# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.utils.attention import PatchEmbed3D
from src.rendering.gs import GaussianRenderer
from src.rendering.gs_deferred import GaussianRendererDeferred

from src.models.utils.cosmos_1_tokenizer import load_cosmos_1_decoder
from src.models.utils.render import subsample_pixels_spatio_temporal, query_z_with_indices, subsample_x_and_rays
from src.models.utils.model import get_model_blocks, ConvTranspose3dFactorized, MultiStageConvTranspose3d, ConvTranspose3dReduced, forward_checkpointing, PositionalEmbedding

class LatentRecon(nn.Module):
    def __init__(
        self,
        opt,
    ):
        super().__init__()
        self.opt = opt
        
        # Main blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.enc_blocks = get_model_blocks(
            self.opt.enc_embed_dim,
            self.opt.enc_depth,
            self.opt.enc_num_heads,
            self.opt.mlp_ratio,
            self.opt.use_mamba,
            self.opt.llrm_7m1t,
            norm_layer,
            self.opt.use_qk_norm,
            self.opt.llrm_7m1t_index,
            )
        self.enc_norm = norm_layer(self.opt.enc_embed_dim)

        # Reduce number of channels after main blocks
        if self.opt.num_block_channels_reduce is not None:
            self.blocks_out_channels = self.opt.num_block_channels_reduce
            self.block_out = nn.Linear(self.opt.enc_embed_dim, self.blocks_out_channels)
        else:
            self.blocks_out_channels = self.opt.enc_embed_dim
            self.block_out = None

        # Patch dimensions
        patch_size_video = [
            self.opt.patch_size_temporal,
            self.opt.patch_size,
            self.opt.patch_size,
            ]
        patch_size_plucker = [
            self.opt.latent_time_compression,
            self.opt.latent_spat_compression * self.opt.patch_size,
            self.opt.latent_spat_compression * self.opt.patch_size,
            ]
        
        # If time embedded already with VAE, use same patchification as image latents
        if self.opt.time_embedding:
            if self.opt.time_embedding_vae:
                patch_size_time = patch_size_video
                time_embedding_dim = self.opt.num_latent_c
            else:
                patch_size_time = [
                    self.opt.latent_time_compression,
                    self.opt.patch_size,
                    self.opt.patch_size,
                    ]
                time_embedding_dim = self.opt.time_embedding_dim
        self.stride_size_out = [
            self.opt.latent_time_compression // self.opt.patch_size_out_factor[0],
            self.opt.latent_spat_compression * self.opt.patch_size // self.opt.patch_size_out_factor[1],
            self.opt.latent_spat_compression * self.opt.patch_size // self.opt.patch_size_out_factor[2],
            ]
        
        # Patch embeddings for non-latent input (t h w -> t_latent h_latent/2 w_latent/2)
        if self.opt.use_rgb_decoder:
            self.padding_time = self.padding_plucker = (0, 0, 0)
            self.patch_size_extra_t = 0
        else:
            # If time embedded already with VAE, use same patchification as image latents
            if self.opt.time_embedding_vae:
                self.padding_time = (0, 0, 0)
            else:
                self.padding_time = (2, 0, 0)
            self.padding_plucker = (self.opt.latent_time_compression//2, 0, 0)
            self.patch_size_extra_t = 1
        self.patch_size_out = [self.stride_size_out[0] + self.patch_size_extra_t, self.stride_size_out[1], self.stride_size_out[2]]

        # Patch embeddings for video latents (t_latent h_latent w_latent -> t_latent h_latent/2 w_latent/2)
        if self.opt.use_patch_embeddings_encoder:
            self.patch_embed = PatchEmbed3D(patch_size_video, self.opt.num_latent_c, self.opt.enc_embed_dim)
        
        # Plucker
        if self.opt.plucker_embedding_vae:
            patch_size_plucker = patch_size_video
            self.padding_plucker = 0
            # Encoded rays_dxo and rays_d with vae separately
            if self.opt.plucker_embedding_vae_fuse_type == 'concat':
                num_plucker_in_channels = 2 * self.opt.num_latent_c
        else:
            num_plucker_in_channels = 6
        self.patch_plucker_embed = PatchEmbed3D(patch_size_plucker, num_plucker_in_channels, self.opt.enc_embed_dim, zero_init=True, padding=self.padding_plucker)
        
        # Encode time conditionings
        if self.opt.time_embedding:
            self.patch_time_embed = PatchEmbed3D(patch_size_time, time_embedding_dim, self.opt.enc_embed_dim, zero_init=True, padding=self.padding_time)
            self.patch_time_embed_tgt = PatchEmbed3D(patch_size_time, time_embedding_dim, self.opt.enc_embed_dim, zero_init=True, padding=self.padding_time)
        
        # Positional embedding (if plucker is not used)
        if self.opt.use_pos_embedding:
            self.pos_embedding = PositionalEmbedding(**self.opt.pos_embedding_kwargs)
        
        # Output layers
        self.output_dims = self.opt.output_dims

        # Learn mask that prunes gaussians
        if self.opt.sub_sample_gaussians_type == 'learned':
            self.output_dims += 1
        
        # Learn position offsets
        if self.opt.gaussians_predict_offset:
            self.output_dims += 3
        
        # Set transposed conv decoding module
        transposed_conv_kwargs = {}
        if self.opt.transposed_conv_type == 'factorized':
            transposed_conv_module = ConvTranspose3dFactorized
        elif self.opt.transposed_conv_type == 'reduce_transposed':
            transposed_conv_module = ConvTranspose3dReduced
            if self.opt.transposed_conv_hidden_channels is not None:
                transposed_conv_kwargs['hidden_channels'] = self.opt.transposed_conv_hidden_channels
        elif self.opt.transposed_conv_type == 'multi_stage_transpose':
            transposed_conv_module = MultiStageConvTranspose3d
        else:
            transposed_conv_module = nn.ConvTranspose3d

        # Decode into RGB with cosmos decoder or just one deconv
        if self.opt.use_cosmos_decoder:
            self.opt.decoder_cosmos_kwargs['out_channels'] = self.output_dims
            self.decoder_cosmos, tokenizer_config = load_cosmos_1_decoder(self.opt.vae_path, self.opt.decoder_cosmos_kwargs)
            deconv_out_channels = tokenizer_config['channels']
            # Set up new patchification based on cosmos upsampling
            temp_factor = int(self.opt.latent_time_compression/self.opt.decoder_cosmos_kwargs['temporal_compression'])
            spat_factor = int(self.opt.latent_spat_compression/self.opt.decoder_cosmos_kwargs['spatial_compression'])
            self.patch_size_out = [
                int(1 / self.opt.patch_size_out_factor[0] * temp_factor),
                int(self.opt.patch_size / self.opt.patch_size_out_factor[1] * spat_factor),
                int(self.opt.patch_size / self.opt.patch_size_out_factor[2] * spat_factor),
            ]
            self.stride_size_out = self.patch_size_out
            patch_size_out_deconv = self.patch_size_out
            stride_size_out_deconv = self.stride_size_out
            self.deconv = transposed_conv_module(self.blocks_out_channels, deconv_out_channels, patch_size_out_deconv, stride=stride_size_out_deconv, padding=0)
        else:
            if self.opt.use_patch_embeddings_encoder:
                self.padding_deconv = (self.opt.latent_time_compression//2, 0, 0)
                self.deconv = transposed_conv_module(self.blocks_out_channels, self.output_dims, self.patch_size_out, stride=self.stride_size_out, padding=self.padding_deconv, **transposed_conv_kwargs)     
        
        # Initialize weights
        for module_name, module in self.named_children():
            module.apply(self._init_weights)

        # Gaussian renderer
        if self.opt.deferred_bp:
            self.gs = GaussianRendererDeferred(opt)
        else:
            self.gs = GaussianRenderer(opt)

        # Gaussian misc
        scale_cap = opt.gaussian_scale_cap
        scale_shift = 1 - math.log(scale_cap)
        self.scale_act = lambda x: torch.minimum(torch.exp(x-scale_shift),torch.tensor([scale_cap],device=x.device,dtype=x.dtype))
        self.opacity_act = lambda x: torch.sigmoid(x-2.0)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5
        self.dnear = self.opt.dnear
        self.dfar = self.opt.dfar

    def forward_gaussians(self, images_input, plucker_embedding, rays_os, rays_ds, time_embeddings, num_input_multi_views=None):
        # Compute embeddings per view independently, reshape multi-view from temporal into batch
        images_input = self.reshape_mv_temp_to_batch(images_input, num_input_multi_views=num_input_multi_views)
        plucker_embedding = self.reshape_mv_temp_to_batch(plucker_embedding, num_input_multi_views=num_input_multi_views)
        rays_os = self.reshape_mv_temp_to_batch(rays_os, num_input_multi_views=num_input_multi_views)
        rays_ds = self.reshape_mv_temp_to_batch(rays_ds, num_input_multi_views=num_input_multi_views)

        B, V, C, H, W = images_input.shape
        h = int(H//self.opt.patch_size)
        w = int(W//self.opt.patch_size)

        # Patchify input images
        if self.opt.use_patch_embeddings_encoder:
            x = forward_checkpointing(self.patch_embed, images_input, gradient_checkpoint=self.opt.gradient_checkpoint_transformer)
        else:
            x = rearrange(images_input, 'b t c h w -> b (t h w) c')

        # Time embedding
        if self.opt.time_embedding and self.opt.get('use_time_embedding', True):
            x_time_emb, x_time_emb_tgt = self.get_time_embedding(time_embeddings, V, num_input_multi_views=num_input_multi_views)
            x = x + x_time_emb + x_time_emb_tgt

        # Add Plucker embeddings
        if self.opt.use_plucker:
            x = x + forward_checkpointing(self.patch_plucker_embed, plucker_embedding, gradient_checkpoint=self.opt.gradient_checkpoint_transformer)

        # Reshape views into THW dimension for joint attention across input views
        if self.opt.process_multi_views:
            x = self.reshape_mv_batch_to_temp(x, num_input_multi_views=num_input_multi_views)

        # Add positional embedding
        if self.opt.use_pos_embedding:
            x = self.pos_embedding(x)

        # Main blocks
        for blk_idx, blk in enumerate(self.enc_blocks):
            x = forward_checkpointing(blk, x , gradient_checkpoint=self.opt.gradient_checkpoint_transformer)      
        x = forward_checkpointing(self.enc_norm, x, gradient_checkpoint=self.opt.gradient_checkpoint_transformer)

        # Additional output block
        if self.block_out is not None:
            x = forward_checkpointing(self.block_out, x, gradient_checkpoint=self.opt.gradient_checkpoint_transformer)

        # Reshape multi-view dimension back to batch dimension to decode independently
        if self.opt.process_multi_views:
            x = self.reshape_mv_temp_to_batch(x, num_input_multi_views=num_input_multi_views)
        
        # Decode temporally and spatially into higher dimensions with convolutions
        x = rearrange(x, 'b (t h w) c -> b c t h w', h=h, w=w)
        if self.opt.use_patch_embeddings_encoder:
            x = forward_checkpointing(self.deconv, x, gradient_checkpoint=self.opt.gradient_checkpoint_conv)
        elif self.opt.use_cosmos_decoder:
            x = self.decoder_cosmos(x, gradient_checkpoint=self.opt.gradient_checkpoint_conv)

        # Get predicted subsampling mask
        if self.opt.sub_sample_gaussians_factor is not None:
            if self.opt.sub_sample_gaussians_type == 'learned':
                x, x_mask = x[:, :-1], x[:, [-1]]
            else:
                x_mask = None

        # Subsample number of gaussians
        if self.opt.sub_sample_gaussians and self.opt.sub_sample_gaussians_factor is not None:
            x = forward_checkpointing(self.subsample_x_and_rays_wrapper, x, rays_os, rays_ds, x_mask, gradient_checkpoint=self.opt.gradient_checkpoint_conv)
        else:
            x = rearrange(x, 'b c t h w -> b (t h w) c')
            rays_os = rearrange(rays_os, 'b t c h w -> b (t h w) c')
            rays_ds = rearrange(rays_ds, 'b t c h w -> b (t h w) c')
            x_mask = None

        # Set up gaussian attributes
        x = forward_checkpointing(self.gaussian_processing, x, rays_os, rays_ds, gradient_checkpoint=self.opt.gradient_checkpoint_conv)

        # Merge viewpoints into one gaussian vector
        if self.opt.fuse_multi_views or not self.training:
            x = self.reshape_mv_batch_to_temp(x, num_input_multi_views)
        
        # Optionally prune gaussians as in Long LRM
        x = self.gaussian_pruning(x)
        return x, x_mask

    def forward(self, data, skip_loss=False):

        results = {}
        loss = 0

        rays_os = data['rays_os']
        rays_ds = data['rays_ds']
        plucker_embedding = data['plucker_embedding']
        images = data['images_input_embed']
        time_embeddings = data['time_embeddings']
        cam_view = data['cam_view']
        if len(cam_view.shape) == 5:
            # B 1 T C D -> B T C D
            cam_view = cam_view.squeeze(1)
        intrinsics = data['intrinsics']
        num_input_multi_views = data['num_input_multi_views']
        B = images.shape[0]
        
        # use the first view to predict gaussians
        gaussians, gaussians_mask = self.forward_gaussians(images, plucker_embedding, rays_os, rays_ds, time_embeddings=time_embeddings, num_input_multi_views=num_input_multi_views) # [B, N, 14]
        
        # always use white background
        add_gs_render_kwargs = {}
        if self.opt.deferred_bp:
            bg_color = [1, 1, 1]
            add_gs_render_kwargs = {'patch_size': self.opt.gs_render_patch_size}
        else:
            bg_color = torch.ones(3, dtype=gaussians.dtype, device=gaussians.device)     

        # render predictions
        results = self.gs.render(gaussians, cam_view, bg_color=bg_color, intrinsics=intrinsics, **add_gs_render_kwargs)
        
        # output
        if self.training:
            out = {}
            out['images_pred'] = results['images_pred']
            out['depths_pred'] = results['depths_pred']

            # opacity loss
            if self.opt.lambda_opacity > 0:
                opacity = gaussians[..., 3]
            else:
                opacity = None
            out["opacity_pred"] = opacity
        else:
            out = results
            out['gaussians_mask_pred'] = gaussians_mask
            out['gaussians'] = gaussians
        return out
    
    def gaussian_processing(self, x: torch.Tensor, rays_os: torch.Tensor, rays_ds: torch.Tensor):
        # Pixel-aligned gaussians
        if self.opt.gaussians_predict_offset:
            pos_offset = x[..., -3:]
            x = x[..., :-3]
        distance, rgb, scaling, rotation, opacity = x.split([1, 3, 3, 4, 1], dim=-1)   
        w = torch.sigmoid(distance + self.opt.pre_sigmoid_distance_shift)
        depths = self.dnear * (1 - w) + self.dfar * w
        pos = rays_os + rays_ds * depths

        # Add offsets to gaussian positions
        if self.opt.gaussians_predict_offset and self.opt.use_gaussians_predict_offset:
            if self.opt.gaussians_predict_offset_act == 'clamp':
                pos_offset = pos_offset.clamp(self.opt.gaussians_predict_offset_range[0], self.opt.gaussians_predict_offset_range[1])
            elif self.opt.gaussians_predict_offset_act == 'tanh':
                pos_offset = self.opt.gaussians_predict_offset_range[1] * torch.tanh(pos_offset)
            pos = pos + pos_offset
        
        # Activations
        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)
        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        return gaussians
    
    def get_time_embedding(self, time_embeddings: torch.Tensor, V: int, num_input_multi_views: int = None):
        # Split into input and target time embeddings
        if self.opt.time_embedding_vae:
            num_in_times = V
        else:
            time_embeddings = repeat(time_embeddings, 'b t c -> b t c h w', h=H, w=W)
            num_in_times = self.opt.num_input_views
        time_embeddings_input = time_embeddings[:, :num_in_times]
        time_embeddings_target = time_embeddings[:, num_in_times: num_in_times + 1]
        # Use the same target timestep for all input images
        time_embeddings_target = repeat(time_embeddings_target, 'b 1 c h w -> b t c h w', t=num_in_times)
        # Encode times
        x_time_emb = self.patch_time_embed(time_embeddings_input)
        x_time_emb_tgt = self.patch_time_embed_tgt(time_embeddings_target)
        # Repeat time embeddings for each view
        x_time_emb = self.repeat_to_mv(x_time_emb, num_input_multi_views=num_input_multi_views)
        x_time_emb_tgt = self.repeat_to_mv(x_time_emb_tgt, num_input_multi_views=num_input_multi_views)
        return x_time_emb, x_time_emb_tgt
    
    def gaussian_pruning(self, gaussians: torch.Tensor):
        # Gaussian pruning following Long LRM
        prune_ratio = self.opt.gaussians_prune_ratio
        if prune_ratio > 0:
            opacity = gaussians[:, :, [3]]
            num_gaussians = gaussians.shape[1]
            keep_ratio = 1 - prune_ratio
            random_ratio = self.opt.gaussians_random_ratio
            random_ratio = keep_ratio * random_ratio
            keep_ratio = keep_ratio - random_ratio
            num_keep = int(num_gaussians * keep_ratio)
            num_keep_random = int(num_gaussians * random_ratio)
            # rank by opacity
            idx_sort = opacity.argsort(dim=1, descending=True)
            keep_idx = idx_sort[:, :num_keep]
            if num_keep_random > 0:
                rest_idx = idx_sort[:, num_keep:]
                random_idx = rest_idx[:, torch.randperm(rest_idx.shape[1])[:num_keep_random]]
                keep_idx = torch.cat([keep_idx, random_idx], dim=1)
            gaussians = gaussians.gather(1, keep_idx.expand(-1, -1, gaussians.shape[-1]))
        return gaussians     
    
    def subsample_x_and_rays_wrapper(self, x, rays_os, rays_ds, x_mask):
        return subsample_x_and_rays(
            x, rays_os, rays_ds, x_mask,
            self.opt.sub_sample_gaussians_factor,
            self.opt.sub_sample_gaussians_type,
            self.opt.sub_sample_gaussians_type_tokens,
            self.opt.sub_sample_gaussians_temperature,
            self.training,
        )
    
    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def subsample_views(self, x: torch.Tensor, num_input_multi_views: int):
        x = rearrange(x, '(b v) ... -> b v ...', v=num_input_multi_views)
        x = x[:, :self.opt.num_target_multi_views]
        x = rearrange(x, 'b v ... -> (b v) ...')
        return x
    
    def reshape_mv_temp_to_batch(self, x, num_input_multi_views=None):
        if num_input_multi_views is None:
            num_input_multi_views = self.num_input_multi_views
        if num_input_multi_views != 1:
            if len(x.shape) == 5:
                x = rearrange(x, 'b (v t) c h w -> (b v) t c h w', v=num_input_multi_views)
            elif len(x.shape) == 3:
                x = rearrange(x, 'b (v d) c -> (b v) d c', v=num_input_multi_views)
        return x
    
    def reshape_mv_batch_to_temp(self, x, num_input_multi_views=None):
        if num_input_multi_views is None:
            num_input_multi_views = self.num_input_multi_views
        if num_input_multi_views != 1:
            if len(x.shape) == 5:
                x = rearrange(x, '(b v) t c h w -> b (v t) c h w', v=num_input_multi_views)
            elif len(x.shape) == 3:
                x = rearrange(x, '(b v) d c -> b (v d) c', v=num_input_multi_views)
        return x
    
    def reshape_mv_batch_to_mv(self, x, num_input_multi_views=None):
        if num_input_multi_views is None:
            num_input_multi_views = self.num_input_multi_views
        if num_input_multi_views != 1:
            if len(x.shape) == 5:
                x = rearrange(x, '(b v) t c h w -> b v t c h w', v=num_input_multi_views)
            elif len(x.shape) == 3:
                x = rearrange(x, '(b v) d c -> b v d c', v=num_input_multi_views)
        return x
    
    def reshape_mv_batch_to_view(self, x, num_input_multi_views=None):
        if num_input_multi_views is None:
            num_input_multi_views = self.num_input_multi_views
        if num_input_multi_views != 1:
            if len(x.shape) == 3:
                x = rearrange(x, '(b v) d c -> (b d) v c', v=num_input_multi_views)
        return x

    def repeat_to_mv(self, x, num_input_multi_views=None):
        if num_input_multi_views is None:
            num_input_multi_views = self.num_input_multi_views
        if num_input_multi_views != 1:
            if len(x.shape) == 5:
                x = repeat(x, 'b t c h w -> (b v) t c h w', v=num_input_multi_views)
            elif len(x.shape) == 3:
                x = repeat(x, 'b d c -> (b v) d c', v=num_input_multi_views)
        return x