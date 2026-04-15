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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import einops
from typing import Tuple

from src.models.utils.attention import Block
from src.models.utils.mamba2 import Mamba2Block
from src.models.utils.cosmos_1_tokenizer import load_cosmos_1_tokenizer

def load_vae(vae_backbone, vae_path):
    if vae_backbone == 'cosmos1':
        vae = load_cosmos_1_tokenizer(vae_path, load_decoder=True, load_jit=True)
    return vae
    
def encode_cosmos1(vae, video):
    sample = vae.encode(video)[0]
    return sample

def encode_video_model(vae, video, vae_backbone):
    if vae_backbone == 'cosmos1':
        encode_func = encode_cosmos1
    return encode_func(vae, video)

def encode_video(vae, video, vae_backbone):
    chunk_size = get_encoder_chunk_size(video)
    with torch.no_grad():
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        samples = []
        for chunk_idx in range(0, video.shape[0], chunk_size):
            video_batch = video[chunk_idx: chunk_idx + chunk_size]
            sample = encode_video_model(vae, video_batch, vae_backbone)
            samples.append(sample)
        samples = torch.cat(samples, 0)
        samples = samples.permute(0, 2, 1, 3, 4) # [B, F, C, H, W]
    return samples

def get_encoder_chunk_size(video, encoder=True):
    if encoder:
        encoder_chunk_sizes = {49: {480: 4, 256: 10, 128: 20}, 121: {704: 1}}
    else:
        encoder_chunk_sizes = {49: {480: 4, 256: 10, 128: 20}, 121: {704: 1}}
    B, T, C, H, W = video.shape
    chunk_size = B
    if T in encoder_chunk_sizes:
        encoder_chunk_sizes_T = encoder_chunk_sizes[T]
        if H in encoder_chunk_sizes_T:
            chunk_size = encoder_chunk_sizes_T[H]
    return chunk_size

def encode_multi_view_video(vae, video, num_input_multi_views, vae_backbone):
    if num_input_multi_views != 1:
        video = einops.rearrange(video, 'b (v t) c h w -> (b v) t c h w', v=num_input_multi_views)
    model_input = encode_video(vae, video, vae_backbone)
    if num_input_multi_views != 1:
        model_input = einops.rearrange(model_input, '(b v) t c h w -> b (v t) c h w', v=num_input_multi_views)
    return model_input

def decode_multi_view_latents(vae, latents: torch.Tensor, num_input_multi_views: int, vae_backbone: str):
    if num_input_multi_views != 1:
        latents = einops.rearrange(latents, 'b (v t) c h w -> (b v) t c h w', v=num_input_multi_views)
    chunk_size = get_encoder_chunk_size(latents, encoder=False)
    video = []
    for chunk_idx in range(0, latents.shape[0], chunk_size):
        latents_batch = latents[chunk_idx: chunk_idx + chunk_size]
        video_batch = decode_video_model(vae, latents_batch, vae_backbone)
        video.append(video_batch)
    video = torch.cat(video, 0)
    if num_input_multi_views != 1:
        video = einops.rearrange(video, '(b v) c t h w -> b c (v t) h w', v=num_input_multi_views)
    video = video.transpose(1, 2)
    video = video.clip(-1, 1)
    return video
    
def decode_cosmos1(vae, video):
    video = einops.rearrange(video, 'b t c h w -> b c t h w')
    sample = vae.decode(video)
    return sample

def decode_video_model(vae, video, vae_backbone):
    if vae_backbone == 'cosmos1':
        decode_func = decode_cosmos1
    return decode_func(vae, video)

def encode_plucker_vae(batch, encode_video, plucker_key='plucker_embedding'):
    batch[plucker_key] = torch.cat((
        encode_video(batch[plucker_key][:, :, :3]),
        encode_video(batch[plucker_key][:, :, 3:])), 
        2)
    return batch

def encode_latent_time_vae(batch, encode_video, img_size, time_keys=['time_embeddings', 'time_embeddings_target']):
    time_embeddings_out = []
    for k in time_keys:
        batch[k] = repeat_time_spatially(batch[k], img_size)
        time_embeddings_out.append(encode_video(batch[k] * 2 - 1))
    batch[time_keys[0]] = torch.cat(time_embeddings_out, 1)
    del batch[time_keys[1]]
    return batch

def repeat_time_spatially(time_embeddings: torch.Tensor, img_size: Tuple[int, int]):
    return einops.repeat(time_embeddings, 'b t c -> b t c h w', h=img_size[0], w=img_size[1])

def get_model_blocks(enc_embed_dim, enc_depth, enc_num_heads, mlp_ratio, use_mamba, llrm_7m1t, norm_layer, use_qk_norm, index_transformer_block=8):
    if use_mamba:
        # Mix of mamba2 and transformer        
        if llrm_7m1t:
            enc_blocks = []
            for i in range(enc_depth):
                if (i + 1) % index_transformer_block == 0:
                    block_mamba = Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, use_qk_norm=use_qk_norm)
                else:
                    block_mamba = Mamba2Block(enc_embed_dim)
                enc_blocks.append(block_mamba)
            enc_blocks = nn.ModuleList(enc_blocks)
        # Only mamba2
        else:
            enc_blocks = nn.ModuleList([
                Mamba2Block(enc_embed_dim)
            for i in range(enc_depth)])
    else:
        enc_blocks = nn.ModuleList(
            [Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, use_qk_norm=use_qk_norm)
            for i in range(enc_depth)]
            )
    return enc_blocks

def forward_checkpointing(layer, *args, gradient_checkpoint=False):
    if not gradient_checkpoint:
        return layer(*args)

    # Identify tensor positions and values
    tensor_positions = [(i, arg) for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)]
    tensor_indices, tensor_args = zip(*tensor_positions) if tensor_positions else ([], [])

    def wrapped(*tensors_in):
        args_copy = list(args)
        for i, t in zip(tensor_indices, tensors_in):
            args_copy[i] = t
        return layer(*args_copy)

    return torch.utils.checkpoint.checkpoint(wrapped, *tensor_args, use_reentrant=False)

def timestep_embedding(timesteps, dim, max_period=10000, use_orig=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if use_orig:
        dim -= 1
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    if use_orig:
        embedding = torch.cat([timesteps[:, None], embedding], dim=-1)

    return embedding

class ConvTranspose3dFactorized(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 upsample_mode='trilinear', use_channel_reduction=True, gradient_checkpoint=True):
        super().__init__()

        self.scale_factor = stride
        self.upsample_mode = upsample_mode
        self.use_channel_reduction = use_channel_reduction
        self.gradient_checkpoint = gradient_checkpoint

        if self.use_channel_reduction:
            self.channel_reducer = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            conv_in_channels = out_channels
        else:
            conv_in_channels = in_channels

        self.conv = nn.Conv3d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def _interpolate(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ['linear', 'bilinear', 'trilinear'] else None
        )

    def _conv(self, x):
        return self.conv(x)

    def _channel_reduce(self, x):
        return self.channel_reducer(x)

    def forward(self, x):
        if self.use_channel_reduction:
            x = forward_checkpointing(self._channel_reduce, x, gradient_checkpoint=self.gradient_checkpoint)
        x = forward_checkpointing(self._interpolate, x, gradient_checkpoint=self.gradient_checkpoint)
        x = forward_checkpointing(self._conv, x, gradient_checkpoint=self.gradient_checkpoint)
        return x

class ConvTranspose3dFactorized(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 upsample_mode='trilinear', use_channel_reduction=True, gradient_checkpoint=True):
        super().__init__()

        self.scale_factor = stride
        self.upsample_mode = upsample_mode
        self.use_channel_reduction = use_channel_reduction
        self.gradient_checkpoint = gradient_checkpoint

        if self.use_channel_reduction:
            self.channel_reducer = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            conv_in_channels = out_channels
        else:
            conv_in_channels = in_channels

        self.conv = nn.Conv3d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def _interpolate(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ['linear', 'bilinear', 'trilinear'] else None
        )

    def _conv(self, x):
        return self.conv(x)

    def _channel_reduce(self, x):
        return self.channel_reducer(x)

    def forward(self, x):
        if self.use_channel_reduction:
            x = forward_checkpointing(self._channel_reduce, x, gradient_checkpoint=self.gradient_checkpoint)
        x = forward_checkpointing(self._interpolate, x, gradient_checkpoint=self.gradient_checkpoint)
        x = forward_checkpointing(self._conv, x, gradient_checkpoint=self.gradient_checkpoint)
        return x

class ConvTranspose3dReduced(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 hidden_channels: int = None, use_channel_reduction: bool = True):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.use_channel_reduction = use_channel_reduction

        if self.use_channel_reduction:
            self.channel_reducer = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
            conv_in_channels = hidden_channels
        else:
            conv_in_channels = in_channels
        self.conv = nn.ConvTranspose3d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def _conv(self, x):
        return self.conv(x)

    def _channel_reduce(self, x):
        return self.channel_reducer(x)

    def forward(self, x: torch.Tensor, gradient_checkpoint: bool = False):
        if self.use_channel_reduction:
            x = forward_checkpointing(self._channel_reduce, x, gradient_checkpoint=gradient_checkpoint)
        x = forward_checkpointing(self._conv, x, gradient_checkpoint=gradient_checkpoint)
        return x

class MultiStageConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        multi_stage=True,
        norm_layer=None,
        activation=nn.ReLU(inplace=True),
        gradient_checkpoint=False,
    ):
        super().__init__()
        self.multi_stage = multi_stage
        self.gradient_checkpoint = gradient_checkpoint

        if not multi_stage:
            self.upsampler = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.pre_pad = None
        else:
            self.target_kernel_size = kernel_size
            self.target_stride = stride
            self.target_padding = padding

            sD, sH, sW = stride
            assert all(s == sH for s in stride[1:]), "Only symmetric spatial strides supported."

            temporal_stages = int(torch.log2(torch.tensor(sD)).item()) if sD > 1 else 0
            spatial_stages = int(torch.log2(torch.tensor(sH)).item())

            self.temporal_blocks = nn.ModuleList()
            for i in range(temporal_stages):
                in_ch = in_channels if i == 0 else out_channels
                self.temporal_blocks.append(nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch,
                        out_channels,
                        kernel_size=(3, 1, 1),
                        stride=(2, 1, 1),
                        padding=(1, 0, 0),
                        output_padding=(1, 0, 0),
                    ),
                    *( [norm_layer(out_channels)] if norm_layer else [] ),
                    *( [activation] if activation else [] )
                ))

            self.spatial_blocks = nn.ModuleList()
            for i in range(spatial_stages):
                in_ch = in_channels if (i == 0 and temporal_stages == 0) else out_channels
                self.spatial_blocks.append(nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch,
                        out_channels,
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                        output_padding=(0, 1, 1),
                    ),
                    *( [norm_layer(out_channels)] if norm_layer else [] ),
                    *( [activation] if activation else [] )
                ))

            self.pre_pad = None

    def _compute_input_padding(self, input_shape):
        D, H, W = input_shape[-3:]
        sD, sH, sW = self.target_stride
        kD, kH, kW = self.target_kernel_size
        pD, pH, pW = self.target_padding

        target_D = (D - 1) * sD - 2 * pD + kD
        spatial_out_H = (H - 1) * sH - 2 * pH + kH
        spatial_out_W = (W - 1) * sW - 2 * pW + kW

        approx_H = H * (2 ** len(self.spatial_blocks))
        approx_W = W * (2 ** len(self.spatial_blocks))

        pad_H = spatial_out_H - approx_H
        pad_W = spatial_out_W - approx_W
        pad = [0, pad_W, 0, pad_H, 0, 0]  # spatial padding only

        return nn.ConstantPad3d(pad, 0.0)

    def forward(self, x):
        if not self.multi_stage:
            return forward_checkpointing(self.upsampler, x, gradient_checkpoint=self.gradient_checkpoint)
        if self.pre_pad is None:
            self.pre_pad = self._compute_input_padding(x.shape)
        x = self.pre_pad(x)

        # Temporal upsampling stages with +1 growth each time
        for block in self.temporal_blocks:
            x = forward_checkpointing(block, x, gradient_checkpoint=self.gradient_checkpoint)

        # Spatial upsampling stages
        for block in self.spatial_blocks:
            x = forward_checkpointing(block, x, gradient_checkpoint=self.gradient_checkpoint)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_length: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_length, dim)

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_embedding = self.pos_embed(positions)  # (1, seq_len, dim)
        return x + pos_embedding