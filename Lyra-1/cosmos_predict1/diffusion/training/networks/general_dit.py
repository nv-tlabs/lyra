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

"""
A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.
It allows us easy to switch building blocks used and their order. Its instantiation includes
* transformer on fully flattened tokens
* factored spatial and temporal attention
* factored non-overlap spatial and temporal attention
* mixing of above attention types

Limitations:

* In favor of simplicity and cleanness, many ops are not fused and we can do better
* such as combining mutiple adaln MLPs into one inside one transformer block.
* we use reshape heavily, which may be not efficient when its occurs unnecessary CUDA memory copy

Purpose:
* A prototype for testing different attention types and their combinations
* Idealy, we want to know where we should allocate our resources / FLOPS / memory via extensive empirical studies
"""

from collections.abc import Container
from typing import List, Optional, Tuple

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms

from cosmos_predict1.diffusion.module.attention import get_normalization
from cosmos_predict1.diffusion.training.conditioner import DataType
from cosmos_predict1.diffusion.training.module.blocks import (
    DITBuildingBlock,
    FinalLayer,
    GeneralDITTransformerBlock,
    PatchEmbed,
    SDXLTimestepEmbedding,
    SDXLTimesteps,
)
from cosmos_predict1.diffusion.training.module.position_embedding import (
    LearnableEmb3D,
    LearnableEmb3D_FPS_Aware,
    LearnablePosEmbAxis,
    SinCosPosEmb,
    SinCosPosEmb_FPS_Aware,
    SinCosPosEmbAxis,
    VideoRopePosition3DEmb,
    VideoRopePositionEmb,
)
from cosmos_predict1.diffusion.training.tensor_parallel import gather_along_first_dim, scatter_along_first_dim
from cosmos_predict1.utils import log


class GeneralDIT(nn.Module):
    """
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.
        Attributes:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (tuple of int): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        block_config (str): Configuration of the transformer block, e.g., 'FA-CA-MLP', means
            full attention, cross attention, and MLP in sequence in one transformer block.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of residual blocks per resolution in the transformer.
        num_heads (int): Number of heads in the multi-head self-attention layers.
        spatial_attn_win_size (int): Window size for the spatial attention mechanism.
        temporal_attn_win_size (int): Window size for the temporal attention mechanism.
        mlp_ratio (float): Expansion ratio for the MLP (multi-layer perceptron) blocks in the transformer.
        use_memory_save (bool): If True, utilizes checkpointing to reduce memory usage during training. (Deprecated)
        use_checkpoint (bool): If True, utilizes checkpointing to reduce memory usage during training for all blocks.
        crossattn_emb_channels (int): Number of embedding channels used in the cross-attention layers.
        use_cross_attn_mask (bool): If True, applies a mask during cross-attention operations to manage sequence alignment.
        pos_emb_cls (str): Type of positional embeddings used ('sincos' for sinusoidal or other types).
        pos_emb_learnable (bool): Specifies if positional embeddings are learnable.
        pos_emb_interpolation (str): Method used for interpolating positional embeddings, e.g., 'crop' for cropping adjustments.
        block_x_format (str, optional): The format of the input tensor for the transformer block. Defaults to "BTHWD". Only support 'BTHWD' and 'THWBD'.
        legacy_patch_emb (bool): If True, applies 3D convolutional layers for video inputs, otherwise, use Linear! This is for backward compatibility.
        rope_h_extrapolation_ratio (float): Ratio of the height extrapolation for the rope positional embedding.
        rope_w_extrapolation_ratio (float): Ratio of the width extrapolation for the rope positional embedding.
        rope_t_extrapolation_ratio (float): Ratio of the temporal extrapolation for the rope positional embedding.
    Note:
        block_config support block type:
        * spatial_sa, ssa: spatial self attention
        * temporal_sa, tsa: temporal self attention
        * cross_attn, ca: cross attention
        * full_attn: full attention on all flatten tokens
        * mlp, ff: feed forward block
        * use '-' to separate different building blocks, e.g., 'FA-CA-MLP' means full attention, cross attention, and MLP in sequence in one transformer block.

    Example:
        >>> # full attention, cross attention, and MLP
        >>> option1_block_config = 'FA-CA-MLP'
        >>> model_1 = GeneralDIT(
                max_img_h=64, max_img_w=64, max_frames=32, in_channels=16, out_channels=16,
                patch_spatial=2, patch_temporal=1, model_channels=768, num_blocks=10,
                num_heads=16, mlp_ratio=4.0,
                spatial_attn_win_size=1, temporal_attn_win_size=1,
                block_config=option1_block_config
            )
        >>> option2_block_config = 'SSA-CA-MLP-TSA-CA-MLP'
        >>> model_2 = GeneralDIT(
                max_img_h=64, max_img_w=64, max_frames=32, in_channels=16, out_channels=16,
                patch_spatial=2, patch_temporal=1, model_channels=768, num_blocks=10,
                num_heads=16, mlp_ratio=4.0,
                spatial_attn_win_size=1, temporal_attn_win_size=1,
                block_config=option2_block_config
            )
        >>> # option3 model
        >>> model_3 = GeneralDIT(
                max_img_h=64, max_img_w=64, max_frames=32, in_channels=16, out_channels=16,
                patch_spatial=2, patch_temporal=1, model_channels=768, num_blocks=10,
                num_heads=16, mlp_ratio=4.0,
                spatial_attn_win_size=1, temporal_attn_win_size=2,
                block_config=option2_block_config
            )
        >>> # Process input tensor through the model
        >>> output = model(input_tensor)
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        block_config: str = "FA-CA-MLP",
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        window_block_indexes: list = [],  # index for window attention block
        window_sizes: list = [],  # window size for window attention block in the order of T, H, W
        spatial_attn_win_size: int = 1,
        temporal_attn_win_size: int = 1,
        mlp_ratio: float = 4.0,
        use_memory_save: bool = False,
        use_checkpoint: bool = False,
        block_x_format: str = "BTHWD",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_cross_attn_mask: bool = False,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,  # 1 for getty video
        max_fps: int = 30,  # 120 for getty video but let's use 30
        additional_timestamp_channels: dict = None,  # Follow SDXL, in format of {condition_name : dimension}
        affline_emb_norm: bool = False,  # whether or not to normalize the affine embedding
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        layer_mask: list = None,  # whether or not a layer is used. For controlnet encoder
        legacy_patch_emb: bool = True,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = True,
        extra_per_block_abs_pos_emb_type: str = "learnable",
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.use_cross_attn_mask = use_cross_attn_mask
        self.concat_padding_mask = concat_padding_mask
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.additional_timestamp_channels = additional_timestamp_channels
        self.affline_emb_norm = affline_emb_norm
        self.legacy_patch_emb = legacy_patch_emb
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_per_block_abs_pos_emb_type = extra_per_block_abs_pos_emb_type.lower()
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio

        self.build_patch_embed()
        self.build_pos_embed()
        self.cp_group = None
        self.sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
        self.block_x_format = block_x_format
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            SDXLTimesteps(model_channels),
            SDXLTimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        self.blocks = nn.ModuleDict()
        self.block_config = block_config
        self.use_memory_save = use_memory_save
        self.use_checkpoint = use_checkpoint

        assert (
            len(window_block_indexes) == 0 or block_config == "FA-CA-MLP"
        ), "Block config must be FA-CA-MLP if using a combination of window attention and global attention"

        layer_mask = [False] * num_blocks if layer_mask is None else layer_mask
        assert (
            len(layer_mask) == num_blocks
        ), f"Layer mask length {len(layer_mask)} does not match num_blocks {num_blocks}"
        for idx in range(num_blocks):
            if layer_mask[idx]:
                continue
            self.blocks[f"block{idx}"] = GeneralDITTransformerBlock(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                block_config=block_config,
                window_sizes=(
                    window_sizes if idx in window_block_indexes else []
                ),  # There will be bug if using "WA-CA-MLP"
                mlp_ratio=mlp_ratio,
                spatial_attn_win_size=spatial_attn_win_size,
                temporal_attn_win_size=temporal_attn_win_size,
                x_format=self.block_x_format,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
                use_checkpoint=use_checkpoint,
            )

        self.build_decode_head()
        self.build_additional_timestamp_embedder()
        if self.affline_emb_norm:
            log.critical("Building affine embedding normalization layer")
            self.affline_norm = get_normalization("R", model_channels)
        else:
            self.affline_norm = nn.Identity()
        self.init_weights()

        if self.use_memory_save:
            log.critical("Using checkpointing to save memory! only verified in 14B base model training!")
            for block in self.blocks.values():
                block.set_memory_save()

    def init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder[1].linear_1.weight, std=0.02)
        if self.t_embedder[1].linear_1.bias is not None:
            nn.init.constant_(self.t_embedder[1].linear_1.bias, 0)
        nn.init.normal_(self.t_embedder[1].linear_2.weight, std=0.02)
        if self.t_embedder[1].linear_2.bias is not None:
            nn.init.constant_(self.t_embedder[1].linear_2.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for transformer_block in self.blocks.values():
            for block in transformer_block.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                if block.adaLN_modulation[-1].bias is not None:
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Tensor parallel
        if parallel_state.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
            self.initialize_tensor_parallel_weights()

    def initialize_tensor_parallel_weights(self):
        """
        Initialize weights for tensor parallel layers.

        This function performs the following steps:
        1. Retrieves the tensor parallel rank.
        2. Saves the current random state.
        3. Sets a new random seed based on the tensor parallel rank.
        4. Initializes weights for attention and MLP layers in each block.
        5. Restores the original random state.

        The use of different random seeds for each rank ensures
        unique initializations across parallel processes.
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # Save the current random state
        rng_state = torch.get_rng_state()

        # Set a new random seed based on the tensor parallel rank
        torch.manual_seed(tp_rank)

        for block in self.blocks.values():
            for layer in block.blocks:
                if layer.block_type in ["full_attn", "fa", "cross_attn", "ca"]:
                    # Initialize weights for attention layers
                    torch.nn.init.xavier_uniform_(layer.block.attn.to_q[0].weight)
                    torch.nn.init.xavier_uniform_(layer.block.attn.to_k[0].weight)
                    torch.nn.init.xavier_uniform_(layer.block.attn.to_v[0].weight)
                    torch.nn.init.xavier_uniform_(layer.block.attn.to_out[0].weight)
                elif layer.block_type in ["mlp", "ff"]:
                    # Initialize weights for MLP layers
                    torch.nn.init.xavier_uniform_(layer.block.layer1.weight)
                    torch.nn.init.xavier_uniform_(layer.block.layer2.weight)
                else:
                    raise ValueError(f"Unknown block type {layer.block_type}")

        # Restore the original random state
        torch.set_rng_state(rng_state)

    def build_decode_head(self):
        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
            keep_spatio=True,
            legacy_patch_emb=self.legacy_patch_emb,
        )
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if self.legacy_patch_emb:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def build_additional_timestamp_embedder(self):
        if self.additional_timestamp_channels:
            self.additional_timestamp_embedder = nn.ModuleDict()
            for cond_name, cond_emb_channels in self.additional_timestamp_channels.items():
                log.critical(
                    f"Building additional timestamp embedder for {cond_name} with {cond_emb_channels} channels"
                )
                self.additional_timestamp_embedder[cond_name] = nn.Sequential(
                    SDXLTimesteps(cond_emb_channels),
                    SDXLTimestepEmbedding(cond_emb_channels, cond_emb_channels),
                )

    def prepare_additional_timestamp_embedder(self, **kwargs):
        condition_concat = []

        for cond_name, embedder in self.additional_timestamp_embedder.items():
            condition_concat.append(embedder(kwargs[cond_name])[0])
        embedding = torch.cat(condition_concat, dim=1)
        if embedding.shape[1] < self.model_channels:
            embedding = nn.functional.pad(embedding, (0, self.model_channels - embedding.shape[1]))
        return embedding

    def build_pos_embed(self):
        if self.pos_emb_cls == "sincos":
            cls_type = SinCosPosEmb
        elif self.pos_emb_cls == "learnable":
            cls_type = LearnableEmb3D
        elif self.pos_emb_cls == "sincos_fps_aware":
            cls_type = SinCosPosEmb_FPS_Aware
        elif self.pos_emb_cls == "learnable_fps_aware":
            cls_type = LearnableEmb3D_FPS_Aware
        elif self.pos_emb_cls == "rope":
            cls_type = VideoRopePositionEmb
        elif self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.critical(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
        )
        self.pos_embedder = cls_type(
            **kwargs,
        )

        assert self.extra_per_block_abs_pos_emb is True, "extra_per_block_abs_pos_emb must be True"

        if self.extra_per_block_abs_pos_emb:
            assert self.extra_per_block_abs_pos_emb_type in [
                "learnable",
            ], f"Unknown extra_per_block_abs_pos_emb_type {self.extra_per_block_abs_pos_emb_type}"
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the `self.pos_embedder`
                with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D, None, extra_pos_emb

    def decoder_head(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        origin_shape: Tuple[int, int, int, int, int],  # [B, C, T, H, W]
        crossattn_mask: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del crossattn_emb, crossattn_mask
        B, C, T_before_patchify, H_before_patchify, W_before_patchify = origin_shape
        x_BT_HW_D = rearrange(x_B_T_H_W_D, "B T H W D -> (B T) (H W) D")
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D, adaln_lora_B_3D=adaln_lora_B_3D)
        # This is to ensure x_BT_HW_D has the correct shape because
        # when we merge T, H, W into one dimension, x_BT_HW_D has shape (B * T * H * W, 1*1, D).
        x_BT_HW_D = x_BT_HW_D.view(
            B * T_before_patchify // self.patch_temporal,
            H_before_patchify // self.patch_spatial * W_before_patchify // self.patch_spatial,
            -1,
        )
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
            t=self.patch_temporal,
            B=B,
        )
        return x_B_D_T_H_W

    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        del kwargs
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")
            timesteps_B_D = timesteps_B_D + scalar_feature.mean(dim=1)

        if self.additional_timestamp_channels:
            additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                bs=x.shape[0],
                fps=fps,
                h=image_size[:, 0],
                w=image_size[:, 1],
                org_h=image_size[:, 2],
                org_w=image_size[:, 3],
            )

            affline_emb_B_D += additional_cond_B_D
            affline_scale_log_info["additional_cond_B_D"] = additional_cond_B_D.detach()

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D
        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks["block0"].x_format == "THWBD":
            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")

            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

            if self.sequence_parallel:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                # Sequence parallel requires the input tensor to be scattered along the first dimension.
                assert self.block_config == "FA-CA-MLP"  # Only support this block config for now
                T, H, W, B, D = x.shape
                # variable name x_T_H_W_B_D is no longer valid. x is reshaped to THW*1*1*b*D and will be reshaped back in FinalLayer
                x = x.view(T * H * W, 1, 1, B, D)
                assert x.shape[0] % parallel_state.get_tensor_model_parallel_world_size() == 0
                x = scatter_along_first_dim(x, tp_group)

                if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.view(
                        T * H * W, 1, 1, B, D
                    )
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = scatter_along_first_dim(
                        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, tp_group
                    )

        elif self.blocks["block0"].x_format == "BTHWD":
            x = x_B_T_H_W_D
        else:
            raise ValueError(f"Unknown x_format {self.blocks[0].x_format}")
        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output

    def forward_blocks_regular(
        self,
        x,
        affline_emb_B_D,
        crossattn_emb,
        crossattn_mask,
        rope_emb_L_1_1_D,
        adaln_lora_B_3D,
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        feature_indices,
        original_shape,
        x_ctrl,
        return_features_early,
    ):
        features = []
        for name, block in self.blocks.items():
            assert (
                self.blocks["block0"].x_format == block.x_format
            ), f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"
            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )

            # Extract features
            block_idx = int(name.split("block")[-1])
            if block_idx in feature_indices:
                B, C, T, H, W = original_shape
                H = H // self.patch_spatial
                W = W // self.patch_spatial
                T = T // self.patch_temporal
                if self.sequence_parallel:
                    x_feat = gather_along_first_dim(x, parallel_state.get_tensor_model_parallel_group())
                    x_B_T_H_W_D = rearrange(x_feat, "(T H W) 1 1 B D -> B T H W D", T=T, H=H, W=W)
                else:
                    x_feat = x
                    if self.blocks["block0"].x_format == "THWBD":
                        x_B_T_H_W_D = rearrange(x_feat, "T H W B D -> B T H W D", T=T, H=H, W=W)
                    elif self.blocks["block0"].x_format == "BTHWD":
                        x_B_T_H_W_D = x_feat
                    else:
                        raise ValueError(f"Unknown x_format {self.blocks[-1].x_format}")

                features.append(x_B_T_H_W_D)

            if x_ctrl is not None and name in x_ctrl:
                x = x + x_ctrl[name]
            # If we have all of the features, we can exit early
            if return_features_early and len(features) == len(feature_indices):
                return features

        if self.blocks["block0"].x_format == "THWBD":
            x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")
        elif self.blocks["block0"].x_format == "BTHWD":
            x_B_T_H_W_D = x
        else:
            raise ValueError(f"Unknown x_format {self.blocks[-1].x_format}")

        x_B_D_T_H_W = self.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            crossattn_emb=None,
            origin_shape=original_shape,
            crossattn_mask=None,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        if len(feature_indices) == 0:
            # no features requested, return only the model output
            return x_B_D_T_H_W
        else:
            # score and features； score, features
            return x_B_D_T_H_W, features

    def forward_blocks_memory_save(
        self,
        x,
        affline_emb_B_D,
        crossattn_emb,
        crossattn_mask,
        rope_emb_L_1_1_D,
        adaln_lora_B_3D,
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        feature_indices,
        original_shape,
        x_ctrl,
        return_features_early,
    ):
        x_before_gate = 0
        x_skip = rearrange(x, "T H W B D -> (T H W) B D")
        assert self.blocks["block0"].x_format == "THWBD"
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            extra_per_block_pos_emb = rearrange(extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "T H W B D -> (T H W) B D")
        else:
            extra_per_block_pos_emb = None
        gate_L_B_D = 1.0

        features = []
        for name, block in self.blocks.items():
            gate_L_B_D, x_before_gate, x_skip = block(
                x_before_gate,
                x_skip,
                gate_L_B_D,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_per_block_pos_emb,
            )

            # Extract features.
            # Convert the block index in the memory save mode to the block index in the regular mode.
            block_idx = int(name.split("block")[-1]) - 1
            if block_idx in feature_indices:
                B, C, T_before_patchify, H_before_patchify, W_before_patchify = original_shape
                H = H_before_patchify // self.patch_spatial
                W = W_before_patchify // self.patch_spatial
                T = T_before_patchify // self.patch_temporal
                if self.sequence_parallel:
                    x_feat = gather_along_first_dim(x_skip, parallel_state.get_tensor_model_parallel_group())
                    x_B_T_H_W_D = rearrange(x_feat, "(T H W) 1 1 B D -> B T H W D", T=T, H=H, W=W)
                else:
                    x_feat = x_skip
                    x_B_T_H_W_D = rearrange(x_feat, "(T H W) B D -> B T H W D", T=T, H=H, W=W)

                features.append(x_B_T_H_W_D)

            new_name = f"block{block_idx}"
            if x_ctrl is not None and new_name in x_ctrl:
                x_ctrl_ = x_ctrl[new_name]
                x_ctrl_ = rearrange(x_ctrl_, "T H W B D -> (T H W) B D")
                x_skip = x_skip + x_ctrl_
            # If we have all of the features, we can exit early
            if return_features_early and len(features) == len(feature_indices):
                return features

        x_THW_B_D_before_gate = x_before_gate
        x_THW_B_D_skip = x_skip

        B, C, T_before_patchify, H_before_patchify, W_before_patchify = original_shape
        x_BT_HW_D_before_gate = rearrange(
            x_THW_B_D_before_gate,
            "(T H W) B D -> (B T) (H W) D",
            T=T_before_patchify // self.patch_temporal,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
        )
        x_BT_HW_D_skip = rearrange(
            x_THW_B_D_skip,
            "(T H W) B D -> (B T) (H W) D",
            T=T_before_patchify // self.patch_temporal,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
        )

        x_BT_HW_D = self.final_layer.forward_with_memory_save(
            x_BT_HW_D_before_gate=x_BT_HW_D_before_gate,
            x_BT_HW_D_skip=x_BT_HW_D_skip,
            gate_L_B_D=gate_L_B_D,
            emb_B_D=affline_emb_B_D,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        # This is to ensure x_BT_HW_D has the correct shape because
        # when we merge T, H, W into one dimension, x_BT_HW_D has shape (B * T * H * W, 1*1, D).
        x_BT_HW_D = x_BT_HW_D.view(
            B * T_before_patchify // self.patch_temporal,
            H_before_patchify // self.patch_spatial * W_before_patchify // self.patch_spatial,
            -1,
        )
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            H=H_before_patchify // self.patch_spatial,
            W=W_before_patchify // self.patch_spatial,
            t=self.patch_temporal,
            B=B,
        )
        if len(feature_indices) == 0:
            # no features requested, return only the model output
            return x_B_D_T_H_W
        else:
            # score and features； score, features
            return x_B_D_T_H_W, features

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        x_ctrl: Optional[dict] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        feature_indices: Optional[Container[int]] = None,
        return_features_early: bool = False,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            condition_video_augment_sigma: (B,) used in lvg(long video generation), we add noise with this sigma to augment condition input, the lvg model will condition on the condition_video_augment_sigma value;
                we need forward_before_blocks pass to the forward_before_blocks function.
        """
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        inputs = self.forward_before_blocks(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )
        x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
            inputs["x"],
            inputs["affline_emb_B_D"],
            inputs["crossattn_emb"],
            inputs["crossattn_mask"],
            inputs["rope_emb_L_1_1_D"],
            inputs["adaln_lora_B_3D"],
            inputs["original_shape"],
        )
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"]
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"

        if self.use_memory_save:
            return self.forward_blocks_memory_save(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D,
                adaln_lora_B_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                feature_indices,
                original_shape,
                x_ctrl,
                return_features_early,
            )

        return self.forward_blocks_regular(
            x,
            affline_emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D,
            adaln_lora_B_3D,
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            feature_indices,
            original_shape,
            x_ctrl,
            return_features_early,
        )

    @property
    def fsdp_wrap_block_cls(self):
        return DITBuildingBlock

    def enable_context_parallel(self, cp_group: ProcessGroup):
        cp_ranks = get_process_group_ranks(cp_group)
        cp_size = len(cp_ranks)
        # Set these attributes for spliting the data after embedding.
        self.cp_group = cp_group
        # Set these attributes for computing the loss.
        self.cp_size = cp_size

        self.pos_embedder.enable_context_parallel(cp_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(cp_group)
        # Loop through the model to set up context parallel.
        for block in self.blocks.values():
            for layer in block.blocks:
                if layer.block_type in ["mlp", "ff"]:
                    continue
                elif layer.block_type in ["cross_attn", "ca"]:
                    continue
                else:
                    layer.block.attn.attn_op.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())

        log.debug(f"[CP] Enable context parallelism with size {cp_size}")

    def disable_context_parallel(self):
        self.cp_group = None
        self.cp_size = None

        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # Loop through the model to disable context parallel.
        for block in self.blocks.values():
            for layer in block.blocks:
                if layer.block_type in ["mlp", "ff"]:
                    continue
                elif layer.block_type in ["cross_attn", "ca"]:
                    continue
                else:
                    layer.block.attn.attn_op.cp_group = None
                    layer.block.attn.attn_op.cp_ranks = None
                    layer.block.attn.attn_op.cp_stream = None

        log.debug("[CP] Disable context parallelism.")

    def enable_sequence_parallel(self):
        self._set_sequence_parallel(True)

    def disable_sequence_parallel(self):
        self._set_sequence_parallel(False)

    def _set_sequence_parallel(self, status: bool):
        self.sequence_parallel = status
        self.final_layer.sequence_parallel = status
        for block in self.blocks.values():
            for layer in block.blocks:
                if layer.block_type in ["full_attn", "fa", "cross_attn", "ca"]:
                    layer.block.attn.to_q[0].sequence_parallel = status
                    layer.block.attn.to_k[0].sequence_parallel = status
                    layer.block.attn.to_v[0].sequence_parallel = status
                    layer.block.attn.to_out[0].sequence_parallel = status
                    layer.block.attn.attn_op.sequence_parallel = status
                elif layer.block_type in ["mlp", "ff"]:
                    layer.block.layer1.sequence_parallel = status
                    layer.block.layer2.sequence_parallel = status
                else:
                    raise ValueError(f"Unknown block type {layer.block_type}")

    @property
    def is_context_parallel_enabled(self):
        return self.cp_group is not None
