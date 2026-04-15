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
from typing import Optional

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from megatron.core import parallel_state
from torch import nn
from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from cosmos_predict1.diffusion.module.attention import Attention, GPT2FeedForward
from cosmos_predict1.diffusion.training.tensor_parallel import gather_along_first_dim
from cosmos_predict1.utils import log


class SDXLTimesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        in_dype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(in_dype)


class SDXLTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.critical(
            f"Using AdaLN LoRA Flag: {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_3D = emb
            emb_B_D = sample
        else:
            emb_B_D = emb
            adaln_lora_B_3D = None

        return emb_B_D, adaln_lora_B_3D


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches and embedding each
    patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    - keep_spatio (bool): If True, the spatial dimensions are kept separate in the output tensor, otherwise, they are flattened. Default: False.
    - legacy_patch_emb (bool): If True, applies 3D convolutional layers for video inputs, otherwise, use Linear! The legacy model is for backward compatibility. Default: True.
    The output shape of the module depends on the `keep_spatio` flag. If `keep_spatio`=True, the output retains the spatial dimensions.
    Otherwise, the spatial dimensions are flattened into a single dimension.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
        keep_spatio=False,
        legacy_patch_emb: bool = True,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        assert keep_spatio, "Only support keep_spatio=True"
        self.keep_spatio = keep_spatio
        self.legacy_patch_emb = legacy_patch_emb

        if legacy_patch_emb:
            self.proj = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                bias=bias,
            )
            self.out = Rearrange("b c t h w -> b t h w c")
        else:
            self.proj = nn.Sequential(
                Rearrange(
                    "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                    r=temporal_patch_size,
                    m=spatial_patch_size,
                    n=spatial_patch_size,
                ),
                nn.Linear(
                    in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=bias
                ),
            )
            self.out = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class ExtraTokenPatchEmbed(PatchEmbed):
    def __init__(self, *args, out_channels: int = 768, keep_spatio: bool = False, **kwargs):
        assert keep_spatio, "ExtraTokenPatchEmbed only supports keep_spatio=True"
        super().__init__(*args, out_channels=out_channels, keep_spatio=keep_spatio, **kwargs)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, 1, 1, out_channels))
        self.spatial_token = nn.Parameter(torch.randn(1, 1, 1, 1, out_channels))

    def forward(self, x):
        x_B_T_H_W_C = super().forward(x)
        B, T, H, W, C = x_B_T_H_W_C.shape
        x_B_T_H_W_C = torch.cat(
            [
                x_B_T_H_W_C,
                self.temporal_token.repeat(B, 1, H, W, 1),
            ],
            dim=1,
        )
        x_B_T_H_W_C = torch.cat(
            [
                x_B_T_H_W_C,
                self.spatial_token.repeat(B, T, H, 1, 1),
            ],
            dim=3,
        )
        return x_B_T_H_W_C


class ExpertChoiceMoEGate(nn.Module):
    """
    ExpertChoiceMoEGate determines which tokens go
    to which experts (and how much to weigh each expert).

    Args:
        hidden_size (int): Dimensionality of input features.
        num_experts (int): Number of experts (E).
        capacity (int): Capacity (number of tokens) each expert can process (C).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity = capacity

        self.router = nn.Parameter(torch.empty((self.num_experts, self.hidden_size)))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.router)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input of shape (B, S, D)
        Returns:
            gating (Tensor): Gating weights of shape (B, E, C),
                             where E = num_experts, C = capacity (top-k).
            dispatch (Tensor): Dispatch mask of shape (B, E, C, S).
            index (Tensor): Indices of top-k tokens for each expert,
                            shape (B, E, C).
        """
        B, S, D = x.shape
        E, C = self.num_experts, self.capacity

        # token-expert affinity scores
        logits = torch.einsum("bsd,de->bse", x, self.router)
        affinity = torch.nn.functional.softmax(logits, dim=-1)  # (B, S, E)

        # gather topk tokens for each expert
        affinity_t = affinity.transpose(1, 2)  # (B, E, S)

        # select top-k tokens for each expert
        gating, index = torch.topk(affinity_t, k=C, dim=-1)  # (B, E, C)

        # one-hot dispatch mask
        dispatch = torch.nn.functional.one_hot(index, num_classes=S).float()  # (B, E, C, S)

        return gating, dispatch, index


class ExpertChoiceMoELayer(nn.Module):
    """
    ExpertChoiceMoELayer uses the ExpertChoiceMoEGate to route tokens
    to experts, process them, and then combine the outputs.

    Args:
        gate_hidden_size (int): Dimensionality of input features.
        ffn_hidden_size (int): Dimension of hidden layer in each expert feedforward (e.g., GPT2FeedForward).
        num_experts (int): Number of experts (E).
        capacity (int): Capacity (number of tokens) each expert can process (C).
        expert_cls (nn.Module): The class to instantiate each expert. Defaults to GPT2FeedForward.
        expert_kwargs (dict): Extra kwargs to pass to each expert class.
    """

    def __init__(
        self,
        gate_hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        capacity: int,
        expert_class: nn.Module = GPT2FeedForward,
        expert_kwargs=None,
    ):
        super().__init__()
        if not expert_kwargs:
            expert_kwargs = {}

        self.gate_hidden_size = gate_hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.capacity = capacity

        self.gate = ExpertChoiceMoEGate(gate_hidden_size, num_experts, capacity)

        self.experts = nn.ModuleList(
            [expert_class(gate_hidden_size, ffn_hidden_size, **expert_kwargs) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input of shape (B, S, D).

        Returns:
            x_out (Tensor): Output of shape (B, S, D), after dispatching tokens
                            to experts and combining their outputs.
        """
        B, S, D = x.shape
        E, C = self.num_experts, self.capacity

        # gating: (B, E, C)
        # dispatch: (B, E, C, S)
        gating, dispatch, index = self.gate(x)

        # collect input tokens for each expert
        x_in = torch.einsum("becs,bsd->becd", dispatch, x)

        # process through each expert
        expert_outputs = [self.experts[e](x_in[:, e]) for e in range(E)]

        x_e = torch.stack(expert_outputs, dim=1)  # (B, E, C, D)

        # gating: (B, E, C), dispatch: (B, E, C, S), x_e: (B, E, C, d)
        # x_out: (B, S, D)
        # each token is placed back to its location with weighting
        x_out = torch.einsum("becs,bec,becd->bsd", dispatch, gating, x_e)

        return x_out


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.sequence_parallel = getattr(parallel_state, "sequence_parallel", False)

    def forward(
        self,
        x_BT_HW_D,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)
        if self.sequence_parallel:
            x_T_B_HW_D = rearrange(x_BT_HW_D, "(b t) hw d -> t b hw d", b=B, t=T)
            x_T_B_HW_D = gather_along_first_dim(x_T_B_HW_D, parallel_state.get_tensor_model_parallel_group())
            x_BT_HW_D = rearrange(x_T_B_HW_D, "t b hw d -> (b t) hw d", b=B)

        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D

    def forward_with_memory_save(
        self,
        x_BT_HW_D_before_gate: torch.Tensor,
        x_BT_HW_D_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D_before_gate.shape[0] // B
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        gate_BT_1_D = repeat(gate_L_B_D, "1 b d -> (b t) 1 d", t=T)

        def _fn(_x_before_gate, _x_skip):
            previous_block_out = _x_skip + gate_BT_1_D * _x_before_gate
            _x = modulate(self.norm_final(previous_block_out), shift_BT_D, scale_BT_D)
            return self.linear(_x)

        return torch.utils.checkpoint.checkpoint(_fn, x_BT_HW_D_before_gate, x_BT_HW_D_skip, use_reentrant=False)


class VideoAttn(nn.Module):
    """
    Implements video attention with optional cross-attention capabilities.

    This module supports both self-attention within the video frames and cross-attention
    with an external context. It's designed to work with flattened spatial dimensions
    to accommodate for video input.

    Attributes:
        x_dim (int): Dimensionality of the input feature vectors.
        context_dim (Optional[int]): Dimensionality of the external context features.
            If None, the attention does not utilize external context.
        num_heads (int): Number of attention heads.
        bias (bool): If true, bias is added to the query, key, value projections.
        x_format (str): The shape format of x tenosor.
        n_views (int): Extra parameter used in multi-view diffusion model. It indicated total number of view we model together.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        bias: bool = False,
        x_format: str = "BTHWD",
        n_views: int = 1,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.x_format = x_format
        if self.x_format == "BTHWD":
            qkv_format = "bshd"
        elif self.x_format == "THWBD":
            qkv_format = "sbhd"
        else:
            raise NotImplementedError(f"Unsupported x_format: {self.x_format}")

        self.attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_bias=bias,
            qkv_norm="RRI",
            out_bias=bias,
            qkv_format=qkv_format,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for video attention.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D) representing batches of video data.
            context (Tensor): Context tensor of shape (B, M, D) or (M, B, D), where M is the sequence length of the context.
            crossattn_mask (Optional[Tensor]): An optional mask for cross-attention mechanisms.
            rope_emb_L_1_1_D (Optional[Tensor]): Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training. transformer_engine format

        Returns:
            Tensor: The output tensor with applied attention, maintaining the input shape.
        """

        if self.x_format == "BTHWD":
            if context is not None and self.n_views > 1:
                x_B_T_H_W_D = rearrange(x, "b (v t) h w d -> (v b) t h w d", v=self.n_views)
                context_B_M_D = rearrange(context, "b (v m) d -> (v b) m d", v=self.n_views)
            else:
                x_B_T_H_W_D = x
                context_B_M_D = context
            B, T, H, W, D = x_B_T_H_W_D.shape
            x_B_THW_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            x_B_THW_D = self.attn(x_B_THW_D, context_B_M_D, crossattn_mask, rope_emb=rope_emb_L_1_1_D)

            # reshape it back to video format
            x_B_T_H_W_D = rearrange(x_B_THW_D, "b (t h w) d -> b t h w d", h=H, w=W)
            if context is not None and self.n_views > 1:
                x_B_T_H_W_D = rearrange(x_B_T_H_W_D, "(v b) t h w d -> b (v t) h w d", v=self.n_views)
            return x_B_T_H_W_D
        elif self.x_format == "THWBD":
            if context is not None and self.n_views > 1:
                x_T_H_W_B_D = rearrange(x, "(v t) h w b d -> t h w (v b) d", v=self.n_views)
                context_M_B_D = rearrange(context, "(v m) b d -> m (v b) d", v=self.n_views)
            else:
                x_T_H_W_B_D = x
                context_M_B_D = context
            T, H, W, B, D = x_T_H_W_B_D.shape
            x_THW_B_D = rearrange(x_T_H_W_B_D, "t h w b d -> (t h w) b d")
            x_THW_B_D = self.attn(
                x_THW_B_D,
                context_M_B_D,
                crossattn_mask,
                rope_emb=rope_emb_L_1_1_D,
            )
            x_T_H_W_B_D = rearrange(x_THW_B_D, "(t h w) b d -> t h w b d", h=H, w=W)
            if context is not None and self.n_views > 1:
                x_T_H_W_B_D = rearrange(x_T_H_W_B_D, "t h w (v b) d -> (v t) h w b d", v=self.n_views)
            return x_T_H_W_B_D
        else:
            raise NotImplementedError(f"Unsupported x_format: {self.x_format}")


def checkpoint_norm_state(norm_state, x, scale, shift):
    normalized = norm_state(x)
    return normalized * (1 + scale) + shift


class DITBuildingBlock(nn.Module):
    """
    DIT Building Block for constructing various types of attention or MLP blocks dynamically based on a specified block type.

    This class instantiates different types of buildig block / attn and MLP based on config, and applies crossponding forward pass during training.

    Attributes:
        block_type (str): Type of block to be used ('spatial_sa', 'temporal_sa', 'cross_attn', 'full_attn', 'mlp').
        x_dim (int): Dimensionality of the input features.
        context_dim (Optional[int]): Dimensionality of the external context, required for cross attention blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Multiplier for the dimensionality of the MLP hidden layer compared to input.
        spatial_win_size (int): Window size for spatial self-attention.
        temporal_win_size (int): Window size for temporal self-attention.
        bias (bool): Whether to include bias in attention and MLP computations.
        mlp_dropout (float): Dropout rate for MLP blocks.
        n_views (int): Extra parameter used in multi-view diffusion model. It indicated total number of view we model together.
    """

    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_sizes: list = [],
        spatial_win_size: int = 1,
        temporal_win_size: int = 1,
        bias: bool = False,
        mlp_dropout: float = 0.0,
        x_format: str = "BTHWD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        n_views: int = 1,
    ) -> None:
        block_type = block_type.lower()

        super().__init__()
        self.x_format = x_format
        if block_type in ["cross_attn", "ca"]:
            self.block = VideoAttn(
                x_dim,
                context_dim,
                num_heads,
                bias=bias,
                x_format=self.x_format,
                n_views=n_views,
            )
        elif block_type in ["full_attn", "fa"]:
            self.block = VideoAttn(x_dim, None, num_heads, bias=bias, x_format=self.x_format)
        elif block_type in ["mlp", "ff"]:
            self.block = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.block_type = block_type
        self.use_adaln_lora = use_adaln_lora

        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, self.n_adaln_chunks * x_dim, bias=False))

    def forward_with_attn_memory_save(
        self,
        x_before_gate: torch.Tensor,
        x_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        del crossattn_mask
        assert isinstance(self.block, VideoAttn), "only support VideoAttn impl"
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_L_B_D, scale_L_B_D, _gate_L_B_D = (
            shift_B_D.unsqueeze(0),
            scale_B_D.unsqueeze(0),
            gate_B_D.unsqueeze(0),
        )

        def _fn(_x_before_gate, _x_skip, _context):
            previous_block_out = _x_skip + gate_L_B_D * _x_before_gate
            if extra_per_block_pos_emb is not None:
                previous_block_out = previous_block_out + extra_per_block_pos_emb
            _normalized_x = self.norm_state(previous_block_out)
            normalized_x = _normalized_x * (1 + scale_L_B_D) + shift_L_B_D
            # context = normalized_x if _context is None else _context
            context = normalized_x if self.block.attn.is_selfattn else _context
            return (
                self.block.attn.to_q[0](normalized_x),
                self.block.attn.to_k[0](context),
                self.block.attn.to_v[0](context),
                previous_block_out,
            )

        q, k, v, previous_block_out = torch.utils.checkpoint.checkpoint(
            _fn, x_before_gate, x_skip, crossattn_emb, use_reentrant=False
        )

        def attn_fn(_q, _k, _v):
            q, k, v = map(
                lambda t: rearrange(
                    t,
                    "b ... (n c) -> b ... n c",
                    n=self.block.attn.heads // self.block.attn.tp_size,
                    c=self.block.attn.dim_head,
                ),
                (_q, _k, _v),
            )
            q = self.block.attn.to_q[1](q)
            k = self.block.attn.to_k[1](k)
            v = self.block.attn.to_v[1](v)
            if self.block.attn.is_selfattn and rope_emb_L_1_1_D is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb_L_1_1_D, tensor_format=self.block.attn.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb_L_1_1_D, tensor_format=self.block.attn.qkv_format, fused=True)

            if self.block.attn.is_selfattn:
                return q, k, v

            seq_dim = self.block.attn.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            return self.block.attn.attn_op(
                q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None
            )  # [B, Mq, H, V]

        assert self.block.attn.backend == "transformer_engine", "Only support transformer_engine backend for now."

        if self.block.attn.is_selfattn:
            q, k, v = torch.utils.checkpoint.checkpoint(attn_fn, q, k, v, use_reentrant=False)
            seq_dim = self.block.attn.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            softmax_attn_output = self.block.attn.attn_op(
                q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None
            )  # [B, Mq, H, V]
        else:
            softmax_attn_output = torch.utils.checkpoint.checkpoint(attn_fn, q, k, v, use_reentrant=False)
        attn_out = self.block.attn.to_out(softmax_attn_output)
        return _gate_L_B_D, attn_out, previous_block_out

    def forward_with_x_attn_memory_save(
        self,
        x_before_gate: torch.Tensor,
        x_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        del crossattn_mask
        assert isinstance(self.block, VideoAttn)
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_L_B_D, scale_L_B_D, _gate_L_B_D = (
            shift_B_D.unsqueeze(0),
            scale_B_D.unsqueeze(0),
            gate_B_D.unsqueeze(0),
        )

        def _fn(_x_before_gate, _x_skip, _context):
            previous_block_out = _x_skip + gate_L_B_D * _x_before_gate
            if extra_per_block_pos_emb is not None:
                previous_block_out = previous_block_out + extra_per_block_pos_emb
            _normalized_x = self.norm_state(previous_block_out)
            normalized_x = _normalized_x * (1 + scale_L_B_D) + shift_L_B_D
            # context = normalized_x if _context is None else _context
            context = normalized_x if self.block.attn.is_selfattn else _context
            return (
                self.block.attn.to_q[0](normalized_x),
                self.block.attn.to_k[0](context),
                self.block.attn.to_v[0](context),
                previous_block_out,
            )

        q, k, v, previous_block_out = torch.utils.checkpoint.checkpoint(
            _fn, x_before_gate, x_skip, crossattn_emb, use_reentrant=False
        )

        def x_attn_fn(_q, _k, _v):
            q, k, v = map(
                lambda t: rearrange(
                    t,
                    "b ... (n c) -> b ... n c",
                    n=self.block.attn.heads // self.block.attn.tp_size,
                    c=self.block.attn.dim_head,
                ),
                (_q, _k, _v),
            )
            q = self.block.attn.to_q[1](q)
            k = self.block.attn.to_k[1](k)
            v = self.block.attn.to_v[1](v)
            if self.block.attn.is_selfattn and rope_emb_L_1_1_D is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb_L_1_1_D, tensor_format=self.block.attn.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb_L_1_1_D, tensor_format=self.block.attn.qkv_format, fused=True)

            seq_dim = self.block.attn.qkv_format.index("s")
            assert (
                q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
            ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
            softmax_attn_output = self.block.attn.attn_op(
                q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None
            )  # [B, Mq, H, V]
            return self.block.attn.to_out(softmax_attn_output)

        assert self.block.attn.backend == "transformer_engine", "Only support transformer_engine backend for now."

        attn_out = torch.utils.checkpoint.checkpoint(x_attn_fn, q, k, v, use_reentrant=False)
        return _gate_L_B_D, attn_out, previous_block_out

    def forward_with_ffn_memory_save(
        self,
        x_before_gate: torch.Tensor,
        x_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        del crossattn_emb, crossattn_mask, rope_emb_L_1_1_D
        assert isinstance(self.block, GPT2FeedForward)
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_L_B_D, scale_L_B_D, _gate_L_B_D = (
            shift_B_D.unsqueeze(0),
            scale_B_D.unsqueeze(0),
            gate_B_D.unsqueeze(0),
        )

        def _fn(_x_before_gate, _x_skip):
            previous_block_out = _x_skip + gate_L_B_D * _x_before_gate
            if extra_per_block_pos_emb is not None:
                previous_block_out = previous_block_out + extra_per_block_pos_emb
            _normalized_x = self.norm_state(previous_block_out)
            normalized_x = _normalized_x * (1 + scale_L_B_D) + shift_L_B_D

            assert self.block.dropout.p == 0.0, "we skip dropout to save memory"

            return self.block.layer1(normalized_x), previous_block_out

        intermediate_output, previous_block_out = torch.utils.checkpoint.checkpoint(
            _fn, x_before_gate, x_skip, use_reentrant=False
        )

        def _fn2(_x):
            _x = self.block.activation(_x)
            return self.block.layer2(_x)

        return (
            _gate_L_B_D,
            torch.utils.checkpoint.checkpoint(_fn2, intermediate_output, use_reentrant=False),
            previous_block_out,
        )

    def forward_with_ffn_memory_save_upgrade(
        self,
        x_before_gate: torch.Tensor,
        x_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        del crossattn_emb, crossattn_mask, rope_emb_L_1_1_D
        assert isinstance(self.block, GPT2FeedForward)
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_L_B_D, scale_L_B_D, _gate_L_B_D = (
            shift_B_D.unsqueeze(0),
            scale_B_D.unsqueeze(0),
            gate_B_D.unsqueeze(0),
        )

        def _fn2(_x):
            _x = self.block.activation(_x)
            return self.block.layer2(_x)

        def _fn(_x_before_gate, _x_skip):
            previous_block_out = _x_skip + gate_L_B_D * _x_before_gate
            if extra_per_block_pos_emb is not None:
                previous_block_out = previous_block_out + extra_per_block_pos_emb
            _normalized_x = self.norm_state(previous_block_out)
            normalized_x = _normalized_x * (1 + scale_L_B_D) + shift_L_B_D

            assert self.block.dropout.p == 0.0, "we skip dropout to save memory"

            return _fn2(self.block.layer1(normalized_x)), previous_block_out

        output, previous_block_out = torch.utils.checkpoint.checkpoint(_fn, x_before_gate, x_skip, use_reentrant=False)

        return (
            _gate_L_B_D,
            output,
            previous_block_out,
        )

    def forward_with_memory_save(
        self,
        x_before_gate: torch.Tensor,
        x_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        if isinstance(self.block, VideoAttn):
            if self.block.attn.is_selfattn:
                fn = self.forward_with_attn_memory_save
            else:
                fn = self.forward_with_x_attn_memory_save
        else:
            # fn = self.forward_with_ffn_memory_save
            fn = self.forward_with_ffn_memory_save_upgrade
        return fn(
            x_before_gate,
            x_skip,
            gate_L_B_D,
            emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D,
            adaln_lora_B_3D,
            extra_per_block_pos_emb,
        )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for dynamically configured blocks with adaptive normalization.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D).
            emb_B_D (Tensor): Embedding tensor for adaptive layer normalization modulation.
            crossattn_emb (Tensor): Tensor for cross-attention blocks.
            crossattn_mask (Optional[Tensor]): Optional mask for cross-attention.
            rope_emb_L_1_1_D (Optional[Tensor]): Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training. transformer_engine format

        Returns:
            Tensor: The output tensor after processing through the configured block and adaptive normalization.
        """
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        if self.x_format == "BTHWD":
            shift_B_1_1_1_D, scale_B_1_1_1_D, gate_B_1_1_1_D = (
                shift_B_D.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                scale_B_D.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                gate_B_D.unsqueeze(1).unsqueeze(2).unsqueeze(3),
            )
            if self.block_type in ["spatial_sa", "temporal_sa", "window_attn", "ssa", "tsa", "wa"]:
                x = x + gate_B_1_1_1_D * self.block(
                    self.norm_state(x) * (1 + scale_B_1_1_1_D) + shift_B_1_1_1_D,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                )
            elif self.block_type in ["full_attn", "fa"]:
                x = x + gate_B_1_1_1_D * self.block(
                    self.norm_state(x) * (1 + scale_B_1_1_1_D) + shift_B_1_1_1_D,
                    context=None,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                )
            elif self.block_type in ["cross_attn", "ca"]:
                x = x + gate_B_1_1_1_D * self.block(
                    self.norm_state(x) * (1 + scale_B_1_1_1_D) + shift_B_1_1_1_D,
                    crossattn_emb,
                    crossattn_mask,
                )
            elif self.block_type in ["mlp", "ff"]:
                x = x + gate_B_1_1_1_D * self.block(
                    self.norm_state(x) * (1 + scale_B_1_1_1_D) + shift_B_1_1_1_D,
                )
            else:
                raise ValueError(f"Unknown block type: {self.block_type}")
        elif self.x_format == "THWBD":
            shift_1_1_1_B_D, scale_1_1_1_B_D, gate_1_1_1_B_D = (
                shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            )

            if self.block_type in ["mlp", "ff"]:
                x = x + gate_1_1_1_B_D * self.block(
                    torch.utils.checkpoint.checkpoint(
                        checkpoint_norm_state, self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D, use_reentrant=False
                    ),
                )
            elif self.block_type in ["full_attn", "fa"]:
                x = x + gate_1_1_1_B_D * self.block(
                    torch.utils.checkpoint.checkpoint(
                        checkpoint_norm_state, self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D, use_reentrant=False
                    ),
                    context=None,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                )
            elif self.block_type in ["cross_attn", "ca"]:
                x = x + gate_1_1_1_B_D * self.block(
                    torch.utils.checkpoint.checkpoint(
                        checkpoint_norm_state, self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D, use_reentrant=False
                    ),
                    context=crossattn_emb,
                    crossattn_mask=crossattn_mask,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                )
            else:
                raise ValueError(f"Unknown block type: {self.block_type}")
        else:
            raise NotImplementedError(f"Unsupported x_format: {self.x_format}")
        return x


class GeneralDITTransformerBlock(nn.Module):
    """
    This class is a wrapper for a list of DITBuildingBlock.
    It's not essential, refactor it if needed.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        block_config: str,
        mlp_ratio: float = 4.0,
        window_sizes: list = [],
        spatial_attn_win_size: int = 1,
        temporal_attn_win_size: int = 1,
        use_checkpoint: bool = False,
        x_format: str = "BTHWD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        n_views: int = 1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.x_format = x_format
        for block_type in block_config.split("-"):
            self.blocks.append(
                DITBuildingBlock(
                    block_type,
                    x_dim,
                    context_dim,
                    num_heads,
                    mlp_ratio,
                    window_sizes,
                    spatial_attn_win_size,
                    temporal_attn_win_size,
                    x_format=self.x_format,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    n_views=n_views,
                )
            )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D,
                adaln_lora_B_3D,
                extra_per_block_pos_emb,
            )
        else:
            return self._forward(
                x, emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, extra_per_block_pos_emb
            )

    def _forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x = x + extra_per_block_pos_emb
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
            )
        return x

    def set_memory_save(self, mode: bool = True):
        # (qsh) to make fsdp happy!
        #! IMPORTANT!
        if mode:
            self.forward = self.forward_with_memory_save
            for block in self.blocks:
                block.forward = block.forward_with_memory_save
        else:
            raise NotImplementedError("Not implemented yet.")

    def forward_with_memory_save(
        self,
        x_before_gate: torch.Tensor,
        x_skip: torch.Tensor,
        gate_L_B_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ):
        for block in self.blocks:
            gate_L_B_D, x_before_gate, x_skip = block.forward(
                x_before_gate,
                x_skip,
                gate_L_B_D,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D,
                adaln_lora_B_3D,
                extra_per_block_pos_emb,
            )
            extra_per_block_pos_emb = None
        return gate_L_B_D, x_before_gate, x_skip
