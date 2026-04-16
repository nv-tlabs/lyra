# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import math
import torch.nn as nn
from typing import Optional, List, Tuple
import torch.amp as amp
from einops import rearrange
from torchvision import transforms
from torch.distributed._composable.fsdp import fully_shard
from lyra_2._src.networks.wan2pt1 import (
    WanLayerNorm,
    WanSelfAttention,
    WAN_CROSSATTENTION_CLASSES,
    VideoSize,
    sinusoidal_embedding_1d,
    Head,
    MLPProj,
    VideoRopePosition3DEmb,
)
from lyra_2._src.callbacks.model_weights_stats import WeightTrainingStat
from lyra_2._src.modules.selective_activation_checkpoint import (
    SACConfig,
    CheckpointMode,
    mm_only_context_fn,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from lyra_2._ext.imaginaire.utils import log
from einops import repeat
from lyra_2._src.utils.context_parallel import (
    split_inputs_cp,
    cat_outputs_cp_with_grad,
)
from torch.distributed import ProcessGroup, get_process_group_ranks


class Lyra2AttentionBlock(nn.Module):
    """Attention block copied from WanAttentionBlock with optional camera/buffer embedding.

    If cam_dim > 0, constructs a camera encoder and injects the camera embedding
    into the self-attention input (pre-attention add). If buffer_dim > 0, injects
    buffer embeddings similarly.
    """
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        cp_comm_type="p2p",
        cam_dim: int = 0,
        buffer_dim: int = 0,
        buffer_sincos_multires: int = 0,
        inject_kq_only: bool = False,
        buffer_mlp_squeeze_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.inject_kq_only = bool(inject_kq_only)
        self.buffer_mlp_squeeze_dim = int(buffer_mlp_squeeze_dim)

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, cp_comm_type)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps, cp_comm_type
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # optional camera/buffer encoders
        self.cam_dim = cam_dim
        self.cam_encoder = nn.Linear(self.cam_dim, self.dim, bias=False) if self.cam_dim > 0 else None
        self.buffer_dim = buffer_dim
        self.buffer_sincos_multires = int(buffer_sincos_multires)
        buffer_embed_dim = self.buffer_dim
        if self.buffer_sincos_multires > 0 and self.buffer_dim > 0:
            buffer_embed_dim = self.buffer_dim * 2 * self.buffer_sincos_multires
        if self.buffer_dim > 0:
            if self.buffer_mlp_squeeze_dim > 0:
                self.buffer_encoder = nn.Sequential(
                    nn.Linear(buffer_embed_dim, self.buffer_mlp_squeeze_dim, bias=False),
                    nn.Linear(self.buffer_mlp_squeeze_dim, self.dim, bias=False),
                )
            else:
                self.buffer_encoder = nn.Linear(buffer_embed_dim, self.dim, bias=False)
        else:
            self.buffer_encoder = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.cross_attn.init_weights()

        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

        std = 1.0 / (self.dim ** 0.5)
        torch.nn.init.trunc_normal_(self.modulation, std=std)
        if self.cam_encoder is not None:
            torch.nn.init.trunc_normal_(self.cam_encoder.weight, std=std, a=-3 * std, b=3 * std)
        if self.buffer_encoder is not None:
            if isinstance(self.buffer_encoder, nn.Sequential):
                for layer in self.buffer_encoder:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.trunc_normal_(layer.weight, std=std, a=-3 * std, b=3 * std)
            else:
                torch.nn.init.trunc_normal_(self.buffer_encoder.weight, std=std, a=-3 * std, b=3 * std)

    @staticmethod
    def _sincos_embed(x: torch.Tensor, multires: int) -> torch.Tensor:
        if multires <= 0:
            return x
        x_float = x.float()
        embeds = []
        for i in range(int(multires)):
            freq = (2.0 ** i) * math.pi
            embeds.append(torch.sin(x_float * freq))
            embeds.append(torch.cos(x_float * freq))
        out = torch.cat(embeds, dim=-1)
        return out.type_as(x)

    def forward(
        self,
        x,
        e,
        seq_lens,
        video_size: VideoSize,
        freqs,
        context,
        context_lens,
        camera: Optional[torch.Tensor] = None,
        buffer: Optional[torch.Tensor] = None,
    ):
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # Self-attention with optional camera injection
        if camera is not None:
            assert self.cam_encoder is not None
            cam_emb = self.cam_encoder(camera)
        else:
            if self.cam_encoder is not None:
                raise ValueError("cam_encoder is enabled but camera tokens are None")
            cam_emb = 0
        if buffer is not None:
            if self.inject_kq_only:
                validity = buffer[..., -1:]  # [B, L, 1]
                buffer = buffer[..., :-1]
            assert self.buffer_encoder is not None
            if self.buffer_sincos_multires > 0:
                buffer = self._sincos_embed(buffer, self.buffer_sincos_multires)
            buf_emb = self.buffer_encoder(buffer)
            if self.inject_kq_only:
                buf_emb = buf_emb * validity
        else:
            if self.buffer_sincos_multires > 0 and self.buffer_encoder is not None:
                raise ValueError("buffer_sincos_multires>0 requires buffer tokens, but buffer is None")
            buf_emb = 0

        y = (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x)

        if self.inject_kq_only:
            kq_bias = cam_emb + buf_emb
            if isinstance(kq_bias, (int, float)) and kq_bias == 0:
                kq_bias = None
            y = self.self_attn(y, seq_lens, video_size, freqs, kq_bias=kq_bias)
        else:
            y = self.self_attn(y + cam_emb + buf_emb, seq_lens, video_size, freqs)

        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].type_as(x)

        # cross-attn + ffn (same as base)
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).type_as(x))
            with amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5].type_as(x)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Lyra2WanModel(WeightTrainingStat):
    """WAN backbone with Lyra2 modifications inlined.

    - Copies core logic from WanModel (no subclassing).
    - Adds clean patch embeddings and Lyra2-aware forward.
    - Optionally injects camera conditioning (Plücker) via attention blocks.
    """

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        concat_padding_mask: bool = False,
        sac_config: SACConfig = SACConfig(),
        cp_comm_type: str = "p2p",
        postpone_checkpoint: bool = False,
        conv_patchify: bool = False,
        use_plucker_condition: bool = False,
        buffer_in_dim: int = 0,
        buffer_pixelshuffle: bool = False,
        buffer_sincos_multires: int = 0,
        use_correspondence: bool = False,
        inject_kq_only: bool = False,
        buffer_mlp_squeeze_dim: int = 0,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.concat_padding_mask = concat_padding_mask
        self.cp_comm_type = cp_comm_type
        self.conv_patchify = conv_patchify
        self.use_plucker_condition = bool(use_plucker_condition)
        self.buffer_in_dim = int(buffer_in_dim)
        self.buffer_pixelshuffle = bool(buffer_pixelshuffle)
        self.buffer_sincos_multires = int(buffer_sincos_multires)
        self.use_correspondence = bool(use_correspondence)
        self.inject_kq_only = bool(inject_kq_only)
        self.buffer_mlp_squeeze_dim = int(buffer_mlp_squeeze_dim)

        # Clean-embedding holders (lazy init)
        self.clean_patch_embeddings: nn.ModuleList | None = None
        self.clean_kernel_sizes: list[int] | None = None
        self.clean_kernel_types: list[str] | None = None
        self.patch_embedding_buffer: nn.Linear | None = None

        # CP state
        self.cp_group: Optional[ProcessGroup] = None
        self._is_context_parallel_enabled: bool = False

        # embeddings
        in_dim_eff = in_dim + 1 if self.concat_padding_mask else in_dim
        if self.conv_patchify:
            self.patch_embedding = nn.Conv3d(in_dim_eff, dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.patch_embedding = nn.Linear(in_dim_eff * patch_size[0] * patch_size[1] * patch_size[2], dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        cam_dim = 1536 if self.use_plucker_condition else 0
        buffer_dim = 0
        if self.use_correspondence:
            if self.buffer_pixelshuffle:
                pt, ph, pw = self.patch_size
                buffer_dim = int(self.buffer_in_dim) * pt * ph * pw
            else:
                buffer_dim = self.dim
        self.blocks = nn.ModuleList(
            [
                Lyra2AttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    self.cp_comm_type,
                    cam_dim=cam_dim,
                    buffer_dim=buffer_dim,
                    buffer_sincos_multires=self.buffer_sincos_multires,
                    inject_kq_only=self.inject_kq_only,
                    buffer_mlp_squeeze_dim=self.buffer_mlp_squeeze_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # rope position embedding
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.rope_position_embedding = VideoRopePosition3DEmb(
            head_dim=d,
            len_h=128,
            len_w=128,
            len_t=32,
        )

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()
        # SAC (same behavior as base)
        self.sac_config = sac_config
        if not postpone_checkpoint:
            self.enable_selective_checkpoint(sac_config, self.blocks)

    # ------------------------- Clean Embeddings -------------------------
    def init_clean_patch_embeddings(self, clean_latent_frame_kernel_sizes: List[int], clean_latent_frame_kernel_types: List[str] | None = None) -> None:
        """Construct clean patch embedding layers without copying weights.

        This only creates `nn.Linear` layers with the correct input feature size
        for each enlarged patch. Weight copying from the base `patch_embedding`
        is deferred to `copy_weights_to_clean_patch_embeddings` and should be
        called at training start.
        """
        if self.clean_patch_embeddings is not None:
            return

        pt, ph, pw = self.patch_size
        in_dim = self.in_dim + (1 if self.concat_padding_mask else 0)
        base_linear: nn.Linear = self.patch_embedding
        assert isinstance(base_linear, nn.Linear)

        # Create holders
        self.clean_patch_embeddings = nn.ModuleList()
        # Ensure newly created layers match dtype/device of the base embedding
        base_dtype = base_linear.weight.dtype
        base_device = base_linear.weight.device
        base_bias_is_not_none = base_linear.bias is not None

        # Default to temporal kernels if types are not provided (backward compatible)
        if clean_latent_frame_kernel_types is None:
            clean_latent_frame_kernel_types = ["k"] * len(clean_latent_frame_kernel_sizes)

        assert len(clean_latent_frame_kernel_types) == len(clean_latent_frame_kernel_sizes), "kernel sizes/types length mismatch"

        for k, t in zip(clean_latent_frame_kernel_sizes, clean_latent_frame_kernel_types):
            if t == "s":  # spatial-only packing (T unchanged)
                new_pt, new_ph, new_pw = pt, ph * k, pw * k
            else:  # default temporal-style packing (THW)
                new_pt, new_ph, new_pw = pt * k, ph * k, pw * k
            new_in_features = in_dim * new_pt * new_ph * new_pw
            clean_lin = nn.Linear(new_in_features, self.dim, bias=base_bias_is_not_none)
            clean_lin = clean_lin.to(dtype=base_dtype, device=base_device)
            self.clean_patch_embeddings.append(clean_lin)

        # Mark structure ready but weights not yet copied
        self.clean_kernel_sizes = list(clean_latent_frame_kernel_sizes)
        self.clean_kernel_types = list(clean_latent_frame_kernel_types)
        log.info(
            f"Constructed {len(clean_latent_frame_kernel_sizes)} clean patch embedding layers (weights not yet copied)."
        )

    def init_patch_embedding_buffer(self, buffer_in_dim: int) -> None:
        """Construct an extra patch embedding for buffer inputs."""
        if buffer_in_dim <= 0 or self.patch_embedding_buffer is not None:
            return
        pt, ph, pw = self.patch_size
        base_linear: nn.Linear = self.patch_embedding
        assert isinstance(base_linear, nn.Linear)

        # Ensure newly created layer matches dtype/device of the base embedding
        base_dtype = base_linear.weight.dtype
        base_device = base_linear.weight.device
        base_bias_is_not_none = base_linear.bias is not None

        in_dim_eff = self.in_dim + (1 if self.concat_padding_mask else 0)
        total_in_features = (in_dim_eff + int(buffer_in_dim)) * pt * ph * pw
        buf_lin = nn.Linear(total_in_features, self.dim, bias=base_bias_is_not_none)
        buf_lin = buf_lin.to(dtype=base_dtype, device=base_device)
        self.patch_embedding_buffer = buf_lin
        self.buffer_in_dim = int(buffer_in_dim)
        log.info(
            f"Constructed patch_embedding_buffer with extra_in_dim={self.buffer_in_dim} (weights not yet copied)."
        )

    def copy_weights_to_clean_patch_embeddings(self) -> None:
        """Copy/base-initialize clean patch embeddings from `self.patch_embedding`.

        Tiling/averaging follows Conv3d style weight expansion:
        - Reshape base weight [dim, c*pt*ph*pw] to [dim, c, pt, ph, pw]
        - For temporal kernels ('k'): tile k along (pt, ph, pw), divide by k^3
        - For spatial kernels ('s'): tile k along (ph, pw), divide by k^2 (pt unchanged)
        Bias is copied directly if present.
        """
        assert self.clean_patch_embeddings is not None, "Call init_clean_patch_embeddings first."
        assert self.clean_kernel_sizes is not None, "clean_kernel_sizes must be set in init_clean_patch_embeddings."
        assert self.clean_kernel_types is not None, "clean_kernel_types must be set in init_clean_patch_embeddings."

        pt, ph, pw = self.patch_size
        in_dim = self.in_dim + (1 if self.concat_padding_mask else 0)
        base_linear: nn.Linear = self.patch_embedding
        assert isinstance(base_linear, nn.Linear)

        with torch.no_grad():
            base_weight = base_linear.weight.detach()  # [dim, in_dim*pt*ph*pw]
            base_bias = base_linear.bias.detach() if base_linear.bias is not None else None

            for clean_lin, k, t in zip(self.clean_patch_embeddings, self.clean_kernel_sizes, self.clean_kernel_types):
                # Prepare tiled weights
                tiled = rearrange(
                    base_weight,
                    "o (c pt ph pw) -> o c pt ph pw",
                    c=in_dim,
                    pt=pt,
                    ph=ph,
                    pw=pw,
                )
                if t == "s":
                    # Only H, W are expanded; T unchanged
                    tiled = repeat(
                        tiled,
                        "o c pt ph pw -> o c pt (ph hk) (pw wk)",
                        hk=k,
                        wk=k,
                    )
                    divisor = (k ** 2)
                else:
                    # Temporal-style: expand along T, H, W
                    tiled = repeat(
                        tiled,
                        "o c pt ph pw -> o c (pt tk) (ph hk) (pw wk)",
                        tk=k,
                        hk=k,
                        wk=k,
                    )
                    divisor = (k ** 3)
                tiled = rearrange(tiled, "o c pt ph pw -> o (c pt ph pw)")
                tiled = tiled / divisor
                clean_lin.weight.copy_(tiled)
                clean_lin.bias.copy_(base_bias)

            if self.patch_embedding_buffer is not None and not self.buffer_pixelshuffle:
                buf_lin = self.patch_embedding_buffer
                buf_lin.weight.zero_()
                buf_lin.weight[:, : base_weight.shape[1]].copy_(base_weight)
                if buf_lin.bias is not None and base_bias is not None:
                    buf_lin.bias.copy_(base_bias)
                log.info(
                    f"patch_embedding_buffer weight shape={tuple(buf_lin.weight.shape)}, "
                    f"base patch_embedding weight shape={tuple(base_weight.shape)}"
                )

        log.info("Copied base patch_embedding weights into clean_patch_embeddings and marked initialized.")

    # --------------------------- Utilities ----------------------------
    @staticmethod
    def _pad_for_linear_patch(x: torch.Tensor, kernel: Tuple[int, int, int]) -> torch.Tensor:
        """Pad a BCHWT tensor so T/H/W are divisible by kernel.

        Args:
            x: Tensor [B, C, T, H, W]
            kernel: (kt, kh, kw)
        """
        _, _, t, h, w = x.shape
        kt, kh, kw = kernel
        pad_t = (kt - (t % kt)) % kt
        pad_h = (kh - (h % kh)) % kh
        pad_w = (kw - (w % kw)) % kw
        if pad_t or pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
        return x

    def _patchify_linear(self, x: torch.Tensor, patch: Tuple[int, int, int], lin: nn.Linear) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Patchify via einops+Linear, returning tokens and (f,h,w) grid size.

        Args:
            x: [B, C, T, H, W]
            patch: (pt, ph, pw)
            lin: Linear mapping from flattened patch to dim
        Returns:
            x_tokens: [B, (f*h*w), dim]
            grid_size: (f, h, w)
        """
        pt, ph, pw = patch
        x = self._pad_for_linear_patch(x, patch)
        b, c, t, h, w = x.shape
        f, hh, ww = t // pt, h // ph, w // pw
        # b c (f pt) (hh ph) (ww pw) -> b f hh ww (c pt ph pw)
        x = rearrange(x, "b c (f pt) (hh ph) (ww pw) -> b f hh ww (c pt ph pw)", f=f, pt=pt, hh=hh, ph=ph, ww=ww, pw=pw)
        x = lin(x)
        x = rearrange(x, "b f h w d -> b (f h w) d")
        return x, (f, hh, ww)

    def _pixelshuffle_tokens(self, x: torch.Tensor, patch: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Rearrange patches into channels without a linear projection."""
        pt, ph, pw = patch
        x = self._pad_for_linear_patch(x, patch)
        b, c, t, h, w = x.shape
        f, hh, ww = t // pt, h // ph, w // pw
        x = rearrange(
            x,
            "b c (f pt) (hh ph) (ww pw) -> b (f hh ww) (c pt ph pw)",
            f=f,
            pt=pt,
            hh=hh,
            ph=ph,
            ww=ww,
            pw=pw,
        )
        return x, (f, hh, ww)

    # ------------------------- Lyra2 Path -------------------------
    def _patchify_lyra2(
        self,
        x: torch.Tensor,
        framepack_indices: torch.Tensor,
        framepack_splits: List[int],
        framepack_kernel_ids: List[int],
        framepack_kernel_types: List[str] | None = None,
        camera: Optional[torch.Tensor] = None,
        buffer_B_C_T_H_W: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Tuple[int, int], Tuple[int, int, int]]:
        """Lyra2-aware patchify.

        Splits time dimension according to `framepack_splits`. For each chunk,
        - if kernel_id == -1: use base patch size and base patch_embedding
        - else: use clean_patch_embeddings[kernel_id] with enlarged patch sizes
          If corresponding kernel type is 's', enlarge only H and W (T unchanged)

        Returns:
            x_tokens: [B, L, dim]
            freqs_tokens: [L, 1, 1, head_dim] as produced by rope embedder per token
            camera_tokens: [B, L, cam_dim] if camera is provided, else None
            buffer_tokens: [B, L, dim] if buffer is injected, else None
            gen_range: (gen_start, gen_end) token range for generation part
            gen_grid: (f, h, w) grid for generation part
        """
        assert self.clean_patch_embeddings is not None, (
            "clean_patch_embeddings must be initialized before using Lyra2"
        )
        xs = x[:,:,framepack_indices].split(framepack_splits, dim=2)  # split along T
        inds = framepack_indices.split(framepack_splits)

        # Determine base T,H,W for token grid and precompute base RoPE freqs once
        _, _, T_total, H, W = x.shape
        pt, ph, pw = self.patch_size
        f_base = T_total // pt
        h_base = H // ph
        w_base = W // pw
        freqs_base = self.rope_position_embedding.generate_embeddings(
            B_T_H_W_C=torch.Size([x.shape[0], f_base, h_base, w_base, self.dim // self.num_heads])
        )
        # (f*h*w,1,1,d) -> (1,d,f,h,w)
        freqs_base_5d = rearrange(freqs_base, "(f h w) 1 1 d -> 1 d f h w", f=f_base, h=h_base, w=w_base)

        token_chunks: List[torch.Tensor] = []
        freq_chunks: List[torch.Tensor] = []
        cam_chunks: List[torch.Tensor] = [] if camera is not None else []
        use_buffer_tokens = bool(self.use_correspondence and buffer_B_C_T_H_W is not None)
        buf_chunks: List[torch.Tensor] = [] if use_buffer_tokens else []
        buf_validity_chunks: List[torch.Tensor] = [] if use_buffer_tokens else []
        buf_splits = None
        buffer_full_match = False
        if buffer_B_C_T_H_W is not None:
            buffer_full_match = (int(buffer_B_C_T_H_W.shape[2]) == int(x.shape[2]))
            if buffer_full_match:
                buf_splits = buffer_B_C_T_H_W[:, :, framepack_indices].split(framepack_splits, dim=2)
        gen_start = None
        gen_end = None
        total_tokens = 0
        gen_grid = (0, 0, 0)
        buffer_token_dim = None
        if use_buffer_tokens:
            if self.buffer_pixelshuffle:
                buffer_token_dim = int(self.buffer_in_dim) * pt * ph * pw
            else:
                buffer_token_dim = int(self.dim)

        for i, x_chunk in enumerate(xs):
            kid = framepack_kernel_ids[i]
            ktype = None
            if framepack_kernel_types is not None and i < len(framepack_kernel_types):
                ktype = framepack_kernel_types[i]
            if kid == -1:
                # Generated/new segment uses base embedding
                if buffer_B_C_T_H_W is not None and self.use_correspondence:
                    x_tokens, (f, h, w) = self._patchify_linear(x_chunk, self.patch_size, self.patch_embedding)
                    buf = buf_splits[i] if buffer_full_match else buffer_B_C_T_H_W
                    buf = buf.to(dtype=x_chunk.dtype, device=x_chunk.device)
                    assert buf.shape[2] == x_chunk.shape[2], (
                        f"Buffer T={buf.shape[2]} must match latent T={x_chunk.shape[2]}"
                    )
                    assert buf.shape[-2:] == x_chunk.shape[-2:], "Buffer spatial size must match latent size."
                    if not self.buffer_pixelshuffle:
                        raise ValueError("use_correspondence requires buffer_pixelshuffle=True")
                    buf_tokens, _ = self._pixelshuffle_tokens(buf, self.patch_size)
                    buf_chunks.append(buf_tokens)
                    buf_validity_chunks.append(
                        torch.ones(buf_tokens.shape[0], buf_tokens.shape[1], 1, device=buf_tokens.device, dtype=buf_tokens.dtype)
                    )
                else:
                    x_tokens, (f, h, w) = self._patchify_linear(x_chunk, self.patch_size, self.patch_embedding)
                token_chunks.append(x_tokens)
                if gen_start is None:
                    gen_start = total_tokens
                total_tokens += x_tokens.shape[1]
                gen_end = total_tokens
                gen_grid = (f, h, w)

                # Slice base precomputed freqs along T using provided indices (with padding if needed)
                t_idx = inds[i].to(device=freqs_base_5d.device, dtype=torch.long)
                if f > t_idx.numel():
                    pad_t = f - t_idx.numel()
                    t_idx = torch.cat([t_idx, t_idx[-1:].repeat(pad_t)], dim=0)
                freqs_sel = freqs_base_5d[:, :, t_idx, :, :]
                freqs_tokens = rearrange(freqs_sel[0], "d f h w -> (f h w) 1 1 d")
                freq_chunks.append(freqs_tokens)

                # Camera tokens for generated segment (no pooling; slice T to match f)
                if camera is not None:
                    cam_base_5d = camera  # [B, D_cam, f_base, h_base, w_base]
                    cam_t_idx = t_idx.to(device=cam_base_5d.device, dtype=torch.long)
                    cam_sel = cam_base_5d[:, :, cam_t_idx, :, :]
                    cam_tokens = rearrange(cam_sel, "b d f h w -> b (f h w) d").type_as(x_tokens)
                    cam_chunks.append(cam_tokens)
            else:
                # History/clean segment uses enlarged clean embedding
                assert self.clean_kernel_sizes is not None
                kernel_factor = int(self.clean_kernel_sizes[kid])
                clean_lin = self.clean_patch_embeddings[kid]
                if (self.clean_kernel_types is not None and self.clean_kernel_types[kid] == "s") or ktype == "s":
                    # Spatial-only packing
                    enlarged_patch = (
                        self.patch_size[0],
                        self.patch_size[1] * kernel_factor,
                        self.patch_size[2] * kernel_factor,
                    )
                    pool_k = (
                        1,
                        enlarged_patch[1] // self.patch_size[1],
                        enlarged_patch[2] // self.patch_size[2],
                    )
                else:
                    # Temporal packing
                    enlarged_patch = (
                        self.patch_size[0] * kernel_factor,
                        self.patch_size[1] * kernel_factor,
                        self.patch_size[2] * kernel_factor,
                    )
                    pool_k = (
                        enlarged_patch[0] // self.patch_size[0],
                        enlarged_patch[1] // self.patch_size[1],
                        enlarged_patch[2] // self.patch_size[2],
                    )
                x_tokens, (f, h, w) = self._patchify_linear(x_chunk, enlarged_patch, clean_lin)
                token_chunks.append(x_tokens)
                total_tokens += x_tokens.shape[1]
                if use_buffer_tokens:
                    if buffer_full_match:
                        buf = buf_splits[i].to(dtype=x_chunk.dtype, device=x_chunk.device)
                        if ktype == "s":
                            if kernel_factor <= 1:
                                buf_tokens, _ = self._pixelshuffle_tokens(buf, self.patch_size)
                            else:
                                pool_k_buf = (1, pool_k[1], pool_k[2])
                                buf_pooled = torch.nn.functional.avg_pool3d(
                                    buf.float(),
                                    kernel_size=pool_k_buf,
                                    stride=pool_k_buf,
                                )
                                buf_tokens, _ = self._pixelshuffle_tokens(buf_pooled.type_as(buf), self.patch_size)
                            buf_is_real = True
                        else:
                            buf_tokens = torch.full(
                                (x_tokens.shape[0], x_tokens.shape[1], int(buffer_token_dim)),
                                -1.0,
                                device=x_tokens.device,
                                dtype=x_tokens.dtype,
                            )
                            buf_is_real = False
                    else:
                        # Non-full buffer (gen-only): history buffer tokens should be empty.
                        fill_val = -1.0 if ktype != "s" else 0.0
                        buf_tokens = torch.full(
                            (x_tokens.shape[0], x_tokens.shape[1], int(buffer_token_dim)),
                            fill_val,
                            device=x_tokens.device,
                            dtype=x_tokens.dtype,
                        )
                        buf_is_real = False
                    buf_chunks.append(buf_tokens)
                    validity_val = 1.0 if buf_is_real else 0.0
                    buf_validity_chunks.append(torch.full(
                        (buf_tokens.shape[0], buf_tokens.shape[1], 1),
                        validity_val,
                        device=buf_tokens.device,
                        dtype=buf_tokens.dtype,
                    ))

                # Pool from base freqs with kernel equal to ratio enlarged/base
                # Slice along T using provided indices and pad so dims divisible by pool_k (except when pool_k[0]==1)
                t_idx = inds[i].to(device=freqs_base_5d.device, dtype=torch.long)
                if pool_k[0] > 1:
                    pad_t = (-t_idx.numel()) % pool_k[0]
                    if pad_t:
                        t_idx = torch.cat([t_idx, t_idx[-1:].repeat(pad_t)], dim=0)
                freqs_sel = freqs_base_5d[:, :, t_idx, :, :]
                pad_h = (-h_base) % pool_k[1]
                pad_w = (-w_base) % pool_k[2]
                if pad_h or pad_w:
                    freqs_sel = torch.nn.functional.pad(
                        freqs_sel,
                        (0, pad_w, 0, pad_h, 0, 0),  # pad W, H (T handled by index pad)
                        mode="replicate",
                    )
                freqs_pooled = torch.nn.functional.avg_pool3d(
                    freqs_sel.float(),
                    kernel_size=(pool_k[0], pool_k[1], pool_k[2]),
                    stride=(pool_k[0], pool_k[1], pool_k[2]),
                )
                freqs_tokens = rearrange(freqs_pooled[0], "d f h w -> (f h w) 1 1 d")
                freq_chunks.append(freqs_tokens)

                # Camera pooling mirrors freqs pooling
                if camera is not None:
                    cam_base_5d = camera  # [B, D_cam, f_base, h_base, w_base]
                    cam_t_idx = inds[i].to(device=cam_base_5d.device, dtype=torch.long)
                    if pool_k[0] > 1:
                        pad_t_cam = (-cam_t_idx.numel()) % pool_k[0]
                        if pad_t_cam:
                            cam_t_idx = torch.cat([cam_t_idx, cam_t_idx[-1:].repeat(pad_t_cam)], dim=0)
                    cam_sel = cam_base_5d[:, :, cam_t_idx, :, :]
                    cam_pad_h = (-h_base) % pool_k[1]
                    cam_pad_w = (-w_base) % pool_k[2]
                    if cam_pad_h or cam_pad_w:
                        cam_sel = torch.nn.functional.pad(
                            cam_sel,
                            (0, cam_pad_w, 0, cam_pad_h, 0, 0),
                            mode="replicate",
                        )
                    cam_pooled = torch.nn.functional.avg_pool3d(
                        cam_sel.float(),
                        kernel_size=(pool_k[0], pool_k[1], pool_k[2]),
                        stride=(pool_k[0], pool_k[1], pool_k[2]),
                    )
                    cam_tokens = rearrange(cam_pooled, "b d f h w -> b (f h w) d").type_as(x_tokens)
                    cam_chunks.append(cam_tokens)
        x_tokens = torch.cat(token_chunks, dim=1)
        freqs_tokens = torch.cat(freq_chunks, dim=0)
        camera_tokens = torch.cat(cam_chunks, dim=1) if camera is not None else None
        buffer_tokens = torch.cat(buf_chunks, dim=1) if use_buffer_tokens else None

        # When inject_kq_only is enabled, append a per-token validity indicator channel
        # (1.0 = real data, 0.0 = dummy) so the attention block can mask out
        # dummy positions after buffer encoding.
        if self.inject_kq_only and buffer_tokens is not None:
            validity_tokens = torch.cat(buf_validity_chunks, dim=1)
            buffer_tokens = torch.cat([buffer_tokens, validity_tokens], dim=-1)

        assert gen_start is not None and gen_end is not None
        return x_tokens, freqs_tokens, camera_tokens, buffer_tokens, (gen_start, gen_end), gen_grid

    # ---------------------------- Forward -----------------------------
    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        seq_len: int | None = None,
        frame_cond_crossattn_emb_B_L_D: torch.Tensor | None = None,
        y_B_C_T_H_W: torch.Tensor | None = None,
        y_buffer_B_C_T_H_W: torch.Tensor | None = None,
        padding_mask: Optional[torch.Tensor] = None,
        is_uncond: bool = False,
        slg_layers=None,
        **kwargs,
    ):
        # Choose path: base-like or Lyra2
        framepack_keys = {"framepack_indices", "framepack_splits", "framepack_kernel_ids"}
        use_framepack = framepack_keys.issubset(set(kwargs.keys()))

        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        if not use_framepack:
            # Base WanModel forward (with minor compatibility tweaks)
            if self.concat_padding_mask and padding_mask is not None:
                padding_mask = transforms.functional.resize(
                    padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
                )
                x_B_C_T_H_W = torch.cat(
                    [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
                )

            if self.conv_patchify:
                x_B_D_T_H_W = self.patch_embedding(x_B_C_T_H_W)
                x_B_T_H_W_D = rearrange(x_B_D_T_H_W, "b d t h w -> b t h w d")
            else:
                x_B_T_H_W_D = rearrange(
                    x_B_C_T_H_W,
                    "b c (t kt) (h kh) (w kw) -> b t h w (c kt kh kw)",
                    kt=self.patch_size[0],
                    kh=self.patch_size[1],
                    kw=self.patch_size[2],
                )
                x_B_T_H_W_D = self.patch_embedding(x_B_T_H_W_D)

            video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])
            x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)
            seq_len = seq_lens.max().item()
            assert seq_lens.max() == seq_len

            with amp.autocast("cuda", dtype=torch.float32):
                e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
                e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))

            if crossattn_emb.dim() == 4:
                crossattn_emb = crossattn_emb.squeeze(1)
            context_lens = None
            context_B_L_D = self.text_embedding(crossattn_emb)
            if frame_cond_crossattn_emb_B_L_D is not None:
                context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)
                context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

            kwargs_blocks = dict(
                e=e0_B_6_D,
                seq_lens=seq_lens,
                video_size=video_size,
                freqs=self.rope_position_embedding(x_B_T_H_W_D),
                context=context_B_L_D,
                context_lens=context_lens,
            )

            for block_idx, block in enumerate(self.blocks):
                if slg_layers is not None and block_idx in slg_layers and is_uncond:
                    continue
                x_B_L_D = block(x_B_L_D, **kwargs_blocks)

            x_B_L_D = self.head(x_B_L_D, e_B_D)
            t, h, w = video_size
            x_B_C_T_H_W = rearrange(
                x_B_L_D,
                "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
                nt=self.patch_size[0],
                nh=self.patch_size[1],
                nw=self.patch_size[2],
                t=t,
                h=h,
                w=w,
                d=self.out_dim,
            )
            return x_B_C_T_H_W

        # Lyra2 path
        # Optional camera extraction (Plücker condition path). Expect last 384 channels appended to x.
        camera_5d = None
        if self.use_plucker_condition:
            camera_ch = 384
            assert x_B_C_T_H_W.size(1) >= camera_ch, "Input missing appended camera channels (384)."
            camera = x_B_C_T_H_W[:, -camera_ch:]
            x_B_C_T_H_W = x_B_C_T_H_W[:, :-camera_ch]
            camera_5d = rearrange(
                camera,
                "b c t (h h2) (w w2) -> b (c h2 w2) t h w",
                h2=2, w2=2,
            )

        if self.concat_padding_mask and padding_mask is not None:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        framepack_indices = kwargs["framepack_indices"]
        framepack_splits = kwargs["framepack_splits"]
        framepack_kernel_ids = kwargs["framepack_kernel_ids"]
        framepack_kernel_types = kwargs.get("framepack_kernel_types", None)

        assert self.clean_patch_embeddings is not None

        x_tokens, freqs_tokens, camera_tokens, buffer_tokens, (gen_start, gen_end), (f_gen, h_gen, w_gen) = self._patchify_lyra2(
            x_B_C_T_H_W,
            framepack_indices,
            framepack_splits,
            framepack_kernel_ids,
            framepack_kernel_types,
            camera=camera_5d,
            buffer_B_C_T_H_W=y_buffer_B_C_T_H_W,
        )

        # Context Parallel after Lyra2 patchify: split tokens along L if enabled
        cp_enabled = getattr(self, "is_context_parallel_enabled", False)
        cp_group = getattr(self, "cp_group", None)
        if cp_enabled and cp_group is not None:
            L = x_tokens.size(1)
            cp_size = cp_group.size()
            assert L % cp_size == 0, f"Token length {L} must be divisible by cp_size {cp_size}"
            assert freqs_tokens.shape[0] % cp_size == 0, (
                f"Freq tokens length {freqs_tokens.shape[0]} must be divisible by cp_size {cp_size}"
            )
            x_tokens = split_inputs_cp(x_tokens, seq_dim=1, cp_group=cp_group)
            freqs_tokens = split_inputs_cp(freqs_tokens, seq_dim=0, cp_group=cp_group)
            if camera_tokens is not None:
                assert camera_tokens.size(1) % cp_size == 0, (
                    f"Camera tokens length {camera_tokens.size(1)} must be divisible by cp_size {cp_size}"
                )
                camera_tokens = split_inputs_cp(camera_tokens, seq_dim=1, cp_group=cp_group)
            if buffer_tokens is not None:
                assert buffer_tokens.size(1) % cp_size == 0, (
                    f"Buffer tokens length {buffer_tokens.size(1)} must be divisible by cp_size {cp_size}"
                )
                buffer_tokens = split_inputs_cp(buffer_tokens, seq_dim=1, cp_group=cp_group)

        with amp.autocast("cuda", dtype=torch.float32):
            e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
            e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))

        if crossattn_emb.dim() == 4:
            crossattn_emb = crossattn_emb.squeeze(1)
        context_B_L_D = self.text_embedding(crossattn_emb)
        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        assert x_tokens.shape[0] == 1
        seq_lens = torch.tensor([x_tokens.size(1)] * x_tokens.size(0), dtype=torch.long, device=x_tokens.device)
        if self.use_plucker_condition:
            kwargs_blocks = dict(
                e=e0_B_6_D,
                seq_lens=seq_lens,
                video_size=VideoSize(T=1, H=1, W=x_tokens.size(1)),
                freqs=freqs_tokens,
                context=context_B_L_D,
                context_lens=None,
                camera=camera_tokens,
                buffer=buffer_tokens,
            )
        else:
            kwargs_blocks = dict(
                e=e0_B_6_D,
                seq_lens=seq_lens,
                video_size=VideoSize(T=1, H=1, W=x_tokens.size(1)),
                freqs=freqs_tokens,
                context=context_B_L_D,
                context_lens=None,
                buffer=buffer_tokens,
            )

        x_B_L_D = x_tokens
        for block in self.blocks:
            x_B_L_D = block(x_B_L_D, **kwargs_blocks)

        x_B_L_D = self.head(x_B_L_D, e_B_D)

        if cp_enabled and cp_group is not None:
            x_B_L_D = cat_outputs_cp_with_grad(x_B_L_D, seq_dim=1, cp_group=cp_group)

        x_gen = x_B_L_D[:, gen_start:gen_end]
        x_B_C_T_H_W = rearrange(
            x_gen,
            "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=f_gen,
            h=h_gen,
            w=w_gen,
            pt=self.patch_size[0],
            ph=self.patch_size[1],
            pw=self.patch_size[2],
            c=self.out_dim,
        )
        return x_B_C_T_H_W

    def init_weights(self):
        # Match base WanModel.init_weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for block in self.blocks:
            block.init_weights()
        self.head.init_weights()

        if isinstance(self.patch_embedding, nn.Linear):
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
            if self.patch_embedding.bias is not None:
                nn.init.zeros_(self.patch_embedding.bias)
        if self.patch_embedding_buffer is not None:
            nn.init.xavier_uniform_(self.patch_embedding_buffer.weight.flatten(1))
            if self.patch_embedding_buffer.bias is not None:
                nn.init.zeros_(self.patch_embedding_buffer.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    def fully_shard(self, mesh, **fsdp_kwargs):
        for i, block in enumerate(self.blocks):
            fully_shard(block, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.head, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(self.text_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.time_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.patch_embedding, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.time_projection, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.clean_patch_embeddings is not None:
            for lin in self.clean_patch_embeddings:
                fully_shard(lin, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # For Lyra2, we split after patchify; disable CP inside rope embedder
        self.rope_position_embedding.disable_context_parallel()
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True
        self.cp_group = process_group

    def disable_context_parallel(self):
        # Lyra2 CP is applied post-patchify via token-splitting; simply drop the group flag
        self.cp_group = None
        self._is_context_parallel_enabled = False

    @property
    def is_context_parallel_enabled(self) -> bool:
        return self._is_context_parallel_enabled

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList):
        if sac_config.mode == CheckpointMode.NONE:
            return
        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(blocks)}"
        )
        _context_fn = sac_config.get_context_fn()
        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                blocks.register_module(block_id, block)
        self.register_module(
            "head",
            ptd_checkpoint_wrapper(
                self.head,
                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )
