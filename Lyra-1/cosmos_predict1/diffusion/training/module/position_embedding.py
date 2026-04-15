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

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks

from cosmos_predict1.diffusion.module.attention import normalize
from cosmos_predict1.diffusion.module.timm import trunc_normal_
from cosmos_predict1.diffusion.training.context_parallel import split_inputs_cp


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size_h,
    grid_size_w,
    grid_size_t,
    spatial_interpolation_scale,
    temporal_interpolation_scale,
    concat=True,
):
    grid_h = np.arange(grid_size_h, dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(grid_size_w, dtype=np.float32) / spatial_interpolation_scale
    grid_t = np.arange(grid_size_t, dtype=np.float32) / temporal_interpolation_scale

    grid = np.meshgrid(grid_w, grid_h, grid_t, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(3, 1, grid_size_h, grid_size_w, grid_size_t)

    if concat:
        per_axis = embed_dim // 3
        per_axis = (per_axis // 2) * 2  # make it even (for sin/cos split)
        dim_h, dim_w = per_axis, per_axis
        dim_t = embed_dim - dim_h - dim_w
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[0])  # (H*W, D/3)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[1])  # (H*W, D/3)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid[2])  # (H*W, D/3)

        return np.concatenate([emb_h, emb_w, emb_t], axis=1)  # (H*W*T, D)
    else:
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[0])  # (H*W)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[1])  # (H*W)
        emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[2])  # (H*W)

        return emb_h + emb_w + emb_t  # (H*W*T, D)


class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp_group = None

    def enable_context_parallel(self, cp_group: ProcessGroup):
        self.cp_group = cp_group

    def disable_context_parallel(self):
        self.cp_group = None

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split. It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self.cp_group is not None:
            cp_ranks = get_process_group_ranks(self.cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        if self.cp_group is not None:
            if isinstance(self, VideoRopePosition3DEmb):
                seq_dim = 0
            else:
                seq_dim = 1
            embeddings = split_inputs_cp(x=embeddings, seq_dim=seq_dim, cp_group=self.cp_group)
        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError


class SinCosPosEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        is_learnable: bool = False,
        interpolation: Literal["crop", "resize", "crop_resize"] = "crop",
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
        init_length_for_resize: int = 16,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): "crop", "resize", "crop_resize". "crop" means we crop the positional embedding to the length of the input sequence. "resize" means we resize the positional embedding to the length of the input sequence. "crop_resize" (inference only) means we first crop the positional embedding to init_length_for_resize, then resize it to the length of the input sequence.
            init_length_for_resize (int): used when interpolation is "crop_resize", where we "resize" embedding during inference for model trained with "crop". We first "crop" the pos_embed to this length (used during training), then run the "resize", default 16
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        self.init_length_for_resize = init_length_for_resize
        param = get_3d_sincos_pos_embed(
            model_channels, len_h, len_w, len_t, spatial_interpolation_scale, temporal_interpolation_scale
        )
        param = rearrange(param, "(h w t) c -> 1 t h w c", h=len_h, w=len_w)
        if is_learnable:
            self.pos_embed = nn.Parameter(
                torch.from_numpy(param).float(),
            )
        else:
            self.register_buffer("pos_embed", torch.from_numpy(param).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C
        if self.interpolation == "crop":
            return self.pos_embed[:, :T, :H, :W]
        if self.interpolation == "resize":
            return rearrange(
                F.interpolate(
                    rearrange(self.pos_embed, "1 t h w c -> 1 c h w t"),
                    size=(H, W, T),
                    mode="linear",
                    align_corners=False,
                ),
                "1 c h w t -> 1 t h w c",
            )
        if self.interpolation == "crop_resize":
            pos_embed_crop = self.pos_embed[:, : self.init_length_for_resize, :H, :W]  # B,T,H,W,C
            _, t, h, w, c = pos_embed_crop.shape

            pos_embed_crop_resize_t = rearrange(
                F.interpolate(
                    rearrange(pos_embed_crop, "1 t h w c -> 1 (c h w) t"),
                    size=(T),
                    mode="linear",
                ),
                "1 (c h w) t -> 1 t h w c",
                c=c,
                h=h,
                w=w,
            )
            pos_embed_crop_resize = rearrange(
                F.interpolate(
                    rearrange(pos_embed_crop_resize_t, "1 t h w c -> 1 (c t) h w"),
                    size=(H, W),
                    mode="bilinear",
                ),
                "1 (c t) h w -> 1 t h w c",
                c=c,
            )
            return pos_embed_crop_resize

        raise ValueError(f"Unknown interpolation method {self.interpolation}")


class SinCosPosEmb_FPS_Aware(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        min_fps: int,  # 1 for getty video
        max_fps: int,  # 120 for getty video
        is_learnable: bool = False,
        interpolation: str = "crop",
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        self.max_fps = max_fps
        self.min_fps = min_fps
        if self.interpolation == "crop":
            param = get_3d_sincos_pos_embed(
                model_channels,
                len_h,
                len_w,
                len_t * int(max_fps / min_fps),
                spatial_interpolation_scale,
                temporal_interpolation_scale,
            )  # should be max_seq_length * (max_fps / min_fps)
        elif self.interpolation == "resize":
            param = get_3d_sincos_pos_embed(
                model_channels, len_h, len_w, len_t, spatial_interpolation_scale, temporal_interpolation_scale
            )  # time embedding based min fps
        else:
            ValueError(f"Unknown interpolation method {self.interpolation}")
        param = rearrange(param, "(h w t) c -> 1 t h w c", h=len_h, w=len_w)
        if is_learnable:
            self.pos_embed = nn.Parameter(
                torch.from_numpy(param).float(),
            )
        else:
            self.register_buffer("pos_embed", torch.from_numpy(param).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C

        if self.interpolation == "crop":
            if T > 1:
                return torch.cat(
                    [
                        self.pos_embed[:, : (int(self.max_fps / curr_fps) * T) : int(self.max_fps / curr_fps), :H, :W]
                        for curr_fps in fps
                    ],
                    0,
                )
            else:
                return self.pos_embed[:, :T, :H, :W]  # image model
        elif self.interpolation == "resize":
            if T > 1:
                return torch.cat(
                    [
                        rearrange(
                            F.interpolate(
                                rearrange(self.pos_embed, "1 t h w c -> 1 c h w t"),
                                size=(H, W, T * int(curr_fps / self.min_fps)),
                                mode="trilinear",
                                align_corners=True,  # important: align corner need to be true
                            )[:, :, :H, :W, :T],
                            "1 c h w t -> 1 t h w c",
                        )
                        for curr_fps in fps
                    ],
                    0,
                )
            else:
                # grab self.pos_embed at time step 0 and resize spatially
                return rearrange(
                    F.interpolate(
                        rearrange(self.pos_embed[:, 0, ::], "1 h w c -> 1 c h w"),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    "1 c h w -> 1 h w c",
                )
        raise ValueError(f"Unknown interpolation method {self.interpolation}")


class LearnableEmb3D(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        interpolation: str = "crop",
        is_learnable: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs  # unused
        super().__init__()
        assert is_learnable is True
        self.interpolation = interpolation
        self.pos_embed = nn.Parameter(torch.zeros(1, len_t, len_h, len_w, model_channels))
        trunc_normal_(self.pos_embed, std=0.02)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C
        if self.interpolation == "crop":
            return self.pos_embed[:, :T, :H, :W]
        if self.interpolation == "resize":
            return rearrange(
                F.interpolate(
                    rearrange(self.pos_embed, "1 t h w c -> 1 c h w t"),
                    size=(H, W, T),
                    mode="linear",
                    align_corners=False,
                ),
                "1 c h w t -> 1 t h w c",
            )
        raise ValueError(f"Unknown interpolation method {self.interpolation}")


class LearnableEmb3D_FPS_Aware(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        min_fps: int,  # 1 for getty video
        max_fps: int,  # 120 for getty video
        interpolation: str = "crop",
        is_learnable: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        assert is_learnable is True
        self.interpolation = interpolation
        self.max_fps = max_fps
        self.min_fps = min_fps

        if self.interpolation == "crop":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, len_t * int(max_fps / min_fps), len_h, len_w, model_channels)
            )  # should be max_seq_length * (max_fps / min_fps)
        elif self.interpolation == "resize":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, len_t, len_h, len_w, model_channels)
            )  # time embedding based min fps
        else:
            ValueError(f"Unknown interpolation method {self.interpolation}")

        trunc_normal_(self.pos_embed, std=0.02)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C

        if self.interpolation == "crop":
            if T > 1:
                return torch.cat(
                    [
                        self.pos_embed[:, : (int(self.max_fps / curr_fps) * T) : int(self.max_fps / curr_fps), :H, :W]
                        for curr_fps in fps
                    ],
                    0,
                )
            else:
                return self.pos_embed[:, :T, :H, :W]  # image model
        elif self.interpolation == "resize":
            if T > 1:
                return torch.cat(
                    [
                        rearrange(
                            F.interpolate(
                                rearrange(self.pos_embed, "1 t h w c -> 1 c h w t"),
                                size=(H, W, T * int(curr_fps / self.min_fps)),
                                mode="trilinear",
                                align_corners=True,  # important: align corner need to be true
                            )[:, :, :H, :W, :T],
                            "1 c h w t -> 1 t h w c",
                        )
                        for curr_fps in fps
                    ],
                    0,
                )
            else:
                # grab self.pos_embed at time step 0 and resize spatially
                return rearrange(
                    F.interpolate(
                        rearrange(self.pos_embed[:, 0, ::], "1 h w c -> 1 c h w"),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    "1 c h w -> 1 h w c",
                )
        raise ValueError(f"Unknown interpolation method {self.interpolation}")


class VideoRopePositionEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(len_h * len_w * len_t, dtype=torch.float))

        self.register_buffer(
            "dim_range", torch.arange(0, head_dim, 2)[: (head_dim // 2)].float().cuda() / head_dim, persistent=False
        )

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor], ntk_factor: float = 1.0):
        theta = 10000.0 * ntk_factor

        # original_dtype = self.dim_range.dtype
        freq = 1.0 / (theta ** self.dim_range.float())
        _, T, H, W, _ = B_T_H_W_C
        length = T * H * W
        emb_L_D = torch.outer(self.seq[:length], freq)
        return rearrange(torch.cat([emb_L_D, emb_L_D], dim=-1), "l d -> l 1 1 d").float()


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        self._dim_h = dim_h
        self._dim_t = dim_t

    def reset_parameters(self) -> None:
        if self.dim_spatial_range.device == torch.device("meta"):
            return

        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)

        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor. Defaults to None.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor. Defaults to None.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor. Defaults to None.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        uniform_fps = (fps is None) or (fps.min() == fps.max())
        assert (
            uniform_fps or B == 1 or T == 1
        ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w}) configured for positional embedding. Please adjust the input size or increase the maximum dimensions in the model configuration."
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        # apply sequence scaling in temporal dimension
        if fps is None:  # image case
            assert T == 1, "T should be 1 for image batch."
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class SinCosPosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        dim = model_channels
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        # rescale pos id is equivalent to rescale frequency
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos=np.arange(len_h) * 1.0 / h_extrapolation_ratio)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos=np.arange(len_w) * 1.0 / w_extrapolation_ratio)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos=np.arange(len_t) * 1.0 / t_extrapolation_ratio)

        self.register_buffer("pos_emb_h", torch.from_numpy(emb_h).float(), persistent=False)
        self.register_buffer("pos_emb_w", torch.from_numpy(emb_w).float(), persistent=False)
        self.register_buffer("pos_emb_t", torch.from_numpy(emb_t).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = torch.cat(
                [
                    repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W),
                    repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W),
                    repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H),
                ],
                dim=-1,
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
            return emb

        raise ValueError(f"Unknown interpolation method {self.interpolation}")


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        trunc_normal_(self.pos_emb_h, std=0.02)
        trunc_normal_(self.pos_emb_w, std=0.02)
        trunc_normal_(self.pos_emb_t, std=0.02)

    def reset_parameters(self):
        if self.pos_emb_h.device == torch.device("meta"):
            return

        trunc_normal_(self.pos_emb_h, std=0.02)
        trunc_normal_(self.pos_emb_w, std=0.02)
        trunc_normal_(self.pos_emb_t, std=0.02)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        return normalize(emb, dim=-1, eps=1e-6)


class MultiviewVideoPositionEmb(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.cp_group = None

    def enable_context_parallel(self, cp_group: ProcessGroup):
        self.cp_group = cp_group

    def disable_context_parallel(self):
        self.cp_group = None

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split. It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self.cp_group is not None:
            cp_ranks = get_process_group_ranks(self.cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        if self.cp_group is not None:
            if isinstance(self, MultiviewVideoRopePosition3DEmb):
                seq_dim = 1
                embeddings = rearrange(embeddings, "(V T) H W D -> V (T H W) 1 1 D", V=self.n_views).float()
                # rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()
                embeddings = split_inputs_cp(x=embeddings, seq_dim=seq_dim, cp_group=self.cp_group)
                embeddings = rearrange(embeddings, "V T 1 1 D -> (V T) 1 1 D", V=self.n_views).float()
            else:
                seq_dim = 1
                embeddings = rearrange(embeddings, "B (V T) H W C -> (B V) T H W C", V=self.n_views)
                embeddings = split_inputs_cp(x=embeddings, seq_dim=seq_dim, cp_group=self.cp_group)
                embeddings = rearrange(embeddings, "(B V) T H W C -> B (V T) H W C", V=self.n_views)
        else:
            if isinstance(self, MultiviewVideoRopePosition3DEmb):
                embeddings = rearrange(embeddings, "t h w d -> (t h w) 1 1 d").float()

        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError


class MultiviewVideoRopePosition3DEmb(MultiviewVideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        n_views: int = 4,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.n_views = n_views
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().cuda() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().cuda() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def generate_embedding_for_batch(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor. Defaults to None.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor. Defaults to None.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor. Defaults to None.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        uniform_fps = (fps is None) or (fps.min() == fps.max())
        assert uniform_fps  # only support uniform fps now

        assert (
            uniform_fps or B == 1 or T == 1
        ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w}) configured for positional embedding. Please adjust the input size or increase the maximum dimensions in the model configuration."
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        # apply sequence scaling in temporal dimension
        if fps is None:  # image case
            assert T == 1, "T should be 1 for image batch."
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return em_T_H_W_D

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size. The camera view dimension is merged in the T dimension

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time * Views, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor. Defaults to None.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor. Defaults to None.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor. Defaults to None.

        Returns:
            Not specified in the original code snippet.
        """

        B, T, H, W, C = B_T_H_W_C

        single_view_B_T_H_W_C = (B, T // self.n_views, H, W, C)
        em_T_H_W_D = torch.cat(
            [
                self.generate_embedding_for_batch(
                    single_view_B_T_H_W_C,
                    fps=fps,
                    h_ntk_factor=h_ntk_factor,
                    w_ntk_factor=w_ntk_factor,
                    t_ntk_factor=t_ntk_factor,
                )
                for item in range(self.n_views)
            ],
            dim=0,
        )

        return em_T_H_W_D
        # return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class MultiviewSinCosPosEmbAxis(MultiviewVideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        n_views: int = 4,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        self.n_views = n_views
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"

        dim = model_channels
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        # rescale pos id is equivalent to rescale frequency
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos=np.arange(len_h) * 1.0 / h_extrapolation_ratio)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos=np.arange(len_w) * 1.0 / w_extrapolation_ratio)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos=np.arange(len_t) * 1.0 / t_extrapolation_ratio)

        self.register_buffer("pos_emb_h", torch.from_numpy(emb_h).float(), persistent=False)
        self.register_buffer("pos_emb_w", torch.from_numpy(emb_w).float(), persistent=False)
        self.register_buffer("pos_emb_t", torch.from_numpy(emb_t).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C

        single_view_T = T // self.n_views

        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:single_view_T]
            emb = torch.cat(
                [
                    torch.cat(
                        [
                            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W),
                            repeat(emb_h_H, "h d-> b t h w d", b=B, t=single_view_T, w=W),
                            repeat(emb_w_W, "w d-> b t h w d", b=B, t=single_view_T, h=H),
                        ],
                        dim=-1,
                    )
                    for _ in range(self.n_views)
                ],
                1,
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
            return emb

        raise ValueError(f"Unknown interpolation method {self.interpolation}")
