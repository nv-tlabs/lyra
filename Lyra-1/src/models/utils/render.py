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

import torch
from typing import Tuple
import einops
from einops import rearrange
from plyfile import PlyData, PlyElement
import kiui
import kiui.op
import numpy as np

from src.models.utils.data import ray_condition
from src.models.utils.token_pruning import process_tensors

def get_plucker_embedding_and_rays(intrinsics_input: torch.Tensor, c2ws_input: torch.Tensor, img_size: Tuple[int, int], patch_size_out_factor: Tuple[int, int, int], flip_flag: torch.Tensor, get_batch_index: bool = True, dtype: torch.dtype = None, out_dtype: torch.dtype = None):
    dtype_orig = intrinsics_input.dtype
    if dtype is not None:
        intrinsics_input = intrinsics_input.to(dtype)
        c2ws_input = c2ws_input.to(dtype)
        flip_flag = flip_flag.to(dtype)
    else:
        dtype = dtype_orig
    if out_dtype is None:
        out_dtype = dtype_orig
    device = intrinsics_input.device
    plucker_embedding, rays_os, rays_ds = ray_condition(intrinsics_input, c2ws_input, img_size[0], img_size[1], device=device, flip_flag=flip_flag, get_batch_index=get_batch_index)
    if patch_size_out_factor[1] != 1 or patch_size_out_factor[2] != 1:
        # NOTE: Intrinsics here are assumed to be scaled already w.r.t image dimensions and not normalized
        intrinsics_resize_factors = torch.tensor(patch_size_out_factor[1:] * 2, dtype=dtype, device=device)
        intrinsics_resized = intrinsics_input/intrinsics_resize_factors
        img_size_patch_h = img_size[0]//patch_size_out_factor[1]
        img_size_patch_w = img_size[1]//patch_size_out_factor[2]
        _, rays_os, rays_ds = ray_condition(intrinsics_resized, c2ws_input, img_size_patch_h, img_size_patch_w, device=device, flip_flag=flip_flag, get_batch_index=get_batch_index)
    plucker_embedding = plucker_embedding.to(out_dtype)
    rays_os = rays_os.to(out_dtype)
    rays_ds = rays_ds.to(out_dtype)
    return plucker_embedding, rays_os, rays_ds

def downscale_intrinsics(intrinsics: torch.Tensor, factor: int = 2):
    for h_i, w_i in [(0, 0), (0, 2), (1, 1), (1, 2)]:
        intrinsics[:, :, h_i, w_i] /= 2
    return intrinsics

def subsample_pixels_spatio_temporal(dimensions: list, m_dims: list, device: torch.device):
    """
    Subsamples pixels from tensors with shape (B, T, H, W) by randomly selecting pixels
    based on temporal and spatial dimensions (T, H, W). Batch dimension (B) is NOT subsampled.

    Args:
        dimensions (list): A list of four integers [B, T, H, W] representing the dimensions of the tensor.
        m_dims (list): List of three integers [m_t, m_h, m_w] representing the number of samples for each dimension.
        device (torch.device): The device on which the tensor operations should occur.

    Returns:
        b_idx (torch.Tensor): (B, m_t * m_h * m_w) tensor of batch indices.
        t_idx (torch.Tensor): (B, m_t * m_h * m_w) tensor of time indices.
        h_idx (torch.Tensor): (B, m_t * m_h * m_w) tensor of height indices.
        w_idx (torch.Tensor): (B, m_t * m_h * m_w) tensor of width indices.
    """
    B, T, H, W = dimensions  # Unpack the dimensions from the input list
    m_t, m_h, m_w = m_dims  # Extract m_t, m_h, m_w from the list

    assert m_t <= T and m_h <= H and m_w <= W, "Requested samples exceed tensor dimensions."

    # Step 1: Sample t, h, w indices PER batch (B samples per dim)
    t_indices = torch.multinomial(torch.ones(T, device=device).expand(B, -1), m_t, replacement=False)  # (B, m_t)
    h_indices = torch.multinomial(torch.ones(H, device=device).expand(B, -1), m_h, replacement=False)  # (B, m_h)
    w_indices = torch.multinomial(torch.ones(W, device=device).expand(B, -1), m_w, replacement=False)  # (B, m_w)

    # Step 2: Cartesian product via broadcasting (tiny tensors only)
    t_grid = t_indices[:, :, None, None]  # (B, m_t, 1, 1)
    h_grid = h_indices[:, None, :, None]  # (B, 1, m_h, 1)
    w_grid = w_indices[:, None, None, :]  # (B, 1, 1, m_w)

    t_grid = t_grid.expand(-1, m_t, m_h, m_w)
    h_grid = h_grid.expand(-1, m_t, m_h, m_w)
    w_grid = w_grid.expand(-1, m_t, m_h, m_w)

    # Step 3: Make coordinates
    b_idx = torch.arange(B, device=device)[:, None].expand(B, m_t * m_h * m_w)  # (B, m_t * m_h * m_w)
    t_idx = t_grid.reshape(B, -1)  # (B, m_t * m_h * m_w)
    h_idx = h_grid.reshape(B, -1)  # (B, m_t * m_h * m_w)
    w_idx = w_grid.reshape(B, -1)  # (B, m_t * m_h * m_w)

    return b_idx, t_idx, h_idx, w_idx

def query_z_with_indices(indices, z):
    """
    Query tensor z at given (b, t, h, w) indices.
    
    Args:
        indices: list of 4 tensors [b_idx, t_idx, h_idx, w_idx], each of shape (B, N)
        z: tensor of shape (B, T, H, W, C)
        
    Returns:
        Tensor of shape (B, N, C)
    """
    b_idx, t_idx, h_idx, w_idx = indices  # each (B, N)
    B, T, H, W, C = z.shape
    N = t_idx.shape[1]

    # Step 1: Flatten z from (B, T, H, W, C) → (B, T*H*W, C)
    z_flat = rearrange(z, 'b t h w c -> b (t h w) c')  # (B, T*H*W, C)

    # Step 2: Compute flat index
    flat_idx = (t_idx * H * W) + (h_idx * W) + w_idx  # (B, N)

    # Step 3: Gather values using batch indexing
    # flat_idx: (B, N) → need to add batch dim for gather
    z_values = torch.gather(z_flat, dim=1, index=flat_idx.unsqueeze(-1).expand(-1, -1, C))  # (B, N, C)

    return z_values

def subsample_x_and_rays(x: torch.Tensor, rays_os: torch.Tensor, rays_ds: torch.Tensor, x_mask: torch.Tensor, sub_sample_gaussians_factor: list, sub_sample_gaussians_type: 'str', sub_sample_gaussians_type_tokens: str, temperature: float, training: bool):
    device = x.device
    # Compute subsample indices
    sub_sample_gaussians_factor = torch.tensor(sub_sample_gaussians_factor, device=device)
    x_shape = torch.tensor(x.shape[-3:], device=device)
    t_g_out, h_g_out, w_g_out = (x_shape/sub_sample_gaussians_factor).int().tolist()

    # Randomly mask pixels
    if sub_sample_gaussians_type == 'random':
        if not (sub_sample_gaussians_factor == 1).all():
            b_g_in, (t_g_in, h_g_in, w_g_in) = x.shape[0], x.shape[2:]
            bthw_g = subsample_pixels_spatio_temporal([b_g_in, t_g_in, h_g_in, w_g_in], [t_g_out, h_g_out, w_g_out], device)

            # Reshape tensors to query b, t, h, w
            x = rearrange(x, 'b c t h w -> b t h w c')
            rays_os = rearrange(rays_os, 'b t c h w -> b t h w c')
            rays_ds = rearrange(rays_ds, 'b t c h w -> b t h w c')

            # Query with subsampled indices
            x = query_z_with_indices(bthw_g, x)
            rays_os = query_z_with_indices(bthw_g, rays_os)
            rays_ds = query_z_with_indices(bthw_g, rays_ds)
        else:
            x = rearrange(x, 'b c t h w -> b (t h w) c')
            rays_os = rearrange(rays_os, 'b t c h w -> b (t h w) c')
            rays_ds = rearrange(rays_ds, 'b t c h w -> b (t h w) c')
        x_mask = None


    # Use learned mask to prune
    elif sub_sample_gaussians_type == 'learned':

        # Reshape to same format
        rays_os = rearrange(rays_os, 'b t c h w -> b c t h w')
        rays_ds = rearrange(rays_ds, 'b t c h w -> b c t h w')

        # Case 1: Structured pruning (per frame pruning and spatial per frame)
        if sub_sample_gaussians_type_tokens == 'local':
            x, (rays_os, rays_ds), x_mask = process_tensors(
                tokens=x,
                mask_logits=x_mask,
                other_tensors=[rays_os, rays_ds],
                k_t=t_g_out,                       # select t_g_out frames out of T
                k_hw=h_g_out * w_g_out,               # select 1/h_g_out * w_g_out spatial tokens
                temperature=temperature,
                training=training,  # differentiable Gumbel-Softmax
            )
        # Case 2: Global total pruning (select k tokens jointly across T and HW)
        elif sub_sample_gaussians_type_tokens == 'global':
            x, (rays_os, rays_ds), x_mask = process_tensors(
                tokens=x,
                mask_logits=x_mask,
                other_tensors=[rays_os, rays_ds],
                total_k=t_g_out * h_g_out * w_g_out,       # select k tokens globally (joint T and HW selection)
                temperature=temperature,
                training=training,  # inference: real top-k selection
            )

        # Reshape to channel last
        x = rearrange(x, 'b c n -> b n c')
        rays_os = rearrange(rays_os, 'b c n -> b n c')
        rays_ds = rearrange(rays_ds, 'b c n -> b n c')
    if training:
        x_mask = None
    return x, rays_os, rays_ds, x_mask

def save_ply(gaussians, path, scale_factor=None):
    # gaussians: [B, N, 14]
    assert gaussians.shape[0] == 1, 'only support batch size 1'
    # Scale positions and Gaussian sizes
    if scale_factor is not None:
        print(f"Scale factor {scale_factor} for gaussians")
        gaussians[0, :, 0:3] *= scale_factor
        gaussians[0, :, 4:7] *= scale_factor
    torch.save(gaussians, path)
    print(f"Saved gaussians to {path}")

def save_ply_orig(gaussians, path, compatible=True, scale_factor=None, prune_factor=0.005, prune=False):
    # gaussians: [B, N, 14]
    # compatible: save pre-activated gaussians as in the original paper

    assert gaussians.shape[0] == 1, 'only support batch size 1'

    from plyfile import PlyData, PlyElement
    
    means3D = gaussians[0, :, 0:3].contiguous().float()
    opacity = gaussians[0, :, 3:4].contiguous().float()
    scales = gaussians[0, :, 4:7].contiguous().float()
    rotations = gaussians[0, :, 7:11].contiguous().float()
    shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

    # Scale positions and Gaussian sizes
    if scale_factor is not None:
        print(f"Scale factor {scale_factor} for gaussians")
        means3D *= scale_factor
        scales *= scale_factor

    # prune by opacity
    if prune:
        mask = opacity.squeeze(-1) >= prune_factor
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = kiui.op.inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(path)
    print(f"Saved gaussians to {path}")