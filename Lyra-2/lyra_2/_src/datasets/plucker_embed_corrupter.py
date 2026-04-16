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
from packaging import version as pver


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def ray_condition(K, c2w, H, W, device, flip_flag=None, use_ray_o=False):
    # c2w: B, V, 4, 4
    # K: B, V, 4
    # If K is None, use constant forward z-direction in camera space
    B, V = c2w.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    if K is None:
        directions = torch.zeros(B, V, H * W, 3, device=device, dtype=c2w.dtype)
        directions[..., 2] = 1.0
    else:
        fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1
        zs = torch.ones_like(i)  # [B, V, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)
        directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    if use_ray_o:
        plucker = torch.cat([rays_o, rays_d], dim=-1)
    else:
        rays_dxo = torch.cross(rays_o, rays_d)  # B, V, HW, 3
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker
