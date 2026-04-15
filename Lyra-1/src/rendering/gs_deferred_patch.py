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
from torch.utils.checkpoint import _get_autocast_kwargs
from gsplat.rendering import rasterization

class DeferredBPPatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, features, scaling, rotation, opacity, C2W, Ks, width, height, near_plane, far_plane, backgrounds, patch_size, raster_kwargs):
        """
        Forward rendering with the addition of near_plane and far_plane.
        """
        assert (xyz.dim() == 3) and (
            features.dim() == 3
        ) and (scaling.dim() == 3) and (rotation.dim() == 3), f"xyz: {xyz.shape}, features: {features.shape}, scaling: {scaling.shape}, rotation: {rotation.shape}, opacity: {opacity.shape}"
        assert height % patch_size[0] == 0 and width % patch_size[1] == 0, f'patch_size must be divisible by H ({height} / {patch_size[0]}) and W ({width} / {patch_size[1]})!'

        ctx.save_for_backward(xyz, features, scaling, rotation, opacity)  # save tensors for backward
        ctx.height = height
        ctx.width = width
        ctx.C2W = C2W
        ctx.Ks = Ks
        ctx.patch_size = patch_size
        ctx.backgrounds = backgrounds
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        ctx.raster_kwargs = raster_kwargs

        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.manual_seeds = []

        with torch.no_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            device = C2W.device
            b, v = C2W.shape[:2]
            colors = torch.zeros(b, v, 3, height, width, device=device)
            alphas = torch.zeros(b, v, 1, height, width, device=device)
            depths = torch.zeros(b, v, 1, height, width, device=device)  # We will store depth here

            for i in range(b):
                ctx.manual_seeds.append([])

                for j in range(v):
                    Ks_ij = Ks[i, j]
                    fx, fy, cx, cy = Ks_ij[0, 0], Ks_ij[1, 1], Ks_ij[0, 2], Ks_ij[1, 2]
                    for m in range(0, ctx.width // ctx.patch_size[1]):
                        for n in range(0, ctx.height // ctx.patch_size[0]):
                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)

                            new_fx = fx
                            new_fy = fy
                            new_cx = cx - m * ctx.patch_size[1]
                            new_cy = cy - n * ctx.patch_size[0]
                            
                            new_K = torch.tensor([[new_fx, 0., new_cx], [0., new_fy, new_cy], [0., 0., 1.]], dtype=torch.float32, device=device)

                            rgbd, alpha, _ = rasterization(
                                means=xyz[i],
                                quats=rotation[i],
                                scales=scaling[i],
                                opacities=opacity[i].squeeze(-1),
                                colors=features[i],
                                viewmats=C2W[i, j][None],
                                Ks=new_K[None],
                                width=ctx.patch_size[1],
                                height=ctx.patch_size[0],
                                near_plane=ctx.near_plane,  # Use near_plane here
                                far_plane=ctx.far_plane,    # Use far_plane here
                                backgrounds=ctx.backgrounds[i, j][None],
                                render_mode="RGB+ED",  # RGB + Depth (last channel)
                                **raster_kwargs,
                            )

                            # Permute and clamp the rendered image and alpha
                            rendered_image = rgbd[0, :, :, :3].permute(2, 0, 1).clamp(0, 1)  # (1, 3, H, W)
                            rendered_alpha = alpha[0].permute(2, 0, 1).clamp(0, 1)  # (1, 1, H, W)
                            rendered_depth = rgbd[0, :, :, 3:].permute(2, 0, 1)  # Depth is the last channel

                            # Store the results in the final output tensors
                            colors[i, j, :, n * ctx.patch_size[0]:(n + 1) * ctx.patch_size[0], m * ctx.patch_size[1]:(m + 1) * ctx.patch_size[1]] = rendered_image
                            alphas[i, j, :, n * ctx.patch_size[0]:(n + 1) * ctx.patch_size[0], m * ctx.patch_size[1]:(m + 1) * ctx.patch_size[1]] = rendered_alpha
                            depths[i, j, :, n * ctx.patch_size[0]:(n + 1) * ctx.patch_size[0], m * ctx.patch_size[1]:(m + 1) * ctx.patch_size[1]] = rendered_depth

        return colors, alphas, depths

    @staticmethod
    def backward(ctx, grad_colors, grad_alphas, grad_depths):
        """
        Backward process.
        """
        xyz, features, scaling, rotation, opacity = ctx.saved_tensors
        raster_kwargs = ctx.raster_kwargs

        xyz_nosync = xyz.detach().clone()
        xyz_nosync.requires_grad = True
        xyz_nosync.grad = None

        features_nosync = features.detach().clone()
        features_nosync.requires_grad = True
        features_nosync.grad = None

        scaling_nosync = scaling.detach().clone()
        scaling_nosync.requires_grad = True
        scaling_nosync.grad = None

        rotation_nosync = rotation.detach().clone()
        rotation_nosync.requires_grad = True
        rotation_nosync.grad = None

        opacity_nosync = opacity.detach().clone()
        opacity_nosync.requires_grad = True
        opacity_nosync.grad = None

        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            device = ctx.C2W.device
            dtype = ctx.C2W.dtype
            b, v = ctx.C2W.shape[:2]

            for i in range(b):
                ctx.manual_seeds.append([])

                for j in range(v):
                    Ks_ij = ctx.Ks[i, j]
                    fx, fy, cx, cy = Ks_ij[0, 0], Ks_ij[1, 1], Ks_ij[0, 2], Ks_ij[1, 2]
                    for m in range(0, ctx.width // ctx.patch_size[1]):
                        for n in range(0, ctx.height // ctx.patch_size[0]):
                            grad_colors_split = grad_colors[i, j, :, n * ctx.patch_size[0]:(n + 1) * ctx.patch_size[0], m * ctx.patch_size[1]:(m + 1) * ctx.patch_size[1]]
                            grad_alphas_split = grad_alphas[i, j, :, n * ctx.patch_size[0]:(n + 1) * ctx.patch_size[0], m * ctx.patch_size[1]:(m + 1) * ctx.patch_size[1]]
                            grad_depths_split = grad_depths[i, j, :, n * ctx.patch_size[0]:(n + 1) * ctx.patch_size[0], m * ctx.patch_size[1]:(m + 1) * ctx.patch_size[1]]

                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)

                            new_fx = fx
                            new_fy = fy
                            new_cx = cx - m * ctx.patch_size[1]
                            new_cy = cy - n * ctx.patch_size[0]

                            new_K = torch.tensor([[new_fx, 0., new_cx], [0., new_fy, new_cy], [0., 0., 1.]], dtype=dtype, device=device)

                            rgbd, alpha, _ = rasterization(
                                means=xyz_nosync[i],
                                quats=rotation_nosync[i],
                                scales=scaling_nosync[i],
                                opacities=opacity_nosync[i].squeeze(-1),
                                colors=features_nosync[i],
                                viewmats=ctx.C2W[i, j][None],
                                Ks=new_K[None],
                                width=ctx.patch_size[1],
                                height=ctx.patch_size[0],
                                near_plane=ctx.near_plane,
                                far_plane=ctx.far_plane,
                                backgrounds=ctx.backgrounds[i, j][None],
                                render_mode="RGB+ED",
                                **raster_kwargs,
                            )

                            # Permute and clamp the rendered image and alpha
                            rendered_image = rgbd[0, :, :, :3].permute(2, 0, 1)
                            rendered_image = rendered_image.clamp(0, 1)
                            rendered_alpha = alpha[0].permute(2, 0, 1) #.clamp(0, 1)
                            rendered_depth = rgbd[0, :, :, 3:].permute(2, 0, 1)


                            # Concatenate rendered output and gradients
                            render_split = torch.cat([rendered_image, rendered_alpha, rendered_depth], dim=0)  # (1, H, W, 5)
                            grad_split = torch.cat([grad_colors_split, grad_alphas_split, grad_depths_split], dim=0)  # Same shape as render_split
                            render_split.backward(grad_split)

        # Return the gradients for the inputs that were used in forward pass
        return xyz_nosync.grad, features_nosync.grad, scaling_nosync.grad, rotation_nosync.grad, opacity_nosync.grad, None, None, None, None, None, None, None, None, None
