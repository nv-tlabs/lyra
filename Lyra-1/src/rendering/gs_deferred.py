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

import numpy as np
import torch
from gsplat.rendering import rasterization
import kiui
import torch.nn.functional as F
import einops

from src.models.utils.render import downscale_intrinsics
from src.rendering.gs_deferred_patch import DeferredBPPatch

class DeferredBP(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_w2c, test_intr, 
               W, H, near_plane, far_plane, backgrounds, raster_kwargs):
        rgbd, alpha, _ = rasterization(
            means=xyz, 
            quats=rotation, 
            scales=scale, 
            opacities=opacity, 
            colors=feature,
            viewmats=test_w2c, 
            Ks=test_intr, 
            width=W, 
            height=H, 
            near_plane=near_plane, 
            far_plane=far_plane,
            backgrounds=backgrounds,
            render_mode="RGB+ED",
            **raster_kwargs,
        ) # (1, H, W, 3) 
        image, depth = rgbd[..., :3], rgbd[..., 3:]
        return image, alpha, depth     # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_w2cs, test_intr,
                W, H, near_plane, far_plane, backgrounds, raster_kwargs):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_w2cs, test_intr, backgrounds)
        ctx.W = W
        ctx.H = H
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        ctx.raster_kwargs = raster_kwargs
        with torch.no_grad():
            B, V = test_intr.shape[:2]
            images = torch.zeros(B, V, H, W, 3).to(xyz.device)
            alphas = torch.zeros(B, V, H, W, 1).to(xyz.device)
            depths = torch.zeros(B, V, H, W, 1).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    image, alpha, depth = DeferredBP.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                        test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
                        W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1],
                        raster_kwargs
                    )
                    images[ib, iv:iv+1] = image
                    alphas[ib, iv:iv+1] = alpha
                    depths[ib, iv:iv+1] = depth
        images = images.requires_grad_()
        alphas = alphas.requires_grad_()
        depths = depths.requires_grad_()
        return images, alphas, depths

    @staticmethod
    def backward(ctx, images_grad, alphas_grad, depths_grad):
        xyz, feature, scale, rotation, opacity, test_w2cs, test_intr, backgrounds = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        raster_kwargs = ctx.raster_kwargs
        with torch.enable_grad():
            B, V = test_intr.shape[:2]
            for ib in range(B):
                for iv in range(V):
                    image, alpha, depth = DeferredBP.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                        test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
                        W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1],
                        raster_kwargs,
                    )
                    render_split = torch.cat([image, alpha, depth], dim=-1)
                    grad_split = torch.cat([images_grad[ib, iv:iv+1], alphas_grad[ib, iv:iv+1], depths_grad[ib, iv:iv+1]], dim=-1) 
                    render_split.backward(grad_split)

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None, None

class GaussianRendererDeferred:
    def __init__(self, opt):
        self.opt = opt
        if self.opt.deferred_bp:
            self.render_func = self.render_deferred
        else:
            self.render_func = self.render_standard
        self.oom_downscale_factors = [1, 2, 4, 8]
        self.use_3dgut = self.opt.get('use_3dgut', False)
        if self.use_3dgut:
            self.raster_kwargs = {'with_ut': True, 'with_eval3d': True, 'packed': False}
        else:
            # Packed = False does not work currently with background in new gsplat
            self.raster_kwargs = {'with_ut': False, 'with_eval3d': False, 'packed': False}

    def render(self, gaussians, cam_view, bg_color=None, intrinsics=None, patch_size=None):
        B, V = cam_view.shape[:2]
        # pos, opacity, scale, rotation, shs
        means3D = gaussians[..., 0:3].contiguous().float()
        opacity = gaussians[..., 3:4].contiguous().float().squeeze(-1)
        scales = gaussians[..., 4:7].contiguous().float()
        rotations = gaussians[..., 7:11].contiguous().float()
        rgbs = gaussians[..., 11:].contiguous().float() # [N, 3]

        viewmat = cam_view.float().transpose(3, 2)  # [B, V, 4, 4]
        Ks = torch.tensor([[[[view_intrinsic[0],0.,view_intrinsic[2]],[0.,view_intrinsic[1],view_intrinsic[3]],[0., 0., 1.]] for view_intrinsic in batch_intrinsic] for batch_intrinsic in intrinsics], dtype=means3D.dtype, device=means3D.device)
        backgrounds = torch.tensor([[bg_color for _ in range(V)] for _ in range(B)], dtype=means3D.dtype, device=means3D.device) if bg_color is not None else torch.ones(B, V, 3, dtype=means3D.dtype, device=means3D.device)

        H, W = self.opt.img_size
        near_plane, far_plane = self.opt.znear, self.opt.zfar
        # Downscale images until no OOM error (sometimes the GS rendering runs OOM for many intersections)
        for factor_idx, downscale_factor in enumerate(self.oom_downscale_factors):
            out_dict = self.render_func(means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane, patch_size)
            # try:
            #     if downscale_factor == 1:
            #         out_dict = self.render_func(means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane, patch_size)
            #     else:
            #         out_dict = self.render_downscale(
            #             means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane, patch_size,
            #             B, downscale_factor
            #         )
            #     # If successful, break out of loop
            #     break
            # except Exception:
            #     if factor_idx == len(self.oom_downscale_factors) - 1:
            #         # Re-raise the last exception if all factors failed
            #         raise e
            #     else:
            #         # Try the next downscale_factor
            #         continue
        return out_dict
    
    def render_downscale(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane, patch_size, B, downscale_factor):
        print(f"Cuda Error for rendering on {means3D.device}! Switch to {downscale_factor}x low res")
        Ks_resized = downscale_intrinsics(Ks.clone(), factor=downscale_factor)
        H_resized, W_resized = H //downscale_factor, W //downscale_factor
        out_dict = self.render_func(means3D, opacity, scales, rotations, rgbs, viewmat, Ks_resized, backgrounds, H_resized, W_resized, near_plane, far_plane, patch_size)
        for k in ["images_pred", "alphas_pred", "depths_pred"]:
            out_dict[k] = einops.rearrange(out_dict[k], 'b v c h w -> (b v) c h w')
            out_dict[k] = F.interpolate(out_dict[k], size=(H, W), mode='nearest')
            out_dict[k] = einops.rearrange(out_dict[k], '(b v) c h w -> b v c h w', b=B)
        return out_dict

    def render_deferred(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane, patch_size=None):
        # If patch_size is None, use regular rendering (DeferredBP)
        if patch_size is None:
            images, alphas, depths = DeferredBP.apply(
                means3D, rgbs, scales, rotations, opacity, 
                viewmat, Ks, W, H, near_plane, far_plane,
                backgrounds, self.raster_kwargs,
            )
            return {
                "images_pred": images.permute(0, 1, 4, 2, 3),  # [B, V, 3, H, W]
                "alphas_pred": alphas.permute(0, 1, 4, 2, 3),  # [B, V, 1, H, W]
                "depths_pred": depths.permute(0, 1, 4, 2, 3),  # [B, V, 1, H, W]
            }
        else:
            # Patch-based rendering (DeferredBPPatch)
            images, alphas, depths = DeferredBPPatch.apply(
                means3D, rgbs, scales, rotations, opacity, 
                viewmat, Ks, W, H, near_plane, far_plane, 
                backgrounds, patch_size, self.raster_kwargs,
            )
            return {
                "images_pred": images,  # [B, V, 3, H, W] already in correct shape
                "alphas_pred": alphas,  # [B, V, 1, H, W] already in correct shape
                "depths_pred": depths,  # [B, V, 1, H, W] now returned by patch version
            }   
                
    def render_standard(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane, patch_size=None):
        # gaussians: [B, N, 14]
        # cam_pos: [B, V, 3]
        B, V = Ks.shape[:2]

        # loop of loop...
        images, alphas, depths = [], [], []
        for b in range(B):
            rendered_image_all, rendered_alpha_all, _ = rasterization(
                means=means3D[b],
                quats=rotations[b],
                scales=scales[b],
                opacities=opacity[b],
                colors=rgbs[b],
                viewmats=viewmat[b],
                Ks=Ks[b],
                width=W,
                height=H,
                near_plane=near_plane,
                far_plane=far_plane,
                backgrounds=backgrounds[b],
                render_mode="RGB+ED",
                **self.raster_kwargs,
            )
            for rendered_image, rendered_alpha in zip(rendered_image_all, rendered_alpha_all):
                depths.append(rendered_image[...,3:].permute(2, 0, 1))
                images.append(rendered_image[...,:3].permute(2, 0, 1))
                alphas.append(rendered_alpha.permute(2, 0, 1))
                
        images, alphas, depths = torch.stack(images), torch.stack(alphas), torch.stack(depths)
        images, alphas, depths = images.view(B, V, *images.shape[1:]), alphas.view(B, V, *alphas.shape[1:]), depths.view(B, V, *depths.shape[1:])

        return {
            "images_pred": images, # [B, V, 3, H, W]
            "alphas_pred": alphas, # [B, V, 1, H, W]
            "depths_pred": depths, # [B, V, 1, H, W]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
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
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
