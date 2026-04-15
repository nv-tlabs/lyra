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

class GaussianRenderer:
    def __init__(self, opt):
        self.opt = opt
        self.gs_view_chunk_size = self.opt.get('gs_view_chunk_size', 1)
        
    def render(self, gaussians, cam_view, bg_color=None, intrinsics=None):
        # gaussians: [B, N, 14]
        # cam_pos: [B, V, 3]
        B, V = cam_view.shape[:2]

        # loop of loop...
        images, alphas, depths = [], [], []
        for b in range(B):
            for v in range(0, V, self.gs_view_chunk_size):
                # pos, opacity, scale, rotation, shs
                means3D = gaussians[b, :, 0:3].contiguous().float()
                opacity = gaussians[b, :, 3:4].contiguous().float()
                scales = gaussians[b, :, 4:7].contiguous().float()
                rotations = gaussians[b, :, 7:11].contiguous().float()
                rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

                # render novel views
                view_matrix = cam_view[b, v:v+self.gs_view_chunk_size].float()
                V_sub = view_matrix.shape[0]
                viewmat = view_matrix.transpose(2, 1) # [V, 4, 4]
                view_intrinsics = intrinsics[b, v: v+self.gs_view_chunk_size]
                Ks = [torch.tensor([[view_intrinsic[0],0.,view_intrinsic[2]],[0.,view_intrinsic[1],view_intrinsic[3]],[0., 0., 1.]],dtype=means3D.dtype, device=means3D.device) for view_intrinsic in view_intrinsics]
                rendered_image_all, rendered_alpha_all, info = rasterization(
                    means=means3D,
                    quats=rotations,
                    scales=scales,
                    opacities=opacity.squeeze(-1),
                    colors=rgbs,
                    viewmats=viewmat,
                    Ks=torch.stack(Ks),
                    width=self.opt.img_size[1],
                    height=self.opt.img_size[0],
                    near_plane=self.opt.znear,
                    far_plane=self.opt.zfar,
                    packed=False,
                    backgrounds=torch.stack([bg_color for _ in range(V_sub)]) if bg_color is not None else None,
                    render_mode="RGB+ED",
                )
                for rendered_image, rendered_alpha in zip(rendered_image_all, rendered_alpha_all):
                    depths.append(rendered_image[...,3:].permute(2, 0, 1))
                    rendered_image = rendered_image[...,:3].permute(2, 0, 1)
                    images.append(rendered_image)
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