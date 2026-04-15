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
import torch.nn.functional as F
import torchvision.transforms as transforms
from packaging import version as pver
import random
import json
from typing import Union, Callable, Tuple
import OpenEXR
import Imath

class ImageTransform:
    def __init__(self, crop_size, sample_size, max_crop, use_flip=False, ):
        self.use_flip = use_flip
        self.crop_size = crop_size
        self.max_crop = max_crop
        self.sample_size = sample_size
        self.crop_transform = transforms.CenterCrop(crop_size) if crop_size else lambda x: x
        self.resize_transform = (
            transforms.Resize(sample_size) if sample_size else lambda x: x
        )
        self.resize_transform_depth = (
            transforms.Resize(sample_size, interpolation=transforms.InterpolationMode.NEAREST) if sample_size else lambda x: x
        )

    def preprocess_images(self, images, depths=None):
        # Returns the preprocessed images along with an image transform object
        # which describes the transformation on the image
        video = images

        if self.use_flip:
            assert False
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(
                images.shape[0], dtype=torch.bool, device=video.device
            )

        ori_h, ori_w = video.shape[-2:]
        if self.max_crop:
            # scale up to largest croppable size
            crop_ratio = min(ori_h/self.crop_size[0], ori_w/self.crop_size[1])
            new_crop_size = (int(self.crop_size[0]*crop_ratio), int(self.crop_size[1]*crop_ratio))
            self.crop_transform = transforms.CenterCrop(new_crop_size)
            

        video = self.crop_transform(video)
        if depths is not None:
            depths = self.crop_transform(depths)
        # print('after crop',video.shape)
        new_h, new_w = video.shape[-2:]
        # NOTE! I'm using u,v convention here instead of h,w
        shift = ((new_w - ori_w) / 2, (new_h - ori_h) / 2)

        # resize:
        ori_h, ori_w = video.shape[-2:]
        # new_h, new_w = self.sample_size
        video = self.resize_transform(video)
        if depths is not None:
            depths = self.resize_transform_depth(depths)
        new_h, new_w = video.shape[-2:]
        scale = (new_w/ori_w, new_h/ori_h)

        if self.use_flip:
            video = self.flip_transform(video, flip_flag)
            if depths is not None:
                depths = self.flip_transform(depths)
        # print('shift, scale',shift, scale)
        # return video, shift, scale, flip_flag
        return video, depths, shift, scale, flip_flag

    def apply_img_transform(self, i, j, shift, scale):
        # takes pixel uv coordinates in un-transformed space and converts to new
        # coordinates of image after crop and resize

        # first shift, then scale
        i = (i + shift[0]) * scale[0]
        j = (j + shift[1]) * scale[1]
        return i, j



def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_grid_uvs(batch_shape, H, W, device, dtype=None, flip_flag=None, nh=None, nw=None, margin=0):
    if dtype is None: dtype = torch.float32
    if nh is None: nh = H
    if nw is None: nw = W
    # c2w: B, V, 4, 4
    # K: B, V, 4
    # c2w @ dirctions
    B, V = batch_shape 

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
        torch.linspace(0, W - 1, nw, device=device, dtype=dtype),
    )
    i = i.reshape([1, 1, nh * nw]).expand([B, V, nh * nw]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, nh * nw]).expand([B, V, nh * nw]) + 0.5          # [B, V, HxW]

    if margin != 0:
        marginw = 1-2*margin
        i = marginw * i + margin * W
        j = marginw * j + margin * H

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
            torch.linspace(W - 1, 0, nw, device=device, dtype=dtype)
        )
        i_flip = i_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        j_flip = j_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip
    return i,j


def get_rays_from_uvs(i,j,K,c2w):
    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    # printarr(directions, c2w)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    return rays_o, rays_d

def project_to_uvs(pts,K,c2w):
    w2c = torch.linalg.inv(c2w)
    cam_pts = torch.einsum("...ij,...vj->...vi", w2c[...,:3,:3], pts) + w2c[..., None, :3, 3]

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    xs = cam_pts[...,0]
    ys = cam_pts[...,1]
    zs = cam_pts[...,2]


    us = (fx * xs/zs) + cx
    vs = (fy * ys/zs) + cy
    uvs = torch.stack([us,vs], dim=-1)
    return uvs, zs

def get_rays(K, c2w, H, W, device, flip_flag=None, nh=None, nw=None):
    i,j = get_grid_uvs(K.shape[:2], H=H, W=W, dtype=K.dtype, device=device, flip_flag=flip_flag, nh=nh, nw=nw)
    return get_rays_from_uvs(i,j,K,c2w)


def ray_condition(K, c2w, H, W, device, flip_flag=None, get_batch_index=True):
    batch_shape = K.shape[:2]
    B, V = batch_shape
    rays_o, rays_d = get_rays(K, c2w, H, W, device, flip_flag=flip_flag)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6).permute(0, 1, 4, 2, 3).contiguous()
    rays_o = rays_o.reshape(B, c2w.shape[1], H, W, 3).permute(0, 1, 4, 2, 3).contiguous()
    rays_d = rays_d.reshape(B, c2w.shape[1], H, W, 3).permute(0, 1, 4, 2, 3).contiguous()
    if get_batch_index:
        plucker = plucker[0]
        rays_o = rays_o[0]
        rays_d = rays_d[0]
    return plucker, rays_o, rays_d

def mirror_frame_indices(sampling_num_frames: int, total_num_frames: int, video_mirror_clip_length: int = None, stride: int = 1, start_index: int = None, return_target: bool = False):
    if video_mirror_clip_length is None:
        video_mirror_clip_length = total_num_frames

    if total_num_frames > video_mirror_clip_length:
        idx = random.randint(0, total_num_frames-video_mirror_clip_length)
        mapping = list(range(idx, idx + video_mirror_clip_length))
        total_num_frames = video_mirror_clip_length
    else:
        mapping = list(range(total_num_frames))
    n_repeat = max((sampling_num_frames * stride - total_num_frames) // (total_num_frames - 1), 0) + 1
    mapping_repeat = mapping.copy()
    for i in range(n_repeat):
        if i % 2 == 0:
            mapping_repeat += mapping[-2::-1]
        else:
            mapping_repeat += mapping[1:]
    if start_index is None:
        start_index = random.randint(0, len(mapping_repeat) - sampling_num_frames * stride)
    sample_idx = list(range(start_index, start_index + sampling_num_frames * stride, stride))
    sample_idx = [mapping_repeat[idx] for idx in sample_idx]
    return sample_idx

def weighted_sample(arr: np.ndarray, num_samples: int, 
                    bias: Union[str, Callable[[np.ndarray], np.ndarray]] = 'uniform') -> np.ndarray:
    """
    Sample elements from a numpy array with optional weighting toward the end.

    Parameters:
    - arr: np.ndarray - Input array to sample from.
    - num_samples: int - Number of samples to draw.
    - bias: str or callable - Weighting strategy: 'uniform', 'linear', 'squared', 'exponential',
                              or a custom function that takes an array of [0, 1] positions.

    Returns:
    - np.ndarray - Sampled array of values.
    - np.ndarray - Sampled array of probabilities.
    """
    n = len(arr)

    if bias == 'uniform':
        probabilities = None  # uniform by default in np.random.choice
    else:
        # Relative position in array: from 0 (start) to 1 (end)
        positions = np.linspace(0, 1, n)

        if bias == 'linear':
            weights = positions
        elif bias == 'squared':
            weights = np.square(positions)
        elif bias == 'exponential':
            weights = np.exp(3 * positions)  # the factor controls steepness; adjust as needed
        elif callable(bias):
            weights = bias(positions)
        else:
            raise ValueError("Invalid bias type. Use 'uniform', 'linear', 'squared', 'exponential', or a custom callable.")

        probabilities = weights / weights.sum()

    sampled_inds = np.random.choice(np.arange(len(arr)), size=num_samples, replace=False, p=probabilities)
    sampled_vals = arr[sampled_inds]
    sampled_probabilities = probabilities[sampled_inds] if probabilities is not None else probabilities
    return sampled_vals, sampled_probabilities

def read_exr_depth_to_numpy(exr_file) -> np.ndarray:
    header = exr_file.header()
    dw = header["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1

    # Dynamically detect pixel type
    chan_info = header['channels']['Z']
    pix_type = chan_info.type.v

    # Map OpenEXR pixel types to NumPy dtypes
    if pix_type == Imath.PixelType(Imath.PixelType.HALF).v:
        dtype = np.float16
        bytes_per_pixel = 2
    elif pix_type == Imath.PixelType(Imath.PixelType.FLOAT).v:
        dtype = np.float32
        bytes_per_pixel = 4
    elif pix_type == Imath.PixelType(Imath.PixelType.DOUBLE).v:
        dtype = np.float64
        bytes_per_pixel = 8
    else:
        raise ValueError(f"Unknown EXR pixel type: {pix_type}")

    # Read and reshape
    raw = exr_file.channel("Z")
    depth_map = np.frombuffer(raw, dtype=dtype).reshape(h, w)
    return depth_map

def merge_input_target_data_dicts(data_fields_input, data_fields_target, original_output_dict_input, original_output_dict_target):
    original_output_dict = {}
    data_fields = list(set(data_fields_input + data_fields_target))
    for data_field in data_fields:
        if data_field in original_output_dict_input and data_field in original_output_dict_target:
            out_data_field = torch.cat((original_output_dict_input[data_field], original_output_dict_target[data_field]))
        else:
            if data_field in original_output_dict_input:
                out_data_field = original_output_dict_input[data_field]
            elif data_field in original_output_dict_target:
                out_data_field = original_output_dict_target[data_field]
        original_output_dict[data_field] = out_data_field
    return original_output_dict

def write_dict_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def read_json_to_dict(filename):
    with open(filename, "r") as f:
        json_dict = json.load(f)
    return json_dict