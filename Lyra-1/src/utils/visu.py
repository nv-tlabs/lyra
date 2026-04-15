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
import os
from diffusers.utils import export_to_video
import torchvision
from torchvision.io import read_video
import numpy as np
import imageio
from einops import rearrange
import math
import cv2

def save_video(test_video_out, outdir, name='sample_grid', fps=8):
    test_video_out = reshape_video_grid(test_video_out)
    test_video_out = test_video_out.numpy()
    test_video_out = (test_video_out.transpose(0,2,3,1) * 255).astype(np.uint8)
    imageio.mimwrite(os.path.join(outdir, f'{name}.mp4'), test_video_out, fps=fps)

def wave_func(values, wave_pos, wave_length=1.0):
    """Cosine-squared falloff within wave band, zero outside."""
    dist = (values - wave_pos) / wave_length
    mask = np.abs(dist) <= 1.0
    wave = np.zeros_like(values, dtype=np.float32)
    wave[mask] = np.cos(dist[mask] * np.pi / 2.0) ** 2
    return wave

def generate_wave_video(image_tensor: torch.Tensor,
                        depth_tensor: torch.Tensor,
                        batch_idx: int = 0,
                        frame_idx: int = 0,
                        n_frames: int = 24,
                        wave_length: float = 1.0,
                        wave_color=(255, 255, 255),
                        wave_color_front = [255, 230, 200],
                        wave_color_back = [200, 220, 255],
                        use_gradient_color: bool = True,
                        pre_frames: int = 24) -> torch.Tensor:
    """
    Generates a wave propagation video and returns it as a torch.Tensor
    in shape [T, 3, H, W], range [0.0, 1.0].
    """
    assert image_tensor.ndim == 5 and image_tensor.shape[2] == 3
    assert depth_tensor.ndim == 5 and depth_tensor.shape[2] == 1

    image = image_tensor[batch_idx, frame_idx].detach().cpu().numpy()  # (3, H, W)
    depth = depth_tensor[batch_idx, frame_idx, 0].detach().cpu().numpy()  # (H, W)

    image = np.transpose(image, (1, 2, 0)).astype(np.float32) * 255.0  # (H, W, 3)
    depth = depth.astype(np.float32)

    assert image.shape[:2] == depth.shape

    min_depth, max_depth = depth.min(), depth.max()
    if max_depth - min_depth < 1e-5:
        max_depth = min_depth + 1.0

    if use_gradient_color:
        wave_color_front = np.array(wave_color_front, dtype=np.float32)   # Warm white
        wave_color_back  = np.array(wave_color_back, dtype=np.float32)   # Cool metallic blue
        depth_norm = (depth - min_depth) / (max_depth - min_depth)
        wave_color_map = (1. - depth_norm[..., None]) * wave_color_front + depth_norm[..., None] * wave_color_back
    else:
        wave_color_map = np.array(wave_color, dtype=np.float32).reshape(1, 1, 3)

    frames_np = []

    # Pre-video: hold initial frame
    initial_frame = np.clip(image, 0, 255).astype(np.uint8)
    frames_np.extend([initial_frame] * pre_frames)

    # Wave animation
    for i in range(n_frames + 1):
        ratio = i / n_frames
        curr_depth = (max_depth - min_depth) * ratio + min_depth

        wave = wave_func(depth, curr_depth, wave_length)[..., None]
        wave = np.clip(wave, 0.0, 1.0)

        frame = image * (1.0 - wave) + wave * wave_color_map
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames_np.append(frame)

    # Convert frames to torch.Tensor in [0,1], shape: (T, 3, H, W)
    frames_np = np.stack(frames_np, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    frames_np = np.transpose(frames_np, (0, 3, 1, 2))                  # (T, 3, H, W)
    frames_tensor = torch.from_numpy(frames_np)
    frames_tensor = frames_tensor[None]
    return frames_tensor

def create_depth_visu(x, cmap='jet', data_range=None, out_float=True, min_max_perc=[0.01, 0.99]): #min_max_perc=[0., 1.]):
    B, T, C, H, W = x.shape
    dtype = x.dtype
    device = x.device
    if data_range is None:
        x_flat = x.view(x.shape[0], -1)
        x_flat = x_flat.cpu().numpy()
        x_min = np.percentile(x_flat, min_max_perc[0]*100) #x_flat.amin(1).view(-1, 1, 1, 1, 1)
        x_max = np.percentile(x_flat, min_max_perc[1]*100) #x_flat.amax(1).view(-1, 1, 1, 1, 1)
        x = x.clip(x_min, x_max)
    else:
        x_min, x_max = data_range
    x = (x - x_min) / (x_max - x_min)
    x = rearrange(x, 'b t c h w -> (b t) h w c')
    x_np = x.cpu().numpy()
    x_np = (x_np * 255.0).astype(np.uint8)
    if cmap == "jet":
        color_map = cv2.COLORMAP_JET
    elif cmap == "inferno":
        color_map = cv2.COLORMAP_INFERNO
    x_np = [cv2.applyColorMap(x_np_i, color_map) for x_np_i in x_np]
    x = torch.from_numpy(np.array(x_np))
    x = rearrange(x, '(b t) h w c -> b t c h w', b=B)
    x = x.to(device=device, dtype=dtype)
    if out_float:
        x = x/255
    return x

def reshape_video_grid(video_tensor):
    b, t, c, h, w = video_tensor.shape
    N1 = N2 = int(math.sqrt(b))
    if N1 * N2 != b:
        N1 = 1
        N2 = b
    assert N1 * N2 == b, "Batch size must be a perfect square"

    # Rearrange using einops
    grid_video = rearrange(video_tensor, "(N1 N2) t c h w -> t c (N1 h) (N2 w)", N1=N1, N2=N2)

    return grid_video