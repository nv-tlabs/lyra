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

from functools import cache
import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from torchvision.io import read_video
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Union, Optional
import os

def compute_std(val_sum, val_avg, count):
    variance = (val_sum / count) - (val_avg ** 2)
    std = torch.sqrt(torch.clamp(variance, min=0))
    return std

def plot_average_metric_per_frame(
    psnr_avg: Union[torch.Tensor, list],
    out_path: str = "average_psnr_per_frame.png",
    psnr_std: Optional[Union[torch.Tensor, list]] = None,
    metric_name: str = "PSNR",
) -> None:
    """
    Plots and saves a line plot of average PSNR values per frame, with optional std error bars.

    Args:
        psnr_avg (Tensor or list): 1D tensor or list of average PSNR values (length = num_frames).
        out_path (str): Path to save the output PNG file.
        psnr_std (Tensor or list, optional): 1D tensor or list of standard deviation values for each frame.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if isinstance(psnr_avg, list):
        psnr_avg = torch.tensor(psnr_avg)
    if psnr_std is not None and isinstance(psnr_std, list):
        psnr_std = torch.tensor(psnr_std)

    num_frames = len(psnr_avg)
    x = list(range(num_frames))

    plt.figure(figsize=(12, 4))

    if psnr_std is not None:
        plt.errorbar(
            x, psnr_avg.tolist(), yerr=psnr_std.tolist(),
            fmt='-o', color='steelblue', ecolor='lightgray', elinewidth=1, capsize=3, linewidth=2
        )
    else:
        plt.plot(x, psnr_avg.tolist(), color='steelblue', linewidth=2)

    plt.xlabel('Frame Index')
    plt.ylabel(f'Average {metric_name}')
    plt.title('Average {metric_name} per Frame across Dataset')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlim(0, num_frames - 1)
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def read_mp4_to_tensor(path: str, device: torch.device) -> torch.Tensor:
    """
    Reads an MP4 video file using torchvision and returns a tensor of shape (B, C, H, W),
    with values in [0, 1] as torch.float32.
    
    Args:
        path (str): Path to the MP4 file.
        
    Returns:
        torch.Tensor: Tensor of shape (B, C, H, W) with dtype float32 and values in [0, 1].
    """
    video, _, _ = read_video(path, pts_unit='sec')  # shape: (B, H, W, C)
    video = video.permute(0, 3, 1, 2)  # (B, C, H, W)
    video = video.float() / 255.0     # Normalize to [0, 1]
    return video.to(device)

def resize_and_crop_video(
    video: torch.Tensor,
    target_height: int,
    target_width: int,
    direct_crop: bool = False
) -> torch.Tensor:
    """
    Resize and center-crop a video tensor to the target resolution.

    Args:
        video (Tensor): Input tensor of shape (B, C, H, W), values in [0, 1].
        target_height (int): Desired output height.
        target_width (int): Desired output width.
        direct_crop (bool): If True, skip resizing and only crop.

    Returns:
        Tensor: Resized and cropped tensor of shape (B, C, target_height, target_width).
    """
    B, C, H, W = video.shape

    if not direct_crop:
        # Determine scale factor to preserve aspect ratio
        scale_h = target_height / H
        scale_w = target_width / W

        # Scale based on the smaller factor (resize one side to target)
        scale = max(scale_h, scale_w)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        # Resize using bilinear interpolation
        video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # Crop center
    _, _, H_new, W_new = video.shape
    top = (H_new - target_height) // 2
    left = (W_new - target_width) // 2

    cropped = video[:, :, top:top + target_height, left:left + target_width]
    return cropped

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()

@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    sub_batch_size: int = 32,
) -> Float[Tensor, " batch"]:
    lpips_model = get_lpips(predicted.device)
    B = ground_truth.shape[0]
    scores = []

    for i in range(0, B, sub_batch_size):
        gt_chunk = ground_truth[i : i + sub_batch_size]
        pred_chunk = predicted[i : i + sub_batch_size]
        value = lpips_model(gt_chunk, pred_chunk, normalize=True)
        scores.append(value[:, 0, 0, 0])

    return torch.cat(scores, dim=0)

@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)