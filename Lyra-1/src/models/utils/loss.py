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
import torch.nn.functional as F
import einops

def resize_images_lpips(images, img_size, img_size_min):
    image_out_size = (min(img_size[0], img_size_min), min(img_size[1], int(img_size_min/img_size[0]*img_size[1])))
    return F.interpolate(
        images.view(-1, 3, img_size[0], img_size[1]) * 2 - 1, image_out_size,
        mode='bilinear',
        align_corners=False
        )

def normalize_depth(depth: torch.Tensor, valid_mask: torch.Tensor):
    depth = einops.rearrange(depth, 'b t 1 h w -> b t (h w)')
    valid_mask = einops.rearrange(valid_mask, 'b t 1 h w -> b t (h w)').float()

    # Count valid pixels (avoid zero count)
    valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)

    # Mask invalid pixels
    depth_valid = depth * valid_mask

    # Median over valid pixels
    depth_median = torch.median(depth_valid, dim=-1, keepdim=True)[0]

    # Centered depth (subtract median), mask invalids
    depth_centered = (depth_valid - depth_median) * valid_mask

    # Mean absolute deviation (only valid pixels)
    depth_var = depth_centered.abs().sum(dim=-1, keepdim=True) / valid_count

    # Clamp variance (avoid zero or inf)
    depth_var = torch.clamp(depth_var, min=1e-3, max=1e3)

    # Normalize centered depth
    depth_normalized = depth_centered / depth_var

    return depth_normalized

def compute_depth_loss(pred_depths: torch.Tensor, gt_depths: torch.Tensor):
    # Valid mask computed once: depth > 0, finite, not nan
    valid_mask = (gt_depths > 0) & torch.isfinite(gt_depths)

    # Normalize using the same valid mask
    pred_depths_norm = normalize_depth(pred_depths, valid_mask)
    gt_depths_norm = normalize_depth(gt_depths, valid_mask)

    # Flatten valid_mask to (b, t, h*w) to match normal[ized tensors
    valid_mask_float = einops.rearrange(valid_mask.float(), 'b t 1 h w -> b t (h w)')

    # Apply mask before loss
    loss_depth = F.smooth_l1_loss(
        pred_depths_norm * valid_mask_float,
        gt_depths_norm * valid_mask_float
    )
    return loss_depth

def compute_lpips_loss_in_chunks(lpips_loss_module, gt_images, pred_images, lpips_img_size, lpips_img_size_min, chunk_size=64):
    """
    Computes LPIPS loss with chunking along the V dimension and uses gradient checkpointing.

    Args:
        lpips_loss_module: A callable LPIPS loss module.
        gt_images (Tensor): Ground truth images of shape (B, V, C, H, W).
        pred_images (Tensor): Predicted images of shape (B, V, C, H, W).
        lpips_img_size (int): Target image size for LPIPS.
        lpips_img_size_min (int): Minimum image size for LPIPS.
        chunk_size (int): Number of V elements to process at once. Default is 64.

    Returns:
        Tensor: Scalar LPIPS loss averaged over all (B * V) image pairs.
    """
    B, V, C, H, W = gt_images.shape
    total_loss = []
    num_chunks = (V + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, V)

        gt_chunk = gt_images[:, start:end].reshape(-1, C, H, W)
        pred_chunk = pred_images[:, start:end].reshape(-1, C, H, W)

        gt_chunk = resize_images_lpips(gt_chunk, lpips_img_size, lpips_img_size_min)
        pred_chunk = resize_images_lpips(pred_chunk, lpips_img_size, lpips_img_size_min)

        loss_chunk = torch.utils.checkpoint.checkpoint(
            lpips_loss_module,
            gt_chunk,
            pred_chunk,
            use_reentrant=False
        )
        total_loss.append(loss_chunk)
    total_loss = torch.cat(total_loss, 0)
    total_loss = total_loss.mean((2, 3))
    return total_loss
    
def compute_loss(accelerator, train_loss, pred_images, gt_images, pred_depths, gt_depths, pred_opacity, config, lpips_loss_module=None, lpips_img_size=None):
    # MSE loss
    loss = F.mse_loss(pred_images, gt_images)
    # LPIPS loss
    if config.get('lambda_lpips', 0) > 0:
        if config.lpips_chunk_size is not None:
            loss_lpips = compute_lpips_loss_in_chunks(lpips_loss_module, gt_images, pred_images, lpips_img_size, config.lpips_img_size_min, config.lpips_chunk_size)
        else:
            loss_lpips = lpips_loss_module(
                resize_images_lpips(gt_images, lpips_img_size, lpips_img_size_min),
                resize_images_lpips(pred_images, lpips_img_size, lpips_img_size_min),
            )
        loss_lpips = loss_lpips.mean()
        loss = loss + config.lambda_lpips * loss_lpips
    # SSIM Loss
    if config.get('lambda_ssim', 0) > 0:
        ssim_img_size = config.img_size
        loss_ssim = fused_ssim(
            pred_images.view(-1, 3, ssim_img_size[0], ssim_img_size[1]).float(),
            gt_images.view(-1, 3, ssim_img_size[0], ssim_img_size[1]).float()
            )
        loss_ssim = (1 - loss_ssim) / 2
        loss = loss + config.lambda_ssim * loss_ssim

    # Depth loss
    if config.get('lambda_depth', 0) > 0:
        loss_depth = compute_depth_loss(pred_depths, gt_depths)
        loss = loss + config.lambda_depth * loss_depth
    # Opacity loss
    if config.get('lambda_opacity', 0) > 0:
        loss_opacity = pred_opacity.to(pred_images.dtype).sigmoid().mean()
        loss = loss + config.lambda_opacity * loss_opacity

    # Average loss
    loss = loss.mean()

    # Gather the losses across all processes for logging (if we use distributed training).
    avg_loss = accelerator.gather(loss.repeat(config.batch_size)).mean()
    train_loss += avg_loss.item() / config.gradient_accumulation_steps
    return train_loss, loss