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

import os
import torch
from tqdm import tqdm
import time

from src.eval.metrics import get_lpips, compute_lpips, compute_ssim, compute_psnr, resize_and_crop_video, read_mp4_to_tensor, plot_average_metric_per_frame, compute_std
from src.models.btimer.core.utils.data import write_dict_to_json, read_json_to_dict

def compute_metrics(path_data_pred: str, path_data_gt: str, out_path: str = None, H_target: int = None, W_target: int = None, num_scenes: int = None, num_frames_eval: int = None):
    device = torch.device("cuda:0")
    lpips_module = get_lpips(device)
    file_names = sorted(os.listdir(path_data_pred))
    if num_scenes is not None:
        file_names = file_names[:num_scenes]

    psnr_sum, ssim_sum, lpips_sum = None, None, None
    count = 0
    out_path_metrics_main = os.path.join(out_path, 'metrics')
    os.makedirs(out_path_metrics_main, exist_ok=True)
    time.sleep(30)
    for file_name in tqdm(file_names):
        path_video_pred = os.path.join(path_data_pred, file_name)
        path_video_gt = os.path.join(path_data_gt, file_name)
        path_metrics_out = os.path.join(out_path_metrics_main, file_name.replace('.mp4', '.json'))
        # Initialize accumulation tensors
        if psnr_sum is None:
            video_pred = read_mp4_to_tensor(path_video_pred, device)
            T = video_pred.shape[0]
            psnr_sum = torch.zeros(T, device=device)
            ssim_sum = torch.zeros(T, device=device)
            lpips_sum = torch.zeros(T, device=device)
        if not os.path.isfile(path_metrics_out):
            # Read videos
            video_pred = read_mp4_to_tensor(path_video_pred, device)
            video_gt = read_mp4_to_tensor(path_video_gt, device)
            T, C, H, W = video_pred.shape
            # Cut gt to the same frames
            video_gt = video_gt[:T]
            # Resize and crop videos to target res
            video_gt = resize_and_crop_video(video_gt, H, W)
            # Additional resize and crop
            if H_target is not None and W_target is not None:
                video_pred = resize_and_crop_video(video_pred, H_target, W_target)
                video_gt = resize_and_crop_video(video_gt, H_target, W_target)
            # Optionally shorten
            T_gt = video_gt.shape[0]
            if T_gt != T:
                pad_mask = torch.zeros(T, device=device, dtype=bool)
                pad_mask[T_gt:] = True
                video_pred = video_pred[:T_gt]
            # Compute metrics
            psnr = compute_psnr(video_gt, video_pred)
            ssim = compute_ssim(video_gt, video_pred)
            lpips = compute_lpips(video_gt, video_pred)

            # Optionally pad
            if T_gt != T:
                psnr_full = torch.zeros(T, device=device)
                ssim_full = torch.zeros(T, device=device)
                lpips_full = torch.zeros(T, device=device)
                psnr_full[:T_gt] = psnr
                ssim_full[:T_gt] = ssim
                lpips_full[:T_gt] = lpips
                psnr = psnr_full
                ssim = ssim_full
                lpips = lpips_full

            metrics_dict_out = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
            metrics_dict_out = {k: v.tolist() for k, v in metrics_dict_out.items()}
            write_dict_to_json(metrics_dict_out, path_metrics_out)
        else:
            metrics_dict_out = read_json_to_dict(path_metrics_out)
            metrics_dict_out = {k: torch.tensor(v, device=device) for k, v in metrics_dict_out.items()}
            psnr = metrics_dict_out['psnr']
            ssim = metrics_dict_out['ssim']
            lpips = metrics_dict_out['lpips']
        psnr_sum += psnr
        ssim_sum += ssim
        lpips_sum += lpips
        count += 1
    # Compute average PSNR per frame
    psnr_avg = psnr_sum / count
    ssim_avg = ssim_sum / count
    lpips_avg = lpips_sum / count
    psnr_std = compute_std(psnr_sum, psnr_avg, count)
    ssim_std = compute_std(ssim_sum, ssim_avg, count)
    lpips_std = compute_std(lpips_sum, lpips_avg, count)

    # Save histogram
    print("psnr_avg", psnr_avg, "ssim_avg", ssim_avg, "lpips_avg", lpips_avg)
    if num_frames_eval is not None:
        print(f"psnr_avg for first {num_frames_eval} frames: {psnr_avg[:num_frames_eval].mean()}")
        print(f"ssim_avg for first {num_frames_eval} frames: {ssim_avg[:num_frames_eval].mean()}")
        print(f"lpips_avg for first {num_frames_eval} frames: {lpips_avg[:num_frames_eval].mean()}")
    plot_average_metric_per_frame(psnr_avg, os.path.join(out_path, "psnr.png"), psnr_std, metric_name="PSNR")
    plot_average_metric_per_frame(ssim_avg, os.path.join(out_path, "ssim.png"), ssim_std, metric_name="SSIM")
    plot_average_metric_per_frame(lpips_avg, os.path.join(out_path, "lpips.png"), lpips_std, metric_name="LPIPS")
    

if __name__ == '__main__':
    path_data_pred = '/path/to/rgb'
    path_data_gt = '/path/to/gt_rgb/'
    out_path = 'outputs/eval'
    
    H_target, W_target = None, None
    num_scenes = 1000
    num_frames_eval = 121
    out_path = os.path.join(out_path, f"img_res_{H_target}_{W_target}", f"num_scenes_{num_scenes}")
    compute_metrics(path_data_pred, path_data_gt, out_path, H_target, W_target, num_scenes, num_frames_eval)