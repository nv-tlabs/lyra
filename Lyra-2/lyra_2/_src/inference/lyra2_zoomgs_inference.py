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
"""
Single-image → zoom-in + zoom-out video generation.

Pipeline:
1. Read a single input image.
2. Load its text caption from a pre-generated .txt file (see scripts/gemini_caption.py).
3. Produce a zoom-in and a zoom-out video using Lyra2 AR spatial generation.
4. Save individual + combined videos.

GS reconstruction is handled separately by vipe_da3_gs_recon.py.
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.visualize.video import save_img_or_video
from lyra_2._src.inference.lyra2_ar_inference import (
    save_output,
    safe_to,
    run_lyra2_sample,
)
from lyra_2._src.inference.camera_traj_utils import (
    build_camera_trajectory,
    CAMERA_TRAJECTORY_CHOICES,
)
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

torch.enable_grad(False)
torch.backends.cudnn.enabled = False

# ---------------------------------------------------------------------------
# DA3 single-image depth (reused from lyra2_ar_inference_from_image)
# ---------------------------------------------------------------------------

def _da3_infer_depth_intrinsics_single(
    da3_model,
    img_rgb_uint8: torch.Tensor,
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DA3 single-image depth: RGB uint8 HWC -> (image_chw01, depth_hw, K_33, mask_hw)."""
    Ht, Wt = target_hw
    img_np = img_rgb_uint8.detach().cpu().numpy()
    img_resized = cv2.resize(img_np, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    image_chw01 = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    image_chw01 = image_chw01.permute(2, 0, 1).unsqueeze(0).contiguous()

    images = [img_resized.astype(np.uint8)]
    prediction = da3_model.inference(
        image=images,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        infer_gs=False,
        process_res=int(max(Ht, Wt)),
        process_res_method="upper_bound_resize",
        export_dir=None,
        export_format="mini_npz",
    )

    depths_np = getattr(prediction, "depth", None)
    if depths_np is None:
        raise RuntimeError("DA3 prediction has no 'depth' field.")
    if isinstance(depths_np, torch.Tensor):
        depth_np = depths_np[0].detach().cpu().numpy()
    else:
        depth_np = np.asarray(depths_np)[0]
    Hd, Wd = depth_np.shape[-2:]

    depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if (Hd, Wd) != (Ht, Wt):
        depth_t = F.interpolate(depth_t, size=(Ht, Wt), mode="bilinear", align_corners=False)
    depth_hw = depth_t[0, 0]
    depth_hw = torch.nan_to_num(depth_hw, nan=1e4).clamp(min=0, max=1e4)
    mask_hw = (depth_hw < 999.9).to(dtype=torch.float32)

    try:
        ixts_np = getattr(prediction, "intrinsics", None)
        if ixts_np is None:
            raise AttributeError
        if isinstance(ixts_np, torch.Tensor):
            K_np = ixts_np[0].detach().cpu().numpy()
        else:
            K_np = np.asarray(ixts_np)[0]
        K_33 = torch.from_numpy(K_np.astype(np.float32))
        scale_x = float(Wt) / float(Wd)
        scale_y = float(Ht) / float(Hd)
        K_33 = K_33.clone()
        K_33[0, 0] *= scale_x
        K_33[1, 1] *= scale_y
        K_33[0, 2] *= scale_x
        K_33[1, 2] *= scale_y
    except Exception:
        fx = fy = max(Ht, Wt) * 1.5
        cx, cy = Wt / 2.0, Ht / 2.0
        K_33 = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)

    return image_chw01, depth_hw, K_33, mask_hw


def _camera_centers_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image zoom-in/zoom-out video generation"
    )
    # Input
    parser.add_argument("--input_image_path", type=str, required=True,
                        help="Path to a single image or a folder of images")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_start_idx", type=int, default=0)
    parser.add_argument("--sample_id", type=int, default=None,
                        help="Run only the sample at this index (0-based). "
                             "Overrides --num_samples and --sample_start_idx.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional explicit prompt applied to ALL images.")
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="Directory containing per-image .txt caption files. "
                             "Each file should be named <image_stem>.txt. "
                             "When set, Gemini captioning is skipped entirely.")
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="Text appended to every prompt.")

    # Model and generation
    parser.add_argument("--experiment", type=str, default="lyra_framepack_spatial")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model")
    parser.add_argument("--output_path", type=str, default="inference/lyra2_zoomgs")
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--num_sampling_step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=161,
                        help="Default frames per direction. Overridden by --num_frames_zoom_in / --num_frames_zoom_out.")
    parser.add_argument("--num_frames_zoom_in", type=int, default=81,
                        help="Frames for zoom-in. Falls back to --num_frames if not set.")
    parser.add_argument("--num_frames_zoom_out", type=int, default=241,
                        help="Frames for zoom-out. Falls back to --num_frames if not set.")
    parser.add_argument("--resolution", type=str, default="480,832", help="H,W")
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--lora_paths", type=str, nargs="+",
                        default=["checkpoints/lora/realism_boost.safetensors",
                                 "checkpoints/lora/detail_enhancer.safetensors"])
    parser.add_argument("--lora_weights", type=float, nargs="+", default=[0.4, 0.4])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload_when_prompt", action="store_true")

    # Camera trajectory for zoom
    parser.add_argument("--zoom_in_trajectory", type=str, default="horizontal_zoom",
                        choices=list(CAMERA_TRAJECTORY_CHOICES),
                        help="Camera trajectory for zoom-in video.")
    parser.add_argument("--zoom_out_trajectory", type=str, default="horizontal_zoom",
                        choices=list(CAMERA_TRAJECTORY_CHOICES),
                        help="Camera trajectory for zoom-out video.")
    parser.add_argument("--zoom_in_direction", type=str, default="right",
                        choices=["left", "right", "up", "down"],
                        help="Direction for zoom-in (right = forward along z).")
    parser.add_argument("--zoom_out_direction", type=str, default="left",
                        choices=["left", "right", "up", "down"],
                        help="Direction for zoom-out (left = backward along z).")
    parser.add_argument("--zoom_in_strength", type=float, default=0.5)
    parser.add_argument("--zoom_out_strength", type=float, default=1.5)

    # Depth backend
    parser.add_argument("--use_moge_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Align DA3 depth to MoGe scale during seeding (default: True).")
    parser.add_argument("--ground_plane_align", action="store_true",
                        help="Fit a ground plane from depth and move camera parallel to it.")
    parser.add_argument("--ground_plane_bottom_frac", type=float, default=0.4,
                        help="Fraction of the image (from bottom) to use for ground plane fitting.")
    parser.add_argument("--zoom_out_upward_shift", type=float, default=0.05,
                        help="Extra linear upward shift (along ground normal) applied to zoom-out "
                             "trajectory. 0 = disabled. Units are in camera-space translation.")
    parser.add_argument("--zoom_out_upward_ratio", type=float, default=0.15,
                        help="Ratio of upward component added to the zoom-out backward trajectory. "
                             "0 = pure z-axis retreat, 0.15 = slight diagonal upward tilt. "
                             "Applied independently of --zoom_out_upward_shift.")
    parser.add_argument("--depth_backend", type=str, default="da3", choices=["da3"])
    parser.add_argument("--da3_model_name", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3_model_path_custom", type=str, default="checkpoints/recon/model.pt")
    parser.add_argument("--da3_frame_interval", type=int, default=8)
    parser.add_argument("--da3_max_history_frames", type=int, default=10)
    parser.add_argument("--da3_include_ar_chunk_last_frames", action="store_true")
    parser.add_argument("--da3_use_predicted_pose", action="store_true",
                        help="Use DA3-predicted camera poses (aligned to pipeline coords) for cache updates.")
    parser.add_argument("--da3_predicted_pose_continuation", action="store_true",
                        help="Apply DA3-predicted pose alignment for continuation segments.")

    # Misc flags needed by run_lyra2_sample internals
    parser.add_argument("--ablate_same_t5", action="store_true")
    parser.add_argument("--use_dmd_scheduler", action="store_true")
    parser.add_argument("--warp_chunk_size", type=int, default=None)
    parser.add_argument("--num_retrieval_views", type=int, default=1)
    parser.add_argument("--disable_cache_update", action="store_true")
    parser.add_argument("--multiview_ids", type=int, nargs="+", default=None)
    parser.add_argument("--offload_da3_diffusion", action="store_true")

    return parser.parse_args()




def _build_image_list(path: str) -> List[str]:
    if os.path.isdir(path):
        exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if os.path.splitext(f)[1] in exts]
        if not files:
            raise FileNotFoundError(f"No images found in folder: {path}")
        return files
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input image not found: {path}")
        return [path]


def _fit_ground_normal_from_depth(
    depth_hw: torch.Tensor,
    K_33: torch.Tensor,
    mask_hw: torch.Tensor,
    bottom_frac: float = 0.4,
    ransac_iters: int = 200,
    ransac_thresh: float = 0.05,
) -> torch.Tensor | None:
    """Fit a ground plane from the bottom portion of the depth map.

    Returns the plane normal in camera space (pointing 'up' away from ground),
    or None if fitting fails.
    """
    H, W = depth_hw.shape
    y_start = int(H * (1.0 - bottom_frac))

    valid = (mask_hw[y_start:] > 0.5) & (depth_hw[y_start:] > 0.01) & (depth_hw[y_start:] < 500.0)
    if valid.sum() < 50:
        return None

    ys, xs = torch.where(valid)
    ys = ys + y_start
    depths = depth_hw[ys, xs]

    fx, fy = K_33[0, 0], K_33[1, 1]
    cx, cy = K_33[0, 2], K_33[1, 2]
    X = (xs.float() - cx) / fx * depths
    Y = (ys.float() - cy) / fy * depths
    Z = depths
    pts = torch.stack([X, Y, Z], dim=-1)  # (N, 3)

    N_pts = pts.shape[0]
    best_normal = None
    best_inliers = 0

    for _ in range(ransac_iters):
        idx = torch.randint(0, N_pts, (3,))
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        n = torch.cross(v1, v2, dim=0)
        norm = n.norm()
        if norm < 1e-8:
            continue
        n = n / norm
        d = -torch.dot(n, p0)
        dists = (pts @ n + d).abs()
        inlier_count = (dists < ransac_thresh * Z.abs().clamp(min=0.1)).sum().item()
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = n

    if best_normal is None:
        return None

    # Ensure normal points "up" in camera space (negative y direction = up in image coords)
    if best_normal[1] > 0:
        best_normal = -best_normal

    log.info(
        f"[ground_plane] Fitted normal: [{best_normal[0]:.4f}, {best_normal[1]:.4f}, {best_normal[2]:.4f}], "
        f"inliers: {best_inliers}/{N_pts}",
        rank0_only=True,
    )
    return best_normal


def _correct_trajectory_ground_parallel(
    w2cs_T_44: torch.Tensor,
    ground_normal_cam: torch.Tensor,
) -> torch.Tensor:
    """Re-project w2c translations so camera moves parallel to the ground plane.

    The original trajectory's translation direction (typically camera z-axis) is
    projected onto the ground plane, preserving the total displacement magnitude.
    Camera orientation (rotation) is kept unchanged.
    """
    T = w2cs_T_44.shape[0]
    n = ground_normal_cam.to(w2cs_T_44.device, dtype=w2cs_T_44.dtype)

    t0 = w2cs_T_44[0, :3, 3]
    displacements = w2cs_T_44[:, :3, 3] - t0.unsqueeze(0)  # (T, 3)

    # Project each displacement onto the ground plane
    n_dot_d = (displacements * n.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (T, 1)
    projected = displacements - n_dot_d * n.unsqueeze(0)  # (T, 3)

    # Preserve original displacement magnitudes
    orig_norms = displacements.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    projected = projected * (orig_norms / proj_norms)
    # First frame stays at origin (no displacement)
    projected[0] = 0.0

    corrected = w2cs_T_44.clone()
    corrected[:, :3, 3] = t0.unsqueeze(0) + projected
    return corrected


def _generate_one_direction(
    *,
    model,
    args: argparse.Namespace,
    img_bchw: torch.Tensor,
    depth_hw: torch.Tensor,
    mask_hw: torch.Tensor,
    K_33: torch.Tensor,
    t5_embeddings: torch.Tensor,
    neg_t5_embeddings: torch.Tensor,
    trajectory: str,
    direction: str,
    strength: float,
    N: int,
    da3_model=None,
    process_group=None,
    log_prefix: str = "",
    ground_normal_cam: torch.Tensor | None = None,
    upward_shift: float = 0.0,
    zoom_out_upward_ratio: float = 0.0,
) -> dict | None:
    """Run AR spatial inference for a single camera trajectory direction."""
    device = model.tensor_kwargs.get("device", None)
    H, W = img_bchw.shape[-2:]

    initial_w2c = torch.eye(4, dtype=torch.float32, device=device)
    center_depth = torch.quantile(depth_hw[mask_hw > 0.5], 0.25)

    w2cs_T_44, Ks_T_33 = build_camera_trajectory(
        initial_w2c,
        K_33.to(initial_w2c),
        center_depth,
        N,
        trajectory,
        direction,
        strength,
    )

    if zoom_out_upward_ratio > 0.0:
        cam_centers = _camera_centers_from_w2c(w2cs_T_44)
        z_disp = cam_centers[:, 2] - cam_centers[0, 2]
        backward_amount = (-z_disp).clamp(min=0)
        upward_amount = backward_amount * zoom_out_upward_ratio
        cam_centers_shifted = cam_centers.clone()
        cam_centers_shifted[:, 1] -= upward_amount
        R = w2cs_T_44[:, :3, :3]
        new_t = -(R @ cam_centers_shifted.unsqueeze(-1)).squeeze(-1)
        w2cs_T_44 = w2cs_T_44.clone()
        w2cs_T_44[:, :3, 3] = new_t
        log.info(
            f"{log_prefix} [upward_tilt] Added upward ratio={zoom_out_upward_ratio:.3f}, "
            f"max_upward={upward_amount.max().item():.4f}",
            rank0_only=True,
        )

    if ground_normal_cam is not None:
        w2cs_T_44 = _correct_trajectory_ground_parallel(w2cs_T_44, ground_normal_cam)

        if upward_shift > 0.0:
            n = ground_normal_cam.to(w2cs_T_44.device, dtype=w2cs_T_44.dtype)
            T = w2cs_T_44.shape[0]
            ramp = torch.linspace(0, upward_shift, T, device=w2cs_T_44.device, dtype=w2cs_T_44.dtype)
            w2cs_T_44 = w2cs_T_44.clone()
            w2cs_T_44[:, :3, 3] -= ramp.unsqueeze(-1) * n.unsqueeze(0)

    w2cs_b_t_44 = w2cs_T_44.unsqueeze(0).to(dtype=torch.float32)
    Ks_b_t_33 = Ks_T_33.unsqueeze(0).to(dtype=torch.float32)

    depth_b_thw = depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1).to(device=device)

    data_batch = {
        "video": img_bchw.unsqueeze(2),
        "t5_text_embeddings": t5_embeddings,
        "neg_t5_text_embeddings": neg_t5_embeddings,
        "fps": torch.tensor([args.fps], dtype=torch.int32, device=device),
        "padding_mask": torch.zeros((1, 1, H, W), dtype=model.tensor_kwargs["dtype"], device=device),
        "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=device),
        "camera_w2c": w2cs_b_t_44,
        "intrinsics": Ks_b_t_33,
        "depth": depth_b_thw,
    }

    skip_keys = {"camera_w2c", "intrinsics", "depth"}
    data_batch = safe_to(
        data_batch,
        device=model.tensor_kwargs.get("device", None),
        dtype=model.tensor_kwargs.get("dtype", None),
        skip_keys=skip_keys,
    )

    saved_num_frames = args.num_frames
    args.num_frames = N
    try:
        result = run_lyra2_sample(
            model,
            data_batch,
            args,
            process_group=process_group,
            da3_model=da3_model,
            show_progress=True,
            log_prefix=log_prefix,
        )
    finally:
        args.num_frames = saved_num_frames

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()

    process_group = None
    if args.context_parallel_size > 1:
        import imaginaire
        from megatron.core import parallel_state
        imaginaire.utils.distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
        process_group = parallel_state.get_context_parallel_group()

    os.makedirs(args.output_path, exist_ok=True)
    misc.set_random_seed(seed=args.seed, by_rank=True)

    # Negative prompt embeddings
    negative_prompt_data = torch.load(
        "checkpoints/text_encoder/negative_prompt.pt", map_location="cpu", weights_only=False
    )

    # Load Lyra2 model
    experiment_opts = [
        "model.config.use_mp_policy_fsdp=False",
        "model.config.keep_original_net_dtype=False",
    ]
    if args.lora_paths:
        experiment_opts += ["model.config.net.postpone_checkpoint=True"]
    model, config = load_model_from_checkpoint(
        config_file="lyra_2/_src/configs/config.py",
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_dir,
        enable_fsdp=False,
        instantiate_ema=False,
        load_ema_to_reg=False,
        experiment_opts=experiment_opts,
    )
    if args.lora_paths:
        lora_names = []
        for lora_path in args.lora_paths:
            lora_name = model.load_lora_weights(lora_path)
            lora_names.append(lora_name)
        model.set_weights_and_activate_adapters(lora_names, args.lora_weights)
        if hasattr(model, "net") and hasattr(model.net, "enable_selective_checkpoint"):
            model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)

    desired_dtype = model.tensor_kwargs.get("dtype", None)
    desired_device = model.tensor_kwargs.get("device", None)
    if desired_dtype is not None:
        model.net = model.net.to(device=desired_device, dtype=desired_dtype)
        log.info(f"Casted model.net to dtype={desired_dtype}", rank0_only=True)

    assert getattr(model.config, "important_start", True) is True
    assert getattr(model.config, "encode_video_from_start", True) is True
    assert not getattr(model.config, "use_hd_map_cond", False)

    model.eval()
    if args.context_parallel_size > 1:
        model.net.enable_context_parallel(process_group)

    if args.warp_chunk_size is not None:
        model.config.warp_chunk_size = args.warp_chunk_size
        model.warp_chunk_size = args.warp_chunk_size

    # Resolution
    target_h, target_w = [int(x) for x in args.resolution.split(",")]

    # Load DA3 model
    from lyra_2._src.inference.depth_utils import load_da3_model
    da3_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    da3_model = load_da3_model(
        da3_model_name=args.da3_model_name,
        da3_model_path_custom=args.da3_model_path_custom,
        device=da3_device,
    )
    da3_model.eval()

    # Optionally load MoGe model for depth scale alignment
    moge_model = None
    if args.use_moge_scale:
        from lyra_2._src.inference.depth_utils import load_moge_model
        moge_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        moge_model = load_moge_model(moge_device)
        moge_model.eval()
        log.info("MoGe model loaded for depth scale alignment.", rank0_only=True)

    # Resolve image(s)
    all_image_paths = _build_image_list(args.input_image_path)
    if args.sample_id is not None:
        if args.sample_id < 0 or args.sample_id >= len(all_image_paths):
            raise IndexError(
                f"--sample_id {args.sample_id} out of range [0, {len(all_image_paths) - 1}]"
            )
        image_paths = [all_image_paths[args.sample_id]]
    else:
        image_paths = all_image_paths[args.sample_start_idx:args.sample_start_idx + args.num_samples]

    videos_dir = os.path.join(args.output_path, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for img_idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        per_image_dir = os.path.join(args.output_path, base_name)
        os.makedirs(per_image_dir, exist_ok=True)

        combined_video_path = os.path.join(videos_dir, f"{base_name}.mp4")
        if os.path.exists(combined_video_path):
            log.info(f"Skipping {img_path} (combined video already exists at {combined_video_path})", rank0_only=True)
            continue

        log.info(f"Processing [{img_idx}]: {img_path}", rank0_only=True)
        misc.set_random_seed(seed=args.seed, by_rank=True)

        # Read image
        bgr = cv2.imread(img_path)
        if bgr is None:
            log.error(f"Cannot read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb)  # H,W,3 uint8

        # Step 1: Depth & intrinsics
        log.info("Running DA3 single-image depth...", rank0_only=True)
        image_chw01, depth_hw, K_33, mask_hw = _da3_infer_depth_intrinsics_single(
            da3_model=da3_model,
            img_rgb_uint8=rgb_t,
            target_hw=(target_h, target_w),
        )
        H, W = image_chw01.shape[-2:]

        # Step 1b: Optionally align DA3 depth to MoGe scale
        if args.use_moge_scale and moge_model is not None:
            log.info("Aligning DA3 depth to MoGe scale...", rank0_only=True)
            from lyra_2._src.inference.depth_utils import moge_infer_depth_intrinsics

            moge_model.to(desired_device)
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.MATH]
            ):
                _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
                    moge_model,
                    rgb_t,
                    depth_pred_hw=(target_h, target_w),
                    target_hw=(target_h, target_w),
                )

            da3_d = depth_hw.to(moge_depth_hw.device)
            da3_m = mask_hw.to(moge_mask_hw.device)

            valid_mask = (da3_m > 0.5) & (moge_mask_hw > 0.5)
            if valid_mask.sum() > 10:
                d_da3_vals = da3_d[valid_mask]
                d_moge_vals = moge_depth_hw[valid_mask]

                inv_da3 = 1.0 / (d_da3_vals + 1e-6)
                inv_moge = 1.0 / (d_moge_vals + 1e-6)

                numerator = (inv_da3 * inv_moge).sum()
                denominator = (inv_da3 * inv_da3).sum()

                if denominator > 1e-8:
                    scale = numerator / denominator
                    log.info(f"Global inverse-depth scale factor: {scale.item()}", rank0_only=True)
                    if scale > 1e-6:
                        depth_hw = depth_hw / scale.to(depth_hw.device)
                    else:
                        log.warning(f"Scale too small ({scale.item()}), skipping alignment.", rank0_only=True)
                else:
                    log.warning("Denominator too small for LS scale alignment.", rank0_only=True)
            else:
                log.warning("Not enough overlapping valid pixels for scale alignment.", rank0_only=True)

            # Free MoGe GPU memory before video generation
            moge_model.cpu()
            del moge_depth_hw, moge_mask_hw, da3_d, da3_m
            torch.cuda.empty_cache()
            gc.collect()

        img_bchw = image_chw01.to(device=desired_device) * 2.0 - 1.0  # [-1,1]

        # Step 2: Load caption from .txt file or use explicit prompt
        if args.prompt:
            caption = args.prompt
            log.info(f"Using provided prompt: {caption}", rank0_only=True)
        elif args.prompt_dir:
            txt_path = os.path.join(args.prompt_dir, f"{base_name}.txt")
            if not os.path.isfile(txt_path):
                log.error(f"Caption file not found: {txt_path}  (expected for image {base_name})")
                continue
            with open(txt_path, "r") as f:
                caption = f.read().strip()
            log.info(f"Loaded caption from {txt_path}: {caption}", rank0_only=True)
        else:
            raise RuntimeError(
                "No caption source specified. Use --prompt for a global prompt, "
                "or --prompt_dir pointing to a folder of <image_stem>.txt files. "
                "Run scripts/gemini_caption.py first to generate captions."
            )

        if args.prompt_suffix:
            caption = caption.rstrip() + " " + args.prompt_suffix

        # Step 2b: T5 embeddings
        from lyra_2._src.inference.get_t5_emb import get_umt5_embedding, get_umt5_embedding_offloaded
        if args.offload_when_prompt:
            t5 = get_umt5_embedding_offloaded(caption, device=desired_device).to(dtype=desired_dtype)
        else:
            t5 = get_umt5_embedding(caption, device=desired_device).to(dtype=desired_dtype)
        if t5.dim() == 2:
            t5 = t5.unsqueeze(0)
        elif t5.dim() == 3 and t5.shape[0] != 1:
            t5 = t5[:1]
        neg_t5 = misc.to(negative_prompt_data["t5_text_embeddings"], **model.tensor_kwargs)

        N_in = int(args.num_frames_zoom_in or args.num_frames)
        N_out = int(args.num_frames_zoom_out or args.num_frames)

        # Step 2c: Optionally fit ground plane for trajectory alignment
        ground_normal = None
        if args.ground_plane_align:
            ground_normal = _fit_ground_normal_from_depth(
                depth_hw, K_33, mask_hw,
                bottom_frac=args.ground_plane_bottom_frac,
            )
            if ground_normal is None:
                log.warning("Ground plane fitting failed, using original trajectory.", rank0_only=True)

        # Step 3: Generate zoom-in video
        log.info(f"=== Generating ZOOM-IN video ({args.zoom_in_trajectory} {args.zoom_in_direction} str={args.zoom_in_strength}, N={N_in}) ===", rank0_only=True)
        result_in = _generate_one_direction(
            model=model,
            args=args,
            img_bchw=img_bchw,
            depth_hw=depth_hw,
            mask_hw=mask_hw,
            K_33=K_33,
            t5_embeddings=t5,
            neg_t5_embeddings=neg_t5,
            trajectory=args.zoom_in_trajectory,
            direction=args.zoom_in_direction,
            strength=args.zoom_in_strength,
            N=N_in,
            da3_model=da3_model,
            process_group=process_group,
            log_prefix=f"{base_name}_zoom_in",
            ground_normal_cam=ground_normal,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3b: Generate zoom-out video
        log.info(f"=== Generating ZOOM-OUT video ({args.zoom_out_trajectory} {args.zoom_out_direction} str={args.zoom_out_strength}, N={N_out}) ===", rank0_only=True)
        result_out = _generate_one_direction(
            model=model,
            args=args,
            img_bchw=img_bchw,
            depth_hw=depth_hw,
            mask_hw=mask_hw,
            K_33=K_33,
            t5_embeddings=t5,
            neg_t5_embeddings=neg_t5,
            trajectory=args.zoom_out_trajectory,
            direction=args.zoom_out_direction,
            strength=args.zoom_out_strength,
            N=N_out,
            da3_model=da3_model,
            process_group=process_group,
            log_prefix=f"{base_name}_zoom_out",
            upward_shift=args.zoom_out_upward_shift,
            ground_normal_cam=ground_normal,
            zoom_out_upward_ratio=args.zoom_out_upward_ratio,
        )

        if result_in is None and result_out is None:
            log.warning(f"Both zoom-in and zoom-out failed for {img_path}", rank0_only=True)
            continue

        # Save individual direction videos
        for tag, res in [("zoom_in", result_in), ("zoom_out", result_out)]:
            if res is None:
                continue
            vid_stem = os.path.join(per_image_dir, tag)
            to_show = []
            if res.get("warp_video") is not None:
                to_show.append(res["warp_video"])
            to_show.append(res["video"])
            save_output(to_show, vid_stem + ".mp4")
            log.info(f"Saved {tag} video: {vid_stem}.mp4", rank0_only=True)

        # Combine zoom-out (reversed) + zoom-in into a single video
        videos_to_combine = []
        if result_out is not None:
            videos_to_combine.append(result_out["video"].flip(dims=[2]))
        if result_in is not None:
            videos_to_combine.append(result_in["video"])

        combined_video = torch.cat(videos_to_combine, dim=2)  # [B, C, T_total, H, W]
        log.info(f"Combined video: {combined_video.shape[2]} frames from both directions", rank0_only=True)

        combined_01 = (combined_video[0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()
        save_img_or_video(combined_01, combined_video_path.replace(".mp4", ""), fps=args.fps)
        log.info(f"Saved combined video: {combined_video_path}", rank0_only=True)

        per_image_combined = os.path.join(per_image_dir, "combined")
        save_img_or_video(combined_01, per_image_combined, fps=args.fps)

        del combined_video, combined_01
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Clean up distributed
    if args.context_parallel_size > 1:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
        try:
            import torch.distributed as dist
            dist.destroy_process_group()
        except Exception:
            pass

    log.info("Done.", rank0_only=True)
