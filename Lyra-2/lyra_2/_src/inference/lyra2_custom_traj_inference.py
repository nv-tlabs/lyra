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
Single-image → video generation along a user-supplied camera trajectory.

Pipeline:
1. Read a single input image.
2. Load per-chunk text captions from a captions.json file (or single --prompt).
3. Load camera trajectory (w2c + intrinsics) from an .npz file,
   take the first ``num_frames`` poses.
4. Produce a video using FramePack AR spatial generation with per-chunk T5 embeddings.
5. Save the output video.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

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
from lyra_2._src.inference.lyra2_zoomgs_inference import (
    _da3_infer_depth_intrinsics_single,
    _build_image_list,
)
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

torch.enable_grad(False)
torch.backends.cudnn.enabled = False


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory(
    path: str,
    num_frames: int,
    target_hw: tuple[int, int] | None = None,
    pose_scale: float = 1.0,
):
    """Load camera trajectory from an .npz file.

    Expected keys:
        w2c        – (N, 4, 4) world-to-camera matrices  (float32/64)
        intrinsics – (N, 3, 3) camera intrinsic matrices  (float32/64)
        image_height, image_width – original resolution the intrinsics refer to

    If *target_hw* is provided and differs from the stored resolution,
    intrinsics are rescaled accordingly.

    Returns the first *num_frames* entries as torch tensors.
    """
    data = np.load(path)
    w2c = torch.from_numpy(data["w2c"][:num_frames].astype(np.float32))
    intrinsics = torch.from_numpy(data["intrinsics"][:num_frames].astype(np.float32))

    if pose_scale != 1.0:
        w2c[:, :3, 3] *= pose_scale

    if target_hw is not None and "image_height" in data and "image_width" in data:
        orig_h, orig_w = int(data["image_height"]), int(data["image_width"])
        tgt_h, tgt_w = target_hw
        if (orig_h, orig_w) != (tgt_h, tgt_w):
            sx = tgt_w / orig_w
            sy = tgt_h / orig_h
            intrinsics[:, 0, 0] *= sx
            intrinsics[:, 0, 2] *= sx
            intrinsics[:, 1, 1] *= sy
            intrinsics[:, 1, 2] *= sy

    return w2c, intrinsics


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image video generation with a custom camera trajectory"
    )
    # Input
    parser.add_argument("--input_image_path", type=str, required=True,
                        help="Path to a single image or a folder of images")
    parser.add_argument("--trajectory_path", type=str, required=True,
                        help="Path to .npz trajectory file (or a folder of per-image .npz files). "
                             "Expected keys: w2c (N,4,4), intrinsics (N,3,3), "
                             "image_height, image_width.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_start_idx", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional explicit prompt applied to ALL images (single caption).")
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="Directory containing per-image .txt caption files (single caption).")
    parser.add_argument("--captions_path", type=str, default=None,
                        help="Path to captions.json (or dir with per-image .json files). "
                             "JSON maps frame-index strings to caption text. "
                             "Each AR chunk uses the caption whose key is <= current frame.")
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="Text appended to every prompt.")

    # Model and generation
    parser.add_argument("--experiment", type=str, default="lyra_framepack_spatial")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model")
    parser.add_argument("--output_path", type=str, default="inference/lyra2_custom_traj")
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--num_sampling_step", type=int, default=35)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=161,
                        help="Number of frames to generate (taken from the start of the trajectory).")
    parser.add_argument("--pose_scale", type=float, default=1.1,
                        help="Scale factor applied to w2c translation vectors.")
    parser.add_argument("--resolution", type=str, default="480,832", help="H,W")
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--lora_paths", type=str, default=None, nargs="+")
    parser.add_argument("--lora_weights", type=float, default=None, nargs="+")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload_when_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # Depth backend
    parser.add_argument("--use_moge_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Align DA3 depth to MoGe scale (default: True).")
    parser.add_argument("--depth_backend", type=str, default="da3", choices=["da3"])
    parser.add_argument("--da3_model_name", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3_model_path_custom", type=str, default=None)
    parser.add_argument("--da3_frame_interval", type=int, default=8)
    parser.add_argument("--da3_max_history_frames", type=int, default=10)
    parser.add_argument("--da3_include_ar_chunk_last_frames", action="store_true")
    parser.add_argument("--da3_use_predicted_pose", action="store_true")
    parser.add_argument("--da3_predicted_pose_continuation", action="store_true")

    # Misc flags needed by run_lyra2_sample internals
    parser.add_argument("--ablate_same_t5", action="store_true")
    parser.add_argument("--use_dmd_scheduler", action="store_true")
    parser.add_argument("--warp_chunk_size", type=int, default=None)
    parser.add_argument("--num_retrieval_views", type=int, default=1)
    parser.add_argument("--disable_cache_update", action="store_true")
    parser.add_argument("--multiview_ids", type=int, nargs="+", default=None)
    parser.add_argument("--offload_da3_diffusion", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()

    if args.debug:
        import debugpy
        debugpy.listen(5678)
        log.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()

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

    # ---- Load FramePack model ----
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

    # ---- Load DA3 model ----
    from lyra_2._src.inference.depth_utils import load_da3_model
    da3_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    da3_model = load_da3_model(
        da3_model_name=args.da3_model_name,
        da3_model_path_custom=args.da3_model_path_custom,
        device=da3_device,
    )
    da3_model.eval()

    # ---- Optionally load MoGe model for depth scale alignment ----
    moge_model = None
    if args.use_moge_scale:
        from lyra_2._src.inference.depth_utils import load_moge_model
        moge_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        moge_model = load_moge_model(moge_device)
        moge_model.eval()
        log.info("MoGe model loaded for depth scale alignment.", rank0_only=True)

    # ---- Resolve image(s) ----
    image_paths = _build_image_list(args.input_image_path)[
        args.sample_start_idx : args.sample_start_idx + args.num_samples
    ]

    # Resolve trajectory file(s): single file shared across images, or per-image files in a folder.
    traj_is_dir = os.path.isdir(args.trajectory_path)

    # Resolve captions source: per-chunk JSON or single caption
    captions_is_dir = args.captions_path is not None and os.path.isdir(args.captions_path)

    N = int(args.num_frames)

    for img_idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        video_path = os.path.join(args.output_path, f"{base_name}.mp4")
        if os.path.exists(video_path):
            log.info(f"Skipping {img_path} (video already exists at {video_path})", rank0_only=True)
            continue

        log.info(f"Processing [{img_idx}]: {img_path}", rank0_only=True)
        misc.set_random_seed(seed=args.seed, by_rank=True)

        # ---- Load trajectory ----
        if traj_is_dir:
            traj_file = os.path.join(args.trajectory_path, f"{base_name}.npz")
        else:
            traj_file = args.trajectory_path
        if not os.path.isfile(traj_file):
            log.error(f"Trajectory file not found: {traj_file}")
            continue

        w2cs_T_44, Ks_T_33 = load_trajectory(traj_file, N, target_hw=(target_h, target_w), pose_scale=args.pose_scale)
        log.info(f"Loaded trajectory: {w2cs_T_44.shape[0]} frames from {traj_file}", rank0_only=True)

        # ---- Read image ----
        bgr = cv2.imread(img_path)
        if bgr is None:
            log.error(f"Cannot read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb)

        # ---- Depth & intrinsics for the first frame (via DA3) ----
        log.info("Running DA3 single-image depth...", rank0_only=True)
        image_chw01, depth_hw, _K_33_da3, mask_hw = _da3_infer_depth_intrinsics_single(
            da3_model=da3_model,
            img_rgb_uint8=rgb_t,
            target_hw=(target_h, target_w),
        )
        H, W = image_chw01.shape[-2:]

        # ---- Optionally align DA3 depth to MoGe scale ----
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

            moge_model.cpu()
            del moge_depth_hw, moge_mask_hw, da3_d, da3_m
            torch.cuda.empty_cache()
            gc.collect()

        img_bchw = image_chw01.to(device=desired_device) * 2.0 - 1.0

        # ---- Load captions ----
        from lyra_2._src.inference.get_t5_emb import get_umt5_embedding, get_umt5_embedding_offloaded
        neg_t5 = misc.to(negative_prompt_data["t5_text_embeddings"], **model.tensor_kwargs)

        captions_file = None
        if args.captions_path is not None:
            if captions_is_dir:
                captions_file = os.path.join(args.captions_path, f"{base_name}.json")
            else:
                captions_file = args.captions_path
            if not os.path.isfile(captions_file):
                log.warning(f"Captions file not found: {captions_file}, falling back to single caption")
                captions_file = None

        use_chunk_captions = False
        if captions_file is not None:
            with open(captions_file, "r") as f:
                captions_dict = json.load(f)
            chunk_keys_int = sorted(int(k) for k in captions_dict)
            chunk_keys_int = [k for k in chunk_keys_int if k < N]
            if len(chunk_keys_int) > 1:
                use_chunk_captions = True
                log.info(f"Loaded {len(chunk_keys_int)} chunk captions from {captions_file}", rank0_only=True)

                chunk_keys = torch.tensor(chunk_keys_int, dtype=torch.long, device=desired_device)
                chunk_embs = []
                chunk_masks = []
                for ck in chunk_keys_int:
                    cap = captions_dict[str(ck)]
                    if args.prompt_suffix:
                        cap = cap.rstrip() + " " + args.prompt_suffix
                    if args.offload_when_prompt:
                        emb = get_umt5_embedding_offloaded(cap, device=desired_device).to(dtype=desired_dtype)
                    else:
                        emb = get_umt5_embedding(cap, device=desired_device).to(dtype=desired_dtype)
                    if emb.dim() == 3:
                        emb = emb[0]
                    S, D = emb.shape
                    S = min(S, 512)
                    D = min(D, 4096)
                    padded_emb = torch.zeros(512, 4096, dtype=desired_dtype, device=desired_device)
                    padded_emb[:S, :D] = emb[:S, :D]
                    padded_mask = torch.zeros(512, dtype=desired_dtype, device=desired_device)
                    padded_mask[:S] = 1.0
                    chunk_embs.append(padded_emb)
                    chunk_masks.append(padded_mask)

                t5_chunk_embeddings = torch.stack(chunk_embs).unsqueeze(0)
                t5_chunk_mask = torch.stack(chunk_masks).unsqueeze(0)
                t5_chunk_keys = chunk_keys.unsqueeze(0)
                sample_frame_indices = torch.arange(N, dtype=torch.long, device=desired_device).unsqueeze(0)
                t5 = t5_chunk_embeddings[:, 0, :, :]
            else:
                single_caption = captions_dict.get(str(chunk_keys_int[0]), "") if chunk_keys_int else ""
                if args.prompt_suffix:
                    single_caption = single_caption.rstrip() + " " + args.prompt_suffix

        if not use_chunk_captions:
            if args.prompt:
                caption = args.prompt
            elif captions_file is not None:
                caption = single_caption
            elif args.prompt_dir:
                txt_path = os.path.join(args.prompt_dir, f"{base_name}.txt")
                if not os.path.isfile(txt_path):
                    log.error(f"Caption file not found: {txt_path}")
                    continue
                with open(txt_path, "r") as f:
                    caption = f.read().strip()
                log.info(f"Loaded caption from {txt_path}", rank0_only=True)
            else:
                raise RuntimeError(
                    "No caption source specified. Use --captions_path, --prompt, or --prompt_dir."
                )
            if args.prompt_suffix:
                caption = caption.rstrip() + " " + args.prompt_suffix
            if args.offload_when_prompt:
                t5 = get_umt5_embedding_offloaded(caption, device=desired_device).to(dtype=desired_dtype)
            else:
                t5 = get_umt5_embedding(caption, device=desired_device).to(dtype=desired_dtype)
            if t5.dim() == 2:
                t5 = t5.unsqueeze(0)
            elif t5.dim() == 3 and t5.shape[0] != 1:
                t5 = t5[:1]

        # ---- Assemble data batch ----
        w2cs_b_t_44 = w2cs_T_44.unsqueeze(0).to(dtype=torch.float32, device=desired_device)
        Ks_b_t_33 = Ks_T_33.unsqueeze(0).to(dtype=torch.float32, device=desired_device)
        depth_b_thw = depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1).to(device=desired_device)

        data_batch = {
            "video": img_bchw.unsqueeze(2),
            "t5_text_embeddings": t5,
            "neg_t5_text_embeddings": neg_t5,
            "fps": torch.tensor([args.fps], dtype=torch.int32, device=desired_device),
            "padding_mask": torch.zeros((1, 1, H, W), dtype=model.tensor_kwargs["dtype"], device=desired_device),
            "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=desired_device),
            "camera_w2c": w2cs_b_t_44,
            "intrinsics": Ks_b_t_33,
            "depth": depth_b_thw,
        }

        if use_chunk_captions:
            data_batch["t5_chunk_keys"] = t5_chunk_keys
            data_batch["t5_chunk_embeddings"] = t5_chunk_embeddings
            data_batch["t5_chunk_mask"] = t5_chunk_mask
            data_batch["sample_frame_indices"] = sample_frame_indices

        skip_keys = {"camera_w2c", "intrinsics", "depth", "t5_chunk_keys", "sample_frame_indices"}
        data_batch = safe_to(
            data_batch,
            device=model.tensor_kwargs.get("device", None),
            dtype=model.tensor_kwargs.get("dtype", None),
            skip_keys=skip_keys,
        )

        # ---- Run AR inference ----
        log.info(f"=== Generating video ({N} frames) ===", rank0_only=True)
        result = run_lyra2_sample(
            model,
            data_batch,
            args,
            process_group=process_group,
            da3_model=da3_model,
            show_progress=True,
            log_prefix=f"{base_name}_custom_traj",
        )

        if result is None:
            log.warning(f"Generation failed for {img_path}", rank0_only=True)
            continue

        # ---- Save output video ----
        video_01 = (result["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()
        save_img_or_video(video_01, video_path.replace(".mp4", ""), fps=args.fps)
        log.info(f"Saved video: {video_path}", rank0_only=True)

        del result, data_batch
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
