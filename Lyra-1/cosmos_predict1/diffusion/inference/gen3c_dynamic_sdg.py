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

import argparse
import os
import copy
import torch
from pathlib import Path
import random
import numpy as np
from typing import Dict, Any
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache4D
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.data_loader_utils import load_data_auto_detect
from cosmos_predict1.diffusion.inference.vipe_utils import load_vipe_data
import torch.nn.functional as F
torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    ) # TODO: do we need this?
    parser.add_argument(
        "--input_image_path",
        type=str,
        help="Input image path for generating a single video",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left",
            "right",
            "up",
            "down",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
        ],
        default="left",
        help="Select a trajectory type from the available options (default: original)",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="center_facing",
        help="Controls camera rotation during movement: center_facing (rotate to look at center), no_rotation (keep orientation), or trajectory_aligned (rotate in the direction of movement)",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera from the center of the scene",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--vipe_path",
        type=str,
        default=None,
        help="Optional: path to VIPE clip root or the mp4 file under rgb/. If set, load VIPE-formatted data directly.",
    )
    parser.add_argument(
        "--vipe_starting_frame_idx",
        type=int,
        default=0,
        help="Starting frame index within the VIPE rgb mp4 to use as the reference frame.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    parser.add_argument(
        "--center_depth_quantile",
        action="store_true",
        help="If set, does not use center depth of 1.0 but quantile, which is needed for raw vipe results.",
    )
    parser.add_argument(
        "--multi_trajectory",
        action="store_true",
        help="If set, do multi-trajectory generation used by the 3DGS decoder.",
    )
    parser.add_argument(
        "--camera_gen_kwargs",
        type=Dict[str, Any],
        default={},
    )
    parser.add_argument(
        "--total_movement_distance_factor",
        type=float,
        default=1.0,
        help="Multiply multi trajectory setup with movement distance factor (larger means more movement but potentially more artifacts)",
    )
    parser.add_argument(
        "--flip_supervision",
        action="store_true",
        help="If set, this generates flipped camera trajectory supervision videos for all multi camera trajectories (only required for training).",
    )
    return parser.parse_args()

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"


def demo(args):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

    # Initialize video2world generation model pipeline
    pipeline = Gen3cPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )

    sample_n_frames = pipeline.model.chunk_size

    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        visual_input_path = args.vipe_path if args.vipe_path is not None else args.input_image_path
        prompts = [{"prompt": args.prompt, "visual_input": visual_input_path}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue
        current_video_path = input_dict.get("visual_input", None)
        if current_video_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue

        try:
            if args.vipe_path is not None:
                (
                    image_bchw_float,
                    depth_b1hw,
                    mask_b1hw,
                    initial_w2c_b44,
                    intrinsics_b33,
                ) = load_vipe_data(
                    vipe_root_or_mp4=args.vipe_path,
                    starting_frame_idx=args.vipe_starting_frame_idx,
                    resize_hw=(720, 1280),
                    crop_hw=(704, 1280),
                    num_frames=args.num_video_frames,
                )
            else:
                (
                    image_bchw_float,
                    depth_b1hw,
                    mask_b1hw,
                    initial_w2c_b44,
                    intrinsics_b33,
                ) = load_data_auto_detect(current_video_path)
        except Exception as e:
            log.critical(f"Failed to load visual input from {current_video_path}: {e}")
            continue

        image_bchw_float = image_bchw_float.to(device)
        depth_b1hw = depth_b1hw.to(device)
        mask_b1hw = mask_b1hw.to(device)
        initial_w2c_b44 = initial_w2c_b44.to(device)
        intrinsics_b33 = intrinsics_b33.to(device)

        # Reverse frame order before generation
        if args.flip_supervision:
            image_bchw_float = image_bchw_float.flip(dims=[0])
            depth_b1hw = depth_b1hw.flip(dims=[0])
            mask_b1hw = mask_b1hw.flip(dims=[0])
            initial_w2c_b44 = initial_w2c_b44.flip(dims=[0])
            intrinsics_b33 = intrinsics_b33.flip(dims=[0])

        cache = Cache4D(
            input_image=image_bchw_float.clone(), # [B, C, H, W]
            input_depth=depth_b1hw,       # [B, 1, H, W]
            input_mask=mask_b1hw,         # [B, 1, H, W]
            input_w2c=initial_w2c_b44,  # [B, 4, 4]
            input_intrinsics=intrinsics_b33,# [B, 3, 3]
            filter_points_threshold=args.filter_points_threshold,
            input_format=["F", "C", "H", "W"],
            foreground_masking=args.foreground_masking,
        )

        initial_cam_w2c_for_traj = initial_w2c_b44
        initial_cam_intrinsics_for_traj = intrinsics_b33

        # Generate camera trajectory using the new utility function
        try:
            # Set the center depth to 1.0 for already scaled depth/poses, otherwise use depth to determine it
            center_depth = torch.quantile(depth_b1hw[0], 0.5) if args.center_depth_quantile else 1.0
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=center_depth,
                device=device.type,
                **args.camera_gen_kwargs,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            continue

        log.info(f"Generating 0 - {sample_n_frames} frames")

        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, 0:sample_n_frames],
            generated_intrinsics[:, 0:sample_n_frames],
            start_frame_idx=0,
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())
        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=image_bchw_float[0].unsqueeze(0).unsqueeze(2),
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
            return_latents=True,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt, latents = generated_output

        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1) # Overlap by 1 frame
            end_frame_idx = start_frame_idx + sample_n_frames

            log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")

            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
            pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0 # (C,H,W), range [0,1]

            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
                start_frame_idx=start_frame_idx,
            )

            if args.save_buffer:
                all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())


            pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1 # (B,C,T,H,W), range [-1,1]
            generated_output = pipeline.generate(
                prompt=current_prompt,
                image_path=pred_image_for_depth_bcthw_minus1_1,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
                return_latents=True,
            )
            video_new, prompt, latents_new = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)
            latents = torch.cat([latents, latents_new[1:]], axis=0)

        # Final video processing
        final_video_to_save = video
        final_width = args.width

        if args.save_buffer and all_rendered_warps:
            squeezed_warps = [t.squeeze(0) for t in all_rendered_warps] # Each is (T_chunk, n_i, C, H, W)

            if squeezed_warps:
                n_max = max(t.shape[1] for t in squeezed_warps)

                padded_t_list = []
                for sq_t in squeezed_warps:
                    # sq_t shape: (T_chunk, n_i, C, H, W)
                    current_n_i = sq_t.shape[1]
                    padding_needed_dim1 = n_max - current_n_i

                    pad_spec = (0,0, # W
                                0,0, # H
                                0,0, # C
                                0,padding_needed_dim1, # n_i
                                0,0) # T_chunk
                    padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                    padded_t_list.append(padded_t)

                full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)

                T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
                buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
                buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
                buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
                buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))

                final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)
                log.info(f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}")
            else:
                log.info("No warp buffers to save.")


        # Output file name
        clip_name = Path(current_video_path).stem
        if prompt is not None and prompt != "":
            clip_name = f"{clip_name}_{prompt}"
        if args.batch_input_path is not None:
            clip_name = f"{clip_name}_{i}"

        # Save pose
        generated_c2ws = generated_w2cs.inverse()
        if args.flip_supervision:
            generated_w2cs = generated_w2cs.flip(dims=[1])
        pose_save_path = os.path.join(
            args.video_save_folder,
            "pose",
            f"{clip_name}.npz",
        )
        os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
        pose_list = []
        for i in range(generated_c2ws.shape[1]):
            pose = generated_c2ws[0, i].cpu().numpy()
            pose = pose.reshape(4, 4)
            pose_list.append((i, pose))
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        np.savez(
            pose_save_path,
            data=pose_data,
            inds=pose_inds,
        )

        # Save intrinsics
        if args.flip_supervision:
            generated_intrinsics = generated_intrinsics.flip(dims=[1])
        intrinsics_save_path = os.path.join(
            args.video_save_folder,
            "intrinsics",
            f"{clip_name}.npz",
        )
        os.makedirs(os.path.dirname(intrinsics_save_path), exist_ok=True)
        intrinsics_list = []
        for i in range(generated_intrinsics.shape[1]):
            intrinsics = generated_intrinsics[0, i].cpu().numpy()
            intrinsics_fxfycxcy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            intrinsics_list.append((i, intrinsics_fxfycxcy))
        intrinsics_data = np.stack(
            [intrinsics for _, intrinsics in intrinsics_list], axis=0
        )
        intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
        np.savez(
            intrinsics_save_path,
            data=intrinsics_data,
            inds=intrinsics_inds,
        )

        # Save latent
        latent_save_path = os.path.join(
            args.video_save_folder,
            "latent",
            f"{clip_name}.pkl",
        )
        os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
        video_latent = latents.detach().float().cpu().numpy()
        torch.save(video_latent, latent_save_path)

        # Save rgb video
        if args.flip_supervision:
            final_video_to_save = np.flip(final_video_to_save, axis=0)
        video_save_path = os.path.join(
            args.video_save_folder,
            "rgb",
            f"{clip_name}.mp4",
        )

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        # Save video
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=8,
            video_save_path=video_save_path,
        )
        log.info(f"Saved video to {video_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()

def demo_multi_trajectory(args):
    video_save_folder = args.video_save_folder
    flip_supervision = args.flip_supervision
    
    # Define trajectories
    args.camera_gen_kwargs = {'radius_x_factor': 0.15, 'radius_y_factor': 0.10, 'num_circles': 2}
    trajectories_list = []
    trajectories = {
        "left": {"traj_idx": 0, "movement_distance_range": [0.2, 0.3]},
        "right": {"traj_idx": 1, "movement_distance_range": [0.2, 0.3]},
        "up": {"traj_idx": 2, "movement_distance_range": [0.1, 0.2]},
        "zoom_out": {"traj_idx": 3, "movement_distance_range": [0.3, 0.4]},
        "zoom_in": {"traj_idx": 4, "movement_distance_range": [0.3, 0.4]},
        "clockwise": {"traj_idx": 5, "movement_distance_range": [0.4, 0.6]},
    }
    trajectories_list.append(trajectories)

    # Add flipped supervision for training
    if flip_supervision:
        num_trajectories = len(trajectories)
        trajectories_flipped = {}
        for traj_k, traj_dict in trajectories.items():
            # Main trajectories (first half of indices)
            traj_dict['flip_supervision'] = False
            # Flipped trajectories (second half of indices)
            traj_dict_flipped = copy.deepcopy(traj_dict)
            traj_dict_flipped["traj_idx"] += num_trajectories
            traj_dict_flipped['flip_supervision'] = True
            trajectories_flipped[traj_k] = traj_dict_flipped
        trajectories_list.append(trajectories_flipped)
    
    # Generate for each trajectory independently
    for trajectories in trajectories_list:
        for traj, traj_dict in trajectories.items():
            args.video_save_folder = os.path.join(video_save_folder, str(traj_dict["traj_idx"]))
            args.trajectory = traj
            args.movement_distance = random.uniform(
                traj_dict["movement_distance_range"][0],
                traj_dict["movement_distance_range"][1]
                ) * args.total_movement_distance_factor
            if flip_supervision:
                args.flip_supervision = traj_dict["flip_supervision"]
            demo(args)

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    if args.multi_trajectory:
        demo_multi_trajectory(args)
    else:
        demo(args)