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
import cv2
import torch
import numpy as np
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    validate_args,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_BufferSelector
import torch.nn.functional as F
torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
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
        "--npz_path",
        type=str,
        required=True,
        help="Path to NPZ exported by export_vipe_npz.py",
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
            "none",
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
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
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
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()


def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

    # Removed MoGe depth utilities; data comes from NPZ

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

    frame_buffer_max = 2 # pipeline.model.frame_buffer_max
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_n_frames = pipeline.model.chunk_size
    # Load NPZ data
    npz = np.load(args.npz_path)
    images_key = torch.tensor(npz["images_key_frames"], dtype=torch.float32, device=device)  # (N, C, H, W), [-1,1]
    depth_key = torch.tensor(npz["depth_key_frames"], dtype=torch.float32, device=device)    # (N, 1, H, W)
    mask_key = torch.tensor(npz["mask_key_frames"], dtype=torch.float32, device=device)      # (N, 1, H, W)
    K_key = torch.tensor(npz["K_key_frames"], dtype=torch.float32, device=device)            # (N, 3, 3) camera intrinsics in openCV convention
    w2cs_all_np = npz["w2cs_all"] # trajectory to generate the video
    Ks_all_np = npz["Ks_all"] if "Ks_all" in npz else None
    # Use stored key-frame w2cs directly
    w2c_key = torch.tensor(npz["w2cs_key_frames"], dtype=torch.float32, device=device)       # (N, 4, 4) world to camera transformation matrix of key frames

    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue

        input_image_bNCHW = images_key.unsqueeze(0)  # [1, N, C, H, W]
        input_depth_bN1HW = depth_key.unsqueeze(0)   # [1, N, 1, H, W]
        input_mask_bN1HW = mask_key.unsqueeze(0)     # [1, N, 1, H, W]
        input_w2c_bN44 = w2c_key.unsqueeze(0)        # [1, N, 4, 4]
        input_K_bN33 = K_key.unsqueeze(0)            # [1, N, 3, 3]

        cache = Cache3D_BufferSelector(
            frame_buffer_max=frame_buffer_max,
            input_image=input_image_bNCHW,
            input_depth=input_depth_bN1HW,
            input_mask=input_mask_bN1HW,
            input_w2c=input_w2c_bN44,
            input_intrinsics=input_K_bN33,
            filter_points_threshold=args.filter_points_threshold,
            input_format=["B", "N", "C", "H", "W"],
            foreground_masking=args.foreground_masking,
        )

        generated_w2cs = torch.tensor(w2cs_all_np, dtype=torch.float32, device=device)[:args.num_video_frames].unsqueeze(0)  # [1, T, 4, 4]
        if Ks_all_np is not None:
            generated_intrinsics = torch.tensor(Ks_all_np, dtype=torch.float32, device=device)[:args.num_video_frames].unsqueeze(0)  # [1, T, 3, 3]
        else:
            last_K = K_key[-1].unsqueeze(0).repeat(generated_w2cs.shape[1], 1, 1)
            generated_intrinsics = last_K.unsqueeze(0)

        log.info(f"Generating 0 - {sample_n_frames} frames")
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, 0:sample_n_frames],
            generated_intrinsics[:, 0:sample_n_frames],
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())

        seeding_bcthw_minus1_1 = input_image_bNCHW[:, 0].unsqueeze(2)  # [B,C,1,H,W]
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=seeding_bcthw_minus1_1,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt = generated_output

        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1) # Overlap by 1 frame
            end_frame_idx = start_frame_idx + sample_n_frames

            log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")

            # No cache update; reuse the multi-frame buffer and render for the next segment
            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
            )

            if args.save_buffer:
                all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
            pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0 # (C,H,W), range [0,1]
            pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1 # (B,C,T,H,W), range [-1,1]
            generated_output = pipeline.generate(
                prompt=current_prompt,
                image_path=pred_image_for_depth_bcthw_minus1_1,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
            video_new, prompt = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)

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


        video_save_path = os.path.join(
            args.video_save_folder,
            f"{i if args.batch_input_path else args.video_save_name}.mp4"
        )

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        # Save video
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )
        log.info(f"Saved video to {video_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)