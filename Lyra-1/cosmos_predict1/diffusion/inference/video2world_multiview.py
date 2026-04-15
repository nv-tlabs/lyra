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

import torch

from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    get_input_sizes,
    remove_argument,
    validate_args,
)
from cosmos_predict1.diffusion.inference.world_generation_pipeline import (
    DiffusionVideo2WorldMultiviewGenerationPipeline,
)
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)
    remove_argument(parser, "width")
    remove_argument(parser, "height")
    remove_argument(parser, "num_video_frames")
    parser.add_argument("--height", type=int, default=480, help="Height of video to sample")
    parser.add_argument("--width", type=int, default=848, help="Width of video to sample")

    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=57,
        choices=[57],
        help="Number of video frames to sample, this is per-camera frame number.",
    )
    # Add video2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview",
        ],
    )
    parser.add_argument(
        "--prompt_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the left. ",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing to the right.",
        help="Text prompt for generating right camera view video",
    )

    parser.add_argument(
        "--prompt_back",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing backwards.",
        help="Text prompt for generating rear camera view video",
    )
    parser.add_argument(
        "--prompt_back_left",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
        help="Text prompt for generating left camera view video",
    )
    parser.add_argument(
        "--prompt_back_right",
        type=str,
        default="The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
        help="Text prompt for generating right camera view video",
    )
    parser.add_argument(
        "--frame_repeat_negative_condition",
        type=float,
        default=10.0,
        help="frame_repeat number to be used as negative condition",
    )
    parser.add_argument(
        "--input_image_or_video_path",
        type=str,
        help="Input video/image path for generating a single video",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of input frames for video2world prediction",
        choices=[1, 9],
    )

    return parser.parse_args()


def demo(args):
    """Run multi-view video-to-world generation demo.

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
    validate_args(args, inference_type)

    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

    # Initialize video2world generation model pipeline
    pipeline = DiffusionVideo2WorldMultiviewGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        frame_repeat_negative_condition=args.frame_repeat_negative_condition,
        seed=args.seed,
        num_input_frames=args.num_input_frames,
    )

    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [
            {
                "prompt": args.prompt,
                "prompt_left": args.prompt_left,
                "prompt_right": args.prompt_right,
                "prompt_back": args.prompt_back,
                "prompt_back_left": args.prompt_back_left,
                "prompt_back_right": args.prompt_back_right,
                "visual_input": args.input_image_or_video_path,
            }
        ]

    os.makedirs(args.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_image_or_video_path = input_dict.pop("visual_input", None)
        if current_image_or_video_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue
        current_prompt = input_dict

        # Check input frames
        if not check_input_frames(current_image_or_video_path, args.num_input_frames):
            continue
        log.warning("Visual input is provided, overriding --height and --width arguments.")
        args.height, args.width = get_input_sizes(current_image_or_video_path)

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_or_video_path=current_image_or_video_path,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        [video_grid, video], prompt = generated_output

        if args.batch_input_path:
            video_save_path = os.path.join(args.video_save_folder, f"{i}.mp4")
            video_grid_save_path = os.path.join(args.video_save_folder, f"{i}_grid.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
            video_grid_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}_grid.mp4")
            prompt_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.txt")

        # Save video
        save_video(
            video=video,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=10,
            video_save_path=video_save_path,
        )
        save_video(
            video=video_grid,
            fps=args.fps,
            H=args.height * 2,
            W=args.width * 3,
            video_save_quality=5,
            video_save_path=video_grid_save_path,
        )
        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            for key, value in prompt.items():
                f.write(value.encode("utf-8"))
                f.write("\n".encode("utf-8"))

        log.info(f"Saved video to {video_save_path}")
        log.info(f"Saved prompt to {prompt_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
