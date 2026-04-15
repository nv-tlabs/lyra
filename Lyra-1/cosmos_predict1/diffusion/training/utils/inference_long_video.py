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
from contextlib import contextmanager
from typing import Tuple, Union

import einops
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as transforms_F
from matplotlib import pyplot as plt

from cosmos_predict1.diffusion.training.models.extend_model import ExtendDiffusionModel
from cosmos_predict1.utils import log
from cosmos_predict1.utils.easy_io import easy_io

"""This file contain functions needed for long video generation,
* function `generate_video_from_batch_with_loop` is used by `single_gpu_sep20`

"""


@contextmanager
def switch_config_for_inference(model):
    """For extend model inference, we need to make sure the condition_location is set to "first_n" and apply_corruption_to_condition_region is False.
    This context manager changes the model configuration to the correct settings for inference, and then restores the original settings when exiting the context.
    Args:
        model (ExtendDiffusionModel): video generation model
    """
    # Store the current condition_location
    current_condition_location = model.config.conditioner.video_cond_bool.condition_location
    if current_condition_location != "first_n" and current_condition_location != "first_and_last_1":
        current_condition_location = "first_n"
    current_apply_corruption_to_condition_region = (
        model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region
    )
    try:
        log.info(
            "Change the condition_location to 'first_n' for inference, and apply_corruption_to_condition_region to False"
        )
        # Change the condition_location to "first_n" for inference
        model.config.conditioner.video_cond_bool.condition_location = current_condition_location
        if current_apply_corruption_to_condition_region == "gaussian_blur":
            model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region = "clean"
        elif current_apply_corruption_to_condition_region == "noise_with_sigma":
            model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region = "noise_with_sigma_fixed"
        # Yield control back to the calling context
        yield
    finally:
        # Restore the original condition_location after exiting the context
        log.info(
            f"Restore the original condition_location {current_condition_location}, apply_corruption_to_condition_region {current_apply_corruption_to_condition_region}"
        )
        model.config.conditioner.video_cond_bool.condition_location = current_condition_location
        model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region = (
            current_apply_corruption_to_condition_region
        )


def visualize_latent_tensor_bcthw(tensor, nrow=1, show_norm=False, save_fig_path=None):
    """Debug function to display a latent tensor as a grid of images.
    Args:
        tensor (torch.Tensor): tensor in shape BCTHW
        nrow (int): number of images per row
        show_norm (bool): whether to display the norm of the tensor
        save_fig_path (str): path to save the visualization

    """
    log.info(
        f"display latent tensor shape {tensor.shape}, max={tensor.max()}, min={tensor.min()}, mean={tensor.mean()}, std={tensor.std()}"
    )
    tensor = tensor.float().cpu().detach()
    tensor = einops.rearrange(tensor, "b c (t n) h w -> (b t h) (n w) c", n=nrow)  # .numpy()
    # display the grid
    tensor_mean = tensor.mean(-1)
    tensor_norm = tensor.norm(dim=-1)
    log.info(f"tensor_norm, tensor_mean {tensor_norm.shape}, {tensor_mean.shape}")
    plt.figure(figsize=(20, 20))
    plt.imshow(tensor_mean)
    plt.title(f"mean {tensor_mean.mean()}, std {tensor_mean.std()}")
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        log.info(f"save to {os.path.abspath(save_fig_path)}")
        plt.savefig(save_fig_path, bbox_inches="tight", pad_inches=0)
    plt.show()
    if show_norm:
        plt.figure(figsize=(20, 20))
        plt.imshow(tensor_norm)
        plt.show()


def visualize_tensor_bcthw(tensor: torch.Tensor, nrow=4, save_fig_path=None):
    """Debug function to display a tensor as a grid of images.
    Args:
        tensor (torch.Tensor): tensor in shape BCTHW
        nrow (int): number of images per row
        save_fig_path (str): path to save the visualization
    """
    log.info(f"display {tensor.shape}, {tensor.max()}, {tensor.min()}")
    assert tensor.max() < 200, f"tensor max {tensor.max()} > 200, the data range is likely wrong"
    tensor = tensor.float().cpu().detach()
    tensor = einops.rearrange(tensor, "b c t h w -> (b t) c h w")
    # use torchvision to save the tensor as a grid of images
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)
    if save_fig_path is not None:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        log.info(f"save to {os.path.abspath(save_fig_path)}")
        torchvision.utils.save_image(tensor, save_fig_path)
    # display the grid
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def compute_num_frames_condition(model: "ExtendDiffusionModel", num_of_latent_overlap: int, downsample_factor=8) -> int:
    """This function computes the number of condition pixel frames given the number of latent frames to overlap.
    Args:
        model (ExtendDiffusionModel): Video generation model
        num_of_latent_overlap (int): Number of latent frames to overlap
        downsample_factor (int): Downsample factor for temporal reduce
    Returns:
        int: Number of condition frames in output space
    """
    # Access the VAE: use tokenizer.video_vae if it exists, otherwise use tokenizer directly
    vae = model.tokenizer.video_vae if hasattr(model.tokenizer, "video_vae") else model.tokenizer

    # Check if the VAE is causal (default to True if attribute not found)
    if getattr(vae, "is_casual", True):
        # For causal model
        num_frames_condition = num_of_latent_overlap // vae.latent_chunk_duration * vae.pixel_chunk_duration
        if num_of_latent_overlap % vae.latent_chunk_duration == 1:
            num_frames_condition += 1
        elif num_of_latent_overlap % vae.latent_chunk_duration > 1:
            num_frames_condition += 1 + (num_of_latent_overlap % vae.latent_chunk_duration - 1) * downsample_factor
    else:
        num_frames_condition = num_of_latent_overlap * downsample_factor

    return num_frames_condition


def read_video_or_image_into_frames_BCTHW(
    input_path: str,
    input_path_format: str = None,
    H: int = None,
    W: int = None,
    normalize: bool = True,
    max_frames: int = -1,
    also_return_fps: bool = False,
) -> torch.Tensor:
    """Read video or image from file and convert it to tensor. The frames will be normalized to [-1, 1].
    Args:
        input_path (str): path to the input video or image, end with .mp4 or .png or .jpg
        H (int): height to resize the video
        W (int): width to resize the video
    Returns:
        torch.Tensor: video tensor in shape (1, C, T, H, W), range [-1, 1]
    """
    log.info(f"Reading video from {input_path}")

    loaded_data = easy_io.load(input_path, file_format=input_path_format, backend_args=None)
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(loaded_data)  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 0
    else:
        frames, meta_data = loaded_data
        fps = int(meta_data.get("fps"))
    if max_frames != -1:
        frames = frames[:max_frames]
    input_tensor = np.stack(frames, axis=0)
    input_tensor = einops.rearrange(input_tensor, "t h w c -> t c h w")
    if normalize:
        input_tensor = input_tensor / 128.0 - 1.0
        input_tensor = torch.from_numpy(input_tensor).bfloat16()  # TCHW
        log.info(f"Raw data shape: {input_tensor.shape}")
        if H is not None and W is not None:
            input_tensor = transforms_F.resize(
                input_tensor,
                size=(H, W),  # type: ignore
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
    input_tensor = einops.rearrange(input_tensor, "(b t) c h w -> b c t h w", b=1)
    if normalize:
        input_tensor = input_tensor.to("cuda")
    log.info(f"Load shape {input_tensor.shape} value {input_tensor.min()}, {input_tensor.max()}")
    if also_return_fps:
        return input_tensor, fps
    return input_tensor


def create_condition_latent_from_input_frames(
    model: ExtendDiffusionModel,
    input_frames: torch.Tensor,
    num_frames_condition: int = 25,
):
    """Create condition latent for video generation. It will take the last num_frames_condition frames from the input frames as condition latent.
    Args:
        model (ExtendDiffusionModel): Video generation model
        input_frames (torch.Tensor): Video tensor in shape (1,C,T,H,W), range [-1, 1]
        num_frames_condition (int): Number of condition frames
    Returns:
        torch.Tensor: Condition latent in shape B,C,T,H,W
    """
    B, C, T, H, W = input_frames.shape
    # Dynamically access the VAE: use tokenizer.video_vae if it exists, otherwise use tokenizer directly
    vae = model.tokenizer.video_vae if hasattr(model.tokenizer, "video_vae") else model.tokenizer
    num_frames_encode = vae.pixel_chunk_duration  # Access pixel_chunk_duration from the VAE
    log.info(
        f"num_frames_encode not set, set it based on pixel chunk duration and model state shape: {num_frames_encode}"
    )

    log.info(
        f"Create condition latent from input frames {input_frames.shape}, value {input_frames.min()}, {input_frames.max()}, dtype {input_frames.dtype}"
    )

    assert (
        input_frames.shape[2] >= num_frames_condition
    ), f"input_frames not enough for condition, require at least {num_frames_condition}, got {input_frames.shape[2]}, {input_frames.shape}"
    assert (
        num_frames_encode >= num_frames_condition
    ), f"num_frames_encode should be larger than num_frames_condition, got {num_frames_encode}, {num_frames_condition}"

    # Put the conditional frames at the beginning of the video, and pad the end with zeros
    if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
        condition_frames_first = input_frames[:, :, :num_frames_condition]
        condition_frames_last = input_frames[:, :, -num_frames_condition:]
        padding_frames = condition_frames_first.new_zeros(B, C, num_frames_encode + 1 - 2 * num_frames_condition, H, W)
        encode_input_frames = torch.cat([condition_frames_first, padding_frames, condition_frames_last], dim=2)
    else:
        condition_frames = input_frames[:, :, -num_frames_condition:]
        padding_frames = condition_frames.new_zeros(B, C, num_frames_encode - num_frames_condition, H, W)
        encode_input_frames = torch.cat([condition_frames, padding_frames], dim=2)

    log.info(
        f"create latent with input shape {encode_input_frames.shape} including padding {num_frames_encode - num_frames_condition} at the end"
    )
    if hasattr(model, "n_views"):
        encode_input_frames = einops.rearrange(encode_input_frames, "(B V) C T H W -> B C (V T) H W", V=model.n_views)
    if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
        latent1 = model.encode(encode_input_frames[:, :, :num_frames_encode])  # BCTHW
        latent2 = model.encode(encode_input_frames[:, :, num_frames_encode:])
        latent = torch.cat([latent1, latent2], dim=2)  # BCTHW
    else:
        latent = model.encode(encode_input_frames)
    return latent, encode_input_frames


def get_condition_latent(
    model: ExtendDiffusionModel,
    conditioned_image_or_video_path: str,
    num_of_latent_condition: int = 4,
    state_shape: list[int] = None,
    input_path_format: str = None,
    frame_index: int = 0,
    frame_stride: int = 1,
):
    if state_shape is None:
        state_shape = model.state_shape
    if num_of_latent_condition == 0:
        log.info("No condition latent needed, return empty latent")
        condition_latent = (
            torch.zeros(
                [
                    1,
                ]
                + state_shape
            )
            .to(torch.bfloat16)
            .cuda()
        )
        return condition_latent, None

    H, W = (
        state_shape[-2] * model.vae.spatial_compression_factor,
        state_shape[-1] * model.vae.spatial_compression_factor,
    )
    input_frames = read_video_or_image_into_frames_BCTHW(
        conditioned_image_or_video_path,
        input_path_format=input_path_format,
        H=H,
        W=W,
    )
    if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
        start_frame = frame_index * frame_stride
        end_frame = (frame_index + 1) * frame_stride
        input_frames = torch.cat(
            [input_frames[:, :, start_frame : start_frame + 1], input_frames[:, :, end_frame : end_frame + 1]], dim=2
        ).contiguous()  # BCTHW

    num_frames_condition = compute_num_frames_condition(
        model, num_of_latent_condition, downsample_factor=model.vae.temporal_compression_factor
    )

    condition_latent, _ = create_condition_latent_from_input_frames(model, input_frames, num_frames_condition)
    condition_latent = condition_latent.to(torch.bfloat16)
    return condition_latent, input_frames


def generate_video_from_batch_with_loop(
    model: ExtendDiffusionModel,
    state_shape: list[int],
    is_negative_prompt: bool,
    data_batch: dict,
    condition_latent: torch.Tensor,
    # hyper-parameters for inference
    num_of_loops: int,
    num_of_latent_overlap_list: list[int],
    guidance: float,
    num_steps: int,
    seed: int,
    add_input_frames_guidance: bool = False,
    augment_sigma_list: list[float] = None,
    data_batch_list: Union[None, list[dict]] = None,
    visualize: bool = False,
    save_fig_path: str = None,
    skip_reencode: int = 0,
    return_noise: bool = False,
) -> Tuple[np.array, list, list, torch.Tensor] | Tuple[np.array, list, list, torch.Tensor, torch.Tensor]:
    """Generate video with loop, given data batch. The condition latent will be updated at each loop.
    Args:
        model (ExtendDiffusionModel)
        state_shape (list): shape of the state tensor
        is_negative_prompt (bool): whether to use negative prompt

        data_batch (dict): data batch for video generation
        condition_latent (torch.Tensor): condition latent in shape BCTHW

        num_of_loops (int): number of loops to generate video
        num_of_latent_overlap_list (list[int]): list number of latent frames to overlap between clips, different clips can have different overlap
        guidance (float): The guidance scale to use during sample generation; defaults to 5.0.
        num_steps (int): number of steps for diffusion sampling
        seed (int): random seed for sampling
        add_input_frames_guidance (bool): whether to add image guidance, default is False
        augment_sigma_list (list): list of sigma value for the condition corruption at different clip, used when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed". default is None

        data_batch_list (list): list of data batch for video generation, used when num_of_loops >= 1, to support multiple prompts in auto-regressive generation. default is None
        visualize (bool): whether to visualize the latent and grid, default is False
        save_fig_path (str): path to save the visualization, default is None

        skip_reencode (int): whether to skip re-encode the input frames, default is 0
        return_noise (bool): whether to return the initial noise used for sampling, used for ODE pairs generation. Default is False
    Returns:
        np.array: generated video in shape THWC, range [0, 255]
        list: list of condition latent, each in shape BCTHW
        list: list of sample latent, each in shape BCTHW
        torch.Tensor: initial noise used for sampling, shape BCTHW (if return_noise is True)
    """

    if data_batch_list is None:
        data_batch_list = [data_batch for _ in range(num_of_loops)]
    if visualize:
        assert save_fig_path is not None, "save_fig_path should be set when visualize is True"

    # Generate video with loop
    condition_latent_list = []
    decode_latent_list = []  # list collect the latent token to be decoded at the end
    sample_latent = []
    grid_list = []

    augment_sigma_list = (
        model.config.conditioner.video_cond_bool.apply_corruption_to_condition_region_sigma_value
        if augment_sigma_list is None
        else augment_sigma_list
    )

    for i in range(num_of_loops):
        num_of_latent_overlap_i = num_of_latent_overlap_list[i]
        num_of_latent_overlap_i_plus_1 = (
            num_of_latent_overlap_list[i + 1]
            if i + 1 < len(num_of_latent_overlap_list)
            else num_of_latent_overlap_list[-1]
        )
        if condition_latent.shape[2] < state_shape[1]:
            # Padding condition latent to state shape
            log.info(f"Padding condition latent {condition_latent.shape} to state shape {state_shape}")
            b, c, t, h, w = condition_latent.shape
            condition_latent = torch.cat(
                [
                    condition_latent,
                    condition_latent.new_zeros(b, c, state_shape[1] - t, h, w),
                ],
                dim=2,
            ).contiguous()
            log.info(f"after padding, condition latent shape {condition_latent.shape}")
        log.info(f"Generate video loop {i} / {num_of_loops}")
        if visualize:
            log.info(f"Visualize condition latent {i}")
            visualize_latent_tensor_bcthw(
                condition_latent[:, :, :4].float(),
                nrow=4,
                save_fig_path=os.path.join(save_fig_path, f"loop_{i:02d}_condition_latent_first_4.png"),
            )  # BCTHW

        condition_latent_list.append(condition_latent)

        if i < len(augment_sigma_list):
            condition_video_augment_sigma_in_inference = augment_sigma_list[i]
            log.info(f"condition_video_augment_sigma_in_inference {condition_video_augment_sigma_in_inference}")
        else:
            condition_video_augment_sigma_in_inference = augment_sigma_list[-1]
        assert not add_input_frames_guidance, "add_input_frames_guidance should be False, not supported"

        sample = model.generate_samples_from_batch(
            data_batch_list[i],
            guidance=guidance,
            state_shape=state_shape,
            num_steps=num_steps,
            is_negative_prompt=is_negative_prompt,
            seed=seed + i,
            condition_latent=condition_latent,
            num_condition_t=num_of_latent_overlap_i,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            return_noise=return_noise,
        )

        if return_noise:
            sample, noise = sample

        if visualize:
            log.info(f"Visualize sampled latent {i} 4-8 frames")
            visualize_latent_tensor_bcthw(
                sample[:, :, 4:8].float(),
                nrow=4,
                save_fig_path=os.path.join(save_fig_path, f"loop_{i:02d}_sample_latent_last_4.png"),
            )  # BCTHW

            diff_between_sample_and_condition = (sample - condition_latent)[:, :, :num_of_latent_overlap_i]
            log.info(
                f"Visualize diff between sample and condition latent {i} first 4 frames {diff_between_sample_and_condition.mean()}"
            )

        sample_latent.append(sample)
        T = condition_latent.shape[2]
        assert num_of_latent_overlap_i <= T, f"num_of_latent_overlap should be < T, get {num_of_latent_overlap_i}, {T}"

        if model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
            assert skip_reencode, "skip_reencode should be turned on when sample_tokens_start_from_p_or_i is True"
            if i == 0:
                decode_latent_list.append(sample)
            else:
                decode_latent_list.append(sample[:, :, num_of_latent_overlap_i:])
        else:
            # Interpolator mode. Decode the first and last as an image.
            if model.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
                grid_BCTHW_1 = (1.0 + model.decode(sample[:, :, :-1, ...])).clamp(0, 2) / 2  # [B, 3, T-1, H, W], [0, 1]
                grid_BCTHW_2 = (1.0 + model.decode(sample[:, :, -1:, ...])).clamp(0, 2) / 2  # [B, 3, 1, H, W], [0, 1]
                grid_BCTHW = torch.cat([grid_BCTHW_1, grid_BCTHW_2], dim=2)  # [B, 3, T, H, W], [0, 1]
            else:
                grid_BCTHW = (1.0 + model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W], [0, 1]

            if visualize:
                log.info(f"Visualize grid {i}")
                visualize_tensor_bcthw(
                    grid_BCTHW.float(), nrow=5, save_fig_path=os.path.join(save_fig_path, f"loop_{i:02d}_grid.png")
                )
            grid_np_THWC = (
                (grid_BCTHW[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
            )  # THW3, range [0, 255]

            # Post-process the output: cut the conditional frames from the output if it's not the first loop
            num_cond_frames = compute_num_frames_condition(
                model, num_of_latent_overlap_i_plus_1, downsample_factor=model.tokenizer.temporal_compression_factor
            )
            if i == 0:
                new_grid_np_THWC = grid_np_THWC  # First output, dont cut the conditional frames
            else:
                new_grid_np_THWC = grid_np_THWC[
                    num_cond_frames:
                ]  # Remove the conditional frames from the output, since it's overlapped with previous loop
            grid_list.append(new_grid_np_THWC)

            # Prepare the next loop: re-compute the condition latent
            if hasattr(model, "n_views"):
                grid_BCTHW = einops.rearrange(grid_BCTHW, "B C (V T) H W -> (B V) C T H W", V=model.n_views)
            condition_frame_input = grid_BCTHW[:, :, -num_cond_frames:] * 2 - 1  # BCTHW, range [0, 1] to [-1, 1]
        if skip_reencode:
            # Use the last num_of_latent_overlap latent token as condition latent
            log.info(f"Skip re-encode the condition frames, use the last {num_of_latent_overlap_i_plus_1} latent token")
            condition_latent = sample[:, :, -num_of_latent_overlap_i_plus_1:]
        else:
            # Re-encode the condition frames to get the new condition latent
            condition_latent, _ = create_condition_latent_from_input_frames(
                model, condition_frame_input, num_frames_condition=num_cond_frames
            )  # BCTHW
        condition_latent = condition_latent.to(torch.bfloat16)

    # save videos
    if model.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
        # decode all video together
        decode_latent_list = torch.cat(decode_latent_list, dim=2)
        grid_BCTHW = (1.0 + model.decode(decode_latent_list)).clamp(0, 2) / 2  # [B, 3, T, H, W], [0, 1]
        video_THWC = (
            (grid_BCTHW[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
        )  # THW3, range [0, 255]
    else:
        video_THWC = np.concatenate(grid_list, axis=0)  # THW3, range [0, 255]

    if return_noise:
        return video_THWC, condition_latent_list, sample_latent, noise
    return video_THWC, condition_latent_list, sample_latent
