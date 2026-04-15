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

from typing import Optional

import torch
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import VideoExtendCondition
from cosmos_predict1.diffusion.model.model_t2w import DiffusionT2WModel, broadcast_condition
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.utils import log, misc


class DiffusionV2WModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Optional[int] = None
    ) -> VideoExtendCondition:
        """Adds conditioning masks to VideoExtendCondition object.

        Creates binary indicators and input masks for conditional video generation.

        Args:
            latent_state: Input latent tensor (B,C,T,H,W)
            condition: VideoExtendCondition object to update
            num_condition_t: Number of frames to condition on

        Returns:
            Updated VideoExtendCondition with added masks:
            - condition_video_indicator: Binary tensor marking condition regions
            - condition_video_input_mask: Input mask for network
            - gt_latent: Ground truth latent tensor
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        # Only in inference to decide the condition region
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.debug(
            f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        condition_video_indicator[:, :, :num_condition_t] += 1.0

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        return condition

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        condition_augment_sigma: float = None,
        add_input_frames_guidance: bool = False,
    ) -> Tensor:
        """Generates video samples conditioned on input frames.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility
            state_shape: Shape of output tensor (defaults to model's state shape)
            n_sample: Number of samples to generate (defaults to batch size)
            is_negative_prompt: Whether to use negative prompting
            num_steps: Number of denoising steps
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_augment_sigma: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            Generated video samples tensor
        """
        assert condition_latent is not None, "condition_latent should be provided"
        condition, uncondition = self._get_conditions(
            data_batch, is_negative_prompt, condition_latent, num_condition_t, add_input_frames_guidance
        )

        self.scheduler.set_timesteps(num_steps)
        if n_sample is None:
            n_sample = condition_latent.shape[0]
        xt = torch.randn(size=(n_sample,) + tuple(state_shape), **self.tensor_kwargs) * self.scheduler.init_noise_sigma

        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)

        for t in self.scheduler.timesteps:
            self.scheduler._init_step_index(t)
            sigma = self.scheduler.sigmas[self.scheduler.step_index].to(**self.tensor_kwargs)
            # Form new noise from latent
            xt = xt.to(**self.tensor_kwargs)
            new_xt, latent, indicator = self._augment_noise_with_latent(
                xt, sigma, condition, condition_augment_sigma=condition_augment_sigma, seed=seed
            )
            new_xt = new_xt.to(**self.tensor_kwargs)
            new_xt_scaled = self.scheduler.scale_model_input(new_xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            net_output_cond = self.net(x=new_xt_scaled, timesteps=t, **condition.to_dict())
            net_output_uncond = self.net(x=new_xt_scaled, timesteps=t, **uncondition.to_dict())
            net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            # Replace indicated output with latent
            latent_unscaled = self._reverse_precondition_output(latent, xt=new_xt, sigma=sigma)
            new_output = indicator * latent_unscaled + (1 - indicator) * net_output
            # Compute the previous noisy sample x_t -> x_t-1
            xt = self.scheduler.step(new_output, t, new_xt).prev_sample
        samples = xt

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        add_input_frames_guidance: bool = False,
    ):
        """Get the conditions for the model.

        Args:
            data_batch: Input data dictionary
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            condition: Input conditions
            uncondition: Conditions removed/reduced to minimum (unconditioned)
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )
        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )
        assert condition.gt_latent.allclose(uncondition.gt_latent)

        # For inference, check if parallel_state is initialized
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def _augment_noise_with_latent(
        self,
        xt: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_augment_sigma: float = 0.001,
        seed: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Augments the conditional frames with noise during inference.

        Args:
            xt (Tensor): noise
            sigma (Tensor): noise level for the generation region
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            condition_augment_sigma (float): sigma for condition video augmentation in inference
            seed (int): random seed for reproducibility
        Returns:
            new_xt (Tensor): new latent-augmented noise tensor in shape B,C,T,H,W
            latent (Tensor): ground-truth latent tensor in shape B,C,T,H,W
            indicator (Tensor): ground-truth latent binary indicator tensor in shape B,C,T,H,W

        """
        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        augment_sigma = condition_augment_sigma
        latent = condition.gt_latent
        indicator = condition.condition_video_indicator
        if augment_sigma >= sigma:
            indicator = torch.zeros_like(indicator)
        # Now apply the augment_sigma to the gt_latent
        noise = misc.arch_invariant_rand(
            latent.shape,
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )
        augment_latent = latent + noise * augment_sigma
        augment_latent = self.scheduler.precondition_inputs(augment_latent, augment_sigma)
        augment_latent_unscaled = self._reverse_precondition_input(augment_latent, sigma)
        if self.net.is_context_parallel_enabled:
            latent = split_inputs_cp(condition.gt_latent, seq_dim=2, cp_group=self.net.cp_group)
            indicator = split_inputs_cp(indicator, seq_dim=2, cp_group=self.net.cp_group)
            augment_latent_unscaled = split_inputs_cp(augment_latent_unscaled, seq_dim=2, cp_group=self.net.cp_group)
        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_xt = indicator * augment_latent_unscaled + (1 - indicator) * xt
        return new_xt, latent, indicator

    def _reverse_precondition_input(self, xt: Tensor, sigma: Tensor) -> Tensor:
        c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        xt_unscaled = xt / c_in
        return xt_unscaled

    def _reverse_precondition_output(self, latent: Tensor, xt: Tensor, sigma: Tensor) -> Tensor:
        sigma_data = self.scheduler.config.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        latent_unscaled = (latent - c_skip * xt) / c_out
        return latent_unscaled
