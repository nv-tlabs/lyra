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

from typing import Optional, Union

import torch
from einops import rearrange
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import VideoExtendCondition
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.utils import log, misc


class DiffusionMultiviewV2WModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_views = config.net.n_views

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        encoded_state = self.tokenizer.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        decoded_state = self.tokenizer.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return decoded_state

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
        """Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        condition_video_indicator is a binary tensor indicating the condition region in the latent state. 1x1xTx1x1 tensor.
        condition_video_input_mask will be concat with the input for the network.
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        condition_video_indicator = rearrange(
            condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views
        )
        # Only in inference to decide the condition region
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.info(
            f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        condition_video_indicator[:, :, :num_condition_t] += 1.0
        condition_video_indicator = rearrange(
            condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views
        )

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
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
        n_sample: int | None = None,
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
            xt = rearrange(xt, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)
            xt = rearrange(xt, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        for t in self.scheduler.timesteps:
            self.scheduler._init_step_index(t)
            sigma = self.scheduler.sigmas[self.scheduler.step_index].to(**self.tensor_kwargs)
            # Form new noise from latent
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
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        return samples

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
            latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            indicator = rearrange(indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            augment_latent_unscaled = rearrange(
                augment_latent_unscaled, "B C (V T) H W -> (B V) C T H W", V=self.n_views
            )

            latent = split_inputs_cp(latent, seq_dim=2, cp_group=self.net.cp_group)
            indicator = split_inputs_cp(indicator, seq_dim=2, cp_group=self.net.cp_group)
            augment_latent_unscaled = split_inputs_cp(augment_latent_unscaled, seq_dim=2, cp_group=self.net.cp_group)

            latent = rearrange(latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
            indicator = rearrange(indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
            augment_latent_unscaled = rearrange(
                augment_latent_unscaled, "(B V) C T H W -> B C (V T) H W", V=self.n_views
            )
        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_xt = indicator * augment_latent_unscaled + (1 - indicator) * xt
        return new_xt, latent, indicator
