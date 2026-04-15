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

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import VideoExtendCondition
from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel, broadcast_condition
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.modules.res_sampler import Sampler
from cosmos_predict1.utils import log, misc

IS_PREPROCESSED_KEY = "is_preprocessed"
from cosmos_predict1.diffusion.modules.denoiser_scaling import EDMScaling
from cosmos_predict1.diffusion.types import DenoisePrediction


class DiffusionWorldInterpolatorWModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.is_extend_model = True
        self.num_valid_latents = config.latent_shape[1] - config.num_latents_to_drop
        self.input_image_key = getattr(self.config, "input_image_key", None)
        self.input_data_key = self.config.input_data_key
        self.sampler = Sampler()  # Added to resolve the AttributeError
        self.scaling = EDMScaling(self.sigma_data)

    def denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: VideoExtendCondition) -> DenoisePrediction:
        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)
        condition_dict = {
            k: v.to(self.precision) if isinstance(v, torch.Tensor) else v for k, v in condition.to_dict().items()
        }
        net_output = self.net(
            x=batch_mul(c_in, xt),
            timesteps=c_noise,
            **condition_dict,
        )
        logvar = self.model.logvar(c_noise) if hasattr(self.model, "logvar") else None
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)
        return DenoisePrediction(x0_pred, eps_pred, logvar)

    def _normalize_video_databatch_inplace(self, data_batch: Dict[str, Tensor]) -> None:
        if self.input_data_key in data_batch:
            if IS_PREPROCESSED_KEY not in data_batch or not data_batch[IS_PREPROCESSED_KEY]:
                assert data_batch[self.input_data_key].dtype == torch.uint8, "Video data must be uint8."
                data_batch[self.input_data_key] = data_batch[self.input_data_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: Dict[str, Tensor]) -> None:
        if self.input_image_key in data_batch:
            if IS_PREPROCESSED_KEY not in data_batch or not data_batch[IS_PREPROCESSED_KEY]:
                data_batch[self.input_image_key] = rearrange(
                    data_batch[self.input_image_key], "b c h w -> b c 1 h w"
                ).contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def is_image_batch(self, data_batch: Dict[str, Tensor]) -> bool:
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, "Batch must contain either image or video data, not both or neither."
        return is_image

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Optional[int] = None
    ) -> VideoExtendCondition:
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype)
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.debug(
            f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        condition_video_indicator[:, :, :num_condition_t] += 1.0
        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator
        B, C, T, H, W = latent_state.shape
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"
        if condition.video_cond_bool:
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:
            condition.condition_video_input_mask = zeros_padding
        return condition

    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        add_input_frames_guidance: bool = False,
    ):
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
        augment_sigma = condition_augment_sigma
        latent = condition.gt_latent
        indicator = condition.condition_video_indicator
        if augment_sigma >= sigma:
            indicator = torch.zeros_like(indicator)
        noise = misc.arch_invariant_rand(latent.shape, torch.float32, self.tensor_kwargs["device"], seed)
        augment_latent = latent + noise * augment_sigma
        augment_latent = self.scheduler.precondition_inputs(augment_latent, augment_sigma)
        augment_latent_unscaled = self._reverse_precondition_input(augment_latent, sigma)
        if self.net.is_context_parallel_enabled:
            latent = split_inputs_cp(condition.gt_latent, seq_dim=2, cp_group=self.net.cp_group)
            indicator = split_inputs_cp(indicator, seq_dim=2, cp_group=self.net.cp_group)
            augment_latent_unscaled = split_inputs_cp(augment_latent_unscaled, seq_dim=2, cp_group=self.net.cp_group)
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
        latent_unscaled = latent / c_out - c_skip * xt
        return latent_unscaled

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed_inference: int = 1,
    ) -> Callable:
        assert condition_latent is not None, "condition_latent must be provided for video generation."
        condition, uncondition = self._get_conditions(
            data_batch,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            add_input_frames_guidance=add_input_frames_guidance,
        )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_xt, cond_latent, cond_indicator = self._augment_noise_with_latent(
                noise_x,
                sigma,
                condition,
                condition_augment_sigma=condition_video_augment_sigma_in_inference or 0.001,
                seed=seed_inference,
            )
            cond_pred = self.denoise(cond_xt, sigma, condition)
            cond_x0 = cond_pred.x0_pred_replaced if hasattr(cond_pred, "x0_pred_replaced") else cond_pred.x0
            uncond_xt, _, _ = self._augment_noise_with_latent(
                noise_x,
                sigma,
                uncondition,
                condition_augment_sigma=condition_video_augment_sigma_in_inference or 0.001,
                seed=seed_inference,
            )
            uncond_pred = self.denoise(uncond_xt, sigma, uncondition)
            uncond_x0 = uncond_pred.x0_pred_replaced if hasattr(uncond_pred, "x0_pred_replaced") else uncond_pred.x0
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        return_noise: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        self._normalize_video_databatch_inplace(data_batch)
        # self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            log.debug("image batch, call base model generate_samples_from_batch")
            return super().generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                is_negative_prompt=is_negative_prompt,
                num_steps=num_steps,
            )
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
                log.debug(f"Default Video state shape is used. {self.state_shape}")
                state_shape = self.state_shape
        assert condition_latent is not None, "condition_latent should be provided"
        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            seed_inference=seed,
        )
        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape), torch.float32, self.tensor_kwargs["device"], seed
            )
            * 80
        )
        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=80)
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
        if return_noise:
            if self.net.is_context_parallel_enabled:
                x_sigma_max = cat_outputs_cp(x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            return samples, x_sigma_max / 80
        return samples
