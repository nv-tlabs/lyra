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

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from diffusers import EDMEulerScheduler
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import BaseVideoCondition
from cosmos_predict1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_predict1.diffusion.module import parallel
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate


@dataclass
class VideoLatentDiffusionDecoderCondition(BaseVideoCondition):
    # latent_condition will concat to the input of network, along channel dim;
    # cfg will make latent_condition all zero padding.
    latent_condition: Optional[torch.Tensor] = None
    latent_condition_sigma: Optional[torch.Tensor] = None


class LatentDiffusionDecoderModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)
        """
        latent_corruptor: the corruption module is used to corrupt the latents. It add gaussian noise to the latents.
        pixel_corruptor: the corruption module is used to corrupt the pixels. It apply gaussian blur kernel to pixels in a temporal consistent way.
        tokenizer_corruptor: the corruption module is used to simulate tokenizer reconstruction errors.

        diffusion decoder noise augmentation pipeline for continuous token condition model:
        condition: GT_video [T, H, W]
                        -> tokenizer_corruptor~(8x8x8) encode -> latent_corruptor -> tokenizer_corruptor~(8x8x8) decode
                        -> pixel corruptor
                        -> tokenizer~(1x8x8) encode -> condition [T, H/8, W/8]
        GT: GT_video [T, H, W] -> tokenizer~(1x8x8) -> x_t [T, H/8, W/8].

        diffusion decoder noise augmentation pipeline for discrete token condition model:
        condition: GT_video [T, H, W]
                -> pixel corruptor
                -> discrete tokenizer encode -> condition [T, T/8, H/16, W/16]
        GT: GT_video [T, H, W] -> tokenizer~(8x8x8) -> x_t [T, T/8, H/8, W/8].

        """
        self.latent_corruptor = lazy_instantiate(config.latent_corruptor)
        self.pixel_corruptor = lazy_instantiate(config.pixel_corruptor)
        self.tokenizer_corruptor = lazy_instantiate(config.tokenizer_corruptor)

        if self.latent_corruptor:
            self.latent_corruptor.to(**self.tensor_kwargs)
        if self.pixel_corruptor:
            self.pixel_corruptor.to(**self.tensor_kwargs)

        if self.tokenizer_corruptor:
            if hasattr(self.tokenizer_corruptor, "reset_dtype"):
                self.tokenizer_corruptor.reset_dtype()
        else:
            assert self.pixel_corruptor is not None

        self.diffusion_decoder_cond_sigma_low = config.diffusion_decoder_cond_sigma_low
        self.diffusion_decoder_cond_sigma_high = config.diffusion_decoder_cond_sigma_high
        self.diffusion_decoder_corrupt_prob = config.diffusion_decoder_corrupt_prob
        if hasattr(config, "condition_on_tokenizer_corruptor_token"):
            self.condition_on_tokenizer_corruptor_token = config.condition_on_tokenizer_corruptor_token
        else:
            self.condition_on_tokenizer_corruptor_token = False

        self.scheduler = EDMEulerScheduler(sigma_max=80, sigma_min=0.02, sigma_data=self.sigma_data)

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        apply_corruptor: bool = False,
        corrupt_sigma: float = 0.01,
        preencode_condition: bool = False,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to self.state_shape if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            preencode_condition (bool): use pre-computed condition if true, save tokenizer's inference time memory/
        """
        if not preencode_condition:
            self._normalize_video_databatch_inplace(data_batch)
            self._augment_image_dim_inplace(data_batch)
        if n_sample is None:
            n_sample = data_batch[self.input_data_key].shape[0]

        condition, uncondition = self._get_conditions(
            data_batch,
            is_negative_prompt=is_negative_prompt,
            apply_corruptor=apply_corruptor,
            corrupt_sigma=corrupt_sigma,
            preencode_condition=preencode_condition,
        )

        self.scheduler.set_timesteps(num_steps)

        xt = torch.randn(size=(n_sample,) + tuple(state_shape)) * self.scheduler.init_noise_sigma

        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)

        for t in self.scheduler.timesteps:
            xt = xt.to(**self.tensor_kwargs)
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
            net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
            net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            # Compute the previous noisy sample x_t -> x_t-1
            xt = self.scheduler.step(net_output, t, xt).prev_sample
        samples = xt

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
        apply_corruptor: bool = True,
        corrupt_sigma: float = 1.5,
        preencode_condition: bool = False,
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
        self._add_latent_conditions_to_data_batch(
            data_batch,
            apply_corruptor=apply_corruptor,
            corrupt_sigma=corrupt_sigma,
            preencode_condition=preencode_condition,
        )

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # For inference, check if parallel_state is initialized
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            condition.latent_condition = split_inputs_cp(condition.latent_condition, seq_dim=2, cp_group=cp_group)
            condition.latent_condition_sigma = split_inputs_cp(
                condition.latent_condition_sigma, seq_dim=2, cp_group=cp_group
            )
            uncondition.latent_condition = split_inputs_cp(uncondition.latent_condition, seq_dim=2, cp_group=cp_group)
            uncondition.latent_condition_sigma = split_inputs_cp(
                uncondition.latent_condition_sigma, seq_dim=2, cp_group=cp_group
            )
        return condition, uncondition

    def _add_latent_conditions_to_data_batch(
        self,
        data_batch: dict,
        apply_corruptor: bool = True,
        corrupt_sigma: float = 1.5,
        preencode_condition: bool = False,
    ):
        # Latent state
        raw_state = data_batch[self.input_data_key]

        if self.condition_on_tokenizer_corruptor_token:
            if preencode_condition:
                latent_condition = raw_state.to(torch.int32).contiguous()
                corrupted_pixel = self.tokenizer_corruptor.decode(latent_condition[:, 0])
            else:
                corrupted_pixel = (
                    self.pixel_corruptor(raw_state) if apply_corruptor and self.pixel_corruptor else raw_state
                )
                latent_condition = self.tokenizer_corruptor.encode(corrupted_pixel)
                latent_condition = latent_condition[1] if isinstance(latent_condition, tuple) else latent_condition
                corrupted_pixel = self.tokenizer_corruptor.decode(latent_condition)
                latent_condition = latent_condition.unsqueeze(1)
        else:
            if preencode_condition:
                latent_condition = raw_state
                corrupted_pixel = self.decode(latent_condition)
            else:
                corrupted_pixel = (
                    self.pixel_corruptor(raw_state) if apply_corruptor and self.pixel_corruptor else raw_state
                )
                latent_condition = self.encode(corrupted_pixel).contiguous()

        sigma = (
            torch.rand((latent_condition.shape[0],)).to(**self.tensor_kwargs) * corrupt_sigma
        )  # small value to indicate clean video
        c_noise_cond = self.scheduler.precondition_noise(sigma=sigma)
        if corrupt_sigma != self.diffusion_decoder_cond_sigma_low and self.diffusion_decoder_corrupt_prob > 0:
            sigma_expand = sigma.view((-1,) + (1,) * (latent_condition.dim() - 1))
            noise = sigma_expand * torch.randn_like(latent_condition)
            latent_condition = latent_condition + noise
        data_batch["latent_condition_sigma"] = torch.ones_like(latent_condition[:, 0:1, ::]) * c_noise_cond
        data_batch["latent_condition"] = latent_condition


def broadcast_condition(condition: BaseVideoCondition, to_tp: bool = True, to_cp: bool = True) -> BaseVideoCondition:
    condition_kwargs = {}
    for k, v in condition.to_dict().items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"{k} requires gradient. the current impl does not support it"
        condition_kwargs[k] = parallel.broadcast(v, to_tp=to_tp, to_cp=to_cp)
    condition = type(condition)(**condition_kwargs)
    return condition
