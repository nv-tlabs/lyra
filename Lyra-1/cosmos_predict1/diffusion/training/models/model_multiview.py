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

import copy
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_predict1.diffusion.training.conditioner import DataType
from cosmos_predict1.diffusion.training.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.training.models.model import DiffusionModel, broadcast_condition
from cosmos_predict1.diffusion.training.models.model_image import CosmosCondition, diffusion_fsdp_class_decorator
from cosmos_predict1.utils import log, misc


class MultiviewDiffusionModel(DiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_views = config.n_views

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        encoded_state = self.vae.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        decoded_state = self.vae.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return decoded_state

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: CosmosCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        if self.is_image_batch(data_batch):
            # Turn off CP
            self.net.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    self.net.enable_context_parallel(cp_group)
                    log.debug("[CP] Split x0 and epsilon")

                    x0 = rearrange(x0, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                    epsilon = rearrange(epsilon, "B C (V T) H W -> (B V) C T H W", V=self.n_views)

                    x0 = split_inputs_cp(x=x0, seq_dim=2, cp_group=self.net.cp_group)
                    epsilon = split_inputs_cp(x=epsilon, seq_dim=2, cp_group=self.net.cp_group)

                    x0 = rearrange(x0, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
                    epsilon = rearrange(epsilon, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        output_batch, kendall_loss, pred_mse, edm_loss = super(
            DiffusionModel, self
        ).compute_loss_with_epsilon_and_sigma(data_batch, x0_from_data_batch, x0, condition, epsilon, sigma)
        if not self.is_image_batch(data_batch):
            if self.loss_reduce == "sum" and parallel_state.get_context_parallel_world_size() > 1:
                kendall_loss *= parallel_state.get_context_parallel_world_size()

        return output_batch, kendall_loss, pred_mse, edm_loss

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        guidance_other: Union[float, None] = None,
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
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
        x0_fn = self.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=is_negative_prompt, guidance_other=guidance_other
        )
        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.sde.sigma_max
        )
        if self.net.is_context_parallel_enabled:
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_views)

            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        samples = self.sampler(
            x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max, solver_option=solver_option
        )
        if self.net.is_context_parallel_enabled:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        return samples

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        guidance_other: Union[float, None] = None,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        if guidance_other is not None:
            # assume this is for inference time trajectory guidance for now
            assert not parallel_state.is_initialized(), "Parallel state not supported with two guidances."
            condition_other = copy.deepcopy(uncondition)
            condition_other.trajectory = condition.trajectory

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                cond_x0 = self.denoise(noise_x, sigma, condition).x0
                uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
                cond_other_x0 = self.denoise(noise_x, sigma, condition_other).x0

                raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0) + guidance_other * (cond_other_x0 - uncond_x0)

                if "guided_image" in data_batch:
                    assert False, "not supported"
                return raw_x0

        else:

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                cond_x0 = self.denoise(noise_x, sigma, condition).x0
                uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
                raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
                if "guided_image" in data_batch:
                    # replacement trick that enables inpainting with base model
                    assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                    guide_image = data_batch["guided_image"]
                    guide_mask = data_batch["guided_mask"]
                    raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
                return raw_x0

        return x0_fn


@diffusion_fsdp_class_decorator
class FSDPDiffusionModel(MultiviewDiffusionModel):
    pass
