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

from cosmos_predict1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.utils import log, misc


class DiffusionMultiviewT2WModel(DiffusionT2WModel):
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

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling.

        This function generates samples from either image or video data batches using diffusion sampling.
        It handles both conditional and unconditional generation with classifier-free guidance.

        Args:
            data_batch (dict): Raw data batch from the training data loader
            guidance (float, optional): Classifier-free guidance weight. Defaults to 1.5.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            state_shape (tuple | None, optional): Shape of the state tensor. Uses self.state_shape if None. Defaults to None.
            n_sample (int | None, optional): Number of samples to generate. Defaults to 1.
            is_negative_prompt (bool, optional): Whether to use negative prompt for unconditional generation. Defaults to False.
            num_steps (int, optional): Number of diffusion sampling steps. Defaults to 35.

        Returns:
            Tensor: Generated samples after diffusion sampling
        """
        condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)

        self.scheduler.set_timesteps(num_steps)

        xt = torch.randn(size=(n_sample,) + tuple(state_shape)) * self.scheduler.init_noise_sigma

        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = rearrange(xt, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)
            xt = rearrange(xt, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

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
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        return samples
