# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Callable
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
from contextlib import contextmanager


class TrainTimeWeight:
    def __init__(
        self,
        noise_scheduler,
        weight: str = "uniform",
    ):
        self.weight = weight
        self.noise_scheduler = noise_scheduler
        if self.weight == "reweighting":
            x = self.noise_scheduler.timesteps.cuda()
            y = torch.exp(-2 * (( x - self.noise_scheduler.config.num_train_timesteps / 2) / self.noise_scheduler.config.num_train_timesteps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (self.noise_scheduler.config.num_train_timesteps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing.cpu()

    def __call__(
        self,
        t,
        tensor_kwargs
    ) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        elif self.weight == "reweighting":
            timestep_id = torch.argmin((self.noise_scheduler.timesteps.to(**tensor_kwargs) - t).abs())
            wts = self.linear_timesteps_weights[timestep_id]
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")

        return wts


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,

    ):
        self.distribution = distribution
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,)) * (self.max_timestep_boundary - self.min_timestep_boundary) + self.min_timestep_boundary
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype))  # .to(device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Time distribution '{self.dist}' is not implemented.")

        return t


class RectifiedFlow:
    def __init__(
        self,
        velocity_field: Callable,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
        train_time_weight_method: str = "uniform",
        use_dynamic_shift: bool = False,
        shift: int= 3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        r"""Initialize the RectifiedFlow class.

        Args:
            velocity_field (`Callable`):
                A function that predicts the velocity given the current state and time.
            train_time_distribution (`TrainTimeSampler` or `str`, *optional*, defaults to `"uniform"`):
                Distribution for sampling training times.
                Can be an instance of `TrainTimeSampler` or a string specifying the distribution type.
            train_time_weight (`TrainTimeWeight` or `str`, *optional*, defaults to `"uniform"`):
                Weight applied to training times.
                Can be an instance of `TrainTimeWeight` or a string specifying the weight type.
        """
        self.velocity_field = velocity_field
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution,max_timestep_boundary,min_timestep_boundary)
        )


        if use_dynamic_shift:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=use_dynamic_shift)
        else:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        self.train_time_weight = TrainTimeWeight(self.noise_scheduler, train_time_weight_method)
        self.use_t_in_reverse_order = True

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    @contextmanager
    def temporary_use_t_in_reverse_order(self, use_t_in_reverse_order: bool):
        """
        Context manager to temporarily set use_t_in_reverse_order value.
        
        Args:
            use_t_in_reverse_order (bool): The temporary value to set for use_t_in_reverse_order.
            
        Example:
            with rectified_flow.temporary_use_t_in_reverse_order(False):
                # use_t_in_reverse_order is temporarily set to False
                # ... do some operations ...
            # use_t_in_reverse_order is restored to its original value
        """
        original_value = self.use_t_in_reverse_order
        try:
            self.use_t_in_reverse_order = use_t_in_reverse_order
            yield
        finally:
            self.use_t_in_reverse_order = original_value

    def sample_train_time(self, batch_size: int):
        r"""This method calls the `TrainTimeSampler` to sample training times.

        Returns:
            t (`torch.Tensor`):
                A tensor of sampled training times with shape `(batch_size,)`,
                matching the class specified `device` and `dtype`.
        """
        time = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)
        return time

    def get_discrete_timestamp(self, u, tensor_kwargs):
        r"""This method map time from 0,1 to discrete steps
        """
        assert 0 <= u.min() and u.max() <= 1, "Time must be in [0, 1]"
        indices = (u.squeeze() * self.noise_scheduler.config.num_train_timesteps).long()
        if not self.use_t_in_reverse_order:
            indices = self.noise_scheduler.config.num_train_timesteps - indices - 1
        timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)[indices]
        return timesteps.unsqueeze(0)

    def get_sigmas(self,  timesteps, tensor_kwargs):

        sigmas = self.noise_scheduler.sigmas.to(**tensor_kwargs)
        schedule_timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)

        step_indices = [(schedule_timesteps == t).nonzero()  for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        return sigma


    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        sigmas: torch.Tensor | None,
        t: torch.Tensor | None = None,
    ):
        r"""
        This method computes interpolation `X_t` and their time derivatives `dotX_t` at the specified time points `t`.
        Note that `x_0` is the noise, and `x_1` is the clean data. This is aligned with the notation in the recified flow community,
        but different from the notation in the diffusion community.

        Args:
            x_0 (`torch.Tensor`):
                noise, shape `(B, D1, D2, ..., Dn)`, where `B` is the batch size, and `D1, D2, ..., Dn` are the data dimensions.
            x_1 (`torch.Tensor`):
                clean data, with the same shape as `x_0`
            sigmas (`torch.Tensor`):
                A tensor of sigmas, with shape `(B,)`, where each value is in `[0, 1]`.
            t (`torch.Tensor`):
                A tensor of time, with shape `(B,)`, where each value is in `[0, 1]`.

        Returns:
            (x_t, dot_x_t) (`Tuple[torch.Tensor, torch.Tensor]`):
                - x_t (`torch.Tensor`): The interpolated state, with shape `(B, D1, D2, ..., Dn)`.
                - dot_x_t (torch.Tensor): The time derivative of the interpolated state, with the same shape as `x_t`.
        """
        if sigmas is None:
            assert t is not None, "t must be provided when sigmas is None."
            timesteps = self.get_discrete_timestamp(t, {"device": self.device, "dtype": self.dtype})
            sigmas = self.get_sigmas(timesteps, {"device": self.device, "dtype": self.dtype})
        else:
            assert t is None, "t must be None when sigmas is provided."
            sigmas = sigmas.to(device=self.device, dtype=self.dtype)

        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of x_0 and x_1 must match."
        assert sigmas.shape[0] == x_1.shape[0], "Batch size of sigmas must match x_1."
        # Reshape t to match dimensions of x_1
        sigmas = sigmas.view(sigmas.shape[0], *([1] * (len(x_1.shape) - 1)))
        x_t = x_0 * sigmas  + x_1 * (1 - sigmas)
        dot_x_t = x_0 - x_1
        return x_t, dot_x_t

    def get_x0_from_flow_prediction(
        self,
        x_t: torch.Tensor,
        dot_x_t: torch.Tensor,
        t: torch.Tensor | None = None,
        sigmas: torch.Tensor | None = None,
    ):
        r"""
        Convert flow matching's prediction to x0 prediction.
        x_t: the input noisy data with shape [B, D1, D2, ..., Dn]
        dot_x_t: the prediction with shape [B, D1, D2, ..., Dn]
        t: the timestep with shape [B,], where each value is in [0, 1]
        sigmas: the sigmas with shape [B,], where each value is in [0, 1]

        pred = noise(x_0) - x_1
        x_t = (1-sigma_t) * x_1 + sigma_t * noise(x_0)
        we have x_1 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        assert t is not None or sigmas is not None, "Either t or sigmas must be provided."
        if sigmas is not None:
            assert t is None, "t and sigmas cannot be provided at the same time."
            sigmas = sigmas.to(device=self.device, dtype=self.dtype)
        else:
            timesteps = self.get_discrete_timestamp(t, {"device": self.device, "dtype": self.dtype})
            sigmas = self.get_sigmas(timesteps, {"device": self.device, "dtype": self.dtype})

        # Reshape t to match dimensions of x_1
        sigmas = sigmas.view(sigmas.shape[0], *([1] * (len(x_t.shape) - 1)))
        original_dtype = x_t.dtype
        x_t, dot_x_t, sigmas = map(lambda x: x.to(dtype=torch.float64), (x_t, dot_x_t, sigmas))
        x_1_pred = x_t - sigmas * dot_x_t
        x_1_pred = x_1_pred.to(dtype=original_dtype)
        return x_1_pred
