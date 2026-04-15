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

import numpy as np
import torch
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.training.conditioner import DataType, VideoExtendCondition
from cosmos_predict1.diffusion.training.models.extend_model import ExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model import DiffusionModel as BaseModel
from cosmos_predict1.diffusion.training.models.model import broadcast_condition
from cosmos_predict1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_predict1.utils import log


class InterpolatorDiffusionModel(ExtendDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.is_extend_model = True
        self.num_valid_latents = config.latent_shape[1] - config.num_latents_to_drop
        self.pixel_chunk_duration = config.vae.video_vae.pixel_chunk_duration
        self.input_image_key = getattr(self.config, "input_image_key", None)
        self.input_data_key = self.config.input_data_key

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], num_condition_t: Union[int, None] = None
    ) -> Tuple[Tensor, VideoExtendCondition]:
        raw_state, latent_state, condition = BaseModel.get_data_and_condition(self, data_batch)
        num_valid_frames = raw_state.shape[2] - self.pixel_chunk_duration + 1
        raw_state, latent_state = (
            raw_state[:, :, :num_valid_frames, ...],
            latent_state[:, :, : self.num_valid_latents, ...],
        )  # [B, C, T, H, W]
        raw_state, latent_state = raw_state.contiguous(), latent_state.contiguous()
        if condition.data_type == DataType.VIDEO:
            if self.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
                latent_state = self.sample_tokens_start_from_p_or_i(latent_state)
            condition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, condition, num_condition_t=1
            )
            if self.config.conditioner.video_cond_bool.add_pose_condition:
                condition = self.add_condition_pose(data_batch, condition)
        log.debug(f"condition.data_type {condition.data_type}")
        return raw_state, latent_state, condition

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
        if self.config.conditioner.video_cond_bool.condition_location == "first_n":
            # Only in inference to decide the condition region
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            log.info(
                f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
            )
            condition_video_indicator[:, :, :num_condition_t] += 1.0
        elif self.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
            # Should be used for both training and inference. The first and last frame will be condition frames.
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            log.info(
                f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
            )
            condition_video_indicator[:, :, :num_condition_t] += 1.0
            condition_video_indicator[:, :, -num_condition_t:] += 1.0
        elif self.config.conditioner.video_cond_bool.condition_location == "first_random_n":
            # Only in training
            num_condition_t_max = self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_max
            assert (
                num_condition_t_max <= T
            ), f"num_condition_t_max should be less than T, get {num_condition_t_max}, {T}"
            assert num_condition_t_max >= self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min
            num_condition_t = torch.randint(
                self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min,
                num_condition_t_max + 1,
                (1,),
            ).item()
            condition_video_indicator[:, :, :num_condition_t] += 1.0

        elif self.config.conditioner.video_cond_bool.condition_location == "random":
            # Only in training
            condition_rate = self.config.conditioner.video_cond_bool.random_conditon_rate
            flag = torch.ones(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype) * condition_rate
            condition_video_indicator = torch.bernoulli(flag).type(latent_dtype).to(latent_state.device)
        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
            )
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

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition


@diffusion_fsdp_class_decorator
class FSDPInterpolatorDiffusionModel(InterpolatorDiffusionModel):
    pass
