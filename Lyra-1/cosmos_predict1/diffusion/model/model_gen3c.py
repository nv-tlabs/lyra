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
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel, broadcast_condition


class DiffusionGen3CModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.frame_buffer_max = config.frame_buffer_max
        self.chunk_size = 121

    def encode_warped_frames(
        self, 
        condition_state: torch.Tensor, 
        condition_state_mask: torch.Tensor, 
        dtype: torch.dtype,
        ):

        assert condition_state.dim() == 6
        condition_state_mask = (condition_state_mask * 2 - 1).repeat(1, 1, 1, 3, 1, 1)
        latent_condition = []
        for i in range(condition_state.shape[2]):
            current_video_latent = self.encode(
                condition_state[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)
            ).contiguous()  # 1, 16, 8, 88, 160

            current_mask_latent = self.encode(
                condition_state_mask[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)
            ).contiguous()
            latent_condition.append(current_video_latent)
            latent_condition.append(current_mask_latent)
        for _ in range(self.frame_buffer_max - condition_state.shape[2]):
            latent_condition.append(torch.zeros_like(current_video_latent))
            latent_condition.append(torch.zeros_like(current_mask_latent))

        latent_condition = torch.cat(latent_condition, dim=1)
        return latent_condition

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

        # encode warped frames
        condition_state, condition_state_mask = (
            data_batch["condition_state"],
            data_batch["condition_state_mask"],
        )
        latent_condition = self.encode_warped_frames(
            condition_state, condition_state_mask, self.tensor_kwargs["dtype"]
        )

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )
        condition = self.add_condition_pose(latent_condition, condition)

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )
        uncondition = self.add_condition_pose(latent_condition, uncondition, drop_out_latent = True)
        assert condition.gt_latent.allclose(uncondition.gt_latent)

        # For inference, check if parallel_state is initialized
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def add_condition_pose(self, latent_condition: torch.Tensor, condition: VideoExtendCondition,
                           drop_out_latent: bool = False) -> VideoExtendCondition:
        """Add pose condition to the condition object. For camera control model
        Args:
            data_batch (Dict): data batch, with key "plucker_embeddings", in shape B,T,C,H,W
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        if drop_out_latent:
            condition.condition_video_pose = torch.zeros_like(latent_condition.contiguous())
        else:
            condition.condition_video_pose = latent_condition.contiguous()

        to_cp = self.net.is_context_parallel_enabled

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition
