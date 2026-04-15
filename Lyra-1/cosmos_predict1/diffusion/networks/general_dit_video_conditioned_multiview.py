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

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torchvision import transforms

from cosmos_predict1.diffusion.conditioner import DataType
from cosmos_predict1.diffusion.module.parallel import split_inputs_cp
from cosmos_predict1.diffusion.networks.general_dit_multiview import MultiviewGeneralDIT
from cosmos_predict1.utils import log


class MultiviewVideoExtendGeneralDIT(MultiviewGeneralDIT):
    def __init__(self, *args, in_channels, **kwargs):
        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels + 1, **kwargs)
        log.info(f"VideoExtendGeneralDIT in_channels: {in_channels + 1}")

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args:
        condition_video_augment_sigma: (B) tensor of sigma value for the conditional input augmentation
        condition_video_pose: (B, 1, T, H, W) tensor of pose condition
        """
        B, C, T, H, W = x.shape

        if data_type == DataType.VIDEO:
            assert (
                condition_video_input_mask is not None
            ), "condition_video_input_mask is required for video data type; check if your model_obj is extend_model.FSDPDiffusionModel or the base DiffusionModel"
            if self.cp_group is not None:
                condition_video_input_mask = rearrange(
                    condition_video_input_mask, "B C (V T) H W -> B C V T H W", V=self.n_views
                )
                condition_video_input_mask = split_inputs_cp(
                    condition_video_input_mask, seq_dim=3, cp_group=self.cp_group
                )
                condition_video_input_mask = rearrange(
                    condition_video_input_mask, "B C V T H W -> B C (V T) H W", V=self.n_views
                )
            input_list = [x, condition_video_input_mask]
            if condition_video_pose is not None:
                if condition_video_pose.shape[2] > T:
                    log.warning(
                        f"condition_video_pose has more frames than the input video: {condition_video_pose.shape} > {x.shape}"
                    )
                    condition_video_pose = condition_video_pose[:, :, :T, :, :].contiguous()
                input_list.append(condition_video_pose)
            x = torch.cat(
                input_list,
                dim=1,
            )

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )
