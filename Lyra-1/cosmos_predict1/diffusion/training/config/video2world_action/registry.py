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
from typing import Dict, Optional

import attrs
import torch
from hydra.core.config_store import ConfigStore

from cosmos_predict1.diffusion.conditioner import VideoExtendCondition, VideoExtendConditioner
from cosmos_predict1.diffusion.config.base.conditioner import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    ReMapkey,
    TextConfig,
    VideoCondBoolConfig,
)
from cosmos_predict1.diffusion.training.config.video2world_action.experiment import register_experiments
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


@dataclass
class ActionConditionalVideoExtendCondition(VideoExtendCondition):
    action: Optional[torch.Tensor] = None


class ActionConditionalVideoExtendConditioner(VideoExtendConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> ActionConditionalVideoExtendCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "action" in batch, "ActionConditionalVideoExtendConditioner requires 'action' in batch"
        output["action"] = batch["action"]
        return ActionConditionalVideoExtendCondition(**output)


@attrs.define(slots=False)
class ActionConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `action`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="action", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "action"


ActionConditionalVideoExtendConditionerConfig: LazyDict = L(ActionConditionalVideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
    action=ActionConfig(),
)


def register_configs():
    cs = ConfigStore.instance()

    register_experiments(cs)

    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="action_conditional_video_cond",
        node=ActionConditionalVideoExtendConditionerConfig,
    )
