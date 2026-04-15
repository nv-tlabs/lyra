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

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from cosmos_predict1.diffusion.conditioner import GeneralConditioner
from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.training.context_parallel import split_inputs_cp
from cosmos_predict1.utils.misc import count_params


class DataType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._is_trainable = None
        self._dropout_rate = None
        self._input_key = None
        self._return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]):
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool):
        self._return_dict = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @dropout_rate.deleter
    def dropout_rate(self):
        del self._dropout_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @is_return_dict.deleter
    def is_return_dict(self):
        del self._return_dict

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])).type_as(in_tensor),
            in_tensor,
        )

    def details(self) -> str:
        return ""

    def summary(self) -> str:
        input_key = self.input_key if self.input_key is not None else getattr(self, "input_keys", None)
        return (
            f"{self.__class__.__name__} \n\tinput key: {input_key}"
            f"\n\tParam count: {count_params(self, False)} \n\tTrainable: {self.is_trainable}"
            f"\n\tDropout rate: {self.dropout_rate}"
            f"\n\t{self.details()}"
        )


class TrajectoryAttr(AbstractEmbModel):
    def __init__(self, traj_dim: int):
        super().__init__()
        self.traj_dim = traj_dim

    def forward(self, traj: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "trajectory": traj,
        }

    def details(self) -> str:
        return f"Traj dim : {self.traj_dim} \n\tOutput key: [trajectory]"


class FrameRepeatAttr(AbstractEmbModel):
    def __init__(self):
        super().__init__()

    def forward(self, frame_repeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "frame_repeat": frame_repeat / 10.0,
        }

    def details(self) -> str:
        return "Frame repeat, Output key: [frame_repeat]"


@dataclass
class BaseVideoCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    data_type: DataType = DataType.VIDEO
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    image_size: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None
    trajectory: Optional[torch.Tensor] = None
    frame_repeat: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class VideoExtendCondition(BaseVideoCondition):
    video_cond_bool: Optional[torch.Tensor] = None  # whether or not it conditioned on video
    gt_latent: Optional[torch.Tensor] = None
    condition_video_indicator: Optional[torch.Tensor] = None  # 1 for condition region

    # condition_video_input_mask will concat to the input of network, along channel dim;
    # Will be concat with the input tensor
    condition_video_input_mask: Optional[torch.Tensor] = None
    # condition_video_augment_sigma: (B, T) tensor of sigma value for the conditional input augmentation, only valid when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    condition_video_augment_sigma: Optional[torch.Tensor] = None
    # pose conditional input, will be concat with the input tensor
    condition_video_pose: Optional[torch.Tensor] = None


@dataclass
class VideoLatentDiffusionDecoderCondition(BaseVideoCondition):
    # latent_condition will concat to the input of network, along channel dim;
    # cfg will make latent_condition all zero padding.
    latent_condition: Optional[torch.Tensor] = None
    latent_condition_sigma: Optional[torch.Tensor] = None

    def get_condition_for_cp(self, cp_group):
        self.latent_condition = split_inputs_cp(x=self.latent_condition, seq_dim=2, cp_group=cp_group)
        self.latent_condition_sigma = split_inputs_cp(x=self.latent_condition_sigma, seq_dim=2, cp_group=cp_group)


class VideoConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> BaseVideoCondition:
        output = super()._forward(batch, override_dropout_rate)
        return BaseVideoCondition(**output)


class VideoDiffusionDecoderConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoLatentDiffusionDecoderCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoLatentDiffusionDecoderCondition(**output)


class VideoExtendConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoExtendCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoExtendCondition(**output)


class VideoConditionerWithTraingOnlyEmb(GeneralConditioner):
    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Processes the provided data batch to generate two sets of outputs: conditioned and unconditioned. This method
        manipulates the dropout rates of embedders to simulate two scenarios — one where all conditions are applied
        (conditioned), and one where they are removed or reduced to the minimum (unconditioned).

        This method first sets the dropout rates to zero for the conditioned scenario to fully apply the embedders' effects.
        For the unconditioned scenario, it sets the dropout rates to 1 (or to 0 if the initial unconditional dropout rate
        is insignificant) to minimize the embedders' influences, simulating an unconditioned generation.

        Parameters:
            data_batch (Dict): The input data batch that contains all necessary information for embedding processing. The
                            data is expected to match the required format and keys expected by the embedders.

        Returns:
            Tuple[Any, Any]: A tuple containing two condition:
                - The first one contains the outputs with all embedders fully applied (conditioned outputs).
                - The second one contains the outputs with embedders minimized or not applied (unconditioned outputs).
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            if isinstance(embedder, FrameRepeatAttr):
                cond_dropout_rates[emb_name] = 1.0
            else:
                cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)
        return condition, un_condition

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> BaseVideoCondition:
        output = super()._forward(batch, override_dropout_rate)
        return BaseVideoCondition(**output)


class VideoExtendConditionerWithTraingOnlyEmb(VideoConditionerWithTraingOnlyEmb):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoExtendCondition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoExtendCondition(**output)


@dataclass
class BaseWithCtrlCondition(VideoExtendCondition):
    control_input_canny: Optional[torch.Tensor] = None
    control_input_blur: Optional[torch.Tensor] = None
    control_input_canny_blur: Optional[torch.Tensor] = None
    control_input_depth: Optional[torch.Tensor] = None
    control_input_segmentation: Optional[torch.Tensor] = None
    control_input_depth_segmentation: Optional[torch.Tensor] = None
    control_input_mask: Optional[torch.Tensor] = None
    control_input_human_kpts: Optional[torch.Tensor] = None
    control_input_upscale: Optional[torch.Tensor] = None
    control_input_identity: Optional[torch.Tensor] = None
    control_input_multi: Optional[torch.Tensor] = None
    base_model: Optional[torch.nn.Module] = None
    hint_key: Optional[str] = None
    control_weight: Optional[float] = 1.0
    num_layers_to_use: Optional[int] = -1


class VideoConditionerWithCtrl(VideoExtendConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> BaseWithCtrlCondition:
        output = super()._forward(batch, override_dropout_rate)
        output["hint_key"] = batch["hint_key"]
        if "control_weight" in batch:
            output["control_weight"] = batch["control_weight"]
        if "num_layers_to_use" in batch:
            output["num_layers_to_use"] = batch["num_layers_to_use"]
        return BaseWithCtrlCondition(**output)


class BooleanFlag(AbstractEmbModel):
    def __init__(self, output_key: Optional[str] = None):
        super().__init__()
        self.output_key = output_key

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        del args, kwargs
        key = self.output_key if self.output_key else self.input_key
        return {key: self.flag}

    def random_dropout_input(
        self, in_tensor: torch.Tensor, dropout_rate: Optional[float] = None, key: Optional[str] = None
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        self.flag = torch.bernoulli((1.0 - dropout_rate) * torch.ones(1)).bool().to(device=in_tensor.device)
        return in_tensor

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\t This is a boolean flag"
