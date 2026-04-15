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

from typing import Dict, List, Optional

import attrs
import torch

from cosmos_predict1.diffusion.conditioner import BaseConditionEntry, TextAttr, VideoConditioner, VideoExtendConditioner
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class TextConfig:
    obj: LazyDict = L(TextAttr)()  # No arguments
    dropout_rate: float = 0.2
    input_keys: List[str] = attrs.field(factory=lambda: ["t5_text_embeddings", "t5_text_mask"])


class BooleanFlag(BaseConditionEntry):
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


class ReMapkey(BaseConditionEntry):
    def __init__(self, output_key: Optional[str] = None, dtype: Optional[str] = None):
        super().__init__()
        self.output_key = output_key
        self.dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
            "int": torch.int32,
            "long": torch.int64,
        }[dtype]

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = self.output_key if self.output_key else self.input_key
        if isinstance(element, torch.Tensor):
            element = element.to(dtype=self.dtype)
        return {key: element}


class FrameRepeatAttr(BaseConditionEntry):
    def __init__(self):
        super().__init__()

    def forward(self, frame_repeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "frame_repeat": frame_repeat / 10.0,
        }


@attrs.define(slots=False)
class FPSConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="fps", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "fps"


@attrs.define(slots=False)
class PaddingMaskConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `padding_mask`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="padding_mask", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "padding_mask"


@attrs.define(slots=False)
class ImageSizeConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `image_size`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="image_size", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "image_size"


@attrs.define(slots=False)
class NumFramesConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `num_frames`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="num_frames", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "num_frames"


@attrs.define(slots=False)
class FrameRepeatConfig:
    """
    Remap and process key from the input dictionary to the output dictionary. For `frame_repeat`.
    """

    obj: LazyDict = L(FrameRepeatAttr)()
    dropout_rate: float = 0.0
    input_key: str = "frame_repeat"


@attrs.define(slots=False)
class VideoCondBoolConfig:
    obj: LazyDict = L(BooleanFlag)(output_key="video_cond_bool")
    dropout_rate: float = 0.2
    input_key: str = "fps"  # This is a placeholder, we never use this value
    # Config below are for long video generation only
    compute_loss_for_condition_region: bool = False  # Compute loss for condition region

    # How to sample condition region during training. "first_random_n" set the first n frames to be condition region, n is random, "random" set the condition region to be random,
    condition_location: str = "first_random_n"
    random_conditon_rate: float = 0.5  # The rate to sample the condition region randomly
    first_random_n_num_condition_t_max: int = 4  # The maximum number of frames to sample as condition region, used when condition_location is "first_random_n"
    first_random_n_num_condition_t_min: int = 0  # The minimum number of frames to sample as condition region, used when condition_location is "first_random_n"

    # How to dropout value of the conditional input frames
    cfg_unconditional_type: str = "zero_condition_region_condition_mask"  # Unconditional type. "zero_condition_region_condition_mask" set the input to zero for condition region, "noise_x_condition_region" set the input to x_t, same as the base model

    # How to corrupt the condition region
    apply_corruption_to_condition_region: str = "noise_with_sigma"  # Apply corruption to condition region, option: "gaussian_blur", "noise_with_sigma", "clean" (inference), "noise_with_sigma_fixed" (inference)
    # Inference only option: list of sigma value for the corruption at different chunk id, used when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    apply_corruption_to_condition_region_sigma_value: list[float] = [0.001, 0.2] + [
        0.5
    ] * 10  # Sigma value for the corruption, used when apply_corruption_to_condition_region is "noise_with_sigma_fixed"

    # Add augment_sigma condition to the network
    condition_on_augment_sigma: bool = False
    # The following arguments is to match with previous implementation where we use train sde to sample augment sigma (with adjust video noise turn on)
    augment_sigma_sample_p_mean: float = 0.0  # Mean of the augment sigma
    augment_sigma_sample_p_std: float = 1.0  # Std of the augment sigma
    augment_sigma_sample_multiplier: float = 4.0  # Multipler of augment sigma

    # Add pose condition to the network
    add_pose_condition: bool = False

    # Sample PPP... from IPPP... sequence
    sample_tokens_start_from_p_or_i: bool = False

    # Normalize the input condition latent
    normalize_condition_latent: bool = False


@attrs.define(slots=False)
class LatentConditionConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `latent condition`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="latent_condition", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "latent_condition"


@attrs.define(slots=False)
class LatentConditionSigmaConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `latent condition`.
    """

    obj: LazyDict = L(ReMapkey)(output_key="latent_condition_sigma", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "latent_condition_sigma"


BaseVideoConditionerConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
)

VideoConditionerFpsSizePaddingConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
)

VideoExtendConditionerConfig: LazyDict = L(VideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
)

VideoConditionerFpsSizePaddingFrameRepeatConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    frame_repeat=FrameRepeatConfig(),
)

VideoExtendConditionerFrameRepeatConfig: LazyDict = L(VideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
    frame_repeat=FrameRepeatConfig(),
)
