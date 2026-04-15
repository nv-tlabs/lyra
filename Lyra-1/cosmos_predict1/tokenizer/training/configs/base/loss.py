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

"""Loss config options

Loss weights are scheduled using a piecewise linear LR schedule. The schedule is defined by a list of boundaries and values.

`boundaries` is a list of integers representing the iteration at which the weight value changes.
`values` is a list of floats representing the weight value at each boundary. It should have one more value than `boundaries`.

Example:
 A loss's weight will be:
    values[0] when step <= boundaries[0],
    values[1] when step > boundaries[0] and step <= boundaries[1],
    ..., and
    values[-1] when step > boundaries[-1].
"""
import attrs

from cosmos_predict1.tokenizer.training.losses import ReduceMode
from cosmos_predict1.tokenizer.training.losses.continuous import (
    ColorLoss,
    FlowLoss,
    KLLoss,
    PerceptualLoss,
    TokenizerLoss,
    VideoConsistencyLoss,
)
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class KLConfig:
    # each step is greater than boundaries[-1], so weight=values[-1]
    boundaries: list[int] = [0]
    values: list[float] = [1e-6]


@attrs.define(slots=False)
class PerceptualConfig:
    lpips_boundaries: list[int] = [500000]
    lpips_values: list[float] = [0.1, 0.073]
    # Layer weights for linearly combining the multi-layer vgg-based losses.
    layer_weights: list[float] = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
    # Gram loss, whether to turn on, and what weights to use.
    gram_enabled: bool = True
    gram_boundaries: list[int] = [500000]
    gram_values: list[float] = [0.0, 0.062]
    # Corr loss, whether to turn on, and what weights to use.
    corr_enabled: bool = False
    corr_boundaries: list[int] = [0]
    corr_values: list[float] = [0.0]
    # In the example training memory usage dropped from 64.03 GiB to 60.54 GiB
    # with checkpointing enabled for this loss for about 3.2% slowdown.
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
    checkpoint_activations: bool = False


@attrs.define(slots=False)
class ColorConfig:
    # Color (RGB) basic loss and its weight schedule.
    norm: str = "L1"
    boundaries: list[int] = [0]
    values: list[float] = [1.0]


@attrs.define(slots=False)
class FlowConfig:
    # Flow loss and its weight schedule.
    boundaries: list[int] = [250000]
    values: list[float] = [0.0, 0.01]
    scale: int = 2
    # Flow loss depends on RAFT, as such it requires a specific dtype.
    dtype: str = "bfloat16"
    # In the example training memory usage dropped from 28GB to 23GB
    # with checkpointing enabled for this loss
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
    checkpoint_activations: bool = False
    enabled: bool = False


@attrs.define(slots=False)
class VideoConsistencyConfig:
    # Add consistency loss between overlapped video frames
    boundaries: list[int] = [250000]
    values: list[float] = [0.0, 0.01]
    enabled: bool = False
    num_frames: int = 9
    step: int = 1


@attrs.define(slots=False)
class VideoLoss:
    # The combined loss function, and its reduction mode.
    color: LazyDict = L(ColorLoss)(config=ColorConfig())
    kl: LazyDict = L(KLLoss)(config=KLConfig())
    perceptual: LazyDict = L(PerceptualLoss)(config=PerceptualConfig())
    flow: LazyDict = L(FlowLoss)(config=FlowConfig())
    video_consistency: LazyDict = L(VideoConsistencyLoss)(config=VideoConsistencyConfig())
    reduce: str = ReduceMode.MEAN.value  # model.config.loss.config.reduce={'MEAN', 'SUM', 'SUM_PER_FRAME'}


VideoLossConfig: LazyDict = L(TokenizerLoss)(config=VideoLoss())
