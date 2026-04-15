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

"""callbacks config options:

BASIC_CALLBACKS: always recommended to use
"""

from cosmos_predict1.tokenizer.training.callbacks import (
    AdaptCkptStateDict,
    ExpandLossMask,
    GradClipCallback,
    TorchCompile,
)
from cosmos_predict1.utils.callback import EMAModelCallback, LowPrecisionCallback, ProgressBarCallback
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L

BASIC_CALLBACKS = dict(
    low_precision=L(LowPrecisionCallback)(update_iter=1, config=PLACEHOLDER, trainer=PLACEHOLDER),
    grad_clip=L(GradClipCallback)(grad_clip_norm=1, verbose=False, config=PLACEHOLDER, trainer=PLACEHOLDER),
    ema=L(EMAModelCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER),
    progress_bar=L(ProgressBarCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER),
    expand_loss_mask=L(ExpandLossMask)(kernel_size=51, config=PLACEHOLDER, trainer=PLACEHOLDER),
    adapt_ckpt_state_dict=L(AdaptCkptStateDict)(config=PLACEHOLDER, trainer=PLACEHOLDER),
    torch_compile=L(TorchCompile)(
        compile_after_iterations=8,
        compile_network=False,
        compile_loss=False,
        compile_loss_keys=["flow", "perceptual"],
    ),
)
