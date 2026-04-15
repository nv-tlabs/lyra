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

"""optimizer config options:

fused_adam - FusedAdamConfig
adamw - AdamWConfig
"""

import torch

from cosmos_predict1.utils import fused_adam
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.scheduler import WarmupCosineLR, WarmupLambdaLR

FusedAdamConfig: LazyDict = L(fused_adam.FusedAdam)(
    capturable=True,
    master_weights=True,
    adam_w_mode=True,
    params=PLACEHOLDER,
    lr=1e-4,
    betas=(0.5, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

AdamWConfig: LazyDict = L(torch.optim.AdamW)(
    params=PLACEHOLDER,
    lr=1e-4,
    betas=(0.5, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

WarmupLRConfig: LazyDict = L(WarmupLambdaLR)(optimizer=PLACEHOLDER, warmup=5000)

FusedAdamDiscConfig: LazyDict = L(fused_adam.FusedAdam)(
    capturable=True,
    master_weights=True,
    adam_w_mode=True,
    params=PLACEHOLDER,
    lr=4e-4,
    betas=(0.5, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

WarmupLRDiscConfig: LazyDict = L(WarmupLambdaLR)(optimizer=PLACEHOLDER, warmup=5000)

WarmupCosineLRConfig: LazyDict = L(WarmupCosineLR)(
    optimizer=PLACEHOLDER, warmup_iters=5000, lr_decay_iters=1000000, min_lr=1e-8
)
