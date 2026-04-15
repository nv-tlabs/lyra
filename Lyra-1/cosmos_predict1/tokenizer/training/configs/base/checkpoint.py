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

"""checkpoints config options:

CHECKPOINT_LOCAL: store at local file system

"""
import attrs

from cosmos_predict1.utils import config
from cosmos_predict1.utils.config import make_freezable
from cosmos_predict1.utils.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class ExperimentConfig:
    # Enables enforcing experiment naming.
    enabled: bool = True
    # The project, e.g. edify_video4.
    project: str = None
    # The valid groups, e.g ["video"].
    groups: list[str] = None
    # The approved name prefixes, e.g. ["DV1024", "DI256"].
    name_prefixes: list[str] = None


@make_freezable
@attrs.define(slots=False)
class TokenizerCheckpointConfig(config.CheckpointConfig):
    # Experiment naming configs.
    experiment: ExperimentConfig = attrs.field(factory=ExperimentConfig)


jit_config = config.JITConfig(
    enabled=True,
    input_shape=[1, 3, 1024, 1024],
)

experiment_config = ExperimentConfig(
    enabled=True,
    project="cosmos_tokenizer",
    groups=["debug", "video"],
    name_prefixes=[
        f"{base}{size}" if base in ["CI", "DI"] else f"{base}{size}_Causal"
        for base in ["CI", "DI", "CV", "DV"]
        for size in [256, 320, 480, 512, 720, 1024, 1080]
    ]
    + [f"{base}{size}" for base in ["CV", "DV"] for size in [256, 320, 512, 720]]
    + ["mock"],
)

CHECKPOINT_LOCAL: LazyDict = attrs.asdict(
    TokenizerCheckpointConfig(
        save_iter=5000,
        jit=jit_config,
        experiment=experiment_config,
    )
)
