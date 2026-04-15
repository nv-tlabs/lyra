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

"""Default config for cosmos/tokenizer project."""

from typing import Any, List

import attrs

from cosmos_predict1.tokenizer.training.configs.base.model import DefaultModelConfig
from cosmos_predict1.tokenizer.training.configs.registry import register_configs
from cosmos_predict1.tokenizer.training.trainer import TokenizerTrainer
from cosmos_predict1.utils import config
from cosmos_predict1.utils.config_helper import import_all_modules_from_package


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock_video720"},
            {"data_val": "mock_video720"},
            {"optimizer": "fused_adam"},
            {"scheduler": "warmup"},
            {"network": "continuous_factorized_video"},
            {"loss": "video"},
            {"metric": "reconstruction"},
            {"checkpoint": "local"},
            {"callbacks": "basic"},
            {"experiment": None},
        ]
    )


def make_config():
    c = Config(
        model=DefaultModelConfig,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
        checkpoint=None,
    )
    c.job.project = "posttraining"
    c.job.group = "debug"
    c.job.name = "default_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = TokenizerTrainer
    c.trainer.run_validation = True

    c.trainer.seed = 1234
    c.trainer.max_iter = 10_000_000
    c.trainer.validation_iter = 5000
    c.trainer.max_val_iter = 1
    c.trainer.logging_iter = 100

    c.trainer.callbacks = None
    c.trainer.ddp.static_graph = True
    c.trainer.ddp.find_unused_parameters = False
    register_configs()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_predict1.tokenizer.training.configs.experiments")

    return c
