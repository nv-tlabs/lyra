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

from typing import Any, List

import attrs

from cosmos_predict1.diffusion.training.config.base.model import DefaultModelConfig
from cosmos_predict1.diffusion.training.config.text2world.registry import (
    register_configs as register_configs_text2world,
)
from cosmos_predict1.diffusion.training.config.video2world.registry import (
    register_configs as register_configs_video2world,
)
from cosmos_predict1.diffusion.training.config.video2world_action.registry import (
    register_configs as register_configs_video2world_action,
)
from cosmos_predict1.diffusion.training.config.video2world_instruction.registry import (
    register_configs as register_configs_video2world_instruction,
)
from cosmos_predict1.diffusion.training.models.model import DiffusionModel
from cosmos_predict1.utils import config
from cosmos_predict1.utils.config_helper import import_all_modules_from_package
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.trainer import Trainer


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": None},
            {"net": None},
            {"conditioner": "add_fps_image_size_padding_mask"},
            {"fsdp": None},
            {"ema": "power"},
            {"vae": "vae1"},
            {"checkpoint": "pbss"},
            {"ckpt_klass": "fsdp"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(DiffusionModel)(
        config=PLACEHOLDER,
    )


def make_config():
    c = Config(
        model=DefaultModelConfig(),
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_predict1"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    c.checkpoint = None

    # Call this function to register config groups.
    register_configs_text2world()
    register_configs_video2world()
    register_configs_video2world_instruction()
    register_configs_video2world_action()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_predict1.diffusion.training.config.text2world", reload=True)
    import_all_modules_from_package("cosmos_predict1.diffusion.training.config.video2world", reload=True)
    import_all_modules_from_package("cosmos_predict1.diffusion.training.config.video2world_instruction", reload=True)
    import_all_modules_from_package("cosmos_predict1.diffusion.training.config.video2world_action", reload=True)

    return c
