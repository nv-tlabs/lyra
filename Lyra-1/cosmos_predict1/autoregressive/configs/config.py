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

"""Default config for cosmos_ar project."""

import os
from typing import Any, List

import attrs

from cosmos_predict1.autoregressive.configs.registry import register_configs
from cosmos_predict1.autoregressive.trainer import Trainer
from cosmos_predict1.utils import config, log
from cosmos_predict1.utils.config_helper import import_all_modules_from_package


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": None},
            {"data_train": "mock_video"},
            {"data_val": None},
            {"optimizer": "fused_adamw"},
            {"scheduler": "warmup_cosine_lr"},
            {"checkpoint": "local"},
            {"callbacks": "basic"},
            {"global_config": None},
            {"experiment": None},
        ]
    )

    def validate(self) -> None:
        """Validate that the config has all required fields."""
        assert self.job.project != "", "job.project is not set"
        assert self.job.group != "", "job.group is not set"
        assert self.job.name != "", "job.name is not set"
        log.info("Validating config for cosmos_autoregressive job")
        # FSDP config check
        if self.model.model_config.fsdp_enabled:
            assert self.trainer.distributed_parallelism == "fsdp"
        else:
            assert self.trainer.distributed_parallelism == "ddp"

        # Transformer Engine config check
        if self.model.model_config.backend == "transformer_engine":
            assert (
                "NVTE_FLASH_ATTN" in os.environ and os.environ["NVTE_FLASH_ATTN"] == "1"
            )  # Enable Flash attention for transformer engine

        # TP, CP config check
        if self.model_parallel is not None:
            if self.model_parallel.context_parallel_size > 1:
                assert (
                    self.model.model_config.backend == "transformer_engine"
                ), "Context parallelism is only supported in transformer engine."

            if self.model_parallel.tensor_model_parallel_size > 1:
                assert (
                    self.model.model_config.set_parallel_mode
                ), "Tensor model parallelism is only supported in parallel mode."

            if self.model_parallel.sequence_parallel:
                assert (
                    self.model_parallel.tensor_model_parallel_size > 1
                ), "Sequence parallelism is only supported in tensor model parallelism."
                assert (
                    self.model.model_config.backend == "transformer_engine"
                ), "Sequence parallelism is only supported in transformer engine."


def make_config():
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
        checkpoint=None,
    )

    c.job.project = "cosmos_autoregressive"
    c.job.group = "debug"
    c.job.name = "default_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.run_validation = True

    c.trainer.seed = 0
    c.trainer.max_iter = 10
    c.trainer.logging_iter = 1

    c.trainer.callbacks = None
    register_configs()
    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_predict1.autoregressive.configs.experiment")
    return c
