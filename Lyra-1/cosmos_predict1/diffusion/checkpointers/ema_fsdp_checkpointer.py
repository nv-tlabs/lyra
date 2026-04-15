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

import warnings

import attrs

from cosmos_predict1.utils import log
from cosmos_predict1.utils.config import CheckpointConfig as BaseCheckpointConfig
from cosmos_predict1.utils.config import make_freezable
from cosmos_predict1.utils.fsdp_checkpointer import FSDPCheckpointer as BaseFSDPCheckpointer


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig(BaseCheckpointConfig):
    load_ema_to_reg: bool = False


class FSDPCheckpointer(BaseFSDPCheckpointer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.config_checkpoint, CheckpointConfig):
            warnings.warn(
                "The 'config_checkpoint' is not an instance of 'CheckpointConfig'. "
                "This behavior is deprecated and will not be supported in future versions. "
                "Please update 'config_checkpoint' to be of type 'CheckpointConfig'.",
                DeprecationWarning,
            )

            self.load_ema_to_reg = False
        else:
            self.load_ema_to_reg = self.config_checkpoint.load_ema_to_reg

        log.critical(f"load_ema_to_reg: {self.load_ema_to_reg}", rank0_only=False)

    def load_model_during_init(self, model, is_ema: bool = False, ema_id: int = 0):
        if self.load_ema_to_reg and is_ema is False:
            is_ema = True
            ema_id = 0
            log.critical("Loading EMA model to regular model during initialization.", rank0_only=False)
        super().load_model_during_init(model, is_ema, ema_id)
