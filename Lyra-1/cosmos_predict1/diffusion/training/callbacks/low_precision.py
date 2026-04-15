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

import torch

from cosmos_predict1.diffusion.training.trainer import Trainer
from cosmos_predict1.utils.callback import LowPrecisionCallback as BaseCallback
from cosmos_predict1.utils.config import Config
from cosmos_predict1.utils.model import Model


class LowPrecisionCallback(BaseCallback):
    """
    Config with non-primitive type makes it difficult to override the option.
    The callback gets precision from model.precision instead.
    """

    def __init__(self, config: Config, trainer: Trainer, update_iter: int):
        self.config = config
        self.trainer = trainer
        self.update_iter = update_iter

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        assert model.precision in [
            torch.bfloat16,
            torch.float16,
            torch.half,
        ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision
