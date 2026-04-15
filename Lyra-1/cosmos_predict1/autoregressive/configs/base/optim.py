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

from cosmos_predict1.utils.lazy_config import LazyCall as L


class LambdaLinearWarmupScheduler:
    """
    A learning rate scheduler that implements linear warm-up and cool-down.

    This scheduler provides three phases:
    1. Warm-up: Learning rate linearly increases from 0 to 1.
    2. Constant: Learning rate remains at 1.
    3. Cool-down: Learning rate linearly decreases from 1 to 0.

    Args:
        warmup_steps (int): Number of steps for the warm-up phase.
        warmup_offset (int): Starts warmup from this offset.
        max_iter (int, optional): Total number of iterations. Required if cooldown_steps is provided.
        cooldown_steps (int, optional): Number of steps for the cool-down phase.

    Raises:
        ValueError: If cooldown_steps is provided without max_iter, or if an invalid step is given.
    """

    def __init__(self, warmup_steps: int, warmup_offset: int = 0, max_iter: int = None, cooldown_steps: int = None):
        self.warmup_steps = warmup_steps
        self.warmup_offset = warmup_offset
        self.max_iter = max_iter
        self.cooldown_steps = cooldown_steps

        if cooldown_steps is not None:
            if max_iter is None:
                raise ValueError("max_iter must be specified when cooldown_steps is provided")
            self.cooldown_start = max_iter - cooldown_steps
        else:
            self.cooldown_start = None

    def __call__(self, step):
        # Warm-up phase
        if step < self.warmup_offset:
            return 0

        if step < self.warmup_steps + self.warmup_offset:
            return float(step - self.warmup_offset) / float(max(1, self.warmup_steps))

        # Constant phase (no cool-down)
        elif self.cooldown_steps is None:
            return 1.0

        # Constant phase (before cool-down starts)
        elif step < self.cooldown_start:
            return 1.0

        # Cool-down phase
        elif self.cooldown_start <= step < self.max_iter:
            cooldown_progress = (step - self.cooldown_start) / self.cooldown_steps
            return 1.0 - cooldown_progress

        # After max_iter
        elif step >= self.max_iter:
            return 0.0

        # Unexpected case
        else:
            raise ValueError(f"Invalid step {step}")


LambdaLinearLR = L(torch.optim.lr_scheduler.LambdaLR)(
    optimizer=None,
    lr_lambda=L(LambdaLinearWarmupScheduler)(warmup_steps=5000),
)
