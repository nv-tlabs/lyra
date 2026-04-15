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

import time

import torch
from torch import Tensor

from cosmos_predict1.diffusion.training.callbacks.every_n import EveryN
from cosmos_predict1.utils import log
from cosmos_predict1.utils.distributed import rank0_only
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer


class IterSpeed(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
    """

    def __init__(self, *args, hit_thres: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()

    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            log.info(
                f"Iteration {iteration}: "
                f"Hit counter: {self.hit_counter + 1}/{self.hit_thres} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s"
            )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @rank0_only
    def every_n_impl(
        self,
        trainer: Trainer,
        model: Model,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / self.every_n / self.step_size

        log.info(f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | Loss: {loss.item():.4f}")

        self.time = cur_time
