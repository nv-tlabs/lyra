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

from dataclasses import dataclass
from typing import Tuple

import torch
from megatron.core import parallel_state
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from cosmos_predict1.utils import distributed
from cosmos_predict1.utils.callbacks.grad_clip import GradClip as GradClipImage
from cosmos_predict1.utils.callbacks.grad_clip import _fused_nan_to_num
from cosmos_predict1.utils.model import Model


@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        self.state += cur_state
        self.iter_count += 1

    def get_stat(self) -> Tuple[float, float]:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
            avg_state = avg_state.item()
        else:
            avg_state = 0
        self.reset()
        return avg_state


class GradClip(GradClipImage):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_mag_log = _MagnitudeRecord()
        self.video_mag_log = _MagnitudeRecord()
        self._cur_state = None

    def on_training_step_start(self, model: Model, data_batch: dict[str, torch.Tensor], iteration: int = 0) -> None:
        if model.is_image_batch(data_batch):
            self._cur_state = self.img_mag_log
        else:
            self._cur_state = self.video_mag_log

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del optimizer, scheduler
        if isinstance(model_ddp, distributed.DistributedDataParallel):
            model = model_ddp.module
        else:
            model = model_ddp
        params = []
        if self.model_key is not None:
            items = self.model_key.split(".")
            for item in items:
                model = getattr(model, item)
            if self.force_finite:
                for param in model.parameters():
                    if param.grad is not None:
                        params.append(param.grad)
                        # torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
                _fused_nan_to_num(params)

            if isinstance(model, FSDP) and self.fsdp_enabled:
                total_norm = model.clip_grad_norm_(self.clip_norm)
            else:
                if parallel_state.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
                    total_norm = model_ddp.module.clip_grad_norm_(self.clip_norm)
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm, foreach=True)

            self._cur_state.update(total_norm)
