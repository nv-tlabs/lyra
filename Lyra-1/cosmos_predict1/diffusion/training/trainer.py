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

from cosmos_predict1.diffusion.training.utils.checkpointer import MultiRankCheckpointer
from cosmos_predict1.utils.fsdp_checkpointer import FSDPCheckpointer
from cosmos_predict1.utils.trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        if config.trainer.distributed_parallelism == "ddp":
            self.checkpointer = MultiRankCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        elif config.trainer.distributed_parallelism == "fsdp":
            self.checkpointer = FSDPCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            raise ValueError(f"Unsupported distributed parallelism: {config.trainer.distributed_parallelism}")
