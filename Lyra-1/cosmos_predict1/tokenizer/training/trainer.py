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
import torch.utils.data

from cosmos_predict1.tokenizer.training.checkpointer import TokenizerCheckpointer
from cosmos_predict1.utils import ema, misc
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer


class TokenizerTrainer(Trainer):
    """The tokenizers traine, extended from Trainer.

    It extends model training functionality.

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        super(TokenizerTrainer, self).__init__(config)
        self.model_config = config.model.config
        self.checkpointer = TokenizerCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)

    @torch.no_grad()
    def validate(self, model: Model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (Model): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        # Evaluate on the full validation set.
        for val_iter, data_batch in enumerate(dataloader_val):
            if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                break
            data_batch = misc.to(data_batch, device="cuda")
            self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
            output_batch, _ = model.validation_step(data_batch, iteration)
            with ema.ema_scope(model, enabled=model.config.ema.enabled):
                ema_output_batch, loss = model.validation_step(data_batch, iteration, ema_model=True)
                output_batch.update(ema_output_batch)
            self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)
