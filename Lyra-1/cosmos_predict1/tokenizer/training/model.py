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

"""Implements the forward op for training, validation, and inference."""

from typing import Any

import torch

from cosmos_predict1.tokenizer.training.datasets.utils import IMAGE_KEY, INPUT_KEY, MASK_KEY, RECON_KEY, VIDEO_KEY
from cosmos_predict1.tokenizer.training.losses.continuous import RECON_CONSISTENCY_KEY, VIDEO_CONSISTENCY_LOSS
from cosmos_predict1.utils import ema
from cosmos_predict1.utils.lazy_config import LazyDict, instantiate
from cosmos_predict1.utils.model import Model

PREDICTION = "prediction"
EMA_PREDICTION = "ema_prediction"


class TokenizerModel(Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.network = instantiate(config.network)
        self.loss = instantiate(config.loss)
        self.metric = instantiate(config.metric)
        self.precision = getattr(torch, config.precision)
        if self.config.ema.enabled:
            self.ema = ema.EMAModelTracker(
                self,
                beta=self.config.ema.beta,
                torch_compile_buffer_renaming=self.config.ema.torch_compile_buffer_renaming,
            )
        self.init_input_keys()

    def init_input_keys(self):
        self.image_key = IMAGE_KEY
        self.video_key = VIDEO_KEY

    def get_input_key(self, data_batch: dict[str, torch.Tensor]) -> str:
        if self.image_key in data_batch:
            return self.image_key
        elif self.video_key in data_batch:
            return self.video_key
        else:
            raise ValueError("Input key not found in data_batch.")

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the network.

        Args:
            optimizer_config: The optimizer config for the net.
            scheduler_config: The scheduler config for the net.

        Returns:
            optimizer (torch.optim.Optimizer): The net optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The net optimization scheduler.
        """
        optimizer_config.params = self.network.parameters()
        optimizer = instantiate(optimizer_config)
        scheduler_config.optimizer = optimizer
        scheduler = instantiate(scheduler_config)

        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.ema.to(dtype=torch.float32)
        self.network = self.network.to(dtype=self.precision, memory_format=memory_format)
        self.loss = self.loss.to(dtype=self.precision, memory_format=memory_format)

    def state_dict(
        self, destination: dict[str, Any] = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        original_state_dict = super(TokenizerModel, self).state_dict(destination, prefix, keep_vars)

        # Filter out '.loss' and 'ema.loss-' keys from the state dict.
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if not k.startswith("loss.")}
        filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if not k.startswith("ema.loss-")}
        filtered_state_dict = {
            k: v for k, v in filtered_state_dict.items() if not k.startswith("network.encoder.patcher")
        }
        filtered_state_dict = {
            k: v for k, v in filtered_state_dict.items() if not k.startswith("network.decoder.unpatcher")
        }

        return filtered_state_dict

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> None:
        own_state = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in own_state}

        # Load only filtered state dict.
        super(TokenizerModel, self).load_state_dict(filtered_state_dict, strict=False)

        # If strict is True, ensure all parameters are loaded (except the excluded ones)
        missing_keys = set(own_state.keys()) - set(filtered_state_dict.keys())
        if missing_keys and strict:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")

    def _on_before_network_forward(self, data_batch: dict[str, torch.Tensor]) -> None:
        consistency_loss = self.loss.loss_modules[VIDEO_CONSISTENCY_LOSS]
        if hasattr(consistency_loss, "enabled") and consistency_loss.enabled:
            _input_key = self.get_input_key(data_batch)
            if _input_key is self.video_key:
                data_batch[_input_key] = consistency_loss.shuffle(data_batch[_input_key])
        return

    def _on_after_network_forward(
        self, data_batch: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor]
    ) -> None:
        consistency_loss = self.loss.loss_modules[VIDEO_CONSISTENCY_LOSS]
        if hasattr(consistency_loss, "enabled") and consistency_loss.enabled:
            _input_key = self.get_input_key(data_batch)
            if _input_key is self.video_key:
                data_batch[_input_key] = consistency_loss.unshuffle(data_batch[_input_key])
                output_batch[RECON_CONSISTENCY_KEY] = torch.ones_like(output_batch[RECON_KEY]) * output_batch[RECON_KEY]
                output_batch[RECON_KEY] = consistency_loss.unshuffle(output_batch[RECON_KEY])
        return

    def _network_forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # A callback proxy to modify the input before the forward pass.
        self._on_before_network_forward(data_batch)

        # Do the forward pass.
        tensor_batch = data_batch[self.get_input_key(data_batch)]
        output_batch = self.network(tensor_batch)
        output_batch = output_batch if self.network.training else output_batch._asdict()

        # A callback proxy to modify the output after the forward pass.
        self._on_after_network_forward(data_batch, output_batch)
        return output_batch

    def training_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images = data_batch[_input_key], output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        return dict({PREDICTION: recon_images, **loss_dict}), loss_value

    @torch.no_grad()
    def validation_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
        ema_model: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images = data_batch[_input_key], output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        metric_dict = self.metric(input_images, output_dict, iteration)
        loss_dict.update(metric_dict)
        prediction_key = EMA_PREDICTION if ema_model else PREDICTION
        return dict({prediction_key: recon_images, **loss_dict}), loss_value

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        return dict({PREDICTION: output_dict[RECON_KEY]})
