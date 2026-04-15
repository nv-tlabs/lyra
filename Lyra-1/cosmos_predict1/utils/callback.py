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

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

import omegaconf
import torch
import torch.utils.data
import tqdm

from cosmos_predict1.utils import distributed, log
from cosmos_predict1.utils.lazy_config import instantiate
from cosmos_predict1.utils.misc import get_local_tensor_if_DTensor

if TYPE_CHECKING:
    from cosmos_predict1.utils.config import Config
    from cosmos_predict1.utils.model import Model
    from cosmos_predict1.utils.trainer import Trainer


class CallBackGroup:
    """A class for hosting a collection of callback objects.

    It is used to execute callback functions of multiple callback objects with the same method name.
    When callbackgroup.func(args) is executed, internally it loops through the objects in self._callbacks and runs
    self._callbacks[0].func(args), self._callbacks[1].func(args), etc. The method name and arguments should match.

    Attributes:
        _callbacks (list[Callback]): List of callback objects.
    """

    def __init__(self, config: Config, trainer: Trainer) -> None:
        """Initializes the list of callback objects.

        Args:
            config (Config): The config object for the codebase.
            trainer (Trainer): The main trainer.
        """
        self._callbacks = []
        callback_configs = config.trainer.callbacks
        if callback_configs:
            if isinstance(callback_configs, list) or isinstance(callback_configs, omegaconf.listconfig.ListConfig):
                warnings.warn(
                    "The 'config.trainer.callbacks' parameter should be a dict instead of a list. "
                    "Please update your code",
                    DeprecationWarning,
                    stacklevel=2,
                )
                callback_configs = {f"callback_{i}": v for i, v in enumerate(callback_configs)}
            for callback_name, current_callback_cfg in callback_configs.items():
                if "_target_" not in current_callback_cfg:
                    log.critical(
                        f"Callback {callback_name} is missing the '_target_' field. \n SKip {current_callback_cfg}"
                    )
                    continue
                log.critical(f"Instantiating callback {callback_name}: {current_callback_cfg}")
                _callback = instantiate(current_callback_cfg)
                assert isinstance(_callback, Callback), f"{current_callback_cfg} is not a valid callback."
                _callback.config = config
                _callback.trainer = trainer
                self._callbacks.append(_callback)

    def __getattr__(self, method_name: str) -> Callable:
        """Loops through the callback objects to call the corresponding callback function.

        Args:
            method_name (str): Callback method name.
        """

        def multi_callback_wrapper(*args, **kwargs) -> None:
            for callback in self._callbacks:
                assert hasattr(callback, method_name)
                method = getattr(callback, method_name)
                assert callable(method)
                _ = method(*args, **kwargs)

        return multi_callback_wrapper


class Callback:
    """The base class for all callbacks.

    All callbacks should inherit from this class and adhere to the established method names and signatures.
    """

    def __init__(self, config: Optional["Config"] = None, trainer: Optional["Trainer"] = None):
        """Initializes a Callback object.

        Args:
            config (Optional[Config]): The configuration object for the codebase, if available.
            trainer (Optional[Trainer]): The main trainer handling the training loop, if available.

        Notes:
            The config and trainer parameters are optional to maintain backward compatibility.
            In future releases, these parameters will be removed. Upon using these parameters, a deprecation
            warning will be issued.

        """
        if config is not None or trainer is not None:
            warnings.warn(
                "The 'config' and 'trainer' parameters are deprecated and will be removed in a future release. "
                "Please update your code to create Callback instances without these parameters.",
                DeprecationWarning,
                stacklevel=2,
            )
        del config, trainer

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        pass

    def on_training_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        pass

    def on_before_forward(self, iteration: int = 0) -> None:
        pass

    def on_after_forward(self, iteration: int = 0) -> None:
        pass

    def on_before_backward(
        self, model_ddp: distributed.DistributedDataParallel, loss: torch.Tensor, iteration: int = 0
    ) -> None:
        pass

    def on_after_backward(self, model_ddp: distributed.DistributedDataParallel, iteration: int = 0) -> None:
        pass

    def on_before_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_after_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_optimizer_init_start(self) -> None:
        pass

    def on_optimizer_init_end(self) -> None:
        pass

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_start(
        self, model: Model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        pass

    def on_validation_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_end(self, model: Model, iteration: int = 0) -> None:
        pass

    def on_load_checkpoint_start(self, model: Model) -> None:
        pass

    def on_load_checkpoint_end(self, model: Model) -> None:
        pass

    def on_load_checkpoint(self, model: Model, state_dict: dict[Any]) -> None:
        pass

    def on_save_checkpoint_start(self, model: Model, iteration: int = 0) -> None:
        pass

    def on_save_checkpoint_end(self, model: Model, iteration: int = 0) -> None:
        pass

    def on_save_checkpoint_success(self, iteration: int = 0) -> None:
        pass

    def on_save_checkpoint(self, model: Model, state_dict: dict[Any]) -> None:
        pass

    def on_train_end(self, model: Model, iteration: int = 0) -> None:
        pass

    def on_app_end(self) -> None:
        pass


class EMAModelCallback(Callback):
    """The callback class for tracking EMA model weights."""

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        # Set up the EMA model weight tracker.
        if model.config.ema.enabled:
            assert hasattr(model, "ema"), "EMA should be initialized from Model"
            # EMA model must be kept in FP32 precision.
            model.ema = model.ema.to(dtype=torch.float32)
        else:
            assert not hasattr(model, "ema"), "There should be no EMA initialized."

    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Update the EMA model with the new regular weights.
        if model.config.ema.enabled:
            model.ema.update_average(model, iteration)


class ProgressBarCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.train_pbar.update()

    @distributed.rank0_only
    def on_validation_start(
        self, model: Model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        if self.config.trainer.max_val_iter is not None:
            num_iter = self.config.trainer.max_val_iter
        else:
            num_iter = len(dataloader_val)
        assert num_iter is not None and num_iter > 0, f"Invalid number of validation iterations: {num_iter}"
        self.val_pbar = tqdm.trange(num_iter, desc="Validating", position=1, leave=False)

    @distributed.rank0_only
    def on_validation_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.val_pbar.update()

    @distributed.rank0_only
    def on_validation_end(self, model: Model, iteration: int = 0) -> None:
        self.val_pbar.close()

    @distributed.rank0_only
    def on_train_end(self, model: Model, iteration: int = 0) -> None:
        self.trainer.checkpointer.finalize()
        self.train_pbar.close()


class IterationLoggerCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        # self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")
        self.start_iteration_time = time.time()
        self.elapsed_iteration_time = 0

    @distributed.rank0_only
    def on_training_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        self.start_iteration_time = time.time()

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.elapsed_iteration_time += time.time() - self.start_iteration_time

        if iteration % self.config.trainer.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / self.config.trainer.logging_iter
            log.info(f"Iteration: {iteration}, average iter time: {avg_time:2f}, total loss {loss.item():4f}")

            self.elapsed_iteration_time = 0


class GradClipCallback(Callback):
    """The callback class for gradient clipping."""

    def __init__(
        self,
        config: Optional["Config"] = None,
        trainer: Optional["Trainer"] = None,
        grad_clip_norm: float = 1.0,
    ):
        super().__init__(config, trainer)
        self.grad_clip_norm = grad_clip_norm

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_ddp.module.parameters(), max_norm=self.grad_clip_norm)


class LowPrecisionCallback(Callback):
    """The callback class handling low precision training"""

    def __init__(self, update_iter: int, config: Optional["Config"] = None, trainer: Optional["Trainer"] = None):
        super().__init__(config, trainer)
        self.update_iter = update_iter

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        assert model.precision in [
            torch.bfloat16,
            torch.float16,
            torch.half,
        ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_validation_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if iteration % self.update_iter == 0:
            if getattr(optimizer, "master_weights", False):
                params, master_params = [], []
                for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
                    for p, p_master in zip(group["params"], group_master["params"]):
                        params.append(get_local_tensor_if_DTensor(p.data))
                        master_params.append(p_master.data)
                torch._foreach_copy_(params, master_params)
