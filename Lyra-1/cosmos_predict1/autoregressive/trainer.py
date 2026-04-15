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

import signal

import torch
import torch.distributed as dist
import torch.utils.data
from megatron.core import parallel_state

from cosmos_predict1.checkpointer.tp import Checkpointer as TensorParallelCheckpointer
from cosmos_predict1.utils import distributed, ema, log, misc
from cosmos_predict1.utils.checkpointer import Checkpointer
from cosmos_predict1.utils.fsdp_checkpointer import FSDPCheckpointer
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer


class Trainer(Trainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        if config.trainer.distributed_parallelism == "ddp":
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                self.checkpointer = TensorParallelCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)
                log.critical("Using Tensor Parallelism Checkpointer")
            else:
                self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)

        elif config.trainer.distributed_parallelism == "fsdp":
            self.checkpointer = FSDPCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            raise ValueError(f"Unsupported distributed parallelism: {config.trainer.distributed_parallelism}")

    """
    Modify the original trainer to log average loss (averaging across all devices and gradient accumulation)
    """

    def train(
        self,
        model: Model,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (Model): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        log.info(f"Model Architecture:\n {model}")
        model.on_train_start(self.config.trainer.memory_format)
        # Initialize the optimizer and scheduler.
        self.callbacks.on_optimizer_init_start()

        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)

        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        # Set the scheduler to the current iteration.
        scheduler.last_epoch = iteration
        scheduler._step_count = iteration + 1

        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        self.callbacks.on_before_dataloading(iteration)
        accumulated_loss = 0.0

        while True:
            dataloader_train_iter = iter(dataloader_train)
            while True:
                self.callbacks.on_before_dataloading(iteration)
                try:
                    data_batch = next(dataloader_train_iter)
                except StopIteration:
                    break
                self.callbacks.on_after_dataloading(iteration)
                # If max_iter is reached, exit the training loop.
                if iteration >= self.config.trainer.max_iter:
                    _end_training = True
                    break
                # Move all tensors in the data batch to GPU device.

                data_batch = misc.to(data_batch, device="cuda")
                # The actual training step.
                self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                model_ddp.train()
                output_batch, loss, grad_accum_iter = self.training_step(
                    model_ddp,
                    optimizer,
                    scheduler,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_iter=grad_accum_iter,
                )

                # Accumulate loss
                accumulated_loss += loss.detach()

                # If the gradients are still being accumulated, continue to load the next training batch.
                if grad_accum_iter != 0:
                    if self.enable_one_logger:
                        # Callback for skipped OneLoggerCallback.on_training_step_end()
                        self.one_logger.on_train_batch_end(set_barrier=False)
                    continue
                # Do the following when an actual optimizer (update) step has been made.
                iteration += 1

                # Average loss over accumulation steps
                grad_accum_avg_loss = accumulated_loss / self.config.trainer.grad_accum_iter
                # Average loss across all devices
                device_avg_loss = grad_accum_avg_loss.clone()
                dist.all_reduce(device_avg_loss, op=dist.ReduceOp.SUM)
                device_avg_loss /= dist.get_world_size()
                # Reset accumulation variables
                accumulated_loss = 0.0

                self.callbacks.on_training_step_end(
                    model, data_batch, output_batch, device_avg_loss, iteration=iteration
                )

                # self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)

                # Validation.
                if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                    self.validate(model, dataloader_val, iteration=iteration)
                # Save checkpoint.
                if iteration % self.config.checkpoint.save_iter == 0:
                    self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                # This iteration is successful; reset the timeout signal.
                signal.alarm(self.config.trainer.timeout_period)
            if _end_training:
                break
        log.success("Done with training.")
        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            with self.training_timer("forward"):
                output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                loss_scaled.backward()
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_after_backward()
                else:
                    model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                else:
                    model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

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
        with ema.ema_scope(model, enabled=getattr(model.config.ema, "enabled", False)):
            for val_iter, data_batch in enumerate(dataloader_val):
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)
