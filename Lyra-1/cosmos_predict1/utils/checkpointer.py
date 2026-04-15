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

import os
import threading
from typing import TYPE_CHECKING

import torch

from cosmos_predict1.utils import callback, distributed, log, misc
from cosmos_predict1.utils.model import Model

if TYPE_CHECKING:
    from cosmos_predict1.utils.config import CheckpointConfig, JobConfig


class Checkpointer:
    """The checkpointer class. Supports checkpoint saving/loading to local disk."""

    def __init__(self, config_checkpoint: CheckpointConfig, config_job: JobConfig, callbacks: callback.CallBackGroup):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        # Set the callback functions.
        self.callbacks = callbacks
        self.checkpoint_dir_local = f"{config_job.path_local}/checkpoints"
        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None

    def save(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (Model): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}.pt"

        if distributed.get_rank() == 0:
            state_dict = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            state_dict = misc.to(state_dict, device="cpu")
            self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
            # Wait for previous saver thread to end.
            if self.save_thread:
                self.save_thread.join()
            # Run the checkpoint saver in a separate thread.
            self.save_thread = threading.Thread(
                target=self._save_worker_local,
                daemon=False,
                args=(state_dict, checkpoint_file, distributed.get_rank()),
            )
            self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @misc.timer("checkpoint saving (local)")
    def _save_worker_local(self, state_dict: dict[str, torch.Tensor], checkpoint_file: str, rank: int = 0) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_local, checkpoint_file)
        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
        try:
            torch.save(state_dict, checkpoint_path)
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            log.success(f"Saved checkpoint (local): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to save (local): {e}")

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (Model): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = self.checkpoint_dir_local
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
            only_resume_scheduler = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                resume = self.load_training_state
                only_resume_scheduler = self.only_load_scheduler_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
                only_resume_scheduler = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            log.info(f"Loading checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            log.info("- Loading the model...")
            if "model" in state_dict:
                model.load_state_dict(state_dict["model"], strict=self.strict_resume)
            else:
                model.load_state_dict(state_dict, strict=self.strict_resume)
            if resume or only_resume_scheduler:
                iteration = state_dict["iteration"]
                assert scheduler
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
            else:
                iteration = 0
            if resume:
                assert optimizer
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                log.success(f"Done with loading the checkpoint (iteration {iteration}).")
            else:
                log.success("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model)

        return iteration

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
        if os.path.isfile(latest_path):
            checkpoint_file = open(latest_path).read().strip()
        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"File not found (local): {checkpoint_path}")

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()
