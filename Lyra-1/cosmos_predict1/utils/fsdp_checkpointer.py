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

import gc
import os
import threading

import torch
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from cosmos_predict1.utils import callback, distributed, log, misc
from cosmos_predict1.utils.config import CheckpointConfig, JobConfig
from cosmos_predict1.utils.easy_io import easy_io
from cosmos_predict1.utils.fsdp_optim_fix import scatter_full_optim_state_dict
from cosmos_predict1.utils.model import Model


class FSDPCheckpointer:
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
        self.load_path = config_checkpoint.load_path
        self.load_training_state = config_checkpoint.load_training_state
        self.save_thread = None
        self.config_checkpoint = config_checkpoint

    def _load_ckpt_file_during_init(self):
        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = self.checkpoint_dir_local
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
            log.critical(f"[Checkpoint] Found latest checkpoint file: {latest_checkpoint_file}")
            log.critical(f"[Checkpoint] Loading from local path: {checkpoint_path}")
            log.critical("[Checkpoint] Will resume full training state (model, optimizer, scheduler)")
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                resume = self.load_training_state
                log.critical(f"[Checkpoint] Using specified checkpoint path: {checkpoint_path}")
                if resume:
                    log.critical("[Checkpoint] Will load complete training state (model, optimizer, scheduler)")
                else:
                    log.critical("[Checkpoint] Will load model weights only (no optimizer/scheduler state)")
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
                log.critical("[Checkpoint] No checkpoint path specified")
                log.critical("[Checkpoint] Starting fresh training with random initialization")
        return checkpoint_path, resume

    @misc.timer("FSDP.load_model_during_init")
    def load_model_during_init(self, model, is_ema=False, ema_id: int = 0):
        if ema_id > 0:
            assert is_ema, "ema_id should be used with is_ema=True"
        checkpoint_path, _ = self._load_ckpt_file_during_init()
        if checkpoint_path is not None:
            tag = "reg" if not is_ema else "ema"
            default_checkpoint_path = checkpoint_path.replace(".pt", f"_{tag}_model.pt")
            if not os.path.exists(default_checkpoint_path):
                default_checkpoint_path = checkpoint_path  # starting from the release checkpoint
                log.warning(f"is_ema={is_ema} model is not found. Loading from {default_checkpoint_path}")
            if tag == "ema" and ema_id > 0:
                _checkpoint_path = checkpoint_path.replace(".pt", f"_RANK{ema_id}.pt")
                _checkpoint_path = _checkpoint_path.replace(".pt", f"_{tag}_model.pt")
                if self._check_checkpoint_exists(_checkpoint_path, is_raise=False):
                    default_checkpoint_path = _checkpoint_path
                else:
                    print(
                        f"{distributed.get_rank()}: Checkpoint not found: {_checkpoint_path} "
                        f"(fallback to {default_checkpoint_path})"
                    )
            checkpoint_path = default_checkpoint_path
            self._check_checkpoint_exists(checkpoint_path)

            log.info(f"Loading checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            log.info("- Loading the model...")
            if self.strict_resume:
                log.info(model.load_state_dict(state_dict, strict=self.strict_resume))
            else:
                log.critical("\t Using non-strict model")
                from cosmos_predict1.diffusion.training.utils.checkpointer import non_strict_load_model

                log.info(non_strict_load_model(model, state_dict))
            log.info("-finish model loading")
        else:
            log.info(f"is_ema={is_ema} model is not found and loaded.")

    @misc.timer("FSDP.load_optim_scheduler_during_init")
    def load_optim_scheduler_during_init(self, fsdp_model, optimizer, scheduler):
        checkpoint_path, resume = self._load_ckpt_file_during_init()
        log.critical(f"Loading optimizer and scheduler: {checkpoint_path} (resume: {resume}")
        if checkpoint_path is not None:
            if resume:
                checkpoint_path = checkpoint_path.replace(".pt", "_optim.pt")
                self._check_checkpoint_exists(checkpoint_path)
                if distributed.get_rank() == 0:
                    log.info(f"Loading checkpoint (local): {checkpoint_path}")
                    state_dict = torch.load(
                        checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False
                    )
                    log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
                    log.info("- Loading the optimizer (FSDP scatter)...")
                else:
                    state_dict = {
                        "optimizer": None,
                        "scheduler": None,
                    }
                distributed.barrier()
                sharded_optimizer_state_dict = scatter_full_optim_state_dict(  # <---- FSDP
                    state_dict["optimizer"],
                    fsdp_model,
                )
                log.info("- Loading the optimizer (FSDP load_state_dict)...")
                log.info(optimizer.load_state_dict(sharded_optimizer_state_dict))
                log.critical("Skip loading the scheduler...")
                return
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])

    @misc.timer("FSDP get_optim_scheduler_state")
    def get_optim_scheduler_state(self, optim, fsdp_model, scheduler):
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            optim_statedict = FSDP.full_optim_state_dict(fsdp_model, optim)
        scheduler_statedict = scheduler.state_dict()
        return {
            "optimizer": optim_statedict,
            "scheduler": scheduler_statedict,
        }

    def save(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
        async_saving: bool = True,
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

        model_state_dict = model.state_dict_model()
        optim_scheduler_state_dict = self.get_optim_scheduler_state(optimizer, model.model, scheduler)
        torch.cuda.empty_cache()
        state_dict = dict(
            iteration=iteration,
        )
        self.callbacks.on_save_checkpoint(model, state_dict=state_dict)

        postfix, replicate_idx, shard_idx, total_ema_num = model.get_ckpt_postfix()
        if replicate_idx == 0 and shard_idx == 0:
            pass  # save whole; it is rank0
        elif replicate_idx < total_ema_num and shard_idx == 0:
            model_state_dict["model"] = None  # only save ema
            optim_scheduler_state_dict = None
            state_dict = None
        else:
            return

        checkpoint_file = f"iter_{iteration:09}{postfix}.pt"
        if async_saving:
            # Wait for previous saver thread to end.
            if self.save_thread:
                self.save_thread.join()
            # Run the checkpoint saver in a separate thread.
            self.save_thread = threading.Thread(
                target=self._save_worker_local,
                daemon=False,
                args=(
                    model_state_dict,
                    optim_scheduler_state_dict,
                    state_dict,
                    checkpoint_file,
                    distributed.get_rank(),
                ),
            )
            self.save_thread.start()
            log.info("checkpoint saving from an async thread")
        else:
            torch.cuda.empty_cache()
            # Run the checkpoint saver in the current thread.
            self._save_worker_local(
                model_state_dict, optim_scheduler_state_dict, state_dict, checkpoint_file, distributed.get_rank()
            )
            log.info("checkpoint saved within the main thread")
            del model_state_dict, optim_scheduler_state_dict, state_dict
            gc.collect()
            torch.cuda.empty_cache()
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @misc.timer("checkpoint saving (local)")
    def _save_worker_local(
        self,
        model_state_dict: dict[str, torch.Tensor],
        optim_scheduler_state_dict: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor],
        checkpoint_file: str,
        rank: int = 0,
    ) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_local, checkpoint_file)
        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
        try:
            model_state_dict, ema_model_state_dict = model_state_dict["model"], model_state_dict["ema"]
            if model_state_dict is not None:
                torch.save(model_state_dict, checkpoint_path.replace(".pt", "_reg_model.pt"))
            if ema_model_state_dict is not None:
                torch.save(ema_model_state_dict, checkpoint_path.replace(".pt", "_ema_model.pt"))
            if optim_scheduler_state_dict is not None:
                torch.save(optim_scheduler_state_dict, checkpoint_path.replace(".pt", "_optim.pt"))
            if state_dict is not None:
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
            model (FSDPDiffModle): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        del optimizer, grad_scaler
        checkpoint_path, resume = self._load_ckpt_file_during_init()
        iteration = 0
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            log.info(f"Loading checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
            log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            if resume:
                iteration = state_dict["iteration"]
            log.success("Done with loading the checkpoint.")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model)

        if scheduler is not None:
            scheduler.last_epoch = iteration
            log.critical(f"resume scheduler from {iteration}", rank0_only=False)

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
        if checkpoint_file is None:
            log.warning(f"Latest ckpt file not found: {latest_path}")
        else:
            log.info(f"Found latest checkpoint: {checkpoint_file}")
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

    def _check_checkpoint_exists(self, checkpoint_path: str, is_raise: bool = True) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            if is_raise:
                raise FileNotFoundError(f"File not found (local): {checkpoint_path}")
            return False
        return True

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()


class FSDPInferenceCheckpointer:
    def __init__(
        self,
        ckpt_path: str,
        strict_resume: bool = True,
    ):
        self.ckpt_path = ckpt_path
        self.strict_resume = strict_resume

    @misc.timer("FSDPInferenceCheckpointer.load_model_during_init")
    def load_model_during_init(self, model, is_ema=False, ema_id: int = 0):
        del ema_id
        if is_ema:
            log.warning("EMA model is not supported in inference mode.")
            return
        assert easy_io.exists(self.ckpt_path)
        log.info(f"Loading from {self.ckpt_path}")
        state_dict = torch.load(self.ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
        if self.strict_resume:
            log.info(model.load_state_dict(state_dict, strict=self.strict_resume))
        else:
            log.critical("\t Using non-strict model")
            from cosmos_predict1.diffusion.training.utils.checkpointer import non_strict_load_model

            log.info(non_strict_load_model(model, state_dict))
        log.info("-finish model loading")

    def load_optim_scheduler_during_init(self, *args, **kwargs):
        """
        We do not do load in inference mode. The function is here to maintain the same interface to avoid errors.
        """
        pass

    def save(self, *args, **kwargs):
        """
        We do not save anything in inference mode. The function is here to maintain the same interface to avoid errors.
        """
        pass

    def load(self, *args, **kwargs):
        """
        We do not do load in inference mode. The function is here to maintain the same interface to avoid errors.
        """
        return 0
