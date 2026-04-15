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

import torch
from torch._dynamo.eval_frame import OptimizedModule as torch_OptimizedModule

from cosmos_predict1.utils import callback, distributed, ema, log, misc
from cosmos_predict1.utils.checkpointer import Checkpointer
from cosmos_predict1.utils.config import CheckpointConfig, JobConfig
from cosmos_predict1.utils.model import Model


class TokenizerCheckpointer(Checkpointer):
    """The tokenizer checkpointer, extends the shared checkpointer.

    Supports checkpoint saving/loading to local disk:
        - network weights and training optimizer states.
        - optionally, export a TorchScript version of the EMA model.
    """

    def __init__(self, config_checkpoint: CheckpointConfig, config_job: JobConfig, callbacks: callback.CallBackGroup):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.callbacks = callbacks
        self.config_jit = config_checkpoint.jit

    def save(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = -1,
        **ignore_kwargs,
    ) -> None:
        """Saves network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (Model): The PyTorch model.
            optimizer: The model optimizer.
            scheduler: The optimization scheduler.
            grad_scaler: The gradient scaler (for mixed precision training).
            iteration: Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)
        model.eval()
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
                args=(state_dict, self._get_ema_jit(model), checkpoint_file, distributed.get_rank()),
            )
            self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @misc.timer("checkpoint saving (local)")
    def _save_worker_local(
        self,
        state_dict: dict[str, torch.Tensor],
        jit_models: dict[str, torch.ScriptModule],
        checkpoint_file: str,
        rank: int = 0,
    ) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict: The state dict of the model/optimizer/scheduler.
            ema_jit: A dict of TorchScript EMA model, representing the encoder, decoder and full model.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_local, checkpoint_file)
        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
        try:
            torch.save(state_dict, checkpoint_path)
            for key, jit_model in jit_models.items():
                checkpoint_jit = checkpoint_path.replace(".pt", f"_{key}.jit")
                torch.jit.save(jit_model, checkpoint_jit)
                log.success(f"Saved checkpoint: {checkpoint_jit}")
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            log.success(f"Saved checkpoint (local): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to save (local): {e}")

    def _get_ema_jit(self, model: Model) -> dict[str, torch.ScriptModule]:
        """Returns a TorchScript version of ema models compiled by PyTorch JIT."""
        if not self.config_jit.enabled:
            return dict()
        input_shape = tuple(self.config_jit.input_shape)
        example_input = torch.randn(input_shape)
        dtype = getattr(torch, self.config_jit.dtype)
        example_input = example_input.to(self.config_jit.device).to(dtype)
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            _model = model.network
            if isinstance(_model, torch_OptimizedModule):
                _model = _model._orig_mod

            # Make sure jit model output consistenly during consecutive calls
            # Check here: https://github.com/pytorch/pytorch/issues/74534
            torch._C._jit_set_texpr_fuser_enabled(False)

            ema_jit = torch.jit.trace(_model, example_input, strict=self.config_jit.strict)
            encoder_jit = torch.jit.trace(_model.encoder_jit(), example_input, strict=self.config_jit.strict)
            decoder_example = encoder_jit(example_input)
            if isinstance(decoder_example, tuple):
                decoder_example = decoder_example[0]
            else:
                assert isinstance(decoder_example, torch.Tensor), "decoder_example should be a tensor or tuple"
            decoder_jit = torch.jit.trace(_model.decoder_jit(), decoder_example, strict=self.config_jit.strict)
        return {"ema": ema_jit, "enc": encoder_jit, "dec": decoder_jit}
