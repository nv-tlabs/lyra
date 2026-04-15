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

"""Tokenizer callbacks extended from base callbacks."""

import math
from typing import Any, Optional

import numpy as np
import torch
from torch._dynamo.eval_frame import OptimizedModule as torch_OptimizedModule

from cosmos_predict1.utils import callback, distributed, log
from cosmos_predict1.utils.config import Config
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_VIDEO_CONSISTENCY_LOSS = "video_consistency"


def make_video_grid(video, nrow=None, padding=1):
    r"""Make a grid of videos for visualization.
    Args:
        video (tensor): video of size B x C x T x H x W.
        nrow (int): number of rows in the grid.
        padding (int): size of paddings between videos.
    """
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().detach().numpy() * _UINT8_MAX_F).astype("uint8")
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    video_grid = np.zeros((t, (padding + h) * nrow + padding, (padding + w) * ncol + padding, c), dtype="uint8")

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r : start_r + h, start_c : start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    return video


def compute_weight_norm(model):
    weight_norm = dict()
    for layer_name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(f"[weight] {layer_name} NaN detected in gradients")
        weight_norm[f"{layer_name}"] = torch.norm(param, p=2).item()
    return weight_norm


def compute_grad_norm(model):
    grad_norm = dict()
    for layer_name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise ValueError(f"[grad] {layer_name} NaN detected in gradients")
            grad_norm[f"{layer_name}"] = torch.norm(param.grad, p=2).item()
    return grad_norm


class AdaptCkptStateDict(callback.Callback):
    def __init__(self, config: Config, trainer: Trainer):
        super().__init__(config, trainer)

    def on_save_checkpoint(self, model: Model, state_dict: dict[Any, Any]) -> None:
        """Adapt the state dict should the model be a compiled one."""
        if not isinstance(model.network, torch_OptimizedModule):
            return

        def _uncompiled_key(k):
            if k.startswith("network._orig_mod"):
                return k.replace("network._orig_mod", "network")
            elif k.startswith("ema.network-_orig_mod"):
                return k.replace("ema.network-_orig_mod", "ema.network")
            return k

        fixed_keys_state_dict = {}

        for k, v in state_dict["model"].items():
            fixed_keys_state_dict[_uncompiled_key(k)] = v

        state_dict["model"] = fixed_keys_state_dict

    def on_load_checkpoint(self, model: Model, state_dict: dict[Any, Any]) -> None:
        """Adapt the state dict should the model be a compiled one."""
        if not isinstance(model.network, torch_OptimizedModule):
            return

        def _compiled_key(k):
            if k.startswith("network."):
                return k.replace("network", "network._orig_mod")
            elif k.startswith("ema.network-"):
                return k.replace("ema.network", "ema.network-_orig_mod")
            return k

        fixed_keys_state_dict = {}

        for k, v in state_dict["model"].items():
            fixed_keys_state_dict[_compiled_key(k)] = v

        state_dict["model"] = fixed_keys_state_dict


class GradClipCallback(callback.GradClipCallback):
    """The verbose tokenizer callback for gradient clipping."""

    def __init__(self, grad_clip_norm: float, config: Config, trainer: Trainer, verbose: bool):
        super().__init__(config, trainer, grad_clip_norm)
        self.verbose = verbose

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        grad_scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model_ddp.module.parameters(), max_norm=self.grad_clip_norm)
        if torch.isnan(total_norm):
            raise ValueError("[gradient clipping] NaN detected in gradient norms")
        if torch.isfinite(total_norm) and total_norm > self.grad_clip_norm and self.verbose:
            if model_ddp.module.network.training:
                log.warning(
                    f"[net:{iteration:07d}] Gradient norm {total_norm} > {self.grad_clip_norm}. Clipping gradients."
                )
            else:
                log.warning(
                    f"[unknown:{iteration:07d}] Gradient norm {total_norm} > {self.grad_clip_norm}. Clipping gradients."
                )


class ExpandLossMask(callback.Callback):
    def __init__(self, kernel_size: int, config: Config, trainer: Trainer):
        super().__init__(config, trainer)
        self.kernel_size = kernel_size

    def on_training_step_start(self, model: Model, data: dict[str, Any], iteration: int = 0) -> None:
        """Expand loss_mask with max pooling (to cover some partial human regions)"""

        if "loss_mask" not in data.keys():
            return

        assert data["loss_mask"].ndim == 4 or data["loss_mask"].ndim == 5, "ndim of loss_mask must be 4 or 5"

        kernel_size = self.kernel_size
        if data["loss_mask"].ndim == 4:
            data["loss_mask"] = torch.nn.functional.max_pool2d(
                data["loss_mask"], kernel_size, stride=1, padding=kernel_size // 2
            )
        else:
            data["loss_mask"] = torch.nn.functional.max_pool3d(
                data["loss_mask"],
                (1, kernel_size, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2, kernel_size // 2),
            )


class TorchCompile(callback.Callback):
    """
    Callback to use torch.compile() on network or modules in losses(FlowLoss and PerceptualLoss) or both.
    We compile them at later iteration as it prevents NCCL timeouts when times are very unstable during first iterations
    """

    _TORCH_DYNAMO_CACHE_SIZE = 128

    def __init__(
        self,
        compile_after_iterations: int = 8,
        compile_network: bool = False,
        compile_loss: bool = False,
        compile_loss_keys: list[str] = ["flow", "perceptual"],
    ):
        self.initial_iteration: Optional[int] = None
        self.compile_after_iterations: int = compile_after_iterations

        self.compile_network: bool = compile_network
        self.compile_loss: bool = compile_loss

        self.compile_loss_keys: list[str] = compile_loss_keys

        if self.compile_network or self.compile_loss:
            torch._dynamo.config.cache_size_limit = TorchCompile._TORCH_DYNAMO_CACHE_SIZE

            # Hack to make ".training" work on "torch.compile()" module.
            # Value of ".training" is incorrectly set on torch.compile() module, when .eval() or .train()
            # is invoked, but is correctly set on original module and this hack accesses that value
            # I've created issue about this: https://github.com/pytorch/pytorch/issues/132986
            torch_OptimizedModule.training = property(
                lambda self: self._orig_mod.training, lambda self, value: None, lambda self: None
            )

    def on_training_step_start(self, model: Model, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        if not (self.compile_network or self.compile_loss):
            return

        if self.initial_iteration is None:
            log.info(f"Compilation will done on iteration {iteration + self.compile_after_iterations}")
            self.initial_iteration = iteration

            if self.compile_network:
                if model.config.ema.enabled is True and model.config.ema.torch_compile_buffer_renaming is False:
                    log.warning(
                        '"model.config.ema.torch_compile_buffer_renaming" should be turned on for the EMA to work with torch.compile(), network will not be compiled'
                    )

        if iteration - self.initial_iteration == self.compile_after_iterations:
            if self.compile_network:
                if model.config.ema.enabled is True and model.config.ema.torch_compile_buffer_renaming is False:
                    log.warning(
                        '"model.config.ema.torch_compile_buffer_renaming" should be turned on for the EMA to work with torch.compile(), skipping network compilation'
                    )
                else:
                    log.info("Compiling network")
                    model.network = torch.compile(model.network, dynamic=False)

            if self.compile_loss:
                for key in self.compile_loss_keys:
                    if key not in model.loss.loss_modules:
                        log.warning(f"Loss module for compilation with key: {key} not found")
                    else:
                        if (
                            hasattr(model.loss.loss_modules[key], "checkpoint_activations")
                            and getattr(model.loss.loss_modules[key], "checkpoint_activations") is True
                        ):
                            log.warning(
                                f"torch.compile() doesn't work with activation checkpointing, skipping compilation for loss with key: {key}"
                            )
                        else:
                            log.info(f"Compiling loss with key: {key}")
                            model.loss.loss_modules[key].torch_compile()
