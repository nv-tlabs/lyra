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
from typing import Any, Dict, Optional, Type, TypeVar, Union

import attrs
import torch
from megatron.core import ModelParallelConfig

from cosmos_predict1.utils import callback
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.misc import Color

T = TypeVar("T")


def _is_attrs_instance(obj: object) -> bool:
    """
    Helper function to check if an object is an instance of an attrs-defined class.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs-defined class, False otherwise.
    """
    return hasattr(obj, "__attrs_attrs__")


def make_freezable(cls: T) -> T:
    """
    A decorator that adds the capability to freeze instances of an attrs-defined class.

    NOTE: This requires the wrapped attrs to be defined with attrs.define(slots=False) because we need
    to hack on a "_is_frozen" attribute.

    This decorator enhances an attrs-defined class with the ability to be "frozen" at runtime.
    Once an instance is frozen, its attributes cannot be changed. It also recursively freezes
    any attrs-defined objects that are attributes of the class.

    Usage:
        @make_freezable
        @attrs.define(slots=False)
        class MyClass:
            attribute1: int
            attribute2: str

        obj = MyClass(1, 'a')
        obj.freeze()  # Freeze the instance
        obj.attribute1 = 2  # Raises AttributeError

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with added freezing capability.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError(
            "make_freezable cannot be used with classes that do not define __dict__. Make sure that the wrapped "
            "class was defined with `@attrs.define(slots=False)`"
        )

    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:  # noqa: ANN001
        """
        Override __setattr__ to allow modifications during initialization
        and prevent modifications once the instance is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)  # type: ignore

    cls.__setattr__ = setattr_override  # type: ignore

    def freeze(self: object) -> None:
        """
        Freeze the instance and all its attrs-defined attributes.
        """
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True  # type: ignore

    cls.freeze = freeze  # type: ignore

    return cls


def _pretty_print_attrs_instance(obj: object, indent: int = 0, use_color: bool = False) -> str:
    """
    Recursively pretty prints attrs objects with color.
    """

    assert attrs.has(obj.__class__)

    lines: list[str] = []
    for attribute in attrs.fields(obj.__class__):
        value = getattr(obj, attribute.name)
        if attrs.has(value.__class__):
            if use_color:
                lines.append("   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ":")
            else:
                lines.append("   " * indent + "* " + attribute.name + ":")
            lines.append(_pretty_print_attrs_instance(value, indent + 1, use_color))
        else:
            if use_color:
                lines.append(
                    "   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ": " + Color.yellow(value)
                )
            else:
                lines.append("   " * indent + "* " + attribute.name + ": " + str(value))
    return "\n".join(lines)


def pretty_print_overrides(overrides: Optional[list[str]] = None, use_color: bool = False) -> str:
    """
    Pretty prints overrides.
    """

    lines: list[str] = []
    lines.append(Color.cyan("* ") + Color.green("overrides") + ": ")
    for override in overrides:
        if override == "--":
            continue
        attribute_name, attribute_value = override.split("=")
        if use_color:
            lines.append("   " + Color.cyan("* ") + Color.green(attribute_name) + ": " + Color.yellow(attribute_value))
        else:
            lines.append("   " + "* " + attribute_name + ": " + str(attribute_value))

    return "\n".join(lines)


@make_freezable
@attrs.define(slots=False)
class JobConfig:
    # Project name.
    project: str = ""
    # Experiment name.
    group: str = ""
    # Run/job name.
    name: str = ""

    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"

    @property
    def path_local(self) -> str:
        local_root = os.environ.get("OUTPUT_ROOT", "checkpoints")
        return f"{local_root}/{self.path}"


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    # Enable tracking a set of exponential moving average (EMA) weights.
    enabled: bool = False
    # EMA decay rate.
    beta: float = 0.9999
    # Enable removing "_orig_mod-" from buffer names that is added by torch.compile
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class DDPConfig:
    # Traverse the computation graph to find parameters that don't receive gradients.
    find_unused_parameters: bool = False
    # Set to True if the computation graph does not change during the whole training loop.
    static_graph: bool = True
    # Set to True if we want to synchronize buffers. Set to False if the sync is going to be handled elsewhere.
    broadcast_buffers: bool = True


@make_freezable
@attrs.define(slots=False)
class CuDNNConfig:
    # Set to True for better reproducibility of the results (only using deterministic cudnn functions).
    deterministic: bool = False
    # If set to True, cudnn will benchmark several algorithms and pick the fastest one.
    benchmark: bool = True


@make_freezable
@attrs.define(slots=False)
class JITConfig:
    # Enable exporting a JIT compiled model.
    enabled: bool = False
    # Input tensor shape, for example input.
    input_shape: Union[list[int], None] = None
    # Device to compile onto.
    device: str = "cuda"
    # # Data type to compile onto.
    dtype: str = "bfloat16"
    # Strict mode for PyTorch JIT.
    strict: bool = True


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    # possible checkpoint class
    type: Optional[Dict] = None
    # for dcp, whether to use async mode
    dcp_async_mode_enabled: bool = False
    # Save the checkpoint every N iterations.
    save_iter: int = 999999999
    # Path of model weights to resume the checkpoint from.
    load_path: str = ""
    # Whether to load the training states (optimizer/scheduler/grad-scaler) from the checkpoint path.
    load_training_state: bool = False
    # Whether to load the scheduler state only from the checkpoint path. If load_training_state is True, this will be ignored.
    only_load_scheduler_state: bool = False
    # Load state_dict to the models in strict mode.
    strict_resume: bool = True
    # Print detailed information during checkpoint saving/loading.
    verbose: bool = True
    # Configs for JIT compiling EMA model.
    jit: JITConfig = attrs.field(factory=JITConfig)
    # keys not to resume from the checkpoint, choices: ["model", "optim", "scheduler", "trainer"]
    keys_not_to_resume: list[str] = []
    # Whether to use the local filesystem for broadcasting checkpoint data (used for Tensor Parallel Checkpointer).
    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False
    async_saving: bool = True


@make_freezable
@attrs.define(slots=False)
class TrainerConfig:
    from cosmos_predict1.utils.trainer import Trainer

    type: Type[Trainer] = Trainer
    # Set the callback class.
    # Defaults to the callbacks below.
    callbacks: LazyDict = LazyDict(
        dict(
            ema=L(callback.EMAModelCallback)(),
            progress_bar=L(callback.ProgressBarCallback)(),
        )
    )
    # distributed parallelism strategy
    distributed_parallelism: str = "ddp"
    # Distributed data parallel configs.
    ddp: DDPConfig = attrs.field(factory=DDPConfig)
    # cuDNN configs.
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    # Set the random seed.
    seed: int = 0
    # Gradient scaler arguments (for torch.amp.GradScaler).
    grad_scaler_args: dict = attrs.field(factory=lambda: dict(enabled=False))
    # Maximum number of iterations to train the model.
    max_iter: int = 999999999
    # Maximum number of iterations to validate the model. If None, validate on the entire dataset.
    max_val_iter: int | None = None
    # How often we log the training stats.
    logging_iter: int = 100
    # Whether we want to run the validation routines.
    run_validation: bool = True
    # How often we evaluate on the validation set.
    validation_iter: int = 999999999
    # Kill the process after N seconds since the last iteration (usually means dead job).
    timeout_period: int = 999999999
    # Tensor memory organization format.
    memory_format: torch.memory_format = torch.preserve_format
    # Gradient accumulation (update step every N iteration).
    grad_accum_iter: int = 1
    # # Profiling config
    # profiling: Profiling = attrs.field(factory=Profiling)


@make_freezable
@attrs.define(slots=False)
class Config:
    """Config for a job.

    See /README.md/Configuration System for more info.
    """

    # Model configs.
    model: LazyDict
    # Optimizer configs.
    optimizer: LazyDict = LazyDict(dict(dummy=None))
    # Scheduler configs.
    scheduler: LazyDict = LazyDict(dict(dummy=None))
    # Training data configs.
    dataloader_train: LazyDict = LazyDict(dict(dummy=None))
    # Validation data configs.
    dataloader_val: LazyDict = LazyDict(dict(dummy=None))

    # Training job configs.
    job: JobConfig = attrs.field(factory=JobConfig)

    # Trainer configs.
    trainer: TrainerConfig = attrs.field(factory=TrainerConfig)

    # Megatron-Core configs
    model_parallel: ModelParallelConfig = attrs.field(factory=ModelParallelConfig)

    # Checkpointer configs.
    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)

    def pretty_print(self, use_color: bool = False) -> str:
        return _pretty_print_attrs_instance(self, 0, use_color)

    # Training job configs.
    job: JobConfig = attrs.field(factory=JobConfig)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def validate(self) -> None:
        """Validate that the config has all required fields."""
        assert self.job.project != "", "Project name is required."
        assert self.job.group != "", "Group name is required."
        assert self.job.name != "", "Job name is required."
