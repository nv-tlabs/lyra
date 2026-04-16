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


"""
Distributed checkpoint (DCP) directory structure and storage backends.

The checkpointer saves model state in a sharded format across multiple processes:

self.save_dirname/
├── iter_000000005/                    # Checkpoint at iteration 5
│   ├── model/                         # Model state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── optim/                        # Optimizer state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── scheduler/                    # Learning rate scheduler state
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   └── trainer/                      # Additional training state
│       ├── __0_0.distcp              # Shard 0 from rank 0
│       └── __1_0.distcp              # Shard 1 from rank 1
└── latest_checkpoint.txt             # Points to most recent checkpoint folder, e.g. iter_000000005

Storage path format:
  self.save_dirname = "{config_job.path_local}/checkpoints"

The sharded format enables efficient distributed saving/loading by:
1. Parallelizing I/O across processes
2. Reducing memory usage per process
3. Supporting both local and cloud storage backends
"""

import enum
import functools
import multiprocessing
import os
import queue
import re
import time
import warnings
from collections import namedtuple
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.utils import _api_bc_check, _DistWrapper, _profile
from torch.distributed.tensor import distribute_tensor

from lyra_2._ext.imaginaire.checkpointer.base import AbstractCheckpointer
from lyra_2._ext.imaginaire.checkpointer.s3_filesystem import S3StorageReader, S3StorageWriter
from lyra_2._ext.imaginaire.config import CheckpointConfig, JobConfig
from lyra_2._ext.imaginaire.model import ImaginaireModel
from lyra_2._ext.imaginaire.utils import callback, distributed, log, misc
from lyra_2._ext.imaginaire.utils.easy_io import easy_io
from lyra_2._src.models.wan_t2v_model import WANDiffusionModel as DiffusionModel

try:
    """
    We override the default load function to skip loadding _extra_state keys created by transformer-engine.
    """
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner as _DefaultLoadPlanner
    from torch.distributed.checkpoint.default_planner import (
        DTensor,
        LoadPlan,
        _create_read_items,
        _version,
        flatten_state_dict,
    )
    from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

    def create_default_local_load_plan(
        state_dict: dict[str, Any], metadata: Metadata, strict: bool = True, dcp_allow_mismatched_size: bool = False
    ) -> LoadPlan:
        requests = []
        """
        Create the ``LoadPlan`` used by DefaultLoadPlanner.

        It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

        The default behavior is to match key exactly between state_dict and metadata.
        It handles resharding by issuing multiple read requests against storage in order to match
        load requirements.
        """

        for fqn, obj in state_dict.items():
            if fqn.endswith("._extra_state"):  # dirty TE attention package!
                continue
            # ignore state_dict keys which do not exist in `state_dict` if strict=False
            if fqn not in metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                else:
                    log.warning(f"Local load plan: Missing key in checkpoint state_dict: {fqn}.")
                    continue

            md = metadata.state_dict_metadata[fqn]

            if not dcp_allow_mismatched_size:
                if (
                    isinstance(md, TensorStorageMetadata)
                    and getattr(obj, "size", None) is not None
                    and md.size != obj.size()
                ):
                    if not strict:
                        log.critical(f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}")
                        continue
                    else:
                        raise ValueError(
                            f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
                        )
            # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
            # gets called only when the current rank is part of the mesh for the corresponding DTensor.
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    requests += _create_read_items(fqn, md, obj)
            else:
                requests += _create_read_items(fqn, md, obj)

        return LoadPlan(requests)

    class DefaultLoadPlanner(_DefaultLoadPlanner):
        def set_partial_channel_weight(self, dcp_allow_mismatched_size: bool):
            self.dcp_allow_mismatched_size = dcp_allow_mismatched_size

        def create_local_plan(self) -> LoadPlan:
            assert self.metadata is not None
            if self.flatten_state_dict:
                # To support checkpoints that are saved before v2.4, we have to
                # differentiate if the missing keys are due to old checkpoints.
                # The contracts are:
                # 1. There are 3 cases when we found a missing key.
                #    1.1 Actual missing key, but allow_partial_load is False
                #    1.2 Actual missing key, but allow_partial load is True
                #    1.3 Old checkpoint, but allow_partial_load is False
                #    1.4 Old checkpoint, but allow_partial_load is True
                # 2. If we found a missing key, we first convert the keys back to
                #    the key format of v2.3
                # 3. If the previous missing keys are in the v2.3 keys, we assume
                #    this is a old checkpoint.
                # 4. Pass the state_dict to `create_default_local_load_plan()`,
                #    which has the logic to check missing for allow_partial_load.
                # So for 1.2 and 1.4 cases, we delegate allow_partial_load check to
                # `create_default_local_load_plan()`. The logic here is to determine
                # whether the checkpoint belong to 2.3 (or before) or 2.4 (or after).
                current_keys = set(self.state_dict.keys())
                load_keys = set(self.metadata.state_dict_metadata.keys())
                missing_keys = load_keys - current_keys
                if missing_keys:
                    _version._derived_version = "2_3"
                    old_state_dict, old_mappings = flatten_state_dict(self.original_state_dict)
                    old_keys = set(old_state_dict.keys())
                    if old_keys & missing_keys:
                        self.state_dict, self.mappings = old_state_dict, old_mappings
                    # _derived_version is only used by flatten_state_dict now.
                    # Set it back to None so that later we can save to a new version.
                    _version._derived_version = None

            return create_default_local_load_plan(
                self.state_dict,
                self.metadata,
                not self.allow_partial_load,
                getattr(self, "dcp_allow_mismatched_size", False),
            )

    log.info("for the back comptiable pytorch! New DefaultLoadPlanner class is created.")

    @_dcp_method_logger(log_exceptions=True)
    @_api_bc_check
    def load(
        state_dict: dict[str, Any],
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_reader: Optional[StorageReader] = None,
        planner: Optional[DefaultLoadPlanner] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        no_dist: bool = False,
    ) -> None:
        """
        We override the default load function to perform a load plan check for mismatched/missing keys.
        ==========================Original Doc string=====================================
        Load a checkpoint into a distributed state dict in SPMD style.

        Each rank must have the same keys in their ``state_dict`` provided to this
        API. Mismatched keys may result in hangs or errors. If unsure, you can use
        the ``utils._assert_same_keys`` API to check (but may incur communication
        costs).

        Each rank will try to read the least amount of data necessary
        to fullfill the requested `state_dict`. When loading :class:`ShardedTensor`
        or :class:`DTensor` instances, each rank only reads data for their local shards.

        For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
        load will first call ``state_dict`` before attempting deserialization, followed by
        ``load_state_dict`` once the deserialization is complete.
        For each non-``Stateful`` object, load will deserailize the object, and then replace
        it in the ``state_dict`` with the deserialized object.

        .. warning::
            All tensors in ``state_dict`` must be allocated on their
            destination device *prior to* calling this function.

            All non-tensor data is loaded using `torch.load()` and modified in place
            on state_dict.

        .. warning::
            Users must call `load_state_dict` on the root module to ensure load
            pos-processing and non-tensor data properly propagates.

        .. note:
            If no process group is initialized, this function will assume the intent
            is to load a checkpoint into the local process. This can be useful in the
            case of local inference, and when using regular Tensors (as opposed to DTensor
            or ShardedTensor)

        .. note:
            Rank 0 is assumed to be the coordinator rank.

        Args:
            state_dict (Dict[str, Any]): The state_dict to load the checkpoint into.
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is a key-value store.
                (Default: ``None``)
            storage_reader (Optional[StorageReader]):
                Instance of StorageWriter used to perform reads. If this is not
                specified, DCP will automatically infer the reader based on the
                checkpoint_id. If checkpoint_id is also None, an exception will
                be raised. (Default: ``None``)
            planner (Optional[LoadPlanner]):
                Instance of LoadPlanner. If this is not specificed, the default
                planner will be used. (Default: ``None``)
            process_group (Optional[ProcessGroup]):
                ProcessGroup to be used for cross-rank synchronization.
                (Default: ``None``)
            no_dist (bool): If ``True``, this function will assume the intent is to load
                a checkpoint without using cross-rank synchronization. (Default: ``False``)
        Returns:
            None.

        Examples
            >>> # xdoctest: +SKIP
            >>> my_model = MyModule()
            >>> optimizer = Adagrad(my_model.parameters())
            >>> model_state_dict = my_model.state_dict()
            >>> fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(
            ...     "/checkpoint/1"
            ... )

            >>> torch.distributed.checkpoint.load_state_dict(
            >>>     state_dict=model_state_dict,
            >>>     storage_reader=fs_storage_reader,
            >>> )

            >>> # module.load_state_dict() function might have customized steps
            >>> # to flush the state_dict, must call it to
            >>> # ensure correct behavior.
            >>> my_model.load_state_dict(model_state_dict)

        .. note::
            load_state_dict uses collectives to coordinate reads across ranks.
            For NCCL-based process groups, internal tensor representations of
            objects must be moved to the GPU device before communication takes place.
            In this case, the device used is given by ``torch.cuda.current_device()``
            and it is the user's responsibility to ensure that this is set so that each
            rank has an individual GPU, via ``torch.cuda.set_device()``.
        """

        no_dist = no_dist or (not torch.distributed.is_available()) or (not torch.distributed.is_initialized())
        if no_dist:
            warnings.warn(
                "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process."
            )

        with _profile():
            storage_reader = cast(StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True))

            # All ranks must have the same keys in their `state_dict` provided to
            # this API.  See documentation for more details.
            # Here we simply sort the keys to ensure that all ranks load values in
            # the same order.
            keys = sorted(state_dict.keys())

            statetful_sd = {}
            for key in keys:
                if key not in state_dict:
                    continue
                elem = state_dict[key]
                statetful_sd[key] = elem.state_dict() if isinstance(elem, Stateful) else elem

            _load_state_dict(
                state_dict=statetful_sd,
                storage_reader=storage_reader,
                process_group=process_group,
                no_dist=no_dist,
                planner=planner,
            )
            for key in keys:
                if key not in state_dict:
                    continue
                elem = state_dict[key]
                if isinstance(elem, Stateful):
                    # If the state_dict is a Stateful object,
                    # DCP does an in-place load in the original state dict.
                    elem.load_state_dict(statetful_sd[key])
                else:
                    # Otherwise, replace the state_dict with the loaded state_dict.
                    state_dict[key] = statetful_sd[key]

    def _load_state_dict(
        state_dict: dict[str, Any],
        storage_reader: StorageReader,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        planner: Optional[DefaultLoadPlanner] = None,
    ) -> None:
        torch._C._log_api_usage_once("torch.distributed.checkpoint.load_state_dict")

        distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
        if planner is None:
            planner = DefaultLoadPlanner()

        ckpt_kwargs = {}
        if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
            ckpt_kwargs["checkpoint_id"] = ckpt_id
            ckpt_kwargs["process_group"] = distW.group

        @_dcp_method_logger(**ckpt_kwargs)
        def local_step():
            assert planner is not None
            metadata = storage_reader.read_metadata()
            planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
            storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

            local_plan = planner.create_local_plan()
            local_plan = storage_reader.prepare_local_plan(local_plan)
            return local_plan

        @_dcp_method_logger(**ckpt_kwargs)
        def global_step(all_local_plans):
            assert planner is not None
            all_local_plans = planner.create_global_plan(all_local_plans)
            all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
            return all_local_plans

        central_plan: LoadPlan = distW.reduce_scatter("plan", local_step, global_step)
        if distW.is_coordinator:
            # Compare central_plan items with storage_reader.storage_data keys
            dest_fqns = set()
            storage_fqns = set()

            # Extract FQNs from central_plan items
            for item in central_plan.items:
                if hasattr(item, "dest_index") and hasattr(item.dest_index, "fqn"):
                    dest_fqns.add(item.dest_index.fqn)
                if hasattr(item, "storage_index") and hasattr(item.storage_index, "fqn"):
                    storage_fqns.add(item.storage_index.fqn)

            # Get storage data keys
            storage_data_keys = set()
            if hasattr(storage_reader, "storage_data") and storage_reader.storage_data is not None:
                storage_data_keys = set(item[0].fqn for item in storage_reader.storage_data.items())
            state_dict_keys = set(state_dict.keys())
            # Compare sets and log differences
            # Remove any item that has "_extra_state" as substring in the sets
            state_dict_keys = {fqn for fqn in state_dict_keys if "_extra_state" not in fqn}
            dest_fqns = {fqn for fqn in dest_fqns if "_extra_state" not in fqn}
            storage_fqns = {fqn for fqn in storage_fqns if "_extra_state" not in fqn}
            storage_data_keys = {fqn for fqn in storage_data_keys if "_extra_state" not in fqn}

            log.info("=== Load Plan FQN Analysis ===")
            log.info(f"State Dict FQNs count: {len(state_dict_keys)}")
            log.info(f"Destination FQNs count (without _extra_state): {len(dest_fqns)}")
            log.info(f"Loaded FQNs count (without _extra_state): {len(storage_fqns)}")
            log.info(f"In Storage keys count (without _extra_state): {len(storage_data_keys)}")

            # Find missing keys in each direction
            state_dict_missing_from_dest = state_dict_keys - dest_fqns
            storage_data_missing_from_storage_fqns = storage_data_keys - storage_fqns

            if state_dict_missing_from_dest:
                log.info(
                    f"State Dict FQNs missing from load plan ({len(state_dict_missing_from_dest)} items): {sorted(state_dict_missing_from_dest)}"
                )
            else:
                log.info("✓ All State Dict FQNs found in storage_data")

            if storage_data_missing_from_storage_fqns:
                # If there are more than 100 "net_ema" keys in storage_data_missing_from_storage_fqns, summarize them
                net_ema_keys = {k for k in storage_data_missing_from_storage_fqns if "net_ema" in k}
                if len(net_ema_keys) > 100:
                    storage_data_missing_from_storage_fqns = storage_data_missing_from_storage_fqns - net_ema_keys
                    storage_data_missing_from_storage_fqns = set(
                        storage_data_missing_from_storage_fqns
                    )  # ensure set type
                    storage_data_missing_from_storage_fqns.add("net_ema")
                    log.info(
                        f"Summarized {len(net_ema_keys)} 'net_ema' keys as 'net_ema' in missing storage data keys."
                    )
                log.info(
                    f"Storage data keys not loaded by load plan ({len(storage_data_missing_from_storage_fqns)} items): {sorted(storage_data_missing_from_storage_fqns)}"
                )
            else:
                log.info("✓ All storage data keys found in Loaded FQNs")

            log.info("=== End Load Plan FQN Analysis ===")

        @_dcp_method_logger(**ckpt_kwargs)
        def read_data():
            assert planner is not None
            final_local_plan = planner.finish_plan(central_plan)
            all_reads = storage_reader.read_data(final_local_plan, planner)

            all_reads.wait()
            return None

        _ = distW.all_gather("read", read_data)

except ImportError as e:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    log.critical(f"{e}, using default planner")

StateDictItemPath = namedtuple("StateDictItemPath", ["state_dict", "save_path"])


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model: Union[nn.Module, List[nn.Module]], load_ema_to_reg: bool = False):
        self.model = [model] if isinstance(model, nn.Module) else model
        self.load_ema_to_reg = load_ema_to_reg
        if self.load_ema_to_reg:
            assert isinstance(model, DiffusionModel)

    def state_dict(self) -> Dict[str, Any]:
        _state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}
        if self.load_ema_to_reg:
            if not self.model[0].config.ema.enabled:
                all_keys = list(_state_dict.keys())
                assert all(k.startswith("net.") for k in all_keys), "All keys must start with net."
                for k in all_keys:
                    _state_dict[k.replace("net.", "net_ema.")] = _state_dict.pop(k)
            else:
                log.warning("EMA is enabled, will only load EMA weights from checkpoint file.")
                all_keys = list(_state_dict.keys())
                for k in all_keys:
                    if k.startswith("net_ema."):
                        break
                else:
                    raise ValueError("No EMA keys found in state_dict")
                # do not load .net keys, since we do not need them anyway.
                _state_dict = {k: _state_dict[k] for k in all_keys if not k.startswith("net.")}

        if hasattr(self.model[0].config, "lora_config") and self.model[0].config.lora_config.enabled:
            """
            When using LoRA, `inject_adapter_in_model` modifies the target modules in place.
            For example, `blocks[0].attn.q_proj.weight` will be modified to `blocks[0].attn.q_proj.base_layer.weight`.
            This means that the model will have the key `blocks[0].attn.q_proj.base_layer.weight`,
            but the checkpoint will have the key `blocks[0].attn.q_proj.weight`.
            We need to map the model key to the checkpoint key.
            """
            self.checkpoint_to_model_key = {}
            mapping_keys = {"base_layer.": "", "base_model.model.": ""}
            keys_to_update = []
            for k in _state_dict.keys():
                new_key = k
                for from_key, to_key in mapping_keys.items():
                    new_key = new_key.replace(from_key, to_key)
                if new_key != k:
                    keys_to_update.append((k, new_key))
                    self.checkpoint_to_model_key[new_key] = k
            for k, new_key in keys_to_update:
                _state_dict[new_key] = _state_dict.pop(k)

        return _state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if hasattr(self.model[0].config, "lora_config") and self.model[0].config.lora_config.enabled:
            if hasattr(self, "checkpoint_to_model_key"):
                for checkpoint_key, model_key in self.checkpoint_to_model_key.items():
                    state_dict[model_key] = state_dict.pop(checkpoint_key)
            else:
                raise ValueError("checkpoint_to_model_key is not set by `state_dict`")

        if self.load_ema_to_reg:
            if not self.model[0].config.ema.enabled:
                all_keys = list(state_dict.keys())
                assert all(k.startswith("net_ema.") for k in all_keys), "All keys must start with net_ema."
                for k in all_keys:
                    state_dict[k.replace("net_ema.", "net.")] = state_dict.pop(k)
            else:
                log.warning("EMA is enabled, will load EMA weights to regular model weights")
                all_keys = list(state_dict.keys())
                assert all(not k.startswith("net.") for k in all_keys), "No .net keys should be in state_dict"
                for k in all_keys:
                    if k.startswith("net_ema."):
                        state_dict[k.replace("net_ema.", "net.")] = torch.clone(state_dict[k])
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    def __init__(self, iteration: int, elapsed_time: float, succeeded: bool):
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        self.succeeded = succeeded

    def __str__(self):
        return f"SaveDone(iteration={self.iteration}, elapsed_time={self.elapsed_time}, succeeded={self.succeeded})"


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    checkpoint_config: CheckpointConfig,
    job_config: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        checkpoint_config: Configuration settings for checkpoint saving behavior
        job_config: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init()

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(checkpoint_config, job_config, None, disable_async=True)

    try:
        while True:
            log.debug("Checkpoint background process is ready for next task, waiting for new state_dict")
            received_data = receiver_queue.get()
            log.debug("Received new state_dict")

            if isinstance(received_data, Terminate):
                log.info("Received termination signal in checkpoint background process, closing sender queue")
                sender_queue.put(Terminate())
                sender_queue.close()
                return

            assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
            state_dict, checkpoint_path = received_data

            # Save checkpoint and measure time taken
            start_time = time.monotonic()
            iteration = state_dict["trainer"][0]["iteration"]
            elapsed_time = 0
            succeeded = False
            try:
                checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)
                elapsed_time = time.monotonic() - start_time
                log.info(
                    f"Checkpoint saved successfully in background process. Time taken: {elapsed_time:.2f} seconds, iteration: {iteration}"
                )
                succeeded = True
            except Exception as e:
                log.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
                # continue because if the thread exits, the main thread keeps on adding to the queue
            finally:
                if elapsed_time == 0:
                    elapsed_time = time.monotonic() - start_time
                sender_queue.put(SaveDone(iteration, elapsed_time, succeeded))

    finally:
        log.info("Cleaning up: destroying distributed process group")
        torch.distributed.destroy_process_group()


class DistributedCheckpointer(AbstractCheckpointer):
    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: Optional[callback.CallBackGroup] = None,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.dcp_async_mode_enabled:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if disable_async:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()

    def keys_to_resume_during_load(self) -> Tuple[Set, Union[str, None]]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []

        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path and not self.load_path.endswith(".pth"):
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"
                    if not re.search(r"/checkpoints/iter_\d{9}/?$", checkpoint_path):
                        old_ckpt_path = checkpoint_path
                        # If path doesn't end with specific checkpoint, read latest checkpoint file
                        latest_ckpt_path = os.path.join(checkpoint_path, "checkpoints/latest_checkpoint.txt")
                        if easy_io.exists(latest_ckpt_path, backend_key=self.load_s3_backend_key):
                            checkpoint_file = easy_io.load(
                                latest_ckpt_path, backend_key=self.load_s3_backend_key
                            ).strip()
                            checkpoint_path = f"{checkpoint_path}/checkpoints/{checkpoint_file}"
                        else:
                            log.warning(
                                f"Latest checkpoint file {latest_ckpt_path} not found, load from {old_ckpt_path}"
                            )
                            checkpoint_path = old_ckpt_path
                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            elif self.load_path and self.load_path.endswith(".pth"):
                checkpoint_path = self.load_path
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.KEYS_TO_SAVE, f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]
        return set(resume_keys), checkpoint_path

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.info(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0

        if checkpoint_path is not None and not checkpoint_path.endswith(".pth"):
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                load_planner = DefaultLoadPlanner(allow_partial_load=True)
                if hasattr(load_planner, "set_partial_channel_weight"):
                    log.info(f"set_partial_channel_weight: {self.config_checkpoint.dcp_allow_mismatched_size}")
                    load_planner.set_partial_channel_weight(self.config_checkpoint.dcp_allow_mismatched_size)
                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.info(f"Start loading checkpoint from {checkpoint_path}")
                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                torch.distributed.barrier()
                log.info(f"starting {cur_key_ckpt_full_path}", rank0_only=False)
                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    load(_state_dict, storage_reader=storage_reader, planner=load_planner)
                    _model_wrapper.load_state_dict(_state_dict)
                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _optim_wrapper = OptimizerWrapper(model, optimizer)
                    _state_dict = _optim_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    _optim_wrapper.load_state_dict(_state_dict)
                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    scheduler.load_state_dict(_state_dict)
                elif key == "trainer":
                    log.info("- Loading the trainer...")
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")
            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=_state_dict)
            log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        elif checkpoint_path is not None and checkpoint_path.endswith(".pth"):
            state = easy_io.load(checkpoint_path)
            model_state = model.net.state_dict()

            for k, v in list(state.items()):
                tgt = model_state.get(k, None)
                if tgt is None:
                    continue
                # If target param/buffer is a DTensor and checkpoint value is not, distribute it
                if isinstance(tgt, DTensor) and not isinstance(v, DTensor):
                    # Match device, dtype, and placements from the target DTensor
                    v = v.to(tgt.device, dtype=tgt.dtype, copy=False)
                    v = distribute_tensor(v, tgt.device_mesh, tgt.placements)
                    state[k] = v
                # If target is a plain Tensor but checkpoint is a DTensor, bring it local
                if not isinstance(tgt, DTensor) and isinstance(v, DTensor):
                    state[k] = v.to_local().to(tgt.device, dtype=tgt.dtype, copy=False)

            model.load_state_dict(state, strict=False, pretrain_copy=True)
            # Clear unused reserved memory from fp32
            torch.cuda.empty_cache()
            log.critical(f"Loaded checkpoint from {checkpoint_path}. **** This only happen at iteration 0 **** ")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def _async_with_pinned_memory(self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]) -> None:
        try:
            from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        if self.cpu_offload_state_dict is None:
            log.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_ckpt_file = checkpoint_file

        self.maybe_wait_for_staging()

    def maybe_wait_for_staging(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM and self.staging:
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))

            sync_func()
            self.staging = False

    def get_previous_checkpoint_results(self, wait_for: int = 0) -> None:
        """Get the results of previously submitted checkpoints and pass them to callbacks if checkpoint succeeded"""
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            try:
                start_time = time.monotonic()
                while not self.mp_queue_recv.empty() or wait_for > 0:
                    try:
                        ret = self.mp_queue_recv.get(timeout=1)
                        if isinstance(ret, Terminate):
                            log.info("Received termination event from checkpoint background process")
                            break
                        save_done: SaveDone = ret
                        log.logger.info(f"Received checkpoint save result: {save_done}")
                        if self.callbacks is not None and save_done.succeeded:
                            self.callbacks.on_save_checkpoint_success(
                                iteration=save_done.iteration, elapsed_time=save_done.elapsed_time
                            )
                    except queue.Empty:
                        elapsed_time = time.monotonic() - start_time
                        if elapsed_time > wait_for:
                            break
            except (EOFError, BrokenPipeError):
                log.info("Queue was closed by checkpoint background process")

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
            )
        return FileSystemReader(checkpoint_path)

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for k, (v, full_checkpoint_path) in to_save_dict.items():
            storage_writer = self.get_storage_writer(full_checkpoint_path)
            dcp.save(
                v,
                storage_writer=storage_writer,
                planner=DefaultSavePlanner(dedup_save_to_lowest_rank=True),
            )

        if distributed.is_rank0():
            print(f"Saving last checkpoint file {checkpoint_file}")
            self._write_latest_checkpoint_file(checkpoint_file)

        log.critical(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}", rank0_only=True)

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.get_previous_checkpoint_results(wait_for=0)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"
        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": OptimizerWrapper(model, optimizer).state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
            },
        }
        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            try:
                self.save_state_dict_worker(to_save_dict, checkpoint_file)
            finally:
                if self.callbacks is not None:
                    self.callbacks.on_save_checkpoint_success(
                        iteration=iteration, elapsed_time=time.monotonic() - start_time
                    )

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.get_previous_checkpoint_results(wait_for=60)
                self.mp.join()
