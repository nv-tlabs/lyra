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

# flake8: noqa
# isort: skip_file

"""
torch 2.2 has bugs in loading optimizer states for FSDP in hybrid mode
torch impl uses state.rank and dist.rank() inconsistently
The file fix the bugs. Verified it works for hybrid mode and fullly sharded mode
Please use the `scatter_full_optim_state_dict` in the code to replace the corresponding function in torch 2.2
"""

import copy
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._optim_utils import (
    _flatten_optim_state,
    _FSDPState,
    _get_fqn_to_fsdp_param_info,
    _get_param_to_fqns,
    _OptimStateKey,
    _PosDimTensorInfo,
    _shard_orig_param_state,
    tree_map_only,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import _rekey_sharded_optim_state_dict


def _broadcast_processed_state(
    fsdp_state: _FSDPState,
    optim_state: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
) -> Dict[str, Any]:
    objects: List[Any] = [None]
    if fsdp_state.rank == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),
            optim_state,
        )
    dist.broadcast_object_list(objects, src=0, group=group)
    if dist.get_rank() == 0:
        return optim_state
    else:
        return objects[0]


def _broadcast_state(fsdp_state: _FSDPState, state: Any, group: Optional[dist.ProcessGroup]) -> Any:
    if dist.get_rank() == 0:
        if not isinstance(state, torch.Tensor) or state.dim() == 0:
            return state
        tensor = state.to(fsdp_state.compute_device)
    else:
        if isinstance(state, torch.Tensor):
            assert state.dim() == 0, (
                "For non-zero ranks, a tensor state should have zero dimension, "
                "but got the state with shape {state.shape()}."
            )
            return state
        elif not isinstance(state, _PosDimTensorInfo):
            return state
        tensor = torch.zeros(state.shape, dtype=state.dtype, device=fsdp_state.compute_device)
    dist.broadcast(tensor, src=0, group=group)
    return tensor


def _flatten_optim_state_dict(
    optim_state_dict: Dict[str, Any],
    model: nn.Module,
    use_orig_params: bool = False,
    optim: Optional[torch.optim.Optimizer] = None,
    rank0_only: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    SimpleProfiler.reset()

    unflat_osd = optim_state_dict
    if "state" not in unflat_osd and not rank0_only:
        raise ValueError('`optim_state_dict` must have the keys "state"' "to be a valid optimizer state dict")
    param_to_fqns = _get_param_to_fqns(model)
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state

    # Broadcast unflat_osd without non-scalar tensor if rank0_only is True.
    if rank0_only:
        unflat_osd = _broadcast_processed_state(fsdp_state, unflat_osd, group=group)

    # Construct the "state" part
    flat_osd_state: Dict[Union[_OptimStateKey, str], Any] = {}
    unflat_osd_state = unflat_osd["state"]
    all_state_keys = set(unflat_osd_state.keys())

    for param, fqns in param_to_fqns.items():
        fqn = fqns[0]
        if fqn not in unflat_osd_state:
            continue
        all_state_keys.difference_update(fqns)

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name in unflat_osd_state[fqn].keys():
                    unflat_osd_state[fqn][state_name] = _broadcast_state(
                        fsdp_state, unflat_osd_state[fqn][state_name], group=group
                    )
            fqn = fqns[0]
        if fqn in fqn_to_fsdp_param_info:
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
                    flat_state = _shard_orig_param_state(
                        fsdp_param_info,
                        fqn,
                        unflat_osd_state[fqn],
                    )
            else:
                flat_state = _flatten_optim_state(
                    fsdp_param_info,
                    unflat_osd_state,
                    fqns,
                )
            key = _OptimStateKey(tuple(fqns), True)
            # Only include non-empty states since as expected by
            # `torch.optim.Optimizer` s unless the optimizer is KeyedOptimizer
            # or NamedOptimizer.
            if flat_state:
                flat_osd_state[key] = flat_state
            elif use_orig_params:
                assert len(fqns) == 1, f"use_orig_params is True but there are multiple FQNs, {fqns}."
                if optim is not None:  # NamedOptimizer or KeyedOptimizer case.
                    state = optim.state.get(param, None)  # type: ignore[call-overload]
                    if state is not None:
                        flat_osd_state[key] = copy.deepcopy(state)
                    else:
                        warnings.warn(f"optim_state[{key}] is not on rank{fsdp_state.rank}.")

            else:
                raise RuntimeError(f"The state of {key} is empty. This should happen when " "use_orig_params=True.")
        else:  # do not flatten non-FSDP parameters' states
            assert len(fqns) == 1
            key = _OptimStateKey(tuple(fqns), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[fqn])

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name, param_state in list(unflat_osd_state[fqn].items()):
                    if fsdp_state.rank > 0:
                        # Deference the tensor so that PyTorch can collect the memory.
                        del unflat_osd_state[fqn][state_name]
                    else:
                        # Move the tensor in the original osd back to CPU to make the
                        # original osd unaffected.
                        unflat_osd_state[fqn][state_name] = unflat_osd_state[fqn][state_name].cpu()

    # Handle user-defined state, states that are not associated with parameters.
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = _broadcast_state(fsdp_state, user_state, group=group)
        flat_osd_state[key] = copy.copy(user_state)

    SimpleProfiler.dump_and_reset("FSDP _flatten_optim_state_dict() profiling: ")
    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    # Only copy param_groups if it exists in unflat_osd
    if "param_groups" in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
        return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}
    else:
        return {"state": flat_osd_state}


def _optim_state_dict_to_load_impl(
    optim_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    full_state_dict: bool = True,
    rank0_only: bool = False,
    is_named_optimizer: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    The internal API that is used by all the load optim_state_dict implementations.
    Given model, optim, and the saved optim_state_dict, this API adds the FSDP
    internal information and internal sharding to the optim_state_dict.
    """
    if full_state_dict:
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
    else:
        using_optim_input = False
        assert optim_input is None and not rank0_only

    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[0]._use_orig_params
    assert all(
        use_orig_params == m._use_orig_params for m in FullyShardedDataParallel.fsdp_modules(model)
    ), "Not all FSDP modules have the same _use_orig_params value"

    if rank0_only and dist.get_rank(group) > 0:
        optim_state_dict = {}
    sharded_osd = _flatten_optim_state_dict(
        optim_state_dict,
        model=model,
        use_orig_params=use_orig_params,
        optim=(optim if is_named_optimizer else None),
        rank0_only=rank0_only,
        group=group,
    )
    return _rekey_sharded_optim_state_dict(
        sharded_osd,
        model=model,
        optim=optim,
        optim_input=optim_input,
        using_optim_input=using_optim_input,
        is_named_optimizer=is_named_optimizer,
    )


def scatter_full_optim_state_dict(
    full_optim_state_dict: Optional[Dict[str, Any]],
    model: torch.nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    group: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Scatters the full optimizer state dict from rank 0 to all other ranks,
    returning the sharded optimizer state dict on each rank. The return
    value is the same as :meth:`shard_full_optim_state_dict`, and on rank
    0, the first argument should be the return value of
    :meth:`full_optim_state_dict`.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> model, optim = ...
        >>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
        >>> # Define new model with possibly different world size
        >>> new_model, new_optim, new_group = ...
        >>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
        >>> new_optim.load_state_dict(sharded_osd)

    .. note:: Both :meth:`shard_full_optim_state_dict` and
        :meth:`scatter_full_optim_state_dict` may be used to get the
        sharded optimizer state dict to load. Assuming that the full
        optimizer state dict resides in CPU memory, the former requires
        each rank to have the full dict in CPU memory, where each rank
        individually shards the dict without any communication, while the
        latter requires only rank 0 to have the full dict in CPU memory,
        where rank 0 moves each shard to GPU memory (for NCCL) and
        communicates it to ranks appropriately. Hence, the former has
        higher aggregate CPU memory cost, while the latter has higher
        communication cost.

    Args:
        full_optim_state_dict (Optional[Dict[str, Any]]): Optimizer state
            dict corresponding to the unflattened parameters and holding
            the full non-sharded optimizer state if on rank 0; the argument
            is ignored on nonzero ranks.
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            correspond to the optimizer state in ``full_optim_state_dict``.
        optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
            Input passed into the optimizer representing either a
            :class:`list` of parameter groups or an iterable of parameters;
            if ``None``, then this method assumes the input was
            ``model.parameters()``. This argument is deprecated, and there
            is no need to pass it in anymore. (Default: ``None``)
        optim (Optional[torch.optim.Optimizer]): Optimizer that will load
            the state dict returned by this method. This is the preferred
            argument to use over ``optim_input``. (Default: ``None``)
        group (dist.ProcessGroup): Model's process group or ``None`` if
            using the default process group. (Default: ``None``)

    Returns:
        Dict[str, Any]: The full optimizer state dict now remapped to
        flattened parameters instead of unflattened parameters and
        restricted to only include this rank's part of the optimizer state.
    """
    FullyShardedDataParallel._warn_legacy_optim_state_dict("scatter_full_optim_state_dict", "optim_state_dict_to_load")
    return _optim_state_dict_to_load_impl(
        optim_state_dict=full_optim_state_dict,
        model=model,
        optim_input=optim_input,
        optim=optim,
        full_state_dict=True,
        rank0_only=True,
        is_named_optimizer=False,
        group=group,
    )
