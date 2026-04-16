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


import math

# Backported gradient-clipping utilities from newer PyTorch / torchtitan.
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import DeviceMesh
from torch.distributed._tensor.api import DTensor
from torch.nn.utils.clip_grad import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
    _no_grad,
    _tensor_or_tensors,
)


# https://github.com/pytorch/torchtitan/blob/d4c86e3758a84cf23e2e879ab3c995cba9d5e410/torchtitan/utils.py#L354
@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)  # prevent generators from being exhausted

    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


# from: pytorch/pytorch/torch/nn/utils/clip_grad.py
@_no_grad
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] = (
        _group_tensors_by_device_and_dtype(
            [tensors]  # type: ignore[list-item]
        )
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_tensors])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


get_total_norm = _get_total_norm


# from: pytorch/pytorch/torch/nn/utils/clip_grad.py
@_no_grad
def _clip_grads_with_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    r"""Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm.

    The gradients will be scaled by the following calculation

    .. math::
        grad = grad * \frac{max\_norm}{total\_norm + 1e-6}

    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.clip_grad_norm_` with a pre-calculated
    total norm.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        total_norm (Tensor): total norm of the gradients to use for clipping
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] = (
        _group_tensors_by_device_and_dtype([grads])
    )  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)


clip_grads_with_norm_ = _clip_grads_with_norm_
