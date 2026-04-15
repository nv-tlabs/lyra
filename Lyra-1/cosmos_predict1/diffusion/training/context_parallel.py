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

import torch
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, get_process_group_ranks, get_world_size


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Split input tensor along the sequence dimension for checkpoint parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the checkpoint parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimension along which to split the input (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A slice of the input tensor corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    assert x.shape[seq_dim] % cp_size == 0, f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    x = x.view(*x.shape[:seq_dim], cp_size, x.shape[seq_dim] // cp_size, *x.shape[(seq_dim + 1) :])
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    # Note that the new sequence length is the original sequence length / cp_size
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenate outputs from different ranks in the checkpoint parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    # Get the world size (number of processes in the group)
    world_size = get_world_size(cp_group)

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    # Concatenate the gathered tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)


def cat_outputs_cp_with_grad(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenate outputs from different ranks in the context parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    It retains computational graph locally for each rank by replacing the portion of the tensor with original output.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    # Get the world size (number of processes in the group)
    cp_size = cp_group.size()
    assert cp_size > 0, "cp_size should be greater than 0"

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(cp_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to gather tensors: {e}")

    rank = cp_group.rank()
    gathered_tensors[rank] = x
    # Concatenate the gathered tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)
