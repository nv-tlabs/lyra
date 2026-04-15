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

from typing import List

import torch
import torch.distributed as dist
from megatron.core import mpu, parallel_state
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.autograd import Function
from torch.distributed import broadcast, get_process_group_ranks
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.module.rmsnorm import RMSNorm as RMSNormTE
from transformer_engine.pytorch.module.rmsnorm import _RMSNorm

from cosmos_predict1.utils import log


def get_batch_on_this_cp_rank(inputs):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = parallel_state.get_context_parallel_world_size()

    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        seq_dim = 1  # if key != 'attention_mask' else 2
        inputs = inputs.view(
            *inputs.shape[0:seq_dim],
            2 * cp_size,
            inputs.shape[seq_dim] // (2 * cp_size),
            *inputs.shape[(seq_dim + 1) :],
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        inputs = inputs.index_select(seq_dim, index)
        inputs = inputs.view(*inputs.shape[0:seq_dim], -1, *inputs.shape[(seq_dim + 2) :])

    return inputs


def gather_batch_from_cp_ranks(outputs):
    """
    Gather and reconstruct the full batch from chunks distributed across GPUs in a context parallel group.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if cp_size > 1:
        seq_dim = 1  # Assuming sequence dimension is 1

        try:
            # Reshape output to separate the two chunks
            chunk_size = outputs.shape[seq_dim] // 2
            outputs = outputs.view(*outputs.shape[:seq_dim], 2, chunk_size, *outputs.shape[seq_dim + 1 :])

            # Prepare a list to gather all chunks from all ranks
            gathered_chunks = [torch.zeros_like(outputs) for _ in range(cp_size)]

            # Gather all chunks
            dist.barrier()
            dist.all_gather(gathered_chunks, outputs, group=parallel_state.get_context_parallel_group())
            dist.barrier()

            # Reorder chunks
            reordered_chunks = [None] * (2 * cp_size)
            for i in range(cp_size):
                reordered_chunks[i] = gathered_chunks[i].select(seq_dim, 0)
                reordered_chunks[2 * cp_size - 1 - i] = gathered_chunks[i].select(seq_dim, 1)

            # Concatenate all chunks
            outputs = torch.cat(reordered_chunks, dim=seq_dim)
        except Exception as e:
            log.info(f"[Rank {cp_rank}] Error in gather_batch_from_cp_ranks: {str(e)}")
            raise

    return outputs


def broadcast_data_batch_in_tp_cp_group(data_batch):
    """
    Broadcast data batch across tensor model parallel and context parallel groups.
    """
    keys = sorted(data_batch.keys())
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group() if tp_size > 1 else None
    cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
    tp_ranks = get_process_group_ranks(tp_group) if tp_size > 1 else None
    cp_ranks = get_process_group_ranks(cp_group) if cp_size > 1 else None
    if tp_size > 1 or cp_size > 1:
        for key in keys:
            tensor = data_batch[key]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.contiguous()
                if tp_size > 1:
                    broadcast(tensor, min(tp_ranks), group=tp_group)
                if cp_size > 1:
                    broadcast(tensor, min(cp_ranks), group=cp_group)


def allreduce_layernorm_grads(model: List[torch.nn.Module], tensor_model_parallel_size: int, sequence_parallel: bool):
    """
    All-reduce layernorm grads (for sequence parallelism).
    Note:
    - We skip QK Normalization layers and the last normalization layer of Transformer,
      since we use AllReduceBWDRMSNormTE for these layers, which already applies all-reduce in the backward pass.
    - TransformerEngine's LayernormLinear and LayernormMLP modules have `*.layer_norm_weight` parameters that
      we must all-reduce in the backward pass as well. So we implement this function to cover these parameters.
    """
    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if tensor_model_parallel_size > 1 and sequence_parallel:
        grads = []
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith(".layer_norm_weight"):  # TP  # Q-layernorm  # K-layernorm
                    grad = param.grad
                    if grad is not None:
                        grads.append(grad.data)

        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


def sync_1d_parameters(model: torch.nn.Module, process_group=None):
    """
    Synchronize layernorm parameters (1D) across ranks by performing all-reduce with mean operation.
    LayerNorm parameters are identified by having ndim==1.
    Note: If parameters other than LayerNorm are 1D, they will also be synchronized.

    Args:
        model (torch.nn.Module): The model containing layernorm parameters
        process_group (optional): The process group to perform all-reduce.
                                If None, uses the default process group.
    """
    if not torch.distributed.is_initialized():
        return
    # Synchronize each 1D parameter (layernorm parameters)
    for name, param in model.named_parameters():
        if param.ndim == 1 and param.requires_grad:  # LayerNorm weights/biases are 1D
            torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.AVG, group=process_group)


class AllReduceBWD(Function):
    """
    Custom autograd Function that performs an all-reduce operation during the backward pass.

    Args:
        tensor (Tensor): The input tensor.
        process_group: The process group to perform the all-reduce operation.

    Returns:
        Tensor: The input tensor in the forward pass, and the all-reduced gradient in the backward pass.
    """

    @staticmethod
    def forward(ctx, tensor, process_group):
        ctx.process_group = process_group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class AllReduceBWDRMSNormTE(RMSNormTE):
    """
    A custom RMSNorm layer that applies all-reduce operation during backward pass.
    Used in tensor parallel training with Transformer Engine.

    Args:
        hidden_size (int): The size of the hidden dimension.
        process_group: Megatron Core's process group.
        **kwargs: Additional arguments to be passed to RMSNormTE.
    """

    def __init__(self, hidden_size, process_group, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.process_group = process_group

    @no_torch_dynamo()
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """RMSNorm FWD"""

        # Set the activation type for AMP.
        TransformerEngineBaseModule.set_activation_dtype(self, inp)

        if torch.is_grad_enabled():
            fwd_fn = _RMSNorm.apply
            args = []
        else:
            fwd_fn = _RMSNorm.forward
            args = [None]

        args += (
            inp,
            AllReduceBWD.apply(self.weight, self.process_group),
            self.eps,
            self.fwd_rmsnorm_sm_margin,
            self.bwd_rmsnorm_sm_margin,
            self.inf_rmsnorm_sm_margin,
            self.zero_centered_gamma,
            torch.is_grad_enabled(),
            self.activation_dtype,
        )

        return fwd_fn(*args)
