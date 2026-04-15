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

from typing import Callable

import torch
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear as McoreColumnParallelLinear
from megatron.core.tensor_parallel import RowParallelLinear as McoreRowParallelLinear
from megatron.core.tensor_parallel import VocabParallelEmbedding as McoreVocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from torch.distributed import _functional_collectives as funcol
from torch.distributed._functional_collectives import all_reduce


class VocabParallelEmbedding(torch.nn.Module):
    """
    Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        precision (str): precision of the embedding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        precision: str = "bfloat16",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        (self.vocab_start_index, self.vocab_end_index) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings,
            parallel_state.get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = torch.nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=getattr(torch, precision),
            )
        )

    def forward(self, input_):
        """Forward.

        Args:
            input_ (torch.Tensor): Input tensor.
        """
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        output = self.weight[masked_input]
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output[input_mask, :] = 0.0

        output = all_reduce(output, "sum", group=parallel_state.get_tensor_model_parallel_group())
        return output


class ColumnParallelLinear(McoreColumnParallelLinear):
    """
    A modified version of Mcore's ColumnParallelLinear that only returns the output tensor.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor):
        """
        Performs the forward pass of the column parallel linear layer.

        Args:
            input_ (torch.Tensor): The input tensor.
            weight (Optional[torch.Tensor], optional): The weight tensor. If None, uses the layer's own weight.

        Returns:
            torch.Tensor: The output tensor after the linear transformation.
        """
        output, _ = super().forward(input_)
        return output


class RowParallelLinear(McoreRowParallelLinear):
    """
    A modified version of Mcore's RowParallelLinear that only returns the output tensor.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor):
        """
        Performs the forward pass of the Row Parallel linear layer.

        Args:
            input_ (torch.Tensor): The input tensor.
            weight (Optional[torch.Tensor], optional): The weight tensor. If None, uses the layer's own weight.

        Returns:
            torch.Tensor: The output tensor after the linear transformation.
        """
        output, _ = super().forward(input_)
        return output


class TrainingVocabParallelEmbedding(McoreVocabParallelEmbedding):
    """
    Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.

    Keyword Args:
        sequence_parallel (bool): Decides whether to perform ReduceScatter after embedding lookup
        batch_first (bool): If True, then output tensor shape is [batch, seq, feature]. If False, then shape becomes
            [seq, batch, feature]. Note: We assume the input tensor is always in the shape of [seq, batch].
        config: A megatron.core.ModelParallelConfig object
        use_inference_allreduce (bool): If True, then Megatron's allreduce in the forward pass is disabled, and the pytorch's
            allreduce is used instead (inference mode only).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        sequence_parallel: bool = False,
        batch_first: bool = False,
        config: ModelParallelConfig,
        use_inference_allreduce: bool = False,
    ):
        super(TrainingVocabParallelEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_method=init_method,
            config=config,
        )
        self.sequence_parallel = sequence_parallel
        if sequence_parallel:
            # If sequence parallel, then the output tensor should be in the shape of [seq, batch, feature]
            batch_first = False
        self.batch_first = batch_first
        self.use_inference_allreduce = use_inference_allreduce

    def forward(self, input_):
        """Forward.

        Args:
            input_ (torch.Tensor): Input tensor.
        """
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        output = self.weight[masked_input]
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output[input_mask, :] = 0.0

        if self.sequence_parallel:
            assert not self.batch_first
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            output = output.transpose(0, 1).contiguous()
            if not self.use_inference_allreduce:
                output = reduce_scatter_to_sequence_parallel_region(output)
        else:
            # Reduce across all the model parallel GPUs.
            if not self.use_inference_allreduce:
                output = reduce_from_tensor_model_parallel_region(output)
            if not self.batch_first:
                # Shape: [b, s, h] --> [s, b, h]
                output = output.transpose(0, 1).contiguous()

        if self.use_inference_allreduce:
            output = funcol.all_reduce(output, "sum", group=parallel_state.get_tensor_model_parallel_group())
        return output
