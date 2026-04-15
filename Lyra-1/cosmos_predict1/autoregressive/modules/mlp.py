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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import ModelParallelConfig, parallel_state
from torch.distributed import _functional_collectives as funcol
from torch.distributed._functional_collectives import all_reduce

from cosmos_predict1.autoregressive.modules.linear import ColumnParallelLinear, RowParallelLinear


def compute_llama3_ffn_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: float) -> int:
    """
    Computes the feedforward network dimensionality.

    Args:
        dim (int): The embedding dimensionality.
        multiple_of (int): The multiple to round up the hidden dimensionality.
        ffn_dim_multiplier (float): The multiplier for the hidden dimensionality.

    Returns:
        The feedforward network dimensionality.
    """
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)  # custom dim factor
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    # Round up hidden dimensionality to the nearest multiple
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        tensor_parallel_size: int = 1,
    ):
        """
        Initializes the multilayer perceptron (MLP) module.

        Args:
            dim: The input and output dimensionality.
            hidden_dim: The dimensionality of the hidden layer.
        """
        super().__init__()
        self.tp_size = tensor_parallel_size
        self.w1 = nn.Linear(dim, hidden_dim // self.tp_size, bias=False)
        self.w2 = nn.Linear(hidden_dim // self.tp_size, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim // self.tp_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP module.

        Args:
            x: The input tensor of shape (batch_size, dim).

        Returns:
            The output tensor of shape (batch_size, dim).
        """
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.tp_size > 1:
            output = all_reduce(output, "sum", group=parallel_state.get_tensor_model_parallel_group())
        return output


class TrainingMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_dropout: float = 0.0,
        set_parallel_mode: bool = False,
        model_parallel: Optional[ModelParallelConfig] = None,
        inference: bool = False,
    ):
        """
        Initializes the multilayer perceptron (MLP) module.

        Args:
            dim: The input and output dimensionality.
            hidden_dim: The dimensionality of the hidden layer.
            hidden_dropout: Dropout after the attention and feed-forward layers (following TransformerEngine's
                implementation in its TransformerLayer class).
            set_parallel_mode: Whether to use column and row parallel linear layers.
            model_parallel: The model parallel configuration.
            inference: Whether the model is used for inference.
        """
        super().__init__()
        self.hidden_dropout = hidden_dropout
        if model_parallel and model_parallel.tensor_model_parallel_size > 1:
            self.tp_size = model_parallel.tensor_model_parallel_size
        else:
            self.tp_size = 1
        if set_parallel_mode and not inference:
            kwargs = {"bias": False, "init_method": lambda x: x, "config": model_parallel}
            # Using column and row parallel linear layers
            self.w1 = ColumnParallelLinear(dim, hidden_dim, gather_output=False, **kwargs)
            self.w2 = RowParallelLinear(hidden_dim, dim, input_is_parallel=True, skip_bias_add=True, **kwargs)
            self.w3 = ColumnParallelLinear(dim, hidden_dim, gather_output=False, **kwargs)
        else:
            self.w1 = nn.Linear(dim, hidden_dim // self.tp_size, bias=False)
            self.w2 = nn.Linear(hidden_dim // self.tp_size, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim // self.tp_size, bias=False)

        self.inference = inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP module.

        Args:
            x: The input tensor of shape (batch_size, dim).

        Returns:
            The output tensor of shape (batch_size, dim).
        """
        x = F.dropout(x, p=self.hidden_dropout, training=self.training)
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        output = F.dropout(output, p=self.hidden_dropout, training=self.training)

        if self.inference and self.tp_size > 1:
            output = funcol.all_reduce(output, "sum", group=parallel_state.get_tensor_model_parallel_group())
        return output

    def init_weights(self, init_std: float):
        """
        Initializes the weights of the MLP module.
        """
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
