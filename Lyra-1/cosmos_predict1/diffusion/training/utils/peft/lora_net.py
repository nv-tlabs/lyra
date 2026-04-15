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
import transformer_engine as te
from megatron.core import parallel_state
from torch import nn

from cosmos_predict1.utils import log


class LoRALinearLayer(nn.Module):
    """
    ported from
    https://github.com/huggingface/diffusers/blob/7a32b6beeb0cfdefed645253dce23d9b0a78597f/src/diffusers/models/attention_processor.py#L470.
    """

    def __init__(self, in_features, out_features, rank=4, linear=False):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        if linear:
            down = nn.Linear(in_features, rank, bias=False)
            up = nn.Linear(rank, out_features, bias=False)
        else:
            down = nn.Conv1d(in_features, rank, 1, bias=False)
            up = nn.Conv1d(rank, out_features, 1, bias=False)

        nn.init.normal_(down.weight, std=1 / rank)
        nn.init.zeros_(up.weight)
        self.net = nn.Sequential(down, up)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.net[0].weight.dtype

        up_hidden_states = self.net(hidden_states.to(dtype))

        return up_hidden_states.to(orig_dtype)


class TELoRALinearLayer(nn.Module):
    """
    ported from
    https://github.com/huggingface/diffusers/blob/7a32b6beeb0cfdefed645253dce23d9b0a78597f/src/diffusers/models/attention_processor.py#L470.
    """

    def __init__(self, in_features, out_features, rank, linear, tp_size, tp_group, sequence_parallel, parallel_mode):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        if linear:
            down = te.pytorch.Linear(
                in_features,
                rank,
                bias=False,
                tp_size=1,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
                parallel_mode=None,
            )
            up = te.pytorch.Linear(
                rank,
                out_features,
                bias=False,
                tp_size=tp_size,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
                parallel_mode=parallel_mode,
            )
        else:
            down = te.pytorch.Conv1d(
                in_features,
                rank,
                1,
                bias=False,
                tp_size=1,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
                parallel_mode=None,
            )
            up = te.pytorch.Conv1d(
                rank,
                out_features,
                1,
                bias=False,
                tp_size=tp_size,
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
                parallel_mode=parallel_mode,
            )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        # Create generator
        gen = torch.Generator(device=down.weight.device)
        # Save the current random state
        gen_state = gen.get_state()

        # Set constant seed for non-tp layers
        log.info(f"rank {tp_rank}: setting seed to 0")
        gen.manual_seed(0)
        nn.init.normal_(down.weight, std=1 / rank, generator=gen)
        # Set a new random seed based on the tensor parallel rank
        gen.manual_seed(tp_rank)
        log.info(f"rank {tp_rank}: setting seed to {tp_rank}")
        nn.init.zeros_(up.weight)
        # Restore the original random state
        gen.set_state(gen_state)

        self.net = nn.Sequential(down, up)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.net[0].weight.dtype
        up_hidden_states = self.net(hidden_states.to(dtype))

        return up_hidden_states.to(orig_dtype)
