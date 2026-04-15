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

import os

import torch
import torch.distributed as dist
from torch.autograd import Function


class AllGather(Function):
    @staticmethod
    def forward(ctx, tensor, process_group):
        world_size = dist.get_world_size(process_group)
        ctx.world_size = world_size
        ctx.rank = process_group.rank()

        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor.contiguous(), process_group)
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        rank = ctx.rank

        # Split the gradient tensor
        grad_chunks = grad_output.chunk(world_size)

        # Select the gradient chunk for the current rank
        grad_input = grad_chunks[rank]
        return grad_input, None


def gather_along_first_dim(tensor, process_group):
    return AllGather.apply(tensor, process_group)


class Scatter(Function):
    @staticmethod
    def forward(ctx, tensor, process_group):
        world_size = dist.get_world_size(process_group)
        ctx.world_size = world_size
        ctx.process_group = process_group
        rank = process_group.rank()

        # Split the tensor
        tensor_chunks = tensor.chunk(world_size)

        # Select the tensor chunk for the current rank
        return tensor_chunks[rank]

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        process_group = ctx.process_group

        # Gather the gradient tensor
        gathered_grads = [torch.zeros_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gathered_grads, grad_output.contiguous(), process_group)
        return torch.cat(gathered_grads, dim=0), None


def scatter_along_first_dim(tensor, process_group):
    return Scatter.apply(tensor, process_group)


if __name__ == "__main__":
    # Torch global setup for distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Create a tensor with gradients
    x = torch.randn(10, 1, requires_grad=True, device="cuda")

    # Perform all_gather with gradient support
    y = gather_along_first_dim(x, dist.group.WORLD)
    print(f"{y.shape=}")
    y = scatter_along_first_dim(y, dist.group.WORLD)
    print(f"{y.shape=}")

    # Use the result in your computation
    loss = y.sum()
    loss.backward()

    # x.grad now contains the gradients
    print(x.grad)
