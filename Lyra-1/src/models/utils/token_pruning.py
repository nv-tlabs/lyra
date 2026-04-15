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
import torch.nn.functional as F
from typing import List, Tuple, Optional

# Sample Gumbel noise for Gumbel-Softmax trick
def sample_gumbel(shape: torch.Size, eps: float = 1e-6, device=None, dtype=None) -> torch.Tensor:
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U.clamp(min=eps, max=1 - eps)))

# Select the top-k or softmax probabilities (with Gumbel-Softmax for differentiability)
def select_topk(
    logits: torch.Tensor,
    k: int,
    method: str,
    temperature: float,
    hard: bool,
    eps: float
) -> torch.Tensor:
    B, N = logits.shape

    if method == 'topk':
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, topk_idx, 1.0)
    elif method == 'softmax':
        gumbel_noise = sample_gumbel(logits.shape, eps=eps, device=logits.device, dtype=logits.dtype)
        y = (logits + gumbel_noise) / temperature
        y_soft = F.softmax(y, dim=-1)

        if hard:
            topk_idx = y_soft.topk(k, dim=-1).indices
            hard_mask = torch.zeros_like(y_soft).scatter(-1, topk_idx, 1.0)
            mask = hard_mask - y_soft.detach() + y_soft
        else:
            mask = y_soft
    else:
        raise ValueError(f"Unknown method: {method}")

    return mask

# Perform global selection of k tokens across the entire tensor
def global_selection(
    mask_logits: torch.Tensor,  # (B, T, H, W)
    total_k: int,
    method: str,
    temperature: float,
    hard: bool,
    eps: float
) -> torch.Tensor:
    B, T, H, W = mask_logits.shape
    N = T * H * W
    logits_flat = mask_logits.reshape(B, N)
    mask_flat = select_topk(logits_flat, total_k, method, temperature, hard, eps)
    mask = mask_flat.reshape(B, T, H, W)
    return mask

# Perform structured selection: k_t and k_hw tokens independently for time and space
def structured_selection(
    mask_logits: torch.Tensor,  # (B, T, H, W)
    k_t: int,
    k_hw: int,
    method: str,
    temperature: float,
    hard: bool,
    eps: float
) -> torch.Tensor:
    B, T, H, W = mask_logits.shape

    # Temporal selection
    logits_t = mask_logits.mean(dim=[2, 3])  # (B, T)
    mask_t = select_topk(logits_t, k_t, method, temperature, hard, eps)  # (B, T)

    # Spatial selection per frame
    mask_spatial = []
    for b in range(B):
        mask_b = []
        for t in range(T):
            logits_hw = mask_logits[b, t].reshape(-1)  # (H*W,)
            mask_hw = select_topk(logits_hw.unsqueeze(0), k_hw, method, temperature, hard, eps)
            mask_b.append(mask_hw.reshape(H, W))
        mask_b = torch.stack(mask_b, dim=0)  # (T, H, W)
        mask_spatial.append(mask_b)
    mask_spatial = torch.stack(mask_spatial, dim=0)  # (B, T, H, W)

    # Combine temporal and spatial
    mask = mask_spatial * mask_t.unsqueeze(-1).unsqueeze(-1)  # (B, T, H, W)
    return mask

# Apply the mask and select tokens and other tensors based on the selected indices
def apply_mask_and_select(
    tokens: torch.Tensor,             # (B, C, T, H, W)
    other_tensors: List[torch.Tensor],
    mask: torch.Tensor               # (B, T, H, W)
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    B, C, T, H, W = tokens.shape
    N = T * H * W

    tokens_flat = tokens.reshape(B, C, N)
    mask_flat = mask.reshape(B, N)

    selected_tokens = []
    selected_others = [[] for _ in other_tensors]

    for b in range(B):
        idx = mask_flat[b].nonzero(as_tuple=False).squeeze(-1)
        selected_tokens.append(tokens_flat[b, :, idx])

        for i, t in enumerate(other_tensors):
            t_flat = t.reshape(B, -1, N)
            selected = t_flat[b, :, idx]
            selected_others[i].append(selected)

    tokens_out = torch.stack(selected_tokens, dim=0)  # (B, C, k)
    others_out = [torch.stack(x, dim=0) for x in selected_others]  # list of (B, C_other, k)

    return tokens_out, others_out

# Main process function to prune tokens and other tensors based on mask logits
def process_tensors(
    tokens: torch.Tensor,           # (B, C, T, H, W)
    mask_logits: torch.Tensor,      # (B, 1, T, H, W)
    other_tensors: List[torch.Tensor],
    total_k: Optional[int] = None,
    k_t: Optional[int] = None,
    k_hw: Optional[int] = None,
    temperature: float = 1.0,
    eps: float = 1e-6,
    training: bool = True,
    soft_inference: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    If training=True -> uses softmax + hard sampling (Gumbel-Softmax trick)
    If training=False -> uses topk (non-differentiable)
    """
    B, C, T, H, W = tokens.shape
    mask_logits = mask_logits.squeeze(1)  # (B, T, H, W)

    if training or soft_inference:
        method = 'softmax'
        hard = True
    else:
        method = 'topk'
        hard = False  # ignored in topk mode

    if total_k is not None:
        mask = global_selection(mask_logits, total_k, method, temperature, hard, eps)
    elif k_t is not None and k_hw is not None:
        mask = structured_selection(mask_logits, k_t, k_hw, method, temperature, hard, eps)
    else:
        raise ValueError("Provide either total_k or both k_t and k_hw.")
    tokens_out, others_out = apply_mask_and_select(tokens, other_tensors, mask)
    return tokens_out, others_out, mask


if __name__ == '__main__':
    # Case 1: Structured pruning (select 60 frames and 1/4 spatial tokens)
    temperature = 1.0
    training = False
    k_t = 9
    k_h = 90
    k_w = 160
    B, T, C, H, W = 2, 17, 3, 180, 320
    tokens = torch.randn(B, C, T, H, W)
    mask_logits = torch.randn(B, 1, T, H, W)

    # Other tensors with different channels
    other1 = torch.randn(B, 6, T, H, W)
    other2 = torch.randn(B, 9, T, H, W)
    
    tokens_out, others_out = process_tensors(
        tokens=tokens,
        mask_logits=mask_logits,
        other_tensors=[other1, other2],
        k_t=k_t,                       # select 60 frames out of 121
        k_hw=k_h * k_w,               # select 1/4 spatial tokens (since 720x1280 is 2*2)
        temperature=temperature,
        training=training,  # differentiable Gumbel-Softmax
    )
    # Case 2: Global total pruning (select k tokens jointly across T and HW)
    tokens_out, others_out = process_tensors(
        tokens=tokens,
        mask_logits=mask_logits,
        other_tensors=[other1, other2],
        total_k=k_t * k_h * k_w,       # select k tokens globally (joint T and HW selection)
        temperature=temperature,
        training=training,  # inference: real top-k selection
    )
