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
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from cosmos_predict1.diffusion.module.attention import Attention
from cosmos_predict1.diffusion.training.utils.peft.lora_net import LoRALinearLayer, TELoRALinearLayer
from cosmos_predict1.diffusion.utils.customization.customization_manager import CustomizationType

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False


def enable_attn_lora(attn: Attention, peft_control: dict) -> None:
    """
    Enable LoRA for the attention block based on the peft_control dictionary.

    Args:
        attn (Attention): The attention block to configure.
        peft_control (dict): Dictionary containing PEFT configuration.
    """
    attn.peft_lora_enabled = False
    if peft_control:
        try:
            if peft_control["customization_type"] == CustomizationType.LORA:
                attn.peft_lora_enabled = True
            else:
                raise Exception(f"Unsupported Customization type {peft_control['customization_type']}")
        except KeyError as e:
            raise KeyError(f"peft_control dictionary expected to have attribute {e.args[0]}.")


def configure_attn_lora(attn: Attention, peft_control: dict) -> None:
    """
    Configure LoRA for the attention block based on the peft_control dictionary.

    Args:
        attn (Attention): The attention block to configure.
        peft_control (dict): Dictionary containing PEFT configuration.
    """
    try:
        attn.q_lora_enabled = peft_control.get("to_q", {}).get("activate", False)
        attn.k_lora_enabled = peft_control.get("to_k", {}).get("activate", False)
        attn.v_lora_enabled = peft_control.get("to_v", {}).get("activate", False)
        attn.out_lora_enabled = peft_control.get("to_out", {}).get("activate", False)
        if attn.q_lora_enabled:
            attn.q_lora_rank = peft_control["to_q"]["lora_rank"]
            attn.q_lora_scale = float(peft_control["to_q"]["lora_scale"])
        if attn.k_lora_enabled:
            attn.k_lora_rank = peft_control["to_k"]["lora_rank"]
            attn.k_lora_scale = float(peft_control["to_k"]["lora_scale"])
        if attn.v_lora_enabled:
            attn.v_lora_rank = peft_control["to_v"]["lora_rank"]
            attn.v_lora_scale = float(peft_control["to_v"]["lora_scale"])
        if attn.out_lora_enabled:
            attn.out_lora_rank = peft_control["to_out"]["lora_rank"]
            attn.out_lora_scale = float(peft_control["to_out"]["lora_scale"])
    except KeyError as e:
        raise KeyError(f"All layers (to_q, etc) specified must have attribute {e.args[0]}.")
    except ValueError as e:
        raise ValueError(f"Could not convert string to float: {e}")


def cal_qkv_lora(
    self,
    x: torch.Tensor,
    context: torch.Tensor = None,
    mask: torch.Tensor = None,
    rope_emb: torch.Tensor = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del kwargs
    """
    Calculate the Q, K, V matrices with LoRA adjustments. Derived from cosmos_predict1/diffusion/module/attention.py cal_qkv.

    Args:
        x (torch.Tensor): Input tensor.
        context (torch.Tensor, optional): Context tensor
        mask (torch.Tensor, optional): Mask tensor
        rope_emb (torch.Tensor, optional): Rotary positional embedding

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The Q, K, V matrices.
    """

    q = self.to_q[0](x)
    context = x if context is None else context
    k = self.to_k[0](context)
    v = self.to_v[0](context)

    if self.peft_lora_enabled:
        try:
            if self.q_lora_enabled:
                q_lora = self.to_q_lora(x)
                q = q + self.q_lora_scale * q_lora
            if self.k_lora_enabled:
                k_lora = self.to_k_lora(context)
                k = k + self.k_lora_scale * k_lora
            if self.v_lora_enabled:
                v_lora = self.to_v_lora(context)
                v = v + self.v_lora_scale * v_lora
        except AttributeError as e:
            raise AttributeError(f"lora enabled, but missing class attribute {e.args[0]} of Attention block")

    q, k, v = map(
        lambda t: rearrange(t, "b ... (n c) -> b ... n c", n=self.heads // self.tp_size, c=self.dim_head),
        (q, k, v),
    )

    def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
        return q, k, v

    q, k, v = checkpoint(apply_norm_and_rotary_pos_emb, q, k, v, rope_emb, use_reentrant=False)

    return q, k, v


def cal_attn_lora(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the attention output with LoRA adjustments. Derived from cosmos_predict1/diffusion/module/attention.py cal_attn.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        mask (torch.Tensor, optional): Mask tensor.

    Returns:
        torch.Tensor: The attention output.
    """
    if self.backend == "transformer_engine":
        seq_dim = self.qkv_format.index("s")
        assert (
            q.shape[seq_dim] > 1 and k.shape[seq_dim] > 1
        ), "Seqlen must be larger than 1 for TE Attention starting with 1.8 TE version."
        attn_out = self.attn_op(q, k, v, core_attention_bias_type="no_bias", core_attention_bias=None)  # [B, Mq, H, V]
        out = self.to_out(attn_out)

        if self.peft_lora_enabled and self.out_lora_enabled:
            try:
                out_lora = self.to_out_lora(attn_out)
                out = out + self.out_lora_scale * out_lora
            except AttributeError as e:
                raise AttributeError(f"l1 lora enabled, but missing class attribute {e.args[0]} of FeedForward block")

        return out
    elif self.backend == "torch":
        attn_out = self.attn_op(q, k, v, mask=mask)  # [B, Mq, H, V]
        attn_out = rearrange(attn_out, " b ... n c -> b ... (n c)")
        out = self.to_out(attn_out)

        if self.peft_lora_enabled and self.out_lora_enabled:
            try:
                out_lora = self.to_out_lora(attn_out)
                out = out + self.out_lora_scale * out_lora
            except AttributeError as e:
                raise AttributeError(f"l1 lora enabled, but missing class attribute {e.args[0]} of FeedForward block")

        return out
    else:
        raise ValueError(f"Backend {self.backend} not found")


def build_attn_lora(attn: Attention, peft_control: dict) -> None:
    """
    Configure, build and add LoRA layers to the attention block.

    Args:
        attn (Attention): The attention block to add LoRA layers to.
        peft_control (dict): Dictionary containing PEFT configuration.
    """
    enable_attn_lora(attn, peft_control)
    configure_attn_lora(attn, peft_control)
    if attn.peft_lora_enabled:
        query_dim = attn.query_dim
        inner_dim = attn.inner_dim
        context_dim = attn.context_dim
        tp_group = parallel_state.get_tensor_model_parallel_group(check_initialized=False) if USE_MEGATRON else None

        if attn.tp_size == 1:
            if attn.q_lora_enabled:
                attn.to_q_lora = LoRALinearLayer(query_dim, inner_dim, rank=attn.q_lora_rank, linear=True)
            if attn.k_lora_enabled:
                attn.to_k_lora = LoRALinearLayer(context_dim, inner_dim, rank=attn.k_lora_rank, linear=True)
            if attn.v_lora_enabled:
                attn.to_v_lora = LoRALinearLayer(context_dim, inner_dim, rank=attn.v_lora_rank, linear=True)
            if attn.out_lora_enabled:
                attn.to_out_lora = LoRALinearLayer(inner_dim, query_dim, rank=attn.out_lora_rank, linear=True)
        else:
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
            if attn.q_lora_enabled:
                attn.to_q_lora = TELoRALinearLayer(
                    query_dim,
                    inner_dim,
                    rank=attn.q_lora_rank,
                    linear=True,
                    tp_size=attn.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                )
            if attn.k_lora_enabled:
                attn.to_k_lora = TELoRALinearLayer(
                    context_dim,
                    inner_dim,
                    rank=attn.k_lora_rank,
                    linear=True,
                    tp_size=attn.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                )
            if attn.v_lora_enabled:
                attn.to_v_lora = TELoRALinearLayer(
                    context_dim,
                    inner_dim,
                    rank=attn.v_lora_rank,
                    linear=True,
                    tp_size=attn.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="column",
                )
            if attn.out_lora_enabled:
                attn.to_out_lora = TELoRALinearLayer(
                    inner_dim,
                    query_dim,
                    rank=attn.out_lora_rank,
                    linear=True,
                    tp_size=attn.tp_size,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    parallel_mode="row",
                )
    attn.cal_qkv = cal_qkv_lora.__get__(attn, attn.__class__)
    attn.cal_attn = cal_attn_lora.__get__(attn, attn.__class__)
