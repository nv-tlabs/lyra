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
from typing import Any, Optional, Tuple, Union

import torch
from megatron.core import ModelParallelConfig, parallel_state
from torch import nn
from torch.distributed import _functional_collectives as funcol
from transformer_engine.pytorch.attention import _SplitAlongDim, apply_rotary_pos_emb, check_set_window_size
from transformer_engine.pytorch.constants import AttnBiasTypes
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.module.linear import Linear as LinearTE
from transformer_engine.pytorch.module.rmsnorm import RMSNorm as RMSNormTE

from cosmos_predict1.autoregressive.modules.embedding import RotaryPositionEmbedding
from cosmos_predict1.autoregressive.modules.linear import ColumnParallelLinear, RowParallelLinear
from cosmos_predict1.autoregressive.modules.normalization import create_norm
from cosmos_predict1.autoregressive.utils.parallel import AllReduceBWDRMSNormTE


class GQA(nn.Module):
    """
    Grouped Query Attention (GQA) with KV cache (only supported for inference).
    """

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: Union[int, None],
        dim: int,
        max_batch_size: int,
        max_seq_len: int,
        context_dim: Optional[int] = None,
        inference: bool = True,
        flash_attn: bool = True,
        use_qk_normalization: bool = False,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        set_parallel_mode: Optional[bool] = False,
        model_parallel: Optional[ModelParallelConfig] = None,
        attention_tp: Optional[bool] = False,
        causal_mask: Optional[bool] = True,
        head_dim: Optional[int] = None,
        fuse_qkv: bool = False,
        precision: str = "bfloat16",
        attention_type: str = "self",
    ):
        """
        Initializes the GQA module.

        Args:
            n_heads (int): The number of attention heads.
            n_kv_heads (int, optional): The number of key-value attention heads. None defaults to n_heads.
            dim (int): The dimensionality of the input and output.
            max_batch_size (int): The maximum batch size.
            max_seq_len (int): The maximum sequence length.
            context_dim (int, optional): The dimensionality of the context for cross-attn. Defaults to None.
            inference (bool, optional): Whether the model is in inference mode. Defaults to True.
            flash_attn (bool, optional): Whether to use Flash attention. Defaults to True.
            use_qk_normalization (bool, optional): Whether to apply QK normalization. Defaults to False.
            norm_type (str, optional): The type of normalization layer. Defaults to "rmsnorm".
            norm_eps (float, optional): The epsilon value for normalization. Defaults to 1e-5.
            attention_dropout (float, optional): Dropout rate for attention. Defaults to 0.0.
            tp_group (int, optional): The tensor parallel group.
            set_parallel_mode (bool, optional): Whether to set parallel mode which enables parallel linear. Defaults to False.
            model_parallel (ModelParallelConfig, optional): The Megatron model parallel configuration.
            attention_tp (bool, optional): Whether to use tensor parallelism for attention layers. Defaults to False.
            causal_mask (bool, optional): Whether to use causal mask. Defaults to True.
            head_dim (int, optional): The dimensionality of each attention head. If None, defaults to dim // n_heads.
            fuse_qkv (bool, optional): Whether to fuse QKV projections. Defaults to False.
            precision (str, optional): The precision of the model. Defaults to "bfloat16".
            attention_type (str, optional): The type of attention. Defaults to "self".
        """
        super().__init__()
        assert attention_type in ["self", "cross", "full"], f"Invalid attention type: {attention_type}"
        self.attention_type = attention_type
        self.model_parallel = model_parallel
        if self.model_parallel and self.model_parallel.tensor_model_parallel_size > 1 and attention_tp:
            self.tp_size = self.model_parallel.tensor_model_parallel_size
        else:
            self.tp_size = 1

        context_dim = dim if context_dim is None else context_dim

        self.dim = dim
        self.context_dim = context_dim
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads // self.tp_size
        self.n_local_heads = n_heads // self.tp_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads if head_dim is None else head_dim
        assert flash_attn, "Flash attention is required."
        self.attention_dropout = attention_dropout
        self.causal_mask = causal_mask
        self.fuse_qkv = fuse_qkv
        self.precision = precision

        if fuse_qkv:
            assert context_dim == dim, f"Fuse QKV requires context_dim ({context_dim}) to be equal to dim ({dim})"
            self.total_head_dim = (n_heads + 2 * self.n_kv_heads) * self.head_dim
            self.total_local_head_dim = (self.n_local_heads + 2 * self.n_local_kv_heads) * self.head_dim

        if set_parallel_mode and attention_tp and not inference:
            kwargs = {"bias": False, "init_method": lambda x: x, "config": self.model_parallel}
            # Using column and row parallel linear layers
            if fuse_qkv:
                self.wqkv = ColumnParallelLinear(dim, self.total_head_dim, **kwargs)
            else:
                self.wq = ColumnParallelLinear(dim, n_heads * self.head_dim, **kwargs)
                self.wk = ColumnParallelLinear(context_dim, self.n_kv_heads * self.head_dim, **kwargs)
                self.wv = ColumnParallelLinear(context_dim, self.n_kv_heads * self.head_dim, **kwargs)

            # Linear layer for output projection
            self.wo = RowParallelLinear(
                n_heads * self.head_dim, dim, input_is_parallel=True, skip_bias_add=True, **kwargs
            )

        else:
            # Linear layers for query, key, and value projections
            if fuse_qkv:
                self.wqkv = nn.Linear(dim, self.total_local_head_dim, bias=False)
            else:
                self.wq = nn.Linear(dim, self.n_local_heads * self.head_dim, bias=False)
                self.wk = nn.Linear(context_dim, self.n_local_kv_heads * self.head_dim, bias=False)
                self.wv = nn.Linear(context_dim, self.n_local_kv_heads * self.head_dim, bias=False)
            self.wo = nn.Linear(self.n_local_heads * self.head_dim, dim, bias=False)

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        if inference and self.attention_type == "self":
            # Cache for key and value tensors
            self.init_kv_cache()

        # QK normalization layers
        if use_qk_normalization:
            assert n_heads % self.tp_size == 0, "n_heads must be divisible by tensor_model_parallel_size"
            assert self.n_kv_heads % self.tp_size == 0, "n_kv_heads must be divisible by tensor_model_parallel_size"
            self.q_norm = create_norm(norm_type, dim=self.head_dim, eps=norm_eps)
            self.k_norm = create_norm(norm_type, dim=self.head_dim, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.use_qk_normalization = use_qk_normalization
        self.inference = inference

        if fuse_qkv:
            # Register hook to load fused QKV weights
            self._register_load_state_dict_pre_hook(self.load_hook)

        self.to(dtype=getattr(torch, self.precision))

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def init_kv_cache(self, dtype=None):
        cache_shape = (self.max_batch_size, self.n_local_kv_heads, self.max_seq_len, self.head_dim)
        if dtype is None:
            dtype = getattr(torch, self.precision)
        if self.attention_type == "self":
            self.cache_k = torch.zeros(cache_shape, dtype=dtype).cuda()
            self.cache_v = torch.zeros(cache_shape, dtype=dtype).cuda()

    def set_inference_flag(self, flag):
        self.inference = flag
        if flag and self.attention_type == "self":
            if self.cache_k is None or self.cache_v is None:
                self.init_kv_cache()

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbedding,
        input_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of GQA.

        Args:
            x: The input tensor of shape (batch_size, seq_len, dim).
            rope: The rotary positional embedding module.
            input_pos: The starting position of the current sequence.
            mask: The attention mask tensor.
            context: The context tensor of shape (batch_size, context_len, dim).

        Returns:
            The output tensor after applying GQA.
        """
        bsz, seqlen, _ = x.shape

        # Use one single module to handle both self-attn and cross-attn
        context = x if context is None else context
        context_len = seqlen if context is None else context.shape[1]

        if self.fuse_qkv:
            q_size = self.n_local_heads * self.head_dim
            kv_size = self.n_local_kv_heads * self.head_dim
            xq, xk, xv = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        else:
            # Compute query, key, and value projections
            xq = self.wq(x)
            xk, xv = self.wk(context), self.wv(context)

        # Reshape projections
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, context_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, context_len, self.n_local_kv_heads, self.head_dim)

        # QK normalization
        if self.use_qk_normalization:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Apply rotary positional embeddings to queries and keys
        # Only apply RoPE to self-attention!
        if self.attention_type in ["self", "full"]:
            xq, xk = rope(xq, xk, input_pos, seqlen)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        # xq: (bs, n_local_heads, seqlen, head_dim)
        # xk: (bs, n_kv_heads, cache_len + context_len, head_dim)
        # xv: (bs, n_kv_heads, cache_len + context_len, head_dim)
        if self.inference and self.attention_type == "self":
            # Update cache with current key and value tensors
            assert input_pos is not None
            self.cache_k[:bsz, :, input_pos] = xk
            self.cache_v[:bsz, :, input_pos] = xv
            keys, values = (
                self.cache_k[:bsz, :, :],
                self.cache_v[:bsz, :, :],
            )
        else:
            keys, values = xk, xv

        # Repeat keys and values if necessary
        keys = keys.repeat_interleave(self.n_rep, dim=1)  # (bs, n_local_heads, cache_len + context_len, head_dim)
        values = values.repeat_interleave(self.n_rep, dim=1)  # (bs, n_local_heads, cache_len + context_len, head_dim)

        if self.attention_type == "self" and self.causal_mask:
            # During inference, `is_causal` should be set to False when KV cache is pre-computed and used,
            # since the masking is handled outside this attention module.
            # During training, `is_causal` should be set to None to use the default behavior of FlashAttention.
            is_causal = False if self.inference else None
        else:
            # This is used for full-attention transformer (e.g., ViT)
            # also for the cross-attn, it's always full-attn w/o causal
            is_causal = False
        output = scaled_dot_product_attention(
            xq,
            keys,
            values,
            head_dim=self.head_dim,
            mask=mask,
            is_causal=is_causal,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.view(bsz, seqlen, -1)
        output = self.wo(output)

        if self.inference and self.tp_size > 1:
            output = funcol.all_reduce(output, "sum", group=parallel_state.get_tensor_model_parallel_group())
        return output

    def init_weights(self, init_std: float):
        """
        Initializes the weights of all modules.
        """
        if self.fuse_qkv:
            nn.init.trunc_normal_(self.wqkv.weight, mean=0.0, std=0.02)
        else:
            for linear in (self.wq, self.wk, self.wv):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        if self.use_qk_normalization:
            torch.nn.init.ones_(self.q_norm.weight)
            torch.nn.init.ones_(self.k_norm.weight)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    PyTorch's native implementation of Flash Attention 2.

    If `is_causal` is given, then the causal attention mask is applied accordingly:
    - If `is_causal` is True, the standard upper-left causal attention masking is applied.
    - If `is_causal` is False, no attention mask is applied, unless an explicit mask tensor is
      provided (i.e., `mask is not None`).

    If `is_causal` is not given (i.e., `is_causal is None`), then the attention mask is applied
    based on the provided mask tensor:
    - If no explicit attention mask is given (i.e., `mask is None`), `is_causal` is set to True,
    leading to the standard upper-left causal attention masking.
    - If an attention mask is given (i.e., `mask is not None`), the provided mask is used,
    and `is_causal` is set to False.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        head_dim (int): Dimension of each attention head
        mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
        is_causal (Optional[bool], optional): Whether to apply causal attention mask. Defaults to None.
        dropout_p (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor after applying scaled dot-product attention
    """
    scale = 1.0 / math.sqrt(head_dim)
    if is_causal is None:
        is_causal = mask is None
    y = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=scale,
        is_causal=is_causal,
    )
    return y.transpose(1, 2).contiguous()


def enable_different_context_dim_in_te_ca(
    te_mha_module,
    context_dim,
    args,
):
    """
    Hijacks the MultiheadAttention (MHA) module from TransformerEngine (TE) to use a different context-dim for KV calculation.
    """
    self = te_mha_module

    common_gemm_kwargs = {
        "fuse_wgrad_accumulation": args["fuse_wgrad_accumulation"],
        "tp_group": self.tp_group,
        "tp_size": self.tp_size,
        "get_rng_state_tracker": self.get_rng_state_tracker,
        "sequence_parallel": self.sequence_parallel,
        "params_dtype": self.params_dtype,
    }

    self.key_value = LinearTE(
        context_dim,
        2 * self.hidden_size_kv,
        init_method=None,
        bias=args["bias"],
        return_bias=False,
        parallel_mode="column" if args["set_parallel_mode"] else None,
        parameters_split=("key", "value") if not args["fuse_qkv_params"] else None,
        **common_gemm_kwargs,
    )


def enable_qk_normalization_in_te_mha(
    te_mha_module,
    norm_eps: float,
    is_self_attn: bool = True,
):
    """
    Hijacks the MultiheadAttention (MHA) module from TransformerEngine (TE) to use our `te_mha_forward_with_qk_norm`.
    The `te_mha_forward_with_qk_norm` function is just a copy of the TE MHA's forward function (source code at
    https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py) with the addition
    of several lines of code for the QK normalization operations.
    """
    self = te_mha_module

    # First, we add the QK norm layers (RMSNorm class) to the TE's MHA module in advance for our custom forward function.
    if is_self_attn:
        common_kwargs = dict(
            eps=norm_eps,
            device=self.layernorm_qkv.layer_norm_weight.device,
            sequence_parallel=self.layernorm_qkv.sequence_parallel,
            params_dtype=self.layernorm_qkv.layer_norm_weight.dtype,
            zero_centered_gamma=self.layernorm_qkv.zero_centered_gamma,
        )
    else:
        common_kwargs = dict(
            eps=norm_eps,
            device=self.layernorm_query.query_weight.device,
            sequence_parallel=self.layernorm_query.sequence_parallel,
            params_dtype=self.layernorm_query.query_weight.dtype,
            zero_centered_gamma=self.layernorm_query.zero_centered_gamma,
        )
    if parallel_state.model_parallel_is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        self.q_norm = AllReduceBWDRMSNormTE(
            self.hidden_size_per_attention_head, process_group=tp_group, **common_kwargs
        )
        self.k_norm = AllReduceBWDRMSNormTE(
            self.hidden_size_per_attention_head, process_group=tp_group, **common_kwargs
        )
    else:
        self.q_norm = RMSNormTE(self.hidden_size_per_attention_head, **common_kwargs)
        self.k_norm = RMSNormTE(self.hidden_size_per_attention_head, **common_kwargs)

    # Second, we define the custom forward function for the TE's MHA module, with the QK normalization operations.
    def te_mha_forward_with_qk_norm(
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[Any] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        fast_zero_fill: bool = True,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward propagation for MultiheadAttention layer.

        """
        # hidden_states: [sq, b, h]

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        if window_size is None:
            window_size = self.window_size
        window_size = check_set_window_size(attn_mask_type, window_size)

        if "padding" in attn_mask_type and attention_mask is not None:
            for mask in attention_mask:
                assert mask.dtype == torch.bool, "Attention mask must be in boolean type!"

        assert (
            core_attention_bias_type in AttnBiasTypes
        ), f"core_attention_bias_type {core_attention_bias_type} is not supported!"

        # =================================================
        # Pre-allocate memory for key-values for inference
        # =================================================

        if inference_params and self.layer_number is not None:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_length
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size, hidden_states.dtype)
                inference_value_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size, hidden_states.dtype)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                (
                    inference_key_memory,
                    inference_value_memory,
                ) = inference_params.key_value_memory_dict[self.layer_number]

        # ======================
        # Query, Key, and Value
        # ======================

        # fp8_mha = FP8GlobalStateManager.is_fp8_enabled() and FP8GlobalStateManager.get_fp8_recipe().fp8_mha
        # fp8_kwargs = {"fp8_output": fp8_mha and rotary_pos_emb is None}
        fp8_kwargs = {}

        layernorm_output = None
        if self.attention_type == "self":
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
            layernorm_qkv_outputs = self.layernorm_qkv(
                hidden_states, is_first_microbatch=is_first_microbatch, **fp8_kwargs
            )
            mixed_x_layer = layernorm_qkv_outputs

            num_queries_per_key_value = self.num_attention_heads_per_partition // self.num_gqa_groups_per_partition
            # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, (np/ng + 2), ng, hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                (num_queries_per_key_value + 2),
                self.num_gqa_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            # split along third last dimension
            split_dim = -3

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            #  [sq, b, (np/ng + 2), ng, hn]
            #  --> [sq, b, np/ng, np, hn], [sq, b, 1, ng, hn], [sq, b, 1, ng, hn]
            query_layer, key_layer, value_layer = _SplitAlongDim.apply(
                mixed_x_layer, split_dim, (num_queries_per_key_value, 1, 1)
            )
            # query: -> [sq, b, np, hn]
            # key, value: -> [sq, b, ng, hn]
            query_layer, key_layer, value_layer = (
                x.reshape(x.size(0), x.size(1), -1, self.hidden_size_per_attention_head)
                for x in (query_layer, key_layer, value_layer)
            )
        elif self.attention_type == "cross":
            # Attention heads [sk, b, h] --> [sk, b, (ng * 2 * hn)]
            mixed_kv_layer = self.key_value(encoder_output, is_first_microbatch=is_first_microbatch, **fp8_kwargs)

            # [sq, b, (ng * 2 * hn)] --> [sq, b, 2 * ng, hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                2 * self.num_gqa_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            # split along second last dimension
            split_dim = -2

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # mixed_kv_layer --> 2 [sk, b, ng, hn]
            key_layer, value_layer = _SplitAlongDim.apply(
                mixed_kv_layer,
                split_dim,
                mixed_kv_layer.shape[split_dim] // 2,
            )
            key_layer, value_layer = (
                x.reshape(
                    x.size(0),
                    x.size(1),
                    -1,
                    self.hidden_size_per_attention_head,
                )
                for x in (key_layer, value_layer)
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            layernorm_query_outputs = self.layernorm_query(
                hidden_states, is_first_microbatch=is_first_microbatch, **fp8_kwargs
            )
            query_layer = layernorm_query_outputs

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ======================================================
        # Apply QK normalization (RMSNorm)
        # ======================================================

        # Must use torch.reshape to flatten the tensor, otherwise an error will be triggered in TE's RMSNorm module.
        query_layer = self.q_norm(query_layer.reshape(-1, self.hidden_size_per_attention_head)).view(query_layer.shape)
        key_layer = self.k_norm(key_layer.reshape(-1, self.hidden_size_per_attention_head)).view(key_layer.shape)

        # ======================================================
        # Apply relative positional encoding (rotary embedding)
        # ======================================================

        if rotary_pos_emb is not None:
            assert not isinstance(query_layer, Float8Tensor) and not isinstance(
                key_layer, Float8Tensor
            ), "RoPE is not supported for Float8Tensors!"
            # duplicate the pos_emb for self attention
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2

            q_pos_emb, k_pos_emb = rotary_pos_emb

            # adjust key and value for inference
            if inference_params is not None:
                if self.qkv_format == "sbhd":
                    sequence_length = key_layer.size(0)
                elif self.qkv_format == "bshd":
                    sequence_length = key_layer.size(1)
                else:
                    raise ValueError(f"QKV format {self.qkv_format} not supported for KV caching.")

                sequence_start = inference_params.sequence_len_offset
                sequence_end = sequence_start + sequence_length

                q_pos_emb = q_pos_emb[sequence_start:sequence_end, ...]
                k_pos_emb = k_pos_emb[sequence_start:sequence_end, ...]

            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb, self.qkv_format, fused=True)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb, self.qkv_format, fused=True)

        # ===========================
        # Core attention computation
        # ===========================
        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_format=self.qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            checkpoint_core_attention=checkpoint_core_attention,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            fast_zero_fill=fast_zero_fill,
            inference_params=inference_params,
        )

        # ===================
        # Output. [sq, b, h]
        # ===================

        projection_output = self.proj(
            context_layer,
            is_first_microbatch=is_first_microbatch,
        )

        if self.return_bias:
            attention_output, attention_bias = projection_output
        else:
            attention_output, attention_bias = projection_output, None

        outputs = (attention_output,)
        if self.return_bias:
            outputs += (attention_bias,)
        if self.input_layernorm and self.return_layernorm_output:
            outputs += (layernorm_output,)
        return outputs if len(outputs) > 1 else outputs[0]

    # Finally, we replace the forward method of given TE's MHA module with our custom forward function.
    self.forward = te_mha_forward_with_qk_norm


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask
