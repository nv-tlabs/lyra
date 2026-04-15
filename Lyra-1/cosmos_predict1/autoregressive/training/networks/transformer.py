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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import transformer_engine as te
from megatron.core import InferenceParams, ModelParallelConfig, parallel_state
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from torch.distributed import ProcessGroup
from torch.distributed import _functional_collectives as funcol
from torch.distributed import broadcast, get_process_group_ranks
from torch.nn.modules.module import _IncompatibleKeys
from transformer_engine.pytorch.module.linear import Linear as LinearTE
from transformer_engine.pytorch.module.rmsnorm import RMSNorm as RMSNormTE

from cosmos_predict1.utils import log

_ACTION_DIM = 8
from cosmos_predict1.autoregressive.modules.embedding import (
    RotaryPositionEmbeddingPytorch,
    RotaryPositionEmbeddingPytorchV2,
    RotaryPositionEmbeddingTE,
    SinCosPosEmbAxisTE,
    get_pos_emb_on_this_cp_rank,
    get_pos_emb_on_this_sptp_rank,
)
from cosmos_predict1.autoregressive.modules.linear import ColumnParallelLinear, TrainingVocabParallelEmbedding
from cosmos_predict1.autoregressive.modules.mlp import TrainingMLP, compute_llama3_ffn_hidden_dim
from cosmos_predict1.autoregressive.modules.normalization import create_norm
from cosmos_predict1.autoregressive.training.modules.attention import (
    GQA,
    create_group_causal_attn_mask,
    enable_different_context_dim_in_te_ca,
    enable_qk_normalization_in_te_mha,
)
from cosmos_predict1.autoregressive.utils.checkpoint import process_state_dict, substrings_to_ignore
from cosmos_predict1.autoregressive.utils.misc import maybe_convert_to_namespace
from cosmos_predict1.autoregressive.utils.parallel import (
    AllReduceBWDRMSNormTE,
    allreduce_layernorm_grads,
    sync_1d_parameters,
)

_MLP_HIDDEN_DIM_DIVISOR = (
    4  # hidden dim of the action embedding layer is action_embedding_dim // _MLP_HIDDEN_DIM_DIVISOR
)

_T5_NUM_TOKENS = 512


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of an attention layer and a feed-forward layer.
    """

    def __init__(self, layer_id: int, model_parallel: Optional[ModelParallelConfig] = None, args=None):
        """
        Initializes the TransformerBlock module.

        Args:
            layer_id: The ID of the transformer block.
            args: The model arguments containing hyperparameters.
        """
        super().__init__()
        args = maybe_convert_to_namespace(args)
        attention_args = {
            "n_heads": args["n_heads"],
            "n_kv_heads": args["n_kv_heads"],
            "dim": args["dim"],
            "context_dim": None,
            "max_batch_size": args["max_batch_size"],
            "max_seq_len": args["max_seq_len"],
            "inference": args["inference"],
            "flash_attn": args["flash_attn"],
            "use_qk_normalization": args["use_qk_normalization"],
            "attention_dropout": getattr(args, "attention_dropout", 0.0),
            "set_parallel_mode": args["set_parallel_mode"],
            "model_parallel": model_parallel,
            "attention_tp": args["attention_tp"],
            "causal_mask": args["causal_mask"],
            "head_dim": args["head_dim"],
            "fuse_qkv": getattr(args, "fuse_qkv", False),
            "precision": getattr(args, "precision", "bfloat16"),
            "attention_type": getattr(args, "attention_type", "self"),
        }
        self.attention = GQA(**attention_args)

        self.has_cross_attention = False
        self.cross_attention, self.cross_attention_norm = None, None

        if args["insert_cross_attn"] and layer_id % args["insert_cross_attn_every_k_layers"] == 0:
            self.has_cross_attention = True
            cross_attention_args = attention_args.copy()
            cross_attention_args.update(
                {"context_dim": args["context_dim"], "fuse_qkv": False, "attention_type": "cross"}
            )
            self.cross_attention = GQA(**cross_attention_args)
            self.cross_attention_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])

        self.feed_forward = TrainingMLP(
            dim=args["dim"],
            hidden_dim=(
                compute_llama3_ffn_hidden_dim(
                    dim=args["dim"], multiple_of=args["multiple_of"], ffn_dim_multiplier=args["ffn_dim_multiplier"]
                )
                if args["ffn_hidden_size"] is None
                else args["ffn_hidden_size"]
            ),
            hidden_dropout=getattr(args, "hidden_dropout", 0.0),
            set_parallel_mode=args["set_parallel_mode"],
            model_parallel=model_parallel,
            inference=args["inference"],
        )
        self.layer_id = layer_id
        self.num_layers = args["n_layers"]
        self.attention_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])
        self.ffn_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])

        # If `True`, then each transformer block init uses its layer ID, and if `False`, each uses the
        # total number of transformer blocks. Default is `True` (following the TorchTitan implementation of Llama3).
        if getattr(args, "depth_init", True):
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbeddingPytorch,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerBlock module.

        Args:
            x: The input tensor.
            input_pos: The position of the current sequence. Used in inference (with KV cache) only.
            freqs_cis: The precomputed frequency values for rotary position embeddings.
            mask: The attention mask tensor.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.

        Returns:
            The output tensor after applying the transformer block.
        """
        # Apply attention and residual connection
        h = x + self.attention(self.attention_norm(x), rope=rope, input_pos=input_pos, mask=mask)

        # If insert cross-attention, apply CA and residual connection
        if self.has_cross_attention:
            h = h + self.cross_attention(
                self.cross_attention_norm(h), rope=rope, input_pos=input_pos, mask=context_mask, context=context
            )

        # Apply feed-forward network and residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        """
        Initializes the weights of the transformer block.
        """
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)

        if self.has_cross_attention:
            self.cross_attention_norm.reset_parameters()
            self.cross_attention.init_weights(self.weight_init_std)
            # zero-init the final output layer of cross-attention
            # nn.init.zeros_(self.cross_attention.wo.weight)


class TransformerBlockTE(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`.

    Args:
        layer_id (int): The ID of the transformer block.
        args: The model arguments containing hyperparameters.
    """

    def __init__(
        self,
        layer_id: int,
        args,
        tp_group: Optional[ProcessGroup] = None,
        set_parallel_mode: bool = False,
        attn_input_format: str = "bshd",
    ):
        attention_args = {
            "hidden_size": args["dim"],
            "ffn_hidden_size": (
                compute_llama3_ffn_hidden_dim(
                    dim=args["dim"], multiple_of=args["multiple_of"], ffn_dim_multiplier=args["ffn_dim_multiplier"]
                )
                if args["ffn_hidden_size"] is None
                else args["ffn_hidden_size"]
            ),
            "num_attention_heads": args["n_heads"],
            "bias": False,
            "layernorm_epsilon": args["norm_eps"],
            "hidden_dropout": getattr(args, "hidden_dropout", 0.0),
            "attention_dropout": getattr(args, "attention_dropout", 0.0),
            "normalization": "RMSNorm",
            "activation": "swiglu",
            "attn_input_format": attn_input_format,
            "num_gqa_groups": args["n_kv_heads"],
            "fuse_wgrad_accumulation": False,
            "fuse_qkv_params": False,
            "tp_group": tp_group,
            "sequence_parallel": args["sequence_parallel"],
            "set_parallel_mode": set_parallel_mode,
            "layer_number": layer_id + 1,
            "self_attn_mask_type": "causal" if args["causal_mask"] else "no_mask",
            "kv_channels": args["head_dim"],  # If None, te.pytorch.TransformerLayer defaults it to dim // n_heads
            "layer_type": "encoder",
        }
        self.has_cross_attention = False
        if args["insert_cross_attn"] and layer_id % args["insert_cross_attn_every_k_layers"] == 0:
            self.has_cross_attention = True
            attention_args["layer_type"] = "decoder"
        super().__init__(**attention_args)
        if args["use_qk_normalization"]:
            # Add QK normalization layers and replace the forward function of original Multi-Head Attention module with
            # our custom one to add QK normalization operations.
            enable_qk_normalization_in_te_mha(self.self_attention, norm_eps=args["norm_eps"], is_self_attn=True)

            if self.has_cross_attention:
                enable_qk_normalization_in_te_mha(self.inter_attention, norm_eps=args["norm_eps"], is_self_attn=False)

        if self.has_cross_attention:
            enable_different_context_dim_in_te_ca(
                self.inter_attention, context_dim=args["context_dim"], args=attention_args
            )

        self.layer_id = layer_id
        self.num_layers = args["n_layers"]
        # If `True`, then each transformer block init uses its layer ID, and if `False`, each uses the
        # total number of transformer blocks. Default is `True` (following the TorchTitan implementation of Llama3).
        if getattr(args, "depth_init", True):
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5
        self.args = args
        self.inference = args["inference"]

    def set_inference_flag(self, flag: bool):
        """
        Set the inference flag for the transformer layers.
        """
        self.inference = flag

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor],
        inference_params: Optional[InferenceParams] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`.

        Args:
            x (torch.Tensor): The input tensor.
            mask (Optional[torch.Tensor]): The attention mask tensor.
            inference_params (Optional[InferenceParams]): Inference parameters used for caching key-value pairs in the TE backend.
                                                          It is not applicable for the PyTorch backend and should be set to None in that case.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.

        Returns:
            torch.Tensor: The output tensor after applying the transformer block
        """

        inference_params = None if not self.inference else inference_params
        output = super().forward(
            x,
            attention_mask=mask,
            rotary_pos_emb=rotary_pos_emb.to(x.device),
            inference_params=inference_params,
            encoder_output=context,
            enc_dec_attn_mask=context_mask,
        )
        return output

    def init_weights(self):
        """
        Initializes the weights of the transformer block.
        """
        # Self Attention
        attn_layer = self.self_attention.layernorm_qkv
        for linear_weight in [attn_layer.query_weight, attn_layer.key_weight, attn_layer.value_weight]:
            nn.init.trunc_normal_(linear_weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.self_attention.proj.weight, mean=0.0, std=self.weight_init_std)

        # Cross Attention
        if self.has_cross_attention:
            nn.init.trunc_normal_(self.inter_attention.layernorm_query.query_weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.inter_attention.key_value.key_weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.inter_attention.key_value.value_weight, mean=0.0, std=0.02)
            # zero-init the final output layer of cross-attention
            if self.args["zero_init_cross_attn_proj"]:
                nn.init.zeros_(self.inter_attention.proj.weight)
            else:
                nn.init.trunc_normal_(self.inter_attention.proj.weight, mean=0.0, std=self.weight_init_std)

        # RMS Normalization
        for norm_weight in (self.layernorm_mlp.layer_norm_weight, self.self_attention.layernorm_qkv.layer_norm_weight):
            torch.nn.init.ones_(norm_weight)

        # In the case of QK Normalization, we also reset the parameters of the QK normalization layers.
        if self.args["use_qk_normalization"]:
            for norm_weight in [self.self_attention.q_norm.weight, self.self_attention.k_norm.weight]:
                torch.nn.init.ones_(norm_weight)

        # MLP
        for linear_weight in (self.layernorm_mlp.fc1_weight, self.layernorm_mlp.fc2_weight):
            nn.init.trunc_normal_(linear_weight, mean=0.0, std=self.weight_init_std)
        # The fc1_weight is a fused weight of w1 and w2 in the MLP of the PyTorch backend, where w1 is initialized with
        # a different std (0.02 by TorchTitan). So we re-initialize the w1 part of the fused weight below.
        split_point = self.layernorm_mlp.fc1_weight.shape[0] // 2
        nn.init.trunc_normal_(self.layernorm_mlp.fc1_weight[:split_point], mean=0.0, std=0.02)


class Transformer(nn.Module):
    """
    The Transformer network consisting of transformer blocks.
    """

    def __init__(self, params, model_parallel=None, tokenizer_config=None, init_weights: bool = True):
        """
        Initializes the Transformer module.

        Args:
            params: The model parameters containing hyperparameters.
            model_parallel: The model parallel configuration.
            tokenizer_config: The model tokenizer configuration.
            init_weights (bool): Whether to initialize the weights of the transformer following
                TorchTitan's Llama3 initialization scheme.
        """
        super().__init__()
        # Check if self.params is an OmegaConf DictConfig instance
        self.params = maybe_convert_to_namespace(params)
        self.vocab_size = params["vocab_size"]
        self.n_layers = params["n_layers"]
        self.precision = getattr(torch, params["precision"])
        self.inference = params["inference"]
        self.backend = params["backend"]
        self.tokenizer_config = tokenizer_config
        self.model_parallel = model_parallel
        self.num_video_frames = params["num_video_frames"]

        self.token_emb_dropout = nn.Dropout(getattr(params, "embedding_dropout", 0.0))

        tp_group = self._get_tp_group()

        # Sequence parallelism requires the first dimension to be the sequence dimension. When sequence parallelism
        # is enabled, we transpose the first two dimensions of the input tensor, and specify the format as "sbhd",
        # (sequence, batch, head, dim). Otherwise, the input format is "bshd" (batch, sequence, head, dim).
        self.attn_input_format = "bshd" if not params["sequence_parallel"] else "sbhd"

        # Token embeddings
        self.tok_embeddings = self._create_token_embeddings(self.model_parallel)
        self.rope_config = self._create_rope_config()

        if self.backend == "pytorch":
            self._initialize_pytorch_backend(model_parallel)
        elif self.backend == "transformer_engine":
            self._initialize_transformer_engine_backend(tp_group)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.output = self._create_output_projection(model_parallel)

        # Action conditioning
        self.use_action_condition = getattr(params, "use_action_condition", False)
        if self.use_action_condition:
            self.action_dim = getattr(
                params, "action_dim", _ACTION_DIM
            )  # e.g., [Δx, Δy, Δz, rx, ry, rz, gripper_open, zero_pad]
            self.action_embedding_dim = self.params["action_embedding_dim"]  # 1024
            self.action_embedding_mode = getattr(params, "action_embedding_mode", "mlp")  # Default to mlp mode
            self.group_causal_mask_mode = getattr(
                params, "group_causal_mask_mode", None
            )  # Default to None, 'causal' or 'group_diagonal'
            self.action_embedding_layers = self._create_action_projection()

        if params["sequence_parallel"]:
            if model_parallel is None:
                setattr(params, "sequence_parallel", False)
                log.critical("model_parallel is None. Disabling sequence parallelism.")
                self.sequence_parallel_enabled = False
            else:
                assert self.backend == "transformer_engine", f"Invalid backend: {self.backend} for sequence parallelism"
                assert (
                    params["tensor_model_parallel_size"] > 1
                ), f"Invalid tensor_model_parallel_size: {params['tensor_model_parallel_size']}"
                self.sequence_parallel_enabled = True
        else:
            self.sequence_parallel_enabled = False

        if init_weights:
            self.init_weights()

        # Set default value for peft_last_n_layers and peft_every_n_layers
        self.peft_last_n_layers = getattr(params, "peft_last_n_layers", 0)
        self.peft_every_n_layers = getattr(params, "peft_every_n_layers", 0)
        if self.peft_last_n_layers > 0 or self.peft_every_n_layers > 0:
            self._setup_peft()

        # Freeze network parameters for finetuning w/ cross-attention
        self.has_cross_attention = getattr(params, "insert_cross_attn", False)
        if self.has_cross_attention:
            self.ca_every_k_layers = getattr(params, "insert_cross_attn_every_k_layers", 1)
            self.finetune_layers_with_cross_attn = getattr(params, "finetune_layers_with_cross_attn", False)
            self.finetune_layers_without_cross_attn = getattr(params, "finetune_layers_without_cross_attn", False)
            self._setup_cross_attn_ft()

        if self.params["apply_abs_pos_emb"]:
            self.pos_emb_config = self._create_abs_pos_emb_config()
            self.pos_emb, self.abs_pos_emb = self._initialize_abs_pos_emb()
            if self.attn_input_format == "sbhd":
                self.abs_pos_emb = self.abs_pos_emb.transpose(0, 1).contiguous()
            self._broadcast_pos_emb(self.abs_pos_emb, tp_group)

    def _initialize_pytorch_backend(self, model_parallel):
        self.layers = nn.ModuleList(
            [
                TransformerBlock(layer_id, model_parallel, self.params).to(self.precision)
                for layer_id in range(self.n_layers)
            ]
        )
        self.norm = create_norm(self.params["norm_type"], dim=self.params["dim"], eps=self.params["norm_eps"]).to(
            self.precision
        )
        pytorch_rope_version = getattr(self.params, "pytorch_rope_version", "v2")
        if pytorch_rope_version == "v1":
            self.rope = RotaryPositionEmbeddingPytorch(**self.rope_config)
        elif pytorch_rope_version == "v2":
            training_type = self.tokenizer_config.training_type if self.tokenizer_config is not None else None
            self.rope = RotaryPositionEmbeddingPytorchV2(
                seq_len=self.params["max_seq_len"], training_type=training_type, **self.rope_config
            )
            self._broadcast_pos_emb(self.rope.cos_cached, tp_group=self._get_tp_group())
            self._broadcast_pos_emb(self.rope.sin_cached, tp_group=self._get_tp_group())
        else:
            raise ValueError(f"Unknown pytorch_rope_version: {pytorch_rope_version}")

        self.causal_mask = torch.tril(
            torch.ones(self.params["max_seq_len"], self.params["max_seq_len"], dtype=torch.bool)
        ).cuda()

    def _initialize_transformer_engine_backend(self, tp_group):
        self.layers = self._create_transformer_layers(tp_group)
        if self.params["sequence_parallel"]:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            self.norm = AllReduceBWDRMSNormTE(
                self.params["dim"],
                process_group=tp_group,
                eps=self.params["norm_eps"],
                sequence_parallel=True,
            ).to(self.precision)
        else:
            self.norm = RMSNormTE(self.params["dim"], eps=self.params["norm_eps"]).to(self.precision)
        self.rope, self.rotary_pos_emb = self._initialize_rope()
        self._broadcast_pos_emb(self.rotary_pos_emb, tp_group)

    def _create_rope_config(self) -> Dict:
        shape_map = {
            "3D": self.params["video_latent_shape"],
            "2D": self.params["image_latent_shape"],
            "1D": None,
        }
        latent_shape = shape_map.get(self.params["rope_dim"], None)
        head_dim = self.params["head_dim"]
        if head_dim is None:
            head_dim = self.params["dim"] // self.params["n_heads"]
        return {
            "dim": head_dim,
            "max_position_embeddings": self.params["max_seq_len"],
            "original_max_position_embeddings": self.params["original_seq_len"],
            "rope_theta": self.params["rope_theta"],
            "apply_yarn": self.params["apply_yarn"],
            "scale": self.params["yarn_scale"],
            "beta_fast": self.params["yarn_beta_fast"],
            "beta_slow": self.params["yarn_beta_slow"],
            "rope_dim": self.params["rope_dim"],
            "latent_shape": latent_shape,
            "original_latent_shape": self.params["original_latent_shape"],
            "pad_to_multiple_of": self.params["pad_to_multiple_of"],
        }

    def _create_abs_pos_emb_config(self):
        shape_map = {
            "3D": self.params["video_latent_shape"],
            "2D": self.params["image_latent_shape"],
            "1D": None,
        }
        latent_shape = shape_map.get(self.params["rope_dim"], None)
        return {
            "dim": self.params["dim"],
            "latent_shape": latent_shape,
            "pad_to_multiple_of": self.params["pad_to_multiple_of"],
        }

    def _create_token_embeddings(self, model_parallel=None, vocab_size: int = None):
        """
        Create token embeddings.

        Args:
            model_parallel: The model parallel configuration.

        Returns:
            nn.Module: Token embeddings module.
        """
        if vocab_size is None:
            vocab_size = self.params["vocab_size"]
        tp_size = self.params["tensor_model_parallel_size"]
        if tp_size > 1:
            # For inference in the PyTorch backend, we use PyTorch's allreduce (tracable) in the forward pass to enable torch.compile.
            use_inference_allreduce = self.inference and self.params["backend"] == "pytorch"
            emb = TrainingVocabParallelEmbedding(
                vocab_size,
                self.params["dim"],
                init_method=lambda x: x,
                config=model_parallel,
                sequence_parallel=self.params["sequence_parallel"],
                batch_first=not self.params["sequence_parallel"],
                use_inference_allreduce=use_inference_allreduce,
            ).to(self.precision)
            return emb
        else:
            return nn.Embedding(vocab_size, self.params["dim"]).to(self.precision)

    def _create_action_projection(self):
        """
        Create the action projection layer.

        Returns:
            nn.Module: Action projection layer.
        """
        assert self.action_embedding_mode == "mlp", f"Invalid action embedding mode: {self.action_embedding_mode}"

        # This method is not working well. (option 1. default) exp102e
        hidden_dim = self.action_embedding_dim // _MLP_HIDDEN_DIM_DIVISOR
        action_embedding_layers = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_embedding_dim),
        )

        return action_embedding_layers

    def _get_tp_group(
        self,
    ):
        """
        Get tensor parallel process group if applicable.

        Returns:
            torch.distributed.ProcessGroup or None: Tensor parallel process group if tensor parallelism is enabled, else None.
        """
        if self.params["tensor_model_parallel_size"] > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            log.info(f"Using tensor model parallel group: {tp_group}")
            return tp_group

        return None

    def _create_transformer_layers(self, tp_group):
        """
        Create the transformer layers.

        Args:
            tp_group (torch.distributed.ProcessGroup or None): Tensor parallel process group.

        Returns:
            nn.ModuleList: List of transformer layers.
        """
        return nn.ModuleList(
            [
                TransformerBlockTE(
                    layer_id,
                    self.params,
                    tp_group,
                    set_parallel_mode=self.params["set_parallel_mode"],
                    attn_input_format=self.attn_input_format,
                ).to(self.precision)
                for layer_id in range(self.params["n_layers"])
            ]
        )

    def _create_output_projection(self, model_parallel=None, vocab_size: int = None):
        """
        Create the output projection layer.

        Args:
            model_parallel: The model parallel configuration.
            vocab_size (int): Vocabulary size (to override the default vocab size).
        Returns:
            LinearTE: Output projection layer.
        """
        if vocab_size is None:
            vocab_size = self.params["vocab_size"]
        if self.params["tensor_model_parallel_size"] > 1:
            if self.params["backend"] == "pytorch" and self.inference:
                tp_size = self.params["tensor_model_parallel_size"]
                layer = nn.Linear(self.params["dim"], vocab_size // tp_size, bias=False).to(self.precision)
                return layer
            else:
                layer = ColumnParallelLinear(
                    self.params["dim"],
                    vocab_size,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x,
                    config=model_parallel,
                ).to(self.precision)
                return layer
        else:
            # No Tensor Parallelism
            if self.params["backend"] == "pytorch":
                return nn.Linear(self.params["dim"], vocab_size, bias=False).to(self.precision)
            elif self.params["backend"] == "transformer_engine":
                return LinearTE(self.params["dim"], vocab_size, bias=False).to(self.precision)
            else:
                raise ValueError("Unknown backend: " + self.params["backend"])

    def _initialize_rope(
        self,
    ):
        """
        Initialize the rotary position embedding.

        Returns:
            tuple: (RotaryPositionEmbeddingTE, torch.Tensor) The RoPE module and the rotary position embeddings.
        """
        rope = RotaryPositionEmbeddingTE(**self.rope_config)
        training_type = self.tokenizer_config.training_type if self.tokenizer_config is not None else None
        rotary_pos_emb = rope.forward(seq_len=self.params["max_seq_len"], training_type=training_type)
        return rope, rotary_pos_emb

    def _initialize_abs_pos_emb(self):
        pos_emb = SinCosPosEmbAxisTE(**self.pos_emb_config)
        training_type = self.tokenizer_config.training_type if self.tokenizer_config is not None else None
        abs_pos_emb = pos_emb.forward(training_type=training_type)
        return pos_emb, abs_pos_emb

    def _broadcast_pos_emb(self, pos_emb, tp_group):
        """
        Broadcast the position embeddings across the tensor parallel group.

        Args:
            pos_emb (torch.Tensor): Position embeddings to broadcast.
            tp_group (torch.distributed.ProcessGroup or None): Tensor parallel process group.
        """
        if self.params["tensor_model_parallel_size"] > 1:
            broadcast(pos_emb, min(get_process_group_ranks(tp_group)), group=tp_group)

    def _setup_peft(self):
        """
        Set up Parameter Efficient Fine-Tuning (PEFT) by selectively freezing and unfreezing layers.

        This method configures the model for fine-tuning by:
        1. Freezing all parameters in the model.
        2. Unfreezing the embedding, normalization and output layers.
        3. Unfreezing the first and last (peft_last_n_layers - 1) transformer layers if peft_last_n_layers is set,
           or unfreezing every n layers (flamingo style) if peft_every_n_layers is set.
        """
        # Ensure only one of peft_last_n_layers and peft_every_n_layers is set
        assert (
            self.peft_last_n_layers == 0 or self.peft_every_n_layers == 0
        ), "Only one of peft_last_n_layers and peft_every_n_layers can be set."

        # First, freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze embedding, normalization and output layers
        for param in self.tok_embeddings.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True
        for param in self.output.parameters():
            param.requires_grad = True

        # PEFT last n layers
        if self.peft_last_n_layers > 0:
            # Ensure peft_last_n_layers is at least 2
            assert self.peft_last_n_layers >= 2, "peft_last_n_layers must be at least 2"

            # Unfreeze specific transformer layers
            total_layers = len(self.layers)
            for i, layer in enumerate(self.layers):
                if i == 0 or i >= total_layers - self.peft_last_n_layers + 1:
                    # Unfreeze the first layer and the last (peft_last_n_layers - 1) layers
                    for param in layer.parameters():
                        param.requires_grad = True

            log.info(
                f"PEFT setup complete. Trainable components: embeddings, un-embedding, normalization layer, "
                f"first transformer layer, last {self.peft_last_n_layers - 1} transformer layers."
            )
        # PEFT every n layers (flamingo style, e.g. every 4 layers = layer 0,1,2,4,5,6,... frozen, layer 3,7,11,... is trainable)
        else:
            trainable_layers = []
            for i, layer in enumerate(self.layers, 1):
                if i % self.peft_every_n_layers == 0:
                    for param in layer.parameters():
                        param.requires_grad = True
                    trainable_layers.append(i - 1)

            log.info(
                f"PEFT setup complete. Trainable components: embeddings, un-embedding, normalization layer, "
                f"every {self.peft_every_n_layers} transformer layers (layer idx {trainable_layers}; total {len(trainable_layers)} layers)."
            )

    def _setup_cross_attn_ft(self):
        """
        Set up Cross Attention Fine-Tuning by selectively freezing and unfreezing layers.

        This method configures the model for fine-tuning by:
        1. Freezing all parameters in the model.
        2. Unfreezing the embedding, normalization and output layers.
        3. Unfreezing all the added cross-attention layers.
        4. If `finetune_layers_with_cross_attn` is True, unfreeze the transformer layers for layers with cross attention.
        5. If `finetune_layers_without_cross_attn` is True, unfreeze the transformer layers for layers without cross attention.
        6. If 'use_action_condition' is True, unfreeze the action embedding layers.
        """
        assert self.has_cross_attention, "Must insert cross-attention layers for finetuning."
        finetune_layer_num = 0

        # First, freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze embedding, normalization and output layers
        for param in self.tok_embeddings.parameters():
            param.requires_grad = True
        for param in self.norm.parameters():
            param.requires_grad = True
        for param in self.output.parameters():
            param.requires_grad = True

        # Unfreeze all the added cross-attention layers
        total_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i % self.ca_every_k_layers == 0:
                if self.params["backend"] == "pytorch":
                    for param in layer.cross_attention.parameters():
                        param.requires_grad = True
                elif self.params["backend"] == "transformer_engine":
                    for param in layer.inter_attention.parameters():
                        param.requires_grad = True
                else:
                    raise ValueError("Unknown backend: " + self.params["backend"])

        # Unfreeze the transformer layers for layers with cross attention
        if self.finetune_layers_with_cross_attn:
            for i, layer in enumerate(self.layers):
                if i % self.ca_every_k_layers == 0:
                    for param in layer.parameters():
                        param.requires_grad = True
                    finetune_layer_num += 1

        # Unfreeze the transformer layers for layers without cross attention
        if self.finetune_layers_without_cross_attn:
            for i, layer in enumerate(self.layers):
                if i % self.ca_every_k_layers != 0:
                    for param in layer.parameters():
                        param.requires_grad = True
                    finetune_layer_num += 1

        # Unfreeze the action embedding layers
        if self.use_action_condition:
            for param in self.action_embedding_layers.parameters():
                param.requires_grad = True

        log.info(
            f"cross attention finetune setup complete. Trainable components: cross-attention layer, "
            f"fully trainable transformer layer number is {finetune_layer_num}."
        )

    def enable_context_parallel(self, cp_group: ProcessGroup):
        """
        Enable context parallelism for the transformer model.

        This method sets up context parallelism by configuring the context parallel group
        and updating each transformer layer to support context parallelism.

        Args:
            cp_group (ProcessGroup): The process group for context parallelism.

        Notes:
            - Updates the model's context parallel group and size.
            - Configures each transformer layer for context parallelism.
            - Enables context parallelism for the rotary position embedding if using the transformer engine backend.
        """
        cp_ranks = get_process_group_ranks(cp_group)
        cp_size = len(cp_ranks)
        # Set these attributes for spliting the data after embedding.
        self.cp_group = cp_group
        # Set these attributes for computing the loss.
        self.cp_size = cp_size

        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, TransformerBlockTE):
                layer.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())
            elif hasattr(layer, "module") and isinstance(layer.module, TransformerBlockTE):
                layer.module.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream())
            else:
                log.warning(f"Layer {layer_idx} does not support context parallelism")

    def set_inference_flag(self, flag: bool):
        """
        Set the inference flag for the transformer layers.
        """
        log.info(f"Setting inference flag to {flag}")
        self.inference = flag
        if self.inference:
            self.eval()
        if self.params["backend"] == "pytorch":
            for layer in self.layers:
                layer.attention.set_inference_flag(flag)
        elif self.params["backend"] == "transformer_engine":
            for layer in self.layers:
                layer.set_inference_flag(flag)

        self._maybe_change_sequence_parallel_status(enable=False)

    def _maybe_change_sequence_parallel_status(self, enable: bool):
        """
        Change the sequence parallel status of the transformer layers.
        """
        if enable and not self.sequence_parallel_enabled:
            for name, module in self.named_modules():
                if hasattr(module, "sequence_parallel"):
                    assert isinstance(
                        module.sequence_parallel, bool
                    ), f"Invalid type of {name}: {type(module.sequence_parallel)}"
                    setattr(module, "sequence_parallel", True)
            self.sequence_parallel_enabled = True
        elif not enable and self.sequence_parallel_enabled:
            for name, module in self.named_modules():
                if hasattr(module, "sequence_parallel"):
                    assert isinstance(
                        module.sequence_parallel, bool
                    ), f"Invalid type of {name}: {type(module.sequence_parallel)}"
                    setattr(module, "sequence_parallel", False)
            self.sequence_parallel_enabled = False

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        total_seq_len: Optional[int] = None,
        return_hidden_states: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer module.

        Args:
            tokens (torch.Tensor, optional): The input tensor of token IDs.
            input_pos (Optional[torch.Tensor]): The position of the current sequence. Used in inference with KV cache. PyTorch backend only.
            inference_params (InferenceParams, optional): Parameters for inference.
            token_embeddings (torch.Tensor, optional): Precomputed token embeddings. If provided, tokens should be None.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.
            action (Optional[torch.Tensor]): The robot action tensor for conditioning.
            total_seq_len (Optional[int]): The total sequence length (before applying context parallelism).
            return_hidden_states (bool): Whether to return hidden states.
        Returns:
            The output tensor after applying the transformer layers.
        """

        # Turn on/off sequence parallelism based on the training status
        self._maybe_change_sequence_parallel_status(enable=self.training and self.params["sequence_parallel"])

        # Token embeddings
        assert (
            tokens is None or token_embeddings is None
        ), "Either tokens or token_embeddings should be provided, not both."

        if token_embeddings is None:
            seq_len = tokens.shape[1]
            h = self.token_emb_dropout(self.tok_embeddings(tokens))
        else:
            seq_len = token_embeddings.shape[1]
            h = self.token_emb_dropout(token_embeddings)

        if mask is None:
            # Create attention mask
            mask = self._create_attention_mask(input_pos=input_pos)

        # Action embedding
        if self.use_action_condition and action is not None:
            assert self.action_embedding_mode == "mlp", f"Invalid action embedding mode: {self.action_embedding_mode}"
            # change action type to bfloat16, of shape [batch_size, action_dim]
            action = action.to(torch.bfloat16)
            # action_emb shape: [batch_size, action_dim, action_embedding_dim]
            action_emb = self.action_embedding_layers(action).unsqueeze(1).repeat(1, self.action_dim, 1)

            # Use action_emb as context
            if self.params["concat_action_to_context"]:
                context = torch.zeros(
                    (action_emb.shape[0], _T5_NUM_TOKENS, self.action_embedding_dim), device=h.device, dtype=h.dtype
                )
                # context[:, -1, :] = action_emb[:, 0, :] # overwrite the last token with action_emb
                context = torch.cat([context, action_emb[:, 0:1, :]], dim=1)
            else:
                context = action_emb  # [batch_size, action_dim, action_embedding_dim]

            # Create context mask
            if self.group_causal_mask_mode is not None:
                num_temporal_groups = self.num_video_frames - 1  # number of latent frames
                num_query_per_group = seq_len // num_temporal_groups  # number of latent tokens per frame
                num_key_per_group = self.action_dim // num_temporal_groups
                context_mask = create_group_causal_attn_mask(
                    num_temporal_groups=num_temporal_groups,
                    num_query_per_group=num_query_per_group,
                    num_key_per_group=num_key_per_group,
                    mode=self.group_causal_mask_mode,
                )  # [L (query), S (key)]
                context_mask = context_mask.unsqueeze(0)  # [1, L (query), S (key)]
                context_mask = context_mask.repeat(context.shape[0], 1, 1)  # [batch_size, L (query), S (key)]
                context_mask = context_mask.to(context.device)
            else:
                context_mask = torch.ones(
                    (context.shape[0], context.shape[1]), device=context.device, dtype=torch.bool
                )  # [batch_size, action_dim]

        # Prepare layer arguments
        layer_kwargs = self._prepare_layer_kwargs(
            total_seq_len=total_seq_len,
            input_pos=input_pos,
            mask=mask,
            inference_params=inference_params,
            context=context,
            context_mask=context_mask,
        )

        # Apply transformer layers
        for layer in self.layers:
            if self.params["apply_abs_pos_emb"]:
                h = self.apply_abs_pos_emb(h, input_pos=input_pos, total_seq_len=total_seq_len)
            h = layer(h, **layer_kwargs)

        # Apply final layer normalization
        h = self.norm(h)
        if return_hidden_states:
            return h

        # Output linear projection
        output = self.output(h)
        output = self.process_output(output)
        return output

    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Adjusts the shape and layout of tensor based on tensor parallelism and attention input format.

        The function performs two operations:
        1. If the tensor model parallelism is enabled (`tensor_model_parallel_size > 1`), it gathers the tensor from
        the tensor-parallel regions and reshapes it accordingly.
        2. If the attention input format is `"sbhd"` (Sequence, Batch, Hidden Dimension), it transposes the tensor
        to the format `(Batch, Sequence, Hidden Dimension)` for further processing.

        Args:
            output [torch.Tensor]: The tensor before modification.

        Returns:
            output [torch.Tensor]: The tensor after modification.

        """
        if self.params["tensor_model_parallel_size"] > 1:
            if self.params["backend"] == "pytorch" and self.inference:
                # Use PyTorch all gather
                output = funcol.all_gather_tensor(
                    output, gather_dim=-1, group=parallel_state.get_tensor_model_parallel_group()
                )
            else:
                # [*, *, hidden_dim // tp_size] --> [*, *, hidden_dim]
                output = gather_from_tensor_model_parallel_region(output)
        if self.attn_input_format == "sbhd":
            # [seq_len, batch_size, hidden_dim] --> [batch_size, seq_len, hidden_dim]
            output = output.transpose(0, 1).contiguous()
        return output

    def _create_attention_mask(self, input_pos: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Creates an attention mask for the transformer layers.

        Args:
            input_pos[torch.Tensor]: The position of input sequence (used for inference only).

        Returns:
            Optional[torch.Tensor]: The attention mask, or None for causal mask.
        """

        if self.backend == "pytorch" and self.inference:
            assert input_pos is not None, "input_pos must be provided for inference"
            mask = self.causal_mask[input_pos]
            return mask
        else:
            return None  # None means causal mask

    def _prepare_layer_kwargs(
        self,
        total_seq_len: Optional[int],
        input_pos: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inference_params: Optional[InferenceParams],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Prepares the keyword arguments for transformer layers.

        Args:
            total_seq_len (Optional[int]): The total sequence length (before applying context parallelism).
            seq_len (Optional[int]): The length of the input sequence.
            input_pos (Optional[torch.Tensor]): The position of the current sequence.
            mask (Optional[torch.Tensor]): The attention mask.
            inference_params (Optional[InferenceParams]): Parameters for inference.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for the transformer layers.
        """
        if context is not None:
            context = context.to(self.precision)

            if self.attn_input_format == "sbhd":
                context = context.transpose(0, 1).contiguous()
        if self.backend == "pytorch":
            if isinstance(mask, torch.Tensor) and mask.ndim == 2:
                mask = mask[None, None, :, :]
            if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 2:
                context_mask = context_mask[None, None, :, :]

        layer_kwargs = {
            "mask": mask,
            "context": context,
            "context_mask": context_mask,
        }

        if self.backend == "pytorch":
            layer_kwargs["input_pos"] = input_pos
            layer_kwargs["rope"] = self.rope
        elif self.backend == "transformer_engine":
            rotary_pos_emb = self.rotary_pos_emb
            try:
                cp_size = parallel_state.get_context_parallel_world_size()
            except (AssertionError, RuntimeError):
                # Fallback if context parallel group isn't initialized
                cp_size = 1
                log.warning("Context parallel group not initialized, falling back to size 1")
            else:
                cp_size = 1
            if cp_size > 1:
                assert input_pos is None, "input_pos must be None for context parallelism"
                rotary_pos_emb = rotary_pos_emb[:total_seq_len]
                rotary_pos_emb = get_pos_emb_on_this_cp_rank(rotary_pos_emb, 0)

            layer_kwargs["rotary_pos_emb"] = rotary_pos_emb
            layer_kwargs["inference_params"] = inference_params

        return layer_kwargs

    def apply_abs_pos_emb(
        self, x: torch.Tensor, input_pos: int = None, total_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Applies the absolute position embeddings to the input tensor.
        """
        abs_pos_emb = self.abs_pos_emb
        if total_seq_len is not None:
            # Truncate the absolute position embeddings to the total sequence length
            abs_pos_emb = (
                abs_pos_emb[:total_seq_len, :, :]
                if self.attn_input_format == "sbhd"
                else abs_pos_emb[:, :total_seq_len, :]
            )
        cp_size = parallel_state.get_context_parallel_world_size() if self.training else 1
        if cp_size > 1:
            assert input_pos is None
            seq_dim = 0 if self.attn_input_format == "sbhd" else 1
            abs_pos_emb = get_pos_emb_on_this_cp_rank(abs_pos_emb, seq_dim=seq_dim)
        if self.attn_input_format == "sbhd":
            if self.sequence_parallel_enabled:
                # Training
                assert input_pos is None, "input_pos must be None when training with sequence parallelism"
                abs_pos_emb = get_pos_emb_on_this_sptp_rank(abs_pos_emb, seq_dim=0)
            else:
                # Inference or Evaluation
                abs_pos_emb = abs_pos_emb[input_pos, :, :] if input_pos is not None else abs_pos_emb
        else:
            abs_pos_emb = abs_pos_emb[:, input_pos, :] if input_pos is not None else abs_pos_emb
        return x + abs_pos_emb

    @torch.no_grad()
    def expand_vocab(
        self, new_vocab_size: int, init_method: str = "gaussian", multiple_of=64, expand_output_layer=True
    ):
        """
        Expands the vocabulary of the model to the new size.

        Args:
            new_vocab_size (int): The new vocabulary size.
            init_method (str): The initialization method for new embeddings.
                               Can be "zero" or "gaussian". Default is "gaussian".
            multiple_of (int): The new vocabulary size must be a multiple of this value. Defaults to 64 to fully
                leverage the power of NVIDIA TensorCore (source 1: https://x.com/karpathy/status/1621578354024677377,
                source 2: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)
            expand_output_layer (bool): Whether to also expand the output layer. Defaults to True.

        Returns:
            None
        """

        tp_size = self.params["tensor_model_parallel_size"]
        if new_vocab_size <= self.vocab_size:
            raise ValueError(
                f"New vocabulary size ({new_vocab_size}) must be " f"larger than current size ({self.vocab_size})"
            )
        if new_vocab_size % multiple_of != 0:
            log.critical(f"New vocabulary size must be a multiple of {multiple_of}. Obtained {new_vocab_size}.")
            new_vocab_size = (new_vocab_size // multiple_of + 1) * multiple_of
            log.critical(f"Rounded vocabulary size to {new_vocab_size}.")
        # Resize token embeddings
        old_embeddings = self.tok_embeddings
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        tensor_kwargs = {"device": old_embeddings.weight.device, "dtype": old_embeddings.weight.dtype}
        self.tok_embeddings = self._create_token_embeddings(
            model_parallel=self.model_parallel, vocab_size=new_vocab_size
        ).to(**tensor_kwargs)
        # Initialize new embeddings
        if init_method not in ["zero", "gaussian"]:
            raise ValueError(f"Unknown initialization method: {init_method}")
        # The default initialization of nn.Embedding is Gaussian, so we don't need to do anything
        # if init_method == "gaussian". Only if init_method == "zero", we need to zero out the new embeddings.
        if init_method == "zero":
            self.tok_embeddings.weight.data[self.vocab_size // tp_size :].zero_()

        # Copy old embeddings
        log.info(
            f"old_embeddings: {old_embeddings.weight.data.shape}, new_embeddings: {self.tok_embeddings.weight.data.shape}, vocab_size: {self.vocab_size}"
        )
        self.tok_embeddings.weight.data[: self.vocab_size // tp_size] = old_embeddings.weight.data
        self.tok_embeddings.weight.requires_grad = old_embeddings_requires_grad
        # Resize output layer
        old_output = self.output
        old_output_requires_grad = old_output.weight.requires_grad
        self.output = self._create_output_projection(
            self.model_parallel, vocab_size=new_vocab_size if expand_output_layer else None
        )

        # Initialize new output weights
        if init_method == "zero":
            self.output.weight.data[self.vocab_size // tp_size :].zero_()
        elif init_method == "gaussian":
            # Follows the parameter initialization in TorchTitan:
            # https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py
            final_out_std = self.params["dim"] ** -0.5
            cutoff_factor = 3
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        # Copy old output weights
        self.output.weight.data[: self.vocab_size // tp_size] = old_output.weight.data
        self.output.weight.requires_grad = old_output_requires_grad

        # Update vocab size
        self.vocab_size = new_vocab_size
        log.critical(f"Expanded vocabulary size to {new_vocab_size}")

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters`` (copied from github.com/pytorch/torchtitan)]
        Modules may define ``reset_parameters`` to initialize parameter values. ``reset_parameters`` is meant to only
        initialize directly owned parameters/buffers, not those of their child modules, and it can be used to give the
        initial values for these tensors. Separately, users may want custom initialization for their modules, different
        from that in ``reset_parameters``. For this, we define ``init_weights``. We only call it in the constructor of
        this ``Transformer`` root module to avoid reinitializing tensors.
        """

        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers:
            layer.init_weights()
        if self.backend == "pytorch":
            self.norm.reset_parameters()
        elif self.backend == "transformer_engine":
            nn.init.ones_(self.norm.weight)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        final_out_std = self.params["dim"] ** -0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

        if self.use_action_condition:
            for layer in self.action_embedding_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def state_dict(self, *args, **kwargs):
        """
        Process the state dict (e.g., remove "_extra_state" keys imposed by TransformerEngine for FP8).
        """
        state_dict = super().state_dict(*args, **kwargs)
        return process_state_dict(state_dict)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """
        Ignore the missing keys with substrings matching `substring_to_ignore` (e.g., "_extra_state" keys imposed by
        TransformerEngine for FP8).
        """
        state_dict = process_state_dict(state_dict)
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            actual_missing_keys = []
            for key in missing_keys:
                if not any(substring in key for substring in substrings_to_ignore):
                    actual_missing_keys.append(key)
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}")
            missing_keys = actual_missing_keys
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def on_after_backward(self, *args, **kwargs):
        """
        All-reduce layernorm grads for tensor/sequence parallelism.
        Reference implementation: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/finalize_model_grads.py
        """
        allreduce_layernorm_grads(
            [self],
            tensor_model_parallel_size=self.params["tensor_model_parallel_size"],
            sequence_parallel=self.params["sequence_parallel"],
        )

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """Hook before zero_grad() is called.

        Args:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            iteration (int): Current iteration number.
        """
        if self.params["sync_1d_parameters"]:
            if self.params["tensor_model_parallel_size"] > 1:
                sync_1d_parameters(self, process_group=parallel_state.get_tensor_model_parallel_group())
            if self.params["context_parallel_size"] > 1:
                sync_1d_parameters(self, process_group=parallel_state.get_context_parallel_group())
