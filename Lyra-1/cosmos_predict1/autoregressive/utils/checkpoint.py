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

from cosmos_predict1.utils import log

# Substrings to ignore when processing state dicts
substrings_to_ignore = [
    "_extra_state",  # Extra states (BytesIO type) added by TransformerEngine for FP8 handling
]


def identify_checkpoint_backend(state_dict: dict[str, torch.Tensor]) -> str:
    """
    Identify the backend of the checkpoint (PyTorch or TransformerEngine)

    Args:
        state_dict (dict[str, torch.Tensor]): The state dict to check

    Returns:
        str: The backend of the checkpoint
    """
    for key in state_dict.keys():
        if "self_attention.layernorm_qkv.query_weight" in key:
            return "transformer_engine"
        elif "attention.wq.weight" in key:
            return "pytorch"
    raise ValueError("Could not identify the backend of the checkpoint")


def get_partial_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """
    Get a partial state dict with keys starting with the given prefix
    """
    return {k: v for k, v in state_dict.items() if k.startswith(prefix)}


def process_state_dict(
    state_dict: dict[str, torch.Tensor],
    device: str = None,
    dtype: torch.dtype = None,
    prefix_to_remove: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """
    - Remove items with substring "_extra_state" in keys (TransformerEngine adds these for FP8)
    - Move tensors to specified device and dtype if provided

    Args:
        state_dict (dict[str, torch.Tensor]): The state dict to process
        device (str, optional): The device to move tensors to. Defaults to None.
        dtype (torch.dtype, optional): The dtype to move tensors to. Defaults to None.
        prefix_to_remove (str, optional): The prefix to remove from the keys of the state dict. Defaults to None.

    Returns:
        dict[str, torch.Tensor]: The processed state dict
    """
    new_state_dict = {}
    tensor_kwargs = {}
    if device is not None:
        tensor_kwargs["device"] = device
    if dtype is not None:
        tensor_kwargs["dtype"] = dtype

    for key, value in state_dict.items():
        # Check if any of the substrings to ignore are in the key
        skip = False
        for substr in substrings_to_ignore:
            if substr in key:
                skip = True
                break
        if skip:
            continue
        if len(tensor_kwargs) > 0:
            value = value.to(**tensor_kwargs)
        if prefix_to_remove is not None and key.startswith(prefix_to_remove):
            key = key[len(prefix_to_remove) :]
        new_state_dict[key] = value
    return new_state_dict


def obtain_tensor_parallel_state_dict(
    whole_model_state_dict: dict[str, torch.Tensor],
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    model_config,
    target_backend: str = None,
) -> dict[str, torch.Tensor]:
    """
    Obtain the tensor parallel state dict shard for the current rank.

    Args:
        whole_model_state_dict (dict[str, torch.Tensor]): The complete model state dict.
        tensor_parallel_size (int): The number of tensor parallel devices.
        tensor_parallel_rank (int): The rank of the current tensor parallel device.
        model_config: The model configuration.
        target_backend (str, optional): The target backend format ('pytorch', 'transformer_engine', or 'huggingface'). If not specified, the source backend will be used.

    Returns:
        dict[str, torch.Tensor]: The updated state dict shard for the current tensor parallel rank.
    """
    new_state_dict_shard = {}
    whole_model_state_dict = process_state_dict(whole_model_state_dict)
    source_backend = identify_checkpoint_backend(whole_model_state_dict)
    if source_backend != "pytorch":
        # Convert the checkpoint to PyTorch backend for checkpoint sharding
        whole_model_state_dict = maybe_convert_checkpoint_to_backend(
            whole_model_state_dict, target_backend="pytorch", model_config=model_config, source_backend=source_backend
        )

    n_heads = model_config["n_heads"]
    n_kv_heads = model_config["n_kv_heads"]
    dim = model_config["dim"]
    context_dim = model_config["context_dim"]
    for key, value in whole_model_state_dict.items():
        prefix = "model." if key.startswith("model.") else ""  # LLM's model prefix
        prefix = "transformer." if key.startswith("transformer.") else prefix  # VIT's model prefix
        key = key.replace(prefix, "")
        if key.startswith("layers."):
            layer_index = int(key.split("layers.")[1].split(".")[0])
            if layer_index >= model_config["n_layers"]:
                log.warning(
                    f"Layer index {layer_index} is greater than the number of layers {model_config['n_layers']}. Skipping this layer."
                )
                continue
            if ".attention.wq.weight" in key or "cross_attention.wq.weight" in key:
                value = torch.chunk(value.view(n_heads, -1, dim), tensor_parallel_size, dim=0)[tensor_parallel_rank]
                value = value.reshape(-1, dim)
            elif ".attention.wk.weight" in key or ".attention.wv.weight" in key:
                value = torch.chunk(value.view(n_kv_heads, -1, dim), tensor_parallel_size, dim=0)[tensor_parallel_rank]
                value = value.reshape(-1, dim)
            elif "cross_attention.wk.weight" in key or "cross_attention.wv.weight" in key:
                assert context_dim is not None
                value = torch.chunk(value.view(n_kv_heads, -1, context_dim), tensor_parallel_size, dim=0)[
                    tensor_parallel_rank
                ]
                value = value.reshape(-1, context_dim)
            elif "feed_forward.w1.weight" in key or "feed_forward.w3.weight" in key or "medusa_head" in key:
                value = torch.chunk(value, tensor_parallel_size, dim=0)[tensor_parallel_rank]
            elif "feed_forward.w2.weight" in key or ".attention.wo.weight" in key or "cross_attention.wo.weight" in key:
                value = torch.chunk(value, tensor_parallel_size, dim=1)[tensor_parallel_rank]
        else:
            # Handle non-layer weights
            if key == "tok_embeddings.weight" or key == "output.weight" or "medusa_head" in key:
                value = torch.chunk(value, tensor_parallel_size, dim=0)[tensor_parallel_rank]
        new_state_dict_shard[prefix + key] = value

    if target_backend is None:
        target_backend = source_backend

    new_state_dict_shard = maybe_convert_checkpoint_to_backend(
        new_state_dict_shard,
        target_backend=target_backend,
        model_config=model_config,
        is_tensor_parallel_shard=True,
        tensor_parallel_size=tensor_parallel_size,
    )

    return new_state_dict_shard


def merge_tensor_parallel_state_dicts(
    state_dict_shards: list[dict[str, torch.Tensor]],
    model_config,
    target_backend: str = None,
) -> dict[str, torch.Tensor]:
    """
    Merge tensor parallel state dict shards into a whole model state dict.

    Args:
        state_dict_shards (List[Dict[str, torch.Tensor]]): The list of state dict shards to merge.
        model_config: The model configuration.
        target_backend (str, optional): The target backend format ('pytorch', 'transformer_engine', or 'huggingface'). If not specified, the source backend will be used.

    Returns:
        Dict[str, torch.Tensor]: The merged state dict.
    """
    state_dict_shards = [process_state_dict(shard, device="cpu") for shard in state_dict_shards]
    tensor_parallel_size = len(state_dict_shards)
    source_backend = identify_checkpoint_backend(state_dict_shards[0])
    if source_backend != "pytorch":
        log.critical(f"Converting from {source_backend} to PyTorch backend for tensor parallel checkpoint merging.")
        state_dict_shards = [
            maybe_convert_checkpoint_to_backend(
                shard,
                target_backend="pytorch",
                model_config=model_config,
                source_backend=source_backend,
                is_tensor_parallel_shard=True,
                tensor_parallel_size=tensor_parallel_size,
            )
            for shard in state_dict_shards
        ]

    n_heads = model_config["n_heads"]
    n_kv_heads = model_config["n_kv_heads"]
    n_local_heads = n_heads // tensor_parallel_size
    n_local_kv_heads = n_kv_heads // tensor_parallel_size
    dim = model_config["dim"]
    context_dim = model_config["context_dim"]
    head_dim = model_config["head_dim"]
    if head_dim is None:
        head_dim = model_config["dim"] // model_config["n_heads"]
    query_dim = head_dim * n_heads
    key_value_dim = head_dim * n_kv_heads
    merged_state_dict = {}

    for key in state_dict_shards[0].keys():
        prefix = "model." if key.startswith("model.") else ""
        key_without_prefix = key[len(prefix) :]
        if key_without_prefix.startswith("layers."):
            layer_index = int(key_without_prefix.split("layers.")[1].split(".")[0])
            if layer_index >= model_config["n_layers"]:
                log.warning(
                    f"Layer index {layer_index} is greater than the number of layers {model_config['n_layers']}. Skipping this layer."
                )
                continue
        if key_without_prefix == "tok_embeddings.weight" or key_without_prefix == "output.weight":
            merged_state_dict[key] = torch.cat([shard[key] for shard in state_dict_shards], dim=0)
        elif ".attention.wq.weight" in key or "cross_attention.wq.weight" in key:
            chunks = [shard[key].view(n_local_heads, head_dim, dim) for shard in state_dict_shards]
            merged_state_dict[key] = torch.cat(chunks, dim=0).reshape(query_dim, dim)
        elif ".attention.wk.weight" in key or ".attention.wv.weight" in key:
            chunks = [shard[key].view(n_local_kv_heads, head_dim, dim) for shard in state_dict_shards]
            merged_state_dict[key] = torch.cat(chunks, dim=0).reshape(key_value_dim, dim)
        elif "cross_attention.wk.weight" in key or "cross_attention.wv.weight" in key:
            chunks = [shard[key].view(n_local_kv_heads, head_dim, context_dim) for shard in state_dict_shards]
            merged_state_dict[key] = torch.cat(chunks, dim=0).reshape(key_value_dim, context_dim)
        elif "feed_forward.w1.weight" in key or "feed_forward.w3.weight" in key or "medusa_head" in key:
            merged_state_dict[key] = torch.cat([shard[key] for shard in state_dict_shards], dim=0)
        elif "feed_forward.w2.weight" in key or ".attention.wo.weight" in key or "cross_attention.wo.weight" in key:
            merged_state_dict[key] = torch.cat([shard[key] for shard in state_dict_shards], dim=1)
        else:
            avg_tensor = torch.stack([shard[key] for shard in state_dict_shards]).mean(dim=0)
            # make sure shard-0 is close to the average tensor
            assert torch.allclose(state_dict_shards[0][key], avg_tensor, atol=5e-2, rtol=0.1), (
                f"Shard-0 tensor {key} is not close to the average tensor. "
                f"Max diff: {torch.max(torch.abs(state_dict_shards[0][key] - avg_tensor))}, "
            )
            merged_state_dict[key] = avg_tensor
            assert "norm" in key, f"Assumed the key {key} is a norm layer, which should be the same across shards."

    if target_backend is None:
        target_backend = source_backend
    return maybe_convert_checkpoint_to_backend(
        merged_state_dict, target_backend=target_backend, model_config=model_config
    )


def te_to_pytorch_state_dict(
    te_state_dict: Dict[str, torch.Tensor], model_config, tensor_parallel_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Convert a TransformerEngine state dict to PyTorch state dict

    Args:
        te_state_dict (Mapping[str, torch.Tensor]): The TransformerEngine state dict
        model_config: The model configuration
        tensor_parallel_size (int): The tensor parallel size. Defaults to 1 (i.e., not a tensor parallel shard).

    Returns:
        Mapping[str, torch.Tensor]: The PyTorch state dict
    """

    if hasattr(model_config, "asdict"):
        model_config = model_config.asdict()

    pytorch_state_dict = {}
    replacement_rules = [
        # Self-attention modules
        (".self_attention.layernorm_qkv.layer_norm_weight", ".attention_norm.weight"),
        (".self_attention.layernorm_qkv.query_weight", ".attention.wq.weight"),
        (".self_attention.layernorm_qkv.key_weight", ".attention.wk.weight"),
        (".self_attention.layernorm_qkv.value_weight", ".attention.wv.weight"),
        (".self_attention.proj.weight", ".attention.wo.weight"),
        (".self_attention.", ".attention."),  # Handle the rest modules such as q_norm and k_norm
        # MLP modules
        (".layernorm_mlp.layer_norm_weight", ".ffn_norm.weight"),
        (".layernorm_mlp.fc2_weight", ".feed_forward.w2.weight"),
        # Cross-attention modules
        (".inter_attention.layernorm_query.query_weight", ".cross_attention.wq.weight"),
        (".inter_attention.key_value.key_weight", ".cross_attention.wk.weight"),
        (".inter_attention.key_value.value_weight", ".cross_attention.wv.weight"),
        (".inter_attention.proj.weight", ".cross_attention.wo.weight"),
        (".inter_attention.layernorm_query.layer_norm_weight", ".cross_attention_norm.weight"),
        (".inter_attention.", ".cross_attention."),  # Handle the rest modules such as q_norm and k_norm
    ]
    head_dim = model_config["head_dim"]
    if head_dim is None:
        head_dim = model_config["dim"] // model_config["n_heads"]
    for old_key, value in te_state_dict.items():
        new_key = old_key
        for old_substr, new_substr in replacement_rules:
            if old_substr in new_key:
                new_key = new_key.replace(old_substr, new_substr)
                break

        # Handle the fused w1 and w3 case
        if "layernorm_mlp.fc1_weight" in old_key:
            fused_weight = value
            split_point = fused_weight.shape[0] // 2
            w1_weight = fused_weight[:split_point]
            w3_weight = fused_weight[split_point:]

            w1_key = new_key.replace("layernorm_mlp.fc1_weight", "feed_forward.w1.weight")
            w3_key = new_key.replace("layernorm_mlp.fc1_weight", "feed_forward.w3.weight")

            pytorch_state_dict[w1_key] = w1_weight
            pytorch_state_dict[w3_key] = w3_weight
        else:
            if model_config["pytorch_rope_version"] == "v1":
                # If the model use qk normalization, we will use the same PyTorch RoPE operations as the TE version.
                # Thus, we do not need to permute the weights.
                if "query_weight" in old_key:
                    value = inverse_permute_weight(
                        value,
                        n_heads=model_config["n_heads"] // tensor_parallel_size,
                        dim1=head_dim * model_config["n_heads"] // tensor_parallel_size,
                        dim2=model_config["dim"],
                    )
                elif "key_weight" in old_key:
                    value = inverse_permute_weight(
                        value,
                        n_heads=model_config["n_kv_heads"] // tensor_parallel_size,
                        dim1=head_dim * model_config["n_kv_heads"] // tensor_parallel_size,
                        dim2=model_config["context_dim"] if "inter_attention" in old_key else model_config["dim"],
                    )
            pytorch_state_dict[new_key] = value

    return pytorch_state_dict


def pytorch_to_te_state_dict(
    pytorch_state_dict: Dict[str, torch.Tensor], model_config, tensor_parallel_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Convert a PyTorch state dict to TransformerEngine state dict

    Args:
        pytorch_state_dict (Mapping[str, torch.Tensor]): The PyTorch state dict
        model_config: The model configuration
        tensor_parallel_size (int): The tensor parallel size. Defaults to 1 (i.e., not a tensor parallel shard).

    Returns:
        Mapping[str, torch.Tensor]: The TransformerEngine
    """

    if hasattr(model_config, "asdict"):
        model_config = model_config.asdict()

    te_state_dict = {}

    replacement_rules = [
        # Self-attention modules
        (".attention_norm.weight", ".self_attention.layernorm_qkv.layer_norm_weight"),
        (".attention.wq.weight", ".self_attention.layernorm_qkv.query_weight"),
        (".attention.wk.weight", ".self_attention.layernorm_qkv.key_weight"),
        (".attention.wv.weight", ".self_attention.layernorm_qkv.value_weight"),
        (".attention.wo.weight", ".self_attention.proj.weight"),
        (".attention.", ".self_attention."),
        # MLP modules
        (".ffn_norm.weight", ".layernorm_mlp.layer_norm_weight"),
        (".feed_forward.w2.weight", ".layernorm_mlp.fc2_weight"),
        # Cross-attention modules
        (".cross_attention_norm.weight", ".inter_attention.layernorm_query.layer_norm_weight"),
        (".cross_attention.wq.weight", ".inter_attention.layernorm_query.query_weight"),
        (".cross_attention.wk.weight", ".inter_attention.key_value.key_weight"),
        (".cross_attention.wv.weight", ".inter_attention.key_value.value_weight"),
        (".cross_attention.wo.weight", ".inter_attention.proj.weight"),
        (".cross_attention.", ".inter_attention."),
    ]
    head_dim = model_config["head_dim"]
    if head_dim is None:
        head_dim = model_config["dim"] // model_config["n_heads"]
    for old_key, value in pytorch_state_dict.items():
        new_key = old_key
        for new_substr, old_substr in replacement_rules:
            if new_substr in new_key:
                new_key = new_key.replace(new_substr, old_substr)
                break

        # Handle the split w1 and w3 case
        if "feed_forward.w1.weight" in old_key:
            w1_weight = value
            w3_key = old_key.replace("feed_forward.w1.weight", "feed_forward.w3.weight")
            if w3_key in pytorch_state_dict:
                w3_weight = pytorch_state_dict[w3_key]
                fused_weight = torch.cat([w1_weight, w3_weight], dim=0)
                new_key = new_key.replace("feed_forward.w1.weight", "layernorm_mlp.fc1_weight")
                te_state_dict[new_key] = fused_weight
            else:
                te_state_dict[new_key] = value
        elif "feed_forward.w3.weight" in old_key:
            # Skip w3 weights as they're handled with w1
            continue
        else:
            if model_config["pytorch_rope_version"] == "v1":
                # If the model use qk normalization, we will use the same PyTorch RoPE operations as the TE version.
                # Thus, we do not need to permute the weights.
                if "attention.wq" in old_key:
                    value = permute_weight(
                        value,
                        n_heads=model_config["n_heads"] // tensor_parallel_size,
                        dim1=head_dim * model_config["n_heads"] // tensor_parallel_size,
                        dim2=model_config["dim"],
                    )
                elif "attention.wk" in old_key:
                    value = permute_weight(
                        value,
                        n_heads=model_config["n_kv_heads"] // tensor_parallel_size,
                        dim1=head_dim * model_config["n_kv_heads"] // tensor_parallel_size,
                        dim2=model_config["context_dim"] if "cross_attention" in old_key else model_config["dim"],
                    )
            te_state_dict[new_key] = value

    return te_state_dict


def permute_weight(w: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """
    Helper function for converting checkpoints from PyTorch to TransformerEngine
    Permute the query weight or key weight of each attention layer
    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

    Args:
        w (torch.Tensor): The weight tensor to permute
        n_heads (int): The number of attention heads
        dim1 (int): The first dimension of the weight tensor
        dim2 (int): The second dimension of the weight tensor

    Returns:
        torch.Tensor: The permuted weight tensor
    """
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def inverse_permute_weight(w: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """
    Helper function for converting checkpoints from TransformerEngine to PyTorch
    Permute the query weight or key weight of each attention layer

    Args:
        w (torch.Tensor): The weight tensor to permute
        n_heads (int): The number of attention heads
        dim1 (int): The first dimension of the weight tensor
        dim2 (int): The second dimension of the weight tensor

    Returns:
        torch.Tensor: The permuted weight tensor
    """
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def pytorch_to_hf_state_dict(
    state_dict: Dict[str, torch.Tensor], model_config: Dict[str, Any], tensor_parallel_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Convert a PyTorch state dict to HuggingFace format for LLM models.

    Args:
        state_dict (Mapping[str, torch.Tensor]):
            The original PyTorch model's state dictionary.
            This is a mapping where keys are layer names and values are the corresponding PyTorch tensors
            containing the model weights.

        model_config (Mapping[str, Any]):
            The configuration of the model. This dictionary contains parameters such as:
            - n_layers: (int) The number of transformer layers.
            - n_heads: (int) The number of attention heads.
            - dim: (int) The hidden size of the model.
            - n_kv_heads: (int, optional) The number of key-value heads for multi-query attention.

    Returns:
        Mapping[str, torch.Tensor]:
            The converted HuggingFace state dictionary. This dictionary maps HuggingFace transformer-compatible
            layer names to the corresponding model weights.
    """
    not_supported_key_substrings = ["cross_attention", "q_norm", "k_norm"]
    for key in state_dict.keys():
        if any(substr in key for substr in not_supported_key_substrings):
            raise ValueError(f"Key {key} is not supported in HuggingFace format.")
    assert tensor_parallel_size == 1, "Tensor parallel size > 1 is not supported for HuggingFace model export."

    hf_state_dict = {}

    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    dim = model_config["dim"]
    head_dim = model_config["head_dim"]
    if head_dim is None:
        head_dim = model_config["dim"] // model_config["n_heads"]

    num_key_value_heads = model_config.get("n_kv_heads", n_heads)
    key_value_dim = head_dim * num_key_value_heads

    for layer_i in range(n_layers):
        pt_prefix = f"layers.{layer_i}."
        hf_prefix = f"model.layers.{layer_i}."

        wq = state_dict[f"{pt_prefix}attention.wq.weight"]
        wk = state_dict[f"{pt_prefix}attention.wk.weight"]
        if model_config["pytorch_rope_version"] == "v1":
            wq = permute_weight(
                wq,
                n_heads=n_heads,
                dim1=dim,
                dim2=dim,
            )
            wk = permute_weight(
                wk,
                n_heads=num_key_value_heads,
                dim1=key_value_dim,
                dim2=dim,
            )
        hf_state_dict[f"{hf_prefix}self_attn.q_proj.weight"] = wq
        hf_state_dict[f"{hf_prefix}self_attn.k_proj.weight"] = wk
        hf_state_dict[f"{hf_prefix}self_attn.v_proj.weight"] = state_dict[f"{pt_prefix}attention.wv.weight"]
        hf_state_dict[f"{hf_prefix}self_attn.o_proj.weight"] = state_dict[f"{pt_prefix}attention.wo.weight"]
        hf_state_dict[f"{hf_prefix}mlp.gate_proj.weight"] = state_dict[f"{pt_prefix}feed_forward.w1.weight"]
        hf_state_dict[f"{hf_prefix}mlp.down_proj.weight"] = state_dict[f"{pt_prefix}feed_forward.w2.weight"]
        hf_state_dict[f"{hf_prefix}mlp.up_proj.weight"] = state_dict[f"{pt_prefix}feed_forward.w3.weight"]
        hf_state_dict[f"{hf_prefix}input_layernorm.weight"] = state_dict[f"{pt_prefix}attention_norm.weight"]
        hf_state_dict[f"{hf_prefix}post_attention_layernorm.weight"] = state_dict[f"{pt_prefix}ffn_norm.weight"]

    # Add non-layer weights
    hf_state_dict["model.embed_tokens.weight"] = state_dict["tok_embeddings.weight"]
    hf_state_dict["model.norm.weight"] = state_dict["norm.weight"]
    hf_state_dict["lm_head.weight"] = state_dict["output.weight"]

    return hf_state_dict


def maybe_convert_checkpoint_to_backend(
    state_dict: Dict[str, torch.Tensor],
    target_backend: str,
    model_config,
    source_backend: str = None,
    is_tensor_parallel_shard: bool = False,
    tensor_parallel_size: int = None,
):
    """
    Identify the backend of the checkpoint and convert to the target backend if necessary.

    This function checks the current backend of the state_dict and converts it to the target backend
    if they don't match. It supports conversions between PyTorch, TransformerEngine, and HuggingFace backends.

    Args:
        state_dict (Dict[str, torch.Tensor]): The model state dictionary to convert.
        target_backend (str): The desired backend format ('pytorch', 'transformer_engine', or 'huggingface').
        model_config: Configuration of the model, used in conversion process.
        source_backend (str, optional): The current backend of the state_dict. If not specified, the function will identify the backend.
        is_tensor_parallel_shard (bool, optional): Whether the state_dict is a tensor parallel shard. Defaults to False.
        tensor_parallel_size (int, optional): The tensor parallel size. If not specified, the model_config will be modified.
    Returns:
        Dict[str, torch.Tensor]: The converted state dictionary in the target backend format.

    Raises:
        ValueError: If the conversion between the identified backend and target backend is not supported.
    """
    # Identify the current backend of the checkpoint
    state_dict = process_state_dict(state_dict)  # Remove unnecessary keys
    if source_backend is None:
        source_backend = identify_checkpoint_backend(state_dict)
    if source_backend == target_backend:
        return state_dict
    else:
        if tensor_parallel_size is None:
            tensor_parallel_size = model_config["tensor_parallel_size"] if is_tensor_parallel_shard else 1
        # Convert to target backend
        if source_backend == "pytorch" and target_backend == "transformer_engine":
            return pytorch_to_te_state_dict(state_dict, model_config, tensor_parallel_size=tensor_parallel_size)
        elif source_backend == "transformer_engine" and target_backend == "pytorch":
            return te_to_pytorch_state_dict(state_dict, model_config, tensor_parallel_size=tensor_parallel_size)
        elif source_backend == "pytorch" and target_backend == "huggingface":
            return pytorch_to_hf_state_dict(state_dict, model_config, tensor_parallel_size=tensor_parallel_size)
        else:
            raise ValueError(f"Conversion from {source_backend} to {target_backend} is not supported.")
