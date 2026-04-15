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

import copy
from typing import Callable, List, Optional

import torch
from megatron.core import ModelParallelConfig

from cosmos_predict1.autoregressive.configs.base.model import ModelConfig, TrainingModelConfig
from cosmos_predict1.autoregressive.configs.base.tokenizer import (
    TextTokenizerConfig,
    TokenizerConfig,
    VideoTokenizerConfig,
    create_discrete_video_fsq_tokenizer_state_dict_config,
)
from cosmos_predict1.autoregressive.tokenizer.image_text_tokenizer import ImageTextTokenizer
from cosmos_predict1.autoregressive.tokenizer.text_tokenizer import TextTokenizer
from cosmos_predict1.autoregressive.training.model import AutoRegressiveTrainingModel
from cosmos_predict1.utils import log
from cosmos_predict1.utils.config import EMAConfig
from cosmos_predict1.utils.lazy_config import LazyCall as L

# Common architecture specifications
BASE_CONFIG = {"n_kv_heads": 8, "norm_type": "rmsnorm", "norm_eps": 1e-5, "ffn_hidden_size": 14336}
COSMOS_ARCHITECTURES = {
    "1b": {
        "n_layers": 16,
        "dim": 2048,
        "n_heads": 32,
    },
    "4b": {
        "n_layers": 16,
        "dim": 4096,
        "n_heads": 32,
    },
    "12b": {
        "n_layers": 40,
        "dim": 5120,
        "n_heads": 32,
        "head_dim": 128,
    },
}

COSMOS_YARN_CONFIG = {
    "original_latent_shape": [3, 40, 64],
    "apply_yarn": True,
    "yarn_beta_fast": 4,
    "yarn_beta_slow": 1,
    "yarn_scale": 2,
}

# Llama3 architecture specifications for different model sizes
LLAMA3_ARCHITECTURES = {
    "8b": {
        "n_layers": 32,
        "dim": 4096,
        "n_heads": 32,
        "ffn_hidden_size": 14336,
    },
}
# Llama3.1 uses YaRN for long context support (context of 128k tokens)
LLAMA_YARN_CONFIG = {
    "apply_yarn": True,
    "yarn_scale": 8,
    "yarn_beta_fast": 4,
    "yarn_beta_slow": 1,
}

# Mistral architecture specifications for different model sizes
MISTRAL_ARCHITECTURES = {
    "12b": {
        "n_layers": 40,
        "dim": 5120,
        "n_heads": 32,
        "ffn_hidden_size": 14336,
        "head_dim": 128,
    },
}

PIXTRAL_VISION_ARCHITECTURES = {
    "12b": {"vision_encoder": "pixtral-12b-vit", "mm_projector": "mlp"},
}


def get_model_arch_specs(model_size: str, model_family: str = "mistral", pretrained: bool = False) -> dict:
    """
    Get the model architecture specifications for the given model size, model family and pretrained status.

    Args:
        model_size (str): Model size. Choices: "1b", "3b", "4b", "7b", etc.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral"
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        dict: A dictionary containing the model architecture specifications.
    """
    arch_specs = copy.deepcopy(BASE_CONFIG)
    model_size = model_size.lower()
    if model_family.startswith("cosmos"):
        arch_specs.update(COSMOS_ARCHITECTURES[model_size])
    elif model_family.startswith("llama"):
        arch_specs.update(LLAMA3_ARCHITECTURES[model_size])
    elif model_family in ["mistral", "pixtral"]:
        arch_specs.update(MISTRAL_ARCHITECTURES[model_size])
        if model_family == "pixtral":
            arch_specs.update(PIXTRAL_VISION_ARCHITECTURES[model_size])
    else:
        raise ValueError(f"Model family {model_family} is not supported.")

    if pretrained:
        if model_family == "cosmos":
            if model_size == "12b":
                arch_specs.update(COSMOS_YARN_CONFIG)
                log.debug(f"Using YaRN for RoPE extension with config: {COSMOS_YARN_CONFIG}")
            else:
                pass
        elif model_family in ["llama", "llama3"]:
            pretrained_specs = {
                "rope_theta": 500000,
                "max_seq_len": 8192,
                "vocab_size": 128256,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "llama3.1":
            pretrained_specs = {
                "rope_theta": 500000,
                "max_seq_len": 131072,
                "original_seq_len": 8192,
                "vocab_size": 128256,
                **LLAMA_YARN_CONFIG,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "mistral":
            assert model_size == "12b", "We only support Mistral-Nemo-12B model."
            pretrained_specs = {
                "rope_theta": 1000000,
                "max_seq_len": 128000,
                "vocab_size": 131072,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "pixtral":
            assert model_size == "12b", "We only support Pixtral 12B model."
            pretrained_specs = {"rope_theta": 1000000000, "max_seq_len": 128000, "vocab_size": 131072}
            arch_specs.update(pretrained_specs)
        else:
            raise ValueError(f"Model family {model_family} doesn't have a pretrained config.")

    return arch_specs


def create_text_model_config(
    model_ckpt_path: str,
    tokenizer_path: str,
    tensor_model_parallel_size: int = 1,
    model_family: str = "mistral",
    model_size: str = "12b",
    is_instruct_model: bool = True,
    max_seq_len: int = None,
    max_batch_size: int = 1,
    rope_dim: str = "1D",
    add_special_tokens: bool = True,
    pytorch_rope_version: str = None,
) -> dict:
    """Create a text model for training or inference.
    Args:
        model_ckpt_path (str): Path to the model checkpoint.
        tokenizer_path (str): Path to the tokenizer folder.
        tensor_model_parallel_size (int): Number of tensor model parallel groups.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral".
        model_size (str): Model size. Choices: "1b", "3b", "4b", "7b", "8b", "72b", etc.
        is_instruct_model (bool): Whether the model is an instruct model.
        inference (bool): Whether to create the model for inference.
        max_seq_len (int): Maximum sequence length.
        max_batch_size (int): Maximum batch size.
        rope_dim (str): RoPE dimension. Choices: "1D", "3D".
        add_special_tokens (bool): Whether to add special tokens.
    Returns:
        dict: A dictionary containing the model configuration, which can be used to instantiate the model object.
    """
    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=True)
    if max_seq_len is not None:
        # Override the max_seq_len if provided
        model_arch_specs["max_seq_len"] = max_seq_len
    if pytorch_rope_version is not None:
        model_arch_specs["pytorch_rope_version"] = pytorch_rope_version
    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        precision="bfloat16",
        ckpt_path=model_ckpt_path,
        use_qk_normalization=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rope_dim=rope_dim,
        **model_arch_specs,
    )

    tokenizer_config = TokenizerConfig(
        text_tokenizer=TextTokenizerConfig(
            config=L(TextTokenizer)(
                model_family=model_family,
                is_instruct_model=is_instruct_model,
                local_path=tokenizer_path,
            ),
            data_key="text",
            tokenizer_offset=model_config.vocab_size,
            tokenize_here=False,
            vocab_size=model_config.vocab_size,
        ),
        seq_len=model_config.max_seq_len,
        training_type="text_only",
        add_special_tokens=add_special_tokens,
    )
    return model_config, tokenizer_config


def create_vision_language_model_config(
    model_ckpt_path: str,
    tokenizer_ckpt_path: str,
    tensor_model_parallel_size: int = 1,
    model_family: str = "pixtral",
    model_size: str = "12b",
    is_instruct_model: bool = True,
    max_batch_size: int = 1,
    rope_dim: str = "1D",
    add_special_tokens: bool = True,
    max_seq_len: int = None,
    vision_encoder_in_channels: int = 3,
    fuse_qkv: bool = False,
    pytorch_rope_version: str = None,
) -> dict:
    """Create a vision-language model for training or inference.
    Args:
        model_ckpt_path (str): Path to the model checkpoint.
        tokenizer_ckpt_path (str): Path to the tokenizer checkpoint.
        tensor_model_parallel_size (int): Number of tensor model parallel groups.
        model_family (str): Model family. Choices: "pixtral".
        model_size (str): Model size. Choices: "12b".
        is_instruct_model (bool): Whether the model is an instruct model.
        rope_dim (str): RoPE dimension. Choices: "1D".
        add_special_tokens (bool): Whether to add special tokens.
        max_seq_len (int): Maximum sequence length.
        vision_encoder_in_channels (int): Number of channels in the input image for the vision encoder. Default is 3, you can specify to int larger than 3. E.g. if you have 4 channel images where last channel is binary mask, set this to 4.
        fuse_qkv (bool): Whether to fuse the QKV linear layers.
    Returns:
        dict: A dictionary containing the model configuration, which can be used to instantiate the model object.
    """
    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=True)
    if max_seq_len is not None:
        # Override the max_seq_len if provided
        model_arch_specs["max_seq_len"] = max_seq_len
    if pytorch_rope_version is not None:
        model_arch_specs["pytorch_rope_version"] = pytorch_rope_version

    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        precision="bfloat16",
        ckpt_path=model_ckpt_path,
        use_qk_normalization=False,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rope_dim=rope_dim,
        vision_encoder_in_channels=vision_encoder_in_channels,
        fuse_qkv=fuse_qkv,
        **model_arch_specs,
    )
    # Vision-language tokenizer
    tokenizer_config = TokenizerConfig(
        text_tokenizer=TextTokenizerConfig(
            config=L(ImageTextTokenizer)(
                model_family=model_family,
                is_instruct_model=is_instruct_model,
                image_processor_path=tokenizer_ckpt_path,
                tokenizer_path=tokenizer_ckpt_path,
            ),
            data_key="image_text_interleaved",
            tokenizer_offset=model_config.vocab_size,
            tokenize_here=False,
            vocab_size=model_config.vocab_size,
        ),
        seq_len=model_config.max_seq_len,
        training_type="image_text_interleaved",
        add_special_tokens=add_special_tokens,
    )
    return model_config, tokenizer_config


def create_video2world_model_config(
    model_ckpt_path: str,
    tokenizer_ckpt_path: str,
    tensor_model_parallel_size: int = 1,
    model_family: str = "cosmos",
    model_size: str = "4b",
    pixel_chunk_duration: int = 9,
    num_video_frames: int = 36,
    compression_ratio: List[int] = [8, 16, 16],
    original_seq_len: int = 8192,
    num_condition_latents_t: int = 1,
    num_tokens_to_ignore: int = -1,
    batch_size: int = 2,
    video_tokenizer_config_creator: Callable = create_discrete_video_fsq_tokenizer_state_dict_config,
    rope_dim: str = "3D",
    add_special_tokens: bool = True,
    video_height: int = 384,
    video_width: int = 640,
    use_qk_normalization: bool = True,
    insert_cross_attn: bool = False,
    insert_cross_attn_every_k_layers: int = 1,
    context_dim: int = 1024,
    training_type: str = "video_to_video",
    pad_to_multiple_of: Optional[int] = 64,
    vocab_size: int = 64000,
    apply_abs_pos_emb: bool = False,
) -> dict:
    """Create a video-to-world model config.
    Args:
        tensor_model_parallel_size (int): Number of tensor model parallel groups.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral".
        model_size (str): Model size. Choices: "1b", "8b", "3b".
        pixel_chunk_duration (int): Number of frames in each chunk.
        num_video_frames (int): Number of video frames.
        compression_ratio (List[int]): Compression ratio for the video frames. Choices: [8, 16, 16] or [4, 8, 8].
        original_seq_len (int): Original sequence length.
        apply_yarn (bool): Whether to apply YaRN for long context scaling.
        yarn_beta_fast (Optional[int]): Fast beta for YaRN.
        yarn_beta_slow (Optional[int]): Slow beta for YaRN.
        yarn_scale (Optional[int]): Scale factor for ctx extension.
        use_qk_normalization (bool): Whether to use Query-Key normalization.
        training_type (str): Type of training task.
        batch_size (int): Batch size.
        video_tokenizer_config_creator (Callable): Method that takes "pixel_chunk_duration: int" and "version: str" as arguments and returns video tokenizer config
        video_tokenizer_version (str): Version of the video tokenizer.
        num_condition_latents_t (int): Number of conditioning latent channels
        num_tokens_to_ignore (int) = Number of tokens to ignore. This takes the precedence
        video_height (int): Height of the video frame. Defaults to 384.
        video_width (int): Width of the video frame. Defaults to 640.
        rope_dim (str): RoPE dimension. Choices: "1D", "3D".
        add_special_tokens (bool): Whether to add special tokens, use False for 2D/3D RoPE.
        pad_to_multiple_of (int): Pad the token sequence length to the nearest multiple of this number. Defaults to 64.
        vocab_size (int): Vocabulary size.
        apply_abs_pos_emb (bool): Whether to apply absolute positional embeddings.
    Returns:
        dict: A dictionary containing the model configuration representing the model object, can be instantiated.
    """
    assert (
        pixel_chunk_duration % compression_ratio[0] == 1
    ), f"pixel_chunk_duration({pixel_chunk_duration}) should be k*n + 1 (k={compression_ratio[0]})"
    latent_chunk_duration = (pixel_chunk_duration - 1) // compression_ratio[0] + 1
    latent_height = video_height // compression_ratio[1]
    latent_width = video_width // compression_ratio[2]
    # Do some math to compute the video latent shape and sequence length
    assert (
        num_video_frames % pixel_chunk_duration == 0
    ), f"num_video_frames {num_video_frames} should be divisible by pixel_chunk_duration {pixel_chunk_duration}"
    video_latent_shape = [
        num_video_frames // pixel_chunk_duration * latent_chunk_duration,
        latent_height,
        latent_width,
    ]
    # product of video_latent_shape
    num_token_video_latent = video_latent_shape[0] * video_latent_shape[1] * video_latent_shape[2]
    if add_special_tokens:
        seq_len = num_token_video_latent + 3  # Sequence length per batch, max_seq_len + 3
        seq_len = (seq_len + 63) // 64 * 64  # Round up to multiple of 64
    # for text to video, we need to add <bov> token to indicate the start of the video
    elif training_type == "text_to_video":
        seq_len = num_token_video_latent + 1
    else:
        seq_len = num_token_video_latent

    if seq_len % pad_to_multiple_of != 0:
        # Round up to the nearest multiple of pad_to_multiple_of
        seq_len = ((seq_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=True)

    # Whether skip the loss for first chunk or not, note the first token is already skipped when computing the loss
    # If num_tokens_to_ignore is specified, use it.
    # Else compute it from num_condition_latents_t
    if num_tokens_to_ignore < 0:
        num_tokens_to_ignore = latent_height * latent_width * num_condition_latents_t
        if not add_special_tokens and num_condition_latents_t > 0:
            # If there are no special tokens (bov), do a -1 so that you can compute the loss
            # from the first token of the next chunk
            num_tokens_to_ignore -= 1

    model_config = ModelConfig(
        video_height=video_height,
        video_width=video_width,
        max_seq_len=seq_len,
        max_batch_size=batch_size,
        precision="bfloat16",
        ckpt_path=model_ckpt_path,
        use_qk_normalization=use_qk_normalization,
        vocab_size=64000,
        original_seq_len=original_seq_len,
        tensor_model_parallel_size=tensor_model_parallel_size,
        video_latent_shape=video_latent_shape,
        num_video_frames=num_video_frames,
        rope_dim=rope_dim,
        pad_to_multiple_of=pad_to_multiple_of,
        insert_cross_attn=insert_cross_attn,
        insert_cross_attn_every_k_layers=insert_cross_attn_every_k_layers,
        context_dim=context_dim,
        apply_abs_pos_emb=apply_abs_pos_emb,
        **model_arch_specs,
    )

    video_tokenizer_config = video_tokenizer_config_creator(
        tokenizer_ckpt_path, pixel_chunk_duration, compression_ratio
    )
    tokenizer_config = TokenizerConfig(
        text_tokenizer=None,
        video_tokenizer=VideoTokenizerConfig(
            config=video_tokenizer_config,
            data_key="video",
            tokenizer_offset=0,  # Since there is no text embeddings in the model. Note this only apply when the model is trained from scratch. If we use text pretrained model, the offset will be vocab_size of text token.
            tokenize_here=True,
            max_seq_len=num_token_video_latent,
            vocab_size=vocab_size,
        ),
        seq_len=seq_len,
        training_type=training_type,
        add_special_tokens=add_special_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return model_config, tokenizer_config


def create_video2world_model(
    tensor_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    shard_checkpoint: bool = False,
    model_family: str = "cosmos",
    model_size: str = "1b",
    backend: str = "pytorch",
    pixel_chunk_duration: int = 9,
    num_video_frames: int = 36,
    compression_ratio: List[int] = [8, 16, 16],
    original_seq_len: int = 8192,
    apply_yarn: bool = False,
    yarn_beta_fast: Optional[int] = None,
    yarn_beta_slow: Optional[int] = None,
    yarn_scale: Optional[int] = None,
    num_condition_latents_t: int = 1,
    num_tokens_to_ignore: int = -1,
    batch_size: int = 1,
    fsdp_enabled: bool = False,
    act_ckpt_enabled: bool = False,
    video_tokenizer_config_creator: Callable = create_discrete_video_fsq_tokenizer_state_dict_config,
    rope_dim: str = "3D",
    add_special_tokens: bool = False,
    video_height: int = 384,
    video_width: int = 640,
    original_latent_shape: Optional[List[int]] = None,
    use_qk_normalization: bool = True,
    sequence_parallel: bool = False,
    insert_cross_attn: bool = False,
    insert_cross_attn_every_k_layers: int = 1,
    context_dim: int = 1024,
    finetune_layers_with_cross_attn: bool = False,
    finetune_layers_without_cross_attn: bool = False,
    use_action_condition: bool = False,
    action_embedding_mode: Optional[str] = "mlp",
    action_dim: int = 8,  # ACTION_DIM,
    action_embedding_dim: int = 1024,
    group_causal_mask_mode: Optional[str] = None,
    training_type: str = "video_to_video",
    pad_to_multiple_of: Optional[int] = 1,
    z_loss_coeff: float = 1e-4,
    temporal_overlap: int = 0,
    embedding_dropout: float = 0.0,
    insert_medusa_head: bool = False,
    ft_medusa_option: str = "fft",
    medusa_num_heads: int = 7,
    medusa_num_layers: int = 1,
    medusa_concat_heads: bool = True,
    fuse_qkv: bool = False,
    zero_init_cross_attn_proj: bool = False,
    concat_action_to_context: bool = False,
    tokenizer_ckpt_path: str = "checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/ema.jit",
) -> dict:
    """Create a video-to-video model for training.
    Args:
        tensor_model_parallel_size (int): Number of tensor model parallel groups.
        context_parallel_size (int): Number of context parallel groups.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral".
        model_size (str): Model size. Choices: "1b", "8b", "3b".
        backend (str): Backend for the model. Choices: "pytorch", "transformer_engine".
        pixel_chunk_duration (int): Number of frames in each chunk.
        num_video_frames (int): Number of video frames.
        compression_ratio (List[int]): Compression ratio for the video frames. Choices: [8, 16, 16] or [4, 8, 8].
        original_seq_len (int): Original sequence length.
        apply_yarn (bool): Whether to apply YaRN for long context scaling.
        yarn_beta_fast (Optional[int]): Fast beta for YaRN.
        yarn_beta_slow (Optional[int]): Slow beta for YaRN.
        yarn_scale (Optional[int]): Scale factor for ctx extension.
        fsdp_enabled (bool): Whether Fully Sharded Data Parallel (FSDP) is enabled.
        act_ckpt_enabled (bool): Whether activation checkpointing is enabled.
        use_qk_normalization (bool): Whether to use Query-Key normalization.
        training_type (str): Type of training task.
        batch_size (int): Batch size.
        video_tokenizer_config_creator (Callable): Method that takes "pixel_chunk_duration: int" and "version: str" as arguments and returns video tokenizer config
        video_tokenizer_version (str): Version of the video tokenizer.
        num_condition_latents_t (int): Number of conditioning latent channels
        num_tokens_to_ignore (int) = Number of tokens to ignore. This takes the precedence
        video_height (int): Height of the video frame. Defaults to 384.
        video_width (int): Width of the video frame. Defaults to 640.
        rope_dim (str): RoPE dimension. Choices: "1D", "2D", "3D".
        add_special_tokens (bool): Whether to add special tokens, use False for 2D/3D RoPE.
        original_latent_shape (list): Original latent shape before RoPE scaling.
        sequence_parallel (bool): Whether to enable sequence parallelism.
        insert_cross_attn (bool): Whether to insert the cross-attention layers after each multi-head self-attention (MSA) layer.
        insert_cross_attn_every_k_layers (int): Insert cross-attention layers every k TransformerLayers.
        context_dim (Optional[int]): The dimensionality of cross-attention embedding, e.g., T5 embed feature dim.
        finetune_layers_with_cross_attn (bool): Whether to finetune Transformer layers w/ CA (cross-attn).
        finetune_layers_without_cross_attn (bool): Whether to finetune Transformer layers w/o CA (cross-attn).
        use_action_condition (bool): Whether to use action condition.
        action_embedding_mode (Optional[str]): The mode of the robot action embedding. Choices: "matrix", "mlp".
        action_dim (int): Dimension of the raw robot action tensor (e.g., 7 for DROID, [Δx, Δy, Δz, rx, ry, rz, gripper_open]).
        action_embedding_dim (int): Dimension of the action embedding.
        group_causal_mask_mode (Optional[str]): The mode of the group causal mask. Choices: "causal", "group_diagonal".
        pad_to_multiple_of (int): Pad the token sequence length to the nearest multiple of this number. Defaults to 64.
        z_loss_coeff (float): Coefficient for the z loss.
        temporal_overlap (int): Temporal overlap in the latent space.
        embedding_dropout (float): Dropout rate for the embeddings.
        insert_medusa_head (bool): Whether to insert the Medusa head.
        ft_medusa_option (str): Options on which layers to finetune, choices like:
            "fft": fully fine-tune both medusa heads and all LLM backbone;
            "head": fine-tune medusa heads;
            "head_out": fine-tune medusa heads, and the output layer;
            "head_out_last_k_layer": fine-tune medusa heads, the output layer, and the last k layer(s) of the LLM backbone.
        medusa_num_heads (int): Number of heads in the Medusa head.
        medusa_num_layers (int): Number of layers in the Medusa head.
        medusa_concat_heads (bool): Whether to concatenate multiple medusa heads into fused matrix, only applicable when medusa_num_layers = 1.
        fuse_qkv (bool): Whether to fuse the QKV linear layers.
        zero_init_cross_attn_proj (bool): Whether to zero-initialize the cross-attention projection weights (default False).
        concat_action_to_context (bool): Whether to concatenate the action embedding to the context (default False).
    Returns:
        dict: A dictionary containing the model configuration representing the model object, can be instantiated.
    """
    assert (
        pixel_chunk_duration % compression_ratio[0] == 1
    ), f"pixel_chunk_duration({pixel_chunk_duration}) should be k*n + 1 (k={compression_ratio[0]})"
    latent_chunk_duration = (pixel_chunk_duration - 1) // compression_ratio[0] + 1
    latent_height = video_height // compression_ratio[1]
    latent_width = video_width // compression_ratio[2]
    # Compute the video latent shape and sequence length
    if temporal_overlap == 0:
        assert (
            num_video_frames % pixel_chunk_duration == 0
        ), f"num_video_frames {num_video_frames} should be divisible by pixel_chunk_duration {pixel_chunk_duration}"
        video_latent_shape = [
            num_video_frames // pixel_chunk_duration * latent_chunk_duration,
            latent_height,
            latent_width,
        ]

    else:
        # Calculate temporal overlap in the latent space
        temporal_overlap_latent = temporal_overlap // compression_ratio[0]

        # Calculate the effective number of latent chunks for the video
        latent_chunks = (num_video_frames - temporal_overlap) // (pixel_chunk_duration - temporal_overlap)

        # Compute the total duration of the latent chunks, accounting for overlap
        effective_latent_duration = (
            latent_chunk_duration - temporal_overlap_latent
        ) * latent_chunks + temporal_overlap_latent

        # Define the shape of the video in the latent space
        video_latent_shape = [
            effective_latent_duration,  # Temporal dimension
            latent_height,  # Height in the latent space
            latent_width,  # Width in the latent space
        ]

    # product of video_latent_shape
    num_token_video_latent = video_latent_shape[0] * video_latent_shape[1] * video_latent_shape[2]
    if add_special_tokens:
        seq_len = num_token_video_latent + 3  # Sequence length per batch, max_seq_len + 3
        seq_len = (seq_len + 63) // 64 * 64  # Round up to multiple of 64
    # for text to video, we need to add <bov> token to indicate the start of the video
    elif training_type == "text_to_video":
        seq_len = num_token_video_latent + 1
    else:
        seq_len = num_token_video_latent

    if seq_len % pad_to_multiple_of != 0:
        # Round up to the nearest multiple of pad_to_multiple_of
        seq_len = ((seq_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=False)

    inference = False  # False for training, True for inference
    # set_parallel_mode = True
    set_parallel_mode = tensor_model_parallel_size > 1
    attention_tp = True

    if context_parallel_size > 1:
        assert backend == "transformer_engine", "Context parallelism is only supported in transformer engine."

    if tensor_model_parallel_size > 1:
        assert set_parallel_mode, "Tensor model parallelism is only supported in parallel mode."

    # Whether skip the loss for first chunk or not, note the first token is already skipped when computing the loss
    # If num_tokens_to_ignore is specified, use it.
    # Else compute it from num_condition_latents_t
    if num_tokens_to_ignore < 0:
        num_tokens_to_ignore = latent_height * latent_width * num_condition_latents_t
        if not add_special_tokens and num_condition_latents_t > 0:
            # If there are no special tokens (bov), do a -1 so that you can compute the loss
            # from the first token of the next chunk
            num_tokens_to_ignore -= 1

    model_config = TrainingModelConfig(
        video_height=video_height,
        video_width=video_width,
        max_seq_len=seq_len,
        max_batch_size=batch_size,
        inference=inference,
        backend=backend,
        precision="bfloat16",
        ema=EMAConfig(enabled=False),
        act_ckpt_enabled=act_ckpt_enabled,
        fsdp_enabled=fsdp_enabled,
        cache_dir=None,
        ckpt_path="checkpoints/Cosmos-Predict1-4B/model.pt",
        use_qk_normalization=use_qk_normalization,
        vocab_size=64000,
        ignore_first_num_tokens=num_tokens_to_ignore,
        apply_yarn=apply_yarn,
        yarn_beta_fast=yarn_beta_fast,
        yarn_beta_slow=yarn_beta_slow,
        original_seq_len=original_seq_len,
        yarn_scale=yarn_scale,
        context_parallel_size=context_parallel_size,
        tensor_model_parallel_size=tensor_model_parallel_size,
        set_parallel_mode=set_parallel_mode,
        attention_tp=attention_tp,
        video_latent_shape=video_latent_shape,
        num_video_frames=num_video_frames,
        rope_dim=rope_dim,
        original_latent_shape=original_latent_shape,
        pad_to_multiple_of=pad_to_multiple_of,
        sequence_parallel=sequence_parallel,
        insert_cross_attn=insert_cross_attn,
        insert_cross_attn_every_k_layers=insert_cross_attn_every_k_layers,
        context_dim=context_dim,
        finetune_layers_with_cross_attn=finetune_layers_with_cross_attn,
        finetune_layers_without_cross_attn=finetune_layers_without_cross_attn,
        use_action_condition=use_action_condition,
        action_embedding_mode=action_embedding_mode,
        action_dim=action_dim,
        action_embedding_dim=action_embedding_dim,
        group_causal_mask_mode=group_causal_mask_mode,
        z_loss_coeff=z_loss_coeff,
        embedding_dropout=embedding_dropout,
        insert_medusa_head=insert_medusa_head,
        ft_medusa_option=ft_medusa_option,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        medusa_concat_heads=medusa_concat_heads,
        fuse_qkv=fuse_qkv,
        zero_init_cross_attn_proj=zero_init_cross_attn_proj,
        concat_action_to_context=concat_action_to_context,
        **model_arch_specs,
    )

    tokenizer_config = TokenizerConfig(
        text_tokenizer=None,
        video_tokenizer=VideoTokenizerConfig(
            config=video_tokenizer_config_creator(
                ckpt_path=tokenizer_ckpt_path, pixel_chunk_duration=pixel_chunk_duration
            ),
            data_key="video",
            tokenizer_offset=0,
            vocab_size=64000,
            tokenize_here=True,
            max_seq_len=num_token_video_latent,
            temporal_overlap=temporal_overlap,
        ),
        seq_len="${model.model_config.max_seq_len}",
        training_type=training_type,
        add_special_tokens=add_special_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    model_parallel = ModelParallelConfig(
        bf16=True,
        params_dtype=getattr(torch, "bfloat16"),
    )
    model_parallel.tensor_model_parallel_size = "${model.model_config.tensor_model_parallel_size}"
    model_parallel.context_parallel_size = "${model.model_config.context_parallel_size}"
    model_parallel.sequence_parallel = "${model.model_config.sequence_parallel}"
    return L(AutoRegressiveTrainingModel.build)(
        seed=0,
        train_from_scratch=True,
        model_config=model_config,
        fsdp_checkpointer=None,
        tokenizer_config=tokenizer_config,
        model_parallel=model_parallel,
        shard_checkpoint=shard_checkpoint,
    )
