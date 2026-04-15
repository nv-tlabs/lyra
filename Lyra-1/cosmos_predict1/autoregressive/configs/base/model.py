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

import attrs

from cosmos_predict1.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos_predict1.utils import config

_ACTION_DIM = 8
from cosmos_predict1.utils.lazy_config import LazyDict


@attrs.define
class ModelConfig:
    """
    A class to hold model configuration arguments.

    Args:
        dim (int): The dimensionality of the input and output of each transformer block.
        n_layers (int): Number of layers in the transformer.
        n_heads (int): Number of attention heads.
        n_kv_heads (Optional[int]): Number of key-value heads. If None, defaults to n_heads. Note: this is equivalent to
            `num_gqa_groups` in TransformerEngine, where GQA means Grouped Query Attention.
        head_dim (Optional[int]): Dimensionality of each head. If None, defaults to dim // n_heads.
        vocab_size (int): Vocabulary size.
        ffn_hidden_size (int): Hidden size for feedforward network.
        norm_eps (float): Epsilon value for normalization.
        rope_theta (float): Theta value for rotary positional embeddings.
        apply_abs_pos_emb (bool): Whether to apply absolute position embeddings.
        max_batch_size (int): Maximum batch size for inference.
        max_seq_len (int): Maximum sequence length for input text.
        fuse_qkv (bool): Whether to fuse QKV in attention. Defaults to True.
        causal_mask (bool): Whether to use causal mask. Defaults to True.
        norm_type (str): Type of normalization layer. Choices: "rmsnorm", "fused_rmsnorm", "layernorm", "np_layernorm".
        precision (str): Data type for the model.
        use_qk_normalization (bool): Whether to enable QK normalization.
        tensor_model_parallel_size (int): Tensor model parallel size. Defaults to 1.
        ckpt_dir (str): Checkpoint directory.
        ckpt_path (str): Checkpoint path.
        apply_yarn (Optional[bool]): Whether to apply YaRN (long-context extension).
        yarn_scale (Optional[float]): Scale factor for YaRN.
        yarn_beta_fast (Optional[int]): Beta fast variable for YaRN (i.e., low_freq_factor in Llama 3.1 RoPE scaling code)
        yarn_beta_slow (Optional[int]): Beta slow variable for YaRN (i.e., high_freq_factor in Llama 3.1 RoPE scaling code)
        original_seq_len (Optional[int]): Original sequence length.
        vision_encoder (Optional[str]): Vision encoder name.
        mm_projector (Optional[str]): Multi-modal projector name.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3, you can specify to int larger than 3. E.g. if you have 4-channel images with the last channel as the alpha channel, set this to 4.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "3D".
        pytorch_rope_version (Optional[str]): Version of the PyTorch RoPE implementation. Choices: "v1", "v2".
        original_latent_shape (Optional[list]): Original shape of the latent tensor needed for rope extension.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3.
        insert_cross_attn (bool): Whether to insert the cross-attention layers after each multi-head self-attention (MSA) layer.
        insert_cross_attn_every_k_layers (int): Insert cross-attention layers every k TransformerLayers.
        context_dim (Optional[int]): The dimensionality of cross-attention embedding, e.g., T5 embed feature dim.
        num_video_frames (Optional[int]): Number of video frames.
        video_height (Optional[int]): Raw video pixel height dimension.
        video_width (Optional[int]): Raw video pixel width dimension.
        video_latent_shape (Optional[list]): Video tokenizer output dimension, in (T,H,W).
    """

    dim: int = attrs.field(default=4096)
    n_layers: int = attrs.field(default=32)
    n_heads: int = attrs.field(default=32)
    n_kv_heads: Optional[int] = attrs.field(default=8)
    head_dim: Optional[int] = attrs.field(default=None)
    vocab_size: int = attrs.field(default=128256)
    ffn_hidden_size: int = attrs.field(default=14336)
    norm_eps: float = attrs.field(default=1e-5)
    rope_theta: float = attrs.field(default=500000)
    apply_abs_pos_emb: bool = attrs.field(default=False)
    max_batch_size: int = attrs.field(default=1)
    max_seq_len: int = attrs.field(default=8192)
    fuse_qkv: bool = attrs.field(default=False)
    causal_mask: bool = attrs.field(default=True)
    norm_type: str = attrs.field(default="rmsnorm")
    precision: str = attrs.field(default="bfloat16")
    use_qk_normalization: bool = False
    tokenizer: Optional[TokenizerConfig] = None
    tensor_model_parallel_size: int = attrs.field(default=1)
    ckpt_dir: Optional[str] = attrs.field(default=None)
    ckpt_path: Optional[str] = attrs.field(
        default=None
    )  # If not None, load the model from this path instead of ckpt_dir
    apply_yarn: Optional[bool] = attrs.field(default=False)
    yarn_scale: Optional[float] = attrs.field(default=None)
    yarn_beta_fast: Optional[int] = attrs.field(default=None)
    yarn_beta_slow: Optional[int] = attrs.field(default=None)
    original_seq_len: Optional[int] = attrs.field(default=None)
    vision_encoder: Optional[str] = attrs.field(default=None)
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    mm_projector: Optional[str] = attrs.field(default=None)
    rope_dim: Optional[str] = attrs.field(default="1D")
    pytorch_rope_version: Optional[str] = attrs.field(default="v2")
    original_latent_shape: Optional[list] = None
    pad_to_multiple_of: Optional[int] = None
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    insert_cross_attn: bool = False
    insert_cross_attn_every_k_layers: int = 1
    context_dim: Optional[int] = attrs.field(default=1024)
    # For video training
    num_video_frames: Optional[int] = None
    # Raw video pixel dimension
    video_height: Optional[int] = None
    video_width: Optional[int] = None
    # Video tokenizer output dimension, in (T,H,W), it's computed by num_video_frames/temporal_compress_factor, video_height/spatial_compression_fact, video_width/spatial_compression_fact
    video_latent_shape: Optional[list] = None

    def __getitem__(self, item):
        return getattr(self, item)


@attrs.define
class TrainingModelConfig:
    """
    A class to hold model configuration arguments.

    Args:
        dim (int): The dimensionality of the input and output of each transformer block.
        n_layers (int): Number of layers in the transformer.
        n_heads (int): Number of attention heads.
        n_kv_heads (Optional[int]): Number of key-value heads. If None, defaults to n_heads. Note: this is equivalent to
            `num_gqa_groups` in TransformerEngine, where GQA means Grouped Query Attention.
        head_dim (Optional[int]): Dimensionality of each head. If None, defaults to dim // n_heads.
        vocab_size (int): Vocabulary size.
        multiple_of (int): Ensures the hidden layer size is a multiple of this value for SwiGLU activation.
        ffn_dim_multiplier (Optional[float]): Multiplier for feedforward network dimension.
        ffn_hidden_size (Optional[int]): Hidden size for feedforward network. If None, use ffn_dim_multiplier to compute it.
        norm_eps (float): Epsilon value for normalization.
        rope_theta (float): Theta value for rotary positional embeddings.
        apply_abs_pos_emb (bool): Whether to apply absolute position embeddings.
        max_batch_size (int): Maximum batch size for inference (determines KV cache size).
        max_seq_len (int): Maximum sequence length for input text (determines KV cache size).
        fuse_qkv (bool): Whether to fuse QKV in attention. Flag for the pytorch backend.
        causal_mask (bool): Whether to use causal mask. Defaults to True.
        flash_attn (bool): Whether to use Flash attention.
        norm_type (str): Type of normalization layer. Choices: "rmsnorm", "fused_rmsnorm", "layernorm", "np_layernorm".
        backend (str): Backend for the model.
        precision (str): Data type for the model.
        ema (config.EMAConfig): Configuration for exponential moving average.
        embedding_dropout(float): Dropout rate for the embedding layer.
        attention_dropout(float): Dropout rate for attention.
        hidden_dropout(float): Dropout after the attention and feed-forward layers (following TransformerEngine's
                implementation in its TransformerLayer class).
        use_qk_normalization (bool): Whether to enable QK normalization.
        inference (bool): Whether the model is used for inference.
        act_ckpt_enabled (bool): Whether to enable activation checkpointing.
        fsdp_enabled (bool): Whether to enable FSDP.
        fsdp (LazyDict): Configuration for FSDP.
        ckpt_dir (str): Checkpoint directory.
        ckpt_path (str): Checkpoint path.
        cache_dir (str): Cache directory.
        apply_yarn (Optional[bool]): Whether to apply YaRN (long-context extension).
        yarn_scale (Optional[float]): Scale factor for YaRN.
        yarn_beta_fast (Optional[int]): Beta fast variable for YaRN (i.e., low_freq_factor in Llama 3.1 RoPE scaling code)
        yarn_beta_slow (Optional[int]): Beta slow variable for YaRN (i.e., high_freq_factor in Llama 3.1 RoPE scaling code)
        original_seq_len (Optional[int]): Original sequence length.
        depth_init (bool): If `True`, then each transformer block init uses its layer ID, and if `False`, each uses the
            total number of transformer blocks. Defaults to `True` (following the TorchTitan implementation of Llama3).
        context_parallel_size (int): Context parallel size. Defaults to 1.
        tensor_model_parallel_size (int): Tensor model parallel size. Defaults to 1.
        sequence_parallel (bool): Whether to use sequence parallelism. Defaults to False.
        set_parallel_mode (bool): It is a boolean flag used by TransformerEngine to handle Tensor Parallelism.
            Essentially, it is equivalent to `tensor_model_parallel_size > 1`. Defaults to `False`.
        attention_tp (bool): Whether to use tensor parallelism for attention layers.
        mm_projector (Optional[str]): Multimodal projector used for vision-language modeling. Defaults to None.
            Choices: "identity", "linear", "mlp", "mlp_downsample".
        video_latent_shape (Optional[list]): Shape of the video latent tensor. [T, H, W]
        image_latent_shape (Optional[list]): Shape of the image latent tensor. [H, W]
        num_video_frames (Optional[int]): Number of video frames.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "2D", "3D".
        pytorch_rope_version (Optional[str]): Version of the RoPE for the `pytorch` backend. "v1" is the Llama implementation, and "v2" is HuggingFace/TransformerEngine implementation.
        original_latent_shape (Optional[list]): Original shape of the latent tensor needed for rope extension.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
        peft_last_n_layers (Optional[int]): Number of last few layers to fine-tune in Parameter Efficient Fine-Tuning (PEFT). When this and peft_every_n_layers are both 0, it means all layers are fine-tuned (FFT).
        peft_every_n_layers (Optional[int]): In Parameter Efficient Fine-Tuning (PEFT), every n layers are unfrozen and can be trained (in flamingo style). When this and peft_last_n_layers are both 0,
            it means all layers are fine-tuned (FFT). For example, for a 40 layer model, n=8 means training layers 7, 15, 23, 31, 39, which includes the final layer.
            It is advised to pick n such that the final layer is included.
        freeze_vision_encoder (bool): Whether to freeze the vision encoder in vision-language model training. Defaults to False.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3, you can specify to int larger than 3. E.g. if you have 4-channel images with the last channel as the alpha channel, set this to 4.
        insert_cross_attn (bool): Whether to insert the cross-attention layers after each multi-head self-attention (MSA) layer.
        insert_cross_attn_every_k_layers (int): Insert cross-attention layers every k TransformerLayers.
        context_dim (Optional[int]): The dimensionality of cross-attention embedding, e.g., T5 embed feature dim.
        finetune_layers_with_cross_attn (bool): Whether to finetune Transformer layers w/ CA (cross-attn).
        finetune_layers_without_cross_attn (bool): Whether to finetune Transformer layers w/o CA (cross-attn).
        use_action_condition (bool): Whether to use the robot action condition.
        action_embedding_mode (Optional[str]): The mode of the robot action embedding. Choices: "matrix", "mlp".
        action_dim (Optional[int]): The dimensionality of the raw robot action tensor (e.g., 7 for DROID, [Δx, Δy, Δz, rx, ry, rz, gripper_open]).
        action_embedding_dim (Optional[int]): The dimensionality of the robot action embedding.
        group_causal_mask_mode (Optional[str]): The mode of the group causal mask. Choices: "causal", "group_diagonal".
        sync_1d_parameters (bool): Whether to synchronize layernorm parameters (1D) across tensor parallel ranks (default True).
            Note: this is to ensure all TP-ranks have the same layernorm parameters.
        z_loss_coeff (float): The coefficient for the z-loss.
        insert_medusa_head (bool): Whether to insert the Medusa head.
        ft_medusa_option (str): Options on which layers to finetune, choices like:
            "fft": fully fine-tune both medusa heads and all LLM backbone;
            "head": fine-tune medusa heads;
            "head_out": fine-tune medusa heads, and the output layer;
            "head_out_last_k_layer": fine-tune medusa heads, the output layer, and the last k layer(s) of the LLM backbone.
        medusa_num_heads (int): Number of heads in the Medusa head.
        medusa_num_layers (int): Number of layers in the Medusa head.
        medusa_concat_heads (bool): Whether to concatenate multiple medusa heads into fused matrix, only applicable when medusa_num_layers = 1.
        zero_init_cross_attn_proj (bool): Whether to initialize the cross-attn proj layer with zeros (default False).
        concat_action_to_context (bool): Whether to concatenate the action embedding to the context (default False).
    """

    dim: int = attrs.field(default=4096)
    n_layers: int = attrs.field(default=32)
    n_heads: int = attrs.field(default=32)
    n_kv_heads: Optional[int] = attrs.field(default=8)
    head_dim: Optional[int] = attrs.field(default=None)
    vocab_size: int = attrs.field(default=128256)
    multiple_of: int = attrs.field(default=1024)  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = attrs.field(default=1.3)
    ffn_hidden_size: Optional[int] = attrs.field(default=None)
    norm_eps: float = attrs.field(default=1e-5)
    rope_theta: float = attrs.field(default=500000)
    apply_abs_pos_emb: bool = attrs.field(default=False)
    max_batch_size: int = attrs.field(default=1)
    max_seq_len: int = attrs.field(default=8192)
    fuse_qkv: bool = attrs.field(default=False)
    causal_mask: bool = attrs.field(default=True)
    flash_attn: bool = attrs.field(default=True)
    norm_type: str = attrs.field(default="rmsnorm")
    backend: str = attrs.field(default="pytorch")
    precision: str = attrs.field(default="bfloat16")
    ema: config.EMAConfig = config.EMAConfig(enabled=False)
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_qk_normalization: bool = False
    tokenizer: Optional[TokenizerConfig] = None
    inference: bool = False
    act_ckpt_enabled: bool = False
    fsdp_enabled: bool = False
    context_parallel_size: int = attrs.field(default=1)
    tensor_model_parallel_size: int = attrs.field(default=1)
    sequence_parallel: bool = attrs.field(default=False)
    set_parallel_mode: bool = attrs.field(default=False)
    fsdp: LazyDict = LazyDict(
        dict(
            policy="auto",  # choices: ["size", "auto"]
            min_num_params=1024,  # Used as policy == "size"
            sharding_strategy="hybrid",  # Choices: ["full", "hybrid"]. "full" means sharding_group_size = world_size
            sharding_group_size=8,  # If None, defaults to min(world_size, 8). Recommends 8 for training on 8-GPU nodes.
        )
    )
    ckpt_dir: Optional[str] = attrs.field(default="")
    ckpt_path: Optional[str] = attrs.field(
        default=None
    )  # If not None, load the model from this path instead of ckpt_dir
    cache_dir: Optional[str] = attrs.field(default="/project/cosmos/ar/cache")
    apply_yarn: Optional[bool] = attrs.field(default=False)
    yarn_scale: Optional[float] = attrs.field(default=None)
    yarn_beta_fast: Optional[int] = attrs.field(default=None)
    yarn_beta_slow: Optional[int] = attrs.field(default=None)
    original_seq_len: Optional[int] = attrs.field(default=None)
    depth_init: bool = attrs.field(default=True)
    ignore_first_num_tokens: int = 0
    z_loss_coeff: float = 1e-4
    attention_tp: bool = False
    vision_encoder: Optional[str] = attrs.field(default=None)
    mm_projector: Optional[str] = attrs.field(default=None)
    rope_dim: Optional[str] = attrs.field(default="1D")
    pytorch_rope_version: Optional[str] = attrs.field(default="v2")
    original_latent_shape: Optional[list] = None
    pad_to_multiple_of: Optional[int] = None
    peft_last_n_layers: Optional[int] = attrs.field(default=0)
    peft_every_n_layers: Optional[int] = attrs.field(default=0)
    freeze_vision_encoder: bool = False
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    insert_cross_attn: bool = False
    insert_cross_attn_every_k_layers: int = 1
    context_dim: Optional[int] = attrs.field(default=1024)
    finetune_layers_with_cross_attn: bool = False
    finetune_layers_without_cross_attn: bool = False
    use_action_condition: bool = False
    action_embedding_mode: Optional[str] = attrs.field(default="mlp")
    action_dim: Optional[int] = attrs.field(default=_ACTION_DIM)
    action_embedding_dim: Optional[int] = attrs.field(default=1024)
    group_causal_mask_mode: Optional[str] = attrs.field(default=None)
    sync_1d_parameters: bool = True
    # hyper-parameters for the medusa head configs
    insert_medusa_head: bool = False
    ft_medusa_option: str = "fft"
    medusa_num_heads: int = 7
    medusa_num_layers: int = 1
    medusa_concat_heads: bool = True
    # For video training
    num_video_frames: Optional[int] = None
    # Raw video pixel dimension
    video_height: Optional[int] = None
    video_width: Optional[int] = None
    # Video tokenizer output dimension, in (T,H,W), it's computed by num_video_frames/temporal_compress_factor, video_height/spatial_compression_fact, video_width/spatial_compression_fact
    video_latent_shape: Optional[list] = None
    # For image training
    image_latent_shape: Optional[list] = None
    # For robot training (action)
    zero_init_cross_attn_proj: bool = False
    # For robot training (action)
    concat_action_to_context: bool = False

    def __getitem__(self, item):
        return getattr(self, item)
