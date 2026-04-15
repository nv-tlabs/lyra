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

import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from megatron.core import InferenceParams, ModelParallelConfig, parallel_state
from safetensors.torch import load_file
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict1.autoregressive.configs.base.model import TrainingModelConfig as ModelConfig
from cosmos_predict1.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos_predict1.autoregressive.modules.mm_projector import MultimodalProjector
from cosmos_predict1.autoregressive.networks.vit import VisionTransformer, get_vit_config

# from cosmos_predict1.autoregressive.training.networks.transformer_medusa import TransformerMedusa
from cosmos_predict1.autoregressive.tokenizer.tokenizer import DiscreteMultimodalTokenizer
from cosmos_predict1.autoregressive.training.networks.transformer import (
    Transformer,
    TransformerBlock,
    TransformerBlockTE,
)
from cosmos_predict1.autoregressive.utils.checkpoint import (
    get_partial_state_dict,
    maybe_convert_checkpoint_to_backend,
    obtain_tensor_parallel_state_dict,
    process_state_dict,
    substrings_to_ignore,
)
from cosmos_predict1.autoregressive.utils.misc import random_dropout
from cosmos_predict1.autoregressive.utils.parallel import broadcast_data_batch_in_tp_cp_group, get_batch_on_this_cp_rank
from cosmos_predict1.autoregressive.utils.sampling import (
    decode_n_tokens,
    decode_one_token,
    prefill,
    sample_top_k,
    sample_top_p,
)
from cosmos_predict1.diffusion.training.utils.fsdp_helper import apply_fsdp_checkpointing, hsdp_device_mesh
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.misc import download_from_s3_with_cache, sync_s3_dir_to_local
from cosmos_predict1.utils.model import Model


class AutoRegressiveTrainingModel(Model):
    """
    A class to build and use a Llama model for text generation.

    Methods:
        build: Build a Llama instance by initializing and loading a model checkpoint.
        generate: Generate text sequences based on provided prompts using the language generation model.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: DiscreteMultimodalTokenizer,
        config: ModelConfig,
        model_parallel: ModelParallelConfig = None,
        vision_encoder: VisionTransformer = None,
        mm_projector: MultimodalProjector = None,
    ):
        """
        Initialize the Llama instance with a model and tokenizer.

        Args:
            model (Transformer): The Transformer model for text generation.
            tokenizer (Tokenizer): The tokenizer for encoding and decoding text.
            config (Config): The configuration for the Llama model.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.precision = self.model.precision
        self.vision_encoder = vision_encoder
        self.mm_projector = mm_projector
        assert (self.vision_encoder is None) == (self.mm_projector is None), (
            "vision_encoder and mm_projector should be " "both None or not None simultaneously"
        )
        self.model_parallel = model_parallel
        self.monitor_output_logits = False
        self.inference_params = None
        # self.insert_medusa_head = self.config.insert_medusa_head

        if self.config.freeze_vision_encoder and vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            log.critical("Vision encoder parameters are frozen.")

        num_params = self.get_num_params()
        log.info(f"Number of model parameters: {round(num_params / 1e9, 3)}B")

    def get_num_params(
        self,
    ) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        broadcast_data_batch_in_tp_cp_group(data_batch)
        # get the context embedding and mask
        context = data_batch.get("context", None)
        context_mask = data_batch.get("context_mask", None)
        if context is not None:
            if self.config.embedding_dropout > 0:
                context = random_dropout(
                    context,
                    self.config.embedding_dropout,
                )
            context = misc.to(context, device="cuda")
        if context_mask is not None:
            context_mask = misc.to(context_mask, device="cuda")
        action = data_batch.get("action", None)
        if action is not None:
            action = misc.to(action, device="cuda")
        # Input tokens
        tokens, token_boundaries = self.tokenizer.tokenize(data_batch)
        tokens = misc.to(tokens, device="cuda")
        # Tokens to predict
        labels = data_batch.get("labels", None)
        # Token Mask (Note: this is not attention mask)
        masks = data_batch.get("token_mask", None)
        apply_token_mask = masks is not None
        if masks is None:
            masks = torch.ones_like(tokens, dtype=torch.bool)
        masks = misc.to(masks, device="cuda")
        assert (
            data_batch.get("labels", None) is None or apply_token_mask
        ), "The code is not tested for the case when both labels and token_mask are provided."

        if self.config.ignore_first_num_tokens > 0:
            assert self.config.ignore_first_num_tokens < masks.shape[1]
            masks[:, : self.config.ignore_first_num_tokens] = False
        seq_len = tokens.shape[1]

        # Boradcast inputs to TP and CP ranks, alternatively we can use the `_broadcast` function from cosmos/diffusion/v1
        # Currently we only handled video tokens (with label and mask) and text tokens (with mask), action and other inputs might also need to be handled
        if parallel_state.get_context_parallel_world_size() > 1:
            # Turn on CP
            cp_group = parallel_state.get_context_parallel_group()
            self.model.enable_context_parallel(cp_group)
            tokens = get_batch_on_this_cp_rank(tokens)
            masks = get_batch_on_this_cp_rank(masks)
            if labels is not None:
                labels = get_batch_on_this_cp_rank(labels)
        if self.vision_encoder is None:
            logits = self.model.forward(
                tokens=tokens,
                input_pos=None,
                context=context,
                context_mask=context_mask,
                action=action,
                total_seq_len=seq_len,
            )
        else:
            assert "images" in data_batch
            images = data_batch["images"]
            if images.ndim == 5:
                # The shape is (batch_size, n_images_per_sample, C, H, W). Flatten the first two dimensions.
                images = images.view(-1, *images.shape[2:])
            assert images.ndim == 4, f"Invalid shape: {images.shape}"
            token_embeddings = self.embed_vision_language_features(tokens, images)
            logits = self.model.forward(
                token_embeddings=token_embeddings,
                input_pos=None,
                context=context,
                context_mask=context_mask,
                action=action,
                total_seq_len=seq_len,
            )

        if labels is None:
            # For auto-regressive models, the labels are the same as the
            # input tokens shifted by one position
            logits = logits[:, :-1]
            masks = masks[:, :-1]
            labels = tokens[:, 1:].clone()

        batch_size = tokens.shape[0]
        # Apply ignore_index
        for sample_num in range(batch_size):
            if self.tokenizer.training_type == "text_to_video":
                # For text-to-video training, we do not compute the loss of text part
                # Hence, we set the labels of text tokens to that of ignore_index
                if len(token_boundaries["text"]) > 0:
                    labels[sample_num][0 : token_boundaries["text"][sample_num][1] - 1] = self.tokenizer.ignore_index
            elif self.tokenizer.training_type == "class_to_image":
                # For class-to-image training, we do not compute the loss of class part
                # Hence, we set the labels of class tokens to that of ignore_index
                labels[sample_num][0 : token_boundaries["class"][sample_num][1] - 1] = self.tokenizer.ignore_index

        ignore_index = self.tokenizer.ignore_index
        if self.config.ignore_first_num_tokens > 0 or apply_token_mask:
            labels[~masks] = ignore_index

        output_batch = {
            "encode_tokens": tokens,
            "logits": logits.detach(),
            "labels": labels.detach(),
            "ignore_index": ignore_index,
        }

        if self.monitor_output_logits:
            self.gather_output_logits_stats(logits, labels, output_batch, ignore_index)

        logits = logits.flatten(0, 1)
        labels = labels.flatten(0, 1)

        # Main cross entropy loss
        ce_loss = F.cross_entropy(
            input=logits,
            target=labels,
            ignore_index=ignore_index,  # ignore prompt (turn prompt tokens into pad_id here)
        )

        # Z-loss
        log_z = torch.logsumexp(logits, dim=-1)  # shape: [B, seq_len]
        z_loss = self.config.z_loss_coeff * (log_z**2).mean()

        # Combined loss
        total_loss = ce_loss + z_loss

        return output_batch, total_loss  # skip returning output logits

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a validation step for the model, which is the same as the training step (but without backpropagation).
        """
        return self.training_step(data_batch, iteration)

    @torch.no_grad()
    def gather_output_logits_stats(
        self, logits: torch.Tensor, labels: torch.Tensor, output_batch: Dict, ignore_index: int = None
    ):
        """
        Gather statistics of the output logits, including mean, norm, and max values.
        """
        bs, seq_len, dim = logits.shape
        logits = logits.reshape(-1, dim)
        if ignore_index is not None:
            select_index = labels.view(-1) != ignore_index
            acc = labels.view(-1)[select_index] == logits.argmax(dim=1)[select_index]
            acc = acc.float().mean().view(-1, 1)

            logits = logits[select_index]
        output_batch.update(
            {
                "logits_mean": logits.mean(dim=1).detach(),
                "logits_norm": torch.linalg.vector_norm(logits, dim=1).detach(),
                "logits_max": logits.max(dim=1).values.detach(),
                "acc": acc.detach() * 100,
            }
        )

    @torch.no_grad()
    def image_encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the image input state to continuous latent and discrete indices.
        """
        latent, indices = self.tokenizer.image_tokenizer.encode(state)
        return latent, indices

    @torch.no_grad()
    def image_decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode the discrete indices to RGB images.
        """
        return self.tokenizer.image_tokenizer.decode(indices)

    @torch.no_grad()
    def video_encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the video input state to continuous latent and discrete indices.
        """
        latent, indices = self.tokenizer.video_tokenizer.encode(state)
        return latent, indices

    @torch.no_grad()
    def video_decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode the discrete indices to RGB videos.
        """
        if self.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap > 0:
            return self.tokenizer.video_tokenizer.decode_with_overlap(
                indices, temporal_overlap=self.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap
            )
        else:
            return self.tokenizer.video_tokenizer.decode(indices)

    @staticmethod
    def load_llm_checkpoint(
        ckpt_path: str = "",
        model: Transformer = None,
        **kwargs,
    ) -> None:
        """
        Load a LLM checkpoint from the specified path.
        """
        with misc.timer(f"loading checkpoint from {ckpt_path}"):
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                mmap=True,  # load the checkpoint in memory-mapped mode
            )
        llm_checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
        llm_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="model.")
        llm_checkpoint = process_state_dict(llm_checkpoint, prefix_to_remove="model.")
        with misc.timer("loading state_dict into model"):
            missing_keys, unexpected_keys = model.load_state_dict(llm_checkpoint, strict=True)

    @staticmethod
    def build(
        seed: int = 1,
        train_from_scratch: bool = False,
        model_config: ModelConfig = ModelConfig(),
        fsdp_checkpointer: Any = None,
        tokenizer_config: TokenizerConfig = None,
        model_parallel: ModelParallelConfig = None,
        shard_checkpoint: bool = True,
        download_rank_sync: bool = True,
        **kwargs,
    ) -> "AutoRegressiveTrainingModel":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            train_from_scratch (bool, optional): Flag indicating whether to train the model from scratch. Defaults to False.
            model_config (ModelConfig, optional): The model configuration for the Llama instance. Defaults to ModelConfig().
            fsdp_checkpointer (Any, optional): The FSDP checkpointer for the Llama instance. Defaults to None.
            tokenizer_config (TokenizerConfig, optional): The tokenizer configuration for the Llama instance. Defaults to None.
            shard_checkpoint (bool, optional): Whether to split the checkpoint by Tensor Parallelism before loading. Defaults to False.
            download_rank_sync (bool, optional): Whether to download the checkpoint in a rank-synchronized manner. Defaults to True.
        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory.

        Note:
            This method sets the device to CUDA and loads the pre-trained model and tokenizer.
        """
        tensor_parallel_size = 1 if model_parallel is None else model_parallel.tensor_model_parallel_size
        # seed must be the same in all processes
        torch.manual_seed(seed)

        # Initialize model configuration parameters
        llama_params = {}

        # Load checkpoint and model parameters
        if not train_from_scratch:
            if model_config.ckpt_path is None:
                # If ckpt_path is not provided, we assume the model checkpoint is saved in the ckpt_dir
                ckpt_dir = sync_s3_dir_to_local(
                    s3_dir=model_config.ckpt_dir,
                    s3_credential_path=model_config.s3_credential_path,
                    cache_dir=model_config.cache_dir,
                )

                # We prioritize safetensors version over the pytorch version, since the former is
                # much faster for checkpoint loading.
                checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
                if len(checkpoints) == 0:
                    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

                assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
                assert (
                    len(checkpoints) == 1
                ), f"multiple checkpoint files found in {ckpt_dir} (currently only one is supported)"
                ckpt_path = str(checkpoints[0])  # Assuming single checkpoint for non-parallel case

                if os.path.exists(Path(ckpt_dir) / "params.json"):
                    with open(Path(ckpt_dir) / "params.json", "r") as f:
                        llama_params = json.loads(f.read())
                else:
                    log.info(
                        f"No params.json found in the checkpoint directory ({ckpt_dir}). "
                        f"Using default model config."
                    )

            else:
                # If ckpt_path is provided, we load the model from the specified path,
                # and use the default model configuration
                ckpt_path = download_from_s3_with_cache(
                    s3_path=model_config.ckpt_path,
                    s3_credential_path=model_config.s3_credential_path,
                    cache_dir=model_config.cache_dir,
                    rank_sync=download_rank_sync,
                )

            for key, value in llama_params.items():
                # Override the default model configuration with the parameters from the checkpoint
                setattr(model_config, key, value)

            with misc.timer(f"loading checkpoint from {ckpt_path}"):
                if ckpt_path.endswith("safetensors"):
                    # Load with safetensors API
                    checkpoint = load_file(ckpt_path, device="cpu")
                else:
                    # The pytorch version
                    checkpoint = torch.load(
                        ckpt_path,
                        map_location="cpu",
                        mmap=True,  # load the checkpoint in memory-mapped mode
                    )
            llm_checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint

            # If the checkpoint backend is different from the model backend, convert the checkpoint
            # to be compatible with the model backend
            # If shard_checkpoint is True, the loaded checkpoint is the whole model checkpoint (will be sharded later)
            # instead of a tensor-parallel sharded checkpoint
            llm_checkpoint = maybe_convert_checkpoint_to_backend(
                llm_checkpoint,
                target_backend=model_config.backend,
                model_config=model_config,
                tensor_parallel_size=tensor_parallel_size if not shard_checkpoint else 1,
                is_tensor_parallel_shard=tensor_parallel_size > 1 and not shard_checkpoint,
            )
            if model_config.vision_encoder is not None:
                # For vanilla VLM ckpt before fine-tuning, `checkpoint['model']` only contains LLM weights, and `checkpoint['vision_encoder']`
                #   and `checkpoint['mm_projector']` are both for those weights
                # For fine-tuned VLM ckpt, `checkpoint['model']` contains all LLM, mm_projector and vision_encoder weights
                if "vision_encoder" in checkpoint:
                    log.info("Using pretrained vision_encoder")
                    vit_checkpoint = checkpoint["vision_encoder"]
                else:
                    log.info("Using fine-tuned vision_encoder")
                    vit_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="vision_encoder.")
                    vit_checkpoint = process_state_dict(vit_checkpoint, prefix_to_remove="vision_encoder.")
                if "mm_projector" in checkpoint:
                    log.info("Using pretrained mm_projector")
                    projector_checkpoint = checkpoint["mm_projector"]
                else:
                    log.info("Using fine-tuned mm_projector")
                    projector_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="mm_projector.")
                    projector_checkpoint = process_state_dict(projector_checkpoint, prefix_to_remove="mm_projector.")
                assert (
                    len(vit_checkpoint) > 0 and len(projector_checkpoint) > 0
                ), "vit_checkpoint and projector_checkpoint cannot be empty. We do not support random initialization for vision_encoder and mm_projector."

        tokenizer = DiscreteMultimodalTokenizer(tokenizer_config)

        precision = getattr(torch, model_config.precision)
        torch.set_default_dtype(precision)
        log.info(f"Setting torch default dtype to {precision}")

        # if model_config.insert_medusa_head:
        #     model = TransformerMedusa(
        #         params=model_config,
        #         model_parallel=model_parallel,
        #         tokenizer_config=tokenizer_config,
        #         init_weights=train_from_scratch,
        #     )
        # else:
        model = Transformer(
            params=model_config,
            model_parallel=model_parallel,
            tokenizer_config=tokenizer_config,
            init_weights=train_from_scratch,
        )
        model_kwargs = {}
        # [Optional] Initialize vision encoder and multimodal projector (for vision-language tasks)
        if model_config.vision_encoder is not None:
            assert model_config.mm_projector is not None, "mm_projector must be provided if vision_encoder is provided."
            vit_config = get_vit_config(model_config.vision_encoder)
            vision_encoder = VisionTransformer.build(
                vit_config,
                hidden_dropout=model_config["hidden_dropout"],
                attention_dropout=model_config["attention_dropout"],
                set_parallel_mode=model_config["set_parallel_mode"],
                model_parallel=model_parallel,
                attention_tp=tensor_parallel_size > 1,
            )

            mm_projector = MultimodalProjector(
                mm_projector_type=model_config.mm_projector, in_dim=vit_config["dim"], out_dim=model_config["dim"]
            )
            model_kwargs.update({"vision_encoder": vision_encoder, "mm_projector": mm_projector})

        # Perform vocab expansion
        if tokenizer.vocab_size > model.vocab_size:
            log.info(f"Expanding vocab size to {tokenizer.vocab_size}")
            # For text-to-video training, we only expand the embedding layer but not the output (unembedding) layer,
            expand_output_layer = not (tokenizer.training_type == "text_to_video")
            model.expand_vocab(tokenizer.vocab_size, init_method="gaussian", expand_output_layer=expand_output_layer)

        if not train_from_scratch:
            if shard_checkpoint:
                # Shard the checkpoint according to tensor parallelism.
                with misc.timer("sharding checkpoint according to tensor parallelism"):
                    if model_parallel is not None:
                        assert model_parallel.tensor_model_parallel_size == model_config["tensor_model_parallel_size"]
                    llm_checkpoint = obtain_tensor_parallel_state_dict(
                        llm_checkpoint,
                        tensor_parallel_size=tensor_parallel_size,
                        tensor_parallel_rank=parallel_state.get_tensor_model_parallel_rank(),
                        model_config=model_config,
                    )
                if model_config.vision_encoder is not None:
                    # Shard vision encoder and multimodal projector weights
                    vit_checkpoint = obtain_tensor_parallel_state_dict(
                        vit_checkpoint,
                        tensor_parallel_size=tensor_parallel_size,
                        tensor_parallel_rank=parallel_state.get_tensor_model_parallel_rank(),
                        model_config=vit_config,
                    )

            if model_config.vision_encoder is not None:
                # Take the LLM weights (starting with "model.") from the VLM checkpoint
                llm_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="model.")
            # Remove the "model." prefix in the state_dict
            llm_checkpoint = process_state_dict(llm_checkpoint, prefix_to_remove="model.")
            with misc.timer("loading state_dict into model"):
                missing_keys, unexpected_keys = model.load_state_dict(llm_checkpoint, strict=True)
            # Remove keys with "_extra_state" suffix in missing_keys (defined by TransformerEngine for FP8 usage)
            missing_keys = [k for k in missing_keys if not k.endswith("_extra_state")]
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

            if model_config.vision_encoder is not None:
                # Load vision encoder and multimodal projector weights
                vision_encoder.load_state_dict(vit_checkpoint)
                mm_projector.load_state_dict(projector_checkpoint)
                if model_config.vision_encoder_in_channels != 3:
                    vision_encoder.expand_in_channels(model_config.vision_encoder_in_channels)

        model = model.to(precision)  # ensure model parameters are in the correct precision
        log.info(f"Model config: {model_config}")

        # if model_config.insert_medusa_head:
        #     from projects.cosmos.ar.v1.model_medusa import LlamaMedusa

        #     model_class = LlamaMedusa
        # else:
        model_class = AutoRegressiveTrainingModel
        if model_config.fsdp_enabled:
            raise NotImplementedError("FSDP is not implemented for AutoRegressiveTrainingModel")
            # model_kwargs["fsdp_checkpointer"] = fsdp_checkpointer
            # model_class = FSDPLlama
        return model_class(model, tokenizer, model_config, **model_kwargs)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        logit_clipping_range: list = [],
        seed: int = 0,
        images: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            top_k (int, optional): Top-k value for top-k sampling. Defaults to None. If not None, top-k sampling will be used instead of top-p sampling.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        assert top_k is None or top_p is None, f"Only one of top_k ({top_k} or top_p ({top_p} should be specified."
        if top_p is not None:
            log.info(f"Using top-p sampling with p={top_p} and temperature={temperature}")
        elif top_k is not None:
            log.info(f"Using top-k sampling with k={top_k} and temperature={temperature}")
        else:
            log.info("Not applying top-k or top-p sampling. Will use top-k sampling with k=None")

        self.model.set_inference_flag(True)
        misc.set_random_seed(seed)
        # Initialization and Assertions
        if isinstance(self.model.params, list):
            # During training, model.params is a list
            log.info(
                f"Find self.model.params is a list, use self.config instead. Get max_batch_size={self.config.max_batch_size}, max_seq_len={self.config.max_seq_len}"
            )
            params = self.config
        else:
            params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        if self.config.backend == "transformer_engine":
            self.inference_params = InferenceParams(
                max_batch_size=params.max_batch_size, max_sequence_length=params.max_seq_len
            )

        # Calculate Prompt Lengths
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = params.max_seq_len
        assert (
            max_gen_len + max_prompt_len <= total_len
        ), f"max_gen_len + max_prompt_len={max_gen_len + max_prompt_len} exceeds max_seq_len={total_len}"

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")

        # Fill tokens tensor with prompt tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        # Flag to check if image embeddings have been passed to the model - we only need to pass them once
        # since we have KV cache.
        passed_image_embeddings = False

        # If all prompts are of max length, compute initial logits and logprobs
        if min_prompt_len == total_len:
            input_pos = torch.arange(tokens.shape[1], dtype=torch.long, device="cuda")
            if images is None:
                logits = self.model.forward(
                    tokens=tokens,
                    input_pos=input_pos,
                    inference_params=self.inference_params,
                    context=context,
                    context_mask=context_mask,
                    action=action,
                )
            else:
                token_embeddings = self.embed_vision_language_features(tokens, images)
                logits = self.model.forward(
                    token_embeddings=token_embeddings,
                    input_pos=input_pos,
                    inference_params=self.inference_params,
                    context=context,
                    context_mask=context_mask,
                    action=action,
                )
                passed_image_embeddings = True
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), dtype=torch.long, device="cuda")

        # Main generation loop
        log.info(f"Start generating the next {total_len - min_prompt_len} tokens. This will take a while..")
        for cur_pos in range(min_prompt_len, total_len):
            input_pos = torch.arange(prev_pos, cur_pos, dtype=torch.long, device="cuda")
            if images is not None and not passed_image_embeddings:
                token_embeddings = self.embed_vision_language_features(tokens[:, prev_pos:cur_pos], images)
                logits = self.model.forward(
                    token_embeddings=token_embeddings,
                    input_pos=input_pos,
                    inference_params=self.inference_params,
                    context=context,
                    context_mask=context_mask,
                    action=action,
                )
                passed_image_embeddings = True
            else:
                logits = self.model.forward(
                    tokens=tokens[:, prev_pos:cur_pos],
                    input_pos=input_pos,
                    inference_params=self.inference_params,
                    context=context,
                    context_mask=context_mask,
                    action=action,
                )

            if self.config.backend == "transformer_engine":
                self.inference_params.sequence_len_offset += logits.shape[1]

            # Apply temperature scaling and nucleus sampling
            if len(logit_clipping_range) > 0:
                min_clip_index = logit_clipping_range[0]
                max_clip_index = logit_clipping_range[1]
                logits_clipped = logits[:, :, min_clip_index:max_clip_index]
            else:
                logits_clipped = logits
                min_clip_index = 0

            if temperature > 0:
                if top_p is not None:
                    next_token = sample_top_p(logits_clipped, temperature=temperature, top_p=top_p)[0]
                else:
                    next_token = sample_top_k(logits_clipped, temperature=temperature, top_k=top_k)[0]
            else:
                next_token = torch.argmax(logits_clipped[:, -1, :], dim=-1)

            next_token += min_clip_index

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # Calculate log probabilities if requested
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            # Check if end-of-sequence token is reached
            eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
            prev_pos = cur_pos
            # Break the loop if all sequences have reached an end-of-sequence token
            if all(eos_reached):
                log.info(f"Reach end of sequence, current pos: {cur_pos}; maximum pos: {total_len}")
                break
        # Convert log probabilities to list if required
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []

        # Process and collect the output tokens and log probabilities
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        self.model.set_inference_flag(False)
        return (out_tokens, out_logprobs if logprobs else None)

    @torch.no_grad()
    def fast_generate(
        self,
        prompt_tokens: List[List[int]] | torch.Tensor,
        max_gen_len: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_gen_seq: int = 1,
        logprobs: bool = False,
        echo: bool = False,
        seed: int = 0,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        compile_decode: bool = True,
        compile_prefill: bool = False,
        verbose: bool = True,
        stop_tokens: Optional[Set[int]] = None,
    ):
        """
        Fast auto-regressive generation. Currently only supports input batch size = 1.
        Args:
            prompt_tokens (List[List[int]] | torch.Tensor): A single prompt of shape (1, seq_len).
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_k (int, optional): Top-k value for top-k sampling. Defaults to None.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to None.
            num_gen_seq (int, optional): Number of outputs to generate given the same prompt. Defaults to 1. When temperature == 0, num_gen_seq must be 1 because the generation is deterministic.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            logit_clipping_range (list, optional): Range of logits to clip. Defaults to [].
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            compile_decode (bool, optional): Flag indicating whether to compile the decoding function. Defaults to True.
            compile_prefill (bool, optional): Flag indicating whether to compile the prefill function. Defaults to False.
            verbose (bool, optional): Flag indicating whether to print the the time. Defaults to False.
        """
        assert (
            top_p is None or top_k is None
        ), f"Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
        if top_p is not None:
            log.info(f"Using top-p sampling with p={top_p} and temperature={temperature}")
        elif top_k is not None:
            log.info(f"Using top-k sampling with k={top_k} and temperature={temperature}")
        else:
            log.info("Not applying top-k or top-p sampling. Will use top-k sampling with k=None")

        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        # Experimental features to reduce compilation times, will be on by default in future
        torch._inductor.config.fx_graph_cache = True
        # torch._functorch.config.enable_autograd_cache = True

        self.model.set_inference_flag(True)
        misc.set_random_seed(seed)

        assert not logprobs, "logprobs are not supported for fast_generate yet"
        # Examine if the function prefil and decode_one_token functions are compiled yet. If not, compile them based on the flags
        if compile_decode and not getattr(self, "inference_decode_compiled", False):
            self.decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
            self.inference_decode_compiled = True
            log.critical("Compiled decode_one_token function. Note: the first run will be slower due to compilation")
        if compile_prefill and not getattr(self, "inference_prefill_compiled", False):
            self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
            self.inference_prefill_compiled = True
            log.critical("Compiled prefill function. Note: the first run will be slower due to compilation")

        if not hasattr(self, "decode_one_token"):
            self.decode_one_token = decode_one_token
        if not hasattr(self, "prefill"):
            self.prefill = prefill

        # Initialization and Assertions
        if isinstance(self.model.params, list):
            # During training, model.params is a list
            log.info(
                f"Find self.model.params is a list, use self.config instead. Get max_batch_size={self.config.max_batch_size}, max_seq_len={self.config.max_seq_len}"
            )
            params = self.config
        else:
            params = self.model.params
        if isinstance(prompt_tokens, list):
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.view(1, -1)
        else:
            assert prompt_tokens.ndim == 2, f"prompt_tokens has shape {prompt_tokens.shape}"
        batch_size, prompt_len = prompt_tokens.shape
        total_len = min(params.max_seq_len, max_gen_len + prompt_len)
        if max_gen_len + prompt_len > params.max_seq_len:
            log.warning(
                f"max_gen_len + prompt_len={max_gen_len + prompt_len} exceeds max_seq_len={params.max_seq_len}, truncate max_gen_len to {params.max_seq_len - prompt_len}"
            )
            max_gen_len = params.max_seq_len - prompt_len

        if context_mask is not None:
            context_mask = context_mask.to(dtype=torch.bool)
            if context_mask.ndim == 2:
                assert (
                    context_mask.shape[0] == batch_size
                ), f"batch_size mismatch: {context_mask.shape[0]} != {batch_size}"
                # Unsqueeze it to make it of shape [batch_size, 1, 1, context_seq_len]
                context_mask = context_mask.view(batch_size, 1, 1, -1)

        if num_gen_seq > 1:
            assert (
                batch_size == 1
            ), f"num_gen_seq > 1 is only supported for a single prompt, got {len(prompt_tokens)} prompts"
            log.critical(f"Generating {num_gen_seq} sequences with the same prompt")
            assert (
                num_gen_seq <= params.max_batch_size
            ), f"num_gen_seq={num_gen_seq} exceeds max_batch_size={params.max_batch_size}"
            # repeat the prompt tokens for num_gen_seq times
            prompt_tokens = prompt_tokens.repeat(num_gen_seq, 1)
            assert prompt_tokens.shape == (
                num_gen_seq,
                prompt_len,
            ), f"prompt_tokens must be of shape (num_gen_seq, seq_len), got {prompt_tokens.shape}"
            batch_size = len(prompt_tokens)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(batch_size, total_len, dtype=prompt_tokens.dtype, device=prompt_tokens.device)
        empty[:, :prompt_len] = prompt_tokens
        seq = empty
        input_pos = torch.arange(0, prompt_len, device="cuda")

        if verbose:
            prefill_start = time.time()

        # Prefill stage
        next_token = self.prefill(
            self.model,
            prompt_tokens,
            input_pos=input_pos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        if verbose:
            prefill_time = time.time() - prefill_start

        seq[:, [prompt_len]] = next_token.to(dtype=seq.dtype)
        input_pos = torch.tensor([prompt_len], dtype=torch.long, device="cuda")
        stop_tokens = self.tokenizer.stop_tokens if stop_tokens is None else stop_tokens
        stop_tokens = torch.tensor(list(stop_tokens), dtype=torch.long, device="cuda")

        if verbose:
            decode_start = time.time()
        # Decode stage
        generated_tokens = decode_n_tokens(
            self.model,
            next_token.view(batch_size, -1),
            input_pos,
            max_gen_len - 1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
            decode_one_token_function=self.decode_one_token,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        gen_len = len(generated_tokens)
        if verbose:
            decode_time = time.time() - decode_start
            prefill_throughput = prompt_len / prefill_time
            decode_throughput = gen_len / decode_time
            log.info(f"[Prefill] Time: {prefill_time:.2f}s; Throughput: {prefill_throughput:.2f} tokens/s")
            log.info(f"[Decode] Time: {decode_time:.2f}s; Throughput: {decode_throughput:.2f} tokens/s")

        generated_tokens = torch.cat(generated_tokens, dim=1)

        log.critical(f"generated_tokens: {generated_tokens.shape}")
        seq = seq[:, : prompt_len + 1 + gen_len]
        seq[:, prompt_len + 1 :] = generated_tokens
        if not echo:
            seq = seq[:, prompt_len:]
        return seq, None

    def embed_vision_language_features(self, input_ids: torch.Tensor, images: torch.tensor) -> torch.Tensor:
        """
        Embed vision and language features into a combined representation.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            images (torch.tensor): Input images.

        Returns:
            torch.Tensor: Combined vision-language features.

        Raises:
            AssertionError: If vision encoder or mm projector is not initialized,
                            or if dimensions mismatch.
        """
        # Ensure vision encoder and mm projector are initialized
        assert self.vision_encoder is not None
        assert self.mm_projector is not None

        # Get image token ID and validate it
        image_token_id = self.vision_encoder.image_token_id
        assert isinstance(image_token_id, int) and image_token_id >= 0, f"Invalid image_token_id: {image_token_id}"

        # Identify text and image locations in the input
        text_locations = input_ids != image_token_id
        image_locations = input_ids == image_token_id

        # Process text features
        text_features = self.model.tok_embeddings(input_ids[text_locations])

        # Process image features
        images = images.to(device=text_features.device, dtype=text_features.dtype)
        vit_outputs = self.vision_encoder(images)
        image_features = self.mm_projector(vit_outputs)

        # Get dimensions
        B, seq_len = input_ids.shape
        N_total = B * seq_len
        N_txt, D_txt = text_features.shape
        N_img, N_patch, D_img = image_features.shape

        # Reshape image features
        image_features = image_features.reshape(N_img * N_patch, D_img)

        # Validate dimensions
        assert D_txt == D_img, f"Text features dim {D_txt} should be equal to image features dim {D_img}"
        assert (
            N_total == N_txt + N_img * N_patch
        ), f"seq_len {seq_len} should be equal to N_txt + N_img*N_Patch {(N_txt, N_img * N_patch, image_locations.sum().item())}"

        # Combine text and image features
        combined_features = torch.empty(
            (B, seq_len, D_txt),
            dtype=text_features.dtype,
            device=text_features.device,
        )
        combined_features[text_locations, :] = text_features
        combined_features[image_locations, :] = image_features

        return combined_features

    def on_after_backward(self, iteration: int = 0):
        """
        Hook after loss.backward() is called.

        This method is called immediately after the backward pass, allowing for custom operations
        or modifications to be performed on the gradients before the optimizer step.

        So far, this method is used to all-reduce layernorm grads for tensor/sequence parallelism.

        Args:
            iteration (int): Current iteration number.
        """
        for module in self.children():
            if hasattr(module, "on_after_backward"):
                module.on_after_backward(iteration)

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """Hook before zero_grad() is called.

        Args:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            iteration (int): Current iteration number.
        """
        for module in self.children():
            if hasattr(module, "on_before_zero_grad"):
                module.on_before_zero_grad(optimizer, scheduler, iteration)

    @property
    def fsdp_wrap_block_cls(self):
        """
        Return the transformer block class to wrap with FSDP.
        """
        if self.config.backend == "pytorch":
            return TransformerBlock
        elif self.config.backend == "transformer_engine":
            return TransformerBlockTE
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

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
        actual_missing_keys = []
        for key in missing_keys:
            if not any(substring in key for substring in substrings_to_ignore):
                actual_missing_keys.append(key)
        if strict:
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}")
        return _IncompatibleKeys(actual_missing_keys, unexpected_keys)


# class FSDPLlama(Llama):
#     def __init__(
#         self, model: Transformer, tokenizer: DiscreteMultimodalTokenizer, config: ModelConfig, fsdp_checkpointer: Any
#     ):
#         self.fsdp_checkpointer = fsdp_checkpointer
#         super().__init__(model, tokenizer, config)
#         self.set_up_fsdp()

#     def set_up_fsdp(self):
#         """
#         Set up FSDP for the model.
#         """

#         model = self.model
#         # detach the model from the parent class
#         self.model = None
#         del self.model

#         # build FSDP sharding strategy and device_mesh
#         strategy = {
#             "full": ShardingStrategy.FULL_SHARD,
#             "hybrid": ShardingStrategy.HYBRID_SHARD,
#             "none": ShardingStrategy.NO_SHARD,
#         }[self.config.fsdp["sharding_strategy"]]
#         log.critical(f"Using {strategy} sharding strategy for FSDP")

#         if self.config.fsdp["sharding_strategy"] == "hybrid":
#             sharding_group_size = self.config.fsdp["sharding_group_size"]
#             device_mesh = hsdp_device_mesh(
#                 sharding_group_size=sharding_group_size,
#             )
#         else:
#             device_mesh = hsdp_device_mesh(
#                 sharding_group_size=distributed.get_world_size(),
#             )
#         parallel_state.fsdp_device_mesh = device_mesh

#         if distributed.get_rank() == 0:
#             # only load model in rank0 to reduce network traffic and sync later
#             self.fsdp_checkpointer.load_model_during_init(model, is_ema=False)

#         if not hasattr(self, "fsdp_wrap_block_cls"):
#             raise ValueError("Networks does not have fsdp_wrap_block_cls attribute, please check the net definition")
#         fsdp_blocks_cls = self.fsdp_wrap_block_cls
#         fsdp_blocks_cls = (
#             list(fsdp_blocks_cls) if isinstance(fsdp_blocks_cls, (list, tuple, set)) else [fsdp_blocks_cls]
#         )
#         log.critical(f"Using FSDP blocks {fsdp_blocks_cls}")

#         log.critical(f"Using wrap policy {self.config.fsdp['policy']}")

#         if self.config.fsdp["policy"] == "size":
#             # Size based policy won't work for transformers because the tokenizers need to be accessible at multiple
#             # layers (input / output). This is handled by this sharding strategy.
#             min_num_params = self.config.fsdp["min_num_params"]
#             log.critical(f"Using {min_num_params} as the minimum number of parameters for auto-wrap policy")
#             log.info("If using a Transformer model. Please use the transformer wrap policy.")
#             wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
#         else:
#             # Use the auto wrap policy for transformers
#             wrap_policy = functools.partial(
#                 transformer_auto_wrap_policy,
#                 transformer_layer_cls=set(fsdp_blocks_cls),
#             )
#         tensor_kwargs = {"device": "cuda", "dtype": model.precision}

#         # Wrap the model with FSDP and attach it back to this class
#         self.model = FSDP(
#             model.to(**tensor_kwargs),
#             sync_module_states=True,  # it can reduce network traffic by only loading model in rank0 and sync
#             sharding_strategy=strategy,
#             auto_wrap_policy=wrap_policy,
#             device_id=torch.cuda.current_device(),
#             device_mesh=device_mesh,
#             limit_all_gathers=True,
#             use_orig_params=True,  # Do not flatten the parameter structure. Useful for layer_dependent lrs, etc.
#         )

#         if self.config.act_ckpt_enabled:
#             # Apply activation checkpointing
#             apply_fsdp_checkpointing(self.model, list_block_cls=fsdp_blocks_cls)

#         # Clean up memory
#         torch.cuda.empty_cache()

#     def state_dict(self) -> Dict:
#         raise NotImplementedError("FSDPLlama does not support state_dict, use state_dict_model and FSDPCheckpointer")

#     @misc.timer("FSDP state_dict_model")
#     def state_dict_model(self) -> Dict:
#         """
#         Get the model state_dict for checkpoint saving in the FSDP mode.
#         """
#         with FSDP.summon_full_params(self.model):
#             pass
#         with FSDP.state_dict_type(
#             self.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#         ):
#             model_state = self.model.state_dict()
#         # No support for EMA yet.
#         ema_model_state = None
#         return {
#             "model": model_state,
#             "ema": ema_model_state,
#         }

#     def load_state_dict(self, state_dict: Dict, strict: bool = True, assign: bool = False) -> None:
#         raise NotImplementedError("FSDPLlama does not support load_state_dict, using FSDPCheckpointer")

#     def init_optimizer_scheduler(
#         self, optimizer_config: LazyDict, scheduler_config: LazyDict
#     ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
#         """
#         Initialize the optimizer and scheduler for FSDP model.

#         Args:
#             optimizer_config (LazyDict): The optimizer configuration.
#             scheduler_config (LazyDict): The scheduler configuration.

#         Returns:
#             tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]: The optimizer and scheduler.
#         """
#         optimizer, scheduler = super().init_optimizer_scheduler(optimizer_config, scheduler_config)
#         self.fsdp_checkpointer.load_optim_scheduler_during_init(
#             self.model,
#             optimizer,
#             scheduler,
#         )
#         return optimizer, scheduler

#     def get_ckpt_postfix(self) -> Tuple[str, int]:
#         """Get the checkpoint file postfix. check FSDPCheckpointer for more details

#         Returns:
#             postfix (str): The postfix of the checkpoint file.
#             replicate_idx, shard_idx (int), current gpu replicate_idx, shard_idx in FSDP \
#                 we will not save each ema model in each GPU, \
#                 ema model with same rate will be saved once
#             total_ema_num (int)
#         """
#         replicate_idx, shard_idx = parallel_state.fsdp_device_mesh.get_coordinate()
#         # !!! EMA is not supported
#         if replicate_idx == 0:
#             return "", 0, shard_idx, 0
#         return "", replicate_idx, shard_idx, 0
