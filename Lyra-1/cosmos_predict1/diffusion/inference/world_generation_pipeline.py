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

import gc
import os
from typing import Any, Optional

import einops
import numpy as np
import torch

from cosmos_predict1.diffusion.inference.inference_utils import (
    generate_world_from_text,
    generate_world_from_video,
    get_condition_latent,
    get_condition_latent_multiview,
    get_video_batch,
    get_video_batch_for_multiview_model,
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
    read_video_or_image_into_frames_BCTHW,
)
from cosmos_predict1.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos_predict1.diffusion.model.model_t2w_multiview import DiffusionMultiviewT2WModel
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos_predict1.diffusion.model.model_v2w_multiview import DiffusionMultiviewV2WModel
from cosmos_predict1.diffusion.model.model_world_interpolator import DiffusionWorldInterpolatorWModel
from cosmos_predict1.diffusion.prompt_upsampler.text2world_prompt_upsampler_inference import (
    create_prompt_upsampler,
    run_chat_completion,
)
from cosmos_predict1.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
)
from cosmos_predict1.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    run_chat_completion as run_chat_completion_vlm,
)
from cosmos_predict1.diffusion.training.utils.inference_long_video import generate_video_from_batch_with_loop
from cosmos_predict1.utils import log
from cosmos_predict1.utils.base_world_generation_pipeline import BaseWorldGenerationPipeline

MODEL_NAME_DICT = {
    # text2world
    "Cosmos-Predict1-7B-Text2World": "Cosmos_Predict1_Text2World_7B",
    "Cosmos-Predict1-14B-Text2World": "Cosmos_Predict1_Text2World_14B",
    "Cosmos-Predict1-7B-Text2World_post-trained": "Cosmos_Predict1_Text2World_7B_Post_trained",
    "Cosmos-Predict1-14B-Text2World_post-trained": "Cosmos_Predict1_Text2World_14B_Post_trained",
    # text2world low-memory
    "Cosmos-Predict1-7B-Text2World_post-trained-4gpu_80gb": "Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_80gb",
    "Cosmos-Predict1-7B-Text2World_post-trained-8gpu_40gb": "Cosmos_Predict1_Text2World_7B_Post_trained_8gpu_40gb",
    "Cosmos-Predict1-7B-Text2World_post-trained-4gpu_40gb": "Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_40gb",
    # text2world lora
    "Cosmos-Predict1-7B-Text2World_post-trained-lora": "Cosmos_Predict1_Text2World_7B_Post_trained_lora",
    # video2world
    "Cosmos-Predict1-7B-Video2World": "Cosmos_Predict1_Video2World_7B",
    "Cosmos-Predict1-14B-Video2World": "Cosmos_Predict1_Video2World_14B",
    "Cosmos-Predict1-7B-Video2World_post-trained": "Cosmos_Predict1_Video2World_7B_Post_trained",
    "Cosmos-Predict1-14B-Video2World_post-trained": "Cosmos_Predict1_Video2World_14B_Post_trained",
    # video2world low-memory
    "Cosmos-Predict1-7B-Video2World_post-trained-4gpu_80gb": "Cosmos_Predict1_Video2World_7B_Post_trained_4gpu_80gb",
    "Cosmos-Predict1-7B-Video2World_post-trained-8gpu_40gb": "Cosmos_Predict1_Video2World_7B_Post_trained_8gpu_40gb",
    "Cosmos-Predict1-7B-Video2World_post-trained-4gpu_40gb": "Cosmos_Predict1_Video2World_7B_Post_trained_4gpu_40gb",
    # video2world lora
    "Cosmos-Predict1-7B-Video2World_post-trained-lora": "Cosmos_Predict1_Video2World_7B_Post_trained_lora",
    "Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview": "Cosmos_Predict1_Text2World_7B_Multiview",
    "Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview": "Cosmos_Predict1_Video2World_7B_Multiview",
    "Cosmos-Predict1-7B-WorldInterpolator": "Cosmos_Predict1_WorldInterpolator_7B",
    # Gen3c
    "Gen3C-Cosmos-7B": "GEN3C_Cosmos_7B",
}


class DiffusionText2WorldGenerationPipeline(BaseWorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        disable_guardrail: bool = False,
        disable_prompt_encoder: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
    ):
        """Initialize the diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            disable_guardrail: Whether to disable guardrail
            disable_prompt_encoder: Whether to disable prompt encoder
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
        """
        assert inference_type in [
            "text2world",
            "video2world",
            "world_interpolator",
        ], "Invalid inference_type, must be 'text2world' or 'video2world'"

        self.model_name = MODEL_NAME_DICT[checkpoint_name]
        self.guidance = guidance
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.seed = seed

        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_guardrail_models=offload_guardrail_models,
            disable_guardrail=disable_guardrail,
            disable_prompt_encoder=disable_prompt_encoder,
        )
        self.prompt_upsampler_dir = prompt_upsampler_dir
        self.enable_prompt_upsampler = enable_prompt_upsampler
        self.offload_prompt_upsampler = offload_prompt_upsampler

        self.prompt_upsampler = None
        if enable_prompt_upsampler and not offload_prompt_upsampler:
            self._load_prompt_upsampler_model()

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/config.py",
            model_class=DiffusionT2WModel,
        )

    def _load_network(self):
        load_network_model(self.model, f"{self.checkpoint_dir}/{self.checkpoint_name}/model.pt")

    def _load_tokenizer(self):
        load_tokenizer_model(self.model, f"{self.checkpoint_dir}/Cosmos-Tokenize1-CV8x8x8-720p")

    def _offload_prompt_upsampler_model(self):
        """Move prompt enhancement model to CPU/disk.

        Offloads prompt upsampling model after processing input
        to reduce GPU memory usage.
        """
        if self.prompt_upsampler:
            del self.prompt_upsampler
            self.prompt_upsampler = None
            gc.collect()
            torch.cuda.empty_cache()

    def _run_prompt_upsampler_on_prompt(self, prompt: str) -> str:
        """Enhance the input prompt using the prompt upsampler model.

        Args:
            prompt: Raw text prompt to be enhanced

        Returns:
            str: Enhanced version of the input prompt with more descriptive details
        """
        upsampled_prompt = run_chat_completion(self.prompt_upsampler, prompt)
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _run_prompt_upsampler_on_prompt_with_offload(self, *args: Any, **kwargs: Any) -> str:
        """Enhance prompt with prompt upsampler model.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Enhanced prompt string
        """
        if self.offload_prompt_upsampler:
            self._load_prompt_upsampler_model()

        enhanced_prompt = self._run_prompt_upsampler_on_prompt(*args, **kwargs)

        if self.offload_prompt_upsampler:
            self._offload_prompt_upsampler_model()

        return enhanced_prompt

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return video

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate video latents using the diffusion model.

        Args:
            embedding: Text embedding tensor from text encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            torch.Tensor: Generated video latents before tokenizer decoding

        Note:
            The model and tokenizer are automatically offloaded after inference
            if offloading is enabled in the config.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        # Generate video frames
        sample = generate_world_from_text(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=True if negative_prompt_embedding is not None else False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
        )

        return sample

    def _run_model_with_offload(
        self, prompt_embedding: torch.Tensor, negative_prompt_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from text encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation
        """
        if self.offload_network:
            self._load_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        sample = self._run_model(prompt_embedding, negative_prompt_embedding)

        if self.offload_network:
            self._offload_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()
        return sample

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        word_limit_to_skip_upsampler: Optional[int] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            negative_prompt: Optional text to guide what not to generate
            word_limit_to_skip_upsampler: Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        if not self.disable_guardrail:
            log.info("Run guardrail on prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Input text prompt is not safe")
                return None
            log.info("Pass guardrail on prompt")

        # Enhance prompt
        if self.enable_prompt_upsampler:
            word_count = len(prompt.split())
            if word_limit_to_skip_upsampler is None or word_count <= word_limit_to_skip_upsampler:
                log.info("Run prompt upsampler on prompt")
                prompt = self._run_prompt_upsampler_on_prompt_with_offload(prompt)
                if not self.disable_guardrail:
                    log.info("Run guardrail on upsampled prompt")
                    is_safe = self._run_guardrail_on_prompt_with_offload(prompt=prompt)
                    if not is_safe:
                        log.critical("Upsampled text prompt is not safe")
                        return None
                    log.info("Pass guardrail on upsampled prompt")
            else:
                log.info(
                    f"Skip prompt upsampler for better robustness because the number of words ({word_count}) in the prompt is greater than {word_limit_to_skip_upsampler}"
                )

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
        )
        log.info("Finish generation")

        if not self.disable_guardrail:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                return None
            log.info("Pass guardrail on generated video")

        return video, prompt


class DiffusionVideo2WorldGenerationPipeline(DiffusionText2WorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        disable_guardrail: bool = False,
        disable_prompt_encoder: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            disable_guardrail: Whether to disable guardrail
            disable_prompt_encoder: Whether to disable prompt encoder
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
        """
        self.num_input_frames = num_input_frames
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=enable_prompt_upsampler,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            disable_guardrail=disable_guardrail,
            disable_prompt_encoder=disable_prompt_encoder,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
        )

    def _run_prompt_upsampler_on_prompt(self, image_or_video_path: str) -> str:
        """Enhance the input prompt using visual context from the conditioning image.

        Args:
            image_or_video_path: Path to conditioning image or video used for visual context

        Returns:
            str: Enhanced prompt incorporating visual details from the image
        """
        dialog = prepare_dialog(image_or_video_path)
        upsampled_prompt = run_chat_completion_vlm(
            self.prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
        )
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_vlm_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/config.py",
            model_class=DiffusionV2WModel,
        )

    def _run_model(
        self,
        embedding: torch.Tensor,
        condition_latent: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            condition_latent: Latent tensor from conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        # Generate video frames
        video = generate_world_from_video(
            model=self.model,
            state_shape=self.model.state_shape,
            is_negative_prompt=True,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
            condition_latent=condition_latent,
            num_input_frames=self.num_input_frames,
        )

        return video

    def _run_tokenizer_encoding(self, image_or_video_path: str) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            image_or_video_path: Path to conditioning image

        Returns:
            torch.Tensor: Latent tensor from tokenizer encoding
        """
        condition_latent = get_condition_latent(
            model=self.model,
            input_image_or_video_path=image_or_video_path,
            num_input_frames=self.num_input_frames,
            state_shape=self.model.state_shape,
        )

        return condition_latent

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            image_or_video_path: Path to conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        condition_latent = self._run_tokenizer_encoding(image_or_video_path)

        if self.offload_network:
            self._load_network()

        sample = self._run_model(prompt_embedding, condition_latent, negative_prompt_embedding)

        if self.offload_network:
            self._offload_network()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def generate(
        self,
        prompt: str,
        image_or_video_path: str,
        negative_prompt: Optional[str] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and optional image.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            image_or_video_path: Path to conditioning image or video
            negative_prompt: Optional text to guide what not to generate

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """

        log.info(f"Run with image or video path: {image_or_video_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        if self.enable_prompt_upsampler:
            log.info("Run prompt upsampler on image or video, input prompt is not used")
            prompt = self._run_prompt_upsampler_on_prompt_with_offload(image_or_video_path=image_or_video_path)

        log.info(f"Run with prompt: {prompt}")
        if not self.disable_guardrail:
            log.info(f"Run guardrail on {'upsampled' if self.enable_prompt_upsampler else 'text'} prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical(f"Input {'upsampled' if self.enable_prompt_upsampler else 'text'} prompt is not safe")
                return None
            log.info(f"Pass guardrail on {'upsampled' if self.enable_prompt_upsampler else 'text'} prompt")
        else:
            log.info("Not running guardrail")

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            image_or_video_path=image_or_video_path,
        )
        log.info("Finish generation")

        if not self.disable_guardrail:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                return None
            log.info("Pass guardrail on generated video")

        return video, prompt


class DiffusionText2WorldMultiviewGenerationPipeline(DiffusionText2WorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        disable_guardrail: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        n_views: int = 6,
        frame_repeat_negative_condition: int = 10,
        seed: int = 0,
    ):
        """Initialize the diffusion multi-view world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            disable_guardrail: Whether to disable guardrail
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            n_views: Number of views
            frame_repeat_negative_condition: Number of frames to repeat to be used as negative condition.
            seed: Random seed for sampling
        """
        assert inference_type in [
            "text2world",
            "video2world",
        ], "Invalid inference_type, must be 'text2world' or 'video2world'"

        self.n_views = n_views
        self.frame_repeat_negative_condition = frame_repeat_negative_condition
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=False,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            disable_guardrail=disable_guardrail,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/config.py",
            model_class=DiffusionMultiviewT2WModel,
        )

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video_segments = einops.rearrange(video, "b c (v t) h w -> b c v t h w", v=self.n_views)
        grid_video = torch.stack(
            [video_segments[:, :, i] for i in [1, 0, 2, 4, 3, 5]],
            dim=2,
        )
        grid_video = einops.rearrange(grid_video, "b c (h w) t h1 w1 -> b c t (h h1) (w w1)", h=2, w=3)
        grid_video = (grid_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return [grid_video, video]

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate video latents using the diffusion model.

        Args:
            embedding: Text embedding tensor from text encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            torch.Tensor: Generated video latents before tokenizer decoding

        Note:
            The model and tokenizer are automatically offloaded after inference
            if offloading is enabled in the config.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch_for_multiview_model(
            model=self.model,
            prompt_embedding=embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames * len(embedding),  # number of views
            frame_repeat_negative_condition=self.frame_repeat_negative_condition,
        )

        # Generate video frames
        sample = generate_world_from_text(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
        )

        return sample

    def generate(
        self,
        prompt: dict,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Convert prompt to embeddings
        2. Generate video frames using diffusion

        Args:
            prompt: A dictionary of text description of desired video.
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")

        prompts = [
            prompt["prompt"],
            prompt["prompt_left"],
            prompt["prompt_right"],
            prompt["prompt_back"],
            prompt["prompt_back_left"],
            prompt["prompt_back_right"],
        ]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        videos = self._run_model_with_offload(
            prompt_embeddings,
        )
        log.info("Finish generation")

        return videos, prompt


class DiffusionVideo2WorldMultiviewGenerationPipeline(DiffusionText2WorldMultiviewGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        disable_guardrail: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
        n_views: int = 6,
        frame_repeat_negative_condition: int = 10,
    ):
        """Initialize diffusion world multi-view generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            disable_guardrail: Whether to disable guardrail
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
        """
        self.num_input_frames = num_input_frames
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            disable_guardrail=disable_guardrail,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
            n_views=n_views,
            frame_repeat_negative_condition=frame_repeat_negative_condition,
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/config.py",
            model_class=DiffusionMultiviewV2WModel,
        )

    def _run_model(
        self,
        embedding: torch.Tensor,
        condition_latent: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
        data_batch: dict = None,
        state_shape: list = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            condition_latent: Latent tensor from conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Generate video frames
        video = generate_world_from_video(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
            condition_latent=condition_latent,
            num_input_frames=self.num_input_frames,
        )

        return video

    def _run_tokenizer_encoding(self, image_or_video_path: str, state_shape: list) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            image_or_video_path: Path to conditioning image

        Returns:
            torch.Tensor: Latent tensor from tokenizer encoding
        """
        condition_latent, condition_frames = get_condition_latent_multiview(
            model=self.model,
            input_image_or_video_path=image_or_video_path,
            num_input_frames=self.num_input_frames,
            state_shape=state_shape,
        )

        return condition_latent, condition_frames

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            image_or_video_path: Path to conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        data_batch, state_shape = get_video_batch_for_multiview_model(
            model=self.model,
            prompt_embedding=prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames * len(prompt_embedding),  # number of views
            frame_repeat_negative_condition=self.frame_repeat_negative_condition,
        )

        condition_latent, condition_frames = self._run_tokenizer_encoding(image_or_video_path, state_shape)

        if self.offload_network:
            self._load_network()

        sample = self._run_model(prompt_embedding, condition_latent, negative_prompt_embedding, data_batch, state_shape)

        if self.offload_network:
            self._offload_network()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def generate(
        self,
        prompt: dict,
        image_or_video_path: str,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Convert prompt to embeddings
        2. Generate video frames using diffusion

        Args:
            prompt: A dictionary of text description of desired video.
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")

        prompts = [
            prompt["prompt"],
            prompt["prompt_left"],
            prompt["prompt_right"],
            prompt["prompt_back"],
            prompt["prompt_back_left"],
            prompt["prompt_back_right"],
        ]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embeddings,
            image_or_video_path=image_or_video_path,
        )
        log.info("Finish generation")

        return video, prompt


class DiffusionWorldInterpolatorGenerationPipeline(DiffusionVideo2WorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        disable_guardrail: bool = False,
        guidance: float = -1.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 11,
        num_input_frames: int = 1,
        num_frame_pairs: int = 1,
        frame_index_start: int = 0,
        frame_stride: int = 1,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            disable_guardrail: Whether to disable guardrail
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
        """
        self.num_input_frames = num_input_frames
        self.num_frame_pairs = num_frame_pairs
        self.frame_index_start = frame_index_start
        self.frame_stride = frame_stride
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self.fps = fps

        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=enable_prompt_upsampler,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            disable_guardrail=disable_guardrail,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
            num_input_frames=num_input_frames,
        )

    def _run_prompt_upsampler_on_prompt(self, image_or_video_path: str) -> str:
        """Enhance the input prompt using visual context from the conditioning image.

        Args:
            image_or_video_path: Path to conditioning image or video used for visual context

        Returns:
            str: Enhanced prompt incorporating visual details from the image
        """
        dialog = prepare_dialog(image_or_video_path)
        upsampled_prompt = run_chat_completion_vlm(
            self.prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
        )
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_vlm_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/config.py",
            model_class=DiffusionWorldInterpolatorWModel,
        )

    @torch.inference_mode()
    def _run_model(
        self,
        condition_latent: torch.Tensor | None = None,
        negative_prompt_embedding: torch.Tensor | None = None,
        num_of_loops: int = 1,
        num_of_latent_overlap_list: list[int] = [1],
        augment_sigma_list: list[float] = [0.001],
        add_input_frames_guidance: float = 0,
        skip_reencode: int = 0,
        state_shape: list = None,
        raw_data_batch: dict = None,
    ) -> np.ndarray:
        """Generate video frames using the diffusion model, supporting chunk processing for video extension.

        Args:
            condition_latent: Latent tensor from conditioning image or video (optional for video extension).
            negative_prompt_embedding: Optional embedding for negative prompt guidance.
            num_of_loops: Number of loops for generating video segments.
            num_of_latent_overlap_list: List of overlaps for latent conditions in each loop.
            augment_sigma_list: List of sigma values for augmentation.
            add_input_frames_guidance: Guidance strength for input frames.
            skip_reencode: Whether to skip reencoding.
            frame_index_start: Starting index for frame pairs.
            num_frame_pairs: Number of frame pairs to process.
            frame_stride: Stride between frame pairs.
            is_interpolator_model: Whether the model is an interpolator.
            input_frames: Input video frames for interpolation (optional).

        Returns:
            np.ndarray: Generated video frames in shape (T, H, W, C).
        """
        video_np_THWC, _, _ = generate_video_from_batch_with_loop(
            model=self.model,
            data_batch=raw_data_batch,
            condition_latent=condition_latent,
            num_of_loops=num_of_loops,
            num_of_latent_overlap_list=num_of_latent_overlap_list,
            guidance=self.guidance,
            state_shape=state_shape,
            num_steps=self.num_steps,
            seed=self.seed,
            is_negative_prompt=True if negative_prompt_embedding is not None else False,
            visualize=False,
            save_fig_path=None,
            augment_sigma_list=augment_sigma_list,
            add_input_frames_guidance=add_input_frames_guidance,
            skip_reencode=skip_reencode,
        )

        return video_np_THWC

    def _run_tokenizer_encoding(
        self, image_or_video_path: str, frame_index: int = 0, frame_stride: int = 1
    ) -> torch.Tensor:
        """Encode image to latent space

        Args:
            image_or_video_path: Path to conditioning image
            frame_index: Starting frame index for encoding
            frame_stride: Stride between frames for encoding

        Returns:
            torch.Tensor: Latent tensor from tokenizer encoding
        """
        condition_latent = get_condition_latent(
            model=self.model,
            input_image_or_video_path=image_or_video_path,
            num_input_frames=self.num_input_frames,
            state_shape=self.model.state_shape,
            frame_index=frame_index,
            frame_stride=frame_stride,
        )

        return condition_latent

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
        frame_index_start: int = 0,
        num_frame_pairs: int = 1,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            image_or_video_path: Path to conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance
            frame_index_start: Starting index for frame pairs
            num_frame_pairs: Number of frame pairs to process

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        # Prepare video batch and state shape
        raw_data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        H, W = (
            state_shape[-2] * self.model.tokenizer.spatial_compression_factor,
            state_shape[-1] * self.model.tokenizer.spatial_compression_factor,
        )

        input_path_format = image_or_video_path.split(".")[-1]
        input_frames = read_video_or_image_into_frames_BCTHW(
            image_or_video_path,
            input_path_format=input_path_format,
            H=H,
            W=W,
        )

        num_frames = input_frames.shape[2]
        num_frame_pairs = num_frame_pairs or num_frames // self.frame_stride
        frame_stride = self.frame_stride

        video_output = []
        for frame_index in range(frame_index_start, num_frame_pairs):
            print(f"Processing frame pair {frame_index + 1} / {num_frame_pairs}...")

            condition_latent = self._run_tokenizer_encoding(image_or_video_path, frame_index, frame_stride)

            video_np_THWC = self._run_model(
                condition_latent=condition_latent,
                negative_prompt_embedding=negative_prompt_embedding,
                raw_data_batch=raw_data_batch,
                state_shape=state_shape,
            )

            # Convert to tensor, rearrange, and normalize to [0, 1]
            video_0_1 = einops.rearrange(torch.from_numpy(video_np_THWC), "t h w c -> c t h w") / 255.0

            # Handle overlap by skipping the first frame of subsequent segments
            if len(video_output) == 0:
                video_output.append(video_0_1)
            else:
                video_output.append(video_0_1[:, 1:, :, :])  # Skip first frame to avoid duplication

        # Concatenate all segments
        video_tensor = torch.cat(video_output, dim=1)  # Shape: (C, total_num_frames, H, W)

        # Convert to NumPy array for guardrail: [T, H, W, C], uint8, [0, 255]
        video_np = (video_tensor.permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()  # Shape: (T, H, W, C)

        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._offload_tokenizer()

        return video_np

    def generate(
        self,
        prompt: str,
        image_or_video_path: str,
        negative_prompt: Optional[str] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and optional image.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            image_or_video_path: Path to conditioning image or video
            negative_prompt: Optional text to guide what not to generate

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with image or video path: {image_or_video_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        if not self.disable_guardrail and not self.enable_prompt_upsampler:
            log.info("Run guardrail on prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Input text prompt is not safe")
                return None
            log.info("Pass guardrail on prompt")
        else:
            log.info("Run prompt upsampler on image or video, input prompt is not used")
            prompt = self._run_prompt_upsampler_on_prompt_with_offload(image_or_video_path=image_or_video_path)

            if not self.disable_guardrail:
                log.info("Run guardrail on upsampled prompt")
                is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
                if not is_safe:
                    log.critical("Upsampled text prompt is not safe")
                    return None
                log.info("Pass guardrail on upsampled prompt")

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            image_or_video_path=image_or_video_path,
            frame_index_start=self.frame_index_start,
            num_frame_pairs=self.num_frame_pairs,
        )
        log.info("Finish generation")

        if not self.disable_guardrail:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                return None
            log.info("Pass guardrail on generated video")

        return video, prompt
