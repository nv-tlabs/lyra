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

from typing import Any, Optional

import torch

from cosmos_predict1.diffusion.inference.inference_utils import (
    generate_world_from_video,
    get_video_batch,
    load_model_by_config,
)
from cosmos_predict1.diffusion.model.model_gen3c import DiffusionGen3CModel
from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline
from cosmos_predict1.utils import log

class Gen3cPipeline(DiffusionVideo2WorldGenerationPipeline):
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
        """
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
            num_input_frames=1,
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/config.py",
            model_class=DiffusionGen3CModel,
        )

    def generate(
        self,
        prompt: str,
        image_path: str,
        rendered_warp_images: torch.Tensor,
        rendered_warp_masks: torch.Tensor,
        negative_prompt: Optional[str] = None,
        return_latents: bool = False,
    ) -> Any:
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
            image_  path: Path to conditioning image
            rendered_warp_images: Rendered warp images
            rendered_warp_masks: Rendered warp masks
            negative_prompt: Optional text to guide what not to generate

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        if type(image_path) == str:
            log.info(f"Run with image path: {image_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

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
        gen_dict = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            image_or_video_path=image_path,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
            return_latents=return_latents,
        )
        video = gen_dict["video"]
        log.info("Finish generation")

        if not self.disable_guardrail:
            log.info("Run guardrail on generated video")
            video = self._run_guardrail_on_video_with_offload(video)
            if video is None:
                log.critical("Generated video is not safe")
                return None
            log.info("Pass guardrail on generated video")
        if return_latents:
            return video, prompt, gen_dict["latents"]
        else:
            return video, prompt

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        rendered_warp_images: torch.Tensor,
        rendered_warp_masks: torch.Tensor,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Any:
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

        sample = self._run_model(prompt_embedding, condition_latent, rendered_warp_images, rendered_warp_masks, negative_prompt_embedding)

        if return_latents:
            latents = sample

        if self.offload_network:
            self._offload_network()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()

        out_dict = {"video": sample}
        if return_latents:
            out_dict["latents"] = latents
        return out_dict

    def _run_model(
        self,
        embedding: torch.Tensor,
        condition_latent: torch.Tensor,
        rendered_warp_images: torch.Tensor,
        rendered_warp_masks: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
    ) -> Any:
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )
        data_batch["condition_state"] = rendered_warp_images
        data_batch["condition_state_mask"] = rendered_warp_masks
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
