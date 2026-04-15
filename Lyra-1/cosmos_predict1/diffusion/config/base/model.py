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

from typing import Any, Dict, List, Optional

import attrs

from cosmos_predict1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class DefaultModelConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    sigma_data: float = 0.5
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    latent_shape: List[int] = [16, 24, 44, 80]  # 24 corresponig to 136 frames
    input_image_key: str = "images_1024"
    adjust_video_noise: bool = False  # Added field with default value
    context_parallel_size: int = 1  # Added field with default value
    # `num_latents_to_drop` is a flag that helps satisfy (1I,N*P,1I) latents setup.
    # Since the tokenizer is causal and has the `T+1` input frames setup, it's
    # challenging to encode arbitrary number of frames. To circumvent this,
    # we sample as many frames, run the tokenizer twice, and discard the last
    # chunk's P-latents, ensuring the requirement: I-latents for the input frames
    # and P-latent for the-to-be-predicted in-between frames.
    # By default, this flag does not have any effect.
    num_latents_to_drop: int = 0  # number of P-latents to discard after encoding

    sde: Optional[Dict] = None
    vae: Optional[Dict] = None  # Add this line to include the vae field
    peft_control: LazyDict | None = None
    frame_buffer_max: Optional[int] = 1


@attrs.define(slots=False)
class LatentDiffusionDecoderModelConfig(DefaultModelConfig):
    tokenizer_corruptor: LazyDict = None
    latent_corruptor: LazyDict = None
    pixel_corruptor: LazyDict = None
    diffusion_decoder_cond_sigma_low: float = None
    diffusion_decoder_cond_sigma_high: float = None
    diffusion_decoder_corrupt_prob: float = None
    condition_on_tokenizer_corruptor_token: bool = False


@attrs.define(slots=False)
class MultiviewModelConfig(DefaultModelConfig):
    n_views: int = 4
