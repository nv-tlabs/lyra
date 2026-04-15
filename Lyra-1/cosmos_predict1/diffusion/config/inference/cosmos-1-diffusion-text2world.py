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

from hydra.core.config_store import ConfigStore

from cosmos_predict1.diffusion.training.utils.peft.lora_config import get_fa_ca_qv_lora_config
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_Predict1_Text2World_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "add_fps_image_size_padding_mask"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_Predict1_Text2World_7B",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            net=dict(
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=2.0,
            ),
        ),
    )
)

Cosmos_Predict1_Text2World_14B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_14b"},
            {"override /conditioner": "add_fps_image_size_padding_mask"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_Predict1_Text2World_14B",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            net=dict(
                rope_h_extrapolation_ratio=2.0,
                rope_t_extrapolation_ratio=2.0,
                rope_w_extrapolation_ratio=2.0,
                extra_h_extrapolation_ratio=2.0,
                extra_t_extrapolation_ratio=2.0,
                extra_w_extrapolation_ratio=2.0,
            ),
        ),
    )
)

Cosmos_Predict1_Text2World_7B_Post_trained: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B",
        ],
        job=dict(
            name="Cosmos_Predict1_Text2World_7B_Post_trained",
        ),
    )
)

Cosmos_Predict1_Text2World_14B_Post_trained: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_14B",
        ],
        job=dict(
            name="Cosmos_Predict1_Text2World_14B_Post_trained",
        ),
    )
)

Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_80gb: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B",
        ],
        job=dict(
            name="Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_80gb",
        ),
        model=dict(
            latent_shape=[  # 384x384 resolution
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            tokenizer=dict(
                video_vae=dict(pixel_chunk_duration=121, spatial_resolution="384"),
            ),
        ),
    )
)

Cosmos_Predict1_Text2World_7B_Post_trained_8gpu_40gb: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B",
        ],
        job=dict(
            name="Cosmos_Predict1_Text2World_7B_Post_trained_8gpu_40gb",
        ),
        model=dict(
            latent_shape=[  # 384x384 resolution
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            tokenizer=dict(
                video_vae=dict(pixel_chunk_duration=33, spatial_resolution="384"),
            ),
        ),
    )
)

Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_40gb: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B",
        ],
        job=dict(
            name="Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_40gb",
        ),
        model=dict(
            latent_shape=[  # 384x384 resolution
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            tokenizer=dict(
                video_vae=dict(pixel_chunk_duration=17, spatial_resolution="384"),
            ),
        ),
    )
)

Cosmos_Predict1_Text2World_7B_Post_trained_lora: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_Text2World_7B_Post_trained",
        ],
        job=dict(
            name="Cosmos_Predict1_Text2World_7B_Post_trained_lora",
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=27, rank=8, scale=1),
        ),
    )
)

cs = ConfigStore.instance()

for _item in [
    Cosmos_Predict1_Text2World_7B,
    Cosmos_Predict1_Text2World_14B,
    Cosmos_Predict1_Text2World_7B_Post_trained,
    Cosmos_Predict1_Text2World_14B_Post_trained,
    Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_80gb,
    Cosmos_Predict1_Text2World_7B_Post_trained_8gpu_40gb,
    Cosmos_Predict1_Text2World_7B_Post_trained_4gpu_40gb,
    Cosmos_Predict1_Text2World_7B_Post_trained_lora,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
