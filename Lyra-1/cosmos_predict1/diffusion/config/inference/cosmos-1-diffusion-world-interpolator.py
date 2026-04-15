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

from cosmos_predict1.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
from cosmos_predict1.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_Predict1_WorldInterpolator_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        model=dict(
            sde=L(EDMSDE)(
                p_mean=0.0,
                p_std=1.0,
                sigma_max=80,
                sigma_min=0.0002,
            ),
            input_image_key="images_1024",
            latent_shape=[
                16,
                4,
                88,
                160,
            ],
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=9,
                )
            ),
            vae=dict(  # Added VAE field
                pixel_chunk_duration=9,
                latent_ch=16,
            ),
            adjust_video_noise=True,
            num_latents_to_drop=1,
            context_parallel_size=1,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_and_last_1",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                    apply_corruption_to_condition_region_sigma_value=[0.001],
                ),
                text=dict(
                    dropout_rate=0.5,
                ),
            ),
            net=L(VideoExtendGeneralDIT)(
                extra_per_block_abs_pos_emb=True,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=2.0,
                extra_per_block_abs_pos_emb_type="learnable",
            ),
        ),
        job=dict(group="WorldInterpolator", name="Cosmos_Predict1_WorldInterpolator_7B"),
    )
)

Cosmos_Predict1_WorldInterpolator_7B_Post_trained: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/Cosmos_Predict1_WorldInterpolator_7B",
        ],
        job=dict(
            name="Cosmos_Predict1_WorldInterpolator_7B_Post_trained",
        ),
    )
)


cs = ConfigStore.instance()
for _item in [
    Cosmos_Predict1_WorldInterpolator_7B,
    Cosmos_Predict1_WorldInterpolator_7B_Post_trained,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
