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
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

GEN3C_Cosmos_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            "_self_",
        ],
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            conditioner=dict(video_cond_bool=dict()),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=2.0,
                in_channels=16 + 16 * 4 + 1 # 16: video_latent, 16 * 4: (warped_frames + warped_frames_mask) * buffer 2, 1: mask
            ),
            frame_buffer_max=2,
        ),
        job=dict(group="Gen3c", name="GEN3C_Cosmos_7B"),
    )
)

cs = ConfigStore.instance()
for _item in [
    GEN3C_Cosmos_7B,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
