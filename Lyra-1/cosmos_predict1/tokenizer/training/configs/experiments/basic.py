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

"""Config settings for cosmos/tokenizer (basic image and video setting)"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils.lazy_config import LazyDict

CAUSAL_VIDEO_BASIC: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_factorized_video"},
            {"override /data_train": "mock_video720"},
            {"override /data_val": "mock_video720"},
            {"override /loss": "video"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            "_self_",
        ],
        model=dict(
            config=dict(
                loss=dict(
                    config=dict(
                        perceptual=dict(
                            config=dict(
                                lpips_boundaries=[0],
                                lpips_values=[0.1],
                                gram_enabled=False,
                                gram_boundaries=[0],
                            )
                        ),
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[1.0],
                                num_frames=32,
                                step=8,
                            )
                        ),
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[1_000_000],
                                values=[0.0, 0.01],
                                scale=2,
                                dtype="bfloat16",
                                checkpoint_activations=False,
                            )
                        ),
                    )
                )
            )
        ),
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=720,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="basic_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(load_path=None, jit=dict(input_shape=[1, 3, 17, 512, 512])),
    )
)

cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="video_basic", node=CAUSAL_VIDEO_BASIC)
