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

"""
    This file contains a basic configuration for video2video experiments.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.autoregressive.configs.base.model_config import create_video2world_model
from cosmos_predict1.autoregressive.configs.base.model_parallel import create_model_parallel_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


"""
   Finetune 4B model with TP=1, pytorch backend, low resolution tealrobot data, frames 33, chunk 33.
   Usage:
   torchrun --nproc_per_node=1 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=base_4b_example_tealrobotsmall_tp1
"""
base_4b_example_tealrobotsmall_tp1: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "tealrobot_video_small"},
            {
                "override /callbacks": [
                    "basic",
                    "video_teacher_forcing",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /optimizer": "fused_adamw"},
            {"override /scheduler": "warmup_cosine_lr"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="autoregressive_base",
            name="base_4b_example_tealrobotsmall_tp1",
        ),
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=1,
            batch_size=1,
            pixel_chunk_duration=33,
            num_video_frames=33,
            video_height=384,
            video_width=640,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
        ),
        trainer=dict(
            max_iter=50000,
            grad_accum_iter=1,
            grad_scaler_args=dict(enabled=False),
            run_validation=False,  # No need for validation as epoch <= 1
            distributed_parallelism="ddp",
            callbacks=dict(
                vid_sampling_tf=dict(
                    every_n=500,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Predict1-4B/model.pt",
            load_training_state=False,
            strict_resume=True,
            save_iter=1000,
        ),
        model_parallel=create_model_parallel_config(),
    ),
)


"""
   Finetune 4B model with TP=4, pytorch backend, high resolution tealrobot data, frame 33, chunk 33.
   Usage:
   torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=base_4b_example_tealrobot_tp4
"""
base_4b_example_tealrobot_tp4: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "tealrobot_video"},
            {
                "override /callbacks": [
                    "basic",
                    "video_teacher_forcing",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /optimizer": "fused_adamw"},
            {"override /scheduler": "warmup_cosine_lr"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="autoregressive_base",
            name="base_4b_example_tealrobot_tp4",
        ),
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=4,
            batch_size=1,
            pixel_chunk_duration=33,
            num_video_frames=33,
            video_height=640,
            video_width=848,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
        ),
        trainer=dict(
            max_iter=50000,
            grad_accum_iter=1,
            grad_scaler_args=dict(enabled=False),
            run_validation=False,  # No need for validation as epoch <= 1
            distributed_parallelism="ddp",
            callbacks=dict(
                vid_sampling_tf=dict(
                    every_n=500,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Predict1-4B/model.pt",
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
        ),
        model_parallel=create_model_parallel_config(),
    ),
)


def register_experiments(cs):
    # Register the experiments
    for _item in [
        base_4b_example_tealrobotsmall_tp1,
        base_4b_example_tealrobot_tp4,
    ]:
        cs.store(
            group="experiment",
            package="_global_",
            name=_item["job"]["name"],
            node=_item,
        )
