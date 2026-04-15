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

import torch
from hydra.core.config_store import ConfigStore

from cosmos_predict1.autoregressive.configs.base.callbacks import BASIC_CALLBACKS, VIDEO_TEACHER_FORCING_CALLBACK
from cosmos_predict1.autoregressive.configs.base.dataloader import get_tealrobot_video
from cosmos_predict1.autoregressive.configs.base.optim import LambdaLinearLR
from cosmos_predict1.autoregressive.configs.experiment.video2video.basic import register_experiments
from cosmos_predict1.utils import config, log
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.scheduler import WarmupCosineLR


def register_checkpoint(cs):
    checkpoint_local = config.CheckpointConfig(
        save_iter=5000,
        broadcast_via_filesystem=True,
    )
    cs.store(group="checkpoint", package="checkpoint", name="local", node=checkpoint_local)


def register_callbacks(cs):
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="video_teacher_forcing",
        node=VIDEO_TEACHER_FORCING_CALLBACK,
    )


def register_scheduler(cs):
    cs.store(
        group="scheduler",
        package="scheduler",
        name="warmup_cosine_lr",
        node=L(WarmupCosineLR)(optimizer=None, warmup_iters=5000, lr_decay_iters="${trainer.max_iter}", min_lr=1e-8),
    )
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearLR)


def register_optimizer(cs):
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fused_adamw",
        node=L(torch.optim.AdamW)(params=None, lr=1e-3, weight_decay=0.05, fused=True),
    )
    cs.store(
        group="optimizer",
        package="optimizer",
        name="sgd",
        node=L(torch.optim.SGD)(params=None, lr=5e-6, momentum=0.9),
    )


def register_training_data(cs):
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="tealrobot_video_small",
        node=get_tealrobot_video(num_frames=33, video_size=[384, 640]),
    )
    cs.store(group="data_train", package="dataloader_train", name="tealrobot_video", node=get_tealrobot_video())


def register_configs():
    log.info("Registering configs for autoregressive_base")
    cs = ConfigStore.instance()
    register_callbacks(cs)
    register_checkpoint(cs)
    register_optimizer(cs)
    register_scheduler(cs)
    register_training_data(cs)
    register_experiments(cs)
