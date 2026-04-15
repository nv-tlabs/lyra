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

"""registry for commandline override options for config."""
from hydra.core.config_store import ConfigStore

from cosmos_predict1.tokenizer.training.configs.base.callback import BASIC_CALLBACKS
from cosmos_predict1.tokenizer.training.configs.base.checkpoint import CHECKPOINT_LOCAL
from cosmos_predict1.tokenizer.training.configs.base.data import DATALOADER_OPTIONS
from cosmos_predict1.tokenizer.training.configs.base.loss import VideoLossConfig
from cosmos_predict1.tokenizer.training.configs.base.metric import DiscreteTokenizerMetricConfig, MetricConfig
from cosmos_predict1.tokenizer.training.configs.base.net import (
    CausalContinuousFactorizedVideoTokenizerConfig,
    CausalDiscreteFactorizedVideoTokenizerConfig,
    ContinuousImageTokenizerConfig,
    DiscreteImageTokenizerConfig,
)
from cosmos_predict1.tokenizer.training.configs.base.optim import (
    AdamWConfig,
    FusedAdamConfig,
    WarmupCosineLRConfig,
    WarmupLRConfig,
)


def register_training_data(cs):
    for data_source in ["mock", "hdvila"]:
        for resolution in ["1080", "720", "480", "360", "256"]:
            cs.store(
                group="data_train",
                package="dataloader_train",
                name=f"{data_source}_video{resolution}",  # `davis_video720`
                node=DATALOADER_OPTIONS["video_loader_basic"](
                    dataset_name=f"{data_source}_video",
                    is_train=True,
                    resolution=resolution,
                ),
            )


def register_val_data(cs):
    for data_source in ["mock", "hdvila"]:
        for resolution in ["1080", "720", "480", "360", "256"]:
            cs.store(
                group="data_val",
                package="dataloader_val",
                name=f"{data_source}_video{resolution}",  # `davis_video720`
                node=DATALOADER_OPTIONS["video_loader_basic"](
                    dataset_name=f"{data_source}_video",
                    is_train=False,
                    resolution=resolution,
                ),
            )


def register_net(cs):
    cs.store(
        group="network", package="model.config.network", name="continuous_image", node=ContinuousImageTokenizerConfig
    )
    cs.store(group="network", package="model.config.network", name="discrete_image", node=DiscreteImageTokenizerConfig)

    cs.store(
        group="network",
        package="model.config.network",
        name="continuous_factorized_video",
        node=CausalContinuousFactorizedVideoTokenizerConfig,
    )
    cs.store(
        group="network",
        package="model.config.network",
        name="discrete_factorized_video",
        node=CausalDiscreteFactorizedVideoTokenizerConfig,
    )


def register_optim(cs):
    cs.store(group="optimizer", package="optimizer", name="fused_adam", node=FusedAdamConfig)
    cs.store(group="optimizer", package="optimizer", name="adamw", node=AdamWConfig)


def register_scheduler(cs):
    cs.store(group="scheduler", package="scheduler", name="warmup", node=WarmupLRConfig)
    cs.store(
        group="scheduler",
        package="scheduler",
        name="warmup_cosine",
        node=WarmupCosineLRConfig,
    )


def register_loss(cs):
    cs.store(group="loss", package="model.config.loss", name="video", node=VideoLossConfig)


def register_metric(cs):
    cs.store(group="metric", package="model.config.metric", name="reconstruction", node=MetricConfig)
    cs.store(group="metric", package="model.config.metric", name="code_usage", node=DiscreteTokenizerMetricConfig)


def register_checkpoint(cs):
    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)


def register_callback(cs):
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)


def register_configs():
    cs = ConfigStore.instance()

    register_training_data(cs)
    register_val_data(cs)

    register_net(cs)

    register_optim(cs)
    register_scheduler(cs)
    register_loss(cs)
    register_metric(cs)
    register_checkpoint(cs)

    register_callback(cs)
