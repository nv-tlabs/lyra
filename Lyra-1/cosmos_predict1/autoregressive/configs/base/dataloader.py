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

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.autoregressive.configs.base.dataset import VideoDatasetConfig
from cosmos_predict1.autoregressive.datasets.video_dataset import VideoDataset
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyCall as L

DATALOADER_OPTIONS = {}


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def dataloader_register(key):
    log.info(f"registering dataloader {key}...")

    def decorator(func):
        DATALOADER_OPTIONS[key] = func
        return func

    return decorator


@dataloader_register("tealrobot_video")
def get_tealrobot_video(
    batch_size: int = 1,
    dataset_dir: str = "datasets/cosmos_nemo_assets/videos/",
    sequence_interval: int = 1,
    num_frames: int = 33,
    video_size: list[int, int] = [640, 848],
    start_frame_interval: int = 1,
):
    dataset = L(VideoDataset)(
        config=VideoDatasetConfig(
            dataset_dir=dataset_dir,
            sequence_interval=sequence_interval,
            num_frames=num_frames,
            video_size=video_size,
            start_frame_interval=start_frame_interval,
        )
    )
    return L(DataLoader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
