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

"""dataloader config options

Available dataloader options:
    image_loader_basic
    video_loader_basic
    joint_image_video_loader_basic
"""

from torch.utils.data import DataLoader

from cosmos_predict1.tokenizer.training.configs.base.mock_data import get_mock_video_dataloader
from cosmos_predict1.tokenizer.training.datasets.dataset_provider import dataset_entry
from cosmos_predict1.utils.lazy_config import LazyCall

DATALOADER_OPTIONS = {}


def dataloader_register(key):
    def decorator(func):
        DATALOADER_OPTIONS[key] = func
        return func

    return decorator


@dataloader_register("video_loader_basic")
def get_video_dataloader(
    dataset_name,
    is_train,
    batch_size=1,
    num_video_frames=25,
    resolution="720",
    crop_height=128,
    num_workers=8,
):
    if dataset_name.startswith("mock"):
        return get_mock_video_dataloader(
            batch_size=batch_size,
            is_train=is_train,
            num_video_frames=num_video_frames,
            resolution=resolution,
            crop_height=crop_height,
        )
    return LazyCall(DataLoader)(
        dataset=LazyCall(dataset_entry)(
            dataset_name=dataset_name,
            dataset_type="video",
            is_train=is_train,
            resolution=resolution,
            crop_height=crop_height,
            num_video_frames=num_video_frames,
        ),
        batch_size=batch_size,  # 2
        num_workers=num_workers,  # 8
        prefetch_factor=2,
        shuffle=None,  # do we need this?
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )
