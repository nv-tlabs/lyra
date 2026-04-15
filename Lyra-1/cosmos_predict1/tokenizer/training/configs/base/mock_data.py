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
from torch.utils.data import DataLoader

from cosmos_predict1.tokenizer.training.datasets.mock_dataset import CombinedDictDataset, LambdaDataset
from cosmos_predict1.tokenizer.training.datasets.utils import VIDEO_KEY, VIDEO_VAL_CROP_SIZE_INFO, get_crop_size_info
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

_IMAGE_ASPECT_RATIO = "1,1"
_VIDEO_ASPECT_RATIO = "16,9"


def get_video_dataset(
    is_train: bool,
    resolution: str,
    crop_height: int,
    num_video_frames: int,
):
    if is_train:
        crop_sizes = get_crop_size_info(crop_height)
        log.info(
            f"[video] training num_frames={num_video_frames}, crop_height={crop_height} and crop_sizes: {crop_sizes}."
        )
    else:
        if crop_height is None:
            crop_sizes = VIDEO_VAL_CROP_SIZE_INFO[resolution]
        else:
            crop_sizes = get_crop_size_info(crop_height)
        log.info(f"[video] validation num_frames={num_video_frames}, crop_sizes: {crop_sizes}")

    h = crop_sizes[_VIDEO_ASPECT_RATIO][1]
    w = crop_sizes[_VIDEO_ASPECT_RATIO][0]

    def video_fn():
        return 2 * torch.rand(3, num_video_frames, h, w) - 1

    return CombinedDictDataset(
        **{
            VIDEO_KEY: LambdaDataset(video_fn),
        }
    )


def get_mock_video_dataloader(
    batch_size: int, is_train: bool = True, num_video_frames: int = 9, resolution: str = "720", crop_height: int = 128
) -> LazyDict:
    """A function to get mock video dataloader.

    Args:
        batch_size: The batch size.
        num_video_frames: The number of video frames.
        resolution: The resolution. Defaults to "1024".

    Returns:
        LazyDict: A LazyDict object specifying the video dataloader.
    """
    if resolution not in VIDEO_VAL_CROP_SIZE_INFO:
        resolution = "720"
    return L(DataLoader)(
        dataset=L(get_video_dataset)(
            is_train=is_train,
            resolution=resolution,
            crop_height=crop_height,
            num_video_frames=num_video_frames,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
