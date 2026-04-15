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

"""Dataset config class."""

import attrs

from cosmos_predict1.utils.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class VideoDatasetConfig:
    """
    Args:
        dataset_dir (str): Base path to the dataset directory
        sequence_interval (int): Interval between sampled frames in a sequence
        num_frames (int): Number of frames to load per sequence
        video_size (list): Target size [H,W] for video frames
        start_frame_interval (int): Interval between starting frames of sequences
    """

    dataset_dir: str = "datasets/cosmos_nemo_assets/videos/"
    sequence_interval: int = 1
    num_frames: int = 33
    video_size: list[int, int] = [640, 848]
    start_frame_interval: int = 1
