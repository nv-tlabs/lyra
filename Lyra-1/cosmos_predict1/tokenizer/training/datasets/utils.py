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

"""Utilities for datasets creation."""

IMAGE_KEY = "images"
VIDEO_KEY = "video"
RECON_KEY = "reconstructions"
LATENT_KEY = "latent"
INPUT_KEY = "INPUT"
MASK_KEY = "loss_mask"

_SPATIAL_ALIGN = 16


import math
from typing import Union

import torch
from PIL import Image

# This is your "for short_side=720" map:
_ASPECT_SIZE_DICT = {
    "1,1": (720, 720),
    "4,3": (960, 720),
    "3,4": (720, 960),
    "16,9": (1280, 720),
    "9,16": (720, 1280),
}


VIDEO_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {  # 1080p doesn't have 1:1
        "4,3": (1440, 1072),
        "3,4": (1072, 1440),
        "16,9": (1920, 1072),
        "9,16": (1072, 1920),
    },
    "720": {"1,1": (720, 720), "4,3": (960, 720), "3,4": (720, 960), "16,9": (1280, 720), "9,16": (720, 1280)},
    "480": {"1,1": (480, 480), "4,3": (640, 480), "3,4": (480, 640), "16,9": (854, 480), "9,16": (480, 854)},
    "512": {"1,1": (512, 512), "4,3": (672, 512), "3,4": (512, 672), "16,9": (896, 512), "9,16": (512, 896)},
    "360": {"1,1": (320, 320), "4,3": (416, 320), "3,4": (320, 416), "16,9": (544, 320), "9,16": (320, 544)},
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "128": {  # Note that we set res lower than 256 to the same resolution as 256
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (448, 256),
        "9,16": (256, 448),
    },
}

VIDEO_VAL_CROP_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {  # 1080p doesn't have 1:1
        "4,3": (1424, 1072),
        "3,4": (1072, 1424),
        "16,9": (1904, 1072),
        "9,16": (1072, 1904),
        "16,10": (1715, 1072),
    },
    "720": {"1,1": (704, 704), "4,3": (944, 704), "3,4": (704, 944), "16,9": (1264, 704), "9,16": (704, 1264)},
    "480": {"1,1": (464, 464), "4,3": (624, 464), "3,4": (464, 624), "16,9": (848, 464), "9,16": (464, 848)},
    "360": {"1,1": (320, 320), "4,3": (416, 320), "3,4": (320, 416), "16,9": (544, 320), "9,16": (320, 544)},
    "512": {"1,1": (512, 512), "4,3": (672, 512), "3,4": (512, 672), "16,9": (896, 512), "9,16": (512, 896)},
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "128": {  # Note that we set res lower than 256 to the same resolution as 256
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
        "16,10": (410, 256),
    },
}


def _pick_closest_aspect_ratio(height, width):
    """
    Given a video's height and width, return the closest aspect ratio key
    from aspect_dict.
    """
    if height == 0:
        return "1,1"  # fallback if something weird, to avoid div by zero

    actual_ratio = width / height

    best_key = None
    min_diff = math.inf

    for ratio_key, (w_target, h_target) in _ASPECT_SIZE_DICT.items():
        # for "16,9" -> (1280, 720), ratio is 1280/720 = 1.7777...
        ratio = w_target / h_target
        diff = abs(actual_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_key = ratio_key

    return best_key


def categorize_aspect_and_store(data_sample):
    """
    data_sample: a dict with 'video' shaped [C,T,H,W].
    We will determine the aspect ratio, pick the closest "1,1", "4,3", etc.,
    and store a new dict entry.
    """
    # Suppose 'video' is [C, T, H, W].
    video_tensor = data_sample["video"]
    H = video_tensor.shape[-2]
    W = video_tensor.shape[-1]
    data_sample["aspect_ratio"] = _pick_closest_aspect_ratio(H, W)
    return data_sample


def get_crop_size_info(crop_sz: int = 128):
    aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]
    crop_sizes = dict()
    for aspect_ratio in aspect_ratios:
        if aspect_ratio[0] < aspect_ratio[1]:
            crop_h = crop_sz // _SPATIAL_ALIGN * _SPATIAL_ALIGN
            crop_w = int(crop_h * aspect_ratio[0] / aspect_ratio[1] + 0.5)
            crop_w = crop_w // _SPATIAL_ALIGN * _SPATIAL_ALIGN
        else:
            crop_w = crop_sz // _SPATIAL_ALIGN * _SPATIAL_ALIGN
            crop_h = int(crop_w * aspect_ratio[1] / aspect_ratio[0] + 0.5)
            crop_h = crop_h // _SPATIAL_ALIGN * _SPATIAL_ALIGN
        key = f"{aspect_ratio[0]},{aspect_ratio[1]}"
        crop_sizes.update({key: (crop_w, crop_h)})
    return crop_sizes


def obtain_image_size(data_dict: dict, input_keys: list) -> tuple[int, int]:
    r"""Function for obtaining the image size from the data dict.

    Args:
        data_dict (dict): Input data dict
        input_keys (list): List of input keys
    Returns:
        width (int): Width of the input image
        height (int): Height of the input image
    """

    data1 = data_dict[input_keys[0]]
    if isinstance(data1, Image.Image):
        width, height = data1.size
    elif isinstance(data1, torch.Tensor):
        height, width = data1.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


def obtain_augmentation_size(data_dict: dict, augmentor_cfg: dict) -> Union[int, tuple]:
    r"""Function for obtaining size of the augmentation.
    When dealing with multi-aspect ratio dataloaders, we need to
    find the augmentation size from the aspect ratio of the data.

    Args:
        data_dict (dict): Input data dict
        augmentor_cfg (dict): Augmentor config
    Returns:
        aug_size (int): Size of augmentation
    """
    if "__url__" in data_dict and "aspect_ratio" in data_dict["__url__"].meta.opts:
        aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]
        aug_size = augmentor_cfg["size"][aspect_ratio]
    else:  # Non-webdataset format
        aspect_ratio = data_dict["aspect_ratio"]
        aug_size = augmentor_cfg["size"][aspect_ratio]
    return aug_size
