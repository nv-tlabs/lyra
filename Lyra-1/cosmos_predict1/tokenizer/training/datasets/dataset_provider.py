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

"""Implementations of dataset settings and augmentations for tokenization

Run this command to interactively debug:
python3 -m cosmos_predict1.tokenizer.training.datasets.dataset_provider

"""

from cosmos_predict1.tokenizer.training.datasets.augmentation_provider import (
    video_train_augmentations,
    video_val_augmentations,
)
from cosmos_predict1.tokenizer.training.datasets.utils import categorize_aspect_and_store
from cosmos_predict1.tokenizer.training.datasets.video_dataset import Dataset
from cosmos_predict1.utils.lazy_config import instantiate

_VIDEO_PATTERN_DICT = {
    "hdvila_video": "datasets/hdvila/videos/*.mp4",
}


def apply_augmentations(data_dict, augmentations_dict):
    """
    Loop over each LazyCall object and apply it to data_dict in place.
    """
    for aug_name, lazy_aug in augmentations_dict.items():
        aug_instance = instantiate(lazy_aug)
        data_dict = aug_instance(data_dict)
    return data_dict


class AugmentDataset(Dataset):
    def __init__(self, base_dataset, augmentations_dict):
        """
        base_dataset: the video dataset instance
        augmentations_dict: the dictionary returned by
                            video_train_augmentations() or video_val_augmentations()
        """
        self.base_dataset = base_dataset

        # Pre-instantiate every augmentation ONCE:
        self.augmentations = []
        for aug_name, lazy_aug in augmentations_dict.items():
            aug_instance = instantiate(lazy_aug)  # build the actual augmentation
            self.augmentations.append((aug_name, aug_instance))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get the raw sample from the base dataset
        data = self.base_dataset[index]
        data = categorize_aspect_and_store(data)

        # Apply each pre-instantiated augmentation
        for aug_name, aug_instance in self.augmentations:
            data = aug_instance(data)

        return data


def dataset_entry(
    dataset_name: str,
    dataset_type: str,
    is_train: bool = True,
    resolution="720",
    crop_height=256,
    num_video_frames=25,
) -> AugmentDataset:
    if dataset_type != "video":
        raise ValueError(f"Dataset type {dataset_type} is not supported")

    # Instantiate the video dataset
    base_dataset = Dataset(
        video_pattern=_VIDEO_PATTERN_DICT[dataset_name.lower()],
        num_video_frames=num_video_frames,
    )

    # Pick the training or validation augmentations
    if is_train:
        aug_dict = video_train_augmentations(
            input_keys=["video"],  # adjust if necessary
            resolution=resolution,
            crop_height=crop_height,
        )
    else:
        aug_dict = video_val_augmentations(
            input_keys=["video"],
            resolution=resolution,
            crop_height=crop_height,
        )

    # Wrap the dataset with the augmentations
    return AugmentDataset(base_dataset, aug_dict)


if __name__ == "__main__":
    # Example usage / quick test
    dataset = dataset_entry(
        dataset_name="davis_video",
        dataset_type="video",
        is_train=False,
        resolution="720",
        crop_height=256,
        num_video_frames=25,
    )

    # 2) Print out some basic info:
    print(f"Total samples in dataset: {len(dataset)}")

    # 3) Grab one sample (or a few) to check shapes, keys, etc.
    if len(dataset) > 0:
        sample_idx = 0
        sample = dataset[sample_idx]
        print(f"Sample index {sample_idx} keys: {list(sample.keys())}")
        if "video" in sample:
            print("Video shape:", sample["video"].shape)
        if "video_name" in sample:
            print("Video metadata:", sample["video_name"])
        print("---\nSample loaded successfully.\n")
    else:
        print("Dataset has no samples!")
