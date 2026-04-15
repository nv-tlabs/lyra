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
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict1/diffusion/posttrain/datasets/dataset_3D.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import json
import pickle
import random
import traceback
import warnings

import numpy as np
import torch

from cosmos_predict1.diffusion.training.datasets.dataset_3D import Dataset_3D
from cosmos_predict1.utils import log


class Dataset_3DBinary(Dataset_3D):
    def __init__(
        self,
        train_annotation_path,
        val_annotation_path,
        test_annotation_path,
        video_path,
        sequence_interval,
        num_frames,
        cam_ids,
        accumulate_action,
        video_size,
        val_start_frame_interval,
        debug=False,
        normalize=False,
        pre_encode=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        mode="train",
    ):
        """Dataset class for loading 3D robot action-conditional data.

        This dataset loads robot trajectories consisting of RGB video frames, robot states
        (arm positions and binary gripper states), and computes relative actions between
        consecutive frames.
        """

        super().__init__(
            train_annotation_path=train_annotation_path,
            val_annotation_path=val_annotation_path,
            test_annotation_path=test_annotation_path,
            video_path=video_path,
            sequence_interval=sequence_interval,
            num_frames=num_frames,
            cam_ids=cam_ids,
            accumulate_action=accumulate_action,
            video_size=video_size,
            val_start_frame_interval=val_start_frame_interval,
            debug=debug,
            normalize=normalize,
            pre_encode=pre_encode,
            do_evaluate=do_evaluate,
            load_t5_embeddings=load_t5_embeddings,
            load_action=load_action,
            mode=mode,
        )

        log.info("Dataset_3DBinary: in this dataset, we binarize the gripper state to 0 or 1.")

    def _get_json_action(self, label, frame_ids):
        all_action = np.array(label["action"])
        actions = all_action[frame_ids[:-1]]
        return torch.from_numpy(actions)

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]
            with open(ann_file, "r") as f:
                label = json.load(f)
            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions *= self.c_act_scaler

            data = dict()
            if self.load_action:
                data["action"] = actions.float()
                json_action = self._get_json_action(label, frame_ids).float()
                json_action[:, :6] = data["action"][:, :6]
                data["action"] = json_action

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file

            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                data["__key__"] = label["original_path"]

            # Just add these to fit the interface
            if self.load_t5_embeddings:
                t5_embedding_path = ann_file.replace(".json", ".pickle")
                with open(t5_embedding_path, "rb") as f:
                    data["t5_text_embeddings"] = torch.from_numpy(pickle.load(f)[0])
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4)  # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256)

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    dataset = Dataset_3DBinary(
        train_annotation_path="datasets/bridge/annotation/train",
        val_annotation_path="datasets/bridge/annotation/val",
        test_annotation_path="datasets/bridge/annotation/test",
        video_path="datasets/bridge/",
        sequence_interval=1,
        num_frames=2,
        cam_ids=[0],
        accumulate_action=False,
        video_size=[256, 320],
        val_start_frame_interval=1,
        mode="train",
        load_t5_embeddings=True,
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        print(
            (
                f"{idx=} "
                f"{dataset[idx]['video'].sum()=}\n"
                f"{dataset[idx]['video'].shape=}\n"
                f"{dataset[idx]['video_name']=}\n"
                f"{dataset[idx]['action'].sum()=}\n"
                f"{dataset[idx]['json_action'].sum()=}\n"
                "---"
            )
        )

    from IPython import embed

    embed()
