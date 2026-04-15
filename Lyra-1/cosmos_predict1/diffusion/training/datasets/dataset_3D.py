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
import os
import pickle
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict1.diffusion.training.datasets.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)


class Dataset_3D(Dataset):
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

        This dataset loads robot trajectories consisting of RGB video frames, robot states (arm positions and gripper states),
        and computes relative actions between consecutive frames.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            cam_ids (list): List of camera IDs to sample from
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data. Defaults to False.
            normalize (bool, optional): Whether to normalize video frames. Defaults to False.
            pre_encode (bool, optional): Whether to pre-encode video frames. Defaults to False.
            do_evaluate (bool, optional): Whether in evaluation mode. Defaults to False.
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings. Defaults to False.
            load_action (bool, optional): Whether to load actions. Defaults to True.
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'. Defaults to 'train'.

        The dataset loads robot trajectories and computes:
        - RGB video frames from specified camera views
        - Robot arm states (xyz position + euler angles)
        - Gripper states (binary open/closed)
        - Relative actions between consecutive frames

        Actions are computed as relative transforms between frames:
        - Translation: xyz offset in previous frame's coordinate frame
        - Rotation: euler angles of relative rotation
        - Gripper: binary gripper state

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - action: Action tensor [T-1,7]
            - video_name: Dict with episode/frame metadata
            - latent: Pre-encoded video features if pre_encode=True
        """

        super().__init__()
        if mode == "train":
            self.data_path = train_annotation_path
            self.start_frame_interval = 1
        elif mode == "val":
            self.data_path = val_annotation_path
            self.start_frame_interval = val_start_frame_interval
        elif mode == "test":
            self.data_path = test_annotation_path
            self.start_frame_interval = val_start_frame_interval
        self.video_path = video_path
        self.sequence_interval = sequence_interval
        self.mode = mode
        self.sequence_length = num_frames
        self.normalize = normalize
        self.pre_encode = pre_encode
        self.load_t5_embeddings = load_t5_embeddings
        self.load_action = load_action

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action

        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.c_act_scaler = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.data_path)

        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),  # 288 512
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        return ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        with open(ann_file, "r") as f:
            ann = json.load(f)

        n_frames = len(ann["state"])
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["ann_file"] = ann_file
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            video_path = label["videos"][cam_id]["video_path"]
            video_path = os.path.join(self.video_path, video_path)
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                print(t_videos.shape)
                writer = imageio.get_writer(filename, fps=4)  # fps 是帧率
                for frame in t_videos:
                    writer.append_data(frame)  # 1 4 13 23 # fp16 24 76 456 688

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, temp_cam_id

    def _get_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        all_cont_gripper_states = np.array(label["continuous_gripper_state"])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        assert arm_states.shape[0] == self.sequence_length
        assert cont_gripper_states.shape[0] == self.sequence_length
        return arm_states, cont_gripper_states

    def _get_all_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        all_cont_gripper_states = np.array(label["continuous_gripper_state"])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        return arm_states, cont_gripper_states

    def _get_all_actions(self, arm_states, gripper_states, accumulate_action):
        action_num = arm_states.shape[0] - 1
        action = np.zeros((action_num, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, action_num + 1):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, action_num + 1):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

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

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file

            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
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
    dataset = Dataset_3D(
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
                "---"
            )
        )

    from IPython import embed

    embed()
