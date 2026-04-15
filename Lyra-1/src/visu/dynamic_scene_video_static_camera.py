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

import os
import cv2
from glob import glob
import argparse

def collect_video_indices(main_path, sub_path, max_time_idx):
    """Find all unique {i} indices across timestamps."""
    video_indices = set()
    for time_idx in range(0, max_time_idx + 1):
        timestamp_folder = os.path.join(main_path, str(time_idx), sub_path)
        if os.path.isdir(timestamp_folder):
            for vid in glob(os.path.join(timestamp_folder, "rgb_*_view_idx_0_0.4.mp4")):
                fname = os.path.basename(vid)
                try:
                    idx = int(fname.split("_")[1])  # extract {i}
                    video_indices.add(idx)
                except ValueError:
                    pass
    return sorted(video_indices)

def collect_frames(main_path, sub_path, video_indices, num_frames, max_time_idx):
    """Collect the first `num_frames` frames from each video for each timestamp."""
    frames_dict = {i: [] for i in video_indices}
    for time_idx in range(0, max_time_idx + 1):
        timestamp_folder = os.path.join(main_path, str(time_idx), sub_path)
        if not os.path.isdir(timestamp_folder):
            print(f"Skip {timestamp_folder}")
            continue

        for i in video_indices:
            vid_path = os.path.join(timestamp_folder, f"rgb_{i}_view_idx_0_0.4.mp4")
            if not os.path.isfile(vid_path):
                print(f"Skip {vid_path}")
                continue

            cap = cv2.VideoCapture(vid_path)
            frame_count = 0
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_dict[i].append(frame)
                frame_count += 1
            cap.release()
    return frames_dict

def save_concatenated_videos(frames_dict, output_path, fps):
    """Save concatenated frames as videos using H264 codec if possible."""
    os.makedirs(output_path, exist_ok=True)
    for i, frames in frames_dict.items():
        if not frames:
            continue
        height, width, _ = frames[0].shape
        out_path = os.path.join(output_path, f"{i}.mp4")

        # Try H264 first
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # If writer not opened, fallback to mp4v
        if not writer.isOpened():
            print(f"Warning: 'avc1' codec not supported, falling back to 'mp4v' for video {i}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Concatenate frames from videos across timestamps.")
    parser.add_argument("--main_path", required=True, help="Path containing all time_idx folders")
    parser.add_argument("--sub_path", required=True, help="Path after the time_idx folder")
    parser.add_argument("--output_path", required=True, help="Folder to save output videos")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to extract from each video")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output videos")
    parser.add_argument("--max_time_idx", type=int, default=120, help="Maximum time index to loop over")

    args = parser.parse_args()

    print("Collecting video indices...")
    video_indices = collect_video_indices(args.main_path, args.sub_path, args.max_time_idx)

    print(f"Extracting first {args.num_frames} frames from each video...")
    frames_dict = collect_frames(args.main_path, args.sub_path, video_indices, args.num_frames, args.max_time_idx)

    print("Saving concatenated videos...")
    save_concatenated_videos(frames_dict, args.output_path, fps=args.fps)

    print(f"Done! Saved videos to: {args.output_path}")