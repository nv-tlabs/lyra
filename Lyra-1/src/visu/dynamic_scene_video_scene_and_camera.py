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
import argparse

def create_video_from_frames_opencv(frames, output_video_path, fps):
    if len(frames) == 0:
        raise ValueError("No frames to write to video")

    height, width = frames[0].shape[:2]

    if not output_video_path.lower().endswith(".mp4"):
        output_video_path += ".mp4"

    # Try avc1 fourcc for H264
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("avc1 codec failed, trying H264...")
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise RuntimeError("Failed to open video writer with H264 codec. Check your OpenCV/ffmpeg installation.")

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def extract_frames_from_videos(base_path, sub_path_after_time_index, output_dir, max_time_idx, fps):
    os.makedirs(output_dir, exist_ok=True)

    # Find all scene indices by inspecting time_idx=0 folder for files like rgb_{scene_idx}_view_idx_0_0.4.mp4
    time0_path = os.path.join(base_path, "0", sub_path_after_time_index)
    if not os.path.isdir(time0_path):
        raise FileNotFoundError(f"Time index 0 path does not exist: {time0_path}")

    scene_files = [f for f in os.listdir(time0_path) if f.startswith("rgb_") and f.endswith(".mp4")]
    
    # Extract scene_idx from filenames like rgb_{scene_idx}_view_idx_0_0.4.mp4
    scene_indices = []
    for f in scene_files:
        parts = f.split("_")
        if len(parts) >= 2 and parts[0] == "rgb":
            try:
                scene_idx = int(parts[1])
                scene_indices.append(scene_idx)
            except Exception:
                continue
    scene_indices = sorted(set(scene_indices))

    print(f"Found scene indices: {scene_indices}")

    # Collect existing time indices (folders) in base_path
    existing_time_indices = []
    for ti in range(max_time_idx + 1):
        folder = os.path.join(base_path, str(ti), sub_path_after_time_index)
        if os.path.isdir(folder):
            existing_time_indices.append(ti)
    print(f"Using existing time indices: {existing_time_indices}")

    for scene_idx in scene_indices:
        frames = []
        for idx, time_idx in enumerate(existing_time_indices):
            video_path = os.path.join(base_path, str(time_idx), sub_path_after_time_index,
                                      f"rgb_{scene_idx}_view_idx_0_0.4.mp4")

            if not os.path.isfile(video_path):
                print(f"Missing video for scene {scene_idx}, time_idx {time_idx}: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"No frames found in video {video_path}")
                cap.release()
                continue

            frame_idx = idx % total_frames  # wrap frame idx if needed
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Failed to read frame {frame_idx} from {video_path}")
                cap.release()
                continue

            frames.append(frame)
            cap.release()

        if len(frames) == 0:
            print(f"No frames extracted for scene {scene_idx}, skipping video creation.")
            continue

        output_video_path = os.path.join(output_dir, f"{scene_idx}.mp4")
        print(f"Writing video for scene {scene_idx} with {len(frames)} frames to {output_video_path}")

        create_video_from_frames_opencv(frames, output_video_path, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames across time indices and create concatenated videos per scene.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path containing time index folders")
    parser.add_argument("--sub_path_after_time_index", type=str, required=True, help="Sub path inside each time index folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated videos")
    parser.add_argument("--max_time_idx", type=int, default=120, help="Maximum time index to consider")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second for output video")

    args = parser.parse_args()

    extract_frames_from_videos(
        args.base_path,
        args.sub_path_after_time_index,
        args.output_dir,
        args.max_time_idx,
        args.fps
    )
