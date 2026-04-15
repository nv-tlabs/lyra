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
import argparse
import subprocess
import math
import re

def natural_sort_key(s):
    """Sort strings containing numbers in human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_video_size(video_path):
    """Return (width, height) of video."""
    width = int(subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "default=nw=1:nk=1",
        video_path
    ], text=True).strip())

    height = int(subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height",
        "-of", "default=nw=1:nk=1",
        video_path
    ], text=True).strip())

    return width, height

def make_grids(input_files, grid_w, grid_h, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    videos_per_grid = grid_w * grid_h

    # Trim list so only complete grids are processed
    total_complete_videos = (len(input_files) // videos_per_grid) * videos_per_grid
    input_files = input_files[:total_complete_videos]

    total_grids = len(input_files) // videos_per_grid

    # Assume all videos have same resolution
    vid_w, vid_h = get_video_size(input_files[0])

    for grid_idx in range(total_grids):
        start_idx = grid_idx * videos_per_grid
        end_idx = start_idx + videos_per_grid
        chunk = input_files[start_idx:end_idx]

        inputs = []
        for vid in chunk:
            inputs.extend(["-i", vid])

        # Layout row by row
        layout_parts = []
        for row in range(grid_h):
            for col in range(grid_w):
                x = col * vid_w
                y = row * vid_h
                layout_parts.append(f"{x}_{y}")
        layout_str = "|".join(layout_parts)

        filter_cmd = f"xstack=inputs={videos_per_grid}:layout={layout_str}[outv]"

        output_path = os.path.join(output_dir, f"{grid_idx}.mp4")
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_cmd,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-crf", "20",       # Good quality and VSCode compatible
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"Creating grid {grid_idx} ({start_idx}..{end_idx-1}) → {output_path}")
        subprocess.run(cmd, check=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Create video grids without resizing and viewable in VSCode.")
    parser.add_argument("input_dir", help="Directory with videos to combine.")
    parser.add_argument("--grid_width", type=int, required=True, help="Number of videos per row.")
    parser.add_argument("--grid_height", type=int, required=True, help="Number of videos per column.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save grid videos.")
    return parser.parse_args()

def main():
    args = parse_args()
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".mp4")]
    files.sort(key=natural_sort_key)  # Natural numeric order
    files = [os.path.join(args.input_dir, f) for f in files]

    if not files:
        print("No .mp4 files found in input_dir")
        return

    make_grids(files, args.grid_width, args.grid_height, args.output_dir)

if __name__ == "__main__":
    main()