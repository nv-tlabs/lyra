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
import subprocess
from typing import List, Union, Optional
import re
import argparse

def natural_key(s):
    """Generate a key that treats digits as integers for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def natural_sort(strings):
    """Sort a list of strings in human-friendly order (e.g., 'rgb_9' < 'rgb_10')."""
    return sorted(strings, key=natural_key)

def create_teaser_video(
    mp4_paths_or_dir: Union[List[str], str],
    wave_output_dir: str,
    single_output_dir: str,
    grid_output_dir: Optional[str] = None,
    grid_rows: int = 2,
    grid_cols: int = 3,
    use_mirror_views: bool = False,
    use_mirror_wave: bool = False,
    fps: int = 30,
):
    if isinstance(mp4_paths_or_dir, str):
        all_mp4s = [
            os.path.join(mp4_paths_or_dir, f) for f in natural_sort(os.listdir(mp4_paths_or_dir))
            if f.endswith(".mp4") and
               "_wave_" not in f and "_view_idx_" not in f and
               'grid_dataset' not in f and f.startswith("rgb")
        ]
    else:
        all_mp4s = mp4_paths_or_dir

    os.makedirs(wave_output_dir, exist_ok=True)
    os.makedirs(single_output_dir, exist_ok=True)
    if grid_output_dir is not None:
        os.makedirs(grid_output_dir, exist_ok=True)

    def find_subvideos(main_path):
        base = os.path.basename(main_path)
        name_parts = base.split('_')
        sample_idx = None
        param = None
        if len(name_parts) >= 3:
            try:
                sample_idx = int(name_parts[-2])
                param = name_parts[-1].replace(".mp4", "")
            except Exception:
                pass
        if sample_idx is None or param is None:
            raise RuntimeError(f"Cannot parse sample idx and param from {base}")

        wave_name = f"rgb_wave_{sample_idx}_{param}.mp4"
        wave_path = os.path.join(os.path.dirname(main_path), wave_name)
        if not os.path.isfile(wave_path):
            raise FileNotFoundError(f"Wave video not found: {wave_path}")

        views = []
        dir_path = os.path.dirname(main_path)
        for f in os.listdir(dir_path):
            if f.startswith(f"rgb_{sample_idx}_view_idx_") and f.endswith(f"_{param}.mp4"):
                views.append(f)

        views = natural_sort(views)
        views_paths = [os.path.join(dir_path, v) for v in views]
        return sample_idx, wave_path, views_paths

    # Save wave videos separately (with optional mirror wave)
    for idx, main_vid_path in enumerate(all_mp4s):
        sample_idx, wave_vid, _ = find_subvideos(main_vid_path)

        input_files = [wave_vid]
        input_labels = ["wave"]
        if use_mirror_wave:
            input_files.append(wave_vid)
            input_labels.append("wave_mirror")

        inputs_ffmpeg = []
        for fpath in input_files:
            inputs_ffmpeg.extend(["-i", fpath])

        filter_parts = []
        for i, label in enumerate(input_labels):
            if "mirror" in label:
                filter_parts.append(f"[{i}:v] setpts=PTS-STARTPTS,reverse [v{i}];")
            else:
                filter_parts.append(f"[{i}:v] setpts=PTS-STARTPTS [v{i}];")

        segments = [f"[v{i}]" for i in range(len(input_labels))]
        seg_str = "".join(segments)
        filter_parts.append(f"{seg_str} concat=n={len(segments)}:v=1:a=0 [outv];")

        filtergraph = "".join(filter_parts)

        wave_output_path = os.path.join(wave_output_dir, f"{sample_idx}.mp4")
        cmd = ["ffmpeg", "-y"] + inputs_ffmpeg + [
            "-filter_complex", filtergraph,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-r", str(fps),
            wave_output_path,
        ]
        print(f"Creating wave video: {wave_output_path}")
        subprocess.run(cmd, check=True)

    # Create single videos from views only (with optional mirror views), no wave here
    for idx, main_vid_path in enumerate(all_mp4s):
        sample_idx, _, views_vids = find_subvideos(main_vid_path)

        if not views_vids:
            print(f"No views found for sample {sample_idx}, skipping single video creation.")
            continue

        input_files = []
        input_labels = []

        for v in views_vids:
            input_files.append(v)
            input_labels.append("view")
            if use_mirror_views:
                input_files.append(v)
                input_labels.append("view_mirror")

        inputs_ffmpeg = []
        for fpath in input_files:
            inputs_ffmpeg.extend(["-i", fpath])

        filter_parts = []
        for i, label in enumerate(input_labels):
            if "mirror" in label:
                filter_parts.append(f"[{i}:v] setpts=PTS-STARTPTS,reverse [v{i}];")
            else:
                filter_parts.append(f"[{i}:v] setpts=PTS-STARTPTS [v{i}];")

        segments = [f"[v{i}]" for i in range(len(input_labels))]
        seg_str = "".join(segments)
        filter_parts.append(f"{seg_str} concat=n={len(segments)}:v=1:a=0 [outv];")

        filtergraph = "".join(filter_parts)

        single_output_path = os.path.join(single_output_dir, f"{sample_idx}.mp4")
        cmd = ["ffmpeg", "-y"] + inputs_ffmpeg + [
            "-filter_complex", filtergraph,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-r", str(fps),
            single_output_path,
        ]

        print(f"Creating single video (views only): {single_output_path}")
        subprocess.run(cmd, check=True)

    if grid_output_dir is None:
        return

    # Create grids from single concatenated videos (views only)
    single_videos = natural_sort([f for f in os.listdir(single_output_dir) if f.endswith(".mp4")])
    single_videos_paths = [os.path.join(single_output_dir, f) for f in single_videos]

    batch_size = grid_rows * grid_cols

    for i in range(0, len(single_videos_paths), batch_size):
        batch = single_videos_paths[i:i+batch_size]
        if len(batch) < batch_size:
            print(f"Skipping last incomplete batch of size {len(batch)} for grid")
            break

        inputs_ffmpeg = []
        for f in batch:
            inputs_ffmpeg.extend(["-i", f])

        filter_parts = []
        for idx_in_batch in range(len(batch)):
            filter_parts.append(f"[{idx_in_batch}:v] setpts=PTS-STARTPTS [v{idx_in_batch}];")

        for r in range(grid_rows):
            row_labels = "".join(f"[v{r*grid_cols + c}]" for c in range(grid_cols))
            filter_parts.append(f"{row_labels} hstack=inputs={grid_cols} [row{r}];")

        rows_str = "".join(f"[row{r}]" for r in range(grid_rows))
        if grid_rows == 1:
            filter_parts.append(f"{rows_str} copy [outv];")
        else:
            filter_parts.append(f"{rows_str} vstack=inputs={grid_rows} [outv];")

        filtergraph = "".join(filter_parts)

        grid_output_path = os.path.join(grid_output_dir, f"grid_{i//batch_size}.mp4")

        cmd = ["ffmpeg", "-y"] + inputs_ffmpeg + [
            "-filter_complex", filtergraph,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-r", str(fps),
            grid_output_path,
        ]

        print(f"Creating grid video: {grid_output_path}")
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create teaser videos with separate wave and views-only outputs, and optional grids.')

    parser.add_argument('--mp4_input', type=str, required=True, help='Input folder path containing mp4 files')

    parser.add_argument('--wave_output_dir', type=str, required=True, help='Output folder path for wave only videos')
    parser.add_argument('--single_output_dir', type=str, required=True, help='Output folder path for views-only concatenated videos')
    parser.add_argument('--grid_output_dir', type=str, default=None, help='Output folder path for grid videos (optional)')

    parser.add_argument('--use_mirror_views', action='store_true', help='Enable mirror views')
    parser.add_argument('--use_mirror_wave', action='store_true', help='Enable mirror wave effect')

    parser.add_argument('--grid_rows', type=int, default=2, help='Number of grid rows')
    parser.add_argument('--grid_cols', type=int, default=3, help='Number of grid columns')

    parser.add_argument('--fps', type=int, default=30, help='Output frame rate (fps)')

    args = parser.parse_args()

    create_teaser_video(
        args.mp4_input,
        wave_output_dir=args.wave_output_dir,
        single_output_dir=args.single_output_dir,
        grid_output_dir=args.grid_output_dir,
        use_mirror_views=args.use_mirror_views,
        use_mirror_wave=args.use_mirror_wave,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        fps=args.fps,
    )
