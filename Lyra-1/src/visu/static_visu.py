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
from typing import List, Union
import re

def natural_key(s):
    """Generate a key that treats digits as integers for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def natural_sort(strings):
    """Sort a list of strings in human-friendly order (e.g., 'rgb_9' < 'rgb_10')."""
    return sorted(strings, key=natural_key)

def create_teaser_video(
    mp4_paths_or_dir: Union[List[str], str],
    output_path: str,
    grid_rows: int = 2,
    grid_cols: int = 3,
    use_mirror_views: bool = False,
    use_mirror_wave: bool = False,
):
    # Determine list of main mp4 paths
    if isinstance(mp4_paths_or_dir, str):
        # Directory input: find all mp4s ignoring wave/view subvideos and grid_dataset
        all_mp4s = [
            os.path.join(mp4_paths_or_dir, f) for f in natural_sort(os.listdir(mp4_paths_or_dir))
            if f.endswith(".mp4") and
               "_wave_" not in f and "_view_idx_" not in f and
               'grid_dataset' not in f and f.startswith("rgb")
        ]
    else:
        all_mp4s = mp4_paths_or_dir

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    batch_size = grid_rows * grid_cols
    temp_files = []

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

        def get_view_idx(filename):
            try:
                parts = filename.split('_')
                vi_pos = parts.index("view_idx")
                return int(parts[vi_pos + 1])
            except Exception:
                return 99999
        views = natural_sort(views)
        views_paths = [os.path.join(dir_path, v) for v in views]
        return wave_path, views_paths

    for i in range(0, len(all_mp4s), batch_size):
        batch = all_mp4s[i:i+batch_size]
        if len(batch) < batch_size:
            print(f"Skipping last incomplete batch of size {len(batch)}")
            break

        input_files = []
        input_labels = []

        for main_vid_path in batch:
            wave_vid, views_vids = find_subvideos(main_vid_path)
            # Wave
            input_files.append(wave_vid)
            input_labels.append("wave")
            if use_mirror_wave:
                input_files.append(wave_vid)
                input_labels.append("wave_mirror")

            # Views
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
        for idx, label in enumerate(input_labels):
            if "mirror" in label:
                # Temporal mirroring: reverse filter
                filter_parts.append(f"[{idx}:v] setpts=PTS-STARTPTS,reverse [v{idx}];")
            else:
                filter_parts.append(f"[{idx}:v] setpts=PTS-STARTPTS [v{idx}];")

        cursor = 0
        samples_segments = []
        for main_vid_path in batch:
            wave_vid, views_vids = find_subvideos(main_vid_path)
            segs_for_sample = []

            segs_for_sample.append(f"[v{cursor}]")
            cursor += 1
            if use_mirror_wave:
                segs_for_sample.append(f"[v{cursor}]")
                cursor += 1

            for _ in views_vids:
                segs_for_sample.append(f"[v{cursor}]")
                cursor += 1
                if use_mirror_views:
                    segs_for_sample.append(f"[v{cursor}]")
                    cursor += 1

            samples_segments.append(segs_for_sample)

        concat_labels = []
        filter_concat_count = 0
        for segs in samples_segments:
            count = len(segs)
            if count == 1:
                concat_labels.append(segs[0])
            else:
                segs_str = "".join(segs)
                filter_parts.append(f"{segs_str} concat=n={count}:v=1:a=0 [ct{filter_concat_count}];")
                concat_labels.append(f"[ct{filter_concat_count}]")
                filter_concat_count += 1

        for r in range(grid_rows):
            row_labels = concat_labels[r * grid_cols:(r + 1) * grid_cols]
            row_str = "".join(row_labels)
            filter_parts.append(f"{row_str} hstack=inputs={grid_cols} [row{r}];")

        rows_str = "".join(f"[row{r}]" for r in range(grid_rows))
        if grid_rows == 1:
            filter_parts.append(f"{rows_str} copy [outv];")
        else:
            filter_parts.append(f"{rows_str} vstack=inputs={grid_rows} [outv];")

        filtergraph = "".join(filter_parts)

        temp_out = os.path.join(output_dir, f"temp_batch_{i}.mp4")

        cmd = ["ffmpeg", "-y"] + inputs_ffmpeg + [
            "-filter_complex", filtergraph,
            "-map", "[outv]",
            "-c:v", "libx264",
            temp_out,
        ]

        subprocess.run(cmd, check=True)
        temp_files.append(temp_out)

    concat_list_path = os.path.join(output_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for tf in temp_files:
            f.write(f"file '{os.path.abspath(tf)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy", output_path
    ], check=True)

    os.remove(concat_list_path)
    for tf in temp_files:
        if os.path.exists(tf):
            os.remove(tf)


if __name__ == '__main__':
    grid_rows = 2
    grid_cols = 4
    use_mirror_views = True
    use_mirror_wave = True
    mp4_input = "/path/to/static_view_indices_fixed_5_0_1_2_3_4"
    output_path = '/path/to/teaser.mp4'

    create_teaser_video(mp4_input, output_path, use_mirror_views=use_mirror_views, use_mirror_wave=use_mirror_wave, grid_rows=grid_rows, grid_cols=grid_cols)