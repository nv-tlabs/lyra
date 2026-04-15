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
import tempfile
import shutil
from fractions import Fraction

def ffprobe_fps(path):
    """Return float FPS for the first video stream, preferring avg_frame_rate."""
    def probe(field):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", f"stream={field}",
            "-of", "default=nw=1:nk=1",
            path
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        return out

    val = probe("avg_frame_rate")
    if val in ("0/0", "", "N/A"):
        val = probe("r_frame_rate")
    try:
        return float(Fraction(val))
    except Exception:
        return 30.0  # fallback default FPS

def escape_concat_path(p):
    """Quote a path for ffmpeg concat list file."""
    return "'" + p.replace("'", r"'\''") + "'"

def subsample_one(input_path, stride, temp_path, target_fps):
    """
    Keep every Nth frame, normalize timing, and set CFR to target_fps.
    Audio is dropped to avoid desync; can be adapted to speed audio if needed.
    """
    vf = f"select='not(mod(n\\,{stride}))',setpts=N/FRAME_RATE/TB,fps=fps={target_fps}"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-an",  # remove audio for sync safety
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-movflags", "+faststart",
        temp_path
    ]
    subprocess.run(cmd, check=True)

def process_and_concat(input_dirs, strides, output_dir):
    if len(strides) != len(input_dirs):
        raise ValueError("Number of strides must match number of input directories.")

    os.makedirs(output_dir, exist_ok=True)

    # Get common MP4 filenames across all dirs
    def mp4s(d):
        return set(f for f in os.listdir(d)
                   if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(".mp4"))
    common_files = set.intersection(*(mp4s(d) for d in input_dirs))
    if not common_files:
        print("No common .mp4 filenames across the provided directories.")
        return

    temp_root = tempfile.mkdtemp(prefix="stride_concat_")

    try:
        for filename in sorted(common_files):
            # Get FPS from the first directory’s file
            ref_input = os.path.join(input_dirs[0], filename)
            fps = ffprobe_fps(ref_input)

            processed = []
            for i, (d, stride) in enumerate(zip(input_dirs, strides)):
                inp = os.path.join(d, filename)
                seg = os.path.join(
                    temp_root,
                    f"{i:02d}__{os.path.basename(os.path.normpath(d))}__{filename}"
                )
                print(f"[{filename}] {d} stride={stride} → {seg}")
                subsample_one(inp, stride, seg, fps)
                processed.append(seg)

            # Create concat list file
            list_path = os.path.join(temp_root, f"{filename}.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in processed:
                    f.write(f"file {escape_concat_path(os.path.abspath(p))}\n")

            # Final concat output
            out_path = os.path.join(output_dir, filename)
            print(f"[{filename}] concatenating {len(processed)} segments → {out_path}")
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                out_path
            ]
            subprocess.run(concat_cmd, check=True)

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporally concat matching MP4s from multiple directories, "
                    "after per-directory frame subsampling (stride)."
    )
    p.add_argument("input_dirs", nargs="+",
                   help="Input directories (order = concat order). Each must contain matching MP4 filenames.")
    p.add_argument("-s", "--strides", nargs="+", type=int, required=True,
                   help="Stride per directory (e.g. -s 2 3 5). Must match number of input dirs.")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Directory to write concatenated outputs (filenames preserved).")
    return p.parse_args()

def main():
    args = parse_args()
    process_and_concat(args.input_dirs, args.strides, args.output_dir)

if __name__ == "__main__":
    main()
