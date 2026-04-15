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
import shutil
import argparse
from omegaconf import OmegaConf

def copy_latest_checkpoint(source_dir, dest_dir, delete_ckpts=False, load_config=True):
    prefix = "checkpoint-"
    checkpoints = []
    if load_config:
        source_config = OmegaConf.load(source_dir)
        source_dir = source_config.output_dir
        dest_config = OmegaConf.load(dest_dir)
        dest_dir = dest_config.output_dir
    # Find all valid checkpoint directories
    for name in os.listdir(source_dir):
        full_path = os.path.join(source_dir, name)
        if os.path.isdir(full_path) and name.startswith(prefix):
            try:
                num = int(name[len(prefix):])
                checkpoints.append((num, name))
            except ValueError:
                continue

    if len(checkpoints) == 0:
        print("No valid checkpoint folders found.")
        return

    # Sort checkpoints by number (ascending)
    checkpoints.sort()

    print("🔍 Found checkpoint folders:")
    for num, name in checkpoints:
        print(f"  - {name}")

    # Copy the latest checkpoint
    max_num, latest_checkpoint = checkpoints[-1]
    src_path = os.path.join(source_dir, latest_checkpoint)
    dest_path = os.path.join(dest_dir, latest_checkpoint)
    if os.path.isfile(dest_path):
        print(f"Latest checkpoint already exists at destination: {dest_path}")
    else:
        print(f"Copying latest checkpoint: {latest_checkpoint}")
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        print("Copy completed.")

    if delete_ckpts:
        # Determine which checkpoints to delete (keep the two newest)
        to_delete = checkpoints[:-2] if len(checkpoints) > 2 else []
        if to_delete:
            print("\n⚠️ The following checkpoint folders will be deleted:")
            for num, name in to_delete:
                print(f"  - {name}")

            confirm = input("Do you want to delete these folders? (y/n): ").strip().lower()
            if confirm == 'y':
                for _, name in to_delete:
                    path_to_delete = os.path.join(source_dir, name)
                    shutil.rmtree(path_to_delete)
                    print(f"🗑️ Deleted: {name}")
                print("Old checkpoints deleted.")
            else:
                print("Deletion cancelled.")
        else:
            print("Nothing to delete — fewer than 3 checkpoints exist.")

def main():
    parser = argparse.ArgumentParser(description="Copy the latest checkpoint and optionally delete older ones.")
    parser.add_argument("source", help="Path to the source directory containing checkpoint folders.")
    parser.add_argument("destination", help="Path to the destination directory.")

    args = parser.parse_args()
    copy_latest_checkpoint(args.source, args.destination)

if __name__ == "__main__":
    main()