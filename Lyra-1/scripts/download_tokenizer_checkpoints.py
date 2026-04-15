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

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.download_guardrail_checkpoints import download_guardrail_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A script to download NVIDIA Cosmos-Tokenizer1 models from Hugging Face"
    )
    parser.add_argument(
        "--tokenizer_types",
        nargs="*",
        default=[
            "CV8x8x8-720p",
            "DV8x16x16-720p",
            "CI8x8-360p",
            "CI16x16-360p",
            "CV4x8x8-360p",
            "DI8x8-360p",
            "DI16x16-360p",
            "DV4x8x8-360p",
        ],  # Download all by default
        choices=[
            "CV8x8x8-720p",
            "DV8x16x16-720p",
            "CI8x8-360p",
            "CI16x16-360p",
            "CV4x8x8-360p",
            "DI8x8-360p",
            "DI16x16-360p",
            "DV4x8x8-360p",
        ],
        help="Which tokenizer model types to download. Possible values: CV8x8x8-720p, DV8x16x16-720p, CV4x8x8-360p, DV4x8x8-360p",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    args = parser.parse_args()
    return args


MD5_CHECKSUM_LOOKUP = {
    "Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit": "7f658580d5cf617ee1a1da85b1f51f0d",
    "Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit": "ff21a63ed817ffdbe4b6841111ec79a8",
    "Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit": "f5834d03645c379bc0f8ad14b9bc0299",
    "Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt": "f07680ad7eefae57d698778e2a0c7c96",
    "Cosmos-Tokenize1-CI16x16-360p/autoencoder.jit": "98f8fdf2ada5537705d6d1bc22c63cf1",
    "Cosmos-Tokenize1-CI16x16-360p/decoder.jit": "dd31a73a8c7062bab25492401d83b473",
    "Cosmos-Tokenize1-CI16x16-360p/encoder.jit": "7be1dadea5a1c283996ca1ce5b1a95a9",
    "Cosmos-Tokenize1-CI8x8-360p/autoencoder.jit": "b2ff9280b12a97202641bb2a41d7b271",
    "Cosmos-Tokenize1-CI8x8-360p/decoder.jit": "57fb213cd88c0a991e9d400875164571",
    "Cosmos-Tokenize1-CI8x8-360p/encoder.jit": "138fe257df41d7a43c17396c23086565",
    "Cosmos-Tokenize1-CV4x8x8-360p/autoencoder.jit": "0690ff725700128424d082b44a1eda08",
    "Cosmos-Tokenize1-CV4x8x8-360p/decoder.jit": "7573744ec14cb1b2abdf9c80318b7224",
    "Cosmos-Tokenize1-CV4x8x8-360p/encoder.jit": "fe3a7193defcb2db0b849b6df480b5e6",
    "Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit": "7f658580d5cf617ee1a1da85b1f51f0d",
    "Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit": "ff21a63ed817ffdbe4b6841111ec79a8",
    "Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit": "f5834d03645c379bc0f8ad14b9bc0299",
    "Cosmos-Tokenize1-DI16x16-360p/autoencoder.jit": "88195130b86c3434d3d4b0e0376def6b",
    "Cosmos-Tokenize1-DI16x16-360p/decoder.jit": "bf27a567388902acbd8abcc3a5afd8dd",
    "Cosmos-Tokenize1-DI16x16-360p/encoder.jit": "12bae3a56c79a7ca0beb774843ee8c58",
    "Cosmos-Tokenize1-DI8x8-360p/autoencoder.jit": "1d638e6034fcd43619bc1cdb343ebe56",
    "Cosmos-Tokenize1-DI8x8-360p/decoder.jit": "b9b5eccaa7ab9ffbccae3b05b3903311",
    "Cosmos-Tokenize1-DI8x8-360p/encoder.jit": "2bfa3c189aacdf9dc8faf17bcc30dd82",
    "Cosmos-Tokenize1-DV4x8x8-360p/autoencoder.jit": "ff8802dc4497be60dc24a8f692833eed",
    "Cosmos-Tokenize1-DV4x8x8-360p/decoder.jit": "f9a7d4bd24e4d2ee210cfd5f21550ce8",
    "Cosmos-Tokenize1-DV4x8x8-360p/encoder.jit": "7af30a0223b2984d9d27dd3054fcd7af",
    "Cosmos-Tokenize1-DV8x16x16-720p/autoencoder.jit": "606b8585b637f06057725cbb67036ae6",
    "Cosmos-Tokenize1-DV8x16x16-720p/decoder.jit": "f0c8a9d992614a43e7ce24ebfc901e26",
    "Cosmos-Tokenize1-DV8x16x16-720p/encoder.jit": "95186b0410346a3f0cf250b76daec452",
}


def get_md5_checksum(checkpoints_dir, model_name):
    print("---------------------")
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name):
            print(f"Verifying checkpoint {key}...")
            file_path = checkpoints_dir.joinpath(key)
            # File must exist
            if not Path(file_path).exists():
                print(f"Checkpoint {key} does not exist.")
                return False
            # File must match give MD5 checksum
            with open(file_path, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != value:
                print(f"MD5 checksum of checkpoint {key} does not match.")
                return False
    print(f"Model checkpoints for {model_name} exist with matched MD5 checksums.")
    return True


def main(args) -> None:
    ORG_NAME = "nvidia"

    # Mapping from size argument to Hugging Face repository name
    model_map = {
        "CV8x8x8-720p": "Cosmos-Tokenize1-CV8x8x8-720p",
        "DV8x16x16-720p": "Cosmos-Tokenize1-DV8x16x16-720p",
        "CI8x8-360p": "Cosmos-Tokenize1-CI8x8-360p",
        "CI16x16-360p": "Cosmos-Tokenize1-CI16x16-360p",
        "CV4x8x8-360p": "Cosmos-Tokenize1-CV4x8x8-360p",
        "DI8x8-360p": "Cosmos-Tokenize1-DI8x8-360p",
        "DI16x16-360p": "Cosmos-Tokenize1-DI16x16-360p",
        "DV4x8x8-360p": "Cosmos-Tokenize1-DV4x8x8-360p",
    }

    # Create local checkpoints folder
    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = dict(allow_patterns=["README.md", "model.pt", "mean_std.pt", "config.json", "*.jit"])

    # Download the requested Tokenizer models
    for tokenizer_type in args.tokenizer_types:
        model_name = model_map[tokenizer_type]
        repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)

        if not get_md5_checksum(checkpoints_dir, model_name):
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo_id} to {local_dir}...")
            snapshot_download(
                repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False, **download_kwargs
            )

    download_guardrail_checkpoints(args.checkpoint_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
