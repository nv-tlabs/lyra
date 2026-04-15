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
import json
import os
import shutil
from glob import glob
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from scripts.download_guardrail_checkpoints import download_guardrail_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Download NVIDIA Cosmos Predict1 Gen3C models from Hugging Face")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    args = parser.parse_args()
    return args


def convert_pixtral_checkpoint(checkpoint_dir: str, checkpoint_name: str, vit_type: str):
    """
    Main function to convert Pixtral vision model weights to checkpoint and optionally verify and save the converted checkpoint.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory
        checkpoint_name (str): Name of the checkpoint
        vit_type (str): Type of ViT used in the Pixtral model

    This function performs the following steps:
    0. Download the checkpoint from Hugging Face
    1. Loads the original Pixtral checkpoint
    2. Splits the checkpoint into vision encoder, projector, and LLM weights
    3. Reorganizes the weights to match the expected format
    4. Extracts and verifies the vision encoder configuration
    5. Optionally verifies the converted checkpoint by loading it into a VisionTransformer
    6. Optionally saves the converted checkpoint and configuration
    """

    save_dir = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    # Save the converted checkpoint
    save_path = os.path.join(save_dir, "model.pt")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Checkpoint {save_path} already exists and is not empty")
        return

    pixtral_ckpt_dir = os.path.join(checkpoint_dir, "Pixtral-12B-2409")
    os.makedirs(pixtral_ckpt_dir, exist_ok=True)
    repo_id = "mistralai/Pixtral-12B-2409"
    print(f"Downloading {repo_id} to {pixtral_ckpt_dir}...")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["params.json", "consolidated.safetensors"],
        local_dir=pixtral_ckpt_dir,
        local_dir_use_symlinks=False,
    )
    orig_dtype = torch.get_default_dtype()
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)

    # Load checkpoint file
    ckpt_files = glob(os.path.join(pixtral_ckpt_dir, "*.safetensors"))
    assert len(ckpt_files) == 1, "ckpt_dir should contain only one file"
    ckpt_path = ckpt_files[0]
    ckpt = load_file(ckpt_path)

    # Split checkpoint into weights of vision encoder, projector, and LLM
    vit_key_prefix = "vision_encoder."
    vit_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith(vit_key_prefix):
            vit_ckpt[key.lstrip(vit_key_prefix)] = value

    projector_key_prefix = "vision_language_adapter."
    projector_ckpt = {}
    substring_replacement_map = {
        "w_in.": "projector.0.",
        "w_out.": "projector.2.",
    }
    for key, value in ckpt.items():
        if key.startswith(projector_key_prefix):
            key = key.lstrip(projector_key_prefix)
            for old, new in substring_replacement_map.items():
                key = key.replace(old, new)
            projector_ckpt[key] = value

    llm_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith(vit_key_prefix) or key.startswith(projector_key_prefix):
            continue
        llm_ckpt[key] = value

    vlm_ckpt = {}
    for key, value in llm_ckpt.items():
        vlm_ckpt["model." + key] = value
    for key, value in projector_ckpt.items():
        vlm_ckpt["mm_projector." + key] = value
    for key, value in vit_ckpt.items():
        vlm_ckpt["vision_encoder." + key] = value

    # Load config
    config_path = os.path.join(pixtral_ckpt_dir, "params.json")
    with open(config_path, "r") as f:
        pixtral_config = json.load(f)

    # Extract the vision encoder configuration
    vision_encoder_config = {
        "dim": pixtral_config["vision_encoder"]["hidden_size"],
        "num_channels": pixtral_config["vision_encoder"]["num_channels"],
        "image_size": pixtral_config["vision_encoder"]["image_size"],
        "patch_size": pixtral_config["vision_encoder"]["patch_size"],
        "rope_theta": pixtral_config["vision_encoder"]["rope_theta"],
        "ffn_hidden_size": pixtral_config["vision_encoder"]["intermediate_size"],
        "n_layers": pixtral_config["vision_encoder"]["num_hidden_layers"],
        "n_heads": pixtral_config["vision_encoder"]["num_attention_heads"],
        "n_kv_heads": pixtral_config["vision_encoder"]["num_attention_heads"],
        "norm_type": "rmsnorm",
        "norm_eps": pixtral_config["norm_eps"],
        "image_token_id": pixtral_config["vision_encoder"]["image_token_id"],
    }
    # Configuration for the 400M ViT of Pixtral 12B VLM
    vit_config = dict(
        dim=1024,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        rope_theta=10000,
        ffn_hidden_size=4096,
        n_layers=24,
        n_heads=16,
        n_kv_heads=16,
        norm_type="rmsnorm",
        norm_eps=1e-5,
        image_token_id=10,
    )
    # Compare the two configurations
    for key, value in vit_config.items():
        assert vision_encoder_config[key] == value, f"Mismatch in {key}: {vision_encoder_config[key]} != {value}"

    llm_config_keys = [
        "dim",
        "n_layers",
        "head_dim",
        "hidden_dim",
        "n_heads",
        "n_kv_heads",
        "rope_theta",
        "norm_eps",
        "vocab_size",
    ]
    assert set(list(pixtral_config.keys())) == set(llm_config_keys + ["vision_encoder"]), "Config keys mismatch"
    replace_map = {
        "hidden_dim": "ffn_hidden_size",
    }
    llm_config = {}
    for k, v in pixtral_config.items():
        if k in llm_config_keys:
            llm_config[replace_map.get(k, k)] = v
        elif k == "vision_encoder":
            llm_config["vision_encoder"] = vit_type
        else:
            raise ValueError(f"Unknown key: {k}")

    ckpt_to_save = {"model": vlm_ckpt, "mm_projector": projector_ckpt, "vision_encoder": vit_ckpt}
    torch.save(ckpt_to_save, save_path)
    print(f"Model saved to {save_path}")

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(llm_config, f)

    torch.set_default_dtype(orig_dtype)  # Reset the default dtype

    # Remove the original Pixtral checkpoint
    shutil.rmtree(pixtral_ckpt_dir, ignore_errors=True)
    print(f"Removed {pixtral_ckpt_dir}")


MD5_CHECKSUM_LOOKUP = {
    "Cosmos-Predict1-14B-Text2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-14B-Text2World/model.pt": "c69d1c6e51dc78b959040e8c4035a29b",
    "Cosmos-Predict1-14B-Video2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-14B-Video2World/model.pt": "eaa7aa3678f61d88108c41d7fe201b18",
    "Cosmos-Predict1-7B-WorldInterpolator/model.pt": "48a0bdc99d5e41eee05ba8597c4851da",
    "Cosmos-Predict1-7B-Text2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-7B-Text2World/model.pt": "fe9ed68e16cf37b10e7414c9b3ee81e1",
    "Cosmos-Predict1-7B-Video2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-7B-Video2World/model.pt": "ebcdb19c4c4a6a0e1e0bb65e346f6867",
    "Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt": "f07680ad7eefae57d698778e2a0c7c96",
    "Cosmos-Tokenize1-CV8x8x8-720p/image_mean_std.pt": "9f19fd3312fc1198e4905ada02e68bce",
    "Cosmos-UpsamplePrompt1-12B-Text2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-UpsamplePrompt1-12B-Text2World/model.pt": "52d7a6b8b1ac44d856b4c1ea3f8c8c74",
    "Cosmos-Predict1-7B-Text2World-Sample-AV-Multiview/model.pt": "e3a6ef070deaae0678acd529dc749ea4",
    "Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview/model.pt": "1653f87dce3d558ee01416593552a91c",
    "Gen3C-Cosmos-7B/model.pt": "38644bf823aa5272acef60cfad8bc0f7",
    "google-t5/t5-11b/pytorch_model.bin": "f890878d8a162e0045a25196e27089a3",
    "google-t5/t5-11b/tf_model.h5": "e081fc8bd5de5a6a9540568241ab8973",
}


def get_md5_checksum(checkpoints_dir, model_name):
    print("---------------------")
    # Check if there are any expected files for this model
    expected_files = [key for key in MD5_CHECKSUM_LOOKUP if key.startswith(model_name + "/")]
    if not expected_files:
        # No expected files in MD5_CHECKSUM_LOOKUP, check if the directory exists and has content
        model_dir = checkpoints_dir / model_name
        if not model_dir.exists() or not any(model_dir.iterdir()):
            print(f"Directory for {model_name} does not exist or is empty. Download required.")
            return False
        else:
            print(f"Directory for {model_name} exists and contains files. Assuming download is complete.")
            return True
    # Proceed with checksum verification for models with expected files
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name + "/"):
            print(f"Verifying checkpoint {key}...")
            file_path = checkpoints_dir.joinpath(key)
            # File must exist
            if not Path(file_path).exists():
                print(f"Checkpoint {key} does not exist.")
                return False
            # File must match given MD5 checksum
            with open(file_path, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != value:
                print(f"MD5 checksum of checkpoint {key} does not match.")
                return False
    print(f"Model checkpoints for {model_name} exist with matched MD5 checksums.")
    return True


def main(args):
    ORG_NAME = "nvidia"

    # Additional models that are always downloaded
    extra_models = [
        "Cosmos-Tokenize1-CV8x8x8-720p",
        "google-t5/t5-11b",
    ]

    # Create local checkpoints folder
    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = dict(
        allow_patterns=[
            "README.md",
            "model.pt",
            "mean_std.pt",
            "image_mean_std.pt",
            "config.json",
            "*.jit",
            "guardrail/*",
        ]
    )

    # Download the requested diffusion models
    model_name = "Gen3C-Cosmos-7B"
    repo_id = f"{ORG_NAME}/{model_name}"
    local_dir = checkpoints_dir.joinpath(model_name)

    if not get_md5_checksum(checkpoints_dir, model_name):
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False, **download_kwargs
        )

    # Download the always-included models
    for model_name in extra_models:
        if model_name == "google-t5/t5-11b":
            repo_id = model_name
        else:
            repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)

        if not get_md5_checksum(checkpoints_dir, model_name):
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo_id} to {local_dir}...")
            # Download all files for Guardrail
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
