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

import torch
from safetensors.torch import load as safetensors_torch_load

from lyra_2._ext.imaginaire.utils import log
from lyra_2._ext.imaginaire.utils.easy_io import easy_io


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    byte_stream = easy_io.load(file_path, file_format="byte")
    state_dict = safetensors_torch_load(byte_stream)
    return state_dict


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = easy_io.load(file_path, file_format="pt", map_location="cpu", weights_only=False)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


# based on https://github.com/huggingface/diffusers/blob/b793debd9d09225582943a1e9cb4ccdab30f1b37/src/diffusers/loaders/lora_conversion_utils.py#L1817
# since our model is the same as non-diffusers Wan, we only need to change the lora keys:
# 1. add adapter_name to the key, 2. change lora keys
def _convert_non_diffusers_wan_lora_to_diffusers(state_dict, adapter_name="default"):
    converted_state_dict = {}
    if any("diffusion_model." in k for k in state_dict.keys()):
        original_state_dict = {k[len("diffusion_model.") :]: v for k, v in state_dict.items()}
    else:
        original_state_dict = state_dict

    block_numbers = {int(k.split(".")[1]) for k in original_state_dict if k.startswith("blocks.")}
    min_block = min(block_numbers)
    max_block = max(block_numbers)

    is_i2v_lora = any("k_img" in k for k in original_state_dict) and any("v_img" in k for k in original_state_dict)
    lora_down_key = "lora_A" if any("lora_A" in k for k in original_state_dict) else "lora_down"
    lora_up_key = "lora_B" if any("lora_B" in k for k in original_state_dict) else "lora_up"

    def get_alpha_scales(down_weight, key):
        rank = down_weight.shape[0]
        alpha = original_state_dict.pop(key + ".alpha").item()
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        return scale_down, scale_up

    diff_keys = [k for k in original_state_dict if k.endswith((".diff_b", ".diff"))]
    if diff_keys:
        for diff_k in diff_keys:
            param = original_state_dict[diff_k]
            # The magnitudes of the .diff-ending weights are very low (most are below 1e-4, some are upto 1e-3,
            # and 2 of them are about 1.6e-2 [the case with AccVideo lora]). The low magnitudes mostly correspond
            # to norm layers. Ignoring them is the best option at the moment until a better solution is found. It
            # is okay to ignore because they do not affect the model output in a significant manner.
            threshold = 1.6e-2
            absdiff = param.abs().max() - param.abs().min()
            all_zero = torch.all(param == 0).item()
            all_absdiff_lower_than_threshold = absdiff < threshold
            if all_zero or all_absdiff_lower_than_threshold:
                log.debug(
                    f"Removed {diff_k} key from the state dict as it's all zeros, or values lower than hardcoded threshold."
                )
                original_state_dict.pop(diff_k)

    # For the `diff_b` keys, we treat them as lora_bias.
    # https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig.lora_bias

    for i in range(min_block, max_block + 1):
        # Self-attention
        for o, c in zip(["q", "k", "v", "o"], ["q", "k", "v", "o"]):
            original_key = f"blocks.{i}.self_attn.{o}.{lora_down_key}.weight"
            converted_down_key = f"blocks.{i}.self_attn.{c}.lora_A.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_down_key] = original_state_dict.pop(original_key)

            original_key = f"blocks.{i}.self_attn.{o}.{lora_up_key}.weight"
            converted_up_key = f"blocks.{i}.self_attn.{c}.lora_B.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_up_key] = original_state_dict.pop(original_key)

            alpha_key = f"blocks.{i}.self_attn.{o}.alpha"
            if alpha_key in original_state_dict:
                down_weight = converted_state_dict[converted_down_key]
                up_weight = converted_state_dict[converted_up_key]
                scale_down, scale_up = get_alpha_scales(down_weight, f"blocks.{i}.self_attn.{o}")
                converted_state_dict[converted_down_key] = down_weight * scale_down
                converted_state_dict[converted_up_key] = up_weight * scale_up

            original_key = f"blocks.{i}.self_attn.{o}.diff_b"
            converted_key = f"blocks.{i}.self_attn.{c}.lora_B.{adapter_name}.bias"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

        # Cross-attention
        for o, c in zip(["q", "k", "v", "o"], ["q", "k", "v", "o"]):
            original_key = f"blocks.{i}.cross_attn.{o}.{lora_down_key}.weight"
            converted_down_key = f"blocks.{i}.cross_attn.{c}.lora_A.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_down_key] = original_state_dict.pop(original_key)

            original_key = f"blocks.{i}.cross_attn.{o}.{lora_up_key}.weight"
            converted_up_key = f"blocks.{i}.cross_attn.{c}.lora_B.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_up_key] = original_state_dict.pop(original_key)

            alpha_key = f"blocks.{i}.cross_attn.{o}.alpha"
            if alpha_key in original_state_dict:
                down_weight = converted_state_dict[converted_down_key]
                up_weight = converted_state_dict[converted_up_key]
                scale_down, scale_up = get_alpha_scales(down_weight, f"blocks.{i}.cross_attn.{o}")
                converted_state_dict[converted_down_key] = down_weight * scale_down
                converted_state_dict[converted_up_key] = up_weight * scale_up

            original_key = f"blocks.{i}.cross_attn.{o}.diff_b"
            converted_key = f"blocks.{i}.cross_attn.{c}.lora_B.{adapter_name}.bias"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

        if is_i2v_lora:
            for o, c in zip(["k_img", "v_img"], ["k_img", "v_img"]):
                original_key = f"blocks.{i}.cross_attn.{o}.{lora_down_key}.weight"
                converted_down_key = f"blocks.{i}.cross_attn.{c}.lora_A.{adapter_name}.weight"
                if original_key in original_state_dict:
                    converted_state_dict[converted_down_key] = original_state_dict.pop(original_key)

                original_key = f"blocks.{i}.cross_attn.{o}.{lora_up_key}.weight"
                converted_up_key = f"blocks.{i}.cross_attn.{c}.lora_B.{adapter_name}.weight"
                if original_key in original_state_dict:
                    converted_state_dict[converted_up_key] = original_state_dict.pop(original_key)

                alpha_key = f"blocks.{i}.cross_attn.{o}.alpha"
                if alpha_key in original_state_dict:
                    down_weight = converted_state_dict[converted_down_key]
                    up_weight = converted_state_dict[converted_up_key]
                    scale_down, scale_up = get_alpha_scales(down_weight, f"blocks.{i}.cross_attn.{o}")
                    converted_state_dict[converted_down_key] = down_weight * scale_down
                    converted_state_dict[converted_up_key] = up_weight * scale_up

                original_key = f"blocks.{i}.cross_attn.{o}.diff_b"
                converted_key = f"blocks.{i}.cross_attn.{c}.lora_B.{adapter_name}.bias"
                if original_key in original_state_dict:
                    converted_state_dict[converted_key] = original_state_dict.pop(original_key)

        # FFN
        for o, c in zip(["ffn.0", "ffn.2"], ["ffn.0", "ffn.2"]):
            original_key = f"blocks.{i}.{o}.{lora_down_key}.weight"
            converted_down_key = f"blocks.{i}.{c}.lora_A.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_down_key] = original_state_dict.pop(original_key)

            original_key = f"blocks.{i}.{o}.{lora_up_key}.weight"
            converted_up_key = f"blocks.{i}.{c}.lora_B.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_up_key] = original_state_dict.pop(original_key)

            alpha_key = f"blocks.{i}.{o}.alpha"
            if alpha_key in original_state_dict:
                down_weight = converted_state_dict[converted_down_key]
                up_weight = converted_state_dict[converted_up_key]
                scale_down, scale_up = get_alpha_scales(down_weight, f"blocks.{i}.{o}")
                converted_state_dict[converted_down_key] = down_weight * scale_down
                converted_state_dict[converted_up_key] = up_weight * scale_up

            original_key = f"blocks.{i}.{o}.diff_b"
            converted_key = f"blocks.{i}.{c}.lora_B.{adapter_name}.bias"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

    # Remaining.
    if original_state_dict:
        if any("time_projection" in k for k in original_state_dict):
            original_key = f"time_projection.1.{lora_down_key}.weight"
            converted_down_key = f"time_projection.1.lora_A.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_down_key] = original_state_dict.pop(original_key)

            original_key = f"time_projection.1.{lora_up_key}.weight"
            converted_up_key = f"time_projection.1.lora_B.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_up_key] = original_state_dict.pop(original_key)

            alpha_key = f"time_projection.1.alpha"
            if alpha_key in original_state_dict:
                down_weight = converted_state_dict[converted_down_key]
                up_weight = converted_state_dict[converted_up_key]
                scale_down, scale_up = get_alpha_scales(down_weight, f"time_projection.1")
                converted_state_dict[converted_down_key] = down_weight * scale_down
                converted_state_dict[converted_up_key] = up_weight * scale_up

            if "time_projection.1.diff_b" in original_state_dict:
                converted_state_dict[f"time_projection.1.lora_B.{adapter_name}.bias"] = original_state_dict.pop(
                    "time_projection.1.diff_b"
                )

        if any("head.head" in k for k in state_dict):
            converted_state_dict[f"head.head.lora_A.{adapter_name}.weight"] = original_state_dict.pop(
                f"head.head.{lora_down_key}.weight"
            )
            converted_state_dict[f"head.head.lora_B.{adapter_name}.weight"] = original_state_dict.pop(
                f"head.head.{lora_up_key}.weight"
            )
            if "head.head.diff_b" in original_state_dict:
                converted_state_dict[f"head.head.lora_B.{adapter_name}.bias"] = original_state_dict.pop(
                    "head.head.diff_b"
                )

            alpha_key = f"head.head.alpha"
            if alpha_key in original_state_dict:
                down_weight = converted_state_dict[f"head.head.lora_A.{adapter_name}.weight"]
                up_weight = converted_state_dict[f"head.head.lora_B.{adapter_name}.weight"]
                scale_down, scale_up = get_alpha_scales(down_weight, f"head.head")
                converted_state_dict[f"head.head.lora_A.{adapter_name}.weight"] = down_weight * scale_down
                converted_state_dict[f"head.head.lora_B.{adapter_name}.weight"] = up_weight * scale_up

        for text_time in ["text_embedding", "time_embedding"]:
            if any(text_time in k for k in original_state_dict):
                for b_n in [0, 2]:
                    diffusers_b_n = b_n
                    diffusers_name = text_time
                    if any(f"{text_time}.{b_n}" in k for k in original_state_dict):
                        converted_state_dict[f"{diffusers_name}.{diffusers_b_n}.lora_A.{adapter_name}.weight"] = (
                            original_state_dict.pop(f"{text_time}.{b_n}.{lora_down_key}.weight")
                        )
                        converted_state_dict[f"{diffusers_name}.{diffusers_b_n}.lora_B.{adapter_name}.weight"] = (
                            original_state_dict.pop(f"{text_time}.{b_n}.{lora_up_key}.weight")
                        )
                        alpha_key = f"{text_time}.{b_n}.alpha"
                        if alpha_key in original_state_dict:
                            down_weight = converted_state_dict[
                                f"{diffusers_name}.{diffusers_b_n}.lora_A.{adapter_name}.weight"
                            ]
                            up_weight = converted_state_dict[
                                f"{diffusers_name}.{diffusers_b_n}.lora_B.{adapter_name}.weight"
                            ]
                            scale_down, scale_up = get_alpha_scales(down_weight, f"{text_time}.{b_n}")
                            converted_state_dict[f"{diffusers_name}.{diffusers_b_n}.lora_A.{adapter_name}.weight"] = (
                                down_weight * scale_down
                            )
                            converted_state_dict[f"{diffusers_name}.{diffusers_b_n}.lora_B.{adapter_name}.weight"] = (
                                up_weight * scale_up
                            )

                    if f"{text_time}.{b_n}.diff_b" in original_state_dict:
                        converted_state_dict[f"{diffusers_name}.{diffusers_b_n}.lora_B.{adapter_name}.bias"] = (
                            original_state_dict.pop(f"{text_time}.{b_n}.diff_b")
                        )

        for img_ours, img_theirs in [
            ("img_emb.proj.1", "img_emb.proj.1"),
            ("img_emb.proj.3", "img_emb.proj.3"),
        ]:
            original_key = f"{img_theirs}.{lora_down_key}.weight"
            converted_key = f"{img_ours}.lora_A.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

            original_key = f"{img_theirs}.{lora_up_key}.weight"
            converted_key = f"{img_ours}.lora_B.{adapter_name}.weight"
            if original_key in original_state_dict:
                converted_state_dict[converted_key] = original_state_dict.pop(original_key)

            alpha_key = f"{img_ours}.alpha"
            if alpha_key in original_state_dict:
                down_weight = converted_state_dict[f"{img_ours}.lora_A.{adapter_name}.weight"]
                up_weight = converted_state_dict[f"{img_ours}.lora_B.{adapter_name}.weight"]
                scale_down, scale_up = get_alpha_scales(down_weight, f"{img_ours}")
                converted_state_dict[f"{img_ours}.lora_A.{adapter_name}.weight"] = down_weight * scale_down
                converted_state_dict[f"{img_ours}.lora_B.{adapter_name}.weight"] = up_weight * scale_up

    if len(original_state_dict) > 0:
        diff = all(".diff" in k for k in original_state_dict)
        if diff:
            diff_keys = {k for k in original_state_dict if k.endswith(".diff")}
            if not all("lora" not in k for k in diff_keys):
                raise ValueError
            log.info(
                "The remaining `state_dict` contains `diff` keys which we do not handle yet. If you see performance issues, please file an issue: "
                "https://github.com/huggingface/diffusers//issues/new"
            )
        else:
            raise ValueError(f"`state_dict` should be empty at this point but has {original_state_dict.keys()=}")

    return converted_state_dict


def _convert_musubi_wan_lora_to_non_diffusers_wan(state_dict):
    # https://github.com/kohya-ss/musubi-tuner
    converted_state_dict = {}
    original_state_dict = {k[len("lora_unet_") :]: v for k, v in state_dict.items()}

    num_blocks = len({k.split("blocks_")[1].split("_")[0] for k in original_state_dict})
    is_i2v_lora = any("k_img" in k for k in original_state_dict) and any("v_img" in k for k in original_state_dict)

    def get_alpha_scales(down_weight, key):
        rank = down_weight.shape[0]
        alpha = original_state_dict.pop(key + ".alpha").item()
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        return scale_down, scale_up

    for i in range(num_blocks):
        # Self-attention
        for o, c in zip(["q", "k", "v", "o"], ["q", "k", "v", "o"]):
            down_weight = original_state_dict.pop(f"blocks_{i}_self_attn_{o}.lora_down.weight")
            up_weight = original_state_dict.pop(f"blocks_{i}_self_attn_{o}.lora_up.weight")
            scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_self_attn_{o}")
            converted_state_dict[f"blocks.{i}.self_attn.{c}.lora_down.weight"] = down_weight * scale_down
            converted_state_dict[f"blocks.{i}.self_attn.{c}.lora_up.weight"] = up_weight * scale_up

        # Cross-attention
        for o, c in zip(["q", "k", "v", "o"], ["q", "k", "v", "o"]):
            down_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_down.weight")
            up_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_up.weight")
            scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_cross_attn_{o}")
            converted_state_dict[f"blocks.{i}.cross_attn.{c}.lora_down.weight"] = down_weight * scale_down
            converted_state_dict[f"blocks.{i}.cross_attn.{c}.lora_up.weight"] = up_weight * scale_up

        if is_i2v_lora:
            for o, c in zip(["k_img", "v_img"], ["k_img", "v_img"]):
                down_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_down.weight")
                up_weight = original_state_dict.pop(f"blocks_{i}_cross_attn_{o}.lora_up.weight")
                scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_cross_attn_{o}")
                converted_state_dict[f"blocks.{i}.cross_attn.{c}.lora_down.weight"] = down_weight * scale_down
                converted_state_dict[f"blocks.{i}.cross_attn.{c}.lora_up.weight"] = up_weight * scale_up

        # FFN
        for o, c in zip(["ffn_0", "ffn_2"], ["ffn.0", "ffn.2"]):
            down_weight = original_state_dict.pop(f"blocks_{i}_{o}.lora_down.weight")
            up_weight = original_state_dict.pop(f"blocks_{i}_{o}.lora_up.weight")
            scale_down, scale_up = get_alpha_scales(down_weight, f"blocks_{i}_{o}")
            converted_state_dict[f"blocks.{i}.{c}.lora_down.weight"] = down_weight * scale_down
            converted_state_dict[f"blocks.{i}.{c}.lora_up.weight"] = up_weight * scale_up

    if len(original_state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {original_state_dict.keys()=}")

    return converted_state_dict
