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

from cosmos_predict1.diffusion.training.utils.peft.lora_attn import build_attn_lora
from cosmos_predict1.diffusion.utils.customization.customization_manager import CustomizationType
from cosmos_predict1.utils import log
from cosmos_predict1.utils.misc import count_params


def get_all_lora_params(model):
    """
    Get all LoRA weight parameters in the model
    """
    lora_modules = [mod for name, mod in model.named_modules() if "lora.net.0" in name or "lora.net.1" in name]
    lora_params = [(name, param) for mod in lora_modules for name, param in mod.named_parameters()]
    log.info(f"Found {len(lora_params)} LoRA weight matrices")
    return lora_params


def setup_lora_requires_grad(model):
    """
    Freeze all model parameters except LoRA parameters.
    """
    num_param = count_params(model, verbose=True)
    log.critical(f"Model has {num_param * 1e-6:.2f}M parameters before freezing")
    lora_params = get_all_lora_params(model)
    num_lora_param = sum([p.numel() for _, p in lora_params])
    log.info(f"Total number of LoRA parameters: {num_lora_param * 1e-6:.2f}M")
    if num_lora_param > 0:
        log.info("Freezing all parameters")
        model.requires_grad_(False)
        log.info("Unfreezing LoRA parameters")
        for name, param in lora_params:
            # log.info(f"Unfreezing loRA : {name}")
            param.requires_grad_(True)
        num_param = count_params(model, verbose=True)
        log.critical(f"Model has {num_param * 1e-6:.2f}M parameters after freezing")
    return num_lora_param


def add_lora_layers(model, peft_control_config):
    for i, block_name in enumerate(model.net.blocks):
        block = model.net.blocks[block_name]
        peft_control = peft_control_config.get(i, {})
        for j, subblock in enumerate(block.blocks):
            block_type = subblock.block_type
            peft_control_subblock = peft_control.get(block_type.upper(), {})
            customization_type = peft_control_subblock.get("customization_type", None)
            if customization_type == CustomizationType.LORA:
                if block_type.upper() in ["CA", "FA"]:
                    build_attn_lora(subblock.block.attn, peft_control_subblock)
