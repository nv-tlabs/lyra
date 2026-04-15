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

"""
Usage:
    pytest -s cosmos_predict1/diffusion/training/utils/peft/lora_attn_test.py
"""

import copy

import pytest
import torch
import torch.nn as nn
from einops import rearrange, repeat
from loguru import logger

from cosmos_predict1.diffusion.config.base.net import FADITV2Config
from cosmos_predict1.diffusion.training.utils.layer_control.peft_control_config_parser import LayerControlConfigParser
from cosmos_predict1.diffusion.training.utils.peft.peft import add_lora_layers, get_all_lora_params
from cosmos_predict1.utils.lazy_config import instantiate


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        dummy_net = copy.deepcopy(FADITV2Config)
        dummy_net.num_blocks = 2
        dummy_net.model_channels = 256
        dummy_net.num_heads = 8
        self.net = instantiate(dummy_net).cuda()


@pytest.fixture()
def block1_peft_control():
    """
    This config has the following edits for the following blocks:
    Block 0: FA, CA edits for ALL sub-blocks
    """
    config = {
        "enabled": "True",
        "edits": [
            {
                "blocks": "\\b\\d*([1])\\b",
                "customization_type": "LoRA",
                "rank": 8,
                "scale": 0.6,
                "block_edit": [
                    "FA[to_q:8:0.8, to_k:16:1.2, to_v:4:64, to_out:8]",
                    "CA[to_q:16, to_k:16, to_v:4, to_out:32]",
                ],
            },
        ],
        "customization_type": "LoRA",
        "rank": 8,
        "scale": 0.8,
    }
    config_parser = LayerControlConfigParser(config)
    return config_parser.parse()


def test_model_without_lora():
    model = DummyModel()
    lora_params = get_all_lora_params(model)
    actual = len(lora_params)
    expected = 0
    assert actual == expected, f"Expected {expected} LoRA layers, got {actual}"


def test_model_with_lora(block1_peft_control):
    model = DummyModel()
    add_lora_layers(model, block1_peft_control)
    lora_params = get_all_lora_params(model)
    actual = len(lora_params)
    expected = 16
    assert actual == expected, f"Expected {expected} LoRA layers, got {actual}"


def test_model_cal_qkv_lora_matches_base_version_at_init(block1_peft_control):
    model = DummyModel()
    # isolate a single attention layer
    block_idx = 1
    attn = model.net.blocks[f"block{block_idx}"].blocks[0].block.attn
    x = torch.rand(2, 16, 256).cuda()  # batch size, seq len, embed size

    q_base, k_base, v_base = attn.cal_qkv(x)
    add_lora_layers(model, block1_peft_control)
    model.cuda()
    q_lora, k_lora, v_lora = attn.cal_qkv(x)

    assert torch.allclose(q_base, q_lora)
    assert torch.allclose(k_base, k_lora)
    assert torch.allclose(v_base, v_lora)


def test_model_cal_qkv_lora_with_non_zero_lora(block1_peft_control):
    model = DummyModel()
    block_idx = 1
    self_attn = model.net.blocks[f"block{block_idx}"].blocks[0].block.attn
    cross_attn = model.net.blocks[f"block{block_idx}"].blocks[1].block.attn
    # Set q_norm and k_norm to Identity
    for attn in [self_attn, cross_attn]:
        attn.to_q[0].weight.data.fill_(0.1)
        attn.to_k[0].weight.data.fill_(0.1)
        attn.to_v[0].weight.data.fill_(0.1)
        attn.to_q[1] = nn.Identity()  # Set normalization to Identity
        attn.to_k[1] = nn.Identity()
        attn.to_v[1] = nn.Identity()
        attn.to_q[1].cuda()
        attn.to_k[1].cuda()
        attn.to_v[1].cuda()

    q_base, k_base, v_base = {}, {}, {}
    x = torch.ones(2, 16, 256).cuda()  # batch size, seq len, embed size
    cross_attn_context = torch.ones(2, 16, 1024).cuda()
    context_dim = {"FA": 256, "CA": 1024}
    input_context = {"FA": x, "CA": cross_attn_context}

    # Compute base qkv for both self and cross attention
    for attn_name, attn in [("FA", self_attn), ("CA", cross_attn)]:
        q_base[attn_name], k_base[attn_name], v_base[attn_name] = attn.cal_qkv(x, input_context[attn_name])
    # add lora layers
    add_lora_layers(model, block1_peft_control)
    model.cuda()

    # compute lora qkv with non-zero lora weights
    for attn_name, attn in [("FA", self_attn), ("CA", cross_attn)]:
        attn.to_q_lora.net[0].weight.data.fill_(0.1)
        attn.to_q_lora.net[1].weight.data.fill_(0.2)

        attn.to_k_lora.net[0].weight.data.fill_(0.3)
        attn.to_k_lora.net[1].weight.data.fill_(0.4)

        attn.to_v_lora.net[0].weight.data.fill_(0.5)
        attn.to_v_lora.net[1].weight.data.fill_(0.6)

        q_lora, k_lora, v_lora = attn.cal_qkv(x, input_context[attn_name])

        # Compare with expected lora qkv
        self_attn_q_lora_scale = float(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_q", {}).get("lora_scale")
        )
        self_attn_q_lora_rank = int(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_q", {}).get("lora_rank")
        )
        q_lora_diff = 256 * 0.1 * self_attn_q_lora_rank * 0.2

        self_attn_k_lora_scale = float(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_k", {}).get("lora_scale")
        )
        self_attn_k_lora_rank = int(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_k", {}).get("lora_rank")
        )
        k_lora_diff = context_dim[attn_name] * 0.3 * self_attn_k_lora_rank * 0.4

        self_attn_v_lora_scale = float(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_v", {}).get("lora_scale")
        )
        self_attn_v_lora_rank = int(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_v", {}).get("lora_rank")
        )
        v_lora_diff = context_dim[attn_name] * 0.5 * self_attn_v_lora_rank * 0.6

        expected_q_lora = q_base[attn_name] + self_attn_q_lora_scale * q_lora_diff
        expected_k_lora = k_base[attn_name] + self_attn_k_lora_scale * k_lora_diff
        expected_v_lora = v_base[attn_name] + self_attn_v_lora_scale * v_lora_diff
        logger.info(f"attn_name: {attn_name}, q_lora: {q_lora.shape}, expected_q_lora: {expected_q_lora.shape}")
        assert torch.allclose(
            q_lora, expected_q_lora, rtol=1e-2
        ), f"q_lora: {q_lora[0, 0, 0, :2]}, expected_q_lora: {expected_q_lora[0, 0, 0, :2]}"
        assert torch.allclose(
            k_lora, expected_k_lora, rtol=1e-2
        ), f"k_lora: {k_lora[0, 0, 0, :2]}, expected_k_lora: {expected_k_lora[0, 0, 0, :2]}"
        assert torch.allclose(
            v_lora, expected_v_lora, rtol=1e-2
        ), f"v_lora: {v_lora[0, 0, 0, :2]}, expected_v_lora: {expected_v_lora[0, 0, 0, :2]}"


def test_model_cal_attn_lora_matches_base_version_at_init(block1_peft_control):
    model = DummyModel()
    q = torch.rand(2, 16, 8, 32).cuda()
    k = torch.rand(2, 16, 8, 32).cuda()
    v = torch.rand(2, 16, 8, 32).cuda()

    # isolate a single attention layer
    block_idx = 1
    attn = model.net.blocks[f"block{block_idx}"].blocks[0].block.attn
    attn_output_base = attn.cal_attn(q, k, v)  # [2, 16, 256]

    add_lora_layers(model, block1_peft_control)
    model.cuda()
    attn_output_lora = attn.cal_attn(q, k, v)

    assert torch.allclose(attn_output_base, attn_output_lora)


def test_model_cal_attn_lora_with_non_zero_output_lora(block1_peft_control):
    model = DummyModel()
    block_idx = 1
    self_attn = model.net.blocks[f"block{block_idx}"].blocks[0].block.attn
    cross_attn = model.net.blocks[f"block{block_idx}"].blocks[1].block.attn
    for attn_name, attn in [("FA", self_attn), ("CA", cross_attn)]:
        # Overwrite attn_op to return ones of shape [2, 16, 256] and output_dropout to be Identity
        class OnesAttnOp(nn.Module):
            def forward(self, *args, **kwargs):
                return torch.ones([2, 16, 256]).cuda()

        attn.attn_op = OnesAttnOp()
        attn.to_out[0].weight.data.fill_(0.1)
        attn.to_out[1] = nn.Identity()  # Remove dropout

        # Compute base attn output
        q = torch.rand(2, 16, 8, 32).cuda()
        k = torch.rand(2, 16, 8, 32).cuda()
        v = torch.rand(2, 16, 8, 32).cuda()
        attn_output_base = attn.cal_attn(q, k, v)

        # Add lora layers
        add_lora_layers(model, block1_peft_control)
        model.cuda()
        # Set lora weights to non-zero
        attn.to_out_lora.net[0].weight.data.fill_(0.1)
        attn.to_out_lora.net[1].weight.data.fill_(0.2)

        # Compute lora attn output
        attn_output_lora = attn.cal_attn(q, k, v)

        # Compare with expected lora attn output
        output_lora_scale = float(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_out", {}).get("lora_scale")
        )
        output_lora_rank = int(
            block1_peft_control.get(block_idx, {}).get(attn_name, {}).get("to_out", {}).get("lora_rank")
        )

        expected_attn_output_lora = attn_output_base + output_lora_scale * 256 * 0.1 * output_lora_rank * 0.2
        assert torch.allclose(
            attn_output_lora, expected_attn_output_lora, rtol=1e-2
        ), f"attn_output_lora: {attn_output_lora[0, 0, :2]}, expected_attn_output_lora: {expected_attn_output_lora[0, 0, :2]}"
