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

# Adapted for Long-LRM by Ziwen 2024
# from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

class Mamba2SingleScan(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        conv_init,
        expand,
        headdim,
        ngroups,
        A_init_range,
        dt_min,
        dt_max,
        dt_init_floor,
        dt_limit,
        learnable_init_states,
        activation,
        bias,
        conv_bias,
        # Fused kernel and sharding options
        chunk_size,
        device,
        dtype,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

    def forward(self, zxbcdt):
        """
        zxbcdt: (B, L, D)
        Returns: same shape as input 
        """
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # Fully fused path
        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=self.D,
            chunk_size=self.chunk_size,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=False,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        return out

class Mamba2MultiScan(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        conv_init,
        expand,
        headdim,
        ngroups,
        A_init_range,
        dt_min,
        dt_max,
        dt_init_floor,
        dt_limit,
        learnable_init_states,
        activation,
        bias,
        conv_bias,
        # Fused kernel and sharding options
        chunk_size,
        scan_type, # single, bi
        device,
        dtype,
        if_divide_out,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        assert scan_type in ["single", "bi"]
        self.scan_type = scan_type
        self.if_divide_out = if_divide_out

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        self.mamba_scans = nn.ModuleList()
        self.scan_num = 1
        if scan_type == "bi":
            self.scan_num = 2
        for _ in range(self.scan_num):
            self.mamba_scans.append(
                Mamba2SingleScan(
                    d_model,
                    d_state,
                    d_conv,
                    conv_init,
                    expand,
                    headdim,
                    ngroups,
                    A_init_range,
                    dt_min,
                    dt_max,
                    dt_init_floor,
                    dt_limit,
                    learnable_init_states,
                    activation,
                    bias,
                    conv_bias,
                    chunk_size,
                    device,
                    dtype,
                )
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as input
        """
        batch, seqlen, dim = hidden_states.shape

        xz = self.in_proj(hidden_states)  # (B, L, d_in_proj), [z,x,B,C,dt]

        xzs = [xz]
        if self.scan_type == "bi":
            xzs.append(xz.flip([1]))

        outs = []
        for i in range(self.scan_num):
            out = self.mamba_scans[i](xzs[i])
            if i == 0:
                outs.append(out)
            elif i == 1:
                outs.append(out.flip([1]))

        out = sum(outs)
        if self.if_divide_out:
            out = out / self.scan_num

        out = self.out_proj(out)

        return out

class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=256,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        scan_type="bi", # single, bi
        device=None,
        dtype=None,
        if_divide_out=False,
        norm_cls="rms_norm",
    ):
        super().__init__()
        assert norm_cls in ["rms_norm", "layer_norm"]
        if norm_cls=="rms_norm":
            self.norm = RMSNorm(d_model)
        elif norm_cls=="layer_norm":
            self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2MultiScan(d_model, d_state, d_conv, conv_init, expand, headdim, ngroups, A_init_range, dt_min, 
                            dt_max, dt_init_floor, dt_limit, learnable_init_states, activation, bias, conv_bias, 
                            chunk_size, scan_type, device, dtype, if_divide_out)
        
    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        x = x + self.mamba(self.norm(x))
        return x


if __name__ == "__main__":
    # Test Mamba2Block
    batch_size = 4
    seq_len = 128
    input_dim = 256
    
    model = Mamba2Block(d_model=input_dim, device="cuda").to("cuda")
    input = torch.randn(batch_size, seq_len, input_dim).to("cuda")
    output = model(input)

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
