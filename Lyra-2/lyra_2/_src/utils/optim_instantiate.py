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

import re

import torch
from omegaconf import ListConfig
from torch import nn

from lyra_2._ext.imaginaire.utils import log


def get_regular_param_group(net: nn.Module):
    """
    seperate the parameters of the network into two groups: decay and no_decay.
    based on nano_gpt codebase.
    """
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return decay_params, nodecay_params


def get_base_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    net_decay_param, net_nodecay_param = get_regular_param_group(model)

    num_decay_params = sum(p.numel() for p in net_decay_param)
    num_nodecay_params = sum(p.numel() for p in net_nodecay_param)
    net_param_total = num_decay_params + num_nodecay_params
    log.info(f"total num parameters : {net_param_total:,}")

    param_group = [
        {
            "params": net_decay_param + net_nodecay_param,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    for k, v in kwargs.items():
        if isinstance(v, ListConfig):
            kwargs[k] = list(v)

    return opt_cls(param_group, **kwargs)


def get_multiple_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    lr_overrides: list[dict] = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Get an optimizer with multiple learning rates for different parts of the model,
    allowing pattern matching for parameter names.

    The logic is:
    1. All parameters are initially considered for the default learning rate.
    2. We iterate through lr_overrides. If a parameter's name matches a pattern,
       it's moved to a group with the specified learning rate. A parameter is only
       assigned to the *first* pattern it matches.

    Args:
        model (nn.Module): The model to optimize.
        lr (float): The default learning rate.
        weight_decay (float): The default weight decay.
        optim_type (str): The type of optimizer to use ('adamw' or 'fusedadam').
        lr_overrides (list[dict], optional): A list of dicts with keys:
              - 'pattern' (str): The pattern to match (required)
              - 'lr' (float): The learning rate for matching params (required)
              - 'match_type' (str): 'regex', 'contains', 'startswith', 'endswith' (default: 'contains')

            Example:
            [
                {'pattern': 'cross_view_attn', 'lr': 2e-4, 'match_type': 'contains'},
                {'pattern': 'text_encoder', 'lr': 1e-5, 'match_type': 'contains'},
            ]

            This is Hydra-friendly and can be overridden from command line like:
            optimizer.lr_overrides.0.lr=1e-4
            optimizer.lr_overrides.1.pattern=vision_encoder
        **kwargs: Additional arguments for the optimizer.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Initialize groups for parameters with overridden LR
    override_groups = {}  # (lr, has_decay) -> [params]
    override_groups_name = {}  # (lr, has_decay) -> [name]

    # Initialize lists for parameters with default LR
    default_decay_params = []
    default_nodecay_params = []

    # Temporarily hold all params to check against overrides
    unassigned_params = list(param_dict.items())

    # First, assign params that match an override pattern
    if lr_overrides:
        override_list = lr_overrides

        for name, p in list(unassigned_params):
            assigned = False
            for override_item in override_list:
                pattern = override_item["pattern"]
                special_lr = override_item["lr"]
                match_type = override_item.get("match_type", "contains")

                # Determine if the parameter name matches
                matched = False
                if match_type == "regex":
                    matched = re.match(pattern, name) is not None
                elif match_type == "contains":
                    matched = pattern in name
                elif match_type == "startswith":
                    matched = name.startswith(pattern)
                elif match_type == "endswith":
                    matched = name.endswith(pattern)
                else:
                    raise ValueError(
                        f"Unknown match_type: {match_type}. Must be one of: regex, contains, startswith, endswith"
                    )

                if matched:
                    has_decay = p.dim() >= 2
                    group_key = (special_lr, has_decay)
                    if group_key not in override_groups:
                        override_groups[group_key] = []
                    if group_key not in override_groups_name:
                        override_groups_name[group_key] = []
                    override_groups[group_key].append(p)
                    override_groups_name[group_key].append(name)
                    assigned = True
                    break  # Assign to first matching pattern
            if assigned:
                # Remove from unassigned list; this is a bit inefficient but clear
                unassigned_params = [(n, param) for n, param in unassigned_params if n != name]

    # Assign all remaining params to default groups
    for name, p in unassigned_params:
        if p.dim() >= 2:
            default_decay_params.append(p)
        else:
            default_nodecay_params.append(p)

    # Build final param_groups list for the optimizer
    final_param_groups = []
    if default_decay_params:
        final_param_groups.append({"params": default_decay_params, "lr": lr, "weight_decay": weight_decay})
    if default_nodecay_params:
        final_param_groups.append({"params": default_nodecay_params, "lr": lr, "weight_decay": 0.0})

    for (special_lr, has_decay), params in override_groups.items():
        final_param_groups.append(
            {"params": params, "lr": special_lr, "weight_decay": weight_decay if has_decay else 0.0}
        )

    # print the parameter names in each group
    for (special_lr, has_decay), params in override_groups.items():
        log.critical(f"special_lr {special_lr}: {override_groups_name[(special_lr, has_decay)]}")

    # Log parameter group information
    total_params = 0
    log.critical("Optimizer parameter groups:")
    for i, group in enumerate(final_param_groups):
        group_params = sum(p.numel() for p in group["params"])
        total_params += group_params
        log.critical(
            f"  Group {i}: num_params={group_params:,}, lr={group['lr']:.1e}, weight_decay={group['weight_decay']}"
        )
    log.critical(f"Total trainable parameters: {total_params:,}")

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    for k, v in kwargs.items():
        if isinstance(v, ListConfig):
            kwargs[k] = list(v)

    return opt_cls(final_param_groups, **kwargs)
