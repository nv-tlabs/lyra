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
import random
import numpy as np
import torch
from accelerate.utils import (
    is_xpu_available,
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

RNG_STATE_NAME = "random_states"

def save_random_state(output_dir, process_index):
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    if is_xpu_available():
        states["torch_xpu_manual_seed"] = torch.xpu.get_rng_state_all()
    else:
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    if is_torch_xla_available():
        states["xm_seed"] = xm.get_rng_state()
    output_states_file = os.path.join(output_dir, states_name)
    torch.save(states, output_states_file)

def load_random_state(input_dir, process_index):
    try:
        states = torch.load(os.path.join(input_dir, f"{RNG_STATE_NAME}_{process_index}.pkl"))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(states["torch_xpu_manual_seed"])
        else:
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        if is_torch_xla_available():
            xm.set_rng_state(states["xm_seed"])
    except Exception:
        print(f"Failed to load random states from {input_dir}")