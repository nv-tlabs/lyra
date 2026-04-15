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

from src.models.data.radym_wrapper import RadymWrapper

dataset_registry = {}

dataset_registry['default'] = {
    "has_latents": False,
    "is_w2c": False,
    "is_generated_cosmos_latent": False,
    "sampling_buckets": None,
    "end_view_idx": None,
    "start_view_target_idx": None,
    "end_view_target_idx": None,
}

# Training static

dataset_registry['lyra_static'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "/path/to/static",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Training dynamic

dataset_registry['lyra_dynamic'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "/path/to/dynamic",
        "is_static": False,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
        "end_view_idx": 5,
        "start_view_target_idx": 6,
        "end_view_target_idx": 11,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Static demo (pre-generated)

dataset_registry['lyra_static_demo'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/static/diffusion_output",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Static demo (self-generated)

dataset_registry['lyra_static_demo_generated'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/static/diffusion_output_generated",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Dynamic demo (pre-generated)

dataset_registry['lyra_dynamic_demo'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/dynamic/diffusion_output",
        "is_static": False,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
        "end_view_idx": 5,
        "start_view_target_idx": 6,
        "end_view_target_idx": 11,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Dynamic demo (self-generated)

dataset_registry['lyra_dynamic_demo_generated'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/dynamic/diffusion_output_generated",
        "is_static": False,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
        "end_view_idx": 5,
        "start_view_target_idx": 6,
        "end_view_target_idx": 11,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}
