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

import torch
from megatron.core import ModelParallelConfig

from cosmos_predict1.utils.lazy_config import LazyDict


def create_model_parallel_config():
    model_parallel = ModelParallelConfig(bf16=True, params_dtype=getattr(torch, "bfloat16"))
    model_parallel.tensor_model_parallel_size = "${model.model_parallel.tensor_model_parallel_size}"
    model_parallel.context_parallel_size = "${model.model_parallel.context_parallel_size}"
    model_parallel.sequence_parallel = "${model.model_parallel.sequence_parallel}"
    MODEL_PARALLELS = LazyDict(
        dict(
            model_parallel_bf16=model_parallel,
        ),
        flags={"allow_objects": True},
    )
    return MODEL_PARALLELS["model_parallel_bf16"]
