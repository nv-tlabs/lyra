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
from omegaconf import OmegaConf
from typing import List

def load_and_merge_configs(config_paths: List[str]):
    """
    Load and merge multiple OmegaConf configs in order.
    Later configs override earlier ones.
    Any missing keys in later configs are added to the schema as None.

    Args:
        config_paths (List[str]): List of paths to config files. 
                                  The first config acts as the base schema.

    Returns:
        OmegaConf.DictConfig: The merged configuration.
    """
    if not config_paths:
        raise ValueError("No config paths provided.")

    # Start with the first config as schema
    schema = OmegaConf.load(config_paths[0])

    # Iteratively merge the rest
    for path in config_paths[1:]:
        cfg = OmegaConf.load(path)

        # Add missing keys into schema
        missing_keys = set(cfg.keys()) - set(schema.keys())
        for key in missing_keys:
            OmegaConf.update(schema, key, None, force_add=True)

        # Merge current config into schema
        schema = OmegaConf.merge(schema, cfg)

    return schema


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

dtype_map = {
    'float32': torch.float32,
    'float': torch.float32,
    'float64': torch.float64,
    'double': torch.float64,
    'float16': torch.float16,
    'half': torch.float16,
    'bfloat16': torch.bfloat16,
    'int32': torch.int32,
    'int64': torch.int64,
    'long': torch.int64,
}
