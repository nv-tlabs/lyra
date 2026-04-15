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
from omegaconf import DictConfig, OmegaConf


class CustomSimpleNamespace:
    """
    A simple namespace class that supports both attribute-style and dictionary-style access.
    """

    def __init__(self, d):
        self._d = d

    def __getattr__(self, attr):
        # Attribute-style access: config.key
        try:
            return self._d[attr]
        except KeyError:
            raise AttributeError(f"'CustomSimpleNamespace' object has no attribute '{attr}'")

    def __getitem__(self, key):
        # Dictionary-style access: config['key']
        return self._d[key]


def maybe_convert_to_namespace(config):
    """
    This function cast a OmegaConf's DictConfig or a standard dict to CustomSimpleNamespace, which supports both
    attribute-style and dictionary-style access.
    Note: We need to convert OmegaConf's DictConfig since it is not compatible with torch.compile.
    """
    # If input is OmegaConf's DictConfig, convert to a standard dict
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    if isinstance(config, dict):
        return CustomSimpleNamespace(config)
    else:
        return config


def random_dropout(embeddings, drop_rate):
    r"""
    Function to perform random dropout for embeddings.
    When we drop embeddings, we zero them out.
    Args:
        embeddings (tensor): Input embeddings
        drop_rate (float): Rate of dropping the embedding.
    """
    num_samples = embeddings.shape[0]
    # Create a shape (num_samples, 1, 1, 1, 1, ...) depending on embeddings dim.
    # This is done to ensure we can broadcast the zero_flag to the embeddings.
    # embeddings.ndim is 3 for images, and 4 for videos, and the corresponding
    # shapes are (num_samples, 1, 1) and (num_samples, 1, 1, 1) respectively.
    tensor_shape = (num_samples,) + tuple([1] * (embeddings.ndim - 1))
    zero_flag = torch.ones(tensor_shape).to(embeddings.dtype) * (1 - drop_rate)
    zero_flag = torch.bernoulli(zero_flag).to(embeddings.device)
    embeddings = embeddings * zero_flag
    return embeddings
