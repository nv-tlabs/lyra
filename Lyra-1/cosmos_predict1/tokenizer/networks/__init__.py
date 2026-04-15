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

from enum import Enum

from cosmos_predict1.tokenizer.networks.configs import continuous_image_8x8_360p as continuous_image_8x8_360p_dict
from cosmos_predict1.tokenizer.networks.configs import continuous_image_16x16_360p as continuous_image_16x16_360p_dict
from cosmos_predict1.tokenizer.networks.configs import continuous_video_4x8x8_360p as continuous_video_4x8x8_360p_dict
from cosmos_predict1.tokenizer.networks.configs import continuous_video_8x8x8_720p as continuous_video_8x8x8_720p_dict
from cosmos_predict1.tokenizer.networks.configs import discrete_image_8x8_360p as discrete_image_8x8_360p_dict
from cosmos_predict1.tokenizer.networks.configs import discrete_image_16x16_360p as discrete_image_16x16_360p_dict
from cosmos_predict1.tokenizer.networks.configs import discrete_video_4x8x8_360p as discrete_video_4x8x8_360p_dict
from cosmos_predict1.tokenizer.networks.configs import discrete_video_8x16x16_720p as discrete_video_8x16x16_720p_dict
from cosmos_predict1.tokenizer.networks.continuous_image import ContinuousImageTokenizer
from cosmos_predict1.tokenizer.networks.continuous_video import CausalContinuousVideoTokenizer
from cosmos_predict1.tokenizer.networks.discrete_image import DiscreteImageTokenizer
from cosmos_predict1.tokenizer.networks.discrete_video import CausalDiscreteVideoTokenizer


class TokenizerConfigs(Enum):
    """Continuous Image (CI) Tokenizer Configs"""

    # Cosmos-Tokenize1-CI8x8-360p
    CI8x8_360p = continuous_image_8x8_360p_dict

    # Cosmos-Tokenize1-CI16x16-360p
    CI16x16_360p = continuous_image_16x16_360p_dict

    """Discrete Image (DI) Tokenizer Configs"""
    # Cosmos-Tokenize1-DI8x8-360p
    DI8x8_360p = discrete_image_8x8_360p_dict

    # Cosmos-Tokenize1-DI16x16-360p
    DI16x16_360p = discrete_image_16x16_360p_dict

    """Causal Continuous Video (CV) Tokenizer Configs"""
    # Cosmos-Tokenize1-CV8x8x8-720p
    CV8x8x8_720p = continuous_video_8x8x8_720p_dict

    # Cosmos-Tokenize1-CV4x8x8-360p
    CV4x8x8_360p = continuous_video_4x8x8_360p_dict

    """Causal Discrete Video (DV) Tokenizer Configs"""
    # Cosmos-Tokenize1-DV8x16x16-720p
    DV8x16x16_720p = discrete_video_8x16x16_720p_dict

    # Cosmos-Tokenize1-DV4x8x8-360p
    DV4x8x8_360p = discrete_video_4x8x8_360p_dict


class TokenizerModels(Enum):
    CI = ContinuousImageTokenizer
    DI = DiscreteImageTokenizer
    CV = CausalContinuousVideoTokenizer
    DV = CausalDiscreteVideoTokenizer
