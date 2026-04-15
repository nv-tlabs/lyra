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

"""Metric configurations for the tokenizer model.

Support for PSNR or SSIM, there are validation only metrics.
"""
import attrs

from cosmos_predict1.tokenizer.training.metrics import CodeUsageMetric, PSNRMetric, SSIMMetric, TokenizerMetric
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class Metric:
    # The combined loss function, and its reduction mode.
    PSNR: LazyDict = L(PSNRMetric)()
    SSIM: LazyDict = L(SSIMMetric)()


@attrs.define(slots=False)
class DiscreteTokenizerMetric:
    # with code usage (perplexity PPL), for discrete tokenizers only
    PSNR: LazyDict = L(PSNRMetric)()
    SSIM: LazyDict = L(SSIMMetric)()
    CodeUsage: LazyDict = L(CodeUsageMetric)(codebook_size=64000)


MetricConfig: LazyDict = L(TokenizerMetric)(config=Metric())

DiscreteTokenizerMetricConfig: LazyDict = L(TokenizerMetric)(config=DiscreteTokenizerMetric())
