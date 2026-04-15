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
from pathlib import Path

import numpy as np

from cosmos_predict1.auxiliary.guardrail.blocklist.blocklist import Blocklist
from cosmos_predict1.auxiliary.guardrail.common.core import GuardrailRunner
from cosmos_predict1.auxiliary.guardrail.face_blur_filter.face_blur_filter import RetinaFaceFilter
from cosmos_predict1.auxiliary.guardrail.llamaGuard3.llamaGuard3 import LlamaGuard3
from cosmos_predict1.auxiliary.guardrail.video_content_safety_filter.video_content_safety_filter import (
    VideoContentSafetyFilter,
)
from cosmos_predict1.utils import log


def create_text_guardrail_runner(checkpoint_dir: str) -> GuardrailRunner:
    """Create the text guardrail runner."""
    return GuardrailRunner(safety_models=[Blocklist(checkpoint_dir), LlamaGuard3(checkpoint_dir)])


def create_video_guardrail_runner(checkpoint_dir: str) -> GuardrailRunner:
    """Create the video guardrail runner."""
    return GuardrailRunner(
        safety_models=[VideoContentSafetyFilter(checkpoint_dir)],
        postprocessors=[RetinaFaceFilter(checkpoint_dir)],
    )


def run_text_guardrail(prompt: str, guardrail_runner: GuardrailRunner) -> bool:
    """Run the text guardrail on the prompt, checking for content safety.

    Args:
        prompt: The text prompt.
        guardrail_runner: The text guardrail runner.

    Returns:
        bool: Whether the prompt is safe.
    """
    is_safe, message = guardrail_runner.run_safety_check(prompt)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
    return is_safe


def run_video_guardrail(frames: np.ndarray, guardrail_runner: GuardrailRunner) -> np.ndarray | None:
    """Run the video guardrail on the frames, checking for content safety and applying face blur.

    Args:
        frames: The frames of the generated video.
        guardrail_runner: The video guardrail runner.

    Returns:
        The processed frames if safe, otherwise None.
    """
    is_safe, message = guardrail_runner.run_safety_check(frames)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
        return None

    frames = guardrail_runner.postprocess(frames)
    return frames
