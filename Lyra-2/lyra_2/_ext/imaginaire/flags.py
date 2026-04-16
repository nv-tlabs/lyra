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

"""Feature flags."""

import os
from dataclasses import dataclass


def _parse_bool(value: str) -> bool:
    """Parse string to a boolean."""
    return value.lower() in ["true", "1", "yes", "y"]


INTERNAL = _parse_bool(os.environ.get("COSMOS_INTERNAL", "0"))
"""Whether to enable internal features."""

SMOKE = _parse_bool(os.environ.get("COSMOS_SMOKE", "0"))
"""Whether to enable smoke test.

Disables expensive operations such as checkpoint loading.
"""

VERBOSE = _parse_bool(os.environ.get("COSMOS_VERBOSE", "0"))
"""Whether to enable verbose output."""

EXPERIMENTAL_CHECKPOINTS = _parse_bool(os.environ.get("COSMOS_EXPERIMENTAL_CHECKPOINTS", "0"))
"""Whether to enable experimental checkpoints."""


@dataclass
class Flags:
    internal: bool = INTERNAL
    smoke: bool = SMOKE
    verbose: bool = VERBOSE
    experimental_checkpoints: bool = EXPERIMENTAL_CHECKPOINTS


FLAGS = Flags()
"""Convenience object for accessing flags."""
