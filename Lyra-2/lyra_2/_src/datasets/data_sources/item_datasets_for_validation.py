# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import os


@dataclasses.dataclass
class ItemDatasetConfig:
    path: str
    length: int


def get_itemdataset_option_local(name: str) -> ItemDatasetConfig:
    return ITEMDATASET_OPTIONS_LOCAL[name]


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

ITEMDATASET_OPTIONS_LOCAL = {
    "empty_string_umt5": ItemDatasetConfig(
        path=os.path.join(_REPO_ROOT, "checkpoints", "empty_string_umt5.pt"),
        length=1,
    ),
}
