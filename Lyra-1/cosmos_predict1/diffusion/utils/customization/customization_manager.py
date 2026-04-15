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


class CustomizationType(Enum):
    LORA = 1
    REPLACE = 2

    @classmethod
    def from_value(cls, value):
        """Convert both int and str to the corresponding enum."""
        if isinstance(value, str):
            value = value.lower()
            if value == "lora":
                return cls.LORA
            elif value == "replace":
                return cls.REPLACE
            elif value == "":
                return None
            else:
                raise ValueError("Customization type must be lora or replace")
        raise TypeError("CustomizationType must be specified as a string.")
