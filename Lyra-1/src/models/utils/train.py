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

def get_most_recent_checkpoint(output_dir: str) -> str | None:
    """
    Returns the most recent checkpoint directory from the given output directory.

    Args:
        output_dir (str): Path to the directory containing checkpoints.

    Returns:
        str | None: The name of the most recent checkpoint directory, or None if none exist.
    """
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return dirs[-1] if dirs else None