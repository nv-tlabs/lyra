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

DATAVERSE_CONFIG = {
    "dl3dv_long_moge_chunk_81_480p_dav3_hsg": {
        "dataset_cfg": {
            "target": "lyra_2._src.datasets.radym.Radym",
            "params": {
                "root_path": "",
                "filter_list_path": "",
            },
        },
        "data_name": "dl3dv_long_moge_chunk_81",
        "sample_n_frames": 1000,
        "video_mirror": True,
        "video_mirror_when_short_only": True,
        "sample_size": [480, 854],
        "crop_size": [480, 832],
        "t5_embedding_path": "",
    }
}
