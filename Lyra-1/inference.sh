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

cfg_file_name=3dgs_res_176_320_views_17.yaml
# cfg_file_name=3dgs_res_176_320_views_49.yaml
# cfg_file_name=3dgs_res_352_640_views_49.yaml
# cfg_file_name=3dgs_res_704_1280_views_49.yaml
# cfg_file_name=3dgs_res_704_1280_views_121.yaml
# cfg_file_name=3dgs_res_704_1280_views_121_multi_6.yaml
# cfg_file_name=3dgs_res_704_1280_views_121_multi_6_prune.yaml
# cfg_file_name=3dgs_res_704_1280_views_121_multi_6_dynamic.yaml
# cfg_file_name=3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml

out_dir_main=outputs/lyra

cfg=configs/inference/$cfg_file_name
accelerate launch sample.py --config $cfg