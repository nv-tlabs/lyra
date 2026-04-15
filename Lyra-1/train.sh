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

config_file=configs/accelerate/accelerate_config.yaml
train_file=train.py

### Static 3D Generation ###

# Stage 1
config1=configs/training/3dgs_res_176_320_views_17.yaml
# accelerate launch --config_file $config_file $train_file --config $config1

# Stage 2
config2=configs/training/3dgs_res_176_320_views_49.yaml
python src/utils/copy_to_resume.py $config1 $config2
# accelerate launch --config_file $config_file $train_file --config $config2

# Stage 3
config3=configs/training/3dgs_res_352_640_views_49.yaml
python src/utils/copy_to_resume.py $config2 $config3
# accelerate launch --config_file $config_file $train_file --config $config3

# Stage 4
config4=configs/training/3dgs_res_704_1280_views_49.yaml
python src/utils/copy_to_resume.py $config3 $config4
# accelerate launch --config_file $config_file $train_file --config $config4

# Stage 5
config5=configs/training/3dgs_res_704_1280_views_121.yaml
python src/utils/copy_to_resume.py $config4 $config5
# accelerate launch --config_file $config_file $train_file --config $config5

# Stage 6
config6=configs/training/3dgs_res_704_1280_views_121_multi_6.yaml
python src/utils/copy_to_resume.py $config5 $config6
# accelerate launch --config_file $config_file $train_file --config $config6

# Stage 7
config7=configs/training/3dgs_res_704_1280_views_121_multi_6_prune.yaml
python src/utils/copy_to_resume.py $config6 $config7
# accelerate launch --config_file $config_file $train_file --config $config7

### Dynamic 3D Generation ###
# Stage 8
config8=configs/training/3dgs_res_704_1280_views_121_multi_6_dynamic.yaml
python src/utils/copy_to_resume.py $config6 $config8
accelerate launch --config_file $config_file $train_file --config $config8

config9=configs/training/3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml
python src/utils/copy_to_resume.py $config8 $config9
accelerate launch --config_file $config_file $train_file --config $config9
