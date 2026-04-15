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

max_time_idx=120
fps=30
static_scene_time_idx_list=(0 120)

### Define input folder with 0...120 as subfolders
in_path_main=/path/to/time_indices/
out_path_main=/path/to/output/
###

in_path_sub=set_target_single/sample_stride_1/static_view_indices_fixed_5_0_1_2_3_4
out_path_main=out_path_main/$fps
out_path_static_camera_single_videos=$out_path_main/static_camera/single_videos/
out_path_scene_camera_single_videos=$out_path_main/scene_camera/single_videos/

python src/visu/dynamic_scene_video_scene_and_camera.py \
  --base_path $in_path_main \
  --sub_path_after_time_index $in_path_sub \
  --output_dir $out_path_scene_camera_single_videos \
  --max_time_idx $max_time_idx \
  --fps $fps

for static_scene_time_idx in "${static_scene_time_idx_list[@]}"; do
    in_path_videos_static_scene=$in_path_main/$static_scene_time_idx/$in_path_sub
    out_path_static_scene_single_videos=$out_path_main/static_scene/single_videos/$static_scene_time_idx/
    out_path_static_scene_wave_videos=$out_path_main/static_scene/wave_videos/$static_scene_time_idx/
    python src/visu/dynamic_scene_video_static_scene.py \
    --mp4_input $in_path_videos_static_scene \
    --single_output_dir $out_path_static_scene_single_videos \
    --wave_output_dir $out_path_static_scene_wave_videos \
    --use_mirror_views \
    --use_mirror_wave \
    --fps $fps
done

python src/visu/dynamic_scene_video_static_camera.py \
    --main_path $in_path_main \
    --sub_path $in_path_sub \
    --output_path $out_path_static_camera_single_videos \
    --num_frames 1 \
    --fps $fps \
    --max_time_idx $max_time_idx

path1=$out_path_main/static_scene/wave_videos/0/
path2=$out_path_main/scene_camera/single_videos/
path3=$out_path_main/static_camera/single_videos/
path4=$out_path_main/static_scene/single_videos/120/
out_path_main_single_videos=$out_path_main/main/single_videos/
python src/visu/dynamic_scene_merge.py $path1 $path2 $path2 $path3 $path3 $path4 -s 1 1 1 1 1 2 -o $out_path_main_single_videos

out_path_main_grid_videos=$out_path_main/main/grid_videos/
python src/visu/dynamic_grid.py $out_path_main_single_videos --grid_width 4 --grid_height 2 -o $out_path_main_grid_videos
