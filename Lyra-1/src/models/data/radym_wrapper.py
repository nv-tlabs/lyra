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

import torch
import os
from typing import Any, List, Optional
from src.models.data.radym import Radym

class RadymWrapper(Radym):
    def __init__(self, is_static: bool = True, is_multi_view: bool = False, **kwargs):
        super().__init__(**kwargs)
    
        # For recon code base
        self.is_static = is_static
        self.sample_list = self.mp4_file_paths
        self.num_cameras = len([camera_name for camera_name in os.listdir(self.root_path) if camera_name != 'flag']) if is_multi_view else 1
        if is_multi_view:
            self.n_views = self.num_cameras
    
    def __len__(self):
        return len(self.sample_list)
    
    def count_frames(self, video_idx: int):
        return self.num_frames(video_idx)
    
    def count_cameras(self, video_idx: int):
        return self.num_cameras
    
    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
        camera_convention: str = "opencv",
    ):
        assert camera_convention == 'opencv', f"No support for camera convention {camera_convention}"
        if view_indices is None or len(view_indices) == 0:
            view_indices = list(range(self.count_cameras(idx)))
        final_dict = None
        for view_idx in view_indices:
            output_dict = self._read_data(
                idx, frame_indices, [view_idx], data_fields,
            )
            if final_dict is None:
                final_dict = output_dict
            else:
                for k in final_dict:
                    if k == "__key__":
                        continue
                    final_dict[k] = torch.concatenate([final_dict[k], output_dict[k]])
        return final_dict