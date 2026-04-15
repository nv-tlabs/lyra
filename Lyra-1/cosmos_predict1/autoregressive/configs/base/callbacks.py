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

from cosmos_predict1.autoregressive.callbacks.video_sampling_teacher_forcing import VideoSamplingTeacherForcing
from cosmos_predict1.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.lazy_config import LazyCall as L

BASIC_CALLBACKS = dict(
    progress_bar=L(ProgressBarCallback)(),
    grad_clip=L(GradClip)(clip_norm=1.0, fsdp_enabled="${model.model_config.fsdp_enabled}", model_key="model"),
)

VIDEO_TEACHER_FORCING_CALLBACK = dict(
    vid_sampling_tf=L(VideoSamplingTeacherForcing)(
        every_n=500,
        video_latent_shape="${model.model_config.video_latent_shape}",
        num_frames_to_display=4,
        save_folder="video_sampling_teacher_forcing",
    )
)
