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

"""registry for commandline override options for config."""
from cosmos_predict1.utils.lazy_config import LazyDict


def create_debug_job_with_mock_data(full_experiment_name):
    job_dict = dict(
        defaults=[
            f"/experiment/{full_experiment_name.replace('-', '_')}",
            {"override /data_train": "mock_video360"},
            {"override /data_val": "mock_video360"},
            "_self_",
        ],
        job=dict(group="debug", name=f"mock_{full_experiment_name}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
        trainer=dict(
            max_iter=2,
            logging_iter=1,
            max_val_iter=1,
            validation_iter=2,
        ),
        checkpoint=dict(
            strict_resume=False,
            load_training_state=False,
            save_iter=2,
        ),
    )
    return LazyDict(job_dict)
