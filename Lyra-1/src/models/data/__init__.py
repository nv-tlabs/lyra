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

import time
import torch
from src.models.data.provider import Provider

def get_multi_dataloader(opt, accelerator=None):
    train_datasets, test_datasets = get_datasets(opt, accelerator)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )


    return train_dataloader, test_dataloader

def get_datasets(opt, accelerator=None):
    train_datasets = []
    test_datasets = []

    for idx in range(len(opt.data_mode)):
        begin_time = time.time()
        if isinstance(opt.data_mode[idx], str):
            dataset_name, num_repeat = opt.data_mode[idx], 1
        else:
            dataset_name, num_repeat = opt.data_mode[idx]

        train_dataset = Provider(dataset_name, opt, training=True, num_repeat=num_repeat)
        train_datasets.append(train_dataset)

        test_dataset = Provider(dataset_name, opt, training=False, num_repeat=num_repeat)
        test_datasets.append(test_dataset)
        if accelerator is None or accelerator.is_main_process:
            print(f"Loaded {dataset_name}, train size: {len(train_dataset)}, test size: {len(test_dataset)}, loading took {time.time() - begin_time} seconds")

    return train_datasets, test_datasets
