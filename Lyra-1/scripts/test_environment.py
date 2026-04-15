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

import argparse
import importlib
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="store_true",
        help="Whether to check training-specific dependencies",
    )
    return parser.parse_args()


def check_packages(package_list):
    global all_success
    for package in package_list:
        try:
            _ = importlib.import_module(package)
        except Exception as e:
            print(f"\033[91m[ERROR]\033[0m Package not successfully imported: \033[93m{package}\033[0m")
            all_success = False
        else:
            print(f"\033[92m[SUCCESS]\033[0m {package} found")


args = parse_args()

if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
    detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\033[91m[ERROR]\033[0m Python 3.10+ is required. You have: \033[93m{detected}\033[0m")
    sys.exit(1)

if "CONDA_PREFIX" not in os.environ:
    print("\033[93m[WARNING]\033[0m Cosmos should be run under a conda environment.")

print("Attempting to import critical packages...")

packages = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "megatron.core",
    "transformer_engine",
]
packages_training = [
    "apex.multi_tensor_apply",
]
all_success = True

check_packages(packages)
if args.training:
    check_packages(packages_training)

if all_success:
    print("-----------------------------------------------------------")
    print("\033[92m[SUCCESS]\033[0m Cosmos environment setup is successful!")
