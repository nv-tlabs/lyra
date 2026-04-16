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

from contextlib import ExitStack, contextmanager
from typing import Generator

from lyra_2._ext.imaginaire.utils.misc import timer


@contextmanager
def data_loader_init() -> Generator[None, None, None]:
    """
    Wrap the data loader initialization with multiple context managers used for telemetry and one logger.
    """
    contexts = [
        timer("init_data_loader"),
    ]
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in contexts]


@contextmanager
def model_init(set_barrier: bool = False) -> Generator[None, None, None]:
    """
    Wrap the instantiation of the model with multiple context managers used for telemetry and one logger.
    """
    contexts = [
        timer("init_model"),
    ]
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in contexts]


@contextmanager
def distributed_init() -> Generator[None, None, None]:
    """
    Wrap the distributed initialization, used for telemetry and timers
    """
    contexts = [
        timer("init_distributed"),
    ]
    with ExitStack() as stack:
        yield [stack.enter_context(cm) for cm in contexts]
