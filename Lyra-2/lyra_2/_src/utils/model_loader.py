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


import importlib
import os

import torch
import torch.distributed.checkpoint as dcp

from lyra_2._ext.imaginaire.checkpointer.dcp import DefaultLoadPlanner, DistributedCheckpointer, ModelWrapper
from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override
from lyra_2._ext.imaginaire.utils.easy_io import easy_io


def load_model_from_checkpoint(
    experiment_name,
    checkpoint_path,
    config_file="lyra_2/_src/configs/t2v_wan/config.py",
    enable_fsdp=False,
    instantiate_ema=True,
    load_ema_to_reg=False,
    seed=0,
    experiment_opts: list[str] = [],
    strict=True,
):
    """
    experiment_name: experiment name
    checkpoint_dir: path to iteration_model
    config_file: config file path
    enable_fsdp: enable fsdp
    seed: random seed
    """
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if instantiate_ema is False and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    misc.set_random_seed(seed=seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    if not enable_fsdp:
        # disable fsdp
        config.model.config.fsdp_shard_size = 1
    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()
        # Convert the model parameters to bf16
        model.on_train_start()

    if checkpoint_path.endswith(".pth"):
        log.info(f"Loading model from consolidated checkpoint {checkpoint_path}")

        model.load_state_dict(easy_io.load(checkpoint_path), strict=strict)
    else:
        log.info(f"Loading model from dcp checkpoint {checkpoint_path}")

        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        cur_key_ckpt_full_path = os.path.join(checkpoint_path, "model")
        storage_reader = checkpointer.get_storage_reader(cur_key_ckpt_full_path)

        _model_wrapper = ModelWrapper(model, load_ema_to_reg=load_ema_to_reg)
        _state_dict = _model_wrapper.state_dict()
        dcp.load(
            _state_dict,
            storage_reader=storage_reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        _model_wrapper.load_state_dict(_state_dict)

    torch.cuda.empty_cache()

    return model, config
