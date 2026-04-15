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
import os
from typing import Any, Set

import torch

from cosmos_predict1.checkpointer.ddp import Checkpointer as DDPCheckpointer
from cosmos_predict1.utils import distributed, log
from cosmos_predict1.utils.model import Model


class Checkpointer(DDPCheckpointer):
    """
    Checkpointer class for PEFT in distributed training. This class is similar to the DDP checkpointer,
    with the exception that the `broadcast_via_filesystem` functionality is not supported, and it supports
    loading pre-trained model without any postfix.

    Note:
    - Fully Sharded Data Parallelism (FSDP) is not supported by this checkpointer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.broadcast_via_filesystem:
            raise ValueError("self.broadcast_via_filesystem=False is not implemented for PEFT checkpointer.")

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: Model) -> str:
        """
        Overwrite the `add_type_postfix_to_checkpoint_path` function of the base class (DDP checkpointer)
        to load pre-trained model without any postfix.
        """
        checkpoint_path = super().add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
        checkpoint_path = checkpoint_path.replace("model_model.pt", "model.pt")
        return checkpoint_path

    def load_broadcast_state_dict(self, checkpoint_path: str, model: Model, resume_keys: Set) -> dict[str, Any]:
        """
        Load state_dict and broadcast for PEFT checkpointer.

        This function is identical to the `load_broadcast_state_dict` function of the base class (DDP checkpointer),
        with the exception that the `broadcast_via_filesystem` functionality is not supported.

        Args:
            checkpoint_path (str): The base path of the checkpoint.
            model (Model): The model being loaded.
            resume_keys (Set): Set of keys to resume from the checkpoint.

        Returns:
            dict[str, Any]: A dictionary containing the loaded state for each resumed key.
        """
        state_dict = {}
        sorted_resume_keys = sorted(resume_keys)
        # Step 1: Download checkpoints for every GPU of DDP-rank 0 and CP-rank 0.
        if self.rank_dp_w_cp == 0:
            for key in sorted_resume_keys:
                _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                if os.path.exists(local_cache_path):
                    # If the local checkpoint exists, we can directly load it
                    self.print(f"Checkpoint is already in local cache: {local_cache_path}. Loading...")
                    _state_dict = torch.load(
                        local_cache_path, map_location=lambda storage, loc: storage, weights_only=False
                    )
                else:
                    # Pre-trained model is not in local cache, so we need to load it from the checkpoint path
                    self.print(f"Loading checkpoint from: {_ckpt_path}")
                    _state_dict = torch.load(_ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
                state_dict[key] = _state_dict

        # Ensure all ranks wait for the download to complete
        distributed.barrier()

        # Step 2: Broadcast checkpoint data
        log.info(
            "Start broadcasting checkpoint from the source rank to all other ranks in the same DDP group.",
            rank0_only=True,
        )
        for key in sorted_resume_keys:
            if self.broadcast_via_filesystem:
                # Load the checkpoint from the local filesystem for other ranks
                if self.rank_dp_w_cp != 0:
                    _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                    local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                    if os.path.exists(local_cache_path):
                        self.print(f"Loading checkpoint from: {local_cache_path}")
                        state_dict[key] = torch.load(
                            local_cache_path, map_location=lambda storage, loc: storage, weights_only=False
                        )
                    else:
                        self.print(f"Loading checkpoint from: {_ckpt_path}")
                        state_dict[key] = torch.load(
                            _ckpt_path, map_location=lambda storage, loc: storage, weights_only=False
                        )

            else:
                raise ValueError("self.broadcast_via_filesystem=False is not implemented for PEFT checkpointer.")

        return state_dict
