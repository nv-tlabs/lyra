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

import copy
from typing import Dict

from hydra.core.config_store import ConfigStore

from cosmos_predict1.checkpointer.peft_checkpointer import Checkpointer as PEFTCheckpointer
from cosmos_predict1.diffusion.checkpointers.ema_fsdp_checkpointer import CheckpointConfig, FSDPCheckpointer
from cosmos_predict1.diffusion.conditioner import VideoExtendConditioner
from cosmos_predict1.diffusion.config.base.conditioner import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from cosmos_predict1.diffusion.training.conditioner import VideoConditioner
from cosmos_predict1.diffusion.training.config.base.optim import FusedAdamWConfig, LambdaLinearSchedulerConfig
from cosmos_predict1.diffusion.training.config.base.vae import get_cosmos_tokenizer_comp8x8x8
from cosmos_predict1.diffusion.training.config.text2world.experiment import register_experiments
from cosmos_predict1.diffusion.training.networks.general_dit import GeneralDIT
from cosmos_predict1.utils.ema import PowerEMATracker
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

FSDP_CHECKPOINTER: Dict[str, str] = L(FSDPCheckpointer)()
PEFT_CHECKPOINTER: Dict[str, str] = L(PEFTCheckpointer)()
VideoExtendConditionerConfig: LazyDict = L(VideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
)


VideoConditionerFpsSizePaddingConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
)


def register_conditioner(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_cond",
        node=VideoExtendConditionerConfig,
    )

    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingConfig,
    )


def register_checkpoint_credential(cs):
    CHECKPOINT_LOCAL = CheckpointConfig(
        save_iter=1000,
        load_path="",
        load_training_state=False,
        strict_resume=True,
    )

    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)


def register_checkpointer(cs):
    cs.store(group="ckpt_klass", package="checkpoint.type", name="fsdp", node=FSDP_CHECKPOINTER)
    cs.store(group="ckpt_klass", package="checkpoint.type", name="peft", node=PEFT_CHECKPOINTER)


FADITV2Config: LazyDict = L(GeneralDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    block_config="FA-CA-MLP",
    spatial_attn_win_size=1,
    temporal_attn_win_size=1,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    block_x_format="THWBD",
    additional_timestamp_channels=None,
    affline_emb_norm=True,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    legacy_patch_emb=False,
)

FADITV2_14B_Config = copy.deepcopy(FADITV2Config)
FADITV2_14B_Config.model_channels = 5120
FADITV2_14B_Config.num_heads = 40
FADITV2_14B_Config.num_blocks = 36


def register_net(cs):
    cs.store(group="net", package="model.net", name="faditv2_7b", node=FADITV2Config)
    cs.store(group="net", package="model.net", name="faditv2_14b", node=FADITV2_14B_Config)


def register_vae(cs):
    cs.store(
        group="vae",
        package="model.vae",
        name="cosmos_diffusion_tokenizer_comp8x8x8",
        node=get_cosmos_tokenizer_comp8x8x8(resolution="720", chunk_duration=121),
    )


PowerEMAConfig: LazyDict = L(PowerEMATracker.initialize_multi_rank_ema)(
    model=PLACEHOLDER, enabled=True, rate=0.10, num=3
)


def register_ema(cs):
    cs.store(group="ema", package="model.ema", name="power", node=PowerEMAConfig)


def register_optimizer(cs):
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)


def register_scheduler(cs):
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)


def register_configs():
    cs = ConfigStore.instance()

    register_optimizer(cs)
    register_scheduler(cs)

    register_net(cs)
    register_conditioner(cs)
    register_vae(cs)

    register_ema(cs)

    register_checkpoint_credential(cs)
    register_checkpointer(cs)

    register_experiments(cs)
