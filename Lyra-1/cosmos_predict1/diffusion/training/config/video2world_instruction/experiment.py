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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_predict1.diffusion.training.datasets.dataset_3D import Dataset_3D
from cosmos_predict1.diffusion.training.models.extend_model import FSDPExtendDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.utils import log
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

cs = ConfigStore.instance()
base_path = "datasets/bridge/"
train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=57,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    load_action=False,
    load_t5_embeddings=True,
)

bridge_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=57,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
    load_action=False,
    load_t5_embeddings=True,
)


dataloader_train = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)
dataloader_val = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)


video2world_instruction_bridge_57frames = LazyDict(  # This experiment is used to verify the expanded config is the same as BASE002_101_512N_FSDP_LR-143_VideoImage_1-1
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world_instruction",
            name="video2world_instruction_bridge_57frames",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=500,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2_000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            # Use 16x8x32x40 latent shape for training
            latent_shape=[
                16,  # Latent channel dim
                8,  # Latent temporal dim
                32,  # Latent height dim
                40,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=False,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            # Use Image VAE for training
            vae=dict(pixel_chunk_duration=57),
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    first_random_n_num_condition_t_max=1,
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                )
            ),
        ),
        # using the video extend model for training
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps~(when resume from 310000)
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
    )
)


def register_experiments(cs):
    # Register the experiments
    for _item in [
        video2world_instruction_bridge_57frames,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
