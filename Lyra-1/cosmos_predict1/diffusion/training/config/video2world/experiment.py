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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_predict1.diffusion.training.datasets.dataset_video import Dataset
from cosmos_predict1.diffusion.training.models.extend_model import FSDPExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model_peft import PEFTExtendDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.diffusion.training.utils.peft.lora_config import get_fa_ca_qv_lora_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

n_length = 15
num_frames = 8 * n_length + 1  # 121

# HDVILA example
example_video_dataset_hdvila = L(Dataset)(
    dataset_dir="datasets/hdvila",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_hdvila = L(DataLoader)(
    dataset=example_video_dataset_hdvila,
    sampler=L(get_sampler)(dataset=example_video_dataset_hdvila),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# Cosmos-NeMo-Assets example
example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# Cosmos-NeMo-Assets examples with more affordable GPUs setup (4 GPUs or 40GB VRAM)
n_length_4gpu_80gb = 15
num_frames_4gpu_80gb = 8 * n_length_4gpu_80gb + 1  # 121
example_video_dataset_cosmos_nemo_assets_4gpu_80gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_4gpu_80gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering the content aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_4gpu_80gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

n_length_8gpu_40gb = 3
num_frames_8gpu_40gb = 8 * n_length_8gpu_40gb + 1  # 25
example_video_dataset_cosmos_nemo_assets_8gpu_40gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_8gpu_40gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_8gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

n_length_4gpu_40gb = 3
num_frames_4gpu_40gb = 8 * n_length_4gpu_40gb + 1  # 25
example_video_dataset_cosmos_nemo_assets_4gpu_40gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_4gpu_40gb,
    video_size=(192, 192),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_4gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

# Cosmos-NeMo-Assets 480x848 example for lora
example_video_dataset_cosmos_nemo_assets_480_848 = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(480, 848),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_480_848 = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_480_848,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_480_848),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

dataloader_val_cosmos_nemo_assets_480_848 = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_480_848,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_480_848),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

video2world_7b_example_hdvila = LazyDict(
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
            group="diffusion_video2world",
            name="video2world_7b_example_hdvila",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
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
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_hdvila,
    )
)


video2world_7b_example_cosmos_nemo_assets = LazyDict(
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
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
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
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets,
    )
)

video2world_7b_example_cosmos_nemo_assets_4gpu_80gb = LazyDict(
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
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_80gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
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
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_80gb,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_80gb,
    )
)

video2world_7b_example_cosmos_nemo_assets_8gpu_40gb = LazyDict(
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
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_8gpu_40gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=2000,
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
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_8gpu_40gb,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_8gpu_40gb,
    )
)

video2world_7b_example_cosmos_nemo_assets_4gpu_40gb = LazyDict(
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
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_40gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=2000,
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
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                24,  # Latent height dim
                24,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_40gb,
                spatial_resolution="192",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_40gb,
    )
)

video2world_7b_lora_example_cosmos_nemo_assets = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_example_cosmos_nemo_assets",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=1000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=5000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
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
            context_parallel_size=4,
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_480_848,
        dataloader_val=dataloader_val_cosmos_nemo_assets_480_848,
    )
)


def register_experiments(cs):
    # Register the experiments
    for _item in [
        video2world_7b_example_hdvila,
        video2world_7b_example_cosmos_nemo_assets,
        video2world_7b_example_cosmos_nemo_assets_4gpu_80gb,
        video2world_7b_example_cosmos_nemo_assets_8gpu_40gb,
        video2world_7b_example_cosmos_nemo_assets_4gpu_40gb,
        video2world_7b_lora_example_cosmos_nemo_assets,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
