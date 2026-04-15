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
import logging
import math
import os
import random
import shutil
from pathlib import Path
import einops

import signal
import sys

import accelerate
import numpy as np
import imageio
import PIL
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig
)

import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs, AutocastKwargs
from accelerate import FullyShardedDataParallelPlugin, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from einops import rearrange, repeat

from transformers.utils import ContextManagers

from typing import Dict, Optional, Tuple, List, Union
from omegaconf import OmegaConf, ListConfig
from dataclasses import dataclass, asdict

import diffusers

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from peft import LoraConfig, get_peft_model_state_dict

from src.utils.random_state_utils import save_random_state
from src.models.recon.model_latent_recon import LatentRecon

from kiui.lpips import LPIPS
from fused_ssim import fused_ssim

from src.models.data import get_multi_dataloader
from src.models.utils.model import encode_latent_time_vae, encode_plucker_vae, repeat_time_spatially
from src.models.utils.cosmos_1_tokenizer import load_cosmos_1_tokenizer
from src.models.utils.render import get_plucker_embedding_and_rays
from src.models.utils.misc import dtype_map
from src.models.utils.model import encode_multi_view_video, load_vae, encode_video
from src.models.utils.loss import compute_loss
from src.models.utils.train import get_most_recent_checkpoint
import time

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.3")

logger = get_logger(__name__, log_level="INFO")

def prepare_config(
    config: Dict
):
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.local_rank:
        config.local_rank = env_local_rank

    # Set paths
    config.model_pipeline = config.get('model_pipeline', {})
    if config.model_pipeline.get('vae_path', None) is None:
        config.model_pipeline['vae_path'] = config.pretrained_model_name_or_path    
    config.model_pipeline['use_lora'] = config.model_pipeline.get('use_lora', False)

def load_model_weights(path_ckpt, transformer):
    path_ckpt_model = os.path.join(path_ckpt, 'pytorch_model', 'mp_rank_00_model_states.pt')
    model_state = torch.load(path_ckpt_model, map_location="cpu")
    model_state = {f'module.{k}': v for k, v in model_state['module'].items()}
    transformer.load_state_dict(model_state, strict=False)

def resume_from_ckpt(config, accelerator, transformer):
    global_step = 0
    first_epoch = 0
    loaded_accelerator = False
    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        is_latest_resume = False
        if config.resume_from_checkpoint_dir is not None:
            path = os.path.basename(config.resume_from_checkpoint)
            path_ckpt = os.path.join(config.resume_from_checkpoint_dir, config.resume_from_checkpoint)
        else:
            if config.resume_from_checkpoint != "latest":
                path = os.path.basename(config.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                path = get_most_recent_checkpoint(config.output_dir)
                if path is None:
                    # This must be our first time since no checkpoint was found
                    logger.warning(f"No latest resume checkpoint found, assuming this is our first training session!")
                else:
                    is_latest_resume = True
            if path is not None:
                path_ckpt = os.path.join(config.output_dir, path)
            else:
                path_ckpt = None
        if path_ckpt is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path_ckpt}")
            try:
                accelerator.load_state(path_ckpt) # will also resume the random seed states # type: ignore
                loaded_accelerator = True
            except Exception as e:
                # Only load model without optimizer
                print("Failed to load checkpoint: Try to only load model weights")
                try:
                    load_model_weights(path_ckpt, transformer)
                    print("Loaded only model weights")
                except:
                    logger.warning(f"Failed to load checkpoint: {e}")
                    if is_latest_resume:
                        logger.warning("Remove the broken checkpoint and exit.")
                        if accelerator.is_main_process:
                            # remove the checkpoint if it fails to load
                            if path.endswith("bkup"):
                                logger.warning("Debug NOT removing the broken checkpoint.")
                            else:
                                shutil.rmtree(path_ckpt) # type: ignore
                            
                    exit(1)
            
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = 0
    else:
        initial_global_step = 0
    return initial_global_step, global_step, first_epoch, loaded_accelerator

def main(
    config: Dict,
    wandb_run_name,
    wandb_group_name,
    app_start_time,
):
    prepare_config(config)
    logging_dir = os.path.join(config.output_dir, config.logging_dir) # type: ignore

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    find_unused_parameters = (
        (config.gradient_accumulation_steps > 1) and
        (config.model_pipeline.get('unet_trainable_modules', None) is not None)
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    autocast_kwargs = AutocastKwargs(cache_enabled=config.autocast_cache_enabled)

    if config.use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
            use_orig_params=True, # fucking useless True
        )
        fsdp_plugin.use_orig_params = True # Stupid stupid design in accelerate
        assert not config.use_ema, "FSDP does not support EMAModel yet, please consider DeepSpeed"
    else:
        fsdp_plugin = None
    
    if config.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            zero_stage=2,
            gradient_clipping=config.max_grad_norm
        )
        deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.batch_size
        if config.deepspeed_type is None:
            config.deepspeed_type = config.mixed_precision
        if config.deepspeed_type == 'fp16':
            deepspeed_plugin.deepspeed_config['fp16'] = {
                "enabled": 'auto',
                "auto_cast": True,
                "initial_scale_power": 16,
            }
        elif config.deepspeed_type == 'bf16':
            deepspeed_plugin.deepspeed_config['bf16'] = {
                "enabled": True,
            }
    else:
        deepspeed_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.log_with,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs, autocast_kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        # the seed across all processes to make sure that models initialized in the same way
        set_seed(config.seed) 
    else:
        print("Not setting a seed")

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    vae = None
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = load_vae(config.vae_backbone, config.vae_path)
    
    for k, v in config.items():
        if isinstance(v, list) or isinstance(v, ListConfig):
            config[k] = tuple(v)
    config = OmegaConf.structured(config)
    
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision
    transformer = LatentRecon(
        config,
    )
    # Use lpips loss
    if config.lambda_lpips > 0:
        lpips_img_size = config.img_size if not isinstance(config.img_size, int) else [config.img_size, config.img_size]
        lpips_loss_module = LPIPS(net='vgg')
        lpips_loss_module.requires_grad_(False)
        lpips_loss_module = lpips_loss_module.to(accelerator.device)
    else:
        lpips_loss_module = None
    if config.resume_pretrained_model_ckpt:
        # TODO don't do this if we are resuming latest
        logger.info(f"Loading pretrain ckpt from: {config.resume_pretrained_model_ckpt}")
        data = torch.load(config.resume_pretrained_model_ckpt)
        transformer.load_state_dict(data["module"])

    # Freeze vae and transformer
    transformer.train()
    transformer.requires_grad_(False)
    if config.set_transformer_dtype:
        for module in transformer.modules():
            module.to(accelerator.device, dtype=weight_dtype)
    modules_dtype = [vae]      
    for module in modules_dtype:
        if module is not None:
            module.requires_grad_(False)
            module.to(accelerator.device, dtype=weight_dtype)
    if config.compile_frozen_modules:
        vae.encode = torch.compile(vae.encode)

    # Add lora support
    lora_params = []
    if config.model_pipeline['use_lora']:
        transformer_lora_config = LoraConfig(
            r=config.model_pipeline.get('lora_rank', 64),
            lora_alpha=config.model_pipeline.get('lora_alpha', 64),
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)
        lora_params = [name for name, p in transformer.named_parameters() if p.requires_grad] 

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # NOTE: currently only save and load the transformer model
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                    
                    if config.model_pipeline['use_lora']:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        model.save_lora_weights(os.path.join(output_dir, "lora"), transformer_lora_layers=transformer_lora_layers_to_save)

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = GSLRMLatent.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        if accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb # type: ignore
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []
    param_names = []

    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    transformer_trainable_modules = config.model_pipeline.get('transformer_trainable_modules', None) # None means all trainable
    if transformer_trainable_modules is not None:
        
        for name, param in transformer.named_parameters():
            for module in transformer_trainable_modules:
                
                if module in name:
                    parameters_list.append(param)
                    param_names.append(name)
                    param.requires_grad = True
                    break
        
        for name, param in transformer.named_parameters():
            if name in lora_params and name not in param_names:
                parameters_list.append(param)
                param_names.append(name)
                param.requires_grad = True
    else:
        transformer.requires_grad_(True)
        parameters_list = transformer.parameters()
        param_names = [name for name, param in transformer.named_parameters()]

    # fsdp - prepare model in advance of optimizer creation
    if accelerator.distributed_type == DistributedType.FSDP:
        transformer = accelerator.prepare(transformer)
    logger.info("***** Parameters list *****")
    logger.info(f"{param_names}")
    optimizer = optimizer_cls(
        parameters_list,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    global_batch_size = config.batch_size * accelerator.num_processes

    overrode_max_train_steps = False

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes, # NOTE: accelerate iterates num_proc times per step, not a bug
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if accelerator.distributed_type == DistributedType.FSDP:
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        pop_keys = []
        for k, v in tracker_config.items():
            if v is not None and not isinstance(v, (int, float, str, bool, torch.Tensor)):
                pop_keys.append(k)
        for k in pop_keys: 
            tracker_config.pop(k)

        init_kwargs = {
            "wandb": {
                "name": wandb_run_name,
                "dir": config.output_dir,
                "group": wandb_group_name,
                "tags": ["cosmos_3dgs"],
                "resume": "auto",
            },
        }

        accelerator.init_trackers(config.experiment_name, config=tracker_config, init_kwargs=init_kwargs)


    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    initial_global_step, global_step, first_epoch, loaded_accelerator = resume_from_ckpt(config, accelerator, transformer)
    # Overwrite learning rate
    if config.lr_overwrite:
        print(f"Set new optimizer with learning rate {config.learning_rate}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.learning_rate
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes, # NOTE: accelerate iterates num_proc times per step, not a bug
            num_training_steps=config.max_train_steps * accelerator.num_processes,
        )

    # Initialize DataLoaders
    if (initial_global_step == 0 or not loaded_accelerator) and config.seed is not None: # reset the seed again, 
        set_seed(config.seed, device_specific=True) # differ in each process, for data loading
        print(f"Set seed to {config.seed}")
    
    # DataLoaders creation:
    wds_loader = True
    train_dataloader, test_dataloader = get_multi_dataloader(config, accelerator)

    def train_step(batch, train_loss, num_input_multi_views):
        threedgs_kwargs = config
        # Convert images to latent space
        if wds_loader:
            batch_keys = list(batch.keys())
            for k in batch_keys:
                if isinstance(batch[k], torch.Tensor):                       
                    batch[k] = batch[k].to(accelerator.device, non_blocking=True)
                    # 3DGS rendering with full precision
                    if k not in ['intrinsics_input', 'c2ws_input', 'cam_view', 'intrinsics']:
                        batch[k] = batch[k].to(weight_dtype)
        # Read data
        gt_images = batch['images_output']
        gt_depths = batch.get('depths_output', None)

        # Handle variable size in multi_views
        if 'num_input_multi_views' in batch:
            assert (batch['num_input_multi_views'][0] == batch['num_input_multi_views']).all(), f"Not supporting multi batch size for variable multi-view"
            num_input_multi_views = int(batch['num_input_multi_views'][0].item())
            batch['num_input_multi_views'] = num_input_multi_views

        # Encode video
        if 'rgb_latents' in batch:
            model_input = batch['rgb_latents'].to(weight_dtype)
        else:
            video = batch['images_input_vae'].to(weight_dtype)
            if threedgs_kwargs.use_rgb_decoder:
                model_input = video
            else:
                # Encode each multi-view video independently
                model_input = encode_multi_view_video(vae, video, num_input_multi_views, config.vae_backbone)
        batch['images_input_embed'] = model_input

        # Compute plucker with GPU for speed
        if threedgs_kwargs.get('compute_plucker_cuda', True):
            batch['plucker_embedding'], batch['rays_os'], batch['rays_ds'] = get_plucker_embedding_and_rays(
                batch['intrinsics_input'],
                batch['c2ws_input'],
                threedgs_kwargs.img_size,
                threedgs_kwargs.patch_size_out_factor,
                batch['flip_flag'],
                get_batch_index=False,
                dtype=dtype_map[threedgs_kwargs.compute_plucker_dtype],
                out_dtype=weight_dtype
                )

        # Encode time and plucker
        if threedgs_kwargs.get('use_time_embedding', False) and threedgs_kwargs.get('time_embedding_vae', False):
            batch = encode_latent_time_vae(batch, lambda x: encode_video(vae, x, config.vae_backbone), threedgs_kwargs.img_size)
        if threedgs_kwargs.get('plucker_embedding_vae', False):
            batch = encode_plucker_vae(batch, lambda x: encode_multi_view_video(vae, x, num_input_multi_views, config.vae_backbone))
        
        # Main model pass
        model_output = transformer(batch)

        # Compute losses
        pred_images = model_output['images_pred'].to(gt_images.dtype)
        pred_depths = model_output['depths_pred'].to(gt_images.dtype)
        pred_opacity = model_output['opacity_pred']
        train_loss, loss = compute_loss(accelerator, train_loss, pred_images, gt_images, pred_depths, gt_depths, pred_opacity, config, lpips_loss_module, lpips_img_size)
        
        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)
            # double check the nan/inf, sometime doesn't work in DDP for no reason
            if optimizer.scaler is not None:
                optimizer.scaler._check_inf_per_device(optimizer.optimizer)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        return train_loss

    # Train!
    total_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info(f"  Output dir: {config.output_dir}")
    
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    ) # type: ignore
    break_loop = False
    while True:
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                train_loss = train_step(batch, train_loss, config.num_input_multi_views)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }, step=global_step)
                train_loss = 0.0

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # don't count "permanent_checkpointing_steps" checkpoints towards limit
                            checkpoints = [ckpt for ckpt in checkpoints if 
                                (int(ckpt.split("-")[1]) % config.permanent_checkpointing_steps) != 0]
                            print(checkpoints)

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint) # type: ignore
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}") # type: ignore
                    if accelerator.is_main_process or accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if config.save_multi_random_states and not accelerator.is_main_process:
                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}") # type: ignore
                        os.makedirs(save_path, exist_ok=True)
                        save_random_state(save_path, accelerator.process_index)

                if config.job_stop_steps is not None and global_step % config.job_stop_steps == 0:
                    logger.info('Reach Job Stop Steps')
                    break_loop = True
                    break

            logs = {"step_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0], "dir": config.output_dir}
            if optimizer.step_was_skipped:
                logs["overflow"] = 1
                logs["scaler"] = optimizer.scaler._scale.item() if optimizer.scaler is not None else 1
                logger.warning(f"Gradient overflow.  Skipping step {global_step}, scaler {logs['scaler']}")
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps: # type: ignore
                logger.info('Reach Max Train Steps')
                break_loop = True
                break

        if break_loop:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()        
    accelerator.end_training()


if __name__ == "__main__":
    app_start_time = time.time_ns() / 1_000_000
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--config_default', type=str, default='configs/training/default.yaml')
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Name of run for wandb",)
    parser.add_argument('--wandb_group_name', type=str, default=None)
    args, unknown = parser.parse_known_args()
    schema = OmegaConf.load(args.config_default)
    config = OmegaConf.load(args.config)
    missing_keys = set(config.keys()) - set(schema.keys())
    for key in missing_keys:
        OmegaConf.update(schema, key, None, force_add=True)
    config = OmegaConf.merge(schema, config)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)

    try:
        main(config, args.wandb_run_name, args.wandb_group_name, app_start_time)
    finally:
        wandb.finish()