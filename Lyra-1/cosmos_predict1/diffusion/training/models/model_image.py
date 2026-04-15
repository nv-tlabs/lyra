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

import functools
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.module.blocks import FourierFeatures
from cosmos_predict1.diffusion.module.pretrained_vae import BaseVAE
from cosmos_predict1.diffusion.modules.denoiser_scaling import EDMScaling
from cosmos_predict1.diffusion.modules.res_sampler import COMMON_SOLVER_OPTIONS, Sampler
from cosmos_predict1.diffusion.training.functional.loss import create_per_sample_loss_mask
from cosmos_predict1.diffusion.training.utils.fsdp_helper import apply_fsdp_checkpointing, hsdp_device_mesh
from cosmos_predict1.diffusion.training.utils.optim_instantiate import get_base_scheduler
from cosmos_predict1.diffusion.types import DenoisePrediction
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.ema import FastEmaModelUpdater
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_predict1.utils.model import Model


@dataclass
class CosmosCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class DiffusionModel(Model):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # how many sample have been processed
        self.sample_counter = 0
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.warning(f"DiffusionModel: precision {self.precision}")
        # Timer passed to network to detect slow ranks.
        # 1. set data keys and data information
        self.sigma_data = config.sigma_data
        self.state_shape = list(config.latent_shape)
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.sde = lazy_instantiate(config.sde)
        self.sampler = Sampler()
        self.scaling = EDMScaling(self.sigma_data)

        # 3. vae
        with misc.timer("DiffusionModel: set_up_vae"):
            self.vae: BaseVAE = lazy_instantiate(config.vae)
            assert (
                self.vae.latent_ch == self.state_shape[0]
            ), f"latent_ch {self.vae.latent_ch} != state_shape {self.state_shape[0]}"

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_masking: Optional[Dict] = config.loss_masking
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        log.critical(f"Enable loss masking: {config.loss_mask_enabled}")

        # 5. diffusion neural networks part
        self.set_up_model()

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key

    def build_model(self) -> torch.nn.ModuleDict:
        config = self.config
        net = lazy_instantiate(config.net)
        conditioner = lazy_instantiate(config.conditioner)
        logvar = torch.nn.Sequential(
            FourierFeatures(num_channels=128, normalize=True), torch.nn.Linear(128, 1, bias=False)
        )

        return torch.nn.ModuleDict(
            {
                "net": net,
                "conditioner": conditioner,
                "logvar": logvar,
            }
        )

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        self.model = self.build_model()
        if config.ema.enabled:
            with misc.timer("DiffusionModel: instantiate ema"):
                config.ema.model = self.model
                self.model_ema = lazy_instantiate(config.ema)
                config.ema.model = None
        else:
            self.model_ema = None

    @property
    def net(self):
        return self.model.net

    @property
    def conditioner(self):
        return self.model.conditioner

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the model_ema
        """
        if self.config.ema.enabled:
            self.model_ema.update_average(self.model, iteration)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.model_ema.to(dtype=torch.float32)
        if hasattr(self.vae, "reset_dtype"):
            self.vae.reset_dtype()
        self.model = self.model.to(memory_format=memory_format, **self.tensor_kwargs)

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning(
                    "torch.compile in Pytorch version older than 2.3 doesn't work well with activation checkpointing.\n"
                    "It's very likely there will be no significant speedup from torch.compile.\n"
                    "Please use at least 24.04 Pytorch container."
                )
            # Increasing cache size. It's required because of the model size and dynamic input shapes resulting in
            # multiple different triton kernels. For 28 TransformerBlocks, the cache limit of 256 should be enough for
            # up to 9 different input shapes, as 28*9 < 256. If you have more Blocks or input shapes, and you observe
            # graph breaks at each Block (detectable with torch._dynamo.explain) or warnings about
            # exceeding cache limit, you may want to increase this size.
            # Starting with 24.05 Pytorch container, the default value is 256 anyway.
            # You can read more about it in the comments in Pytorch source code under path torch/_dynamo/cache_size.py.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape. It incurs higher compilation costs
            # at initial iterations, but can result in more specialized and efficient kernels.
            # dynamic=True currently throws errors in pytorch 2.3.
            self.model.net = torch.compile(self.model.net, dynamic=False, disable=not self.config.use_torch_compile)

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: CosmosCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0_from_data_batch: raw image/video
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean, std = self.sde.marginal_prob(x0, sigma)
        # Generate noisy observations
        xt = mean + batch_mul(std, epsilon)  # corrupted data
        # make prediction
        model_pred = self.denoise(xt, sigma, condition)
        # loss weights for different noise levels
        weights_per_sigma = self.get_per_sigma_loss_weights(sigma=sigma)
        # extra weight for each sample, for example, aesthetic weight, camera weight
        weights_per_sample = self.get_per_sample_weight(data_batch, x0_from_data_batch.shape[0])
        # extra loss mask for each sample, for example, human faces, hands
        loss_mask_per_sample = self.get_per_sample_loss_mask(data_batch, x0_from_data_batch.shape, x0.shape)
        pred_mse = (x0 - model_pred.x0) ** 2 * loss_mask_per_sample
        edm_loss = batch_mul(pred_mse, weights_per_sigma * weights_per_sample)
        if self.config.loss_add_logvar:
            kendall_loss = batch_mul(edm_loss, torch.exp(-model_pred.logvar).view(-1)).flatten(
                start_dim=1
            ) + model_pred.logvar.view(-1, 1)
        else:
            kendall_loss = edm_loss.flatten(start_dim=1)
        output_batch = {
            "x0": x0,
            "xt": xt,
            "sigma": sigma,
            "weights_per_sigma": weights_per_sigma,
            "weights_per_sample": weights_per_sample,
            "loss_mask_per_sample": loss_mask_per_sample,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse.mean(),
            "edm_loss": edm_loss.mean(),
        }
        return output_batch, kendall_loss, pred_mse, edm_loss

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the diffusion model.

        This method is responsible for executing one iteration of the model's training. It involves:
        1. Adding noise to the input data using the SDE process.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        x0_from_data_batch, x0, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma, epsilon = self.draw_training_sigma_and_epsilon(x0.size(), condition)

        output_batch, kendall_loss, pred_mse, edm_loss = self.compute_loss_with_epsilon_and_sigma(
            data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

    def denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: CosmosCondition) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (CosmosCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
        """

        if getattr(self.config, "use_dummy_temporal_dim", False):
            # When using video DiT model for image, we need to use a dummy temporal dimension.
            xt = xt.unsqueeze(2)

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)
        # get precondition for the network
        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        # forward pass through the network
        net_output = self.net(
            x=batch_mul(c_in, xt),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps=c_noise,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        )

        logvar = self.model.logvar(c_noise)
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)

        # get noise prediction based on sde
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        if getattr(self.config, "use_dummy_temporal_dim", False):
            x0_pred = x0_pred.squeeze(2)
            eps_pred = eps_pred.squeeze(2)

        return DenoisePrediction(x0_pred, eps_pred, logvar)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.sigma_data)

    def draw_training_sigma_and_epsilon(self, x0_size: int, condition: Any) -> torch.Tensor:
        del condition
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, **self.tensor_kwargs)
        return self.sde.sample_t(batch_size).to(**self.tensor_kwargs), epsilon

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, CosmosCondition]:
        """
        processing data batch draw from data loader and return data and condition that used for denoising task

        Returns:
            raw_state (tensor): the image / video data that feed to vae
            latent_state (tensor): nosie-free state, the vae latent state
            condition (CosmosCondition): condition information for conditional generation. Generated from conditioner
        """
        raw_state = data_batch[self.input_data_key]
        latent_state = self.encode(raw_state)
        condition = self.conditioner(data_batch)
        return raw_state, latent_state, condition

    def get_per_sample_weight(self, data_batch: dict[str, torch.Tensor], batch_size: int):
        r"""
        extra weight for each sample, for example, aesthetic weight
        Args:
            data_batch: raw data batch draw from the training data loader.
            batch_size: int, the batch size of the input data
        """
        aesthetic_cfg = getattr(self.config, "aesthetic_finetuning", None)
        if (aesthetic_cfg is not None) and getattr(aesthetic_cfg, "enabled", False):
            sample_weight = data_batch["aesthetic_weight"]
        else:
            sample_weight = torch.ones(batch_size, **self.tensor_kwargs)

        camera_cfg = getattr(self.config, "camera_sample_weight", None)
        if (camera_cfg is not None) and getattr(camera_cfg, "enabled", False):
            sample_weight *= 1 + (data_batch["camera_attributes"][:, 1:].sum(dim=1) != 0) * (camera_cfg.weight - 1)
        return sample_weight

    def get_per_sample_loss_mask(self, data_batch, raw_x_shape, latent_x_shape):
        """
        extra loss mask for each sample, for example, human faces, hands.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            raw_x_shape (tuple): shape of the input data. We need the raw_x_shape for necessary resize operation.
            latent_x_shape (tuple): shape of the latent data
        """
        if self.config.loss_mask_enabled:
            raw_x_shape = [raw_x_shape[0], 1, *raw_x_shape[2:]]
            weights = create_per_sample_loss_mask(
                self.loss_masking, data_batch, raw_x_shape, torch.get_default_dtype(), "cuda"
            )
            return F.interpolate(weights, size=latent_x_shape[2:], mode="bilinear")

        return 1.0

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def generate_samples(self, batch_size: int, condition: CosmosCondition) -> torch.Tensor:
        """
        Generate samples with given condition. It is WITHOUT classifier-free-guidance.

        Args:
            batch_size (int):
            condition (CosmosCondition): condition information generated from self.conditioner
        """
        x_sigma_max = torch.randn(batch_size, *self.state_shape, **self.tensor_kwargs) * self.sde.sigma_max

        def x0_fn(x, t):
            return self.denoise(x, t, condition).x0  # ODE function

        return self.sampler(x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max)

    def generate_cfg_samples(
        self, batch_size: int, condition: CosmosCondition, uncondition: CosmosCondition, guidance=1.5
    ) -> torch.Tensor:
        """
        Generate samples with with classifier-free-guidance.

        Args:
            batch_size (int):
            condition (CosmosCondition): condition information generated from self.conditioner
            uncondition (CosmosCondition): uncondition information, possibily generated from self.conditioner
        """
        x_sigma_max = torch.randn(batch_size, *self.state_shape, **self.tensor_kwargs) * self.sde.sigma_max

        def x0_fn(x, t):
            cond_x0 = self.denoise(x, t, condition).x0
            uncond_x0 = self.denoise(x, t, uncondition).x0
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return self.sampler(x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max)

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Optional[Tuple] = None,
        n_sample: Optional[int] = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
    ) -> torch.Tensor:
        """
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to self.state_shape if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        batch_size = n_sample or data_batch[self.input_data_key].shape[0]
        state_shape = state_shape or self.state_shape
        x_sigma_max = (
            misc.arch_invariant_rand(
                (batch_size,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.sde.sigma_max
        )
        return self.sampler(
            x0_fn, x_sigma_max, sigma_max=self.sde.sigma_max, num_steps=num_steps, solver_option=solver_option
        )

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Current code does nothing.
        """
        return {}, torch.tensor(0).to(**self.tensor_kwargs)

    @torch.no_grad()
    def forward(self, xt, t, condition: CosmosCondition):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (CosmosCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred) and optional confidence (logvar).
        """
        return self.denoise(xt, t, condition)

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = lazy_instantiate(optimizer_config, model=self.model)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the current state of the model as a dictionary.

        Returns:
            Dict: The current state of the model as a dictionary.
        """
        return {
            "model": self.model.state_dict(),
            "ema": self.model_ema.state_dict() if self.config.ema.enabled else None,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                     those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                     matching keys one-by-one. This is typically used when loading parts of state dicts
                                     or using customized loading procedures. Defaults to False.
        """
        if strict:
            # the converted tpsp checkpoint has "ema" and it is None
            if self.config.ema.enabled and state_dict["ema"] is not None:
                ema_results: _IncompatibleKeys = self.model_ema.load_state_dict(
                    state_dict["ema"], strict=strict, assign=assign
                )
            reg_results: _IncompatibleKeys = self.model.load_state_dict(
                state_dict["model"], strict=strict, assign=assign
            )
            if self.config.ema.enabled and state_dict["ema"] is not None:
                return _IncompatibleKeys(
                    ema_results.missing_keys + reg_results.missing_keys,
                    ema_results.unexpected_keys + reg_results.unexpected_keys,
                )
            return reg_results
        else:
            from cosmos_predict1.diffusion.training.utils.checkpointer import non_strict_load_model

            log.critical("load model in non-strict mode")
            if "model" in state_dict:
                log.critical(non_strict_load_model(self.model, state_dict["model"]), rank0_only=False)
            else:
                log.critical(non_strict_load_model(self.model, state_dict), rank0_only=False)
            if self.config.ema.enabled and "ema" in state_dict and state_dict["ema"] is not None:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.model_ema, state_dict["ema"]), rank0_only=False)

    def get_ckpt_postfix(self) -> Tuple[str, int, int]:
        """Get the checkpoint file postfix.

        Args:
            iteration (int): The current iteration number.

        Returns:
            postfix (str): The postfix of the checkpoint file.
            rank_to_save ema (int), we will not save each ema model in each rank, \
                ema model with same rate will be saved once
            total_ema_num (int)
        """
        total_ema_num = min(self.config.ema.num, distributed.get_world_size())
        rank = distributed.get_rank()
        if rank == 0:
            return "", 0, total_ema_num
        if self.config.ema.enabled:
            if rank < self.config.ema.num:
                return f"_RANK{rank}", rank, total_ema_num
        return "", 0, total_ema_num  # use rank 0 to save the checkpoint

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            self.model_ema.cache(self.model.parameters(), is_cpu=is_cpu)
            self.model_ema.copy_to(self.model)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")


T = TypeVar("T", bound=DiffusionModel)


def diffusion_fsdp_class_decorator(base_class: Type[T]) -> Type[T]:
    """
    Decorator for the FSDP class for the diffusion model, which handles the FSDP specific logic for the diffusion model.
    """

    class FSDPClass(base_class):
        """
        Handle FSDP specific logic for the diffusion model. Including:
        - FSDP model initialization
        - FSDP model / optimizer save and loading
        - Different from the original DiffusionModel, the impl of multi-rank EMA is a bit hacky. \
            We need to make sure sharded model weights for EMA and regular model are the same.
        """

        def __init__(self, config, fsdp_checkpointer: Any):
            self.fsdp_checkpointer = fsdp_checkpointer
            super().__init__(config)

        def set_up_model(self):
            config = self.config

            # 1. build FSDP sharding strategy and device_mesh
            strategy = {
                "full": ShardingStrategy.FULL_SHARD,
                "hybrid": ShardingStrategy.HYBRID_SHARD,
            }[config.fsdp.sharding_strategy]
            log.critical(f"Using {strategy} sharding strategy for FSDP")

            if config.fsdp.sharding_strategy == "hybrid":
                sharding_group_size = getattr(config.fsdp, "sharding_group_size", 8)
                device_mesh = hsdp_device_mesh(
                    sharding_group_size=sharding_group_size,
                )
                shard_group = device_mesh.get_group(mesh_dim="shard")
                replicate_group = device_mesh.get_group(mesh_dim="replicate")
                fsdp_process_group = (shard_group, replicate_group)
            else:
                device_mesh = hsdp_device_mesh(
                    sharding_group_size=distributed.get_world_size(),
                )
                shard_group = device_mesh.get_group(mesh_dim="shard")
                fsdp_process_group = shard_group

            # We piggyback the `device_mesh` to megatron-core's `parallel_state` for global access.
            # This is not megatron-core's original API.
            parallel_state.fsdp_device_mesh = device_mesh

            def get_wrap_policy(_model):
                if not hasattr(_model.net, "fsdp_wrap_block_cls"):
                    raise ValueError(
                        "Networks does not have fsdp_wrap_block_cls attribute, please check the net definition"
                    )
                fsdp_blocks_cls = _model.net.fsdp_wrap_block_cls
                fsdp_blocks_cls = (
                    list(fsdp_blocks_cls) if isinstance(fsdp_blocks_cls, (list, tuple, set)) else [fsdp_blocks_cls]
                )
                log.critical(f"Using FSDP blocks {fsdp_blocks_cls}")

                log.critical(f"Using wrap policy {config.fsdp.policy}")
                if config.fsdp.policy == "size":
                    min_num_params = getattr(config.fsdp, "min_num_params", 100)
                    log.critical(f"Using {min_num_params} as the minimum number of parameters for auto-wrap policy")
                    wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
                else:
                    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

                    wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=set(fsdp_blocks_cls),
                    )
                return wrap_policy

            # 2. build naive pytorch model and load weights if exists
            replica_idx, shard_idx = device_mesh.get_coordinate()
            # 2.1 handle ema case first, since float32 is more expensive
            if config.ema.enabled:
                with misc.timer("Creating PyTorch model and loading weights for ema"):
                    model_ema = self.build_model().float()
                    model_ema.cuda().eval().requires_grad_(False)
                    if distributed.get_rank() == 0:
                        # only load model in rank0 to reduce network traffic
                        self.fsdp_checkpointer.load_model_during_init(model_ema, is_ema=True)
                    # sync ema model weights from rank0
                    with misc.timer("Sync model states for EMA model"):
                        #! this is IMPORTANT, see the following comment about regular model for details
                        #! we broadcast the ema model first, since it is fp32 and costs more memory
                        distributed.sync_model_states(model_ema, device_mesh.get_group(mesh_dim="shard"))
                        torch.cuda.empty_cache()
                        distributed.sync_model_states(model_ema, device_mesh.get_group(mesh_dim="replicate"))
                        torch.cuda.empty_cache()
                    # for ema model with dfiferent rate, we download the model when necessary
                    if shard_idx == 0 and replica_idx > 0 and replica_idx < config.ema.num:
                        print("loading ema model in rank", replica_idx)
                        self.fsdp_checkpointer.load_model_during_init(
                            model_ema,
                            is_ema=True,
                            ema_id=replica_idx,
                        )
                        print("finish loading ema model in rank", replica_idx)
                # 2.1.2 create FSDP model for ema model
                with misc.timer("Creating FSDP model for EMA model"):
                    self.model_ema = FSDP(
                        model_ema,
                        sync_module_states=True,  # it can reduce network traffic by only loading model in rank0 and sync
                        process_group=device_mesh.get_group(mesh_dim=1),
                        sharding_strategy=ShardingStrategy.FULL_SHARD,
                        auto_wrap_policy=get_wrap_policy(model_ema),
                        device_id=torch.cuda.current_device(),
                        limit_all_gathers=True,
                    )

                # extra ema model upate logic to the model
                self.model_ema_worker = FastEmaModelUpdater()
                s = 0.1
                replica_idx, shard_idx = device_mesh.get_coordinate()
                divider = 2**replica_idx if replica_idx < config.ema.num else 1
                if replica_idx < config.ema.num:
                    if shard_idx == 0:
                        print(f"EMA: rank {replica_idx}, rate {config.ema.rate / divider}")
                s = config.ema.rate / divider
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                torch.cuda.empty_cache()

            # 2.2 handle regular model
            with misc.timer("Creating PyTorch model and loading weights for regular model"):
                model = self.build_model().cuda().to(**self.tensor_kwargs)

                if distributed.get_rank() == 0:
                    # only load model in rank0 to reduce network traffic and sync later
                    self.fsdp_checkpointer.load_model_during_init(model, is_ema=False)

                #! overwrite the forward method so that it will invoke the FSDP-specific pre- and post-forward sharding logic
                model.forward = super().training_step
                #! this is IMPORTANT, though following two lines are identical to sync_module_states=True in FSDP
                #! we do it twice so that following line can warm up and avoid OOM in aws 128+ nodes settings
                #! qsh hypothesize that it is due to overhead of initialization of nccl network communication;
                #! without it, peak mem : reg_model + ema_model + FSDP overhead + nccl communication initialization overhead
                #! with it, peak men: reg_model + ema_model + FSDP overhead
                #! it is tricky, but it works!
                with misc.timer("Sync model states for regular model"):
                    distributed.sync_model_states(model, device_mesh.get_group(mesh_dim="shard"))
                    torch.cuda.empty_cache()
                    distributed.sync_model_states(model, device_mesh.get_group(mesh_dim="replicate"))
                    torch.cuda.empty_cache()

                with misc.timer("Creating FSDP model"):
                    self.model = FSDP(
                        model.to(**self.tensor_kwargs),
                        sync_module_states=True,  # it can reduce network traffic by only loading model in rank0 and sync
                        sharding_strategy=strategy,
                        auto_wrap_policy=get_wrap_policy(model),
                        process_group=fsdp_process_group,
                        limit_all_gathers=True,
                    )

                    if self.config.fsdp.checkpoint:
                        fsdp_blocks_cls = model.net.fsdp_wrap_block_cls
                        fsdp_blocks_cls = (
                            list(fsdp_blocks_cls)
                            if isinstance(fsdp_blocks_cls, (list, tuple, set))
                            else [fsdp_blocks_cls]
                        )
                        log.critical(f"Applying FSDP checkpointing with FSDP blocks: {fsdp_blocks_cls}")
                        apply_fsdp_checkpointing(self.model, list_block_cls=fsdp_blocks_cls)

            torch.cuda.empty_cache()

        def on_before_zero_grad(
            self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
        ) -> None:
            del scheduler, optimizer

            if self.config.ema.enabled:
                # calculate beta for EMA update
                if iteration == 0:
                    beta = 0.0
                else:
                    i = iteration + 1
                    beta = (1 - 1 / i) ** (self.ema_exp_coefficient + 1)
                self.model_ema_worker.update_average(self.model, self.model_ema, beta=beta)

        def training_step(
            self, data_batch: Dict[str, torch.Tensor], iteration: int
        ) -> Tuple[Dict[str, torch.Tensor] | torch.Tensor]:
            # ! Important!!!
            # ! make sure the training step is the same as the forward method~(training_step in the super class)
            # ! this is necessary to trigger the FSDP-specific pre- and post-forward sharding logic
            return self.model(data_batch, iteration)

        def state_dict(self) -> Dict:
            raise NotImplementedError(
                "FSDPDiffModle does not support state_dict, use state_dict_model and FSDPCheckpointer"
            )

        @misc.timer("FSDP state_dict_model")
        def state_dict_model(self) -> Dict:
            with FSDP.summon_full_params(self.model):
                pass
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                model_state = self.model.state_dict()
            if self.config.ema.enabled:
                with FSDP.summon_full_params(self.model_ema):
                    pass
                with FSDP.state_dict_type(
                    self.model_ema,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    ema_model_state = self.model_ema.state_dict()
            else:
                ema_model_state = None
            return {
                "model": model_state,
                "ema": ema_model_state,
            }

        def load_state_dict(self, state_dict: Dict, strict: bool = True, assign: bool = False) -> None:
            raise NotImplementedError("FSDPDiffModle does not support load_state_dict, using FSDPCheckpointer")

        def init_optimizer_scheduler(
            self, optimizer_config: LazyDict, scheduler_config: LazyDict
        ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
            optimizer, scheduler = super().init_optimizer_scheduler(optimizer_config, scheduler_config)
            self.fsdp_checkpointer.load_optim_scheduler_during_init(
                self.model,
                optimizer,
                scheduler,
            )
            return optimizer, scheduler

        @contextmanager
        def ema_scope(self, context=None, is_cpu=False):
            if self.config.ema.enabled:
                self.model_ema_worker.cache(self.model.parameters(), is_cpu=is_cpu)
                self.model_ema_worker.copy_to(src_model=self.model_ema, tgt_model=self.model)
                if context is not None:
                    log.info(f"{context}: Switched to EMA weights")
            try:
                yield None
            finally:
                if self.config.ema.enabled:
                    self.model_ema_worker.restore(self.model.parameters())
                    if context is not None:
                        log.info(f"{context}: Restored training weights")

        def get_ckpt_postfix(self) -> Tuple[str, int]:
            """Get the checkpoint file postfix. check FSDPCheckpointer for more details

            Args:
                iteration (int): The current iteration number.

            Returns:
                postfix (str): The postfix of the checkpoint file.
                replicate_idx, shard_idx (int), current gpu replicate_idx, shard_idx in FSDP \
                    we will not save each ema model in each GPU, \
                    ema model with same rate will be saved once
                total_ema_num (int)
            """
            mesh_shape = parallel_state.fsdp_device_mesh.shape
            total_ema_num = min(self.config.ema.num, mesh_shape[0])
            replicate_idx, shard_idx = parallel_state.fsdp_device_mesh.get_coordinate()
            if replicate_idx == 0:
                return "", 0, shard_idx, total_ema_num
            if self.config.ema.enabled:
                if replicate_idx < self.config.ema.num:
                    return f"_RANK{replicate_idx}", replicate_idx, shard_idx, total_ema_num
            return "", replicate_idx, shard_idx, total_ema_num

    return FSDPClass


@diffusion_fsdp_class_decorator
class FSDPDiffusionModel(DiffusionModel):
    pass
