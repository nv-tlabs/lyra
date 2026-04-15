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

import torch
from diffusers import EDMEulerScheduler
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import BaseVideoCondition
from cosmos_predict1.diffusion.module import parallel
from cosmos_predict1.diffusion.module.blocks import FourierFeatures
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.module.pretrained_vae import BaseVAE
from cosmos_predict1.diffusion.training.utils.layer_control.peft_control_config_parser import LayerControlConfigParser
from cosmos_predict1.diffusion.training.utils.peft.peft import add_lora_layers, setup_lora_requires_grad
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.distributed import get_rank
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate


class DiffusionT2WModel(torch.nn.Module):
    """Text-to-world diffusion model that generates video frames from text descriptions.

    This model implements a diffusion-based approach for generating videos conditioned on text input.
    It handles the full pipeline including encoding/decoding through a VAE, diffusion sampling,
    and classifier-free guidance.
    """

    def __init__(self, config):
        """Initialize the diffusion model.

        Args:
            config: Configuration object containing model parameters and architecture settings
        """
        super().__init__()
        # Initialize trained_data_record with defaultdict, key: image, video, iteration
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.debug(f"DiffusionModel: precision {self.precision}")
        # Timer passed to network to detect slow ranks.
        # 1. set data keys and data information
        self.sigma_data = config.sigma_data
        self.state_shape = list(config.latent_shape)
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.scheduler = EDMEulerScheduler(sigma_max=80, sigma_min=0.0002, sigma_data=self.sigma_data)
        self.tokenizer = None
        self.model = None

    @property
    def net(self):
        return self.model.net

    @property
    def conditioner(self):
        return self.model.conditioner

    @property
    def logvar(self):
        return self.model.logvar

    def set_up_tokenizer(self, tokenizer_dir: str):
        self.tokenizer: BaseVAE = lazy_instantiate(self.config.tokenizer)
        self.tokenizer.load_weights(tokenizer_dir)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self, memory_format: torch.memory_format = torch.preserve_format):
        """Initialize the core model components including network, conditioner and logvar."""
        self.model = self.build_model()
        if self.config.peft_control and self.config.peft_control.enabled:
            log.info("Setting up LoRA layers")
            peft_control_config_parser = LayerControlConfigParser(config=self.config.peft_control)
            peft_control_config = peft_control_config_parser.parse()
            add_lora_layers(self.model, peft_control_config)
            num_lora_params = setup_lora_requires_grad(self.model)
            self.model.requires_grad_(False)
            if num_lora_params == 0:
                raise ValueError("No LoRA parameters found. Please check the model configuration.")
        self.model = self.model.to(memory_format=memory_format, **self.tensor_kwargs)

    def build_model(self) -> torch.nn.ModuleDict:
        """Construct the model's neural network components.

        Returns:
            ModuleDict containing the network, conditioner and logvar components
        """
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

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode input state into latent representation using VAE.

        Args:
            state: Input tensor to encode

        Returns:
            Encoded latent representation scaled by sigma_data
        """
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to pixel space using VAE.

        Args:
            latent: Latent tensor to decode

        Returns:
            Decoded tensor in pixel space
        """
        return self.tokenizer.decode(latent / self.sigma_data)

    def setup_data_key(self) -> None:
        """Configure input data keys for video and image data."""
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling.

        This function generates samples from either image or video data batches using diffusion sampling.
        It handles both conditional and unconditional generation with classifier-free guidance.

        Args:
            data_batch (dict): Raw data batch from the training data loader
            guidance (float, optional): Classifier-free guidance weight. Defaults to 1.5.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            state_shape (tuple | None, optional): Shape of the state tensor. Uses self.state_shape if None. Defaults to None.
            n_sample (int | None, optional): Number of samples to generate. Defaults to 1.
            is_negative_prompt (bool, optional): Whether to use negative prompt for unconditional generation. Defaults to False.
            num_steps (int, optional): Number of diffusion sampling steps. Defaults to 35.

        Returns:
            Tensor: Generated samples after diffusion sampling
        """
        condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)

        self.scheduler.set_timesteps(num_steps)

        xt = torch.randn(size=(n_sample,) + tuple(state_shape)) * self.scheduler.init_noise_sigma
        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)

        for t in self.scheduler.timesteps:
            xt = xt.to(**self.tensor_kwargs)
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
            net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
            net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            # Compute the previous noisy sample x_t -> x_t-1
            xt = self.scheduler.step(net_output, t, xt).prev_sample
        samples = xt

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
    ):
        """Get the conditions for the model.

        Args:
            data_batch: Input data dictionary
            is_negative_prompt: Whether to use negative prompting

        Returns:
            condition: Input conditions
            uncondition: Conditions removed/reduced to minimum (unconditioned)
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition


def broadcast_condition(condition: BaseVideoCondition, to_tp: bool = True, to_cp: bool = True) -> BaseVideoCondition:
    condition_kwargs = {}
    for k, v in condition.to_dict().items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"{k} requires gradient. the current impl does not support it"
        condition_kwargs[k] = parallel.broadcast(v, to_tp=to_tp, to_cp=to_cp)
    condition = type(condition)(**condition_kwargs)
    return condition
