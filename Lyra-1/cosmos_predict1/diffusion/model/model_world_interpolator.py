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

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import VideoExtendCondition
from cosmos_predict1.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel, broadcast_condition
from cosmos_predict1.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.modules.res_sampler import Sampler
from cosmos_predict1.diffusion.training.conditioner import DataType
from cosmos_predict1.diffusion.training.models.model import _broadcast
from cosmos_predict1.utils import log, misc

IS_PREPROCESSED_KEY = "is_preprocessed"
from dataclasses import dataclass, fields

from cosmos_predict1.diffusion.modules.denoiser_scaling import EDMScaling
from cosmos_predict1.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_predict1.diffusion.types import DenoisePrediction


@dataclass
class VideoDenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    eps: Optional[torch.Tensor] = None  # noise prediction
    logvar: Optional[torch.Tensor] = None  # log variance of noise prediction
    net_in: Optional[torch.Tensor] = None  # input to the network
    net_x0_pred: Optional[torch.Tensor] = None  # prediction of x0 from the network
    xt: Optional[torch.Tensor] = None  # input to the network, before multiply with c_in
    x0_pred_replaced: Optional[torch.Tensor] = None  # x0 prediction with condition region replaced by gt_latent


@dataclass
class CosmosCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    scalar_feature: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class DiffusionWorldInterpolatorWModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.is_extend_model = True
        self.num_valid_latents = config.latent_shape[1] - config.num_latents_to_drop
        self.setup_data_key()  # Initialize input_data_key and input_image_key
        self.sampler = Sampler()
        self.scaling = EDMScaling(self.sigma_data)
        self.sde = EDMSDE(
            p_mean=0.0,
            p_std=1.0,
            sigma_max=80,
            sigma_min=0.0002,
        )

    def setup_data_key(self) -> None:
        """Initialize data keys for image and video inputs."""
        self.input_data_key = self.config.input_data_key
        self.input_image_key = self.config.input_image_key

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """Determine if the data batch is an image batch or a video batch.

        Args:
            data_batch (dict[str, Tensor]): Input data batch.

        Returns:
            bool: True if the batch is an image batch, False if it is a video batch.

        Raises:
            AssertionError: If both or neither of input_image_key and input_data_key are present.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert (
            is_image != is_video
        ), "Only one of the input_image_key or input_data_key should be present in the data_batch."
        return is_image

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        Args:
            data_batch (dict[str, Tensor]): Dictionary containing the video data.
            input_key (str, optional): Key for the video data in the batch. Defaults to self.input_data_key.

        Side Effects:
            Modifies the video data tensor in-place to scale from [0, 255] to [-1, 1].
        """
        input_key = self.input_data_key if input_key is None else input_key
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all(
                    (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
                ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """Augments image data in-place by adding a temporal dimension.

        Args:
            data_batch (dict[str, Tensor]): Dictionary containing the image data.
            input_key (str, optional): Key for the image data in the batch. Defaults to self.input_image_key.

        Side Effects:
            Modifies the image data tensor in-place to add a temporal dimension (B,C,H,W -> B,C,1,H,W).
        """
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert (
                    data_batch[input_key].shape[2] == 1
                ), f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def normalize_condition_latent(self, condition_latent: torch.Tensor) -> torch.Tensor:
        """Normalize the condition latent tensor to have zero mean and unit variance."""
        condition_latent_2D = rearrange(condition_latent, "b c t h w -> b c t (h w)")
        mean = condition_latent_2D.mean(dim=-1)
        std = condition_latent_2D.std(dim=-1)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1)
        condition_latent = (condition_latent - mean) / std
        return condition_latent

    def draw_augment_sigma_and_epsilon(
        self, size: int, condition: VideoExtendCondition, p_mean: float, p_std: float, multiplier: float
    ) -> Tuple[Tensor, Tensor]:
        """Draw sigma and epsilon for augmenting conditional latent frames."""
        is_video_batch = condition.data_type == DataType.VIDEO
        del condition
        batch_size = size[0]
        epsilon = torch.randn(size, **self.tensor_kwargs)

        gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]

        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        sigma_B = torch.exp(log_sigma).to(**self.tensor_kwargs)

        sigma_B = _broadcast(sigma_B * multiplier, to_tp=True, to_cp=is_video_batch)
        epsilon = _broadcast(epsilon, to_tp=True, to_cp=is_video_batch)
        return sigma_B, epsilon

    def augment_conditional_latent_frames(
        self,
        condition: VideoExtendCondition,
        cfg_video_cond_bool: VideoCondBoolConfig,
        gt_latent: Tensor,
        condition_video_augment_sigma_in_inference: float = 0.001,
        sigma: Tensor = None,
        seed_inference: int = 1,
    ) -> Union[VideoExtendCondition, Tensor]:
        """Augment the condition input with noise."""
        if cfg_video_cond_bool.apply_corruption_to_condition_region == "noise_with_sigma":
            augment_sigma, _ = self.draw_augment_sigma_and_epsilon(
                gt_latent.shape,
                condition,
                cfg_video_cond_bool.augment_sigma_sample_p_mean,
                cfg_video_cond_bool.augment_sigma_sample_p_std,
                cfg_video_cond_bool.augment_sigma_sample_multiplier,
            )
            noise = torch.randn(*gt_latent.shape, **self.tensor_kwargs)
        elif cfg_video_cond_bool.apply_corruption_to_condition_region == "noise_with_sigma_fixed":
            log.debug(
                f"condition_video_augment_sigma_in_inference={condition_video_augment_sigma_in_inference}, sigma={sigma.flatten()[0]}"
            )
            assert (
                condition_video_augment_sigma_in_inference is not None
            ), "condition_video_augment_sigma_in_inference should be provided"
            augment_sigma = condition_video_augment_sigma_in_inference

            if augment_sigma >= sigma.flatten()[0]:
                log.debug("augment_sigma larger than sigma or other frame, remove condition")
                condition.condition_video_indicator = condition_video_indicator * 0

            augment_sigma = torch.tensor([augment_sigma], **self.tensor_kwargs)
            noise = misc.arch_invariant_rand(
                gt_latent.shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed_inference,
            )
        else:
            raise ValueError(f"does not support {cfg_video_cond_bool.apply_corruption_to_condition_region}")

        augment_latent = gt_latent + noise * augment_sigma.view(-1, 1, 1, 1, 1)
        _, _, c_in_augment, c_noise_augment = self.scaling(sigma=augment_sigma)

        if cfg_video_cond_bool.condition_on_augment_sigma:
            if condition.condition_video_indicator.sum() > 0:
                condition.condition_video_augment_sigma = c_noise_augment
            else:
                condition.condition_video_augment_sigma = torch.zeros_like(c_noise_augment)

        augment_latent_cin = batch_mul(augment_latent, c_in_augment)
        _, _, c_in, _ = self.scaling(sigma=sigma)
        augment_latent_cin = batch_mul(augment_latent_cin, 1 / c_in)

        return condition, augment_latent_cin

    def super_denoise(self, xt: torch.Tensor, sigma: torch.Tensor, condition: CosmosCondition) -> DenoisePrediction:
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

    def drop_out_condition_region(
        self, augment_latent: Tensor, noise_x: Tensor, cfg_video_cond_bool: VideoCondBoolConfig
    ) -> Tensor:
        """Drop out the conditional region for CFG on input frames."""
        if cfg_video_cond_bool.cfg_unconditional_type == "zero_condition_region_condition_mask":
            augment_latent_drop = torch.zeros_like(augment_latent)
        elif cfg_video_cond_bool.cfg_unconditional_type == "noise_x_condition_region":
            augment_latent_drop = noise_x
        else:
            raise NotImplementedError(
                f"cfg_unconditional_type {cfg_video_cond_bool.cfg_unconditional_type} not implemented"
            )
        return augment_latent_drop

    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
        seed_inference: int = 1,
    ) -> VideoDenoisePrediction:
        """Denoise the noisy input tensor for video data."""
        assert (
            condition.gt_latent is not None
        ), "find None gt_latent in condition, likely didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition"
        gt_latent = condition.gt_latent
        cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

        condition_latent = gt_latent

        if cfg_video_cond_bool.normalize_condition_latent:
            condition_latent = self.normalize_condition_latent(condition_latent)

        condition, augment_latent = self.augment_conditional_latent_frames(
            condition,
            cfg_video_cond_bool,
            condition_latent,
            condition_video_augment_sigma_in_inference,
            sigma,
            seed_inference=seed_inference,
        )
        condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]
        if parallel_state.get_context_parallel_world_size() > 1:
            cp_group = parallel_state.get_context_parallel_group()
            condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
            augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
            gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

        if not condition.video_cond_bool:
            augment_latent = self.drop_out_condition_region(augment_latent, noise_x, cfg_video_cond_bool)

        new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
        denoise_pred = self.super_denoise(new_noise_xt, sigma, condition)

        x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0
        if cfg_video_cond_bool.compute_loss_for_condition_region:
            x0_pred = denoise_pred.x0
        else:
            x0_pred = x0_pred_replaced

        return VideoDenoisePrediction(
            x0=x0_pred,
            eps=batch_mul(noise_x - x0_pred, 1.0 / sigma),
            logvar=denoise_pred.logvar,
            net_in=batch_mul(1.0 / torch.sqrt(self.sigma_data**2 + sigma**2), new_noise_xt),
            net_x0_pred=denoise_pred.x0,
            xt=new_noise_xt,
            x0_pred_replaced=x0_pred_replaced,
        )

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Union[torch.Tensor, None] = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        return_noise: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Generate samples from the batch. Supports condition latent for video generation.

        Args:
            data_batch (Dict): Input data batch.
            guidance (float): Guidance scale for classifier-free guidance.
            seed (int): Random seed for reproducibility.
            state_shape (Tuple | None): Shape of the latent state, defaults to self.state_shape if None.
            n_sample (int | None): Number of samples to generate, inferred from batch if None.
            is_negative_prompt (bool): Use negative prompt for unconditioned generation.
            num_steps (int): Number of sampling steps.
            condition_latent (torch.Tensor | None): Latent tensor (B,C,T,H,W) as condition for video generation.
            num_condition_t (int | None): Number of condition frames in T dimension.
            condition_video_augment_sigma_in_inference (float): Sigma for augmenting condition video in inference.
            add_input_frames_guidance (bool): Apply guidance to input frames for CFG.
            return_noise (bool): Return initial noise along with samples.

        Returns:
            Tensor | Tuple[Tensor, Tensor]: Generated samples, or (samples, noise) if return_noise is True.
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if is_image_batch:
            log.debug("image batch, call base model generate_samples_from_batch")
            return super().generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                is_negative_prompt=is_negative_prompt,
                num_steps=num_steps,
            )
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W
            else:
                log.debug(f"Default Video state shape is used. {self.state_shape}")
                state_shape = self.state_shape

        assert condition_latent is not None, "condition_latent should be provided"

        x0_fn = self.get_x0_fn_from_batch_with_condition_latent(
            data_batch,
            guidance,
            is_negative_prompt=is_negative_prompt,
            condition_latent=condition_latent,
            num_condition_t=num_condition_t,
            condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
            add_input_frames_guidance=add_input_frames_guidance,
            seed_inference=seed,
        )

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape), torch.float32, self.tensor_kwargs["device"], seed
            )
            * self.sde.sigma_max
        )
        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        if return_noise:
            if self.net.is_context_parallel_enabled:
                x_sigma_max = cat_outputs_cp(x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            return samples, x_sigma_max / self.sde.sigma_max

        return samples

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        seed_inference: int = 1,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` for denoising based on the data batch and condition latent.

        Args:
            data_batch (Dict): Input data batch.
            guidance (float): Guidance scale.
            is_negative_prompt (bool): Use negative prompt for unconditioned generation.
            condition_latent (torch.Tensor): Latent tensor (B,C,T,H,W) as condition.
            num_condition_t (int | None): Number of condition frames.
            condition_video_augment_sigma_in_inference (float): Sigma for condition augmentation.
            add_input_frames_guidance (bool): Apply guidance to input frames.
            seed_inference (int): Seed for inference noise.

        Returns:
            Callable: Function `x0_fn(noise_x, sigma)` returning denoised prediction.
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            condition = self.add_condition_pose(data_batch, condition)

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )
        if self.config.conditioner.video_cond_bool.add_pose_condition:
            uncondition = self.add_condition_pose(data_batch, uncondition)

        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed_inference=seed_inference,
            ).x0_pred_replaced
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
                condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                seed_inference=seed_inference,
            ).x0_pred_replaced
            return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Union[int, None] = None
    ) -> VideoExtendCondition:
        """Add condition_video_indicator and condition_video_input_mask to the condition object for video conditioning.
        condition_video_indicator is a binary tensor indicating the condition region in the latent state. 1x1xTx1x1 tensor.
        condition_video_input_mask will be concat with the input for the network.
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region
        if self.config.conditioner.video_cond_bool.condition_location == "first_n":
            # Only in inference to decide the condition region
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            log.info(
                f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
            )
            condition_video_indicator[:, :, :num_condition_t] += 1.0
        elif self.config.conditioner.video_cond_bool.condition_location == "first_and_last_1":
            # Should be used for both training and inference. The first and last frame will be condition frames.
            assert num_condition_t is not None, "num_condition_t should be provided"
            assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
            log.info(
                f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
            )
            condition_video_indicator[:, :, :num_condition_t] += 1.0
            condition_video_indicator[:, :, -num_condition_t:] += 1.0
        elif self.config.conditioner.video_cond_bool.condition_location == "first_random_n":
            # Only in training
            num_condition_t_max = self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_max
            assert (
                num_condition_t_max <= T
            ), f"num_condition_t_max should be less than T, get {num_condition_t_max}, {T}"
            assert num_condition_t_max >= self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min
            num_condition_t = torch.randint(
                self.config.conditioner.video_cond_bool.first_random_n_num_condition_t_min,
                num_condition_t_max + 1,
                (1,),
            ).item()
            condition_video_indicator[:, :, :num_condition_t] += 1.0

        elif self.config.conditioner.video_cond_bool.condition_location == "random":
            # Only in training
            condition_rate = self.config.conditioner.video_cond_bool.random_conditon_rate
            flag = torch.ones(1, 1, T, 1, 1, device=latent_state.device).type(latent_dtype) * condition_rate
            condition_video_indicator = torch.bernoulli(flag).type(latent_dtype).to(latent_state.device)
        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
            )
        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition

    def add_condition_pose(self, data_batch: Dict, condition: VideoExtendCondition) -> VideoExtendCondition:
        """
        Adds pose condition to the condition object for camera control.

        Args:
            data_batch (Dict): Data batch with 'plucker_embeddings' or 'plucker_embeddings_downsample'.
            condition (VideoExtendCondition): Condition object to update.

        Returns:
            VideoExtendCondition: Updated condition object.
        """
        assert (
            "plucker_embeddings" in data_batch or "plucker_embeddings_downsample" in data_batch.keys()
        ), f"plucker_embeddings should be in data_batch. only find {data_batch.keys()}"
        plucker_embeddings = (
            data_batch["plucker_embeddings"]
            if "plucker_embeddings_downsample" not in data_batch.keys()
            else data_batch["plucker_embeddings_downsample"]
        )
        condition.condition_video_pose = rearrange(plucker_embeddings, "b t c h w -> b c t h w").contiguous()
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition
