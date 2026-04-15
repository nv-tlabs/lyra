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

from cosmos_predict1.diffusion.config.base.conditioner import VideoCondBoolConfig
from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.training.conditioner import DataType, VideoExtendCondition
from cosmos_predict1.diffusion.training.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.training.models.model import DiffusionModel as BaseModel
from cosmos_predict1.diffusion.training.models.model import _broadcast, broadcast_condition
from cosmos_predict1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_predict1.utils import log, misc


@dataclass
class VideoDenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    eps: Optional[torch.Tensor] = None  # noise prediction
    logvar: Optional[torch.Tensor] = None  # log variance of noise prediction, can be used a confidence / uncertainty
    net_in: Optional[torch.Tensor] = None  # input to the network
    net_x0_pred: Optional[torch.Tensor] = None  # prediction of x0 from the network
    xt: Optional[torch.Tensor] = None  # input to the network, before muliply with c_in
    x0_pred_replaced: Optional[torch.Tensor] = None  # x0 prediction with condition region replaced by gt_latent


def normalize_condition_latent(condition_latent):
    """Normalize the condition latent tensor to have zero mean and unit variance
    Args:
        condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W
    """
    condition_latent_2D = rearrange(condition_latent, "b c t h w -> b c t (h w)")
    mean = condition_latent_2D.mean(dim=-1)
    std = condition_latent_2D.std(dim=-1)
    # bct -> bct11
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(-1).unsqueeze(-1)
    condition_latent = (condition_latent - mean) / std
    return condition_latent


class ExtendDiffusionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.is_extend_model = True

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor], num_condition_t: Union[int, None] = None
    ) -> Tuple[Tensor, VideoExtendCondition]:
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        if condition.data_type == DataType.VIDEO:
            if self.config.conditioner.video_cond_bool.sample_tokens_start_from_p_or_i:
                latent_state = self.sample_tokens_start_from_p_or_i(latent_state)
            condition = self.add_condition_video_indicator_and_video_input_mask(
                latent_state, condition, num_condition_t=num_condition_t
            )
            if self.config.conditioner.video_cond_bool.add_pose_condition:
                condition = self.add_condition_pose(data_batch, condition)
        log.debug(f"condition.data_type {condition.data_type}")
        return raw_state, latent_state, condition

    def draw_augment_sigma_and_epsilon(
        self, size: int, condition: VideoExtendCondition, p_mean: float, p_std: float, multiplier: float
    ) -> Tensor:
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
        """This function is used to augment the condition input with noise
        Args:
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            cfg_video_cond_bool (VideoCondBoolConfig): video condition bool config
            gt_latent (Tensor): ground truth latent tensor in shape B,C,T,H,W
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
            sigma (Tensor): noise level for the generation region
        Returns:
            VideoExtendCondition: updated condition object
                condition_video_augment_sigma: sigma for the condition region, feed to the network
            augment_latent (Tensor): augmented latent tensor in shape B,C,T,H,W

        """

        if cfg_video_cond_bool.apply_corruption_to_condition_region == "noise_with_sigma":
            # Training only, sample sigma for the condition region
            augment_sigma, _ = self.draw_augment_sigma_and_epsilon(
                gt_latent.shape,
                condition,
                cfg_video_cond_bool.augment_sigma_sample_p_mean,
                cfg_video_cond_bool.augment_sigma_sample_p_std,
                cfg_video_cond_bool.augment_sigma_sample_multiplier,
            )
            noise = torch.randn(*gt_latent.shape, **self.tensor_kwargs)

        elif cfg_video_cond_bool.apply_corruption_to_condition_region == "noise_with_sigma_fixed":
            # Inference only, use fixed sigma for the condition region
            log.debug(
                f"condition_video_augment_sigma_in_inference={condition_video_augment_sigma_in_inference}, sigma={sigma.flatten()[0]}"
            )
            assert (
                condition_video_augment_sigma_in_inference is not None
            ), "condition_video_augment_sigma_in_inference should be provided"
            augment_sigma = condition_video_augment_sigma_in_inference

            if augment_sigma >= sigma.flatten()[0]:
                # This is a inference trick! If the sampling sigma is smaller than the augment sigma, we will start denoising the condition region together.
                # This is achieved by setting all region as `generation`, i.e. value=0
                log.debug("augment_sigma larger than sigma or other frame, remove condition")
                condition.condition_video_indicator = condition.condition_video_indicator * 0

            augment_sigma = torch.tensor([augment_sigma], **self.tensor_kwargs)

            # Inference, use fixed seed
            noise = misc.arch_invariant_rand(
                gt_latent.shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed_inference,
            )
        else:
            raise ValueError(f"does not support {cfg_video_cond_bool.apply_corruption_to_condition_region}")

        # Now apply the augment_sigma to the gt_latent

        augment_latent = gt_latent + noise * augment_sigma.view(-1, 1, 1, 1, 1)
        _, _, c_in_augment, c_noise_augment = self.scaling(sigma=augment_sigma)

        if cfg_video_cond_bool.condition_on_augment_sigma:  # model takes augment_sigma as input
            if condition.condition_video_indicator.sum() > 0:  # has condition frames
                condition.condition_video_augment_sigma = c_noise_augment
            else:  # no condition frames
                condition.condition_video_augment_sigma = torch.zeros_like(c_noise_augment)

        # Multiply the whole latent with c_in_augment
        augment_latent_cin = batch_mul(augment_latent, c_in_augment)

        # Since the whole latent will multiply with c_in later, we devide the value to cancel the effect
        _, _, c_in, _ = self.scaling(sigma=sigma)
        augment_latent_cin = batch_mul(augment_latent_cin, 1 / c_in)

        return condition, augment_latent_cin

    def drop_out_condition_region(
        self, augment_latent: Tensor, noise_x: Tensor, cfg_video_cond_bool: VideoCondBoolConfig
    ) -> Tensor:
        """Use for CFG on input frames, we drop out the conditional region
        There are two option:
        1. when we dropout, we set the region to be zero
        2. when we dropout, we set the region to be noise_x
        """
        # Unconditional case, use for cfg
        if cfg_video_cond_bool.cfg_unconditional_type == "zero_condition_region_condition_mask":
            # Set the condition location input to be zero
            augment_latent_drop = torch.zeros_like(augment_latent)
        elif cfg_video_cond_bool.cfg_unconditional_type == "noise_x_condition_region":
            # Set the condition location input to be noise_x, i.e., same as base model training
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
        """
        Denoise the noisy input tensor.

        Args:
            noise_x (Tensor): Noisy input tensor.
            sigma (Tensor): Noise level.
            condition (VideoExtendCondition): Condition for denoising.
            condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference

        Returns:
            Tensor: Denoised output tensor.
        """
        if condition.data_type == DataType.IMAGE:
            pred = super().denoise(noise_x, sigma, condition)
            log.debug(f"hit image denoise, noise_x shape {noise_x.shape}, sigma shape {sigma.shape}", rank0_only=False)
            return VideoDenoisePrediction(
                x0=pred.x0,
                eps=pred.eps,
                logvar=pred.logvar,
                xt=noise_x,
            )
        else:
            assert (
                condition.gt_latent is not None
            ), f"find None gt_latent in condition, likely didn't call self.add_condition_video_indicator_and_video_input_mask when preparing the condition or this is a image batch but condition.data_type is wrong, get {noise_x.shape}"
            gt_latent = condition.gt_latent
            cfg_video_cond_bool: VideoCondBoolConfig = self.config.conditioner.video_cond_bool

            condition_latent = gt_latent

            if cfg_video_cond_bool.normalize_condition_latent:
                condition_latent = normalize_condition_latent(condition_latent)

            # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
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
                # Unconditional case, drop out the condition region
                augment_latent = self.drop_out_condition_region(augment_latent, noise_x, cfg_video_cond_bool)

            # Compose the model input with condition region (augment_latent) and generation region (noise_x)
            new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
            # Call the abse model
            denoise_pred = super().denoise(new_noise_xt, sigma, condition)

            x0_pred_replaced = condition_video_indicator * gt_latent + (1 - condition_video_indicator) * denoise_pred.x0
            if cfg_video_cond_bool.compute_loss_for_condition_region:
                # We also denoise the conditional region
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
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half

            add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames
            return_noise (bool): return the initial noise or not, used for ODE pairs generation
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
            seed_inference=seed,  # Use for noise of augment sigma
        )

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape), torch.float32, self.tensor_kwargs["device"], seed
            )
            * self.sde.sigma_max
        )
        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

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
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.
        Different from the base model, this function support condition latent as input, it will add the condition information into the condition and uncondition object.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true
        - condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
        - add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
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
        # For inference, check if parallel_state is initialized
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
        """Add pose condition to the condition object. For camera control model
        Args:
            data_batch (Dict): data batch, with key "plucker_embeddings", in shape B,T,C,H,W
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
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
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition

    def sample_tokens_start_from_p_or_i(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Sample the PPP... from the IPPP... sequence, only for video sequence
        Args:
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
        Returns:
            torch.Tensor: sampled PPP tensor in shape B,C,T,H,W
        """
        B, C, T, H, W = latent_state.shape
        latent_dtype = latent_state.dtype
        T_target = self.state_shape[1]
        latent_state_sample = torch.zeros((B, C, T_target, H, W), dtype=latent_dtype, device=latent_state.device)
        t_start = torch.randint(0, T - T_target + 1, (1,))
        # broadcast to other device
        latent_state_sample = latent_state[:, :, t_start : t_start + T_target].contiguous()
        if parallel_state.is_initialized():
            latent_state_sample = _broadcast(latent_state_sample, to_tp=True, to_cp=True)

        return latent_state_sample


@diffusion_fsdp_class_decorator
class FSDPExtendDiffusionModel(ExtendDiffusionModel):
    pass
