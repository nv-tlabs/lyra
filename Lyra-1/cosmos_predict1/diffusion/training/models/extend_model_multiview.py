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

from typing import Callable, Dict, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.functional.batch_ops import batch_mul
from cosmos_predict1.diffusion.training.conditioner import DataType, VideoExtendCondition
from cosmos_predict1.diffusion.training.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.training.models.extend_model import (
    ExtendDiffusionModel,
    VideoDenoisePrediction,
    normalize_condition_latent,
)
from cosmos_predict1.diffusion.training.models.model import DiffusionModel, broadcast_condition
from cosmos_predict1.diffusion.training.models.model_image import CosmosCondition, diffusion_fsdp_class_decorator
from cosmos_predict1.utils import log


class MultiviewExtendDiffusionModel(ExtendDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_views = config.n_views

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        encoded_state = self.vae.encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views) * self.sigma_data
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
        decoded_state = self.vae.decode(latent / self.sigma_data)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return decoded_state

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition: CosmosCondition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        if self.is_image_batch(data_batch):
            # Turn off CP
            self.net.disable_context_parallel()
        else:
            if parallel_state.is_initialized():
                if parallel_state.get_context_parallel_world_size() > 1:
                    # Turn on CP
                    cp_group = parallel_state.get_context_parallel_group()
                    self.net.enable_context_parallel(cp_group)
                    log.debug("[CP] Split x0 and epsilon")

                    x0 = rearrange(x0, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                    epsilon = rearrange(epsilon, "B C (V T) H W -> (B V) C T H W", V=self.n_views)

                    x0 = split_inputs_cp(x=x0, seq_dim=2, cp_group=self.net.cp_group)
                    epsilon = split_inputs_cp(x=epsilon, seq_dim=2, cp_group=self.net.cp_group)

                    x0 = rearrange(x0, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
                    epsilon = rearrange(epsilon, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        output_batch, kendall_loss, pred_mse, edm_loss = super(
            DiffusionModel, self
        ).compute_loss_with_epsilon_and_sigma(data_batch, x0_from_data_batch, x0, condition, epsilon, sigma)
        if not self.is_image_batch(data_batch):
            if self.loss_reduce == "sum" and parallel_state.get_context_parallel_world_size() > 1:
                kendall_loss *= parallel_state.get_context_parallel_world_size()

        return output_batch, kendall_loss, pred_mse, edm_loss

    def denoise(
        self,
        noise_x: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_video_augment_sigma_in_inference: float = 0.001,
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
            pred = super(DiffusionModel, self).denoise(noise_x, sigma, condition)
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
                condition, cfg_video_cond_bool, condition_latent, condition_video_augment_sigma_in_inference, sigma
            )
            condition_video_indicator = condition.condition_video_indicator  # [B, 1, T, 1, 1]
            if parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()

                condition_video_indicator = rearrange(
                    condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views
                )
                augment_latent = rearrange(augment_latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
                gt_latent = rearrange(gt_latent, "B C (V T) H W -> (B V) C T H W", V=self.n_views)

                condition_video_indicator = split_inputs_cp(condition_video_indicator, seq_dim=2, cp_group=cp_group)
                augment_latent = split_inputs_cp(augment_latent, seq_dim=2, cp_group=cp_group)
                gt_latent = split_inputs_cp(gt_latent, seq_dim=2, cp_group=cp_group)

                condition_video_indicator = rearrange(
                    condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views
                )
                augment_latent = rearrange(augment_latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
                gt_latent = rearrange(gt_latent, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

            if not condition.video_cond_bool:
                # Unconditional case, drop out the condition region
                augment_latent = self.drop_out_condition_region(augment_latent, noise_x, cfg_video_cond_bool)
            # Compose the model input with condition region (augment_latent) and generation region (noise_x)
            new_noise_xt = condition_video_indicator * augment_latent + (1 - condition_video_indicator) * noise_x
            # Call the abse model

            denoise_pred = super(DiffusionModel, self).denoise(new_noise_xt, sigma, condition)

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

        condition_video_indicator = rearrange(
            condition_video_indicator, "B C (V T) H W -> (B V) C T H W", V=self.n_views
        )
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
            num_condition_t = torch.randint(0, num_condition_t_max + 1, (1,)).item()
            condition_video_indicator[:, :, :num_condition_t] += 1.0

        else:
            raise NotImplementedError(
                f"condition_location {self.config.conditioner.video_cond_bool.condition_location} not implemented; training={self.training}"
            )

        condition_video_indicator = rearrange(
            condition_video_indicator, "(B V) C T H W -> B C (V T) H W", V=self.n_views
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

    def get_x0_fn_from_batch_with_condition_latent(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        condition_latent: torch.Tensor = None,
        num_condition_t: Union[int, None] = None,
        condition_video_augment_sigma_in_inference: float = None,
        add_input_frames_guidance: bool = False,
        guidance_other: Union[float, None] = None,
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
        if guidance_other is not None:  # and guidance_other != guidance:
            import copy

            assert not parallel_state.is_initialized(), "Parallel state not supported with two guidances."
            condition_other = copy.deepcopy(uncondition)
            condition_other.trajectory = condition.trajectory

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                cond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                ).x0_pred_replaced
                uncond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    uncondition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                ).x0_pred_replaced
                cond_other_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition_other,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                ).x0_pred_replaced
                return cond_x0 + guidance * (cond_x0 - uncond_x0) + guidance_other * (cond_other_x0 - uncond_x0)

        else:

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                cond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    condition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                ).x0_pred_replaced
                uncond_x0 = self.denoise(
                    noise_x,
                    sigma,
                    uncondition,
                    condition_video_augment_sigma_in_inference=condition_video_augment_sigma_in_inference,
                ).x0_pred_replaced
                return cond_x0 + guidance * (cond_x0 - uncond_x0)

        return x0_fn

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
        guidance_other: Union[float, None] = None,
    ) -> Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Different from the base model, this function support condition latent as input, it will create a differnt x0_fn if condition latent is given.
        If this feature is stablized, we could consider to move this function to the base model.

        Args:
            condition_latent (Optional[torch.Tensor]): latent tensor in shape B,C,T,H,W as condition to generate video.
            num_condition_t (Optional[int]): number of condition latent T, if None, will use the whole first half

            add_input_frames_guidance (bool): add guidance to the input frames, used for cfg on input frames
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
            guidance_other=guidance_other,
        )

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        x_sigma_max = (
            torch.randn(n_sample, *state_shape, **self.tensor_kwargs, generator=generator) * self.sde.sigma_max
        )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = rearrange(x_sigma_max, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            x_sigma_max = rearrange(x_sigma_max, "(B V) C T H W -> B C (V T) H W", V=self.n_views)

        samples = self.sampler(x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max)
        if self.net.is_context_parallel_enabled:
            samples = rearrange(samples, "B C (V T) H W -> (B V) C T H W", V=self.n_views)
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)
            samples = rearrange(samples, "(B V) C T H W -> B C (V T) H W", V=self.n_views)
        return samples


@diffusion_fsdp_class_decorator
class FSDPExtendDiffusionModel(MultiviewExtendDiffusionModel):
    pass
