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

"""
A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.
It allows us easy to switch building blocks used and their order. Its instantiation includes
* transformer on fully flattened tokens
* factored spatial and temporal attention
* factored non-overlap spatial and temporal attention
* mixing of above attention types

Limitations:

* In favor of simplicity and cleanness, many ops are not fused and we can do better
* such as combining mutiple adaln MLPs into one inside one transformer block.
* we use reshape heavily, which may be not efficient when its occurs unnecessary CUDA memory copy

Purpose:
* A prototype for testing different attention types and their combinations
* Idealy, we want to know where we should allocate our resources / FLOPS / memory via extensive empirical studies
"""

from collections.abc import Container
from typing import List, Optional, Tuple

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import nn

from cosmos_predict1.diffusion.module.timm import Mlp
from cosmos_predict1.diffusion.training.conditioner import DataType
from cosmos_predict1.diffusion.training.context_parallel import split_inputs_cp
from cosmos_predict1.diffusion.training.networks.general_dit import GeneralDIT
from cosmos_predict1.diffusion.training.tensor_parallel import scatter_along_first_dim
from cosmos_predict1.utils import log


class ActionConditionalGeneralDIT(GeneralDIT):
    """
    ActionConditionalGeneralDIT is a subclass of GeneralDIT that take `action` as condition.
    Action embedding is would be added to timestep embedding.
    """

    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
        """
        del kwargs
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")
            timesteps_B_D = timesteps_B_D + scalar_feature.mean(dim=1)

        if self.additional_timestamp_channels:
            additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                bs=x.shape[0],
                fps=fps,
                h=image_size[:, 0],
                w=image_size[:, 1],
                org_h=image_size[:, 2],
                org_w=image_size[:, 3],
            )

            affline_emb_B_D += additional_cond_B_D
            affline_scale_log_info["additional_cond_B_D"] = additional_cond_B_D.detach()

        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D
        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks["block0"].x_format == "THWBD":
            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")

            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

            if self.sequence_parallel:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                # Sequence parallel requires the input tensor to be scattered along the first dimension.
                assert self.block_config == "FA-CA-MLP"  # Only support this block config for now
                T, H, W, B, D = x.shape
                # variable name x_T_H_W_B_D is no longer valid. x is reshaped to THW*1*1*b*D and will be reshaped back in FinalLayer
                x = x.view(T * H * W, 1, 1, B, D)
                assert x.shape[0] % parallel_state.get_tensor_model_parallel_world_size() == 0
                x = scatter_along_first_dim(x, tp_group)

                if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.view(
                        T * H * W, 1, 1, B, D
                    )
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = scatter_along_first_dim(
                        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, tp_group
                    )

        elif self.blocks["block0"].x_format == "BTHWD":
            x = x_B_T_H_W_D
        else:
            raise ValueError(f"Unknown x_format {self.blocks[0].x_format}")
        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        x_ctrl: Optional[dict] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        feature_indices: Optional[Container[int]] = None,
        return_features_early: bool = False,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks
            feature_indices: A set of feature indices (a set of integers) decides which blocks
                to extract features from. If the set is non-empty, then features will be returned.
                By default, feature_indices=None means extract no features.
            return_features_early: If true, the forward pass returns the features once the set is complete.
                This means the forward pass will not finish completely and no final output is returned.
            condition_video_augment_sigma: (B,) used in lvg(long video generation), we add noise with this sigma to augment condition input, the lvg model will condition on the condition_video_augment_sigma value;
                we need forward_before_blocks pass to the forward_before_blocks function.
        """
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            # Exit immediately if user requested this.
            return []

        inputs = self.forward_before_blocks(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            action=action,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )
        x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
            inputs["x"],
            inputs["affline_emb_B_D"],
            inputs["crossattn_emb"],
            inputs["crossattn_mask"],
            inputs["rope_emb_L_1_1_D"],
            inputs["adaln_lora_B_3D"],
            inputs["original_shape"],
        )
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"]
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"

        if self.use_memory_save:
            return self.forward_blocks_memory_save(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D,
                adaln_lora_B_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                feature_indices,
                original_shape,
                x_ctrl,
                return_features_early,
            )

        return self.forward_blocks_regular(
            x,
            affline_emb_B_D,
            crossattn_emb,
            crossattn_mask,
            rope_emb_L_1_1_D,
            adaln_lora_B_3D,
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            feature_indices,
            original_shape,
            x_ctrl,
            return_features_early,
        )


class ActionConditionalVideoExtendGeneralDIT(ActionConditionalGeneralDIT):
    """
    ActionConditionalVideoExtendGeneralDIT is a subclass of ActionConditionalGeneralDIT that take `action` as condition.
    Action embedding is would be added to timestep embedding.
    """

    def __init__(self, *args, in_channels=16 + 1, add_augment_sigma_embedding=False, **kwargs):
        self.add_augment_sigma_embedding = add_augment_sigma_embedding

        # extra channel for video condition mask
        super().__init__(*args, in_channels=in_channels, **kwargs)
        log.info(f"VideoExtendGeneralDIT in_channels: {in_channels}")

        assert hasattr(self, "model_channels"), "model_channels attribute is missing"
        self.action_embedder_B_D = Mlp(
            in_features=7,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D = Mlp(
            in_features=7,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        video_cond_bool: Optional[torch.Tensor] = None,
        condition_video_indicator: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        condition_video_pose: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Args:
        condition_video_augment_sigma: (B) tensor of sigma value for the conditional input augmentation
        condition_video_pose: (B, 1, T, H, W) tensor of pose condition
        """
        B, C, T, H, W = x.shape

        if data_type == DataType.VIDEO:
            assert (
                condition_video_input_mask is not None
            ), "condition_video_input_mask is required for video data type; check if your model_obj is extend_model.FSDPDiffusionModel or the base DiffusionModel"
            if self.cp_group is not None:
                condition_video_input_mask = split_inputs_cp(
                    condition_video_input_mask, seq_dim=2, cp_group=self.cp_group
                )
                condition_video_indicator = split_inputs_cp(
                    condition_video_indicator, seq_dim=2, cp_group=self.cp_group
                )
                if condition_video_pose is not None:
                    condition_video_pose = split_inputs_cp(condition_video_pose, seq_dim=2, cp_group=self.cp_group)
            # log.critical(f"hit video case, video_cond_bool: {video_cond_bool}, condition_video_indicator: {condition_video_indicator.flatten()}, condition_video_input_mask: {condition_video_input_mask.shape}, {condition_video_input_mask[:,:,:,0,0]}", rank0_only=False)
            input_list = [x, condition_video_input_mask]
            if condition_video_pose is not None:
                if condition_video_pose.shape[2] > T:
                    log.warning(
                        f"condition_video_pose has more frames than the input video: {condition_video_pose.shape} > {x.shape}"
                    )
                    condition_video_pose = condition_video_pose[:, :, :T, :, :].contiguous()
                input_list.append(condition_video_pose)
            x = torch.cat(
                input_list,
                dim=1,
            )

        if data_type == DataType.IMAGE:
            # For image, we dont have condition_video_input_mask, or condition_video_pose
            # We need to add the extra channel for video condition mask
            padding_channels = self.in_channels - x.shape[1]
            x = torch.cat([x, torch.zeros((B, padding_channels, T, H, W), dtype=x.dtype, device=x.device)], dim=1)
        else:
            assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels, got {x.shape[1]}"
        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            action=action,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )

    def forward_before_blocks(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            crossattn_mask: (B, N) tensor of cross-attention masks

            condition_video_augment_sigma: (B, T) tensor of sigma value for the conditional input augmentation
        """
        del kwargs
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        original_shape = x.shape
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
        )
        # logging affline scale information
        affline_scale_log_info = {}

        timesteps_B_D, adaln_lora_B_3D = self.t_embedder(timesteps.flatten())
        affline_emb_B_D = timesteps_B_D
        affline_scale_log_info["timesteps_B_D"] = timesteps_B_D.detach()

        # Add action conditioning
        assert action is not None, "Action is required for action-conditional training"
        if action is not None:
            action = action[:, 0, :]  # Since we are now training on 1 frame, we only need the first frame action.
            action_embedding_B_D = self.action_embedder_B_D(action)
            action_embedding_B_3D = self.action_embedder_B_3D(action)
            timesteps_B_D = timesteps_B_D + action_embedding_B_D
            adaln_lora_B_3D = adaln_lora_B_3D + action_embedding_B_3D

        if scalar_feature is not None:
            raise NotImplementedError("Scalar feature is not implemented yet.")
            timesteps_B_D = timesteps_B_D + scalar_feature.mean(dim=1)
        if self.additional_timestamp_channels:
            additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                bs=x.shape[0],
                fps=fps,
                h=image_size[:, 0],
                w=image_size[:, 1],
                org_h=image_size[:, 2],
                org_w=image_size[:, 3],
            )

            affline_emb_B_D += additional_cond_B_D
            affline_scale_log_info["additional_cond_B_D"] = additional_cond_B_D.detach()
        if self.add_augment_sigma_embedding:
            if condition_video_augment_sigma is None:
                # Handling image case
                # Note: for video case, when there is not condition frames, we also set it as zero, see extend_model augment_conditional_latent_frames function
                assert data_type == DataType.IMAGE, "condition_video_augment_sigma is required for video data type"
                condition_video_augment_sigma = torch.zeros_like(timesteps.flatten())

            affline_augment_sigma_emb_B_D, adaln_lora_sigma_emb_B_3D = self.augment_sigma_embedder(
                condition_video_augment_sigma.flatten()
            )
            affline_emb_B_D = affline_emb_B_D + affline_augment_sigma_emb_B_D
        affline_scale_log_info["affline_emb_B_D"] = affline_emb_B_D.detach()
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        # for logging purpose
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = affline_emb_B_D
        self.crossattn_emb = crossattn_emb
        self.crossattn_mask = crossattn_mask

        if self.use_cross_attn_mask:
            crossattn_mask = crossattn_mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, length]
        else:
            crossattn_mask = None

        if self.blocks["block0"].x_format == "THWBD":
            x = rearrange(x_B_T_H_W_D, "B T H W D -> T H W B D")
            if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = rearrange(
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, "B T H W D -> T H W B D"
                )
            crossattn_emb = rearrange(crossattn_emb, "B M D -> M B D")
            if crossattn_mask:
                crossattn_mask = rearrange(crossattn_mask, "B M -> M B")

            if self.sequence_parallel:
                tp_group = parallel_state.get_tensor_model_parallel_group()
                # Sequence parallel requires the input tensor to be scattered along the first dimension.
                assert self.block_config == "FA-CA-MLP"  # Only support this block config for now
                T, H, W, B, D = x.shape
                # variable name x_T_H_W_B_D is no longer valid. x is reshaped to THW*1*1*b*D and will be reshaped back in FinalLayer
                x = x.view(T * H * W, 1, 1, B, D)
                assert x.shape[0] % parallel_state.get_tensor_model_parallel_world_size() == 0
                x = scatter_along_first_dim(x, tp_group)

                if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.view(
                        T * H * W, 1, 1, B, D
                    )
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = scatter_along_first_dim(
                        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, tp_group
                    )

        elif self.blocks["block0"].x_format == "BTHWD":
            x = x_B_T_H_W_D
        else:
            raise ValueError(f"Unknown x_format {self.blocks[0].x_format}")
        output = {
            "x": x,
            "affline_emb_B_D": affline_emb_B_D,
            "crossattn_emb": crossattn_emb,
            "crossattn_mask": crossattn_mask,
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_3D": adaln_lora_B_3D,
            "original_shape": original_shape,
            "extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        return output
