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

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import amp_C
import torch
from apex.multi_tensor_apply import multi_tensor_applier
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import broadcast_object_list, get_process_group_ranks
from torch.distributed.utils import _verify_param_shape_across_processes

from cosmos_predict1.diffusion.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_predict1.diffusion.training.conditioner import BaseVideoCondition, DataType
from cosmos_predict1.diffusion.training.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict1.diffusion.training.models.model_image import CosmosCondition
from cosmos_predict1.diffusion.training.models.model_image import DiffusionModel as ImageModel
from cosmos_predict1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_predict1.utils import distributed, log, misc

l2_norm_impl = amp_C.multi_tensor_l2norm
multi_tensor_scale_impl = amp_C.multi_tensor_scale

# key to check if the video data is normalized or image data is converted to video data
# to avoid apply normalization or augment image dimension multiple times
# It is due to we do not have normalization and augment image dimension in the dataloader and move it to the model
IS_PREPROCESSED_KEY = "is_preprocessed"


def robust_broadcast(tensor: torch.Tensor, src: int, pg, is_check_shape: bool = False) -> torch.Tensor:
    """
    Perform a robust broadcast operation that works regardless of tensor shapes on different ranks.

    Args:
        tensor (torch.Tensor): The tensor to broadcast (on src rank) or receive (on other ranks).
        src (int): The source rank for the broadcast. Defaults to 0.

    Returns:
        torch.Tensor: The broadcasted tensor on all ranks.
    """
    # First, broadcast the shape of the tensor
    if distributed.get_rank() == src:
        shape = torch.tensor(tensor.shape).cuda()
    else:
        shape = torch.empty(tensor.dim(), dtype=torch.long).cuda()
    if is_check_shape:
        _verify_param_shape_across_processes(pg, [shape])
    torch.distributed.broadcast(shape, src, group=pg)

    # Resize the tensor on non-src ranks if necessary
    if distributed.get_rank() != src:
        tensor = tensor.new_empty(shape.tolist()).type_as(tensor)

    # Now broadcast the tensor data
    torch.distributed.broadcast(tensor, src, group=pg)

    return tensor


def _broadcast(item: torch.Tensor | str | None, to_tp: bool = True, to_cp: bool = True) -> torch.Tensor | str | None:
    """
    Broadcast the item from the minimum rank in the specified group(s).
    Since global rank = tp_rank + cp_rank * tp_size + ...
    First broadcast in the tp_group and then in the cp_group will
    ensure that the item is broadcasted across ranks in cp_group and tp_group.

    Parameters:
    - item: The item to broadcast (can be a torch.Tensor, str, or None).
    - to_tp: Whether to broadcast to the tensor model parallel group.
    - to_cp: Whether to broadcast to the context parallel group.
    """
    if not parallel_state.is_initialized():
        return item
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()

    to_tp = to_tp and parallel_state.get_tensor_model_parallel_world_size() > 1
    to_cp = to_cp and parallel_state.get_context_parallel_world_size() > 1

    if to_tp:
        min_tp_rank = min(get_process_group_ranks(tp_group))

    if to_cp:
        min_cp_rank = min(get_process_group_ranks(cp_group))

    if isinstance(item, torch.Tensor):  # assume the device is cuda
        # log.info(f"{item.shape}", rank0_only=False)
        if to_tp:
            # torch.distributed.broadcast(item, min_tp_rank, group=tp_group)
            item = robust_broadcast(item, min_tp_rank, tp_group)
        if to_cp:
            # torch.distributed.broadcast(item, min_cp_rank, group=cp_group)
            item = robust_broadcast(item, min_cp_rank, cp_group)
    elif item is not None:
        broadcastable_list = [item]
        if to_tp:
            # log.info(f"{broadcastable_list}", rank0_only=False)
            broadcast_object_list(broadcastable_list, min_tp_rank, group=tp_group)
        if to_cp:
            broadcast_object_list(broadcastable_list, min_cp_rank, group=cp_group)

        item = broadcastable_list[0]
    return item


def broadcast_condition(condition: BaseVideoCondition, to_tp: bool = True, to_cp: bool = True) -> BaseVideoCondition:
    condition_kwargs = {}
    for k, v in condition.to_dict().items():
        if isinstance(v, torch.Tensor):
            assert not v.requires_grad, f"{k} requires gradient. the current impl does not support it"
        condition_kwargs[k] = _broadcast(v, to_tp=to_tp, to_cp=to_cp)
    condition = type(condition)(**condition_kwargs)
    return condition


class DiffusionModel(ImageModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize trained_data_record with defaultdict, key: image, video, iteration
        self.trained_data_record = {
            "image": 0,
            "video": 0,
            "iteration": 0,
        }
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.state_shape[1])
        else:
            self.video_noise_multiplier = 1.0

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert (
            is_image != is_video
        ), "Only one of the input_image_key or input_data_key should be present in the data_batch."
        return is_image

    def draw_training_sigma_and_epsilon(self, size: int, condition: BaseVideoCondition) -> Tensor:
        sigma_B, epsilon = super().draw_training_sigma_and_epsilon(size, condition)
        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B = _broadcast(sigma_B * multiplier, to_tp=True, to_cp=is_video_batch)
        epsilon = _broadcast(epsilon, to_tp=True, to_cp=is_video_batch)
        return sigma_B, epsilon

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        save generated videos
        """
        raw_data, x0, condition = self.get_data_and_condition(data)
        guidance = data["guidance"]
        data = misc.to(data, **self.tensor_kwargs)
        sample = self.generate_samples_from_batch(
            data,
            guidance=guidance,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
        )
        sample = self.decode(sample)
        gt = raw_data
        caption = data["ai_caption"]
        return {"gt": gt, "result": sample, "caption": caption}, torch.tensor([0]).to(**self.tensor_kwargs)

    def training_step(self, data_batch: Dict[str, Tensor], iteration: int) -> Tuple[Dict[str, Tensor] | Tensor]:
        input_key = self.input_data_key  # by default it is video key
        if self.is_image_batch(data_batch):
            input_key = self.input_image_key
        batch_size = data_batch[input_key].shape[0]
        self.trained_data_record["image" if self.is_image_batch(data_batch) else "video"] += (
            batch_size * self.data_parallel_size
        )
        self.trained_data_record["iteration"] += 1
        return super().training_step(data_batch, iteration)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["trained_data_record"] = self.trained_data_record
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if "trained_data_record" in state_dict and hasattr(self, "trained_data_record"):
            trained_data_record = state_dict.pop("trained_data_record")
            if trained_data_record:
                assert set(trained_data_record.keys()) == set(self.trained_data_record.keys())
                for k, v in trained_data_record.items():
                    self.trained_data_record[k] = v
        else:
            log.warning("trained_data_record not found in the state_dict.")
        return super().load_state_dict(state_dict, strict, assign)

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
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
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert (
                    data_batch[input_key].shape[2] == 1
                ), f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def get_data_and_condition(self, data_batch: dict[str, Tensor]) -> Tuple[Tensor, BaseVideoCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        input_key = self.input_data_key  # by default it is video key
        is_image_batch = self.is_image_batch(data_batch)
        is_video_batch = not is_image_batch

        # Broadcast data and condition across TP and CP groups.
        # sort keys to make sure the order is same, IMPORTANT! otherwise, nccl will hang!
        local_keys = sorted(list(data_batch.keys()))
        # log.critical(f"all keys {local_keys}", rank0_only=False)
        for key in local_keys:
            data_batch[key] = _broadcast(data_batch[key], to_tp=True, to_cp=is_video_batch)

        if is_image_batch:
            input_key = self.input_image_key

        # Latent state
        raw_state = data_batch[input_key]
        latent_state = self.encode(raw_state).contiguous()

        # Condition
        condition = self.conditioner(data_batch)
        if is_image_batch:
            condition.data_type = DataType.IMAGE
        else:
            condition.data_type = DataType.VIDEO

        # VAE has randomness. CP/TP group should have the same encoded output.

        latent_state = _broadcast(latent_state, to_tp=True, to_cp=is_video_batch)
        condition = broadcast_condition(condition, to_tp=True, to_cp=is_video_batch)

        return raw_state, latent_state, condition

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format)
        if parallel_state.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
            sequence_parallel = getattr(parallel_state, "sequence_parallel", False)
            if sequence_parallel:
                self.net.enable_sequence_parallel()

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
                    x0 = split_inputs_cp(x=x0, seq_dim=2, cp_group=self.net.cp_group)
                    epsilon = split_inputs_cp(x=epsilon, seq_dim=2, cp_group=self.net.cp_group)

        output_batch, kendall_loss, pred_mse, edm_loss = super().compute_loss_with_epsilon_and_sigma(
            data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
        )
        if not self.is_image_batch(data_batch):
            if self.loss_reduce == "sum" and parallel_state.get_context_parallel_world_size() > 1:
                kendall_loss *= parallel_state.get_context_parallel_world_size()

        return output_batch, kendall_loss, pred_mse, edm_loss

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

        to_cp = self.net.is_context_parallel_enabled
        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        return_noise: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
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
            return_noise (bool): return the initial noise or not, used for ODE pairs generation
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        if n_sample is None:
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            if is_image_batch:
                state_shape = (self.state_shape[0], 1, *self.state_shape[2:])  # C,T,H,W

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.sde.sigma_max
        )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)

        samples = self.sampler(
            x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max, solver_option=solver_option
        )
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        if return_noise:
            if self.net.is_context_parallel_enabled:
                x_sigma_max = cat_outputs_cp(x_sigma_max, seq_dim=2, cp_group=self.net.cp_group)
            return samples, x_sigma_max / self.sde.sigma_max

        return samples

    def on_after_backward(self, iteration: int = 0):
        finalize_model_grads([self])

    def get_grad_norm(
        self,
        norm_type: Union[int, float] = 2,
        filter_fn: Callable[[str, torch.nn.Parameter], bool] | None = None,
    ) -> float:
        """Calculate the norm of gradients, handling model parallel parameters.

        This function is adapted from torch.nn.utils.clip_grad.clip_grad_norm_
        with added functionality to handle model parallel parameters.

        Args:
            norm_type (float or int): Type of norm to use. Can be 2 for L2 norm.
                'inf' for infinity norm is not supported.
            filter_fn (callable, optional): Function to filter parameters for norm calculation.
                Takes parameter name and parameter as input, returns True if this parameter is sharded else False.

        Returns:
            float: Total norm of the parameters (viewed as a single vector).

        Note:
            - Uses NVIDIA's multi-tensor applier for efficient norm calculation.
            - Handles both model parallel and non-model parallel parameters separately.
            - Currently only supports L2 norm (norm_type = 2).
        """
        # Get model parallel group if parallel state is initialized
        if parallel_state.is_initialized():
            model_parallel_group = parallel_state.get_model_parallel_group()
        else:
            model_parallel_group = None

        # Default filter function to identify tensor parallel parameters
        if filter_fn is None:

            def is_tp(name, param):
                return (
                    any(key in name for key in ["to_q.0", "to_k.0", "to_v.0", "to_out.0", "layer1", "layer2"])
                    and "_extra_state" not in name
                )

            filter_fn = is_tp

        # Separate gradients into model parallel and non-model parallel
        without_mp_grads_for_norm = []
        with_mp_grads_for_norm = []
        for name, param in self.named_parameters():
            if param.grad is not None:
                if filter_fn(name, param):
                    with_mp_grads_for_norm.append(param.grad.detach())
                else:
                    without_mp_grads_for_norm.append(param.grad.detach())

        # Only L2 norm is currently supported
        if norm_type != 2.0:
            raise NotImplementedError(f"Norm type {norm_type} is not supported. Only L2 norm (2.0) is implemented.")

        # Calculate L2 norm using NVIDIA's multi-tensor applier
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Calculate norm for non-model parallel gradients
        without_mp_grad_norm = torch.tensor([0], dtype=torch.float, device="cuda")
        if without_mp_grads_for_norm:
            without_mp_grad_norm, _ = multi_tensor_applier(
                l2_norm_impl,
                dummy_overflow_buf,
                [without_mp_grads_for_norm],
                False,  # no per-parameter norm
            )

        # Calculate norm for model parallel gradients
        with_mp_grad_norm = torch.tensor([0], dtype=torch.float, device="cuda")
        if with_mp_grads_for_norm:
            with_mp_grad_norm, _ = multi_tensor_applier(
                l2_norm_impl,
                dummy_overflow_buf,
                [with_mp_grads_for_norm],
                False,  # no per-parameter norm
            )

        # Square the norms as we'll be summing across model parallel GPUs
        total_without_mp_norm = without_mp_grad_norm**2
        total_with_mp_norm = with_mp_grad_norm**2

        # Sum across all model-parallel GPUs
        torch.distributed.all_reduce(total_with_mp_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)

        # Combine norms from model parallel and non-model parallel gradients
        total_norm = (total_with_mp_norm.item() + total_without_mp_norm.item()) ** 0.5

        return total_norm

    def clip_grad_norm_(self, max_norm: float):
        """
        This function performs gradient clipping to prevent exploding gradients.
        It calculates the total norm of the gradients, and if it exceeds the
        specified max_norm, scales the gradients down proportionally.

        Args:
            max_norm (float): The maximum allowed norm for the gradients.

        Returns:
            torch.Tensor: The total norm of the gradients before clipping.

        Note:
            This implementation uses NVIDIA's multi-tensor applier for efficiency.
        """
        # Collect gradients from all parameters that require gradients
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach())

        # Calculate the total norm of the gradients
        total_norm = self.get_grad_norm()

        # Compute the clipping coefficient
        clip_coeff = max_norm / (total_norm + 1.0e-6)

        # Apply gradient clipping if the total norm exceeds max_norm
        if clip_coeff < 1.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
            # Apply the scaling to the gradients using multi_tensor_applier for efficiency
            multi_tensor_applier(multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff)

        return torch.tensor([total_norm])


def _allreduce_layernorm_grads(model: List[torch.nn.Module]):
    """
    All-reduce the following layernorm grads:
    - When tensor parallel is enabled, all-reduce grads of QK-layernorm
    - When sequence parallel, all-reduce grads of AdaLN, t_embedder, additional_timestamp_embedder,
    and affline_norm.
    """
    sequence_parallel = getattr(parallel_state, "sequence_parallel", False)

    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        grads = []
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if not param.requires_grad:
                    continue

                if "to_q.1" in name or "to_k.1" in name:  # TP  # Q-layernorm  # K-layernorm
                    # grad = param.main_grad
                    grad = param.grad
                    if grad is not None:
                        grads.append(grad.data)

                if sequence_parallel:  # TP + SP
                    if (
                        "t_embedder" in name
                        or "adaLN_modulation" in name
                        or "additional_timestamp_embedder" in name
                        or "affline_norm" in name
                        or "input_hint_block" in name
                        or "zero_blocks" in name
                    ):
                        # grad = param.main_grad
                        grad = param.grad
                        if grad is not None:
                            grads.append(grad.data)

        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)


def finalize_model_grads(model: List[torch.nn.Module]):
    """
    All-reduce layernorm grads for tensor/sequence parallelism.
    Reference implementation: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/finalize_model_grads.py#L99
    """

    _allreduce_layernorm_grads(model)


@diffusion_fsdp_class_decorator
class FSDPDiffusionModel(DiffusionModel):
    pass
