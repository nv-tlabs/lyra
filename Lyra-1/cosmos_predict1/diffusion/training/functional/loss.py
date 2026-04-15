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

from typing import Tuple, Union

import torch

from cosmos_predict1.diffusion.functional.batch_ops import batch_mul


def create_per_sample_loss_mask(
    loss_masking_cfg: dict,
    data_batch: dict,
    x_shape: Tuple[int],
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
):
    """
    Creates a per-sample loss mask based on the given configuration and input data batch.

    This function generates a dictionary of loss masks for each specified key in the loss masking configuration.
    For keys present in both the configuration and the data batch, the corresponding data batch value is used.
    For keys present only in the configuration, a tensor of zeros with the specified shape is created.
    Additionally, it computes loss mask weights for each key based on the configuration values and adjusts them
    based on the presence of certain keys in the data batch, such as "skip_face" and "object_loss_map".

    Note:
    - The original `loss_masking_cfg` and `data_batch` are not modified by this function.
    - For image data, it is assumed that the channel is always the first dimension.
    - `skip_face` is for face regions that should be skipped during training, the key is provided so that we can generate
    diverse human and avoid collapse to a single face given certain prompts. The issue happens for getty projects,
    where face distribution in the dataset is high unbalanced that single man face can be shown in more than 100+ images.

    Parameters:
        loss_masking_cfg (dict): Configuration for loss masking, specifying which keys to include and their weights.
        data_batch (dict): The batch of data containing actual data points and potential mask indicators like "skip_face".
        x_shape (tuple): The shape of the input data, used to initialize zero masks for keys not in the data batch.
        dtype (torch.dtype): The data type for the tensors in the loss masks.
        device (str, optional): The device on which to create the tensors. Defaults to 'cuda'.

    Returns:
        dict: A dictionary containing combined loss masks adjusted according to the `loss_masking_cfg` and `data_batch`.

    Raises:
        AssertionError: If "skip_face" is not present in `data_batch`.

    Note: `create_combined_loss_mask` is assumed to be a separate function that combines individual loss masks into a
    single mask or set of masks based on the given parameters. Its behavior should be documented separately.
    """
    loss_mask_data: dict = {}
    for key in loss_masking_cfg:
        if key not in data_batch:
            loss_mask_data[key] = torch.zeros((x_shape[0], 1, x_shape[2], x_shape[3]), device=device)
        else:
            loss_mask_data[key] = data_batch[key]

    if "skip_face" not in data_batch:
        # When skip_face is not there in data_dict, use 0 as default. This will not skip any sample.
        data_batch["skip_face"] = torch.zeros((x_shape[0],), dtype=dtype, device=device)

    loss_mask_weight: dict = {}
    for k, v in loss_masking_cfg.items():
        loss_mask_weight[k] = torch.tensor(v, device=device).expand(data_batch["skip_face"].size())

    if "human_face_mask" in loss_mask_weight:
        loss_mask_weight["human_face_mask"] = (1 - data_batch["skip_face"]) * loss_mask_weight["human_face_mask"]

    if "object_loss_map" in data_batch:
        loss_mask_weight["object_loss_map"] = torch.ones(data_batch["object_loss_map"].shape[0], device=device)

    return create_combined_loss_mask(loss_mask_data, x_shape, dtype, device, loss_mask_weight)


def create_combined_loss_mask(data, x_shape, dtype, device="cuda", loss_masking=None):
    """
    Creates a combined loss mask from multiple input masks.

    This function combines several loss masks into a single mask. In regions where masks overlap,
    the highest value is assigned. Non-overlapping regions are assigned a default value of 1.
    Regions with a mask value of zero are explicitly zeroed out, which is essential for padded loss calculations.

    Example:
        Given the following masks and weights:
            mask1: [0, 1, 1, 1, 0, 0], weight: 2
            mask2: [1, 0, 1, 0, 0, 0], weight: 4
            mask3: [0, 1, 0, 0, 0, 0], weight: 0
        The resulting combined loss mask would be:
            [4, 0, 4, 2, 1, 1]

    Parameters:
        data (dict): Contains the loss masks and their weights.
        x_shape (tuple): The shape of the output mask.
        dtype: The data type for the output mask.
        device: The device on which the output mask will be allocated.
        loss_masking: The loss masking weight configuration.

    Returns:
        torch.Tensor: The combined loss mask.
    """

    loss_mask = torch.ones(x_shape, dtype=dtype, device=device)
    zero_mask = torch.ones(x_shape, dtype=dtype, device=device)

    if loss_masking:
        for key in loss_masking:
            # Repeat mask along channel's dimension. ndim=4 for images.
            repeat_dims = (1, x_shape[1]) + tuple([1] * (data[key].ndim - 2))
            mask_key = torch.tile(data[key], dims=repeat_dims)
            weight_key = loss_masking[key]

            # handle zero weight case
            is_zero_weight = (weight_key == 0).float()[:, None, None, None]
            zero_mask = zero_mask * (
                (1 - is_zero_weight) * torch.ones(x_shape, dtype=dtype, device=device)
                + is_zero_weight * (1 - mask_key.bool().float())
            )

            # calculate weights
            no_mask_region = (mask_key.bool() == 0).float()
            loss_mask = batch_mul(mask_key, weight_key) + batch_mul(no_mask_region, loss_mask)

    loss_mask_final = loss_mask * zero_mask
    return loss_mask_final
