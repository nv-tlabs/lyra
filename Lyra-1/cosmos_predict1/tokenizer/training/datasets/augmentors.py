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

"""Additional augmentors for image and video training loops."""

from typing import Any, Optional

import omegaconf
import torch
import torchvision.transforms.functional as transforms_F
from loguru import logger as logging

from cosmos_predict1.tokenizer.training.datasets.utils import obtain_augmentation_size, obtain_image_size
from cosmos_predict1.utils import log


class Augmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise ValueError("Augmentor not implemented")


class LossMask(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs data normalization.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        assert self.args is not None, "Please specify args"
        mask_config = self.args["masking"]

        input_key = self.input_keys[0]
        default_mask = torch.ones_like(data_dict[input_key])
        loss_mask = mask_config["nonhuman_mask"] * default_mask
        for curr_key in mask_config:
            if curr_key not in self.input_keys:
                continue
            curr_mask = data_dict[curr_key]
            curr_weight = mask_config[curr_key]
            curr_loss_mask = curr_mask * curr_weight + (1 - curr_mask) * loss_mask
            loss_mask = torch.max(curr_loss_mask, loss_mask)
            _ = data_dict.pop(curr_key)
        data_dict["loss_mask"] = loss_mask
        return data_dict


class UnsqueezeImage(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs horizontal flipping.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        for key in self.input_keys:
            data_dict[key] = data_dict[key].unsqueeze(1)

        return data_dict


class RandomReverse(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random temporal reversing of frames.

        Args:
            data_dict (dict): Input data dict, CxTxHxW
        Returns:
            data_dict (dict): Output dict where videos are randomly reversed.
        """
        assert self.args is not None
        p = self.args.get("prob", 0.5)
        coin_flip = torch.rand(1).item() <= p
        for key in self.input_keys:
            if coin_flip:
                data_dict[key] = torch.flip(data_dict[key], dims=[1])

        return data_dict


class RenameInputKeys(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Rename the input keys from the data dict.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with keys renamed.
        """
        assert len(self.input_keys) == len(self.output_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            if input_key in data_dict:
                data_dict[output_key] = data_dict.pop(input_key)
        return data_dict


class CropResizeAugmentor(Augmentor):
    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        crop_args: Optional[dict] = None,
        resize_args: Optional[dict] = None,
        args: Optional[dict] = None,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.crop_args = crop_args
        self.resize_args = resize_args
        self.crop_op = RandomCrop(input_keys, output_keys, crop_args)
        self.resize_op = ResizeSmallestSideAspectPreserving(input_keys, output_keys, resize_args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random temporal reversing of frames.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where videso are randomly reversed.
        """
        assert self.args is not None
        p = self.args.get("prob", 0.1)

        if p > 0.0:
            crop_img_size = obtain_augmentation_size(data_dict, self.crop_args)
            crop_width, crop_height = crop_img_size
            orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
            if orig_w < crop_width or orig_h < crop_height:
                log.warning(
                    f"Data size ({orig_w}, {orig_h}) is smaller than crop size ({crop_width}, {crop_height}), skip the crop augmentation."
                )
            coin_flip = torch.rand(1).item() <= p
            if coin_flip and crop_width <= orig_w and crop_height <= orig_h:
                data_dict = self.crop_op(data_dict)
                return data_dict

        data_dict = self.resize_op(data_dict)
        data_dict = self.crop_op(data_dict)

        return data_dict


class CenterCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs center crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("size" in self.args), "Please specify size in args"

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        for key in self.input_keys:
            data_dict[key] = transforms_F.center_crop(data_dict[key], [height, width])

        # We also add the aug params we use. This will be useful for other transforms
        crop_x0 = (orig_w - width) // 2
        crop_y0 = (orig_h - height) // 2
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        data_dict["padding_mask"] = torch.zeros((1, cropping_params["crop_h"], cropping_params["crop_w"]))
        return data_dict


class RandomCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        # Obtaining random crop coords
        try:
            crop_x0 = int(torch.randint(0, orig_w - width + 1, size=(1,)).item())
            crop_y0 = int(torch.randint(0, orig_h - height + 1, size=(1,)).item())
        except Exception as e:
            logging.warning(
                f"Random crop failed. Performing center crop, original_size(wxh): {orig_w}x{orig_h}, random_size(wxh): {width}x{height}"
            )
            for key in self.input_keys:
                data_dict[key] = transforms_F.center_crop(data_dict[key], [height, width])
            crop_x0 = (orig_w - width) // 2
            crop_y0 = (orig_h - height) // 2

        # We also add the aug params we use. This will be useful for other transforms
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params

        # We must perform same random cropping for all input keys
        for key in self.input_keys:
            data_dict[key] = transforms_F.crop(data_dict[key], crop_y0, crop_x0, height, width)
        return data_dict


class HorizontalFlip(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs horizontal flipping.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        flip_enabled = getattr(self.args, "enabled", True)
        if flip_enabled:
            p = getattr(self.args, "prob", 0.5)
            coin_flip = torch.rand(1).item() > p
            for key in self.input_keys:
                if coin_flip:
                    data_dict[key] = transforms_F.hflip(data_dict[key])

        return data_dict


class Normalize(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs data normalization.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        for key in self.input_keys:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(dtype=torch.get_default_dtype()).div(255)
            else:
                data_dict[key] = transforms_F.to_tensor(data_dict[key])  # division by 255 is applied in to_tensor()

            data_dict[key] = transforms_F.normalize(tensor=data_dict[key], mean=mean, std=std)
        return data_dict


class ReflectionPadding(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs reflection padding. This function also returns a padding mask.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """

        assert self.args is not None, "Please specify args in augmentation"
        if self.output_keys is None:
            self.output_keys = self.input_keys

        # Obtain image and augmentation sizes
        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        target_size = obtain_augmentation_size(data_dict, self.args)

        assert isinstance(target_size, (tuple, omegaconf.listconfig.ListConfig)), "Please specify target size as tuple"
        target_w, target_h = target_size

        target_w = int(target_w)
        target_h = int(target_h)

        # Calculate padding vals
        padding_left = int((target_w - orig_w) / 2)
        padding_right = target_w - orig_w - padding_left
        padding_top = int((target_h - orig_h) / 2)
        padding_bottom = target_h - orig_h - padding_top
        padding_vals = [padding_left, padding_top, padding_right, padding_bottom]

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            if max(padding_vals[0], padding_vals[2]) >= orig_w or max(padding_vals[1], padding_vals[3]) >= orig_h:
                # In this case, we can't perform reflection padding. This is because padding values
                # are larger than the image size. So, perform edge padding instead.
                data_dict[out_key] = transforms_F.pad(data_dict[inp_key], padding_vals, padding_mode="edge")
            else:
                # Perform reflection padding
                data_dict[out_key] = transforms_F.pad(data_dict[inp_key], padding_vals, padding_mode="reflect")

            if out_key != inp_key:
                del data_dict[inp_key]

        # Return padding_mask when padding is performed.
        # Padding mask denotes which pixels are padded.
        padding_mask = torch.ones((1, target_h, target_w))
        padding_mask[:, padding_top : (padding_top + orig_h), padding_left : (padding_left + orig_w)] = 0
        data_dict["padding_mask"] = padding_mask
        data_dict["image_size"] = torch.tensor([target_h, target_w, orig_h, orig_w], dtype=torch.float)

        return data_dict


class ResizeSmallestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to smaller side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=out_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSide(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs resizing to larger side

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            out_size = obtain_augmentation_size(data_dict, self.args)
            assert isinstance(out_size, int), "Arg size in resize should be an integer"
            orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)

            scaling_ratio = min(out_size / orig_w, out_size / orig_h)
            target_size = [int(scaling_ratio * orig_h), int(scaling_ratio * orig_w)]

            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )
            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeSmallestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the smaller ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the smaller of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(
            img_size, (tuple, omegaconf.listconfig.ListConfig)
        ), f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert (
            target_size[0] >= img_h and target_size[1] >= img_w
        ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict


class ResizeLargestSideAspectPreserving(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs aspect-ratio preserving resizing.
        Image is resized to the dimension which has the larger ratio of (size / target_size).
        First we compute (w_img / w_target) and (h_img / h_target) and resize the image
        to the dimension that has the larger of these ratios.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are resized
        """

        if self.output_keys is None:
            self.output_keys = self.input_keys
        assert self.args is not None, "Please specify args in augmentations"

        img_size = obtain_augmentation_size(data_dict, self.args)
        assert isinstance(
            img_size, (tuple, omegaconf.listconfig.ListConfig)
        ), f"Arg size in resize should be a tuple, get {type(img_size)}, {img_size}"
        img_w, img_h = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert (
            target_size[0] <= img_h and target_size[1] <= img_w
        ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"

        for inp_key, out_key in zip(self.input_keys, self.output_keys):
            data_dict[out_key] = transforms_F.resize(
                data_dict[inp_key],
                size=target_size,  # type: ignore
                interpolation=getattr(self.args, "interpolation", transforms_F.InterpolationMode.BICUBIC),
                antialias=True,
            )

            if out_key != inp_key:
                del data_dict[inp_key]
        return data_dict
