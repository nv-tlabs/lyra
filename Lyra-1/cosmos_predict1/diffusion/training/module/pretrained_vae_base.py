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

import os
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from cosmos_predict1.utils.distributed import rank0_first
from cosmos_predict1.utils.misc import load_from_s3_with_cache


class BaseVAE(torch.nn.Module, ABC):
    """
    Abstract base class for a Variational Autoencoder (VAE).

    All subclasses should implement the methods to define the behavior for encoding
    and decoding, along with specifying the latent channel size.
    """

    def __init__(self, channel: int = 3, name: str = "vae"):
        super().__init__()
        self.channel = channel
        self.name = name

    @property
    def latent_ch(self) -> int:
        """
        Returns the number of latent channels in the VAE.
        """
        return self.channel

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into a latent representation.

        Args:
        - state (torch.Tensor): The input tensor to encode.

        Returns:
        - torch.Tensor: The encoded latent tensor.
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to the original space.

        Args:
        - latent (torch.Tensor): The latent tensor to decode.

        Returns:
        - torch.Tensor: The decoded tensor.
        """
        pass

    @property
    def spatial_compression_factor(self) -> int:
        """
        Returns the spatial reduction factor for the VAE.
        """
        raise NotImplementedError("The spatial_compression_factor property must be implemented in the derived class.")


class BasePretrainedImageVAE(BaseVAE):
    """
    A base class for pretrained Variational Autoencoder (VAE) that loads mean and standard deviation values
    from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.
    Derived classes should load pre-trained encoder and decoder components from a remote store

    Attributes:
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ) -> None:
        super().__init__(latent_ch, name)
        dtype = torch.bfloat16 if is_bf16 else torch.float32
        self.dtype = dtype
        self.is_image = is_image
        self.mean_std_fp = mean_std_fp
        self.name = name

        self.backend_args = None

        self.register_mean_std(mean_std_fp)

    def register_mean_std(self, mean_std_fp: str) -> None:
        latent_mean, latent_std = torch.load(mean_std_fp, map_location="cuda", weights_only=True)
        target_shape = [1, self.latent_ch, 1, 1] if self.is_image else [1, self.latent_ch, 1, 1, 1]
        self.register_buffer(
            "latent_mean",
            latent_mean.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the input state to latent space; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = state.dtype
        latent_mean = self.latent_mean.to(in_dtype)
        latent_std = self.latent_std.to(in_dtype)
        encoded_state = self.encoder(state.to(self.dtype))
        if isinstance(encoded_state, torch.Tensor):
            pass
        elif isinstance(encoded_state, tuple):
            assert isinstance(encoded_state[0], torch.Tensor)
            encoded_state = encoded_state[0]
        else:
            raise ValueError("Invalid type of encoded state")
        return (encoded_state.to(in_dtype) - latent_mean) / latent_std

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode the input latent to state; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = latent.dtype
        latent = latent * self.latent_std.to(in_dtype) + self.latent_mean.to(in_dtype)
        return self.decoder(latent.to(self.dtype)).to(in_dtype)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.decoder.to(self.dtype)
        self.encoder.to(self.dtype)


class JITVAE(BasePretrainedImageVAE):
    """
    A JIT compiled Variational Autoencoder (VAE) that loads pre-trained encoder
    and decoder components from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The JIT compiled encoder loaded from storage.
        decoder (Module): The JIT compiled decoder loaded from storage.
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        name (str): Name of the model, used for differentiating cache file paths.
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ):
        super().__init__(name, mean_std_fp, latent_ch, is_image, is_bf16)
        self.load_encoder(enc_fp)
        self.load_decoder(dec_fp)

    def load_encoder(self, enc_fp: str) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - enc_fp (str): File path to the encoder's JIT file on the remote store.
        """
        self.encoder = torch.jit.load(enc_fp, map_location="cuda")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.dtype)

    def load_decoder(self, dec_fp: str) -> None:
        """
        Load the decoder from the remote store.

        Args:
        - dec_fp (str): File path to the decoder's JIT file on the remote store.
        """
        self.decoder = torch.jit.load(dec_fp, map_location="cuda")
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)


class StateDictVAE(BasePretrainedImageVAE):
    """
    A Variational Autoencoder (VAE) that loads pre-trained weights into
    provided encoder and decoder components from a remote store, handles data type conversions,
    and normalization using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The encoder with weights loaded from storage.
        decoder (Module): The decoder with weights loaded from storage.
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        vae (Module): Instance of VAE with not loaded weights
        name (str): Name of the model, used for differentiating cache file paths.
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        vae: torch.nn.Module,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ):
        super().__init__(name, mean_std_fp, latent_ch, is_image, is_bf16)

        self.load_encoder_and_decoder(enc_fp, dec_fp, vae)

    def load_encoder_and_decoder(self, enc_fp: str, dec_fp: str, vae: torch.nn.Module) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - vae_fp (str): File path to the vae's state dict file on the remote store.
        - vae (str): VAE module into which weights will be loaded.
        """
        state_dict_enc = load_from_s3_with_cache(
            enc_fp,
            f"vae/{self.name}_enc.jit",
            easy_io_kwargs={"map_location": torch.device(torch.cuda.current_device())},
            backend_args=self.backend_args,
        )

        state_dict_dec = load_from_s3_with_cache(
            dec_fp,
            f"vae/{self.name}_dec.jit",
            easy_io_kwargs={"map_location": torch.device(torch.cuda.current_device())},
            backend_args=self.backend_args,
        )

        jit_weights_state_dict = state_dict_enc.state_dict() | state_dict_dec.state_dict()
        jit_weights_state_dict = {
            k: v
            for k, v in jit_weights_state_dict.items()
            # Global variables captured by JIT
            if k
            not in (
                "encoder.patcher.wavelets",
                "encoder.patcher._arange",
                "decoder.unpatcher.wavelets",
                "decoder.unpatcher._arange",
            )
        }

        vae.load_state_dict(jit_weights_state_dict)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        vae.to(self.dtype)

        self.vae = vae
        self.encoder = self.vae.encode
        self.decoder = self.vae.decode

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.vae.to(self.dtype)


class SDVAE(BaseVAE):
    def __init__(self, batch_size=16, count_std: bool = False, is_downsample: bool = True) -> None:
        super().__init__(channel=4, name="sd_vae")
        self.dtype = torch.bfloat16
        self.register_buffer(
            "scale",
            torch.tensor([4.17, 4.62, 3.71, 3.28], dtype=self.dtype).reciprocal().reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            -1.0 * torch.tensor([5.81, 3.25, 0.12, -2.15], dtype=self.dtype).reshape(1, -1, 1, 1) * self.scale,
            persistent=False,
        )
        self.batch_size = batch_size
        self.count_std = count_std
        self.is_downsample = is_downsample
        self.load_vae()
        self.reset_dtype()

    def reset_dtype(self, *args, **kwargs):
        del args, kwargs
        self.vae.to(self.dtype)

    @rank0_first
    def load_vae(self) -> None:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import diffusers

        vae_name = "stabilityai/sd-vae-ft-mse"
        try:
            vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, local_files_only=True)
        except:  # noqa: E722
            # Could not load the model from cache; try without local_files_only.
            vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name)
        self.vae = vae.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        state : pixel range [-1, 1]
        """
        if self.is_downsample:
            _h, _w = state.shape[-2:]
            state = F.interpolate(state, size=(_h // 2, _w // 2), mode="bilinear", align_corners=False)
        in_dtype = state.dtype
        state = state.to(self.dtype)
        state = (state + 1.0) / 2.0
        latent_dist = self.vae.encode(state)["latent_dist"]
        mean, std = latent_dist.mean, latent_dist.std
        if self.count_std:
            latent = mean + torch.randn_like(mean) * std
        else:
            latent = mean
        latent = latent * self.scale
        latent = latent + self.bias
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        latent = latent.to(self.dtype)
        latent = latent - self.bias
        latent = latent / self.scale
        latent = torch.cat([self.vae.decode(batch)["sample"] for batch in latent.split(self.batch_size)])
        if self.is_downsample:
            _h, _w = latent.shape[-2:]
            latent = F.interpolate(latent, size=(_h * 2, _w * 2), mode="bilinear", align_corners=False)
        return latent.to(in_dtype) * 2 - 1.0

    @property
    def spatial_compression_factor(self) -> int:
        return 8
