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
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules import Module

from cosmos_predict1.diffusion.training.module.pretrained_vae_base import JITVAE, BaseVAE, StateDictVAE
from cosmos_predict1.utils import log


class VideoTokenizerInterface(ABC):
    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        pass

    @property
    @abstractmethod
    def spatial_compression_factor(self):
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self):
        pass

    @property
    @abstractmethod
    def spatial_resolution(self):
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        pass

    @property
    def is_chunk_overlap(self):
        return False


class BasePretrainedVideoTokenizer(ABC):
    """
    Base class for a pretrained video tokenizer that handles chunking of video data for efficient processing.

    Args:
        pixel_chunk_duration (int): The duration (in number of frames) of each chunk of video data at the pixel level.
        temporal_compress_factor (int): The factor by which the video data is temporally compressed during processing.
        max_enc_batch_size (int): The maximum batch size to process in one go during encoding to avoid memory overflow.
        max_dec_batch_size (int): The maximum batch size to process in one go during decoding to avoid memory overflow.

    The class introduces parameters for managing temporal chunks (`pixel_chunk_duration` and `temporal_compress_factor`)
    which define how video data is subdivided and compressed during the encoding and decoding processes. The
    `max_enc_batch_size` and `max_dec_batch_size` parameters allow processing in smaller batches to handle memory
    constraints.
    """

    def __init__(
        self,
        pixel_chunk_duration: int = 17,
        temporal_compress_factor: int = 8,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
    ):
        self._pixel_chunk_duration = pixel_chunk_duration
        self._temporal_compress_factor = temporal_compress_factor
        self.max_enc_batch_size = max_enc_batch_size
        self.max_dec_batch_size = max_dec_batch_size

    def register_mean_std(self, mean_std_fp: str) -> None:
        latent_mean, latent_std = torch.load(mean_std_fp, map_location="cuda", weights_only=True)
        latent_mean = latent_mean.view(self.latent_ch, -1)[:, : self.latent_chunk_duration]
        latent_std = latent_std.view(self.latent_ch, -1)[:, : self.latent_chunk_duration]

        target_shape = [1, self.latent_ch, self.latent_chunk_duration, 1, 1]

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

    def transform_encode_state_shape(self, state: torch.Tensor) -> torch.Tensor:
        """
        Rearranges the input state tensor to the required shape for encoding video data. Mainly for chunk based encoding
        """
        B, C, T, H, W = state.shape
        assert (
            T % self.pixel_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {self.pixel_chunk_duration}"
        return rearrange(state, "b c (n t) h w -> (b n) c t h w", t=self.pixel_chunk_duration)

    def transform_decode_state_shape(self, latent: torch.Tensor) -> None:
        B, _, T, _, _ = latent.shape
        assert (
            T % self.latent_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {self.latent_chunk_duration}"
        return rearrange(latent, "b c (n t) h w -> (b n) c t h w", t=self.latent_chunk_duration)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if self._temporal_compress_factor == 1:
            _, _, origin_T, _, _ = state.shape
            state = rearrange(state, "b c t h w -> (b t) c 1 h w")
        B, C, T, H, W = state.shape
        state = self.transform_encode_state_shape(state)
        # use max_enc_batch_size to avoid OOM
        if state.shape[0] > self.max_enc_batch_size:
            latent = []
            for i in range(0, state.shape[0], self.max_enc_batch_size):
                latent.append(super().encode(state[i : i + self.max_enc_batch_size]))
            latent = torch.cat(latent, dim=0)
        else:
            latent = super().encode(state)

        latent = rearrange(latent, "(b n) c t h w -> b c (n t) h w", b=B)
        if self._temporal_compress_factor == 1:
            latent = rearrange(latent, "(b t) c 1 h w -> b c t h w", t=origin_T)
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch of latent representations into video frames by applying temporal chunking. Similar to encode,
        it handles video data by processing smaller temporal chunks to reconstruct the original video dimensions.

        It can also decode single frame image data.

        Args:
            latent (torch.Tensor): The latent space tensor containing encoded video data.

        Returns:
            torch.Tensor: The decoded video tensor reconstructed from latent space.
        """
        if self._temporal_compress_factor == 1:
            _, _, origin_T, _, _ = latent.shape
            latent = rearrange(latent, "b c t h w -> (b t) c 1 h w")
        B, _, T, _, _ = latent.shape
        latent = self.transform_decode_state_shape(latent)
        # use max_enc_batch_size to avoid OOM
        if latent.shape[0] > self.max_dec_batch_size:
            state = []
            for i in range(0, latent.shape[0], self.max_dec_batch_size):
                state.append(super().decode(latent[i : i + self.max_dec_batch_size]))
            state = torch.cat(state, dim=0)
        else:
            state = super().decode(latent)
        assert state.shape[2] == self.pixel_chunk_duration
        state = rearrange(state, "(b n) c t h w -> b c (n t) h w", b=B)
        if self._temporal_compress_factor == 1:
            return rearrange(state, "(b t) c 1 h w -> b c t h w", t=origin_T)
        return state

    @property
    def pixel_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        # return self._latent_chunk_duration
        assert (self.pixel_chunk_duration - 1) % self.temporal_compression_factor == 0, (
            f"Pixel chunk duration {self.pixel_chunk_duration} is not divisible by latent chunk duration "
            f"{self.latent_chunk_duration}"
        )
        return (self.pixel_chunk_duration - 1) // self.temporal_compression_factor + 1

    @property
    def temporal_compression_factor(self):
        return self._temporal_compress_factor

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        assert (
            num_pixel_frames % self.pixel_chunk_duration == 0
        ), f"Temporal dimension {num_pixel_frames} is not divisible by chunk_length {self.pixel_chunk_duration}"
        return num_pixel_frames // self.pixel_chunk_duration * self.latent_chunk_duration

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        assert (
            num_latent_frames % self.latent_chunk_duration == 0
        ), f"Temporal dimension {num_latent_frames} is not divisible by chunk_length {self.latent_chunk_duration}"
        return num_latent_frames // self.latent_chunk_duration * self.pixel_chunk_duration


class VideoJITTokenizer(BasePretrainedVideoTokenizer, JITVAE, VideoTokenizerInterface):
    """
    Instance of BasePretrainedVideoVAE that loads encoder and decoder from JIT scripted module file
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_bf16: bool = True,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 8,
        pixel_chunk_duration: int = 17,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        spatial_resolution: str = "720",
    ):
        super().__init__(pixel_chunk_duration, temporal_compression_factor, max_enc_batch_size, max_dec_batch_size)
        super(BasePretrainedVideoTokenizer, self).__init__(enc_fp, dec_fp, name, mean_std_fp, latent_ch, False, is_bf16)

        self._spatial_compression_factor = spatial_compression_factor
        self._spatial_resolution = spatial_resolution

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def spatial_resolution(self) -> str:
        return self._spatial_resolution


class VideoStateDictTokenizer(BasePretrainedVideoTokenizer, StateDictVAE, VideoTokenizerInterface):
    """
    Instance of BasePretrainedVideoVAE that loads encoder and decoder from state_dict checkpoint
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        vae: torch.nn.Module,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_bf16: bool = True,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 8,
        pixel_chunk_duration: int = 17,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        spatial_resolution: str = "720",
    ):
        super().__init__(pixel_chunk_duration, temporal_compression_factor, max_enc_batch_size, max_dec_batch_size)
        super(BasePretrainedVideoTokenizer, self).__init__(
            enc_fp, dec_fp, vae, name, mean_std_fp, latent_ch, is_image=False, is_bf16=is_bf16
        )

        self._spatial_compression_factor = spatial_compression_factor
        self._spatial_resolution = spatial_resolution

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def spatial_resolution(self) -> str:
        return self._spatial_resolution


class VideoJITVAEChunkWiseTokenizer(VideoJITTokenizer):
    """
    Do temporal chunk wise encoding and decoding.
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        name: str,
        mean_std_fp: str,
        spatial_compression_factor: int,
        latent_ch: int = 16,
        is_bf16: bool = True,
        full_duration: int = 121,
        chunk_duration: int = 49,
        temporal_compression_factor: int = 8,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        spatial_resolution="720",
        overlap_size: int = 9,
    ):
        self._latent_chunk_duration = (
            chunk_duration - 1
        ) // temporal_compression_factor + 1  # need to set before super init
        self._latent_full_duration = (full_duration - 1) // temporal_compression_factor + 1
        super().__init__(
            enc_fp=enc_fp,
            dec_fp=dec_fp,
            name=name,
            mean_std_fp=mean_std_fp,
            latent_ch=latent_ch,
            is_bf16=is_bf16,
            pixel_chunk_duration=chunk_duration,
            temporal_compression_factor=temporal_compression_factor,
            max_enc_batch_size=max_enc_batch_size,
            max_dec_batch_size=max_dec_batch_size,
            spatial_resolution=spatial_resolution,
            spatial_compression_factor=spatial_compression_factor,
        )
        self.overlap_size = overlap_size
        self.full_duration = full_duration
        # make sure full_duration is divisible by chunk_duration with pre-set overlap size
        assert (full_duration - overlap_size) % (chunk_duration - overlap_size) == 0

    @property
    def latent_chunk_duration(self) -> int:
        return self._latent_chunk_duration

    @property
    def latent_full_duration(self) -> int:
        return self._latent_full_duration

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        assert (
            num_pixel_frames % self.full_duration == 0
        ), f"Temporal dimension {num_pixel_frames} is not divisible by chunk_length {self.full_duration}"
        return num_pixel_frames // self.full_duration * self.latent_full_duration

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        assert (
            num_latent_frames % self.latent_full_duration == 0
        ), f"Temporal dimension {num_latent_frames} is not divisible by chunk_length {self.latent_full_duration}"
        return num_latent_frames // self.latent_full_duration * self.full_duration

    def transform_encode_state_shape(self, state: torch.Tensor) -> torch.Tensor:
        # This is a hack impl, should be improved later
        return state

    def transform_decode_state_shape(self, latent: torch.Tensor) -> torch.Tensor:
        # This is a hack impl, should be improved later
        return latent

    def _impl_encode(self, state: torch.Tensor) -> torch.Tensor:
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
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape

        assert state.shape[2] == self.full_duration

        # Calculate the number of overlapping windows/chunks
        # Each window has a duration of self.pixel_chunk_duration frames
        # The overlap between consecutive windows is self.overlap_size frames
        num_windows = (T - self.pixel_chunk_duration) // (self.pixel_chunk_duration - self.overlap_size)
        # Calculate the total number of frames covered by the windows
        num_windowed_frames = self.pixel_chunk_duration + num_windows * (self.pixel_chunk_duration - self.overlap_size)

        assert num_windowed_frames == T  # only handle case where number frames can be separated equally
        # Prepare a list to hold overlapping chunks of the input state
        pack_list = [state[:, :, : self.pixel_chunk_duration]] + [
            state[
                :,
                :,
                (ii + 1)
                * (self.pixel_chunk_duration - self.overlap_size) : (ii + 1)
                * (self.pixel_chunk_duration - self.overlap_size)
                + self.pixel_chunk_duration,
            ]
            for ii in range(num_windows)
        ]

        latent = self._impl_encode(torch.cat(pack_list, dim=0))
        latent = rearrange(latent, "(n b) c t h w -> n b c t h w", b=B)
        # Calculate the overlap size in the latent space, accounting for any temporal compression
        # For example, if the network downsamples temporally by a factor of 4, adjust the overlap accordingly
        overlap_latent = (self.overlap_size - 1) // self.temporal_compression_factor + 1
        # Concatenate the latent representations from each chunk/window
        # For the first chunk, include all latent frames
        # For subsequent chunks, exclude the overlapping latent frames at the beginning
        out = torch.cat([latent[0]] + [latent[i, :, :, overlap_latent:] for i in range(1, len(latent))], dim=2)
        return out

    @torch.no_grad()
    def maybe_pad_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Since the decoder expect the latent to be window_size + (window_size - decode_overlap_size) * N, we need to pad the latent to match the expected size
        Args:
            latent (torch.Tensor): [B, C, T, H, W]
        Returns:
            latent: torch.Tensor, the padded latent
            padding_t: int, the number of padding latent t
        """

        # Calculate the overlap size and window size in the latent space, considering any temporal compression
        decode_overlap_size = (self.overlap_size - 1) // self.temporal_compression_factor + 1
        # Calculate the number of windows/chunks for decoding
        window_size = (self.pixel_chunk_duration - 1) // self.temporal_compression_factor + 1
        B, C, current_latent_t, H, W = latent.shape

        if current_latent_t < window_size:
            # If the current latent tensor is smaller than the window size, pad it to the window size
            target_latent_t = window_size
        else:
            # Calculate the target latent frame number for decoding
            target_latent_t = window_size + math.ceil(
                (current_latent_t - window_size) / (window_size - decode_overlap_size)
            ) * (window_size - decode_overlap_size)

        padding_t = target_latent_t - current_latent_t
        if padding_t != 0:
            log.info(
                f"Padding latent from {current_latent_t} to {target_latent_t} for decoding purpose. current window_size: {window_size}, decode_overlap_size: {decode_overlap_size}"
            )
            padding = latent.new_zeros(B, C, padding_t, H, W)
            latent = torch.cat([latent, padding], dim=2).contiguous()
        return latent, padding_t

    @torch.no_grad()
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        state, padding_t = self.maybe_pad_latent(state)
        B, C, num_latents, H, W = state.shape

        # Calculate the overlap size and window size in the latent space, considering any temporal compression
        decode_overlap_size = (self.overlap_size - 1) // self.temporal_compression_factor + 1
        # Calculate the number of windows/chunks for decoding
        window_size = (self.pixel_chunk_duration - 1) // self.temporal_compression_factor + 1

        num_windows = (num_latents - window_size) // (window_size - decode_overlap_size) + 1
        decoded_frames = []
        # Start decoding with the initial window of latent frames
        current_state = state[:, :, :window_size]
        for i in range(num_windows):
            # Decode the current window to get the reconstructed frames
            window_frames = super().decode(current_state)
            decoded_frames.append(window_frames)
            # Re-encode the overlapping frames at the end of the decoded window to obtain the last latent frame
            # This is necessary due to the casual first frame design
            last_latent = self._impl_encode(window_frames[:, :, -self.overlap_size : -self.overlap_size + 1])[:, :, 0:1]
            # Calculate the start and end indices for the next chunk of latent frames
            start_idx = window_size + i * (window_size - decode_overlap_size) - decode_overlap_size + 1
            end_idx = start_idx + window_size - 1
            # Prepare the next state by concatenating the last latent frame with the next chunk of latent frames
            current_state = torch.cat([last_latent, state[:, :, start_idx:end_idx]], dim=2)
        # Remove overlapping frames (e.g., 17 frames) from all windows except the first one.
        for i in range(1, num_windows):
            decoded_frames[i] = decoded_frames[i][:, :, self.overlap_size :]
        video_tensor = torch.cat(decoded_frames, dim=2)
        return video_tensor

    @property
    def is_chunk_overlap(self):
        return True


class DebugMeanStdVideoJITVAE(VideoJITTokenizer):
    """
    A class for one
    """

    def register_mean_std(self, mean_std_fp: str) -> None:
        target_shape = [1, self.latent_ch, 1, 1, 1]
        self.register_buffer(
            "latent_mean",
            # latent_mean.to(self.dtype).reshape(*target_shape),
            torch.zeros(*target_shape, dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            torch.ones(*target_shape, dtype=self.dtype),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if T == 1:
            return JITVAE.encode(self, state)
        return super().encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B, _, T, _, _ = latent.shape
        if T == 1:
            return JITVAE.decode(self, latent)
        return super().decode(latent)


class DebugMeanStdVideoJITVAEChunkWiseTokenizer(VideoJITVAEChunkWiseTokenizer):
    def register_mean_std(self, mean_std_fp: str) -> None:
        target_shape = [1, self.latent_ch, 1, 1, 1]
        self.register_buffer(
            "latent_mean",
            # latent_mean.to(self.dtype).reshape(*target_shape),
            torch.zeros(*target_shape, dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            torch.ones(*target_shape, dtype=self.dtype),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if T == 1:
            return JITVAE.encode(self, state)
        return super().encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B, _, T, _, _ = latent.shape
        if T == 1:
            return JITVAE.decode(self, latent)
        return super().decode(latent)


class JointImageVideoTokenizer(BaseVAE, VideoTokenizerInterface):
    def __init__(
        self,
        image_vae: torch.nn.Module,
        video_vae: torch.nn.Module,
        name: str,
        latent_ch: int = 16,
        squeeze_for_image: bool = True,
    ):
        super().__init__(latent_ch, name)
        self.image_vae = image_vae
        self.video_vae = video_vae
        self.squeeze_for_image = squeeze_for_image

    def encode_image(self, state: torch.Tensor) -> torch.Tensor:
        if self.squeeze_for_image:
            return self.image_vae.encode(state.squeeze(2)).unsqueeze(2)
        return self.image_vae.encode(state)

    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        if self.squeeze_for_image:
            return self.image_vae.decode(latent.squeeze(2)).unsqueeze(2)
        return self.image_vae.decode(latent)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if T == 1:
            return self.encode_image(state)

        return self.video_vae.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = latent.shape
        if T == 1:
            return self.decode_image(latent)
        return self.video_vae.decode(latent)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.image_vae.reset_dtype()
        self.video_vae.reset_dtype()

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        return self.video_vae.get_latent_num_frames(num_pixel_frames)

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        return self.video_vae.get_pixel_num_frames(num_latent_frames)

    @property
    def spatial_compression_factor(self):
        return self.video_vae.spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self.video_vae.temporal_compression_factor

    @property
    def spatial_resolution(self) -> str:
        return self.video_vae.spatial_resolution

    @property
    def pixel_chunk_duration(self) -> int:
        return self.video_vae.pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.video_vae.latent_chunk_duration


class JointImageVideoSharedJITTokenizer(JointImageVideoTokenizer):
    """
    First version of the ImageVideoVAE trained with Fitsum.
    We have to use seperate mean and std for image and video due to non-causal nature of the model.
    """

    def __init__(self, image_vae: Module, video_vae: Module, name: str, latent_ch: int = 16):
        super().__init__(image_vae, video_vae, name, latent_ch, squeeze_for_image=False)
        assert isinstance(image_vae, JITVAE)
        assert isinstance(
            video_vae, VideoJITTokenizer
        ), f"video_vae should be an instance of VideoJITVAE, got {type(video_vae)}"
        # a hack to make the image_vae and video_vae share the same encoder and decoder
        self.image_vae.encoder = self.video_vae.encoder
        self.image_vae.decoder = self.video_vae.decoder


class JointImageVideoStateDictTokenizer(JointImageVideoTokenizer):
    """
    Copy of ImageVideoVAE1 that uses plain torch.nn.Module instead of JITed one so
    that it can be used witch torch.compile()
    """

    def __init__(self, image_vae: Module, video_vae: Module, name: str, latent_ch: int = 16):
        super().__init__(image_vae, video_vae, name, latent_ch, squeeze_for_image=False)

        assert isinstance(image_vae, StateDictVAE)
        assert isinstance(video_vae, VideoStateDictTokenizer)
        # a hack to make the image_vae and video_vae share the same encoder and decoder

        # nn.Module
        del self.image_vae.vae
        # Just method
        del self.image_vae.encoder
        # Just method
        del self.image_vae.decoder

        self.image_vae.vae = self.video_vae.vae
        self.image_vae.encoder = self.video_vae.encoder
        self.image_vae.decoder = self.video_vae.decoder


class DummyJointImageVideoTokenizer(BaseVAE, VideoTokenizerInterface):
    def __init__(
        self,
        name: str = "dummy_joint_image_video",
        pixel_ch: int = 3,
        latent_ch: int = 16,
        pixel_chunk_duration: int = 17,
        latent_chunk_duration: int = 3,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 8,
        spatial_resolution: str = "720",
    ):
        self.pixel_ch = pixel_ch
        self._pixel_chunk_duration = pixel_chunk_duration
        self._latent_chunk_duration = latent_chunk_duration
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        self._spatial_resolution = spatial_resolution
        super().__init__(latent_ch, name)

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self._temporal_compression_factor

    @property
    def spatial_resolution(self) -> str:
        return self._spatial_resolution

    @property
    def pixel_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self._latent_chunk_duration

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        assert (
            num_pixel_frames % self.pixel_chunk_duration == 0
        ), f"Temporal dimension {num_pixel_frames} is not divisible by chunk_length {self.pixel_chunk_duration}"
        return num_pixel_frames // self.pixel_chunk_duration * self.latent_chunk_duration

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        assert (
            num_latent_frames % self.latent_chunk_duration == 0
        ), f"Temporal dimension {num_latent_frames} is not divisible by chunk_length {self.latent_chunk_duration}"
        return num_latent_frames // self.latent_chunk_duration * self.pixel_chunk_duration

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if T == 1:
            state_B_T_C_H_W = F.interpolate(
                rearrange(state, "b c t h w -> b t c h w"),
                size=(self.latent_ch, H // self.spatial_compression_factor, W // self.spatial_compression_factor),
                mode="trilinear",
                align_corners=False,
            )
            return rearrange(state_B_T_C_H_W, "b t c h w -> b c t h w").contiguous()
        assert (
            T % self.pixel_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {self.pixel_chunk_duration}"
        num_frames = T // self.pixel_chunk_duration * self.latent_chunk_duration

        state_B_C_T_H_W = F.interpolate(
            state,
            size=(self.latent_ch, H // self.spatial_compression_factor, W // self.spatial_compression_factor),
            mode="trilinear",
            align_corners=False,
        )
        state_B_H_W_T_C = rearrange(state_B_C_T_H_W, "b c t h w -> b h w t c")
        state_B_H_W_T_C = F.interpolate(
            state_B_H_W_T_C,
            size=(W // self.spatial_compression_factor, num_frames, self.latent_ch),
            mode="trilinear",
            align_corners=False,
        )
        return rearrange(state_B_H_W_T_C, "b h w t c -> b c t h w").contiguous()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = latent.shape
        if T == 1:
            latent_B_T_C_H_W = F.interpolate(
                rearrange(latent, "b c t h w -> b t c h w"),
                size=(self.pixel_ch, H * self.spatial_compression_factor, W * self.spatial_compression_factor),
                mode="trilinear",
                align_corners=False,
            )
            return rearrange(latent_B_T_C_H_W, "b t c h w -> b c t h w").contiguous()

        assert (
            T % self.latent_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {self.latent_chunk_duration}"
        num_frames = T * self.pixel_chunk_duration // self.latent_chunk_duration

        latent_B_H_W_T_C = rearrange(latent, "b c t h w -> b h w t c")
        latent_B_H_W_T_C = F.interpolate(
            latent_B_H_W_T_C,
            size=(W * self.spatial_compression_factor, num_frames, self.pixel_ch),
            mode="trilinear",
            align_corners=False,
        )
        latent_B_C_T_H_W = rearrange(latent_B_H_W_T_C, "b h w t c -> b c t h w")

        state = F.interpolate(
            latent_B_C_T_H_W,
            size=(num_frames, H * self.spatial_compression_factor, W * self.spatial_compression_factor),
            mode="trilinear",
            align_corners=False,
        )

        return state.contiguous()
