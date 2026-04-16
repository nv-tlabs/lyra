# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from typing import Any, Optional, List, cast
import random
from statistics import NormalDist
import numpy as np
import torch
from einops import rearrange
import attrs
import gc
from lyra_2._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from lyra_2._ext.imaginaire.utils import log
from lyra_2._ext.imaginaire.utils import misc
from lyra_2._src.modules.conditioner import DataType, T2VCondition
from lyra_2._src.models.wan_t2v_model import WANDiffusionModel, T2VModelConfig
from megatron.core import parallel_state
from torch.distributed.tensor import DTensor
from torch.distributed._composable.fsdp import fully_shard
from lyra_2._src.utils.dtensor_helper import broadcast_dtensor_model_states
from lyra_2._src.utils.context_parallel import broadcast
from lyra_2._src.datasets.forward_warp_utils_pytorch import (
    unproject_points,
    forward_warp_multiframes,
)
from lyra_2._src.datasets.plucker_embed_corrupter import (
    ray_condition,
)


WAN2PT1_I2V_COND_LATENT_KEY = "i2v_WAN2PT1_cond_latents"
LYRA2_BUFFER_SINCOS_MULTIRES = 2
LYRA2_BUFFER_MLP_SQUEEZE_DIM = 256
LYRA2_CORRESPONDENCE_CHANNELS_PER_SLOT = 4 * 8 * 8


@attrs.define(slots=False)
class Lyra2T2VConfig(T2VModelConfig):
    """Configuration for Lyra2 spatial model"""
    init_framepack_weights: bool = True # disable this if resume training

    # Lyra2 AR configuration
    framepack_type: str = "f16k4f2k2f1k1_g3"  # fkfkfk_gN where N=new latent frames
    num_frames_per_latent: int = 4           # frames per latent tokenization (video tokenizer)
    max_segments: int = 10
    starting_frame_ratio: float = 0.1

    apply_corruption_to_spatial_region: str = "none"
    augment_sigma_sample_p_mean: float = 0.0
    augment_sigma_sample_p_std: float = 1.0
    augment_sigma_sample_multiplier: float = 1.0
    condition_video_augment_sigma_in_inference: float = 0.001

    # Stage-A self-augmentation (training-time short denoise) configuration
    self_aug_enabled: bool = False
    self_aug_steps: int = 5
    # Number of discrete inference-like timesteps
    self_aug_num_discrete_timesteps: int = 8
    self_aug_guidance: Optional[float] = None
    self_aug_scheduler_shift: Optional[float] = None
    self_aug_every_k: int = 1
    self_aug_prob: float = 1.0
    self_aug_max_T: int = 50
    self_aug_copy_chunk: bool = False
    self_aug_encode_gt_with_clean_history: bool = False
    self_aug_i2v_ratio: float = 0.3 # ensure i2v case is trained

    spatial_memory_stride: int = 8
    spatial_memory_skip_recent: int = 100
    spatial_memory_use_image: bool = False
    spatial_memory_dropout_rate: float = 0.1

    # Optional: comma-separated list of submodule names in Lyra2AttentionBlock to train.
    # If provided, all other parameters are frozen.
    framepack_trainable_modules: Optional[str] = None

    # This port is intentionally collapsed to the single target branch:
    # accumulated correspondence + multibuffer + depth-augmented slots + K/Q-only injection.
    # Drop spatial memory cache completely with this probability.
    spatial_memory_drop_rate: float = 0.1
    # Max number of spatial buffers to keep (pad with zeros if fewer are available).
    # If None, defaults to the number of spatial history slots from framepack_type.
    multibuffer_max_spatial_frames: Optional[int] = None
    warp_chunk_size: int = 2


class Lyra2Model(WANDiffusionModel):
    """Lyra2 spatial model"""

    def __init__(self, config: Lyra2T2VConfig):
        super().__init__(config)
        # Transient diagnostics/visualization cache controls and storage
        self._collect_return_condition_state: bool = False
        self._latest_condition_state_pixels = None
        self._latest_plucker_rays_pixels = None
        self._latest_gt_gen_pixels = None
        # Parse Lyra2 AR metadata
        self._init_lyra2_metadata()
        self.framepack_weights_initialized = False
        self._cached_spatial_coords: Optional[torch.Tensor] = None
        self._cached_spatial_coords_meta: Optional[tuple[int, int, int, torch.device, torch.dtype]] = None

        self._spatial_history_positions = self._compute_spatial_history_positions()
        log.info(
            f"Lyra2Model spatial history positions: {self._spatial_history_positions}, "
            f"spatial_memory_stride={self.config.spatial_memory_stride}, "
            f"spatial_memory_skip_recent={self.config.spatial_memory_skip_recent}",
            rank0_only=True,
        )

    def _compute_spatial_history_positions(self) -> tuple[int, ...]:
        positions: list[int] = []
        offset = 0
        for count, kernel_type in zip(self.framepack_clean_latent_frame_splits, self.framepack_clean_latent_frame_kernel_types):
            cnt = int(count)
            if kernel_type == "s":
                positions.extend(range(offset, offset + cnt))
            offset += cnt
        return tuple(positions)

    def _apply_spatial_region_corruption(self, latents: torch.Tensor, cond_latent: torch.Tensor) -> None:
        if len(self._spatial_history_positions) == 0:
            return
        spatial_idx = torch.tensor(self._spatial_history_positions, device=latents.device, dtype=torch.long)
        if spatial_idx.numel() == 0:
            return
        latents_slice = latents[:, :, spatial_idx]
        latents_corrupted, aug_sigma = self.augment_conditional_latent_frames(
            latents_slice,
            target_mode=self.config.apply_corruption_to_spatial_region,
        )
        latents[:, :, spatial_idx] = latents_corrupted
        cond_latent[:, :16, spatial_idx] = latents_corrupted

    def _prepare_video_window(self, video, start=None, cur_segment_id=None):
        """Step 1: crop/pad video to the current segment window and return bookkeeping.

        Returns:
            video_win: cropped/padded video on correct dtype/device
            video_indices: absolute frame indices into the original video timeline for each frame in video_win
            start: chosen start index
            cur_segment_id: chosen segment id
            chunk_len: number of frames in the window
            to_repeat_front: number of frames repeated at the front (for history padding)
        """
        video_length = int(video.shape[2])
        video_indices = torch.arange(video_length, device=video.device)
        cfg = self.config

        # Choose start
        if start is None:
            start = int(np.random.randint(0, max(1, int(video_length * self.framepack_starting_frame_ratio))))

        # Choose segment id
        if cur_segment_id is None:
            max_segments = (video_length - start - 1) // int(self.framepack_num_new_video_frames)
            if cfg.self_aug_enabled:
                max_segments = max_segments - 1
            max_segments = max(1, min(max_segments, int(self.framepack_max_segments)))
            cur_segment_id = int(np.random.randint(0, max_segments))

        # number of frames to keep
        chunk_len = (cur_segment_id + 1) * int(self.framepack_num_new_video_frames) + 1

        # Crop-right or pad-right by repeating the last available frame
        if start + chunk_len > video_length:
            video_win = video[:, :, start:]
            video_indices = video_indices[start:]
            to_repeat = start + chunk_len - video_length
            video_win = torch.cat([video_win, video_win[:, :, -1:].repeat(1, 1, to_repeat, 1, 1)], dim=2)
            video_indices = torch.cat([video_indices, video_indices[-1:].repeat(to_repeat)], dim=0)
        else:
            video_win = video[:, :, start:start + chunk_len]
            video_indices = video_indices[start:start + chunk_len]

        # Lyra2 first iter, condition is i iiii iiii ...
        to_repeat_front = (self.framepack_num_history_latent - 1) * int(self.framepack_num_frames_per_latent)
        if to_repeat_front > 0:
            video_win = torch.cat([video_win[:, :, :1].repeat(1, 1, to_repeat_front, 1, 1), video_win], dim=2)
            video_indices = torch.cat([video_indices[:1].repeat(to_repeat_front), video_indices], dim=0)

        video_win = video_win.to(dtype=self.tensor_kwargs["dtype"], device=self.tensor_kwargs["device"])

        if self._collect_return_condition_state:
            self._latest_gt_gen_pixels = video_win[:, :, -int(self.framepack_num_new_video_frames):].contiguous()

        return video_win, video_indices, int(start), int(cur_segment_id), int(chunk_len)

    def build_net(self):
        """Add clean patch embeddings before FSDP so they are sharded/initialized correctly."""
        config = self.config
        if config.use_mp_policy_fsdp:
            fsdp_kwargs = {"mp_policy": torch.distributed.fsdp.MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=False,
            )}
        else:
            fsdp_kwargs = {}
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            # Initialize clean patch embeddings BEFORE FSDP sharding
            log.info("Constructing clean patch embeddings before FSDP")
            # Derive kernel splits from instance attributes if metadata is already initialized
            # (e.g. called from a second build_net()), otherwise fall back to parsing the
            # framepack_type config string directly.  This is necessary because
            # build_net() is called from WANDiffusionModel.__init__ -> set_up_model()
            # before Lyra2Model.__init__ calls _init_lyra2_metadata().
            try:
                splits = self.framepack_clean_latent_frame_splits
                kernel_sizes = self.framepack_clean_latent_frame_kernel_sizes
                kernel_types = getattr(self, "framepack_clean_latent_frame_kernel_types", ["k"] * len(kernel_sizes))
            except AttributeError:
                fp_splits = config.framepack_type.split("_")
                fk_substring = fp_splits[0]
                segments = [seg for seg in fk_substring.split("f")[1:] if len(seg) > 0]
                splits = []
                kernel_sizes = []
                kernel_types = []
                for seg in segments:
                    i = 0
                    while i < len(seg) and seg[i].isdigit():
                        i += 1
                    assert i > 0 and i < len(seg), f"Invalid framepack segment: {seg}"
                    f_count = int(seg[:i])
                    t = seg[i]
                    assert t in ("k", "s"), f"Unknown kernel type {t} in segment {seg}"
                    ksize = int(seg[i + 1:])
                    splits.append(f_count)
                    kernel_sizes.append(ksize)
                    kernel_types.append(t)

            def _cfg_set(cfg, key, value):
                if isinstance(cfg, dict):
                    cfg[key] = value
                    return
                try:
                    cfg[key] = value
                except Exception:
                    setattr(cfg, key, value)

            max_spatial = config.multibuffer_max_spatial_frames
            if max_spatial is None:
                max_spatial = sum(s for s, t in zip(splits, kernel_types) if t == "s")
            max_spatial = int(max_spatial)

            buffer_in_dim = 0
            if max_spatial > 0:
                buffer_in_dim = LYRA2_CORRESPONDENCE_CHANNELS_PER_SLOT * max_spatial

            _cfg_set(config.net, "buffer_pixelshuffle", True)
            _cfg_set(config.net, "buffer_in_dim", int(buffer_in_dim))
            _cfg_set(config.net, "buffer_sincos_multires", LYRA2_BUFFER_SINCOS_MULTIRES)
            _cfg_set(config.net, "use_correspondence", True)
            _cfg_set(config.net, "use_plucker_condition", True)
            _cfg_set(config.net, "inject_kq_only", True)
            _cfg_set(config.net, "buffer_mlp_squeeze_dim", LYRA2_BUFFER_MLP_SQUEEZE_DIM)

            with torch.device(init_device):
                net = lazy_instantiate(config.net)

            net.init_clean_patch_embeddings(kernel_sizes, kernel_types)
            if hasattr(net, "buffer_pixelshuffle"):
                net.buffer_pixelshuffle = True

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh, **fsdp_kwargs)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, reshard_after_forward=True, **fsdp_kwargs)

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"

            if config.framepack_trainable_modules:
                whitelist = [p.strip() for p in config.framepack_trainable_modules.split(",") if p.strip()]
                if whitelist:
                    log.info(f"Freezing model and unfreezing Lyra2AttentionBlock layers matching: {whitelist}")

                    for param in net.parameters():
                        param.requires_grad = False

                    trainable_param_names = set()

                    for name, module in net.named_modules():
                        if type(module).__name__ == "Lyra2AttentionBlock":
                            for sub_name, param in module.named_parameters():
                                if any(pattern in sub_name for pattern in whitelist):
                                    param.requires_grad = True
                                    full_name = f"{name}.{sub_name}"
                                    trainable_param_names.add(full_name)

                    if any("clean_patch_embeddings" in p for p in whitelist):
                        for name, param in net.named_parameters():
                            if "clean_patch_embeddings" in name:
                                param.requires_grad = True
                                trainable_param_names.add(name)

                    if "patch_embedding" in whitelist:
                        for name, param in net.named_parameters():
                            if "patch_embedding" in name:
                                param.requires_grad = True
                                trainable_param_names.add(name)
                    if "patch_embedding_buffer" in whitelist:
                        for name, param in net.named_parameters():
                            if "patch_embedding_buffer" in name:
                                param.requires_grad = True
                                trainable_param_names.add(name)

                    log.info(f"Enabled gradients for {len(trainable_param_names)} parameters: {sorted(list(trainable_param_names))}")


        return net

    @torch.no_grad()
    def decode(self, latents):
        """Decode video from latents"""
        if self.tokenizer.model.video_std.shape[2] > 1:
            self.tokenizer.model.video_std = self.tokenizer.model.video_std[:,:,:1]
            self.tokenizer.model.video_mean = self.tokenizer.model.video_mean[:,:,:1]
        return self.tokenizer.decode(latents)

    def augment_conditional_latent_frames(
        self,
        gt_latent,
        condition_video_augment_sigma_in_inference=0.001,
        seed_inference=1,
        augment_sigma=None,
        target_mode="none",
    ):
        if target_mode == "none":
            return gt_latent

        elif target_mode == "noise_with_sigma":
            # Training only, sample sigma for the condition region
            if augment_sigma is None:
                augment_sigma, _ = self.draw_augment_sigma_and_epsilon_gen3c(
                gt_latent.shape,
                self.config.augment_sigma_sample_p_mean,
                self.config.augment_sigma_sample_p_std,
                self.config.augment_sigma_sample_multiplier,
            )
            noise = torch.randn(*gt_latent.shape, **self.tensor_kwargs)

        elif target_mode == "noise_with_sigma_fixed":
            # Inference only, use fixed sigma for the condition region
            log.debug(f"condition_video_augment_sigma_in_inference={condition_video_augment_sigma_in_inference}")
            assert (
                condition_video_augment_sigma_in_inference is not None
            ), "condition_video_augment_sigma_in_inference should be provided"
            s = float(condition_video_augment_sigma_in_inference)
            B, _, T, _, _ = gt_latent.shape
            if augment_sigma is None:
                augment_sigma = torch.full((B, T), s, device=self.tensor_kwargs["device"], dtype=torch.float32).to(**self.tensor_kwargs)

            # Inference, use fixed seed
            noise = misc.arch_invariant_rand(
                gt_latent.shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed_inference,
            )
        else:
            raise ValueError(f"does not support {target_mode}")

        B, _, T, _, _ = gt_latent.shape
        augment_latent = gt_latent + noise * augment_sigma.view(B, 1, T, 1, 1)

        return augment_latent, augment_sigma

    def draw_augment_sigma_and_epsilon_gen3c(self, size, p_mean, p_std, multiplier):
        B, _, T, _, _ = size
        epsilon = torch.randn(size, **self.tensor_kwargs)

        gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        cdf_vals = np.random.uniform(size=(B * T))
        samples_interval_gaussian = [gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]
        log_sigma = torch.tensor(samples_interval_gaussian, device=self.tensor_kwargs["device"], dtype=torch.float32).view(B, T)
        sigma_B = torch.exp(log_sigma).to(**self.tensor_kwargs)
        return sigma_B, epsilon

    def _init_lyra2_metadata(self):
        cfg = self.config
        fp_splits = cfg.framepack_type.split("_")
        if fp_splits[0].startswith("f") and fp_splits[1].startswith("g"):
            self.framepack_num_new_latent_frames = int(fp_splits[1].split("g")[1])
            fk_substring = fp_splits[0]
        else:
            raise ValueError(
                f"Unsupported framepack_type: {cfg.framepack_type}. Expected fk..._g..."
            )

        # Parse segments supporting both temporal ('k') and spatial ('s') kernels per segment
        segments = [seg for seg in fk_substring.split("f")[1:] if len(seg) > 0]
        splits: List[int] = []
        kernel_sizes: List[int] = []
        kernel_types: List[str] = []
        for seg in segments:
            # seg like '2k2' or '2s2'
            i = 0
            while i < len(seg) and seg[i].isdigit():
                i += 1
            assert i > 0 and i < len(seg), f"Invalid framepack segment: {seg}"
            f_count = int(seg[:i])
            t = seg[i]
            assert t in ("k", "s"), f"Unknown kernel type {t} in segment {seg}"
            ksize = int(seg[i+1:])
            splits.append(f_count)
            kernel_sizes.append(ksize)
            kernel_types.append(t)

        self.framepack_clean_latent_frame_splits: List[int] = splits
        self.framepack_clean_latent_frame_kernel_sizes: List[int] = kernel_sizes
        self.framepack_clean_latent_frame_kernel_types: List[str] = kernel_types
        # Cache commonly used counts (temporal vs spatial history slots) for downstream helpers.
        # Temporal slots correspond to kernel_type == 'k'; spatial slots correspond to 's'.
        self.framepack_num_temporal_hist = int(
            sum(s for s, t in zip(self.framepack_clean_latent_frame_splits, self.framepack_clean_latent_frame_kernel_types) if t == "k")
        )
        self.framepack_num_spatial_hist = int(
            sum(s for s, t in zip(self.framepack_clean_latent_frame_splits, self.framepack_clean_latent_frame_kernel_types) if t == "s")
        )
        log.info(
            f"Lyra2: splits={self.framepack_clean_latent_frame_splits}, "
            f"kernel_sizes={self.framepack_clean_latent_frame_kernel_sizes}, "
            f"kernel_types={self.framepack_clean_latent_frame_kernel_types}, "
            f"new_latent_frames={self.framepack_num_new_latent_frames}"
        )

        max_num_clean_latent_frames = sum(self.framepack_clean_latent_frame_splits)
        self.framepack_total_max_num_latent_frames = (
            max_num_clean_latent_frames + self.framepack_num_new_latent_frames
        )

        # framepack_splits: e.g., [16, 2, 1, 9]
        framepack_splits = self.framepack_clean_latent_frame_splits + [
            self.framepack_num_new_latent_frames
        ]
        # framepack_indices: 0, 1, ..., 18 (history), 19, 20, ..., 27 (new)
        framepack_indices = torch.arange(self.framepack_total_max_num_latent_frames)
        # framepack_kernel_ids: 0, 1, 2, -1 (new frames -- using the original patch embedding kernel)
        framepack_kernel_ids = list(range(len(self.framepack_clean_latent_frame_kernel_sizes))) + [-1]
        self.framepack_params = {
            "framepack_indices": framepack_indices,
            "framepack_splits": framepack_splits,
            "framepack_kernel_ids": framepack_kernel_ids,
            "framepack_kernel_types": self.framepack_clean_latent_frame_kernel_types,
        }
        self.framepack_max_segments = cfg.max_segments
        self.framepack_starting_frame_ratio = cfg.starting_frame_ratio
        self.framepack_num_frames_per_latent = cfg.num_frames_per_latent
        self.framepack_num_new_video_frames = (
            self.framepack_num_frames_per_latent * self.framepack_num_new_latent_frames
        )
        self.framepack_num_history_latent = max_num_clean_latent_frames

    def get_data_and_condition(
        self,
        data_batch,
        dropout=False,
    ):
        """Prepare Lyra2 latents and I2V conditioning, always using tokenizer."""
        if self.is_image_batch(data_batch):
            raise ValueError("Lyra2 expects video inputs.")

        # Normalize and tokenize video into fixed-length latents for Lyra2
        # Frank: skip assertion to save memory
        _flag = data_batch.get("is_preprocessed", False)
        if not _flag:
            self._normalize_video_databatch_inplace(data_batch)

        raw_state = data_batch[self.input_data_key]
        latent_state, last_hist_frame, cond_latent, cond_latent_mask = self._tokenizing_video_to_latents(raw_state, dropout=dropout, data_batch=data_batch)

        # Populate images key for CLIP embedding: use last frame in history segment
        data_batch["last_hist_frame"] = last_hist_frame
        data_batch["cond_latent_mask"] = cond_latent_mask

        data_batch[WAN2PT1_I2V_COND_LATENT_KEY] = cond_latent

        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        return raw_state, latent_state, condition

    def get_x0_fn_from_batch(
        self,
        data_batch,
        guidance=1.5,
        is_negative_prompt=False,
        seed=None,
    ):
        """Prepare Lyra2-aware inference closure and initial latents.

        - Conditions are prepared the same way as training and broadcast for CP.
        - Returns a closure that stitches the generated-region prediction back into a
          full-length tensor with zeros over history frames, and an initial latent
          state whose history is clean x0 and generated region is random noise.
        """
        # Prepare normalized/tokenized latents and condition
        # data_batch_uncond = data_batch.copy()
        _, x0_latents, _ = self.get_data_and_condition(data_batch)
        # _, x0_latents_uncond, _ = self.get_data_and_condition(data_batch_uncond, dropout=True)

        is_image_batch = self.is_image_batch(data_batch)
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
            # _, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch_uncond)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
            # _, uncondition = self.conditioner.get_condition_uncondition(data_batch_uncond)

        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # Enable CP and broadcast conditions (no temporal split; net handles CP internally)
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(None, uncondition, None, None)

        if not parallel_state.is_initialized():
            assert not self.net.is_context_parallel_enabled, "parallel_state is not initialized, context parallel should be turned off."

        # Build initial latents: clean history + random noise on generated region
        T_hist = self.framepack_total_max_num_latent_frames - self.framepack_num_new_latent_frames
        init_latents = torch.zeros_like(x0_latents, dtype=torch.float32)
        init_latents[:, :, :T_hist] = x0_latents[:, :, :T_hist].to(dtype=torch.float32)

        gen_shape = tuple(x0_latents[:, :, T_hist:].shape)
        gen_noise = misc.arch_invariant_rand(
            gen_shape,
            torch.float32,
            self.tensor_kwargs["device"],
            seed if seed is not None else 0,
        )
        init_latents[:, :, T_hist:] = gen_noise

        def x0_fn(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            # noise_x_uncond = noise_x.clone()
            cond_v_gen = self.denoise(noise_x, timestep, condition)

            # noise_x_uncond[:, :, :T_hist] = x0_latents_uncond[:, :, :T_hist].to(noise_x_uncond) # set history region of x to encoding of zeros
            uncond_v_gen = self.denoise(noise_x, timestep, uncondition)
            gen_v = uncond_v_gen + guidance * (cond_v_gen - uncond_v_gen)

            vt_full = torch.zeros_like(noise_x, dtype=gen_v.dtype)
            vt_full[:, :, T_hist:] = gen_v # zero predicted noise for history frames to keep history unchanged.
            return vt_full

        return x0_fn, init_latents

    @torch.no_grad()
    def inference(self, history_latents, cond_latent, guidance, seed, num_steps, shift, t5_text_embeddings, neg_t5_text_embeddings, **kwargs):
        # 1) Validate history latent length
        T_hist_expected = self.framepack_total_max_num_latent_frames - self.framepack_num_new_latent_frames
        assert (
            history_latents.shape[2] == T_hist_expected
        ), f"history_latents has T={history_latents.shape[2]} but expected {T_hist_expected}"

        # 2) Build conditioner inputs from provided kwargs and history latents
        # Required: last_hist_frame in pixel space [B,3,H,W]
        assert (
            "last_hist_frame" in kwargs
        ), "last_hist_frame (pixel) is required in kwargs for Lyra2 inference"
        last_hist_frame = kwargs["last_hist_frame"]

        # Optional: fps, padding_mask passthrough if provided
        data_batch = {
            "t5_text_embeddings": t5_text_embeddings,
            "neg_t5_text_embeddings": neg_t5_text_embeddings,
            "last_hist_frame": last_hist_frame,
        }

        # Build cond media latents = [history, zeros(gen_region)] and its mask (1 on history, 0 on gen)
        B, C, T_hist, H, W = history_latents.shape
        T_new = self.framepack_num_new_latent_frames
        assert cond_latent.shape[2] == T_hist + T_new, "cond_latent must have T_hist + T_new frames"

        mask = kwargs.get("cond_latent_mask", None)
        if mask is None:
            mask = torch.ones((B, 4, T_hist + T_new, H, W), dtype=history_latents.dtype, device=history_latents.device)

        data_batch["cond_latent_mask"] = mask
        data_batch[WAN2PT1_I2V_COND_LATENT_KEY] = cond_latent
        data_batch["cond_latent_buffer"] = kwargs.get("cond_latent_buffer", None)

        # Add-through extras if present
        if "fps" in kwargs and kwargs["fps"] is not None:
            data_batch["fps"] = kwargs["fps"]
        if "padding_mask" in kwargs and kwargs["padding_mask"] is not None:
            data_batch["padding_mask"] = kwargs["padding_mask"]

        # 3) Build condition / uncondition with negative prompt
        is_image_batch = False
        condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # 4) Init latents: keep history clean, random on generated region
        init_latents = torch.zeros((B, C, T_hist + T_new, H, W), dtype=torch.float32, device=self.tensor_kwargs["device"])  # float32 for sampler math
        init_latents[:, :, :T_hist] = history_latents.to(dtype=torch.float32, device=self.tensor_kwargs["device"])  # clean history

        gen_shape = (B, C, T_new, H, W)
        gen_noise = misc.arch_invariant_rand(
            gen_shape,
            torch.float32,
            self.tensor_kwargs["device"],
            seed if seed is not None else 0,
        )
        init_latents[:, :, T_hist:] = gen_noise

        # 5) CP broadcast (no temporal split for Lyra2; net handles internally)
        cp_group = self.get_context_parallel_group()
        if cp_group is not None:
            init_latents = broadcast(init_latents.contiguous(), cp_group)
            condition = condition.broadcast(cp_group)
            uncondition = uncondition.broadcast(cp_group)
        else:
            # Some network variants may not expose the property until enabled; use getattr
            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )

        # 6) Define x0_fn following Lyra2 get_x0_fn_from_batch semantics (zeros over history)
        def x0_fn(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v_gen = self.denoise(noise_x, timestep, condition)
            uncond_v_gen = self.denoise(noise_x, timestep, uncondition)
            gen_v = uncond_v_gen + guidance * (cond_v_gen - uncond_v_gen)
            vt_full = torch.zeros_like(noise_x, dtype=gen_v.dtype)
            vt_full[:, :, T_hist:] = gen_v
            return vt_full

        # 7) Sampling loop
        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
        timesteps = self.sample_scheduler.timesteps

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed if seed is not None else 0)

        latents = init_latents
        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = torch.stack([t])
            velocity_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = temp_x0.squeeze(0)

        # 8) Return only the newly generated latent chunk
        return latents[:, :, T_hist:]

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        return_condition_state: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key


        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps, device=self.tensor_kwargs["device"], shift=shift)

        timesteps = self.sample_scheduler.timesteps

        # Indicate whether to collect visualization/condition payloads during tokenization
        prev_collect_flag = getattr(self, "_collect_return_condition_state", False)
        self._collect_return_condition_state = bool(return_condition_state)
        x0_fn, init_latents = self.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=is_negative_prompt, seed=seed
        )
        latents = init_latents

        # Broadcast initial latents across CP ranks; do not split (net handles CP internally)
        cp_group = self.get_context_parallel_group()
        if cp_group is not None:
            latents = broadcast(latents.contiguous(), cp_group)

        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            noise_pred = x0_fn(latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = temp_x0.squeeze(0)


        if return_condition_state:
            cond_parts = []
            gt_vis = getattr(self, "_latest_gt_gen_pixels", None)
            cond_pixels = getattr(self, "_latest_condition_state_pixels", None)
            if cond_pixels is not None:
                cond_parts.append(cond_pixels)
            rays = getattr(self, "_latest_plucker_rays_pixels", None)
            cond_plucker = None
            if rays is not None and isinstance(rays, dict):
                ray_o = rays.get("ray_origin", None)
                ray_d = rays.get("ray_direction", None)
                if ray_o is not None and ray_d is not None:
                    # Convert to [B, 6, T, H, W]
                    cond_plucker = torch.cat([
                        ray_o.permute(0, 4, 1, 2, 3).contiguous(),
                        ray_d.permute(0, 4, 1, 2, 3).contiguous(),
                    ], dim=1)
            if cond_plucker is not None:
                cond_parts.append(cond_plucker)
            cond_state = None
            if len(cond_parts) > 0:
                if len(cond_parts) == 1:
                    cond_state = cond_parts[0]
                else:
                    target_T = max(part.shape[2] for part in cond_parts)
                    target_HW = {(part.shape[3], part.shape[4]) for part in cond_parts}
                    assert len(target_HW) == 1, (
                        f"Cannot concatenate condition parts with different spatial sizes: {target_HW}"
                    )
                    aligned_parts = []
                    for part in cond_parts:
                        if part.shape[2] < target_T:
                            pad_frames = target_T - part.shape[2]
                            pad = torch.zeros(
                                (part.shape[0], part.shape[1], pad_frames, part.shape[3], part.shape[4]),
                                dtype=part.dtype,
                                device=part.device,
                            )
                            part = torch.cat([pad, part], dim=2)
                        elif part.shape[2] > target_T:
                            part = part[:, :, -target_T:]
                        aligned_parts.append(part)
                    cond_state = torch.cat(aligned_parts, dim=1)
            # Clear cached references immediately after assembling return payload
            self._latest_condition_state_pixels = None
            self._latest_plucker_rays_pixels = None
            self._latest_gt_gen_pixels = None
            # Restore flag
            self._collect_return_condition_state = prev_collect_flag
            if gt_vis is not None and cond_state is not None:
                return latents, cond_state, gt_vis
            if cond_state is not None:
                return latents, cond_state
            if gt_vis is not None:
                return latents, None, gt_vis
        # Restore flag when not returning condition state
        self._collect_return_condition_state = prev_collect_flag
        return latents

    # ------------------------ Inference utilities ------------------------
    def denoise(self, xt_B_C_T_H_W, timestep, condition):
        """Lyra2-aware denoise: return only generated-region prediction from net."""
        vt_pred_gen = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timestep,
            **condition.to_dict(),
            **self.framepack_params,
        ).float()
        return vt_pred_gen

    # ------------------------ Training utilities ------------------------
    def _clone_vae_cache(self, cache_list):
        """Clone a VAE encoder cache list, cloning tensors to avoid aliasing."""
        cloned = []
        for it in cache_list:
            if torch.is_tensor(it):
                cloned.append(it.clone())
            else:
                cloned.append(it)
        return cloned

    def _vae_encode_range_stream(self, x_vid, start_t, end_t, skip_first_frame: bool = False):
        """Stream-encode frames in [start_t, end_t) using current encoder caches, return features."""
        vae_wrap = self.tokenizer.model   # WanVAE wrapper
        vae_core = vae_wrap.model         # WanVAE_ core
        temporal_window = vae_wrap.temporal_window

        feats = []
        # First frame handling only when starting from t=0 and not resuming from a prior cache
        if start_t == 0 and not bool(skip_first_frame):
            vae_core._enc_conv_idx = [0]
            out0 = vae_core.encoder(
                x_vid[:, :, :1, :, :],
                feat_cache=vae_core._enc_feat_map,
                feat_idx=vae_core._enc_conv_idx,
            )
            feats.append(out0)
            start_t = 1
        # Full temporal_window chunks
        pos = start_t
        while pos + temporal_window <= end_t:
            vae_core._enc_conv_idx = [0]
            out_w = vae_core.encoder(
                x_vid[:, :, pos: pos + temporal_window, :, :],
                feat_cache=vae_core._enc_feat_map,
                feat_idx=vae_core._enc_conv_idx,
            )
            feats.append(out_w)
            pos += temporal_window
        # Remainder
        if pos < end_t:
            vae_core._enc_conv_idx = [0]
            out_r = vae_core.encoder(
                x_vid[:, :, pos: end_t, :, :],
                feat_cache=vae_core._enc_feat_map,
                feat_idx=vae_core._enc_conv_idx,
            )
            feats.append(out_r)
        if len(feats) == 1:
            return feats[0]
        return torch.cat(feats, dim=2)

    @torch.no_grad()
    def vae_encode_with_cache(self, enc_cache, video, start_t=None, end_t=None, return_cache=False):
        """Resume encoder from `cache` and stream-encode `video[start_t:end_t)` into features.

        Args:
            cache: Encoder feature cache captured from VAE encoder state.
                   Accepts either a list of per-layer cached tensors, or a legacy (enc_cache, enc_idx) tuple.
            video: [B, C, T, H, W] pixels in the model's pixel range.
            start_t: start index (inclusive). Defaults to 0.
            end_t: end index (exclusive). Defaults to T.
        Returns:
            Tensor of encoder features over the requested range.
        """
        vae_wrap = self.tokenizer.model   # WanVAE wrapper
        vae_core = vae_wrap.model         # WanVAE_ core
        with vae_wrap.context:
            if not vae_wrap.is_amp:
                x_vid = video.to(vae_wrap.dtype)
            else:
                x_vid = video
            # Restore encoder caches
            vae_core._enc_feat_map = self._clone_vae_cache(enc_cache)
            # Always start layer-iteration index at 0 when resuming with an explicit cache
            vae_core._enc_conv_idx = [0]
            # Resolve range
            T_total = int(x_vid.shape[2])
            s = 0 if start_t is None else int(start_t)
            e = T_total if end_t is None or int(end_t) < 0 else int(end_t)
            # When resuming from a provided cache, avoid the I-frame special case even if s==0
            feats = self._vae_encode_range_stream(x_vid, s, e, skip_first_frame=True)
            if return_cache:
                cache_current = self._clone_vae_cache(vae_core._enc_feat_map)
                return feats, cache_current
            else:
                return feats

    @torch.no_grad()
    def _encoder_feats_to_normalized_latents(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        """Project encoder features to [mu, logvar], take mu, and apply channel-wise and per-frame normalization.

        Matches the normalization used by Wan2pt1VAEInterface.encode in this codebase.
        """
        vae_iface = self.tokenizer
        vae_wrap = vae_iface.model   # WanVAE wrapper
        vae_core = vae_wrap.model    # WanVAE_ core
        with vae_wrap.context:
            mu_logvar = vae_core.conv1(encoder_feats)
            mu, _ = mu_logvar.chunk(2, dim=1)
            # Channel-wise normalization
            mean_c = vae_wrap.scale[0]
            inv_std_c = vae_wrap.scale[1]
            if torch.is_tensor(mean_c):
                mu = (mu - mean_c.view(1, vae_core.z_dim, 1, 1, 1).type_as(mu)) * inv_std_c.view(1, vae_core.z_dim, 1, 1, 1).type_as(mu)
            else:
                mu = (mu - mean_c) * inv_std_c
            # Per-frame normalization
            if int(mu.shape[2]) == 1:
                latents = (mu - vae_wrap.img_mean.type_as(mu)) / vae_wrap.img_std.type_as(mu)
            else:
                latents = (mu - vae_wrap.video_mean[:, :, :1].type_as(mu)) / vae_wrap.video_std[:, :, :1].type_as(mu)
        return latents

    @torch.no_grad()
    def _vae_encode_with_shared_prefix(self, video, gen_cond_pixels=None, return_cache=False):
        """Efficiently encode full latents and zero-tailed conditional latents by reusing VAE encoder caches.

        Returns (latents, cond_latent), both normalized the same way as tokenizer.encode.
        """
        # Build zero-tail video for I2V conditioning
        _video = video.clone()
        if gen_cond_pixels is None:
            _video[:, :, -self.framepack_num_new_video_frames:] = 0
        else:
            _video[:, :, -self.framepack_num_new_video_frames:] = gen_cond_pixels

        # Stream-encode prefix once,
        # snapshot encoder caches, then continue with zero-tail and real-tail separately.
        vae_iface = self.tokenizer  # Wan2pt1VAEInterface
        vae_wrap = vae_iface.model   # WanVAE wrapper
        vae_core = vae_wrap.model    # WanVAE_ core

        T_total = int(video.shape[2])
        T_hist = T_total - int(self.framepack_num_new_video_frames)

        # Use VAE's autocast/dtype context for consistency
        with vae_wrap.context:
            if not vae_wrap.is_amp:
                video_cast = video.to(vae_wrap.dtype)
                video_zero_cast = _video.to(vae_wrap.dtype)
            else:
                video_cast = video
                video_zero_cast = _video

            # Reset encoder caches and build prefix features once
            vae_core.clear_cache()
            vae_core._enc_conv_idx = [0]
            prefix_feats = self._vae_encode_range_stream(video_cast, 0, T_hist)

            # Snapshot caches post-prefix (per-layer feature map cache only)
            enc_cache_after_prefix = self._clone_vae_cache(vae_core._enc_feat_map)

            # Continue with zero-tail from cached state
            zero_tail_feats = self.vae_encode_with_cache(
                enc_cache_after_prefix,
                video_zero_cast,
                start_t=T_hist,
                end_t=T_total,
            )

            # Continue with real-tail from the SAME cached state
            real_tail_feats = self.vae_encode_with_cache(
                enc_cache_after_prefix,
                video_cast,
                start_t=T_hist,
                end_t=T_total,
            )

            # Stitch features and run final 1x1 causal conv to produce [mu, logvar]
            feats_cond = torch.cat([prefix_feats, zero_tail_feats], dim=2) if T_hist < T_total else prefix_feats
            feats_full = torch.cat([prefix_feats, real_tail_feats], dim=2) if T_hist < T_total else prefix_feats
            cond_latent = self._encoder_feats_to_normalized_latents(feats_cond)
            latents = self._encoder_feats_to_normalized_latents(feats_full)

        in_dtype = video.dtype
        latents = latents.contiguous().to(in_dtype)
        cond_latent = cond_latent.contiguous().to(in_dtype)
        # Clear encoder feature caches to release memory

        if return_cache:
            cache_after_prefix = enc_cache_after_prefix
            cache_current = self._clone_vae_cache(vae_core._enc_feat_map)
            return latents, cond_latent, cache_after_prefix, cache_current
        else:
            return latents, cond_latent

    # ------------------------ Training utilities ------------------------
    def prepare_latent_conditon(
        self,
        condition_state,
        condition_state_mask,
        dtype,
        condition_video_augment_sigma_in_inference=0.001,
        seed_inference=1,
        is_testing=True,
    ):
        """Encode warped condition frames into VAE latents.

        Only support the simple case where condition_state has 5 dims: [B, C, T, H, W].
        """
        if condition_state.dim() == 5:
            # Prepend the first frame to satisfy VAE's I-frame requirement,
            # then drop the first latent token after encoding.
            first = condition_state[:, :, :1]
            condition_state = torch.cat([first, condition_state], dim=2)
            latent_condition = self.encode(condition_state.to(dtype)).contiguous()
            latent_condition = latent_condition[:, :, 1:]  # drop the inserted I-frame token
        else:
            raise NotImplementedError("prepare_latent_conditon only supports 5D condition_state in Lyra2Model")
        return latent_condition

    # ----------------------------- Training -----------------------------
    def training_step(
        self, data_batch, iteration
    ):
        """Lyra2 training: loss only on generated latents; history latents are clean.
        """
        if not self.framepack_weights_initialized and iteration == 0 and self.config.init_framepack_weights:
            self.net.copy_weights_to_clean_patch_embeddings()
            self.framepack_weights_initialized = True
        self._update_train_stats(data_batch)
        dropout = False

        prob = float(getattr(self.config, "self_aug_prob", 1.0))
            # Synchronized decision across devices via broadcast from rank 0
        rand_tensor = torch.zeros(1, device=self.tensor_kwargs["device"], dtype=torch.float32)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                rand_tensor.uniform_(0.0, 1.0)
            torch.distributed.broadcast(rand_tensor, src=0)
        else:
            rand_tensor.uniform_(0.0, 1.0)
        rand_val = float(rand_tensor.item())
        # ========================= Stage A: Self-augmentation (optional) =========================
        if (
            getattr(self.config, "self_aug_enabled", False)
            and (iteration % int(getattr(self.config, "self_aug_every_k", 1)) == 0)
            and rand_val <= prob
        ):
            if float(rand_tensor.item()) > self.config.self_aug_i2v_ratio:
                raw_state, x0_latents, condition = self.get_data_and_condition(data_batch, dropout=dropout)
                with misc.timer("self_aug_training_step", debug=False), torch.no_grad():
                    # Indices and shapes
                    T_hist = self.framepack_total_max_num_latent_frames - self.framepack_num_new_latent_frames
                    stride = int(self.framepack_num_frames_per_latent)

                    # Build init latents: keep history clean; tail will be partially noised via RectifiedFlow
                    init_latents_aug = torch.zeros_like(x0_latents, dtype=torch.float32)
                    init_latents_aug[:, :, :T_hist] = x0_latents[:, :, :T_hist].to(torch.float32).clone()

                    # Stage-A RectifiedFlow: uniform time with max_T scaling
                    stageA_flow = self.rectified_flow

                    B = int(x0_latents.shape[0])
                    tA_single = stageA_flow.sample_train_time(1).to(**self.flow_matching_kwargs)  # [1] e.g. 0.193
                    timestepA_single = stageA_flow.get_discrete_timestamp(tA_single, self.flow_matching_kwargs)  # [1], e.g. 954
                    max_T = int(getattr(self.config, "self_aug_max_T", 50))
                    scale = max_T / 1000.0
                    timestepA_single = (timestepA_single.float() * scale).floor().clamp(min=1).to(dtype=timestepA_single.dtype)

                    # Find the nearest value in stageA_flow.noise_scheduler.timesteps to the single timestep
                    # stageA_flow.noise_scheduler.timesteps is a 1D tensor of available timesteps
                    available_timesteps = stageA_flow.noise_scheduler.timesteps.to(timestepA_single.device, dtype=timestepA_single.dtype)

                    # timestepA_single: [1], available_timesteps: [N]
                    # Find the index of the closest available timestep
                    diff = torch.abs(available_timesteps - timestepA_single)  # [N]
                    nearest_index = diff.argmin()  # scalar
                    timestepA_single = available_timesteps[nearest_index].unsqueeze(0)  # [1]

                    timestepsA = timestepA_single.expand(B, 1)  # [B, 1] - broadcast to batch
                    sigmasA = stageA_flow.get_sigmas(timestepsA, self.flow_matching_kwargs)  # [B]
                    sigmasA_B1 = rearrange(sigmasA, "b -> b 1")

                    log.info(f"self aug timestep={timestepA_single}", rank0_only=False)

                    # Build epsilon only for the tail region and form xt via interpolation
                    eps_tail = torch.randn_like(
                        x0_latents[:, :, T_hist:].to(torch.float32), dtype=self.flow_matching_kwargs["dtype"]
                    ) #e.g. ([1, 16, 9, 56, 96])

                    xt_tail, _ = stageA_flow.get_interpolation(
                        eps_tail, x0_latents[:, :, T_hist:].to(torch.float32), sigmasA_B1
                    )
                    init_latents_aug[:, :, T_hist:] = xt_tail

                    # Build Stage-A CFG conditions (video) and CP broadcast if needed
                    is_image_batch = self.is_image_batch(data_batch)
                    condition_A, uncondition_A = self.conditioner.get_condition_with_negative_prompt(data_batch)
                    condition_A = condition_A.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
                    uncondition_A = uncondition_A.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

                    cp_group = self.get_context_parallel_group()
                    if cp_group is not None:
                        init_latents_aug = broadcast(init_latents_aug.contiguous(), cp_group)
                        condition_A = condition_A.broadcast(cp_group)
                        uncondition_A = uncondition_A.broadcast(cp_group)
                        sigmasA = broadcast(sigmasA.contiguous(), cp_group)

                    # x0_fn: zeros history, predict only tail with CFG
                    guidance_A = float(getattr(self.config, "self_aug_guidance", 1.0) or 1.0)

                    def x0_fn_A(noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                        cond_v_gen = self.denoise(noise_x, timestep, condition_A)
                        if guidance_A  == 1.0:
                            gen_v = cond_v_gen
                        else:
                            uncond_v_gen = self.denoise(noise_x, timestep, uncondition_A)
                            gen_v = uncond_v_gen + guidance_A * (cond_v_gen - uncond_v_gen)
                        vt_full = torch.zeros_like(noise_x, dtype=gen_v.dtype)
                        vt_full[:, :, T_hist:] = gen_v
                        return vt_full

                    # Short sampling for Stage-A
                    steps_A = int(getattr(self.config, "self_aug_steps", 3))
                    shift_A = (
                        float(self.config.self_aug_scheduler_shift)
                        if getattr(self.config, "self_aug_scheduler_shift", None) is not None
                        else 1.0
                    )

                    sigmas_steps =np.linspace(sigmasA.squeeze().item(), 0.0, steps_A)
                    self.sample_scheduler.set_timesteps(
                        steps_A, device=self.tensor_kwargs["device"], sigmas=sigmas_steps, shift=shift_A
                    )
                    timesteps_iter = self.sample_scheduler.timesteps

                    latents_A = init_latents_aug
                    for _, t in enumerate(timesteps_iter):
                        latent_model_input = latents_A
                        timestep = torch.stack([t])
                        vt_pred = x0_fn_A(latent_model_input, timestep.unsqueeze(0))
                        temp_x0 = self.sample_scheduler.step(
                            vt_pred.unsqueeze(0),
                            t,
                            latents_A[0].unsqueeze(0),
                            return_dict=False,
                        )[0]
                        latents_A = temp_x0.squeeze(0)

                    # Decode Stage-A tail to pixels, stitch into the pixelized version of current latent window, re-encode
                    # Cached streaming decode/encode update
                    latents_new = latents_A[:, :, T_hist:]
                    assert latents_new.shape[2] == self.framepack_num_new_latent_frames
                    history_latents = data_batch["_stage_a_full_latents"][:, :, : -self.framepack_num_new_latent_frames]
                    latents_to_decode = torch.cat([history_latents, latents_new], dim=2)
                    full_frames = self.decode(latents_to_decode.to(self.tensor_kwargs["dtype"]))
                    # First latent frame is the start I-frame; remaining are V-frames.
                    chunk_frames = full_frames[:, :, -(self.framepack_total_max_num_latent_frames - 1)*self.framepack_num_frames_per_latent :]
                    del history_latents
                    del latents_to_decode, latents_new, full_frames

                    # Get the frame indices and metadata from stage A
                    video_indices = data_batch.get("_stage_a_video_indices")
                    stage_a_start = data_batch.get("_stage_a_start")
                    stage_a_cur_segment_id = data_batch.get("_stage_a_cur_segment_id")

                    # Create a modified data_batch with augmented frames inserted at correct positions
                    data_batch_aug = data_batch
                    original_video = data_batch[self.input_data_key]
                    # Insert chunk_frames into the original video at the correct positions
                    if video_indices is not None and len(video_indices) > 0:
                        # Create a copy of the original video
                        augmented_video = original_video.clone()
                        original_indices = video_indices.cpu().numpy()
                        # Only replace frames that actually exist in the original video and haven't been repeated/padded
                        valid_indices = original_indices[original_indices < original_video.shape[2]]

                        if len(valid_indices) > 0:
                            # Map the valid indices to the corresponding frames in video_px_aug2
                            # align the tails
                            if len(valid_indices) > chunk_frames.shape[2]:
                                n_extra = len(valid_indices) - chunk_frames.shape[2]
                                valid_indices = valid_indices[n_extra:]
                            elif len(valid_indices) < chunk_frames.shape[2]:
                                n_extra = chunk_frames.shape[2] - len(valid_indices)
                                chunk_frames = chunk_frames[-n_extra:]

                            if not self.config.self_aug_copy_chunk:
                                valid_indices = valid_indices[-self.framepack_num_new_video_frames:]
                                chunk_frames = chunk_frames[:, :, -self.framepack_num_new_video_frames:]
                            max_aug_frames = min(len(valid_indices), chunk_frames.shape[2])
                            for i in range(max_aug_frames):
                                orig_idx = int(valid_indices[i])
                                augmented_video[:, :, orig_idx] = chunk_frames[:, :, i].clamp(-1, 1)

                        data_batch_aug[self.input_data_key] = augmented_video
                    else:
                        raise ValueError("Fallback not implemented")

                    # Preserve the original start to maintain consistency
                    if stage_a_start is not None:
                        data_batch_aug["start"] = stage_a_start
                    # increment cur_segment_id by 1
                    if stage_a_cur_segment_id is not None:
                        data_batch_aug["cur_segment_id"] = stage_a_cur_segment_id + 1

                # Regenerate x0_latents and condition with the preserved parameters
                raw_state, x0_latents, condition = self.get_data_and_condition(data_batch_aug, dropout=dropout)
            else:
                data_batch["cur_segment_id"] = 0 # train i2v
                data_batch["is_i2v"] = True
                raw_state, x0_latents, condition = self.get_data_and_condition(data_batch, dropout=dropout)
        else:
            raw_state, x0_latents, condition = self.get_data_and_condition(data_batch, dropout=dropout)

        # Remove a series of cache and input keys from data_batch if present
        if hasattr(self.tokenizer.model.model, "clear_cache"):
            try:
                self.tokenizer.model.model.clear_cache()
                del original_video
                del augmented_video
            except Exception:
                pass

        del raw_state
        for k in [
            "_stage_a_vae_cache_T-2",
            "_stage_a_vae_cache_T-1",
            "stage_a_full_latents",
            "t5_chunk_embeddings",
            "t5_chunk_mask",
            "depth",
            "_stage_a_full_latents",
            "camera_w2c",
            "intrinsics",
            "t5_chunk_keys",
            "sample_frame_indices",
            "control_input_world_scenario",
            WAN2PT1_I2V_COND_LATENT_KEY,
            "video",
            "last_hist_frame",
        ]:
            if k in data_batch:
                del data_batch[k]
        data_batch["video"] = torch.zeros([1])

        gc.collect()
        torch.cuda.empty_cache()

        # Sample times
        batch_size = x0_latents.size(0)
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.flow_matching_kwargs)
        t_B = rearrange(t_B, "b -> b 1")

        # Build epsilon BEFORE CP split: zeros on history, noise on generated region.
        T_hist = self.framepack_total_max_num_latent_frames - self.framepack_num_new_latent_frames
        epsilon_full = x0_latents.clone()
        epsilon_full[:, :, T_hist:] = torch.randn_like(
            x0_latents[:, :, T_hist:], dtype=self.flow_matching_kwargs["dtype"]
        )

        # Enable CP (without splitting T) and broadcast inputs so all CP ranks see identical data;
        # CP splitting happens after Lyra2 patchify inside net
        cp_group = self.get_context_parallel_group()
        if cp_group is not None:
            x0_latents = broadcast(x0_latents.contiguous(), cp_group)
            epsilon_full = broadcast(epsilon_full.contiguous(), cp_group)
            t_B = broadcast(t_B.contiguous(), cp_group)
            condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()
        # CP after Lyra2 patchify
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.flow_matching_kwargs)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.flow_matching_kwargs)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        # Compute interpolation directly on already-split tensors
        xt, vt = self.rectified_flow.get_interpolation(epsilon_full, x0_latents, sigmas)

        # Net forward: it returns only generated region prediction when Lyra2 params are provided
        vt_pred_gen = self.net(
            x_B_C_T_H_W=xt.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            **condition.to_dict(),
            **self.framepack_params,
        )

        # Loss only over generated region; align with base model weighting
        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.flow_matching_kwargs)
        vt_gen_target = vt[:, :, T_hist:].to(vt_pred_gen.dtype)
        per_instance_loss = torch.mean((vt_pred_gen - vt_gen_target) ** 2, dim=list(range(1, vt_pred_gen.dim())))
        loss = torch.mean(time_weights_B * per_instance_loss)
        output_batch = {"edm_loss": loss}

        return output_batch, loss



    def _select_temporal_history_indices(self, T_hist_total, num_temporal_hist):
        """Select temporal history latent indices: always include important start (0) and most recent rest."""
        temporal_rest_needed = max(0, num_temporal_hist - 1)
        recent_start = max(0, T_hist_total - temporal_rest_needed)
        temporal_rest = list(range(recent_start, T_hist_total))
        return [0] + temporal_rest

    def _compose_selected_indices(
        self,
        splits,
        types,
        T_hist_total,
        temporal_selected,
        spatial_selected,
    ):
        device = self.tensor_kwargs["device"]

        temporal_pool = [idx for idx in temporal_selected if idx != 0]
        spatial_pool = list(spatial_selected)
        ordered_past: list[int] = []
        for seg_idx, (cnt, tp) in enumerate(zip(splits, types)):
            if tp == "k":
                for j in range(cnt):
                    if seg_idx == 0 and j == 0:
                        ordered_past.append(0)
                    else:
                        ordered_past.append(temporal_pool.pop(0))
            else:
                for _ in range(cnt):
                    if len(spatial_pool) == 0:
                        ordered_past.append(T_hist_total - 1)
                    else:
                        ordered_past.append(spatial_pool.pop(0))

        selected_idx = torch.tensor(ordered_past, device=device, dtype=torch.long)
        return selected_idx

    @staticmethod
    def _build_canonical_spatial_coords(
        H: int,
        W: int,
        num_spatial_hist: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if num_spatial_hist <= 0:
            return None
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base_xy = torch.stack([xx, yy], dim=0)  # [2,H,W]
        base_xy = base_xy.unsqueeze(0).repeat(num_spatial_hist, 1, 1, 1)  # [N,2,H,W]
        if num_spatial_hist == 1:
            zs = torch.zeros(1, device=device, dtype=dtype)
        else:
            zs = torch.linspace(-1.0, 1.0, num_spatial_hist, device=device, dtype=dtype)
        z = zs.view(num_spatial_hist, 1, 1, 1).expand(num_spatial_hist, 1, H, W)
        coords = torch.cat([base_xy, z], dim=1)  # [N,3,H,W]
        return coords

    def _get_cached_spatial_coords(
        self,
        H: int,
        W: int,
        num_spatial_hist: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        meta = (H, W, num_spatial_hist, device, dtype)
        if self._cached_spatial_coords is None or self._cached_spatial_coords_meta != meta:
            self._cached_spatial_coords = self._build_canonical_spatial_coords(
                H=H,
                W=W,
                num_spatial_hist=num_spatial_hist,
                device=device,
                dtype=dtype,
            )
            self._cached_spatial_coords_meta = meta
        return self._cached_spatial_coords

    @staticmethod
    def _pixelshuffle_hw_to_latent(
        x: torch.Tensor,
        *,
        h8: int = 8,
        w8: int = 8,
    ) -> torch.Tensor:
        return rearrange(
            x,
            "b c t (h h8) (w w8) -> b (c h8 w8) t h w",
            h8=h8,
            w8=w8,
        )

    def _coord_pixels_to_latents(
        self,
        coord_pixels: torch.Tensor,
        *,
        dtype: torch.dtype,
        target_t: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert warped coordinate pixels to latent grid via subsample+pixelshuffle."""
        frames_per_lat = int(self.framepack_num_frames_per_latent)
        F = int(coord_pixels.shape[2])
        start = max(frames_per_lat - 1, 0)
        time_idx = list(range(start, F, frames_per_lat))
        time_idx_t = torch.tensor(time_idx, device=coord_pixels.device, dtype=torch.long)
        coord_sel = coord_pixels[:, :, time_idx_t]
        coord_lat = self._pixelshuffle_hw_to_latent(coord_sel)
        if target_t is not None and int(coord_lat.shape[2]) != int(target_t):
            raise ValueError(f"Unexpected coord_lat.shape[2]={coord_lat.shape[2]} != target_t={target_t}")
        return coord_lat.to(dtype=dtype)

    def _apply_camera_controls(
        self,
        cond_latent,
        selected_idx,
        video_indices,
        spatial_selected_frame_ids: Optional[torch.Tensor] = None,
        spatial_selected_coords: Optional[torch.Tensor] = None,
        *,
        video: Optional[torch.Tensor] = None,
        camera_w2c: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        buffer_depth_B_1_H_W: Optional[torch.Tensor] = None,
        spatial_cache: Optional[Sparse3DCache] = None,
        is_training: bool = True,
    ):
        cfg = self.config
        if video is None or camera_w2c is None or intrinsics is None:
            return cond_latent, None
        buffer_cond_latents: Optional[torch.Tensor] = None
        T_new_lat = int(self.framepack_num_new_latent_frames)
        # Depth warp / HD map condition overwriting the tail of cond_latent
        with misc.timer("camera_pose_condition - build"):
            device = cond_latent.device
            B = int(cond_latent.shape[0])
            # Find the buffer frame absolute index
            rel_buffer_idx = video_indices.shape[0] - self.framepack_num_new_video_frames - 1
            abs_buffer_idx = int(video_indices[rel_buffer_idx].item())
            abs_gen_indices = video_indices[-self.framepack_num_new_video_frames:]

            F = int(abs_gen_indices.numel())
            target_w2cs = camera_w2c[:, abs_gen_indices]
            target_intrinsics = intrinsics[:, abs_gen_indices]

            spatial_condition_pixels_list: list[torch.Tensor] = []

            with misc.timer("camera_pose_condition - corruptor"):
                # _warp_multisrc: shared helper for accumulated PCD / correspondence warping.
                def _warp_multisrc(
                    src_rgb: torch.Tensor,
                    src_depth: torch.Tensor,
                    src_w2c: torch.Tensor,
                    src_K: torch.Tensor,
                    *,
                    return_depth: bool = False,
                ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
                    # Repeat sources across target frames, flatten (B,F) into batch, then chunk-wise warp.
                    src_rgb_bf = rearrange(
                        src_rgb.unsqueeze(1).repeat(1, F, 1, 1, 1, 1),
                        "b f n c h w -> (b f) n c h w",
                    )
                    src_depth_bf = rearrange(
                        src_depth.unsqueeze(1).repeat(1, F, 1, 1, 1, 1),
                        "b f n c h w -> (b f) n c h w",
                    )
                    src_w2c_bf = rearrange(
                        src_w2c.unsqueeze(1).repeat(1, F, 1, 1, 1),
                        "b f n c d -> (b f) n c d",
                    )
                    src_K_bf = rearrange(
                        src_K.unsqueeze(1).repeat(1, F, 1, 1, 1),
                        "b f n c d -> (b f) n c d",
                    )
                    tgt_w2c_bf = rearrange(target_w2cs.to(dtype=torch.float32), "b f c d -> (b f) c d")
                    tgt_K_bf = rearrange(target_intrinsics.to(dtype=torch.float32), "b f c d -> (b f) c d")

                    warp_chunk_size = self.config.warp_chunk_size
                    warped_imgs_list: list[torch.Tensor] = []
                    warped_masks_list: list[torch.Tensor] = []
                    warped_depths_list: list[torch.Tensor] = []
                    for i in range(0, int(src_rgb_bf.shape[0]), int(warp_chunk_size)):
                        w_img, w_mask, w_depth, _ = forward_warp_multiframes(
                            src_rgb_bf[i : i + warp_chunk_size],
                            mask1=None,
                            depth1=src_depth_bf[i : i + warp_chunk_size],
                            transformation1=src_w2c_bf[i : i + warp_chunk_size],
                            transformation2=tgt_w2c_bf[i : i + warp_chunk_size],
                            intrinsic1=src_K_bf[i : i + warp_chunk_size],
                            intrinsic2=tgt_K_bf[i : i + warp_chunk_size],
                            is_image=True,
                            render_depth=return_depth,
                            world_points1=None,
                            clean_points=True,
                            clean_points_continuity=True,
                        )
                        warped_imgs_list.append(w_img)
                        warped_masks_list.append(w_mask)
                        if return_depth:
                            assert w_depth is not None
                            warped_depths_list.append(w_depth)
                    warped_imgs_bf = torch.cat(warped_imgs_list, dim=0)
                    # warped_masks_bf = torch.cat(warped_masks_list, dim=0)  # currently unused downstream
                    warped_imgs_bf = warped_imgs_bf.contiguous()
                    warped_imgs_bf_reshaped = rearrange(warped_imgs_bf, "(b f) c h w -> b f c h w", b=B, f=F)
                    warped_imgs = warped_imgs_bf_reshaped.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,F,H,W]
                    if not return_depth:
                        return warped_imgs, None
                    warped_depths_bf = torch.cat(warped_depths_list, dim=0).contiguous()
                    if warped_depths_bf.ndim == 3:
                        warped_depths_bf = warped_depths_bf.unsqueeze(1)
                    warped_depths_bf_reshaped = rearrange(
                        warped_depths_bf, "(b f) c h w -> b f c h w", b=B, f=F
                    )
                    warped_depths = warped_depths_bf_reshaped.permute(0, 2, 1, 3, 4).contiguous()  # [B,1,F,H,W]
                    return warped_imgs, warped_depths

                # -- Multiview-safe accessors ----------------------------------------
                # For multiview inputs stored with negative frame_ids, retrieve
                # camera/depth/rgb from spatial_cache instead of tensor indexing
                # (which would wrap around with negative indices).
                def _mv_w2c(f_id: int) -> torch.Tensor:
                    if int(f_id) < 0:
                        _, w, _ = spatial_cache.get_rgbd_by_frame_id(int(f_id))
                        return w.to(device=device, dtype=torch.float32)
                    return camera_w2c[:, int(f_id)].to(dtype=torch.float32)

                def _mv_K(f_id: int) -> torch.Tensor:
                    if int(f_id) < 0:
                        _, _, k = spatial_cache.get_rgbd_by_frame_id(int(f_id))
                        return k.to(device=device, dtype=torch.float32)
                    return intrinsics[:, int(f_id)].to(dtype=torch.float32)

                # -- end multiview-safe accessors ------------------------------------

                # Accumulated correspondence warp:
                # warp each spatial frame's canonical coordinates separately and
                # append normalized depth as the fourth channel for each slot.
                spatial_frame_ids: list[int] = (
                    [int(x) for x in spatial_selected_frame_ids.tolist()]
                    if spatial_selected_frame_ids is not None
                    else []
                )
                spatial_coords: Optional[torch.Tensor] = spatial_selected_coords

                spatial_unique: list[int] = []
                seen_sp: set[int] = set()
                keep_unique_idx: list[int] = []
                for j, idx in enumerate(spatial_frame_ids):
                    if idx not in seen_sp:
                        spatial_unique.append(int(idx))
                        seen_sp.add(int(idx))
                        keep_unique_idx.append(j)
                spatial_frame_ids = spatial_unique
                if spatial_coords is not None and spatial_coords.numel() > 0:
                    spatial_coords = spatial_coords[:, keep_unique_idx] if keep_unique_idx else spatial_coords[:, :0]

                max_spatial = cfg.multibuffer_max_spatial_frames
                if max_spatial is None:
                    max_spatial = int(self.framepack_num_spatial_hist)
                max_spatial = int(max_spatial)

                assert buffer_depth_B_1_H_W is not None, "buffer_depth_B_1_H_W is required for accumulated correspondence warping"
                assert spatial_cache is not None and spatial_cache._store_values, (
                    "spatial_cache(store_values=True) is required to provide depth for spatial frames"
                )

                # Buffer warp for main conditioning tail
                buf_rgb = video[:, :, abs_buffer_idx].to(dtype=torch.float32).unsqueeze(1)  # [B,1,C,H,W]
                buf_depth = buffer_depth_B_1_H_W.to(device=device, dtype=torch.float32).unsqueeze(1)  # [B,1,1,H,W]
                buf_w2c = camera_w2c[:, abs_buffer_idx].to(dtype=torch.float32).unsqueeze(1)  # [B,1,4,4]
                buf_K = intrinsics[:, abs_buffer_idx].to(dtype=torch.float32).unsqueeze(1)  # [B,1,3,3]
                condition_state_pixels, _ = _warp_multisrc(buf_rgb, buf_depth, buf_w2c, buf_K)

                if max_spatial > 0:
                    spatial_latents: list[torch.Tensor] = []
                    spatial_warped_coords: list[torch.Tensor] = []
                    spatial_warped_depths: list[torch.Tensor] = []
                    for j_rev in range(len(spatial_frame_ids) - 1, -1, -1):  # due to left padding.
                        f_id = spatial_frame_ids[j_rev]
                        assert spatial_coords is not None, (
                            "spatial_selected_coords is required for correspondence multibuffer warping"
                        )
                        src_rgb = spatial_coords[:, j_rev : j_rev + 1].to(
                            device=device, dtype=torch.float32, non_blocking=True
                        )  # [B,1,3,H,W]
                        d, _w, _k = spatial_cache.get_rgbd_by_frame_id(int(f_id))
                        src_depth = d.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
                        src_w2c = _mv_w2c(f_id).unsqueeze(1)
                        src_K = _mv_K(f_id).unsqueeze(1)
                        warped_coords, warped_depth = _warp_multisrc(
                            src_rgb, src_depth, src_w2c, src_K, return_depth=True
                        )
                        assert warped_depth is not None
                        spatial_warped_depths.append(warped_depth)
                        spatial_condition_pixels_list.append(warped_coords)
                        spatial_warped_coords.append(warped_coords)

                    depth_norm_per_spatial: list[torch.Tensor] = []
                    if len(spatial_warped_depths) > 0:
                        depth_stack = torch.cat(spatial_warped_depths, dim=1)  # [B,N,F,H,W]
                        dmin = depth_stack.amin(dim=(1, 2, 3, 4), keepdim=True)
                        dmax = depth_stack.amax(dim=(1, 2, 3, 4), keepdim=True)
                        depth_stack = 2.0 * (depth_stack - dmin) / torch.clamp(dmax - dmin, min=1e-6) - 1.0
                        depth_norm_per_spatial = [depth_stack[:, i : i + 1] for i in range(int(depth_stack.shape[1]))]

                    for i, warped_coords in enumerate(spatial_warped_coords):
                        warped_for_latent = torch.cat([warped_coords, depth_norm_per_spatial[i]], dim=1)
                        coord_lat = self._coord_pixels_to_latents(
                            warped_for_latent, dtype=cond_latent.dtype, target_t=T_new_lat,
                        )
                        spatial_latents.append(coord_lat)

                    if len(spatial_latents) < max_spatial:
                        H_lat = int(cond_latent.shape[3])
                        W_lat = int(cond_latent.shape[4])
                        pad_lat = torch.full(
                            (B, LYRA2_CORRESPONDENCE_CHANNELS_PER_SLOT, T_new_lat, H_lat, W_lat),
                            -1.0,
                            device=cond_latent.device,
                            dtype=cond_latent.dtype,
                        )
                        pad_lat[:, 3 * 8 * 8 :, :, :, :] = 1.0
                        spatial_latents.extend([pad_lat] * (max_spatial - len(spatial_latents)))
                    buffer_cond_latents = torch.cat(spatial_latents, dim=1)
            if self._collect_return_condition_state:
                if len(spatial_condition_pixels_list) > 0:
                    # Show buffer warp + all per-frame spatial warps (like multibuffer vis).
                    vis_list = [condition_state_pixels] + spatial_condition_pixels_list
                    condition_vis = torch.cat(vis_list, dim=1)
                    self._latest_condition_state_pixels = condition_vis
                else:
                    self._latest_condition_state_pixels = condition_state_pixels

            with misc.timer("camera_pose_condition - encode camera tail"):
                camera_latent = self.prepare_latent_conditon(condition_state_pixels, None, cond_latent.dtype)
                assert int(camera_latent.shape[2]) == T_new_lat, (
                    f"Unexpected camera latent T={camera_latent.shape[2]} != {T_new_lat}"
                )
            cond_latent[:, :, -T_new_lat:] = camera_latent.type_as(cond_latent)

        # Plucker ray condition concatenated along channels
        with misc.timer("plucker_condition - build"):
            device = cond_latent.device
            # Gather intrinsics and poses per selected latent (absolute indices)
            K_sel = intrinsics[:, video_indices].to(device=device, dtype=torch.float32)
            w2c_sel = camera_w2c[:, video_indices].to(device=device, dtype=torch.float32)
            c2w_sel = torch.inverse(w2c_sel)
            # c2w_ref for absolute pose mode: last history frame (buffer)
            c2w_ref_inv = w2c_sel[:, -self.framepack_num_new_video_frames - 1:-self.framepack_num_new_video_frames]  # [B,1,4,4]

            # Build intrinsics vector
            fx = K_sel[..., 0, 0]
            fy = K_sel[..., 1, 1]
            cx = K_sel[..., 0, 2]
            cy = K_sel[..., 1, 2]
            K_vec = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, V, 4]

            # Downsample along time to align with VAE latents BEFORE computing rays
            frames_per_lat = int(self.framepack_num_frames_per_latent)
            v_len = int(K_sel.shape[1])
            # subsample long T to 0, 4, 8, 12, ... to match VAE latent indexing used elsewhere
            time_idx = [0] + [i for i in range(4, v_len, frames_per_lat)]
            time_idx_t = torch.tensor(time_idx, device=device, dtype=torch.long)

            # Select downsampled intrinsics and poses
            K_vec_ds = K_vec[:, time_idx_t]              # [B, T_ds, 4]
            c2w_sel_ds = c2w_sel[:, time_idx_t]          # [B, T_ds, 4, 4]

            # Compute relative camera-to-world transforms on the downsampled timeline
            # absolute to the buffer (last history) pose
            c2w_rel_ds = torch.matmul(c2w_ref_inv, c2w_sel_ds)      # [B, T_ds, 4, 4]

            H_pix = int(video.shape[-2])
            W_pix = int(video.shape[-1])
            # Compute Plücker rays on the downsampled timeline
            plucker_sel_B_T_H_W_6 = ray_condition(
                K_vec_ds,
                c2w_rel_ds,
                H_pix,
                W_pix,
                device=device,
                flip_flag=None,
                use_ray_o=True,
            )  # [B, T_ds, H, W, 6]

            # [B, T, H, W, 6] -> [B, 6, T, H, W]
            plucker_5d = plucker_sel_B_T_H_W_6.permute(0, 4, 1, 2, 3).contiguous()

            # Spatial rearrange to latent grid: 6 -> 6*8*8 channels, then reorder by selected_idx
            assert H_pix % 8 == 0 and W_pix % 8 == 0
            plucker_down384 = rearrange(
                plucker_5d,
                "b c t (h h8) (w w8) -> b (c h8 w8) t h w",
                h8=8,
                w8=8,
            )  # [B,384,T,H/8,W/8]
            plucker_down384 = plucker_down384[:, :, selected_idx]

            cond_latent = torch.cat([cond_latent, plucker_down384.type_as(cond_latent)], dim=1)

        return cond_latent, buffer_cond_latents

    @torch.no_grad()
    def _prepare_lyra2_inputs(
        self,
        history_full: torch.Tensor,
        gen_cond: torch.Tensor,
        spatial_cache: Optional[Sparse3DCache],
        video: torch.Tensor,
        buffer_depth_B_1_H_W: Optional[torch.Tensor],
        camera_w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        video_indices: torch.Tensor,
        *,
        is_training: bool = True,
        spatial_cache_skip_last_n: int = 0,
        num_retrieval_views: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Prepare (latents, cond_latent, mask) for Lyra2Model.

        This consolidates the history selection + camera-control tail + optional image-based
        spatial memory insertion.

        Args:
            history_full: [B, C_lat, T_hist, H, W]
            gen_cond:     [B, C_lat, T_new,  H, W] (conditioning tail latents, e.g. zero-tail)
            camera_w2c:   [B, T_video, 4, 4]
            intrinsics:   [B, T_video, 3, 3]
            video_indices: [T_window] absolute indices into the original video timeline
            spatial_cache: pre-built Sparse3DCache instance (built outside; inference uses AR-updated cache)
            video: full video tensor for RGB lookup [B, C, T, H, W]
            buffer_depth_B_1_H_W: last history-frame depth for warping (NOT stored in cache)
            is_training: enables stochastic retrieval only during training
            num_retrieval_views: Number of evenly-spaced target views within the generation
                window used for multi-view coverage retrieval. 1 = single last-frame (legacy).
        """
        cfg = self.config
        device = history_full.device

        num_temporal_hist = int(self.framepack_num_temporal_hist)
        num_spatial_hist = int(self.framepack_num_spatial_hist)

        T_hist_total = int(history_full.shape[2])
        T_new_lat = self.framepack_num_new_latent_frames
        # We don't pass gen_gt into this helper: build a dummy tail for shape bookkeeping.
        gen_lat_dummy = torch.zeros_like(gen_cond)

        temporal_selected = self._select_temporal_history_indices(T_hist_total, num_temporal_hist)

        spatial_selected_frame_ids_t: Optional[torch.Tensor] = None
        spatial_selected: list[int] = []
        spatial_coords_all: Optional[torch.Tensor] = None
        spatial_selected_coords: Optional[torch.Tensor] = None

        if num_spatial_hist > 0:
            H = int(video.shape[-2])
            W = int(video.shape[-1])
            spatial_coords_all = self._get_cached_spatial_coords(
                H=H,
                W=W,
                num_spatial_hist=num_spatial_hist,
                device=device,
                dtype=torch.float32,
            )  # [num_spatial_hist, 3, H, W]

        # Pre-compute retrieval targets (multi-view or single last-frame).
        if num_spatial_hist > 0 and num_retrieval_views > 1 and not is_training:
            T_new_pix = int(self.framepack_num_new_video_frames)
            gen_start = int(video_indices.shape[0]) - T_new_pix
            pts = torch.linspace(0, T_new_pix - 1, num_retrieval_views + 1).long().tolist()
            target_offsets = pts[1:]
            target_abs_list = [int(video_indices[gen_start + off].item()) for off in target_offsets]
            retrieval_w2c = torch.stack(
                [camera_w2c[:, idx].to(device=device, dtype=torch.float32) for idx in target_abs_list], dim=1
            )  # [B, V, 4, 4]
            retrieval_K = torch.stack(
                [intrinsics[:, idx].to(device=device, dtype=torch.float32) for idx in target_abs_list], dim=1
            )  # [B, V, 3, 3]
            log.info(
                f"Multi-view retrieval: {num_retrieval_views} views at abs indices {target_abs_list}",
                rank0_only=True,
            )
        elif num_spatial_hist > 0:
            last_abs = int(video_indices[-1].item())
            retrieval_w2c = camera_w2c[:, last_abs].to(device=device, dtype=torch.float32)  # [B, 4, 4]
            retrieval_K = intrinsics[:, last_abs].to(device=device, dtype=torch.float32)  # [B, 3, 3]

        if cfg.spatial_memory_use_image and num_spatial_hist > 0:
            assert spatial_cache is not None, "spatial_cache is required when spatial_memory_use_image=True"
            H = int(video.shape[-2])
            W = int(video.shape[-1])

            retrieved = spatial_cache.retrieve(
                retrieval_w2c,
                retrieval_K,
                (H, W),
                num_latents=num_spatial_hist,
                skip_last_n=int(spatial_cache_skip_last_n),
                random=bool(is_training),
                max_coverage=not bool(is_training),
            )
            spatial_selected_frame_ids_t = torch.tensor([int(fi) for (_li, fi) in retrieved], device=device, dtype=torch.long)
            if spatial_coords_all is not None:
                n_retrieved = int(spatial_selected_frame_ids_t.numel())
                offset = max(0, int(num_spatial_hist - n_retrieved))
                coords_sel = spatial_coords_all[offset : offset + n_retrieved]
                if coords_sel.numel() > 0:
                    B0 = int(history_full.shape[0])
                    spatial_selected_coords = coords_sel.unsqueeze(0).repeat(B0, 1, 1, 1, 1)

            # Post-retrieval spatial memory dropout: randomly drop 1..N of selected spatial frames.
            if is_training and float(cfg.spatial_memory_drop_rate) > 0:
                N_sel = int(spatial_selected_frame_ids_t.numel()) if spatial_selected_frame_ids_t is not None else 0
                if N_sel > 0 and torch.rand(1, device=device).item() < float(cfg.spatial_memory_drop_rate):
                    num_drop = int(torch.randint(1, N_sel + 1, (1,), device=device).item())
                    perm = torch.randperm(N_sel, device=device)
                    keep_idx = perm[num_drop:].sort().values
                    spatial_selected_frame_ids_t = spatial_selected_frame_ids_t[keep_idx]
                    # Re-align canonical coords with the new count so that the right-aligned
                    # offset matches the left-padding of image tokens at s-positions.
                    if spatial_coords_all is not None:
                        n_post = int(spatial_selected_frame_ids_t.numel())
                        offset_post = max(0, int(num_spatial_hist - n_post))
                        coords_post = spatial_coords_all[offset_post : offset_post + n_post]
                        if coords_post.numel() > 0:
                            B0 = int(history_full.shape[0])
                            spatial_selected_coords = coords_post.unsqueeze(0).repeat(B0, 1, 1, 1, 1)
                        else:
                            spatial_selected_coords = None
                    log.info(
                        f"Spatial memory dropout: dropped {num_drop}/{N_sel}, kept {N_sel - num_drop}",
                        rank0_only=True,
                    )

            # For the base history selection, use temporal-only indices (spatial slots will be inserted later).
            splits_temp = [s for s, t in zip(self.framepack_clean_latent_frame_splits, self.framepack_clean_latent_frame_kernel_types) if t == "k"]
            types_temp = ["k"] * len(splits_temp)
            selected_idx_hist = self._compose_selected_indices(
                splits=splits_temp,
                types=types_temp,
                T_hist_total=T_hist_total,
                temporal_selected=temporal_selected,
                spatial_selected=[],
            )
        elif num_spatial_hist > 0:
            raise NotImplementedError(
                "Lyra2Model is collapsed to the image-token target branch and requires spatial_memory_use_image=True"
            )
        else:
            selected_idx_hist = self._compose_selected_indices(
                splits=self.framepack_clean_latent_frame_splits,
                types=self.framepack_clean_latent_frame_kernel_types,
                T_hist_total=T_hist_total,
                temporal_selected=temporal_selected,
                spatial_selected=[],
            )
        # Mask: history always clean; generation tail masked out only when no pose conditioning.
        B, C_lat, _, H_lat, W_lat = history_full.shape
        mask_hist = torch.ones(B, 4, int(selected_idx_hist.shape[0]), H_lat, W_lat, dtype=history_full.dtype, device=device)
        mask_gen = torch.ones(B, 4, T_new_lat, H_lat, W_lat, dtype=history_full.dtype, device=device)

        # Reorder history and concatenate tails.
        latents_hist = history_full[:, :, selected_idx_hist]
        cond_hist = latents_hist
        latents = torch.cat([latents_hist, gen_lat_dummy], dim=2)
        cond_latent = torch.cat([cond_hist, gen_cond], dim=2)
        mask = torch.cat([mask_hist, mask_gen], dim=2)

        # Build full selected_idx for plucker alignment (history indices + gen indices).
        gen_idx = torch.arange(T_hist_total, T_hist_total + T_new_lat, device=device, dtype=torch.long)
        selected_idx_full = torch.cat([selected_idx_hist, gen_idx], dim=0)

        # Camera controls (depth warp + plucker) applied after reordering.
        cond_latent, buffer_cond_latents = self._apply_camera_controls(
            cond_latent,
            selected_idx_full,
            video_indices,
            spatial_selected_frame_ids=spatial_selected_frame_ids_t,
            spatial_selected_coords=spatial_selected_coords,
            video=video,
            camera_w2c=camera_w2c,
            intrinsics=intrinsics,
            buffer_depth_B_1_H_W=buffer_depth_B_1_H_W,
            spatial_cache=spatial_cache,
            is_training=bool(is_training),
        )

        # Optional image-based spatial memory insertion (encode retrieved frames and interleave).
        if cfg.spatial_memory_use_image:
            spatial_image_ids: list[int] = spatial_selected_frame_ids_t.tolist() if spatial_selected_frame_ids_t is not None and spatial_selected_frame_ids_t.numel() > 0 else []
            if self.framepack_num_spatial_hist <= 0:
                spatial_image_ids = []
            if len(spatial_image_ids) < self.framepack_num_spatial_hist:
                spatial_image_ids = [video_indices[0].item()] * (self.framepack_num_spatial_hist - len(spatial_image_ids)) + spatial_image_ids

            spatial_latents_list = []
            for t in spatial_image_ids:
                if int(t) < 0:
                    mv_rgb = spatial_cache.get_rgb_by_frame_id(int(t)).to(device=device, dtype=video.dtype)
                    if mv_rgb.dim() == 4:
                        mv_rgb = mv_rgb.unsqueeze(2)
                    spatial_latents_list.append(self.encode(mv_rgb))
                else:
                    spatial_latents_list.append(self.encode(video[:, :, t : t + 1]))
            spatial_latents = torch.cat(spatial_latents_list, dim=2)

            spatial_plucker = None
            if spatial_image_ids:
                # Multiview-safe: build K_sel/w2c_sel per-element for negative frame IDs.
                if any(int(t) < 0 for t in spatial_image_ids):
                    K_list, w2c_list = [], []
                    for t in spatial_image_ids:
                        if int(t) < 0:
                            _, w, k = spatial_cache.get_rgbd_by_frame_id(int(t))
                            w2c_list.append(w.to(device=device, dtype=torch.float32))
                            K_list.append(k.to(device=device, dtype=torch.float32))
                        else:
                            w2c_list.append(camera_w2c[:, int(t)].to(dtype=torch.float32))
                            K_list.append(intrinsics[:, int(t)].to(dtype=torch.float32))
                    K_sel = torch.stack(K_list, dim=1)  # [B, T_sp, 3, 3]
                    w2c_sel = torch.stack(w2c_list, dim=1)
                else:
                    spatial_image_ids_t = torch.tensor(spatial_image_ids, device=device, dtype=torch.long)
                    K_sel = intrinsics[:, spatial_image_ids_t]  # [B, T_sp, 3, 3]
                    w2c_sel = camera_w2c[:, spatial_image_ids_t]
                c2w_sel = torch.inverse(w2c_sel)
                ref_idx = video_indices[-int(self.framepack_num_new_video_frames) - 1]
                c2w_ref_inv = camera_w2c[:, ref_idx : ref_idx + 1]
                c2w_rel = torch.matmul(c2w_ref_inv, c2w_sel)
                fx = K_sel[..., 0, 0]
                fy = K_sel[..., 1, 1]
                cx = K_sel[..., 0, 2]
                cy = K_sel[..., 1, 2]
                K_vec_sp = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, T_sp, 4]
                plucker = ray_condition(
                    K_vec_sp,
                    c2w_rel,
                    int(video.shape[-2]),
                    int(video.shape[-1]),
                    device=device,
                    flip_flag=None,
                    use_ray_o=True,
                )
                plucker_5d = plucker.permute(0, 4, 1, 2, 3).contiguous()
                spatial_plucker = rearrange(
                    plucker_5d,
                    "b c t (h h8) (w w8) -> b (c h8 w8) t h w",
                    h8=8,
                    w8=8,
                )

            # Insert spatial latents into history according to (splits, types), keep gen tail at end.
            final_latents = []
            final_cond = []
            final_mask = []
            gen_ptr = latents.shape[2] - int(self.framepack_num_new_latent_frames)
            current_temp_lat = latents[:, :, :gen_ptr]
            current_gen_lat = latents[:, :, gen_ptr:]
            current_temp_cond = cond_latent[:, :, :gen_ptr]
            current_gen_cond = cond_latent[:, :, gen_ptr:]
            current_temp_mask = mask[:, :, :gen_ptr]
            current_gen_mask = mask[:, :, gen_ptr:]
            t_cursor = 0
            s_cursor = 0
            B0 = latents.shape[0]
            for s_cnt, tp in zip(self.framepack_clean_latent_frame_splits, self.framepack_clean_latent_frame_kernel_types):
                if tp == "k":
                    final_latents.append(current_temp_lat[:, :, t_cursor : t_cursor + s_cnt])
                    final_cond.append(current_temp_cond[:, :, t_cursor : t_cursor + s_cnt])
                    final_mask.append(current_temp_mask[:, :, t_cursor : t_cursor + s_cnt])
                    t_cursor += int(s_cnt)
                elif tp == "s":
                    chunk_lat = spatial_latents[:, :, s_cursor : s_cursor + s_cnt]
                    final_latents.append(chunk_lat)
                    if spatial_plucker is not None:
                        chunk_plucker = spatial_plucker[:, :, s_cursor : s_cursor + s_cnt]
                        chunk_cond = torch.cat([chunk_lat, chunk_plucker.type_as(chunk_lat)], dim=1)
                    else:
                        chunk_cond = chunk_lat
                    final_cond.append(chunk_cond)
                    chunk_mask = torch.ones(
                        B0,
                        4,
                        int(s_cnt),
                        latents.shape[3],
                        latents.shape[4],
                        device=device,
                        dtype=latents.dtype,
                    )
                    final_mask.append(chunk_mask)
                    s_cursor += int(s_cnt)

            final_latents.append(current_gen_lat)
            final_cond.append(current_gen_cond)
            final_mask.append(current_gen_mask)
            latents = torch.cat(final_latents, dim=2)
            cond_latent = torch.cat(final_cond, dim=2)
            mask = torch.cat(final_mask, dim=2)

        # Pad buffer_cond_latents to full length and inject spatial 3D coordinates.
        B0 = int(cond_latent.shape[0])
        H_lat = int(cond_latent.shape[3])
        W_lat = int(cond_latent.shape[4])
        T_hist = int(cond_latent.shape[2]) - int(self.framepack_num_new_latent_frames)

        _max_spatial = cfg.multibuffer_max_spatial_frames
        if _max_spatial is None:
            _max_spatial = int(self.framepack_num_spatial_hist)
        _max_spatial = int(_max_spatial)

        C_coord_lat = LYRA2_CORRESPONDENCE_CHANNELS_PER_SLOT * _max_spatial

        def _coords_to_slotted_latent(coords_b: torch.Tensor) -> torch.Tensor:
            """[B, 3, T, H, W] raw pixel coords → [B, _max_spatial*4*8*8, T, H_lat, W_lat]."""
            d_minus = torch.full(
                (B0, 1, coords_b.shape[2], coords_b.shape[3], coords_b.shape[4]),
                -1.0, device=coords_b.device, dtype=coords_b.dtype,
            )
            d_plus = torch.ones_like(d_minus)
            lat_minus = self._pixelshuffle_hw_to_latent(
                torch.cat([coords_b, d_minus], dim=1)
            ).to(dtype=cond_latent.dtype)
            lat_plus = self._pixelshuffle_hw_to_latent(
                torch.cat([coords_b, d_plus], dim=1)
            ).to(dtype=cond_latent.dtype)
            if _max_spatial <= 0:
                return lat_minus[:, :0]
            if _max_spatial == 1:
                return lat_minus
            return torch.cat([lat_minus] + [lat_plus] * (_max_spatial - 1), dim=1)

        buffer_hist_chunks: list[torch.Tensor] = []
        C_buf = int(buffer_cond_latents.shape[1]) if buffer_cond_latents is not None else C_coord_lat

        # Convert spatial coords to slotted latent space.
        spatial_coords_latent: Optional[torch.Tensor] = None
        if spatial_coords_all is not None and num_spatial_hist > 0:
            spatial_coords_all_b = spatial_coords_all.unsqueeze(0).permute(0, 2, 1, 3, 4).repeat(B0, 1, 1, 1, 1)
            spatial_coords_latent = _coords_to_slotted_latent(spatial_coords_all_b)

        # Assemble per-chunk history buffer with debug logging.
        s_cursor = 0
        _debug_labels: list[str] = []
        for _pos, (s_cnt, tp) in enumerate(zip(self.framepack_clean_latent_frame_splits, self.framepack_clean_latent_frame_kernel_types)):
            s_cnt = int(s_cnt)
            if tp == "s" and spatial_coords_latent is not None:
                chunk = spatial_coords_latent[:, :, s_cursor : s_cursor + s_cnt]
                buffer_hist_chunks.append(chunk)
                s_cursor += int(s_cnt)
                _debug_labels.append(f"pos{_pos}:f{s_cnt}{tp}=spatial_coord")
            else:
                buffer_hist_chunks.append(
                    torch.zeros((B0, C_buf, s_cnt, H_lat, W_lat), device=cond_latent.device, dtype=cond_latent.dtype)
                )
                _debug_labels.append(f"pos{_pos}:f{s_cnt}{tp}=zeros")
        log.info(
            f"Buffer hist padding: {_debug_labels}, C_buf={C_buf}, _max_spatial={_max_spatial}",
            rank0_only=True,
        )

        buffer_hist = torch.cat(buffer_hist_chunks, dim=2) if buffer_hist_chunks else torch.zeros(
            (B0, C_buf, T_hist, H_lat, W_lat),
            device=cond_latent.device,
            dtype=cond_latent.dtype,
        )
        if buffer_cond_latents is None:
            buffer_tail = torch.zeros(
                (B0, C_buf, T_new_lat, H_lat, W_lat),
                device=cond_latent.device,
                dtype=cond_latent.dtype,
            )
        else:
            buffer_tail = buffer_cond_latents.to(dtype=cond_latent.dtype)
        buffer_cond_latents = torch.cat([buffer_hist, buffer_tail], dim=2)

        return latents, cond_latent, mask, buffer_cond_latents

    @torch.no_grad()
    def _tokenizing_video_to_latents(self, video, dropout=False, data_batch=None):
        cfg = self.config
        assert data_batch is not None, "Lyra2Model._tokenizing_video_to_latents requires data_batch"
        with misc.timer("_tokenizing_video_to_latents(spatial) - total"):
            # Step 1: windowing
            video, video_indices, start, cur_segment_id, chunk_len = self._prepare_video_window(
                video,
                data_batch.get("start") if data_batch is not None else None,
                data_batch.get("cur_segment_id") if data_batch is not None else None,
            )

            # Encode latents and cond_latent with shared prefix
            with misc.timer("vae_encoding - shared prefix"):
                if cfg.self_aug_enabled:
                    if "_stage_a_vae_cache_T-2" not in data_batch: # self aug step. Save vae cache
                        out = self._vae_encode_with_shared_prefix(video, None, return_cache=True)
                        latents, cond_latent, cache_after_prefix, cache_current = cast(tuple[torch.Tensor, torch.Tensor, Any, Any], out)
                        data_batch["_stage_a_full_latents"] = latents.clone()
                        data_batch["_stage_a_vae_cache_T-2"] = cache_after_prefix
                        data_batch["_stage_a_vae_cache_T-1"] = cache_current
                    else:
                        # Temporal slices along T dimension
                        prev_gen_chunk_aug = video[:, :, -2 * self.framepack_num_new_video_frames : -1 * self.framepack_num_new_video_frames] # self-augmented previous chunk
                        curr_gen_chunk = video[:, :, -1 * self.framepack_num_new_video_frames :] # clean current chunk
                        # 1) Encode self-augmented previous-chunk
                        feat1_enc, cache_after_prev = self.vae_encode_with_cache(
                            data_batch["_stage_a_vae_cache_T-2"],
                            prev_gen_chunk_aug,
                            start_t=0,
                            end_t=prev_gen_chunk_aug.shape[2],
                            return_cache=True,
                        )
                        # 2) Encode zero-tail for the next chunk
                        zeros_last = torch.zeros_like(prev_gen_chunk_aug)
                        feat2_enc = self.vae_encode_with_cache(
                            cache_after_prev,
                            zeros_last,
                            start_t=0,
                            end_t=zeros_last.shape[2],
                            return_cache=False,
                        )
                        # 3) Encode GT with clean cache
                        feat3_enc = self.vae_encode_with_cache(
                            data_batch["_stage_a_vae_cache_T-1"],
                            curr_gen_chunk,
                            start_t=0,
                            end_t=curr_gen_chunk.shape[2],
                            return_cache=False,
                        )
                        # Convert encoder feats to normalized latents using shared helper
                        lat1 = self._encoder_feats_to_normalized_latents(feat1_enc)
                        lat2 = self._encoder_feats_to_normalized_latents(feat2_enc)
                        lat3 = self._encoder_feats_to_normalized_latents(feat3_enc)
                        # Cast to input dtype and stitch
                        in_dtype = video.dtype
                        lat1 = lat1.contiguous().to(in_dtype)
                        lat2 = lat2.contiguous().to(in_dtype)
                        lat3 = lat3.contiguous().to(in_dtype)
                        # replace previous chunk with self-augmented latents, and concatenate with clean gt / zero latents
                        latents = torch.cat([data_batch["_stage_a_full_latents"][:, :, :-self.framepack_num_new_latent_frames], lat1, lat3], dim=2)
                        cond_latent = torch.cat([data_batch["_stage_a_full_latents"][:, :, :-self.framepack_num_new_latent_frames], lat1, lat2], dim=2)
                        del data_batch["_stage_a_full_latents"]
                else:
                    out2 = self._vae_encode_with_shared_prefix(video, None, return_cache=False)
                    latents, cond_latent = cast(tuple[torch.Tensor, torch.Tensor], out2)

            history_full = latents[:, :, : -self.framepack_num_new_latent_frames]
            gen_gt = latents[:, :, -self.framepack_num_new_latent_frames :]
            gen_cond = cond_latent[:, :, -self.framepack_num_new_latent_frames :]

            # Build Sparse3DCache OUTSIDE _prepare_lyra2_inputs.
            # In inference, this cache is maintained incrementally during AR, so the build logic differs.
            splits = self.framepack_clean_latent_frame_splits
            types = self.framepack_clean_latent_frame_kernel_types
            num_spatial_hist = int(sum(s for s, t in zip(splits, types) if t == "s"))
            num_temporal_hist = int(sum(s for s, t in zip(splits, types) if t == "k"))
            _ = num_temporal_hist  # explicit for readability

            spatial_cache: Optional[Sparse3DCache] = None
            buffer_depth_B_1_H_W: Optional[torch.Tensor] = None
            # Last history frame (buffer) absolute index; depth passed explicitly for warping and NOT cached.
            rel_buffer_idx = video_indices.shape[0] - self.framepack_num_new_video_frames - 1
            abs_buffer_idx = int(video_indices[rel_buffer_idx].item())
            buffer_depth_B_1_H_W = data_batch["depth"][:, abs_buffer_idx].to(device=latents.device, dtype=torch.float32)
            if buffer_depth_B_1_H_W.dim() == 3:
                buffer_depth_B_1_H_W = buffer_depth_B_1_H_W.unsqueeze(1)

            if num_spatial_hist > 0:
                spatial_cache = Sparse3DCache(
                    downsample=4,
                    store_device=str(latents.device),
                    store_values=True,
                )

                if cfg.spatial_memory_use_image:
                    # Image-based spatial memory: cache over global timeline, excluding frames near the current window.
                    is_i2v = bool(data_batch.get("is_i2v", False))
                    use_only_first = is_i2v
                    if use_only_first:
                        log.info(
                            f"Spatial memory: only use first frame (is_i2v={is_i2v}).",
                            rank0_only=True,
                        )
                        if int(data_batch["video"].shape[2]) > 0:
                            spatial_cache.add(
                                data_batch["depth"][:, video_indices[0].item()].to(device=latents.device, dtype=torch.float32),
                                data_batch["camera_w2c"][:, video_indices[0].item()].to(device=latents.device, dtype=torch.float32),
                                data_batch["intrinsics"][:, video_indices[0].item()].to(device=latents.device, dtype=torch.float32),
                                latent_index=0,
                                frame_id=video_indices[0].item(),
                            )
                    else:
                        skip_recent = int(cfg.spatial_memory_skip_recent)
                        stride = max(int(cfg.spatial_memory_stride), 1)
                        t0 = int(video_indices[-int(self.framepack_num_new_video_frames)].item())
                        t1 = int(video_indices[-1].item())
                        abs_buffer_idx = int(video_indices[video_indices.shape[0] - self.framepack_num_new_video_frames - 1].item())
                        for t in range(int(data_batch["video"].shape[2])):
                            if t == abs_buffer_idx:
                                continue
                            if t < (t0 - skip_recent) or t > (t1 + skip_recent):
                                if (t % stride == 0) and t != 0:
                                    spatial_cache.add(
                                        data_batch["depth"][:, t].to(device=latents.device, dtype=torch.float32),
                                        data_batch["camera_w2c"][:, t].to(device=latents.device, dtype=torch.float32),
                                        data_batch["intrinsics"][:, t].to(device=latents.device, dtype=torch.float32),
                                        latent_index=int(t),
                                        frame_id=int(t),
                                    )

            # Prepare final (latents, cond_latent, mask) using the unified helper.
            with misc.timer("post - prepare_lyra2_inputs"):
                latents, cond_latent, mask, buffer_cond_latents = self._prepare_lyra2_inputs(
                    history_full=history_full,
                    gen_cond=gen_cond,
                    spatial_cache=spatial_cache,
                    video=data_batch["video"].to(device=latents.device, dtype=latents.dtype),
                    buffer_depth_B_1_H_W=buffer_depth_B_1_H_W,
                    camera_w2c=data_batch["camera_w2c"],
                    intrinsics=data_batch["intrinsics"],
                    video_indices=video_indices,
                    is_training=True,
                )
            data_batch["cond_latent_buffer"] = buffer_cond_latents
            # Replace dummy tail latents with the actual ground-truth tail for training.
            latents = torch.cat([latents[:, :, : -self.framepack_num_new_latent_frames], gen_gt], dim=2)

            # Recompute visualization rays so they align with final latent order.
            if self._collect_return_condition_state:
                C_lat = latents.shape[1]
                plucker_grid = cond_latent[:, C_lat:, :, :, :]  # [B, 6*8*8, T_lat, H_lat, W_lat]
                if plucker_grid.numel() > 0:
                    plucker_lat = rearrange(
                        plucker_grid,
                        "b (c h8 w8) t h w -> b t (h h8) (w w8) c",
                        h8=8,
                        w8=8,
                    )  # [B, T_lat, H_pix, W_pix, 6]
                    B_vis, T_lat, H_pix, W_pix, _ = plucker_lat.shape
                    F = int(self.framepack_num_frames_per_latent)
                    if T_lat <= 1 or F <= 1:
                        plucker_vis = plucker_lat
                    else:
                        head = plucker_lat[:, :1]  # first latent -> 1 frame
                        tail = plucker_lat[:, 1:]  # remaining latents
                        tail_rep = tail.unsqueeze(2).repeat(1, 1, F, 1, 1, 1)
                        tail_rep = tail_rep.reshape(B_vis, (T_lat - 1) * F, H_pix, W_pix, 6)
                        plucker_vis = torch.cat([head, tail_rep], dim=1)  # [B, 1 + (T_lat-1)*F, H, W, 6]
                    M_all = plucker_vis[..., 0:3]
                    d_all = plucker_vis[..., 3:6]
                    self._latest_plucker_rays_pixels = {
                        "ray_origin": M_all.detach(),
                        "ray_direction": d_all.detach(),
                    }

            # Optional corruption/augmentation (must always run, regardless of visualization flags).
            if self.config.apply_corruption_to_spatial_region != "none":
                self._apply_spatial_region_corruption(latents, cond_latent)

            # last history frame in pixel space (from pre-encoded video timeline)
            last_hist_frame = video[:, :, -self.framepack_num_new_video_frames - 1].clone()

            data_batch["_stage_a_video_indices"] = video_indices
            data_batch["_stage_a_start"] = start
            data_batch["_stage_a_cur_segment_id"] = cur_segment_id
            data_batch["_stage_a_chunk_len"] = chunk_len
            data_batch["_stage_a_video_shape"] = video.shape

            if "t5_chunk_keys" in data_batch:
                # absolute index of the last history frame (first frame before generation)
                rel_gen_first_idx = int(video_indices[-int(self.framepack_num_new_video_frames)].item())
                sample_frame_indices = data_batch["sample_frame_indices"]  # [B, F]
                t5_chunk_keys = data_batch["t5_chunk_keys"]               # [B, K]
                t5_chunk_embeddings = data_batch["t5_chunk_embeddings"]   # [B, K, 512, 4096]
                t5_chunk_mask = data_batch["t5_chunk_mask"]               # [B, K, 512]
                assert torch.is_tensor(sample_frame_indices) and torch.is_tensor(t5_chunk_keys)
                assert torch.is_tensor(t5_chunk_embeddings) and torch.is_tensor(t5_chunk_mask)
                B = int(t5_chunk_keys.shape[0])
                # Per-sample absolute index into original sequence
                first_abs_idx_B = sample_frame_indices[:, rel_gen_first_idx].to(dtype=torch.long)  # [B]
                selected_emb_list = []
                selected_mask_list = []
                for b in range(B):
                    keys_b = t5_chunk_keys[b]  # [K], ascending
                    K = int(keys_b.numel())
                    val = int(first_abs_idx_B[b].item())
                    # strictly smaller w.r.t first index where key > val, then minus 1
                    pos = torch.searchsorted(keys_b, torch.tensor([val], device=keys_b.device, dtype=keys_b.dtype), right=True).item()
                    sel_idx = max(0, min(int(pos) - 1, K - 1))
                    emb_b = t5_chunk_embeddings[b, sel_idx]  # [512, 4096]
                    msk_b = t5_chunk_mask[b, sel_idx]        # [512]

                    sel_key = int(keys_b[sel_idx].item()) if K > 0 else -1

                    selected_emb_list.append(emb_b)
                    selected_mask_list.append(msk_b)
                data_batch["t5_text_embeddings"] = torch.stack(selected_emb_list, dim=0)  # [B, 512, 4096]
                data_batch["t5_text_mask"] = torch.stack(selected_mask_list, dim=0)      # [B, 512]
            return latents, last_hist_frame, cond_latent, mask


class Sparse3DCache:
    def __init__(
        self,
        downsample: int = 4,
        store_device: str = "cuda",
        store_values: bool = False,
    ) -> None:
        self.downsample = int(downsample)
        self._store_device = str(store_device)
        self._store_values = bool(store_values)
        self._world_points: list[torch.Tensor] = []  # each: [B, H', W', 3]
        self._latent_indices: list[int] = []        # latent index per entry
        self._frame_ids: list[int] = []             # original video frame id per entry
        # Optional raw RGBD camera storage for value lookup (used in inference warping).
        self._depths: list[torch.Tensor] = []
        self._w2cs: list[torch.Tensor] = []
        self._Ks: list[torch.Tensor] = []
        # Multiview RGB storage keyed by frame_id (only populated for multiview inputs).
        self._rgbs: dict[int, torch.Tensor] = {}

    @staticmethod
    def _scale_intrinsics(intrinsic: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale pinhole intrinsics for a downsampled grid by factor `scale` (e.g., 1/4)."""
        assert intrinsic.dim() == 3 and intrinsic.shape[-2:] == (3, 3)
        K = intrinsic.clone()
        K[:, 0, 0] = K[:, 0, 0] * scale
        K[:, 1, 1] = K[:, 1, 1] * scale
        K[:, 0, 2] = K[:, 0, 2] * scale
        K[:, 1, 2] = K[:, 1, 2] * scale
        return K

    def add(
        self,
        depth_B_1_H_W: torch.Tensor,
        w2c_B_4_4: torch.Tensor,
        K_B_3_3: torch.Tensor,
        latent_index: int,
        frame_id: Optional[int] = None,
    ) -> None:
        ds = self.downsample
        # Subsample depth and scale intrinsics accordingly
        depth_ds = depth_B_1_H_W[:, :, ::ds, ::ds]
        scale = 1.0 / float(ds)
        K_scaled = self._scale_intrinsics(K_B_3_3, scale)
        # Valid mask where depth > 0
        mask_valid = (depth_ds > 0)
        world_pts: torch.Tensor = unproject_points(
            depth=depth_ds,
            w2c=w2c_B_4_4,
            intrinsic=K_scaled,
            is_depth=True,
            is_ftheta=False,
            mask=mask_valid,
            return_sparse=False,
        )  # [B, H', W', 3]
        if self._store_device == "cpu":
            world_pts = world_pts.detach().to("cpu", non_blocking=True)
        self._world_points.append(world_pts)
        self._latent_indices.append(int(latent_index))
        self._frame_ids.append(int(latent_index) if frame_id is None else int(frame_id))
        if self._store_values:
            # Store full-res values (no downsample) for later retrieval by frame_id.
            d = depth_B_1_H_W.detach()
            w = w2c_B_4_4.detach()
            k = K_B_3_3.detach()
            if self._store_device == "cpu":
                d = d.to("cpu", non_blocking=True)
                w = w.to("cpu", non_blocking=True)
                k = k.to("cpu", non_blocking=True)
            self._depths.append(d)
            self._w2cs.append(w)
            self._Ks.append(k)

    def store_rgb(self, frame_id: int, rgb: torch.Tensor) -> None:
        """Store RGB pixels for a frame (used for multiview inputs with negative frame_id)."""
        t = rgb.detach()
        if self._store_device == "cpu":
            t = t.to("cpu", non_blocking=True)
        self._rgbs[int(frame_id)] = t

    def get_rgb_by_frame_id(self, frame_id: int) -> torch.Tensor:
        """Return stored RGB for a frame_id. Raises KeyError if not found."""
        fid = int(frame_id)
        if fid not in self._rgbs:
            raise KeyError(f"frame_id={fid} not found in Sparse3DCache RGB storage")
        return self._rgbs[fid]

    def get_rgbd_by_frame_id(self, frame_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (depth, w2c, K) for a stored frame_id.

        Requires store_values=True at construction time.
        """
        if not self._store_values:
            raise RuntimeError("Sparse3DCache.get_rgbd_by_frame_id requires store_values=True")
        # Prefer most recent match if duplicated.
        for i in range(len(self._frame_ids) - 1, -1, -1):
            if int(self._frame_ids[i]) == int(frame_id):
                return self._depths[i], self._w2cs[i], self._Ks[i]
        raise KeyError(f"frame_id={int(frame_id)} not found in Sparse3DCache")

    def update_by_frame_id(
        self,
        frame_id: int,
        depth_B_1_H_W: torch.Tensor,
        w2c_B_4_4: torch.Tensor,
        K_B_3_3: torch.Tensor,
    ) -> bool:
        """Replace depth/w2c/K and recompute world points for an existing frame_id.

        Returns True if the frame was found and updated, False otherwise.
        """
        fid = int(frame_id)
        idx = None
        for i in range(len(self._frame_ids)):
            if int(self._frame_ids[i]) == fid:
                idx = i
                break
        if idx is None:
            return False

        # Ensure all tensors are on the same device for unproject_points.
        compute_device = depth_B_1_H_W.device
        _depth = depth_B_1_H_W.to(compute_device)
        _w2c = w2c_B_4_4.to(compute_device)
        _K = K_B_3_3.to(compute_device)

        ds = self.downsample
        depth_ds = _depth[:, :, ::ds, ::ds]
        scale = 1.0 / float(ds)
        K_scaled = self._scale_intrinsics(_K, scale)
        mask_valid = (depth_ds > 0)
        world_pts: torch.Tensor = unproject_points(
            depth=depth_ds,
            w2c=_w2c,
            intrinsic=K_scaled,
            is_depth=True,
            is_ftheta=False,
            mask=mask_valid,
            return_sparse=False,
        )
        if self._store_device == "cpu":
            world_pts = world_pts.detach().to("cpu", non_blocking=True)
        self._world_points[idx] = world_pts
        if self._store_values:
            d = depth_B_1_H_W.detach()
            w = w2c_B_4_4.detach()
            k = K_B_3_3.detach()
            if self._store_device == "cpu":
                d = d.to("cpu", non_blocking=True)
                w = w.to("cpu", non_blocking=True)
                k = k.to("cpu", non_blocking=True)
            self._depths[idx] = d
            self._w2cs[idx] = w
            self._Ks[idx] = k
        return True


    @torch.no_grad()
    def retrieve(
        self,
        target_w2c_B_4_4: torch.Tensor,
        target_K_B_3_3: torch.Tensor,
        target_hw: tuple[int, int],
        num_latents: int,
        skip_last_n: int = 0,
        random: bool = False,
        max_coverage: bool = False,
        depth_threshold: float = 0.1,
    ) -> list[tuple[int, int]]:
        """Retrieve the best candidate frames from the cache.

        Args:
            target_w2c_B_4_4: Target world-to-camera matrices.
                Single view [B, 4, 4] or multi-view [B, V, 4, 4].
            target_K_B_3_3: Target intrinsics.
                Single view [B, 3, 3] or multi-view [B, V, 3, 3].
            target_hw: (H, W) of the target image in pixels.
            num_latents: Maximum number of candidates to return.
            skip_last_n: Skip the most recent N entries in the cache.
            random: Stochastic sampling (training only).
            max_coverage: Greedy set-cover maximizing pixel coverage.
                When multi-view targets are given, coverage is maximized
                across the union of all views' pixels.
            depth_threshold: Tolerance for depth-based occlusion filtering.
        """
        Ht, Wt = target_hw
        num_total = len(self._world_points)
        if num_total == 0 or num_latents <= 0:
            return []
        device = target_w2c_B_4_4.device
        ds = self.downsample
        scale = 1.0 / float(ds)
        Ht_ds = int((Ht + ds - 1) // ds)
        Wt_ds = int((Wt + ds - 1) // ds)

        # Handle multi-view [B, V, 4, 4] vs single-view [B, 4, 4]
        if target_w2c_B_4_4.dim() == 4:
            num_views = int(target_w2c_B_4_4.shape[1])
            w2c_views = [target_w2c_B_4_4[:, v] for v in range(num_views)]
            K_views = [target_K_B_3_3[:, v] for v in range(num_views)]
        else:
            num_views = 1
            w2c_views = [target_w2c_B_4_4]
            K_views = [target_K_B_3_3]

        s = int(skip_last_n) if skip_last_n is not None else 0
        avail = max(0, num_total - max(0, s))
        if avail <= 0:
            return []

        num_cands = avail
        pts_list = self._world_points[:avail]

        # Vectorized projection of all (view, candidate) pairs at once.
        # pts_stacked: [C, B, H', W', 3]
        pts_stacked = torch.stack([p.to(device=device) for p in pts_list], dim=0)
        C, Bp, Hp, Wp, _ = pts_stacked.shape

        # Homogeneous coordinates: [C, B, H', W', 4, 1]
        ones_hw = torch.ones(C, Bp, Hp, Wp, 1, device=device, dtype=pts_stacked.dtype)
        pts_homo = torch.cat([pts_stacked, ones_hw], dim=-1).unsqueeze(-1)

        # Build per-view downsampled intrinsics: [V, B, 3, 3] and [V, B, 4, 4]
        K_ds_views = [self._scale_intrinsics(K_v, scale) for K_v in K_views]
        w2c_stack = torch.stack(w2c_views, dim=0)       # [V, B, 4, 4]
        K_ds_stack = torch.stack(K_ds_views, dim=0)      # [V, B, 3, 3]

        # Broadcast matmul: w2c [V,1,B,1,1,4,4] x pts_homo [1,C,B,H',W',4,1]
        cam_homo = torch.matmul(
            w2c_stack[:, None, :, None, None],   # [V, 1, B, 1, 1, 4, 4]
            pts_homo[None],                      # [1, C, B, H', W', 4, 1]
        )                                        # [V, C, B, H', W', 4, 1]
        cam_pts = cam_homo[..., :3, :]           # [V, C, B, H', W', 3, 1]

        # Broadcast matmul: K [V,1,B,1,1,3,3] x cam_pts [V,C,B,H',W',3,1]
        proj = torch.matmul(
            K_ds_stack[:, None, :, None, None],  # [V, 1, B, 1, 1, 3, 3]
            cam_pts,                             # [V, C, B, H', W', 3, 1]
        )                                        # [V, C, B, H', W', 3, 1]

        z_all = proj[..., 2, 0]                          # [V, C, B, H', W']
        u_all = proj[..., 0, 0] / (z_all + 1e-7)
        v_all = proj[..., 1, 0] / (z_all + 1e-7)
        x_all = u_all.round().long()
        y_all = v_all.round().long()
        valid = (z_all > 0) & (x_all >= 0) & (x_all < Wt_ds) & (y_all >= 0) & (y_all < Ht_ds)

        if not valid.any():
            log.info(
                f"Sparse3DCache.retrieve: no valid projections for any of {num_cands} candidates "
                f"(frame_ids={self._frame_ids[:avail]})",
                rank0_only=True,
            )
            return []

        # valid dims: [V, C, B, H', W'] → nonzero gives (view_ids, cand_ids, b_idx, _, _)
        view_ids, cand_ids, b_idx, _, _ = valid.nonzero(as_tuple=True)
        y_idx = y_all[valid]
        x_idx = x_all[valid]
        z_vals = z_all[valid].to(torch.float32)

        Btot = Bp
        pixels_per_view = Btot * Ht_ds * Wt_ds
        # Each (view, batch, y, x) gets a unique key so depth fusion is per-view.
        lin_keys = view_ids * pixels_per_view + b_idx * (Ht_ds * Wt_ds) + y_idx * Wt_ds + x_idx
        n_keys = num_views * pixels_per_view

        inf_val = torch.tensor(float('inf'), device=device, dtype=z_vals.dtype)
        min_depth = torch.full((n_keys,), inf_val, device=device, dtype=z_vals.dtype)
        min_depth.scatter_reduce_(0, lin_keys, z_vals, reduce='amin', include_self=True)

        min_d_for_pts = min_depth[lin_keys]
        if max_coverage:
            keep = z_vals <= (min_d_for_pts + float(depth_threshold))
            if not keep.any():
                return []

            lin_keys_keep = lin_keys[keep]
            cand_keep = cand_ids[keep].to(torch.long)

            flat_idx = cand_keep * n_keys + lin_keys_keep
            mask_flat = torch.zeros((num_cands * n_keys,), device=device, dtype=torch.bool)
            mask_flat.scatter_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.bool))
            mask = mask_flat.view(num_cands, n_keys)

            k = min(int(num_latents), num_cands)
            if k <= 0:
                return []

            # Pre-cover pixels from the temporally closest frame (largest frame_id)
            # because its warping is already included in the network condition.
            # Only applies when the most recent frame_id > 0 (skip the seed frame_id=0
            # and multiview inputs with negative IDs).
            avail_frame_ids = self._frame_ids[:avail]
            max_frame_id = max(avail_frame_ids)
            excluded: set[int] = set()
            if max_frame_id > 0:
                last_cand_idx = int(max(range(avail), key=lambda i: avail_frame_ids[i]))
                covered = mask[last_cand_idx].clone()
                excluded.add(last_cand_idx)
                log.info(
                    f"Sparse3DCache.retrieve(max_coverage): pre-covering pixels from temporally closest "
                    f"frame_id={avail_frame_ids[last_cand_idx]} (cand_idx={last_cand_idx}, "
                    f"pixels={int(covered.sum().item())})",
                    rank0_only=True,
                )
            else:
                covered = torch.zeros((n_keys,), device=device, dtype=torch.bool)

            selected: list[int] = []
            for _ in range(k):
                additional = (mask & (~covered)).sum(dim=1)
                exclude_indices = list(selected) + list(excluded)
                if len(exclude_indices) > 0:
                    additional[torch.tensor(exclude_indices, device=device)] = -1
                best = int(torch.argmax(additional).item())
                if additional[best].item() <= 0:
                    break
                selected.append(best)
                covered |= mask[best]

            if len(selected) == 0:
                return []
            top_ids = selected
        else:
            is_min = z_vals <= (min_d_for_pts + 1e-6)
            big_int = torch.iinfo(torch.long).max
            cid_masked = torch.where(is_min, cand_ids.to(torch.long), torch.full_like(cand_ids, big_int, dtype=torch.long))

            owner_lin = torch.full((n_keys,), -1, device=device, dtype=torch.long)
            owner_lin_tmp = torch.full((n_keys,), big_int, device=device, dtype=torch.long)
            owner_lin_tmp.scatter_reduce_(0, lin_keys, cid_masked, reduce='amin', include_self=True)
            owner_lin = torch.where(owner_lin_tmp == big_int, owner_lin, owner_lin_tmp)

            valid_owner = owner_lin[owner_lin >= 0]
            counts = torch.bincount(valid_owner, minlength=num_cands)

            scores_t = counts.float()
            scores = scores_t.tolist()

            score_map = {
                int(self._latent_indices[i]): {"score": float(scores[i]), "frame_id": int(self._frame_ids[i])}
                for i in range(num_cands)
            }
            log.info(f"Sparse3DCache.retrieve scores (latent_index -> score): {score_map}", rank0_only=True)

            if random and num_latents > 0:
                max_score = scores_t.max() if scores_t.numel() > 0 else scores_t.new_tensor(1.0)
                weights = torch.clamp(scores_t, min=0.0) + max_score * 0.02

                k = min(int(num_latents), scores_t.shape[0])
                if k <= 0:
                    return []
                sampled_ids = torch.multinomial(weights, num_samples=k, replacement=False)
                top_ids = [int(i) for i in sampled_ids.tolist()]

            else:
                top_ids = sorted(range(num_cands), key=lambda i: scores[i], reverse=True)[:num_latents]

        top_ids_reversed = top_ids[::-1]
        return [(self._latent_indices[i], self._frame_ids[i]) for i in top_ids_reversed]


