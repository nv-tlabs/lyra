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

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import einops
from typing import List, Dict
from copy import deepcopy
import os

from src.models.utils.model import timestep_embedding
from src.models.utils.data import ImageTransform, mirror_frame_indices, weighted_sample, merge_input_target_data_dicts
from src.models.data.registry import dataset_registry
from src.models.utils.cosmos_1_tokenizer import load_cosmos_latent_statistics, denormalize_latents

from src.models.data.datafield import DataField
from src.models.data.registry import dataset_registry

class Provider(Dataset):
    def __init__(self, dataset_name, opt, training=True, num_repeat=1):
        self.opt = opt

        # Get dataset setup
        dataset_kwargs = dataset_registry[dataset_name]['kwargs']
        dataset_kwargs = deepcopy(dataset_kwargs)

        # Set dataset parameters
        for key, default in dataset_registry['default'].items():
            setattr(self, key, dataset_kwargs.pop(key, default))
        self.start_view_idx = dataset_kwargs.get("start_view_idx", 0)

        # Create dataset
        self.dataset = dataset_registry[dataset_name]['cls'](**dataset_kwargs)
        self.scene_scale = dataset_registry[dataset_name]['scene_scale']
        self.max_gap, self.min_gap = dataset_registry[dataset_name]['max_gap'], dataset_registry[dataset_name]['min_gap']
        self.training = training
        self.dataset.sample_list *= num_repeat
        self.dataset.sample_list = sorted(self.dataset.sample_list)

        # Use part of training data for validation
        if self.opt.subsample_data_train_val:
            num_test_scenes = self.opt.get('num_test_scenes', self.opt.batch_size)
            num_train_images = self.opt.get('num_train_images', -num_test_scenes)
            unique_sample_list = list({os.path.basename(f) for f in self.dataset.sample_list})
            num_unique_samples = len(unique_sample_list)
            if self.training:
                self.dataset.sample_list = self.dataset.sample_list[:min(num_train_images, num_unique_samples)]
            else:
                self.dataset.sample_list = self.dataset.sample_list[-min(num_test_scenes, num_unique_samples):]
        
        # Image transformations (crop and resize)
        self._setup_image_transforms(
            sample_size=self.opt.img_size,
            crop_size=self.opt.img_size,
            use_flip=False,
            max_crop=True,
        )

        # Data fields
        self.load_latents = self.opt.load_latents and self.has_latents
        self.data_fields = [DataField.IMAGE_RGB.value, DataField.CAMERA_C2W_TRANSFORM.value, DataField.CAMERA_INTRINSICS.value]
        if self.load_latents:
            self.data_fields_latents = [DataField.LATENT_RGB.value, DataField.CAMERA_C2W_TRANSFORM.value, DataField.CAMERA_INTRINSICS.value]
        if self.opt.use_depth:
            self.data_fields.append(DataField.METRIC_DEPTH.value)
        # Load latent statistics to denormalize generated latents
        if self.is_generated_cosmos_latent:
            self.latent_mean, self.latent_std = load_cosmos_latent_statistics(self.opt.vae_path)
        else:
            self.latent_mean, self.latent_std = None, None
        
        # Number of target views
        self.num_target_views = self.opt.num_views - self.opt.num_input_views

    def __len__(self):
        return len(self.dataset)

    def _setup_image_transforms(self, sample_size, crop_size, use_flip, max_crop=False):
        self.image_transform = ImageTransform(crop_size=crop_size, sample_size=sample_size, use_flip=use_flip, max_crop=max_crop)
        self.input_normalizer_vae = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)

    def _preprocess(self, file_name, rgbs, c2ws, intrinsics, depths, timesteps, latents=None, target_index=None, num_input_multi_views=None):
        # Filter: keep depths > 0 and finite, else set to 0
        if depths is not None:
            valid_mask = (depths > 0) & torch.isfinite(depths)
            depths = torch.where(valid_mask, depths, torch.zeros_like(depths))
        
        # Crop and resize
        rgbs, depths, shift, scale, flip_flag = self.image_transform.preprocess_images(rgbs, depths)
        intrinsics = torch.stack([intrinsics[...,0]*scale[0], intrinsics[...,1]*scale[1], (intrinsics[...,2]+shift[0])*scale[0], (intrinsics[...,3]+shift[1])*scale[1]], dim=-1)
        
        # Relative pose
        if self.is_w2c:
            c2ws = c2ws.inverse()
        if self.opt.relative_translation_scale:
            c2w_ref = c2ws[0]
            c2w_rel = torch.inverse(c2w_ref)[None]
            c2ws_new = c2w_rel @ c2ws
            target_cam_c2w = c2ws_new[[0]]
            c2ws = c2ws_new

        # Scaling the scene
        c2ws = self._norm_camera_scale(c2ws)
        
        # Split rgb into input and target
        num_total_input_frames = num_input_multi_views * self.opt.num_input_views
        if self.load_latents:
            images_input_vae = None
            images_output = rgbs
            rgb_latents = latents
        else:
            images_input_vae = self.input_normalizer_vae(rgbs[:num_total_input_frames])
            images_output = rgbs[num_total_input_frames:]
            rgb_latents = None
        
        # Time embedding
        if not self.training:
            timesteps = timesteps[:self.opt.num_input_views + 1]
        if self.opt.time_embedding:
            timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min() + self.opt.timesteps_eps) # normalize to 0 to 1 # [TV]
            time_embeddings = timestep_embedding(timesteps, self.opt.time_embedding_dim, use_orig=self.opt.time_embedding_use_orig)    # [TV, D]
        else:
            time_embeddings = False

        # Split cameras into input and target
        intrinsics_input = intrinsics[:num_total_input_frames]
        c2ws_input = c2ws[:num_total_input_frames]
        cam_view = torch.inverse(c2ws[num_total_input_frames:]).transpose(1, 2) # [V, 4, 4]
        intrinsics = intrinsics[num_total_input_frames:]

        # Split time_embeddings into source and target for vae encoder (assume last one is target index)
        if self.opt.time_embedding_vae:
            time_embeddings_target = time_embeddings[[-1]]
            time_embeddings = time_embeddings[:self.opt.num_input_views]
        else:
            time_embeddings_target = [False]
    
        # Prepare final output
        out_dict = {
            'images_output': images_output,
            'intrinsics': intrinsics,
            'cam_view': cam_view,
            'time_embeddings': time_embeddings,
            'time_embeddings_target': time_embeddings_target,
            'num_input_multi_views': num_input_multi_views,
            'intrinsics_input': intrinsics_input,
            'c2ws_input': c2ws_input,
            'flip_flag': flip_flag,
            'file_name': file_name,
            'target_index': target_index,
        }

        # Additional outputs
        if self.opt.use_depth:
            if not self.load_latents:
                depths_output = depths[num_total_input_frames:]
            else:
                depths_output = depths
            out_dict['depths_output'] = depths_output
        if images_input_vae is not None:
            out_dict['images_input_vae'] = images_input_vae
        if rgb_latents is not None:
            if self.is_generated_cosmos_latent:
                rgb_latents = denormalize_latents(rgb_latents, self.latent_std, self.latent_mean, num_input_multi_views)
            out_dict['rgb_latents'] = rgb_latents
        if not self.opt.compute_plucker_cuda:
            out_dict['plucker_embedding'] = plucker_embedding
            out_dict['rays_os'] = rays_os
            out_dict['rays_ds'] = rays_ds
        return out_dict
    
    def _norm_camera_scale(self, c2ws: torch.Tensor):
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * self.scene_scale
        return c2ws
    
    def _get_view_indices(self, num_views: int, camera_count: int, start_view_idx: int = 0):
        if num_views > camera_count:
            view_indices = np.random.permutation(np.arange(num_views)%camera_count)
        else:
            view_indices = np.random.permutation(np.arange(camera_count))[:num_views]
        view_indices = view_indices + start_view_idx
        return view_indices
    
    def _get_view_indices_from_input(self, view_indices_input: List[int]):
        num_input_views = len(view_indices_input)
        if self.num_target_views > num_input_views:
            view_indices_i = np.random.permutation(np.arange(self.num_target_views)%num_input_views)
        else:
            view_indices_i = np.random.permutation(np.arange(num_input_views))[:self.num_target_views]
        view_indices_target = view_indices_input[view_indices_i]
        return view_indices_target
    
    def _get_num_input_multi_views(self):
        if self.opt.sample_num_input_multi_views and self.training:
            num_input_multi_views = np.random.randint(low=1, high=self.opt.num_input_multi_views + 1)
        else:
            num_input_multi_views = self.opt.num_input_multi_views
        return num_input_multi_views

    def _get_indices_dynamic(self, idx: int):
        total_num_frames = self.dataset.count_frames(idx)
        camera_count = self.get_camera_count(idx)
        num_input_multi_views = self._get_num_input_multi_views()
        num_input_multi_views = min(num_input_multi_views, camera_count)

        # If there are not enough frames, try to mirror video
        if total_num_frames <= self.opt.num_input_views:
            if self.opt.mirror_dynamic:
                # Start with 0 always if video is too short
                start_idx = 0
                frame_indices = mirror_frame_indices(self.opt.num_input_views, total_num_frames, start_index=start_idx)
                frame_indices = np.array(frame_indices)
                target_indices = frame_indices
            else:
                assert total_num_frames >= self.opt.num_input_views, f'Frame number {total_num_frames} is smaller than number of input views {self.opt.num_input_views}.'
        else:
            context_gap = np.random.randint(self.min_gap, self.max_gap + 1)
            context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views)
            start_frame = np.random.randint(0, total_num_frames-context_gap)
            if not self.training:
                start_frame = 0
            end_frame = start_frame + context_gap
            inbetween_indices = np.sort(np.random.permutation(np.arange(start_frame + 1, end_frame))[:self.opt.num_input_views - 2])
            frame_indices = np.array([start_frame, *inbetween_indices, end_frame])
            target_indices = np.arange(start_frame, end_frame+1)
        target_index = np.random.permutation(target_indices)[:1]
        if not self.opt.use_interp_target:
            target_index = np.random.permutation(frame_indices)[:1]
        
        # Just take num_input_multi_views (dynamic) cameras when there is a monocular video
        view_indices_input = np.random.permutation(np.arange(camera_count))[:num_input_multi_views]
        if self.opt.select_target_views_input_dynamic:
            view_indices_target = self._get_view_indices_from_input(view_indices_input)
        else:
            if self.end_view_target_idx is not None and self.start_view_target_idx is not None:
                num_additional_target_views = self.end_view_target_idx - self.start_view_target_idx + 1
            else:
                num_additional_target_views = 0
            view_indices_target = self._get_view_indices(self.num_target_views, camera_count + num_additional_target_views)
        if not self.training:
            view_indices_input = np.sort(view_indices_input)
            view_indices_target = np.sort(view_indices_target)
        view_indices = np.concatenate((view_indices_input, view_indices_target))
        
        # Set manual target index
        self._set_target_index_manually(target_index, frame_indices)

        # Set indices manually at inference time
        if not self.training:
            if self.opt.static_view_indices_sampling == 'fixed':
                view_indices = np.array(self.opt.static_view_indices_fixed)
            if self.opt.set_manual_time_idx:
                target_index = frame_indices
            # Subsampled output views
            if self.opt.target_index_subsample:
                target_index = target_index[::self.opt.target_index_subsample]
            view_indices = np.concatenate((view_indices, view_indices))
        
        # Append target index to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])

        return frame_indices, view_indices, num_input_multi_views
    
    def _get_indices_static_multi_view(self, start_frame: int = None, end_frame: int = None, total_num_frames: int = None, mirror_frames: bool = False, num_input_multi_views: int = None):
        frame_indices_views = []
        # Create multiple frame indices for each view index, can create duplicates across views for overlap
        for view_idx in range(num_input_multi_views):
            # Create mirrored frame indices for too short videos
            if mirror_frames:
                frame_indices = mirror_frame_indices(self.opt.num_input_views, total_num_frames)
            else:
                frame_indices = self.get_random_static_indices(start_frame, end_frame)
            frame_indices_views.append(frame_indices)
        frame_indices_views = np.concatenate(frame_indices_views)
        return frame_indices_views
    
    def get_random_static_indices(self, start_frame: int, end_frame: int):
        frame_indices_range = np.arange(start_frame + 1, end_frame)
        inbetween_indices = np.sort(np.random.permutation(frame_indices_range)[:self.opt.num_input_views - 2])
        frame_indices = np.array([start_frame, *inbetween_indices, end_frame])
        return frame_indices
    
    def get_camera_count(self, idx: int):
        if self.end_view_idx is not None:
            camera_count = self.end_view_idx + 1 - self.start_view_idx
        else:
            camera_count = self.dataset.count_cameras(idx)
        return camera_count

    def _get_indices_static(self, idx: int):
        total_num_frames = self.dataset.count_frames(idx)
        camera_count = self.get_camera_count(idx)
        num_input_multi_views = self._get_num_input_multi_views()
        num_input_multi_views = min(num_input_multi_views, camera_count)

        # Check if there are enough frames for the model input
        if total_num_frames <= self.opt.num_input_views:
            if total_num_frames == self.opt.num_input_views:
                mirror_frames, start_frame, end_frame = False, 0, total_num_frames - 1
            else:
                mirror_frames, start_frame, end_frame = True, None, None
            if self.opt.mirror_static:
                frame_indices = self._get_indices_static_multi_view(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    total_num_frames=total_num_frames,
                    mirror_frames=mirror_frames,
                    num_input_multi_views=num_input_multi_views,
                    )
                target_indices = frame_indices[:self.opt.num_input_views]
            else:
                assert total_num_frames >= self.opt.num_input_views, f'Frame number {total_num_frames} is smaller than number of input views {self.opt.num_input_views}.'
        else:
            # Sample frame range
            context_gap = np.random.randint(self.min_gap, self.max_gap + 1)
            context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views)

            # Sample input frame indices
            start_frame = np.random.randint(0, total_num_frames-context_gap)
            end_frame = start_frame + context_gap
            frame_indices = self._get_indices_static_multi_view(start_frame, start_frame + context_gap, num_input_multi_views=num_input_multi_views)
            target_indices = np.arange(start_frame, end_frame+1)
        target_index, _ = weighted_sample(target_indices, self.num_target_views, self.opt.static_frame_sampling)
            
        # Set manually the frame indices at inference time
        if not self.training:
            frame_indices = einops.repeat(np.arange(0, self.opt.num_input_views), 't -> (v t)', v=num_input_multi_views)

            # Normal output views
            target_index = frame_indices.copy()
            target_index = target_index[:self.opt.num_input_views]

            # Subsampled output views
            if self.opt.target_index_subsample:
                target_index = target_index[::self.opt.target_index_subsample]
        
            # Set manual target index
            if not self.opt.set_manual_time_idx:
                self._set_target_index_manually(target_index, frame_indices)
        
        # Append target index to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])

        # Sample view indices (default static dataset only has one view)
        view_indices = self._sample_view_indices_bucket(camera_count, num_input_multi_views)

        return frame_indices, view_indices, num_input_multi_views

    def _sample_view_indices_bucket(self, camera_count: int, num_input_multi_views: int):
        # Sample view indices (default static dataset only has one view)
        if self.opt.static_view_indices_sampling == 'empty':
            view_indices = []
        elif self.opt.static_view_indices_sampling == 'random':
            view_indices = self._get_view_indices(num_input_multi_views, camera_count, self.start_view_idx).tolist()
            view_indices = [str(view_idx) for view_idx in view_indices]
        elif self.opt.static_view_indices_sampling == 'random_bucket':
            view_indices = self._sample_view_indices_from_bucket(camera_count, num_input_multi_views)
        elif self.opt.static_view_indices_sampling == 'fixed':
            view_indices = self.opt.static_view_indices_fixed
        return view_indices
    
    def _sample_view_indices_from_bucket(self, camera_count: int, num_input_multi_views: int):
        assert sum([len(bucket) for bucket in self.sampling_buckets]) == camera_count, f"{sum([len(bucket) for bucket in self.sampling_buckets])} vs. {camera_count}"
        # Get bucket indices
        view_indices_buckets = self._get_view_indices(num_input_multi_views, len(self.sampling_buckets), self.start_view_idx).tolist()
        # Get view from bucket
        view_indices = [np.random.permutation(self.sampling_buckets[view_indices_bucket - self.start_view_idx])[:1].item() for view_indices_bucket in view_indices_buckets]
        return view_indices
    
    def _set_target_index_manually(self, target_index: np.ndarray, frame_indices: np.ndarray):
        if not self.training and self.opt.target_index_manual is not None:
            target_index[:] = self.opt.target_index_manual
    
    def _read_latent_data_static(self, idx: int, frame_indices_input: np.ndarray, frame_indices_target: np.ndarray, view_indices: np.ndarray, num_input_multi_views: int, data_fields_input: DataField, data_fields_target: DataField):
        original_output_dict_input = None

        # Read data for each view index
        for multi_view_idx in range(num_input_multi_views):
            frame_indices_input_view = frame_indices_input[multi_view_idx * self.opt.num_input_views: (multi_view_idx + 1) * self.opt.num_input_views]
            view_indices_view = [view_indices[multi_view_idx]]
            original_output_dict_input_view = self.dataset.get_data(idx, data_fields=data_fields_input, frame_indices=frame_indices_input_view, view_indices=view_indices_view)
            if original_output_dict_input is None:
                original_output_dict_input = original_output_dict_input_view
            else:
                for output_dict_k_view, output_dict_v_view in original_output_dict_input.items():
                    if output_dict_k_view == "__key__":
                        continue
                    original_output_dict_input[output_dict_k_view] = torch.cat((output_dict_v_view, original_output_dict_input_view[output_dict_k_view]))           
        original_output_dict_target = self.dataset.get_data(idx, data_fields=data_fields_target, frame_indices=frame_indices_target, view_indices=view_indices)
        return original_output_dict_input, original_output_dict_target
    
    def get_data_dynamic(self, idx: int, frame_indices: List[int], view_indices: List[int], num_input_multi_views: int):
        # Split frame and view indices into input and target
        frame_indices_input = frame_indices[:self.opt.num_input_views]
        frame_indices_target = frame_indices[self.opt.num_input_views:]
        view_indices_input = view_indices[:num_input_multi_views]
        view_indices_target = view_indices[num_input_multi_views:]

        # Split data fields for input if latents are used
        data_fields_target = self.data_fields
        if self.load_latents:
            data_fields_input = self.data_fields_latents
        else:
            data_fields_input = self.data_fields
        
        # Read data
        original_output_dict_input = self.dataset.get_data(idx, data_fields=data_fields_input, frame_indices=frame_indices_input, view_indices=view_indices_input)
        original_output_dict_target = self.dataset.get_data(idx, data_fields=data_fields_target, frame_indices=frame_indices_target, view_indices=view_indices_target)

        # Merge input and target
        original_output_dict = merge_input_target_data_dicts(data_fields_input, data_fields_target, original_output_dict_input, original_output_dict_target)
        return original_output_dict
    
    def get_data_static(self, idx: int, frame_indices: np.ndarray, view_indices: np.ndarray, num_input_multi_views: int):
        num_total_input_frames = num_input_multi_views * self.opt.num_input_views
        
        # Directly load input latents
        if self.load_latents:
            # Split into input and target frame indices
            frame_indices_input = frame_indices[:num_total_input_frames]
            frame_indices_target = frame_indices[num_total_input_frames:]
            data_fields_target = self.data_fields
            data_fields_input = self.data_fields_latents
            original_output_dict_input, original_output_dict_target = self._read_latent_data_static(idx, frame_indices_input, frame_indices_target, view_indices, num_input_multi_views, data_fields_input, data_fields_target)
            
            # Merge input and target
            original_output_dict = merge_input_target_data_dicts(data_fields_input, data_fields_target, original_output_dict_input, original_output_dict_target)
        else:
            # For multi-view training, save repeated read operations
            if num_input_multi_views > 1:
                assert len(view_indices) == 0, f'Assuming that all frames come from the same view index, but {len(view_indices)} view indices found'
                frame_indices_unique, frame_indices_unique_rev = np.unique(frame_indices, return_inverse=True)
            else:
                frame_indices_unique = frame_indices
            original_output_dict_unique = self.dataset.get_data(idx, data_fields=self.data_fields, frame_indices=frame_indices_unique, view_indices=view_indices)
            
            # Sample from unique reads
            if num_input_multi_views > 1:
                for data_field in self.data_fields:
                    original_output_dict_unique[data_field] = original_output_dict_unique[data_field][frame_indices_unique_rev]
            original_output_dict = original_output_dict_unique
        return original_output_dict

    def get_item(self, idx):
        # File name for inference
        file_name = self.dataset.mp4_file_paths[idx].stem

        # Sample view and frame indices
        _get_indices_fn = self._get_indices_static if self.dataset.is_static else self._get_indices_dynamic
        frame_indices, view_indices, num_input_multi_views = _get_indices_fn(idx)

        # Read data depending on static or dynamic data
        if self.dataset.is_static:
            original_output_dict = self.get_data_static(idx, frame_indices, view_indices, num_input_multi_views)
        else:
            original_output_dict = self.get_data_dynamic(idx, frame_indices, view_indices, num_input_multi_views)
        
        # Get rgb and camera matrices
        rgbs, c2ws, intrinsics = original_output_dict[DataField.IMAGE_RGB.value], original_output_dict[DataField.CAMERA_C2W_TRANSFORM.value], original_output_dict[DataField.CAMERA_INTRINSICS.value]
        
        # Optionally get depth if available
        depths = original_output_dict.get(DataField.METRIC_DEPTH.value, None)
        if depths is not None:
            depths = depths[:, None]
        
        # Optionally get rgb latents as input to the model
        latents = original_output_dict.get(DataField.LATENT_RGB.value, None)

        # Set manual target time index
        if not self.training and self.opt.set_manual_time_idx:
            self._set_target_index_manually(frame_indices[self.opt.num_input_views:], frame_indices[:self.opt.num_input_views])
        
        # Convert frame indices to timesteps for the model input
        target_index = torch.from_numpy(frame_indices[self.opt.num_input_views:])
        timesteps = torch.from_numpy(frame_indices).float()

        # Export final output dict
        return self._preprocess(file_name, rgbs, c2ws, intrinsics, depths, timesteps, latents, target_index, num_input_multi_views)

    def __getitem__(self, idx):
        count = 0
        while True:
            try:
                results = self.get_item(idx)
                break
            except Exception as e:
                count += 1
                if count > 20:
                    print(f"data loader error count {count}: {e}")
                idx = np.random.randint(0, len(self.dataset))
        return results