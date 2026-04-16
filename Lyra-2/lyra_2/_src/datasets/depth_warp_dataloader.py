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

import importlib
import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from lyra_2._src.datasets.base import DataField
except ImportError:
    pass
from lyra_2._ext.imaginaire.utils import log
from lyra_2._ext.imaginaire.lazy_config import LazyCall as L, instantiate
from lyra_2._src.datasets.config_dataverse import DATAVERSE_CONFIG
from lyra_2._src.datasets.utils import VIDEO_RES_SIZE_INFO

try:
    from megatron.core import parallel_state
except ImportError:
    pass

import omegaconf
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def _instantiate_from_config(config, **additional_kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    additional_kwargs.update(config.get("params", dict()))
    return _get_obj_from_str(config["target"])(**additional_kwargs)


def _resize_intrinsics(intrinsics, old_size, new_size, crop_size=None):
    """intrinsics: (N, 3, 3), sizes: (h, w)"""
    intrinsics_copy = intrinsics.clone() if isinstance(intrinsics, torch.Tensor) else np.copy(intrinsics)
    intrinsics_copy[:, 0, :] *= new_size[1] / old_size[1]
    intrinsics_copy[:, 1, :] *= new_size[0] / old_size[0]
    if crop_size is not None:
        intrinsics_copy[:, 0, -1] -= (new_size[1] - crop_size[1]) / 2
        intrinsics_copy[:, 1, -1] -= (new_size[0] - crop_size[0]) / 2
    return intrinsics_copy


def _intrinsics_from_fxfycxcy_batch(intrinsics):
    m = torch.zeros((intrinsics.shape[0], 3, 3), device=intrinsics.device)
    m[:, 0, 0] = intrinsics[:, 0]
    m[:, 1, 1] = intrinsics[:, 1]
    m[:, 0, 2] = intrinsics[:, 2]
    m[:, 1, 2] = intrinsics[:, 3]
    m[:, 2, 2] = 1
    return m


def _dict_collation_fn(samples):
    """Collate a list of sample dicts into a batched dict."""
    batched = {key: [s[key] for s in samples] for key in samples[0]}
    result = {}
    for key, vals in batched.items():
        if isinstance(vals[0], bool):
            result[key] = vals[0]
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.from_numpy(np.array(vals))
        elif isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


def _sample_frame_indices(total_frames, N, video_mirror=False):
    """Sample N frame indices starting from 0 with stride 1.
    If video_mirror is True, extends short clips by mirroring before sampling.
    """
    if video_mirror:
        mapping = list(range(total_frames))
        n_repeat = max((N - total_frames) // (total_frames - 1), 0) + 1
        mapping_repeat = mapping.copy()
        for i in range(n_repeat):
            mapping_repeat += mapping[-2::-1] if i % 2 == 0 else mapping[1:]
        return [mapping_repeat[i] for i in range(N)]
    else:
        if total_frames < N:
            idx = list(range(total_frames))
            idx += [total_frames - 1] * (N - total_frames)
            return idx
        return list(range(N))


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

class IterativeGEN3CDataLoader:
    """Wraps multiple dataloaders with ratio-based sampling."""

    def __init__(self, dataloaders):
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []
        for dataset_name, dataloader_data in dataloaders.items():
            if dataset_name in ("image_data", "video_data"):
                continue
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"]))
            self.data_ratios.append(dataloader_data["ratio"])
        self.ratio_sum = sum(self.data_ratios)
        self.data_len = sum(len(d) for d in self.dataloader_list)
        self.dataloaders = [iter(dl) for dl in self.dataloader_list]

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        while True:
            data_id = random.randint(0, self.ratio_sum - 1)
            cumsum = 0
            for i, r in enumerate(self.data_ratios):
                cumsum += r
                if data_id < cumsum:
                    break
            output = next(self.dataloaders[i])
            output["dataset_name"] = self.dataset_name_list[i]
            yield output


def get_gen3c_multiple_video_dataloader(
    dataset_list: list[str],
    dataset_weight_list: list[float],
    shuffle=True,
    num_workers=4,
    prefetch_factor=4,
    mode="random",
) -> omegaconf.dictconfig.DictConfig:
    dataloader_dict = {
        name: {
            "dataloader": L(MyDataLoader)(
                dataset=L(get_depth_warp_dataset)(dataset_name=name),
                batch_size=1,
                num_workers=num_workers,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
            ),
            "ratio": weight,
        }
        for name, weight in zip(dataset_list, dataset_weight_list)
    }
    return L(IterativeGEN3CDataLoader)(dataloaders=dataloader_dict)


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        kw.pop("dataloaders", None)
        super().__init__(dataset.build_dataset(), batch_size, collate_fn=_dict_collation_fn, *args, **kw)


def get_depth_warp_dataset(dataset_name="dl3dv_long_moge_chunk_81_480p_dav3_hsg", resolution="720", chunk_size=256, **kwargs):
    return DepthWarpDataset(dataset_name, resolution, chunk_size, **kwargs)


class DepthWarpDataset:
    def __init__(self, dataset_name, resolution, chunk_size, **kwargs):
        self.video_size = VIDEO_RES_SIZE_INFO[resolution]
        self.dataset_config = OmegaConf.merge(
            OmegaConf.create(DATAVERSE_CONFIG[dataset_name]),
            OmegaConf.create(kwargs),
        )

    def build_dataset(self):
        return InfiniteCommonDataset(**OmegaConf.to_container(self.dataset_config, resolve=True))


class InfiniteCommonDataset:
    def __init__(
        self,
        dataset_cfg,
        data_name="",
        batch_size=1,
        sample_n_frames=8,
        sample_size=[320, 512],
        crop_size=None,
        video_mirror=False,
        video_mirror_when_short_only=False,
        t5_embedding_path=None,
    ):
        self.dataset = _instantiate_from_config(dataset_cfg)
        self.data_name = data_name
        self.n_data = self.dataset.num_videos()
        self.t5_embedding_path = t5_embedding_path

        if parallel_state.is_initialized():
            dp_group_id = parallel_state.get_data_parallel_rank()
            dp_world_size = parallel_state.get_data_parallel_world_size()
            log.critical(
                f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, "
                + f"TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, "
                + f"DP: {dp_group_id}, DP World size: {dp_world_size}"
            )
        else:
            dp_world_size = 1
            dp_group_id = 0
        self.n_data_per_node = self.n_data // dp_world_size
        self.data_start_idx = dp_group_id * self.n_data_per_node

        self.multiplier = (2000000 * batch_size) // self.n_data_per_node
        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size
        self.crop_size = crop_size if crop_size is not None else sample_size
        self.video_mirror = video_mirror
        self.video_mirror_when_short_only = video_mirror_when_short_only

        self.img_transform = transforms.Compose([
            transforms.Resize(sample_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(self.crop_size),
        ])
        self.norm_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def __len__(self):
        return self.multiplier * self.n_data_per_node

    def _get_frame_indices(self, n_total_frames):
        if self.video_mirror_when_short_only:
            use_mirror = self.video_mirror and (n_total_frames < self.sample_n_frames)
        else:
            use_mirror = self.video_mirror
        return _sample_frame_indices(n_total_frames, self.sample_n_frames, use_mirror)

    def _transform(self, images, w2c, depths, intrinsics):
        _, _, H, W = images.shape
        return {
            "video": self.norm_image(self.img_transform(images)).permute(1, 0, 2, 3).contiguous(),
            "camera_w2c": w2c,
            "depth": self.img_transform(depths),
            "intrinsics": _resize_intrinsics(intrinsics, [H, W], self.sample_size, self.crop_size),
            "is_preprocessed": True,
        }

    def _to_w2c_and_intrinsics(self, data):
        data[DataField.METRIC_DEPTH] = data[DataField.METRIC_DEPTH].unsqueeze(1)
        data[DataField.CAMERA_C2W_TRANSFORM] = torch.from_numpy(
            np.stack([np.linalg.inv(m.numpy()) for m in data[DataField.CAMERA_C2W_TRANSFORM]])
        )
        data[DataField.CAMERA_INTRINSICS] = _intrinsics_from_fxfycxcy_batch(data[DataField.CAMERA_INTRINSICS])
        return data

    def __getitem__(self, idx):
        data_idx = (idx % self.n_data_per_node) + self.data_start_idx
        assert data_idx < self.n_data

        frame_indices = self._get_frame_indices(self.dataset.num_frames(data_idx))

        data = self.dataset._read_data(
            video_idx=data_idx,
            data_fields=[DataField.IMAGE_RGB, DataField.CAMERA_C2W_TRANSFORM, DataField.CAMERA_INTRINSICS],
            frame_idxs=frame_indices,
            view_idxs=[0],
        )
        depth_data = self.dataset._read_data(
            video_idx=data_idx,
            data_fields=[DataField.METRIC_DEPTH],
            frame_idxs=frame_indices,
            view_idxs=[0],
        )
        data.update(depth_data)
        data = self._to_w2c_and_intrinsics(data)

        N = self.sample_n_frames
        sample = self._transform(
            data[DataField.IMAGE_RGB][-N:].clone(),
            data[DataField.CAMERA_C2W_TRANSFORM][-N:].clone(),
            data[DataField.METRIC_DEPTH][-N:].clone(),
            data[DataField.CAMERA_INTRINSICS][-N:].clone(),
        )

        # T5 chunk embeddings
        t5_path = os.path.join(self.t5_embedding_path, data["__key__"] + ".pkl")
        if not os.path.exists(t5_path):
            print(f"t5 embedding path {t5_path} does not exist")
            return self.__getitem__(np.random.randint(0, len(self)))
        t5_pickle = pickle.load(open(t5_path, "rb"))
        keys = [int(k) for k in t5_pickle]
        embeddings = [torch.as_tensor(t5_pickle[k]["embedding"]) for k in t5_pickle]
        order = np.argsort(keys)
        sorted_keys = np.asarray(keys, dtype=np.int64)[order]
        sorted_embs = [embeddings[i] for i in order]
        cutoff = int(np.searchsorted(sorted_keys, int(frame_indices[-1]), side="right"))
        sorted_keys = sorted_keys[:cutoff]
        sorted_embs = sorted_embs[:cutoff]
        assert len(sorted_embs) > 0
        chunk_emb = torch.zeros(len(sorted_embs), 512, 4096)
        chunk_mask = torch.zeros(len(sorted_embs), 512)
        for i, e in enumerate(sorted_embs):
            s, d = min(e.shape[0], 512), min(e.shape[1], 4096)
            chunk_emb[i, :s, :d] = e[:s, :d]
            chunk_mask[i, :s] = 1.0
        del t5_pickle
        sample["t5_chunk_keys"] = torch.from_numpy(sorted_keys)
        sample["t5_chunk_embeddings"] = chunk_emb
        sample["t5_chunk_mask"] = chunk_mask

        sample["sample_frame_indices"] = torch.as_tensor(frame_indices, dtype=torch.long)
        sample["num_frames"] = N
        sample["image_size"] = torch.as_tensor(self.crop_size)
        sample["fps"] = 24
        sample["__key__"] = data["__key__"]
        sample["clip_name"] = f"{self.data_name}-{data['__key__']}-{data_idx:d}-000-001"
        sample["padding_mask"] = torch.zeros(1, self.crop_size[0], self.crop_size[1])
        del data
        return sample
