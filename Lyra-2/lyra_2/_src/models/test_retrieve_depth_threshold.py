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

import ast
import pathlib
from typing import Optional

import torch

from lyra_2._src.datasets.forward_warp_utils_pytorch import unproject_points


class _Log:
    def info(self, *args, **kwargs):
        pass


def _load_sparse3d_cache():
    # Importing lyra2_model needs the whole training stack (megatron, hydra, ...), which the cache
    # does not use; build just the class from source so this test needs only torch and einops.
    src = (pathlib.Path(__file__).parent / "lyra2_model.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Sparse3DCache")
    ns = dict(torch=torch, Optional=Optional, log=_Log(), unproject_points=unproject_points)
    exec("from __future__ import annotations\n" + ast.get_source_segment(src, node), ns)
    return ns["Sparse3DCache"]


Sparse3DCache = _load_sparse3d_cache()

H, W = 48, 64
K = torch.tensor([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])[None]
SCALES = (0.01, 0.1, 1.0, 10.0, 100.0)


def _wall(z, cols):
    """Fronto-parallel wall at depth z; columns outside `cols` are invalid (depth 0)."""
    d = torch.full((H, W), float(z))
    d[:, ~cols(torch.arange(W))] = 0.0
    return d


def _select(walls, **kw):
    cache = Sparse3DCache(downsample=1, store_device="cpu")
    for i, d in enumerate(walls):
        cache.add(d[None, None], torch.eye(4)[None], K, latent_index=i, frame_id=-1)
    out = cache.retrieve(torch.eye(4)[None], K, (H, W), 3, max_coverage=True, **kw)
    return sorted(i for i, _ in out)


def _tilted_wall_scene(s):
    """The same scene dilated by s: A near/left, B the same wall 8% deeper (full width), C near/right."""
    return [
        _wall(1.00 * s, lambda u: u < 32),
        _wall(1.08 * s, lambda u: u >= 0),
        _wall(1.00 * s, lambda u: u >= 48),
    ]


def test_default_rule_is_unchanged():
    # Absolute 0.1 tolerance, as before: the decision flips with scale (see the next test).
    assert _select(_tilted_wall_scene(1.0)) == [1]
    assert _select(_tilted_wall_scene(2.0)) == [0, 1, 2]
    assert _select(_tilted_wall_scene(4.0)) == [0, 1, 2]
    assert _select(_tilted_wall_scene(4.0), depth_threshold=0.5) == [1]


def test_zero_relative_threshold_equals_default():
    for s in (1.0, 2.0, 4.0):
        assert _select(_tilted_wall_scene(s), depth_threshold_rel=0.0) == _select(_tilted_wall_scene(s))


def test_relative_threshold_is_scale_equivariant():
    for s in SCALES:
        assert _select(_tilted_wall_scene(s), depth_threshold=0.0, depth_threshold_rel=0.1) == [1], s


def test_relative_threshold_still_rejects_occluded_surface():
    for s in SCALES:
        walls = [_wall(1.0 * s, lambda u: u < 32), _wall(1.5 * s, lambda u: u >= 0)]
        assert _select(walls, depth_threshold=0.0, depth_threshold_rel=0.1) == [0, 1], s
