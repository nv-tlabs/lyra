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

"""Benchmark peak memory and wall time of ``Sparse3DCache.retrieve``.

Compares the original single-shot projection (``chunk_size=None``) with the
chunked projection for several cache sizes, view counts and retrieval modes.

Each measurement runs in a fresh subprocess so allocator reuse from a previous
run cannot pollute the peak-RSS number. When CUDA is available the script also
reports ``torch.cuda.max_memory_allocated``; otherwise it reports the measured
CPU RSS delta and an analytic transient-tensor size.

Examples::

    python Lyra-2/scripts/benchmark_sparse3dcache_retrieve.py --quick
    python Lyra-2/scripts/benchmark_sparse3dcache_retrieve.py --full --json-out metrics.json
    python Lyra-2/scripts/benchmark_sparse3dcache_retrieve.py --gpu-parity
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import pathlib
import statistics
import subprocess
import sys
import threading
import time

import torch

try:
    import psutil
except Exception:  # pragma: no cover - psutil is optional
    psutil = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "lyra_2" / "_src" / "models" / "lyra2_model.py"


class _StubLog:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def load_sparse3dcache():
    """Import the real class, or exec only it from source in a minimal env."""
    try:
        from lyra_2._src.models.lyra2_model import Sparse3DCache  # type: ignore

        return Sparse3DCache
    except Exception:
        pass

    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sparse3DCache"
    )
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)

    def _unavailable_unproject(*args, **kwargs):
        raise RuntimeError("unproject_points is unavailable in the minimal benchmark loader")

    namespace = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "unproject_points": _unavailable_unproject,
        "log": _StubLog(),
    }
    exec(compile(module, str(MODEL_PATH), "exec"), namespace)
    return namespace["Sparse3DCache"]


Sparse3DCache = load_sparse3dcache()


def build_cache(num_candidates: int, batch: int, hp: int, wp: int, seed: int = 0):
    cache = Sparse3DCache(downsample=4, store_device="cpu", store_values=False)
    generator = torch.Generator().manual_seed(seed)
    for index in range(num_candidates):
        points = torch.empty(batch, hp, wp, 3)
        points[..., 0].uniform_(-1.5, 1.5, generator=generator)
        points[..., 1].uniform_(-1.5, 1.5, generator=generator)
        points[..., 2].uniform_(0.5, 3.0, generator=generator)
        cache._world_points.append(points)
        cache._latent_indices.append(index)
        cache._frame_ids.append(index)
    return cache


def build_targets(batch: int, views: int, hw: tuple[int, int]):
    ht, wt = hw
    poses = []
    intrinsics = []
    for view in range(views):
        angle = 0.05 * view
        rotation = torch.tensor(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        pose = torch.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = torch.tensor([0.2 * view, 0.1 * view, 3.0])
        poses.append(pose)
        intrinsics.append(
            torch.tensor(
                [
                    [0.9 * wt, 0.0, wt / 2.0],
                    [0.0, 0.9 * ht, ht / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
    if views == 1:
        return (
            poses[0].unsqueeze(0).expand(batch, 4, 4).contiguous(),
            intrinsics[0].unsqueeze(0).expand(batch, 3, 3).contiguous(),
        )
    return (
        torch.stack(poses, dim=0).unsqueeze(0).expand(batch, views, 4, 4).contiguous(),
        torch.stack(intrinsics, dim=0).unsqueeze(0).expand(batch, views, 3, 3).contiguous(),
    )


def analytic_transient_mb(candidates: int, views: int, batch: int, hp: int, wp: int) -> float:
    """Bytes of the large projection tensors, independent of hardware."""
    elements = candidates * batch * hp * wp
    bytes_per_float = 4
    # points stack (3) + homogeneous points (4) + camera-space points (3*V)
    # + projected points (3*V) + z/u/v/x/y/valid bookkeeping.
    tensors = elements * (3 + 4 + 3 * views + 3 * views)
    bookkeeping = elements * views * (2 * bytes_per_float + 3)
    return (tensors * bytes_per_float + bookkeeping) / (1024.0 * 1024.0)


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


class _RssSampler:
    def __init__(self):
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            value = _rss_mb()
            if value is not None:
                self.peak = max(self.peak, value)
            time.sleep(0.002)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        return False


def run_worker(config: dict) -> dict:
    torch.manual_seed(0)
    num_candidates = int(config["candidates"])
    views = int(config["views"])
    hw = (int(config["height"]), int(config["width"]))
    downsample = 4
    hp = (hw[0] + downsample - 1) // downsample
    wp = (hw[1] + downsample - 1) // downsample
    chunk_size = config.get("chunk_size")
    mode = config["mode"]
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA is not available")

    cache = build_cache(num_candidates, batch=1, hp=hp, wp=wp, seed=7)
    w2c, k = build_targets(batch=1, views=views, hw=hw)
    if device == "cuda":
        # Keep the points resident so each projection is a pure GPU matmul.
        cache._world_points = [p.to("cuda") for p in cache._world_points]
        w2c = w2c.to("cuda")
        k = k.to("cuda")

    def call(chunk):
        torch.manual_seed(0)
        return cache.retrieve(
            w2c,
            k,
            hw,
            num_latents=8,
            skip_last_n=0,
            random=(mode == "random"),
            max_coverage=(mode == "coverage"),
            chunk_size=chunk,
        )

    # Warmup: allocate allocator arenas before the measured call.
    call(chunk_size)
    gc.collect()

    baseline = _rss_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with _RssSampler() as sampler:
        start = time.perf_counter()
        selection = call(chunk_size)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    rss_delta = None
    if baseline is not None and sampler.peak > 0:
        rss_delta = max(0.0, sampler.peak - baseline)

    # The chunked path only ever materializes `chunk_size` candidates' worth
    # of projection tensors; the reference materializes all of them.
    effective = num_candidates
    if chunk_size is not None and 0 < int(chunk_size) < num_candidates:
        effective = int(chunk_size)

    return {
        "path": "chunked" if chunk_size else "reference",
        "device": device,
        "candidates": num_candidates,
        "views": views,
        "height": hw[0],
        "width": hw[1],
        "mode": mode,
        "chunk_size": chunk_size,
        "secs": elapsed,
        "selected": len(selection),
        "selection": selection,
        "analytic_transient_mb": analytic_transient_mb(effective, views, 1, hp, wp),
        "rss_delta_mb": rss_delta,
        "cuda_peak_mb": (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else None
        ),
    }


def _spawn(config: dict) -> dict:
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", json.dumps(config)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed for {config}:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _median(values):
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _spawn_repeated(config: dict, repeats: int) -> dict:
    runs = [_spawn(config) for _ in range(max(1, repeats))]
    first = runs[0]
    selection = first["selection"]
    if any(run["selection"] != selection for run in runs):
        raise RuntimeError(f"selection changed across repeats for {config}")
    return {
        **first,
        "secs": _median([run["secs"] for run in runs]),
        "rss_delta_mb": _median([run["rss_delta_mb"] for run in runs]),
        "cuda_peak_mb": _median([run["cuda_peak_mb"] for run in runs]),
        "repeats": len(runs),
    }


def gpu_parity(config: dict) -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available; GPU parity not measured.")
        return 0
    num_candidates = int(config["candidates"])
    views = int(config["views"])
    hw = (int(config["height"]), int(config["width"]))
    downsample = 4
    hp = (hw[0] + downsample - 1) // downsample
    wp = (hw[1] + downsample - 1) // downsample
    cache = build_cache(num_candidates, batch=1, hp=hp, wp=wp)
    cache._world_points = [point.to("cuda") for point in cache._world_points]
    w2c, k = build_targets(batch=1, views=views, hw=hw)
    w2c = w2c.to("cuda")
    k = k.to("cuda")
    for mode in ("coverage", "random"):
        torch.manual_seed(0)
        reference = cache.retrieve(
            w2c, k, hw, num_latents=8, random=(mode == "random"),
            max_coverage=(mode == "coverage"), chunk_size=None,
        )
        torch.manual_seed(0)
        chunked = cache.retrieve(
            w2c, k, hw, num_latents=8, random=(mode == "random"),
            max_coverage=(mode == "coverage"), chunk_size=32,
        )
        if chunked != reference:
            print(f"FAIL: GPU parity mismatch for mode={mode}: {chunked} != {reference}")
            return 1
    print("PASS: GPU parity matches for coverage and random modes.")
    return 0


def _matrix(quick: bool, full: bool) -> list[dict]:
    if quick:
        candidates = [128, 512]
        views = [1]
        modes = ["coverage", "random"]
    elif full:
        candidates = [64, 128, 256, 512]
        views = [1, 3]
        modes = ["coverage", "random"]
    else:
        candidates = [64, 128, 256, 512]
        views = [1]
        modes = ["coverage", "random"]
    matrix = []
    for count in candidates:
        for view_count in views:
            for mode in modes:
                matrix.append(
                    {
                        "candidates": count,
                        "views": view_count,
                        "height": 480,
                        "width": 832,
                        "mode": mode,
                    }
                )
    return matrix


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'cands':>6} {'views':>5} {'mode':>8} {'path':>9} {'chunk':>5} "
        f"{'secs':>8} {'sel':>3} {'analytic MB':>12} {'RSS MB':>8} {'CUDA MB':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        rss = row["rss_delta_mb"]
        cuda = row["cuda_peak_mb"]
        print(
            f"{row['candidates']:>6} {row['views']:>5} {row['mode']:>8} {row['path']:>9} "
            f"{str(row['chunk_size']):>5} {row['secs']:>8.4f} {row['selected']:>3} "
            f"{row['analytic_transient_mb']:>12.1f} "
            f"{(rss if rss is not None else float('nan')):>8.1f} "
            f"{(cuda if cuda is not None else float('nan')):>8.1f}"
        )


def summarise(rows: list[dict]) -> dict:
    grouped: dict[tuple, dict] = {}
    for row in rows:
        key = (row["candidates"], row["views"], row["mode"])
        grouped.setdefault(key, {})[row["path"]] = row

    comparisons = []
    for key, paths in sorted(grouped.items()):
        if "reference" not in paths or "chunked" not in paths:
            continue
        old = paths["reference"]
        new = paths["chunked"]
        comparisons.append(
            {
                "candidates": key[0],
                "views": key[1],
                "mode": key[2],
                "parity": old["selection"] == new["selection"],
                "selected": old["selected"],
                "old_secs": old["secs"],
                "chunked_secs": new["secs"],
                "latency_ratio": new["secs"] / old["secs"] if old["secs"] else None,
                "old_analytic_mb": old["analytic_transient_mb"],
                "chunked_analytic_mb": new["analytic_transient_mb"],
                "analytic_reduction": (
                    old["analytic_transient_mb"] / new["analytic_transient_mb"]
                    if new["analytic_transient_mb"]
                    else None
                ),
                "old_rss_mb": old["rss_delta_mb"],
                "chunked_rss_mb": new["rss_delta_mb"],
                "rss_reduction": (
                    old["rss_delta_mb"] / new["rss_delta_mb"]
                    if old["rss_delta_mb"] and new["rss_delta_mb"]
                    else None
                ),
                "old_cuda_mb": old.get("cuda_peak_mb"),
                "chunked_cuda_mb": new.get("cuda_peak_mb"),
                "cuda_reduction": (
                    old["cuda_peak_mb"] / new["cuda_peak_mb"]
                    if old.get("cuda_peak_mb") and new.get("cuda_peak_mb")
                    else None
                ),
            }
        )

    ratios = [c["analytic_reduction"] for c in comparisons if c["analytic_reduction"]]
    latency = [c["latency_ratio"] for c in comparisons if c["latency_ratio"]]
    rss_ratios = [c["rss_reduction"] for c in comparisons if c["rss_reduction"]]
    cuda_ratios = [c["cuda_reduction"] for c in comparisons if c["cuda_reduction"]]
    return {
        "comparisons": comparisons,
        "all_parity": all(c["parity"] for c in comparisons) if comparisons else False,
        "max_analytic_reduction": max(ratios) if ratios else None,
        "median_analytic_reduction": statistics.median(ratios) if ratios else None,
        "median_latency_ratio": statistics.median(latency) if latency else None,
        "max_rss_reduction": max(rss_ratios) if rss_ratios else None,
        "max_cuda_reduction": max(cuda_ratios) if cuda_ratios else None,
        "median_cuda_reduction": statistics.median(cuda_ratios) if cuda_ratios else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small matrix")
    parser.add_argument("--full", action="store_true", help="candidate x view matrix")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=1, help="fresh subprocess repeats per row")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="device for the matrix; --gpu-parity is separate",
    )
    parser.add_argument("--gpu-parity", action="store_true")
    parser.add_argument(
        "--require-fair",
        action="store_true",
        help="fail if any comparison lacks parity or uses mismatched selections",
    )
    args = parser.parse_args()

    if args.worker is not None:
        print(json.dumps(run_worker(json.loads(args.worker))))
        return 0

    if args.gpu_parity:
        return gpu_parity(
            {"candidates": 512, "views": 1, "height": 480, "width": 832}
        )

    matrix = _matrix(args.quick, args.full)
    rows = []
    for config in matrix:
        for chunk_size in (None, 32):
            worker_config = dict(config)
            worker_config["chunk_size"] = chunk_size
            worker_config["device"] = args.device
            rows.append(_spawn_repeated(worker_config, args.repeats))

    print_table(rows)
    summary = summarise(rows)

    print("\ncomparisons (chunked vs reference):")
    header = (
        f"{'cands':>6} {'views':>5} {'mode':>8} {'parity':>6} {'analytic x':>10} "
        f"{'RSS x':>7} {'CUDA x':>7} {'latency x':>9}"
    )
    print(header)
    print("-" * len(header))
    for comparison in summary["comparisons"]:
        print(
            f"{comparison['candidates']:>6} {comparison['views']:>5} {comparison['mode']:>8} "
            f"{str(comparison['parity']):>6} "
            f"{(comparison['analytic_reduction'] or float('nan')):>10.2f} "
            f"{(comparison['rss_reduction'] or float('nan')):>7.2f} "
            f"{(comparison['cuda_reduction'] or float('nan')):>7.2f} "
            f"{(comparison['latency_ratio'] or float('nan')):>9.2f}"
        )

    print("\nsummary:")
    print(f"  all parity:              {summary['all_parity']}")
    print(f"  median analytic drop:    {summary['median_analytic_reduction']:.2f}x")
    print(f"  max analytic drop:       {summary['max_analytic_reduction']:.2f}x")
    print(f"  max measured RSS drop:   {summary['max_rss_reduction']}")
    print(f"  max CUDA peak drop:      {summary['max_cuda_reduction']}")
    print(f"  median CUDA peak drop:   {summary['median_cuda_reduction']}")
    print(f"  median latency ratio:    {summary['median_latency_ratio']:.2f}x")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")

    if args.require_fair:
        fair = summary["all_parity"] and len(summary["comparisons"]) == len(matrix)
        if not fair:
            print(
                "\nFAIL: --require-fair saw a parity mismatch, a mismatched matrix, "
                "or an empty comparison set."
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
