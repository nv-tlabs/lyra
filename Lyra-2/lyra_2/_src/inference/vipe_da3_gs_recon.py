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

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from plyfile import PlyData

from lyra_2._src.inference.depth_utils import load_da3_model


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECON_DA3_MODEL_PATH = REPO_ROOT / "checkpoints" / "recon" / "model.pt"


@dataclass(slots=True)
class VIPEOutputs:
    intrinsics: torch.Tensor
    extrinsics_c2w: torch.Tensor
    depth: torch.Tensor
    frame_ids: torch.Tensor


def _ensure_da3_on_syspath() -> Path:
    da3_src_root = Path(__file__).resolve().parent / "depth_anything_3" / "src"
    if str(da3_src_root) not in sys.path:
        sys.path.insert(0, str(da3_src_root))
    return da3_src_root


def _to_rgb_tensor_0_1(frames: torch.Tensor) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be a torch.Tensor, got {type(frames)}")
    if frames.ndim != 4:
        raise ValueError(f"frames must be 4D, got shape {tuple(frames.shape)}")

    if frames.shape[-1] == 3:
        rgb = frames
    elif frames.shape[1] == 3:
        rgb = frames.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"frames must be (T,H,W,3) or (T,3,H,W), got {tuple(frames.shape)}")

    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    else:
        rgb = rgb.float()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

    return rgb.clamp_(0.0, 1.0)


def _compose_vipe_config(
    *,
    overrides: Sequence[str],
    config_dir: str | Path,
    config_name: str = "default",
):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    cfg_dir = Path(config_dir).resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _import_vipe_class():
    vipe_root = Path(__file__).resolve().parent / "vipe"
    if not (vipe_root / "vipe" / "__init__.py").is_file() or not (vipe_root / "configs").is_dir():
        raise ImportError(f"VIPE submodule not found at {vipe_root}")

    import_root = vipe_root
    config_dir = vipe_root / "configs"
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

    try:
        from vipe.pipeline import make_pipeline  # type: ignore
        from vipe.streams.base import VideoFrame, VideoStream  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import VIPE from {import_root}") from e

    class InMemoryVideoStream(VideoStream):  # type: ignore[misc]
        def __init__(
            self,
            frames_rgb_0_1_thwc: torch.Tensor,
            fps_value: float = 16.0,
            stream_name: str = "inmem",
            device: Optional[torch.device] = None,
        ) -> None:
            super().__init__()
            self.frames_rgb_0_1_thwc = _to_rgb_tensor_0_1(frames_rgb_0_1_thwc)
            self.fps_value = float(fps_value)
            self.stream_name = stream_name
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def frame_size(self) -> tuple[int, int]:
            _, h, w, _ = self.frames_rgb_0_1_thwc.shape
            return (h, w)

        def name(self) -> str:
            return self.stream_name

        def fps(self) -> float:
            return self.fps_value

        def __len__(self) -> int:
            return int(self.frames_rgb_0_1_thwc.shape[0])

        def __getitem__(self, idx: int) -> VideoFrame:
            if idx < 0:
                idx = len(self) + idx
            if idx < 0 or idx >= len(self):
                raise IndexError(idx)
            rgb = self.frames_rgb_0_1_thwc[idx].to(self.device, non_blocking=True)
            return VideoFrame(raw_frame_idx=idx, rgb=rgb)

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    class VIPEWrapper:
        def __init__(
            self,
            overrides: Sequence[str],
            *,
            config_dir: str | Path | None = None,
            config_name: str = "default",
            device: str | torch.device | None = None,
            fast_mode: bool = True,
        ) -> None:
            if config_dir is None:
                config_dir = _config_dir
            self.cfg = _compose_vipe_config(
                overrides=overrides,
                config_dir=config_dir,
                config_name=config_name,
            )
            if fast_mode:
                if self.cfg.pipeline.init.get("instance") is not None:
                    self.cfg.pipeline.init.instance = None
                if self.cfg.pipeline.post.get("compute_backward_flow") is not None:
                    self.cfg.pipeline.post.compute_backward_flow = False
                if self.cfg.pipeline.output.get("save_viz") is not None:
                    self.cfg.pipeline.output.save_viz = False
                if self.cfg.pipeline.output.get("save_artifacts") is not None:
                    self.cfg.pipeline.output.save_artifacts = False
                if self.cfg.pipeline.output.get("save_metrics") is not None:
                    self.cfg.pipeline.output.save_metrics = False
                if self.cfg.pipeline.output.get("save_slam_map") is not None:
                    self.cfg.pipeline.output.save_slam_map = False
            self.pipeline = make_pipeline(self.cfg.pipeline)
            self.pipeline.return_output_streams = True
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        def infer_frames(
            self,
            frames: torch.Tensor,
            *,
            fps: float = 30.0,
            name: str = "inmem",
        ) -> VIPEOutputs:
            stream = InMemoryVideoStream(
                frames_rgb_0_1_thwc=frames,
                fps_value=float(fps),
                stream_name=name,
                device=self.device,
            )
            out = self.pipeline.run(stream)
            assert out.output_streams is not None and len(out.output_streams) == 1
            output_stream = out.output_streams[0]

            intr_list = []
            c2w_list = []
            depth_list = []
            frame_ids = []
            for frame in output_stream:
                f = frame.cpu()
                frame_ids.append(f.raw_frame_idx)
                assert f.intrinsics is not None and f.pose is not None
                intr_list.append(f.intrinsics.float())
                c2w_list.append(f.pose.matrix().float())
                if f.metric_depth is not None:
                    depth_metric = f.metric_depth.float()
                else:
                    height, width = stream.frame_size()
                    depth_metric = torch.zeros((height, width), dtype=torch.float32)
                depth_list.append(depth_metric)

            return VIPEOutputs(
                intrinsics=torch.stack(intr_list, dim=0),
                extrinsics_c2w=torch.stack(c2w_list, dim=0),
                depth=torch.stack(depth_list, dim=0),
                frame_ids=torch.tensor(frame_ids, dtype=torch.int64),
            )

    _config_dir = config_dir
    return VIPEWrapper


def _vipe_default_overrides(output_path: Path) -> List[str]:
    return [
        "pipeline=default",
        "pipeline.slam.optimize_intrinsics=false",
        "pipeline.post.depth_align_model=null",
        "pipeline.output.save_artifacts=false",
        "pipeline.output.save_viz=false",
        f"pipeline.output.path={output_path}",
    ]


def _intrinsics_vec_to_k33(intrinsics_vec: torch.Tensor) -> torch.Tensor:
    if intrinsics_vec.ndim != 2 or intrinsics_vec.shape[1] < 4:
        raise ValueError(f"Expected intrinsics shape (T,>=4), got {tuple(intrinsics_vec.shape)}")
    fx, fy, cx, cy = intrinsics_vec[:, 0], intrinsics_vec[:, 1], intrinsics_vec[:, 2], intrinsics_vec[:, 3]
    t = int(intrinsics_vec.shape[0])
    k = torch.zeros((t, 3, 3), dtype=intrinsics_vec.dtype, device=intrinsics_vec.device)
    k[:, 0, 0] = fx
    k[:, 1, 1] = fy
    k[:, 0, 2] = cx
    k[:, 1, 2] = cy
    k[:, 2, 2] = 1.0
    return k


def _probe_video(video_path: str) -> Tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if frame_count <= 0:
        frame_count = 0
    if fps <= 1e-6:
        fps = 30.0
    return frame_count, fps


def _sample_indices(num_frames: int, stride: int, max_views: int = 0) -> List[int]:
    stride = max(int(stride), 1)
    indices = list(range(0, int(num_frames), stride))
    if max_views > 0:
        indices = indices[: int(max_views)]
    return indices


def _uniform_subsample_indices(num_frames: int, max_frames: int) -> List[int]:
    num_frames = int(num_frames)
    max_frames = int(max_frames)
    if num_frames <= 0:
        return []
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    return np.floor(np.linspace(0, num_frames - 1, num=max_frames)).astype(np.int64).tolist()


def _read_video_frames_rgb(video_path: str, indices: List[int]) -> List[np.ndarray]:
    if not indices:
        return []

    wanted = set(int(i) for i in indices)
    last_needed = int(max(wanted))
    frames: List[np.ndarray] = []
    read_ids: List[int] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if frame_idx in wanted:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                read_ids.append(frame_idx)
                if len(frames) == len(wanted):
                    break
            if frame_idx >= last_needed:
                break
            frame_idx += 1
    finally:
        cap.release()

    if len(frames) != len(wanted):
        missing = sorted(list(wanted - set(read_ids)))
        print(
            f"[vipe_da3_gs] Warning: requested {len(wanted)} frames, got {len(frames)}. "
            f"Missing={missing[:10]}"
        )

    return frames


def _compute_aligned_pred_w2c(pred_extr_np: np.ndarray, input_w2c_np: np.ndarray) -> np.ndarray:
    _ensure_da3_on_syspath()
    from depth_anything_3.utils.geometry import affine_inverse_np  # type: ignore
    from depth_anything_3.utils.pose_align import align_poses_umeyama  # type: ignore

    pred_44 = pred_extr_np.copy()
    if pred_44.shape[-2] == 3:
        pad = np.zeros((*pred_44.shape[:-2], 4, 4), dtype=pred_44.dtype)
        pad[..., :3, :4] = pred_44
        pad[..., 3, 3] = 1.0
        pred_44 = pad

    inp_44 = input_w2c_np.copy()
    if inp_44.shape[-2] == 3:
        pad = np.zeros((*inp_44.shape[:-2], 4, 4), dtype=inp_44.dtype)
        pad[..., :3, :4] = inp_44
        pad[..., 3, 3] = 1.0
        inp_44 = pad

    r, t, s = align_poses_umeyama(pred_44, inp_44)
    r_inv = r.T
    pred_c2w = affine_inverse_np(pred_44)

    aligned_c2w = np.zeros_like(pred_c2w)
    aligned_c2w[:, :3, :3] = np.einsum("ij,njk->nik", r_inv, pred_c2w[:, :3, :3])
    trans_shifted = pred_c2w[:, :3, 3] - t[None, :]
    aligned_c2w[:, :3, 3] = np.einsum("ij,nj->ni", r_inv, trans_shifted) / s
    aligned_c2w[:, 3, 3] = 1.0

    return affine_inverse_np(aligned_c2w).astype(np.float32)


def _pad_to_44(mat: np.ndarray) -> np.ndarray:
    if mat.shape[-2:] == (4, 4):
        return mat
    padded = np.zeros((*mat.shape[:-2], 4, 4), dtype=mat.dtype)
    padded[..., :3, :4] = mat[..., :3, :4]
    padded[..., 3, 3] = 1.0
    return padded


def _interpolate_w2c(
    w2c_keyframes: np.ndarray,
    key_indices: List[int],
    n_total: int,
) -> np.ndarray:
    w2c_keyframes = _pad_to_44(w2c_keyframes)

    if len(key_indices) == 1:
        return np.repeat(w2c_keyframes[:1], n_total, axis=0).astype(np.float32)

    from scipy.spatial.transform import Rotation, Slerp

    c2w = np.linalg.inv(w2c_keyframes)
    times_key = np.array(key_indices, dtype=np.float64)
    rotations = Rotation.from_matrix(c2w[:, :3, :3])
    translations = c2w[:, :3, 3].astype(np.float64)

    slerp = Slerp(times_key, rotations)
    times_all = np.arange(n_total, dtype=np.float64)
    times_clamped = np.clip(times_all, times_key[0], times_key[-1])

    rotations_interp = slerp(times_clamped)
    translations_interp = np.column_stack(
        [np.interp(times_clamped, times_key, translations[:, dim]) for dim in range(3)]
    )

    c2w_dense = np.zeros((n_total, 4, 4), dtype=np.float64)
    c2w_dense[:, :3, :3] = rotations_interp.as_matrix()
    c2w_dense[:, :3, 3] = translations_interp
    c2w_dense[:, 3, 3] = 1.0
    return np.linalg.inv(c2w_dense).astype(np.float32)


def _load_gaussian_ply_to_gaussians(ply_path: str, device: torch.device):
    _ensure_da3_on_syspath()
    from depth_anything_3.specs import Gaussians  # type: ignore

    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"No 'vertex' element in PLY: {ply_path}")
    vertices = ply["vertex"].data
    names = list(vertices.dtype.names or [])

    def _stack_props(prefix: str, count: int) -> np.ndarray:
        props = []
        for idx in range(count):
            key = f"{prefix}{idx}"
            if key not in names:
                raise ValueError(f"Missing '{key}' in PLY: {ply_path}")
            props.append(vertices[key].astype(np.float32, copy=False))
        return np.stack(props, axis=1)

    means = np.stack(
        [
            vertices["x"].astype(np.float32, copy=False),
            vertices["y"].astype(np.float32, copy=False),
            vertices["z"].astype(np.float32, copy=False),
        ],
        axis=1,
    )
    scales = np.exp(_stack_props("scale_", 3))
    rotations = _stack_props("rot_", 4)
    opacities = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float32, copy=False)))

    f_dc = _stack_props("f_dc_", 3)
    f_rest_keys = sorted(
        [key for key in names if key.startswith("f_rest_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    if f_rest_keys:
        f_rest = np.stack([vertices[key].astype(np.float32, copy=False) for key in f_rest_keys], axis=1)
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected f_rest size {f_rest.shape[1]} in PLY: {ply_path}")
        d_sh = f_rest.shape[1] // 3 + 1
        f_rest = f_rest.reshape(f_rest.shape[0], 3, d_sh - 1)
        harmonics = np.concatenate([f_dc[:, :, None], f_rest], axis=2)
    else:
        harmonics = f_dc[:, :, None]

    return Gaussians(
        means=torch.from_numpy(means).to(device=device).unsqueeze(0),
        scales=torch.from_numpy(scales).to(device=device).unsqueeze(0),
        rotations=torch.from_numpy(rotations).to(device=device).unsqueeze(0),
        harmonics=torch.from_numpy(harmonics).to(device=device).unsqueeze(0),
        opacities=torch.from_numpy(opacities).to(device=device).unsqueeze(0),
    )


def _save_video_mp4(video_path: str, frames_thwc: np.ndarray, fps: float) -> None:
    if frames_thwc.ndim != 4 or frames_thwc.shape[-1] != 3:
        raise ValueError(f"Expected frames shape (T,H,W,3), got {tuple(frames_thwc.shape)}")

    t = int(frames_thwc.shape[0])
    if t == 0:
        raise ValueError("No frames to save.")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    frames_uint8 = frames_thwc
    if frames_uint8.dtype != np.uint8:
        frames_uint8 = np.clip(frames_uint8, 0, 255).astype(np.uint8)
    frames_list = [frame for frame in frames_uint8]
    clip = ImageSequenceClip(frames_list, fps=float(max(fps, 1.0)))
    try:
        clip.write_videofile(
            video_path,
            codec="libx264",
            audio=False,
            fps=float(max(fps, 1.0)),
            ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
        )
    finally:
        clip.close()


def _collect_vipe_images(
    video_path: str,
    vipe_stride: int,
    max_frames: int,
    max_views: int,
) -> tuple[List[np.ndarray], List[int], float]:
    frame_count, fps = _probe_video(video_path)

    if frame_count > 0:
        total = frame_count
        if max_frames > 0:
            total = min(total, max_frames)
        indices_vipe = _sample_indices(total, vipe_stride, max_views)
        images_vipe = _read_video_frames_rgb(video_path, indices_vipe)
        return images_vipe, indices_vipe, fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frames_tmp: List[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames_tmp.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames_tmp) >= max_frames:
                break
    finally:
        cap.release()

    indices_vipe = _sample_indices(len(frames_tmp), vipe_stride, max_views)
    images_vipe = [frames_tmp[idx] for idx in indices_vipe]
    return images_vipe, indices_vipe, fps


def _build_output_dir(input_video_path: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    input_video = Path(input_video_path).expanduser().resolve()
    return input_video.with_name(f"{input_video.stem}_gs_ours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-video VIPE pose estimation + DA3 Gaussian reconstruction."
    )
    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no_vipe",
        action="store_true",
        help="Skip VIPE pose estimation; DA3 reconstructs without input poses and its predicted poses are used for rendering.",
    )
    parser.add_argument("--vipe_overrides", type=str, nargs="+", default=None)
    parser.add_argument("--vipe_full_mode", action="store_true")

    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument(
        "--da3_max_frames",
        type=int,
        default=128,
        help="Uniformly subsample VIPE frames to at most this many views for DA3.",
    )

    parser.add_argument(
        "--da3_model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    parser.add_argument(
        "--da3_model_path_custom",
        type=str,
        default=str(DEFAULT_RECON_DA3_MODEL_PATH),
    )
    parser.add_argument("--da3_process_res", type=int, default=None)
    parser.add_argument(
        "--da3_process_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=0,
        help="If > 0, use as DA3 short-side cap via lower_bound_resize.",
    )

    parser.add_argument("--gs_down_ratio", type=int, default=2)
    parser.add_argument("--gs_scale_extra_multiplier", type=float, default=1.0)
    parser.add_argument("--gs_ply_prune_opacity_percentile", type=float, default=None)
    parser.add_argument(
        "--no_gs_ds_feature_mode",
        dest="gs_ds_feature_mode",
        action="store_false",
        help="Disable the default release-friendly GS feature downsampling mode.",
    )
    parser.set_defaults(gs_ds_feature_mode=True)

    parser.add_argument(
        "--use_da3_render_pose",
        dest="use_da3_render_pose",
        action="store_true",
        help="Render with DA3-aligned predicted poses interpolated to VIPE cadence.",
    )
    parser.add_argument(
        "--no_da3_render_pose",
        dest="use_da3_render_pose",
        action="store_false",
        help="Render with raw VIPE poses instead.",
    )
    parser.set_defaults(use_da3_render_pose=True)

    parser.add_argument("--render_fps", type=float, default=None)
    parser.add_argument("--render_chunk_size", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_video = Path(args.input_video_path).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir = _build_output_dir(str(input_video), args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done_marker = output_dir / ".done"
    if done_marker.is_file() and not args.force:
        print(f"[vipe_da3_gs] Skipping {input_video.name}: {done_marker} exists. Use --force to re-run.")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[vipe_da3_gs] Input video: {input_video}")
    print(f"[vipe_da3_gs] Output dir:  {output_dir}")
    print(f"[vipe_da3_gs] Device:      {device}")

    da3_model_path_custom = None
    if args.da3_model_path_custom:
        da3_model_path_custom = str(Path(args.da3_model_path_custom).expanduser().resolve())
        if not Path(da3_model_path_custom).is_file():
            raise FileNotFoundError(f"DA3 checkpoint not found: {da3_model_path_custom}")
        print(f"[vipe_da3_gs] DA3 ckpt:    {da3_model_path_custom}")

    print("[vipe_da3_gs] Loading DA3 model...")
    da3_model = load_da3_model(
        da3_model_name=args.da3_model_name,
        da3_model_path_custom=da3_model_path_custom,
        device=str(device),
    )
    da3_model.eval()

    skip_vipe = bool(args.no_vipe)

    if not skip_vipe:
        VIPE = _import_vipe_class()

    print("[vipe_da3_gs] Reading video frames...")
    images_all, indices_all, fps = _collect_vipe_images(
        str(input_video),
        vipe_stride=1,
        max_frames=args.max_frames,
        max_views=0,
    )
    if not images_all:
        raise RuntimeError("No frames read from video.")

    indices_da3_rel = _uniform_subsample_indices(len(images_all), args.da3_max_frames)
    if not indices_da3_rel:
        raise RuntimeError("No frames selected for DA3.")

    images_da3 = [images_all[idx] for idx in indices_da3_rel]
    indices_da3 = [indices_all[idx] for idx in indices_da3_rel]
    eff_fps = float(fps)

    print(
        f"[vipe_da3_gs] fps={fps:.4g}, no_vipe={skip_vipe}, da3_max_frames={args.da3_max_frames}, "
        f"frames_all={len(images_all)}, frames_da3={len(images_da3)}"
    )

    if skip_vipe:
        w2c_np_vipe_full = None
        k_np_vipe_full = None
        w2c_np_da3 = None
        k_np_da3 = None
    else:
        frames_np = np.stack(images_all, axis=0).astype(np.float32) / 255.0
        frames_thwc = torch.from_numpy(frames_np).contiguous()

    with tempfile.TemporaryDirectory(prefix="vipe_da3_gs_") as tmpdir:
        if not skip_vipe:
            vipe_output_path = Path(tmpdir) / "vipe_out"
            vipe_output_path.mkdir(parents=True, exist_ok=True)
            vipe_overrides = args.vipe_overrides or _vipe_default_overrides(
                vipe_output_path,
            )

            print("[vipe_da3_gs] Loading VIPE...")
            vipe_kwargs = {"fast_mode": not bool(args.vipe_full_mode)}
            vipe = VIPE(vipe_overrides, **vipe_kwargs)

            print("[vipe_da3_gs] Running VIPE...")
            vipe_out = vipe.infer_frames(frames_thwc, fps=eff_fps, name=input_video.stem)

            c2w = vipe_out.extrinsics_c2w.to(dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)
            intrinsics_vipe = _intrinsics_vec_to_k33(vipe_out.intrinsics.to(dtype=torch.float32))

            w2c_np_vipe_full = w2c.cpu().numpy().astype(np.float32)
            k_np_vipe_full = intrinsics_vipe.cpu().numpy().astype(np.float32)
            w2c_np_da3 = w2c_np_vipe_full[indices_da3_rel]
            k_np_da3 = k_np_vipe_full[indices_da3_rel]

            np.savez(
                output_dir / "vipe_predictions.npz",
                frame_ids=vipe_out.frame_ids.cpu().numpy().astype(np.int64),
                w2c_vipe=w2c_np_vipe_full,
                intrinsics_vipe=k_np_vipe_full,
                w2c_da3=w2c_np_da3,
                intrinsics_da3=k_np_da3,
                indices_vipe=np.asarray(indices_all, dtype=np.int64),
                indices_da3=np.asarray(indices_da3, dtype=np.int64),
                fps=np.asarray([eff_fps], dtype=np.float32),
                input_video_path=np.asarray([str(input_video)]),
            )

        if args.da3_process_res is not None:
            da3_process_res = int(args.da3_process_res)
            da3_process_method = str(args.da3_process_method)
        elif int(args.max_resolution) > 0:
            da3_process_res = int(args.max_resolution)
            da3_process_method = "lower_bound_resize"
        else:
            h0, w0 = images_da3[0].shape[:2]
            da3_process_res = int(max(h0, w0))
            da3_process_method = "upper_bound_resize"

        _ensure_da3_on_syspath()
        from depth_anything_3.utils.gsply_helpers import save_gaussian_ply  # type: ignore

        if skip_vipe:
            # Pass 1: DA3 without input poses to get predicted extrinsics
            print(
                f"[vipe_da3_gs] DA3 pass 1 (pose estimation): views={len(images_da3)} "
                f"process_res={da3_process_res} infer_gs=False"
            )
            pred_pose = da3_model.inference(
                image=images_da3,
                extrinsics=None,
                intrinsics=None,
                align_to_input_extrinsics=False,
                align_to_input_ext_scale=False,
                infer_gs=False,
                process_res=da3_process_res,
                process_res_method=da3_process_method,
                export_dir=None,
                export_format="mini_npz",
            )
            if pred_pose.extrinsics is None or pred_pose.intrinsics is None:
                raise RuntimeError("DA3 pass 1 did not return predicted poses.")
            w2c_np_da3 = _pad_to_44(np.asarray(pred_pose.extrinsics, dtype=np.float32))
            k_np_da3 = np.asarray(pred_pose.intrinsics, dtype=np.float32)
            da3_pred_w2c = w2c_np_da3.copy()
            da3_pred_k = k_np_da3.copy()
            print(
                f"[vipe_da3_gs] DA3 pass 1 done. "
                f"extrinsics {w2c_np_da3.shape}, intrinsics {k_np_da3.shape}"
            )

            del pred_pose
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Pass 2: DA3 with pass-1 poses as condition for GS reconstruction
            print(
                f"[vipe_da3_gs] DA3 pass 2 (GS recon): views={len(images_da3)} "
                f"process_res={da3_process_res} infer_gs=True"
            )

        else:
            print(
                f"[vipe_da3_gs] DA3 single pass: views={len(images_da3)} "
                f"process_res={da3_process_res}"
            )

        pred = da3_model.inference(
            image=images_da3,
            extrinsics=w2c_np_da3,
            intrinsics=k_np_da3,
            align_to_input_extrinsics=False,
            align_to_input_ext_scale=False,
            infer_gs=True,
            process_res=da3_process_res,
            process_res_method=da3_process_method,
            export_dir=None,
            export_format="mini_npz",
            use_aligned_pred_cam=True,
            gs_down_ratio=args.gs_down_ratio,
            gs_scale_extra_multiplier=args.gs_scale_extra_multiplier,
            gs_ds_feature_mode=args.gs_ds_feature_mode,
        )

        if not skip_vipe:
            aligned_w2c_da3 = None
            if args.use_da3_render_pose and pred.extrinsics is not None:
                aligned_w2c_da3 = _compute_aligned_pred_w2c(
                    np.asarray(pred.extrinsics, dtype=np.float32),
                    w2c_np_da3,
                )

        final_ply_path = output_dir / "reconstructed_scene.ply"
        depth_t = torch.from_numpy(np.asarray(pred.depth, dtype=np.float32)).float()
        save_gaussian_ply(
            pred.gaussians,
            str(final_ply_path),
            ctx_depth=depth_t.unsqueeze(-1),
            prune_by_opacity_percentile=args.gs_ply_prune_opacity_percentile,
            prune_border_gs=False
            if (
                args.gs_ply_prune_opacity_percentile is not None
                and args.gs_ply_prune_opacity_percentile > 0
            )
            else True,
        )
        print(f"[vipe_da3_gs] Saved PLY to {final_ply_path}")

        if skip_vipe:
            w2c_render = _interpolate_w2c(da3_pred_w2c, indices_da3_rel, len(images_all))
            if da3_pred_k is not None:
                k_da3_first = da3_pred_k[0:1]
                k_render = np.repeat(k_da3_first, len(images_all), axis=0).astype(np.float32)
            else:
                raise RuntimeError("DA3 did not return predicted intrinsics; cannot render without VIPE.")
            print("[vipe_da3_gs] Rendering with DA3 predicted poses (no VIPE).")
        elif args.use_da3_render_pose and aligned_w2c_da3 is not None:
            w2c_render = _interpolate_w2c(aligned_w2c_da3, indices_da3_rel, len(images_all))
            k_render = k_np_vipe_full
            print("[vipe_da3_gs] Rendering with DA3-aligned poses.")
        else:
            w2c_render = w2c_np_vipe_full
            k_render = k_np_vipe_full
            if args.use_da3_render_pose:
                print("[vipe_da3_gs] Warning: DA3 poses unavailable, falling back to VIPE poses.")
            else:
                print("[vipe_da3_gs] Rendering with VIPE poses.")

        cameras_data = {
            "w2c_render": w2c_render,
            "indices_da3": np.asarray(indices_da3, dtype=np.int64),
            "fps": np.asarray([eff_fps], dtype=np.float32),
            "no_vipe": np.asarray([int(skip_vipe)], dtype=np.int32),
        }
        if not skip_vipe:
            cameras_data.update({
                "w2c_vipe": w2c_np_vipe_full,
                "intrinsics_vipe": k_np_vipe_full,
                "w2c_da3": w2c_np_da3,
                "intrinsics_da3": k_np_da3,
                "indices_vipe": np.asarray(indices_all, dtype=np.int64),
                "use_da3_render_pose": np.asarray([int(args.use_da3_render_pose)], dtype=np.int32),
            })
        else:
            cameras_data.update({
                "w2c_da3_pred": da3_pred_w2c,
                "intrinsics_da3_pred": da3_pred_k,
                "intrinsics_render": k_render,
            })
        np.savez(output_dir / "cameras.npz", **cameras_data)

        del pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode  # type: ignore

        gs_device = device
        if hasattr(da3_model, "model"):
            try:
                gs_device = next(da3_model.model.parameters()).device
            except StopIteration:
                gs_device = device

        gaussians = _load_gaussian_ply_to_gaussians(str(final_ply_path), device=gs_device)
        render_extr = torch.from_numpy(w2c_render).to(device=gs_device, dtype=gaussians.means.dtype)[None]
        render_intr = torch.from_numpy(k_render).to(device=gs_device, dtype=gaussians.means.dtype)[None]
        if render_extr.shape[-2:] == (3, 4):
            pad = torch.tensor([0, 0, 0, 1], device=gs_device, dtype=gaussians.means.dtype).view(1, 1, 1, 4)
            render_extr = torch.cat(
                [render_extr, pad.expand(render_extr.shape[0], render_extr.shape[1], -1, -1)],
                dim=-2,
            )

        render_h, render_w = images_all[0].shape[:2]
        render_fps = float(args.render_fps) if args.render_fps is not None else float(max(1, round(eff_fps)))
        print(
            f"[vipe_da3_gs] Rendering {render_extr.shape[1]} frames at {render_h}x{render_w} "
            f"(fps={render_fps:.2f}, chunk_size={args.render_chunk_size})..."
        )
        color, depth = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=render_extr,
            intrinsics=render_intr,
            image_shape=(render_h, render_w),
            chunk_size=int(args.render_chunk_size),
            trj_mode="original",
            use_sh=True,
            color_mode="RGB+ED",
            enable_tqdm=True,
        )

        frames_render = (
            color[0].clamp(0.0, 1.0).mul(255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
        )
        video_path = output_dir / "gs_trajectory.mp4"
        _save_video_mp4(str(video_path), frames_render, fps=render_fps)
        print(f"[vipe_da3_gs] Saved GS render video to {video_path}")

        del gaussians, render_extr, render_intr, color, depth, frames_render
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    done_marker.write_text("done\n")
    print("[vipe_da3_gs] Done.")


if __name__ == "__main__":
    main()
