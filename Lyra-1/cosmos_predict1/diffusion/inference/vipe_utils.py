import os
import zipfile
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _center_crop(tensor_bchw: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    _, _, h, w = tensor_bchw.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return tensor_bchw[:, :, top : top + crop_h, left : left + crop_w]


def _adjust_intrinsics_for_resize_and_crop(
    intrinsics_3x3: np.ndarray,
    src_hw: Tuple[int, int],
    resize_hw: Tuple[int, int],
    crop_hw: Tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_hw
    resize_h, resize_w = resize_hw
    crop_h, crop_w = crop_hw

    K = intrinsics_3x3.copy()

    sx = resize_w / float(src_w)
    sy = resize_h / float(src_h)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy

    off_x = max((resize_w - crop_w) // 2, 0)
    off_y = max((resize_h - crop_h) // 2, 0)
    K[0, 2] -= off_x
    K[1, 2] -= off_y

    return K


def _intrinsics_from_fxfycxcy(fxfycxcy: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in fxfycxcy]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _load_pose_matrix_for_frame(pose_npz_path: str, frame_idx: int) -> np.ndarray:
    data = np.load(pose_npz_path)
    inds = data["inds"]
    arr = data["data"]
    pos = int(np.searchsorted(inds, frame_idx))
    if not (0 <= pos < len(inds)) or int(inds[pos]) != int(frame_idx):
        raise FileNotFoundError(f"Pose for frame {frame_idx} not found in {pose_npz_path}")
    mat = arr[pos]
    if mat.shape == (16,):
        mat = mat.reshape(4, 4)
    assert mat.shape == (4, 4)
    return mat.astype(np.float32)


def _load_intrinsics_for_frame(intrinsics_npz_path: str, frame_idx: int) -> np.ndarray:
    data = np.load(intrinsics_npz_path)
    inds = data["inds"]
    arr = data["data"]
    pos = int(np.searchsorted(inds, frame_idx))
    if not (0 <= pos < len(inds)) or int(inds[pos]) != int(frame_idx):
        raise FileNotFoundError(
            f"Intrinsics for frame {frame_idx} not found in {intrinsics_npz_path}"
        )
    item = arr[pos]
    if item.shape == (3, 3):
        K = item.astype(np.float32)
    elif item.shape[-1] == 4:
        K = _intrinsics_from_fxfycxcy(item)
    else:
        raise ValueError(
            f"Unsupported intrinsics format {item.shape} in {intrinsics_npz_path}"
        )
    return K


def _read_depth_from_zip(zip_path: str, frame_idx: int) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore
    except ImportError as e:
        raise ImportError("OpenEXR package is required to read VIPE depth EXR files") from e

    fname = f"{frame_idx:05d}.exr"
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(fname, "r") as f:
            exr = OpenEXR.InputFile(f)
            dw = exr.header()["dataWindow"]
            height = dw.max.y - dw.min.y + 1
            width = dw.max.x - dw.min.x + 1
            depth = np.frombuffer(exr.channel("Z"), np.float16).astype(np.float32)
            depth = depth.reshape(height, width)
    return depth


def _read_mask_from_zip(zip_path: str, frame_idx: int) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    fname = f"{frame_idx:05d}.png"
    if not os.path.exists(zip_path):
        return None
    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            with zf.open(fname, "r") as f:
                buf = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except KeyError:
            return None
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    mask = (img > 0).astype(np.float32)
    return mask


def _read_mp4_frame(mp4_path: str, frame_idx: int) -> np.ndarray:
    try:
        from decord import VideoReader  # type: ignore
    except ImportError as e:
        raise ImportError("decord is required to read VIPE rgb mp4") from e
    vr = VideoReader(mp4_path, num_threads=4)
    if frame_idx < 0 or frame_idx >= len(vr):
        raise IndexError(
            f"Requested frame_idx {frame_idx} is out of bounds for video length {len(vr)}"
        )
    frame = vr.get_batch([frame_idx])
    try:
        frame_np = frame.asnumpy()
    except AttributeError:
        frame_np = frame.numpy()
    frame_np = frame_np[0]  # (H, W, C)
    return frame_np


def _find_clip_paths(vipe_root_or_mp4: str, video_idx: int = 0) -> Tuple[str, str, str, str, Optional[str]]:
    if vipe_root_or_mp4.endswith(".mp4"):
        mp4_path = vipe_root_or_mp4
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        root = os.path.dirname(os.path.dirname(mp4_path))
    else:
        rgb_dir = os.path.join(vipe_root_or_mp4, "rgb")
        mp4_files = [
            os.path.join(rgb_dir, f)
            for f in sorted(os.listdir(rgb_dir))
            if f.endswith(".mp4")
        ]
        if len(mp4_files) == 0:
            raise FileNotFoundError(f"No mp4 found under {rgb_dir}")
        mp4_path = mp4_files[video_idx]
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        root = vipe_root_or_mp4

    depth_zip = os.path.join(root, "depth", f"{base}.zip")
    pose_npz = os.path.join(root, "pose", f"{base}.npz")
    intr_npz = os.path.join(root, "intrinsics", f"{base}.npz")
    mask_zip = os.path.join(root, "mask", f"{base}.zip")
    if not os.path.exists(mask_zip):
        mask_zip = None
    return mp4_path, depth_zip, pose_npz, intr_npz, mask_zip


def load_vipe_data(
    vipe_root_or_mp4: str,
    starting_frame_idx: int,
    resize_hw: Tuple[int, int] = (720, 1280),
    crop_hw: Tuple[int, int] = (704, 1280),
    num_frames: int = 121,
    read_mask: bool = False,
    video_idx: int = 0,
):
    mp4_path, depth_zip, pose_npz, intr_npz, mask_zip = _find_clip_paths(vipe_root_or_mp4, video_idx=video_idx)

    # Read the sequence of RGB frames
    try:
        from decord import VideoReader  # type: ignore
    except ImportError as e:
        raise ImportError("decord is required to read VIPE rgb mp4") from e
    vr = VideoReader(mp4_path, num_threads=4)
    total_len = len(vr)
    # If starting index is beyond the video, clamp to last frame
    if starting_frame_idx >= total_len:
        starting_frame_idx = max(0, total_len - 1)
    last_available_idx = total_len - 1
    # Build index list and repeat the last available frame if not enough
    frame_indices = list(range(starting_frame_idx, min(starting_frame_idx + num_frames, total_len)))
    while len(frame_indices) < num_frames:
        frame_indices.append(last_available_idx)
    batch = vr.get_batch(frame_indices)
    try:
        frames_np = batch.asnumpy()
    except AttributeError:
        frames_np = batch.numpy()
    # frames_np: (T, H, W, C) in [0,255]
    frames_np = frames_np.astype(np.float32) / 255.0
    src_h, src_w = frames_np.shape[1], frames_np.shape[2]

    # Load per-frame pose (c2w) and intrinsics, convert and adjust K
    w2cs_list = []
    Ks_list = []
    for fidx in frame_indices:
        c2w_44 = _load_pose_matrix_for_frame(pose_npz, fidx)
        w2c_44 = np.linalg.inv(c2w_44).astype(np.float32)
        w2cs_list.append(w2c_44)

        K_src = _load_intrinsics_for_frame(intr_npz, fidx)
        K_adj = _adjust_intrinsics_for_resize_and_crop(K_src, (src_h, src_w), resize_hw, crop_hw)
        Ks_list.append(K_adj)

    w2cs_np = np.stack(w2cs_list, axis=0)  # (T, 4, 4)
    Ks_np = np.stack(Ks_list, axis=0)      # (T, 3, 3)

    # Depth/mask for the whole sequence
    depth_list = []
    mask_list = []
    for fidx in frame_indices:
        d_hw = _read_depth_from_zip(depth_zip, fidx)
        depth_list.append(d_hw)
        if read_mask and mask_zip:
            m_hw = _read_mask_from_zip(mask_zip, fidx)
        else:
            m_hw = None
        mask_list.append(m_hw)

    # Convert to torch and apply resize/crop
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    depth_seq = torch.from_numpy(np.stack(depth_list, axis=0)).unsqueeze(1).contiguous()  # (T,1,H,W)
    mask_seq_np = []
    for m in mask_list:
        if m is None:
            mask_seq_np.append(np.ones((src_h, src_w), dtype=np.float32))
        else:
            mask_seq_np.append(m.astype(np.float32))
    mask_seq = torch.from_numpy(np.stack(mask_seq_np, axis=0)).unsqueeze(1).contiguous()  # (T,1,H,W)

    rh, rw = resize_hw
    ch, cw = crop_hw

    frames_t = F.interpolate(frames_t, size=(rh, rw), mode="bilinear", align_corners=False)
    depth_seq = F.interpolate(depth_seq, size=(rh, rw), mode="bilinear", align_corners=False)
    mask_seq = F.interpolate(mask_seq, size=(rh, rw), mode="nearest")

    frames_t = _center_crop(frames_t, ch, cw)  # (T, C, ch, cw)
    depth_seq = _center_crop(depth_seq, ch, cw)    # (T, 1, ch, cw)
    mask_seq = _center_crop(mask_seq, ch, cw)      # (T, 1, ch, cw)

    frames_t = frames_t * 2.0 - 1.0  # to [-1, 1]

    # Full sequences (T, ...)
    w2cs_T44 = torch.from_numpy(w2cs_np).contiguous()
    Ks_T33 = torch.from_numpy(Ks_np).contiguous()

    return (
        frames_t,
        depth_seq,
        mask_seq,
        w2cs_T44,
        Ks_T33,  
    )
