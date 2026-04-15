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

from typing import Optional, Tuple
import numpy as np
import torch
import os
import torch.nn.functional as F
try:
    import warp as wp
except ImportError:
    raise ImportError("NVIDIA Warp is required for ray-triangle intersection")

_warp_initialized = False
_ray_triangle_intersection_func = None

def _init_warp():
    global _warp_initialized, _ray_triangle_intersection_func
    
    if not _warp_initialized:
        print(f"Initializing Warp library (local_rank {os.getenv('LOCAL_RANK')})...")
        wp.init()
        _warp_initialized = True
        print(f"Warp library initialized successfully (local_rank {os.getenv('LOCAL_RANK')})")
    
    if _ray_triangle_intersection_func is None:
        try:
            from .ray_triangle_intersection_warp import ray_triangle_intersection_warp
            _ray_triangle_intersection_func = ray_triangle_intersection_warp
            print(f"Warp: ray_triangle_intersection_warp kernel loaded (local_rank {os.getenv('LOCAL_RANK')})")
        except ImportError:
            from ray_triangle_intersection_warp import ray_triangle_intersection_warp
            _ray_triangle_intersection_func = ray_triangle_intersection_warp
            print(f"Warp: ray_triangle_intersection_warp kernel loaded (local_rank {os.getenv('LOCAL_RANK')})")


def points_to_mesh(points, mask, resolution=None):
    """
    Convert a grid of 3D points to a triangle mesh based on mask.
    
    Args:
        points: Tensor of shape [H, W, 3] containing 3D points
        mask: Tensor of shape [H, W] containing binary mask
        resolution: Optional tuple (new_H, new_W) to resize to
    
    Returns:
        vertices: Tensor of shape [N, 3] containing unique vertices
        faces: Tensor of shape [M, 3] containing triangle indices
    """
    H, W = points.shape[:2]
    
    # Resize if resolution is provided
    if resolution is not None:
        new_H, new_W = resolution
        # Resize points using bilinear interpolation
        points = points.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        points = F.interpolate(points, size=(new_H, new_W), mode='bilinear', align_corners=False)
        points = points.squeeze(0).permute(1, 2, 0)  # [new_H, new_W, 3]
        
        # Resize mask using nearest neighbor
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        mask = F.interpolate(mask, size=(new_H, new_W), mode='nearest')
        mask = mask.squeeze(0).squeeze(0).bool()  # [new_H, new_W]
        
        H, W = new_H, new_W
    
    # Create vertex indices grid
    vertex_indices = torch.arange(H * W, device=points.device).reshape(H, W)
    
    # Find 2x2 patches where at least one vertex is in the mask
    # Create shifted views for efficient neighbor checking
    mask_tl = mask[:-1, :-1]  # top-left
    mask_tr = mask[:-1, 1:]   # top-right
    mask_bl = mask[1:, :-1]   # bottom-left
    mask_br = mask[1:, 1:]    # bottom-right
    
    # A patch is valid if any of its 4 vertices is in the mask
    valid_patches = mask_tl | mask_tr | mask_bl | mask_br  # [H-1, W-1]
    
    # Get indices of valid patches
    valid_h, valid_w = torch.where(valid_patches)
    
    # For each valid patch, create two triangles
    # Triangle 1: (u,v), (u,v+1), (u+1,v)
    # Triangle 2: (u,v+1), (u+1,v+1), (u+1,v)
    n_valid = len(valid_h)
    
    if n_valid == 0:
        # No valid patches, return empty mesh
        return torch.empty((0, 3), device=points.device), torch.empty((0, 3), dtype=torch.long, device=points.device)
    
    # Vectorized triangle creation
    idx_tl = vertex_indices[valid_h, valid_w]        # top-left
    idx_tr = vertex_indices[valid_h, valid_w + 1]    # top-right
    idx_bl = vertex_indices[valid_h + 1, valid_w]    # bottom-left
    idx_br = vertex_indices[valid_h + 1, valid_w + 1]  # bottom-right
    
    # Create faces (2 triangles per patch)
    faces1 = torch.stack([idx_tl, idx_tr, idx_bl], dim=1)  # [n_valid, 3]
    faces2 = torch.stack([idx_tr, idx_br, idx_bl], dim=1)  # [n_valid, 3]
    faces = torch.cat([faces1, faces2], dim=0)  # [2*n_valid, 3]
    
    # Flatten points to get vertices
    vertices = points.reshape(-1, 3)  # [H*W, 3]
    
    # Optional: Remove unused vertices and remap faces
    # First, find which vertices are actually used
    used_vertices = torch.unique(faces.flatten())
    
    # Create a mapping from old indices to new indices
    new_idx_map = torch.full((H * W,), -1, dtype=torch.long, device=points.device)
    new_idx_map[used_vertices] = torch.arange(len(used_vertices), device=points.device)
    
    # Extract only used vertices
    vertices = vertices[used_vertices]
    
    # Remap face indices
    faces = new_idx_map[faces.flatten()].reshape(-1, 3)
    
    return vertices, faces

def get_max_exponent_for_dtype(dtype):
    # Set the maximum exponent based on dtype
    if dtype == torch.bfloat16:
        return 80.0  # Safe maximum exponent for bfloat16
    elif dtype == torch.float16:
        return 10.0  # Safe maximum exponent for float16
    elif dtype == torch.float32:
        return 80.0  # Safe maximum exponent for float32
    elif dtype == torch.float64:
        return 700.0  # Safe maximum exponent for float64
    else:
        return 80.0  # Default safe value

def inverse_with_conversion(mtx):
    return torch.linalg.inv(mtx.to(torch.float32)).to(mtx.dtype)


def get_camera_rays(h, w, intrinsic: np.ndarray) -> np.ndarray:
    """Backproject 2D pixels into 3D rays."""
    device = intrinsic.device
    x1d = torch.arange(0, w, device=device, dtype=intrinsic.dtype)[None]
    y1d = torch.arange(0, h, device=device, dtype=intrinsic.dtype)[:, None]
    x2d = x1d.repeat([h, 1])  # .to(intrinsic)  # (h, w)
    y2d = y1d.repeat([1, w])  # .to(intrinsic)  # (h, w)
    ones_2d = torch.ones(size=(h, w), device=device, dtype=intrinsic.dtype)  # .to(intrinsic)  # (h, w)
    pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

    intrinsic1_inv = inverse_with_conversion(intrinsic)  # (b, 3, 3)
    intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
    # Normalize the rays
    unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo).squeeze(-1)
    # Normalize the rays
    norm = torch.norm(unnormalized_pos, dim=-1, keepdim=True)
    norm[norm == 0] = 1
    return unnormalized_pos / norm


def forward_warp(
    frame1: torch.Tensor,
    mask1: Optional[torch.Tensor],
    depth1: Optional[torch.Tensor],
    transformation1: Optional[torch.Tensor],
    transformation2: torch.Tensor,
    intrinsic1: Optional[torch.Tensor],
    intrinsic2: Optional[torch.Tensor],
    is_image=True,
    conditioned_normal1=None,
    cameraray_filtering=False,
    is_depth=True,
    render_depth=False,
    world_points1=None,
    foreground_masking=False,
    boundary_mask=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
    bilinear splatting.
    All arrays should be torch tensors with batch dimension and channel first
    :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                    bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                    method accordingly.
    :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
    :param depth1: (b, 1, h, w)
    :param transformation1: (b, 4, 4) extrinsic transformation matrix (camera-to-world pose) of first view. Required if depth1 is not None, or if cleaning is enabled.
    :param transformation2: (b, 4, 4) extrinsic transformation matrix (camera-to-world pose) of second view.
    :param intrinsic1: (b, 3, 3) camera intrinsic matrix. Required if depth1 is not None.
    :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional (defaults to intrinsic1 if provided).
    :param is_image: bool, whether frame1 represents image data (affects clipping and fill value).
    :param conditioned_normal1: Optional (b, 3, h, w) normals for filtering.
    :param cameraray_filtering: bool, use camera rays for filtering instead of normals.
    :param is_depth: bool, whether depth1 represents depth along Z or distance to camera center. Used only if depth1 is not None.
    :param render_depth: bool, whether to also render and return the warped depth map.
    :param world_points1: Optional (b, h, w, 3) world points. Required if depth1 is None.
    :param foreground_masking: bool, enable foreground occlusion masking using mesh rendering.
    :param boundary_mask: Optional (b, h, w) mask for mesh generation, required if foreground_masking is True.
    """
    device = frame1.device
    b, c, h, w = frame1.shape
    dtype = frame1.dtype
    if mask1 is None:
        mask1 = torch.ones(size=(b, 1, h, w), device=device, dtype=frame1.dtype)
    if intrinsic2 is None:
        assert intrinsic1 is not None, "intrinsic2 cannot be derived if intrinsic1 is None and intrinsic2 is None"
        intrinsic2 = intrinsic1.clone()

    if depth1 is None:
        assert world_points1.shape == (b, h, w, 3)
        if foreground_masking:
            trans_points1, cam_points_target = project_points(world_points1, transformation2, intrinsic2, return_cam_points=True)
        else:
            trans_points1 = project_points(world_points1, transformation2, intrinsic2)
    else:
        # assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        depth1 = torch.nan_to_num(depth1, nan=1e4)
        depth1 = torch.clamp(depth1, min=0, max=1e4)
        if foreground_masking:
            trans_points1, cam_points_target = compute_transformed_points(
                depth1, transformation1, transformation2, intrinsic1, is_depth, intrinsic2, return_cam_points=True
            )
        else:
            trans_points1 = compute_transformed_points(
                depth1, transformation1, transformation2, intrinsic1, is_depth, intrinsic2
            )
    mask1 = mask1 * (trans_points1[:, :, :, 2, 0].unsqueeze(1) > 0)
    trans_coordinates = trans_points1[:, :, :, :2, 0] / (trans_points1[:, :, :, 2:3, 0] + 1e-7)
    trans_coordinates = trans_coordinates.permute(0, 3, 1, 2)  # b, 2, h, w
    trans_depth1 = trans_points1[:, :, :, 2, 0].unsqueeze(1)

    grid = create_grid(b, h, w, device=device, dtype=dtype)  # .to(trans_coordinates)
    flow12 = trans_coordinates - grid
    if conditioned_normal1 is not None or cameraray_filtering:
        camera_rays = get_camera_rays(h, w, intrinsic1)  # b, h, w, 3
        transformation = torch.bmm(transformation2, inverse_with_conversion(transformation1))
        transformation[:, :3, 3] = 0
        trans_4d = transformation[:, None, None]
        if cameraray_filtering:  # use normal for filtering
            conditioned_normal1 = camera_rays
            inversion_vector = torch.tensor([-1, -1, -1], dtype=camera_rays.dtype, device=camera_rays.device).view(
                1, 1, 1, 3, 1
            )
        else:  # use normal for filtering
            assert conditioned_normal1.shape == (b, 3, h, w)
            inversion_vector = torch.tensor([-1, 1, 1], dtype=camera_rays.dtype, device=camera_rays.device).view(
                1, 1, 1, 3, 1
            )
            conditioned_normal1 = conditioned_normal1.permute(0, 2, 3, 1)
        # rotate normal into target camera spaces
        normal_4d = conditioned_normal1.unsqueeze(-1)
        b, _, h, w = depth1.shape
        ones_2d = torch.ones(size=(h, w), device=device, dtype=dtype)  # .to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])
        normal_4d_homo = torch.cat([normal_4d * inversion_vector, ones_4d], dim=3)

        trans_normal = torch.matmul(trans_4d, normal_4d_homo).squeeze(-1)[..., :3]  # b, h, w, 3
        dot_product = torch.sum(trans_normal * camera_rays, dim=-1)

        # Create binary mask for angles < 90 degrees
        binary_mask = dot_product > 0
        # import ipdb;ipdb.set_trace()
        mask1 *= binary_mask.unsqueeze(1)
    warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=is_image)
    warped_depth2 = None
    if render_depth or foreground_masking:
        warped_depth2 = bilinear_splatting(trans_depth1, mask1, trans_depth1, flow12, None, is_image=False)[0][:, 0]
    if foreground_masking:
        for batch_idx in range(b):
            assert boundary_mask is not None
            mesh_mask = boundary_mask[batch_idx]
            
            mesh_downsample_factor = 4
            vertices_masked, faces_masked = points_to_mesh(
                cam_points_target[batch_idx], 
                mesh_mask, 
                resolution=(h // mesh_downsample_factor, w // mesh_downsample_factor)
            )
            
            if vertices_masked.shape[0] == 0 or faces_masked.shape[0] == 0:
                continue
            
            ray_scale_factor = 1
            ray_downsampled_h = h // ray_scale_factor
            ray_downsampled_w = w // ray_scale_factor
            current_intrinsic_batch = intrinsic2[batch_idx:batch_idx+1]
            scaled_intrinsic = current_intrinsic_batch.clone()
            
            scaled_intrinsic[0, 0, 0] /= ray_scale_factor  # fx
            scaled_intrinsic[0, 1, 1] /= ray_scale_factor  # fy
            scaled_intrinsic[0, 0, 2] /= ray_scale_factor  # cx
            scaled_intrinsic[0, 1, 2] /= ray_scale_factor  # cy
            
            camera_rays = get_camera_rays(ray_downsampled_h, ray_downsampled_w, scaled_intrinsic)  # (1, h_ds, w_ds, 3)
            camera_rays = camera_rays[0]  # (h_ds, w_ds, 3)
            
            ray_origins = torch.zeros((ray_downsampled_h, ray_downsampled_w, 3), device=device, dtype=dtype)
            
            mesh_depth = ray_triangle_intersection(
                ray_origins,
                camera_rays,
                vertices_masked,
                faces_masked,
                device
            ) 
            ray_z = camera_rays[:, :, 2]  # (h, w)
            mesh_z_depth = mesh_depth * ray_z  # Convert to z-depth
            mesh_z_depth = F.interpolate(mesh_z_depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0).squeeze(0)
            
            warped_depth_batch = warped_depth2[batch_idx]  # (h, w)

            
            mesh_valid = mesh_z_depth > 0
            mesh_closer = ((mesh_z_depth + 0.02) < warped_depth_batch) & mesh_valid
            
            mask2[batch_idx, 0] = mask2[batch_idx, 0] * (~mesh_closer).float()
            warped_frame2[batch_idx] = (warped_frame2[batch_idx] + 1) * (~mesh_closer.unsqueeze(0)).float() - 1
            warped_depth2[batch_idx] = warped_depth2[batch_idx] * (~mesh_closer.unsqueeze(0)).float()
    return warped_frame2, mask2, warped_depth2, flow12

def reliable_depth_mask_range_batch(depth, window_size=5, ratio_thresh=0.05, eps=1e-6):
    assert window_size % 2 == 1, "Window size must be odd."
    if depth.dim() == 3:   # Input shape: (B, H, W)
        depth_unsq = depth.unsqueeze(1)
    elif depth.dim() == 4:  # Already has shape (B, 1, H, W)
        depth_unsq = depth
    else:
        raise ValueError("depth tensor must be of shape (B, H, W) or (B, 1, H, W)")
    
    local_max = torch.nn.functional.max_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_min = -torch.nn.functional.max_pool2d(-depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_mean = torch.nn.functional.avg_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    ratio = (local_max - local_min) / (local_mean + eps)
    reliable_mask = (ratio < ratio_thresh) & (depth_unsq > 0)
    
    return reliable_mask

def double_forward_warp(
    frame1: torch.Tensor,
    mask1: torch.Tensor,
    depth1: torch.Tensor,
    intrinsic1: torch.Tensor,
    double_proj_w2cs: torch.Tensor,
):
    """
    Double projection using forward warping with your APIs.
    
    1. Warps frame1 from the original view (identity transformation)
       to the target view defined by double_proj_w2cs.
    2. Computes a warped flow field and then warps the intermediate result
       back to the original view using the original depth.
       
    :param frame1: (b, 3, h, w) original image.
    :param mask1: (b, 1, h, w) valid mask.
    :param depth1: (b, 1, h, w) depth map.
    :param intrinsic1: (b, 3, 3) intrinsic matrix.
    :param double_proj_w2cs: (b, 4, 4) target view transformation.
    :return: twice_warped_frame1, warped_frame2, None, None
    """
    b, c, h, w = frame1.shape
    device, dtype = frame1.device, frame1.dtype

    if mask1 is None:
        mask1 = torch.ones((b, 1, h, w), device=device, dtype=dtype)

    # Use identity transformation for the original view.
    identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)

    trans_points = compute_transformed_points(
        depth1, identity, double_proj_w2cs, intrinsic1, is_depth=True, intrinsic2=intrinsic1
    )
    trans_coordinates = trans_points[:, :, :, :2, 0] / (trans_points[:, :, :, 2:3, 0] + 1e-7)
    trans_depth = trans_points[:, :, :, 2, 0]

    grid = create_grid(b, h, w, device=device, dtype=dtype)
    flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

    warped_frame2, mask2 = bilinear_splatting(
        frame1, mask1, trans_depth.unsqueeze(1), flow12, None, is_image=True, n_views=1, depth_weight_scale=50
    )

    warped_flow, _ = bilinear_splatting(
        flow12, mask1, trans_depth.unsqueeze(1), flow12, None, is_image=False, n_views=1, depth_weight_scale=50
    )

    twice_warped_frame1, twice_warped_mask1 = bilinear_splatting(
        warped_frame2, mask2, depth1, -warped_flow, None, is_image=True, n_views=1, depth_weight_scale=50
    )

    return twice_warped_frame1, twice_warped_mask1, warped_frame2, mask2


def unproject_points(depth: torch.Tensor, 
                     w2c: torch.Tensor, 
                     intrinsic: torch.Tensor, 
                     is_depth: bool = True, 
                     mask: Optional[torch.Tensor] = None):

    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype
    if mask is None:
        mask = depth > 0
    if mask.dim() == depth.dim() and mask.shape[1] == 1:
        mask = mask[:, 0]

    idx = torch.nonzero(mask)
    if idx.numel() == 0:
        return torch.zeros((b, h, w, 3), device=device, dtype=dtype)

    b_idx, y_idx, x_idx = idx[:, 0], idx[:, 1], idx[:, 2]


    intrinsic_inv = inverse_with_conversion(intrinsic)  # (b, 3, 3)

    x_valid = x_idx.to(dtype)
    y_valid = y_idx.to(dtype)
    ones = torch.ones_like(x_valid)
    pos = torch.stack([x_valid, y_valid, ones], dim=1).unsqueeze(-1)  # (N, 3, 1)

    intrinsic_inv_valid = intrinsic_inv[b_idx]  # (N, 3, 3)
    unnormalized_pos = torch.matmul(intrinsic_inv_valid, pos)  # (N, 3, 1)

    depth_valid = depth[b_idx, 0, y_idx, x_idx].view(-1, 1, 1)
    if is_depth:
        world_points_cam = depth_valid * unnormalized_pos
    else:
        norm_val = torch.norm(unnormalized_pos, dim=1, keepdim=True)
        direction = unnormalized_pos / (norm_val + 1e-8)
        world_points_cam = depth_valid * direction

    ones_h = torch.ones((world_points_cam.shape[0], 1, 1), 
                        device=device, dtype=dtype)
    world_points_homo = torch.cat([world_points_cam, ones_h], dim=1)  # (N, 4, 1)

    trans = inverse_with_conversion(w2c)  # (b, 4, 4)
    trans_valid = trans[b_idx]  # (N, 4, 4)
    world_points_transformed = torch.matmul(trans_valid, world_points_homo)  # (N, 4, 1)
    sparse_points = world_points_transformed[:, :3, 0]  # (N, 3)

    out_points = torch.zeros((b, h, w, 3), device=device, dtype=dtype)
    out_points[b_idx, y_idx, x_idx, :] = sparse_points
    return out_points

def project_points(world_points: torch.Tensor, w2c: torch.Tensor, intrinsic: torch.Tensor, return_cam_points: bool = False):
    """
    Projects 3D world points back into 2D pixel space.
    """
    world_points = world_points.unsqueeze(-1)  # (b, h, w, 3) -> # (b, h, w, 3, 1)
    b, h, w, _, _ = world_points.shape

    ones_4d = torch.ones((b, h, w, 1, 1), device=world_points.device, dtype=world_points.dtype)  # (b, h, w, 1, 1)
    world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)

    # Apply transformation2 to convert world points to camera space
    trans_4d = w2c[:, None, None]  # (b, 1, 1, 4, 4)
    camera_points_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)

    # Remove homogeneous coordinate and project to image plane
    camera_points = camera_points_homo[:, :, :, :3]  # (b, h, w, 3, 1)
    intrinsic_4d = intrinsic[:, None, None]  # (b, 1, 1, 3, 3)
    projected_points = torch.matmul(intrinsic_4d, camera_points)  # (b, h, w, 3, 1)

    if return_cam_points:
        # Return both projected points and camera space points
        cam_points_3d = camera_points.squeeze(-1)  # (b, h, w, 3)
        return projected_points, cam_points_3d
    else:
        return projected_points


def unproject_depth_torch(
    depth1: torch.Tensor,
    transformation1: torch.Tensor,
    intrinsic1: torch.Tensor,
) -> torch.Tensor:
    b, c, h, w = depth1.shape
    assert depth1.shape == (b, 1, h, w)
    assert transformation1.shape == (b, 4, 4)
    assert intrinsic1.shape == (b, 3, 3)
    device = depth1.device
    x1d = torch.arange(0, w, device=device)[None]
    y1d = torch.arange(0, h, device=device)[:, None]
    x2d = x1d.repeat([h, 1])  # .to(depth1)  # (h, w)
    y2d = y1d.repeat([1, w])  # .to(depth1)  # (h, w)
    ones_2d = torch.ones(size=(h, w), device=device)  # .to(depth1)  # (h, w)
    ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
    pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

    intrinsic1_inv = inverse_with_conversion(intrinsic1)  # (b, 3, 3)
    intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)

    depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)

    unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
    world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)

    world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
    trans_4d = transformation1[:, None, None]  # (b, 1, 1, 4, 4)
    trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
    trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
    trans_world = trans_world.squeeze(dim=-1)
    return trans_world


def compute_transformed_points(
    depth1: torch.Tensor,
    transformation1: torch.Tensor,
    transformation2: torch.Tensor,
    intrinsic1: torch.Tensor,
    is_depth: bool = True,
    intrinsic2: Optional[torch.Tensor] = None,
    return_cam_points: bool = False,
):
    """
    Computes transformed position for each pixel location
    """
    b, _, h, w = depth1.shape
    if intrinsic2 is None:
        intrinsic2 = intrinsic1.clone()
    transformation = torch.bmm(
        transformation2, inverse_with_conversion(transformation1)
    )  # (b, 4, 4) transformation is w2c
    device = depth1.device
    x1d = torch.arange(0, w, device=device, dtype=depth1.dtype)[None]
    y1d = torch.arange(0, h, device=device, dtype=depth1.dtype)[:, None]
    x2d = x1d.repeat([h, 1])  # .to(depth1)  # (h, w)
    y2d = y1d.repeat([1, w])  # .to(depth1)  # (h, w)
    ones_2d = torch.ones(size=(h, w), device=device, dtype=depth1.dtype)  # .to(depth1)  # (h, w)
    ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
    pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

    intrinsic1_inv = inverse_with_conversion(intrinsic1)  # (b, 3, 3)
    intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
    intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
    depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
    trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

    unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
    if is_depth:
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
    else:  # if 'depth' is defined as distance to camera center
        direction_vectors = unnormalized_pos / torch.norm(unnormalized_pos, dim=-2, keepdim=True)  # (b, h, w, 3, 1)
        world_points = depth_4d * direction_vectors  # (b, h, w, 3, 1)

    world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
    trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
    trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
    trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
    
    if return_cam_points:
        # Return both projected points and camera space points
        cam_points = trans_world.squeeze(-1)  # (b, h, w, 3)
        return trans_norm_points, cam_points
    else:
        return trans_norm_points


def bilinear_splatting(
    frame1: torch.Tensor,
    mask1: Optional[torch.Tensor],
    depth1: torch.Tensor,
    flow12: torch.Tensor,
    flow12_mask: Optional[torch.Tensor],
    is_image: bool = False,
    n_views=1,
    depth_weight_scale=50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear splatting
    :param frame1: (b,c,h,w)
    :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
    :param depth1: (b,1,h,w)
    :param flow12: (b,2,h,w)
    :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
    :param is_image: if true, output will be clipped to (-1,1) range
    :return: warped_frame2: (b,c,h,w)
             mask2: (b,1,h,w): 1 for known and 0 for unknown
    """
    b, c, h, w = frame1.shape
    device = frame1.device
    dtype = frame1.dtype
    if mask1 is None:
        mask1 = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)  # .to(frame1)
    if flow12_mask is None:
        flow12_mask = torch.ones(size=(b, 1, h, w), device=device, dtype=dtype)  # .to(flow12)
    grid = create_grid(b, h, w, device=device, dtype=dtype).to(dtype)  # .to(frame1)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()
    trans_pos_offset = torch.stack(
        [torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1), torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)],
        dim=1,
    )
    trans_pos_floor = torch.stack(
        [torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1), torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)],
        dim=1,
    )
    trans_pos_ceil = torch.stack(
        [torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1), torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)],
        dim=1,
    )

    prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
        1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
    )
    prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
        1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
    )
    prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
        1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
    )
    prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
        1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
    )

    # Calculate depth weights, preventing overflow and removing saturation
    # Clamp depth to be non-negative before log1p
    clamped_depth1 = torch.clamp(depth1, min=0)
    log_depth1 = torch.log1p(clamped_depth1) # Use log1p for better precision near 0
    # Normalize and scale log depth
    exponent = log_depth1 / (log_depth1.max() + 1e-7) * depth_weight_scale
    # Clamp exponent before exp to prevent overflow
    max_exponent = get_max_exponent_for_dtype(depth1.dtype)
    clamped_exponent = torch.clamp(exponent, max=max_exponent)
    # Compute depth weights with added epsilon for stability when dividing later
    depth_weights = torch.exp(clamped_exponent) + 1e-7


    weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
    weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

    warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=dtype, device=device)  # .to(frame1)
    warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=dtype, device=device)  # .to(frame1)

    frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
    batch_indices = torch.arange(b, device=device, dtype=torch.long)[:, None, None]  # .to(frame1.device)
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_nw, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), frame1_cl * weight_sw, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_ne, accumulate=True
    )
    warped_frame.index_put_(
        (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), frame1_cl * weight_se, accumulate=True
    )

    warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]), weight_nw, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]), weight_sw, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]), weight_ne, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]), weight_se, accumulate=True)
    if n_views > 1:
        warped_frame = warped_frame.reshape(b // n_views, n_views, h + 2, w + 2, c).sum(1)
        warped_weights = warped_weights.reshape(b // n_views, n_views, h + 2, w + 2, 1).sum(1)

    warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
    warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
    cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
    cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]
    cropped_weights = torch.nan_to_num(cropped_weights, nan=1000.0)

    mask = cropped_weights > 0
    zero_value = -1 if is_image else 0
    zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
    warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
    mask2 = mask.to(frame1)
    if is_image:
        # assert warped_frame2.min() >= -1.1  # Allow for rounding errors
        # assert warped_frame2.max() <= 1.1
        warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
    return warped_frame2, mask2

def create_grid(b: int, h: int, w: int, device="cpu", dtype=torch.float) -> torch.Tensor:
    """
    Create a dense grid of (x,y) coordinates of shape (b, 2, h, w).
    """
    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, y], dim=1)

def ray_triangle_intersection(
    ray_origins: torch.Tensor,  # (H, W, 3)
    ray_directions: torch.Tensor,  # (H, W, 3)
    vertices: torch.Tensor,  # (N, 3)
    faces: torch.Tensor,  # (M, 3)
    device: torch.device
) -> torch.Tensor:
    """
    Compute ray-triangle intersections for all rays and triangles.
    Returns depth map of shape (H, W) with intersection distances.
    
    Uses NVIDIA Warp acceleration for fast performance.
    """
    _init_warp()
    return _ray_triangle_intersection_func(
        ray_origins, ray_directions, vertices, faces, device
    )