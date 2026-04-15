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

import torch
import warp as wp
import numpy as np

# Initialize Warp with CUDA
wp.init()

@wp.kernel
def ray_triangle_intersection_kernel(
    ray_origins: wp.array2d(dtype=wp.float32),      # (H*W, 3)
    ray_directions: wp.array2d(dtype=wp.float32),   # (H*W, 3)
    vertices: wp.array2d(dtype=wp.float32),         # (N, 3)
    faces: wp.array2d(dtype=wp.int32),              # (M, 3)
    depth_map: wp.array(dtype=wp.float32),          # (H*W,)
    num_triangles: wp.int32,
    epsilon: wp.float32
):
    """
    Warp kernel for ray-triangle intersection using Möller–Trumbore algorithm.
    Each thread processes one ray against all triangles.
    """
    # Get thread index (ray index)
    ray_idx = wp.tid()
    
    # Get ray origin and direction
    ray_origin = wp.vec3(
        ray_origins[ray_idx, 0],
        ray_origins[ray_idx, 1],
        ray_origins[ray_idx, 2]
    )
    ray_dir = wp.vec3(
        ray_directions[ray_idx, 0],
        ray_directions[ray_idx, 1],
        ray_directions[ray_idx, 2]
    )
    
    # Initialize minimum distance
    min_t = wp.float32(1e10)
    
    # Iterate through all triangles
    for tri_idx in range(num_triangles):
        # Get triangle vertex indices
        i0 = faces[tri_idx, 0]
        i1 = faces[tri_idx, 1]
        i2 = faces[tri_idx, 2]
        
        # Get triangle vertices
        v0 = wp.vec3(vertices[i0, 0], vertices[i0, 1], vertices[i0, 2])
        v1 = wp.vec3(vertices[i1, 0], vertices[i1, 1], vertices[i1, 2])
        v2 = wp.vec3(vertices[i2, 0], vertices[i2, 1], vertices[i2, 2])
        
        # Compute edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Möller–Trumbore algorithm
        h = wp.cross(ray_dir, edge2)
        a = wp.dot(edge1, h)
        
        # Check if ray is parallel to triangle
        if wp.abs(a) < epsilon:
            continue
            
        f = 1.0 / a
        s = ray_origin - v0
        u = f * wp.dot(s, h)
        
        # Check if intersection is within triangle (u >= 0 and u <= 1)
        if u < 0.0 or u > 1.0:
            continue
            
        q = wp.cross(s, edge1)
        v = f * wp.dot(ray_dir, q)
        
        # Check if intersection is within triangle (v >= 0 and u + v <= 1)
        if v < 0.0 or (u + v) > 1.0:
            continue
            
        # Compute t (distance along ray)
        t = f * wp.dot(edge2, q)
        
        # Only consider intersections in front of camera (t > 0)
        if t > epsilon and t < min_t:
            min_t = t
    
    # Write result
    if min_t < 1e10:
        depth_map[ray_idx] = min_t
    else:
        depth_map[ray_idx] = 0.0


@wp.kernel
def ray_triangle_intersection_tiled_kernel(
    ray_origins: wp.array2d(dtype=wp.float32),      # (H*W, 3)
    ray_directions: wp.array2d(dtype=wp.float32),   # (H*W, 3)
    vertices: wp.array2d(dtype=wp.float32),         # (N, 3)
    faces: wp.array2d(dtype=wp.int32),              # (M, 3)
    depth_map: wp.array(dtype=wp.float32),          # (H*W,)
    tri_start: wp.int32,                            # Start triangle index for this tile
    tri_end: wp.int32,                              # End triangle index for this tile
    epsilon: wp.float32
):
    """
    Tiled version of ray-triangle intersection kernel.
    Processes a subset of triangles to improve memory access patterns.
    """
    # Get thread index (ray index)
    ray_idx = wp.tid()
    
    # Get ray origin and direction
    ray_origin = wp.vec3(
        ray_origins[ray_idx, 0],
        ray_origins[ray_idx, 1],
        ray_origins[ray_idx, 2]
    )
    ray_dir = wp.vec3(
        ray_directions[ray_idx, 0],
        ray_directions[ray_idx, 1],
        ray_directions[ray_idx, 2]
    )
    
    # Get current minimum distance
    min_t = depth_map[ray_idx]
    if min_t == 0.0:
        min_t = wp.float32(1e10)
    
    # Process triangles in this tile
    for tri_idx in range(tri_start, tri_end):
        # Get triangle vertex indices
        i0 = faces[tri_idx, 0]
        i1 = faces[tri_idx, 1]
        i2 = faces[tri_idx, 2]
        
        # Get triangle vertices
        v0 = wp.vec3(vertices[i0, 0], vertices[i0, 1], vertices[i0, 2])
        v1 = wp.vec3(vertices[i1, 0], vertices[i1, 1], vertices[i1, 2])
        v2 = wp.vec3(vertices[i2, 0], vertices[i2, 1], vertices[i2, 2])
        
        # Compute edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Möller–Trumbore algorithm
        h = wp.cross(ray_dir, edge2)
        a = wp.dot(edge1, h)
        
        # Check if ray is parallel to triangle
        if wp.abs(a) < epsilon:
            continue
            
        f = 1.0 / a
        s = ray_origin - v0
        u = f * wp.dot(s, h)
        
        # Check if intersection is within triangle (u >= 0 and u <= 1)
        if u < 0.0 or u > 1.0:
            continue
            
        q = wp.cross(s, edge1)
        v = f * wp.dot(ray_dir, q)
        
        # Check if intersection is within triangle (v >= 0 and u + v <= 1)
        if v < 0.0 or (u + v) > 1.0:
            continue
            
        # Compute t (distance along ray)
        t = f * wp.dot(edge2, q)
        
        # Only consider intersections in front of camera (t > 0)
        if t > epsilon and t < min_t:
            min_t = t
    
    # Write result using atomic min to handle concurrent updates
    if min_t < 1e10:
        wp.atomic_min(depth_map, ray_idx, min_t)


def ray_triangle_intersection_warp(
    ray_origins: torch.Tensor,      # (H, W, 3)
    ray_directions: torch.Tensor,   # (H, W, 3)
    vertices: torch.Tensor,         # (N, 3)
    faces: torch.Tensor,            # (M, 3)
    device: torch.device
) -> torch.Tensor:
    """
    Compute ray-triangle intersections using NVIDIA Warp for maximum GPU acceleration.
    
    This implementation uses Warp kernels to achieve the best possible performance
    on NVIDIA GPUs by:
    1. Using native CUDA kernels through Warp
    2. Tiling triangles for better memory access patterns
    3. Using atomic operations for concurrent updates
    4. Minimizing memory transfers
    
    Args:
        ray_origins: (H, W, 3) ray origins in camera space
        ray_directions: (H, W, 3) ray directions (should be normalized)
        vertices: (N, 3) mesh vertices
        faces: (M, 3) triangle face indices
        device: torch device (must be CUDA)
    
    Returns:
        depth_map: (H, W) depth values, 0 where no intersection
    """
    H, W = ray_origins.shape[:2]
    num_rays = H * W
    num_triangles = faces.shape[0]
    
    # Reshape rays to 2D arrays
    ray_origins_flat = ray_origins.reshape(-1, 3).contiguous()
    ray_directions_flat = ray_directions.reshape(-1, 3).contiguous()
    
    # Convert PyTorch tensors to Warp arrays (as float arrays, not vec3)
    wp_ray_origins = wp.from_torch(ray_origins_flat, dtype=wp.float32)
    wp_ray_directions = wp.from_torch(ray_directions_flat, dtype=wp.float32)
    wp_vertices = wp.from_torch(vertices.contiguous(), dtype=wp.float32)
    wp_faces = wp.from_torch(faces.int().contiguous(), dtype=wp.int32)
    
    # Create output depth map
    depth_map_flat = torch.zeros(num_rays, device=device, dtype=torch.float32)
    wp_depth_map = wp.from_torch(depth_map_flat, dtype=wp.float32)
    
    # Choose implementation based on problem size
    if num_triangles < 10000:
        # For smaller meshes, use simple kernel
        wp.launch(
            kernel=ray_triangle_intersection_kernel,
            dim=num_rays,
            inputs=[
                wp_ray_origins,
                wp_ray_directions,
                wp_vertices,
                wp_faces,
                wp_depth_map,
                num_triangles,
                1e-8  # epsilon
            ],
            device=f"cuda:{device.index}" if device.index is not None else "cuda:0"
        )
    else:
        # For larger meshes, use tiled approach for better memory access
        triangle_tile_size = 10000  # Process triangles in tiles
        
        # Initialize depth map to infinity
        depth_map_flat.fill_(float('inf'))
        
        # Process triangles in tiles
        for tri_start in range(0, num_triangles, triangle_tile_size):
            tri_end = min(tri_start + triangle_tile_size, num_triangles)
            
            wp.launch(
                kernel=ray_triangle_intersection_tiled_kernel,
                dim=num_rays,
                inputs=[
                    wp_ray_origins,
                    wp_ray_directions,
                    wp_vertices,
                    wp_faces,
                    wp_depth_map,
                    tri_start,
                    tri_end,
                    1e-8  # epsilon
                ],
                device=f"cuda:{device.index}" if device.index is not None else "cuda:0"
            )
        
        # Convert infinity back to 0
        depth_map_flat[depth_map_flat == float('inf')] = 0.0
    
    # Synchronize to ensure kernel completion
    wp.synchronize()
    
    # Reshape back to 2D
    depth_map = depth_map_flat.reshape(H, W)
    
    return depth_map
