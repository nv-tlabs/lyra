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
import math
import torch.nn.functional as F
from .forward_warp_utils_pytorch import unproject_points

def apply_transformation(Bx4x4, another_matrix):
    B = Bx4x4.shape[0]
    if another_matrix.dim() == 2:
        another_matrix = another_matrix.unsqueeze(0).expand(B, -1, -1)  # Make another_matrix compatible with batch size
    transformed_matrix = torch.bmm(Bx4x4, another_matrix)  # Shape: (B, 4, 4)

    return transformed_matrix


def look_at_matrix(camera_pos, target, invert_pos=True):
    """Creates a 4x4 look-at matrix, keeping the camera pointing towards a target."""
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)

    up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)  # assuming Y-up coordinate system
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    look_at = torch.eye(4, device=camera_pos.device)
    look_at[0, :3] = right
    look_at[1, :3] = up
    look_at[2, :3] = forward
    look_at[:3, 3] = (-camera_pos) if invert_pos else camera_pos

    return look_at

def create_horizontal_trajectory(
    world_to_camera_matrix, center_depth, positive=True, n_steps=13, distance=0.1, device="cuda", axis="x", camera_rotation="center_facing"
):
    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    # Spiral motion key points
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)

    for i in range(n_steps):
        if axis == "x": # pos - right
            x = i * distance * center_depth / n_steps * (1 if positive else -1)
            y = 0
            z = 0
        elif axis == "y": # pos - down
            x = 0
            y = i * distance * center_depth / n_steps * (1 if positive else -1)
            z = 0
        elif axis == "z": # pos - in
            x = 0
            y = 0
            z = i * distance * center_depth / n_steps * (1 if positive else -1)
        else:
            raise ValueError("Axis should be x, y or z")

        translation_positions.append(torch.tensor([x, y, z], device=device))

    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if camera_rotation == "trajectory_aligned":
            _look_at = look_at + pos * 2
        elif camera_rotation == "center_facing":
            _look_at = look_at
        elif camera_rotation == "no_rotation":
            _look_at = look_at + pos
        else:
            raise ValueError("Camera rotation should be center_facing or trajectory_aligned")
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_spiral_trajectory(
    world_to_camera_matrix,
    center_depth,
    radius_x=0.03,
    radius_y=0.02,
    radius_z=0.0,
    positive=True,
    camera_rotation="center_facing",
    n_steps=13,
    device="cuda",
    start_from_zero=True,
    num_circles=1,
):

    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)

    # Spiral motion key points
    trajectory = []
    spiral_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)  # world_to_camera_matrix[:3, 3].clone()

    example_scale = 1.0

    theta_max = 2 * math.pi * num_circles

    for i in range(n_steps):
        # theta = 2 * math.pi * i / (n_steps-1)  # angle for each point
        theta = theta_max * i / (n_steps - 1)  # angle for each point
        if start_from_zero:
            x = radius_x * (math.cos(theta) - 1) * (1 if positive else -1) * (center_depth / example_scale)
        else:
            x = radius_x * (math.cos(theta)) * (center_depth / example_scale)

        y = radius_y * math.sin(theta) * (center_depth / example_scale)
        z = radius_z * math.sin(theta) * (center_depth / example_scale)
        spiral_positions.append(torch.tensor([x, y, z], device=device))

    for pos in spiral_positions:
        if camera_rotation == "center_facing":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at)
        elif camera_rotation == "trajectory_aligned":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at + pos * 2)
        elif camera_rotation == "no_rotation":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at + pos)
        else:
            raise ValueError("Camera rotation should be center_facing, trajectory_aligned or no_rotation")
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def generate_camera_trajectory(
    trajectory_type: str,
    initial_w2c: torch.Tensor,  # Shape: (4, 4)
    initial_intrinsics: torch.Tensor,  # Shape: (3, 3)
    num_frames: int,
    movement_distance: float,
    camera_rotation: str,
    center_depth: float = 1.0,
    device: str = "cuda",
    num_circles: int = 1,
    radius_x_factor: float = 1.0,
    radius_y_factor: float = 1.0,
):
    """
    Generates a sequence of camera poses (world-to-camera matrices) and intrinsics
    for a specified trajectory type.

    Args:
        trajectory_type: Type of trajectory (e.g., "left", "right", "up", "down", "zoom_in", "zoom_out").
        initial_w2c: Initial world-to-camera matrix (4x4 tensor or num_framesx4x4 tensor).
        initial_intrinsics: Camera intrinsics matrix (3x3 tensor or num_framesx3x3 tensor).
        num_frames: Number of frames (steps) in the trajectory.
        movement_distance: Distance factor for the camera movement.
        camera_rotation: Type of camera rotation ('center_facing', 'no_rotation', 'trajectory_aligned').
        center_depth: Depth of the center point the camera might focus on.
        device: Computation device ("cuda" or "cpu").
        num_circles: Number of circles for spiral
        radius_x_factor: Multiple strength in x direction with factor for spiral
        radius_y_factor: Multiple strength in x direction with factor for spiral

    Returns:
        A tuple (generated_w2cs, generated_intrinsics):
        - generated_w2cs: Batch of world-to-camera matrices for the trajectory (1, num_frames, 4, 4 tensor).
        - generated_intrinsics: Batch of camera intrinsics for the trajectory (1, num_frames, 3, 3 tensor).
    """
    if trajectory_type in ["clockwise", "counterclockwise"]:
        radius_x = movement_distance * radius_x_factor
        radius_y = movement_distance * radius_y_factor
        new_w2cs_seq = create_spiral_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=trajectory_type == "clockwise",
            device=device,
            camera_rotation=camera_rotation,
            radius_x=radius_x,
            radius_y=radius_y,
            num_circles=num_circles,
        )
    else:
        if trajectory_type == "left":
            positive = False
            axis = "x"
        elif trajectory_type == "right":
            positive = True
            axis = "x"
        elif trajectory_type == "up":
            positive = False  # Assuming 'up' means camera moves in negative y direction if y points down
            axis = "y"
        elif trajectory_type == "down":
            positive = True # Assuming 'down' means camera moves in positive y direction if y points down
            axis = "y"
        elif trajectory_type == "zoom_in":
            positive = True  # Assuming 'zoom_in' means camera moves in positive z direction (forward)
            axis = "z"
        elif trajectory_type == "zoom_out":
            positive = False # Assuming 'zoom_out' means camera moves in negative z direction (backward)
            axis = "z"
        else:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")

        # Generate world-to-camera matrices using create_horizontal_trajectory
        new_w2cs_seq = create_horizontal_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=positive,
            axis=axis,
            distance=movement_distance,
            device=device,
            camera_rotation=camera_rotation,
        )

    generated_w2cs = new_w2cs_seq.unsqueeze(0)  # Shape: [1, num_frames, 4, 4]
    if initial_intrinsics.dim() == 2:
        generated_intrinsics = initial_intrinsics.unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1)
    else:
        generated_intrinsics = initial_intrinsics.unsqueeze(0)

    return generated_w2cs, generated_intrinsics


def _align_inv_depth_to_depth(
    source_inv_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply affine transformation to align source inverse depth to target depth.

    Args:
        source_inv_depth: Inverse depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    target_inv_depth = 1.0 / target_depth
    source_mask = source_inv_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    outlier_quantiles = torch.tensor([0.1, 0.9], device=source_inv_depth.device)

    source_data_low, source_data_high = torch.quantile(source_inv_depth[source_mask], outlier_quantiles)
    target_data_low, target_data_high = torch.quantile(target_inv_depth[target_mask], outlier_quantiles)
    source_mask = (source_inv_depth > source_data_low) & (source_inv_depth < source_data_high)
    target_mask = (target_inv_depth > target_data_low) & (target_inv_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_inv_depth[mask].view(-1, 1)
    target_data = target_inv_depth[mask].view(-1, 1)

    ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
    source_data_h = torch.cat([source_data, ones], dim=1)
    transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

    scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
    aligned_inv_depth = source_inv_depth * scale + bias

    return 1.0 / aligned_inv_depth


def align_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    k: torch.Tensor = None,
    c2w: torch.Tensor = None,
    alignment_method: str = "rigid",
    num_iters: int = 100,
    lambda_arap: float = 0.1,
    smoothing_kernel_size: int = 3,
) -> torch.Tensor:
    if alignment_method == "rigid":
        source_inv_depth = 1.0 / source_depth
        source_depth = _align_inv_depth_to_depth(source_inv_depth, target_depth, target_mask)
        return source_depth
    elif alignment_method == "non_rigid":
        if k is None or c2w is None:
            raise ValueError("Camera intrinsics (k) and camera-to-world matrix (c2w) are required for non-rigid alignment")
            
        source_inv_depth = 1.0 / source_depth
        source_depth = _align_inv_depth_to_depth(source_inv_depth, target_depth, target_mask)
        
        # Initialize scale map
        sc_map = torch.ones_like(source_depth).float().to(source_depth.device).requires_grad_(True)
        optimizer = torch.optim.Adam(params=[sc_map], lr=0.001)
        
        # Unproject target depth
        target_unprojected = unproject_points(
            target_depth.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            c2w.unsqueeze(0),  # Add batch dimension
            k.unsqueeze(0),  # Add batch dimension
            is_depth=True,
            mask=target_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        ).squeeze(0)  # Remove batch dimension
        
        # Create smoothing kernel
        smoothing_kernel = torch.ones(
            (1, 1, smoothing_kernel_size, smoothing_kernel_size),
            device=source_depth.device
        ) / (smoothing_kernel_size**2)
        
        for _ in range(num_iters):
            # Unproject scaled source depth
            source_unprojected = unproject_points(
                (source_depth * sc_map).unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                c2w.unsqueeze(0),  # Add batch dimension
                k.unsqueeze(0),  # Add batch dimension
                is_depth=True,
                mask=target_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            ).squeeze(0)  # Remove batch dimension
            
            # Data loss
            data_loss = torch.abs(source_unprojected[target_mask] - target_unprojected[target_mask]).mean()
            
            # Apply smoothing filter to sc_map
            sc_map_reshaped = sc_map.unsqueeze(0).unsqueeze(0)
            sc_map_smoothed = F.conv2d(
                sc_map_reshaped,
                smoothing_kernel,
                padding=smoothing_kernel_size // 2
            ).squeeze(0).squeeze(0)
            
            # ARAP loss
            arap_loss = torch.abs(sc_map_smoothed - sc_map).mean()
            
            # Total loss
            loss = data_loss + lambda_arap * arap_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return source_depth * sc_map
    else:
        raise ValueError(f"Unsupported alignment method: {alignment_method}")
