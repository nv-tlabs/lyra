# SPDX-FileCopyrightText: Copyright (c) 2026 MiLO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original work at github.com/MiLO83/DownToEarth/blob/main/voxgaussian/pipeline/uvw_atlas.py
# (MIT-licensed there). Re-licensed Apache-2.0 here to match the rest of the
# Lyra-2 codebase. Either license is fine to adopt.
#
# Bidirectional world-coord ↔ atlas-position bijection with mathematically
# inherited identity. Drop-in utility for Lyra 2's canonical-coord pipeline —
# see _build_canonical_world_coords in lyra2_model.py for the integration.
#
# Property: each voxel's (u, v, w) is bijective with both
#   (a) its 2D atlas pixel (atlas_x, atlas_y), and
#   (b) when packed as RGB bytes, the displayed pixel color is the coord.
#
# Identity is free — the mapping is computed, never stored. The application
# only pays VRAM for the *payload* stored at each atlas slot (class, color,
# Gaussian-splat params, latent features, etc.).
#
# Bit-width family (one per use-case):
#   bits=8   → 256³ voxels per axis, 16.7M total, 2.5 m world @ 1 cm cells
#   bits=16  → 65,536³, 281T total, 655 m world @ 1 cm cells
#   bits=32  → 4.29B³, 7.9e28 total, 42,000 km world @ 1 cm cells

from typing import Tuple
import torch


# ─── Forward / inverse bijection (resolution-parametric) ──────────────────


def voxel_to_atlas(
    u: torch.Tensor,           # any-shape integer tensor in [0, res)
    v: torch.Tensor,
    w: torch.Tensor,
    res: int = 256,
    tiles_per_row: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(u, v, w) → (atlas_x, atlas_y). O(1), arithmetic only.

    Default 256³ → 4096² (16 tiles × 16 tiles × 256² per tile = 16,777,216).
    For other resolutions, choose tiles_per_row = ceil(sqrt(res)) such that
    tiles_per_row * (res / tiles_per_row) >= res.

    >>> u = torch.tensor([0, 255, 128, 0]); v = torch.tensor([0, 255, 128, 0]); w = torch.tensor([0, 255, 0, 16])
    >>> ax, ay = voxel_to_atlas(u, v, w)
    >>> ax.tolist(), ay.tolist()
    ([0, 4095, 128, 0], [0, 4095, 128, 256])
    """
    tile_col = w % tiles_per_row
    tile_row = torch.div(w, tiles_per_row, rounding_mode='floor')
    atlas_x = tile_col * res + u
    atlas_y = tile_row * res + v
    return atlas_x, atlas_y


def atlas_to_voxel(
    atlas_x: torch.Tensor,
    atlas_y: torch.Tensor,
    res: int = 256,
    tiles_per_row: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(atlas_x, atlas_y) → (u, v, w). Inverse of voxel_to_atlas.

    >>> ax = torch.tensor([0, 4095, 128, 0]); ay = torch.tensor([0, 4095, 128, 256])
    >>> u, v, w = atlas_to_voxel(ax, ay)
    >>> u.tolist(), v.tolist(), w.tolist()
    ([0, 255, 128, 0], [0, 255, 128, 0], [0, 255, 0, 16])
    """
    u = atlas_x % res
    v = atlas_y % res
    tile_col = torch.div(atlas_x, res, rounding_mode='floor')
    tile_row = torch.div(atlas_y, res, rounding_mode='floor')
    w = tile_row * tiles_per_row + tile_col
    return u, v, w


# ─── World ↔ atlas (quantizing + dequantizing world coords) ───────────────


def world_to_quantized(
    world_xyz: torch.Tensor,         # (..., 3) float, in world units
    world_min: torch.Tensor,         # (3,) float, scene origin
    world_max: torch.Tensor,         # (3,) float, scene extent
    bits_per_axis: int = 16,
) -> torch.Tensor:                   # (..., 3) integer of matching width
    """Quantize world coords to integer-packed coords at `bits_per_axis`.

    The integer triple is the canonical 'address' — byte-perfect identity
    is preserved at bits=8 / 16 / 32 (uint dtypes). At 16-bit float (not
    used by this function) the identity is exact only to mantissa-2^10.
    """
    if bits_per_axis == 8:
        out_dtype = torch.uint8
    elif bits_per_axis == 16:
        out_dtype = torch.int32   # PyTorch lacks uint16; pack as int32, mask
    elif bits_per_axis == 32:
        out_dtype = torch.int64   # int64 mask to 32 bits at use-site
    else:
        raise ValueError(f"bits_per_axis must be 8 / 16 / 32; got {bits_per_axis}")

    n = (1 << bits_per_axis) - 1
    extent = (world_max - world_min)
    normed = (world_xyz - world_min) / torch.where(
        extent.abs() > 1e-9, extent, torch.ones_like(extent)
    )
    return (normed.clamp(0, 1) * n).round().to(out_dtype)


def quantized_to_world(
    quantized_xyz: torch.Tensor,
    world_min: torch.Tensor,
    world_max: torch.Tensor,
    bits_per_axis: int = 16,
) -> torch.Tensor:
    """Inverse: dequantize integer-packed coords back to world space."""
    n = (1 << bits_per_axis) - 1
    return world_min + (quantized_xyz.float() / n) * (world_max - world_min)


# ─── Tile-id mapping for disk-backed unbounded variant ────────────────────


def world_to_tile_id(
    world_xyz: torch.Tensor,         # (..., 3) float in world units
    tile_size_m: float,              # physical size of one tile, in meters
) -> torch.Tensor:                   # (..., 3) integer tile coords
    """Map world coords to integer tile IDs. The lower bits address inside
    the tile; the upper bits address which tile file on disk.

    For disk-backed streaming: tile_id_to_filename(tile_id) gives the file
    on the SSD; once loaded, intra-tile coords address the in-VRAM atlas
    slot via voxel_to_atlas() above.
    """
    return torch.floor(world_xyz / tile_size_m).to(torch.int64)


def tile_id_to_world_origin(
    tile_id: torch.Tensor,
    tile_size_m: float,
) -> torch.Tensor:
    """Inverse: integer tile coords back to the world position of the
    tile's origin corner."""
    return tile_id.float() * tile_size_m


# ─── World ↔ canonical RGB (the headline property: bytes ARE coords) ──────


def world_to_canonical_rgb(
    world_xyz: torch.Tensor,
    world_min: torch.Tensor,
    world_max: torch.Tensor,
    bits_per_axis: int = 8,
) -> torch.Tensor:
    """World 3-vec → packed (R, G, B) byte triple matching the bit-width.

    At bits=8: (..., 3) uint8 — the pixel color IS the quantized coord.
    At bits=16: (..., 3) int32, masked to 16 bits — pack to RG16UI texture.
    At bits=32: (..., 3) int64, masked to 32 bits — pack to RGB32UI texture.

    Used as input to ControlNet-style spatial conditioning (the Lyra 2
    `_build_canonical_*_coords` family).
    """
    return world_to_quantized(world_xyz, world_min, world_max, bits_per_axis)


def canonical_rgb_to_world(
    canonical_rgb: torch.Tensor,
    world_min: torch.Tensor,
    world_max: torch.Tensor,
    bits_per_axis: int = 8,
) -> torch.Tensor:
    """Inverse: read pixel bytes back as world coords. No projection math."""
    return quantized_to_world(canonical_rgb, world_min, world_max, bits_per_axis)


# ─── Self-test ────────────────────────────────────────────────────────────


def verify_bijection(res: int = 256, tiles_per_row: int = 16) -> bool:
    """Verify (atlas_to_voxel ∘ voxel_to_atlas) == identity for all res³ coords.

    Returns True iff round-trip is exact across the entire address space.
    Run with res=64 in CI; res=256 in pre-release.
    """
    coords = torch.cartesian_prod(
        torch.arange(res), torch.arange(res), torch.arange(res)
    )
    u, v, w = coords[:, 0], coords[:, 1], coords[:, 2]
    ax, ay = voxel_to_atlas(u, v, w, res=res, tiles_per_row=tiles_per_row)
    u2, v2, w2 = atlas_to_voxel(ax, ay, res=res, tiles_per_row=tiles_per_row)
    return bool((u == u2).all() and (v == v2).all() and (w == w2).all())


if __name__ == "__main__":
    import doctest
    failures, tests = doctest.testmod(verbose=False)
    print(f"doctest: {tests - failures}/{tests} passed")
    print("Exhaustive bijection check (64³ pairs)...", end=" ", flush=True)
    print("OK" if verify_bijection(res=64, tiles_per_row=8) else "FAILED")
