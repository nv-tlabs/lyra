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


# ─── Occupancy bitmap — same-grid 1-bit-per-voxel empty/full toggle ───────


class OccupancyBitmap:
    """Same-resolution 1-bit-per-voxel companion to the canonical-coord atlas.

    Stores one bit per voxel answering "is this slot populated?". Lets the
    cross-attention or shader skip the multi-byte main-atlas read for empty
    voxels via a single-bit test that's 24-32x cheaper in storage *and*
    bandwidth.

    Memory math:

      Grid       Voxels      RGBA8 atlas    RGB only    1-bit mask    Ratio
      256-cubed  16.7M       67 MB          50 MB       2.1 MB        24x
      512-cubed  134M        537 MB         403 MB      16.8 MB       24x
      1024-cubed 1.07B       4.3 GB         3.2 GB      134 MB        24x

    Layout: 8 voxels packed per byte along the X axis of the 2D atlas
    (preserves cache locality for horizontal-scan access patterns).

    For Lyra 2 specifically, this pairs with the canonical-coord encoding
    from _build_canonical_world_coords as the explicit emptiness signal —
    the cross-attention doesn't have to learn that (0, 0, 0) is special;
    a dedicated 1-bit channel says so structurally.

    >>> bmap = OccupancyBitmap(atlas_w=64, atlas_h=64)
    >>> bmap.nbytes
    512
    >>> bmap.get(0, 0)
    False
    >>> bmap.set(0, 0)
    >>> bmap.get(0, 0)
    True
    >>> bmap.count_occupied()
    1
    """

    def __init__(self, atlas_w: int = 4096, atlas_h: int = 4096):
        if atlas_w % 8 != 0:
            raise ValueError(
                f"atlas_w must be divisible by 8 for X-axis bit-packing, got {atlas_w}"
            )
        self.atlas_w = atlas_w
        self.atlas_h = atlas_h
        # NumPy is fine here — this is a CPU-side build-time data structure;
        # the GPU consumes the resulting bytes as a R8UI texture sampled via
        # texelFetch + shift + AND. See module docstring for the GLSL pattern.
        import numpy as np
        self._np = np
        self.bits = np.zeros((atlas_h, atlas_w // 8), dtype=np.uint8)

    def __repr__(self) -> str:
        n = self.count_occupied()
        total = self.atlas_w * self.atlas_h
        pct = (100.0 * n / total) if total > 0 else 0.0
        return (f"OccupancyBitmap({self.atlas_w}x{self.atlas_h}, "
                f"{self.nbytes / 1024:.1f} KB, {n}/{total} occupied = {pct:.3f}%)")

    @property
    def nbytes(self) -> int:
        """Total bytes used by the bitmap."""
        return int(self.bits.nbytes)

    def set(self, atlas_x: int, atlas_y: int, occupied: bool = True) -> None:
        """Set / clear the bit at (atlas_x, atlas_y)."""
        byte_idx = atlas_x >> 3
        bit_idx = atlas_x & 7
        if occupied:
            self.bits[atlas_y, byte_idx] |= self._np.uint8(1 << bit_idx)
        else:
            self.bits[atlas_y, byte_idx] &= self._np.uint8(~(1 << bit_idx) & 0xFF)

    def get(self, atlas_x: int, atlas_y: int) -> bool:
        """Read the bit at (atlas_x, atlas_y)."""
        byte_idx = atlas_x >> 3
        bit_idx = atlas_x & 7
        return bool((self.bits[atlas_y, byte_idx] >> bit_idx) & 1)

    def count_occupied(self) -> int:
        """Total number of bits set across the whole bitmap."""
        return int(self._np.unpackbits(self.bits).sum())

    def occupancy_fraction(self) -> float:
        """Fraction of voxels populated, in [0, 1]."""
        total = self.atlas_w * self.atlas_h
        return self.count_occupied() / total if total > 0 else 0.0

    def fill_from_voxel_iter(self, voxels, res: int = 256,
                              tiles_per_row: int = 16) -> int:
        """Bulk-set occupancy from any iterable of (u, v, w) tuples.

        Vectorised via numpy bitwise_or.at. Returns the count of voxels written.
        """
        np = self._np
        voxels = list(voxels)
        if not voxels:
            return 0
        coords = np.asarray(voxels, dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Expected list of (u, v, w) triples; got shape {coords.shape}"
            )
        u = coords[:, 0]
        v = coords[:, 1]
        w_ = coords[:, 2]
        # Pure-numpy version of voxel_to_atlas (no torch needed at build time).
        ax = (w_ % tiles_per_row) * res + u
        ay = (w_ // tiles_per_row) * res + v
        byte_idx = ax >> 3
        bit_mask = (1 << (ax & 7)).astype(np.uint8)
        np.bitwise_or.at(self.bits, (ay, byte_idx), bit_mask)
        return len(voxels)

    def to_bytes(self) -> bytes:
        """Raw byte buffer in row-major order."""
        return self.bits.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, atlas_w: int = 4096,
                    atlas_h: int = 4096) -> "OccupancyBitmap":
        """Reconstruct from to_bytes() output."""
        import numpy as np
        expected = atlas_h * (atlas_w // 8)
        if len(data) != expected:
            raise ValueError(
                f"Byte length {len(data)} does not match expected {expected} "
                f"for {atlas_w}x{atlas_h} atlas"
            )
        out = cls(atlas_w, atlas_h)
        out.bits = np.frombuffer(data, dtype=np.uint8).reshape(
            (atlas_h, atlas_w // 8)
        ).copy()
        return out

    def as_torch_tensor(self) -> "torch.Tensor":
        """Return as a torch uint8 tensor for GPU upload.

        Suitable for binding to a R8UI texture. The GLSL fetch pattern is in
        the module docstring.
        """
        return torch.from_numpy(self.bits)


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
    print("Exhaustive bijection check (64-cubed pairs)...", end=" ", flush=True)
    print("OK" if verify_bijection(res=64, tiles_per_row=8) else "FAILED")

    print("\nOccupancyBitmap self-test:")
    bmap = OccupancyBitmap(atlas_w=512, atlas_h=512)  # 64-cubed grid
    print(f"  empty: {bmap!r}")
    # Set a few voxels via bulk-fill
    test_voxels = [(0, 0, 0), (1, 1, 1), (63, 63, 63), (32, 16, 8)]
    n = bmap.fill_from_voxel_iter(test_voxels, res=64, tiles_per_row=8)
    print(f"  after fill_from_voxel_iter({n} voxels): {bmap!r}")
    # Verify each voxel reads back
    res = 64
    tiles_per_row = 8
    ok = True
    for (u, v, w) in test_voxels:
        ax = (w % tiles_per_row) * res + u
        ay = (w // tiles_per_row) * res + v
        if not bmap.get(ax, ay):
            ok = False
            print(f"    FAIL: voxel ({u}, {v}, {w}) not set")
    print(f"  round-trip: {'OK' if ok else 'FAILED'}")
    # Bytes round-trip
    raw = bmap.to_bytes()
    reconstructed = OccupancyBitmap.from_bytes(raw, atlas_w=512, atlas_h=512)
    same = (reconstructed.bits == bmap.bits).all()
    print(f"  to_bytes / from_bytes: {'OK' if same else 'FAILED'}")
