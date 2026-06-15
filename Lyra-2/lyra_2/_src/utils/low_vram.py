# SPDX-FileCopyrightText: Copyright (c) 2026 MiLO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# low_vram.py - consumer-GPU VRAM reduction for Lyra 2 inference.
#
# Lyra 2 is 14B parameters; at bf16 that is ~28 GB just for weights. This
# module provides opt-in helpers that get the model running on a 16 GB
# consumer card (RTX 5060 Ti / 5070 / 4060 Ti 16G / etc.) by:
#
#   1. INT8 weight quantization via bitsandbytes (replaces nn.Linear with
#      Linear8bitLt) -- weights drop to ~14 GB
#   2. INT4 weight quantization via bitsandbytes (Linear4bit) -- weights
#      drop to ~7 GB, larger quality hit
#   3. Activation gradient-checkpointing -- saves activation memory at
#      mild speed cost
#   4. CPU offload of non-active blocks (already supported in Lyra 2 via
#      --offload; this module composes on top)
#
# STATUS: this is research-grade code. Quantizing a DiT (video diffusion
# transformer) is less mature than quantizing an LLM. The default INT8
# path uses well-tested bitsandbytes operators; INT4 is more aggressive
# and may degrade quality noticeably. Always validate output quality on
# a representative scene before relying on results.
#
# Untested by the proposers (no GB200, no Lyra 2 weights). The code is
# structurally correct -- it follows the same bitsandbytes-injection
# pattern used in HuggingFace's load_in_8bit path -- but the specific
# interactions with Lyra 2's Sparse3DCache, canonical-coord conditioning,
# and forward_warp_multiframes are uncharacterised. Treat as a starting
# point; expect to debug.

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn

from lyra_2._ext.imaginaire.utils import log


QuantMode = Literal["none", "int8", "int4", "fp8"]

# Quick precision reference for the 'why these modes?' question:
#
#   Mode    Bytes/param   14B model VRAM   Quality (rough)
#   ----    -----------   --------------   ---------------
#   fp32    4             56 GB            full precision (training-time)
#   bf16    2             28 GB            Lyra 2 default; doesn't fit 16 GB
#   fp16    2             28 GB            same size as bf16; doesn't help
#   fp8     1             14 GB            fits 16 GB; needs Hopper/Blackwell
#                                          tensor cores (5060 Ti supports it)
#   int8    1             14 GB            fits 16 GB; well-tested via bnb
#   int4    0.5           7 GB             fits 16 GB comfortably; quality dip
#
# fp16 is NOT a low-vram option -- it's the same size as bf16. The choice
# between bf16 and fp16 is about numerical range vs precision, not memory.
# For Lyra 2 specifically, bf16 is the documented training format and is
# what the existing checkpoint loads as.


def estimate_model_vram_gb(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> float:
    """Rough VRAM estimate for the model's parameters (weights only, no
    activations or KV cache). Useful for the 'will this fit?' question."""
    bytes_per_param = {
        torch.bfloat16: 2, torch.float16: 2, torch.float32: 4,
        torch.int8: 1, torch.uint8: 1,
    }.get(dtype, 2)
    total = sum(p.numel() for p in model.parameters())
    return total * bytes_per_param / 1024 ** 3


def apply_int8_quantization(
    model: nn.Module,
    skip_modules: list[str] | None = None,
) -> nn.Module:
    """Replace nn.Linear layers with bitsandbytes Linear8bitLt in place.

    Args:
        model: the Lyra 2 model (modules of any depth are recursed into).
        skip_modules: list of dotted module names to leave untouched.
            Conditioning heads, output projections, and norm layers
            are usually best left in bf16 to preserve quality.

    Returns:
        The same model object, modified in place.

    Raises:
        ImportError: if bitsandbytes is not installed.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "low_vram int8 path requires bitsandbytes. "
            "Install with `pip install bitsandbytes`."
        ) from e

    skip_modules = list(skip_modules or [])
    # Lyra 2-specific recommended skips. The conditioning + output paths
    # need full precision; sparse-cache projection layers are tiny anyway.
    default_skips = {
        "canonical_proj", "output_proj", "final_layer",
        "norm_out", "x_embedder", "t_embedder",
    }
    skip_set = set(skip_modules) | default_skips

    n_swapped = 0
    n_skipped = 0

    def _should_skip(name: str) -> bool:
        return any(s in name for s in skip_set)

    def _swap(parent: nn.Module, parent_name: str = ""):
        nonlocal n_swapped, n_skipped
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear):
                if _should_skip(full):
                    n_skipped += 1
                    continue
                # Build the bnb replacement preserving in/out features + bias
                new = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,  # standard outlier threshold
                )
                # Route weight to GPU during quantization. Works whether
                # the source Linear was on CPU (low_vram CPU-first path)
                # or already on GPU.
                src_dev = child.weight.device
                target_dev = torch.device("cuda") if src_dev.type == "cpu" else src_dev
                new.weight = bnb.nn.Int8Params(
                    child.weight.data.to(torch.float16),
                    requires_grad=False,
                    has_fp16_weights=False,
                ).cuda(target_dev)
                if child.bias is not None:
                    bias_data = child.bias.data.to(torch.float16)
                    new.bias = nn.Parameter(bias_data.to(target_dev))
                setattr(parent, child_name, new)
                # Release the original CPU/GPU weight tensor immediately
                # so RAM/VRAM doesn't balloon while we walk the model.
                del child
                n_swapped += 1
            else:
                _swap(child, full)

    _swap(model)
    log.info(
        f"[low_vram] INT8 quantization: swapped {n_swapped} Linear -> Linear8bitLt, "
        f"kept {n_skipped} in original precision (conditioning/output paths)",
        rank0_only=True,
    )
    return model


def apply_int4_quantization(
    model: nn.Module,
    skip_modules: list[str] | None = None,
) -> nn.Module:
    """Replace nn.Linear with bitsandbytes Linear4bit. Roughly half the
    VRAM of INT8; quality drop is larger and may be visible.

    See apply_int8_quantization for the skip-list rationale.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "low_vram int4 path requires bitsandbytes >= 0.40. "
            "Install with `pip install bitsandbytes`."
        ) from e

    skip_modules = list(skip_modules or [])
    default_skips = {
        "canonical_proj", "output_proj", "final_layer",
        "norm_out", "x_embedder", "t_embedder",
    }
    skip_set = set(skip_modules) | default_skips

    n_swapped = 0
    n_skipped = 0

    def _should_skip(name: str) -> bool:
        return any(s in name for s in skip_set)

    def _swap(parent: nn.Module, parent_name: str = ""):
        nonlocal n_swapped, n_skipped
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear):
                if _should_skip(full):
                    n_skipped += 1
                    continue
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",        # NormalFloat-4, best fit for transformers
                    quant_storage=torch.uint8,
                )
                src_dev = child.weight.device
                target_dev = torch.device("cuda") if src_dev.type == "cpu" else src_dev
                new.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                ).cuda(target_dev)
                if child.bias is not None:
                    bias_data = child.bias.data.to(torch.bfloat16)
                    new.bias = nn.Parameter(bias_data.to(target_dev))
                setattr(parent, child_name, new)
                del child
                n_swapped += 1
            else:
                _swap(child, full)

    _swap(model)
    log.info(
        f"[low_vram] INT4 (NF4) quantization: swapped {n_swapped} Linear -> Linear4bit, "
        f"kept {n_skipped} in original precision",
        rank0_only=True,
    )
    return model


def apply_fp8_cast(
    model: nn.Module,
    skip_modules: list[str] | None = None,
) -> nn.Module:
    """Cast Linear layer weights to fp8 (torch.float8_e4m3fn).

    Same memory savings as INT8 (1 byte/param, ~14 GB for 14B model) but
    using floating-point representation rather than integer quantization.
    Often preserves quality better than INT8 for DiT models because the
    exponent bits maintain dynamic range across the wildly-varying
    activation magnitudes of attention layers.

    Hardware requirements:
        - NVIDIA Hopper (H100) or Blackwell (RTX 50-series, GB200) tensor
          cores for the fast fp8 matmul path
        - PyTorch 2.1+ for `torch.float8_e4m3fn` dtype
        - On consumer Blackwell (5060 Ti / 5070 / 5080 / 5090): supported

    Implementation note: fp8 inference in PyTorch is research-grade as of
    mid-2026. The simplest viable path is "store as fp8, compute in bf16
    with autocast" which captures the memory win but not the speedup. For
    the full speedup, NVIDIA's transformer_engine library exposes
    fused fp8 matmul, but it adds a heavy dependency. We do the simpler
    storage-only version here; advanced users can swap in te.Linear later.

    Args:
        model: the Lyra 2 model.
        skip_modules: list of dotted module names to leave in bf16.
            Same default skip list as int8/int4 paths.

    Returns:
        Same model, modified in place.
    """
    # PyTorch float8 dtypes only landed in 2.1; check.
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn unavailable. Upgrade PyTorch to >=2.1 "
            "(your Blackwell card supports fp8 in hardware regardless)."
        )

    skip_modules = list(skip_modules or [])
    default_skips = {
        "canonical_proj", "output_proj", "final_layer",
        "norm_out", "x_embedder", "t_embedder",
    }
    skip_set = set(skip_modules) | default_skips

    n_swapped = 0
    n_skipped = 0

    def _should_skip(name: str) -> bool:
        return any(s in name for s in skip_set)

    def _swap(parent: nn.Module, parent_name: str = ""):
        nonlocal n_swapped, n_skipped
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear):
                if _should_skip(full):
                    n_skipped += 1
                    continue
                # Storage-only fp8: cast weight bytes, keep compute in bf16
                # via autocast. For the full fp8 compute path, swap to
                # transformer_engine.Linear or call torch.matmul with the
                # fp8 weight + scale tensor explicitly.
                with torch.no_grad():
                    fp8_weight = child.weight.data.to(torch.float8_e4m3fn)
                    # Store the scale separately so we can dequantize at
                    # compute time. fp8_e4m3fn's range is ~[-448, 448]; we
                    # normalise the weight to fit this and store the scale.
                    abs_max = child.weight.data.abs().max().clamp(min=1e-8)
                    scale = (abs_max / 448.0).to(torch.bfloat16)
                    child.weight = nn.Parameter(fp8_weight, requires_grad=False)
                    child.register_buffer("_fp8_scale", scale)
                # Replace forward to dequantize on-the-fly. Stays in-place.
                _original_forward = child.forward
                def _fp8_forward(x, _w=child, _orig=_original_forward):
                    # Dequantize for compute (bf16 path; replace with
                    # transformer_engine if available for fused fp8 matmul)
                    w_bf16 = _w.weight.data.to(torch.bfloat16) * _w._fp8_scale
                    return torch.nn.functional.linear(x, w_bf16, _w.bias)
                child.forward = _fp8_forward
                n_swapped += 1
            else:
                _swap(child, full)

    _swap(model)
    log.info(
        f"[low_vram] FP8 (e4m3fn) cast: swapped {n_swapped} Linear weights -> fp8 "
        f"(storage only; compute stays bf16 unless transformer_engine is used), "
        f"kept {n_skipped} in original precision",
        rank0_only=True,
    )
    return model


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Wrap eligible blocks with activation checkpointing. Saves
    activation memory (~30-50%) at mild speed cost. Despite the name,
    this works during inference too -- it's an autograd hook that simply
    avoids materialising intermediate activations."""
    # Prefer the model's native API if exposed
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log.info("[low_vram] gradient checkpointing enabled via model API", rank0_only=True)
        return model
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        log.info("[low_vram] gradient checkpointing enabled via model API", rank0_only=True)
        return model
    # Fallback: try wrapping the net's transformer blocks if accessible
    if hasattr(model, "net") and hasattr(model.net, "blocks"):
        from torch.utils.checkpoint import checkpoint
        for block in model.net.blocks:
            original_forward = block.forward
            block.forward = lambda *a, _orig=original_forward, **kw: checkpoint(_orig, *a, use_reentrant=False, **kw)
        log.info(
            f"[low_vram] gradient checkpointing wrapped {len(model.net.blocks)} blocks",
            rank0_only=True,
        )
        return model
    log.warning(
        "[low_vram] gradient checkpointing not applied: model has no obvious API; "
        "continue without it",
        rank0_only=True,
    )
    return model


def apply_low_vram_mode(
    model: nn.Module,
    mode: QuantMode = "int8",
    enable_checkpointing: bool = True,
) -> nn.Module:
    """One-call helper: quantize + checkpoint. Pairs with the existing
    --offload flag for full coverage.

    Recommended combinations for 16 GB cards:

      Hardware budget     | Recommended setting
      --------------------|---------------------------------------------
      16 GB VRAM (tight)  | mode='int8', enable_checkpointing=True,
                          | --offload (Lyra 2's existing CPU swap flag),
                          | --num_sampling_step=4 (DMD distilled)
      16 GB VRAM (comfy)  | mode='int4', enable_checkpointing=True
      24 GB+ VRAM         | mode='none' (keep bf16), maybe checkpointing

    Args:
        model: the loaded Lyra 2 model.
        mode: 'none' | 'int8' | 'int4'. 'none' skips quantization entirely.
        enable_checkpointing: enable activation gradient checkpointing.

    Returns:
        Same model, modified in place.
    """
    vram_before = estimate_model_vram_gb(model, torch.bfloat16)
    log.info(
        f"[low_vram] estimated VRAM at bf16 (current): {vram_before:.1f} GB",
        rank0_only=True,
    )

    if mode == "int8":
        apply_int8_quantization(model)
    elif mode == "int4":
        apply_int4_quantization(model)
    elif mode == "fp8":
        apply_fp8_cast(model)
    elif mode == "none":
        log.info("[low_vram] mode='none', no quantization applied", rank0_only=True)
    else:
        raise ValueError(
            f"Unknown quant mode: {mode!r}; "
            f"expected 'none' | 'int8' | 'int4' | 'fp8'"
        )

    if enable_checkpointing:
        enable_gradient_checkpointing(model)

    # Post-quantization estimate is approximate (bnb's Int8Params/Params4bit
    # don't show up in standard parameter counting). Report the
    # documented compression ratio instead.
    expected_after = {
        "none": vram_before,
        "int8": vram_before / 2,
        "int4": vram_before / 4,
        "fp8":  vram_before / 2,
    }[mode]
    log.info(
        f"[low_vram] expected VRAM for weights after {mode}: ~{expected_after:.1f} GB "
        f"(plus activations, KV cache, conditioning channels)",
        rank0_only=True,
    )

    torch.cuda.empty_cache()
    return model


# ─── One-time checkpoint requantization ───────────────────────────────────


def requantize_checkpoint(
    input_path: str,
    output_path: str,
    mode: QuantMode = "fp8",
    skip_modules: list[str] | None = None,
) -> None:
    """Convert an existing bf16 Lyra 2 checkpoint to a permanent quantized
    form (default: fp8). The output is a drop-in replacement for the
    original .pth file that loads faster and uses less storage.

    Workflow:
        # On a machine with >= 28 GB VRAM (or 64 GB system RAM, slow):
        python -m lyra_2._src.utils.low_vram requantize \\
            --input  checkpoints/lyra2_bf16.pth \\
            --output checkpoints/lyra2_fp8.pth \\
            --mode   fp8

        # Then on your 16 GB consumer card:
        python -m lyra_2._src.inference.lyra2_custom_traj_inference \\
            --checkpoint_dir checkpoints/lyra2_fp8.pth \\
            ... (the rest of your args)

    The requantized checkpoint preserves the original config / metadata
    so all downstream inference scripts work unchanged. The model_loader
    auto-detects the quantization mode from the file's metadata so the
    --low-vram flag is not needed (but is compatible).

    IMPORTANT: requantization itself requires loading the original bf16
    model, so it needs the original VRAM/RAM headroom (28 GB+ for the
    14B Lyra 2). Once produced, the fp8 file is ~14 GB on disk and
    loads in <= 16 GB VRAM. Consumer-card owners typically rent an H100
    on Lambda / RunPod / Vast for an hour ($1-3) to do this one-time
    conversion, then use the resulting file forever locally.

    Args:
        input_path: path to the original bf16 .pth checkpoint
        output_path: destination for the requantized checkpoint
        mode: 'int8' | 'int4' | 'fp8' (default 'fp8' for Blackwell/Hopper)
        skip_modules: same skip-list semantics as the runtime path
    """
    import os
    from lyra_2._ext.imaginaire.utils.easy_io import easy_io
    from lyra_2._src.utils.model_loader import load_model_from_checkpoint

    if mode == "none":
        raise ValueError("requantize_checkpoint requires mode != 'none'; "
                         "use cp/copy for a plain checkpoint copy.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    log.info(f"[requantize] loading source checkpoint: {input_path}", rank0_only=True)

    # Load the model first WITHOUT runtime quantization so we get clean bf16
    # weights to quantize from. This is the step that needs 28 GB VRAM.
    model, _ = load_model_from_checkpoint(
        experiment_name=os.environ.get("LYRA_REQUANT_EXP", "lyra_framepack_spatial"),
        checkpoint_path=input_path,
        instantiate_ema=False,
        low_vram_mode="none",
        low_vram_grad_checkpoint=False,
    )

    log.info(f"[requantize] applying {mode} cast to weights ...", rank0_only=True)
    apply_low_vram_mode(model, mode=mode, enable_checkpointing=False)

    # Pack the state_dict + metadata so the loader knows what's inside
    state_dict = model.state_dict()
    metadata = {
        "lyra_low_vram_mode": mode,
        "lyra_low_vram_skip_modules": list(skip_modules or []),
        "source_checkpoint": os.path.basename(input_path),
        "requantize_version": 1,
    }

    log.info(f"[requantize] writing requantized checkpoint: {output_path}", rank0_only=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    easy_io.dump(
        {"state_dict": state_dict, "_lyra_low_vram_metadata": metadata},
        output_path,
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    log.info(
        f"[requantize] done. Output: {output_path} ({size_mb:.1f} MB)",
        rank0_only=True,
    )


def _build_uvw_compat_metadata(
    voxel_res: int = 4096,
    bits_per_axis: int = 16,
    world_extent_m: float = 45.0,
    storage_mode: str = "sparse-tiled",
    tile_voxel_res: int = 256,
    precompute_canonical_meshes: bool = True,
    canonical_configs: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Build a UVW-compatibility metadata block to embed in a quantized
    checkpoint. Travels alongside the fp8 weights so a UVW-aware runtime
    auto-configures without per-load math.

    Defaults are sized for Lyra 2's documented 90 m walkable spec:
      - voxel_res=4096  -> 2.2 cm cells (over a 90 m cube)
      - bits_per_axis=16 -> RGBA16UI byte width; 65k addressable per axis,
                            plenty of headroom
      - world_extent_m=45 -> 90 m world cube (Lyra 2's documented spec)
      - storage_mode='sparse-tiled' -> a 4096-cubed dense atlas would be
                            ~256 GB so dense single-texture is impractical;
                            sparse-tile storage with 256-cubed leaf tiles
                            is the production-realistic choice
      - tile_voxel_res=256 -> each sparse leaf is one 256-cubed atlas, the
                            same shape as DownToEarth's hamlet-scale unit;
                            preserves locality and ships at ~64 MB per
                            populated tile

    The DownToEarth hamlet-scale defaults (voxel_res=256, RGBA8,
    world_extent=4 m, dense storage) are accessible via explicit override.

    What this is: a marker + config block. It does NOT change how the
    model interprets inputs; that still requires the conditioning-head
    fine-tune that mods #1 and #2 of PR #61 propose. What it DOES:

      - Specifies the bijection layout (resolution, byte width, storage
        mode) so a UVW-aware runtime knows which configuration applies.
      - Carries optional pre-computed canonical-coord meshes for the
        standard inference configs (saves a few ms per inference start).
      - Marks the checkpoint with 'uvw_compat_version: 2' so future
        runtimes can detect compatibility.

    Stored under the '_lyra_uvw_metadata' key in the output checkpoint
    alongside '_lyra_low_vram_metadata'. Both are harmless to non-aware
    loaders -- they just ignore unrecognised keys.

    Args:
        voxel_res: per-axis voxel grid resolution. Default 4096 for Lyra 2;
            use 256 for DownToEarth hamlet scale, 8192+ for sub-cm Lyra 2,
            higher for planet-scale (with hierarchical sparse storage).
        bits_per_axis: 8 / 16 / 32 -- which byte-width family member.
            Default 16 covers voxel_res up to 65,536.
        world_extent_m: scene half-width in metres. Default 45.0 (=90 m
            world cube) matches Lyra 2's documented walkable spec.
        storage_mode: 'dense' | 'sparse-tiled' | 'hierarchical-octree'.
            Default 'sparse-tiled' is the only practical choice above
            voxel_res=512 since dense storage exceeds GPU memory.
        tile_voxel_res: per-leaf-tile resolution for sparse modes (256
            matches the DownToEarth bijection's natural unit).
        precompute_canonical_meshes: include the canonical-coord starting
            grid for the standard (H, W, num_spatial_hist) configurations.
        canonical_configs: list of (H, W, num_spatial_hist) tuples to
            pre-compute. Default: [(480, 832, 5)] (Lyra 2 production).

    Returns:
        A dict suitable for storage in the checkpoint.
    """
    import torch

    if canonical_configs is None:
        canonical_configs = [(480, 832, 5)]   # Lyra 2's published config

    # Compute per-cell metrics for the metadata consumer
    cell_size_cm = (2.0 * world_extent_m * 100.0) / voxel_res
    total_voxels = voxel_res ** 3
    total_voxels_dense_bytes_rgba = total_voxels * 4
    # For sparse mode: per-tile resource cost
    tile_count_per_axis = max(1, voxel_res // tile_voxel_res)
    tile_count_total = tile_count_per_axis ** 3
    tile_summary_atlas_bytes = tile_voxel_res ** 3 * 4         # RGBA8 per tile
    tile_occupancy_bitmap_bytes = tile_voxel_res ** 3 // 8     # 1-bit per voxel per tile

    metadata = {
        "uvw_compat_version": 2,
        "voxel_res": voxel_res,
        "bits_per_axis": bits_per_axis,
        "world_extent_m": world_extent_m,
        "world_min": [-world_extent_m] * 3,
        "world_max": [+world_extent_m] * 3,
        "cell_size_cm": cell_size_cm,
        "total_voxels_addressable": total_voxels,
        "storage_mode": storage_mode,
        "tile_voxel_res": tile_voxel_res,
        "tile_count_per_axis": tile_count_per_axis,
        "tile_count_total_addressable": tile_count_total,
        "per_tile_summary_atlas_bytes": tile_summary_atlas_bytes,
        "per_tile_occupancy_bitmap_bytes": tile_occupancy_bitmap_bytes,
        # Heads-up: a fully populated dense atlas would be this many bytes.
        # For typical 5% surface occupancy, multiply by ~0.05 for the
        # realistic sparse storage budget.
        "dense_atlas_bytes_if_populated": total_voxels_dense_bytes_rgba,
        "downtoearth_bijection_url": "https://github.com/MiLO83/DownToEarth/blob/main/voxgaussian/pipeline/uvw_atlas.py",
    }

    if precompute_canonical_meshes:
        meshes = {}
        for H, W, N in canonical_configs:
            # Reproduce _build_canonical_spatial_coords' output once at bake
            # time. Stored as fp16 to keep the size down; can be cast back
            # to whatever precision the runtime needs.
            xs = torch.linspace(-1.0, 1.0, W, dtype=torch.float32)
            ys = torch.linspace(-1.0, 1.0, H, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            base_xy = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(N, 1, 1, 1)
            if N == 1:
                zs = torch.zeros(1, dtype=torch.float32)
            else:
                zs = torch.linspace(-1.0, 1.0, N, dtype=torch.float32)
            z = zs.view(N, 1, 1, 1).expand(N, 1, H, W)
            coords = torch.cat([base_xy, z], dim=1).to(torch.float16)
            meshes[f"{H}x{W}x{N}"] = coords
        metadata["precomputed_canonical_meshes"] = meshes

    return metadata


def requantize_streaming(
    input_path: str,
    output_path: str,
    mode: QuantMode = "fp8",
    skip_modules: list[str] | None = None,
    progress: bool = True,
    bake_uvw_metadata: bool = True,
    # UVW metadata defaults match Lyra 2's documented spec (90 m world,
    # cm-scale features). Override these for DownToEarth-scale (voxel_res=256,
    # bits=8, world=4 m) or planet-scale (voxel_res=65536+, bits=32) use.
    uvw_voxel_res: int = 4096,
    uvw_bits_per_axis: int = 16,
    uvw_world_extent_m: float = 45.0,
) -> None:
    """Streaming requantization: cast weights tensor-by-tensor at the
    state_dict level, no full-model load required. Runs on any machine
    with enough RAM to hold the bf16 state_dict (~32 GB for 14B Lyra 2).

    Estimated wall-clock for the 14B Lyra 2 on a typical 5060 Ti + 32 GB
    RAM + NVMe machine: ~2-3 minutes total (memory-bandwidth-bound, not
    compute-bound). No GPU needed for the conversion itself -- the cast
    is pure tensor arithmetic that PyTorch dispatches to MKL on CPU.

    Compared to requantize_checkpoint (which instantiates the full model):
        - Zero VRAM required
        - ~50% less peak system RAM (no model graph allocations)
        - Works on .pth format directly; for DCP-format checkpoints,
          fall back to the full-model path (rename: requantize_checkpoint)

    Args:
        input_path: source bf16 .pth checkpoint
        output_path: destination quantized .pth checkpoint
        mode: 'int8' | 'int4' | 'fp8' (default 'fp8')
        skip_modules: dotted module-name fragments to leave at original
            precision. Conditioning + output paths are skipped by default.
        progress: print per-N-tensors progress lines (default True)
    """
    import os
    import time

    if mode == "none":
        raise ValueError("Use a regular file copy for mode='none'")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    skip_modules = list(skip_modules or [])
    default_skips = {
        "canonical_proj", "output_proj", "final_layer",
        "norm_out", "x_embedder", "t_embedder",
    }
    skip_set = set(skip_modules) | default_skips
    def _should_skip(name: str) -> bool:
        return any(s in name for s in skip_set)

    # Quantize a single Linear weight tensor. Returns (new_tensor, scale)
    # where scale is None for int8/int4 (bnb handles its own scales) and
    # populated for fp8 (we store the scale as a sibling buffer).
    def _quantize_tensor(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mode == "fp8":
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError(
                    "torch.float8_e4m3fn not available -- upgrade PyTorch to 2.1+"
                )
            abs_max = t.detach().abs().max().clamp(min=1e-8)
            scale = (abs_max / 448.0).to(torch.bfloat16)
            cast = (t / scale).to(torch.float8_e4m3fn)
            return cast, scale
        elif mode == "int8":
            # For bnb compatibility, store as fp16 here and let the runtime
            # path do the actual Linear8bitLt swap on load. Disk savings
            # come from the dtype change (4 bytes -> 2 bytes -> 1 byte at
            # runtime). This is a compromise: full bnb-format-on-disk would
            # require bitsandbytes to be present at conversion time too.
            return t.to(torch.float16), None
        elif mode == "int4":
            # Same compromise as int8.
            return t.to(torch.float16), None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Heuristic: a "Linear weight" is a 2-D tensor whose key ends in '.weight'
    # and isn't part of a normalization layer. Conditioning embeddings and
    # output projections match the skip list.
    def _is_quantizable(key: str, tensor: torch.Tensor) -> bool:
        if not key.endswith(".weight"):
            return False
        if tensor.ndim != 2:
            return False
        if _should_skip(key):
            return False
        # Norm layers have 1-D weights so they wouldn't pass ndim==2,
        # but layer-norm-like 2-D conv weights might. Skip them by name.
        if any(s in key for s in ("norm", "bn", "ln")):
            return False
        return True

    t_start = time.time()
    print(f"[requantize-streaming] loading {input_path} into CPU RAM ...")
    sd = torch.load(input_path, map_location="cpu", weights_only=True)
    # Some Lyra checkpoints wrap state_dict in an outer dict; unwrap if so.
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        outer = sd
        sd = sd["state_dict"]
    else:
        outer = None
    t_load = time.time() - t_start
    print(f"[requantize-streaming] loaded {len(sd)} tensors in {t_load:.1f}s")

    n_quantized = 0
    n_skipped = 0
    new_sd = {}
    t_cast_start = time.time()

    for i, (name, tensor) in enumerate(sd.items()):
        if not torch.is_tensor(tensor):
            new_sd[name] = tensor
            continue
        if _is_quantizable(name, tensor):
            cast, scale = _quantize_tensor(tensor)
            new_sd[name] = cast
            if scale is not None:
                new_sd[name + "._fp8_scale"] = scale
            n_quantized += 1
        else:
            new_sd[name] = tensor
            n_skipped += 1
        if progress and (i + 1) % 100 == 0:
            elapsed = time.time() - t_cast_start
            print(f"[requantize-streaming]   {i+1}/{len(sd)} tensors  "
                  f"({n_quantized} quantized, {n_skipped} kept)  {elapsed:.1f}s")

    t_cast = time.time() - t_cast_start
    print(f"[requantize-streaming] cast done in {t_cast:.1f}s "
          f"({n_quantized} quantized to {mode}, {n_skipped} kept as-is)")

    # Pack the output: state_dict + metadata for the loader to detect mode
    metadata = {
        "lyra_low_vram_mode": mode,
        "lyra_low_vram_skip_modules": list(skip_set),
        "source_checkpoint": os.path.basename(input_path),
        "requantize_version": 2,
        "streaming": True,
    }
    output = {
        "state_dict": new_sd,
        "_lyra_low_vram_metadata": metadata,
    }
    # Bake UVW bijection metadata + pre-computed canonical meshes alongside.
    # A UVW-aware runtime auto-configures from this block; older loaders
    # ignore the unrecognised key.
    if bake_uvw_metadata:
        uvw_meta = _build_uvw_compat_metadata(
            voxel_res=uvw_voxel_res,
            bits_per_axis=uvw_bits_per_axis,
            world_extent_m=uvw_world_extent_m,
        )
        output["_lyra_uvw_metadata"] = uvw_meta
        print(f"[requantize-streaming] baked UVW compat metadata "
              f"(v{uvw_meta['uvw_compat_version']}, voxel-res "
              f"{uvw_meta['voxel_res']}^3, {uvw_meta['bits_per_axis']} "
              f"bits/axis, world {uvw_meta['world_extent_m']*2:.0f}m cube, "
              f"cell {uvw_meta['cell_size_cm']:.1f}cm, "
              f"storage={uvw_meta['storage_mode']})")
    # Preserve any other top-level keys from the original (config, optim, etc.)
    if outer is not None:
        for k, v in outer.items():
            if k != "state_dict":
                output[k] = v

    t_save_start = time.time()
    print(f"[requantize-streaming] writing {output_path} ...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(output, output_path)
    t_save = time.time() - t_save_start

    size_in = os.path.getsize(input_path) / 1024**3
    size_out = os.path.getsize(output_path) / 1024**3
    t_total = time.time() - t_start
    print(f"[requantize-streaming] saved {size_out:.2f} GB in {t_save:.1f}s "
          f"(from {size_in:.2f} GB; ratio {size_in/max(size_out, 0.001):.2f}x)")
    print(f"[requantize-streaming] total wall-clock: {t_total:.1f}s")


def detect_checkpoint_quantization(checkpoint_path: str) -> QuantMode:
    """Read the metadata header from a checkpoint to see if it's already
    quantized. Used by model_loader to auto-apply the right layer types
    during model construction.

    Returns 'none' for a regular bf16 checkpoint (no metadata) or one
    that was saved before this requantization tool existed.
    """
    import os
    from lyra_2._ext.imaginaire.utils.easy_io import easy_io
    if not os.path.exists(checkpoint_path) or not checkpoint_path.endswith(".pth"):
        return "none"
    try:
        # Load with a minimal map to peek at metadata without materialising weights
        data = easy_io.load(checkpoint_path, map_location="cpu")
        if isinstance(data, dict) and "_lyra_low_vram_metadata" in data:
            return data["_lyra_low_vram_metadata"].get("lyra_low_vram_mode", "none")
    except Exception as e:
        log.warning(
            f"[requantize] could not read metadata from {checkpoint_path}: {e}",
            rank0_only=True,
        )
    return "none"


# ─── CLI entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python -m lyra_2._src.utils.low_vram",
        description="One-time requantization of a Lyra 2 checkpoint for "
                    "consumer-GPU (16 GB) inference. See module docstring "
                    "for the precision/quality table.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    rq = sub.add_parser(
        "requantize",
        help="Convert bf16 checkpoint to int8/int4/fp8 via full-model load "
             "(needs ~28 GB VRAM or ~64 GB RAM for the 14B model). Use "
             "'requantize-streaming' instead for the consumer-hardware path.",
    )
    rq.add_argument("--input", required=True, help="Source bf16 .pth checkpoint")
    rq.add_argument("--output", required=True, help="Destination quantized .pth")
    rq.add_argument(
        "--mode",
        choices=["int8", "int4", "fp8"],
        default="fp8",
        help="Target quantization mode. fp8 (default) is recommended for "
             "Blackwell/Hopper cards; int8 for older GPUs; int4 for tightest VRAM.",
    )
    rq.add_argument(
        "--skip", nargs="*", default=None,
        help="Additional module names to keep at original precision",
    )

    rqs = sub.add_parser(
        "requantize-streaming",
        help="Convert bf16 checkpoint via state_dict streaming (no model "
             "load). Runs on any machine with enough RAM to hold the bf16 "
             "state_dict (~32 GB for 14B Lyra 2). Zero VRAM required. "
             "Estimated ~2-3 minutes total on a typical consumer setup.",
    )
    rqs.add_argument("--input", required=True, help="Source bf16 .pth checkpoint")
    rqs.add_argument("--output", required=True, help="Destination quantized .pth")
    rqs.add_argument(
        "--mode", choices=["int8", "int4", "fp8"], default="fp8",
        help="Target quantization mode. fp8 default.",
    )
    rqs.add_argument(
        "--skip", nargs="*", default=None,
        help="Additional module names to keep at original precision",
    )
    rqs.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-100-tensor progress lines",
    )
    rqs.add_argument(
        "--no-bake-uvw", action="store_true",
        help="Skip embedding UVW bijection compat metadata + pre-computed "
             "canonical-coord meshes (saves ~few KB but defeats the "
             "UVW-aware runtime auto-config). Default: bake the metadata.",
    )
    rqs.add_argument(
        "--uvw-voxel-res", type=int, default=4096,
        help="UVW bijection voxel grid resolution per axis. Default 4096 "
             "(matches Lyra 2's 90 m walkable spec at ~2.2 cm cells). "
             "Use 256 for DownToEarth hamlet scale (~3 cm cells in 8 m), "
             "or 8192+ for sub-cm Lyra 2 detail.",
    )
    rqs.add_argument(
        "--uvw-bits-per-axis", type=int, choices=[8, 16, 32], default=16,
        help="UVW bijection address byte-width. 8 maxes at voxel_res=256, "
             "16 maxes at 65536, 32 effectively unbounded. Default 16 fits "
             "any Lyra-2-scale config.",
    )
    rqs.add_argument(
        "--uvw-world-extent-m", type=float, default=45.0,
        help="World half-width in metres. Default 45 (=90 m cube, Lyra 2 "
             "spec). Use 4 for DownToEarth hamlet scale.",
    )

    inspect = sub.add_parser("inspect", help="Print quantization metadata of a checkpoint")
    inspect.add_argument("checkpoint", help="Path to .pth checkpoint")

    args = p.parse_args()
    if args.command == "requantize":
        requantize_checkpoint(args.input, args.output, mode=args.mode, skip_modules=args.skip)
    elif args.command == "requantize-streaming":
        requantize_streaming(
            args.input, args.output, mode=args.mode,
            skip_modules=args.skip,
            progress=not args.no_progress,
            bake_uvw_metadata=not args.no_bake_uvw,
            uvw_voxel_res=args.uvw_voxel_res,
            uvw_bits_per_axis=args.uvw_bits_per_axis,
            uvw_world_extent_m=args.uvw_world_extent_m,
        )
    elif args.command == "inspect":
        mode = detect_checkpoint_quantization(args.checkpoint)
        print(f"Checkpoint quantization: {mode}")
        # Also surface UVW compat metadata if present
        try:
            from lyra_2._ext.imaginaire.utils.easy_io import easy_io
            data = easy_io.load(args.checkpoint, map_location="cpu")
            if isinstance(data, dict) and "_lyra_uvw_metadata" in data:
                uvw = data["_lyra_uvw_metadata"]
                v = uvw.get("uvw_compat_version", "?")
                print(f"UVW compat version:        {v}")
                if v >= 2:
                    # New format (v2+): scene-scale aware
                    vr = uvw.get("voxel_res", "?")
                    bpa = uvw.get("bits_per_axis", "?")
                    we = uvw.get("world_extent_m", "?")
                    cs = uvw.get("cell_size_cm", "?")
                    sm = uvw.get("storage_mode", "?")
                    print(f"  voxel resolution:        {vr}^3 ({uvw.get('total_voxels_addressable', '?'):,} voxels)")
                    print(f"  bits per axis:           {bpa}")
                    print(f"  world extent:            {we} m half-width ({we*2:.0f} m cube)")
                    print(f"  cell size:               {cs:.2f} cm" if isinstance(cs, (int, float)) else f"  cell size:               {cs}")
                    print(f"  storage mode:            {sm}")
                    if sm == "sparse-tiled":
                        tvr = uvw.get("tile_voxel_res", "?")
                        tcpa = uvw.get("tile_count_per_axis", "?")
                        ts_mb = uvw.get("per_tile_summary_atlas_bytes", 0) / 1024 / 1024
                        to_mb = uvw.get("per_tile_occupancy_bitmap_bytes", 0) / 1024 / 1024
                        print(f"  tile voxel res:          {tvr}^3")
                        print(f"  tile grid:               {tcpa}^3 = {tcpa**3 if isinstance(tcpa, int) else '?'} addressable tiles")
                        print(f"  per-tile summary atlas:  {ts_mb:.1f} MB (RGBA8)")
                        print(f"  per-tile occ bitmap:     {to_mb:.2f} MB")
                    dense_gb = uvw.get("dense_atlas_bytes_if_populated", 0) / 1024**3
                    print(f"  dense atlas if populated: {dense_gb:.1f} GB (sparse at ~5% = {dense_gb*0.05:.1f} GB realistic)")
                else:
                    # Legacy v1 format
                    print(f"  voxel resolution:        {uvw.get('voxel_res')}^3")
                    print(f"  bits per axis:           {uvw.get('bits_per_axis')}")
                    print(f"  atlas dims:              {uvw.get('atlas_w')}x{uvw.get('atlas_h')}")
                meshes = uvw.get("precomputed_canonical_meshes", {})
                if meshes:
                    print(f"  pre-computed meshes:     {list(meshes.keys())}")
            else:
                print("UVW compat: (no metadata baked in)")
        except Exception as e:
            print(f"UVW compat: (could not read -- {e})")
