"""
Reference implementation of the per-frame demand-streaming loop.

The 2-bit OccupancyBitmap stores (occupancy, render-flag) per voxel.
The runtime renderer sets the render-flag via imageAtomicOr on first-hit
per pixel; the streamer reads the per-chunk OR of those bits to decide
which chunks to keep resident in VRAM.

This module gives a pure-Python reference implementation of the
streamer side: a chunk cache with hysteresis-based eviction, an
async loader interface, and the canonical per-frame loop. It runs
without a GPU and is intended as documentation-by-code for anyone
implementing the runtime side.

Classes:
    ChunkCache       — VRAM-resident chunk store with hysteresis
    ChunkLoader      — abstract async loader interface
    DiskChunkLoader  — concrete loader from a directory of chunk files
    StreamingSession — per-frame loop wrapper

Example:
    >>> from lyra_2._src.datasets.uvw_atlas import OccupancyBitmap
    >>> bitmap = OccupancyBitmap(atlas_w=4096, atlas_h=4096, bits_per_voxel=2)
    >>> loader = DiskChunkLoader("/path/to/chunks/")
    >>> cache = ChunkCache(loader, max_resident=200, evict_after_n_cold_frames=60)
    >>> session = StreamingSession(bitmap, cache)
    >>>
    >>> for frame_idx in range(num_frames):
    ...     with session.frame():
    ...         pass  # raymarcher runs here; sets touched bits via shader
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol


# ───────────────────────── Loader interface ────────────────────────────────

class ChunkLoader(Protocol):
    """Abstract loader. Concrete impls fetch chunks from disk / network."""

    def fetch(self, chunk_id: tuple) -> bytes:
        """Block until chunk_id is loaded; return its bytes."""
        ...

    def fetch_async(self, chunk_id: tuple) -> "Future[bytes]":
        """Schedule a non-blocking load; return a Future."""
        ...


class DiskChunkLoader:
    """Reference loader that reads chunks from a directory.

    File naming convention: chunks/cx_cy_cz.bin

    Threading model: a small ThreadPoolExecutor for parallel reads.
    NVMe is fine with high read-queue depth; spinning disks aren't.
    """

    def __init__(self, root: str | Path, n_threads: int = 4):
        self.root = Path(root)
        self.executor = ThreadPoolExecutor(max_workers=n_threads)

    def _path(self, chunk_id: tuple) -> Path:
        cx, cy, cz = chunk_id
        return self.root / f"{cx}_{cy}_{cz}.bin"

    def fetch(self, chunk_id: tuple) -> bytes:
        return self._path(chunk_id).read_bytes()

    def fetch_async(self, chunk_id: tuple) -> "Future[bytes]":
        return self.executor.submit(self.fetch, chunk_id)


# ───────────────────────── Chunk cache ─────────────────────────────────────

@dataclass
class CachedChunk:
    chunk_id: tuple
    data: bytes
    last_touched_frame: int = 0
    cold_frames: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(self.data)


class ChunkCache:
    """
    VRAM-resident chunk store with hysteresis-based eviction.

    Policy:
      - LRU-ordered map of chunk_id -> CachedChunk
      - Frame counter increments each frame_complete()
      - On mark_touched(): chunk.last_touched_frame = current_frame,
        chunk.cold_frames = 0
      - On frame_complete(): for each NOT-touched chunk, cold_frames += 1
      - Eviction trigger: cold_frames > evict_after_n_cold_frames
      - Optional max_resident cap forces LRU eviction even of "warm" chunks
        when the cache exceeds size

    Async loading:
      - request(chunk_id) returns a Future or completed chunk if already
        resident. Background loader fills in.
      - mark_touched() also requests the chunk if not resident
    """

    def __init__(self, loader: ChunkLoader, max_resident: int = 200,
                 evict_after_n_cold_frames: int = 60):
        self.loader = loader
        self.max_resident = max_resident
        self.evict_after_n_cold_frames = evict_after_n_cold_frames
        self._resident: OrderedDict[tuple, CachedChunk] = OrderedDict()
        self._pending: dict[tuple, "Future[bytes]"] = {}
        self._lock = threading.Lock()
        self._current_frame = 0
        self._evicted_count = 0
        self._loaded_count = 0

    # ---- mutations ----

    def mark_touched(self, chunk_id: tuple) -> Optional[CachedChunk]:
        """
        Mark a chunk as touched this frame. Returns its data if resident,
        None if it's still being loaded (renderer should use LOD fallback).
        Triggers async load if missing.
        """
        with self._lock:
            chunk = self._resident.get(chunk_id)
            if chunk is not None:
                chunk.last_touched_frame = self._current_frame
                chunk.cold_frames = 0
                self._resident.move_to_end(chunk_id)   # LRU bump
                return chunk
            if chunk_id not in self._pending:
                self._pending[chunk_id] = self.loader.fetch_async(chunk_id)
            return None

    def mark_touched_many(self, chunk_ids: Iterable[tuple]) -> None:
        for cid in chunk_ids:
            self.mark_touched(cid)

    def _ingest_pending(self) -> None:
        """Move completed pending fetches into resident set."""
        finished = [(cid, fut) for cid, fut in self._pending.items() if fut.done()]
        for cid, fut in finished:
            try:
                data = fut.result()
                self._resident[cid] = CachedChunk(
                    chunk_id=cid, data=data,
                    last_touched_frame=self._current_frame, cold_frames=0,
                )
                self._resident.move_to_end(cid)
                self._loaded_count += 1
            except Exception:
                pass  # log in production
            del self._pending[cid]

    def _evict_cold(self) -> None:
        """Remove chunks that haven't been touched recently."""
        to_remove = [
            cid for cid, chunk in self._resident.items()
            if chunk.cold_frames > self.evict_after_n_cold_frames
        ]
        for cid in to_remove:
            del self._resident[cid]
            self._evicted_count += 1

    def _evict_lru_if_over_cap(self) -> None:
        """Force LRU eviction when cache is over max_resident."""
        while len(self._resident) > self.max_resident:
            cid, _ = self._resident.popitem(last=False)   # oldest
            self._evicted_count += 1

    def frame_complete(self) -> None:
        """Call once per frame, after rendering + mark_touched calls."""
        with self._lock:
            self._ingest_pending()
            for chunk in self._resident.values():
                if chunk.last_touched_frame != self._current_frame:
                    chunk.cold_frames += 1
            self._evict_cold()
            self._evict_lru_if_over_cap()
            self._current_frame += 1

    # ---- queries ----

    @property
    def resident_count(self) -> int:
        with self._lock:
            return len(self._resident)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(c.size_bytes for c in self._resident.values())

    def stats(self) -> dict:
        with self._lock:
            return {
                "resident": len(self._resident),
                "pending": len(self._pending),
                "evicted_total": self._evicted_count,
                "loaded_total": self._loaded_count,
                "frame": self._current_frame,
                "bytes_resident": sum(c.size_bytes for c in self._resident.values()),
            }


# ───────────────────────── Per-frame session ──────────────────────────────

class StreamingSession:
    """
    Wraps the per-frame demand-streaming loop. Use as a context manager
    inside the renderer's frame loop.

    Sequence:
        1. enter: bitmap.clear_touched()
        2. user code: raymarcher runs, sets touched bits on first-hit
        3. exit:  touched_chunks = bitmap.touched_chunks()
                  cache.mark_touched_many(touched_chunks)
                  cache.frame_complete()
    """

    def __init__(self, bitmap, cache: ChunkCache,
                 chunk_size: int = 16, res: int = 256,
                 tiles_per_row: int = 16):
        self.bitmap = bitmap
        self.cache = cache
        self.chunk_size = chunk_size
        self.res = res
        self.tiles_per_row = tiles_per_row

    @contextmanager
    def frame(self):
        # Pre-render: clear render-flag bits, keep occupancy bits.
        self.bitmap.clear_touched()
        try:
            yield self
        finally:
            # Post-render: OR-reduce touched bits to chunk keys; tell the
            # cache; advance frame counter (which evicts cold chunks).
            touched = self.bitmap.touched_chunks(
                chunk_size=self.chunk_size,
                res=self.res,
                tiles_per_row=self.tiles_per_row,
            )
            self.cache.mark_touched_many(touched)
            self.cache.frame_complete()


__all__ = ["ChunkLoader", "DiskChunkLoader", "CachedChunk",
           "ChunkCache", "StreamingSession"]
