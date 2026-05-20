"""
Reference shader bindings for the OccupancyBitmap 2-bit pattern.

This module provides ready-to-paste GLSL / WGSL / CUDA snippets for
wiring the 2-bit OccupancyBitmap (`lyra_2/_src/datasets/uvw_atlas.py`)
into a runtime voxel renderer.

The 2-bit bitmap encodes (occupancy, render-flag) per voxel, addressed
via the UVW bijection — voxel coord (u, v, w) → atlas pixel (ax, ay).
Bit layout per byte (4 voxels packed):

    voxel n: bit (n*2)     = occupancy
             bit (n*2 + 1) = render-flag (touched this frame)

    byte = [v0_occ | v0_flag | v1_occ | v1_flag | v2_occ | v2_flag | v3_occ | v3_flag]
              MSB                                                              LSB

The runtime per-frame loop is:

    1. clear_touched()                # masked-AND, preserves occupancy
    2. raymarch every screen pixel    # imageAtomicOr the flag on each first-hit
    3. touched_chunks()               # OR-reduce to chunk keys → streamer

The snippets below are reference implementations. Drop into your
renderer of choice; the same UVW lookup serves both reads (sample
voxel RGBA from atlas at (ax, ay)) and writes (set flag bit at
(ax >> 2, ay)).
"""

# ───────────────────────── GLSL (OpenGL / WebGPU) ──────────────────────────

GLSL_VOXEL_TO_ATLAS = """
// UVW bijection — voxel grid coord (u, v, w) → atlas pixel (ax, ay).
// Inverse: atlas_to_voxel below.
//
// Defaults: res=256, tiles_per_row=16. A 256x256 voxel grid packs into
// a 4096x4096 atlas as a 16x16 grid of 256x256 tiles, one tile per
// W-plane.
const int RES = 256;
const int TILES_PER_ROW = 16;

ivec2 voxel_to_atlas(ivec3 voxel_xyz) {
    int u = voxel_xyz.x;
    int v = voxel_xyz.y;
    int w = voxel_xyz.z;
    int tile_col = w % TILES_PER_ROW;
    int tile_row = w / TILES_PER_ROW;
    return ivec2(tile_col * RES + u, tile_row * RES + v);
}

ivec3 atlas_to_voxel(ivec2 atlas_xy) {
    int u = atlas_xy.x % RES;
    int v = atlas_xy.y % RES;
    int tile_col = atlas_xy.x / RES;
    int tile_row = atlas_xy.y / RES;
    int w = tile_row * TILES_PER_ROW + tile_col;
    return ivec3(u, v, w);
}
"""

GLSL_OCCUPANCY_2BIT_READ = """
// Read occupancy + render-flag for a voxel via the 2-bit bitmap.
// stateTexture is bound as usampler2D (R8UI) with dimensions
// (atlas_w / 4, atlas_h). One texelFetch per voxel; one shift+AND
// per bit query.
uniform usampler2D stateTexture;

bool is_occupied(ivec3 voxel_xyz) {
    ivec2 a = voxel_to_atlas(voxel_xyz);
    uint byte_v = texelFetch(stateTexture, ivec2(a.x >> 2, a.y), 0).r;
    return ((byte_v >> ((a.x & 3) * 2)) & 1u) == 1u;
}

bool is_rendered_this_frame(ivec3 voxel_xyz) {
    ivec2 a = voxel_to_atlas(voxel_xyz);
    uint byte_v = texelFetch(stateTexture, ivec2(a.x >> 2, a.y), 0).r;
    return ((byte_v >> ((a.x & 3) * 2 + 1)) & 1u) == 1u;
}
"""

GLSL_RENDER_FLAG_WRITE = """
// Mark a voxel as touched (render-flag = 1) for this frame. Called by
// the raymarcher on first-hit per pixel. stateImage is the SAME memory
// as stateTexture, bound as a writable imageBuffer / image2D for
// atomic ops.
//
// Requires:
//   GL_ARB_shader_image_load_store or GLSL 4.20+
//   layout(r8ui) binding for atomic ops on the image
//
// Note: imageAtomicOr is NOT supported in WebGL2 fragment shaders.
// For WebGL2, use a separate render-to-id-target pass and CPU readback.
// For WebGPU / Vulkan / OpenGL 4.5+, this works in fragment shaders.

layout(r8ui) uniform uimage2D stateImage;

void mark_touched(ivec3 voxel_xyz) {
    ivec2 a = voxel_to_atlas(voxel_xyz);
    uint flag_bit = 1u << ((a.x & 3) * 2 + 1);
    imageAtomicOr(stateImage, ivec2(a.x >> 2, a.y), flag_bit);
}
"""

GLSL_DDA_RAYMARCH = """
// Reference DDA voxel-raymarch loop using the 2-bit bitmap.
// Returns the RGBA of the first occupied voxel hit, or fog color.
//
// Inputs:
//   atlasTexture - the main RGBA8 voxel atlas (sampler2D, R8UI for state)
//   ray_origin, ray_dir - in voxel grid coords
//
// On first hit:
//   - read voxel color from atlasTexture at the SAME atlas address
//   - mark_touched() sets the render-flag bit
//   - return color

uniform sampler2D atlasTexture;   // RGBA8 voxel data
const int MAX_STEPS = 384;

vec4 raymarch_voxels(vec3 ray_origin, vec3 ray_dir) {
    ivec3 v = ivec3(floor(ray_origin));
    vec3 step_dir = sign(ray_dir);
    vec3 rdInv = 1.0 / max(abs(ray_dir), vec3(1e-6));
    vec3 tMax = abs((vec3(v) + max(step_dir, 0.0) - ray_origin) * rdInv);
    vec3 tDelta = rdInv;

    for (int i = 0; i < MAX_STEPS; i++) {
        if (is_occupied(v)) {
            mark_touched(v);
            ivec2 a = voxel_to_atlas(v);   // SAME bijection lookup
            return texelFetch(atlasTexture, a, 0);
        }
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                v.x += int(step_dir.x); tMax.x += tDelta.x;
            } else {
                v.z += int(step_dir.z); tMax.z += tDelta.z;
            }
        } else {
            if (tMax.y < tMax.z) {
                v.y += int(step_dir.y); tMax.y += tDelta.y;
            } else {
                v.z += int(step_dir.z); tMax.z += tDelta.z;
            }
        }
    }
    return vec4(0.5, 0.7, 0.9, 1.0);   // sky fallback
}
"""


# ───────────────────────── WGSL (WebGPU) ───────────────────────────────────

WGSL_VOXEL_TO_ATLAS = """
// WebGPU version of voxel_to_atlas. Same logic as the GLSL above.
const RES: i32 = 256;
const TILES_PER_ROW: i32 = 16;

fn voxel_to_atlas(voxel_xyz: vec3<i32>) -> vec2<i32> {
    let tile_col = voxel_xyz.z % TILES_PER_ROW;
    let tile_row = voxel_xyz.z / TILES_PER_ROW;
    return vec2<i32>(tile_col * RES + voxel_xyz.x, tile_row * RES + voxel_xyz.y);
}
"""

WGSL_OCCUPANCY_2BIT = """
// WebGPU storage texture for the 2-bit bitmap. Atomic ops on storage
// textures are supported in compute shaders and (in newer specs)
// fragment shaders with the appropriate extension.

@group(0) @binding(0) var<storage, read_write> stateBuffer: array<atomic<u32>>;

fn is_occupied(voxel_xyz: vec3<i32>) -> bool {
    let a = voxel_to_atlas(voxel_xyz);
    // Pack 4 bytes per u32 word; voxel n at word (n>>2), within-word position varies
    let voxel_index_in_atlas = a.y * (atlas_w / 4) + (a.x >> 2);
    let byte_in_word = a.x & 3;  // 0..3
    let word = atomicLoad(&stateBuffer[voxel_index_in_atlas]);
    let bit_pos = (byte_in_word * 8) + ((a.x & 3) * 2);  // careful: 2-bit packing
    return ((word >> u32(bit_pos)) & 1u) == 1u;
}

fn mark_touched(voxel_xyz: vec3<i32>) {
    let a = voxel_to_atlas(voxel_xyz);
    let voxel_index_in_atlas = a.y * (atlas_w / 4) + (a.x >> 2);
    let bit_pos = (a.x & 3) * 2 + 1;
    atomicOr(&stateBuffer[voxel_index_in_atlas], 1u << u32(bit_pos));
}
"""


# ───────────────────────── CUDA (for native runtime) ───────────────────────

CUDA_VOXEL_TO_ATLAS = """
// CUDA helper for the voxel<->atlas bijection. Use in a __device__
// function inside your custom raymarcher kernel.
__device__ __forceinline__
int2 voxel_to_atlas(int u, int v, int w, int res=256, int tiles_per_row=16) {
    int tile_col = w % tiles_per_row;
    int tile_row = w / tiles_per_row;
    return make_int2(tile_col * res + u, tile_row * res + v);
}
"""

CUDA_TOUCH_BITMAP = """
// CUDA: atomic-or the render-flag bit for voxel (u, v, w).
// state_bitmap is a uint8_t* with row-stride = atlas_w / 4.
__device__ __forceinline__
void mark_touched_cuda(uint8_t* state_bitmap, int row_stride,
                       int u, int v, int w) {
    int2 a = voxel_to_atlas(u, v, w);
    int byte_idx = a.y * row_stride + (a.x >> 2);
    uint8_t flag_bit = 1u << ((a.x & 3) * 2 + 1);
    atomicOr((unsigned int*)((uintptr_t)(state_bitmap + byte_idx) & ~3u),
             ((unsigned int)flag_bit) << (((uintptr_t)(state_bitmap + byte_idx) & 3u) * 8));
}
"""


# ───────────────────────── per-frame Python orchestrator ───────────────────

PYTHON_FRAME_LOOP = """
# Reference per-frame loop combining clear_touched + render + readback.
# This is the CPU side; the GPU side does the actual raymarching via
# the GLSL/WGSL/CUDA above.

from lyra_2._src.datasets.uvw_atlas import OccupancyBitmap

def frame(bitmap: OccupancyBitmap, renderer, streamer):
    # 1. Zero the render-flag bits (preserves occupancy via masked AND).
    #    ~10 us for a 4 MB bitmap on a 5060 Ti.
    bitmap.clear_touched()

    # 2. Render the frame. The raymarcher writes the touched bits via
    #    imageAtomicOr on the same uint8 buffer.
    renderer.render(bitmap)

    # 3. Read back which chunks the rays actually visited this frame.
    #    OR-reduces the per-voxel touched bitmap to chunk granularity.
    touched_chunks = bitmap.touched_chunks(chunk_size=16)

    # 4. Update the streamer: load missing chunks, mark cold chunks for
    #    eviction. With hysteresis (don't evict on first miss).
    streamer.update(touched_chunks)
"""


# ───────────────────────── Convenience ─────────────────────────────────────

def all_snippets() -> dict:
    """Return all reference shader / kernel / orchestrator snippets as
    a dict keyed by name. Useful for pasting into a renderer's docs."""
    return {
        "GLSL_VOXEL_TO_ATLAS":        GLSL_VOXEL_TO_ATLAS,
        "GLSL_OCCUPANCY_2BIT_READ":   GLSL_OCCUPANCY_2BIT_READ,
        "GLSL_RENDER_FLAG_WRITE":     GLSL_RENDER_FLAG_WRITE,
        "GLSL_DDA_RAYMARCH":          GLSL_DDA_RAYMARCH,
        "WGSL_VOXEL_TO_ATLAS":        WGSL_VOXEL_TO_ATLAS,
        "WGSL_OCCUPANCY_2BIT":        WGSL_OCCUPANCY_2BIT,
        "CUDA_VOXEL_TO_ATLAS":        CUDA_VOXEL_TO_ATLAS,
        "CUDA_TOUCH_BITMAP":          CUDA_TOUCH_BITMAP,
        "PYTHON_FRAME_LOOP":          PYTHON_FRAME_LOOP,
    }


if __name__ == "__main__":
    # Print all snippets for inspection / docs generation.
    for name, snippet in all_snippets().items():
        print(f"\n=== {name} ===")
        print(snippet)
