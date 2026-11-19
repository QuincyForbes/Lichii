// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions from nxnative (https://github.com/seedds/nxnative), Copyright (c) 2026 seedds, MIT
// License (LICENSES/MIT.txt): rb_swap, rb_swap_rows and the unwritten-attachment store rule.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_recorder.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_switches.h"

namespace Metal {

namespace {

/// Apple GPUs' largest render target side.
constexpr u32 MaxRenderTargetSize = 16384;

/// A stage's argument tables as Metal's validation layer names them on Apple's own GPUs (a Mac's
/// other GPUs take 128 textures).
constexpr u32 BufferSlots = 31;
constexpr u32 TextureSlots = 96;
constexpr u32 SamplerSlots = 16;

/// EDEN_NX2_METAL_UNWRITTEN_STORE (on unless 0), on Apple GPUs: an attachment a pass loads and
/// no draw of it can write is not stored again (MTLStoreActionDontCare; the tile it would write
/// back holds what memory has already). nxnative's DM_NVN_UNWRITTEN_DC (MIT, nvn_mtl.m sd_finish).
bool UnwrittenStoresEnabled(const Device& device) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_UNWRITTEN_STORE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled && device.AppleFamily() != 0;
}

/// EDEN_NX2_MSAA_DISCARD=0 (as Vulkan's): MSAA passes keep their samples even when they begin
/// with a clear; otherwise such a pass resolves into the image's shadow and does not store them.
bool MsaaDiscardEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_MSAA_DISCARD");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Metal checks a slot past a stage's tables in its validation layer only.
bool SlotFits(u32 slot, u32 slots, const char* kind) {
    if (slot < slots || !PipelineCheckEnabled()) {
        return true;
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Metal, "Metal: {} slot {} not bound, a stage has {}", kind, slot,
                    slots);
    }
    return false;
}

constexpr const char* BlitSource = R"(
#include <metal_stdlib>
using namespace metal;

struct BlitOut {
    float4 position [[position]];
    float2 uv;
};

// One triangle over the viewport; uv_rect is the source rectangle (normalized) at its corners.
vertex BlitOut blit_vertex(uint id [[vertex_id]], constant float4& uv_rect [[buffer(0)]]) {
    const float2 corner = float2(float((id << 1) & 2), float(id & 2));
    BlitOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = mix(uv_rect.xy, uv_rect.zw, corner);
    return out;
}

fragment float4 blit_float(BlitOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                           sampler smp [[sampler(0)]]) {
    return src.sample(smp, in.uv);
}

fragment uint4 blit_uint(BlitOut in [[stage_in]], texture2d<uint> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
}

fragment int4 blit_sint(BlitOut in [[stage_in]], texture2d<int> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
}

struct DepthOut {
    float depth [[depth(any)]];
};

fragment DepthOut blit_depth(BlitOut in [[stage_in]], depth2d<float> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    DepthOut out;
    out.depth = src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
    return out;
}

struct DepthStencilOut {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

// The stencil goes out as the reference value of a Replace stencil operation.
fragment DepthStencilOut blit_depth_stencil(BlitOut in [[stage_in]],
                                            depth2d<float> src [[texture(0)]],
                                            texture2d<uint> src_stencil [[texture(1)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    const uint2 texel = uint2(clamp(in.uv * size, float2(0.0), size - 1.0));
    DepthStencilOut out;
    out.depth = src.read(texel);
    out.stencil = src_stencil.read(texel).x;
    return out;
}

struct ClearOut {
    float4 position [[position]];
};

// One triangle over the viewport at the depth a clear writes.
vertex ClearOut clear_vertex(uint id [[vertex_id]], constant float& depth [[buffer(0)]]) {
    const float2 corner = float2(float((id << 1) & 2), float(id & 2));
    ClearOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, depth, 1.0);
    return out;
}

fragment float4 clear_float(constant float4& color [[buffer(0)]]) {
    return color;
}

fragment uint4 clear_uint(constant uint4& color [[buffer(0)]]) {
    return color;
}

fragment int4 clear_sint(constant int4& color [[buffer(0)]]) {
    return color;
}

fragment void clear_none() {}

#define CLEAR_RT_FN(idx) \
    struct ClearFloatRt##idx { float4 c [[color(idx)]]; }; \
    fragment ClearFloatRt##idx clear_float_rt##idx(constant float4& color [[buffer(0)]]) { return {color}; } \
    struct ClearUintRt##idx { uint4 c [[color(idx)]]; }; \
    fragment ClearUintRt##idx clear_uint_rt##idx(constant uint4& color [[buffer(0)]]) { return {color}; } \
    struct ClearSintRt##idx { int4 c [[color(idx)]]; }; \
    fragment ClearSintRt##idx clear_sint_rt##idx(constant int4& color [[buffer(0)]]) { return {color}; }
CLEAR_RT_FN(0) CLEAR_RT_FN(1) CLEAR_RT_FN(2) CLEAR_RT_FN(3)
CLEAR_RT_FN(4) CLEAR_RT_FN(5) CLEAR_RT_FN(6) CLEAR_RT_FN(7)
#undef CLEAR_RT_FN

struct ClearFloatMrt {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
    float4 c5 [[color(5)]];
    float4 c6 [[color(6)]];
    float4 c7 [[color(7)]];
};

fragment ClearFloatMrt clear_float_mrt(constant float4* colors [[buffer(0)]]) {
    return {colors[0], colors[1], colors[2], colors[3],
            colors[4], colors[5], colors[6], colors[7]};
}

// Sample copies: the guest sees a multisampled image as a plain one holding sample s of pixel p
// at texel (p << shift) | (s & (2^shift.x - 1), s >> shift.x), as Vulkan's MSAA copy passes.
struct SampleParams {
    uint2 shift;
    uint samples;
    uint layer;
};

uint SampleOf(uint2 texel, constant SampleParams& p) {
    const uint2 mask = (uint2(1) << p.shift) - 1;
    return min((texel.x & mask.x) | ((texel.y & mask.y) << p.shift.x), p.samples - 1);
}

uint2 TexelOf(uint2 pixel, uint sample_index, constant SampleParams& p) {
    const uint mask_x = (1u << p.shift.x) - 1;
    return (pixel << p.shift) | uint2(sample_index & mask_x, sample_index >> p.shift.x);
}

fragment float4 expand_float(ClearOut in [[stage_in]], texture2d_ms_array<float> src [[texture(0)]],
                             constant SampleParams& p [[buffer(0)]]) {
    const uint2 texel = uint2(in.position.xy);
    return src.read(texel >> p.shift, p.layer, SampleOf(texel, p));
}

fragment uint4 expand_uint(ClearOut in [[stage_in]], texture2d_ms_array<uint> src [[texture(0)]],
                           constant SampleParams& p [[buffer(0)]]) {
    const uint2 texel = uint2(in.position.xy);
    return src.read(texel >> p.shift, p.layer, SampleOf(texel, p));
}

fragment int4 expand_sint(ClearOut in [[stage_in]], texture2d_ms_array<int> src [[texture(0)]],
                          constant SampleParams& p [[buffer(0)]]) {
    const uint2 texel = uint2(in.position.xy);
    return src.read(texel >> p.shift, p.layer, SampleOf(texel, p));
}

fragment DepthOut expand_depth(ClearOut in [[stage_in]],
                               depth2d_ms_array<float> src [[texture(0)]],
                               constant SampleParams& p [[buffer(0)]]) {
    const uint2 texel = uint2(in.position.xy);
    DepthOut out;
    out.depth = src.read(texel >> p.shift, p.layer, SampleOf(texel, p));
    return out;
}

fragment float4 compress_float(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                               texture2d<float> src [[texture(0)]],
                               constant SampleParams& p [[buffer(0)]]) {
    return src.read(TexelOf(uint2(in.position.xy), sample_index, p));
}

fragment uint4 compress_uint(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                             texture2d<uint> src [[texture(0)]],
                             constant SampleParams& p [[buffer(0)]]) {
    return src.read(TexelOf(uint2(in.position.xy), sample_index, p));
}

fragment int4 compress_sint(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                            texture2d<int> src [[texture(0)]],
                            constant SampleParams& p [[buffer(0)]]) {
    return src.read(TexelOf(uint2(in.position.xy), sample_index, p));
}

// Format conversions between depth and colour images, as Vulkan's convert_*.frag: texel for
// texel, the destination's size.
fragment float4 convert_depth_to_color(ClearOut in [[stage_in]], depth2d<float> src [[texture(0)]]) {
    const float depth = src.read(uint2(in.position.xy));
    return float4(depth, depth, depth, 1.0);
}

fragment DepthOut convert_color_to_depth(ClearOut in [[stage_in]],
                                         texture2d<float> src [[texture(0)]]) {
    DepthOut out;
    out.depth = src.read(uint2(in.position.xy)).r;
    return out;
}

fragment DepthOut convert_abgr8_to_d32f(ClearOut in [[stage_in]],
                                        texture2d<float> src [[texture(0)]]) {
    const float4 color = src.read(uint2(in.position.xy)).abgr;
    DepthOut out;
    out.depth = color.a * (color.r + color.g + color.b) / 3.0;
    return out;
}

fragment float4 convert_x8d24_to_abgr8(ClearOut in [[stage_in]],
                                       depth2d<float> src [[texture(0)]]) {
    const uint depth = uint(round(src.read(uint2(in.position.xy)) * 16777215.0));
    return unpack_unorm4x8_to_float(depth);
}

fragment DepthOut convert_abgr8_to_x8d24(ClearOut in [[stage_in]],
                                         texture2d<float> src [[texture(0)]]) {
    const float4 color = src.read(uint2(in.position.xy));
    const uint depth = uint(round(color.r * 255.0)) | (uint(round(color.g * 255.0)) << 8) |
                       (uint(round(color.b * 255.0)) << 16);
    const uint scaled = as_type<uint>(float(depth) * (1.0 / 16777216.0));
    DepthOut out;
    out.depth = as_type<float>(scaled + uint(depth != 0u));
    return out;
}

uint4 DepthStencilBytes(float depth, uint stencil, bool stencil_first) {
    // The 24-bit value rounded, not Vulkan's float scaling by 2^32 - 1, which loses its low
    // bits in float and carries into the next byte.
    const uint depth24 = uint(round(saturate(depth) * 16777215.0));
    const uint3 bytes = (uint3(depth24) >> uint3(16, 8, 0)) & 0xFFu;
    return stencil_first ? uint4(stencil, bytes) : uint4(bytes, stencil);
}

fragment float4 convert_d24s8_to_abgr8(ClearOut in [[stage_in]],
                                       depth2d<float> src [[texture(0)]],
                                       texture2d<uint> src_stencil [[texture(1)]]) {
    const uint2 texel = uint2(in.position.xy);
    const uint4 bytes = DepthStencilBytes(src.read(texel), src_stencil.read(texel).r, true);
    return float4(bytes.wzyx) / 255.0;
}

fragment float4 convert_s8d24_to_abgr8(ClearOut in [[stage_in]],
                                       depth2d<float> src [[texture(0)]],
                                       texture2d<uint> src_stencil [[texture(1)]]) {
    const uint2 texel = uint2(in.position.xy);
    const uint4 bytes = DepthStencilBytes(src.read(texel), src_stencil.read(texel).r, false);
    return float4(bytes.wzyx) / 255.0;
}

fragment DepthStencilOut convert_abgr8_to_d24s8(ClearOut in [[stage_in]],
                                                texture2d<float> src [[texture(0)]]) {
    const uint4 color = uint4(src.read(uint2(in.position.xy)).abgr * 255.0);
    const uint4 bytes = color << uint4(24, 16, 8, 0);
    const uint value = bytes.x | bytes.y | bytes.z | bytes.w;
    DepthStencilOut out;
    out.depth = float(value & 0x00FFFFFFu) / 16777215.0;
    out.stencil = value >> 24;
    return out;
}

// Nearest scaling of what Blit can't draw (ScalePass): sample for sample, or the stencil.
struct ScaleParams {
    uint2 src_size;
    uint2 dst_size;
    uint layer;
};

uint2 ScaledTexel(float2 position, constant ScaleParams& p) {
    return min(uint2(position) * p.src_size / p.dst_size, p.src_size - 1);
}

fragment float4 scale_samples_float(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                                    texture2d_ms_array<float> src [[texture(0)]],
                                    constant ScaleParams& p [[buffer(0)]]) {
    return src.read(ScaledTexel(in.position.xy, p), p.layer, sample_index);
}

fragment uint4 scale_samples_uint(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                                  texture2d_ms_array<uint> src [[texture(0)]],
                                  constant ScaleParams& p [[buffer(0)]]) {
    return src.read(ScaledTexel(in.position.xy, p), p.layer, sample_index);
}

fragment int4 scale_samples_sint(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                                 texture2d_ms_array<int> src [[texture(0)]],
                                 constant ScaleParams& p [[buffer(0)]]) {
    return src.read(ScaledTexel(in.position.xy, p), p.layer, sample_index);
}

fragment DepthOut scale_samples_depth(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                                      depth2d_ms_array<float> src [[texture(0)]],
                                      constant ScaleParams& p [[buffer(0)]]) {
    DepthOut out;
    out.depth = src.read(ScaledTexel(in.position.xy, p), p.layer, sample_index);
    return out;
}

struct StencilOut {
    uint stencil [[stencil]];
};

fragment StencilOut scale_stencil(ClearOut in [[stage_in]],
                                  texture2d_array<uint> src [[texture(0)]],
                                  constant ScaleParams& p [[buffer(0)]]) {
    StencilOut out;
    out.stencil = src.read(ScaledTexel(in.position.xy, p), p.layer).r;
    return out;
}

// Guest depth and stencil bytes (DepthLayout) to and from depth textures.
struct DepthCopyParams {
    int2 origin;
    uint row_words;
    uint layout; // 0: depth << 8 | stencil, 1: float depth, then the stencil word
    uint2 extent;
    uint has_stencil;
    uint pad;
};

DepthStencilOut DepthFromWords(uint2 texel, device const uint* words,
                               constant DepthCopyParams& p) {
    DepthStencilOut out;
    if (p.layout == 0) {
        const uint word = words[texel.y * p.row_words + texel.x];
        out.depth = float(word >> 8) / 16777215.0;
        out.stencil = word & 0xFFu;
    } else if (p.layout == 2) {
        const uint word = words[texel.y * p.row_words + texel.x];
        out.depth = float(word & 0xFFFFFFu) / 16777215.0;
        out.stencil = word >> 24;
    } else {
        const uint index = texel.y * p.row_words + texel.x * 2;
        out.depth = as_type<float>(words[index]);
        out.stencil = words[index + 1] & 0xFFu;
    }
    return out;
}

fragment DepthOut depth_from_buffer(ClearOut in [[stage_in]],
                                    device const uint* words [[buffer(0)]],
                                    constant DepthCopyParams& p [[buffer(1)]]) {
    DepthOut out;
    out.depth = DepthFromWords(uint2(int2(in.position.xy) - p.origin), words, p).depth;
    return out;
}

fragment DepthStencilOut depth_stencil_from_buffer(ClearOut in [[stage_in]],
                                                   device const uint* words [[buffer(0)]],
                                                   constant DepthCopyParams& p [[buffer(1)]]) {
    return DepthFromWords(uint2(int2(in.position.xy) - p.origin), words, p);
}

// nxnative dm/gfx/msl/rb_swap.metal (MIT, Copyright (c) 2026 seedds): the red / blue exchange of
// the 565 family, or reversed nibbles (shift 4), in place on rows of 16-bit texels.
struct SwapParams {
    uint offset;
    uint width;
    uint height;
    uint row_bytes;
    uint image_bytes;
    uint shift;
};

kernel void swap_packed(uint3 id [[thread_position_in_grid]],
                        device uchar* bytes [[buffer(0)]],
                        constant SwapParams& p [[buffer(1)]]) {
    if (id.x >= p.width || id.y >= p.height) {
        return;
    }
    device ushort* texel = (device ushort*)(bytes + p.offset + id.z * p.image_bytes +
                                            id.y * p.row_bytes) + id.x;
    const uint v = *texel;
    const uint m = 31u | (31u << p.shift);
    *texel = p.shift == 4
                 ? ushort(v << 12 | (v & 0xf0u) << 4 | (v >> 4 & 0xf0u) | v >> 12)
                 : ushort((v & ~m) | (v & 31u) << p.shift | (v >> p.shift & 31u));
}

// A buffer copy on an open compute encoder (EDEN_NX2_METAL_COMPUTE_COPY): 16 bytes per thread.
struct CopyParams {
    ulong src;
    ulong dst;
    ulong size;
};

kernel void copy_bytes(uint id [[thread_position_in_grid]],
                       device const uchar* src [[buffer(0)]],
                       device uchar* dst [[buffer(1)]],
                       constant CopyParams& p [[buffer(2)]]) {
    const ulong at = ulong(id) * 16;
    if (at >= p.size) {
        return;
    }
    const ulong s = p.src + at;
    const ulong d = p.dst + at;
    if (at + 16 <= p.size) {
        if (((s | d) & 15) == 0) {
            *(device uint4*)(dst + d) = *(device const uint4*)(src + s);
            return;
        }
        if (((s | d) & 3) == 0) {
            device const uint* from = (device const uint*)(src + s);
            device uint* to = (device uint*)(dst + d);
            to[0] = from[0];
            to[1] = from[1];
            to[2] = from[2];
            to[3] = from[3];
            return;
        }
    }
    const uint n = uint(min(ulong(16), p.size - at));
    for (uint i = 0; i < n; ++i) {
        dst[d + i] = src[s + i];
    }
}

// A resolve shadow's colour into every sample of its multisampled texture (RestoreSamples).
fragment float4 restore_samples_float(ClearOut in [[stage_in]],
                                      texture2d_array<float> src [[texture(0)]],
                                      constant uint& layer [[buffer(0)]]) {
    return src.read(uint2(in.position.xy), layer);
}

// The 16-bit packed formats as RGBA8 on the host (Pack16Mode); these move whole regions between
// the texture and the guest's words. 1 A5B5G5R1 (alpha 15-11, blue 10-6, green 5-1, red 0),
// 2 R5G6B5, 3 B5G6R5, 4 A1R5G5B5, 5 A1B5G5R5, 6 A4B4G4R4.
struct PackParams {
    uint2 origin;
    uint2 extent;
    uint row_words;
    uint mode;
};

float4 Unpack16(uint v, uint mode) {
    switch (mode) {
    case 1: return float4(float(v & 1u), float((v >> 1) & 31u) / 31.0, float((v >> 6) & 31u) / 31.0,
                          float((v >> 11) & 31u) / 31.0);
    case 2: return float4(float(v >> 11) / 31.0, float((v >> 5) & 63u) / 63.0, float(v & 31u) / 31.0, 1.0);
    case 3: return float4(float(v & 31u) / 31.0, float((v >> 5) & 63u) / 63.0, float(v >> 11) / 31.0, 1.0);
    case 4: return float4(float((v >> 10) & 31u) / 31.0, float((v >> 5) & 31u) / 31.0,
                          float(v & 31u) / 31.0, float(v >> 15));
    case 5: return float4(float(v & 31u) / 31.0, float((v >> 5) & 31u) / 31.0,
                          float((v >> 10) & 31u) / 31.0, float(v >> 15));
    default: return float4(float(v & 15u), float((v >> 4) & 15u), float((v >> 8) & 15u),
                           float(v >> 12)) / 15.0;
    }
}

uint Pack16(float4 c, uint mode) {
    c = saturate(c);
    switch (mode) {
    case 1: return (uint(round(c.a * 31.0)) << 11) | (uint(round(c.b * 31.0)) << 6) |
                   (uint(round(c.g * 31.0)) << 1) | (c.r >= 0.5 ? 1u : 0u);
    case 2: return (uint(round(c.r * 31.0)) << 11) | (uint(round(c.g * 63.0)) << 5) |
                   uint(round(c.b * 31.0));
    case 3: return (uint(round(c.b * 31.0)) << 11) | (uint(round(c.g * 63.0)) << 5) |
                   uint(round(c.r * 31.0));
    case 4: return (c.a >= 0.5 ? 0x8000u : 0u) | (uint(round(c.r * 31.0)) << 10) |
                   (uint(round(c.g * 31.0)) << 5) | uint(round(c.b * 31.0));
    case 5: return (c.a >= 0.5 ? 0x8000u : 0u) | (uint(round(c.b * 31.0)) << 10) |
                   (uint(round(c.g * 31.0)) << 5) | uint(round(c.r * 31.0));
    default: return (uint(round(c.a * 15.0)) << 12) | (uint(round(c.b * 15.0)) << 8) |
                    (uint(round(c.g * 15.0)) << 4) | uint(round(c.r * 15.0));
    }
}

kernel void pack16_from_buffer(uint2 id [[thread_position_in_grid]],
                               texture2d<float, access::write> dst [[texture(0)]],
                               device const ushort* words [[buffer(0)]],
                               constant PackParams& p [[buffer(1)]]) {
    if (any(id >= p.extent)) {
        return;
    }
    dst.write(Unpack16(words[id.y * p.row_words + id.x], p.mode), p.origin + id);
}

kernel void pack16_to_buffer(uint2 id [[thread_position_in_grid]],
                             texture2d<float> src [[texture(0)]],
                             device ushort* words [[buffer(0)]],
                             constant PackParams& p [[buffer(1)]]) {
    if (any(id >= p.extent)) {
        return;
    }
    words[id.y * p.row_words + id.x] = ushort(Pack16(src.read(p.origin + id), p.mode));
}

kernel void depth_to_buffer(uint2 id [[thread_position_in_grid]],
                            depth2d<float> depth [[texture(0)]],
                            texture2d<uint> stencil [[texture(1)]],
                            device uint* words [[buffer(0)]],
                            constant DepthCopyParams& p [[buffer(1)]]) {
    if (any(id >= p.extent)) {
        return;
    }
    const uint2 texel = uint2(p.origin) + id;
    const float value = depth.read(texel);
    const uint stencil_value = p.has_stencil != 0 ? stencil.read(texel).r : 0u;
    if (p.layout == 0) {
        words[id.y * p.row_words + id.x] =
            (uint(round(saturate(value) * 16777215.0)) << 8) | stencil_value;
    } else if (p.layout == 2) {
        words[id.y * p.row_words + id.x] =
            (stencil_value << 24) | uint(round(saturate(value) * 16777215.0));
    } else {
        const uint index = id.y * p.row_words + id.x * 2;
        words[index] = as_type<uint>(value);
        words[index + 1] = stencil_value;
    }
}

fragment DepthOut compress_depth(ClearOut in [[stage_in]], uint sample_index [[sample_id]],
                                 depth2d<float> src [[texture(0)]],
                                 constant SampleParams& p [[buffer(0)]]) {
    DepthOut out;
    out.depth = src.read(TexelOf(uint2(in.position.xy), sample_index, p));
    return out;
}
)";

/// The stencil-only view format of a depth-stencil format (invalid when it has none).
MTLPixelFormat StencilViewFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth32Float_Stencil8:
        return MTLPixelFormatX32_Stencil8;
#if !TARGET_OS_IPHONE
    case MTLPixelFormatDepth24Unorm_Stencil8:
        return MTLPixelFormatX24_Stencil8;
#endif
    default:
        return MTLPixelFormatInvalid;
    }
}

/// The stencil-only view of a depth-stencil texture a blit reads, or nil. With the pipeline check
/// the texture is checked first, as CreateTextureView checks its views: Metal aborts on a view
/// of a texture made without MTLTextureUsagePixelFormatView, of another type family or of a
/// multisampled texture.
id<MTLTexture> StencilView(id<MTLTexture> texture) {
    const MTLPixelFormat format = StencilViewFormat(texture.pixelFormat);
    if (format == MTLPixelFormatInvalid) {
        return nil;
    }
    if (PipelineCheckEnabled()) {
        const char* reason = nullptr;
        if ((texture.usage & MTLTextureUsagePixelFormatView) == 0) {
            reason = "made without format views";
        } else if (texture.sampleCount != 1) {
            reason = "multisampled";
        } else if (texture.textureType != MTLTextureType2D &&
                   texture.textureType != MTLTextureType2DArray) {
            reason = "not a 2D texture";
        }
        if (reason != nullptr) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Metal, "Metal: stencil view refused: {}; format {} type {}",
                            reason, static_cast<u32>(texture.pixelFormat),
                            static_cast<u32>(texture.textureType));
            }
            return nil;
        }
    }
    return [texture newTextureViewWithPixelFormat:format];
}

void ApplyAttachment(MTLRenderPassAttachmentDescriptor* descriptor, const Attachment& attachment) {
    descriptor.texture = AsTexture(attachment.texture);
    descriptor.level = attachment.level;
    descriptor.slice = attachment.slice;
    descriptor.depthPlane = attachment.depth_plane;
}

id<MTLTexture> RootOf(id<MTLTexture> texture) {
    while (texture.parentTexture != nil) {
        texture = texture.parentTexture;
    }
    return texture;
}

// EDEN_NX2_METAL_CLEAR_FOLD=0: every clear is a render pass of its own, as this backend started.
bool ClearFoldEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_CLEAR_FOLD");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_STATE_CACHE=0: every draw and dispatch sets all its state and bindings again.
bool StateCacheEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_STATE_CACHE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// Vulkan's EDEN_NX2_DYNSTATE_CACHE=0 (read under EDEN_NX2_METAL_VULKAN_SWITCHES): every draw sets
// its draw state (pipeline, depth-stencil state, viewport, scissor, culling, depth bias, stencil
// references, blend colour) again, its bindings staying cached.
bool DrawStateCacheEnabled() {
    static const bool enabled = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_DYNSTATE_CACHE");
        return StateCacheEnabled() && !(value != nullptr && value[0] == '0');
    }();
    return enabled;
}

// Vulkan's EDEN_NX2_VERTEX_INPUT_CACHE=0 (read under EDEN_NX2_METAL_VULKAN_SWITCHES): every draw
// binds its vertex buffers again.
bool VertexInputCacheEnabled() {
    static const bool enabled = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_VERTEX_INPUT_CACHE");
        return StateCacheEnabled() && !(value != nullptr && value[0] == '0');
    }();
    return enabled;
}

// EDEN_NX2_METAL_AHEAD=0: uploads and kernels are recorded where they come, ending the open
// render encoder.
bool AheadEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_AHEAD");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_AHEAD_SPLIT=0: no command buffer is committed early to let a render pass begin
// a segment.
bool AheadSplitEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_AHEAD_SPLIT");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_FENCE_IDLE=0: a guest fence commits a command buffer even when nothing was
// recorded since the last one.
bool FenceIdleEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FENCE_IDLE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// Vulkan's EDEN_NX2_FENCE_FLUSH_US (read under EDEN_NX2_METAL_VULKAN_SWITCHES); 0 with
// EDEN_NX2_METAL_FENCE_BATCH=0, else defaults to a bounded 300 us window on GPUBack while the GPU
// already has >= 2 committed command buffers in flight.
bool FenceFlushExplicit() {
    static const bool explicit_window = [] {
        return VulkanSwitch("EDEN_NX2_FENCE_FLUSH_US") != nullptr ||
               (VulkanSwitchesEnabled() &&
                VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush));
    }();
    return explicit_window;
}

s64 FenceFlushIntervalUs() {
    static const s64 interval_us = [] {
        const char* const batch = std::getenv("EDEN_NX2_METAL_FENCE_BATCH");
        if (batch != nullptr && batch[0] == '0') {
            return s64{0};
        }
        if (const char* const value = VulkanSwitch("EDEN_NX2_FENCE_FLUSH_US")) {
            return static_cast<s64>(std::atoll(value));
        }
        if (VulkanSwitchesEnabled() &&
            VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush)) {
            return s64{5000};
        }
        return s64{300};
    }();
    return interval_us;
}

// EDEN_NX2_METAL_FENCE_BATCH=2: fence work is left pending off the split's back too, where only
// the caller commits it (the smoke test).
bool FenceBatchAnyThread() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FENCE_BATCH");
        return value != nullptr && value[0] == '2';
    }();
    return enabled;
}

/// A switch on unless its variable is 0, logged when first read.
bool SwitchOn(const char* name, const char* what) {
    const char* const value = std::getenv(name);
    const bool on = value == nullptr || value[0] != '0';
    LOG_INFO(Render_Metal, "Metal: {}: {} ({}={})", what, on ? "on" : "off", name, on ? 1 : 0);
    return on;
}

// EDEN_NX2_METAL_MERGE_AHEAD=0: the uploads and the kernels recorded ahead of a segment go into
// two command buffers (uploads, then kernels) instead of one encoded in recording order.
bool MergeAheadEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_MERGE_AHEAD", "one ahead command buffer per segment");
    return enabled;
}

// EDEN_NX2_METAL_AHEAD_ANY=0: a draw's kernels go ahead only while the segment holds nothing but
// the render pass they draw into. Otherwise they go ahead after any work as long as no buffer
// they bind was bound or copied in order in the segment (else they fall back in order).
bool AheadAnyEnabled() {
    static const bool enabled =
        AheadEnabled() &&
        SwitchOn("EDEN_NX2_METAL_AHEAD_ANY", "draw kernels ahead of a busy segment, checked");
    return enabled;
}

// EDEN_NX2_METAL_CLEAR_NARROW=0: copies, blits and dispatches record the waiting clear first
// whatever textures they touch; otherwise only those on its textures do.
bool ClearNarrowEnabled() {
    static const bool enabled =
        ClearFoldEnabled() &&
        SwitchOn("EDEN_NX2_METAL_CLEAR_NARROW", "waiting clears flushed by their textures only");
    return enabled;
}

// EDEN_NX2_METAL_DEPTH_KILL=0: a pass's depth and stencil are stored even when the next thing
// done to them is a clear of the whole attachment (on Apple GPUs).
bool DepthKillEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_DEPTH_KILL", "depth/stencil not stored before a full clear");
    return enabled;
}

// EDEN_NX2_METAL_EMPTY_SKIP=1: a Flush with nothing recorded, with the GPU done with every
// earlier tick, retires the tick without a command buffer. Off by default: OoT had no such
// flushes, and it split batched fences (EDEN_NX2_FENCE_FLUSH_US) over two ticks (smoke test).
bool EmptySkipEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_EMPTY_SKIP");
        const bool on = value != nullptr && value[0] == '1';
        LOG_INFO(Render_Metal, "Metal: empty flushes retired without a command buffer: {}",
                 on ? "on" : "off");
        return on;
    }();
    return enabled;
}

// EDEN_NX2_METAL_LIST_END_DEFER=<n>: a command list end leaves its work pending while the GPU has
// at least n committed ticks not done yet (default 2, bounded within 300 us).
bool ListEndDeferExplicit() {
    static const bool explicit_depth = std::getenv("EDEN_NX2_METAL_LIST_END_DEFER") != nullptr;
    return explicit_depth;
}

u64 ListEndDeferDepth() {
    static const u64 depth = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_LIST_END_DEFER");
        const u64 parsed = value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 10))
                                            : u64{2};
        LOG_INFO(Render_Metal,
                 "Metal: list ends left pending while the GPU is {} ticks behind: {} "
                 "(EDEN_NX2_METAL_LIST_END_DEFER={})",
                 parsed, parsed != 0 ? "on" : "off", parsed);
        return parsed;
    }();
    return depth;
}

// EDEN_NX2_METAL_IDLE_COMMIT_ALL=1: the split's idle hook commits everything recorded, not only
// work a fence or list end left pending.
bool IdleCommitAllEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_IDLE_COMMIT_ALL");
        const bool on = value != nullptr && value[0] != '0';
        LOG_INFO(Render_Metal, "Metal: idle GPU thread commits all recorded work: {}",
                 on ? "on" : "off");
        return on;
    }();
    return enabled;
}

bool InPassClearEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_INPASS_CLEAR", "same-target mid-pass clears drawn in pass");
    return enabled;
}

// EDEN_NX2_METAL_STATIC_LABELS=0: command buffers and render passes get labels formatted with
// their tick and pass number again (an NSString format per command buffer and per pass on the
// recording thread); otherwise constant labels (failure logs still name the tick).
bool StaticLabelsEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_STATIC_LABELS", "constant command buffer and pass labels");
    return enabled;
}

// EDEN_NX2_METAL_CB_PREMAKE=0: every command buffer is made on the recording thread when it is
// needed. Otherwise a serial queue keeps a few made ahead (commandBufferWithDescriptor and the
// system's frame pacing hook on it run there): the GPU thread takes one, uncommitted and not
// enqueued, so the commit order stays the recording thread's. With constant labels only.
// EDEN_NX2_METAL_RECORDER=1: Metal encoding on an encoder thread (metal_recorder.h); the
// recording thread appends commands to a stream it replays and commits in order. Read at boot.
bool RecorderEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_RECORDER");
        const bool on = value != nullptr && value[0] == '1';
        LOG_INFO(Render_Metal, "Metal: encoding on an encoder thread: {} (EDEN_NX2_METAL_RECORDER)",
                 on ? "on" : "off");
        return on;
    }();
    return enabled;
}

bool CommandBufferPremakeEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_CB_PREMAKE", "command buffers made ahead off the GPU thread");
    return enabled;
}

/// Command buffers made ahead on a serial queue (EDEN_NX2_METAL_CB_PREMAKE). They hold slots of
/// the queue's 64 until taken; Count of them at most.
class CommandBufferPremake {
public:
    static constexpr size_t Count = 3;

    ~CommandBufferPremake() {
        if (refill_queue != nil) {
            stopping.store(true, std::memory_order_relaxed);
            dispatch_sync(refill_queue, ^{
                          });
        }
        std::scoped_lock lock{mutex};
        ready.clear();
    }

    /// One made ahead (nil when none is ready); a refill is started either way.
    id<MTLCommandBuffer> Take(id<MTLCommandQueue> queue, MTLCommandBufferDescriptor* descriptor) {
        id<MTLCommandBuffer> buffer = nil;
        {
            std::scoped_lock lock{mutex};
            if (!ready.empty()) {
                buffer = ready.front();
                ready.pop_front();
            }
        }
        if (buffer != nil) {
            hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            misses.fetch_add(1, std::memory_order_relaxed);
        }
        Refill(queue, descriptor);
        return buffer;
    }

    std::atomic<u64> hits{};
    std::atomic<u64> misses{};

private:
    void Refill(id<MTLCommandQueue> queue, MTLCommandBufferDescriptor* descriptor) {
        if (refill_pending.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        if (refill_queue == nil) {
            const dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
                DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
            refill_queue = dispatch_queue_create("Lichii command buffers", attr);
        }
        CommandBufferPremake* const self = this;
        dispatch_async(refill_queue, ^{
          for (;;) {
              {
                  std::scoped_lock lock{self->mutex};
                  if (self->ready.size() >= Count ||
                      self->stopping.load(std::memory_order_relaxed)) {
                      break;
                  }
              }
              id<MTLCommandBuffer> made = nil;
              @autoreleasepool {
                  made = [queue commandBufferWithDescriptor:descriptor];
              }
              if (made == nil) {
                  break;
              }
              std::scoped_lock lock{self->mutex};
              self->ready.push_back(made);
          }
          self->refill_pending.store(false, std::memory_order_release);
        });
    }

    std::mutex mutex;
    std::deque<id<MTLCommandBuffer>> ready;
    std::atomic<bool> refill_pending{};
    std::atomic<bool> stopping{};
    dispatch_queue_t refill_queue = nil;
};

// EDEN_NX2_METAL_IMAGE_NOTE_SKIP=0: a draw's texture cache views are noted for the segment as
// texel buffer views are (NoteSegmentTexture walks each one's parents for a buffer under it,
// once per segment). Otherwise only views that may be on a buffer are: the texture cache makes
// its images with newTextureWithDescriptor (device or heap), never on a buffer.
bool ImageNoteSkipEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_IMAGE_NOTE_SKIP", "image views not noted as buffer views");
    return enabled;
}

// EDEN_NX2_METAL_SEGMENT_NOTE_CACHE=0: every texture bind of in-order work walks the texture's
// parents for the buffer under it (NoteSegmentTexture), even when the segment noted it already.
bool SegmentNoteCacheEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_SEGMENT_NOTE_CACHE", "texture binds noted once per segment");
    return enabled;
}

// EDEN_NX2_METAL_COMPUTE_COPY=0: a buffer copy always goes to a blit encoder, ending the open
// compute encoder (the segment's or the ahead kernels'). Otherwise a copy of up to
// EDEN_NX2_METAL_COMPUTE_COPY_MAX bytes (256 KiB) is a dispatch on that open compute encoder,
// and (unless =1) an ahead copy with no ahead encoder open opens the kernels' compute encoder
// rather than a blit one, so the ahead command buffer's copies and kernels share one encoder.
u32 ComputeCopyMode() {
    static const u32 mode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_COMPUTE_COPY");
        const u32 read = value == nullptr ? 2U : value[0] == '0' ? 0U : value[0] == '1' ? 1U : 2U;
        LOG_INFO(Render_Metal,
                 "Metal: buffer copies as dispatches on open compute encoders: {} "
                 "(EDEN_NX2_METAL_COMPUTE_COPY={})",
                 read == 0 ? "off" : read == 1 ? "on" : "on, ahead ones opening one", read);
        return read;
    }();
    return mode;
}

bool ComputeCopyEnabled() {
    return ComputeCopyMode() != 0;
}

u64 ComputeCopyMax() {
    static const u64 max_bytes = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_COMPUTE_COPY_MAX");
        return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 0))
                                : u64{256 * 1024};
    }();
    return max_bytes;
}

constexpr std::array<const char*, static_cast<size_t>(CommitReason::Count)> CommitReasonNames{
    "other",  "present",   "list_end",  "fence", "fence_deferred", "idle",  "wait", "finish",
    "staging", "draw_budget", "list_copy", "query", "condition",      "split", "ahead",
};

/// What ended the open render encoder (the "Metal pass ends:" line).
enum class PassEnd : u8 {
    NewPass,       ///< BeginRendering on other targets
    Clear,         ///< a clear pass (Clear, a waiting clear recorded)
    Helper,        ///< a pass of the scheduler's own (blit, convert, resolve, depth upload...)
    Blit,          ///< a copy in order
    Compute,       ///< a kernel in order
    Commit,        ///< Flush
    Split,         ///< SplitBefore
    CommandBuffer, ///< CommandBuffer() (present, MetalFX)
    EndRendering,  ///< the rasterizer, texture cache or presenter ended it
    Other,
    Count,
};

constexpr std::array<const char*, static_cast<size_t>(PassEnd::Count)> PassEndNames{
    "new_pass", "clear", "helper", "blit", "compute", "commit", "split", "command_buffer",
    "end_rendering", "other",
};

/// A compute bind of a kernel, kept to replay it on another encoder.
struct ComputeBind {
    enum class Kind : u8 { Buffer, Bytes, Texture, Sampler, UseBuffer, UseTexture };
    Kind kind;
    bool write;
    u32 slot;
    void* object;
    u32 offset; ///< Bytes: where in compute_bind_bytes
    u32 size;
};

/// What one stage of an encoder has bound; a new encoder has nothing bound, as this starts.
struct StageBindings {
    std::array<const void*, BufferSlots> buffers{};
    std::array<u32, BufferSlots> offsets{};
    std::array<const void*, TextureSlots> textures{};
    std::array<const void*, SamplerSlots> samplers{};
};

/// A buffer slot holding bytes (setBytes): its address stands for a buffer no other equals.
const char InlineBytes{};

/// A buffer or texture noted in segment `serial` (EDEN_NX2_METAL_SEGMENT_NOTE_CACHE).
struct SegmentNote {
    const void* object;
    u64 serial;
    bool writes_shared;
};

enum class Bound {
    Same,   ///< Nothing to set
    Offset, ///< The same buffer at another offset
    Other,
};

void ReportFailure(id<MTLCommandBuffer> finished, u64 tick, std::atomic<u32>* errors_logged) {
    if (finished.status != MTLCommandBufferStatusError || errors_logged->fetch_add(1) >= 16) {
        return;
    }
    NSError* const error = finished.error;
    std::string encoders;
    NSArray* const infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> info in infos) {
        // MTLCommandEncoderErrorState: 0 unknown, 1 completed, 2 affected, 3 pending, 4 faulted.
        if (info.errorState != MTLCommandEncoderErrorStateCompleted) {
            encoders += fmt::format("{}{} (state {})", encoders.empty() ? "" : ", ",
                                    info.label.UTF8String != nullptr ? info.label.UTF8String : "?",
                                    static_cast<int>(info.errorState));
        }
    }
    LOG_ERROR(Render_Metal, "Metal: command buffer {} ({}) failed: code {}: {}; encoders: {}", tick,
              finished.label.UTF8String != nullptr ? finished.label.UTF8String : "?",
              static_cast<long>(error.code),
              error.localizedDescription.UTF8String != nullptr
                  ? error.localizedDescription.UTF8String
                  : "no description",
              encoders.empty() ? "none reported" : encoders);
}

/// Adds `clear` to the waiting `pending` when one render pass can do both.
bool MergeClear(ClearPass& pending, const ClearPass& clear) {
    if (pending.width != clear.width || pending.height != clear.height) {
        return false;
    }
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.clear_color[index] && pending.clear_color[index] &&
            pending.color[index] != clear.color[index]) {
            return false;
        }
    }
    // One depth-stencil texture a pass.
    const Attachment* texture = nullptr;
    const std::array<const Attachment*, 4> depth_stencil{
        pending.clear_depth ? &pending.depth : nullptr,
        pending.clear_stencil ? &pending.stencil : nullptr,
        clear.clear_depth ? &clear.depth : nullptr,
        clear.clear_stencil ? &clear.stencil : nullptr,
    };
    for (const Attachment* const attachment : depth_stencil) {
        if (attachment == nullptr) {
            continue;
        }
        if (texture != nullptr && *texture != *attachment) {
            return false;
        }
        texture = attachment;
    }
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.clear_color[index]) {
            pending.color[index] = clear.color[index];
            pending.clear_color[index] = true;
            pending.color_value[index] = clear.color_value[index];
        }
    }
    if (clear.clear_depth) {
        pending.depth = clear.depth;
        pending.clear_depth = true;
        pending.depth_value = clear.depth_value;
    }
    if (clear.clear_stencil) {
        pending.stencil = clear.stencil;
        pending.clear_stencil = true;
        pending.stencil_value = clear.stencil_value;
    }
    return true;
}

/// Whether a render pass on `targets` can fold all or a non-conflicting non-empty subset of
/// `clear` into its load actions, leaving any unconsumed attachments in `pending_clear`.
bool CanFoldClearInto(const ClearPass& clear, const RenderTargetSet& targets) {
    if (clear.width != targets.width || clear.height != targets.height) {
        return false;
    }
    const auto same_root = [](TextureHandle a, TextureHandle b) {
        return a != nullptr && b != nullptr && RootOf(AsTexture(a)) == RootOf(AsTexture(b));
    };
    const auto targets_use_root = [&](TextureHandle tex) {
        if (tex == nullptr) {
            return false;
        }
        for (const Attachment& att : targets.color) {
            if (same_root(att.texture, tex)) {
                return true;
            }
        }
        return (targets.has_depth || targets.has_stencil) && same_root(targets.depth.texture, tex);
    };
    bool any_match = false;
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (!clear.clear_color[index]) {
            continue;
        }
        if (targets.color[index] == clear.color[index]) {
            any_match = true;
        } else if (targets.color[index].texture != nullptr ||
                   targets_use_root(clear.color[index].texture)) {
            return false;
        }
    }
    if (clear.clear_depth) {
        if (targets.has_depth && clear.depth == targets.depth) {
            any_match = true;
        } else if (targets.has_depth || targets_use_root(clear.depth.texture)) {
            return false;
        }
    }
    if (clear.clear_stencil) {
        if (targets.has_stencil && clear.stencil == targets.depth) {
            any_match = true;
        } else if (targets.has_stencil || targets_use_root(clear.stencil.texture)) {
            return false;
        }
    }
    return any_match;
}

/// Consumes the attachments of `pending` that match `targets`, keeping any remaining ones in
/// `pending` (`has_remainder` tells whether `pending` still holds anything).
ClearPass ConsumeClearSubset(ClearPass& pending, const RenderTargetSet& targets,
                             bool& has_remainder) {
    ClearPass consumed{};
    consumed.width = pending.width;
    consumed.height = pending.height;
    has_remainder = false;
    for (size_t index = 0; index < pending.color.size(); ++index) {
        if (!pending.clear_color[index]) {
            continue;
        }
        if (targets.color[index] == pending.color[index]) {
            consumed.color[index] = pending.color[index];
            consumed.clear_color[index] = true;
            consumed.color_value[index] = pending.color_value[index];
            pending.color[index] = {};
            pending.clear_color[index] = false;
        } else {
            has_remainder = true;
        }
    }
    if (pending.clear_depth) {
        if (targets.has_depth && pending.depth == targets.depth) {
            consumed.depth = pending.depth;
            consumed.clear_depth = true;
            consumed.depth_value = pending.depth_value;
            pending.depth = {};
            pending.clear_depth = false;
        } else {
            has_remainder = true;
        }
    }
    if (pending.clear_stencil) {
        if (targets.has_stencil && pending.stencil == targets.depth) {
            consumed.stencil = pending.stencil;
            consumed.clear_stencil = true;
            consumed.stencil_value = pending.stencil_value;
            pending.stencil = {};
            pending.clear_stencil = false;
        } else {
            has_remainder = true;
        }
    }
    return consumed;
}

struct InPassClearKey {
    std::array<u32, 8> color_formats{};
    u32 depth_format{};
    u32 stencil_format{};
    u8 samples{};
    u8 color_clear_mask{};

    bool operator==(const InPassClearKey&) const noexcept = default;
};

struct InPassClearKeyHash {
    size_t operator()(const InPassClearKey& key) const noexcept {
        u64 h = 14695981039346656037ULL;
        const auto mix = [&h](u64 v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        for (const u32 fmt : key.color_formats) {
            mix(fmt);
        }
        mix(key.depth_format);
        mix(key.stencil_format);
        mix((u64{key.samples} << 8) | key.color_clear_mask);
        return static_cast<size_t>(h);
    }
};

} // Anonymous namespace

struct Scheduler::Impl {
    explicit Impl(const Device& device_) : device{device_}, queue{device_.GetQueue()} {
        // MTLGPUFamilyApple10 and setDepthTestMinBound:maxBound: came with the macOS / iOS 26
        // SDKs; an older SDK builds without the depth bounds test.
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
        if (@available(macOS 26.0, iOS 26.0, *)) {
            depth_bounds = [device_.GetDevice() supportsFamily:MTLGPUFamilyApple10];
        }
#endif
        NSError* error = nil;
        library = [device.GetDevice() newLibraryWithSource:@(BlitSource) options:nil error:&error];
        if (library == nil) {
            LOG_ERROR(Render_Metal, "Metal: blit shaders failed to build: {}",
                      error.localizedDescription.UTF8String);
        }
        MTLSamplerDescriptor* sampler_desc = [MTLSamplerDescriptor new];
        sampler_desc.minFilter = MTLSamplerMinMagFilterNearest;
        sampler_desc.magFilter = MTLSamplerMinMagFilterNearest;
        nearest = [device.GetDevice() newSamplerStateWithDescriptor:sampler_desc];
        sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
        sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
        linear = [device.GetDevice() newSamplerStateWithDescriptor:sampler_desc];
        visibility = [device.GetDevice() newBufferWithLength:VisibilitySlots * sizeof(u64)
                                                     options:MTLResourceStorageModeShared];
        std::memset(visibility.contents, 0, VisibilitySlots * sizeof(u64));
        event = [device.GetDevice() newSharedEvent];
        if (RecorderEnabled()) {
            recorder = std::make_unique<Rec::Recorder>(queue);
        }
    }

    id<MTLCommandBuffer> NewCommandBuffer(NSString* label) {
        // EDEN_NX2_METAL_ENCODER_STATUS=0: no per-encoder status in command buffer errors.
        static const bool encoder_status = [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_ENCODER_STATUS");
            return value == nullptr || value[0] != '0';
        }();
        id<MTLCommandBuffer> buffer = nil;
        if (recorder != nullptr) {
            // A proxy; the encoder thread makes the command buffer.
            MTLCommandBufferDescriptor* descriptor = nil;
            if (encoder_status) {
                static MTLCommandBufferDescriptor* const recorder_descriptor = [] {
                    MTLCommandBufferDescriptor* const made = [MTLCommandBufferDescriptor new];
                    made.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
                    return made;
                }();
                descriptor = recorder_descriptor;
            }
            buffer = recorder->NewCommandBuffer(descriptor);
            buffer.label = label;
            return buffer;
        }
        if (encoder_status) {
            // One descriptor for all (EDEN_NX2_METAL_STATIC_LABELS): only read by Metal.
            static MTLCommandBufferDescriptor* const shared_descriptor = [] {
                MTLCommandBufferDescriptor* const made = [MTLCommandBufferDescriptor new];
                made.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
                return made;
            }();
            MTLCommandBufferDescriptor* descriptor = shared_descriptor;
            if (!StaticLabelsEnabled()) {
                descriptor = [MTLCommandBufferDescriptor new];
                descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            } else if (CommandBufferPremakeEnabled()) {
                buffer = premade.Take(queue, descriptor);
            }
            if (buffer == nil) {
                buffer = [queue commandBufferWithDescriptor:descriptor];
            }
        } else {
            buffer = [queue commandBuffer];
        }
        buffer.label = label;
        return buffer;
    }

    id<MTLCommandBuffer> Commands() {
        if (command_buffer == nil) {
            command_buffer =
                NewCommandBuffer(StaticLabelsEnabled()
                                     ? @"Lichii"
                                     : [NSString stringWithFormat:@"Lichii %llu", current_tick]);
        }
        return command_buffer;
    }

    /// The command buffer of what is recorded ahead (EDEN_NX2_METAL_MERGE_AHEAD).
    id<MTLCommandBuffer> AheadBuffer() {
        if (ahead_buffer == nil) {
            ahead_buffer =
                NewCommandBuffer(StaticLabelsEnabled()
                    ? @"Lichii ahead"
                    : [NSString stringWithFormat:@"Lichii %llu ahead", current_tick]);
        }
        return ahead_buffer;
    }

    void EndKernelEncoder() {
        if (kernel_compute == nil) {
            return;
        }
        if (active_compute == kernel_compute) {
            active_compute = nil;
            active_bound = nullptr;
            compute_clobbered_by_copy = false;
        }
        [kernel_compute endEncoding];
        kernel_compute = nil;
    }

    /// The blit encoder of the copies recorded ahead of the segment: in the ahead command
    /// buffer after the kernels recorded before them (EDEN_NX2_METAL_MERGE_AHEAD), else in a
    /// command buffer committed before the kernels' and the segment's.
    id<MTLBlitCommandEncoder> UploadEncoder() {
        if (upload_blit == nil) {
            if (MergeAheadEnabled()) {
                if (kernel_compute != nil && active_compute == kernel_compute) {
                    // In the middle of a kernel's binds: they go to a new encoder after the copy.
                    kernel_resume = true;
                }
                EndKernelEncoder();
                upload_blit = [AheadBuffer() blitCommandEncoder];
            } else {
                upload_buffer = NewCommandBuffer(
                    StaticLabelsEnabled()
                        ? @"Lichii uploads"
                        : [NSString stringWithFormat:@"Lichii %llu uploads", current_tick]);
                upload_blit = [upload_buffer blitCommandEncoder];
            }
            upload_blit.label = @"Uploads";
            ++encoders;
        }
        ++blit_commands;
        ++ahead_copies;
        return upload_blit;
    }

    /// The compute encoder of the kernels recorded ahead of the segment.
    id<MTLComputeCommandEncoder> KernelEncoder() {
        if (kernel_compute == nil) {
            id<MTLCommandBuffer> target = nil;
            if (MergeAheadEnabled()) {
                if (upload_blit != nil) {
                    [upload_blit endEncoding];
                    upload_blit = nil;
                }
                target = AheadBuffer();
            } else {
                kernel_buffer = NewCommandBuffer(
                    StaticLabelsEnabled()
                        ? @"Lichii kernels"
                        : [NSString stringWithFormat:@"Lichii %llu kernels", current_tick]);
                target = kernel_buffer;
            }
            kernel_compute = [target computeCommandEncoder];
            kernel_compute.label = @"Kernels";
            kernel_bound = {};
            kernel_pipeline = nullptr;
            ++encoders;
        }
        return kernel_compute;
    }

    /// Commits a command buffer of the tick being recorded that is not its last one: the tick
    /// is done when the one Flush commits is.
    void CommitEarly(id<MTLCommandBuffer> buffer, u64 tick, CommitReason reason) {
        if (buffer == nil) {
            return;
        }
        commit_counts[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
        std::atomic<u64>* const gpu_time = &gpu_ns;
        std::atomic<u32>* const errors = &errors_logged;
        [buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            const CFTimeInterval busy = finished.GPUEndTime - finished.GPUStartTime;
            if (busy > 0.0) {
                gpu_time->fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
            }
            ReportFailure(finished, tick, errors);
        }];
        [buffer commit];
        if (recorder != nullptr) {
            recorder->Publish();
        }
        ++command_buffers;
        in_flight.emplace_back(tick, buffer);
    }

    /// The uploads and kernels recorded ahead, before the segment they belong to.
    void CommitAhead(u64 tick) {
        if (upload_blit != nil) {
            [upload_blit endEncoding];
            upload_blit = nil;
        }
        EndKernelEncoder();
        kernel_resume = false;
        kernel_checked = false;
        const id<MTLCommandBuffer> uploads = upload_buffer;
        const id<MTLCommandBuffer> kernels = kernel_buffer;
        const id<MTLCommandBuffer> ahead = ahead_buffer;
        upload_buffer = nil;
        kernel_buffer = nil;
        ahead_buffer = nil;
        CommitEarly(uploads, tick, CommitReason::Ahead);
        CommitEarly(kernels, tick, CommitReason::Ahead);
        CommitEarly(ahead, tick, CommitReason::Ahead);
    }

    void NextSegment() {
        ++segment_serial;
        segment_encoders = 0;
        segment_render_passes = 0;
        ahead_blocked = false;
        in_order_buffers.clear();
        segment_bound.clear();
        segment_written_textures.clear();
        segment_touched_textures.clear();
        segment_shared_writes = false;
        segment_opaque = false;
        checked_failed = false;
        ahead_query_relaxed = false;
        compute_clobbered_by_copy = false;
    }

    /// A copy recorded in order reads or writes `buffer`: no copy into it goes ahead any more.
    void NoteInOrder(BufferHandle buffer) {
        if (std::find(in_order_buffers.rbegin(), in_order_buffers.rend(), buffer) ==
            in_order_buffers.rend()) {
            in_order_buffers.push_back(buffer);
        }
    }

    /// In-order work writes `buffer` (a copy's destination).
    void NoteInOrderWrite(BufferHandle buffer) {
        NoteInOrder(buffer);
        NoteSegmentBuffer(buffer, true);
    }

    /// EDEN_NX2_METAL_SEGMENT_NOTE_CACHE: whether the segment noted `object` (a buffer or a
    /// texture) already as this note would (writing, or as `writes_shared` asks): what a note
    /// adds stays until NextSegment. Else it is recorded as noted, the caller noting it now. The
    /// segment's command buffer holds what its work binds: an address names one object.
    bool NotedInSegment(const void* object, bool writes_shared) {
        if (!SegmentNoteCacheEnabled()) {
            return false;
        }
        const uintptr_t key = reinterpret_cast<uintptr_t>(object);
        SegmentNote& note =
            segment_notes[((key >> 4) ^ (key >> 12)) % segment_notes.size()];
        if (note.object == object && note.serial == segment_serial) {
            if (note.writes_shared || !writes_shared) {
                ++segment_note_skips;
                return true;
            }
            note.writes_shared = true;
            return false;
        }
        note = {object, segment_serial, writes_shared};
        return false;
    }

    /// EDEN_NX2_METAL_AHEAD_ANY: in-order work of the segment modifies `buffer`; read-only UBO/SSBO
    /// binds (`writes` false) do not conflict with ahead compute kernels.
    void NoteSegmentBuffer(const void* buffer, bool writes) {
        if (!writes || buffer == nullptr || !AheadAnyEnabled() || NotedInSegment(buffer, true)) {
            return;
        }
        if (AsBuffer(const_cast<void*>(buffer)).storageMode == MTLStorageModeShared) {
            segment_shared_writes = true;
        } else {
            segment_bound.insert(buffer);
        }
    }

    void NoteTextureTouched(const void* texture) {
        if (texture == nullptr) {
            return;
        }
        const uintptr_t key = reinterpret_cast<uintptr_t>(texture);
        TouchedNote& cached =
            segment_touched_cache[((key >> 4) ^ (key >> 12)) % segment_touched_cache.size()];
        if (cached.object == texture && cached.serial == segment_serial) {
            return;
        }
        cached = {texture, segment_serial};
        const id<MTLTexture> root = RootOf(AsTexture(const_cast<void*>(texture)));
        if (root != nil) {
            segment_touched_textures.insert((__bridge const void*)root);
        }
    }

    void NoteTextureWrite(const void* texture) {
        if (texture == nullptr) {
            return;
        }
        const id<MTLTexture> root = RootOf(AsTexture(const_cast<void*>(texture)));
        if (root != nil) {
            const void* const raw_root = (__bridge const void*)root;
            segment_written_textures.insert(raw_root);
            segment_touched_textures.insert(raw_root);
        }
    }

    /// The buffer a texture view of a buffer is on.
    void NoteSegmentTexture(const void* texture, bool writes) {
        if (!writes || texture == nullptr || !AheadAnyEnabled() || NotedInSegment(texture, true)) {
            return;
        }
        const id<MTLBuffer> buffer = RootOf(AsTexture(const_cast<void*>(texture))).buffer;
        if (buffer != nil) {
            NoteSegmentBuffer((__bridge const void*)buffer, true);
        }
    }

    /// Whether the checked kernel binding `buffer` may see something in-order work of the
    /// segment writes (it would run before that work).
    bool CheckedConflict(const void* buffer) const {
        if (buffer == nullptr) {
            return false;
        }
        if (AsBuffer(const_cast<void*>(buffer)).storageMode == MTLStorageModeShared) {
            // Ring allocations the kernel writes or CPU uploads it reads.
            return segment_shared_writes;
        }
        return segment_bound.contains(buffer);
    }

    void LogComputeBind(ComputeBind::Kind kind, u32 slot, void* object, u32 offset, bool write) {
        compute_binds.push_back({kind, write, slot, object, offset, 0});
    }

    void LogComputeBytes(u32 slot, const void* data, u32 size) {
        const u32 at = static_cast<u32>(compute_bind_bytes.size());
        const u8* const bytes = static_cast<const u8*>(data);
        compute_bind_bytes.insert(compute_bind_bytes.end(), bytes, bytes + size);
        compute_binds.push_back({ComputeBind::Kind::Bytes, false, slot, nullptr, at, size});
    }

    /// Sets the kernel's pipeline and binds again on the active compute encoder (a new one).
    void ReplayComputeBinds(const void** bound_pipeline, bool in_order) {
        if (active_compute == nil) {
            return;
        }
        if (!StateCacheEnabled() || *bound_pipeline != compute_bind_pipeline) {
            *bound_pipeline = compute_bind_pipeline;
            ++state_sets;
            [active_compute
                setComputePipelineState:(__bridge id<MTLComputePipelineState>)compute_bind_pipeline];
        }
        for (const ComputeBind& bind : compute_binds) {
            switch (bind.kind) {
            case ComputeBind::Kind::Buffer:
                SetBuffer(ShaderStage::Compute, bind.slot, bind.object, bind.offset);
                if (in_order) {
                    NoteSegmentBuffer(bind.object, true);
                }
                break;
            case ComputeBind::Kind::Bytes:
                if (bind.slot < BufferSlots) {
                    active_bound->buffers[bind.slot] = &InlineBytes;
                }
                ++state_sets;
                [active_compute setBytes:compute_bind_bytes.data() + bind.offset
                                  length:bind.size
                                 atIndex:bind.slot];
                break;
            case ComputeBind::Kind::Texture:
                if (NoteObject(active_bound->textures, bind.slot, bind.object)) {
                    [active_compute setTexture:AsTexture(bind.object) atIndex:bind.slot];
                }
                if (in_order) {
                    NoteSegmentTexture(bind.object, true);
                }
                break;
            case ComputeBind::Kind::Sampler:
                if (NoteObject(active_bound->samplers, bind.slot, bind.object)) {
                    [active_compute setSamplerState:AsSampler(bind.object) atIndex:bind.slot];
                }
                break;
            case ComputeBind::Kind::UseBuffer:
                [active_compute useResource:AsBuffer(bind.object)
                                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                if (in_order) {
                    NoteSegmentBuffer(bind.object, true);
                }
                break;
            case ComputeBind::Kind::UseTexture:
                [active_compute useResource:AsTexture(bind.object)
                                      usage:bind.write
                                                ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                                                : MTLResourceUsageRead];
                if (in_order) {
                    NoteSegmentTexture(bind.object, true);
                }
                break;
            }
        }
    }

    /// EDEN_NX2_METAL_COMPUTE_COPY: a buffer copy as a dispatch on the open compute encoder
    /// `encoder` (bound as `bound`, its pipeline in `bound_pipeline`), in place of ending it for
    /// a blit encoder and opening another after the copy. Metal orders the dispatch with the
    /// encoder's others on the buffers' tracked hazards, as it orders the encoders. False when
    /// the copy kernel is not there (the caller blits).
    bool CopyByCompute(id<MTLComputeCommandEncoder> encoder, StageBindings& bound,
                       const void** bound_pipeline, BufferHandle src, u64 src_offset,
                       BufferHandle dst, u64 dst_offset, u64 size) {
        if (copy_kernel == nil) {
            if (copy_kernel_failed || library == nil) {
                return false;
            }
            NSError* error = nil;
            const id<MTLFunction> function = [library newFunctionWithName:@"copy_bytes"];
            copy_kernel = function != nil
                              ? [device.GetDevice() newComputePipelineStateWithFunction:function
                                                                                  error:&error]
                              : nil;
            if (copy_kernel == nil) {
                copy_kernel_failed = true;
                LOG_ERROR(Render_Metal, "Metal: buffer copy kernel failed: {}",
                          error != nil ? error.localizedDescription.UTF8String : "no function");
                return false;
            }
        }
        const void* const copy_pipe = (__bridge const void*)copy_kernel;
        if (!StateCacheEnabled() || *bound_pipeline != copy_pipe) {
            *bound_pipeline = copy_pipe;
            ++state_sets;
            [encoder setComputePipelineState:copy_kernel];
        } else {
            ++state_skips;
        }
        switch (NoteBuffer(bound, 0, src, 0)) {
        case Bound::Same:
            break;
        case Bound::Offset:
            [encoder setBufferOffset:0 atIndex:0];
            break;
        case Bound::Other:
            [encoder setBuffer:AsBuffer(src) offset:0 atIndex:0];
            break;
        }
        switch (NoteBuffer(bound, 1, dst, 0)) {
        case Bound::Same:
            break;
        case Bound::Offset:
            [encoder setBufferOffset:0 atIndex:1];
            break;
        case Bound::Other:
            [encoder setBuffer:AsBuffer(dst) offset:0 atIndex:1];
            break;
        }
        bound.buffers[2] = &InlineBytes;
        ++state_sets;
        const std::array<u64, 3> params{src_offset, dst_offset, size};
        [encoder setBytes:params.data() length:sizeof(params) atIndex:2];
        const u64 threads = (size + 15) / 16;
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((threads + 63) / 64), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (active_compute == encoder) {
            compute_clobbered_by_copy = true;
        }
        return true;
    }

    /// The segment's compute encoder, opened when there is none, made the active one.
    void OpenInOrderCompute() {
        if (compute == nil) {
            EndEncoder(PassEnd::Compute);
            compute = [Commands() computeCommandEncoder];
            compute.label = @"Compute";
            compute_bound = {};
            compute_pipeline = nullptr;
            ++encoders;
            ++segment_encoders;
        }
        active_compute = compute;
        active_bound = &compute_bound;
        compute_clobbered_by_copy = false;
    }

    /// Before a compute bind or dispatch: the ahead kernel an upload interrupted goes on in a
    /// new encoder, or restores slots 0..2 if CopyByCompute used the active encoder.
    void ResumeKernel() {
        if (kernel_resume) {
            kernel_resume = false;
            compute_clobbered_by_copy = false;
            active_compute = KernelEncoder();
            active_bound = &kernel_bound;
            ReplayComputeBinds(&kernel_pipeline, false);
            return;
        }
        if (!compute_clobbered_by_copy) {
            return;
        }
        compute_clobbered_by_copy = false;
        if (active_compute == nil) {
            return;
        }
        const void** const bound_pipeline =
            active_compute == kernel_compute ? &kernel_pipeline : &compute_pipeline;
        if (compute_bind_pipeline != nullptr &&
            (!StateCacheEnabled() || *bound_pipeline != compute_bind_pipeline)) {
            *bound_pipeline = compute_bind_pipeline;
            ++state_sets;
            [active_compute
                setComputePipelineState:(__bridge id<MTLComputePipelineState>)compute_bind_pipeline];
        }
        for (const ComputeBind& bind : compute_binds) {
            if (bind.slot >= 3) {
                continue;
            }
            if (bind.kind == ComputeBind::Kind::Buffer) {
                SetBuffer(ShaderStage::Compute, bind.slot, bind.object, bind.offset);
            } else if (bind.kind == ComputeBind::Kind::Bytes) {
                active_bound->buffers[bind.slot] = &InlineBytes;
                ++state_sets;
                [active_compute setBytes:compute_bind_bytes.data() + bind.offset
                                  length:bind.size
                                 atIndex:bind.slot];
            }
        }
    }

    /// The checked kernel conflicts with in-order work: it goes in order from here on, with the
    /// binds it has made so far.
    void LeaveChecked(TextureHandle texture) {
        kernel_checked = false;
        checked_failed = true;
        ++ahead_fallbacks;
        // As the rasterizer does for a kernel that ends the pass it draws into.
        if (render != nil && render_targets == ahead_query_targets && AheadSplitEnabled() &&
            !Wanted(ahead_query_targets)) {
            wanted[wanted_count++ % wanted.size()] = ahead_query_targets;
        }
        if (!ClearNarrowEnabled()) {
            FlushClear();
        } else if (texture != nullptr) {
            Touch(texture);
        }
        OpenInOrderCompute();
        ReplayComputeBinds(&compute_pipeline, true);
    }

    /// Before a compute bind of `buffer`: checks the checked kernel, notes in-order binds.
    void ComputeBuffer(const void* buffer, bool writes = true) {
        ResumeKernel();
        if (kernel_checked && CheckedConflict(buffer)) {
            LeaveChecked(nullptr);
        }
        if (active_compute != nil && active_compute == compute) {
            NoteSegmentBuffer(buffer, writes);
        }
    }

    /// Before a compute bind of `texture`: a checked kernel only falls back in order when the
    /// root texture (or its backing buffer) was modified in the segment or has a pending clear;
    /// an in-order one records a waiting clear of it first.
    void ComputeTexture(TextureHandle texture, bool write = true) {
        ResumeKernel();
        if (texture == nullptr) {
            return;
        }
        NoteTextureTouched(texture);
        if (kernel_checked) {
            const id<MTLTexture> root_tex = RootOf(AsTexture(texture));
            const void* const raw_root = (__bridge const void*)root_tex;
            const id<MTLBuffer> backing = root_tex != nil ? root_tex.buffer : nil;
            if (segment_written_textures.contains(raw_root) || PendingClearOn(texture) ||
                (backing != nil && CheckedConflict((__bridge const void*)backing))) {
                LeaveChecked(texture);
            }
        }
        if (active_compute == nil || active_compute != compute) {
            return;
        }
        if (write) {
            NoteTextureWrite(texture);
        }
        NoteSegmentTexture(texture, write);
        if (pending_clear && PendingClearOn(texture)) {
            EndEncoder();
            FlushClear();
            OpenInOrderCompute();
            ReplayComputeBinds(&compute_pipeline, true);
        }
    }

    /// Commits what the tick recorded so far; the tick goes on in a new segment.
    void Split() {
        EndEncoder(PassEnd::Split);
        CommitAhead(current_tick);
        const id<MTLCommandBuffer> segment = command_buffer;
        command_buffer = nil;
        CommitEarly(segment, current_tick, CommitReason::Split);
        NextSegment();
        ++splits;
        ++tick_splits;
    }

    bool Wanted(const RenderTargetSet& targets) const {
        const auto end = wanted.begin() + (std::min)(wanted_count, wanted.size());
        return std::find(wanted.begin(), end, targets) != end;
    }

    /// 64 KiB of zeros bound in place of what Metal must not be given.
    id<MTLBuffer> ZeroBuffer() {
        if (zero_buffer == nil) {
            zero_buffer = [device.GetDevice() newBufferWithLength:64 * 1024
                                                          options:MTLResourceStorageModePrivate];
        }
        return zero_buffer;
    }

    /// What binding `buffer` at `offset` takes, with `bound` as the encoder has it.
    Bound NoteBuffer(StageBindings& bound, u32 slot, const void* buffer, u32 offset) {
        if (!StateCacheEnabled() || slot >= bound.buffers.size()) {
            ++state_sets;
            return Bound::Other;
        }
        const bool same_buffer = bound.buffers[slot] == buffer;
        if (same_buffer && (buffer == nullptr || bound.offsets[slot] == offset)) {
            ++state_skips;
            return Bound::Same;
        }
        bound.buffers[slot] = buffer;
        bound.offsets[slot] = offset;
        ++state_sets;
        return same_buffer ? Bound::Offset : Bound::Other;
    }

    /// Whether `object` has to be set where `bound` is.
    template <size_t N>
    bool NoteObject(std::array<const void*, N>& bound, u32 slot, const void* object) {
        if (!StateCacheEnabled() || slot >= N) {
            ++state_sets;
            return true;
        }
        if (bound[slot] == object) {
            ++state_skips;
            return false;
        }
        bound[slot] = object;
        ++state_sets;
        return true;
    }

    void SetBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset) {
        // Metal checks an offset past the buffer's end only in its validation layer; without it
        // the GPU reads past the allocation. Zeros instead.
        if (buffer != nullptr && PipelineCheckEnabled() && offset >= AsBuffer(buffer).length) {
            Coverage::NoteGuardWith("buffer offset past its end", [&] {
                return fmt::format("offset {} of a {}-byte buffer", offset,
                                   static_cast<u64>(AsBuffer(buffer).length));
            });
            buffer = (__bridge BufferHandle)ZeroBuffer();
            offset = 0;
        }
        switch (stage) {
        case ShaderStage::Vertex:
            if (render == nil) {
                return;
            }
            switch (NoteBuffer(vertex_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [render setVertexBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [render setVertexBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        case ShaderStage::Fragment:
            if (render == nil) {
                return;
            }
            switch (NoteBuffer(fragment_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [render setFragmentBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [render setFragmentBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        case ShaderStage::Object:
            if (render == nil) {
                return;
            }
            switch (NoteBuffer(object_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [render setObjectBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [render setObjectBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        case ShaderStage::Mesh:
            if (render == nil) {
                return;
            }
            switch (NoteBuffer(mesh_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [render setMeshBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [render setMeshBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        case ShaderStage::Compute:
            if (active_compute == nil) {
                return;
            }
            switch (NoteBuffer(*active_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [active_compute setBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [active_compute setBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        }
    }

    /// The progress trace: the first of each kind of work at Info, so a log that stops shows how
    /// far the renderer got.
    void CountDraw() {
        if (draws++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first draw encoded (command buffer {}, render pass {})",
                     current_tick, render_pass_serial);
        }
    }

    void CountDispatch() {
        if (dispatches++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first dispatch encoded (command buffer {})",
                     current_tick);
        }
    }

    /// Sets the store actions BeginRendering left open (EDEN_NX2_METAL_UNWRITTEN_STORE).
    void FinishStores() {
        if (deferred_color == 0 && !deferred_depth && !deferred_stencil && !cleared_depth_open &&
            !cleared_stencil_open) {
            kill_depth = false;
            kill_stencil = false;
            return;
        }
        // A pass whose draws never set their state is taken to write everything.
        const u8 color = pass_state_set ? written_color : 0xFF;
        const bool depth = !pass_state_set || written_depth;
        const bool stencil = !pass_state_set || written_stencil;
        const auto action = [this](bool written) {
            if (!written) {
                ++stores_dropped;
            }
            return written ? MTLStoreActionStore : MTLStoreActionDontCare;
        };
        for (u32 index = 0; index < 8; ++index) {
            if ((deferred_color >> index) & 1U) {
                [render setColorStoreAction:action((color >> index) & 1U) atIndex:index];
            }
        }
        // EDEN_NX2_METAL_DEPTH_KILL: a clear of the whole attachment comes next.
        if (deferred_depth || cleared_depth_open) {
            [render setDepthStoreAction:action((cleared_depth_open || depth) && !kill_depth)];
        }
        if (deferred_stencil || cleared_stencil_open) {
            [render setStencilStoreAction:action((cleared_stencil_open || stencil) &&
                                                 !kill_stencil)];
        }
        deferred_color = 0;
        deferred_depth = false;
        deferred_stencil = false;
        cleared_depth_open = false;
        cleared_stencil_open = false;
        kill_depth = false;
        kill_stencil = false;
    }

    /// Q27: a multisampled colour texture's single-sampled resolve shadow. `valid`: it holds the
    /// resolve of the samples as they are; `dropped`: the samples were not stored (the pass
    /// cleared them and resolved, MTLStoreActionMultisampleResolve), so anything that needs them
    /// restores them from the shadow first.
    struct ShadowEntry {
        void* shadow{};
        bool valid{};
        bool dropped{};
    };
    std::unordered_map<void*, ShadowEntry> shadows;
    u64 samples_dropped{};
    u64 samples_restored{};

    id<MTLRenderPipelineState> RestorePipeline(MTLPixelFormat format, NSUInteger samples) {
        const u64 key = (u64{1} << 62) | (static_cast<u64>(format) << 8) | static_cast<u64>(samples);
        if (const auto it = scale_pipelines.find(key); it != scale_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        if (library != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Restore samples";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:@"restore_samples_float"];
            desc.rasterSampleCount = samples;
            desc.colorAttachments[0].pixelFormat = format;
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
        }
        scale_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The samples of `texture` written back from its shadow (each sample the resolved colour).
    void RestoreSamples(void* texture, ShadowEntry& entry) {
        entry.dropped = false;
        const id<MTLTexture> dst = AsTexture(texture);
        const id<MTLTexture> src = AsTexture(entry.shadow);
        if (dst == nil || src == nil) {
            return;
        }
        ClearBefore({texture, entry.shadow});
        const id<MTLRenderPipelineState> pipeline = RestorePipeline(dst.pixelFormat, dst.sampleCount);
        if (pipeline == nil) {
            return;
        }
        ++samples_restored;
        const u32 width = static_cast<u32>(dst.width);
        const u32 height = static_cast<u32>(dst.height);
        for (u32 layer = 0; layer < dst.arrayLength; ++layer) {
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            ApplyAttachment(pass.colorAttachments[0],
                            Attachment{.texture = texture, .level = 0, .slice = layer});
            pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.renderTargetWidth = width;
            pass.renderTargetHeight = height;
            const id<MTLRenderCommandEncoder> encoder = RenderEncoder(pass);
            if (encoder == nil) {
                return;
            }
            [encoder setRenderPipelineState:pipeline];
            [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width),
                                             static_cast<f64>(height), 0.0, 1.0}];
            const f32 depth_value = 0.0f;
            [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
            [encoder setFragmentTexture:src atIndex:0];
            [encoder setFragmentBytes:&layer length:sizeof(layer) atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [encoder endEncoding];
        }
    }

    /// Something reads the samples of `texture`.
    void SamplesNeeded(const void* texture) {
        NoteTextureTouched(texture);
        if (shadows.empty()) {
            return;
        }
        if (const auto it = shadows.find(const_cast<void*>(texture));
            it != shadows.end() && it->second.dropped) {
            RestoreSamples(it->first, it->second);
        }
    }

    /// Something writes part or all of the samples of `texture`: the resolve no longer holds.
    void Written(const void* texture) {
        if (shadows.empty()) {
            return;
        }
        if (const auto it = shadows.find(const_cast<void*>(texture)); it != shadows.end()) {
            if (it->second.dropped) {
                RestoreSamples(it->first, it->second);
            }
            it->second.valid = false;
        }
    }

    void EndEncoder(PassEnd why = PassEnd::Other) {
        if (blit != nil) {
            [blit endEncoding];
            blit = nil;
        }
        if (render != nil) {
            ++pass_ends[static_cast<size_t>(why)];
            FinishStores();
            [render endEncoding];
            render = nil;
        }
        if (pass_open) {
            pass_open = false;
            pass_records.back().draws = draws - pass_records.back().draws_begin;
        }
        if (compute != nil) {
            if (active_compute == compute) {
                active_compute = nil;
                active_bound = nullptr;
                compute_clobbered_by_copy = false;
            }
            [compute endEncoding];
            compute = nil;
        }
    }

    id<MTLBlitCommandEncoder> BlitEncoder() {
        if (blit == nil) {
            EndEncoder(PassEnd::Blit);
            blit = [Commands() blitCommandEncoder];
            blit.label = @"Blit";
            ++encoders;
            ++segment_encoders;
        }
        ++blit_commands;
        return blit;
    }

    /// why: what the pass is (NewPass: a pass for draws, Clear, else Helper), what it ends.
    id<MTLRenderCommandEncoder> RenderEncoder(MTLRenderPassDescriptor* pass,
                                              PassEnd why = PassEnd::Helper) {
        EndEncoder(why);
        ++passes_begun[why == PassEnd::NewPass ? 0 : why == PassEnd::Clear ? 1 : 2];
        ++encoders;
        ++segment_encoders;
        ++segment_render_passes;
        ++render_passes;
        for (u32 i = 0; i < 8; ++i) {
            if (const id<MTLTexture> tex = pass.colorAttachments[i].texture; tex != nil) {
                NoteTextureWrite((__bridge const void*)tex);
            }
            if (const id<MTLTexture> res = pass.colorAttachments[i].resolveTexture; res != nil) {
                NoteTextureWrite((__bridge const void*)res);
            }
        }
        if (const id<MTLTexture> tex = pass.depthAttachment.texture; tex != nil) {
            NoteTextureWrite((__bridge const void*)tex);
        }
        if (const id<MTLTexture> res = pass.depthAttachment.resolveTexture; res != nil) {
            NoteTextureWrite((__bridge const void*)res);
        }
        if (const id<MTLTexture> tex = pass.stencilAttachment.texture; tex != nil) {
            NoteTextureWrite((__bridge const void*)tex);
        }
        if (const id<MTLTexture> res = pass.stencilAttachment.resolveTexture; res != nil) {
            NoteTextureWrite((__bridge const void*)res);
        }
        // A timed frame (EDEN_NX2_PASS_TIMES): the pass's start and end on the GPU's clock.
        const size_t sample = pass_records.size() * 2;
        const bool timed =
            pass_timing && pass_samples != nil && sample + 2 <= pass_samples.sampleCount;
        if (timed) {
            MTLRenderPassSampleBufferAttachmentDescriptor* const attachment =
                pass.sampleBufferAttachments[0];
            attachment.sampleBuffer = pass_samples;
            attachment.startOfVertexSampleIndex = sample;
            attachment.endOfVertexSampleIndex = MTLCounterDontSample;
            attachment.startOfFragmentSampleIndex = MTLCounterDontSample;
            attachment.endOfFragmentSampleIndex = sample + 1;
        }
        const id<MTLRenderCommandEncoder> encoder =
            [Commands() renderCommandEncoderWithDescriptor:pass];
        if (timed) {
            pass.sampleBufferAttachments[0].sampleBuffer = nil;
            if (encoder != nil) {
                const id<MTLTexture> color = pass.colorAttachments[0].texture;
                const id<MTLTexture> depth = pass.depthAttachment.texture;
                u32 pass_width = static_cast<u32>(pass.renderTargetWidth);
                u32 pass_height = static_cast<u32>(pass.renderTargetHeight);
                if (pass_width == 0 || pass_height == 0) {
                    if (color != nil) {
                        const u32 level = static_cast<u32>(pass.colorAttachments[0].level);
                        pass_width = (std::max)(static_cast<u32>(color.width) >> level, 1U);
                        pass_height = (std::max)(static_cast<u32>(color.height) >> level, 1U);
                    } else if (depth != nil) {
                        const u32 level = static_cast<u32>(pass.depthAttachment.level);
                        pass_width = (std::max)(static_cast<u32>(depth.width) >> level, 1U);
                        pass_height = (std::max)(static_cast<u32>(depth.height) >> level, 1U);
                    }
                }
                pass_records.push_back({
                    .width = pass_width,
                    .height = pass_height,
                    .color_format = color != nil ? static_cast<u32>(color.pixelFormat) : 0U,
                    .depth_format = depth != nil ? static_cast<u32>(depth.pixelFormat) : 0U,
                    .draws_begin = draws,
                    .draws = 0,
                });
                pass_open = true;
            }
        }
        if (encoder == nil) {
            // Metal refused the pass (draws into it are dropped): name it once.
            if (refused_passes++ < 4) {
                LOG_ERROR(Render_Metal, "Metal: render pass refused ({}x{}, colour 0 {}, depth {})",
                          pass.renderTargetWidth, pass.renderTargetHeight,
                          pass.colorAttachments[0].texture != nil
                              ? static_cast<u32>(pass.colorAttachments[0].texture.pixelFormat)
                              : 0U,
                          pass.depthAttachment.texture != nil
                              ? static_cast<u32>(pass.depthAttachment.texture.pixelFormat)
                              : 0U);
            }
            return nil;
        }
        encoder.label = StaticLabelsEnabled()
                            ? @"Pass"
                            : [NSString stringWithFormat:@"Pass %llu", render_pass_serial + 1];
        return encoder;
    }

    /// A render pass that only clears.
    void RecordClear(const ClearPass& clear) {
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        bool any = false;
        for (size_t index = 0; index < clear.color.size(); ++index) {
            if (clear.color[index].texture == nullptr || !clear.clear_color[index]) {
                continue;
            }
            MTLRenderPassColorAttachmentDescriptor* color = pass.colorAttachments[index];
            ApplyAttachment(color, clear.color[index]);
            if (const auto shadow = shadows.find(clear.color[index].texture); shadow != shadows.end()) {
                shadow->second.valid = false;
                shadow->second.dropped = false;
            }
            const auto& value = clear.color_value[index];
            color.loadAction = MTLLoadActionClear;
            color.storeAction = MTLStoreActionStore;
            color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
            any = true;
        }
        if (clear.depth.texture != nullptr && clear.clear_depth) {
            ApplyAttachment(pass.depthAttachment, clear.depth);
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = clear.depth_value;
            any = true;
        }
        if (clear.stencil.texture != nullptr && clear.clear_stencil) {
            ApplyAttachment(pass.stencilAttachment, clear.stencil);
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = clear.stencil_value;
            any = true;
        }
        if (!any) {
            return;
        }
        pass.renderTargetWidth = std::clamp(clear.width, 1U, MaxRenderTargetSize);
        pass.renderTargetHeight = std::clamp(clear.height, 1U, MaxRenderTargetSize);
        [RenderEncoder(pass, PassEnd::Clear) endEncoding];
    }

    /// The clear that waits for a render pass gets one of its own: work that may read or write
    /// its attachments comes next.
    void FlushClear() {
        if (pending_clear) {
            const ClearPass clear = *pending_clear;
            pending_clear.reset();
            RecordClear(clear);
            pending_clear_textures.clear();
        }
    }

    void HoldClear(const ClearPass& clear) {
        pending_clear_textures.clear();
        for (const Attachment& attachment : clear.color) {
            if (attachment.texture != nullptr) {
                pending_clear_textures.push_back(AsTexture(attachment.texture));
            }
        }
        for (const Attachment* const attachment : {&clear.depth, &clear.stencil}) {
            if (attachment->texture != nullptr) {
                pending_clear_textures.push_back(AsTexture(attachment->texture));
            }
        }
    }

    /// A copy reads or writes `texture`: the waiting clear of it (or of what it is a view of)
    /// goes first.
    void Touch(TextureHandle texture) {
        if (pending_clear && PendingClearOn(texture)) {
            FlushClear();
        }
    }

    /// Whether the waiting clear is on `texture` (or what it is a view of).
    bool PendingClearOn(TextureHandle texture) const {
        if (!pending_clear || texture == nullptr) {
            return false;
        }
        const id<MTLTexture> root = RootOf(AsTexture(texture));
        const auto same = [&](const Attachment& attachment) {
            return attachment.texture != nullptr && RootOf(AsTexture(attachment.texture)) == root;
        };
        return std::ranges::any_of(pending_clear->color, same) || same(pending_clear->depth) ||
               same(pending_clear->stencil);
    }

    /// Work on `textures` comes next: the waiting clear is recorded first when it is on one of
    /// them (EDEN_NX2_METAL_CLEAR_NARROW), else whatever it is on.
    void ClearBefore(std::initializer_list<TextureHandle> textures) {
        if (!pending_clear) {
            return;
        }
        if (!ClearNarrowEnabled()) {
            FlushClear();
            return;
        }
        for (const TextureHandle texture : textures) {
            Touch(texture);
        }
    }

    id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format, BlitBlend blend) {
        const u64 key = (static_cast<u64>(format) << 8) | static_cast<u64>(blend);
        if (const auto it = blit_pipelines.find(key); it != blit_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        NSString* fragment = nil;
        switch (format_class) {
        case FormatClass::Float: fragment = @"blit_float"; break;
        case FormatClass::Uint: fragment = @"blit_uint"; break;
        case FormatClass::Sint: fragment = @"blit_sint"; break;
        case FormatClass::Depth: fragment = @"blit_depth"; break;
        case FormatClass::DepthStencil:
            fragment = StencilViewFormat(format) != MTLPixelFormatInvalid ? @"blit_depth_stencil"
                                                                          : @"blit_depth";
            break;
        default: break;
        }
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Blit";
            desc.vertexFunction = [library newFunctionWithName:@"blit_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
                desc.depthAttachmentPixelFormat = format;
                if (format_class == FormatClass::DepthStencil) {
                    desc.stencilAttachmentPixelFormat = format;
                }
            } else {
                desc.colorAttachments[0].pixelFormat = format;
            }
            if (blend != BlitBlend::None && format_class == FormatClass::Float) {
                MTLRenderPipelineColorAttachmentDescriptor* color = desc.colorAttachments[0];
                color.blendingEnabled = YES;
                color.sourceRGBBlendFactor = blend == BlitBlend::Premultiplied
                                                 ? MTLBlendFactorOne
                                                 : MTLBlendFactorSourceAlpha;
                color.sourceAlphaBlendFactor = color.sourceRGBBlendFactor;
                color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: blit pipeline for format {} failed: {}",
                          static_cast<u32>(format), error.localizedDescription.UTF8String);
            }
        }
        blit_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The depth blits' state: every fragment writes its depth, and the stencil it outputs.
    id<MTLDepthStencilState> DepthWriteAlways() {
        if (depth_write_always == nil) {
            MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
            desc.depthCompareFunction = MTLCompareFunctionAlways;
            desc.depthWriteEnabled = YES;
            MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = MTLCompareFunctionAlways;
            stencil.depthStencilPassOperation = MTLStencilOperationReplace;
            stencil.readMask = 0xFF;
            stencil.writeMask = 0xFF;
            desc.frontFaceStencil = stencil;
            desc.backFaceStencil = stencil;
            depth_write_always = [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
        }
        return depth_write_always;
    }

    /// The pipeline of a drawn clear: colour 0 with a write mask, or depth and stencil alone.
    id<MTLRenderPipelineState> ClearPipeline(MTLPixelFormat color, u8 mask, MTLPixelFormat depth,
                                             bool stencil, NSUInteger samples) {
        const u64 key = (static_cast<u64>(color) << 32) | (static_cast<u64>(depth) << 12) |
                        (static_cast<u64>(samples) << 5) | (static_cast<u64>(mask) << 1) |
                        (stencil ? 1U : 0U);
        if (const auto it = clear_pipelines.find(key); it != clear_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        NSString* fragment = nil;
        if (color != MTLPixelFormatInvalid) {
            switch (ClassOf(static_cast<u32>(color))) {
            case FormatClass::Float: fragment = @"clear_float"; break;
            case FormatClass::Uint: fragment = @"clear_uint"; break;
            case FormatClass::Sint: fragment = @"clear_sint"; break;
            default: break;
            }
        }
        if (library != nil && (fragment != nil || color == MTLPixelFormatInvalid)) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Clear";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.rasterSampleCount = samples;
            if (fragment != nil) {
                desc.fragmentFunction = [library newFunctionWithName:fragment];
                desc.colorAttachments[0].pixelFormat = color;
                MTLColorWriteMask write = MTLColorWriteMaskNone;
                write |= (mask & 1) != 0 ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
                write |= (mask & 2) != 0 ? MTLColorWriteMaskGreen : MTLColorWriteMaskNone;
                write |= (mask & 4) != 0 ? MTLColorWriteMaskBlue : MTLColorWriteMaskNone;
                write |= (mask & 8) != 0 ? MTLColorWriteMaskAlpha : MTLColorWriteMaskNone;
                desc.colorAttachments[0].writeMask = write;
            }
            if (depth != MTLPixelFormatInvalid && !stencil) {
                desc.depthAttachmentPixelFormat = depth;
            } else if (depth != MTLPixelFormatInvalid) {
                // A stencil-only format has no depth to attach.
                if (depth != MTLPixelFormatStencil8) {
                    desc.depthAttachmentPixelFormat = depth;
                }
                desc.stencilAttachmentPixelFormat = depth;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: clear pipeline for formats {} and {} failed: {}",
                          static_cast<u32>(color), static_cast<u32>(depth),
                          error.localizedDescription.UTF8String);
            }
        }
        clear_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The pipeline of a sample copy into `format` with `samples` per pixel (CopySamples).
    id<MTLRenderPipelineState> SamplePipeline(MTLPixelFormat format, NSUInteger samples,
                                              bool expand) {
        const u64 key = (static_cast<u64>(format) << 8) | (static_cast<u64>(samples) << 1) |
                        (expand ? 1U : 0U);
        if (const auto it = sample_pipelines.find(key); it != sample_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        NSString* fragment = nil;
        switch (format_class) {
        case FormatClass::Float: fragment = expand ? @"expand_float" : @"compress_float"; break;
        case FormatClass::Uint: fragment = expand ? @"expand_uint" : @"compress_uint"; break;
        case FormatClass::Sint: fragment = expand ? @"expand_sint" : @"compress_sint"; break;
        case FormatClass::Depth:
        case FormatClass::DepthStencil:
            fragment = expand ? @"expand_depth" : @"compress_depth";
            break;
        default: break;
        }
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Sample copy";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            desc.rasterSampleCount = samples;
            if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
                desc.depthAttachmentPixelFormat = format;
                if (format_class == FormatClass::DepthStencil) {
                    desc.stencilAttachmentPixelFormat = format;
                }
            } else {
                desc.colorAttachments[0].pixelFormat = format;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: sample copy pipeline for format {} failed: {}",
                          static_cast<u32>(format), error.localizedDescription.UTF8String);
            }
        }
        sample_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The pipeline of a nearest scaling into `format` with `samples` per pixel (ScaleNearest).
    id<MTLRenderPipelineState> ScalePipeline(MTLPixelFormat format, NSUInteger samples) {
        const u64 key = (static_cast<u64>(format) << 8) | static_cast<u64>(samples);
        if (const auto it = scale_pipelines.find(key); it != scale_pipelines.end()) {
            return it->second;
        }
        NSString* fragment = nil;
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        switch (format_class) {
        case FormatClass::Float: fragment = @"scale_samples_float"; break;
        case FormatClass::Uint: fragment = @"scale_samples_uint"; break;
        case FormatClass::Sint: fragment = @"scale_samples_sint"; break;
        case FormatClass::Depth: fragment = @"scale_samples_depth"; break;
        case FormatClass::Stencil: fragment = samples == 1 ? @"scale_stencil" : nil; break;
        default: break;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Scale";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            desc.rasterSampleCount = samples;
            if (format_class == FormatClass::Depth) {
                desc.depthAttachmentPixelFormat = format;
            } else if (format_class == FormatClass::Stencil) {
                desc.stencilAttachmentPixelFormat = format;
            } else {
                desc.colorAttachments[0].pixelFormat = format;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: scaling pipeline for format {} failed: {}",
                          static_cast<u32>(format), error.localizedDescription.UTF8String);
            }
        }
        scale_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The pipeline of a format conversion into `format` (Convert).
    id<MTLRenderPipelineState> ConvertPipeline(ConvertKind kind, MTLPixelFormat format) {
        const u64 key = (static_cast<u64>(format) << 8) | static_cast<u64>(kind);
        if (const auto it = convert_pipelines.find(key); it != convert_pipelines.end()) {
            return it->second;
        }
        NSString* fragment = nil;
        switch (kind) {
        case ConvertKind::DepthToColor: fragment = @"convert_depth_to_color"; break;
        case ConvertKind::ColorToDepth: fragment = @"convert_color_to_depth"; break;
        case ConvertKind::Abgr8ToD32f: fragment = @"convert_abgr8_to_d32f"; break;
        case ConvertKind::X8d24ToAbgr8: fragment = @"convert_x8d24_to_abgr8"; break;
        case ConvertKind::Abgr8ToX8d24: fragment = @"convert_abgr8_to_x8d24"; break;
        case ConvertKind::D24s8ToAbgr8: fragment = @"convert_d24s8_to_abgr8"; break;
        case ConvertKind::S8d24ToAbgr8: fragment = @"convert_s8d24_to_abgr8"; break;
        case ConvertKind::Abgr8ToD24s8: fragment = @"convert_abgr8_to_d24s8"; break;
        }
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        const bool depth_target =
            format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
        const bool to_depth = kind == ConvertKind::ColorToDepth ||
                              kind == ConvertKind::Abgr8ToD32f ||
                              kind == ConvertKind::Abgr8ToX8d24 || kind == ConvertKind::Abgr8ToD24s8;
        id<MTLRenderPipelineState> pipeline = nil;
        // The destination has to be what the conversion writes.
        if (library != nil && depth_target == to_depth &&
            (format_class == FormatClass::Float || depth_target) &&
            (kind != ConvertKind::Abgr8ToD24s8 || format_class == FormatClass::DepthStencil)) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Convert";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            if (depth_target) {
                desc.depthAttachmentPixelFormat = format;
                if (format_class == FormatClass::DepthStencil) {
                    desc.stencilAttachmentPixelFormat = format;
                }
            } else {
                desc.colorAttachments[0].pixelFormat = format;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: conversion {} into format {} failed: {}",
                          static_cast<u32>(kind), static_cast<u32>(format),
                          error.localizedDescription.UTF8String);
            }
        }
        convert_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The state of a drawn clear: depth written or kept, the stencil bits of `mask` replaced.
    id<MTLDepthStencilState> ClearDepthState(bool depth, bool stencil, u32 mask) {
        const u32 key = (depth ? 0x200U : 0U) | (stencil ? 0x100U | (mask & 0xFFU) : 0U);
        if (const auto it = clear_depth_states.find(key); it != clear_depth_states.end()) {
            return it->second;
        }
        MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled = depth ? YES : NO;
        if (stencil) {
            MTLStencilDescriptor* face = [MTLStencilDescriptor new];
            face.stencilCompareFunction = MTLCompareFunctionAlways;
            face.depthStencilPassOperation = MTLStencilOperationReplace;
            face.readMask = 0xFF;
            face.writeMask = mask & 0xFFU;
            desc.frontFaceStencil = face;
            desc.backFaceStencil = face;
        }
        const id<MTLDepthStencilState> state =
            [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
        clear_depth_states.emplace(key, state);
        return state;
    }

    id<MTLRenderPipelineState> InPassClearPipeline(const InPassClearKey& key) {
        if (const auto it = inpass_clear_pipelines.find(key); it != inpass_clear_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        NSString* fragment = nil;
        const u32 clear_count = static_cast<u32>(std::popcount(key.color_clear_mask));
        if (clear_count == 0) {
            fragment = @"clear_none";
        } else if (clear_count == 1) {
            const u32 rt = static_cast<u32>(std::countr_zero(key.color_clear_mask));
            switch (ClassOf(key.color_formats[rt])) {
            case FormatClass::Float:
                fragment = [NSString stringWithFormat:@"clear_float_rt%u", rt];
                break;
            case FormatClass::Uint:
                fragment = [NSString stringWithFormat:@"clear_uint_rt%u", rt];
                break;
            case FormatClass::Sint:
                fragment = [NSString stringWithFormat:@"clear_sint_rt%u", rt];
                break;
            default:
                break;
            }
        } else {
            bool all_float = true;
            for (const u32 fmt : key.color_formats) {
                if (fmt != 0 && ClassOf(fmt) != FormatClass::Float) {
                    all_float = false;
                    break;
                }
            }
            if (all_float) {
                fragment = @"clear_float_mrt";
            }
        }
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"In-pass clear";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            desc.rasterSampleCount = (std::max)(u32{key.samples}, 1U);
            for (u32 i = 0; i < 8; ++i) {
                if (key.color_formats[i] != 0) {
                    desc.colorAttachments[i].pixelFormat =
                        static_cast<MTLPixelFormat>(key.color_formats[i]);
                    desc.colorAttachments[i].writeMask =
                        ((key.color_clear_mask >> i) & 1U) != 0 ? MTLColorWriteMaskAll
                                                                : MTLColorWriteMaskNone;
                }
            }
            if (key.depth_format != 0) {
                desc.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(key.depth_format);
            }
            if (key.stencil_format != 0) {
                desc.stencilAttachmentPixelFormat =
                    static_cast<MTLPixelFormat>(key.stencil_format);
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: in-pass clear pipeline failed: {}",
                          error != nil ? error.localizedDescription.UTF8String : "no reason");
            }
        }
        inpass_clear_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    bool TryDrawInPassClear(const ClearPass& clear_pass) {
        if (!InPassClearEnabled() || render == nil || render_targets.layers > 1 ||
            render_targets.width == 0 || render_targets.height == 0 ||
            clear_pass.width < render_targets.width || clear_pass.height < render_targets.height) {
            return false;
        }
        u8 color_clear_mask = 0;
        for (size_t i = 0; i < clear_pass.color.size(); ++i) {
            if (!clear_pass.clear_color[i]) {
                continue;
            }
            if (render_targets.color[i].texture == nullptr ||
                clear_pass.color[i] != render_targets.color[i]) {
                return false;
            }
            color_clear_mask |= static_cast<u8>(1U << i);
        }
        if (clear_pass.clear_depth &&
            (!render_targets.has_depth || clear_pass.depth != render_targets.depth)) {
            return false;
        }
        if (clear_pass.clear_stencil &&
            (!render_targets.has_stencil || clear_pass.stencil != render_targets.depth)) {
            return false;
        }
        InPassClearKey key{};
        key.color_clear_mask = color_clear_mask;
        for (size_t i = 0; i < render_targets.color.size(); ++i) {
            if (const id<MTLTexture> tex = AsTexture(render_targets.color[i].texture); tex != nil) {
                key.color_formats[i] = static_cast<u32>(tex.pixelFormat);
                if (key.samples == 0) {
                    key.samples = static_cast<u8>(tex.sampleCount);
                }
            }
        }
        if (render_targets.has_depth || render_targets.has_stencil) {
            if (const id<MTLTexture> tex = AsTexture(render_targets.depth.texture); tex != nil) {
                if (render_targets.has_depth) {
                    key.depth_format = static_cast<u32>(tex.pixelFormat);
                }
                if (render_targets.has_stencil) {
                    key.stencil_format = static_cast<u32>(tex.pixelFormat);
                }
                if (key.samples == 0) {
                    key.samples = static_cast<u8>(tex.sampleCount);
                }
            }
        }
        if (key.samples == 0) {
            return false;
        }
        const id<MTLRenderPipelineState> pipeline = InPassClearPipeline(key);
        if (pipeline == nil) {
            return false;
        }
        if (visibility_slot != -1) {
            [render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
            visibility_slot = -1;
        }
        [render setRenderPipelineState:pipeline];
        if (render_targets.has_depth || render_targets.has_stencil) {
            [render setDepthStencilState:ClearDepthState(clear_pass.clear_depth,
                                                         clear_pass.clear_stencil, 0xFF)];
            if (clear_pass.clear_stencil) {
                [render setStencilReferenceValue:clear_pass.stencil_value];
            }
        }
        const u32 target_w = std::clamp(render_targets.width, 1U, MaxRenderTargetSize);
        const u32 target_h = std::clamp(render_targets.height, 1U, MaxRenderTargetSize);
        [render setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(target_w),
                                        static_cast<f64>(target_h), 0.0, 1.0}];
        [render setScissorRect:MTLScissorRect{0, 0, target_w, target_h}];
        [render setCullMode:MTLCullModeNone];
        [render setTriangleFillMode:MTLTriangleFillModeFill];
        [render setDepthClipMode:MTLDepthClipModeClip];
        [render setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
        if (depth_bounds) {
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
            if (@available(macOS 26.0, iOS 26.0, *)) {
                [render setDepthTestMinBound:0.0f maxBound:1.0f];
            }
#endif
        }
        const f32 depth_val = std::clamp(static_cast<f32>(clear_pass.depth_value), 0.0f, 1.0f);
        [render setVertexBytes:&depth_val length:sizeof(depth_val) atIndex:0];
        vertex_bound.buffers[0] = &InlineBytes;
        if (color_clear_mask != 0) {
            if (std::popcount(color_clear_mask) == 1) {
                const u32 rt = static_cast<u32>(std::countr_zero(color_clear_mask));
                std::array<u32, 4> bits{};
                const FormatClass fmt_class = ClassOf(key.color_formats[rt]);
                for (size_t component = 0; component < bits.size(); ++component) {
                    const f64 val = clear_pass.color_value[rt][component];
                    if (fmt_class == FormatClass::Uint) {
                        bits[component] = static_cast<u32>(val);
                    } else if (fmt_class == FormatClass::Sint) {
                        bits[component] = static_cast<u32>(static_cast<s32>(val));
                    } else {
                        bits[component] = std::bit_cast<u32>(static_cast<f32>(val));
                    }
                }
                [render setFragmentBytes:bits.data() length:sizeof(bits) atIndex:0];
            } else {
                std::array<f32, 32> colors{};
                for (size_t i = 0; i < 8; ++i) {
                    for (size_t c = 0; c < 4; ++c) {
                        colors[i * 4 + c] = static_cast<f32>(clear_pass.color_value[i][c]);
                    }
                }
                [render setFragmentBytes:colors.data() length:sizeof(colors) atIndex:0];
            }
            fragment_bound.buffers[0] = &InlineBytes;
        }
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        written_color |= color_clear_mask;
        written_depth = written_depth || clear_pass.clear_depth;
        written_stencil = written_stencil || clear_pass.clear_stencil;
        render_state_valid = false;
        ++inpass_clears;
        return true;
    }

    const Device& device;
    id<MTLCommandQueue> queue;
    /// EDEN_NX2_METAL_CB_PREMAKE; after `queue`: destroyed (its refills drained) before it.
    CommandBufferPremake premade;
    id<MTLDepthStencilState> depth_write_always = nil;
    /// By ClearPipeline's formats, mask and sample count.
    std::unordered_map<u64, id<MTLRenderPipelineState>> clear_pipelines;
    std::unordered_map<InPassClearKey, id<MTLRenderPipelineState>, InPassClearKeyHash>
        inpass_clear_pipelines;
    std::unordered_map<u32, id<MTLDepthStencilState>> clear_depth_states;
    std::unordered_map<u64, id<MTLRenderPipelineState>> sample_pipelines;
    std::unordered_map<u64, id<MTLRenderPipelineState>> convert_pipelines;
    std::unordered_map<u64, id<MTLRenderPipelineState>> scale_pipelines;
    std::unordered_map<u64, id<MTLRenderPipelineState>> depth_upload_pipelines;
    id<MTLComputePipelineState> depth_download = nil;
    id<MTLComputePipelineState> swap_packed = nil;
    id<MTLComputePipelineState> pack16_from = nil;
    id<MTLComputePipelineState> pack16_to = nil;
    id<MTLBuffer> zero_buffer = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    /// The render encoder draws go to, open until other work needs the command buffer.
    id<MTLRenderCommandEncoder> render = nil;
    RenderTargetSet render_targets{};
    /// A clear not recorded yet: the load actions of the next render pass on its attachments
    /// (EDEN_NX2_METAL_CLEAR_FOLD), with its textures held until then.
    std::optional<ClearPass> pending_clear;
    std::vector<id<MTLTexture>> pending_clear_textures;
    u64 clears_folded{};
    id<MTLComputeCommandEncoder> compute = nil;
    /// EDEN_NX2_METAL_STATE_CACHE: what the open encoders have set, so a draw or dispatch sets
    /// only what differs from the one before it.
    StageBindings vertex_bound;
    StageBindings fragment_bound;
    StageBindings object_bound;
    StageBindings mesh_bound;
    DrawState render_state{};
    bool render_state_valid{};
    /// The device's GPU has the depth bounds test (the Apple10 family).
    bool depth_bounds{};
    /// P07: attachments the open pass loaded (not cleared) have their store action decided when
    /// it ends: stored when a draw could write them, dropped otherwise. Bits are colour
    /// targets; the pass's draws OR what they can write into `written_*`.
    u8 deferred_color{};
    bool deferred_depth{};
    bool deferred_stencil{};
    u8 written_color{};
    bool written_depth{};
    bool written_stencil{};
    bool pass_state_set{};
    /// The open pass cleared its depth or stencil attachment by its load action
    /// (RenderPassWroteDepthStencil).
    bool depth_stencil_cleared{};
    u64 stores_dropped{};
    StageBindings compute_bound;
    const void* compute_pipeline{};
    StageBindings kernel_bound;
    const void* kernel_pipeline{};
    /// The compute encoder binds and dispatches go to (the segment's or the kernels'), and what
    /// it has bound.
    id<MTLComputeCommandEncoder> active_compute = nil;
    StageBindings* active_bound{};
    u64 state_sets{};
    u64 state_skips{};
    /// EDEN_NX2_METAL_AHEAD: the command buffers committed before the segment's, with the one
    /// encoder each has.
    id<MTLCommandBuffer> upload_buffer = nil;
    id<MTLBlitCommandEncoder> upload_blit = nil;
    id<MTLCommandBuffer> kernel_buffer = nil;
    id<MTLComputeCommandEncoder> kernel_compute = nil;
    /// EDEN_NX2_METAL_MERGE_AHEAD: the one command buffer of the uploads and kernels recorded
    /// ahead (upload_blit and kernel_compute are its encoders, at most one of them open).
    id<MTLCommandBuffer> ahead_buffer = nil;
    /// An upload ended the ahead kernel being bound: its encoder is opened again, its binds
    /// replayed, by the next bind or dispatch.
    bool kernel_resume{};
    bool compute_clobbered_by_copy{};
    /// The binds of the kernel since its BeginCompute, to replay on another encoder.
    std::vector<ComputeBind> compute_binds;
    std::vector<u8> compute_bind_bytes;
    void* compute_bind_pipeline{};
    /// EDEN_NX2_METAL_AHEAD_ANY: private buffers modified in order in the segment, root textures
    /// written or touched in the segment, whether in-order work wrote a shared buffer, and whether
    /// it did something not tracked at all.
    std::unordered_set<const void*> segment_bound;
    std::unordered_set<const void*> segment_written_textures;
    std::unordered_set<const void*> segment_touched_textures;
    struct TouchedNote {
        const void* object;
        u64 serial;
    };
    std::array<TouchedNote, 512> segment_touched_cache{};
    bool segment_shared_writes{};
    /// Buffers and textures noted in the segment, by address
    /// (EDEN_NX2_METAL_SEGMENT_NOTE_CACHE).
    std::array<SegmentNote, 512> segment_notes{};
    u64 segment_note_skips{};
    /// EDEN_NX2_METAL_COMPUTE_COPY: the copy kernel, and copies recorded as its dispatches.
    id<MTLComputePipelineState> copy_kernel = nil;
    bool copy_kernel_failed{};
    u64 compute_copies_ahead{};
    u64 compute_copies_in_order{};
    bool segment_opaque{};
    /// The kernel being bound went ahead of a segment with other work: each buffer it binds is
    /// checked; a conflict moves it in order (and every later one of the segment).
    bool kernel_checked{};
    bool checked_failed{};
    /// The last CanRecordAhead answered yes by the check alone, for a draw into these.
    bool ahead_query_relaxed{};
    RenderTargetSet ahead_query_targets{};
    u64 ahead_checked{};
    u64 ahead_fallbacks{};
    /// EDEN_NX2_METAL_DEPTH_KILL: the open pass's depth and stencil cleared by its load action
    /// have their store action left open; a clear of all of them next drops them.
    bool cleared_depth_open{};
    bool cleared_stencil_open{};
    bool kill_depth{};
    bool kill_stencil{};
    u64 segment_serial{1};
    /// Encoders and render passes the segment's command buffer got so far.
    u32 segment_encoders{};
    u32 segment_render_passes{};
    u64 force_split_serial{};
    bool ahead_blocked{};
    /// Buffers the segment's copies read or write in order.
    std::vector<const void*> in_order_buffers;
    /// Segments the tick being recorded committed already.
    u32 tick_splits{};
    /// Render passes that are to begin a segment (WantSplitBefore).
    std::array<RenderTargetSet, 8> wanted{};
    size_t wanted_count{};
    u64 splits{};
    u64 ahead_copies{};
    u64 ahead_kernels{};
    u64 idle_fences{};
    u64 inpass_clears{};
    /// EDEN_NX2_METAL_FENCE_BATCH: work a fence left pending, since when, and the last commit.
    bool flush_pending{};
    std::chrono::steady_clock::time_point flush_pending_since{};
    std::chrono::steady_clock::time_point last_commit{};
    u64 deferred_fences{};
    u64 idle_commits{};
    /// EDEN_NX2_PASS_TIMES: one frame's render passes timed at their stage boundaries.
    struct PassRecord {
        u32 width;
        u32 height;
        u32 color_format;
        u32 depth_format;
        u64 draws_begin;
        u64 draws;
    };
    id<MTLCounterSampleBuffer> pass_samples = nil;
    bool pass_times_unsupported{};
    bool pass_timing{};
    bool pass_open{};
    std::optional<u64> pass_resolve_tick;
    std::chrono::steady_clock::time_point pass_next{};
    std::vector<PassRecord> pass_records;
    u64 pass_gpu_ns_before{};
    /// The GPU's clock and the CPU's steady clock read together when the timed frame began.
    MTLTimestamp pass_clock_gpu{};
    std::chrono::steady_clock::time_point pass_clock_steady{};
    std::string pass_times_line;
    id<MTLBuffer> visibility = nil;
    u64 render_pass_serial{};
    /// The open render encoder's counting slot (-1: not counting).
    s64 visibility_slot{-1};
    id<MTLLibrary> library = nil;
    id<MTLSamplerState> nearest = nil;
    id<MTLSamplerState> linear = nil;
    /// By format << 8 | BlitBlend.
    std::unordered_map<u64, id<MTLRenderPipelineState>> blit_pipelines;

    u64 current_tick{1};
    std::atomic<u64> submitted_tick{0};
    std::atomic<u64> completed_tick{0};
    id<MTLSharedEvent> event = nil;

    u64 CompletedTick() noexcept {
        u64 completed = completed_tick.load(std::memory_order_acquire);
        if (event != nil) {
            const u64 signaled = [event signaledValue];
            while (completed < signaled &&
                   !completed_tick.compare_exchange_weak(completed, signaled,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire)) {
            }
            if (signaled > completed) {
                completed = signaled;
            }
        }
        return completed;
    }
    /// Wakes WaitSubmitted when completed_tick moves.
    std::mutex completed_mutex;
    std::condition_variable completed_cv;
    /// AbandonWaits: WaitCompleted returns at once.
    std::atomic<bool> waits_abandoned{false};
    /// GPU time of the command buffers completed since TakeGpuNanoseconds.
    std::atomic<u64> gpu_ns{0};
    std::deque<std::pair<u64, id<MTLCommandBuffer>>> in_flight;

    u64 encoders{};
    u64 command_buffers{};
    u64 blit_commands{};
    u64 render_passes{};
    /// Since boot (the progress trace).
    u64 draws{};
    u64 dispatches{};
    u64 committed{};
    /// Presents showing a guest layer, and those that had none to show.
    u64 presents{};
    u64 blank_presents{};
    u32 refused_passes{};
    /// Since TakeCommitLine: command buffers committed by CommitReason (Flush may run off the
    /// recording thread: Wait from a fence operation), Flushes that found nothing recorded, and
    /// those of them retired without a command buffer (EDEN_NX2_METAL_EMPTY_SKIP).
    std::array<std::atomic<u64>, static_cast<size_t>(CommitReason::Count)> commit_counts{};
    std::atomic<u64> empty_flushes{0};
    std::atomic<u64> empty_skipped{0};
    /// List ends left pending (DeferListEnd).
    u64 list_ends_deferred{};
    /// Since TakePassEndLine: render encoders ended by PassEnd, and passes begun (draws, clears,
    /// the scheduler's own).
    std::array<u64, static_cast<size_t>(PassEnd::Count)> pass_ends{};
    std::array<u64, 3> passes_begun{};
    std::chrono::steady_clock::time_point heartbeat = std::chrono::steady_clock::now();
    std::atomic<bool> completed_logged{false};
    std::atomic<u32> errors_logged{0};
    /// EDEN_NX2_METAL_RECORDER: the encoder thread's stream. Last: destroyed (drained, joined)
    /// before the rest.
    std::unique_ptr<Rec::Recorder> recorder;
};

Scheduler::Scheduler(const Device& device) : impl{std::make_unique<Impl>(device)} {}

Scheduler::~Scheduler() {
    Finish();
}

u64 Scheduler::CurrentTick() const noexcept {
    return impl->current_tick;
}

bool Scheduler::IsFree(u64 tick) const noexcept {
    if (impl->completed_tick.load(std::memory_order_acquire) >= tick) {
        return true;
    }
    return impl->CompletedTick() >= tick;
}

void Scheduler::Wait(u64 tick, CommitReason reason) {
    if (tick >= impl->current_tick) {
        Flush(reason);
    }
    if (impl->recorder != nullptr) {
        // The tick's commands were published by its Flush; the encoder thread commits them and
        // the completion handler moves completed_tick (only after the real commit).
        if (!IsFree(tick)) {
            std::unique_lock lock{impl->completed_mutex};
            impl->completed_cv.wait(lock, [&] { return IsFree(tick); });
        }
        while (!impl->in_flight.empty() && impl->in_flight.front().first <= tick) {
            impl->in_flight.pop_front();
        }
        return;
    }
    while (!impl->in_flight.empty() && impl->in_flight.front().first <= tick) {
        [impl->in_flight.front().second waitUntilCompleted];
        impl->in_flight.pop_front();
    }
}

bool Scheduler::IsSubmitted(u64 tick) const noexcept {
    return impl->submitted_tick.load(std::memory_order_acquire) >= tick;
}

namespace {
Scheduler::SplitHooks split_hooks{};

bool SplitBackRunning() {
    return split_hooks.back_running != nullptr && split_hooks.back_running();
}
bool SplitOnBackThread() {
    return split_hooks.on_back_thread != nullptr && split_hooks.on_back_thread();
}
bool SplitOwnsScheduler() {
    return split_hooks.owns_scheduler == nullptr || split_hooks.owns_scheduler();
}
void SplitRequestIdleHook() {
    if (split_hooks.request_idle_hook != nullptr) {
        split_hooks.request_idle_hook();
    }
}
} // Anonymous namespace

void Scheduler::SetSplitHooks(const SplitHooks& hooks) noexcept {
    split_hooks = hooks;
}

void Scheduler::WaitSubmitted(u64 tick) {
    WaitCompleted(tick);
}

void Scheduler::WaitCompleted(u64 tick) {
    if (IsFree(tick)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    auto next_request = start;
    auto next_warning = start + std::chrono::seconds{2};
    std::unique_lock lock{impl->completed_mutex};
    while (!IsFree(tick) && !impl->waits_abandoned.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const bool submitted = IsSubmitted(tick);
        if (!submitted && now >= next_request && SplitBackRunning()) {
            // Fence work left pending (FenceTick, RequestCommit): the back commits it in its
            // idle hook, between entries when asked, else before it sleeps. Asked again now and
            // then: a request the back drops while quiet finds the work committed already.
            SplitRequestIdleHook();
            next_request = now + std::chrono::milliseconds{5};
        }
        if (now >= next_warning) {
            next_warning = now + std::chrono::seconds{2};
            const char* const back_site =
                VideoCommon::back_wait_site.load(std::memory_order_relaxed);
            LOG_WARNING(Render_Metal,
                        "Metal: fencing thread waited {} s for tick {} ({}; submitted up to {}, "
                        "completed up to {}); GPU back thread blocked in: {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick, submitted ? "committed" : "not committed",
                        impl->submitted_tick.load(std::memory_order_relaxed),
                        impl->completed_tick.load(std::memory_order_relaxed),
                        back_site != nullptr ? back_site : "no tracked wait");
        }
        // Completion handlers and AbandonWaits notify under the mutex: the timeout only paces the
        // requests and warnings above.
        impl->completed_cv.wait_for(lock, submitted ? std::chrono::milliseconds{100}
                                                    : std::chrono::milliseconds{5});
    }
}

void Scheduler::WaitAnyThread(u64 tick, CommitReason reason) {
    if (SplitOwnsScheduler()) {
        Wait(tick, reason);
    } else {
        WaitCompleted(tick);
    }
}

void Scheduler::RequestCommit() {
    if (!impl->flush_pending) {
        impl->flush_pending = true;
        impl->flush_pending_since = std::chrono::steady_clock::now();
    }
}

void Scheduler::AbandonWaits() {
    impl->waits_abandoned.store(true, std::memory_order_release);
    {
        std::scoped_lock lock{impl->completed_mutex};
    }
    impl->completed_cv.notify_all();
}

bool Scheduler::HasPendingWork() const noexcept {
    return impl->command_buffer != nil || impl->pending_clear.has_value() ||
           impl->upload_buffer != nil || impl->kernel_buffer != nil ||
           impl->ahead_buffer != nil || impl->tick_splits != 0;
}

u64 Scheduler::Flush(CommitReason reason) {
    if (!HasPendingWork()) {
        impl->empty_flushes.fetch_add(1, std::memory_order_relaxed);
        // Every earlier tick is done and nothing was recorded for this one: it is done too.
        const u64 empty_tick = impl->current_tick;
        if (EmptySkipEnabled() && impl->CompletedTick() + 1 >= empty_tick) {
            impl->empty_skipped.fetch_add(1, std::memory_order_relaxed);
            ++impl->current_tick;
            impl->submitted_tick.store(empty_tick, std::memory_order_release);
            u64 previous = impl->completed_tick.load(std::memory_order_relaxed);
            while (previous < empty_tick &&
                   !impl->completed_tick.compare_exchange_weak(previous, empty_tick,
                                                               std::memory_order_release)) {
            }
            {
                std::scoped_lock lock{impl->completed_mutex};
            }
            impl->completed_cv.notify_all();
            impl->NextSegment();
            impl->tick_splits = 0;
            impl->flush_pending = false;
            while (!impl->in_flight.empty() && IsFree(impl->in_flight.front().first)) {
                impl->in_flight.pop_front();
            }
            return empty_tick;
        }
    }
    impl->commit_counts[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    impl->FlushClear();
    impl->EndEncoder(PassEnd::Commit);
    const id<MTLCommandBuffer> command_buffer = impl->Commands();
    const u64 tick = impl->current_tick++;
    std::atomic<u64>* const completed = &impl->completed_tick;
    std::atomic<u64>* const gpu_ns = &impl->gpu_ns;
    std::atomic<bool>* const completed_logged = &impl->completed_logged;
    std::atomic<u32>* const errors_logged = &impl->errors_logged;
    std::mutex* const completed_mutex = &impl->completed_mutex;
    std::condition_variable* const completed_cv = &impl->completed_cv;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
        const CFTimeInterval busy = finished.GPUEndTime - finished.GPUStartTime;
        if (busy > 0.0) {
            gpu_ns->fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
        }
        if (!completed_logged->exchange(true)) {
            LOG_INFO(Render_Metal, "Metal: first command buffer completed (tick {}, status {}, "
                                   "GPU {:.2f} ms)",
                     tick, static_cast<int>(finished.status), busy * 1e3);
        }
        ReportFailure(finished, tick, errors_logged);
        u64 previous = completed->load(std::memory_order_relaxed);
        while (previous < tick &&
               !completed->compare_exchange_weak(previous, tick, std::memory_order_release)) {
        }
        {
            std::scoped_lock lock{*completed_mutex};
        }
        completed_cv->notify_all();
    }];
    impl->CommitAhead(tick);
    if (impl->event != nil) {
        [command_buffer encodeSignalEvent:impl->event value:tick];
    }
    [command_buffer commit];
    if (impl->recorder != nullptr) {
        // Committed by the encoder thread after what was recorded before. submitted_tick still
        // moves now: it means "will complete without the owner doing more", which holds.
        impl->recorder->Publish();
    }
    impl->submitted_tick.store(tick, std::memory_order_release);
    ++impl->command_buffers;
    if (impl->committed++ == 0) {
        LOG_INFO(Render_Metal, "Metal: first command buffer committed (tick {})", tick);
    }
    // Until the first guest frame, a line every 5 s: where a boot that never shows one stands.
    if (impl->presents == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - impl->heartbeat >= std::chrono::seconds{5}) {
            impl->heartbeat = now;
            LOG_INFO(Render_Metal,
                     "Metal: no guest frame presented yet; {} command buffers committed, {} "
                     "completed, {} draws, {} dispatches, {} presents without a guest picture",
                     impl->committed, impl->completed_tick.load(std::memory_order_relaxed),
                     impl->draws, impl->dispatches, impl->blank_presents);
        }
    }
    impl->command_buffer = nil;
    impl->NextSegment();
    impl->tick_splits = 0;
    impl->flush_pending = false;
    impl->last_commit = std::chrono::steady_clock::now();
    impl->in_flight.emplace_back(tick, command_buffer);
    while (!impl->in_flight.empty() && IsFree(impl->in_flight.front().first)) {
        impl->in_flight.pop_front();
    }
    return tick;
}

void Scheduler::Finish(CommitReason reason) {
    Wait(Flush(reason), reason);
}

u64 Scheduler::FenceTick() {
    if (FenceIdleEnabled() && !HasPendingWork()) {
        // Everything before the fence is in the ticks committed so far (tick 0 is always free).
        ++impl->idle_fences;
        return impl->current_tick - 1;
    }
    const s64 interval_us = FenceFlushIntervalUs();
    const auto now = std::chrono::steady_clock::now();
    const bool on_back = SplitOnBackThread() || VideoCommon::GpuSplit::OnBackThread();
    const bool explicit_window = FenceFlushExplicit() || FenceBatchAnyThread();
    const bool can_coalesce =
        interval_us > 0 && (on_back || FenceBatchAnyThread()) &&
        now - impl->last_commit < std::chrono::microseconds{interval_us} &&
        (explicit_window ||
         (GpuBehind() && (!impl->flush_pending ||
                          now - impl->flush_pending_since < std::chrono::microseconds{interval_us})));
    if (can_coalesce) {
        // The tick being recorded: committed with whatever comes next, at the latest when the
        // back goes idle.
        if (!impl->flush_pending) {
            impl->flush_pending = true;
            impl->flush_pending_since = now;
        }
        ++impl->deferred_fences;
        return impl->current_tick;
    }
    return Flush(CommitReason::Fence);
}

bool Scheduler::DeferListEnd() {
    const u64 depth = ListEndDeferDepth();
    const bool on_back = SplitOnBackThread() || VideoCommon::GpuSplit::OnBackThread();
    if (depth == 0 || !on_back || impl->in_flight.empty()) {
        return false;
    }
    // Committed ticks the GPU has not finished: while there are enough, a command buffer of its
    // own for this work would only wait behind them.
    const u64 completed = impl->CompletedTick();
    if (completed + depth >= impl->current_tick) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!ListEndDeferExplicit()) {
        const auto window = std::chrono::microseconds{(std::max)(FenceFlushIntervalUs(), s64{300})};
        if (now - impl->last_commit >= window ||
            (impl->flush_pending && now - impl->flush_pending_since >= window)) {
            return false;
        }
    }
    if (!impl->flush_pending) {
        impl->flush_pending = true;
        impl->flush_pending_since = now;
    }
    ++impl->list_ends_deferred;
    return true;
}

bool Scheduler::FlushDeferredFromIdle(bool requested) {
    // Whatever is recorded goes now when pending, requested by a waiter, or under IDLE_COMMIT_ALL.
    if (!impl->flush_pending && !((requested || IdleCommitAllEnabled()) && HasPendingWork())) {
        return false;
    }
    ++impl->idle_commits;
    Flush(CommitReason::Idle);
    return true;
}

void Scheduler::FlushDeferred() {
    if (impl->flush_pending) {
        Flush(CommitReason::FenceDeferred);
    }
}

bool Scheduler::DeferredFlushMayWait() const {
    return impl->flush_pending &&
           std::chrono::steady_clock::now() - impl->flush_pending_since <
               std::chrono::microseconds{FenceFlushIntervalUs()};
}

bool Scheduler::GpuBehind() const noexcept {
    return impl->CompletedTick() + 2 < impl->current_tick;
}

std::array<u64, 2> Scheduler::TakeDeferredFences() noexcept {
    return {std::exchange(impl->deferred_fences, 0), std::exchange(impl->idle_commits, 0)};
}

void Scheduler::CopyBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset,
                           u64 size, bool ahead) {
    if (size == 0 || src == nullptr || dst == nullptr) {
        return;
    }
    const bool goes_ahead =
        ahead && CanUploadAhead() && (impl->render != nil || impl->segment_render_passes != 0);
    if (!goes_ahead) {
        impl->NoteInOrder(src);
        impl->NoteInOrderWrite(dst);
    }
    // EDEN_NX2_METAL_COMPUTE_COPY: on the compute encoder open where the blit would go.
    if (ComputeCopyEnabled() && size <= ComputeCopyMax() && src != dst) {
        Impl& state = *impl;
        if (goes_ahead) {
            // The ahead command buffer's kernels, open (so no upload blit is): the copy lands
            // after the kernels recorded before it, as UploadEncoder's would.
            if (MergeAheadEnabled() && state.upload_blit == nil && state.kernel_compute == nil &&
                ComputeCopyMode() == 2 && !state.copy_kernel_failed && state.library != nil) {
                state.KernelEncoder();
            }
            if (MergeAheadEnabled() && state.kernel_compute != nil && state.upload_blit == nil &&
                state.CopyByCompute(state.kernel_compute, state.kernel_bound,
                                    &state.kernel_pipeline, src, src_offset, dst, dst_offset,
                                    size)) {
                ++state.ahead_copies;
                ++state.compute_copies_ahead;
                return;
            }
        } else if (state.compute != nil && state.render == nil && state.blit == nil &&
                   state.CopyByCompute(state.compute, state.compute_bound,
                                       &state.compute_pipeline, src, src_offset, dst, dst_offset,
                                       size)) {
            ++state.compute_copies_in_order;
            return;
        }
    }
    const id<MTLBlitCommandEncoder> encoder =
        goes_ahead ? impl->UploadEncoder() : impl->BlitEncoder();
    [encoder copyFromBuffer:AsBuffer(src)
               sourceOffset:src_offset
                   toBuffer:AsBuffer(dst)
          destinationOffset:dst_offset
                       size:size];
}

u64 Scheduler::SegmentSerial() const noexcept {
    return impl->segment_serial;
}

bool Scheduler::CanUploadAhead() const noexcept {
    return AheadEnabled() && !impl->ahead_blocked;
}

bool Scheduler::CanUploadTextureAhead(TextureHandle texture) const noexcept {
    if (texture == nullptr || impl->render == nil || !CanUploadAhead() ||
        impl->PendingClearOn(texture)) {
        return false;
    }
    if (!impl->shadows.empty() && impl->shadows.contains(texture)) {
        return false;
    }
    const id<MTLTexture> root = RootOf(AsTexture(texture));
    return root != nil && !impl->segment_touched_textures.contains((__bridge const void*)root);
}

bool Scheduler::UsedInOrder(BufferHandle buffer) const noexcept {
    return std::find(impl->in_order_buffers.begin(), impl->in_order_buffers.end(), buffer) !=
           impl->in_order_buffers.end();
}

bool Scheduler::CanRecordAhead(const RenderTargetSet& targets) const noexcept {
    if (!AheadEnabled() || impl->ahead_blocked) {
        return false;
    }
    if (impl->pending_clear && !CanFoldClearInto(*impl->pending_clear, targets)) {
        return false;
    }
    if (impl->segment_encoders == 0 || (impl->segment_encoders == 1 && IsRenderingTo(targets))) {
        impl->ahead_query_relaxed = false;
        return true;
    }
    // EDEN_NX2_METAL_AHEAD_ANY: ahead after other work, each buffer the kernels bind checked
    // against what the segment's in-order work bound (BeginCompute).
    if (AheadAnyEnabled() && !impl->segment_opaque && !impl->checked_failed) {
        impl->ahead_query_relaxed = true;
        impl->ahead_query_targets = targets;
        return true;
    }
    return false;
}

void Scheduler::BlockAhead() noexcept {
    impl->ahead_blocked = true;
}

bool Scheduler::IsRenderingTo(const RenderTargetSet& targets) const noexcept {
    return impl->render != nil && impl->render_targets == targets;
}

void Scheduler::WantSplitBefore(const RenderTargetSet& targets) {
    if (!AheadEnabled() || !AheadSplitEnabled()) {
        return;
    }
    if (!IsRenderingTo(targets)) {
        impl->force_split_serial = impl->segment_serial;
    }
    if (!impl->Wanted(targets)) {
        impl->wanted[impl->wanted_count++ % impl->wanted.size()] = targets;
    }
}

void Scheduler::SplitBefore(const RenderTargetSet& targets) {
    // A pass that other work keeps ending is not given a command buffer each time.
    constexpr u32 MaxTickSplits = 2;
    if (!AheadEnabled() || !AheadSplitEnabled() || IsRenderingTo(targets) ||
        impl->tick_splits >= MaxTickSplits || !impl->Wanted(targets)) {
        return;
    }
    if (impl->segment_encoders == 0 ||
        (impl->force_split_serial != impl->segment_serial && CanRecordAhead(targets))) {
        return;
    }
    // A clear waiting for this pass stays its load actions, in the new segment.
    if (impl->pending_clear && !CanFoldClearInto(*impl->pending_clear, targets)) {
        impl->FlushClear();
    }
    if (impl->segment_encoders != 0) {
        impl->Split();
    }
}

void Scheduler::FillBuffer(BufferHandle dst, u64 offset, u64 size, u8 value) {
    if (size == 0 || dst == nullptr) {
        return;
    }
    impl->NoteInOrderWrite(dst);
    [impl->BlitEncoder() fillBuffer:AsBuffer(dst) range:NSMakeRange(offset, size) value:value];
}

void Scheduler::CopyBufferToTexture(BufferHandle src, u64 offset, u32 bytes_per_row,
                                    u32 bytes_per_image, TextureHandle dst, u32 level, u32 slice,
                                    Offset3D origin, Extent3D extent, bool ahead) {
    impl->Written(dst);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    const bool goes_ahead = ahead && CanUploadTextureAhead(dst) && !UsedInOrder(src);
    if (!goes_ahead) {
        impl->Touch(dst);
        impl->NoteInOrder(src);
        impl->NoteTextureWrite(dst);
    }
    const id<MTLBlitCommandEncoder> encoder =
        goes_ahead ? impl->UploadEncoder() : impl->BlitEncoder();
    [encoder copyFromBuffer:AsBuffer(src)
               sourceOffset:offset
          sourceBytesPerRow:bytes_per_row
        sourceBytesPerImage:bytes_per_image
                 sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                  toTexture:AsTexture(dst)
           destinationSlice:slice
           destinationLevel:level
          destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)];
}

void Scheduler::CopyTextureToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin,
                                    Extent3D extent, BufferHandle dst, u64 offset,
                                    u32 bytes_per_row, u32 bytes_per_image) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    impl->Touch(src);
    impl->NoteTextureTouched(src);
    impl->NoteInOrderWrite(dst);
    [impl->BlitEncoder() copyFromTexture:AsTexture(src)
                             sourceSlice:slice
                             sourceLevel:level
                            sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                              sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                                toBuffer:AsBuffer(dst)
                       destinationOffset:offset
                  destinationBytesPerRow:bytes_per_row
                destinationBytesPerImage:bytes_per_image];
}

void Scheduler::CopyTexture(TextureHandle src, u32 src_level, u32 src_slice, Offset3D src_origin,
                            Extent3D extent, TextureHandle dst, u32 dst_level, u32 dst_slice,
                            Offset3D dst_origin) {
    impl->SamplesNeeded(src);
    impl->Written(dst);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    impl->Touch(src);
    impl->Touch(dst);
    impl->NoteTextureWrite(dst);
    [impl->BlitEncoder() copyFromTexture:AsTexture(src)
                             sourceSlice:src_slice
                             sourceLevel:src_level
                            sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                              sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                               toTexture:AsTexture(dst)
                        destinationSlice:dst_slice
                        destinationLevel:dst_level
                       destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
}

void Scheduler::BreakBlits() {
    impl->EndEncoder();
}

void Scheduler::Clear(const ClearPass& clear) {
    if (!ClearFoldEnabled()) {
        impl->RecordClear(clear);
        return;
    }
    // The clear waits for the next render pass: its load actions when the pass is on the same
    // attachments, saving the clear a pass of its own that stores what the next one loads.
    ClearPass wanted = clear;
    bool any = false;
    for (size_t index = 0; index < wanted.color.size(); ++index) {
        if (wanted.color[index].texture == nullptr || !wanted.clear_color[index]) {
            wanted.color[index] = {};
            wanted.clear_color[index] = false;
        } else {
            any = true;
        }
    }
    if (wanted.depth.texture == nullptr || !wanted.clear_depth) {
        wanted.depth = {};
        wanted.clear_depth = false;
    } else {
        any = true;
    }
    if (wanted.stencil.texture == nullptr || !wanted.clear_stencil) {
        wanted.stencil = {};
        wanted.clear_stencil = false;
    } else {
        any = true;
    }
    if (!any) {
        return;
    }
    if (impl->TryDrawInPassClear(wanted)) {
        return;
    }
    // EDEN_NX2_METAL_DEPTH_KILL: the open pass's depth or stencil is cleared whole before
    // anything else uses it: the pass does not store it.
    if (impl->render != nil && DepthKillEnabled() && impl->render_targets.layers <= 1) {
        const RenderTargetSet& open = impl->render_targets;
        const auto whole = [&wanted](const Attachment& attachment) {
            const id<MTLTexture> texture = AsTexture(attachment.texture);
            return texture != nil &&
                   wanted.width >= (std::max)(static_cast<u32>(texture.width) >> attachment.level,
                                              1U) &&
                   wanted.height >=
                       (std::max)(static_cast<u32>(texture.height) >> attachment.level, 1U);
        };
        if (wanted.clear_depth && open.has_depth && wanted.depth == open.depth &&
            whole(wanted.depth)) {
            impl->kill_depth = true;
        }
        if (wanted.clear_stencil && open.has_stencil && wanted.stencil == open.depth &&
            whole(wanted.stencil)) {
            impl->kill_stencil = true;
        }
    }
    // The draws and dispatches recorded so far come before it.
    if (impl->render != nil || impl->compute != nil) {
        impl->EndEncoder(PassEnd::Clear);
    }
    if (impl->pending_clear) {
        if (MergeClear(*impl->pending_clear, wanted)) {
            impl->HoldClear(*impl->pending_clear);
            return;
        }
        impl->FlushClear();
    }
    impl->pending_clear = wanted;
    impl->HoldClear(wanted);
}

bool Scheduler::ClearRectangle(const ClearRect& clear) {
    impl->Written(clear.color.texture);
    impl->Written(clear.depth.texture);
    impl->ClearBefore({clear.color.texture, clear.depth.texture});
    const id<MTLTexture> color = clear.color_mask != 0 ? AsTexture(clear.color.texture) : nil;
    const id<MTLTexture> depth =
        clear.clear_depth || clear.clear_stencil ? AsTexture(clear.depth.texture) : nil;
    if (color == nil && depth == nil) {
        return false;
    }
    const MTLPixelFormat color_format = color != nil ? color.pixelFormat : MTLPixelFormatInvalid;
    const MTLPixelFormat depth_format = depth != nil ? depth.pixelFormat : MTLPixelFormatInvalid;
    const bool stencil = depth != nil && (depth_format == MTLPixelFormatStencil8 ||
                                          StencilViewFormat(depth_format) != MTLPixelFormatInvalid);
    const NSUInteger samples = color != nil ? color.sampleCount : depth.sampleCount;
    const id<MTLRenderPipelineState> pipeline =
        impl->ClearPipeline(color_format, clear.color_mask, depth_format, stencil, samples);
    if (pipeline == nil) {
        return false;
    }
    const u32 limit_w = std::clamp(clear.width, 1U, MaxRenderTargetSize);
    const u32 limit_h = std::clamp(clear.height, 1U, MaxRenderTargetSize);
    const u32 x = (std::min)(clear.rect[0], limit_w);
    const u32 y = (std::min)(clear.rect[1], limit_h);
    const u32 w = (std::min)(clear.rect[2], limit_w - x);
    const u32 h = (std::min)(clear.rect[3], limit_h - y);
    if (w == 0 || h == 0) {
        return true;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (color != nil) {
        ApplyAttachment(pass.colorAttachments[0], clear.color);
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (depth != nil && depth_format != MTLPixelFormatStencil8) {
        ApplyAttachment(pass.depthAttachment, clear.depth);
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (stencil) {
        ApplyAttachment(pass.stencilAttachment, clear.depth);
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    pass.renderTargetWidth = limit_w;
    pass.renderTargetHeight = limit_h;
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder setRenderPipelineState:pipeline];
    if (depth != nil) {
        [encoder setDepthStencilState:impl->ClearDepthState(clear.clear_depth,
                                                            stencil && clear.clear_stencil,
                                                            clear.stencil_mask)];
        [encoder setStencilReferenceValue:clear.stencil_value];
    }
    [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y), static_cast<f64>(w),
                                     static_cast<f64>(h), 0.0, 1.0}];
    [encoder setScissorRect:MTLScissorRect{x, y, w, h}];
    const f32 depth_value = std::clamp(clear.depth_value, 0.0f, 1.0f);
    [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
    if (color != nil) {
        // 16 bytes read as float4, uint4 or int4 by the format's class.
        std::array<u32, 4> bits{};
        const FormatClass format_class = ClassOf(static_cast<u32>(color_format));
        for (size_t component = 0; component < bits.size(); ++component) {
            const f64 value = clear.color_value[component];
            if (format_class == FormatClass::Uint) {
                bits[component] = static_cast<u32>(value);
            } else if (format_class == FormatClass::Sint) {
                bits[component] = static_cast<u32>(static_cast<s32>(value));
            } else {
                bits[component] = std::bit_cast<u32>(static_cast<f32>(value));
            }
        }
        [encoder setFragmentBytes:bits.data() length:sizeof(bits) atIndex:0];
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::Blit(const BlitPass& blit) {
    impl->Written(blit.dst.texture);
    impl->SamplesNeeded(blit.src);
    impl->ClearBefore({blit.src, blit.dst.texture});
    const id<MTLTexture> src = AsTexture(blit.src);
    if (src == nil || blit.dst.texture == nullptr) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(blit.dst_format);
    const id<MTLRenderPipelineState> pipeline = impl->BlitPipeline(format, blit.blend);
    if (pipeline == nil) {
        return false;
    }
    const auto [dx0, dy0, dx1, dy1] = blit.dst_rect;
    const auto [sx0, sy0, sx1, sy1] = blit.src_rect;
    const f32 width = static_cast<f32>(src.width);
    const f32 height = static_cast<f32>(src.height);
    // A mirrored destination becomes a mirrored source rectangle over a forward viewport.
    std::array<f32, 4> uv{sx0 / width, sy0 / height, sx1 / width, sy1 / height};
    if (dx1 < dx0) {
        std::swap(uv[0], uv[2]);
    }
    if (dy1 < dy0) {
        std::swap(uv[1], uv[3]);
    }
    const s32 x = (std::min)(dx0, dx1);
    const s32 y = (std::min)(dy0, dy1);
    const s32 w = std::abs(dx1 - dx0);
    const s32 h = std::abs(dy1 - dy0);
    if (w == 0 || h == 0) {
        return true;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const FormatClass format_class = ClassOf(blit.dst_format);
    const bool depth = format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
    if (depth) {
        // Depth only: the fragment writes it; stencil keeps what it held or is cleared to 0.
        ApplyAttachment(pass.depthAttachment, blit.dst);
        pass.depthAttachment.loadAction = blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (format_class == FormatClass::DepthStencil) {
            ApplyAttachment(pass.stencilAttachment, blit.dst);
            pass.stencilAttachment.loadAction =
                blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    } else {
        ApplyAttachment(pass.colorAttachments[0], blit.dst);
        pass.colorAttachments[0].loadAction =
            blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    // A depth-stencil source's stencil, read through a stencil view.
    id<MTLTexture> src_stencil = nil;
    if (format_class == FormatClass::DepthStencil &&
        StencilViewFormat(format) != MTLPixelFormatInvalid) {
        src_stencil = StencilView(src);
        if (src_stencil == nil) {
            return false;
        }
    }
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    [encoder setRenderPipelineState:pipeline];
    if (depth) {
        [encoder setDepthStencilState:impl->DepthWriteAlways()];
        [encoder setFragmentTexture:src_stencil atIndex:1];
    }
    [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                     static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
    const s32 limit_w = static_cast<s32>(blit.dst_width);
    const s32 limit_h = static_cast<s32>(blit.dst_height);
    const s32 sx = std::clamp(x, 0, limit_w);
    const s32 sy = std::clamp(y, 0, limit_h);
    const s32 sw = std::clamp(x + w, 0, limit_w) - sx;
    const s32 sh = std::clamp(y + h, 0, limit_h) - sy;
    if (sw <= 0 || sh <= 0) {
        [encoder endEncoding];
        return true;
    }
    [encoder setScissorRect:MTLScissorRect{static_cast<NSUInteger>(sx),
                                           static_cast<NSUInteger>(sy),
                                           static_cast<NSUInteger>(sw),
                                           static_cast<NSUInteger>(sh)}];
    [encoder setVertexBytes:uv.data() length:sizeof(uv) atIndex:0];
    [encoder setFragmentTexture:src atIndex:0];
    [encoder setFragmentSamplerState:blit.linear ? impl->linear : impl->nearest atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::CopySamples(const SampleCopy& copy, bool expand) {
    impl->SamplesNeeded(copy.src);
    impl->Written(copy.dst.texture);
    impl->ClearBefore({copy.src, copy.dst.texture});
    const id<MTLTexture> src = AsTexture(copy.src);
    const id<MTLTexture> dst = AsTexture(copy.dst.texture);
    if (src == nil || dst == nil || copy.samples == 0) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(copy.dst_format);
    const id<MTLRenderPipelineState> pipeline =
        impl->SamplePipeline(format, dst.sampleCount, expand);
    if (pipeline == nil) {
        return false;
    }
    const u32 width = std::clamp(copy.dst_width, 1U, MaxRenderTargetSize);
    const u32 height = std::clamp(copy.dst_height, 1U, MaxRenderTargetSize);
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const FormatClass format_class = ClassOf(copy.dst_format);
    const bool depth =
        format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
    if (depth) {
        ApplyAttachment(pass.depthAttachment, copy.dst);
        pass.depthAttachment.loadAction = MTLLoadActionDontCare;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (format_class == FormatClass::DepthStencil) {
            // The stencil stays as it was: only depth is copied.
            ApplyAttachment(pass.stencilAttachment, copy.dst);
            pass.stencilAttachment.loadAction = MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    } else {
        ApplyAttachment(pass.colorAttachments[0], copy.dst);
        pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder setRenderPipelineState:pipeline];
    if (depth) {
        // Depth written, stencil kept.
        [encoder setDepthStencilState:impl->ClearDepthState(true, false, 0)];
    }
    [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width), static_cast<f64>(height),
                                     0.0, 1.0}];
    const f32 depth_value = 0.0f;
    [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
    const std::array<u32, 4> params{copy.shift[0], copy.shift[1], copy.samples, copy.src_layer};
    [encoder setFragmentBytes:params.data() length:sizeof(params) atIndex:0];
    [encoder setFragmentTexture:src atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::Convert(const ConvertPass& convert) {
    impl->Written(convert.dst.texture);
    impl->SamplesNeeded(convert.src);
    impl->ClearBefore({convert.src, convert.dst.texture});
    const id<MTLTexture> src = AsTexture(convert.src);
    if (src == nil || convert.dst.texture == nullptr) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(convert.dst_format);
    const id<MTLRenderPipelineState> pipeline = impl->ConvertPipeline(convert.kind, format);
    if (pipeline == nil) {
        return false;
    }
    id<MTLTexture> src_stencil = nil;
    if (convert.kind == ConvertKind::D24s8ToAbgr8 || convert.kind == ConvertKind::S8d24ToAbgr8) {
        src_stencil = StencilView(src);
        if (src_stencil == nil) {
            return false;
        }
    }
    const u32 width = std::clamp(convert.width, 1U, MaxRenderTargetSize);
    const u32 height = std::clamp(convert.height, 1U, MaxRenderTargetSize);
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const FormatClass format_class = ClassOf(convert.dst_format);
    const bool depth =
        format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
    if (depth) {
        ApplyAttachment(pass.depthAttachment, convert.dst);
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (format_class == FormatClass::DepthStencil) {
            ApplyAttachment(pass.stencilAttachment, convert.dst);
            pass.stencilAttachment.loadAction = MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    } else {
        ApplyAttachment(pass.colorAttachments[0], convert.dst);
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder setRenderPipelineState:pipeline];
    if (depth) {
        // Only the D24S8 conversion writes stencil; the others keep it.
        [encoder setDepthStencilState:convert.kind == ConvertKind::Abgr8ToD24s8
                                          ? impl->DepthWriteAlways()
                                          : impl->ClearDepthState(true, false, 0)];
    }
    [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width), static_cast<f64>(height),
                                     0.0, 1.0}];
    const f32 depth_value = 0.0f;
    [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
    [encoder setFragmentTexture:src atIndex:0];
    if (src_stencil != nil) {
        [encoder setFragmentTexture:src_stencil atIndex:1];
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::DepthFromBuffer(BufferHandle src, u64 offset, u32 bytes_per_row,
                                TextureHandle dst_handle, u32 level, u32 slice, Offset3D origin,
                                Extent3D extent, DepthLayout layout) {
    impl->ClearBefore({dst_handle});
    const id<MTLTexture> dst = AsTexture(dst_handle);
    if (src == nullptr || dst == nil || extent.width == 0 || extent.height == 0 ||
        offset % 4 != 0 || bytes_per_row % 4 != 0) {
        return false;
    }
    const FormatClass format_class = ClassOf(static_cast<u32>(dst.pixelFormat));
    const bool stencil = format_class == FormatClass::DepthStencil;
    if (format_class != FormatClass::Depth && !stencil) {
        return false;
    }
    Impl& state = *impl;
    const u64 key = static_cast<u64>(dst.pixelFormat);
    auto it = state.depth_upload_pipelines.find(key);
    if (it == state.depth_upload_pipelines.end()) {
        id<MTLRenderPipelineState> pipeline = nil;
        if (state.library != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Depth upload";
            desc.vertexFunction = [state.library newFunctionWithName:@"clear_vertex"];
            desc.fragmentFunction = [state.library
                newFunctionWithName:stencil ? @"depth_stencil_from_buffer" : @"depth_from_buffer"];
            desc.depthAttachmentPixelFormat = dst.pixelFormat;
            if (stencil) {
                desc.stencilAttachmentPixelFormat = dst.pixelFormat;
            }
            NSError* error = nil;
            pipeline = [state.device.GetDevice() newRenderPipelineStateWithDescriptor:desc
                                                                                error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: depth upload pipeline failed: {}",
                          error.localizedDescription.UTF8String);
            }
        }
        it = state.depth_upload_pipelines.emplace(key, pipeline).first;
    }
    if (it->second == nil) {
        return false;
    }
    const u32 width = (std::max)(static_cast<u32>(dst.width) >> level, 1U);
    const u32 height = (std::max)(static_cast<u32>(dst.height) >> level, 1U);
    const u32 x = (std::min)(origin.x, width);
    const u32 y = (std::min)(origin.y, height);
    const u32 w = (std::min)(extent.width, width - x);
    const u32 h = (std::min)(extent.height, height - y);
    if (w == 0 || h == 0) {
        return true;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    ApplyAttachment(pass.depthAttachment, Attachment{.texture = dst_handle, .level = level,
                                                     .slice = slice});
    pass.depthAttachment.loadAction = MTLLoadActionLoad;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    if (stencil) {
        ApplyAttachment(pass.stencilAttachment, Attachment{.texture = dst_handle, .level = level,
                                                           .slice = slice});
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    state.NoteInOrder(src);
    const id<MTLRenderCommandEncoder> encoder = state.RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder setRenderPipelineState:it->second];
    [encoder setDepthStencilState:state.DepthWriteAlways()];
    [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y), static_cast<f64>(w),
                                     static_cast<f64>(h), 0.0, 1.0}];
    [encoder setScissorRect:MTLScissorRect{x, y, w, h}];
    const f32 depth_value = 0.0f;
    [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
    const std::array<u32, 8> params{x, y, bytes_per_row / 4, static_cast<u32>(layout), w, h,
                                    stencil ? 1U : 0U, 0};
    // The region's first texel is at offset: the origin is subtracted in the shader.
    [encoder setFragmentBuffer:AsBuffer(src) offset:offset atIndex:0];
    [encoder setFragmentBytes:params.data() length:sizeof(params) atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::DepthToBuffer(TextureHandle src_handle, u32 level, u32 slice, Offset3D origin,
                              Extent3D extent, BufferHandle dst, u64 offset, u32 bytes_per_row,
                              DepthLayout layout) {
    impl->ClearBefore({src_handle});
    const id<MTLTexture> src = AsTexture(src_handle);
    if (src == nil || dst == nullptr || extent.width == 0 || extent.height == 0 ||
        offset % 4 != 0 || bytes_per_row % 4 != 0) {
        return false;
    }
    const FormatClass format_class = ClassOf(static_cast<u32>(src.pixelFormat));
    const bool has_stencil = format_class == FormatClass::DepthStencil;
    if (format_class != FormatClass::Depth && !has_stencil) {
        return false;
    }
    Impl& state = *impl;
    if (state.depth_download == nil && state.library != nil) {
        NSError* error = nil;
        state.depth_download = [state.device.GetDevice()
            newComputePipelineStateWithFunction:[state.library newFunctionWithName:@"depth_to_buffer"]
                                          error:&error];
        if (state.depth_download == nil) {
            LOG_ERROR(Render_Metal, "Metal: depth download kernel failed: {}",
                      error.localizedDescription.UTF8String);
        }
    }
    if (state.depth_download == nil) {
        return false;
    }
    if (src.textureType != MTLTextureType2D && src.textureType != MTLTextureType2DArray) {
        return false;
    }
    // A 2D view of the level and slice (the depth, and the stencil through its own view).
    const id<MTLTexture> depth =
        [src newTextureViewWithPixelFormat:src.pixelFormat
                               textureType:MTLTextureType2D
                                    levels:NSMakeRange(level, 1)
                                    slices:NSMakeRange(slice, 1)];
    id<MTLTexture> stencil = nil;
    if (has_stencil) {
        stencil = StencilView(depth);
        if (stencil == nil) {
            return false;
        }
    }
    if (depth == nil) {
        return false;
    }
    state.EndEncoder(PassEnd::Compute);
    state.NoteTextureTouched(src_handle);
    state.NoteInOrderWrite(dst);
    ++state.encoders;
    ++state.segment_encoders;
    const id<MTLComputeCommandEncoder> encoder = [state.Commands() computeCommandEncoder];
    encoder.label = @"Depth download";
    [encoder setComputePipelineState:state.depth_download];
    [encoder setTexture:depth atIndex:0];
    [encoder setTexture:stencil atIndex:1];
    [encoder setBuffer:AsBuffer(dst) offset:offset atIndex:0];
    const std::array<u32, 8> params{origin.x, origin.y, bytes_per_row / 4,
                                    static_cast<u32>(layout), extent.width, extent.height,
                                    has_stencil ? 1U : 0U, 0};
    [encoder setBytes:params.data() length:sizeof(params) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake((extent.width + 7) / 8, (extent.height + 7) / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder endEncoding];
    return true;
}

bool Scheduler::UpdateSparseTiles(TextureHandle texture_handle,
                                  std::span<const SparseTileOp> ops) {
    const id<MTLTexture> texture = AsTexture(texture_handle);
    if (texture == nil || ops.empty()) {
        return texture != nil;
    }
    impl->ClearBefore({texture_handle});
    impl->NoteTextureWrite(texture_handle);
    impl->EndEncoder();
    ++impl->encoders;
    ++impl->segment_encoders;
    const id<MTLResourceStateCommandEncoder> encoder = [impl->Commands() resourceStateCommandEncoder];
    encoder.label = @"Sparse tiles";
    for (const SparseTileOp& op : ops) {
        [encoder updateTextureMapping:texture
                                 mode:op.map ? MTLSparseTextureMappingModeMap
                                             : MTLSparseTextureMappingModeUnmap
                               region:MTLRegionMake2D(op.x, op.y, op.width, op.height)
                             mipLevel:op.level
                                slice:0];
    }
    [encoder endEncoding];
    return true;
}

bool Scheduler::Pack16(u32 mode, bool to_buffer, TextureHandle texture_handle, u32 level, u32 slice,
                         Offset3D origin, Extent3D extent, BufferHandle buffer, u64 offset,
                         u32 bytes_per_row) {
    const id<MTLTexture> texture = AsTexture(texture_handle);
    if (texture == nil || buffer == nullptr || extent.width == 0 || extent.height == 0 ||
        offset % 2 != 0 || bytes_per_row % 2 != 0 ||
        (texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray)) {
        return false;
    }
    impl->ClearBefore({texture_handle});
    Impl& state = *impl;
    id<MTLComputePipelineState> pipeline = to_buffer ? state.pack16_to : state.pack16_from;
    if (pipeline == nil && state.library != nil) {
        NSError* error = nil;
        pipeline = [state.device.GetDevice()
            newComputePipelineStateWithFunction:
                [state.library newFunctionWithName:to_buffer ? @"pack16_to_buffer"
                                                             : @"pack16_from_buffer"]
                                          error:&error];
        if (pipeline == nil) {
            LOG_ERROR(Render_Metal, "Metal: Packed 16-bit kernel failed: {}",
                      error.localizedDescription.UTF8String);
        } else if (to_buffer) {
            state.pack16_to = pipeline;
        } else {
            state.pack16_from = pipeline;
        }
    }
    const id<MTLTexture> view = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                                           textureType:MTLTextureType2D
                                                                levels:NSMakeRange(level, 1)
                                                                slices:NSMakeRange(slice, 1)];
    if (pipeline == nil || view == nil) {
        return false;
    }
    state.EndEncoder(PassEnd::Compute);
    if (to_buffer) {
        state.NoteTextureTouched(texture_handle);
        state.NoteInOrderWrite(buffer);
    } else {
        state.NoteTextureWrite(texture_handle);
        state.NoteInOrder(buffer);
    }
    ++state.encoders;
    ++state.segment_encoders;
    const id<MTLComputeCommandEncoder> encoder = [state.Commands() computeCommandEncoder];
    encoder.label = @"Packed 16-bit copy";
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:view atIndex:0];
    [encoder setBuffer:AsBuffer(buffer) offset:offset atIndex:0];
    const std::array<u32, 6> params{origin.x, origin.y, extent.width, extent.height,
                                    bytes_per_row / 2, mode};
    [encoder setBytes:params.data() length:sizeof(params) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake((extent.width + 7) / 8, (extent.height + 7) / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder endEncoding];
    return true;
}

bool Scheduler::SwapPacked(BufferHandle buffer, u64 offset, u32 width, u32 rows, u32 images,
                           u32 bytes_per_row, u32 bytes_per_image, u32 shift) {
    if (buffer == nullptr || width == 0 || rows == 0 || images == 0 || offset % 2 != 0 ||
        offset + u64{images} * bytes_per_image > UINT32_MAX) {
        return false;
    }
    impl->ClearBefore({});
    Impl& state = *impl;
    if (state.swap_packed == nil && state.library != nil) {
        NSError* error = nil;
        state.swap_packed = [state.device.GetDevice()
            newComputePipelineStateWithFunction:[state.library newFunctionWithName:@"swap_packed"]
                                          error:&error];
        if (state.swap_packed == nil) {
            LOG_ERROR(Render_Metal, "Metal: packed texel swap kernel failed: {}",
                      error.localizedDescription.UTF8String);
        }
    }
    if (state.swap_packed == nil) {
        return false;
    }
    state.EndEncoder(PassEnd::Compute);
    state.NoteInOrderWrite(buffer);
    ++state.encoders;
    ++state.segment_encoders;
    const id<MTLComputeCommandEncoder> encoder = [state.Commands() computeCommandEncoder];
    encoder.label = @"Packed texel swap";
    [encoder setComputePipelineState:state.swap_packed];
    [encoder setBuffer:AsBuffer(buffer) offset:0 atIndex:0];
    const std::array<u32, 6> params{static_cast<u32>(offset), width, rows, bytes_per_row,
                                    bytes_per_image, shift};
    [encoder setBytes:params.data() length:sizeof(params) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake((width + 31) / 32, (rows + 7) / 8, images)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    return true;
}

bool Scheduler::ScaleNearest(const ScalePass& scale) {
    impl->SamplesNeeded(scale.src);
    impl->Written(scale.dst.texture);
    impl->ClearBefore({scale.src, scale.dst.texture});
    const id<MTLTexture> src = AsTexture(scale.src);
    const id<MTLTexture> dst = AsTexture(scale.dst.texture);
    if (src == nil || dst == nil || scale.src_width == 0 || scale.src_height == 0) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(scale.dst_format);
    const id<MTLRenderPipelineState> pipeline = impl->ScalePipeline(format, dst.sampleCount);
    if (pipeline == nil) {
        return false;
    }
    const u32 width = std::clamp(scale.dst_width, 1U, MaxRenderTargetSize);
    const u32 height = std::clamp(scale.dst_height, 1U, MaxRenderTargetSize);
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const FormatClass format_class = ClassOf(scale.dst_format);
    MTLRenderPassAttachmentDescriptor* attachment = nil;
    if (format_class == FormatClass::Depth) {
        attachment = pass.depthAttachment;
    } else if (format_class == FormatClass::Stencil) {
        attachment = pass.stencilAttachment;
    } else {
        attachment = pass.colorAttachments[0];
    }
    ApplyAttachment(attachment, scale.dst);
    attachment.loadAction = MTLLoadActionDontCare;
    attachment.storeAction = MTLStoreActionStore;
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder setRenderPipelineState:pipeline];
    if (format_class == FormatClass::Depth) {
        [encoder setDepthStencilState:impl->ClearDepthState(true, false, 0)];
    } else if (format_class == FormatClass::Stencil) {
        [encoder setDepthStencilState:impl->DepthWriteAlways()];
    }
    [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width), static_cast<f64>(height),
                                     0.0, 1.0}];
    const f32 depth_value = 0.0f;
    [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
    const std::array<u32, 8> params{scale.src_width, scale.src_height, width, height,
                                    scale.src_layer, 0, 0, 0};
    [encoder setFragmentBytes:params.data() length:sizeof(params) atIndex:0];
    [encoder setFragmentTexture:src atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::Resolve(TextureHandle src_handle, u32 src_slice, const Attachment& dst, u32 width,
                        u32 height) {
    impl->ClearBefore({src_handle, dst.texture});
    impl->SamplesNeeded(src_handle);
    const id<MTLTexture> src = AsTexture(src_handle);
    const id<MTLTexture> target = AsTexture(dst.texture);
    if (src == nil || target == nil || src.sampleCount < 2 || target.sampleCount != 1 ||
        src.pixelFormat != target.pixelFormat) {
        return false;
    }
    const FormatClass format_class = ClassOf(static_cast<u32>(src.pixelFormat));
    if (format_class == FormatClass::Stencil) {
        return false;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    MTLRenderPassAttachmentDescriptor* attachment = nil;
    if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
        MTLRenderPassDepthAttachmentDescriptor* depth = pass.depthAttachment;
        depth.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
        attachment = depth;
    } else {
        attachment = pass.colorAttachments[0];
    }
    attachment.texture = src;
    attachment.slice = src_slice;
    attachment.loadAction = MTLLoadActionLoad;
    attachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
    attachment.resolveTexture = target;
    attachment.resolveLevel = dst.level;
    attachment.resolveSlice = dst.slice;
    attachment.resolveDepthPlane = dst.depth_plane;
    pass.renderTargetWidth = std::clamp(width, 1U, MaxRenderTargetSize);
    pass.renderTargetHeight = std::clamp(height, 1U, MaxRenderTargetSize);
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder endEncoding];
    return true;
}

void Scheduler::BeginRendering(const RenderTargetSet& targets) {
    if (impl->render != nil && impl->render_targets == targets) {
        return;
    }
    // The waiting clear: this pass's load actions (all or a matching subset), or a pass of its
    // own first.
    std::optional<ClearPass> clear;
    if (impl->pending_clear) {
        if (CanFoldClearInto(*impl->pending_clear, targets)) {
            bool has_remainder = false;
            clear = ConsumeClearSubset(*impl->pending_clear, targets, has_remainder);
            if (has_remainder) {
                impl->HoldClear(*impl->pending_clear);
            } else {
                impl->pending_clear.reset();
                impl->pending_clear_textures.clear();
            }
            ++impl->clears_folded;
        } else {
            impl->FlushClear();
        }
    }
    const bool defer_stores = UnwrittenStoresEnabled(impl->device);
    u8 defer_color = 0;
    bool defer_depth = false;
    bool defer_stencil = false;
    bool kill_open_depth = false;
    bool kill_open_stencil = false;
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < targets.color.size(); ++index) {
        if (targets.color[index].texture == nullptr) {
            continue;
        }
        MTLRenderPassColorAttachmentDescriptor* const color = pass.colorAttachments[index];
        ApplyAttachment(color, targets.color[index]);
        color.loadAction = MTLLoadActionLoad;
        color.storeAction = MTLStoreActionStore;
        const bool cleared = clear && clear->clear_color[index];
        // Q27: a multisampled colour target with a resolve shadow resolves in the pass.
        bool resolving = false;
        if (!impl->shadows.empty()) {
            if (const auto shadow = impl->shadows.find(targets.color[index].texture);
                shadow != impl->shadows.end()) {
                Impl::ShadowEntry& entry = shadow->second;
                if (!cleared && entry.dropped) {
                    impl->RestoreSamples(shadow->first, entry);
                }
                if (targets.layers == 1) {
                    resolving = true;
                    color.resolveTexture = AsTexture(entry.shadow);
                    color.resolveLevel = 0;
                    color.resolveSlice = targets.color[index].slice;
                    const bool drop = MsaaDiscardEnabled() && cleared;
                    color.storeAction = drop ? MTLStoreActionMultisampleResolve
                                             : MTLStoreActionStoreAndMultisampleResolve;
                    entry.valid = true;
                    entry.dropped = drop;
                    if (drop) {
                        ++impl->samples_dropped;
                    }
                } else {
                    entry.valid = false;
                }
            }
        }
        if (cleared) {
            const auto& value = clear->color_value[index];
            color.loadAction = MTLLoadActionClear;
            color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
        } else if (defer_stores && !resolving) {
            color.storeAction = MTLStoreActionUnknown;
            defer_color |= static_cast<u8>(1U << index);
        }
    }
    if (targets.has_depth) {
        ApplyAttachment(pass.depthAttachment, targets.depth);
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (clear && clear->clear_depth) {
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.clearDepth = clear->depth_value;
            if (defer_stores && DepthKillEnabled()) {
                pass.depthAttachment.storeAction = MTLStoreActionUnknown;
                kill_open_depth = true;
            }
        } else if (defer_stores) {
            pass.depthAttachment.storeAction = MTLStoreActionUnknown;
            defer_depth = true;
        }
    }
    if (targets.has_stencil) {
        ApplyAttachment(pass.stencilAttachment, targets.depth);
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        if (clear && clear->clear_stencil) {
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.clearStencil = clear->stencil_value;
            if (defer_stores && DepthKillEnabled()) {
                pass.stencilAttachment.storeAction = MTLStoreActionUnknown;
                kill_open_stencil = true;
            }
        } else if (defer_stores) {
            pass.stencilAttachment.storeAction = MTLStoreActionUnknown;
            defer_stencil = true;
        }
    }
    // Draws without attachments (occlusion queries) still need a size, within Metal's limit.
    pass.renderTargetWidth = std::clamp(targets.width, 1U, MaxRenderTargetSize);
    pass.renderTargetHeight = std::clamp(targets.height, 1U, MaxRenderTargetSize);
    const bool targetless = !targets.has_depth && !targets.has_stencil &&
                            std::ranges::all_of(targets.color, [](const Attachment& color) {
                                return color.texture == nullptr;
                            });
    if (targetless && PipelineCheckEnabled()) {
        // Without attachments nothing else tells Metal the pass has one sample per pixel, as
        // the pipelines drawn into it have (MoltenVK sets it for such a pass too).
        pass.defaultRasterSampleCount = 1;
    }
    if (targets.layers > 1 && !targetless) {
        // A program's layer output picks among this many slices from each attachment's own.
        // Metal requires every attachment to have them (array and cube textures only).
        NSUInteger layers = targets.layers;
        const auto fit = [&layers](const Attachment& attachment) {
            const id<MTLTexture> texture = AsTexture(attachment.texture);
            if (texture == nil) {
                return;
            }
            const MTLTextureType type = texture.textureType;
            const bool array = type == MTLTextureType2DArray ||
                               type == MTLTextureType2DMultisampleArray ||
                               type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
            const NSUInteger slices =
                texture.arrayLength * (type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray
                                           ? 6
                                           : 1);
            layers = array && attachment.slice < slices
                         ? (std::min)(layers, slices - attachment.slice)
                         : 1;
        };
        for (const Attachment& color : targets.color) {
            fit(color);
        }
        if (targets.has_depth || targets.has_stencil) {
            fit(targets.depth);
        }
        if (layers > 1) {
            pass.renderTargetArrayLength = layers;
        } else {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_INFO(Render_Metal, "Metal: a pass of {} layers has an attachment of one",
                         targets.layers);
            }
        }
    }
    pass.visibilityResultBuffer = impl->visibility;
    impl->render = impl->RenderEncoder(pass, PassEnd::NewPass);
    impl->deferred_color = defer_color;
    impl->deferred_depth = defer_depth;
    impl->deferred_stencil = defer_stencil;
    impl->cleared_depth_open = kill_open_depth && impl->render != nil;
    impl->cleared_stencil_open = kill_open_stencil && impl->render != nil;
    impl->kill_depth = false;
    impl->kill_stencil = false;
    impl->written_color = 0;
    impl->written_depth = false;
    impl->written_stencil = false;
    impl->pass_state_set = false;
    impl->depth_stencil_cleared = clear && (clear->clear_depth || clear->clear_stencil);
    impl->render_targets = targets;
    impl->visibility_slot = -1;
    impl->vertex_bound = {};
    impl->fragment_bound = {};
    impl->object_bound = {};
    impl->mesh_bound = {};
    impl->render_state = {};
    impl->render_state_valid = false;
    ++impl->render_pass_serial;
}

bool Scheduler::IsRendering() const noexcept {
    return impl->render != nil;
}

bool Scheduler::RenderPassWroteDepthStencil() const noexcept {
    // Clears outside the load actions end the encoder first (Clear, ClearRectangle); every draw
    // sets its state, defaulting to "writes everything", before it is encoded.
    return impl->render == nil || impl->depth_stencil_cleared || impl->written_depth ||
           impl->written_stencil;
}

void Scheduler::EndRendering() {
    if (impl->render != nil) {
        impl->EndEncoder(PassEnd::EndRendering);
    }
}

void Scheduler::SetVisibility(std::optional<u32> slot) {
    const s64 wanted = slot ? static_cast<s64>(*slot) : -1;
    if (impl->render == nil || wanted == impl->visibility_slot) {
        return;
    }
    impl->visibility_slot = wanted;
    if (slot) {
        [impl->render setVisibilityResultMode:MTLVisibilityResultModeCounting
                                       offset:*slot * sizeof(u64)];
    } else {
        [impl->render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }
}

BufferHandle Scheduler::VisibilityBuffer() const noexcept {
    return (__bridge void*)impl->visibility;
}

u64* Scheduler::VisibilityResults() noexcept {
    return static_cast<u64*>(impl->visibility.contents);
}

void Scheduler::SetResolveShadow(TextureHandle msaa, TextureHandle shadow) {
    if (msaa == nullptr) {
        return;
    }
    if (shadow == nullptr) {
        impl->shadows.erase(msaa);
    } else {
        impl->shadows[msaa] = Impl::ShadowEntry{.shadow = shadow};
    }
}

TextureHandle Scheduler::ResolveShadow(TextureHandle msaa) const noexcept {
    const auto it = impl->shadows.find(msaa);
    return it != impl->shadows.end() && it->second.valid ? it->second.shadow : nullptr;
}

u64 Scheduler::SamplesDropped() const noexcept {
    return impl->samples_dropped;
}

u64 Scheduler::SamplesRestored() const noexcept {
    return impl->samples_restored;
}

u64 Scheduler::StoresDropped() const noexcept {
    return impl->stores_dropped;
}

u64 Scheduler::RenderPassSerial() const noexcept {
    return impl->render_pass_serial;
}

void Scheduler::SetDrawState(const DrawState& state) {
    const id<MTLRenderCommandEncoder> encoder = impl->render;
    if (encoder == nil) {
        return;
    }
    impl->pass_state_set = true;
    impl->written_color |= state.color_written;
    impl->written_depth = impl->written_depth || state.depth_written;
    impl->written_stencil = impl->written_stencil || state.stencil_written;
    // What the encoder has from the draw before stays as it is (EDEN_NX2_METAL_STATE_CACHE).
    const bool cached = DrawStateCacheEnabled() && impl->render_state_valid;
    const DrawState& last = impl->render_state;
    const auto changed = [&](bool differs) {
        if (cached && !differs) {
            ++impl->state_skips;
            return false;
        }
        ++impl->state_sets;
        return true;
    };
    if (changed(last.pipeline != state.pipeline)) {
        [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)state.pipeline];
    }
    if (state.depth_stencil != nullptr && changed(last.depth_stencil != state.depth_stencil)) {
        [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)state.depth_stencil];
    }
    const auto to_viewport = [](const Viewport& viewport) {
        return MTLViewport{viewport.x,      viewport.y,     viewport.width,
                           viewport.height, viewport.znear, viewport.zfar};
    };
    // Metal rejects scissors past the attachments.
    const u32 target_width = std::clamp(impl->render_targets.width, 1U, MaxRenderTargetSize);
    const u32 target_height = std::clamp(impl->render_targets.height, 1U, MaxRenderTargetSize);
    const auto to_scissor = [&](const Scissor& scissor) {
        const u32 x = (std::min)(scissor.x, target_width - 1);
        const u32 y = (std::min)(scissor.y, target_height - 1);
        const u32 width = (std::min)(scissor.width, target_width - x);
        const u32 height = (std::min)(scissor.height, target_height - y);
        return MTLScissorRect{x, y, (std::max)(width, 1U), (std::max)(height, 1U)};
    };
    if (state.viewport_count > 1) {
        // A program writes the viewport index: all of them.
        const size_t count = (std::min)(size_t{state.viewport_count}, size_t{16});
        if (changed(last.viewport_count != state.viewport_count ||
                    last.viewport != state.viewport ||
                    last.more_viewports != state.more_viewports)) {
            std::array<MTLViewport, 16> viewports{};
            viewports[0] = to_viewport(state.viewport);
            for (size_t index = 1; index < count; ++index) {
                viewports[index] = to_viewport(state.more_viewports[index - 1]);
            }
            [encoder setViewports:viewports.data() count:count];
        }
        if (changed(last.viewport_count != state.viewport_count ||
                    last.scissor != state.scissor || last.more_scissors != state.more_scissors)) {
            std::array<MTLScissorRect, 16> scissors{};
            scissors[0] = to_scissor(state.scissor);
            for (size_t index = 1; index < count; ++index) {
                scissors[index] = to_scissor(state.more_scissors[index - 1]);
            }
            [encoder setScissorRects:scissors.data() count:count];
        }
    } else {
        if (changed(last.viewport_count != 1 || last.viewport != state.viewport)) {
            [encoder setViewport:to_viewport(state.viewport)];
        }
        if (changed(last.viewport_count != 1 || last.scissor != state.scissor)) {
            [encoder setScissorRect:to_scissor(state.scissor)];
        }
    }
    if (changed(last.fill_lines != state.fill_lines)) {
        [encoder setTriangleFillMode:state.fill_lines ? MTLTriangleFillModeLines
                                                      : MTLTriangleFillModeFill];
    }
    if (changed(last.cull_mode != state.cull_mode)) {
        [encoder setCullMode:static_cast<MTLCullMode>(state.cull_mode)];
    }
    if (changed(last.front_ccw != state.front_ccw)) {
        [encoder setFrontFacingWinding:state.front_ccw ? MTLWindingCounterClockwise
                                                       : MTLWindingClockwise];
    }
    if (changed(last.depth_clamp != state.depth_clamp)) {
        [encoder setDepthClipMode:state.depth_clamp ? MTLDepthClipModeClamp
                                                    : MTLDepthClipModeClip];
    }
    if (changed(last.depth_bias != state.depth_bias || last.slope_scale != state.slope_scale ||
                last.bias_clamp != state.bias_clamp)) {
        [encoder setDepthBias:state.depth_bias slopeScale:state.slope_scale clamp:state.bias_clamp];
    }
    if (impl->depth_bounds && changed(last.depth_bounds_min != state.depth_bounds_min ||
                                      last.depth_bounds_max != state.depth_bounds_max)) {
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
        if (@available(macOS 26.0, iOS 26.0, *)) {
            // A public Metal 4-era call, but the GPU must have the test: its family is checked.
            [encoder setDepthTestMinBound:state.depth_bounds_min maxBound:state.depth_bounds_max];
        }
#endif
    }
    if (changed(last.stencil_front_ref != state.stencil_front_ref ||
                last.stencil_back_ref != state.stencil_back_ref)) {
        [encoder setStencilFrontReferenceValue:state.stencil_front_ref
                            backReferenceValue:state.stencil_back_ref];
    }
    if (changed(last.blend_color != state.blend_color)) {
        [encoder setBlendColorRed:state.blend_color[0]
                            green:state.blend_color[1]
                             blue:state.blend_color[2]
                            alpha:state.blend_color[3]];
    }
    // A draw without a depth-stencil state leaves the encoder's.
    void* const depth_stencil =
        state.depth_stencil != nullptr ? state.depth_stencil : last.depth_stencil;
    impl->render_state = state;
    impl->render_state.depth_stencil = depth_stencil;
    impl->render_state_valid = true;
}

void Scheduler::BindVertexBuffer(u32 slot, BufferHandle buffer, u32 offset) {
    if (SlotFits(slot, BufferSlots, "vertex buffer")) {
        if (!VertexInputCacheEnabled()) {
            // Bound again whatever the encoder has.
            impl->vertex_bound.buffers[slot] = &InlineBytes;
        }
        impl->SetBuffer(ShaderStage::Vertex, slot, buffer, offset);
    }
}

void Scheduler::BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset,
                           bool write) {
    if (SlotFits(slot, BufferSlots, "buffer")) {
        if (stage == ShaderStage::Compute) {
            impl->ComputeBuffer(buffer);
            impl->LogComputeBind(ComputeBind::Kind::Buffer, slot, buffer, offset, true);
        } else if (write && impl->render != nil) {
            impl->NoteSegmentBuffer(buffer, true);
        }
        impl->SetBuffer(stage, slot, buffer, offset);
    }
}

u64 Scheduler::UseBuffer(ShaderStage stage, BufferHandle buffer, u32 offset, bool write) {
    if (buffer == nullptr) {
        return 0;
    }
    if (stage == ShaderStage::Compute) {
        impl->ComputeBuffer(buffer);
        impl->LogComputeBind(ComputeBind::Kind::UseBuffer, 0, buffer, offset, true);
    } else if (write && impl->render != nil) {
        impl->NoteSegmentBuffer(buffer, true);
    }
    const id<MTLBuffer> metal_buffer = AsBuffer(buffer);
    switch (stage) {
    case ShaderStage::Vertex:
        if (impl->render != nil) {
            [impl->render useResource:metal_buffer
                                usage:MTLResourceUsageRead | MTLResourceUsageWrite
                               stages:MTLRenderStageVertex];
        }
        break;
    case ShaderStage::Fragment:
        if (impl->render != nil) {
            [impl->render useResource:metal_buffer
                                usage:MTLResourceUsageRead | MTLResourceUsageWrite
                               stages:MTLRenderStageFragment];
        }
        break;
    case ShaderStage::Compute:
        if (impl->active_compute != nil) {
            [impl->active_compute useResource:metal_buffer
                                        usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        break;
    case ShaderStage::Object:
        // A mesh pipeline's spilled buffers are read through their addresses too: without this
        // they were neither resident nor hazard-tracked for the object and mesh functions.
        if (impl->render != nil) {
            [impl->render useResource:metal_buffer
                                usage:MTLResourceUsageRead | MTLResourceUsageWrite
                               stages:MTLRenderStageObject];
        }
        break;
    case ShaderStage::Mesh:
        if (impl->render != nil) {
            [impl->render useResource:metal_buffer
                                usage:MTLResourceUsageRead | MTLResourceUsageWrite
                               stages:MTLRenderStageMesh];
        }
        break;
    }
    return metal_buffer.gpuAddress + offset;
}

void Scheduler::BindBytes(ShaderStage stage, u32 slot, const void* data, u32 size) {
    if (!SlotFits(slot, BufferSlots, "buffer")) {
        return;
    }
    ++impl->state_sets;
    switch (stage) {
    case ShaderStage::Vertex:
        if (slot < BufferSlots) {
            impl->vertex_bound.buffers[slot] = &InlineBytes;
        }
        [impl->render setVertexBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Fragment:
        if (slot < BufferSlots) {
            impl->fragment_bound.buffers[slot] = &InlineBytes;
        }
        [impl->render setFragmentBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Object:
        if (slot < BufferSlots) {
            impl->object_bound.buffers[slot] = &InlineBytes;
        }
        [impl->render setObjectBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Mesh:
        if (slot < BufferSlots) {
            impl->mesh_bound.buffers[slot] = &InlineBytes;
        }
        [impl->render setMeshBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Compute:
        impl->ResumeKernel();
        impl->LogComputeBytes(slot, data, size);
        if (impl->active_compute == nil) {
            return;
        }
        if (slot < BufferSlots) {
            impl->active_bound->buffers[slot] = &InlineBytes;
        }
        [impl->active_compute setBytes:data length:size atIndex:slot];
        break;
    }
}

void Scheduler::BindTexture(ShaderStage stage, u32 slot, TextureHandle texture, bool on_buffer,
                            bool write) {
    if (!SlotFits(slot, TextureSlots, "texture")) {
        return;
    }
    if (stage == ShaderStage::Compute) {
        impl->ComputeTexture(texture, true);
        impl->LogComputeBind(ComputeBind::Kind::Texture, slot, texture, 0, true);
    } else if (impl->render != nil) {
        impl->NoteTextureTouched(texture);
        if (write) {
            impl->NoteTextureWrite(texture);
        }
        if (on_buffer || !ImageNoteSkipEnabled()) {
            impl->NoteSegmentTexture(texture, write);
        }
    }
    switch (stage) {
    case ShaderStage::Vertex:
        if (impl->render != nil && impl->NoteObject(impl->vertex_bound.textures, slot, texture)) {
            [impl->render setVertexTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    case ShaderStage::Fragment:
        if (impl->render != nil &&
            impl->NoteObject(impl->fragment_bound.textures, slot, texture)) {
            [impl->render setFragmentTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    case ShaderStage::Object:
        if (impl->render != nil && impl->NoteObject(impl->object_bound.textures, slot, texture)) {
            [impl->render setObjectTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    case ShaderStage::Mesh:
        if (impl->render != nil && impl->NoteObject(impl->mesh_bound.textures, slot, texture)) {
            [impl->render setMeshTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    case ShaderStage::Compute:
        if (impl->active_compute != nil &&
            impl->NoteObject(impl->active_bound->textures, slot, texture)) {
            [impl->active_compute setTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    }
}

u64 Scheduler::UseTexture(ShaderStage stage, TextureHandle texture, bool write, bool on_buffer) {
    if (texture == nullptr) {
        return 0;
    }
    if (stage == ShaderStage::Compute) {
        impl->ComputeTexture(texture, write);
        impl->LogComputeBind(ComputeBind::Kind::UseTexture, 0, texture, 0, write);
    } else if (impl->render != nil) {
        impl->NoteTextureTouched(texture);
        if (write) {
            impl->NoteTextureWrite(texture);
        }
        if (on_buffer || !ImageNoteSkipEnabled()) {
            impl->NoteSegmentTexture(texture, write);
        }
    }
    const id<MTLTexture> metal_texture = AsTexture(texture);
    const MTLResourceUsage usage =
        write ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
    switch (stage) {
    case ShaderStage::Vertex:
    case ShaderStage::Fragment:
    case ShaderStage::Object:
    case ShaderStage::Mesh:
        if (impl->render != nil) {
            const MTLRenderStages stages =
                stage == ShaderStage::Vertex     ? MTLRenderStageVertex
                : stage == ShaderStage::Fragment ? MTLRenderStageFragment
                : stage == ShaderStage::Object   ? MTLRenderStageObject
                                                 : MTLRenderStageMesh;
            [impl->render useResource:metal_texture usage:usage stages:stages];
        }
        break;
    case ShaderStage::Compute:
        if (impl->active_compute != nil) {
            [impl->active_compute useResource:metal_texture usage:usage];
        }
        break;
    }
    return metal_texture.gpuResourceID._impl;
}

u64 Scheduler::SamplerResourceId(void* sampler) noexcept {
    if (sampler == nullptr) {
        return 0;
    }
    // Guest samplers are made with supportArgumentBuffers (CreateSampler).
    return AsSampler(sampler).gpuResourceID._impl;
}

void Scheduler::BindSampler(ShaderStage stage, u32 slot, void* sampler) {
    if (!SlotFits(slot, SamplerSlots, "sampler")) {
        return;
    }
    if (stage == ShaderStage::Compute) {
        impl->ResumeKernel();
        impl->LogComputeBind(ComputeBind::Kind::Sampler, slot, sampler, 0, false);
    }
    switch (stage) {
    case ShaderStage::Vertex:
        if (impl->render != nil && impl->NoteObject(impl->vertex_bound.samplers, slot, sampler)) {
            [impl->render setVertexSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    case ShaderStage::Fragment:
        if (impl->render != nil &&
            impl->NoteObject(impl->fragment_bound.samplers, slot, sampler)) {
            [impl->render setFragmentSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    case ShaderStage::Object:
        if (impl->render != nil && impl->NoteObject(impl->object_bound.samplers, slot, sampler)) {
            [impl->render setObjectSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    case ShaderStage::Mesh:
        if (impl->render != nil && impl->NoteObject(impl->mesh_bound.samplers, slot, sampler)) {
            [impl->render setMeshSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    case ShaderStage::Compute:
        if (impl->active_compute != nil &&
            impl->NoteObject(impl->active_bound->samplers, slot, sampler)) {
            [impl->active_compute setSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    }
}

void Scheduler::DrawIndirect(u32 primitive, BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
                  indirectBuffer:AsBuffer(buffer)
            indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::DrawIndexedIndirect(u32 primitive, u32 index_size, BufferHandle index_buffer,
                                    u64 index_offset, BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr || index_buffer == nullptr) {
        return;
    }
    [impl->render drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                              indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                            indexBuffer:AsBuffer(index_buffer)
                      indexBufferOffset:index_offset
                         indirectBuffer:AsBuffer(buffer)
                   indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::BeginCompute(void* pipeline, bool ahead) {
    Impl& state = *impl;
    state.kernel_resume = false;
    state.kernel_checked = false;
    state.compute_clobbered_by_copy = false;
    state.compute_binds.clear();
    state.compute_bind_bytes.clear();
    state.compute_bind_pipeline = pipeline;
    bool goes_ahead =
        ahead && AheadEnabled() && (state.render != nil || state.segment_render_passes != 0);
    bool checked = false;
    if (goes_ahead && AheadAnyEnabled()) {
        // Past what CanRecordAhead allowed before EDEN_NX2_METAL_AHEAD_ANY: checked by its binds.
        const bool plain =
            !state.ahead_query_relaxed &&
            (state.segment_encoders == 0 || (state.segment_encoders == 1 && state.render != nil));
        if (!plain) {
            checked = true;
            goes_ahead = !state.segment_opaque && !state.checked_failed && !state.ahead_blocked;
        }
    }
    const void** bound_pipeline = &state.compute_pipeline;
    if (goes_ahead) {
        state.active_compute = state.KernelEncoder();
        state.active_bound = &state.kernel_bound;
        bound_pipeline = &state.kernel_pipeline;
        state.kernel_checked = checked;
        ++state.ahead_kernels;
        if (checked) {
            ++state.ahead_checked;
        }
    } else {
        // EDEN_NX2_METAL_CLEAR_NARROW: a waiting clear goes first when a texture bind is on it
        // (ComputeTexture).
        if (!ClearNarrowEnabled()) {
            state.FlushClear();
        }
        state.OpenInOrderCompute();
    }
    if (StateCacheEnabled() && *bound_pipeline == pipeline) {
        ++state.state_skips;
        return;
    }
    *bound_pipeline = pipeline;
    ++state.state_sets;
    [state.active_compute setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
}

void Scheduler::Dispatch(std::array<u32, 3> groups, std::array<u32, 3> group_size) {
    impl->ResumeKernel();
    const id<MTLComputeCommandEncoder> encoder = impl->active_compute;
    if (encoder == nil || groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return;
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
            threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1], group_size[2])];
    impl->CountDispatch();
}

void Scheduler::DispatchIndirect(BufferHandle buffer, u64 offset, std::array<u32, 3> group_size) {
    impl->ComputeBuffer(buffer, false);
    const id<MTLComputeCommandEncoder> encoder = impl->active_compute;
    if (encoder == nil || buffer == nullptr) {
        return;
    }
    [encoder dispatchThreadgroupsWithIndirectBuffer:AsBuffer(buffer)
                               indirectBufferOffset:offset
                              threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1],
                                                                group_size[2])];
    impl->CountDispatch();
}

void Scheduler::DispatchStageIn(u32 first, u32 count, u32 first_instance, u32 instances) {
    impl->ResumeKernel();
    const id<MTLComputeCommandEncoder> encoder = impl->active_compute;
    if (encoder == nil || count == 0 || instances == 0) {
        return;
    }
    [encoder setStageInRegion:MTLRegionMake2D(first, first_instance, count, instances)];
    [encoder dispatchThreads:MTLSizeMake(count, instances, 1)
       threadsPerThreadgroup:MTLSizeMake((std::min)(count, 64U), 1, 1)];
    impl->CountDispatch();
}

void Scheduler::SetTessellationFactorBuffer(BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render setTessellationFactorBuffer:AsBuffer(buffer) offset:offset instanceStride:0];
}

void Scheduler::DrawPatches(u32 control_points, u32 patch_count) {
    if (impl->render == nil || patch_count == 0) {
        return;
    }
    [impl->render drawPatches:control_points
                   patchStart:0
                   patchCount:patch_count
             patchIndexBuffer:nil
       patchIndexBufferOffset:0
                instanceCount:1
                 baseInstance:0];
    impl->CountDraw();
}

void Scheduler::Draw(u32 primitive, u32 first_vertex, u32 vertex_count, u32 instance_count,
                     u32 base_instance) {
    if (impl->render == nil || vertex_count == 0 || instance_count == 0) {
        return;
    }
    [impl->render drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
                     vertexStart:first_vertex
                     vertexCount:vertex_count
                   instanceCount:instance_count
                    baseInstance:base_instance];
    impl->CountDraw();
}

void Scheduler::DrawMeshThreadgroups(u32 groups, u32 object_threads, u32 mesh_threads) {
    if (impl->render == nil || groups == 0) {
        return;
    }
    [impl->render drawMeshThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerObjectThreadgroup:MTLSizeMake((std::max)(object_threads, 1U), 1, 1)
             threadsPerMeshThreadgroup:MTLSizeMake((std::max)(mesh_threads, 1U), 1, 1)];
    impl->CountDraw();
}

void Scheduler::DrawMeshThreadgroupsIndirect(BufferHandle buffer, u64 offset, u32 object_threads,
                                             u32 mesh_threads) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render drawMeshThreadgroupsWithIndirectBuffer:AsBuffer(buffer)
                                    indirectBufferOffset:offset
                             threadsPerObjectThreadgroup:MTLSizeMake((std::max)(object_threads, 1U),
                                                                     1, 1)
                               threadsPerMeshThreadgroup:MTLSizeMake((std::max)(mesh_threads, 1U),
                                                                     1, 1)];
    impl->CountDraw();
}

void Scheduler::DrawIndexed(u32 primitive, u32 index_count, u32 index_size,
                            BufferHandle index_buffer, u64 index_offset, u32 instance_count,
                            s32 base_vertex, u32 base_instance) {
    if (impl->render == nil || index_count == 0 || instance_count == 0 || index_buffer == nullptr) {
        return;
    }
    [impl->render drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                             indexCount:index_count
                              indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                            indexBuffer:AsBuffer(index_buffer)
                      indexBufferOffset:index_offset
                          instanceCount:instance_count
                             baseVertex:base_vertex
                           baseInstance:base_instance];
    impl->CountDraw();
}

void* Scheduler::CommandBuffer(bool allow_proxy) {
    impl->FlushClear();
    impl->EndEncoder(PassEnd::CommandBuffer);
    impl->segment_opaque = true;
    // What the caller encodes is not known here: nothing goes ahead of it by its targets.
    ++impl->segment_encoders;
    if (impl->recorder != nullptr && !allow_proxy) {
        // Sync point: the caller (MetalFX) encodes on the real command buffer on this thread,
        // after the encoder thread replayed everything before it.
        const id<MTLCommandBuffer> proxy = impl->Commands();
        impl->recorder->Drain("CommandBuffer()");
        return (__bridge void*)Rec::Recorder::RealOf(proxy);
    }
    return (__bridge void*)impl->Commands();
}

void* Scheduler::SharedEvent() const noexcept {
    return (__bridge void*)impl->event;
}

u64 Scheduler::TakeGpuNanoseconds() noexcept {
    return impl->gpu_ns.exchange(0, std::memory_order_relaxed);
}

void Scheduler::NotePresent(bool guest_picture) {
    if (!guest_picture) {
        if (impl->blank_presents++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first drawable presented, without a guest picture");
        }
        return;
    }
    if (impl->presents++ == 0) {
        LOG_INFO(Render_Metal, "Metal: first guest frame presented ({} command buffers, {} draws, "
                               "{} dispatches before it)",
                 impl->committed, impl->draws, impl->dispatches);
    }
    PassTimesFrameEnd();
}

void Scheduler::PassTimesFrameEnd() {
    static const f64 interval_s = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_PASS_TIMES");
        return value != nullptr ? std::atof(value) : 0.0;
    }();
    if (interval_s <= 0.0 || impl->pass_times_unsupported) {
        return;
    }
    Impl& state = *impl;
    if (state.pass_timing) {
        // The timed frame ends with the command buffer this present commits.
        state.pass_timing = false;
        state.pass_resolve_tick = state.current_tick;
        return;
    }
    if (state.pass_resolve_tick && IsFree(*state.pass_resolve_tick)) {
        state.pass_resolve_tick.reset();
        ResolvePassTimes();
    }
    const auto now = std::chrono::steady_clock::now();
    if (state.pass_resolve_tick || now < state.pass_next) {
        return;
    }
    if (state.pass_samples == nil) {
        const id<MTLDevice> mtl_device = state.device.GetDevice();
        id<MTLCounterSet> timestamps = nil;
        for (id<MTLCounterSet> set in mtl_device.counterSets) {
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestamps = set;
            }
        }
        if (timestamps == nil ||
            ![mtl_device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            state.pass_times_unsupported = true;
            LOG_WARNING(Render_Metal, "Perf Metal passes: this GPU takes no timestamps at render "
                                      "pass boundaries");
            return;
        }
        MTLCounterSampleBufferDescriptor* const descriptor = [MTLCounterSampleBufferDescriptor new];
        descriptor.counterSet = timestamps;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = 2 * 1024;
        NSError* error = nil;
        state.pass_samples = [mtl_device newCounterSampleBufferWithDescriptor:descriptor
                                                                        error:&error];
        if (state.pass_samples == nil) {
            state.pass_times_unsupported = true;
            LOG_WARNING(Render_Metal, "Perf Metal passes: no counter sample buffer: {}",
                        error != nil ? error.localizedDescription.UTF8String : "unknown");
            return;
        }
        LOG_INFO(Render_Metal, "Perf Metal passes: one frame timed every {} s", interval_s);
    }
    state.pass_next = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                std::chrono::duration<f64>(interval_s));
    state.pass_records.clear();
    state.pass_gpu_ns_before = state.gpu_ns.load(std::memory_order_relaxed);
    MTLTimestamp cpu = 0;
    [state.device.GetDevice() sampleTimestamps:&cpu gpuTimestamp:&state.pass_clock_gpu];
    state.pass_clock_steady = std::chrono::steady_clock::now();
    state.pass_timing = true;
}

void Scheduler::ResolvePassTimes() {
    Impl& state = *impl;
    const size_t passes = state.pass_records.size();
    if (passes == 0) {
        return;
    }
    NSData* const data = [state.pass_samples resolveCounterRange:NSMakeRange(0, passes * 2)];
    if (data == nil || data.length < passes * 2 * sizeof(MTLCounterResultTimestamp)) {
        state.pass_times_line = "no timestamps resolved";
        LOG_WARNING(Render_Metal, "Perf Metal passes: no timestamps resolved");
        return;
    }
    const auto* const stamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
    // The GPU clock against the steady clock since the timed frame began.
    MTLTimestamp cpu = 0;
    MTLTimestamp gpu = 0;
    [state.device.GetDevice() sampleTimestamps:&cpu gpuTimestamp:&gpu];
    const f64 steady_ns = static_cast<f64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                             state.pass_clock_steady)
            .count());
    struct Timed {
        size_t index;
        f64 ms;
    };
    std::vector<Timed> timed;
    f64 total_ms = 0.0;
    for (size_t index = 0; index < passes; ++index) {
        const u64 begin = stamps[index * 2].timestamp;
        const u64 end = stamps[index * 2 + 1].timestamp;
        if (begin == MTLCounterErrorValue || end == MTLCounterErrorValue || end < begin) {
            continue;
        }
        timed.push_back({index, static_cast<f64>(end - begin)});
    }
    // The timestamps' unit: CPU nanoseconds per GPU tick, as Vulkan's timestampPeriod.
    f64 ns_per_tick = 1.0;
    if (gpu > state.pass_clock_gpu && steady_ns > 0.0) {
        ns_per_tick = steady_ns / static_cast<f64>(gpu - state.pass_clock_gpu);
    }
    for (Timed& entry : timed) {
        entry.ms *= ns_per_tick / 1e6;
        total_ms += entry.ms;
    }
    std::ranges::sort(timed, [](const Timed& a, const Timed& b) { return a.ms > b.ms; });
    std::string top;
    for (size_t i = 0; i < (std::min)(timed.size(), size_t{8}); ++i) {
        const Impl::PassRecord& record = state.pass_records[timed[i].index];
        top += fmt::format("{}{}x{} format {}{} {} draws {:.3f} ms", i != 0 ? ", " : "",
                           record.width, record.height, record.color_format,
                           record.depth_format != 0 ? fmt::format(" + {}", record.depth_format)
                                                    : std::string{},
                           record.draws, timed[i].ms);
    }
    const f64 frame_ms =
        static_cast<f64>(state.gpu_ns.load(std::memory_order_relaxed) - state.pass_gpu_ns_before) /
        1e6;
    state.pass_times_line =
        fmt::format("{} render passes ({} timed) take {:.2f} ms of GPU time, the frame's command "
                    "buffers {:.2f} ms | passes: {} | since boot: {} kernels ahead after other "
                    "work, {} of them moved back in order",
                    passes, timed.size(), total_ms, frame_ms, top, state.ahead_checked,
                    state.ahead_fallbacks);
    LOG_INFO(Render_Metal, "Perf Metal passes: {}", state.pass_times_line);
}

std::string Scheduler::TakePassTimesLine() {
    return std::exchange(impl->pass_times_line, std::string{});
}

std::string Scheduler::TakeCommitLine(double frames) {
    const double per = frames > 0.0 ? frames : 1.0;
    std::string line;
    for (size_t i = 0; i < impl->commit_counts.size(); ++i) {
        const u64 count = impl->commit_counts[i].exchange(0, std::memory_order_relaxed);
        if (count != 0) {
            line += fmt::format("{}={:.1f}/frame ", CommitReasonNames[i],
                                static_cast<double>(count) / per);
        }
    }
    line += fmt::format("(empty flushes {:.1f}/frame, {:.1f} without a command buffer; list ends "
                        "left pending {:.1f}/frame)",
                        static_cast<double>(impl->empty_flushes.exchange(0)) / per,
                        static_cast<double>(impl->empty_skipped.exchange(0)) / per,
                        static_cast<double>(std::exchange(impl->list_ends_deferred, 0)) / per);
    // EDEN_NX2_METAL_CB_PREMAKE.
    line += fmt::format(" (command buffers made ahead {:.1f}/frame, made when needed {:.1f}/frame)",
                        static_cast<double>(impl->premade.hits.exchange(0)) / per,
                        static_cast<double>(impl->premade.misses.exchange(0)) / per);
    if (impl->recorder != nullptr) {
        line += impl->recorder->TakeLine(frames);
    }
    return line;
}

std::string Scheduler::TakePassEndLine(double frames) {
    const double per = frames > 0.0 ? frames : 1.0;
    std::string line;
    for (size_t i = 0; i < impl->pass_ends.size(); ++i) {
        if (const u64 count = std::exchange(impl->pass_ends[i], 0); count != 0) {
            line += fmt::format("{}={:.1f}/frame ", PassEndNames[i],
                                static_cast<double>(count) / per);
        }
    }
    line += fmt::format("(passes begun per frame: draw={:.1f} clear={:.1f} helper={:.1f})",
                        static_cast<double>(std::exchange(impl->passes_begun[0], 0)) / per,
                        static_cast<double>(std::exchange(impl->passes_begun[1], 0)) / per,
                        static_cast<double>(std::exchange(impl->passes_begun[2], 0)) / per);
    // EDEN_NX2_METAL_COMPUTE_COPY / EDEN_NX2_METAL_SEGMENT_NOTE_CACHE.
    line += fmt::format(" (copies as dispatches per frame: ahead={:.1f} in_order={:.1f}; texture "
                        "notes skipped per frame {:.1f}; in-pass clears {:.1f}/frame)",
                        static_cast<double>(std::exchange(impl->compute_copies_ahead, 0)) / per,
                        static_cast<double>(std::exchange(impl->compute_copies_in_order, 0)) / per,
                        static_cast<double>(std::exchange(impl->segment_note_skips, 0)) / per,
                        static_cast<double>(std::exchange(impl->inpass_clears, 0)) / per);
    return line;
}

std::array<u64, 11> Scheduler::TakeCounts() noexcept {
    return {std::exchange(impl->encoders, 0),      std::exchange(impl->command_buffers, 0),
            std::exchange(impl->blit_commands, 0), std::exchange(impl->render_passes, 0),
            std::exchange(impl->clears_folded, 0), std::exchange(impl->state_sets, 0),
            std::exchange(impl->state_skips, 0),   std::exchange(impl->splits, 0),
            std::exchange(impl->ahead_copies, 0),  std::exchange(impl->ahead_kernels, 0),
            std::exchange(impl->idle_fences, 0)};
}

} // namespace Metal
