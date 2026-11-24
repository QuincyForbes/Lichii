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

#include "common/container/unordered_set.h"
#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/nx2_debug_control.h"
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

fragment float4 blit_float_srgb_to_linear(BlitOut in [[stage_in]],
                                          texture2d<float> src [[texture(0)]],
                                          sampler smp [[sampler(0)]]) {
    float4 color = src.sample(smp, in.uv);
    const float3 c = color.rgb;
    const float3 lo = c / 12.92f;
    const float3 hi = pow((c + 0.055f) / 1.055f, float3(2.4f));
    color.rgb = select(hi, lo, c <= 0.04045f);
    return color;
}

fragment float4 blit_float_square(BlitOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                                  sampler smp [[sampler(0)]]) {
    float4 color = src.sample(smp, in.uv);
    color.rgb = color.rgb * color.rgb;
    return color;
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

struct FillParams {
    ulong dst;
    ulong size;
    uint pattern;
    uint pad;
};

kernel void fill_pattern(uint id [[thread_position_in_grid]],
                         device uchar* dst [[buffer(0)]],
                         constant FillParams& p [[buffer(1)]]) {
    const ulong at = ulong(id) * 16;
    if (at >= p.size) {
        return;
    }
    const ulong d = p.dst + at;
    if (at + 16 <= p.size) {
        if ((d & 15) == 0) {
            *(device uint4*)(dst + d) = uint4(p.pattern);
            return;
        }
        if ((d & 3) == 0) {
            device uint* to = (device uint*)(dst + d);
            to[0] = p.pattern;
            to[1] = p.pattern;
            to[2] = p.pattern;
            to[3] = p.pattern;
            return;
        }
    }
    const uint n = uint(min(ulong(16), p.size - at));
    for (uint i = 0; i < n; ++i) {
        dst[d + i] = uchar((p.pattern >> (((at + i) & 3u) * 8u)) & 0xffu);
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

#define UnretainedBuffer(handle) ((__bridge __unsafe_unretained id<MTLBuffer>)(handle))
#define UnretainedTexture(handle) ((__bridge __unsafe_unretained id<MTLTexture>)(handle))
#define UnretainedSampler(handle) ((__bridge __unsafe_unretained id<MTLSamplerState>)(handle))

void ApplyAttachment(MTLRenderPassAttachmentDescriptor* __unsafe_unretained descriptor,
                     const Attachment& attachment) {
    descriptor.texture = UnretainedTexture(attachment.texture);
    descriptor.level = attachment.level;
    descriptor.slice = attachment.slice;
    descriptor.depthPlane = attachment.depth_plane;
}

const void* RootTextureHandle(const void* handle) noexcept {
    id<MTLTexture> __unsafe_unretained texture = UnretainedTexture(handle);
    while (texture != nil) {
        const id<MTLTexture> __unsafe_unretained parent = texture.parentTexture;
        if (parent == nil) {
            break;
        }
        texture = parent;
    }
    return (__bridge const void*)texture;
}

void UseTextureResources(__unsafe_unretained id<MTLRenderCommandEncoder> encoder,
                         __unsafe_unretained id<MTLTexture> metal_texture, MTLResourceUsage usage,
                         MTLRenderStages stages, bool on_buffer) {
    [encoder useResource:metal_texture usage:usage stages:stages];
    id<MTLBuffer> __unsafe_unretained backing = on_buffer ? metal_texture.buffer : nil;
    for (id<MTLTexture> __unsafe_unretained parent = metal_texture.parentTexture; parent != nil;
         parent = parent.parentTexture) {
        [encoder useResource:parent usage:usage stages:stages];
        if (on_buffer && backing == nil) {
            backing = parent.buffer;
        }
    }
    if (backing != nil) {
        [encoder useResource:backing usage:usage stages:stages];
    }
}

void UseTextureResources(__unsafe_unretained id<MTLComputeCommandEncoder> encoder,
                         __unsafe_unretained id<MTLTexture> metal_texture, MTLResourceUsage usage,
                         bool on_buffer) {
    [encoder useResource:metal_texture usage:usage];
    id<MTLBuffer> __unsafe_unretained backing = on_buffer ? metal_texture.buffer : nil;
    for (id<MTLTexture> __unsafe_unretained parent = metal_texture.parentTexture; parent != nil;
         parent = parent.parentTexture) {
        [encoder useResource:parent usage:usage];
        if (on_buffer && backing == nil) {
            backing = parent.buffer;
        }
    }
    if (backing != nil) {
        [encoder useResource:backing usage:usage];
    }
}

void AppendTextureReadResources(std::vector<const void*>& out,
                                __unsafe_unretained id<MTLTexture> metal_texture, bool on_buffer,
                                const void* known_root = nullptr) {
    if (metal_texture == nil) {
        return;
    }
    const void* const raw = (__bridge const void*)metal_texture;
    out.push_back(raw);
    if (!on_buffer && known_root != nullptr && known_root == raw) {
        return;
    }
    id<MTLBuffer> __unsafe_unretained backing = on_buffer ? metal_texture.buffer : nil;
    for (id<MTLTexture> __unsafe_unretained parent = metal_texture.parentTexture; parent != nil;
         parent = parent.parentTexture) {
        out.push_back((__bridge const void*)parent);
        if (on_buffer && backing == nil) {
            backing = parent.buffer;
        }
    }
    if (backing != nil) {
        out.push_back((__bridge const void*)backing);
    }
}

struct RenderUseEntry {
    const void* resource{};
    MTLResourceUsage usage{};
    MTLRenderStages stages{};
};

void AppendTextureUseResources(std::vector<RenderUseEntry>& out,
                               __unsafe_unretained id<MTLTexture> metal_texture,
                               MTLResourceUsage usage, MTLRenderStages stages, bool on_buffer,
                               const void* known_root = nullptr) {
    if (metal_texture == nil) {
        return;
    }
    const void* const raw = (__bridge const void*)metal_texture;
    out.push_back({raw, usage, stages});
    if (!on_buffer && known_root != nullptr && known_root == raw) {
        return;
    }
    id<MTLBuffer> __unsafe_unretained backing = on_buffer ? metal_texture.buffer : nil;
    for (id<MTLTexture> __unsafe_unretained parent = metal_texture.parentTexture; parent != nil;
         parent = parent.parentTexture) {
        out.push_back({(__bridge const void*)parent, usage, stages});
        if (on_buffer && backing == nil) {
            backing = parent.buffer;
        }
    }
    if (backing != nil) {
        out.push_back({(__bridge const void*)backing, usage, stages});
    }
}

void AppendRenderTextureReadUse(std::vector<RenderUseEntry>& pass_out,
                                std::vector<const void*>& batch_out,
                                __unsafe_unretained id<MTLTexture> metal_texture,
                                MTLRenderStages stages, bool on_buffer,
                                const void* known_root = nullptr) {
    if (metal_texture == nil) {
        return;
    }
    const void* const raw = (__bridge const void*)metal_texture;
    pass_out.push_back({raw, MTLResourceUsageRead, stages});
    batch_out.push_back(raw);
    if (!on_buffer && known_root != nullptr && known_root == raw) {
        return;
    }
    id<MTLBuffer> __unsafe_unretained backing = on_buffer ? metal_texture.buffer : nil;
    for (id<MTLTexture> __unsafe_unretained parent = metal_texture.parentTexture; parent != nil;
         parent = parent.parentTexture) {
        const void* const raw_parent = (__bridge const void*)parent;
        pass_out.push_back({raw_parent, MTLResourceUsageRead, stages});
        batch_out.push_back(raw_parent);
        if (on_buffer && backing == nil) {
            backing = parent.buffer;
        }
    }
    if (backing != nil) {
        const void* const raw_backing = (__bridge const void*)backing;
        pass_out.push_back({raw_backing, MTLResourceUsageRead, stages});
        batch_out.push_back(raw_backing);
    }
}

std::pair<u32, u32> MipExtent(__unsafe_unretained id<MTLTexture> texture, u32 level) {
    return {(std::max)(static_cast<u32>(texture.width) >> level, 1U),
            (std::max)(static_cast<u32>(texture.height) >> level, 1U)};
}

struct SubresourceKey {
    const void* root{};
    u32 level{};
    u32 slice{};
    u32 depth_plane{};

    bool operator==(const SubresourceKey&) const noexcept = default;
};

SubresourceKey SubresourceOf(const Attachment& att) noexcept {
    if (att.texture == nullptr) {
        return {};
    }
    id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(att.texture);
    u32 level = att.level;
    u32 slice = att.slice;
    while (tex != nil) {
        const id<MTLTexture> __unsafe_unretained parent = tex.parentTexture;
        if (parent == nil) {
            break;
        }
        level += static_cast<u32>(tex.parentRelativeLevel);
        slice += static_cast<u32>(tex.parentRelativeSlice);
        tex = parent;
    }
    return {(__bridge const void*)tex, level, slice, att.depth_plane};
}

bool SameAttachmentSubresource(const Attachment& a, const Attachment& b) noexcept {
    if (a.texture == b.texture) {
        return a.texture == nullptr ||
               (a.level == b.level && a.slice == b.slice && a.depth_plane == b.depth_plane);
    }
    if (a.texture == nullptr || b.texture == nullptr) {
        return false;
    }
    const id<MTLTexture> __unsafe_unretained ta = UnretainedTexture(a.texture);
    const id<MTLTexture> __unsafe_unretained tb = UnretainedTexture(b.texture);
    if (ta == nil || tb == nil || ta.pixelFormat != tb.pixelFormat ||
        ta.sampleCount != tb.sampleCount) {
        return false;
    }
    return SubresourceOf(a) == SubresourceOf(b);
}

inline size_t PtrHashIndex(const void* ptr, u32 bits) noexcept {
    const u64 k = static_cast<u64>(reinterpret_cast<uintptr_t>(ptr));
    return static_cast<size_t>((k * 0x9E3779B97F4A7C15ULL) >> (64U - bits));
}

inline u64 PtrBloomBit(const void* ptr) noexcept {
    const u64 k = static_cast<u64>(reinterpret_cast<uintptr_t>(ptr));
    return 1ULL << (((k * 0x9E3779B97F4A7C15ULL) >> 58) & 63U);
}

u32 TextureBlockHeight(MTLPixelFormat format) noexcept {
    switch (format) {
#if !TARGET_OS_IPHONE
    case MTLPixelFormatBC1_RGBA:
    case MTLPixelFormatBC1_RGBA_sRGB:
    case MTLPixelFormatBC2_RGBA:
    case MTLPixelFormatBC2_RGBA_sRGB:
    case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB:
    case MTLPixelFormatBC4_RUnorm:
    case MTLPixelFormatBC4_RSnorm:
    case MTLPixelFormatBC5_RGUnorm:
    case MTLPixelFormatBC5_RGSnorm:
    case MTLPixelFormatBC6H_RGBFloat:
    case MTLPixelFormatBC6H_RGBUfloat:
    case MTLPixelFormatBC7_RGBAUnorm:
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
        return 4;
#endif
    case MTLPixelFormatEAC_R11Unorm:
    case MTLPixelFormatEAC_R11Snorm:
    case MTLPixelFormatEAC_RG11Unorm:
    case MTLPixelFormatEAC_RG11Snorm:
    case MTLPixelFormatEAC_RGBA8:
    case MTLPixelFormatEAC_RGBA8_sRGB:
    case MTLPixelFormatETC2_RGB8:
    case MTLPixelFormatETC2_RGB8_sRGB:
    case MTLPixelFormatETC2_RGB8A1:
    case MTLPixelFormatETC2_RGB8A1_sRGB:
    case MTLPixelFormatASTC_4x4_sRGB:
    case MTLPixelFormatASTC_4x4_LDR:
    case MTLPixelFormatASTC_4x4_HDR:
    case MTLPixelFormatASTC_5x4_sRGB:
    case MTLPixelFormatASTC_5x4_LDR:
    case MTLPixelFormatASTC_5x4_HDR:
        return 4;
    case MTLPixelFormatASTC_5x5_sRGB:
    case MTLPixelFormatASTC_5x5_LDR:
    case MTLPixelFormatASTC_5x5_HDR:
    case MTLPixelFormatASTC_6x5_sRGB:
    case MTLPixelFormatASTC_6x5_LDR:
    case MTLPixelFormatASTC_6x5_HDR:
    case MTLPixelFormatASTC_8x5_sRGB:
    case MTLPixelFormatASTC_8x5_LDR:
    case MTLPixelFormatASTC_8x5_HDR:
    case MTLPixelFormatASTC_10x5_sRGB:
    case MTLPixelFormatASTC_10x5_LDR:
    case MTLPixelFormatASTC_10x5_HDR:
        return 5;
    case MTLPixelFormatASTC_6x6_sRGB:
    case MTLPixelFormatASTC_6x6_LDR:
    case MTLPixelFormatASTC_6x6_HDR:
    case MTLPixelFormatASTC_8x6_sRGB:
    case MTLPixelFormatASTC_8x6_LDR:
    case MTLPixelFormatASTC_8x6_HDR:
    case MTLPixelFormatASTC_10x6_sRGB:
    case MTLPixelFormatASTC_10x6_LDR:
    case MTLPixelFormatASTC_10x6_HDR:
        return 6;
    case MTLPixelFormatASTC_8x8_sRGB:
    case MTLPixelFormatASTC_8x8_LDR:
    case MTLPixelFormatASTC_8x8_HDR:
    case MTLPixelFormatASTC_10x8_sRGB:
    case MTLPixelFormatASTC_10x8_LDR:
    case MTLPixelFormatASTC_10x8_HDR:
        return 8;
    case MTLPixelFormatASTC_10x10_sRGB:
    case MTLPixelFormatASTC_10x10_LDR:
    case MTLPixelFormatASTC_10x10_HDR:
    case MTLPixelFormatASTC_12x10_sRGB:
    case MTLPixelFormatASTC_12x10_LDR:
    case MTLPixelFormatASTC_12x10_HDR:
        return 10;
    case MTLPixelFormatASTC_12x12_sRGB:
    case MTLPixelFormatASTC_12x12_LDR:
    case MTLPixelFormatASTC_12x12_HDR:
        return 12;
    default:
        return 1;
    }
}

u64 TextureCopyBufferBytes(__unsafe_unretained id<MTLTexture> tex, u32 bytes_per_row,
                           u32 bytes_per_image, Extent3D extent) noexcept {
    const u32 bh = tex != nil ? TextureBlockHeight(tex.pixelFormat) : 1U;
    const u32 rows = (std::max)((extent.height + bh - 1U) / bh, 1U);
    const u64 slice_bytes = u64{bytes_per_row} * rows;
    return bytes_per_image != 0 && extent.depth > 1
               ? u64{bytes_per_image} * (extent.depth - 1U) + slice_bytes
               : slice_bytes;
}

std::pair<u32, u32> EffectivePassExtent(const RenderTargetSet& targets) noexcept {
    u32 w = std::clamp(targets.width, 1U, MaxRenderTargetSize);
    u32 h = std::clamp(targets.height, 1U, MaxRenderTargetSize);
    for (const Attachment& c : targets.color) {
        if (c.texture != nullptr) {
            if (const id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(c.texture);
                tex != nil) {
                const auto [mw, mh] = MipExtent(tex, c.level);
                w = (std::min)(w, mw);
                h = (std::min)(h, mh);
            }
        }
    }
    if ((targets.has_depth || targets.has_stencil) && targets.depth.texture != nullptr) {
        if (const id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(targets.depth.texture);
            tex != nil) {
            const auto [mw, mh] = MipExtent(tex, targets.depth.level);
            w = (std::min)(w, mw);
            h = (std::min)(h, mh);
        }
    }
    return {w, h};
}

struct WantedPassKey {
    std::array<SubresourceKey, 8> color{};
    std::array<u32, 8> color_formats{};
    SubresourceKey depth{};
    u32 depth_format{};
    u32 width{};
    u32 height{};
    u32 layers{1};
    bool has_depth{};
    bool has_stencil{};

    bool operator==(const WantedPassKey&) const noexcept = default;
};

WantedPassKey MakeWantedPassKey(const RenderTargetSet& targets) noexcept {
    WantedPassKey out{};
    const auto [eff_w, eff_h] = EffectivePassExtent(targets);
    out.width = eff_w;
    out.height = eff_h;
    out.layers = targets.layers;
    out.has_depth = targets.has_depth;
    out.has_stencil = targets.has_stencil;
    for (size_t i = 0; i < targets.color.size(); ++i) {
        if (targets.color[i].texture != nullptr) {
            out.color[i] = SubresourceOf(targets.color[i]);
            if (const id<MTLTexture> __unsafe_unretained tex =
                    UnretainedTexture(targets.color[i].texture);
                tex != nil) {
                out.color_formats[i] = static_cast<u32>(tex.pixelFormat);
            }
        }
    }
    if ((targets.has_depth || targets.has_stencil) && targets.depth.texture != nullptr) {
        out.depth = SubresourceOf(targets.depth);
        if (const id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(targets.depth.texture);
            tex != nil) {
            out.depth_format = static_cast<u32>(tex.pixelFormat);
        }
    }
    return out;
}

std::array<u32, 4> PackClearColorWords(const std::array<f64, 4>& color, u32 format) {
    std::array<u32, 4> bits{};
    const FormatClass fmt_class = ClassOf(format);
    for (size_t component = 0; component < bits.size(); ++component) {
        const f64 val = color[component];
        if (fmt_class == FormatClass::Uint) {
            bits[component] = static_cast<u32>(val);
        } else if (fmt_class == FormatClass::Sint) {
            bits[component] = static_cast<u32>(static_cast<s32>(val));
        } else {
            bits[component] = std::bit_cast<u32>(static_cast<f32>(val));
        }
    }
    return bits;
}

void SetPipelineTargetFormat(MTLRenderPipelineDescriptor* desc, MTLPixelFormat format,
                             FormatClass format_class) {
    if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
        desc.depthAttachmentPixelFormat = format;
        if (format_class == FormatClass::DepthStencil) {
            desc.stencilAttachmentPixelFormat = format;
        }
    } else {
        desc.colorAttachments[0].pixelFormat = format;
    }
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
// EDEN_NX2_METAL_FENCE_BATCH=0, else defaults to a bounded 450 us window on GPUBack while the
// GPU already has >= 1 committed command buffer in flight or a render pass is open.
bool FenceFlushExplicit() {
    static const bool explicit_window = [] {
        return std::getenv("EDEN_NX2_METAL_FENCE_FLUSH_US") != nullptr ||
               VulkanSwitch("EDEN_NX2_FENCE_FLUSH_US") != nullptr ||
               VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush);
    }();
    return explicit_window;
}

s64 FenceFlushIntervalUs() {
    static const s64 interval_us = [] {
        const char* const batch = std::getenv("EDEN_NX2_METAL_FENCE_BATCH");
        if (batch != nullptr && batch[0] == '0') {
            return s64{0};
        }
        if (const char* const metal_value = std::getenv("EDEN_NX2_METAL_FENCE_FLUSH_US")) {
            return static_cast<s64>(std::atoll(metal_value));
        }
        if (const char* const value = VulkanSwitch("EDEN_NX2_FENCE_FLUSH_US")) {
            return static_cast<s64>(std::atoll(value));
        }
        if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush)) {
            return s64{5000};
        }
        return s64{450};
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

// EDEN_NX2_METAL_LAZY_AHEAD=0: open AheadBuffer() for ahead-eligible copies even when
// `command_buffer` has no active render pass (or already has an open compute encoder that can run
// the copy without a barrier).
bool LazyAheadEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_LAZY_AHEAD", "ahead command buffer opened only while a pass is open");
    return enabled;
}

// EDEN_NX2_METAL_DEFER_DOWNLOADS=0: staging downloads record immediately instead of batching to
// segment end (or before an overlapping GPU write).
bool DeferDownloadsEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_DEFER_DOWNLOADS", "staging downloads batched to segment end");
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
// at least n committed ticks not done yet (default 1, bounded within 1000 us).
bool ListEndDeferExplicit() {
    static const bool explicit_depth = std::getenv("EDEN_NX2_METAL_LIST_END_DEFER") != nullptr;
    return explicit_depth;
}

u64 ListEndDeferDepth() {
    static const u64 depth = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_LIST_END_DEFER");
        const u64 parsed = value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 10))
                                            : u64{1};
        LOG_INFO(Render_Metal,
                 "Metal: list ends left pending while the GPU is {} ticks behind: {} "
                 "(EDEN_NX2_METAL_LIST_END_DEFER={})",
                 parsed, parsed != 0 ? "on" : "off", parsed);
        return parsed;
    }();
    return depth;
}

// EDEN_NX2_METAL_IDLE_COMMIT_ALL=1: the split's idle hook commits everything recorded even while
// earlier command buffers are still running on the GPU.
bool IdleCommitAllEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_IDLE_COMMIT_ALL");
        const bool on = value != nullptr && value[0] == '1';
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
// EDEN_NX2_METAL_RECORDER=0/1: Metal encoding on an encoder thread (metal_recorder.h); the
// recording thread appends commands to a stream it replays and commits in order. Default on for
// macOS (frees GPUBack from Objective-C Metal encoding overhead); off on iOS unless opted in.
bool RecorderEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_RECORDER");
#if defined(__APPLE__) && !TARGET_OS_IPHONE
        const bool on = value == nullptr || value[0] != '0';
#else
        const bool on = value != nullptr && value[0] == '1';
#endif
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

// EDEN_NX2_METAL_CONCURRENT_COMPUTE=0: compute encoders use MTLDispatchTypeSerial (full GPU drain
// between every dispatch). Default on: MTLDispatchTypeConcurrent with explicit scoped barriers from
// ComputeHazardState on RAW/WAW/WAR overlaps.
bool ConcurrentComputeEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_CONCURRENT_COMPUTE", "concurrent compute encoders");
    return enabled;
}

// EDEN_NX2_METAL_COMPUTE_COPY=0: buffer copies always go to a blit encoder, ending an open
// compute encoder. Default =1: ahead copies batch onto UploadEncoder's hardware DMA blit encoder,
// while in-order copies of up to EDEN_NX2_METAL_COMPUTE_COPY_MAX bytes (256 KiB) reuse an
// already-open in-order compute encoder without breaking it; =2 also opens the ahead compute
// encoder when none is open.
u32 ComputeCopyMode() {
    static const u32 mode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_COMPUTE_COPY");
        const u32 read = value == nullptr ? 1U : value[0] == '0' ? 0U : value[0] == '1' ? 1U : 2U;
        LOG_INFO(Render_Metal,
                 "Metal: buffer copies as dispatches on open compute encoders: {} "
                 "(EDEN_NX2_METAL_COMPUTE_COPY={})",
                 read == 0 ? "off" : read == 1 ? "on" : "on, ahead ones opening one", read);
        return read;
    }();
    return mode;
}

// EDEN_NX2_METAL_RANGE_BINDS=0: stage bindings and read-only residency use single-slot calls.
bool RangeBindsEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_RANGE_BINDS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled && StateCacheEnabled();
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

// EDEN_NX2_METAL_MESH_WAVE_BATCH=0: ahead emulated task-mesh draws emit task->dist->mesh per draw
// instead of batching all wave-0 tasks, then all distributions, then all mesh kernels.
bool MeshWaveBatchEnabled() {
    static const bool enabled =
        SwitchOn("EDEN_NX2_METAL_MESH_WAVE_BATCH", "ahead emulated mesh wave batching");
    return enabled;
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
    bool on_buffer;
    bool write;
    u32 slot;
    void* object;
    u32 offset; ///< Bytes: where in compute_bind_bytes for Kind::Bytes, buffer offset otherwise
    u32 size;
    u64 hazard_read_size{~u64{0}};
    u64 hazard_write_size{~u64{0}};
    bool track_hazard{true};
    bool disjoint_write{false};
};

/// What one stage of an encoder has bound; a new encoder has nothing bound, as this starts.
struct StageBindings {
    static constexpr u32 MaxInlineCachedBytes = 2048;
    struct InlineSlot {
        u32 slot{~0U};
        u32 size{};
        std::array<u8, MaxInlineCachedBytes> bytes{};
    };
    std::array<const void*, BufferSlots> buffers{};
    std::array<NSUInteger, BufferSlots> offsets{};
    std::array<u64, BufferSlots> lengths{};
    std::array<const void*, TextureSlots> textures{};
    std::array<const void*, SamplerSlots> samplers{};
    u32 dirty_buffers{};
    u32 dirty_offset_only{};
    std::array<u64, 2> dirty_textures{};
    u16 dirty_samplers{};
    std::array<InlineSlot, 4> inline_slots{};
    u32 inline_cursor{};

    bool HasDirty() const noexcept {
        return (dirty_buffers | dirty_samplers) != 0 ||
               (dirty_textures[0] | dirty_textures[1]) != 0;
    }

    void MarkBufferDirty(u32 slot, bool offset_only) noexcept {
        if (slot < BufferSlots) {
            const u32 bit = 1U << slot;
            const bool already_dirty_new_buf =
                (dirty_buffers & bit) != 0 && (dirty_offset_only & bit) == 0;
            dirty_buffers |= bit;
            if (offset_only && !already_dirty_new_buf) {
                dirty_offset_only |= bit;
            } else {
                dirty_offset_only &= ~bit;
            }
        }
    }

    void ClearBufferDirty(u32 slot) noexcept {
        if (slot < BufferSlots) {
            const u32 bit = 1U << slot;
            dirty_buffers &= ~bit;
            dirty_offset_only &= ~bit;
        }
    }

    void MarkTextureDirty(u32 slot) noexcept {
        if (slot < TextureSlots) {
            dirty_textures[slot >> 6] |= 1ULL << (slot & 63U);
        }
    }

    void ClearTextureDirty(u32 slot) noexcept {
        if (slot < TextureSlots) {
            dirty_textures[slot >> 6] &= ~(1ULL << (slot & 63U));
        }
    }

    void MarkSamplerDirty(u32 slot) noexcept {
        if (slot < SamplerSlots) {
            dirty_samplers |= static_cast<u16>(1U << slot);
        }
    }

    void InvalidateDirty(const void* clobbered_tag) noexcept {
        u32 bm = dirty_buffers;
        while (bm != 0) {
            const u32 s = static_cast<u32>(std::countr_zero(bm));
            bm &= bm - 1U;
            buffers[s] = clobbered_tag;
        }
        for (u32 word_idx = 0; word_idx < 2; ++word_idx) {
            u64 tm = dirty_textures[word_idx];
            while (tm != 0) {
                const u32 s = word_idx * 64U + static_cast<u32>(std::countr_zero(tm));
                tm &= tm - 1ULL;
                if (s < TextureSlots) {
                    textures[s] = clobbered_tag;
                }
            }
        }
        u32 sm = dirty_samplers;
        while (sm != 0) {
            const u32 s = static_cast<u32>(std::countr_zero(sm));
            sm &= sm - 1U;
            samplers[s] = clobbered_tag;
        }
        dirty_buffers = 0;
        dirty_offset_only = 0;
        dirty_textures = {};
        dirty_samplers = 0;
    }

    bool MatchesOrRecordInline(u32 slot, const void* data, u32 size, const void* inline_tag) {
        ClearBufferDirty(slot);
        if (slot >= BufferSlots || size > MaxInlineCachedBytes) {
            if (slot < BufferSlots) {
                buffers[slot] = inline_tag;
                for (InlineSlot& entry : inline_slots) {
                    if (entry.slot == slot) {
                        entry.slot = ~0U;
                    }
                }
            }
            return false;
        }
        if (buffers[slot] == inline_tag) {
            for (InlineSlot& entry : inline_slots) {
                if (entry.slot == slot) {
                    if (entry.size == size &&
                        (size == 0 || std::memcmp(entry.bytes.data(), data, size) == 0)) {
                        return true;
                    }
                    entry.size = size;
                    if (size != 0) {
                        std::memcpy(entry.bytes.data(), data, size);
                    }
                    return false;
                }
            }
        }
        buffers[slot] = inline_tag;
        for (InlineSlot& entry : inline_slots) {
            if (entry.slot == slot || entry.slot == ~0U) {
                entry.slot = slot;
                entry.size = size;
                if (size != 0) {
                    std::memcpy(entry.bytes.data(), data, size);
                }
                return false;
            }
        }
        InlineSlot& victim = inline_slots[inline_cursor++ & (inline_slots.size() - 1)];
        victim.slot = slot;
        victim.size = size;
        if (size != 0) {
            std::memcpy(victim.bytes.data(), data, size);
        }
        return false;
    }
};

/// Tracks byte ranges on buffers and root textures read and written since the last scoped barrier
/// inside one open MTLComputeCommandEncoder (MTLDispatchTypeConcurrent) so independent dispatches
/// and CopyByCompute uploads run concurrently while RAW, WAW, and WAR hazards emit scoped barriers.
struct HazardBufferSpan {
    const void* buffer{};
    u64 begin{};
    u64 end{};

    [[nodiscard]] bool Overlaps(const HazardBufferSpan& o) const noexcept {
        return buffer == o.buffer && begin < o.end && o.begin < end;
    }
};

struct HazardSlotBufferSpan {
    const void* buffer{};
    u64 read_begin{};
    u64 read_end{};
    u64 write_begin{};
    u64 write_end{};
    bool disjoint_write{};
};

struct HazardSlotTextureEntry {
    const void* root{};
    HazardBufferSpan backing{};
    bool write{};
};

struct ComputeHazardState {
    using BufferSpan = HazardBufferSpan;
    using SlotBufferSpan = HazardSlotBufferSpan;
    using SlotTextureEntry = HazardSlotTextureEntry;

    std::vector<BufferSpan> read_buffers;
    std::vector<BufferSpan> written_buffers;
    std::vector<BufferSpan> disjoint_written_buffers;
    std::vector<const void*> read_textures;
    std::vector<const void*> written_textures;
    std::array<SlotBufferSpan, BufferSlots> slot_buffers{};
    u32 slot_buffers_mask{};
    std::array<SlotTextureEntry, TextureSlots> slot_textures{};
    std::array<u64, 2> slot_textures_mask{};
    std::vector<BufferSpan> pending_read_buffers;
    std::vector<BufferSpan> pending_write_buffers;
    std::vector<const void*> pending_read_textures;
    std::vector<const void*> pending_write_textures;
    bool has_dispatch_since_buffer_barrier{};
    u64 barriers_emitted{};

    void Reset() noexcept {
        read_buffers.clear();
        written_buffers.clear();
        disjoint_written_buffers.clear();
        read_textures.clear();
        written_textures.clear();
        has_dispatch_since_buffer_barrier = false;
        ResetPending();
    }

    void ResetPending() noexcept {
        slot_buffers_mask = 0;
        slot_textures_mask = {};
        pending_read_buffers.clear();
        pending_write_buffers.clear();
        pending_read_textures.clear();
        pending_write_textures.clear();
    }

    static u64 SaturatingEnd(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return offset;
        }
        if (size == ~u64{0} || offset > ~u64{0} - size) {
            return ~u64{0};
        }
        return offset + size;
    }

    static bool Contains(const std::vector<const void*>& list, const void* item) noexcept {
        return std::find(list.begin(), list.end(), item) != list.end();
    }

    static void AddUnique(std::vector<const void*>& list, const void* item) {
        if (item != nullptr && !Contains(list, item)) {
            list.push_back(item);
        }
    }

    static bool OverlapsAny(const std::vector<BufferSpan>& list, const BufferSpan& span) noexcept {
        if (span.buffer == nullptr || span.end <= span.begin) {
            return false;
        }
        for (const BufferSpan& existing : list) {
            if (existing.Overlaps(span)) {
                return true;
            }
        }
        return false;
    }

    static void AddSpan(std::vector<BufferSpan>& list, const BufferSpan& span) {
        if (span.buffer == nullptr || span.end <= span.begin) {
            return;
        }
        for (BufferSpan& existing : list) {
            if (existing.buffer != span.buffer) {
                continue;
            }
            if (span.begin >= existing.begin && span.end <= existing.end) {
                return;
            }
            if (span.begin <= existing.end && existing.begin <= span.end) {
                existing.begin = (std::min)(existing.begin, span.begin);
                existing.end = (std::max)(existing.end, span.end);
                return;
            }
        }
        list.push_back(span);
    }

    void ClearBufferHazards() noexcept {
        read_buffers.clear();
        written_buffers.clear();
        disjoint_written_buffers.clear();
        has_dispatch_since_buffer_barrier = false;
    }

    void ClearSlotBuffer(u32 slot) noexcept {
        if (slot < BufferSlots) {
            slot_buffers_mask &= ~(1U << slot);
        }
    }

    void BindSlotBuffer(u32 slot, const void* buffer, bool write, u64 offset = 0,
                        u64 read_size = ~u64{0}, u64 write_size = ~u64{0},
                        bool disjoint_write = false) {
        if (slot >= BufferSlots) {
            NoteBuffer(buffer, write, offset, read_size, write_size);
            return;
        }
        if (buffer == nullptr) {
            slot_buffers_mask &= ~(1U << slot);
            return;
        }
        const u64 r_end = SaturatingEnd(offset, read_size);
        const u64 eff_write = write ? (write_size != 0 ? write_size : read_size) : 0;
        const u64 w_end = write ? SaturatingEnd(offset, eff_write) : offset;
        if (r_end <= offset && w_end <= offset) {
            slot_buffers_mask &= ~(1U << slot);
            return;
        }
        slot_buffers[slot] = {buffer, offset, r_end, offset, w_end, disjoint_write};
        slot_buffers_mask |= (1U << slot);
    }

    void NoteBuffer(const void* buffer, bool write, u64 offset = 0, u64 read_size = ~u64{0},
                    u64 write_size = ~u64{0}) {
        if (buffer == nullptr) {
            return;
        }
        const BufferSpan rspan{buffer, offset, SaturatingEnd(offset, read_size)};
        AddSpan(pending_read_buffers, rspan);
        if (write) {
            const u64 eff_write = write_size != 0 ? write_size : read_size;
            const BufferSpan wspan{buffer, offset, SaturatingEnd(offset, eff_write)};
            AddSpan(pending_write_buffers, wspan);
        }
    }

    [[nodiscard]] bool HasCopyHazard(const void* src, u64 src_offset, u64 size, const void* dst,
                                     u64 dst_offset) const noexcept {
        const BufferSpan src_span{src, src_offset, SaturatingEnd(src_offset, size)};
        const BufferSpan dst_span{dst, dst_offset, SaturatingEnd(dst_offset, size)};
        return OverlapsAny(written_buffers, src_span) ||
               OverlapsAny(disjoint_written_buffers, src_span) ||
               OverlapsAny(written_buffers, dst_span) ||
               OverlapsAny(disjoint_written_buffers, dst_span) ||
               OverlapsAny(read_buffers, dst_span);
    }

    [[nodiscard]] bool HasFillHazard(const void* dst, u64 dst_offset, u64 size) const noexcept {
        const BufferSpan dst_span{dst, dst_offset, SaturatingEnd(dst_offset, size)};
        return OverlapsAny(written_buffers, dst_span) ||
               OverlapsAny(disjoint_written_buffers, dst_span) ||
               OverlapsAny(read_buffers, dst_span);
    }

    static BufferSpan BackingBufferSpan(const void* raw_root) noexcept {
        if (raw_root == nullptr) {
            return {};
        }
        const id<MTLTexture> __unsafe_unretained root_tex = UnretainedTexture(raw_root);
        const id<MTLBuffer> __unsafe_unretained backing = root_tex != nil ? root_tex.buffer : nil;
        if (backing == nil) {
            return {};
        }
        const u64 off = static_cast<u64>(root_tex.bufferOffset);
        const u64 row_bytes = static_cast<u64>(root_tex.bufferBytesPerRow);
        const u64 rows = (std::max)(static_cast<u64>(root_tex.height), u64{1});
        const u64 bytes = row_bytes != 0 ? row_bytes * rows : ~u64{0};
        return {(__bridge const void*)backing, off, SaturatingEnd(off, bytes)};
    }

    void BindSlotTexture(u32 slot, TextureHandle texture, bool on_buffer, bool write,
                         const void* known_root = nullptr) {
        if (slot >= TextureSlots) {
            NoteTexture(texture, on_buffer, write, known_root);
            return;
        }
        const void* const raw_root =
            texture != nullptr ? (known_root != nullptr ? known_root : RootTextureHandle(texture))
                               : nullptr;
        if (raw_root == nullptr) {
            slot_textures_mask[slot >> 6] &= ~(1ULL << (slot & 63U));
            return;
        }
        slot_textures[slot] = {
            raw_root,
            on_buffer ? BackingBufferSpan(raw_root) : BufferSpan{},
            write,
        };
        slot_textures_mask[slot >> 6] |= (1ULL << (slot & 63U));
    }

    void NoteTexture(TextureHandle texture, bool on_buffer, bool write,
                     const void* known_root = nullptr) {
        if (texture == nullptr) {
            return;
        }
        const void* const raw_root =
            known_root != nullptr ? known_root : RootTextureHandle(texture);
        if (raw_root == nullptr) {
            return;
        }
        AddUnique(pending_read_textures, raw_root);
        if (write) {
            AddUnique(pending_write_textures, raw_root);
        }
        if (on_buffer) {
            const BufferSpan backing_span = BackingBufferSpan(raw_root);
            if (backing_span.buffer != nullptr) {
                AddSpan(pending_read_buffers, backing_span);
                if (write) {
                    AddSpan(pending_write_buffers, backing_span);
                }
            }
        }
    }

    void BeforeDispatch(__unsafe_unretained id<MTLComputeCommandEncoder> encoder,
                        bool concurrent = true, BufferSpan extra_read = {}) {
        if (!concurrent) {
            return;
        }
        MTLBarrierScope scope = 0;
        const bool have_buf_history =
            !written_buffers.empty() || !disjoint_written_buffers.empty() || !read_buffers.empty();
        if (have_buf_history) {
            if (extra_read.buffer != nullptr &&
                (OverlapsAny(written_buffers, extra_read) ||
                 OverlapsAny(disjoint_written_buffers, extra_read))) {
                scope |= MTLBarrierScopeBuffers;
            }
            u32 bmask = slot_buffers_mask;
            while (bmask != 0 && (scope & MTLBarrierScopeBuffers) == 0) {
                const u32 s = static_cast<u32>(std::countr_zero(bmask));
                bmask &= bmask - 1U;
                const SlotBufferSpan& entry = slot_buffers[s];
                if (entry.read_end > entry.read_begin) {
                    const BufferSpan rspan{entry.buffer, entry.read_begin, entry.read_end};
                    if (OverlapsAny(written_buffers, rspan) ||
                        OverlapsAny(disjoint_written_buffers, rspan)) {
                        scope |= MTLBarrierScopeBuffers;
                        break;
                    }
                }
                if (entry.write_end > entry.write_begin) {
                    const BufferSpan wspan{entry.buffer, entry.write_begin, entry.write_end};
                    if (OverlapsAny(written_buffers, wspan) ||
                        (!entry.disjoint_write && OverlapsAny(disjoint_written_buffers, wspan)) ||
                        OverlapsAny(read_buffers, wspan)) {
                        scope |= MTLBarrierScopeBuffers;
                        break;
                    }
                }
            }
            if ((scope & MTLBarrierScopeBuffers) == 0) {
                for (const BufferSpan& r : pending_read_buffers) {
                    if (OverlapsAny(written_buffers, r) ||
                        OverlapsAny(disjoint_written_buffers, r)) {
                        scope |= MTLBarrierScopeBuffers;
                        break;
                    }
                }
            }
            if ((scope & MTLBarrierScopeBuffers) == 0) {
                for (const BufferSpan& w : pending_write_buffers) {
                    if (OverlapsAny(written_buffers, w) ||
                        OverlapsAny(disjoint_written_buffers, w) || OverlapsAny(read_buffers, w)) {
                        scope |= MTLBarrierScopeBuffers;
                        break;
                    }
                }
            }
        }
        const bool have_tex_history = !written_textures.empty() || !read_textures.empty();
        if (have_tex_history || have_buf_history) {
            for (u32 widx = 0; widx < 2; ++widx) {
                u64 tmask = slot_textures_mask[widx];
                while (tmask != 0) {
                    const u32 s = widx * 64U + static_cast<u32>(std::countr_zero(tmask));
                    tmask &= tmask - 1ULL;
                    const SlotTextureEntry& entry = slot_textures[s];
                    if ((scope & MTLBarrierScopeTextures) == 0 && have_tex_history &&
                        (Contains(written_textures, entry.root) ||
                         (entry.write && Contains(read_textures, entry.root)))) {
                        scope |= MTLBarrierScopeTextures;
                    }
                    if (entry.backing.buffer != nullptr && have_buf_history &&
                        (OverlapsAny(written_buffers, entry.backing) ||
                         OverlapsAny(disjoint_written_buffers, entry.backing) ||
                         (entry.write && OverlapsAny(read_buffers, entry.backing)))) {
                        scope |= MTLBarrierScopeBuffers | MTLBarrierScopeTextures;
                    }
                }
            }
        }
        if ((scope & MTLBarrierScopeTextures) == 0 && have_tex_history) {
            for (const void* tex : pending_read_textures) {
                if (Contains(written_textures, tex)) {
                    scope |= MTLBarrierScopeTextures;
                    break;
                }
            }
        }
        if ((scope & MTLBarrierScopeTextures) == 0 && have_tex_history) {
            for (const void* tex : pending_write_textures) {
                if (Contains(written_textures, tex) || Contains(read_textures, tex)) {
                    scope |= MTLBarrierScopeTextures;
                    break;
                }
            }
        }
        if (scope != 0 && encoder != nil) {
            [encoder memoryBarrierWithScope:scope];
            ++barriers_emitted;
            if ((scope & MTLBarrierScopeBuffers) != 0) {
                ClearBufferHazards();
            }
            if ((scope & MTLBarrierScopeTextures) != 0) {
                read_textures.clear();
                written_textures.clear();
            }
        }
    }

    void AfterDispatch(bool concurrent = true, BufferSpan extra_read = {}) {
        if (!concurrent) {
            return;
        }
        if (extra_read.buffer != nullptr) {
            AddSpan(read_buffers, extra_read);
        }
        u32 bmask = slot_buffers_mask;
        while (bmask != 0) {
            const u32 s = static_cast<u32>(std::countr_zero(bmask));
            bmask &= bmask - 1U;
            const SlotBufferSpan& entry = slot_buffers[s];
            if (entry.read_end > entry.read_begin) {
                AddSpan(read_buffers, {entry.buffer, entry.read_begin, entry.read_end});
            }
            if (entry.write_end > entry.write_begin) {
                AddSpan(entry.disjoint_write ? disjoint_written_buffers : written_buffers,
                        {entry.buffer, entry.write_begin, entry.write_end});
            }
        }
        for (const BufferSpan& r : pending_read_buffers) {
            AddSpan(read_buffers, r);
        }
        for (const BufferSpan& w : pending_write_buffers) {
            AddSpan(written_buffers, w);
        }
        for (u32 widx = 0; widx < 2; ++widx) {
            u64 tmask = slot_textures_mask[widx];
            while (tmask != 0) {
                const u32 s = widx * 64U + static_cast<u32>(std::countr_zero(tmask));
                tmask &= tmask - 1ULL;
                const SlotTextureEntry& entry = slot_textures[s];
                AddUnique(read_textures, entry.root);
                if (entry.write) {
                    AddUnique(written_textures, entry.root);
                }
                if (entry.backing.buffer != nullptr) {
                    AddSpan(read_buffers, entry.backing);
                    if (entry.write) {
                        AddSpan(written_buffers, entry.backing);
                    }
                }
            }
        }
        for (const void* tex : pending_read_textures) {
            AddUnique(read_textures, tex);
        }
        for (const void* tex : pending_write_textures) {
            AddUnique(written_textures, tex);
        }
        has_dispatch_since_buffer_barrier = true;
    }
};

struct ReusablePassDescriptor {
    MTLRenderPassDescriptor* pass = nil;
    std::array<MTLRenderPassColorAttachmentDescriptor*, 8> colors{};
    MTLRenderPassDepthAttachmentDescriptor* depth = nil;
    MTLRenderPassStencilAttachmentDescriptor* stencil = nil;
    u8 dirty_colors{};
    bool dirty_depth{};
    bool dirty_stencil{};

    ReusablePassDescriptor() {
        pass = [MTLRenderPassDescriptor new];
        for (u32 i = 0; i < 8; ++i) {
            colors[i] = pass.colorAttachments[i];
        }
        depth = pass.depthAttachment;
        stencil = pass.stencilAttachment;
    }

    MTLRenderPassDescriptor* Acquire() noexcept {
        Reset();
        return pass;
    }

    MTLRenderPassColorAttachmentDescriptor* Color(size_t index) noexcept {
        dirty_colors |= static_cast<u8>(1U << index);
        return colors[index];
    }

    MTLRenderPassDepthAttachmentDescriptor* Depth() noexcept {
        dirty_depth = true;
        return depth;
    }

    MTLRenderPassStencilAttachmentDescriptor* Stencil() noexcept {
        dirty_stencil = true;
        return stencil;
    }

    void Reset() noexcept {
        while (dirty_colors != 0) {
            const u32 i = static_cast<u32>(std::countr_zero(dirty_colors));
            dirty_colors &= dirty_colors - 1U;
            MTLRenderPassColorAttachmentDescriptor* __unsafe_unretained c = colors[i];
            c.texture = nil;
            c.level = 0;
            c.slice = 0;
            c.depthPlane = 0;
            c.loadAction = MTLLoadActionDontCare;
            c.storeAction = MTLStoreActionDontCare;
            c.resolveTexture = nil;
            c.resolveLevel = 0;
            c.resolveSlice = 0;
            c.resolveDepthPlane = 0;
        }
        if (dirty_depth) {
            dirty_depth = false;
            MTLRenderPassDepthAttachmentDescriptor* __unsafe_unretained d = depth;
            d.texture = nil;
            d.level = 0;
            d.slice = 0;
            d.depthPlane = 0;
            d.loadAction = MTLLoadActionDontCare;
            d.storeAction = MTLStoreActionDontCare;
            d.clearDepth = 1.0;
            d.resolveTexture = nil;
            d.resolveLevel = 0;
            d.resolveSlice = 0;
            d.resolveDepthPlane = 0;
            d.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
        }
        if (dirty_stencil) {
            dirty_stencil = false;
            MTLRenderPassStencilAttachmentDescriptor* __unsafe_unretained s = stencil;
            s.texture = nil;
            s.level = 0;
            s.slice = 0;
            s.depthPlane = 0;
            s.loadAction = MTLLoadActionDontCare;
            s.storeAction = MTLStoreActionDontCare;
            s.clearStencil = 0;
            s.resolveTexture = nil;
            s.resolveLevel = 0;
            s.resolveSlice = 0;
            s.resolveDepthPlane = 0;
        }
        pass.renderTargetWidth = 0;
        pass.renderTargetHeight = 0;
        pass.renderTargetArrayLength = 0;
        pass.defaultRasterSampleCount = 0;
        pass.visibilityResultBuffer = nil;
    }
};

/// A buffer slot holding bytes (setBytes): its address stands for a buffer no other equals.
const char InlineBytes{};
const char ClobberedSlot{};

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
        const u32 min_w = (std::min)(std::clamp(pending.width, 1U, MaxRenderTargetSize),
                                     std::clamp(clear.width, 1U, MaxRenderTargetSize));
        const u32 min_h = (std::min)(std::clamp(pending.height, 1U, MaxRenderTargetSize),
                                     std::clamp(clear.height, 1U, MaxRenderTargetSize));
        const auto fits_both = [&](const Attachment& att) {
            const id<MTLTexture> tex = AsTexture(att.texture);
            if (tex == nil) {
                return true;
            }
            const auto [mip_w, mip_h] = MipExtent(tex, att.level);
            return mip_w <= min_w && mip_h <= min_h;
        };
        for (size_t i = 0; i < pending.color.size(); ++i) {
            if ((pending.clear_color[i] && !fits_both(pending.color[i])) ||
                (clear.clear_color[i] && !fits_both(clear.color[i]))) {
                return false;
            }
        }
        if ((pending.clear_depth && !fits_both(pending.depth)) ||
            (pending.clear_stencil && !fits_both(pending.stencil)) ||
            (clear.clear_depth && !fits_both(clear.depth)) ||
            (clear.clear_stencil && !fits_both(clear.stencil))) {
            return false;
        }
    }
    NSUInteger samples = 0;
    const auto check_samples = [&](const Attachment& att) {
        const id<MTLTexture> tex = AsTexture(att.texture);
        if (tex == nil) {
            return true;
        }
        const NSUInteger tex_samples = tex.sampleCount;
        if (samples == 0) {
            samples = tex_samples;
            return true;
        }
        return samples == tex_samples;
    };
    for (size_t index = 0; index < pending.color.size(); ++index) {
        if (pending.clear_color[index] && !check_samples(pending.color[index])) {
            return false;
        }
    }
    if ((pending.clear_depth && !check_samples(pending.depth)) ||
        (pending.clear_stencil && !check_samples(pending.stencil))) {
        return false;
    }
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.clear_color[index]) {
            if (!check_samples(clear.color[index])) {
                return false;
            }
            if (pending.clear_color[index] &&
                !SameAttachmentSubresource(pending.color[index], clear.color[index])) {
                return false;
            }
        }
    }
    if ((clear.clear_depth && !check_samples(clear.depth)) ||
        (clear.clear_stencil && !check_samples(clear.stencil))) {
        return false;
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
        if (texture != nullptr && !SameAttachmentSubresource(*texture, *attachment)) {
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
    pending.width = (std::max)(pending.width, clear.width);
    pending.height = (std::max)(pending.height, clear.height);
    const u32 p_min_w = pending.min_width != 0 ? pending.min_width : pending.width;
    const u32 p_min_h = pending.min_height != 0 ? pending.min_height : pending.height;
    const u32 c_min_w = clear.min_width != 0 ? clear.min_width : clear.width;
    const u32 c_min_h = clear.min_height != 0 ? clear.min_height : clear.height;
    pending.min_width = (std::max)(p_min_w, c_min_w);
    pending.min_height = (std::max)(p_min_h, c_min_h);
    return true;
}

u32 EffectiveTargetLayers(const RenderTargetSet& targets) noexcept {
    if (targets.layers <= 1) {
        return 1;
    }
    const bool targetless = !targets.has_depth && !targets.has_stencil &&
                            std::ranges::all_of(targets.color, [](const Attachment& color) {
                                return color.texture == nullptr;
                            });
    if (targetless) {
        return 1;
    }
    NSUInteger layers = targets.layers;
    const auto fit = [&layers](const Attachment& attachment) {
        const id<MTLTexture> __unsafe_unretained texture = UnretainedTexture(attachment.texture);
        if (texture == nil) {
            return;
        }
        const MTLTextureType type = texture.textureType;
        const bool array = type == MTLTextureType2DArray ||
                           type == MTLTextureType2DMultisampleArray ||
                           type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
        const NSUInteger slices =
            texture.arrayLength *
            (type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray ? 6 : 1);
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
    return static_cast<u32>((std::max)(layers, NSUInteger{1}));
}

bool ClearExtentFitsAttachment(const ClearPass& clear, const Attachment& att, u32 target_w,
                               u32 target_h) noexcept {
    const u32 clear_w = std::clamp(clear.width, 1U, MaxRenderTargetSize);
    const u32 clear_h = std::clamp(clear.height, 1U, MaxRenderTargetSize);
    const u32 clear_min_w =
        std::clamp(clear.min_width != 0 ? clear.min_width : clear.width, 1U, clear_w);
    const u32 clear_min_h =
        std::clamp(clear.min_height != 0 ? clear.min_height : clear.height, 1U, clear_h);
    if (target_w >= clear_min_w && target_w <= clear_w && target_h >= clear_min_h &&
        target_h <= clear_h) {
        return true;
    }
    const id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(att.texture);
    if (tex == nil) {
        return false;
    }
    const auto [mip_w, mip_h] = MipExtent(tex, att.level);
    const u32 eff_tw = (std::min)(target_w, mip_w);
    const u32 eff_th = (std::min)(target_h, mip_h);
    return eff_tw >= (std::min)(clear_min_w, mip_w) && eff_tw <= (std::min)(clear_w, mip_w) &&
           eff_th >= (std::min)(clear_min_h, mip_h) && eff_th <= (std::min)(clear_h, mip_h);
}

/// Whether a render pass on `targets` can fold all or a non-conflicting non-empty subset of
/// `clear` into its load actions, leaving any unconsumed attachments in `pending_clears`.
bool CanFoldClearInto(const ClearPass& clear, const RenderTargetSet& targets) {
    const u32 eff_layers = EffectiveTargetLayers(targets);
    if (eff_layers > 1 || clear.width == 0 || clear.height == 0 || targets.width == 0 ||
        targets.height == 0) {
        return false;
    }
    const auto [target_w, target_h] = EffectivePassExtent(targets);
    const auto same_effective_extent = [&](const Attachment& att) {
        return ClearExtentFitsAttachment(clear, att, target_w, target_h);
    };
    const auto targets_use_sub = [&](const Attachment& clear_att) {
        if (clear_att.texture == nullptr) {
            return false;
        }
        const SubresourceKey csub = SubresourceOf(clear_att);
        for (const Attachment& att : targets.color) {
            if (att.texture != nullptr) {
                const SubresourceKey tsub = SubresourceOf(att);
                if (tsub.root == csub.root && tsub.level == csub.level &&
                    tsub.depth_plane == csub.depth_plane && csub.slice >= tsub.slice &&
                    csub.slice < tsub.slice + eff_layers) {
                    return true;
                }
            }
        }
        if ((targets.has_depth || targets.has_stencil) && targets.depth.texture != nullptr) {
            const SubresourceKey tsub = SubresourceOf(targets.depth);
            if (tsub.root == csub.root && tsub.level == csub.level &&
                tsub.depth_plane == csub.depth_plane && csub.slice >= tsub.slice &&
                csub.slice < tsub.slice + eff_layers) {
                return true;
            }
        }
        return false;
    };
    bool any_match = false;
    u8 matched_target_slots = 0;
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (!clear.clear_color[index] || clear.color[index].texture == nullptr) {
            continue;
        }
        size_t matched_rt = targets.color.size();
        for (size_t rt = 0; rt < targets.color.size(); ++rt) {
            if (((matched_target_slots >> rt) & 1U) == 0 &&
                SameAttachmentSubresource(targets.color[rt], clear.color[index])) {
                matched_rt = rt;
                break;
            }
        }
        if (matched_rt < targets.color.size()) {
            if (!same_effective_extent(clear.color[index])) {
                return false;
            }
            matched_target_slots |= static_cast<u8>(1U << matched_rt);
            any_match = true;
        } else if (targets_use_sub(clear.color[index])) {
            return false;
        }
    }
    if (clear.clear_depth && clear.depth.texture != nullptr) {
        if (targets.has_depth && SameAttachmentSubresource(clear.depth, targets.depth)) {
            if (!same_effective_extent(clear.depth)) {
                return false;
            }
            any_match = true;
        } else if (targets_use_sub(clear.depth)) {
            return false;
        }
    }
    if (clear.clear_stencil && clear.stencil.texture != nullptr) {
        if (targets.has_stencil && SameAttachmentSubresource(clear.stencil, targets.depth)) {
            if (!same_effective_extent(clear.stencil)) {
                return false;
            }
            any_match = true;
        } else if (targets_use_sub(clear.stencil)) {
            return false;
        }
    }
    return any_match;
}

void MergeConsumedClear(ClearPass& dst, const ClearPass& consumed) {
    for (size_t index = 0; index < consumed.color.size(); ++index) {
        if (consumed.clear_color[index]) {
            dst.color[index] = consumed.color[index];
            dst.clear_color[index] = true;
            dst.color_value[index] = consumed.color_value[index];
        }
    }
    if (consumed.clear_depth) {
        dst.depth = consumed.depth;
        dst.clear_depth = true;
        dst.depth_value = consumed.depth_value;
    }
    if (consumed.clear_stencil) {
        dst.stencil = consumed.stencil;
        dst.clear_stencil = true;
        dst.stencil_value = consumed.stencil_value;
    }
    dst.width = (std::max)(dst.width, consumed.width);
    dst.height = (std::max)(dst.height, consumed.height);
}

/// Consumes the attachments of `pending` that match `targets`, keeping any remaining ones in
/// `pending` (`has_remainder` tells whether `pending` still holds anything).
ClearPass ConsumeClearSubset(ClearPass& pending, const RenderTargetSet& targets,
                             bool& has_remainder) {
    ClearPass consumed{};
    consumed.width = pending.width;
    consumed.height = pending.height;
    has_remainder = false;
    u8 matched_target_slots = 0;
    for (size_t index = 0; index < pending.color.size(); ++index) {
        if (!pending.clear_color[index] || pending.color[index].texture == nullptr) {
            pending.color[index] = {};
            pending.clear_color[index] = false;
            continue;
        }
        size_t matched_rt = targets.color.size();
        for (size_t rt = 0; rt < targets.color.size(); ++rt) {
            if (((matched_target_slots >> rt) & 1U) == 0 &&
                SameAttachmentSubresource(targets.color[rt], pending.color[index])) {
                matched_rt = rt;
                break;
            }
        }
        if (matched_rt < targets.color.size()) {
            matched_target_slots |= static_cast<u8>(1U << matched_rt);
            consumed.color[matched_rt] = pending.color[index];
            consumed.clear_color[matched_rt] = true;
            consumed.color_value[matched_rt] = pending.color_value[index];
            pending.color[index] = {};
            pending.clear_color[index] = false;
        } else {
            has_remainder = true;
        }
    }
    if (pending.clear_depth && pending.depth.texture != nullptr) {
        if (targets.has_depth && SameAttachmentSubresource(pending.depth, targets.depth)) {
            consumed.depth = pending.depth;
            consumed.clear_depth = true;
            consumed.depth_value = pending.depth_value;
            pending.depth = {};
            pending.clear_depth = false;
        } else {
            has_remainder = true;
        }
    } else {
        pending.depth = {};
        pending.clear_depth = false;
    }
    if (pending.clear_stencil && pending.stencil.texture != nullptr) {
        if (targets.has_stencil && SameAttachmentSubresource(pending.stencil, targets.depth)) {
            consumed.stencil = pending.stencil;
            consumed.clear_stencil = true;
            consumed.stencil_value = pending.stencil_value;
            pending.stencil = {};
            pending.clear_stencil = false;
        } else {
            has_remainder = true;
        }
    } else {
        pending.stencil = {};
        pending.clear_stencil = false;
    }
    return consumed;
}

struct InPassClearKey {
    std::array<u32, 8> color_formats{};
    u32 depth_format{};
    u32 stencil_format{};
    u32 color_write_masks{};
    u8 samples{};

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
        mix((u64{key.samples} << 32) | key.color_write_masks);
        return static_cast<size_t>(h);
    }
};

} // Anonymous namespace

struct Scheduler::Impl {
    explicit Impl(const Device& device_)
        : device{device_}, queue{device_.GetQueue()},
          state_cache_enabled{StateCacheEnabled()},
          draw_state_cache_enabled{DrawStateCacheEnabled()},
          vertex_input_cache_enabled{VertexInputCacheEnabled()},
          range_binds_enabled{RangeBindsEnabled()},
          pipeline_check_enabled{PipelineCheckEnabled()},
          ahead_enabled{AheadEnabled()},
          ahead_split_enabled{AheadSplitEnabled()},
          ahead_any_enabled{AheadAnyEnabled()},
          merge_ahead_enabled{MergeAheadEnabled()},
          lazy_ahead_enabled{LazyAheadEnabled()},
          defer_downloads_enabled{DeferDownloadsEnabled()},
          clear_fold_enabled{ClearFoldEnabled()},
          clear_narrow_enabled{ClearNarrowEnabled()},
          depth_kill_enabled{DepthKillEnabled()},
          unwritten_stores_enabled{UnwrittenStoresEnabled(device_)},
          msaa_discard_enabled{MsaaDiscardEnabled()},
          inpass_clear_enabled{InPassClearEnabled()},
          static_labels_enabled{StaticLabelsEnabled()},
          image_note_skip_enabled{ImageNoteSkipEnabled()},
          segment_note_cache_enabled{SegmentNoteCacheEnabled()},
          concurrent_compute_enabled{ConcurrentComputeEnabled()},
          mesh_wave_batch_enabled{MeshWaveBatchEnabled()},
          compute_copy_mode{ComputeCopyMode()},
          compute_copy_enabled{ComputeCopyEnabled()},
          compute_copy_max{ComputeCopyMax()} {
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
        pending_clears.reserve(MaxPendingClears);
        for (auto& vec : pending_render_read_use) {
            vec.reserve(64);
        }
        pass_used_resources.reserve(64);
        pass_touched_roots.reserve(32);
        pass_sampled_roots.reserve(32);
        pending_in_order_compute_read_use.reserve(64);
        pending_kernel_compute_read_use.reserve(64);
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
            if (!static_labels_enabled) {
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

    void ClearPendingAheadTracking() noexcept {
        ahead_encoders = 0;
        kernel_binding = false;
        ahead_written_spans.clear();
        ahead_read_spans.clear();
        ahead_touched_textures.clear();
        ahead_written_textures.clear();
    }

    void NoteAheadReadSpan(const void* buffer, u64 offset, u64 size) {
        if (command_buffer == nil && buffer != nullptr && size != 0) {
            ComputeHazardState::AddSpan(
                ahead_read_spans,
                {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
        }
    }

    void NoteAheadWriteSpan(const void* buffer, u64 offset, u64 size) {
        if (command_buffer == nil && buffer != nullptr && size != 0) {
            ComputeHazardState::AddSpan(
                ahead_written_spans,
                {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
        }
    }

    void NoteAheadTexture(const void* raw_root, bool write) {
        if (command_buffer == nil && raw_root != nullptr) {
            ComputeHazardState::AddUnique(ahead_touched_textures, raw_root);
            if (write) {
                ComputeHazardState::AddUnique(ahead_written_textures, raw_root);
            }
        }
    }

    /// Adopts pre-pass `ahead_buffer` work directly as `command_buffer` when the segment's first
    /// in-order encoder opens, promoting its accessed spans so later mid-pass ahead operations
    /// stay ordered after it without polluting `WrittenInOrder`.
    void AdoptAheadBeforeMain() {
        if (command_buffer != nil || !merge_ahead_enabled || !ahead_any_enabled ||
            ahead_buffer == nil || upload_buffer != nil || kernel_buffer != nil ||
            kernel_binding || kernel_resume) {
            return;
        }
        if (upload_blit != nil) {
            [upload_blit endEncoding];
            upload_blit = nil;
        }
        EndKernelEncoder();
        kernel_checked = false;
        command_buffer = ahead_buffer;
        ahead_buffer = nil;
        segment_encoders += ahead_encoders;
        for (const ComputeHazardState::BufferSpan& span : ahead_written_spans) {
            ComputeHazardState::AddSpan(adopted_written_spans, span);
        }
        for (const ComputeHazardState::BufferSpan& span : ahead_read_spans) {
            ComputeHazardState::AddSpan(adopted_read_spans, span);
        }
        for (const void* root : ahead_touched_textures) {
            segment_touched_textures.insert(root);
        }
        for (const void* root : ahead_written_textures) {
            segment_written_textures.insert(root);
            segment_touched_textures.insert(root);
        }
        ClearPendingAheadTracking();
    }

    id<MTLCommandBuffer> Commands() {
        if (command_buffer == nil) {
            AdoptAheadBeforeMain();
        }
        if (command_buffer == nil) {
            command_buffer =
                NewCommandBuffer(static_labels_enabled
                                     ? @"Lichii"
                                     : [NSString stringWithFormat:@"Lichii %llu", current_tick]);
        }
        return command_buffer;
    }

    /// The command buffer of what is recorded ahead (EDEN_NX2_METAL_MERGE_AHEAD).
    id<MTLCommandBuffer> AheadBuffer() {
        if (ahead_buffer == nil) {
            ahead_buffer =
                NewCommandBuffer(static_labels_enabled
                    ? @"Lichii ahead"
                    : [NSString stringWithFormat:@"Lichii %llu ahead", current_tick]);
        }
        return ahead_buffer;
    }

    void EndKernelEncoder() {
        if (!ahead_wave_flushing && (!ahead_waves[0].Empty() || !ahead_waves[1].Empty())) {
            FlushAheadComputeWaves();
        }
        if (kernel_compute == nil) {
            return;
        }
        if (active_compute == kernel_compute) {
            active_compute = nil;
            active_bound = nullptr;
            active_hazards = nullptr;
            compute_clobbered_by_copy = false;
        }
        pending_kernel_compute_read_use.clear();
        kernel_hazards.Reset();
        [kernel_compute endEncoding];
        kernel_compute = nil;
    }

    /// The blit encoder of the copies recorded ahead of the segment: in the ahead command
    /// buffer after the kernels recorded before them (EDEN_NX2_METAL_MERGE_AHEAD), else in a
    /// command buffer committed before the kernels' and the segment's.
    id<MTLBlitCommandEncoder> UploadEncoder() {
        if (upload_blit == nil) {
            if (merge_ahead_enabled) {
                if (kernel_compute != nil && active_compute == kernel_compute && kernel_binding) {
                    // In the middle of a kernel's binds: they go to a new encoder after the copy.
                    kernel_resume = true;
                }
                EndKernelEncoder();
                upload_blit = [AheadBuffer() blitCommandEncoder];
                if (command_buffer == nil) {
                    ++ahead_encoders;
                }
            } else {
                if (upload_buffer == nil) {
                    upload_buffer = NewCommandBuffer(
                        static_labels_enabled
                            ? @"Lichii uploads"
                            : [NSString stringWithFormat:@"Lichii %llu uploads", current_tick]);
                }
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
            id<MTLCommandBuffer> __unsafe_unretained target = nil;
            if (merge_ahead_enabled) {
                if (upload_blit != nil) {
                    [upload_blit endEncoding];
                    upload_blit = nil;
                }
                target = AheadBuffer();
                if (command_buffer == nil) {
                    ++ahead_encoders;
                }
            } else {
                if (kernel_buffer == nil) {
                    kernel_buffer = NewCommandBuffer(
                        static_labels_enabled
                            ? @"Lichii kernels"
                            : [NSString stringWithFormat:@"Lichii %llu kernels", current_tick]);
                }
                target = kernel_buffer;
            }
            kernel_compute =
                concurrent_compute_enabled
                    ? [target computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                    : [target computeCommandEncoder];
            kernel_compute.label = @"Kernels";
            kernel_bound = {};
            kernel_hazards.Reset();
            kernel_pipeline = nullptr;
            kernel_compute_serial = ++next_compute_encoder_serial;
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
        const std::shared_ptr<CompletionState> state = completion;
        [buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            const CFTimeInterval gpu_start = finished.GPUStartTime;
            const CFTimeInterval gpu_end = finished.GPUEndTime;
            const CFTimeInterval busy =
                (gpu_start > 0.0 && gpu_end > gpu_start) ? (gpu_end - gpu_start) : 0.0;
            if (busy > 0.0) {
                state->gpu_ns.fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
            }
            ReportFailure(finished, tick, &state->errors_logged);
        }];
        [buffer commit];
        if (recorder != nullptr) {
            recorder->Publish();
        }
        ++command_buffers;
        in_flight.emplace_back(tick, buffer);
    }

    /// When a segment recorded only onto ahead command buffer(s) and never opened `command_buffer`,
    /// adopt the last ordered ahead command buffer as `command_buffer` instead of committing an
    /// extra empty buffer.
    void AdoptAheadIfMainEmpty() {
        if (command_buffer != nil ||
            (ahead_buffer == nil && upload_buffer == nil && kernel_buffer == nil)) {
            return;
        }
        if (upload_blit != nil) {
            [upload_blit endEncoding];
            upload_blit = nil;
        }
        EndKernelEncoder();
        kernel_resume = false;
        kernel_checked = false;
        ClearPendingAheadTracking();
        if (ahead_buffer != nil) {
            command_buffer = ahead_buffer;
            ahead_buffer = nil;
        } else if (kernel_buffer != nil) {
            command_buffer = kernel_buffer;
            kernel_buffer = nil;
        } else if (upload_buffer != nil) {
            command_buffer = upload_buffer;
            upload_buffer = nil;
        }
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
        ClearPendingAheadTracking();
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

    bool WantsAheadEncoder(bool can_compute_copy_without_barrier = false) const noexcept {
        if (!lazy_ahead_enabled || command_buffer == nil || ahead_buffer != nil ||
            upload_buffer != nil || kernel_buffer != nil || render != nil || blit_render != nil) {
            return true;
        }
        if (compute != nil) {
            return !can_compute_copy_without_barrier;
        }
        return false;
    }

    void NextSegment() {
        ++segment_serial;
        segment_encoders = 0;
        segment_render_passes = 0;
        ahead_blocked = false;
        want_in_order_compute = false;
        leaving_render_pass = false;
        ClearPendingAheadTracking();
        adopted_written_spans.clear();
        adopted_read_spans.clear();
        in_order_spans.clear();
        in_order_written_spans.clear();
        in_order_untracked_read_spans.clear();
        untracked_in_order_buffers.clear();
        deferred_downloads.clear();
        segment_bound.clear();
        segment_written_textures.clear();
        segment_touched_textures.clear();
        segment_opaque = false;
        checked_failed = false;
        ahead_query_relaxed = false;
        compute_clobbered_by_copy = false;
        for (PendingClearEntry& entry : pending_clears) {
            entry.blocked_ahead = false;
        }
    }

    void NoteActiveBuffer(const void* buffer) {
        if (buffer == nullptr) {
            return;
        }
        ActiveBufferNote& cached =
            active_buffer_cache[PtrHashIndex(buffer, ActiveBufferCacheBits)];
        if (cached.object == buffer && cached.tick == current_tick) {
            return;
        }
        cached = {buffer, current_tick, 0, false};
        active_buffers.insert(buffer);
    }

    u64 NoteActiveBufferAndLength(const void* buffer, bool need_length) {
        if (buffer == nullptr) {
            return 0;
        }
        ActiveBufferNote& cached =
            active_buffer_cache[PtrHashIndex(buffer, ActiveBufferCacheBits)];
        if (cached.object == buffer && cached.tick == current_tick) {
            if (need_length && !cached.has_length) {
                const id<MTLBuffer> __unsafe_unretained mtl = UnretainedBuffer(buffer);
                cached.length = mtl != nil ? static_cast<u64>(mtl.length) : 0;
                cached.has_length = true;
            }
            return cached.length;
        }
        u64 len = 0;
        if (need_length) {
            const id<MTLBuffer> __unsafe_unretained mtl = UnretainedBuffer(buffer);
            len = mtl != nil ? static_cast<u64>(mtl.length) : 0;
        }
        cached = {buffer, current_tick, len, need_length};
        active_buffers.insert(buffer);
        return len;
    }

    /// A copy recorded in order reads or writes `[offset, offset + size)` of `buffer`.
    void NoteInOrder(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0}) {
        if (buffer != nullptr && size != 0) {
            ComputeHazardState::AddSpan(
                in_order_spans,
                {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
        }
        NoteActiveBuffer(buffer);
    }

    void NoteAheadWrite(BufferHandle buffer) noexcept {
        if (buffer != nullptr) {
            ahead_write_notes[PtrHashIndex(buffer, AheadWriteNotesBits)] = {
                buffer, ++ahead_write_clock};
        }
    }

    /// In-order work writes `buffer` (a copy's destination or storage write).
    void NoteInOrderWrite(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0}) {
        NoteInOrder(buffer, offset, size);
        if (buffer != nullptr && size != 0) {
            ComputeHazardState::AddSpan(
                in_order_written_spans,
                {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
        }
        NoteSegmentBuffer(buffer, true);
    }

    void NoteUntrackedInOrder(BufferHandle buffer) {
        if (buffer != nullptr &&
            std::find(untracked_in_order_buffers.rbegin(), untracked_in_order_buffers.rend(),
                      buffer) == untracked_in_order_buffers.rend()) {
            untracked_in_order_buffers.push_back(buffer);
        }
    }

    void NoteUntrackedInOrderRead(BufferHandle buffer, u64 offset, u64 size) {
        if (buffer == nullptr || size == 0) {
            return;
        }
        if (size == ~u64{0}) {
            NoteUntrackedInOrder(buffer);
            return;
        }
        ComputeHazardState::AddSpan(
            in_order_untracked_read_spans,
            {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
    }

    bool UsedInOrder(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0}) const noexcept {
        if (buffer == nullptr || size == 0) {
            return false;
        }
        return ComputeHazardState::OverlapsAny(
            in_order_spans,
            {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
    }

    bool WrittenInOrder(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0}) const noexcept {
        if (buffer == nullptr || size == 0) {
            return false;
        }
        return ComputeHazardState::OverlapsAny(
            in_order_written_spans,
            {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
    }

    bool UntrackedReadInOrder(BufferHandle buffer, u64 offset = 0,
                              u64 size = ~u64{0}) const noexcept {
        if (buffer == nullptr || size == 0) {
            return false;
        }
        return ComputeHazardState::OverlapsAny(
            in_order_untracked_read_spans,
            {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
    }

    bool AdoptedWritten(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0}) const noexcept {
        if (buffer == nullptr || size == 0) {
            return false;
        }
        return ComputeHazardState::OverlapsAny(
            adopted_written_spans,
            {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
    }

    bool AdoptedRead(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0}) const noexcept {
        if (buffer == nullptr || size == 0) {
            return false;
        }
        return ComputeHazardState::OverlapsAny(
            adopted_read_spans,
            {buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)});
    }

    /// EDEN_NX2_METAL_SEGMENT_NOTE_CACHE: whether the segment noted `object` (a buffer or a
    /// texture) already as this note would (writing, or as `writes_shared` asks): what a note
    /// adds stays until NextSegment. Else it is recorded as noted, the caller noting it now. The
    /// segment's command buffer holds what its work binds: an address names one object.
    bool NotedInSegment(const void* object, bool writes_shared) {
        if (!segment_note_cache_enabled) {
            return false;
        }
        SegmentNote& note = segment_notes[PtrHashIndex(object, SegmentNotesBits)];
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
        if (!writes || buffer == nullptr || !ahead_any_enabled || NotedInSegment(buffer, true)) {
            return;
        }
        segment_bound.insert(buffer);
    }

    void AddPassTouchedRoot(const void* raw_root) {
        if (raw_root == nullptr) {
            return;
        }
        const u64 bit = PtrBloomBit(raw_root);
        if ((pass_touched_bloom & bit) == 0 ||
            !ComputeHazardState::Contains(pass_touched_roots, raw_root)) {
            pass_touched_bloom |= bit;
            pass_touched_roots.push_back(raw_root);
        }
    }

    void AddPassSampledRoot(const void* raw_root) {
        if (raw_root == nullptr) {
            return;
        }
        const u64 bit = PtrBloomBit(raw_root);
        if ((pass_sampled_bloom & bit) == 0 ||
            !ComputeHazardState::Contains(pass_sampled_roots, raw_root)) {
            pass_sampled_bloom |= bit;
            pass_sampled_roots.push_back(raw_root);
        }
    }

    bool IsPassTouchedRoot(const void* raw_root) const noexcept {
        return raw_root != nullptr && (pass_touched_bloom & PtrBloomBit(raw_root)) != 0 &&
               ComputeHazardState::Contains(pass_touched_roots, raw_root);
    }

    bool IsPassSampledRoot(const void* raw_root) const noexcept {
        return raw_root != nullptr && (pass_sampled_bloom & PtrBloomBit(raw_root)) != 0 &&
               ComputeHazardState::Contains(pass_sampled_roots, raw_root);
    }

    const void* CachedRootTexture(const void* texture) const noexcept {
        if (texture == nullptr) {
            return nullptr;
        }
        const TouchedNote& cached =
            segment_touched_cache[PtrHashIndex(texture, SegmentTouchedCacheBits)];
        if (cached.object == texture && cached.serial == segment_serial) {
            return cached.root;
        }
        return RootTextureHandle(texture);
    }

    const void* NoteTextureTouched(const void* texture) {
        if (texture == nullptr) {
            return nullptr;
        }
        TouchedNote& cached =
            segment_touched_cache[PtrHashIndex(texture, SegmentTouchedCacheBits)];
        if (cached.object == texture && cached.serial == segment_serial) {
            if (render != nil && cached.pass_serial != render_pass_serial) {
                cached.pass_serial = render_pass_serial;
                AddPassTouchedRoot(cached.root);
            }
            return cached.root;
        }
        const void* const raw_root = RootTextureHandle(texture);
        if (raw_root == nullptr) {
            return nullptr;
        }
        const u64 pserial = render != nil ? render_pass_serial : ~u64{0};
        if (render != nil) {
            AddPassTouchedRoot(raw_root);
        }
        cached = {texture, raw_root, segment_serial, pserial, ~u64{0}};
        segment_touched_textures.insert(raw_root);
        return raw_root;
    }

    void NoteTextureWrite(const void* texture) {
        if (texture == nullptr) {
            return;
        }
        if (const void* const raw_root = NoteTextureTouched(texture); raw_root != nullptr) {
            segment_written_textures.insert(raw_root);
        }
    }

    /// The buffer a texture view of a buffer is on.
    void NoteSegmentTexture(const void* texture, bool writes, const void* known_root = nullptr) {
        if (!writes || texture == nullptr || !ahead_any_enabled || NotedInSegment(texture, true)) {
            return;
        }
        const void* const raw_root =
            known_root != nullptr ? known_root : CachedRootTexture(texture);
        const ComputeHazardState::BufferSpan span = ComputeHazardState::BackingBufferSpan(raw_root);
        if (span.buffer != nullptr && span.end > span.begin) {
            const u64 sz = span.end == ~u64{0} ? ~u64{0} : (span.end - span.begin);
            NoteInOrderWrite(const_cast<void*>(span.buffer), span.begin, sz);
        }
    }

    const void* NoteRenderTextureUse(TextureHandle texture, bool on_buffer, bool write) {
        if (render == nil || texture == nullptr) {
            return nullptr;
        }
        if (pending_clear_roots_bloom != 0) {
            const void* const pre_root = CachedRootTexture(texture);
            if (PendingClearOnRoot(pre_root)) {
                FlushClearAheadForRoot(pre_root);
            }
        }
        if (render == nil) {
            return nullptr;
        }
        const void* const raw_root = NoteTextureTouched(texture);
        if (raw_root != nullptr) {
            TouchedNote& cached =
                segment_touched_cache[PtrHashIndex(texture, SegmentTouchedCacheBits)];
            if (cached.object == texture && cached.serial == segment_serial) {
                if (cached.sampled_pass_serial != render_pass_serial) {
                    cached.sampled_pass_serial = render_pass_serial;
                    AddPassSampledRoot(raw_root);
                }
            } else {
                AddPassSampledRoot(raw_root);
            }
        }
        if (write && raw_root != nullptr) {
            segment_written_textures.insert(raw_root);
        }
        if (on_buffer && raw_root != nullptr) {
            const ComputeHazardState::BufferSpan bspan =
                ComputeHazardState::BackingBufferSpan(raw_root);
            if (bspan.buffer != nullptr) {
                NoteActiveBuffer(bspan.buffer);
                if (write && bspan.end > bspan.begin) {
                    const u64 bsz = bspan.end == ~u64{0} ? ~u64{0} : (bspan.end - bspan.begin);
                    FlushDeferredDownloadsFor(const_cast<void*>(bspan.buffer), bspan.begin, bsz,
                                              true);
                    if (render == nil) {
                        return raw_root;
                    }
                    NoteTextureTouched(texture);
                    AddPassSampledRoot(raw_root);
                }
            }
        }
        if (on_buffer || !image_note_skip_enabled) {
            NoteSegmentTexture(texture, write, raw_root);
        }
        return raw_root;
    }

    /// Whether the checked kernel binding `buffer` may see something in-order or adopted ahead
    /// work of the segment writes (or overwrite a span an earlier untracked/adopted read needs).
    bool CheckedConflict(const void* buffer, u64 offset = 0, u64 size = ~u64{0},
                         bool writes = false) const {
        if (buffer == nullptr || size == 0) {
            return false;
        }
        const BufferHandle handle = const_cast<void*>(buffer);
        return WrittenInOrder(handle, offset, size) || AdoptedWritten(handle, offset, size) ||
               (writes && (UntrackedReadInOrder(handle, offset, size) ||
                           AdoptedRead(handle, offset, size)));
    }

    void LogComputeBind(ComputeBind::Kind kind, u32 slot, void* object, u32 offset, bool write,
                        bool on_buffer = true, u64 hazard_read_size = ~u64{0},
                        u64 hazard_write_size = ~u64{0}, bool track_hazard = true,
                        bool disjoint_write = false) {
        compute_binds.push_back({kind, on_buffer, write, slot, object, offset, 0, hazard_read_size,
                                 hazard_write_size, track_hazard, disjoint_write});
    }

    void LogComputeBytes(u32 slot, const void* data, u32 size) {
        const u32 at = static_cast<u32>(compute_bind_bytes.size());
        const u8* const bytes = static_cast<const u8*>(data);
        compute_bind_bytes.insert(compute_bind_bytes.end(), bytes, bytes + size);
        compute_binds.push_back({ComputeBind::Kind::Bytes, false, false, slot, nullptr, at, size});
    }

    void ReplayComputeBindList(std::span<const ComputeBind> binds, std::span<const u8> bytes,
                               void* pipeline, const void** bound_pipeline, bool in_order) {
        const id<MTLComputeCommandEncoder> __unsafe_unretained enc = active_compute;
        if (enc == nil) {
            return;
        }
        if (active_hazards != nullptr) {
            active_hazards->ResetPending();
        }
        if (!state_cache_enabled || *bound_pipeline != pipeline) {
            *bound_pipeline = pipeline;
            ++state_sets;
            [enc setComputePipelineState:(__bridge __unsafe_unretained id<MTLComputePipelineState>)
                                             pipeline];
        } else {
            ++state_skips;
        }
        for (const ComputeBind& bind : binds) {
            switch (bind.kind) {
            case ComputeBind::Kind::Buffer:
                if (active_hazards != nullptr) {
                    active_hazards->BindSlotBuffer(bind.slot, bind.object, bind.write, bind.offset,
                                                   bind.hazard_read_size, bind.hazard_write_size,
                                                   bind.disjoint_write);
                }
                SetBuffer(ShaderStage::Compute, bind.slot, bind.object, bind.offset);
                if (in_order && bind.write) {
                    const u64 eff_write = bind.hazard_write_size != 0 ? bind.hazard_write_size
                                                                      : bind.hazard_read_size;
                    NoteInOrderWrite(bind.object, bind.offset, eff_write);
                }
                break;
            case ComputeBind::Kind::Bytes:
                if (active_hazards != nullptr) {
                    active_hazards->ClearSlotBuffer(bind.slot);
                }
                if (bind.slot < BufferSlots) {
                    active_bound->buffers[bind.slot] = &ClobberedSlot;
                    active_bound->ClearBufferDirty(bind.slot);
                }
                ++state_sets;
                [enc setBytes:bytes.data() + bind.offset length:bind.size atIndex:bind.slot];
                break;
            case ComputeBind::Kind::Texture: {
                const void* const known_root =
                    bind.track_hazard && bind.object != nullptr ? CachedRootTexture(bind.object)
                                                                : nullptr;
                if (bind.track_hazard && active_hazards != nullptr) {
                    active_hazards->BindSlotTexture(bind.slot, bind.object, bind.on_buffer,
                                                    bind.write, known_root);
                }
                if (NoteObject(active_bound->textures, bind.slot, bind.object)) {
                    if (bind.slot < TextureSlots && range_binds_enabled) {
                        active_bound->MarkTextureDirty(bind.slot);
                    } else {
                        [enc setTexture:UnretainedTexture(bind.object) atIndex:bind.slot];
                    }
                }
                if (bind.track_hazard && in_order && bind.object != nullptr) {
                    if (bind.write) {
                        NoteTextureWrite(bind.object);
                    } else {
                        NoteTextureTouched(bind.object);
                    }
                    if (bind.write) {
                        NoteSegmentTexture(bind.object, true, known_root);
                    }
                }
                break;
            }
            case ComputeBind::Kind::Sampler:
                if (NoteObject(active_bound->samplers, bind.slot, bind.object)) {
                    if (bind.slot < SamplerSlots && range_binds_enabled) {
                        active_bound->MarkSamplerDirty(bind.slot);
                    } else {
                        [enc setSamplerState:UnretainedSampler(bind.object) atIndex:bind.slot];
                    }
                }
                break;
            case ComputeBind::Kind::UseBuffer:
                if (active_hazards != nullptr) {
                    active_hazards->NoteBuffer(bind.object, bind.write, bind.offset,
                                               bind.hazard_read_size, bind.hazard_write_size);
                }
                if (!bind.write && bind.object != nullptr && range_binds_enabled &&
                    recorder == nullptr) {
                    ActiveComputeReadUse().push_back(bind.object);
                } else if (bind.object != nullptr) {
                    [enc useResource:UnretainedBuffer(bind.object)
                               usage:bind.write ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                                                : MTLResourceUsageRead];
                }
                if (in_order && bind.write) {
                    const u64 eff_write = bind.hazard_write_size != 0 ? bind.hazard_write_size
                                                                      : bind.hazard_read_size;
                    NoteInOrderWrite(bind.object, bind.offset, eff_write);
                }
                break;
            case ComputeBind::Kind::UseTexture: {
                const void* const known_root =
                    bind.track_hazard && bind.object != nullptr ? CachedRootTexture(bind.object)
                                                                : nullptr;
                if (bind.track_hazard && active_hazards != nullptr) {
                    active_hazards->NoteTexture(bind.object, bind.on_buffer, bind.write,
                                                known_root);
                }
                if (!bind.write && bind.object != nullptr && range_binds_enabled &&
                    recorder == nullptr) {
                    AppendTextureReadResources(ActiveComputeReadUse(),
                                               UnretainedTexture(bind.object), bind.on_buffer,
                                               known_root);
                } else if (bind.object != nullptr) {
                    UseTextureResources(enc, UnretainedTexture(bind.object),
                                        bind.write ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                                                   : MTLResourceUsageRead,
                                        bind.on_buffer);
                }
                if (bind.track_hazard && in_order && bind.object != nullptr) {
                    if (bind.write) {
                        NoteTextureWrite(bind.object);
                    } else {
                        NoteTextureTouched(bind.object);
                    }
                    if (bind.write) {
                        NoteSegmentTexture(bind.object, true, known_root);
                    }
                }
                break;
            }
            }
        }
        FlushComputeBindings();
    }

    /// Sets the kernel's pipeline and binds again on the active compute encoder (a new one).
    void ReplayComputeBinds(const void** bound_pipeline, bool in_order) {
        ReplayComputeBindList(compute_binds, compute_bind_bytes, compute_bind_pipeline,
                              bound_pipeline, in_order);
    }

    void FlushAheadComputeWaves() {
        if (ahead_wave_flushing || (ahead_waves[0].Empty() && ahead_waves[1].Empty())) {
            return;
        }
        ahead_wave_flushing = true;
        const id<MTLComputeCommandEncoder> __unsafe_unretained enc = KernelEncoder();
        active_compute = enc;
        active_bound = &kernel_bound;
        active_hazards = &kernel_hazards;
        compute_clobbered_by_copy = false;
        kernel_resume = false;
        for (size_t w = 0; w < ahead_waves.size(); ++w) {
            AheadWaveQueue& q = ahead_waves[w];
            for (const DeferredAheadDispatch& d : q.dispatches) {
                ReplayComputeBindList(
                    std::span<const ComputeBind>(q.binds.data() + d.bind_begin, d.bind_count),
                    q.bytes, d.pipeline, &kernel_pipeline, false);
                if (d.indirect) {
                    const ComputeHazardState::BufferSpan ind_span{
                        d.indirect_buffer, d.indirect_offset,
                        ComputeHazardState::SaturatingEnd(d.indirect_offset, 12)};
                    kernel_hazards.BeforeDispatch(enc, concurrent_compute_enabled, ind_span);
                    [enc dispatchThreadgroupsWithIndirectBuffer:UnretainedBuffer(d.indirect_buffer)
                                           indirectBufferOffset:d.indirect_offset
                                          threadsPerThreadgroup:MTLSizeMake(d.group_size[0],
                                                                            d.group_size[1],
                                                                            d.group_size[2])];
                    kernel_hazards.AfterDispatch(concurrent_compute_enabled, ind_span);
                } else {
                    kernel_hazards.BeforeDispatch(enc, concurrent_compute_enabled);
                    [enc dispatchThreadgroups:MTLSizeMake(d.groups[0], d.groups[1], d.groups[2])
                        threadsPerThreadgroup:MTLSizeMake(d.group_size[0], d.group_size[1],
                                                          d.group_size[2])];
                    kernel_hazards.AfterDispatch(concurrent_compute_enabled);
                }
                CountDispatch();
            }
            q.Clear();
        }
        ahead_wave_read_spans.clear();
        ahead_wave_flushing = false;
    }

    void AbortRecordingAheadWave(TextureHandle conflict_texture) {
        const int w = recording_ahead_wave;
        if (w < 0 || w >= static_cast<int>(ahead_waves.size())) {
            return;
        }
        AheadWaveQueue& q = ahead_waves[static_cast<size_t>(w)];
        compute_binds.clear();
        compute_bind_bytes.clear();
        compute_bind_pipeline = recording_wave_pipeline;
        if (recording_wave_bytes_start < q.bytes.size()) {
            compute_bind_bytes.insert(compute_bind_bytes.end(),
                                      q.bytes.begin() + recording_wave_bytes_start, q.bytes.end());
        }
        for (size_t i = recording_wave_bind_start; i < q.binds.size(); ++i) {
            ComputeBind b = q.binds[i];
            if (b.kind == ComputeBind::Kind::Bytes) {
                b.offset -= recording_wave_bytes_start;
            }
            compute_binds.push_back(b);
        }
        q.binds.resize(recording_wave_bind_start);
        q.bytes.resize(recording_wave_bytes_start);
        recording_ahead_wave = -1;
        ahead_wave_checked = false;
        FlushAheadComputeWaves();
        active_compute = kernel_compute;
        active_bound = &kernel_bound;
        active_hazards = &kernel_hazards;
        LeaveChecked(conflict_texture);
    }

    /// EDEN_NX2_METAL_COMPUTE_COPY: a buffer copy as a dispatch on the open compute encoder
    /// `encoder` (bound as `bound`, its pipeline in `bound_pipeline`), in place of ending it for
    /// a blit encoder and opening another after the copy. False when the copy kernel is not there.
    bool CopyByCompute(__unsafe_unretained id<MTLComputeCommandEncoder> encoder,
                       StageBindings& bound, ComputeHazardState& hazards,
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
        if (encoder == kernel_compute && !ahead_wave_read_spans.empty()) {
            const ComputeHazardState::BufferSpan dst_span{
                dst, dst_offset, ComputeHazardState::SaturatingEnd(dst_offset, size)};
            if (ComputeHazardState::OverlapsAny(ahead_wave_read_spans, dst_span)) {
                FlushAheadComputeWaves();
            }
        }
        if (concurrent_compute_enabled &&
            hazards.HasCopyHazard(src, src_offset, size, dst, dst_offset)) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ++hazards.barriers_emitted;
            hazards.ClearBufferHazards();
        }
        const void* const copy_pipe = (__bridge const void*)copy_kernel;
        if (!state_cache_enabled || *bound_pipeline != copy_pipe) {
            *bound_pipeline = copy_pipe;
            ++state_sets;
            [encoder setComputePipelineState:copy_kernel];
        } else {
            ++state_skips;
        }
        switch (NoteBuffer(bound, 0, src, 0)) {
        case Bound::Same:
            if ((bound.dirty_buffers & 1U) != 0) {
                [encoder setBuffer:UnretainedBuffer(src) offset:0 atIndex:0];
            }
            break;
        case Bound::Offset:
            if ((bound.dirty_buffers & 1U) != 0) {
                [encoder setBuffer:UnretainedBuffer(src) offset:0 atIndex:0];
            } else {
                [encoder setBufferOffset:0 atIndex:0];
            }
            break;
        case Bound::Other: {
            const id<MTLBuffer> __unsafe_unretained mtl_src = UnretainedBuffer(src);
            if (0 < BufferSlots) {
                bound.lengths[0] = NoteActiveBufferAndLength(src, pipeline_check_enabled);
            }
            [encoder setBuffer:mtl_src offset:0 atIndex:0];
            break;
        }
        }
        bound.ClearBufferDirty(0);
        switch (NoteBuffer(bound, 1, dst, 0)) {
        case Bound::Same:
            if ((bound.dirty_buffers & 2U) != 0) {
                [encoder setBuffer:UnretainedBuffer(dst) offset:0 atIndex:1];
            }
            break;
        case Bound::Offset:
            if ((bound.dirty_buffers & 2U) != 0) {
                [encoder setBuffer:UnretainedBuffer(dst) offset:0 atIndex:1];
            } else {
                [encoder setBufferOffset:0 atIndex:1];
            }
            break;
        case Bound::Other: {
            const id<MTLBuffer> __unsafe_unretained mtl_dst = UnretainedBuffer(dst);
            if (1 < BufferSlots) {
                bound.lengths[1] = NoteActiveBufferAndLength(dst, pipeline_check_enabled);
            }
            [encoder setBuffer:mtl_dst offset:0 atIndex:1];
            break;
        }
        }
        bound.ClearBufferDirty(1);
        bound.buffers[2] = &ClobberedSlot;
        bound.ClearBufferDirty(2);
        ++state_sets;
        const std::array<u64, 3> params{src_offset, dst_offset, size};
        [encoder setBytes:params.data() length:sizeof(params) atIndex:2];
        const u64 threads = (size + 15) / 16;
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((threads + 63) / 64), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (concurrent_compute_enabled) {
            ComputeHazardState::AddSpan(
                hazards.read_buffers,
                {src, src_offset, ComputeHazardState::SaturatingEnd(src_offset, size)});
            ComputeHazardState::AddSpan(
                hazards.written_buffers,
                {dst, dst_offset, ComputeHazardState::SaturatingEnd(dst_offset, size)});
        }
        if (active_compute == encoder) {
            compute_clobbered_by_copy = true;
        }
        return true;
    }

    bool FillByCompute(__unsafe_unretained id<MTLComputeCommandEncoder> encoder,
                       StageBindings& bound, ComputeHazardState& hazards,
                       const void** bound_pipeline, BufferHandle dst, u64 dst_offset, u64 size,
                       u32 pattern) {
        if (fill_kernel == nil) {
            if (fill_kernel_failed || library == nil) {
                return false;
            }
            NSError* error = nil;
            const id<MTLFunction> function = [library newFunctionWithName:@"fill_pattern"];
            fill_kernel = function != nil
                              ? [device.GetDevice() newComputePipelineStateWithFunction:function
                                                                                  error:&error]
                              : nil;
            if (fill_kernel == nil) {
                fill_kernel_failed = true;
                LOG_ERROR(Render_Metal, "Metal: buffer fill kernel failed: {}",
                          error != nil ? error.localizedDescription.UTF8String : "no function");
                return false;
            }
        }
        if (encoder == kernel_compute && !ahead_wave_read_spans.empty()) {
            const ComputeHazardState::BufferSpan dst_span{
                dst, dst_offset, ComputeHazardState::SaturatingEnd(dst_offset, size)};
            if (ComputeHazardState::OverlapsAny(ahead_wave_read_spans, dst_span)) {
                FlushAheadComputeWaves();
            }
        }
        if (concurrent_compute_enabled && hazards.HasFillHazard(dst, dst_offset, size)) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ++hazards.barriers_emitted;
            hazards.ClearBufferHazards();
        }
        const void* const fill_pipe = (__bridge const void*)fill_kernel;
        if (!state_cache_enabled || *bound_pipeline != fill_pipe) {
            *bound_pipeline = fill_pipe;
            ++state_sets;
            [encoder setComputePipelineState:fill_kernel];
        } else {
            ++state_skips;
        }
        switch (NoteBuffer(bound, 0, dst, 0)) {
        case Bound::Same:
            if ((bound.dirty_buffers & 1U) != 0) {
                [encoder setBuffer:UnretainedBuffer(dst) offset:0 atIndex:0];
            }
            break;
        case Bound::Offset:
            if ((bound.dirty_buffers & 1U) != 0) {
                [encoder setBuffer:UnretainedBuffer(dst) offset:0 atIndex:0];
            } else {
                [encoder setBufferOffset:0 atIndex:0];
            }
            break;
        case Bound::Other: {
            const id<MTLBuffer> __unsafe_unretained mtl_dst = UnretainedBuffer(dst);
            if (0 < BufferSlots) {
                bound.lengths[0] = NoteActiveBufferAndLength(dst, pipeline_check_enabled);
            }
            [encoder setBuffer:mtl_dst offset:0 atIndex:0];
            break;
        }
        }
        bound.ClearBufferDirty(0);
        bound.buffers[1] = &ClobberedSlot;
        bound.ClearBufferDirty(1);
        ++state_sets;
        struct FillParamsHost {
            u64 dst;
            u64 size;
            u32 pattern;
            u32 pad;
        } const params{dst_offset, size, pattern, 0};
        [encoder setBytes:&params length:sizeof(params) atIndex:1];
        const u64 threads = (size + 15) / 16;
        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>((threads + 63) / 64), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (concurrent_compute_enabled) {
            ComputeHazardState::AddSpan(
                hazards.written_buffers,
                {dst, dst_offset, ComputeHazardState::SaturatingEnd(dst_offset, size)});
        }
        if (active_compute == encoder) {
            compute_clobbered_by_copy = true;
        }
        return true;
    }

    /// The segment's compute encoder, opened when there is none, made the active one.
    void OpenInOrderCompute() {
        want_in_order_compute = false;
        if (compute == nil) {
            EndEncoder(PassEnd::Compute);
            compute = concurrent_compute_enabled
                          ? [Commands()
                                computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                          : [Commands() computeCommandEncoder];
            compute.label = @"Compute";
            compute_bound = {};
            compute_hazards.Reset();
            compute_pipeline = nullptr;
            in_order_compute_serial = ++next_compute_encoder_serial;
            ++encoders;
            ++segment_encoders;
        }
        active_compute = compute;
        active_bound = &compute_bound;
        active_hazards = &compute_hazards;
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
            active_hazards = &kernel_hazards;
            ReplayComputeBinds(&kernel_pipeline, false);
            return;
        }
        if (!compute_clobbered_by_copy) {
            return;
        }
        compute_clobbered_by_copy = false;
        const id<MTLComputeCommandEncoder> __unsafe_unretained enc = active_compute;
        if (enc == nil) {
            return;
        }
        const void** const bound_pipeline =
            enc == kernel_compute ? &kernel_pipeline : &compute_pipeline;
        if (compute_bind_pipeline != nullptr &&
            (!state_cache_enabled || *bound_pipeline != compute_bind_pipeline)) {
            *bound_pipeline = compute_bind_pipeline;
            ++state_sets;
            [enc setComputePipelineState:(__bridge __unsafe_unretained id<MTLComputePipelineState>)
                                             compute_bind_pipeline];
        }
        for (const ComputeBind& bind : compute_binds) {
            if (bind.slot >= 3) {
                continue;
            }
            if (bind.kind == ComputeBind::Kind::Buffer) {
                SetBuffer(ShaderStage::Compute, bind.slot, bind.object, bind.offset);
            } else if (bind.kind == ComputeBind::Kind::Bytes) {
                active_bound->buffers[bind.slot] = &ClobberedSlot;
                active_bound->ClearBufferDirty(bind.slot);
                ++state_sets;
                [enc setBytes:compute_bind_bytes.data() + bind.offset
                       length:bind.size
                      atIndex:bind.slot];
            }
        }
    }

    /// The checked kernel conflicts with in-order work: it goes in order from here on, with the
    /// binds it has made so far.
    void LeaveChecked(TextureHandle texture) {
        kernel_checked = false;
        kernel_binding = false;
        checked_failed = true;
        ++ahead_fallbacks;
        if (active_bound != nullptr) {
            active_bound->InvalidateDirty(&ClobberedSlot);
        }
        pending_kernel_compute_read_use.clear();
        kernel_compute_serial = ++next_compute_encoder_serial;
        if (active_hazards != nullptr) {
            active_hazards->ResetPending();
        }
        // As the rasterizer does for a kernel that ends the pass it draws into.
        if (render != nil && ahead_split_enabled) {
            const WantedPassKey open_key = MakeWantedPassKey(render_targets);
            if (open_key == ahead_query_key) {
                NoteWantedKey(ahead_query_key, true);
            }
        }
        if (!clear_narrow_enabled) {
            FlushClear();
        } else if (texture != nullptr) {
            Touch(texture);
        }
        OpenInOrderCompute();
        ReplayComputeBinds(&compute_pipeline, true);
    }

    /// Before a compute bind of `buffer`: checks the checked kernel, notes in-order binds.
    void ComputeBuffer(std::optional<u32> slot, const void* buffer, bool writes = true,
                       u64 offset = 0, u64 read_size = ~u64{0}, u64 write_size = ~u64{0},
                       bool add_to_pending = true, bool disjoint_write = false) {
        ResumeKernel();
        const u64 eff_write = write_size != 0 ? write_size : read_size;
        const u64 conflict_size = writes ? (std::max)(read_size, eff_write) : read_size;
        if (kernel_checked && CheckedConflict(buffer, offset, conflict_size, writes)) {
            LeaveChecked(nullptr);
        }
        if (writes && active_compute != nil && active_compute == compute) {
            FlushDeferredDownloadsFor(const_cast<void*>(buffer), offset, eff_write, false);
            if (compute == nil) {
                OpenInOrderCompute();
                ReplayComputeBinds(&compute_pipeline, true);
            }
        }
        if (active_compute != nil) {
            NoteActiveBuffer(buffer);
        }
        if (add_to_pending && active_hazards != nullptr) {
            if (slot) {
                active_hazards->BindSlotBuffer(*slot, buffer, writes, offset, read_size,
                                               write_size, disjoint_write);
            } else {
                active_hazards->NoteBuffer(buffer, writes, offset, read_size, write_size);
            }
        }
        if (active_compute != nil && active_compute == compute) {
            if (writes) {
                NoteInOrderWrite(const_cast<void*>(buffer), offset, eff_write);
            } else {
                NoteSegmentBuffer(buffer, false);
            }
        } else if (active_compute != nil && active_compute == kernel_compute) {
            NoteAheadReadSpan(buffer, offset, read_size);
            if (writes) {
                NoteAheadWriteSpan(buffer, offset, eff_write);
            }
        }
    }

    /// Before a compute bind of `texture`: a checked kernel only falls back in order when the
    /// root texture (or its backing buffer) was modified in the segment or has a pending clear;
    /// an in-order one records a waiting clear of it first.
    const void* ComputeTexture(std::optional<u32> slot, TextureHandle texture,
                               bool on_buffer = true, bool write = true, bool track_hazard = true) {
        ResumeKernel();
        if (texture == nullptr || !track_hazard) {
            if (slot && active_hazards != nullptr) {
                active_hazards->BindSlotTexture(*slot, nullptr, false, false, nullptr);
            }
            return nullptr;
        }
        const bool was_in_order = active_compute != nil && active_compute == compute;
        const void* const raw_root = CachedRootTexture(texture);
        const u64 restored_before = samples_restored;
        if (write) {
            WrittenRoot(texture, raw_root);
        } else {
            RestoreSamplesIfDropped(texture, raw_root);
        }
        const bool restored = samples_restored != restored_before;
        const ComputeHazardState::BufferSpan bspan =
            on_buffer ? ComputeHazardState::BackingBufferSpan(raw_root)
                      : ComputeHazardState::BufferSpan{};
        const u64 bsz = bspan.end == ~u64{0} ? ~u64{0} : (bspan.end - bspan.begin);
        if (active_compute != nil && active_compute != compute) {
            if (restored || PendingClearOnRoot(raw_root)) {
                LeaveChecked(texture);
            } else if (kernel_checked) {
                if ((write ? segment_touched_textures : segment_written_textures)
                        .contains(raw_root) ||
                    (bspan.buffer != nullptr &&
                     CheckedConflict(bspan.buffer, bspan.begin, bsz, write))) {
                    LeaveChecked(texture);
                }
            }
            if (active_compute == kernel_compute) {
                NoteAheadTexture(raw_root, write);
                if (bspan.buffer != nullptr && bspan.end > bspan.begin) {
                    NoteAheadReadSpan(bspan.buffer, bspan.begin, bsz);
                    if (write) {
                        NoteAheadWriteSpan(bspan.buffer, bspan.begin, bsz);
                    }
                }
            }
        } else if (was_in_order && compute == nil) {
            OpenInOrderCompute();
            ReplayComputeBinds(&compute_pipeline, true);
        }
        if (active_compute != nil && active_compute == compute && pending_clear_roots_bloom != 0 &&
            PendingClearOnRoot(raw_root)) {
            EndEncoder();
            if (!clear_narrow_enabled) {
                FlushClear();
            } else {
                FlushClearForRoot(raw_root);
            }
            OpenInOrderCompute();
            ReplayComputeBinds(&compute_pipeline, true);
        }
        if (on_buffer && write && active_compute != nil && active_compute == compute &&
            bspan.buffer != nullptr && bspan.end > bspan.begin) {
            FlushDeferredDownloadsFor(const_cast<void*>(bspan.buffer), bspan.begin, bsz, false);
            if (compute == nil) {
                OpenInOrderCompute();
                ReplayComputeBinds(&compute_pipeline, true);
            }
        }
        if (on_buffer && active_compute != nil && bspan.buffer != nullptr) {
            NoteActiveBuffer(bspan.buffer);
        }
        if (active_hazards != nullptr) {
            if (slot) {
                active_hazards->BindSlotTexture(*slot, texture, on_buffer, write, raw_root);
            } else {
                active_hazards->NoteTexture(texture, on_buffer, write, raw_root);
            }
        }
        if (active_compute == nil || active_compute != compute) {
            return raw_root;
        }
        if (write) {
            NoteTextureWrite(texture);
        } else {
            NoteTextureTouched(texture);
        }
        if (on_buffer || !image_note_skip_enabled) {
            NoteSegmentTexture(texture, write, raw_root);
        }
        return raw_root;
    }

    /// Commits what the tick recorded so far; the tick goes on in a new segment.
    void Split() {
        EndEncoder(PassEnd::Split);
        AdoptAheadIfMainEmpty();
        FlushDeferredDownloads();
        if (blit != nil) {
            [blit endEncoding];
            blit = nil;
        }
        CommitAhead(current_tick);
        const id<MTLCommandBuffer> segment = command_buffer;
        command_buffer = nil;
        CommitEarly(segment, current_tick, CommitReason::Split);
        NextSegment();
        ++splits;
        ++tick_splits;
    }

    struct WantedEntry {
        WantedPassKey key{};
        u64 serial{};
        bool force_split{};
    };
    static constexpr u64 WantedPassTTLSegments = 24;

    void NoteWantedKey(const WantedPassKey& key, bool force_split = false) noexcept {
        const size_t count = (std::min)(wanted_count, wanted.size());
        for (size_t i = 0; i < count; ++i) {
            if (wanted[i].key == key) {
                wanted[i].serial = segment_serial;
                wanted[i].force_split = wanted[i].force_split || force_split;
                return;
            }
        }
        wanted[wanted_count++ % wanted.size()] = WantedEntry{key, segment_serial, force_split};
    }

    const WantedEntry* FindWantedKey(const WantedPassKey& key) const noexcept {
        const size_t count = (std::min)(wanted_count, wanted.size());
        for (size_t i = 0; i < count; ++i) {
            if (wanted[i].key == key &&
                segment_serial - wanted[i].serial <= WantedPassTTLSegments) {
                return &wanted[i];
            }
        }
        return nullptr;
    }

    const WantedEntry* FindWanted(const RenderTargetSet& targets) const noexcept {
        if (wanted_count == 0) {
            return nullptr;
        }
        return FindWantedKey(MakeWantedPassKey(targets));
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
        if (!state_cache_enabled || slot >= bound.buffers.size()) {
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
        if (!state_cache_enabled || slot >= N) {
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
        StageBindings* bound = nullptr;
        switch (stage) {
        case ShaderStage::Vertex:
            if (render == nil) {
                return;
            }
            bound = &vertex_bound;
            break;
        case ShaderStage::Fragment:
            if (render == nil) {
                return;
            }
            bound = &fragment_bound;
            break;
        case ShaderStage::Object:
            if (render == nil) {
                return;
            }
            bound = &object_bound;
            break;
        case ShaderStage::Mesh:
            if (render == nil) {
                return;
            }
            bound = &mesh_bound;
            break;
        case ShaderStage::Compute:
            if (active_compute == nil) {
                return;
            }
            bound = active_bound;
            break;
        }
        const bool cached = state_cache_enabled && slot < BufferSlots;
        const bool same_buffer = cached && bound->buffers[slot] == buffer;
        if (same_buffer && (buffer == nullptr || bound->offsets[slot] == offset)) {
            ++state_skips;
            return;
        }
        // Metal checks an offset past the buffer's end only in its validation layer; without it
        // the GPU reads past the allocation. Checked after the redundant-bind memo, using the
        // cached length when the same buffer stays bound at a new offset.
        id<MTLBuffer> __unsafe_unretained mtl_buffer = UnretainedBuffer(buffer);
        u64 buf_len = 0;
        if (mtl_buffer != nil) {
            buf_len = same_buffer ? bound->lengths[slot]
                                  : NoteActiveBufferAndLength(buffer, pipeline_check_enabled);
            if (pipeline_check_enabled && offset >= buf_len) {
                Coverage::NoteGuardWith("buffer offset past its end", [&] {
                    return fmt::format("offset {} of a {}-byte buffer", offset, buf_len);
                });
                mtl_buffer = ZeroBuffer();
                buffer = (__bridge BufferHandle)mtl_buffer;
                buf_len = NoteActiveBufferAndLength(buffer, true);
                offset = 0;
            }
        }
        Bound kind = Bound::Other;
        if (cached) {
            const bool still_same = bound->buffers[slot] == buffer;
            if (still_same && (buffer == nullptr || bound->offsets[slot] == offset)) {
                ++state_skips;
                return;
            }
            bound->buffers[slot] = buffer;
            bound->offsets[slot] = offset;
            ++state_sets;
            kind = still_same ? Bound::Offset : Bound::Other;
            if (kind == Bound::Other) {
                bound->lengths[slot] = buf_len;
            }
        } else {
            ++state_sets;
        }
        if (slot < BufferSlots && range_binds_enabled) {
            bound->MarkBufferDirty(slot, kind == Bound::Offset);
            return;
        }
        const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = render;
        const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = active_compute;
        switch (stage) {
        case ShaderStage::Vertex:
            if (kind == Bound::Offset) {
                [r_enc setVertexBufferOffset:offset atIndex:slot];
            } else {
                [r_enc setVertexBuffer:mtl_buffer offset:offset atIndex:slot];
            }
            break;
        case ShaderStage::Fragment:
            if (kind == Bound::Offset) {
                [r_enc setFragmentBufferOffset:offset atIndex:slot];
            } else {
                [r_enc setFragmentBuffer:mtl_buffer offset:offset atIndex:slot];
            }
            break;
        case ShaderStage::Object:
            if (kind == Bound::Offset) {
                [r_enc setObjectBufferOffset:offset atIndex:slot];
            } else {
                [r_enc setObjectBuffer:mtl_buffer offset:offset atIndex:slot];
            }
            break;
        case ShaderStage::Mesh:
            if (kind == Bound::Offset) {
                [r_enc setMeshBufferOffset:offset atIndex:slot];
            } else {
                [r_enc setMeshBuffer:mtl_buffer offset:offset atIndex:slot];
            }
            break;
        case ShaderStage::Compute:
            if (kind == Bound::Offset) {
                [c_enc setBufferOffset:offset atIndex:slot];
            } else {
                [c_enc setBuffer:mtl_buffer offset:offset atIndex:slot];
            }
            break;
        }
    }

    std::vector<const void*>& ActiveComputeReadUse() noexcept {
        return active_compute == compute ? pending_in_order_compute_read_use
                                         : pending_kernel_compute_read_use;
    }

    void FlushStageBindingsImpl(ShaderStage stage) {
        const bool allow_batch = range_binds_enabled && recorder == nullptr;
        if (stage == ShaderStage::Compute) {
            const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = active_compute;
            StageBindings* const bound = active_bound;
            if (c_enc == nil || bound == nullptr) {
                return;
            }
            std::vector<const void*>& read_use = ActiveComputeReadUse();
            if (!read_use.empty()) {
                const size_t n = read_use.size();
                if (n >= 2 && allow_batch) {
                    const auto* const ptr =
                        reinterpret_cast<const __unsafe_unretained id<MTLResource>*>(
                            read_use.data());
                    [c_enc useResources:ptr count:n usage:MTLResourceUsageRead];
                    ++residency_batch_calls;
                } else {
                    for (const void* raw : read_use) {
                        [c_enc useResource:(__bridge __unsafe_unretained id<MTLResource>)raw
                                     usage:MTLResourceUsageRead];
                    }
                }
                read_use.clear();
            }
            if (!bound->HasDirty()) {
                return;
            }
            u32 bmask = bound->dirty_buffers;
            const u32 boff = bound->dirty_offset_only;
            bound->dirty_buffers = 0;
            bound->dirty_offset_only = 0;
            while (bmask != 0) {
                const u32 start = static_cast<u32>(std::countr_zero(bmask));
                const u32 len = static_cast<u32>(std::countr_one(bmask >> start));
                bmask &= ~(((1U << len) - 1U) << start);
                if (len >= 2 && allow_batch) {
                    const auto* const bufs =
                        reinterpret_cast<const __unsafe_unretained id<MTLBuffer>*>(
                            &bound->buffers[start]);
                    [c_enc setBuffers:bufs
                              offsets:&bound->offsets[start]
                            withRange:NSMakeRange(start, len)];
                    ++range_bind_calls;
                } else {
                    for (u32 i = 0; i < len; ++i) {
                        const u32 s = start + i;
                        if (((boff >> s) & 1U) != 0) {
                            [c_enc setBufferOffset:bound->offsets[s] atIndex:s];
                        } else {
                            [c_enc setBuffer:UnretainedBuffer(bound->buffers[s])
                                      offset:bound->offsets[s]
                                     atIndex:s];
                        }
                    }
                }
            }
            u64 tlo = bound->dirty_textures[0];
            u64 thi = bound->dirty_textures[1];
            bound->dirty_textures = {};
            while (tlo != 0 || thi != 0) {
                u32 start = 0;
                u32 len = 0;
                if (tlo != 0) {
                    start = static_cast<u32>(std::countr_zero(tlo));
                    len = static_cast<u32>(std::countr_one(tlo >> start));
                    tlo &= (len == 64U) ? 0ULL : ~(((1ULL << len) - 1ULL) << start);
                    if (start + len == 64U && thi != 0) {
                        const u32 hi_len =
                            (std::min)(static_cast<u32>(std::countr_one(thi)), TextureSlots - 64U);
                        len += hi_len;
                        thi &= (hi_len == 64U) ? 0ULL : ~((1ULL << hi_len) - 1ULL);
                    }
                } else {
                    const u32 hi_start = static_cast<u32>(std::countr_zero(thi));
                    start = 64U + hi_start;
                    if (start >= TextureSlots) {
                        break;
                    }
                    len = (std::min)(static_cast<u32>(std::countr_one(thi >> hi_start)),
                                     TextureSlots - start);
                    thi &= (len == 64U) ? 0ULL : ~(((1ULL << len) - 1ULL) << hi_start);
                }
                const u32 end = start + len;
                if (len >= 2 && allow_batch) {
                    const auto* const texs =
                        reinterpret_cast<const __unsafe_unretained id<MTLTexture>*>(
                            &bound->textures[start]);
                    [c_enc setTextures:texs withRange:NSMakeRange(start, len)];
                    ++range_bind_calls;
                } else {
                    for (u32 s = start; s < end; ++s) {
                        [c_enc setTexture:UnretainedTexture(bound->textures[s]) atIndex:s];
                    }
                }
            }
            u32 smask = bound->dirty_samplers;
            bound->dirty_samplers = 0;
            while (smask != 0) {
                const u32 start = static_cast<u32>(std::countr_zero(smask));
                const u32 len = static_cast<u32>(std::countr_one(smask >> start));
                smask &= ~(((1U << len) - 1U) << start);
                if (len >= 2 && allow_batch) {
                    const auto* const smps =
                        reinterpret_cast<const __unsafe_unretained id<MTLSamplerState>*>(
                            &bound->samplers[start]);
                    [c_enc setSamplerStates:smps withRange:NSMakeRange(start, len)];
                    ++range_bind_calls;
                } else {
                    for (u32 i = 0; i < len; ++i) {
                        const u32 s = start + i;
                        [c_enc setSamplerState:UnretainedSampler(bound->samplers[s]) atIndex:s];
                    }
                }
            }
            return;
        }

        const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = render;
        size_t stage_idx = 0;
        StageBindings* bound = nullptr;
        MTLRenderStages mtl_stage = MTLRenderStageVertex;
        switch (stage) {
        case ShaderStage::Vertex:
            stage_idx = 0;
            bound = &vertex_bound;
            mtl_stage = MTLRenderStageVertex;
            break;
        case ShaderStage::Fragment:
            stage_idx = 1;
            bound = &fragment_bound;
            mtl_stage = MTLRenderStageFragment;
            break;
        case ShaderStage::Object:
            stage_idx = 2;
            bound = &object_bound;
            mtl_stage = MTLRenderStageObject;
            break;
        case ShaderStage::Mesh:
            stage_idx = 3;
            bound = &mesh_bound;
            mtl_stage = MTLRenderStageMesh;
            break;
        case ShaderStage::Compute:
            return;
        }
        std::vector<const void*>& read_use = pending_render_read_use[stage_idx];
        if (r_enc == nil) {
            read_use.clear();
            return;
        }
        if (!read_use.empty()) {
            const size_t n = read_use.size();
            if (n >= 2 && allow_batch) {
                const auto* const ptr =
                    reinterpret_cast<const __unsafe_unretained id<MTLResource>*>(read_use.data());
                [r_enc useResources:ptr count:n usage:MTLResourceUsageRead stages:mtl_stage];
                ++residency_batch_calls;
            } else {
                for (const void* raw : read_use) {
                    [r_enc useResource:(__bridge __unsafe_unretained id<MTLResource>)raw
                                 usage:MTLResourceUsageRead
                                stages:mtl_stage];
                }
            }
            read_use.clear();
        }
        if (!bound->HasDirty()) {
            return;
        }
        u32 bmask = bound->dirty_buffers;
        const u32 boff = bound->dirty_offset_only;
        bound->dirty_buffers = 0;
        bound->dirty_offset_only = 0;
        while (bmask != 0) {
            const u32 start = static_cast<u32>(std::countr_zero(bmask));
            const u32 len = static_cast<u32>(std::countr_one(bmask >> start));
            bmask &= ~(((1U << len) - 1U) << start);
            if (len >= 2 && allow_batch) {
                const auto* const bufs =
                    reinterpret_cast<const __unsafe_unretained id<MTLBuffer>*>(
                        &bound->buffers[start]);
                const NSUInteger* const offs = &bound->offsets[start];
                const NSRange range = NSMakeRange(start, len);
                switch (stage) {
                case ShaderStage::Vertex:
                    [r_enc setVertexBuffers:bufs offsets:offs withRange:range];
                    break;
                case ShaderStage::Fragment:
                    [r_enc setFragmentBuffers:bufs offsets:offs withRange:range];
                    break;
                case ShaderStage::Object:
                    [r_enc setObjectBuffers:bufs offsets:offs withRange:range];
                    break;
                case ShaderStage::Mesh:
                    [r_enc setMeshBuffers:bufs offsets:offs withRange:range];
                    break;
                case ShaderStage::Compute:
                    break;
                }
                ++range_bind_calls;
            } else {
                for (u32 i = 0; i < len; ++i) {
                    const u32 s = start + i;
                    const bool is_off = ((boff >> s) & 1U) != 0;
                    const NSUInteger off = bound->offsets[s];
                    const id<MTLBuffer> __unsafe_unretained buf =
                        UnretainedBuffer(bound->buffers[s]);
                    switch (stage) {
                    case ShaderStage::Vertex:
                        if (is_off) {
                            [r_enc setVertexBufferOffset:off atIndex:s];
                        } else {
                            [r_enc setVertexBuffer:buf offset:off atIndex:s];
                        }
                        break;
                    case ShaderStage::Fragment:
                        if (is_off) {
                            [r_enc setFragmentBufferOffset:off atIndex:s];
                        } else {
                            [r_enc setFragmentBuffer:buf offset:off atIndex:s];
                        }
                        break;
                    case ShaderStage::Object:
                        if (is_off) {
                            [r_enc setObjectBufferOffset:off atIndex:s];
                        } else {
                            [r_enc setObjectBuffer:buf offset:off atIndex:s];
                        }
                        break;
                    case ShaderStage::Mesh:
                        if (is_off) {
                            [r_enc setMeshBufferOffset:off atIndex:s];
                        } else {
                            [r_enc setMeshBuffer:buf offset:off atIndex:s];
                        }
                        break;
                    case ShaderStage::Compute:
                        break;
                    }
                }
            }
        }
        u64 tlo = bound->dirty_textures[0];
        u64 thi = bound->dirty_textures[1];
        bound->dirty_textures = {};
        while (tlo != 0 || thi != 0) {
            u32 start = 0;
            u32 len = 0;
            if (tlo != 0) {
                start = static_cast<u32>(std::countr_zero(tlo));
                len = static_cast<u32>(std::countr_one(tlo >> start));
                tlo &= (len == 64U) ? 0ULL : ~(((1ULL << len) - 1ULL) << start);
                if (start + len == 64U && thi != 0) {
                    const u32 hi_len =
                        (std::min)(static_cast<u32>(std::countr_one(thi)), TextureSlots - 64U);
                    len += hi_len;
                    thi &= (hi_len == 64U) ? 0ULL : ~((1ULL << hi_len) - 1ULL);
                }
            } else {
                const u32 hi_start = static_cast<u32>(std::countr_zero(thi));
                start = 64U + hi_start;
                if (start >= TextureSlots) {
                    break;
                }
                len = (std::min)(static_cast<u32>(std::countr_one(thi >> hi_start)),
                                 TextureSlots - start);
                thi &= (len == 64U) ? 0ULL : ~(((1ULL << len) - 1ULL) << hi_start);
            }
            const u32 end = start + len;
            if (len >= 2 && allow_batch) {
                const auto* const texs =
                    reinterpret_cast<const __unsafe_unretained id<MTLTexture>*>(
                        &bound->textures[start]);
                const NSRange range = NSMakeRange(start, len);
                switch (stage) {
                case ShaderStage::Vertex:
                    [r_enc setVertexTextures:texs withRange:range];
                    break;
                case ShaderStage::Fragment:
                    [r_enc setFragmentTextures:texs withRange:range];
                    break;
                case ShaderStage::Object:
                    [r_enc setObjectTextures:texs withRange:range];
                    break;
                case ShaderStage::Mesh:
                    [r_enc setMeshTextures:texs withRange:range];
                    break;
                case ShaderStage::Compute:
                    break;
                }
                ++range_bind_calls;
            } else {
                for (u32 s = start; s < end; ++s) {
                    const id<MTLTexture> __unsafe_unretained tex =
                        UnretainedTexture(bound->textures[s]);
                    switch (stage) {
                    case ShaderStage::Vertex:
                        [r_enc setVertexTexture:tex atIndex:s];
                        break;
                    case ShaderStage::Fragment:
                        [r_enc setFragmentTexture:tex atIndex:s];
                        break;
                    case ShaderStage::Object:
                        [r_enc setObjectTexture:tex atIndex:s];
                        break;
                    case ShaderStage::Mesh:
                        [r_enc setMeshTexture:tex atIndex:s];
                        break;
                    case ShaderStage::Compute:
                        break;
                    }
                }
            }
        }
        u32 smask = bound->dirty_samplers;
        bound->dirty_samplers = 0;
        while (smask != 0) {
            const u32 start = static_cast<u32>(std::countr_zero(smask));
            const u32 len = static_cast<u32>(std::countr_one(smask >> start));
            smask &= ~(((1U << len) - 1U) << start);
            if (len >= 2 && allow_batch) {
                const auto* const smps =
                    reinterpret_cast<const __unsafe_unretained id<MTLSamplerState>*>(
                        &bound->samplers[start]);
                const NSRange range = NSMakeRange(start, len);
                switch (stage) {
                case ShaderStage::Vertex:
                    [r_enc setVertexSamplerStates:smps withRange:range];
                    break;
                case ShaderStage::Fragment:
                    [r_enc setFragmentSamplerStates:smps withRange:range];
                    break;
                case ShaderStage::Object:
                    [r_enc setObjectSamplerStates:smps withRange:range];
                    break;
                case ShaderStage::Mesh:
                    [r_enc setMeshSamplerStates:smps withRange:range];
                    break;
                case ShaderStage::Compute:
                    break;
                }
                ++range_bind_calls;
            } else {
                for (u32 i = 0; i < len; ++i) {
                    const u32 s = start + i;
                    const id<MTLSamplerState> __unsafe_unretained smp =
                        UnretainedSampler(bound->samplers[s]);
                    switch (stage) {
                    case ShaderStage::Vertex:
                        [r_enc setVertexSamplerState:smp atIndex:s];
                        break;
                    case ShaderStage::Fragment:
                        [r_enc setFragmentSamplerState:smp atIndex:s];
                        break;
                    case ShaderStage::Object:
                        [r_enc setObjectSamplerState:smp atIndex:s];
                        break;
                    case ShaderStage::Mesh:
                        [r_enc setMeshSamplerState:smp atIndex:s];
                        break;
                    case ShaderStage::Compute:
                        break;
                    }
                }
            }
        }
    }

    void FlushRenderBindings() {
        if (vertex_bound.HasDirty() || !pending_render_read_use[0].empty()) {
            FlushStageBindingsImpl(ShaderStage::Vertex);
        }
        if (fragment_bound.HasDirty() || !pending_render_read_use[1].empty()) {
            FlushStageBindingsImpl(ShaderStage::Fragment);
        }
        if (object_bound.HasDirty() || !pending_render_read_use[2].empty()) {
            FlushStageBindingsImpl(ShaderStage::Object);
        }
        if (mesh_bound.HasDirty() || !pending_render_read_use[3].empty()) {
            FlushStageBindingsImpl(ShaderStage::Mesh);
        }
    }

    void FlushComputeBindings() {
        if ((active_bound != nullptr && active_bound->HasDirty()) ||
            !ActiveComputeReadUse().empty()) {
            FlushStageBindingsImpl(ShaderStage::Compute);
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
        if (deferred_color == 0 && cleared_color_open == 0 && !deferred_depth &&
            !deferred_stencil && !cleared_depth_open && !cleared_stencil_open) {
            kill_color = 0;
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
        const u8 open_color = static_cast<u8>(deferred_color | cleared_color_open);
        for (u32 index = 0; index < 8; ++index) {
            if ((open_color >> index) & 1U) {
                const bool cleared_open = ((cleared_color_open >> index) & 1U) != 0;
                const bool killed = ((kill_color >> index) & 1U) != 0;
                [render setColorStoreAction:action((cleared_open || ((color >> index) & 1U)) &&
                                                   !killed)
                                    atIndex:index];
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
        cleared_color_open = 0;
        deferred_depth = false;
        deferred_stencil = false;
        cleared_depth_open = false;
        cleared_stencil_open = false;
        kill_color = 0;
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
            MTLRenderPassDescriptor* const pass = helper_pass_desc.Acquire();
            MTLRenderPassColorAttachmentDescriptor* const color0 = helper_pass_desc.Color(0);
            ApplyAttachment(color0, Attachment{.texture = texture, .level = 0, .slice = layer});
            color0.loadAction = MTLLoadActionDontCare;
            color0.storeAction = MTLStoreActionStore;
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

    bool NeedsSampleRestoreRoot(const void* root) const noexcept {
        if (shadows.empty() || root == nullptr) {
            return false;
        }
        const auto it = shadows.find(const_cast<void*>(root));
        return it != shadows.end() && it->second.dropped;
    }

    void RestoreSamplesIfDropped(const void* texture, const void* known_root = nullptr) {
        if (shadows.empty() || texture == nullptr) {
            return;
        }
        const void* const root = known_root != nullptr ? known_root : CachedRootTexture(texture);
        if (const auto it = shadows.find(const_cast<void*>(root != nullptr ? root : texture));
            it != shadows.end() && it->second.dropped) {
            RestoreSamples(it->first, it->second);
        }
    }

    /// Something reads the samples of `texture`.
    void SamplesNeeded(const void* texture) {
        const void* const raw_root = NoteTextureTouched(texture);
        RestoreSamplesIfDropped(texture, raw_root);
    }

    void WrittenRoot(const void* texture, const void* known_root = nullptr) {
        if (shadows.empty() || texture == nullptr) {
            return;
        }
        const void* const root = known_root != nullptr ? known_root : CachedRootTexture(texture);
        if (const auto it = shadows.find(const_cast<void*>(root != nullptr ? root : texture));
            it != shadows.end()) {
            if (it->second.dropped) {
                RestoreSamples(it->first, it->second);
            }
            it->second.valid = false;
        }
    }

    /// Something writes part or all of the samples of `texture`: the resolve no longer holds.
    void Written(const void* texture) {
        WrittenRoot(texture);
    }

    void EndEncoder(PassEnd why = PassEnd::Other) {
        want_in_order_compute = false;
        if (blit != nil) {
            [blit endEncoding];
            blit = nil;
        }
        if (blit_render != nil) {
            [blit_render endEncoding];
            blit_render = nil;
            blit_dst = {};
            blit_dst_format = 0;
        }
        if (render != nil) {
            for (auto& vec : pending_render_read_use) {
                vec.clear();
            }
            pass_used_resources.clear();
            ++pass_ends[static_cast<size_t>(why)];
            FinishStores();
            [render endEncoding];
            render = nil;
            in_render_pass.store(false, std::memory_order_relaxed);
        }
        if (pass_open) {
            pass_open = false;
            pass_records.back().draws = draws - pass_records.back().draws_begin;
        }
        if (compute != nil) {
            if (active_compute == compute) {
                active_compute = nil;
                active_bound = nullptr;
                active_hazards = nullptr;
                compute_clobbered_by_copy = false;
            }
            pending_in_order_compute_read_use.clear();
            compute_hazards.Reset();
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
        ReusablePassDescriptor* const rpd =
            pass == draw_pass_desc.pass     ? &draw_pass_desc
            : pass == helper_pass_desc.pass ? &helper_pass_desc
                                            : nullptr;
        if (rpd != nullptr) {
            const bool is_draw_pass = pass == draw_pass_desc.pass;
            u8 colors = rpd->dirty_colors;
            while (colors != 0) {
                const u32 i = static_cast<u32>(std::countr_zero(colors));
                colors &= colors - 1U;
                MTLRenderPassColorAttachmentDescriptor* const __unsafe_unretained c = rpd->colors[i];
                if (const id<MTLTexture> __unsafe_unretained tex = c.texture; tex != nil) {
                    if (!is_draw_pass || c.loadAction == MTLLoadActionClear ||
                        c.storeAction != MTLStoreActionUnknown) {
                        NoteTextureWrite((__bridge const void*)tex);
                    } else {
                        NoteTextureTouched((__bridge const void*)tex);
                    }
                }
                if (const id<MTLTexture> __unsafe_unretained res = c.resolveTexture; res != nil) {
                    NoteTextureWrite((__bridge const void*)res);
                }
            }
            if (rpd->dirty_depth) {
                if (const id<MTLTexture> __unsafe_unretained tex = rpd->depth.texture; tex != nil) {
                    if (!is_draw_pass || rpd->depth.loadAction == MTLLoadActionClear ||
                        rpd->depth.storeAction != MTLStoreActionUnknown) {
                        NoteTextureWrite((__bridge const void*)tex);
                    } else {
                        NoteTextureTouched((__bridge const void*)tex);
                    }
                }
                if (const id<MTLTexture> __unsafe_unretained res = rpd->depth.resolveTexture;
                    res != nil) {
                    NoteTextureWrite((__bridge const void*)res);
                }
            }
            if (rpd->dirty_stencil) {
                if (const id<MTLTexture> __unsafe_unretained tex = rpd->stencil.texture;
                    tex != nil) {
                    if (!is_draw_pass || rpd->stencil.loadAction == MTLLoadActionClear ||
                        rpd->stencil.storeAction != MTLStoreActionUnknown) {
                        NoteTextureWrite((__bridge const void*)tex);
                    } else {
                        NoteTextureTouched((__bridge const void*)tex);
                    }
                }
                if (const id<MTLTexture> __unsafe_unretained res = rpd->stencil.resolveTexture;
                    res != nil) {
                    NoteTextureWrite((__bridge const void*)res);
                }
            }
        } else {
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
                const id<MTLTexture> __unsafe_unretained color =
                    rpd != nullptr ? rpd->colors[0].texture : pass.colorAttachments[0].texture;
                const id<MTLTexture> __unsafe_unretained depth =
                    rpd != nullptr ? rpd->depth.texture : pass.depthAttachment.texture;
                u32 pass_width = static_cast<u32>(pass.renderTargetWidth);
                u32 pass_height = static_cast<u32>(pass.renderTargetHeight);
                if (pass_width == 0 || pass_height == 0) {
                    if (color != nil) {
                        const u32 lvl = static_cast<u32>(
                            rpd != nullptr ? rpd->colors[0].level : pass.colorAttachments[0].level);
                        std::tie(pass_width, pass_height) = MipExtent(color, lvl);
                    } else if (depth != nil) {
                        const u32 lvl = static_cast<u32>(
                            rpd != nullptr ? rpd->depth.level : pass.depthAttachment.level);
                        std::tie(pass_width, pass_height) = MipExtent(depth, lvl);
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
                const id<MTLTexture> __unsafe_unretained c0 =
                    rpd != nullptr ? rpd->colors[0].texture : pass.colorAttachments[0].texture;
                const id<MTLTexture> __unsafe_unretained d0 =
                    rpd != nullptr ? rpd->depth.texture : pass.depthAttachment.texture;
                LOG_ERROR(Render_Metal, "Metal: render pass refused ({}x{}, colour 0 {}, depth {})",
                          pass.renderTargetWidth, pass.renderTargetHeight,
                          c0 != nil ? static_cast<u32>(c0.pixelFormat) : 0U,
                          d0 != nil ? static_cast<u32>(d0.pixelFormat) : 0U);
            }
            if (rpd != nullptr) {
                rpd->Reset();
            }
            return nil;
        }
        if (rpd != nullptr) {
            rpd->Reset();
        }
        encoder.label = static_labels_enabled
                            ? @"Pass"
                            : [NSString stringWithFormat:@"Pass %llu", render_pass_serial + 1];
        return encoder;
    }

    MTLRenderPassDescriptor* BuildClearDescriptor(const ClearPass& clear) {
        MTLRenderPassDescriptor* const pass = helper_pass_desc.Acquire();
        bool any = false;
        for (size_t index = 0; index < clear.color.size(); ++index) {
            if (clear.color[index].texture == nullptr || !clear.clear_color[index]) {
                continue;
            }
            MTLRenderPassColorAttachmentDescriptor* const color = helper_pass_desc.Color(index);
            ApplyAttachment(color, clear.color[index]);
            if (const auto shadow = shadows.find(clear.color[index].texture);
                shadow != shadows.end()) {
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
            MTLRenderPassDepthAttachmentDescriptor* const depth = helper_pass_desc.Depth();
            ApplyAttachment(depth, clear.depth);
            depth.loadAction = MTLLoadActionClear;
            depth.storeAction = MTLStoreActionStore;
            depth.clearDepth = clear.depth_value;
            any = true;
        }
        if (clear.stencil.texture != nullptr && clear.clear_stencil) {
            MTLRenderPassStencilAttachmentDescriptor* const stencil = helper_pass_desc.Stencil();
            ApplyAttachment(stencil, clear.stencil);
            stencil.loadAction = MTLLoadActionClear;
            stencil.storeAction = MTLStoreActionStore;
            stencil.clearStencil = clear.stencil_value;
            any = true;
        }
        if (!any) {
            helper_pass_desc.Reset();
            return nil;
        }
        pass.renderTargetWidth = std::clamp(clear.width, 1U, MaxRenderTargetSize);
        pass.renderTargetHeight = std::clamp(clear.height, 1U, MaxRenderTargetSize);
        return pass;
    }

    /// A render pass that only clears.
    void RecordClear(const ClearPass& clear) {
        @autoreleasepool {
            MTLRenderPassDescriptor* const pass = BuildClearDescriptor(clear);
            if (pass != nil) {
                [RenderEncoder(pass, PassEnd::Clear) endEncoding];
            }
        }
    }

    static constexpr size_t MaxPendingClears = 32;

    struct PendingClearEntry {
        ClearPass clear{};
        std::array<id<MTLTexture>, 10> textures{};
        std::array<SubresourceKey, 10> subs{};
        u64 roots_bloom{};
        bool blocked_ahead{};

        bool HasAny() const noexcept {
            return clear.clear_depth || clear.clear_stencil ||
                   std::ranges::any_of(clear.clear_color, [](bool v) { return v; });
        }

        void Refresh() {
            u64 bloom = 0;
            for (size_t i = 0; i < clear.color.size(); ++i) {
                if (clear.clear_color[i] && clear.color[i].texture != nullptr) {
                    textures[i] = AsTexture(clear.color[i].texture);
                    subs[i] = SubresourceOf(clear.color[i]);
                    if (subs[i].root != nullptr) {
                        bloom |= PtrBloomBit(subs[i].root);
                    }
                } else {
                    textures[i] = nil;
                    subs[i] = {};
                }
            }
            if (clear.clear_depth && clear.depth.texture != nullptr) {
                textures[8] = AsTexture(clear.depth.texture);
                subs[8] = SubresourceOf(clear.depth);
                if (subs[8].root != nullptr) {
                    bloom |= PtrBloomBit(subs[8].root);
                }
            } else {
                textures[8] = nil;
                subs[8] = {};
            }
            if (clear.clear_stencil && clear.stencil.texture != nullptr) {
                textures[9] = AsTexture(clear.stencil.texture);
                subs[9] = SubresourceOf(clear.stencil);
                if (subs[9].root != nullptr) {
                    bloom |= PtrBloomBit(subs[9].root);
                }
            } else {
                textures[9] = nil;
                subs[9] = {};
            }
            roots_bloom = bloom;
        }

        bool TouchesRoot(const void* want_root) const noexcept {
            if (want_root == nullptr || (roots_bloom & PtrBloomBit(want_root)) == 0) {
                return false;
            }
            for (const SubresourceKey& sub : subs) {
                if (sub.root == want_root) {
                    return true;
                }
            }
            return false;
        }

        bool TouchesSubresource(const SubresourceKey& want) const noexcept {
            if (want.root == nullptr || (roots_bloom & PtrBloomBit(want.root)) == 0) {
                return false;
            }
            for (const SubresourceKey& sub : subs) {
                if (sub == want) {
                    return true;
                }
            }
            return false;
        }

        bool SharesAnySubresourceWith(const std::array<SubresourceKey, 10>& other) const noexcept {
            for (const SubresourceKey& s : other) {
                if (s.root != nullptr && TouchesSubresource(s)) {
                    return true;
                }
            }
            return false;
        }
    };

    void RebuildPendingClearBloom() noexcept {
        u64 bloom = 0;
        for (const PendingClearEntry& entry : pending_clears) {
            bloom |= entry.roots_bloom;
        }
        pending_clear_roots_bloom = bloom;
    }

    static std::array<SubresourceKey, 9> TargetSubresourcesOf(const RenderTargetSet& targets) {
        std::array<SubresourceKey, 9> subs{};
        for (size_t i = 0; i < targets.color.size(); ++i) {
            subs[i] = SubresourceOf(targets.color[i]);
        }
        subs[8] = (targets.has_depth || targets.has_stencil) ? SubresourceOf(targets.depth)
                                                             : SubresourceKey{};
        return subs;
    }

    static bool SubresourceMatchesTargets(const SubresourceKey& sub,
                                          const std::array<SubresourceKey, 9>& target_subs,
                                          u32 layers) noexcept {
        if (sub.root == nullptr) {
            return false;
        }
        for (const SubresourceKey& tsub : target_subs) {
            if (tsub.root == sub.root && tsub.level == sub.level &&
                tsub.depth_plane == sub.depth_plane && sub.slice >= tsub.slice &&
                sub.slice < tsub.slice + layers) {
                return true;
            }
        }
        return false;
    }

    bool CanFlushClearAheadForRoot(const void* want_root) const noexcept {
        if (want_root == nullptr || !ahead_enabled || !merge_ahead_enabled || ahead_blocked ||
            segment_opaque) {
            return false;
        }
        if ((pending_clear_roots_bloom & PtrBloomBit(want_root)) == 0) {
            return true;
        }
        for (const PendingClearEntry& entry : pending_clears) {
            if (entry.TouchesRoot(want_root) && entry.blocked_ahead) {
                return false;
            }
        }
        return true;
    }

    void EraseOrRefreshClears(auto&& strip_entry) {
        std::erase_if(pending_clears, [&](PendingClearEntry& entry) {
            if (!strip_entry(entry)) {
                return false;
            }
            if (entry.HasAny()) {
                entry.Refresh();
                return false;
            }
            return true;
        });
        RebuildPendingClearBloom();
    }

    void DropSubresourceClear(const Attachment& att, bool drop_color, bool drop_depth,
                              bool drop_stencil) {
        if (pending_clears.empty() || att.texture == nullptr) {
            return;
        }
        const SubresourceKey want = SubresourceOf(att);
        if (want.root == nullptr) {
            return;
        }
        if (drop_color && !shadows.empty()) {
            if (const auto it = shadows.find(const_cast<void*>(want.root)); it != shadows.end()) {
                it->second.valid = false;
                it->second.dropped = false;
            }
        }
        EraseOrRefreshClears([&](PendingClearEntry& entry) {
            bool changed = false;
            if (drop_color) {
                for (size_t i = 0; i < entry.clear.color.size(); ++i) {
                    if (entry.clear.clear_color[i] && entry.subs[i] == want) {
                        entry.clear.color[i] = {};
                        entry.clear.clear_color[i] = false;
                        changed = true;
                    }
                }
            }
            if (drop_depth && entry.clear.clear_depth && entry.subs[8] == want) {
                entry.clear.depth = {};
                entry.clear.clear_depth = false;
                changed = true;
            }
            if (drop_stencil && entry.clear.clear_stencil && entry.subs[9] == want) {
                entry.clear.stencil = {};
                entry.clear.clear_stencil = false;
                changed = true;
            }
            return changed;
        });
    }

    void DropMipClear(TextureHandle dst, u32 level, u32 slice, Offset3D origin, Extent3D extent) {
        if (pending_clears.empty() || dst == nullptr || origin.x != 0 || origin.y != 0 ||
            origin.z != 0) {
            return;
        }
        const id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(dst);
        if (tex == nil) {
            return;
        }
        const auto [mip_w, mip_h] = MipExtent(tex, level);
        const u32 mip_d = (std::max)(static_cast<u32>(tex.depth) >> level, 1U);
        if (extent.width < mip_w || extent.height < mip_h || extent.depth < mip_d) {
            return;
        }
        if (mip_d == 1) {
            DropSubresourceClear(Attachment{.texture = dst, .level = level, .slice = slice}, true,
                                 true, true);
            return;
        }
        const SubresourceKey base = SubresourceOf(
            Attachment{.texture = dst, .level = level, .slice = slice, .depth_plane = 0});
        if (base.root == nullptr) {
            return;
        }
        const auto matches_3d = [&](const SubresourceKey& sub) noexcept {
            return sub.root == base.root && sub.level == base.level && sub.slice == base.slice &&
                   sub.depth_plane < extent.depth;
        };
        EraseOrRefreshClears([&](PendingClearEntry& entry) {
            bool changed = false;
            for (size_t i = 0; i < entry.clear.color.size(); ++i) {
                if (entry.clear.clear_color[i] && matches_3d(entry.subs[i])) {
                    entry.clear.color[i] = {};
                    entry.clear.clear_color[i] = false;
                    changed = true;
                }
            }
            if (entry.clear.clear_depth && matches_3d(entry.subs[8])) {
                entry.clear.depth = {};
                entry.clear.clear_depth = false;
                changed = true;
            }
            if (entry.clear.clear_stencil && matches_3d(entry.subs[9])) {
                entry.clear.stencil = {};
                entry.clear.clear_stencil = false;
                changed = true;
            }
            return changed;
        });
    }

    void FlushClearForSubresource(const SubresourceKey& want) {
        if (pending_clears.empty() || want.root == nullptr ||
            (pending_clear_roots_bloom & PtrBloomBit(want.root)) == 0) {
            return;
        }
        std::vector<PendingClearEntry> matching;
        std::erase_if(pending_clears, [&](PendingClearEntry& entry) {
            if (entry.TouchesSubresource(want)) {
                matching.push_back(std::move(entry));
                return true;
            }
            return false;
        });
        if (!matching.empty()) {
            RebuildPendingClearBloom();
            for (const PendingClearEntry& entry : matching) {
                RecordClear(entry.clear);
            }
        }
    }

    void FlushClearAheadForSubresource(const SubresourceKey& want) {
        if (pending_clears.empty() || want.root == nullptr ||
            (pending_clear_roots_bloom & PtrBloomBit(want.root)) == 0) {
            return;
        }
        std::vector<PendingClearEntry> matching;
        std::erase_if(pending_clears, [&](PendingClearEntry& entry) {
            if (entry.TouchesSubresource(want)) {
                matching.push_back(std::move(entry));
                return true;
            }
            return false;
        });
        if (!matching.empty()) {
            RebuildPendingClearBloom();
            for (const PendingClearEntry& entry : matching) {
                RecordClearOnAheadBuffer(entry.clear);
            }
        }
    }

    bool CanDeferEntryAcross(const PendingClearEntry& entry,
                             const std::array<SubresourceKey, 9>& target_subs,
                             u32 layers) const noexcept {
        for (const SubresourceKey& sub : entry.subs) {
            if (sub.root != nullptr && SubresourceMatchesTargets(sub, target_subs, layers)) {
                return false;
            }
        }
        return true;
    }

    bool EntryCanFoldOrDefer(const PendingClearEntry& entry, const RenderTargetSet& targets,
                             const std::array<SubresourceKey, 9>& target_subs,
                             u32 eff_layers) const {
        return CanFoldClearInto(entry.clear, targets) ||
               CanDeferEntryAcross(entry, target_subs, eff_layers);
    }

    bool LayeredAttachmentCanFold(const Attachment& base_att, u32 eff_layers, u32 target_w,
                                  u32 target_h, int aspect) const noexcept {
        const id<MTLTexture> __unsafe_unretained base_tex = UnretainedTexture(base_att.texture);
        const SubresourceKey base_sub = SubresourceOf(base_att);
        if (base_tex == nil || base_sub.root == nullptr) {
            return false;
        }
        std::array<f64, 4> cval{};
        f64 dval = 0.0;
        u32 sval = 0;
        const auto slice_fits = [&](const ClearPass& c, const SubresourceKey& have,
                                    const Attachment& att, const SubresourceKey& want) noexcept {
            const id<MTLTexture> __unsafe_unretained t = UnretainedTexture(att.texture);
            return have == want && t != nil && t.pixelFormat == base_tex.pixelFormat &&
                   t.sampleCount == base_tex.sampleCount &&
                   ClearExtentFitsAttachment(c, att, target_w, target_h);
        };
        for (u32 s = 0; s < eff_layers; ++s) {
            const SubresourceKey want{base_sub.root, base_sub.level, base_sub.slice + s,
                                      base_sub.depth_plane};
            bool ok = false;
            for (const PendingClearEntry& entry : pending_clears) {
                if (aspect == 0) {
                    for (size_t ci = 0; ci < entry.clear.color.size(); ++ci) {
                        if (entry.clear.clear_color[ci] &&
                            slice_fits(entry.clear, entry.subs[ci], entry.clear.color[ci], want) &&
                            (s == 0 || entry.clear.color_value[ci] == cval)) {
                            cval = entry.clear.color_value[ci];
                            ok = true;
                            break;
                        }
                    }
                } else if (aspect == 1) {
                    if (entry.clear.clear_depth &&
                        slice_fits(entry.clear, entry.subs[8], entry.clear.depth, want) &&
                        (s == 0 || entry.clear.depth_value == dval)) {
                        dval = entry.clear.depth_value;
                        ok = true;
                    }
                } else if (entry.clear.clear_stencil &&
                           slice_fits(entry.clear, entry.subs[9], entry.clear.stencil, want) &&
                           (s == 0 || entry.clear.stencil_value == sval)) {
                    sval = entry.clear.stencil_value;
                    ok = true;
                }
                if (ok) {
                    break;
                }
            }
            if (!ok) {
                return false;
            }
        }
        return true;
    }

    void TryFoldLayeredClears(std::optional<ClearPass>& out_clear, const RenderTargetSet& targets,
                              u32 eff_layers) {
        if (eff_layers <= 1 || pending_clears.empty() || targets.width == 0 ||
            targets.height == 0) {
            return;
        }
        const auto [target_w, target_h] = EffectivePassExtent(targets);
        bool any_modified = false;
        for (size_t rt = 0; rt < targets.color.size(); ++rt) {
            const Attachment& base_att = targets.color[rt];
            if (!LayeredAttachmentCanFold(base_att, eff_layers, target_w, target_h, 0)) {
                continue;
            }
            const SubresourceKey base_sub = SubresourceOf(base_att);
            std::array<f64, 4> color_val{};
            for (u32 s = 0; s < eff_layers; ++s) {
                const SubresourceKey want{base_sub.root, base_sub.level, base_sub.slice + s,
                                          base_sub.depth_plane};
                for (PendingClearEntry& entry : pending_clears) {
                    bool found = false;
                    for (size_t ci = 0; ci < entry.clear.color.size(); ++ci) {
                        if (entry.clear.clear_color[ci] && entry.subs[ci] == want) {
                            if (s == 0) {
                                color_val = entry.clear.color_value[ci];
                            }
                            entry.clear.color[ci] = {};
                            entry.clear.clear_color[ci] = false;
                            found = true;
                            break;
                        }
                    }
                    if (found) {
                        break;
                    }
                }
            }
            if (!out_clear) {
                out_clear.emplace();
            }
            out_clear->color[rt] = base_att;
            out_clear->clear_color[rt] = true;
            out_clear->color_value[rt] = color_val;
            out_clear->width = (std::max)(out_clear->width, target_w);
            out_clear->height = (std::max)(out_clear->height, target_h);
            any_modified = true;
        }
        if (targets.has_depth &&
            LayeredAttachmentCanFold(targets.depth, eff_layers, target_w, target_h, 1)) {
            const SubresourceKey base_sub = SubresourceOf(targets.depth);
            f64 depth_val = 0.0;
            for (u32 s = 0; s < eff_layers; ++s) {
                const SubresourceKey want{base_sub.root, base_sub.level, base_sub.slice + s,
                                          base_sub.depth_plane};
                for (PendingClearEntry& entry : pending_clears) {
                    if (entry.clear.clear_depth && entry.subs[8] == want) {
                        if (s == 0) {
                            depth_val = entry.clear.depth_value;
                        }
                        entry.clear.depth = {};
                        entry.clear.clear_depth = false;
                        break;
                    }
                }
            }
            if (!out_clear) {
                out_clear.emplace();
            }
            out_clear->depth = targets.depth;
            out_clear->clear_depth = true;
            out_clear->depth_value = depth_val;
            out_clear->width = (std::max)(out_clear->width, target_w);
            out_clear->height = (std::max)(out_clear->height, target_h);
            any_modified = true;
        }
        if (targets.has_stencil &&
            LayeredAttachmentCanFold(targets.depth, eff_layers, target_w, target_h, 2)) {
            const SubresourceKey base_sub = SubresourceOf(targets.depth);
            u32 stencil_val = 0;
            for (u32 s = 0; s < eff_layers; ++s) {
                const SubresourceKey want{base_sub.root, base_sub.level, base_sub.slice + s,
                                          base_sub.depth_plane};
                for (PendingClearEntry& entry : pending_clears) {
                    if (entry.clear.clear_stencil && entry.subs[9] == want) {
                        if (s == 0) {
                            stencil_val = entry.clear.stencil_value;
                        }
                        entry.clear.stencil = {};
                        entry.clear.clear_stencil = false;
                        break;
                    }
                }
            }
            if (!out_clear) {
                out_clear.emplace();
            }
            out_clear->stencil = targets.depth;
            out_clear->clear_stencil = true;
            out_clear->stencil_value = stencil_val;
            out_clear->width = (std::max)(out_clear->width, target_w);
            out_clear->height = (std::max)(out_clear->height, target_h);
            any_modified = true;
        }
        if (any_modified) {
            EraseOrRefreshClears([](PendingClearEntry&) { return true; });
        }
    }

    /// Whether every waiting clear can either fold into `targets` or stay deferred across it.
    bool CanFoldPendingClears(const RenderTargetSet& targets) const {
        if (pending_clears.empty()) {
            return true;
        }
        const u32 eff_layers = EffectiveTargetLayers(targets);
        const auto target_subs = TargetSubresourcesOf(targets);
        if (eff_layers <= 1) {
            for (const PendingClearEntry& entry : pending_clears) {
                if (!EntryCanFoldOrDefer(entry, targets, target_subs, eff_layers)) {
                    return false;
                }
            }
            return true;
        }
        const auto [target_w, target_h] = EffectivePassExtent(targets);
        std::array<bool, 8> color_folds{};
        for (size_t rt = 0; rt < targets.color.size(); ++rt) {
            color_folds[rt] =
                LayeredAttachmentCanFold(targets.color[rt], eff_layers, target_w, target_h, 0);
        }
        const bool depth_folds =
            targets.has_depth &&
            LayeredAttachmentCanFold(targets.depth, eff_layers, target_w, target_h, 1);
        const bool stencil_folds =
            targets.has_stencil &&
            LayeredAttachmentCanFold(targets.depth, eff_layers, target_w, target_h, 2);
        const auto in_slice_range = [](const SubresourceKey& sub, const SubresourceKey& base,
                                       u32 layers) noexcept {
            return base.root != nullptr && sub.root == base.root && sub.level == base.level &&
                   sub.depth_plane == base.depth_plane && sub.slice >= base.slice &&
                   sub.slice < base.slice + layers;
        };
        for (const PendingClearEntry& entry : pending_clears) {
            for (size_t ci = 0; ci < entry.clear.color.size(); ++ci) {
                if (!entry.clear.clear_color[ci] ||
                    !SubresourceMatchesTargets(entry.subs[ci], target_subs, eff_layers)) {
                    continue;
                }
                bool covered = false;
                for (size_t rt = 0; rt < targets.color.size(); ++rt) {
                    if (color_folds[rt] &&
                        in_slice_range(entry.subs[ci], target_subs[rt], eff_layers)) {
                        covered = true;
                        break;
                    }
                }
                if (!covered) {
                    return false;
                }
            }
            if (entry.clear.clear_depth &&
                SubresourceMatchesTargets(entry.subs[8], target_subs, eff_layers) && !depth_folds) {
                return false;
            }
            if (entry.clear.clear_stencil &&
                SubresourceMatchesTargets(entry.subs[9], target_subs, eff_layers) &&
                !stencil_folds) {
                return false;
            }
        }
        return true;
    }

    void FlushClearForRoot(const void* want_root) {
        if (pending_clears.empty() || want_root == nullptr ||
            (pending_clear_roots_bloom & PtrBloomBit(want_root)) == 0) {
            return;
        }
        std::vector<PendingClearEntry> matching;
        std::erase_if(pending_clears, [&](PendingClearEntry& entry) {
            if (entry.TouchesRoot(want_root)) {
                matching.push_back(std::move(entry));
                return true;
            }
            return false;
        });
        if (!matching.empty()) {
            RebuildPendingClearBloom();
            for (const PendingClearEntry& entry : matching) {
                RecordClear(entry.clear);
            }
        }
    }

    id<MTLRenderCommandEncoder> AheadRenderEncoder(MTLRenderPassDescriptor* pass,
                                                   NSString* label = @"Ahead Helper",
                                                   size_t kind_idx = 2) {
        if (upload_blit != nil) {
            [upload_blit endEncoding];
            upload_blit = nil;
        }
        if (kernel_compute != nil && active_compute == kernel_compute && kernel_binding) {
            kernel_resume = true;
        }
        EndKernelEncoder();
        const id<MTLRenderCommandEncoder> encoder =
            [AheadBuffer() renderCommandEncoderWithDescriptor:pass];
        helper_pass_desc.Reset();
        if (encoder != nil) {
            encoder.label = label;
            if (command_buffer == nil) {
                ++ahead_encoders;
            }
            ++passes_begun[kind_idx];
            ++encoders;
            ++render_passes;
        }
        return encoder;
    }

    bool CanRunHelperPassAhead(const void* src_root, TextureHandle dst_tex,
                               const void* dst_root) const {
        if (render == nil || !ahead_enabled || !merge_ahead_enabled || ahead_blocked ||
            segment_opaque || !WantsAheadEncoder()) {
            return false;
        }
        if (!clear_narrow_enabled && !pending_clears.empty()) {
            return false;
        }
        if (src_root == nullptr || dst_root == nullptr || src_root == dst_root) {
            return false;
        }
        if (NeedsSampleRestoreRoot(src_root) || segment_written_textures.contains(src_root)) {
            return false;
        }
        if (PendingClearOnRoot(src_root) && !CanFlushClearAheadForRoot(src_root)) {
            return false;
        }
        if (segment_touched_textures.contains(dst_root)) {
            return false;
        }
        if (PendingClearOnRoot(dst_root) && !CanFlushClearAheadForRoot(dst_root)) {
            return false;
        }
        if (!shadows.empty() &&
            (shadows.contains(dst_tex) || shadows.contains(const_cast<void*>(dst_root)))) {
            return false;
        }
        return ComputeHazardState::BackingBufferSpan(src_root).buffer == nullptr &&
               ComputeHazardState::BackingBufferSpan(dst_root).buffer == nullptr;
    }

    void RecordClearOnAheadBuffer(const ClearPass& clear) {
        MTLRenderPassDescriptor* const pass = BuildClearDescriptor(clear);
        if (pass == nil) {
            return;
        }
        if (command_buffer == nil) {
            for (size_t i = 0; i < clear.color.size(); ++i) {
                if (clear.clear_color[i] && clear.color[i].texture != nullptr) {
                    NoteAheadTexture(CachedRootTexture(clear.color[i].texture), true);
                }
            }
            if (clear.clear_depth && clear.depth.texture != nullptr) {
                NoteAheadTexture(CachedRootTexture(clear.depth.texture), true);
            }
            if (clear.clear_stencil && clear.stencil.texture != nullptr) {
                NoteAheadTexture(CachedRootTexture(clear.stencil.texture), true);
            }
        }
        const id<MTLRenderCommandEncoder> encoder = AheadRenderEncoder(pass, @"Ahead Clear", 1);
        if (encoder != nil) {
            [encoder endEncoding];
        }
    }

    void RestoreRenderStageBindings(const std::array<StageBindings, 4>& saved_stages) {
        if (render == nil) {
            return;
        }
        constexpr std::array<ShaderStage, 4> stages{
            ShaderStage::Vertex, ShaderStage::Fragment, ShaderStage::Object, ShaderStage::Mesh};
        const std::array<StageBindings*, 4> dsts{
            &vertex_bound, &fragment_bound, &object_bound, &mesh_bound};
        for (size_t idx = 0; idx < 4; ++idx) {
            const StageBindings& src = saved_stages[idx];
            StageBindings& dst = *dsts[idx];
            const ShaderStage st = stages[idx];
            for (u32 s = 0; s < BufferSlots; ++s) {
                const void* const buf = src.buffers[s];
                if (buf == nullptr || buf == &ClobberedSlot) {
                    continue;
                }
                if (buf == &InlineBytes) {
                    for (const StageBindings::InlineSlot& inl : src.inline_slots) {
                        if (inl.slot == s) {
                            (void)dst.MatchesOrRecordInline(s, inl.bytes.data(), inl.size,
                                                            &InlineBytes);
                            if (st == ShaderStage::Vertex) {
                                [render setVertexBytes:inl.bytes.data() length:inl.size atIndex:s];
                            } else if (st == ShaderStage::Fragment) {
                                [render setFragmentBytes:inl.bytes.data()
                                                  length:inl.size
                                                 atIndex:s];
                            } else if (st == ShaderStage::Object) {
                                [render setObjectBytes:inl.bytes.data() length:inl.size atIndex:s];
                            } else {
                                [render setMeshBytes:inl.bytes.data() length:inl.size atIndex:s];
                            }
                            break;
                        }
                    }
                } else {
                    SetBuffer(st, s, const_cast<void*>(buf), static_cast<u32>(src.offsets[s]));
                }
            }
            for (u32 s = 0; s < TextureSlots; ++s) {
                const void* const tex = src.textures[s];
                if (tex == nullptr || tex == &ClobberedSlot) {
                    continue;
                }
                dst.textures[s] = tex;
                if (range_binds_enabled) {
                    dst.MarkTextureDirty(s);
                } else {
                    const id<MTLTexture> __unsafe_unretained mtl_tex = UnretainedTexture(tex);
                    if (st == ShaderStage::Vertex) {
                        [render setVertexTexture:mtl_tex atIndex:s];
                    } else if (st == ShaderStage::Fragment) {
                        [render setFragmentTexture:mtl_tex atIndex:s];
                    } else if (st == ShaderStage::Object) {
                        [render setObjectTexture:mtl_tex atIndex:s];
                    } else {
                        [render setMeshTexture:mtl_tex atIndex:s];
                    }
                }
            }
            for (u32 s = 0; s < SamplerSlots; ++s) {
                const void* const smp = src.samplers[s];
                if (smp == nullptr || smp == &ClobberedSlot) {
                    continue;
                }
                dst.samplers[s] = smp;
                if (range_binds_enabled) {
                    dst.MarkSamplerDirty(s);
                } else {
                    const id<MTLSamplerState> __unsafe_unretained mtl_smp = UnretainedSampler(smp);
                    if (st == ShaderStage::Vertex) {
                        [render setVertexSamplerState:mtl_smp atIndex:s];
                    } else if (st == ShaderStage::Fragment) {
                        [render setFragmentSamplerState:mtl_smp atIndex:s];
                    } else if (st == ShaderStage::Object) {
                        [render setObjectSamplerState:mtl_smp atIndex:s];
                    } else {
                        [render setMeshSamplerState:mtl_smp atIndex:s];
                    }
                }
            }
        }
        FlushRenderBindings();
    }

    bool MatchesOpenTargets(const RenderTargetSet& targets) noexcept {
        if (render == nil) {
            return false;
        }
        if (render_targets.layers != targets.layers ||
            render_targets.has_depth != targets.has_depth ||
            render_targets.has_stencil != targets.has_stencil) {
            return false;
        }
        for (size_t i = 0; i < render_targets.color.size(); ++i) {
            if (!SameAttachmentSubresource(render_targets.color[i], targets.color[i])) {
                return false;
            }
        }
        if ((render_targets.has_depth || render_targets.has_stencil) &&
            !SameAttachmentSubresource(render_targets.depth, targets.depth)) {
            return false;
        }
        if ((targets.width == render_targets.width ||
             targets.width == render_targets_raw_width) &&
            (targets.height == render_targets.height ||
             targets.height == render_targets_raw_height)) {
            return true;
        }
        if (std::pair{render_targets.width, render_targets.height} ==
            EffectivePassExtent(targets)) {
            render_targets_raw_width = targets.width;
            render_targets_raw_height = targets.height;
            return true;
        }
        return false;
    }

    void BeginRenderingImpl(const RenderTargetSet& targets);
    void SetDrawStateImpl(const DrawState& state);

    void InterruptAndRestoreRenderPass(auto&& record_fn) {
        if (render == nil) {
            record_fn();
            return;
        }
        const RenderTargetSet saved_targets = render_targets;
        const bool had_state = render_state_valid;
        const DrawState saved_state = render_state;
        const s64 saved_vis = visibility_slot;
        const std::array<StageBindings, 4> saved_stages{
            vertex_bound, fragment_bound, object_bound, mesh_bound};
        const std::vector<RenderUseEntry> saved_uses = pass_used_resources;
        record_fn();
        BeginRenderingImpl(saved_targets);
        if (render != nil) {
            if (had_state) {
                SetDrawStateImpl(saved_state);
            }
            pass_used_resources = saved_uses;
            for (const RenderUseEntry& u : saved_uses) {
                [render useResource:(__bridge __unsafe_unretained id<MTLResource>)u.resource
                              usage:u.usage
                             stages:u.stages];
            }
            RestoreRenderStageBindings(saved_stages);
            if (saved_vis >= 0) {
                visibility_slot = saved_vis;
                [render setVisibilityResultMode:MTLVisibilityResultModeCounting
                                         offset:static_cast<NSUInteger>(saved_vis) * sizeof(u64)];
            }
        }
    }

    void FlushDeferredDownloads() {
        if (deferred_downloads.empty()) {
            return;
        }
        std::vector<DeferredDownload> pending = std::move(deferred_downloads);
        deferred_downloads.clear();
        const id<MTLBlitCommandEncoder> __unsafe_unretained enc = BlitEncoder();
        blit_commands += pending.size() - 1;
        for (const DeferredDownload& d : pending) {
            [enc copyFromBuffer:UnretainedBuffer(d.src)
                   sourceOffset:d.src_offset
                       toBuffer:UnretainedBuffer(d.dst)
              destinationOffset:d.dst_offset
                           size:d.size];
        }
    }

    void FlushDeferredDownloadsFor(BufferHandle buffer, u64 offset, u64 size,
                                   bool preserve_render) {
        if (deferred_downloads.empty() || buffer == nullptr || size == 0) {
            return;
        }
        const ComputeHazardState::BufferSpan write_span{
            buffer, offset, ComputeHazardState::SaturatingEnd(offset, size)};
        bool any_overlap = false;
        for (const DeferredDownload& entry : deferred_downloads) {
            const ComputeHazardState::BufferSpan read_span{
                entry.src, entry.src_offset,
                ComputeHazardState::SaturatingEnd(entry.src_offset, entry.size)};
            if (read_span.Overlaps(write_span)) {
                any_overlap = true;
                break;
            }
        }
        if (!any_overlap) {
            return;
        }
        std::vector<DeferredDownload> matching;
        std::erase_if(deferred_downloads, [&](const DeferredDownload& entry) {
            const ComputeHazardState::BufferSpan read_span{
                entry.src, entry.src_offset,
                ComputeHazardState::SaturatingEnd(entry.src_offset, entry.size)};
            if (read_span.Overlaps(write_span)) {
                matching.push_back(entry);
                return true;
            }
            return false;
        });
        if (matching.empty()) {
            return;
        }
        if (compute != nil && render == nil && blit == nil && blit_render == nil) {
            for (const DeferredDownload& d : matching) {
                if (compute_copy_enabled && d.size <= compute_copy_max && d.src != d.dst &&
                    compute != nil && render == nil && blit == nil && blit_render == nil &&
                    CopyByCompute(compute, compute_bound, compute_hazards, &compute_pipeline, d.src,
                                  d.src_offset, d.dst, d.dst_offset, d.size)) {
                    ++compute_copies_in_order;
                } else {
                    const id<MTLBlitCommandEncoder> __unsafe_unretained enc = BlitEncoder();
                    [enc copyFromBuffer:UnretainedBuffer(d.src)
                           sourceOffset:d.src_offset
                               toBuffer:UnretainedBuffer(d.dst)
                      destinationOffset:d.dst_offset
                                   size:d.size];
                }
            }
            return;
        }
        const auto emit_blit = [&] {
            const id<MTLBlitCommandEncoder> __unsafe_unretained enc = BlitEncoder();
            blit_commands += matching.size() - 1;
            for (const DeferredDownload& d : matching) {
                [enc copyFromBuffer:UnretainedBuffer(d.src)
                       sourceOffset:d.src_offset
                           toBuffer:UnretainedBuffer(d.dst)
                  destinationOffset:d.dst_offset
                               size:d.size];
            }
        };
        if (preserve_render && render != nil) {
            InterruptAndRestoreRenderPass(emit_blit);
        } else {
            emit_blit();
        }
    }

    void FlushClearAheadForRoot(const void* want_root) {
        if (pending_clears.empty() || want_root == nullptr ||
            (pending_clear_roots_bloom & PtrBloomBit(want_root)) == 0) {
            return;
        }
        std::vector<PendingClearEntry> matching;
        std::erase_if(pending_clears, [&](PendingClearEntry& entry) {
            if (entry.TouchesRoot(want_root)) {
                matching.push_back(std::move(entry));
                return true;
            }
            return false;
        });
        if (matching.empty()) {
            return;
        }
        RebuildPendingClearBloom();
        const bool can_ahead = ahead_enabled && merge_ahead_enabled && !ahead_blocked &&
                               !segment_opaque && WantsAheadEncoder() &&
                               std::ranges::none_of(matching, [](const PendingClearEntry& entry) {
                                   return entry.blocked_ahead;
                               });
        if (can_ahead) {
            @autoreleasepool {
                for (const PendingClearEntry& entry : matching) {
                    RecordClearOnAheadBuffer(entry.clear);
                }
            }
            return;
        }
        InterruptAndRestoreRenderPass([&] {
            for (const PendingClearEntry& entry : matching) {
                RecordClear(entry.clear);
            }
        });
    }

    /// Flushes matching `pending_clears` entries onto AheadBuffer() when possible, or in order
    /// (restoring any interrupted draw render pass) when an earlier in-order touch precedes them.
    void FlushClearAheadFor(TextureHandle texture) {
        if (pending_clear_roots_bloom == 0 || texture == nullptr) {
            return;
        }
        FlushClearAheadForRoot(CachedRootTexture(texture));
    }

    /// Flushes all remaining waiting clears in recording order, coalescing compatible passes.
    void FlushClear() {
        if (pending_clears.empty()) {
            return;
        }
        std::vector<PendingClearEntry> all = std::move(pending_clears);
        pending_clears.clear();
        pending_clear_roots_bloom = 0;
        std::vector<ClearPass> batches;
        batches.reserve(all.size());
        for (const PendingClearEntry& entry : all) {
            if (!batches.empty() && MergeClear(batches.back(), entry.clear)) {
                continue;
            }
            batches.push_back(entry.clear);
        }
        for (const ClearPass& batch : batches) {
            RecordClear(batch);
        }
    }

    /// A copy reads or writes `texture`: any waiting clear on its root texture goes first.
    void Touch(TextureHandle texture) {
        if (pending_clear_roots_bloom != 0 && texture != nullptr) {
            FlushClearForRoot(CachedRootTexture(texture));
        }
    }

    bool PendingClearOnRoot(const void* want_root) const noexcept {
        if (want_root == nullptr || (pending_clear_roots_bloom & PtrBloomBit(want_root)) == 0) {
            return false;
        }
        for (const PendingClearEntry& entry : pending_clears) {
            if (entry.TouchesRoot(want_root)) {
                return true;
            }
        }
        return false;
    }

    /// Whether any waiting clear is on `texture` (or what it is a view of).
    bool PendingClearOn(TextureHandle texture) const {
        if (pending_clear_roots_bloom == 0 || texture == nullptr) {
            return false;
        }
        return PendingClearOnRoot(CachedRootTexture(texture));
    }

    /// Work on `textures` comes next: waiting clears on them are recorded first
    /// (EDEN_NX2_METAL_CLEAR_NARROW), else all waiting clears are recorded.
    void ClearBefore(std::initializer_list<TextureHandle> textures) {
        if (pending_clear_roots_bloom == 0) {
            return;
        }
        if (!clear_narrow_enabled) {
            FlushClear();
            return;
        }
        for (const TextureHandle texture : textures) {
            Touch(texture);
        }
    }

    id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format, BlitBlend blend,
                                            BlitColorOp color_op) {
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        const BlitColorOp effective_op =
            format_class == FormatClass::Float ? color_op : BlitColorOp::Copy;
        const u64 key = (static_cast<u64>(format) << 16) | (static_cast<u64>(effective_op) << 8) |
                        static_cast<u64>(blend);
        if (const auto it = blit_pipelines.find(key); it != blit_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        NSString* fragment = nil;
        switch (format_class) {
        case FormatClass::Float:
            switch (effective_op) {
            case BlitColorOp::Copy:
                fragment = @"blit_float";
                break;
            case BlitColorOp::SrgbToLinear:
                fragment = @"blit_float_srgb_to_linear";
                break;
            case BlitColorOp::Square:
                fragment = @"blit_float_square";
                break;
            }
            break;
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
            @autoreleasepool {
                MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
                desc.label = @"Blit";
                desc.vertexFunction = [library newFunctionWithName:@"blit_vertex"];
                desc.fragmentFunction = [library newFunctionWithName:fragment];
                SetPipelineTargetFormat(desc, format, format_class);
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
                pipeline =
                    [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: blit pipeline for format {} failed: {}",
                              static_cast<u32>(format),
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
            }
        }
        blit_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The depth blits' state: every fragment writes its depth, and the stencil it outputs.
    id<MTLDepthStencilState> DepthWriteAlways() {
        if (depth_write_always == nil) {
            @autoreleasepool {
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
            @autoreleasepool {
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
                pipeline =
                    [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: clear pipeline for formats {} and {} failed: {}",
                              static_cast<u32>(color), static_cast<u32>(depth),
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
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
            @autoreleasepool {
                MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
                desc.label = @"Sample copy";
                desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
                desc.fragmentFunction = [library newFunctionWithName:fragment];
                desc.rasterSampleCount = samples;
                SetPipelineTargetFormat(desc, format, format_class);
                NSError* error = nil;
                pipeline =
                    [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: sample copy pipeline for format {} failed: {}",
                              static_cast<u32>(format),
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
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
            @autoreleasepool {
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
                pipeline =
                    [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: scaling pipeline for format {} failed: {}",
                              static_cast<u32>(format),
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
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
            @autoreleasepool {
                MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
                desc.label = @"Convert";
                desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
                desc.fragmentFunction = [library newFunctionWithName:fragment];
                SetPipelineTargetFormat(desc, format, format_class);
                NSError* error = nil;
                pipeline =
                    [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: conversion {} into format {} failed: {}",
                              static_cast<u32>(kind), static_cast<u32>(format),
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
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
        id<MTLDepthStencilState> state = nil;
        @autoreleasepool {
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
            state = [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
        }
        clear_depth_states.emplace(key, state);
        return state;
    }

    id<MTLRenderPipelineState> InPassClearPipeline(const InPassClearKey& key) {
        if (const auto it = inpass_clear_pipelines.find(key); it != inpass_clear_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        @autoreleasepool {
            u8 active_rt_mask = 0;
            for (u32 i = 0; i < 8; ++i) {
                if (((key.color_write_masks >> (i * 4)) & 0xFU) != 0) {
                    active_rt_mask |= static_cast<u8>(1U << i);
                }
            }
            NSString* fragment = nil;
            const u32 clear_count = static_cast<u32>(std::popcount(active_rt_mask));
            if (clear_count == 0) {
                fragment = @"clear_none";
            } else if (clear_count == 1) {
                const u32 rt = static_cast<u32>(std::countr_zero(active_rt_mask));
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
                        const u32 nibble = (key.color_write_masks >> (i * 4)) & 0xFU;
                        MTLColorWriteMask mtl_mask = MTLColorWriteMaskNone;
                        if ((nibble & 1U) != 0) {
                            mtl_mask |= MTLColorWriteMaskRed;
                        }
                        if ((nibble & 2U) != 0) {
                            mtl_mask |= MTLColorWriteMaskGreen;
                        }
                        if ((nibble & 4U) != 0) {
                            mtl_mask |= MTLColorWriteMaskBlue;
                        }
                        if ((nibble & 8U) != 0) {
                            mtl_mask |= MTLColorWriteMaskAlpha;
                        }
                        desc.colorAttachments[i].writeMask = mtl_mask;
                    }
                }
                if (key.depth_format != 0) {
                    desc.depthAttachmentPixelFormat =
                        static_cast<MTLPixelFormat>(key.depth_format);
                }
                if (key.stencil_format != 0) {
                    desc.stencilAttachmentPixelFormat =
                        static_cast<MTLPixelFormat>(key.stencil_format);
                }
                NSError* error = nil;
                pipeline =
                    [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: in-pass clear pipeline failed: {}",
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
            }
        }
        inpass_clear_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    bool FillInPassClearTargetFormats(InPassClearKey& key, u32& pass_w, u32& pass_h) const {
        if (render_targets.width == 0 || render_targets.height == 0) {
            return false;
        }
        pass_w = std::clamp(render_targets.width, 1U, MaxRenderTargetSize);
        pass_h = std::clamp(render_targets.height, 1U, MaxRenderTargetSize);
        const auto fit_attachment = [&](__unsafe_unretained id<MTLTexture> tex, u32 level) -> bool {
            const u8 tex_samples = static_cast<u8>(tex.sampleCount);
            if (key.samples == 0) {
                key.samples = tex_samples;
            } else if (key.samples != tex_samples) {
                return false;
            }
            const auto [mip_w, mip_h] = MipExtent(tex, level);
            pass_w = (std::min)(pass_w, mip_w);
            pass_h = (std::min)(pass_h, mip_h);
            return true;
        };
        for (size_t i = 0; i < render_targets.color.size(); ++i) {
            if (const id<MTLTexture> __unsafe_unretained tex =
                    UnretainedTexture(render_targets.color[i].texture);
                tex != nil) {
                const u32 fmt = static_cast<u32>(tex.pixelFormat);
                const FormatClass fmt_class = ClassOf(fmt);
                if (fmt_class != FormatClass::Float && fmt_class != FormatClass::Uint &&
                    fmt_class != FormatClass::Sint) {
                    return false;
                }
                key.color_formats[i] = fmt;
                if (!fit_attachment(tex, render_targets.color[i].level)) {
                    return false;
                }
            }
        }
        if (render_targets.has_depth || render_targets.has_stencil) {
            const id<MTLTexture> __unsafe_unretained tex =
                UnretainedTexture(render_targets.depth.texture);
            if (tex == nil) {
                return false;
            }
            const u32 fmt = static_cast<u32>(tex.pixelFormat);
            const FormatClass fmt_class = ClassOf(fmt);
            if (render_targets.has_depth) {
                if (fmt_class != FormatClass::Depth && fmt_class != FormatClass::DepthStencil) {
                    return false;
                }
                key.depth_format = fmt;
            }
            if (render_targets.has_stencil) {
                if (fmt_class != FormatClass::Stencil && fmt_class != FormatClass::DepthStencil) {
                    return false;
                }
                key.stencil_format = fmt;
            }
            if (!fit_attachment(tex, render_targets.depth.level)) {
                return false;
            }
        }
        return key.samples != 0 && pass_w != 0 && pass_h != 0;
    }

    void EmitInPassClearDraw(id<MTLRenderPipelineState> pipeline, u8 active_rt_mask,
                             const InPassClearKey& key, const std::array<f64, 4>* color_values,
                             bool single_color_value, bool clear_depth, f32 depth_val,
                             bool clear_stencil, u32 stencil_value, u32 stencil_mask,
                             u32 x, u32 y, u32 w, u32 h) {
        [render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        if (visibility_slot != -1) {
            visibility_slot = -1;
            ++visibility_serial;
        }
        [render setRenderPipelineState:pipeline];
        if (render_targets.has_depth || render_targets.has_stencil) {
            [render setDepthStencilState:ClearDepthState(clear_depth, clear_stencil, stencil_mask)];
            if (clear_stencil) {
                [render setStencilReferenceValue:stencil_value];
            }
        }
        [render setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                        static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
        [render setScissorRect:MTLScissorRect{x, y, w, h}];
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
        const f32 clamped_depth = std::clamp(depth_val, 0.0f, 1.0f);
        [render setVertexBytes:&clamped_depth length:sizeof(clamped_depth) atIndex:0];
        vertex_bound.buffers[0] = &ClobberedSlot;
        vertex_bound.ClearBufferDirty(0);
        if (active_rt_mask != 0) {
            if (std::popcount(active_rt_mask) == 1) {
                const u32 rt = static_cast<u32>(std::countr_zero(active_rt_mask));
                const std::array<u32, 4> bits = PackClearColorWords(
                    single_color_value ? color_values[0] : color_values[rt], key.color_formats[rt]);
                [render setFragmentBytes:bits.data() length:sizeof(bits) atIndex:0];
            } else {
                std::array<f32, 32> colors{};
                for (size_t i = 0; i < 8; ++i) {
                    for (size_t c = 0; c < 4; ++c) {
                        colors[i * 4 + c] = static_cast<f32>(color_values[i][c]);
                    }
                }
                [render setFragmentBytes:colors.data() length:sizeof(colors) atIndex:0];
            }
            fragment_bound.buffers[0] = &ClobberedSlot;
            fragment_bound.ClearBufferDirty(0);
        }
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        const u8 newly_written_color = static_cast<u8>(active_rt_mask & ~written_color);
        if (newly_written_color != 0) {
            u8 bits = newly_written_color;
            while (bits != 0) {
                const u32 rt = static_cast<u32>(std::countr_zero(bits));
                bits &= bits - 1U;
                NoteTextureWrite(render_targets.color[rt].texture);
            }
        }
        if ((clear_depth && !written_depth) || (clear_stencil && !written_stencil)) {
            NoteTextureWrite(render_targets.depth.texture);
        }
        written_color |= active_rt_mask;
        written_depth = written_depth || clear_depth;
        written_stencil = written_stencil || clear_stencil;
        render_state = {};
        render_state_valid = false;
        ++inpass_clears;
    }

    bool TryDrawInPassClear(const ClearPass& clear_pass) {
        if (!inpass_clear_enabled || render == nil || EffectiveTargetLayers(render_targets) > 1 ||
            clear_pass.from_bound_mismatch || clear_pass.width == 0 || clear_pass.height == 0) {
            return false;
        }
        u8 active_rt_mask = 0;
        u32 color_write_masks = 0;
        std::array<std::array<f64, 4>, 8> mapped_colors{};
        for (size_t i = 0; i < clear_pass.color.size(); ++i) {
            if (!clear_pass.clear_color[i]) {
                continue;
            }
            size_t matched_rt = render_targets.color.size();
            if (SameAttachmentSubresource(clear_pass.color[i], render_targets.color[i]) &&
                ((active_rt_mask >> i) & 1U) == 0) {
                matched_rt = i;
            } else {
                for (size_t rt = 0; rt < render_targets.color.size(); ++rt) {
                    if (((active_rt_mask >> rt) & 1U) == 0 &&
                        SameAttachmentSubresource(clear_pass.color[i], render_targets.color[rt])) {
                        matched_rt = rt;
                        break;
                    }
                }
            }
            if (matched_rt >= render_targets.color.size()) {
                return false;
            }
            active_rt_mask |= static_cast<u8>(1U << matched_rt);
            color_write_masks |= 0xFU << (matched_rt * 4);
            mapped_colors[matched_rt] = clear_pass.color_value[i];
        }
        if (clear_pass.clear_depth &&
            (!render_targets.has_depth ||
             !SameAttachmentSubresource(clear_pass.depth, render_targets.depth))) {
            return false;
        }
        if (clear_pass.clear_stencil &&
            (!render_targets.has_stencil ||
             !SameAttachmentSubresource(clear_pass.stencil, render_targets.depth))) {
            return false;
        }
        InPassClearKey key{};
        key.color_write_masks = color_write_masks;
        u32 pass_w = 0;
        u32 pass_h = 0;
        if (!FillInPassClearTargetFormats(key, pass_w, pass_h)) {
            return false;
        }
        const auto covers_cleared = [&](const Attachment& att) -> bool {
            const id<MTLTexture> __unsafe_unretained tex = UnretainedTexture(att.texture);
            if (tex == nil) {
                return false;
            }
            const auto [mip_w, mip_h] = MipExtent(tex, att.level);
            const u32 want_w = (std::min)(std::clamp(clear_pass.width, 1U, MaxRenderTargetSize),
                                          mip_w);
            const u32 want_h = (std::min)(std::clamp(clear_pass.height, 1U, MaxRenderTargetSize),
                                          mip_h);
            return pass_w >= want_w && pass_h >= want_h;
        };
        for (size_t i = 0; i < clear_pass.color.size(); ++i) {
            if (clear_pass.clear_color[i] && !covers_cleared(clear_pass.color[i])) {
                return false;
            }
        }
        if (clear_pass.clear_depth && !covers_cleared(clear_pass.depth)) {
            return false;
        }
        if (clear_pass.clear_stencil && !covers_cleared(clear_pass.stencil)) {
            return false;
        }
        const id<MTLRenderPipelineState> pipeline = InPassClearPipeline(key);
        if (pipeline == nil) {
            return false;
        }
        EmitInPassClearDraw(pipeline, active_rt_mask, key, mapped_colors.data(), false,
                            clear_pass.clear_depth, static_cast<f32>(clear_pass.depth_value),
                            clear_pass.clear_stencil, clear_pass.stencil_value, 0xFFU, 0, 0,
                            pass_w, pass_h);
        return true;
    }

    bool TryDrawInPassClearRect(const ClearRect& clear) {
        if (!inpass_clear_enabled || render == nil || EffectiveTargetLayers(render_targets) > 1) {
            return false;
        }
        u8 active_rt_mask = 0;
        u32 color_write_masks = 0;
        if (clear.color_mask != 0 && clear.color.texture != nullptr) {
            const u32 rt = clear.color_index;
            if (rt >= 8 || !SameAttachmentSubresource(clear.color, render_targets.color[rt])) {
                return false;
            }
            active_rt_mask = static_cast<u8>(1U << rt);
            color_write_masks = u32{clear.color_mask & 0xFU} << (rt * 4);
        }
        const bool want_depth = clear.clear_depth && clear.depth.texture != nullptr;
        const bool want_stencil = clear.clear_stencil && clear.depth.texture != nullptr;
        if (want_depth && (!render_targets.has_depth ||
                           !SameAttachmentSubresource(clear.depth, render_targets.depth))) {
            return false;
        }
        if (want_stencil && (!render_targets.has_stencil ||
                             !SameAttachmentSubresource(clear.depth, render_targets.depth))) {
            return false;
        }
        if (active_rt_mask == 0 && !want_depth && !want_stencil) {
            return false;
        }
        InPassClearKey key{};
        key.color_write_masks = color_write_masks;
        u32 pass_w = 0;
        u32 pass_h = 0;
        if (!FillInPassClearTargetFormats(key, pass_w, pass_h)) {
            return false;
        }
        const u32 x = (std::min)(clear.rect[0], pass_w);
        const u32 y = (std::min)(clear.rect[1], pass_h);
        const u32 w = (std::min)(clear.rect[2], pass_w - x);
        const u32 h = (std::min)(clear.rect[3], pass_h - y);
        if (w == 0 || h == 0) {
            return true;
        }
        const id<MTLRenderPipelineState> pipeline = InPassClearPipeline(key);
        if (pipeline == nil) {
            return false;
        }
        EmitInPassClearDraw(pipeline, active_rt_mask, key, &clear.color_value, true, want_depth,
                            clear.depth_value, want_stencil, clear.stencil_value,
                            clear.stencil_mask, x, y, w, h);
        return true;
    }

    const Device& device;
    id<MTLCommandQueue> queue;
    const bool state_cache_enabled;
    const bool draw_state_cache_enabled;
    const bool vertex_input_cache_enabled;
    const bool range_binds_enabled;
    const bool pipeline_check_enabled;
    const bool ahead_enabled;
    const bool ahead_split_enabled;
    const bool ahead_any_enabled;
    const bool merge_ahead_enabled;
    const bool lazy_ahead_enabled;
    const bool defer_downloads_enabled;
    const bool clear_fold_enabled;
    const bool clear_narrow_enabled;
    const bool depth_kill_enabled;
    const bool unwritten_stores_enabled;
    const bool msaa_discard_enabled;
    const bool inpass_clear_enabled;
    const bool static_labels_enabled;
    const bool image_note_skip_enabled;
    const bool segment_note_cache_enabled;
    const bool concurrent_compute_enabled;
    const bool mesh_wave_batch_enabled;
    const u32 compute_copy_mode;
    const bool compute_copy_enabled;
    const u64 compute_copy_max;
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
    id<MTLRenderCommandEncoder> blit_render = nil;
    Attachment blit_dst{};
    u32 blit_dst_format{};
    /// The render encoder draws go to, open until other work needs the command buffer.
    id<MTLRenderCommandEncoder> render = nil;
    RenderTargetSet render_targets{};
    u32 render_targets_raw_width{};
    u32 render_targets_raw_height{};
    /// Clears not recorded yet: folded into the load actions of later render passes on their
    /// attachments (EDEN_NX2_METAL_CLEAR_FOLD), with their textures retained until then.
    std::vector<PendingClearEntry> pending_clears;
    u64 pending_clear_roots_bloom{};
    u64 last_midpass_clear_pass_serial{~u64{0}};
    u64 clears_folded{};
    id<MTLComputeCommandEncoder> compute = nil;
    /// EDEN_NX2_METAL_STATE_CACHE: what the open encoders have set, so a draw or dispatch sets
    /// only what differs from the one before it.
    StageBindings vertex_bound;
    StageBindings fragment_bound;
    StageBindings object_bound;
    StageBindings mesh_bound;
    std::array<std::vector<const void*>, 4> pending_render_read_use{};
    std::vector<RenderUseEntry> pass_used_resources;
    std::vector<const void*> pending_in_order_compute_read_use;
    std::vector<const void*> pending_kernel_compute_read_use;
    u64 range_bind_calls{};
    u64 residency_batch_calls{};
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
    u8 color_cleared_mask{};
    bool pass_state_set{};
    /// The open pass cleared its depth or stencil attachment by its load action
    /// (RenderPassWroteDepthStencil).
    bool depth_stencil_cleared{};
    u64 stores_dropped{};
    StageBindings compute_bound;
    ComputeHazardState compute_hazards;
    const void* compute_pipeline{};
    StageBindings kernel_bound;
    ComputeHazardState kernel_hazards;
    const void* kernel_pipeline{};
    /// The compute encoder binds and dispatches go to (the segment's or the kernels'), and what
    /// it has bound.
    id<MTLComputeCommandEncoder> active_compute = nil;
    StageBindings* active_bound{};
    ComputeHazardState* active_hazards{};
    ReusablePassDescriptor draw_pass_desc;
    ReusablePassDescriptor helper_pass_desc;
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
    bool kernel_binding{};
    bool want_in_order_compute{};
    bool compute_clobbered_by_copy{};
    /// The binds of the kernel since its BeginCompute, to replay on another encoder.
    std::vector<ComputeBind> compute_binds;
    std::vector<u8> compute_bind_bytes;
    void* compute_bind_pipeline{};
    /// EDEN_NX2_METAL_MESH_WAVE_BATCH: deferred dependent waves on KernelEncoder() (0 = dist,
    /// 1 = mesh) so all wave-0 task dispatches in a batch run before one barrier, then all
    /// distributions, then one barrier, then all mesh kernels.
    struct DeferredAheadDispatch {
        void* pipeline{};
        u32 bind_begin{};
        u32 bind_count{};
        bool indirect{};
        std::array<u32, 3> groups{};
        std::array<u32, 3> group_size{};
        BufferHandle indirect_buffer{};
        u64 indirect_offset{};
    };
    struct AheadWaveQueue {
        std::vector<DeferredAheadDispatch> dispatches;
        std::vector<ComputeBind> binds;
        std::vector<u8> bytes;
        void Clear() noexcept {
            dispatches.clear();
            binds.clear();
            bytes.clear();
        }
        [[nodiscard]] bool Empty() const noexcept {
            return dispatches.empty();
        }
    };
    std::array<AheadWaveQueue, 2> ahead_waves;
    std::vector<ComputeHazardState::BufferSpan> ahead_wave_read_spans;
    int recording_ahead_wave{-1};
    bool ahead_wave_checked{};
    bool ahead_wave_flushing{};
    void* recording_wave_pipeline{};
    u32 recording_wave_bind_start{};
    u32 recording_wave_bytes_start{};
    /// EDEN_NX2_METAL_AHEAD_ANY: buffers modified in order in the segment, root textures
    /// written or touched in the segment, and whether in-order work did something not tracked.
    Common::unordered_set<const void*> segment_bound;
    Common::unordered_set<const void*> segment_written_textures;
    Common::unordered_set<const void*> segment_touched_textures;
    std::vector<const void*> pass_touched_roots;
    std::vector<const void*> pass_sampled_roots;
    u64 pass_touched_bloom{};
    u64 pass_sampled_bloom{};
    static constexpr u32 SegmentTouchedCacheBits = 11;
    static constexpr u32 RenderUseCacheBits = 12;
    static constexpr u32 ComputeUseCacheBits = 11;
    static constexpr u32 SegmentNotesBits = 11;
    static constexpr u32 ActiveBufferCacheBits = 11;
    static constexpr u32 AheadWriteNotesBits = 10;

    struct TouchedNote {
        const void* object{};
        const void* root{};
        u64 serial{};
        u64 pass_serial{~u64{0}};
        u64 sampled_pass_serial{~u64{0}};
    };
    std::array<TouchedNote, 1U << SegmentTouchedCacheBits> segment_touched_cache{};
    /// Per-encoder residency and GPU address / resource ID memo for UseBuffer and UseTexture.
    struct UseResourceNote {
        const void* resource{};
        u64 encoder_serial{};
        u64 id_or_addr{};
        u8 read_stages{};
        u8 write_stages{};
        u8 buffer_read_stages{};
        u8 buffer_write_stages{};
    };
    std::array<UseResourceNote, 1U << RenderUseCacheBits> render_use_cache{};
    std::array<UseResourceNote, 1U << ComputeUseCacheBits> compute_use_cache{};
    u64 next_compute_encoder_serial{1};
    u64 in_order_compute_serial{};
    u64 kernel_compute_serial{};
    /// Buffers and textures noted in the segment, by address
    /// (EDEN_NX2_METAL_SEGMENT_NOTE_CACHE).
    std::array<SegmentNote, 1U << SegmentNotesBits> segment_notes{};
    u64 segment_note_skips{};
    /// EDEN_NX2_METAL_COMPUTE_COPY: the copy and fill kernels, and operations recorded as their
    /// dispatches.
    id<MTLComputePipelineState> copy_kernel = nil;
    id<MTLComputePipelineState> fill_kernel = nil;
    bool copy_kernel_failed{};
    bool fill_kernel_failed{};
    u64 compute_copies_ahead{};
    u64 compute_copies_in_order{};
    bool segment_opaque{};
    /// The kernel being bound went ahead of a segment with other work: each buffer it binds is
    /// checked; a conflict moves it in order (and every later one of the segment).
    bool kernel_checked{};
    bool checked_failed{};
    /// The last CanRecordAhead answered yes by the check alone, for a draw into these.
    bool ahead_query_relaxed{};
    WantedPassKey ahead_query_key{};
    u64 ahead_checked{};
    u64 ahead_fallbacks{};
    /// EDEN_NX2_METAL_DEPTH_KILL: the open pass's attachments cleared by its load action
    /// have their store action left open; a clear of all of them next drops them.
    u8 cleared_color_open{};
    bool cleared_depth_open{};
    bool cleared_stencil_open{};
    u8 kill_color{};
    bool kill_depth{};
    bool kill_stencil{};
    u64 segment_serial{1};
    /// Encoders and render passes the segment's command buffer got so far.
    u32 segment_encoders{};
    u32 segment_render_passes{};
    u64 force_split_serial{};
    bool ahead_blocked{};
    /// Buffers referenced by any uncommitted command buffer of the current tick.
    struct ActiveBufferNote {
        const void* object{};
        u64 tick{};
        u64 length{};
        bool has_length{};
    };
    std::array<ActiveBufferNote, 1U << ActiveBufferCacheBits> active_buffer_cache{};
    Common::unordered_set<const void*> active_buffers;
    u32 ahead_encoders{};
    std::vector<ComputeHazardState::BufferSpan> ahead_written_spans;
    std::vector<ComputeHazardState::BufferSpan> ahead_read_spans;
    std::vector<const void*> ahead_touched_textures;
    std::vector<const void*> ahead_written_textures;
    std::vector<ComputeHazardState::BufferSpan> adopted_written_spans;
    std::vector<ComputeHazardState::BufferSpan> adopted_read_spans;
    /// Byte spans the segment's copies read or write in order.
    std::vector<ComputeHazardState::BufferSpan> in_order_spans;
    /// Byte spans the segment's work writes on the GPU in order.
    std::vector<ComputeHazardState::BufferSpan> in_order_written_spans;
    /// Byte spans read in order by operations without Buffer::tracker coverage.
    std::vector<ComputeHazardState::BufferSpan> in_order_untracked_read_spans;
    struct DeferredDownload {
        BufferHandle src{};
        u64 src_offset{};
        BufferHandle dst{};
        u64 dst_offset{};
        u64 size{};
    };
    std::vector<DeferredDownload> deferred_downloads;
    struct AheadWriteNote {
        const void* object{};
        u64 serial{};
    };
    std::array<AheadWriteNote, 1U << AheadWriteNotesBits> ahead_write_notes{};
    u64 ahead_write_clock{};
    /// Buffers touched in order with unbounded range and without Buffer::tracker coverage.
    std::vector<const void*> untracked_in_order_buffers;
    /// Segments the tick being recorded committed already.
    u32 tick_splits{};
    /// Render passes that are to begin a segment (WantSplitBefore).
    std::array<WantedEntry, 32> wanted{};
    size_t wanted_count{};
    u64 splits{};
    u64 ahead_copies{};
    u64 ahead_kernels{};
    u64 idle_fences{};
    u64 inpass_clears{};
    /// EDEN_NX2_METAL_FENCE_BATCH: work a fence left pending, since when, and the last commit.
    bool flush_pending{};
    std::chrono::steady_clock::time_point flush_pending_since{};
    bool list_end_pending{};
    std::chrono::steady_clock::time_point list_end_pending_since{};
    bool leaving_render_pass{};
    std::atomic<bool> in_render_pass{false};
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
    u64 visibility_serial{1};
    id<MTLLibrary> library = nil;
    id<MTLSamplerState> nearest = nil;
    id<MTLSamplerState> linear = nil;
    /// By (format << 16) | (BlitColorOp << 8) | BlitBlend.
    std::unordered_map<u64, id<MTLRenderPipelineState>> blit_pipelines;

    /// Shared with Metal command-buffer completion blocks so libdispatch callbacks that run right
    /// after waitUntilCompleted / MTLSharedEvent signals never touch destroyed Scheduler members.
    struct CompletionState {
        std::atomic<u64> completed_tick{0};
        std::mutex completed_mutex;
        std::condition_variable completed_cv;
        std::atomic<u64> gpu_ns{0};
        std::atomic<bool> completed_logged{false};
        std::atomic<u32> errors_logged{0};
        std::atomic<u32> submit_waiters{0};
        std::atomic<u32> urgent_submit_waiters{0};
    };
    std::shared_ptr<CompletionState> completion = std::make_shared<CompletionState>();

    u64 current_tick{1};
    std::atomic<u64> submitted_tick{0};
    id<MTLSharedEvent> event = nil;

    u64 CompletedTick() noexcept {
        u64 completed = completion->completed_tick.load(std::memory_order_acquire);
        if (event != nil) {
            const u64 signaled = [event signaledValue];
            while (completed < signaled &&
                   !completion->completed_tick.compare_exchange_weak(
                       completed, signaled, std::memory_order_acq_rel, std::memory_order_acquire)) {
            }
            if (signaled > completed) {
                completed = signaled;
            }
        }
        return completed;
    }
    /// AbandonWaits: WaitCompleted returns at once.
    std::atomic<bool> waits_abandoned{false};
    std::deque<std::pair<u64, id<MTLCommandBuffer>>> in_flight;

    u64 encoders{};
    u64 command_buffers{};
    u64 blit_commands{};
    u64 render_passes{};
    /// Since boot (the progress trace).
    u64 draws{};
    u64 draws_at_commit{};
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
    if (impl->completion->completed_tick.load(std::memory_order_acquire) >= tick) {
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
            std::unique_lock lock{impl->completion->completed_mutex};
            impl->completion->completed_cv.wait(lock, [&] { return IsFree(tick); });
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
    WaitCompleted(tick, false);
}

void Scheduler::WaitCompleted(u64 tick, bool allow_batch_grace) {
    if (IsFree(tick)) {
        return;
    }
    static const s64 explicit_patience_us = [] {
        const char* const value = std::getenv("EDEN_NX2_FENCE_PATIENCE_US");
        return value != nullptr ? (std::max)(static_cast<s64>(std::atoll(value)), s64{0})
                                : s64{-1};
    }();
    const auto start = std::chrono::steady_clock::now();
    const s64 interval_us = FenceFlushIntervalUs();
    s64 grace_us = 0;
    if (!VideoCore::Nx2Debug::DelayedFences() && SplitBackRunning()) {
        if (allow_batch_grace && explicit_patience_us >= 0) {
            grace_us = explicit_patience_us;
        } else if (interval_us > 0) {
            const bool gpu_busy = impl->CompletedTick() + 1 < tick;
            const bool pass_open = impl->in_render_pass.load(std::memory_order_relaxed);
            if (allow_batch_grace && gpu_busy) {
                grace_us = interval_us;
            } else if (pass_open) {
                grace_us = (std::min)(interval_us, allow_batch_grace ? s64{250} : s64{100});
            }
        }
    }
    auto next_request = start + std::chrono::microseconds{grace_us};
    auto next_warning = start + std::chrono::seconds{2};
    const bool use_shared_event = impl->event != nil && impl->recorder == nullptr;
    const auto maybe_warn = [&](std::chrono::steady_clock::time_point now, bool submitted) {
        if (now < next_warning) {
            return;
        }
        next_warning = now + std::chrono::seconds{2};
        const char* const back_site = VideoCommon::back_wait_site.load(std::memory_order_relaxed);
        LOG_WARNING(Render_Metal,
                    "Metal: fencing thread waited {} s for tick {} ({}; submitted up to {}, "
                    "completed up to {}); GPU back thread blocked in: {}",
                    std::chrono::duration_cast<std::chrono::seconds>(now - start).count(), tick,
                    submitted ? "committed" : "not committed",
                    impl->submitted_tick.load(std::memory_order_relaxed),
                    impl->completion->completed_tick.load(std::memory_order_relaxed),
                    back_site != nullptr ? back_site : "no tracked wait");
    };
    if (!IsSubmitted(tick) || !use_shared_event) {
        struct SubmitWaiterGuard {
            std::atomic<u32>& counter;
            std::atomic<u32>* urgent;
            SubmitWaiterGuard(std::atomic<u32>& c, std::atomic<u32>* u) noexcept
                : counter{c}, urgent{u} {
                counter.fetch_add(1, std::memory_order_acq_rel);
                if (urgent != nullptr) {
                    urgent->fetch_add(1, std::memory_order_acq_rel);
                }
            }
            ~SubmitWaiterGuard() {
                if (urgent != nullptr) {
                    urgent->fetch_sub(1, std::memory_order_acq_rel);
                }
                counter.fetch_sub(1, std::memory_order_acq_rel);
            }
        } waiter_guard{impl->completion->submit_waiters,
                       allow_batch_grace ? nullptr : &impl->completion->urgent_submit_waiters};
        std::unique_lock lock{impl->completion->completed_mutex};
        while (!IsFree(tick) && !impl->waits_abandoned.load(std::memory_order_acquire)) {
            const bool submitted = IsSubmitted(tick);
            if (submitted && use_shared_event) {
                break;
            }
            const auto now = std::chrono::steady_clock::now();
            if (!submitted && now < next_request && explicit_patience_us < 0 &&
                (allow_batch_grace ? impl->CompletedTick() + 1 >= tick : true) &&
                !impl->in_render_pass.load(std::memory_order_relaxed)) {
                next_request = now;
            }
            if (!submitted && now >= next_request && SplitBackRunning()) {
                // Fence work left pending (FenceTick, RequestCommit): the back commits it in its
                // idle hook, between entries when asked, else before it sleeps.
                SplitRequestIdleHook();
                next_request =
                    now + std::chrono::microseconds{allow_batch_grace ? s64{250} : s64{100}};
            }
            maybe_warn(now, submitted);
            if (submitted) {
                impl->completion->completed_cv.wait_for(lock, std::chrono::milliseconds{100});
            } else if (now < next_request) {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::microseconds>(next_request - now);
                impl->completion->completed_cv.wait_for(
                    lock, (std::min)(remaining, std::chrono::microseconds{100}));
            } else {
                impl->completion->completed_cv.wait_for(lock, std::chrono::milliseconds{5});
            }
        }
    }
    if (use_shared_event) {
        // Wait directly on the hardware IOGPU MTLSharedEvent rather than waiting for libdispatch's
        // default-QoS completion handler queue to run under heavy P-core load.
        while (!IsFree(tick) && !impl->waits_abandoned.load(std::memory_order_acquire)) {
            [impl->event waitUntilSignaledValue:tick timeoutMS:5];
            if (IsFree(tick)) {
                break;
            }
            maybe_warn(std::chrono::steady_clock::now(), true);
        }
    }
}

void Scheduler::WaitAnyThread(u64 tick, CommitReason reason, bool allow_batch_grace) {
    if (SplitOwnsScheduler()) {
        Wait(tick, reason);
    } else {
        WaitCompleted(tick, allow_batch_grace);
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
        std::scoped_lock lock{impl->completion->completed_mutex};
    }
    impl->completion->completed_cv.notify_all();
}

bool Scheduler::HasPendingWork() const noexcept {
    return impl->command_buffer != nil || !impl->pending_clears.empty() ||
           !impl->deferred_downloads.empty() || impl->upload_buffer != nil ||
           impl->kernel_buffer != nil || impl->ahead_buffer != nil || impl->tick_splits != 0;
}

u64 Scheduler::Flush(CommitReason reason) {
    const bool keep_clears =
        impl->clear_fold_enabled &&
        (reason == CommitReason::Fence || reason == CommitReason::FenceDeferred ||
         reason == CommitReason::Idle || reason == CommitReason::ListEnd ||
         reason == CommitReason::ListCopy || reason == CommitReason::DrawBudget ||
         reason == CommitReason::Staging || reason == CommitReason::Query ||
         reason == CommitReason::Condition);
    if (!keep_clears && !impl->pending_clears.empty()) {
        @autoreleasepool {
            impl->FlushClear();
        }
    }
    const bool has_gpu_commands = impl->command_buffer != nil ||
                                  !impl->deferred_downloads.empty() ||
                                  impl->upload_buffer != nil || impl->kernel_buffer != nil ||
                                  impl->ahead_buffer != nil || impl->tick_splits != 0;
    if (!has_gpu_commands) {
        impl->empty_flushes.fetch_add(1, std::memory_order_relaxed);
        // Every earlier tick is done and nothing was recorded for this one: it is done too.
        const u64 empty_tick = impl->current_tick;
        if (EmptySkipEnabled() && impl->CompletedTick() + 1 >= empty_tick) {
            impl->empty_skipped.fetch_add(1, std::memory_order_relaxed);
            ++impl->current_tick;
            if (impl->event != nil) {
                impl->event.signaledValue = empty_tick;
            }
            impl->submitted_tick.store(empty_tick, std::memory_order_release);
            u64 previous = impl->completion->completed_tick.load(std::memory_order_relaxed);
            while (previous < empty_tick &&
                   !impl->completion->completed_tick.compare_exchange_weak(
                       previous, empty_tick, std::memory_order_release)) {
            }
            {
                std::scoped_lock lock{impl->completion->completed_mutex};
            }
            impl->completion->completed_cv.notify_all();
            impl->active_buffers.clear();
            impl->NextSegment();
            impl->tick_splits = 0;
            impl->flush_pending = false;
            impl->list_end_pending = false;
            impl->draws_at_commit = impl->draws;
            while (!impl->in_flight.empty() && IsFree(impl->in_flight.front().first)) {
                impl->in_flight.pop_front();
            }
            return empty_tick;
        }
    }
    @autoreleasepool {
        impl->commit_counts[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
        if (!keep_clears) {
            impl->FlushClear();
        }
        impl->EndEncoder(PassEnd::Commit);
        impl->AdoptAheadIfMainEmpty();
        impl->FlushDeferredDownloads();
        if (impl->blit != nil) {
            [impl->blit endEncoding];
            impl->blit = nil;
        }
        const id<MTLCommandBuffer> command_buffer = impl->Commands();
        const u64 tick = impl->current_tick++;
        const std::shared_ptr<Impl::CompletionState> state = impl->completion;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            const CFTimeInterval gpu_start = finished.GPUStartTime;
            const CFTimeInterval gpu_end = finished.GPUEndTime;
            const CFTimeInterval busy =
                (gpu_start > 0.0 && gpu_end > gpu_start) ? (gpu_end - gpu_start) : 0.0;
            if (busy > 0.0) {
                state->gpu_ns.fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
            }
            if (!state->completed_logged.exchange(true)) {
                LOG_INFO(Render_Metal,
                         "Metal: first command buffer completed (tick {}, status {}, "
                         "GPU {:.2f} ms)",
                         tick, static_cast<int>(finished.status), busy * 1e3);
            }
            ReportFailure(finished, tick, &state->errors_logged);
            u64 previous = state->completed_tick.load(std::memory_order_relaxed);
            while (previous < tick && !state->completed_tick.compare_exchange_weak(
                                          previous, tick, std::memory_order_release)) {
            }
            {
                std::scoped_lock lock{state->completed_mutex};
            }
            state->completed_cv.notify_all();
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
        if (impl->completion->submit_waiters.load(std::memory_order_acquire) != 0) {
            {
                std::scoped_lock lock{impl->completion->completed_mutex};
            }
            impl->completion->completed_cv.notify_all();
        }
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
                         impl->committed,
                         impl->completion->completed_tick.load(std::memory_order_relaxed),
                         impl->draws, impl->dispatches, impl->blank_presents);
            }
        }
        impl->command_buffer = nil;
        impl->active_buffers.clear();
        impl->NextSegment();
        impl->tick_splits = 0;
        impl->flush_pending = false;
        impl->list_end_pending = false;
        impl->draws_at_commit = impl->draws;
        impl->last_commit = std::chrono::steady_clock::now();
        impl->in_flight.emplace_back(tick, command_buffer);
        while (!impl->in_flight.empty() && IsFree(impl->in_flight.front().first)) {
            impl->in_flight.pop_front();
        }
        return tick;
    }
}

void Scheduler::Finish(CommitReason reason) {
    Wait(Flush(reason), reason);
}

bool Scheduler::DeferFlush() {
    const s64 interval_us = FenceFlushIntervalUs();
    if (interval_us <= 0) {
        return false;
    }
    const bool on_back = SplitOnBackThread() || VideoCommon::GpuSplit::OnBackThread();
    if (!on_back && !FenceBatchAnyThread()) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    const bool explicit_window = FenceFlushExplicit() || FenceBatchAnyThread();
    const bool in_pass = impl->render != nil;
    const auto window = std::chrono::microseconds{interval_us};
    bool can_coalesce = false;
    if (explicit_window) {
        can_coalesce = now - impl->last_commit < window;
    } else {
        const bool front_urgent =
            VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
            impl->completion->urgent_submit_waiters.load(std::memory_order_relaxed) != 0;
        if (!in_pass && front_urgent) {
            return false;
        }
        const bool gpu_busy = GpuBehind();
        if (gpu_busy || in_pass) {
            const auto active_window =
                gpu_busy ? window : std::chrono::microseconds{(std::min)(interval_us, s64{250})};
            can_coalesce =
                !impl->flush_pending || now - impl->flush_pending_since < active_window;
        }
    }
    if (!can_coalesce) {
        return false;
    }
    if (!impl->flush_pending) {
        impl->flush_pending = true;
        impl->flush_pending_since = now;
    }
    return true;
}

u64 Scheduler::FenceTick() {
    if (FenceIdleEnabled() && !HasPendingWork()) {
        // Everything before the fence is in the ticks committed so far (tick 0 is always free).
        ++impl->idle_fences;
        return impl->current_tick - 1;
    }
    if (DeferFlush()) {
        ++impl->deferred_fences;
        return impl->current_tick;
    }
    return Flush(CommitReason::Fence);
}

bool Scheduler::DeferListEnd() {
    const u64 depth = ListEndDeferDepth();
    const bool on_back = SplitOnBackThread() || VideoCommon::GpuSplit::OnBackThread();
    if (depth == 0 || !on_back) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    const bool in_pass = impl->render != nil;
    const bool explicit_window = FenceFlushExplicit();
    const u64 completed = impl->CompletedTick();
    const bool gpu_busy = !impl->in_flight.empty() && completed + depth < impl->current_tick;
    if (!gpu_busy && !in_pass && !explicit_window) {
        return false;
    }
    if (!ListEndDeferExplicit()) {
        const s64 interval_us = (std::max)(FenceFlushIntervalUs(), s64{250});
        const auto window = std::chrono::microseconds{interval_us};
        const auto active_window =
            (gpu_busy || explicit_window)
                ? window
                : std::chrono::microseconds{(std::min)(interval_us, s64{250})};
        if ((!gpu_busy && !in_pass && now - impl->last_commit >= window) ||
            (impl->flush_pending && now - impl->flush_pending_since >= active_window) ||
            (impl->list_end_pending && now - impl->list_end_pending_since >= active_window)) {
            return false;
        }
    }
    if (!impl->list_end_pending) {
        impl->list_end_pending = true;
        impl->list_end_pending_since = now;
    }
    ++impl->list_ends_deferred;
    return true;
}

bool Scheduler::FlushDeferredFromIdle(bool requested) {
    const bool idle_all = IdleCommitAllEnabled() && HasPendingWork();
    const bool on_back = SplitOnBackThread() || VideoCommon::GpuSplit::OnBackThread();
    const bool urgent =
        VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
        impl->completion->urgent_submit_waiters.load(std::memory_order_relaxed) != 0;
    if (requested) {
        if (!impl->flush_pending && !impl->list_end_pending && !idle_all) {
            return false;
        }
        if (on_back && !idle_all && impl->render != nil && !urgent) {
            const auto since =
                impl->flush_pending ? impl->flush_pending_since : impl->list_end_pending_since;
            const s64 max_hold_us =
                (std::min)((std::max)(FenceFlushIntervalUs(), s64{150}), s64{250});
            if (std::chrono::steady_clock::now() - since <
                std::chrono::microseconds{max_hold_us}) {
                return false;
            }
        }
        Flush(CommitReason::FenceDeferred);
        return true;
    }
    if (!idle_all) {
        if (!impl->flush_pending && !impl->list_end_pending) {
            return false;
        }
        const s64 hold_us = (std::max)(FenceFlushIntervalUs(), s64{250});
        const auto since =
            impl->flush_pending ? impl->flush_pending_since : impl->list_end_pending_since;
        const bool within_window =
            std::chrono::steady_clock::now() - since < std::chrono::microseconds{hold_us};
        if (on_back && !urgent &&
            impl->completion->submit_waiters.load(std::memory_order_relaxed) == 0 &&
            ((within_window && (GpuBehind() || impl->render != nil)) ||
             (!impl->flush_pending && GpuBehind()))) {
            return false;
        }
    }
    ++impl->idle_commits;
    Flush(CommitReason::Idle);
    return true;
}

void Scheduler::FlushDeferred() {
    if (impl->flush_pending || impl->list_end_pending) {
        Flush(CommitReason::FenceDeferred);
    }
}

void Scheduler::FlushExpiredDeferred() {
    if (!impl->flush_pending && !impl->list_end_pending) {
        return;
    }
    const s64 interval_us = FenceFlushIntervalUs();
    const auto now = std::chrono::steady_clock::now();
    const bool in_pass = impl->render != nil && !impl->leaving_render_pass;
    const bool explicit_window = FenceFlushExplicit() || FenceBatchAnyThread();
    const bool gpu_busy = explicit_window || GpuBehind();
    const bool front_urgent =
        !explicit_window &&
        (VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
         impl->completion->urgent_submit_waiters.load(std::memory_order_relaxed) != 0);
    const auto window = std::chrono::microseconds{(std::max)(interval_us, s64{250})};
    const auto fence_limit =
        in_pass ? (gpu_busy ? window * 2
                            : std::chrono::microseconds{(std::min)(interval_us, s64{250})})
                : window;
    const auto list_limit = in_pass
                                ? (gpu_busy ? window * 2
                                            : std::chrono::microseconds{(std::min)(interval_us, s64{250})})
                                : window;
    const bool fence_expired =
        impl->flush_pending &&
        (interval_us <= 0 || (!in_pass && (front_urgent || !gpu_busy)) ||
         now - impl->flush_pending_since >= fence_limit);
    const bool list_expired =
        impl->list_end_pending &&
        ((!in_pass && (front_urgent || !gpu_busy)) ||
         now - impl->list_end_pending_since >= list_limit);
    if (fence_expired || list_expired) {
        Flush(impl->flush_pending ? CommitReason::FenceDeferred : CommitReason::ListEnd);
    }
}

bool Scheduler::DeferredFlushMayWait() const {
    const auto window = std::chrono::microseconds{FenceFlushIntervalUs()};
    const auto now = std::chrono::steady_clock::now();
    return (impl->flush_pending && now - impl->flush_pending_since < window) ||
           (impl->list_end_pending && now - impl->list_end_pending_since < window);
}

bool Scheduler::GpuBehind() const noexcept {
    return impl->CompletedTick() + 1 < impl->current_tick;
}

bool Scheduler::GpuCaughtUp() const noexcept {
    return impl->CompletedTick() + 1 >= impl->current_tick;
}

u64 Scheduler::DrawsSinceCommit() const noexcept {
    return impl->draws - impl->draws_at_commit;
}

std::array<u64, 2> Scheduler::TakeDeferredFences() noexcept {
    return {std::exchange(impl->deferred_fences, 0), std::exchange(impl->idle_commits, 0)};
}

void Scheduler::CopyBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset,
                           u64 size, bool ahead, bool range_tracked, bool src_range_tracked) {
    if (size == 0 || src == nullptr || dst == nullptr) {
        return;
    }
    impl->NoteActiveBuffer(src);
    impl->NoteActiveBuffer(dst);
    const bool can_compute_no_barrier =
        impl->compute_copy_enabled && size <= impl->compute_copy_max && src != dst &&
        !impl->compute_hazards.has_dispatch_since_buffer_barrier &&
        !impl->compute_hazards.HasCopyHazard(src, src_offset, size, dst, dst_offset);
    const bool goes_ahead = ahead && CanCopyBufferAhead() &&
                            impl->WantsAheadEncoder(can_compute_no_barrier) &&
                            !impl->WrittenInOrder(src, src_offset, size) &&
                            !impl->WrittenInOrder(dst, dst_offset, size) &&
                            !impl->UntrackedReadInOrder(dst, dst_offset, size) &&
                            !impl->AdoptedWritten(src, src_offset, size) &&
                            !impl->AdoptedWritten(dst, dst_offset, size) &&
                            !impl->AdoptedRead(dst, dst_offset, size);
    if (goes_ahead) {
        impl->NoteAheadWrite(dst);
        impl->NoteAheadReadSpan(src, src_offset, size);
        impl->NoteAheadWriteSpan(dst, dst_offset, size);
    } else {
        impl->FlushDeferredDownloadsFor(dst, dst_offset, size, false);
        impl->NoteInOrder(src, src_offset, size);
        impl->NoteInOrderWrite(dst, dst_offset, size);
        if (!range_tracked && !src_range_tracked) {
            impl->NoteUntrackedInOrderRead(src, src_offset, size);
        }
    }
    // EDEN_NX2_METAL_COMPUTE_COPY: on the compute encoder open where the blit would go.
    if (impl->compute_copy_enabled && size <= impl->compute_copy_max && src != dst) {
        Impl& state = *impl;
        if (goes_ahead) {
            // The ahead command buffer's kernels, open (so no upload blit is): the copy lands
            // after the kernels recorded before it, as UploadEncoder's would.
            if (state.merge_ahead_enabled && state.upload_blit == nil &&
                state.kernel_compute == nil && state.compute_copy_mode == 2 &&
                !state.copy_kernel_failed && state.library != nil) {
                state.KernelEncoder();
            }
            if (state.merge_ahead_enabled && state.kernel_compute != nil &&
                state.upload_blit == nil &&
                state.CopyByCompute(state.kernel_compute, state.kernel_bound, state.kernel_hazards,
                                    &state.kernel_pipeline, src, src_offset, dst, dst_offset,
                                    size)) {
                ++state.ahead_copies;
                ++state.compute_copies_ahead;
                return;
            }
        } else {
            if (state.compute == nil && state.want_in_order_compute && state.render == nil &&
                state.blit == nil && state.blit_render == nil) {
                state.OpenInOrderCompute();
            }
            if (state.compute != nil && state.render == nil && state.blit == nil &&
                state.blit_render == nil &&
                state.CopyByCompute(state.compute, state.compute_bound, state.compute_hazards,
                                    &state.compute_pipeline, src, src_offset, dst, dst_offset,
                                    size)) {
                ++state.compute_copies_in_order;
                return;
            }
        }
    }
    const id<MTLBlitCommandEncoder> __unsafe_unretained encoder =
        goes_ahead ? impl->UploadEncoder() : impl->BlitEncoder();
    [encoder copyFromBuffer:UnretainedBuffer(src)
               sourceOffset:src_offset
                   toBuffer:UnretainedBuffer(dst)
          destinationOffset:dst_offset
                       size:size];
}

void Scheduler::DownloadBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset,
                               u64 size) {
    if (size == 0 || src == nullptr || dst == nullptr) {
        return;
    }
    if (!impl->defer_downloads_enabled) {
        CopyBuffer(src, src_offset, dst, dst_offset, size, false, false, true);
        return;
    }
    impl->NoteActiveBuffer(src);
    impl->NoteActiveBuffer(dst);
    impl->NoteInOrder(src, src_offset, size);
    impl->NoteUntrackedInOrderRead(src, src_offset, size);
    impl->NoteInOrderWrite(dst, dst_offset, size);
    if (impl->blit != nil) {
        ++impl->blit_commands;
        [impl->blit copyFromBuffer:UnretainedBuffer(src)
                      sourceOffset:src_offset
                          toBuffer:UnretainedBuffer(dst)
                 destinationOffset:dst_offset
                              size:size];
        return;
    }
    impl->deferred_downloads.push_back({src, src_offset, dst, dst_offset, size});
}

void Scheduler::FlushDeferredDownloadsFor(BufferHandle buffer, u64 offset, u64 size) {
    impl->FlushDeferredDownloadsFor(buffer, offset, size, impl->render != nil);
}

u64 Scheduler::SegmentSerial() const noexcept {
    return impl->segment_serial;
}

bool Scheduler::CanUploadAhead() const noexcept {
    return impl->ahead_enabled && !impl->ahead_blocked && !impl->segment_opaque;
}

bool Scheduler::CanCopyBufferAhead() const noexcept {
    return CanUploadAhead();
}

bool Scheduler::CanWriteBufferAhead(BufferHandle dst, u64 offset, u64 size) const noexcept {
    if (dst == nullptr || !CanCopyBufferAhead() || UsedUntrackedInOrder(dst)) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    return !impl->WrittenInOrder(dst, offset, size) &&
           !impl->UntrackedReadInOrder(dst, offset, size) &&
           !impl->AdoptedWritten(dst, offset, size) &&
           !impl->AdoptedRead(dst, offset, size);
}

bool Scheduler::CanReadBufferAhead(BufferHandle src, u64 offset, u64 size) const noexcept {
    if (src == nullptr || !CanCopyBufferAhead() || UsedUntrackedInOrder(src)) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    return !impl->WrittenInOrder(src, offset, size) &&
           !impl->AdoptedWritten(src, offset, size);
}

bool Scheduler::CanOrderAheadCopies() const noexcept {
    return impl->merge_ahead_enabled;
}

u64 Scheduler::AheadCopyEpoch() const noexcept {
    return impl->ahead_copies;
}

u64 Scheduler::BufferAheadWriteSerial(BufferHandle buffer) const noexcept {
    if (buffer == nullptr) {
        return 0;
    }
    const auto& note = impl->ahead_write_notes[PtrHashIndex(buffer, Impl::AheadWriteNotesBits)];
    return note.object == nullptr ? 0 : note.serial;
}

bool Scheduler::HasOpenInOrderCompute() const noexcept {
    return impl->compute != nil || impl->want_in_order_compute;
}

bool Scheduler::IsBufferActiveInOpenCommands(BufferHandle buffer) const {
    if (buffer == nullptr) {
        return false;
    }
    const bool open_commands = impl->command_buffer != nil ||
                               !impl->deferred_downloads.empty() ||
                               impl->upload_buffer != nil || impl->kernel_buffer != nil ||
                               impl->ahead_buffer != nil || impl->tick_splits != 0;
    if (!open_commands) {
        return false;
    }
    if (impl->active_buffers.contains(buffer) || impl->segment_bound.contains(buffer)) {
        return true;
    }
    if (impl->UsedInOrder(buffer)) {
        return true;
    }
    const auto in_stage = [buffer](const StageBindings& bound) noexcept {
        return std::find(bound.buffers.begin(), bound.buffers.end(), buffer) !=
               bound.buffers.end();
    };
    if (impl->render != nil &&
        (in_stage(impl->vertex_bound) || in_stage(impl->fragment_bound) ||
         in_stage(impl->object_bound) || in_stage(impl->mesh_bound))) {
        return true;
    }
    if (impl->compute != nil && in_stage(impl->compute_bound)) {
        return true;
    }
    if (impl->kernel_compute != nil && in_stage(impl->kernel_bound)) {
        return true;
    }
    return false;
}

bool Scheduler::CanUploadTextureAhead(TextureHandle texture) const noexcept {
    if (texture == nullptr || impl->segment_opaque || !CanUploadAhead()) {
        return false;
    }
    if (!impl->merge_ahead_enabled && impl->render == nil) {
        return false;
    }
    const void* const raw_root = impl->CachedRootTexture(texture);
    if (raw_root == nullptr) {
        return false;
    }
    if (impl->PendingClearOnRoot(raw_root) && !impl->CanFlushClearAheadForRoot(raw_root)) {
        return false;
    }
    if (!impl->shadows.empty() &&
        (impl->shadows.contains(texture) ||
         impl->shadows.contains(const_cast<void*>(raw_root)))) {
        return false;
    }
    return !impl->segment_touched_textures.contains(raw_root);
}

bool Scheduler::UsedInOrder(BufferHandle buffer, u64 offset, u64 size) const noexcept {
    return impl->UsedInOrder(buffer, offset, size);
}

bool Scheduler::WrittenInOrder(BufferHandle buffer, u64 offset, u64 size) const noexcept {
    return impl->WrittenInOrder(buffer, offset, size);
}

bool Scheduler::UsedUntrackedInOrder(BufferHandle buffer) const noexcept {
    return std::find(impl->untracked_in_order_buffers.begin(),
                     impl->untracked_in_order_buffers.end(),
                     buffer) != impl->untracked_in_order_buffers.end();
}

bool Scheduler::CanRecordAhead(const RenderTargetSet& targets) const noexcept {
    if (!impl->ahead_enabled || impl->ahead_blocked) {
        return false;
    }
    if (!impl->pending_clears.empty() && !impl->CanFoldPendingClears(targets)) {
        return false;
    }
    if (impl->segment_encoders == 0 || (impl->segment_encoders == 1 && IsRenderingTo(targets))) {
        impl->ahead_query_relaxed = false;
        return true;
    }
    // EDEN_NX2_METAL_AHEAD_ANY: ahead after other work, each buffer the kernels bind checked
    // against what the segment's in-order work bound (BeginCompute).
    if (impl->ahead_any_enabled && !impl->segment_opaque && !impl->checked_failed) {
        impl->ahead_query_relaxed = true;
        impl->ahead_query_key = MakeWantedPassKey(targets);
        return true;
    }
    return false;
}

void Scheduler::BlockAhead() noexcept {
    impl->ahead_blocked = true;
}

bool Scheduler::IsRenderingTo(const RenderTargetSet& targets) const noexcept {
    return impl->MatchesOpenTargets(targets);
}

void Scheduler::WantSplitBefore(const RenderTargetSet& targets) {
    if (!impl->ahead_enabled || !impl->ahead_split_enabled) {
        return;
    }
    const bool force = !IsRenderingTo(targets);
    if (force) {
        impl->force_split_serial = impl->segment_serial;
    }
    impl->NoteWantedKey(MakeWantedPassKey(targets), force);
}

void Scheduler::WantSplitBeforeCurrentPass() {
    if (impl->render != nil && impl->ahead_enabled && impl->ahead_split_enabled) {
        impl->NoteWantedKey(MakeWantedPassKey(impl->render_targets), true);
    }
}

void Scheduler::SplitBefore(const RenderTargetSet& targets) {
    if (!IsRenderingTo(targets) && (impl->flush_pending || impl->list_end_pending)) {
        const bool on_back = SplitOnBackThread() || VideoCommon::GpuSplit::OnBackThread();
        if (on_back && HasPendingWork()) {
            impl->leaving_render_pass = true;
            FlushExpiredDeferred();
            impl->leaving_render_pass = false;
        }
    }
    constexpr u32 MaxTickSplits = 10;
    if (!impl->ahead_enabled || !impl->ahead_split_enabled || IsRenderingTo(targets) ||
        impl->tick_splits >= MaxTickSplits || impl->segment_encoders == 0) {
        return;
    }
    const Impl::WantedEntry* const wanted_entry = impl->FindWanted(targets);
    if (wanted_entry == nullptr) {
        return;
    }
    const bool same_segment_force = impl->force_split_serial == impl->segment_serial;
    const bool can_ahead = CanRecordAhead(targets);
    if (!same_segment_force && can_ahead) {
        return;
    }
    if (!impl->pending_clears.empty() && !impl->CanFoldPendingClears(targets)) {
        const u32 eff_layers = EffectiveTargetLayers(targets);
        const auto target_subs = impl->TargetSubresourcesOf(targets);
        std::vector<Impl::PendingClearEntry> to_flush;
        std::erase_if(impl->pending_clears, [&](Impl::PendingClearEntry& entry) {
            if (!impl->EntryCanFoldOrDefer(entry, targets, target_subs, eff_layers)) {
                to_flush.push_back(std::move(entry));
                return true;
            }
            return false;
        });
        if (!to_flush.empty()) {
            impl->RebuildPendingClearBloom();
            for (const Impl::PendingClearEntry& entry : to_flush) {
                impl->RecordClear(entry.clear);
            }
        }
    }
    if (impl->segment_encoders != 0) {
        impl->Split();
    }
}

void Scheduler::FillBuffer(BufferHandle dst, u64 offset, u64 size, u8 value, bool ahead,
                           bool range_tracked) {
    FillBufferPattern(dst, offset, size, u32{value} * 0x01010101U, ahead, range_tracked);
}

void Scheduler::FillBufferPattern(BufferHandle dst, u64 offset, u64 size, u32 pattern, bool ahead,
                                  [[maybe_unused]] bool range_tracked) {
    if (size == 0 || dst == nullptr) {
        return;
    }
    impl->NoteActiveBuffer(dst);
    const u8 byte = static_cast<u8>(pattern);
    const bool uniform_byte = pattern == u32{byte} * 0x01010101U;
    const u64 compute_fill_max = uniform_byte ? (std::min)(impl->compute_copy_max, u64{65536})
                                              : impl->compute_copy_max;
    const bool can_compute_no_barrier =
        impl->compute_copy_enabled && size <= compute_fill_max &&
        !impl->compute_hazards.has_dispatch_since_buffer_barrier &&
        !impl->compute_hazards.HasFillHazard(dst, offset, size);
    const bool goes_ahead = ahead && CanCopyBufferAhead() &&
                            impl->WantsAheadEncoder(can_compute_no_barrier) &&
                            !impl->WrittenInOrder(dst, offset, size) &&
                            !impl->UntrackedReadInOrder(dst, offset, size) &&
                            !impl->AdoptedWritten(dst, offset, size) &&
                            !impl->AdoptedRead(dst, offset, size);
    if (goes_ahead) {
        impl->NoteAheadWrite(dst);
        impl->NoteAheadWriteSpan(dst, offset, size);
    } else {
        impl->FlushDeferredDownloadsFor(dst, offset, size, false);
        impl->NoteInOrderWrite(dst, offset, size);
    }
    if (impl->compute_copy_enabled && size <= compute_fill_max) {
        Impl& state = *impl;
        if (goes_ahead) {
            if (state.merge_ahead_enabled && !uniform_byte && state.upload_blit == nil &&
                state.kernel_compute == nil && state.compute_copy_mode == 2 &&
                !state.fill_kernel_failed && state.library != nil) {
                state.KernelEncoder();
            }
            if (state.merge_ahead_enabled && state.kernel_compute != nil &&
                state.upload_blit == nil &&
                state.FillByCompute(state.kernel_compute, state.kernel_bound, state.kernel_hazards,
                                    &state.kernel_pipeline, dst, offset, size, pattern)) {
                ++state.ahead_copies;
                ++state.compute_copies_ahead;
                return;
            }
        } else {
            if (state.compute == nil && state.want_in_order_compute && state.render == nil &&
                state.blit == nil && state.blit_render == nil) {
                state.OpenInOrderCompute();
            }
            if (state.compute != nil && state.render == nil && state.blit == nil &&
                state.blit_render == nil &&
                state.FillByCompute(state.compute, state.compute_bound, state.compute_hazards,
                                    &state.compute_pipeline, dst, offset, size, pattern)) {
                ++state.compute_copies_in_order;
                return;
            }
        }
    }
    if (uniform_byte) {
        const id<MTLBlitCommandEncoder> __unsafe_unretained encoder =
            goes_ahead ? impl->UploadEncoder() : impl->BlitEncoder();
        [encoder fillBuffer:UnretainedBuffer(dst) range:NSMakeRange(offset, size) value:byte];
        return;
    }
    Impl& state = *impl;
    if (goes_ahead && state.merge_ahead_enabled && !state.fill_kernel_failed &&
        state.library != nil) {
        state.KernelEncoder();
        if (state.kernel_compute != nil &&
            state.FillByCompute(state.kernel_compute, state.kernel_bound, state.kernel_hazards,
                                &state.kernel_pipeline, dst, offset, size, pattern)) {
            ++state.ahead_copies;
            ++state.compute_copies_ahead;
            return;
        }
    }
    state.OpenInOrderCompute();
    static_cast<void>(state.FillByCompute(state.compute, state.compute_bound, state.compute_hazards,
                                          &state.compute_pipeline, dst, offset, size, pattern));
}

void Scheduler::CopyBufferToTexture(BufferHandle src, u64 offset, u32 bytes_per_row,
                                    u32 bytes_per_image, TextureHandle dst, u32 level, u32 slice,
                                    Offset3D origin, Extent3D extent, bool ahead,
                                    bool src_range_tracked) {
    if (src == nullptr || dst == nullptr) {
        impl->Written(dst);
        return;
    }
    impl->DropMipClear(dst, level, slice, origin, extent);
    const void* const dst_root = impl->CachedRootTexture(dst);
    impl->WrittenRoot(dst, dst_root);
    impl->NoteActiveBuffer(src);
    const id<MTLTexture> __unsafe_unretained mtl_dst = UnretainedTexture(dst);
    const u64 copy_bytes = TextureCopyBufferBytes(mtl_dst, bytes_per_row, bytes_per_image, extent);
    const bool auto_hoist =
        !ahead && impl->render != nil && impl->merge_ahead_enabled && !UsedUntrackedInOrder(src) &&
        ComputeHazardState::BackingBufferSpan(dst_root).buffer == nullptr;
    const bool goes_ahead = (ahead || auto_hoist) && CanUploadTextureAhead(dst) &&
                            impl->WantsAheadEncoder() &&
                            !impl->WrittenInOrder(src, offset, copy_bytes) &&
                            !impl->AdoptedWritten(src, offset, copy_bytes);
    if (goes_ahead) {
        if (impl->PendingClearOnRoot(dst_root)) {
            impl->FlushClearAheadForRoot(dst_root);
        }
        impl->NoteAheadReadSpan(src, offset, copy_bytes);
        impl->NoteAheadTexture(dst_root, true);
    } else {
        if ((ahead || auto_hoist) && impl->render != nil && CanUploadAhead()) {
            if (dst_root != nullptr && !impl->IsPassTouchedRoot(dst_root) &&
                !impl->PendingClearOnRoot(dst_root)) {
                WantSplitBeforeCurrentPass();
            }
        }
        impl->Touch(dst);
        impl->NoteInOrder(src, offset, copy_bytes);
        if (!src_range_tracked) {
            impl->NoteUntrackedInOrderRead(src, offset, copy_bytes);
        }
        impl->NoteTextureWrite(dst);
    }
    const id<MTLBlitCommandEncoder> __unsafe_unretained encoder =
        goes_ahead ? impl->UploadEncoder() : impl->BlitEncoder();
    [encoder copyFromBuffer:UnretainedBuffer(src)
               sourceOffset:offset
          sourceBytesPerRow:bytes_per_row
        sourceBytesPerImage:bytes_per_image
                 sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                  toTexture:mtl_dst
           destinationSlice:slice
           destinationLevel:level
          destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)];
}

void Scheduler::CopyTextureToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin,
                                    Extent3D extent, BufferHandle dst, u64 offset,
                                    u32 bytes_per_row, u32 bytes_per_image,
                                    [[maybe_unused]] bool dst_range_tracked) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    const id<MTLTexture> __unsafe_unretained mtl_src = UnretainedTexture(src);
    const u64 copy_bytes = TextureCopyBufferBytes(mtl_src, bytes_per_row, bytes_per_image, extent);
    impl->FlushDeferredDownloadsFor(dst, offset, copy_bytes, false);
    impl->Touch(src);
    impl->NoteTextureTouched(src);
    impl->NoteInOrderWrite(dst, offset, copy_bytes);
    const id<MTLBlitCommandEncoder> __unsafe_unretained encoder = impl->BlitEncoder();
    [encoder copyFromTexture:mtl_src
                 sourceSlice:slice
                 sourceLevel:level
                sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                  sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                    toBuffer:UnretainedBuffer(dst)
           destinationOffset:offset
      destinationBytesPerRow:bytes_per_row
    destinationBytesPerImage:bytes_per_image];
}

void Scheduler::CopyTexture(TextureHandle src, u32 src_level, u32 src_slice, Offset3D src_origin,
                            Extent3D extent, TextureHandle dst, u32 dst_level, u32 dst_slice,
                            Offset3D dst_origin, bool ahead) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    const void* const src_root = impl->CachedRootTexture(src);
    const void* const dst_root = impl->CachedRootTexture(dst);
    if (src_root != dst_root) {
        impl->DropMipClear(dst, dst_level, dst_slice, dst_origin, extent);
    }
    const bool src_needs_restore = impl->NeedsSampleRestoreRoot(src_root);
    const bool src_clear_ok =
        !impl->PendingClearOnRoot(src_root) || impl->CanFlushClearAheadForRoot(src_root);
    const ComputeHazardState::BufferSpan src_bspan = ComputeHazardState::BackingBufferSpan(src_root);
    const ComputeHazardState::BufferSpan dst_bspan = ComputeHazardState::BackingBufferSpan(dst_root);
    const u64 src_bsz = src_bspan.end == ~u64{0} ? ~u64{0} : (src_bspan.end - src_bspan.begin);
    const u64 dst_bsz = dst_bspan.end == ~u64{0} ? ~u64{0} : (dst_bspan.end - dst_bspan.begin);
    const bool backing_ok =
        ahead ? ((src_bspan.buffer == nullptr ||
                  CanReadBufferAhead(const_cast<void*>(src_bspan.buffer), src_bspan.begin,
                                     src_bsz)) &&
                 (dst_bspan.buffer == nullptr ||
                  CanWriteBufferAhead(const_cast<void*>(dst_bspan.buffer), dst_bspan.begin,
                                      dst_bsz)))
              : (src_bspan.buffer == nullptr && dst_bspan.buffer == nullptr);
    const bool ahead_order_ok =
        impl->merge_ahead_enabled ||
        (!ComputeHazardState::Contains(impl->ahead_written_textures, src_root) &&
         !ComputeHazardState::Contains(impl->ahead_touched_textures, dst_root));
    const bool want_hoist = ahead || (impl->render != nil && impl->merge_ahead_enabled);
    const bool goes_ahead = want_hoist && backing_ok && ahead_order_ok &&
                            CanUploadTextureAhead(dst) && impl->WantsAheadEncoder() &&
                            src_root != nullptr && src_root != dst_root && !src_needs_restore &&
                            src_clear_ok && !impl->segment_written_textures.contains(src_root);
    if (goes_ahead) {
        if (impl->PendingClearOnRoot(src_root)) {
            impl->FlushClearAheadForRoot(src_root);
        }
        if (impl->PendingClearOnRoot(dst_root)) {
            impl->FlushClearAheadForRoot(dst_root);
        }
        impl->NoteAheadTexture(src_root, false);
        impl->NoteAheadTexture(dst_root, true);
        impl->WrittenRoot(dst, dst_root);
        const id<MTLBlitCommandEncoder> __unsafe_unretained encoder = impl->UploadEncoder();
        [encoder copyFromTexture:UnretainedTexture(src)
                     sourceSlice:src_slice
                     sourceLevel:src_level
                    sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                      sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                       toTexture:UnretainedTexture(dst)
                destinationSlice:dst_slice
                destinationLevel:dst_level
               destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
        return;
    }
    if ((ahead || (src_bspan.buffer == nullptr && dst_bspan.buffer == nullptr)) &&
        impl->render != nil && CanUploadAhead() && src_root != nullptr && dst_root != nullptr &&
        !impl->IsPassTouchedRoot(src_root) && !impl->IsPassTouchedRoot(dst_root) &&
        !impl->PendingClearOnRoot(src_root) && !impl->PendingClearOnRoot(dst_root)) {
        WantSplitBeforeCurrentPass();
    }
    impl->SamplesNeeded(src);
    impl->WrittenRoot(dst, dst_root);
    impl->Touch(src);
    impl->Touch(dst);
    impl->NoteTextureWrite(dst);
    const id<MTLBlitCommandEncoder> __unsafe_unretained encoder = impl->BlitEncoder();
    [encoder copyFromTexture:UnretainedTexture(src)
                 sourceSlice:src_slice
                 sourceLevel:src_level
                sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                  sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                   toTexture:UnretainedTexture(dst)
            destinationSlice:dst_slice
            destinationLevel:dst_level
           destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
}

void Scheduler::BreakBlits() {
    impl->EndEncoder();
}

void Scheduler::Clear(const ClearPass& clear) {
    if (!impl->clear_fold_enabled) {
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
    if (impl->render != nil && impl->depth_kill_enabled &&
        EffectiveTargetLayers(impl->render_targets) <= 1) {
        const RenderTargetSet& open = impl->render_targets;
        const auto whole = [&wanted](const Attachment& attachment) {
            const id<MTLTexture> __unsafe_unretained texture =
                UnretainedTexture(attachment.texture);
            if (texture == nil) {
                return false;
            }
            const auto [mip_w, mip_h] = MipExtent(texture, attachment.level);
            return wanted.width >= mip_w && wanted.height >= mip_h;
        };
        for (size_t index = 0; index < wanted.color.size(); ++index) {
            if (!wanted.clear_color[index] || wanted.color[index].texture == nullptr) {
                continue;
            }
            for (size_t rt = 0; rt < open.color.size(); ++rt) {
                if (open.color[rt].texture != nullptr &&
                    SameAttachmentSubresource(wanted.color[index], open.color[rt]) &&
                    whole(wanted.color[index])) {
                    impl->kill_color |= static_cast<u8>(1U << rt);
                }
            }
        }
        if (wanted.clear_depth && open.has_depth &&
            SameAttachmentSubresource(wanted.depth, open.depth) && whole(wanted.depth)) {
            impl->kill_depth = true;
        }
        if (wanted.clear_stencil && open.has_stencil &&
            SameAttachmentSubresource(wanted.stencil, open.depth) && whole(wanted.stencil)) {
            impl->kill_stencil = true;
        }
    }
    Impl::PendingClearEntry wanted_entry{.clear = wanted};
    wanted_entry.Refresh();
    // The draws and dispatches recorded so far come before it.
    if (impl->render != nil) {
        const u32 eff_open_layers = EffectiveTargetLayers(impl->render_targets);
        const auto open_subs = impl->TargetSubresourcesOf(impl->render_targets);
        bool must_end_render = false;
        for (const SubresourceKey& sub : wanted_entry.subs) {
            if (sub.root == nullptr) {
                continue;
            }
            if (Impl::SubresourceMatchesTargets(sub, open_subs, eff_open_layers) ||
                impl->IsPassSampledRoot(sub.root)) {
                must_end_render = true;
                break;
            }
            if (impl->IsPassTouchedRoot(sub.root)) {
                wanted_entry.blocked_ahead = true;
            }
        }
        if (must_end_render) {
            wanted_entry.blocked_ahead = true;
            impl->EndEncoder(PassEnd::Clear);
        }
    }
    if (!wanted_entry.blocked_ahead) {
        for (const SubresourceKey& sub : wanted_entry.subs) {
            if (sub.root != nullptr && impl->segment_touched_textures.contains(sub.root)) {
                wanted_entry.blocked_ahead = true;
                break;
            }
        }
    }
    size_t conflict_count = 0;
    size_t conflict_idx = 0;
    for (size_t i = 0; i < impl->pending_clears.size(); ++i) {
        if (impl->pending_clears[i].SharesAnySubresourceWith(wanted_entry.subs)) {
            ++conflict_count;
            conflict_idx = i;
        }
    }
    if (conflict_count == 1) {
        if (MergeClear(impl->pending_clears[conflict_idx].clear, wanted)) {
            impl->pending_clears[conflict_idx].blocked_ahead =
                impl->pending_clears[conflict_idx].blocked_ahead || wanted_entry.blocked_ahead;
            impl->pending_clears[conflict_idx].Refresh();
            impl->RebuildPendingClearBloom();
            return;
        }
    }
    if (conflict_count != 0) {
        std::vector<Impl::PendingClearEntry> to_flush;
        std::erase_if(impl->pending_clears, [&](Impl::PendingClearEntry& entry) {
            if (entry.SharesAnySubresourceWith(wanted_entry.subs)) {
                to_flush.push_back(std::move(entry));
                return true;
            }
            return false;
        });
        impl->RebuildPendingClearBloom();
        for (const Impl::PendingClearEntry& entry : to_flush) {
            impl->RecordClear(entry.clear);
        }
    }
    if (impl->pending_clears.size() >= Impl::MaxPendingClears) {
        for (size_t i = impl->pending_clears.size(); i-- > 0;) {
            if (impl->pending_clears[i].blocked_ahead == wanted_entry.blocked_ahead &&
                MergeClear(impl->pending_clears[i].clear, wanted)) {
                impl->pending_clears[i].Refresh();
                impl->RebuildPendingClearBloom();
                return;
            }
        }
        Impl::PendingClearEntry oldest = std::move(impl->pending_clears.front());
        impl->pending_clears.erase(impl->pending_clears.begin());
        impl->RebuildPendingClearBloom();
        impl->RecordClear(oldest.clear);
    }
    if (impl->render != nil) {
        impl->last_midpass_clear_pass_serial = impl->render_pass_serial;
    }
    impl->pending_clear_roots_bloom |= wanted_entry.roots_bloom;
    impl->pending_clears.push_back(std::move(wanted_entry));
}

bool Scheduler::ClearRectangle(const ClearRect& clear) {
    if (impl->TryDrawInPassClearRect(clear)) {
        return true;
    }
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

    @autoreleasepool {
        MTLRenderPassDescriptor* const pass = impl->helper_pass_desc.Acquire();
        if (color != nil) {
            MTLRenderPassColorAttachmentDescriptor* const c0 = impl->helper_pass_desc.Color(0);
            ApplyAttachment(c0, clear.color);
            c0.loadAction = MTLLoadActionLoad;
            c0.storeAction = MTLStoreActionStore;
        }
        if (depth != nil && depth_format != MTLPixelFormatStencil8) {
            MTLRenderPassDepthAttachmentDescriptor* const d0 = impl->helper_pass_desc.Depth();
            ApplyAttachment(d0, clear.depth);
            d0.loadAction = MTLLoadActionLoad;
            d0.storeAction = MTLStoreActionStore;
        }
        if (stencil) {
            MTLRenderPassStencilAttachmentDescriptor* const s0 = impl->helper_pass_desc.Stencil();
            ApplyAttachment(s0, clear.depth);
            s0.loadAction = MTLLoadActionLoad;
            s0.storeAction = MTLStoreActionStore;
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
        [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                         static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
        [encoder setScissorRect:MTLScissorRect{x, y, w, h}];
        const f32 depth_value = std::clamp(clear.depth_value, 0.0f, 1.0f);
        [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
        if (color != nil) {
            const std::array<u32, 4> bits =
                PackClearColorWords(clear.color_value, static_cast<u32>(color_format));
            [encoder setFragmentBytes:bits.data() length:sizeof(bits) atIndex:0];
        }
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        return true;
    }
}

bool Scheduler::Blit(const BlitPass& blit) {
    const id<MTLTexture> src = AsTexture(blit.src);
    const id<MTLTexture> dst_tex = AsTexture(blit.dst.texture);
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(blit.dst_format);
    const id<MTLRenderPipelineState> pipeline =
        (src != nil && dst_tex != nil) ? impl->BlitPipeline(format, blit.blend, blit.color_op)
                                       : nil;
    if (pipeline == nil) {
        impl->Written(blit.dst.texture);
        impl->SamplesNeeded(blit.src);
        impl->ClearBefore({blit.src, blit.dst.texture});
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
    const s32 limit_w = static_cast<s32>(blit.dst_width);
    const s32 limit_h = static_cast<s32>(blit.dst_height);
    const s32 sx = std::clamp(x, 0, limit_w);
    const s32 sy = std::clamp(y, 0, limit_h);
    const s32 sw = std::clamp(x + w, 0, limit_w) - sx;
    const s32 sh = std::clamp(y + h, 0, limit_h) - sy;
    if (w == 0 || h == 0 || ((sw <= 0 || sh <= 0) && blit.keep_dst)) {
        impl->Written(blit.dst.texture);
        impl->SamplesNeeded(blit.src);
        impl->ClearBefore({blit.src});
        return true;
    }

    @autoreleasepool {
        const FormatClass format_class = ClassOf(blit.dst_format);
        const bool depth =
            format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
        id<MTLTexture> src_stencil = nil;
        if (format_class == FormatClass::DepthStencil &&
            StencilViewFormat(format) != MTLPixelFormatInvalid) {
            src_stencil = StencilView(src);
            if (src_stencil == nil) {
                impl->Written(blit.dst.texture);
                impl->SamplesNeeded(blit.src);
                impl->ClearBefore({blit.src, blit.dst.texture});
                return false;
            }
        }
        const auto [mip_w, mip_h] = MipExtent(dst_tex, blit.dst.level);
        const void* const src_root = impl->CachedRootTexture(blit.src);
        const void* const dst_root = impl->CachedRootTexture(blit.dst.texture);
        const bool covers_whole =
            sw > 0 && sh > 0 && blit.blend == BlitBlend::None && sx == 0 && sy == 0 &&
            static_cast<u32>(sw) >= mip_w && static_cast<u32>(sh) >= mip_h;
        if (covers_whole && src_root != dst_root) {
            impl->DropSubresourceClear(blit.dst, !depth, depth,
                                       format_class == FormatClass::DepthStencil &&
                                           src_stencil != nil);
        }
        const bool goes_ahead =
            impl->CanRunHelperPassAhead(src_root, blit.dst.texture, dst_root);
        if (!goes_ahead) {
            if (impl->render != nil && CanUploadAhead() && src_root != nullptr &&
                dst_root != nullptr && !impl->IsPassTouchedRoot(src_root) &&
                !impl->IsPassTouchedRoot(dst_root) && !impl->PendingClearOnRoot(src_root) &&
                !impl->PendingClearOnRoot(dst_root) &&
                ComputeHazardState::BackingBufferSpan(src_root).buffer == nullptr &&
                ComputeHazardState::BackingBufferSpan(dst_root).buffer == nullptr) {
                WantSplitBeforeCurrentPass();
            }
            impl->WrittenRoot(blit.dst.texture, dst_root);
            impl->SamplesNeeded(blit.src);
            impl->ClearBefore({blit.src});
        } else {
            if (impl->PendingClearOnRoot(src_root)) {
                impl->FlushClearAheadForRoot(src_root);
            }
            impl->WrittenRoot(blit.dst.texture, dst_root);
        }

        std::optional<ClearPass> folded_clear;
        if (!impl->pending_clears.empty() && src_root != dst_root) {
            if (!impl->clear_narrow_enabled) {
                impl->FlushClear();
            } else if (!covers_whole && blit.keep_dst && impl->PendingClearOnRoot(dst_root)) {
                RenderTargetSet blit_targets{};
                blit_targets.width = mip_w;
                blit_targets.height = mip_h;
                blit_targets.layers = 1;
                if (depth) {
                    blit_targets.depth = blit.dst;
                    blit_targets.has_depth = true;
                    blit_targets.has_stencil = format_class == FormatClass::DepthStencil;
                } else {
                    blit_targets.color[0] = blit.dst;
                }
                std::erase_if(impl->pending_clears, [&](Impl::PendingClearEntry& entry) {
                    if (!CanFoldClearInto(entry.clear, blit_targets)) {
                        return false;
                    }
                    bool has_remainder = false;
                    const ClearPass consumed =
                        ConsumeClearSubset(entry.clear, blit_targets, has_remainder);
                    if (!folded_clear) {
                        folded_clear = consumed;
                    } else {
                        MergeConsumedClear(*folded_clear, consumed);
                    }
                    if (!has_remainder) {
                        return true;
                    }
                    entry.Refresh();
                    return false;
                });
                if (folded_clear) {
                    impl->RebuildPendingClearBloom();
                    ++impl->clears_folded;
                }
            }
            if (impl->PendingClearOnRoot(dst_root)) {
                if (goes_ahead) {
                    impl->FlushClearAheadForSubresource(SubresourceOf(blit.dst));
                } else {
                    impl->FlushClearForSubresource(SubresourceOf(blit.dst));
                }
            }
        }

        const bool can_reuse_blit =
            !goes_ahead && impl->blit_render != nil && blit.keep_dst && !folded_clear &&
            src_root != nullptr && src_root != dst_root &&
            blit.dst_format == impl->blit_dst_format &&
            SameAttachmentSubresource(blit.dst, impl->blit_dst);
        id<MTLRenderCommandEncoder> encoder = can_reuse_blit ? impl->blit_render : nil;
        if (encoder == nil) {
            MTLRenderPassDescriptor* const pass = impl->helper_pass_desc.Acquire();
            if (depth) {
                MTLRenderPassDepthAttachmentDescriptor* const d0 = impl->helper_pass_desc.Depth();
                ApplyAttachment(d0, blit.dst);
                if (covers_whole) {
                    d0.loadAction = MTLLoadActionDontCare;
                } else if (folded_clear && folded_clear->clear_depth) {
                    d0.loadAction = MTLLoadActionClear;
                    d0.clearDepth = folded_clear->depth_value;
                } else {
                    d0.loadAction = blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
                }
                d0.storeAction = MTLStoreActionStore;
                if (format_class == FormatClass::DepthStencil) {
                    MTLRenderPassStencilAttachmentDescriptor* const s0 =
                        impl->helper_pass_desc.Stencil();
                    ApplyAttachment(s0, blit.dst);
                    if (covers_whole && src_stencil != nil) {
                        s0.loadAction = MTLLoadActionDontCare;
                    } else if (folded_clear && folded_clear->clear_stencil) {
                        s0.loadAction = MTLLoadActionClear;
                        s0.clearStencil = folded_clear->stencil_value;
                    } else {
                        s0.loadAction = blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
                    }
                    s0.storeAction = MTLStoreActionStore;
                }
            } else {
                MTLRenderPassColorAttachmentDescriptor* const c0 = impl->helper_pass_desc.Color(0);
                ApplyAttachment(c0, blit.dst);
                if (covers_whole) {
                    c0.loadAction = MTLLoadActionDontCare;
                } else if (folded_clear && folded_clear->clear_color[0]) {
                    const auto& cv = folded_clear->color_value[0];
                    c0.loadAction = MTLLoadActionClear;
                    c0.clearColor = MTLClearColorMake(cv[0], cv[1], cv[2], cv[3]);
                } else if (blit.keep_dst) {
                    c0.loadAction = MTLLoadActionLoad;
                } else {
                    c0.loadAction = MTLLoadActionClear;
                    c0.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
                }
                c0.storeAction = MTLStoreActionStore;
            }
            pass.renderTargetWidth = mip_w;
            pass.renderTargetHeight = mip_h;
            if (goes_ahead) {
                impl->NoteAheadTexture(src_root, false);
                impl->NoteAheadTexture(dst_root, true);
                encoder = impl->AheadRenderEncoder(pass);
                if (encoder == nil) {
                    return false;
                }
            } else {
                encoder = impl->RenderEncoder(pass);
                if (encoder == nil) {
                    return false;
                }
                impl->blit_render = encoder;
                impl->blit_dst = blit.dst;
                impl->blit_dst_format = blit.dst_format;
            }
        }
        if (sw <= 0 || sh <= 0) {
            if (goes_ahead) {
                [encoder endEncoding];
            }
            return true;
        }
        [encoder setRenderPipelineState:pipeline];
        if (depth) {
            [encoder setDepthStencilState:format_class == FormatClass::DepthStencil
                                              ? impl->DepthWriteAlways()
                                              : impl->ClearDepthState(true, false, 0)];
            [encoder setFragmentTexture:src_stencil atIndex:1];
        }
        [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                         static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
        [encoder setScissorRect:MTLScissorRect{static_cast<NSUInteger>(sx),
                                               static_cast<NSUInteger>(sy),
                                               static_cast<NSUInteger>(sw),
                                               static_cast<NSUInteger>(sh)}];
        [encoder setVertexBytes:uv.data() length:sizeof(uv) atIndex:0];
        [encoder setFragmentTexture:src atIndex:0];
        [encoder setFragmentSamplerState:blit.linear ? impl->linear : impl->nearest atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        if (goes_ahead) {
            [encoder endEncoding];
        }
        return true;
    }
}

bool Scheduler::CopySamples(const SampleCopy& copy, bool expand) {
    const id<MTLTexture> src = AsTexture(copy.src);
    const id<MTLTexture> dst = AsTexture(copy.dst.texture);
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(copy.dst_format);
    const id<MTLRenderPipelineState> pipeline =
        (src != nil && dst != nil && copy.samples != 0)
            ? impl->SamplePipeline(format, dst.sampleCount, expand)
            : nil;
    const u32 width = std::clamp(copy.dst_width, 1U, MaxRenderTargetSize);
    const u32 height = std::clamp(copy.dst_height, 1U, MaxRenderTargetSize);
    if (pipeline != nil && RootTextureHandle(copy.src) != RootTextureHandle(copy.dst.texture)) {
        const auto [mip_w, mip_h] = MipExtent(dst, copy.dst.level);
        if (width >= mip_w && height >= mip_h) {
            const FormatClass fmt_class = ClassOf(copy.dst_format);
            const bool is_depth =
                fmt_class == FormatClass::Depth || fmt_class == FormatClass::DepthStencil;
            impl->DropSubresourceClear(copy.dst, !is_depth, is_depth, false);
        }
    }
    impl->SamplesNeeded(copy.src);
    impl->Written(copy.dst.texture);
    impl->ClearBefore({copy.src, copy.dst.texture});
    if (pipeline == nil) {
        return false;
    }
    @autoreleasepool {
        MTLRenderPassDescriptor* const pass = impl->helper_pass_desc.Acquire();
        const FormatClass format_class = ClassOf(copy.dst_format);
        const bool depth =
            format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
        if (depth) {
            MTLRenderPassDepthAttachmentDescriptor* const d0 = impl->helper_pass_desc.Depth();
            ApplyAttachment(d0, copy.dst);
            d0.loadAction = MTLLoadActionDontCare;
            d0.storeAction = MTLStoreActionStore;
            if (format_class == FormatClass::DepthStencil) {
                // The stencil stays as it was: only depth is copied.
                MTLRenderPassStencilAttachmentDescriptor* const s0 =
                    impl->helper_pass_desc.Stencil();
                ApplyAttachment(s0, copy.dst);
                s0.loadAction = MTLLoadActionLoad;
                s0.storeAction = MTLStoreActionStore;
            }
        } else {
            MTLRenderPassColorAttachmentDescriptor* const c0 = impl->helper_pass_desc.Color(0);
            ApplyAttachment(c0, copy.dst);
            c0.loadAction = MTLLoadActionDontCare;
            c0.storeAction = MTLStoreActionStore;
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
        [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width),
                                         static_cast<f64>(height), 0.0, 1.0}];
        const f32 depth_value = 0.0f;
        [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
        const std::array<u32, 4> params{copy.shift[0], copy.shift[1], copy.samples, copy.src_layer};
        [encoder setFragmentBytes:params.data() length:sizeof(params) atIndex:0];
        [encoder setFragmentTexture:src atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        return true;
    }
}

bool Scheduler::Convert(const ConvertPass& convert) {
    const id<MTLTexture> src = AsTexture(convert.src);
    const id<MTLTexture> dst_tex = AsTexture(convert.dst.texture);
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(convert.dst_format);
    const id<MTLRenderPipelineState> pipeline =
        (src != nil && dst_tex != nil) ? impl->ConvertPipeline(convert.kind, format) : nil;
    if (pipeline == nil) {
        impl->Written(convert.dst.texture);
        impl->SamplesNeeded(convert.src);
        impl->ClearBefore({convert.src, convert.dst.texture});
        return false;
    }
    @autoreleasepool {
        id<MTLTexture> src_stencil = nil;
        if (convert.kind == ConvertKind::D24s8ToAbgr8 ||
            convert.kind == ConvertKind::S8d24ToAbgr8) {
            src_stencil = StencilView(src);
            if (src_stencil == nil) {
                impl->Written(convert.dst.texture);
                impl->SamplesNeeded(convert.src);
                impl->ClearBefore({convert.src, convert.dst.texture});
                return false;
            }
        }
        const u32 width = std::clamp(convert.width, 1U, MaxRenderTargetSize);
        const u32 height = std::clamp(convert.height, 1U, MaxRenderTargetSize);
        const auto [mip_w, mip_h] = MipExtent(dst_tex, convert.dst.level);
        const FormatClass format_class = ClassOf(convert.dst_format);
        const bool depth =
            format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
        const void* const src_root = impl->CachedRootTexture(convert.src);
        const void* const dst_root = impl->CachedRootTexture(convert.dst.texture);
        const bool covers_whole = src_root != dst_root && width >= mip_w && height >= mip_h;
        if (covers_whole) {
            impl->DropSubresourceClear(convert.dst, !depth, depth,
                                       convert.kind == ConvertKind::Abgr8ToD24s8);
        }
        const bool goes_ahead =
            impl->CanRunHelperPassAhead(src_root, convert.dst.texture, dst_root);
        if (goes_ahead) {
            if (impl->PendingClearOnRoot(src_root)) {
                impl->FlushClearAheadForRoot(src_root);
            }
            if (impl->PendingClearOnRoot(dst_root)) {
                impl->FlushClearAheadForSubresource(SubresourceOf(convert.dst));
            }
            impl->WrittenRoot(convert.dst.texture, dst_root);
        } else {
            impl->WrittenRoot(convert.dst.texture, dst_root);
            impl->SamplesNeeded(convert.src);
            impl->ClearBefore({convert.src, convert.dst.texture});
        }
        MTLRenderPassDescriptor* const pass = impl->helper_pass_desc.Acquire();
        if (depth) {
            MTLRenderPassDepthAttachmentDescriptor* const d0 = impl->helper_pass_desc.Depth();
            ApplyAttachment(d0, convert.dst);
            d0.loadAction = covers_whole ? MTLLoadActionDontCare : MTLLoadActionLoad;
            d0.storeAction = MTLStoreActionStore;
            if (format_class == FormatClass::DepthStencil) {
                MTLRenderPassStencilAttachmentDescriptor* const s0 =
                    impl->helper_pass_desc.Stencil();
                ApplyAttachment(s0, convert.dst);
                s0.loadAction = (covers_whole && convert.kind == ConvertKind::Abgr8ToD24s8)
                                    ? MTLLoadActionDontCare
                                    : MTLLoadActionLoad;
                s0.storeAction = MTLStoreActionStore;
            }
        } else {
            MTLRenderPassColorAttachmentDescriptor* const c0 = impl->helper_pass_desc.Color(0);
            ApplyAttachment(c0, convert.dst);
            c0.loadAction = covers_whole ? MTLLoadActionDontCare : MTLLoadActionLoad;
            c0.storeAction = MTLStoreActionStore;
        }
        pass.renderTargetWidth = width;
        pass.renderTargetHeight = height;
        if (goes_ahead) {
            impl->NoteAheadTexture(src_root, false);
            impl->NoteAheadTexture(dst_root, true);
        }
        const id<MTLRenderCommandEncoder> encoder =
            goes_ahead ? impl->AheadRenderEncoder(pass) : impl->RenderEncoder(pass);
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
        [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width),
                                         static_cast<f64>(height), 0.0, 1.0}];
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
            @autoreleasepool {
                MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
                desc.label = @"Depth upload";
                desc.vertexFunction = [state.library newFunctionWithName:@"clear_vertex"];
                desc.fragmentFunction = [state.library
                    newFunctionWithName:stencil ? @"depth_stencil_from_buffer"
                                                : @"depth_from_buffer"];
                desc.depthAttachmentPixelFormat = dst.pixelFormat;
                if (stencil) {
                    desc.stencilAttachmentPixelFormat = dst.pixelFormat;
                }
                NSError* error = nil;
                pipeline = [state.device.GetDevice() newRenderPipelineStateWithDescriptor:desc
                                                                                    error:&error];
                if (pipeline == nil) {
                    LOG_ERROR(Render_Metal, "Metal: depth upload pipeline failed: {}",
                              error != nil ? error.localizedDescription.UTF8String : "no reason");
                }
            }
        }
        it = state.depth_upload_pipelines.emplace(key, pipeline).first;
    }
    if (it->second == nil) {
        return false;
    }
    const auto [width, height] = MipExtent(dst, level);
    const u32 x = (std::min)(origin.x, width);
    const u32 y = (std::min)(origin.y, height);
    const u32 w = (std::min)(extent.width, width - x);
    const u32 h = (std::min)(extent.height, height - y);
    if (w == 0 || h == 0) {
        return true;
    }
    @autoreleasepool {
        MTLRenderPassDescriptor* const pass = state.helper_pass_desc.Acquire();
        MTLRenderPassDepthAttachmentDescriptor* const d0 = state.helper_pass_desc.Depth();
        ApplyAttachment(d0, Attachment{.texture = dst_handle, .level = level, .slice = slice});
        d0.loadAction = MTLLoadActionLoad;
        d0.storeAction = MTLStoreActionStore;
        if (stencil) {
            MTLRenderPassStencilAttachmentDescriptor* const s0 = state.helper_pass_desc.Stencil();
            ApplyAttachment(s0, Attachment{.texture = dst_handle, .level = level, .slice = slice});
            s0.loadAction = MTLLoadActionLoad;
            s0.storeAction = MTLStoreActionStore;
        }
        pass.renderTargetWidth = width;
        pass.renderTargetHeight = height;
        const u64 read_bytes = u64{bytes_per_row} * h;
        state.NoteInOrder(src, offset, read_bytes);
        state.NoteUntrackedInOrderRead(src, offset, read_bytes);
        const id<MTLRenderCommandEncoder> encoder = state.RenderEncoder(pass);
        if (encoder == nil) {
            return false;
        }
        [encoder setRenderPipelineState:it->second];
        [encoder setDepthStencilState:stencil ? state.DepthWriteAlways()
                                              : state.ClearDepthState(true, false, 0)];
        [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                         static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
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
    @autoreleasepool {
        if (state.depth_download == nil && state.library != nil) {
            NSError* error = nil;
            state.depth_download = [state.device.GetDevice()
                newComputePipelineStateWithFunction:[state.library
                                                        newFunctionWithName:@"depth_to_buffer"]
                                              error:&error];
            if (state.depth_download == nil) {
                LOG_ERROR(Render_Metal, "Metal: depth download kernel failed: {}",
                          error != nil ? error.localizedDescription.UTF8String : "no reason");
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
        const u64 write_bytes = u64{bytes_per_row} * extent.height;
        state.FlushDeferredDownloadsFor(dst, offset, write_bytes, false);
        state.NoteTextureTouched(src_handle);
        state.NoteInOrderWrite(dst, offset, write_bytes);
        state.OpenInOrderCompute();
        const id<MTLComputeCommandEncoder> __unsafe_unretained encoder = state.compute;
        state.compute_hazards.ResetPending();
        state.compute_hazards.NoteTexture(src_handle, false, false);
        state.compute_hazards.NoteBuffer(dst, true, offset, 0, write_bytes);
        state.compute_hazards.BeforeDispatch(encoder, state.concurrent_compute_enabled);
        state.compute_pipeline = (__bridge const void*)state.depth_download;
        state.compute_bound.textures[0] = &ClobberedSlot;
        state.compute_bound.textures[1] = &ClobberedSlot;
        state.compute_bound.buffers[0] = &ClobberedSlot;
        state.compute_bound.buffers[1] = &ClobberedSlot;
        state.compute_bound.ClearTextureDirty(0);
        state.compute_bound.ClearTextureDirty(1);
        state.compute_bound.ClearBufferDirty(0);
        state.compute_bound.ClearBufferDirty(1);
        [encoder setComputePipelineState:state.depth_download];
        [encoder setTexture:depth atIndex:0];
        [encoder setTexture:stencil atIndex:1];
        [encoder setBuffer:UnretainedBuffer(dst) offset:offset atIndex:0];
        const std::array<u32, 8> params{origin.x, origin.y, bytes_per_row / 4,
                                        static_cast<u32>(layout), extent.width, extent.height,
                                        has_stencil ? 1U : 0U, 0};
        [encoder setBytes:params.data() length:sizeof(params) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake((extent.width + 7) / 8, (extent.height + 7) / 8,
                                                  1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        state.compute_hazards.AfterDispatch(state.concurrent_compute_enabled);
        state.CountDispatch();
        return true;
    }
}

bool Scheduler::UpdateSparseTiles(TextureHandle texture_handle,
                                  std::span<const SparseTileOp> ops) {
    const id<MTLTexture> texture = AsTexture(texture_handle);
    if (texture == nil || ops.empty()) {
        return texture != nil;
    }
    @autoreleasepool {
        impl->ClearBefore({texture_handle});
        impl->NoteTextureWrite(texture_handle);
        impl->EndEncoder();
        ++impl->encoders;
        ++impl->segment_encoders;
        const id<MTLResourceStateCommandEncoder> encoder =
            [impl->Commands() resourceStateCommandEncoder];
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
    @autoreleasepool {
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
                          error != nil ? error.localizedDescription.UTF8String : "no reason");
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
        const u64 copy_bytes = u64{bytes_per_row} * extent.height;
        if (to_buffer) {
            state.FlushDeferredDownloadsFor(buffer, offset, copy_bytes, false);
            state.NoteTextureTouched(texture_handle);
            state.NoteInOrderWrite(buffer, offset, copy_bytes);
        } else {
            state.NoteTextureWrite(texture_handle);
            state.NoteInOrder(buffer, offset, copy_bytes);
            state.NoteUntrackedInOrderRead(buffer, offset, copy_bytes);
        }
        state.OpenInOrderCompute();
        const id<MTLComputeCommandEncoder> __unsafe_unretained encoder = state.compute;
        state.compute_hazards.ResetPending();
        state.compute_hazards.NoteTexture(texture_handle, false, !to_buffer);
        state.compute_hazards.NoteBuffer(buffer, to_buffer, offset,
                                         to_buffer ? u64{0} : copy_bytes,
                                         to_buffer ? copy_bytes : u64{0});
        state.compute_hazards.BeforeDispatch(encoder, state.concurrent_compute_enabled);
        state.compute_pipeline = (__bridge const void*)pipeline;
        state.compute_bound.textures[0] = &ClobberedSlot;
        state.compute_bound.buffers[0] = &ClobberedSlot;
        state.compute_bound.buffers[1] = &ClobberedSlot;
        state.compute_bound.ClearTextureDirty(0);
        state.compute_bound.ClearBufferDirty(0);
        state.compute_bound.ClearBufferDirty(1);
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:view atIndex:0];
        [encoder setBuffer:UnretainedBuffer(buffer) offset:offset atIndex:0];
        const std::array<u32, 6> params{origin.x, origin.y, extent.width, extent.height,
                                        bytes_per_row / 2, mode};
        [encoder setBytes:params.data() length:sizeof(params) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake((extent.width + 7) / 8, (extent.height + 7) / 8,
                                                  1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        state.compute_hazards.AfterDispatch(state.concurrent_compute_enabled);
        state.CountDispatch();
        return true;
    }
}

bool Scheduler::SwapPacked(BufferHandle buffer, u64 offset, u32 width, u32 rows, u32 images,
                           u32 bytes_per_row, u32 bytes_per_image, u32 shift) {
    if (buffer == nullptr || width == 0 || rows == 0 || images == 0 || offset % 2 != 0 ||
        offset + u64{images} * bytes_per_image > UINT32_MAX) {
        return false;
    }
    @autoreleasepool {
        impl->ClearBefore({});
        Impl& state = *impl;
        if (state.swap_packed == nil && state.library != nil) {
            NSError* error = nil;
            state.swap_packed = [state.device.GetDevice()
                newComputePipelineStateWithFunction:[state.library
                                                        newFunctionWithName:@"swap_packed"]
                                              error:&error];
            if (state.swap_packed == nil) {
                LOG_ERROR(Render_Metal, "Metal: packed texel swap kernel failed: {}",
                          error != nil ? error.localizedDescription.UTF8String : "no reason");
            }
        }
        if (state.swap_packed == nil) {
            return false;
        }
        const u64 swap_bytes =
            (images > 1 ? u64{images - 1} * bytes_per_image : u64{0}) + u64{rows} * bytes_per_row;
        state.FlushDeferredDownloadsFor(buffer, offset, swap_bytes, false);
        state.NoteInOrderWrite(buffer, offset, swap_bytes);
        state.OpenInOrderCompute();
        const id<MTLComputeCommandEncoder> __unsafe_unretained encoder = state.compute;
        state.compute_hazards.ResetPending();
        state.compute_hazards.NoteBuffer(buffer, true, offset, swap_bytes, swap_bytes);
        state.compute_hazards.BeforeDispatch(encoder, state.concurrent_compute_enabled);
        state.compute_pipeline = (__bridge const void*)state.swap_packed;
        state.compute_bound.buffers[0] = &ClobberedSlot;
        state.compute_bound.buffers[1] = &ClobberedSlot;
        state.compute_bound.ClearBufferDirty(0);
        state.compute_bound.ClearBufferDirty(1);
        [encoder setComputePipelineState:state.swap_packed];
        [encoder setBuffer:UnretainedBuffer(buffer) offset:0 atIndex:0];
        const std::array<u32, 6> params{static_cast<u32>(offset), width, rows, bytes_per_row,
                                        bytes_per_image, shift};
        [encoder setBytes:params.data() length:sizeof(params) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake((width + 31) / 32, (rows + 7) / 8, images)
                threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        state.compute_hazards.AfterDispatch(state.concurrent_compute_enabled);
        state.CountDispatch();
        return true;
    }
}

bool Scheduler::ScaleNearest(const ScalePass& scale) {
    const id<MTLTexture> src = AsTexture(scale.src);
    const id<MTLTexture> dst = AsTexture(scale.dst.texture);
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(scale.dst_format);
    const id<MTLRenderPipelineState> pipeline =
        (src != nil && dst != nil && scale.src_width != 0 && scale.src_height != 0)
            ? impl->ScalePipeline(format, dst.sampleCount)
            : nil;
    const u32 width = std::clamp(scale.dst_width, 1U, MaxRenderTargetSize);
    const u32 height = std::clamp(scale.dst_height, 1U, MaxRenderTargetSize);
    const void* const src_root = impl->CachedRootTexture(scale.src);
    const void* const dst_root = impl->CachedRootTexture(scale.dst.texture);
    if (pipeline != nil && src_root != dst_root) {
        const auto [mip_w, mip_h] = MipExtent(dst, scale.dst.level);
        if (width >= mip_w && height >= mip_h) {
            const FormatClass fmt_class = ClassOf(scale.dst_format);
            impl->DropSubresourceClear(scale.dst,
                                       fmt_class != FormatClass::Depth &&
                                           fmt_class != FormatClass::Stencil,
                                       fmt_class == FormatClass::Depth,
                                       fmt_class == FormatClass::Stencil);
        }
    }
    const bool goes_ahead =
        pipeline != nil && impl->CanRunHelperPassAhead(src_root, scale.dst.texture, dst_root);
    if (goes_ahead) {
        if (impl->PendingClearOnRoot(src_root)) {
            impl->FlushClearAheadForRoot(src_root);
        }
        if (impl->PendingClearOnRoot(dst_root)) {
            impl->FlushClearAheadForSubresource(SubresourceOf(scale.dst));
        }
        impl->WrittenRoot(scale.dst.texture, dst_root);
    } else {
        impl->SamplesNeeded(scale.src);
        impl->WrittenRoot(scale.dst.texture, dst_root);
        impl->ClearBefore({scale.src, scale.dst.texture});
    }
    if (pipeline == nil) {
        return false;
    }
    @autoreleasepool {
        MTLRenderPassDescriptor* const pass = impl->helper_pass_desc.Acquire();
        const FormatClass format_class = ClassOf(scale.dst_format);
        MTLRenderPassAttachmentDescriptor* attachment = nil;
        if (format_class == FormatClass::Depth) {
            attachment = impl->helper_pass_desc.Depth();
        } else if (format_class == FormatClass::Stencil) {
            attachment = impl->helper_pass_desc.Stencil();
        } else {
            attachment = impl->helper_pass_desc.Color(0);
        }
        ApplyAttachment(attachment, scale.dst);
        attachment.loadAction = MTLLoadActionDontCare;
        attachment.storeAction = MTLStoreActionStore;
        pass.renderTargetWidth = width;
        pass.renderTargetHeight = height;
        if (goes_ahead) {
            impl->NoteAheadTexture(src_root, false);
            impl->NoteAheadTexture(dst_root, true);
        }
        const id<MTLRenderCommandEncoder> encoder =
            goes_ahead ? impl->AheadRenderEncoder(pass) : impl->RenderEncoder(pass);
        if (encoder == nil) {
            return false;
        }
        [encoder setRenderPipelineState:pipeline];
        if (format_class == FormatClass::Depth) {
            [encoder setDepthStencilState:impl->ClearDepthState(true, false, 0)];
        } else if (format_class == FormatClass::Stencil) {
            [encoder setDepthStencilState:impl->ClearDepthState(false, true, 0xFF)];
        }
        [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(width),
                                         static_cast<f64>(height), 0.0, 1.0}];
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
}

bool Scheduler::Resolve(TextureHandle src_handle, u32 src_slice, const Attachment& dst, u32 width,
                        u32 height) {
    const id<MTLTexture> src = AsTexture(src_handle);
    const id<MTLTexture> target = AsTexture(dst.texture);
    const bool valid = src != nil && target != nil && src.sampleCount >= 2 &&
                       target.sampleCount == 1 && src.pixelFormat == target.pixelFormat;
    const FormatClass format_class =
        valid ? ClassOf(static_cast<u32>(src.pixelFormat)) : FormatClass::Stencil;
    if (valid && format_class != FormatClass::Stencil &&
        RootTextureHandle(src_handle) != RootTextureHandle(dst.texture)) {
        const auto [mip_w, mip_h] = MipExtent(target, dst.level);
        if (std::clamp(width, 1U, MaxRenderTargetSize) >= mip_w &&
            std::clamp(height, 1U, MaxRenderTargetSize) >= mip_h) {
            const bool is_depth =
                format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
            impl->DropSubresourceClear(dst, !is_depth, is_depth, false);
        }
    }
    impl->ClearBefore({src_handle, dst.texture});
    impl->SamplesNeeded(src_handle);
    if (!valid || format_class == FormatClass::Stencil) {
        return false;
    }
    @autoreleasepool {
        MTLRenderPassDescriptor* const pass = impl->helper_pass_desc.Acquire();
        MTLRenderPassAttachmentDescriptor* attachment = nil;
        if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
            MTLRenderPassDepthAttachmentDescriptor* const depth = impl->helper_pass_desc.Depth();
            depth.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
            attachment = depth;
        } else {
            attachment = impl->helper_pass_desc.Color(0);
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
}

void Scheduler::BeginRendering(const RenderTargetSet& targets) {
    impl->BeginRenderingImpl(targets);
}

void Scheduler::Impl::BeginRenderingImpl(const RenderTargetSet& targets) {
    if (MatchesOpenTargets(targets)) {
        return;
    }
    Impl* const self = this;
    @autoreleasepool {
        // Consume all foldable subsets across `pending_clears` into this pass's load actions;
        // keep remaining unconsumed entries deferred across the pass unless they overlap `targets`.
        const u32 eff_layers = EffectiveTargetLayers(targets);
        std::optional<ClearPass> clear;
        if (!self->pending_clears.empty()) {
            if (eff_layers > 1) {
                self->TryFoldLayeredClears(clear, targets, eff_layers);
            }
            const auto target_subs = self->TargetSubresourcesOf(targets);
            std::vector<Impl::PendingClearEntry> keep;
            std::vector<Impl::PendingClearEntry> to_flush;
            keep.reserve(self->pending_clears.size());
            for (Impl::PendingClearEntry& entry : self->pending_clears) {
                if (eff_layers <= 1 && CanFoldClearInto(entry.clear, targets)) {
                    bool has_remainder = false;
                    const ClearPass consumed =
                        ConsumeClearSubset(entry.clear, targets, has_remainder);
                    if (!clear) {
                        clear = consumed;
                    } else {
                        MergeConsumedClear(*clear, consumed);
                    }
                    if (!has_remainder) {
                        continue;
                    }
                    entry.Refresh();
                }
                if (self->CanDeferEntryAcross(entry, target_subs, eff_layers)) {
                    keep.push_back(std::move(entry));
                } else {
                    to_flush.push_back(std::move(entry));
                }
            }
            self->pending_clears = std::move(keep);
            self->RebuildPendingClearBloom();
            if (clear) {
                ++self->clears_folded;
            }
            for (const Impl::PendingClearEntry& entry : to_flush) {
                self->RecordClear(entry.clear);
            }
        }
        if (!self->shadows.empty()) {
            for (size_t index = 0; index < targets.color.size(); ++index) {
                if (targets.color[index].texture == nullptr) {
                    continue;
                }
                const bool cleared = clear && clear->clear_color[index];
                if (!cleared) {
                    if (const auto shadow = self->shadows.find(targets.color[index].texture);
                        shadow != self->shadows.end() && shadow->second.dropped) {
                        self->RestoreSamples(shadow->first, shadow->second);
                    }
                }
            }
        }
        const bool defer_stores = self->unwritten_stores_enabled;
        u8 defer_color = 0;
        u8 kill_open_color = 0;
        bool defer_depth = false;
        bool defer_stencil = false;
        bool kill_open_depth = false;
        bool kill_open_stencil = false;
        MTLRenderPassDescriptor* const pass = self->draw_pass_desc.Acquire();
        for (size_t index = 0; index < targets.color.size(); ++index) {
            if (targets.color[index].texture == nullptr) {
                continue;
            }
            MTLRenderPassColorAttachmentDescriptor* const __unsafe_unretained color =
                self->draw_pass_desc.Color(index);
            ApplyAttachment(color, targets.color[index]);
            color.loadAction = MTLLoadActionLoad;
            color.storeAction = MTLStoreActionStore;
            const bool cleared = clear && clear->clear_color[index];
            // Q27: a multisampled colour target with a resolve shadow resolves in the pass.
            bool resolving = false;
            if (!self->shadows.empty()) {
                if (const auto shadow = self->shadows.find(targets.color[index].texture);
                    shadow != self->shadows.end()) {
                    Impl::ShadowEntry& entry = shadow->second;
                    if (targets.layers == 1) {
                        resolving = true;
                        color.resolveTexture = UnretainedTexture(entry.shadow);
                        color.resolveLevel = 0;
                        color.resolveSlice = targets.color[index].slice;
                        const bool drop = self->msaa_discard_enabled && cleared;
                        color.storeAction = drop ? MTLStoreActionMultisampleResolve
                                                 : MTLStoreActionStoreAndMultisampleResolve;
                        entry.valid = true;
                        entry.dropped = drop;
                        if (drop) {
                            ++self->samples_dropped;
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
                if (defer_stores && self->depth_kill_enabled && !resolving) {
                    color.storeAction = MTLStoreActionUnknown;
                    kill_open_color |= static_cast<u8>(1U << index);
                }
            } else if (defer_stores && !resolving) {
                color.storeAction = MTLStoreActionUnknown;
                defer_color |= static_cast<u8>(1U << index);
            }
        }
        if (targets.has_depth) {
            MTLRenderPassDepthAttachmentDescriptor* const __unsafe_unretained depth =
                self->draw_pass_desc.Depth();
            ApplyAttachment(depth, targets.depth);
            depth.loadAction = MTLLoadActionLoad;
            depth.storeAction = MTLStoreActionStore;
            if (clear && clear->clear_depth) {
                depth.loadAction = MTLLoadActionClear;
                depth.clearDepth = clear->depth_value;
                if (defer_stores && self->depth_kill_enabled) {
                    depth.storeAction = MTLStoreActionUnknown;
                    kill_open_depth = true;
                }
            } else if (defer_stores) {
                depth.storeAction = MTLStoreActionUnknown;
                defer_depth = true;
            }
        }
        if (targets.has_stencil) {
            MTLRenderPassStencilAttachmentDescriptor* const __unsafe_unretained stencil =
                self->draw_pass_desc.Stencil();
            ApplyAttachment(stencil, targets.depth);
            stencil.loadAction = MTLLoadActionLoad;
            stencil.storeAction = MTLStoreActionStore;
            if (clear && clear->clear_stencil) {
                stencil.loadAction = MTLLoadActionClear;
                stencil.clearStencil = clear->stencil_value;
                if (defer_stores && self->depth_kill_enabled) {
                    stencil.storeAction = MTLStoreActionUnknown;
                    kill_open_stencil = true;
                }
            } else if (defer_stores) {
                stencil.storeAction = MTLStoreActionUnknown;
                defer_stencil = true;
            }
        }
        // Draws without attachments (occlusion queries) still need a size, within Metal's limit.
        const auto [eff_w, eff_h] = EffectivePassExtent(targets);
        pass.renderTargetWidth = eff_w;
        pass.renderTargetHeight = eff_h;
        const bool targetless = !targets.has_depth && !targets.has_stencil &&
                                std::ranges::all_of(targets.color, [](const Attachment& color) {
                                    return color.texture == nullptr;
                                });
        if (targetless && self->pipeline_check_enabled) {
            // Without attachments nothing else tells Metal the pass has one sample per pixel, as
            // the pipelines drawn into it have (MoltenVK sets it for such a pass too).
            pass.defaultRasterSampleCount = 1;
        }
        if (targets.layers > 1 && !targetless) {
            if (eff_layers > 1) {
                pass.renderTargetArrayLength = eff_layers;
            } else {
                static std::atomic<bool> logged{};
                if (!logged.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Metal, "Metal: a pass of {} layers has an attachment of one",
                             targets.layers);
                }
            }
        }
        pass.visibilityResultBuffer = self->visibility;
        self->render = self->RenderEncoder(pass, PassEnd::NewPass);
        self->in_render_pass.store(self->render != nil, std::memory_order_relaxed);
        self->deferred_color = defer_color;
        self->cleared_color_open = self->render != nil ? kill_open_color : 0;
        self->deferred_depth = defer_depth;
        self->deferred_stencil = defer_stencil;
        self->cleared_depth_open = kill_open_depth && self->render != nil;
        self->cleared_stencil_open = kill_open_stencil && self->render != nil;
        self->kill_color = 0;
        self->kill_depth = false;
        self->kill_stencil = false;
        self->written_color = 0;
        self->written_depth = false;
        self->written_stencil = false;
        u8 cleared_colors = 0;
        if (clear) {
            for (size_t i = 0; i < clear->clear_color.size(); ++i) {
                if (clear->clear_color[i] && targets.color[i].texture != nullptr) {
                    cleared_colors |= static_cast<u8>(1U << i);
                }
            }
        }
        self->color_cleared_mask = cleared_colors;
        self->pass_state_set = false;
        self->depth_stencil_cleared = clear && (clear->clear_depth || clear->clear_stencil);
        self->render_targets = targets;
        self->render_targets_raw_width = targets.width;
        self->render_targets_raw_height = targets.height;
        self->render_targets.width = eff_w;
        self->render_targets.height = eff_h;
        self->pass_touched_roots.clear();
        self->pass_sampled_roots.clear();
        self->pass_touched_bloom = 0;
        self->pass_sampled_bloom = 0;
        for (const Attachment& c : targets.color) {
            if (const void* const r = self->CachedRootTexture(c.texture); r != nullptr) {
                self->AddPassTouchedRoot(r);
            }
        }
        if ((targets.has_depth || targets.has_stencil) && targets.depth.texture != nullptr) {
            if (const void* const r = self->CachedRootTexture(targets.depth.texture);
                r != nullptr) {
                self->AddPassTouchedRoot(r);
            }
        }
        self->visibility_slot = -1;
        self->vertex_bound = {};
        self->fragment_bound = {};
        self->object_bound = {};
        self->mesh_bound = {};
        for (auto& vec : self->pending_render_read_use) {
            vec.clear();
        }
        self->pass_used_resources.clear();
        self->render_state = {};
        self->render_state_valid = false;
        ++self->render_pass_serial;
        ++self->visibility_serial;
    }
}

bool Scheduler::IsRendering() const noexcept {
    return impl->render != nil;
}

const RenderTargetSet& Scheduler::OpenRenderTargets() const noexcept {
    return impl->render_targets;
}

bool Scheduler::HasPendingClearFor(const RenderTargetSet& targets) const noexcept {
    if (impl->pending_clears.empty()) {
        return false;
    }
    const auto target_subs = impl->TargetSubresourcesOf(targets);
    for (const Impl::PendingClearEntry& entry : impl->pending_clears) {
        for (const SubresourceKey& sub : entry.subs) {
            if (sub.root != nullptr &&
                Impl::SubresourceMatchesTargets(sub, target_subs, targets.layers)) {
                return true;
            }
        }
    }
    return false;
}

bool Scheduler::HasPendingClears() const noexcept {
    return !impl->pending_clears.empty();
}

void Scheduler::PrepareSampledTextureForPass(TextureHandle texture) {
    if (impl->pending_clear_roots_bloom != 0 && texture != nullptr) {
        const void* const want_root = impl->CachedRootTexture(texture);
        if (impl->PendingClearOnRoot(want_root)) {
            impl->FlushClearAheadForRoot(want_root);
        }
    }
}

void Scheduler::FlushPendingClearsForTargets(const RenderTargetSet& targets) {
    if (impl->pending_clears.empty()) {
        return;
    }
    const u32 eff_layers = EffectiveTargetLayers(targets);
    const auto target_subs = impl->TargetSubresourcesOf(targets);
    std::vector<Impl::PendingClearEntry> to_flush;
    std::erase_if(impl->pending_clears, [&](Impl::PendingClearEntry& entry) {
        for (const SubresourceKey& sub : entry.subs) {
            if (sub.root != nullptr &&
                Impl::SubresourceMatchesTargets(sub, target_subs, eff_layers)) {
                to_flush.push_back(std::move(entry));
                return true;
            }
        }
        return false;
    });
    if (!to_flush.empty()) {
        impl->RebuildPendingClearBloom();
        const bool ahead_ok = impl->ahead_enabled && impl->merge_ahead_enabled &&
                              !impl->ahead_blocked && !impl->segment_opaque &&
                              impl->WantsAheadEncoder();
        @autoreleasepool {
            for (const Impl::PendingClearEntry& entry : to_flush) {
                if (ahead_ok && !entry.blocked_ahead) {
                    impl->RecordClearOnAheadBuffer(entry.clear);
                } else {
                    impl->RecordClear(entry.clear);
                }
            }
        }
    }
}

bool Scheduler::RenderPassWroteDepthStencil() const noexcept {
    // Clears outside the load actions end the encoder first (Clear, ClearRectangle); every draw
    // sets its state, defaulting to "writes everything", before it is encoded.
    return impl->render == nil || impl->depth_stencil_cleared || impl->written_depth ||
           impl->written_stencil;
}

u8 Scheduler::RenderPassWroteColorMask() const noexcept {
    return impl->render == nil ? 0xFFU
                               : static_cast<u8>(impl->color_cleared_mask | impl->written_color);
}

void Scheduler::EndRendering() {
    if (impl->render != nil || impl->blit_render != nil) {
        impl->EndEncoder(PassEnd::EndRendering);
    }
}

void Scheduler::SetVisibility(std::optional<u32> slot) {
    const s64 wanted = slot ? static_cast<s64>(*slot) : -1;
    if (impl->render == nil || wanted == impl->visibility_slot) {
        return;
    }
    if (impl->visibility_slot != -1) {
        ++impl->visibility_serial;
    }
    impl->visibility_slot = wanted;
    if (slot) {
        impl->NoteInOrderWrite((__bridge void*)impl->visibility, u64{*slot} * sizeof(u64),
                               sizeof(u64));
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

void Scheduler::DropPendingClears(TextureHandle texture) {
    if (impl->pending_clear_roots_bloom == 0 || texture == nullptr) {
        return;
    }
    const void* const want_root = RootTextureHandle(texture);
    if (want_root == nullptr ||
        (impl->pending_clear_roots_bloom & PtrBloomBit(want_root)) == 0) {
        return;
    }
    std::erase_if(impl->pending_clears, [want_root](Impl::PendingClearEntry& entry) {
        if (!entry.TouchesRoot(want_root)) {
            return false;
        }
        bool any_left = false;
        for (size_t i = 0; i < entry.clear.color.size(); ++i) {
            if (!entry.clear.clear_color[i]) {
                continue;
            }
            if (entry.subs[i].root == want_root) {
                entry.clear.color[i] = {};
                entry.clear.clear_color[i] = false;
            } else {
                any_left = true;
            }
        }
        if (entry.clear.clear_depth) {
            if (entry.subs[8].root == want_root) {
                entry.clear.depth = {};
                entry.clear.clear_depth = false;
            } else {
                any_left = true;
            }
        }
        if (entry.clear.clear_stencil) {
            if (entry.subs[9].root == want_root) {
                entry.clear.stencil = {};
                entry.clear.clear_stencil = false;
            } else {
                any_left = true;
            }
        }
        if (any_left) {
            entry.Refresh();
            return false;
        }
        return true;
    });
    impl->RebuildPendingClearBloom();
}

TextureHandle Scheduler::ResolveShadow(TextureHandle msaa) const noexcept {
    if (impl->shadows.empty() || msaa == nullptr) {
        return nullptr;
    }
    const void* const root = RootTextureHandle(msaa);
    const auto it = impl->shadows.find(const_cast<void*>(root != nullptr ? root : msaa));
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

u64 Scheduler::VisibilitySerial() const noexcept {
    return impl->visibility_serial;
}

void Scheduler::SetDrawState(const DrawState& state) {
    impl->SetDrawStateImpl(state);
}

void Scheduler::Impl::SetDrawStateImpl(const DrawState& state) {
    Impl* const self = this;
    const id<MTLRenderCommandEncoder> __unsafe_unretained encoder = self->render;
    if (encoder == nil) {
        return;
    }
    self->pass_state_set = true;
    const u8 newly_written_color = static_cast<u8>(state.color_written & ~self->written_color);
    if (newly_written_color != 0) {
        u8 bits = newly_written_color;
        while (bits != 0) {
            const u32 rt = static_cast<u32>(std::countr_zero(bits));
            bits &= bits - 1U;
            self->NoteTextureWrite(self->render_targets.color[rt].texture);
        }
    }
    if ((state.depth_written && !self->written_depth) ||
        (state.stencil_written && !self->written_stencil)) {
        self->NoteTextureWrite(self->render_targets.depth.texture);
    }
    self->written_color |= state.color_written;
    self->written_depth = self->written_depth || state.depth_written;
    self->written_stencil = self->written_stencil || state.stencil_written;
    // What the encoder has from the draw before stays as it is (EDEN_NX2_METAL_STATE_CACHE).
    const bool cached = self->draw_state_cache_enabled && self->render_state_valid;
    const DrawState& last = self->render_state;
    if (cached && state.viewport_count == 1 && last.viewport_count == 1 &&
        last.pipeline == state.pipeline &&
        (state.depth_stencil == nullptr || last.depth_stencil == state.depth_stencil) &&
        last.viewport == state.viewport && last.scissor == state.scissor &&
        last.fill_lines == state.fill_lines && last.cull_mode == state.cull_mode &&
        last.front_ccw == state.front_ccw && last.depth_clamp == state.depth_clamp &&
        last.depth_bias == state.depth_bias && last.slope_scale == state.slope_scale &&
        last.bias_clamp == state.bias_clamp &&
        (!self->depth_bounds || (last.depth_bounds_min == state.depth_bounds_min &&
                                 last.depth_bounds_max == state.depth_bounds_max)) &&
        last.stencil_front_ref == state.stencil_front_ref &&
        last.stencil_back_ref == state.stencil_back_ref &&
        last.blend_color == state.blend_color) {
        self->state_skips += 10U + (state.depth_stencil != nullptr ? 1U : 0U) +
                             (self->depth_bounds ? 1U : 0U);
        return;
    }
    bool any_changed = !cached;
    const auto changed = [&](bool differs) {
        if (cached && !differs) {
            ++self->state_skips;
            return false;
        }
        any_changed = true;
        ++self->state_sets;
        return true;
    };
    if (changed(last.pipeline != state.pipeline)) {
        [encoder setRenderPipelineState:(__bridge __unsafe_unretained id<MTLRenderPipelineState>)
                                            state.pipeline];
    }
    if (state.depth_stencil != nullptr && changed(last.depth_stencil != state.depth_stencil)) {
        [encoder setDepthStencilState:(__bridge __unsafe_unretained id<MTLDepthStencilState>)
                                          state.depth_stencil];
    }
    const auto to_viewport = [](const Viewport& viewport) {
        return MTLViewport{viewport.x,      viewport.y,     viewport.width,
                           viewport.height, viewport.znear, viewport.zfar};
    };
    // Metal rejects scissors past the attachments; BeginRenderingImpl already clamped
    // render_targets.width/height to the effective mip extent.
    const u32 target_width = (std::max)(self->render_targets.width, 1U);
    const u32 target_height = (std::max)(self->render_targets.height, 1U);
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
    if (self->depth_bounds && changed(last.depth_bounds_min != state.depth_bounds_min ||
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
    if (!any_changed) {
        return;
    }
    // A draw without a depth-stencil state leaves the encoder's.
    void* const depth_stencil =
        state.depth_stencil != nullptr ? state.depth_stencil : last.depth_stencil;
    if (state.viewport_count <= 1 && self->render_state.viewport_count <= 1) {
        DrawState& dst = self->render_state;
        dst.pipeline = state.pipeline;
        dst.depth_stencil = depth_stencil;
        dst.viewport = state.viewport;
        dst.scissor = state.scissor;
        dst.cull_mode = state.cull_mode;
        dst.front_ccw = state.front_ccw;
        dst.depth_clamp = state.depth_clamp;
        dst.depth_bias = state.depth_bias;
        dst.slope_scale = state.slope_scale;
        dst.bias_clamp = state.bias_clamp;
        dst.stencil_front_ref = state.stencil_front_ref;
        dst.stencil_back_ref = state.stencil_back_ref;
        dst.blend_color = state.blend_color;
        dst.viewport_count = state.viewport_count;
        dst.fill_lines = state.fill_lines;
        dst.depth_bounds_min = state.depth_bounds_min;
        dst.depth_bounds_max = state.depth_bounds_max;
        dst.color_written = state.color_written;
        dst.depth_written = state.depth_written;
        dst.stencil_written = state.stencil_written;
    } else {
        self->render_state = state;
        self->render_state.depth_stencil = depth_stencil;
    }
    self->render_state_valid = true;
}

void Scheduler::BindVertexBuffer(u32 slot, BufferHandle buffer, u32 offset) {
    if (SlotFits(slot, BufferSlots, "vertex buffer")) {
        if (!impl->vertex_input_cache_enabled) {
            // Bound again whatever the encoder has.
            impl->vertex_bound.buffers[slot] = &InlineBytes;
            impl->vertex_bound.ClearBufferDirty(slot);
        }
        impl->SetBuffer(ShaderStage::Vertex, slot, buffer, offset);
    }
}

void Scheduler::BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset, bool write,
                           u64 hazard_read_size, u64 hazard_write_size, bool disjoint_write) {
    if (SlotFits(slot, BufferSlots, "buffer")) {
        if (stage == ShaderStage::Compute) {
            if (impl->recording_ahead_wave >= 0) {
                const u64 eff_write =
                    hazard_write_size != 0 ? hazard_write_size : hazard_read_size;
                const u64 conflict_size =
                    write ? (std::max)(hazard_read_size, eff_write) : hazard_read_size;
                if (impl->ahead_wave_checked &&
                    impl->CheckedConflict(buffer, offset, conflict_size, write)) {
                    impl->AbortRecordingAheadWave(nullptr);
                } else {
                    if (buffer != nullptr) {
                        impl->NoteActiveBuffer(buffer);
                        impl->NoteAheadReadSpan(buffer, offset, hazard_read_size);
                        if (hazard_read_size != 0) {
                            ComputeHazardState::AddSpan(
                                impl->ahead_wave_read_spans,
                                {buffer, offset,
                                 ComputeHazardState::SaturatingEnd(offset, hazard_read_size)});
                        }
                        if (write) {
                            impl->NoteAheadWriteSpan(buffer, offset, eff_write);
                        }
                    }
                    auto& q = impl->ahead_waves[static_cast<size_t>(impl->recording_ahead_wave)];
                    q.binds.push_back({ComputeBind::Kind::Buffer, true, write, slot, buffer, offset,
                                       0, hazard_read_size, hazard_write_size, true,
                                       disjoint_write});
                    return;
                }
            }
            impl->ComputeBuffer(slot, buffer, write, offset, hazard_read_size, hazard_write_size,
                                true, disjoint_write);
            if (impl->active_compute != nil && impl->active_compute == impl->kernel_compute) {
                impl->kernel_binding = true;
            }
            impl->LogComputeBind(ComputeBind::Kind::Buffer, slot, buffer, offset, write, true,
                                 hazard_read_size, hazard_write_size, true, disjoint_write);
        } else if (write && impl->render != nil) {
            const u64 eff_write =
                hazard_write_size != 0 ? hazard_write_size : hazard_read_size;
            impl->FlushDeferredDownloadsFor(buffer, offset, eff_write, true);
            if (impl->render != nil) {
                impl->NoteInOrderWrite(buffer, offset, eff_write);
            }
        }
        impl->SetBuffer(stage, slot, buffer, offset);
    }
}

u64 Scheduler::UseBuffer(ShaderStage stage, BufferHandle buffer, u32 offset, bool write,
                         u64 known_gpu_address, u64 hazard_read_size, u64 hazard_write_size) {
    if (buffer == nullptr) {
        return 0;
    }
    const MTLResourceUsage usage =
        write ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
    if (stage == ShaderStage::Compute) {
        if (impl->recording_ahead_wave >= 0) {
            const u64 eff_write = hazard_write_size != 0 ? hazard_write_size : hazard_read_size;
            const u64 conflict_size =
                write ? (std::max)(hazard_read_size, eff_write) : hazard_read_size;
            if (impl->ahead_wave_checked &&
                impl->CheckedConflict(buffer, offset, conflict_size, write)) {
                impl->AbortRecordingAheadWave(nullptr);
            } else {
                impl->NoteActiveBuffer(buffer);
                impl->NoteAheadReadSpan(buffer, offset, hazard_read_size);
                if (hazard_read_size != 0) {
                    ComputeHazardState::AddSpan(
                        impl->ahead_wave_read_spans,
                        {buffer, offset,
                         ComputeHazardState::SaturatingEnd(offset, hazard_read_size)});
                }
                if (write) {
                    impl->NoteAheadWriteSpan(buffer, offset, eff_write);
                }
                auto& q = impl->ahead_waves[static_cast<size_t>(impl->recording_ahead_wave)];
                q.binds.push_back({ComputeBind::Kind::UseBuffer, true, write, 0, buffer, offset, 0,
                                   hazard_read_size, hazard_write_size, true, false});
                const u64 base_addr =
                    known_gpu_address != 0
                        ? known_gpu_address
                        : static_cast<u64>(UnretainedBuffer(buffer).gpuAddress);
                return base_addr + offset;
            }
        }
        impl->ComputeBuffer(std::nullopt, buffer, write, offset, hazard_read_size,
                            hazard_write_size);
        if (impl->active_compute != nil && impl->active_compute == impl->kernel_compute) {
            impl->kernel_binding = true;
        }
        impl->LogComputeBind(ComputeBind::Kind::UseBuffer, 0, buffer, offset, write, true,
                             hazard_read_size, hazard_write_size);
        const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = impl->active_compute;
        const u64 enc_serial = c_enc == nil                 ? 0
                               : c_enc == impl->compute     ? impl->in_order_compute_serial
                                                            : impl->kernel_compute_serial;
        Impl::UseResourceNote& cached =
            impl->compute_use_cache[PtrHashIndex(buffer, Impl::ComputeUseCacheBits)];
        if (impl->state_cache_enabled && c_enc != nil && cached.resource == buffer &&
            cached.encoder_serial == enc_serial &&
            ((write ? cached.write_stages : cached.read_stages) & 1U) != 0) {
            return cached.id_or_addr + offset;
        }
        const id<MTLBuffer> __unsafe_unretained metal_buffer = UnretainedBuffer(buffer);
        const u64 base_addr =
            known_gpu_address != 0 ? known_gpu_address : static_cast<u64>(metal_buffer.gpuAddress);
        if (c_enc != nil) {
            if (!write && impl->range_binds_enabled && impl->recorder == nullptr) {
                impl->ActiveComputeReadUse().push_back(buffer);
            } else {
                [c_enc useResource:metal_buffer usage:usage];
            }
            const bool same_enc = cached.resource == buffer && cached.encoder_serial == enc_serial;
            cached = {
                buffer,
                enc_serial,
                base_addr,
                1U,
                static_cast<u8>((same_enc ? cached.write_stages : 0U) | (write ? 1U : 0U)),
                0,
                0,
            };
        }
        return base_addr + offset;
    }

    if (impl->render != nil && write) {
        const u64 eff_write = hazard_write_size != 0 ? hazard_write_size : hazard_read_size;
        impl->FlushDeferredDownloadsFor(buffer, offset, eff_write, true);
        if (impl->render != nil) {
            impl->NoteInOrderWrite(buffer, offset, eff_write);
        }
    }
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    const u8 stage_bit = static_cast<u8>(
        stage == ShaderStage::Vertex     ? 1U
        : stage == ShaderStage::Fragment ? 2U
        : stage == ShaderStage::Object   ? 4U
                                         : 8U);
    Impl::UseResourceNote& cached =
        impl->render_use_cache[PtrHashIndex(buffer, Impl::RenderUseCacheBits)];
    if (impl->state_cache_enabled && r_enc != nil && cached.resource == buffer &&
        cached.encoder_serial == impl->render_pass_serial &&
        ((write ? cached.write_stages : cached.read_stages) & stage_bit) != 0) {
        return cached.id_or_addr + offset;
    }
    if (r_enc != nil) {
        impl->NoteActiveBuffer(buffer);
    }
    const id<MTLBuffer> __unsafe_unretained metal_buffer = UnretainedBuffer(buffer);
    const u64 base_addr =
        known_gpu_address != 0 ? known_gpu_address : static_cast<u64>(metal_buffer.gpuAddress);
    if (r_enc != nil) {
        const MTLRenderStages mtl_stage =
            stage == ShaderStage::Vertex     ? MTLRenderStageVertex
            : stage == ShaderStage::Fragment ? MTLRenderStageFragment
            : stage == ShaderStage::Object   ? MTLRenderStageObject
                                             : MTLRenderStageMesh;
        impl->pass_used_resources.push_back({buffer, usage, mtl_stage});
        if (!write && impl->range_binds_enabled && impl->recorder == nullptr) {
            const size_t stage_idx =
                stage == ShaderStage::Vertex     ? 0
                : stage == ShaderStage::Fragment ? 1
                : stage == ShaderStage::Object   ? 2
                                                 : 3;
            impl->pending_render_read_use[stage_idx].push_back(buffer);
        } else {
            [r_enc useResource:metal_buffer usage:usage stages:mtl_stage];
        }
        const bool same_pass =
            cached.resource == buffer && cached.encoder_serial == impl->render_pass_serial;
        cached = {
            buffer,
            impl->render_pass_serial,
            base_addr,
            static_cast<u8>((same_pass ? cached.read_stages : 0U) | stage_bit),
            static_cast<u8>((same_pass ? cached.write_stages : 0U) | (write ? stage_bit : 0U)),
            0,
            0,
        };
    }
    return base_addr + offset;
}

void Scheduler::BindBytes(ShaderStage stage, u32 slot, const void* data, u32 size) {
    if (!SlotFits(slot, BufferSlots, "buffer")) {
        return;
    }
    const bool caching = impl->state_cache_enabled;
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    switch (stage) {
    case ShaderStage::Vertex:
        if (r_enc == nil) {
            return;
        }
        if (caching &&
            impl->vertex_bound.MatchesOrRecordInline(slot, data, size, &InlineBytes)) {
            ++impl->state_skips;
            return;
        }
        if (!caching && slot < BufferSlots) {
            impl->vertex_bound.buffers[slot] = &InlineBytes;
        }
        ++impl->state_sets;
        [r_enc setVertexBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Fragment:
        if (r_enc == nil) {
            return;
        }
        if (caching &&
            impl->fragment_bound.MatchesOrRecordInline(slot, data, size, &InlineBytes)) {
            ++impl->state_skips;
            return;
        }
        if (!caching && slot < BufferSlots) {
            impl->fragment_bound.buffers[slot] = &InlineBytes;
        }
        ++impl->state_sets;
        [r_enc setFragmentBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Object:
        if (r_enc == nil) {
            return;
        }
        if (caching &&
            impl->object_bound.MatchesOrRecordInline(slot, data, size, &InlineBytes)) {
            ++impl->state_skips;
            return;
        }
        if (!caching && slot < BufferSlots) {
            impl->object_bound.buffers[slot] = &InlineBytes;
        }
        ++impl->state_sets;
        [r_enc setObjectBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Mesh:
        if (r_enc == nil) {
            return;
        }
        if (caching &&
            impl->mesh_bound.MatchesOrRecordInline(slot, data, size, &InlineBytes)) {
            ++impl->state_skips;
            return;
        }
        if (!caching && slot < BufferSlots) {
            impl->mesh_bound.buffers[slot] = &InlineBytes;
        }
        ++impl->state_sets;
        [r_enc setMeshBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Compute: {
        if (impl->recording_ahead_wave >= 0) {
            auto& q = impl->ahead_waves[static_cast<size_t>(impl->recording_ahead_wave)];
            const u32 at = static_cast<u32>(q.bytes.size());
            const u8* const bytes = static_cast<const u8*>(data);
            q.bytes.insert(q.bytes.end(), bytes, bytes + size);
            q.binds.push_back(
                {ComputeBind::Kind::Bytes, false, false, slot, nullptr, at, size});
            return;
        }
        impl->ResumeKernel();
        if (impl->active_hazards != nullptr) {
            impl->active_hazards->ClearSlotBuffer(slot);
        }
        if (impl->active_compute != nil && impl->active_compute == impl->kernel_compute) {
            impl->kernel_binding = true;
        }
        impl->LogComputeBytes(slot, data, size);
        const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = impl->active_compute;
        if (c_enc == nil) {
            return;
        }
        if (caching &&
            impl->active_bound->MatchesOrRecordInline(slot, data, size, &InlineBytes)) {
            ++impl->state_skips;
            return;
        }
        if (!caching && slot < BufferSlots) {
            impl->active_bound->buffers[slot] = &InlineBytes;
        }
        ++impl->state_sets;
        [c_enc setBytes:data length:size atIndex:slot];
        break;
    }
    }
}

void Scheduler::BindTexture(ShaderStage stage, u32 slot, TextureHandle texture, bool on_buffer,
                            bool write, bool track_hazard) {
    if (!SlotFits(slot, TextureSlots, "texture")) {
        return;
    }
    if (stage == ShaderStage::Compute) {
        if (impl->recording_ahead_wave >= 0) {
            bool conflict = false;
            if (texture != nullptr && track_hazard) {
                const void* const raw_root = impl->CachedRootTexture(texture);
                const u64 restored_before = impl->samples_restored;
                if (write) {
                    impl->WrittenRoot(texture, raw_root);
                } else {
                    impl->RestoreSamplesIfDropped(texture, raw_root);
                }
                const bool restored = impl->samples_restored != restored_before;
                const ComputeHazardState::BufferSpan bspan =
                    on_buffer ? ComputeHazardState::BackingBufferSpan(raw_root)
                              : ComputeHazardState::BufferSpan{};
                const u64 bsz = bspan.end == ~u64{0} ? ~u64{0} : (bspan.end - bspan.begin);
                if (restored || impl->PendingClearOnRoot(raw_root) ||
                    (impl->ahead_wave_checked &&
                     ((write ? impl->segment_touched_textures : impl->segment_written_textures)
                          .contains(raw_root) ||
                      (bspan.buffer != nullptr &&
                       impl->CheckedConflict(bspan.buffer, bspan.begin, bsz, write))))) {
                    conflict = true;
                    impl->AbortRecordingAheadWave(texture);
                } else {
                    impl->NoteAheadTexture(raw_root, write);
                    if (bspan.buffer != nullptr && bspan.end > bspan.begin) {
                        impl->NoteActiveBuffer(bspan.buffer);
                        impl->NoteAheadReadSpan(bspan.buffer, bspan.begin, bsz);
                        ComputeHazardState::AddSpan(impl->ahead_wave_read_spans, bspan);
                        if (write) {
                            impl->NoteAheadWriteSpan(bspan.buffer, bspan.begin, bsz);
                        }
                    }
                }
            }
            if (!conflict) {
                auto& q = impl->ahead_waves[static_cast<size_t>(impl->recording_ahead_wave)];
                q.binds.push_back({ComputeBind::Kind::Texture, on_buffer, write, slot, texture, 0,
                                   0, ~u64{0}, ~u64{0}, track_hazard, false});
                return;
            }
        }
        impl->ComputeTexture(slot, texture, on_buffer, write, track_hazard);
        if (impl->active_compute != nil && impl->active_compute == impl->kernel_compute) {
            impl->kernel_binding = true;
        }
        impl->LogComputeBind(ComputeBind::Kind::Texture, slot, texture, 0, write, on_buffer,
                             ~u64{0}, ~u64{0}, track_hazard);
        const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = impl->active_compute;
        if (c_enc != nil && impl->NoteObject(impl->active_bound->textures, slot, texture)) {
            if (slot < TextureSlots && impl->range_binds_enabled) {
                impl->active_bound->MarkTextureDirty(slot);
            } else {
                [c_enc setTexture:UnretainedTexture(texture) atIndex:slot];
            }
        }
        return;
    }
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil) {
        return;
    }
    StageBindings* bound = nullptr;
    switch (stage) {
    case ShaderStage::Vertex:
        bound = &impl->vertex_bound;
        break;
    case ShaderStage::Fragment:
        bound = &impl->fragment_bound;
        break;
    case ShaderStage::Object:
        bound = &impl->object_bound;
        break;
    case ShaderStage::Mesh:
        bound = &impl->mesh_bound;
        break;
    case ShaderStage::Compute:
        return;
    }
    if (!write && impl->state_cache_enabled && slot < TextureSlots &&
        bound->textures[slot] == texture &&
        (impl->last_midpass_clear_pass_serial != impl->render_pass_serial ||
         impl->pending_clear_roots_bloom == 0 || !impl->PendingClearOn(texture))) {
        ++impl->state_skips;
        return;
    }
    if (track_hazard) {
        impl->NoteRenderTextureUse(texture, on_buffer, write);
    }
    const id<MTLRenderCommandEncoder> __unsafe_unretained cur_enc = impl->render;
    if (cur_enc == nil || !impl->NoteObject(bound->textures, slot, texture)) {
        return;
    }
    if (slot < TextureSlots && impl->range_binds_enabled) {
        bound->MarkTextureDirty(slot);
        return;
    }
    const id<MTLTexture> __unsafe_unretained mtl_tex = UnretainedTexture(texture);
    switch (stage) {
    case ShaderStage::Vertex:
        [cur_enc setVertexTexture:mtl_tex atIndex:slot];
        break;
    case ShaderStage::Fragment:
        [cur_enc setFragmentTexture:mtl_tex atIndex:slot];
        break;
    case ShaderStage::Object:
        [cur_enc setObjectTexture:mtl_tex atIndex:slot];
        break;
    case ShaderStage::Mesh:
        [cur_enc setMeshTexture:mtl_tex atIndex:slot];
        break;
    case ShaderStage::Compute:
        break;
    }
}

u64 Scheduler::UseTexture(ShaderStage stage, TextureHandle texture, bool write, bool on_buffer,
                          u64 known_gpu_resource_id, bool track_hazard) {
    if (texture == nullptr) {
        return 0;
    }
    const MTLResourceUsage usage =
        write ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
    if (stage == ShaderStage::Compute) {
        if (impl->recording_ahead_wave >= 0) {
            bool conflict = false;
            if (track_hazard) {
                const void* const raw_root = impl->CachedRootTexture(texture);
                const u64 restored_before = impl->samples_restored;
                if (write) {
                    impl->WrittenRoot(texture, raw_root);
                } else {
                    impl->RestoreSamplesIfDropped(texture, raw_root);
                }
                const bool restored = impl->samples_restored != restored_before;
                const ComputeHazardState::BufferSpan bspan =
                    on_buffer ? ComputeHazardState::BackingBufferSpan(raw_root)
                              : ComputeHazardState::BufferSpan{};
                const u64 bsz = bspan.end == ~u64{0} ? ~u64{0} : (bspan.end - bspan.begin);
                if (restored || impl->PendingClearOnRoot(raw_root) ||
                    (impl->ahead_wave_checked &&
                     ((write ? impl->segment_touched_textures : impl->segment_written_textures)
                          .contains(raw_root) ||
                      (bspan.buffer != nullptr &&
                       impl->CheckedConflict(bspan.buffer, bspan.begin, bsz, write))))) {
                    conflict = true;
                    impl->AbortRecordingAheadWave(texture);
                } else {
                    impl->NoteAheadTexture(raw_root, write);
                    if (bspan.buffer != nullptr && bspan.end > bspan.begin) {
                        impl->NoteActiveBuffer(bspan.buffer);
                        impl->NoteAheadReadSpan(bspan.buffer, bspan.begin, bsz);
                        ComputeHazardState::AddSpan(impl->ahead_wave_read_spans, bspan);
                        if (write) {
                            impl->NoteAheadWriteSpan(bspan.buffer, bspan.begin, bsz);
                        }
                    }
                }
            }
            if (!conflict) {
                auto& q = impl->ahead_waves[static_cast<size_t>(impl->recording_ahead_wave)];
                q.binds.push_back({ComputeBind::Kind::UseTexture, on_buffer, write, 0, texture, 0,
                                   0, ~u64{0}, ~u64{0}, track_hazard, false});
                return known_gpu_resource_id != 0
                           ? known_gpu_resource_id
                           : UnretainedTexture(texture).gpuResourceID._impl;
            }
        }
        const void* const raw_root =
            impl->ComputeTexture(std::nullopt, texture, on_buffer, write, track_hazard);
        if (impl->active_compute != nil && impl->active_compute == impl->kernel_compute) {
            impl->kernel_binding = true;
        }
        impl->LogComputeBind(ComputeBind::Kind::UseTexture, 0, texture, 0, write, on_buffer,
                             ~u64{0}, ~u64{0}, track_hazard);
        const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = impl->active_compute;
        const u64 enc_serial = c_enc == nil                 ? 0
                               : c_enc == impl->compute     ? impl->in_order_compute_serial
                                                            : impl->kernel_compute_serial;
        Impl::UseResourceNote& cached =
            impl->compute_use_cache[PtrHashIndex(texture, Impl::ComputeUseCacheBits)];
        const u8 have = on_buffer ? (write ? cached.buffer_write_stages : cached.buffer_read_stages)
                                  : (write ? cached.write_stages : cached.read_stages);
        if (impl->state_cache_enabled && c_enc != nil && cached.resource == texture &&
            cached.encoder_serial == enc_serial && (have & 1U) != 0) {
            return cached.id_or_addr;
        }
        const id<MTLTexture> __unsafe_unretained metal_texture = UnretainedTexture(texture);
        const u64 res_id =
            known_gpu_resource_id != 0 ? known_gpu_resource_id : metal_texture.gpuResourceID._impl;
        if (c_enc != nil) {
            if (!write && impl->range_binds_enabled && impl->recorder == nullptr) {
                AppendTextureReadResources(impl->ActiveComputeReadUse(), metal_texture, on_buffer,
                                           raw_root);
            } else {
                UseTextureResources(c_enc, metal_texture, usage, on_buffer);
            }
            const bool same_enc = cached.resource == texture && cached.encoder_serial == enc_serial;
            cached = {
                texture,
                enc_serial,
                res_id,
                1U,
                static_cast<u8>((same_enc ? cached.write_stages : 0U) | (write ? 1U : 0U)),
                static_cast<u8>((same_enc ? cached.buffer_read_stages : 0U) |
                                (on_buffer ? 1U : 0U)),
                static_cast<u8>((same_enc ? cached.buffer_write_stages : 0U) |
                                (on_buffer && write ? 1U : 0U)),
            };
        }
        return res_id;
    }

    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    const u8 stage_bit = static_cast<u8>(
        stage == ShaderStage::Vertex     ? 1U
        : stage == ShaderStage::Fragment ? 2U
        : stage == ShaderStage::Object   ? 4U
                                         : 8U);
    Impl::UseResourceNote& cached =
        impl->render_use_cache[PtrHashIndex(texture, Impl::RenderUseCacheBits)];
    const u8 have = on_buffer ? (write ? cached.buffer_write_stages : cached.buffer_read_stages)
                              : (write ? cached.write_stages : cached.read_stages);
    if (impl->state_cache_enabled && r_enc != nil &&
        (impl->last_midpass_clear_pass_serial != impl->render_pass_serial ||
         impl->pending_clear_roots_bloom == 0 || !impl->PendingClearOn(texture)) &&
        cached.resource == texture && cached.encoder_serial == impl->render_pass_serial &&
        (have & stage_bit) != 0) {
        return cached.id_or_addr;
    }
    const void* const raw_root =
        track_hazard ? impl->NoteRenderTextureUse(texture, on_buffer, write) : nullptr;
    const id<MTLRenderCommandEncoder> __unsafe_unretained cur_enc = impl->render;
    const id<MTLTexture> __unsafe_unretained metal_texture = UnretainedTexture(texture);
    const u64 res_id =
        known_gpu_resource_id != 0 ? known_gpu_resource_id : metal_texture.gpuResourceID._impl;
    if (cur_enc != nil) {
        const MTLRenderStages stages =
            stage == ShaderStage::Vertex     ? MTLRenderStageVertex
            : stage == ShaderStage::Fragment ? MTLRenderStageFragment
            : stage == ShaderStage::Object   ? MTLRenderStageObject
                                             : MTLRenderStageMesh;
        if (!write && impl->range_binds_enabled && impl->recorder == nullptr) {
            const size_t stage_idx =
                stage == ShaderStage::Vertex     ? 0
                : stage == ShaderStage::Fragment ? 1
                : stage == ShaderStage::Object   ? 2
                                                 : 3;
            AppendRenderTextureReadUse(impl->pass_used_resources,
                                       impl->pending_render_read_use[stage_idx], metal_texture,
                                       stages, on_buffer, raw_root);
        } else {
            AppendTextureUseResources(impl->pass_used_resources, metal_texture, usage, stages,
                                      on_buffer, raw_root);
            UseTextureResources(cur_enc, metal_texture, usage, stages, on_buffer);
        }
        const bool same_pass =
            cached.resource == texture && cached.encoder_serial == impl->render_pass_serial;
        cached = {
            texture,
            impl->render_pass_serial,
            res_id,
            static_cast<u8>((same_pass ? cached.read_stages : 0U) | stage_bit),
            static_cast<u8>((same_pass ? cached.write_stages : 0U) | (write ? stage_bit : 0U)),
            static_cast<u8>((same_pass ? cached.buffer_read_stages : 0U) |
                            (on_buffer ? stage_bit : 0U)),
            static_cast<u8>((same_pass ? cached.buffer_write_stages : 0U) |
                            (on_buffer && write ? stage_bit : 0U)),
        };
    }
    return res_id;
}

u64 Scheduler::BufferGpuAddress(BufferHandle buffer) noexcept {
    if (buffer == nullptr) {
        return 0;
    }
    return static_cast<u64>(UnretainedBuffer(buffer).gpuAddress);
}

u64 Scheduler::TextureResourceId(TextureHandle texture) noexcept {
    if (texture == nullptr) {
        return 0;
    }
    return UnretainedTexture(texture).gpuResourceID._impl;
}

u64 Scheduler::SamplerResourceId(void* sampler) noexcept {
    if (sampler == nullptr) {
        return 0;
    }
    // Guest samplers are made with supportArgumentBuffers (CreateSampler).
    return UnretainedSampler(sampler).gpuResourceID._impl;
}

void Scheduler::FlushStageBindings(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        if (!impl->vertex_bound.HasDirty() && impl->pending_render_read_use[0].empty()) {
            return;
        }
        break;
    case ShaderStage::Fragment:
        if (!impl->fragment_bound.HasDirty() && impl->pending_render_read_use[1].empty()) {
            return;
        }
        break;
    case ShaderStage::Object:
        if (!impl->object_bound.HasDirty() && impl->pending_render_read_use[2].empty()) {
            return;
        }
        break;
    case ShaderStage::Mesh:
        if (!impl->mesh_bound.HasDirty() && impl->pending_render_read_use[3].empty()) {
            return;
        }
        break;
    case ShaderStage::Compute:
        if ((impl->active_bound == nullptr || !impl->active_bound->HasDirty()) &&
            impl->ActiveComputeReadUse().empty()) {
            return;
        }
        break;
    }
    impl->FlushStageBindingsImpl(stage);
}

void Scheduler::BindSampler(ShaderStage stage, u32 slot, void* sampler) {
    if (!SlotFits(slot, SamplerSlots, "sampler")) {
        return;
    }
    if (stage == ShaderStage::Compute) {
        if (impl->recording_ahead_wave >= 0) {
            auto& q = impl->ahead_waves[static_cast<size_t>(impl->recording_ahead_wave)];
            q.binds.push_back({ComputeBind::Kind::Sampler, false, false, slot, sampler, 0, 0});
            return;
        }
        impl->ResumeKernel();
        if (impl->active_compute != nil && impl->active_compute == impl->kernel_compute) {
            impl->kernel_binding = true;
        }
        impl->LogComputeBind(ComputeBind::Kind::Sampler, slot, sampler, 0, false);
        const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = impl->active_compute;
        if (c_enc != nil && impl->NoteObject(impl->active_bound->samplers, slot, sampler)) {
            if (slot < SamplerSlots && impl->range_binds_enabled) {
                impl->active_bound->MarkSamplerDirty(slot);
            } else {
                [c_enc setSamplerState:UnretainedSampler(sampler) atIndex:slot];
            }
        }
        return;
    }
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil) {
        return;
    }
    StageBindings* bound = nullptr;
    switch (stage) {
    case ShaderStage::Vertex:
        bound = &impl->vertex_bound;
        break;
    case ShaderStage::Fragment:
        bound = &impl->fragment_bound;
        break;
    case ShaderStage::Object:
        bound = &impl->object_bound;
        break;
    case ShaderStage::Mesh:
        bound = &impl->mesh_bound;
        break;
    case ShaderStage::Compute:
        return;
    }
    if (!impl->NoteObject(bound->samplers, slot, sampler)) {
        return;
    }
    if (slot < SamplerSlots && impl->range_binds_enabled) {
        bound->MarkSamplerDirty(slot);
        return;
    }
    const id<MTLSamplerState> __unsafe_unretained mtl_smp = UnretainedSampler(sampler);
    switch (stage) {
    case ShaderStage::Vertex:
        [r_enc setVertexSamplerState:mtl_smp atIndex:slot];
        break;
    case ShaderStage::Fragment:
        [r_enc setFragmentSamplerState:mtl_smp atIndex:slot];
        break;
    case ShaderStage::Object:
        [r_enc setObjectSamplerState:mtl_smp atIndex:slot];
        break;
    case ShaderStage::Mesh:
        [r_enc setMeshSamplerState:mtl_smp atIndex:slot];
        break;
    case ShaderStage::Compute:
        break;
    }
}

void Scheduler::DrawIndirect(u32 primitive, BufferHandle buffer, u64 offset) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || buffer == nullptr) {
        return;
    }
    impl->FlushRenderBindings();
    impl->NoteActiveBuffer(buffer);
    [r_enc drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
           indirectBuffer:UnretainedBuffer(buffer)
     indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::DrawIndexedIndirect(u32 primitive, u32 index_size, BufferHandle index_buffer,
                                    u64 index_offset, BufferHandle buffer, u64 offset) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || buffer == nullptr || index_buffer == nullptr) {
        return;
    }
    impl->FlushRenderBindings();
    impl->NoteActiveBuffer(index_buffer);
    impl->NoteActiveBuffer(buffer);
    [r_enc drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                       indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                     indexBuffer:UnretainedBuffer(index_buffer)
               indexBufferOffset:index_offset
                  indirectBuffer:UnretainedBuffer(buffer)
            indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::PrepareCompute() {
    if (!impl->compute_copy_enabled || impl->compute != nil) {
        return;
    }
    if (!impl->clear_narrow_enabled) {
        impl->FlushClear();
    }
    if (impl->render != nil || impl->blit_render != nil) {
        impl->EndEncoder(PassEnd::Compute);
    }
    impl->want_in_order_compute = true;
}

void Scheduler::BeginCompute(void* pipeline, bool ahead, bool keep_ahead_waves) {
    Impl& state = *impl;
    state.recording_ahead_wave = -1;
    state.ahead_wave_checked = false;
    state.kernel_resume = false;
    state.kernel_checked = false;
    state.kernel_binding = false;
    state.compute_clobbered_by_copy = false;
    state.compute_binds.clear();
    state.compute_bind_bytes.clear();
    state.compute_bind_pipeline = pipeline;
    bool goes_ahead =
        ahead && state.ahead_enabled &&
        (keep_ahead_waves || state.render != nil || state.ahead_buffer != nil ||
         state.upload_buffer != nil || state.kernel_buffer != nil);
    bool checked = false;
    if (goes_ahead && state.ahead_any_enabled) {
        // Past what CanRecordAhead allowed before EDEN_NX2_METAL_AHEAD_ANY: checked by its binds.
        const bool plain =
            !state.ahead_query_relaxed && state.deferred_downloads.empty() &&
            (state.segment_encoders == 0 || (state.segment_encoders == 1 && state.render != nil));
        if (!plain) {
            checked = true;
            goes_ahead = !state.segment_opaque && !state.checked_failed && !state.ahead_blocked;
        }
    }
    const void** bound_pipeline = &state.compute_pipeline;
    if (goes_ahead) {
        if (!keep_ahead_waves && (!state.ahead_waves[0].Empty() || !state.ahead_waves[1].Empty())) {
            state.FlushAheadComputeWaves();
        }
        state.active_compute = state.KernelEncoder();
        state.active_bound = &state.kernel_bound;
        state.active_hazards = &state.kernel_hazards;
        bound_pipeline = &state.kernel_pipeline;
        state.kernel_checked = checked;
        state.kernel_binding = true;
        ++state.ahead_kernels;
        if (checked) {
            ++state.ahead_checked;
        }
    } else {
        // EDEN_NX2_METAL_CLEAR_NARROW: a waiting clear goes first when a texture bind is on it
        // (ComputeTexture).
        if (!state.clear_narrow_enabled) {
            state.FlushClear();
        }
        state.OpenInOrderCompute();
    }
    if (state.active_hazards != nullptr) {
        state.active_hazards->ResetPending();
    }
    if (state.state_cache_enabled && *bound_pipeline == pipeline) {
        ++state.state_skips;
        return;
    }
    *bound_pipeline = pipeline;
    ++state.state_sets;
    const id<MTLComputeCommandEncoder> __unsafe_unretained c_enc = state.active_compute;
    [c_enc
        setComputePipelineState:(__bridge __unsafe_unretained id<MTLComputePipelineState>)pipeline];
}

bool Scheduler::ActiveComputeIsAhead() const noexcept {
    return impl->active_compute != nil && impl->active_compute == impl->kernel_compute &&
           impl->recording_ahead_wave < 0;
}

bool Scheduler::BeginAheadWave(u32 wave_index, void* pipeline) {
    Impl& state = *impl;
    if (!state.mesh_wave_batch_enabled || !state.concurrent_compute_enabled ||
        wave_index >= state.ahead_waves.size() || state.kernel_compute == nil ||
        pipeline == nullptr) {
        return false;
    }
    state.recording_ahead_wave = static_cast<int>(wave_index);
    const bool plain =
        !state.ahead_query_relaxed && state.deferred_downloads.empty() &&
        (state.segment_encoders == 0 || (state.segment_encoders == 1 && state.render != nil));
    state.ahead_wave_checked = state.ahead_any_enabled && !plain;
    state.recording_wave_pipeline = pipeline;
    const auto& q = state.ahead_waves[wave_index];
    state.recording_wave_bind_start = static_cast<u32>(q.binds.size());
    state.recording_wave_bytes_start = static_cast<u32>(q.bytes.size());
    return true;
}

void Scheduler::Dispatch(std::array<u32, 3> groups, std::array<u32, 3> group_size) {
    if (impl->recording_ahead_wave >= 0) {
        const size_t w = static_cast<size_t>(impl->recording_ahead_wave);
        impl->recording_ahead_wave = -1;
        impl->ahead_wave_checked = false;
        if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
            impl->ahead_waves[w].binds.resize(impl->recording_wave_bind_start);
            impl->ahead_waves[w].bytes.resize(impl->recording_wave_bytes_start);
            return;
        }
        auto& q = impl->ahead_waves[w];
        q.dispatches.push_back({
            .pipeline = impl->recording_wave_pipeline,
            .bind_begin = impl->recording_wave_bind_start,
            .bind_count = static_cast<u32>(q.binds.size()) - impl->recording_wave_bind_start,
            .indirect = false,
            .groups = groups,
            .group_size = group_size,
        });
        ++impl->ahead_kernels;
        return;
    }
    impl->ResumeKernel();
    impl->kernel_binding = false;
    const id<MTLComputeCommandEncoder> __unsafe_unretained encoder = impl->active_compute;
    if (encoder == nil || groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return;
    }
    impl->FlushComputeBindings();
    if (impl->active_hazards != nullptr) {
        impl->active_hazards->BeforeDispatch(encoder, impl->concurrent_compute_enabled);
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
            threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1], group_size[2])];
    if (impl->active_hazards != nullptr) {
        impl->active_hazards->AfterDispatch(impl->concurrent_compute_enabled);
    }
    impl->CountDispatch();
}

void Scheduler::DispatchIndirect(BufferHandle buffer, u64 offset, std::array<u32, 3> group_size) {
    if (impl->recording_ahead_wave >= 0) {
        if (impl->ahead_wave_checked && impl->CheckedConflict(buffer, offset, 12, false)) {
            impl->AbortRecordingAheadWave(nullptr);
        } else {
            const size_t w = static_cast<size_t>(impl->recording_ahead_wave);
            impl->recording_ahead_wave = -1;
            impl->ahead_wave_checked = false;
            if (buffer == nullptr) {
                impl->ahead_waves[w].binds.resize(impl->recording_wave_bind_start);
                impl->ahead_waves[w].bytes.resize(impl->recording_wave_bytes_start);
                return;
            }
            impl->NoteActiveBuffer(buffer);
            impl->NoteAheadReadSpan(buffer, offset, 12);
            ComputeHazardState::AddSpan(
                impl->ahead_wave_read_spans,
                {buffer, offset, ComputeHazardState::SaturatingEnd(offset, 12)});
            auto& q = impl->ahead_waves[w];
            q.dispatches.push_back({
                .pipeline = impl->recording_wave_pipeline,
                .bind_begin = impl->recording_wave_bind_start,
                .bind_count = static_cast<u32>(q.binds.size()) - impl->recording_wave_bind_start,
                .indirect = true,
                .group_size = group_size,
                .indirect_buffer = buffer,
                .indirect_offset = offset,
            });
            ++impl->ahead_kernels;
            return;
        }
    }
    impl->ComputeBuffer(std::nullopt, buffer, false, offset, 12, 0, false);
    impl->kernel_binding = false;
    const id<MTLComputeCommandEncoder> __unsafe_unretained encoder = impl->active_compute;
    if (encoder == nil || buffer == nullptr) {
        return;
    }
    impl->FlushComputeBindings();
    const ComputeHazardState::BufferSpan ind_span{
        buffer, offset, ComputeHazardState::SaturatingEnd(offset, 12)};
    if (impl->active_hazards != nullptr) {
        impl->active_hazards->BeforeDispatch(encoder, impl->concurrent_compute_enabled, ind_span);
    }
    [encoder dispatchThreadgroupsWithIndirectBuffer:UnretainedBuffer(buffer)
                               indirectBufferOffset:offset
                              threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1],
                                                                group_size[2])];
    if (impl->active_hazards != nullptr) {
        impl->active_hazards->AfterDispatch(impl->concurrent_compute_enabled, ind_span);
    }
    impl->CountDispatch();
}

void Scheduler::DispatchStageIn(u32 first, u32 count, u32 first_instance, u32 instances) {
    impl->ResumeKernel();
    impl->kernel_binding = false;
    const id<MTLComputeCommandEncoder> __unsafe_unretained encoder = impl->active_compute;
    if (encoder == nil || count == 0 || instances == 0) {
        return;
    }
    impl->FlushComputeBindings();
    if (impl->active_hazards != nullptr) {
        impl->active_hazards->BeforeDispatch(encoder, impl->concurrent_compute_enabled);
    }
    [encoder setStageInRegion:MTLRegionMake2D(first, first_instance, count, instances)];
    [encoder dispatchThreads:MTLSizeMake(count, instances, 1)
       threadsPerThreadgroup:MTLSizeMake((std::min)(count, 64U), 1, 1)];
    if (impl->active_hazards != nullptr) {
        impl->active_hazards->AfterDispatch(impl->concurrent_compute_enabled);
    }
    impl->CountDispatch();
}

void Scheduler::SetTessellationFactorBuffer(BufferHandle buffer, u64 offset) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || buffer == nullptr) {
        return;
    }
    impl->NoteActiveBuffer(buffer);
    [r_enc setTessellationFactorBuffer:UnretainedBuffer(buffer) offset:offset instanceStride:0];
}

void Scheduler::DrawPatches(u32 control_points, u32 patch_count) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || patch_count == 0) {
        return;
    }
    impl->FlushRenderBindings();
    [r_enc drawPatches:control_points
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
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || vertex_count == 0 || instance_count == 0) {
        return;
    }
    impl->FlushRenderBindings();
    [r_enc drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
              vertexStart:first_vertex
              vertexCount:vertex_count
            instanceCount:instance_count
             baseInstance:base_instance];
    impl->CountDraw();
}

void Scheduler::DrawMeshThreadgroups(u32 groups, u32 object_threads, u32 mesh_threads) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || groups == 0) {
        return;
    }
    impl->FlushRenderBindings();
    [r_enc drawMeshThreadgroups:MTLSizeMake(groups, 1, 1)
   threadsPerObjectThreadgroup:MTLSizeMake((std::max)(object_threads, 1U), 1, 1)
     threadsPerMeshThreadgroup:MTLSizeMake((std::max)(mesh_threads, 1U), 1, 1)];
    impl->CountDraw();
}

void Scheduler::DrawMeshThreadgroupsIndirect(BufferHandle buffer, u64 offset, u32 object_threads,
                                             u32 mesh_threads) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || buffer == nullptr) {
        return;
    }
    impl->FlushRenderBindings();
    impl->NoteActiveBuffer(buffer);
    [r_enc drawMeshThreadgroupsWithIndirectBuffer:UnretainedBuffer(buffer)
                             indirectBufferOffset:offset
                      threadsPerObjectThreadgroup:MTLSizeMake((std::max)(object_threads, 1U), 1, 1)
                        threadsPerMeshThreadgroup:MTLSizeMake((std::max)(mesh_threads, 1U), 1, 1)];
    impl->CountDraw();
}

void Scheduler::DrawIndexed(u32 primitive, u32 index_count, u32 index_size,
                            BufferHandle index_buffer, u64 index_offset, u32 instance_count,
                            s32 base_vertex, u32 base_instance) {
    const id<MTLRenderCommandEncoder> __unsafe_unretained r_enc = impl->render;
    if (r_enc == nil || index_count == 0 || instance_count == 0 || index_buffer == nullptr) {
        return;
    }
    impl->FlushRenderBindings();
    impl->NoteActiveBuffer(index_buffer);
    [r_enc drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                      indexCount:index_count
                       indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                     indexBuffer:UnretainedBuffer(index_buffer)
               indexBufferOffset:index_offset
                   instanceCount:instance_count
                      baseVertex:base_vertex
                    baseInstance:base_instance];
    impl->CountDraw();
}

void* Scheduler::CommandBuffer(bool allow_proxy) {
    impl->FlushClear();
    impl->EndEncoder(PassEnd::CommandBuffer);
    impl->FlushDeferredDownloads();
    if (impl->blit != nil) {
        [impl->blit endEncoding];
        impl->blit = nil;
    }
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

void* Scheduler::CommandBufferForTextures(std::span<const TextureHandle> read_textures,
                                          TextureHandle write_texture) {
    if (!impl->pending_clears.empty()) {
        if (!impl->clear_narrow_enabled) {
            impl->FlushClear();
        } else {
            for (const TextureHandle tex : read_textures) {
                impl->Touch(tex);
            }
            impl->Touch(write_texture);
        }
    }
    for (const TextureHandle tex : read_textures) {
        if (tex != nullptr) {
            impl->SamplesNeeded(tex);
        }
    }
    if (write_texture != nullptr) {
        impl->Written(write_texture);
        impl->NoteTextureWrite(write_texture);
        impl->NoteSegmentTexture(write_texture, true);
    }
    impl->EndEncoder(PassEnd::CommandBuffer);
    ++impl->segment_encoders;
    if (impl->recorder != nullptr) {
        const id<MTLCommandBuffer> proxy = impl->Commands();
        impl->recorder->Drain("CommandBufferForTextures()");
        return (__bridge void*)Rec::Recorder::RealOf(proxy);
    }
    return (__bridge void*)impl->Commands();
}

void* Scheduler::SharedEvent() const noexcept {
    return (__bridge void*)impl->event;
}

u64 Scheduler::TakeGpuNanoseconds() noexcept {
    return impl->completion->gpu_ns.exchange(0, std::memory_order_relaxed);
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
    state.pass_gpu_ns_before = state.completion->gpu_ns.load(std::memory_order_relaxed);
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
    const u64 gpu_now = state.completion->gpu_ns.load(std::memory_order_relaxed);
    const f64 frame_ms =
        gpu_now >= state.pass_gpu_ns_before
            ? static_cast<f64>(gpu_now - state.pass_gpu_ns_before) / 1e6
            : 0.0;
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
    const u64 barriers = std::exchange(impl->compute_hazards.barriers_emitted, 0) +
                         std::exchange(impl->kernel_hazards.barriers_emitted, 0);
    line += fmt::format(
        " (copies as dispatches per frame: ahead={:.1f} in_order={:.1f}; compute barriers={:.1f}/"
        "frame; texture notes skipped per frame {:.1f}; in-pass clears {:.1f}/frame)",
        static_cast<double>(std::exchange(impl->compute_copies_ahead, 0)) / per,
        static_cast<double>(std::exchange(impl->compute_copies_in_order, 0)) / per,
        static_cast<double>(barriers) / per,
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

std::array<u64, 2> Scheduler::TakeBatchCounts() noexcept {
    return {std::exchange(impl->range_bind_calls, 0),
            std::exchange(impl->residency_batch_calls, 0)};
}

#undef UnretainedSampler
#undef UnretainedTexture
#undef UnretainedBuffer

} // namespace Metal
