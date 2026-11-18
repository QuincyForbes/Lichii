// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke's texture cases (metal-coverage.md sections E, F, G and R): multisampled
// images through their sample grid and resolves, depth and stencil bytes to and from buffers,
// depth <-> colour conversions, samplers, and the guards that keep Metal from undefined input.
// Each is drawn on the runner's device, read back and compared.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_smoke_textures.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/texture_cache/decode_bc.h"
#include "video_core/texture_cache/decode_etc.h"
#include "video_core/textures/texture.h"

namespace SmokeTextures {

namespace {

using Metal::Object;
using Metal::TextureKind;

constexpr uint32_t Rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);

bool Close(int a, int b, int tolerance) {
    return std::abs(a - b) <= tolerance;
}

/// A shared buffer of `size` bytes and its contents.
std::pair<Object, uint8_t*> SharedBuffer(const Metal::Device& device, size_t size) {
    uint8_t* contents = nullptr;
    Object buffer = Metal::CreateBuffer(device, size, true, &contents);
    if (contents != nullptr) {
        std::memset(contents, 0, size);
    }
    return {std::move(buffer), contents};
}

/// Samples are copied into a 4-sample texture from their texel grid (TextureCacheRuntime's
/// uploads and CopyImageMSAA), back into another grid (downloads, plain reads of the image) and
/// resolved (BlitImage of the whole image).
int RunSampleCopies() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Pixels = 4;
    constexpr uint32_t Grid = Pixels * 2;
    const Object samples = Metal::CreateTexture(device, {.kind = TextureKind::e2DMultisampleArray,
                                                         .format = Rgba8,
                                                         .width = Pixels,
                                                         .height = Pixels,
                                                         .samples = 4,
                                                         .render_target = true});
    const Metal::TextureDesc grid{.kind = TextureKind::e2DArray,
                                  .format = Rgba8,
                                  .width = Grid,
                                  .height = Grid,
                                  .render_target = true};
    const Object grid_in = Metal::CreateTexture(device, grid);
    const Object grid_out = Metal::CreateTexture(device, grid);
    const Object resolved = Metal::CreateTexture(device, {.kind = TextureKind::e2DArray,
                                                          .format = Rgba8,
                                                          .width = Pixels,
                                                          .height = Pixels,
                                                          .render_target = true});
    auto [upload, upload_bytes] = SharedBuffer(device, Grid * Grid * 4);
    auto [grid_back, grid_bytes] = SharedBuffer(device, Grid * Grid * 4);
    auto [resolve_back, resolve_bytes] = SharedBuffer(device, Pixels * Pixels * 4);
    if (!samples || !grid_in || !grid_out || !resolved || upload_bytes == nullptr ||
        grid_bytes == nullptr || resolve_bytes == nullptr) {
        std::printf("metal-smoke: FAIL sample copies: textures\n");
        return 1;
    }
    for (uint32_t y = 0; y < Grid; ++y) {
        for (uint32_t x = 0; x < Grid; ++x) {
            uint8_t* const texel = upload_bytes + (y * Grid + x) * 4;
            texel[0] = static_cast<uint8_t>(x * 30);
            texel[1] = static_cast<uint8_t>(y * 30);
            texel[2] = static_cast<uint8_t>((x ^ y) * 20);
            texel[3] = 255;
        }
    }
    scheduler.CopyBufferToTexture(upload.Get(), 0, Grid * 4, 0, grid_in.Get(), 0, 0, {},
                                  {Grid, Grid, 1});
    scheduler.BreakBlits();
    const Object view = Metal::CreateTextureView(grid_in.Get(), Rgba8, TextureKind::e2D, 0, 1, 0,
                                                 1, Metal::IdentitySwizzle);
    const bool compressed = scheduler.CopySamples(
        Metal::SampleCopy{.src = view.Get(),
                          .dst = {.texture = samples.Get()},
                          .dst_format = Rgba8,
                          .dst_width = Pixels,
                          .dst_height = Pixels,
                          .samples = 4,
                          .shift = {1, 1}},
        false);
    const bool expanded = scheduler.CopySamples(
        Metal::SampleCopy{.src = samples.Get(),
                          .dst = {.texture = grid_out.Get()},
                          .dst_format = Rgba8,
                          .dst_width = Grid,
                          .dst_height = Grid,
                          .samples = 4,
                          .shift = {1, 1}},
        true);
    const bool resolve =
        scheduler.Resolve(samples.Get(), 0, {.texture = resolved.Get()}, Pixels, Pixels);
    scheduler.CopyTextureToBuffer(grid_out.Get(), 0, 0, {}, {Grid, Grid, 1}, grid_back.Get(), 0,
                                  Grid * 4, 0);
    scheduler.CopyTextureToBuffer(resolved.Get(), 0, 0, {}, {Pixels, Pixels, 1},
                                  resolve_back.Get(), 0, Pixels * 4, 0);
    scheduler.Finish();
    uint32_t grid_wrong = 0;
    for (uint32_t index = 0; index < Grid * Grid * 4; ++index) {
        grid_wrong += upload_bytes[index] == grid_bytes[index] ? 0 : 1;
    }
    uint32_t resolve_wrong = 0;
    for (uint32_t y = 0; y < Pixels; ++y) {
        for (uint32_t x = 0; x < Pixels; ++x) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                int sum = 0;
                for (uint32_t sample = 0; sample < 4; ++sample) {
                    const uint32_t gx = x * 2 + (sample & 1);
                    const uint32_t gy = y * 2 + (sample >> 1);
                    sum += upload_bytes[(gy * Grid + gx) * 4 + channel];
                }
                const int got = resolve_bytes[(y * Pixels + x) * 4 + channel];
                resolve_wrong += Close(got, sum / 4, 2) ? 0 : 1;
            }
        }
    }
    const bool passed = compressed && expanded && resolve && grid_wrong == 0 && resolve_wrong == 0;
    std::printf("metal-smoke: sample copies 8x8 grid -> 4x4 x4 -> 8x8 grid: %u of %u bytes "
                "wrong; resolve %s, %u of %u channels off the average: %s\n",
                grid_wrong, Grid * Grid * 4, resolve ? "done" : "REFUSED", resolve_wrong,
                Pixels * Pixels * 4, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// G16: resolution scaling of what Blit can't draw (Scheduler::ScaleNearest): a 4-sample 4x4
/// image to 8x8 sample for sample, and a stencil-only 4x4 one to 8x8.
int RunScaleNearest() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Small = 4;
    constexpr uint32_t Large = 8;
    const auto multisampled = [&](uint32_t side) {
        return Metal::CreateTexture(device, {.kind = TextureKind::e2DMultisampleArray,
                                             .format = Rgba8,
                                             .width = side,
                                             .height = side,
                                             .samples = 4,
                                             .render_target = true});
    };
    const auto grid = [&](uint32_t side) {
        return Metal::CreateTexture(device, {.kind = TextureKind::e2DArray,
                                             .format = Rgba8,
                                             .width = side * 2,
                                             .height = side * 2,
                                             .render_target = true});
    };
    const Object small = multisampled(Small);
    const Object large = multisampled(Large);
    const Object small_grid = grid(Small);
    const Object large_grid = grid(Large);
    const uint32_t stencil8 = static_cast<uint32_t>(MTLPixelFormatStencil8);
    const auto stencil_texture = [&](uint32_t side) {
        return Metal::CreateTexture(device, {.kind = TextureKind::e2DArray,
                                             .format = stencil8,
                                             .width = side,
                                             .height = side,
                                             .render_target = true});
    };
    const Object stencil_small = stencil_texture(Small);
    const Object stencil_large = stencil_texture(Large);
    auto [upload, upload_bytes] = SharedBuffer(device, Small * 2 * Small * 2 * 4);
    auto [back, back_bytes] = SharedBuffer(device, Large * 2 * Large * 2 * 4);
    auto [stencil_up, stencil_up_bytes] = SharedBuffer(device, Small * Small);
    auto [stencil_back, stencil_back_bytes] = SharedBuffer(device, Large * Large);
    if (!small || !large || !small_grid || !large_grid || !stencil_small || !stencil_large ||
        upload_bytes == nullptr || back_bytes == nullptr || stencil_up_bytes == nullptr ||
        stencil_back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL scale nearest: textures\n");
        return 1;
    }
    constexpr uint32_t SmallGrid = Small * 2;
    constexpr uint32_t LargeGrid = Large * 2;
    for (uint32_t index = 0; index < SmallGrid * SmallGrid; ++index) {
        upload_bytes[index * 4] = static_cast<uint8_t>(index * 3);
        upload_bytes[index * 4 + 1] = static_cast<uint8_t>(255 - index);
        upload_bytes[index * 4 + 2] = 0;
        upload_bytes[index * 4 + 3] = 255;
    }
    for (uint32_t index = 0; index < Small * Small; ++index) {
        stencil_up_bytes[index] = static_cast<uint8_t>(index * 11 + 1);
    }
    scheduler.CopyBufferToTexture(upload.Get(), 0, SmallGrid * 4, 0, small_grid.Get(), 0, 0, {},
                                  {SmallGrid, SmallGrid, 1});
    scheduler.CopyBufferToTexture(stencil_up.Get(), 0, Small, 0, stencil_small.Get(), 0, 0, {},
                                  {Small, Small, 1});
    scheduler.BreakBlits();
    const Object small_view = Metal::CreateTextureView(small_grid.Get(), Rgba8, TextureKind::e2D,
                                                       0, 1, 0, 1, Metal::IdentitySwizzle);
    const bool compressed = scheduler.CopySamples(Metal::SampleCopy{.src = small_view.Get(),
                                                                    .dst = {.texture = small.Get()},
                                                                    .dst_format = Rgba8,
                                                                    .dst_width = Small,
                                                                    .dst_height = Small,
                                                                    .samples = 4,
                                                                    .shift = {1, 1}},
                                                  false);
    const bool scaled = scheduler.ScaleNearest(Metal::ScalePass{.src = small.Get(),
                                                                .src_width = Small,
                                                                .src_height = Small,
                                                                .dst = {.texture = large.Get()},
                                                                .dst_format = Rgba8,
                                                                .dst_width = Large,
                                                                .dst_height = Large});
    const bool stencil_scaled =
        scheduler.ScaleNearest(Metal::ScalePass{.src = stencil_small.Get(),
                                                .src_width = Small,
                                                .src_height = Small,
                                                .dst = {.texture = stencil_large.Get()},
                                                .dst_format = stencil8,
                                                .dst_width = Large,
                                                .dst_height = Large});
    const bool expanded = scheduler.CopySamples(Metal::SampleCopy{.src = large.Get(),
                                                                  .dst = {.texture = large_grid.Get()},
                                                                  .dst_format = Rgba8,
                                                                  .dst_width = LargeGrid,
                                                                  .dst_height = LargeGrid,
                                                                  .samples = 4,
                                                                  .shift = {1, 1}},
                                                true);
    scheduler.CopyTextureToBuffer(large_grid.Get(), 0, 0, {}, {LargeGrid, LargeGrid, 1},
                                  back.Get(), 0, LargeGrid * 4, 0);
    scheduler.CopyTextureToBuffer(stencil_large.Get(), 0, 0, {}, {Large, Large, 1},
                                  stencil_back.Get(), 0, Large, 0);
    scheduler.Finish();
    // Pixel p of the large image holds pixel p / 2 of the small one, every sample its own.
    uint32_t wrong = 0;
    for (uint32_t y = 0; y < LargeGrid; ++y) {
        for (uint32_t x = 0; x < LargeGrid; ++x) {
            const uint32_t pixel_x = x >> 1;
            const uint32_t pixel_y = y >> 1;
            const uint32_t sx = ((pixel_x / 2) << 1) | (x & 1);
            const uint32_t sy = ((pixel_y / 2) << 1) | (y & 1);
            const uint8_t* const got = back_bytes + (y * LargeGrid + x) * 4;
            const uint8_t* const want = upload_bytes + (sy * SmallGrid + sx) * 4;
            wrong += std::memcmp(got, want, 4) == 0 ? 0 : 1;
        }
    }
    uint32_t stencil_wrong = 0;
    for (uint32_t y = 0; y < Large; ++y) {
        for (uint32_t x = 0; x < Large; ++x) {
            stencil_wrong +=
                stencil_back_bytes[y * Large + x] == stencil_up_bytes[(y / 2) * Small + x / 2] ? 0
                                                                                              : 1;
        }
    }
    const bool passed = compressed && scaled && stencil_scaled && expanded && wrong == 0 &&
                        stencil_wrong == 0;
    std::printf("metal-smoke: scale nearest 4x4 -> 8x8: 4 samples %s, %u of %u samples wrong; "
                "stencil %s, %u of %u wrong: %s\n",
                scaled ? "scaled" : "REFUSED", wrong, LargeGrid * LargeGrid,
                stencil_scaled ? "scaled" : "REFUSED", stencil_wrong, Large * Large,
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// The guest's bytes of 24-bit depth with stencil (and of 32-bit float depth with stencil, in a
/// region) into Depth32Float_Stencil8 and back (Scheduler::DepthFromBuffer / DepthToBuffer).
int RunDepthBytes() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Side = 8;
    const uint32_t ds_format = static_cast<uint32_t>(MTLPixelFormatDepth32Float_Stencil8);
    const Metal::TextureDesc desc{.kind = TextureKind::e2DArray,
                                  .format = ds_format,
                                  .width = Side,
                                  .height = Side,
                                  .render_target = true};
    const Object packed = Metal::CreateTexture(device, desc);
    const Object floats = Metal::CreateTexture(device, desc);
    auto [packed_in, packed_in_bytes] = SharedBuffer(device, Side * Side * 4);
    auto [packed_out, packed_out_bytes] = SharedBuffer(device, Side * Side * 4);
    constexpr uint32_t Region = 4;
    constexpr Metal::Offset3D Origin{2, 3, 0};
    auto [float_in, float_in_bytes] = SharedBuffer(device, Region * Region * 8);
    auto [float_out, float_out_bytes] = SharedBuffer(device, Region * Region * 8);
    if (!packed || !floats || packed_in_bytes == nullptr || packed_out_bytes == nullptr ||
        float_in_bytes == nullptr || float_out_bytes == nullptr) {
        std::printf("metal-smoke: FAIL depth bytes: textures\n");
        return 1;
    }
    auto* const packed_words = reinterpret_cast<uint32_t*>(packed_in_bytes);
    for (uint32_t index = 0; index < Side * Side; ++index) {
        const uint32_t depth = (index * 262147U) & 0xFFFFFFU;
        packed_words[index] = (depth << 8) | ((index * 37U) & 0xFFU);
    }
    auto* const float_words = reinterpret_cast<uint32_t*>(float_in_bytes);
    for (uint32_t index = 0; index < Region * Region; ++index) {
        const float depth = static_cast<float>(index) / 64.0f;
        std::memcpy(&float_words[index * 2], &depth, sizeof(depth));
        float_words[index * 2 + 1] = 0xABCD0000U | (index * 13U & 0xFFU);
    }
    const bool packed_up = scheduler.DepthFromBuffer(packed_in.Get(), 0, Side * 4, packed.Get(),
                                                     0, 0, {}, {Side, Side, 1},
                                                     Metal::DepthLayout::Packed24_8);
    const bool float_up = scheduler.DepthFromBuffer(float_in.Get(), 0, Region * 8, floats.Get(), 0,
                                                    0, Origin, {Region, Region, 1},
                                                    Metal::DepthLayout::Float32_X24S8);
    const bool packed_down = scheduler.DepthToBuffer(packed.Get(), 0, 0, {}, {Side, Side, 1},
                                                     packed_out.Get(), 0, Side * 4,
                                                     Metal::DepthLayout::Packed24_8);
    const bool float_down = scheduler.DepthToBuffer(floats.Get(), 0, 0, Origin, {Region, Region, 1},
                                                    float_out.Get(), 0, Region * 8,
                                                    Metal::DepthLayout::Float32_X24S8);
    scheduler.Finish();
    const auto* const packed_back = reinterpret_cast<const uint32_t*>(packed_out_bytes);
    uint32_t packed_wrong = 0;
    for (uint32_t index = 0; index < Side * Side; ++index) {
        const bool stencil = (packed_back[index] & 0xFF) == (packed_words[index] & 0xFF);
        const bool depth = Close(static_cast<int>(packed_back[index] >> 8),
                                 static_cast<int>(packed_words[index] >> 8), 1);
        packed_wrong += stencil && depth ? 0 : 1;
    }
    const auto* const float_back = reinterpret_cast<const uint32_t*>(float_out_bytes);
    uint32_t float_wrong = 0;
    for (uint32_t index = 0; index < Region * Region; ++index) {
        const bool depth = float_back[index * 2] == float_words[index * 2];
        const bool stencil = float_back[index * 2 + 1] == (float_words[index * 2 + 1] & 0xFF);
        float_wrong += depth && stencil ? 0 : 1;
    }
    const bool passed = packed_up && float_up && packed_down && float_down && packed_wrong == 0 &&
                        float_wrong == 0;
    std::printf("metal-smoke: depth bytes into Depth32Float_Stencil8 and back: D24S8 8x8 %u of "
                "%u wrong, D32F_S8 4x4 at 2,3 %u of %u wrong: %s\n",
                packed_wrong, Side * Side, float_wrong, Region * Region,
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Vulkan's convert_*.frag on Metal (Scheduler::Convert): ABGR8 -> D24S8 -> ABGR8, ABGR8 ->
/// X8D24 -> ABGR8 and D32F -> R32F, each read back.
int RunConversions() {
    using Metal::ConvertKind;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Side = 8;
    const uint32_t ds_format = static_cast<uint32_t>(MTLPixelFormatDepth32Float_Stencil8);
    const uint32_t d_format = static_cast<uint32_t>(MTLPixelFormatDepth32Float);
    const uint32_t r32f = static_cast<uint32_t>(MTLPixelFormatR32Float);
    const auto texture = [&](uint32_t format) {
        return Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                             .format = format,
                                             .width = Side,
                                             .height = Side,
                                             .render_target = true});
    };
    const Object colour = texture(Rgba8);
    const Object depth_stencil = texture(ds_format);
    const Object colour_back = texture(Rgba8);
    const Object depth = texture(d_format);
    const Object colour_x8 = texture(Rgba8);
    const Object depth_floats = texture(d_format);
    const Object reds = texture(r32f);
    auto [upload, upload_bytes] = SharedBuffer(device, Side * Side * 4);
    auto [back, back_bytes] = SharedBuffer(device, Side * Side * 4);
    auto [x8_back, x8_bytes] = SharedBuffer(device, Side * Side * 4);
    auto [float_in, float_in_bytes] = SharedBuffer(device, Side * Side * 8);
    auto [red_back, red_bytes] = SharedBuffer(device, Side * Side * 4);
    if (!colour || !depth_stencil || !colour_back || !depth || !colour_x8 || !depth_floats ||
        !reds || upload_bytes == nullptr || back_bytes == nullptr || x8_bytes == nullptr ||
        float_in_bytes == nullptr || red_bytes == nullptr) {
        std::printf("metal-smoke: FAIL conversions: textures\n");
        return 1;
    }
    for (uint32_t index = 0; index < Side * Side * 4; ++index) {
        upload_bytes[index] = static_cast<uint8_t>(index * 7 + 3);
    }
    auto* const float_words = reinterpret_cast<uint32_t*>(float_in_bytes);
    std::vector<float> depths(Side * Side);
    for (uint32_t index = 0; index < Side * Side; ++index) {
        depths[index] = static_cast<float>(index) / 128.0f;
        std::memcpy(&float_words[index * 2], &depths[index], sizeof(float));
    }
    scheduler.CopyBufferToTexture(upload.Get(), 0, Side * 4, 0, colour.Get(), 0, 0, {},
                                  {Side, Side, 1});
    scheduler.BreakBlits();
    const auto convert = [&](const Object& src, const Object& dst, uint32_t format,
                             ConvertKind kind) {
        return scheduler.Convert(Metal::ConvertPass{.src = src.Get(),
                                                    .dst = {.texture = dst.Get()},
                                                    .dst_format = format,
                                                    .width = Side,
                                                    .height = Side,
                                                    .kind = kind});
    };
    const bool to_d24s8 = convert(colour, depth_stencil, ds_format, ConvertKind::Abgr8ToD24s8);
    const bool from_d24s8 = convert(depth_stencil, colour_back, Rgba8, ConvertKind::D24s8ToAbgr8);
    const bool to_x8d24 = convert(colour, depth, d_format, ConvertKind::Abgr8ToX8d24);
    const bool from_x8d24 = convert(depth, colour_x8, Rgba8, ConvertKind::X8d24ToAbgr8);
    const bool floats_up = scheduler.DepthFromBuffer(float_in.Get(), 0, Side * 8,
                                                     depth_floats.Get(), 0, 0, {}, {Side, Side, 1},
                                                     Metal::DepthLayout::Float32_X24S8);
    const bool to_red = convert(depth_floats, reds, r32f, ConvertKind::DepthToColor);
    // Into a format no conversion writes: refused, nothing drawn.
    const bool wrong_refused = !convert(colour, colour_back, Rgba8, ConvertKind::Abgr8ToD24s8);
    scheduler.CopyTextureToBuffer(colour_back.Get(), 0, 0, {}, {Side, Side, 1}, back.Get(), 0,
                                  Side * 4, 0);
    scheduler.CopyTextureToBuffer(colour_x8.Get(), 0, 0, {}, {Side, Side, 1}, x8_back.Get(), 0,
                                  Side * 4, 0);
    scheduler.CopyTextureToBuffer(reds.Get(), 0, 0, {}, {Side, Side, 1}, red_back.Get(), 0,
                                  Side * 4, 0);
    scheduler.Finish();
    uint32_t d24s8_wrong = 0;
    uint32_t x8d24_wrong = 0;
    for (uint32_t index = 0; index < Side * Side; ++index) {
        const uint8_t* const in = upload_bytes + index * 4;
        const uint8_t* const out = back_bytes + index * 4;
        // Red, green and blue go through 24-bit depth, alpha through the stencil.
        d24s8_wrong += Close(in[0], out[0], 1) && Close(in[1], out[1], 1) &&
                               Close(in[2], out[2], 1) && in[3] == out[3]
                           ? 0
                           : 1;
        const uint8_t* const x8 = x8_bytes + index * 4;
        x8d24_wrong += Close(in[0], x8[0], 1) && Close(in[1], x8[1], 1) &&
                               Close(in[2], x8[2], 1) && x8[3] == 0
                           ? 0
                           : 1;
    }
    const auto* const red_values = reinterpret_cast<const float*>(red_bytes);
    uint32_t red_wrong = 0;
    for (uint32_t index = 0; index < Side * Side; ++index) {
        red_wrong += red_values[index] == depths[index] ? 0 : 1;
    }
    const bool passed = to_d24s8 && from_d24s8 && to_x8d24 && from_x8d24 && floats_up &&
                        to_red && wrong_refused && d24s8_wrong == 0 && x8d24_wrong == 0 &&
                        red_wrong == 0;
    std::printf("metal-smoke: conversions: ABGR8 -> D24S8 -> ABGR8 %u wrong, ABGR8 -> X8D24 -> "
                "ABGR8 %u wrong, D32F -> R32F %u wrong (of %u), into the wrong kind %s: %s\n",
                d24s8_wrong, x8d24_wrong, red_wrong, Side * Side,
                wrong_refused ? "refused" : "DRAWN", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// R11, R13: a stencil view of a texture made without format views is refused instead of
/// aborting, and a buffer bound at an offset past its end is swapped for zeros.
int RunGuards() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:8
                                                          height:8
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModePrivate;
    const Object no_views = Metal::Retain([device.GetDevice() newTextureWithDescriptor:desc]);
    const Object target = Metal::CreateTexture(
        device, {.kind = TextureKind::e2D, .format = Rgba8, .width = 8, .height = 8,
                 .render_target = true});
    const Object small = Metal::CreateBuffer(device, 64, false, nullptr);
    if (!no_views || !target || !small) {
        std::printf("metal-smoke: FAIL guards: textures\n");
        return 1;
    }
    (void)Metal::Coverage::TakeLine();
    const bool view_refused = !scheduler.Convert(Metal::ConvertPass{.src = no_views.Get(),
                                                                    .dst = {.texture = target.Get()},
                                                                    .dst_format = Rgba8,
                                                                    .width = 8,
                                                                    .height = 8,
                                                                    .kind = Metal::ConvertKind::D24s8ToAbgr8});
    Metal::RenderTargetSet targets{};
    targets.color[0] = {.texture = target.Get()};
    targets.width = 8;
    targets.height = 8;
    scheduler.BeginRendering(targets);
    scheduler.BindVertexBuffer(0, small.Get(), 128);
    scheduler.BindBuffer(Metal::ShaderStage::Fragment, 0, small.Get(), 64);
    scheduler.EndRendering();
    scheduler.Finish();
    const std::string line = Metal::Coverage::TakeLine();
    const bool offsets_guarded = line.find("buffer offset past its end 2") != std::string::npos;
    const bool passed = view_refused && offsets_guarded;
    std::printf("metal-smoke: guards: stencil view of a texture without format views %s; "
                "offsets past the end (\"%s\"): %s\n",
                view_refused ? "refused" : "MADE", line.c_str(), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// R12 and R15: a guest target of 8 samples gets the device's most, and every placeholder
/// texture is made and filled with zeros.
int RunSamplesAndPlaceholders() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const Metal::PipelineLimits limits = Metal::DevicePipelineLimits(device);
    const uint32_t samples = Metal::SupportedSamples(8, limits);
    const Metal::TextureDesc eight{.kind = TextureKind::e2DMultisampleArray,
                                   .format = Rgba8,
                                   .width = 4,
                                   .height = 4,
                                   .samples = samples,
                                   .render_target = true};
    const bool checked = Metal::CheckTexture(eight, limits).empty();
    const Object multisampled = checked ? Metal::CreateTexture(device, eight) : Object{};
    const bool device_eight = [device.GetDevice() supportsTextureSampleCount:8];
    const bool samples_right = device_eight ? samples == 8 : samples == 4;
    auto [zeros, zero_bytes] = SharedBuffer(device, 256);
    auto [back, back_bytes] = SharedBuffer(device, 4);
    uint32_t made = 0;
    uint32_t wanted = 0;
    bool zero = true;
    for (uint32_t type = 0; type < Shader::NUM_TEXTURE_TYPES; ++type) {
        if (type == static_cast<uint32_t>(Shader::TextureType::Buffer)) {
            continue;
        }
        ++wanted;
        const Metal::TextureDesc desc = Metal::PlaceholderTexture(type);
        const Object placeholder = Metal::CreateTexture(device, desc);
        if (!placeholder || zero_bytes == nullptr || back_bytes == nullptr) {
            continue;
        }
        ++made;
        for (uint32_t layer = 0; layer < desc.layers; ++layer) {
            scheduler.CopyBufferToTexture(zeros.Get(), 0, 4, 4, placeholder.Get(), 0, layer, {},
                                          {});
        }
        scheduler.BreakBlits();
        std::memset(back_bytes, 0xFF, 4);
        scheduler.CopyTextureToBuffer(placeholder.Get(), 0, desc.layers - 1, {}, {}, back.Get(),
                                      0, 4, 4);
        scheduler.Finish();
        zero = zero && back_bytes[0] == 0 && back_bytes[1] == 0 && back_bytes[2] == 0 &&
               back_bytes[3] == 0;
    }
    const bool passed = checked && multisampled && samples_right && made == wanted && zero;
    std::printf("metal-smoke: 8 samples drawn at %u (device has 8: %s), texture %s; "
                "placeholders %u of %u made, %s: %s\n",
                samples, device_eight ? "yes" : "no", multisampled ? "made" : "NOT MADE", made,
                wanted, zero ? "zeros" : "NOT ZERO", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// E04, E11: formats Metal lacks get the wider one and CPU-expanded texels.
int RunExpandedFormats() {
    using VideoCore::Surface::PixelFormat;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const bool converted = Metal::IsConverted(device, PixelFormat::G4R4_UNORM) &&
                           Metal::IsConverted(device, PixelFormat::R32G32B32_FLOAT);
    const Metal::FormatInfo rg8 = Metal::SurfaceFormat(device, PixelFormat::G4R4_UNORM, true);
    const Metal::FormatInfo rgba32 =
        Metal::SurfaceFormat(device, PixelFormat::R32G32B32_FLOAT, true);
    constexpr uint32_t Texels = 4;
    const std::vector<uint8_t> guest{0xF0, 0x0F, 0x5A, 0x00};
    std::vector<uint8_t> expanded(Texels * 2);
    VideoCommon::BufferImageCopy copy{};
    copy.buffer_row_length = Texels;
    copy.buffer_image_height = 1;
    copy.image_subresource.num_layers = 1;
    copy.image_extent = {Texels, 1, 1};
    VideoCommon::ExpandTexels(guest, expanded, copy, PixelFormat::G4R4_UNORM);
    const Object texture = Metal::CreateTexture(
        device, {.kind = TextureKind::e2D, .format = rg8.format, .width = Texels, .height = 1});
    const Object wide = Metal::CreateTexture(
        device, {.kind = TextureKind::e2D, .format = rgba32.format, .width = 2, .height = 2});
    auto [upload, upload_bytes] = SharedBuffer(device, expanded.size());
    auto [back, back_bytes] = SharedBuffer(device, expanded.size());
    if (!texture || !wide || upload_bytes == nullptr || back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL expanded formats: textures (RG8 %u, RGBA32F %u)\n",
                    rg8.format, rgba32.format);
        return 1;
    }
    std::memcpy(upload_bytes, expanded.data(), expanded.size());
    scheduler.CopyBufferToTexture(upload.Get(), 0, Texels * 2, 0, texture.Get(), 0, 0, {},
                                  {Texels, 1, 1});
    scheduler.BreakBlits();
    scheduler.CopyTextureToBuffer(texture.Get(), 0, 0, {}, {Texels, 1, 1}, back.Get(), 0,
                                  Texels * 2, 0);
    scheduler.Finish();
    const std::vector<uint8_t> expected{0xFF, 0x00, 0x00, 0xFF, 0x55, 0xAA, 0x00, 0x00};
    const bool same = std::memcmp(back_bytes, expected.data(), expected.size()) == 0;
    const bool passed = converted && same;
    std::printf("metal-smoke: expanded formats: G4R4 as RG8 %s, R32G32B32F as RGBA32F made, "
                "converted %s: %s\n",
                same ? "read back" : "WRONG", converted ? "yes" : "NO", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

constexpr const char* StorageSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void write_texels(uint2 id [[thread_position_in_grid]],
                         texture2d<float, access::write> image [[texture(0)]]) {
    image.write(float4(0.25, 0.5, 0.75, 1.0), id);
}
)";

/// F09: an sRGB image written by a program through a view of its linear format (as the game's
/// UNORM views of its sRGB images): the texture cache gives such textures shader-write usage.
int RunSrgbStorage() {
    using VideoCore::Surface::PixelFormat;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const id<MTLDevice> mtl = device.GetDevice();
    const Metal::FormatInfo srgb = Metal::SurfaceFormat(device, PixelFormat::A8B8G8R8_SRGB, false);
    const uint32_t linear = Metal::LinearTwin(srgb.format);
    const Object texture = Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                                         .format = srgb.format,
                                                         .width = 4,
                                                         .height = 4,
                                                         .shader_write = linear != 0});
    const Object view = texture ? Metal::CreateTextureView(texture.Get(), linear, TextureKind::e2D,
                                                           0, 1, 0, 1, Metal::IdentitySwizzle)
                                : Object{};
    NSError* error = nil;
    const id<MTLLibrary> library = [mtl newLibraryWithSource:@(StorageSource)
                                                     options:[MTLCompileOptions new]
                                                       error:&error];
    const id<MTLComputePipelineState> kernel =
        library != nil
            ? [mtl newComputePipelineStateWithFunction:[library newFunctionWithName:@"write_texels"]
                                                 error:&error]
            : nil;
    auto [back, back_bytes] = SharedBuffer(device, 4 * 4 * 4);
    if (linear == 0 || !texture || !view || kernel == nil || back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL sRGB storage: linear twin %u, texture %d, view %d, %s\n",
                    linear, static_cast<bool>(texture), static_cast<bool>(view),
                    error != nil ? error.localizedDescription.UTF8String : "kernel built");
        return 1;
    }
    const id<MTLCommandQueue> queue = [mtl newCommandQueue];
    const id<MTLCommandBuffer> commands = [queue commandBuffer];
    const id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:kernel];
    [encoder setTexture:Metal::AsTexture(view.Get()) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(4, 4, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    scheduler.CopyTextureToBuffer(texture.Get(), 0, 0, {}, {4, 4, 1}, back.Get(), 0, 16, 0);
    scheduler.Finish();
    uint32_t wrong = 0;
    for (uint32_t texel = 0; texel < 16; ++texel) {
        const uint8_t* const bytes = back_bytes + texel * 4;
        // The linear view stores the values as they are (no sRGB encoding).
        wrong += Close(bytes[0], 64, 1) && Close(bytes[1], 128, 1) && Close(bytes[2], 191, 1) &&
                         bytes[3] == 255
                     ? 0
                     : 1;
    }
    const bool passed = wrong == 0;
    std::printf("metal-smoke: sRGB storage through a linear view: %u of 16 texels wrong: %s\n",
                wrong, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

constexpr const char* LayerSource = R"(
#include <metal_stdlib>
using namespace metal;

struct LayerOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};

vertex LayerOut layer_vertex(uint id [[vertex_id]], uint instance [[instance_id]]) {
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    LayerOut out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.layer = instance;
    return out;
}

fragment float4 layer_fragment(LayerOut in [[stage_in]]) {
    return in.layer == 0 ? float4(1.0, 0.0, 0.0, 1.0) : float4(0.0, 0.0, 1.0, 1.0);
}
)";

/// F11: a program that picks the layer draws into a 2-layer target through the scheduler's
/// render pass (RenderTargetSet::layers -> renderTargetArrayLength).
int RunLayeredTarget() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const id<MTLDevice> mtl = device.GetDevice();
    NSError* error = nil;
    const id<MTLLibrary> library = [mtl newLibraryWithSource:@(LayerSource)
                                                     options:[MTLCompileOptions new]
                                                       error:&error];
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"layer_vertex"];
    desc.fragmentFunction = [library newFunctionWithName:@"layer_fragment"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    const Object pipeline = Metal::Retain(
        library != nil ? [mtl newRenderPipelineStateWithDescriptor:desc error:&error] : nil);
    const Object depth_stencil =
        Metal::Retain([mtl newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]]);
    constexpr uint32_t Side = 4;
    const Object target = Metal::CreateTexture(device, {.kind = TextureKind::e2DArray,
                                                        .format = Rgba8,
                                                        .width = Side,
                                                        .height = Side,
                                                        .layers = 2,
                                                        .render_target = true});
    auto [back, back_bytes] = SharedBuffer(device, Side * Side * 4 * 2);
    if (!pipeline || !depth_stencil || !target || back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL layered target: %s\n",
                    error != nil ? error.localizedDescription.UTF8String : "textures");
        return 1;
    }
    Metal::RenderTargetSet targets{};
    targets.color[0] = {.texture = target.Get()};
    targets.width = Side;
    targets.height = Side;
    targets.layers = 2;
    scheduler.BeginRendering(targets);
    Metal::DrawState state{};
    state.pipeline = pipeline.Get();
    state.depth_stencil = depth_stencil.Get();
    state.viewport = {0.0f, 0.0f, float(Side), float(Side), 0.0f, 1.0f};
    state.scissor = {0, 0, Side, Side};
    scheduler.SetDrawState(state);
    scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 3, 2, 0);
    scheduler.EndRendering();
    for (uint32_t layer = 0; layer < 2; ++layer) {
        scheduler.CopyTextureToBuffer(target.Get(), 0, layer, {}, {Side, Side, 1}, back.Get(),
                                      layer * Side * Side * 4, Side * 4, 0);
    }
    scheduler.Finish();
    const auto texel = [&](uint32_t layer) {
        uint32_t value = 0;
        std::memcpy(&value, back_bytes + layer * Side * Side * 4 + (Side + 1) * 4, sizeof(value));
        return value;
    };
    const uint32_t first = texel(0);
    const uint32_t second = texel(1);
    const bool passed = first == 0xFF0000FFU && second == 0xFFFF0000U;
    std::printf("metal-smoke: layered target, layer from the program: layer 0 %08x (red), "
                "layer 1 %08x (blue): %s\n",
                first, second, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// M07: a read-only storage binding whose guest range spans two mappings gets the two segments
/// gathered into one buffer (BufferCacheRuntime::BindMultiRangeStorageBuffer); a written one is
/// left to the plain path.
int RunMultiRange() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Metal::StagingBufferPool pool(device, scheduler);
    Metal::BufferCacheRuntime runtime(device, scheduler, pool);
    auto [first, first_bytes] = SharedBuffer(device, 64);
    auto [second, second_bytes] = SharedBuffer(device, 64);
    auto [back, back_bytes] = SharedBuffer(device, 48);
    if (first_bytes == nullptr || second_bytes == nullptr || back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL multi-range: buffers\n");
        return 1;
    }
    for (uint32_t index = 0; index < 64; ++index) {
        first_bytes[index] = static_cast<uint8_t>(index);
        second_bytes[index] = static_cast<uint8_t>(0x80 + index);
    }
    // 16 bytes from offset 48 of the first mapping, then 32 from offset 4 of the second.
    runtime.ResetMultiRange();
    runtime.PushMultiRangeSource(first.Get(), 48, 16);
    runtime.PushMultiRangeSource(second.Get(), 4, 32);
    const bool written_refused = !runtime.BindMultiRangeStorageBuffer(7, true);
    const bool bound = runtime.BindMultiRangeStorageBuffer(7, false);
    const auto& bindings = runtime.PendingBindings();
    const Metal::Binding gathered = bindings.empty() ? Metal::Binding{} : bindings.back();
    const std::array copy{VideoCommon::BufferCopy{.src_offset = 0, .dst_offset = 0, .size = 48}};
    if (gathered.buffer != nullptr) {
        runtime.CopyBuffer(back.Get(), gathered.buffer, copy, false);
    }
    scheduler.Finish();
    uint32_t wrong = 0;
    for (uint32_t index = 0; index < 48; ++index) {
        const uint8_t expected = index < 16 ? static_cast<uint8_t>(48 + index)
                                            : static_cast<uint8_t>(0x80 + 4 + index - 16);
        wrong += back_bytes[index] == expected ? 0 : 1;
    }
    const bool passed = written_refused && bound && gathered.size == 48 && wrong == 0;
    std::printf("metal-smoke: multi-range storage binding of 2 segments: %s, %u bytes, %u of 48 "
                "wrong; written binding %s: %s\n",
                bound ? "gathered" : "NOT BOUND", gathered.size, wrong,
                written_refused ? "left to the plain path" : "GATHERED", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// M05: a thread that is not the recording one (the GPU's front thread reading command lists,
/// a guest core landing a download) waits for a committed tick without touching the command
/// buffers in flight (BufferCacheRuntime::WaitSubmittedTick).
int RunCrossThreadWait() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Metal::StagingBufferPool pool(device, scheduler);
    Metal::BufferCacheRuntime runtime(device, scheduler, pool);
    auto [target, target_bytes] = SharedBuffer(device, 4096);
    if (target_bytes == nullptr) {
        std::printf("metal-smoke: FAIL cross-thread wait: buffer\n");
        return 1;
    }
    scheduler.FillBuffer(target.Get(), 0, 4096, 0x5A);
    const uint64_t tick = scheduler.CurrentTick();
    scheduler.Flush();
    const bool submitted = scheduler.IsSubmitted(tick);
    bool free_after = false;
    std::thread waiter([&] {
        runtime.WaitSubmittedTick(tick);
        free_after = scheduler.IsFree(tick);
    });
    waiter.join();
    const bool filled = target_bytes[0] == 0x5A && target_bytes[4095] == 0x5A;
    const bool passed = submitted && free_after && filled;
    std::printf("metal-smoke: wait for a committed tick from another thread: submitted %s, done "
                "%s, contents %s: %s\n",
                submitted ? "yes" : "NO", free_after ? "yes" : "NO", filled ? "written" : "NOT",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

constexpr const char* SamplerSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut sampler_vertex(uint id [[vertex_id]]) {
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    VertexOut out;
    out.position = float4(positions[id], 0.0, 1.0);
    return out;
}

// uv = position * p.xy + p.zw: a constant point, or one texel a pixel.
fragment float4 sampler_fragment(VertexOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                                 sampler s [[sampler(0)]], constant float4& p [[buffer(0)]]) {
    return t.sample(s, in.position.xy * p.xy + p.zw);
}
)";

/// F14-F18: border colours, the mirror-once modes, LOD bias and min / max reduction as the
/// renderer's CreateSampler makes them from a guest sampler, sampled by a draw.
int RunSamplers() {
    using Tegra::Texture::SamplerReduction;
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    using Tegra::Texture::TSCEntry;
    using Tegra::Texture::WrapMode;
    Metal::Device device;
    const id<MTLDevice> mtl = device.GetDevice();
    NSError* error = nil;
    const id<MTLLibrary> library = [mtl newLibraryWithSource:@(SamplerSource)
                                                     options:[MTLCompileOptions new]
                                                       error:&error];
    MTLRenderPipelineDescriptor* pipeline_desc = [MTLRenderPipelineDescriptor new];
    pipeline_desc.vertexFunction = [library newFunctionWithName:@"sampler_vertex"];
    pipeline_desc.fragmentFunction = [library newFunctionWithName:@"sampler_fragment"];
    pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    const id<MTLRenderPipelineState> pipeline =
        library != nil ? [mtl newRenderPipelineStateWithDescriptor:pipeline_desc error:&error] : nil;
    if (pipeline == nil) {
        std::printf("metal-smoke: FAIL samplers: %s\n",
                    error != nil ? error.localizedDescription.UTF8String : "no pipeline");
        return 1;
    }
    const id<MTLCommandQueue> queue = [mtl newCommandQueue];
    const auto texture = [&](uint32_t width, uint32_t height, uint32_t levels,
                             const std::vector<std::vector<uint32_t>>& texels) {
        MTLTextureDescriptor* desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:width
                                                              height:height
                                                           mipmapped:levels > 1];
        desc.mipmapLevelCount = levels;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        const id<MTLTexture> result = [mtl newTextureWithDescriptor:desc];
        for (uint32_t level = 0; level < levels; ++level) {
            const uint32_t w = std::max(width >> level, 1U);
            const uint32_t h = std::max(height >> level, 1U);
            [result replaceRegion:MTLRegionMake2D(0, 0, w, h)
                      mipmapLevel:level
                        withBytes:texels[level].data()
                      bytesPerRow:w * 4];
        }
        return result;
    };
    constexpr uint32_t Red = 0xFF0000FF;
    constexpr uint32_t Green = 0xFF00FF00;
    const id<MTLTexture> red_green = texture(2, 1, 1, {{Red, Green}});
    const id<MTLTexture> mips = texture(2, 2, 2, {{Red, Red, Red, Red}, {Green}});
    const id<MTLTexture> black_white = texture(2, 1, 1, {{0xFF000000, 0xFFFFFFFF}});
    MTLTextureDescriptor* target_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:2
                                                          height:2
                                                       mipmapped:NO];
    target_desc.usage = MTLTextureUsageRenderTarget;
    target_desc.storageMode = MTLStorageModeShared;
    const id<MTLTexture> target = [mtl newTextureWithDescriptor:target_desc];

    // The first pixel of a draw sampling `source` with the guest sampler.
    const auto draw = [&](id<MTLTexture> source, const TSCEntry& tsc, std::array<float, 4> p) {
        const Object sampler = Metal::CreateSampler(device, tsc, {});
        const id<MTLCommandBuffer> commands = [queue commandBuffer];
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        const id<MTLRenderCommandEncoder> encoder =
            [commands renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:pipeline];
        [encoder setFragmentTexture:source atIndex:0];
        [encoder setFragmentSamplerState:Metal::AsSampler(sampler.Get()) atIndex:0];
        [encoder setFragmentBytes:p.data() length:sizeof(p) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        uint32_t texel = 0;
        [target getBytes:&texel bytesPerRow:8 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
        return texel;
    };
    const auto tsc_of = [](WrapMode wrap, TextureFilter filter) {
        TSCEntry tsc{};
        tsc.raw = {};
        tsc.wrap_u.Assign(wrap);
        tsc.wrap_v.Assign(WrapMode::ClampToEdge);
        tsc.wrap_p.Assign(WrapMode::ClampToEdge);
        tsc.mag_filter.Assign(filter);
        tsc.min_filter.Assign(filter);
        tsc.mipmap_filter.Assign(TextureMipmapFilter::None);
        tsc.max_lod_clamp.Assign(256);
        return tsc;
    };
    const std::array<float, 4> Outside{0.0f, 0.0f, 1.5f, 0.5f};
    const std::array<float, 4> Mirrored{0.0f, 0.0f, -0.75f, 0.5f};
    uint32_t wrong = 0;
    std::string report;
    const auto expect = [&](const char* name, uint32_t got, uint32_t wanted, int tolerance = 1) {
        const bool right = [&] {
            for (uint32_t shift = 0; shift < 32; shift += 8) {
                if (!Close(static_cast<int>((got >> shift) & 0xFF),
                           static_cast<int>((wanted >> shift) & 0xFF), tolerance)) {
                    return false;
                }
            }
            return true;
        }();
        wrong += right ? 0 : 1;
        char text[96];
        std::snprintf(text, sizeof(text), "%s%s %08x%s", report.empty() ? "" : ", ", name, got,
                      right ? "" : " WRONG");
        report += text;
    };
    TSCEntry border = tsc_of(WrapMode::Border, TextureFilter::Nearest);
    border.border_color = {0.9f, 0.9f, 0.9f, 1.0f};
    expect("border near white", draw(red_green, border, Outside), 0xFFFFFFFF);
    border.border_color = {0.2f, 0.3f, 0.4f, 0.1f};
    expect("border near transparent", draw(red_green, border, Outside), 0x00000000);
    // Linear OpenGL clamp reaches the border, as Vulkan's.
    TSCEntry clamp = tsc_of(WrapMode::Clamp, TextureFilter::Linear);
    clamp.border_color = {1.0f, 1.0f, 1.0f, 1.0f};
    expect("clamp linear", draw(red_green, clamp, Outside), 0xFFFFFFFF);
    expect("mirror once to edge",
           draw(red_green, tsc_of(WrapMode::MirrorOnceClampToEdge, TextureFilter::Nearest),
                Mirrored),
           Green);
    expect("mirror once to border",
           draw(red_green, tsc_of(WrapMode::MirrorOnceBorder, TextureFilter::Nearest), Mirrored),
           Green);
    expect("mirror once OpenGL",
           draw(red_green, tsc_of(WrapMode::MirrorOnceClampOGL, TextureFilter::Nearest),
                Mirrored),
           Green);
    const bool bias = Metal::SamplerBiasSupported(device);
    if (bias) {
        const std::array<float, 4> PerTexel{0.5f, 0.5f, 0.0f, 0.0f};
        TSCEntry mip = tsc_of(WrapMode::ClampToEdge, TextureFilter::Nearest);
        mip.mipmap_filter.Assign(TextureMipmapFilter::Nearest);
        expect("lod bias 0", draw(mips, mip, PerTexel), Red);
        mip.mip_lod_bias.Assign(256); // 1.0
        expect("lod bias 1", draw(mips, mip, PerTexel), Green);
        // Metal reduces only with linear min, mag and mip filters.
        const std::array<float, 4> Middle{0.0f, 0.0f, 0.5f, 0.5f};
        TSCEntry reduce = tsc_of(WrapMode::ClampToEdge, TextureFilter::Linear);
        reduce.mipmap_filter.Assign(TextureMipmapFilter::Linear);
        expect("average", draw(black_white, reduce, Middle), 0xFF808080, 2);
        reduce.reduction_filter.Assign(SamplerReduction::Min);
        expect("min", draw(black_white, reduce, Middle), 0xFF000000);
        reduce.reduction_filter.Assign(SamplerReduction::Max);
        expect("max", draw(black_white, reduce, Middle), 0xFFFFFFFF);
    }
    const bool passed = wrong == 0;
    std::printf("metal-smoke: samplers: %s%s: %s\n", report.c_str(),
                bias ? "" : "; LOD bias and reduction SKIP (need the macOS / iOS 26 SDK and OS on an Apple7 GPU)",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

} // Anonymous namespace

/// E09, E10: the 16-bit packed formats Metal has with red and blue exchanged are render targets
/// holding Metal's order; the swap on the way to guest memory (Scheduler::SwapPacked, and
/// SwapPackedTexels on the CPU) gives the guest's bits (red at bit 0).
int RunPackedTargets() {
    using VideoCore::Surface::PixelFormat;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    struct Case {
        PixelFormat format;
        std::array<double, 4> color;
        uint16_t metal; ///< what Metal's render target holds for the colour
        uint16_t guest; ///< and the guest's bits
        const char* name;
    };
    const std::array<Case, 4> cases{{
        {PixelFormat::B5G6R5_UNORM, {1, 0, 0, 1}, 0xF800, 0x001F, "B5G6R5"},
        {PixelFormat::B5G6R5_UNORM, {0, 0, 1, 1}, 0x001F, 0xF800, "B5G6R5 blue"},
        {PixelFormat::A1B5G5R5_UNORM, {1, 0, 0, 1}, 0xFC00, 0x801F, "A1B5G5R5"},
        {PixelFormat::A4B4G4R4_UNORM, {1, 0, 0, 0}, 0xF000, 0x000F, "A4B4G4R4"},
    }};
    int failed = 0;
    std::string report;
    for (const Case& test : cases) {
        const Metal::FormatInfo host = Metal::SurfaceFormat(device, test.format, false);
        const bool plain = host.attachable && host.swizzle == Metal::IdentitySwizzle &&
                           Metal::PackedSwapShift(test.format) != 0;
        const Object target = Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                                            .format = host.format,
                                                            .width = 8,
                                                            .height = 8,
                                                            .render_target = host.attachable});
        auto [back, back_bytes] = SharedBuffer(device, 8 * 8 * 2);
        if (!plain || !target || back_bytes == nullptr) {
            report += std::string{" "} + test.name + " NOT A TARGET";
            ++failed;
            continue;
        }
        Metal::ClearPass clear{};
        clear.color[0] = {.texture = target.Get()};
        clear.clear_color[0] = true;
        clear.color_value[0] = test.color;
        clear.width = 8;
        clear.height = 8;
        scheduler.Clear(clear);
        scheduler.BreakBlits();
        scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {8, 8, 1}, back.Get(), 0, 16, 0);
        scheduler.BreakBlits();
        scheduler.Finish();
        uint16_t metal_texel{};
        std::memcpy(&metal_texel, back_bytes, 2);
        // GPU swap in place, then the CPU's over the same bytes must undo it.
        const bool gpu = scheduler.SwapPacked(back.Get(), 0, 8, 8, 1, 16, 128,
                                              Metal::PackedSwapShift(test.format));
        scheduler.Finish();
        uint16_t guest_texel{};
        std::memcpy(&guest_texel, back_bytes, 2);
        Metal::SwapPackedTexels(back_bytes, 8, 8, 1, 16, 128, Metal::PackedSwapShift(test.format));
        uint16_t again{};
        std::memcpy(&again, back_bytes, 2);
        const bool ok = metal_texel == test.metal && gpu && guest_texel == test.guest &&
                        again == test.metal;
        if (!ok) {
            ++failed;
            report += fmt::format(" {} WRONG (metal {:04x} want {:04x}, guest {:04x} want {:04x})",
                                  test.name, metal_texel, test.metal, guest_texel, test.guest);
        }
    }
    std::printf("metal-smoke: packed 16-bit render targets (B5G6R5, A1B5G5R5, A4B4G4R4) hold "
                "Metal's order, swapped to guest bits on the GPU and CPU:%s %s\n",
                report.c_str(), failed == 0 ? "PASS" : "FAIL");
    return failed == 0 ? 0 : 1;
}

constexpr const char* ShadowSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_one_per_half(uint2 id [[thread_position_in_grid]],
                             texture2d<uint, access::read_write> image [[texture(0)]]) {
    image.write(image.read(id) + 0x00010001u, id);
}
)";

/// G19: a packed R16 surface a program addresses bytewise goes through an R32Uint shadow of half
/// the width (RasterizerMetal::DispatchCompute): texels through a buffer into the shadow, the
/// program works on it, and back.
int RunByteShadow() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const id<MTLDevice> mtl = device.GetDevice();
    constexpr uint32_t Width = 8;
    constexpr uint32_t Height = 4;
    const Object original = Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                                          .format = Metal::Fmt::R16Float,
                                                          .width = Width,
                                                          .height = Height});
    const Object shadow = Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                                        .format = Metal::Fmt::R32Uint,
                                                        .width = Width / 2,
                                                        .height = Height,
                                                        .shader_write = true});
    auto [bridge, bridge_bytes] = SharedBuffer(device, Width * Height * 2);
    NSError* error = nil;
    const id<MTLLibrary> library = [mtl newLibraryWithSource:@(ShadowSource)
                                                     options:[MTLCompileOptions new]
                                                       error:&error];
    const id<MTLComputePipelineState> kernel =
        library != nil ? [mtl newComputePipelineStateWithFunction:
                                  [library newFunctionWithName:@"add_one_per_half"]
                                                            error:&error]
                       : nil;
    if (!original || !shadow || bridge_bytes == nullptr || kernel == nil) {
        std::printf("metal-smoke: FAIL byte-addressed shadow: textures, buffer or kernel (%s)\n",
                    error != nil ? error.localizedDescription.UTF8String : "made");
        return 1;
    }
    std::array<uint16_t, Width * Height> texels{};
    for (uint32_t index = 0; index < texels.size(); ++index) {
        texels[index] = static_cast<uint16_t>(0x1000 + index * 3);
    }
    std::memcpy(bridge_bytes, texels.data(), sizeof(texels));
    scheduler.CopyBufferToTexture(bridge.Get(), 0, Width * 2, 0, original.Get(), 0, 0, {},
                                  {Width, Height, 1});
    scheduler.Finish();
    // In: the original's bytes through the bridge into the shadow.
    std::memset(bridge_bytes, 0, sizeof(texels));
    scheduler.CopyTextureToBuffer(original.Get(), 0, 0, {}, {Width, Height, 1}, bridge.Get(), 0,
                                  Width * 2, 0);
    scheduler.BreakBlits();
    scheduler.CopyBufferToTexture(bridge.Get(), 0, Width * 2, 0, shadow.Get(), 0, 0, {},
                                  {Width / 2, Height, 1});
    scheduler.Finish();
    const id<MTLCommandQueue> queue = [mtl newCommandQueue];
    const id<MTLCommandBuffer> commands = [queue commandBuffer];
    const id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:kernel];
    [encoder setTexture:Metal::AsTexture(shadow.Get()) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(Width / 2, Height, 1)
        threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    // Out: the shadow's texels through the bridge into the original, then read back.
    scheduler.CopyTextureToBuffer(shadow.Get(), 0, 0, {}, {Width / 2, Height, 1}, bridge.Get(), 0,
                                  Width * 2, 0);
    scheduler.BreakBlits();
    scheduler.CopyBufferToTexture(bridge.Get(), 0, Width * 2, 0, original.Get(), 0, 0, {},
                                  {Width, Height, 1});
    scheduler.BreakBlits();
    std::memset(bridge_bytes, 0, sizeof(texels));
    scheduler.CopyTextureToBuffer(original.Get(), 0, 0, {}, {Width, Height, 1}, bridge.Get(), 0,
                                  Width * 2, 0);
    scheduler.Finish();
    uint32_t wrong = 0;
    for (uint32_t index = 0; index < texels.size(); ++index) {
        uint16_t value{};
        std::memcpy(&value, bridge_bytes + index * 2, 2);
        wrong += value == static_cast<uint16_t>(texels[index] + 1) ? 0 : 1;
    }
    const bool passed = wrong == 0;
    std::printf("metal-smoke: packed R16 surface through its R32 shadow: %u of %u halves wrong: "
                "%s\n",
                wrong, Width * Height, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// G18: a texture over a sparse heap takes memory for the tiles mapped only; a mapped tile holds
/// what is written to it, an unmapped one reads zero, and unmapping gives the page back.
int RunSparseTiles() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    if (!Metal::SupportsSparseTextures(device)) {
        std::printf("metal-smoke: sparse tiles: this GPU or OS has no tier 2 sparse textures: "
                    "SKIP\n");
        return 0;
    }
    constexpr uint32_t Size = 2048;
    auto sparse = Metal::CreateSparseTexture(device, {.kind = TextureKind::e2D,
                                                      .format = Metal::Fmt::R8Unorm,
                                                      .width = Size,
                                                      .height = Size}, 4u << 20);
    if (!sparse) {
        std::printf("metal-smoke: FAIL sparse tiles: texture not made\n");
        return 1;
    }
    const uint32_t tw = sparse->tile_width;
    const uint32_t th = sparse->tile_height;
    auto [upload, upload_bytes] = SharedBuffer(device, size_t{tw} * th);
    auto [back, back_bytes] = SharedBuffer(device, size_t{tw} * th * 2);
    if (upload_bytes == nullptr || back_bytes == nullptr) {
        return 1;
    }
    for (uint32_t index = 0; index < tw * th; ++index) {
        upload_bytes[index] = static_cast<uint8_t>(index * 7 + 1);
    }
    const uint64_t before = Metal::SparseHeapUsedBytes(sparse->heap.Get());
    const std::array<Metal::SparseTileOp, 1> map{{{.x = 1, .y = 1, .width = 1, .height = 1}}};
    scheduler.UpdateSparseTiles(sparse->texture.Get(), map);
    scheduler.CopyBufferToTexture(upload.Get(), 0, tw, 0, sparse->texture.Get(), 0, 0,
                                  {tw, th, 0}, {tw, th, 1});
    scheduler.BreakBlits();
    scheduler.CopyTextureToBuffer(sparse->texture.Get(), 0, 0, {tw, th, 0}, {tw, th, 1},
                                  back.Get(), 0, tw, 0);
    scheduler.CopyTextureToBuffer(sparse->texture.Get(), 0, 0, {tw * 3, th * 3, 0}, {tw, th, 1},
                                  back.Get(), size_t{tw} * th, tw, 0);
    scheduler.Finish();
    const uint64_t mapped_bytes = Metal::SparseHeapUsedBytes(sparse->heap.Get());
    const bool mapped_ok = std::memcmp(back_bytes, upload_bytes, size_t{tw} * th) == 0;
    bool unmapped_zero = true;
    for (uint32_t index = 0; index < tw * th; ++index) {
        unmapped_zero = unmapped_zero && back_bytes[size_t{tw} * th + index] == 0;
    }
    const std::array<Metal::SparseTileOp, 1> unmap{{{.x = 1, .y = 1, .map = false}}};
    scheduler.UpdateSparseTiles(sparse->texture.Get(), unmap);
    scheduler.Finish();
    const uint64_t after = Metal::SparseHeapUsedBytes(sparse->heap.Get());
    const bool passed = mapped_ok && unmapped_zero && before == 0 &&
                        mapped_bytes == sparse->tile_bytes && after == 0;
    std::printf("metal-smoke: sparse tiles (%ux%u texels, %u bytes): mapped tile %s, unmapped "
                "reads %s, heap used %llu -> %llu -> %llu bytes: %s\n",
                tw, th, sparse->tile_bytes, mapped_ok ? "holds its data" : "WRONG",
                unmapped_zero ? "zero" : "NOT ZERO", static_cast<unsigned long long>(before),
                static_cast<unsigned long long>(mapped_bytes),
                static_cast<unsigned long long>(after), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// P07: an attachment a pass loads and no draw of it writes is not stored again on Apple GPUs
/// (MTLStoreActionDontCare), and what the image held before is still there afterwards; one a draw
/// writes is stored.
int RunUnwrittenStore() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const id<MTLDevice> mtl = device.GetDevice();
    NSError* error = nil;
    const id<MTLLibrary> library = [mtl newLibraryWithSource:@(LayerSource)
                                                     options:[MTLCompileOptions new]
                                                       error:&error];
    const auto make = [&](MTLColorWriteMask mask) {
        MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
        desc.vertexFunction = [library newFunctionWithName:@"layer_vertex"];
        desc.fragmentFunction = [library newFunctionWithName:@"layer_fragment"];
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        desc.colorAttachments[0].writeMask = mask;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        return Metal::Retain(library != nil ? [mtl newRenderPipelineStateWithDescriptor:desc
                                                                                    error:&error]
                                            : nil);
    };
    const Object writes = make(MTLColorWriteMaskAll);
    const Object keeps = make(MTLColorWriteMaskNone);
    const Object depth_stencil =
        Metal::Retain([mtl newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]]);
    constexpr uint32_t Side = 4;
    const Object target = Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                                        .format = Rgba8,
                                                        .width = Side,
                                                        .height = Side,
                                                        .render_target = true});
    auto [back, back_bytes] = SharedBuffer(device, Side * Side * 4);
    if (!writes || !keeps || !target || back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL unwritten store: %s\n",
                    error != nil ? error.localizedDescription.UTF8String : "textures");
        return 1;
    }
    Metal::RenderTargetSet targets{};
    targets.color[0] = {.texture = target.Get()};
    targets.width = Side;
    targets.height = Side;
    const auto draw = [&](const Object& pipeline, uint8_t written) {
        scheduler.BeginRendering(targets);
        Metal::DrawState state{};
        state.pipeline = pipeline.Get();
        state.depth_stencil = depth_stencil.Get();
        state.viewport = {0.0f, 0.0f, float(Side), float(Side), 0.0f, 1.0f};
        state.scissor = {0, 0, Side, Side};
        state.color_written = written;
        scheduler.SetDrawState(state);
        scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 3, 1, 0);
        scheduler.EndRendering();
    };
    // Cleared to blue, drawn red (stored); a pass drawing nothing to it (not stored, twice).
    Metal::ClearPass clear{};
    clear.color[0] = {.texture = target.Get()};
    clear.clear_color[0] = true;
    clear.color_value[0] = {0, 0, 1, 1};
    clear.width = Side;
    clear.height = Side;
    scheduler.Clear(clear);
    scheduler.BreakBlits();
    const uint64_t before = scheduler.StoresDropped();
    draw(writes, 0x01);
    const uint64_t after_write = scheduler.StoresDropped();
    draw(keeps, 0x00);
    draw(keeps, 0x00);
    const uint64_t after_keep = scheduler.StoresDropped();
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Side, Side, 1}, back.Get(), 0,
                                  Side * 4, 0);
    scheduler.Finish();
    uint32_t texel = 0;
    std::memcpy(&texel, back_bytes + (Side + 1) * 4, sizeof(texel));
    const uint64_t expected = device.AppleFamily() != 0 ? 2 : 0;
    const bool passed = texel == 0xFF0000FFU && after_write == before &&
                        after_keep - after_write == expected;
    std::printf("metal-smoke: unwritten attachments: image %08x (red), %llu stores dropped (%llu "
                "expected on this GPU): %s\n",
                texel, static_cast<unsigned long long>(after_keep - after_write),
                static_cast<unsigned long long>(expected), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

constexpr const char* ReadAllSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void read_all(uint2 id [[thread_position_in_grid]],
                     texture2d<float> image [[texture(0)]],
                     device float4* out [[buffer(0)]],
                     constant uint& width [[buffer(1)]]) {
    out[id.y * width + id.x] = image.read(id);
}
)";

/// E21: ETC2 and EAC for a device without them (Macs without an Apple GPU, or
/// EDEN_NX2_METAL_ETC=cpu): VideoCommon::DecodeEtc against this GPU's own decode of the same
/// random blocks, and the conversion switch.
int RunEtcDecode() {
    using VideoCore::Surface::PixelFormat;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const id<MTLDevice> mtl = device.GetDevice();
    if (device.AppleFamily() == 0) {
        std::printf("metal-smoke: ETC2 / EAC decode: no GPU decode to compare with: SKIP\n");
        return 0;
    }
    setenv("EDEN_NX2_METAL_ETC", "cpu", 1);
    const bool forced = Metal::IsConverted(device, PixelFormat::ETC2_RGB_UNORM) &&
                        Metal::IsConverted(device, PixelFormat::EAC_R11_UNORM);
    unsetenv("EDEN_NX2_METAL_ETC");
    const bool native = !Metal::IsConverted(device, PixelFormat::ETC2_RGB_UNORM);
    NSError* error = nil;
    const id<MTLLibrary> library = [mtl newLibraryWithSource:@(ReadAllSource)
                                                     options:[MTLCompileOptions new]
                                                       error:&error];
    const id<MTLComputePipelineState> kernel =
        library != nil ? [mtl newComputePipelineStateWithFunction:
                                  [library newFunctionWithName:@"read_all"]
                                                            error:&error]
                       : nil;
    if (kernel == nil) {
        std::printf("metal-smoke: FAIL ETC decode: kernel\n");
        return 1;
    }
    struct Case {
        PixelFormat format;
        uint32_t metal;
        uint32_t block_bytes;
        const char* name;
    };
    const std::array<Case, 7> cases{{
        {PixelFormat::ETC2_RGB_UNORM, Metal::Fmt::ETC2_RGB8, 8, "ETC2 RGB"},
        {PixelFormat::ETC2_RGB_PTA_UNORM, Metal::Fmt::ETC2_RGB8A1, 8, "ETC2 RGB+A1"},
        {PixelFormat::ETC2_RGBA_UNORM, Metal::Fmt::EAC_RGBA8, 16, "ETC2 RGBA"},
        {PixelFormat::EAC_R11_UNORM, Metal::Fmt::EAC_R11Unorm, 8, "EAC R11"},
        {PixelFormat::EAC_R11_SNORM, Metal::Fmt::EAC_R11Snorm, 8, "EAC R11 signed"},
        {PixelFormat::EAC_R11G11_UNORM, Metal::Fmt::EAC_RG11Unorm, 16, "EAC RG11"},
        {PixelFormat::EAC_R11G11_SNORM, Metal::Fmt::EAC_RG11Snorm, 16, "EAC RG11 signed"},
    }};
    constexpr uint32_t Side = 64;
    constexpr uint32_t Blocks = (Side / 4) * (Side / 4);
    int failed = 0;
    std::string report;
    for (const Case& test : cases) {
        std::vector<uint8_t> blocks(size_t{Blocks} * test.block_bytes);
        uint64_t state = 0x9E3779B97F4A7C15ULL ^ test.metal;
        for (uint8_t& byte : blocks) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            byte = static_cast<uint8_t>(state >> 56);
        }
        const Object texture = Metal::CreateTexture(
            device, {.kind = TextureKind::e2D, .format = test.metal, .width = Side, .height = Side});
        auto [upload, upload_bytes] = SharedBuffer(device, blocks.size());
        auto [out, out_bytes] = SharedBuffer(device, size_t{Side} * Side * 16);
        if (!texture || upload_bytes == nullptr || out_bytes == nullptr) {
            report += std::string{" "} + test.name + " NOT MADE";
            ++failed;
            continue;
        }
        std::memcpy(upload_bytes, blocks.data(), blocks.size());
        scheduler.CopyBufferToTexture(upload.Get(), 0, (Side / 4) * test.block_bytes, 0,
                                      texture.Get(), 0, 0, {}, {Side, Side, 1});
        scheduler.Finish();
        const id<MTLCommandQueue> queue = [mtl newCommandQueue];
        const id<MTLCommandBuffer> commands = [queue commandBuffer];
        const id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:kernel];
        [encoder setTexture:Metal::AsTexture(texture.Get()) atIndex:0];
        [encoder setBuffer:Metal::AsBuffer(out.Get()) offset:0 atIndex:0];
        const uint32_t width = Side;
        [encoder setBytes:&width length:sizeof(width) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(Side, Side, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        VideoCommon::BufferImageCopy copy{};
        copy.image_extent = {Side, Side, 1};
        copy.image_subresource.num_layers = 1;
        std::vector<uint8_t> decoded(size_t{Side} * Side * VideoCommon::EtcDecodedBytes(test.format));
        VideoCommon::DecodeEtc(blocks, decoded, copy, test.format);
        const bool wide = test.format == PixelFormat::EAC_R11_UNORM ||
                          test.format == PixelFormat::EAC_R11_SNORM ||
                          test.format == PixelFormat::EAC_R11G11_UNORM ||
                          test.format == PixelFormat::EAC_R11G11_SNORM;
        const bool is_signed = test.format == PixelFormat::EAC_R11_SNORM ||
                               test.format == PixelFormat::EAC_R11G11_SNORM;
        const bool rg = test.format == PixelFormat::EAC_R11G11_UNORM ||
                        test.format == PixelFormat::EAC_R11G11_SNORM;
        const float* gpu = reinterpret_cast<const float*>(out_bytes);
        std::array<uint32_t, 4> bad_blocks{};
        uint32_t bad = 0;
        uint32_t first_bad = ~0U;
        for (uint32_t texel = 0; texel < Side * Side; ++texel) {
            std::array<float, 4> want{0, 0, 0, 1};
            if (wide) {
                const uint16_t* v = reinterpret_cast<const uint16_t*>(decoded.data()) +
                                    texel * (rg ? 2 : 1);
                for (uint32_t c = 0; c < (rg ? 2U : 1U); ++c) {
                    want[c] = is_signed ? std::max(static_cast<int16_t>(v[c]) / 32767.0f, -1.0f)
                                        : v[c] / 65535.0f;
                }
            } else {
                for (uint32_t c = 0; c < 4; ++c) {
                    want[c] = decoded[texel * 4 + c] / 255.0f;
                }
            }
            bool texel_bad = false;
            for (uint32_t c = 0; c < (wide ? (rg ? 2U : 1U) : 4U); ++c) {
                texel_bad = texel_bad || std::fabs(gpu[texel * 4 + c] - want[c]) > (wide ? 0.002f : 0.006f);
            }
            if (texel_bad) {
                ++bad;
                first_bad = std::min(first_bad, texel);
            }
        }
        if (bad != 0) {
            ++failed;
            const uint32_t block = (first_bad / Side / 4) * (Side / 4) + (first_bad % Side) / 4;
            report += fmt::format(" {}: {} of {} texels differ (first in block {} texel {} of {}: gpu {:.4f} {:.4f} {:.4f} {:.4f}, cpu {:02x}{:02x}{:02x}{:02x}: {:02x})", test.name,
                                  bad, Side * Side, block, (first_bad % 4) * 4 + (first_bad / Side) % 4,
                                  first_bad, gpu[first_bad * 4], gpu[first_bad * 4 + 1], gpu[first_bad * 4 + 2], gpu[first_bad * 4 + 3],
                                  decoded[first_bad * VideoCommon::EtcDecodedBytes(test.format)],
                                  decoded[first_bad * VideoCommon::EtcDecodedBytes(test.format) + 1],
                                  decoded[first_bad * VideoCommon::EtcDecodedBytes(test.format) + 2],
                                  decoded[first_bad * VideoCommon::EtcDecodedBytes(test.format) + 3],
                                  fmt::join(std::span(blocks.data() + block * test.block_bytes,
                                                      test.block_bytes), ""));
        }
    }
    const bool passed = failed == 0 && forced && native;
    std::printf("metal-smoke: ETC2 / EAC CPU decode against this GPU's (7 formats, 64x64 random "
                "blocks), forced by EDEN_NX2_METAL_ETC=cpu %s:%s %s\n",
                forced ? "yes" : "NO", report.c_str(), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// E09: A5B5G5R1 (5-bit alpha on top, 1-bit red) is an RGBA8 render target converted to and from
/// the guest's words by kernels.
int RunPacked5551() {
    using VideoCore::Surface::PixelFormat;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const Metal::FormatInfo host = Metal::SurfaceFormat(device, PixelFormat::A5B5G5R1_UNORM, false);
    const Object target = Metal::CreateTexture(device, {.kind = TextureKind::e2D,
                                                        .format = host.format,
                                                        .width = 8,
                                                        .height = 8,
                                                        .render_target = host.attachable,
                                                        .shader_write = host.storage});
    auto [words, word_bytes] = SharedBuffer(device, 8 * 8 * 2);
    auto [back, back_bytes] = SharedBuffer(device, 8 * 8 * 4);
    if (!target || !host.attachable || host.swizzle != Metal::IdentitySwizzle ||
        word_bytes == nullptr || back_bytes == nullptr) {
        std::printf("metal-smoke: FAIL A5B5G5R1 target: texture\n");
        return 1;
    }
    // Rendered: cleared to magenta with alpha 1 is red 1, green 0, blue 31, alpha 31.
    Metal::ClearPass clear{};
    clear.color[0] = {.texture = target.Get()};
    clear.clear_color[0] = true;
    clear.color_value[0] = {1, 0, 1, 1};
    clear.width = 8;
    clear.height = 8;
    scheduler.Clear(clear);
    scheduler.BreakBlits();
    const bool down = scheduler.Pack5551(true, target.Get(), 0, 0, {}, {8, 8, 1}, words.Get(), 0, 16);
    scheduler.Finish();
    uint16_t rendered{};
    std::memcpy(&rendered, word_bytes, 2);
    // Uploaded: guest 0x0842 is alpha 1, blue 1, green 1, red 0.
    for (uint32_t index = 0; index < 64; ++index) {
        const uint16_t value = 0x0842;
        std::memcpy(word_bytes + index * 2, &value, 2);
    }
    const bool up = scheduler.Pack5551(false, target.Get(), 0, 0, {}, {8, 8, 1}, words.Get(), 0, 16);
    scheduler.BreakBlits();
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {8, 8, 1}, back.Get(), 0, 32, 0);
    scheduler.Finish();
    const std::array<uint8_t, 4> want{0, 8, 8, 8};
    const bool uploaded = std::memcmp(back_bytes, want.data(), 4) == 0;
    const bool passed = down && up && rendered == 0xFFC1 && uploaded;
    std::printf("metal-smoke: A5B5G5R1 as an RGBA8 target: rendered magenta reads back as %04x "
                "(ffc1), guest 0842 uploads as %02x%02x%02x%02x (00080808): %s\n",
                rendered, back_bytes[0], back_bytes[1], back_bytes[2], back_bytes[3],
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

int RunAll() {
    int result = 0;
    for (const auto test : {RunSampleCopies, RunScaleNearest, RunDepthBytes, RunConversions, RunGuards,
                            RunSamplesAndPlaceholders, RunExpandedFormats, RunSrgbStorage,
                            RunLayeredTarget, RunMultiRange, RunCrossThreadWait,
                            RunSamplers, RunPackedTargets, RunByteShadow, RunSparseTiles,
                            RunUnwrittenStore, RunEtcDecode,
                            RunPacked5551}) {
        const int outcome = test();
        result = result != 0 ? result : outcome;
    }
    return result;
}

} // namespace SmokeTextures
