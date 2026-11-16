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
#include <vector>

#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_smoke_textures.h"
#include "video_core/texture_cache/decode_bc.h"
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
    const bool bias = Metal::SamplerBiasSupported();
    if (bias) {
        const std::array<float, 4> PerTexel{0.5f, 0.5f, 0.0f, 0.0f};
        TSCEntry mip = tsc_of(WrapMode::ClampToEdge, TextureFilter::Nearest);
        mip.mipmap_filter.Assign(TextureMipmapFilter::Nearest);
        expect("lod bias 0", draw(mips, mip, PerTexel), Red);
        mip.mip_lod_bias.Assign(256); // 1.0
        expect("lod bias 1", draw(mips, mip, PerTexel), Green);
        const std::array<float, 4> Middle{0.0f, 0.0f, 0.5f, 0.5f};
        TSCEntry reduce = tsc_of(WrapMode::ClampToEdge, TextureFilter::Linear);
        expect("average", draw(black_white, reduce, Middle), 0xFF808080, 2);
        reduce.reduction_filter.Assign(SamplerReduction::Min);
        expect("min", draw(black_white, reduce, Middle), 0xFF000000);
        reduce.reduction_filter.Assign(SamplerReduction::Max);
        expect("max", draw(black_white, reduce, Middle), 0xFFFFFFFF);
    }
    const bool passed = wrong == 0;
    std::printf("metal-smoke: samplers: %s%s: %s\n", report.c_str(),
                bias ? "" : "; LOD bias and reduction SKIP (need the macOS / iOS 26 SDK and OS)",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

} // Anonymous namespace

int RunAll() {
    int result = 0;
    for (const auto test : {RunSampleCopies, RunDepthBytes, RunConversions, RunGuards,
                            RunSamplesAndPlaceholders, RunExpandedFormats, RunSamplers}) {
        const int outcome = test();
        result = result != 0 ? result : outcome;
    }
    return result;
}

} // namespace SmokeTextures
