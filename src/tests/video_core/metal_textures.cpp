// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/texture_cache/decode_bc.h"

namespace {

using VideoCore::Surface::PixelFormat;

VideoCommon::BufferImageCopy Copy(u32 width, u32 height, u32 row_length) {
    VideoCommon::BufferImageCopy copy{};
    copy.buffer_row_length = row_length;
    copy.buffer_image_height = height;
    copy.image_subresource.num_layers = 1;
    copy.image_extent = {width, height, 1};
    return copy;
}

} // Anonymous namespace

TEST_CASE("MetalTextures[ExpandRgb32Float]", "[video_core]") {
    REQUIRE(VideoCommon::IsExpandedFormat(PixelFormat::R32G32B32_FLOAT));
    REQUIRE(VideoCommon::ConvertedBytesPerBlock(PixelFormat::R32G32B32_FLOAT) == 16U);
    // Two texels a row, a row padded to three texels.
    const std::array<f32, 9 * 2> texels{1, 2, 3, 4, 5, 6, 0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0};
    std::vector<u8> input(sizeof(texels));
    std::memcpy(input.data(), texels.data(), sizeof(texels));
    std::vector<u8> output(2 * 2 * 16);
    VideoCommon::ExpandTexels(input, output, Copy(2, 2, 3), PixelFormat::R32G32B32_FLOAT);
    std::array<f32, 16> expanded{};
    std::memcpy(expanded.data(), output.data(), output.size());
    const std::array<f32, 16> expected{1, 2, 3, 1, 4, 5, 6, 1, 7, 8, 9, 1, 10, 11, 12, 1};
    REQUIRE(expanded == expected);
}

TEST_CASE("MetalTextures[ExpandG4R4]", "[video_core]") {
    REQUIRE(VideoCommon::ConvertedBytesPerBlock(PixelFormat::G4R4_UNORM) == 2U);
    const std::vector<u8> input{0xF0, 0x0F, 0x5A, 0x00};
    std::vector<u8> output(4 * 2);
    VideoCommon::ExpandTexels(input, output, Copy(4, 1, 4), PixelFormat::G4R4_UNORM);
    // Red is the high nibble, as Vulkan's R4G4_UNORM_PACK8.
    const std::vector<u8> expected{0xFF, 0x00, 0x00, 0xFF, 0x55, 0xAA, 0x00, 0x00};
    REQUIRE(output == expected);
}

TEST_CASE("MetalTextures[ExpandedHostFormats]", "[video_core]") {
    REQUIRE(Metal::SurfaceFormat(PixelFormat::R32G32B32_FLOAT, true).format ==
            Metal::Fmt::RGBA32Float);
    REQUIRE(Metal::SurfaceFormat(PixelFormat::G4R4_UNORM, true).format == Metal::Fmt::RG8Unorm);
    REQUIRE(Metal::SurfaceFormat(PixelFormat::R32G32B32_FLOAT, false).format ==
            Metal::Fmt::Invalid);
}

TEST_CASE("MetalTextures[SupportedSamples]", "[video_core]") {
    const Metal::PipelineLimits apple{.sample_counts = (1U << 1) | (1U << 2) | (1U << 4)};
    REQUIRE(Metal::SupportedSamples(8, apple) == 4U);
    REQUIRE(Metal::SupportedSamples(16, apple) == 4U);
    REQUIRE(Metal::SupportedSamples(4, apple) == 4U);
    REQUIRE(Metal::SupportedSamples(2, apple) == 2U);
    REQUIRE(Metal::SupportedSamples(1, apple) == 1U);
    REQUIRE(Metal::SupportedSamples(0, apple) == 1U);
    const Metal::PipelineLimits eight{.sample_counts = apple.sample_counts | (1U << 8)};
    REQUIRE(Metal::SupportedSamples(8, eight) == 8U);
}

TEST_CASE("MetalTextures[GuardLine]", "[video_core]") {
    (void)Metal::Coverage::TakeLine();
    Metal::Coverage::NoteGuard("test guard", "first");
    Metal::Coverage::NoteGuard("test guard");
    const std::string line = Metal::Coverage::TakeLine();
    REQUIRE(line.find("guards: test guard 2") != std::string::npos);
    REQUIRE(Metal::Coverage::TakeLine().empty());
}

namespace {

Metal::TextureFacts FakeFacts(Metal::TextureHandle texture) {
    switch (reinterpret_cast<uintptr_t>(texture)) {
    case 1: // Colour, sRGB of the pipeline's RGBA8 format, one level and slice
        return {.format = Metal::Fmt::RGBA8Unorm_sRGB};
    case 2: // Colour with 4 samples
        return {.format = Metal::Fmt::RGBA8Unorm, .samples = 4};
    case 3: // Depth-stencil
        return {.format = Metal::Fmt::Depth32Float_Stencil8};
    default:
        return {.format = 0};
    }
}

} // Anonymous namespace

TEST_CASE("MetalTextures[FitAttachments]", "[video_core]") {
    (void)Metal::Coverage::TakeLine();
    Metal::RenderPipelineDesc desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.samples = 1;
    desc.color_formats[0] = Metal::Fmt::RGBA8Unorm;
    desc.color_formats[1] = Metal::Fmt::RGBA8Unorm;
    desc.depth_format = Metal::Fmt::Depth32Float;
    Metal::RenderTargetSet targets{};
    targets.color[0] = {.texture = reinterpret_cast<Metal::TextureHandle>(1), .level = 3,
                        .slice = 2};
    targets.color[1] = {.texture = reinterpret_cast<Metal::TextureHandle>(2)};
    targets.depth = {.texture = reinterpret_cast<Metal::TextureHandle>(3)};
    targets.has_depth = true;
    Metal::FitAttachments(desc, targets, FakeFacts);
    // The level and slice fit the texture, the pipeline takes its format of the same kind.
    REQUIRE(targets.color[0].level == 0U);
    REQUIRE(targets.color[0].slice == 0U);
    REQUIRE(desc.color_formats[0] == Metal::Fmt::RGBA8Unorm_sRGB);
    // Another sample count, another depth format: left out of the pass and the pipeline.
    REQUIRE(targets.color[1].texture == nullptr);
    REQUIRE(desc.color_formats[1] == 0U);
    REQUIRE(!targets.has_depth);
    REQUIRE(desc.depth_format == 0U);
    const std::string line = Metal::Coverage::TakeLine();
    REQUIRE(line.find("attachment level past the texture 1") != std::string::npos);
    REQUIRE(line.find("attachment slice past the texture 1") != std::string::npos);
    REQUIRE(line.find("pipeline and pass formats differ 3") != std::string::npos);
}

TEST_CASE("MetalTextures[Placeholders]", "[video_core]") {
    const Metal::PipelineLimits limits{};
    for (u32 type = 0; type < Shader::NUM_TEXTURE_TYPES; ++type) {
        if (type == static_cast<u32>(Shader::TextureType::Buffer)) {
            continue;
        }
        const Metal::TextureDesc desc = Metal::PlaceholderTexture(type);
        INFO("texture type " << type);
        REQUIRE(Metal::CheckTexture(desc, limits).empty());
        REQUIRE(desc.kind != Metal::TextureKind::e1D);
        REQUIRE(desc.kind != Metal::TextureKind::e1DArray);
    }
}