// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions from nxnative (https://github.com/seedds/nxnative), Copyright (c) 2026 seedds, MIT
// License (LICENSES/MIT.txt): rb_swap, rb_swap_rows and the unwritten-attachment store rule.

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fmt/format.h>

#include "video_core/renderer_metal/metal_formats.h"

namespace Metal {

using VideoCore::Surface::PixelFormat;

namespace {

// zortos293's TryTransformSwizzleIfNeeded: A5B5G5R1 as R5G5B5A1 read A<->R, G<->B.
constexpr std::array<u8, 4> SwapSpecial{Swizzle::Alpha, Swizzle::Blue, Swizzle::Green,
                                        Swizzle::Red};

struct Entry {
    u32 format;
    bool attachable;
    bool storage;
    std::array<u8, 4> swizzle = IdentitySwizzle;
};

constexpr bool A = true;  // attachable
constexpr bool S = true;  // storage
constexpr bool N = false;

// Vulkan's SurfaceFormat table (maxwell_to_vk.cpp) with MoltenVK's Vulkan -> Metal format map.
Entry Lookup(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8B8G8R8_UNORM: return {Fmt::RGBA8Unorm, A, S};
    case PixelFormat::A8B8G8R8_SNORM: return {Fmt::RGBA8Snorm, A, S};
    case PixelFormat::A8B8G8R8_SINT: return {Fmt::RGBA8Sint, A, S};
    case PixelFormat::A8B8G8R8_UINT: return {Fmt::RGBA8Uint, A, S};
    case PixelFormat::R5G6B5_UNORM: return {Fmt::B5G6R5Unorm, A, N};
    // The 16-bit packed formats Metal has only with red and blue exchanged (or, for RGBA4, the
    // nibbles reversed) hold Metal's order in the host texture: bytes are converted where they
    // cross to guest memory (PackedSwapShift), so the texture is a render target with no swizzle.
    case PixelFormat::B5G6R5_UNORM: return {Fmt::B5G6R5Unorm, A, N};
    case PixelFormat::A1R5G5B5_UNORM: return {Fmt::BGR5A1Unorm, A, N};
    case PixelFormat::A2B10G10R10_UNORM: return {Fmt::RGB10A2Unorm, A, S};
    case PixelFormat::A2B10G10R10_UINT: return {Fmt::RGB10A2Uint, A, S};
    case PixelFormat::A2R10G10B10_UNORM: return {Fmt::BGR10A2Unorm, A, N};
    case PixelFormat::A1B5G5R5_UNORM: return {Fmt::BGR5A1Unorm, A, N};
    case PixelFormat::A5B5G5R1_UNORM: return {Fmt::A1BGR5Unorm, N, N, SwapSpecial};
    case PixelFormat::R8_UNORM: return {Fmt::R8Unorm, A, S};
    case PixelFormat::R8_SNORM: return {Fmt::R8Snorm, A, S};
    case PixelFormat::R8_SINT: return {Fmt::R8Sint, A, S};
    case PixelFormat::R8_UINT: return {Fmt::R8Uint, A, S};
    case PixelFormat::R16G16B16A16_FLOAT: return {Fmt::RGBA16Float, A, S};
    case PixelFormat::R16G16B16A16_UNORM: return {Fmt::RGBA16Unorm, A, S};
    case PixelFormat::R16G16B16A16_SNORM: return {Fmt::RGBA16Snorm, A, S};
    case PixelFormat::R16G16B16A16_SINT: return {Fmt::RGBA16Sint, A, S};
    case PixelFormat::R16G16B16A16_UINT: return {Fmt::RGBA16Uint, A, S};
    case PixelFormat::B10G11R11_FLOAT: return {Fmt::RG11B10Float, A, S};
    case PixelFormat::R32G32B32A32_UINT: return {Fmt::RGBA32Uint, A, S};
    case PixelFormat::BC1_RGBA_UNORM: return {Fmt::BC1_RGBA, N, N};
    case PixelFormat::BC2_UNORM: return {Fmt::BC2_RGBA, N, N};
    case PixelFormat::BC3_UNORM: return {Fmt::BC3_RGBA, N, N};
    case PixelFormat::BC4_UNORM: return {Fmt::BC4_RUnorm, N, N};
    case PixelFormat::BC4_SNORM: return {Fmt::BC4_RSnorm, N, N};
    case PixelFormat::BC5_UNORM: return {Fmt::BC5_RGUnorm, N, N};
    case PixelFormat::BC5_SNORM: return {Fmt::BC5_RGSnorm, N, N};
    case PixelFormat::BC7_UNORM: return {Fmt::BC7_RGBAUnorm, N, N};
    case PixelFormat::BC6H_UFLOAT: return {Fmt::BC6H_RGBUfloat, N, N};
    case PixelFormat::BC6H_SFLOAT: return {Fmt::BC6H_RGBFloat, N, N};
    case PixelFormat::ASTC_2D_4X4_UNORM: return {Fmt::ASTC_4x4_LDR, N, N};
    case PixelFormat::B8G8R8A8_UNORM: return {Fmt::BGRA8Unorm, A, S};
    case PixelFormat::R32G32B32A32_FLOAT: return {Fmt::RGBA32Float, A, S};
    case PixelFormat::R32G32B32A32_SINT: return {Fmt::RGBA32Sint, A, S};
    case PixelFormat::R32G32_FLOAT: return {Fmt::RG32Float, A, S};
    case PixelFormat::R32G32_SINT: return {Fmt::RG32Sint, A, S};
    case PixelFormat::R32_FLOAT: return {Fmt::R32Float, A, S};
    case PixelFormat::R16_FLOAT: return {Fmt::R16Float, A, S};
    case PixelFormat::R16_UNORM: return {Fmt::R16Unorm, A, S};
    case PixelFormat::R16_SNORM: return {Fmt::R16Snorm, A, S};
    case PixelFormat::R16_UINT: return {Fmt::R16Uint, A, S};
    case PixelFormat::R16_SINT: return {Fmt::R16Sint, A, S};
    case PixelFormat::R16G16_UNORM: return {Fmt::RG16Unorm, A, S};
    case PixelFormat::R16G16_FLOAT: return {Fmt::RG16Float, A, S};
    case PixelFormat::R16G16_UINT: return {Fmt::RG16Uint, A, S};
    case PixelFormat::R16G16_SINT: return {Fmt::RG16Sint, A, S};
    case PixelFormat::R16G16_SNORM: return {Fmt::RG16Snorm, A, S};
    // Metal has no 96-bit texture format; texel buffers of it are read as raw words (M3).
    case PixelFormat::R32G32B32_FLOAT: return {Fmt::Invalid, N, N};
    case PixelFormat::A8B8G8R8_SRGB: return {Fmt::RGBA8Unorm_sRGB, A, N};
    case PixelFormat::R8G8_UNORM: return {Fmt::RG8Unorm, A, S};
    case PixelFormat::R8G8_SNORM: return {Fmt::RG8Snorm, A, S};
    case PixelFormat::R8G8_SINT: return {Fmt::RG8Sint, A, S};
    case PixelFormat::R8G8_UINT: return {Fmt::RG8Uint, A, S};
    case PixelFormat::R32G32_UINT: return {Fmt::RG32Uint, A, S};
    case PixelFormat::R16G16B16X16_FLOAT: return {Fmt::RGBA16Float, A, S};
    case PixelFormat::R32_UINT: return {Fmt::R32Uint, A, S};
    case PixelFormat::R32_SINT: return {Fmt::R32Sint, A, S};
    case PixelFormat::ASTC_2D_8X8_UNORM: return {Fmt::ASTC_8x8_LDR, N, N};
    case PixelFormat::ASTC_2D_8X5_UNORM: return {Fmt::ASTC_8x5_LDR, N, N};
    case PixelFormat::ASTC_2D_5X4_UNORM: return {Fmt::ASTC_5x4_LDR, N, N};
    case PixelFormat::B8G8R8A8_SRGB: return {Fmt::BGRA8Unorm_sRGB, A, N};
    case PixelFormat::BC1_RGBA_SRGB: return {Fmt::BC1_RGBA_sRGB, N, N};
    case PixelFormat::BC2_SRGB: return {Fmt::BC2_RGBA_sRGB, N, N};
    case PixelFormat::BC3_SRGB: return {Fmt::BC3_RGBA_sRGB, N, N};
    case PixelFormat::BC7_SRGB: return {Fmt::BC7_RGBAUnorm_sRGB, N, N};
    case PixelFormat::A4B4G4R4_UNORM: return {Fmt::ABGR4Unorm, A, N};
    // Metal has no 8-bit 4:4 format: approximated (only the high nibble survives).
    case PixelFormat::G4R4_UNORM: return {Fmt::R8Unorm, N, N};
    case PixelFormat::ASTC_2D_4X4_SRGB: return {Fmt::ASTC_4x4_sRGB, N, N};
    case PixelFormat::ASTC_2D_8X8_SRGB: return {Fmt::ASTC_8x8_sRGB, N, N};
    case PixelFormat::ASTC_2D_8X5_SRGB: return {Fmt::ASTC_8x5_sRGB, N, N};
    case PixelFormat::ASTC_2D_5X4_SRGB: return {Fmt::ASTC_5x4_sRGB, N, N};
    case PixelFormat::ASTC_2D_5X5_UNORM: return {Fmt::ASTC_5x5_LDR, N, N};
    case PixelFormat::ASTC_2D_5X5_SRGB: return {Fmt::ASTC_5x5_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X8_UNORM: return {Fmt::ASTC_10x8_LDR, N, N};
    case PixelFormat::ASTC_2D_10X8_SRGB: return {Fmt::ASTC_10x8_sRGB, N, N};
    case PixelFormat::ASTC_2D_6X6_UNORM: return {Fmt::ASTC_6x6_LDR, N, N};
    case PixelFormat::ASTC_2D_6X6_SRGB: return {Fmt::ASTC_6x6_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X6_UNORM: return {Fmt::ASTC_10x6_LDR, N, N};
    case PixelFormat::ASTC_2D_10X6_SRGB: return {Fmt::ASTC_10x6_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X5_UNORM: return {Fmt::ASTC_10x5_LDR, N, N};
    case PixelFormat::ASTC_2D_10X5_SRGB: return {Fmt::ASTC_10x5_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X10_UNORM: return {Fmt::ASTC_10x10_LDR, N, N};
    case PixelFormat::ASTC_2D_10X10_SRGB: return {Fmt::ASTC_10x10_sRGB, N, N};
    case PixelFormat::ASTC_2D_12X10_UNORM: return {Fmt::ASTC_12x10_LDR, N, N};
    case PixelFormat::ASTC_2D_12X10_SRGB: return {Fmt::ASTC_12x10_sRGB, N, N};
    case PixelFormat::ASTC_2D_12X12_UNORM: return {Fmt::ASTC_12x12_LDR, N, N};
    case PixelFormat::ASTC_2D_12X12_SRGB: return {Fmt::ASTC_12x12_sRGB, N, N};
    case PixelFormat::ASTC_2D_8X6_UNORM: return {Fmt::ASTC_8x6_LDR, N, N};
    case PixelFormat::ASTC_2D_8X6_SRGB: return {Fmt::ASTC_8x6_sRGB, N, N};
    case PixelFormat::ASTC_2D_6X5_UNORM: return {Fmt::ASTC_6x5_LDR, N, N};
    case PixelFormat::ASTC_2D_6X5_SRGB: return {Fmt::ASTC_6x5_sRGB, N, N};
    case PixelFormat::E5B9G9R9_FLOAT: return {Fmt::RGB9E5Float, N, N};
    case PixelFormat::ETC2_RGB_UNORM: return {Fmt::ETC2_RGB8, N, N};
    case PixelFormat::ETC2_RGBA_UNORM: return {Fmt::EAC_RGBA8, N, N};
    case PixelFormat::ETC2_RGB_PTA_UNORM: return {Fmt::ETC2_RGB8A1, N, N};
    case PixelFormat::ETC2_RGB_SRGB: return {Fmt::ETC2_RGB8_sRGB, N, N};
    case PixelFormat::ETC2_RGBA_SRGB: return {Fmt::EAC_RGBA8_sRGB, N, N};
    case PixelFormat::ETC2_RGB_PTA_SRGB: return {Fmt::ETC2_RGB8A1_sRGB, N, N};
    case PixelFormat::EAC_R11_UNORM: return {Fmt::EAC_R11Unorm, N, N};
    case PixelFormat::EAC_R11_SNORM: return {Fmt::EAC_R11Snorm, N, N};
    case PixelFormat::EAC_R11G11_UNORM: return {Fmt::EAC_RG11Unorm, N, N};
    case PixelFormat::EAC_R11G11_SNORM: return {Fmt::EAC_RG11Snorm, N, N};
    case PixelFormat::D32_FLOAT: return {Fmt::Depth32Float, A, N};
    case PixelFormat::D16_UNORM: return {Fmt::Depth16Unorm, A, N};
    // Apple GPUs have no 24-bit depth: 32-bit float depth, as MoltenVK does.
    case PixelFormat::X8_D24_UNORM: return {Fmt::Depth32Float, A, N};
    case PixelFormat::S8_UINT: return {Fmt::Stencil8, A, N};
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
    case PixelFormat::D32_FLOAT_S8_UINT: return {Fmt::Depth32Float_Stencil8, A, N};
    default: return {Fmt::Invalid, N, N};
    }
}

} // Anonymous namespace

u32 PackedSwapShift(PixelFormat format) {
    switch (format) {
    case PixelFormat::B5G6R5_UNORM: return 11;
    case PixelFormat::A1B5G5R5_UNORM: return 10;
    case PixelFormat::A4B4G4R4_UNORM: return 4;
    default: return 0;
    }
}

void SwapPackedTexels(u8* data, u32 width, u32 rows, u32 images, size_t row_pitch,
                      size_t image_pitch, u32 shift) {
    // nxnative dm/gfx/nvn/nvn_mtl.m rb_swap_rows (MIT).
    const u16 mask = static_cast<u16>(31U | 31U << shift);
    for (u32 z = 0; z < images; ++z) {
        for (u32 y = 0; y < rows; ++y) {
            u8* const row = data + z * image_pitch + y * row_pitch;
            for (u32 x = 0; x < width; ++x) {
                u16 v;
                std::memcpy(&v, row + x * 2, 2);
                if (shift == 4) {
                    v = static_cast<u16>(v << 12 | (v & 0xf0U) << 4 | (v >> 4 & 0xf0U) | v >> 12);
                } else {
                    v = static_cast<u16>((v & ~mask) | (v & 31U) << shift | (v >> shift & 31U));
                }
                std::memcpy(row + x * 2, &v, 2);
            }
        }
    }
}

u32 LinearTwin(u32 mtl_format) {
    switch (mtl_format) {
    case Fmt::RGBA8Unorm_sRGB: return Fmt::RGBA8Unorm;
    case Fmt::BGRA8Unorm_sRGB: return Fmt::BGRA8Unorm;
    default: return 0;
    }
}

u32 StorageFormat(Shader::ImageFormat image_format) {
    switch (image_format) {
    case Shader::ImageFormat::Typeless:
        return 0;
    case Shader::ImageFormat::R8_UINT:
        return Fmt::R8Uint;
    case Shader::ImageFormat::R8_SINT:
        return Fmt::R8Sint;
    case Shader::ImageFormat::R16_UINT:
        return Fmt::R16Uint;
    case Shader::ImageFormat::R16_SINT:
        return Fmt::R16Sint;
    case Shader::ImageFormat::R32_UINT:
        return Fmt::R32Uint;
    case Shader::ImageFormat::R32G32_UINT:
        return Fmt::RG32Uint;
    case Shader::ImageFormat::R32G32B32A32_UINT:
        return Fmt::RGBA32Uint;
    }
    return 0;
}

bool FormatsExpanded() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FORMATS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

FormatInfo SurfaceFormat(PixelFormat format, bool converted) {
    if (converted) {
        const bool srgb = VideoCore::Surface::IsPixelFormatSRGB(format);
        u32 decoded = srgb ? Fmt::RGBA8Unorm_sRGB : Fmt::RGBA8Unorm;
        switch (format) {
        case PixelFormat::BC4_UNORM: decoded = Fmt::R8Unorm; break;
        case PixelFormat::BC4_SNORM: decoded = Fmt::R8Snorm; break;
        case PixelFormat::BC5_UNORM: decoded = Fmt::RG8Unorm; break;
        case PixelFormat::BC5_SNORM: decoded = Fmt::RG8Snorm; break;
        case PixelFormat::BC6H_UFLOAT:
        case PixelFormat::BC6H_SFLOAT: decoded = Fmt::RGBA16Float; break;
        // VideoCommon::ExpandTexels.
        case PixelFormat::R32G32B32_FLOAT: decoded = Fmt::RGBA32Float; break;
        case PixelFormat::G4R4_UNORM: decoded = Fmt::RG8Unorm; break;
        default: break;
        }
        return FormatInfo{
            .format = decoded,
            .attachable = true,
            .storage = !srgb,
            .format_class = FormatClass::Float,
        };
    }
    const Entry entry = Lookup(format);
    return FormatInfo{
        .format = entry.format,
        .attachable = entry.attachable,
        .storage = entry.storage,
        .format_class = ClassOf(entry.format),
        .swizzle = entry.swizzle,
    };
}

FormatClass ClassOf(u32 mtl_format) {
    switch (mtl_format) {
    case Fmt::R8Uint:
    case Fmt::R16Uint:
    case Fmt::R32Uint:
    case Fmt::RG8Uint:
    case Fmt::RG16Uint:
    case Fmt::RG32Uint:
    case Fmt::RGBA8Uint:
    case Fmt::RGB10A2Uint:
    case Fmt::RGBA16Uint:
    case Fmt::RGBA32Uint:
        return FormatClass::Uint;
    case Fmt::R8Sint:
    case Fmt::R16Sint:
    case Fmt::R32Sint:
    case Fmt::RG8Sint:
    case Fmt::RG16Sint:
    case Fmt::RG32Sint:
    case Fmt::RGBA8Sint:
    case Fmt::RGBA16Sint:
    case Fmt::RGBA32Sint:
        return FormatClass::Sint;
    case Fmt::Depth16Unorm:
    case Fmt::Depth32Float:
        return FormatClass::Depth;
    case Fmt::Stencil8:
        return FormatClass::Stencil;
    case Fmt::Depth32Float_Stencil8:
        return FormatClass::DepthStencil;
    default:
        return FormatClass::Float;
    }
}

std::string FormatName(u32 mtl_format) {
    switch (mtl_format) {
#define X(name, value)                                                                             \
    case value:                                                                                    \
        return #name;
        METAL_PIXEL_FORMATS(X)
#undef X
    default:
        return fmt::format("{}", mtl_format);
    }
}

} // namespace Metal
