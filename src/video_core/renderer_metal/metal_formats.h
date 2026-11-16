// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The guest format table for plain C++: Metal's pixel formats by number, so the table and the
// checks that read it also run where there is no Metal (the offline replay). metal_resources.mm
// asserts every number against Metal's headers.

#pragma once

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/surface.h"

namespace Metal {

#define METAL_PIXEL_FORMATS(X)                                                                     \
    X(Invalid, 0)                                                                                  \
    X(R8Unorm, 10)                                                                                 \
    X(R8Snorm, 12)                                                                                 \
    X(R8Uint, 13)                                                                                  \
    X(R8Sint, 14)                                                                                  \
    X(R16Unorm, 20)                                                                                \
    X(R16Snorm, 22)                                                                                \
    X(R16Uint, 23)                                                                                 \
    X(R16Sint, 24)                                                                                 \
    X(R16Float, 25)                                                                                \
    X(RG8Unorm, 30)                                                                                \
    X(RG8Snorm, 32)                                                                                \
    X(RG8Uint, 33)                                                                                 \
    X(RG8Sint, 34)                                                                                 \
    X(B5G6R5Unorm, 40)                                                                             \
    X(A1BGR5Unorm, 41)                                                                             \
    X(ABGR4Unorm, 42)                                                                              \
    X(BGR5A1Unorm, 43)                                                                             \
    X(R32Uint, 53)                                                                                 \
    X(R32Sint, 54)                                                                                 \
    X(R32Float, 55)                                                                                \
    X(RG16Unorm, 60)                                                                               \
    X(RG16Snorm, 62)                                                                               \
    X(RG16Uint, 63)                                                                                \
    X(RG16Sint, 64)                                                                                \
    X(RG16Float, 65)                                                                               \
    X(RGBA8Unorm, 70)                                                                              \
    X(RGBA8Unorm_sRGB, 71)                                                                         \
    X(RGBA8Snorm, 72)                                                                              \
    X(RGBA8Uint, 73)                                                                               \
    X(RGBA8Sint, 74)                                                                               \
    X(BGRA8Unorm, 80)                                                                              \
    X(BGRA8Unorm_sRGB, 81)                                                                         \
    X(RGB10A2Unorm, 90)                                                                            \
    X(RGB10A2Uint, 91)                                                                             \
    X(RG11B10Float, 92)                                                                            \
    X(RGB9E5Float, 93)                                                                             \
    X(BGR10A2Unorm, 94)                                                                            \
    X(RG32Uint, 103)                                                                               \
    X(RG32Sint, 104)                                                                               \
    X(RG32Float, 105)                                                                              \
    X(RGBA16Unorm, 110)                                                                            \
    X(RGBA16Snorm, 112)                                                                            \
    X(RGBA16Uint, 113)                                                                             \
    X(RGBA16Sint, 114)                                                                             \
    X(RGBA16Float, 115)                                                                            \
    X(RGBA32Uint, 123)                                                                             \
    X(RGBA32Sint, 124)                                                                             \
    X(RGBA32Float, 125)                                                                            \
    X(BC1_RGBA, 130)                                                                               \
    X(BC1_RGBA_sRGB, 131)                                                                          \
    X(BC2_RGBA, 132)                                                                               \
    X(BC2_RGBA_sRGB, 133)                                                                          \
    X(BC3_RGBA, 134)                                                                               \
    X(BC3_RGBA_sRGB, 135)                                                                          \
    X(BC4_RUnorm, 140)                                                                             \
    X(BC4_RSnorm, 141)                                                                             \
    X(BC5_RGUnorm, 142)                                                                            \
    X(BC5_RGSnorm, 143)                                                                            \
    X(BC6H_RGBFloat, 150)                                                                          \
    X(BC6H_RGBUfloat, 151)                                                                         \
    X(BC7_RGBAUnorm, 152)                                                                          \
    X(BC7_RGBAUnorm_sRGB, 153)                                                                     \
    X(EAC_R11Unorm, 170)                                                                           \
    X(EAC_R11Snorm, 172)                                                                           \
    X(EAC_RG11Unorm, 174)                                                                          \
    X(EAC_RG11Snorm, 176)                                                                          \
    X(EAC_RGBA8, 178)                                                                              \
    X(EAC_RGBA8_sRGB, 179)                                                                         \
    X(ETC2_RGB8, 180)                                                                              \
    X(ETC2_RGB8_sRGB, 181)                                                                         \
    X(ETC2_RGB8A1, 182)                                                                            \
    X(ETC2_RGB8A1_sRGB, 183)                                                                       \
    X(ASTC_4x4_sRGB, 186)                                                                          \
    X(ASTC_5x4_sRGB, 187)                                                                          \
    X(ASTC_5x5_sRGB, 188)                                                                          \
    X(ASTC_6x5_sRGB, 189)                                                                          \
    X(ASTC_6x6_sRGB, 190)                                                                          \
    X(ASTC_8x5_sRGB, 192)                                                                          \
    X(ASTC_8x6_sRGB, 193)                                                                          \
    X(ASTC_8x8_sRGB, 194)                                                                          \
    X(ASTC_10x5_sRGB, 195)                                                                         \
    X(ASTC_10x6_sRGB, 196)                                                                         \
    X(ASTC_10x8_sRGB, 197)                                                                         \
    X(ASTC_10x10_sRGB, 198)                                                                        \
    X(ASTC_12x10_sRGB, 199)                                                                        \
    X(ASTC_12x12_sRGB, 200)                                                                        \
    X(ASTC_4x4_LDR, 204)                                                                           \
    X(ASTC_5x4_LDR, 205)                                                                           \
    X(ASTC_5x5_LDR, 206)                                                                           \
    X(ASTC_6x5_LDR, 207)                                                                           \
    X(ASTC_6x6_LDR, 208)                                                                           \
    X(ASTC_8x5_LDR, 210)                                                                           \
    X(ASTC_8x6_LDR, 211)                                                                           \
    X(ASTC_8x8_LDR, 212)                                                                           \
    X(ASTC_10x5_LDR, 213)                                                                          \
    X(ASTC_10x6_LDR, 214)                                                                          \
    X(ASTC_10x8_LDR, 215)                                                                          \
    X(ASTC_10x10_LDR, 216)                                                                         \
    X(ASTC_12x10_LDR, 217)                                                                         \
    X(ASTC_12x12_LDR, 218)                                                                         \
    X(Depth16Unorm, 250)                                                                           \
    X(Depth32Float, 252)                                                                           \
    X(Stencil8, 253)                                                                               \
    X(Depth32Float_Stencil8, 260)

/// MTLPixelFormat values, named as Metal names them.
namespace Fmt {
enum : u32 {
#define X(name, value) name = value,
    METAL_PIXEL_FORMATS(X)
#undef X
};
} // namespace Fmt

/// MTLTextureSwizzle values.
namespace Swizzle {
enum : u8 { Zero = 0, One = 1, Red = 2, Green = 3, Blue = 4, Alpha = 5 };
} // namespace Swizzle

/// EDEN_NX2_METAL_FORMATS (on unless 0): guest formats Metal has no counterpart for are expanded
/// on the CPU at upload (R32G32B32_FLOAT as RGBA32Float, G4R4_UNORM as RG8Unorm,
/// VideoCommon::ExpandTexels); 0: the first gets no texture and the second is read as R8.
[[nodiscard]] bool FormatsExpanded();

/// The host format of a guest format on any device (SurfaceFormat in metal_resources.h).
[[nodiscard]] FormatInfo SurfaceFormat(VideoCore::Surface::PixelFormat format, bool converted);

/// Metal's name of a pixel format, or its number.
[[nodiscard]] std::string FormatName(u32 mtl_format);

} // namespace Metal
