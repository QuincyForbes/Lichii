// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"

namespace Metal {

using VideoCore::Surface::PixelFormat;

namespace {

constexpr u8 Swz(MTLTextureSwizzle swizzle) {
    return static_cast<u8>(swizzle);
}

constexpr std::array<u8, 4> SwapRedBlue{Swz(MTLTextureSwizzleBlue), Swz(MTLTextureSwizzleGreen),
                                        Swz(MTLTextureSwizzleRed), Swz(MTLTextureSwizzleAlpha)};
// zortos293's TryTransformSwizzleIfNeeded: A5B5G5R1 as R5G5B5A1 read A<->R, G<->B.
constexpr std::array<u8, 4> SwapSpecial{Swz(MTLTextureSwizzleAlpha), Swz(MTLTextureSwizzleBlue),
                                        Swz(MTLTextureSwizzleGreen), Swz(MTLTextureSwizzleRed)};
constexpr std::array<u8, 4> Reversed{Swz(MTLTextureSwizzleAlpha), Swz(MTLTextureSwizzleBlue),
                                     Swz(MTLTextureSwizzleGreen), Swz(MTLTextureSwizzleRed)};

struct Entry {
    MTLPixelFormat format;
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
    case PixelFormat::A8B8G8R8_UNORM: return {MTLPixelFormatRGBA8Unorm, A, S};
    case PixelFormat::A8B8G8R8_SNORM: return {MTLPixelFormatRGBA8Snorm, A, S};
    case PixelFormat::A8B8G8R8_SINT: return {MTLPixelFormatRGBA8Sint, A, S};
    case PixelFormat::A8B8G8R8_UINT: return {MTLPixelFormatRGBA8Uint, A, S};
    case PixelFormat::R5G6B5_UNORM: return {MTLPixelFormatB5G6R5Unorm, A, N};
    case PixelFormat::B5G6R5_UNORM: return {MTLPixelFormatB5G6R5Unorm, N, N, SwapRedBlue};
    case PixelFormat::A1R5G5B5_UNORM: return {MTLPixelFormatBGR5A1Unorm, A, N};
    case PixelFormat::A2B10G10R10_UNORM: return {MTLPixelFormatRGB10A2Unorm, A, S};
    case PixelFormat::A2B10G10R10_UINT: return {MTLPixelFormatRGB10A2Uint, A, S};
    case PixelFormat::A2R10G10B10_UNORM: return {MTLPixelFormatBGR10A2Unorm, A, N};
    case PixelFormat::A1B5G5R5_UNORM: return {MTLPixelFormatBGR5A1Unorm, A, N, SwapRedBlue};
    case PixelFormat::A5B5G5R1_UNORM: return {MTLPixelFormatA1BGR5Unorm, N, N, SwapSpecial};
    case PixelFormat::R8_UNORM: return {MTLPixelFormatR8Unorm, A, S};
    case PixelFormat::R8_SNORM: return {MTLPixelFormatR8Snorm, A, S};
    case PixelFormat::R8_SINT: return {MTLPixelFormatR8Sint, A, S};
    case PixelFormat::R8_UINT: return {MTLPixelFormatR8Uint, A, S};
    case PixelFormat::R16G16B16A16_FLOAT: return {MTLPixelFormatRGBA16Float, A, S};
    case PixelFormat::R16G16B16A16_UNORM: return {MTLPixelFormatRGBA16Unorm, A, S};
    case PixelFormat::R16G16B16A16_SNORM: return {MTLPixelFormatRGBA16Snorm, A, S};
    case PixelFormat::R16G16B16A16_SINT: return {MTLPixelFormatRGBA16Sint, A, S};
    case PixelFormat::R16G16B16A16_UINT: return {MTLPixelFormatRGBA16Uint, A, S};
    case PixelFormat::B10G11R11_FLOAT: return {MTLPixelFormatRG11B10Float, A, S};
    case PixelFormat::R32G32B32A32_UINT: return {MTLPixelFormatRGBA32Uint, A, S};
    case PixelFormat::BC1_RGBA_UNORM: return {MTLPixelFormatBC1_RGBA, N, N};
    case PixelFormat::BC2_UNORM: return {MTLPixelFormatBC2_RGBA, N, N};
    case PixelFormat::BC3_UNORM: return {MTLPixelFormatBC3_RGBA, N, N};
    case PixelFormat::BC4_UNORM: return {MTLPixelFormatBC4_RUnorm, N, N};
    case PixelFormat::BC4_SNORM: return {MTLPixelFormatBC4_RSnorm, N, N};
    case PixelFormat::BC5_UNORM: return {MTLPixelFormatBC5_RGUnorm, N, N};
    case PixelFormat::BC5_SNORM: return {MTLPixelFormatBC5_RGSnorm, N, N};
    case PixelFormat::BC7_UNORM: return {MTLPixelFormatBC7_RGBAUnorm, N, N};
    case PixelFormat::BC6H_UFLOAT: return {MTLPixelFormatBC6H_RGBUfloat, N, N};
    case PixelFormat::BC6H_SFLOAT: return {MTLPixelFormatBC6H_RGBFloat, N, N};
    case PixelFormat::ASTC_2D_4X4_UNORM: return {MTLPixelFormatASTC_4x4_LDR, N, N};
    case PixelFormat::B8G8R8A8_UNORM: return {MTLPixelFormatBGRA8Unorm, A, S};
    case PixelFormat::R32G32B32A32_FLOAT: return {MTLPixelFormatRGBA32Float, A, S};
    case PixelFormat::R32G32B32A32_SINT: return {MTLPixelFormatRGBA32Sint, A, S};
    case PixelFormat::R32G32_FLOAT: return {MTLPixelFormatRG32Float, A, S};
    case PixelFormat::R32G32_SINT: return {MTLPixelFormatRG32Sint, A, S};
    case PixelFormat::R32_FLOAT: return {MTLPixelFormatR32Float, A, S};
    case PixelFormat::R16_FLOAT: return {MTLPixelFormatR16Float, A, S};
    case PixelFormat::R16_UNORM: return {MTLPixelFormatR16Unorm, A, S};
    case PixelFormat::R16_SNORM: return {MTLPixelFormatR16Snorm, A, S};
    case PixelFormat::R16_UINT: return {MTLPixelFormatR16Uint, A, S};
    case PixelFormat::R16_SINT: return {MTLPixelFormatR16Sint, A, S};
    case PixelFormat::R16G16_UNORM: return {MTLPixelFormatRG16Unorm, A, S};
    case PixelFormat::R16G16_FLOAT: return {MTLPixelFormatRG16Float, A, S};
    case PixelFormat::R16G16_UINT: return {MTLPixelFormatRG16Uint, A, S};
    case PixelFormat::R16G16_SINT: return {MTLPixelFormatRG16Sint, A, S};
    case PixelFormat::R16G16_SNORM: return {MTLPixelFormatRG16Snorm, A, S};
    // Metal has no 96-bit texture format; texel buffers of it are read as raw words (M3).
    case PixelFormat::R32G32B32_FLOAT: return {MTLPixelFormatInvalid, N, N};
    case PixelFormat::A8B8G8R8_SRGB: return {MTLPixelFormatRGBA8Unorm_sRGB, A, N};
    case PixelFormat::R8G8_UNORM: return {MTLPixelFormatRG8Unorm, A, S};
    case PixelFormat::R8G8_SNORM: return {MTLPixelFormatRG8Snorm, A, S};
    case PixelFormat::R8G8_SINT: return {MTLPixelFormatRG8Sint, A, S};
    case PixelFormat::R8G8_UINT: return {MTLPixelFormatRG8Uint, A, S};
    case PixelFormat::R32G32_UINT: return {MTLPixelFormatRG32Uint, A, S};
    case PixelFormat::R16G16B16X16_FLOAT: return {MTLPixelFormatRGBA16Float, A, S};
    case PixelFormat::R32_UINT: return {MTLPixelFormatR32Uint, A, S};
    case PixelFormat::R32_SINT: return {MTLPixelFormatR32Sint, A, S};
    case PixelFormat::ASTC_2D_8X8_UNORM: return {MTLPixelFormatASTC_8x8_LDR, N, N};
    case PixelFormat::ASTC_2D_8X5_UNORM: return {MTLPixelFormatASTC_8x5_LDR, N, N};
    case PixelFormat::ASTC_2D_5X4_UNORM: return {MTLPixelFormatASTC_5x4_LDR, N, N};
    case PixelFormat::B8G8R8A8_SRGB: return {MTLPixelFormatBGRA8Unorm_sRGB, A, N};
    case PixelFormat::BC1_RGBA_SRGB: return {MTLPixelFormatBC1_RGBA_sRGB, N, N};
    case PixelFormat::BC2_SRGB: return {MTLPixelFormatBC2_RGBA_sRGB, N, N};
    case PixelFormat::BC3_SRGB: return {MTLPixelFormatBC3_RGBA_sRGB, N, N};
    case PixelFormat::BC7_SRGB: return {MTLPixelFormatBC7_RGBAUnorm_sRGB, N, N};
    // Vulkan's R4G4B4A4 with the components reversed, as MoltenVK's ABGR4Unorm.
    case PixelFormat::A4B4G4R4_UNORM: return {MTLPixelFormatABGR4Unorm, N, N, Reversed};
    // Metal has no 8-bit 4:4 format: approximated (only the high nibble survives).
    case PixelFormat::G4R4_UNORM: return {MTLPixelFormatR8Unorm, N, N};
    case PixelFormat::ASTC_2D_4X4_SRGB: return {MTLPixelFormatASTC_4x4_sRGB, N, N};
    case PixelFormat::ASTC_2D_8X8_SRGB: return {MTLPixelFormatASTC_8x8_sRGB, N, N};
    case PixelFormat::ASTC_2D_8X5_SRGB: return {MTLPixelFormatASTC_8x5_sRGB, N, N};
    case PixelFormat::ASTC_2D_5X4_SRGB: return {MTLPixelFormatASTC_5x4_sRGB, N, N};
    case PixelFormat::ASTC_2D_5X5_UNORM: return {MTLPixelFormatASTC_5x5_LDR, N, N};
    case PixelFormat::ASTC_2D_5X5_SRGB: return {MTLPixelFormatASTC_5x5_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X8_UNORM: return {MTLPixelFormatASTC_10x8_LDR, N, N};
    case PixelFormat::ASTC_2D_10X8_SRGB: return {MTLPixelFormatASTC_10x8_sRGB, N, N};
    case PixelFormat::ASTC_2D_6X6_UNORM: return {MTLPixelFormatASTC_6x6_LDR, N, N};
    case PixelFormat::ASTC_2D_6X6_SRGB: return {MTLPixelFormatASTC_6x6_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X6_UNORM: return {MTLPixelFormatASTC_10x6_LDR, N, N};
    case PixelFormat::ASTC_2D_10X6_SRGB: return {MTLPixelFormatASTC_10x6_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X5_UNORM: return {MTLPixelFormatASTC_10x5_LDR, N, N};
    case PixelFormat::ASTC_2D_10X5_SRGB: return {MTLPixelFormatASTC_10x5_sRGB, N, N};
    case PixelFormat::ASTC_2D_10X10_UNORM: return {MTLPixelFormatASTC_10x10_LDR, N, N};
    case PixelFormat::ASTC_2D_10X10_SRGB: return {MTLPixelFormatASTC_10x10_sRGB, N, N};
    case PixelFormat::ASTC_2D_12X10_UNORM: return {MTLPixelFormatASTC_12x10_LDR, N, N};
    case PixelFormat::ASTC_2D_12X10_SRGB: return {MTLPixelFormatASTC_12x10_sRGB, N, N};
    case PixelFormat::ASTC_2D_12X12_UNORM: return {MTLPixelFormatASTC_12x12_LDR, N, N};
    case PixelFormat::ASTC_2D_12X12_SRGB: return {MTLPixelFormatASTC_12x12_sRGB, N, N};
    case PixelFormat::ASTC_2D_8X6_UNORM: return {MTLPixelFormatASTC_8x6_LDR, N, N};
    case PixelFormat::ASTC_2D_8X6_SRGB: return {MTLPixelFormatASTC_8x6_sRGB, N, N};
    case PixelFormat::ASTC_2D_6X5_UNORM: return {MTLPixelFormatASTC_6x5_LDR, N, N};
    case PixelFormat::ASTC_2D_6X5_SRGB: return {MTLPixelFormatASTC_6x5_sRGB, N, N};
    case PixelFormat::E5B9G9R9_FLOAT: return {MTLPixelFormatRGB9E5Float, N, N};
    case PixelFormat::ETC2_RGB_UNORM: return {MTLPixelFormatETC2_RGB8, N, N};
    case PixelFormat::ETC2_RGBA_UNORM: return {MTLPixelFormatEAC_RGBA8, N, N};
    case PixelFormat::ETC2_RGB_PTA_UNORM: return {MTLPixelFormatETC2_RGB8A1, N, N};
    case PixelFormat::ETC2_RGB_SRGB: return {MTLPixelFormatETC2_RGB8_sRGB, N, N};
    case PixelFormat::ETC2_RGBA_SRGB: return {MTLPixelFormatEAC_RGBA8_sRGB, N, N};
    case PixelFormat::ETC2_RGB_PTA_SRGB: return {MTLPixelFormatETC2_RGB8A1_sRGB, N, N};
    case PixelFormat::EAC_R11_UNORM: return {MTLPixelFormatEAC_R11Unorm, N, N};
    case PixelFormat::EAC_R11_SNORM: return {MTLPixelFormatEAC_R11Snorm, N, N};
    case PixelFormat::EAC_R11G11_UNORM: return {MTLPixelFormatEAC_RG11Unorm, N, N};
    case PixelFormat::EAC_R11G11_SNORM: return {MTLPixelFormatEAC_RG11Snorm, N, N};
    case PixelFormat::D32_FLOAT: return {MTLPixelFormatDepth32Float, A, N};
    case PixelFormat::D16_UNORM: return {MTLPixelFormatDepth16Unorm, A, N};
    // Apple GPUs have no 24-bit depth: 32-bit float depth, as MoltenVK does.
    case PixelFormat::X8_D24_UNORM: return {MTLPixelFormatDepth32Float, A, N};
    case PixelFormat::S8_UINT: return {MTLPixelFormatStencil8, A, N};
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
    case PixelFormat::D32_FLOAT_S8_UINT: return {MTLPixelFormatDepth32Float_Stencil8, A, N};
    default: return {MTLPixelFormatInvalid, N, N};
    }
}

MTLTextureType TextureType(TextureKind kind) {
    switch (kind) {
    case TextureKind::e1D: return MTLTextureType1D;
    case TextureKind::e1DArray: return MTLTextureType1DArray;
    case TextureKind::e2D: return MTLTextureType2D;
    case TextureKind::e2DArray: return MTLTextureType2DArray;
    case TextureKind::e2DMultisample: return MTLTextureType2DMultisample;
    case TextureKind::e2DMultisampleArray: return MTLTextureType2DMultisampleArray;
    case TextureKind::e3D: return MTLTextureType3D;
    case TextureKind::Cube: return MTLTextureTypeCube;
    case TextureKind::CubeArray: return MTLTextureTypeCubeArray;
    }
    return MTLTextureType2D;
}

} // Anonymous namespace

u64 DeviceMemoryBudget(const Device& device) {
    return device.MemoryBudget();
}

bool IsConverted(const Device& device, PixelFormat format) {
    if (VideoCore::Surface::IsPixelFormatASTC(format)) {
        // Apple GPUs (every Apple GPU family) read ASTC; Intel and AMD Macs don't.
        return device.AppleFamily() == 0;
    }
    if (VideoCore::Surface::IsPixelFormatBCn(format)) {
        return !device.SupportsBCTextures();
    }
    return false;
}

FormatInfo SurfaceFormat(const Device& device, PixelFormat format, bool converted) {
    if (converted) {
        const bool srgb = VideoCore::Surface::IsPixelFormatSRGB(format);
        MTLPixelFormat decoded = srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
        switch (format) {
        case PixelFormat::BC4_UNORM: decoded = MTLPixelFormatR8Unorm; break;
        case PixelFormat::BC4_SNORM: decoded = MTLPixelFormatR8Snorm; break;
        case PixelFormat::BC5_UNORM: decoded = MTLPixelFormatRG8Unorm; break;
        case PixelFormat::BC5_SNORM: decoded = MTLPixelFormatRG8Snorm; break;
        case PixelFormat::BC6H_UFLOAT:
        case PixelFormat::BC6H_SFLOAT: decoded = MTLPixelFormatRGBA16Float; break;
        default: break;
        }
        return FormatInfo{
            .format = static_cast<u32>(decoded),
            .attachable = true,
            .storage = !srgb,
            .format_class = FormatClass::Float,
        };
    }
    const Entry entry = Lookup(format);
    return FormatInfo{
        .format = static_cast<u32>(entry.format),
        .attachable = entry.attachable,
        .storage = entry.storage,
        .format_class = ClassOf(static_cast<u32>(entry.format)),
        .swizzle = entry.swizzle,
    };
}

FormatClass ClassOf(u32 mtl_format) {
    switch (static_cast<MTLPixelFormat>(mtl_format)) {
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA32Uint:
        return FormatClass::Uint;
    case MTLPixelFormatR8Sint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA32Sint:
        return FormatClass::Sint;
    case MTLPixelFormatDepth16Unorm:
    case MTLPixelFormatDepth32Float:
        return FormatClass::Depth;
    case MTLPixelFormatStencil8:
        return FormatClass::Stencil;
    case MTLPixelFormatDepth32Float_Stencil8:
        return FormatClass::DepthStencil;
    default:
        return FormatClass::Float;
    }
}

u32 DrawableFormat() {
    return static_cast<u32>(MTLPixelFormatBGRA8Unorm);
}

Object CreateTexture(const Device& device, const TextureDesc& desc) {
    if (PipelineCheckEnabled()) {
        // Metal aborts the process for a texture descriptor it rejects.
        const std::string reason = CheckTexture(desc, DevicePipelineLimits(device));
        if (!reason.empty()) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal, "Metal: texture refused: {}; kind {} format {} "
                            "{}x{}x{} levels {} layers {} samples {}",
                            reason, static_cast<u32>(desc.kind), desc.format, desc.width,
                            desc.height, desc.depth, desc.levels, desc.layers, desc.samples);
            }
            return {};
        }
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = TextureType(desc.kind);
    descriptor.pixelFormat = static_cast<MTLPixelFormat>(desc.format);
    descriptor.width = (std::max)(desc.width, 1U);
    descriptor.height = (std::max)(desc.height, 1U);
    descriptor.depth = (std::max)(desc.depth, 1U);
    descriptor.mipmapLevelCount = (std::max)(desc.levels, 1U);
    // Cube textures count faces in arrayLength / 6.
    const bool cube = desc.kind == TextureKind::Cube || desc.kind == TextureKind::CubeArray;
    descriptor.arrayLength = (std::max)(cube ? desc.layers / 6 : desc.layers, 1U);
    descriptor.sampleCount = (std::max)(desc.samples, 1U);
    descriptor.storageMode = MTLStorageModePrivate;
    // Format views reinterpret texture cache images (ImageView formats, copies between
    // compatible formats). It costs lossless compression on some GPUs; narrowing it to images
    // that need it is a later step (docs/nx2/metal-renderer.md).
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (desc.render_target) {
        usage |= MTLTextureUsageRenderTarget;
    }
    if (desc.shader_write) {
        usage |= MTLTextureUsageShaderWrite;
    }
    descriptor.usage = usage;
    // Metal orders the encoders until the scheduler tracks hazards itself (untracked heaps).
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    return Retain([device.GetDevice() newTextureWithDescriptor:descriptor]);
}

Object CreateTextureView(TextureHandle texture, u32 format, TextureKind kind, u32 base_level,
                         u32 levels, u32 base_layer, u32 layers, std::array<u8, 4> swizzle) {
    const id<MTLTexture> base = AsTexture(texture);
    if (base == nil) {
        return {};
    }
    if (PipelineCheckEnabled()) {
        // A range of levels or slices the texture doesn't have aborts the process.
        const bool cube = base.textureType == MTLTextureTypeCube ||
                          base.textureType == MTLTextureTypeCubeArray;
        const u64 slices =
            base.textureType == MTLTextureType3D ? 1 : u64{base.arrayLength} * (cube ? 6 : 1);
        if (levels == 0 || layers == 0 || u64{base_level} + levels > base.mipmapLevelCount ||
            u64{base_layer} + layers > slices) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal, "Metal: texture view refused: levels {}+{} of {}, "
                            "slices {}+{} of {}, kind {} format {}",
                            base_level, levels, static_cast<u32>(base.mipmapLevelCount),
                            base_layer, layers, slices, static_cast<u32>(kind), format);
            }
            return {};
        }
    }
    const MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
        static_cast<MTLTextureSwizzle>(swizzle[0]), static_cast<MTLTextureSwizzle>(swizzle[1]),
        static_cast<MTLTextureSwizzle>(swizzle[2]), static_cast<MTLTextureSwizzle>(swizzle[3]));
    return Retain([base newTextureViewWithPixelFormat:static_cast<MTLPixelFormat>(format)
                                          textureType:TextureType(kind)
                                               levels:NSMakeRange(base_level, levels)
                                               slices:NSMakeRange(base_layer, layers)
                                              swizzle:channels]);
}

u32 TextureBufferAlignment(const Device& device, u32 format) {
    return static_cast<u32>([device.GetDevice()
        minimumTextureBufferAlignmentForPixelFormat:static_cast<MTLPixelFormat>(format)]);
}

Object CreateTextureBufferView(BufferHandle buffer, u64 offset, u32 elements, u32 bytes_per_texel,
                               u32 format, bool writable) {
    const id<MTLBuffer> object = AsBuffer(buffer);
    if (object == nil || elements == 0) {
        return {};
    }
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (writable) {
        usage |= MTLTextureUsageShaderWrite;
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        textureBufferDescriptorWithPixelFormat:static_cast<MTLPixelFormat>(format)
                                         width:elements
                               resourceOptions:object.resourceOptions
                                         usage:usage];
    return Retain([object newTextureWithDescriptor:descriptor
                                            offset:offset
                                       bytesPerRow:u64{elements} * bytes_per_texel]);
}

Object CreateBuffer(const Device& device, u64 size, bool shared, u8** contents) {
    const MTLResourceOptions options =
        shared ? (MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)
               : MTLResourceStorageModePrivate;
    const id<MTLBuffer> buffer = [device.GetDevice() newBufferWithLength:(std::max)(size, u64{16})
                                                                 options:options];
    if (contents != nullptr) {
        *contents = buffer != nil && shared ? static_cast<u8*>(buffer.contents) : nullptr;
    }
    return Retain(buffer);
}

Object CreateLinearClampSampler(const Device& device) {
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = MTLSamplerMinMagFilterLinear;
    descriptor.magFilter = MTLSamplerMinMagFilterLinear;
    descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    return Retain([device.GetDevice() newSamplerStateWithDescriptor:descriptor]);
}

Object CreateSampler(const Device& device, const Tegra::Texture::TSCEntry& tsc) {
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    using Tegra::Texture::WrapMode;
    const auto filter = [](TextureFilter value) {
        return value == TextureFilter::Linear ? MTLSamplerMinMagFilterLinear
                                              : MTLSamplerMinMagFilterNearest;
    };
    const auto address = [](WrapMode mode) {
        switch (mode) {
        case WrapMode::Wrap: return MTLSamplerAddressModeRepeat;
        case WrapMode::Mirror: return MTLSamplerAddressModeMirrorRepeat;
        case WrapMode::ClampToEdge:
        case WrapMode::Clamp:
        case WrapMode::MirrorOnceClampOGL: return MTLSamplerAddressModeClampToEdge;
        case WrapMode::Border:
        case WrapMode::MirrorOnceBorder: return MTLSamplerAddressModeClampToBorderColor;
        case WrapMode::MirrorOnceClampToEdge: return MTLSamplerAddressModeMirrorClampToEdge;
        }
        return MTLSamplerAddressModeRepeat;
    };
    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.minFilter = filter(tsc.min_filter.Value());
    descriptor.magFilter = filter(tsc.mag_filter.Value());
    switch (tsc.mipmap_filter.Value()) {
    case TextureMipmapFilter::Nearest: descriptor.mipFilter = MTLSamplerMipFilterNearest; break;
    case TextureMipmapFilter::Linear: descriptor.mipFilter = MTLSamplerMipFilterLinear; break;
    default: descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
    }
    descriptor.sAddressMode = address(tsc.wrap_u.Value());
    descriptor.tAddressMode = address(tsc.wrap_v.Value());
    descriptor.rAddressMode = address(tsc.wrap_p.Value());
    // Metal has three border colours: the guest's nearest one.
    const auto border = tsc.BorderColor();
    if (border[3] < 0.5f) {
        descriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    } else if (border[0] + border[1] + border[2] > 1.5f) {
        descriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
    } else {
        descriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
    }
    if (tsc.depth_compare_enabled != 0) {
        // Tegra's DepthCompareFunc has MTLCompareFunction's values.
        descriptor.compareFunction = static_cast<MTLCompareFunction>(tsc.depth_compare_func.Value());
    }
    descriptor.lodMinClamp = tsc.MinLod();
    descriptor.lodMaxClamp = (std::max)(tsc.MaxLod(), tsc.MinLod());
    descriptor.maxAnisotropy =
        static_cast<NSUInteger>(std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f));
    descriptor.normalizedCoordinates = YES;
    descriptor.supportArgumentBuffers = YES;
    return Retain([device.GetDevice() newSamplerStateWithDescriptor:descriptor]);
}

namespace {
[[nodiscard]] NSString* ToNSString(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}
} // Anonymous namespace

Object CreateLibrary(const Device& device, std::string_view source, std::string& error) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    // The recompiler marks what must stay exact (NoContraction); the rest may use fast math.
    options.fastMathEnabled = YES;
    NSError* ns_error = nil;
    const id<MTLLibrary> library = [device.GetDevice() newLibraryWithSource:ToNSString(source)
                                                                    options:options
                                                                      error:&ns_error];
    if (library == nil) {
        error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
    }
    return Retain(library);
}

Object CreateFunction(const void* library, std::string_view name) {
    const id<MTLLibrary> object = (__bridge id<MTLLibrary>)library;
    return object != nil ? Retain([object newFunctionWithName:ToNSString(name)]) : Object{};
}

Object CreateComputeState(const Device& device, MTLComputePipelineDescriptor* descriptor,
                          std::string& error, PipelineArchive* archive) {
    const id<MTLBinaryArchive> binary =
        archive != nullptr ? (__bridge id<MTLBinaryArchive>)archive->Handle() : nil;
    const auto start = std::chrono::steady_clock::now();
    id<MTLComputePipelineState> state = nil;
    if (binary != nil) {
        descriptor.binaryArchives = @[ binary ];
        state = [device.GetDevice()
            newComputePipelineStateWithDescriptor:descriptor
                                          options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                       reflection:nil
                                            error:nil];
    }
    const bool hit = state != nil;
    NSError* ns_error = nil;
    if (!hit) {
        state = [device.GetDevice() newComputePipelineStateWithDescriptor:descriptor
                                                                  options:MTLPipelineOptionNone
                                                               reflection:nil
                                                                    error:&ns_error];
    }
    if (state == nil) {
        error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
        return {};
    }
    if (binary != nil) {
        archive->NoteBuild(hit, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - start)
                                                     .count()));
        if (!hit) {
            archive->NoteAdded([binary addComputePipelineFunctionsWithDescriptor:descriptor
                                                                           error:nil]);
        }
    }
    return Retain(state);
}

Object CreateComputePipelineState(const Device& device, const void* function, std::string& error,
                                  PipelineArchive* archive) {
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = (__bridge id<MTLFunction>)function;
    return CreateComputeState(device, descriptor, error, archive);
}

u64 AllocatedSize(const void* resource) {
    const id<MTLResource> object = (__bridge id<MTLResource>)resource;
    return object != nil ? object.allocatedSize : 0;
}

void SetLabel(const void* resource, std::string_view label) {
    const id<MTLResource> object = (__bridge id<MTLResource>)resource;
    if (object != nil) {
        object.label = [[NSString alloc] initWithBytes:label.data()
                                                length:label.size()
                                              encoding:NSUTF8StringEncoding];
    }
}

} // namespace Metal
