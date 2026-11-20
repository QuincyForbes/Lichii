// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "video_core/texture_cache/decode_etc.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/texture_cache/decode_bc.h"

// MTLSamplerDescriptor's lodBias and reductionMode came with the macOS / iOS 26 SDKs.
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
#define LICHII_METAL_SAMPLER_BIAS 1
#else
#define LICHII_METAL_SAMPLER_BIAS 0
#endif

namespace Metal {

using VideoCore::Surface::PixelFormat;

namespace {

// metal_formats.h names Metal's formats by number.
#define X(name, value) static_assert(static_cast<u32>(MTLPixelFormat##name) == value);
METAL_PIXEL_FORMATS(X)
#undef X
#define X(name) static_assert(static_cast<u32>(MTLTextureSwizzle##name) == u32{Swizzle::name});
X(Zero) X(One) X(Red) X(Green) X(Blue) X(Alpha)
#undef X

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

TextureKind KindOf(MTLTextureType type) {
    switch (type) {
    case MTLTextureType1D: return TextureKind::e1D;
    case MTLTextureType1DArray: return TextureKind::e1DArray;
    case MTLTextureType2DArray: return TextureKind::e2DArray;
    case MTLTextureType2DMultisample: return TextureKind::e2DMultisample;
    case MTLTextureType2DMultisampleArray: return TextureKind::e2DMultisampleArray;
    case MTLTextureType3D: return TextureKind::e3D;
    case MTLTextureTypeCube: return TextureKind::Cube;
    case MTLTextureTypeCubeArray: return TextureKind::CubeArray;
    default: return TextureKind::e2D;
    }
}

/// Bytes of a texel of a plain colour format, 0 for the others.
u32 TexelBytes(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatA8Unorm:
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Unorm_sRGB:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
        return 1;
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Unorm_sRGB:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatB5G6R5Unorm:
    case MTLPixelFormatA1BGR5Unorm:
    case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm:
        return 2;
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatBGR10A2Unorm:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
        return 4;
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA16Float:
        return 8;
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
    case MTLPixelFormatRGBA32Float:
        return 16;
    default:
        return 0;
    }
}

/// MoltenVK's view classes, 0 for a format in none: plain colour formats by texel size, a
/// compressed format with its sRGB, signed and HDR variants.
u32 ViewClass(MTLPixelFormat format) {
    constexpr u32 Compressed = 0x100;
    if (const u32 bytes = TexelBytes(format)) {
        return bytes;
    }
    const u32 raw = static_cast<u32>(format);
    // Each of these is followed by its variant.
    if ((format >= MTLPixelFormatBC1_RGBA && format <= MTLPixelFormatBC7_RGBAUnorm_sRGB) ||
        (format >= MTLPixelFormatEAC_RGBA8 && format <= MTLPixelFormatETC2_RGB8A1_sRGB)) {
        return Compressed + (raw & ~1U);
    }
    if (format == MTLPixelFormatEAC_R11Unorm || format == MTLPixelFormatEAC_R11Snorm) {
        return Compressed + static_cast<u32>(MTLPixelFormatEAC_R11Unorm);
    }
    if (format == MTLPixelFormatEAC_RG11Unorm || format == MTLPixelFormatEAC_RG11Snorm) {
        return Compressed + static_cast<u32>(MTLPixelFormatEAC_RG11Unorm);
    }
    // ASTC's sRGB, LDR and HDR formats list the block sizes in one order.
    for (const MTLPixelFormat first : {MTLPixelFormatASTC_4x4_sRGB, MTLPixelFormatASTC_4x4_LDR,
                                       MTLPixelFormatASTC_4x4_HDR}) {
        const u32 last = static_cast<u32>(first) + static_cast<u32>(MTLPixelFormatASTC_12x12_LDR) -
                         static_cast<u32>(MTLPixelFormatASTC_4x4_LDR);
        if (raw >= static_cast<u32>(first) && raw <= last) {
            return Compressed + raw - static_cast<u32>(first) +
                   static_cast<u32>(MTLPixelFormatASTC_4x4_LDR);
        }
    }
    return 0;
}

/// Why Metal would reject by assertion a view in format `to` of a texture in format `from`, or
/// nil: it casts between colour formats of one texel size, between the variants of a compressed
/// format and to the stencil half of a depth-stencil format, nothing else.
const char* ViewFormatReason(MTLPixelFormat from, MTLPixelFormat to) {
    if (from == to) {
        return nullptr;
    }
    const auto depth_stencil = [](MTLPixelFormat format) {
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        return format_class == FormatClass::Depth || format_class == FormatClass::Stencil ||
               format_class == FormatClass::DepthStencil || format == MTLPixelFormatX32_Stencil8;
    };
    if (depth_stencil(from) || depth_stencil(to)) {
        return from == MTLPixelFormatDepth32Float_Stencil8 && to == MTLPixelFormatX32_Stencil8
                   ? nullptr
                   : "a depth or stencil format seen as another format";
    }
    const u32 from_class = ViewClass(from);
    const u32 to_class = ViewClass(to);
    if (from_class != 0 && to_class != 0 && from_class != to_class) {
        return "formats of different texel sizes or compressions";
    }
    return nullptr;
}

MTLTextureDescriptor* MakeTextureDescriptor(const TextureDesc& desc) {
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
    // compatible formats); internal targets, MSAA textures and pure depth omit the flag so Apple
    // GPUs keep lossless compression on them.
    const bool pure_depth = desc.format == Fmt::Depth32Float || desc.format == Fmt::Depth16Unorm;
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (desc.pixel_format_view && desc.samples <= 1 && !pure_depth) {
        usage |= MTLTextureUsagePixelFormatView;
    }
    if (desc.render_target) {
        usage |= MTLTextureUsageRenderTarget;
    }
    if (desc.shader_write) {
        usage |= MTLTextureUsageShaderWrite;
    }
    descriptor.usage = usage;
    // Metal orders the encoders until the scheduler tracks hazards itself (untracked heaps).
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    return descriptor;
}

} // Anonymous namespace

const void* DeviceHandle(const Device& device) {
    return (__bridge const void*)device.GetDevice();
}

u64 DeviceMemoryBudget(const Device& device) {
    return device.MemoryBudget();
}

u32 DeviceAppleFamily(const Device& device) {
    return device.AppleFamily();
}

bool DeviceSupportsMeshShaders(const Device& device) {
    return device.SupportsMeshShaders();
}

std::unique_ptr<Device, void (*)(Device*)> CreateOfflineDevice() {
    constexpr auto deleter = +[](Device* ptr) { delete ptr; };
    try {
        return {new Device(), deleter};
    } catch (...) {
        return {nullptr, deleter};
    }
}

void InAutoreleasePool(const std::function<void()>& work) {
    @autoreleasepool {
        work();
    }
}

bool IsConverted(const Device& device, PixelFormat format) {
    if (VideoCore::Surface::IsPixelFormatASTC(format)) {
        // Apple GPUs (every Apple GPU family) read ASTC; Intel and AMD Macs don't.
        return device.AppleFamily() == 0;
    }
    if (VideoCore::Surface::IsPixelFormatBCn(format)) {
        return !device.SupportsBCTextures();
    }
    if (VideoCommon::IsEtcFormat(format)) {
        // Apple GPUs read ETC2 and EAC; Intel and AMD Macs don't (decoded on the CPU,
        // VideoCommon::DecodeEtc). EDEN_NX2_METAL_ETC=cpu decodes them on any GPU.
        const char* const value = std::getenv("EDEN_NX2_METAL_ETC");
        return device.AppleFamily() == 0 || (value != nullptr && std::strcmp(value, "cpu") == 0);
    }
    // Metal has no 96-bit and no 8-bit 4:4 format.
    if (VideoCommon::IsExpandedFormat(format)) {
        return FormatsExpanded();
    }
    return false;
}

bool PackedAsRgba8(const Device& device) {
    const char* const value = std::getenv("EDEN_NX2_METAL_PACKED");
    return device.AppleFamily() == 0 || (value != nullptr && std::strcmp(value, "rgba8") == 0);
}

FormatFallback FallbackOf(const Device& device, PixelFormat format) {
    if (Pack16Mode(format) != 0 &&
        (Pack16Mode(format) == 1 || PackedAsRgba8(device))) {
        return FormatFallback::Stand_in;
    }
    if (IsConverted(device, format)) {
        return VideoCommon::IsExpandedFormat(format) ? FormatFallback::Expanded
                                                     : FormatFallback::Decoded;
    }
    switch (format) {
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM: return FormatFallback::Widened;
    default: return FormatFallback::Native;
    }
}

FormatInfo SurfaceFormat(const Device& device, PixelFormat format, bool converted) {
    const u32 mode = Pack16Mode(format);
    if (!converted && mode > 1 && PackedAsRgba8(device)) {
        return FormatInfo{.format = Fmt::RGBA8Unorm, .attachable = true, .storage = true,
                          .format_class = FormatClass::Float};
    }
    return SurfaceFormat(format, converted);
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
    @autoreleasepool {
        MTLTextureDescriptor* descriptor = MakeTextureDescriptor(desc);
        return Retain([device.GetDevice() newTextureWithDescriptor:descriptor]);
    }
}

bool SupportsSparseTextures(const Device& device) {
    // Apple6 and later (A13, M1) have sparse textures; tier 2 (unmapped tiles read zero, no
    // residency query needed) is told by Metal 4's SDK and OS.
    if (device.AppleFamily() < 6) {
        return false;
    }
#if LICHII_METAL_SAMPLER_BIAS // an older SDK can't check the texture's tier
    if (@available(macOS 26.0, iOS 26.0, *)) {
        return true;
    }
#endif
    return false;
}

std::optional<SparseTexture> CreateSparseTexture(const Device& device, const TextureDesc& desc,
                                                 u64 heap_bytes) {
    if (!SupportsSparseTextures(device) || desc.kind != TextureKind::e2D || desc.samples != 1) {
        return std::nullopt;
    }
    @autoreleasepool {
        const id<MTLDevice> mtl = device.GetDevice();
        MTLTextureDescriptor* descriptor = MakeTextureDescriptor(desc);
        const MTLSize tile = [mtl sparseTileSizeWithTextureType:descriptor.textureType
                                                    pixelFormat:descriptor.pixelFormat
                                                    sampleCount:1];
        const u32 tile_bytes = static_cast<u32>([mtl sparseTileSizeInBytes]);
        if (tile.width == 0 || tile.height == 0 || tile_bytes == 0) {
            return std::nullopt;
        }
        MTLHeapDescriptor* heap_descriptor = [MTLHeapDescriptor new];
        heap_descriptor.type = MTLHeapTypeSparse;
        heap_descriptor.storageMode = MTLStorageModePrivate;
        heap_descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
        heap_descriptor.size = Common::AlignUp(heap_bytes, u64{tile_bytes});
        const id<MTLHeap> heap = [mtl newHeapWithDescriptor:heap_descriptor];
        if (heap == nil) {
            return std::nullopt;
        }
        const id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor];
        if (texture == nil) {
            return std::nullopt;
        }
#if LICHII_METAL_SAMPLER_BIAS // sparseTextureTier came with the same 26 SDKs
        if (@available(macOS 26.0, iOS 26.0, *)) {
            if (texture.sparseTextureTier != MTLTextureSparseTier2) {
                return std::nullopt;
            }
        }
#endif
        const u32 levels = desc.levels;
        const u32 tail_level =
            static_cast<u32>((std::min)(static_cast<u32>(texture.firstMipmapInTail), levels));
        return SparseTexture{
            .texture = Retain(texture),
            .heap = Retain(heap),
            .tile_width = static_cast<u32>(tile.width),
            .tile_height = static_cast<u32>(tile.height),
            .tile_bytes = tile_bytes,
            .tail_level = tail_level,
            .tail_tiles = tail_level < levels ? (std::max)(1U, static_cast<u32>(Common::DivCeil(
                                                                   u64{texture.tailSizeInBytes},
                                                                   u64{tile_bytes})))
                                              : 0U,
            .heap_bytes = static_cast<u64>(heap.size),
        };
    }
}

u64 SparseHeapUsedBytes(const void* heap) {
    const id<MTLHeap> __unsafe_unretained mtl_heap = (__bridge __unsafe_unretained id<MTLHeap>)heap;
    return mtl_heap != nil ? static_cast<u64>(mtl_heap.usedSize) : 0;
}

Object CreateTextureView(TextureHandle texture, u32 format, TextureKind kind, u32 base_level,
                         u32 levels, u32 base_layer, u32 layers, std::array<u8, 4> swizzle) {
    const id<MTLTexture> __unsafe_unretained base =
        (__bridge __unsafe_unretained id<MTLTexture>)texture;
    if (base == nil) {
        return {};
    }
    if (PipelineCheckEnabled()) {
        // A range of levels or slices the texture doesn't have, a view of another family of
        // texture types and a format Metal doesn't cast to abort the process.
        const bool cube = base.textureType == MTLTextureTypeCube ||
                          base.textureType == MTLTextureTypeCubeArray;
        const u64 slices =
            base.textureType == MTLTextureType3D ? 1 : u64{base.arrayLength} * (cube ? 6 : 1);
        std::string reason;
        if (levels == 0 || layers == 0 || u64{base_level} + levels > base.mipmapLevelCount ||
            u64{base_layer} + layers > slices) {
            reason = "levels or slices past the texture";
        } else if (base.pixelFormat != static_cast<MTLPixelFormat>(format) &&
                   (base.usage & MTLTextureUsagePixelFormatView) == 0) {
            reason = "made without format views";
        } else if (reason = CheckTextureView(KindOf(base.textureType), kind, layers);
                   reason.empty()) {
            if (const char* const cast = ViewFormatReason(
                    base.pixelFormat, static_cast<MTLPixelFormat>(format))) {
                reason = cast;
            }
        }
        if (!reason.empty()) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal, "Metal: texture view refused: {}; levels {}+{} of {}, "
                            "slices {}+{} of {}, kind {} of {}, format {} of {}",
                            reason, base_level, levels, static_cast<u32>(base.mipmapLevelCount),
                            base_layer, layers, slices, static_cast<u32>(kind),
                            static_cast<u32>(KindOf(base.textureType)), format,
                            static_cast<u32>(base.pixelFormat));
            }
            return {};
        }
    }
    @autoreleasepool {
        const MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
            static_cast<MTLTextureSwizzle>(swizzle[0]), static_cast<MTLTextureSwizzle>(swizzle[1]),
            static_cast<MTLTextureSwizzle>(swizzle[2]), static_cast<MTLTextureSwizzle>(swizzle[3]));
        return Retain([base newTextureViewWithPixelFormat:static_cast<MTLPixelFormat>(format)
                                              textureType:TextureType(kind)
                                                   levels:NSMakeRange(base_level, levels)
                                                   slices:NSMakeRange(base_layer, layers)
                                                  swizzle:channels]);
    }
}

u32 TextureBufferAlignment(const Device& device, u32 format) {
    return static_cast<u32>([device.GetDevice()
        minimumTextureBufferAlignmentForPixelFormat:static_cast<MTLPixelFormat>(format)]);
}

Object CreateTextureBufferView(BufferHandle buffer, u64 offset, u32 elements, u32 bytes_per_texel,
                               u32 format, bool writable) {
    const id<MTLBuffer> __unsafe_unretained object =
        (__bridge __unsafe_unretained id<MTLBuffer>)buffer;
    if (object == nil || elements == 0) {
        return {};
    }
    @autoreleasepool {
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
}

Object CreateBuffer(const Device& device, u64 size, bool shared, u8** contents,
                    bool untracked_write_combined) {
    @autoreleasepool {
        MTLResourceOptions options = MTLResourceStorageModePrivate;
        if (shared) {
            options = untracked_write_combined
                          ? (MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined |
                             MTLResourceHazardTrackingModeUntracked)
                          : (MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache);
        }
        const id<MTLBuffer> buffer =
            [device.GetDevice() newBufferWithLength:(std::max)(size, u64{16}) options:options];
        if (contents != nullptr) {
            *contents = buffer != nil && shared ? static_cast<u8*>(buffer.contents) : nullptr;
        }
        return Retain(buffer);
    }
}

Object CreateLinearClampSampler(const Device& device) {
    @autoreleasepool {
        MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
        descriptor.minFilter = MTLSamplerMinMagFilterLinear;
        descriptor.magFilter = MTLSamplerMinMagFilterLinear;
        descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
        return Retain([device.GetDevice() newSamplerStateWithDescriptor:descriptor]);
    }
}

bool SamplerBiasSupported(const Device& device) {
#if LICHII_METAL_SAMPLER_BIAS
    if (@available(iOS 26.0, macOS 26.0, *)) {
        // Older GPUs take the properties and sample without them (CI's paravirtual Apple5).
        return [device.GetDevice() supportsFamily:MTLGPUFamilyApple7];
    }
#endif
    return false;
}

u64 SamplerReductionBits(const Tegra::Texture::TSCEntry& tsc) {
    using Tegra::Texture::SamplerReduction;
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    const SamplerReduction reduction = tsc.reduction_filter.Value();
    if (reduction == SamplerReduction::WeightedAverage ||
        tsc.mag_filter.Value() != TextureFilter::Linear ||
        tsc.min_filter.Value() != TextureFilter::Linear) {
        return 0;
    }
    u64 bits = reduction == SamplerReduction::Min ? 1 : 2;
    if (tsc.mipmap_filter.Value() == TextureMipmapFilter::Linear) {
        bits |= 4;
    } else if (tsc.mipmap_filter.Value() == TextureMipmapFilter::Nearest) {
        bits |= 8;
    }
    return bits;
}

Object CreateSampler(const Device& device, const Tegra::Texture::TSCEntry& tsc,
                     const SamplerOptions& options) {
    using Tegra::Texture::SamplerReduction;
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    using Tegra::Texture::WrapMode;
    const bool plain = options.plain;
    const auto filter = [&](TextureFilter value) {
        return value == TextureFilter::Linear && !options.force_nearest
                   ? MTLSamplerMinMagFilterLinear
                   : MTLSamplerMinMagFilterNearest;
    };
    // As Vulkan's MaxwellToVK::Sampler::WrapMode: the OpenGL clamp reaches the border when the
    // guest filters, and both mirror-once modes Metal lacks mirror once and clamp to the edge.
    const auto address = [&](WrapMode mode) {
        switch (mode) {
        case WrapMode::Wrap: return MTLSamplerAddressModeRepeat;
        case WrapMode::Mirror: return MTLSamplerAddressModeMirrorRepeat;
        case WrapMode::ClampToEdge: return MTLSamplerAddressModeClampToEdge;
        case WrapMode::Clamp:
            return !plain && tsc.mag_filter.Value() == TextureFilter::Linear
                       ? MTLSamplerAddressModeClampToBorderColor
                       : MTLSamplerAddressModeClampToEdge;
        case WrapMode::Border: return MTLSamplerAddressModeClampToBorderColor;
        case WrapMode::MirrorOnceClampToEdge: return MTLSamplerAddressModeMirrorClampToEdge;
        case WrapMode::MirrorOnceBorder:
            return plain ? MTLSamplerAddressModeClampToBorderColor
                         : MTLSamplerAddressModeMirrorClampToEdge;
        case WrapMode::MirrorOnceClampOGL:
            return plain ? MTLSamplerAddressModeClampToEdge
                         : MTLSamplerAddressModeMirrorClampToEdge;
        }
        return MTLSamplerAddressModeRepeat;
    };
    @autoreleasepool {
        MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
        descriptor.minFilter = filter(tsc.min_filter.Value());
        descriptor.magFilter = filter(tsc.mag_filter.Value());
        switch (tsc.mipmap_filter.Value()) {
        case TextureMipmapFilter::Nearest: descriptor.mipFilter = MTLSamplerMipFilterNearest; break;
        case TextureMipmapFilter::Linear:
            descriptor.mipFilter =
                options.force_nearest ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
            break;
        default: descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
        }
        descriptor.sAddressMode = address(tsc.wrap_u.Value());
        descriptor.tAddressMode = address(tsc.wrap_v.Value());
        descriptor.rAddressMode = address(tsc.wrap_p.Value());
        // Metal has three border colours and no custom one: the guest's nearest (exact for
        // transparent black, opaque black and opaque white).
        const auto border = options.srgb_border ? tsc.SrgbBorderColor() : tsc.BorderColor();
        if (border[3] < 0.5f) {
            descriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
        } else if (border[0] + border[1] + border[2] > 1.5f) {
            descriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
        } else {
            descriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
        }
        if (tsc.depth_compare_enabled != 0 && !options.drop_compare) {
            // Tegra's DepthCompareFunc has MTLCompareFunction's values.
            descriptor.compareFunction =
                static_cast<MTLCompareFunction>(tsc.depth_compare_func.Value());
        }
        if (descriptor.mipFilter == MTLSamplerMipFilterNotMipmapped) {
            descriptor.lodMinClamp = 0.0f;
            descriptor.lodMaxClamp = 0.25f;
        } else {
            descriptor.lodMinClamp = tsc.MinLod();
            descriptor.lodMaxClamp = (std::max)(tsc.MaxLod(), tsc.MinLod());
            // Vulkan's EDEN_NX2_FORCE_LOD0 (read under EDEN_NX2_METAL_VULKAN_SWITCHES): mipmapped
            // samplers read the top level only (a debug aid for mip layout problems).
            static const bool force_lod0 = VulkanSwitch("EDEN_NX2_FORCE_LOD0") != nullptr;
            if (force_lod0) {
                descriptor.lodMinClamp = 0.0f;
                descriptor.lodMaxClamp = 0.25f;
            }
        }
        f32 anisotropy = std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f);
        if (options.force_nearest) {
            anisotropy = 1.0f;
        } else if (options.reduce_anisotropy) {
            anisotropy = static_cast<f32>(1U << tsc.max_anisotropy.Value());
        }
        descriptor.maxAnisotropy = static_cast<NSUInteger>(anisotropy);
#if LICHII_METAL_SAMPLER_BIAS
        if (@available(iOS 26.0, macOS 26.0, *)) {
            if (!plain && SamplerBiasSupported(device)) {
                descriptor.lodBias =
                    std::clamp(tsc.LodBias() + options.added_lod_bias, -15.0f, 15.0f);
                switch (tsc.reduction_filter.Value()) {
                case SamplerReduction::Min:
                    descriptor.reductionMode = MTLSamplerReductionModeMinimum;
                    break;
                case SamplerReduction::Max:
                    descriptor.reductionMode = MTLSamplerReductionModeMaximum;
                    break;
                default: break;
                }
            }
        }
#endif
        descriptor.normalizedCoordinates = YES;
        descriptor.supportArgumentBuffers = YES;
        return Retain([device.GetDevice() newSamplerStateWithDescriptor:descriptor]);
    }
}

namespace {
[[nodiscard]] NSString* ToNSString(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}

[[nodiscard]] MTLCompileOptions* MakeLibraryCompileOptions() {
    MTLCompileOptions* options = [MTLCompileOptions new];
    // The recompiler marks what must stay exact (NoContraction); the rest may use fast math.
    // EDEN_NX2_METAL_FAST_MATH=1: full fast math (A/B).
    static const bool full_fast_math = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FAST_MATH");
        return value != nullptr && value[0] == '1';
    }();
    // EDEN_NX2_METAL_SAFE_MATH=1: no fast math at all, positions invariant across programs
    // (A/B: a depth pre-pass and an EQUAL-tested colour pass of the same geometry must agree
    // bit for bit).
    static const bool safe_math = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SAFE_MATH");
        return value != nullptr && value[0] == '1';
    }();
    if (safe_math) {
        if (@available(iOS 18.0, macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
            options.fastMathEnabled = NO;
        }
    } else if (full_fast_math) {
        options.fastMathEnabled = YES;
    } else if (@available(iOS 18.0, macOS 15.0, *)) {
        // Relaxed keeps fast math's reassociation but still honours NaN and Inf, as MoltenVK
        // does; full fast math let one NaN (normalize(0), 0 * inf in specular terms) spread
        // through the lighting and bloom passes into white blobs.
        options.mathMode = MTLMathModeRelaxed;
    } else {
        options.fastMathEnabled = YES;
    }
    if (@available(iOS 17.0, macOS 14.0, *)) {
        if (options.languageVersion < MTLLanguageVersion3_1) {
            options.languageVersion = MTLLanguageVersion3_1;
        }
    }
    if (@available(iOS 14.0, macOS 11.0, *)) {
        options.preserveInvariance = YES;
    }
    return options;
}

void ConfigureComputeDescriptor(MTLComputePipelineDescriptor* descriptor, u32 workgroup_threads,
                                u32 immutable_buffers) {
    if (workgroup_threads > 0) {
        const u32 clamped = (std::min)(workgroup_threads, 1024U);
        descriptor.maxTotalThreadsPerThreadgroup = clamped;
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth =
            (clamped >= 32 && (clamped % 32) == 0) ? YES : NO;
    }
    u32 mask = immutable_buffers & ((1U << 31) - 1U);
    while (mask != 0) {
        const u32 slot = static_cast<u32>(std::countr_zero(mask));
        mask &= mask - 1U;
        descriptor.buffers[slot].mutability = MTLMutabilityImmutable;
    }
}
} // Anonymous namespace

Object CreateLibrary(const Device& device, std::string_view source, std::string& error) {
    @autoreleasepool {
        MTLCompileOptions* options = MakeLibraryCompileOptions();
        NSError* ns_error = nil;
        const id<MTLLibrary> library = [device.GetDevice() newLibraryWithSource:ToNSString(source)
                                                                        options:options
                                                                          error:&ns_error];
        if (library == nil) {
            error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
        }
        return Retain(library);
    }
}

void CreateLibraryAsync(const Device& device, std::string_view msl,
                        std::function<void(Object, std::string)> on_done) {
    @autoreleasepool {
        MTLCompileOptions* options = MakeLibraryCompileOptions();
        NSString* ns_source = ToNSString(msl);
        if (ns_source == nil) {
            if (on_done) {
                on_done(Object{}, "invalid UTF-8 MSL source");
            }
            return;
        }
        auto callback =
            std::make_shared<std::function<void(Object, std::string)>>(std::move(on_done));
        [device.GetDevice()
            newLibraryWithSource:ns_source
                         options:options
               completionHandler:^(id<MTLLibrary> library, NSError* ns_error) {
                   @autoreleasepool {
                       std::string error;
                       if (library == nil) {
                           error = ns_error != nil && ns_error.localizedDescription != nil
                                       ? ns_error.localizedDescription.UTF8String
                                       : "unknown error";
                       }
                       Object retained = Retain(library);
                       if (*callback) {
                           (*callback)(std::move(retained), std::move(error));
                       }
                   }
               }];
    }
}

Object CreateFunction(const void* library, std::string_view name) {
    const id<MTLLibrary> __unsafe_unretained object =
        (__bridge __unsafe_unretained id<MTLLibrary>)library;
    if (object == nil) {
        return {};
    }
    @autoreleasepool {
        return Retain([object newFunctionWithName:ToNSString(name)]);
    }
}

Object CreateComputeState(const Device& device, MTLComputePipelineDescriptor* descriptor,
                          std::string& error, PipelineArchive* archive) {
    @autoreleasepool {
        const id<MTLBinaryArchive> binary =
            archive != nullptr ? (__bridge id<MTLBinaryArchive>)archive->Handle() : nil;
        const auto start = std::chrono::steady_clock::now();
        id<MTLComputePipelineState> state = nil;
        std::shared_lock<std::shared_mutex> building;
        if (binary != nil) {
            building = std::shared_lock{archive->BuildMutex()};
            descriptor.binaryArchives = @[ binary ];
            state = [device.GetDevice()
                newComputePipelineStateWithDescriptor:descriptor
                                              options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                           reflection:nil
                                                error:nil];
        }
        if (building.owns_lock()) {
            // Release right after the probe: holding the archive lock across the fallback compile
            // (and the writer-preferring lock of an addition) serialized every builder.
            building.unlock();
        }
        const bool hit = state != nil;
        if (!hit && descriptor.binaryArchives != nil) {
            descriptor.binaryArchives = nil;
        }
        NSError* ns_error = nil;
        if (!hit) {
            state = [device.GetDevice() newComputePipelineStateWithDescriptor:descriptor
                                                                      options:MTLPipelineOptionNone
                                                                   reflection:nil
                                                                        error:&ns_error];
        }
        if (building.owns_lock()) {
            building.unlock();
        }
        if (state == nil) {
            error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
            return {};
        }
        if (binary != nil) {
            archive->NoteBuild(
                hit, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count()));
            if (!hit) {
                auto* const copy = [descriptor copy];
                archive->DeferAdd([binary, copy] {
                    return [binary addComputePipelineFunctionsWithDescriptor:copy error:nil] == YES;
                });
            }
        }
        return Retain(state);
    }
}

Object CreateComputePipelineState(const Device& device, const void* function, std::string& error,
                                  PipelineArchive* archive, u32 workgroup_threads,
                                  u32 immutable_buffers) {
    const id<MTLFunction> __unsafe_unretained mtl_function =
        (__bridge __unsafe_unretained id<MTLFunction>)function;
    if (mtl_function == nil) {
        error = "compute function is null";
        return {};
    }
    @autoreleasepool {
        if (workgroup_threads == 0 && immutable_buffers == 0 && archive == nullptr) {
            NSError* ns_error = nil;
            const id<MTLComputePipelineState> state =
                [device.GetDevice() newComputePipelineStateWithFunction:mtl_function
                                                                  error:&ns_error];
            if (state == nil) {
                error =
                    ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
                return {};
            }
            return Retain(state);
        }
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = mtl_function;
        ConfigureComputeDescriptor(descriptor, workgroup_threads, immutable_buffers);
        return CreateComputeState(device, descriptor, error, archive);
    }
}

void CreateComputeStateAsync(const Device& device, const void* library, std::string_view entry_name,
                             u32 workgroup_threads, PipelineArchive* archive,
                             std::function<void(Object, std::string)> on_done,
                             u32 immutable_buffers) {
    @autoreleasepool {
        const id<MTLLibrary> __unsafe_unretained mtl_lib =
            (__bridge __unsafe_unretained id<MTLLibrary>)library;
        NSString* const ns_entry = mtl_lib != nil ? ToNSString(entry_name) : nil;
        const id<MTLFunction> function =
            mtl_lib != nil && ns_entry != nil ? [mtl_lib newFunctionWithName:ns_entry] : nil;
        if (function == nil) {
            if (on_done) {
                on_done(Object{}, "compute entry function not found");
            }
            return;
        }
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        ConfigureComputeDescriptor(descriptor, workgroup_threads, immutable_buffers);

        const id<MTLBinaryArchive> binary =
            archive != nullptr ? (__bridge id<MTLBinaryArchive>)archive->Handle() : nil;
        const auto start = std::chrono::steady_clock::now();
        if (binary != nil && archive->HasEntries()) {
            id<MTLComputePipelineState> hit_state = nil;
            {
                std::shared_lock building{archive->BuildMutex()};
                descriptor.binaryArchives = @[ binary ];
                hit_state = [device.GetDevice()
                    newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                               reflection:nil
                                                    error:nil];
            }
            descriptor.binaryArchives = nil;
            if (hit_state != nil) {
                archive->NoteBuild(
                    true, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - start)
                                               .count()));
                Object retained = Retain(hit_state);
                if (on_done) {
                    on_done(std::move(retained), std::string{});
                }
                return;
            }
        }

        MTLComputePipelineDescriptor* const copy = binary != nil ? [descriptor copy] : nil;
        auto callback =
            std::make_shared<std::function<void(Object, std::string)>>(std::move(on_done));
        [device.GetDevice()
            newComputePipelineStateWithDescriptor:descriptor
                                          options:MTLPipelineOptionNone
                                completionHandler:^(id<MTLComputePipelineState> state,
                                                    MTLComputePipelineReflection*,
                                                    NSError* ns_error) {
                                    @autoreleasepool {
                                        std::string error;
                                        if (state == nil) {
                                            error = ns_error != nil &&
                                                            ns_error.localizedDescription != nil
                                                        ? ns_error.localizedDescription.UTF8String
                                                        : "unknown error";
                                        } else if (binary != nil && archive != nullptr) {
                                            archive->NoteBuild(
                                                false,
                                                static_cast<u64>(
                                                    std::chrono::duration_cast<
                                                        std::chrono::nanoseconds>(
                                                        std::chrono::steady_clock::now() - start)
                                                        .count()));
                                            archive->DeferAdd([binary, copy] {
                                                return [binary
                                                           addComputePipelineFunctionsWithDescriptor:
                                                               copy
                                                                                               error:
                                                                                                   nil] ==
                                                       YES;
                                            });
                                        }
                                        Object retained = Retain(state);
                                        if (*callback) {
                                            (*callback)(std::move(retained), std::move(error));
                                        }
                                    }
                                }];
    }
}

u64 DeviceAllocatedSize(const Device& device) {
    return device.AllocatedSize();
}

TextureFacts DescribeTexture(TextureHandle texture) {
    const id<MTLTexture> __unsafe_unretained object =
        (__bridge __unsafe_unretained id<MTLTexture>)texture;
    if (object == nil) {
        return {.format = 0};
    }
    const bool cube =
        object.textureType == MTLTextureTypeCube || object.textureType == MTLTextureTypeCubeArray;
    return TextureFacts{
        .format = static_cast<u32>(object.pixelFormat),
        .samples = static_cast<u32>(object.sampleCount),
        .levels = static_cast<u32>(object.mipmapLevelCount),
        .slices = object.textureType == MTLTextureType3D
                      ? 1U
                      : static_cast<u32>(object.arrayLength) * (cube ? 6U : 1U),
        .depth = static_cast<u32>(object.depth),
    };
}

u64 AllocatedSize(const void* resource) {
    const id<MTLResource> __unsafe_unretained object =
        (__bridge __unsafe_unretained id<MTLResource>)resource;
    return object != nil ? object.allocatedSize : 0;
}

void SetLabel(const void* resource, std::string_view label) {
    const id<MTLResource> __unsafe_unretained object =
        (__bridge __unsafe_unretained id<MTLResource>)resource;
    if (object != nil) {
        @autoreleasepool {
            object.label = ToNSString(label);
        }
    }
}

} // namespace Metal
