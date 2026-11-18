// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <optional>
#include <string_view>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/dlss_contract.h"
#include "video_core/nx2.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_fsr3_upscaler.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_present.h"
#include "video_core/renderer_metal/metal_texture_cache.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/decode_bc.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/util.h"

namespace Metal {

using VideoCommon::ImageFlagBits;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewFlagBits;
using VideoCommon::ImageViewInfo;
using VideoCommon::ImageViewType;
using VideoCore::Surface::PixelFormat;

namespace {

/// Logs a message the first time `flag` is seen unset.
#define LOG_ONCE(flag, ...)                                                                        \
    do {                                                                                           \
        static std::atomic_bool flag{false};                                                       \
        if (!flag.exchange(true)) {                                                                \
            LOG_WARNING(Render_Metal, __VA_ARGS__);                                                \
        }                                                                                          \
    } while (false)

[[nodiscard]] bool IsColor(FormatClass format_class) {
    return format_class == FormatClass::Float || format_class == FormatClass::Uint ||
           format_class == FormatClass::Sint;
}

[[nodiscard]] bool IsDepth(FormatClass format_class) {
    return format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
}

// EDEN_NX2_METAL_COPIES=0: blits between depth images and copies between block-compressed
// and plain formats are skipped, as this backend started.
[[nodiscard]] bool CopiesEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_COPIES");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_SAMPLERS=0: one sampler state per guest sampler, without LOD bias, reduction
// or the variants a view asks for, Clamp and the mirror-once modes as this backend started.
[[nodiscard]] bool SamplersEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SAMPLERS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_MSAA=0: multisampled images are neither uploaded, downloaded, copied to or from
// single-sampled ones nor resolved, and one bound where a shader reads a plain texture is bound
// as it is, as this backend started.
[[nodiscard]] bool MsaaEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MSAA");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// The conversion Vulkan's ConvertImage draws for these formats (vk_texture_cache.cpp).
[[nodiscard]] std::optional<ConvertKind> ConversionFor(PixelFormat dst, PixelFormat src) {
    const bool abgr8 = src == PixelFormat::A8B8G8R8_UNORM || src == PixelFormat::B8G8R8A8_UNORM;
    const bool abgr8_any = abgr8 || src == PixelFormat::A8B8G8R8_SRGB ||
                           src == PixelFormat::B8G8R8A8_SRGB;
    switch (dst) {
    case PixelFormat::R16_UNORM:
        if (src == PixelFormat::D16_UNORM) {
            return ConvertKind::DepthToColor;
        }
        break;
    case PixelFormat::A8B8G8R8_SRGB:
    case PixelFormat::B8G8R8A8_SRGB:
    case PixelFormat::B8G8R8A8_UNORM:
        if (src == PixelFormat::D32_FLOAT) {
            return ConvertKind::DepthToColor;
        }
        break;
    case PixelFormat::A8B8G8R8_UNORM:
        switch (src) {
        case PixelFormat::X8_D24_UNORM: return ConvertKind::X8d24ToAbgr8;
        case PixelFormat::S8_UINT_D24_UNORM: return ConvertKind::D24s8ToAbgr8;
        case PixelFormat::D24_UNORM_S8_UINT: return ConvertKind::S8d24ToAbgr8;
        case PixelFormat::D32_FLOAT: return ConvertKind::DepthToColor;
        default: break;
        }
        break;
    case PixelFormat::R32_FLOAT:
        if (src == PixelFormat::D32_FLOAT) {
            return ConvertKind::DepthToColor;
        }
        break;
    case PixelFormat::D16_UNORM:
        if (src == PixelFormat::R16_UNORM) {
            return ConvertKind::ColorToDepth;
        }
        break;
    case PixelFormat::X8_D24_UNORM:
        if (src == PixelFormat::A8B8G8R8_UNORM) {
            return ConvertKind::Abgr8ToX8d24;
        }
        break;
    case PixelFormat::S8_UINT_D24_UNORM:
        if (abgr8) {
            return ConvertKind::Abgr8ToD24s8;
        }
        break;
    case PixelFormat::D32_FLOAT:
        if (abgr8_any) {
            return ConvertKind::Abgr8ToD32f;
        }
        if (src == PixelFormat::R32_FLOAT) {
            return ConvertKind::ColorToDepth;
        }
        break;
    case PixelFormat::D24_UNORM_S8_UINT:
        if (abgr8_any) {
            return ConvertKind::Abgr8ToD24s8;
        }
        break;
    default:
        break;
    }
    // Vulkan's fallback: a depth image into a colour one (the bloom fix, B5G6R5).
    if (src == PixelFormat::D32_FLOAT &&
        VideoCore::Surface::GetFormatType(dst) == VideoCore::Surface::SurfaceType::ColorTexture &&
        (dst == PixelFormat::B5G6R5_UNORM || Settings::values.fix_bloom_effects.GetValue())) {
        return ConvertKind::DepthToColor;
    }
    return std::nullopt;
}

// EDEN_NX2_METAL_LAYERS=0: a render pass has the bound slice of each attachment only, as this
// backend started (a program's layer output is not honoured).
[[nodiscard]] bool LayersEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_LAYERS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] TextureKind KindOf(const ImageInfo& info) {
    switch (info.type) {
    case ImageType::e1D:
        // With the pipeline check a 2D texture a texel high, which the shaders read as one
        // (SpirvToMsl): Metal aborts on a 1D texture with levels and on a 2D view of one.
        return PipelineCheckEnabled() ? TextureKind::e2DArray : TextureKind::e1DArray;
    case ImageType::e3D:
        return TextureKind::e3D;
    case ImageType::e2D:
    case ImageType::Linear:
    default:
        return info.num_samples > 1 ? TextureKind::e2DMultisampleArray : TextureKind::e2DArray;
    }
}

[[nodiscard]] u8 HostSwizzle(u8 guest, const std::array<u8, 4>& format_swizzle) {
    using Tegra::Texture::SwizzleSource;
    switch (static_cast<SwizzleSource>(guest)) {
    case SwizzleSource::Zero:
        return 0;
    case SwizzleSource::OneInt:
    case SwizzleSource::OneFloat:
        return 1;
    case SwizzleSource::R:
    case SwizzleSource::G:
    case SwizzleSource::B:
    case SwizzleSource::A:
        return format_swizzle[guest - static_cast<u8>(SwizzleSource::R)];
    }
    return 0;
}

[[nodiscard]] Offset3D ToOffset(const VideoCommon::Offset3D& offset) {
    return Offset3D{static_cast<u32>(offset.x), static_cast<u32>(offset.y),
                    static_cast<u32>(offset.z)};
}

/// The copy extent clipped to the level (Metal rejects regions past it; block-compressed
/// levels smaller than a block are copied whole).
[[nodiscard]] Extent3D ClipExtent(const ImageInfo& info, u32 samples_x, u32 samples_y, s32 level,
                                  VideoCommon::Offset3D offset, VideoCommon::Extent3D extent) {
    const u32 width = (std::max)((info.size.width >> samples_x) >> level, 1U);
    const u32 height = (std::max)((info.size.height >> samples_y) >> level, 1U);
    const u32 depth =
        info.type == ImageType::e3D ? (std::max)(info.size.depth >> level, 1U) : 1U;
    const auto clip = [](u32 limit, s32 start, u32 size) {
        const u32 begin = static_cast<u32>((std::max)(start, 0));
        return begin >= limit ? 0U : (std::min)(size, limit - begin);
    };
    return Extent3D{clip(width, offset.x, extent.width), clip(height, offset.y, extent.height),
                    clip(depth, offset.z, extent.depth)};
}

// EDEN_NX2_METAL_SPARSE=0: large images over Switch 2 sparse regions are made whole.
[[nodiscard]] bool SparseTilesEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SPARSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// A large single 2D image in a Switch 2 sparse region (title 1's world-sized R8/RG16/R16 masks,
// 170-341 MB, of which the game maps a few MiB) gets memory by sparse tile where the guest memory
// under it can hold data (Vulkan's UsesSparseTiles). Formats of one texel per block.
[[nodiscard]] bool UsesSparseTiles(const Device& device, const ImageInfo& info,
                                   const FormatInfo& host, bool converted, GPUVAddr gpu_addr) {
    if (!SparseTilesEnabled() || converted || host.format == 0 || !SupportsSparseTextures(device) ||
        info.type != ImageType::e2D || info.resources.layers != 1 || info.num_samples != 1 ||
        info.size.depth != 1 || info.block.width != 0 ||
        VideoCore::Surface::DefaultBlockWidth(info.format) != 1 ||
        VideoCore::Surface::DefaultBlockHeight(info.format) != 1 ||
        VideoCore::Surface::GetFormatType(info.format) !=
            VideoCore::Surface::SurfaceType::ColorTexture) {
        return false;
    }
    const u32 size = VideoCommon::CalculateGuestSizeInBytes(info);
    return size >= (32U << 20) && Tegra::Nx2InSparseRegion(gpu_addr, size);
}

/// The sparse heap of an image of `guest_bytes`: an eighth of the image, the most that is
/// mapped for the game's masks (EDEN_NX2_METAL_SPARSE_MIB sets it), within 16 and 256 MiB.
[[nodiscard]] u64 SparseHeapBytes(u32 guest_bytes) {
    static const u64 forced = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SPARSE_MIB");
        return value != nullptr ? u64{std::strtoull(value, nullptr, 10)} << 20 : u64{0};
    }();
    if (forced != 0) {
        return forced;
    }
    return std::clamp<u64>(Common::AlignUp(guest_bytes / 8, u64{1} << 20), u64{16} << 20,
                           u64{256} << 20);
}

} // Anonymous namespace

TextureCacheRuntime::TextureCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                         StagingBufferPool& staging_buffer_pool_)
    : device{device_}, scheduler{scheduler_}, staging_buffer_pool{staging_buffer_pool_},
      sample_limits{DevicePipelineLimits(device_)} {}

TextureCacheRuntime::~TextureCacheRuntime() = default;

void TextureCacheRuntime::Finish() {
    scheduler.Finish();
}

StagingBufferRef TextureCacheRuntime::UploadStagingBuffer(size_t size, bool deferred) {
    return staging_buffer_pool.Request(size, deferred);
}

StagingBufferRef TextureCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_buffer_pool.Request(size, deferred);
}

void TextureCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_buffer_pool.FreeDeferred(ref);
}

u64 TextureCacheRuntime::GetDeviceLocalMemory() const {
    return DeviceMemoryBudget(device);
}

u64 TextureCacheRuntime::GetDeviceMemoryUsage() const {
    return allocated_bytes + staging_buffer_pool.HeldBytes();
}

bool TextureCacheRuntime::ShouldReinterpret(Image& dst, Image& src) const noexcept {
    // Two formats of one Metal format keep their guest bits apart when one of them is swapped.
    return dst.HostFormat().format != src.HostFormat().format ||
           dst.PackedShift() != src.PackedShift() || dst.Pack16() != src.Pack16();
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const VideoCommon::ImageCopy> copies) {
    if (ShouldReinterpret(dst, src)) {
        ReinterpretImage(dst, src, copies);
        return;
    }
    if (dst.Handle() == nullptr || src.Handle() == nullptr ||
        dst.info.num_samples != src.info.num_samples) {
        return;
    }
    const auto [sx, sy] = VideoCommon::SamplesLog2(src.info.num_samples);
    const bool src_3d = src.info.type == ImageType::e3D;
    const bool dst_3d = dst.info.type == ImageType::e3D;
    for (const VideoCommon::ImageCopy& copy : copies) {
        const Extent3D extent = ClipExtent(src.info, sx, sy, copy.src_subresource.base_level,
                                           copy.src_offset, copy.extent);
        if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
            continue;
        }
        const s32 layers = (std::max)(copy.src_subresource.num_layers, 1);
        for (s32 layer = 0; layer < layers; ++layer) {
            scheduler.CopyTexture(
                src.Handle(), static_cast<u32>(copy.src_subresource.base_level),
                src_3d ? 0U : static_cast<u32>(copy.src_subresource.base_layer + layer),
                ToOffset(copy.src_offset), extent, dst.Handle(),
                static_cast<u32>(copy.dst_subresource.base_level),
                dst_3d ? 0U : static_cast<u32>(copy.dst_subresource.base_layer + layer),
                ToOffset(copy.dst_offset));
        }
    }
}

void TextureCacheRuntime::BarrierFeedbackLoop() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FRAGMENT_BARRIER");
        return value == nullptr || value[0] != '0';
    }();
    // The draw samples the bound depth/stencil texture: ending the pass makes what its earlier
    // draws wrote visible. If none wrote it (nor a load-action clear), memory already holds what
    // the attachment has, and Metal allows sampling the attachment while nothing writes it, as
    // Vulkan's (Eglantine bdaf482d3f). EDEN_NX2_METAL_DEPTH_READ_KEEP_PASS=0 ends it every time.
    static const bool keep_pass = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DEPTH_READ_KEEP_PASS");
        return value == nullptr || value[0] != '0';
    }();
    if (enabled && scheduler.IsRendering()) {
        if (keep_pass && !scheduler.RenderPassWroteDepthStencil()) {
            Coverage::NoteGuard("depth feedback loop pass kept");
            return;
        }
        scheduler.EndRendering();
        Coverage::NoteGuard("depth feedback loop pass ends");
    }
}

bool TextureCacheRuntime::CanUploadMSAA() const noexcept {
    return MsaaEnabled();
}

bool TextureCacheRuntime::CanDownloadMsaa(const ImageInfo&) const {
    return MsaaEnabled();
}

void TextureCacheRuntime::CopyImageMSAA(Image& dst, Image& src,
                                        std::span<const VideoCommon::ImageCopy>) {
    // Whole images, as Vulkan's MSAA copy passes: every sample becomes a texel of the
    // single-sampled image, or every texel a sample.
    const bool expand = src.HostSamples() > 1 && dst.HostSamples() == 1;
    const bool compress = src.HostSamples() == 1 && dst.HostSamples() > 1;
    Image& multisampled = expand ? src : dst;
    Image& single = expand ? dst : src;
    if (!MsaaEnabled() || (!expand && !compress) || src.Handle() == nullptr ||
        dst.Handle() == nullptr || ClassOf(src.HostFormat().format) != ClassOf(dst.HostFormat().format) ||
        !dst.HostFormat().attachable) {
        LOG_ONCE(warned, "Metal: a copy of {} x{} into {} x{} is not done", src.info.format,
                 src.info.num_samples, dst.info.format, dst.info.num_samples);
        return;
    }
    const auto [shift_x, shift_y] = VideoCommon::SamplesLog2(multisampled.info.num_samples);
    const u32 layers = static_cast<u32>(
        (std::min)(src.info.resources.layers, dst.info.resources.layers));
    for (u32 layer = 0; layer < layers; ++layer) {
        SampleCopy copy{
            .src = src.Handle(),
            .src_layer = layer,
            .dst = {.texture = dst.Handle(), .slice = layer},
            .dst_format = dst.HostFormat().format,
            .dst_width = (std::max)(dst.info.size.width >> (compress ? shift_x : 0), 1U),
            .dst_height = (std::max)(dst.info.size.height >> (compress ? shift_y : 0), 1U),
            .samples = multisampled.HostSamples(),
            .shift = {static_cast<u32>(shift_x), static_cast<u32>(shift_y)},
        };
        Object view;
        if (compress) {
            view = CreateTextureView(single.Handle(), single.HostFormat().format,
                                     TextureKind::e2D, 0, 1, layer, 1, IdentitySwizzle);
            copy.src = view.Get();
            copy.src_layer = 0;
        }
        if (!scheduler.CopySamples(copy, expand)) {
            LOG_ONCE(warned_copy, "Metal: a sample copy of {} into {} failed", src.info.format,
                     dst.info.format);
            return;
        }
    }
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const VideoCommon::ImageCopy> copies) {
    // Through a buffer: texels keep their bytes, as a copy between formats of one size does.
    const PixelFormat src_format = src.info.format;
    const PixelFormat dst_format = dst.info.format;
    const bool converted = True(src.flags & ImageFlagBits::Converted) ||
                           True(dst.flags & ImageFlagBits::Converted);
    const u32 src_block_width = VideoCore::Surface::DefaultBlockWidth(src_format);
    const u32 src_block_height = VideoCore::Surface::DefaultBlockHeight(src_format);
    const u32 dst_block_width = VideoCore::Surface::DefaultBlockWidth(dst_format);
    const u32 dst_block_height = VideoCore::Surface::DefaultBlockHeight(dst_format);
    const bool src_texels = src_block_width == 1 && src_block_height == 1;
    const bool dst_texels = dst_block_width == 1 && dst_block_height == 1;
    // Blocks keep their bytes as texels do: a texel of the one image is a block of the other.
    const bool blocks_fit =
        (src_texels && dst_texels) ||
        (CopiesEnabled() && (src_texels || dst_texels ||
                             (src_block_width == dst_block_width &&
                              src_block_height == dst_block_height)));
    // Depth and stencil go through the buffer in the guest's bytes too (Image::DepthBytesLayout).
    const auto copyable = [](const Image& image) {
        return IsColor(image.HostFormat().format_class) ||
               (CopiesEnabled() && image.CanCopyWithBuffers());
    };
    if (dst.Handle() == nullptr || src.Handle() == nullptr || converted ||
        src.info.num_samples != 1 || dst.info.num_samples != 1 || !copyable(src) ||
        !copyable(dst) || !blocks_fit ||
        VideoCore::Surface::BytesPerBlock(src_format) !=
            VideoCore::Surface::BytesPerBlock(dst_format)) {
        LOG_ONCE(warned, "Metal: reinterpreting {} as {} is not done yet", src_format,
                 dst_format);
        return;
    }
    const u32 bytes_per_block = VideoCore::Surface::BytesPerBlock(src_format);
    for (const VideoCommon::ImageCopy& copy : copies) {
        // Whole blocks, or up to the edge of the level: Metal aborts on anything else.
        if (copy.src_offset.x % static_cast<s32>(src_block_width) != 0 ||
            copy.src_offset.y % static_cast<s32>(src_block_height) != 0 ||
            copy.dst_offset.x % static_cast<s32>(dst_block_width) != 0 ||
            copy.dst_offset.y % static_cast<s32>(dst_block_height) != 0) {
            LOG_ONCE(warned_offset, "Metal: a copy of {} as {} starts inside a block, skipped",
                     src_format, dst_format);
            continue;
        }
        const Extent3D extent = ClipExtent(
            src.info, 0, 0, copy.src_subresource.base_level, copy.src_offset,
            VideoCommon::Extent3D{Common::AlignUp(copy.extent.width, src_block_width),
                                  Common::AlignUp(copy.extent.height, src_block_height),
                                  copy.extent.depth});
        if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
            continue;
        }
        const u32 blocks_x = Common::DivCeil(extent.width, src_block_width);
        const u32 blocks_y = Common::DivCeil(extent.height, src_block_height);
        const Extent3D dst_extent = ClipExtent(
            dst.info, 0, 0, copy.dst_subresource.base_level, copy.dst_offset,
            VideoCommon::Extent3D{blocks_x * dst_block_width, blocks_y * dst_block_height,
                                  extent.depth});
        if (dst_extent.width == 0 || dst_extent.height == 0 || dst_extent.depth == 0) {
            continue;
        }
        const u32 bytes_per_row = blocks_x * bytes_per_block;
        const u32 bytes_per_image = bytes_per_row * blocks_y;
        const u64 size = u64{bytes_per_image} * extent.depth;
        if (size > reinterpret_buffer_size) {
            reinterpret_buffer_size = std::bit_ceil(size);
            reinterpret_buffer = CreateBuffer(device, reinterpret_buffer_size, false, nullptr);
        }
        const bool src_3d = src.info.type == ImageType::e3D;
        const bool dst_3d = dst.info.type == ImageType::e3D;
        const s32 layers = (std::max)(copy.src_subresource.num_layers, 1);
        const std::optional<DepthLayout> src_depth = src.DepthBytesLayout();
        const std::optional<DepthLayout> dst_depth = dst.DepthBytesLayout();
        for (s32 layer = 0; layer < layers; ++layer) {
            const u32 src_level = static_cast<u32>(copy.src_subresource.base_level);
            const u32 src_slice =
                src_3d ? 0U : static_cast<u32>(copy.src_subresource.base_layer + layer);
            const u32 dst_level = static_cast<u32>(copy.dst_subresource.base_level);
            const u32 dst_slice =
                dst_3d ? 0U : static_cast<u32>(copy.dst_subresource.base_layer + layer);
            if (src_depth) {
                scheduler.DepthToBuffer(src.Handle(), src_level, src_slice,
                                        ToOffset(copy.src_offset), extent,
                                        reinterpret_buffer.Get(), 0, bytes_per_row, *src_depth);
            } else if (src.Pack16() != 0 && !src_3d) {
                scheduler.Pack16(src.Pack16(), true, src.Handle(), src_level, src_slice,
                                 ToOffset(copy.src_offset), extent, reinterpret_buffer.Get(), 0,
                                 bytes_per_row);
            } else {
                scheduler.CopyTextureToBuffer(src.Handle(), src_level, src_slice,
                                              ToOffset(copy.src_offset), extent,
                                              reinterpret_buffer.Get(), 0, bytes_per_row,
                                              src_3d ? bytes_per_image : 0U);
            }
            scheduler.BreakBlits();
            if (const u32 shift = src.PackedShift(); shift != 0) {
                scheduler.SwapPacked(reinterpret_buffer.Get(), 0, blocks_x, blocks_y,
                                     src_3d ? extent.depth : 1U, bytes_per_row,
                                     bytes_per_image, shift);
            }
            if (const u32 shift = dst.PackedShift(); shift != 0) {
                scheduler.SwapPacked(reinterpret_buffer.Get(), 0, blocks_x, blocks_y,
                                     dst_3d ? dst_extent.depth : 1U, bytes_per_row,
                                     bytes_per_image, shift);
            }
            if (dst_depth) {
                scheduler.DepthFromBuffer(reinterpret_buffer.Get(), 0, bytes_per_row,
                                          dst.Handle(), dst_level, dst_slice,
                                          ToOffset(copy.dst_offset), dst_extent, *dst_depth);
            } else if (dst.Pack16() != 0 && !dst_3d) {
                scheduler.Pack16(dst.Pack16(), false, dst.Handle(), dst_level, dst_slice,
                                 ToOffset(copy.dst_offset), dst_extent, reinterpret_buffer.Get(),
                                 0, bytes_per_row);
            } else {
                scheduler.CopyBufferToTexture(reinterpret_buffer.Get(), 0, bytes_per_row,
                                              dst_3d ? bytes_per_image : 0U, dst.Handle(),
                                              dst_level, dst_slice, ToOffset(copy.dst_offset),
                                              dst_extent);
            }
            scheduler.BreakBlits();
        }
    }
}

void TextureCacheRuntime::ConvertImage(Framebuffer* dst, ImageView& dst_view,
                                       ImageView& src_view) {
    if (const std::optional<ConvertKind> kind = ConversionFor(dst_view.format, src_view.format);
        kind && CopiesEnabled()) {
        // Metal aborts on a render pass into a texture made without render target use.
        if (!SurfaceFormat(device, dst_view.format, false).attachable ||
            !scheduler.Convert(ConvertPass{
                .src = src_view.Handle(Shader::TextureType::Color2D),
                .dst = dst_view.RenderTarget(),
                .dst_format = dst_view.HostFormat(),
                .width = dst != nullptr ? dst->Width() : dst_view.size.width,
                .height = dst != nullptr ? dst->Height() : dst_view.size.height,
                .kind = *kind,
            })) {
            LOG_ONCE(warned_convert, "Metal: a conversion of {} into {} was dropped",
                     src_view.format, dst_view.format);
        }
        return;
    }
    const Region2D region{
        .start = {0, 0},
        .end = {static_cast<s32>(dst_view.size.width), static_cast<s32>(dst_view.size.height)},
    };
    BlitImage(dst, dst_view, src_view, region, region, Tegra::Engines::Fermi2D::Filter::Point,
              Tegra::Engines::Fermi2D::Operation::SrcCopy);
}

void TextureCacheRuntime::BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                                    const Region2D& dst_region, const Region2D& src_region,
                                    Tegra::Engines::Fermi2D::Filter filter,
                                    Tegra::Engines::Fermi2D::Operation operation) {
    TextureHandle source = src.Handle(Shader::TextureType::Color2D);
    std::array<s32, 4> src_rect{src_region.start.x, src_region.start.y, src_region.end.x,
                                src_region.end.y};
    if (src.Samples() > 1 && dst.Samples() == 1 && MsaaEnabled()) {
        // A resolve, as Vulkan's vkCmdResolveImage, when both rectangles are the whole of one
        // size: Metal's resolve store action takes the whole render area.
        const bool whole = src_region.start.x == 0 && src_region.start.y == 0 &&
                           dst_region.start.x == 0 && dst_region.start.y == 0 &&
                           src_region.end.x == dst_region.end.x &&
                           src_region.end.y == dst_region.end.y &&
                           dst_region.end.x == static_cast<s32>(dst.size.width) &&
                           dst_region.end.y == static_cast<s32>(dst.size.height);
        // Q27: the resolve its passes made already, while nothing wrote the samples since.
        if (whole && src.HostFormat() == dst.HostFormat()) {
            const Attachment target = dst.RenderTarget();
            if (const TextureHandle shadow = scheduler.ResolveShadow(src.ImageHandle());
                shadow != nullptr && target.depth_plane == 0) {
                scheduler.CopyTexture(shadow, 0, static_cast<u32>(src.range.base.layer), {},
                                      {dst.size.width, dst.size.height, 1}, target.texture,
                                      target.level, target.slice, {});
                return;
            }
        }
        if (whole && src.HostFormat() == dst.HostFormat() &&
            scheduler.Resolve(src.ImageHandle(), static_cast<u32>(src.range.base.layer),
                              dst.RenderTarget(), dst.size.width, dst.size.height)) {
            return;
        }
        // Else the samples as texels, scaled into the rectangle (the filter averages them).
        if (!src.PrepareSampleGrid()) {
            LOG_ONCE(warned_msaa, "Metal: a blit from a multisampled {} image was dropped",
                     src.format);
            return;
        }
        source = src.SampledHandle(Shader::TextureType::Color2D, false);
        const auto [shift_x, shift_y] = VideoCommon::SamplesLog2(src.GuestSamples());
        src_rect = {src_rect[0] << shift_x, src_rect[1] << shift_y, src_rect[2] << shift_x,
                    src_rect[3] << shift_y};
        Coverage::NoteGuard("multisampled blit through its samples");
    }
    const FormatClass dst_class = dst.HostClass();
    const FormatClass src_class = src.HostClass();
    // Depth to depth: the fragment writes the depth it reads, and the stencil between images
    // of one format (Scheduler::Blit). Filtered blits of depth are nearest, as Vulkan's.
    const bool depth = CopiesEnabled() && IsDepth(src_class) && src.Samples() == 1 &&
                       (dst_class == FormatClass::Depth ||
                        (dst_class == FormatClass::DepthStencil &&
                         dst.HostFormat() == src.HostFormat()));
    if (!depth && (!IsColor(dst_class) || !IsColor(src_class))) {
        LOG_ONCE(warned_depth, "Metal: a blit of format {} into {} is not done yet", src.format,
                 dst.format);
        return;
    }
    const BlitPass pass{
        .src = source,
        .src_rect = src_rect,
        .dst = dst.RenderTarget(),
        .dst_format = dst.HostFormat(),
        .dst_rect = {dst_region.start.x, dst_region.start.y, dst_region.end.x, dst_region.end.y},
        .dst_width = dst_framebuffer != nullptr ? dst_framebuffer->Width() : dst.size.width,
        .dst_height = dst_framebuffer != nullptr ? dst_framebuffer->Height() : dst.size.height,
        .linear = !depth && filter == Tegra::Engines::Fermi2D::Filter::Bilinear,
        .keep_dst = true,
    };
    if (!scheduler.Blit(pass)) {
        LOG_ONCE(warned_format, "Metal: a blit into format {} was dropped", dst.format);
    }
}

bool TextureCacheRuntime::WantsTemporalUpscaleDepth() const {
    // FSR 3.1 replaces DLSS; the temporal stand-in only runs when FSR cannot (no depth, a GPU
    // without what its passes need, a failed start). EDEN_NX2_DLSS_REPLACEMENT=standin forces
    // the stand-in for A/B runs, as on Vulkan.
    static const bool force_standin = [] {
        const char* const value = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return value != nullptr && std::string_view{value} == "standin";
    }();
    if (force_standin) {
        return false;
    }
    if (Settings::MetalFxTemporalChosen() && !temporal_unsupported) {
        return true;
    }
    return !fsr3_unsupported;
}

bool TextureCacheRuntime::TemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst,
                                          ImageView& input_color, ImageView& motion_vectors,
                                          ImageView* depth, std::array<u32, 2> render_size,
                                          std::array<f32, 2> jitter,
                                          std::array<f32, 2> motion_scale, bool use_history) {
    if (dst_framebuffer == nullptr) {
        return false;
    }
    if (use_history && depth != nullptr && WantsTemporalUpscaleDepth() &&
        MetalFxTemporalUpscale(dst_framebuffer, dst, input_color, motion_vectors, *depth,
                               render_size, jitter, motion_scale)) {
        return true;
    }
    if (use_history && depth != nullptr && WantsTemporalUpscaleDepth() && !fsr3_unsupported &&
        Fsr3TemporalUpscale(dst_framebuffer, dst, input_color, motion_vectors, *depth,
                            render_size, jitter, motion_scale)) {
        return true;
    }
    return TemporalStandIn(dst_framebuffer, dst, input_color, motion_vectors, render_size, jitter,
                           motion_scale, use_history);
}

bool TextureCacheRuntime::Fsr3TemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst,
                                              ImageView& input_color, ImageView& motion_vectors,
                                              ImageView& depth_view,
                                              std::array<u32, 2> render_size,
                                              std::array<f32, 2> jitter,
                                              std::array<f32, 2> motion_scale) {
    using PixelFormat = VideoCore::Surface::PixelFormat;
    namespace DlssContract = VideoCommon::DlssContract;
    constexpr bool use_history = true;
    ImageView* const depth = &depth_view;
    const auto plain = [](const ImageView& view) {
        return view.range.base.level == 0 && view.range.base.layer == 0;
    };
    const auto fits = [](const ImageView& view, std::array<u32, 2> requested) {
        return DlssContract::FitsExtent(requested, {view.size.width, view.size.height});
    };
    const std::array<u32, 2> output_size{dst_framebuffer->Width(), dst_framebuffer->Height()};
    if (!plain(dst) || !plain(input_color) || !plain(motion_vectors) || !plain(*depth) ||
        dst_framebuffer->Samples() != 1 || !IsColor(dst.HostClass()) ||
        !fits(input_color, render_size) || !fits(dst, output_size) ||
        !fits(motion_vectors, render_size) || !fits(*depth, render_size) ||
        motion_vectors.format != PixelFormat::R16G16_FLOAT ||
        depth->format != PixelFormat::D32_FLOAT || input_color.ImageHandle() == dst.ImageHandle() ||
        !DlssContract::ValidColorFormats(input_color.format, dst.format, use_history) ||
        !DlssContract::ValidGeometry(render_size, output_size, jitter, motion_scale)) {
        return false;
    }
    if (!fsr3) {
        fsr3 = std::make_unique<Fsr3Upscaler>(device, scheduler);
    }
    if (!fsr3->Prepare()) {
        LOG_WARNING(Render_Metal, "FSR 3.1 needs shaders this GPU lacks; using the temporal "
                                  "stand-in");
        fsr3_unsupported = true;
        return false;
    }
    // A gap in the evaluations (a menu, a movie, a load) starts the history again.
    const auto now = std::chrono::steady_clock::now();
    const f32 frame_time_ms =
        fsr3_last_evaluation
            ? std::chrono::duration<f32, std::milli>(now - *fsr3_last_evaluation).count()
            : 1000.0f / 30.0f;
    const bool reset = !fsr3_last_evaluation || frame_time_ms > 250.0f;
    fsr3_last_evaluation = now;
    // FSR's own RCAS, set by the sharpness slider as on Vulkan; EDEN_NX2_TEMPORAL_SHARPEN
    // overrides it.
    static const char* const sharpen_override = std::getenv("EDEN_NX2_TEMPORAL_SHARPEN");
    const int slider = Settings::values.fsr_sharpening_slider.GetValue();
    bool sharpen = slider < 200;
    f32 stops = static_cast<f32>(slider) / 100.0f;
    if (sharpen_override != nullptr) {
        sharpen = std::string_view{sharpen_override} != "off";
        stops = static_cast<f32>(std::strtod(sharpen_override, nullptr));
    }
    const TextureHandle output = fsr3->Upscale({
        .color = input_color.Handle(Shader::TextureType::Color2D),
        .depth = depth->Handle(Shader::TextureType::Color2D),
        .motion = motion_vectors.Handle(Shader::TextureType::Color2D),
        .render_size = render_size,
        .upscale_size = output_size,
        .jitter = jitter,
        .motion_vector_scale = {motion_scale[0] * static_cast<f32>(render_size[0]),
                                motion_scale[1] * static_cast<f32>(render_size[1])},
        .frame_time_ms = frame_time_ms,
        .reset = reset,
        .sharpen = sharpen,
        .sharpness = 1.0f - stops / 2.0f,
    });
    const s32 width = static_cast<s32>(output_size[0]);
    const s32 height = static_cast<s32>(output_size[1]);
    if (output == nullptr ||
        !scheduler.Blit(BlitPass{
            .src = output,
            .src_rect = {0, 0, width, height},
            .dst = dst.RenderTarget(),
            .dst_format = dst.HostFormat(),
            .dst_rect = {0, 0, width, height},
            .dst_width = output_size[0],
            .dst_height = output_size[1],
            .linear = false,
            .keep_dst = true,
        })) {
        LOG_WARNING(Render_Metal, "FSR 3.1 evaluation unavailable; using the temporal stand-in");
        fsr3_unsupported = true;
        return false;
    }
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Metal, "Metal: DLSS replaced by FSR 3.1: {}x{} -> {}x{}", render_size[0],
                 render_size[1], output_size[0], output_size[1]);
    }
    return true;
}

bool TextureCacheRuntime::TemporalStandInEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_TEMPORAL_STANDIN");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

bool TextureCacheRuntime::TemporalStandIn(Framebuffer* dst_framebuffer, ImageView& dst,
                                          ImageView& input_color, ImageView& motion_vectors,
                                          std::array<u32, 2> render_size,
                                          std::array<f32, 2> jitter,
                                          std::array<f32, 2> motion_scale, bool use_history) {
    using PixelFormat = VideoCore::Surface::PixelFormat;
    namespace DlssContract = VideoCommon::DlssContract;
    if (!TemporalStandInEnabled() || standin_failed) {
        return false;
    }
    // Vulkan's checks for its stand-in (TextureCacheRuntime::TemporalUpscale).
    const auto plain = [](const ImageView& view) {
        return view.range.base.level == 0 && view.range.base.layer == 0;
    };
    const auto fits = [](const ImageView& view, std::array<u32, 2> requested) {
        return DlssContract::FitsExtent(requested, {view.size.width, view.size.height});
    };
    const std::array<u32, 2> output_size{dst_framebuffer->Width(), dst_framebuffer->Height()};
    if (!plain(dst) || !plain(input_color) || dst_framebuffer->Samples() != 1 ||
        !IsColor(dst.HostClass()) || !fits(input_color, render_size) || !fits(dst, output_size) ||
        input_color.ImageHandle() == dst.ImageHandle() ||
        !DlssContract::ValidColorFormats(input_color.format, dst.format, use_history) ||
        !DlssContract::ValidGeometry(render_size, output_size, jitter, motion_scale)) {
        return false;
    }
    if (use_history && (!plain(motion_vectors) || !fits(motion_vectors, render_size) ||
                        motion_vectors.format != PixelFormat::R16G16_FLOAT ||
                        input_color.size.width != motion_vectors.size.width ||
                        input_color.size.height != motion_vectors.size.height)) {
        return false;
    }
    if (!present_passes) {
        present_passes = std::make_unique<PresentPasses>(device, scheduler);
    }
    if (!present_passes->TemporalFilter({
            .color = input_color.Handle(Shader::TextureType::Color2D),
            .motion = use_history ? motion_vectors.Handle(Shader::TextureType::Color2D) : nullptr,
            .target = dst.RenderTarget(),
            .format = dst.HostFormat(),
            .size = output_size,
            .jitter = jitter,
            .motion_scale = motion_scale,
            .use_history = use_history,
        })) {
        LOG_WARNING(Render_Metal, "Metal: temporal stand-in unavailable; the game's DLSS runs");
        standin_failed = true;
        return false;
    }
    ++standin_frames;
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Metal, "Metal: DLSS replaced by the temporal stand-in: {}x{} -> {}x{}",
                 render_size[0], render_size[1], output_size[0], output_size[1]);
    }
    return true;
}

bool TextureCacheRuntime::MetalFxTemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst,
                                                 ImageView& input_color,
                                                 ImageView& motion_vectors, ImageView& depth_view,
                                                 std::array<u32, 2> render_size,
                                                 std::array<f32, 2> jitter,
                                                 std::array<f32, 2> motion_scale) {
    using PixelFormat = VideoCore::Surface::PixelFormat;
    namespace DlssContract = VideoCommon::DlssContract;
    constexpr bool use_history = true;
    ImageView* const depth = &depth_view;
    // Images are never rescaled on Metal: each view's size is its texture's.
    const auto plain = [](const ImageView& view) {
        return view.range.base.level == 0 && view.range.base.layer == 0;
    };
    const auto fits = [](const ImageView& view, std::array<u32, 2> requested) {
        return DlssContract::FitsExtent(requested, {view.size.width, view.size.height});
    };
    const std::array<u32, 2> output_size{dst_framebuffer->Width(), dst_framebuffer->Height()};
    if (!plain(dst) || !plain(input_color) || !plain(motion_vectors) || !plain(*depth) ||
        dst_framebuffer->Samples() != 1 || !IsColor(dst.HostClass()) ||
        !fits(input_color, render_size) || !fits(dst, output_size) ||
        !fits(motion_vectors, render_size) || !fits(*depth, render_size) ||
        motion_vectors.format != PixelFormat::R16G16_FLOAT ||
        depth->format != PixelFormat::D32_FLOAT ||
        input_color.ImageHandle() == dst.ImageHandle() ||
        !DlssContract::ValidColorFormats(input_color.format, dst.format, use_history) ||
        !DlssContract::ValidGeometry(render_size, output_size, jitter, motion_scale)) {
        return false;
    }
    if (!temporal) {
        if (!TemporalUpscaler::IsSupported(device)) {
            LOG_WARNING(Render_Metal, "MetalFX temporal: not supported on this GPU");
            temporal_unsupported = true;
            return false;
        }
        temporal = std::make_unique<TemporalUpscaler>();
    }
    // A gap in the evaluations (a menu, a movie, a load) starts the history again.
    const auto now = std::chrono::steady_clock::now();
    const bool reset = !temporal_last_evaluation ||
                       now - *temporal_last_evaluation > std::chrono::milliseconds{250};
    temporal_last_evaluation = now;
    const TextureHandle output = temporal->Upscale(
        device, scheduler,
        {
            .color = input_color.Handle(Shader::TextureType::Color2D),
            .depth = depth->Handle(Shader::TextureType::Color2D),
            .motion = motion_vectors.Handle(Shader::TextureType::Color2D),
            .render_size = render_size,
            .output_size = output_size,
            .jitter = jitter,
            .motion_vector_scale = {motion_scale[0] * static_cast<f32>(render_size[0]),
                                    motion_scale[1] * static_cast<f32>(render_size[1])},
            .reset = reset,
            .depth_reversed = true,
            .auto_exposure = true,
        });
    if (output == nullptr) {
        temporal_unsupported = true;
        return false;
    }
    const s32 width = static_cast<s32>(output_size[0]);
    const s32 height = static_cast<s32>(output_size[1]);
    if (!scheduler.Blit(BlitPass{
            .src = output,
            .src_rect = {0, 0, width, height},
            .dst = dst.RenderTarget(),
            .dst_format = dst.HostFormat(),
            .dst_rect = {0, 0, width, height},
            .dst_width = output_size[0],
            .dst_height = output_size[1],
            .linear = false,
            .keep_dst = true,
        })) {
        temporal_unsupported = true;
        return false;
    }
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Metal, "Metal: DLSS replaced by MetalFX temporal: {}x{} -> {}x{}",
                 render_size[0], render_size[1], output_size[0], output_size[1]);
    }
    return true;
}

bool TextureCacheRuntime::UpscaleWithCameraMotion(Framebuffer* dst_framebuffer,
                                                  ImageView& input_color, ImageView& depth,
                                                  std::array<u32, 2> render_size,
                                                  const std::array<f32, 16>& reprojection,
                                                  std::array<f32, 2> jitter, bool reset) {
    if (dst_framebuffer == nullptr || temporal_unsupported) {
        return false;
    }
    // Images are never rescaled on Metal: each view's size is its texture's. Vulkan's checks
    // (TextureCacheRuntime::UpscaleWithCameraMotion): the output at least the render size, both
    // inputs single-sampled.
    const std::array<u32, 2> output_size{dst_framebuffer->Width(), dst_framebuffer->Height()};
    if (output_size[0] < render_size[0] || output_size[1] < render_size[1] ||
        dst_framebuffer->Samples() != 1 || !IsColor(input_color.HostClass()) ||
        input_color.size.width < render_size[0] || input_color.size.height < render_size[1] ||
        depth.size.width < render_size[0] || depth.size.height < render_size[1]) {
        return false;
    }
    const Attachment target = dst_framebuffer->Colors()[0];
    if (target.texture == nullptr) {
        return false;
    }
    if (!temporal) {
        if (!TemporalUpscaler::IsSupported(device)) {
            LOG_WARNING(Render_Metal, "MetalFX temporal: not supported on this GPU");
            temporal_unsupported = true;
            return false;
        }
        temporal = std::make_unique<TemporalUpscaler>();
    }
    if (!camera_motion_pass) {
        camera_motion_pass = std::make_unique<CameraMotionPass>(device, scheduler);
    }
    if (!camera_motion_pass->Record(depth.Handle(Shader::TextureType::Color2D), render_size,
                                    reprojection)) {
        temporal_unsupported = true;
        return false;
    }
    const TextureHandle output = temporal->Upscale(
        device, scheduler,
        {
            .color = input_color.Handle(Shader::TextureType::Color2D),
            .depth = depth.Handle(Shader::TextureType::Color2D),
            .motion = camera_motion_pass->Motion(),
            .render_size = render_size,
            .output_size = output_size,
            .jitter = jitter,
            // The motion vectors are in render pixels.
            .motion_vector_scale = {1.0f, 1.0f},
            .reset = reset,
            // Ocarina's scene has its depth at 0 at the camera and is tonemapped already.
            .depth_reversed = false,
            .auto_exposure = false,
        });
    if (output == nullptr) {
        temporal_unsupported = true;
        return false;
    }
    const s32 width = static_cast<s32>(output_size[0]);
    const s32 height = static_cast<s32>(output_size[1]);
    if (!scheduler.Blit(BlitPass{
            .src = output,
            .src_rect = {0, 0, width, height},
            .dst = target,
            .dst_format = dst_framebuffer->ColorFormats()[0],
            .dst_rect = {0, 0, width, height},
            .dst_width = output_size[0],
            .dst_height = output_size[1],
            .linear = false,
            .keep_dst = true,
        })) {
        temporal_unsupported = true;
        return false;
    }
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Metal, "Metal: anti-aliasing replaced by MetalFX temporal: {}x{} -> {}x{}",
                 render_size[0], render_size[1], output_size[0], output_size[1]);
    }
    return true;
}

void TextureCacheRuntime::AccelerateImageUpload(Image&, const StagingBufferRef&,
                                                std::span<const VideoCommon::SwizzleParameters>,
                                                u32, u32) {
    // Images never ask for it: CanAccelerateImageUpload is false.
    UNREACHABLE();
}

Image::Image(TextureCacheRuntime& runtime_, const ImageInfo& info_, GPUVAddr gpu_addr_,
             VAddr cpu_addr_)
    : VideoCommon::ImageBase(info_, gpu_addr_, cpu_addr_), runtime{&runtime_} {
    // Texel buffers read the buffer cache's bytes as they are.
    const bool converted =
        info.type != ImageType::Buffer && IsConverted(runtime->device, info.format);
    if (converted) {
        flags |= ImageFlagBits::Converted;
        flags |= ImageFlagBits::CostlyLoad;
    }
    host_format = SurfaceFormat(runtime->device, info.format, converted);
    if (info.type != ImageType::Buffer && !converted) {
        const u32 mode = Pack16Mode(info.format);
        if (mode == 1 || (mode != 0 && PackedAsRgba8(runtime->device))) {
            pack16 = mode;
        } else {
            packed_shift = PackedSwapShift(info.format);
        }
    }
    if (info.type == ImageType::Buffer) {
        // Texel buffers are views of the buffer cache's buffers (milestone 3).
        return;
    }
    if (host_format.format == 0) {
        LOG_ONCE(warned, "Metal: format {} has no Metal counterpart yet", info.format);
        return;
    }
    kind = KindOf(info);
    const auto [samples_x, samples_y] = VideoCommon::SamplesLog2(info.num_samples);
    // The pixel grid stays the guest's; only the samples per pixel drop to what the device has.
    u32 samples = info.num_samples;
    if (PipelineCheckEnabled()) {
        samples = SupportedSamples(samples, runtime->sample_limits);
        if (samples != info.num_samples) {
            Coverage::NoteGuardWith("fewer samples per pixel", [&] {
                return fmt::format("{} samples drawn at {}, the GPU's most", info.num_samples,
                                   samples);
            });
        }
    }
    if (samples == 1 && kind == TextureKind::e2DMultisampleArray) {
        kind = TextureKind::e2DArray;
    }
    desc = TextureDesc{
        .kind = kind,
        .format = host_format.format,
        .width = info.size.width >> samples_x,
        .height = info.type == ImageType::e1D ? 1U : info.size.height >> samples_y,
        .depth = info.type == ImageType::e3D ? info.size.depth : 1U,
        .levels = static_cast<u32>(info.resources.levels),
        .layers = info.type == ImageType::e3D ? 1U : static_cast<u32>(info.resources.layers),
        .samples = samples,
        .render_target = host_format.attachable,
        // sRGB images take writes through a linear view (Vulkan's mutable format).
        .shader_write = (host_format.storage ||
                         (FormatsExpanded() && LinearTwin(host_format.format) != 0)) &&
                        samples == 1,
    };
    if (UsesSparseTiles(runtime->device, info, host_format, converted, gpu_addr_)) {
        if (std::optional<SparseTexture> made = CreateSparseTexture(
                runtime->device, desc, SparseHeapBytes(VideoCommon::CalculateGuestSizeInBytes(info)))) {
            texture = std::move(made->texture);
            sparse = SparseTiles{
                .heap = std::move(made->heap),
                .tile_width = made->tile_width,
                .tile_height = made->tile_height,
                .tile_bytes = made->tile_bytes,
                .tail_level = made->tail_level,
                .tail_tiles = made->tail_tiles,
                .heap_bytes = made->heap_bytes,
            };
            for (u32 level = 0; level < sparse.tail_level; ++level) {
                const u32 width = (std::max)(desc.width >> level, 1U);
                const u32 height = (std::max)(desc.height >> level, 1U);
                sparse.mapped.emplace_back(size_t{Common::DivCeil(width, sparse.tile_width)} *
                                               Common::DivCeil(height, sparse.tile_height),
                                           false);
            }
            flags |= ImageFlagBits::Unbacked | ImageFlagBits::SparseTiles;
            bytes = sparse.heap_bytes;
            runtime->allocated_bytes += bytes;
            LOG_INFO(Render_Metal,
                     "Metal: sparse tiled image {:#x}: {}x{} {} levels, tile {}x{}, mip tail from "
                     "{}, heap {} MiB",
                     gpu_addr_, desc.width, desc.height, desc.levels, sparse.tile_width,
                     sparse.tile_height, sparse.tail_level, sparse.heap_bytes >> 20);
            return;
        }
    }
    texture = CreateTexture(runtime->device, desc);
    if (!texture) {
        LOG_ERROR(Render_Metal, "Metal: texture {}x{}x{} format {} levels {} layers {} failed",
                  desc.width, desc.height, desc.depth, info.format, desc.levels, desc.layers);
        return;
    }
    bytes = AllocatedSize(texture.Get());
    // Q27: a multisampled colour image gets a single-sampled shadow its passes resolve into, which
    // BlitImage's resolves consume (EDEN_NX2_METAL_MSAA_SHADOW=0: none).
    static const bool shadows_enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MSAA_SHADOW");
        return value == nullptr || value[0] != '0';
    }();
    if (shadows_enabled && desc.samples > 1 && info.type == ImageType::e2D &&
        host_format.attachable && host_format.format_class == FormatClass::Float &&
        MsaaEnabled()) {
        resolve_shadow = CreateTexture(runtime->device, TextureDesc{.kind = TextureKind::e2DArray,
                                                                    .format = desc.format,
                                                                    .width = desc.width,
                                                                    .height = desc.height,
                                                                    .layers = desc.layers,
                                                                    .render_target = true});
        if (resolve_shadow) {
            bytes += AllocatedSize(resolve_shadow.Get());
            runtime->scheduler.SetResolveShadow(texture.Get(), resolve_shadow.Get());
        }
    }
    runtime->allocated_bytes += bytes;
}

Image::Image(const VideoCommon::NullImageParams& params) : VideoCommon::ImageBase{params} {}

Image::~Image() {
    if (resolve_shadow && runtime != nullptr) {
        runtime->scheduler.SetResolveShadow(texture.Get(), nullptr);
    }
    if (texture && runtime != nullptr) {
        runtime->allocated_bytes -= bytes + scaled_bytes;
    }
}

bool TextureCacheRuntime::CanRescaleImage(const ImageInfo& info) const {
    constexpr u32 MaxTextureSize = 16384;
    const auto& resolution = Settings::values.resolution_info;
    const FormatInfo host = SurfaceFormat(device, info.format, IsConverted(device, info.format));
    // Multisampled colour and depth images are scaled sample for sample (EDEN_NX2_METAL_MSAA),
    // stencil-only ones with the stencil exported (EDEN_NX2_METAL_COPIES); a multisampled
    // depth-stencil one is not (Metal writes no stencil per sample from a fragment).
    const bool samples_fit =
        info.num_samples == 1 ||
        (MsaaEnabled() && host.format_class != FormatClass::Stencil &&
         host.format_class != FormatClass::DepthStencil);
    const bool stencil_fits =
        host.format_class != FormatClass::Stencil || (CopiesEnabled() && info.num_samples == 1);
    return info.type == ImageType::e2D && samples_fit && stencil_fits && host.format != 0 &&
           host.attachable && resolution.ScaleUp(info.size.width) <= MaxTextureSize &&
           resolution.ScaleUp(info.size.height) <= MaxTextureSize;
}

bool Image::ScaleUp(bool ignore) {
    const auto& resolution = Settings::values.resolution_info;
    if (!resolution.active || rescaled || !texture || runtime == nullptr) {
        return false;
    }
    if (!scaled_texture) {
        TextureDesc scaled = desc;
        scaled.width = resolution.ScaleUp(desc.width);
        if (info.type == ImageType::e2D) {
            scaled.height = resolution.ScaleUp(desc.height);
        }
        scaled_texture = CreateTexture(runtime->device, scaled);
        if (!scaled_texture) {
            LOG_ONCE(warned_scaled, "Metal: a scaled {}x{} texture failed; images stay native",
                     scaled.width, scaled.height);
            return false;
        }
        scaled_bytes = AllocatedSize(scaled_texture.Get());
        runtime->allocated_bytes += scaled_bytes;
        has_scaled = true;
        ignore = false;
    }
    rescaled = true;
    flags |= ImageFlagBits::Rescaled;
    if (!ignore && !CopyScaled(true)) {
        rescaled = false;
        flags &= ~ImageFlagBits::Rescaled;
        return false;
    }
    return true;
}

bool Image::ScaleDown(bool ignore) {
    if (!Settings::values.resolution_info.active || !rescaled) {
        return false;
    }
    if (!ignore && !CopyScaled(false)) {
        return false;
    }
    rescaled = false;
    flags &= ~ImageFlagBits::Rescaled;
    return true;
}

bool Image::CopyScaled(bool scale_up) {
    const TextureHandle src = scale_up ? texture.Get() : scaled_texture.Get();
    const TextureHandle dst = scale_up ? scaled_texture.Get() : texture.Get();
    if (src == nullptr || dst == nullptr || kind == TextureKind::e3D || !host_format.attachable) {
        return false;
    }
    const auto& resolution = Settings::values.resolution_info;
    const bool is_2d = info.type == ImageType::e2D;
    // Multisampled and stencil-only images: what Blit can't draw, sample for sample or the
    // stencil exported, nearest (CanRescaleImage lets them through).
    const bool multisampled = desc.samples > 1;
    if (multisampled || host_format.format_class == FormatClass::Stencil) {
        for (u32 level = 0; level < desc.levels; ++level) {
            const u32 width = (std::max)(desc.width >> level, 1U);
            const u32 height = (std::max)(desc.height >> level, 1U);
            const u32 scaled_width = (std::max)(resolution.ScaleUp(desc.width) >> level, 1U);
            const u32 scaled_height = (std::max)(resolution.ScaleUp(desc.height) >> level, 1U);
            // The shader reads the level as level 0 of a view (stencil images have levels).
            Object view;
            TextureHandle source = src;
            if (!multisampled) {
                view = CreateTextureView(src, host_format.format, TextureKind::e2DArray, level, 1,
                                         0, desc.layers, IdentitySwizzle);
                source = view.Get();
            }
            for (u32 layer = 0; layer < desc.layers; ++layer) {
                if (!runtime->scheduler.ScaleNearest(ScalePass{
                        .src = source,
                        .src_layer = layer,
                        .src_width = scale_up ? width : scaled_width,
                        .src_height = scale_up ? height : scaled_height,
                        .dst = {.texture = dst, .level = level, .slice = layer},
                        .dst_format = host_format.format,
                        .dst_width = scale_up ? scaled_width : width,
                        .dst_height = scale_up ? scaled_height : height,
                    })) {
                    LOG_ONCE(warned_scale, "Metal: rescaling a {} x{} image failed", info.format,
                             desc.samples);
                    return false;
                }
            }
        }
        return true;
    }
    const bool depth = host_format.format_class == FormatClass::Depth ||
                       host_format.format_class == FormatClass::DepthStencil;
    const bool integer = host_format.format_class == FormatClass::Uint ||
                         host_format.format_class == FormatClass::Sint;
    for (u32 level = 0; level < desc.levels; ++level) {
        const u32 width = (std::max)(desc.width >> level, 1U);
        const u32 height = (std::max)(desc.height >> level, 1U);
        const u32 scaled_width = (std::max)(resolution.ScaleUp(desc.width) >> level, 1U);
        const u32 scaled_height =
            is_2d ? (std::max)(resolution.ScaleUp(desc.height) >> level, 1U) : height;
        const u32 src_width = scale_up ? width : scaled_width;
        const u32 src_height = scale_up ? height : scaled_height;
        const u32 dst_width = scale_up ? scaled_width : width;
        const u32 dst_height = scale_up ? scaled_height : height;
        for (u32 layer = 0; layer < desc.layers; ++layer) {
            // The blit reads a 2D texture: a view of the level and layer unless that is all.
            Object view;
            TextureHandle source = src;
            if (desc.levels != 1 || desc.layers != 1 || kind != TextureKind::e2D) {
                view = CreateTextureView(src, host_format.format, TextureKind::e2D, level, 1,
                                         layer, 1, IdentitySwizzle);
                source = view.Get();
            }
            const bool copied = runtime->scheduler.Blit(BlitPass{
                .src = source,
                .src_rect = {0, 0, static_cast<s32>(src_width), static_cast<s32>(src_height)},
                .dst = {.texture = dst, .level = level, .slice = layer},
                .dst_format = host_format.format,
                .dst_rect = {0, 0, static_cast<s32>(dst_width), static_cast<s32>(dst_height)},
                .dst_width = dst_width,
                .dst_height = dst_height,
                // As Vulkan's BlitScale: colour filtered, depth and integers nearest.
                .linear = !depth && !integer,
                .keep_dst = false,
            });
            if (!copied) {
                LOG_ONCE(warned_copy, "Metal: rescaling an image of format {} failed",
                         info.format);
                return false;
            }
        }
    }
    return true;
}

std::pair<u32, u32> Image::CopyPitches(const VideoCommon::BufferImageCopy& copy) const {
    const bool converted = True(flags & ImageFlagBits::Converted);
    const u32 block_width = converted ? 1U : VideoCore::Surface::DefaultBlockWidth(info.format);
    const u32 block_height = converted ? 1U : VideoCore::Surface::DefaultBlockHeight(info.format);
    const u32 bytes_per_block = converted ? VideoCommon::ConvertedBytesPerBlock(info.format)
                                          : VideoCore::Surface::BytesPerBlock(info.format);
    const u32 row_texels =
        copy.buffer_row_length != 0 ? copy.buffer_row_length : copy.image_extent.width;
    const u32 image_rows =
        copy.buffer_image_height != 0 ? copy.buffer_image_height : copy.image_extent.height;
    const u32 bytes_per_row = Common::DivCeil(row_texels, block_width) * bytes_per_block;
    return {bytes_per_row, Common::DivCeil(image_rows, block_height) * bytes_per_row};
}

TextureHandle Image::SampleGrid() {
    if (!sample_grid && texture && runtime != nullptr) {
        TextureDesc grid = desc;
        grid.kind = TextureKind::e2DArray;
        grid.width = info.size.width;
        grid.height = info.size.height;
        grid.levels = 1;
        grid.samples = 1;
        grid.render_target = true;
        grid.shader_write = false;
        sample_grid = CreateTexture(runtime->device, grid);
        if (sample_grid) {
            bytes += AllocatedSize(sample_grid.Get());
            runtime->allocated_bytes += AllocatedSize(sample_grid.Get());
        }
    }
    return sample_grid.Get();
}

bool Image::CopySampleGrid(bool expand) {
    const TextureHandle grid = SampleGrid();
    if (grid == nullptr || desc.samples < 2) {
        return false;
    }
    // The grid is the native size: a rescaled image's drawing comes down to the native texture.
    if (expand && rescaled && !CopyScaled(false)) {
        return false;
    }
    const auto [shift_x, shift_y] = VideoCommon::SamplesLog2(info.num_samples);
    for (u32 layer = 0; layer < desc.layers; ++layer) {
        SampleCopy copy{
            .src = texture.Get(),
            .src_layer = layer,
            .dst = {.texture = grid, .slice = layer},
            .dst_format = host_format.format,
            .dst_width = info.size.width,
            .dst_height = info.size.height,
            .samples = desc.samples,
            .shift = {static_cast<u32>(shift_x), static_cast<u32>(shift_y)},
        };
        Object view;
        if (!expand) {
            view = CreateTextureView(grid, host_format.format, TextureKind::e2D, 0, 1, layer, 1,
                                     IdentitySwizzle);
            copy.src = view.Get();
            copy.src_layer = 0;
            copy.dst = {.texture = texture.Get(), .slice = layer};
            copy.dst_width = desc.width;
            copy.dst_height = desc.height;
        }
        if (!runtime->scheduler.CopySamples(copy, expand)) {
            LOG_ONCE(warned, "Metal: samples of a multisampled {} image not copied", info.format);
            return false;
        }
    }
    return true;
}

bool Image::CanCopyWithBuffers() const {
    if (!texture || (info.num_samples != 1 && !MsaaEnabled())) {
        return false;
    }
    if (desc.samples > 1 && (host_format.format_class == FormatClass::Stencil ||
                             host_format.format_class == FormatClass::DepthStencil)) {
        LOG_ONCE(warned_stencil, "Metal: multisampled stencil ({}) is not uploaded or downloaded",
                 info.format);
        return false;
    }
    switch (host_format.format_class) {
    case FormatClass::Depth:
    case FormatClass::Stencil:
    case FormatClass::DepthStencil:
        // Depth32Float and Depth16Unorm hold the guest's bytes; the rest go through
        // Scheduler::DepthFromBuffer / DepthToBuffer (or are stencil bytes).
        if (info.format == PixelFormat::D32_FLOAT || info.format == PixelFormat::D16_UNORM ||
            CopiesEnabled()) {
            return true;
        }
        LOG_ONCE(warned, "Metal: depth and stencil uploads and downloads ({}) are skipped",
                 info.format);
        return false;
    default:
        return true;
    }
}

std::optional<DepthLayout> Image::DepthBytesLayout() const {
    switch (info.format) {
    case PixelFormat::X8_D24_UNORM:
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
        return DepthLayout::Packed24_8;
    case PixelFormat::D32_FLOAT_S8_UINT:
        return DepthLayout::Float32_X24S8;
    default:
        return std::nullopt;
    }
}

bool Image::BindTiles(std::span<const VideoCommon::LevelRect> rects) {
    if (!texture || sparse.tile_width == 0) {
        return false;
    }
    std::vector<SparseTileOp> ops;
    std::vector<std::pair<u32, size_t>> fresh; // Tiles this call maps: level, index
    bool needs_tail = false;
    u64 added = 0;
    for (const VideoCommon::LevelRect& rect : rects) {
        const u32 level = static_cast<u32>(rect.level);
        if (level >= sparse.tail_level) {
            needs_tail = true;
            continue;
        }
        if (rect.width == 0 || rect.height == 0) {
            continue;
        }
        const u32 width = (std::max)(desc.width >> level, 1U);
        const u32 tiles_x = Common::DivCeil(width, sparse.tile_width);
        for (u32 ty = rect.y / sparse.tile_height;
             ty <= (rect.y + rect.height - 1) / sparse.tile_height; ++ty) {
            for (u32 tx = rect.x / sparse.tile_width;
                 tx <= (rect.x + rect.width - 1) / sparse.tile_width; ++tx) {
                const size_t index = size_t{ty} * tiles_x + tx;
                if (sparse.mapped[level][index]) {
                    continue;
                }
                sparse.mapped[level][index] = true; // Not mapped twice within this call
                fresh.emplace_back(level, index);
                ops.push_back({.x = tx, .y = ty, .width = 1, .height = 1, .level = level});
                added += sparse.tile_bytes;
            }
        }
    }
    const bool tail = needs_tail && sparse.tail_level < desc.levels && !sparse.tail_mapped;
    if (tail) {
        ops.push_back({.x = 0, .y = 0, .width = sparse.tail_tiles, .height = 1,
                       .level = sparse.tail_level});
        added += u64{sparse.tail_tiles} * sparse.tile_bytes;
    }
    // A failed bind leaves the tiles as they were, so the next call tries them again.
    const auto roll_back = [&] {
        for (const auto& [level, index] : fresh) {
            sparse.mapped[level][index] = false;
        }
    };
    if (sparse.bound_bytes + added > sparse.heap_bytes) {
        LOG_WARNING(Render_Metal, "Metal: sparse tiled image {:#x}: its {} MiB heap is full",
                    gpu_addr, sparse.heap_bytes >> 20);
        roll_back();
        return false;
    }
    if (ops.empty()) {
        return true;
    }
    if (!runtime->scheduler.UpdateSparseTiles(texture.Get(), ops)) {
        roll_back();
        return false;
    }
    sparse.bound_bytes += added;
    sparse.tail_mapped = sparse.tail_mapped || tail;
    return true;
}

void Image::UploadMemory(BufferHandle buffer, size_t offset,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    if (!CanCopyWithBuffers()) {
        return;
    }
    // A multisampled image takes its samples as texels of the grid (the guest's layout).
    const bool multisampled = desc.samples > 1;
    const TextureHandle target = multisampled ? SampleGrid() : texture.Get();
    if (target == nullptr) {
        return;
    }
    const std::optional<DepthLayout> depth_layout = DepthBytesLayout();
    const bool is_3d = info.type == ImageType::e3D;
    for (const VideoCommon::BufferImageCopy& copy : copies) {
        if (copy.image_subresource.base_level >= info.resources.levels) {
            continue;
        }
        const Extent3D extent = ClipExtent(info, 0, 0, copy.image_subresource.base_level,
                                           copy.image_offset, copy.image_extent);
        if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
            continue;
        }
        const auto [bytes_per_row, bytes_per_image] = CopyPitches(copy);
        const s32 layers = is_3d ? 1 : (std::max)(copy.image_subresource.num_layers, 1);
        for (s32 layer = 0; layer < layers; ++layer) {
            if (pack16 != 0 && !is_3d && !multisampled) {
                if (!runtime->scheduler.Pack16(
                        pack16, false, target, static_cast<u32>(copy.image_subresource.base_level),
                        static_cast<u32>(copy.image_subresource.base_layer + layer),
                        ToOffset(copy.image_offset), extent, buffer,
                        offset + copy.buffer_offset + u64{bytes_per_image} * layer,
                        bytes_per_row)) {
                    LOG_ONCE(warned_pack, "Metal: an upload of {} was dropped", info.format);
                }
                continue;
            }
            if (depth_layout) {
                if (!runtime->scheduler.DepthFromBuffer(
                        buffer, offset + copy.buffer_offset + u64{bytes_per_image} * layer,
                        bytes_per_row, target, static_cast<u32>(copy.image_subresource.base_level),
                        static_cast<u32>(copy.image_subresource.base_layer + layer),
                        ToOffset(copy.image_offset), extent, *depth_layout)) {
                    LOG_ONCE(warned_depth, "Metal: an upload of {} was dropped", info.format);
                }
                continue;
            }
            runtime->scheduler.CopyBufferToTexture(
                buffer, offset + copy.buffer_offset + u64{bytes_per_image} * layer, bytes_per_row,
                is_3d ? bytes_per_image : 0U, target,
                static_cast<u32>(copy.image_subresource.base_level),
                is_3d ? 0U : static_cast<u32>(copy.image_subresource.base_layer + layer),
                ToOffset(copy.image_offset), extent);
        }
    }
    if (multisampled) {
        runtime->scheduler.BreakBlits();
        CopySampleGrid(false);
    }
    // Uploads go to the native texture; a rescaled image draws from the scaled one.
    if (rescaled) {
        CopyScaled(true);
    }
}

template <typename F>
void Image::ForEachCopyRegion(std::span<const VideoCommon::BufferImageCopy> copies,
                              F&& region) const {
    const bool is_3d = info.type == ImageType::e3D;
    for (const VideoCommon::BufferImageCopy& copy : copies) {
        if (copy.image_subresource.base_level >= info.resources.levels) {
            continue;
        }
        const Extent3D extent = ClipExtent(info, 0, 0, copy.image_subresource.base_level,
                                           copy.image_offset, copy.image_extent);
        if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
            continue;
        }
        const auto [bytes_per_row, bytes_per_image] = CopyPitches(copy);
        const s32 layers = is_3d ? 1 : (std::max)(copy.image_subresource.num_layers, 1);
        for (s32 layer = 0; layer < layers; ++layer) {
            region(copy.buffer_offset + u64{bytes_per_image} * layer, extent.width, extent.height,
                   is_3d ? extent.depth : 1U, bytes_per_row, bytes_per_image);
        }
    }
}

void Image::UploadMemory(const StagingBufferRef& map,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    if (packed_shift != 0 && CanCopyWithBuffers()) {
        // Guest bytes into Metal's order, on the CPU: the staging buffer is this upload's own.
        ForEachCopyRegion(copies, [&](u64 offset, u32 width, u32 rows, u32 images, u32 row_bytes,
                                      u32 image_bytes) {
            const u64 end = offset + u64{images - 1} * image_bytes + u64{rows - 1} * row_bytes +
                            u64{width} * 2;
            if (end <= map.mapped_span.size()) {
                SwapPackedTexels(map.mapped_span.data() + offset, width, rows, images, row_bytes,
                                 image_bytes, packed_shift);
            }
        });
    }
    UploadMemory(map.buffer, map.offset, copies);
}

void Image::DownloadMemory(BufferHandle buffer, size_t offset,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    std::array buffers{buffer};
    std::array offsets{offset};
    DownloadMemory(buffers, offsets, copies);
}

void Image::DownloadMemory(std::span<BufferHandle> buffers, std::span<size_t> offsets,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    if (!CanCopyWithBuffers()) {
        return;
    }
    // Downloads read the native texture: bring a rescaled image's drawing to it first.
    if (rescaled) {
        CopyScaled(false);
    }
    const bool multisampled = desc.samples > 1;
    if (multisampled && !CopySampleGrid(true)) {
        return;
    }
    const TextureHandle source = multisampled ? sample_grid.Get() : texture.Get();
    const std::optional<DepthLayout> depth_layout = DepthBytesLayout();
    const bool is_3d = info.type == ImageType::e3D;
    for (size_t index = 0; index < buffers.size(); ++index) {
        for (const VideoCommon::BufferImageCopy& copy : copies) {
            if (copy.image_subresource.base_level >= info.resources.levels) {
                continue;
            }
            const Extent3D extent = ClipExtent(info, 0, 0, copy.image_subresource.base_level,
                                               copy.image_offset, copy.image_extent);
            if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
                continue;
            }
            const auto [bytes_per_row, bytes_per_image] = CopyPitches(copy);
            const s32 layers = is_3d ? 1 : (std::max)(copy.image_subresource.num_layers, 1);
            for (s32 layer = 0; layer < layers; ++layer) {
                if (pack16 != 0 && !is_3d && !multisampled) {
                    if (!runtime->scheduler.Pack16(
                            pack16, true, source, static_cast<u32>(copy.image_subresource.base_level),
                            static_cast<u32>(copy.image_subresource.base_layer + layer),
                            ToOffset(copy.image_offset), extent, buffers[index],
                            offsets[index] + copy.buffer_offset + u64{bytes_per_image} * layer,
                            bytes_per_row)) {
                        LOG_ONCE(warned_pack, "Metal: a download of {} was dropped", info.format);
                    }
                    continue;
                }
                if (depth_layout) {
                    if (!runtime->scheduler.DepthToBuffer(
                            source, static_cast<u32>(copy.image_subresource.base_level),
                            static_cast<u32>(copy.image_subresource.base_layer + layer),
                            ToOffset(copy.image_offset), extent, buffers[index],
                            offsets[index] + copy.buffer_offset + u64{bytes_per_image} * layer,
                            bytes_per_row, *depth_layout)) {
                        LOG_ONCE(warned_depth, "Metal: a download of {} was dropped",
                                 info.format);
                    }
                    continue;
                }
                runtime->scheduler.CopyTextureToBuffer(
                    source, static_cast<u32>(copy.image_subresource.base_level),
                    is_3d ? 0U : static_cast<u32>(copy.image_subresource.base_layer + layer),
                    ToOffset(copy.image_offset), extent, buffers[index],
                    offsets[index] + copy.buffer_offset + u64{bytes_per_image} * layer,
                    bytes_per_row, is_3d ? bytes_per_image : 0U);
            }
        }
        if (packed_shift != 0) {
            // Metal's order into the guest's, in the buffer, after the copies.
            ForEachCopyRegion(copies, [&](u64 offset, u32 width, u32 rows, u32 images,
                                          u32 row_bytes, u32 image_bytes) {
                if (!runtime->scheduler.SwapPacked(buffers[index], offsets[index] + offset, width,
                                                   rows, images, row_bytes, image_bytes,
                                                   packed_shift)) {
                    LOG_ONCE(warned_swap, "Metal: a download of {} keeps Metal's channel order",
                             info.format);
                }
            });
        }
    }
}

void Image::DownloadMemory(const StagingBufferRef& map,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    DownloadMemory(map.buffer, map.offset, copies);
}

ImageView::ImageView(TextureCacheRuntime& runtime, const ImageViewInfo& info, ImageId image_id_,
                     Image& image, const SlotVector<Image>& slot_images_)
    : VideoCommon::ImageViewBase{info, image.info, image_id_, image.gpu_addr},
      // The texture cache owns the images; views reach theirs for the sample grid.
      slot_images{const_cast<SlotVector<Image>*>(&slot_images_)}, image_texture{image.Handle()}, samples{image.HostSamples()},
      guest_samples{image.info.num_samples}, image_kind{image.Kind()} {
    const bool converted = True(image.flags & ImageFlagBits::Converted);
    const FormatInfo host = SurfaceFormat(runtime.device, format, converted);
    host_format = host.format;
    is_render_target = info.IsRenderTarget();
    is_3d_slice = True(flags & ImageViewFlagBits::Slice);
    if (!is_render_target) {
        swizzle = {HostSwizzle(info.x_source, host.swizzle), HostSwizzle(info.y_source, host.swizzle),
                   HostSwizzle(info.z_source, host.swizzle), HostSwizzle(info.w_source, host.swizzle)};
    }
    // A render target in another format than its image draws through a view of that format.
    if (image_texture != nullptr && host_format != image.HostFormat().format && !is_3d_slice &&
        image_kind != TextureKind::e3D) {
        render_view = CreateTextureView(image_texture, host_format, image_kind, 0,
                                        static_cast<u32>(image.info.resources.levels), 0,
                                        static_cast<u32>(image.info.resources.layers),
                                        IdentitySwizzle);
    }
}

ImageView::ImageView(TextureCacheRuntime&, const ImageInfo& info, const ImageViewInfo& view_info,
                     GPUVAddr gpu_addr_)
    : VideoCommon::ImageViewBase{info, view_info, gpu_addr_},
      buffer_size{static_cast<u32>(VideoCommon::CalculateGuestSizeInBytes(info))} {}

ImageView::ImageView(TextureCacheRuntime&, const ImageInfo& info, const ImageViewInfo& view_info)
    : VideoCommon::ImageViewBase{info, view_info, 0} {}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams& params)
    : VideoCommon::ImageViewBase{params} {}

ImageView::~ImageView() = default;

TextureHandle ImageView::Handle(Shader::TextureType texture_type) {
    if (image_texture == nullptr || texture_type == Shader::TextureType::Buffer) {
        return nullptr;
    }
    Object& view = views[static_cast<size_t>(texture_type)];
    if (view) {
        return view.Get();
    }
    const bool multisample = image_kind == TextureKind::e2DMultisampleArray;
    TextureKind view_kind = TextureKind::e2D;
    u32 base_layer = static_cast<u32>(range.base.layer);
    u32 layers = static_cast<u32>(range.extent.layers);
    switch (texture_type) {
    case Shader::TextureType::Color1D:
        view_kind = PipelineCheckEnabled() ? TextureKind::e2D : TextureKind::e1D;
        layers = 1;
        break;
    case Shader::TextureType::ColorArray1D:
        view_kind = PipelineCheckEnabled() ? TextureKind::e2DArray : TextureKind::e1DArray;
        break;
    case Shader::TextureType::Color2D:
    case Shader::TextureType::Color2DRect:
        view_kind = multisample ? TextureKind::e2DMultisample : TextureKind::e2D;
        layers = 1;
        break;
    case Shader::TextureType::ColorArray2D:
        view_kind = multisample ? TextureKind::e2DMultisampleArray : TextureKind::e2DArray;
        break;
    case Shader::TextureType::Color3D:
        view_kind = TextureKind::e3D;
        base_layer = 0;
        layers = 1;
        break;
    case Shader::TextureType::ColorCube:
        view_kind = TextureKind::Cube;
        layers = 6;
        break;
    case Shader::TextureType::ColorArrayCube:
        view_kind = TextureKind::CubeArray;
        layers = (std::max)(layers / 6, 1U) * 6;
        break;
    default:
        return nullptr;
    }
    // 3D images only have 3D views (2D slices of them are render targets, RenderTarget).
    if ((image_kind == TextureKind::e3D) != (view_kind == TextureKind::e3D)) {
        view_kind = image_kind == TextureKind::e3D ? TextureKind::e3D : TextureKind::e2D;
        base_layer = image_kind == TextureKind::e3D ? 0 : base_layer;
        layers = 1;
    }
    // Nor does Metal view a multisampled texture as anything else (it aborts).
    if (multisample && PipelineCheckEnabled() && view_kind != TextureKind::e2DMultisample &&
        view_kind != TextureKind::e2DMultisampleArray) {
        view_kind = TextureKind::e2DMultisample;
        layers = 1;
    }
    view = CreateTextureView(image_texture, host_format, view_kind,
                             static_cast<u32>(range.base.level),
                             static_cast<u32>(range.extent.levels), base_layer, layers, swizzle);
    return view.Get();
}

TextureHandle ImageView::StorageHandle(Shader::TextureType texture_type,
                                       Shader::ImageFormat image_format) {
    if (!FormatsExpanded()) {
        return Handle(texture_type);
    }
    if (image_texture == nullptr || texture_type == Shader::TextureType::Buffer) {
        return nullptr;
    }
    u32 view_format = StorageFormat(image_format);
    if (view_format == 0) {
        const u32 linear = LinearTwin(host_format);
        view_format = linear != 0 ? linear : host_format;
    }
    if (view_format == host_format && swizzle == IdentitySwizzle) {
        return Handle(texture_type);
    }
    for (const StorageVariant& existing : storage_views) {
        if (existing.type == texture_type && existing.format == view_format) {
            return existing.view ? existing.view.Get() : Handle(texture_type);
        }
    }
    const bool multisample = image_kind == TextureKind::e2DMultisampleArray;
    TextureKind view_kind = TextureKind::e2D;
    u32 base_layer = static_cast<u32>(range.base.layer);
    u32 layers = static_cast<u32>(range.extent.layers);
    switch (texture_type) {
    case Shader::TextureType::Color1D:
        view_kind = PipelineCheckEnabled() ? TextureKind::e2D : TextureKind::e1D;
        layers = 1;
        break;
    case Shader::TextureType::ColorArray1D:
        view_kind = PipelineCheckEnabled() ? TextureKind::e2DArray : TextureKind::e1DArray;
        break;
    case Shader::TextureType::Color2D:
    case Shader::TextureType::Color2DRect:
        view_kind = multisample ? TextureKind::e2DMultisample : TextureKind::e2D;
        layers = 1;
        break;
    case Shader::TextureType::ColorArray2D:
        view_kind = multisample ? TextureKind::e2DMultisampleArray : TextureKind::e2DArray;
        break;
    case Shader::TextureType::Color3D:
        view_kind = TextureKind::e3D;
        base_layer = 0;
        layers = 1;
        break;
    case Shader::TextureType::ColorCube:
        view_kind = TextureKind::Cube;
        layers = 6;
        break;
    case Shader::TextureType::ColorArrayCube:
        view_kind = TextureKind::CubeArray;
        layers = (std::max)(layers / 6, 1U) * 6;
        break;
    default:
        return nullptr;
    }
    if ((image_kind == TextureKind::e3D) != (view_kind == TextureKind::e3D)) {
        view_kind = image_kind == TextureKind::e3D ? TextureKind::e3D : TextureKind::e2D;
        base_layer = image_kind == TextureKind::e3D ? 0 : base_layer;
        layers = 1;
    }
    if (multisample && PipelineCheckEnabled() && view_kind != TextureKind::e2DMultisample &&
        view_kind != TextureKind::e2DMultisampleArray) {
        view_kind = TextureKind::e2DMultisample;
        layers = 1;
    }
    Object storage = CreateTextureView(image_texture, view_format, view_kind,
                                       static_cast<u32>(range.base.level),
                                       static_cast<u32>(range.extent.levels), base_layer, layers,
                                       IdentitySwizzle);
    const TextureHandle result = storage ? storage.Get() : Handle(texture_type);
    storage_views.push_back(StorageVariant{
        .type = texture_type,
        .format = view_format,
        .view = std::move(storage),
    });
    return result;
}

bool ImageView::PrepareSampleGrid() {
    if (samples == 1 || !MsaaEnabled() || slot_images == nullptr || !image_id) {
        return false;
    }
    Image& image = (*slot_images)[image_id];
    if (!image.CopySampleGrid(true)) {
        return false;
    }
    if (const TextureHandle grid = image.SampleGrid(); grid != sample_grid) {
        sample_grid = grid;
        grid_views = {};
    }
    return true;
}

TextureHandle ImageView::SampledHandle(Shader::TextureType texture_type, bool multisample_read) {
    if (samples == 1 || multisample_read || sample_grid == nullptr || !MsaaEnabled()) {
        return Handle(texture_type);
    }
    const bool array = texture_type == Shader::TextureType::ColorArray2D;
    if (texture_type != Shader::TextureType::Color2D &&
        texture_type != Shader::TextureType::Color2DRect && !array) {
        return Handle(texture_type);
    }
    Object& view = grid_views[array ? 1 : 0];
    if (!view) {
        view = CreateTextureView(sample_grid, host_format,
                                 array ? TextureKind::e2DArray : TextureKind::e2D, 0, 1,
                                 static_cast<u32>(range.base.layer),
                                 array ? static_cast<u32>(range.extent.layers) : 1U, swizzle);
    }
    return view.Get();
}

Attachment ImageView::RenderTarget() const noexcept {
    if (is_3d_slice) {
        return Attachment{
            .texture = image_texture,
            .level = static_cast<u32>(range.base.level),
            .slice = 0,
            .depth_plane = static_cast<u32>(range.base.layer),
        };
    }
    return Attachment{
        .texture = render_view ? render_view.Get() : image_texture,
        .level = static_cast<u32>(range.base.level),
        .slice = static_cast<u32>(range.base.layer),
    };
}

f32 TextureCacheRuntime::OcarinaLodBias() {
    if (!ocarina_lod_bias) {
        const bool ocarina = Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL;
        // MetalFX temporal takes Vulkan's Dlss backend (RasterizerMetal::OcarinaTemporalEnabled);
        // FSR 3.1 has no Metal path.
        ocarina_lod_bias = ocarina && OcarinaTemporal::Enabled() &&
                                   Settings::MetalFxTemporalChosen()
                               ? VideoCommon::OcarinaFsrLodBias(
                                     VideoCommon::OcarinaUpscaleBackend::Dlss,
                                     Settings::OcarinaDlssModeFor(
                                         Settings::values.oot_resolution.GetValue()))
                               : 0.0f;
    }
    return *ocarina_lod_bias;
}

u64 Sampler::ReductionBits() const noexcept {
    return SamplerReductionBits(tsc);
}

Sampler::Sampler(TextureCacheRuntime& runtime_, const Tegra::Texture::TSCEntry& config)
    : device{&runtime_.device}, runtime{&runtime_}, tsc{config} {
    using Tegra::Texture::SamplerReduction;
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    if (!SamplersEnabled()) {
        sampler = CreateSampler(*device, tsc, SamplerOptions{.plain = true});
        return;
    }
    if (const f32 bias = runtime->OcarinaLodBias();
        bias != 0.0f && tsc.mipmap_filter.Value() != TextureMipmapFilter::None) {
        added_lod_bias = bias;
    }
    has_added_anisotropy = std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f) >
                           static_cast<f32>(1U << tsc.max_anisotropy.Value());
    has_linear_filtering = tsc.mag_filter.Value() == TextureFilter::Linear ||
                           tsc.min_filter.Value() == TextureFilter::Linear ||
                           tsc.mipmap_filter.Value() == TextureMipmapFilter::Linear;
    has_depth_comparison = tsc.depth_compare_enabled != 0;
    has_srgb_border = tsc.srgb_conversion != 0 && tsc.SrgbBorderColor() != tsc.BorderColor();
    sampler = CreateSampler(*device, tsc, SamplerOptions{.added_lod_bias = added_lod_bias});
    with_variants = true;
    if (!SamplerBiasSupported(*device) &&
        (tsc.LodBias() != 0.0f || added_lod_bias != 0.0f ||
         tsc.reduction_filter.Value() != SamplerReduction::WeightedAverage)) {
        LOG_ONCE(warned_bias, "Metal: sampler LOD bias {} and reduction {} need macOS / iOS 26 on an Apple7 GPU; "
                 "sampled without them", tsc.LodBias() + added_lod_bias,
                 static_cast<u32>(tsc.reduction_filter.Value()));
    }
}

void* Sampler::HandleFor(const ImageView& view, bool is_depth) {
    if (!with_variants) {
        return sampler.Get();
    }
    enum : u8 { ReduceAnisotropy = 1, ForceNearest = 2, DropCompare = 4, DropBias = 8, Srgb = 16 };
    u8 key = 0;
    if (has_added_anisotropy && !view.SupportsAnisotropy()) {
        key |= ReduceAnisotropy;
    }
    if (has_linear_filtering && VideoCore::Surface::IsPixelFormatInteger(view.format)) {
        key |= ForceNearest;
    }
    // A compare sampler read by a plain sample: Maxwell ignores the comparison.
    if (has_depth_comparison && !is_depth) {
        key |= DropCompare;
    }
    if (added_lod_bias != 0.0f) {
        const Image* const image = view.BackingImage();
        if (!VideoCommon::KeepOcarinaFsrLodBias(
                Settings::IsDockedMode(), false,
                image != nullptr && True(image->flags & ImageFlagBits::GpuModified))) {
            key |= DropBias;
        }
    }
    if (has_srgb_border && VideoCore::Surface::IsPixelFormatSRGB(view.format)) {
        key |= Srgb;
    }
    if (key == 0) {
        return sampler.Get();
    }
    if (const auto it = std::ranges::find(variants, key, &Variant::key); it != variants.end()) {
        return it->state.Get();
    }
    Object state = CreateSampler(*device, tsc,
                                 SamplerOptions{
                                     .added_lod_bias = (key & DropBias) != 0 ? 0.0f : added_lod_bias,
                                     .reduce_anisotropy = (key & ReduceAnisotropy) != 0,
                                     .force_nearest = (key & ForceNearest) != 0,
                                     .drop_compare = (key & DropCompare) != 0,
                                     .srgb_border = (key & Srgb) != 0,
                                 });
    void* const handle = state.Get();
    variants.push_back(Variant{.key = key, .state = std::move(state)});
    return handle;
}

Framebuffer::Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key)
    : width{key.size.width}, height{key.size.height} {
    // The fewest layers among the attachments' views; a 3D slice takes none beyond its own.
    std::optional<u32> view_layers;
    const auto note_layers = [&view_layers](const ImageView& view) {
        const u32 count = True(view.flags & ImageViewFlagBits::Slice)
                              ? 1U
                              : static_cast<u32>((std::max)(view.range.extent.layers, 1));
        view_layers = (std::min)(view_layers.value_or(count), count);
    };
    for (size_t index = 0; index < NUM_RT; ++index) {
        if (color_buffers[index] != nullptr) {
            note_layers(*color_buffers[index]);
        }
    }
    if (depth_buffer != nullptr) {
        note_layers(*depth_buffer);
    }
    if (LayersEnabled()) {
        layers = view_layers.value_or(1);
    }
    for (size_t index = 0; index < NUM_RT; ++index) {
        if (color_buffers[index] == nullptr) {
            continue;
        }
        colors[index] = color_buffers[index]->RenderTarget();
        color_formats[index] = color_buffers[index]->HostFormat();
        color_layers[index] = static_cast<u32>(color_buffers[index]->range.extent.layers);
        samples = color_buffers[index]->Samples();
    }
    if (depth_buffer != nullptr) {
        depth = depth_buffer->RenderTarget();
        depth_layers = static_cast<u32>(depth_buffer->range.extent.layers);
        depth_format = depth_buffer->HostFormat();
        samples = depth_buffer->Samples();
        const FormatClass format_class = depth_buffer->HostClass();
        has_depth = format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
        has_stencil =
            format_class == FormatClass::Stencil || format_class == FormatClass::DepthStencil;
    }
}

Framebuffer::~Framebuffer() = default;

} // namespace Metal
