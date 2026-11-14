// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>

#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/logging.h"
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

[[nodiscard]] TextureKind KindOf(const ImageInfo& info) {
    switch (info.type) {
    case ImageType::e1D:
        return TextureKind::e1DArray;
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

} // Anonymous namespace

TextureCacheRuntime::TextureCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                         StagingBufferPool& staging_buffer_pool_)
    : device{device_}, scheduler{scheduler_}, staging_buffer_pool{staging_buffer_pool_} {}

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
    return dst.HostFormat().format != src.HostFormat().format;
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const VideoCommon::ImageCopy> copies) {
    if (dst.HostFormat().format != src.HostFormat().format) {
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

void TextureCacheRuntime::CopyImageMSAA(Image& dst, Image& src,
                                        std::span<const VideoCommon::ImageCopy> copies) {
    LOG_ONCE(warned, "Metal: copies between sample counts ({} -> {}) are not done yet",
             src.info.num_samples, dst.info.num_samples);
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const VideoCommon::ImageCopy> copies) {
    // Through a buffer: texels keep their bytes, as a copy between formats of one size does.
    const PixelFormat src_format = src.info.format;
    const PixelFormat dst_format = dst.info.format;
    const bool converted = True(src.flags & ImageFlagBits::Converted) ||
                           True(dst.flags & ImageFlagBits::Converted);
    if (dst.Handle() == nullptr || src.Handle() == nullptr || converted ||
        src.info.num_samples != 1 || dst.info.num_samples != 1 ||
        !IsColor(src.HostFormat().format_class) || !IsColor(dst.HostFormat().format_class) ||
        VideoCore::Surface::DefaultBlockWidth(src_format) != 1 ||
        VideoCore::Surface::DefaultBlockWidth(dst_format) != 1 ||
        VideoCore::Surface::BytesPerBlock(src_format) !=
            VideoCore::Surface::BytesPerBlock(dst_format)) {
        LOG_ONCE(warned, "Metal: reinterpreting {} as {} is not done yet", src_format,
                 dst_format);
        return;
    }
    const u32 bytes_per_texel = VideoCore::Surface::BytesPerBlock(src_format);
    for (const VideoCommon::ImageCopy& copy : copies) {
        const Extent3D extent = ClipExtent(src.info, 0, 0, copy.src_subresource.base_level,
                                           copy.src_offset, copy.extent);
        if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
            continue;
        }
        const u32 bytes_per_row = extent.width * bytes_per_texel;
        const u32 bytes_per_image = bytes_per_row * extent.height;
        const u64 size = u64{bytes_per_image} * extent.depth;
        if (size > reinterpret_buffer_size) {
            reinterpret_buffer_size = std::bit_ceil(size);
            reinterpret_buffer = CreateBuffer(device, reinterpret_buffer_size, false, nullptr);
        }
        const bool src_3d = src.info.type == ImageType::e3D;
        const bool dst_3d = dst.info.type == ImageType::e3D;
        const s32 layers = (std::max)(copy.src_subresource.num_layers, 1);
        for (s32 layer = 0; layer < layers; ++layer) {
            scheduler.CopyTextureToBuffer(
                src.Handle(), static_cast<u32>(copy.src_subresource.base_level),
                src_3d ? 0U : static_cast<u32>(copy.src_subresource.base_layer + layer),
                ToOffset(copy.src_offset), extent, reinterpret_buffer.Get(), 0, bytes_per_row,
                src_3d ? bytes_per_image : 0U);
            scheduler.BreakBlits();
            scheduler.CopyBufferToTexture(
                reinterpret_buffer.Get(), 0, bytes_per_row, dst_3d ? bytes_per_image : 0U,
                dst.Handle(), static_cast<u32>(copy.dst_subresource.base_level),
                dst_3d ? 0U : static_cast<u32>(copy.dst_subresource.base_layer + layer),
                ToOffset(copy.dst_offset), extent);
            scheduler.BreakBlits();
        }
    }
}

void TextureCacheRuntime::ConvertImage(Framebuffer* dst, ImageView& dst_view,
                                       ImageView& src_view) {
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
    if (!IsColor(dst.HostClass()) || !IsColor(src.HostClass())) {
        LOG_ONCE(warned_depth, "Metal: depth and stencil blits are not done yet");
        return;
    }
    const TextureHandle source = src.Handle(Shader::TextureType::Color2D);
    const BlitPass pass{
        .src = source,
        .src_rect = {src_region.start.x, src_region.start.y, src_region.end.x, src_region.end.y},
        .dst = dst.RenderTarget(),
        .dst_format = dst.HostFormat(),
        .dst_rect = {dst_region.start.x, dst_region.start.y, dst_region.end.x, dst_region.end.y},
        .dst_width = dst_framebuffer != nullptr ? dst_framebuffer->Width() : dst.size.width,
        .dst_height = dst_framebuffer != nullptr ? dst_framebuffer->Height() : dst.size.height,
        .linear = filter == Tegra::Engines::Fermi2D::Filter::Bilinear,
        .keep_dst = true,
    };
    if (!scheduler.Blit(pass)) {
        LOG_ONCE(warned_format, "Metal: a blit into format {} was dropped", dst.format);
    }
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
    const bool converted = IsConverted(runtime->device, info.format);
    if (converted) {
        flags |= ImageFlagBits::Converted;
        flags |= ImageFlagBits::CostlyLoad;
    }
    host_format = SurfaceFormat(runtime->device, info.format, converted);
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
    const TextureDesc desc{
        .kind = kind,
        .format = host_format.format,
        .width = info.size.width >> samples_x,
        .height = info.type == ImageType::e1D ? 1U : info.size.height >> samples_y,
        .depth = info.type == ImageType::e3D ? info.size.depth : 1U,
        .levels = static_cast<u32>(info.resources.levels),
        .layers = info.type == ImageType::e3D ? 1U : static_cast<u32>(info.resources.layers),
        .samples = info.num_samples,
        .render_target = host_format.attachable,
        .shader_write = host_format.storage && info.num_samples == 1,
    };
    texture = CreateTexture(runtime->device, desc);
    if (!texture) {
        LOG_ERROR(Render_Metal, "Metal: texture {}x{}x{} format {} levels {} layers {} failed",
                  desc.width, desc.height, desc.depth, info.format, desc.levels, desc.layers);
        return;
    }
    bytes = AllocatedSize(texture.Get());
    runtime->allocated_bytes += bytes;
}

Image::Image(const VideoCommon::NullImageParams& params) : VideoCommon::ImageBase{params} {}

Image::~Image() {
    if (texture && runtime != nullptr) {
        runtime->allocated_bytes -= bytes;
    }
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

bool Image::CanCopyWithBuffers() const {
    if (!texture || info.num_samples != 1) {
        return false;
    }
    switch (host_format.format_class) {
    case FormatClass::Depth:
        // Depth32Float and Depth16Unorm hold the guest's bytes; 24-bit depth does not.
        if (info.format == PixelFormat::D32_FLOAT || info.format == PixelFormat::D16_UNORM) {
            return true;
        }
        [[fallthrough]];
    case FormatClass::Stencil:
    case FormatClass::DepthStencil:
        LOG_ONCE(warned, "Metal: depth and stencil uploads and downloads ({}) are not done yet",
                 info.format);
        return false;
    default:
        return true;
    }
}

void Image::UploadMemory(BufferHandle buffer, size_t offset,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    if (!CanCopyWithBuffers()) {
        return;
    }
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
            runtime->scheduler.CopyBufferToTexture(
                buffer, offset + copy.buffer_offset + u64{bytes_per_image} * layer, bytes_per_row,
                is_3d ? bytes_per_image : 0U, texture.Get(),
                static_cast<u32>(copy.image_subresource.base_level),
                is_3d ? 0U : static_cast<u32>(copy.image_subresource.base_layer + layer),
                ToOffset(copy.image_offset), extent);
        }
    }
}

void Image::UploadMemory(const StagingBufferRef& map,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
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
                runtime->scheduler.CopyTextureToBuffer(
                    texture.Get(), static_cast<u32>(copy.image_subresource.base_level),
                    is_3d ? 0U : static_cast<u32>(copy.image_subresource.base_layer + layer),
                    ToOffset(copy.image_offset), extent, buffers[index],
                    offsets[index] + copy.buffer_offset + u64{bytes_per_image} * layer,
                    bytes_per_row, is_3d ? bytes_per_image : 0U);
            }
        }
    }
}

void Image::DownloadMemory(const StagingBufferRef& map,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    DownloadMemory(map.buffer, map.offset, copies);
}

ImageView::ImageView(TextureCacheRuntime& runtime, const ImageViewInfo& info, ImageId image_id_,
                     Image& image, const SlotVector<Image>&)
    : VideoCommon::ImageViewBase{info, image.info, image_id_, image.gpu_addr},
      image_texture{image.Handle()}, samples{image.info.num_samples}, image_kind{image.Kind()} {
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
      buffer_size{VideoCommon::CalculateGuestSizeInBytes(info)} {}

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
        view_kind = TextureKind::e1D;
        layers = 1;
        break;
    case Shader::TextureType::ColorArray1D:
        view_kind = TextureKind::e1DArray;
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
    view = CreateTextureView(image_texture, host_format, view_kind,
                             static_cast<u32>(range.base.level),
                             static_cast<u32>(range.extent.levels), base_layer, layers, swizzle);
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

Sampler::Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& config)
    : sampler{CreateSampler(runtime.device, config)} {}

Framebuffer::Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key)
    : width{key.size.width}, height{key.size.height} {
    for (size_t index = 0; index < NUM_RT; ++index) {
        if (color_buffers[index] == nullptr) {
            continue;
        }
        colors[index] = color_buffers[index]->RenderTarget();
        color_formats[index] = color_buffers[index]->HostFormat();
        samples = color_buffers[index]->Samples();
    }
    if (depth_buffer != nullptr) {
        depth = depth_buffer->RenderTarget();
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
