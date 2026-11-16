// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/dlss_contract.h"
#include "video_core/renderer_metal/metal_graphics.h"
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
    if (dst.Handle() == nullptr || src.Handle() == nullptr || converted ||
        src.info.num_samples != 1 || dst.info.num_samples != 1 ||
        !IsColor(src.HostFormat().format_class) || !IsColor(dst.HostFormat().format_class) ||
        !blocks_fit ||
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
                ToOffset(copy.dst_offset), dst_extent);
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
    const TextureHandle source = src.Handle(Shader::TextureType::Color2D);
    const BlitPass pass{
        .src = source,
        .src_rect = {src_region.start.x, src_region.start.y, src_region.end.x, src_region.end.y},
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
    return Settings::MetalFxTemporalChosen() && !temporal_unsupported;
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
    return TemporalStandIn(dst_framebuffer, dst, input_color, motion_vectors, render_size, jitter,
                           motion_scale, use_history);
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
    desc = TextureDesc{
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
        runtime->allocated_bytes -= bytes + scaled_bytes;
    }
}

bool TextureCacheRuntime::CanRescaleImage(const ImageInfo& info) const {
    constexpr u32 MaxTextureSize = 16384;
    const auto& resolution = Settings::values.resolution_info;
    const FormatInfo host = SurfaceFormat(device, info.format, IsConverted(device, info.format));
    return info.type == ImageType::e2D && info.num_samples == 1 && host.format != 0 &&
           host.attachable && host.format_class != FormatClass::Stencil &&
           resolution.ScaleUp(info.size.width) <= MaxTextureSize &&
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
    if (src == nullptr || dst == nullptr || info.num_samples != 1 ||
        kind == TextureKind::e3D || !host_format.attachable) {
        return false;
    }
    // Depth-stencil images carry both (the blit writes the stencil from the shader); stencil-only
    // ones are not rescaled (CanRescaleImage).
    if (host_format.format_class == FormatClass::Stencil) {
        return false;
    }
    const auto& resolution = Settings::values.resolution_info;
    const bool is_2d = info.type == ImageType::e2D;
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
    // Uploads go to the native texture; a rescaled image draws from the scaled one.
    if (rescaled) {
        CopyScaled(true);
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
    // Downloads read the native texture: bring a rescaled image's drawing to it first.
    if (rescaled) {
        CopyScaled(false);
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
