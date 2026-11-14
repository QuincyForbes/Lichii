// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The texture cache runtime (VideoCommon::TextureCache<TextureCacheParams>), modelled on
// OpenGL's minimal one. Plain C++: Metal calls go through metal_resources.h and Scheduler.

#pragma once

#include <array>
#include <optional>
#include <span>

#include "shader_recompiler/shader_info.h"
#include "video_core/game_dlss_status.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/texture_cache_base.h"

namespace Metal {

class Device;
class Framebuffer;
class Image;
class ImageView;
class Sampler;

using Common::SlotVector;
using VideoCommon::ImageId;
using VideoCommon::NUM_RT;
using VideoCommon::Region2D;

class TextureCacheRuntime {
public:
    explicit TextureCacheRuntime(const Device& device, Scheduler& scheduler,
                                 StagingBufferPool& staging_buffer_pool);
    ~TextureCacheRuntime();

    void Finish();

    StagingBufferRef UploadStagingBuffer(size_t size, bool deferred = false);

    StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void TickFrame() {}

    void FlushDeferredClear() {}

    [[nodiscard]] u64 GetDeviceLocalMemory() const;

    [[nodiscard]] u64 GetDeviceMemoryUsage() const;

    [[nodiscard]] bool CanReportMemoryUsage() const noexcept {
        return true;
    }

    /// Metal copies texels only between textures of one format: any other pair is reinterpreted.
    [[nodiscard]] bool ShouldReinterpret(Image& dst, Image& src) const noexcept;

    [[nodiscard]] bool CanUploadMSAA() const noexcept {
        return false;
    }

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void CopyImageMSAA(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ReinterpretImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view);

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                   const Region2D& dst_region, const Region2D& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    /// The DLSS stand-ins (MetalFX temporal comes with milestone 4): the game keeps its own
    /// upscaler.
    bool TemporalUpscale(Framebuffer*, ImageView&, ImageView&, ImageView&, ImageView*,
                         std::array<u32, 2>, std::array<f32, 2>, std::array<f32, 2>, bool) {
        return false;
    }

    [[nodiscard]] bool WantsTemporalUpscaleDepth() const noexcept {
        return false;
    }

    [[nodiscard]] std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(ImageView&,
                                                                              ImageView&) {
        return std::nullopt;
    }

    [[nodiscard]] bool CanAccelerateImageUpload(Image&) const noexcept {
        return false;
    }

    void AccelerateImageUpload(Image&, const StagingBufferRef&,
                               std::span<const VideoCommon::SwizzleParameters>, u32 z_start,
                               u32 z_count);

    void InsertUploadMemoryBarrier() {}

    void TransitionImageLayout(Image&) {}

    [[nodiscard]] bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }

    [[nodiscard]] bool HasNativeBgr() const noexcept {
        return true;
    }

    void BarrierFeedbackLoop() const noexcept {}

    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;

    /// GPU bytes the texture cache's textures hold.
    u64 allocated_bytes{};
    /// Shared buffer for reinterpreting copies, grown as needed.
    Object reinterpret_buffer;
    u64 reinterpret_buffer_size{};
};

class Image : public VideoCommon::ImageBase {
public:
    explicit Image(TextureCacheRuntime&, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
                   VAddr cpu_addr);
    explicit Image(const VideoCommon::NullImageParams&);

    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    void UploadMemory(BufferHandle buffer, size_t offset,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void UploadMemory(const StagingBufferRef& map,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(BufferHandle buffer, size_t offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(std::span<BufferHandle> buffers, std::span<size_t> offsets,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(const StagingBufferRef& map,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    [[nodiscard]] TextureHandle Handle() const noexcept {
        return texture.Get();
    }

    /// The host format (MTLPixelFormat) and what it is.
    [[nodiscard]] const FormatInfo& HostFormat() const noexcept {
        return host_format;
    }

    [[nodiscard]] TextureKind Kind() const noexcept {
        return kind;
    }

    [[nodiscard]] bool IsRescaled() const noexcept {
        return false;
    }

    /// Resolution scaling is not done on Metal yet.
    bool ScaleUp(bool = false) {
        return false;
    }

    bool ScaleDown(bool = false) {
        return false;
    }

    u64 allocation_tick{};

private:
    /// Bytes a row of the copy holds in the buffer, and a layer (2D) or slice (3D).
    [[nodiscard]] std::pair<u32, u32> CopyPitches(const VideoCommon::BufferImageCopy& copy) const;

    /// Depth and stencil data are not copied to or from buffers yet (Metal needs per-aspect
    /// copies and has no 24-bit depth). Logs once.
    [[nodiscard]] bool CanCopyWithBuffers() const;

    TextureCacheRuntime* runtime{};
    Object texture;
    FormatInfo host_format{};
    TextureKind kind{TextureKind::e2D};
    u64 bytes{};
};

class ImageView : public VideoCommon::ImageViewBase {
public:
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&,
                       const SlotVector<Image>&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo&,
                       const VideoCommon::ImageViewInfo&, GPUVAddr);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo&,
                       const VideoCommon::ImageViewInfo&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams&);

    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    /// The view shaders sample as `texture_type`, made on first use. Null for null views.
    [[nodiscard]] TextureHandle Handle(Shader::TextureType texture_type);

    /// The attachment that renders into this view's first level and layer.
    [[nodiscard]] Attachment RenderTarget() const noexcept;

    [[nodiscard]] u32 HostFormat() const noexcept {
        return host_format;
    }

    [[nodiscard]] FormatClass HostClass() const noexcept {
        return ClassOf(host_format);
    }

    [[nodiscard]] TextureHandle ImageHandle() const noexcept {
        return image_texture;
    }

    [[nodiscard]] GPUVAddr GpuAddr() const noexcept {
        return gpu_addr;
    }

    [[nodiscard]] u32 BufferSize() const noexcept {
        return buffer_size;
    }

    [[nodiscard]] u32 Samples() const noexcept {
        return samples;
    }

private:
    TextureHandle image_texture{};
    u32 samples{1};
    TextureKind image_kind{TextureKind::e2D};
    u32 host_format{};
    std::array<u8, 4> swizzle{IdentitySwizzle};
    std::array<Object, Shader::NUM_TEXTURE_TYPES> views;
    Object render_view; ///< Only when the view's format is not its image's
    bool is_render_target{};
    bool is_3d_slice{};
    u32 buffer_size{};
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry&);

    [[nodiscard]] void* Handle() const noexcept {
        return sampler.Get();
    }

private:
    Object sampler;
};

class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key);

    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    Framebuffer(Framebuffer&&) = default;
    Framebuffer& operator=(Framebuffer&&) = default;

    [[nodiscard]] const std::array<Attachment, NUM_RT>& Colors() const noexcept {
        return colors;
    }

    [[nodiscard]] const std::array<u32, NUM_RT>& ColorFormats() const noexcept {
        return color_formats;
    }

    [[nodiscard]] const Attachment& Depth() const noexcept {
        return depth;
    }

    [[nodiscard]] u32 DepthFormat() const noexcept {
        return depth_format;
    }

    [[nodiscard]] u32 Samples() const noexcept {
        return samples;
    }

    [[nodiscard]] bool HasDepth() const noexcept {
        return has_depth;
    }

    [[nodiscard]] bool HasStencil() const noexcept {
        return has_stencil;
    }

    [[nodiscard]] u32 Width() const noexcept {
        return width;
    }

    [[nodiscard]] u32 Height() const noexcept {
        return height;
    }

private:
    std::array<Attachment, NUM_RT> colors{};
    std::array<u32, NUM_RT> color_formats{};
    Attachment depth{};
    u32 depth_format{};
    u32 samples{1};
    bool has_depth{};
    bool has_stencil{};
    u32 width{};
    u32 height{};
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = false;
    static constexpr bool HAS_MSAA_DOWNLOADS = false;

    using Runtime = Metal::TextureCacheRuntime;
    using Image = Metal::Image;
    using ImageAlloc = Metal::ImageAlloc;
    using ImageView = Metal::ImageView;
    using Sampler = Metal::Sampler;
    using Framebuffer = Metal::Framebuffer;
    using AsyncBuffer = Metal::StagingBufferRef;
    using BufferType = Metal::BufferHandle;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace Metal
