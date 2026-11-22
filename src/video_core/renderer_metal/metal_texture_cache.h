// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The texture cache runtime (VideoCommon::TextureCache<TextureCacheParams>), modelled on
// OpenGL's minimal one. Plain C++: Metal calls go through metal_resources.h and Scheduler.

#pragma once

#include <array>
#include <chrono>
#include <list>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shader_recompiler/shader_info.h"
#include "video_core/game_dlss_status.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/renderer_metal/metal_temporal.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/texture_cache_base.h"

namespace Metal {

class Device;
class Framebuffer;
class CameraMotionPass;
class Image;
class ImageView;
class Fsr3Upscaler;
class PresentPasses;
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

    void TickFrame();

    void FlushDeferredClear() {}

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return scheduler.CurrentTick();
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return scheduler.IsFree(tick);
    }

    [[nodiscard]] u64 PoolFrame() const noexcept {
        return pool_frame;
    }

    [[nodiscard]] u64 PooledImageBytes() const noexcept {
        return pooled_texture_bytes;
    }

    [[nodiscard]] std::pair<Object, u64> AcquireTexture(const TextureDesc& tex_desc,
                                                        bool* out_from_pool = nullptr);
    void ReleaseTexture(const TextureDesc& tex_desc, Object&& tex, u64 tex_bytes);
    void TrimTexturePool(u64 max_bytes);

    [[nodiscard]] u64 GetDeviceLocalMemory() const;

    [[nodiscard]] u64 GetDeviceMemoryUsage() const;

    [[nodiscard]] bool CanReportMemoryUsage() const noexcept {
        return true;
    }

    [[nodiscard]] u32 GetMaxScaledExtent() const noexcept {
        return 16384;
    }

    /// Resolution scaling takes single-sampled 2D render targets Metal can blit (no stencil-only
    /// ones) whose scaled size fits a texture.
    [[nodiscard]] bool CanRescaleImage(const VideoCommon::ImageInfo& info) const;

    /// Metal copies texels only between textures of one format: any other pair is reinterpreted.
    [[nodiscard]] bool ShouldReinterpret(Image& dst, Image& src) const noexcept;

    /// Multisampled images go through a single-sampled texture holding their samples as texels
    /// (EDEN_NX2_METAL_MSAA).
    [[nodiscard]] bool CanUploadMSAA() const noexcept;

    [[nodiscard]] bool CanDownloadMsaa(const VideoCommon::ImageInfo& info) const;

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void CopyImageMSAA(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ReinterpretImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view);

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                   const Region2D& dst_region, const Region2D& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    /// MetalFX temporal upscaling in place of Ocarina of Time's anti-aliasing, as the Vulkan
    /// runtime's DLSS camera-motion upscale: motion vectors from the scene depth and the camera
    /// (`reprojection`; host_shaders/camera_motion_vectors.comp through the translator), MetalFX
    /// with the host's jitter, the result drawn into `dst_framebuffer`. False leaves the game's own.
    bool UpscaleWithCameraMotion(Framebuffer* dst_framebuffer, ImageView& input_color,
                                 ImageView& depth, std::array<u32, 2> render_size,
                                 const std::array<f32, 16>& reprojection,
                                 std::array<f32, 2> jitter, bool reset);

    /// MetalFX or FSR 3.1 temporal upscaling in place of the game's DLSS, else Vulkan's temporal
    /// stand-in (EDEN_NX2_METAL_TEMPORAL_STANDIN); false leaves the game's own upscaler.
    bool TemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& input_color,
                         ImageView& motion_vectors, ImageView* depth,
                         std::array<u32, 2> render_size, std::array<f32, 2> jitter,
                         std::array<f32, 2> motion_scale, bool use_history,
                         std::optional<std::array<u32, 2>> render_region = std::nullopt);

    /// Frames the temporal stand-in drew since the last call.
    [[nodiscard]] u64 TakeStandInFrames() noexcept {
        return std::exchange(standin_frames, 0);
    }

    [[nodiscard]] bool WantsTemporalUpscaleDepth() const;

    [[nodiscard]] std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        ImageView&, ImageView&, std::optional<std::array<u32, 2>> src_region = std::nullopt) {
        (void)src_region;
        return std::nullopt;
    }

    [[nodiscard]] bool CanAccelerateImageUpload(Image&) const noexcept {
        return false;
    }

    void AccelerateImageUpload(Image&, const StagingBufferRef&,
                               std::span<const VideoCommon::SwizzleParameters>, u32 z_start,
                               u32 z_count);

    void InsertUploadMemoryBarrier() {}

    void TransitionImageLayout(Image& image);

    [[nodiscard]] bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }

    [[nodiscard]] bool HasNativeBgr() const noexcept {
        return true;
    }

    /// A draw samples the depth image its pass writes (TextureCache::CheckFeedbackLoop): the pass
    /// ends, storing what was drawn, so the draw's own pass reads it, as Vulkan's barrier orders
    /// it (EDEN_NX2_METAL_FRAGMENT_BARRIER).
    void BarrierFeedbackLoop();

    /// The LOD bias Ocarina of Time's mipmapped samplers get while a jittered temporal upscaler
    /// runs (VideoCommon::OcarinaFsrLodBias with MetalFX as Vulkan's Dlss backend), else 0.
    [[nodiscard]] f32 OcarinaLodBias();

    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;
    /// The sample counts the device's textures take.
    PipelineLimits sample_limits;

    /// GPU bytes the texture cache's textures hold.
    u64 allocated_bytes{};
    std::unique_ptr<TemporalUpscaler> temporal;
    std::unique_ptr<CameraMotionPass> camera_motion_pass;
    bool temporal_unsupported{};
    std::optional<std::chrono::steady_clock::time_point> temporal_last_evaluation;
    std::optional<u64> temporal_last_frame;
    std::array<u32, 2> temporal_last_render{};
    std::array<u32, 2> temporal_last_output{};
    std::unique_ptr<PresentPasses> present_passes;
    std::unique_ptr<Fsr3Upscaler> fsr3;
    bool fsr3_unsupported{};
    std::optional<std::chrono::steady_clock::time_point> fsr3_last_evaluation;
    std::optional<u64> fsr3_last_frame;
    std::array<u32, 2> fsr3_last_render{};
    std::array<u32, 2> fsr3_last_output{};
    bool standin_failed{};
    std::optional<u64> standin_last_frame;
    std::array<u32, 2> standin_last_render{};
    u64 standin_frames{};
    f32 temporal_lod_bias{};
    u64 temporal_lod_frame{};
    bool ocarina_upscaler_failed{};

private:
    void ZeroTexture(TextureHandle tex, const TextureDesc& tex_desc,
                     VideoCore::Surface::PixelFormat guest_format);
    Object pool_zero_buffer;
    u64 pool_zero_size{};
    void NoteTemporalLodBias(std::array<u32, 2> render_size,
                             std::array<u32, 2> output_size) noexcept;
    static bool TemporalStandInEnabled();
    bool MetalFxTemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst,
                                ImageView& input_color, ImageView& motion_vectors,
                                ImageView& depth_view, std::array<u32, 2> render_size,
                                std::array<f32, 2> jitter, std::array<f32, 2> motion_scale);
    /// FSR 3.1 in place of the game's DLSS (Vulkan's UpscaleWithFsr3), before the stand-in.
    bool Fsr3TemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& input_color,
                             ImageView& motion_vectors, ImageView& depth_view,
                             std::array<u32, 2> render_size, std::array<f32, 2> jitter,
                             std::array<f32, 2> motion_scale);
    bool TemporalStandIn(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& input_color,
                         ImageView& motion_vectors, std::array<u32, 2> render_size,
                         std::array<f32, 2> jitter, std::array<f32, 2> motion_scale,
                         bool use_history);

public:

    /// Shared buffer for reinterpreting copies, grown as needed.
    Object reinterpret_buffer;
    u64 reinterpret_buffer_size{};

    std::optional<f32> ocarina_lod_bias;

private:
    struct TextureDescHash {
        size_t operator()(const TextureDesc& d) const noexcept;
    };
    struct PooledTexture {
        TextureDesc desc{};
        Object texture;
        u64 bytes{};
        u64 frame{};
    };
    void PopOldestPooledTexture();

    std::list<PooledTexture> texture_pool_lru;
    std::unordered_map<TextureDesc, std::vector<std::list<PooledTexture>::iterator>,
                       TextureDescHash>
        texture_pool_index;
    u64 pooled_texture_bytes{};
    u64 pool_frame{};
    u64 pool_hits{};
    u64 pool_misses{};
    u64 pool_recycled{};
    u64 pool_evicted{};
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
                      std::span<const VideoCommon::BufferImageCopy> copies, bool ahead = false);

    void UploadMemory(const StagingBufferRef& map,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(BufferHandle buffer, size_t offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(std::span<BufferHandle> buffers, std::span<size_t> offsets,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    void DownloadMemory(const StagingBufferRef& map,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    /// The texture draws and copies use: the scaled one while the image is rescaled.
    [[nodiscard]] TextureHandle Handle() const noexcept {
        return rescaled ? scaled_texture.Get() : texture.Get();
    }

    /// The host format (MTLPixelFormat) and what it is.
    [[nodiscard]] const FormatInfo& HostFormat() const noexcept {
        return host_format;
    }

    [[nodiscard]] TextureKind Kind() const noexcept {
        return kind;
    }

    /// Samples per pixel of the texture (the guest's, or the device's most when it has fewer).
    [[nodiscard]] u32 HostSamples() const noexcept {
        return desc.samples;
    }

    [[nodiscard]] bool IsRescaled() const noexcept {
        return rescaled;
    }

    /// Resolution scaling: draws go to a texture at the scaled size, which gets the native
    /// texture's contents unless `ignore`.
    bool ScaleUp(bool ignore = false);

    bool ScaleDown(bool ignore = false);

    /// A multisampled image's samples as texels of a single-sampled 2D array texture, as the
    /// guest sees the image (made on first use; uploads, downloads and plain sampling go through
    /// it).
    [[nodiscard]] TextureHandle SampleGrid();

    /// Copies the samples into the grid (`expand`) or the grid into the samples.
    bool CopySampleGrid(bool expand);

    /// Whether the image's data is copied to and from buffers (multisampled stencil is not;
    /// 24-bit depth and the formats with stencil only under EDEN_NX2_METAL_COPIES). Logs once.
    [[nodiscard]] bool CanCopyWithBuffers() const;

    /// How the guest lays out the image's depth and stencil when Metal's blit copies can't take
    /// its bytes (24-bit depth, depth with stencil), else nothing.
    [[nodiscard]] std::optional<DepthLayout> DepthBytesLayout() const;

    /// SparseTiles images (G18, as Vulkan's): maps the tiles under `rects` (and the mip tail) of
    /// the sparse texture; false (nothing mapped) when its heap is full.
    bool BindTiles(std::span<const VideoCommon::LevelRect> rects);

    /// The sparse tile in texels, the first level of the mip tail, and what the image holds.
    [[nodiscard]] u32 SparseTileWidth() const noexcept {
        return sparse.tile_width;
    }
    [[nodiscard]] u32 SparseTileHeight() const noexcept {
        return sparse.tile_height;
    }
    [[nodiscard]] s32 SparseTailLevel() const noexcept {
        return static_cast<s32>(sparse.tail_level);
    }
    [[nodiscard]] u64 SparseBoundBytes() const noexcept {
        return sparse.heap_bytes;
    }

    /// PackedSwapShift of the guest format (0: the texture holds the guest's bytes).
    [[nodiscard]] u32 PackedShift() const noexcept {
        return packed_shift;
    }

    /// Pack16Mode of a packed format held as RGBA8 (moved to and from the guest's words by
    /// kernels), else 0.
    [[nodiscard]] u32 Pack16() const noexcept {
        return pack16;
    }

    [[nodiscard]] bool FromPool() const noexcept {
        return from_pool;
    }

    [[nodiscard]] bool TakePoolZeroNeeded() noexcept {
        return std::exchange(needs_pool_zero, false);
    }

    [[nodiscard]] TextureHandle BaseHandle() const noexcept {
        return texture.Get();
    }

    [[nodiscard]] const TextureDesc& BaseDesc() const noexcept {
        return desc;
    }

    [[nodiscard]] const TextureDesc& ActiveDesc() const noexcept {
        return rescaled ? scaled_desc : desc;
    }

    void ReleaseToPool();

    u64 allocation_tick{};

private:
    /// Calls `region(buffer offset, width, rows, images, bytes per row, bytes per image)` for
    /// every run of texels a list of buffer copies moves (Image::PackedShift images).
    template <typename F>
    void ForEachCopyRegion(std::span<const VideoCommon::BufferImageCopy> copies, F&& region) const;

    /// Copies every level and layer between the native and the scaled texture (scaled to fit).
    bool CopyScaled(bool scale_up);

    /// Bytes a row of the copy holds in the buffer, and a layer (2D) or slice (3D).
    [[nodiscard]] std::pair<u32, u32> CopyPitches(const VideoCommon::BufferImageCopy& copy) const;

    struct SparseTiles {
        Object heap;
        u32 tile_width{};
        u32 tile_height{};
        u32 tile_bytes{};
        u32 tail_level{};
        u32 tail_tiles{};
        u64 heap_bytes{};
        u64 bound_bytes{};
        bool tail_mapped{};
        std::vector<std::vector<bool>> mapped; ///< Per level before the tail, per tile
    };

    TextureCacheRuntime* runtime{};
    u32 packed_shift{};
    u32 pack16{};
    Object resolve_shadow; ///< Q27: the single-sampled resolve of a multisampled colour image
    SparseTiles sparse;
    Object texture;
    /// The texture at the scaled size, made on the first ScaleUp.
    Object scaled_texture;
    Object sample_grid;
    TextureDesc desc{};
    TextureDesc scaled_desc{};
    FormatInfo host_format{};
    TextureKind kind{TextureKind::e2D};
    u64 bytes{};
    u64 base_texture_bytes{};
    u64 scaled_bytes{};
    bool rescaled{};
    bool from_pool{};
    bool needs_pool_zero{};
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
    [[nodiscard]] TextureHandle Handle(Shader::TextureType texture_type,
                                       u64* out_gpu_resource_id = nullptr);

    /// Cached MTLResourceID of `Handle(texture_type)` (0 for null views).
    [[nodiscard]] u64 GpuResourceId(Shader::TextureType texture_type) {
        u64 res_id = 0;
        static_cast<void>(Handle(texture_type, &res_id));
        return res_id;
    }

    /// The view shaders read/write as a storage image (`info.image_descriptors`), using a linear
    /// format for sRGB images and the declared `image_format` when typed, with identity swizzle.
    [[nodiscard]] TextureHandle StorageHandle(Shader::TextureType texture_type,
                                              Shader::ImageFormat image_format,
                                              u64* out_gpu_resource_id = nullptr);

    /// What a program reads as `texture_type`: a multisampled image read as a plain texture
    /// (`multisample_read` false) reads the grid of its samples (UseSampleGrid) with
    /// EDEN_NX2_METAL_MSAA, as the guest sees it.
    [[nodiscard]] TextureHandle SampledHandle(Shader::TextureType texture_type,
                                              bool multisample_read,
                                              u64* out_gpu_resource_id = nullptr);

    /// Fills the image's sample grid (Image::SampleGrid) for a draw that reads this
    /// multisampled view as a plain texture; false when it can't (the view is bound as it is).
    bool PrepareSampleGrid();

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

    /// The guest's samples per pixel (Samples() may be fewer on this GPU).
    [[nodiscard]] u32 GuestSamples() const noexcept {
        return guest_samples;
    }

    /// The image the view shows (null for null and buffer views).
    [[nodiscard]] const Image* BackingImage() const noexcept {
        return slot_images != nullptr && image_id ? &(*slot_images)[image_id] : nullptr;
    }

private:
    struct StorageVariant {
        Shader::TextureType type{};
        u32 format{};
        Object view;
        u64 gpu_resource_id{};
    };

    SlotVector<Image>* slot_images{};
    TextureHandle image_texture{};
    u32 samples{1};
    u32 guest_samples{1};
    TextureKind image_kind{TextureKind::e2D};
    u32 host_format{};
    std::array<u8, 4> swizzle{IdentitySwizzle};
    std::array<Object, Shader::NUM_TEXTURE_TYPES> views;
    std::array<u64, Shader::NUM_TEXTURE_TYPES> view_resource_ids{};
    std::vector<StorageVariant> storage_views;
    TextureHandle sample_grid{};
    std::array<Object, 2> grid_views; ///< 2D and 2D array views of sample_grid
    std::array<u64, 2> grid_view_resource_ids{};
    Object render_view; ///< Only when the view's format is not its image's
    bool is_render_target{};
    bool is_3d_slice{};
    u32 buffer_size{};
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry&);

    /// SamplerReductionBits of the guest sampler.
    [[nodiscard]] u64 ReductionBits() const noexcept;
    [[nodiscard]] const Tegra::Texture::TSCEntry& Descriptor() const noexcept {
        return tsc;
    }

    /// The sampler state without the changes a view asks for.
    [[nodiscard]] void* Handle() const noexcept {
        return sampler.Get();
    }

    [[nodiscard]] u64 GpuResourceId() const noexcept {
        return gpu_resource_id;
    }

    /// The sampler state for sampling `view` (`is_depth`: a depth comparison), as Vulkan's
    /// Sampler::HandleFor picks its variants: no added anisotropy where the view takes none, no
    /// filtering of integer views, no comparison for a plain sample, no Ocarina upscaler bias for
    /// images the GPU wrote, the sRGB border colour for sRGB views.
    [[nodiscard]] void* HandleFor(const ImageView& view, bool is_depth,
                                  u64* out_gpu_resource_id = nullptr);

private:
    struct Variant {
        u16 key{};
        Object state;
        u64 gpu_resource_id{};
    };

    const Device* device{};
    TextureCacheRuntime* runtime{};
    Tegra::Texture::TSCEntry tsc{};
    Object sampler;
    u64 gpu_resource_id{};
    std::vector<Variant> variants;
    f32 added_lod_bias{};
    bool has_added_anisotropy{};
    bool has_linear_filtering{};
    bool has_depth_comparison{};
    bool has_srgb_border{};
    bool with_variants{};
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

    /// Layers of each colour target's view, and of the depth target's, from its slice on.
    [[nodiscard]] const std::array<u32, NUM_RT>& ColorLayers() const noexcept {
        return color_layers;
    }

    [[nodiscard]] u32 DepthLayers() const noexcept {
        return depth_layers;
    }

    /// The layers every attachment's view has (RenderTargetSet::layers).
    [[nodiscard]] u32 Layers() const noexcept {
        return layers;
    }

private:
    std::array<Attachment, NUM_RT> colors{};
    std::array<u32, NUM_RT> color_formats{};
    std::array<u32, NUM_RT> color_layers{};
    u32 depth_layers{};
    Attachment depth{};
    u32 depth_format{};
    u32 samples{1};
    bool has_depth{};
    bool has_stencil{};
    u32 width{};
    u32 height{};
    u32 layers{1};
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = true;
    static constexpr bool HAS_MSAA_DOWNLOADS = true;

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
