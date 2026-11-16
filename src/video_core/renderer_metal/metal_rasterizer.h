// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>

#include "common/common_types.h"
#include "video_core/buffer_cache/nx2_write_handoff.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/oot_network.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_fence_manager.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/renderer_metal/metal_texture_cache.h"

namespace Tegra {
class GPU;
struct FramebufferConfig;
} // namespace Tegra

namespace Metal {

class Device;
class Scheduler;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& buffer_operand,
                            const Tegra::DMA::ImageOperand& image_operand);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
};

/// The guest framebuffer a present shows, found in the texture cache.
struct DisplayTexture {
    TextureHandle texture{};
    u32 width{};
    u32 height{};
    u32 scaled_width{};
    u32 scaled_height{};
};

/// Takes the GPU's work into the texture and buffer caches. Milestone 2: memory, copies, blits,
/// clears and DMA run on Metal; draws and dispatches are counted, not drawn (milestone 3).
class RasterizerMetal final : public VideoCore::RasterizerInterface,
                              protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerMetal(Tegra::GPU& gpu, Tegra::MaxwellDeviceMemoryManager& device_memory,
                             const Device& device, Scheduler& scheduler);
    ~RasterizerMetal() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    bool DrawMeshTasksFromGpu(DAddr count_addr) override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    bool LandDownloadForCpuRead(DAddr addr, u64 size, u64 tick) override;
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool FencesQuiet(bool strict, u64& epoch) override;
    u64 FenceWorkEpoch() const override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    bool AccelerateConditionalRendering() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    bool AccelerateTemporalUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                   const Tegra::Engines::Fermi2D::Surface& dst,
                                   const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                   const Tegra::Texture::TICEntry* depth,
                                   std::array<f32, 2> jitter, std::array<f32, 2> motion_scale,
                                   bool use_history) override;
    VideoCommon::OotNetwork::BypassState* OotNetworkBypass() override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;

    /// The texture cache's image at the framebuffer's address, if it has one.
    [[nodiscard]] std::optional<DisplayTexture> AccelerateDisplay(
        const Tegra::FramebufferConfig& config, DAddr framebuffer_addr);

    struct WorkCounts {
        u64 draws;
        u64 draws_with_pipeline; ///< Their pipeline translated and built by Metal
        u64 draws_encoded;       ///< Recorded on a render encoder
        u64 draws_converted;     ///< Of them, with indices rewritten (quads, fans, 8-bit)
        u64 texel_copies;        ///< Texel buffers read through an aligned copy
        u64 occlusion_queries;   ///< Sample counts written back
        u64 indices_rewritten_on_gpu; ///< Of the rewritten, by a kernel (GPU-written indices)
        u64 mesh_draws;          ///< Task/mesh draws through the compute emulation
        u64 mesh_groups;         ///< Their mesh workgroups
        u64 ocarina_bilinear;    ///< Ocarina upscale passes replaced by the bilinear pass
        u64 tessellated_draws;
        u64 tessellated_patches;
        u64 dispatches;
        u64 dispatches_with_pipeline;
        u64 clears;
        u64 barrier_pass_ends;   ///< Render passes ended by a fragment barrier
        u64 staging_flushes;     ///< Command buffers committed for the staging buffers they took
        u64 staging_waits;       ///< Of them, waited for (the pool was over its cap)
        u64 staging_freed;       ///< Staging buffers freed
        u64 staging_bytes;       ///< Held by the staging pool now
    };

    /// Work taken since the last call (the renderer's 10 s report).
    [[nodiscard]] WorkCounts TakeWorkCounts() noexcept {
        return {std::exchange(draws, 0), std::exchange(draws_with_pipeline, 0),
                std::exchange(draws_encoded, 0), std::exchange(draws_converted, 0),
                std::exchange(texel_copies, 0), std::exchange(occlusion_queries, 0),
                std::exchange(indices_rewritten_on_gpu, 0), std::exchange(mesh_draws, 0),
                std::exchange(mesh_groups, 0), ocarina.TakePasses(),
                std::exchange(tessellated_draws, 0), std::exchange(tessellated_patches, 0),
                std::exchange(dispatches, 0), std::exchange(dispatches_with_pipeline, 0),
                std::exchange(clears, 0), std::exchange(barrier_pass_ends, 0),
                std::exchange(staging_flushes, 0), std::exchange(staging_waits, 0),
                staging_pool.TakeFreed(), staging_pool.HeldBytes()};
    }

    struct MeshCounts {
        u64 native_draws;   ///< Task/mesh draws with Metal mesh functions
        u64 from_gpu;       ///< Of them, with the task count the GPU wrote (DrawMeshTasksFromGpu)
        u64 indirect;       ///< Of them, from guest indirect records (DrawIndirect)
        u64 emulated_draws; ///< Through the compute emulation
    };

    /// Task/mesh draws since the last call (the renderer's 10 s "Perf Metal mesh:" line).
    [[nodiscard]] MeshCounts TakeMeshCounts() noexcept {
        return {std::exchange(native_mesh_draws, 0), std::exchange(native_mesh_from_gpu, 0),
                std::exchange(native_mesh_indirect, 0), std::exchange(emulated_mesh_draws, 0)};
    }

    [[nodiscard]] PipelineCache& Pipelines() noexcept {
        return pipeline_cache;
    }

    [[nodiscard]] const TextureCacheRuntime& TextureRuntime() const noexcept {
        return texture_cache_runtime;
    }

    [[nodiscard]] const BufferCacheRuntime& BufferRuntime() const noexcept {
        return buffer_cache_runtime;
    }

private:
    /// What a tessellation pipeline's vertex kernel runs over: vertices from `first` or, with an
    /// index_size (2 or 4; 8-bit indices are widened to 4), `count` indices from first_index;
    /// and instances. PrepareDraw finds index_buffer and index_offset.
    struct TessellationDraw {
        u32 first;
        u32 count;
        u32 first_instance;
        u32 instances;
        u32 index_size;
        bool widen_indices;
        u32 first_index;
        BufferHandle index_buffer;
        u64 index_offset;
    };

    /// Before a draw, clear or dispatch: commits the command buffer being recorded once the
    /// staging buffers it took pass the pool's mark, and waits for it when the pool is over its
    /// cap (EDEN_NX2_METAL_STAGING_BOUND).
    void FlushWork();

    /// Binds everything the draw needs and opens its render pass; false to skip it
    /// (prepare_refusal says why, empty for a draw of nothing).
    bool PrepareDraw(GraphicsPipeline& pipeline, bool is_indexed, u32 mesh_workgroups = 0,
                     TessellationDraw* tess_draw = nullptr);

    /// Draw's work ahead of the pipeline: a vertex count the GPU wrote is read into the draw
    /// state, a draw past 16M vertices is refused, Bananza's reset draw is done as writes. False
    /// when the draw is over.
    bool BeginDraw(bool is_indexed, u32 instance_count);

    /// Vulkan's TryNx2IndirectReset; true when the draw was done as writes to guest memory.
    bool TryIndirectReset(bool is_indexed, u32 instance_count);

    /// Vulkan's TrackOotTonemappedImage: render target 0 of a tonemap draw is what replaces the
    /// anti-aliasing network's output (oot_network.h).
    void TrackOotTonemappedImage();

    /// Whether Ocarina's upscale pass goes to MetalFX temporal upscaling: the game, the
    /// upscaler setting (MetalFX Temporal) or EDEN_NX2_DLSS_REPLACEMENT, and the GPU.
    [[nodiscard]] bool OcarinaTemporalEnabled();

    /// Vulkan's TrackOcarinaCamera: the camera matrices of a draw into the scene's targets.
    void TrackOcarinaCamera();

    /// Vulkan's TryOcarinaUpscale without `replace`: the scene depth its reader pass names.
    void TrackOcarinaDepth();

    /// A texture descriptor of the compute engine's table, if mapped and within its limit.
    [[nodiscard]] std::optional<Tegra::Texture::TICEntry> ReadComputeTic(u32 index);

    /// Vulkan's TryOcarinaUpscale: the upscale pass (views: the scene, then the target) replaced
    /// by OcarinaTemporal; false when the game's pass (or the bilinear one) must run.
    bool TryOcarinaTemporal(u64 shader_hash, std::span<const VideoCommon::ImageViewInOut> views);

    /// Counts a replaced or declined upscale pass and logs the summary every 600 (Vulkan's
    /// CountOcarinaUpscale).
    void CountOcarinaTemporal(bool replaced, const char* reason = nullptr);

    /// Counts the draw PrepareDraw skipped.
    void NoteUnprepared();

    /// Whether the draw's indices are rewritten so flat inputs take the last vertex, as the
    /// game asks (EDEN_NX2_METAL_PROVOKING_LAST=1, pipelines with flat inputs).
    [[nodiscard]] bool ProvokingLastWanted(const GraphicsPipeline& pipeline, bool is_indexed,
                                           u32 count, u32 index_size, bool restart);

    /// Counts a draw not done, and logs the cause with the draw's state the first time.
    void RefuseDraw(std::string_view cause);

    /// The programs, topology and targets of the draw at hand, for the log.
    [[nodiscard]] std::string DescribeDraw() const;

    /// Sets the rescaling_words bits of a stage's texture and image descriptors (views from the
    /// stage's first), counting on from texture_bit and image_bit.
    void AddRescalingBits(const Shader::Info& info,
                          std::span<const VideoCommon::ImageViewInOut> views, u32& texture_bit,
                          u32& image_bit);

    /// Copies written texel buffers read through aligned copies back (ends the open encoder).
    void FinishTexelWritebacks();

    /// Encodes the GPU rewrite of index_rewrite (fills its output; left empty when it can't).
    /// ahead: before the segment, leaving the open render pass as it is.
    void RewriteIndices(bool ahead);

    /// The mesh threadgroups of a task/mesh draw: a count on the CPU, or (natively drawn only)
    /// MTLDispatchThreadgroupsIndirectArguments in `buffer`, `draws` of them `stride` bytes apart.
    /// With mesh_indirect.wanted, PrepareDraw writes them and they replace buffer and offset.
    struct MeshGroups {
        u32 count{};
        BufferHandle buffer{};
        u64 offset{};
        u32 draws{1};
        u32 stride{3 * sizeof(u32)};
    };

    /// A task/mesh draw: natively while the pipeline has Metal mesh functions, else through the
    /// emulation (a CPU count only). False when nothing was encoded because neither can draw it
    /// that way (DrawMeshTasksFromGpu's caller then reads the count).
    bool DrawMesh(GraphicsPipeline& pipeline, bool is_indexed, const MeshGroups& groups);

    /// DrawIndirect of a task/mesh pipeline: natively, the guest's records rewritten on the GPU
    /// into Metal's threadgroup counts; emulated, the records read from guest memory.
    void DrawMeshIndirect(GraphicsPipeline& pipeline);

    /// Encodes mesh_indirect's rewrite of guest draw records into threadgroup counts (fills its
    /// output; left empty when the kernel can't be made). ahead: as RewriteIndices.
    void ConvertMeshIndirect(bool ahead);

    /// Encodes the widening of a sanitised layout's 10:10:10:2 integer source (VertexSource::
    /// unpack) into unpacked_vertices[slot]; left empty (zeros are bound) when it can't be made.
    /// ahead: as RewriteIndices.
    void UnpackVertices(u32 slot, const VertexSource& source, bool ahead);

    /// Encodes the copies of the attributes VertexSources gathered into gathered_vertices (left
    /// empty, zeros bound, when the kernel can't be made). ahead: as RewriteIndices.
    void GatherVertices(const VertexSources& sources, bool ahead);

    /// Binds mesh_buffers to the program's slots of the emulation's bindings.
    void BindMeshBuffers(const StageProgram& program, ShaderStage metal_stage,
                         const MeshEmulation& mesh);

    struct MeshArena {
        Object buffer;
        Object counter;
        u8* counter_contents{};
        u64 size{};
        u64 tick{};
        bool counted{}; ///< the counter of its last command buffer was read
    };

    /// The output arena of the command buffer being recorded (task programs' mesh output).
    [[nodiscard]] MeshArena& AcquireMeshArena();

    /// texture_buffer views of a stage's texel buffer bindings, in order (null where none can be
    /// made). Ranges at offsets Metal can't view are read through an aligned copy.
    [[nodiscard]] std::vector<TextureHandle> PrepareTexelBuffers(std::span<const Binding> bindings,
                                                                 const Shader::Info& info);

    /// Binds a stage's buffers (the buffer cache's pending bindings for it), textures, samplers,
    /// images and push constants into the slots the translator gave them. `views` and
    /// samplers start at the stage's first.
    void BindStageResources(StageProgram& program, ShaderStage metal_stage,
                            std::span<const Binding> pending,
                            std::span<const TextureHandle> texel_buffer_views,
                            std::span<const VideoCommon::ImageViewInOut> views,
                            std::span<const VideoCommon::SamplerId> samplers, u32 render_width,
                            u32 render_height);

    /// Sets the open render pass's occlusion counting for the next draw.
    void CountOcclusion();

    [[nodiscard]] u32 AllocateVisibilitySlot();

    void QueryFallback(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                       VideoCommon::QueryPropertiesFlags flags, u32 payload);

    /// Writes a query result to the device address its report resolved to (as Vulkan's query
    /// cache does): `wide` writes the 64-bit value and the timestamp after it.
    void WriteQueryResult(DAddr address, bool wide, u64 value);
    void NoteUnmappedReport(GPUVAddr gpu_addr, VideoCommon::QueryType type);

    /// The inline write itself: guest memory, then the caches (AccelerateInlineToMemory adds
    /// the GPU-sized dispatches' copies).
    void InlineToMemory(GPUVAddr address, size_t copy_size, std::span<const u8> memory);

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    const Device& device;
    Scheduler& scheduler;

    StagingBufferPool staging_pool;
    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    PipelineCache pipeline_cache;
    OcarinaUpscale ocarina;
    OcarinaTemporal ocarina_temporal;
    QueryCache query_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;
    /// Guest writes that found the buffer cache's lock held (EDEN_NX2_WRITE_HANDOFF).
    VideoCommon::Nx2WriteHandoff write_handoff;

    struct TexelViewKey {
        BufferHandle buffer;
        u64 offset;
        u32 elements;
        u32 format;
        bool writable;

        bool operator==(const TexelViewKey&) const noexcept = default;
    };
    struct TexelViewHash {
        size_t operator()(const TexelViewKey& key) const noexcept {
            return std::hash<const void*>{}(key.buffer) ^ (key.offset * 0x9E3779B97F4A7C15ULL) ^
                   (u64{key.elements} << 20) ^ (u64{key.format} << 8) ^ (key.writable ? 1U : 0U);
        }
    };

    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    /// Cleared every frame: views of deleted buffers must not live on.
    std::unordered_map<TexelViewKey, Object, TexelViewHash> texel_view_cache;
    u64 texel_copies{};
    struct TexelWriteback {
        BufferHandle copy;
        u64 copy_offset;
        BufferHandle buffer;
        u64 offset;
        u64 size;
    };
    std::vector<TexelWriteback> texel_writebacks;

    /// The visibility slots of the occlusion counter since its last reset, one per render pass.
    std::vector<u32> occlusion_slots;
    u64 occlusion_pass{~u64{0}};
    u32 next_visibility_slot{};
    std::array<u64, Scheduler::VisibilitySlots> visibility_ticks{};
    u64 occlusion_queries{};
    u64 unmapped_reports{};
    /// Indirect draws whose count was read from guest memory (Metal has no count buffer).
    u64 indirect_counts_read{};
    ::Common::unordered_map<u64, Object> depth_stencil_states;
    /// What an attribute of a vertex stream without a buffer reads, made on first use.
    Object zero_vertices;

    u64 draws{};
    u64 draws_with_pipeline{};
    u64 draws_encoded{};
    u64 draws_converted{};
    u64 mesh_draws{};
    u64 mesh_groups{};
    /// Why PrepareDraw skipped the draw; empty for a draw of nothing.
    std::string_view prepare_refusal;
    /// Shader::Backend::SPIRV::RescalingLayout of the draw or dispatch being bound.
    std::array<u32, 8> rescaling_words{};

    /// The GPU index rewrite of the draw being prepared.
    struct IndexRewrite {
        bool wanted{};
        IndexConvert::RewriteParams params{};
        StagingBufferRef output;
    } index_rewrite;
    Object index_kernel_library;
    Object index_kernel;
    bool index_kernel_failed{};
    u64 indices_rewritten_on_gpu{};

    /// The buffers between the tessellation stages of the draw being prepared.
    struct TessellationBuffers {
        StagingBufferRef vertices;
        StagingBufferRef control_points;
        StagingBufferRef patch_constants;
        StagingBufferRef factors;
        u32 patches{};
        u32 output_points{};
    } tess_buffers;
    u64 tessellated_draws{};
    u64 tessellated_patches{};

    /// The mesh emulation buffers of the draw being prepared.
    struct MeshBuffers {
        StagingBufferRef output;
        StagingBufferRef payload;
        StagingBufferRef offsets;
        StagingBufferRef args;
        StagingBufferRef counter;
    } mesh_buffers;
    std::vector<MeshArena> mesh_arenas;
    /// The rewrite of the guest's indirect records of a native task/mesh draw being prepared.
    struct MeshIndirect {
        bool wanted{};
        bool has_count{}; ///< Else every record is drawn
        BufferHandle records{};
        u64 records_offset{};
        BufferHandle count{};
        u64 count_offset{};
        u32 draws{};
        u32 stride_words{};
        StagingBufferRef output;
    } mesh_indirect;
    /// The draw's widened 10:10:10:2 integer attributes, by vertex buffer slot.
    std::array<StagingBufferRef, Maxwell::NumVertexArrays> unpacked_vertices{};
    Object vertex_unpack_library;
    Object vertex_unpack_kernel;
    bool vertex_unpack_failed{};
    /// The draw's gathered attributes (VertexSources::gather_slot).
    StagingBufferRef gathered_vertices;
    Object vertex_gather_library;
    Object vertex_gather_kernel;
    bool vertex_gather_failed{};
    Object mesh_indirect_library;
    Object mesh_indirect_kernel;
    bool mesh_indirect_failed{};
    u64 native_mesh_draws{};
    u64 native_mesh_from_gpu{};
    u64 native_mesh_indirect{};
    u64 emulated_mesh_draws{};
    size_t mesh_arena_current{~size_t{}};
    u64 mesh_arena_size{u64{32} << 20};
    /// Guest index data read for draws whose indices are rewritten.
    std::vector<u8> converted_source;
    u64 dispatches{};
    u64 dispatches_with_pipeline{};
    u64 clears{};
    /// The render pass (Scheduler::RenderPassSerial) a fragment barrier waits in; 0: none.
    u64 fragment_barrier_pass{};
    /// The last render pass with a draw that writes storage.
    u64 pass_stores{};
    u64 barrier_pass_ends{};
    u64 staging_flushes{};
    u64 staging_waits{};
    VideoCommon::OotNetwork::BypassState oot_network_bypass;

    /// Frames ended (TickFrame), Vulkan's render_trace_frame for the Ocarina paths.
    u64 oot_frame{1};
    /// Vulkan's oot_motion: this frame's camera and scene depth.
    struct OcarinaMotion {
        const Tegra::MemoryManager* camera_channel{};
        const Tegra::MemoryManager* depth_channel{};
        u64 camera_frame{};
        u64 depth_frame{};
        GPUVAddr depth_address{};
        u32 depth_tic{};
        std::array<u32, 2> extent{};
        /// The scene targets' size: wider than the extent at a render scale the game pads.
        std::array<u32, 2> image_extent{};
        VideoCommon::OcarinaMatrix current{};
        VideoCommon::OcarinaMatrix previous{};

        [[nodiscard]] bool SceneSized(u32 width, u32 height) const noexcept {
            return (width == extent[0] || width == image_extent[0]) &&
                   (height == extent[1] || height == image_extent[1]);
        }
    } oot_motion;
    const char* oot_decline{};
    u64 oot_decline_repeats{};
    u64 oot_replaced{};
    u64 oot_declined{};
    std::array<std::pair<const char*, u64>, 8> oot_decline_reasons{};
};

} // namespace Metal
