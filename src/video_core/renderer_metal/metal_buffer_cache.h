// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The buffer cache runtime (VideoCommon::BufferCache<BufferCacheParams>) in the shape of
// Vulkan's: bindings are appended in the order the pipeline declares them (Bindings), and
// the pipeline sets them on its encoder at the draw. Plain C++.

#pragma once

#include <array>
#include <span>
#include <vector>

#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/surface.h"

namespace Metal {

class BufferCacheRuntime;
class Device;

class Buffer : public VideoCommon::BufferBase {
public:
    explicit Buffer(BufferCacheRuntime&, VideoCommon::NullBufferParams null_params);
    explicit Buffer(BufferCacheRuntime& runtime, VAddr cpu_addr_, u64 size_bytes_,
                    bool sparse_compatible_);
    ~Buffer();

    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    [[nodiscard]] BufferHandle Handle() const noexcept {
        return buffer.Get();
    }

    operator BufferHandle() const noexcept {
        return buffer.Get();
    }

    void MarkUsage(u64, u64) noexcept {}

private:
    BufferCacheRuntime* runtime{};
    Object buffer;
    u64 bytes{};
};

/// One resource the next draw or dispatch binds, in binding order.
struct Binding {
    enum class Kind : u8 { Buffer, TexelBuffer };
    Kind kind{Kind::Buffer};
    BufferHandle buffer{};
    u32 offset{};
    u32 size{};
    VideoCore::Surface::PixelFormat format{};
};

class BufferCacheRuntime {
    friend Buffer;

    using PrimitiveTopology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;

public:
    static constexpr u32 Nx2MeshMirrorSlots = 4096;
    static constexpr u32 Nx2MeshMirrorSlotSize = 3 * sizeof(u32);

    explicit BufferCacheRuntime(const Device& device, Scheduler& scheduler,
                                StagingBufferPool& staging_pool);

    void TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept;

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return scheduler.CurrentTick();
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return scheduler.IsFree(tick);
    }

    void Wait(u64 tick) {
        scheduler.Wait(tick);
    }

    /// Metal records on the GPU thread only, so this waits as Wait does.
    void WaitSubmittedTick(u64 tick, u64 writer = 0) {
        scheduler.Wait(tick);
    }

    void Finish() {
        scheduler.Finish();
    }

    [[nodiscard]] u64 GetDeviceLocalMemory() const;

    [[nodiscard]] u64 GetDeviceMemoryUsage() const noexcept {
        return allocated_bytes;
    }

    [[nodiscard]] bool CanReportMemoryUsage() const noexcept {
        return true;
    }

    [[nodiscard]] u64 GetBufferAllocationTail() const noexcept {
        return 0;
    }

    /// Satisfies Metal's constant buffer offset rule on every Apple, Intel and AMD GPU.
    [[nodiscard]] u32 GetUniformBufferAlignment() const noexcept {
        return 256;
    }

    [[nodiscard]] u32 GetStorageBufferAlignment() const noexcept {
        return 16;
    }

    [[nodiscard]] u64 GetMaxStorageBufferRange() const noexcept {
        return u64{1} << 30;
    }

    [[nodiscard]] StagingBufferRef UploadStagingBuffer(size_t size);

    [[nodiscard]] StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    [[nodiscard]] bool CanReorderUpload(const Buffer&,
                                        std::span<const VideoCommon::BufferCopy>) const noexcept {
        return false;
    }

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void PreCopyBarrier() {}

    void CopyBuffer(BufferHandle dst_buffer, BufferHandle src_buffer,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                    bool can_reorder_upload = false);

    void PostCopyBarrier() {}

    void ClearBuffer(BufferHandle dest_buffer, u32 offset, size_t size, u32 value);

    [[nodiscard]] BufferHandle Nx2MeshMirror();

    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 num_indices,
                         u32 base_vertex, BufferHandle buffer, u32 offset, u32 size);

    void BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count);

    void BindVertexBuffer(u32 index, BufferHandle buffer, u32 offset, u32 size, u32 stride);

    void BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    void BindTransformFeedbackBuffer(u32 index, BufferHandle buffer, u32 offset, u32 size);

    void BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    /// A stream slice of exactly `size` bytes, bound next.
    std::span<u8> BindMappedUniformBuffer(size_t stage, u32 binding_index, u32 size);

    void BindUniformBuffer(const Buffer& buffer, u32 offset, u32 size) {
        bindings.push_back({Binding::Kind::Buffer, buffer.Handle(), offset, size, {}});
    }

    void BindStorageBuffer(const Buffer& buffer, u32 offset, u32 size, bool) {
        bindings.push_back({Binding::Kind::Buffer, buffer.Handle(), offset, size, {}});
    }

    void BindTextureBuffer(Buffer& buffer, u32 offset, u32 size,
                           VideoCore::Surface::PixelFormat format) {
        bindings.push_back({Binding::Kind::TexelBuffer, buffer.Handle(), offset, size, format});
    }

    /// The bindings appended since the last call, for the pipeline at the draw.
    [[nodiscard]] std::vector<Binding>& PendingBindings() noexcept {
        return bindings;
    }

    struct IndexState {
        BufferHandle buffer{};
        u32 offset{};
        u32 size{};
        u32 count{};
        u32 base_vertex{};
        IndexFormat format{};
        PrimitiveTopology topology{};
        bool quads{}; ///< Quads drawn as triangles (BindQuadIndexBuffer)
        u32 quad_first{};
    };

    struct VertexState {
        BufferHandle buffer{};
        u32 offset{};
        u32 size{};
        u32 stride{};
    };

    [[nodiscard]] const IndexState& Index() const noexcept {
        return index_state;
    }

    [[nodiscard]] const std::array<VertexState, VideoCommon::NUM_VERTEX_BUFFERS>& Vertices()
        const noexcept {
        return vertex_states;
    }

private:
    const Device& device;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;

    std::vector<Binding> bindings;
    IndexState index_state{};
    std::array<VertexState, VideoCommon::NUM_VERTEX_BUFFERS> vertex_states{};
    std::array<VertexState, VideoCommon::NUM_TRANSFORM_FEEDBACK_BUFFERS> feedback_states{};

    Object nx2_mesh_mirror;
    Object clear_pattern;
    u64 clear_pattern_size{};
    u64 allocated_bytes{};
};

struct BufferCacheParams {
    using Runtime = Metal::BufferCacheRuntime;
    using Buffer = Metal::Buffer;
    using Async_Buffer = Metal::StagingBufferRef;
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using BufferCache = VideoCommon::BufferCache<BufferCacheParams>;

} // namespace Metal
