// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The buffer cache runtime (VideoCommon::BufferCache<BufferCacheParams>) in the shape of
// Vulkan's: bindings are appended in the order the pipeline declares them (Bindings), and
// the pipeline sets them on its encoder at the draw. Plain C++.

#pragma once

#include <array>
#include <span>
#include <unordered_map>
#include <vector>

#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/buffer_cache/usage_tracker.h"
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

    /// The bytes are read or written by what the scheduler's segment records in order: an
    /// upload of them does not go ahead of it (EDEN_NX2_METAL_AHEAD).
    void MarkUsage(u64 offset, u64 size) noexcept;

    [[nodiscard]] bool IsRegionUsed(u64 offset, u64 size) const noexcept;

private:
    BufferCacheRuntime* runtime{};
    Object buffer;
    u64 bytes{};
    VideoCommon::UsageTracker tracker;
    /// The scheduler segment the tracker holds the uses of.
    u64 usage_serial{};
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

    /// Callers outside the GPU thread (the list reads of BufferCache's Nx2 paths, with the GPU
    /// split in two threads) wait for a committed tick through the scheduler's completion
    /// signal: Scheduler::Wait commits and walks the command buffers in flight, which only the
    /// recording thread may. A tick not committed yet can only be the recording thread's own.
    void WaitSubmittedTick(u64 tick, u64 writer = 0) {
        if (scheduler.IsSubmitted(tick)) {
            scheduler.WaitSubmitted(tick);
        } else {
            scheduler.Wait(tick);
        }
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

    /// Whether the upload may go ahead of the segment: nothing recorded in it uses the bytes.
    [[nodiscard]] bool CanReorderUpload(
        const Buffer& buffer, std::span<const VideoCommon::BufferCopy> copies) const noexcept;

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

    /// A storage buffer whose guest range spans several mappings (BufferCache's multi-range
    /// bindings, EDEN_NX2_METAL_MULTI_RANGE): Metal binds one buffer a slot and has no sparse
    /// aliasing here, so a read-only binding gets its segments gathered into one buffer, as
    /// Vulkan's gathered fallback; a written one stays on the plain path, as on Vulkan without
    /// sparse buffers.
    [[nodiscard]] bool PrefersSparseSources() const noexcept {
        return false;
    }

    void ResetMultiRange() noexcept {
        multi_range_sources.clear();
        multi_range_total = 0;
    }

    void PushMultiRangeSource(const Buffer& buffer, u32 offset, u32 size) {
        PushMultiRangeSource(buffer.Handle(), offset, size);
    }

    void PushMultiRangeSource(BufferHandle buffer, u32 offset, u32 size) {
        multi_range_sources.push_back({buffer, offset, size});
        multi_range_total += size;
    }

    /// Gathers the pushed segments into the buffer kept for `key` and binds it next.
    bool BindMultiRangeStorageBuffer(u64 key, bool is_written);

    /// Gathered buffers are refilled at every bind (BindFreshMultiRangeStorage).
    void InvalidateMultiRange(u64) noexcept {}

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
    struct MultiRangeSource {
        BufferHandle buffer{};
        u32 offset{};
        u32 size{};
    };
    struct MultiRangeBuffer {
        Object buffer;
        u64 size{};
        u64 frame{};
    };
    std::vector<MultiRangeSource> multi_range_sources;
    u64 multi_range_total{};
    std::unordered_map<u64, MultiRangeBuffer> multi_range_buffers;
    u64 frame{};
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
