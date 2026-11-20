// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "common/alignment.h"
#include "video_core/gpu_split.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_resources.h"

namespace Metal {

namespace {
bool SharedBuffersEnabled(const Device& device) {
    static const bool force_private = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_PRIVATE_BUFFERS");
        return value != nullptr && value[0] == '1';
    }();
    return !force_private && DeviceAppleFamily(device) >= 7;
}
} // Anonymous namespace

Buffer::Buffer(BufferCacheRuntime&, VideoCommon::NullBufferParams null_params)
    : VideoCommon::BufferBase(null_params), tracker{0} {}

Buffer::Buffer(BufferCacheRuntime& runtime_, VAddr cpu_addr_, u64 size_bytes_, bool)
    : VideoCommon::BufferBase(cpu_addr_, size_bytes_), runtime{&runtime_},
      bytes{Common::AlignUp((std::max)(size_bytes_, u64{16}), 16) + 65536},
      tracker{size_bytes_} {
    buffer = CreateBuffer(runtime_.device, bytes, runtime_.shared_buffers, &host_ptr);
    freshly_zeroed_ = buffer.Get() != nullptr;
    runtime->allocated_bytes += bytes;
}

Buffer::Buffer(Buffer&& other) noexcept
    : VideoCommon::BufferBase(std::move(other)),
      runtime{std::exchange(other.runtime, nullptr)},
      buffer{std::move(other.buffer)},
      host_ptr{std::exchange(other.host_ptr, nullptr)},
      bytes{std::exchange(other.bytes, 0)},
      tracker{std::move(other.tracker)},
      usage_serial{std::exchange(other.usage_serial, 0)},
      last_usage_tick{std::exchange(other.last_usage_tick, 0)},
      freshly_zeroed_{std::exchange(other.freshly_zeroed_, false)} {
    if (runtime != nullptr && buffer) {
        runtime->RelocateBuffer(&other, this);
    }
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (buffer && runtime != nullptr) {
        runtime->RelocateBuffer(this, nullptr);
        if (runtime->write_hook != nullptr) {
            runtime->write_hook(runtime->write_hook_user, buffer.Get(), 0, ~u64{0});
        }
        runtime->allocated_bytes -= bytes;
    }
    VideoCommon::BufferBase::operator=(std::move(other));
    runtime = std::exchange(other.runtime, nullptr);
    buffer = std::move(other.buffer);
    host_ptr = std::exchange(other.host_ptr, nullptr);
    bytes = std::exchange(other.bytes, 0);
    tracker = std::move(other.tracker);
    usage_serial = std::exchange(other.usage_serial, 0);
    last_usage_tick = std::exchange(other.last_usage_tick, 0);
    freshly_zeroed_ = std::exchange(other.freshly_zeroed_, false);
    if (runtime != nullptr && buffer) {
        runtime->RelocateBuffer(&other, this);
    }
    return *this;
}

Buffer::~Buffer() {
    if (buffer && runtime != nullptr) {
        runtime->RelocateBuffer(this, nullptr);
        if (runtime->write_hook != nullptr) {
            runtime->write_hook(runtime->write_hook_user, buffer.Get(), 0, ~u64{0});
        }
        runtime->allocated_bytes -= bytes;
    }
}

void Buffer::MarkUsage(u64 offset, u64 size) noexcept {
    if (runtime == nullptr) {
        return;
    }
    last_usage_tick = runtime->scheduler.CurrentTick();
    if (const u64 serial = runtime->scheduler.SegmentSerial(); serial != usage_serial) {
        usage_serial = serial;
        tracker.Reset();
    }
    tracker.Track(offset, size);
}

bool Buffer::IsRegionUsed(u64 offset, u64 size) const noexcept {
    return runtime != nullptr && usage_serial == runtime->scheduler.SegmentSerial() &&
           tracker.IsUsed(offset, size);
}

bool Buffer::IsInFlight() const noexcept {
    if (runtime == nullptr || !buffer) {
        return false;
    }
    const u64 current = runtime->scheduler.CurrentTick();
    return last_usage_tick >= current || !runtime->scheduler.IsFree(last_usage_tick) ||
           runtime->scheduler.IsBufferActiveInOpenCommands(Handle());
}

void Buffer::RefreshTick() noexcept {
    if (runtime != nullptr) {
        last_usage_tick = runtime->scheduler.CurrentTick();
    }
}

void Buffer::NoteUsed() const noexcept {
    if (runtime != nullptr) {
        last_usage_tick = runtime->scheduler.CurrentTick();
    }
}

bool Buffer::ConsumeFreshlyZeroed(u32 offset) noexcept {
    const bool zeroed = std::exchange(freshly_zeroed_, false);
    return zeroed && offset == 0;
}

BufferCacheRuntime::BufferCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                       StagingBufferPool& staging_pool_)
    : device{device_}, scheduler{scheduler_}, staging_pool{staging_pool_},
      uniform_buffer_alignment{DeviceAppleFamily(device_) >= 7 ? 16U : 256U},
      shared_buffers{SharedBuffersEnabled(device_)} {}

void BufferCacheRuntime::TickFrame(Common::SlotVector<Buffer>&) noexcept {
    ++frame;
    if (upload_scratch.capacity() > (2ULL << 20) && frame - upload_scratch_large_frame > 60) {
        std::vector<u8>{}.swap(upload_scratch);
        upload_scratch_size = 0;
    }
    // Gathered buffers of bindings not seen for a while go.
    std::erase_if(multi_range_buffers, [this](auto& entry) {
        if (frame - entry.second.frame < 64) {
            return false;
        }
        allocated_bytes -= entry.second.size;
        return true;
    });
}

bool BufferCacheRuntime::BindMultiRangeStorageBuffer(u64 key, bool is_written) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MULTI_RANGE");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled || is_written || multi_range_sources.empty() || multi_range_total == 0 ||
        multi_range_total > GetMaxStorageBufferRange()) {
        return false;
    }
    MultiRangeBuffer& entry = multi_range_buffers[key];
    if (!entry.buffer || entry.size < multi_range_total) {
        allocated_bytes -= entry.size;
        entry.size = Common::AlignUp(multi_range_total, u64{256});
        entry.buffer = CreateBuffer(device, entry.size, false, nullptr);
        if (!entry.buffer) {
            entry.size = 0;
            return false;
        }
        allocated_bytes += entry.size;
    }
    entry.frame = frame;
    // In order in the segment: the copies land before this draw reads them and after the draws
    // before it read the last gather (Metal tracks the buffer's hazards).
    u64 dst_offset = 0;
    for (const MultiRangeSource& source : multi_range_sources) {
        const std::array copy{VideoCommon::BufferCopy{
            .src_offset = source.offset,
            .dst_offset = dst_offset,
            .size = source.size,
        }};
        CopyBuffer(entry.buffer.Get(), source.buffer, copy, false);
        dst_offset += source.size;
    }
    bindings.push_back({Binding::Kind::Buffer, is_written, entry.buffer.Get(), 0,
                        static_cast<u32>(multi_range_total), {}, nullptr});
    Coverage::NoteGuard("storage buffer gathered from several mappings");
    return true;
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    return DeviceMemoryBudget(device);
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    if (shared_buffers) {
        const size_t alloc_size = (std::max<size_t>)(size, 1);
        if (upload_scratch.size() < alloc_size) {
            upload_scratch.resize(alloc_size);
        }
        if (alloc_size > (2ULL << 20)) {
            upload_scratch_large_frame = frame;
        }
        upload_scratch_size = size;
        return StagingBufferRef{
            .buffer = &upload_scratch_tag,
            .offset = 0,
            .mapped_span = std::span<u8>(upload_scratch.data(), alloc_size),
        };
    }
    return staging_pool.RequestUpload(size, 16);
}

StagingBufferRef BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, deferred);
}

void BufferCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

void BufferCacheRuntime::WaitSubmittedTick(u64 tick, u64) {
    if (scheduler.IsFree(tick)) {
        return;
    }
    if (auto& split = VideoCommon::GpuSplit::Get();
        split.Active() && !VideoCommon::GpuSplit::OnBackThread()) {
        split.Publish();
    }
    if (scheduler.IsSubmitted(tick)) {
        scheduler.WaitSubmitted(tick);
    } else if (VideoCommon::GpuSplit::OwnsScheduler()) {
        scheduler.Wait(tick);
    } else {
        scheduler.WaitCompleted(tick);
    }
}

bool BufferCacheRuntime::CanReorderUpload(
    const Buffer& buffer, std::span<const VideoCommon::BufferCopy> copies) const noexcept {
    return scheduler.CanCopyBufferAhead() && !scheduler.UsedInOrder(buffer.Handle()) &&
           std::ranges::none_of(copies, [&](const VideoCommon::BufferCopy& copy) {
               return buffer.IsRegionUsed(copy.dst_offset, copy.size);
           });
}

void BufferCacheRuntime::CopyBuffer(BufferHandle dst_buffer, BufferHandle src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool,
                                    bool can_reorder_upload) {
    if (src_buffer == &upload_scratch_tag) {
        if (upload_scratch_size == 0 || copies.empty()) {
            return;
        }
        const StagingBufferRef staged = staging_pool.RequestUpload(upload_scratch_size, 16);
        std::memcpy(staged.mapped_span.data(), upload_scratch.data(), upload_scratch_size);
        for (const VideoCommon::BufferCopy& copy : copies) {
            if (write_hook != nullptr) {
                write_hook(write_hook_user, dst_buffer, copy.dst_offset, copy.size);
            }
            scheduler.CopyBuffer(staged.buffer, staged.offset + copy.src_offset, dst_buffer,
                                 copy.dst_offset, copy.size, can_reorder_upload);
        }
        return;
    }
    // A blit encoder runs its copies in order; Metal orders encoders on tracked resources.
    // The scheduler notes the buffers of the copies it records in order.
    for (const VideoCommon::BufferCopy& copy : copies) {
        if (write_hook != nullptr) {
            write_hook(write_hook_user, dst_buffer, copy.dst_offset, copy.size);
        }
        scheduler.CopyBuffer(src_buffer, copy.src_offset, dst_buffer, copy.dst_offset, copy.size,
                             can_reorder_upload);
    }
}

void BufferCacheRuntime::CopyBuffer(Buffer& dst_buffer, BufferHandle src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    if (src_buffer == &upload_scratch_tag) {
        if (upload_scratch_size == 0 || copies.empty()) {
            return;
        }
        if (dst_buffer.HostPtr() != nullptr && !dst_buffer.IsInFlight()) {
            u8* const dst_ptr = dst_buffer.HostPtr();
            for (const VideoCommon::BufferCopy& copy : copies) {
                if (copy.size == 0) {
                    continue;
                }
                if (write_hook != nullptr) {
                    write_hook(write_hook_user, dst_buffer.Handle(), copy.dst_offset, copy.size);
                }
                std::memcpy(dst_ptr + copy.dst_offset, upload_scratch.data() + copy.src_offset,
                            copy.size);
            }
            return;
        }
        const StagingBufferRef staged = staging_pool.RequestUpload(upload_scratch_size, 16);
        std::memcpy(staged.mapped_span.data(), upload_scratch.data(), upload_scratch_size);
        const bool ahead = can_reorder_upload && scheduler.CanCopyBufferAhead();
        dst_buffer.NoteUsed();
        if (!ahead) {
            for (const VideoCommon::BufferCopy& copy : copies) {
                dst_buffer.MarkUsage(copy.dst_offset, copy.size);
            }
        }
        for (const VideoCommon::BufferCopy& copy : copies) {
            if (write_hook != nullptr) {
                write_hook(write_hook_user, dst_buffer.Handle(), copy.dst_offset, copy.size);
            }
            scheduler.CopyBuffer(staged.buffer, staged.offset + copy.src_offset,
                                 dst_buffer.Handle(), copy.dst_offset, copy.size, ahead);
        }
        return;
    }
    const bool ahead = can_reorder_upload && scheduler.CanCopyBufferAhead();
    dst_buffer.NoteUsed();
    if (!ahead) {
        for (const VideoCommon::BufferCopy& copy : copies) {
            dst_buffer.MarkUsage(copy.dst_offset, copy.size);
        }
    }
    CopyBuffer(dst_buffer.Handle(), src_buffer, copies, barrier, ahead);
}

void BufferCacheRuntime::CopyBuffer(BufferHandle dst_buffer, Buffer& src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    const bool ahead = can_reorder_upload && scheduler.CanCopyBufferAhead();
    src_buffer.NoteUsed();
    if (!ahead) {
        for (const VideoCommon::BufferCopy& copy : copies) {
            src_buffer.MarkUsage(copy.src_offset, copy.size);
        }
    }
    CopyBuffer(dst_buffer, src_buffer.Handle(), copies, barrier, ahead);
}

void BufferCacheRuntime::CopyBuffer(Buffer& dst_buffer, Buffer& src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    const bool ahead = can_reorder_upload && scheduler.CanCopyBufferAhead();
    src_buffer.NoteUsed();
    dst_buffer.NoteUsed();
    if (!ahead) {
        for (const VideoCommon::BufferCopy& copy : copies) {
            src_buffer.MarkUsage(copy.src_offset, copy.size);
            dst_buffer.MarkUsage(copy.dst_offset, copy.size);
        }
    }
    CopyBuffer(dst_buffer.Handle(), src_buffer.Handle(), copies, barrier, ahead);
}

void BufferCacheRuntime::ClearBuffer(Buffer& dest_buffer, u32 offset, size_t size, u32 value) {
    if (dest_buffer.ConsumeFreshlyZeroed(offset) && value == 0) {
        if (write_hook != nullptr && size != 0) {
            write_hook(write_hook_user, dest_buffer.Handle(), offset, size);
        }
        return;
    }
    if (dest_buffer.HostPtr() != nullptr && !dest_buffer.IsInFlight()) {
        if (write_hook != nullptr && size != 0) {
            write_hook(write_hook_user, dest_buffer.Handle(), offset, size);
        }
        u8* const dst = dest_buffer.HostPtr() + offset;
        const u8 byte = static_cast<u8>(value);
        if (value == u32{byte} * 0x01010101U) {
            std::memset(dst, byte, size);
        } else {
            size_t at = 0;
            for (; at + sizeof(u32) <= size; at += sizeof(u32)) {
                std::memcpy(dst + at, &value, sizeof(u32));
            }
            if (at < size) {
                std::memcpy(dst + at, &value, size - at);
            }
        }
        return;
    }
    dest_buffer.NoteUsed();
    ClearBuffer(dest_buffer.Handle(), offset, size, value);
}

void BufferCacheRuntime::ClearBuffer(BufferHandle dest_buffer, u32 offset, size_t size,
                                     u32 value) {
    if (write_hook != nullptr) {
        write_hook(write_hook_user, dest_buffer, offset, size);
    }
    const u8 byte = static_cast<u8>(value);
    if (value == u32{byte} * 0x01010101U) {
        scheduler.FillBuffer(dest_buffer, offset, size, byte);
        return;
    }
    // fillBuffer takes one byte: other patterns are copied from a staging buffer.
    const StagingBufferRef pattern = staging_pool.RequestUpload(size, 16);
    for (size_t at = 0; at + sizeof(u32) <= size; at += sizeof(u32)) {
        std::memcpy(pattern.mapped_span.data() + at, &value, sizeof(u32));
    }
    scheduler.CopyBuffer(pattern.buffer, pattern.offset, dest_buffer, offset, size);
}

BufferHandle BufferCacheRuntime::Nx2MeshMirror() {
    if (!nx2_mesh_mirror) {
        nx2_mesh_mirror = CreateBuffer(device, u64{Nx2MeshMirrorSlots} * Nx2MeshMirrorSlotSize,
                                       false, nullptr);
    }
    return nx2_mesh_mirror.Get();
}

void BufferCacheRuntime::RelocateBuffer(const Buffer* from, Buffer* to) noexcept {
    for (Binding& binding : bindings) {
        if (binding.buffer_obj == from) {
            binding.buffer_obj = to;
        }
    }
    if (index_state.buffer_obj == from) {
        index_state.buffer_obj = to;
    }
    for (VertexState& state : vertex_states) {
        if (state.buffer_obj == from) {
            state.buffer_obj = to;
        }
    }
    for (VertexState& state : feedback_states) {
        if (state.buffer_obj == from) {
            state.buffer_obj = to;
        }
    }
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 num_indices, u32 base_vertex, BufferHandle buffer,
                                         u32 offset, u32 size) {
    index_state = IndexState{
        .buffer = buffer,
        .offset = offset,
        .size = size,
        .count = num_indices,
        .base_vertex = base_vertex,
        .format = index_format,
        .topology = topology,
    };
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 num_indices, u32 base_vertex, Buffer& buffer,
                                         u32 offset, u32 size) {
    buffer.NoteUsed();
    const BufferHandle handle = buffer.Handle();
    index_state = IndexState{
        .buffer = handle,
        .offset = offset,
        .size = size,
        .count = num_indices,
        .base_vertex = base_vertex,
        .format = index_format,
        .topology = topology,
        .buffer_obj = handle != nullptr ? &buffer : nullptr,
    };
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    index_state = IndexState{
        .count = count,
        .topology = topology,
        .quads = true,
        .quad_first = first,
    };
}

void BufferCacheRuntime::BindVertexBuffer(u32 index, BufferHandle buffer, u32 offset, u32 size,
                                          u32 stride) {
    if (index < vertex_states.size()) {
        vertex_states[index] = VertexState{buffer, offset, size, stride, nullptr};
    }
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& host_bindings) {
    for (size_t index = 0; index < host_bindings.buffers.size(); ++index) {
        const size_t binding = host_bindings.min_index + index;
        if (binding >= vertex_states.size()) {
            break;
        }
        Buffer* const buffer = host_bindings.buffers[index];
        if (buffer != nullptr) {
            buffer->NoteUsed();
        }
        const BufferHandle handle = buffer != nullptr ? buffer->Handle() : nullptr;
        vertex_states[binding] = VertexState{
            .buffer = handle,
            .offset = static_cast<u32>(host_bindings.offsets[index]),
            .size = static_cast<u32>(host_bindings.sizes[index]),
            .stride = static_cast<u32>(host_bindings.strides[index]),
            .buffer_obj = handle != nullptr ? buffer : nullptr,
        };
    }
}

void BufferCacheRuntime::BindTransformFeedbackBuffer(u32 index, BufferHandle buffer, u32 offset,
                                                     u32 size) {
    if (index < feedback_states.size()) {
        if (write_hook != nullptr && buffer != nullptr && size != 0) {
            write_hook(write_hook_user, buffer, offset, size);
        }
        feedback_states[index] = VertexState{buffer, offset, size, 0, nullptr};
    }
}

void BufferCacheRuntime::BindTransformFeedbackBuffers(
    VideoCommon::HostBindings<Buffer>& host_bindings) {
    for (size_t index = 0; index < host_bindings.buffers.size(); ++index) {
        if (index >= feedback_states.size()) {
            break;
        }
        Buffer* const buffer = host_bindings.buffers[index];
        if (buffer != nullptr) {
            buffer->NoteUsed();
        }
        const BufferHandle handle = buffer != nullptr ? buffer->Handle() : nullptr;
        const u32 offset = static_cast<u32>(host_bindings.offsets[index]);
        const u32 size = static_cast<u32>(host_bindings.sizes[index]);
        if (write_hook != nullptr && handle != nullptr && size != 0) {
            write_hook(write_hook_user, handle, offset, size);
        }
        feedback_states[index] = VertexState{
            .buffer = handle,
            .offset = offset,
            .size = size,
            .stride = 0,
            .buffer_obj = handle != nullptr ? buffer : nullptr,
        };
    }
}

std::span<u8> BufferCacheRuntime::BindMappedUniformBuffer(size_t, u32, u32 size) {
    const u32 alloc_size = StreamUniformAllocationSize(size);
    stream_uniform_readable_bytes = 0;
    const StagingBufferRef ref =
        staging_pool.RequestUpload(alloc_size, GetUniformBufferAlignment());
    if (alloc_size > size) {
        std::memset(ref.mapped_span.data() + size, 0, alloc_size - size);
    }
    last_stream_uniform = StreamUniformRef{
        .buffer = ref.buffer,
        .offset = static_cast<u32>(ref.offset),
    };
    RebindStreamUniform(last_stream_uniform, size);
    return alloc_size > size ? ref.mapped_span.first(size) : ref.mapped_span;
}

} // namespace Metal
