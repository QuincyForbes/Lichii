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
    gpu_address = Scheduler::BufferGpuAddress(buffer.Get());
    freshly_zeroed_ = buffer.Get() != nullptr;
    runtime->allocated_bytes += bytes;
}

Buffer::Buffer(Buffer&& other) noexcept
    : VideoCommon::BufferBase(std::move(other)),
      runtime{std::exchange(other.runtime, nullptr)},
      buffer{std::move(other.buffer)},
      gpu_address{std::exchange(other.gpu_address, 0)},
      host_ptr{std::exchange(other.host_ptr, nullptr)},
      bytes{std::exchange(other.bytes, 0)},
      tracker{std::move(other.tracker)},
      usage_serial{std::exchange(other.usage_serial, 0)},
      last_pass_serial{std::exchange(other.last_pass_serial, ~u64{0})},
      pass_interval_begin{std::exchange(other.pass_interval_begin, 0)},
      pass_interval_end{std::exchange(other.pass_interval_end, 0)},
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
    gpu_address = std::exchange(other.gpu_address, 0);
    host_ptr = std::exchange(other.host_ptr, nullptr);
    bytes = std::exchange(other.bytes, 0);
    tracker = std::move(other.tracker);
    usage_serial = std::exchange(other.usage_serial, 0);
    last_pass_serial = std::exchange(other.last_pass_serial, ~u64{0});
    pass_interval_begin = std::exchange(other.pass_interval_begin, 0);
    pass_interval_end = std::exchange(other.pass_interval_end, 0);
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
    if (size != 0 && runtime->scheduler.IsRendering()) {
        const u64 pserial = runtime->scheduler.RenderPassSerial();
        const u64 end = size > ~u64{0} - offset ? ~u64{0} : offset + size;
        if (last_pass_serial != pserial) {
            last_pass_serial = pserial;
            pass_interval_begin = offset;
            pass_interval_end = end;
        } else {
            pass_interval_begin = (std::min)(pass_interval_begin, offset);
            pass_interval_end = (std::max)(pass_interval_end, end);
        }
    }
    tracker.Track(offset, size);
}

bool Buffer::IsRegionUsed(u64 offset, u64 size) const noexcept {
    return runtime != nullptr && usage_serial == runtime->scheduler.SegmentSerial() &&
           tracker.IsUsed(offset, size);
}

bool Buffer::MayBeUsedInCurrentPass(u64 offset, u64 size) const noexcept {
    return size != 0 && runtime != nullptr && runtime->scheduler.IsRendering() &&
           last_pass_serial == runtime->scheduler.RenderPassSerial() &&
           offset < pass_interval_end &&
           pass_interval_begin < (size > ~u64{0} - offset ? ~u64{0} : offset + size);
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
    entry.frame = frame;
    const u64 cur_serial = scheduler.SegmentSerial();
    for (MultiRangeSource& source : multi_range_sources) {
        source.ahead_serial = scheduler.BufferAheadWriteSerial(source.buffer);
    }
    const bool any_written_in_order =
        std::ranges::any_of(multi_range_sources, [&](const MultiRangeSource& source) {
            return scheduler.WrittenInOrder(source.buffer, source.offset, source.size) ||
                   scheduler.UsedUntrackedInOrder(source.buffer);
        });
    const auto can_read_sources_ahead = [&]() noexcept {
        return std::ranges::all_of(multi_range_sources, [&](const MultiRangeSource& source) {
            return scheduler.CanReadBufferAhead(source.buffer, source.offset, source.size);
        });
    };
    const auto matches_cached_prefix = [&]() noexcept {
        if (!entry.buffer || entry.size < multi_range_total ||
            multi_range_sources.size() > entry.last_sources.size()) {
            return false;
        }
        const size_t last_idx = multi_range_sources.size() - 1;
        for (size_t i = 0; i < last_idx; ++i) {
            if (!(multi_range_sources[i] == entry.last_sources[i])) {
                return false;
            }
        }
        const MultiRangeSource& want_tail = multi_range_sources[last_idx];
        const MultiRangeSource& have_tail = entry.last_sources[last_idx];
        return want_tail.buffer == have_tail.buffer && want_tail.offset == have_tail.offset &&
               want_tail.ahead_serial == have_tail.ahead_serial && want_tail.size <= have_tail.size;
    };
    const u32 gathered_size = static_cast<u32>(multi_range_total);
    if (entry.segment_serial == cur_serial && !any_written_in_order && matches_cached_prefix()) {
        bindings.push_back({Binding::Kind::Buffer, is_written, entry.buffer.Get(), 0, gathered_size,
                            {}, nullptr, entry.gpu_address, gathered_size,
                            is_written ? gathered_size : 0U});
        Coverage::NoteGuard("storage buffer gathered from several mappings");
        return true;
    }
    if (entry.buffer && entry.segment_serial == cur_serial &&
        multi_range_total <= (u64{16} << 20)) {
        const StagingBufferRef scratch =
            staging_pool.Stream(static_cast<size_t>(multi_range_total), 256);
        if (scratch.buffer != nullptr && scratch.offset <= UINT32_MAX) {
            const bool gather_ahead =
                !any_written_in_order && scheduler.CanOrderAheadCopies() &&
                can_read_sources_ahead() &&
                scheduler.CanWriteBufferAhead(scratch.buffer, scratch.offset, multi_range_total);
            u64 dst_offset = scratch.offset;
            for (const MultiRangeSource& source : multi_range_sources) {
                const std::array copy{VideoCommon::BufferCopy{
                    .src_offset = source.offset,
                    .dst_offset = dst_offset,
                    .size = source.size,
                }};
                CopyBuffer(scratch.buffer, source.buffer, copy, false, gather_ahead, true, true);
                dst_offset += source.size;
            }
            bindings.push_back({Binding::Kind::Buffer, is_written, scratch.buffer,
                                static_cast<u32>(scratch.offset), gathered_size, {}, nullptr,
                                Scheduler::BufferGpuAddress(scratch.buffer), gathered_size,
                                is_written ? gathered_size : 0U});
            Coverage::NoteGuard("storage buffer gathered from several mappings");
            return true;
        }
    }
    if (!entry.buffer || entry.size < multi_range_total) {
        allocated_bytes -= entry.size;
        entry.size = Common::AlignUp(multi_range_total, u64{256});
        entry.buffer = CreateBuffer(device, entry.size, false, nullptr);
        entry.gpu_address = Scheduler::BufferGpuAddress(entry.buffer.Get());
        entry.segment_serial = 0;
        if (!entry.buffer) {
            entry.size = 0;
            return false;
        }
        allocated_bytes += entry.size;
    }
    const bool gather_ahead =
        entry.segment_serial != cur_serial && !any_written_in_order &&
        scheduler.CanOrderAheadCopies() && can_read_sources_ahead() &&
        scheduler.CanWriteBufferAhead(entry.buffer.Get(), 0, multi_range_total);
    u64 dst_offset = 0;
    for (const MultiRangeSource& source : multi_range_sources) {
        const std::array copy{VideoCommon::BufferCopy{
            .src_offset = source.offset,
            .dst_offset = dst_offset,
            .size = source.size,
        }};
        CopyBuffer(entry.buffer.Get(), source.buffer, copy, false, gather_ahead, true, true);
        dst_offset += source.size;
    }
    for (MultiRangeSource& source : multi_range_sources) {
        source.ahead_serial = scheduler.BufferAheadWriteSerial(source.buffer);
    }
    entry.segment_serial = cur_serial;
    entry.last_sources = multi_range_sources;
    bindings.push_back({Binding::Kind::Buffer, is_written, entry.buffer.Get(), 0, gathered_size, {},
                        nullptr, entry.gpu_address, gathered_size, is_written ? gathered_size : 0U});
    Coverage::NoteGuard("storage buffer gathered from several mappings");
    return true;
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    return DeviceMemoryBudget(device);
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    static const bool use_scratch = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_UPLOAD_SCRATCH");
        return value != nullptr && value[0] == '1';
    }();
    if (use_scratch && shared_buffers) {
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
    const StagingBufferRef ref = staging_pool.Request(size, deferred);
    last_download_staging = ref.buffer;
    return ref;
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

bool BufferCacheRuntime::CanDirectUpload(const Buffer& buffer) const noexcept {
    return VideoCommon::GpuSplit::OwnsScheduler() && buffer.HostPtr() != nullptr &&
           !buffer.IsInFlight();
}

bool BufferCacheRuntime::CanReorderUpload(
    const Buffer& buffer, std::span<const VideoCommon::BufferCopy> copies) const noexcept {
    const BufferHandle handle = buffer.Handle();
    if (handle == nullptr || !scheduler.CanCopyBufferAhead() ||
        scheduler.UsedUntrackedInOrder(handle)) {
        return false;
    }
    return std::ranges::none_of(copies, [&](const VideoCommon::BufferCopy& copy) {
        return buffer.IsRegionUsed(copy.dst_offset, copy.size) ||
               !scheduler.CanWriteBufferAhead(handle, copy.dst_offset, copy.size);
    });
}

bool BufferCacheRuntime::CanReorderCopy(
    const Buffer& dst_buffer, const Buffer& src_buffer,
    std::span<const VideoCommon::BufferCopy> copies) const noexcept {
    const BufferHandle src_handle = src_buffer.Handle();
    if (src_handle == nullptr || !scheduler.CanOrderAheadCopies() ||
        !CanReorderUpload(dst_buffer, copies) || scheduler.UsedUntrackedInOrder(src_handle)) {
        return false;
    }
    if (&dst_buffer == &src_buffer) {
        for (size_t i = 0; i < copies.size(); ++i) {
            const u64 s0 = copies[i].src_offset;
            const u64 s1 = s0 + copies[i].size;
            for (size_t j = 0; j < copies.size(); ++j) {
                const u64 d0 = copies[j].dst_offset;
                const u64 d1 = d0 + copies[j].size;
                if (s0 < d1 && d0 < s1) {
                    return false;
                }
            }
        }
    }
    return std::ranges::all_of(copies, [&](const VideoCommon::BufferCopy& copy) {
        return scheduler.CanReadBufferAhead(src_handle, copy.src_offset, copy.size);
    });
}

bool BufferCacheRuntime::ShouldStreamDirtyUniformInPass(const Buffer& buffer, u32 offset,
                                                        u32 size) const noexcept {
    if (!scheduler.IsRendering() || size == 0 || CanDirectUpload(buffer)) {
        return false;
    }
    // SynchronizeBuffer uploads dirty memory at 4 KiB device-page granularity.
    const u64 page_begin = Common::AlignDown(u64{offset}, u64{4096});
    const u64 page_end = (std::min)(Common::AlignUp(u64{offset} + size, u64{4096}),
                                    static_cast<u64>(buffer.SizeBytes()));
    const u64 check_offset = page_end > page_begin ? page_begin : u64{offset};
    const u64 check_size = page_end > page_begin ? (page_end - page_begin) : u64{size};
    const std::array copy{VideoCommon::BufferCopy{
        .src_offset = 0,
        .dst_offset = check_offset,
        .size = check_size,
    }};
    return !CanReorderUpload(buffer, copy);
}

bool BufferCacheRuntime::ShouldStreamDirtyComputeUniform(const Buffer& buffer, u32 offset,
                                                         u32 size) const noexcept {
    if (!scheduler.HasOpenInOrderCompute() || size == 0 || CanDirectUpload(buffer)) {
        return false;
    }
    const u64 page_begin = Common::AlignDown(u64{offset}, u64{4096});
    const u64 page_end = (std::min)(Common::AlignUp(u64{offset} + size, u64{4096}),
                                    static_cast<u64>(buffer.SizeBytes()));
    const u64 check_offset = page_end > page_begin ? page_begin : u64{offset};
    const u64 check_size = page_end > page_begin ? (page_end - page_begin) : u64{size};
    const std::array copy{VideoCommon::BufferCopy{
        .src_offset = 0,
        .dst_offset = check_offset,
        .size = check_size,
    }};
    return !CanReorderUpload(buffer, copy);
}

void BufferCacheRuntime::CopyBuffer(BufferHandle dst_buffer, BufferHandle src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool,
                                    bool can_reorder_upload, bool range_tracked,
                                    bool src_range_tracked) {
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
                                 copy.dst_offset, copy.size, can_reorder_upload, range_tracked,
                                 src_range_tracked);
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
                             can_reorder_upload, range_tracked, src_range_tracked);
    }
}

void BufferCacheRuntime::CopyBuffer(Buffer& dst_buffer, BufferHandle src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    const BufferHandle dst_handle = dst_buffer.Handle();
    const bool can_ahead =
        scheduler.CanCopyBufferAhead() && !scheduler.UsedUntrackedInOrder(dst_handle);
    if (src_buffer == &upload_scratch_tag) {
        if (upload_scratch_size == 0 || copies.empty()) {
            return;
        }
        if (CanDirectUpload(dst_buffer)) {
            u8* const dst_ptr = dst_buffer.HostPtr();
            for (const VideoCommon::BufferCopy& copy : copies) {
                if (copy.size == 0) {
                    continue;
                }
                if (write_hook != nullptr) {
                    write_hook(write_hook_user, dst_handle, copy.dst_offset, copy.size);
                }
                std::memcpy(dst_ptr + copy.dst_offset, upload_scratch.data() + copy.src_offset,
                            copy.size);
            }
            return;
        }
        const StagingBufferRef staged = staging_pool.RequestUpload(upload_scratch_size, 16);
        std::memcpy(staged.mapped_span.data(), upload_scratch.data(), upload_scratch_size);
        dst_buffer.NoteUsed();
        for (const VideoCommon::BufferCopy& copy : copies) {
            const bool ahead =
                can_ahead && !dst_buffer.IsRegionUsed(copy.dst_offset, copy.size) &&
                scheduler.CanWriteBufferAhead(dst_handle, copy.dst_offset, copy.size);
            if (!ahead) {
                if (can_ahead && scheduler.IsRendering() &&
                    !dst_buffer.MayBeUsedInCurrentPass(copy.dst_offset, copy.size)) {
                    scheduler.WantSplitBeforeCurrentPass();
                }
                dst_buffer.MarkUsage(copy.dst_offset, copy.size);
            }
            if (write_hook != nullptr) {
                write_hook(write_hook_user, dst_handle, copy.dst_offset, copy.size);
            }
            scheduler.CopyBuffer(staged.buffer, staged.offset + copy.src_offset, dst_handle,
                                 copy.dst_offset, copy.size, ahead, true);
        }
        return;
    }
    dst_buffer.NoteUsed();
    if (can_reorder_upload && can_ahead) {
        CopyBuffer(dst_handle, src_buffer, copies, barrier, true, true);
        return;
    }
    for (const VideoCommon::BufferCopy& copy : copies) {
        const bool ahead =
            can_ahead && !dst_buffer.IsRegionUsed(copy.dst_offset, copy.size) &&
            scheduler.CanWriteBufferAhead(dst_handle, copy.dst_offset, copy.size);
        if (!ahead) {
            if (can_ahead && scheduler.IsRendering() &&
                !dst_buffer.MayBeUsedInCurrentPass(copy.dst_offset, copy.size)) {
                scheduler.WantSplitBeforeCurrentPass();
            }
            dst_buffer.MarkUsage(copy.dst_offset, copy.size);
        }
        if (write_hook != nullptr) {
            write_hook(write_hook_user, dst_handle, copy.dst_offset, copy.size);
        }
        scheduler.CopyBuffer(src_buffer, copy.src_offset, dst_handle, copy.dst_offset, copy.size,
                             ahead, true);
    }
}

void BufferCacheRuntime::CopyBuffer(BufferHandle dst_buffer, Buffer& src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    if (!can_reorder_upload && dst_buffer != nullptr && dst_buffer == last_download_staging &&
        dst_buffer != nx2_mesh_mirror.Get()) {
        src_buffer.NoteUsed();
        for (const VideoCommon::BufferCopy& copy : copies) {
            src_buffer.MarkUsage(copy.src_offset, copy.size);
            if (write_hook != nullptr && copy.size != 0) {
                write_hook(write_hook_user, dst_buffer, copy.dst_offset, copy.size);
            }
            scheduler.DownloadBuffer(src_buffer.Handle(), copy.src_offset, dst_buffer,
                                     copy.dst_offset, copy.size);
        }
        return;
    }
    const bool ahead = can_reorder_upload && scheduler.CanCopyBufferAhead();
    src_buffer.NoteUsed();
    if (!ahead) {
        for (const VideoCommon::BufferCopy& copy : copies) {
            src_buffer.MarkUsage(copy.src_offset, copy.size);
        }
    }
    CopyBuffer(dst_buffer, src_buffer.Handle(), copies, barrier, ahead, false, true);
}

void BufferCacheRuntime::CopyBuffer(Buffer& dst_buffer, Buffer& src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    if (can_reorder_upload && CanDirectUpload(dst_buffer) && CanDirectUpload(src_buffer)) {
        u8* const dst_ptr = dst_buffer.HostPtr();
        const u8* const src_ptr = src_buffer.HostPtr();
        for (const VideoCommon::BufferCopy& copy : copies) {
            if (copy.size == 0) {
                continue;
            }
            if (write_hook != nullptr) {
                write_hook(write_hook_user, dst_buffer.Handle(), copy.dst_offset, copy.size);
            }
            std::memcpy(dst_ptr + copy.dst_offset, src_ptr + copy.src_offset, copy.size);
        }
        return;
    }
    const bool ahead = can_reorder_upload && scheduler.CanCopyBufferAhead();
    src_buffer.NoteUsed();
    dst_buffer.NoteUsed();
    if (!ahead) {
        const bool check_split = scheduler.CanCopyBufferAhead() && scheduler.IsRendering();
        for (const VideoCommon::BufferCopy& copy : copies) {
            if (check_split && !src_buffer.MayBeUsedInCurrentPass(copy.src_offset, copy.size) &&
                !dst_buffer.MayBeUsedInCurrentPass(copy.dst_offset, copy.size)) {
                scheduler.WantSplitBeforeCurrentPass();
            }
            src_buffer.MarkUsage(copy.src_offset, copy.size);
            dst_buffer.MarkUsage(copy.dst_offset, copy.size);
        }
    }
    CopyBuffer(dst_buffer.Handle(), src_buffer.Handle(), copies, barrier, ahead, true, true);
}

bool BufferCacheRuntime::ClearBuffer(Buffer& dest_buffer, u32 offset, size_t size, u32 value) {
    if (dest_buffer.ConsumeFreshlyZeroed(offset) && value == 0) {
        if (write_hook != nullptr && size != 0) {
            write_hook(write_hook_user, dest_buffer.Handle(), offset, size);
        }
        return false;
    }
    if (CanDirectUpload(dest_buffer)) {
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
        return false;
    }
    const BufferHandle dest_handle = dest_buffer.Handle();
    const bool ahead = !dest_buffer.IsRegionUsed(offset, size) &&
                       scheduler.CanWriteBufferAhead(dest_handle, offset, size);
    dest_buffer.NoteUsed();
    if (!ahead) {
        if (scheduler.CanCopyBufferAhead() && scheduler.IsRendering() &&
            !dest_buffer.MayBeUsedInCurrentPass(offset, size)) {
            scheduler.WantSplitBeforeCurrentPass();
        }
        dest_buffer.MarkUsage(offset, size);
    }
    if (write_hook != nullptr && size != 0) {
        write_hook(write_hook_user, dest_handle, offset, size);
    }
    scheduler.FillBufferPattern(dest_handle, offset, size, value, ahead, true);
    return !ahead;
}

void BufferCacheRuntime::ClearBuffer(BufferHandle dest_buffer, u32 offset, size_t size,
                                     u32 value) {
    if (write_hook != nullptr && size != 0) {
        write_hook(write_hook_user, dest_buffer, offset, size);
    }
    scheduler.FillBufferPattern(dest_buffer, offset, size, value, false, false);
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
    if (ref.buffer != cached_stream_buffer) {
        cached_stream_buffer = ref.buffer;
        cached_stream_gpu_address = Scheduler::BufferGpuAddress(ref.buffer);
    }
    last_stream_uniform = StreamUniformRef{
        .buffer = ref.buffer,
        .gpu_address = cached_stream_gpu_address,
        .offset = static_cast<u32>(ref.offset),
    };
    RebindStreamUniform(last_stream_uniform, size);
    return alloc_size > size ? ref.mapped_span.first(size) : ref.mapped_span;
}

} // namespace Metal
