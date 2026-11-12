// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/buffer_cache/uniform_cache_history.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/perf_report.h"
#include "common/platform_events.h"
#include "common/range_sets.inc"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/multi_range_refresh.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/churn_counters.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/host_budget.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() = default;

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (!aggressive_gc && gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            return false;
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        DeleteBuffer(buffer_id);
    }
    Common::PerfReport::Add(Common::PerfReport::BufferCollected, evicted.size());
}

template <class P>
void BufferCache<P>::TickFrame() {
    // Read once per frame: FindBuffer asked for it on every binding of every draw.
    fence_behavior_accurate = Settings::IsGPUFenceBehaviorAccurate();
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    static const bool log_bc = std::getenv("EDEN_NX2_TC_LOG") != nullptr;
    if (static u64 frames{}; log_bc && ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU, "Buffer cache: {} buffers {} MiB; largest {}", count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()), ", "));
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    AdvanceUniformCacheHistory(channel_state->uniform_cache_hits);
    AdvanceUniformCacheHistory(channel_state->uniform_cache_shots);

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // Host memory pressure (iOS): collect unused buffers for a while, without downloading
    // GPU-written ones.
    if (const u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
        pressure != seen_host_pressure) {
        seen_host_pressure = pressure;
        host_pressure_frames = 30;
    }
    const bool host_pressure = host_pressure_frames != 0;
    if (host_pressure) {
        --host_pressure_frames;
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget);
        // title 1's buffers were ~200 MiB of an 8 GB GPU's ~6.8 GB in use, yet this collector
        // ran aggressively every frame on the device-wide figure, evicting up to 64 buffers (and
        // downloading the GPU-written ones, ~5% of the GPU thread) that were recreated later.
        // Collect when the buffers themselves are large. At 1 GiB it held geometry of scenes
        // long gone: title 1's title screen grew to ~0.9 GB of buffers in ten minutes.
        // EDEN_NX2_BC_COLLECT_MIB moves the threshold (aggressive at twice it); iOS sets it from
        // the process budget, since buffers share memory with everything else there.
        static const u64 configured_bytes = [] {
            const char* value = std::getenv("EDEN_NX2_BC_COLLECT_MIB");
            const u64 mib = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
            return mib != 0 ? mib << 20
                            : (HostBudget::Simulated() ? HostBudget::SimulatedBufferCollect()
                                                       : u64{512_MiB});
        }();
        if (collect_bytes == 0) {
            collect_bytes = configured_bytes;
        }
        // iOS: past that floor, a third of the room left to the process (re-read every 30
        // frames, up to 1 GiB). Buffers the collector deletes are uploaded again in full at
        // their next use: an iPhone with 400-600 MiB of buffers collected at 207 MiB, aggressively
        // (with downloads and a wait for the GPU) every frame. Small moves are ignored.
        if (HostBudget::FollowsHeadroom() && frame_tick % 30 == 0) {
            const u64 available = HostBudget::Available(
                HostBudget::Simulated() && runtime.CanReportMemoryUsage()
                    ? runtime.GetDeviceMemoryUsage()
                    : 0);
            if (available != 0) {
                const u64 room =
                    available > HostBudget::Reserve ? available - HostBudget::Reserve : 0;
                const u64 target =
                    std::clamp(buffer_bytes + room / 3, configured_bytes,
                               (std::max)(configured_bytes, u64{1_GiB}));
                if (target + 32_MiB < collect_bytes || target > collect_bytes + 64_MiB) {
                    LOG_INFO(Common_Memory,
                             "Buffer cache with {} MiB left to the process: collects from {} MiB "
                             "({} MiB of buffers)",
                             available >> 20, target >> 20, buffer_bytes >> 20);
                    collect_bytes = target;
                }
            }
        }
        if (buffer_bytes >= collect_bytes || host_pressure) {
            RunGarbageCollector(buffer_bytes >= collect_bytes * 2);
        }
    } else {
        if (runtime.CanReportMemoryUsage()) {
            total_used_memory = runtime.GetDeviceMemoryUsage();
        }
        if (total_used_memory >= minimum_memory || host_pressure) {
            RunGarbageCollector(total_used_memory >= critical_memory);
        }
    }
    ++frame_tick;
    delayed_destruction_ring.Tick();

    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data) {
    Nx2NoteListHostWrite(device_addr, size);
    Core::g_buffer_write_pages.Mark(device_addr, size);
    NoteUniformWrite(device_addr, size);
    // The written range lies within its pages: when they hold no GPU-written data, both lookups
    // below find nothing, so one lookup answers them.
    if (Nx2Perf::On(Nx2Perf::Opt::WriteIntersections) && keep_page_gpu_data) {
        const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
        if (!gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
            if (!Nx2Perf::Verify()) {
                static const bool dirty_write_skip = [] {
                    const char* value = std::getenv("EDEN_NX2_DIRTY_WRITE_SKIP");
                    return value == nullptr || value[0] != '0';
                }();
                if (dirty_write_skip &&
                    memory_tracker.IsRegionFullyCpuModified(device_addr, size)) {
                    MarkVertexBuffersStale(device_addr, size);
                    return;
                }
                memory_tracker.MarkRegionAsCpuModified(device_addr, size);
                NoteCpuDirty(device_addr, size);
                MarkVertexBuffersStale(device_addr, size);
                return;
            }
            static std::atomic<u32> logged{};
            if (IsRegionGpuModified(device_addr, size) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: GPU-written data in {:#x}+{:#x} outside its pages",
                          device_addr, size);
            }
        }
    }
    if (IsRegionGpuModified(device_addr, size)) {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    // Only the scheduler's owner may download (it records and submits): unmaps and other writes
    // reach this from CPU and service threads too, and submitting there raced the GPU thread's
    // own submissions (crashes in Scheduler::SubmitExecution).
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data && VideoCommon::GpuSplit::OwnsScheduler() &&
        gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        // Switch 2 titles: upload the CPU's bytes into the buffer instead, as CachedWriteMemory
        // does, so the page stays GPU-owned and nothing waits for the host GPU.
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_INLINE_PAGE_WRITES");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        const bool inline_writes =
            forced >= 0 ? forced == 1 : Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
        if (inline_writes && IsRegionRegistered(device_addr, size)) {
            tmp_buffer.resize_destructive(size);
            device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);
            InlineMemoryImplementation(device_addr, size, tmp_buffer);
            return;
        }
        DownloadMemory(page_begin, page_end - page_begin);
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    NoteCpuDirty(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
        // A preflushed region's data reaches guest memory when its fence pops (PopAsyncBuffers
        // clears gpu_modified_ranges then), while fast GPU fences tell the guest at once that the
        // GPU is done. A read in between saw the previous words: Ocarina of Time meters the frame
        // on the GPU and its CPU derives the exposure from that, which then jumped between
        // values and washed the picture out. Such a read flushes on demand instead.
        area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                               device_addr_end_aligned - device_addr_start_aligned);
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::PreflushedRead(!area->preemtive);
        }
        // The core consumed this trap, or keeps the area: the next GPU write must arm again.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
        return area;
    }
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    // A preflushable mark persists and downloads later GPU writes at every fence. Native titles
    // keep read traps and flush on demand unless EDEN_NX2_PREFLUSH=1, which nx2_switches.cpp sets
    // by default: preflushing Ocarina of Time's ~170 CPU reads a second of GPU-written pages was
    // worth ~3 fps. A third of its loads into Castle Town then rendered overexposed and tinted
    // (stale exposure data), which the on-demand flush above is for.
    static const std::optional<bool> forced_preflush = []() -> std::optional<bool> {
        const char* value = std::getenv("EDEN_NX2_PREFLUSH");
        if (value != nullptr) {
            return value[0] != '0';
        }
        return std::nullopt;
    }();
    const bool preflush =
        forced_preflush.value_or(!Tegra::g_nx2_ampere.load(std::memory_order_relaxed));
    if (!area->preemtive) {
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Marked(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned, preflush);
        }
        if (preflush) {
            memory_tracker.MarkRegionAsPreflushable(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        } else {
            if constexpr (USE_MEMORY_MAPS) {
                cpu_read_ranges.Add(area->start_address, area->end_address - area->start_address);
                download_cpu_read_ranges = true;
            }
            // The core consumed this trap before GetFlushArea. Forget it so the next GPU write
            // really rearms the core instead of being skipped by the deduplication cache.
            read_trapped_ranges.Subtract(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        }
    } else {
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    if constexpr (USE_MEMORY_MAPS) {
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
            cpu_read_ranges.ForEach([&](DAddr lower, DAddr upper) {
                gpu_modified_ranges.ForEachInRange(lower, upper - lower,
                                                   [&](DAddr start, DAddr end) {
                                                       ranges.emplace_back(start, end - start);
                                                   });
            });
            DownloadRangesMemory(ranges);
            return;
        }
    }
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        const DAddr start = (std::max)(device_addr, buffer.CpuAddr());
        const DAddr end = (std::min)(device_addr + size, buffer.CpuAddr() + buffer.SizeBytes());
        if (start < end) {
            DownloadBufferMemory(buffer, start, end - start);
        }
    });
}

template <class P>
void BufferCache<P>::DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges) {
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    boost::container::small_vector<Download, 16> downloads;
    u64 total_size_bytes = 0;
    for (const auto& [range_addr, range_size] : ranges) {
        ForEachBufferInRange(range_addr, range_size, [&](BufferId buffer_id, Buffer& buffer) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const DAddr start = (std::max)(range_addr, buffer_addr);
            const DAddr end = (std::min)(range_addr + range_size, buffer_addr + buffer.SizeBytes());
            if (start >= end) {
                return;
            }
            memory_tracker.ForEachDownloadRangeAndClear(
                start, end - start, [&](u64 device_addr_out, u64 out_size) {
                    gpu_modified_ranges.ForEachInRange(
                        device_addr_out, out_size, [&](DAddr lower, DAddr upper) {
                            downloads.push_back({buffer_id, BufferCopy{
                                                                .src_offset = lower - buffer_addr,
                                                                .dst_offset = total_size_bytes,
                                                                .size = upper - lower,
                                                            }});
                            total_size_bytes += Common::AlignUp(upper - lower, 64);
                        });
                    ClearDownload(device_addr_out, out_size);
                    gpu_modified_ranges.Subtract(device_addr_out, out_size);
                });
        });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    // The CPU writes collected from now on store their bytes once this is in guest memory.
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const Download& download : downloads) {
        const DAddr address = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        landing_begin = (std::min)(landing_begin, address);
        landing_end = (std::max)(landing_end, address + download.copy.size);
    }
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadMemoryRanges(std::span<const std::pair<DAddr, u64>> ranges) {
    constexpr bool has_mapped = requires(Buffer& buffer) {
        buffer.Mapped();
        buffer.LastUsageTick();
    };
    if constexpr (!USE_MEMORY_MAPS) {
        for (const auto& [device_addr, size] : ranges) {
            DownloadMemory(device_addr, size);
        }
    } else {
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        std::vector<Download> staged;
        std::vector<Download> mapped;
        u64 total_size_bytes = 0;
        u64 mapped_tick = 0;
        for (const auto& [device_addr, size] : ranges) {
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(device_addr, buffer_addr);
                const DAddr end = (std::min)(device_addr + size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                bool is_mapped = false;
                if constexpr (has_mapped) {
                    is_mapped = !buffer.Mapped().empty();
                    if (is_mapped) {
                        mapped_tick = (std::max)(mapped_tick, buffer.LastUsageTick());
                    }
                }
                memory_tracker.ForEachDownloadRangeAndClear(
                    begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            device_addr_out, range_size, [&](DAddr start, DAddr range_end) {
                                const BufferCopy copy{
                                    .src_offset = start - buffer_addr,
                                    .dst_offset = total_size_bytes,
                                    .size = range_end - start,
                                };
                                if (is_mapped) {
                                    mapped.push_back({buffer_id, copy});
                                    return;
                                }
                                staged.push_back({buffer_id, copy});
                                total_size_bytes += Common::AlignUp(range_end - start, 64);
                            });
                        ClearDownload(device_addr_out, range_size);
                        gpu_modified_ranges.Subtract(device_addr_out, range_size);
                    });
            });
        }
        if (staged.empty() && mapped.empty()) {
            return;
        }
        // Once the host GPU ran the copies, inside a landing over their bytes: the CPU writes
        // collected meanwhile keep their bytes too (they were looked at before the wait).
        const auto land = [&](const Async_Buffer* staging) {
            DAddr landing_begin = ~DAddr{0};
            DAddr landing_end = 0;
            const auto cover = [&](const Download& download) {
                const DAddr address =
                    slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
                landing_begin = (std::min)(landing_begin, address);
                landing_end = (std::max)(landing_end, address + download.copy.size);
            };
            std::ranges::for_each(staged, cover);
            std::ranges::for_each(mapped, cover);
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            if (staging != nullptr) {
                for (const Download& download : staged) {
                    WriteDownloadedMemory(
                        cpu_written,
                        slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset,
                        staging->mapped_span.data() + (download.copy.dst_offset - staging->offset),
                        download.copy.size);
                }
            }
            if constexpr (has_mapped) {
                for (const Download& download : mapped) {
                    Buffer& buffer = slot_buffers[download.buffer_id];
                    WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + download.copy.src_offset,
                                          buffer.Mapped().data() + download.copy.src_offset,
                                          download.copy.size);
                }
            }
        };
        if (staged.empty()) {
            if constexpr (has_mapped) {
                runtime.Wait(mapped_tick);
            }
            land(nullptr);
        } else {
            auto staging = runtime.DownloadStagingBuffer(total_size_bytes);
            for (Download& download : staged) {
                download.copy.dst_offset += staging.offset;
                Buffer& buffer = slot_buffers[download.buffer_id];
                buffer.MarkUsage(download.copy.src_offset, download.copy.size);
                runtime.CopyBuffer(staging.buffer, buffer, std::span{&download.copy, 1}, true);
            }
            runtime.Finish();
            land(&staging);
        }
    }
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount,
                             bool inline_grid_copy) {
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);
    NoteUniformWrite(*cpu_dest_address, amount);
    Nx2NoteListHostWrite(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        if (!inline_grid_copy) {
            QueueFenceDownload(new_base_address, size);
        }
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        if (!inline_grid_copy) {
            gpu_modified_serial.fetch_add(1, std::memory_order_relaxed);
            if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
                Tegra::Nx2MarkGpuWritten(pair.first, pair.second);
                Nx2NoteListWrite(buffer_b, pair.first, pair.second);
            }
        }
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads && !inline_grid_copy) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    if (!inline_grid_copy) {
        Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
            tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
        tmp.SetAddressAndSize(*cpu_dest_address, amount);
    }
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    const size_t size = amount * sizeof(u32);
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, size);
    if (!dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dst_address, size);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);
    NoteUniformWrite(*cpu_dst_address, size);
    Nx2NoteListHostWrite(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    // Each caller records GPU work that reads or writes these bytes now (indirect dispatches, DMA
    // copies between images and buffers, query results, conditional rendering): an upload of them
    // later in this submission must run after it (CanReorderUpload).
    buffer.MarkUsage(buffer.Offset(device_addr), size);
    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::MarkUsed(DAddr device_addr, u64 size) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || size == 0 || page >= page_table.size()) {
        return;
    }
    const BufferId buffer_id = page_table[page];
    if (!buffer_id) {
        return;
    }
    Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        buffer.MarkUsage(buffer.Offset(device_addr), size);
    }
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    UniformBindMemo& memo = uniform_bind_memo[stage][index];
    Binding& current = channel_state->uniform_buffers[stage][index];
    // EDEN_NX2_EXP18 & 1: the same GPU range bound again while the mappings and the set of
    // buffers are unchanged translates to the same device range, and the buffer it resolved to
    // is still what FindBuffer returns: keep the binding with its buffer_id. The contents are
    // read at the draw as before. FindBuffer waits for writes with accurate GPU fences: keep that.
    const bool use_memo = (VideoCore::Nx2Debug::Experiment(18) & 1) != 0 &&
                          !fence_behavior_accurate;
    if (use_memo && memo.epoch == binding_epoch && memo.gpu_addr == gpu_addr &&
        memo.size == size && memo.channel == static_cast<const void*>(channel_state) &&
        current.device_addr == memo.device_addr && current.size == size &&
        memo.as_id == gpu_memory->GetID() && memo.map_serial == gpu_memory->MapSerial()) {
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        return;
    }
    // Before the translation: a map in between makes the memo miss next time.
    const u64 map_serial = gpu_memory->MapSerial();
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        current = NULL_BINDING;
        memo.epoch = 0;
        return;
    }
    current = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
    if (use_memo) {
        memo = UniformBindMemo{
            .epoch = binding_epoch,
            .map_serial = map_serial,
            .as_id = gpu_memory->GetID(),
            .channel = channel_state,
            .gpu_addr = gpu_addr,
            .device_addr = *device_addr,
            .size = size,
        };
    } else {
        memo.epoch = 0;
    }
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    uniform_bind_memo[stage][index].epoch = 0;
    auto& current = channel_state->uniform_buffers[stage][index];
    if (current.device_addr == 0 && current.size == 0 && current.buffer_id == NULL_BUFFER_ID)
        [[likely]] {
        return;
    }
    current = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    ++native_bank_draw;
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    BindHostGraphicsUniformBuffers(stage);
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        BindHostGraphicsStorageBuffers(stage);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        BindHostGraphicsTextureBuffers(stage);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers();
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers();
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    if (channel_state->enabled_uniform_buffer_masks == mask &&
        channel_state->uniform_buffer_sizes == sizes) [[likely]] {
        return;
    }
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
        channel_state->enabled_uniform_buffer_masks = mask;
    }
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) {
        return;
    }
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written,
                                               u32 indirect_offset) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        LOG_ERROR(HW_GPU, "Invalid graphics storage buffer binding: stage {}, index {}",
                  stage, ssbo_index);
        return false;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= u64{1} << ssbo_index;
    channel_state->written_storage_buffers[stage] |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    const GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
    // EDEN_NX2_GRAPHICS_SSBO_BARE=0 takes the word after a native graphics pointer as its size
    // again: much smaller bindings to synchronize per draw, for A/B runs.
    static const bool bare_pointers = [] {
        const char* const value = std::getenv("EDEN_NX2_GRAPHICS_SSBO_BARE");
        return value == nullptr || value[0] != '0';
    }();
    const bool native_pointer =
        (bare_pointers && maxwell3d->GetNativeMacroEngine().native_turing) ||
        indirect_offset != ~0U;
    channel_state->storage_buffers[stage][ssbo_index] =
        BindCachedStorageBuffer(stage, ssbo_index, ssbo_addr, cbuf_index, cbuf_offset, is_written,
                                native_pointer, indirect_offset);
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) {
        return;
    }
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    channel_state->native_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 indirect_offset,
                                              const Shader::StorageWriteExtent& write_extent) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    const GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    const bool native_pointer = kepler_compute->IsNativeMode();
    channel_state->native_compute_storage_buffers =
        (channel_state->native_compute_storage_buffers & ~(u64{1} << ssbo_index)) |
        (u64{native_pointer ? 1U : 0U} << ssbo_index);
    GPUVAddr resolved_pointer = 0;
    const Binding binding = BindCachedStorageBuffer(
        NUM_STAGES, ssbo_index, ssbo_addr, cbuf_index, cbuf_offset, is_written, native_pointer,
        indirect_offset, &resolved_pointer);
    channel_state->compute_storage_buffers[ssbo_index] = binding;

    // How much of the binding the dispatch marks GPU-written. A native pointer has no size, so the
    // binding is up to 8 MiB past it, and marking all of it defers the CPU's later writes to those
    // pages until the next fence and has fences copy the over-marked range back.
    u64 mark = binding.size;
    // EDEN_NX2_EXP1=1: mark the whole binding as before (#415's A/B).
    if (native_pointer && is_written && binding.size != 0 &&
        VideoCore::Nx2Debug::Experiment(1) == 0) {
        switch (Settings::values.native_compute_write_mark.GetValue()) {
        case Settings::NativeComputeWriteMark::Auto: {
            const auto& qmd = launch_desc;
            const std::array<u32, 3> block{qmd.block_dim_x.Value(), qmd.block_dim_y.Value(),
                                           qmd.block_dim_z.Value()};
            const std::array<u32, 3> grid =
                kepler_compute->GetIndirectComputeAddress()
                    ? std::array<u32, 3>{64U, 1U, 1U}
                    : std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                         qmd.grid_dim_z.Value()};
            if (const auto extent = Shader::ResolveStorageWriteExtent(write_extent, block, grid)) {
                // The extent counts from the pointer, the binding from the pointer aligned down.
                const GPUVAddr pointer = resolved_pointer;
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                mark = std::min<u64>(binding.size, pad + *extent);
            }
            break;
        }
        case Settings::NativeComputeWriteMark::Full:
            break;
        case Settings::NativeComputeWriteMark::Cap64K:
            mark = std::min<u64>(binding.size, 64_KiB);
            break;
        }
    }
    channel_state->compute_storage_write_marks[ssbo_index] = static_cast<u32>(mark);
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
    ++cpu_dirty_epoch;
    ++cb_write_serial;
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        if (Nx2Profile::Enabled()) {
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12; page <= (orig_device_addr + copy.size - 1) >> 12;
                 ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Copied(orig_device_addr, copy.size);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
void BufferCache<P>::ClaimCpuWrites(std::span<const std::pair<DAddr, u64>> ranges) {
    for (const auto& [device_addr, size] : ranges) {
        if (IsRegionGpuModified(device_addr, size)) {
            ClearDownload(device_addr, size);
            gpu_modified_ranges.Subtract(device_addr, size);
        }
    }
}

template <class P>
Common::RangeSet<DAddr> BufferCache<P>::PendingCpuWriteRanges() {
    Common::RangeSet<DAddr> ranges;
    if (pending_cpu_writes) {
        pending_cpu_writes(
            [&ranges](DAddr device_addr, size_t size) { ranges.Add(device_addr, size); });
    }
    return ranges;
}

template <class P>
void BufferCache<P>::WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written,
                                           DAddr device_addr, const u8* data, u64 size) {
    Core::g_buffer_write_pages.Mark(device_addr, size);
    NoteUniformWrite(device_addr, size);
    if (!cpu_written.Intersects(device_addr, size)) {
        device_memory.WriteBlockUnsafe(device_addr, data, size);
        return;
    }
    Common::RangeSet<DAddr> parts;
    parts.Add(device_addr, size);
    cpu_written.ForEachInRange(device_addr, size, [&parts](DAddr start, DAddr end) {
        parts.Subtract(start, end - start);
    });
    parts.ForEach([&](DAddr start, DAddr end) {
        device_memory.WriteBlockUnsafe(start, data + (start - device_addr), end - start);
    });
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    // Landed per copy (copies within a page of each other together), each inside a
    // VideoCore::DownloadLanding that it looks at the collected CPU writes in: with preflush a
    // fence's copies span unrelated pages, and one landing over their hull would hold guest
    // writes anywhere in it until all of them landed.
    size_t first = 0;
    while (first < downloads.size()) {
        DAddr landing_begin = static_cast<DAddr>(downloads[first].src_offset);
        DAddr landing_end = landing_begin + downloads[first].size;
        size_t last = first + 1;
        for (; last < downloads.size(); ++last) {
            const DAddr begin = static_cast<DAddr>(downloads[last].src_offset);
            const DAddr end = begin + downloads[last].size;
            if (end + Core::DEVICE_PAGESIZE < landing_begin ||
                begin > landing_end + Core::DEVICE_PAGESIZE) {
                break;
            }
            landing_begin = (std::min)(landing_begin, begin);
            landing_end = (std::max)(landing_end, end);
        }
        const VideoCore::DownloadLanding landing{landing_begin, landing_end};
        // This runs on the fencing thread while the CPU keeps writing: CPU writes collected since
        // the copy was taken are newer than it, and the GPU thread hasn't dropped them from
        // `ranges` yet.
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        for (size_t index = first; index < last; ++index) {
            const auto& copy = downloads[index];
            const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
            const u64 dst_offset = copy.dst_offset - base_offset;
            const u8* read_mapped_memory = base + dst_offset;
            // Only what no write replaced since the copy was taken: the rest is older than guest
            // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
            ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
                WriteDownloadedMemory(cpu_written, start, &read_mapped_memory[start - device_addr],
                                      end - start);
                gpu_modified_ranges.Subtract(start, end - start);
            });
        }
        first = last;
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    if (size <= 64_KiB && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        !Tegra::Nx2MayBeGpuWritten(addr, size)) {
        return false;
    }
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    return gpu_modified_ranges.Intersects(addr, size);
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if constexpr (requires { runtime.BindHostUint8IndexBuffer(std::span<const u8>{}); }) {
        if (draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
            draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
            draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
            runtime.WidensUint8IndicesOnHost()) {
            std::span<const u8> indices;
            const auto& index_buffer = draw_state.index_buffer;
            const u32 guest_size = (std::min)(
                static_cast<u32>(index_buffer.EndAddress() - index_buffer.StartAddress()),
                index_buffer.first + index_buffer.count);
            if (!draw_state.inline_index_draw_indexes.empty()) {
                indices = std::span{draw_state.inline_index_draw_indexes.data(), size};
            } else if (guest_size != 0 &&
                       !IsRegionGpuModified(channel_state->index_buffer.device_addr, guest_size)) {
                guest_uint8_indices.resize(guest_size);
                device_memory.ReadBlockUnsafe(channel_state->index_buffer.device_addr,
                                              guest_uint8_indices.data(), guest_size);
                indices = guest_uint8_indices;
            }
            if (!indices.empty()) {
                runtime.BindHostUint8IndexBuffer(indices);
                return;
            }
        }
    }
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeBuffer(buffer, channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = maxwell3d->dirty.flags;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
        // The draw reads the arguments (GetDrawIndirectBuffer, GetDrawIndirectCount).
        if (binding.buffer_id != NULL_BUFFER_ID) {
            buffer.MarkUsage(buffer.Offset(binding.device_addr), binding.size);
        }
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    // Bank 24 (root table 0, NVN2's push constants) is read from bank 17, which such games do not
    // bind: stream the table's current words as that buffer.
    if (index == 17 && maxwell3d->root_tables.loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = maxwell3d->root_tables.tables[0];
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index, std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    const bool not_gpu_modified =
        has_host_buffer && !memory_tracker.IsRegionGpuModified(device_addr, size);
    const bool use_fast_buffer =
        needs_alignment_stream ||
        (not_gpu_modified && size <= channel_state->uniform_buffer_skip_cache_size);
    if (use_fast_buffer) {
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off. EDEN_NX2_EXP17=1: any stage or slot's copy of
            // the same range is bound again too.
            if (VideoCommon::UniformStreamReuseEnabled()) {
                streamed = VideoCommon::BindSmallUniformReusing(
                    runtime, device_memory, stage, index, binding_index, device_addr, size,
                    VideoCore::Nx2Debug::Experiment(17) != 0, cb_write_serial,
                    UboPageGen(device_addr, size), [this, device_addr, size](u64& gen) {
                        return UboRangeClean(device_addr, size, gen);
                    });
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        }
        return;
    }
    // Classic cached path
    constexpr u64 ocarina_of_time = 0x0400486024DEC000ULL;
    const bool bank_recheck = program_id != ocarina_of_time || Nx2Perf::Verify();
    const bool native_bank = bank_recheck && not_gpu_modified &&
                             (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000 &&
                             size != 0 && gpu_memory != nullptr && gpu_memory->UsesNativeTiling();
    if (const int tracked_mode = native_bank ? VideoCore::Nx2Debug::Experiment(9) : 0;
        tracked_mode != 0) {
        CheckNativeBankPagesTracked(device_addr, size, tracked_mode);
    } else if (native_bank) {
        // The bank size is only an upper limit: stop at the end of the mapped memory.
        u64 mapped_size = 0;
        while (mapped_size < size &&
               device_memory.template GetPointer<u8>(device_addr + mapped_size) != nullptr) {
            mapped_size += Core::DEVICE_PAGESIZE - ((device_addr + mapped_size) & Core::DEVICE_PAGEMASK);
        }
        const u64 check_size = (std::min<u64>)(mapped_size, size);
        // Banks slide through larger buffers, a new address per draw: compare each page with
        // its last contents and mark only the pages that changed.
        const DAddr first_page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = device_addr + check_size;
        const int cost_mode = VideoCore::Nx2Debug::Experiment(4);
        const bool trim = (cost_mode & 1) != 0;
        const bool once_per_draw = (cost_mode & 2) != 0;
        auto& cost = native_bank_stats;
        if (cost_mode != 0) {
            ++cost.binds;
        }
        for (DAddr page = first_page; page < end; page += Core::DEVICE_PAGESIZE) {
            const u8* const guest = device_memory.template GetPointer<u8>(page);
            if (guest == nullptr) {
                continue;
            }
            const u32 begin =
                trim ? static_cast<u32>((std::max)(page, device_addr) - page) : 0;
            const u32 stop = trim ? static_cast<u32>((std::min<DAddr>)(page + Core::DEVICE_PAGESIZE,
                                                                      end) - page)
                                  : static_cast<u32>(Core::DEVICE_PAGESIZE);
            auto [it, inserted] = InsertNativeBankSnapshot(page);
            NativeBankPage& snapshot = it->second;
            snapshot.checked_frame = frame_tick;
            if (once_per_draw && !inserted && snapshot.checked_draw == native_bank_draw &&
                snapshot.checked_begin <= begin && stop <= snapshot.checked_end) {
                ++cost.pages_skipped;
                continue;
            }
            const bool same = !inserted && std::memcmp(snapshot.words.data() + begin,
                                                       guest + begin, stop - begin) == 0;
            if (cost_mode != 0 && !inserted) {
                ++cost.pages_compared;
                cost.bytes_compared += stop - begin;
                cost.bytes_trimmed += Core::DEVICE_PAGESIZE - (stop - begin);
            }
            if (once_per_draw && same) {
                if (snapshot.checked_draw != native_bank_draw ||
                    begin > snapshot.checked_end || stop < snapshot.checked_begin) {
                    if (snapshot.checked_draw != native_bank_draw ||
                        stop - begin > snapshot.checked_end - snapshot.checked_begin) {
                        snapshot.checked_begin = begin;
                        snapshot.checked_end = stop;
                    }
                } else {
                    snapshot.checked_begin = (std::min)(snapshot.checked_begin, begin);
                    snapshot.checked_end = (std::max)(snapshot.checked_end, stop);
                }
                snapshot.checked_draw = native_bank_draw;
            }
            if (same) {
                continue;
            }
            std::memcpy(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE);
            memory_tracker.MarkRegionAsCpuModified(page, Core::DEVICE_PAGESIZE);
            NoteCpuDirty(page, Core::DEVICE_PAGESIZE);
            if (once_per_draw) {
                snapshot.checked_draw = native_bank_draw;
                snapshot.checked_begin = 0;
                snapshot.checked_end = static_cast<u32>(Core::DEVICE_PAGESIZE);
            }
            if (cost_mode != 0 && !inserted) {
                ++cost.pages_changed;
            }
        }
    }
    if (SynchronizeBuffer(buffer, device_addr, size)) {
        ++channel_state->uniform_cache_hits[0];
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    buffer.MarkUsage(offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
void BufferCache<P>::RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        VideoCommon::RefreshMultiRangeSegments(
            std::span{pool}, [this](BufferId id) { return slot_buffers.contains(id); },
            [this](DAddr address, u32 size) {
                return FindBuffer(address, size, runtime.PrefersSparseSources());
            });
    }
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        if (gpu_memory->MaxContinuousRange(binding.gpu_addr, binding.size) >= binding.size) {
            return;
        }
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool,
                                           u64 write_mark_limit) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first > pool.size() ||
            binding.segment_count > pool.size() - binding.segment_first) {
            return false;
        }
        const auto segments = pool.subspan(binding.segment_first, binding.segment_count);
        if (!MultiRangeSegmentsLive(segments,
                                    [this](BufferId id) { return slot_buffers.contains(id); })) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            SynchronizeBuffer(buffer, segment.device_addr, segment.size);
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written && write_mark_limit != 0) {
                // write_mark_limit: the native write mark (BindComputeStorageBuffer), spread over
                // the segments in order.
                const u32 mark = static_cast<u32>(std::min<u64>(segment.size, write_mark_limit));
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, mark);
                if (write_mark_limit != ~u64{0}) {
                    write_mark_limit -= mark;
                }
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return BindFreshMultiRangeStorage(runtime, key, is_written);
    } else {
        return false;
    }
}

template <class P>
std::pair<typename std::unordered_map<DAddr, typename BufferCache<P>::NativeBankPage>::iterator,
          bool>
BufferCache<P>::InsertNativeBankSnapshot(DAddr page) {
    auto [it, inserted] = native_indirect_page_snapshots.try_emplace(page);
    if (!inserted) {
        return {it, false};
    }
    if (native_indirect_page_ring_count < MAX_NATIVE_BANK_SNAPSHOTS) {
        native_indirect_page_ring[native_indirect_page_ring_count++] = page;
        return {it, true};
    }
    size_t victim_slot = native_indirect_page_clock;
    u64 oldest_frame = ~u64{0};
    for (size_t step = 0; step < 8; ++step) {
        const size_t slot = (native_indirect_page_clock + step) & (MAX_NATIVE_BANK_SNAPSHOTS - 1);
        const auto cand = native_indirect_page_snapshots.find(native_indirect_page_ring[slot]);
        if (cand == native_indirect_page_snapshots.end() ||
            cand->second.checked_frame != frame_tick) {
            victim_slot = slot;
            break;
        }
        if (cand->second.checked_frame < oldest_frame) {
            oldest_frame = cand->second.checked_frame;
            victim_slot = slot;
        }
    }
    const DAddr victim_page = native_indirect_page_ring[victim_slot];
    if (victim_page != page) {
        native_indirect_page_snapshots.erase(victim_page);
    }
    native_indirect_page_ring[victim_slot] = page;
    native_indirect_page_clock = (victim_slot + 1) & (MAX_NATIVE_BANK_SNAPSHOTS - 1);
    return {it, true};
}

template <class P>
void BufferCache<P>::CheckNativeBankPagesTracked(DAddr device_addr, u32 size, int mode) {
    // EDEN_NX2_EXP9 (experiment, 0 = off): compare only the pages whose guest memory may have
    // changed since their last compare. 1: pages guest CPU writes were collected for (applied
    // only at the next InvalidateGPUCache, while the tracker holds the page clean), pages the
    // buffer cache saw written, pages flagged at the previous bind, CPU-modified pages, and each
    // page once per frame for writers none of these see. 3: the same without the frame check.
    // 2: every page as with 0, counting the changed pages the rules of 1 and 3 would have skipped.
    static_assert(Core::DevicePageFlags::AddressBits >=
                  Tegra::MaxwellDeviceTraits::device_virtual_bits);
    const DAddr first_page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    const DAddr end = device_addr + size;
    const bool verify = mode == 2;
    const bool frame_check = mode != 3;
    auto& stats = native_bank_tracked_stats;
    ++stats.binds;
    auto& memo = native_bank_memo[((device_addr >> 12) ^ (device_addr >> 18)) &
                                  (native_bank_memo.size() - 1)];
    if (!verify && memo.device_addr == device_addr && memo.size == size &&
        memo.checked_frame == frame_tick && !memo.needs_per_bind &&
        (!memo.has_hot || memo.write_serial == cb_write_serial) &&
        !Core::g_collected_write_pages.Any(first_page, end - first_page) &&
        !Core::g_buffer_write_pages.Any(first_page, end - first_page)) [[likely]] {
        stats.pages_skipped += (end - first_page) >> Core::DEVICE_PAGEBITS;
        return;
    }
    // Most banks have no CPU-modified page: ask per page only when one has.
    const bool any_cpu_dirty = memory_tracker.IsRegionCpuModified(first_page, end - first_page);
    bool needs_per_bind = false;
    bool has_hot = false;
    for (DAddr page = first_page; page < end; page += Core::DEVICE_PAGESIZE) {
        const u8* const guest = device_memory.template GetPointer<u8>(page);
        if (guest == nullptr) {
            break;
        }
        // Take the flags before reading the page: a write flagged later is compared next bind.
        const bool collected = Core::g_collected_write_pages.Take(page);
        const bool written = Core::g_buffer_write_pages.Take(page);
        auto [it, inserted] = InsertNativeBankSnapshot(page);
        NativeBankPage& snapshot = it->second;
        if (inserted) {
            ++stats.pages_new;
            std::memcpy(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE);
            snapshot.checked_frame = frame_tick;
            snapshot.checked_write_serial = cb_write_serial;
            snapshot.recheck = collected || written;
            needs_per_bind |= snapshot.recheck;
            memory_tracker.MarkRegionAsCpuModified(page, Core::DEVICE_PAGESIZE);
            NoteCpuDirty(page, Core::DEVICE_PAGESIZE);
            continue;
        }
        // A flagged write's bytes may land after this compare: compare again at the next bind.
        const bool recheck = std::exchange(snapshot.recheck, collected || written);
        needs_per_bind |= snapshot.recheck;
        const bool is_hot = snapshot.changed_frame != ~u64{0} &&
                            frame_tick - snapshot.changed_frame < HotFrames;
        has_hot |= is_hot;
        NativeBankReason reason = BankNone;
        if (any_cpu_dirty &&
            memory_tracker.IsRegionCpuModified(page, Core::DEVICE_PAGESIZE)) {
            if (!verify) {
                ++stats.pages_skipped;
                continue;
            }
            reason = BankCpuDirty;
        } else if (collected) {
            reason = BankCollected;
        } else if (written) {
            reason = BankWritten;
        } else if (recheck) {
            reason = BankRecheck;
        } else if (is_hot && (snapshot.checked_frame != frame_tick ||
                              snapshot.checked_write_serial != cb_write_serial)) {
            // Changed lately: likely per-draw data, rewritten by paths none of the flags see
            // (EXP9=2 found ~1 such change in 15 s without this rule).
            reason = BankHot;
        } else if (frame_check && snapshot.checked_frame != frame_tick &&
                   ((frame_tick ^ (page >> Core::DEVICE_PAGEBITS)) & 15ULL) == 0) {
            reason = BankFrame;
        }
        snapshot.checked_frame = frame_tick;
        snapshot.checked_write_serial = cb_write_serial;
        if (reason == BankNone && !verify) {
            ++stats.pages_skipped;
            continue;
        }
        ++stats.compared[reason];
        const u8* const snap = snapshot.words.data();
        bool same = true;
#ifdef __AVX2__
        for (size_t off = 0; off < Core::DEVICE_PAGESIZE; off += 128) {
            const __m256i d0 = _mm256_xor_si256(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(snap + off)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off)));
            const __m256i d1 = _mm256_xor_si256(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(snap + off + 32)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off + 32)));
            const __m256i d2 = _mm256_xor_si256(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(snap + off + 64)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off + 64)));
            const __m256i d3 = _mm256_xor_si256(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(snap + off + 96)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off + 96)));
            const __m256i acc = _mm256_or_si256(_mm256_or_si256(d0, d1), _mm256_or_si256(d2, d3));
            if (!_mm256_testz_si256(acc, acc)) {
                same = false;
                break;
            }
        }
#else
        same = std::memcmp(snap, guest, Core::DEVICE_PAGESIZE) == 0;
#endif
        if (same) {
            continue;
        }
        ++stats.changed[reason];
        if (reason == BankHot || reason == BankFrame || reason == BankNone) {
            snapshot.changed_frame = frame_tick;
            has_hot = true;
        }
        std::memcpy(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE);
        memory_tracker.MarkRegionAsCpuModified(page, Core::DEVICE_PAGESIZE);
        NoteCpuDirty(page, Core::DEVICE_PAGESIZE);
    }
    memo.device_addr = device_addr;
    memo.size = size;
    memo.checked_frame = frame_tick;
    memo.write_serial = cb_write_serial;
    memo.needs_per_bind = needs_per_bind;
    memo.has_hot = has_hot;
    if (!Nx2Profile::Enabled() || (stats.binds & 0xfff) != 0) [[likely]] {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.last_log < std::chrono::seconds{5}) {
        return;
    }
    // With EXP9=2 the frame check runs too: "missed" counts what 1 would have skipped, and
    // "missed without the frame check" adds what only the frame check found (3's misses).
    const auto& compared = stats.compared;
    const auto& changed = stats.changed;
    LOG_INFO(HW_GPU,
             "Native bank check (EXP9={}, 5 s): {} binds, {} new pages, {} skipped; compared / "
             "changed: collected CPU writes {} / {}, buffer cache writes {} / {}, recheck {} / {}, "
             "CPU-modified {} / {}, hot {} / {}, frame check {} / {}, verify only {} / {}; missed "
             "{}, missed without the frame check {}",
             mode, stats.binds, stats.pages_new, stats.pages_skipped, compared[BankCollected],
             changed[BankCollected], compared[BankWritten], changed[BankWritten],
             compared[BankRecheck], changed[BankRecheck], compared[BankCpuDirty],
             changed[BankCpuDirty], compared[BankHot], changed[BankHot], compared[BankFrame],
             changed[BankFrame], compared[BankNone],
             changed[BankNone], verify ? changed[BankNone] : 0,
             verify ? changed[BankNone] + changed[BankFrame] : 0);
    stats = {};
    stats.last_log = now;
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            const u32 mark = WrittenStorageMarkSize(size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, false, mark < size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !memory_tracker.IsRegionGpuModified(binding.device_addr, size));
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                static const bool compute_bindcache = [] {
                    const char* value = std::getenv("EDEN_NX2_COMPUTE_BINDCACHE");
                    return value == nullptr || value[0] != '0';
                }();
                if constexpr (VideoCommon::StreamUniformReusable<Runtime>) {
                    if (compute_bindcache && VideoCommon::UniformStreamReuseEnabled() &&
                        VideoCommon::BindSmallUniformReusing(
                            runtime, device_memory, 5, index, binding_index, binding.device_addr,
                            size, VideoCore::Nx2Debug::Experiment(17) != 0, cb_write_serial,
                            UboPageGen(binding.device_addr, size),
                            [this, addr = binding.device_addr, size](u64& gen) {
                                return UboRangeClean(addr, size, gen);
                            })) {
                        return;
                    }
                }
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        // A native program's binding is min(layout, 8 MiB) because its pointer has no size, and
        // marking all of it GPU-written defers the CPU's later writes to those pages. Mark only
        // what BindComputeStorageBuffer worked out the dispatch can write; the binding keeps its
        // size so the shader can still read all of it.
        const bool native =
            ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
        const u64 write_mark_limit =
            native ? u64{channel_state->compute_storage_write_marks[index]} : ~u64{0};
        if (BindMultiRangeStorage(binding, is_written, compute_segments, write_mark_limit)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        // The lower of EDEN_NX2_WRITTEN_SSBO_MARK_KIB's cap and the native write mark.
        const u32 capped_size = WrittenStorageMarkSize(size);
        const u32 mark =
            is_written ? static_cast<u32>(std::min<u64>(capped_size, write_mark_limit)) : 0;
        if (mark != 0) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, true,
                              capped_size < size && capped_size <= write_mark_limit);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
        RefreshMultiRangeSegments(graphics_segments);
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
        RefreshMultiRangeSegments(compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            ++binding_epoch;
            // Draws recorded before still read it (MoltenVK encodes them at submit): destroyed
            // with the deleted buffers, not now.
            static u32 logged{};
            if (logged++ < 8) {
                LOG_INFO(HW_GPU, "Inline index buffer grows from {} to {} bytes",
                         slot_buffers[inline_buffer_id].SizeBytes(), buffer_size);
            }
            delayed_destruction_ring.Push(std::move(slot_buffers[inline_buffer_id]));
            slot_buffers.erase(inline_buffer_id);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) * u32(index_buffer_ref.FormatSizeInBytes());
    IndexBufferMemo& memo = index_buffer_memo;
    const u64 map_serial = gpu_memory->MapSerial();
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 size = (std::min)(address_size, draw_size);
    const bool memo_hit = memo.channel == channel_state && memo.memory == gpu_memory &&
                          memo.begin == gpu_addr_begin && memo.end == gpu_addr_end &&
                          memo.map_serial == map_serial && memo.deletions == binding_epoch &&
                          size != 0 && size <= memo.found_size &&
                          !Settings::IsGPUFenceBehaviorAccurate();
    if (memo_hit && !Nx2Perf::Verify()) {
        channel_state->index_buffer = Binding{
            .device_addr = memo.device_addr,
            .size = size,
            .buffer_id = memo.buffer_id,
        };
        return;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    if (size == 0 || !device_addr) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->index_buffer = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
    if (memo_hit) {
        if (memo.device_addr != *device_addr || memo.buffer_id != buffer_id) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf index buffer memo mismatch: {:#x}+{:#x} kept {:#x} buffer {}, "
                          "found {:#x} buffer {}",
                          gpu_addr_begin, size, memo.device_addr, memo.buffer_id.index,
                          *device_addr, buffer_id.index);
            }
        }
        return;
    }
    memo = IndexBufferMemo{
        .channel = channel_state,
        .memory = gpu_memory,
        .begin = gpu_addr_begin,
        .end = gpu_addr_end,
        .map_serial = map_serial,
        .deletions = binding_epoch,
        .device_addr = *device_addr,
        .found_size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
bool BufferCache<P>::ReuseGeometry(const GeometryMemo& memo, GPUVAddr begin, GPUVAddr end,
                                   u64 extra) const {
    static const bool geometry_memo = [] {
        const char* value = std::getenv("EDEN_NX2_BUFCACHE_GEOMEMO");
        return value == nullptr || value[0] != '0';
    }();
    return geometry_memo && memo.epoch == binding_epoch && memo.begin == begin &&
           memo.end == end && memo.extra == extra &&
           memo.channel == static_cast<const void*>(channel_state) &&
           memo.as_id == gpu_memory->GetID() && memo.map_serial == gpu_memory->MapSerial() &&
           !fence_behavior_accurate;
}

template <class P>
void BufferCache<P>::StoreGeometry(GeometryMemo& memo, GPUVAddr begin, GPUVAddr end, u64 extra,
                                   const Binding& binding) {
    memo = GeometryMemo{
        .epoch = binding_epoch,
        .map_serial = gpu_memory->MapSerial(),
        .as_id = gpu_memory->GetID(),
        .channel = channel_state,
        .begin = begin,
        .end = end,
        .extra = extra,
        .binding = binding,
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    GeometryMemo& memo = vertex_memo[index];
    if (array.enable != 0 && ReuseGeometry(memo, gpu_addr_begin, gpu_addr_end, 1)) {
        channel_state->vertex_buffers[index] = memo.binding;
        UpdateVertexBufferSlot(index, memo.binding);
        return;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    u32 size = address_size; // TODO: Analyze stride and number of vertices
    if (array.enable == 0 || size == 0 || !device_addr) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
    StoreGeometry(memo, gpu_addr_begin, gpu_addr_end, 1, binding);
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    if (channel_state->enabled_uniform_buffer_masks[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    const u64 enabled = channel_state->enabled_storage_buffers[stage];
    const u64 written = channel_state->written_storage_buffers[stage];
    StorageResolveCache& cache = graphics_storage_resolve[stage];
    if (ReuseStorageResolve(cache, channel_state->storage_buffers[stage], enabled, written,
                            graphics_segments)) {
        return;
    }
    const u64 start_epoch = binding_epoch;
    ForEachEnabledBit(enabled, [&](u32 index) {
        Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((written >> index) & 1) != 0;
        ResolveStorageBinding(stage, index, binding, is_written, graphics_segments);
    });
    StoreStorageResolve(cache, channel_state->storage_buffers[stage], enabled, written,
                        graphics_segments, start_epoch);
}

template <class P>
void BufferCache<P>::ResolveStorageBinding(size_t memo_stage, u32 index, Binding& binding,
                                           bool is_written, std::vector<MultiRangeSegment>& pool) {
    StorageBindingMemo& memo = storage_binding_memo[memo_stage][index];
    // EDEN_NX2_EXP18 & 2: what FindBuffer and a ResolveMultiRangeStorage finding one range gave
    // last time, while the binding, the mappings and the set of buffers are unchanged.
    const bool use_memo = (VideoCore::Nx2Debug::Experiment(18) & 2) != 0 &&
                          !fence_behavior_accurate;
    // Before the lookup: a mapping change during it leaves a memo that never matches.
    const u64 map_serial = gpu_memory->MapSerial();
    const size_t as_id = gpu_memory->GetID();
    if (use_memo && memo.epoch == binding_epoch && memo.device_addr == binding.device_addr &&
        memo.gpu_addr == binding.gpu_addr && memo.size == binding.size &&
        memo.is_written == is_written && memo.map_serial == map_serial && memo.as_id == as_id &&
        memo.channel == static_cast<const void*>(channel_state)) {
        binding.buffer_id = memo.buffer_id;
        binding.segment_first = 0;
        binding.segment_count = 0;
        return;
    }
    binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    ResolveMultiRangeStorage(binding, is_written, pool);
    memo.epoch = 0;
    if (use_memo && binding.segment_count == 0) {
        // binding_epoch after the lookup: a buffer it created is part of what the memo holds.
        memo = StorageBindingMemo{
            .epoch = binding_epoch,
            .map_serial = map_serial,
            .as_id = as_id,
            .channel = channel_state,
            .device_addr = binding.device_addr,
            .gpu_addr = binding.gpu_addr,
            .size = binding.size,
            .is_written = is_written,
            .buffer_id = binding.buffer_id,
        };
    }
}

template <class P>
bool BufferCache<P>::ReuseStorageResolve(StorageResolveCache& cache, std::span<Binding> bindings,
                                         u64 enabled, u64 written,
                                         std::vector<MultiRangeSegment>& pool) {
    if (cache.epoch != binding_epoch || cache.enabled != enabled || cache.written != written ||
        cache.channel != static_cast<const void*>(channel_state) ||
        cache.as_id != gpu_memory->GetID() || cache.map_serial != gpu_memory->MapSerial() ||
        fence_behavior_accurate) {
        return false;
    }
    bool same = true;
    ForEachEnabledBit(enabled, [&](u32 index) {
        const Binding& now = bindings[index];
        const Binding& was = cache.bindings[index];
        same &= now.device_addr == was.device_addr && now.gpu_addr == was.gpu_addr &&
                now.size == was.size;
    });
    if (!same) {
        return false;
    }
    const u32 base = static_cast<u32>(pool.size());
    pool.insert(pool.end(), cache.segments.begin(), cache.segments.end());
    ForEachEnabledBit(enabled, [&](u32 index) {
        Binding& binding = bindings[index];
        const Binding& was = cache.bindings[index];
        binding.buffer_id = was.buffer_id;
        binding.segment_first = was.segment_count != 0 ? base + was.segment_first : 0;
        binding.segment_count = was.segment_count;
    });
    return true;
}

template <class P>
void BufferCache<P>::StoreStorageResolve(StorageResolveCache& cache,
                                         std::span<const Binding> bindings, u64 enabled,
                                         u64 written, const std::vector<MultiRangeSegment>& pool,
                                         u64 start_epoch) {
    cache.epoch = 0;
    if (binding_epoch != start_epoch) {
        return;
    }
    cache.segments.clear();
    bool valid = true;
    ForEachEnabledBit(enabled, [&](u32 index) {
        Binding binding = bindings[index];
        if (binding.segment_count != 0) {
            if (binding.segment_first > pool.size() ||
                binding.segment_count > pool.size() - binding.segment_first) {
                valid = false;
                return;
            }
            const auto first = pool.begin() + binding.segment_first;
            binding.segment_first = static_cast<u32>(cache.segments.size());
            cache.segments.insert(cache.segments.end(), first, first + binding.segment_count);
        }
        cache.bindings[index] = binding;
    });
    if (!valid) {
        return;
    }
    cache.enabled = enabled;
    cache.written = written;
    cache.channel = channel_state;
    cache.as_id = gpu_memory->GetID();
    cache.map_serial = gpu_memory->MapSerial();
    cache.epoch = binding_epoch;
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    static const bool compute_bindcache = [] {
        const char* value = std::getenv("EDEN_NX2_COMPUTE_BINDCACHE");
        return value == nullptr || value[0] != '0';
    }();
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = kepler_compute->launch_description;
        const bool enabled = ((launch_desc.const_buffer_enable_mask >> index) & 1) != 0;
        const GPUVAddr cbuf_addr = enabled ? launch_desc.const_buffer_config[index].Address() : 0;
        const u32 cbuf_size = enabled ? launch_desc.const_buffer_config[index].size.Value() : 0U;
        GeometryMemo& memo = compute_uniform_memo[index];
        if (compute_bindcache && enabled &&
            ReuseGeometry(memo, cbuf_addr, cbuf_addr + cbuf_size, 2)) {
            binding = memo.binding;
            return;
        }
        if (enabled) {
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf_addr);
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.size = cbuf_size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        if (enabled) {
            StoreGeometry(memo, cbuf_addr, cbuf_addr + cbuf_size, 2, binding);
        }
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    const u64 enabled = channel_state->enabled_compute_storage_buffers;
    const u64 written = channel_state->written_compute_storage_buffers;
    if (enabled == 0) {
        return;
    }
    if (ReuseStorageResolve(compute_storage_resolve, channel_state->compute_storage_buffers,
                            enabled, written, compute_segments)) {
        return;
    }
    const u64 start_epoch = binding_epoch;
    ForEachEnabledBit(enabled, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written = ((written >> index) & 1) != 0;
        ResolveStorageBinding(NUM_STAGES, index, binding, is_written, compute_segments);
    });
    StoreStorageResolve(compute_storage_resolve, channel_state->compute_storage_buffers, enabled,
                        written, compute_segments, start_epoch);
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

namespace Detail {
/// Whether one range of `set` covers [addr, addr + size) (interval sets join touching ranges).
template <typename Set, typename Addr>
[[nodiscard]] bool RangeCovered(const Set& set, Addr addr, u64 size) {
    if (size == 0) {
        return false;
    }
    u32 pieces = 0;
    bool covered = false;
    set.ForEachInRange(addr, size, [&](Addr lower, Addr upper) {
        ++pieces;
        covered = lower == addr && upper == addr + size;
    });
    return pieces == 1 && covered;
}
} // namespace Detail

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync,
                                       bool capped) {
    if (VideoCommon::GpuSplit::RecordingDispatch()) [[unlikely]] {
        VideoCommon::GpuSplit::Get().RecordDispatchWrite(device_addr, size);
    }
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    static const bool mark_guards = [] {
        const char* const value = std::getenv("EDEN_NX2_BUFCACHE_MARK");
        return value == nullptr || value[0] != '0';
    }();
    // Dispatches and draws write the same bindings again and again: an interval set add that
    // changes nothing still erases and reinserts nodes (EDEN_NX2_BUFCACHE_MARK).
    if (!mark_guards || !Detail::RangeCovered(gpu_modified_ranges, device_addr, size)) {
        gpu_modified_ranges.Add(device_addr, size);
        gpu_modified_serial.fetch_add(1, std::memory_order_relaxed);
    }
    QueueFenceDownload(device_addr, size);
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Tegra::Nx2MarkGpuWritten(device_addr, size);
        Nx2NoteListWrite(buffer_id, device_addr, size, capped);
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size,
                                      bool to_page_end) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    constexpr DAddr page_size = DAddr{1} << page_bits;
                    const DAddr write_end = to_page_end && page == last
                                                ? page_addr + page_size
                                                : (std::min)(device_addr + size,
                                                             page_addr + page_size);
                    Nx2MarkListWords(entry, (std::max)(device_addr, page_addr) - page_addr,
                                     write_end - page_addr, true);
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU, "Command list page {:#x} is written by draws too: its "
                                         "lists wait for the work before the last host sync",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
                if (Tegra::Nx2CmdPageDiag::g_enabled) {
                    Tegra::Nx2CmdPageDiag::NoteWrite(page_addr, device_addr, size,
                                                     !nx2_collect_list_copies);
                }
            }
            ++page;
        }
    }
}

template <class P>
std::optional<u32> BufferCache<P>::Nx2MeshCountSlot(DAddr count_addr) {
    if constexpr (IS_OPENGL) {
        return std::nullopt;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        std::scoped_lock lock{nx2_list_mutex};
        const auto [it, inserted] = nx2_mesh_slots.try_emplace(count_addr);
        if (inserted) {
            if (nx2_mesh_slots.size() > Runtime::Nx2MeshMirrorSlots) {
                nx2_mesh_slots.erase(it);
            } else {
                it->second.index = static_cast<u32>(nx2_mesh_slots.size() - 1);
            }
            return std::nullopt;
        }
        const auto page = nx2_list_pages.find(count_addr & ~(page_size - 1));
        if (!it->second.initialized || page == nx2_list_pages.end() ||
            page->second.writes != it->second.mirrored_writes) {
            return std::nullopt;
        }
        return it->second.index;
    }
}

template <class P>
void BufferCache<P>::Nx2MirrorMeshCounts(Buffer& buffer, DAddr begin, DAddr end, u64 writes) {
    if constexpr (!IS_OPENGL) {
        // Called with nx2_list_mutex held, right after the page's copy (outside a render pass).
        if (nx2_mesh_slots.empty()) {
            return;
        }
        constexpr u32 slot_size = Runtime::Nx2MeshMirrorSlotSize;
        boost::container::small_vector<BufferCopy, 8> copies;
        std::optional<decltype(runtime.Nx2MeshMirror())> mirror;
        for (auto& [count_addr, slot] : nx2_mesh_slots) {
            if (count_addr < begin || count_addr + sizeof(u32) > end) {
                continue;
            }
            if (!mirror) {
                mirror = runtime.Nx2MeshMirror();
            }
            if (!slot.initialized) {
                runtime.ClearBuffer(*mirror, slot.index * slot_size + sizeof(u32),
                                    2 * sizeof(u32), 1);
                slot.initialized = true;
            }
            copies.push_back(BufferCopy{
                .src_offset = buffer.Offset(count_addr),
                .dst_offset = slot.index * slot_size,
                .size = sizeof(u32),
            });
            slot.mirrored_writes = writes;
        }
        if (!copies.empty()) {
            runtime.CopyBuffer(*mirror, buffer, copies, true);
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick, u64 writer) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            entry.copy_writer = writer;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
            Nx2MirrorMeshCounts(buffer, begin, end, entry.writes);
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync)
    -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && !entry.draw_written && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        const bool words = (VideoCore::Nx2Debug::Experiment(40) & 1) != 0;
        u64 tick{};
        u64 writes{};
        u64 writer{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            if (entry.draw_written) {
                // EDEN_NX2_CMDPAGE_DRAIN=1: drain and flush for every list on such pages.
                static const bool always_drain = std::getenv("EDEN_NX2_CMDPAGE_DRAIN") != nullptr;
                if (always_drain) {
                    return Nx2ListRead::Unknown;
                }
                if (!past_sync) {
                    return Nx2ListRead::NeedsSync;
                }
                return entry.synced_writes == entry.writes ? Nx2ListRead::Synced
                                                           : Nx2ListRead::Unknown;
            }
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced, and
            // with EXP40 & 1 a list on none of the words GPU work wrote since the page was flushed
            // is in guest memory too (zortos293/Lichii#805).
            if (entry.synced_writes == entry.writes ||
                (words && entry.flushed && !Nx2HasListWords(entry, offset, offset + out.size()))) {
                return Nx2ListRead::Synced;
            }
            if (!current(entry)) {
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                Nx2ReadListWords(entry, device_addr, offset, out);
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
            writer = entry.copy_writer;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes).
        runtime.WaitSubmittedTick(tick, writer);
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        Nx2ReadListWords(it->second, device_addr, offset, out);
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2ReadListWords(const Nx2ListPage& entry, DAddr device_addr, u64 offset,
                                      std::span<u8> out) {
    if constexpr (!IS_OPENGL) {
        const u8* const copy = entry.copy->mapped_span.data();
        const u64 end = offset + out.size();
        bool all_gpu = true;
        for (u64 word = offset / 4; word < (end + 3) / 4 && all_gpu; ++word) {
            all_gpu = entry.gpu_words.test(static_cast<size_t>(word));
        }
        if (all_gpu || !entry.flushed || (VideoCore::Nx2Debug::Experiment(40) & 1) == 0) {
            std::memcpy(out.data(), copy + offset, out.size());
            return;
        }
        device_memory.ReadBlockUnsafe(device_addr, out.data(), out.size());
        for (u64 byte = offset; byte < end;) {
            const u64 next = (std::min)((byte / 4 + 1) * 4, end);
            if (entry.gpu_words.test(static_cast<size_t>(byte / 4))) {
                std::memcpy(out.data() + (byte - offset), copy + byte, next - byte);
            }
            byte = next;
        }
    }
}

template <class P>
void BufferCache<P>::Nx2MarkListWords(Nx2ListPage& entry, u64 begin, u64 end, bool set) {
    if (begin >= end) {
        return;
    }
    const u64 first = set ? begin / 4 : (begin + 3) / 4;
    const u64 last = (std::min<u64>)(set ? (end + 3) / 4 : end / 4, Nx2ListPageWords);
    for (u64 word = first; word < last; ++word) {
        entry.gpu_words.set(static_cast<size_t>(word), set);
    }
}

template <class P>
bool BufferCache<P>::Nx2HasListWords(const Nx2ListPage& entry, u64 begin, u64 end) {
    const u64 last = (std::min<u64>)((end + 3) / 4, Nx2ListPageWords);
    for (u64 word = begin / 4; word < last; ++word) {
        if (entry.gpu_words.test(static_cast<size_t>(word))) {
            return true;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
        entry.gpu_words.reset();
        entry.flushed = true;
    }
}

template <class P>
void BufferCache<P>::Nx2FlushListPages(std::span<const DAddr> pages) {
    constexpr u64 page_size = u64{1} << Tegra::Nx2WrittenPageBits;
    std::vector<std::pair<DAddr, u64>> ranges;
    for (const DAddr page : pages) {
        ranges.emplace_back(page, page_size);
    }
    std::vector<std::pair<DAddr, u64>> drawn;
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, entry] : nx2_list_pages) {
            if (entry.draw_written && entry.synced_writes != entry.writes) {
                ranges.emplace_back(page, page_size);
                drawn.emplace_back(page, entry.writes);
            }
        }
    }
    DownloadMemoryRanges(ranges);
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, writes] : drawn) {
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.writes == writes) {
                entry.synced_writes = writes;
                entry.gpu_words.reset();
                entry.flushed = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const DAddr page_addr = page << page_bits;
            const auto it = nx2_list_pages.find(page_addr);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
            Nx2MarkListWords(entry, (std::max)(device_addr, page_addr) - page_addr,
                             (std::min)(device_addr + size, page_addr + (DAddr{1} << page_bits)) -
                                 page_addr,
                             false);
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            if (fence_behavior_accurate) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                if (sparse_compatible && !buffer.IsSparseCompatible()) {
                    usable = false;
                }
            }
            if (usable) {
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
bool BufferCache<P>::JoinCopyIsDead(DAddr device_addr, u64 size) {
    // Every page CPU-modified: SynchronizeBuffer uploads such pages whole from guest memory
    // (ForEachModifiedRange reports whole pages) before any bind, DMA copy or download reads
    // the new buffer there, so the bytes a join copy puts in would be overwritten unread. The
    // exceptions keep host data the upload leaves alone or read the host buffer later: a page
    // GPU-modified or Preflushable, GPU writes the upload skips (gpu_modified_ranges), and GPU
    // writes queued or in flight for download.
    if (!memory_tracker.IsRegionWhollyCpuModified(device_addr, size) ||
        gpu_modified_ranges.Intersects(device_addr, size) ||
        uncommitted_gpu_modified_ranges.Intersects(device_addr, size)) {
        return false;
    }
    const auto intersects = [&](const Common::RangeSet<DAddr>& ranges) {
        return ranges.Intersects(device_addr, size);
    };
    return std::ranges::none_of(committed_gpu_modified_ranges, intersects) &&
           std::ranges::none_of(pending_download_ranges, intersects);
}

template <class P>
bool BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    VideoCommon::Churn::Add(VideoCommon::Churn::BuffersJoined);
    // EDEN_NX2_JOIN_SKIP_CPU=1 (fable_answer_17 4): a drifted per-frame range the guest rewrote
    // whole since its last upload needs no copy, the first bind uploads all of it anyway
    // (JoinCopyIsDead). Read on first use: nx2_switches.txt is applied after static init.
    static const bool skip_cpu = [] {
        const char* value = std::getenv("EDEN_NX2_JOIN_SKIP_CPU");
        return value != nullptr && value[0] == '1';
    }();
    const bool copy = !skip_cpu || !JoinCopyIsDead(overlap.CpuAddr(), overlap.SizeBytes());
    if (copy) {
        VideoCommon::Churn::Add(VideoCommon::Churn::BufferJoinBytes, overlap.SizeBytes());
        runtime.CopyBuffer(new_buffer, overlap, copies, true);
    } else {
        VideoCommon::Churn::Add(VideoCommon::Churn::BufferJoinsSkipped);
        VideoCommon::Churn::Add(VideoCommon::Churn::BufferJoinSkippedBytes, overlap.SizeBytes());
    }
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
    return copy;
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible) {
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    VideoCommon::Churn::Add(VideoCommon::Churn::BuffersCreated);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    bool copied = false;
    bool skipped = false;
    for (const BufferId overlap_id : overlap.ids) {
        const bool joined_copy = JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
        copied |= joined_copy;
        skipped |= !joined_copy;
    }
    if (skipped && !copied) {
        // The barrier a join copy ends with (CopyBuffer) also ordered the clear above before a
        // reordered upload in a later submission; keep it when every copy was skipped.
        runtime.PostCopyBarrier();
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    if constexpr (!insert) {
        ++binding_epoch;
    }
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const u64 allocated = Common::AlignUp(size + runtime.GetBufferAllocationTail(), 1024);
    if (insert) {
        buffer_bytes += allocated;
        total_used_memory += allocated;
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= allocated;
        total_used_memory -= allocated;
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    if (buffer_id == NULL_BUFFER_ID) {
        return;
    }
    // EDEN_NX2_EXP18 & 4: the buffer's own field instead of its LRU node (a cache miss) for every
    // bind of the frame; a second touch in the same frame changes nothing.
    if ((VideoCore::Nx2Debug::Experiment(18) & 4) != 0) {
        if (buffer.lru_touched_tick == frame_tick) {
            return;
        }
        buffer.lru_touched_tick = frame_tick;
    }
    lru_cache.Touch(buffer.getLRUID(), frame_tick);
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
    // An upload over a clean word of the tracker (word_manager.h) finds and changes nothing:
    // walk only the dirty ones. Verify walks every word and checks the clean words.
    const bool skip_clean = Nx2Perf::On(Nx2Perf::Opt::CleanWords) && !Nx2Perf::Verify();
    if (Nx2Perf::On(Nx2Perf::Opt::CleanWords) && Nx2Perf::Verify()) {
        static std::atomic<u32> logged{};
        if (!memory_tracker.CleanWordsHold(device_addr, size) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf draw-binds: clean tracker words wrong near {:#x}+{:#x}",
                      device_addr, size);
        }
    }
    static const bool sync_skip_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SYNC_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    static const bool skip_regions_enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SKIP_REGIONS");
        return value == nullptr || value[0] != '0';
    }();
    SyncSkip* skip_entry = nullptr;
    const bool by_regions =
        skip_regions_enabled && SkipRegionCount(device_addr, size) <= MAX_SKIP_REGIONS;
    if (sync_skip_enabled && skip_clean) {
        skip_entry = &sync_skip[((device_addr ^ (u64{size} << 40)) * 0x9E3779B97F4A7C15ULL) >>
                                (64 - SYNC_SKIP_BITS)];
        if (skip_entry->buffer == &buffer && skip_entry->device_addr == device_addr &&
            skip_entry->size == size && skip_entry->epoch == cpu_dirty_epoch &&
            skip_entry->binding_epoch == binding_epoch &&
            (by_regions ? skip_entry->region_sum == SkipRegionSum(device_addr, size)
                        : skip_entry->gen == buffer.cpu_dirty_gen)) {
            return true;
        }
    }
    const auto store_skip = [&] {
        if (skip_entry == nullptr) {
            return;
        }
        *skip_entry = SyncSkip{
            .buffer = &buffer,
            .device_addr = device_addr,
            .size = size,
            .gen = buffer.cpu_dirty_gen,
            .epoch = cpu_dirty_epoch,
            .binding_epoch = binding_epoch,
            .region_sum = by_regions ? SkipRegionSum(device_addr, size) : 0,
        };
    };
    static const bool sync_precheck = [] {
        const char* value = std::getenv("EDEN_NX2_SYNC_PRECHECK");
        return value == nullptr || value[0] != '0';
    }();
    if (sync_precheck && skip_clean && !memory_tracker.IsRegionUploadDirty(device_addr, size)) {
        store_skip();
        return true;
    }
    const auto for_each_upload_range = [&](auto&& func) {
        if (skip_clean) {
            memory_tracker.ForEachDirtyUploadRange(device_addr, size, func);
        } else {
            memory_tracker.ForEachUploadRange(device_addr, size, func);
        }
    };
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        if (total_size_bytes == 0) {
            store_skip();
            return true;
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        store_skip();
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
void BufferCache<P>::NoteCpuDirty(DAddr device_addr, u64 size) {
    ForEachBufferInRange(device_addr, size, [](BufferId, Buffer& buffer) { ++buffer.cpu_dirty_gen; });
    BumpSkipRegions(device_addr, size);
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    Common::PerfReport::Add(Common::PerfReport::BufferUploadBytes, total_size_bytes);
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            if (!native_indirect_page_snapshots.empty()) {
                const DAddr first_page = Common::AlignUp(device_addr, Core::DEVICE_PAGESIZE);
                const DAddr last_page =
                    Common::AlignDown(device_addr + copy.size, Core::DEVICE_PAGESIZE);
                for (DAddr page = first_page; page < last_page; page += Core::DEVICE_PAGESIZE) {
                    if (const auto it = native_indirect_page_snapshots.find(page);
                        it != native_indirect_page_snapshots.end()) {
                        Core::g_collected_write_pages.Take(page);
                        Core::g_buffer_write_pages.Take(page);
                        if (const u8* const guest = device_memory.template GetPointer<u8>(page)) {
                            std::memcpy(it->second.words.data(), guest, Core::DEVICE_PAGESIZE);
                        }
                        it->second.checked_frame = frame_tick;
                        it->second.checked_write_serial = cb_write_serial;
                        it->second.recheck = false;
                    }
                }
            }
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer, bool* pages_clean) {
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    const bool clean = !IsRegionGpuModified(aligned_start, aligned_end - aligned_start);
    if (pages_clean != nullptr) {
        *pages_clean = clean;
    }
    if (clean) {
        return false;
    }
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    Core::g_buffer_write_pages.Mark(dest_address, copy_size);
    NoteUniformWrite(dest_address, copy_size);
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const Download& download : downloads) {
        const DAddr address = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        landing_begin = (std::min)(landing_begin, address);
        landing_end = (std::max)(landing_end, address + download.copy.size);
    }
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    // Guest memory is written once the host GPU ran the copies, inside a landing over their
    // bytes: the CPU writes collected meanwhile keep their bytes too.
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const BufferCopy& copy : copies) {
        landing_begin = (std::min)(landing_begin, buffer.CpuAddr() + copy.src_offset);
        landing_end = (std::max)(landing_end, buffer.CpuAddr() + copy.src_offset + copy.size);
    }
    // EDEN_NX2_DOWNLOAD_STATS=1: how buffer downloads (reactive flushes among them) are made and
    // what they wait for, logged every 5 s.
    static const bool download_stats = std::getenv("EDEN_NX2_DOWNLOAD_STATS") != nullptr;
    struct DownloadStats {
        u64 mapped{}, staged{}, mapped_ns{}, staged_ns{}, bytes{}, writer_done{};
        std::chrono::steady_clock::time_point last{std::chrono::steady_clock::now()};
    };
    static DownloadStats stats;
    const auto stats_begin = download_stats ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    const auto note_download = [&](bool mapped_path) {
        if (!download_stats) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const u64 ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - stats_begin).count());
        (mapped_path ? stats.mapped : stats.staged) += 1;
        (mapped_path ? stats.mapped_ns : stats.staged_ns) += ns;
        stats.bytes += total_size_bytes;
        if (now - stats.last >= std::chrono::seconds{5}) {
            const double window = std::chrono::duration<double>(now - stats.last).count();
            LOG_INFO(HW_GPU,
                     "Download stats: {:.0f} mapped/s ({:.1f} ms/s), {:.0f} staged/s ({:.1f} ms/s), "
                     "{:.0f} KiB/s, writer already done {:.0f}/s",
                     stats.mapped / window, stats.mapped_ns / 1e6 / window, stats.staged / window,
                     stats.staged_ns / 1e6 / window, stats.bytes / 1024.0 / window,
                     stats.writer_done / window);
            stats = DownloadStats{};
        }
    };
    if constexpr (!IS_OPENGL) {
        if (download_stats && runtime.IsFree(buffer.getWriteTick())) {
            ++stats.writer_done;
        }
    }

    if constexpr (requires { buffer.Mapped(); }) {
        if (const auto mapped = buffer.Mapped(); !mapped.empty()) {
            runtime.Wait(buffer.LastUsageTick());
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            for (const BufferCopy& copy : copies) {
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + copy.src_offset,
                                      mapped.data() + copy.src_offset, copy.size);
            }
            note_download(true);
            return;
        }
    }
    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        const VideoCore::DownloadLanding landing{landing_begin, landing_end};
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, copy_mapped_memory, copy.size);
        }
        note_download(false);
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            const VideoCore::DownloadLanding landing{copy_device_addr,
                                                     copy_device_addr + copy.size};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            WriteDownloadedMemory(cpu_written, copy_device_addr, immediate_buffer.data(), copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    ++binding_epoch;
    VideoCommon::Churn::Add(VideoCommon::Churn::BuffersDeleted);
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    cpu_read_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
    bool dirty_index{false};
    boost::container::small_vector<u64, NUM_VERTEX_BUFFERS> dirty_vertex_buffers;
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](std::span<Binding> bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };

    if (channel_state->index_buffer.buffer_id == buffer_id) {
        channel_state->index_buffer.buffer_id = BufferId{};
        dirty_index = true;
    }

    for (u32 index = 0; index < channel_state->vertex_buffers.size(); index++) {
        auto& binding = channel_state->vertex_buffers[index];
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
            UpdateVertexBufferSlot(index, binding);
            dirty_vertex_buffers.push_back(index);
        }
    }
    std::ranges::for_each(channel_state->uniform_buffers, replace);
    std::ranges::for_each(channel_state->storage_buffers, replace);
    replace(channel_state->transform_feedback_buffers);
    replace(channel_state->compute_uniform_buffers);
    replace(channel_state->compute_storage_buffers);

    // Ocarina uses several GPU channels. A deleted slot must not survive in another channel's
    // bindings and later resolve to an unrelated buffer that reuses the same id.
    for (const size_t other_id : this->active_channel_ids) {
        auto& other = this->channel_storage[other_id];
        if (&other == channel_state) {
            continue;
        }
        auto& other_flags = other.maxwell3d.dirty.flags;
        if (other.index_buffer.buffer_id == buffer_id) {
            other.index_buffer.buffer_id = BufferId{};
            other_flags[Dirty::IndexBuffer] = true;
        }
        for (u32 index = 0; index < other.vertex_buffers.size(); ++index) {
            if (other.vertex_buffers[index].buffer_id == buffer_id) {
                other.vertex_buffers[index].buffer_id = BufferId{};
                other_flags[Dirty::VertexBuffers] = true;
                other_flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
        std::ranges::for_each(other.uniform_buffers, replace);
        std::ranges::for_each(other.storage_buffers, replace);
        replace(other.transform_feedback_buffers);
        replace(other.compute_uniform_buffers);
        replace(other.compute_storage_buffers);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            other.dirty_uniform_buffers.fill(~u32{0});
            other.uniform_buffer_binding_sizes.fill({});
        }
        other.has_deleted_buffers = true;
    }

    // Adapted from the pre-refresh branch: merged buffers invalidate segment references too.
    InvalidateMultiRangeSegments(std::span{graphics_segments}, buffer_id);
    InvalidateMultiRangeSegments(std::span{compute_segments}, buffer_id);

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        ++cpu_dirty_epoch;
    }

    Unregister(buffer_id);

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);

    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->uniform_buffer_binding_sizes.fill({});
    }

    auto& flags = maxwell3d->dirty.flags;
    if (dirty_index) {
        flags[Dirty::IndexBuffer] = true;
    }

    if (dirty_vertex_buffers.size() > 0) {
        flags[Dirty::VertexBuffers] = true;
        for (auto index : dirty_vertex_buffers) {
            flags[Dirty::VertexBuffer0 + index] = true;
        }
    }
    channel_state->has_deleted_buffers = true;
}

template <class P>
Binding BufferCache<P>::BindCachedStorageBuffer(size_t memo_stage, size_t ssbo_index,
                                                GPUVAddr cbuf_slot_addr, u32 cbuf_index,
                                                u32 cbuf_offset, bool is_written,
                                                bool native_pointer, u32 indirect_offset,
                                                GPUVAddr* out_pointer) {
    static const bool use_memo = [] {
        const char* const value = std::getenv("EDEN_NX2_SSBO_MEMO");
        return value == nullptr || value[0] != '0';
    }();
    const bool table_pointer = indirect_offset != ~0U;
    const bool reads_size_word = !(table_pointer && !is_written) && !native_pointer;
    const u8 flags = static_cast<u8>(1U | (is_written ? 2U : 0U) | (native_pointer ? 4U : 0U) |
                                     (table_pointer ? 8U : 0U));
    const size_t as_id = gpu_memory->GetID();
    const u64 map_serial = gpu_memory->MapSerial();
    SsboBindMemo& memo = ssbo_bind_memo[memo_stage][ssbo_index];

    if (use_memo && memo.flags == flags &&
        memo.channel == static_cast<const void*>(channel_state) && memo.as_id == as_id &&
        memo.map_serial == map_serial && memo.cbuf_slot_addr == cbuf_slot_addr &&
        memo.indirect_offset == indirect_offset && memo.cbuf_host != nullptr) [[likely]] {
        u64 q0 = 0;
        std::memcpy(&q0, memo.cbuf_host, sizeof(u64));
        if (q0 == memo.cbuf_qword0) [[likely]] {
            if (!table_pointer) {
                u64 q1 = 0;
                if (reads_size_word) {
                    std::memcpy(&q1, memo.cbuf_host + sizeof(u64), sizeof(u64));
                }
                if (q1 == memo.cbuf_qword1) [[likely]] {
                    if (out_pointer != nullptr) {
                        *out_pointer = memo.resolved_pointer;
                    }
                    return memo.binding;
                }
            } else if (q0 == 0) {
                if (out_pointer != nullptr) {
                    *out_pointer = 0;
                }
                return NULL_BINDING;
            } else if (memo.entry_host != nullptr) [[likely]] {
                u64 eq0 = 0;
                std::memcpy(&eq0, memo.entry_host, sizeof(u64));
                u64 eq1 = 0;
                if (reads_size_word) {
                    std::memcpy(&eq1, memo.entry_host + sizeof(u64), sizeof(u64));
                }
                if (eq0 == memo.entry_qword0 && eq1 == memo.entry_qword1) [[likely]] {
                    if (out_pointer != nullptr) {
                        *out_pointer = memo.resolved_pointer;
                    }
                    return memo.binding;
                }
            }
        }
    }

    const auto host_for_gpu = [&](GPUVAddr addr, size_t bytes) -> const u8* {
        if ((addr & Core::Memory::YUZU_PAGEMASK) > Core::Memory::YUZU_PAGESIZE - bytes) {
            return nullptr;
        }
        const std::optional<DAddr> daddr = gpu_memory->GpuToCpuAddress(addr);
        return daddr ? device_memory.template GetPointer<u8>(*daddr) : nullptr;
    };

    const size_t cbuf_bytes = (!table_pointer && reads_size_word) ? 16 : 8;
    const u8* const cbuf_host = use_memo ? host_for_gpu(cbuf_slot_addr, cbuf_bytes) : nullptr;
    u64 cbuf_qword0 = 0;
    u64 cbuf_qword1 = 0;
    if (cbuf_host != nullptr) {
        std::memcpy(&cbuf_qword0, cbuf_host, sizeof(u64));
        if (!table_pointer && reads_size_word) {
            std::memcpy(&cbuf_qword1, cbuf_host + sizeof(u64), sizeof(u64));
        }
    } else if (table_pointer || out_pointer != nullptr) {
        cbuf_qword0 = gpu_memory->Read<u64>(cbuf_slot_addr);
    }

    if (table_pointer && cbuf_qword0 == 0) {
        if (cbuf_host != nullptr) {
            memo = SsboBindMemo{
                .channel = channel_state,
                .as_id = as_id,
                .map_serial = map_serial,
                .cbuf_slot_addr = cbuf_slot_addr,
                .indirect_offset = indirect_offset,
                .flags = flags,
                .cbuf_host = cbuf_host,
                .entry_host = nullptr,
                .cbuf_qword0 = 0,
                .resolved_pointer = 0,
                .binding = NULL_BINDING,
            };
        } else {
            memo.flags = 0;
        }
        if (out_pointer != nullptr) {
            *out_pointer = 0;
        }
        return NULL_BINDING;
    }

    const GPUVAddr ssbo_addr = table_pointer ? (cbuf_qword0 + indirect_offset) : cbuf_slot_addr;
    const size_t entry_bytes = reads_size_word ? 16 : 8;
    const u8* const entry_host =
        (use_memo && table_pointer) ? host_for_gpu(ssbo_addr, entry_bytes) : nullptr;
    u64 entry_qword0 = 0;
    u64 entry_qword1 = 0;
    if (entry_host != nullptr) {
        std::memcpy(&entry_qword0, entry_host, sizeof(u64));
        if (reads_size_word) {
            std::memcpy(&entry_qword1, entry_host + sizeof(u64), sizeof(u64));
        }
    } else if (table_pointer && out_pointer != nullptr) {
        entry_qword0 = gpu_memory->Read<u64>(ssbo_addr);
    }

    const Binding binding =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer, table_pointer);
    if (table_pointer) {
        TraceTablePointer(memo_stage == NUM_STAGES, memo_stage == NUM_STAGES ? 0 : memo_stage,
                          ssbo_index, cbuf_index, cbuf_offset, indirect_offset, cbuf_qword0,
                          binding);
    }

    const GPUVAddr resolved = table_pointer ? entry_qword0 : cbuf_qword0;
    if (out_pointer != nullptr) {
        *out_pointer = resolved;
    }
    if (cbuf_host != nullptr && (!table_pointer || entry_host != nullptr)) {
        memo = SsboBindMemo{
            .channel = channel_state,
            .as_id = as_id,
            .map_serial = map_serial,
            .cbuf_slot_addr = cbuf_slot_addr,
            .indirect_offset = indirect_offset,
            .flags = flags,
            .cbuf_host = cbuf_host,
            .entry_host = entry_host,
            .cbuf_qword0 = cbuf_qword0,
            .cbuf_qword1 = cbuf_qword1,
            .entry_qword0 = entry_qword0,
            .entry_qword1 = entry_qword1,
            .resolved_pointer = resolved,
            .binding = binding,
        };
    } else {
        memo.flags = 0;
    }
    return binding;
}

template <class P>
void BufferCache<P>::TraceTablePointer(bool compute, size_t stage, size_t ssbo_index,
                                       u32 cbuf_index, u32 cbuf_offset, u32 indirect_offset,
                                       GPUVAddr holder, const Binding& binding) {
    // A table entry is read here from guest memory, and again by the shader from the holder's
    // host buffer. Logs each change of a slot's entry or size, and whether the two copies of the
    // entry can differ: GPU-written (host newer) or CPU-written and not yet uploaded (guest newer).
    static const bool enabled = std::getenv("EDEN_NX2_TABLE_POINTER_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    const GPUVAddr entry_addr = holder + indirect_offset;
    const GPUVAddr entry = gpu_memory->Read<u64>(entry_addr);
    const u64 key = (u64{compute} << 63) | (u64{stage & 7} << 60) | (u64{ssbo_index & 63} << 54) |
                    (u64{cbuf_index & 31} << 49) | (u64{cbuf_offset & 0x7ffff} << 30) |
                    (indirect_offset & 0x3fffffff);
    const std::pair<GPUVAddr, u32> state{entry, binding.size};
    const auto [it, inserted] = traced_table_pointers.try_emplace(key, state);
    if (!inserted && it->second == state) {
        return;
    }
    it->second = state;
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    if (++lines > max_lines) {
        if (lines == max_lines + 1) {
            LOG_INFO(HW_GPU, "Table pointer trace stopped after {} lines", max_lines);
        }
        return;
    }
    const std::optional<DAddr> entry_device = gpu_memory->GpuToCpuAddress(entry_addr);
    const bool gpu_modified =
        entry_device && memory_tracker.IsRegionGpuModified(*entry_device, sizeof(u64));
    const bool cpu_modified =
        entry_device && memory_tracker.IsRegionCpuModified(*entry_device, sizeof(u64));
    LOG_INFO(HW_GPU,
             "Table pointer frame={} {} stage={} ssbo={} c{}[{:#x}]+{:#x}: holder={:#x} "
             "entry={:#x} bound={:#x} size={:#x} entry_gpu_modified={} entry_cpu_modified={}",
             frame_tick, compute ? "compute" : "graphics", stage, ssbo_index, cbuf_index,
             cbuf_offset, indirect_offset, holder, entry, binding.device_addr, binding.size,
             gpu_modified, cpu_modified);
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer, bool table_pointer) const {
    const GPUVAddr gpu_addr = gpu_memory->Read<u64>(ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    // The size word after a plain pointer is part of the answer; pool and native pointers ignore
    // it.
    const bool reads_size_word = !(table_pointer && !is_written) && !native_pointer;
    const u64 raw_next_qword = reads_size_word ? gpu_memory->Read<u64>(ssbo_addr + 8) : 0;
    const u64 next_qword =
        (raw_next_qword >> 32) == 0 ? static_cast<u64>(static_cast<u32>(raw_next_qword)) : 0ULL;
    // Every draw rebinds each storage buffer: the layout lookup (a mutex), the range walk and
    // two page walks cost Ocarina's GPU thread ~6%. Answers hold until the page table changes.
    const u32 alignment = runtime.GetStorageBufferAlignment();
    struct CachedBinding {
        size_t owner; ///< MemoryManager id + 1, 0 when empty
        GPUVAddr gpu_addr;
        u64 next_qword;
        u64 serial;
        u32 alignment;
        u32 flags;
        Binding binding;
    };
    constexpr size_t Ways = 4;
    thread_local std::array<std::array<CachedBinding, Ways>, 512> binding_cache{};
    const u32 flags =
        (is_written ? 1U : 0U) | (native_pointer ? 2U : 0U) | (table_pointer ? 4U : 0U);
    const size_t owner = gpu_memory->GetID() + 1;
    const u64 serial = gpu_memory->MapSerial();
    auto& set = binding_cache[((gpu_addr ^ next_qword ^ flags) * 0x9E3779B97F4A7C15ULL) >> 55];
    for (size_t way = 0; way < Ways; ++way) {
        const CachedBinding& cached = set[way];
        if (cached.owner == owner && cached.gpu_addr == gpu_addr &&
            cached.next_qword == next_qword && cached.serial == serial &&
            cached.alignment == alignment && cached.flags == flags) {
            const Binding hit = cached.binding;
            std::rotate(set.begin(), set.begin() + way, set.begin() + way + 1);
            return hit;
        }
    }
    // Per-object pointers into one pool miss the cache above on every new object. A pool is
    // a pointer's kind range and device run: later pointers inside it read their layout size,
    // contiguous size and device address from it while the mappings stay (EDEN_NX2_BUFCACHE_POOLS).
    struct Pool {
        size_t owner{};
        u64 serial{};
        GPUVAddr begin{};
        GPUVAddr kind_end{};  ///< [begin, kind_end) lies in one kind map entry, not INVALID
        GPUVAddr run_end{};   ///< [begin, run_end) is one device run
        bool run_end_exact{}; ///< false: the run goes on past run_end (only 64 MiB were asked)
        DAddr device_begin{};
    };
    static const bool pool_ranges = [] {
        const char* const value = std::getenv("EDEN_NX2_BUFCACHE_POOLS");
        return value == nullptr || value[0] != '0';
    }();
    thread_local std::array<Pool, 8> pools{};
    thread_local u32 pool_cursor{};
    const Pool* pool = nullptr;
    if (pool_ranges) {
        for (const Pool& entry : pools) {
            if (entry.owner == owner && entry.serial == serial && entry.begin <= gpu_addr &&
                gpu_addr < entry.kind_end && gpu_addr < entry.run_end) {
                pool = &entry;
                break;
            }
        }
        if (pool == nullptr) {
            constexpr u64 run_query = 64_MiB;
            const size_t layout = gpu_memory->GetMemoryLayoutSize(gpu_addr);
            const size_t run = layout != 0 ? gpu_memory->MaxContinuousRange(gpu_addr, run_query) : 0;
            const std::optional<DAddr> device = run != 0 ? gpu_memory->GpuToCpuAddress(gpu_addr)
                                                         : std::nullopt;
            if (device) {
                Pool& entry = pools[pool_cursor++ % pools.size()];
                entry = Pool{
                    .owner = owner,
                    .serial = serial,
                    .begin = gpu_addr,
                    .kind_end = gpu_addr + layout,
                    .run_end = gpu_addr + run,
                    .run_end_exact = run < run_query,
                    .device_begin = *device,
                };
                pool = &entry;
            }
        }
    }
    const Binding binding = [&]() -> Binding {
        u32 size = [&]() {
            const u32 memory_layout_size = static_cast<u32>((std::min)(
                pool != nullptr ? u64{pool->kind_end - gpu_addr}
                                : u64{gpu_memory->GetMemoryLayoutSize(gpu_addr)},
                u64{(std::numeric_limits<u32>::max)()}));
            // A pointer read from a table in memory is the base of an object pool, read anywhere in
            // it: bound to the end of its mapping, as far as a storage buffer may reach (at most
            // 1 GiB). Ocarina of Time's grass instances lie 12-18 MiB past theirs (an 8 MiB cap
            // culled every blade), and its light probe records reach about 130 MiB past theirs: a
            // 64 MiB cap left the far probes outside the binding, and the fog took one overbright
            // constant from them. Written pools keep a cap: a write marks the whole binding
            // GPU-modified.
            if (table_pointer && !is_written) {
                const u64 reach = (std::min)(runtime.GetMaxStorageBufferRange(), u64{1_GiB});
                const u32 limit = static_cast<u32>(reach - (std::min)(reach, u64{64_KiB}));
                return (std::min)(memory_layout_size, limit);
            }
            // A native (Switch 2) compute program's pointer is bare: the word after it is another
            // parameter (a dimension, stride or texture handle), not an NVN buffer size. Taking it
            // as one bound Fortune's Weave's DLSS input features to 266 bytes and dropped the rest.
            if (native_pointer) {
                return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
            }
            const u32 packed_size = static_cast<u32>(next_qword);
            const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                            packed_size != 0 && packed_size <= memory_layout_size;
            if (next_qword_is_size) {
                return packed_size;
            }
            return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
        }();
        if ((native_pointer || table_pointer) && size != 0) {
            // Stop where the mapping or its device range does: the binding is one device range, and
            // a written one isn't split into separate ranges without sparse sources.
            if (pool != nullptr && (pool->run_end_exact || gpu_addr + size <= pool->run_end)) {
                size = static_cast<u32>((std::min<u64>)(size, pool->run_end - gpu_addr));
            } else {
                size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, size));
            }
        }

        // Alignment only applies to the offset of the buffer
        const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
        const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

        // Inside a run, device addresses follow GPU addresses (above the device space, the
        // translation reports nothing: leave those to it).
        const auto translate = [&](GPUVAddr address) -> std::optional<DAddr> {
            if (pool != nullptr && address >= pool->begin) {
                const DAddr device = pool->device_begin + (address - pool->begin);
                if ((device >> Tegra::MaxwellDeviceMemoryManager::AS_BITS) == 0) {
                    return device;
                }
            }
            return gpu_memory->GpuToCpuAddress(address);
        };
        const std::optional<DAddr> aligned_device_addr = translate(aligned_gpu_addr);
        if (!aligned_device_addr || size == 0) {
            LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
            return NULL_BINDING;
        }
        const std::optional<DAddr> device_addr = translate(gpu_addr);
        ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
                   cbuf_index);
        // The end address used for size calculation does not need to be aligned
        const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

        u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
        if (is_written) {
            binding_size = aligned_size;
        }
        return Binding{
            .device_addr = *aligned_device_addr,
            .gpu_addr = aligned_gpu_addr,
            .size = binding_size,
            .buffer_id = BufferId{},
        };
    }();
    std::rotate(set.begin(), set.end() - 1, set.end());
    set[0] = {owner, gpu_addr, next_qword, serial, alignment, flags, binding};
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
