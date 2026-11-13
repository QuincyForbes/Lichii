// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <vector>
#include <string_view>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"

#include "common/logging.h"
#include "common/settings.h"
#include "common/dkb_cpu_trace.h"
#include "common/perf_report.h"
#include "common/thread.h"
#include <boost/container/small_vector.hpp>

#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/upload_batch.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#ifdef LICHII_METALFX
#include "video_core/renderer_vulkan/metal_gpu_time.h"
#endif

namespace Vulkan {

namespace {
std::atomic<u64> g_profile_submits{};
std::atomic<u64> g_profile_renderpasses{};
std::atomic<u64> g_profile_finishes{};
std::atomic<u64> g_profile_finish_ns{};
std::atomic<u64> g_profile_window_start{};
std::mutex g_profile_end_mutex;
std::map<std::string, u64> g_profile_ends;
std::map<std::string, u64> g_profile_flushes;
std::map<std::string, u64> g_profile_reopens;

// "Perf sites:" (EDEN_NX2_PERF_REPORT): submissions, render pass ends and reopens by caller.
enum PerfSiteKind : size_t { SiteSubmit, SitePassEnd, SiteReopen, SiteKinds };
struct PerfSite {
    const char* file;
    u32 line;
    u64 count;
};
std::mutex g_perf_site_mutex;
std::array<std::vector<PerfSite>, SiteKinds> g_perf_sites;

void CountPerfSite(PerfSiteKind kind, const char* file, u32 line) {
    if (!Common::PerfReport::Enabled()) {
        return;
    }
    std::scoped_lock lock{g_perf_site_mutex};
    auto& sites = g_perf_sites[kind];
    const auto it = std::ranges::find_if(
        sites, [&](const PerfSite& site) { return site.file == file && site.line == line; });
    if (it != sites.end()) {
        ++it->count;
    } else {
        sites.push_back({file, line, 1});
    }
}

void CountSite(std::map<std::string, u64>& sites, const char* file, u32 line) {
    if (&sites == &g_profile_flushes) {
        CountPerfSite(SiteSubmit, file, line);
    } else if (&sites == &g_profile_ends) {
        CountPerfSite(SitePassEnd, file, line);
    }
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return;
    }
    std::string_view name{file};
    if (const auto slash = name.find_last_of("/\\"); slash != std::string_view::npos) {
        name.remove_prefix(slash + 1);
    }
    std::scoped_lock lock{g_profile_end_mutex};
    ++sites[fmt::format("{}:{}", name, line)];
}

void CountRenderPassEnd(const char* file, u32 line) {
    CountSite(g_profile_ends, file, line);
}

// Passes that began on the framebuffer the last one ended on, by the caller that ended that one.
void CountReopen(const char* file, u32 line, VkExtent2D area) {
    CountPerfSite(SiteReopen, file, line);
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return;
    }
    std::string_view name{file != nullptr ? file : "elsewhere"};
    if (const auto slash = name.find_last_of("/\\"); slash != std::string_view::npos) {
        name.remove_prefix(slash + 1);
    }
    std::scoped_lock lock{g_profile_end_mutex};
    ++g_profile_reopens[fmt::format("{}:{} ({}x{})", name, line, area.width, area.height)];
}

// The callers of Flush and Finish: who asks for the queue submissions.
void CountFlush(const char* file, u32 line) {
    CountSite(g_profile_flushes, file, line);
}

void ReportSubmitCounts() {
    const u64 now = VideoCommon::Nx2Profile::Now();
    u64 start = g_profile_window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_profile_window_start.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 10'000'000'000ULL ||
        !g_profile_window_start.compare_exchange_strong(start, now)) {
        return;
    }
    const double seconds = static_cast<double>(now - start) / 1e9;
    LOG_INFO(Render_Vulkan,
             "Scheduler per second: {:.0f} submits, {:.0f} render passes, {:.0f} finishes "
             "waiting {:.1f} ms",
             static_cast<double>(g_profile_submits.exchange(0)) / seconds,
             static_cast<double>(g_profile_renderpasses.exchange(0)) / seconds,
             static_cast<double>(g_profile_finishes.exchange(0)) / seconds,
             static_cast<double>(g_profile_finish_ns.exchange(0)) / 1e6 / seconds);
    const auto report = [seconds](std::map<std::string, u64>& sites, const char* what) {
        std::vector<std::pair<u64, std::string>> counts;
        {
            std::scoped_lock lock{g_profile_end_mutex};
            for (const auto& [site, count] : sites) {
                counts.emplace_back(count, site);
            }
            sites.clear();
        }
        std::sort(counts.rbegin(), counts.rend());
        for (size_t i = 0; i < counts.size() && i < 16; ++i) {
            LOG_INFO(Render_Vulkan, "{} {:.0f}/s at {}", what,
                     static_cast<double>(counts[i].first) / seconds, counts[i].second);
        }
    };
    report(g_profile_ends, "Render pass ends");
    report(g_profile_reopens, "Render passes reopened on the target they left");
    report(g_profile_flushes, "Submits asked");
}
}

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)},
      upload_batch{std::make_unique<UploadBatch>()} {

    if (Common::DkbFrameTrace::Enabled()) {
        const auto physical = device.GetPhysical();
        trace_timestamp_bits =
            physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
        trace_timestamp_period = physical.GetProperties().limits.timestampPeriod;
        if (trace_timestamp_bits != 0 && trace_timestamp_bits <= 64 && trace_timestamp_period > 0) {
            try {
                trace_query_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxSubmissions * 2,
                    .pipelineStatistics = 0,
                });
                trace_submissions.resize(TraceMaxSubmissions);
                trace_work_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxWork * 2,
                    .pipelineStatistics = 0,
                });
                trace_work_items.resize(TraceMaxWork);
            } catch (const vk::Exception& error) {
                LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp unavailable: {}", error.what());
            }
        } else {
            LOG_INFO(Render_Vulkan, "DKB gpu_timestamp unsupported: valid_bits={} period_ns={}",
                     trace_timestamp_bits, trace_timestamp_period);
        }
    }

    if (Common::PerfReport::Enabled() && !device.IsMoltenVK()) {
        const auto physical = device.GetPhysical();
        gpu_time_bits =
            physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
        gpu_time_period = physical.GetProperties().limits.timestampPeriod;
        if (gpu_time_bits >= 36 && gpu_time_period > 0.0) {
            try {
                gpu_time_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = GpuTimeSlots * 2,
                    .pipelineStatistics = 0,
                });
            } catch (const vk::Exception& error) {
                LOG_WARNING(Render_Vulkan, "Perf: no GPU timestamps: {}", error.what());
            }
        }
    }

    if (const char* const value = std::getenv("EDEN_NX2_PASS_TIMES");
        value != nullptr && std::atof(value) > 0.0) {
        const auto physical = device.GetPhysical();
        const u32 bits =
            physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
        pass_time_period = physical.GetProperties().limits.timestampPeriod;
        pass_time_mask = bits >= 64 ? ~u64{0} : (u64{1} << bits) - 1;
        pass_time_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(std::atof(value)));
        if (bits >= 32 && pass_time_period > 0.0) {
            try {
                pass_time_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = PassTimeQueries,
                    .pipelineStatistics = 0,
                });
                pass_time_records.reserve(PassTimeQueries / 2);
                LOG_INFO(Common_Memory, "Perf passes: one frame timed every {} s", value);
#ifdef LICHII_METALFX
                MetalGpuTime::TimestampNsPerTick();
#endif
            } catch (const vk::Exception& error) {
                LOG_WARNING(Render_Vulkan, "Perf passes: no timestamp queries: {}", error.what());
            }
        } else {
            LOG_WARNING(Render_Vulkan, "Perf passes: timestamps not supported ({} bits)", bits);
        }
    }

    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() {
    if (trace_query_pool || gpu_time_pool || pass_time_pool) {
        WaitWorker();
        worker_thread.request_stop();
        worker_thread.join();
        if (trace_gpu_tick != 0) {
            master_semaphore->Wait(trace_gpu_tick);
        }
        if (pass_time_pending) {
            master_semaphore->Wait(pass_time_wait_tick);
        }
        if (gpu_time_count != 0) {
            master_semaphore->Wait(
                gpu_time_slots[(gpu_time_head + gpu_time_count - 1) % GpuTimeSlots].tick);
        }
    }
}

void Scheduler::BeginGpuTime() {
    if (!gpu_time_pool) {
        return;
    }
    CollectGpuTime();
    if (gpu_time_count == GpuTimeSlots) {
        gpu_time_open = ~0U;
        return;
    }
    const u32 index = (gpu_time_head + gpu_time_count) % GpuTimeSlots;
    current_upload_cmdbuf.ResetQueryPool(*gpu_time_pool, index * 2, 2);
    current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *gpu_time_pool,
                                         index * 2);
    gpu_time_open = index;
}

void Scheduler::EndGpuTime(vk::CommandBuffer cmdbuf, u64 tick) {
    if (gpu_time_open == ~0U) {
        return;
    }
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *gpu_time_pool,
                          gpu_time_open * 2 + 1);
    gpu_time_slots[gpu_time_open] = {.tick = tick, .query = gpu_time_open * 2};
    ++gpu_time_count;
    gpu_time_open = ~0U;
}

void Scheduler::CollectGpuTime() {
    const u64 mask = gpu_time_bits >= 64 ? ~u64{0} : (u64{1} << gpu_time_bits) - 1;
    const u64 second = static_cast<u64>(1e9 / gpu_time_period);
    bool refreshed = false;
    while (gpu_time_count != 0) {
        const GpuTimeSlot& slot = gpu_time_slots[gpu_time_head];
        if (!master_semaphore->IsFree(slot.tick)) {
            if (refreshed) {
                break;
            }
            master_semaphore->Refresh();
            refreshed = true;
            continue;
        }
        struct Result {
            u64 timestamp;
            u64 available;
        };
        std::array<Result, 2> results{};
        const VkResult result = device.GetLogical().GetQueryResults(
            *gpu_time_pool, slot.query, 2, sizeof(results), results.data(), sizeof(Result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY ||
            (result == VK_SUCCESS && (results[0].available == 0 || results[1].available == 0))) {
            break;
        }
        if (result == VK_SUCCESS) {
            const u64 begin = results[0].timestamp & mask;
            const u64 end = results[1].timestamp & mask;
            if (end > begin && end - begin < second * 10) {
                // Submissions overlap on the GPU: count each moment once. A counter that
                // wrapped leaves the last end far ahead; start over from there.
                if (gpu_time_last_end > end + second) {
                    gpu_time_last_end = 0;
                }
                const u64 from = gpu_time_last_end > begin ? (std::min)(gpu_time_last_end, end)
                                                           : begin;
                gpu_time_last_end = (std::max)(gpu_time_last_end, end);
                Common::PerfReport::Add(
                    Common::PerfReport::GpuBusyNs,
                    static_cast<u64>(static_cast<double>(end - from) * gpu_time_period));
                Common::PerfReport::Add(Common::PerfReport::GpuTimedBuffers);
            }
        }
        gpu_time_head = (gpu_time_head + 1) % GpuTimeSlots;
        --gpu_time_count;
    }
}

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                     std::source_location location) {
    CountFlush(location.file_name(), location.line());
    if (state.renderpass) {
        NotePassEnd(location);
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    return signal_value;
}

bool Scheduler::DeferFlush() {
    // Fences come one per compute release: ~11,500 a second in Hyrule Warriors: Age of
    // Imprisonment fights, each a queue submission (~22,000 a second with the others), which
    // kept the Vulkan worker busy submitting. Batched within 500 us: ~10-15% more fps there.
    // Command list copies after dispatches were ~9,000 more a second, of which the front waited
    // for about 5%.
    static const s64 interval_us = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_FLUSH_US");
        if (value != nullptr) {
            return static_cast<s64>(std::atoll(value));
        }
        // 0 by default (EvilOOT's flash fix): Ocarina of Time flashed white with the 500 us batch
        // (sword boulder with preflush: 31 whole-screen flashes in 30 s, 0 without), and ran
        // faster without it (Lake Hylia 52.3 -> 55.4 fps). EDEN_NX2_FENCE_FLUSH_US=500 restores it.
        return VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush) ? s64{5000} : s64{0};
    }();
    if (interval_us <= 0 || !VideoCommon::GpuSplit::OnBackThread() ||
        std::chrono::steady_clock::now() - last_flush >= std::chrono::microseconds{interval_us}) {
        return false;
    }
    flush_pending = true;
    return true;
}

void Scheduler::FlushDeferred() {
    if (flush_pending) {
        Flush();
    }
}

bool Scheduler::WaitSubmitted(u64 tick) {
    if (IsSubmitted(tick)) {
        return true;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    split.RequestIdleHook();
    // No time limit: the back can spend seconds on one entry (a large shader's translation),
    // and a fence reported as submitted before it is would release its downloads too early.
    const auto start = std::chrono::steady_clock::now();
    auto next_log = start + std::chrono::seconds{2};
    for (u32 spins = 0; !IsSubmitted(tick); ++spins) {
        if (!split.BackRunning()) {
            return IsSubmitted(tick);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_log) {
            next_log = now + std::chrono::seconds{2};
            // The back serves the request between entries: one that never does is blocked in
            // an entry. Name the wait when it is a tracked one (BackWaitScope).
            const char* const back_site = VideoCommon::back_wait_site.load(std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan,
                        "Waited {} s for the GPU thread to submit tick {} (submitted up to {}); "
                        "GPU back thread blocked in: {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick, master_semaphore->CurrentTick() - 1,
                        back_site != nullptr ? back_site
                                             : "no tracked wait (an untracked lock or wait, or "
                                               "a long entry)");
            split.RequestIdleHook();
        }
        if (spins < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    return true;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                       std::source_location location) {
    CountFlush(location.file_name(), location.line());
    if (state.renderpass) {
        NotePassEnd(location);
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    const bool profile = VideoCommon::Nx2Profile::Enabled();
    const u64 start = profile ? VideoCommon::Nx2Profile::Now() : 0;
    SubmitExecution(signal_semaphore, wait_semaphore);
    Wait(presubmit_tick);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    if (profile) {
        g_profile_finishes.fetch_add(1, std::memory_order_relaxed);
        g_profile_finish_ns.fetch_add(VideoCommon::Nx2Profile::Now() - start,
                                      std::memory_order_relaxed);
    }
}

void Scheduler::WaitWorker() {
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        if (!work_queue.empty()) {
            waiter_waiting = true;
            wait_cv.wait(ql, [this] { return work_queue.empty(); });
            waiter_waiting = false;
        }
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::MarkFrameEnd(u64 frame) {
    if (Common::DkbFrameTrace::Enabled()) {
        Record([this, frame](vk::CommandBuffer) { trace_frame_end = frame; });
    }
}

void Scheduler::ReportPerfSites(u64 frames) {
    std::array<std::vector<PerfSite>, SiteKinds> sites;
    {
        std::scoped_lock lock{g_perf_site_mutex};
        for (size_t kind = 0; kind < SiteKinds; ++kind) {
            sites[kind] = std::move(g_perf_sites[kind]);
            g_perf_sites[kind].clear();
        }
    }
    if (frames == 0) {
        return;
    }
    const auto top = [frames](std::vector<PerfSite>& list) {
        std::ranges::sort(list, std::greater{}, &PerfSite::count);
        std::string text;
        u64 rest{};
        for (size_t i = 0; i < list.size(); ++i) {
            if (i >= 8) {
                rest += list[i].count;
                continue;
            }
            std::string_view file{list[i].file != nullptr ? list[i].file : "elsewhere"};
            if (const size_t slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
                file.remove_prefix(slash + 1);
            }
            text += fmt::format("{}{}:{} {:.1f}", text.empty() ? "" : ", ", file, list[i].line,
                                static_cast<double>(list[i].count) / static_cast<double>(frames));
        }
        if (rest != 0) {
            text += fmt::format(", others {:.1f}",
                                static_cast<double>(rest) / static_cast<double>(frames));
        }
        return text.empty() ? std::string{"none"} : text;
    };
    LOG_INFO(Common_Memory,
             "Perf sites per frame: submits by caller: {} | render pass ends by caller: {} | "
             "passes reopened on the target the last one left, after: {}",
             top(sites[SiteSubmit]), top(sites[SitePassEnd]), top(sites[SiteReopen]));
}

void Scheduler::NotePassEnd(const std::source_location& location) {
    CountRenderPassEnd(location.file_name(), location.line());
    pass_end_file = location.file_name();
    pass_end_line = location.line();
}

void Scheduler::MarkPassTimesFrame() {
    if (!pass_time_pool) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const bool start = now - pass_time_last >= pass_time_interval;
    if (start) {
        pass_time_last = now;
    }
    std::vector<PassReopen> reopens;
    if (pass_time_host_sampling) {
        reopens = std::move(pass_time_reopens);
    }
    pass_time_reopens.clear();
    pass_time_host_sampling = start;
    Record([this, start, reopens = std::move(reopens)](vk::CommandBuffer cmdbuf) {
        if (pass_time_sampling) {
            pass_time_reopen_report = reopens;
            // The frame's last commands go with the next submission; its results come after it.
            PassTimeCloseLabel(cmdbuf);
            pass_time_sampling = false;
            pass_time_closing = true;
            return;
        }
        if (!start || pass_time_closing || pass_time_pending) {
            return;
        }
        pass_time_sampling = true;
        pass_time_reset = true;
        pass_time_next_query = 0;
        pass_time_submissions = 0;
        pass_time_records.clear();
        pass_time_spans.clear();
    });
}

void Scheduler::PassTimestamp(vk::CommandBuffer cmdbuf, bool begin, const PassShape* shape,
                              u32 draws) {
    pass_time_in_pass = begin;
    if (begin) {
        PassTimeCloseLabel(cmdbuf);
        if (!pass_time_sampling || pass_time_next_query + 4 > PassTimeQueries) {
            return;
        }
        if (std::exchange(pass_time_reset, false)) {
            // The upload command buffer runs before this one in the same submission.
            current_upload_cmdbuf.ResetQueryPool(*pass_time_pool, 0, PassTimeQueries);
        }
        const u32 query = pass_time_next_query;
        pass_time_next_query += 2;
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool, query);
        pass_time_records.push_back({
            .shape = *shape,
            .begin_query = query,
            .submission = pass_time_submissions,
        });
        return;
    }
    if (pass_time_records.empty() || pass_time_records.back().end_query != ~0U) {
        return;
    }
    PassTimeRecord& record = pass_time_records.back();
    record.end_query = record.begin_query + 1;
    record.draws = draws;
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool,
                          record.end_query);
}

void Scheduler::PassTimeLabel(vk::CommandBuffer cmdbuf, const char* file, u32 line) {
    PassTimeCloseLabel(cmdbuf);
    if (!pass_time_sampling || pass_time_in_pass || pass_time_next_query + 4 > PassTimeQueries) {
        return;
    }
    if (std::exchange(pass_time_reset, false)) {
        current_upload_cmdbuf.ResetQueryPool(*pass_time_pool, 0, PassTimeQueries);
    }
    const u32 query = pass_time_next_query;
    pass_time_next_query += 2;
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool, query);
    pass_time_records.push_back({
        .begin_query = query,
        .submission = pass_time_submissions,
        .site_file = file,
        .site_line = line,
    });
    pass_time_open_label = static_cast<u32>(pass_time_records.size() - 1);
}

void Scheduler::PassTimeCloseLabel(vk::CommandBuffer cmdbuf) {
    if (pass_time_open_label == ~0U) {
        return;
    }
    PassTimeRecord& label = pass_time_records[pass_time_open_label];
    pass_time_open_label = ~0U;
    label.end_query = label.begin_query + 1;
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool, label.end_query);
}

u32 Scheduler::PassTimeWorkBegin(vk::CommandBuffer cmdbuf, u64 hash, std::array<u32, 3> size) {
    PassTimeCloseLabel(cmdbuf);
    if (pass_time_next_query + 4 > PassTimeQueries) {
        return ~0U;
    }
    if (std::exchange(pass_time_reset, false)) {
        current_upload_cmdbuf.ResetQueryPool(*pass_time_pool, 0, PassTimeQueries);
    }
    const u32 query = pass_time_next_query;
    pass_time_next_query += 2;
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool, query);
    pass_time_records.push_back({
        .begin_query = query,
        .submission = pass_time_submissions,
        .dispatch = true,
        .hash = hash,
        .groups = size,
    });
    return static_cast<u32>(pass_time_records.size() - 1);
}

void Scheduler::PassTimeWorkEnd(vk::CommandBuffer cmdbuf, u32 record) {
    if (record == ~0U) {
        return;
    }
    PassTimeRecord& work = pass_time_records[record];
    work.end_query = work.begin_query + 1;
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool, work.end_query);
}

void Scheduler::PassTimesSubmitted(u64 tick) {
    PassTimeCloseLabel(current_cmdbuf);
    if (pass_time_open_span != ~0U || ((pass_time_sampling || pass_time_closing) &&
                                       pass_time_next_query + 2 <= PassTimeQueries)) {
        // The whole submission, from the start of its upload command buffer (the first one of
        // the frame: from the end of its uploads) to the end of this one.
        if (pass_time_open_span == ~0U) {
            if (std::exchange(pass_time_reset, false)) {
                current_upload_cmdbuf.ResetQueryPool(*pass_time_pool, 0, PassTimeQueries);
            }
            pass_time_open_span = pass_time_next_query;
            pass_time_next_query += 2;
            current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                                 *pass_time_pool, pass_time_open_span);
        }
        current_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pass_time_pool,
                                      pass_time_open_span + 1);
        pass_time_spans.push_back(pass_time_open_span);
    }
    pass_time_open_span = ~0U;
    if (pass_time_sampling) {
        ++pass_time_submissions;
    } else if (pass_time_closing) {
        pass_time_closing = false;
        pass_time_pending = !pass_time_records.empty() || !pass_time_spans.empty();
        pass_time_wait_tick = tick;
    } else if (pass_time_pending && master_semaphore->IsFree(pass_time_wait_tick)) {
        pass_time_pending = false;
        ReportPassTimes();
    }
}

void Scheduler::ReportPassTimes() {
    std::vector<u64> stamps(pass_time_next_query);
    if (stamps.empty() ||
        device.GetLogical().GetQueryResults(*pass_time_pool, 0, pass_time_next_query,
                                            stamps.size() * sizeof(u64), stamps.data(),
                                            sizeof(u64), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
        return;
    }
    // MoltenVK on an Apple GPU: each timestamp is the start of a blit pass it adds once the Metal
    // encoder the query was recorded in has ended, and it reports a period of 1 ns.
    double period = pass_time_period;
    std::string method;
    if (device.IsMoltenVK()) {
#ifdef LICHII_METALFX
        if (const double measured = MetalGpuTime::TimestampNsPerTick(); measured > 0.0) {
            period = measured;
        }
#endif
        method = fmt::format(
            " | MoltenVK: each time is taken after the Metal encoder its query fell in ends, so a "
            "pass counts from the end of the encoder before it to the end of its own, dispatches "
            "or copies sharing one encoder put its whole time on the first, and this frame ran "
            "{} extra Metal passes (slower than the others); {:.2f} ns per tick (driver says {:.2f})",
            pass_time_next_query, period, pass_time_period);
    }
    const auto ms_between = [&](u32 first, u32 second) {
        const u64 a = stamps[first] & pass_time_mask;
        const u64 b = stamps[second] & pass_time_mask;
        return b > a ? static_cast<double>(b - a) * period / 1e6 : 0.0;
    };
    const size_t timed = static_cast<size_t>(std::ranges::count_if(
        pass_time_records, [](const PassTimeRecord& record) { return record.end_query != ~0U; }));
    const size_t nonzero = static_cast<size_t>(
        std::ranges::count_if(pass_time_records, [&](const PassTimeRecord& record) {
            return record.end_query != ~0U && ms_between(record.begin_query, record.end_query) > 0.0;
        }));
    if (timed >= 8 && nonzero * 10 < timed) {
        method += fmt::format(" | only {} of {} timed blocks took any time: the driver gave one "
                              "time per command buffer, per-pass times are not real",
                              nonzero, timed);
    }
    using VideoCore::Surface::PixelFormat;
    const auto bytes_of = [](u8 format) -> u64 {
        const auto pixel_format = static_cast<PixelFormat>(format);
        return pixel_format == PixelFormat::Invalid ? 0
                                                    : VideoCore::Surface::BytesPerBlock(pixel_format);
    };
    struct Group {
        std::string name;
        u32 passes{};
        u32 draws{};
        double ms{};
        u64 load_bytes{};
        u64 store_bytes{};
    };
    std::vector<Group> groups;
    struct DispatchGroup {
        u64 hash{};
        u32 count{};
        double ms{};
        u64 groups{}; ///< Workgroups; 0 for indirect dispatches
    };
    std::vector<DispatchGroup> dispatches;
    std::vector<Group> labels; ///< Work after leaving a render pass, by the caller that left it
    double in_dispatches{};
    double in_labels{};
    double in_passes{};
    u64 load_bytes{};
    u64 store_bytes{};
    for (const PassTimeRecord& record : pass_time_records) {
        if (record.end_query == ~0U) {
            continue;
        }
        const double ms = ms_between(record.begin_query, record.end_query);
        if (record.site_file != nullptr) {
            in_labels += ms;
            std::string_view file{record.site_file};
            if (const size_t slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
                file.remove_prefix(slash + 1);
            }
            std::string site = fmt::format("{}:{}", file, record.site_line);
            auto it = std::ranges::find(labels, site, &Group::name);
            if (it == labels.end()) {
                it = labels.insert(labels.end(), Group{.name = std::move(site)});
            }
            ++it->passes;
            it->ms += ms;
            continue;
        }
        if (record.dispatch) {
            in_dispatches += ms;
            auto it = std::ranges::find(dispatches, record.hash, &DispatchGroup::hash);
            if (it == dispatches.end()) {
                it = dispatches.insert(dispatches.end(), DispatchGroup{.hash = record.hash});
            }
            ++it->count;
            it->ms += ms;
            it->groups += u64{record.groups[0]} * record.groups[1] * record.groups[2];
            continue;
        }
        in_passes += ms;
        const PassShape& s = record.shape;
        const u64 pixels = u64{s.width} * s.height * (std::max)(u16{1}, s.layers) *
                           (std::max)(u8{1}, s.samples);
        u64 loads{};
        u64 stores{};
        std::string name = fmt::format("{}x{}", s.width, s.height);
        if (s.layers > 1) {
            name += fmt::format("x{}", s.layers);
        }
        if (s.samples > 1) {
            name += fmt::format(" {}x MSAA", s.samples);
        }
        for (u32 slot = 0; slot < s.num_colors; ++slot) {
            const u64 bytes = pixels * bytes_of(s.color_formats[slot]);
            loads += (s.color_clear_mask & (1U << slot)) != 0 ? 0 : bytes;
            stores += (s.color_discard_mask & (1U << slot)) != 0 ? 0 : bytes;
            name += fmt::format(" {}", static_cast<PixelFormat>(s.color_formats[slot]));
        }
        if (bytes_of(s.depth_format) != 0) {
            const u64 bytes = pixels * bytes_of(s.depth_format);
            loads += s.depth_clear ? 0 : bytes;
            stores += s.depth_discard ? 0 : bytes;
            name += fmt::format(" + {}", static_cast<PixelFormat>(s.depth_format));
        }
        load_bytes += loads;
        store_bytes += stores;
        auto it = std::ranges::find(groups, name, &Group::name);
        if (it == groups.end()) {
            it = groups.insert(groups.end(), Group{.name = std::move(name)});
        }
        ++it->passes;
        it->draws += record.draws;
        it->ms += ms;
        it->load_bytes += loads;
        it->store_bytes += stores;
    }
    std::ranges::sort(groups, std::greater{}, &Group::ms);
    std::string top;
    for (size_t i = 0; i < groups.size() && i < 10; ++i) {
        const Group& g = groups[i];
        top += fmt::format("{}{}: {} passes, {:.2f} ms, {} draws, loads {} MiB, stores {} MiB",
                           i == 0 ? "" : " | ", g.name, g.passes, g.ms, g.draws,
                           g.load_bytes >> 20, g.store_bytes >> 20);
    }
    std::ranges::sort(dispatches, std::greater{}, &DispatchGroup::ms);
    std::string top_dispatches;
    for (size_t i = 0; i < dispatches.size() && i < 10; ++i) {
        const DispatchGroup& d = dispatches[i];
        top_dispatches += fmt::format("{}{:016x}: {}x, {:.2f} ms, {} groups", i == 0 ? "" : " | ",
                                      d.hash, d.count, d.ms, d.groups);
    }
    std::ranges::sort(labels, std::greater{}, &Group::ms);
    std::string top_labels;
    for (size_t i = 0; i < labels.size() && i < 10; ++i) {
        top_labels += fmt::format("{}{}: {}x, {:.2f} ms", i == 0 ? "" : " | ", labels[i].name,
                                  labels[i].passes, labels[i].ms);
    }
    std::ranges::sort(pass_time_reopen_report, std::greater{}, &PassReopen::count);
    std::string top_reopens;
    u32 num_reopens{};
    for (size_t i = 0; i < pass_time_reopen_report.size(); ++i) {
        const PassReopen& reopen = pass_time_reopen_report[i];
        num_reopens += reopen.count;
        if (i >= 10) {
            continue;
        }
        std::string_view file{reopen.file != nullptr ? reopen.file : "elsewhere"};
        if (const size_t slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
            file.remove_prefix(slash + 1);
        }
        top_reopens += fmt::format("{}{}:{} on {}x{}: {}x", i == 0 ? "" : " | ", file, reopen.line,
                                   reopen.width, reopen.height, reopen.count);
    }
    static const bool list_passes = [] {
        const char* const value = std::getenv("EDEN_NX2_PASS_LIST");
        return value != nullptr && value[0] != '0';
    }();
    if (list_passes) {
        // The frame in order: P<target> passes, D dispatches, O other work, / a new submission.
        std::vector<u64> targets;
        std::string line;
        u32 part{};
        u32 submission{};
        const auto flush = [&] {
            if (!line.empty()) {
                LOG_INFO(Common_Memory, "Perf pass list {}: {}", part++, line);
                line.clear();
            }
        };
        for (const PassTimeRecord& record : pass_time_records) {
            if (record.end_query == ~0U) {
                continue;
            }
            if (record.submission != submission) {
                submission = record.submission;
                line += "/ ";
            }
            const double ms = ms_between(record.begin_query, record.end_query);
            if (record.site_file != nullptr) {
                std::string_view file{record.site_file};
                if (const size_t slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
                    file.remove_prefix(slash + 1);
                }
                line += fmt::format("O {}:{} {:.3f}, ", file, record.site_line, ms);
            } else if (record.dispatch) {
                line += fmt::format("D {:016x} {:.3f}, ", record.hash, ms);
            } else {
                const PassShape& s = record.shape;
                auto it = std::ranges::find(targets, s.framebuffer);
                if (it == targets.end()) {
                    it = targets.insert(targets.end(), s.framebuffer);
                }
                line += fmt::format("P{} {}x{}", it - targets.begin(), s.width, s.height);
                for (u32 slot = 0; slot < s.num_colors; ++slot) {
                    line += fmt::format(" {}{}", static_cast<PixelFormat>(s.color_formats[slot]),
                                        (s.color_clear_mask & (1U << slot)) != 0 ? " (clear)" : "");
                }
                if (bytes_of(s.depth_format) != 0) {
                    line += fmt::format(" + {}{}", static_cast<PixelFormat>(s.depth_format),
                                        s.depth_clear ? " (clear)" : "");
                }
                line += fmt::format(" {} draws {:.3f}, ", record.draws, ms);
            }
            if (line.size() > 1500) {
                flush();
            }
        }
        flush();
    }
    double submitted{};
    for (const u32 span : pass_time_spans) {
        submitted += ms_between(span, span + 1);
    }
    const auto count = [&](auto&& pred) {
        return static_cast<size_t>(std::ranges::count_if(pass_time_records, pred));
    };
    const size_t num_passes = count([](const PassTimeRecord& record) {
        return !record.dispatch && record.site_file == nullptr;
    });
    const size_t num_dispatches = count([](const PassTimeRecord& record) { return record.dispatch; });
    LOG_INFO(Common_Memory,
             "Perf passes: {} submissions take {:.2f} ms of GPU time: {} render passes {:.2f} ms, "
             "{} guest dispatches {:.2f} ms, other work after leaving a pass {:.2f} ms, the rest "
             "{:.2f} ms; attachments load {} MiB, store {} MiB | passes: {} | dispatches: {} | "
             "other work by caller: {} | {} passes reopened the target the last one left, after: {}{}",
             pass_time_spans.size(), submitted, num_passes, in_passes, num_dispatches,
             in_dispatches, in_labels,
             (std::max)(submitted - in_passes - in_dispatches - in_labels, 0.0), load_bytes >> 20,
             store_bytes >> 20, top, top_dispatches, top_labels, num_reopens, top_reopens, method);
}

void Scheduler::DispatchWork(bool lazy) {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (uploads_pending) {
        FlushUploads();
    }
    static const bool lazy_wake = [] {
        const char* value = std::getenv("EDEN_NX2_LAZY_WAKE");
        return value == nullptr || value[0] != '0';
    }();
    const bool has_work = chunk && !chunk->Empty();
    if (!has_work && !(lazy_queued && !lazy)) {
        return;
    }
    lazy = lazy && lazy_wake;
    bool wake;
    {
        std::scoped_lock ql{queue_mutex};
        if (has_work) {
            work_queue.push(std::move(chunk));
        }
        wake = worker_waiting && !work_queue.empty() && (!lazy || work_queue.size() >= 2);
        lazy_queued = worker_waiting && !wake && !work_queue.empty();
    }
    if (wake) {
        work_cv.notify_one();
    }
    if (has_work) {
        AcquireNewChunk();
    }
}

void Scheduler::RecordUpload(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies) {
    if (VideoCommon::GpuSplit::check_owner) {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::RecordUpload");
    }
    for (const VkBufferCopy& copy : copies) {
        const std::span<const VkBufferCopy> region{&copy, 1};
        if (uploads_pending && !upload_batch->CanAdd(src, dst, region)) {
            FlushUploads();
        }
        upload_batch->Add(src, dst, region);
        uploads_pending = true;
        has_pending_commands = true;
    }
}

void Scheduler::FlushUploads() {
    boost::container::small_vector<VkBufferCopy, 16> copies(upload_batch->copies.begin(),
                                                            upload_batch->copies.end());
    boost::container::small_vector<UploadBatch::Group, 8> groups(upload_batch->groups.begin(),
                                                                 upload_batch->groups.end());
    upload_batch->copies.clear();
    upload_batch->groups.clear();
    uploads_pending = false;
    Record([copies = std::move(copies), groups = std::move(groups)](vk::CommandBuffer cmdbuf) {
        UploadBatch::Record(cmdbuf, std::span<const VkBufferCopy>{copies.data(), copies.size()},
                            std::span<const UploadBatch::Group>{groups.data(), groups.size()});
    });
}

bool Scheduler::RenderPassMatches(const Framebuffer* framebuffer) const {
    const VkExtent2D render_area = framebuffer->RenderArea();
    return state.renderpass != VK_NULL_HANDLE &&
           (framebuffer->RenderPass() == state.renderpass || state.clear_variant) &&
           framebuffer->Handle() == state.framebuffer &&
           render_area.width == state.render_area.width &&
           render_area.height == state.render_area.height;
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count,
                                    u32 color_clear_mask, bool depth_clear,
                                    u32 color_discard_mask, bool depth_discard) {
    if (VideoCommon::Nx2Profile::Enabled()) {
        g_profile_renderpasses.fetch_add(1, std::memory_order_relaxed);
    }
    Common::PerfReport::Add(Common::PerfReport::RenderPasses);
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    pass_draws = 0;
    if (framebuffer_handle == last_pass_framebuffer && clear_value_count == 0) {
        // This pass loads back what the last one stored: on a tiler that is the whole target, twice.
        Common::PerfReport::Add(Common::PerfReport::RenderPassReopens);
        CountReopen(last_pass_end_file, last_pass_end_line, render_area);
        if (pass_time_host_sampling) [[unlikely]] {
            auto it = std::ranges::find_if(pass_time_reopens, [&](const PassReopen& reopen) {
                return reopen.file == last_pass_end_file && reopen.line == last_pass_end_line &&
                       reopen.width == render_area.width && reopen.height == render_area.height;
            });
            if (it == pass_time_reopens.end()) {
                it = pass_time_reopens.insert(
                    pass_time_reopens.end(),
                    PassReopen{.file = last_pass_end_file,
                               .line = last_pass_end_line,
                               .width = static_cast<u16>((std::min)(render_area.width, 0xffffU)),
                               .height = static_cast<u16>((std::min)(render_area.height, 0xffffU))});
            }
            ++it->count;
        }
    }
    const bool time_pass = static_cast<bool>(pass_time_pool);
    PassShape shape{};
    if (time_pass || Common::PerfReport::Enabled()) {
        using VideoCore::Surface::PixelFormat;
        const RenderPassKey& key = framebuffer->RenderPassKeyBase();
        shape = {
            .width = static_cast<u16>((std::min)(render_area.width, 0xffffU)),
            .height = static_cast<u16>((std::min)(render_area.height, 0xffffU)),
            .layers = static_cast<u16>((std::min)(framebuffer->Layers(), 0xffffU)),
            .samples = static_cast<u8>(key.samples),
            .depth_format = static_cast<u8>(key.depth_format),
            .color_clear_mask = static_cast<u8>(color_clear_mask | key.color_clear_mask),
            .color_discard_mask = static_cast<u8>(color_discard_mask | key.color_discard_mask),
            .depth_clear = depth_clear || key.depth_stencil_clear,
            .depth_discard = depth_discard || key.depth_stencil_discard,
            .framebuffer = (u64)framebuffer_handle,
        };
        // The render pass's attachments are the valid formats, in slot order.
        u32 clear_mask{};
        u32 discard_mask{};
        for (u32 slot = 0; slot < key.color_formats.size(); ++slot) {
            if (key.color_formats[slot] == PixelFormat::Invalid) {
                continue;
            }
            clear_mask |= ((shape.color_clear_mask >> slot) & 1U) << shape.num_colors;
            discard_mask |= ((shape.color_discard_mask >> slot) & 1U) << shape.num_colors;
            shape.color_formats[shape.num_colors++] = static_cast<u8>(key.color_formats[slot]);
        }
        shape.color_clear_mask = static_cast<u8>(clear_mask);
        shape.color_discard_mask = static_cast<u8>(discard_mask);
        if (Common::PerfReport::Enabled()) {
            const u64 pixels = u64{render_area.width} * render_area.height *
                               (std::max)(framebuffer->Layers(), 1U) *
                               (std::max)(static_cast<u32>(key.samples), 1U);
            u64 loads{};
            u64 stores{};
            for (u32 index = 0; index < shape.num_colors; ++index) {
                const u64 bytes = pixels * VideoCore::Surface::BytesPerBlock(
                                               static_cast<PixelFormat>(shape.color_formats[index]));
                loads += (shape.color_clear_mask >> index) & 1U ? 0 : bytes;
                stores += (shape.color_discard_mask >> index) & 1U ? 0 : bytes;
            }
            if (key.depth_format != PixelFormat::Invalid) {
                const u64 bytes = pixels * VideoCore::Surface::BytesPerBlock(key.depth_format);
                loads += shape.depth_clear ? 0 : bytes;
                stores += shape.depth_discard ? 0 : bytes;
            }
            Common::PerfReport::Add(Common::PerfReport::AttachmentLoadBytes, loads);
            Common::PerfReport::Add(Common::PerfReport::AttachmentStoreBytes, stores);
        }
    }
    state.renderpass = renderpass;
    state.clear_variant = false;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;
    state.pushed_rescaling_words.Forget();
    state.pushed_render_area.Forget();

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    const bool after_compute = std::exchange(compute_writes_pending, false);
    Record([this, renderpass, framebuffer_handle, render_area, values, clear_value_count,
            after_compute, time_pass, shape](vk::CommandBuffer cmdbuf) {
        if (time_pass) {
            PassTimestamp(cmdbuf, true, &shape, 0);
        }
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        if (after_compute) {
            // Draws read compute output as indirect arguments, vertices and resources; nothing
            // else orders those reads after the dispatch.
            static constexpr VkMemoryBarrier COMPUTE_TO_GRAPHICS{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, COMPUTE_TO_GRAPHICS);
        }
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count,
                        dc.color_clear_mask, dc.depth_stencil, color_discard_mask,
                        depth_stencil_discard);
    // Without discards, the variant only clears instead of loading: its store ops, layouts and
    // attachments are the framebuffer's own.
    state.clear_variant = color_discard_mask == 0 && !depth_stencil_discard;
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (RenderPassMatches(framebuffer)) {
        return false;
    }
    if (IsRenderPassActive()) {
        EndRenderPass();
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (RenderPassMatches(framebuffer)) {
        return false;
    }
    if (IsRenderPassActive()) {
        EndRenderPass();
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    pass_end_file = "deferred clear before a copy";
    pass_end_line = 0;
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer, std::source_location location) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    // A deferred clear's non-discarding pass serves the draws after its first one too: ending it
    // there and beginning the framebuffer's own pass stored the attachments only to load them back
    // (~9-11 render passes a frame in Ocarina of Time). MSAA tiler discard variants keep their
    // distinct renderpass identity so storeOp=DONT_CARE is not reused across separate passes.
    if (RenderPassMatches(framebuffer)) {
        return;
    }
    if (state.renderpass) {
        NotePassEnd(location);
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext(std::source_location location) {
    if (state.renderpass) {
        NotePassEnd(location);
    }
    EndRenderPass();
    if (pass_time_host_sampling) [[unlikely]] {
        // "Perf passes:" names the GPU time up to the next timestamp after the caller.
        Record([this, file = location.file_name(), line = location.line()](vk::CommandBuffer cmdbuf) {
            PassTimeLabel(cmdbuf, file, line);
        });
    }
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling, bool mesh_layout) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling &&
        mesh_layout == state.mesh_layout) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    state.mesh_layout = mesh_layout;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    const Common::DkbCpuTrace::Registration trace_thread{Common::DkbCpuTrace::Role::VulkanWorker};
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    std::stop_callback stop_cb{stop_token, [this] {
        std::scoped_lock lk{queue_mutex};
        work_cv.notify_all();
    }};

    std::vector<std::unique_ptr<CommandChunk>> batch;
    batch.reserve(16);

    while (!stop_token.stop_requested()) {
        batch.clear();
        bool notify_waiter = false;
        {
            std::unique_lock lk{queue_mutex};
            if (work_queue.empty()) {
                worker_waiting = true;
                work_cv.wait(lk, [&] {
                    return !work_queue.empty() || stop_token.stop_requested();
                });
                worker_waiting = false;
            }
            if (stop_token.stop_requested() && work_queue.empty()) {
                return;
            }
            while (!work_queue.empty()) {
                batch.push_back(std::move(work_queue.front()));
                work_queue.pop();
            }
            notify_waiter = waiter_waiting;

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in WaitWorker().
            void(std::exchange(lk, std::unique_lock{execution_mutex}));
            if (notify_waiter) {
                wait_cv.notify_all();
            }

            for (auto& work : batch) {
                const bool has_submit = work->HasSubmit();
                Common::Nx2Timeline::Span timeline_span{has_submit ? "record commands + submit"
                                                                   : "record commands"};
                work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);
                if (has_submit) {
                    AllocateWorkerCommandBuffer();
                }
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};
            for (auto& work : batch) {
                chunk_reserve.emplace_back(std::move(work));
            }
        }
    }
}

bool Scheduler::CheckIndexBufferBound(u64 vertex_hash, u64 fragment_hash) {
    if (index_buffer_bound) {
        return true;
    }
    ++unbound_index_draws;
    if (unbound_index_draws <= 8 || std::has_single_bit(unbound_index_draws)) {
        LOG_ERROR(HW_GPU,
                  "Indexed draw with no index buffer bound in its command buffer: skipped ({} so "
                  "far; shaders {:016x}/{:016x})",
                  unbound_index_draws, vertex_hash, fragment_hash);
    }
    return false;
}

void Scheduler::AllocateWorkerCommandBuffer() {
    index_buffer_bound = false;
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    static constexpr VkMemoryBarrier UPLOAD_ORDER_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    current_upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, UPLOAD_ORDER_BARRIER);
    if (trace_query_pool && !trace_gpu_failed) {
        TraceGpuSubmission();
    }
    BeginGpuTime();
    if (pass_time_sampling && !pass_time_reset && pass_time_next_query + 2 <= PassTimeQueries) {
        pass_time_open_span = pass_time_next_query;
        pass_time_next_query += 2;
        current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                             *pass_time_pool, pass_time_open_span);
    }
}

void Scheduler::TraceShader(u64 hash, u32 stage) {
    if (hash == 0) {
        return;
    }
    for (u32 i = 0; i < trace_work.num_shaders; ++i) {
        if (trace_work.shaders[i].hash == hash && trace_work.shaders[i].stage == stage) {
            return;
        }
    }
    if (trace_work.num_shaders == trace_work.shaders.size()) {
        ++trace_work.untracked_uses;
        return;
    }
    trace_work.shaders[trace_work.num_shaders++] = {hash, stage};
}

void Scheduler::TraceGraphicsPipeline(const std::array<u64, 6>& hashes) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hashes](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.draw_attempts;
        for (u32 stage = 0; stage < hashes.size(); ++stage) {
            TraceShader(hashes[stage], stage);
        }
    });
}

void Scheduler::TraceComputePipeline(u64 hash) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hash](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.dispatch_attempts;
        TraceShader(hash, 6);
    });
}

void Scheduler::ReportWorkTimes() {
    const u32 count = trace_work_count;
    trace_work_count = 0;
    if (count == 0) {
        return;
    }
    struct Result {
        u64 timestamp;
        u64 available;
    };
    std::vector<Result> results(count * 2);
    const auto result = device.GetLogical().GetQueryResults(
        *trace_work_pool, 0, count * 2, results.size() * sizeof(Result), results.data(),
        sizeof(Result), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) {
        return;
    }
    struct Totals {
        double total_ms{};
        double max_ms{};
        u32 count{};
        std::array<u32, 3> max_size{};
    };
    std::map<std::pair<u64, u64>, Totals> by_shaders;
    double total_ms{};
    for (u32 i = 0; i < count; ++i) {
        if (results[i * 2].available == 0 || results[i * 2 + 1].available == 0) {
            continue;
        }
        const double ms = static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                              results[i * 2].timestamp, results[i * 2 + 1].timestamp,
                              trace_timestamp_bits)) *
                          trace_timestamp_period / 1e6;
        const WorkTrace& item = trace_work_items[i];
        Totals& totals = by_shaders[{item.hash, item.fragment_hash}];
        totals.total_ms += ms;
        ++totals.count;
        if (ms > totals.max_ms) {
            totals.max_ms = ms;
            totals.max_size = item.size;
        }
        total_ms += ms;
    }
    std::vector<std::pair<std::pair<u64, u64>, Totals>> sorted(by_shaders.begin(),
                                                               by_shaders.end());
    std::ranges::sort(sorted, std::greater<>{},
                      [](const auto& entry) { return entry.second.total_ms; });
    if (sorted.size() > 8) {
        sorted.resize(8);
    }
    std::string slowest;
    for (const auto& [shaders, totals] : sorted) {
        const auto& [hash, fragment_hash] = shaders;
        slowest += fragment_hash == 0
                       ? fmt::format(" s6:{:016x}", hash)
                       : fmt::format(" s1:{:016x}+s5:{:016x}", hash, fragment_hash);
        slowest += fmt::format("={:.3f}ms/{}x(max {:.3f}ms size {}x{}x{})", totals.total_ms,
                               totals.count, totals.max_ms, totals.max_size[0],
                               totals.max_size[1], totals.max_size[2]);
    }
    LOG_INFO(Render_Vulkan,
             "DKB gpu_work renderer_frame={} timed={} timed_sum_ms={:.3f} slowest:{}",
             trace_sample_frame, count, total_ms, slowest);
}

void Scheduler::TraceGpuSubmission() {
    if (trace_gpu_pending) {
        master_semaphore->Refresh();
        if (!master_semaphore->IsFree(trace_gpu_tick)) {
            return;
        }
        struct Result {
            u64 timestamp;
            u64 available;
        };
        std::array<Result, TraceMaxSubmissions * 2> results{};
        const auto result = device.GetLogical().GetQueryResults(
            *trace_query_pool, 0, trace_gpu_count * 2, trace_gpu_count * 2 * sizeof(Result),
            results.data(), sizeof(Result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY) {
            return;
        }
        if (result != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp disabled: query result {}",
                        static_cast<int>(result));
            trace_gpu_failed = true;
            return;
        }
        double total_ms{};
        std::array<std::pair<double, u32>, 4> slowest{};
        for (u32 i = 0; i < trace_gpu_count * 2; i += 2) {
            if (results[i].available == 0 || results[i + 1].available == 0) {
                return;
            }
            const auto ticks = Common::DkbFrameTrace::TimestampDelta(
                results[i].timestamp, results[i + 1].timestamp, trace_timestamp_bits);
            const double duration_ms = static_cast<double>(ticks) * trace_timestamp_period / 1e6;
            total_ms += duration_ms;
            if (duration_ms >= 10.0 && duration_ms > slowest.back().first) {
                slowest.back() = {duration_ms, i / 2};
                std::ranges::sort(slowest, std::greater<>{});
            }
        }
        const double span_ms = trace_gpu_skipped == 0
            ? static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                  results[0].timestamp, results[trace_gpu_count * 2 - 1].timestamp,
                  trace_timestamp_bits)) * trace_timestamp_period / 1e6
            : -1.0;
        LOG_INFO(Render_Vulkan,
                 "DKB gpu_frame renderer_frame={} last_submission_tick={} "
                 "gpu_frame_span_ms={:.3f} gpu_submission_sum_ms={:.3f} "
                 "measured_submissions={} dropped_submissions={} "
                 "valid_bits={} period_ns={}",
                 trace_sample_frame, trace_gpu_tick, span_ms, total_ms, trace_gpu_count,
                 trace_gpu_skipped, trace_timestamp_bits, trace_timestamp_period);
        for (const auto& [duration_ms, index] : slowest) {
            if (duration_ms < 10.0) {
                continue;
            }
            const auto& work = trace_submissions[index];
            std::string shaders;
            for (u32 i = 0; i < work.num_shaders; ++i) {
                const auto& shader = work.shaders[i];
                shaders += fmt::format("{}s{}:{:016x}", shaders.empty() ? "" : ",",
                                       shader.stage, shader.hash);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB gpu_submission renderer_frame={} tick={} gpu_ms={:.3f} "
                     "draw_attempts={} dispatch_attempts={} shaders={} untracked_uses={}",
                     trace_sample_frame, work.tick, duration_ms, work.draw_attempts,
                     work.dispatch_attempts, shaders, work.untracked_uses);
        }
        ReportWorkTimes();
        trace_gpu_tick = 0;
        trace_gpu_count = 0;
        trace_gpu_skipped = 0;
        trace_gpu_pending = false;
        trace_gpu_rate.next = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    }
    if (!trace_gpu_sampling || trace_gpu_count == TraceMaxSubmissions) {
        return;
    }
    current_upload_cmdbuf.ResetQueryPool(*trace_query_pool, trace_gpu_count * 2, 2);
    if (trace_gpu_count == 0 && trace_work_pool) {
        // Draws are timed inside render passes, where queries cannot be reset.
        current_upload_cmdbuf.ResetQueryPool(*trace_work_pool, 0, TraceMaxWork * 2);
    }
    current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *trace_query_pool,
                                        trace_gpu_count * 2);
    trace_gpu_recording = true;
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (uploads_pending) {
        FlushUploads();
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        g_profile_submits.fetch_add(1, std::memory_order_relaxed);
        ReportSubmitCounts();
    }
    EndPendingOperations();
    InvalidateState();

    if (!wait_semaphore && next_wait_semaphore.load(std::memory_order_relaxed)) {
        wait_semaphore = next_wait_semaphore.exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
    }
    const u64 signal_value = master_semaphore->NextTick();
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        if (pass_time_pool) {
            PassTimesSubmitted(signal_value);
        }
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        if (trace_gpu_recording) {
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_query_pool,
                                 trace_gpu_count * 2 + 1);
            trace_gpu_tick = signal_value;
            trace_work.tick = signal_value;
            trace_submissions[trace_gpu_count] = trace_work;
            ++trace_gpu_count;
            trace_gpu_recording = false;
            trace_work = {};
        } else if (trace_gpu_sampling) {
            ++trace_gpu_skipped;
        }
        EndGpuTime(cmdbuf, signal_value);
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        VkResult result;
        {
            std::scoped_lock lock{submit_mutex};
            result = master_semaphore->SubmitQueue(cmdbuf, upload_cmdbuf, signal_semaphore,
                                                   wait_semaphore, signal_value);
        }
        switch (result) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
        if (trace_frame_end != 0) {
            if (trace_gpu_sampling) {
                trace_gpu_pending = true;
                trace_gpu_sampling = false;
            }
            if (trace_query_pool && !trace_gpu_failed && !trace_gpu_pending &&
                trace_gpu_rate.Ready(std::chrono::steady_clock::now())) {
                trace_sample_frame = trace_frame_end + 1;
                trace_gpu_sampling = true;
            }
            trace_frame_end = 0;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    has_pending_commands = false;
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    ++recording_epoch;
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state.pushed_rescaling_words.Forget();
    state.pushed_render_area.Forget();
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    // Pause, don't reset: a submission can land between a guest's counter reset and its report,
    // and resetting here would drop the samples counted so far (the report then reads too low).
    query_cache->CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        if (!state.renderpass) {
            return;
        }
        ++recording_epoch;
        fragment_barrier.Reset();
        render_pass_has_stores = false;
        last_pass_framebuffer = state.framebuffer;
        last_pass_end_file = std::exchange(pass_end_file, nullptr);
        last_pass_end_line = std::exchange(pass_end_line, 0U);

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        Record([this, num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported(),
                       time_pass = static_cast<bool>(pass_time_pool), draws = pass_draws](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            if (time_pass) {
                PassTimestamp(cmdbuf, false, nullptr, draws);
            }
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        state.renderpass = VkRenderPass{};
        state.clear_variant = false;
        num_renderpass_images = 0;
    }


void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
