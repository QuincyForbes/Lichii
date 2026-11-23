// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <thread>
#if defined(ARCHITECTURE_x86_64)
#include <immintrin.h>
#endif

#include "common/assert.h"
#include "common/dkb_cpu_trace.h"
#include "common/frame_deadline.h"
#include "common/logging.h"
#include "common/nx2_poll.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_hazard.h"
#include "video_core/nx2_job_stats.h"
#include "video_core/nx2_timeline.h"
#include "video_core/nx2.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

namespace {

/// Switch 2: start a command list no sooner than Nx2GpuLatency::Latency() after the game
/// submitted it, or once a game thread waits on the GPU (nx2_gpu_latency.h).
template <typename StopToken>
void HoldForGpuLatency(const SubmitListCommand& list, const StopToken& stop_token) {
    const auto latency = Nx2GpuLatency::Latency();
    if (latency.count() <= 0 || !Nx2GpuLatency::Steady() ||
        !Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
        Tegra::DmaPusher::TrustSubmitted()) {
        return;
    }
    const auto until = list.submit_time + latency;
    for (auto now = std::chrono::steady_clock::now(); now < until;
         now = std::chrono::steady_clock::now()) {
        if (Nx2GpuLatency::GuestWaits() != list.guest_waits || stop_token.stop_requested()) {
            return;
        }
        // Windows sleeps a millisecond or more at a time: sleep only while that much is left.
        if (until - now > std::chrono::microseconds{1500}) {
            std::this_thread::sleep_for(std::chrono::microseconds{500});
        } else {
            std::this_thread::yield();
        }
    }
}

// Tells a CPU thread waiting in PushCommand that its command ran.
void SignalCommand(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

} // Anonymous namespace

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuCommand};
        Common::SetCurrentThreadName("GPU");
        VideoCommon::GpuSplit::MarkGpuThread();
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        Common::SetCurrentThreadToPerformanceCores();
        Common::FrameDeadline::Join(Common::FrameDeadline::Role::GpuFront);
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // When a pending channel last moved or a command arrived (see the waits below).
        auto last_progress = std::chrono::steady_clock::now();
        constexpr std::chrono::nanoseconds PendingPoll = std::chrono::microseconds{500};
        constexpr std::chrono::nanoseconds PendingPollPeriod = std::chrono::microseconds{100};
        auto& wake = Tegra::Nx2PendingWake::Get();
        namespace Profile = VideoCommon::Nx2Profile;
        // Before the split's back thread starts, which reads it too.
        Profile::Initialize();
        auto& split = VideoCommon::GpuSplit::Get();
        u64 pending_map_position{};
        const auto wait_pending_maps = [&] {
            if (pending_map_position == 0) {
                return;
            }
            if (split.Active()) {
                const VideoCommon::Nx2JobStats::Timer drain_timer{
                    VideoCommon::Nx2JobStats::Kind::FrontDrain};
                split.WaitUntil(pending_map_position, VideoCommon::DrainSite::Map);
            }
            pending_map_position = 0;
        };
        while (!stop_token.stop_requested()) {
            Profile::Report();
            Common::FrameDeadline::Tick();
            split.MaybeStart(system);
            if (scheduler.HasPendingChannels()) {
                // Suspended channels wait on semaphores: a new command or a GPU release wakes the
                // wait below at once (Nx2PendingWake), so arm it before looking.
                wake.Arm();
                wait_pending_maps();
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                const bool popped = state.queue.TryPop(next);
                if (resumed || popped) {
                    last_progress = std::chrono::steady_clock::now();
                }
                if (!popped) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    // EDEN_NX2_PUBLISH_WAITS: what the resumed channels sent goes to the back
                    // before this thread waits or sleeps (Commit publishes every 64 words).
                    if (split.Active()) {
                        split.PublishBeforeWait();
                    }
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    const int poll = VideoCore::Nx2Debug::Experiment(6);
                    const auto now = std::chrono::steady_clock::now();
                    if (!Common::PollYield()) {
                        (void)wake.Sleep(std::chrono::microseconds{50});
                    } else if (now - last_progress < PendingPoll) {
                        const auto poll_end =
                            (std::min)(now + PendingPollPeriod, last_progress + PendingPoll);
                        wake.Poll(poll_end);
                    } else {
                        Profile::Add(Profile::Counter::PendingWaits, 1);
                        const u32 reason = wake.Sleep(poll == 1 ? std::chrono::microseconds{50}
                                                                : std::chrono::microseconds{500});
                        const bool by_push = (reason & Tegra::Nx2PendingWake::ByPush) != 0;
                        const bool by_latch =
                            !by_push && (reason & Tegra::Nx2PendingWake::ByLatch) != 0;
                        Profile::Add(Profile::Counter::PendingWakePush, by_push ? 1 : 0);
                        Profile::Add(Profile::Counter::PendingWakeLatch, by_latch ? 1 : 0);
                        Profile::Add(Profile::Counter::PendingWakeTimeout, reason == 0 ? 1 : 0);
                    }
                    continue;
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (!std::holds_alternative<GPUTickCommand>(next.data)) {
                last_progress = std::chrono::steady_clock::now();
            }
            // A blocking (synchronous GPU) submit already keeps the game thread waiting.
            if (const auto* list = std::get_if<SubmitListCommand>(&next.data);
                list != nullptr && !next.block) {
                HoldForGpuLatency(*list, stop_token);
            }
            if (split.Active()) {
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                auto* const rasterizer_ptr = renderer.ReadRasterizer();
                if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                    wait_pending_maps();
                    Profile::Scope scope{Profile::Zone::Push};
                    if (VideoCommon::Nx2Hazard::Enabled()) {
                        split.Defer([epoch = submit_list->epoch] {
                            VideoCommon::Nx2Hazard::SetListEpoch(epoch);
                        });
                    }
                    const u64 timeline_seq = submit_list->timeline_seq;
                    scheduler.Push(system.GPU(), submit_list->channel, std::move(*submit_list->entries));
                    if (timeline_seq != 0) {
                        VideoCommon::Nx2Timeline::OnFront(timeline_seq);
                        split.Defer([timeline_seq] { VideoCommon::Nx2Timeline::OnBack(timeline_seq); });
                    }
                } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); });
                } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                    if (run->before_later_reads) {
                        // Coalesce consecutive map updates on GPUBack and wait once before the
                        // next front-side command read, without requesting the idle commit hook.
                        split.DeferFunction(std::move(run->func), true);
                        pending_map_position = split.WritePosition();
                    } else {
                        split.DeferFunction(std::move(run->func), run->changes_mappings);
                        if (run->changes_mappings) {
                            split.NoteMemoryWrite();
                        }
                    }
                } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                    split.Defer([rasterizer_ptr, addr = flush->addr, size = flush->size] {
                        rasterizer_ptr->FlushRegion(addr, size);
                    });
                } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                    split.Defer(
                        [rasterizer_ptr, addr = invalidate->addr, size = invalidate->size] {
                            rasterizer_ptr->OnCacheInvalidation(addr, size);
                        },
                        VideoCommon::GpuSplit::Effect::None);
                    // It may drop programs that draws queued after it run: early MME reads
                    // past those draws check it like a noted write (MacroReadWriterSet).
                    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet)) {
                        split.NoteDeviceRange(invalidate->addr, invalidate->size);
                    }
                } else {
                    ASSERT(false);
                }
                split.Defer(
                    [state_ptr = &state, fence = next.fence, block = next.block] {
                        SignalCommand(*state_ptr, fence, block);
                    },
                    VideoCommon::GpuSplit::Effect::None);
                split.Publish();
                continue;
            }
            pending_map_position = 0;
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Push};
                VideoCommon::Nx2Hazard::SetListEpoch(submit_list->epoch);
                const u64 timeline_seq = submit_list->timeline_seq;
                scheduler.Push(system.GPU(), submit_list->channel, std::move(*submit_list->entries));
                VideoCommon::Nx2Timeline::OnFront(timeline_seq);
                VideoCommon::Nx2Timeline::OnBack(timeline_seq);
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
            } else if (auto* run = std::get_if<RunCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::Run};
                run->func();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            SignalCommand(state, next.fence, next.block);
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::Run(std::function<void()>&& func, bool is_async, bool before_later_reads,
                        bool changes_mappings) {
    PushCommand(RunCommand{std::move(func), before_later_reads, changes_mappings}, false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
    Tegra::Nx2SemaphoreWaiters::Get().NoteCpuStore(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (VideoCore::Nx2Debug::AccuratePart(8)) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
    Tegra::Nx2SemaphoreWaiters::Get().NoteCpuStore(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);
    Tegra::Nx2PendingWake::Get().Notify(Tegra::Nx2PendingWake::ByPush);

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        Tegra::Nx2PendingWake::Get().Notify(Tegra::Nx2PendingWake::ByPush);
        thread.join();
    }
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
