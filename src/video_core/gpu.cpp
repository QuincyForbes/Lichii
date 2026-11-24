// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/perf_stats.h"
#include "video_core/buffer_cache/cpu_read_stats.h"
#include "video_core/buffer_cache/hot_pages.h"
#include "video_core/cdma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/due_composites.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/pending_maps.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_timestamp.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_base.h"
#include "video_core/shader_notify.h"

namespace Tegra {

struct GPU::Impl {
    explicit Impl(Core::System& system_, bool is_async_, bool use_nvdec_)
        : system{system_}
        , gpu_thread{system_}
        , use_nvdec{use_nvdec_}
        , shader_notify()
        , is_async{is_async_}
    {}

    ~Impl() {
        NotifyShutdown();
        // Rasterizer teardown flushes commands and invalidates the bound engine's dirty flags.
        // Keep channels, their scheduler references, and the graphics context alive until then.
        renderer.reset();
    }

    std::shared_ptr<Control::ChannelState> CreateChannel(s32 channel_id) {
        auto channel_state = std::make_shared<Tegra::Control::ChannelState>(channel_id);
        channels.emplace(channel_id, channel_state);
        {
            std::scoped_lock lk{channel_list_mutex};
            channel_list.push_back(channel_state);
        }
        scheduler.DeclareChannel(channel_state);
        return channel_state;
    }

    void BindChannel(s32 channel_id) {
        if (bound_channel != channel_id) {
            auto it = channels.find(channel_id);
            ASSERT(it != channels.end());
            bound_channel = channel_id;
            current_channel = it->second.get();
            renderer->ReadRasterizer()->BindChannel(*current_channel);
        }
    }

    std::shared_ptr<Control::ChannelState> AllocateChannel() {
        return CreateChannel(new_channel_id++);
    }

    void InitChannel(Control::ChannelState& to_init, u64 program_id) {
        to_init.Init(system, program_id);
        to_init.BindRasterizer(renderer->ReadRasterizer());
        renderer->ReadRasterizer()->InitializeChannel(to_init);
    }

    void InitAddressSpace(Tegra::MemoryManager& memory_manager) {
        memory_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    void ReleaseChannel(Control::ChannelState& to_release) {
        UNIMPLEMENTED();
    }

    /// Binds a renderer to the GPU.
    void BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer_) {
        renderer = std::move(renderer_);
        system.Host1x().memory_manager.BindInterface(renderer->ReadRasterizer());
        system.Host1x().gmmu_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    /// Flush all current written commands into the host GPU for execution.
    void FlushCommands() {
        renderer->ReadRasterizer()->FlushCommands();
    }

    /// Synchronizes CPU writes with Host GPU memory.
    void InvalidateGPUCache() {
        static const bool fast_path = [] {
            const char* value = std::getenv("EDEN_NX2_DIRTY_FASTPATH");
            return value == nullptr || value[0] != '0';
        }();
        static const bool any_flag = [] {
            const char* value = std::getenv("EDEN_NX2_DIRTY_ANY");
            return value == nullptr || value[0] != '0';
        }();
        if (fast_path && any_flag) {
            if (!Core::GPUDirtyMemoryManager::AnyWrites()) {
                return;
            }
            Core::GPUDirtyMemoryManager::BeginGather();
        } else if (fast_path) {
            bool may_have_writes = false;
            for (auto& manager : system.GetGPUDirtyMemoryManager()) {
                may_have_writes |= manager.MayHaveWrites();
            }
            if (!may_have_writes) {
                return;
            }
        }
        // Runs at every fence (~17k/s), on more than one thread: keep each thread's buffer.
        thread_local std::vector<std::pair<DAddr, u64>> thread_writes;
        auto& writes = thread_writes;
        writes.clear();
        std::function<void(PAddr, size_t)> callback_writes(
            [&writes](PAddr address, size_t size) { writes.emplace_back(address, size); });
        system.GatherGPUDirtyMemory(callback_writes);
        if (fast_path && any_flag) {
            Core::GPUDirtyMemoryManager::EndGather();
        }
        if (!writes.empty()) {
            renderer->ReadRasterizer()->OnCacheInvalidations(writes);
        }
    }

    void ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
        if (shutting_down.load(std::memory_order_acquire)) {
            return;
        }
        for (auto& manager : system.GetGPUDirtyMemoryManager()) {
            manager.ForEachPending(callback);
        }
    }

    /// Signal the ending of command list.
    void OnCommandListEnd() {
        VideoCommon::HotPages::g_command_lists.fetch_add(1, std::memory_order_relaxed);
        renderer->ReadRasterizer()->ReleaseFences(false);
        Settings::UpdateGPUAccuracy();
    }

    /// Request a host GPU memory flush from the CPU.
    template <typename Func>
    [[nodiscard]] u64 RequestSyncOperation(Func&& action) {
        std::unique_lock lck{sync_request_mutex};
        const u64 fence = ++last_sync_fence;
        if (sync_requests_dropped) {
            // The GPU threads stopped (NotifyShutdown): nothing runs it, and its waiter returns at
            // once. Not queued, so a stray TickWork can't run it after its waiter's frame is gone.
            return fence;
        }
        sync_requests.emplace_back(std::forward<Func>(action));
        sync_request_count.fetch_add(1, std::memory_order_release);
        return fence;
    }

    /// Obtains current flush request fence id.
    [[nodiscard]] u64 CurrentSyncRequestFence() const {
        return current_sync_fence.load(std::memory_order_relaxed);
    }

    /// Returns once the request of `fence` ran, or once the GPU threads stopped for good
    /// (NotifyShutdown): then it never runs.
    void WaitForSyncOperation(const u64 fence) {
        std::unique_lock lck{sync_request_mutex};
        sync_request_cv.wait(lck, [this, fence] {
            return CurrentSyncRequestFence() >= fence || sync_requests_dropped;
        });
    }

    void WaitForIdle() {
        const u64 fence = RequestSyncOperation([] {});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
    }

    /// Tick pending requests within the GPU.
    void TickWork() {
        // Called once per draw: no lock while nothing is queued.
        if (sync_request_count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::unique_lock lck{sync_request_mutex};
        while (!sync_requests.empty()) {
            auto request = std::move(sync_requests.front());
            sync_requests.pop_front();
            sync_request_count.fetch_sub(1, std::memory_order_relaxed);
            sync_request_mutex.unlock();
            request();
            current_sync_fence.fetch_add(1, std::memory_order_release);
            sync_request_mutex.lock();
            sync_request_cv.notify_all();
        }
    }

    [[nodiscard]] u64 GetTicks() const {
        const bool ampere = g_nx2_ampere.load(std::memory_order_relaxed) && !LegacyTimestamps();
        return ReadGpuTimestamp(system.CoreTiming(), ampere, Settings::values.gpu_clock.GetValue(),
                                present_ns.load(std::memory_order_relaxed));
    }

    /// Whether the running title keeps GPU tick timestamps (UsesLegacyAmpereTimestamps). Known
    /// once the application process exists; until then (program id 0) nanoseconds are used.
    [[nodiscard]] bool LegacyTimestamps() const {
        s8 known = legacy_timestamps.load(std::memory_order_relaxed);
        if (known < 0) {
            const u64 program_id = system.GetApplicationProcessProgramID();
            if (program_id == 0) {
                return false;
            }
            known = UsesLegacyAmpereTimestamps(program_id) ? 1 : 0;
            legacy_timestamps.store(known, std::memory_order_relaxed);
        }
        return known != 0;
    }

    void RendererFrameEndNotify(u32 generated_frames) {
        present_ns.store(static_cast<u64>(system.CoreTiming().GetGlobalTimeNs().count()),
                         std::memory_order_relaxed);
        system.GetPerfStats().EndGameFrame(generated_frames);
    }

    /// Performs any additional setup necessary in order to begin GPU emulation.
    /// This can be used to launch any necessary threads and register any necessary
    /// core timing events.
    void Start() {
        Settings::UpdateGPUAccuracy();
        gpu_thread.StartThread(*renderer, renderer->Context(), scheduler);
    }

    void NotifyShutdown() {
        // Stops the GPU thread and the split's back (GpuSplit::Stop): no TickWork runs after it.
        gpu_thread.NotifyShutdown();
        due_composites.Stop();
        // Host1x outlives GPU in System::Shutdown: cancel any unreached composite fence actions
        // (and wait for any in-flight one) before this Impl or renderer goes away.
        boost::container::small_vector<
            std::pair<u32, Host1x::SyncpointManager::ActionHandle>, 16>
            pending_swap_handles;
        {
            std::scoped_lock swap_lock{request_swap_mutex};
            swap_requests_stopped = true;
            for (auto& req : swap_requests) {
                for (const auto& entry : req.handles) {
                    pending_swap_handles.push_back(entry);
                }
                req.handles.clear();
                req.layers.clear();
                req.pending_fences = 0;
            }
        }
        if (!pending_swap_handles.empty()) {
            auto& syncpoint_manager = system.Host1x().GetSyncpointManager();
            for (const auto& [id, handle] : pending_swap_handles) {
                syncpoint_manager.DeregisterGuestAction(id, handle);
            }
        }
        {
            // The GPU threads stopped: no sync operation runs anymore. Guest threads still run
            // until the kernel suspends them, and one waiting for a read's flush (OnCPURead)
            // would wait forever, in host code where the kernel cannot suspend it
            // (KernelCore::SuspendEmulation then waits for it forever). Drop the requests, which
            // may refer to their waiters' frames, and release the waiters.
            std::scoped_lock requests_lock{sync_request_mutex};
            sync_requests.clear();
            sync_request_count.store(0, std::memory_order_relaxed);
            sync_requests_dropped = true;
        }
        sync_request_cv.notify_all();
        {
            std::scoped_lock channels_lock{channel_list_mutex};
            channel_list.clear();
        }
        std::unique_lock lk{sync_mutex};
        shutting_down.store(true, std::memory_order::release);
        sync_cv.notify_all();
    }

    /// Obtain the CPU Context
    void ObtainContext() {
        if (!cpu_context) {
            cpu_context = renderer->GetRenderWindow().CreateSharedContext();
        }
        cpu_context->MakeCurrent();
    }

    /// Release the CPU Context
    void ReleaseContext() {
        cpu_context->DoneCurrent();
    }

    /// Push GPU command entries to be processed
    void PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
        if (!first_submit_logged.exchange(true, std::memory_order_relaxed)) {
            LOG_INFO(HW_GPU, "Boot: first GPU command list, channel {}, {} entries", channel,
                     entries.command_lists.size());
        }
        // Only drain queued maps before front-side command parsing when an entry's pushbuffer
        // words were deferred behind a pending map, or when snapshots may re-read memory.
        const bool uncopied_overlap =
            !entries.command_lists.empty() && entries.submitted_offsets.empty() &&
            std::ranges::any_of(entries.command_lists, [this](const auto& entry) {
                return entry.size != 0 &&
                       pending_maps.Overlaps(entry.addr, u64{entry.size} * sizeof(u32));
            });
        if (!entries.deferred_entries.empty() || uncopied_overlap ||
            (!entries.command_lists.empty() &&
             Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
             !Tegra::DmaPusher::TrustSubmitted() &&
             pending_maps.Overlaps(0, ~u64{0}))) {
            gpu_thread.Run([] {}, true, true, false);
        }
        gpu_thread.SubmitList(channel, std::move(entries), is_async);
    }

    void AdvanceRetiredReleaseGen(u64 gen) {
        u64 prev = retired_release_gen.load(std::memory_order_relaxed);
        while (prev < gen &&
               !retired_release_gen.compare_exchange_weak(prev, gen, std::memory_order_release)) {
        }
    }

    // Sparse tile updates, idempotent re-maps, fresh VA mappings and VA remaps run on GPUBack in
    // ring order without stalling GPUFront on DrainSite::Map; pending_maps and PushGPUEntries
    // drain before parsing any command list whose pushbuffer overlaps a queued map, and
    // Nx2CommandMemoryReleases keeps earlier lists on their submit-time snapshots.
    bool CanOrderMapWithoutFrontDrain(GPUVAddr gpu_addr, u64 size, bool& already_mapped) {
        already_mapped = false;
        static const bool force_drain = [] {
            const char* value = std::getenv("EDEN_NX2_MAP_DRAIN");
            return value != nullptr && value[0] != '0';
        }();
        if (force_drain) {
            return false;
        }
        if (size == 0 || Tegra::Nx2InSparseRegion(gpu_addr, size)) {
            already_mapped = true;
            return true;
        }
        if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
            !Tegra::Nx2CommandMemoryReleases::Enabled()) {
            return false;
        }
        boost::container::small_vector<const Tegra::MemoryManager*, 4> spaces;
        {
            std::scoped_lock lk{channel_list_mutex};
            for (const auto& ch : channel_list) {
                if (const auto* mm = ch->memory_manager; mm != nullptr &&
                    std::ranges::find(spaces, mm) == spaces.end()) {
                    spaces.push_back(mm);
                }
            }
        }
        if (spaces.empty()) {
            return true;
        }
        bool all_unmapped = true;
        bool all_continuous = true;
        bool had_proven_code = false;
        for (const auto* mm : spaces) {
            const auto submapped = mm->GetSubmappedDeviceRange(gpu_addr, size);
            if (!submapped.empty()) {
                all_unmapped = false;
                for (const auto& [dev_addr, dev_bytes] : submapped) {
                    if (Tegra::Nx2IsProvenCode(dev_addr, dev_bytes)) {
                        had_proven_code = true;
                        break;
                    }
                }
            }
            if (submapped.size() != 1 || submapped[0].second != size) {
                all_continuous = false;
            }
        }
        if (had_proven_code) {
            Tegra::Nx2BumpCodeEpoch();
        }
        if (all_unmapped) {
            return true;
        }
        if (all_continuous && !pending_maps.Overlaps(gpu_addr, size)) {
            auto& releases = Tegra::Nx2CommandMemoryReleases::Get();
            const u64 now = releases.Generation();
            const u64 since = retired_release_gen.load(std::memory_order_acquire);
            if (now <= since) {
                already_mapped = true;
                return true;
            }
            thread_local std::vector<Tegra::Nx2CommandMemoryReleases::Range> overlapped;
            if (releases.Collect(since, now, gpu_addr, gpu_addr + size, overlapped) &&
                overlapped.empty()) {
                already_mapped = true;
                return true;
            }
        }
        return true;
    }

    // Queued without waiting, also with synchronous GPU: callers hold nx2:gpu locks that the GPU
    // back thread takes while running the map (flushes write back into sparse regions).
    void RunInOrder(std::function<void()>&& func, bool changes_mappings) {
        if (!changes_mappings || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            gpu_thread.Run(std::move(func), true, false, changes_mappings);
            return;
        }
        const u64 gen = Tegra::Nx2CommandMemoryReleases::Get().Generation();
        gpu_thread.Run(
            [this, gen, func = std::move(func)] {
                func();
                AdvanceRetiredReleaseGen(gen);
            },
            true, false, false);
    }

    void NotifySyncpointWait() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_SYNCPT_WAIT_FLUSH");
            return value == nullptr || value[0] != '0';
        }();
        if (!enabled || shutting_down.load(std::memory_order_relaxed) || !system.IsPoweredOn()) {
            return;
        }
        auto& split = VideoCommon::GpuSplit::Get();
        if (split.BackRunning()) {
            split.RequestIdleHook();
        }
        if (!syncpt_flush_queued.exchange(true, std::memory_order_acq_rel)) {
            RunInOrder(
                [this] {
                    syncpt_flush_queued.store(false, std::memory_order_release);
                    if (shutting_down.load(std::memory_order_relaxed) || renderer == nullptr) {
                        return;
                    }
                    auto& gpu_split = VideoCommon::GpuSplit::Get();
                    if (gpu_split.BackRunning()) {
                        gpu_split.RequestIdleHook();
                    } else if (VideoCommon::GpuSplit::OwnsScheduler()) {
                        renderer->ReadRasterizer()->ReleaseFences(false);
                    }
                },
                false);
        }
    }

    void MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
        const u64 gen = Tegra::Nx2CommandMemoryReleases::Get().Generation();
        bool already_mapped = false;
        if (CanOrderMapWithoutFrontDrain(gpu_addr, size, already_mapped)) {
            if (already_mapped) {
                gpu_thread.Run(
                    [this, gen, func = std::move(func)] {
                        func();
                        AdvanceRetiredReleaseGen(gen);
                    },
                    true, false, false);
                return;
            }
            const u64 id = pending_maps.Add(gpu_addr, size);
            gpu_thread.Run(
                [this, id, gen, func = std::move(func)] {
                    func();
                    pending_maps.Remove(id);
                    AdvanceRetiredReleaseGen(gen);
                },
                true, false, false);
            return;
        }
        const u64 id = pending_maps.Add(gpu_addr, size);
        gpu_thread.Run(
            [this, id, gen, func = std::move(func)] {
                func();
                pending_maps.Remove(id);
                AdvanceRetiredReleaseGen(gen);
            },
            true, true); // not waited for, as in RunInOrder
    }

    bool IsMapPending(GPUVAddr gpu_addr, u64 size) const {
        return pending_maps.Overlaps(gpu_addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(DAddr addr, u64 size) {
        gpu_thread.FlushRegion(addr, size, is_async);
    }

    VideoCore::RasterizerDownloadArea OnCPURead(DAddr addr, u64 size) {
        namespace ReadStats = VideoCommon::CpuReadStats;
        auto& stats = ReadStats::g_counters;
        const u64 ask_begin = ReadStats::NowNs();
        auto raster_area = renderer->ReadRasterizer()->GetFlushArea(addr, size);
        const u64 ask_end = ReadStats::NowNs();
        ReadStats::Add(stats.asks, 1);
        ReadStats::Add(stats.ask_ns, ask_end - ask_begin);
        if (raster_area.preemtive) {
            return raster_area;
        }
        raster_area.preemtive = true;
        // A guest CPU read of GPU-written memory (reactive flushing): count them, they decide
        // whether buffer data must be copied back at every fence (EDEN_NX2_BUFFER_DOWNLOADS).
        if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
            if (static std::atomic<u64> reads{};
                (reads.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
                LOG_INFO(HW_GPU, "CPU read of GPU-written memory [{:#x}, {:#x}) ({} so far)",
                         raster_area.start_address, raster_area.end_address,
                         reads.load(std::memory_order_relaxed));
            }
        }
        if (raster_area.download_tick != 0) {
            if (renderer->ReadRasterizer()->LandDownloadForCpuRead(
                    raster_area.start_address,
                    raster_area.end_address - raster_area.start_address,
                    raster_area.download_tick)) {
                ReadStats::Add(stats.core_landed, 1);
                ReadStats::Add(stats.core_ns, ReadStats::NowNs() - ask_end);
                return raster_area;
            }
            ReadStats::Add(stats.core_refused, 1);
        }
        const size_t cause = (std::min<size_t>)(raster_area.cause, ReadStats::CauseCount - 1);
        const u64 fence = RequestSyncOperation([this, raster_area, ask_end, cause]() {
            auto& stats = ReadStats::g_counters;
            const u64 flush_begin = ReadStats::NowNs();
            ReadStats::Add(stats.queue_ns[cause], flush_begin - ask_end);
            // Apply collected guest writes before downloading GPU data over guest memory.
            InvalidateGPUCache();
            ReadStats::t_in_read = true;
            renderer->ReadRasterizer()->FlushRegion(raster_area.start_address, raster_area.end_address - raster_area.start_address);
            ReadStats::t_in_read = false;
            ReadStats::Add(stats.flush_ns[cause], ReadStats::NowNs() - flush_begin);
        });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
        ReadStats::Add(stats.waits[cause], 1);
        return raster_area;
    }

    bool IsCleanReadTrapPage(DAddr page_addr) const {
        return renderer->ReadRasterizer()->IsCleanReadTrapPage(page_addr);
    }

    std::optional<VideoCore::RasterizerDownloadArea> TryFastCleanRead(DAddr page_begin,
                                                                      u64 page_bytes) {
        namespace ReadStats = VideoCommon::CpuReadStats;
        const u64 ask_begin = ReadStats::NowNs();
        auto area = renderer->ReadRasterizer()->TryFastCleanRead(page_begin, page_bytes);
        if (area) {
            auto& stats = ReadStats::g_counters;
            ReadStats::Add(stats.asks, 1);
            ReadStats::Add(stats.ask_ns, ReadStats::NowNs() - ask_begin);
        }
        return area;
    }

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
        if (preserve_gpu_writes) {
            VideoCore::RasterizerInterface* rasterizer = renderer->ReadRasterizer();
            if (!rasterizer->TryCacheInvalidation(addr, size)) {
                const u64 fence = RequestSyncOperation([rasterizer, addr, size] {
                    rasterizer->FlushRegion(addr, size, VideoCommon::CacheType::BufferCache);
                    rasterizer->OnCacheInvalidation(addr, size);
                });
                gpu_thread.TickGPU(is_async);
                WaitForSyncOperation(fence);
            }
            return;
        }
        gpu_thread.InvalidateRegion(addr, size);
    }

    bool OnCPUWrite(DAddr addr, u64 size) {
        return renderer->ReadRasterizer()->OnCPUWrite(addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(DAddr addr, u64 size) {
        gpu_thread.FlushAndInvalidateRegion(addr, size, is_async);
    }

    void RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers, std::vector<Service::Nvidia::NvFence>&& fences) {
        if (fences.empty()) {
            pending_composite_fence = RequestSyncOperation(
                [this, composite_layers = std::move(layers)]() mutable {
                    CompositeOnOwner(std::move(composite_layers));
                });
            gpu_thread.TickGPU(is_async);
            return;
        }
        pending_composite_fence = RequestSyncOperation(
            [this, composite_layers = std::move(layers),
             composite_fences = std::move(fences)]() mutable {
                const size_t num_fences = composite_fences.size();
                u32 slot{};
                u32 gen{};
                {
                    std::scoped_lock lk{request_swap_mutex};
                    if (swap_requests_stopped) {
                        return;
                    }
                    if (free_swap_counters.empty()) {
                        slot = static_cast<u32>(swap_requests.size());
                        swap_requests.emplace_back();
                    } else {
                        slot = free_swap_counters.front();
                        free_swap_counters.pop_front();
                    }
                    auto& req = swap_requests[slot];
                    gen = ++req.generation;
                    req.pending_fences = num_fences;
                    req.layers = std::move(composite_layers);
                    req.handles.clear();
                }
                auto& syncpoint_manager = system.Host1x().GetSyncpointManager();
                for (size_t i = 0; i < num_fences; i++) {
                    const u32 id = static_cast<u32>(composite_fences[i].id);
                    const auto handle = syncpoint_manager.RegisterGuestAction(
                        id, composite_fences[i].value,
                        [this, slot, gen] { OnSwapFenceSignaled(slot, gen); });
                    if (handle != 0) {
                        std::scoped_lock lk{request_swap_mutex};
                        auto& req = swap_requests[slot];
                        if (!swap_requests_stopped && req.generation == gen &&
                            req.pending_fences != 0) {
                            req.handles.emplace_back(id, handle);
                        }
                    }
                }
            });
        gpu_thread.TickGPU(is_async);
    }

    void OnSwapFenceSignaled(u32 slot, u32 gen) {
        VideoCommon::DueComposites::Layers ready_layers;
        {
            std::scoped_lock lk{request_swap_mutex};
            auto& req = swap_requests[slot];
            if (swap_requests_stopped || req.generation != gen || req.pending_fences == 0) {
                return;
            }
            if (--req.pending_fences != 0) {
                return;
            }
            ready_layers = std::move(req.layers);
            req.handles.clear();
            free_swap_counters.push_back(slot);
        }
        CompositeOnOwner(std::move(ready_layers));
    }

    /// A present whose fences were all reached, on the thread that reached the last one. Every
    /// present goes through due_composites, so none overtakes one queued for the owner.
    void CompositeOnOwner(VideoCommon::DueComposites::Layers&& layers) {
        const auto composite = [this](const VideoCommon::DueComposites::Layers& due_layers) {
            renderer->Composite(due_layers);
        };
        // A fence another thread reached waits for the owner's next tick. That thread may be one
        // the GPU threads wait for, so it doesn't wait.
        due_composites.Reached(
            std::move(layers), VideoCommon::GpuSplit::OwnsScheduler(), composite, [this] {
                (void)RequestSyncOperation([this] {
                    due_composites.Run(
                        [this](const VideoCommon::DueComposites::Layers& due_layers) {
                            renderer->Composite(due_layers);
                        });
                });
                gpu_thread.TickGPU(true);
            });
    }

    void WaitForComposite() {
        const u64 fence = pending_composite_fence;
        if (fence == 0) {
            return;
        }
        pending_composite_fence = 0;
        if (shutting_down.load(std::memory_order_relaxed)) {
            return;
        }
        WaitForSyncOperation(fence);
    }

    std::vector<u8> GetAppletCaptureBuffer() {
        std::vector<u8> out;

        const auto wait_fence =
            RequestSyncOperation([&] { out = renderer->GetAppletCaptureBuffer(); });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(wait_fence);

        return out;
    }

    Core::System& system;

    // Destruction of thread must be done before all (non trivial)
    // previous members has been destroyed
    VideoCommon::GPUThread::ThreadManager gpu_thread;

    std::unique_ptr<VideoCore::RendererBase> renderer;
    const bool use_nvdec;

    s32 new_channel_id{1};
    /// Shader build notifier
    VideoCore::ShaderNotify shader_notify;
    /// When true, we are about to shut down emulation session, so terminate outstanding tasks
    std::atomic_bool shutting_down{};

    std::array<std::atomic<u32>, Service::Nvidia::MaxSyncPoints> syncpoints{};

    std::array<std::list<u32>, Service::Nvidia::MaxSyncPoints> syncpt_interrupts;

    std::mutex sync_mutex;
    std::mutex device_mutex;

    std::condition_variable sync_cv;

    std::list<std::function<void()>> sync_requests;
    std::atomic<u32> sync_request_count{};
    std::atomic<u64> current_sync_fence{};
    std::atomic<u64> present_ns{};
    mutable std::atomic<s8> legacy_timestamps{-1};
    u64 last_sync_fence{};

    VideoCommon::PendingMaps pending_maps;
    std::mutex channel_list_mutex;
    std::vector<std::shared_ptr<Tegra::Control::ChannelState>> channel_list;
    std::atomic<u64> retired_release_gen{Tegra::Nx2CommandMemoryReleases::Get().Generation()};
    std::atomic<bool> syncpt_flush_queued{};
    std::mutex sync_request_mutex;
    std::condition_variable sync_request_cv;
    /// The GPU threads stopped and the requests were dropped (NotifyShutdown, under the mutex).
    bool sync_requests_dropped{};

    const bool is_async;
    /// Boot milestone: the guest's first command list.
    std::atomic<bool> first_submit_logged{};

    std::unique_ptr<Core::Frontend::GraphicsContext> cpu_context;

    Tegra::Control::Scheduler scheduler;
    ::Common::unordered_map<s32, std::shared_ptr<Tegra::Control::ChannelState>> channels;
    Tegra::Control::ChannelState* current_channel;
    s32 bound_channel{-1};

    struct SwapRequest {
        size_t pending_fences{};
        u32 generation{};
        VideoCommon::DueComposites::Layers layers;
        boost::container::small_vector<
            std::pair<u32, Host1x::SyncpointManager::ActionHandle>, 4>
            handles;
    };
    std::deque<u32> free_swap_counters;
    std::deque<SwapRequest> swap_requests;
    std::mutex request_swap_mutex;
    bool swap_requests_stopped{};
    VideoCommon::DueComposites due_composites;
    u64 pending_composite_fence{};
};

GPU::GPU(Core::System& system, bool is_async, bool use_nvdec)
    : impl{std::make_unique<Impl>(system, is_async, use_nvdec)}
{}

GPU::~GPU() = default;

std::shared_ptr<Control::ChannelState> GPU::AllocateChannel() {
    return impl->AllocateChannel();
}

void GPU::InitChannel(Control::ChannelState& to_init, u64 program_id) {
    impl->InitChannel(to_init, program_id);
}

void GPU::BindChannel(s32 channel_id) {
    impl->BindChannel(channel_id);
}

void GPU::ReleaseChannel(Control::ChannelState& to_release) {
    impl->ReleaseChannel(to_release);
}

void GPU::InitAddressSpace(Tegra::MemoryManager& memory_manager) {
    impl->InitAddressSpace(memory_manager);
}

void GPU::BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer) {
    impl->BindRenderer(std::move(renderer));
}

void GPU::FlushCommands() {
    impl->FlushCommands();
}

void GPU::InvalidateGPUCache() {
    impl->InvalidateGPUCache();
}

bool GPU::HasPendingCPUWrites(DAddr addr, u64 size) {
    for (auto& manager : impl->system.GetGPUDirtyMemoryManager()) {
        if (manager.MayHaveWrites() && manager.Overlaps(addr, size)) {
            return true;
        }
    }
    return false;
}

void GPU::ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
    if (!impl) {
        return;
    }
    impl->ForEachPendingCpuWrite(callback);
}

void GPU::CollectHandoffWrite(DAddr addr, u64 size) {
    impl->system.GetGPUDirtyMemoryManager().back().Collect(addr, size);
}

void GPU::OnCommandListEnd() {
    impl->OnCommandListEnd();
}

u64 GPU::RequestFlush(DAddr addr, std::size_t size) {
    return impl->RequestSyncOperation([this, addr, size]() {
        impl->renderer->ReadRasterizer()->FlushRegion(addr, size);
    });
}

u64 GPU::RunOnGpuThread(std::function<void()>&& action) {
    return impl->RequestSyncOperation(std::move(action));
}

u64 GPU::CurrentSyncRequestFence() const {
    return impl->CurrentSyncRequestFence();
}

void GPU::WaitForSyncOperation(u64 fence) {
    return impl->WaitForSyncOperation(fence);
}

void GPU::TickWork() {
    impl->TickWork();
}

/// Gets a mutable reference to the Host1x interface
Host1x::Host1x& GPU::Host1x() {
    return impl->system.Host1x();
}

/// Gets an immutable reference to the Host1x interface.
const Host1x::Host1x& GPU::Host1x() const {
    return impl->system.Host1x();
}

Engines::Maxwell3D& GPU::Maxwell3D() {
    return impl->current_channel->payload->maxwell_3d;
}

const Engines::Maxwell3D& GPU::Maxwell3D() const {
    return impl->current_channel->payload->maxwell_3d;
}

Engines::KeplerCompute& GPU::KeplerCompute() {
    return impl->current_channel->payload->kepler_compute;
}

const Engines::KeplerCompute& GPU::KeplerCompute() const {
    return impl->current_channel->payload->kepler_compute;
}

Tegra::DmaPusher& GPU::DmaPusher() {
    return impl->current_channel->payload->dma_pusher;
}

const Tegra::DmaPusher& GPU::DmaPusher() const {
    return impl->current_channel->payload->dma_pusher;
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

const VideoCore::RendererBase& GPU::Renderer() const {
    return *impl->renderer;
}

VideoCore::ShaderNotify& GPU::ShaderNotify() {
    return impl->shader_notify;
}

const VideoCore::ShaderNotify& GPU::ShaderNotify() const {
    return impl->shader_notify;
}

void GPU::RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers,
                           std::vector<Service::Nvidia::NvFence>&& fences) {
    impl->RequestComposite(std::move(layers), std::move(fences));
}

void GPU::WaitForComposite() {
    impl->WaitForComposite();
}

std::vector<u8> GPU::GetAppletCaptureBuffer() {
    return impl->GetAppletCaptureBuffer();
}

u64 GPU::GetTicks() const {
    return impl->GetTicks();
}

bool GPU::IsAsync() const {
    return impl->is_async;
}

bool GPU::HasSuspendedChannels() const {
    return impl->scheduler.HasPendingChannelsAnyThread();
}

bool GPU::UseNvdec() const {
    return impl->use_nvdec;
}

void GPU::RendererFrameEndNotify(u32 generated_frames) {
    impl->RendererFrameEndNotify(generated_frames);
}

void GPU::Start() {
    impl->Start();
}

void GPU::NotifyShutdown() {
    impl->NotifyShutdown();
}

void GPU::WaitForIdle() {
    impl->WaitForIdle();
}

void GPU::ObtainContext() {
    impl->ObtainContext();
}

void GPU::ReleaseContext() {
    impl->ReleaseContext();
}

void GPU::PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
    impl->PushGPUEntries(channel, std::move(entries));
}

VideoCore::RasterizerDownloadArea GPU::OnCPURead(PAddr addr, u64 size) {
    return impl->OnCPURead(addr, size);
}

bool GPU::IsCleanReadTrapPage(DAddr page_addr) const {
    return impl->IsCleanReadTrapPage(page_addr);
}

std::optional<VideoCore::RasterizerDownloadArea> GPU::TryFastCleanRead(DAddr page_begin,
                                                                       u64 page_bytes) {
    return impl->TryFastCleanRead(page_begin, page_bytes);
}

void GPU::FlushRegion(DAddr addr, u64 size) {
    impl->FlushRegion(addr, size);
}

void GPU::RunInOrder(std::function<void()>&& func, bool changes_mappings) {
    impl->RunInOrder(std::move(func), changes_mappings);
}

void GPU::NotifySyncpointWait() {
    impl->NotifySyncpointWait();
}

void GPU::MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
    impl->MapInOrder(gpu_addr, size, std::move(func));
}

bool GPU::IsMapPending(GPUVAddr gpu_addr, u64 size) const {
    return impl->IsMapPending(gpu_addr, size);
}

void GPU::InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
    impl->InvalidateRegion(addr, size, preserve_gpu_writes);
}

bool GPU::OnCPUWrite(DAddr addr, u64 size) {
    return impl->OnCPUWrite(addr, size);
}

void GPU::FlushAndInvalidateRegion(DAddr addr, u64 size) {
    impl->FlushAndInvalidateRegion(addr, size);
}

} // namespace Tegra
