// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <queue>
#include <source_location>
#include <span>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/polyfill_thread.h"
#include "common/nx2_timeline.h"
#include "common/settings.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/renderer_vulkan/fragment_barrier_state.h"
#include "video_core/renderer_vulkan/pushed_constants.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon {
template <typename Trait>
class QueryCacheBase;
}

namespace Vulkan {

class CommandPool;
class Device;
class Framebuffer;
class GraphicsPipeline;
class StateTracker;
class UploadBatch;

struct QueryCacheParams;

/// The scheduler abstracts command buffer and fence management with an interface that's able to do
/// OpenGL-like operations on Vulkan command buffers.
class Scheduler {
public:
    explicit Scheduler(const Device& device, StateTracker& state_tracker);
    ~Scheduler();

    /// Sends the current execution context to the GPU.
    u64 Flush(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
              std::source_location location = std::source_location::current());

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
                std::source_location location = std::source_location::current());

    /// On the split GPU thread, a fence or a command list copy submits only if the last
    /// submission is older than a window (EDEN_NX2_FENCE_FLUSH_US, default 500 us, 0 submits at
    /// once); otherwise it goes with the next one. The back submits deferred work before it
    /// sleeps and when another thread asks (WaitSubmitted). True when the flush was deferred.
    bool DeferFlush();
    /// Submits deferred work (the split's back, from its idle hook).
    void FlushDeferred();
    /// Threads other than the split's back: waits until tick is submitted, asking the back to
    /// submit deferred work (logged every 2 s while it waits). False only when the back thread
    /// is not running, so nothing will submit it: the caller then submits it itself.
    bool WaitSubmitted(u64 tick);
    /// Whether everything up to tick has been handed to the worker for submission.
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept {
        return tick < master_semaphore->CurrentTick();
    }

    /// Waits for the worker thread to finish executing everything. After this function returns it's
    /// safe to touch worker resources.
    void WaitWorker();

    /// The next submission without a wait semaphore of its own waits for this binary semaphore
    /// (MetalFX work submitted outside the scheduler, see present/metalfx_context.h).
    void WaitOnNextSubmit(VkSemaphore semaphore) noexcept {
        next_wait_semaphore.store(semaphore, std::memory_order_release);
    }

    void MarkFrameEnd(u64 frame);
    /// EDEN_NX2_PASS_TIMES=<seconds>: every that many seconds, one frame's render passes get GPU
    /// timestamps, logged as "Perf passes:" (the passes by size and format that took the most).
    void MarkPassTimesFrame();
    /// A draw was recorded in the open render pass (for "Perf passes:").
    void NotePassDraw() noexcept {
        ++pass_draws;
    }
    void TraceGraphicsPipeline(const std::array<u64, 6>& hashes);
    void TraceComputePipeline(u64 hash);
    /// Worker thread, inside a recorded command: timestamps one draw or dispatch of a sampled
    /// submission. A draw passes its vertex and fragment hashes, a dispatch its compute hash.
    template <typename Func>
    void TraceGpuWork(vk::CommandBuffer cmdbuf, u64 hash, u64 fragment_hash,
                      std::array<u32, 3> size, Func&& work) {
        if (pass_time_sampling && !pass_time_in_pass) [[unlikely]] {
            // A guest dispatch of a frame "Perf passes:" times.
            const u32 record = PassTimeWorkBegin(cmdbuf, hash, size);
            TraceGpuWorkImpl(cmdbuf, hash, fragment_hash, size, work);
            PassTimeWorkEnd(cmdbuf, record);
            return;
        }
        TraceGpuWorkImpl(cmdbuf, hash, fragment_hash, size, work);
    }

    template <typename Func>
    void TraceGpuWorkImpl(vk::CommandBuffer cmdbuf, u64 hash, u64 fragment_hash,
                          std::array<u32, 3> size, Func& work) {
        if (!trace_gpu_recording || !trace_work_pool || trace_work_count == TraceMaxWork) {
            work();
            return;
        }
        const u32 query = trace_work_count * 2;
        trace_work_items[trace_work_count++] = {hash, fragment_hash, size};
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query);
        work();
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query + 1);
    }

    /// Sends currently recorded work to the worker thread.
    void DispatchWork(bool lazy = false);

    /// Requests to begin a renderpass.
    void RequestRenderpass(const Framebuffer* framebuffer,
                           std::source_location location = std::source_location::current());

    /// Defers a full-attachment color clear so it becomes the next render pass.
    bool DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot, const VkClearValue& value);

    /// Defers a full depth/stencil clear so it becomes the next render pass.
    bool DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value);

    /// Realizes any pending deferred clear before its framebuffer can be moved or freed.
    void FlushDeferredClear();

    /// Requests the current execution context to be able to execute operations only allowed outside
    /// of a renderpass.
    void RequestOutsideRenderPassOperationContext(
        std::source_location location = std::source_location::current());

    void DeferFragmentBarrier() {
        fragment_barrier.Defer(state.renderpass != VK_NULL_HANDLE);
    }

    bool HasFragmentBarrier() const {
        return fragment_barrier.Pending();
    }

    void NoteRenderPassStores() {
        render_pass_has_stores = true;
    }

    bool RenderPassHasStores() const {
        return render_pass_has_stores;
    }

    /// A dispatch was recorded: the next render pass must see what it wrote.
    void NoteComputeWrites() {
        compute_writes_pending = true;
    }

    /// Returns true when a render pass is currently active in the scheduler state.
    bool IsRenderPassActive() const {
        return state.renderpass != VK_NULL_HANDLE;
    }

    /// Whether the active render pass is the one RequestRenderpass(framebuffer) would keep.
    bool RenderPassMatches(const Framebuffer* framebuffer) const;

    /// Update the pipeline to the current execution context.
    bool UpdateGraphicsPipeline(GraphicsPipeline* pipeline);

    /// Worker thread only: whether the bound graphics pipeline failed to build, so the draws
    /// recorded after it must be dropped instead of reaching the driver without a pipeline.
    void SetGraphicsPipelineMissing(bool missing) {
        graphics_pipeline_missing = missing;
    }
    [[nodiscard]] bool GraphicsPipelineMissing() const {
        return graphics_pipeline_missing;
    }

    /// Update the rescaling state. Returns true if the state has to be updated.
    bool UpdateRescaling(bool is_rescaling, bool mesh_layout);

    /// Returns true when the descriptor buffer chunk has to be bound into the command buffer.
    bool UpdateDescriptorBufferChunk(u32 descriptor_chunk);

    /// Draws of pipelines with the plain graphics layout (one push constant range, the same for
    /// all of them): returns true when the draw has to push its rescaling words, false when the
    /// last such draw of this render pass pushed the same ones. See PushedConstants for lands.
    template <typename T>
    [[nodiscard]] bool UpdateRescalingWords(const T& words, bool lands) noexcept {
        return state.pushed_rescaling_words.Update(words, lands);
    }

    /// The same for the render area.
    template <typename T>
    [[nodiscard]] bool UpdateRenderArea(const T& words, bool lands) noexcept {
        return state.pushed_render_area.Update(words, lands);
    }

    /// A draw pushed with another layout (mesh pipelines): the next plain draw pushes again.
    void ForgetGraphicsPushConstants() noexcept {
        state.pushed_rescaling_words.Forget();
        state.pushed_render_area.Forget();
    }

    /// Invalidates current command buffer state except for render passes
    void InvalidateState();

    /// Assigns the query cache.
    void SetQueryCache(VideoCommon::QueryCacheBase<QueryCacheParams>& query_cache_) {
        query_cache = &query_cache_;
    }

    // Registers a callback to perform on queue submission.
    void RegisterOnSubmit(std::function<void()>&& func) {
        on_submit = std::move(func);
    }

    /// Consecutive in-order staging uploads share barriers until their destinations overlap,
    /// a source depends on a pending copy, or the batch is full. Call outside a render pass.
    void RecordUpload(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies);

    /// Send work to a separate thread.
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer, vk::CommandBuffer>
    void RecordWithUploadBuffer(T&& command) {
        if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
            VideoCommon::GpuSplit::CheckOwner("Scheduler::Record");
        }
        if (uploads_pending) {
            FlushUploads();
        }
        has_pending_commands = true;
        if (chunk->Record(command)) {
            return;
        }
        DispatchWork(true);
        (void)chunk->Record(command);
    }

    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void Record(T&& c) {
        this->RecordWithUploadBuffer(
            [command = std::move(c)](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
                command(cmdbuf);
            });
    }

    [[nodiscard]] bool HasPendingCommands() const noexcept {
        return has_pending_commands || uploads_pending || state.renderpass != VK_NULL_HANDLE ||
               deferred_clear.framebuffer != nullptr;
    }

    /// Whether the recording thread submitted `tick` within 20 ms.
    [[nodiscard]] bool WaitForSubmission(u64 tick) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{20};
        while (tick >= master_semaphore->CurrentTick()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{100});
        }
        return true;
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore->CurrentTick();
    }

    /// Changes whenever recorded command buffer state may have been reset (InvalidateState, the
    /// end of a render pass), so a binding recorded at the same epoch is still bound.
    [[nodiscard]] u64 RecordingEpoch() const noexcept {
        return recording_epoch;
    }

    /// In recorded commands (worker thread): an index buffer was bound in this command buffer.
    void NoteIndexBufferBound() noexcept {
        index_buffer_bound = true;
    }

    /// In recorded commands (worker thread): false when an indexed draw would have no index
    /// buffer (MoltenVK then hands Metal nil and the driver crashes); logs the first ones.
    [[nodiscard]] bool CheckIndexBufferBound(u64 vertex_hash, u64 fragment_hash);

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master_semaphore->IsFree(tick);
    }

    /// Waits for the given GPU tick, optionally pacing frames. Threads other than the one that
    /// records (the async present thread) pass may_flush = false: flushing from there records
    /// into the current chunk while that thread does (corrupt command buffers, crashes). They
    /// wait for the recording thread to submit the tick instead, and give up after 20 ms (it may
    /// be waiting for them).
    void Wait(u64 tick, double target_fps = 0.0, bool may_flush = true) {
        const bool trace = Common::DkbFrameTrace::Enabled() && target_fps > 0.0;
        const auto trace_start = trace ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        if (tick > 0 && (may_flush || WaitForSubmission(tick))) {
            if (tick >= master_semaphore->CurrentTick()) {
                Flush();
            }
            VideoCommon::BackWaitScope wait_scope{"Scheduler::Wait (host GPU execution)"};
            Common::Nx2Timeline::Span timeline_span{"wait for host GPU"};
            master_semaphore->Wait(tick);
        }
        const auto trace_waited = trace ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        double trace_requested_ms{};
        double trace_sleep_ms{};
        double trace_slept_ms{};
        if (Settings::values.use_speed_limit.GetValue() && target_fps > 0.0) {
            auto now = std::chrono::steady_clock::now();
            if (last_target_fps != target_fps) {
                frame_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / target_fps));
                max_frame_count = static_cast<int>(0.1 * target_fps);
                last_target_fps = target_fps;
                frame_counter = 0;
                start_time = now;
            }
            frame_counter++;
            auto target_time = start_time + frame_interval * frame_counter;
            if (target_time >= now) {
                constexpr auto spin_tail = std::chrono::milliseconds(1);
                auto sleep_time = target_time - now;
                if (trace) {
                    trace_requested_ms =
                        std::chrono::duration<double, std::milli>(sleep_time).count();
                }
                if (sleep_time > spin_tail * 2) {
                    if (trace) {
                        trace_sleep_ms =
                            std::chrono::duration<double, std::milli>(sleep_time - spin_tail).count();
                    }
                    const auto sleep_start = trace ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
                    std::this_thread::sleep_for(sleep_time - spin_tail);
                    if (trace) {
                        trace_slept_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - sleep_start).count();
                    }
                }
                while (std::chrono::steady_clock::now() < target_time) {
                    std::this_thread::yield();
                }
            } else if (frame_counter > max_frame_count) {
                frame_counter = 0;
                start_time = now;
            }
        }
        if (trace) {
            static thread_local Common::DkbFrameTrace::RateLimit rate;
            const auto end = std::chrono::steady_clock::now();
            if (rate.Ready(end)) {
                LOG_INFO(Render_Vulkan,
                         "DKB host_pace target_fps={} enabled={} may_flush={} "
                         "cpu_wait_wall_ms={:.3f} pace_requested_ms={:.3f} "
                         "sleep_requested_ms={:.3f} sleep_actual_wall_ms={:.3f} "
                         "pace_actual_wall_ms={:.3f}",
                         target_fps, Settings::values.use_speed_limit.GetValue(), may_flush,
                         std::chrono::duration<double, std::milli>(trace_waited - trace_start).count(),
                         trace_requested_ms, trace_sleep_ms, trace_slept_ms,
                         std::chrono::duration<double, std::milli>(end - trace_waited).count());
            }
        }
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore& GetMasterSemaphore() const noexcept {
        return *master_semaphore;
    }

    std::mutex submit_mutex;

private:
    std::chrono::steady_clock::time_point last_flush{};
    bool flush_pending{};

    class Command {
    public:
        virtual ~Command() = default;

        virtual void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const = 0;

        Command* GetNext() const {
            return next;
        }

        void SetNext(Command* next_) {
            next = next_;
        }

    private:
        Command* next = nullptr;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        explicit TypedCommand(T&& command_) : command{std::move(command_)} {}
        ~TypedCommand() override = default;

        TypedCommand(TypedCommand&&) = delete;
        TypedCommand& operator=(TypedCommand&&) = delete;

        void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const override {

            command(cmdbuf, upload_cmdbuf);
        }

    private:
        T command;
    };

    class CommandChunk final {
    public:
        void ExecuteAll(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf);

        template <typename T>
        bool Record(T& command) {
            using FuncType = TypedCommand<T>;
            static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");

            command_offset = Common::AlignUp(command_offset, alignof(FuncType));
            if (command_offset > sizeof(data) - sizeof(FuncType)) {
                return false;
            }
            Command* const current_last = last;
            last = new (data.data() + command_offset) FuncType(std::move(command));

            if (current_last) {
                current_last->SetNext(last);
            } else {
                first = last;
            }
            command_offset += sizeof(FuncType);
            return true;
        }

        void MarkSubmit() {
            submit = true;
        }

        bool Empty() const {
            return command_offset == 0;
        }

        bool HasSubmit() const {
            return submit;
        }

    private:
        Command* first = nullptr;
        Command* last = nullptr;

        size_t command_offset = 0;
        bool submit = false;
        alignas(std::max_align_t) std::array<u8, 0x8000> data;
    };

    struct State {
        VkRenderPass renderpass{};
        VkFramebuffer framebuffer{};
        VkExtent2D render_area = {0, 0};
        /// The pass is a deferred clear's variant of the framebuffer's render pass that differs
        /// from it only in load ops, so later draws on that framebuffer can stay in it.
        bool clear_variant = false;
        GraphicsPipeline* graphics_pipeline = nullptr;
        bool is_rescaling = false;
        bool rescaling_defined = false;
        bool mesh_layout = false;
        bool needs_state_enable_refresh = false;
        u32 descriptor_buffer_chunk = 0;
        bool descriptor_buffer_bound = false;
        /// Known only inside one render pass: commands recorded between passes (compute
        /// dispatches, compute passes) push at the same offsets without InvalidateState.
        PushedConstants<24> pushed_rescaling_words;
        PushedConstants<16> pushed_render_area;
    };

    struct DeferredClear {
        const Framebuffer* framebuffer = nullptr;
        u32 color_clear_mask = 0;
        std::array<VkClearValue, 8> color_values{};
        bool depth_stencil = false;
        VkClearValue depth_stencil_value{};
    };

    /// What one render pass loads and stores: its attachments, clears and discards.
    struct PassShape {
        u16 width = 0;
        u16 height = 0;
        u16 layers = 0;
        u8 samples = 0;
        u8 num_colors = 0;
        std::array<u8, 8> color_formats{};
        u8 depth_format = 0; ///< PixelFormat::Invalid's value when there is none
        u8 color_clear_mask = 0;
        u8 color_discard_mask = 0;
        bool depth_clear = false;
        bool depth_discard = false;
        u64 framebuffer = 0; ///< Handle, to tell the targets of a frame apart
    };

    /// Begins a render pass for the given framebuffer, optionally with clear values.
    void BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                             const VkClearValue* clear_values, u32 clear_value_count,
                             u32 color_clear_mask = 0, bool depth_clear = false,
                             u32 color_discard_mask = 0, bool depth_discard = false);

    /// Worker thread: "Perf passes:" sampling (see MarkPassTimesFrame).
    void PassTimestamp(vk::CommandBuffer cmdbuf, bool begin, const PassShape* shape, u32 draws);
    void PassTimesSubmitted(u64 tick);
    void PassTimeLabel(vk::CommandBuffer cmdbuf, const char* file, u32 line);
    void PassTimeCloseLabel(vk::CommandBuffer cmdbuf);
    u32 PassTimeWorkBegin(vk::CommandBuffer cmdbuf, u64 hash, std::array<u32, 3> size);
    void PassTimeWorkEnd(vk::CommandBuffer cmdbuf, u32 record);
    void ReportPassTimes();
    void NotePassEnd(const std::source_location& location);

    /// If a deferred clear is pending.
    void RealizeDeferredClear();

    void WorkerThread(std::stop_token stop_token);

    void AllocateWorkerCommandBuffer();

    void TraceGpuSubmission();
    void ReportWorkTimes();
    void TraceShader(u64 hash, u32 stage);

    u64 SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore);

    void AllocateNewContext();

    void EndPendingOperations();

    void EndRenderPass();

    void AcquireNewChunk();

    const Device& device;
    StateTracker& state_tracker;

    std::unique_ptr<MasterSemaphore> master_semaphore;
    FragmentBarrierState fragment_barrier;
    bool render_pass_has_stores{};
    bool compute_writes_pending{};
    bool graphics_pipeline_missing{};
    std::unique_ptr<CommandPool> command_pool;
    std::unique_ptr<UploadBatch> upload_batch;
    bool uploads_pending{};
    void FlushUploads();

    vk::QueryPool trace_query_pool;
    static constexpr u32 TraceMaxSubmissions = 512;
    struct SubmissionTrace {
        struct ShaderUse {
            u64 hash{};
            u32 stage{};
        };
        u64 tick{};
        u32 draw_attempts{};
        u32 dispatch_attempts{};
        u32 num_shaders{};
        u32 untracked_uses{};
        std::array<ShaderUse, 32> shaders{};
    };
    SubmissionTrace trace_work;
    std::vector<SubmissionTrace> trace_submissions;
    static constexpr u32 TraceMaxWork = 4096;
    struct WorkTrace {
        u64 hash{};
        u64 fragment_hash{};
        /// Dispatch grid, or vertex and instance counts of a draw; zero when indirect.
        std::array<u32, 3> size{};
    };
    vk::QueryPool trace_work_pool;
    std::vector<WorkTrace> trace_work_items;
    u32 trace_work_count{};
    Common::DkbFrameTrace::RateLimit trace_gpu_rate;
    u64 trace_gpu_tick{};
    u64 trace_frame_end{};
    u64 trace_sample_frame{};
    u32 trace_gpu_count{};
    u64 trace_gpu_skipped{};
    u32 trace_timestamp_bits{};
    float trace_timestamp_period{};
    bool trace_gpu_recording{};
    bool trace_gpu_sampling{};
    bool trace_gpu_pending{};
    bool trace_gpu_failed{};

    /// GPU busy time for the "Perf:" line (Common::PerfReport::GpuBusyNs): a timestamp at the
    /// start and end of every submission, read back once its tick is done. Not on MoltenVK,
    /// where timestamps add Metal passes (Apple reads its command buffers' GPU times instead).
    void BeginGpuTime();
    void EndGpuTime(vk::CommandBuffer cmdbuf, u64 tick);
    void CollectGpuTime();
    static constexpr u32 GpuTimeSlots = 256;
    struct GpuTimeSlot {
        u64 tick{};
        u32 query{};
    };
    vk::QueryPool gpu_time_pool;
    std::array<GpuTimeSlot, GpuTimeSlots> gpu_time_slots{};
    u32 gpu_time_head{};
    u32 gpu_time_count{};
    u32 gpu_time_open{~0U};
    u32 gpu_time_bits{};
    double gpu_time_period{};
    u64 gpu_time_last_end{};

    static constexpr u32 PassTimeQueries = 4096;
    struct PassTimeRecord {
        PassShape shape;
        u32 begin_query{};
        u32 end_query{~0U};
        u32 draws{};
        u32 submission{};
        bool dispatch{};
        u64 hash{};               ///< Dispatches: the compute shader
        std::array<u32, 3> groups{};
        const char* site_file = nullptr; ///< Labels: the caller that left the render pass
        u32 site_line{};
    };
    vk::QueryPool pass_time_pool;
    double pass_time_period{};
    u64 pass_time_mask{};
    std::chrono::steady_clock::duration pass_time_interval{};
    // Recording thread:
    u32 pass_draws{}; ///< Draws in the open render pass
    std::chrono::steady_clock::time_point pass_time_last{};
    bool pass_time_host_sampling{}; ///< This frame may be timed: label the work outside passes
    /// A render pass that began on the framebuffer the last one ended on, loading what that stored.
    struct PassReopen {
        const char* file{}; ///< Caller that ended the pass before; null when it gave no place
        u32 line{};
        u16 width{};
        u16 height{};
        u32 count{};
    };
    VkFramebuffer last_pass_framebuffer{}; ///< Framebuffer of the render pass that ended last
    const char* pass_end_file{}; ///< Caller about to end the open pass
    u32 pass_end_line{};
    const char* last_pass_end_file{};
    u32 last_pass_end_line{};
    std::vector<PassReopen> pass_time_reopens; ///< Of the frame being timed
    // Worker thread:
    u32 pass_time_open_label{~0U};
    std::vector<PassTimeRecord> pass_time_records;
    std::vector<u32> pass_time_spans; ///< First query of each timed submission
    u32 pass_time_open_span{~0U};
    u32 pass_time_next_query{};
    u32 pass_time_submissions{};
    u64 pass_time_wait_tick{};
    bool pass_time_sampling{};
    bool pass_time_in_pass{};
    bool pass_time_reset{};
    bool pass_time_closing{};
    bool pass_time_pending{};
    std::vector<PassReopen> pass_time_reopen_report;

    VideoCommon::QueryCacheBase<QueryCacheParams>* query_cache = nullptr;

    vk::CommandBuffer current_cmdbuf;
    vk::CommandBuffer current_upload_cmdbuf;

    DeferredClear deferred_clear;

    std::unique_ptr<CommandChunk> chunk;
    std::function<void()> on_submit;
    std::atomic<VkSemaphore> next_wait_semaphore{};

    State state;
    u64 recording_epoch = 0;
    bool index_buffer_bound = false; ///< Worker thread: in current_cmdbuf
    u64 unbound_index_draws = 0;     ///< Worker thread

    u32 num_renderpass_images = 0;
    std::array<VkImage, 9> renderpass_images{};
    std::array<VkImageSubresourceRange, 9> renderpass_image_ranges{};

    std::queue<std::unique_ptr<CommandChunk>> work_queue;
    std::vector<std::unique_ptr<CommandChunk>> chunk_reserve;
    std::mutex execution_mutex;
    std::mutex reserve_mutex;
    std::mutex queue_mutex;
    std::condition_variable work_cv;
    std::condition_variable wait_cv;
    bool worker_waiting{}; ///< The worker is waiting for work (guarded by queue_mutex)
    bool waiter_waiting{}; ///< WaitWorker is waiting for work_queue to drain (guarded by queue_mutex)
    bool lazy_queued{};    ///< A lazily queued chunk may wait for a sleeping worker (recording thread)
    bool has_pending_commands{};
    std::jthread worker_thread;

    std::chrono::steady_clock::duration frame_interval{};
    std::chrono::steady_clock::time_point start_time{};
    double last_target_fps{};
    u64 max_frame_count{};
    u64 frame_counter{};
};

} // namespace Vulkan
