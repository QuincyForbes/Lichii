// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "common/common_types.h"
#include "common/nx2_address_wait.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/split_write_tracker.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release (other paths)
    AcquireTrigger, ///< The same, an Ampere semaphore trigger acquire checked when parsed
    AcquirePoll,    ///< The same, a pending acquire polled (Puller::CanResumeAcquire)
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA)
    MacroDmaWrites, ///< The same, waiting only for the queued work that may write it
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    MacroDmaCheck,  ///< EDEN_NX2_CMDPAGE_DIAG: a macro's memory read repeated after a full drain
    DispatchWait,   ///< EXP10: a macro memory read waits for the dispatches sent since it flushed
    FenceWriters,   ///< EXP12: a fence release waits for the writers a semaphore read can't see
    ListCopies,     ///< Lists on GPU-written pages: the dispatches and copies before the last sync
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",       "fetch drain",     "acquire",    "acquire trigger", "acquire poll",
    "semaphore flush", "fence release",
    "list writers", "list sync",      "list flush", "macro dma read",  "macro dma writes",
    "macro dma flush",
    "gp sync fence", "map",           "front start", "front verify",  "macro dma check",
    "macro dma dispatch", "fence writers", "list copy writers",
};

// EDEN_NX2_SPLIT_STATS=1: a cheap stall report, without the zone timers of EDEN_NX2_GPU_PROFILE.
// Only waits are timed (two clock reads per wait). Every 5 s the back logs "GPU split stalls":
// the front's waits for the back in ms/s per DrainSite (and for ring space), the back's idle
// time waiting for the front in ms/s, and OoT 0xe2c calls forwarded to the back.
namespace SplitStats {
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_SPLIT_STATS");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}
inline std::array<std::atomic<u64>, static_cast<size_t>(DrainSite::Count)> g_front_wait_ns{};
inline std::array<std::atomic<u64>, static_cast<size_t>(DrainSite::Count)> g_front_wait_count{};
inline std::atomic<u64> g_ring_full_ns{};    ///< The front waited for ring space
inline std::atomic<u64> g_back_idle_ns{};    ///< The back waited for the front (empty ring)
inline std::atomic<u64> g_back_idle_count{};
inline std::atomic<u64> g_e2c_forwarded{};   ///< OoT 0xe2c calls whose read ran on the back
inline std::atomic<u64> g_e2c_read_front{};  ///< OoT 0xe2c calls that read on the front
// Running totals, never reset (MaybeReport resets the ones above): nx2:vi2's per-frame
// "critical path" line takes differences of these every 600 presents.
inline std::array<std::atomic<u64>, static_cast<size_t>(DrainSite::Count)> g_total_front_wait_ns{};
inline std::atomic<u64> g_total_back_idle_ns{};
/// Always counted (not only with EDEN_NX2_SPLIT_STATS): the renderer's present blocked the
/// thread that composites (Vulkan GetRenderFrame, Metal nextDrawable), i.e. the GPU back.
inline std::atomic<u64> g_total_present_block_ns{};
inline std::atomic<u64> g_total_present_blocks{}; ///< Presents that blocked over 100 us
inline void AddFrontWait(DrainSite site, u64 ns) {
    const auto index = static_cast<size_t>(site);
    g_front_wait_ns[index].fetch_add(ns, std::memory_order_relaxed);
    g_front_wait_count[index].fetch_add(1, std::memory_order_relaxed);
    g_total_front_wait_ns[index].fetch_add(ns, std::memory_order_relaxed);
}
inline void AddPresentBlock(u64 ns) {
    g_total_present_block_ns.fetch_add(ns, std::memory_order_relaxed);
    if (ns > 100'000) {
        g_total_present_blocks.fetch_add(1, std::memory_order_relaxed);
    }
}
/// Logs and resets the totals when 5 s have passed since the last report (the back thread).
void MaybeReport();
} // namespace SplitStats

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// EDEN_NX2_MACRO_SINK=1 (default off): the state writes the front's 3D macros make
    /// (registers outside execution_mask) go to the back's method sink like the command list's
    /// own state writes, instead of one CallMethod each; the back applies its sink, in order,
    /// before the engine's next executable method.
    [[nodiscard]] static bool MacroSink() {
        return macro_sink;
    }
    /// EDEN_NX2_COMPUTE_SINK=1 (default off): native compute writes that only store their value
    /// (native_plain_methods), from macros or the command list, are dropped when the front's
    /// register copy shows the back holds the value already, else sent batched (SinkMethod).
    [[nodiscard]] static bool ComputeSink() {
        return compute_sink;
    }
    /// EDEN_NX2_RING_PAIRS=1 (default off): the state writes that would go to the back's method
    /// sink (the command list's 3D state writes; the 3D macros' with EDEN_NX2_MACRO_SINK, the
    /// compute ones with EDEN_NX2_COMPUTE_SINK) go as (method, value) pairs in Pairs blocks
    /// (PairWrite), which the back applies in one loop per block (EngineInterface::ApplyPairs),
    /// after its method sink.
    [[nodiscard]] static bool RingPairs() {
        return ring_pairs;
    }

    /// The front's writes that took a batched path (EDEN_NX2_MACRO_SINK, EDEN_NX2_RING_PAIRS),
    /// logged every 10 s.
    enum class Batched : u32 {
        MacroSunk,   ///< 3D macro state writes sent to the method sink (or as pairs)
        ComputeSunk, ///< Compute state writes sent to the method sink (or as pairs)
        ComputeSame, ///< Compute state writes dropped: the register held the value
        Pairs,       ///< Writes sent as pairs (any engine, any source)
        PairBlocks,  ///< Pairs blocks opened
        Count,
    };
    void CountBatched(Batched counter, u64 amount = 1) {
        batched[static_cast<size_t>(counter)] += amount;
    }

    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Marks the calling thread as the GPU thread (the front, or the only one without the split).
    static void MarkGpuThread() {
        on_gpu_thread = true;
    }
    /// Whether the calling thread owns the Vulkan scheduler: the back while it runs, else the
    /// GPU thread. Other threads (CPU cores, services) must not record or submit work.
    [[nodiscard]] static bool OwnsScheduler() {
        return instance.BackRunning() ? on_back_thread : on_gpu_thread;
    }
    /// Back: the ring position of the entry being run.
    [[nodiscard]] u64 BackEntryPosition() const {
        return back_entry_position;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work)
    /// and between entries when another thread asks (RequestIdleHook; `requested` true). Returns
    /// whether it submitted.
    /// `pending`: whether the hook would submit (deferred work waits). With EDEN_NX2_IDLE_GRACE_US
    /// the back waits that long for more work before it submits it.
    using IdleHook = bool (*)(void* argument, bool requested);
    using IdlePending = bool (*)(void* argument);
    void SetIdleHook(IdleHook hook, IdlePending pending, void* argument) {
        idle_hook = hook;
        idle_pending = pending;
        idle_argument = argument;
    }
    /// `busy`: whether the GPU still runs work submitted earlier. With EDEN_NX2_IDLE_HOLD_US the
    /// back then sleeps up to that long on deferred work: more work goes into the same command
    /// buffer, and the GPU has no use for one of its own while it is behind.
    void SetIdleBusy(IdlePending busy) {
        idle_busy = busy;
    }
    /// "Perf submits:": how often the back held deferred work, and how often more work came.
    [[nodiscard]] std::pair<u64, u64> TakeIdleHolds() {
        return {idle_holds.exchange(0, std::memory_order_relaxed),
                idle_holds_fed.exchange(0, std::memory_order_relaxed)};
    }
    /// "Perf submits:": how long the back stayed idle after its idle hook submitted, in
    /// IdleGapBuckets (under 50 us, 250 us, 1 ms, 4 ms, longer). Taken (reset) by the caller.
    static constexpr size_t IdleGapBuckets = 5;
    [[nodiscard]] std::array<u64, IdleGapBuckets> TakeIdleGaps() {
        std::array<u64, IdleGapBuckets> gaps;
        for (size_t i = 0; i < IdleGapBuckets; ++i) {
            gaps[i] = idle_gaps[i].exchange(0, std::memory_order_relaxed);
        }
        return gaps;
    }
    /// Experiment(3): asked on the back whenever it has run everything published, whether a
    /// ReleaseFences there would only invalidate the GPU cache (RasterizerInterface::FencesQuiet).
    using FenceQuietCheck = bool (*)(void* argument, bool strict, u64* epoch);
    void SetFenceQuietCheck(FenceQuietCheck check, void* argument) {
        fence_quiet_check = check;
        fence_quiet_argument = argument;
    }
    enum class FenceQuiet : u32 {
        Quiet,     ///< The back found nothing to release with everything sent executed
        Behind,    ///< Work sent is still queued (or unpublished)
        Busy,      ///< The back found fence work with everything sent executed
        Unchecked, ///< The back has not checked at this position (or its idle hook ran since)
    };
    /// Front: what the back's last check found, if it was made with everything sent executed.
    /// `epoch`: the fence work epoch the back saw (compare with FenceWorkEpoch()).
    [[nodiscard]] FenceQuiet FencesQuietNow(u64& epoch) const {
        if (consumed.load(std::memory_order_acquire) != write_position) {
            return FenceQuiet::Behind;
        }
        if (fence_quiet_position.load(std::memory_order_acquire) == write_position) {
            epoch = fence_quiet_epoch.load(std::memory_order_relaxed);
            return FenceQuiet::Quiet;
        }
        return fence_busy_position.load(std::memory_order_relaxed) == write_position
                   ? FenceQuiet::Busy
                   : FenceQuiet::Unchecked;
    }

    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or before it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        WakeOne(wake);
    }
    /// Any thread, a hint (relaxed loads): whether the front waits for the back now, for a ring
    /// position (WaitFor), a drain or a submission (hook_requested), or asleep in Backoff.
    [[nodiscard]] bool FrontWaitsForBack() const {
        return wait_target.load(std::memory_order_relaxed) != ~u64{0} ||
               hook_requested.load(std::memory_order_relaxed) ||
               front_waiting.load(std::memory_order_relaxed);
    }
    /// Any thread: whether the back has already run its idle hook since its last entry.
    [[nodiscard]] bool IsBackQuiet() const {
        return BackQuiet();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    /// `effect`: what the entry may write (SplitWriteTracker; unknown unless the caller knows).
    using Effect = SplitWriteTracker::Effect;
    using WriteKind = SplitWriteTracker::WriteKind;
    static bool WritePipeEnabled();
    /// Up to 64 consecutive non-executable Maxwell3D registers, without a DMA segment.
    void StateMethodRun(Tegra::Engines::EngineInterface* engine, u32 method, const u32* arguments,
                        u32 count, Effect effect);
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment, Effect effect = Effect::Unknown) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words, effect);
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment,
                     Effect effect = Effect::Unknown);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Effect effect = Effect::Unknown) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1, effect);
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    /// EDEN_NX2_MACRO_SINK: a write that only stores a register's value, to an engine whose
    /// execution_mask has the register (native compute): the back puts it in the engine's method
    /// sink tagged EngineInterface::SplitPlainTag, applied before the engine's next other write.
    void SinkMethod(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Sink | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1, Effect::None);
    }
    /// EDEN_NX2_RING_PAIRS: a write that only stores a register's value (no execution_mask side
    /// effect; native compute: native_plain_methods), appended to the open Pairs block. A block
    /// is open only until anything else is reserved or the ring is published (Reserve, Publish),
    /// so published entries never change and a block never crosses the ring's end.
    void PairWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        const u64 pair = u64{method & 0xffff} | (u64{argument} << 32);
        const size_t index = write_position & RingMask;
        if (pairs_header != nullptr && index != 0 && pairs_count < MaxPairs &&
            write_position + 1 - consumed_cache <= RingWords) {
            ring[index] = pair;
            ++pairs_count;
            *pairs_header = static_cast<u64>(Kind::Pairs) | (u64{pairs_count} << 32);
            CountBatched(Batched::Pairs);
            Commit(1, Effect::None);
            return;
        }
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::Pairs) | (u64{1} << 32);
        entry[1] = pair;
        pairs_header = entry;
        pairs_count = 1;
        CountBatched(Batched::Pairs);
        CountBatched(Batched::PairBlocks);
        Commit(2, Effect::None);
    }
    /// Appends a pre-packed span of (method | (u64{value} << 32)) pairs in one ring commit.
    void PairWriteSpan(Tegra::Engines::EngineInterface* engine, const u64* pairs, u32 count) {
        if (count == 0) {
            return;
        }
        SelectEngine(engine);
        const size_t index = write_position & RingMask;
        if (pairs_header != nullptr && index != 0 && index + count <= RingWords &&
            pairs_count + count <= MaxPairs &&
            write_position + count - consumed_cache <= RingWords) {
            std::memcpy(&ring[index], pairs, size_t{count} * sizeof(u64));
            pairs_count += count;
            *pairs_header = static_cast<u64>(Kind::Pairs) | (u64{pairs_count} << 32);
            CountBatched(Batched::Pairs, count);
            Commit(count, Effect::None);
            return;
        }
        u64* const entry = Reserve(1 + size_t{count});
        entry[0] = static_cast<u64>(Kind::Pairs) | (u64{count} << 32);
        std::memcpy(entry + 1, pairs, size_t{count} * sizeof(u64));
        pairs_header = entry;
        pairs_count = count;
        CountBatched(Batched::Pairs, count);
        CountBatched(Batched::PairBlocks);
        Commit(1 + size_t{count}, Effect::None);
    }
    /// Compact Maxwell3D constant-buffer bind: sets const_buffer.{size,address} and binds to stage.
    void CbBind(Tegra::Engines::EngineInterface* engine, u32 stage, u32 raw_config, u32 size,
                GPUVAddr address, bool track_shadow) {
        SelectEngine(engine);
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::CbBind) | (u64{stage & 0x7} << 4) |
                   (track_shadow ? PacketTrackShadow : 0) | (u64{size & 0xfffff} << 12) |
                   (u64{raw_config} << 32);
        entry[1] = address;
        Commit(2, Effect::None);
    }
    /// Compact Maxwell3D constant-buffer data upload (optionally setting const_buffer.offset first).
    void CbUpload(Tegra::Engines::EngineInterface* engine, u32 first_slot, bool incrementing,
                  bool has_offset, bool track_shadow, u32 offset, const u32* words, u32 count) {
        SelectEngine(engine);
        const size_t entry_words = 1 + (size_t{count} + 1) / 2;
        u64* const entry = Reserve(entry_words);
        const u64 flags = (u64{first_slot & 0xf} << 4) | (incrementing ? CbUploadInc : 0) |
                          (has_offset ? CbUploadHasOffset : 0) |
                          (track_shadow ? CbUploadTrack : 0) | (u64{offset} << 32);
        entry[0] = static_cast<u64>(Kind::CbUpload) | flags | (u64{count} << 16);
        entry[entry_words - 1] = 0;
        std::memcpy(entry + 1, words, size_t{count} * sizeof(u32));
        if (count != 0 && (incrementing ? ((first_slot & 0xf) + count < 16)
                                        : (count < MaxCbUploadWords))) {
            cb_upload_header = entry;
            cb_upload_flags = flags;
            cb_upload_first_slot = static_cast<u8>(first_slot & 0xf);
            cb_upload_count = count;
        }
        Commit(entry_words, Effect::None);
        NoteMethodWrite(engine, CbDataBaseMethod + (first_slot & 0xf));
    }
    /// Single-word CBData write that coalesces consecutive 0x8e4+i words into an open CbUpload.
    void CbUploadWord(Tegra::Engines::EngineInterface* engine, u32 slot, bool has_offset,
                      bool track_shadow, u32 offset, u32 word) {
        const u64 track_flag = track_shadow ? CbUploadTrack : 0;
        if (cb_upload_header != nullptr && engine == front_engine && !has_offset &&
            (cb_upload_flags & CbUploadTrack) == track_flag) {
            const bool can_inc = (cb_upload_flags & CbUploadInc) != 0 && cb_upload_count < 16 &&
                                 cb_upload_first_slot + cb_upload_count == slot;
            const bool can_same_slot =
                slot == cb_upload_first_slot && cb_upload_count < MaxCbUploadWords &&
                (cb_upload_count == 1 || (cb_upload_flags & CbUploadInc) == 0);
            if (can_inc || can_same_slot) {
                if ((cb_upload_count & 1) == 1) {
                    if (!can_inc) {
                        cb_upload_flags &= ~CbUploadInc;
                    }
                    ring[(write_position - 1) & RingMask] |= u64{word} << 32;
                    ++cb_upload_count;
                    *cb_upload_header = static_cast<u64>(Kind::CbUpload) | cb_upload_flags |
                                        (u64{cb_upload_count} << 16);
                    NoteMethodWrite(engine, CbDataBaseMethod + (slot & 0xf));
                    return;
                }
                const size_t index = write_position & RingMask;
                if (index != 0 && write_position + 1 - consumed_cache <= RingWords) {
                    if (!can_inc) {
                        cb_upload_flags &= ~CbUploadInc;
                    }
                    ring[index] = u64{word};
                    ++cb_upload_count;
                    *cb_upload_header = static_cast<u64>(Kind::CbUpload) | cb_upload_flags |
                                        (u64{cb_upload_count} << 16);
                    Commit(1, Effect::None);
                    NoteMethodWrite(engine, CbDataBaseMethod + (slot & 0xf));
                    return;
                }
            }
        }
        SelectEngine(engine);
        u64* const entry = Reserve(2);
        const u64 flags = (u64{slot & 0xf} << 4) | CbUploadInc |
                          (has_offset ? CbUploadHasOffset : 0) | track_flag | (u64{offset} << 32);
        entry[0] = static_cast<u64>(Kind::CbUpload) | flags | (u64{1} << 16);
        entry[1] = u64{word};
        cb_upload_header = entry;
        cb_upload_flags = flags;
        cb_upload_first_slot = static_cast<u8>(slot);
        cb_upload_count = 1;
        Commit(2, Effect::None);
        NoteMethodWrite(engine, CbDataBaseMethod + (slot & 0xf));
    }
    /// Compact Maxwell3D indexed draw packet (covers E04, E52, EA0, and staged 0x586..0x585 draws).
    void DrawIndexed(Tegra::Engines::EngineInterface* engine, u32 begin, u32 first, u32 count,
                     bool has_bases, bool has_driver_cb, bool track_shadow, u32 base_vertex,
                     u32 base_instance, u32 instances, Effect effect) {
        const bool with_instances = instances != 1;
        const size_t entry_words = 2 + (has_bases ? 1 : 0) + (with_instances ? 1 : 0);
        SelectEngine(engine);
        u64* const entry = Reserve(entry_words);
        entry[0] = static_cast<u64>(Kind::DrawIndexed) | (has_bases ? DrawHasBases : 0) |
                   (has_driver_cb ? DrawHasDriverCb : 0) |
                   (track_shadow ? PacketTrackShadow : 0) |
                   (with_instances ? DrawHasInstances : 0) | (u64{begin} << 32);
        entry[1] = u64{first} | (u64{count} << 32);
        size_t pos = 2;
        if (has_bases) {
            entry[pos++] = u64{base_vertex} | (u64{base_instance} << 32);
        }
        if (with_instances) {
            entry[pos++] = u64{instances};
        }
        Commit(entry_words, effect);
    }
    /// Compact Maxwell3D array / mesh draw packet (0x586, 0x35d, 0x35e, 0x585).
    void DrawArrays(Tegra::Engines::EngineInterface* engine, u32 begin, u32 first, u32 count,
                    bool track_shadow, Effect effect) {
        SelectEngine(engine);
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::DrawArrays) | (track_shadow ? PacketTrackShadow : 0) |
                   (u64{begin} << 32);
        entry[1] = u64{first} | (u64{count} << 32);
        Commit(2, effect);
    }
    /// Compact Maxwell3D conditional-render evaluation at `address` (OoT 0xe2c OcclusionHle).
    void RenderEnable(Tegra::Engines::EngineInterface* engine, u32 mode, GPUVAddr address,
                      bool track_shadow) {
        SelectEngine(engine);
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::RenderEnable) | (track_shadow ? PacketTrackShadow : 0) |
                   (u64{mode} << 32);
        entry[1] = address;
        Commit(2, Effect::None);
    }

    /// Nx2Perf draw_begin_coalesce: Method (MacroWrite when `direct`) for an executable write
    /// whose immediate repeat the back can apply from a count (DrawManager::AddInstances). The
    /// write is counted into the last entry (bits 24-30) when that one is the same word, sent by
    /// this function, followed by nothing, still unpublished and below 127 repeats; otherwise it
    /// is sent. With `verify` it is always sent, flagged Repeatable when it would have been
    /// counted. Returns whether it was (or would have been) counted.
    bool RepeatableMethod(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                          bool is_last, bool direct, Effect effect, bool verify) {
        const u64 word = static_cast<u64>(Kind::Method) |
                         (direct ? IsLast | Direct : (is_last ? IsLast : 0)) |
                         (u64{method & 0xffff} << 8) | (u64{argument} << 32);
        // The entry at write_position - 1 is unpublished while published_local is below it.
        const bool repeat =
            repeat_end == write_position && engine == front_engine &&
            published_local < write_position &&
            (ring[(write_position - 1) & RingMask] & ~(Repeatable | RepeatMask)) == word &&
            (ring[(write_position - 1) & RingMask] & RepeatMask) != RepeatMask;
        if (repeat && !verify) {
            ring[(write_position - 1) & RingMask] += RepeatOne;
            writes.Entry(effect, write_position);
            if (eager_publish && back_idle.load(std::memory_order_relaxed)) {
                Publish();
            }
            return true;
        }
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        // Reserve may have wrapped: then the flagged entry would not follow the first one.
        const bool flagged = repeat && write_position == repeat_end;
        entry[0] = word | (flagged ? Repeatable : 0);
        Commit(1, effect);
        repeat_end = write_position;
        return flagged;
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    template <typename Func>
    void Defer(const Func& func, Effect effect = Effect::Unknown) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload, effect);
    }
    void DeferFunction(std::function<void()>&& func, bool changes_mappings = true);
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// EDEN_NX2_PUBLISH_WAITS: before the front yields or sleeps, sends the held-back work and
    /// publishes, so the back doesn't idle on work the front already decoded.
    void PublishBeforeWait() {
        static const bool publish_waits = [] {
            const char* value = std::getenv("EDEN_NX2_PUBLISH_WAITS");
            return value == nullptr || value[0] != '0';
        }();
        if (publish_waits) {
            Publish();
        }
    }
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// What sent compute or copy work that may write any memory (EXP10's log line).
    enum class WriterKind : u32 {
        Upload,  ///< Compute inline uploads the front could not bound
        Release, ///< Compute report semaphores the front could not translate
        Compute, ///< Other compute methods (unknown methods)
        Copy,    ///< Copies the front could not bound
        Other,   ///< Writers no engine front classified
        Count,
    };
    /// Compute or copy work was just sent: what can write GPU-written command lists.
    /// `classified`: the engine's front noted what that work writes (NoteWriterRange,
    /// NoteUnboundedWriter, NoteDispatchWriter); otherwise it may write anything.
    void NoteWriter(bool classified = false) {
        last_writer_position = write_position;
        if (!classified) {
            NoteUnboundedWriter(WriterKind::Other);
        }
        if (eager_publish) {
            Publish(); // the fetch that waits for this entry usually follows within microseconds
        }
    }
    /// Compute or copy work just sent may write any memory.
    void NoteUnboundedWriter(WriterKind kind) {
        last_unbounded_writer_position = write_position;
        ++unbounded_writers[static_cast<size_t>(kind)];
    }
    /// Compute or copy work just sent writes only [address, address + size) of `memory` (a copy's
    /// destination, a release, an inline upload); counts as unbounded when the front can't
    /// translate it as the back will.
    void NoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                         WriterKind kind);
    /// How the front noted a compute report semaphore (NoteReleaseWriter, EXP10's log line).
    enum class ReleaseNote : u32 {
        Ranged,       ///< Its bytes, as one device range
        Unmapped,     ///< Translated when the back runs it, to nothing: writes nothing
        NoWrite,      ///< An operation the back ignores (KeplerCompute::FrontNoteWriter)
        PendingMap,   ///< Unbounded: a queued function may still remap it
        Sparse,       ///< Unbounded: in a lazily backed sparse span
        LateUnmapped, ///< Unbounded: unmapped now, translated only when its fence signals
        Split,        ///< Unbounded: not one device range
        Count,
    };
    /// A compute report semaphore just sent writes [address, address + size). `at_run`: the back
    /// translates it when it runs the method and writes nothing when that fails (the query
    /// cache's payload reports), so an address without a mapping (Ocarina of Time releases to GPU
    /// address 0) needs no range; otherwise as NoteWriterRange.
    void NoteReleaseWriter(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                           bool at_run);
    void CountRelease(ReleaseNote note) {
        ++release_notes[static_cast<size_t>(note)];
    }
    [[nodiscard]] const std::array<u64, static_cast<size_t>(ReleaseNote::Count)>&
    ReleaseNotes() const {
        return release_notes;
    }
    /// The noted ranges so far: a check made now covers those below it (WritersMayWriteSince).
    [[nodiscard]] u64 WriterRangeCount() const {
        writer_range_sealed = writer_range_count;
        return writer_range_count;
    }
    /// Whether compute or copy work sent after position `since` may write device bytes
    /// [begin, begin + size): an unbounded writer, or a noted range that overlaps them.
    /// `checked` (optional): the noted ranges below it were found to miss these bytes before;
    /// set to WriterRangeCount() when nothing does.
    [[nodiscard]] bool WritersMayWriteSince(u64 since, u64 begin, u64 size,
                                            u64* checked = nullptr) const {
        if (last_unbounded_writer_position > since) {
            return true;
        }
        // At most MaxWriterScan ranges are compared (a drained read takes a new snapshot), and
        // more than are remembered count as overlapping.
        const u64 end = begin + size;
        const u64 oldest =
            writer_range_count - (std::min<u64>)(writer_range_count, writer_ranges.size());
        const u64 known = checked != nullptr ? *checked : 0;
        u64 index = writer_range_count;
        bool reached = false;
        for (u64 scanned = 0; index > (std::max)(known, oldest); --index, ++scanned) {
            if (scanned == MaxWriterScan) {
                return true;
            }
            const WriterRange& range = writer_ranges[(index - 1) % writer_ranges.size()];
            if (range.position <= since) {
                reached = true;
                break;
            }
            if (range.begin < end && begin < range.end) {
                return true;
            }
        }
        if (!reached && oldest > known) {
            return true;
        }
        if (checked != nullptr) {
            *checked = WriterRangeCount();
        }
        return false;
    }
    [[nodiscard]] u64 LastUnboundedWriterPosition() const {
        return last_unbounded_writer_position;
    }
    [[nodiscard]] const std::array<u64, static_cast<size_t>(WriterKind::Count)>&
    UnboundedWriters() const {
        return unbounded_writers;
    }

    // EXP10: what native compute dispatches write. The front can't tell (their storage buffers
    // and images come from constant buffers and descriptor pools queued work may still write),
    // so the back records, for each dispatch it runs, the device ranges it marks GPU-written in
    // the buffer cache (BufferCache::MarkWrittenBuffer: the storage and texture buffers' write
    // marks, all a flush or fence download brings to guest memory), its written images' guest
    // ranges, its indirect arguments and its QMD releases. A front read checks those once the
    // back ran every dispatch sent since its snapshot, waiting for that when asked (only that
    // far, not for what was sent after).

    /// Native compute work that may dispatch was just sent (a QMD launch or an inline QMD).
    void NoteDispatchWriter();
    /// Why a dispatch the back ran counts as writing anything.
    enum class DispatchUnbounded : u32 {
        Unconfirmed, ///< The rasterizer did not report its writes (another path or backend)
        Replaced,    ///< Replaced by other work (a network bypass, an upscaler)
        Image,       ///< A written image without a guest range
        Count,
    };
    enum class DispatchCheck : u32 {
        Miss,      ///< Every dispatch sent since ran, and none writes the bytes
        Queued,    ///< One is still queued (without the wait)
        Overlap,   ///< One writes the bytes
        Unbounded, ///< One may write anything (DispatchUnbounded, or sent with EXP10 off)
        ScanLimit, ///< Too many writes recorded since to compare
        Count,
    };
    /// Whether the native compute dispatches sent after position `since` may write device bytes
    /// [begin, begin + size). `checked`: the recorded writes below it were found to miss these
    /// bytes before (DispatchWriteCount() at the snapshot); updated on a miss. With `wait`, waits
    /// for the back to run the newest of those dispatches first (`waited` then set).
    DispatchCheck DispatchesMayWriteSince(u64 since, u64 begin, u64 size, bool wait, u64& checked,
                                          bool& waited);
    /// Whether any bounded dispatch has ever written to device bytes [begin, begin + size).
    [[nodiscard]] bool MayBeDispatchWritten(u64 begin, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const u64 first = begin >> 12;
        const u64 last = (begin + size - 1) >> 12;
        if (last < first || last >= DispatchWrittenPageWords * 64) {
            return true;
        }
        if (!dispatch_written_pages) {
            return false;
        }
        for (u64 page = first; page <= last; ++page) {
            if ((dispatch_written_pages[page / 64].load(std::memory_order_relaxed) >>
                 (page % 64)) &
                1) {
                return true;
            }
        }
        return false;
    }
    /// The writes the back recorded so far.
    [[nodiscard]] u64 DispatchWriteCount() const {
        return dispatch_write_count.load(std::memory_order_acquire);
    }
    /// For EXP10's log line: native compute work sent that may dispatch, the dispatches the back
    /// ran whose writes it recorded, and why the others count as writing anything.
    struct DispatchStats {
        u64 sent;
        u64 bounded;
        std::array<u64, static_cast<size_t>(DispatchUnbounded::Count)> unbounded;
    };
    [[nodiscard]] DispatchStats GetDispatchStats() const;

    /// Back: whether the dispatches being run record their writes (BeginDispatchRecord).
    [[nodiscard]] static bool RecordingDispatch() {
        return recording_dispatch;
    }
    /// Back, around a QMD action that may dispatch and release (KeplerCompute).
    void BeginDispatchRecord() {
        recording_dispatch = on_back_thread && back_dispatch_records;
    }
    void EndDispatchRecord() {
        recording_dispatch = false;
    }
    /// Back, while recording: a dispatch starts; the rasterizer recorded all it writes; it ended
    /// (`ran`: it reached the rasterizer, `replaced`: other work ran instead).
    void StartDispatch() {
        dispatch_confirmed = false;
        dispatch_unbounded = false;
    }
    void ConfirmDispatch() {
        dispatch_confirmed = true;
    }
    void FinishDispatch(bool ran, bool replaced);
    /// Back, while recording: the dispatch writes device bytes [begin, begin + size).
    void RecordDispatchWrite(u64 begin, u64 size) {
        if (size != 0) {
            AppendDispatchWrite(begin, begin + size);
        }
    }
    /// Back, while recording: the dispatch may write anything.
    void RecordDispatchUnbounded(DispatchUnbounded reason);
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// Other work that writes guest memory on the back was just sent: inline uploads, reports,
    /// puller methods (semaphores).
    void NoteMemoryWriter() {
        last_memory_writer_position = write_position;
    }
    /// Waits until the back executed every NoteWriter and NoteMemoryWriter work sent so far.
    bool WaitForMemoryWriters();
    /// memory-writing work whose range the front does not note was just sent (KeplerMemory
    /// and Fermi2D methods, syncpoint operations). Ends every macro DMA read snapshot.
    void NoteUnrangedWriter() {
        last_memory_writer_position = write_position;
        last_unranged_position = write_position;
    }
    /// the ring position of everything sent so far.
    [[nodiscard]] u64 SentPosition() const {
        return write_position;
    }
    /// the last compute/copy and unranged writer positions.
    [[nodiscard]] u64 LastWriterPosition() const {
        return last_writer_position;
    }
    [[nodiscard]] u64 LastUnrangedPosition() const {
        return last_unranged_position;
    }
    [[nodiscard]] u64 LastFunctionPosition() const {
        return last_function_position;
    }
    /// Front: whether every queued mapping-changing function has executed, using consumed_cache
    /// first to avoid an atomic load on every macro call and constant-buffer bind.
    [[nodiscard]] bool FunctionsSettled() {
        if (last_function_position <= consumed_cache) {
            return true;
        }
        consumed_cache = consumed.load(std::memory_order_acquire);
        return last_function_position <= consumed_cache;
    }
    /// whether work sent after position `since` may write device bytes
    /// [begin, begin + size): compute or copies (any range), unranged writers, or a noted write
    /// that overlaps the range. Also true when more writes were noted since than are remembered.
    [[nodiscard]] bool MayBeWrittenSince(u64 since, u64 begin, u64 size) const {
        if (last_writer_position > since || last_unranged_position > since) {
            return true;
        }
        return NotedWriteSince(since, begin, size);
    }
    [[nodiscard]] u64 PendingWriteCount() const noexcept {
        return pending_write_count;
    }
    /// Unconsumed PendingWrite sub-ranges on one 4 KiB page so snapshot.pending_writes
    /// always advances to pending_write_count without pinning at same-page writes.
    struct PagePendingWrites {
        struct Entry {
            u64 position;
            u16 begin;
            u16 end;
        };
        static constexpr size_t MaxEntries = 16;
        std::array<Entry, MaxEntries> entries{};
        u32 count{};

        void Add(u64 position, u16 begin, u16 end) noexcept {
            if (count > 0) {
                Entry& last = entries[count - 1];
                if (last.begin <= end && begin <= last.end) {
                    last.begin = (std::min)(last.begin, begin);
                    last.end = (std::max)(last.end, end);
                    last.position = (std::max)(last.position, position);
                    return;
                }
            }
            if (count < MaxEntries) {
                entries[count++] = Entry{position, begin, end};
            } else {
                Entry& last = entries[MaxEntries - 1];
                last.begin = (std::min)(last.begin, begin);
                last.end = (std::max)(last.end, end);
                last.position = (std::max)(last.position, position);
            }
        }
    };

    /// The noted-write part of MayBeWrittenSince: a NoteMemoryWrite sent after `since` (and not
    /// yet consumed by the back) overlaps [begin, begin + size), or more were noted since than
    /// are remembered.
    [[nodiscard]] bool NotedWriteSince(u64 since, u64 begin, u64 size,
                                       u64* checked_writes = nullptr,
                                       PagePendingWrites* page_writes = nullptr) const {
        const u64 cutoff = (std::max)(since, consumed.load(std::memory_order_acquire));
        const u64 end = begin + size;
        const u64 page_begin = begin & ~u64{0xfff};
        const u64 page_end = (end + 0xfff) & ~u64{0xfff};
        const u16 rel_begin = static_cast<u16>(begin - page_begin);
        const u16 rel_end = static_cast<u16>((std::min)(end, page_end) - page_begin);
        bool overlapped = false;
        if (page_writes != nullptr && page_writes->count != 0) {
            u32 dst = 0;
            for (u32 i = 0; i < page_writes->count; ++i) {
                const auto& entry = page_writes->entries[i];
                if (entry.position <= cutoff) {
                    continue;
                }
                if (entry.begin < rel_end && rel_begin < entry.end) {
                    overlapped = true;
                }
                page_writes->entries[dst++] = entry;
            }
            page_writes->count = dst;
        }
        if (last_memory_writer_position <= cutoff || !pending_writes) {
            if (checked_writes != nullptr) {
                *checked_writes = pending_write_count;
            }
            return false;
        }
        const u64 from = checked_writes != nullptr ? *checked_writes : 0;
        if (pending_write_count <= from) {
            return overlapped;
        }
        const u64 to_scan = pending_write_count - from;
        const u64 limit = (std::min<u64>)(to_scan, PendingWriteSlots);
        u64 next_checked = pending_write_count;
        bool reached = to_scan <= PendingWriteSlots;
        for (u64 scanned = 0; scanned < limit; ++scanned) {
            const u64 index = pending_write_count - 1 - scanned;
            const PendingWrite& write = pending_writes[index % PendingWriteSlots];
            if (write.position <= cutoff) {
                reached = true;
                break;
            }
            if (write.begin < end && begin < write.end) {
                if (page_writes == nullptr) {
                    return true;
                }
                overlapped = true;
            }
            if (write.begin < page_end && page_begin < write.end) {
                if (page_writes != nullptr) {
                    const u16 w_begin =
                        static_cast<u16>((std::max)(write.begin, page_begin) - page_begin);
                    const u16 w_end =
                        static_cast<u16>((std::min)(write.end, page_end) - page_begin);
                    page_writes->Add(write.position, w_begin, w_end);
                } else {
                    next_checked = index;
                }
            }
        }
        if (!reached) {
            return true;
        }
        if (checked_writes != nullptr) {
            *checked_writes = next_checked;
        }
        return overlapped;
    }
    /// Work that writes the device range [begin, begin + size) was just sent (an inline upload, a
    /// report, a semaphore release). size ~0 means an unknown range.
    void NoteMemoryWrite(u64 begin, u64 size, const char* what = "unknown") {
        EnsurePendingWrites();
        PendingWrite& write = pending_writes[pending_write_count++ % PendingWriteSlots];
        write.what = what;
        write.begin = begin;
        write.end = size == ~u64{0} ? ~u64{0} : begin + size;
        write.position = write_position;
        write.words = 0;
    }
    /// As NoteMemoryWrite, for a write whose words the front knows (a fence release payload,
    /// linear inline upload data): reads overlay them instead of waiting.
    void NoteMemoryWriteData(u64 begin, const u32* data, u32 words) {
        if (words != 0) {
            EnsurePendingWrites();
        }
        while (words != 0) {
            const u32 count = (std::min<u32>)(words, PendingWriteWords);
            PendingWrite& write = pending_writes[pending_write_count++ % PendingWriteSlots];
            write.what = "known";
            write.begin = begin;
            write.end = begin + u64{count} * sizeof(u32);
            write.position = write_position;
            write.words = count;
            std::memcpy(write.data.data(), data, count * sizeof(u32));
            begin += u64{count} * sizeof(u32);
            data += count;
            words -= count;
        }
    }
    /// Applies to `out` (the words at device address `begin`, read from guest memory after
    /// `since` = ConsumedNow()) the queued writes the front knows the words of. False when an
    /// overlapping queued write's words are unknown (the caller waits instead).
    bool OverlayQueuedWrites(u64 since, u64 begin, u32* out, u32 words);
    /// Waits until the back executed the noted writes that overlap [begin, begin + size).
    /// Returns false when more writes are queued than are remembered (the caller drains).
    bool WaitForWritesTo(u64 begin, u64 size);
    /// Whether a noted write that overlaps [begin, begin + size) is still queued (or too many
    /// are queued to tell).
    bool HasQueuedWritesTo(u64 begin, u64 size);
    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires drain): the guest's GPU work before it is complete before the
    /// commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();
    /// Waits until the back executed the compute and copy work sent before the last NoteSync:
    /// the writers a command list fetched now is ordered after (zortos293/Lichii#805).
    bool WaitForSyncedWriters(DrainSite site = DrainSite::ListWriters);
    /// Whether the back executed the work sent before `position`.
    [[nodiscard]] bool Ran(u64 position) const {
        return consumed.load(std::memory_order_acquire) >= position;
    }
    /// Deferred work that can write memory was just sent (puller semaphore releases, maps).
    void NoteMemoryWrite() {
        last_memory_write_position = write_position;
        last_memory_write_method = DeferredWrite;
    }
    /// A method that writes memory was just sent; the method is kept for the GPU profile.
    void NoteMethodWrite(const Tegra::Engines::EngineInterface* engine, u32 method) {
        previous_memory_write_position = last_memory_write_position;
        previous_memory_write_method = last_memory_write_method;
        last_memory_write_position = write_position;
        // Engines all of whose methods write (compute, DMA, inline-to-memory, 2D) mark method 0.
        last_memory_write_method = engine->memory_write_mask[0] ? OtherEngineWrite : method;
    }
    /// The method just sent writes only [begin, end): a macro memory read waits for it only where
    /// they overlap. Without this call its write counts for every read. EDEN_NX2_RANGED_WAITS=0
    /// keeps every write counting.
    /// `data` (optional): the words it writes there, in address space `space`, which a macro
    /// memory read of them takes without waiting (ForwardRead).
    void RangeLastWrite(GPUVAddr begin, GPUVAddr end, const u32* data = nullptr, size_t space = 0);
    /// For a macro memory read of `words` words at `begin` in address space `space`: when the
    /// newest work that may write them is one constant buffer upload still queued for the back,
    /// copies its words to `out` and returns true; the read then needs no wait.
    /// EDEN_NX2_FORWARD_READS=0 turns it off.
    bool ForwardRead(size_t space, GPUVAddr begin, u32 words, u32* out);
    /// For the front's macro memory reads (MME DMA) of [begin, end): waits until the back executed
    /// the methods that write memory in command order (memory_write_mask, NoteMemoryWrite; ranged
    /// writes only where they overlap), the compute and copy work and the work before the last
    /// host-level wait, all sent so far.
    bool WaitForMemoryWriters(DrainSite site, GPUVAddr begin, GPUVAddr end);
    /// Whether WaitForMemoryWriters(begin, end) would wait now: work that may write the range is
    /// still queued for the back.
    bool HasQueuedMemoryWriters(GPUVAddr begin, GPUVAddr end);
    /// Diagnostics: the newest queued work that may write [begin, end): 0 none, 1 compute or
    /// copy, 2 work before a host sync, 3 an unranged memory write (`method`), 4 a ranged write.
    u32 QueuedWriterKind(GPUVAddr begin, GPUVAddr end, u32& method);
    /// A semaphore release that writes `payload` at device address `device` (from GPU address
    /// `address`) was just sent (a puller release, a 3D payload report).
    void NoteRelease(u64 device, GPUVAddr address, u32 payload, u64 qmd = 0);
    /// Front: the word at `device` (`address`) after queued releases and known writes, or from
    /// guest memory when nothing unpredicted since `ConsumedNow()` may write it.
    std::optional<u32> CurrentReleaseWord(const Tegra::MemoryManager& memory, u64 device,
                                          GPUVAddr address, u64 qmd = 0,
                                          bool* same_qmd = nullptr);
    /// For a semaphore acquire of `device`: the payload a release still queued for the back writes
    /// there, if nothing sent after it may write the word. The back runs the acquire's channel
    /// after that release, so an acquire it meets needs no drain. EDEN_NX2_ACQUIRE_QUEUED=0 off.
    /// `miss` (when no payload): 1 no release recorded, 2 it already ran, 3 a later write.
    std::optional<u32> QueuedReleasePayload(u64 device, GPUVAddr address, u32* miss = nullptr);
    /// EXP12: the ring position past the queued work that may write the u32 at `device` without
    /// the front knowing its words, so a semaphore read that waits for or overlays the noted
    /// writes (EDEN_NX2_SEM_FAST) can't account for it: compute and copies (their QMD and report
    /// releases among them), unranged writers, a release recorded for the word, and queued
    /// functions (maps). Draws that store to the word mark its page GPU-written once they ran.
    [[nodiscard]] u64 UntrackedWriterPosition(u64 device) const {
        u64 target =
            (std::max)({last_writer_position, last_unranged_position, last_function_position});
        if (const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
            release.device == device) {
            target = (std::max)(target, release.position);
        }
        return (std::min)(target, write_position);
    }
    /// EDEN_NX2_ACQUIRE_PRECHECK=2's queued write check (puller.cpp, EXP34): the payload of the
    /// newest release recorded for the u32 at `device` (NoteRelease) while the back has not run it
    /// (its ring position is past `done`), whatever was sent after it (unlike
    /// QueuedReleasePayload). queued_releases keeps one release per slot: a newer release to
    /// another word in the same slot hides it. (From Fritchii v5.5, 0b213954bd.)
    [[nodiscard]] std::optional<u32> QueuedReleaseTo(u64 device, u64 done) const {
        const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
        if (release.device != device || release.position == 0 || release.position <= done) {
            return std::nullopt;
        }
        return release.payload;
    }
    /// Whether a compute or copy writer range noted after position `since` overlaps device bytes
    /// [begin, begin + size), not counting unbounded writers (WritersMayWriteSince counts them).
    /// More ranges than are remembered or scanned count as overlapping.
    [[nodiscard]] bool NotedWriterRangeSince(u64 since, u64 begin, u64 size) const {
        const u64 end = begin + size;
        const u64 oldest =
            writer_range_count - (std::min<u64>)(writer_range_count, writer_ranges.size());
        u64 scanned = 0;
        for (u64 index = writer_range_count; index > oldest; --index, ++scanned) {
            if (scanned == MaxWriterScan) {
                return true;
            }
            const WriterRange& range = writer_ranges[(index - 1) % writer_ranges.size()];
            if (range.position <= since) {
                return false;
            }
            if (range.begin < end && begin < range.end) {
                return true;
            }
        }
        return oldest != 0;
    }
    /// The entry just sent also writes [address, address + size) of `memory`, noted by device
    /// page (other mappings may alias it, and the caches keep state per page). Returns whether
    /// it was noted by range (not as an unknown write).
    bool NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                   WriteKind kind = WriteKind::Other);
    /// The entry just sent writes the constant buffer data `words` (`count` of them, the first
    /// `stored` land: the rest fall past the bank) at `address`. With MacroReadCbOverlay the
    /// stored words are kept by device address until the back ran the entry
    /// (OverlayConstBufferWords).
    void NoteConstBufferWrite(const Tegra::MemoryManager& memory, GPUVAddr address,
                              const u32* words, u32 count, u32 stored);
    /// Whether every constant buffer word queued after `consumed_position` is kept (none was
    /// dropped).
    [[nodiscard]] bool ConstBufferWordsKept(u64 consumed_position) const {
        return cb_words_lost <= consumed_position;
    }
    /// Copies the kept constant buffer words queued after `consumed_position` over the bytes
    /// `out` holds for the device range [begin, begin + size), oldest first: the newest wins.
    void OverlayConstBufferWords(u64 consumed_position, u64 begin, u64 size, u8* out) const;
    /// The next entry's Effect::Draw was classified with this program epoch (MacroReadWriterSet).
    void DrawEpoch(u64 epoch) {
        writes.DrawEpoch(epoch);
    }
    /// The entry just sent invalidates the device range [begin, begin + size) of the caches
    /// (MacroReadWriterSet: early reads check it against proven programs, as a noted write).
    void NoteDeviceRange(u64 begin, u64 size) {
        if (size != 0) {
            writes.Write(begin, begin + size, write_position);
        }
    }
    /// The entry just sent writes guest memory the front can't tell.
    void NoteUnknownWrite() {
        writes.Entry(Effect::Unknown, write_position);
    }
    /// Waits for the queued functions (GPU memory maps among them) before the front translates
    /// addresses, then drops the pending writes the back has run. Returns the back's position.
    u64 SettleWrites();
    /// Drops the pending writes the back has run. Returns the back's position.
    u64 RetireWrites();
    /// Front: a write that lands when a host query completes or a fence signals (counter
    /// reports, delayed releases), not when the back runs it (Tegra::Nx2AsyncReportsSettled).
    /// Call it after the entry that issues the write was sent.
    void NoteAsyncWrite(u64 begin, u64 size) {
        for (u64 slot = begin >> 4; slot <= (begin + size - 1) >> 4; ++slot) {
            // Positions only grow: a bucket holds the newest of the slots that share it.
            async_writes[AsyncWriteBucket(slot)] = write_position;
        }
        any_async_write = true;
    }
    /// Front: the ring position of the newest async write noted to these bytes, or 0. Slots
    /// share buckets, so this may be newer than the bytes' own last write or report one they
    /// never had: callers then only wait longer or take the drained path.
    [[nodiscard]] u64 HasAsyncWrite(u64 begin, u64 size) const {
        u64 newest = 0;
        if (!any_async_write) {
            return newest;
        }
        for (u64 slot = begin >> 4; slot <= (begin + size - 1) >> 4; ++slot) {
            newest = (std::max)(newest, async_writes[AsyncWriteBucket(slot)]);
        }
        return newest;
    }
    /// Front: the ring position after the last entry sent.
    [[nodiscard]] u64 WritePosition() const {
        return write_position;
    }
    /// The back's position right now.
    [[nodiscard]] u64 ConsumedNow() const {
        return consumed.load(std::memory_order_acquire);
    }
    /// Ring position the back must pass to execute possible writers of [begin, end).
    /// Deferred fence writes can still be pending afterwards.
    [[nodiscard]] u64 ReadTarget(u64 begin, u64 end) const {
        return writes.ReadTarget(begin, end);
    }
    [[nodiscard]] const SplitWriteTracker& Writes() const {
        return writes;
    }
    /// Waits until the back consumed `position` (not what came after). Returns whether it had
    /// not yet.
    bool WaitUntil(u64 position, DrainSite site) {
        return WaitFor((std::min)(position, write_position), site);
    }

    static constexpr u64 PairsTag = 5;
    static constexpr u64 CbBindTag = 7;

private:
    friend struct GpuSplitTestAccess;

    enum class Kind : u64 {
        Wrap,         ///< The rest of the ring is unused: continue at its start
        Engine,       ///< The engine the next methods go to: word 1
        Method,       ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,        ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,         ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
        Pairs,        ///< Bits 32-63 count; then one word per write: bits 0-15 method, 32-63 value
        StateBlock,   ///< Bits 8-23 first method, 32-63 count; then packed u32 arguments
        CbBind,       ///< Bits 4-6 stage, 7 track, 12-31 size, 32-63 config; word 1 address
        CbUpload,     ///< Bits 4-7 slot, 8 inc, 9 off, 10 track, 16-31 count, 32-63 offset; words
        DrawIndexed,  ///< Bits 4 bases, 5 cb, 6 inst, 7 track, 32-63 begin; first|count; [bases]; [inst]
        DrawArrays,   ///< Bit 7 track, 32-63 begin; word 1 first|count
        RenderEnable, ///< Bit 7 track, 32-63 mode; word 1 address
    };
    static_assert(PairsTag == static_cast<u64>(Kind::Pairs) &&
                  CbBindTag == static_cast<u64>(Kind::CbBind));
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)
    static constexpr u64 Sink = 1 << 7;   ///< Method: into the method sink, tagged (SinkMethod)
    static constexpr u64 DrawHasBases = u64{1} << 4;
    static constexpr u64 DrawHasDriverCb = u64{1} << 5;
    static constexpr u64 DrawHasInstances = u64{1} << 6;
    static constexpr u64 PacketTrackShadow = u64{1} << 7;
    static constexpr u64 CbUploadInc = u64{1} << 8;
    static constexpr u64 CbUploadHasOffset = u64{1} << 9;
    static constexpr u64 CbUploadTrack = u64{1} << 10;
    static constexpr u32 CbDataBaseMethod = 0x8e4;

    // Read when the split first starts (MaybeStart), after main() applied nx2_switches.txt: a
    // static initialiser would run before it.
    static inline bool macro_sink{};
    static inline bool compute_sink{};
    static inline bool ring_pairs{};
    static constexpr u32 MaxPairs = 4096;
    static constexpr u32 MaxCbUploadWords = 4096;

    /// Method, EDEN_NX2_PERF_VERIFY with draw_begin_coalesce: the front would have counted this
    /// entry into the one before it (RepeatableMethod). Bit 31: bit 7 is Sink's.
    static constexpr u64 Repeatable = u64{1} << 31;
    /// Method: how many more times the back calls it, right after the first (RepeatableMethod),
    /// in bits 24-30 (up to 127)
    static constexpr u64 RepeatOne = u64{1} << 24;
    static constexpr u64 RepeatMask = u64{0x7f} << 24;
    static_assert(((u64{0xf} | IsLast | HasSegment | Direct) & Repeatable) == 0 &&
                  ((u64{0xffff} << 8) & RepeatMask) == 0);

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 64;
    static constexpr size_t EagerWords = 16;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        pairs_header = nullptr;     // What follows ends the open Pairs block
        cb_upload_header = nullptr; // and any open CbUpload block
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords) {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words, Effect effect) {
        write_position += words;
        writes.Entry(effect, write_position);
        const size_t pending = write_position - published_local;
        // EagerPublish: while the back has nothing to do, publish once EagerWords accumulate
        // instead of reading back_idle / back_waiting on every single ring entry.
        if (pending >= publish_every ||
            (pending >= EagerWords &&
             (back_waiting.load(std::memory_order_relaxed) ||
              consumed.load(std::memory_order_relaxed) == published_local ||
              (eager_publish && back_idle.load(std::memory_order_relaxed))))) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// The idle hook (deferred submits), timed as its own profiler zone.
    void RunIdleHook();
    /// After the back stored `consumed`: wakes a front sleeping in Backoff once past wait_target.
    void NotifyFront(u64 position) {
        if (front_waiting.load(std::memory_order_seq_cst) &&
            position >= wait_target.load(std::memory_order_relaxed)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            WakeOne(front_wake);
        }
    }
    /// The same without a position: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            WakeOne(front_wake);
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);
    /// A Method entry with a repeat count (RepeatableMethod): the call, then the repeats.
    void ExecuteRepeated(u64 head);
    /// EDEN_NX2_PERF_VERIFY with draw_begin_coalesce: checks that a Repeatable entry directly
    /// follows the same word.
    void VerifyRepeat(u64 head);

    Core::System* system{};
    bool active{};
    bool eager_publish{}; ///< Nx2Perf::Opt::EagerPublish, read once at Start
    /// Nx2Perf::Verify() with draw_begin_coalesce, read once at Start before the back starts
    bool verify_repeats{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side, in its own 128-byte block: the back writes back_engine and back_entry_position,
    // so keeping them apart avoids cache-line ping-pong on every ring write.
    alignas(128) u64 write_position{};
    u64 last_writer_position{};
    u64 last_memory_writer_position{};
    u64 last_unranged_position{};
    static constexpr u32 PendingWriteWords = 8;
    static constexpr size_t PendingWriteSlots = 8192;
    struct PendingWrite {
        u64 begin{};
        u64 end{};
        u64 position{};
        u32 words{}; ///< Known words from begin, or 0 (unknown contents)
        const char* what{};
        std::array<u32, PendingWriteWords> data{};
    };
    std::unique_ptr<PendingWrite[]> pending_writes;
    u64 pending_write_count{};
    /// Compute and copy writes with a known device range (NoteWriterRange), newest last.
    struct WriterRange {
        u64 position;
        u64 begin;
        u64 end;
    };
    std::array<WriterRange, 1024> writer_ranges{};
    static constexpr u64 MaxWriterScan = 512;
    u64 writer_range_count{};
    u64 last_unbounded_writer_position{};
    std::array<u64, static_cast<size_t>(WriterKind::Count)> unbounded_writers{};
    std::array<u64, static_cast<size_t>(ReleaseNote::Count)> release_notes{};
    /// NoteWriterRange's translation: noted, or why not (a ReleaseNote past Ranged).
    ReleaseNote TryNoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address,
                                   u64 size);
    /// NoteWriterRange's last range and the function position its translation holds for: the
    /// same range again (an inline upload's data words) only moves the newest range's position,
    /// unless a check may have kept what it found for it (below writer_range_sealed).
    mutable u64 writer_range_sealed{};
    const Tegra::MemoryManager* writer_range_memory{};
    GPUVAddr writer_range_address{};
    u64 writer_range_size{};
    u64 writer_range_function{~u64{0}};
    /// Front, EXP10 dispatches (NoteDispatchWriter): the newest sent, the newest sent while the
    /// back did not record (EXP10 off), whether the back was last told to record, and how many.
    u64 last_dispatch_position{};
    u64 last_unrecorded_dispatch_position{};
    bool dispatch_records{};
    u64 dispatches_sent{};
    /// Dispatch writes the back recorded (begin, end; 0, ~0 for anything) with the position of
    /// the entry that ran them, newest last; written by the back, read by the front.
    struct DispatchWrite {
        std::atomic<u64> position{};
        std::atomic<u64> begin{};
        std::atomic<u64> end{};
    };
    static constexpr u64 DispatchWriteSlots = 8192;
    static constexpr u64 MaxDispatchScan = 2048;
    static constexpr size_t DispatchWrittenPageWords = (size_t{1} << (36 - 12)) / 64;
    std::unique_ptr<DispatchWrite[]> dispatch_writes;
    std::unique_ptr<std::atomic<u64>[]> dispatch_written_pages;
    alignas(128) std::atomic<u64> dispatch_write_count{};
    std::atomic<u64> last_unbounded_dispatch_position{};
    std::atomic<u64> dispatches_bounded{};
    std::array<std::atomic<u64>, static_cast<size_t>(DispatchUnbounded::Count)>
        dispatches_unbounded{};
    void EnsurePendingWrites() {
        if (!pending_writes) [[unlikely]] {
            pending_writes = std::make_unique<PendingWrite[]>(PendingWriteSlots);
        }
    }
    void EnsureDispatchTables() {
        if (!dispatch_writes) [[unlikely]] {
            dispatch_writes = std::make_unique<DispatchWrite[]>(DispatchWriteSlots);
        }
        if (!dispatch_written_pages) [[unlikely]] {
            dispatch_written_pages = std::make_unique<std::atomic<u64>[]>(DispatchWrittenPageWords);
        }
    }
    void AppendDispatchWrite(u64 begin, u64 end);
    alignas(128) u64 last_sync_position{};
    u64 last_memory_write_position{};
    static constexpr u32 DeferredWrite = ~0U;
    static constexpr u32 OtherEngineWrite = ~0U - 1;
    /// The 3D method behind last_memory_write_position, or one of the two values above.
    u32 last_memory_write_method{DeferredWrite};
    u64 previous_memory_write_position{};
    u32 previous_memory_write_method{DeferredWrite};
    /// Writes with a known range (RangeLastWrite) the back may not have run yet, oldest first.
    struct RangedWrite {
        u64 position;
        GPUVAddr begin;
        GPUVAddr end;
        u32 method;
        size_t space;           ///< Address space of `data`; 0 when the words are not known
        std::vector<u32> data;  ///< The words written over [begin, end), when known
    };
    /// Ranged writes kept before the oldest counts for every read again (EDEN_NX2_RANGED_MAX).
    static size_t MaxRangedWrites();
    /// The ranged writes in send order. A free slot keeps the storage of its words for the next
    /// write (a deque allocated per write: its block per element with MSVC, and the words).
    class RangedWriteRing {
    public:
        [[nodiscard]] bool empty() const {
            return count == 0;
        }
        [[nodiscard]] size_t size() const {
            return count;
        }
        [[nodiscard]] RangedWrite& front() {
            return slots[head];
        }
        /// The write sent `age` writes before the newest one.
        [[nodiscard]] RangedWrite& Newest(size_t age = 0) {
            return slots[(head + count - 1 - age) & (slots.size() - 1)];
        }
        [[nodiscard]] const RangedWrite& Newest(size_t age = 0) const {
            return slots[(head + count - 1 - age) & (slots.size() - 1)];
        }
        void pop_front() {
            RangedWrite& write = slots[head];
            if (write.data.capacity() > KeptWords) {
                std::vector<u32>{}.swap(write.data);
            } else {
                write.data.clear();
            }
            head = (head + 1) & (slots.size() - 1);
            --count;
        }
        /// The slot of a new newest write: no words, the other fields are the caller's to set.
        [[nodiscard]] RangedWrite& push_back() {
            if (count == slots.size()) {
                Grow();
            }
            ++count;
            return Newest();
        }
        void clear() {
            while (count != 0) {
                pop_front();
            }
            head = 0;
        }

    private:
        /// Words a free slot keeps allocated.
        static constexpr size_t KeptWords = 1024;

        void Grow() {
            std::vector<RangedWrite> grown((std::max<size_t>)(slots.size() * 2, 16));
            for (size_t i = 0; i < count; ++i) {
                grown[i] = std::move(slots[(head + i) & (slots.size() - 1)]);
            }
            slots = std::move(grown);
            head = 0;
        }

        std::vector<RangedWrite> slots; ///< A power of two of them
        size_t head{};
        size_t count{};
    };
    RangedWriteRing ranged_writes;
    static constexpr size_t RangedPageTableSize = 1024;
    static constexpr u64 RangedPageMask = RangedPageTableSize - 1;
    std::array<u64, RangedPageTableSize> ranged_page_max_pos{};
    u64 ranged_large_max_pos{};
    void NoteRangedWritePages(GPUVAddr begin, GPUVAddr end, u64 position) noexcept {
        if (begin >= end) {
            return;
        }
        const u64 first_page = begin >> 12;
        const u64 last_page = (end - 1) >> 12;
        if (last_page - first_page >= 8) {
            if (position > ranged_large_max_pos) {
                ranged_large_max_pos = position;
            }
            return;
        }
        for (u64 page = first_page; page <= last_page; ++page) {
            const size_t slot = static_cast<size_t>((page ^ (page >> 10)) & RangedPageMask);
            if (position > ranged_page_max_pos[slot]) {
                ranged_page_max_pos[slot] = position;
            }
        }
    }
    struct QueuedRelease {
        u64 position;
        u64 device;
        u64 qmd;
        u32 payload;
    };
    std::array<QueuedRelease, 1024> queued_releases{};
    /// Drops consumed ranged writes; returns the position a memory read of [begin, end) waits
    /// for and the method behind it.
    u64 MemoryWriterTarget(u64 done, GPUVAddr begin, GPUVAddr end, u32& target_method);
    u64 last_function_position{};
    SplitWriteTracker writes;
    /// MacroReadCbOverlay: queued constant buffer words, oldest first, popped by RetireWrites.
    struct ConstBufferWord {
        u64 device;
        u64 position;
        u32 value;
    };
    static constexpr size_t MaxConstBufferWords = 4096;
    std::deque<ConstBufferWord> cb_words;
    u64 cb_words_lost{}; ///< Newest position of a constant buffer word not kept
    /// NoteWrite's last translation: memory manager, GPU page, device page, the function
    /// position it was made at (a later function may remap).
    const Tegra::MemoryManager* note_memory{};
    GPUVAddr note_page{~GPUVAddr{0}};
    u64 note_device{};
    u64 note_function{};
    u64 published_local{};
    u64 consumed_cache{};
    /// RepeatableMethod: the ring position after the last entry it sent
    u64 repeat_end{~u64{0}};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};
    /// The open Pairs block's head (PairWrite), and its count.
    u64* pairs_header{};
    u32 pairs_count{};
    /// The open CbUpload block's head (CbUploadWord), flags, starting slot, and word count.
    u64* cb_upload_header{};
    u64 cb_upload_flags{};
    u32 cb_upload_count{};
    u8 cb_upload_first_slot{};
    std::array<u64, static_cast<size_t>(Batched::Count)> batched{};
    std::chrono::steady_clock::time_point batched_since{};
    u32 publishes{};
    void ReportBatched();

    // Back side, in its own 128-byte block.
    alignas(128) Tegra::Engines::EngineInterface* back_engine{};
    /// The position of the entry being run (dispatch writes record it).
    u64 back_entry_position{};
    /// EXP10: the front asked the back to record dispatch writes (NoteDispatchWriter).
    bool back_dispatch_records{};
    /// The dispatch being run: the rasterizer recorded its writes, it recorded an unbounded one.
    bool dispatch_confirmed{};
    bool dispatch_unbounded{};
    static inline thread_local bool recording_dispatch{};
    /// VerifyRepeat: the last entry's word, the Repeatable entries seen and those that did not
    /// follow the same word, since the last report
    u64 verify_previous{};
    u64 verify_mergeable{};
    u64 verify_mismatches{};
    u64 verify_report_ns{};

    /// Whether the back is done with the Vulkan scheduler until it gets more work: set after its
    /// idle hook ran, cleared before it executes entries.
    [[nodiscard]] bool BackQuiet() const {
        return back_quiet.load(std::memory_order_acquire) ||
               !back_running.load(std::memory_order_acquire);
    }

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    static inline thread_local bool on_gpu_thread{};
    IdleHook idle_hook{};
    IdlePending idle_pending{};
    void* idle_argument{};
    std::array<std::atomic<u64>, IdleGapBuckets> idle_gaps{};
    /// Back: when its idle hook last submitted before it waited for work (0 = it did not).
    u64 idle_submit_ns{};
    void NoteIdleGap();
    static u64 IdleGraceNs();
    IdlePending idle_busy{};
    std::atomic<u64> idle_holds{};
    std::atomic<u64> idle_holds_fed{};
    static u64 IdleHoldNs();
    /// EDEN_NX2_IDLE_HOLD_ANY=1: the hold does not ask whether the GPU is behind (to measure it
    /// on a PC, whose GPU rarely is).
    static bool IdleHoldAny();
    /// EDEN_NX2_SPLIT_ADDRESS_WAIT (default 1, 0 = std::atomic::wait): the two threads sleep on
    /// `wake` and `front_wake` in the OS's address wait (Common::Nx2AddressWait). Apple's
    /// std::atomic::wait polls and yields for up to 64 us before it sleeps.
    static bool AddressWaits();
    static void SleepOn(std::atomic<u32>& word, u32 seen) {
        if (AddressWaits()) {
            Common::Nx2AddressWait::Wait(word, seen);
        } else {
            word.wait(seen, std::memory_order_seq_cst);
        }
    }
    static void WakeOne(std::atomic<u32>& word) {
        if (AddressWaits()) {
            Common::Nx2AddressWait::WakeOne(word);
        } else {
            word.notify_one();
        }
    }
    static void WakeAll(std::atomic<u32>& word) {
        if (AddressWaits()) {
            Common::Nx2AddressWait::WakeAll(word);
        } else {
            word.notify_all();
        }
    }
    FenceQuietCheck fence_quiet_check{};
    void* fence_quiet_argument{};
    /// Back: runs fence_quiet_check at `position` (everything published executed).
    void CheckFencesQuiet(u64 position);
    /// Back: forgets the last check (the idle hook submits, which may change the caches).
    void ForgetFencesQuiet() {
        fence_quiet_position.store(~u64{0}, std::memory_order_relaxed);
        fence_busy_position.store(~u64{0}, std::memory_order_relaxed);
    }
    /// Written by the back before `consumed`: the position of its last check that found nothing
    /// to release (with the epoch it saw), or that found work; ~0 for none.
    alignas(128) std::atomic<u64> fence_quiet_position{~u64{0}};
    std::atomic<u64> fence_busy_position{~u64{0}};
    std::atomic<u64> fence_quiet_epoch{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};
    std::atomic<bool> back_quiet{true};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// Front: newest ring position per bucket of 16-byte slots (AsyncWriteBucket). An unordered
    /// map kept every slot ever reported for the whole session: in Ocarina of Time its inserts
    /// were ~6% of the front's working time (counter reports at ever-new addresses).
    static constexpr size_t AsyncWriteBuckets = size_t{1} << 16;
    [[nodiscard]] static size_t AsyncWriteBucket(u64 slot) {
        return static_cast<size_t>((slot ^ (slot >> 16) ^ (slot >> 32)) & (AsyncWriteBuckets - 1));
    }
    alignas(128) std::vector<u64> async_writes = std::vector<u64>(AsyncWriteBuckets);
    bool any_async_write{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<bool> back_idle{}; ///< The back is spinning or sleeping for work (EagerPublish)
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    std::jthread back_thread;
};

} // namespace VideoCommon
