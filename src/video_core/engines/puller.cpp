// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_qmd_prediction.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

namespace {

constexpr u32 Gf100BindClassMask = 0xffff;
constexpr u32 Gf100BindValidMask =  0x1f0000 | Gf100BindClassMask;

// Reads a semaphore word, flushing a GPU-written copy out of the buffer cache first.
u32 ReadSemaphoreWord(DmaPusher& dma_pusher, GPUVAddr address) {
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
        device && dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                          VideoCommon::CacheType::BufferCache)) {
        // The flush downloads through the Vulkan scheduler, which the split's back owns.
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = dma_pusher.rasterizer, device_addr = *device] {
                rasterizer->FlushRegion(device_addr, sizeof(u32), VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::SemaphoreFlush);
    }
    return dma_pusher.memory_manager.Read<u32>(address);
}

// With the GPU thread split (gpu_split.h) the rasterizer belongs to the back thread: work that
// only feeds it runs there in order; reads of what it wrote first wait for it (DrainBack).
template <typename Func>
void OnBack(const Func& func,
            VideoCommon::GpuSplit::Effect effect = VideoCommon::GpuSplit::Effect::Unknown) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer(func, effect);
        if (effect != VideoCommon::GpuSplit::Effect::None ||
            VideoCore::Nx2Debug::DelayedFences()) {
            split.NoteMemoryWrite();
        }
    } else {
        func();
    }
}

// A semaphore release: it writes the payload (and timestamp) at `address` and nothing else.
template <typename Func>
void OnBackRelease(DmaPusher& dma_pusher, GPUVAddr address, const Func& func) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        func();
        return;
    }
    split.Defer(func, VideoCommon::GpuSplit::Effect::None);
    split.NoteMemoryWrite();
    split.NoteWrite(dma_pusher.memory_manager, address, 16);
    // Only delayed fences or synced counter mode leave a payload release unsettled after the
    // back runs it; otherwise SignalFence / Query writes guest memory inline on the back.
    if (VideoCore::Nx2Debug::DelayedFences() || VideoCore::Nx2Debug::AccuratePart(1)) {
        if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address)) {
            split.NoteAsyncWrite(*device, 16);
        }
    }
}

// EDEN_NX2_ACQUIRE_STATS=1 (diagnostic): classifies Ampere acquire triggers before their drain.
// The front notes each queued release (device address -> ring position past it, payload); an
// acquire is "queued-release" when the newest queued release to its word, still ahead of the
// back, satisfies it, "memory" when the word in memory already does and no release to it is
// queued, and "other" otherwise.
struct Nx2QueuedRelease {
    u64 position{};
    u32 payload{};
};
bool AcquireStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_ACQUIRE_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
std::unordered_map<u64, Nx2QueuedRelease>& QueuedReleases() {
    static std::unordered_map<u64, Nx2QueuedRelease> releases; // front thread only
    return releases;
}
struct Nx2AcquireStats {
    u64 total{}, queued_release{}, queued_release_unmet{}, memory{}, other{}, unmet_after{};
};
Nx2AcquireStats& AcquireStats() {
    static Nx2AcquireStats stats; // front thread only
    return stats;
}

bool AcquireEarlyEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_ACQUIRE_EARLY");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool AcquireEarlyVerify() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_ACQUIRE_EARLY_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

struct Nx2AcquireEarlyStats {
    u64 checks{}, early{}, waited{}, declined_gpu_written{}, declined_proof{}, declined_async{},
        unmet{}, verify_mismatches{};
};
Nx2AcquireEarlyStats& AcquireEarlyStats() {
    static Nx2AcquireEarlyStats stats; // front thread only
    return stats;
}

// Most Ampere acquires wait on words the guest CPU released long before (OoT: ~88% are met in
// memory with no release to them queued), yet each drained the whole back first. Here the front
// waits only for the queued entries that could write the word: unknown-effect entries, noted
// writes to its 16 bytes and draws/dispatches not proven to write no memory (as the early MME DMA
// read does, Maxwell3D::FrontDmaReadEarly). Then a page no GPU work ever wrote cannot have a newer
// copy in the buffer cache, and with no pending query result for the word, memory holds what a
// drained read would see. Returns true when the word meets the acquire; false falls back to the
// drained check (which then also covers an unmet word).
bool AcquireEarly(DmaPusher& dma_pusher, GPUVAddr address, const auto& matches) {
    auto& split = VideoCommon::GpuSplit::Get();
    auto& stats = AcquireEarlyStats();
    ++stats.checks;
    const std::optional<DAddr> device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device || Tegra::Nx2IsGpuWrittenRegion(*device) || Tegra::Nx2IsGpuWritten(*device)) {
        ++stats.declined_gpu_written;
        return false;
    }
    const u64 slot_begin = *device & ~u64{0xf};
    const u64 slot_end = slot_begin + 0x10;
    const u64 consumed = split.SettleWrites();
    const bool writer_set =
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet);
    const u64 target = writer_set ? split.Writes().WriterReadTarget(slot_begin, slot_end)
                                  : split.ReadTarget(slot_begin, slot_end);
    if (target > consumed) {
        ++stats.waited;
    }
    split.WaitUntil(target, VideoCommon::DrainSite::AcquireTrigger);
    const u64 consumed_now = split.ConsumedNow();
    if (writer_set && split.Writes().LastGpu() > consumed_now) {
        const u64 epoch = Tegra::g_nx2_code_epoch.load(std::memory_order_acquire);
        if (!split.Writes().DrawsValid(consumed_now, epoch) ||
            split.Writes().AnyPendingWrite(consumed_now, split.Writes().LastGpu(),
                                           [](u64 begin, u64 end) {
                                               return Tegra::Nx2IsProvenCode(begin, end - begin);
                                           })) {
            ++stats.declined_proof;
            return false;
        }
    }
    if (Tegra::Nx2IsGpuWrittenRegion(*device) || Tegra::Nx2IsGpuWritten(*device)) {
        ++stats.declined_gpu_written;
        return false;
    }
    if (const u64 async_position = split.HasAsyncWrite(slot_begin, slot_end - slot_begin);
        async_position != 0 && (async_position > consumed_now ||
                                !Tegra::Nx2AsyncReportsSettled(slot_begin, slot_end - slot_begin))) {
        ++stats.declined_async;
        return false;
    }
    if (!matches(dma_pusher.memory_manager.Read<u32>(address))) {
        ++stats.unmet;
        return false;
    }
    ++stats.early;
    if ((stats.checks % 50000) == 0) {
        LOG_INFO(HW_GPU,
                 "nx2 acquire early: {} checks, {} decided without a drain ({} after a partial "
                 "wait); declined: {} GPU-written, {} draw proofs, {} async results, {} unmet; "
                 "{} verify mismatches",
                 stats.checks, stats.early, stats.waited, stats.declined_gpu_written,
                 stats.declined_proof, stats.declined_async, stats.unmet,
                 stats.verify_mismatches);
    }
    return true;
}

void DrainBack(VideoCommon::DrainSite site = VideoCommon::DrainSite::Acquire) {
    VideoCommon::GpuSplit::Get().Drain(site);
}

// Switch 2 native syncpoint RO shim: resolves `address` to its syncpoint ID when `address` is a
// 4-byte aligned shim word matching MemoryManager::Read<u32>'s exact bounds (guaranteed to read
// SyncpointManager::GetGuestSyncpointValue(id)).
std::optional<u32> NativeSyncpointShimId(const MemoryManager& memory, GPUVAddr address) {
    constexpr u64 shim_bytes = u64{MemoryManager::NumShimSyncpoints} * sizeof(u32);
    if ((address % sizeof(u32)) != 0 || address > ~GPUVAddr{0} - sizeof(u32) ||
        !memory.OverlapsNativeSyncpointShim(address, sizeof(u32)) ||
        !memory.OverlapsNativeSyncpointShim(address, 1) ||
        !memory.OverlapsNativeSyncpointShim(address + sizeof(u32) - 1, 1)) {
        return std::nullopt;
    }
    struct CachedShim {
        size_t memory_id{};
        GPUVAddr base{};
    };
    static thread_local CachedShim cached{};
    const size_t memory_id = memory.GetID() + 1;
    GPUVAddr shim = cached.memory_id == memory_id ? cached.base : 0;
    if (shim == 0 || !memory.OverlapsNativeSyncpointShim(shim, 1) ||
        memory.OverlapsNativeSyncpointShim(shim - 1, 1)) {
        const GPUVAddr page = address & ~GPUVAddr{0xfff};
        if (page != 0 && memory.OverlapsNativeSyncpointShim(page, 1) &&
            !memory.OverlapsNativeSyncpointShim(page - 1, 1)) {
            shim = page;
        } else {
            GPUVAddr lo = address >= shim_bytes ? address - (shim_bytes - 1) : 1;
            GPUVAddr hi = address;
            while (lo < hi) {
                const GPUVAddr mid = lo + (hi - lo) / 2;
                if (memory.OverlapsNativeSyncpointShim(mid, 1)) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            shim = lo;
        }
        cached = {memory_id, shim};
    }
    if (address < shim || address - shim >= shim_bytes ||
        ((address - shim) % sizeof(u32)) != 0) {
        return std::nullopt;
    }
    return static_cast<u32>((address - shim) / sizeof(u32));
}

// Tracks channels suspended on a native syncpoint shim word so SyncpointManager wakes GPUFront as
// soon as IncrementGuest reaches the target value, without capturing Puller* in the callback.
class SyncpointShimWaiters {
public:
    static constexpr size_t MaxWaiters = 64;

    static SyncpointShimWaiters& Get() {
        static SyncpointShimWaiters waiters;
        return waiters;
    }

    void Register(const Puller* owner, Tegra::Host1x::SyncpointManager& syncpoints_mgr,
                  u32 syncpoint_id, std::optional<u32> expected_value, bool latch_on_reach,
                  u64 target_position) {
        if (owner == nullptr) {
            return;
        }
        u32 prev_syncpoint = 0;
        Tegra::Host1x::SyncpointManager::ActionHandle prev_handle = 0;
        Slot* chosen = nullptr;
        u64 token = 0;
        {
            std::scoped_lock lk{mutex};
            for (Slot& slot : slots) {
                const Puller* const current = slot.owner.load(std::memory_order_relaxed);
                if (current == owner) {
                    prev_syncpoint = slot.syncpoint_id;
                    prev_handle = slot.handle;
                    slot.handle = 0;
                    chosen = &slot;
                    break;
                }
                if (current == nullptr && chosen == nullptr) {
                    chosen = &slot;
                }
            }
            if (chosen != nullptr) {
                const bool was_empty = chosen->owner.load(std::memory_order_relaxed) == nullptr;
                token = ++next_token;
                chosen->syncpoint_id = syncpoint_id;
                chosen->expected_value = expected_value.value_or(0);
                chosen->has_expected = expected_value.has_value();
                chosen->target_position.store(target_position, std::memory_order_relaxed);
                chosen->token.store(token, std::memory_order_relaxed);
                chosen->owner.store(owner, std::memory_order_release);
                if (was_empty) {
                    active_count.fetch_add(1, std::memory_order_release);
                }
            }
        }
        if (prev_handle != 0) {
            syncpoints_mgr.DeregisterGuestAction(prev_syncpoint, prev_handle);
        }
        if (!expected_value.has_value()) {
            return;
        }
        if (chosen == nullptr) {
            syncpoints_mgr.RegisterGuestAction(
                syncpoint_id, *expected_value,
                [] { Tegra::Nx2PendingWake::Get().Notify(Tegra::Nx2PendingWake::ByLatch); });
            return;
        }
        const auto handle = syncpoints_mgr.RegisterGuestAction(
            syncpoint_id, *expected_value, [chosen, token, latch_on_reach] {
                if (latch_on_reach) {
                    chosen->latched_token.store(token, std::memory_order_release);
                }
                Tegra::Nx2PendingWake::Get().Notify(Tegra::Nx2PendingWake::ByLatch);
            });
        if (handle != 0) {
            bool stale = false;
            {
                std::scoped_lock lk{mutex};
                if (chosen->owner.load(std::memory_order_relaxed) == owner &&
                    chosen->token.load(std::memory_order_relaxed) == token) {
                    chosen->handle = handle;
                } else {
                    stale = true;
                }
            }
            if (stale) {
                syncpoints_mgr.DeregisterGuestAction(syncpoint_id, handle);
            }
        }
    }

    void EnsureArmedForMask(const Puller* owner, Tegra::Host1x::SyncpointManager& syncpoints_mgr,
                            u32 syncpoint_id, u32 next_expected) {
        if (owner == nullptr || active_count.load(std::memory_order_acquire) == 0) {
            return;
        }
        u64 target_pos = 0;
        {
            std::scoped_lock lk{mutex};
            for (const Slot& slot : slots) {
                if (slot.owner.load(std::memory_order_relaxed) == owner) {
                    if (slot.has_expected && slot.syncpoint_id == syncpoint_id &&
                        slot.expected_value >= next_expected) {
                        return;
                    }
                    target_pos = slot.target_position.load(std::memory_order_relaxed);
                    break;
                }
            }
        }
        Register(owner, syncpoints_mgr, syncpoint_id, next_expected, false, target_pos);
    }

    [[nodiscard]] bool Latched(const Puller* owner) const {
        if (owner == nullptr || active_count.load(std::memory_order_acquire) == 0) {
            return false;
        }
        for (const Slot& slot : slots) {
            if (slot.owner.load(std::memory_order_acquire) == owner) {
                const u64 tok = slot.token.load(std::memory_order_relaxed);
                return tok != 0 && slot.latched_token.load(std::memory_order_acquire) == tok;
            }
        }
        return false;
    }

    [[nodiscard]] u64 TargetPosition(const Puller* owner) const {
        if (owner == nullptr || active_count.load(std::memory_order_acquire) == 0) {
            return 0;
        }
        for (const Slot& slot : slots) {
            if (slot.owner.load(std::memory_order_acquire) == owner) {
                return slot.target_position.load(std::memory_order_relaxed);
            }
        }
        return 0;
    }

    void Clear(const Puller* owner, Tegra::Host1x::SyncpointManager& syncpoints_mgr) {
        if (owner == nullptr || active_count.load(std::memory_order_acquire) == 0) {
            return;
        }
        u32 syncpoint_id = 0;
        Tegra::Host1x::SyncpointManager::ActionHandle handle = 0;
        {
            std::scoped_lock lk{mutex};
            for (Slot& slot : slots) {
                if (slot.owner.load(std::memory_order_relaxed) == owner) {
                    syncpoint_id = slot.syncpoint_id;
                    handle = slot.handle;
                    slot.handle = 0;
                    slot.has_expected = false;
                    slot.token.store(0, std::memory_order_relaxed);
                    slot.owner.store(nullptr, std::memory_order_release);
                    active_count.fetch_sub(1, std::memory_order_release);
                    break;
                }
            }
        }
        if (handle != 0) {
            syncpoints_mgr.DeregisterGuestAction(syncpoint_id, handle);
        }
    }

    void Forget(const Puller* owner) {
        if (owner == nullptr || active_count.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lk{mutex};
        for (Slot& slot : slots) {
            if (slot.owner.load(std::memory_order_relaxed) == owner) {
                slot.handle = 0;
                slot.has_expected = false;
                slot.token.store(0, std::memory_order_relaxed);
                slot.owner.store(nullptr, std::memory_order_release);
                active_count.fetch_sub(1, std::memory_order_release);
                break;
            }
        }
    }

private:
    struct Slot {
        std::atomic<const Puller*> owner{};
        u32 syncpoint_id{};
        u32 expected_value{};
        bool has_expected{};
        Tegra::Host1x::SyncpointManager::ActionHandle handle{};
        std::atomic<u64> target_position{};
        std::atomic<u64> token{};
        std::atomic<u64> latched_token{};
    };

    mutable std::mutex mutex;
    std::array<Slot, MaxWaiters> slots{};
    std::atomic<size_t> active_count{};
    u64 next_token{};
};

// Per-channel poll cache for suspended PendingAcquire checks on GPUFront. Caches the resolved
// DAddr and host u32* pointer (invalidated by MemoryManager::MapSerial and
// DeviceMemoryManager::MappingGeneration) and the settled-readable state so repeated polls on an
// unchanged semaphore word avoid page-table walks, GpuSplit table scans, and BufferCache mutex
// acquisitions in MustFlushRegion.
class AcquirePollCache {
public:
    static constexpr size_t NumSlots = 64;

    struct Entry {
        std::atomic<const Puller*> owner{};
        GPUVAddr address{};
        u32 value{};
        u32 operation{};
        std::chrono::steady_clock::time_point since{};

        size_t memory_id{};
        u64 map_serial{~u64{0}};
        u64 device_gen{~u64{0}};
        bool address_resolved{};
        bool has_device{};
        DAddr device_addr{};
        const u32* host_word{};

        bool readable_settled{};
        bool was_gpu_written{};
        bool had_unconsumed_draws{};
        u64 checked_write_pos{};
        u64 checked_code_epoch{};
        u32 last_word{};

        bool dirty_throttled{};
        u32 last_dirty_word{};
        std::chrono::steady_clock::time_point last_dirty_check{};
    };

    static AcquirePollCache& Get() {
        static AcquirePollCache cache;
        return cache;
    }

    Entry& Resolve(const Puller* owner, DmaPusher& dma_pusher,
                   const Puller::PendingAcquire& acquire) {
        Entry& entry = slots[SlotIndex(owner, dma_pusher.channel_state.bind_id)];
        auto& memory = dma_pusher.memory_manager;
        auto& dev_mem = dma_pusher.system.Host1x().MemoryManager();
        const size_t mem_id = memory.GetID();
        const u64 map_ser = memory.MapSerial();
        const u64 dev_gen = dev_mem.MappingGeneration();
        if (entry.owner.load(std::memory_order_acquire) != owner || !entry.address_resolved ||
            entry.address != acquire.address || entry.value != acquire.value ||
            entry.operation != acquire.operation || entry.since != acquire.since ||
            entry.memory_id != mem_id || entry.map_serial != map_ser ||
            entry.device_gen != dev_gen) {
            entry.owner.store(nullptr, std::memory_order_release);
            entry.readable_settled = false;
            entry.dirty_throttled = false;
            const auto dev = memory.GpuToCpuAddress(acquire.address);
            const u32* host_word = nullptr;
            if (dev && (acquire.address % sizeof(u32)) == 0 &&
                (acquire.address & 0xfffULL) <= 0x1000ULL - sizeof(u32)) {
                const u8* const ptr = dev_mem.GetPointer<u8>(*dev);
                if (ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % alignof(u32)) == 0) {
                    host_word = reinterpret_cast<const u32*>(ptr);
                }
            }
            entry.address = acquire.address;
            entry.value = acquire.value;
            entry.operation = acquire.operation;
            entry.since = acquire.since;
            entry.memory_id = mem_id;
            entry.map_serial = map_ser;
            entry.device_gen = dev_gen;
            entry.has_device = dev.has_value();
            entry.device_addr = dev.value_or(0);
            entry.host_word = host_word;
            entry.address_resolved = true;
            entry.owner.store(owner, std::memory_order_release);
        } else if (entry.has_device && entry.host_word == nullptr &&
                   (acquire.address % sizeof(u32)) == 0 &&
                   (acquire.address & 0xfffULL) <= 0x1000ULL - sizeof(u32)) {
            const u8* const ptr = dev_mem.GetPointer<u8>(entry.device_addr);
            if (ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % alignof(u32)) == 0) {
                entry.host_word = reinterpret_cast<const u32*>(ptr);
            }
        }
        return entry;
    }

    void Invalidate(const Puller* owner, s32 bind_id) noexcept {
        if (owner == nullptr) {
            return;
        }
        Entry& entry = slots[SlotIndex(owner, bind_id)];
        if (entry.owner.load(std::memory_order_relaxed) == owner) {
            entry.address_resolved = false;
            entry.readable_settled = false;
            entry.dirty_throttled = false;
        }
    }

    void Forget(const Puller* owner) noexcept {
        if (owner == nullptr) {
            return;
        }
        for (Entry& entry : slots) {
            if (entry.owner.load(std::memory_order_relaxed) == owner) {
                entry.address_resolved = false;
                entry.readable_settled = false;
                entry.dirty_throttled = false;
                entry.owner.store(nullptr, std::memory_order_release);
            }
        }
    }

private:
    static size_t SlotIndex(const Puller* owner, s32 bind_id) noexcept {
        if (bind_id >= 0 && static_cast<size_t>(bind_id) < NumSlots) {
            return static_cast<size_t>(bind_id);
        }
        const auto raw = reinterpret_cast<uintptr_t>(owner);
        return static_cast<size_t>((raw >> 6) ^ (raw >> 14)) & (NumSlots - 1);
    }

    std::array<Entry, NumSlots> slots{};
};

inline u32 LoadHostWord(const u32* host_word) noexcept {
    return std::atomic_ref<u32>{*const_cast<u32*>(host_word)}.load(std::memory_order_acquire);
}

inline u32 ReadCachedOrMemoryWord(const MemoryManager& memory, GPUVAddr address,
                                  const u32* host_word) {
    if (host_word != nullptr) {
        return LoadHostWord(host_word);
    }
    return memory.Read<u32>(address);
}

// Shared safety checks for reading a settled semaphore word directly from guest memory without
// draining the split: 4-byte aligned GPU/device addresses, outside the native syncpoint shim,
// no pending GMMU function, queued writer, queued release, in-flight async report, or GPU-written
// page/region.
bool SettledSemaphoreChecks(const MemoryManager& memory, GPUVAddr address,
                            std::optional<DAddr> known_device = std::nullopt) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active() || VideoCore::Nx2Debug::DelayedFences() ||
        (address % sizeof(u32)) != 0 ||
        memory.OverlapsNativeSyncpointShim(address, sizeof(u32))) {
        return false;
    }
    const u64 done = split.ConsumedNow();
    if (split.LastFunctionPosition() > done) {
        return false;
    }
    const auto device = known_device ? known_device : memory.GpuToCpuAddress(address);
    if (!device || (*device % sizeof(u32)) != 0) {
        return false;
    }
    if (u32 method = 0; split.QueuedWriterKind(address, address + sizeof(u32), method) != 0) {
        return false;
    }
    if (split.MayBeWrittenSince(done, *device, sizeof(u32))) {
        return false;
    }
    u32 miss = 0;
    if (split.QueuedReleasePayload(*device, address, &miss).has_value() || miss == 3 ||
        split.QueuedReleaseTo(*device, done).has_value()) {
        return false;
    }
    if (const u64 async_position = split.HasAsyncWrite(*device, sizeof(u32));
        async_position != 0 &&
        (async_position > done || !Nx2AsyncReportsSettled(*device, sizeof(u32)))) {
        return false;
    }
    return !Nx2MayBeGpuWritten(*device, sizeof(u32));
}

bool AddSemaphoreWaiter(DmaPusher& dma_pusher, const void* owner,
                        const Puller::PendingAcquire& acquire,
                        std::optional<DAddr> known_device = std::nullopt) {
    const auto device =
        known_device ? known_device : dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device) {
        return true;
    }
    u32* watched_word =
        reinterpret_cast<u32*>(dma_pusher.system.Host1x().MemoryManager().GetPointer<u8>(*device));
    if (reinterpret_cast<uintptr_t>(watched_word) % alignof(u32) != 0) {
        watched_word = nullptr;
    }
    return Nx2SemaphoreWaiters::Get().Add(owner, *device, acquire.value, acquire.operation,
                                          watched_word);
}

// EDEN_NX2_ACQUIRE_PRECHECK=1: a GEQUAL acquire whose word already holds the value continues
// without draining the split, if nothing queued for the back may write the word (a release that
// lowers it, a fence action) and no GPU work ever wrote its page (then no cache holds a newer word
// than guest memory). It counts as a host sync for later GPU-written command lists, as the drain
// did. Equal and mask acquires always drain. Ported from zortos293/Lichii#166 (e56b37fc0a), with
// the queued-writer check added.
// A release still queued for the split's back meets the acquire: the back runs it before the
// acquire's channel goes on, so the acquire needs no drain (GpuSplit::QueuedReleasePayload).
bool MetByQueuedRelease(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                        u32* miss_out = nullptr,
                        std::optional<DAddr> known_device = std::nullopt,
                        const u32* host_word = nullptr) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active()) {
        return false;
    }
    // Equal and mask acquires must not pass on a release whose host fence has yet to run.
    if (VideoCore::Nx2Debug::DelayedFences() && acquire.operation != 0x4) {
        return false;
    }
    auto& memory = dma_pusher.memory_manager;
    const auto device = known_device ? known_device : memory.GpuToCpuAddress(acquire.address);
    if (!device) {
        return false;
    }
    u32 miss = 0;
    const auto payload = split.QueuedReleasePayload(*device, acquire.address, &miss);
    // The release already ran: guest memory holds its word once settled (nothing sent after it
    // may still change the word, no pending GMMU function or async report, and not GPU-written).
    if (!payload && miss == 2 && SettledSemaphoreChecks(memory, acquire.address, *device) &&
        acquire.Matches(ReadCachedOrMemoryWord(memory, acquire.address, host_word))) {
        split.NoteSync();
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresQueued, 1);
        return true;
    }
    if (miss_out) {
        *miss_out = payload ? 4 : miss;
    }
    if (!payload || !acquire.Matches(*payload)) {
        if (miss_out && VideoCommon::Nx2Profile::Enabled()) {
            using Counter = VideoCommon::Nx2Profile::Counter;
            VideoCommon::Nx2Profile::Add(miss == 1   ? Counter::AcquireNoRelease
                                         : miss == 2 ? Counter::AcquireReleaseRan
                                         : miss == 3 ? Counter::AcquireLaterWrite
                                                     : Counter::AcquirePayloadMiss,
                                         1);
        }
        return false;
    }
    split.NoteSync();
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresQueued, 1);
    return true;
}

// EDEN_NX2_ACQUIRE_PRECHECK=2: also while work is queued and on GPU-written pages. The word only
// grows (GEQUAL semaphores count up), so guest memory, if older than a queued release or a cache,
// holds a lower value: one that meets the acquire already, meets it then too. Only a write that
// lowers the word (a reset) could make this wrong (QueuedWriteRefusesMet). Ocarina drained ~2,200
// acquires a second that were met at once. `readable_word`: what SemaphoreReadableNow read, when
// it read the word current.
bool GequalMetWithoutDrain(DmaPusher& dma_pusher, GPUVAddr address, u32 value,
                           std::optional<u32> readable_word = std::nullopt,
                           std::optional<DAddr> known_device = std::nullopt,
                           const u32* host_word = nullptr);

// EDEN_NX2_EXP34 (live; from Fritchii v5.5, f33124b812 and b2e1d011b5): where level 2 would answer
// a GEQUAL acquire met from guest memory alone, send it the normal way when that may be over a
// write still queued for the back. NVN2 resets its counters (a plain release of a low value) at
// scene and pass boundaries; a queued reset leaves the old, higher word in guest memory and the
// channel runs before the producer the reset preceded. 1 = refuse when the front knows any queued
// write to the word: the newest plain release recorded for it (QueuedReleaseTo: puller WriteLong,
// SEM_EXECUTE, 3D and compute 0x6c3 payload releases) with any payload, a noted queued write over
// it (puller semaphores, 3D reports, uploads), a noted compute or copy writer range, or with
// EXP13 a queued QMD prediction record; also when SemaphoreReadableNow read the word current and
// below the value. 2 = only when that queued release writes less than the word read. 0 = off.
// A strict tightening: a refused answer takes the path level 2 takes for a word below the value
// (drain, or the readable fence release). Blind spots: with EXP13=0 the front can't tell which
// word a queued QMD completion release writes, and queued_releases keeps one release per slot.
// Front thread only, as the GpuSplit tables it reads.
bool QueuedWriteRefusesMet(u64 device, u32 word, u32 value, std::optional<u32> readable_word) {
    const int mode = VideoCore::Nx2Debug::Experiment(34);
    if (mode == 0) {
        return false;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    const u64 done = split.ConsumedNow();
    const std::optional<u32> release = split.QueuedReleaseTo(device, done);
    bool refuse = release && *release < word;
    if (mode == 1 && !refuse) {
        refuse = release.has_value() || (readable_word && *readable_word < value) ||
                 split.HasQueuedWritesTo(device, sizeof(u32)) ||
                 split.NotedWriterRangeSince(done, device, sizeof(u32)) ||
                 (VideoCommon::QmdPrediction::Enabled() &&
                  VideoCommon::QmdPrediction::QueuedPayloads(device, done).count != 0);
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        static u64 asked = 0;
        static u64 refused = 0;
        static auto last_report = std::chrono::steady_clock::now();
        ++asked;
        refused += refuse ? 1 : 0;
        if (const auto now = std::chrono::steady_clock::now();
            now - last_report >= std::chrono::seconds{5}) {
            last_report = now;
            LOG_INFO(HW_GPU, "Acquire precheck (level 2, EXP34={}): {} met, {} sent the normal way",
                     mode, asked - refused, refused);
            asked = refused = 0;
        }
    }
    return refuse;
}

bool GequalMetWithoutDrain(DmaPusher& dma_pusher, GPUVAddr address, u32 value,
                           std::optional<u32> readable_word, std::optional<DAddr> known_device,
                           const u32* host_word) {
    static const int level = [] {
        const char* setting = std::getenv("EDEN_NX2_ACQUIRE_PRECHECK");
        return setting == nullptr ? 2 : setting[0] == '2' ? 2 : setting[0] != '0' ? 1 : 0;
    }();
    auto& split = VideoCommon::GpuSplit::Get();
    if (level == 0 || !split.Active()) {
        return false;
    }
    auto& memory = dma_pusher.memory_manager;
    if (memory.OverlapsNativeSyncpointShim(address, sizeof(u32)) ||
        (level == 1 && split.HasQueuedMemoryWriters(address, address + sizeof(u32)))) {
        return false;
    }
    const auto device = known_device ? known_device : memory.GpuToCpuAddress(address);
    if (!device || (level == 1 && Nx2IsGpuWritten(*device))) {
        return false;
    }
    const u32 word = ReadCachedOrMemoryWord(memory, address, host_word);
    if (word < value ||
        (level == 2 && QueuedWriteRefusesMet(*device, word, value, readable_word))) {
        return false;
    }
    split.NoteSync();
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresFast, 1);
    return true;
}

void ReportLearning(double window);

// EDEN_NX2_ACQUIRE_TRACE=1: per semaphore address, the acquires that drained the split, the time
// the drains took, why the queued-release check missed and how each ended. Logged every 5 s.
class AcquireTrace {
public:
    enum Outcome : u32 { MetAfterDrain, MetAfterFences, Suspended, NumOutcomes };

    static bool Enabled() {
        static const bool enabled = std::getenv("EDEN_NX2_ACQUIRE_TRACE") != nullptr;
        return enabled;
    }

    void Record(GPUVAddr address, u32 op, u32 channel, u32 miss, u32 word_before, u32 value,
                u64 drain_ns, Outcome outcome, u32 writer, u32 writer_method) {
        std::scoped_lock lock{mutex};
        Entry& entry = entries[address];
        ++entry.writers[(std::min)(writer, 4U)];
        if (writer == 3) {
            entry.writer_method = writer_method;
        }
        ++entry.count;
        entry.drain_ns += drain_ns;
        ++entry.miss[(std::min)(miss, 4U)];
        ++entry.outcomes[outcome];
        entry.op = op;
        entry.channel = channel;
        entry.behind += word_before < value ? 1 : 0;
        entry.last_value = value;
        entry.last_word = word_before;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report < std::chrono::seconds{5}) {
            return;
        }
        const double window = std::chrono::duration<double>(now - last_report).count();
        last_report = now;
        std::vector<std::pair<GPUVAddr, Entry>> sorted(entries.begin(), entries.end());
        std::ranges::sort(sorted, [](const auto& a, const auto& b) {
            return a.second.drain_ns > b.second.drain_ns;
        });
        u64 total_ns = 0;
        u32 total = 0;
        for (const auto& [_, e] : sorted) {
            total_ns += e.drain_ns;
            total += e.count;
        }
        LOG_INFO(HW_GPU, "Acquire trace: {:.0f} drained acquires/s, {:.1f} ms/s draining, {} addresses",
                 total / window, static_cast<double>(total_ns) / 1e6 / window, sorted.size());
        ReportLearning(window);
        for (size_t i = 0; i < (std::min)(sorted.size(), size_t{12}); ++i) {
            const auto& [entry_address, e] = sorted[i];
            LOG_INFO(HW_GPU,
                     "  {:#x} op {} ch {}: {:.0f}/s, {:.2f} ms/s, word behind {}x (last {:#x} < {:#x}); "
                     "miss none/ran/later/payload-miss {}/{}/{}/{}; met after drain {}, after "
                     "fences {}, suspended {}; queued writer none/compute/sync/unranged/ranged "
                     "{}/{}/{}/{}/{} (unranged method {:#x})",
                     entry_address, e.op, e.channel, e.count / window,
                     static_cast<double>(e.drain_ns) / 1e6 / window, e.behind, e.last_word,
                     e.last_value, e.miss[1], e.miss[2], e.miss[3], e.miss[4],
                     e.outcomes[MetAfterDrain], e.outcomes[MetAfterFences], e.outcomes[Suspended],
                     e.writers[0], e.writers[1], e.writers[2], e.writers[3], e.writers[4],
                     e.writer_method);
        }
        entries.clear();
    }

    static AcquireTrace& Get() {
        static AcquireTrace trace;
        return trace;
    }

private:
    struct Entry {
        u32 count{};
        u64 drain_ns{};
        std::array<u32, 5> miss{};
        std::array<u32, NumOutcomes> outcomes{};
        std::array<u32, 5> writers{};
        u32 writer_method{};
        u32 op{};
        u32 channel{};
        u32 behind{};
        u32 last_value{};
        u32 last_word{};
    };
    std::mutex mutex;
    std::unordered_map<GPUVAddr, Entry> entries;
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

// EDEN_NX2_ACQUIRE_EQUAL=1: an equal or mask acquire whose word is settled and meets the value
// continues without a drain. NVN2's queue-sync slots get one release per frame number, so the
// word changes again only after the acquiring queue moved on.
bool EqualMetWithoutDrain(DmaPusher& dma_pusher, const Puller::PendingAcquire& acquire,
                          std::optional<DAddr> known_device = std::nullopt,
                          const u32* host_word = nullptr) {
    static const bool enabled = [] {
        const char* setting = std::getenv("EDEN_NX2_ACQUIRE_EQUAL");
        return setting == nullptr || setting[0] != '0';
    }();
    auto& memory = dma_pusher.memory_manager;
    if (!enabled || !SettledSemaphoreChecks(memory, acquire.address, known_device) ||
        !acquire.Matches(ReadCachedOrMemoryWord(memory, acquire.address, host_word))) {
        return false;
    }
    VideoCommon::GpuSplit::Get().NoteSync();
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresFast, 1);
    return true;
}

// EDEN_NX2_ACQUIRE_LEARN=1 (experiment): acquires drained per address. An address whose last 32
// drained acquires were all met at once after the drain (releases of queued work: QMD
// completions) is taken as met without one; the back checks the word when it gets there and
// counts misses (a miss turns this off). One whose last 32 all suspended suspends at once:
// that is always safe, the drain told it nothing.
class AcquireLearning {
public:
    static constexpr u32 Streak = 32;

    static bool Enabled() {
        static const bool enabled = [] {
            const char* setting = std::getenv("EDEN_NX2_ACQUIRE_LEARN");
            return setting != nullptr && setting[0] == '1';
        }();
        return enabled;
    }

    struct Stats {
        u32 met{};
        u32 suspended{};
    };

    Stats& Get(GPUVAddr address) {
        return stats[address];
    }

    void NoteMet(GPUVAddr address) {
        Stats& entry = stats[address];
        entry.met = (std::min)(entry.met + 1, Streak);
        entry.suspended = 0;
    }

    void NoteSuspended(GPUVAddr address) {
        Stats& entry = stats[address];
        entry.suspended = (std::min)(entry.suspended + 1, Streak);
        entry.met = 0;
    }

    static inline std::atomic<u32> misses{};
    static inline std::atomic<u64> speculated{};
    static inline std::atomic<u64> early_suspends{};

    static AcquireLearning& Instance() {
        static AcquireLearning learning;
        return learning;
    }

private:
    std::unordered_map<GPUVAddr, Stats> stats;
};

void ReportLearning(double window) {
    static u64 last_speculated = 0, last_early = 0;
    const u64 speculated = AcquireLearning::speculated.load(std::memory_order_relaxed);
    const u64 early = AcquireLearning::early_suspends.load(std::memory_order_relaxed);
    LOG_INFO(HW_GPU, "Acquire learning: {:.0f} taken as met/s, {:.0f} suspended without a drain/s, "
             "{} misses", (speculated - last_speculated) / window, (early - last_early) / window,
             AcquireLearning::misses.load(std::memory_order_relaxed));
    last_speculated = speculated;
    last_early = early;
}

// EDEN_NX2_EXP3: FenceRelease drains skipped and taken, logged every 5 s.
struct FenceReleaseSkips {
    using FenceQuiet = VideoCommon::GpuSplit::FenceQuiet;
    std::array<u64, 4> taken{}; ///< By GpuSplit::FenceQuiet; Quiet = fence work added since
    u64 skipped{};
    std::chrono::steady_clock::time_point since{};

    void Count(FenceQuiet state, bool skip) {
        if (skip) {
            ++skipped;
        } else {
            ++taken[static_cast<size_t>(state)];
        }
        const auto now = std::chrono::steady_clock::now();
        if (since == std::chrono::steady_clock::time_point{}) {
            since = now;
            return;
        }
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
        LOG_INFO(HW_GPU,
                 "Fence release skip (EXP3={}): {:.0f} drains skipped/s, {:.0f} taken/s: {:.0f} "
                 "back behind, {:.0f} fence work pending, {:.0f} fence work added since, {:.0f} "
                 "unchecked",
                 VideoCore::Nx2Debug::Experiment(3), rate(skipped),
                 rate(taken[0] + taken[1] + taken[2] + taken[3]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Behind)]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Busy)]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Quiet)]),
                 rate(taken[static_cast<size_t>(FenceQuiet::Unchecked)]));
        *this = {};
        since = now;
    }
};

// The fence manager records into the scheduler and downloads the CPU's writes over GPU data:
// the back's state, which a drained back still uses from its idle hook (it submits).
// EDEN_NX2_EXP3 (1, or 2 to allow pending fences without delayed fences): no drain when the
// back, having run everything sent, found the release would only invalidate the GPU cache
// (FenceManager::FencesQuiet) and no fence work was added since; the invalidation still runs
// on the back, in order. Fences already made complete without a release.
void ReleaseFences(DmaPusher& dma_pusher) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (VideoCore::Nx2Debug::Experiment(3) != 0 && split.Active()) {
        static FenceReleaseSkips skips;
        u64 epoch{};
        const auto state = split.FencesQuietNow(epoch);
        const bool skip = state == VideoCommon::GpuSplit::FenceQuiet::Quiet &&
                          dma_pusher.rasterizer->FenceWorkEpoch() == epoch;
        skips.Count(state, skip);
        if (skip) {
            OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); });
            return;
        }
    }
    split.Run([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
              VideoCommon::DrainSite::FenceRelease);
}

// EDEN_NX2_EXP12: the fence releases of acquires whose word the front read current but unmet
// (ReleaseFencesForReadable), logged every 5 s. Front thread only.
struct FenceWriterStats {
    using Clock = std::chrono::steady_clock;

    u64 acquires{};   ///< Readable acquires unmet before their fence release
    u64 delayed{};    ///< With EXP12 on, not applied: fences are delayed (GPU accuracy High)
    u64 waits{};      ///< Waits for the back to run the writers the read can't account for
    u64 wait_ns{};
    u64 met{};        ///< Met once the back ran those writers
    u64 full{};       ///< Full drains (EXP12 0 or 3, 1 when still unmet)
    u64 full_ns{};    ///< Their time (from the target on, when there was one)
    u64 full_met{};   ///< Met after the full drain
    u64 suspended{};  ///< EXP12=2: unmet, suspended without the full drain
    u64 saved_ns{};   ///< EXP12=3: the full drains' time past the target, of acquires met there
    u64 resumed{};    ///< Readable acquires suspended after their fence release, resumed since
    u64 resume_ns{};  ///< Their time from the acquire to the resume
    Clock::time_point since{};

    static FenceWriterStats& Get() {
        static FenceWriterStats stats;
        return stats;
    }

    static u64 Nanoseconds(Clock::time_point begin, Clock::time_point end) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    }

    void Tick() {
        const auto now = Clock::now();
        if (since == Clock::time_point{}) {
            since = now;
            return;
        }
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
        const auto ms = [window](u64 ns) { return static_cast<double>(ns) / 1e6 / window; };
        LOG_INFO(HW_GPU,
                 "Fence writer waits (EXP12={}): {:.0f} readable acquires/s unmet before their "
                 "fence release ({:.0f}/s not applied, delayed fences); writer waits {:.0f}/s "
                 "{:.2f} ms/s, met there {:.0f}/s; full drains {:.0f}/s {:.2f} ms/s, met after "
                 "them {:.0f}/s; suspended without one {:.0f}/s; full drain past the writers of "
                 "acquires met there (EXP12=3) {:.2f} ms/s; resumed {:.0f}/s, {:.0f} us from "
                 "acquire to resume",
                 VideoCore::Nx2Debug::Experiment(12), rate(acquires), rate(delayed), rate(waits),
                 ms(wait_ns), rate(met), rate(full), ms(full_ns), rate(full_met), rate(suspended),
                 ms(saved_ns), rate(resumed),
                 resumed != 0 ? static_cast<double>(resume_ns) / 1e3 / resumed : 0.0);
        *this = {};
        since = now;
    }
};

// EXP13 (nx2_qmd_prediction.h): whether a QMD or copy release the front predicted, still queued,
// would meet an unmet readable acquire. Returns its device and the back's release count, for
// QmdPredictionMetBy once the fences ran.
struct QmdPredictionCheck {
    u64 device;
    u64 sequence;
};
std::optional<QmdPredictionCheck> CheckQmdPrediction(DmaPusher& dma_pusher,
                                                     const Puller::PendingAcquire& acquire) {
    namespace Prediction = VideoCommon::QmdPrediction;
    auto& split = VideoCommon::GpuSplit::Get();
    if (!Prediction::Enabled() || !split.Active()) {
        return std::nullopt;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device) {
        return std::nullopt;
    }
    const auto queued = Prediction::QueuedPayloads(*device, split.ConsumedNow());
    const bool meets = std::any_of(queued.payloads.begin(),
                                   queued.payloads.begin() + queued.count,
                                   [&](u32 payload) { return acquire.Matches(payload); });
    Prediction::NoteAcquire(meets              ? Prediction::Coverage::WouldMeet
                            : queued.count != 0 ? Prediction::Coverage::WrongValue
                            : queued.any        ? Prediction::Coverage::AlreadyRan
                                                : Prediction::Coverage::None);
    return QmdPredictionCheck{*device, Prediction::BackSequence()};
}

void QmdPredictionMetBy(const QmdPredictionCheck& check, const Puller::PendingAcquire& acquire,
                        bool met) {
    namespace Prediction = VideoCommon::QmdPrediction;
    using MetBy = Prediction::MetBy;
    if (!met) {
        Prediction::NoteMetBy(MetBy::Unmet);
        return;
    }
    const auto release = Prediction::BackReleaseSince(check.device, check.sequence);
    if (!release) {
        Prediction::NoteMetBy(MetBy::Other);
    } else if (release->reduction) {
        Prediction::NoteMetBy(MetBy::QmdReduction);
    } else if (!acquire.Matches(release->payload)) {
        Prediction::NoteMetBy(MetBy::Other);
    } else {
        Prediction::NoteMetBy(release->writer == Prediction::Writer::Qmd ? MetBy::QmdRelease
                                                                         : MetBy::DmaRelease);
    }
}

} // Anonymous namespace

static bool SemaphoreReadableNow(DmaPusher& dma_pusher, GPUVAddr address, u32& word,
                                 std::optional<DAddr> known_device = std::nullopt,
                                 const u32* host_word = nullptr,
                                 bool* buffer_dirty_out = nullptr);

// Queues an ordered non-blocking probe on GPUBack for a suspended semaphore acquire so GPUFront
// never stalls draining the split ring: flushes any GPU-written BufferCache copy of the word,
// flushes unsettled reports/fences when needed, and latches Nx2SemaphoreWaiters with the result.
static void QueueAsyncSemaphoreProbe(DmaPusher& dma_pusher, GPUVAddr acquire_addr,
                                     DAddr fallback_device_addr) {
    auto& split = VideoCommon::GpuSplit::Get();
    auto* const memory_ptr = &dma_pusher.memory_manager;
    split.Defer(
        [rasterizer = dma_pusher.rasterizer, memory_ptr, acquire_addr, fallback_device_addr] {
            const DAddr device_addr =
                memory_ptr->GpuToCpuAddress(acquire_addr).value_or(fallback_device_addr);
            if (VideoCore::Nx2Debug::DelayedFences() ||
                !Tegra::Nx2AsyncReportsSettled(device_addr, sizeof(u32))) {
                rasterizer->ReleaseFences();
            } else {
                rasterizer->InvalidateGPUCache();
            }
            if (rasterizer->MustFlushRegion(device_addr, sizeof(u32),
                                            VideoCommon::CacheType::BufferCache)) {
                rasterizer->FlushRegion(device_addr, sizeof(u32),
                                        VideoCommon::CacheType::BufferCache);
            }
            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(device_addr,
                                                      memory_ptr->Read<u32>(acquire_addr));
        },
        VideoCommon::GpuSplit::Effect::None);
    split.Publish();
}

// The fence release of an acquire whose word SemaphoreReadableNow read current but unmet: the
// drain only matters for writes that read can't account for. Returns whether the acquire is met.
// EDEN_NX2_EXP12 (live, 0 = off): the back first runs only the queued work that may write the
// word without the front knowing its words (GpuSplit::UntrackedWriterPosition: compute and
// copies, with their QMD and report releases, unranged writers, a release recorded for the word,
// queued functions), the fence release queued behind everything as before. 1 = met there: go
// on, the fences released on the back in order; still unmet: the full drain as before.
// 2 = unmet there too suspends without the full drain (the channel polls as usual). 3 = the full
// drain always, timed past that point (measures what 1 saves). Not with delayed fences, whose
// releases land when the host GPU passed them, not when the back runs them.
static bool ReleaseFencesForReadable(DmaPusher& dma_pusher, const void* owner,
                                     const Puller::PendingAcquire& acquire,
                                     std::optional<DAddr> known_device = std::nullopt) {
    using Clock = FenceWriterStats::Clock;
    auto& stats = FenceWriterStats::Get();
    const auto met = [&] {
        u32 word{};
        return Nx2SemaphoreWaiters::Get().Latched(owner) ||
               (SemaphoreReadableNow(dma_pusher, acquire.address, word) && acquire.Matches(word));
    };
    ++stats.acquires;
    const int mode = VideoCore::Nx2Debug::Experiment(12);
    const auto device =
        known_device ? known_device : dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    auto& split = VideoCommon::GpuSplit::Get();
    if (mode <= 0 || mode > 3 || !device || !split.Active() ||
        VideoCore::Nx2Debug::DelayedFences()) {
        stats.delayed += mode != 0 ? 1 : 0;
        const auto begin = Clock::now();
        ReleaseFences(dma_pusher);
        const bool result = met();
        ++stats.full;
        stats.full_ns += FenceWriterStats::Nanoseconds(begin, Clock::now());
        stats.full_met += result ? 1 : 0;
        stats.Tick();
        return result;
    }
    if (mode == 2) {
        const DAddr device_addr = *device;
        const GPUVAddr acquire_addr = acquire.address;
        auto* const memory_ptr = &dma_pusher.memory_manager;
        split.Defer(
            [rasterizer = dma_pusher.rasterizer, memory_ptr, acquire_addr, device_addr] {
                if (!Tegra::Nx2AsyncReportsSettled(device_addr, sizeof(u32))) {
                    rasterizer->ReleaseFences();
                } else {
                    rasterizer->InvalidateGPUCache();
                }
                Tegra::Nx2SemaphoreWaiters::Get().OnWrite(device_addr,
                                                          memory_ptr->Read<u32>(acquire_addr));
            },
            VideoCommon::GpuSplit::Effect::None);
        split.NoteSync();
        split.Publish();
        ++stats.suspended;
        stats.Tick();
        return false;
    }
    const u64 target = split.UntrackedWriterPosition(*device);
    split.Defer([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                VideoCommon::GpuSplit::Effect::None);
    const auto begin = Clock::now();
    if (split.WaitUntil(target, VideoCommon::DrainSite::FenceWriters)) {
        ++stats.waits;
    }
    const auto reached = Clock::now();
    stats.wait_ns += FenceWriterStats::Nanoseconds(begin, reached);
    const bool met_there = met();
    stats.met += met_there ? 1 : 0;
    if (mode == 3 || (mode == 1 && !met_there)) {
        split.Drain(VideoCommon::DrainSite::FenceRelease);
        const u64 ns = FenceWriterStats::Nanoseconds(reached, Clock::now());
        const bool result = met();
        ++stats.full;
        stats.full_ns += ns;
        stats.full_met += result ? 1 : 0;
        stats.saved_ns += mode == 3 && met_there ? ns : 0;
        stats.Tick();
        return result;
    }
    // A host-level wait for the commands after it, as the drain was; the fences go out at once.
    split.NoteSync();
    split.Publish();
    stats.suspended += met_there ? 0 : 1;
    stats.Tick();
    return met_there;
}

Puller::~Puller() {
    Nx2SemaphoreWaiters::Get().Remove(this);
    SyncpointShimWaiters::Get().Forget(this);
    AcquirePollCache::Get().Forget(this);
}

bool Puller::PendingAcquire::Matches(u32 current) const {
    switch (static_cast<GpuSemaphoreOperation>(operation)) {
    case GpuSemaphoreOperation::AcquireEqual:
        return current == value;
    case GpuSemaphoreOperation::AcquireGequal:
        return current >= value;
    case GpuSemaphoreOperation::AcquireMask:
        return (current & value) != 0;
    default:
        return true;
    }
}

bool Puller::CanResumeAcquire(DmaPusher& dma_pusher) {
    if (!pending_acquire) {
        return true;
    }
    namespace Profile = VideoCommon::Nx2Profile;
    Profile::Add(Profile::Counter::AcquirePolls, 1);
    const bool met = PollAcquire(dma_pusher);
    if (met) {
        Profile::Add(Profile::Counter::AcquirePollsMet, 1);
    }
    return met;
}

bool Puller::PollAcquire(DmaPusher& dma_pusher) {
    static std::atomic<u32> rescued{};
    if (dma_pusher.memory_manager.OverlapsNativeSyncpointShim(pending_acquire->address,
                                                              sizeof(u32))) {
        if (const auto syncpoint_id =
                NativeSyncpointShimId(dma_pusher.memory_manager, pending_acquire->address)) {
            auto& split = VideoCommon::GpuSplit::Get();
            const auto shim_op = static_cast<GpuSemaphoreOperation>(pending_acquire->operation);
            if (SyncpointShimWaiters::Get().Latched(this)) {
                split.NoteSync();
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                return true;
            }
            const u32 shim_word = dma_pusher.memory_manager.Read<u32>(pending_acquire->address);
            const bool target_settled =
                !split.Active() ||
                split.ConsumedNow() >= SyncpointShimWaiters::Get().TargetPosition(this);
            const bool shim_met = shim_op == GpuSemaphoreOperation::AcquireGequal
                                      ? pending_acquire->Matches(shim_word)
                                      : (target_settled && pending_acquire->Matches(shim_word));
            if (shim_met) {
                split.NoteSync();
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                return true;
            }
            ++pending_acquire->unmet_polls;
            if (shim_op == GpuSemaphoreOperation::AcquireMask &&
                !pending_acquire->Matches(shim_word) && pending_acquire->value != 0 &&
                shim_word != ~0U) {
                SyncpointShimWaiters::Get().EnsureArmedForMask(
                    this, dma_pusher.system.Host1x().GetSyncpointManager(), *syncpoint_id,
                    shim_word + 1);
            }
            const bool shim_delayed = VideoCore::Nx2Debug::DelayedFences();
            const auto now = std::chrono::steady_clock::now();
            const bool timeout_due = !pending_acquire->reported &&
                                     now - pending_acquire->since > std::chrono::seconds{2};
            if (split.Active() && !timeout_due) {
                split.Publish();
                if (!split.IsBackQuiet() &&
                    now - pending_acquire->last_drain >= std::chrono::microseconds{250}) {
                    pending_acquire->last_drain = now;
                    split.RequestIdleHook();
                }
                if (shim_delayed &&
                    now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
                    pending_acquire->last_fences = now;
                    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                           VideoCommon::GpuSplit::Effect::None);
                    split.Publish();
                    split.RequestIdleHook();
                }
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                return false;
            }
            if (!split.Active() && !timeout_due) {
                if (shim_delayed &&
                    now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
                    pending_acquire->last_fences = now;
                    ReleaseFences(dma_pusher);
                    if (SyncpointShimWaiters::Get().Latched(this) ||
                        pending_acquire->Matches(
                            dma_pusher.memory_manager.Read<u32>(pending_acquire->address))) {
                        return true;
                    }
                }
                return false;
            }
        }
    }
    // Fast path (EDEN_NX2_SEM_FAST), as for the first check in ProcessSemaphoreTriggerMethod:
    // when the word is current without draining the back, answer from it. Ocarina of Time polls
    // each waiting acquire about 12 times before it is met, and every poll drained the back
    // (about 380 ms/s of the front's time). The drained path below runs now and then for an
    // unmet word, and reports an acquire not met for 2 s.
    auto& split = VideoCommon::GpuSplit::Get();
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        // The back ran a release that met the value and latched this waiter: resume at once.
        split.NoteSync();
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
        return true;
    }
    const bool delayed = Settings::IsGPUFenceBehaviorDefault()
                             ? Settings::IsGPULevelHigh()
                             : Settings::IsGPUFenceBehaviorBalanced() ||
                                   Settings::IsGPUFenceBehaviorAccurate();
    auto& cache_entry = AcquirePollCache::Get().Resolve(this, dma_pusher, *pending_acquire);
    const std::optional<DAddr> known_device =
        cache_entry.has_device ? std::optional<DAddr>{cache_entry.device_addr} : std::nullopt;
    const int sem_fast =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_sem_fast, "EDEN_NX2_SEM_FAST");
    if (cache_entry.readable_settled && cache_entry.host_word != nullptr && split.Active() &&
        sem_fast != 0 && split.WritePosition() == cache_entry.checked_write_pos &&
        !dma_pusher.system.GPU().IsMapPending(pending_acquire->address, sizeof(u32)) &&
        Tegra::Nx2IsGpuWritten(cache_entry.device_addr) == cache_entry.was_gpu_written &&
        (!cache_entry.had_unconsumed_draws ||
         split.ConsumedNow() >= cache_entry.checked_write_pos ||
         Tegra::g_nx2_code_epoch.load(std::memory_order_acquire) ==
             cache_entry.checked_code_epoch)) {
        const u32 word = LoadHostWord(cache_entry.host_word);
        if (pending_acquire->Matches(word)) {
            split.NoteSync();
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
            return true;
        }
        cache_entry.last_word = word;
        ++pending_acquire->unmet_polls;
        const auto now = std::chrono::steady_clock::now();
        const bool timeout_due =
            !pending_acquire->reported && now - pending_acquire->since > std::chrono::seconds{2};
        if (!timeout_due) {
            split.Publish();
            if (delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
                pending_acquire->last_fences = now;
                OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                       VideoCommon::GpuSplit::Effect::None);
                split.Publish();
                split.RequestIdleHook();
            }
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
            return false;
        }
    } else {
        cache_entry.readable_settled = false;
    }
    bool skip_dirty_readable_check = false;
    if (cache_entry.dirty_throttled && cache_entry.host_word != nullptr && split.Active()) {
        const u32 cur = LoadHostWord(cache_entry.host_word);
        const auto now = std::chrono::steady_clock::now();
        if (cur == cache_entry.last_dirty_word && !pending_acquire->Matches(cur) &&
            now - cache_entry.last_dirty_check < std::chrono::microseconds{250}) {
            skip_dirty_readable_check = true;
        } else {
            cache_entry.dirty_throttled = false;
        }
    }
    std::optional<u32> readable_word;
    bool buffer_dirty_checked = false;
    bool buffer_dirty = false;
    if (!skip_dirty_readable_check) {
        if (u32 word{}; SemaphoreReadableNow(dma_pusher, pending_acquire->address, word,
                                             known_device, cache_entry.host_word, &buffer_dirty)) {
            if (pending_acquire->Matches(word)) {
                split.NoteSync();
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                return true;
            }
            readable_word = word;
            ++pending_acquire->unmet_polls;
            if (cache_entry.has_device && cache_entry.host_word != nullptr && split.Active()) {
                const u64 done = split.ConsumedNow();
                if (!split.QueuedReleaseTo(cache_entry.device_addr, done).has_value() &&
                    !split.HasQueuedWritesTo(cache_entry.device_addr, 16)) {
                    const bool gpu_written = Tegra::Nx2IsGpuWritten(cache_entry.device_addr);
                    cache_entry.readable_settled = true;
                    cache_entry.was_gpu_written = gpu_written;
                    cache_entry.had_unconsumed_draws =
                        gpu_written && split.Writes().LastGpu() > done;
                    cache_entry.checked_write_pos = split.WritePosition();
                    cache_entry.checked_code_epoch =
                        Tegra::g_nx2_code_epoch.load(std::memory_order_acquire);
                    cache_entry.last_word = word;
                }
            }
        } else if (buffer_dirty) {
            buffer_dirty_checked = true;
            if (cache_entry.host_word != nullptr) {
                const u32 cur = LoadHostWord(cache_entry.host_word);
                if (!pending_acquire->Matches(cur)) {
                    cache_entry.dirty_throttled = true;
                    cache_entry.last_dirty_word = cur;
                    cache_entry.last_dirty_check = std::chrono::steady_clock::now();
                }
            }
        }
    } else {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::SemGpuWrittenPage, 1);
    }
    if (!cache_entry.readable_settled) {
        if (MetByQueuedRelease(dma_pusher, *pending_acquire, nullptr, known_device,
                               cache_entry.host_word)) {
            return true;
        }
        const auto pending_op = static_cast<GpuSemaphoreOperation>(pending_acquire->operation);
        if (pending_op == GpuSemaphoreOperation::AcquireGequal &&
            GequalMetWithoutDrain(dma_pusher, pending_acquire->address, pending_acquire->value,
                                  readable_word, known_device, cache_entry.host_word)) {
            return true;
        }
        if ((pending_op == GpuSemaphoreOperation::AcquireEqual ||
             pending_op == GpuSemaphoreOperation::AcquireMask) &&
            EqualMetWithoutDrain(dma_pusher, *pending_acquire, known_device,
                                 cache_entry.host_word)) {
            return true;
        }
    }
    if (split.Active()) {
        const auto now = std::chrono::steady_clock::now();
        const bool timeout_due =
            !pending_acquire->reported && now - pending_acquire->since > std::chrono::seconds{2};
        if (!timeout_due) {
            split.Publish();
            if (readable_word.has_value()) {
                if (delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
                    pending_acquire->last_fences = now;
                    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                           VideoCommon::GpuSplit::Effect::None);
                    split.Publish();
                    split.RequestIdleHook();
                }
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                return false;
            }
            if (known_device) {
                const DAddr device_addr = *known_device;
                const u64 done = split.ConsumedNow();
                const u64 async_pos = split.HasAsyncWrite(device_addr, sizeof(u32));
                const bool async_unsettled =
                    async_pos != 0 &&
                    (async_pos > done || !Nx2AsyncReportsSettled(device_addr, sizeof(u32)));
                if (((async_pos != 0 && async_pos <= done && async_unsettled) || delayed) &&
                    !split.IsBackQuiet() &&
                    now - pending_acquire->last_drain >= std::chrono::microseconds{250}) {
                    pending_acquire->last_drain = now;
                    split.RequestIdleHook();
                }
                if ((delayed || async_unsettled) &&
                    now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
                    pending_acquire->last_fences = now;
                    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                           VideoCommon::GpuSplit::Effect::None);
                    split.Publish();
                    split.RequestIdleHook();
                }
                const auto queued_rel = split.QueuedReleaseTo(device_addr, done);
                const bool memory_matches =
                    (!queued_rel || pending_acquire->Matches(*queued_rel)) &&
                    pending_acquire->Matches(ReadCachedOrMemoryWord(
                        dma_pusher.memory_manager, pending_acquire->address, cache_entry.host_word));
                bool need_buffer_flush = false;
                if (!memory_matches && Nx2IsGpuWritten(device_addr) &&
                    split.UntrackedWriterPosition(device_addr) <= done &&
                    now - pending_acquire->last_drain >= std::chrono::microseconds{250}) {
                    pending_acquire->last_drain = now;
                    need_buffer_flush =
                        buffer_dirty_checked
                            ? buffer_dirty
                            : dma_pusher.rasterizer->MustFlushRegion(
                                  device_addr, sizeof(u32), VideoCommon::CacheType::BufferCache);
                }
                if (!memory_matches && !need_buffer_flush) {
                    VideoCommon::Nx2Profile::Add(
                        VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                    return false;
                }
                if (!need_buffer_flush &&
                    now - pending_acquire->last_drain < std::chrono::microseconds{250}) {
                    VideoCommon::Nx2Profile::Add(
                        VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                    return false;
                }
                pending_acquire->last_drain = now;
                QueueAsyncSemaphoreProbe(dma_pusher, pending_acquire->address, device_addr);
                if (need_buffer_flush) {
                    split.RequestIdleHook();
                }
                VideoCommon::Nx2Profile::Add(
                    VideoCommon::Nx2Profile::Counter::AcquirePollsFast, 1);
                return false;
            }
        }
    }
    // The release may still be queued for the split GPU thread's back half (and the caches
    // ReadSemaphoreWord asks are its state).
    DrainBack(VideoCommon::DrainSite::AcquirePoll);
    pending_acquire->last_drain = std::chrono::steady_clock::now();
    const u32 current = ReadSemaphoreWord(dma_pusher, pending_acquire->address);
    if (Nx2SemaphoreWaiters::Get().Latched(this) || SyncpointShimWaiters::Get().Latched(this)) {
        if (!pending_acquire->Matches(current) && rescued.fetch_add(1) < 64) {
            LOG_INFO(HW_GPU, "Channel {} met semaphore {:#x} value {:#x} at its release (now {:#x})",
                     dma_pusher.channel_state.bind_id, pending_acquire->address,
                     pending_acquire->value, current);
        }
        return true;
    }
    if (pending_acquire->Matches(current)) {
        return true;
    }
    // With delayed fence operations (Balanced or Accurate fences, accurate GPU), a release
    // queued after the acquire released the fences waits in the caches for the next fence. With
    // every channel waiting none comes: release them again now and then while waiting.
    if (const auto now = std::chrono::steady_clock::now();
        delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
        pending_acquire->last_fences = now;
        ReleaseFences(dma_pusher);
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            SyncpointShimWaiters::Get().Latched(this) ||
            pending_acquire->Matches(ReadSemaphoreWord(dma_pusher, pending_acquire->address))) {
            return true;
        }
    }
    if (!pending_acquire->reported &&
        std::chrono::steady_clock::now() - pending_acquire->since > std::chrono::seconds{2}) {
        pending_acquire->reported = true;
        const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
        LOG_WARNING(HW_GPU, "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s; "
                    "device {:#x}, buffer cache holds GPU data: {}, words {:08x} {:08x} {:08x} {:08x}",
                    dma_pusher.channel_state.bind_id, pending_acquire->address,
                    pending_acquire->operation, pending_acquire->value, current, device.value_or(0),
                    device && dma_pusher.rasterizer->MustFlushRegion(*device, 16, VideoCommon::CacheType::All),
                    dma_pusher.memory_manager.Read<u32>(pending_acquire->address & ~GPUVAddr{0xf}),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 4),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 8),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 12));
    }
    return false;
}

bool Puller::ResumeAcquire(DmaPusher& dma_pusher) {
    if (!CanResumeAcquire(dma_pusher)) {
        return false;
    }
    // Only a resume from a suspension counts: DmaPusher::Step calls this before every entry, and
    // counting those made the count odd ("suspended") at random, so submit-time snapshots were
    // thrown away for about half of all entries (CommandList::snapshot_suspensions).
    if (pending_acquire) {
        if (pending_acquire->readable_release) {
            auto& stats = FenceWriterStats::Get();
            ++stats.resumed;
            stats.resume_ns += FenceWriterStats::Nanoseconds(
                pending_acquire->since, FenceWriterStats::Clock::now());
        }
        pending_acquire.reset();
        dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
        Nx2SemaphoreWaiters::Get().Remove(this);
        SyncpointShimWaiters::Get().Clear(this, dma_pusher.system.Host1x().GetSyncpointManager());
        AcquirePollCache::Get().Invalidate(this, dma_pusher.channel_state.bind_id);
    }
    return true;
}

void Puller::ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_DEBUG(HW_GPU, "Binding subchannel {} to engine {:#x}", method_call.subchannel, method_call.argument);
    u32 engine = method_call.argument;
    if ((engine & ~Gf100BindClassMask) != 0 && (engine & ~Gf100BindValidMask) == 0) {
        engine &= Gf100BindClassMask;
    }
    // Switch 2 binds Ampere classes; run them on the Maxwell engines (same method layout for
    // most state). Class 0 unbinds the subchannel.
    switch (engine & 0xFFFF) {
    case 0xC797: // AMPERE_B
        engine = static_cast<u32>(EngineID::MAXWELL_B);
        break;
    case 0xC7C0: // AMPERE_COMPUTE_B
        engine = static_cast<u32>(EngineID::KEPLER_COMPUTE_B);
        break;
    case 0xC7B5: // AMPERE_DMA_COPY_B
        engine = static_cast<u32>(EngineID::MAXWELL_DMA_COPY_A);
        break;
    case 0x0:
        if (g_nx2_ampere) {
            LOG_DEBUG(HW_GPU, "Switch 2 unbinds subchannel {}", method_call.subchannel);
            return;
        }
        break;
    default:
        break;
    }

    const auto engine_id = static_cast<EngineID>(engine);
    bound_engines[method_call.subchannel] = engine_id;
    switch (engine_id) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->fermi_2d, method_call.subchannel, EngineTypes::Fermi2D);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_3d, method_call.subchannel, EngineTypes::Maxwell3D);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_compute, method_call.subchannel, EngineTypes::KeplerCompute);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_dma, method_call.subchannel, EngineTypes::MaxwellDMA);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_memory, method_call.subchannel, EngineTypes::KeplerMemory);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->nv01_timer, method_call.subchannel, EngineTypes::Nv01Timer);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine {:04X}", engine_id);
        break;
    }
}

void Puller::ProcessFenceActionMethod(DmaPusher& dma_pusher) {
    switch (regs.fence_action.op) {
    case Puller::FenceOperation::Acquire:
        // UNIMPLEMENTED_MSG("Channel Scheduling pending.");
        // WaitFence(regs.fence_action.syncpoint_id, regs.fence_value);
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
               VideoCommon::GpuSplit::Effect::None);
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case Puller::FenceOperation::Increment:
        OnBack(
            [rasterizer = dma_pusher.rasterizer,
             id = static_cast<u32>(regs.fence_action.syncpoint_id.Value())] {
                rasterizer->SignalSyncPoint(id);
            },
            VideoCommon::GpuSplit::Effect::None);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented operation {}", regs.fence_action.op.Value());
        break;
    }
}

void Puller::ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher) {
    const auto semaphoreOperationMask = 0xF;
    const auto op = GpuSemaphoreOperation(regs.semaphore_trigger & semaphoreOperationMask);
    // EDEN_NX2_SEM_WATCH=<gpu address>: log every release/acquire on the 16 bytes there.
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (watch != 0 && (regs.semaphore_address.SemaphoreAddress() & ~GPUVAddr{0xf}) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} as {} device {:#x} op {} {:#x} value {:#x} (now {:#x})",
                 dma_pusher.channel_state.bind_id, dma_pusher.memory_manager.GetID(),
                 dma_pusher.memory_manager.GpuToCpuAddress(regs.semaphore_address.SemaphoreAddress()).value_or(0),
                 static_cast<u32>(op),
                 regs.semaphore_address.SemaphoreAddress(), regs.semaphore_sequence,
                 dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress()));
    }
    if (op == GpuSemaphoreOperation::WriteLong) {
        const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
        const u32 payload = regs.semaphore_sequence;
        // RELEASE_SIZE (bit 24) set means a 4-byte release without the timestamp; writing 16
        // bytes there clobbers the words next to it (Switch 2 packs semaphores tightly).
        const auto properties = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0
                                    ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout;
        const GPUVAddr mirror = g_nx2_ampere ? Nx2SemaphoreMirror(sequence_address) : 0;
        if (mirror != 0) {
            LOG_DEBUG(HW_GPU, "Switch 2 semaphore {:#x}={:#x} mirrored to {:#x}", sequence_address, payload, mirror);
            VideoCommon::GpuSplit::Get().NoteWrite(dma_pusher.memory_manager, mirror, 16);
        }
        OnBackRelease(dma_pusher, sequence_address,
                      [rasterizer = dma_pusher.rasterizer, sequence_address, mirror, properties, payload] {
                          rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload, properties, payload, 0);
                          if (mirror != 0) {
                              rasterizer->Query(mirror, VideoCommon::QueryType::Payload, properties, payload, 0);
                          }
                      });
        if (AcquireStatsEnabled() && VideoCommon::GpuSplit::Get().Active()) {
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(sequence_address)) {
                QueuedReleases()[*device] = {VideoCommon::GpuSplit::Get().WritePosition(), payload};
            }
            if (mirror != 0) {
                if (const auto mirror_device = dma_pusher.memory_manager.GpuToCpuAddress(mirror)) {
                    QueuedReleases()[*mirror_device] = {
                        VideoCommon::GpuSplit::Get().WritePosition(), payload};
                }
            }
        }
        // After the release's entry was sent, so its position covers it.
        if (auto& split = VideoCommon::GpuSplit::Get();
            mirror != 0 && split.Active() &&
            (VideoCore::Nx2Debug::DelayedFences() || VideoCore::Nx2Debug::AccuratePart(1))) {
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(mirror)) {
                split.NoteAsyncWrite(*device, 16);
            }
        }
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(sequence_address)) {
                split.NoteRelease(*device, sequence_address, payload);
            }
            if (mirror != 0) {
                if (const auto mirror_device = dma_pusher.memory_manager.GpuToCpuAddress(mirror)) {
                    split.NoteRelease(*mirror_device, mirror, payload);
                }
            }
        }
    } else if (g_nx2_ampere) {
        // Switch 2 waits on semaphores the CPU (or another channel) releases, e.g. once it has
        // written the commands this channel runs next: suspend the channel until the value is
        // met instead of reading ahead (approach from changes.patch).
        PendingAcquire acquire{
            .address = regs.semaphore_address.SemaphoreAddress(),
            .value = regs.semaphore_sequence,
            .operation = static_cast<u32>(op),
            .since = std::chrono::steady_clock::now(),
            .last_fences = std::chrono::steady_clock::now(),
            .reported = false,
        };
        if (op != GpuSemaphoreOperation::AcquireEqual && op != GpuSemaphoreOperation::AcquireGequal &&
            op != GpuSemaphoreOperation::AcquireMask) {
            LOG_ERROR(HW_GPU, "Invalid semaphore operation {}", static_cast<u32>(op));
            return;
        }
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireChecks, 1);
        int acquire_class = -1; // 0 queued release, 1 memory, 2 other
        if (AcquireStatsEnabled() && VideoCommon::GpuSplit::Get().Active()) {
            auto& stats = AcquireStats();
            ++stats.total;
            const u64 consumed_now = VideoCommon::GpuSplit::Get().ConsumedNow();
            const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
            const auto it = device ? QueuedReleases().find(*device) : QueuedReleases().end();
            const bool queued = it != QueuedReleases().end() && it->second.position > consumed_now;
            if (queued) {
                const bool met = acquire.Matches(it->second.payload);
                acquire_class = met ? 0 : 2;
                ++(met ? stats.queued_release : stats.queued_release_unmet);
            } else if (acquire.Matches(dma_pusher.memory_manager.Read<u32>(acquire.address))) {
                acquire_class = 1;
                ++stats.memory;
            } else {
                acquire_class = 2;
                ++stats.other;
            }
            if ((stats.total % 20000) == 0) {
                LOG_INFO(HW_GPU,
                         "nx2 acquire stats: {} acquire triggers: {} met by a queued release, "
                         "{} with a queued release that does not meet it, {} met in memory with "
                         "none queued, {} other; {} still unmet after the drain",
                         stats.total, stats.queued_release, stats.queued_release_unmet,
                         stats.memory, stats.other, stats.unmet_after);
            }
        }
        if (const auto syncpoint_id =
                NativeSyncpointShimId(dma_pusher.memory_manager, acquire.address)) {
            auto& split = VideoCommon::GpuSplit::Get();
            const bool delayed = VideoCore::Nx2Debug::DelayedFences();
            const u32 current = dma_pusher.memory_manager.Read<u32>(acquire.address);
            const bool settled_now =
                !delayed && (!split.Active() || split.ConsumedNow() == split.WritePosition());
            if ((op == GpuSemaphoreOperation::AcquireGequal && acquire.Matches(current)) ||
                (op != GpuSemaphoreOperation::AcquireGequal && settled_now &&
                 acquire.Matches(current))) {
                split.NoteSync();
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresFast, 1);
                return;
            }
            std::optional<u32> expected_value;
            bool latch_on_reach = false;
            if (op == GpuSemaphoreOperation::AcquireGequal) {
                expected_value = acquire.value;
            } else if (op == GpuSemaphoreOperation::AcquireEqual) {
                if (current < acquire.value) {
                    expected_value = acquire.value;
                    latch_on_reach = true;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (!acquire.Matches(current) && acquire.value != 0 && current != ~0U) {
                    expected_value = current + 1;
                }
            }
            auto& syncpoints = dma_pusher.system.Host1x().GetSyncpointManager();
            const u64 target_pos = split.Active() ? split.WritePosition() : 0;
            SyncpointShimWaiters::Get().Register(this, syncpoints, *syncpoint_id, expected_value,
                                                 latch_on_reach, target_pos);
            if (!split.Active()) {
                ReleaseFences(dma_pusher);
            }
            const u32 rechecked = dma_pusher.memory_manager.Read<u32>(acquire.address);
            const bool target_reached = !split.Active() || split.ConsumedNow() >= target_pos;
            if (SyncpointShimWaiters::Get().Latched(this) ||
                (op == GpuSemaphoreOperation::AcquireGequal && acquire.Matches(rechecked)) ||
                (op != GpuSemaphoreOperation::AcquireGequal && target_reached &&
                 acquire.Matches(rechecked))) {
                SyncpointShimWaiters::Get().Clear(this, syncpoints);
                split.NoteSync();
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquiresFast, 1);
                return;
            }
            const auto now = std::chrono::steady_clock::now();
            pending_acquire = acquire;
            pending_acquire->last_drain = now;
            pending_acquire->last_fences = now;
            pending_acquire->readable_release = false;
            AcquirePollCache::Get().Invalidate(this, dma_pusher.channel_state.bind_id);
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            split.NoteSync();
            if (split.Active()) {
                if (op != GpuSemaphoreOperation::AcquireGequal && !target_reached &&
                    acquire.Matches(rechecked)) {
                    split.Defer(
                        [] { Tegra::Nx2PendingWake::Get().Notify(Tegra::Nx2PendingWake::ByLatch); },
                        VideoCommon::GpuSplit::Effect::None);
                }
                if (delayed) {
                    OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                           VideoCommon::GpuSplit::Effect::None);
                }
                split.Publish();
                if (delayed) {
                    split.RequestIdleHook();
                }
            }
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireNoDrain, 1);
            if (AcquireTrace::Enabled()) {
                AcquireTrace::Get().Record(acquire.address, acquire.operation,
                                           dma_pusher.channel_state.bind_id, 0, current,
                                           acquire.value, 0, AcquireTrace::Suspended, 0, 0);
            }
            if (AcquireLearning::Enabled()) {
                AcquireLearning::Instance().NoteSuspended(acquire.address);
            }
            return;
        }
        const bool map_pending = dma_pusher.system.GPU().IsMapPending(acquire.address, sizeof(u32));
        const std::optional<DAddr> initial_device =
            map_pending ? std::nullopt : dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
        u32 word{};
        const bool readable =
            SemaphoreReadableNow(dma_pusher, acquire.address, word, initial_device);
        if (readable && acquire.Matches(word)) {
            return;
        }
        u32 miss = 0;
        if (MetByQueuedRelease(dma_pusher, acquire, &miss, initial_device)) {
            return;
        }
        if (op == GpuSemaphoreOperation::AcquireGequal &&
            GequalMetWithoutDrain(dma_pusher, acquire.address, acquire.value,
                                  readable ? std::optional<u32>{word} : std::nullopt,
                                  initial_device)) {
            return;
        }
        if ((op == GpuSemaphoreOperation::AcquireEqual ||
             op == GpuSemaphoreOperation::AcquireMask) &&
            EqualMetWithoutDrain(dma_pusher, acquire, initial_device)) {
            return;
        }
        const bool trace = AcquireTrace::Enabled();
        const u32 word_before = trace ? dma_pusher.memory_manager.Read<u32>(acquire.address) : 0;
        u32 writer_method = 0;
        const u32 writer =
            trace ? VideoCommon::GpuSplit::Get().QueuedWriterKind(
                        acquire.address, acquire.address + sizeof(u32), writer_method)
                  : 0;
        const auto drain_begin = trace ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        const auto record = [&](AcquireTrace::Outcome outcome) {
            if (trace) {
                AcquireTrace::Get().Record(
                    acquire.address, acquire.operation, dma_pusher.channel_state.bind_id, miss,
                    word_before, acquire.value,
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - drain_begin)
                                         .count()),
                    outcome, writer, writer_method);
            }
        };
        const bool learn = AcquireLearning::Enabled();
        if (learn) {
            auto& learning = AcquireLearning::Instance();
            const auto stats = learning.Get(acquire.address);
            if (stats.met >= AcquireLearning::Streak &&
                AcquireLearning::misses.load(std::memory_order_relaxed) == 0) {
                AcquireLearning::speculated.fetch_add(1, std::memory_order_relaxed);
                VideoCommon::GpuSplit::Get().NoteSync();
                OnBack([memory = &dma_pusher.memory_manager, acquire] {
                    if (!acquire.Matches(memory->Read<u32>(acquire.address)) &&
                        AcquireLearning::misses.fetch_add(1) < 8) {
                        LOG_WARNING(HW_GPU, "Learned acquire {:#x} (op {}, {:#x}) not met on the "
                                    "back (word {:#x}): learning off",
                                    acquire.address, acquire.operation, acquire.value,
                                    memory->Read<u32>(acquire.address));
                    }
                });
                return;
            }
            if (stats.suspended >= AcquireLearning::Streak &&
                !acquire.Matches(dma_pusher.memory_manager.Read<u32>(acquire.address))) {
                AcquireLearning::early_suspends.fetch_add(1, std::memory_order_relaxed);
                AddSemaphoreWaiter(dma_pusher, this, acquire, initial_device);
                pending_acquire = acquire;
                AcquirePollCache::Get().Invalidate(this, dma_pusher.channel_state.bind_id);
                dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
                return;
            }
        }
        if (readable) {
            // The word is current (no queued or GPU write the front knows of can change it):
            // the value comes from the CPU or another channel later, and the polls in
            // CanResumeAcquire pick it up without a drain (from zortos293/Lichii#231).
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireNoDrain, 1);
        } else {
            auto& split = VideoCommon::GpuSplit::Get();
            if (AcquireEarlyEnabled() && split.Active() &&
                AcquireEarly(dma_pusher, acquire.address,
                             [&acquire](u32 value) { return acquire.Matches(value); })) {
                if (AcquireEarlyVerify()) {
                    DrainBack(VideoCommon::DrainSite::AcquireTrigger);
                    const u32 verified = ReadSemaphoreWord(dma_pusher, acquire.address);
                    if (!acquire.Matches(verified) &&
                        AcquireEarlyStats().verify_mismatches++ < 64) {
                        LOG_ERROR(HW_GPU,
                                  "nx2 acquire early: {:#x} op {} value {:#x} met early, word "
                                  "{:#x} after a drain does not meet it",
                                  acquire.address, acquire.operation, acquire.value, verified);
                    }
                }
                return;
            }
            // Suspend the channel without draining GPUBack when the word lives on a GPU-written
            // page or has a pending report: other channels keep pushing while GPUBack flushes and
            // latches the semaphore word asynchronously in stream order.
            if (split.Active() &&
                !dma_pusher.system.GPU().IsMapPending(acquire.address, sizeof(u32))) {
                if (const auto device =
                        initial_device
                            ? initial_device
                            : dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
                    device && AddSemaphoreWaiter(dma_pusher, this, acquire, device)) {
                    const auto now = std::chrono::steady_clock::now();
                    pending_acquire = acquire;
                    pending_acquire->last_drain = now;
                    pending_acquire->last_fences = now;
                    pending_acquire->readable_release = false;
                    AcquirePollCache::Get().Invalidate(this, dma_pusher.channel_state.bind_id);
                    dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
                    split.NoteSync();
                    QueueAsyncSemaphoreProbe(dma_pusher, acquire.address, *device);
                    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
                    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireNoDrain,
                                                 1);
                    record(AcquireTrace::Suspended);
                    if (learn) {
                        AcquireLearning::Instance().NoteSuspended(acquire.address);
                    }
                    return;
                }
            }
            DrainBack(VideoCommon::DrainSite::AcquireTrigger);
            const u32 drained = ReadSemaphoreWord(dma_pusher, acquire.address);
            if (acquire_class >= 0 && !acquire.Matches(drained)) {
                ++AcquireStats().unmet_after;
                if (acquire_class != 2) {
                    LOG_WARNING(HW_GPU,
                                "nx2 acquire stats: class {} acquire at {:#x} value {:#x} unmet "
                                "after the drain (word {:#x})",
                                acquire_class, acquire.address, acquire.value, drained);
                }
            }
            if (acquire.Matches(drained)) {
                VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireMetAfterDrain, 1);
                record(AcquireTrace::MetAfterDrain);
                if (learn) {
                    AcquireLearning::Instance().NoteMet(acquire.address);
                }
                return;
            }
        }
        // Registered before the fences run, so a release among them that meets the value counts
        // even if a later one overwrites it. The word in memory is watched for the guest CPU's
        // stores (Nx2SemaphoreWaiters::WordStored).
        const auto current_device =
            readable && initial_device
                ? initial_device
                : dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
        auto& waiters = Nx2SemaphoreWaiters::Get();
        if (!AddSemaphoreWaiter(dma_pusher, this, acquire, current_device)) {
            if (static std::atomic<u32> full{}; full.fetch_add(1) < 8) {
                LOG_ERROR(HW_GPU, "Channel {} waits on semaphore {:#x} unlatched: {} channels wait",
                          dma_pusher.channel_state.bind_id, acquire.address,
                          Nx2SemaphoreWaiters::MaxWaiters);
            }
        }
        bool met_now;
        if (readable) {
            const auto qmd_check = CheckQmdPrediction(dma_pusher, acquire);
            met_now = ReleaseFencesForReadable(dma_pusher, this, acquire, current_device);
            if (qmd_check) {
                QmdPredictionMetBy(*qmd_check, acquire, met_now || waiters.Latched(this));
            }
        } else {
            ReleaseFences(dma_pusher);
            met_now = acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address));
        }
        if (!waiters.Latched(this) && !met_now) {
            pending_acquire = acquire;
            pending_acquire->last_drain = std::chrono::steady_clock::now();
            pending_acquire->readable_release = readable;
            AcquirePollCache::Get().Invalidate(this, dma_pusher.channel_state.bind_id);
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
            record(AcquireTrace::Suspended);
            if (learn) {
                AcquireLearning::Instance().NoteSuspended(acquire.address);
            }
        } else {
            waiters.Remove(this);
            record(AcquireTrace::MetAfterFences);
            if (learn) {
                AcquireLearning::Instance().NoteMet(acquire.address);
            }
        }
    } else {
        DrainBack();
        do {
            const u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
            regs.acquire_source = true;
            regs.acquire_value = regs.semaphore_sequence;
            if (op == GpuSemaphoreOperation::AcquireEqual) {
                regs.acquire_active = true;
                regs.acquire_mode = false;
                if (word != regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireGequal) {
                regs.acquire_active = true;
                regs.acquire_mode = true;
                if (word < regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (word && regs.semaphore_sequence == 0) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else {
                LOG_ERROR(HW_GPU, "Invalid semaphore operation");
            }
        } while (false);
    }
}

void Puller::ProcessSemaphoreRelease(DmaPusher& dma_pusher) {
    const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
    const u32 payload = regs.semaphore_release;
    OnBackRelease(dma_pusher, sequence_address,
                  [rasterizer = dma_pusher.rasterizer, sequence_address, payload] {
                      rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload,
                                        VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
                  });
}

void Puller::ProcessSemaphoreAcquire(DmaPusher& dma_pusher) {
    DrainBack();
    u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
    const auto value = regs.semaphore_acquire;
    while (word != value) {
        regs.acquire_active = true;
        regs.acquire_value = value;
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
        // TODO(kemathe73) figure out how to do the acquire_timeout
        regs.acquire_mode = false;
        regs.acquire_source = false;
    }
}

// Ampere SEM_EXECUTE (method 0x1B): operation in bits 2:0 on the semaphore at
// SEM_ADDR_HI:SEM_ADDR_LO (methods 0x18/0x17) with payload SEM_PAYLOAD_LO (0x19).
void Puller::ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument) {
    const GPUVAddr address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                             (regs.reg_array[0x17] & ~3U);
    const u32 payload = regs.reg_array[0x19];
    const u32 operation = argument & 7;
    if (operation == 1) { // RELEASE
        const GPUVAddr mirror = g_nx2_ampere ? Nx2SemaphoreMirror(address) : 0;
        if (mirror != 0) {
            VideoCommon::GpuSplit::Get().NoteWrite(dma_pusher.memory_manager, mirror, 4);
        }
        OnBackRelease(dma_pusher, address,
                      [rasterizer = dma_pusher.rasterizer, address, mirror, payload] {
                          rasterizer->Query(address, VideoCommon::QueryType::Payload,
                                            VideoCommon::QueryPropertiesFlags::IsAFence, payload,
                                            0);
                          if (mirror != 0) {
                              rasterizer->Query(mirror, VideoCommon::QueryType::Payload,
                                                VideoCommon::QueryPropertiesFlags::IsAFence,
                                                payload, 0);
                          }
                      });
        if (auto& split = VideoCommon::GpuSplit::Get();
            mirror != 0 && split.Active() &&
            (VideoCore::Nx2Debug::DelayedFences() || VideoCore::Nx2Debug::AccuratePart(1))) {
            if (const auto mirror_device = dma_pusher.memory_manager.GpuToCpuAddress(mirror)) {
                split.NoteAsyncWrite(*mirror_device, 4);
            }
        }
        // An acquire it meets while queued needs no drain (from zortos293/Lichii#179).
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address)) {
                split.NoteRelease(*device, address, payload);
            }
            if (mirror != 0) {
                if (const auto mirror_device = dma_pusher.memory_manager.GpuToCpuAddress(mirror)) {
                    split.NoteRelease(*mirror_device, mirror, payload);
                }
            }
        }
        return;
    }
    // Acquires: the result only feeds the warning below, so with the split the pending fences are
    // released in order on the back instead of after a drain (from zortos293/Lichii#179). Later
    // GPU-written list fetches and macro reads still wait for the work before it (NoteSync).
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
        split.NoteSync();
        return;
    }
    // Acquires: check once after flushing pending fences; never block the GPU thread on them.
    static std::atomic<u32> reported{};
    static std::atomic<u32> checks{};
    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_sem_fast, "EDEN_NX2_SEM_FAST") != 0 &&
        VideoCommon::GpuSplit::Get().Active()) {
        // Only the warning below reads the result: release the fences in order instead.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
               VideoCommon::GpuSplit::Effect::None);
        return;
    }
    // The check only feeds the warning below: drain for it during the first checks only.
    if (reported.load(std::memory_order_relaxed) >= 8 ||
        checks.fetch_add(1, std::memory_order_relaxed) >= 256) {
        return;
    }
    DrainBack();
    ReleaseFences(dma_pusher);
    const u32 word = dma_pusher.memory_manager.Read<u32>(address);
    const bool ready = operation == 0   ? word == payload
                       : operation == 2 ? word >= payload
                       : operation == 3 ? static_cast<s32>(word - payload) >= 0
                       : operation == 4 ? (word & payload) != 0
                       : operation == 5 ? (word & payload) == 0
                                        : true;
    if (!ready) {
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(HW_GPU, "SEM_EXECUTE op {} at {:#x}: {:#x} vs {:#x} not met, continuing",
                        operation, address, word, payload);
        }
    }
}

// A semaphore release queued for the split's back: the front's macro DMA reads of those bytes
// wait for it (EDEN_NX2_DMA_WAIT_WRITERS).
// Immediate payload releases write when the back runs them, so reads overlay their payload words.
static void NoteSemaphoreWrite(DmaPusher& dma_pusher, GPUVAddr address, u32 payload,
                               bool is_fence) {
    auto& split = VideoCommon::GpuSplit::Get();
    split.NoteMemoryWriter();
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        if (dma_pusher.system.GPU().IsMapPending(address, 16)) {
            split.NoteMemoryWrite(0, ~u64{0}, "semaphore (pending map)");
        }
        return;
    }
    if (is_fence) {
        if (VideoCore::Nx2Debug::DelayedFences()) {
            split.NoteAsyncWrite(*device, 4);
            split.NoteMemoryWrite(*device, 4, "semaphore (delayed fence)");
        } else {
            split.NoteMemoryWriteData(*device, &payload, 1);
        }
        return;
    }
    if (VideoCore::Nx2Debug::AccuratePart(1)) {
        split.NoteAsyncWrite(*device, 16);
        split.NoteMemoryWrite(*device, 16, "semaphore (timed)");
        return;
    }
    const std::array<u32, 2> low_words{payload, 0U};
    split.NoteMemoryWriteData(*device, low_words.data(), 2);
    split.NoteMemoryWrite(*device + 8, 8, "semaphore timestamp");
}

// EDEN_NX2_SEM_FAST: whether the word at `address` can be read without draining the back: not
// dirty in BufferCache (with no untracked GPU writers queued), after waiting only for the queued
// writes noted to it.
// =2 also overlays the queued fence releases the front knows, instead of waiting for them.
static bool SemaphoreReadableNow(DmaPusher& dma_pusher, GPUVAddr address, u32& word,
                                 std::optional<DAddr> known_device, const u32* host_word,
                                 bool* buffer_dirty_out) {
    if (buffer_dirty_out != nullptr) {
        *buffer_dirty_out = false;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_sem_fast, "EDEN_NX2_SEM_FAST");
    using VideoCommon::Nx2Profile::Counter;
    if (!split.Active() || mode == 0) {
        VideoCommon::Nx2Profile::Add(Counter::SemNoFastPath, 1);
        return false;
    }
    if (dma_pusher.system.GPU().IsMapPending(address, sizeof(u32))) {
        VideoCommon::Nx2Profile::Add(Counter::SemGpuWrittenPage, 1);
        return false;
    }
    const auto device =
        known_device ? known_device : dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        VideoCommon::Nx2Profile::Add(Counter::SemGpuWrittenPage, 1);
        return false;
    }
    if (Nx2IsGpuWritten(*device)) {
        const u64 done = split.ConsumedNow();
        const u64 epoch = Tegra::g_nx2_code_epoch.load(std::memory_order_acquire);
        const bool draws_safe =
            split.Writes().LastGpu() <= done ||
            (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet) &&
             split.Writes().DrawsValid(done, epoch) &&
             !split.Writes().AnyPendingWrite(done, split.Writes().LastGpu(),
                                            [](u64 begin, u64 end) {
                                                return Tegra::Nx2IsProvenCode(begin, end - begin);
                                            }));
        if (!draws_safe || split.UntrackedWriterPosition(*device) > done) {
            VideoCommon::Nx2Profile::Add(Counter::SemGpuWrittenPage, 1);
            return false;
        }
        if (dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                   VideoCommon::CacheType::BufferCache)) {
            if (buffer_dirty_out != nullptr) {
                *buffer_dirty_out = true;
            }
            VideoCommon::Nx2Profile::Add(Counter::SemGpuWrittenPage, 1);
            return false;
        }
    }
    if (const u64 async_position = split.HasAsyncWrite(*device, 4);
        async_position != 0 &&
        (async_position > split.ConsumedNow() || !Nx2AsyncReportsSettled(*device, 4))) {
        VideoCommon::Nx2Profile::Add(Counter::SemAsyncPending, 1);
        return false;
    }
    if (mode >= 2 && !Nx2IsGpuWritten(*device)) {
        // Not only the noted writes: a queued compute or copy writer, an unranged writer or a
        // queued release may change the word too (CurrentReleaseWord declines those, reading
        // memory with the noted writes overlaid otherwise).
        if (const auto current =
                split.CurrentReleaseWord(dma_pusher.memory_manager, *device, address, 0, nullptr)) {
            word = *current;
            return true;
        }
    }
    if (!split.WaitForWritesTo(*device, 16)) {
        VideoCommon::Nx2Profile::Add(Counter::SemWritersWait, 1);
        return false;
    }
    word = ReadCachedOrMemoryWord(dma_pusher.memory_manager, address, host_word);
    return true;
}

/// Calls a GPU puller method.
void Puller::CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    static const bool nx2_trace = std::getenv("EDEN_NX2_PULLER_TRACE") != nullptr;
    if (nx2_trace) {
        LOG_INFO(HW_GPU, "nx2 puller sub={} method={:#x} arg={:#x}", method_call.subchannel,
                 method_call.method, method_call.argument);
    }
    regs.reg_array[method_call.method] = method_call.argument;
    const auto method = static_cast<BufferMethods>(method_call.method);

    switch (method) {
    case BufferMethods::BindObject: {
        ProcessBindMethod(dma_pusher, method_call);
        break;
    }
    case BufferMethods::Nop:
    case BufferMethods::SemaphoreAddressHigh:
    case BufferMethods::SemaphoreAddressLow:
    case BufferMethods::SemaphoreSequencePayload:
    case BufferMethods::SyncpointPayload:
    case BufferMethods::WrcacheFlush:
        break;
    case BufferMethods::RefCnt:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->SignalReference(); },
               VideoCommon::GpuSplit::Effect::None);
        break;
    case BufferMethods::SyncpointOperation:
        ProcessFenceActionMethod(dma_pusher);
        if (VideoCore::Nx2Debug::DelayedFences()) {
            VideoCommon::GpuSplit::Get().NoteMemoryWriter();
        }
        break;
    case BufferMethods::WaitForIdle:
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->WaitForIdle(); },
               VideoCommon::GpuSplit::Effect::None);
        VideoCommon::GpuSplit::Get().NoteSync();
        break;
    case BufferMethods::SemaphoreOperation: {
        ProcessSemaphoreTriggerMethod(dma_pusher);
        if (static_cast<GpuSemaphoreOperation>(regs.semaphore_trigger & 0xF) ==
            GpuSemaphoreOperation::WriteLong) {
            const GPUVAddr sequence_address = regs.semaphore_address.SemaphoreAddress();
            const bool fence = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0;
            NoteSemaphoreWrite(dma_pusher, sequence_address, regs.semaphore_sequence, fence);
            if (const GPUVAddr mirror = g_nx2_ampere ? Nx2SemaphoreMirror(sequence_address) : 0;
                mirror != 0) {
                NoteSemaphoreWrite(dma_pusher, mirror, regs.semaphore_sequence, fence);
            }
        }
        break;
    }
    case BufferMethods::NonStallInterrupt: {
        LOG_ERROR(HW_GPU, "Special puller engine method NonStallInterrupt not implemented");
        break;
    }
    case BufferMethods::MemOpA: {
        LOG_TRACE(HW_GPU, "Memory Operation A");
        break;
    }
    case BufferMethods::MemOpB: {
        // Implement this better.
        OnBack([rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); },
               VideoCommon::GpuSplit::Effect::None);
        break;
    }
    case BufferMethods::MemOpC:
    case BufferMethods::MemOpD: {
        LOG_TRACE(HW_GPU, "Memory Operation C,D");
        break;
    }
    case BufferMethods::SemaphoreAcquire: {
        // On Ampere host classes 0x17..0x1A are SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI.
        if (!g_nx2_ampere) {
            ProcessSemaphoreAcquire(dma_pusher);
        }
        break;
    }
    case BufferMethods::SemaphoreRelease: {
        if (g_nx2_ampere) {
            ProcessSemaphoreExecute(dma_pusher, method_call.argument);
        } else {
            ProcessSemaphoreRelease(dma_pusher);
        }
        if (g_nx2_ampere && (method_call.argument & 7) == 1) {
            const GPUVAddr sem_address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                                         (regs.reg_array[0x17] & ~3U);
            const u32 sem_payload = regs.reg_array[0x19];
            NoteSemaphoreWrite(dma_pusher, sem_address, sem_payload, true);
            if (const GPUVAddr mirror = Nx2SemaphoreMirror(sem_address); mirror != 0) {
                NoteSemaphoreWrite(dma_pusher, mirror, sem_payload, true);
            }
        } else if (!g_nx2_ampere) {
            NoteSemaphoreWrite(dma_pusher, regs.semaphore_address.SemaphoreAddress(),
                               regs.semaphore_release, true);
        }
        break;
    }
    case BufferMethods::Yield: {
        // TODO(Kmather73): Research and implement this method.
        LOG_ERROR(HW_GPU, "Special puller engine method Yield not implemented");
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Special puller engine method {:X} not implemented", method);
        if (static bool dumped = false; g_nx2_ampere && !dumped) {
            dumped = true;
            dma_pusher.DumpCurrentEntry("unknown host method");
        }
        break;
    }
}

/// Calls a GPU engine method.
static bool Nx2MethodTrace() {
    static const bool trace = std::getenv("EDEN_NX2_METHOD_TRACE") != nullptr;
    return trace;
}

void Puller::CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    const EngineID engine = bound_engines[method_call.subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 method sub={} engine={:04X} method={:#x} arg={:#x}",
                 method_call.subchannel, static_cast<u32>(engine), method_call.method,
                 method_call.argument);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU engine multivalue method.
void Puller::CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    const EngineID engine = bound_engines[subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 multimethod sub={} engine={:04X} method={:#x} amount={} first={:#x}",
                 subchannel, static_cast<u32>(engine), method, amount, base_start[0]);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU method.
void Puller::CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method_call.method, method_call.subchannel);
    ASSERT(method_call.subchannel < bound_engines.size());

    if (ExecuteMethodOnEngine(dma_pusher, method_call.method)) {
        CallEngineMethod(dma_pusher, method_call);
    } else {
        CallPullerMethod(dma_pusher, method_call);
    }
}

/// Calls a GPU multivalue method.
void Puller::CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method, subchannel);
    ASSERT(subchannel < bound_engines.size());
    if (ExecuteMethodOnEngine(dma_pusher, method)) {
        CallEngineMultiMethod(dma_pusher, method, subchannel, base_start, amount, methods_pending);
    } else {
        for (u32 i = 0; i < amount; i++) {
            CallPullerMethod(dma_pusher, MethodCall{
                method,
                base_start[i],
                subchannel,
                methods_pending - i,
            });
        }
    }
}

/// Determines where the method should be executed.
[[nodiscard]] bool Puller::ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method) {
    const auto buffer_method = BufferMethods(method);
    return buffer_method >= BufferMethods::NonPullerMethods;
}

} // namespace Tegra::Engines
