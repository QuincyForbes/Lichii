// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

#include "core/device_memory_manager.h"
#include "video_core/nx2_perf.h"

namespace Core {

class GPUDirtyMemoryManager {
public:
    GPUDirtyMemoryManager() {
        for (auto& slot : current) {
            slot.store(default_transform, std::memory_order_relaxed);
        }
        back_buffer.reserve(256);
        front_buffer.reserve(256);
    }

    ~GPUDirtyMemoryManager() = default;

    /// Collects [address, address + size) and returns true when at least one 64-byte granule bit
    /// was newly set since the last Gather.
    bool Collect(PAddr address, size_t size) {
        // A transform covers one 2 KiB page: a write past its page goes in a page at a time. Its
        // mask was otherwise built with shifts of 32 or more, which could leave it wrong or empty
        // and lose the write (zortos293/Lichii#1586, #1591).
        if ((address & page_mask) + size <= page_size || !SplitAcrossPages()) {
            return CollectInPage(address, size);
        }
        bool any_new = false;
        while (size != 0) {
            const size_t in_page = (std::min<size_t>)(size, page_size - (address & page_mask));
            any_new |= CollectInPage(address, in_page);
            address += in_page;
            size -= in_page;
        }
        return any_new;
    }

    /// EDEN_NX2_COLLECT_SPLIT=0: a write crossing a 2 KiB page is collected as one transform, as
    /// before.
    [[nodiscard]] static bool SplitAcrossPages() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_COLLECT_SPLIT");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

private:
    bool CollectInPage(PAddr address, size_t size) {
        const TransformAddress t = BuildTransform(address, size);
        auto& slot = current[t.address & (num_current_slots - 1)];
        TransformAddress tmp = slot.load(std::memory_order_acquire);
        if (tmp.address == t.address && (tmp.mask | t.mask) == tmp.mask) {
            return false;
        }
        TransformAddress original;
        do {
            original = tmp;
            if (tmp.address != t.address) {
                if (IsValid(tmp.address)) {
                    std::scoped_lock lk(guard);
                    const TransformAddress prev = slot.exchange(t, std::memory_order_seq_cst);
                    if (prev.address == t.address) {
                        if ((prev.mask | t.mask) == prev.mask) {
                            return false;
                        }
                        TransformAddress merged = t;
                        merged.mask |= prev.mask;
                        slot.store(merged, std::memory_order_seq_cst);
                    } else if (IsValid(prev.address)) {
                        back_buffer.emplace_back(prev);
                        has_back.store(true, std::memory_order_release);
                    }
                    NoteAnyWrites();
                    return true;
                }
                tmp.address = t.address;
                tmp.mask = 0;
            }
            if ((tmp.mask | t.mask) == tmp.mask) {
                return false;
            }
            tmp.mask |= t.mask;
            if (slot.compare_exchange_weak(original, tmp, std::memory_order_seq_cst,
                                           std::memory_order_acquire)) {
                break;
            }
            tmp = original;
        } while (true);
        if (!IsValid(original.address)) {
            NoteAnyWrites();
        }
        return true;
    }

public:
    [[nodiscard]] static bool AnyWrites() {
        return any_writes.load(std::memory_order_seq_cst) ||
               any_gatherers.load(std::memory_order_seq_cst) != 0;
    }

    static void BeginGather() {
        any_gatherers.fetch_add(1, std::memory_order_seq_cst);
        any_writes.exchange(false, std::memory_order_seq_cst);
    }

    static void EndGather() {
        any_gatherers.fetch_sub(1, std::memory_order_seq_cst);
    }

    [[nodiscard]] u64 GatherEpoch() const noexcept {
        return gather_epoch.load(std::memory_order_acquire);
    }

    [[nodiscard]] const std::atomic<u64>* GatherEpochPtr() const noexcept {
        return &gather_epoch;
    }

    bool MayHaveWrites() const {
        return AnyCurrentValid() ||
               has_back.load(std::memory_order_acquire) ||
               has_front.load(std::memory_order_acquire);
    }

    /// Calls `callback` for each write collected and not yet applied, without consuming them: those
    /// still collecting and those a Gather is handing out.
    void ForEachPending(const std::function<void(PAddr, size_t)>& callback) {
        if (!MayHaveWrites()) {
            return;
        }
        std::vector<TransformAddress> pending;
        {
            std::scoped_lock gather_lk(gather_guard);
            std::scoped_lock lk(guard);
            pending = back_buffer;
            pending.insert(pending.end(), front_buffer.begin(), front_buffer.end());
            for (const auto& slot : current) {
                const TransformAddress t = slot.load(std::memory_order_acquire);
                if (IsValid(t.address)) {
                    pending.push_back(t);
                }
            }
        }
        for (const auto& transform : pending) {
            ForEachRun(transform, callback);
        }
    }

    void Gather(std::function<void(PAddr, size_t)>& callback) {
        // Nothing collected: back_buffer only grows (under guard) while a current slot holds a
        // valid address, and only Gather resets current.
        // Every fence gathers each core's manager; most find nothing.
        if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::DirtyGatherFast) &&
            !AnyCurrentValid() && !has_back.load(std::memory_order_acquire)) {
            return;
        }
        // FlushRegion also gathers off the GPU thread (host1x video decode reads); serialize
        // gatherers so one cannot swap or clear front_buffer while another iterates it.
        std::scoped_lock gather_lk(gather_guard);
        {
            std::scoped_lock lk(guard);
            gather_epoch.fetch_add(1, std::memory_order_seq_cst);
            front_buffer.swap(back_buffer);
            has_back.store(false, std::memory_order_relaxed);
            for (auto& slot : current) {
                const TransformAddress t =
                    slot.exchange(default_transform, std::memory_order_seq_cst);
                if (IsValid(t.address)) {
                    front_buffer.emplace_back(t);
                }
            }
            gather_epoch.fetch_add(1, std::memory_order_seq_cst);
            has_front.store(!front_buffer.empty(), std::memory_order_release);
        }
        for (auto& transform : front_buffer) {
            ForEachRun(transform, callback);
        }
        {
            std::scoped_lock lk(guard);
            front_buffer.clear();
            has_front.store(false, std::memory_order_release);
        }
    }

    /// Returns true when collected writes not yet gathered overlap [address, address + size).
    bool Overlaps(PAddr address, size_t size) {
        const PAddr end = address + size;
        const auto overlaps = [&](TransformAddress t) {
            if (!IsValid(t.address)) {
                return false;
            }
            const PAddr begin = static_cast<PAddr>(t.address) << page_bits;
            if (address >= begin + page_size || end <= begin) {
                return false;
            }
            const size_t minor_bit = address > begin ? (address - begin) >> align_bits : 0;
            const size_t top_bit = ((std::min)(end - begin, PAddr{page_size}) + align_mask) >> align_bits;
            return (t.mask & CreateMask<u32>(top_bit, minor_bit)) != 0;
        };
        std::scoped_lock lk(guard);
        for (const auto& slot : current) {
            if (overlaps(slot.load(std::memory_order_acquire))) {
                return true;
            }
        }
        for (const TransformAddress& t : back_buffer) {
            if (overlaps(t)) {
                return true;
            }
        }
        for (const TransformAddress& t : front_buffer) {
            if (overlaps(t)) {
                return true;
            }
        }
        return false;
    }

private:
    struct alignas(8) TransformAddress {
        u32 address;
        u32 mask;
    };

    template <typename Func>
    static void ForEachRun(const TransformAddress& transform, Func&& callback) {
        size_t offset = 0;
        u64 mask = transform.mask;
        while (mask != 0) {
            const size_t empty_bits = std::countr_zero(mask);
            offset += empty_bits << align_bits;
            mask = mask >> empty_bits;
            const size_t continuous_bits = std::countr_one(mask);
            callback((static_cast<PAddr>(transform.address) << page_bits) + offset,
                     continuous_bits << align_bits);
            mask = continuous_bits < align_size ? (mask >> continuous_bits) : 0;
            offset += continuous_bits << align_bits;
        }
    }

    constexpr static size_t page_bits = DEVICE_PAGEBITS - 1;
    constexpr static size_t page_size = 1ULL << page_bits;
    constexpr static size_t page_mask = page_size - 1;

    constexpr static size_t align_bits = 6U;
    constexpr static size_t align_size = 1U << align_bits;
    constexpr static size_t align_mask = align_size - 1;
    constexpr static size_t num_current_slots = 4U;
    constexpr static TransformAddress default_transform = {.address = ~0U, .mask = 0U};

    static constexpr bool IsValid(PAddr address) {
        return address < (1ULL << 39);
    }

    bool AnyCurrentValid() const noexcept {
        for (const auto& slot : current) {
            if (IsValid(slot.load(std::memory_order_acquire).address)) {
                return true;
            }
        }
        return false;
    }

    static void NoteAnyWrites() {
        if (!any_writes.load(std::memory_order_seq_cst)) {
            any_writes.store(true, std::memory_order_seq_cst);
        }
    }

    static inline std::atomic<bool> any_writes{true};
    static inline std::atomic<u32> any_gatherers{};

    template <typename T>
    T CreateMask(size_t top_bit, size_t minor_bit) {
        T mask = ~T(0);
        mask <<= (sizeof(T) * 8 - top_bit);
        mask >>= (sizeof(T) * 8 - top_bit);
        mask >>= minor_bit;
        mask <<= minor_bit;
        return mask;
    }

    TransformAddress BuildTransform(PAddr address, size_t size) {
        const size_t minor_address = address & page_mask;
        const size_t minor_bit = minor_address >> align_bits;
        const size_t top_bit = (minor_address + size + align_mask) >> align_bits;
        TransformAddress result{};
        result.address = static_cast<u32>(address >> page_bits);
        result.mask = CreateMask<u32>(top_bit, minor_bit);
        return result;
    }

    std::array<std::atomic<TransformAddress>, num_current_slots> current{};
    std::atomic<u64> gather_epoch{2};
    std::atomic<bool> has_back{};
    std::atomic<bool> has_front{};
    std::mutex guard;
    std::mutex gather_guard;
    std::vector<TransformAddress> back_buffer;
    std::vector<TransformAddress> front_buffer;
};

/// One bit per device page of the 36-bit device address space (2 MiB), set by writers and taken
/// by the native uniform bank check (BufferCache, EDEN_NX2_EXP9).
class DevicePageFlags {
public:
    static constexpr u64 AddressBits = 36;

    void Mark(u64 address, u64 size) noexcept {
        constexpr u64 limit = u64{1} << AddressBits;
        if (size == 0 || address >= limit) {
            return;
        }
        u64 page = address >> DEVICE_PAGEBITS;
        const u64 last = ((std::min)(address + size, limit) - 1) >> DEVICE_PAGEBITS;
        while (page <= last) {
            const u64 first_bit = page % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            std::atomic<u64>& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_seq_cst);
            }
            page += bits;
        }
    }

    /// Clears the bit of the page holding `address`, returning whether it was set.
    bool Take(u64 address) noexcept {
        if (address >= (u64{1} << AddressBits)) {
            return false;
        }
        const u64 page = address >> DEVICE_PAGEBITS;
        const u64 bit = u64{1} << (page % 64);
        std::atomic<u64>& word = words[page / 64];
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            return false;
        }
        return (word.fetch_and(~bit, std::memory_order_seq_cst) & bit) != 0;
    }

    /// Returns true if any page in [address, address + size) has its bit set.
    [[nodiscard]] bool Any(u64 address, u64 size) const noexcept {
        constexpr u64 limit = u64{1} << AddressBits;
        if (size == 0 || address >= limit) {
            return false;
        }
        u64 page = address >> DEVICE_PAGEBITS;
        const u64 last = ((std::min)(address + size, limit) - 1) >> DEVICE_PAGEBITS;
        while (page <= last) {
            const u64 first_bit = page % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            if ((words[page / 64].load(std::memory_order_relaxed) & mask) != 0) {
                return true;
            }
            page += bits;
        }
        return false;
    }

private:
    std::array<std::atomic<u64>, (u64{1} << (AddressBits - DEVICE_PAGEBITS)) / 64> words{};
};

/// Pages guest CPU writes went to through the dirty memory managers: the buffer cache sees them
/// only at the next InvalidateGPUCache.
inline DevicePageFlags g_collected_write_pages;
/// Pages whose guest memory the buffer cache saw change (WriteMemory, uploads in place,
/// downloads).
inline DevicePageFlags g_buffer_write_pages;
/// Bumped when the GPU's answer to a guest write of a collected page may have changed
/// (EDEN_NX2_CPU_WRITE_RECLAIM): each core then asks again instead of collecting at once.
inline std::atomic<u64> g_write_answer_epoch{};

} // namespace Core
