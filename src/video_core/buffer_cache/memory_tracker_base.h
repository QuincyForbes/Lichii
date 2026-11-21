// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <deque>
#include <limits>
#include <type_traits>
#include "common/container/unordered_set.h"
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "video_core/buffer_cache/word_manager.h"

namespace VideoCommon {

template <typename DeviceTracker>
class MemoryTrackerBase {
    static constexpr size_t MAX_CPU_PAGE_BITS = 36; // device addresses (MaxwellDeviceTraits)
    static constexpr size_t HIGHER_PAGE_BITS = 22;
    static constexpr size_t HIGHER_PAGE_SIZE = 1ULL << HIGHER_PAGE_BITS;
    static constexpr size_t HIGHER_PAGE_MASK = HIGHER_PAGE_SIZE - 1ULL;
    static constexpr size_t NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - HIGHER_PAGE_BITS);
    static constexpr size_t MANAGER_POOL_SIZE = 32;
    static constexpr size_t WORDS_STACK_NEEDED = HIGHER_PAGE_SIZE / BYTES_PER_WORD;
    using Manager = WordManager<DeviceTracker, WORDS_STACK_NEEDED, HIGHER_PAGE_SIZE>;

public:
    MemoryTrackerBase(DeviceTracker& device_tracker_) : device_tracker{&device_tracker_} {}
    ~MemoryTrackerBase() = default;

    /// Returns the inclusive CPU modified range in a begin end pair
    [[nodiscard]] std::pair<u64, u64> ModifiedCpuRegion(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePairs<true>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->ModifiedRegion(Type::CPU, offset, size);
        });
    }

    /// Returns the inclusive GPU modified range in a begin end pair
    [[nodiscard]] std::pair<u64, u64> ModifiedGpuRegion(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePairs<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->ModifiedRegion(Type::GPU, offset, size);
        });
    }

    /// Returns true if a region has been modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) const noexcept {
        if (query_size == 0) {
            return false;
        }
        std::size_t page_index{query_cpu_addr >> HIGHER_PAGE_BITS};
        u64 page_offset{query_cpu_addr & HIGHER_PAGE_MASK};
        u64 remaining{query_size};
        while (remaining > 0) {
            const u64 amount = (std::min<u64>)(HIGHER_PAGE_SIZE - page_offset, remaining);
            const Manager* const manager = top_tier[page_index];
            if (manager == nullptr || manager->IsRegionModified(Type::CPU, page_offset, amount)) {
                return true;
            }
            ++page_index;
            page_offset = 0;
            remaining -= amount;
        }
        return false;
    }

    /// True when MarkRegionAsCpuModified(query_cpu_addr, query_size) would change no tracker or
    /// protection state (including chunks with no manager yet, which start all CPU-modified and
    /// untracked).
    [[nodiscard]] bool IsRegionCpuModifiedNoOp(VAddr query_cpu_addr, u64 query_size) const noexcept {
        if (query_size == 0) {
            return true;
        }
        std::size_t page_index{query_cpu_addr >> HIGHER_PAGE_BITS};
        u64 page_offset{query_cpu_addr & HIGHER_PAGE_MASK};
        u64 remaining{query_size};
        while (remaining > 0) {
            const u64 amount = (std::min<u64>)(HIGHER_PAGE_SIZE - page_offset, remaining);
            const Manager* const manager = top_tier[page_index];
            if (manager != nullptr && !manager->IsRangeCpuModifiedNoOp(page_offset, amount)) {
                return false;
            }
            ++page_index;
            page_offset = 0;
            remaining -= amount;
        }
        return true;
    }

    /// Returns true if every page of a region is CPU-modified and none is GPU-modified or
    /// Preflushable. False where the tracker has no manager yet (nothing to prove it).
    [[nodiscard]] bool IsRegionWhollyCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        if (query_size == 0) {
            return false;
        }
        const size_t first = query_cpu_addr >> HIGHER_PAGE_BITS;
        const size_t last = (query_cpu_addr + query_size - 1) >> HIGHER_PAGE_BITS;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            if (top_tier[page_index] == nullptr) {
                return false;
            }
        }
        return !IteratePages<false>(query_cpu_addr, query_size,
                                    [](Manager* manager, u64 offset, size_t size) {
                                        return !manager->IsRegionWhollyCpuModified(offset, size);
                                    });
    }

    /// Returns true if a region has been modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        const u64 local = query_cpu_addr & HIGHER_PAGE_MASK;
        if (query_size != 0 && local + query_size <= HIGHER_PAGE_SIZE &&
            local / BYTES_PER_WORD == (local + query_size - 1) / BYTES_PER_WORD) {
            const Manager* const manager = top_tier[query_cpu_addr >> HIGHER_PAGE_BITS];
            return manager != nullptr &&
                   manager->IsWordRangeGpuModified(local, local + query_size - 1);
        }
        return IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::GPU, offset, size);
        });
    }

    [[nodiscard]] bool IsRegionFullyCpuModified(VAddr query_cpu_addr, u64 query_size) const noexcept {
        if (query_size == 0) {
            return false;
        }
        std::size_t page_index{query_cpu_addr >> HIGHER_PAGE_BITS};
        u64 page_offset{query_cpu_addr & HIGHER_PAGE_MASK};
        u64 remaining{query_size};
        while (remaining > 0) {
            const u64 amount = (std::min<u64>)(HIGHER_PAGE_SIZE - page_offset, remaining);
            const Manager* const manager = top_tier[page_index];
            if (manager == nullptr || !manager->IsRangeFullyCpuModified(page_offset, amount)) {
                return false;
            }
            ++page_index;
            page_offset = 0;
            remaining -= amount;
        }
        return true;
    }

    [[nodiscard]] bool IsRegionUploadDirty(VAddr query_cpu_addr, u64 query_size) const noexcept {
        std::size_t page_index{query_cpu_addr >> HIGHER_PAGE_BITS};
        u64 page_offset{query_cpu_addr & HIGHER_PAGE_MASK};
        u64 remaining{query_size};
        while (remaining > 0) {
            const u64 amount = (std::min<u64>)(HIGHER_PAGE_SIZE - page_offset, remaining);
            const Manager* const manager = top_tier[page_index];
            if (manager == nullptr || manager->IsRangeUploadDirty(page_offset, amount)) {
                return true;
            }
            ++page_index;
            page_offset = 0;
            remaining -= amount;
        }
        return false;
    }

    /// Returns true if a region has been marked as Preflushable
    [[nodiscard]] bool IsRegionPreflushable(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::Preflushable, offset, size);
        });
    }

    /// Nx2Perf bind_memo: bumped whenever CPU-modified bits may be set (cpu) and whenever
    /// GPU-modified bits change (gpu), tracked per 4 MiB chunk so unrelated writes don't
    /// invalidate every bound buffer's memo.
    [[nodiscard]] u64 CpuEpoch() const noexcept {
        return cpu_epoch;
    }
    [[nodiscard]] u64 GpuEpoch() const noexcept {
        return gpu_epoch;
    }
    [[nodiscard]] u64 CpuRangeEpoch(VAddr addr, u64 size) const noexcept {
        const size_t first = addr >> HIGHER_PAGE_BITS;
        const size_t last = (addr + (std::max<u64>)(size, 1) - 1) >> HIGHER_PAGE_BITS;
        if (last < first || last - first >= MAX_RANGE_EPOCH_CHUNKS) {
            return (u64{1} << 63) | cpu_epoch;
        }
        u64 sum = structural_cpu_epoch;
        for (size_t page = first; page <= last; ++page) {
            sum += cpu_chunk_epochs[page & (CHUNK_EPOCH_SLOTS - 1)];
        }
        return sum & ~(u64{1} << 63);
    }
    [[nodiscard]] u64 GpuRangeEpoch(VAddr addr, u64 size) const noexcept {
        const size_t first = addr >> HIGHER_PAGE_BITS;
        const size_t last = (addr + (std::max<u64>)(size, 1) - 1) >> HIGHER_PAGE_BITS;
        if (last < first || last - first >= MAX_RANGE_EPOCH_CHUNKS) {
            return (u64{1} << 63) | gpu_epoch;
        }
        u64 sum = structural_gpu_epoch;
        for (size_t page = first; page <= last; ++page) {
            sum += gpu_chunk_epochs[page & (CHUNK_EPOCH_SLOTS - 1)];
        }
        return sum & ~(u64{1} << 63);
    }
    void BumpGpuEpoch() noexcept {
        ++gpu_epoch;
        ++structural_gpu_epoch;
    }
    void BumpGpuRangeEpoch(VAddr addr, u64 size) noexcept {
        ++gpu_epoch;
        const size_t first = addr >> HIGHER_PAGE_BITS;
        const size_t last = (addr + (std::max<u64>)(size, 1) - 1) >> HIGHER_PAGE_BITS;
        if (last < first || last - first >= CHUNK_EPOCH_SLOTS) {
            ++structural_gpu_epoch;
            return;
        }
        for (size_t page = first; page <= last; ++page) {
            ++gpu_chunk_epochs[page & (CHUNK_EPOCH_SLOTS - 1)];
        }
    }
    void BumpCpuRangeEpoch(VAddr addr, u64 size) noexcept {
        ++cpu_epoch;
        const size_t first = addr >> HIGHER_PAGE_BITS;
        const size_t last = (addr + (std::max<u64>)(size, 1) - 1) >> HIGHER_PAGE_BITS;
        if (last < first || last - first >= CHUNK_EPOCH_SLOTS) {
            ++structural_cpu_epoch;
            return;
        }
        for (size_t page = first; page <= last; ++page) {
            ++cpu_chunk_epochs[page & (CHUNK_EPOCH_SLOTS - 1)];
        }
    }

    /// Mark region as CPU modified, notifying the device_tracker about this change
    /// @param written False when no guest write is behind the mark (it never turns a page hot)
    void MarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size, bool written = true) {
        BumpCpuRangeEpoch(dirty_cpu_addr, query_size);
        IteratePages<true>(dirty_cpu_addr, query_size,
                           [this, written](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::CPU, true, manager->cpu_addr + offset, size, written);
            TrackHot(manager);
        });
    }

    /// Whether a page of the region is hot (EDEN_NX2_HOT_PAGES): the guest writes it without the
    /// cache being told, so nothing may remember its contents by the writes it has seen.
    [[nodiscard]] bool IsRegionHot(VAddr query_cpu_addr, u64 query_size) const noexcept {
        if (HotPages::g_mode == 0) {
            return false;
        }
        std::size_t page_index{query_cpu_addr >> HIGHER_PAGE_BITS};
        u64 page_offset{query_cpu_addr & HIGHER_PAGE_MASK};
        u64 remaining{query_size};
        while (remaining > 0) {
            const u64 amount = (std::min<u64>)(HIGHER_PAGE_SIZE - page_offset, remaining);
            const Manager* const manager = top_tier[page_index];
            if (manager != nullptr && manager->window_used &&
                manager->IsRangeHot(page_offset, amount)) {
                return true;
            }
            ++page_index;
            page_offset = 0;
            remaining -= amount;
        }
        return false;
    }

    /// Ends an epoch of EDEN_NX2_HOT_PAGES: the hot pages uploaded in it read as CPU-modified
    /// again, and func(address, size) is called for each run of them. With new_frame the frame
    /// count of the window advances; at its end every hot page cools down.
    template <typename Func>
    void AdvanceHotPages(bool new_frame, Func&& func) {
        if (!epoch_managers.empty()) {
            for (Manager* const manager : epoch_managers) {
                manager->epoch_listed = false;
                ++cpu_chunk_epochs[(manager->cpu_addr >> HIGHER_PAGE_BITS) &
                                   (CHUNK_EPOCH_SLOTS - 1)];
                manager->ReleaseHotClean(func);
            }
            epoch_managers.clear();
            // Released pages read CPU-modified again: bind memos keyed on the epoch must
            // re-synchronize them (a memo hit skipped the upload of a rewritten hot page).
            ++cpu_epoch;
            HotPages::Add(HotPages::g_counters.epochs, 1);
        }
        if (!new_frame || ++window_frames < HotPages::WindowFrames) {
            return;
        }
        window_frames = 0;
        for (Manager* const manager : window_managers) {
            manager->window_listed = false;
            manager->ResetHotWindow();
        }
        window_managers.clear();
    }

    /// Calls func(address) for each hot page of the region uploaded in this epoch.
    template <typename Func>
    void ForEachHotCleanPage(VAddr query_cpu_addr, u64 query_size, Func&& func) {
        IteratePages<false>(query_cpu_addr, query_size,
                            [&func](Manager* manager, u64 offset, size_t size) {
            if (manager->hot_clean_set != 0) {
                manager->ForEachHotCleanPage(offset, size, func);
            }
        });
    }

    /// Unmark region as CPU modified, notifying the device_tracker about this change
    void UnmarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::CPU, false, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the host GPU
    /// @return True when a hot page uploaded in this epoch (EDEN_NX2_HOT_PAGES) lost that state:
    ///         it reads as CPU-modified again although no guest write was reported
    bool MarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        BumpGpuRangeEpoch(dirty_cpu_addr, query_size);
        bool hot_page_dirtied = false;
        IteratePages<true>(dirty_cpu_addr, query_size,
                           [&hot_page_dirtied](Manager* manager, u64 offset, size_t size) {
            hot_page_dirtied |=
                manager->ChangeRegionState(Type::GPU, true, manager->cpu_addr + offset, size);
        });
        if (hot_page_dirtied) {
            BumpCpuRangeEpoch(dirty_cpu_addr, query_size);
        }
        return hot_page_dirtied;
    }

    /// Mark region as modified from the host GPU
    void MarkRegionAsPreflushable(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::Preflushable, true, manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        BumpGpuRangeEpoch(dirty_cpu_addr, query_size);
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::GPU, false, manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsPreflushable(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::Preflushable, false, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the CPU
    /// but don't mark it as modified until FlusHCachedWrites is called.
    void CachedCpuWrite(VAddr dirty_cpu_addr, u64 query_size) {
        bool hot_page_dirtied = false;
        IteratePages<true>(dirty_cpu_addr, query_size,
                           [this, &hot_page_dirtied](Manager* manager, u64 offset, size_t size) {
            const VAddr cpu_address = manager->cpu_addr + offset;
            hot_page_dirtied |= manager->ChangeRegionState(Type::CachedCPU, true, cpu_address, size);
            cached_pages.insert(u32(cpu_address >> HIGHER_PAGE_BITS));
        });
        if (hot_page_dirtied) {
            BumpCpuRangeEpoch(dirty_cpu_addr, query_size);
        }
    }

    /// Flushes cached CPU writes, and notify the device_tracker about the deltas
    void FlushCachedWrites(VAddr query_cpu_addr, u64 query_size) noexcept {
        BumpCpuRangeEpoch(query_cpu_addr, query_size);
        IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, [[maybe_unused]] u64 offset, [[maybe_unused]] size_t size) {
            manager->FlushCachedWrites();
        });
    }

    void FlushCachedWrites() noexcept {
        ++cpu_epoch;
        for (auto id : cached_pages) {
            ++cpu_chunk_epochs[id & (CHUNK_EPOCH_SLOTS - 1)];
            top_tier[id]->FlushCachedWrites();
        }
        cached_pages.clear();
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    template <typename Func>
    void ForEachUploadRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<true>(query_cpu_range, query_size, [this, &func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::CPU, true, manager->cpu_addr + offset, size, func);
            TrackHot(manager);
        });
    }

    /// Call 'func' for each range marked as Preflushable
    template <typename Func>
    void ForEachPreflushableRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<false>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::Preflushable, false, manager->cpu_addr + offset, size, func);
        });
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <typename Func>
    void ForEachDownloadRange(VAddr query_cpu_range, u64 query_size, bool clear, Func&& func) {
        if (clear) {
            BumpGpuRangeEpoch(query_cpu_range, query_size);
        }
        IteratePages<false>(query_cpu_range, query_size, [&func, clear](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::GPU, clear, manager->cpu_addr + offset, size, func);
        });
    }

    template <typename Func>
    void ForEachDownloadRangeAndClear(VAddr query_cpu_range, u64 query_size, Func&& func) {
        BumpGpuRangeEpoch(query_cpu_range, query_size);
        IteratePages<false>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::GPU, true, manager->cpu_addr + offset, size, func);
        });
    }

    /// ForEachUploadRange without the clean words (word_manager.h) at either end of each manager's
    /// part of the range: they have nothing to find or change. A per-word form of PR #161's skip.
    template <typename Func>
    void ForEachDirtyUploadRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<true>(query_cpu_range, query_size, [this, &func](Manager* manager, u64 offset, size_t size) {
            const u64 dirty =
                manager->DirtyWords(offset / BYTES_PER_WORD, (offset + size - 1) / BYTES_PER_WORD);
            if (dirty == 0) {
                return;
            }
            const u64 begin = (std::max<u64>)(offset, std::countr_zero(dirty) * BYTES_PER_WORD);
            const u64 end =
                (std::min<u64>)(offset + size, (64 - std::countl_zero(dirty)) * BYTES_PER_WORD);
            manager->ForEachModifiedRange(Type::CPU, true, manager->cpu_addr + begin,
                                          static_cast<s64>(end - begin), func);
            TrackHot(manager);
        });
    }
    /// Verify mode: true when the clean words of the managers over a range match a recount.
    [[nodiscard]] bool CleanWordsHold(VAddr query_cpu_addr, u64 query_size) const noexcept {
        const size_t first = query_cpu_addr >> HIGHER_PAGE_BITS;
        const size_t last = (query_cpu_addr + (std::max<u64>)(query_size, 1) - 1) >> HIGHER_PAGE_BITS;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            if (top_tier[page_index] && !top_tier[page_index]->CleanWordsHold()) {
                return false;
            }
        }
        return true;
    }

private:
    template <bool create_region_on_fail, typename Func>
    bool IteratePages(VAddr cpu_address, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, Manager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & HIGHER_PAGE_MASK};
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            if (manager) {
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    template <bool create_region_on_fail, typename Func>
    std::pair<u64, u64> IteratePairs(VAddr cpu_address, size_t size, Func&& func) {
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & HIGHER_PAGE_MASK};
        u64 begin = (std::numeric_limits<u64>::max)();
        u64 end = 0;
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            const auto execute = [&] {
                auto [new_begin, new_end] = func(manager, page_offset, copy_amount);
                if (new_begin != 0 || new_end != 0) {
                    const u64 base_address = page_index << HIGHER_PAGE_BITS;
                    begin = (std::min)(new_begin + base_address, begin);
                    end = (std::max)(new_end + base_address, end);
                }
            };
            if (manager) {
                execute();
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                execute();
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        if (begin < end) {
            return std::make_pair(begin, end);
        } else {
            return std::make_pair(0ULL, 0ULL);
        }
    }

    /// Lists a manager for AdvanceHotPages once it has something to release or to reset. Both
    /// stay false while EDEN_NX2_HOT_PAGES is off.
    void TrackHot(Manager* manager) {
        if (manager->hot_clean_set != 0 && !manager->epoch_listed) {
            manager->epoch_listed = true;
            epoch_managers.push_back(manager);
        }
        if (manager->window_used && !manager->window_listed) {
            manager->window_listed = true;
            window_managers.push_back(manager);
        }
    }

    void CreateRegion(std::size_t page_index) {
        ++cpu_epoch; // A new manager starts with every page CPU-modified
        ++cpu_chunk_epochs[page_index & (CHUNK_EPOCH_SLOTS - 1)];
        const VAddr base_cpu_addr = page_index << HIGHER_PAGE_BITS;
        top_tier[page_index] = GetNewManager(base_cpu_addr);
    }

    Manager* GetNewManager(VAddr base_cpu_address) {
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                new (&last_pool[i]) Manager(0, *device_tracker);
                free_managers.push_back(&last_pool[i]);
            }
        }
        Manager* new_manager = free_managers.front();
        new_manager->cpu_addr = base_cpu_address;
        free_managers.pop_front();
        return new_manager;
    }

    static constexpr size_t CHUNK_EPOCH_SLOTS = 1024;
    static constexpr size_t MAX_RANGE_EPOCH_CHUNKS = 8;

    std::array<Manager*, NUM_HIGH_PAGES> top_tier{};
    std::deque<std::array<Manager, MANAGER_POOL_SIZE>> manager_pool;
    std::deque<Manager*> free_managers;
    ::Common::unordered_set<u32> cached_pages;
    /// EDEN_NX2_HOT_PAGES: managers with a hot page uploaded in this epoch, managers with a page
    /// counted or hot in this window, and the frames of the window so far.
    std::vector<Manager*> epoch_managers;
    std::vector<Manager*> window_managers;
    u32 window_frames = 0;
    DeviceTracker* device_tracker = nullptr;
    u64 cpu_epoch{};
    u64 gpu_epoch{};
    u64 structural_cpu_epoch{};
    u64 structural_gpu_epoch{};
    std::array<u64, CHUNK_EPOCH_SLOTS> cpu_chunk_epochs{};
    std::array<u64, CHUNK_EPOCH_SLOTS> gpu_chunk_epochs{};
};

} // namespace VideoCommon
