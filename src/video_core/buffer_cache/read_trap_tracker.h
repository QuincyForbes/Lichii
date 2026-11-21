// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <memory>

#include "common/common_types.h"
#include "common/range_sets.h"
#include "common/range_sets.inc"

namespace VideoCommon {

/// Lock-free two-level bitmap over 4 KiB device pages (covering up to 512 GiB of DAddr space in
/// 2 KiB chunks allocated on demand).
class DevicePageBitmap {
public:
    static constexpr u64 PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;
    static constexpr u64 WordsPerChunk = 256; // 16,384 pages = 64 MiB per 2 KiB chunk
    static constexpr u64 PagesPerChunk = WordsPerChunk * 64;
    static constexpr size_t NumChunks = 8192; // 512 GiB address reach

    DevicePageBitmap() = default;

    ~DevicePageBitmap() {
        for (auto& slot : chunks) {
            delete slot.load(std::memory_order_relaxed);
        }
    }

    DevicePageBitmap(const DevicePageBitmap&) = delete;
    DevicePageBitmap& operator=(const DevicePageBitmap&) = delete;

    void Mark(DAddr address, u64 size) {
        if (size == 0) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last < first || last >= NumChunks * PagesPerChunk) {
            return;
        }
        for (u64 page = first; page <= last;) {
            Chunk* const chunk = EnsureChunk(static_cast<size_t>(page / PagesPerChunk));
            const u64 in_chunk = page % PagesPerChunk;
            const size_t word_idx = static_cast<size_t>(in_chunk / 64);
            const u64 bit = in_chunk % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = chunk->words[word_idx];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_seq_cst);
            }
            page += count;
        }
    }

    void Clear(DAddr address, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last < first || first >= NumChunks * PagesPerChunk) {
            return;
        }
        const u64 clamped_last = (std::min<u64>)(last, NumChunks * PagesPerChunk - 1);
        for (u64 page = first; page <= clamped_last;) {
            const size_t chunk_idx = static_cast<size_t>(page / PagesPerChunk);
            Chunk* const chunk = chunks[chunk_idx].load(std::memory_order_acquire);
            if (chunk == nullptr) {
                page = (u64{chunk_idx} + 1) * PagesPerChunk;
                continue;
            }
            const u64 in_chunk = page % PagesPerChunk;
            const size_t word_idx = static_cast<size_t>(in_chunk / 64);
            const u64 bit = in_chunk % 64;
            const u64 count = (std::min<u64>)(64 - bit, clamped_last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = chunk->words[word_idx];
            if ((word.load(std::memory_order_relaxed) & mask) != 0) {
                word.fetch_and(~mask, std::memory_order_seq_cst);
            }
            page += count;
        }
    }

    void ClearAll() noexcept {
        for (auto& slot : chunks) {
            if (Chunk* const chunk = slot.load(std::memory_order_acquire); chunk != nullptr) {
                for (auto& word : chunk->words) {
                    word.store(0, std::memory_order_relaxed);
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    [[nodiscard]] bool Any(DAddr address, u64 size,
                           std::memory_order order = std::memory_order_acquire) const noexcept {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last < first || last >= NumChunks * PagesPerChunk) {
            return true;
        }
        for (u64 page = first; page <= last;) {
            const size_t chunk_idx = static_cast<size_t>(page / PagesPerChunk);
            const Chunk* const chunk = chunks[chunk_idx].load(order);
            if (chunk == nullptr) {
                page = (u64{chunk_idx} + 1) * PagesPerChunk;
                continue;
            }
            const u64 in_chunk = page % PagesPerChunk;
            const size_t word_idx = static_cast<size_t>(in_chunk / 64);
            const u64 bit = in_chunk % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if ((chunk->words[word_idx].load(order) & mask) != 0) {
                return true;
            }
            page += count;
        }
        return false;
    }

    [[nodiscard]] bool AnySeqCst(DAddr address, u64 size) const noexcept {
        return Any(address, size, std::memory_order_seq_cst);
    }

    [[nodiscard]] bool All(DAddr address, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last < first || last >= NumChunks * PagesPerChunk) {
            return false;
        }
        for (u64 page = first; page <= last;) {
            const size_t chunk_idx = static_cast<size_t>(page / PagesPerChunk);
            const Chunk* const chunk = chunks[chunk_idx].load(std::memory_order_acquire);
            if (chunk == nullptr) {
                return false;
            }
            const u64 in_chunk = page % PagesPerChunk;
            const size_t word_idx = static_cast<size_t>(in_chunk / 64);
            const u64 bit = in_chunk % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if ((chunk->words[word_idx].load(std::memory_order_relaxed) & mask) != mask) {
                return false;
            }
            page += count;
        }
        return true;
    }

    /// Atomically sets unarmed page bits in [address, address + size) and invokes
    /// `on_new_run(run_address, run_bytes)` for each contiguous run of newly set pages.
    template <typename Func>
    bool ArmRuns(DAddr address, u64 size, Func&& on_new_run) {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last < first || last >= NumChunks * PagesPerChunk) {
            on_new_run(address, size);
            return true;
        }
        bool any = false;
        u64 run_begin = 0;
        u64 run_pages = 0;
        const auto flush_run = [&] {
            if (run_pages != 0) {
                on_new_run(run_begin << PageBits, run_pages << PageBits);
                run_pages = 0;
                any = true;
            }
        };
        for (u64 page = first; page <= last;) {
            Chunk* const chunk = EnsureChunk(static_cast<size_t>(page / PagesPerChunk));
            const u64 in_chunk = page % PagesPerChunk;
            const size_t word_idx = static_cast<size_t>(in_chunk / 64);
            const u64 bit = in_chunk % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            const u64 old_word =
                chunk->words[word_idx].fetch_or(mask, std::memory_order_seq_cst);
            u64 newly_set = ~old_word & mask;
            const u64 word_base_page = page - bit;
            if (newly_set == 0) {
                flush_run();
            } else if (newly_set == mask) {
                if (run_pages == 0) {
                    run_begin = page;
                } else if (run_begin + run_pages != page) {
                    flush_run();
                    run_begin = page;
                }
                run_pages += count;
            } else {
                while (newly_set != 0) {
                    const u64 low = static_cast<u64>(std::countr_zero(newly_set));
                    const u64 len = static_cast<u64>(std::countr_one(newly_set >> low));
                    const u64 sub_page = word_base_page + low;
                    if (run_pages != 0 && run_begin + run_pages == sub_page) {
                        run_pages += len;
                    } else {
                        flush_run();
                        run_begin = sub_page;
                        run_pages = len;
                    }
                    newly_set = (low + len >= 64) ? 0 : (newly_set & ~(((u64{1} << len) - 1) << low));
                }
            }
            page += count;
        }
        flush_run();
        return any;
    }

private:
    struct Chunk {
        std::array<std::atomic<u64>, WordsPerChunk> words{};
    };

    Chunk* EnsureChunk(size_t index) {
        Chunk* chunk = chunks[index].load(std::memory_order_acquire);
        if (chunk != nullptr) {
            return chunk;
        }
        auto fresh = std::make_unique<Chunk>();
        if (chunks[index].compare_exchange_strong(chunk, fresh.get(), std::memory_order_seq_cst,
                                                  std::memory_order_seq_cst)) {
            return fresh.release();
        }
        return chunk;
    }

    std::array<std::atomic<Chunk*>, NumChunks> chunks{};
};

class ReadTrapTracker {
public:
    /// Arms this page-aligned range. Calls `arm(address, size)` for each part that was not armed
    /// yet, the parts the core must be notified of, and returns whether there was any.
    template <typename Func>
    bool Arm(DAddr address, u64 size, Func&& arm) {
        return pages.ArmRuns(address, size, std::forward<Func>(arm));
    }

    /// Returns whether the core must be notified to arm this page-aligned range.
    bool Arm(DAddr address, u64 size) {
        return Arm(address, size, [](DAddr, u64) {});
    }

    /// A CPU read consumes the core's trap; unmapping also forgets the range. Thread-safe.
    void Subtract(DAddr address, u64 size) noexcept {
        pages.Clear(address, size);
    }

private:
    DevicePageBitmap pages;
};

} // namespace VideoCommon
