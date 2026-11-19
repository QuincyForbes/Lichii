// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* virtual_buffer.h */
// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"

namespace Common {

#ifdef _WIN32
constexpr u64 HostPageSize = 0x1000;
constexpr u64 HostPageBits = 12;
constexpr u64 HostPageMask = ~(HostPageSize - 1);
bool CommitVectorPage(uintptr_t addr, bool write) noexcept;
#else
inline const u64 HostPageSize = sysconf(_SC_PAGESIZE);
inline const u64 HostPageBits = std::countr_zero(HostPageSize);
inline const u64 HostPageMask = ~(HostPageSize - 1);
#endif

void* AllocateMemoryPages(std::size_t size) noexcept;
void FreeMemoryPages(void* base, std::size_t size) noexcept;

/// A large page-aligned buffer that has optimized memory usage for zero-writes.
template <typename T>
    // MSVC doesn't regard structs with atomics as trivially copyable
    // requires std::is_trivially_copyable_v<T>
class SparseLargeVector final {
public:
    constexpr SparseLargeVector() = default;

    explicit SparseLargeVector(std::size_t count) noexcept {
        Allocate(count * sizeof(T));
    }

    ~SparseLargeVector() noexcept {
        FreeMemoryPages(base_ptr, alloc_size);
    }

    SparseLargeVector(const SparseLargeVector&) = delete;
    SparseLargeVector& operator=(const SparseLargeVector&) = delete;
    SparseLargeVector(SparseLargeVector&& other) = delete;
    SparseLargeVector& operator=(SparseLargeVector&& other) = delete;

    void ResizeAndClear(std::size_t count) noexcept {
        if (const std::size_t new_size = count * sizeof(T); new_size != alloc_size) {
            FreeMemoryPages(base_ptr, alloc_size);
            Allocate(new_size);
        } else if (count != 0) {
            ZeroRegion(0, count);
        }
    }

    /// Returns a reference to the value of the requested index and allocates memory if needed.
    T& GetAndFault(std::size_t index) noexcept {
        if (index >= size()) {
            UNREACHABLE_MSG("Out of bounds RW access on SparseLargeVector @ {}", index);
        }

        if (!IsCommittedPage(index) && !CommitPage(index)) {
            UNREACHABLE_MSG("Cannot access SparseLargeVector index {} with RW permission", index);
        }
        return base_ptr[index];
    }

    /// Returns a reference to the value of the requested index if initialized, or will otherwise return a zero-initialized object.
    const T& GetOrDefault(std::size_t index) const noexcept {
#ifdef _WIN32
        if (!IsCommittedPage(index)) {
            return *reinterpret_cast<const T*>(default_val.data());
        }
#endif
        // On non-Windows, OS page table should optimize this by pointing to a zero page if unallocated.
        return base_ptr[index];
    }

    /// Returns a direct const pointer to [first, last] when every backing host page covering that
    /// inclusive element range is readable (always true in-bounds on POSIX; verified once per 64
    /// host pages on Windows), or nullptr otherwise.
    [[nodiscard]] const T* CommittedSpan(std::size_t first, std::size_t last) const noexcept {
        if (first > last || last >= size()) {
            return nullptr;
        }
#ifdef _WIN32
        const std::size_t first_page = (first * sizeof(T)) >> HostPageBits;
        const std::size_t last_page = (last * sizeof(T)) >> HostPageBits;
        for (std::size_t page = first_page; page <= last_page;) {
            const std::size_t bit = page & 63;
            const std::size_t count = (std::min<std::size_t>)(64 - bit, last_page - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if ((committed_pages[page >> 6].load(std::memory_order_acquire) & mask) != mask) {
                return nullptr;
            }
            page += count;
        }
#endif
        return base_ptr + first;
    }

    void Set(std::size_t index, const T& value) noexcept {
        if (index >= size()) {
            LOG_CRITICAL(Common_Memory, "Out of bounds write on SparseLargeVector @ {}", index);
            return;
        }
        if (IsCommittedPage(index) || CommitPage(index)) {
            base_ptr[index] = value;
        }
    }

    void ZeroRegion(std::size_t start, std::size_t end_) noexcept {
        end_ = (std::min)(end_, size());
        if (start >= end_) {
            return;
        }
        const std::size_t byte_begin = start * sizeof(T);
        const std::size_t byte_end = end_ * sizeof(T);
        const std::size_t first_page = byte_begin >> HostPageBits;
        const std::size_t last_page = (byte_end - 1) >> HostPageBits;
        u8* const raw_base = reinterpret_cast<u8*>(base_ptr);
        for (std::size_t page = first_page; page <= last_page;) {
            if (!IsPageCommitted(page)) {
                ++page;
                continue;
            }
            std::size_t run_end = page + 1;
            while (run_end <= last_page && IsPageCommitted(run_end)) {
                ++run_end;
            }
            const std::size_t clear_begin = (std::max)(byte_begin, page << HostPageBits);
            const std::size_t clear_end = (std::min)(byte_end, run_end << HostPageBits);
            std::memset(raw_base + clear_begin, 0, clear_end - clear_begin);
            page = run_end;
        }
    }

    void CommitRegion(std::size_t index, std::size_t end_) noexcept {
        end_ = (std::min)(end_, size());
        if (index >= end_) {
            return;
        }
        const std::size_t first_page = (index * sizeof(T)) >> HostPageBits;
        const std::size_t last_page = (end_ * sizeof(T) - 1) >> HostPageBits;
        for (std::size_t page = first_page; page <= last_page;) {
            if (IsPageCommitted(page)) {
                ++page;
                continue;
            }
            std::size_t run_end = page + 1;
            while (run_end <= last_page && !IsPageCommitted(run_end)) {
                ++run_end;
            }
#ifdef _WIN32
            for (std::size_t p = page; p < run_end; ++p) {
                CommitHostPage(p);
            }
#else
            void* const addr = reinterpret_cast<u8*>(base_ptr) + (page << HostPageBits);
            const std::size_t bytes = (run_end - page) << HostPageBits;
            if (mprotect(addr, bytes, PROT_READ | PROT_WRITE) != 0) {
                LOG_ERROR(Common_Memory, "Failed to commit large buffer region at page {}, error {}",
                          page, strerror(errno));
                return;
            }
            for (std::size_t p = page; p < run_end;) {
                const std::size_t bit = p & 63;
                const std::size_t n = (std::min<std::size_t>)(64 - bit, run_end - p);
                const u64 mask = (n == 64 ? ~u64{0} : (u64{1} << n) - 1) << bit;
                committed_pages[p >> 6].fetch_or(mask, std::memory_order_release);
                p += n;
            }
#endif
            page = run_end;
        }
    }

    constexpr T& GetUnchecked(std::size_t index) noexcept {
        return base_ptr[index];
    }

    [[nodiscard]] const T& operator[](std::size_t index) const noexcept {
        return GetOrDefault(index);
    }

    [[nodiscard]] constexpr const T* data() const noexcept {
        return base_ptr;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return alloc_size / sizeof(T);
    }

    /// Host bytes committed for writing so far.
    [[nodiscard]] std::size_t CommittedBytes() const noexcept {
        std::size_t pages = 0;
        for (const auto& word : committed_pages) {
            pages += std::popcount(word.load(std::memory_order_relaxed));
        }
        return pages * HostPageSize;
    }

private:
    void Allocate(std::size_t new_size) noexcept {
        alloc_size = new_size;
        if (alloc_size == 0) {
            base_ptr = nullptr;
            committed_pages.clear();
            return;
        }
        base_ptr = static_cast<T*>(AllocateMemoryPages(alloc_size));
        // Each atomic u64 holds commitment bits for 64 host pages.
        const std::size_t denom = HostPageSize * 64;
        committed_pages = std::vector<std::atomic<u64>>((alloc_size + denom - 1) / denom);
    }

    [[nodiscard]] bool IsPageCommitted(std::size_t page_index) const noexcept {
        const u64 val = committed_pages[page_index >> 6].load(std::memory_order_acquire);
        return ((val >> (page_index & 63)) & 1) != 0;
    }

    [[nodiscard]] bool IsCommittedPage(std::size_t index) const noexcept {
        if (index >= size()) {
            LOG_CRITICAL(Common_Memory, "Out of bounds access on large vector @ {}", index);
            return false;
        }
        return IsPageCommitted((index * sizeof(T)) >> HostPageBits);
    }

    bool CommitHostPage(std::size_t page_index) noexcept {
        const uintptr_t page =
            reinterpret_cast<uintptr_t>(base_ptr) + (page_index << HostPageBits);
#ifdef _WIN32
        if (!CommitVectorPage(page, true)) {
            return false;
        }
#else
        if (mprotect(reinterpret_cast<void*>(page), HostPageSize, PROT_READ | PROT_WRITE) != 0) {
            LOG_ERROR(Common_Memory, "Failed to commit large buffer page {}, error {}", page_index,
                      strerror(errno));
            return false;
        }
#endif
        committed_pages[page_index >> 6].fetch_or(1ULL << (page_index & 63),
                                                  std::memory_order_release);
        return true;
    }

    bool CommitPage(std::size_t index) noexcept {
        return CommitHostPage((index * sizeof(T)) >> HostPageBits);
    }

    std::size_t alloc_size{};
    T* base_ptr{};

    std::vector<std::atomic<u64>> committed_pages{};
#ifdef _WIN32
    alignas(T) const std::array<u8, sizeof(T)> default_val{};
#endif
};

} // namespace Common
