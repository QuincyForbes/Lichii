// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "core/hle/kernel/k_page_bitmap.h"
#include "core/hle/kernel/k_spin_lock.h"
#include "core/hle/kernel/k_typed_address.h"
#include "core/hle/kernel/memory_types.h"
#include "core/hle/kernel/svc_results.h"

namespace Kernel {

class KDynamicPageManager {
public:
    class PageBuffer {
    private:
        u8 m_buffer[PageSize];
    };
    static_assert(sizeof(PageBuffer) == PageSize);

private:
    static constexpr size_t SlabPages = Common::BitSize<u64>();
    struct Slab {
        alignas(PageSize) std::array<PageBuffer, SlabPages> pages{};
    };

    PageBuffer* EnsurePage(size_t index) const {
        const size_t slab_idx = index / SlabPages;
        auto& slab = m_slabs[slab_idx];
        if (!slab) [[unlikely]] {
            slab = std::make_unique<Slab>();
        }
        return std::addressof(slab->pages[index % SlabPages]);
    }

public:
    KDynamicPageManager() = default;

    template <typename T>
    T* GetPointer(KVirtualAddress addr) {
        const size_t byte_offset = GetInteger(addr) - GetInteger(m_aligned_address);
        auto* page = EnsurePage(byte_offset / sizeof(PageBuffer));
        return reinterpret_cast<T*>(reinterpret_cast<u8*>(page) +
                                    (byte_offset % sizeof(PageBuffer)));
    }

    template <typename T>
    const T* GetPointer(KVirtualAddress addr) const {
        const size_t byte_offset = GetInteger(addr) - GetInteger(m_aligned_address);
        const auto* page = EnsurePage(byte_offset / sizeof(PageBuffer));
        return reinterpret_cast<const T*>(reinterpret_cast<const u8*>(page) +
                                          (byte_offset % sizeof(PageBuffer)));
    }

    Result Initialize(KVirtualAddress memory, size_t size, size_t align) {
        // We need to have positive size.
        R_UNLESS(size > 0, ResultOutOfMemory);

        // Set addresses.
        m_address = memory;
        m_aligned_address = Common::AlignDown(GetInteger(memory), align);

        // Calculate extents.
        const size_t managed_size = m_address + size - m_aligned_address;
        const size_t overhead_size = Common::AlignUp(
            KPageBitmap::CalculateManagementOverheadSize(managed_size / sizeof(PageBuffer)),
            sizeof(PageBuffer));
        R_UNLESS(overhead_size < size, ResultOutOfMemory);

        // Set tracking fields.
        m_size = Common::AlignDown(size - overhead_size, sizeof(PageBuffer));
        m_count = m_size / sizeof(PageBuffer);

        // Initialize the bitmap in compact host storage; backing slabs allocate lazily on demand.
        const size_t allocatable_region_size =
            (m_address + size - overhead_size) - m_aligned_address;
        ASSERT(allocatable_region_size >= sizeof(PageBuffer));
        const size_t allocatable_pages = allocatable_region_size / sizeof(PageBuffer);

        m_slabs.clear();
        m_slabs.resize(Common::DivCeil(allocatable_pages, SlabPages));
        m_bitmap_storage.assign(overhead_size / sizeof(u64), 0);

        m_page_bitmap.Initialize(m_bitmap_storage.data(), allocatable_pages);

        // Mark managed pages free in the bitmap without touching backing slabs.
        const size_t start_offset = (m_address - m_aligned_address) / sizeof(PageBuffer);
        for (size_t i = 0; i < m_count; i++) {
            m_page_bitmap.SetBit(start_offset + i);
        }

        R_SUCCEED();
    }

    KVirtualAddress GetAddress() const {
        return m_address;
    }
    size_t GetSize() const {
        return m_size;
    }
    size_t GetUsed() const {
        return m_used;
    }
    size_t GetPeak() const {
        return m_peak;
    }
    size_t GetCount() const {
        return m_count;
    }

    PageBuffer* Allocate() {
        // Take the lock.
        // TODO(bunnei): We should disable interrupts here via KScopedInterruptDisable.
        KScopedSpinLock lk(m_lock);

        // Pack allocations sequentially to keep resident slab count minimal.
        s64 soffset = m_page_bitmap.FindFreeBlock(false);
        if (soffset < 0) [[unlikely]] {
            return nullptr;
        }

        const size_t offset = static_cast<size_t>(soffset);

        // Update our tracking.
        m_page_bitmap.ClearBit(offset);
        m_peak = (std::max)(m_peak, (++m_used));

        return EnsurePage(offset);
    }

    PageBuffer* Allocate(size_t count) {
        // Take the lock.
        // TODO(bunnei): We should disable interrupts here via KScopedInterruptDisable.
        KScopedSpinLock lk(m_lock);

        // Find a free range.
        s64 soffset = m_page_bitmap.FindFreeRange(count);
        if (soffset < 0) [[likely]] {
            return nullptr;
        }

        const size_t offset = static_cast<size_t>(soffset);

        // Update our tracking.
        m_page_bitmap.ClearRange(offset, count);
        m_used += count;
        m_peak = (std::max)(m_peak, m_used);

        return EnsurePage(offset);
    }

    void Free(PageBuffer* pb) {
        // Ensure all pages in the heap are zero.
        std::memset(pb, 0, sizeof(PageBuffer));

        // Take the lock.
        // TODO(bunnei): We should disable interrupts here via KScopedInterruptDisable.
        KScopedSpinLock lk(m_lock);

        for (size_t slab_idx = 0; slab_idx < m_slabs.size(); ++slab_idx) {
            if (const auto& slab = m_slabs[slab_idx]; slab) {
                const PageBuffer* const begin = slab->pages.data();
                const PageBuffer* const end = begin + SlabPages;
                if (pb >= begin && pb < end) {
                    const size_t offset = slab_idx * SlabPages + static_cast<size_t>(pb - begin);
                    m_page_bitmap.SetBit(offset);
                    --m_used;
                    return;
                }
            }
        }
        UNREACHABLE_MSG("PageBuffer freed to wrong KDynamicPageManager");
    }

private:
    KSpinLock m_lock;
    KPageBitmap m_page_bitmap;
    size_t m_used{};
    size_t m_peak{};
    size_t m_count{};
    KVirtualAddress m_address{};
    KVirtualAddress m_aligned_address{};
    size_t m_size{};

    std::vector<u64> m_bitmap_storage;
    mutable std::vector<std::unique_ptr<Slab>> m_slabs;
};

} // namespace Kernel
