// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/assert.h"
#include "common/atomic_ops.h"
#include "common/common_funcs.h"
#include "common/common_types.h"

namespace Kernel {

class KernelCore;

namespace impl {

class KSlabHeapImpl {
    YUZU_NON_COPYABLE(KSlabHeapImpl);
    YUZU_NON_MOVEABLE(KSlabHeapImpl);

public:
    struct Node {
        Node* next{};
    };

public:
    KSlabHeapImpl() = default;

    void Initialize(void* memory = nullptr, size_t memory_size = 0, size_t obj_size = 0) {
        ASSERT(m_head == nullptr);
        m_obj_size = obj_size;
        m_next_free = reinterpret_cast<uintptr_t>(memory);
        const size_t num_obj = obj_size != 0 ? (memory_size / obj_size) : 0;
        m_end = m_next_free + num_obj * obj_size;
    }

    Node* GetHead() const {
        return m_head;
    }

    void* Allocate() {
        // KScopedInterruptDisable di;

        m_lock.lock();

        void* ret = m_head.load(std::memory_order_relaxed);
        if (ret != nullptr) [[likely]] {
            m_head.store(static_cast<Node*>(ret)->next, std::memory_order_relaxed);
        } else if (m_next_free < m_end) [[likely]] {
            ret = reinterpret_cast<void*>(m_next_free);
            m_next_free += m_obj_size;
        }

        m_lock.unlock();
        return ret;
    }

    void Free(void* obj) {
        // KScopedInterruptDisable di;

        m_lock.lock();

        Node* node = static_cast<Node*>(obj);
        node->next = m_head.load(std::memory_order_relaxed);
        m_head.store(node, std::memory_order_relaxed);

        m_lock.unlock();
    }

private:
    std::atomic<Node*> m_head{};
    uintptr_t m_next_free{};
    uintptr_t m_end{};
    size_t m_obj_size{};
    std::mutex m_lock;
};

} // namespace impl

template <bool SupportDynamicExpansion>
class KSlabHeapBase : protected impl::KSlabHeapImpl {
    YUZU_NON_COPYABLE(KSlabHeapBase);
    YUZU_NON_MOVEABLE(KSlabHeapBase);

private:
    size_t m_obj_size{};
    uintptr_t m_peak{};
    uintptr_t m_start{};
    uintptr_t m_end{};

private:
    void UpdatePeakImpl(uintptr_t obj) {
        const uintptr_t alloc_peak = obj + this->GetObjectSize();
        uintptr_t cur_peak = m_peak;
        do {
            if (alloc_peak <= cur_peak) {
                break;
            }
        } while (
            !Common::AtomicCompareAndSwap(std::addressof(m_peak), alloc_peak, cur_peak, cur_peak));
    }

public:
    constexpr KSlabHeapBase() = default;

    bool Contains(uintptr_t address) const {
        return m_start <= address && address < m_end;
    }

    void Initialize(size_t obj_size, void* memory, size_t memory_size) {
        // Ensure we don't initialize a slab using null memory.
        ASSERT(memory != nullptr);

        // Set our object size.
        m_obj_size = obj_size;

        // Set our tracking variables.
        const size_t num_obj = (memory_size / obj_size);
        m_start = reinterpret_cast<uintptr_t>(memory);
        m_end = m_start + num_obj * obj_size;
        m_peak = m_start;

        // Initialize the base allocator with bump range [m_start, m_end) so unused pages stay clean.
        KSlabHeapImpl::Initialize(memory, num_obj * obj_size, obj_size);
    }

    size_t GetSlabHeapSize() const {
        return (m_end - m_start) / this->GetObjectSize();
    }

    size_t GetObjectSize() const {
        return m_obj_size;
    }

    void* Allocate() {
        void* obj = KSlabHeapImpl::Allocate();

        return obj;
    }

    void Free(void* obj) {
        // Don't allow freeing an object that wasn't allocated from this heap.
        const bool contained = this->Contains(reinterpret_cast<uintptr_t>(obj));
        ASSERT(contained);
        KSlabHeapImpl::Free(obj);
    }

    size_t GetObjectIndex(const void* obj) const {
        if constexpr (SupportDynamicExpansion) {
            if (!this->Contains(reinterpret_cast<uintptr_t>(obj))) {
                return (std::numeric_limits<size_t>::max)();
            }
        }

        return (reinterpret_cast<uintptr_t>(obj) - m_start) / this->GetObjectSize();
    }

    size_t GetPeakIndex() const {
        return this->GetObjectIndex(reinterpret_cast<const void*>(m_peak));
    }

    uintptr_t GetSlabHeapAddress() const {
        return m_start;
    }

    size_t GetNumRemaining() const {
        // Only calculate the number of remaining objects under debug configuration.
        return 0;
    }
};

template <typename T>
class KSlabHeap final : public KSlabHeapBase<false> {
private:
    using BaseHeap = KSlabHeapBase<false>;

public:
    constexpr KSlabHeap() = default;

    void Initialize(void* memory, size_t memory_size) {
        BaseHeap::Initialize(sizeof(T), memory, memory_size);
    }

    T* Allocate() {
        T* obj = static_cast<T*>(BaseHeap::Allocate());

        if (obj != nullptr) [[likely]] {
            std::construct_at(obj);
        }
        return obj;
    }

    T* Allocate(KernelCore& kernel) {
        T* obj = static_cast<T*>(BaseHeap::Allocate());

        if (obj != nullptr) [[likely]] {
            std::construct_at(obj, kernel);
        }
        return obj;
    }

    void Free(T* obj) {
        BaseHeap::Free(obj);
    }

    size_t GetObjectIndex(const T* obj) const {
        return BaseHeap::GetObjectIndex(obj);
    }
};

} // namespace Kernel
