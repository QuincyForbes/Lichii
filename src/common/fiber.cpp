// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/fiber.h"

#include <boost/context/detail/fcontext.hpp>

namespace Common {

namespace {

#ifndef _WIN32
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

void PauseHint() noexcept {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

std::size_t HostStackPageSize() noexcept {
#ifdef _WIN32
    static const std::size_t page_size = [] {
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        return si.dwPageSize != 0 ? static_cast<std::size_t>(si.dwPageSize) : std::size_t{4096};
    }();
    return page_size;
#else
    static const std::size_t page_size = [] {
        const long res = sysconf(_SC_PAGESIZE);
        return res > 0 ? static_cast<std::size_t>(res) : std::size_t{4096};
    }();
    return page_size;
#endif
}

void AcquireFiber(std::atomic<bool>& in_use) noexcept {
    if (!in_use.exchange(true, std::memory_order_acquire)) [[likely]] {
        return;
    }
    u32 spins = 0;
    while (true) {
        while (in_use.load(std::memory_order_relaxed)) {
            if (++spins < 256) {
                PauseHint();
            } else {
                std::this_thread::yield();
            }
        }
        if (!in_use.exchange(true, std::memory_order_acquire)) {
            return;
        }
    }
}

} // Anonymous namespace

struct Fiber::FiberImpl {
    FiberImpl() = default;

    ~FiberImpl() {
        if (alloc_base != nullptr) {
#ifdef _WIN32
            const BOOL ok = VirtualFree(alloc_base, 0, MEM_RELEASE);
            ASSERT(ok != 0);
#else
            const int res = munmap(alloc_base, total_alloc_size);
            ASSERT(res == 0);
#endif
        }
    }

    FiberImpl(const FiberImpl&) = delete;
    FiberImpl& operator=(const FiberImpl&) = delete;

    void AllocateStack(std::size_t requested_size) {
        const std::size_t page_size = HostStackPageSize();
        usable_stack_size = Common::AlignUp(
            requested_size != 0 ? requested_size : Fiber::DefaultStackSize, page_size);
        total_alloc_size = usable_stack_size + page_size;
#ifdef _WIN32
        alloc_base =
            static_cast<u8*>(VirtualAlloc(nullptr, total_alloc_size, MEM_RESERVE, PAGE_NOACCESS));
        ASSERT(alloc_base != nullptr);
        void* committed =
            VirtualAlloc(alloc_base + page_size, usable_stack_size, MEM_COMMIT, PAGE_READWRITE);
        ASSERT(committed != nullptr);
#else
        void* mapped = mmap(nullptr, total_alloc_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT(mapped != MAP_FAILED);
        alloc_base = static_cast<u8*>(mapped);
        const int prot_res = mprotect(alloc_base, page_size, PROT_NONE);
        ASSERT(prot_res == 0);
#if defined(__linux__) && defined(MADV_NOHUGEPAGE)
        void(madvise(alloc_base + page_size, usable_stack_size, MADV_NOHUGEPAGE));
#endif
#endif
        stack_limit = alloc_base + page_size;
    }

    u8* alloc_base = nullptr;
    u8* stack_limit = nullptr;
    std::size_t usable_stack_size = 0;
    std::size_t total_alloc_size = 0;

    boost::context::detail::fcontext_t context{};
    boost::context::detail::fcontext_t rewind_context{};

    std::atomic<bool> in_use{false};
    FiberImpl* previous_impl = nullptr;
    std::function<void()> entry_point;
    std::function<void()> rewind_point;

    bool is_thread_fiber = false;
    bool released = false;
};

void Fiber::SetRewindPoint(std::function<void()>&& rewind_func) {
    impl->rewind_point = std::move(rewind_func);
}

Fiber::Fiber(std::function<void()>&& entry_point_func, std::size_t stack_size)
    : impl{std::make_unique<FiberImpl>()} {
    impl->entry_point = std::move(entry_point_func);
    impl->AllocateStack(stack_size);
    u8* const stack_base = impl->stack_limit + impl->usable_stack_size;
    impl->context = boost::context::detail::make_fcontext(
        stack_base, impl->usable_stack_size, [](boost::context::detail::transfer_t transfer) -> void {
            auto* self = static_cast<FiberImpl*>(transfer.data);
            ASSERT(self != nullptr && self->previous_impl != nullptr);
            FiberImpl* const prev = self->previous_impl;
            self->previous_impl = nullptr;
            prev->context = transfer.fctx;
            prev->in_use.store(false, std::memory_order_release);
            self->entry_point();
            UNREACHABLE();
        });
}

Fiber::Fiber() : impl{std::make_unique<FiberImpl>()} {}

Fiber::~Fiber() {
    if (!impl) {
        return;
    }
    if (!impl->released) {
        // Make sure the Fiber is not being used
        const bool free_to_destroy = !impl->in_use.exchange(true, std::memory_order_acquire);
        ASSERT_MSG(free_to_destroy, "Destroying a fiber that's still running");
        if (free_to_destroy) {
            impl->in_use.store(false, std::memory_order_release);
        }
    }
}

void Fiber::Exit() {
    ASSERT_MSG(impl->is_thread_fiber, "Exiting non main thread fiber");
    if (impl->is_thread_fiber) {
        impl->in_use.store(false, std::memory_order_release);
        impl->released = true;
    }
}

void Fiber::YieldToImpl(FiberImpl* from_impl, FiberImpl* to_impl) {
    ASSERT(from_impl != nullptr && to_impl != nullptr);
    AcquireFiber(to_impl->in_use);
    to_impl->previous_impl = from_impl;

    auto transfer = boost::context::detail::jump_fcontext(to_impl->context, to_impl);
    auto* self = static_cast<FiberImpl*>(transfer.data);
    ASSERT(self != nullptr);
    if (FiberImpl* const prev = self->previous_impl; prev != nullptr) {
        self->previous_impl = nullptr;
        prev->context = transfer.fctx;
        prev->in_use.store(false, std::memory_order_release);
    } else {
        ASSERT(false && "previous_impl is nullptr!");
    }
}

void Fiber::YieldTo(const std::shared_ptr<Fiber>& from, Fiber& to) {
    ASSERT(from != nullptr);
    YieldToImpl(from->impl.get(), to.impl.get());
}

void Fiber::YieldTo(std::weak_ptr<Fiber> weak_from, Fiber& to) {
    auto from = weak_from.lock();
    ASSERT(from != nullptr);
    YieldToImpl(from->impl.get(), to.impl.get());
}

std::shared_ptr<Fiber> Fiber::ThreadToFiber() {
    std::shared_ptr<Fiber> fiber = std::shared_ptr<Fiber>{new Fiber()};
    fiber->impl->in_use.store(true, std::memory_order_relaxed);
    fiber->impl->is_thread_fiber = true;
    return fiber;
}

} // namespace Common
