// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stop_token>
#include <system_error>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#include <thread>
#else
#include <pthread.h>
#endif

namespace Common {

/// RAII joinable thread started with an explicit stack size on POSIX (falling back to std::thread
/// on Windows). Supports both `void()` and `void(std::stop_token)` callables like `std::jthread`.
class LargeStackThread {
public:
    LargeStackThread() noexcept : stop_source{std::nostopstate} {}

    LargeStackThread(const LargeStackThread&) = delete;
    LargeStackThread& operator=(const LargeStackThread&) = delete;

    LargeStackThread(LargeStackThread&& other) noexcept
        : stop_source{std::move(other.stop_source)},
#if defined(_WIN32)
          thread{std::move(other.thread)}
#else
          thread{other.thread},
          active{std::exchange(other.active, false)}
#endif
    {
    }

    LargeStackThread& operator=(LargeStackThread&& other) noexcept {
        if (this != &other) {
            if (joinable()) {
                stop_source.request_stop();
                join();
            }
            stop_source = std::move(other.stop_source);
#if defined(_WIN32)
            thread = std::move(other.thread);
#else
            thread = other.thread;
            active = std::exchange(other.active, false);
#endif
        }
        return *this;
    }

    ~LargeStackThread() {
        if (joinable()) {
            stop_source.request_stop();
            join();
        }
    }

    [[nodiscard]] bool joinable() const noexcept {
#if defined(_WIN32)
        return thread.joinable();
#else
        return active;
#endif
    }

    void join() {
#if defined(_WIN32)
        thread.join();
#else
        if (active) {
            pthread_join(thread, nullptr);
            active = false;
        }
#endif
    }

    bool request_stop() noexcept {
        return stop_source.request_stop();
    }

    [[nodiscard]] std::stop_token get_stop_token() const noexcept {
        return stop_source.get_token();
    }

private:
    template <typename Fn>
    friend LargeStackThread StartThreadWithStack(std::size_t stack_size, Fn&& fn);

    std::stop_source stop_source;
#if defined(_WIN32)
    std::thread thread;
#else
    pthread_t thread{};
    bool active{false};
#endif
};

/// Starts a joinable thread with at least `stack_size` bytes of stack on POSIX (`stack_size == 0`
/// uses the system default). Falls back to `std::thread` on Windows.
template <typename Fn>
[[nodiscard]] LargeStackThread StartThreadWithStack(std::size_t stack_size, Fn&& fn) {
    LargeStackThread result;
    result.stop_source = std::stop_source{};
    using Decayed = std::decay_t<Fn>;
#if defined(_WIN32)
    static_cast<void>(stack_size);
    if constexpr (std::is_invocable_v<Decayed, std::stop_token>) {
        auto token = result.stop_source.get_token();
        result.thread = std::thread(std::forward<Fn>(fn), std::move(token));
    } else {
        result.thread = std::thread(std::forward<Fn>(fn));
    }
    return result;
#else
    struct State {
        Decayed fn;
        std::stop_token token;
    };
    auto state = std::make_unique<State>(State{std::forward<Fn>(fn), result.stop_source.get_token()});
    const auto entry = [](void* argument) -> void* {
        std::unique_ptr<State> owned{static_cast<State*>(argument)};
        if constexpr (std::is_invocable_v<Decayed, std::stop_token>) {
            owned->fn(std::move(owned->token));
        } else {
            owned->fn();
        }
        return nullptr;
    };
    int rc = -1;
    if (stack_size > 0) {
        pthread_attr_t attributes;
        if (pthread_attr_init(&attributes) == 0) {
            constexpr std::size_t CallerHeadroom = std::size_t{512} << 10;
            if (pthread_attr_setstacksize(&attributes, stack_size + CallerHeadroom) != 0) {
                pthread_attr_setstacksize(&attributes, stack_size);
            }
            rc = pthread_create(&result.thread, &attributes, entry, state.get());
            pthread_attr_destroy(&attributes);
        }
    }
    if (rc != 0) {
        rc = pthread_create(&result.thread, nullptr, entry, state.get());
    }
    if (rc != 0) {
        throw std::system_error(rc, std::system_category(), "pthread_create");
    }
    static_cast<void>(state.release());
    result.active = true;
    return result;
#endif
}

/// Runs `func` on a stack with at least `stack_size` bytes available. On Apple, runs directly on
/// the calling thread when enough stack space remains; otherwise spawns a `stack_size` thread and
/// waits for it. For work that recurses deeply (SPIRV-Cross walks a shader's control flow
/// recursively, one frame per nested block): macOS gives secondary threads 512 KiB and std::thread
/// can't ask for more, and a large game shader overflowed it (SIGBUS "Thread stack size exceeded
/// due to excessive recursion"). An exception `func` throws is rethrown here. Falls back to
/// calling `func` directly when the thread can't be made.
template <typename Func>
void RunOnLargeStack(Func&& func, std::size_t stack_size = std::size_t{16} << 20) {
#if defined(_WIN32)
    static_cast<void>(stack_size);
    std::forward<Func>(func)();
#else
#if defined(__APPLE__)
    char probe = 0;
    const auto sp = reinterpret_cast<std::uintptr_t>(&probe);
    const auto top = reinterpret_cast<std::uintptr_t>(pthread_get_stackaddr_np(pthread_self()));
    const std::size_t total = pthread_get_stacksize_np(pthread_self());
    constexpr std::size_t FrameAllowance = std::size_t{256} << 10;
    const std::size_t required =
        stack_size > FrameAllowance ? stack_size - FrameAllowance : stack_size;
    if (top >= total && sp >= (top - total) && sp <= top && (sp - (top - total)) >= required) {
        std::forward<Func>(func)();
        return;
    }
#endif
    struct Context {
        Func* func;
        std::exception_ptr error;
    } context{&func, nullptr};
    const auto entry = [](void* argument) -> void* {
        auto* const ctx = static_cast<Context*>(argument);
        try {
            (*ctx->func)();
        } catch (...) {
            ctx->error = std::current_exception();
        }
        return nullptr;
    };
    pthread_attr_t attributes;
    pthread_t thread;
    bool started = false;
    if (pthread_attr_init(&attributes) == 0) {
        constexpr std::size_t CallerHeadroom = std::size_t{512} << 10;
        const bool size_ok =
            pthread_attr_setstacksize(&attributes, stack_size + CallerHeadroom) == 0 ||
            pthread_attr_setstacksize(&attributes, stack_size) == 0;
        started = size_ok && pthread_create(&thread, &attributes, entry, &context) == 0;
        pthread_attr_destroy(&attributes);
    }
    if (!started) {
        std::forward<Func>(func)();
        return;
    }
    pthread_join(thread, nullptr);
    if (context.error) {
        std::rethrow_exception(context.error);
    }
#endif
}

} // namespace Common
