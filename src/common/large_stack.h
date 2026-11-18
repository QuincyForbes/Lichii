// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <exception>
#include <utility>

#if defined(_WIN32)
#include <thread>
#else
#include <pthread.h>
#endif

namespace Common {

/// Runs `func` on a thread of its own with a `stack_size` stack and waits for it. For work that
/// recurses deeply (SPIRV-Cross walks a shader's control flow recursively, one frame per nested
/// block): macOS gives secondary threads 512 KiB and std::thread can't ask for more, and a large
/// game shader overflowed it (SIGBUS "Thread stack size exceeded due to excessive recursion").
/// An exception `func` throws is rethrown here. Falls back to calling `func` directly when the
/// thread can't be made.
template <typename Func>
void RunOnLargeStack(Func&& func, std::size_t stack_size = std::size_t{16} << 20) {
#if defined(_WIN32)
    static_cast<void>(stack_size);
    std::forward<Func>(func)();
#else
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
        started = pthread_attr_setstacksize(&attributes, stack_size) == 0 &&
                  pthread_create(&thread, &attributes, entry, &context) == 0;
        pthread_attr_destroy(&attributes);
    }
    if (!started) {
        func();
        return;
    }
    pthread_join(thread, nullptr);
    if (context.error) {
        std::rethrow_exception(context.error);
    }
#endif
}

} // namespace Common
