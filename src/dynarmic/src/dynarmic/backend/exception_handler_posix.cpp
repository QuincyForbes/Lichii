// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2019 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

#include "common/container/unordered_map.h"
#include <fmt/format.h>
#include <sys/mman.h>

#include "common/assert.h"
#include "common/logging.h"
#include "common/common_types.h"
#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/common/context.h"
#if defined(__APPLE__) && defined(ARCHITECTURE_arm64)
#    include <iterator>
#    include <string>
#    include <TargetConditionals.h>
#    include <dlfcn.h>
#    include <mach/mach.h>
#    if TARGET_OS_IPHONE
#        include "common/ios_jit.h"
#    endif
#endif
#if defined(ARCHITECTURE_x86_64)
#    include "dynarmic/backend/x64/block_of_code.h"
#elif defined(ARCHITECTURE_arm64)
#    include <oaknut/code_block.hpp>
#    include "dynarmic/backend/arm64/abi.h"
#elif defined(ARCHITECTURE_riscv64)
#    include "dynarmic/backend/riscv64/code_block.h"
#elif defined(ARCHITECTURE_loongarch64)
#    include "dynarmic/backend/loongarch64/code_block.h"
#else
#    error "Invalid architecture"
#endif

namespace Dynarmic::Backend {

namespace {

struct CodeBlockInfo {
    u64 size;
    std::function<FakeCall(u64)> cb;
};

class SigHandler {
    auto FindCodeBlockInfo(u64 offset) noexcept {
        return std::find_if(code_block_infos.begin(), code_block_infos.end(), [&](auto const& e) {
            return e.first <= offset && e.first + e.second.size > offset;
        });
    }

    ::Common::unordered_map<u64, CodeBlockInfo> code_block_infos;
    std::shared_mutex code_block_infos_mutex;
    struct sigaction old_sa_segv;
    struct sigaction old_sa_bus;
    std::unique_ptr<uint8_t[]> signal_stack_memory;
    bool supports_fast_mem = true;
public:
    SigHandler() noexcept {
        auto const stack_size = std::max<size_t>(SIGSTKSZ, 2 * 1024 * 1024);
        signal_stack_memory = std::make_unique<uint8_t[]>(stack_size);

        stack_t signal_stack{};
        signal_stack.ss_sp = signal_stack_memory.get();
        signal_stack.ss_size = stack_size;
        signal_stack.ss_flags = 0;
        if (sigaltstack(&signal_stack, nullptr) != 0) {
            fmt::print(stderr, "dynarmic: POSIX SigHandler: init failure at sigaltstack\n");
            supports_fast_mem = false;
            return;
        }

        struct sigaction sa{};
        sa.sa_handler = nullptr;
        sa.sa_sigaction = &SigHandler::SigAction;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        if (sigaction(SIGSEGV, &sa, &old_sa_segv) != 0) {
            fmt::print(stderr, "dynarmic: POSIX SigHandler: could not set SIGSEGV handler\n");
            supports_fast_mem = false;
            return;
        }
#if defined(__APPLE__)
        if (sigaction(SIGBUS, &sa, &old_sa_bus) != 0) {
            fmt::print(stderr, "dynarmic: POSIX SigHandler: could not set SIGBUS handler\n");
            supports_fast_mem = false;
            return;
        }
#endif
    }

    void AddCodeBlock(u64 offset, CodeBlockInfo cbi) noexcept {
        std::unique_lock guard(code_block_infos_mutex);
        code_block_infos.insert_or_assign(offset, cbi);
    }
    void RemoveCodeBlock(u64 offset) noexcept {
        std::unique_lock guard(code_block_infos_mutex);
        code_block_infos.erase(offset);
    }

    [[nodiscard]] inline bool SupportsFastmem() const noexcept {
        return supports_fast_mem;
    }

    static void RegisterHandler();
    static void SigAction(int sig, siginfo_t* info, void* raw_context);
};

std::optional<SigHandler> sig_handler;

void SigHandler::RegisterHandler() {
    if (!sig_handler) {
        sig_handler.emplace();
    }
}

#if defined(__APPLE__) && defined(ARCHITECTURE_arm64)
// Image and offset of a host address, or its place in the iOS JIT pool.
std::string DescribeHostAddress(u64 address) {
#    if TARGET_OS_IPHONE
    if (const auto location = ::Common::IosJit::Locate(reinterpret_cast<const void*>(address))) {
        return fmt::format("JIT pool chunk {} +{:#x} ({} view, {})", location->chunk, location->offset,
                           location->writable_view ? "writable" : "executable",
                           location->handed_out ? "handed out" : "not handed out");
    }
#    endif
    Dl_info image{};
    if (dladdr(reinterpret_cast<const void*>(address), &image) != 0 && image.dli_fname != nullptr) {
        const char* const slash = std::strrchr(image.dli_fname, '/');
        return fmt::format("{} +{:#x}", slash != nullptr ? slash + 1 : image.dli_fname,
                           address - reinterpret_cast<u64>(image.dli_fbase));
    }
    return "no image";
}

// What an unhandled fault's one-line report can't tell: which code the thread was in. The frame
// records ({caller's fp, return address}) are read through the kernel, so a bad one ends the walk
// instead of faulting again.
void LogCrashContext(u64 pc, u64 lr, u64 fp) {
    LOG_ERROR(Core, "Crash context: pc in {}; lr {:#018x} in {}", DescribeHostAddress(pc), lr,
              DescribeHostAddress(lr));
    std::string frames;
    for (int depth = 0; depth < 24 && fp != 0 && fp % 8 == 0; ++depth) {
        u64 record[2]{};
        vm_size_t bytes_read = 0;
        if (vm_read_overwrite(mach_task_self(), static_cast<vm_address_t>(fp), sizeof(record),
                              reinterpret_cast<vm_address_t>(&record[0]), &bytes_read) != KERN_SUCCESS ||
            bytes_read != sizeof(record)) {
            break;
        }
        fmt::format_to(std::back_inserter(frames), "{}{:#x} {}", depth != 0 ? " < " : "", record[1],
                       DescribeHostAddress(record[1]));
        if (record[0] <= fp) {
            break;
        }
        fp = record[0];
    }
    LOG_ERROR(Core, "Crash backtrace: {}", frames.empty() ? "no frame records" : frames);
}
#endif

void SigHandler::SigAction(int sig, siginfo_t* info, void* raw_context) {
    DEBUG_ASSERT(sig == SIGSEGV || sig == SIGBUS);
    CTX_DECLARE(raw_context);
    {
        std::shared_lock guard(sig_handler->code_block_infos_mutex);
        if (auto const iter = sig_handler->FindCodeBlockInfo(CTX_PC); iter != sig_handler->code_block_infos.end()) {
            FakeCall fc = iter->second.cb(CTX_PC);
#if defined(ARCHITECTURE_x86_64)
            CTX_SP -= sizeof(u64);
            *std::bit_cast<u64*>(CTX_SP) = fc.ret_rip;
            CTX_PC = fc.call_rip;
#elif defined(ARCHITECTURE_arm64)
            CTX_PC = fc.call_pc;
#elif defined(ARCHITECTURE_riscv64)
            CTX_PC = fc.call_sepc;
#elif defined(ARCHITECTURE_loongarch64)
            CTX_PC = fc.call_pc;
#else
            ASSERT(false);
#endif
            return;
        }
    }
    char thread_name[64]{};
#if defined(__APPLE__) || (defined(__linux__) && !defined(__ANDROID__))
    pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
#endif
    LOG_ERROR(Core, "Unhandled {} at {:#018x} (fault address {}, thread \"{}\")\n",
              sig == SIGSEGV ? "SIGSEGV" : "SIGBUS", CTX_PC, fmt::ptr(info->si_addr), thread_name);
#if defined(__APPLE__) && defined(ARCHITECTURE_arm64)
    // Once: a fault inside this report lands here again.
    static std::atomic_flag context_logged = ATOMIC_FLAG_INIT;
    if (!context_logged.test_and_set()) {
        LogCrashContext(CTX_PC, CTX_LR, mctx->__ss.__fp);
    }
#endif

    struct sigaction* retry_sa = sig == SIGSEGV ? &sig_handler->old_sa_segv : &sig_handler->old_sa_bus;
    if (retry_sa->sa_flags & SA_SIGINFO) {
        retry_sa->sa_sigaction(sig, info, raw_context);
        return;
    }
    if (retry_sa->sa_handler == SIG_DFL) {
        signal(sig, SIG_DFL);
        return;
    }
    if (retry_sa->sa_handler == SIG_IGN) {
        return;
    }
    retry_sa->sa_handler(sig);
}

}  // anonymous namespace

struct ExceptionHandler::Impl final {
    Impl(u64 offset_, u64 size_)
        : offset(offset_)
        , size(size_)
    {
        SigHandler::RegisterHandler();
    }

    void SetCallback(std::function<FakeCall(u64)> cb) {
        sig_handler->AddCodeBlock(offset, CodeBlockInfo{
            .size = size,
            .cb = cb
        });
    }

    ~Impl() {
        sig_handler->RemoveCodeBlock(offset);
    }

private:
    u64 offset;
    u64 size;
};

ExceptionHandler::ExceptionHandler() = default;
ExceptionHandler::~ExceptionHandler() = default;

#if defined(ARCHITECTURE_x86_64)
void ExceptionHandler::Register(X64::BlockOfCode& code) {
    impl = std::make_unique<Impl>(std::bit_cast<u64>(code.getCode()), code.GetTotalCodeSize());
}
#elif defined(ARCHITECTURE_arm64)
void ExceptionHandler::Register(oaknut::CodeBlock& mem, std::size_t size) {
    impl = std::make_unique<Impl>(std::bit_cast<u64>(mem.ptr()), size);
}
#elif defined(ARCHITECTURE_riscv64)
void ExceptionHandler::Register(RV64::CodeBlock& mem, std::size_t size) {
    impl = std::make_unique<Impl>(std::bit_cast<u64>(mem.ptr<u64>()), size);
}
#elif defined(ARCHITECTURE_loongarch64)
void ExceptionHandler::Register(LoongArch64::CodeBlock& mem, std::size_t size) {
    impl = std::make_unique<Impl>(std::bit_cast<u64>(mem.ptr<u64>()), size);
}
#else
#    error "Invalid architecture"
#endif

bool ExceptionHandler::SupportsFastmem() const noexcept {
    return bool(impl) && sig_handler->SupportsFastmem();
}

void ExceptionHandler::SetFastmemCallback(std::function<FakeCall(u64)> cb) {
    impl->SetCallback(cb);
}

}  // namespace Dynarmic::Backend
