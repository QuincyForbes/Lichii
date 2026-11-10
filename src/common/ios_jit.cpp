// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/ios_jit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <libkern/OSCacheControl.h>
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include "common/logging.h"

extern "C" int csops(pid_t pid, unsigned int ops, void* useraddr, size_t usersize);

namespace Common::IosJit {
namespace {

constexpr unsigned int kCsOpsStatus = 0;
constexpr int kCsDebugged = 0x10000000;
constexpr std::size_t kJitPageSize = 0x4000;
constexpr std::uint32_t kBrkF00d = 0xD43E01A0; // brk #0xf00d

struct Region {
    std::uint8_t* rx{};
    std::uint8_t* rw{};
    std::size_t size{};
    std::size_t used{};
};
std::vector<Region> g_pool;
std::mutex g_pool_mutex;
std::atomic<bool> g_pool_ready{false};
std::atomic<std::size_t> g_prepared{0};
std::atomic<State> g_state{State::Waiting};
std::mutex g_error_mutex;
std::string g_error;
std::atomic<bool> g_thread_started{false};
std::atomic<StateCallback> g_state_callback{nullptr};

void NotifyState() {
    if (const StateCallback callback = g_state_callback.load()) {
        callback();
    }
}

void SetState(State state) {
    g_state = state;
    NotifyState();
}

void SetFailed(const std::string& error) {
    {
        std::scoped_lock lock{g_error_mutex};
        g_error = error;
    }
    LOG_CRITICAL(Common, "iOS JIT: {}", error);
    SetState(State::Failed);
}

// brk-based "syscalls" serviced by StikDebug's universal.js (x16 = command).
__attribute__((noinline, optnone)) void* Jit26PrepareRegion(void* addr, std::size_t len) {
    register void* x0 __asm__("x0") = addr;
    register std::size_t x1 __asm__("x1") = len;
    __asm__ volatile(
        "mov x16, #1\n"
        "brk #0xf00d\n"
        : "+r"(x0)
        : "r"(x1)
        : "x16", "memory");
    return x0;
}

__attribute__((noinline, optnone)) void Jit26Detach() {
    __asm__ volatile(
        "mov x16, #0\n"
        "brk #0xf00d\n" ::
            : "x16", "memory");
}

// If nobody services our brk (debugger gone, or attached without the universal script), skip it
// and return nullptr instead of dying. Any other trap gets the default action back so real
// crashes still produce reports. While a debugger is attached the exception goes to it first and
// this handler never runs.
void SigtrapHandler(int sig, siginfo_t*, void* context) {
    auto* uc = static_cast<ucontext_t*>(context);
    const std::uint64_t pc = uc->uc_mcontext->__ss.__pc;
    if ((pc & 3) != 0 || *reinterpret_cast<const std::uint32_t*>(pc) != kBrkF00d) {
        struct sigaction dfl {};
        dfl.sa_handler = SIG_DFL;
        sigaction(sig, &dfl, nullptr);
        return;
    }
    uc->uc_mcontext->__ss.__pc += 4;
    uc->uc_mcontext->__ss.__x[0] = 0;
}

void InstallTrapGuard() {
    struct sigaction sa {};
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = SigtrapHandler;
    sigaction(SIGTRAP, &sa, nullptr);
}

std::size_t AlignUp(std::size_t v, std::size_t a) {
    return (v + a - 1) & ~(a - 1);
}

bool SelfTest(std::uint8_t* rw, std::uint8_t* rx) {
    // mov w0, #42 ; ret
    const std::uint32_t code[2] = {0x52800540, 0xD65F03C0};
    std::memcpy(rw, code, sizeof(code));
    sys_icache_invalidate(rx, sizeof(code));
    auto fn = reinterpret_cast<int (*)()>(rx);
    return fn() == 42;
}

bool PublishPool(std::vector<Region> regions) {
    if (regions.empty() || !SelfTest(regions.front().rw, regions.front().rx)) {
        SetFailed("JIT self-test failed: code written through the alias did not execute");
        return false;
    }
    std::size_t total = 0;
    for (const Region& region : regions) {
        total += region.size;
    }
    {
        std::scoped_lock lock{g_pool_mutex};
        g_pool = std::move(regions);
        g_pool_ready = true;
    }
    LOG_INFO(Common, "iOS JIT: pool ready ({} MiB), self-test passed", total >> 20);
    SetState(State::Ready);
    return true;
}

// A writable alias of a prepared R-X region.
std::uint8_t* MapWritableAlias(std::uint8_t* rx, std::size_t bytes, std::string& error) {
    vm_address_t alias = 0;
    vm_prot_t cur = 0, max = 0;
    kern_return_t kr = vm_remap(mach_task_self(), &alias, bytes, 0, VM_FLAGS_ANYWHERE,
                                mach_task_self(), reinterpret_cast<vm_address_t>(rx), FALSE, &cur,
                                &max, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS) {
        error = std::string("vm_remap of the JIT pool failed: ") + mach_error_string(kr);
        return nullptr;
    }
    kr = vm_protect(mach_task_self(), alias, bytes, FALSE, VM_PROT_READ | VM_PROT_WRITE);
    if (kr != KERN_SUCCESS) {
        error = std::string("vm_protect(RW) of the JIT alias failed: ") + mach_error_string(kr);
        return nullptr;
    }
    return reinterpret_cast<std::uint8_t*>(alias);
}

// TXM: ask the attached universal script for the pool one chunk at a time (the script touches
// every 16 KiB page through the debugger, so one big request showed no progress for minutes),
// alias each chunk writable, detach. Returns false, leaving the state Waiting, when no script
// answered the first request (JIT was enabled without one) so the caller asks again later; true
// once the requests were answered, failed or not.
bool AcquireThroughDebugger(std::size_t bytes) {
    g_prepared = 0;
    SetState(State::Acquiring);
    std::vector<Region> regions;
    std::string error;
    while (g_prepared < bytes) {
        const std::size_t chunk = (std::min)(ChunkBytes, bytes - g_prepared);
        auto* rx = static_cast<std::uint8_t*>(Jit26PrepareRegion(nullptr, chunk));
        if (rx == nullptr) {
            if (regions.empty()) {
                SetState(State::Waiting);
                return false;
            }
            error = "the debugger stopped answering after " +
                    std::to_string(g_prepared.load() >> 20) + " MiB of JIT memory";
            break;
        }
        std::uint8_t* rw = MapWritableAlias(rx, chunk, error);
        if (rw == nullptr) {
            break;
        }
        regions.push_back(Region{rx, rw, chunk, 0});
        g_prepared += chunk;
        LOG_INFO(Common, "iOS JIT: prepared {} of {} MiB", g_prepared.load() >> 20, bytes >> 20);
        NotifyState();
    }
    // Done with the debugger; StikDebug returns to its UI (and to this app when it was asked
    // through its URL scheme).
    Jit26Detach();
    if (!error.empty()) {
        SetFailed(error);
        return true;
    }
    PublishPool(std::move(regions));
    return true;
}

// Pre-TXM: a debugged process may map RWX directly.
bool AcquireRwx(std::size_t bytes) {
    SetState(State::Acquiring);
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) {
        SetFailed(std::string("RWX mapping refused (") + strerror(errno) +
                  "): this iOS version needs StikDebug's universal script");
        return false;
    }
    auto* mem = static_cast<std::uint8_t*>(p);
    LOG_INFO(Common, "iOS JIT: RWX pool at {} ({} MiB)", fmt::ptr(mem), bytes >> 20);
    g_prepared = bytes;
    return PublishPool({Region{mem, mem, bytes, 0}});
}

void AcquireThread(std::size_t bytes, bool txm) {
    using namespace std::chrono_literals;
    LOG_INFO(Common, "iOS JIT: waiting for a debugger (TXM: {}, debugged: {}, traced: {})", txm,
             Debugged(), DebuggerAttached());
    InstallTrapGuard();
    std::chrono::steady_clock::time_point unanswered_since{};
    const auto started = std::chrono::steady_clock::now();
    while (true) {
        if (txm) {
            // StikDebug's protocol: once the process is debugged, call into the script. Its
            // debugserver need not show as P_TRACED, so CS_DEBUGGED is the signal. A call nobody
            // answers (JIT enabled from StikDebug's app list without the script: a plain
            // attach/detach) returns null through the trap guard; keep asking, the frontend
            // requests the script through StikDebug's URL scheme meanwhile.
            if (Debugged() || DebuggerAttached()) {
                if (AcquireThroughDebugger(bytes)) {
                    return;
                }
                const auto now = std::chrono::steady_clock::now();
                if (unanswered_since == std::chrono::steady_clock::time_point{}) {
                    unanswered_since = now;
                    LOG_WARNING(Common, "iOS JIT: debugged, but no universal script answered; "
                                        "waiting for StikDebug to attach with it");
                } else if (now - unanswered_since > 3min) {
                    SetFailed("JIT was enabled without the universal script, which iOS 26+ needs. "
                              "Close this app and launch it again; when StikDebug opens, let it "
                              "enable JIT from this app's request (script universal.js).");
                    return;
                }
                std::this_thread::sleep_for(1s);
                continue;
            }
        } else if (Debugged()) {
            AcquireRwx(bytes);
            return;
        }
        // Quick at first (StikDebug attaches right after the launch request), then 4 Hz for the
        // rest of a session without JIT.
        std::this_thread::sleep_for(std::chrono::steady_clock::now() - started < 15s ? 50ms
                                                                                    : 250ms);
    }
}

} // Anonymous namespace

bool DebuggerAttached() {
    struct kinfo_proc info {};
    std::size_t size = sizeof(info);
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()};
    if (sysctl(mib, 4, &info, &size, nullptr, 0) != 0) {
        return false;
    }
    return (info.kp_proc.p_flag & P_TRACED) != 0;
}

bool Debugged() {
    int flags = 0;
    if (csops(getpid(), kCsOpsStatus, &flags, sizeof(flags)) != 0) {
        return false;
    }
    return (flags & kCsDebugged) != 0;
}

bool PoolReady() {
    return g_pool_ready.load();
}

void StartAcquire(std::size_t bytes, bool txm) {
    if (g_thread_started.exchange(true)) {
        return;
    }
    std::thread(AcquireThread, AlignUp(bytes, kJitPageSize), txm).detach();
}

State GetState() {
    return g_state.load();
}

std::string GetError() {
    std::scoped_lock lock{g_error_mutex};
    return g_error;
}

std::size_t PreparedBytes() {
    return g_prepared.load();
}

void SetStateCallback(StateCallback callback) {
    g_state_callback = callback;
}

bool Alloc(std::size_t bytes, void** wmem, void** xmem) {
    std::scoped_lock lock{g_pool_mutex};
    if (!g_pool_ready) {
        LOG_CRITICAL(Common, "iOS JIT: code block requested before the pool exists");
        return false;
    }
    bytes = AlignUp(bytes, kJitPageSize);
    // First fit: a block never straddles two chunks, which are separate mappings.
    std::size_t left = 0;
    for (Region& region : g_pool) {
        if (region.used + bytes <= region.size) {
            *wmem = region.rw + region.used;
            *xmem = region.rx + region.used;
            region.used += bytes;
            LOG_INFO(Common, "iOS JIT: handed out {} MiB (x={}, w={})", bytes >> 20,
                     fmt::ptr(*xmem), fmt::ptr(*wmem));
            return true;
        }
        left += region.size - region.used;
    }
    LOG_CRITICAL(Common, "iOS JIT: pool exhausted ({} MiB requested, {} MiB left in {} chunks)",
                 bytes >> 20, left >> 20, g_pool.size());
    return false;
}

void Reset() {
    std::scoped_lock lock{g_pool_mutex};
    for (Region& region : g_pool) {
        region.used = 0;
    }
    LOG_INFO(Common, "iOS JIT: pool reset ({} chunks free)", g_pool.size());
}

} // namespace Common::IosJit

extern "C" int lichii_ios_jit_alloc(std::size_t bytes, void** wmem, void** xmem) {
    return Common::IosJit::Alloc(bytes, wmem, xmem) ? 1 : 0;
}
