// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_address_wait.h"

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(__APPLE__) && __has_include(<os/os_sync_wait_on_address.h>)
#define LICHII_OS_SYNC_WAIT 1
#include <array>
#include <cstdint>
#include <cstdlib>
#include <os/os_sync_wait_on_address.h>
#endif

namespace Common::Nx2AddressWait {

namespace {

#ifdef _WIN32
using WaitOnAddressFn = BOOL(WINAPI*)(volatile VOID*, PVOID, SIZE_T, DWORD);
using WakeByAddressFn = VOID(WINAPI*)(PVOID);

struct AddressApi {
    WaitOnAddressFn wait{};
    WakeByAddressFn wake_one{};
    WakeByAddressFn wake_all{};
};

const AddressApi& Api() {
    static const AddressApi api = [] {
        AddressApi result{};
        HMODULE module = GetModuleHandleA("KernelBase.dll");
        if (module == nullptr) {
            module = LoadLibraryA("api-ms-win-core-synch-l1-2-0.dll");
        }
        if (module == nullptr) {
            return result;
        }
        result.wait = reinterpret_cast<WaitOnAddressFn>(GetProcAddress(module, "WaitOnAddress"));
        result.wake_one =
            reinterpret_cast<WakeByAddressFn>(GetProcAddress(module, "WakeByAddressSingle"));
        result.wake_all =
            reinterpret_cast<WakeByAddressFn>(GetProcAddress(module, "WakeByAddressAll"));
        if (result.wait == nullptr || result.wake_one == nullptr || result.wake_all == nullptr) {
            return AddressApi{};
        }
        return result;
    }();
    return api;
}
#endif

#ifdef LICHII_OS_SYNC_WAIT
// Apple's libc++ (LLVM 17-19) std::atomic::wait polls, then calls sched_yield for up to 64 us
// before it sleeps: every idle guest core and every contended kernel lock burned that much CPU
// per wait. os_sync_wait_on_address (iOS 17.4, macOS 14.4) sleeps at once. A wake is a syscall
// even with nobody asleep, so sleepers are counted per address slot and wakes skip empty ones.
// EDEN_NX2_APPLE_OS_SYNC=0 goes back to std::atomic::wait.
bool UseOsSync() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_APPLE_OS_SYNC");
        if (value != nullptr && value[0] == '0') {
            return false;
        }
        if (__builtin_available(iOS 17.4, macOS 14.4, *)) {
            return true;
        }
        return false;
    }();
    return enabled;
}

std::array<std::atomic<u32>, 256> g_sleepers{};

std::atomic<u32>& Sleepers(const void* address) {
    const auto value = reinterpret_cast<std::uintptr_t>(address);
    return g_sleepers[(value >> 2 ^ value >> 10) % g_sleepers.size()];
}

bool HasSleepers(std::atomic<u32>& word) {
    // Pairs with the sleeper's increment before the kernel compares the word: either it sees
    // the waker's store, or the waker sees the count.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return Sleepers(&word).load(std::memory_order_seq_cst) != 0;
}
#endif

} // Anonymous namespace

void Wait(std::atomic<u32>& word, u32 expected) {
#ifdef _WIN32
    if (const AddressApi& api = Api(); api.wait != nullptr) {
        api.wait(static_cast<void*>(&word), &expected, sizeof(u32), INFINITE);
        return;
    }
#endif
#ifdef LICHII_OS_SYNC_WAIT
    if (UseOsSync()) {
        if (__builtin_available(iOS 17.4, macOS 14.4, *)) {
            std::atomic<u32>& sleepers = Sleepers(&word);
            sleepers.fetch_add(1, std::memory_order_seq_cst);
            if (word.load(std::memory_order_seq_cst) == expected) {
                os_sync_wait_on_address(static_cast<void*>(&word), expected, sizeof(u32),
                                        OS_SYNC_WAIT_ON_ADDRESS_NONE);
            }
            sleepers.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
    }
#endif
    word.wait(expected, std::memory_order_relaxed);
}

void WakeOne(std::atomic<u32>& word) {
#ifdef _WIN32
    if (const AddressApi& api = Api(); api.wake_one != nullptr) {
        api.wake_one(static_cast<void*>(&word));
        return;
    }
#endif
#ifdef LICHII_OS_SYNC_WAIT
    if (UseOsSync()) {
        if (__builtin_available(iOS 17.4, macOS 14.4, *)) {
            if (HasSleepers(word)) {
                os_sync_wake_by_address_any(static_cast<void*>(&word), sizeof(u32),
                                            OS_SYNC_WAKE_BY_ADDRESS_NONE);
            }
            return;
        }
    }
#endif
    word.notify_one();
}

void WakeAll(std::atomic<u32>& word) {
#ifdef _WIN32
    if (const AddressApi& api = Api(); api.wake_all != nullptr) {
        api.wake_all(static_cast<void*>(&word));
        return;
    }
#endif
#ifdef LICHII_OS_SYNC_WAIT
    if (UseOsSync()) {
        if (__builtin_available(iOS 17.4, macOS 14.4, *)) {
            if (HasSleepers(word)) {
                os_sync_wake_by_address_all(static_cast<void*>(&word), sizeof(u32),
                                            OS_SYNC_WAKE_BY_ADDRESS_NONE);
            }
            return;
        }
    }
#endif
    word.notify_all();
}

} // namespace Common::Nx2AddressWait
