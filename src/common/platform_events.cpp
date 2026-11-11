// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <malloc/malloc.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include "common/host_memory.h"
#if defined(__APPLE__) && TARGET_OS_IPHONE
#include "common/ios_jit.h"
#endif
#include "common/logging.h"
#include "common/platform_events.h"

namespace Common::PlatformEvents {
namespace {
std::atomic<u32> g_pressure_generation{};
std::atomic<u64> g_low_water{};
std::atomic<long long> g_next_poll_ns{};
std::atomic<u32> g_display_max_fps{};

std::mutex g_listeners_mutex;
std::vector<std::pair<std::size_t, std::function<void()>>> g_listeners;
std::size_t g_next_listener_id = 1;
} // Anonymous namespace

void SignalMemoryPressure() {
    g_pressure_generation.fetch_add(1, std::memory_order_relaxed);
}

u32 MemoryPressureGeneration() {
    return g_pressure_generation.load(std::memory_order_relaxed);
}

u64 AvailableProcessMemory() {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return static_cast<u64>(os_proc_available_memory());
#else
    return 0;
#endif
}

void SetMemoryLowWater(u64 bytes) {
    g_low_water.store(bytes, std::memory_order_relaxed);
}

void PollMemoryPressure() {
    const u64 low_water = g_low_water.load(std::memory_order_relaxed);
    if (low_water == 0) {
        return;
    }
    const long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    long long next = g_next_poll_ns.load(std::memory_order_relaxed);
    if (now < next || !g_next_poll_ns.compare_exchange_strong(next, now + 250'000'000LL,
                                                              std::memory_order_relaxed)) {
        return;
    }
    const u64 available = AvailableProcessMemory();
    if (available != 0 && available < low_water) {
        LOG_WARNING(Common_Memory, "{} MiB left to this process (low water {} MiB): collecting",
                    available >> 20, low_water >> 20);
        SignalMemoryPressure();
        // Give the collectors a second before the next look.
        g_next_poll_ns.store(now + 1'000'000'000LL, std::memory_order_relaxed);
    }
}

ProcessMemory QueryProcessMemory() {
    ProcessMemory memory{};
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS) {
        memory.footprint = info.phys_footprint;
        memory.footprint_peak = info.ledger_phys_footprint_peak;
        memory.compressed = info.compressed;
        memory.reusable = info.reusable;
        memory.internal = info.internal;
        memory.purgeable_nonvolatile = static_cast<u64>(info.ledger_purgeable_nonvolatile);
        memory.external = info.external;
        if (count >= TASK_VM_INFO_REV3_COUNT) {
            memory.graphics = static_cast<u64>(info.ledger_tag_graphics_footprint +
                                               info.ledger_tag_graphics_footprint_compressed);
        }
    }
#elif defined(_WIN32)
    // The closest Windows has: everything resident, guest memory included.
    PROCESS_MEMORY_COUNTERS counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        memory.footprint = counters.WorkingSetSize;
        memory.footprint_peak = counters.PeakWorkingSetSize;
    }
#endif
    return memory;
}

std::string FootprintSplit([[maybe_unused]] const ProcessMemory& memory,
                           [[maybe_unused]] u64 left) {
#ifdef __APPLE__
    // Ledger-tagged guest memory is in the footprint once per resident or compressed page and
    // in none of the ledgers task_info reports; a file's pages are in neither.
    const u64 resident = HostMemory::ScannedResidentBytes();
    u64 guest = resident;
    std::string guest_text;
    switch (HostMemory::ScannedAccounting()) {
    case HostMemory::GuestAccounting::Once:
        guest_text = fmt::format("{} MiB (counted once)", resident >> 20);
        break;
    case HostMemory::GuestAccounting::PerMapping:
        guest_text = fmt::format("{} MiB or more (counted per mapping)", resident >> 20);
        break;
    case HostMemory::GuestAccounting::File:
        guest = 0;
        guest_text = fmt::format("0 MiB (in a file, {} MiB of it resident)", resident >> 20);
        break;
    case HostMemory::GuestAccounting::Uncounted:
        guest = 0;
        guest_text = "not counted yet";
        break;
    }
    u64 jit = 0;
    std::string jit_text = "none";
#if TARGET_OS_IPHONE
    // The pool's pages were all touched when it was prepared. Code is written through a second
    // mapping of them, which an untagged object's footprint may count again: that part is in
    // the rest.
    jit = IosJit::PreparedBytes();
    jit_text = fmt::format("{} MiB ({} handed out)", jit >> 20, IosJit::HandedOutBytes() >> 20);
#endif
    const u64 known = guest + jit + memory.graphics;
    return fmt::format("limit {} MiB | guest {}, JIT pool {}, graphics {} MiB, rest {} MiB",
                       left != 0 ? fmt::format("{}", (memory.footprint + left) >> 20) : "unknown",
                       guest_text, jit_text, memory.graphics >> 20,
                       (memory.footprint - (std::min)(memory.footprint, known)) >> 20);
#else
    return "not known on this OS";
#endif
}

u64 RelieveMallocPressure() {
#ifdef __APPLE__
    return static_cast<u64>(malloc_zone_pressure_relief(nullptr, 0));
#else
    return 0;
#endif
}

std::size_t AddBackgroundListener(std::function<void()> listener) {
    std::scoped_lock lock{g_listeners_mutex};
    const std::size_t id = g_next_listener_id++;
    g_listeners.emplace_back(id, std::move(listener));
    return id;
}

void RemoveBackgroundListener(std::size_t id) {
    std::scoped_lock lock{g_listeners_mutex};
    std::erase_if(g_listeners, [id](const auto& entry) { return entry.first == id; });
}

void NotifyEnteringBackground() {
    std::scoped_lock lock{g_listeners_mutex};
    for (const auto& entry : g_listeners) {
        entry.second();
    }
}

void SetDisplayMaxFps(u32 fps) {
    g_display_max_fps.store(fps, std::memory_order_relaxed);
}

u32 DisplayMaxFps() {
    return g_display_max_fps.load(std::memory_order_relaxed);
}

} // namespace Common::PlatformEvents
