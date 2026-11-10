// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#if defined(_WIN32) || defined(__APPLE__)
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "common/logging.h"
#include "common/platform_events.h"
#include "common/thread.h"
#endif

#include "core/device_memory.h"
#include "hle/kernel/board/nintendo/nx/k_system_control.h"

namespace Core {

#ifdef HAS_NCE
constexpr size_t VirtualReserveSize = 1ULL << 38;
#else
constexpr size_t VirtualReserveSize = 1ULL << 39;
#endif

DeviceMemory::DeviceMemory()
    : buffer{Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize(),
             VirtualReserveSize} {
#if defined(_WIN32) || defined(__APPLE__)
    zero_trim_thread = std::jthread{[this](std::stop_token stop) { TrimZeroPages(stop); }};
#endif
}

DeviceMemory::~DeviceMemory() = default;

#if defined(_WIN32) || defined(__APPLE__)
// Windows can't punch holes in the guest memory section, so pages holding only zeros (games clear
// their heaps while booting: Ocarina leaves ~7 GB) stay resident until they leave the working set.
// A pass comes every 5 s while they keep turning up, backing off to once a minute. Apple counts
// them in the footprint iOS kills the app over, compressed or not.
void DeviceMemory::TrimZeroPages(std::stop_token stop) {
    using namespace std::chrono_literals;
    Common::SetCurrentThreadName("ZeroTrim");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    constexpr size_t Busy = size_t{64} << 20;
    // Footprint growth since the last pass that starts the next one early (Apple).
    constexpr u64 Growth = u64{256} << 20;
    // Under this much left to the process (iOS), passes stay 5 s apart and are always logged.
    constexpr u64 Tight = u64{1} << 30;
    std::mutex mutex;
    std::condition_variable_any wake;
    std::chrono::seconds interval = 5s;
    u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
    u64 footprint_after_pass = Common::PlatformEvents::QueryProcessMemory().footprint;
    bool reported_boot = false;
    while (true) {
        // Host memory pressure (iOS) starts a pass at once; a footprint that grows (a game
        // clearing its heap while booting) one 2 s later, rather than after the backed-off
        // interval.
        for (std::chrono::seconds waited = 0s; waited < interval; waited += 1s) {
            {
                std::unique_lock lock{mutex};
                wake.wait_for(lock, stop, 1s, [] { return false; });
            }
            if (stop.stop_requested()) {
                return;
            }
            // The caches poll the low-water mark once a frame; booting and loading draw few.
            Common::PlatformEvents::PollMemoryPressure();
            if (const u32 now = Common::PlatformEvents::MemoryPressureGeneration();
                now != pressure) {
                pressure = now;
                break;
            }
            if (waited >= 1s && Common::PlatformEvents::QueryProcessMemory().footprint >
                                    footprint_after_pass + Growth) {
                break;
            }
        }
        const auto before = Common::PlatformEvents::QueryProcessMemory();
        const auto start = std::chrono::steady_clock::now();
        const auto [resident, trimmed] = buffer.TrimZeroPages();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        const auto after = Common::PlatformEvents::QueryProcessMemory();
        footprint_after_pass = after.footprint;
        const u64 dropped =
            before.footprint > after.footprint ? before.footprint - after.footprint : 0;
        const u64 available = Common::PlatformEvents::AvailableProcessMemory();
        const bool tight = available != 0 && available < Tight;
        const bool busy = trimmed >= Busy || dropped >= Busy;
        // The first pass after a game boots is always logged.
        const bool boot = !reported_boot && resident >= Growth;
        reported_boot = reported_boot || boot;
        if (!busy && !tight) {
            interval = (std::min)(interval * 2, std::chrono::seconds{60});
            if (!boot) {
                continue;
            }
        } else {
            interval = 5s;
        }
        if (before.footprint == 0) {
            LOG_INFO(HW_Memory,
                     "Trimmed {} MiB of all-zero guest pages out of {} MiB resident in {} ms; the "
                     "kernel zero-filled {} MiB so far",
                     trimmed >> 20, resident >> 20, ms, Common::HostMemory::ZeroFilledBytes() >> 20);
            continue;
        }
        LOG_INFO(HW_Memory,
                 "Trimmed {} MiB of all-zero guest pages out of {} MiB resident in {} ms: footprint "
                 "{} -> {} MiB (reusable {} MiB), {} MiB left; guest clears so far: {} MiB "
                 "memset, {} MiB without faulting",
                 trimmed >> 20, resident >> 20, ms, before.footprint >> 20, after.footprint >> 20,
                 after.reusable >> 20, available >> 20,
                 Common::HostMemory::ZeroFilledBytes() >> 20,
                 Common::HostMemory::ZeroAdvisedBytes() >> 20);
    }
}
#endif

} // namespace Core
