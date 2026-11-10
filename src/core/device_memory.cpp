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
    std::mutex mutex;
    std::condition_variable_any wake;
    std::chrono::seconds interval = 5s;
    u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
    while (true) {
        // Host memory pressure (iOS) starts a pass at once.
        for (std::chrono::seconds waited = 0s; waited < interval; waited += 1s) {
            {
                std::unique_lock lock{mutex};
                wake.wait_for(lock, stop, 1s, [] { return false; });
            }
            if (stop.stop_requested()) {
                return;
            }
            if (const u32 now = Common::PlatformEvents::MemoryPressureGeneration();
                now != pressure) {
                pressure = now;
                break;
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const auto [resident, trimmed] = buffer.TrimZeroPages();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (trimmed < Busy) {
            interval = (std::min)(interval * 2, std::chrono::seconds{60});
            continue;
        }
        interval = 5s;
        LOG_INFO(HW_Memory,
                 "Trimmed {} MiB of all-zero guest pages out of {} MiB resident in {} ms; the kernel "
                 "zero-filled {} MiB so far",
                 trimmed >> 20, resident >> 20, ms, Common::HostMemory::ZeroFilledBytes() >> 20);
    }
}
#endif

} // namespace Core
