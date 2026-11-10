// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>

#include "common/logging.h"
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
#ifdef _WIN32
    zero_trim_thread = std::jthread{[this](std::stop_token stop) { TrimZeroPages(stop); }};
#endif
}

DeviceMemory::~DeviceMemory() = default;

#ifdef _WIN32
// Windows can't punch holes in the guest memory section, so pages holding only zeros (games clear
// their heaps while booting: Ocarina leaves ~7 GB) stay resident until they leave the working set.
// A pass comes every 5 s while they keep turning up, backing off to once a minute. It works in
// slices of a few MiB with a short sleep after each, on the efficiency cores, and only trims pages
// that were already zero on the pass before: whole passes at once took 1.4-4.7 s and lined up with
// the 0.6-0.7 s hitches of a game's first minute. EDEN_NX2_ZERO_TRIM=0 turns it off.
void DeviceMemory::TrimZeroPages(std::stop_token stop) {
    using namespace std::chrono_literals;
    Common::SetCurrentThreadName("ZeroTrim");
    if (const char* value = std::getenv("EDEN_NX2_ZERO_TRIM");
        value != nullptr && value[0] == '0') {
        LOG_INFO(HW_Memory, "Zero page trimming is off (EDEN_NX2_ZERO_TRIM=0)");
        return;
    }
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    Common::SetCurrentThreadToEfficiencyCores();
    constexpr size_t Busy = size_t{64} << 20;
    std::mutex mutex;
    std::condition_variable_any wake;
    const auto sleep = [&](auto duration) {
        std::unique_lock lock{mutex};
        wake.wait_for(lock, stop, duration, [] { return false; });
        return !stop.stop_requested();
    };
    Common::HostMemory::ZeroTrimState state;
    std::chrono::seconds interval = 5s;
    while (sleep(interval)) {
        const auto start = std::chrono::steady_clock::now();
        const auto [resident, trimmed] =
            buffer.TrimZeroPages(&state, [&] { return sleep(2ms); });
        if (stop.stop_requested()) {
            return;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        // Pages found zero for the first time are trimmed by the next pass if they stay zero.
        const size_t zero = state.zero_before.size() * 0x1000;
        if (trimmed < Busy && zero - trimmed < Busy) {
            interval = (std::min)(interval * 2, std::chrono::seconds{60});
            continue;
        }
        interval = 5s;
        LOG_INFO(HW_Memory,
                 "Trimmed {} MiB of all-zero guest pages out of {} MiB resident over {} ms; the "
                 "kernel zero-filled {} MiB so far",
                 trimmed >> 20, resident >> 20, ms, Common::HostMemory::ZeroFilledBytes() >> 20);
    }
}
#endif

} // namespace Core
