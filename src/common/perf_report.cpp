// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/perf_report.h"

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <notify.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#if __has_include(<libproc.h>)
#include <libproc.h>
#else
extern "C" int proc_pid_rusage(int pid, int flavor, rusage_info_t* buffer);
#endif
#endif

namespace Common::PerfReport {

namespace {
std::array<std::atomic<u64>, PlacementCores> g_placement_samples{};
std::array<std::atomic<u64>, PlacementCores> g_placement_high{};
} // Anonymous namespace

std::optional<ProcessCpu> ReadProcessCpu() {
#if defined(__APPLE__) && defined(RUSAGE_INFO_V6)
    rusage_info_v6 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V6, reinterpret_cast<rusage_info_t*>(&info)) != 0) {
        return std::nullopt;
    }
    // The times are in Mach absolute time units.
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t value{};
        mach_timebase_info(&value);
        return value;
    }();
    const auto to_ns = [](u64 ticks) {
        return timebase.denom == 0 ? ticks : ticks * timebase.numer / timebase.denom;
    };
    return ProcessCpu{
        .cpu_ns = to_ns(info.ri_user_time + info.ri_system_time),
        .performance_ns = to_ns(info.ri_user_ptime + info.ri_system_ptime),
        .energy_nj = info.ri_energy_nj,
    };
#else
    return std::nullopt;
#endif
}

u32 HighCpuFirst() {
#ifdef __APPLE__
    static const u32 first = [] {
        int levels = 0;
        size_t size = sizeof(levels);
        if (sysctlbyname("hw.nperflevels", &levels, &size, nullptr, 0) != 0 || levels < 2) {
            return 0U;
        }
        int efficiency = 0;
        size = sizeof(efficiency);
        if (sysctlbyname("hw.perflevel1.logicalcpu", &efficiency, &size, nullptr, 0) != 0) {
            return 0U;
        }
        return static_cast<u32>(efficiency);
    }();
    return first;
#else
    return 0;
#endif
}

int ThermalPressure() {
#ifdef __APPLE__
    static const int token = [] {
        int value = 0;
        return notify_register_check("com.apple.system.thermalpressurelevel", &value) ==
                       NOTIFY_STATUS_OK
                   ? value
                   : -1;
    }();
    u64 state = 0;
    if (token < 0 || notify_get_state(token, &state) != NOTIFY_STATUS_OK) {
        return -1;
    }
    return static_cast<int>(state);
#else
    return -1;
#endif
}

void SampleGuestCorePlacement(std::size_t core) {
#ifdef __APPLE__
    if (!Enabled() || core >= PlacementCores) {
        return;
    }
    thread_local u32 calls = 0;
    if ((++calls & 63) != 0) {
        return;
    }
    if (__builtin_available(iOS 14.2, macOS 11.0, *)) {
        size_t cpu = 0;
        if (pthread_cpu_number_np(&cpu) != 0) {
            return;
        }
        g_placement_samples[core].fetch_add(1, std::memory_order_relaxed);
        const u32 first = HighCpuFirst();
        if (first != 0 && cpu >= first) {
            g_placement_high[core].fetch_add(1, std::memory_order_relaxed);
        }
    }
#else
    static_cast<void>(core);
#endif
}

Placement ReadPlacement() {
    Placement placement{};
    for (std::size_t i = 0; i < PlacementCores; ++i) {
        placement.samples[i] = g_placement_samples[i].load(std::memory_order_relaxed);
        placement.high[i] = g_placement_high[i].load(std::memory_order_relaxed);
    }
    return placement;
}

} // namespace Common::PerfReport
