// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/perf_sched.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include "common/string_util.h"
#elif defined(__APPLE__)
#include <cerrno>
#include <cstddef>
#include <mutex>
#include <thread>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_time.h>
#include <mach/processor_info.h>
#include <mach/thread_info.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>
#include <unistd.h>
#if __has_include(<libproc.h>)
#include <libproc.h>
#else
extern "C" int proc_pidinfo(int pid, int flavor, uint64_t arg, void* buffer, int buffersize);
#endif
#include "common/thread.h"
#endif

namespace Common::PerfSched {

namespace {

struct HostThread {
    u64 id{};
    std::string name;
    std::string qos; ///< QoS class and scheduler priority; the priority alone on Windows
    u64 cpu_ns{};
    std::optional<u64> performance_ns; ///< Of cpu_ns, on performance cores
    std::optional<u64> run_samples;    ///< Samples that found it running or waiting for a CPU
};

struct HostCpu {
    u64 busy{};
    u64 total{};
};

struct HostThreads {
    std::vector<HostThread> threads;
    u64 sample_ticks{};
    std::vector<HostCpu> cpus; ///< System-wide ticks per CPU
    int active_cpus{-1};
    int all_cpus{-1};
    std::string note; ///< What could not be read
};

#ifdef _WIN32

HostThreads ReadHostThreads() {
    HostThreads result;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return result;
    }
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description = reinterpret_cast<GetDescription>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"KernelBase.dll"), "GetThreadDescription")));
    const DWORD pid = GetCurrentProcessId();
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid) {
            continue;
        }
        const HANDLE thread =
            OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
        if (thread == nullptr) {
            continue;
        }
        FILETIME creation{}, exit{}, kernel{}, user{};
        if (GetThreadTimes(thread, &creation, &exit, &kernel, &user)) {
            HostThread host{};
            host.id = entry.th32ThreadID;
            host.cpu_ns = ((u64{kernel.dwHighDateTime} << 32 | kernel.dwLowDateTime) +
                           (u64{user.dwHighDateTime} << 32 | user.dwLowDateTime)) *
                          100;
            host.qos = fmt::format("priority {}", GetThreadPriority(thread));
            PWSTR description = nullptr;
            if (get_description != nullptr && SUCCEEDED(get_description(thread, &description)) &&
                description != nullptr) {
                host.name = Common::UTF16ToUTF8(std::wstring_view{description});
                LocalFree(description);
            }
            result.threads.push_back(std::move(host));
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
    return result;
}

#elif defined(__APPLE__)

// PROC_PIDTHREADCOUNTS (bsd/sys/proc_info.h): a thread's CPU time per performance level, the
// performance cores first. Not in the iOS SDK's headers.
struct ThreadCountsData {
    u64 instructions;
    u64 cycles;
    u64 user_time_mach;
    u64 system_time_mach;
    u64 energy_nj;
};
struct ThreadCounts {
    u16 len;
    u16 reserved0;
    u32 reserved1;
    ThreadCountsData counts[4];
};
constexpr int PidThreadCounts = 34;

u64 MachToNs(u64 ticks) {
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t value{};
        mach_timebase_info(&value);
        return value;
    }();
    return timebase.denom == 0 ? ticks : ticks * timebase.numer / timebase.denom;
}

const char* QosName(qos_class_t qos) {
    switch (qos) {
    case QOS_CLASS_USER_INTERACTIVE:
        return "user-interactive";
    case QOS_CLASS_USER_INITIATED:
        return "user-initiated";
    case QOS_CLASS_DEFAULT:
        return "default";
    case QOS_CLASS_UTILITY:
        return "utility";
    case QOS_CLASS_BACKGROUND:
        return "background";
    default:
        return "no QoS";
    }
}

// A thread's run state is only readable as it is now, so a helper thread samples every thread
// 20 times a second: the share of samples in the run state, less the thread's CPU time, is the
// time it waited for a CPU.
struct RunSampler {
    struct Entry {
        u64 id{};
        thread_act_t port{};
        u64 running{};
    };

    std::mutex mutex;
    std::vector<Entry> entries;
    u64 ticks{};
    std::jthread thread;

    void Loop(std::stop_token stop) {
        Common::SetCurrentThreadName("PerfSched");
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        const thread_act_t self = pthread_mach_thread_np(pthread_self());
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
            std::scoped_lock lock{mutex};
            ++ticks;
            for (Entry& entry : entries) {
                thread_basic_info_data_t info{};
                mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
                if (entry.port != self &&
                    thread_info(entry.port, THREAD_BASIC_INFO,
                                reinterpret_cast<thread_info_t>(&info), &count) == KERN_SUCCESS &&
                    info.run_state == TH_STATE_RUNNING && (info.flags & TH_FLAGS_IDLE) == 0) {
                    ++entry.running;
                }
            }
        }
    }
};

RunSampler& Sampler() {
    static RunSampler sampler;
    return sampler;
}

HostThreads ReadHostThreads() {
    HostThreads result;
    thread_act_array_t ports = nullptr;
    mach_msg_type_number_t port_count = 0;
    if (const kern_return_t status = task_threads(mach_task_self(), &ports, &port_count);
        status != KERN_SUCCESS) {
        result.note = fmt::format("task_threads failed ({})", status);
        return result;
    }
    int counts_error = 0;
    std::vector<RunSampler::Entry> entries;
    for (mach_msg_type_number_t i = 0; i < port_count; ++i) {
        const thread_act_t port = ports[i];
        thread_identifier_info_data_t identifier{};
        mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
        thread_extended_info_data_t extended{};
        mach_msg_type_number_t extended_count = THREAD_EXTENDED_INFO_COUNT;
        if (thread_info(port, THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&identifier),
                        &count) != KERN_SUCCESS ||
            thread_info(port, THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&extended),
                        &extended_count) != KERN_SUCCESS) {
            mach_port_deallocate(mach_task_self(), port);
            continue;
        }
        HostThread host{};
        host.id = identifier.thread_id;
        host.cpu_ns = extended.pth_user_time + extended.pth_system_time;
        extended.pth_name[sizeof(extended.pth_name) - 1] = '\0';
        host.name = extended.pth_name;
        qos_class_t qos = QOS_CLASS_UNSPECIFIED;
        int relative = 0;
        if (const pthread_t pthread = pthread_from_mach_thread_np(port); pthread != nullptr) {
            pthread_get_qos_class_np(pthread, &qos, &relative);
        }
        host.qos = fmt::format("{}/{}", QosName(qos), extended.pth_curpri);
        ThreadCounts counts{};
        const int size = proc_pidinfo(getpid(), PidThreadCounts, identifier.thread_id, &counts,
                                      static_cast<int>(sizeof(counts)));
        if (size >= static_cast<int>(offsetof(ThreadCounts, counts) + 2 * sizeof(ThreadCountsData)) &&
            counts.len >= 2) {
            host.performance_ns =
                MachToNs(counts.counts[0].user_time_mach + counts.counts[0].system_time_mach);
        } else if (counts_error == 0) {
            counts_error = size <= 0 ? errno : -1;
        }
        entries.push_back({.id = host.id, .port = port, .running = 0});
        result.threads.push_back(std::move(host));
    }
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(ports),
                  port_count * sizeof(thread_act_t));
    if (counts_error != 0) {
        result.note = fmt::format("time per performance level unreadable (proc_pidinfo {}: {})",
                                  PidThreadCounts,
                                  counts_error < 0 ? std::string{"short reply"}
                                                   : fmt::format("errno {}", counts_error));
    }

    // Hand the sampler the new ports, keeping each thread's count.
    RunSampler& sampler = Sampler();
    {
        std::scoped_lock lock{sampler.mutex};
        for (RunSampler::Entry& entry : entries) {
            const auto old = std::ranges::find(sampler.entries, entry.id, &RunSampler::Entry::id);
            if (old != sampler.entries.end()) {
                entry.running = old->running;
            }
        }
        for (const RunSampler::Entry& old : sampler.entries) {
            mach_port_deallocate(mach_task_self(), old.port);
        }
        sampler.entries = std::move(entries);
        result.sample_ticks = sampler.ticks;
        for (HostThread& host : result.threads) {
            const auto entry = std::ranges::find(sampler.entries, host.id, &RunSampler::Entry::id);
            if (entry != sampler.entries.end()) {
                host.run_samples = entry->running;
            }
        }
        if (!sampler.thread.joinable()) {
            sampler.thread = std::jthread{[&sampler](std::stop_token stop) { sampler.Loop(stop); }};
        }
    }

    static const host_t host_port = mach_host_self();
    natural_t cpu_count = 0;
    processor_info_array_t info = nullptr;
    mach_msg_type_number_t info_count = 0;
    if (const kern_return_t status = host_processor_info(host_port, PROCESSOR_CPU_LOAD_INFO,
                                                         &cpu_count, &info, &info_count);
        status == KERN_SUCCESS) {
        const auto* const load = reinterpret_cast<const processor_cpu_load_info_data_t*>(info);
        for (natural_t cpu = 0; cpu < cpu_count; ++cpu) {
            const u64 busy = u64{load[cpu].cpu_ticks[CPU_STATE_USER]} +
                             load[cpu].cpu_ticks[CPU_STATE_SYSTEM] +
                             load[cpu].cpu_ticks[CPU_STATE_NICE];
            result.cpus.push_back({busy, busy + load[cpu].cpu_ticks[CPU_STATE_IDLE]});
        }
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(info),
                      info_count * sizeof(integer_t));
    } else {
        result.note += fmt::format("{}host_processor_info failed ({})",
                                   result.note.empty() ? "" : "; ", status);
    }
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname("hw.activecpu", &value, &size, nullptr, 0) == 0) {
        result.active_cpus = value;
    }
    size = sizeof(value);
    if (sysctlbyname("hw.ncpu", &value, &size, nullptr, 0) == 0) {
        result.all_cpus = value;
    }
    return result;
}

#else

HostThreads ReadHostThreads() {
    return {};
}

#endif

struct LatencySnapshot {
    u64 count{};
    u64 sum_ns{};
    std::array<u32, Latency::Buckets> buckets{};
};

LatencySnapshot ReadLatency(const Latency& latency) {
    LatencySnapshot result;
    result.count = latency.count.load(std::memory_order_relaxed);
    result.sum_ns = latency.sum_ns.load(std::memory_order_relaxed);
    for (size_t i = 0; i < Latency::Buckets; ++i) {
        result.buckets[i] = latency.buckets[i].load(std::memory_order_relaxed);
    }
    return result;
}

std::string LatencyText(Latency& latency, LatencySnapshot& previous, double seconds) {
    const LatencySnapshot now = ReadLatency(latency);
    const u64 max_ns = latency.max_ns.exchange(0, std::memory_order_relaxed);
    const u64 count = now.count - previous.count;
    std::string text = fmt::format("{:.0f}/s", static_cast<double>(count) / seconds);
    if (count != 0) {
        // The 99th percentile as its bucket's upper bound.
        const u64 wanted = count - count / 100;
        u64 seen = 0;
        u64 p99_ns = max_ns;
        for (size_t i = 0; i < Latency::Buckets; ++i) {
            seen += now.buckets[i] - previous.buckets[i];
            if (seen >= wanted) {
                p99_ns = (std::min)(Latency::BucketEndNs(i), (std::max)(max_ns, u64{1}));
                break;
            }
        }
        text += fmt::format(", average {:.0f} us, 99% within {:.0f} us, longest {:.0f} us",
                            static_cast<double>(now.sum_ns - previous.sum_ns) / 1e3 /
                                static_cast<double>(count),
                            static_cast<double>(p99_ns) / 1e3, static_cast<double>(max_ns) / 1e3);
    }
    previous = now;
    return text;
}

struct GuestThreadSnapshot {
    u64 run_ns{};
    u64 ready_ns{};
    u64 switches{};
    u64 core_changes{};
};

struct State {
    std::chrono::steady_clock::time_point last{};
    u64 frames{};
    std::array<u64, Counter::Count> counters{};
    LatencySnapshot core_wakes;
    LatencySnapshot timer_late;
    std::array<GuestThreadSnapshot, GuestThreads> guest{};
    HostThreads host;
};

std::string HostThreadsText(const HostThreads& now, const HostThreads& previous, double seconds) {
    struct Row {
        const HostThread* thread{};
        double cpu{};
        std::optional<double> performance;
        std::optional<double> waiting;
    };
    const double ticks = static_cast<double>(now.sample_ticks - previous.sample_ticks);
    std::vector<Row> rows;
    double all_cpu = 0;
    double all_performance_ns = 0;
    double all_cpu_ns = 0;
    double all_waiting = 0;
    for (const HostThread& thread : now.threads) {
        const auto old = std::ranges::find(previous.threads, thread.id, &HostThread::id);
        const bool known = old != previous.threads.end();
        const u64 cpu_ns = thread.cpu_ns - (known ? (std::min)(old->cpu_ns, thread.cpu_ns) : 0);
        Row row{.thread = &thread, .cpu = static_cast<double>(cpu_ns) / 1e7 / seconds};
        if (thread.performance_ns && (!known || old->performance_ns)) {
            const u64 before = known ? *old->performance_ns : 0;
            const u64 performance_ns = *thread.performance_ns - (std::min)(before, *thread.performance_ns);
            all_performance_ns += static_cast<double>(performance_ns);
            all_cpu_ns += static_cast<double>(cpu_ns);
            if (cpu_ns != 0) {
                row.performance = (std::min)(
                    100.0, 100.0 * static_cast<double>(performance_ns) / static_cast<double>(cpu_ns));
            }
        }
        if (thread.run_samples && ticks > 0) {
            const u64 before = known && old->run_samples ? *old->run_samples : 0;
            const double in_run_state =
                100.0 * static_cast<double>(*thread.run_samples - (std::min)(before, *thread.run_samples)) /
                ticks;
            row.waiting = (std::max)(0.0, in_run_state - row.cpu);
            all_waiting += *row.waiting;
        }
        all_cpu += row.cpu;
        rows.push_back(row);
    }
    std::ranges::sort(rows, [](const Row& a, const Row& b) {
        return a.cpu + a.waiting.value_or(0.0) > b.cpu + b.waiting.value_or(0.0);
    });
    std::string text = fmt::format("{} threads, CPU {:.0f}%", now.threads.size(), all_cpu);
    if (all_cpu_ns > 0) {
        text += fmt::format(", {:.0f}% of it on performance cores", 100.0 * all_performance_ns / all_cpu_ns);
    }
    if (ticks > 0) {
        text += fmt::format(", waiting for a CPU {:.0f}% ({:.0f} samples)", all_waiting, ticks);
    }
    constexpr size_t Shown = 10;
    double other_cpu = 0;
    double other_waiting = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& row = rows[i];
        if (i >= Shown) {
            other_cpu += row.cpu;
            other_waiting += row.waiting.value_or(0.0);
            continue;
        }
        text += fmt::format(" | {} ({}): CPU {:.0f}%",
                            row.thread->name.empty() ? fmt::format("thread {}", row.thread->id)
                                                     : row.thread->name,
                            row.thread->qos, row.cpu);
        if (row.performance) {
            text += fmt::format(", {:.0f}% on performance cores", *row.performance);
        }
        if (row.waiting) {
            text += fmt::format(", waiting {:.0f}%", *row.waiting);
        }
    }
    if (rows.size() > Shown) {
        text += fmt::format(" | {} others: CPU {:.0f}%", rows.size() - Shown, other_cpu);
        if (ticks > 0) {
            text += fmt::format(", waiting {:.0f}%", other_waiting);
        }
    }
    if (!now.cpus.empty() && now.cpus.size() == previous.cpus.size()) {
        text += " | every process, busy % per CPU:";
        for (size_t cpu = 0; cpu < now.cpus.size(); ++cpu) {
            const u64 total = now.cpus[cpu].total - previous.cpus[cpu].total;
            const u64 busy = now.cpus[cpu].busy - previous.cpus[cpu].busy;
            text += total == 0 ? std::string{" -"}
                               : fmt::format(" {:.0f}", 100.0 * static_cast<double>(busy) /
                                                            static_cast<double>(total));
        }
        if (const u32 first = PerfReport::HighCpuFirst(); first != 0) {
            text += fmt::format(" (performance from CPU {})", first);
        }
    }
    if (now.active_cpus >= 0) {
        text += fmt::format(" | {} of {} CPUs active", now.active_cpus, now.all_cpus);
    }
    if (!now.note.empty()) {
        text += " | " + now.note;
    }
    return text;
}

std::string GuestThreadsText(State& state, double seconds) {
    struct Row {
        s32 slot{};
        u64 run_ns{};
        u64 ready_ns{};
        u64 ready_max_ns{};
        u64 switches{};
        u64 core_changes{};
        u32 cores{};
    };
    const s32 count = static_cast<s32>((std::min)(
        g_guest_thread_count.load(std::memory_order_relaxed), static_cast<u32>(GuestThreads)));
    std::vector<Row> rows;
    u64 app_ready_ns = 0;
    u64 app_switches = 0;
    u64 app_core_changes = 0;
    u64 app_run_ns = 0;
    u64 other_run_ns = 0;
    for (s32 slot = 0; slot < count; ++slot) {
        GuestThread& thread = g_guest_threads[static_cast<size_t>(slot)];
        GuestThreadSnapshot& previous = state.guest[static_cast<size_t>(slot)];
        const GuestThreadSnapshot now{
            .run_ns = thread.run_ns.load(std::memory_order_relaxed),
            .ready_ns = thread.ready_ns.load(std::memory_order_relaxed),
            .switches = thread.switches.load(std::memory_order_relaxed),
            .core_changes = thread.core_changes.load(std::memory_order_relaxed),
        };
        const Row row{
            .slot = slot,
            .run_ns = now.run_ns - previous.run_ns,
            .ready_ns = now.ready_ns - previous.ready_ns,
            .ready_max_ns = thread.ready_max_ns.exchange(0, std::memory_order_relaxed),
            .switches = now.switches - previous.switches,
            .core_changes = now.core_changes - previous.core_changes,
            .cores = thread.cores.exchange(0, std::memory_order_relaxed),
        };
        previous = now;
        if (!thread.application.load(std::memory_order_relaxed)) {
            other_run_ns += row.run_ns;
            continue;
        }
        app_run_ns += row.run_ns;
        app_ready_ns += row.ready_ns;
        app_switches += row.switches;
        app_core_changes += row.core_changes;
        if (row.switches != 0) {
            rows.push_back(row);
        }
    }
    std::ranges::sort(rows, [](const Row& a, const Row& b) { return a.run_ns > b.run_ns; });
    const auto percent = [&](u64 ns) { return static_cast<double>(ns) / 1e7 / seconds; };
    const auto per_second = [&](u64 value) { return static_cast<double>(value) / seconds; };
    std::string text = fmt::format(
        "game threads: {} ran, on a core {:.0f}%, ready without one {:.0f}%, {:.0f} switches/s, "
        "{:.0f} core changes/s; other guest threads on a core {:.0f}%",
        rows.size(), percent(app_run_ns), percent(app_ready_ns), per_second(app_switches),
        per_second(app_core_changes), percent(other_run_ns));
    // The busiest five, then the three of the rest that changed core most.
    // EDEN_NX2_PERF_SCHED_THREADS=<n> lists the busiest n instead.
    static const size_t busiest = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_SCHED_THREADS");
        return value != nullptr ? static_cast<size_t>(std::strtoul(value, nullptr, 10)) : size_t{5};
    }();
    const size_t shown = busiest + 3;
    if (rows.size() > busiest) {
        std::ranges::sort(rows.begin() + static_cast<std::ptrdiff_t>(busiest), rows.end(),
                          [](const Row& a, const Row& b) { return a.core_changes > b.core_changes; });
    }
    for (size_t i = 0; i < rows.size() && i < shown; ++i) {
        const Row& row = rows[i];
        if (i >= busiest && row.core_changes == 0) {
            break;
        }
        const GuestThread& thread = g_guest_threads[static_cast<size_t>(row.slot)];
        std::string cores;
        for (u32 core = 0; core < 8; ++core) {
            if ((row.cores >> core & 1) != 0) {
                cores += static_cast<char>('0' + core);
            }
        }
        text += fmt::format(
            " | thread {} priority {}: on a core {:.0f}%, ready {:.1f}% (longest {:.1f} ms), "
            "{:.0f} switches/s, {:.0f} core changes/s, cores {}",
            thread.id.load(std::memory_order_relaxed),
            thread.priority.load(std::memory_order_relaxed), percent(row.run_ns),
            percent(row.ready_ns), static_cast<double>(row.ready_max_ns) / 1e6,
            per_second(row.switches), per_second(row.core_changes), cores);
    }
    return text;
}

} // Anonymous namespace

void TickFrame() {
    if (!PerfReport::Enabled()) {
        return;
    }
    static State state;
    ++state.frames;
    const auto now = std::chrono::steady_clock::now();
    const bool first = state.last == std::chrono::steady_clock::time_point{};
    const double seconds = std::chrono::duration<double>(now - state.last).count();
    if (!first && seconds < 10.0) {
        return;
    }
    std::array<u64, Counter::Count> counters{};
    for (size_t i = 0; i < counters.size(); ++i) {
        counters[i] = g_counters[i].load(std::memory_order_relaxed);
    }
    HostThreads host = ReadHostThreads();
    if (first) {
        state.core_wakes = ReadLatency(g_core_wakes);
        state.timer_late = ReadLatency(g_timer_late);
        static_cast<void>(GuestThreadsText(state, 1.0));
    } else {
        const auto delta = [&](Counter counter) { return counters[counter] - state.counters[counter]; };
        const auto per_second = [&](Counter counter) {
            return static_cast<double>(delta(counter)) / seconds;
        };
        const double frames = static_cast<double>((std::max)(state.frames, u64{1}));
        const auto per_frame = [&](Counter counter) {
            return static_cast<double>(delta(counter)) / frames;
        };
        const double presents = static_cast<double>((std::max)(delta(Presents), u64{1}));
        const auto per_present = [&](Counter counter) {
            return static_cast<double>(delta(counter)) / presents;
        };
        if (!host.threads.empty()) {
            LOG_INFO(Common_Memory, "Perf threads: {}", HostThreadsText(host, state.host, seconds));
        }
        const s32 backoff = g_yield_backoff.load(std::memory_order_relaxed);
        LOG_INFO(Common_Memory,
                 "Perf sched: {} | guest core wakes {} | timer thread late {} | waits on the GPU "
                 "per frame: {:.1f} syncpoint events armed ({:.1f} more already reached), {:.2f} ms "
                 "armed to signalled; {:.1f} presents/s, each {:.2f} ms for the frame's fence, {:.2f} "
                 "ms for the vblank | yield backoff {}: runs started {:.0f}/s after a real yield or none, "
                 "{:.0f}/s after a gap, {:.0f}/s after the registers changed; {:.0f}/s yields "
                 "under the gate; real yields {:.0f}/s",
                 GuestThreadsText(state, seconds), LatencyText(g_core_wakes, state.core_wakes, seconds),
                 LatencyText(g_timer_late, state.timer_late, seconds), per_frame(SyncptArmed),
                 per_frame(SyncptReached), per_frame(SyncptWaitNs) / 1e6, per_second(Presents),
                 per_present(PresentFenceNs) / 1e6, per_present(PresentPaceNs) / 1e6,
                 backoff < 0 ? "not reached" : backoff != 0 ? "on" : "off",
                 per_second(YieldRunStarts), per_second(YieldGapResets), per_second(YieldRegResets),
                 per_second(YieldUnderGate), per_second(YieldRealYields));
    }
    state.host = std::move(host);
    state.counters = counters;
    state.frames = 0;
    state.last = now;
}

} // namespace Common::PerfSched
