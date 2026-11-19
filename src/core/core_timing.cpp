// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <fmt/format.h>
#include "common/cpu_features.h"
#include "common/logging.h"
#include "common/perf_sched.h"
#include "common/thread_cpu_clock.h"

#ifdef _WIN32
#include <windows.h>
#undef CreateEvent
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#include "common/scope_exit.h"
#include "common/windows/timer_resolution.h"
#endif

#include "common/settings.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"

namespace Core::Timing {

constexpr s64 MAX_SLICE_LENGTH = 10000;

constexpr u32 CPU_CLOCK_BASE_MHZ = 1020;
constexpr u32 CPU_CLOCK_BOOST_MHZ = 1734;
constexpr u32 CPU_CLOCK_OVERCLOCK_MHZ = 2040;

constexpr u32 CpuClockTargetMhz(Settings::CpuClock clock) {
    switch (clock) {
    case Settings::CpuClock::Boost:
        return CPU_CLOCK_BOOST_MHZ;
    case Settings::CpuClock::Overclock:
        return CPU_CLOCK_OVERCLOCK_MHZ;
    default:
        return CPU_CLOCK_BASE_MHZ;
    }
}

namespace {

void PauseHint() noexcept {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

/// How long before a one-shot event the timer thread stops sleeping in the OS and waits the rest
/// in a hardware wait. OS sleeps end late by the host's timer tick (0.5 ms on Windows) plus
/// scheduling, so the margin follows the 90th percentile of how late the last Samples sleeps
/// ended. Hardware waits are paid from a budget of a fifth of the thread's time: with events
/// denser than the margin, waits fall back to OS sleeps rather than a busy core.
class HostWaitMargin {
public:
    static constexpr size_t Samples = 128;
    static constexpr size_t UpdateEvery = 16;
    static constexpr s64 MaxMarginNs = 2'000'000;
    static constexpr s64 BudgetDivisor = 5;
    static constexpr s64 BurstNs = 2'000'000;

    s64 Get(s64 now) {
        if (last_now) {
            const s64 elapsed = (std::max<s64>)(0, now - *last_now);
            budget_ns = (std::min)(BurstNs, budget_ns + elapsed / BudgetDivisor);
        }
        last_now = now;
        return std::clamp<s64>(budget_ns, 0, estimate_ns);
    }

    void OnOversleep(s64 late_ns) {
        samples[next] = std::clamp<s64>(late_ns, 0, MaxMarginNs);
        next = (next + 1) % Samples;
        count = (std::min)(count + 1, Samples);
        if (++since_update < UpdateEvery) {
            return;
        }
        since_update = 0;
        std::array<s64, Samples> sorted = samples;
        const auto p90 = sorted.begin() + count * 9 / 10;
        std::nth_element(sorted.begin(), p90, sorted.begin() + count);
        estimate_ns = *p90;
    }

    void OnPreciseWait(s64 ns) {
        budget_ns -= ns;
    }

    s64 Estimate() const {
        return estimate_ns;
    }

private:
    std::array<s64, Samples> samples{};
    size_t next{};
    size_t count{};
    size_t since_update{};
    s64 estimate_ns{};
    s64 budget_ns{BurstNs};
    std::optional<s64> last_now;
};

// EDEN_NX2_TIMING_SPIN_US overrides the adaptive margin in microseconds when set.
s64 HostWaitMarginOverrideNs() {
    static const s64 margin_ns = [] {
        const char* v = std::getenv("EDEN_NX2_TIMING_SPIN_US");
        return v != nullptr ? (std::max<s64>)(0, std::atoll(v) * 1000) : -1;
    }();
    return margin_ns;
}

// One second of the timer thread's waits, for EDEN_NX2_HOSTTIMING_STATS.
struct HostWaitWindow {
    s64 start{};
    std::optional<u64> cpu_start;
    u64 waits{};
    u64 woken{};
    s64 sleep_ns{};
    s64 precise_ns{};
    std::vector<s64> wake_late;
    std::vector<s64> event_late;
    std::vector<s64> one_shot_late;
};

thread_local HostWaitWindow t_host_wait;
thread_local HostWaitMargin t_host_wait_margin;
thread_local std::optional<s64> t_next_one_shot_time;

void LogHostWait(s64 now) {
    HostWaitWindow& w = t_host_wait;
    const auto percentiles = [](std::vector<s64>& late) -> std::array<s64, 3> {
        if (late.empty()) {
            return {};
        }
        std::sort(late.begin(), late.end());
        return {late[late.size() / 2] / 1000, late[late.size() * 99 / 100] / 1000,
                late.back() / 1000};
    };
    const auto wake = percentiles(w.wake_late);
    const auto event = percentiles(w.event_late);
    const auto one_shot = percentiles(w.one_shot_late);
    const f64 seconds = (std::max)(1e-9, static_cast<f64>(now - w.start) / 1e9);
    const std::optional<u64> cpu = Common::CurrentThreadCpuTimeNs();
    const s64 override_ns = HostWaitMarginOverrideNs();
    const s64 reported_margin_us =
        (override_ns >= 0 ? override_ns : t_host_wait_margin.Estimate()) / 1000;
    LOG_INFO(Core_Timing,
             "HostTiming wait: {} waits, {} woken by new events, OS sleep {:.1f} ms/s, "
             "precise wait {:.1f} ms/s, margin {} us; wake late p50/p99/max {}/{}/{} us; {} events "
             "late p50/p99/max {}/{}/{} us, {} of them one-shot p50/p99/max {}/{}/{} us; thread "
             "CPU {}",
             w.waits, w.woken, static_cast<f64>(w.sleep_ns) / 1e6 / seconds,
             static_cast<f64>(w.precise_ns) / 1e6 / seconds, reported_margin_us, wake[0], wake[1],
             wake[2], w.event_late.size(), event[0], event[1], event[2], w.one_shot_late.size(),
             one_shot[0], one_shot[1], one_shot[2],
             cpu && w.cpu_start
                 ? fmt::format("{:.1f}%", static_cast<f64>(*cpu - *w.cpu_start) / 1e7 / seconds)
                 : std::string{"unknown"});
    w.start = now;
    w.cpu_start = cpu;
    w.waits = w.woken = 0;
    w.sleep_ns = w.precise_ns = 0;
    w.wake_late.clear();
    w.event_late.clear();
    w.one_shot_late.clear();
}

} // Anonymous namespace

std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback) {
    return std::make_shared<EventType>(std::move(callback), std::move(name));
}

struct CoreTiming::Event {
    s64 time;
    u64 fifo_order;
    EventType* raw_type;
    std::weak_ptr<EventType> type;
    s64 reschedule_time;

    // Sort by time, unless the times are the same, in which case sort by
    // the order added to the queue
    friend bool operator>(const Event& left, const Event& right) noexcept {
        return std::tie(left.time, left.fifo_order) > std::tie(right.time, right.fifo_order);
    }

    friend bool operator<(const Event& left, const Event& right) noexcept {
        return std::tie(left.time, left.fifo_order) < std::tie(right.time, right.fifo_order);
    }
};

CoreTiming::CoreTiming() {
    event_queue.reserve(64);
}
CoreTiming::~CoreTiming() {
    Reset();
#ifdef _WIN32
    if (wake_handle != nullptr) {
        CloseHandle(static_cast<HANDLE>(wake_handle));
        wake_handle = nullptr;
    }
#endif
}

void CoreTiming::Wake() {
    event.Set();
#ifdef _WIN32
    if (wake_handle != nullptr) {
        SetEvent(static_cast<HANDLE>(wake_handle));
    }
#endif
}

void CoreTiming::Initialize(std::function<void()>&& on_thread_init_) {
    Reset();
    on_thread_init = std::move(on_thread_init_);
    event_fifo_id = 0;
    cpu_ticks = 0;
    if (const char* stats = std::getenv("EDEN_NX2_HOSTTIMING_STATS")) {
        host_wait_stats = std::strtol(stats, nullptr, 10) != 0;
    }
    if (is_multicore) {
#ifdef _WIN32
        if (wake_handle == nullptr) {
            wake_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        }
#endif
        timer_thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("HostTiming");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::VeryHigh);
            Common::SetCurrentThreadToPerformanceCores(false);
            on_thread_init();
            void* hires_timer = nullptr;
#ifdef _WIN32
            // EDEN_HIRES_CLOCK=0 or EDEN_NX2_TIMING_SLEEP=0 keeps the plain timed wait.
            const char* const hires_switch = std::getenv("EDEN_HIRES_CLOCK");
            const char* const sleep_switch = std::getenv("EDEN_NX2_TIMING_SLEEP");
            const bool use_hires_timer =
                (hires_switch == nullptr || hires_switch[0] != '0') &&
                (sleep_switch == nullptr || sleep_switch[0] != '0');
            if (use_hires_timer) {
                hires_timer = CreateWaitableTimerExW(
                    nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            }
            SCOPE_EXIT {
                if (hires_timer != nullptr) {
                    CloseHandle(static_cast<HANDLE>(hires_timer));
                }
            };
#endif
            has_started = true;

            // base frequency in MHz: 1ns (10^-9) = 1GHz (10^9)
            while (!stop_token.stop_requested()) {
                while (!paused && !stop_token.stop_requested()) {
                    paused_set = false;
                    if (auto const next_time = Advance(); next_time) {
                        // There are more events left in the queue, wait until the next event.
                        WaitForEvent(*next_time, hires_timer);
                    } else {
                        // Queue is empty, wait until another event is scheduled and signals us to
                        // continue.
                        wait_set = true;
                        event.Wait();
                    }
                    wait_set = false;
                }
                paused_set = true;
                pause_event.Wait();
            }
        });
    }
}

void CoreTiming::WaitForEvent(s64 next_time, void* hires_timer) {
    const std::optional<s64> one_shot_time = std::exchange(t_next_one_shot_time, std::nullopt);
    const s64 start = GetGlobalTimeNs().count();
    if (next_time <= start) {
        return;
    }
    const s64 override_ns = HostWaitMarginOverrideNs();
    const s64 margin_ns = override_ns >= 0 ? override_ns : t_host_wait_margin.Get(start);
    const bool precise = override_ns >= 0
                             ? (next_time - margin_ns <= start)
                             : (one_shot_time && *one_shot_time - margin_ns <= start);
    const s64 wake = override_ns >= 0
                         ? (std::max)(start, next_time - margin_ns)
                         : (one_shot_time ? (std::min)(next_time, *one_shot_time - margin_ns)
                                          : next_time);
    bool signalled = false;
    if (precise) {
#if defined(_WIN32) && defined(ARCHITECTURE_x86_64)
        if (Common::g_cpu_caps.monitorx || Common::g_cpu_caps.waitpkg) {
            signalled = event.WaitFor(std::chrono::nanoseconds{next_time - start});
            if (signalled && wake_handle != nullptr) {
                ResetEvent(static_cast<HANDLE>(wake_handle));
            }
        } else
#endif
        {
            while (true) {
                if (event.IsSet()) {
                    event.Reset();
#ifdef _WIN32
                    if (wake_handle != nullptr) {
                        ResetEvent(static_cast<HANDLE>(wake_handle));
                    }
#endif
                    signalled = true;
                    break;
                }
                if (GetGlobalTimeNs().count() >= next_time) {
                    break;
                }
                PauseHint();
            }
        }
    } else if (wake > start) {
#ifdef _WIN32
        // A high resolution waitable timer (~0.5 ms precise on Windows 10 1803+) keeps the guest's
        // events (vsync among them) on time even where Windows 11 ignores our timer resolution
        // request, without holding a logical processor in MWAITX for the whole wait.
        bool waited_hires = false;
        if (hires_timer != nullptr && wake_handle != nullptr) {
            if (event.IsSet()) {
                event.Reset();
                ResetEvent(static_cast<HANDLE>(wake_handle));
                signalled = true;
                waited_hires = true;
            } else {
                LARGE_INTEGER due{};
                due.QuadPart =
                    -(std::max<LONGLONG>)(1, static_cast<LONGLONG>((wake - start) / 100));
                if (SetWaitableTimerEx(static_cast<HANDLE>(hires_timer), &due, 0, nullptr, nullptr,
                                       nullptr, 0)) {
                    waited_hires = true;
                    const HANDLE handles[2]{static_cast<HANDLE>(wake_handle),
                                            static_cast<HANDLE>(hires_timer)};
                    const DWORD wait_res = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
                    if (wait_res == WAIT_OBJECT_0) {
                        event.Reset();
                        signalled = true;
                    } else if (event.IsSet()) {
                        event.Reset();
                        ResetEvent(static_cast<HANDLE>(wake_handle));
                        signalled = true;
                    }
                }
            }
        }
        if (!waited_hires) {
            signalled = event.SleepFor(std::chrono::nanoseconds{wake - start});
        }
#else
        (void)hires_timer;
        signalled = event.SleepFor(std::chrono::nanoseconds{wake - start});
#endif
    }
    const s64 end = GetGlobalTimeNs().count();
    if (override_ns < 0) {
        if (precise) {
            t_host_wait_margin.OnPreciseWait(end - start);
        } else if (!signalled && wake > start) {
            t_host_wait_margin.OnOversleep(end - wake);
        }
    }
    if (!signalled && (precise || wake == next_time) && Common::PerfSched::Enabled()) {
        Common::PerfSched::g_timer_late.Add(static_cast<u64>((std::max)(s64{0}, end - next_time)));
    }
    if (!host_wait_stats.load(std::memory_order_relaxed)) {
        return;
    }
    HostWaitWindow& w = t_host_wait;
    ++w.waits;
    w.woken += signalled;
    (precise ? w.precise_ns : w.sleep_ns) += end - start;
    if (!precise && !signalled) {
        w.wake_late.push_back(end - wake);
    }
    if (w.start == 0) {
        w.start = end;
        w.cpu_start = Common::CurrentThreadCpuTimeNs();
    } else if (end - w.start >= 1'000'000'000) {
        LogHostWait(end);
    }
}

void CoreTiming::ClearPendingEvents() {
    std::scoped_lock lock{advance_lock, basic_lock};
    event_queue.clear();
    t_next_one_shot_time.reset();
    armed_wait_time = (std::numeric_limits<s64>::max)();
    if (is_multicore) {
        Wake();
    }
}

void CoreTiming::Pause(bool is_paused) {
    paused = is_paused;
    pause_event.Set();

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

void CoreTiming::SyncPause(bool is_paused) {
    if (is_paused == paused && paused_set == paused) {
        return;
    }

    Pause(is_paused);
    if (timer_thread.joinable()) {
        if (!is_paused) {
            pause_event.Set();
        }
        Wake();
        while (paused_set != is_paused) {
            PauseHint();
        }
    }

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

bool CoreTiming::IsRunning() const {
    return !paused_set;
}

bool CoreTiming::HasPendingEvents() const {
    std::scoped_lock lock{basic_lock};
    return !(wait_set && event_queue.empty());
}

void CoreTiming::ScheduleEvent(std::chrono::nanoseconds ns_into_future,
                               const std::shared_ptr<EventType>& event_type, bool absolute_time) {
    ScheduleLoopingEvent(ns_into_future, std::chrono::nanoseconds{0}, event_type, absolute_time);
}

void CoreTiming::RescheduleEvent(std::chrono::nanoseconds ns_into_future,
                                 const std::shared_ptr<EventType>& event_type,
                                 bool absolute_time) {
    if (!event_type) {
        return;
    }
    bool should_wake = false;
    {
        std::scoped_lock scope{basic_lock};
        const s64 target_time =
            (absolute_time ? ns_into_future : GetGlobalTimeNs() + ns_into_future).count();
        const bool was_empty = event_queue.empty();
        const s64 prev_head =
            was_empty ? (std::numeric_limits<s64>::max)() : event_queue.front().time;

        ++event_type->sequence_number;
        const auto erased = std::erase_if(event_queue, [&](const Event& e) {
            return e.raw_type == event_type.get();
        });
        event_queue.push_back(
            Event{target_time, event_fifo_id++, event_type.get(), event_type, 0});
        if (erased > 0) {
            std::make_heap(event_queue.begin(), event_queue.end(), std::greater<>{});
        } else {
            std::push_heap(event_queue.begin(), event_queue.end(), std::greater<>{});
        }

        if (is_multicore &&
            (was_empty || target_time < prev_head || target_time < armed_wait_time)) {
            armed_wait_time = (std::min)(armed_wait_time, target_time);
            should_wake = true;
        }
    }

    if (should_wake) {
        Wake();
    }
}

void CoreTiming::ScheduleLoopingEvent(std::chrono::nanoseconds start_time,
                                      std::chrono::nanoseconds resched_time,
                                      const std::shared_ptr<EventType>& event_type,
                                      bool absolute_time) {
    bool should_wake = false;
    {
        std::scoped_lock scope{basic_lock};
        const s64 target_time =
            (absolute_time ? start_time : GetGlobalTimeNs() + start_time).count();
        const bool was_empty = event_queue.empty();
        const s64 prev_head =
            was_empty ? (std::numeric_limits<s64>::max)() : event_queue.front().time;

        event_queue.push_back(Event{target_time, event_fifo_id++, event_type.get(), event_type,
                                    resched_time.count()});
        std::push_heap(event_queue.begin(), event_queue.end(), std::greater<>{});

        if (is_multicore &&
            (was_empty || target_time < prev_head || target_time < armed_wait_time)) {
            armed_wait_time = (std::min)(armed_wait_time, target_time);
            should_wake = true;
        }
    }

    if (should_wake) {
        Wake();
    }
}

void CoreTiming::UnscheduleEvent(const std::shared_ptr<EventType>& event_type,
                                 UnscheduleEventType type) {
    if (!event_type) {
        return;
    }
    {
        std::scoped_lock lk{basic_lock};
        const auto erased = std::erase_if(event_queue, [&](const Event& e) {
            return e.raw_type == event_type.get();
        });
        if (erased > 0) {
            std::make_heap(event_queue.begin(), event_queue.end(), std::greater<>{});
        }
        event_type->sequence_number++;
    }

    // Force any in-progress events to finish
    if (type == UnscheduleEventType::Wait) {
        std::scoped_lock lk{advance_lock};
    }
}

static u64 GetNextTickCount(u64 next_ticks) {
    if (Settings::values.use_custom_cpu_ticks.GetValue()) {
        return Settings::values.cpu_ticks.GetValue();
    }
    return next_ticks;
}

void CoreTiming::AddTicks(u64 ticks_to_add) {
    const u64 ticks = GetNextTickCount(ticks_to_add);
    cpu_ticks += ticks;
    downcount -= static_cast<s64>(ticks);
}

void CoreTiming::Idle() {
    AddTicks(1000U);
}

void CoreTiming::ResetTicks() {
    downcount = MAX_SLICE_LENGTH;
}

u64 CoreTiming::GetClockTicks() const {
    u64 fres = is_multicore ? Common::g_wall_clock.GetCNTPCT() : Common::g_wall_clock.CPUTickToGuestCNTPCT(cpu_ticks);
    // Switch 2 SDKs convert ticks with a fixed 31.25 MHz, so a scaled counter speeds up every
    // tick-based time (Ocarina with the Dynamic FPS add-on ran ~1.7x fast on Boost).
    if (const u32 target = CpuClockTargetMhz(Settings::values.cpu_clock.GetValue());
        target != CPU_CLOCK_BASE_MHZ && !Settings::CurrentProgramIsSwitch2()) {
        fres = fres * target / CPU_CLOCK_BASE_MHZ;
    }
    if (::Settings::values.sync_core_speed.GetValue()) {
        auto const ticks = f64(fres);
        auto const speed_limit = f64(Settings::SpeedLimit()) * 0.01;
        return u64(ticks / speed_limit);
    }
    return fres;
}

u64 CoreTiming::GetGPUTicks() const {
    return is_multicore
        ? Common::g_wall_clock.GetGPUTick()
        : Common::WallClock::CPUTickToGPUTick(cpu_ticks);
}

std::optional<s64> CoreTiming::Advance() {
    std::scoped_lock lock{advance_lock, basic_lock};
    global_timer = GetGlobalTimeNs().count();

    while (!event_queue.empty() && event_queue.front().time <= global_timer) {
        std::pop_heap(event_queue.begin(), event_queue.end(), std::greater<>{});
        Event evt = std::move(event_queue.back());
        event_queue.pop_back();

        if (const auto event_type{evt.type.lock()}) {
            const auto evt_time = evt.time;
            const auto evt_sequence_num = event_type->sequence_number;
            const std::chrono::nanoseconds late{GetGlobalTimeNs().count() - evt_time};
            if (is_multicore && host_wait_stats.load(std::memory_order_relaxed)) {
                t_host_wait.event_late.push_back(late.count());
                if (evt.reschedule_time == 0) {
                    t_host_wait.one_shot_late.push_back(late.count());
                }
            }

            if (evt.reschedule_time == 0) {
                basic_lock.unlock();

                event_type->callback(evt_time, late);

                basic_lock.lock();
            } else {
                basic_lock.unlock();

                const auto new_schedule_time{event_type->callback(evt_time, late)};

                basic_lock.lock();

                if (evt_sequence_num != event_type->sequence_number) {
                    global_timer = GetGlobalTimeNs().count();
                    continue;
                }

                const auto next_schedule_time{new_schedule_time.has_value()
                                                  ? new_schedule_time.value().count()
                                                  : evt.reschedule_time};

                // If this event was scheduled into a pause, its time now is going to be way
                // behind. Re-set this event to continue from the end of the pause.
                auto next_time{evt_time + next_schedule_time};
                const s64 pause_end = pause_end_time.load(std::memory_order_relaxed);
                if (evt_time < pause_end) {
                    next_time = pause_end + next_schedule_time;
                }

                event_queue.push_back(Event{next_time, event_fifo_id++, event_type.get(), evt.type,
                                            next_schedule_time});
                std::push_heap(event_queue.begin(), event_queue.end(), std::greater<>{});
            }
        }

        global_timer = GetGlobalTimeNs().count();
    }

    if (!event_queue.empty()) {
        const s64 next = event_queue.front().time;
        std::optional<s64> one_shot;
        for (const Event& e : event_queue) {
            if (e.reschedule_time == 0 && (!one_shot || e.time < *one_shot)) {
                one_shot = e.time;
            }
        }
        t_next_one_shot_time = one_shot;
        armed_wait_time = next;
        return next;
    } else {
        t_next_one_shot_time.reset();
        armed_wait_time = (std::numeric_limits<s64>::max)();
        return std::nullopt;
    }
}

void CoreTiming::Reset() {
    paused = true;
    // Request the stop before waking the thread: woken first, it could take the wake-up, see no
    // stop request yet, and go back to waiting on pause_event for good (the join below then hung
    // the frontend at boot).
    if (timer_thread.joinable()) {
        timer_thread.request_stop();
    }
    pause_event.Set();
    Wake();
    if (timer_thread.joinable()) {
        timer_thread.join();
    }
    has_started = false;
}

/// @brief Returns current time in nanoseconds.
std::chrono::nanoseconds CoreTiming::GetGlobalTimeNs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeNS()
        : std::chrono::nanoseconds{Common::WallClock::CPUTickToNS(cpu_ticks)};
}

/// @brief Returns current time in microseconds.
std::chrono::microseconds CoreTiming::GetGlobalTimeUs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeUS()
        : std::chrono::microseconds{Common::WallClock::CPUTickToUS(cpu_ticks)};
}

#ifdef _WIN32
void CoreTiming::SetTimerResolutionNs(std::chrono::nanoseconds ns) {
    timer_resolution_ns = ns.count();
}
#endif

} // namespace Core::Timing
