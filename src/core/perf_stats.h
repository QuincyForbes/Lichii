// SPDX-FileCopyrightText: 2017 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <vector>
#include "common/common_types.h"

namespace Core {

struct PerfStatsResults {
    /// System FPS (LCD VBlanks) in Hz
    double system_fps;
    /// Average game FPS (GPU frame renders) in Hz
    double average_game_fps;
    /// Average generated FPS (FrameGen interpolated frames) in Hz
    double generated_fps;
    /// Total displayed FPS (real game frames + generated frames) in Hz
    double total_fps;
    /// Walltime per system frame, in seconds, excluding any waits
    double frametime;
    /// Ratio of walltime / emulated time elapsed
    double emulation_speed;
};

/// Totals since the game started. Unlike GetAndResetStats, reading them changes nothing, so any
/// number of readers can take rates from the difference of two readings.
struct PerfCounters {
    std::chrono::steady_clock::time_point time;
    std::chrono::microseconds system_time_us;
    u64 game_frames;
    u64 generated_frames;
    u64 system_frames;
    /// Walltime of those system frames, excluding any waits
    std::chrono::steady_clock::duration system_frametime;
};

/**
 * Class to manage and query performance/timing statistics. All public functions of this class are
 * thread-safe unless stated otherwise.
 */
class PerfStats {
public:
    explicit PerfStats(u64 title_id_);
    ~PerfStats();

    using Clock = std::chrono::steady_clock;

    void BeginSystemFrame();
    void EndSystemFrame();
    void EndGameFrame(u32 generated_count = 0);

    PerfStatsResults GetAndResetStats(std::chrono::microseconds current_system_time_us);

    PerfCounters GetCounters(std::chrono::microseconds current_system_time_us) const;

    /**
     * Returns the arithmetic mean of all frametime values stored in the performance history.
     */
    double GetMeanFrametime() const;

    /**
     * Gets the ratio between walltime and the emulated time of the previous system frame. This is
     * useful for scaling inputs or outputs moving between the two time domains.
     */
    double GetLastFrameTimeScale() const;

private:
    mutable std::mutex object_mutex;

    /// Title ID for the game that is running. 0 if there is no game running yet
    u64 title_id{0};
    /// Number of frame times counted, at most an hour of them
    std::size_t current_index{0};
    /// Sum of the counted frame times after the first IgnoreFrames
    double history_sum{};
    /// The counted frame times, kept only while record_frame_times is on (written to a file at the
    /// end, useful for tracking performance regressions with code changes)
    std::vector<double> perf_history;

    /// Point when the cumulative counters were reset
    Clock::time_point reset_point = Clock::now();
    /// System time when the cumulative counters were reset
    std::chrono::microseconds reset_point_system_us{0};

    /// Cumulative duration (excluding v-sync/frame-limiting) of frames since last reset
    Clock::duration accumulated_frametime = Clock::duration::zero();
    /// Cumulative number of system frames (LCD VBlanks) presented since last reset
    u32 system_frames = 0;
    /// Cumulative number of game frames (GSP frame submissions) since last reset
    std::atomic<u32> game_frames = 0;
    /// Cumulative number of generated frames (FrameGen) since last reset
    std::atomic<u32> generated_frames = 0;

    /// Totals for GetCounters, never reset
    std::atomic<u64> total_game_frames = 0;
    std::atomic<u64> total_generated_frames = 0;
    u64 total_system_frames = 0;
    Clock::duration total_frametime = Clock::duration::zero();

    /// Point when the previous system frame ended
    Clock::time_point previous_frame_end = reset_point;
    /// Point when the current system frame began
    Clock::time_point frame_begin = reset_point;
    /// Total visible duration (including frame-limiting, etc.) of the previous system frame
    Clock::duration previous_frame_length = Clock::duration::zero();
    /// Previously computed fps
    double previous_fps = 0;
    /// Previously computed generated fps
    double previous_generated_fps = 0;
};

class SpeedLimiter {
public:
    using Clock = std::chrono::steady_clock;

    void DoSpeedLimiting(std::chrono::microseconds current_system_time_us);

private:
    /// Emulated system time (in microseconds) at the last limiter invocation
    std::chrono::microseconds previous_system_time_us{0};
    /// Walltime at the last limiter invocation
    Clock::time_point previous_walltime = Clock::now();

    /// Accumulated difference between walltime and emulated time
    std::chrono::microseconds speed_limiting_delta_err{0};
};

} // namespace Core
