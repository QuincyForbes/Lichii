// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Bench mode without the engine: the request a host hands the app, the scripted input player and
// recorder, the steady-state test and the frame statistics. lichii_bench.cpp drives them on the
// device; tests/ios_frontend/bench_core.cpp runs them on a desktop.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace LichiiBench {

/// One step of an input script (tools/nx2/ios_bench/README.md lists the JSON forms).
struct Step {
    enum class Kind {
        Wait,        ///< ms
        WaitFrames,  ///< frames game frames
        WaitRunning, ///< until the game runs (not paused, not booting)
        WaitQuiet,   ///< until the JIT compiled under the steady threshold for ms
        Press,       ///< buttons down, ms or frames later up, then after_ms
        Hold,        ///< buttons down
        Release,     ///< buttons up
        Stick,       ///< stick at x, y; back to the centre after ms unless keep
        Mark,        ///< a named time in the result
        Log,         ///< a log line
    };
    Kind kind{Kind::Wait};
    double ms{};
    std::uint64_t frames{};
    double after_ms{};
    std::vector<int> buttons;
    int stick{};
    float x{};
    float y{};
    bool keep{};
    std::string text;
};

/// Button names as in LichiiButton ("A", "ZL", "PLUS", "DUP"...); -1 when unknown.
int ButtonFromName(std::string_view name);
const char* ButtonName(int button);

/// A script is an array of steps or {"steps": [...]}; "repeat" blocks are flattened.
bool ParseSteps(const nlohmann::json& json, std::vector<Step>& out, std::string& error);
nlohmann::json StepsToJson(const std::vector<Step>& steps);

struct SteadyConfig {
    double jit_percent_max{5.0};    ///< JIT compile time, % of one core
    double upload_mib_s_max{8.0};   ///< texture + buffer uploads
    double hold_s{10.0};            ///< every second of this long under both
    double timeout_s{300.0};
    double min_fps{1.0};            ///< and the game drawing
};

struct Request {
    enum class Mode { Run, CaptureSave, Record, Probe };

    Mode mode{Mode::Run};
    std::string run;           ///< label of this run (the host's matrix cell)
    std::string game;          ///< program id (16 hex digits) or a path under Documents
    std::string save;          ///< Documents/bench/saves/<save> goes into the game's save first
    int save_slot{};           ///< informational: the slot the menu script loads
    bool restore_owner_save{true};
    std::vector<std::string> experiments;
    bool keep_saved_experiments{};
    std::vector<std::pair<std::string, std::string>> env;
    std::vector<std::pair<std::string, std::string>> settings;
    std::string log_filter;    ///< appended to the iOS log filter
    std::string jit_provider{"stikdebug"}; ///< or "external" (a Mac debugger, tools/nx2/ios_bench)
    double jit_timeout_s{180.0};
    int thermal_max_state{1};  ///< 0 nominal, 1 fair, 2 serious, 3 critical
    double thermal_wait_s{};   ///< paused this long for the phone to cool, then "too_hot"
    double boot_timeout_s{600.0};
    double script_timeout_s{900.0};
    std::vector<Step> script;  ///< from the title screen to standing in the measured place
    SteadyConfig steady;
    double stand_s{75.0};
    std::vector<Step> pan;     ///< the camera turn, measured as its own window
    double pan_settle_s{2.0};
    double record_s{900.0};
    bool exit_app{true};
    nlohmann::json raw;
};

/// `base` resolves "script"/"pan" file names (Documents/bench). Unknown keys are kept in raw only.
bool ParseRequest(const nlohmann::json& json, const std::filesystem::path& base, Request& out,
                  std::string& error);
const char* ModeName(Request::Mode mode);

/// A 16-hex-digit program id, or nullopt for a path.
std::optional<std::uint64_t> ParseProgramId(std::string_view text);

/// What the player asks of the controller.
struct Action {
    enum class Kind { Button, Stick, Mark, Log };
    Kind kind{};
    int button{};
    bool pressed{};
    int stick{};
    float x{};
    float y{};
    std::string text;
};

struct PlayerContext {
    double now_ms{};
    std::uint64_t frames{};
    bool running{};
    /// Since when the JIT has been quiet, or a negative value while it compiles.
    double quiet_since_ms{-1.0};
};

/// Plays a script step by step; Advance returns what is due by ctx.now_ms.
class Player {
public:
    explicit Player(std::vector<Step> steps);

    std::vector<Action> Advance(const PlayerContext& ctx);
    bool Done() const {
        return index >= steps.size();
    }
    std::size_t Index() const {
        return index;
    }
    std::size_t Count() const {
        return steps.size();
    }

private:
    enum class Phase { Start, Waiting, Released };

    std::vector<Step> steps;
    std::size_t index{};
    Phase phase{Phase::Start};
    double until_ms{};
    std::uint64_t until_frames{};
    bool by_frames{};
};

/// Turns controller input into a script: the recording starts at 0 ms, then waits between
/// changes, buttons as hold/release, sticks thinned to one position per 50 ms that moved at least
/// 0.05 (the return to the centre always).
class Recorder {
public:
    void Button(double now_ms, int button, bool pressed);
    void Stick(double now_ms, int stick, float x, float y);
    /// Ends with the time since the last input as a wait.
    std::vector<Step> Finish(double now_ms);
    std::size_t Events() const {
        return events;
    }

private:
    void Gap(double now_ms);
    void Emit(double now_ms, int stick, float x, float y);
    void FlushPending(int stick);

    std::vector<Step> steps;
    double last_ms{};
    std::size_t events{};
    struct StickState {
        float x{};
        float y{};
        double ms{-1.0};
        bool pending{};
        float pending_x{};
        float pending_y{};
        double pending_ms{};
    };
    StickState sticks[2]{};
};

/// The counters a window takes differences of (Common::PerfReport, PerfStats).
struct Counters {
    double t_s{};
    std::uint64_t frames{};
    std::uint64_t gpu_busy_ns{};
    std::uint64_t gpu_buffers{};
    std::uint64_t jit_compile_ns{};
    std::uint64_t jit_blocks{};
    std::uint64_t texture_upload_bytes{};
    std::uint64_t buffer_upload_bytes{};
    std::uint64_t staging_upload_bytes{};
    std::uint64_t draws{};
    std::uint64_t render_passes{};
    std::uint64_t guest_run_ns{};
    std::uint64_t guest_kernel_ns{};
    std::uint64_t guest_idle_ns{};
    bool cpu_known{};
    std::uint64_t cpu_ns{};
    std::uint64_t performance_ns{};
    std::uint64_t energy_nj{};
    std::uint64_t runnable_ns{};
};

/// Steady state: every second of the last hold_s had the JIT quiet, uploads low and frames drawn.
class SteadyDetector {
public:
    explicit SteadyDetector(SteadyConfig config_) : config{config_} {}

    void Add(const Counters& sample);
    bool Steady() const;
    /// The figures of the last full second ("JIT 2.1 %, uploads 0.4 MiB/s, 28.9 fps").
    std::string Describe() const;
    double LastJitPercent() const;
    void Reset() {
        samples.clear();
    }

private:
    SteadyConfig config;
    std::vector<Counters> samples;
};

double JitPercent(const Counters& a, const Counters& b);
double UploadMibPerSecond(const Counters& a, const Counters& b);

struct FrameStats {
    std::uint64_t frames{};
    double duration_s{};
    double fps_avg{};
    double fps_1pct_low{}; ///< 1000 / the mean of the slowest 1 % of frame times
    double p50_ms{};
    double p90_ms{};
    double p95_ms{};
    double p99_ms{};
    double max_ms{};
};

/// From game frame timestamps (ns, ascending) inside [start_ns, end_ns].
FrameStats ComputeFrameStats(const std::vector<std::uint64_t>& stamps_ns, std::uint64_t start_ns,
                             std::uint64_t end_ns);

nlohmann::json FrameStatsJson(const FrameStats& stats);
/// Rates of a window from its first and last counters.
nlohmann::json WindowCountersJson(const Counters& a, const Counters& b);

} // namespace LichiiBench
