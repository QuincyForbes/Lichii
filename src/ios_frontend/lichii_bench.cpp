// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ios_frontend/lichii_bench.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <dispatch/dispatch.h>
#include <fmt/format.h>
#include <mach/mach.h>
#include <unistd.h>

#include "common/ios_jit.h"
#include "common/logging.h"
#include "common/perf_report.h"
#include "common/platform_events.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/perf_stats.h"
#include "ios_frontend/bench_core.h"
#include "ios_frontend/bridge_host.h"
#include "ios_frontend/controls_host.h"
#include "ios_frontend/lichii_bridge.h"
#include "ios_frontend/lichii_bridge_bench.h"
#include "ios_frontend/lichii_bridge_content.h"
#include "ios_frontend/lichii_bridge_controls.h"
#include "ios_frontend/lichii_bridge_platform.h"

namespace LichiiBench {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr int ResultSchema = 1;

struct BenchState {
    bool active{};
    bool request_ok{};
    std::string request_error;
    Request request;
    std::string trigger; ///< what asked for it: argument, environment, request file
    fs::path source;
    fs::path documents;
    fs::path dir; ///< Documents/bench
    Clock::time_point start{};
    std::string stamp;
    std::string log_filter;
    std::vector<std::string> notes;
};
BenchState g;

double Elapsed() {
    return std::chrono::duration<double>(Clock::now() - g.start).count();
}

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

void Sleep(double seconds) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

double Round(double value, double scale = 100.0) {
    return std::round(value * scale) / scale;
}

// ------------------------------------------------------------------------------ game frames

constexpr std::size_t FrameRing = 16384;
std::array<std::atomic<std::uint64_t>, FrameRing> g_frame_ns{};
std::atomic<std::uint64_t> g_frame_count{0};

void OnGameFrame() {
    const std::uint64_t index = g_frame_count.fetch_add(1, std::memory_order_relaxed);
    g_frame_ns[index % FrameRing].store(NowNs(), std::memory_order_relaxed);
}

std::vector<std::uint64_t> FramesBetween(std::uint64_t first, std::uint64_t end) {
    first = std::max(first, end > FrameRing ? end - FrameRing : 0);
    std::vector<std::uint64_t> stamps;
    stamps.reserve(static_cast<std::size_t>(end - first));
    for (std::uint64_t i = first; i < end; ++i) {
        stamps.push_back(g_frame_ns[i % FrameRing].load(std::memory_order_relaxed));
    }
    std::sort(stamps.begin(), stamps.end());
    return stamps;
}

// -------------------------------------------------------------------------- input recording

std::mutex g_record_mutex;
std::optional<Recorder> g_recorder;
std::optional<Clock::time_point> g_record_start;

double RecordMs() {
    return std::chrono::duration<double, std::milli>(Clock::now() - *g_record_start).count();
}

void TapButton(int player, int button, bool pressed) {
    std::scoped_lock lock{g_record_mutex};
    if (player == 0 && g_recorder && g_record_start) {
        g_recorder->Button(RecordMs(), button, pressed);
    }
}

void TapStick(int player, int stick, float x, float y) {
    std::scoped_lock lock{g_record_mutex};
    if (player == 0 && g_recorder && g_record_start) {
        g_recorder->Stick(RecordMs(), stick, x, y);
    }
}

// -------------------------------------------------------------------------------- helpers

void OnMain(std::function<void()> fn) {
    auto* work = new std::function<void()>(std::move(fn));
    dispatch_sync_f(dispatch_get_main_queue(), work, [](void* context) {
        std::unique_ptr<std::function<void()>> owned{static_cast<std::function<void()>*>(context)};
        (*owned)();
    });
}

void OnMainAsync(std::function<void()> fn) {
    auto* work = new std::function<void()>(std::move(fn));
    dispatch_async_f(dispatch_get_main_queue(), work, [](void* context) {
        std::unique_ptr<std::function<void()>> owned{static_cast<std::function<void()>*>(context)};
        (*owned)();
    });
}

std::string Take(char* text) {
    std::string out = text != nullptr ? text : "";
    lichii_free(text);
    return out;
}

bool WriteJson(const fs::path& path, const json& value) {
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    const fs::path temp = path.string() + ".tmp";
    {
        std::ofstream out{temp, std::ios::trunc};
        if (!out) {
            return false;
        }
        out << value.dump(2) << '\n';
        out.flush();
        if (!out) {
            return false;
        }
    }
    fs::rename(temp, path, error);
    return !error;
}

std::optional<json> ReadJson(const fs::path& path, std::string& error) {
    std::ifstream in{path};
    if (!in) {
        error = "cannot read " + path.string();
        return std::nullopt;
    }
    json value = json::parse(in, nullptr, false, true);
    if (value.is_discarded()) {
        error = path.string() + " is not valid JSON";
        return std::nullopt;
    }
    return value;
}

std::string LocalStamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char text[32];
    std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", &local);
    return text;
}

bool CopyTree(const fs::path& from, const fs::path& to, std::string& error) {
    std::error_code ec;
    fs::create_directories(to, ec);
    fs::copy(from, to, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) {
        error = fmt::format("copying {} to {}: {}", from.string(), to.string(), ec.message());
        return false;
    }
    return true;
}

/// Moves a directory, copying across volumes.
bool MoveTree(const fs::path& from, const fs::path& to, std::string& error) {
    std::error_code ec;
    fs::create_directories(to.parent_path(), ec);
    fs::rename(from, to, ec);
    if (!ec) {
        return true;
    }
    if (!CopyTree(from, to, error)) {
        return false;
    }
    fs::remove_all(from, ec);
    return true;
}

std::uint64_t LedgerPeakMib() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return static_cast<std::uint64_t>(info.ledger_phys_footprint_peak) >> 20;
}

Counters ReadCounters() {
    using namespace Common::PerfReport;
    const auto values = Read();
    Counters c;
    c.t_s = Elapsed();
    c.frames = g_frame_count.load(std::memory_order_relaxed);
    c.gpu_busy_ns = values[GpuBusyNs];
    c.gpu_buffers = values[GpuTimedBuffers];
    c.jit_compile_ns = values[JitCompileNs];
    c.jit_blocks = values[JitBlocks];
    c.texture_upload_bytes = values[TextureUploadBytes];
    c.buffer_upload_bytes = values[BufferUploadBytes];
    c.staging_upload_bytes = values[StagingUploadBytes];
    c.draws = values[Draws];
    c.render_passes = values[RenderPasses];
    c.guest_run_ns = values[GuestRunNs];
    c.guest_kernel_ns = values[GuestKernelNs];
    c.guest_idle_ns = values[GuestIdleNs];
    if (const auto cpu = ReadProcessCpu()) {
        c.cpu_known = true;
        c.cpu_ns = cpu->cpu_ns;
        c.performance_ns = cpu->performance_ns;
        c.energy_nj = cpu->energy_nj;
        c.runnable_ns = cpu->runnable_ns;
    }
    return c;
}

json ThermalJson() {
    return {{"state", lichii_platform_thermal_state()},
            {"pressure", Common::PerfReport::ThermalPressure()},
            {"auto_cool_fps", Common::PlatformEvents::ThermalMaxFps()},
            {"low_power_mode", lichii_platform_low_power_mode() != 0}};
}

/// The "Perf ..." log lines (Perf:, Perf guest:, Perf threads:, Perf sched:, Perf passes:...) as
/// they were logged; tools/nx2/ios_bench/bench_report.py reads the figures out of them.
class PerfLines {
public:
    void Begin() {
        const auto lines = Common::Log::ReadRing(0);
        after = lines.empty() ? 0 : lines.back().sequence;
        kept.clear();
    }
    void Collect() {
        for (auto& line : Common::Log::ReadRing(after)) {
            after = line.sequence;
            if (line.text.find(": Perf") != std::string::npos) {
                kept.push_back(std::move(line.text));
            }
        }
    }
    std::vector<std::string> Take() {
        return std::exchange(kept, {});
    }

private:
    std::uint64_t after{};
    std::vector<std::string> kept;
};

// ------------------------------------------------------------------------- owner's save

struct SaveSwap {
    bool installed{};
    fs::path save_dir;
    fs::path backup;
};
SaveSwap g_save;

fs::path SaveDir(std::uint64_t program_id, const std::string& game_path) {
    // The library scan reads game files on its own thread.
    std::scoped_lock lock{IosFrontend::LoaderMutex()};
    return Take(lichii_content_save_dir(program_id, game_path.c_str()));
}

fs::path BackupRoot() {
    return g.dir / "owner-save-backup";
}

/// Puts an owner save aside before a bench save goes in; an earlier backup (an interrupted run)
/// is the owner's and is kept.
bool InstallBenchSave(std::uint64_t program_id, const std::string& game_path, json& result,
                      std::string& error) {
    const fs::path source = g.dir / "saves" / g.request.save;
    std::error_code ec;
    if (!fs::is_directory(source, ec)) {
        error = "no save " + source.string() +
                " (make one with mode capture_save, or copy a save folder there)";
        return false;
    }
    const fs::path save_dir = SaveDir(program_id, game_path);
    if (save_dir.empty()) {
        error = "the game's save folder is not known";
        return false;
    }
    const fs::path backup = BackupRoot() / fmt::format("{:016X}", program_id);
    if (g.request.restore_owner_save && !fs::exists(backup / "save_dir.txt", ec)) {
        fs::create_directories(backup, ec);
        const bool had_save = fs::exists(save_dir, ec);
        if (had_save && !MoveTree(save_dir, backup / "save", error)) {
            return false;
        }
        std::ofstream{backup / "save_dir.txt"} << save_dir.string() << '\n'
                                               << (had_save ? "had_save" : "no_save") << '\n';
    }
    fs::remove_all(save_dir, ec);
    if (!CopyTree(source, save_dir, error)) {
        return false;
    }
    g_save = {true, save_dir, backup};
    result["save"] = {{"name", g.request.save},
                      {"slot", g.request.save_slot},
                      {"save_dir", save_dir.string()},
                      {"owner_save_kept_in", g.request.restore_owner_save ? backup.string() : ""}};
    LOG_INFO(Frontend, "Bench: save {} installed in {}", g.request.save, save_dir.string());
    return true;
}

void RestoreBackup(const fs::path& backup) {
    std::ifstream in{backup / "save_dir.txt"};
    std::string save_dir;
    std::string had;
    if (!std::getline(in, save_dir) || save_dir.empty()) {
        return;
    }
    std::getline(in, had);
    in.close();
    std::error_code ec;
    fs::remove_all(save_dir, ec);
    std::string error;
    if (had == "had_save" && !MoveTree(backup / "save", save_dir, error)) {
        LOG_ERROR(Frontend, "Bench: the owner save could not be put back from {}: {}",
                  backup.string(), error);
        return;
    }
    fs::remove_all(backup, ec);
    LOG_INFO(Frontend, "Bench: owner save put back in {}", save_dir);
}

void RestoreOwnerSave() {
    if (!g_save.installed) {
        return;
    }
    g_save.installed = false;
    if (g.request.restore_owner_save) {
        RestoreBackup(g_save.backup);
    }
}

// -------------------------------------------------------------------------------- result

json g_result;

/// Documents/bench/status.json: the host polls it to follow the run.
void WriteStatus(std::string_view phase, json extra = json::object()) {
    extra["phase"] = std::string{phase};
    extra["run"] = g.request.run;
    extra["stamp"] = g.stamp;
    extra["t_s"] = Round(Elapsed(), 10.0);
    extra["pid"] = getpid();
    WriteJson(g.dir / "status.json", extra);
}

bool WaitForState(std::initializer_list<int32_t> states, double timeout_s) {
    const double until = Elapsed() + timeout_s;
    while (Elapsed() < until) {
        const int32_t state = lichii_emu_state();
        if (std::find(states.begin(), states.end(), state) != states.end()) {
            return true;
        }
        Sleep(0.1);
    }
    return false;
}

void StopGame() {
    const int32_t state = lichii_emu_state();
    if (state == LICHII_EMU_IDLE || state == LICHII_EMU_FAILED) {
        return;
    }
    WriteStatus("stopping");
    OnMain([] {
        lichii_controls_release_touch();
        lichii_platform_flush_caches();
        if (lichii_emu_state() == LICHII_EMU_WAITING_FOR_JIT) {
            lichii_emu_cancel_launch();
        } else {
            lichii_emu_stop();
        }
    });
    if (!WaitForState({LICHII_EMU_IDLE, LICHII_EMU_FAILED}, 120.0)) {
        g.notes.push_back("the game did not stop within 120 s");
    }
}

void Finish(const std::string& status, const std::string& message) {
    StopGame();
    RestoreOwnerSave();
    g_result["status"] = status;
    g_result["message"] = message;
    g_result["elapsed_s"] = Round(Elapsed(), 10.0);
    g_result["memory"]["ledger_footprint_peak_mib"] = LedgerPeakMib();
    g_result["thermal_end"] = ThermalJson();
    g_result["notes"] = g.notes;
    const std::string base = "result-" + g.stamp;
    const fs::path log = Take(lichii_log_file());
    std::error_code ec;
    if (fs::copy_file(log, g.dir / (base + ".log"), fs::copy_options::overwrite_existing, ec)) {
        g_result["log"] = base + ".log";
    } else {
        g_result["log"] = nullptr;
        g_result["notes"].push_back("log not copied: " + ec.message());
    }
    g_result["result"] = base + ".json";
    WriteJson(g.dir / (base + ".json"), g_result);
    WriteJson(g.dir / "last.json", g_result);
    LOG_INFO(Frontend, "Bench: {} ({}): {}{}", status, g.request.run, message,
             g.request.exit_app ? ", exiting" : "");
    WriteStatus("done", {{"status", status}, {"result", base + ".json"}});
    if (!g.request.exit_app) {
        return;
    }
    OnMainAsync([] { std::exit(0); });
    Sleep(15.0);
    _exit(0);
}

// --------------------------------------------------------------------------------- phases

bool WaitForJit() {
    WriteStatus("jit");
    const double until = Elapsed() + g.request.jit_timeout_s;
    bool asked = false;
    double last_log = 0;
    while (Elapsed() < until) {
        const int32_t state = lichii_jit_state();
        if (state == LICHII_JIT_READY) {
            g_result["timeline"]["jit_ready_s"] = Round(Elapsed());
            g_result["jit"] = {{"provider", g.request.jit_provider},
                               {"pool_mib", lichii_jit_pool_bytes() >> 20}};
            return true;
        }
        if (state == LICHII_JIT_FAILED) {
            Finish("jit_failed", "JIT failed: " + Take(lichii_jit_error()));
            return false;
        }
        if (!asked && state == LICHII_JIT_WAITING && g.request.jit_provider == "stikdebug" &&
            Elapsed() > 1.0) {
            asked = true;
            bool opened = false;
            OnMain([&opened] { opened = lichii_jit_stikdebug_installed() && lichii_jit_request(); });
            LOG_INFO(Frontend, "Bench: asked StikDebug for JIT ({})",
                     opened ? "opened" : "StikDebug not installed or did not open");
        }
        if (Elapsed() - last_log > 15.0) {
            last_log = Elapsed();
            LOG_INFO(Frontend, "Bench: waiting for JIT, state {}, {} of {} MiB prepared", state,
                     lichii_jit_prepared_bytes() >> 20, lichii_jit_pool_bytes() >> 20);
        }
        Sleep(0.25);
    }
    const bool installed = lichii_jit_stikdebug_installed();
    Finish("jit_not_granted",
           fmt::format("no JIT after {:.0f} s (provider {}, StikDebug {}, debugged {}, TXM {}, "
                       "{} MiB prepared). {}",
                       g.request.jit_timeout_s, g.request.jit_provider,
                       installed ? "installed" : "not installed", lichii_jit_debugged(),
                       lichii_jit_needs_script(), lichii_jit_prepared_bytes() >> 20,
                       g.request.jit_provider == "external"
                           ? "Attach the Mac's debugger (tools/nx2/ios_bench/jit_lldb.py)."
                           : "Check StikDebug's pairing file and VPN, then retry."));
    return false;
}

std::optional<std::pair<std::string, std::uint64_t>> ResolveGame(std::string& error) {
    if (const auto program_id = ParseProgramId(g.request.game)) {
        LichiiGame* games = nullptr;
        const std::size_t count = lichii_games_scan(g.documents.c_str(), &games);
        std::optional<std::pair<std::string, std::uint64_t>> found;
        for (std::size_t i = 0; i < count; ++i) {
            if (games[i].program_id == *program_id && !found) {
                found.emplace(games[i].path, *program_id);
            }
        }
        lichii_games_free(games, count);
        if (!found) {
            error = fmt::format("no game {:016X} in Documents ({} games scanned)", *program_id, count);
        }
        return found;
    }
    const fs::path path = g.documents / g.request.game;
    LichiiGame game{};
    if (!lichii_game_read(path.c_str(), &game)) {
        error = "not a bootable game: " + path.string();
        return std::nullopt;
    }
    std::pair<std::string, std::uint64_t> out{game.path, game.program_id};
    lichii_game_clear(&game);
    return out;
}

bool Boot(const std::string& path) {
    WriteStatus("boot");
    bool launched = false;
    OnMain([&] { launched = lichii_emu_launch(path.c_str()); });
    if (!launched) {
        Finish("boot_failed", "the launch was refused (another game running?)");
        return false;
    }
    const double until = Elapsed() + g.request.boot_timeout_s;
    bool resumed = false;
    while (Elapsed() < until) {
        const int32_t state = lichii_emu_state();
        if (state == LICHII_EMU_RUNNING) {
            g_result["timeline"]["running_s"] = Round(Elapsed());
            return true;
        }
        if (state == LICHII_EMU_FAILED) {
            Finish("boot_failed", Take(lichii_emu_error()));
            return false;
        }
        // Loaded while the app was inactive: the game waits paused.
        if (state == LICHII_EMU_PAUSED && !resumed) {
            resumed = true;
            g.notes.push_back("the game started paused (app inactive at the end of the load)");
            OnMain([] { lichii_emu_resume(); });
        }
        Sleep(0.1);
    }
    Finish("boot_timeout", fmt::format("the game was not running after {:.0f} s",
                                       g.request.boot_timeout_s));
    return false;
}

struct Sampler {
    std::optional<Counters> last;
    double quiet_since_ms{-1.0};
    double next_s{};
    std::uint64_t footprint_peak{};
    std::uint64_t available_min{~0ULL};

    /// At most once a second; true when a new sample was taken.
    bool Tick(SteadyDetector* steady) {
        const double now = Elapsed();
        if (now < next_s) {
            return false;
        }
        next_s = now + 1.0;
        const Counters sample = ReadCounters();
        if (last) {
            if (JitPercent(*last, sample) < g.request.steady.jit_percent_max) {
                if (quiet_since_ms < 0) {
                    quiet_since_ms = now * 1000.0;
                }
            } else {
                quiet_since_ms = -1.0;
            }
        }
        last = sample;
        if (steady != nullptr) {
            steady->Add(sample);
        }
        footprint_peak = std::max<std::uint64_t>(footprint_peak, lichii_platform_footprint_mib());
        if (const std::uint64_t left = lichii_platform_available_memory_mib(); left != 0) {
            available_min = std::min(available_min, left);
        }
        return true;
    }
};

void Dispatch(const std::vector<Action>& actions, json& marks) {
    for (const Action& action : actions) {
        switch (action.kind) {
        case Action::Kind::Button:
            OnMainAsync([button = action.button, pressed = action.pressed] {
                lichii_controls_touch_button(0, button, pressed);
            });
            break;
        case Action::Kind::Stick:
            OnMainAsync([stick = action.stick, x = action.x, y = action.y] {
                lichii_controls_touch_stick(0, stick, x, y);
            });
            break;
        case Action::Kind::Mark:
            marks[action.text] = Round(Elapsed());
            LOG_INFO(Frontend, "Bench: mark {} at {:.1f} s", action.text, Elapsed());
            break;
        case Action::Kind::Log:
            LOG_INFO(Frontend, "Bench: {}", action.text);
            break;
        }
    }
}

/// Plays `steps` against the running game; false on timeout or when the game stopped.
bool Play(const std::vector<Step>& steps, double timeout_s, Sampler& sampler, const char* what) {
    Player player{steps};
    const double start = Elapsed();
    const std::uint64_t frames_at_start = g_frame_count.load(std::memory_order_relaxed);
    json& marks = g_result["marks"];
    std::size_t shown = ~std::size_t{0};
    // The script's clock stops while the game is paused (the app in the background).
    double paused_s = 0;
    double last = start;
    bool noted_pause = false;
    while (!player.Done()) {
        const double now = Elapsed();
        if (now - start > timeout_s) {
            g.notes.push_back(fmt::format("{} timed out at step {} of {}", what, player.Index() + 1,
                                          player.Count()));
            return false;
        }
        const int32_t state = lichii_emu_state();
        if (state != LICHII_EMU_RUNNING && state != LICHII_EMU_PAUSED) {
            g.notes.push_back(fmt::format("the game stopped during the {}", what));
            return false;
        }
        if (state == LICHII_EMU_PAUSED) {
            paused_s += now - last;
            if (!noted_pause) {
                noted_pause = true;
                g.notes.push_back(fmt::format("the game was paused during the {}", what));
            }
        }
        last = now;
        sampler.Tick(nullptr);
        PlayerContext ctx;
        ctx.now_ms = (now - start - paused_s) * 1000.0;
        ctx.frames = g_frame_count.load(std::memory_order_relaxed) - frames_at_start;
        ctx.running = state == LICHII_EMU_RUNNING;
        ctx.quiet_since_ms = sampler.quiet_since_ms < 0
                                 ? -1.0
                                 : std::max(0.0, sampler.quiet_since_ms - start * 1000.0);
        Dispatch(player.Advance(ctx), marks);
        if (player.Index() != shown) {
            shown = player.Index();
            WriteStatus(fmt::format("{} {}/{}", what, shown, player.Count()));
        }
        Sleep(0.005);
    }
    OnMainAsync([] { lichii_controls_release_touch(); });
    return true;
}

bool WaitSteady(Sampler& sampler, json& out) {
    WriteStatus("steady");
    SteadyDetector steady{g.request.steady};
    const double start = Elapsed();
    double last_log = start;
    while (Elapsed() - start < g.request.steady.timeout_s) {
        if (sampler.Tick(&steady)) {
            if (steady.Steady()) {
                out = {{"reached", true},
                       {"after_s", Round(Elapsed() - start)},
                       {"last_second", steady.Describe()}};
                g_result["timeline"]["steady_s"] = Round(Elapsed());
                return true;
            }
            if (Elapsed() - last_log >= 10.0) {
                last_log = Elapsed();
                LOG_INFO(Frontend, "Bench: not steady yet: {}", steady.Describe());
            }
        }
        const int32_t state = lichii_emu_state();
        if (state != LICHII_EMU_RUNNING && state != LICHII_EMU_PAUSED) {
            out = {{"reached", false}, {"why", "the game stopped"}};
            return false;
        }
        Sleep(0.05);
    }
    out = {{"reached", false},
           {"after_s", Round(Elapsed() - start)},
           {"last_second", steady.Describe()}};
    return false;
}

/// Thermal gate: above the request's state the game waits paused for the phone to cool.
bool CoolEnough() {
    if (lichii_platform_thermal_state() <= g.request.thermal_max_state) {
        return true;
    }
    if (g.request.thermal_wait_s <= 0) {
        return false;
    }
    WriteStatus("cooling");
    const bool running = lichii_emu_state() == LICHII_EMU_RUNNING;
    if (running) {
        OnMain([] { lichii_emu_pause(); });
    }
    const double start = Elapsed();
    while (Elapsed() - start < g.request.thermal_wait_s &&
           lichii_platform_thermal_state() > g.request.thermal_max_state) {
        Sleep(5.0);
    }
    g_result["timeline"]["cooled_for_s"] = Round(Elapsed() - start);
    if (running) {
        OnMain([] { lichii_emu_resume(); });
    }
    return lichii_platform_thermal_state() <= g.request.thermal_max_state;
}

json MeasureWindow(const char* name, double seconds, const std::vector<Step>* steps,
                   Sampler& sampler) {
    WriteStatus(name);
    PerfLines lines;
    lines.Begin();
    Sampler window;
    window.Tick(nullptr);
    const json thermal_start = ThermalJson();
    const Counters a = ReadCounters();
    const std::uint64_t first_frame = g_frame_count.load(std::memory_order_relaxed);
    const std::uint64_t start_ns = NowNs();
    bool paused = false;
    bool completed = true;
    if (steps != nullptr) {
        // The pan: as long as its script plays.
        Player player{*steps};
        const double start = Elapsed();
        json& marks = g_result["marks"];
        while (!player.Done() && Elapsed() - start < 300.0) {
            PlayerContext ctx;
            ctx.now_ms = (Elapsed() - start) * 1000.0;
            ctx.frames = g_frame_count.load(std::memory_order_relaxed) - first_frame;
            ctx.running = lichii_emu_state() == LICHII_EMU_RUNNING;
            paused |= !ctx.running;
            Dispatch(player.Advance(ctx), marks);
            if (window.Tick(nullptr)) {
                lines.Collect();
            }
            Sleep(0.005);
        }
        completed = player.Done();
        OnMainAsync([] { lichii_controls_release_touch(); });
    } else {
        const double end = Elapsed() + seconds;
        while (Elapsed() < end) {
            paused |= lichii_emu_state() != LICHII_EMU_RUNNING;
            if (window.Tick(nullptr)) {
                lines.Collect();
            }
            Sleep(0.05);
        }
    }
    const std::uint64_t end_ns = NowNs();
    const Counters b = ReadCounters();
    const std::uint64_t end_frame = g_frame_count.load(std::memory_order_relaxed);
    lines.Collect();
    sampler.footprint_peak = std::max(sampler.footprint_peak, window.footprint_peak);
    sampler.available_min = std::min(sampler.available_min, window.available_min);

    json out = FrameStatsJson(ComputeFrameStats(FramesBetween(first_frame, end_frame), start_ns,
                                                end_ns));
    out.update(WindowCountersJson(a, b));
    out["name"] = name;
    out["start_s"] = Round(a.t_s);
    out["thermal_start"] = thermal_start;
    out["thermal_end"] = ThermalJson();
    out["footprint_peak_mib"] = window.footprint_peak;
    out["available_min_mib"] = window.available_min == ~0ULL ? 0 : window.available_min;
    out["paused_or_inactive"] = paused;
    out["completed"] = completed;
    out["perf_lines"] = lines.Take();
    LOG_INFO(Frontend, "Bench: {} window: {} fps avg, {} 1% low, GPU {} ms/frame", name,
             out["fps_avg"].dump(), out["fps_1pct_low"].dump(), out["gpu_ms_per_frame"].dump());
    return out;
}

json SettingsJson() {
    static constexpr const char* Keys[] = {
        "resolution_setup", "oot_graphics_preset", "oot_resolution", "oot_frame_rate",
        "texture_quality", "use_docked_mode", "moltenvk_driver", "gpu_accuracy",
        "use_asynchronous_shaders", "frame_gen", "scaling_filter", "anti_aliasing",
    };
    json out = json::object();
    OnMain([&out] {
        for (const char* key : Keys) {
            out[key] = Take(lichii_settings_get_effective(key));
        }
        for (const auto& [key, value] : g.request.settings) {
            out[key] = Take(lichii_settings_get_effective(key.c_str()));
        }
    });
    return out;
}

json DeviceJson() {
    LichiiDeviceInfo info{};
    if (!lichii_platform_device_info(&info)) {
        return nullptr;
    }
    return {{"machine", info.machine},
            {"chip", info.chip},
            {"gpu", info.gpu_name},
            {"ios", fmt::format("{}.{}", info.os_major, info.os_minor)},
            {"memory_mib", info.physical_memory_mib},
            {"available_at_launch_mib", info.launch_available_memory_mib},
            {"increased_memory_limit", info.increased_memory_limit},
            {"txm", info.txm},
            {"max_fps", info.max_fps}};
}

void Record(const std::string& path, std::uint64_t program_id) {
    if (!Boot(path)) {
        return;
    }
    {
        std::scoped_lock lock{g_record_mutex};
        g_recorder.emplace();
        g_record_start = Clock::now();
    }
    LichiiControls::SetInputTaps(&TapButton, &TapStick);
    WriteStatus("record");
    LOG_INFO(Frontend, "Bench: recording the controls for up to {:.0f} s; stop the game to end",
             g.request.record_s);
    const double until = Elapsed() + g.request.record_s;
    while (Elapsed() < until) {
        const int32_t state = lichii_emu_state();
        if (state != LICHII_EMU_RUNNING && state != LICHII_EMU_PAUSED) {
            break;
        }
        Sleep(0.25);
    }
    LichiiControls::SetInputTaps(nullptr, nullptr);
    std::vector<Step> steps;
    std::size_t events = 0;
    {
        std::scoped_lock lock{g_record_mutex};
        events = g_recorder->Events();
        steps = g_recorder->Finish(RecordMs());
        g_recorder.reset();
    }
    // The recording starts at the first frame; a replay waits for it the same way.
    Step running;
    running.kind = Step::Kind::WaitRunning;
    steps.insert(steps.begin(), running);
    const std::string name = "recorded-" + g.stamp + ".json";
    WriteJson(g.dir / "scripts" / name,
              {{"program_id", fmt::format("{:016X}", program_id)},
               {"recorded", g.stamp},
               {"build", Common::g_scm_rev},
               {"steps", StepsToJson(steps)}});
    g_result["recorded_script"] = "scripts/" + name;
    Finish("ok", fmt::format("{} input changes recorded to scripts/{}", events, name));
}

void Run() {
    Common::SetCurrentThreadName("LichiiBench");
    g_result = {
        {"schema", ResultSchema},
        {"run", g.request.run},
        {"mode", ModeName(g.request.mode)},
        {"stamp", g.stamp},
        {"trigger", g.trigger},
        {"request_file", g.source.string()},
        {"request", g.request.raw},
        {"build",
         {{"sha", Common::g_scm_rev},
          {"branch", Common::g_scm_branch},
          {"describe", Common::g_scm_desc},
          {"name", Common::g_build_name},
          {"version", Take(lichii_build_version())}}},
        {"device", DeviceJson()},
        {"experiments", g.request.experiments},
        {"keep_saved_experiments", g.request.keep_saved_experiments},
        {"env", json::object()},
        {"thermal_launch", ThermalJson()},
        {"timeline", json::object()},
        {"marks", json::object()},
        {"windows", json::array()},
    };
    for (const auto& [key, value] : g.request.env) {
        g_result["env"][key] = value;
    }
    if (!g.request_ok) {
        Finish("bad_request", g.request_error);
        return;
    }
    LOG_INFO(Frontend, "Bench: {} run \"{}\" from {} ({})", ModeName(g.request.mode),
             g.request.run, g.source.string(), g.trigger);

    if (g.request.mode == Request::Mode::Probe) {
        // Setup check: what this launch saw, without a game.
        g_result["probe"] = {{"jit_state", lichii_jit_state()},
                             {"jit_debugged", lichii_jit_debugged()},
                             {"jit_needs_script", lichii_jit_needs_script()},
                             {"stikdebug_installed", lichii_jit_stikdebug_installed()},
                             {"documents", g.documents.string()}};
        if (g.request.raw.value("wait_for_jit", false) && !WaitForJit()) {
            return;
        }
        Finish("ok", "probe");
        return;
    }
    if (g.request.mode == Request::Mode::Run &&
        lichii_platform_thermal_state() > g.request.thermal_max_state &&
        g.request.thermal_wait_s <= 0) {
        Finish("too_hot", fmt::format("thermal state {} at launch (the request allows {})",
                                      lichii_platform_thermal_state(),
                                      g.request.thermal_max_state));
        return;
    }
    WriteStatus("game");
    std::string error;
    const auto game = ResolveGame(error);
    if (!game) {
        Finish("no_game", error);
        return;
    }
    const auto& [path, program_id] = *game;
    g_result["game"] = {{"path", path}, {"program_id", fmt::format("{:016X}", program_id)}};

    if (g.request.mode == Request::Mode::CaptureSave) {
        const fs::path save_dir = SaveDir(program_id, path);
        const fs::path target = g.dir / "saves" / g.request.save;
        std::error_code ec;
        if (save_dir.empty() || !fs::is_directory(save_dir, ec)) {
            Finish("no_save", "the game has no save to capture in " + save_dir.string());
            return;
        }
        fs::remove_all(target, ec);
        if (!CopyTree(save_dir, target, error)) {
            Finish("error", error);
            return;
        }
        Finish("ok", fmt::format("save of {:016X} copied to saves/{}", program_id, g.request.save));
        return;
    }
    if (!g.request.save.empty() && !InstallBenchSave(program_id, path, g_result, error)) {
        Finish("no_save", error);
        return;
    }
    if (!WaitForJit()) {
        return;
    }
    if (g.request.mode == Request::Mode::Record) {
        Record(path, program_id);
        return;
    }
    if (!Boot(path)) {
        return;
    }
    g_result["settings"] = SettingsJson();
    Sampler sampler;
    if (!g.request.script.empty()) {
        WriteStatus("script");
        if (!Play(g.request.script, g.request.script_timeout_s, sampler, "script")) {
            Finish("script_failed", g.notes.empty() ? "the input script did not finish"
                                                    : g.notes.back());
            return;
        }
        g_result["timeline"]["script_done_s"] = Round(Elapsed());
    }
    json steady;
    const bool steady_reached = WaitSteady(sampler, steady);
    g_result["steady"] = steady;
    if (!steady_reached && lichii_emu_state() != LICHII_EMU_RUNNING &&
        lichii_emu_state() != LICHII_EMU_PAUSED) {
        Finish("game_stopped", "the game stopped before it was steady");
        return;
    }
    if (!CoolEnough()) {
        Finish("too_hot", fmt::format("thermal state {} before the measurement (the request "
                                      "allows {})",
                                      lichii_platform_thermal_state(),
                                      g.request.thermal_max_state));
        return;
    }
    g_result["thermal_measure_start"] = ThermalJson();
    if (g.request.stand_s > 0) {
        g_result["windows"].push_back(MeasureWindow("stand", g.request.stand_s, nullptr, sampler));
    }
    if (!g.request.pan.empty()) {
        Sleep(g.request.pan_settle_s);
        g_result["windows"].push_back(MeasureWindow("pan", 0.0, &g.request.pan, sampler));
    }
    g_result["memory"] = {{"footprint_peak_mib", sampler.footprint_peak},
                          {"available_min_mib",
                           sampler.available_min == ~0ULL ? 0 : sampler.available_min}};
    g_result["thermal_measure_end"] = ThermalJson();
    Finish(steady_reached ? "ok" : "steady_timeout",
           steady_reached ? "measured"
                          : "measured without a steady state (see steady.last_second)");
}

std::string Arg(std::string_view arg, std::string_view name, int& i, int argc, char** argv,
                bool& matched) {
    matched = false;
    if (arg == name) {
        matched = true;
        if (i + 1 < argc) {
            return argv[++i];
        }
        return "1";
    }
    if (arg.size() > name.size() && arg.substr(0, name.size()) == name &&
        arg[name.size()] == '=') {
        matched = true;
        return std::string{arg.substr(name.size() + 1)};
    }
    return {};
}

std::vector<std::string> SplitList(std::string_view text) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = std::min(text.find(',', start), text.size());
        std::string item{text.substr(start, end - start)};
        if (!item.empty()) {
            out.push_back(std::move(item));
        }
        start = end + 1;
    }
    return out;
}

} // Anonymous namespace

void Configure(int& argc, char** argv) {
    std::string trigger_value;
    std::string trigger;
    std::string extra_experiments;
    std::string run_label;
    int kept = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        bool matched = false;
        if (std::string value = Arg(arg, "--lichii-bench", i, argc, argv, matched); matched) {
            trigger_value = std::move(value);
            trigger = "argument";
            continue;
        }
        if (std::string value = Arg(arg, "--lichii-bench-experiments", i, argc, argv, matched);
            matched) {
            extra_experiments = std::move(value);
            continue;
        }
        if (std::string value = Arg(arg, "--lichii-bench-run", i, argc, argv, matched); matched) {
            run_label = std::move(value);
            continue;
        }
        argv[kept++] = argv[i];
    }
    argc = kept;
    argv[argc] = nullptr;

    if (const char* value = std::getenv("LICHII_BENCH"); trigger.empty() && value && *value) {
        trigger_value = value;
        trigger = "environment";
    }
    if (const char* value = std::getenv("LICHII_BENCH_EXPERIMENTS"); extra_experiments.empty() && value) {
        extra_experiments = value;
    }
    if (const char* value = std::getenv("LICHII_BENCH_RUN"); run_label.empty() && value) {
        run_label = value;
    }
    const char* const home = std::getenv("HOME");
    g.documents = fs::path{home != nullptr ? home : "."} / "Documents";
    g.dir = g.documents / "bench";
    std::error_code ec;
    if (trigger.empty() && fs::exists(g.dir / "request.json", ec)) {
        trigger_value = "request";
        trigger = "request file";
    }
    if (trigger.empty() || trigger_value == "0" || trigger_value == "off") {
        return;
    }
    g.active = true;
    g.start = Clock::now();
    g.stamp = LocalStamp();
    g.trigger = trigger;
    if (trigger_value == "1" || trigger_value == "request") {
        g.source = g.dir / "request.json";
    } else if (trigger_value.find('/') != std::string::npos ||
               trigger_value.ends_with(".json")) {
        g.source = trigger_value.front() == '/' ? fs::path{trigger_value} : g.dir / trigger_value;
    } else {
        g.source = g.dir / "profiles" / (trigger_value + ".json");
    }
    std::string error;
    if (auto json = ReadJson(g.source, error)) {
        g.request_ok = ParseRequest(*json, g.dir, g.request, error);
    }
    // A request file asks for one run: the next normal launch must not be a bench run.
    if (g.source == g.dir / "request.json") {
        fs::rename(g.source, g.dir / ("request-used-" + g.stamp + ".json"), ec);
    }
    if (!g.request_ok) {
        g.request_error = error;
        LOG_ERROR(Frontend, "Bench: request {} refused: {}", g.source.string(), error);
        return;
    }
    if (!run_label.empty()) {
        g.request.run = run_label;
        g.request.raw["run"] = run_label;
    }
    for (auto& name : SplitList(extra_experiments)) {
        if (std::find(g.request.experiments.begin(), g.request.experiments.end(), name) ==
            g.request.experiments.end()) {
            g.request.experiments.push_back(std::move(name));
        }
    }
    for (const auto& [key, value] : g.request.env) {
        setenv(key.c_str(), value.c_str(), 1);
    }
    // The measurement reads the "Perf:" counters.
    setenv("EDEN_NX2_PERF_REPORT", "1", 0);
    std::vector<const char*> names;
    for (const auto& name : g.request.experiments) {
        names.push_back(name.c_str());
    }
    lichii_platform_override_experiments(names.data(), static_cast<int>(names.size()),
                                         g.request.keep_saved_experiments ? 1 : 0);
    g.log_filter = g.request.log_filter;
    WriteStatus("configured");
    LOG_INFO(Frontend, "Bench: {} run \"{}\" ({}), experiments [{}], {} environment values",
             ModeName(g.request.mode), g.request.run, g.trigger,
             fmt::join(g.request.experiments, ", "), g.request.env.size());
}

bool Active() {
    return g.active;
}

const char* LogFilter() {
    return g.log_filter.c_str();
}

void Start() {
    if (!g.active) {
        // An interrupted bench run left the owner's save aside.
        std::error_code ec;
        for (fs::directory_iterator it{BackupRoot(), ec}, end; !ec && it != end; it.increment(ec)) {
            RestoreBackup(it->path());
        }
        return;
    }
    if (g.request_ok) {
        for (const auto& [key, value] : g.request.settings) {
            const bool ok = lichii_settings_set(key.c_str(), value.c_str());
            LOG_INFO(Frontend, "Bench: setting {} = {}{}", key, value, ok ? "" : " (refused)");
            if (!ok) {
                g.notes.push_back("setting refused: " + key);
            }
        }
    }
    Common::Log::SetRingCapacity(16384);
    Core::PerfStats::SetGameFrameObserver(&OnGameFrame);
    std::thread(Run).detach();
}

void ReapplySettings() {
    if (!g.active || !g.request_ok) {
        return;
    }
    // The game's own value is the one in effect where its config sets the key.
    auto& by_key = Settings::values.linkage.by_key;
    for (const auto& [key, value] : g.request.settings) {
        const auto it = by_key.find(key);
        if (it == by_key.end()) {
            continue;
        }
        Settings::BasicSetting* const setting = it->second;
        const bool custom = !setting->UsingGlobal();
        if (custom) {
            setting->SetGlobal(true);
        }
        setting->LoadString(value);
        if (custom) {
            setting->SetGlobal(false);
            setting->LoadString(value);
            LOG_INFO(Frontend, "Bench: setting {} = {} over the game's own value", key, value);
        }
    }
}

bool BlocksConfigSave() {
    return g.active;
}

} // namespace LichiiBench

extern "C" {

bool lichii_bench_active(void) {
    return LichiiBench::Active();
}

} // extern "C"
