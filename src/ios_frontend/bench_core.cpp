// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ios_frontend/bench_core.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

namespace LichiiBench {
namespace {

// LichiiButton order (lichii_bridge.h).
constexpr std::array<const char*, 20> ButtonNames{
    "A",  "B",  "X",    "Y",     "LS",   "RS", "L",     "R",  "ZL",   "ZR",
    "PLUS", "MINUS", "LEFT", "UP", "RIGHT", "DOWN", "SL", "SR", "HOME", "CAPTURE",
};

std::string Upper(std::string_view text) {
    std::string out{text};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

bool ReadButtons(const nlohmann::json& value, std::vector<int>& out, std::string& error) {
    const auto add = [&](const nlohmann::json& name) {
        if (!name.is_string()) {
            error = "a button is not a name";
            return false;
        }
        const int button = ButtonFromName(name.get<std::string>());
        if (button < 0) {
            error = "unknown button \"" + name.get<std::string>() + "\"";
            return false;
        }
        out.push_back(button);
        return true;
    };
    if (value.is_array()) {
        for (const auto& name : value) {
            if (!add(name)) {
                return false;
            }
        }
        if (out.empty()) {
            error = "no buttons";
            return false;
        }
        return true;
    }
    return add(value);
}

int StickFromName(const nlohmann::json& value) {
    if (value.is_number_integer()) {
        const int stick = value.get<int>();
        return stick == 0 || stick == 1 ? stick : -1;
    }
    if (value.is_string()) {
        const std::string name = Upper(value.get<std::string>());
        if (name == "LEFT" || name == "L") {
            return 0;
        }
        if (name == "RIGHT" || name == "R") {
            return 1;
        }
    }
    return -1;
}

double Number(const nlohmann::json& object, const char* key, double fallback) {
    const auto it = object.find(key);
    return it != object.end() && it->is_number() ? it->get<double>() : fallback;
}

bool ParseStep(const nlohmann::json& json, std::vector<Step>& out, std::string& error, int depth) {
    if (!json.is_object()) {
        error = "a step is not an object";
        return false;
    }
    if (const auto it = json.find("repeat"); it != json.end()) {
        if (depth > 4 || !it->is_number_unsigned() || !json.contains("steps")) {
            error = "\"repeat\" needs a count and \"steps\" (nested at most 4 deep)";
            return false;
        }
        std::vector<Step> body;
        for (const auto& inner : json["steps"]) {
            if (!ParseStep(inner, body, error, depth + 1)) {
                return false;
            }
        }
        const auto count = it->get<std::uint64_t>();
        for (std::uint64_t i = 0; i < count; ++i) {
            out.insert(out.end(), body.begin(), body.end());
        }
        return true;
    }
    Step step;
    step.after_ms = Number(json, "after_ms", 0.0);
    if (json.contains("wait_ms") || json.contains("wait_s")) {
        step.kind = Step::Kind::Wait;
        step.ms = json.contains("wait_ms") ? Number(json, "wait_ms", 0) : Number(json, "wait_s", 0) * 1000;
    } else if (json.contains("wait_frames")) {
        step.kind = Step::Kind::WaitFrames;
        step.frames = static_cast<std::uint64_t>(std::max(0.0, Number(json, "wait_frames", 0)));
    } else if (json.contains("wait_running")) {
        step.kind = Step::Kind::WaitRunning;
    } else if (json.contains("wait_quiet_s")) {
        step.kind = Step::Kind::WaitQuiet;
        step.ms = Number(json, "wait_quiet_s", 0) * 1000;
    } else if (json.contains("press")) {
        step.kind = Step::Kind::Press;
        if (!ReadButtons(json["press"], step.buttons, error)) {
            return false;
        }
        if (json.contains("hold_frames")) {
            step.frames =
                static_cast<std::uint64_t>(std::max(1.0, Number(json, "hold_frames", 1)));
        } else {
            step.ms = Number(json, "hold_ms", 100.0);
        }
    } else if (json.contains("hold")) {
        step.kind = Step::Kind::Hold;
        if (!ReadButtons(json["hold"], step.buttons, error)) {
            return false;
        }
    } else if (json.contains("release")) {
        step.kind = Step::Kind::Release;
        if (!ReadButtons(json["release"], step.buttons, error)) {
            return false;
        }
    } else if (json.contains("stick")) {
        step.kind = Step::Kind::Stick;
        step.stick = StickFromName(json["stick"]);
        if (step.stick < 0) {
            error = "\"stick\" is \"left\" or \"right\"";
            return false;
        }
        step.x = static_cast<float>(std::clamp(Number(json, "x", 0), -1.0, 1.0));
        step.y = static_cast<float>(std::clamp(Number(json, "y", 0), -1.0, 1.0));
        step.keep = json.value("keep", false);
        if (json.contains("frames")) {
            step.frames = static_cast<std::uint64_t>(std::max(0.0, Number(json, "frames", 0)));
        } else {
            step.ms = json.contains("s") ? Number(json, "s", 0) * 1000 : Number(json, "ms", 0);
        }
        if (step.ms <= 0 && step.frames == 0) {
            step.keep = true;
        }
    } else if (json.contains("mark")) {
        step.kind = Step::Kind::Mark;
        step.text = json["mark"].is_string() ? json["mark"].get<std::string>() : "mark";
    } else if (json.contains("log")) {
        step.kind = Step::Kind::Log;
        step.text = json["log"].is_string() ? json["log"].get<std::string>() : "";
    } else {
        error = "unknown step " + json.dump();
        return false;
    }
    if (step.ms < 0 || step.after_ms < 0) {
        error = "negative time in " + json.dump();
        return false;
    }
    out.push_back(std::move(step));
    return true;
}

bool LoadJsonFile(const std::filesystem::path& path, nlohmann::json& out, std::string& error) {
    std::ifstream in{path};
    if (!in) {
        error = "cannot read " + path.string();
        return false;
    }
    out = nlohmann::json::parse(in, nullptr, false, true);
    if (out.is_discarded()) {
        error = path.string() + " is not valid JSON";
        return false;
    }
    return true;
}

/// "script": a file name under base, or the steps themselves.
bool ReadScript(const nlohmann::json& request, const char* key, const std::filesystem::path& base,
                std::vector<Step>& out, std::string& error) {
    const auto it = request.find(key);
    if (it == request.end() || it->is_null()) {
        return true;
    }
    nlohmann::json script = *it;
    if (it->is_string()) {
        if (!LoadJsonFile(base / it->get<std::string>(), script, error)) {
            return false;
        }
    }
    if (!ParseSteps(script, out, error)) {
        error = std::string{key} + ": " + error;
        return false;
    }
    return true;
}

std::vector<std::pair<std::string, std::string>> ReadPairs(const nlohmann::json& object) {
    std::vector<std::pair<std::string, std::string>> out;
    if (!object.is_object()) {
        return out;
    }
    for (const auto& [key, value] : object.items()) {
        if (value.is_string()) {
            out.emplace_back(key, value.get<std::string>());
        } else if (value.is_boolean()) {
            out.emplace_back(key, value.get<bool>() ? "true" : "false");
        } else if (value.is_number_integer()) {
            out.emplace_back(key, std::to_string(value.get<long long>()));
        } else if (value.is_number()) {
            out.emplace_back(key, value.dump());
        }
    }
    return out;
}

double Percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double rank = fraction * static_cast<double>(sorted.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(rank));
    const auto high = std::min(sorted.size() - 1, low + 1);
    return sorted[low] + (sorted[high] - sorted[low]) * (rank - static_cast<double>(low));
}

double Round(double value, double scale = 100.0) {
    return std::round(value * scale) / scale;
}

template <typename T>
double Delta(T after, T before) {
    return after > before ? static_cast<double>(after - before) : 0.0;
}

} // Anonymous namespace

int ButtonFromName(std::string_view name) {
    std::string upper = Upper(name);
    if (upper == "DUP" || upper == "DPAD_UP") {
        upper = "UP";
    } else if (upper == "DDOWN" || upper == "DPAD_DOWN") {
        upper = "DOWN";
    } else if (upper == "DLEFT" || upper == "DPAD_LEFT") {
        upper = "LEFT";
    } else if (upper == "DRIGHT" || upper == "DPAD_RIGHT") {
        upper = "RIGHT";
    } else if (upper == "+" || upper == "START") {
        upper = "PLUS";
    } else if (upper == "-" || upper == "SELECT") {
        upper = "MINUS";
    }
    for (std::size_t i = 0; i < ButtonNames.size(); ++i) {
        if (upper == ButtonNames[i]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

const char* ButtonName(int button) {
    return button >= 0 && button < static_cast<int>(ButtonNames.size()) ? ButtonNames[button] : "?";
}

bool ParseSteps(const nlohmann::json& json, std::vector<Step>& out, std::string& error) {
    const nlohmann::json* list = &json;
    if (json.is_object() && json.contains("steps")) {
        list = &json["steps"];
    }
    if (!list->is_array()) {
        error = "a script is an array of steps or {\"steps\": [...]}";
        return false;
    }
    for (const auto& step : *list) {
        if (!ParseStep(step, out, error, 0)) {
            return false;
        }
    }
    return true;
}

nlohmann::json StepsToJson(const std::vector<Step>& steps) {
    nlohmann::json out = nlohmann::json::array();
    const auto names = [](const std::vector<int>& buttons) {
        if (buttons.size() == 1) {
            return nlohmann::json(ButtonName(buttons.front()));
        }
        nlohmann::json list = nlohmann::json::array();
        for (const int button : buttons) {
            list.push_back(ButtonName(button));
        }
        return list;
    };
    for (const Step& step : steps) {
        nlohmann::json json;
        switch (step.kind) {
        case Step::Kind::Wait:
            json["wait_ms"] = Round(step.ms, 1.0);
            break;
        case Step::Kind::WaitFrames:
            json["wait_frames"] = step.frames;
            break;
        case Step::Kind::WaitRunning:
            json["wait_running"] = true;
            break;
        case Step::Kind::WaitQuiet:
            json["wait_quiet_s"] = step.ms / 1000.0;
            break;
        case Step::Kind::Press:
            json["press"] = names(step.buttons);
            if (step.frames != 0) {
                json["hold_frames"] = step.frames;
            } else {
                json["hold_ms"] = Round(step.ms, 1.0);
            }
            break;
        case Step::Kind::Hold:
            json["hold"] = names(step.buttons);
            break;
        case Step::Kind::Release:
            json["release"] = names(step.buttons);
            break;
        case Step::Kind::Stick:
            json["stick"] = step.stick == 0 ? "left" : "right";
            json["x"] = Round(step.x);
            json["y"] = Round(step.y);
            if (step.keep) {
                json["keep"] = true;
            } else if (step.frames != 0) {
                json["frames"] = step.frames;
            } else {
                json["ms"] = Round(step.ms, 1.0);
            }
            break;
        case Step::Kind::Mark:
            json["mark"] = step.text;
            break;
        case Step::Kind::Log:
            json["log"] = step.text;
            break;
        }
        if (step.after_ms > 0) {
            json["after_ms"] = Round(step.after_ms, 1.0);
        }
        out.push_back(std::move(json));
    }
    return out;
}

const char* ModeName(Request::Mode mode) {
    switch (mode) {
    case Request::Mode::Run:
        return "run";
    case Request::Mode::CaptureSave:
        return "capture_save";
    case Request::Mode::Record:
        return "record";
    case Request::Mode::Probe:
        return "probe";
    }
    return "run";
}

std::optional<std::uint64_t> ParseProgramId(std::string_view text) {
    if (text.size() != 16) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (const char c : text) {
        const int upper = std::toupper(static_cast<unsigned char>(c));
        int digit;
        if (upper >= '0' && upper <= '9') {
            digit = upper - '0';
        } else if (upper >= 'A' && upper <= 'F') {
            digit = upper - 'A' + 10;
        } else {
            return std::nullopt;
        }
        value = value * 16 + static_cast<std::uint64_t>(digit);
    }
    return value;
}

bool ParseRequest(const nlohmann::json& json, const std::filesystem::path& base, Request& out,
                  std::string& error) {
    if (!json.is_object()) {
        error = "the request is not a JSON object";
        return false;
    }
    Request r;
    r.raw = json;
    const std::string mode = json.value("mode", std::string{"run"});
    if (mode == "run") {
        r.mode = Request::Mode::Run;
    } else if (mode == "capture_save") {
        r.mode = Request::Mode::CaptureSave;
    } else if (mode == "record") {
        r.mode = Request::Mode::Record;
    } else if (mode == "probe") {
        r.mode = Request::Mode::Probe;
    } else {
        error = "unknown mode \"" + mode + "\"";
        return false;
    }
    r.run = json.value("run", std::string{});
    r.game = json.value("game", std::string{});
    if (r.game.empty() && r.mode != Request::Mode::Probe) {
        error = "\"game\" is missing (a program id or a path under Documents)";
        return false;
    }
    if (const auto it = json.find("save"); it != json.end()) {
        if (it->is_string()) {
            r.save = it->get<std::string>();
        } else if (it->is_object()) {
            r.save = it->value("name", std::string{});
            r.save_slot = it->value("slot", 0);
            r.restore_owner_save = it->value("restore_owner_save", true);
        }
        if (r.save.find("..") != std::string::npos || r.save.find('/') != std::string::npos) {
            error = "\"save\" names a folder in Documents/bench/saves";
            return false;
        }
    }
    if (r.mode == Request::Mode::CaptureSave && r.save.empty()) {
        error = "capture_save needs \"save\" (the folder name to write)";
        return false;
    }
    if (const auto it = json.find("experiments"); it != json.end() && it->is_array()) {
        for (const auto& name : *it) {
            if (name.is_string() && !name.get<std::string>().empty()) {
                r.experiments.push_back(name.get<std::string>());
            }
        }
    }
    r.keep_saved_experiments = json.value("keep_saved_experiments", false);
    if (const auto it = json.find("env"); it != json.end()) {
        r.env = ReadPairs(*it);
    }
    if (const auto it = json.find("settings"); it != json.end()) {
        r.settings = ReadPairs(*it);
    }
    r.log_filter = json.value("log_filter", std::string{});
    if (const auto it = json.find("jit"); it != json.end() && it->is_object()) {
        r.jit_provider = it->value("provider", r.jit_provider);
        r.jit_timeout_s = Number(*it, "timeout_s", r.jit_timeout_s);
    }
    if (r.jit_provider != "stikdebug" && r.jit_provider != "external") {
        error = "jit.provider is \"stikdebug\" or \"external\"";
        return false;
    }
    if (const auto it = json.find("thermal"); it != json.end() && it->is_object()) {
        r.thermal_max_state = it->value("max_state", r.thermal_max_state);
        r.thermal_wait_s = Number(*it, "wait_s", r.thermal_wait_s);
    }
    r.boot_timeout_s = Number(json, "boot_timeout_s", r.boot_timeout_s);
    r.script_timeout_s = Number(json, "script_timeout_s", r.script_timeout_s);
    if (!ReadScript(json, "script", base, r.script, error)) {
        return false;
    }
    if (const auto it = json.find("steady"); it != json.end() && it->is_object()) {
        r.steady.jit_percent_max = Number(*it, "jit_percent_max", r.steady.jit_percent_max);
        r.steady.upload_mib_s_max = Number(*it, "upload_mib_s_max", r.steady.upload_mib_s_max);
        r.steady.hold_s = std::max(1.0, Number(*it, "hold_s", r.steady.hold_s));
        r.steady.timeout_s = Number(*it, "timeout_s", r.steady.timeout_s);
        r.steady.min_fps = Number(*it, "min_fps", r.steady.min_fps);
    }
    if (const auto it = json.find("measure"); it != json.end() && it->is_object()) {
        r.stand_s = Number(*it, "stand_s", r.stand_s);
        r.pan_settle_s = Number(*it, "pan_settle_s", r.pan_settle_s);
        if (!ReadScript(*it, "pan", base, r.pan, error)) {
            return false;
        }
    }
    r.record_s = Number(json, "record_s", r.record_s);
    r.exit_app = json.value("exit", true);
    out = std::move(r);
    return true;
}

Player::Player(std::vector<Step> steps_) : steps{std::move(steps_)} {}

std::vector<Action> Player::Advance(const PlayerContext& ctx) {
    std::vector<Action> actions;
    const auto buttons = [&](const Step& step, bool pressed) {
        for (const int button : step.buttons) {
            actions.push_back({Action::Kind::Button, button, pressed, 0, 0.0f, 0.0f, {}});
        }
    };
    const auto wait = [&](double ms, std::uint64_t frames) {
        by_frames = frames != 0;
        until_frames = ctx.frames + frames;
        until_ms = ctx.now_ms + ms;
        phase = Phase::Waiting;
    };
    const auto elapsed = [&] {
        return by_frames ? ctx.frames >= until_frames : ctx.now_ms >= until_ms;
    };
    const auto next = [&] {
        ++index;
        phase = Phase::Start;
    };
    // Bounded: a script of only marks and logs finishes in one call.
    for (std::size_t guard = 0; index < steps.size() && guard < steps.size() + 4; ++guard) {
        const Step& step = steps[index];
        switch (step.kind) {
        case Step::Kind::Wait:
        case Step::Kind::WaitFrames:
            if (phase == Phase::Start) {
                wait(step.kind == Step::Kind::Wait ? step.ms : 0.0, step.frames);
            }
            if (!elapsed()) {
                return actions;
            }
            next();
            continue;
        case Step::Kind::WaitRunning:
            if (!ctx.running) {
                return actions;
            }
            next();
            continue;
        case Step::Kind::WaitQuiet:
            if (!ctx.running || ctx.quiet_since_ms < 0 || ctx.now_ms - ctx.quiet_since_ms < step.ms) {
                return actions;
            }
            next();
            continue;
        case Step::Kind::Press:
            if (phase == Phase::Start) {
                buttons(step, true);
                wait(step.ms, step.frames);
                return actions;
            }
            if (phase == Phase::Waiting) {
                if (!elapsed()) {
                    return actions;
                }
                buttons(step, false);
                if (step.after_ms <= 0) {
                    next();
                    continue;
                }
                wait(step.after_ms, 0);
                phase = Phase::Released;
                return actions;
            }
            if (!elapsed()) {
                return actions;
            }
            next();
            continue;
        case Step::Kind::Hold:
        case Step::Kind::Release:
        case Step::Kind::Mark:
        case Step::Kind::Log:
            if (phase == Phase::Start) {
                if (step.kind == Step::Kind::Hold || step.kind == Step::Kind::Release) {
                    buttons(step, step.kind == Step::Kind::Hold);
                } else {
                    actions.push_back({step.kind == Step::Kind::Mark ? Action::Kind::Mark
                                                                     : Action::Kind::Log,
                                       0, false, 0, 0.0f, 0.0f, step.text});
                }
                if (step.after_ms <= 0) {
                    next();
                    continue;
                }
                wait(step.after_ms, 0);
                phase = Phase::Released;
            }
            if (!elapsed()) {
                return actions;
            }
            next();
            continue;
        case Step::Kind::Stick:
            if (phase == Phase::Start) {
                actions.push_back({Action::Kind::Stick, 0, false, step.stick, step.x, step.y, {}});
                if (step.keep) {
                    if (step.after_ms <= 0) {
                        next();
                        continue;
                    }
                    wait(step.after_ms, 0);
                    phase = Phase::Released;
                    return actions;
                }
                wait(step.ms, step.frames);
                return actions;
            }
            if (phase == Phase::Waiting) {
                if (!elapsed()) {
                    return actions;
                }
                actions.push_back({Action::Kind::Stick, 0, false, step.stick, 0.0f, 0.0f, {}});
                if (step.after_ms <= 0) {
                    next();
                    continue;
                }
                wait(step.after_ms, 0);
                phase = Phase::Released;
                return actions;
            }
            if (!elapsed()) {
                return actions;
            }
            next();
            continue;
        }
    }
    return actions;
}

void Recorder::Gap(double now_ms) {
    if (now_ms - last_ms >= 1.0) {
        Step wait;
        wait.kind = Step::Kind::Wait;
        wait.ms = std::round(now_ms - last_ms);
        steps.push_back(wait);
    }
    last_ms = std::max(last_ms, now_ms);
}

void Recorder::Emit(double now_ms, int stick, float x, float y) {
    Gap(now_ms);
    Step step;
    step.kind = Step::Kind::Stick;
    step.stick = stick;
    step.x = x;
    step.y = y;
    step.keep = true;
    steps.push_back(step);
    StickState& state = sticks[stick];
    state.x = x;
    state.y = y;
    state.ms = now_ms;
    ++events;
}

void Recorder::FlushPending(int stick) {
    StickState& state = sticks[stick];
    if (!state.pending) {
        return;
    }
    state.pending = false;
    if (std::abs(state.pending_x - state.x) >= 0.05f || std::abs(state.pending_y - state.y) >= 0.05f) {
        Emit(state.pending_ms, stick, state.pending_x, state.pending_y);
    }
}

void Recorder::Button(double now_ms, int button, bool pressed) {
    FlushPending(0);
    FlushPending(1);
    Gap(now_ms);
    Step step;
    step.kind = pressed ? Step::Kind::Hold : Step::Kind::Release;
    step.buttons = {button};
    steps.push_back(step);
    ++events;
}

void Recorder::Stick(double now_ms, int stick, float x, float y) {
    if (stick != 0 && stick != 1) {
        return;
    }
    StickState& state = sticks[stick];
    if (std::abs(x) < 0.01f && std::abs(y) < 0.01f) {
        state.pending = false;
        if (state.x != 0.0f || state.y != 0.0f) {
            Emit(now_ms, stick, 0.0f, 0.0f);
        }
        return;
    }
    if (state.ms >= 0 && now_ms - state.ms < 50.0) {
        state.pending = true;
        state.pending_x = x;
        state.pending_y = y;
        state.pending_ms = now_ms;
        return;
    }
    state.pending = false;
    if (std::abs(x - state.x) >= 0.05f || std::abs(y - state.y) >= 0.05f) {
        Emit(now_ms, stick, x, y);
    }
}

std::vector<Step> Recorder::Finish(double now_ms) {
    FlushPending(0);
    FlushPending(1);
    Gap(now_ms);
    std::vector<Step> out = std::move(steps);
    steps.clear();
    return out;
}

double JitPercent(const Counters& a, const Counters& b) {
    const double seconds = b.t_s - a.t_s;
    return seconds > 0 ? Delta(b.jit_compile_ns, a.jit_compile_ns) / 1e7 / seconds : 0.0;
}

double UploadMibPerSecond(const Counters& a, const Counters& b) {
    const double seconds = b.t_s - a.t_s;
    const double bytes = Delta(b.texture_upload_bytes, a.texture_upload_bytes) +
                         Delta(b.buffer_upload_bytes, a.buffer_upload_bytes);
    return seconds > 0 ? bytes / (1 << 20) / seconds : 0.0;
}

void SteadyDetector::Add(const Counters& sample) {
    samples.push_back(sample);
    // A little more than the hold window.
    while (samples.size() > 2 && samples.front().t_s < sample.t_s - config.hold_s - 2.0) {
        samples.erase(samples.begin());
    }
}

bool SteadyDetector::Steady() const {
    if (samples.size() < 2) {
        return false;
    }
    const double end = samples.back().t_s;
    if (end - samples.front().t_s < config.hold_s) {
        return false;
    }
    for (std::size_t i = 1; i < samples.size(); ++i) {
        const Counters& a = samples[i - 1];
        const Counters& b = samples[i];
        if (b.t_s <= end - config.hold_s) {
            continue;
        }
        const double seconds = b.t_s - a.t_s;
        if (seconds <= 0) {
            continue;
        }
        if (JitPercent(a, b) >= config.jit_percent_max ||
            UploadMibPerSecond(a, b) >= config.upload_mib_s_max ||
            Delta(b.frames, a.frames) / seconds < config.min_fps) {
            return false;
        }
    }
    return true;
}

double SteadyDetector::LastJitPercent() const {
    return samples.size() < 2 ? 100.0 : JitPercent(samples[samples.size() - 2], samples.back());
}

std::string SteadyDetector::Describe() const {
    if (samples.size() < 2) {
        return "no samples yet";
    }
    const Counters& a = samples[samples.size() - 2];
    const Counters& b = samples.back();
    const double seconds = b.t_s - a.t_s;
    char text[128];
    std::snprintf(text, sizeof(text), "JIT %.1f %%, uploads %.1f MiB/s, %.1f fps",
                  JitPercent(a, b), UploadMibPerSecond(a, b),
                  seconds > 0 ? Delta(b.frames, a.frames) / seconds : 0.0);
    return text;
}

FrameStats ComputeFrameStats(const std::vector<std::uint64_t>& stamps_ns, std::uint64_t start_ns,
                             std::uint64_t end_ns) {
    FrameStats stats;
    stats.duration_s = end_ns > start_ns ? static_cast<double>(end_ns - start_ns) / 1e9 : 0.0;
    std::vector<double> times;
    std::uint64_t previous = 0;
    for (const std::uint64_t stamp : stamps_ns) {
        if (stamp < start_ns || stamp > end_ns) {
            continue;
        }
        ++stats.frames;
        if (previous != 0 && stamp >= previous) {
            times.push_back(static_cast<double>(stamp - previous) / 1e6);
        }
        previous = stamp;
    }
    if (stats.duration_s > 0) {
        stats.fps_avg = static_cast<double>(stats.frames) / stats.duration_s;
    }
    if (times.empty()) {
        return stats;
    }
    std::sort(times.begin(), times.end());
    stats.p50_ms = Percentile(times, 0.50);
    stats.p90_ms = Percentile(times, 0.90);
    stats.p95_ms = Percentile(times, 0.95);
    stats.p99_ms = Percentile(times, 0.99);
    stats.max_ms = times.back();
    const std::size_t worst = std::max<std::size_t>(1, times.size() / 100);
    const double worst_mean =
        std::accumulate(times.end() - static_cast<std::ptrdiff_t>(worst), times.end(), 0.0) /
        static_cast<double>(worst);
    stats.fps_1pct_low = worst_mean > 0 ? 1000.0 / worst_mean : 0.0;
    return stats;
}

nlohmann::json FrameStatsJson(const FrameStats& stats) {
    return {
        {"frames", stats.frames},
        {"duration_s", Round(stats.duration_s)},
        {"fps_avg", Round(stats.fps_avg)},
        {"fps_1pct_low", Round(stats.fps_1pct_low)},
        {"frametime_ms",
         {{"p50", Round(stats.p50_ms)},
          {"p90", Round(stats.p90_ms)},
          {"p95", Round(stats.p95_ms)},
          {"p99", Round(stats.p99_ms)},
          {"max", Round(stats.max_ms)}}},
    };
}

nlohmann::json WindowCountersJson(const Counters& a, const Counters& b) {
    const double seconds = b.t_s - a.t_s;
    const double frames = Delta(b.frames, a.frames);
    const auto per_frame = [&](std::uint64_t after, std::uint64_t before) {
        return frames > 0 ? Round(Delta(after, before) / frames) : 0.0;
    };
    const auto wall_percent = [&](std::uint64_t after, std::uint64_t before) {
        return seconds > 0 ? Round(Delta(after, before) / 1e7 / seconds, 10.0) : 0.0;
    };
    const auto mib_s = [&](std::uint64_t after, std::uint64_t before) {
        return seconds > 0 ? Round(Delta(after, before) / (1 << 20) / seconds) : 0.0;
    };
    nlohmann::json out{
        {"gpu_ms_per_frame",
         b.gpu_buffers > a.gpu_buffers && frames > 0
             ? nlohmann::json(Round(Delta(b.gpu_busy_ns, a.gpu_busy_ns) / 1e6 / frames))
             : nlohmann::json(nullptr)},
        {"gpu_busy_pct", wall_percent(b.gpu_busy_ns, a.gpu_busy_ns)},
        {"command_buffers_per_frame", per_frame(b.gpu_buffers, a.gpu_buffers)},
        {"draws_per_frame", per_frame(b.draws, a.draws)},
        {"render_passes_per_frame", per_frame(b.render_passes, a.render_passes)},
        {"jit_compile_pct", Round(JitPercent(a, b))},
        {"jit_blocks_per_s", seconds > 0 ? Round(Delta(b.jit_blocks, a.jit_blocks) / seconds) : 0.0},
        {"upload_mib_s",
         {{"textures", mib_s(b.texture_upload_bytes, a.texture_upload_bytes)},
          {"buffers", mib_s(b.buffer_upload_bytes, a.buffer_upload_bytes)},
          {"all_staging", mib_s(b.staging_upload_bytes, a.staging_upload_bytes)}}},
        {"guest_cores_pct",
         {{"guest_code", wall_percent(b.guest_run_ns, a.guest_run_ns)},
          {"kernel", wall_percent(b.guest_kernel_ns, a.guest_kernel_ns)},
          {"idle", wall_percent(b.guest_idle_ns, a.guest_idle_ns)}}},
    };
    if (a.cpu_known && b.cpu_known && b.cpu_ns > a.cpu_ns) {
        const double cpu_ns = Delta(b.cpu_ns, a.cpu_ns);
        out["cpu"] = {
            {"process_pct", wall_percent(b.cpu_ns, a.cpu_ns)},
            {"performance_core_share_pct",
             Round(100.0 * Delta(b.performance_ns, a.performance_ns) / cpu_ns, 10.0)},
            {"watts", seconds > 0 ? Round(Delta(b.energy_nj, a.energy_nj) / 1e9 / seconds) : 0.0},
            {"runnable_not_running_pct", wall_percent(b.runnable_ns, a.runnable_ns)},
        };
    } else {
        out["cpu"] = nullptr;
    }
    return out;
}

} // namespace LichiiBench
