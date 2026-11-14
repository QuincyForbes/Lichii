// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ios_frontend/bench_core.h"

namespace {

using namespace LichiiBench;
using nlohmann::json;

std::vector<Step> Steps(const char* text) {
    std::vector<Step> steps;
    std::string error;
    REQUIRE(ParseSteps(json::parse(text), steps, error));
    return steps;
}

PlayerContext At(double ms, std::uint64_t frames = 0, bool running = true) {
    return PlayerContext{ms, frames, running, -1.0};
}

// jit_percent_seconds grows by the JIT's % of a core each second.
Counters Sample(double t, std::uint64_t frames, double jit_percent_seconds,
                std::uint64_t upload_bytes) {
    Counters c;
    c.t_s = t;
    c.frames = frames;
    c.jit_compile_ns = static_cast<std::uint64_t>(jit_percent_seconds * 1e7);
    c.texture_upload_bytes = upload_bytes;
    return c;
}

} // Anonymous namespace

TEST_CASE("Bench: button names follow LichiiButton", "[ios_bench]") {
    CHECK(ButtonFromName("A") == 0);
    CHECK(ButtonFromName("zl") == 8);
    CHECK(ButtonFromName("PLUS") == 10);
    CHECK(ButtonFromName("+") == 10);
    CHECK(ButtonFromName("dup") == 13);
    CHECK(ButtonFromName("DOWN") == 15);
    CHECK(ButtonFromName("CAPTURE") == 19);
    CHECK(ButtonFromName("nope") == -1);
    CHECK(std::string{ButtonName(9)} == "ZR");
}

TEST_CASE("Bench: a press holds, releases and waits", "[ios_bench]") {
    Player player{Steps(R"([{"press": "A", "hold_ms": 100, "after_ms": 50}, {"press": ["ZL", "B"]}])")};
    auto actions = player.Advance(At(0));
    REQUIRE(actions.size() == 1);
    CHECK(actions[0].kind == Action::Kind::Button);
    CHECK(actions[0].button == 0);
    CHECK(actions[0].pressed);
    CHECK(player.Advance(At(99)).empty());
    actions = player.Advance(At(100));
    REQUIRE(actions.size() == 1);
    CHECK_FALSE(actions[0].pressed);
    CHECK(player.Advance(At(149)).empty());
    actions = player.Advance(At(150));
    REQUIRE(actions.size() == 2);
    CHECK(actions[0].button == 8);
    CHECK(actions[1].button == 1);
    CHECK(actions[1].pressed);
    actions = player.Advance(At(250));
    REQUIRE(actions.size() == 2);
    CHECK_FALSE(actions[0].pressed);
    CHECK(player.Done());
}

TEST_CASE("Bench: waits by frames, for running and for a quiet JIT", "[ios_bench]") {
    Player player{Steps(R"([{"wait_running": true}, {"wait_frames": 30}, {"wait_quiet_s": 3},
                            {"mark": "kokiri"}])")};
    CHECK(player.Advance(At(0, 0, false)).empty());
    CHECK(player.Index() == 0);
    CHECK(player.Advance(At(10, 5, true)).empty());
    CHECK(player.Index() == 1);
    CHECK(player.Advance(At(5000, 34, true)).empty());
    CHECK(player.Index() == 1);
    CHECK(player.Advance(At(5001, 35, true)).empty());
    CHECK(player.Index() == 2);
    PlayerContext quiet = At(6000, 40, true);
    quiet.quiet_since_ms = 4000;
    CHECK(player.Advance(quiet).empty());
    quiet.now_ms = 7000;
    const auto actions = player.Advance(quiet);
    REQUIRE(actions.size() == 1);
    CHECK(actions[0].kind == Action::Kind::Mark);
    CHECK(actions[0].text == "kokiri");
    CHECK(player.Done());
}

TEST_CASE("Bench: a stick returns to the centre unless kept", "[ios_bench]") {
    Player player{Steps(R"([{"stick": "right", "x": 1, "y": 0, "s": 8}, {"stick": "left", "y": 1, "keep": true}])")};
    auto actions = player.Advance(At(0));
    REQUIRE(actions.size() == 1);
    CHECK(actions[0].stick == 1);
    CHECK(actions[0].x == 1.0f);
    actions = player.Advance(At(8000));
    REQUIRE(actions.size() == 2);
    CHECK(actions[0].x == 0.0f);
    CHECK(actions[1].stick == 0);
    CHECK(actions[1].y == 1.0f);
    CHECK(player.Done());
}

TEST_CASE("Bench: repeat blocks are flattened", "[ios_bench]") {
    const auto steps =
        Steps(R"({"steps": [{"repeat": 3, "steps": [{"press": "DDOWN"}, {"wait_ms": 200}]}]})");
    REQUIRE(steps.size() == 6);
    CHECK(steps[4].kind == Step::Kind::Press);
    CHECK(steps[4].buttons == std::vector<int>{15});
}

TEST_CASE("Bench: bad steps are refused with a reason", "[ios_bench]") {
    std::vector<Step> steps;
    std::string error;
    CHECK_FALSE(ParseSteps(json::parse(R"([{"press": "Q"}])"), steps, error));
    CHECK(error.find("unknown button") != std::string::npos);
    CHECK_FALSE(ParseSteps(json::parse(R"([{"jump": 1}])"), steps, error));
    CHECK_FALSE(ParseSteps(json::parse(R"([{"stick": "middle"}])"), steps, error));
    CHECK_FALSE(ParseSteps(json::parse(R"({"no": []})"), steps, error));
}

TEST_CASE("Bench: steps survive a round trip through JSON", "[ios_bench]") {
    const auto steps = Steps(R"([{"wait_running": true}, {"press": "A", "hold_frames": 2},
        {"stick": "left", "x": 0.5, "y": -0.25, "ms": 300, "after_ms": 20}, {"hold": "R"},
        {"release": "R"}, {"wait_quiet_s": 2}, {"log": "hi"}])");
    std::vector<Step> again;
    std::string error;
    REQUIRE(ParseSteps(StepsToJson(steps), again, error));
    REQUIRE(again.size() == steps.size());
    CHECK(again[1].frames == 2);
    CHECK(again[2].x == 0.5f);
    CHECK(again[2].after_ms == 20.0);
    CHECK(again[5].ms == 2000.0);
}

TEST_CASE("Bench: the recorder replays what it saw", "[ios_bench]") {
    Recorder recorder;
    recorder.Button(1000, 0, true);
    recorder.Button(1120, 0, false);
    recorder.Stick(1500, 0, 0.0f, 1.0f);
    recorder.Stick(1510, 0, 0.0f, 0.98f); // under 50 ms and under 0.05: dropped
    recorder.Stick(1700, 0, 0.6f, 0.6f);
    recorder.Stick(2000, 0, 0.0f, 0.0f);
    const auto steps = recorder.Finish(2500);
    CHECK(recorder.Events() == 5);

    Player player{steps};
    std::vector<Action> seen;
    for (double t = 0; t <= 3000; t += 10) {
        for (auto& action : player.Advance(At(t))) {
            seen.push_back(action);
        }
    }
    REQUIRE(seen.size() == 5);
    CHECK(seen[0].pressed);
    CHECK_FALSE(seen[1].pressed);
    CHECK(seen[2].y == 1.0f);
    CHECK(seen[3].x == 0.6f);
    CHECK(seen[4].x == 0.0f);
    // The recording starts at 0 ms: 1 s to the press, 120 ms held, then 380 ms to the stick.
    CHECK(steps[0].kind == Step::Kind::Wait);
    CHECK(steps[0].ms == 1000.0);
    CHECK(steps[2].ms == 120.0);
    CHECK(steps[4].ms == 380.0);
}

TEST_CASE("Bench: steady only after a whole quiet hold window", "[ios_bench]") {
    SteadyConfig config;
    config.hold_s = 10;
    SteadyDetector detector{config};
    std::uint64_t frames = 0;
    double jit = 0;
    std::uint64_t uploads = 0;
    // Loading: the JIT compiles 40 % of a core.
    for (int t = 0; t <= 20; ++t) {
        detector.Add(Sample(t, frames, jit, uploads));
        frames += 30;
        jit += 40;
    }
    CHECK_FALSE(detector.Steady());
    // Quiet from 20 s on: 1 % JIT, 1 MiB/s uploads.
    for (int t = 21; t <= 29; ++t) {
        jit += 1;
        uploads += 1 << 20;
        detector.Add(Sample(t, frames, jit, uploads));
        frames += 30;
    }
    CHECK_FALSE(detector.Steady());
    jit += 1;
    detector.Add(Sample(30, frames, jit, uploads));
    frames += 30;
    jit += 1;
    detector.Add(Sample(31, frames, jit, uploads));
    CHECK(detector.Steady());
    CHECK(detector.LastJitPercent() == Catch::Approx(1.0));
    // One upload burst in the window breaks it again.
    uploads += 64u << 20;
    detector.Add(Sample(32, frames + 30, jit, uploads));
    CHECK_FALSE(detector.Steady());
}

TEST_CASE("Bench: a stalled game is never steady", "[ios_bench]") {
    SteadyDetector detector{SteadyConfig{}};
    for (int t = 0; t <= 15; ++t) {
        detector.Add(Sample(t, 100, 0, 0));
    }
    CHECK_FALSE(detector.Steady());
}

TEST_CASE("Bench: frame statistics", "[ios_bench]") {
    std::vector<std::uint64_t> stamps;
    std::uint64_t t = 1'000'000'000;
    // 99 frames of 33.3 ms and one of 100 ms.
    for (int i = 0; i < 100; ++i) {
        t += i == 50 ? 100'000'000 : 33'333'333;
        stamps.push_back(t);
    }
    const auto stats = ComputeFrameStats(stamps, stamps.front(), stamps.back());
    CHECK(stats.frames == 100);
    CHECK(stats.p50_ms == Catch::Approx(33.333).epsilon(0.001));
    CHECK(stats.max_ms == Catch::Approx(100.0));
    CHECK(stats.fps_1pct_low == Catch::Approx(10.0));
    CHECK(stats.fps_avg == Catch::Approx(100.0 / stats.duration_s));
    const auto json = FrameStatsJson(stats);
    CHECK(json["frametime_ms"]["max"] == 100.0);

    const auto empty = ComputeFrameStats({}, 0, 1'000'000'000);
    CHECK(empty.frames == 0);
    CHECK(empty.fps_avg == 0.0);
}

TEST_CASE("Bench: window rates", "[ios_bench]") {
    Counters a;
    a.t_s = 10;
    Counters b = a;
    b.t_s = 20;
    b.frames = 300;
    b.gpu_busy_ns = 300ull * 20'000'000;
    b.gpu_buffers = 3000;
    b.draws = 300 * 1800;
    b.jit_compile_ns = 50'000'000; // 0.5 % of a core over 10 s
    b.texture_upload_bytes = 10u << 20;
    a.cpu_known = b.cpu_known = true;
    b.cpu_ns = 30'000'000'000;     // 300 %
    b.performance_ns = 6'000'000'000;
    b.energy_nj = 15'000'000'000;
    const auto json = WindowCountersJson(a, b);
    CHECK(json["gpu_ms_per_frame"] == 20.0);
    CHECK(json["gpu_busy_pct"] == 60.0);
    CHECK(json["draws_per_frame"] == 1800.0);
    CHECK(json["jit_compile_pct"] == 0.5);
    CHECK(json["upload_mib_s"]["textures"] == 1.0);
    CHECK(json["cpu"]["process_pct"] == 300.0);
    CHECK(json["cpu"]["performance_core_share_pct"] == 20.0);
    CHECK(json["cpu"]["watts"] == 1.5);
    a.cpu_known = false;
    CHECK(WindowCountersJson(a, b)["cpu"].is_null());
}

TEST_CASE("Bench: requests", "[ios_bench]") {
    const auto dir = std::filesystem::temp_directory_path() / "lichii_bench_core_test";
    std::filesystem::create_directories(dir / "scripts");
    std::ofstream{dir / "scripts" / "pan.json"} << R"([{"stick": "right", "x": 1, "s": 6}])";

    Request request;
    std::string error;
    REQUIRE(ParseRequest(json::parse(R"({
        "run": "guest_cores_first-2", "game": "0400486024DEC000",
        "save": {"name": "kokiri", "slot": 1},
        "experiments": ["guest_cores_first"], "env": {"EDEN_NX2_X": 1, "EDEN_NX2_Y": "a"},
        "settings": {"oot_graphics_preset": "0", "use_docked_mode": false},
        "thermal": {"max_state": 1, "wait_s": 120},
        "script": [{"wait_running": true}, {"press": "A"}],
        "steady": {"hold_s": 12},
        "measure": {"stand_s": 60, "pan": "scripts/pan.json"}
    })"), dir, request, error));
    CHECK(request.mode == Request::Mode::Run);
    CHECK(request.save == "kokiri");
    CHECK(request.save_slot == 1);
    CHECK(request.experiments == std::vector<std::string>{"guest_cores_first"});
    REQUIRE(request.env.size() == 2);
    CHECK(request.env[0].second == "1");
    CHECK(request.settings[1].second == "false");
    CHECK(request.thermal_wait_s == 120.0);
    CHECK(request.script.size() == 2);
    CHECK(request.steady.hold_s == 12.0);
    CHECK(request.stand_s == 60.0);
    REQUIRE(request.pan.size() == 1);
    CHECK(request.pan[0].ms == 6000.0);
    CHECK(ParseProgramId(request.game) == 0x0400486024DEC000ull);
    CHECK_FALSE(ParseProgramId("Games/oot.nup").has_value());

    CHECK_FALSE(ParseRequest(json::parse(R"({"mode": "run"})"), dir, request, error));
    CHECK_FALSE(ParseRequest(json::parse(R"({"game": "x", "save": "../y"})"), dir, request, error));
    CHECK_FALSE(ParseRequest(json::parse(R"({"game": "x", "mode": "capture_save"})"), dir, request, error));
    CHECK_FALSE(ParseRequest(json::parse(R"({"game": "x", "jit": {"provider": "magic"}})"), dir, request, error));
    CHECK_FALSE(ParseRequest(json::parse(R"({"game": "x", "script": "missing.json"})"), dir, request, error));
    REQUIRE(ParseRequest(json::parse(R"({"mode": "probe"})"), dir, request, error));
    CHECK(request.mode == Request::Mode::Probe);
    std::filesystem::remove_all(dir);
}
