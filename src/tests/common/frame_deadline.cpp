// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include <catch2/catch_test_macros.hpp>

#include "common/frame_deadline.h"

namespace {

using Common::FrameDeadline::Window;

constexpr u64 Ms = 1'000'000;
constexpr u64 Period = 33'333'333;

/// Feeds a window of frames that each cost `cost(frame)`; returns what the last one declared.
template <typename Cost>
u64 Feed(Window& window, u64& frame, u64& cpu, Cost&& cost) {
    u64 declared = 0;
    for (size_t i = 0; i < Window::Size; ++i) {
        ++frame;
        cpu += cost(i);
        const u64 result = window.Sample(frame, cpu, Period);
        REQUIRE((i + 1 == Window::Size || result == 0));
        declared = result;
    }
    return declared;
}

} // Anonymous namespace

TEST_CASE("Frame deadline: need is over the 90th percentile and the peak", "[common]") {
    std::array<u64, 10> steady{};
    steady.fill(8 * Ms);
    REQUIRE(Window::Need(steady, Period) == 10 * Ms);

    std::array<u64, 10> spike{};
    spike.fill(4 * Ms);
    spike[3] = 20 * Ms;
    REQUIRE(Window::Need(spike, Period) == 25 * Ms);

    std::array<u64, 10> heavy{};
    heavy.fill(40 * Ms);
    REQUIRE(Window::Need(heavy, Period) == Period / 10 * 9);

    std::array<u64, 10> idle{};
    REQUIRE(Window::Need(idle, Period) == Ms);
    REQUIRE(Window::Need({}, Period) == 0);
}

TEST_CASE("Frame deadline: a window declares once and again only when it moved", "[common]") {
    Window window;
    u64 frame = 100;
    u64 cpu = 5 * Ms;
    REQUIRE(window.Sample(frame, cpu, Period) == 0);
    // The same frame again adds nothing.
    REQUIRE(window.Sample(frame, cpu + Ms, Period) == 0);

    REQUIRE(Feed(window, frame, cpu, [](size_t) { return 8 * Ms; }) == 10 * Ms);
    REQUIRE(window.Declared() == 10 * Ms);
    REQUIRE(window.MeanNs() == 8 * Ms);

    // Within 15 %: kept.
    REQUIRE(Feed(window, frame, cpu, [](size_t) { return 9 * Ms; }) == 0);
    REQUIRE(window.Declared() == 10 * Ms);

    REQUIRE(Feed(window, frame, cpu, [](size_t) { return 16 * Ms; }) == 20 * Ms);
    REQUIRE(Feed(window, frame, cpu, [](size_t) { return 4 * Ms; }) == 5 * Ms);
}

TEST_CASE("Frame deadline: frames a thread sat out share its time", "[common]") {
    Window window;
    u64 frame = 0;
    u64 cpu = 0;
    REQUIRE(window.Sample(frame, cpu, Period) == 0);
    u64 declared = 0;
    for (size_t i = 0; i < Window::Size; ++i) {
        frame += 2;
        cpu += 12 * Ms;
        declared = window.Sample(frame, cpu, Period);
    }
    REQUIRE(declared == 7'500'000);
}
