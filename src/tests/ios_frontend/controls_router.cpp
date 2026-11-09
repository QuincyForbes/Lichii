// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ios_frontend/controls_router.h"

namespace {

using namespace LichiiControls;

constexpr int ButtonA = 0;
constexpr int ButtonB = 1;
constexpr int ButtonPlus = 10;
constexpr int ButtonZR = 9;

struct Sink {
    std::array<std::array<bool, ButtonCount>, PlayerCount> buttons{};
    std::array<std::array<std::array<float, 2>, 2>, PlayerCount> sticks{};
    int button_events = 0;

    Router MakeRouter() {
        return Router{[this](int player, int button, bool pressed) {
                          buttons[player][button] = pressed;
                          ++button_events;
                      },
                      [this](int player, int stick, float x, float y) {
                          sticks[player][stick] = {x, y};
                      }};
    }
};

} // Anonymous namespace

TEST_CASE("Touch and controller input combine per player", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int slot = router.Connect(LICHII_PAD_KIND_XBOX);
    REQUIRE(slot == 0);
    REQUIRE(router.Config(slot).player == 0);

    router.TouchButton(0, ButtonA, true);
    router.Element(slot, LICHII_PAD_EAST, true); // by position: EAST is A
    REQUIRE(sink.buttons[0][ButtonA]);
    router.TouchButton(0, ButtonA, false);
    REQUIRE(sink.buttons[0][ButtonA]);
    router.Element(slot, LICHII_PAD_EAST, false);
    REQUIRE_FALSE(sink.buttons[0][ButtonA]);

    router.Element(slot, LICHII_PAD_SOUTH, true);
    REQUIRE(sink.buttons[0][ButtonB]);
    router.Disconnect(slot);
    REQUIRE_FALSE(sink.buttons[0][ButtonB]);
}

TEST_CASE("Only changes are sent", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    router.TouchButton(0, ButtonA, true);
    const int events = sink.button_events;
    router.TouchButton(0, ButtonA, true);
    REQUIRE(sink.button_events == events);
}

TEST_CASE("Second controller goes to player 2 and can be reassigned", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int first = router.Connect(LICHII_PAD_KIND_DUALSENSE);
    const int second = router.Connect(LICHII_PAD_KIND_SWITCH_PRO);
    REQUIRE(router.Config(second).player == 1);

    router.Element(second, LICHII_PAD_EAST, true);
    REQUIRE(sink.buttons[1][ButtonA]);
    router.SetPlayer(second, 0);
    REQUIRE_FALSE(sink.buttons[1][ButtonA]);
    REQUIRE(sink.buttons[0][ButtonA]);
    (void)first;
}

TEST_CASE("Mapping by label and remapping", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int slot = router.Connect(LICHII_PAD_KIND_XBOX);
    PadConfig config = router.Config(slot);
    config.map = DefaultMapping(LICHII_PAD_KIND_XBOX, true);
    config.map[LICHII_PAD_PADDLE1] = ButtonPlus;
    config.map[LICHII_PAD_NORTH] = 1234; // invalid targets are dropped
    router.SetConfig(slot, config);

    REQUIRE(router.Config(slot).map[LICHII_PAD_NORTH] == LICHII_ACTION_NONE);
    router.Element(slot, LICHII_PAD_SOUTH, true);
    REQUIRE(sink.buttons[0][ButtonA]);
    router.Element(slot, LICHII_PAD_PADDLE1, true);
    REQUIRE(sink.buttons[0][ButtonPlus]);
}

TEST_CASE("Analog triggers press at the threshold", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int slot = router.Connect(LICHII_PAD_KIND_DUALSENSE);
    router.Axis(slot, LICHII_PAD_AXIS_R2, 0.2f);
    REQUIRE_FALSE(sink.buttons[0][ButtonZR]);
    router.Axis(slot, LICHII_PAD_AXIS_R2, 0.5f);
    REQUIRE(sink.buttons[0][ButtonZR]);
}

TEST_CASE("Radial deadzone rescales", "[ios][controls]") {
    float x = 0.05f, y = 0.0f;
    ApplyRadialDeadzone(x, y, 0.1f, 0.9f);
    REQUIRE(x == 0.0f);

    x = 0.5f;
    y = 0.0f;
    ApplyRadialDeadzone(x, y, 0.1f, 0.9f);
    REQUIRE(x == Catch::Approx(0.5f));

    x = 0.7f;
    y = 0.7f;
    ApplyRadialDeadzone(x, y, 0.1f, 0.9f);
    REQUIRE(std::sqrt(x * x + y * y) == Catch::Approx(1.0f));
    REQUIRE(x == Catch::Approx(y));
}

TEST_CASE("Strongest stick wins and sticks can swap", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int slot = router.Connect(LICHII_PAD_KIND_XBOX);
    router.TouchStick(0, 0, 0.3f, 0.0f);
    router.Axis(slot, LICHII_PAD_AXIS_LX, -1.0f);
    REQUIRE(sink.sticks[0][0][0] == Catch::Approx(-1.0f));
    router.Axis(slot, LICHII_PAD_AXIS_LX, 0.0f);
    REQUIRE(sink.sticks[0][0][0] == Catch::Approx(0.3f));

    PadConfig config = router.Config(slot);
    config.swap_sticks = true;
    config.sticks[1].invert_y = true;
    router.SetConfig(slot, config);
    router.Axis(slot, LICHII_PAD_AXIS_RY, 1.0f);
    REQUIRE(sink.sticks[0][0][1] == Catch::Approx(-1.0f));
}

TEST_CASE("Menu chord and action mappings", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int slot = router.Connect(LICHII_PAD_KIND_XBOX);
    router.Element(slot, LICHII_PAD_MENU, true);
    REQUIRE(sink.buttons[0][ButtonPlus]);
    router.Element(slot, LICHII_PAD_OPTIONS, true);
    REQUIRE_FALSE(sink.buttons[0][ButtonPlus]);
    REQUIRE(router.TakeActions() == 1u);
    REQUIRE(router.TakeActions() == 0u);
    router.Element(slot, LICHII_PAD_MENU, false);
    router.Element(slot, LICHII_PAD_OPTIONS, false);

    router.SetConfig(slot, [&] {
        PadConfig config = router.Config(slot);
        config.map[LICHII_PAD_PADDLE2] = LICHII_ACTION_SCREENSHOT;
        return config;
    }());
    router.Element(slot, LICHII_PAD_PADDLE2, true);
    REQUIRE(router.TakeActions() == (1u << (LICHII_ACTION_SCREENSHOT - LICHII_ACTION_MENU)));
}

TEST_CASE("Releasing touch keeps controller input", "[ios][controls]") {
    Sink sink;
    Router router = sink.MakeRouter();
    const int slot = router.Connect(LICHII_PAD_KIND_GENERIC);
    router.TouchButton(0, ButtonB, true);
    router.Element(slot, LICHII_PAD_EAST, true);
    router.ReleaseTouch();
    REQUIRE_FALSE(sink.buttons[0][ButtonB]);
    REQUIRE(sink.buttons[0][ButtonA]);
}
