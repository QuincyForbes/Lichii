// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Combines the touch overlay and every physical controller assigned to a player into one
// Switch controller state per player, and reports only what changed. No engine dependencies.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>

#include "ios_frontend/lichii_bridge_controls.h"

namespace LichiiControls {

constexpr int ButtonCount = LICHII_CONTROLS_BUTTON_COUNT;
constexpr int PlayerCount = LICHII_CONTROLS_PLAYER_COUNT;
constexpr int SlotCount = LICHII_PAD_SLOT_COUNT;
constexpr int ElementCount = LICHII_PAD_ELEMENT_COUNT;
constexpr int AxisCount = LICHII_PAD_AXIS_COUNT;
constexpr int FirstAction = LICHII_ACTION_MENU;
constexpr int LastAction = LICHII_ACTION_TOGGLE_SPEED_LIMIT;

using ElementMap = std::array<int, ElementCount>;

ElementMap DefaultMapping(int kind, bool by_label);
bool IsValidTarget(int target);

/// Radial deadzone with rescaling: below `inner` reads 0, past `outer` reads 1, linear between.
void ApplyRadialDeadzone(float& x, float& y, float inner, float outer);

struct StickConfig {
    float inner = 0.10f;
    float outer = 0.95f;
    bool invert_y = false;
};

struct PadConfig {
    int player = -1;
    ElementMap map{};
    std::array<StickConfig, 2> sticks{};
    float trigger_threshold = 0.30f;
    bool swap_sticks = false;
};

struct PlayerState {
    std::array<bool, ButtonCount> buttons{};
    std::array<std::array<float, 2>, 2> sticks{};

    bool operator==(const PlayerState&) const = default;
};

class Router {
public:
    using ButtonSink = std::function<void(int player, int button, bool pressed)>;
    using StickSink = std::function<void(int player, int stick, float x, float y)>;

    Router(ButtonSink button_sink, StickSink stick_sink);

    void TouchButton(int player, int button, bool pressed);
    void TouchStick(int player, int stick, float x, float y);
    void ReleaseTouch();

    int Connect(int kind);
    void Disconnect(int slot);
    bool IsConnected(int slot) const;
    int Kind(int slot) const;
    void SetPlayer(int slot, int player);

    void Element(int slot, int element, bool pressed);
    void Axis(int slot, int axis, float value);

    /// Copy of a slot's configuration (default-constructed for unused slots).
    PadConfig Config(int slot) const;
    /// Replaces a slot's configuration (player included) and re-evaluates its input.
    void SetConfig(int slot, const PadConfig& config);

    void SetMenuChord(bool enabled);
    bool MenuChord() const;
    uint32_t TakeActions();

    /// Re-sends every player's full state (after the engine attaches or reconnects).
    void Resend();
    /// Current combined state of a player.
    PlayerState State(int player) const;

private:
    struct Pad {
        bool used = false;
        int kind = LICHII_PAD_KIND_GENERIC;
        PadConfig config{};
        std::array<bool, ElementCount> elements{};
        std::array<float, AxisCount> axes{};
        bool chord_active = false;
    };

    bool ElementHeld(const Pad& pad, int element) const;
    void UpdateActions(Pad& pad, int element, bool was_held);
    void Recompute(int player, bool force = false);
    PlayerState Combine(int player) const;

    ButtonSink button_sink;
    StickSink stick_sink;
    mutable std::recursive_mutex mutex;
    std::array<PlayerState, PlayerCount> touch{};
    std::array<PlayerState, PlayerCount> sent{};
    std::array<Pad, SlotCount> pads{};
    bool menu_chord = true;
    std::atomic<uint32_t> actions{0};
};

} // namespace LichiiControls
