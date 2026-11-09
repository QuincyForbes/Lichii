// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ios_frontend/controls_router.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace LichiiControls {

namespace {

// InputCommon::VirtualGamepad::VirtualButton values.
enum Button : int {
    ButtonA, ButtonB, ButtonX, ButtonY, StickL, StickR, TriggerL, TriggerR, TriggerZL, TriggerZR,
    ButtonPlus, ButtonMinus, ButtonLeft, ButtonUp, ButtonRight, ButtonDown, ButtonSL, ButtonSR,
    ButtonHome, ButtonCapture,
};

bool ValidPlayer(int player) {
    return player >= 0 && player < PlayerCount;
}

bool ValidSlot(int slot) {
    return slot >= 0 && slot < SlotCount;
}

float Magnitude(const std::array<float, 2>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1]);
}

} // Anonymous namespace

ElementMap DefaultMapping(int kind, bool by_label) {
    (void)kind;
    ElementMap map;
    map.fill(LICHII_ACTION_NONE);
    if (by_label) {
        map[LICHII_PAD_SOUTH] = ButtonA;
        map[LICHII_PAD_EAST] = ButtonB;
        map[LICHII_PAD_WEST] = ButtonX;
        map[LICHII_PAD_NORTH] = ButtonY;
    } else {
        map[LICHII_PAD_SOUTH] = ButtonB;
        map[LICHII_PAD_EAST] = ButtonA;
        map[LICHII_PAD_WEST] = ButtonY;
        map[LICHII_PAD_NORTH] = ButtonX;
    }
    map[LICHII_PAD_L1] = TriggerL;
    map[LICHII_PAD_R1] = TriggerR;
    map[LICHII_PAD_L2] = TriggerZL;
    map[LICHII_PAD_R2] = TriggerZR;
    map[LICHII_PAD_L3] = StickL;
    map[LICHII_PAD_R3] = StickR;
    map[LICHII_PAD_MENU] = ButtonPlus;
    map[LICHII_PAD_OPTIONS] = ButtonMinus;
    map[LICHII_PAD_HOME] = ButtonHome;
    map[LICHII_PAD_SHARE] = ButtonCapture;
    map[LICHII_PAD_DPAD_UP] = ButtonUp;
    map[LICHII_PAD_DPAD_DOWN] = ButtonDown;
    map[LICHII_PAD_DPAD_LEFT] = ButtonLeft;
    map[LICHII_PAD_DPAD_RIGHT] = ButtonRight;
    map[LICHII_PAD_TOUCHPAD] = LICHII_ACTION_MENU;
    return map;
}

bool IsValidTarget(int target) {
    return target == LICHII_ACTION_NONE || (target >= 0 && target < ButtonCount) ||
           (target >= FirstAction && target <= LastAction);
}

void ApplyRadialDeadzone(float& x, float& y, float inner, float outer) {
    inner = std::clamp(inner, 0.0f, 0.9f);
    outer = std::clamp(outer, inner + 0.05f, 1.0f);
    const float length = std::sqrt(x * x + y * y);
    if (length <= inner || length == 0.0f) {
        x = 0.0f;
        y = 0.0f;
        return;
    }
    const float scaled = std::min((length - inner) / (outer - inner), 1.0f);
    x = x / length * scaled;
    y = y / length * scaled;
}

Router::Router(ButtonSink button_sink_, StickSink stick_sink_)
    : button_sink{std::move(button_sink_)}, stick_sink{std::move(stick_sink_)} {}

void Router::TouchButton(int player, int button, bool pressed) {
    if (!ValidPlayer(player) || button < 0 || button >= ButtonCount) {
        return;
    }
    std::scoped_lock lock{mutex};
    touch[player].buttons[button] = pressed;
    Recompute(player);
}

void Router::TouchStick(int player, int stick, float x, float y) {
    if (!ValidPlayer(player) || (stick != 0 && stick != 1)) {
        return;
    }
    std::scoped_lock lock{mutex};
    touch[player].sticks[stick] = {std::clamp(x, -1.0f, 1.0f), std::clamp(y, -1.0f, 1.0f)};
    Recompute(player);
}

void Router::ReleaseTouch() {
    std::scoped_lock lock{mutex};
    for (int player = 0; player < PlayerCount; ++player) {
        touch[player] = {};
        Recompute(player);
    }
}

int Router::Connect(int kind) {
    std::scoped_lock lock{mutex};
    for (int slot = 0; slot < SlotCount; ++slot) {
        if (pads[slot].used) {
            continue;
        }
        std::array<bool, PlayerCount> taken{};
        for (const Pad& pad : pads) {
            if (pad.used && ValidPlayer(pad.config.player)) {
                taken[pad.config.player] = true;
            }
        }
        const auto free_player = std::ranges::find(taken, false);
        pads[slot] = {};
        pads[slot].used = true;
        pads[slot].kind = kind;
        pads[slot].config.map = DefaultMapping(kind, false);
        pads[slot].config.player =
            free_player == taken.end() ? -1 : static_cast<int>(free_player - taken.begin());
        return slot;
    }
    return -1;
}

void Router::Disconnect(int slot) {
    if (!ValidSlot(slot)) {
        return;
    }
    std::scoped_lock lock{mutex};
    const int player = pads[slot].config.player;
    pads[slot] = {};
    if (ValidPlayer(player)) {
        Recompute(player);
    }
}

bool Router::IsConnected(int slot) const {
    std::scoped_lock lock{mutex};
    return ValidSlot(slot) && pads[slot].used;
}

int Router::Kind(int slot) const {
    std::scoped_lock lock{mutex};
    return ValidSlot(slot) && pads[slot].used ? pads[slot].kind : -1;
}

void Router::SetPlayer(int slot, int player) {
    if (!ValidSlot(slot)) {
        return;
    }
    std::scoped_lock lock{mutex};
    PadConfig config = pads[slot].config;
    config.player = ValidPlayer(player) ? player : -1;
    SetConfig(slot, config);
}

bool Router::ElementHeld(const Pad& pad, int element) const {
    if (pad.elements[element]) {
        return true;
    }
    if (element == LICHII_PAD_L2) {
        return pad.axes[LICHII_PAD_AXIS_L2] >= pad.config.trigger_threshold;
    }
    if (element == LICHII_PAD_R2) {
        return pad.axes[LICHII_PAD_AXIS_R2] >= pad.config.trigger_threshold;
    }
    return false;
}

void Router::UpdateActions(Pad& pad, int element, bool was_held) {
    const bool held = ElementHeld(pad, element);
    const int target = pad.config.map[element];
    if (held && !was_held && target >= FirstAction && target <= LastAction) {
        actions.fetch_or(1u << (target - FirstAction));
    }
    if (!menu_chord) {
        pad.chord_active = false;
        return;
    }
    const bool menu = ElementHeld(pad, LICHII_PAD_MENU);
    const bool options = ElementHeld(pad, LICHII_PAD_OPTIONS);
    if (menu && options && !pad.chord_active) {
        pad.chord_active = true;
        actions.fetch_or(1u << (LICHII_ACTION_MENU - FirstAction));
    } else if (!menu && !options) {
        pad.chord_active = false;
    }
}

void Router::Element(int slot, int element, bool pressed) {
    if (!ValidSlot(slot) || element < 0 || element >= ElementCount) {
        return;
    }
    std::scoped_lock lock{mutex};
    Pad& pad = pads[slot];
    if (!pad.used) {
        return;
    }
    const bool was_held = ElementHeld(pad, element);
    pad.elements[element] = pressed;
    UpdateActions(pad, element, was_held);
    if (ValidPlayer(pad.config.player)) {
        Recompute(pad.config.player);
    }
}

void Router::Axis(int slot, int axis, float value) {
    if (!ValidSlot(slot) || axis < 0 || axis >= AxisCount) {
        return;
    }
    std::scoped_lock lock{mutex};
    Pad& pad = pads[slot];
    if (!pad.used) {
        return;
    }
    const bool is_trigger = axis == LICHII_PAD_AXIS_L2 || axis == LICHII_PAD_AXIS_R2;
    const int element = axis == LICHII_PAD_AXIS_L2 ? LICHII_PAD_L2 : LICHII_PAD_R2;
    const bool was_held = is_trigger && ElementHeld(pad, element);
    pad.axes[axis] = std::clamp(value, is_trigger ? 0.0f : -1.0f, 1.0f);
    if (is_trigger) {
        UpdateActions(pad, element, was_held);
    }
    if (ValidPlayer(pad.config.player)) {
        Recompute(pad.config.player);
    }
}

PadConfig Router::Config(int slot) const {
    std::scoped_lock lock{mutex};
    return ValidSlot(slot) && pads[slot].used ? pads[slot].config : PadConfig{};
}

void Router::SetConfig(int slot, const PadConfig& config) {
    if (!ValidSlot(slot)) {
        return;
    }
    std::scoped_lock lock{mutex};
    Pad& pad = pads[slot];
    if (!pad.used) {
        return;
    }
    const int old_player = pad.config.player;
    pad.config = config;
    for (int& target : pad.config.map) {
        if (!IsValidTarget(target)) {
            target = LICHII_ACTION_NONE;
        }
    }
    pad.config.player = ValidPlayer(config.player) ? config.player : -1;
    pad.config.trigger_threshold = std::clamp(config.trigger_threshold, 0.05f, 0.95f);
    for (StickConfig& stick : pad.config.sticks) {
        stick.inner = std::clamp(stick.inner, 0.0f, 0.9f);
        stick.outer = std::clamp(stick.outer, 0.1f, 1.0f);
    }
    if (ValidPlayer(old_player)) {
        Recompute(old_player);
    }
    if (ValidPlayer(pad.config.player) && pad.config.player != old_player) {
        Recompute(pad.config.player);
    }
}

void Router::SetMenuChord(bool enabled) {
    std::scoped_lock lock{mutex};
    menu_chord = enabled;
    for (int player = 0; player < PlayerCount; ++player) {
        Recompute(player);
    }
}

bool Router::MenuChord() const {
    std::scoped_lock lock{mutex};
    return menu_chord;
}

uint32_t Router::TakeActions() {
    return actions.exchange(0);
}

void Router::Resend() {
    std::scoped_lock lock{mutex};
    for (int player = 0; player < PlayerCount; ++player) {
        Recompute(player, true);
    }
}

PlayerState Router::State(int player) const {
    std::scoped_lock lock{mutex};
    return ValidPlayer(player) ? sent[player] : PlayerState{};
}

PlayerState Router::Combine(int player) const {
    PlayerState state = touch[player];
    for (const Pad& pad : pads) {
        if (!pad.used || pad.config.player != player) {
            continue;
        }
        for (int element = 0; element < ElementCount; ++element) {
            const int target = pad.config.map[element];
            if (target < 0 || target >= ButtonCount || !ElementHeld(pad, element)) {
                continue;
            }
            if (pad.chord_active && (element == LICHII_PAD_MENU || element == LICHII_PAD_OPTIONS)) {
                continue;
            }
            state.buttons[target] = true;
        }
        for (int stick = 0; stick < 2; ++stick) {
            const int axis_x = stick == 0 ? LICHII_PAD_AXIS_LX : LICHII_PAD_AXIS_RX;
            const StickConfig& tuning = pad.config.sticks[stick];
            float x = pad.axes[axis_x];
            float y = pad.axes[axis_x + 1];
            ApplyRadialDeadzone(x, y, tuning.inner, tuning.outer);
            if (tuning.invert_y) {
                y = -y;
            }
            const int target = pad.config.swap_sticks ? 1 - stick : stick;
            const std::array<float, 2> value{x, y};
            if (Magnitude(value) > Magnitude(state.sticks[target])) {
                state.sticks[target] = value;
            }
        }
    }
    return state;
}

void Router::Recompute(int player, bool force) {
    const PlayerState next = Combine(player);
    PlayerState& last = sent[player];
    for (int button = 0; button < ButtonCount; ++button) {
        if (force || next.buttons[button] != last.buttons[button]) {
            button_sink(player, button, next.buttons[button]);
        }
    }
    for (int stick = 0; stick < 2; ++stick) {
        if (force || next.sticks[stick] != last.sticks[stick]) {
            stick_sink(player, stick, next.sticks[stick][0], next.sticks[stick][1]);
        }
    }
    last = next;
}

} // namespace LichiiControls
