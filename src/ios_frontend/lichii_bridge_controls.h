// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Controls bridge for the iOS app: on-screen and physical controller input per player (mapping,
// deadzones, triggers), controller motion, rumble polling, frontend actions bound to controller
// buttons and process CPU/memory figures for the performance HUD. Pause, screenshots, frame stats
// and settings are in lichii_bridge.h. Input sent before a game runs is kept and applied when it
// starts.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Mapping targets: 0..LICHII_CONTROLS_BUTTON_COUNT-1 are Switch buttons (LichiiButton values),
/// these are frontend actions.
typedef enum LichiiAction {
    LICHII_ACTION_NONE = -1,
    LICHII_ACTION_MENU = 100,
    LICHII_ACTION_SCREENSHOT = 101,
    LICHII_ACTION_TOGGLE_HUD = 102,
    LICHII_ACTION_TOGGLE_SPEED_LIMIT = 103,
} LichiiAction;

enum {
    LICHII_CONTROLS_BUTTON_COUNT = 20,
    LICHII_CONTROLS_PLAYER_COUNT = 8,
    LICHII_PAD_SLOT_COUNT = 8,
};

/// Physical controller elements by position (GameController: buttonA is SOUTH, buttonB EAST).
typedef enum LichiiPadElement {
    LICHII_PAD_SOUTH = 0,
    LICHII_PAD_EAST,
    LICHII_PAD_WEST,
    LICHII_PAD_NORTH,
    LICHII_PAD_L1,
    LICHII_PAD_R1,
    LICHII_PAD_L2,
    LICHII_PAD_R2,
    LICHII_PAD_L3,
    LICHII_PAD_R3,
    LICHII_PAD_MENU,
    LICHII_PAD_OPTIONS,
    LICHII_PAD_HOME,
    LICHII_PAD_SHARE,
    LICHII_PAD_DPAD_UP,
    LICHII_PAD_DPAD_DOWN,
    LICHII_PAD_DPAD_LEFT,
    LICHII_PAD_DPAD_RIGHT,
    LICHII_PAD_PADDLE1,
    LICHII_PAD_PADDLE2,
    LICHII_PAD_PADDLE3,
    LICHII_PAD_PADDLE4,
    LICHII_PAD_TOUCHPAD,
    LICHII_PAD_ELEMENT_COUNT,
} LichiiPadElement;

typedef enum LichiiPadAxis {
    LICHII_PAD_AXIS_LX = 0,
    LICHII_PAD_AXIS_LY, ///< +1 up
    LICHII_PAD_AXIS_RX,
    LICHII_PAD_AXIS_RY, ///< +1 up
    LICHII_PAD_AXIS_L2, ///< 0..1
    LICHII_PAD_AXIS_R2, ///< 0..1
    LICHII_PAD_AXIS_COUNT,
} LichiiPadAxis;

typedef enum LichiiPadKind {
    LICHII_PAD_KIND_GENERIC = 0,
    LICHII_PAD_KIND_XBOX,
    LICHII_PAD_KIND_DUALSENSE,
    LICHII_PAD_KIND_DUALSHOCK,
    LICHII_PAD_KIND_SWITCH_PRO,
    LICHII_PAD_KIND_JOYCON_PAIR,
    LICHII_PAD_KIND_MFI,
    LICHII_PAD_KIND_KEYBOARD,
} LichiiPadKind;

/* ---------------------------------------------------------------------------- on-screen input */

/// On-screen controller input for `player` (0 = player 1, mirrored to Handheld).
void lichii_controls_touch_button(int32_t player, int32_t button, bool pressed);
/// Stick position, -1..1 on both axes, y +1 up.
void lichii_controls_touch_stick(int32_t player, int32_t stick, float x, float y);
/// Releases every on-screen input (overlay hidden, menu opened, app backgrounded).
void lichii_controls_release_touch(void);
/// Device motion for `player`: gyro in turns per second (rad/s / 2pi), accel in g, Switch axes.
void lichii_controls_motion(int32_t player, uint64_t delta_us, float gyro_x, float gyro_y,
                            float gyro_z, float accel_x, float accel_y, float accel_z);

/* ---------------------------------------------------------------------- physical controllers */

/// Registers a controller; returns its slot or -1 when all are taken. The slot starts with the
/// default mapping (by position) and the first player no other slot uses.
int32_t lichii_pad_connect(int32_t kind);
void lichii_pad_disconnect(int32_t slot);
/// Player 0..7, or -1 to ignore the controller. Several slots may drive one player.
void lichii_pad_set_player(int32_t slot, int32_t player);
int32_t lichii_pad_get_player(int32_t slot);

void lichii_pad_element(int32_t slot, int32_t element, bool pressed);
void lichii_pad_axis(int32_t slot, int32_t axis, float value);
/// Controller motion in the same units as lichii_controls_motion; goes to the slot's player.
void lichii_pad_motion(int32_t slot, uint64_t delta_us, float gyro_x, float gyro_y, float gyro_z,
                       float accel_x, float accel_y, float accel_z);

/// Maps an element to a Switch button, a LichiiAction, or LICHII_ACTION_NONE.
void lichii_pad_set_mapping(int32_t slot, int32_t element, int32_t target);
int32_t lichii_pad_get_mapping(int32_t slot, int32_t element);
/// Default mapping by position (Nintendo layout: SOUTH is B, EAST is A) or by printed label
/// (Xbox/PlayStation: SOUTH is A).
void lichii_pad_reset_mapping(int32_t slot, bool by_label);
/// Radial deadzone: below `inner` reads 0, past `outer` reads full deflection.
void lichii_pad_set_deadzone(int32_t slot, int32_t stick, float inner, float outer);
void lichii_pad_get_deadzone(int32_t slot, int32_t stick, float* inner, float* outer);
/// Analog trigger press point (0.05..0.95).
void lichii_pad_set_trigger_threshold(int32_t slot, float threshold);
float lichii_pad_get_trigger_threshold(int32_t slot);
void lichii_pad_set_invert_y(int32_t slot, int32_t stick, bool invert);
bool lichii_pad_get_invert_y(int32_t slot, int32_t stick);
void lichii_pad_set_swap_sticks(int32_t slot, bool swap);
bool lichii_pad_get_swap_sticks(int32_t slot);
/// Holding + and - (MENU and OPTIONS) together opens the in-game menu (default on).
void lichii_pad_set_menu_chord(bool enabled);
bool lichii_pad_get_menu_chord(void);

/// Frontend actions requested since the last call: bit (action - LICHII_ACTION_MENU) is set.
uint32_t lichii_controls_take_actions(void);

typedef struct LichiiRumble {
    float left_low;
    float left_high;
    float right_low;
    float right_high;
} LichiiRumble;

/// Rumble amplitudes (0..1) the game last asked for on `player` (Handheld included for player
/// 0). False when no game runs.
bool lichii_controls_rumble(int32_t player, LichiiRumble* out);

/* ------------------------------------------------------------------------------ process stats */

typedef struct LichiiSystemStats {
    double cpu_percent;        ///< process CPU time over walltime since the last call, 100 per core
    uint64_t memory_footprint; ///< bytes (phys_footprint), 0 when unknown
    uint64_t memory_available; ///< bytes the process may still allocate, 0 when unknown
    uint32_t cpu_cores;
    uint32_t threads;
} LichiiSystemStats;

/// Any thread. The first call reports cpu_percent -1.
bool lichii_controls_system_stats(LichiiSystemStats* out);

#ifdef __cplusplus
} // extern "C"
#endif
