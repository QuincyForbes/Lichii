// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Numbers for the app's performance overlay: the same ones the desktop frontend shows (Real,
// Generated and Displayed FPS, frame time, speed, scale, shaders building). Settings are returned
// as the enum values lichii_settings_enum_name / lichii_settings_enum_value describe, so a value a
// later build adds (a new frame generation type) needs no change here.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LichiiPerfSnapshot {
    double real_fps;        ///< frames the game rendered per second
    double generated_fps;   ///< frames frame generation added per second
    double displayed_fps;   ///< real + generated
    double frametime_ms;    ///< emulated frame time, excluding waits
    double speed;           ///< emulation speed, 1.0 = full speed

    bool paused;
    bool frame_gen_enabled;
    bool game_dlss_network; ///< a Switch 2 game's own DLSS network replaces AMD FSR 3.1
    bool nvidia_dlss;       ///< always false where NVIDIA Streamline is absent
    int32_t frame_gen_type; ///< "frame_gen_type" enum value
    uint32_t frame_gen_multiplier;
    int32_t scaling_filter; ///< "scaling_filter" enum value
    int32_t gpu_accuracy;   ///< "gpu_accuracy" enum value in effect
    float resolution_scale; ///< internal resolution factor, 1.0 = native

    uint32_t guest_width;   ///< the game's frame before scaling, 0 until the first frame
    uint32_t guest_height;
    uint32_t render_width;  ///< guest size times the resolution scale
    uint32_t render_height;
    uint32_t output_width;  ///< the screen's pixels
    uint32_t output_height;

    uint64_t texture_cache_bytes;    ///< device memory the texture cache holds
    uint64_t texture_cache_expected; ///< where it starts collecting
    int32_t shaders_building;
} LichiiPerfSnapshot;

/// Main thread. Frame rates since the previous call (calls less than 250 ms apart repeat the last
/// rates), from counters nothing else resets, and relaxed atomics for the rest. False while no
/// game runs.
bool lichii_perf_snapshot(LichiiPerfSnapshot* out);

#ifdef __cplusplus
} // extern "C"
#endif
