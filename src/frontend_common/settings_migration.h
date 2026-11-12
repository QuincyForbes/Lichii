// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"
#include "common/settings_enums.h"

// Settings that earlier builds kept as several keys, mapped to the one setting that replaced
// them, so a config keeps doing what it did (Config::MigrateRendererValues).
namespace SettingsMigration {

/// oot_30fps and nx2_frame_rate_limit -> oot_frame_rate. The 30 FPS mode held only while the
/// limit was 60 Hz; at any other limit Ocarina of Time ran at the limit.
Settings::OotFrameRate OotFrameRateFrom(bool oot_30fps, u32 frame_rate_limit);

/// What decided Ocarina of Time's 3D resolution before oot_resolution.
struct OldOcarinaResolution {
    /// oot_render_scale: 0 Game, 1 75%, 2 56%, 3 50%.
    int render_scale = 0;
    /// patch_resolution: the global switch of the resolution built-ins.
    Settings::GamePatchMode patch_resolution = Settings::GamePatchMode::Auto;
    /// The game's Add-Ons choice for "Max Resolution (built-in)", if it made one.
    std::optional<bool> max_resolution_choice;
    /// Max Resolution's platform default (on, except on iOS).
    bool max_resolution_default = true;
    /// ocarina_dlss_mode: 0 Quality, 1 Balanced, 2 Performance, 3 Ultra Performance. Its render
    /// scale applied only with NVIDIA DLSS in the build and on, and FSR 3.1 not chosen.
    int dlss_mode = 0;
    bool dlss_build = false;
    bool nvidia_dlss = false;
    bool fsr3 = false;
};

/// oot_render_scale, patch_resolution, the Max Resolution Add-On and ocarina_dlss_mode ->
/// oot_resolution.
Settings::OotResolution OotResolutionFrom(const OldOcarinaResolution& old);

/// mkw_30fps and mkw_dynamic_fps -> mkw_frame_rate (dynamic took precedence).
Settings::MkwFrameRate MkwFrameRateFrom(bool mkw_30fps, bool mkw_dynamic_fps);

/// oot_anti_aliasing AMD FSR 3.1 (value 2), nvidia_dlss, metalfx_temporal and game_dlss_network ->
/// upscaler. FSR 3.1 won over DLSS and MetalFX in Ocarina of Time, the game this was set for.
Settings::Upscaler UpscalerFrom(bool oot_fsr3, bool nvidia_dlss, bool metalfx_temporal,
                                bool game_dlss_network);

/// Ocarina of Time drew at oot_resolution times resolution_setup; the resolution scale no longer
/// applies to it, so the product becomes the nearest oot_resolution. The game's dynamic
/// resolution stays as it is.
Settings::OotResolution OotResolutionWithScale(Settings::OotResolution resolution,
                                               Settings::ResolutionSetup scale);

} // namespace SettingsMigration
