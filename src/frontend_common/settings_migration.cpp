// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "common/settings.h"
#include "frontend_common/settings_migration.h"

namespace SettingsMigration {

Settings::OotFrameRate OotFrameRateFrom(bool oot_30fps, u32 frame_rate_limit) {
    if (frame_rate_limit != 60) {
        return Settings::OotFrameRate::Unlocked;
    }
    return oot_30fps ? Settings::OotFrameRate::Fps30 : Settings::OotFrameRate::Fps60;
}

Settings::OotResolution OotResolutionFrom(const OldOcarinaResolution& old) {
    using Settings::GamePatchMode;
    using Settings::OotResolution;
    // Max Resolution: the game's choice, then the global switch, then the platform default. The
    // DLSS render scales were never listed, so only the global switch and default decided them.
    const bool global_on = old.patch_resolution == GamePatchMode::On ||
                           (old.patch_resolution == GamePatchMode::Auto &&
                            old.max_resolution_default);
    const bool max_resolution = old.max_resolution_choice.value_or(global_on);
    if (old.dlss_build && old.nvidia_dlss && !old.fsr3 && old.dlss_mode >= 1 &&
        old.dlss_mode <= 3 && global_on) {
        if (!max_resolution) {
            return OotResolution::Game;
        }
        constexpr OotResolution dlss[]{OotResolution::Percent87, OotResolution::Percent75,
                                       OotResolution::Percent50};
        return dlss[old.dlss_mode - 1];
    }
    switch (old.render_scale) {
    case 1:
        return OotResolution::Percent75;
    case 2:
        return OotResolution::Percent56;
    case 3:
        return OotResolution::Percent50;
    default:
        return max_resolution ? OotResolution::Full : OotResolution::Game;
    }
}

Settings::MkwFrameRate MkwFrameRateFrom(bool mkw_30fps, bool mkw_dynamic_fps) {
    if (mkw_dynamic_fps) {
        return Settings::MkwFrameRate::Dynamic;
    }
    return mkw_30fps ? Settings::MkwFrameRate::Fps30 : Settings::MkwFrameRate::Game;
}

Settings::Upscaler UpscalerFrom(bool oot_fsr3, bool nvidia_dlss, bool metalfx_temporal,
                                bool game_dlss_network) {
    using Settings::Upscaler;
    if (oot_fsr3) {
        return Upscaler::Fsr3;
    }
    if (nvidia_dlss) {
        return Upscaler::Dlss;
    }
    if (metalfx_temporal) {
        return Upscaler::MetalFx;
    }
    return game_dlss_network ? Upscaler::Game : Upscaler::Auto;
}

Settings::OotResolution OotResolutionWithScale(Settings::OotResolution resolution,
                                               Settings::ResolutionSetup scale) {
    using Settings::OotResolution;
    using Settings::ResolutionSetup;
    constexpr std::array<std::pair<OotResolution, int>, 8> percent{{
        {OotResolution::Percent50, 50},
        {OotResolution::Percent56, 56},
        {OotResolution::Percent75, 75},
        {OotResolution::Percent87, 87},
        {OotResolution::Full, 100},
        {OotResolution::Percent150, 150},
        {OotResolution::Percent200, 200},
        {OotResolution::Percent300, 300},
    }};
    const auto own = std::ranges::find(percent, resolution, &std::pair<OotResolution, int>::first);
    if (resolution == OotResolution::Game || own == percent.end() ||
        scale == ResolutionSetup::Res1X) {
        return resolution;
    }
    Settings::ResolutionScalingInfo info;
    Settings::TranslateResolutionInfo(scale, info);
    const float product = static_cast<float>(own->second) * info.up_factor;
    // The nearest share; on a tie the smaller one (less GPU work, as the scale was a choice).
    auto best = percent.front();
    for (const auto& entry : percent) {
        if (std::abs(static_cast<float>(entry.second) - product) <
            std::abs(static_cast<float>(best.second) - product)) {
            best = entry;
        }
    }
    return best.first;
}

} // namespace SettingsMigration
