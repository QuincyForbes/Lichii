// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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

} // namespace SettingsMigration
