// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"
#include "frontend_common/config.h"
#include "frontend_common/settings_migration.h"

namespace {

using Settings::GamePatchMode;
using Settings::MkwFrameRate;
using Settings::OotFrameRate;
using Settings::OotResolution;

class MigrationConfig final : public Config {
public:
    explicit MigrationConfig(ConfigType type = ConfigType::GlobalConfig,
                             std::string path = "qt-config.ini")
        : Config(type) {
        config_loc = std::move(path);
    }

    void Load(const std::string& data) {
        config = std::make_unique<CSimpleIniA>();
        REQUIRE(config->LoadData(data) >= 0);
        if (global) {
            ReadDisabledAddOnValues();
        }
        ReadRendererValues();
    }

private:
    void ReloadAllValues() override {}
    void SaveAllValues() override {}
    void ReadHidbusValues() override {}
    void ReadDebugControlValues() override {}
    void ReadPathValues() override {}
    void ReadShortcutValues() override {}
    void ReadUIValues() override {}
    void ReadUIGamelistValues() override {}
    void ReadUILayoutValues() override {}
    void ReadMultiplayerValues() override {}
    void SaveHidbusValues() override {}
    void SaveDebugControlValues() override {}
    void SavePathValues() override {}
    void SaveShortcutValues() override {}
    void SaveUIValues() override {}
    void SaveUIGamelistValues() override {}
    void SaveUILayoutValues() override {}
    void SaveMultiplayerValues() override {}
    std::vector<Settings::BasicSetting*>& FindRelevantList(Settings::Category category) override {
        return Settings::values.linkage.by_category[category];
    }
};

// Reading a config resets every setting it doesn't set; leave the defaults behind.
struct MigrationScope {
    ~MigrationScope() {
        Settings::RestoreGlobalState(false);
        Settings::values.disabled_addons.clear();
        MigrationConfig{}.Load("[Renderer]\n");
    }
};

constexpr bool DlssBuild =
#ifdef EGLANTINE_STREAMLINE
    true;
#else
    false;
#endif

} // namespace

TEST_CASE("Ocarina frame rate from oot_30fps and the frame rate limit", "[settings_migration]") {
    using SettingsMigration::OotFrameRateFrom;
    CHECK(OotFrameRateFrom(false, 60) == OotFrameRate::Fps60);
    CHECK(OotFrameRateFrom(true, 60) == OotFrameRate::Fps30);
    // The 30 FPS mode did nothing at another limit: the game ran at the limit.
    CHECK(OotFrameRateFrom(true, 120) == OotFrameRate::Unlocked);
    CHECK(OotFrameRateFrom(false, 120) == OotFrameRate::Unlocked);
    CHECK(OotFrameRateFrom(false, 30) == OotFrameRate::Unlocked);
}

TEST_CASE("Ocarina resolution from the render scale, Max Resolution and DLSS mode",
          "[settings_migration]") {
    using SettingsMigration::OldOcarinaResolution;
    using SettingsMigration::OotResolutionFrom;
    OldOcarinaResolution old;
    CHECK(OotResolutionFrom(old) == OotResolution::Full);
    old.max_resolution_default = false;
    CHECK(OotResolutionFrom(old) == OotResolution::Game);
    old.patch_resolution = GamePatchMode::On;
    CHECK(OotResolutionFrom(old) == OotResolution::Full);
    old.max_resolution_choice = false;
    CHECK(OotResolutionFrom(old) == OotResolution::Game);

    // A fixed render scale won over Max Resolution either way.
    for (const auto [scale, expected] : {std::pair{1, OotResolution::Percent75},
                                         std::pair{2, OotResolution::Percent56},
                                         std::pair{3, OotResolution::Percent50}}) {
        old.render_scale = scale;
        CHECK(OotResolutionFrom(old) == expected);
    }

    // A DLSS mode's scale won over the render scale, with Max Resolution on.
    old = {};
    old.render_scale = 2;
    old.dlss_build = true;
    old.nvidia_dlss = true;
    for (const auto [mode, expected] : {std::pair{1, OotResolution::Percent87},
                                        std::pair{2, OotResolution::Percent75},
                                        std::pair{3, OotResolution::Percent50}}) {
        old.dlss_mode = mode;
        CHECK(OotResolutionFrom(old) == expected);
    }
    old.dlss_mode = 0;
    CHECK(OotResolutionFrom(old) == OotResolution::Percent56);
    old.dlss_mode = 1;
    old.max_resolution_choice = false;
    CHECK(OotResolutionFrom(old) == OotResolution::Game);
    old.max_resolution_choice.reset();
    // Not with FSR 3.1, DLSS off, or a build without DLSS.
    old.fsr3 = true;
    CHECK(OotResolutionFrom(old) == OotResolution::Percent56);
    old.fsr3 = false;
    old.nvidia_dlss = false;
    CHECK(OotResolutionFrom(old) == OotResolution::Percent56);
    old.nvidia_dlss = true;
    old.dlss_build = false;
    CHECK(OotResolutionFrom(old) == OotResolution::Percent56);
    // The DLSS scales followed only the global switch: off there, the render scale applies.
    old.dlss_build = true;
    old.patch_resolution = GamePatchMode::Off;
    CHECK(OotResolutionFrom(old) == OotResolution::Percent56);
}

TEST_CASE("Mario Kart World frame rate from its two switches", "[settings_migration]") {
    using SettingsMigration::MkwFrameRateFrom;
    CHECK(MkwFrameRateFrom(false, false) == MkwFrameRate::Game);
    CHECK(MkwFrameRateFrom(true, false) == MkwFrameRate::Fps30);
    CHECK(MkwFrameRateFrom(false, true) == MkwFrameRate::Dynamic);
    CHECK(MkwFrameRateFrom(true, true) == MkwFrameRate::Dynamic);
}

TEST_CASE("Config of an earlier build loads into the new settings", "[settings_migration]") {
    MigrationScope scope;

    SECTION("The owner's config: 120 Hz limit, everything else default") {
        MigrationConfig{}.Load("[Renderer]\n"
                               "oot_30fps\\default=true\noot_30fps=false\n"
                               "nx2_frame_rate_limit\\default=false\nnx2_frame_rate_limit=120\n"
                               "patch_frame_rate\\default=true\npatch_frame_rate=0\n"
                               "patch_resolution\\default=true\npatch_resolution=0\n"
                               "ocarina_dlss_mode\\default=true\nocarina_dlss_mode=0\n"
                               "oot_render_scale\\default=true\noot_render_scale=0\n"
                               "mkw_30fps\\default=true\nmkw_30fps=false\n"
                               "mkw_dynamic_fps\\default=true\nmkw_dynamic_fps=false\n");
        CHECK(Settings::values.oot_frame_rate.GetValue() == OotFrameRate::Unlocked);
        CHECK(Settings::values.frame_rate_limit.GetValue() == 120);
        CHECK(Settings::values.oot_resolution.GetValue() == OotResolution::Full);
        CHECK(Settings::values.mkw_frame_rate.GetValue() == MkwFrameRate::Game);
    }

    SECTION("30 FPS mode, DLSS Performance, Mario Kart 30 FPS") {
        MigrationConfig{}.Load("[Renderer]\n"
                               "nvidia_dlss\\default=false\nnvidia_dlss=true\n"
                               "oot_30fps\\default=false\noot_30fps=true\n"
                               "ocarina_dlss_mode\\default=false\nocarina_dlss_mode=2\n"
                               "mkw_30fps\\default=false\nmkw_30fps=true\n");
        CHECK(Settings::values.oot_frame_rate.GetValue() == OotFrameRate::Fps30);
        CHECK(Settings::values.oot_resolution.GetValue() ==
              (DlssBuild ? OotResolution::Percent75 : OotResolution::Full));
        CHECK(Settings::values.mkw_frame_rate.GetValue() == MkwFrameRate::Fps30);
    }

    SECTION("Max Resolution turned off in Ocarina's Add-Ons") {
        MigrationConfig{}.Load("[DisabledAddOns]\nsize=1\n1\\title_id\\default=false\n"
                               "1\\title_id=288309953924349952\n1\\disabled\\size=1\n"
                               "1\\disabled\\1\\d\\default=false\n"
                               "1\\disabled\\1\\d=Max Resolution (built-in)\n"
                               "[Renderer]\noot_30fps\\default=true\noot_30fps=false\n");
        CHECK(Settings::values.oot_resolution.GetValue() == OotResolution::Game);
    }

    SECTION("Nothing changes once the new keys are written") {
        MigrationConfig{}.Load("[Renderer]\n"
                               "oot_30fps\\default=false\noot_30fps=true\n"
                               "oot_render_scale\\default=false\noot_render_scale=3\n"
                               "oot_frame_rate\\default=true\noot_frame_rate=1\n"
                               "oot_resolution\\default=true\noot_resolution=1\n");
        CHECK(Settings::values.oot_frame_rate.GetValue() == OotFrameRate::Fps60);
        CHECK(Settings::values.oot_resolution.GetValue() == OotResolution::Full);
    }

    SECTION("A fresh config keeps the defaults") {
        MigrationConfig{}.Load("[Renderer]\n");
        CHECK(Settings::values.oot_frame_rate.GetValue() == OotFrameRate::Fps60);
        CHECK(Settings::values.oot_resolution.GetValue() == OotResolution::Full);
        CHECK(Settings::values.mkw_frame_rate.GetValue() == MkwFrameRate::Game);
    }

    SECTION("Ocarina's own config keeps its own values") {
        MigrationConfig{}.Load("[Renderer]\n");
        MigrationConfig game{Config::ConfigType::PerGameConfig, "custom/0400486024DEC000.ini"};
        game.Load("[Renderer]\n"
                  "oot_30fps\\use_global=false\noot_30fps\\default=false\noot_30fps=true\n"
                  "oot_render_scale\\use_global=false\noot_render_scale\\default=false\n"
                  "oot_render_scale=2\n"
                  "nx2_frame_rate_limit\\use_global=true\n");
        CHECK_FALSE(Settings::values.oot_frame_rate.UsingGlobal());
        CHECK(Settings::values.oot_frame_rate.GetValue() == OotFrameRate::Fps30);
        CHECK(Settings::values.oot_frame_rate.GetValue(true) == OotFrameRate::Fps60);
        CHECK(Settings::values.oot_resolution.GetValue() == OotResolution::Percent56);
        CHECK(Settings::values.oot_resolution.GetValue(true) == OotResolution::Full);
    }

    SECTION("A game that used the global values gets no values of its own") {
        MigrationConfig{}.Load("[Renderer]\n");
        MigrationConfig game{Config::ConfigType::PerGameConfig, "custom/0400486024DEC000.ini"};
        game.Load("[Renderer]\noot_30fps\\use_global=true\noot_render_scale\\use_global=true\n"
                  "nx2_frame_rate_limit\\use_global=true\n");
        CHECK(Settings::values.oot_frame_rate.UsingGlobal());
        CHECK(Settings::values.oot_resolution.UsingGlobal());
    }
}
