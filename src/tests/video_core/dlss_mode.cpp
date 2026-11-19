// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <optional>
#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"
#include "video_core/dlss_mode.h"

namespace Mode = VideoCommon::DlssMode;
using Preset = Settings::DlssPreset;
using Quality = Settings::DlssQuality;

TEST_CASE("DLSS quality and preset settings round-trip and support per-game overrides",
          "[video_core][dlss_mode]") {
    REQUIRE(Settings::values.nvidia_dlss_preset.GetDefault() == Preset::K);
    REQUIRE(Settings::values.nvidia_dlss_quality.GetDefault() == Quality::Auto);

    Settings::Linkage linkage;
    Settings::SwitchableSetting<Preset> preset{linkage, Preset::K, "nvidia_dlss_preset",
                                               Settings::Category::Renderer};
    preset.LoadString("");
    REQUIRE(preset.GetValue() == Preset::K);
    for (const auto& [name, value] : Settings::EnumMetadata<Preset>::Canonicalizations()) {
        INFO(name);
        preset.SetValue(value);
        const auto saved = preset.ToString();
        preset.LoadString("");
        preset.LoadString(saved);
        REQUIRE(preset.GetValue() == value);
    }
    preset.SetValue(Preset::M);
    preset.SetGlobal(false);
    preset.SetValue(Preset::E);
    REQUIRE(preset.GetValue() == Preset::E);
    REQUIRE(preset.GetValue(true) == Preset::M);
    preset.SetGlobal(true);
    REQUIRE(preset.GetValue() == Preset::M);

    Settings::SwitchableSetting<Quality> quality{linkage, Quality::Auto, "nvidia_dlss_quality",
                                                 Settings::Category::Renderer};
    quality.LoadString("");
    REQUIRE(quality.GetValue() == Quality::Auto);
    for (const auto& [name, value] : Settings::EnumMetadata<Quality>::Canonicalizations()) {
        INFO(name);
        quality.SetValue(value);
        const auto saved = quality.ToString();
        quality.LoadString("");
        quality.LoadString(saved);
        REQUIRE(quality.GetValue() == value);
    }
    quality.SetValue(Quality::Quality);
    quality.SetGlobal(false);
    quality.SetValue(Quality::Dlaa);
    REQUIRE(quality.GetValue() == Quality::Dlaa);
    REQUIRE(quality.GetValue(true) == Quality::Quality);
    quality.SetGlobal(true);
    REQUIRE(quality.GetValue() == Quality::Quality);
}

TEST_CASE("DLSS validates both dimensions, inclusive NGX size bounds and DLAA native sizing",
          "[video_core][dlss_mode]") {
    constexpr Mode::Extent output{3840, 2160};
    constexpr Mode::Range range{{1280, 720}, {2560, 1440}};
    REQUIRE(Mode::Fits(range.min, output, Quality::Quality, range));
    REQUIRE(Mode::Fits(range.max, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({1279, 720}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({1280, 719}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({2561, 1440}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({1920, 1441}, output, Quality::Quality, range));
    REQUIRE_FALSE(Mode::Fits({}, output, Quality::Quality, {}));
    REQUIRE_FALSE(Mode::Fits({1920, 1080}, {0, 2160}, Quality::Quality, {{1, 1}, {3840, 2160}}));
    REQUIRE_FALSE(Mode::Fits({1920, 1080}, {3840, 0}, Quality::Quality, {{1, 1}, {3840, 2160}}));

    constexpr Mode::Range dlaa_range{{1920, 1080}, {3840, 2160}};
    REQUIRE_FALSE(Mode::Fits({2560, 1440}, output, Quality::Dlaa, dlaa_range));
    REQUIRE(Mode::Fits(output, output, Quality::Dlaa, dlaa_range));
}

TEST_CASE("DLSS selects each explicit quality mode and maps render ratios in Auto mode",
          "[video_core][dlss_mode]") {
    constexpr Mode::Extent output{3840, 2160};
    const auto full_ngx = [](Quality mode) -> std::optional<Mode::Range> {
        switch (mode) {
        case Quality::Dlaa:
            return Mode::Range{{3840, 2160}, {3840, 2160}};
        case Quality::Quality:
            return Mode::Range{{2400, 1350}, {2700, 1520}};
        case Quality::Balanced:
            return Mode::Range{{2150, 1200}, {2350, 1320}};
        case Quality::Performance:
            return Mode::Range{{1800, 1000}, {2000, 1200}};
        case Quality::UltraPerformance:
            return Mode::Range{{1200, 680}, {1360, 760}};
        case Quality::Auto:
            break;
        }
        return std::nullopt;
    };

    CHECK(Mode::Preferred({3840, 2160}, output) == Quality::Dlaa);
    CHECK(Mode::Preferred({2560, 1440}, output) == Quality::Quality);
    CHECK(Mode::Preferred({2227, 1253}, output) == Quality::Balanced);
    CHECK(Mode::Preferred({1920, 1080}, output) == Quality::Performance);
    CHECK(Mode::Preferred({1280, 720}, output) == Quality::UltraPerformance);
    CHECK(Mode::Preferred({1280, 720}, {0, 0}) == Quality::UltraPerformance);

    struct Case {
        Quality expected;
        Mode::Extent render;
    };
    for (const Case test : {
             Case{Quality::Dlaa, {3840, 2160}},
             Case{Quality::Quality, {2560, 1440}},
             Case{Quality::Balanced, {2227, 1253}},
             Case{Quality::Performance, {1920, 1080}},
             Case{Quality::UltraPerformance, {1280, 720}},
         }) {
        const auto explicit_choice = Mode::Select(test.expected, test.render, output, full_ngx);
        REQUIRE(explicit_choice.mode == test.expected);
        REQUIRE_FALSE(explicit_choice.fallback);
        REQUIRE(explicit_choice.requested_range.has_value());

        const auto auto_choice = Mode::Select(Quality::Auto, test.render, output, full_ngx);
        REQUIRE(auto_choice.mode == test.expected);
        REQUIRE_FALSE(auto_choice.fallback);
        REQUIRE_FALSE(auto_choice.requested_range.has_value());
    }
}

TEST_CASE("DLSS requested modes fall back to a validated automatic mode",
          "[video_core][dlss_mode]") {
    const auto query = [](Quality mode) -> std::optional<Mode::Range> {
        if (mode == Quality::Quality) {
            return Mode::Range{{2400, 1350}, {2700, 1520}};
        }
        if (mode == Quality::Performance) {
            return Mode::Range{{1800, 1000}, {2000, 1200}};
        }
        if (mode == Quality::Dlaa) {
            return Mode::Range{{1920, 1080}, {3840, 2160}};
        }
        return std::nullopt;
    };

    const auto valid = Mode::Select(Quality::Quality, {2560, 1440}, {3840, 2160}, query);
    REQUIRE(valid.mode == Quality::Quality);
    REQUIRE_FALSE(valid.fallback);

    const auto fallback = Mode::Select(Quality::Quality, {1920, 1080}, {3840, 2160}, query);
    REQUIRE(fallback.mode == Quality::Performance);
    REQUIRE(fallback.fallback);
    REQUIRE(fallback.requested_range.has_value());
    REQUIRE(fallback.requested_range->min.width == 2400);

    const auto dlaa_mismatch = Mode::Select(Quality::Dlaa, {2560, 1440}, {3840, 2160}, query);
    REQUIRE(dlaa_mismatch.mode == Quality::Quality);
    REQUIRE(dlaa_mismatch.fallback);

    const auto unavailable = Mode::Select(Quality::Balanced, {1920, 1080}, {3840, 2160}, query);
    REQUIRE(unavailable.mode == Quality::Performance);
    REQUIRE(unavailable.fallback);
    REQUIRE_FALSE(unavailable.requested_range.has_value());

    const auto explicit_none =
        Mode::Select(Quality::UltraPerformance, {960, 540}, {3840, 2160}, query);
    REQUIRE_FALSE(explicit_none.mode.has_value());
    REQUIRE(explicit_none.fallback);

    const auto auto_none = Mode::Select(Quality::Auto, {2560, 1200}, {3840, 2160}, query);
    REQUIRE_FALSE(auto_none.mode.has_value());
    REQUIRE_FALSE(auto_none.fallback);
}

TEST_CASE("Automatic DLSS checks other modes when its ratio heuristic is invalid",
          "[video_core][dlss_mode]") {
    const auto result = Mode::Select(Quality::Auto, {2560, 1440}, {3840, 2160},
                                     [](Quality mode) -> std::optional<Mode::Range> {
                                         if (mode == Quality::Balanced) {
                                             return Mode::Range{{2000, 1000}, {3000, 1500}};
                                         }
                                         return std::nullopt;
                                     });
    REQUIRE(result.mode == Quality::Balanced);
    REQUIRE_FALSE(result.fallback);
}
