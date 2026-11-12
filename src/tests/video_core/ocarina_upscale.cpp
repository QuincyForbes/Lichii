// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/settings.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/fsr.h"

TEST_CASE("OoT temporal backend selection is opt-in and vendor neutral", "[video_core][ocarina]") {
    using enum VideoCommon::OcarinaUpscaleBackend;
    using VideoCommon::SelectOcarinaUpscaler;
    CHECK(SelectOcarinaUpscaler(false, false, false, "") == Native);
    CHECK(SelectOcarinaUpscaler(false, false, true, "") == Native);
    CHECK(SelectOcarinaUpscaler(true, false, false, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(true, false, true, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(false, true, false, "") == Native);
    CHECK(SelectOcarinaUpscaler(false, true, true, "") == Dlss);
    CHECK(SelectOcarinaUpscaler(true, true, false, "") == Fsr);
    CHECK(SelectOcarinaUpscaler(true, true, true, "") == Dlss);
    CHECK(SelectOcarinaUpscaler(true, true, true, "native") == Native);
    CHECK(SelectOcarinaUpscaler(false, true, true, "fsr") == Fsr);
    CHECK(SelectOcarinaUpscaler(false, false, false, "fsr") == Fsr);
    CHECK(SelectOcarinaUpscaler(false, false, false, "dlss") == Native);
    CHECK(SelectOcarinaUpscaler(false, false, true, "dlss") == Dlss);
    CHECK(SelectOcarinaUpscaler(false, true, true, "dlss") == Dlss);
    CHECK(SelectOcarinaUpscaler(false, false, false, "garbage") == Native);
}

TEST_CASE("One upscaler setting picks Ocarina's upscaler and the DLSS replacements",
          "[video_core][ocarina]") {
    using Settings::Upscaler;
    auto& upscaler = Settings::values.upscaler;
    const auto previous = upscaler.GetValue();
    upscaler.SetValue(Upscaler::Auto);
    CHECK_FALSE(Settings::OcarinaFsrChosen());
    CHECK_FALSE(Settings::NvidiaDlssChosen());
    CHECK_FALSE(Settings::GameDlssNetworkChosen());
    upscaler.SetValue(Upscaler::Fsr3);
    CHECK(Settings::OcarinaFsrChosen());
    CHECK_FALSE(Settings::NvidiaDlssChosen());
    upscaler.SetValue(Upscaler::Dlss);
    CHECK(Settings::NvidiaDlssChosen());
    CHECK_FALSE(Settings::OcarinaFsrChosen());
    upscaler.SetValue(Upscaler::MetalFx);
    CHECK(Settings::MetalFxTemporalChosen());
    upscaler.SetValue(Upscaler::Game);
    CHECK(Settings::GameDlssNetworkChosen());
    upscaler.SetValue(previous);
    // Settings store the numbers, so the order must not change.
    CHECK(static_cast<u32>(Settings::OotAntiAliasing::Game) == 0);
    CHECK(static_cast<u32>(Settings::OotAntiAliasing::NoSharpening) == 1);
    CHECK(static_cast<u32>(Upscaler::Auto) == 0);
    CHECK(static_cast<u32>(Upscaler::MetalFx) == 4);
}

TEST_CASE("OoT jitter uses AMD's phase count for the upscale ratio", "[video_core][ocarina]") {
    using VideoCommon::OcarinaJitterPhases;
    CHECK(OcarinaJitterPhases(2560, 3840) == 18);
    CHECK(OcarinaJitterPhases(2560, 2560) == 8);
    CHECK(OcarinaJitterPhases(1920, 3840) == 32);
    CHECK(OcarinaJitterPhases(0, 3840) == 8);
    CHECK_FALSE(VideoCommon::OcarinaJitterEnabled(VideoCommon::OcarinaUpscaleBackend::Native));
}

TEST_CASE("OoT texture LOD bias follows the upscaler, the console mode and failure",
          "[video_core][ocarina]") {
    using enum VideoCommon::OcarinaUpscaleBackend;
    using Settings::OcarinaDlssMode;
    using VideoCommon::KeepOcarinaFsrLodBias;
    using VideoCommon::OcarinaDlssLodBias;
    using VideoCommon::OcarinaFsrLodBias;
    CHECK(OcarinaFsrLodBias(Fsr) == -1.58f);
    CHECK(OcarinaFsrLodBias(Dlss) == 0.0f);
    CHECK(OcarinaFsrLodBias(Native) == 0.0f);
    CHECK(OcarinaDlssLodBias(OcarinaDlssMode::Quality) == -0.58f);
    CHECK(OcarinaDlssLodBias(OcarinaDlssMode::Balanced) == -0.75f);
    CHECK(OcarinaDlssLodBias(OcarinaDlssMode::Performance) == -1.0f);
    CHECK(OcarinaDlssLodBias(OcarinaDlssMode::UltraPerformance) == -1.58f);
    CHECK(KeepOcarinaFsrLodBias(true, false, false));
    CHECK_FALSE(KeepOcarinaFsrLodBias(false, false, false));
    CHECK_FALSE(KeepOcarinaFsrLodBias(true, false, true));
    CHECK_FALSE(KeepOcarinaFsrLodBias(true, true, false));
    CHECK_FALSE(KeepOcarinaFsrLodBias(true, true, true));
}

TEST_CASE("FSR temporal sharpness uses the existing stops slider", "[video_core][fsr][ocarina]") {
    CHECK(FSR::RcasSharpnessFromStops(0.0f) == 1.0f);
    CHECK(FSR::RcasSharpnessFromStops(0.81f) == 0.595f);
    CHECK(FSR::RcasSharpnessFromStops(1.0f) == 0.5f);
    CHECK(FSR::RcasSharpnessFromStops(2.0f) == 0.0f);
    CHECK(FSR::RcasSharpnessFromStops(3.0f) == 0.0f);
    CHECK(FSR::RcasSharpnessFromStops(-1.0f) == 1.0f);
}
