// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <catch2/catch_test_macros.hpp>
#include "common/device_profile.h"
#include "common/game_presets.h"
#include "common/game_settings.h"
#include "common/settings.h"

namespace {

namespace Profile = Settings::DeviceProfile;

std::string Value(const Profile::Profile& profile, std::string_view key) {
    const auto* value = Profile::Find(profile, key);
    return value != nullptr ? value->display : std::string{"<none>"};
}

Profile::Device IPhone17ProMax() {
    // The owner's iPhone18,2 as its device log reports it.
    Profile::Device device;
    device.os = Profile::Os::IOS;
    device.model = "iPhone18,2";
    device.chip = "A19";
    device.apple_gpu_family = 10;
    device.memory_mib = 11694;
    device.max_fps = 120;
    return device;
}

// Settings changed by a test go back to their defaults.
struct DefaultsScope {
    ~DefaultsScope() {
        Settings::RestoreGlobalState(false);
        for (const auto& [key, setting] : Settings::values.linkage.by_key) {
            setting->LoadString("");
        }
        Settings::SetCurrentProgramID(0);
    }
};

} // namespace

TEST_CASE("iPhone 17 Pro Max gets the phone profile", "[device_profile]") {
    const auto profile = Profile::Recommend(IPhone17ProMax());
    CHECK(profile.tier == Profile::Tier::High);
    CHECK(profile.device_name == "iPhone 17 Pro Max (A19 Pro, 12 GB)");
    CHECK(Value(profile, "resolution_setup") == "Res1X");
    CHECK(Value(profile, "use_docked_mode") == "Handheld");
    CHECK(Value(profile, "oot_resolution") == "Percent56");
    CHECK(Value(profile, "oot_frame_rate") == "Fps30");
    CHECK(Value(profile, "texture_quality") == "Half");
    CHECK(Value(profile, "scaling_filter") == "MetalFxSpatial");
    CHECK(Value(profile, "anti_aliasing") == "None");
    CHECK(Value(profile, "upscaler") == "Auto");
    CHECK(Value(profile, "frame_gen") == "Off");
    CHECK(Value(profile, "moltenvk_driver") == "Legacy");
    CHECK(Value(profile, "gpu_accuracy") == "Low");
    CHECK(Value(profile, "use_asynchronous_shaders") == "On");
    CHECK(Value(profile, "guest_memory_file") == "On");
    CHECK(Value(profile, "auto_cool") == "On");
    CHECK(Value(profile, "guest_wait_backoff") == "On");
    CHECK(profile.jit_cache_mib == 64);
    CHECK_FALSE(profile.jit_cache_reason.empty());
    CHECK(profile.jit_evict_mib == 8);
    CHECK(profile.jit_fault_block_after == 2);
    CHECK(profile.helper_qos);
    // Every value has its reason.
    for (const auto* list : {&profile.settings, &profile.app}) {
        for (const auto& value : *list) {
            CHECK_FALSE(value.reason.empty());
        }
    }
}

TEST_CASE("Apple tiers follow memory and GPU family", "[device_profile]") {
    auto device = IPhone17ProMax();
    device.model = "iPhone15,4"; // iPhone 15: A16, 6 GB
    device.apple_gpu_family = 8;
    device.memory_mib = 5600;
    auto profile = Profile::Recommend(device);
    CHECK(profile.tier == Profile::Tier::Low);
    CHECK(Value(profile, "resolution_setup") == "Res3_4X");
    CHECK(Value(profile, "oot_resolution") == "Percent50");
    CHECK(Value(profile, "texture_quality") == "Quarter");
    CHECK_FALSE(profile.note.empty());

    device.model = "iPad16,6";
    device.is_ipad = true;
    device.apple_gpu_family = 9;
    device.memory_mib = 15800;
    profile = Profile::Recommend(device);
    CHECK(profile.tier == Profile::Tier::Ultra);
    CHECK(Value(profile, "frame_gen") == "On");
    CHECK(Value(profile, "oot_frame_rate") == "Fps60");
    CHECK(Value(profile, "texture_quality") == "Full");
}

TEST_CASE("Desktop profile follows the GPU when it is known", "[device_profile]") {
    Profile::Device device;
    device.os = Profile::Os::Windows;
    device.memory_mib = 32768;
    auto profile = Profile::Recommend(device);
    CHECK(profile.tier == Profile::Tier::Desktop);
    CHECK(Value(profile, "upscaler") == "Auto");
    CHECK(Value(profile, "oot_resolution") == "Full");
    CHECK(Value(profile, "texture_quality") == "Full");
    CHECK(Value(profile, "scaling_filter") == "Fsr3");
    CHECK(Value(profile, "moltenvk_driver") == "<none>");
    CHECK(profile.app.empty());

    device.gpu_known = true;
    device.gpu_name = "NVIDIA GeForce RTX 5090";
    device.vram_mib = 32607;
    device.dlss_capable = true;
    profile = Profile::Recommend(device);
    CHECK(Value(profile, "upscaler") == "Dlss");
    device.os = Profile::Os::Linux;
    CHECK(Value(Profile::Recommend(device), "upscaler") == "Auto");

    device.os = Profile::Os::Windows;
    device.dlss_capable = false;
    device.gpu_integrated = true;
    device.vram_mib = 2048;
    profile = Profile::Recommend(device);
    CHECK(Value(profile, "texture_quality") == "Half");
    CHECK(Value(profile, "oot_resolution") == "Percent75");
}

TEST_CASE("Applying the profile sets the global values and reports changes", "[device_profile]") {
    DefaultsScope scope;
    auto& values = Settings::values;
    const auto profile = Profile::Recommend(IPhone17ProMax());
    values.texture_quality.SetValue(Settings::TextureQuality::Full);
    const auto before = Profile::Differences(profile);
    bool texture = false;
    for (const auto& change : before) {
        if (change.key == "texture_quality") {
            texture = change.from == "Full" && change.to == "Half";
        }
    }
    CHECK(texture);
    const auto applied = Profile::Apply(profile);
    CHECK(applied.size() == before.size());
    CHECK(values.texture_quality.GetValue() == Settings::TextureQuality::Half);
    CHECK(values.oot_resolution.GetValue() == Settings::OotResolution::Percent56);
    CHECK(values.scaling_filter.GetValue() == Settings::ScalingFilter::MetalFxSpatial);
    CHECK(Profile::Differences(profile).empty());

    // A game's own value is kept; the global one underneath changes.
    values.texture_quality.SetValue(Settings::TextureQuality::Full);
    values.texture_quality.SetGlobal(false);
    values.texture_quality.SetValue(Settings::TextureQuality::Quarter);
    Profile::Apply(profile);
    CHECK(values.texture_quality.GetValue() == Settings::TextureQuality::Quarter);
    CHECK(values.texture_quality.GetValue(true) == Settings::TextureQuality::Half);
    const auto reset = Profile::ResetGameOverrides();
    CHECK_FALSE(reset.empty());
    CHECK(values.texture_quality.UsingGlobal());
}

TEST_CASE("Game-specific settings show only for their game", "[device_profile]") {
    using Settings::AppliesToGame;
    constexpr u64 Ocarina = Settings::OcarinaOfTimeProgramId;
    constexpr u64 MarioKart = Settings::MarioKartWorldProgramId;
    constexpr u64 Other = 0x040074A01BF12000ULL;
    CHECK(AppliesToGame("oot_resolution", 0));
    CHECK(AppliesToGame("oot_resolution", Ocarina));
    CHECK_FALSE(AppliesToGame("oot_resolution", MarioKart));
    CHECK_FALSE(AppliesToGame("oot_frame_rate", Other));
    CHECK(AppliesToGame("mkw_frame_rate", MarioKart));
    CHECK_FALSE(AppliesToGame("mkw_frame_rate", Ocarina));
    // Ocarina's resolution replaces the resolution scale for it.
    CHECK_FALSE(AppliesToGame("resolution_setup", Ocarina));
    CHECK(AppliesToGame("resolution_setup", MarioKart));
    CHECK(AppliesToGame("texture_quality", Ocarina));
    // Every key the table names is a setting.
    for (const auto& game : Settings::GamesWithSettings()) {
        for (const auto key : game.keys) {
            CHECK(Settings::values.linkage.by_key.contains(std::string{key}));
        }
    }
}

TEST_CASE("Ocarina's resolution sets its scale; other games keep resolution_setup",
          "[device_profile]") {
    DefaultsScope scope;
    using Settings::OotResolution;
    using Settings::ResolutionSetup;
    auto& values = Settings::values;
    values.resolution_setup.SetValue(ResolutionSetup::Res2X);
    values.oot_resolution.SetValue(OotResolution::Full);
    Settings::SetCurrentProgramID(Settings::MarioKartWorldProgramId);
    CHECK(Settings::EffectiveResolutionSetup() == ResolutionSetup::Res2X);
    Settings::SetCurrentProgramID(Settings::OcarinaOfTimeProgramId);
    CHECK(Settings::EffectiveResolutionSetup() == ResolutionSetup::Res1X);
    values.oot_resolution.SetValue(OotResolution::Percent300);
    CHECK(Settings::EffectiveResolutionSetup() == ResolutionSetup::Res3X);
    CHECK(Settings::OcarinaGameResolution(OotResolution::Percent300) == OotResolution::Full);
    // A temporal upscaler needs 1x.
    values.upscaler.SetValue(Settings::Upscaler::Dlss);
    CHECK(Settings::EffectiveResolutionSetup() == ResolutionSetup::Res1X);
}
