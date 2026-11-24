// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <catch2/catch_test_macros.hpp>
#include "common/device_profile.h"
#include "common/game_presets.h"
#include "common/game_settings.h"
#include "common/settings.h"
#include "common/ui_catalog.h"

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
    CHECK(Value(profile, "backend") == "Metal");
    CHECK(Value(profile, "resolution_setup") == "Res1X");
    CHECK(Value(profile, "use_docked_mode") == "Handheld");
    CHECK(Value(profile, "oot_resolution") == "Percent50");
    CHECK(Value(profile, "oot_graphics_preset") == "PhoneLow");
    CHECK(Value(profile, "oot_frame_rate") == "Fps30");
    CHECK(Value(profile, "texture_quality") == "Quarter");
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
    CHECK(profile.jit_cache_mib == 96);
    CHECK_FALSE(profile.jit_cache_reason.empty());
    CHECK(profile.jit_evict_mib == 12);
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
    CHECK(Value(profile, "backend") == "Metal");
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
    CHECK(profile.device_name == "iPad Pro 13-inch (M4) (M4, 16 GB)");
    CHECK(Value(profile, "backend") == "Metal");
    CHECK(Value(profile, "frame_gen") == "On");
    CHECK(Value(profile, "oot_frame_rate") == "Fps60");
    CHECK(Value(profile, "oot_graphics_preset") == "Balanced");
    CHECK(Value(profile, "texture_quality") == "Full");

    device.model = "iPad13,16"; // M1 iPad Air: Apple GPU family 7, 8 GB
    device.chip = "M1";
    device.apple_gpu_family = 7;
    device.memory_mib = 7900;
    profile = Profile::Recommend(device);
    CHECK(profile.tier == Profile::Tier::Mid);
    CHECK(Value(profile, "backend") == "Metal");
    CHECK(Value(profile, "oot_frame_rate") == "Fps60");
}

TEST_CASE("Desktop profile follows the GPU when it is known", "[device_profile]") {
    Profile::Device device;
    device.os = Profile::Os::Windows;
    device.memory_mib = 32768;
    auto profile = Profile::Recommend(device);
    CHECK(profile.tier == Profile::Tier::Desktop);
    CHECK(Value(profile, "backend") == "Vulkan");
    CHECK(Value(profile, "upscaler") == "Auto");
    CHECK(Value(profile, "oot_resolution") == "Full");
    CHECK(Value(profile, "oot_graphics_preset") == "Balanced");
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
    CHECK(Value(Profile::Recommend(device), "backend") == "Vulkan");
    CHECK(Value(Profile::Recommend(device), "upscaler") == "Auto");

    device.os = Profile::Os::Windows;
    device.dlss_capable = false;
    device.gpu_integrated = true;
    device.vram_mib = 2048;
    device.recommended_vulkan_device = 1;
    profile = Profile::Recommend(device);
    CHECK(Value(profile, "texture_quality") == "Half");
    CHECK(Value(profile, "oot_resolution") == "Percent75");
    CHECK(Value(profile, "vulkan_device") == "1");
}

TEST_CASE("macOS Apple Silicon recommends Direct Metal while Windows and Linux recommend Vulkan",
          "[device_profile]") {
    DefaultsScope scope;
#if defined(__APPLE__)
    CHECK(Settings::values.renderer_backend.GetDefault() == Settings::RendererBackend::Metal);
#elif !defined(__sun__) && !defined(__managarm__)
    CHECK(Settings::values.renderer_backend.GetDefault() == Settings::RendererBackend::Vulkan);
#endif

    Profile::Device mac;
    mac.os = Profile::Os::MacOS;
    mac.model = "Mac16,1";
    mac.chip = "M4";
    mac.apple_gpu_family = 9;
    mac.memory_mib = 16384;
    const auto mac_profile = Profile::Recommend(mac);
    CHECK(mac_profile.tier == Profile::Tier::Desktop);
    CHECK(mac_profile.device_name == "M4, 16 GB RAM");
    CHECK(Value(mac_profile, "backend") == "Metal");
    CHECK(Value(mac_profile, "moltenvk_driver") == "Legacy");
    CHECK(Value(mac_profile, "use_reactive_flushing") == "Off");
    CHECK(Value(mac_profile, "oot_graphics_preset") == "Balanced");
    CHECK(Value(mac_profile, "texture_quality") == "Full");
    for (const auto& value : mac_profile.settings) {
        CHECK_FALSE(value.reason.empty());
    }

    mac.chip = "M1";
    mac.apple_gpu_family = 7;
    mac.memory_mib = 8192;
    const auto m1_profile = Profile::Recommend(mac);
    CHECK(Value(m1_profile, "backend") == "Metal");
    CHECK(Value(m1_profile, "texture_quality") == "Half");

    Profile::Apply(mac_profile);
    CHECK(Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Metal);

#if !defined(__sun__) && !defined(__managarm__)
    Profile::Device win;
    win.os = Profile::Os::Windows;
    win.memory_mib = 16384;
    const auto win_profile = Profile::Recommend(win);
    CHECK(Value(win_profile, "backend") == "Vulkan");
    Profile::Apply(win_profile);
    CHECK(Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);

    Profile::Device lin;
    lin.os = Profile::Os::Linux;
    lin.memory_mib = 16384;
    const auto lin_profile = Profile::Recommend(lin);
    CHECK(Value(lin_profile, "backend") == "Vulkan");
    Profile::Apply(lin_profile);
    CHECK(Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan);
#endif
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
            texture = change.from == "Full" && change.to == "Quarter";
        }
    }
    CHECK(texture);
    const auto applied = Profile::Apply(profile);
    CHECK(applied.size() == before.size());
    CHECK(values.texture_quality.GetValue() == Settings::TextureQuality::Quarter);
    CHECK(values.oot_resolution.GetValue() == Settings::OotResolution::Percent50);
    CHECK(values.oot_graphics_preset.GetValue() == Settings::OotGraphicsPreset::PhoneLow);
    CHECK(values.scaling_filter.GetValue() == Settings::ScalingFilter::MetalFxSpatial);
    CHECK(Profile::Differences(profile).empty());

    // A game's own value is kept; the global one underneath changes.
    values.texture_quality.SetValue(Settings::TextureQuality::Full);
    values.texture_quality.SetGlobal(false);
    values.texture_quality.SetValue(Settings::TextureQuality::Half);
    Profile::Apply(profile);
    CHECK(values.texture_quality.GetValue() == Settings::TextureQuality::Half);
    CHECK(values.texture_quality.GetValue(true) == Settings::TextureQuality::Quarter);
    const auto reset = Profile::ResetGameOverrides();
    CHECK_FALSE(reset.empty());
    CHECK(values.texture_quality.UsingGlobal());
}

TEST_CASE("Resetting a game to recommended writes the profile's values for it",
          "[device_profile]") {
    DefaultsScope scope;
    auto& values = Settings::values;
    const auto profile = Profile::Recommend(IPhone17ProMax());
    // The owner's iPhone: a migrated global config at full textures and the game's own
    // resolution, and Ocarina of Time's own overrides.
    values.texture_quality.SetValue(Settings::TextureQuality::Full);
    values.oot_resolution.SetValue(Settings::OotResolution::Game);
    values.gpu_accuracy.SetValue(Settings::GpuAccuracy::Low);
    values.texture_quality.SetGlobal(false);
    values.texture_quality.SetValue(Settings::TextureQuality::Half);
    values.oot_resolution.SetGlobal(false);
    values.oot_resolution.SetValue(Settings::OotResolution::Percent56);
    values.resolution_setup.SetGlobal(false);
    values.resolution_setup.SetValue(Settings::ResolutionSetup::Res3_4X);
    values.frame_rate_limit.SetGlobal(false);
    values.frame_rate_limit.SetValue(30);

    constexpr u64 Ocarina = Settings::OcarinaOfTimeProgramId;
    const auto planned = Profile::ResetGameToRecommended(profile, Ocarina, true);
    CHECK(values.texture_quality.GetValue() == Settings::TextureQuality::Half);
    const auto changes = Profile::ResetGameToRecommended(profile, Ocarina);
    CHECK(planned.size() == changes.size());

    CHECK_FALSE(values.texture_quality.UsingGlobal());
    CHECK(values.texture_quality.GetValue() == Settings::TextureQuality::Quarter);
    CHECK(values.texture_quality.GetValue(true) == Settings::TextureQuality::Full);
    CHECK(values.oot_resolution.GetValue() == Settings::OotResolution::Percent50);
    CHECK(values.oot_frame_rate.GetValue() == Settings::OotFrameRate::Fps30);
    // Replaced by Ocarina's own resolution, and not in the profile: back to global.
    CHECK(values.resolution_setup.UsingGlobal());
    CHECK(values.frame_rate_limit.UsingGlobal());
    // Already the global value: no override needed.
    CHECK(values.gpu_accuracy.UsingGlobal());
    bool texture = false;
    for (const auto& change : changes) {
        if (change.key == "texture_quality") {
            texture = change.from == "Half" && change.to == "Quarter";
        }
    }
    CHECK(texture);
    // Nothing more to do the second time.
    CHECK(Profile::ResetGameToRecommended(profile, Ocarina, true).empty());
    Settings::RestoreGlobalState(true);
}

TEST_CASE("Phones carry engine switches with reasons", "[device_profile]") {
    const auto profile = Profile::Recommend(IPhone17ProMax());
    REQUIRE_FALSE(profile.tuning.empty());
    CHECK(profile.tuning[0].key == "EDEN_NX2_PIPELINE_EVICT_LEFT_MIB");
    CHECK_FALSE(profile.tuning[0].reason.empty());
    Profile::Device desktop;
    desktop.os = Profile::Os::Windows;
    CHECK(Profile::Recommend(desktop).tuning.empty());
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
    values.upscaler.SetValue(Settings::Upscaler::Fsr4);
    CHECK(Settings::EffectiveResolutionSetup() == ResolutionSetup::Res1X);
}

TEST_CASE("Ocarina of Time's dynamic resolution is fixed under a temporal upscaler",
          "[device_profile]") {
    using Settings::OotResolution;
    using Settings::Upscaler;
    // The upscaler takes only a scene 3/4 of the output: 1080p docked, 810p handheld.
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::MetalFx, false) ==
          OotResolution::Percent75);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Dlss, true) ==
          OotResolution::Full);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Fsr3, false) ==
          OotResolution::Percent75);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Fsr4, false) ==
          OotResolution::Percent75);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Fsr4, true) ==
          OotResolution::Full);
#if defined(__APPLE__)
    const auto previous_backend = Settings::values.renderer_backend.GetValue();
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Metal);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Auto, false) ==
          OotResolution::Percent75);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Auto, true) ==
          OotResolution::Full);
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Auto, false) ==
          OotResolution::Game);
    Settings::values.renderer_backend.SetValue(previous_backend);
#else
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Auto, false) ==
          OotResolution::Game);
#endif
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Game, Upscaler::Game, true) ==
          OotResolution::Game);
    // A resolution the player picked stays.
    CHECK(Settings::OcarinaUpscaledResolution(OotResolution::Percent56, Upscaler::MetalFx, false) ==
          OotResolution::Percent56);
}

TEST_CASE("UiCatalog covers every core setting and resolves cross-references", "[ui_catalog]") {
    for (const auto& [key, basic] : Settings::values.linkage.by_key) {
        INFO(key);
        CHECK(UiCatalog::FindSetting(key) != nullptr);
    }
    const auto* renderer = UiCatalog::FindSetting("backend");
    REQUIRE(renderer != nullptr);
    CHECK(std::string_view{renderer->label} == "Graphics API");
    const auto* fsr_filter =
        UiCatalog::FindChoice(Settings::EnumMetadata<Settings::ScalingFilter>::Index(),
                              static_cast<u32>(Settings::ScalingFilter::Fsr3));
    REQUIRE(fsr_filter != nullptr);
    CHECK(std::string_view{fsr_filter->label} == "AMD FSR 1 Soft");
    const auto* fsr4_upscaler =
        UiCatalog::FindChoice(Settings::EnumMetadata<Settings::Upscaler>::Index(),
                              static_cast<u32>(Settings::Upscaler::Fsr4));
    REQUIRE(fsr4_upscaler != nullptr);
    CHECK(std::string_view{fsr4_upscaler->label} == "AMD FSR 4");
    const auto* optical_flow =
        UiCatalog::FindChoice(Settings::EnumMetadata<Settings::FrameGenType>::Index(),
                              static_cast<u32>(Settings::FrameGenType::Fsr3));
    REQUIRE(optical_flow != nullptr);
    CHECK(std::string_view{optical_flow->label} == "Optical Flow");
    const auto* dlss_fg =
        UiCatalog::FindChoice(Settings::EnumMetadata<Settings::FrameGenType>::Index(),
                              static_cast<u32>(Settings::FrameGenType::NvidiaDlssG));
    REQUIRE(dlss_fg != nullptr);
    CHECK(std::string_view{dlss_fg->label} == "NVIDIA DLSS");
    CHECK(static_cast<u32>(Settings::FrameGenType::NvidiaDlssG) == 3);
    for (const auto& [name, preset] :
         Settings::EnumMetadata<Settings::DlssPreset>::Canonicalizations()) {
        INFO(name);
        CHECK(UiCatalog::FindChoice(Settings::EnumMetadata<Settings::DlssPreset>::Index(),
                                    static_cast<u32>(preset)) != nullptr);
    }
    CHECK(static_cast<u32>(Settings::DlssPreset::M) == 4);
    CHECK(static_cast<u32>(Settings::DlssPreset::E) == 5);
    CHECK(static_cast<u32>(Settings::DlssPreset::F) == 6);
    const auto* dlss_quality_setting = UiCatalog::FindSetting("nvidia_dlss_quality");
    REQUIRE(dlss_quality_setting != nullptr);
    CHECK(std::string_view{dlss_quality_setting->label} == "DLSS Quality");
    for (const auto& [name, quality] :
         Settings::EnumMetadata<Settings::DlssQuality>::Canonicalizations()) {
        INFO(name);
        CHECK(UiCatalog::FindChoice(Settings::EnumMetadata<Settings::DlssQuality>::Index(),
                                    static_cast<u32>(quality)) != nullptr);
    }
    CHECK(std::string_view{UiCatalog::GetText("page.experiments")} == "Memory & Experiments");
    const auto* file_guest = UiCatalog::FindExperiment("file_guest");
    REQUIRE(file_guest != nullptr);
    CHECK(std::string_view{file_guest->label} == "Keep Guest Memory in a File");
    const std::string resolved = UiCatalog::Resolve(
        "Use {exp:file_guest} with {setting:backend}",
        [](std::string_view kind, std::string_view ref) -> std::string {
            if (kind == "exp") {
                return UiCatalog::FindExperiment(ref)->label;
            }
            if (kind == "setting") {
                return UiCatalog::FindSetting(ref)->label;
            }
            return {};
        });
    CHECK(resolved == "Use Keep Guest Memory in a File with Graphics API");
}
