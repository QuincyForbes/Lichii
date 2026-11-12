// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#endif

#include "common/device_profile.h"
#include "common/memory_detect.h"
#include "common/settings.h"

namespace Settings::DeviceProfile {

namespace {

struct AppleModel {
    std::string_view model;
    std::string_view name;
    std::string_view chip;
    int gpu_family;
};

// Phones Lichii is run on, newest first (Metal GPU family: 8 A15/A16, 9 A17 Pro/A18, 10 A19).
constexpr std::array<AppleModel, 18> AppleModels{{
    {"iPhone18,1", "iPhone 17 Pro", "A19 Pro", 10},
    {"iPhone18,2", "iPhone 17 Pro Max", "A19 Pro", 10},
    {"iPhone18,3", "iPhone 17", "A19", 10},
    {"iPhone18,4", "iPhone Air", "A19 Pro", 10},
    {"iPhone17,1", "iPhone 16 Pro", "A18 Pro", 9},
    {"iPhone17,2", "iPhone 16 Pro Max", "A18 Pro", 9},
    {"iPhone17,3", "iPhone 16", "A18", 9},
    {"iPhone17,4", "iPhone 16 Plus", "A18", 9},
    {"iPhone17,5", "iPhone 16e", "A18", 9},
    {"iPhone16,1", "iPhone 15 Pro", "A17 Pro", 9},
    {"iPhone16,2", "iPhone 15 Pro Max", "A17 Pro", 9},
    {"iPhone15,4", "iPhone 15", "A16", 8},
    {"iPhone15,5", "iPhone 15 Plus", "A16", 8},
    {"iPhone15,2", "iPhone 14 Pro", "A16", 8},
    {"iPhone15,3", "iPhone 14 Pro Max", "A16", 8},
    {"iPhone14,7", "iPhone 14", "A15", 8},
    {"iPhone14,8", "iPhone 14 Plus", "A15", 8},
    {"iPhone14,2", "iPhone 13 Pro", "A15", 8},
}};

const AppleModel* FindAppleModel(std::string_view model) {
    for (const AppleModel& entry : AppleModels) {
        if (entry.model == model) {
            return &entry;
        }
    }
    return nullptr;
}

std::mutex g_host_mutex;
std::optional<Device> g_host_details;

#ifdef __APPLE__
std::string Sysctl(const char* name) {
    size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }
    std::string text(size, '\0');
    if (sysctlbyname(name, text.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    text.resize(text.find('\0') == std::string::npos ? size : text.find('\0'));
    return text;
}
#endif

template <typename T>
std::string StringOf(T value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else if constexpr (std::is_enum_v<T>) {
        return std::to_string(static_cast<std::underlying_type_t<T>>(value));
    } else {
        return std::to_string(value);
    }
}

template <typename T>
std::string DisplayOf(T value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? "On" : "Off";
    } else if constexpr (std::is_enum_v<T>) {
        return std::string{CanonicalizeEnum(value)};
    } else {
        return std::to_string(value);
    }
}

template <typename T, bool ranged>
void Add(Profile& profile, const Setting<T, ranged>& setting, T value, std::string reason) {
    profile.settings.push_back(
        {setting.GetLabel(), StringOf(value), DisplayOf(value), std::move(reason)});
}

void AddApp(Profile& profile, std::string key, bool on, std::string reason) {
    profile.app.push_back({std::move(key), StringOf(on), DisplayOf(on), std::move(reason)});
}

struct MobileTier {
    Tier tier;
    u64 min_memory_mib; ///< hw.memsize reports a little under the marketed size
    int min_gpu_family;
    ResolutionSetup resolution;
};

// First match wins (the table lichii_bridge_platform.cpp had since the first iOS builds).
constexpr std::array<MobileTier, 4> MobileTiers{{
    {Tier::Ultra, 14000, 8, ResolutionSetup::Res3_2X},
    {Tier::High, 10000, 9, ResolutionSetup::Res1X},
    {Tier::Mid, 7000, 8, ResolutionSetup::Res1X},
    {Tier::Low, 5000, 8, ResolutionSetup::Res3_4X},
}};

u64 MarketedGb(u64 memory_mib) {
    return (memory_mib + 1023) / 1024;
}

void RecommendApple(const Device& device, Profile& p) {
    auto& v = values;
    const MobileTier* row = nullptr;
    for (const MobileTier& candidate : MobileTiers) {
        if (device.memory_mib >= candidate.min_memory_mib &&
            device.apple_gpu_family >= candidate.min_gpu_family) {
            row = &candidate;
            break;
        }
    }
    p.tier = row != nullptr ? row->tier : Tier::Unsupported;
    const bool phone = !device.is_ipad;
    const bool small = p.tier == Tier::Low || p.tier == Tier::Unsupported;
    if (p.tier == Tier::Unsupported) {
        p.note = "Under 6 GB of memory or older than A15: Switch 2 games will likely be closed by "
                 "iOS.";
    } else if (p.tier == Tier::Low) {
        p.note = "6 GB device: larger games may run out of memory.";
    }

    Add(p, v.resolution_setup, row != nullptr ? row->resolution : ResolutionSetup::Res1_2X,
        "Games other than Ocarina of Time: the tier's scale (1x from 8 GB, 0.75x at 6 GB, "
        "1.5x on 16 GB M-series iPads). Higher costs GPU time and memory the phone can't spare.");
    Add(p, v.use_docked_mode, ConsoleMode::Handheld,
        "Handheld: games draw at their handheld size; Docked draws at TV size, more GPU time and "
        "memory.");
    if (p.tier == Tier::Ultra) {
        Add(p, v.oot_resolution, OotResolution::Game,
            "No iPad run yet: the game's own dynamic resolution.");
    } else {
        Add(p, v.oot_resolution, small ? OotResolution::Percent50 : OotResolution::Percent56,
            small ? "960x540: the smallest fixed size, for 6 GB devices."
                  : "1080x608, the size HookLink runs the game at on iPhones. A fixed size: the "
                    "game's dynamic resolution reads emulated GPU time and doesn't drop on a "
                    "busy phone GPU; 56 % cut GPU power 15 % and render targets ~100 MiB on PC "
                    "(oot-mods).");
    }
    Add(p, v.oot_frame_rate, phone ? OotFrameRate::Fps30 : OotFrameRate::Fps60,
        phone ? "30 FPS: half the GPU work and heat per second; the phone was hot at 1-3 fps "
                "(ios-gpu-time) and Auto-Cool caps at 45/30 anyway. The game keeps its speed."
              : "60 FPS: the game's own rate.");
    Add(p, v.texture_quality,
        small ? TextureQuality::Quarter
              : (p.tier == Tier::Ultra ? TextureQuality::Full : TextureQuality::Half),
        p.tier == Tier::Ultra
            ? "Full: 16 GB has room for the game's textures."
            : "Half: ~1.2 GB less guest and GPU memory in Ocarina of Time (Kokiri 4544 -> 3340 "
              "MiB); builds 57-59 reached the game with it on an iPhone18,2.");
    Add(p, v.scaling_filter, ScalingFilter::MetalFxSpatial,
        "MetalFX Spatial: one cheap pass to the screen; FSR 3 costs a phone GPU too much.");
    Add(p, v.anti_aliasing, AntiAliasing::None, "None: games anti-alias themselves.");
    Add(p, v.upscaler, Upscaler::Auto,
        "Game's own: MetalFX temporal hasn't run on a device yet.");
    Add(p, v.frame_gen, p.tier == Tier::Ultra && device.max_fps >= 120,
        p.tier == Tier::Ultra ? "On where the GPU has room and the display shows 120 Hz."
                              : "Off: generated frames cost GPU time a phone needs for the game.");
    Add(p, v.use_reactive_flushing, false,
        "Off: on unified memory the readback is still a fence wait.");
    Add(p, v.moltenvk_driver, MoltenVkDriver::Legacy,
        "Legacy MoltenVK until a device run proves the native mesh shader one.");
    AddApp(p, "guest_memory_file", true,
           "Guest memory in a file, outside the memory iOS counts: 1.6-1.8 GiB footprint on "
           "builds 56-59; without it iOS closed the game at 6485 of 6656 MiB (build 54). Needs "
           "10 GB free storage.");
    AddApp(p, "auto_cool", true, "Steps the frame rate down while the device is hot.");
}

void RecommendDesktop(const Device& device, Profile& p) {
    auto& v = values;
    p.tier = Tier::Desktop;
    // Less than 6 GB of graphics memory, or memory shared with the CPU.
    const bool small_gpu = device.gpu_known && (device.gpu_integrated ||
                                                (device.vram_mib != 0 && device.vram_mib < 6000));
    const bool small_mac = device.os == Os::MacOS && device.memory_mib < 12000;

    Add(p, v.resolution_setup, ResolutionSetup::Res1X,
        "1x for games other than Ocarina of Time: a higher scale costs fps (Frizti tuning).");
    Add(p, v.use_docked_mode, ConsoleMode::Docked, "Docked: the TV resolution, as on a Switch 2.");
    Add(p, v.oot_resolution, small_gpu ? OotResolution::Percent75 : OotResolution::Full,
        small_gpu ? "75 %: integrated or under 6 GB of graphics memory."
                  : "100 %: the game's full size (1440p docked).");
    Add(p, v.texture_quality,
        small_gpu || small_mac ? TextureQuality::Half : TextureQuality::Full,
        small_gpu || small_mac
            ? "Half: ~1.2 GB less memory in Ocarina of Time; Mario Kart World filled 7.3-7.8 GB "
              "of an 8 GB card."
            : "Full: enough memory for the game's textures.");
    Add(p, v.scaling_filter, ScalingFilter::Fsr3,
        "FSR 1 Soft to the window, the first-run choice since the Frizti builds.");
    Add(p, v.anti_aliasing, AntiAliasing::Fxaa, "FXAA: cheap edge smoothing.");
    const bool dlss = device.os == Os::Windows && device.gpu_known && device.dlss_capable;
    Add(p, v.upscaler, dlss ? Upscaler::Dlss : Upscaler::Auto,
        dlss ? "NVIDIA DLSS: an RTX GPU; Ocarina of Time at 100 % is DLSS Quality to 4K. The "
               "Configure page downloads NVIDIA's files."
             : "Game's own: Ocarina of Time's upscaler, AMD FSR 3.1 for DLSS calls.");
    Add(p, v.frame_gen, false, "Off: adds input lag; turn it on for a 120 Hz display.");
    Add(p, v.nvdec_emulation, NvdecEmulation::Gpu, "GPU: Ocarina of Time's cutscene videos.");
    if (device.os == Os::MacOS) {
        Add(p, v.use_reactive_flushing, false,
            "Off: on unified memory the readback is still a fence wait.");
        Add(p, v.moltenvk_driver, MoltenVkDriver::Legacy,
            "Legacy MoltenVK until a run proves the native mesh shader one.");
    } else {
        Add(p, v.use_reactive_flushing, true, "On: readbacks some games need.");
        Add(p, v.astc_recompression, AstcRecompression::Bc3,
            "BC3: GPUs without ASTC keep textures at a quarter of RGBA8's size.");
    }
}

} // namespace

Device DetectHost() {
    {
        std::scoped_lock lock{g_host_mutex};
        if (g_host_details) {
            return *g_host_details;
        }
    }
    Device device;
#if defined(_WIN32)
    device.os = Os::Windows;
#elif defined(__ANDROID__)
    device.os = Os::Android;
#elif defined(__APPLE__) && TARGET_OS_IPHONE
    device.os = Os::IOS;
#elif defined(__APPLE__)
    device.os = Os::MacOS;
#elif defined(__linux__)
    device.os = Os::Linux;
#endif
    device.memory_mib = Common::GetMemInfo().TotalPhysicalMemory >> 20;
#ifdef __APPLE__
    if (device.os == Os::IOS) {
        utsname name{};
        uname(&name);
        device.model = name.machine;
        device.is_ipad = device.model.starts_with("iPad");
    } else {
        device.model = Sysctl("hw.model");
        device.chip = Sysctl("machdep.cpu.brand_string");
        if (device.chip.starts_with("Apple ")) {
            device.chip.erase(0, 6);
        }
    }
    if (const AppleModel* known = FindAppleModel(device.model)) {
        device.model_name = known->name;
        device.chip = known->chip;
        device.apple_gpu_family = known->gpu_family;
    }
#endif
    return device;
}

void SetHostDetails(const Device& device) {
    std::scoped_lock lock{g_host_mutex};
    g_host_details = device;
}

Profile Recommend(const Device& input) {
    // The model table names the phone and its chip ("A19 Pro"; Metal's GPU name says "A19").
    Device device = input;
    if (const AppleModel* known = FindAppleModel(device.model)) {
        device.model_name = known->name;
        device.chip = known->chip;
        if (device.apple_gpu_family == 0) {
            device.apple_gpu_family = known->gpu_family;
        }
    }
    Profile p;
    auto& v = values;
    if (device.os == Os::IOS) {
        RecommendApple(device, p);
    } else {
        RecommendDesktop(device, p);
    }
    Add(p, v.gpu_accuracy, GpuAccuracy::Low,
        "Low: what the Frizti builds tune Ocarina of Time with; High fixes a few games and costs "
        "GPU time.");
    Add(p, v.use_asynchronous_shaders, true,
        "On: a draw whose shader is still building is skipped instead of freezing the game.");
    Add(p, v.use_disk_shader_cache, true, "On: built shaders are kept for the next start.");
    Add(p, v.use_vulkan_driver_pipeline_cache, true, "On: the driver keeps its pipelines too.");
    Add(p, v.async_presentation, true, "On: the frame pacer runs on the present thread.");
    Add(p, v.vsync_mode, VSyncMode::Mailbox, "Mailbox: the host frame pacer times presents.");

    p.tier_name = std::string{TierName(p.tier)};
    std::string name = device.model_name;
    if (name.empty()) {
        name = !device.gpu_name.empty() ? device.gpu_name : device.model;
    }
    std::string detail = device.chip;
    if (device.memory_mib != 0) {
        detail += fmt::format("{}{} GB", detail.empty() ? "" : ", ", MarketedGb(device.memory_mib));
    }
    if (name.empty()) {
        name = device.os == Os::Windows ? "This PC" : "This computer";
    }
    p.device_name = detail.empty() ? name : fmt::format("{} ({})", name, detail);
    return p;
}

Profile ForHost() {
    return Recommend(DetectHost());
}

const Value* Find(const Profile& profile, std::string_view key) {
    for (const auto* list : {&profile.settings, &profile.app}) {
        for (const Value& value : *list) {
            if (value.key == key) {
                return &value;
            }
        }
    }
    return nullptr;
}

std::vector<Change> Differences(const Profile& profile) {
    std::vector<Change> changes;
    for (const Value& value : profile.settings) {
        const auto it = values.linkage.by_key.find(value.key);
        if (it == values.linkage.by_key.end()) {
            continue;
        }
        BasicSetting* const setting = it->second;
        if (setting->ToStringGlobal() != value.value) {
            const bool global = setting->UsingGlobal();
            setting->SetGlobal(true);
            changes.push_back({value.key, setting->Canonicalize(), value.display});
            setting->SetGlobal(global);
        }
    }
    return changes;
}

std::vector<Change> Apply(const Profile& profile) {
    auto changes = Differences(profile);
    for (const Value& value : profile.settings) {
        const auto it = values.linkage.by_key.find(value.key);
        if (it == values.linkage.by_key.end()) {
            continue;
        }
        BasicSetting* const setting = it->second;
        const bool global = setting->UsingGlobal();
        setting->SetGlobal(true);
        setting->LoadString(value.value);
        setting->SetGlobal(global);
    }
    UpdateRescalingInfo();
    return changes;
}

std::vector<Change> ResetGameOverrides() {
    std::vector<Change> changes;
    for (const auto& [key, setting] : values.linkage.by_key) {
        if (setting->Switchable() && !setting->UsingGlobal()) {
            changes.push_back({key, setting->Canonicalize(), "global"});
            setting->SetGlobal(true);
        }
    }
    return changes;
}

std::string FormatChanges(std::span<const Change> changes) {
    std::string text;
    for (const Change& change : changes) {
        text += fmt::format("{}{}: {} -> {}", text.empty() ? "" : ", ", change.key, change.from,
                            change.to);
    }
    return text;
}

std::string_view TierName(Tier tier) {
    switch (tier) {
    case Tier::Unsupported:
        return "unsupported";
    case Tier::Low:
        return "low";
    case Tier::Mid:
        return "mid";
    case Tier::High:
        return "high";
    case Tier::Ultra:
        return "ultra";
    case Tier::Desktop:
        return "desktop";
    }
    return "desktop";
}

} // namespace Settings::DeviceProfile
