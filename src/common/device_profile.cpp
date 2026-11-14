// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
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
#include "common/game_settings.h"
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
        Add(p, v.oot_resolution, OotResolution::Percent50,
            "960x540, with the Phone Low graphics preset: a fixed size, as the game's dynamic "
            "resolution reads emulated GPU time and doesn't drop on a busy phone GPU. 56 -> 50 % "
            "cut GPU time 11 % and draws 20 % on the PC phone simulation (oot-presets).");
    }
    Add(p, v.oot_graphics_preset,
        p.tier == Tier::Ultra ? OotGraphicsPreset::Console : OotGraphicsPreset::PhoneLow,
        p.tier == Tier::Ultra
            ? "Console: no iPad run yet."
            : "Phone Low: an iPhone18,2 spent 84-119 ms of GPU time per frame in Kokiri (builds "
              "71, 138) with 152-216 render passes. A small reflection probe, no bloom or depth "
              "of field, one small sun shadow cascade: with 50 % and Quarter, render passes 175 -> "
              "105, draws -31 % and GPU time -24 % against 56 % Console "
              "on the PC phone simulation (oot-presets).");
    Add(p, v.oot_frame_rate, phone ? OotFrameRate::Fps30 : OotFrameRate::Fps60,
        phone ? "30 FPS: half the GPU work and heat per second; the phone was hot at 1-3 fps "
                "(ios-gpu-time) and Auto-Cool caps at 45/30 anyway. The game keeps its speed."
              : "60 FPS: the game's own rate.");
    Add(p, v.texture_quality,
        p.tier == Tier::Ultra ? TextureQuality::Full : TextureQuality::Quarter,
        p.tier == Tier::Ultra
            ? "Full: 16 GB has room for the game's textures."
            : "Quarter, with the Phone Low graphics preset: ~1.5 GB less guest and GPU memory "
              "in Ocarina of Time than Full; an iPhone18,2 was at its memory limit in Kokiri at "
              "Quarter (build 71: 5932 of 6656 MiB).");
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
    Add(p, v.guest_wait_backoff, true,
        "On: a game thread that yields or polls the clock for 2 ms sleeps instead of holding one "
        "of the two performance cores (nxnative measured 78 -> 19 % of a core in Fire Emblem: "
        "Three Houses this way). Ocarina of Time never waits that long on a PC.");
    AddApp(p, "guest_memory_file", true,
           "Guest memory in a file, outside the memory iOS counts: 1.6-1.8 GiB footprint on "
           "builds 56-59; without it iOS closed the game at 6485 of 6656 MiB (build 54). Needs "
           "10 GB free storage.");
    AddApp(p, "auto_cool", true, "Steps the frame rate down while the device is hot.");
    // PC phone sim (Ocarina of Time, Kokiri, x64 code with the phone's compact JIT code): 32 MiB
    // per core flushed 13-18 times per 10 s at 5-13 fps; 48 MiB 0-2 flushes, 86 fps; 64 MiB 0-1;
    // 96 MiB none (fullest 74 MiB). arm64 code is somewhat larger than x64's. The pool is six of
    // these, prepared through the debugger at app start (~0.1 s per 32 MiB) and resident.
    p.jit_cache_mib = p.tier == Tier::Ultra ? 96 : p.tier == Tier::High ? 64
                    : p.tier == Tier::Mid  ? 48 : 32;
    p.jit_cache_reason = fmt::format(
        "{} MiB of JIT code per guest core ({} MiB for six): Ocarina of Time in Kokiri needs "
        "40-75 MiB of code per core and recompiled everything several times a second at 32 MiB.",
        p.jit_cache_mib, p.jit_cache_mib * 6);
    p.jit_evict_mib = p.jit_cache_mib / 8;
    p.jit_evict_reason = fmt::format(
        "A full code cache drops its oldest {} MiB and goes on: build 7707654e51 still filled 64 "
        "MiB per core (1-4 full flushes per 10 s, compiling 65-131 % in Kokiri), and each flush "
        "translated all of the hot code again at once.",
        p.jit_evict_mib);
    p.jit_fault_shared = true;
    p.jit_fault_block_after = 2;
    p.jit_fault_reason =
        "Fastmem faults: 2.7k-71k a second on the phone (16 KiB pages: one GPU-tracked or "
        "unmapped 4 KiB page takes three neighbours along) against 1-38 on a PC, and each "
        "translated its block again and left the old copy in the cache. A site found by one core "
        "applies to all six, and a block that faulted twice translates without fastmem.";
    p.helper_qos = true;
    p.helper_qos_reason =
        "Timing, fence and present threads at user-initiated QoS: 12 user-interactive threads "
        "shared two performance cores and the guest cores ran 15-56 % of the time on them.";
    p.tuning.push_back(
        {"EDEN_NX2_PIPELINE_EVICT_LEFT_MIB", "384", "384 MiB",
         "Pipelines are evicted on memory pressure only with under 384 MiB left: an iPhone "
         "18,2 evicted 985 with 1.3 GiB left and skipped ~60 draws while they were built again; "
         "1,041 evicted earlier gave back at most ~28 MiB of Metal memory."});
    p.tuning.push_back(
        {"EDEN_NX2_TC_SCRATCH_KEEP_MIB", "16", "16 MiB",
         "Texture upload scratch over 16 MiB is freed at the next frame instead of kept for up to "
         "256 frames: two 64 MiB malloc blocks in Kokiri (iPhone18,2 malloc census); a large "
         "texture allocates it again."});
    p.tuning.push_back(
        {"EDEN_NX2_TQ_ARRAYS", "1", "on",
         "Texture quality also reduces array textures: Ocarina of Time's two 90-layer terrain "
         "material arrays were 56 MiB each in Kokiri at Quarter (iPhone18,2 census, build 71); "
         "Half leaves 15 MiB, Quarter 4 MiB each."});
    p.tuning.push_back(
        {"EDEN_NX2_TC_REDRAWN_FRAMES", "8", "8 frames",
         "A render target the game drew in 8 or more frames is freed without reading it back "
         "from the GPU: the game draws it again before it uses it. An iPhone18,2 read 63-78 "
         "images back per 10 s in Kokiri (build 71), each a full wait for the GPU."});
    p.tuning.push_back(
        {"EDEN_NX2_TC_REUPLOAD_FIRST", "1", "on",
         "Under memory pressure textures that upload again from guest memory are freed first; "
         "images that must be read back from the GPU only once the cache is past its critical "
         "size or an allocation failed."});
    p.tuning.push_back(
        {"EDEN_NX2_JIT_PACKED_LISTS", "1", "on",
         "What the JIT keeps for each translated block (links to other blocks, fastmem patch "
         "points) in one packed heap block behind a 16-byte table entry: as maps and vectors it "
         "was four to six small blocks each, 237 MiB of 16-64 byte blocks plus 155 MiB of tables "
         "in Kokiri (iPhone18,2 malloc census, build 71)."});
    p.tuning.push_back(
        {"EDEN_NX2_FENCE_FLUSH_US", "2000", "2 ms",
         "The game's fences are submitted together within 2 ms instead of each in a command "
         "buffer of its own: an iPhone18,2 ran 76-95 command buffers a frame in Kokiri (IPA "
         "138), the title screen 7. PC phone sim: 59 -> 21 a frame, no more flashes in Kokiri "
         "(flash watch). Experiments > Submit at Every Fence turns it off."});
    p.tuning.push_back(
        {"EDEN_NX2_PASS_TIMES", "10", "Every 10 s",
         "One frame in 10 s gets GPU timestamps around each render pass (\"Perf passes:\" log "
         "line): build 71 spent 84-103 ms of GPU time per frame on the phone with the Legacy and "
         "the New driver alike, ~15x a PC's, and the log has to say which passes take it."});
    p.tuning.push_back(
        {"EDEN_NX2_OOT_BILINEAR_UPSCALE", "1", "on",
         "Ocarina of Time's upscale pass is one bilinear tap whenever MetalFX Temporal doesn't "
         "take the frame: the game's own pass 9302bc0d6d1d9bc1 took ~40 % of an iPhone18,2's "
         "GPU time in Kokiri (IPA 150, Perf passes). Experiments > Game's Own Upscale Pass turns "
         "it off."});
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
    if (device.os == Os::IOS) {
        // "iPhone 17 Pro Max (A19 Pro, 12 GB)"
        const std::string name = device.model_name.empty() ? device.model : device.model_name;
        std::string detail = device.chip;
        if (device.memory_mib != 0) {
            detail += fmt::format("{}{} GB", detail.empty() ? "" : ", ",
                                  MarketedGb(device.memory_mib));
        }
        p.device_name = detail.empty() ? name : fmt::format("{} ({})", name, detail);
    } else {
        // "NVIDIA GeForce RTX 5090 (32 GB), 64 GB RAM"
        std::string name = device.gpu_name;
        if (name.empty()) {
            name = !device.chip.empty() ? device.chip
                                        : (device.os == Os::Windows ? "This PC" : "This computer");
        } else if (device.vram_mib != 0 && !device.gpu_integrated) {
            name += fmt::format(" ({} GB)", MarketedGb(device.vram_mib));
        }
        p.device_name = device.memory_mib != 0
                            ? fmt::format("{}, {} GB RAM", name, MarketedGb(device.memory_mib))
                            : name;
    }
    return p;
}

Profile ForHost() {
    return Recommend(DetectHost());
}

std::string ApplyTuning(const Profile& profile) {
    std::string text;
    for (const Value& value : profile.tuning) {
        const char* const set = std::getenv(value.key.c_str());
        if (set == nullptr) {
#ifdef _WIN32
            _putenv_s(value.key.c_str(), value.value.c_str());
#else
            setenv(value.key.c_str(), value.value.c_str(), 0);
#endif
        }
        text += fmt::format("{}{}={}{}", text.empty() ? "" : ", ", value.key,
                            set != nullptr ? set : value.value.c_str(),
                            set != nullptr ? " (from the environment)" : "");
    }
    return text;
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

std::vector<Change> ResetGameToRecommended(const Profile& profile, u64 program_id,
                                           bool dry_run) {
    std::vector<Change> changes;
    for (const auto& [key, setting] : values.linkage.by_key) {
        if (!setting->Switchable()) {
            continue;
        }
        const Value* const recommended = Find(profile, key);
        const bool applies = recommended != nullptr && AppliesToGame(key, program_id);
        const bool was_global = setting->UsingGlobal();
        const std::string before = was_global ? std::string{"global"} : setting->Canonicalize();
        if (!applies || setting->ToStringGlobal() == recommended->value) {
            if (!was_global) {
                changes.push_back({key, before, "global"});
                if (!dry_run) {
                    setting->SetGlobal(true);
                }
            }
            continue;
        }
        if (was_global || setting->ToString() != recommended->value) {
            changes.push_back({key, before, recommended->display});
        }
        if (!dry_run) {
            setting->SetGlobal(false);
            setting->LoadString(recommended->value);
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
