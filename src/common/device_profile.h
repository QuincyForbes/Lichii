// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

// The recommended settings for the device Lichii runs on: one table for every frontend. A new
// configuration starts from them (Config::SetUpIni), and "Use recommended settings for this
// device" (Qt Configure, iOS Settings) applies them to an existing one. The values and the
// evidence behind each are in each Add() call's reason.
namespace Settings::DeviceProfile {

enum class Os { Windows, Linux, MacOS, IOS, Android, Other };
enum class GpuVendor { Unknown, Nvidia, Amd, Intel, Apple, Qualcomm, Other };
// Apple phones and tablets, by memory first: Switch 2 games get a 12 GB memory layout, and iOS
// gives an app about 55-70 % of the device's memory.
enum class Tier { Unsupported, Low, Mid, High, Ultra, Desktop };

struct Device {
    Os os = Os::Other;
    std::string model;          ///< "iPhone18,2", "Mac14,2"; empty elsewhere
    std::string model_name;     ///< "iPhone 17 Pro Max", when the model table knows it
    std::string chip;           ///< "A19 Pro", "M2"
    int apple_gpu_family = 0;   ///< Highest MTLGPUFamilyAppleN (7 A14/M1 ... 10 A19/M5)
    bool is_ipad = false;
    u64 memory_mib = 0;         ///< Installed memory
    int max_fps = 0;            ///< Display refresh rate, 0 when unknown

    // The GPU, where the frontend knows it (Qt reads the Vulkan device list when Configure opens).
    bool gpu_known = false;
    std::string gpu_name;
    GpuVendor gpu_vendor = GpuVendor::Unknown;
    bool gpu_integrated = false;
    u64 vram_mib = 0;           ///< Device-local memory
    bool dlss_capable = false;  ///< NVIDIA RTX (tensor cores), in a build with NVIDIA Streamline
};

/// What the host tells without a graphics device: OS, memory, Apple model and chip.
Device DetectHost();
/// The iOS platform layer adds what Metal reports (lichii_platform_init); later DetectHost calls
/// return it.
void SetHostDetails(const Device& device);

struct Value {
    std::string key;     ///< Settings key, or an app item ("guest_memory_file", "auto_cool")
    std::string value;   ///< As LoadString takes it ("true", "3")
    std::string display; ///< "On", "Percent56"
    std::string reason;
};

struct Profile {
    Tier tier = Tier::Desktop;
    std::string tier_name;
    std::string device_name; ///< "iPhone 17 Pro Max (A19 Pro, 12 GB)"
    std::string note;        ///< One line for the UI, may be empty
    std::vector<Value> settings;
    /// Frontend-side choices that are not Settings values (iOS: guest memory in a file, Auto-Cool).
    std::vector<Value> app;
    /// iOS: each guest core's JIT code cache in MiB (the debugger-prepared pool holds one per
    /// guest core, sized at app start); 0 = Common::IosJit's default.
    u32 jit_cache_mib = 0;
    std::string jit_cache_reason;
    /// Engine switches (EDEN_NX2_* environment variables) this device runs with; a variable set
    /// in the environment wins. Applied once at app start (ApplyTuning).
    std::vector<Value> tuning;
};

Profile Recommend(const Device& device);
/// Recommend(DetectHost()).
Profile ForHost();
/// Sets the profile's tuning variables that the environment doesn't set; returns them as
/// "NAME=value" for the log.
std::string ApplyTuning(const Profile& profile);

/// A setting's or app item's value in the profile, nullptr when it has none.
const Value* Find(const Profile& profile, std::string_view key);

struct Change {
    std::string key;
    std::string from;
    std::string to;
};

/// The global values the profile would change (canonical names).
std::vector<Change> Differences(const Profile& profile);
/// Sets the global values; not while a game runs (its per-game values are loaded then).
std::vector<Change> Apply(const Profile& profile);
/// Inside a per-game scope: every setting the game overrides follows the global value again.
std::vector<Change> ResetGameOverrides();
/// Inside a per-game scope: the game gets the profile's value of every setting it can override
/// and that applies to it (an override where the global value differs, the global value where
/// it is the same); its other overrides are removed. Clearing the overrides alone left Ocarina
/// of Time at a global config's full textures and game resolution on an iPhone. `dry_run`: only
/// what would change.
std::vector<Change> ResetGameToRecommended(const Profile& profile, u64 program_id,
                                           bool dry_run = false);

std::string FormatChanges(std::span<const Change> changes);
std::string_view TierName(Tier tier);

} // namespace Settings::DeviceProfile
