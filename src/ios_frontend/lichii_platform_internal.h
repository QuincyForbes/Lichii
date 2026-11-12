// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <string_view>

#include "common/device_profile.h"
#include "ios_frontend/lichii_bridge_platform.h"

// Shared between lichii_bridge_platform.cpp (portable policy) and lichii_platform_ios.mm (UIKit).
namespace LichiiPlatform {

/// `info` as the recommended settings see a device (lichii_platform_init hands it to
/// Settings::DeviceProfile::SetHostDetails).
Settings::DeviceProfile::Device DeviceOf(const LichiiDeviceInfo& info);

/// The per-device defaults for `info` (pure: the recommended settings for that device).
LichiiDeviceDefaults ComputeDefaults(const LichiiDeviceInfo& info);

/// The value `profile` recommends for a settings key, as LoadString takes it ("" if none).
std::string Recommended(const Settings::DeviceProfile::Profile& profile, std::string_view key);

/// Exports the cache budgets of `defaults` to the texture/buffer caches and the .nup loader
/// (environment, kept when already set) and the memory low-water mark; removes the temporary
/// files the loader left when iOS closed a game.
void ApplyMemoryBudgets(const LichiiDeviceDefaults& defaults);

/// Sets the environment of the enabled experiments (before the guest memory is created).
void ApplyExperiments();

/// What to do in this thermal and power state (pure).
LichiiThermalAdvice ComputeThermalAdvice(int thermal_state, int low_power_mode);

} // namespace LichiiPlatform
