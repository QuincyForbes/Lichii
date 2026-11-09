// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "ios_frontend/lichii_bridge_platform.h"

// Shared between lichii_bridge_platform.cpp (portable policy) and lichii_platform_ios.mm (UIKit).
namespace LichiiPlatform {

/// The per-device defaults for `info` (pure: the table in lichii_bridge_platform.cpp).
LichiiDeviceDefaults ComputeDefaults(const LichiiDeviceInfo& info);

/// Exports the cache budgets of `defaults` to the texture/buffer caches (environment, kept when
/// already set) and the memory low-water mark.
void ApplyMemoryBudgets(const LichiiDeviceDefaults& defaults);

/// Sets the environment of the enabled experiments (before the guest memory is created).
void ApplyExperiments();

/// What to do in this thermal and power state (pure).
LichiiThermalAdvice ComputeThermalAdvice(int thermal_state, int low_power_mode);

} // namespace LichiiPlatform
