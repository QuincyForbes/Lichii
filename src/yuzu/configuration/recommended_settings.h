// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "common/device_profile.h"
#include "qt_common/util/vk.h"

class QWidget;

namespace ConfigurationShared {
class Builder;
}

// "Use Recommended Settings for This Device" (Configure) and "Reset to Recommended" (a game's
// Properties): common/device_profile.h with the GPU Configure knows.
namespace RecommendedSettings {

/// The host with the selected Vulkan device.
Settings::DeviceProfile::Device HostDevice(const std::vector<VkDeviceInfo::Record>& records);

/// Lists what the recommended settings change, with the reason for each, and applies them when
/// confirmed. True when applied.
bool ConfirmAndApply(QWidget* parent, const ConfigurationShared::Builder& builder,
                     const std::vector<VkDeviceInfo::Record>& records);

/// Lists what giving the game the recommended settings changes and does it when confirmed (the
/// per-game config must be the one loaded). True when reset.
bool ConfirmAndResetGame(QWidget* parent, const ConfigurationShared::Builder& builder,
                         const std::vector<VkDeviceInfo::Record>& records);

} // namespace RecommendedSettings
