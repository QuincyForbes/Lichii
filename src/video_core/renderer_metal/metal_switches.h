// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace Metal {

/// EDEN_NX2_METAL_VULKAN_SWITCHES (on unless 0): the Metal renderer reads the Vulkan renderer's
/// EDEN_NX2_* switches it has a counterpart for and does what they do there (docs/nx2/
/// metal-coverage.md, section Q); 0 ignores them, as before. Their defaults change nothing.
[[nodiscard]] inline bool VulkanSwitchesEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_VULKAN_SWITCHES");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// A Vulkan switch read under VulkanSwitchesEnabled: its value, or null when unset or ignored.
[[nodiscard]] inline const char* VulkanSwitch(const char* name) {
    return VulkanSwitchesEnabled() ? std::getenv(name) : nullptr;
}

} // namespace Metal
