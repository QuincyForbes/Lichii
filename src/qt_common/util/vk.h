// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "common/common_types.h"
#include "common/device_profile.h"
#include "common/settings_enums.h"
#include "vulkan/vulkan_core.h"

static const std::vector<VkPresentModeKHR> default_present_modes{VK_PRESENT_MODE_IMMEDIATE_KHR,
                                                                 VK_PRESENT_MODE_FIFO_KHR};

// Converts a setting to a present mode (or vice versa)
static inline constexpr VkPresentModeKHR VSyncSettingToMode(Settings::VSyncMode mode) {
    switch (mode) {
    case Settings::VSyncMode::Immediate:
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    case Settings::VSyncMode::Mailbox:
        return VK_PRESENT_MODE_MAILBOX_KHR;
    case Settings::VSyncMode::Fifo:
        return VK_PRESENT_MODE_FIFO_KHR;
    case Settings::VSyncMode::FifoRelaxed:
        return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    default:
        return VK_PRESENT_MODE_FIFO_KHR;
    }
}

static inline constexpr Settings::VSyncMode PresentModeToSetting(VkPresentModeKHR mode) {
    switch (mode) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return Settings::VSyncMode::Immediate;
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return Settings::VSyncMode::Mailbox;
    case VK_PRESENT_MODE_FIFO_KHR:
        return Settings::VSyncMode::Fifo;
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        return Settings::VSyncMode::FifoRelaxed;
    default:
        return Settings::VSyncMode::Fifo;
    }
}

class QWindow;

namespace Settings {
enum class VSyncMode : u32;
}

namespace VkDeviceInfo {
// Short class to record Vulkan driver information for configuration purposes
/// What the configuration UI needs to know about a device's vendor and driver.
struct Capabilities {
    u32 vendor_id{};                    ///< VkPhysicalDeviceProperties::vendorID
    u32 driver_id{};                    ///< VkDriverId, 0 when unknown
    std::vector<std::string> extensions; ///< Device extension names

    [[nodiscard]] bool HasExtension(std::string_view extension) const {
        for (const auto& name : extensions) {
            if (name == extension) {
                return true;
            }
        }
        return false;
    }
};

class Record {
public:
    explicit Record(std::string_view name, const std::vector<VkPresentModeKHR>& vsync_modes,
                    bool has_broken_compute, VkDriverId driver_id, bool dlss_capable,
                    std::string_view driver_version = {}, bool integrated = false,
                    u64 vram_mib = 0, Capabilities capabilities = {},
                    bool fsr4_capable = false);
    ~Record();

    const std::string name;
    const std::vector<VkPresentModeKHR> vsync_support;
    const bool has_broken_compute;
    const VkDriverId driver_id;
    /// NVIDIA's proprietary driver on a GPU with tensor cores (GeForce/Quadro RTX), which DLSS
    /// needs. Streamline makes the final call when a game starts.
    const bool dlss_capable;
    /// AMD Radeon RX 6000/7000/9000 or Ryzen RDNA 3+ GPU on AMD's Windows driver, or Mesa RADV
    /// on Linux with fsr4-vk.
    const bool fsr4_capable;
    /// The driver's own version text (e.g. "581.57"), for support logs.
    const std::string driver_version;
    /// Shares the CPU's memory (VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU).
    const bool integrated;
    /// The largest device-local memory heap.
    const u64 vram_mib;
    const u32 vendor_id;
    const Capabilities capabilities;
};

/// Returns the highest-ranked device index matching `pred` (preferring discrete GPUs, then VRAM),
/// or -1 when none matches.
template <typename Pred>
[[nodiscard]] int FindBestDevice(std::span<const Record> records, Pred&& pred) {
    int best = -1;
    u64 best_score = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (!pred(records[i])) {
            continue;
        }
        const u64 score = Settings::DeviceProfile::ScoreGpuDevice(
            !records[i].integrated, records[i].integrated, records[i].vram_mib);
        if (best < 0 || score > best_score) {
            best = static_cast<int>(i);
            best_score = score;
        }
    }
    return best;
}

[[nodiscard]] inline int FindBestDiscreteDevice(std::span<const Record> records) {
    return FindBestDevice(records, [](const Record& r) { return !r.integrated; });
}

// TODO(crueter): Port some configure_graphics.cpp stuff
void PopulateRecords(std::vector<Record>& records, QWindow* window);
} // namespace VkDeviceInfo
