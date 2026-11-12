// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <vector>
#include <QColor>
#include <QString>
#include <QWidget>
#include <qobjectdefs.h>
#include <vulkan/vulkan_core.h>
#include "common/common_types.h"
#include "common/settings_enums.h"
#include "qt_common/config/shared_translation.h"
#include "qt_common/util/vk.h"
#include "yuzu/configuration/configuration_shared.h"

class QPushButton;
class QEvent;
class QObject;
class QCheckBox;
class QComboBox;
class QLabel;
class NvidiaDlssPanel;

namespace Settings {
enum class NvdecEmulation : u32;
enum class RendererBackend : u32;
} // namespace Settings

namespace Core {
class System;
}

namespace Ui {
class ConfigureGraphics;
}

namespace ConfigurationShared {
class Builder;
}

class ConfigureGraphics : public ConfigurationShared::Tab {
    Q_OBJECT

public:
    explicit ConfigureGraphics(
        const Core::System& system_, std::vector<VkDeviceInfo::Record>& records,
        const std::function<void()>& expose_compute_option,
        const std::function<void(Settings::AspectRatio, Settings::ResolutionSetup)>&
            update_aspect_ratio,
        std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group,
        const ConfigurationShared::Builder& builder, QWidget* parent = nullptr);
    ~ConfigureGraphics() override;

    void ApplyConfiguration() override;
    void SetConfiguration() override;

private:
    void changeEvent(QEvent* event) override;
    void RetranslateUI();

    void Setup(const ConfigurationShared::Builder& builder);

    void PopulateVSyncModeSelection(bool use_setting);
    void UpdateVsyncSetting() const;
    void UpdateBackgroundColorButton(QColor color);
    void UpdateAPILayout();
    void UpdateDeviceSelection(int device);

    void RetrieveVulkanDevices();

    /// Greys out the DLSS and frame generation settings the selected renderer, GPU or build can't
    /// use, with the reason in the tooltip.
    void UpdateFeatureAvailability();
    void UpdateLosslessStatus();

    /* Turns a Vulkan present mode into a textual string for a UI
     * (and eventually for a human to read) */
    const QString TranslateVSyncMode(VkPresentModeKHR mode,
                                     Settings::RendererBackend backend) const;

    Settings::RendererBackend GetCurrentGraphicsBackend() const;

    int FindIndex(u32 enumeration, int value) const;

    std::unique_ptr<Ui::ConfigureGraphics> ui;
    QColor bg_color;

    std::vector<std::function<void(bool)>> apply_funcs{};

    std::vector<VkDeviceInfo::Record>& records;
    std::vector<QString> vulkan_devices;
    std::vector<std::vector<VkPresentModeKHR>> device_present_modes;
    std::vector<VkPresentModeKHR>
        vsync_mode_combobox_enum_map{}; //< Keeps track of which present mode corresponds to which
                                        // selection in the combobox
    u32 vulkan_device{};
    const std::function<void()>& expose_compute_option;
    const std::function<void(Settings::AspectRatio, Settings::ResolutionSetup)> update_aspect_ratio;

    const Core::System& system;
    const ConfigurationShared::ComboboxTranslationMap& combobox_translations;

    QPushButton* api_restore_global_button;
    QComboBox* vulkan_device_combobox;
    QComboBox* api_combobox;
    QComboBox* vsync_mode_combobox;
    QPushButton* vsync_restore_global_button;
    QWidget* vulkan_device_widget;
    QWidget* api_widget;
    QComboBox* aspect_ratio_combobox;
    QComboBox* resolution_combobox;

    QWidget* nvidia_dlss_widget{};
    QCheckBox* nvidia_dlss_checkbox{};
    NvidiaDlssPanel* nvidia_dlss_panel{};
    QWidget* nvidia_dlss_preset_widget{};
    QWidget* ocarina_dlss_mode_widget{};
    QWidget* frame_gen_widget{};
    QComboBox* frame_gen_type_combobox{};
    QLabel* lossless_status_label{};
    QWidget* metalfx_temporal_widget{};
    QComboBox* scaling_filter_combobox{};
};
