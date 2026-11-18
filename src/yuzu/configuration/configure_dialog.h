// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <QDialog>
#include "configuration/shared_widget.h"
#include "qt_common/config/shared_translation.h"
#include "qt_common/util/vk.h"
#include "yuzu/configuration/configuration_shared.h"

namespace Core {
class System;
}

class ConfigureApplets;
class ConfigureAudio;
class ConfigureCpu;
class ConfigureDebugTab;
class ConfigureFilesystem;
class ConfigureFrameGeneration;
class ConfigureGeneral;
class ConfigureGraphics;
class ConfigureGraphicsAdvanced;
class ConfigureGraphicsExtensions;
class ConfigureHotkeys;
class ConfigureInput;
class ConfigureProfileManager;
class ConfigureSystem;
class ConfigureNetwork;
class ConfigureGpu;
class ConfigurePerformance;
class ConfigureShaders;
class ConfigureRam;
class ConfigureUi;
class ConfigureWeb;

class QLabel;
class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QScrollArea;
class HotkeyRegistry;

namespace Performance {
class State;
}

namespace InputCommon {
class InputSubsystem;
}

namespace Ui {
class ConfigureDialog;
}

class ConfigureDialog : public QDialog {
    Q_OBJECT

public:
    explicit ConfigureDialog(QWidget* parent, HotkeyRegistry& registry_,
                             InputCommon::InputSubsystem* input_subsystem,
                             std::vector<VkDeviceInfo::Record>& vk_device_records,
                             Core::System& system_, bool enable_web_config = true);
    ~ConfigureDialog() override;

    void ApplyConfiguration();

    /// Selects the Profiles tab of the System category.
    void SelectProfilesTab();

private slots:
    void OnLanguageChanged(const QString& locale);

signals:
    void LanguageChanged(const QString& locale);
    void ExternalContentDirsChanged();

private:
    void changeEvent(QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void RetranslateUI();
    void FitToAvailableScreen();
    QWidget* WrapTab(QWidget* tab);
    /// The Controls page's tabs, built the first time the page is shown (eight player pages).
    QList<QWidget*> ControlsTabs();

    void HandleApplyButtonClicked();

    void SetConfiguration();
    void UpdateVisibleTabs();
    void PopulateSelectionList();

    /// One page of the sidebar, as the iOS app's settings root lists them: a title, an icon and
    /// the tabs it shows one under the other (Controls keeps its tabs).
    struct Page {
        QString title;
        QString icon;
        /// Sidebar group heading shown before this page, if any.
        QString group;
        QString footer;
        std::vector<std::pair<QString, QWidget*>> sections;
        bool controls{false};
        QScrollArea* area{};
        QLabel* title_label{};
        QLabel* footer_label{};
        std::vector<std::pair<std::size_t, QLabel*>> section_labels;
    };
    QScrollArea* PageArea(Page& page);
    void SelectPage(int index);
    /// The settings search, as the iOS app's: every page's settings by name, and the settings this
    /// platform has no use for with the reason.
    void OnSearchChanged(const QString& text);
    void OpenSearchResult(QListWidgetItem* item);

    std::vector<Page> pages;
    int last_page{0};
    QLineEdit* search_field{};
    QListWidget* search_results{};

    std::unique_ptr<Ui::ConfigureDialog> ui;
    HotkeyRegistry& registry;

    Core::System& system;
    std::unique_ptr<ConfigurationShared::Builder> builder;
    std::vector<ConfigurationShared::Tab*> tab_group;
    std::unordered_map<QWidget*, QScrollArea*> tab_scroll_areas;
    bool initial_geometry_clamped{false};

    std::unique_ptr<ConfigureApplets> applets_tab;
    std::unique_ptr<ConfigureAudio> audio_tab;
    std::unique_ptr<ConfigureCpu> cpu_tab;
    std::unique_ptr<ConfigureDebugTab> debug_tab_tab;
    std::unique_ptr<ConfigureFilesystem> filesystem_tab;
    std::unique_ptr<ConfigureGeneral> general_tab;
    std::unique_ptr<ConfigureGraphicsAdvanced> graphics_advanced_tab;
    std::unique_ptr<ConfigureGraphicsExtensions> graphics_extensions_tab;
    std::unique_ptr<ConfigureFrameGeneration> frame_gen_tab;
    std::unique_ptr<ConfigureUi> ui_tab;
    std::unique_ptr<ConfigureGraphics> graphics_tab;
    std::unique_ptr<ConfigureHotkeys> hotkeys_tab;
    std::unique_ptr<ConfigureInput> input_tab;
    std::unique_ptr<ConfigureNetwork> network_tab;
    std::unique_ptr<Performance::State> performance_state;
    std::unique_ptr<ConfigureGpu> performance_gpu_tab;
    std::unique_ptr<ConfigureShaders> performance_shaders_tab;
    std::unique_ptr<ConfigureRam> performance_ram_tab;
    std::unique_ptr<ConfigureProfileManager> profile_tab;
    std::unique_ptr<ConfigureSystem> system_tab;
    std::unique_ptr<ConfigureWeb> web_tab;
    InputCommon::InputSubsystem* input_subsystem;
};
