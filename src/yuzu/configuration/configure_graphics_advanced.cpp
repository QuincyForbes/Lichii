// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>
#include <QLabel>
#include <qnamespace.h>
#include "common/settings.h"
#include "core/core.h"
#include "qt_common/config/shared_translation.h"
#include "ui_configure_graphics_advanced.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_graphics_advanced.h"
#include "yuzu/configuration/shared_widget.h"

ConfigureGraphicsAdvanced::ConfigureGraphicsAdvanced(
    const Core::System& system_, std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
    const ConfigurationShared::Builder& builder, QWidget* parent)
    : Tab(group_, parent), ui{std::make_unique<Ui::ConfigureGraphicsAdvanced>()}, system{system_} {

    ui->setupUi(this);

    Setup(builder);

    SetConfiguration();

    checkbox_enable_compute_pipelines->setVisible(false);
}

ConfigureGraphicsAdvanced::~ConfigureGraphicsAdvanced() = default;

void ConfigureGraphicsAdvanced::SetConfiguration() {}

void ConfigureGraphicsAdvanced::Setup(const ConfigurationShared::Builder& builder) {
    auto& normal_layout = *ui->normal_target->layout();

    // A map will sort the data for us
    std::map<u32, QWidget*> normal_hold{};

    for (auto setting :
         Settings::values.linkage.by_category[Settings::Category::RendererAdvanced]) {
        ConfigurationShared::Widget* widget = builder.BuildWidget(setting, apply_funcs);

        if (widget == nullptr) {
            continue;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            continue;
        }

        const auto id = setting->Id();

        normal_hold.emplace(id, widget);

        // Keep track of enable_compute_pipelines so we can display it when needed
        if (setting->Id() == Settings::values.enable_compute_pipelines.Id()) {
            checkbox_enable_compute_pipelines = widget;
        }
    }

    for (const auto& [id, widget] : normal_hold) {
        normal_layout.addWidget(widget);
    }

    const auto held = [&normal_hold](const Settings::BasicSetting& setting) -> QWidget* {
        const auto it = normal_hold.find(setting.Id());
        return it != normal_hold.end() ? it->second : nullptr;
    };
    if (Settings::values.renderer_backend.GetValue() != Settings::RendererBackend::Vulkan) {
        const QString reason = tr("Vulkan only: the renderer is set to another API (Graphics tab).");
        ConfigurationShared::SetUnavailable(held(Settings::values.renderer_force_max_clock), reason);
        ConfigurationShared::SetUnavailable(held(Settings::values.frame_pacing_mode), reason);
    }
    // nx2_gsv.h PaceToVblank only holds the two-vblank interval at a 60 Hz vsync event with the
    // speed limited.
    if (const u32 hz = Settings::values.frame_rate_limit.GetValue(); hz != 60) {
        ConfigurationShared::SetUnavailable(
            held(Settings::values.oot_30fps),
            tr("Has no effect while Frame rate limit (Switch 2) is %1 Hz: set it to 60 Hz in the "
               "Graphics tab first.")
                .arg(hz));
    } else if (!Settings::values.use_speed_limit.GetValue()) {
        ConfigurationShared::SetUnavailable(
            held(Settings::values.oot_30fps),
            tr("Has no effect while the speed limit is off (Configure > General)."));
    }
}

void ConfigureGraphicsAdvanced::ApplyConfiguration() {
    const bool is_powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(is_powered_on);
    }
}

void ConfigureGraphicsAdvanced::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureGraphicsAdvanced::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureGraphicsAdvanced::ExposeComputeOption() {
    checkbox_enable_compute_pipelines->setVisible(true);
}
