// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QCheckBox>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QVBoxLayout>
#include "common/game_presets.h"
#include "common/game_settings.h"
#include "common/settings.h"
#include "core/core.h"
#include "yuzu/configuration/configure_per_game_tweaks.h"
#include "yuzu/configuration/shared_widget.h"
#include "yuzu/recommended_settings_dialog.h"
#include "qt_common/config/catalog_text.h"

namespace {

std::vector<std::string> EffectiveGameTweaks(u64 title_id) {
    std::vector<std::string> keys;
    if (const auto* tweaks = Settings::FindGameTweaks(title_id)) {
        for (const auto& key : *tweaks) {
            // Settings listed in GamesWithSettings() (e.g. mkw_frame_rate) belong to
            // ConfigureGraphics; skip them here so ApplyConfiguration() cannot overwrite them.
            if (Settings::GameOfSetting(key) == nullptr) {
                keys.push_back(key);
            }
        }
    }
    return keys;
}

} // namespace

ConfigurePerGameTweaks::ConfigurePerGameTweaks(u64 title_id, const Core::System& system_,
                                               const ConfigurationShared::Builder& builder,
                                               QWidget* parent)
    : QWidget(parent), system{system_} {
    setAccessibleName(tr("Tweaks"));

    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    outer->addWidget(scroll);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setSpacing(12);
    scroll->setWidget(content);

    if (const auto* preset = Settings::FindGamePreset(title_id)) {
        auto* box = new QGroupBox(UiText::Text("page.recommended"), content);
        auto* box_layout = new QVBoxLayout(box);
        box_layout->setSpacing(8);

        auto* intro = new QLabel(tr("These settings are known to work well with %1. Tick the ones "
                                    "you want and press Apply Selected; they show up in the other "
                                    "tabs and are saved with OK.")
                                     .arg(QString::fromUtf8(preset->name)),
                                 box);
        intro->setWordWrap(true);
        box_layout->addWidget(intro);

        auto* frame = new QFrame(box);
        frame->setFrameShape(QFrame::StyledPanel);
        auto* frame_layout = new QVBoxLayout(frame);
        frame_layout->setContentsMargins(0, 0, 0, 0);
        recommended_list = new RecommendedSettingsList(*preset, frame);
        frame_layout->addWidget(recommended_list);
        box_layout->addWidget(frame);

        recommended_status = new QLabel(box);
        recommended_status->setWordWrap(true);
        box_layout->addWidget(recommended_status);

        auto* buttons = new QHBoxLayout;
        select_all_button = new QPushButton(tr("Select All"), box);
        select_none_button = new QPushButton(tr("Select None"), box);
        apply_button = new QPushButton(tr("Apply Selected"), box);
        buttons->addWidget(select_all_button);
        buttons->addWidget(select_none_button);
        buttons->addStretch(1);
        buttons->addWidget(apply_button);
        box_layout->addLayout(buttons);

        connect(select_all_button, &QPushButton::clicked, this,
                [this] { recommended_list->SetAllChecked(true); });
        connect(select_none_button, &QPushButton::clicked, this,
                [this] { recommended_list->SetAllChecked(false); });
        connect(apply_button, &QPushButton::clicked, this, [this] {
            emit ApplyRecommendedRequested(recommended_list->SelectedKeys());
        });
        connect(recommended_list, &RecommendedSettingsList::SelectionChanged, this,
                &ConfigurePerGameTweaks::UpdateRecommendedState);

        layout->addWidget(box);
        UpdateRecommendedState();
    }

    const auto tweaks = EffectiveGameTweaks(title_id);
    if (!tweaks.empty()) {
        auto* box = new QGroupBox(tr("Game Tweaks"), content);
        auto* box_layout = new QVBoxLayout(box);
        auto* intro = new QLabel(tr("Options that only affect this game."), box);
        intro->setWordWrap(true);
        box_layout->addWidget(intro);

        ConfigurationShared::Widget* hyrule_dynamic_widget = nullptr;
        ConfigurationShared::Widget* hyrule_30fps_widget = nullptr;
        int added_widgets = 0;

        for (const auto& key : tweaks) {
            auto* widget = builder.BuildWidget(Settings::values.linkage.by_key.at(key), apply_funcs);
            if (widget == nullptr) {
                continue;
            }
            if (!widget->Valid()) {
                widget->deleteLater();
                continue;
            }
            if (key == "hyrule_dynamic_fps") {
                hyrule_dynamic_widget = widget;
            } else if (key == "hyrule_30fps") {
                hyrule_30fps_widget = widget;
            }
            box_layout->addWidget(widget);
            ++added_widgets;
        }

        if (hyrule_dynamic_widget != nullptr && hyrule_dynamic_widget->checkbox != nullptr &&
            hyrule_30fps_widget != nullptr && hyrule_30fps_widget->checkbox != nullptr) {
            const auto make_exclusive = [](ConfigurationShared::Widget* source,
                                           ConfigurationShared::Widget* other) {
                connect(source->checkbox, &QCheckBox::toggled, other, [other](bool checked) {
                    if (checked && other->checkbox->isChecked()) {
                        other->checkbox->setChecked(false);
                        if (other->restore_button != nullptr) {
                            other->restore_button->setEnabled(true);
                            other->restore_button->setVisible(true);
                        }
                    }
                });
            };
            make_exclusive(hyrule_dynamic_widget, hyrule_30fps_widget);
            make_exclusive(hyrule_30fps_widget, hyrule_dynamic_widget);
            if (hyrule_30fps_widget->checkbox->isChecked() &&
                hyrule_dynamic_widget->checkbox->isChecked()) {
                hyrule_dynamic_widget->checkbox->setChecked(false);
                if (hyrule_dynamic_widget->restore_button != nullptr) {
                    hyrule_dynamic_widget->restore_button->setEnabled(true);
                    hyrule_dynamic_widget->restore_button->setVisible(true);
                }
            }
        }

        if (added_widgets > 0) {
            layout->addWidget(box);
        } else {
            delete box;
        }
    }

    layout->addStretch(1);

    // Only scroll vertically: widen the tab to fit the setting table instead.
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setMinimumWidth(content->minimumSizeHint().width() +
                            scroll->verticalScrollBar()->sizeHint().width());
}

ConfigurePerGameTweaks::~ConfigurePerGameTweaks() = default;

bool ConfigurePerGameTweaks::HasContent(u64 title_id) {
    return Settings::FindGamePreset(title_id) != nullptr ||
           !EffectiveGameTweaks(title_id).empty();
}

void ConfigurePerGameTweaks::ApplyConfiguration() {
    const bool is_powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(is_powered_on);
    }
}

void ConfigurePerGameTweaks::RefreshRecommendations() {
    if (recommended_list != nullptr) {
        recommended_list->Refresh();
    }
}

void ConfigurePerGameTweaks::UpdateRecommendedState() {
    if (recommended_list == nullptr) {
        return;
    }
    const bool running = system.IsPoweredOn();
    const bool has_changes = recommended_list->HasChanges();
    const bool editable = !running && has_changes;
    recommended_list->setEnabled(!running);
    select_all_button->setEnabled(editable);
    select_none_button->setEnabled(editable);
    apply_button->setEnabled(editable && !recommended_list->SelectedKeys().empty());
    if (running) {
        recommended_status->setText(tr("Stop the game to apply recommended settings."));
    } else if (!has_changes) {
        recommended_status->setText(tr("All recommended settings are in use."));
    }
    recommended_status->setVisible(running || !has_changes);
}
