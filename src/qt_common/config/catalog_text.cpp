// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <QAbstractButton>
#include <QComboBox>
#include <QCoreApplication>
#include <QGroupBox>
#include <QLabel>
#include <QWidget>

#include "common/settings.h"
#include "qt_common/config/catalog_text.h"
#include "qt_common/config/uisettings.h"

namespace UiText {

namespace {

QString Translate(const char* context, const char* text) {
    return text == nullptr ? QString() : QCoreApplication::translate(context, text);
}

const Settings::BasicSetting* FindBasicSetting(std::string_view key) {
    const std::string name{key};
    for (const auto* const linkage : {&Settings::values.linkage, &UISettings::values.linkage}) {
        if (const auto it = linkage->by_key.find(name); it != linkage->by_key.end()) {
            return it->second;
        }
    }
    return nullptr;
}

} // namespace

QString Resolve(const QString& text) {
    if (!text.contains(QLatin1Char('{'))) {
        return text;
    }
    const std::string resolved = UiCatalog::Resolve(
        text.toStdString(), [](std::string_view kind, std::string_view ref) -> std::string {
            QString name;
            if (kind == "setting") {
                name = SettingLabel(ref);
            } else if (kind == "exp") {
                name = ExperimentLabel(ref);
            } else if (kind == "text") {
                name = Text(ref);
            } else if (kind == "choice") {
                const std::size_t dot = ref.find('.');
                name = ChoiceLabel(ref.substr(0, dot), ref.substr(dot + 1));
            }
            return name.toStdString();
        });
    return QString::fromStdString(resolved);
}

QString Text(std::string_view id) {
    return Resolve(Translate(UiCatalog::TextsContext, UiCatalog::GetText(id)));
}

QString SettingLabel(std::string_view key) {
    const auto* setting = UiCatalog::FindSetting(key);
    return setting == nullptr ? QString()
                              : Translate(UiCatalog::SettingsContext, setting->label);
}

QString SettingHelp(std::string_view key) {
    const auto* setting = UiCatalog::FindSetting(key);
    return setting == nullptr ? QString()
                              : Resolve(Translate(UiCatalog::SettingsContext, setting->help));
}

QString SettingReason(std::string_view key) {
    const auto* setting = UiCatalog::FindSetting(key);
    return setting == nullptr ? QString()
                              : Resolve(Translate(UiCatalog::SettingsContext, setting->reason));
}

QString ChoiceLabel(u32 enum_index, u32 value) {
    const auto* choice = UiCatalog::FindChoice(enum_index, value);
    return choice == nullptr ? QString()
                             : Translate(UiCatalog::SettingsContext, choice->label);
}

QString ChoiceLabel(std::string_view key, std::string_view canonical) {
    const auto* setting = FindBasicSetting(key);
    if (setting == nullptr) {
        return QString();
    }
    for (const auto& choice : UiCatalog::ChoicesOf(setting->EnumIndex())) {
        if (choice.canonical == canonical) {
            return Translate(UiCatalog::SettingsContext, choice.label);
        }
    }
    return QString();
}

QString ExperimentLabel(std::string_view id) {
    const auto* experiment = UiCatalog::FindExperiment(id);
    return experiment == nullptr ? QString()
                                 : Translate(UiCatalog::TextsContext, experiment->label);
}

void ApplyCatalogue(QWidget* root) {
    const auto set_text = [](QWidget* widget, const QString& text) {
        if (auto* const button = qobject_cast<QAbstractButton*>(widget)) {
            button->setText(text);
        } else if (auto* const label = qobject_cast<QLabel*>(widget)) {
            label->setText(text);
        } else if (auto* const box = qobject_cast<QGroupBox*>(widget)) {
            box->setTitle(text);
        }
    };
    for (QWidget* const widget : root->findChildren<QWidget*>()) {
        if (const QVariant id = widget->property("catalogueText"); id.isValid()) {
            set_text(widget, Text(id.toString().toStdString()));
            continue;
        }
        const QVariant key = widget->property("catalogueKey");
        if (!key.isValid()) {
            continue;
        }
        const std::string name = key.toString().toStdString();
        if (auto* const list = qobject_cast<QComboBox*>(widget)) {
            // The setting's values, in the enum's order (the page selects them by index).
            if (const auto* setting = FindBasicSetting(name); setting != nullptr && list->count() == 0) {
                for (const auto& choice : UiCatalog::ChoicesOf(setting->EnumIndex())) {
                    list->addItem(Translate(UiCatalog::SettingsContext, choice.label),
                                  choice.value);
                }
            }
        } else {
            set_text(widget, SettingLabel(name));
        }
        if (const QString help = SettingHelp(name); !help.isEmpty()) {
            widget->setToolTip(help);
        }
    }
}

} // namespace UiText
