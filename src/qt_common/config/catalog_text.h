// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <string_view>

#include <QString>

#include "common/settings_enums.h"
#include "common/ui_catalog.h"

class QWidget;

/// The desktop app's view of common/ui_catalog: every text translated (the UI's language) and its
/// {setting:...}-style references filled in.
namespace UiText {

/// A text of the catalogue ("page.graphics").
[[nodiscard]] QString Text(std::string_view id);
/// A setting's name, whether or not this platform lists it; empty when it has none.
[[nodiscard]] QString SettingLabel(std::string_view key);
[[nodiscard]] QString SettingHelp(std::string_view key);
/// Why the setting isn't listed on the platforms it isn't shown on; empty when it has no reason.
[[nodiscard]] QString SettingReason(std::string_view key);
/// The name of one value of an enum.
[[nodiscard]] QString ChoiceLabel(u32 enum_index, u32 value);
/// The name of a value of the setting `key` by its enum name ("upscaler", "MetalFx").
[[nodiscard]] QString ChoiceLabel(std::string_view key, std::string_view canonical);
[[nodiscard]] QString ExperimentLabel(std::string_view id);
/// Fills in the references of a translated text.
[[nodiscard]] QString Resolve(const QString& text);

/// The names of an enum's values this platform offers, in the catalogue's order.
template <typename E>
[[nodiscard]] std::map<E, QString> ChoiceNames() {
    std::map<E, QString> names;
    for (const auto& choice : UiCatalog::ChoicesOf(Settings::EnumMetadata<E>::Index())) {
        if ((choice.shown & UiCatalog::ThisPlatform()) != 0) {
            names.emplace(static_cast<E>(choice.value),
                          ChoiceLabel(Settings::EnumMetadata<E>::Index(), choice.value));
        }
    }
    return names;
}

/// Sets the text (or group box title) and tool tip of every widget under `root` that names its
/// setting in the dynamic property "catalogueKey" (set in the .ui files).
void ApplyCatalogue(QWidget* root);

} // namespace UiText
