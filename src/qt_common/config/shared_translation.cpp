// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 Torzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shared_translation.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <QString>
#include "common/settings.h"
#include "common/settings_enums.h"
#include "common/time_zone.h"
#include "common/ui_catalog.h"
#include "qt_common/config/catalog_text.h"
#include "qt_common/config/uisettings.h"

namespace ConfigurationShared {

// The names and help texts are common/ui_catalog.cpp's, also the iOS app's.
std::unique_ptr<TranslationMap> InitializeTranslations(QObject*) {
    auto translations = std::make_unique<TranslationMap>();
    const UiCatalog::Platform platform = UiCatalog::ThisPlatform();
    for (const UiCatalog::Setting& entry : UiCatalog::AllSettings()) {
        const std::string key{entry.key};
        const Settings::BasicSetting* setting = nullptr;
        for (const auto* const linkage : {&Settings::values.linkage, &UISettings::values.linkage}) {
            if (const auto it = linkage->by_key.find(key); it != linkage->by_key.end()) {
                setting = it->second;
                break;
            }
        }
        if (setting == nullptr) {
            continue;
        }
        // A blank name makes no widget: settings this platform doesn't list, and the ones shown
        // as part of another row (a value's check box).
        const QString label = UiCatalog::ShownOn(entry, platform)
                                  ? UiText::SettingLabel(entry.key)
                                  : QString();
        translations->insert({setting->Id(), {label, UiText::SettingHelp(entry.key)}});
    }
    return translations;
}

std::unique_ptr<ComboboxTranslationMap> ComboboxEnumeration(QObject*) {
    auto translations = std::make_unique<ComboboxTranslationMap>();
    const UiCatalog::Platform platform = UiCatalog::ThisPlatform();
    for (const UiCatalog::Choice& choice : UiCatalog::AllChoices()) {
        const u32 index = choice.enum_index();
        // The graphics page fills the VSync modes the driver offers.
        if (index == Settings::EnumMetadata<Settings::VSyncMode>::Index() ||
            (choice.shown & platform) == 0) {
            continue;
        }
        if (index == Settings::EnumMetadata<Settings::RendererBackend>::Index()) {
            const auto backend = static_cast<Settings::RendererBackend>(choice.value);
#ifndef HAS_OPENGL
            if (backend == Settings::RendererBackend::OpenGL_GLSL ||
                backend == Settings::RendererBackend::OpenGL_GLASM ||
                backend == Settings::RendererBackend::OpenGL_SPIRV) {
                continue;
            }
#endif
#ifndef LICHII_METAL_RENDERER
            if (backend == Settings::RendererBackend::Metal) {
                continue;
            }
#endif
        }
        QString label = UiText::ChoiceLabel(index, choice.value);
        if (index == Settings::EnumMetadata<Settings::TimeZone>::Index()) {
            if (choice.value == static_cast<u32>(Settings::TimeZone::Auto)) {
                label = QStringLiteral("%1 (%2)").arg(
                    label, QString::fromStdString(
                               Settings::GetTimeZoneString(Settings::TimeZone::Auto)));
            } else if (choice.value == static_cast<u32>(Settings::TimeZone::Default)) {
                label = QStringLiteral("%1 (%2)").arg(
                    label, QString::fromStdString(Common::TimeZone::GetDefaultTimeZone()));
            }
        }
        (*translations)[index].emplace_back(choice.value, label);
    }
    return translations;
}

} // namespace ConfigurationShared
