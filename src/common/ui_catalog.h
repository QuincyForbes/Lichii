// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <span>
#include <string>
#include <string_view>

#include "common/common_types.h"

/// Every name and help text the apps show for settings, their choices, the settings pages and the
/// iOS app's experiments: written once in ui_catalog.cpp, read by the desktop app (through Qt's
/// translations, qt_common/config/ui_text.h) and the iOS app (through lichii_bridge_ui.h). Texts
/// are English; QT_TRANSLATE_NOOP marks them for lupdate. docs/design/lichii-design.md has the
/// format, and tools/ui/catalog_check.py fails when a name is written anywhere else.
namespace UiCatalog {

/// Where a setting or choice is offered.
enum Platform : u8 {
    Windows = 1 << 0,
    Linux = 1 << 1,
    MacOS = 1 << 2,
    IOS = 1 << 3,
    Android = 1 << 4,
    Desktop = Windows | Linux | MacOS,
    Apple = MacOS | IOS,
    Everywhere = Desktop | IOS,
    Nowhere = 0,
};

/// The platform this build runs on.
[[nodiscard]] Platform ThisPlatform();

/// Translation contexts of the texts below (settings and choices keep the desktop app's
/// "ConfigurationShared", so its translations still apply).
inline constexpr const char* SettingsContext = "ConfigurationShared";
inline constexpr const char* TextsContext = "UiCatalog";

struct Setting {
    std::string_view key;  ///< the config key
    const char* label;     ///< nullptr: internal state, or a form field remembered between uses
    const char* help;      ///< what it does, one or more paragraphs; may be nullptr
    u8 shown = Everywhere; ///< platforms that list it
    const char* reason = nullptr; ///< why it does nothing where it isn't listed (the search says)
};

struct Choice {
    u32 (*enum_index)();  ///< Settings::EnumMetadata<T>::Index of the enum it names a value of
    u32 value;
    std::string_view canonical; ///< the enum value's name
    const char* label;
    u8 shown = Everywhere;
};

struct Text {
    std::string_view id;
    const char* text;
};

/// A switch of the iOS app's Memory & Experiments screen (lichii_platform_set_experiment).
struct Experiment {
    std::string_view id;
    std::string_view section; ///< a section id: its texts are "experiments.<section>.title" and
                              ///< "experiments.<section>.footer"
    const char* label;
};

[[nodiscard]] std::span<const Setting> AllSettings();
[[nodiscard]] std::span<const Choice> AllChoices();
[[nodiscard]] std::span<const Text> AllTexts();
[[nodiscard]] std::span<const Experiment> AllExperiments();
/// Section ids of the experiments screen, in order.
[[nodiscard]] std::span<const std::string_view> ExperimentSections();

[[nodiscard]] const Setting* FindSetting(std::string_view key);
/// The choices of one enum, in the order to offer them.
[[nodiscard]] std::span<const Choice> ChoicesOf(u32 enum_index);
[[nodiscard]] const Choice* FindChoice(u32 enum_index, u32 value);
[[nodiscard]] const Experiment* FindExperiment(std::string_view id);
/// The text with this id; the id itself when there is none (a missing entry shows, not crashes).
[[nodiscard]] const char* GetText(std::string_view id);

/// Whether the setting is listed on `platform`.
[[nodiscard]] bool ShownOn(const Setting& setting, Platform platform);

/// Replaces {setting:key}, {choice:key.Value}, {exp:id} and {text:id} with the name they refer
/// to, so a text that names another setting follows it. `name` returns that name (translated by
/// the caller) for a kind ("setting", "choice", "exp", "text") and a reference.
[[nodiscard]] std::string Resolve(std::string_view text,
                                  const std::function<std::string(std::string_view kind,
                                                                  std::string_view ref)>& name);

} // namespace UiCatalog
