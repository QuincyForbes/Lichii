// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/settings.h"
#include "common/ui_catalog.h"
#include "ios_frontend/lichii_bridge_ui.h"

namespace {

const Settings::BasicSetting* FindBasicSetting(std::string_view key) {
    const auto it = Settings::values.linkage.by_key.find(std::string{key});
    return it == Settings::values.linkage.by_key.end() ? nullptr : it->second;
}

std::string ChoiceLabelString(std::string_view key, std::string_view canonical) {
    const auto* setting = FindBasicSetting(key);
    if (setting == nullptr) {
        return {};
    }
    for (const auto& choice : UiCatalog::ChoicesOf(setting->EnumIndex())) {
        if (choice.canonical == canonical && choice.label != nullptr) {
            return choice.label;
        }
    }
    return {};
}

std::string ResolveText(std::string_view raw);

std::string ResolveReference(std::string_view kind, std::string_view ref) {
    if (kind == "setting") {
        const auto* setting = UiCatalog::FindSetting(ref);
        return setting != nullptr && setting->label != nullptr ? setting->label : std::string{};
    }
    if (kind == "exp") {
        const auto* exp = UiCatalog::FindExperiment(ref);
        return exp != nullptr && exp->label != nullptr ? exp->label : std::string{};
    }
    if (kind == "text") {
        return ResolveText(UiCatalog::GetText(ref));
    }
    if (kind == "choice") {
        const std::size_t dot = ref.find('.');
        if (dot != std::string_view::npos) {
            return ChoiceLabelString(ref.substr(0, dot), ref.substr(dot + 1));
        }
    }
    return {};
}

std::string ResolveText(std::string_view raw) {
    if (raw.find('{') == std::string_view::npos) {
        return std::string{raw};
    }
    return UiCatalog::Resolve(raw, &ResolveReference);
}

struct CachedSetting {
    const char* label = nullptr;
    std::string help;
    std::string reason;
    bool has_help = false;
    bool has_reason = false;
    bool shown = false;
};

struct CachedChoice {
    u32 value = 0;
    std::string canonical;
    const char* label = nullptr;
};

struct CachedExperiment {
    std::string id;
    std::string section;
    const char* label = nullptr;
};

struct CatalogCache {
    std::unordered_map<std::string, std::string> texts;
    std::unordered_map<std::string, CachedSetting> settings;
    std::unordered_map<std::string, std::vector<CachedChoice>> ios_choices;
    std::unordered_map<std::string, std::unordered_map<std::string, const char*>> all_choice_labels;
    std::vector<std::string> experiment_sections;
    std::vector<CachedExperiment> experiments;
    std::unordered_map<std::string, const char*> experiment_labels;

    CatalogCache() {
        for (const auto& entry : UiCatalog::AllTexts()) {
            texts.emplace(std::string{entry.id},
                          entry.text != nullptr ? ResolveText(entry.text) : std::string{});
        }
        for (const auto& entry : UiCatalog::AllSettings()) {
            CachedSetting item;
            item.label = entry.label;
            if (entry.help != nullptr) {
                item.help = ResolveText(entry.help);
                item.has_help = true;
            }
            if (entry.reason != nullptr) {
                item.reason = ResolveText(entry.reason);
                item.has_reason = true;
            }
            item.shown = UiCatalog::ShownOn(entry, UiCatalog::Platform::IOS);
            settings.emplace(std::string{entry.key}, std::move(item));
        }
        for (const auto& [key, basic] : Settings::values.linkage.by_key) {
            auto& visible = ios_choices[key];
            auto& by_name = all_choice_labels[key];
            for (const auto& choice : UiCatalog::ChoicesOf(basic->EnumIndex())) {
                if (choice.label != nullptr) {
                    by_name.emplace(std::string{choice.canonical}, choice.label);
                }
                if ((choice.shown & UiCatalog::Platform::IOS) != 0) {
                    visible.push_back(CachedChoice{
                        choice.value,
                        std::string{choice.canonical},
                        choice.label != nullptr ? choice.label : "",
                    });
                }
            }
        }
        for (const std::string_view section : UiCatalog::ExperimentSections()) {
            experiment_sections.emplace_back(section);
        }
        for (const auto& exp : UiCatalog::AllExperiments()) {
            experiments.push_back(CachedExperiment{
                std::string{exp.id},
                std::string{exp.section},
                exp.label != nullptr ? exp.label : "",
            });
            if (exp.label != nullptr) {
                experiment_labels.emplace(std::string{exp.id}, exp.label);
            }
        }
    }
};

const CatalogCache& Cache() {
    static const CatalogCache cache;
    return cache;
}

} // namespace

extern "C" {

const char* lichii_ui_text(const char* id) {
    if (id == nullptr) {
        return "";
    }
    const auto& texts = Cache().texts;
    const auto it = texts.find(id);
    return it == texts.end() ? id : it->second.c_str();
}

const char* lichii_ui_setting_label(const char* key) {
    if (key == nullptr) {
        return nullptr;
    }
    const auto& settings = Cache().settings;
    const auto it = settings.find(key);
    return it == settings.end() ? nullptr : it->second.label;
}

const char* lichii_ui_setting_help(const char* key) {
    if (key == nullptr) {
        return nullptr;
    }
    const auto& settings = Cache().settings;
    const auto it = settings.find(key);
    return it != settings.end() && it->second.has_help ? it->second.help.c_str() : nullptr;
}

const char* lichii_ui_setting_reason(const char* key) {
    if (key == nullptr) {
        return nullptr;
    }
    const auto& settings = Cache().settings;
    const auto it = settings.find(key);
    return it != settings.end() && it->second.has_reason ? it->second.reason.c_str() : nullptr;
}

bool lichii_ui_setting_shown(const char* key) {
    if (key == nullptr) {
        return false;
    }
    const auto& settings = Cache().settings;
    const auto it = settings.find(key);
    return it != settings.end() && it->second.shown;
}

size_t lichii_ui_setting_choice_count(const char* key) {
    if (key == nullptr) {
        return 0;
    }
    const auto& choices = Cache().ios_choices;
    const auto it = choices.find(key);
    return it == choices.end() ? 0 : it->second.size();
}

bool lichii_ui_setting_choice(const char* key, size_t index, uint32_t* out_value,
                              const char** out_canonical, const char** out_label) {
    if (key == nullptr) {
        return false;
    }
    const auto& choices = Cache().ios_choices;
    const auto it = choices.find(key);
    if (it == choices.end() || index >= it->second.size()) {
        return false;
    }
    const CachedChoice& choice = it->second[index];
    if (out_value != nullptr) {
        *out_value = choice.value;
    }
    if (out_canonical != nullptr) {
        *out_canonical = choice.canonical.c_str();
    }
    if (out_label != nullptr) {
        *out_label = choice.label;
    }
    return true;
}

const char* lichii_ui_choice_label(const char* key, const char* canonical) {
    if (key == nullptr || canonical == nullptr) {
        return nullptr;
    }
    const auto& all = Cache().all_choice_labels;
    const auto key_it = all.find(key);
    if (key_it == all.end()) {
        return nullptr;
    }
    const auto choice_it = key_it->second.find(canonical);
    return choice_it == key_it->second.end() ? nullptr : choice_it->second;
}

size_t lichii_ui_experiment_section_count(void) {
    return Cache().experiment_sections.size();
}

const char* lichii_ui_experiment_section_id(size_t index) {
    const auto& sections = Cache().experiment_sections;
    return index < sections.size() ? sections[index].c_str() : "";
}

size_t lichii_ui_experiment_count(void) {
    return Cache().experiments.size();
}

bool lichii_ui_experiment_info(size_t index, const char** out_id, const char** out_section,
                               const char** out_label) {
    const auto& experiments = Cache().experiments;
    if (index >= experiments.size()) {
        return false;
    }
    const CachedExperiment& exp = experiments[index];
    if (out_id != nullptr) {
        *out_id = exp.id.c_str();
    }
    if (out_section != nullptr) {
        *out_section = exp.section.c_str();
    }
    if (out_label != nullptr) {
        *out_label = exp.label;
    }
    return true;
}

const char* lichii_ui_experiment_label(const char* id) {
    if (id == nullptr) {
        return nullptr;
    }
    const auto& labels = Cache().experiment_labels;
    const auto it = labels.find(id);
    return it == labels.end() ? nullptr : it->second;
}

} // extern "C"
