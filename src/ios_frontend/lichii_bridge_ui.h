// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// C bridge to common/ui_catalog.h for the iOS app. Returned `const char*` pointers stay valid for
// the life of the process; `{setting:...}`, `{choice:...}`, `{exp:...}` and `{text:...}`
// references are already resolved.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A catalogue text by id ("page.graphics", "overlay.real", ...); returns `id` when unknown.
const char* lichii_ui_text(const char* id);

/// Setting metadata from UiCatalog; NULL when the setting has no entry or text.
const char* lichii_ui_setting_label(const char* key);
const char* lichii_ui_setting_help(const char* key);
const char* lichii_ui_setting_reason(const char* key);
/// Whether the setting is shown in the iOS app (`UiCatalog::ShownOn(..., IOS)`).
bool lichii_ui_setting_shown(const char* key);

/// Choices of an enum setting offered on iOS, in the catalogue's display order.
size_t lichii_ui_setting_choice_count(const char* key);
bool lichii_ui_setting_choice(const char* key, size_t index, uint32_t* out_value,
                              const char** out_canonical, const char** out_label);
/// Display name of one choice by its canonical enum name ("MetalFx"), or NULL when not found.
const char* lichii_ui_choice_label(const char* key, const char* canonical);

/// Memory & Experiments sections and switches, in the catalogue's display order.
size_t lichii_ui_experiment_section_count(void);
const char* lichii_ui_experiment_section_id(size_t index);
size_t lichii_ui_experiment_count(void);
bool lichii_ui_experiment_info(size_t index, const char** out_id, const char** out_section,
                               const char** out_label);
const char* lichii_ui_experiment_label(const char* id);

#ifdef __cplusplus
}
#endif
