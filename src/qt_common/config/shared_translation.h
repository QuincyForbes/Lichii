// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2024 Torzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <QObject>
#include <QString>
#include "common/common_types.h"
#include "common/settings_enums.h"

namespace ConfigurationShared {
using TranslationMap = std::map<u32, std::pair<QString, QString>>;
using ComboboxTranslations = std::vector<std::pair<u32, QString>>;
using ComboboxTranslationMap = std::map<u32, ComboboxTranslations>;

std::unique_ptr<TranslationMap> InitializeTranslations(QObject* parent);

std::unique_ptr<ComboboxTranslationMap> ComboboxEnumeration(QObject* parent);

} // namespace ConfigurationShared
