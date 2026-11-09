// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Hooks of the content bridge for the SDL frontend's boot path (yuzu_cmd/yuzu.cpp).

#include <string>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace LichiiContent {

/// Puts the iOS dialog applets (error, software keyboard, profile select, controller) into the
/// system's applet holder. Call before System::Load.
void InstallFrontendApplets(Core::System& system);

/// The applet a boot of `path` starts as (a Service::AM::AppletId value): MiiEdit after
/// lichii_content_prepare_mii_edit returned that path, else Application. Reading resets it.
u32 TakePendingAppletId(const std::string& path);

} // namespace LichiiContent
