// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// C++ side of the content bridge, for the core bridge (lichii_bridge.cpp) that owns the System.

#include <functional>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace FileSys {
class ManualContentProvider;
}

namespace LichiiContent {

struct Host {
    Core::System* system{};
    FileSys::ManualContentProvider* manual_provider{};
    /// Persists Settings::values (disabled_addons, current_user) to the config file.
    std::function<void()> save_config;
    /// True while a game is loaded; firmware and save changes are refused then.
    std::function<bool()> is_running;
};

/// Hands the app's System to the content bridge. The System must have its filesystem and
/// content provider set (CreateFactories done). Until this is called the bridge uses a private
/// System of its own for installs. Pass an empty Host before destroying the System.
void SetHost(Host host);

/// Puts the iOS dialog applets (error, software keyboard, profile select, controller) into the
/// system's applet holder. Call before System::Load.
void InstallFrontendApplets(Core::System& system);

/// The applet the next boot should start as (Service::AM::AppletId value), set by
/// lichii_content_prepare_applet. Application (1) when nothing was requested; reading clears it.
u32 TakePendingAppletId();

} // namespace LichiiContent
