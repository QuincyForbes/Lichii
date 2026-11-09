// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// C++ side of the iOS bridge: the SDL frontend (yuzu_cmd/yuzu.cpp) attaches its objects here and
// polls the requests the app makes; the bridge files (lichii_bridge*.cpp) use them.

#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include "common/common_types.h"
#include "ios_frontend/lichii_bridge.h"

/// Implemented in Swift (AppController.swift): shows the app; called once from SDL_AppInit.
extern "C" void lichii_ui_main(void);

namespace Core {
class System;
}
namespace InputCommon {
class InputSubsystem;
}

namespace IosFrontend {

struct Host {
    Core::System* system{};
    InputCommon::InputSubsystem* input{};
    /// Writes the global config file from Settings::values.
    std::function<void()> save_config;
};

void Attach(const Host& host);
Core::System& System();
InputCommon::InputSubsystem& Input();
void SaveGlobalConfig();
/// Held while game files are read or content is installed: both use the system's filesystem
/// objects (library scans, installs).
std::recursive_mutex& LoaderMutex();

/// Frontend side: requests made through the C API, taken once.
std::optional<std::string> TakeLaunchRequest();
bool TakeStopRequest();
bool TakeCancelRequest();

void SetState(LichiiEmuState state, std::string error = {});
LichiiEmuState GetState();
void SetRunningGame(std::string path, u64 program_id, void* ui_window);
void ClearRunningGame();
u64 RunningProgramId();
std::string RunningGamePath();

/// Loads <program id>.ini over the global values before a boot; RestoreGlobal undoes it.
void ApplyGameConfig(u64 program_id);
void RestoreGlobal();

/// Program id of a game file, 0 when unreadable.
u64 ReadProgramId(const std::string& path);

void PostEvent(LichiiEvent event);
/// A URL or file the system opened with the app (SDL_EVENT_DROP_FILE).
void OpenUrl(std::string url);

/// Pool size the frontend asked the JIT for.
void SetJitPoolBytes(u64 bytes);
/// Ends the background task lichii_jit_request began (the pool is ready or failed).
void EndJitBackgroundTask();

} // namespace IosFrontend
