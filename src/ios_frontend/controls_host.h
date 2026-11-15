// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Core {
class System;
}
namespace InputCommon {
class InputSubsystem;
}

namespace LichiiControls {

/// Engine side of the controls bridge: call once the system and the input subsystem exist, and
/// Detach before either is destroyed. Input received earlier is re-sent on Attach.
void Attach(Core::System& system, InputCommon::InputSubsystem& input);
void Detach();

/// Sees every button and stick change sent to the game (bench mode's input recorder); null
/// removes it. Called on the thread that sent the input.
using ButtonTap = void (*)(int player, int button, bool pressed);
using StickTap = void (*)(int player, int stick, float x, float y);
void SetInputTaps(ButtonTap button, StickTap stick);

} // namespace LichiiControls
