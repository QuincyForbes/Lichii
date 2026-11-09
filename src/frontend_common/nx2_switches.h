// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace FrontendCommon {

/// Puts the NX2 switches (EDEN_NX2_*) into the environment before anything reads them: the
/// built-in Ocarina of Time tuning, then nx2_switches.txt next to the executable, then the user's
/// copy in the emulator's data folder. A variable already in the real environment always wins.
void LoadNx2Switches(const char* argv0);

} // namespace FrontendCommon
