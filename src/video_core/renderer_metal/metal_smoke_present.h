// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke's presentation cases (metal-coverage.md N08 to N10 and N17, N18): the
// post-processing chain, the frame dump with the debug server's capture request, and frame
// generation with FSR 3 and MetalFX.

#pragma once

namespace SmokePresent {

/// Runs every case and prints a line for each; 0 when all passed.
int RunAll();

} // namespace SmokePresent
