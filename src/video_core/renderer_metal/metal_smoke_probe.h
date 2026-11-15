// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke's descriptor probe: what Metal does with descriptors the guest's state can
// lead to (builds them, returns an error, or aborts the process), one child process per suspect.

#pragma once

namespace SmokeProbe {

/// The child's side: builds the suspect `name`, prints the outcome, returns the exit code.
int RunChild(const char* name);

/// Runs every suspect in a child of its own and prints a line for each. 1 when the probe itself
/// doesn't work (the control descriptor doesn't build, no child starts), else 0: an abort of a
/// suspect is a finding, not a failure.
int RunAll();

} // namespace SmokeProbe
