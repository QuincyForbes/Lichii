// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

namespace Core {

class System;

/// Where the guest memory and the host process's memory go, for fitting a game into a memory
/// limit (iOS). Reads every page the application has mapped, once, on the calling thread (a
/// second or two for 12 GiB); pages it had to bring in leave the working set again. Logs the
/// report at Info and returns it.
std::string ProfileMemory(System& system);

/// Live C++ allocations by call stack, as JSON (builds with LICHII_ALLOC_TRACE only): bytes,
/// resident bytes and return addresses per stack. tools/nx2/alloc_trace_report.py names them.
std::string AllocTraceReport();

} // namespace Core
