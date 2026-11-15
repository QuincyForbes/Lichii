// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <string>

namespace Common {

/// Sends stdout and stderr to the file at `path` (appending), so that what the process and the
/// system's frameworks print survives on a device without a console. Does nothing on Windows.
void CaptureConsole(const std::string& path);

/// The last `max_bytes` this process sent to the file CaptureConsole named, as they are in the
/// file; empty without a capture. Bounded and without locks of its own, for a crash handler.
[[nodiscard]] std::string ConsoleCaptureTail(std::size_t max_bytes);

} // namespace Common
