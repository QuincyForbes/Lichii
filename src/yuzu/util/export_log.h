// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

class QWidget;

namespace VkDeviceInfo {
class Record;
}

namespace ExportLog {

/// Prompts for a save location (Desktop by default) and packages eden_log.txt, eden_log.txt.old,
/// sanitized configuration files (qt-config.ini + custom/*.ini), and system-info.txt into a .zip
/// archive for bug reporting, with the OS user name, machine name, and tokens redacted.
void Export(QWidget* parent, const std::vector<VkDeviceInfo::Record>& vk_device_records);

} // namespace ExportLog
