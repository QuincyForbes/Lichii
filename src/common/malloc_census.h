// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common {

/// Apple only: logs the malloc blocks in use by size (Common_Memory, "Memory malloc census"), so
/// a device log shows what the heap holds. Holds each zone's lock while it walks the zone (tens
/// of ms). EDEN_NX2_MALLOC_CENSUS=0 turns it off. Does nothing on other platforms.
void LogMallocCensus(const char* reason);

} // namespace Common
