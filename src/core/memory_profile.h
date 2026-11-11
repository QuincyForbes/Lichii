// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#include "common/common_types.h"

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

struct TouchedMemory {
    struct Entry {
        std::string name; ///< Memory state, " (GPU)" appended where the GPU maps the pages
        u64 backing;      ///< Bytes of backing the application has in this class
        u64 touched;      ///< Of those, bytes used since the measurement started
        u64 touched_16k;  ///< The same where a use keeps the whole 16 KiB host page (iOS)
    };
    bool measuring{}; ///< A measurement is running after this call
    bool known{};     ///< classes, outside and seconds are filled in
    double seconds{}; ///< Since the measurement started
    u64 outside{};    ///< Bytes used that the application doesn't map (kernel, services)
    std::vector<Entry> classes;
    std::string text;
};

/// How much guest memory is used over a stretch of time, per kind of memory: decides whether cold
/// guest data could live outside RAM. reset starts a measurement (HostMemory::ResetTouched: on
/// Windows every page of the process soft-faults once more, a hitch of a second or so); a call
/// without it reports what was used since, and can be repeated; end stops measuring. With a
/// measurement running, a call reports before it resets or ends. Reads no guest memory. Windows
/// counts a page read or written through any mapping; Apple the same from reference bits.
TouchedMemory ProfileTouched(System& system, bool reset, bool end);

} // namespace Core
