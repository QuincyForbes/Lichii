// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>

// Host code ranges of JIT-compiled guest blocks, so the Windows sampler (nx2_sampler) can name
// the guest code a guest CPU thread runs instead of "(jit)". Dynarmic's perf map feeds it on
// Windows (on Linux the same names go to perf's /tmp/perf-<pid>.map). Opt-in only via
// EDEN_NX2_JIT_SYMBOLS=1 so normal JIT block compilation never takes the lock.
namespace Common::JitSymbols {

bool Enabled() noexcept;

void Register(const void* start, const void* end, std::string_view name) noexcept;

/// Forgets every range (the JIT cleared its code cache).
void Clear() noexcept;

/// The name of the block holding the host address, if one is registered.
std::optional<std::string> Lookup(const void* address);

} // namespace Common::JitSymbols

namespace Nx2Sampler {

// EDEN_NX2_SAMPLE=<thread names>: a sampling profiler for named host threads (Windows only).
// Names are comma separated; a name ending in '*' matches by prefix (CPUCore_*). From
// EDEN_NX2_SAMPLE_START seconds (default 60) for EDEN_NX2_SAMPLE_SECONDS (default 30), each
// thread is paused about 1000 times a second and its call stack recorded; the counts per
// function (where the thread is, and everything on its stack) go to EDEN_NX2_SAMPLE_OUT
// (default nx2_samples.txt).
void StartFromEnvironment();

/// Samples the named threads for `seconds` now (blocking), writes the report to `out` (and
/// the folded stacks to `out`.folded) and returns the report.
std::string Sample(const char* names, double seconds, const std::string& out);

} // namespace Nx2Sampler
