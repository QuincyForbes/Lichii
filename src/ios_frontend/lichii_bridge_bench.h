// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Bench mode as the app sees it (lichii_bench.h has the engine side).

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A host started this process for a bench run: the app asks for nothing itself (JIT, launches).
bool lichii_bench_active(void);

#ifdef __cplusplus
}
#endif
