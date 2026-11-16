// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// What became of the guest's draws, dispatches and clears on Metal: drawn as Metal has them,
// drawn through an emulation (by kind), or not done (by cause). The renderer's 10 s
// "Perf Metal coverage:" line. Plain C++.

#pragma once

#include <string>
#include <string_view>

#include "common/common_types.h"

namespace Metal::Coverage {

enum class Work : u32 {
    Draw,
    Dispatch,
    Clear,
    Count,
};

enum class Path : u32 {
    Native,
    /// Task and mesh programs as compute kernels, their output drawn by a vertex program.
    MeshCompute,
    /// Vertex and control programs as compute kernels ahead of Metal's tessellator.
    TessellationKernels,
    /// A topology or index size Metal lacks (quads, fans, loops, 8-bit), through rewritten indices.
    RewrittenIndices,
    /// A multi-draw whose count was read from guest memory, one call per draw.
    IndirectLoop,
    /// A guest pass replaced by one of ours (Ocarina's upscale as a bilinear pass).
    Replaced,
    /// A clear of some channels, a rectangle or some layers as a draw.
    ClearDraw,
    /// Nothing to do (no vertices, everything culled, nothing to clear).
    Empty,
    Count,
};

void Note(Work work, Path path) noexcept;

/// Counts work that was not done; true the first time this cause is seen for this kind of work.
[[nodiscard]] bool NoteRefused(Work work, std::string_view cause);

/// Logs the first refusal of a cause with the state behind it.
void LogRefused(Work work, std::string_view cause, std::string_view state);

/// Counts work that was not done and logs the cause the first time, with detail()'s text.
template <typename Detail>
void Refuse(Work work, std::string_view cause, Detail&& detail) {
    if (NoteRefused(work, cause)) {
        LogRefused(work, cause, detail());
    }
}

inline void Refuse(Work work, std::string_view cause) {
    if (NoteRefused(work, cause)) {
        LogRefused(work, cause, {});
    }
}

/// The counts since the last call as the report line's text; empty when there was no work.
[[nodiscard]] std::string TakeLine();

/// Refusals since the start (smoke tests).
[[nodiscard]] u64 TotalRefused() noexcept;

} // namespace Metal::Coverage
