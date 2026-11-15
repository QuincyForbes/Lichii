// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The rule for how much CPU time a frame thread declares (Window) is a port of nxnative's
// dm/gfx/nvn/nvn_rt.h (seedds/nxnative PR #2569, 4c71083), and the declaration itself of
// rt_declare / rt_tick in dm/gfx/nvn/hle_nvn.c (PR #2137, b06f4e0):
//
// MIT License, Copyright (c) 2026 seedds
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <array>
#include <span>
#include <string>

#include "common/common_types.h"

namespace Common::FrameDeadline {

/// The threads that may declare, as bits of EDEN_NX2_APPLE_FRAME_DEADLINE.
enum class Role : u32 {
    GpuBack = 1,
    GpuFront = 2,
    VulkanWorker = 4,
    GuestCore = 8,
};

/// A thread's CPU time a frame over the last frames, and from it the computation to declare.
class Window {
public:
    static constexpr size_t Size = 120;

    /// What a thread with these per-frame CPU times (ns, reordered) declares for `period_ns`:
    /// a quarter over the 90th percentile or a tenth over the peak, whichever is more, between
    /// 1 ms and 0.9 periods. Declared too low, the kernel holds the thread up every frame.
    [[nodiscard]] static u64 Need(std::span<u64> samples_ns, u64 period_ns);

    /// Takes the thread's CPU clock at frame `frame`. Returns the computation to declare now,
    /// or 0 to keep the last one: only once a window is full, and only when it moved the
    /// declared one by more than 15 %.
    [[nodiscard]] u64 Sample(u64 frame, u64 cpu_ns, u64 period_ns);

    /// The computation Sample last returned.
    [[nodiscard]] u64 Declared() const {
        return declared_ns;
    }

    /// The mean CPU time a frame of the last full window.
    [[nodiscard]] u64 MeanNs() const {
        return mean_ns;
    }

private:
    std::array<u64, Size> samples{};
    size_t count{};
    u64 last_frame{};
    u64 last_cpu_ns{};
    u64 declared_ns{};
    u64 mean_ns{};
    bool primed{};
};

/// The roles that declare (0 = none): EDEN_NX2_APPLE_FRAME_DEADLINE, by default the three GPU
/// threads on iPhone and iPad and none elsewhere. Always 0 off Apple platforms.
[[nodiscard]] u32 Roles();

/// The frame the game aims for, in ns (its swap interval at the display rate, or a lower limit).
/// 0 = unpaced: the frame time measured stands in, in whole 60 Hz refreshes.
void SetPeriodNs(u64 period_ns);

/// Once a frame, from the thread that presents.
void NoteFrame();

#ifdef __APPLE__
/// On a frame thread, once: the thread declares its frame time from then on if its role is on.
void Join(Role role, int index = -1);

/// On a joined thread, before a piece of its work; nothing unless a frame went by.
void Tick();
#else
inline void Join(Role, int = -1) {}
inline void Tick() {}
#endif

/// What each joined thread declared and what the kernel made of it since the last call, for
/// the Perf line; empty when no thread declares.
[[nodiscard]] std::string TakeStatusText();

} // namespace Common::FrameDeadline
