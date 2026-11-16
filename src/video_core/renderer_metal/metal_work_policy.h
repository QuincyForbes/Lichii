// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Plain decisions of the Metal rasterizer that follow Vulkan's switches, kept apart so the unit
// tests check them on every platform (src/tests/video_core/metal_work_policy.cpp).

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>

#include "common/common_types.h"

namespace Metal::WorkPolicy {

/// EDEN_NX2_FLUSH_DRAWS as Metal takes it: commit every n draws (at least 1), 0 when unset.
[[nodiscard]] inline u32 ParseFlushDraws(const char* value) {
    if (value == nullptr) {
        return 0;
    }
    return static_cast<u32>((std::max)(std::strtol(value, nullptr, 10), 1L));
}

/// Counts draws against the budget: true on the draw that is to commit the work.
struct DrawFlushCounter {
    u32 budget{};
    u32 since{};

    [[nodiscard]] bool Note() {
        if (budget == 0 || ++since < budget) {
            return false;
        }
        since = 0;
        return true;
    }
};

/// EDEN_NX2_NO_MESH, EDEN_NX2_NO_MESH_EMULATION and EDEN_NX2_NO_MESH_FILE: whether mesh draws
/// are skipped; the file is looked for on the first mesh draw and then every 1024.
struct MeshSkip {
    bool always{};
    bool has_file{};
    u32 checks{};
    bool file_exists{};

    template <typename Exists>
    [[nodiscard]] bool Skip(Exists&& exists) {
        if (has_file && (checks++ % 1024) == 0) {
            file_exists = exists();
        }
        return always || file_exists;
    }
};

[[nodiscard]] inline f32 Halton(u32 index, u32 base) {
    f32 fraction = 1.0f;
    f32 result = 0.0f;
    while (index > 0) {
        fraction /= static_cast<f32>(base);
        result += fraction * static_cast<f32>(index % base);
        index /= base;
    }
    return result;
}

/// Vulkan's Ocarina jitter: Halton (2, 3) centred on the pixel, the frame's place among `phases`.
[[nodiscard]] inline std::array<f32, 2> OcarinaJitter(u64 frame, u32 phases) {
    const u32 index = static_cast<u32>(frame % (std::max)(phases, 1U)) + 1;
    return {Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f};
}

} // namespace Metal::WorkPolicy
