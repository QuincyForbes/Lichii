// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Plain decisions of the Metal rasterizer that follow Vulkan's switches, kept apart so the unit
// tests check them on every platform (src/tests/video_core/metal_work_policy.cpp).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/pipeline_cache_policy.h"

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

/// Vulkan's host pacer (EDEN_NX2_HOST_PACE) the Metal way: a frame is presented no earlier than
/// k refreshes after the one before (presentDrawable:afterMinimumDuration:), k being the game's
/// frame period in refreshes, so a 30 fps game on a 120 Hz screen shows every frame for four
/// refreshes instead of three and five by turns.
struct PresentPacer {
    f64 refresh_ns{};        ///< The display's refresh period; 0: unknown
    f64 content_ns{};        ///< The game's frame period (moving average)
    u32 frames{};

    /// The time since the previous guest frame reached the present.
    void NoteFrame(f64 interval_ns) {
        // A load or a menu that stalls for a while starts the average again.
        if (interval_ns <= 0.0 || interval_ns > 250e6) {
            frames = 0;
            return;
        }
        content_ns = frames == 0 ? interval_ns : content_ns * 0.9 + interval_ns * 0.1;
        ++frames;
    }

    /// The minimum time from the previous frame's present to this one's, in nanoseconds; 0 to
    /// present at once (refresh unknown, a frame every refresh, or no requested period).
    /// When `requested_ns` > 0 (Nx2Perf::g_guest_frame_interval_ns, the period the game asks for),
    /// k comes from that period as Vulkan's HostPacer::Target does: pacing from arrival times
    /// (`use_arrival`, EDEN_NX2_METAL_PACE_ARRIVAL=1) locked a 60 fps title screen at 4.5
    /// refreshes = 37.5 ms (26.7 fps) on a 120 Hz screen with three drawables.
    [[nodiscard]] f64 MinimumDurationNs(f64 requested_ns = 0.0, bool use_arrival = true) const {
        if (refresh_ns <= 0.0) {
            return 0.0;
        }
        if (requested_ns > 0.0) {
            const f64 k = std::floor(requested_ns / refresh_ns + 0.02);
            return k < 2.0 ? 0.0 : (k - 0.5) * refresh_ns;
        }
        if (!use_arrival || frames < 8) {
            return 0.0;
        }
        const f64 refreshes = content_ns / refresh_ns;
        const f64 k = refreshes < 1.5 ? 1.0 : static_cast<f64>(static_cast<u64>(refreshes + 0.5));
        // Half a refresh early: the drawable still lands on the k-th refresh.
        return k < 2.0 ? 0.0 : (k - 0.5) * refresh_ns;
    }
};

/// A graphics pipeline as eviction sees it.
struct PipelineUse {
    u64 last_use;  ///< The frame of its last draw
    bool used;     ///< A draw of this run took it (else it was built at boot from the records)
    bool current;  ///< The current draw's
};

/// Vulkan's EvictUnusedPipelines on Apple (MoltenVK): only on a memory pressure signal, and with
/// EDEN_NX2_PIPELINE_EVICT_LEFT_MIB only near the limit (`critical`). The indices of the
/// pipelines to destroy, least recently used first.
[[nodiscard]] inline std::vector<size_t> PressureEvictions(std::span<const PipelineUse> pipelines,
                                                           u64 frame, bool critical,
                                                           size_t evict_max) {
    // Unused for ~10 s (near the limit ~60 s); the 256 most recent stay (512), and fewer than 128
    // stale ones are kept too: an iPhone's 1,041 evicted at once gave back at most ~28 MiB.
    const u64 evict_after = critical ? 3600 : 600;
    const size_t keep = critical ? 512 : 256;
    constexpr size_t MinEvicted = 128;
    if (frame <= evict_after || pipelines.size() <= keep) {
        return {};
    }
    const u64 cutoff = frame - evict_after;
    // Near the limit, boot-built pipelines no draw has taken stay up to a budget.
    const size_t max_unused_prebuilt = critical ? 2000 : 0;
    size_t kept_prebuilt = 0;
    std::vector<std::pair<u64, size_t>> candidates;
    for (size_t index = 0; index < pipelines.size(); ++index) {
        const PipelineUse& pipeline = pipelines[index];
        if (pipeline.current || pipeline.last_use >= cutoff) {
            continue;
        }
        if (!pipeline.used && kept_prebuilt < max_unused_prebuilt) {
            ++kept_prebuilt;
            continue;
        }
        candidates.emplace_back(pipeline.last_use, index);
    }
    size_t evicted = VideoCommon::PressurePipelinesToEvict(pipelines.size(), candidates.size(),
                                                           keep, MinEvicted);
    if (critical) {
        evicted = (std::min)(evicted, evict_max);
    }
    if (evicted == 0) {
        return {};
    }
    std::nth_element(candidates.begin(), candidates.begin() + (evicted - 1), candidates.end());
    std::vector<size_t> indices;
    indices.reserve(evicted);
    for (size_t i = 0; i < evicted; ++i) {
        indices.push_back(candidates[i].second);
    }
    return indices;
}

/// Vulkan's Ocarina jitter: Halton (2, 3) centred on the pixel, the frame's place among `phases`.
[[nodiscard]] inline std::array<f32, 2> OcarinaJitter(u64 frame, u32 phases) {
    const u32 index = static_cast<u32>(frame % (std::max)(phases, 1U)) + 1;
    return {Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f};
}

} // namespace Metal::WorkPolicy
