// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// Page untracking a thread applies itself once it drops the buffer cache's lock
/// (RasterizerVulkan::OnCPUWrite, zortos293/Lichii#1518): changing a page's host protection is a
/// system call, and every other thread drawing or reading waited for it. Only untracking waits: a
/// page protected a little longer traps a write that then finds it untracked, but a page tracked
/// late lets a write through unseen.
class DeferredPageCaching {
public:
    using Run = std::pair<VAddr, size_t>;
    using Apply = void (*)(void* tracker, std::span<const Run> runs, s32 delta);

    void Add(void* tracker, Apply apply, std::span<const Run> runs, s32 delta) {
        batches.push_back(Batch{tracker, apply, delta, collected.size(), runs.size()});
        collected.insert(collected.end(), runs.begin(), runs.end());
    }

    /// In the order they were made: one page can be untracked and tracked again in between.
    void ApplyAll() {
        for (const Batch& batch : batches) {
            batch.apply(batch.tracker,
                        std::span<const Run>(collected).subspan(batch.begin, batch.count),
                        batch.delta);
        }
        batches.clear();
        collected.clear();
    }

    [[nodiscard]] bool Empty() const {
        return batches.empty();
    }

private:
    struct Batch {
        void* tracker;
        Apply apply;
        s32 delta;
        size_t begin;
        size_t count;
    };
    std::vector<Batch> batches;
    std::vector<Run> collected;
};

/// Set while the calling thread collects its page untracking instead of applying it.
inline thread_local DeferredPageCaching* t_deferred_page_caching{};

/// EDEN_NX2_PROTECT_UNLOCKED=0: guest CPU writes change page protection under the buffer cache's
/// lock, as before.
[[nodiscard]] inline bool ProtectUnlockedEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PROTECT_UNLOCKED");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

} // namespace VideoCommon
