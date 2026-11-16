// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "core/memory.h"
#include "video_core/buffer_cache/hot_pages.h"
#include "video_core/buffer_cache/memory_tracker_base.h"

namespace {
using Range = std::pair<u64, u64>;
namespace HotPages = VideoCommon::HotPages;

constexpr u64 PAGE = 4096;
constexpr DAddr BASE = 16ULL << 22;

/// Counts the protection of each page as the rasterizer does.
class Protections {
public:
    void UpdatePagesCachedCount(DAddr addr, size_t size, s32 delta) {
        Apply(addr, size, delta);
    }

    void UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta) {
        for (const auto& [addr, size] : ranges) {
            Apply(addr, size, delta);
        }
        calls += ranges.empty() ? 0 : 1;
    }

    [[nodiscard]] int Count(DAddr addr) const {
        const auto it = pages.find(addr / PAGE);
        return it == pages.end() ? 0 : it->second;
    }

    size_t calls = 0;

private:
    void Apply(DAddr addr, size_t size, s32 delta) {
        for (u64 page = addr / PAGE; page < (addr + size + PAGE - 1) / PAGE; ++page) {
            pages[page] += delta;
        }
    }

    ::Common::unordered_map<u64, int> pages;
};

using Tracker = VideoCommon::MemoryTrackerBase<Protections>;

struct ModeGuard {
    explicit ModeGuard(int mode) : old{HotPages::g_mode} {
        HotPages::g_mode = mode;
    }
    ~ModeGuard() {
        HotPages::g_mode = old;
    }
    int old;
};

std::vector<Range> Upload(Tracker& tracker, DAddr addr, u64 size) {
    std::vector<Range> ranges;
    tracker.ForEachUploadRange(addr, size,
                               [&](u64 begin, u64 bytes) { ranges.emplace_back(begin, bytes); });
    return ranges;
}

std::vector<Range> Advance(Tracker& tracker, bool new_frame) {
    std::vector<Range> released;
    tracker.AdvanceHotPages(new_frame,
                            [&](u64 begin, u64 bytes) { released.emplace_back(begin, bytes); });
    return released;
}

/// Upload, trapped write, upload, trapped write: the second trap turns the page hot.
void Promote(Tracker& tracker, DAddr page) {
    Upload(tracker, page, PAGE);
    tracker.MarkRegionAsCpuModified(page + 16, 4);
    Upload(tracker, page, PAGE);
    tracker.MarkRegionAsCpuModified(page + 32, 4);
}
} // Anonymous namespace

TEST_CASE("HotPages: second trapped write promotes, upload leaves it unprotected", "[video_core]") {
    ModeGuard guard{1};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    Upload(*tracker, BASE, PAGE);
    REQUIRE(protections.Count(BASE) == 1);
    tracker->MarkRegionAsCpuModified(BASE + 16, 4);
    REQUIRE(protections.Count(BASE) == 0);
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
    Upload(*tracker, BASE, PAGE);
    REQUIRE(protections.Count(BASE) == 1);
    tracker->MarkRegionAsCpuModified(BASE + 32, 4);
    REQUIRE(tracker->IsRegionHot(BASE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionHot(BASE + PAGE, PAGE));

    REQUIRE(Upload(*tracker, BASE, PAGE) == std::vector<Range>{{BASE, PAGE}});
    REQUIRE(protections.Count(BASE) == 0);
    // Uploaded in this epoch: clean for every query until the epoch ends.
    REQUIRE_FALSE(tracker->IsRegionUploadDirty(BASE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionCpuModified(BASE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionFullyCpuModified(BASE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionWhollyCpuModified(BASE, PAGE));
    REQUIRE(tracker->ModifiedCpuRegion(BASE, PAGE) == Range{0, 0});
    REQUIRE(Upload(*tracker, BASE, PAGE).empty());
}

TEST_CASE("HotPages: each epoch end reports a hot page as written", "[video_core]") {
    ModeGuard guard{2};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    Promote(*tracker, BASE);
    Promote(*tracker, BASE + 2 * PAGE);
    Upload(*tracker, BASE, 3 * PAGE);
    REQUIRE(protections.Count(BASE) == 0);
    REQUIRE(protections.Count(BASE + PAGE) == 1);

    // The buffer cache passes these runs to NoteUniformWrite, NoteCpuDirty and
    // MarkVertexBuffersStale (BufferCache::NoteHotPagesWritten).
    REQUIRE(Advance(*tracker, false) == std::vector<Range>{{BASE, PAGE}, {BASE + 2 * PAGE, PAGE}});
    REQUIRE(tracker->IsRegionUploadDirty(BASE, PAGE));
    REQUIRE(tracker->IsRegionCpuModified(BASE, PAGE));
    REQUIRE(tracker->IsRegionFullyCpuModified(BASE, PAGE));
    REQUIRE(tracker->IsRegionWhollyCpuModified(BASE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionUploadDirty(BASE + PAGE, PAGE));
    // The guest wrote nothing the tracker saw: the next use uploads anyway.
    REQUIRE(Upload(*tracker, BASE, 3 * PAGE) ==
            std::vector<Range>{{BASE, PAGE}, {BASE + 2 * PAGE, PAGE}});
    REQUIRE(protections.Count(BASE) == 0);

    // Nothing uploaded since: nothing to report.
    Upload(*tracker, BASE, PAGE);
    Advance(*tracker, false);
    REQUIRE(Advance(*tracker, false).empty());
}

TEST_CASE("HotPages: a page cools down at the end of the window", "[video_core]") {
    ModeGuard guard{1};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    Promote(*tracker, BASE);
    for (u32 frame = 0; frame + 1 < HotPages::WindowFrames; ++frame) {
        Upload(*tracker, BASE, PAGE);
        Advance(*tracker, true);
        REQUIRE(tracker->IsRegionHot(BASE, PAGE));
        REQUIRE(protections.Count(BASE) == 0);
    }
    Upload(*tracker, BASE, PAGE);
    REQUIRE(Advance(*tracker, true) == std::vector<Range>{{BASE, PAGE}});
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
    // Dirty from the release, then protected again by its next upload.
    REQUIRE(Upload(*tracker, BASE, PAGE) == std::vector<Range>{{BASE, PAGE}});
    REQUIRE(protections.Count(BASE) == 1);
    // One trap is not enough in the new window.
    tracker->MarkRegionAsCpuModified(BASE, 4);
    Upload(*tracker, BASE, PAGE);
    REQUIRE(protections.Count(BASE) == 1);
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
    tracker->MarkRegionAsCpuModified(BASE, 4);
    REQUIRE(tracker->IsRegionHot(BASE, PAGE));
}

TEST_CASE("HotPages: a GPU write demotes a hot page for good", "[video_core]") {
    ModeGuard guard{1};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    Promote(*tracker, BASE);
    Promote(*tracker, BASE + PAGE);
    REQUIRE(tracker->IsRegionHot(BASE + PAGE, PAGE));
    // Not uploaded in this epoch: it already reads as dirty, nothing to report.
    REQUIRE_FALSE(tracker->MarkRegionAsGpuModified(BASE + PAGE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionHot(BASE + PAGE, PAGE));
    Upload(*tracker, BASE, PAGE);
    REQUIRE(protections.Count(BASE) == 0);
    REQUIRE(tracker->MarkRegionAsGpuModified(BASE + 64, 16));
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
    // Write-protected again so downloads and read traps see the GPU's bytes, and still dirty for
    // whatever the guest wrote since the upload.
    REQUIRE(tracker->IsRegionGpuModified(BASE, PAGE));
    REQUIRE(tracker->ModifiedGpuRegion(BASE, PAGE) == Range{BASE, BASE + PAGE});
    REQUIRE(protections.Count(BASE) == 1);
    REQUIRE(protections.Count(BASE + PAGE) == 1);
    REQUIRE(tracker->IsRegionUploadDirty(BASE, PAGE));
    REQUIRE_FALSE(tracker->IsRegionFullyCpuModified(BASE, PAGE));
    REQUIRE(Advance(*tracker, false).empty());

    tracker->UnmarkRegionAsGpuModified(BASE, PAGE);
    for (int round = 0; round < 3; ++round) {
        Upload(*tracker, BASE, PAGE);
        REQUIRE(protections.Count(BASE) == 1);
        tracker->MarkRegionAsCpuModified(BASE, 4);
    }
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
}

TEST_CASE("HotPages: a hot page has nothing to download", "[video_core]") {
    ModeGuard guard{1};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    Promote(*tracker, BASE);
    Upload(*tracker, BASE, PAGE);
    REQUIRE_FALSE(tracker->IsRegionGpuModified(BASE, PAGE));
    REQUIRE(tracker->ModifiedGpuRegion(BASE, PAGE) == Range{0, 0});
    // A download writes guest memory and marks it CPU-modified with no trap behind it.
    tracker->MarkRegionAsCpuModified(BASE, PAGE, false);
    REQUIRE(tracker->IsRegionUploadDirty(BASE, PAGE));
    REQUIRE(tracker->IsRegionHot(BASE, PAGE));
}

TEST_CASE("HotPages: marks without a guest write never promote", "[video_core]") {
    ModeGuard guard{1};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    for (int round = 0; round < 4; ++round) {
        Upload(*tracker, BASE, PAGE);
        REQUIRE(protections.Count(BASE) == 1);
        tracker->MarkRegionAsCpuModified(BASE, PAGE, false);
    }
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
}

TEST_CASE("HotPages: off keeps every page trapped", "[video_core]") {
    ModeGuard guard{0};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    for (int round = 0; round < 4; ++round) {
        Upload(*tracker, BASE, PAGE);
        REQUIRE(protections.Count(BASE) == 1);
        tracker->MarkRegionAsCpuModified(BASE, 4);
        REQUIRE(protections.Count(BASE) == 0);
    }
    REQUIRE_FALSE(tracker->IsRegionHot(BASE, PAGE));
    REQUIRE(Advance(*tracker, true).empty());
}

TEST_CASE("HotPages: hot uploads save protection changes", "[video_core]") {
    ModeGuard guard{1};
    Protections protections;
    std::optional<Tracker> tracker(protections);
    Promote(*tracker, BASE);
    const size_t before = protections.calls;
    const u64 protects = HotPages::g_counters.protect_ranges.load();
    for (int frame = 0; frame < 8; ++frame) {
        Upload(*tracker, BASE, PAGE);
        Advance(*tracker, true);
    }
    REQUIRE(protections.calls == before);
    REQUIRE(HotPages::g_counters.protect_ranges.load() == protects);
}
