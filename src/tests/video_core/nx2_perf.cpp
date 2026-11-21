// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_vulkan/vk_compute_chain.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"

namespace {

using namespace VideoCommon::Nx2Perf;

constexpr size_t Index(Opt opt) {
    return static_cast<size_t>(opt);
}

} // namespace

TEST_CASE("Nx2Perf default mask keeps every switch on", "[nx2_perf]") {
    REQUIRE(ParseOffMask(false, "", "") == u64{0});
}

TEST_CASE("Nx2Perf bank_once_per_frame is on by default and off under SAFE", "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::BankOncePerFrame)] == "bank_once_per_frame");
    REQUIRE(!IsOff(ParseOffMask(false, "", ""), Index(Opt::BankOncePerFrame)));
    REQUIRE(IsOff(ParseOffMask(true, "", ""), Index(Opt::BankOncePerFrame)));
    REQUIRE(ParseOffMask(false, "", "bank_once_per_frame") == Bit(Opt::BankOncePerFrame));
}

TEST_CASE("Nx2Perf OFF=bind_memo flips only its bit", "[nx2_perf]") {
    const auto mask = ParseOffMask(false, "", "bind_memo");
    REQUIRE(mask == Bit(Opt::BindMemo));
    REQUIRE(!IsOff(mask, Index(Opt::IndexRebind)));
}

TEST_CASE("Nx2Perf SAFE is the default-off set", "[nx2_perf]") {
    const auto mask = ParseOffMask(true, "", "");
    REQUIRE(mask == DefaultOffMask());
    for (size_t i = Index(Opt::Bounds); i <= Index(Opt::IndexRebind); ++i) {
        REQUIRE(!IsOff(mask, i));
    }
    for (size_t i = Index(Opt::DownloadBatch); i <= Index(Opt::BindMemo); ++i) {
        REQUIRE(IsOff(mask, i));
    }
}

TEST_CASE("Nx2Perf lists of names", "[nx2_perf]") {
    REQUIRE(ParseOffMask(false, "", "bounds,draw_preamble") ==
            (Bit(Opt::Bounds) | Bit(Opt::DrawPreamble)));
    REQUIRE(ParseOffMask(true, "front_run,download_batch", "") ==
            (DefaultOffMask() & ~(Bit(Opt::FrontRun) | Bit(Opt::DownloadBatch))));
}

TEST_CASE("geometry_consistency: named, and off at start (EDEN_NX2_PERF_ON turns it on)",
          "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::GeometryConsistency)] == "geometry_consistency");
    REQUIRE(!IsOff(DefaultOffMask(), Index(Opt::GeometryConsistency)));
    REQUIRE(ParseOffMask(false, "", "geometry_consistency") == Bit(Opt::GeometryConsistency));
    REQUIRE(!IsOff(ParseOffMask(false, "geometry_consistency", "", Bit(Opt::GeometryConsistency)),
                   Index(Opt::GeometryConsistency)));
}

TEST_CASE("ComputeDispatchChain merges repeated RAR reads and chains 40+ disjoint NVN dispatches",
          "[video_core][nx2_perf][vulkan]") {
    Vulkan::ComputeDispatchChainBase<u64> chain;
    const std::span<const u64> no_imgs{};

    // 8 shared uniform/storage buffers read by every dispatch in a 45-dispatch chain, plus one
    // disjoint 1 KiB NVN/write-extent output slice per dispatch in the same 8 MiB heap.
    std::array<Vulkan::ComputeAccessRange, 8> shared_reads{};
    for (size_t i = 0; i < shared_reads.size(); ++i) {
        const DAddr base = 0x1000'0000ULL + i * 0x10000ULL;
        shared_reads[i] = {base, base + 0x2000ULL};
    }

    const DAddr out_heap = 0x2000'0000ULL;
    const std::array<Vulkan::ComputeAccessRange, 1> first_write{{{out_heap, out_heap + 0x400}}};
    chain.Start(shared_reads, first_write, no_imgs, no_imgs, false);

    // 44 consecutive dispatches: without RAR deduplication/merging, 45 * 8 = 360 > MaxChainRanges.
    for (u32 step = 1; step < 45; ++step) {
        const DAddr slice = out_heap + u64{step} * 0x400ULL;
        const std::array<Vulkan::ComputeAccessRange, 1> cur_write{{{slice, slice + 0x400ULL}}};
        REQUIRE(chain.CanElide(shared_reads, cur_write, no_imgs, no_imgs, false));
        chain.Append(shared_reads, cur_write, no_imgs, no_imgs);
    }

    // RAW against slice 17 inside the merged output range must still force a barrier.
    const DAddr hazard_addr = out_heap + 17ULL * 0x400ULL + 0x80ULL;
    const std::array<Vulkan::ComputeAccessRange, 1> raw_read{{{hazard_addr, hazard_addr + 16}}};
    REQUIRE_FALSE(chain.CanElide(raw_read, {}, no_imgs, no_imgs, false));

    // WAR against shared_reads[3] must still force a barrier.
    const std::array<Vulkan::ComputeAccessRange, 1> war_write{
        {{shared_reads[3].begin + 64, shared_reads[3].begin + 128}}};
    REQUIRE_FALSE(chain.CanElide({}, war_write, no_imgs, no_imgs, false));
}

TEST_CASE("SplitWantedCache avoids cyclic FIFO thrashing across double-buffered passes",
          "[video_core][nx2_perf][vulkan]") {
    Vulkan::SplitWantedCache<64, 4> cache;
    REQUIRE_FALSE(cache.Contains(0));
    cache.Insert(0);
    REQUIRE_FALSE(cache.Contains(0));

    // 96 passes (e.g. 48 passes x double-buffered framebuffers) accessed cyclically: a 32-slot
    // FIFO ring had a 0% hit rate here, whereas the 256-entry 4-way cache retains the working set.
    std::array<u64, 96> keys{};
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = 0x7f00'0000'0000ULL + (i + 1) * 0x1040ULL;
        cache.Insert(keys[i]);
    }
    size_t hits = 0;
    for (const u64 k : keys) {
        if (cache.Contains(k)) {
            ++hits;
        }
    }
    REQUIRE(hits >= 90);
}

TEST_CASE("UpdateDescriptorQueue Rewind reclaims L1 scratch and zeroes entry padding",
          "[video_core][nx2_perf][vulkan]") {
    Vulkan::UpdateDescriptorQueue queue{0x1000, true};

    queue.AcquireScratch(8, false);
    const auto* first_start = queue.UpdateData();
    REQUIRE(first_start != nullptr);
    queue.AddSampledImage((VkImageView)u64{0x1111}, (VkSampler)u64{0x2222});
    queue.AddImage((VkImageView)u64{0x3333});
    queue.AddBuffer((VkBuffer)u64{0x4444}, 64, 256);
    REQUIRE(queue.UpdateCount() == 3);
    const std::array<Vulkan::DescriptorUpdateEntry, 3> snap{first_start[0], first_start[1],
                                                            first_start[2]};

    // Rewind reclaims the scratch cursor so the next draw reuses the exact same cache lines.
    queue.Rewind();
    REQUIRE(queue.UpdateCount() == 0);
    queue.AcquireScratch(8, false);
    REQUIRE(queue.UpdateData() == first_start);
    queue.AddSampledImage((VkImageView)u64{0x1111}, (VkSampler)u64{0x2222});
    queue.AddImage((VkImageView)u64{0x3333});
    queue.AddBuffer((VkBuffer)u64{0x4444}, 64, 256);
    REQUIRE(std::memcmp(first_start, snap.data(), sizeof(snap)) == 0);
}
