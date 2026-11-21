// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

struct ComputeAccessRange {
    DAddr begin{};
    DAddr end{};

    [[nodiscard]] constexpr bool Overlaps(const ComputeAccessRange& other) const noexcept {
        return begin < other.end && other.begin < end;
    }
};

template <typename Vec>
constexpr void AppendComputeRange(Vec& ranges, DAddr begin, DAddr end) {
    if (end <= begin) {
        return;
    }
    for (auto& existing : ranges) {
        if (begin >= existing.begin && end <= existing.end) {
            return;
        }
        if (begin <= existing.end && existing.begin <= end) {
            existing.begin = (std::min)(existing.begin, begin);
            existing.end = (std::max)(existing.end, end);
            return;
        }
    }
    ranges.push_back(ComputeAccessRange{begin, end});
}

template <typename Vec>
[[nodiscard]] constexpr bool RangeCoveredOrMergeable(const Vec& ranges,
                                                     const ComputeAccessRange& r) noexcept {
    for (const auto& existing : ranges) {
        if (r.begin <= existing.end && existing.begin <= r.end) {
            return true;
        }
    }
    return false;
}

template <typename Vec, typename Handle>
[[nodiscard]] constexpr bool ContainsHandle(const Vec& handles, Handle handle) noexcept {
    for (const auto& existing : handles) {
        if (existing == handle) {
            return true;
        }
    }
    return false;
}

template <typename Vec, typename Handle>
constexpr void AppendUniqueHandle(Vec& handles, Handle handle) {
    if (!handle || ContainsHandle(handles, handle)) {
        return;
    }
    handles.push_back(handle);
}

/// 4-way set-associative O(1) cache of render targets / framebuffers that requested before-pass
/// command buffer splitting, avoiding cyclic FIFO thrashing when > 32 passes run per frame.
template <size_t NumSets = 64, size_t NumWays = 4>
class SplitWantedCache {
public:
    static_assert((NumSets & (NumSets - 1)) == 0, "NumSets must be a power of two");

    constexpr void Insert(u64 key) noexcept {
        if (key == 0 || Contains(key)) {
            return;
        }
        const size_t set = SetIndex(key);
        const u8 way = next_way[set];
        sets[set][way] = key;
        next_way[set] = static_cast<u8>((way + 1) % NumWays);
    }

    [[nodiscard]] constexpr bool Contains(u64 key) const noexcept {
        if (key == 0) {
            return false;
        }
        const auto& ways = sets[SetIndex(key)];
        for (size_t w = 0; w < NumWays; ++w) {
            if (ways[w] == key) {
                return true;
            }
        }
        return false;
    }

private:
    [[nodiscard]] static constexpr size_t SetIndex(u64 key) noexcept {
        const u64 mixed = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
        return static_cast<size_t>((mixed ^ (mixed >> 27)) & (NumSets - 1));
    }

    std::array<std::array<u64, NumWays>, NumSets> sets{};
    std::array<u8, NumSets> next_way{};
};

/// Tracks cumulative read/write footprints of consecutive unbarriered compute dispatches so a
/// barrier is emitted iff a RAW, WAW, or WAR hazard (or unbounded global access) exists.
template <typename ImageHandle = VkImage>
class ComputeDispatchChainBase {
public:
    static constexpr size_t MaxChainRanges = 128;
    static constexpr size_t MaxChainImages = 32;

    constexpr void Reset() noexcept {
        active = false;
        unbounded = false;
        reads.clear();
        writes.clear();
        read_images.clear();
        write_images.clear();
    }

    [[nodiscard]] constexpr bool CanElide(std::span<const ComputeAccessRange> cur_reads,
                                          std::span<const ComputeAccessRange> cur_writes,
                                          std::span<const ImageHandle> cur_read_images,
                                          std::span<const ImageHandle> cur_write_images,
                                          bool cur_unbounded) const noexcept {
        if (!active || unbounded || cur_unbounded) {
            return false;
        }
        size_t added_reads = 0;
        for (const ComputeAccessRange& r : cur_reads) {
            if (r.end > r.begin && !RangeCoveredOrMergeable(reads, r)) {
                ++added_reads;
            }
        }
        size_t added_writes = 0;
        for (const ComputeAccessRange& w : cur_writes) {
            if (w.end > w.begin && !RangeCoveredOrMergeable(writes, w)) {
                ++added_writes;
            }
        }
        size_t added_read_imgs = 0;
        for (const ImageHandle& img : cur_read_images) {
            if (img && !ContainsHandle(read_images, img)) {
                ++added_read_imgs;
            }
        }
        size_t added_write_imgs = 0;
        for (const ImageHandle& img : cur_write_images) {
            if (img && !ContainsHandle(write_images, img)) {
                ++added_write_imgs;
            }
        }
        if (reads.size() + added_reads > MaxChainRanges ||
            writes.size() + added_writes > MaxChainRanges ||
            read_images.size() + added_read_imgs > MaxChainImages ||
            write_images.size() + added_write_imgs > MaxChainImages) {
            return false;
        }
        // RAW: current reads vs earlier writes in the chain.
        for (const ComputeAccessRange& r : cur_reads) {
            for (const ComputeAccessRange& w : writes) {
                if (r.Overlaps(w)) {
                    return false;
                }
            }
        }
        for (const ImageHandle& img : cur_read_images) {
            for (const ImageHandle& w_img : write_images) {
                if (img == w_img) {
                    return false;
                }
            }
        }
        // WAW and WAR: current writes vs earlier writes and earlier reads in the chain.
        for (const ComputeAccessRange& w : cur_writes) {
            for (const ComputeAccessRange& cw : writes) {
                if (w.Overlaps(cw)) {
                    return false;
                }
            }
            for (const ComputeAccessRange& cr : reads) {
                if (w.Overlaps(cr)) {
                    return false;
                }
            }
        }
        for (const ImageHandle& img : cur_write_images) {
            for (const ImageHandle& w_img : write_images) {
                if (img == w_img) {
                    return false;
                }
            }
            for (const ImageHandle& r_img : read_images) {
                if (img == r_img) {
                    return false;
                }
            }
        }
        return true;
    }

    constexpr void Start(std::span<const ComputeAccessRange> cur_reads,
                         std::span<const ComputeAccessRange> cur_writes,
                         std::span<const ImageHandle> cur_read_images,
                         std::span<const ImageHandle> cur_write_images,
                         bool cur_unbounded) {
        Reset();
        active = true;
        if (cur_unbounded || cur_reads.size() > MaxChainRanges ||
            cur_writes.size() > MaxChainRanges || cur_read_images.size() > MaxChainImages ||
            cur_write_images.size() > MaxChainImages) {
            unbounded = true;
            return;
        }
        Append(cur_reads, cur_writes, cur_read_images, cur_write_images);
    }

    constexpr void Append(std::span<const ComputeAccessRange> cur_reads,
                          std::span<const ComputeAccessRange> cur_writes,
                          std::span<const ImageHandle> cur_read_images,
                          std::span<const ImageHandle> cur_write_images) {
        for (const ComputeAccessRange& r : cur_reads) {
            AppendComputeRange(reads, r.begin, r.end);
        }
        for (const ComputeAccessRange& w : cur_writes) {
            AppendComputeRange(writes, w.begin, w.end);
        }
        for (const ImageHandle& img : cur_read_images) {
            AppendUniqueHandle(read_images, img);
        }
        for (const ImageHandle& img : cur_write_images) {
            AppendUniqueHandle(write_images, img);
        }
    }

private:
    template <typename T, size_t Cap>
    class FixedVec {
    public:
        constexpr void clear() noexcept {
            count = 0;
        }
        [[nodiscard]] constexpr bool empty() const noexcept {
            return count == 0;
        }
        [[nodiscard]] constexpr size_t size() const noexcept {
            return count;
        }
        [[nodiscard]] constexpr T& back() noexcept {
            return storage[count - 1];
        }
        constexpr void push_back(const T& value) noexcept {
            if (count < Cap) {
                storage[count++] = value;
            }
        }
        [[nodiscard]] constexpr T* begin() noexcept {
            return storage.data();
        }
        [[nodiscard]] constexpr T* end() noexcept {
            return storage.data() + count;
        }
        [[nodiscard]] constexpr const T* begin() const noexcept {
            return storage.data();
        }
        [[nodiscard]] constexpr const T* end() const noexcept {
            return storage.data() + count;
        }

    private:
        std::array<T, Cap> storage{};
        size_t count{};
    };

    bool active{};
    bool unbounded{};
    FixedVec<ComputeAccessRange, MaxChainRanges> reads;
    FixedVec<ComputeAccessRange, MaxChainRanges> writes;
    FixedVec<ImageHandle, MaxChainImages> read_images;
    FixedVec<ImageHandle, MaxChainImages> write_images;
};

using ComputeDispatchChain = ComputeDispatchChainBase<VkImage>;

static_assert([] {
    ComputeDispatchChainBase<u64> chain;
    const std::array<ComputeAccessRange, 1> r_pool{{{0x10000, 0x20000}}};
    const std::array<ComputeAccessRange, 1> w_out1{{{0x30000, 0x31000}}};
    const std::array<ComputeAccessRange, 1> w_out2{{{0x31000, 0x32000}}}; // adjacent, disjoint
    const std::array<ComputeAccessRange, 1> r_out1{{{0x30800, 0x30900}}}; // RAW against w_out1
    const std::array<ComputeAccessRange, 1> w_pool{{{0x18000, 0x19000}}}; // WAR against r_pool
    const std::span<const u64> no_imgs{};

    // First dispatch cannot elide before chain starts.
    if (chain.CanElide(r_pool, w_out1, no_imgs, no_imgs, false)) {
        return false;
    }
    chain.Start(r_pool, w_out1, no_imgs, no_imgs, false);
    // RAR on r_pool + disjoint adjacent write w_out2 -> safe to elide!
    if (!chain.CanElide(r_pool, w_out2, no_imgs, no_imgs, false)) {
        return false;
    }
    chain.Append(r_pool, w_out2, no_imgs, no_imgs);
    // RAW on w_out1 -> must barrier!
    if (chain.CanElide(r_out1, no_imgs.empty() ? std::span<const ComputeAccessRange>{} : w_out1,
                       no_imgs, no_imgs, false)) {
        return false;
    }
    // WAW on w_out2 -> must barrier!
    if (chain.CanElide(r_pool, w_out2, no_imgs, no_imgs, false)) {
        return false;
    }
    // WAR on r_pool -> must barrier!
    if (chain.CanElide({}, w_pool, no_imgs, no_imgs, false)) {
        return false;
    }
    // Unbounded global access -> must barrier!
    if (chain.CanElide(r_pool, {}, no_imgs, no_imgs, true)) {
        return false;
    }
    // Unbacked image WAR and RAW checks.
    const std::array<u64, 1> img_a{11};
    const std::array<u64, 1> img_b{22};
    chain.Start({}, {}, img_a, img_b, false);
    if (!chain.CanElide({}, {}, img_a, {}, false)) { // RAR on img_a -> safe
        return false;
    }
    if (chain.CanElide({}, {}, img_b, {}, false)) { // RAW on img_b -> barrier
        return false;
    }
    if (chain.CanElide({}, {}, {}, img_a, false)) { // WAR on img_a -> barrier
        return false;
    }
    return true;
}());

} // namespace Vulkan
