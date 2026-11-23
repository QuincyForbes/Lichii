// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Vulkan's binding memo (EDEN_NX2_BINDMEMO, renderer_vulkan/pipeline_helper.h) for the Metal
// rasterizer: a draw's or dispatch's texture and image bindings looked up in the texture cache's
// memo (TextureCache::FindBinding) instead of resolving every view and sampler again, the ones
// missed resolved as FillImageViews and GetSamplerId do and recorded. Metal binds the views it
// gets directly, so nothing is kept in a memo's payload.

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "core/memory.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/textures/texture.h"

namespace Metal::BindMemo {

/// Vulkan's EDEN_NX2_BINDMEMO (on unless 0, read under EDEN_NX2_METAL_VULKAN_SWITCHES), and
/// EDEN_NX2_METAL_BINDMEMO=0 to leave it off on Metal alone.
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const metal = std::getenv("EDEN_NX2_METAL_BINDMEMO");
        if (metal != nullptr && metal[0] == '0') {
            return false;
        }
        const char* const vulkan = VulkanSwitch("EDEN_NX2_BINDMEMO");
        return vulkan == nullptr || vulkan[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_METAL_BINDMEMO_SCRATCH=0: Resolve allocates its per-call lists afresh.
[[nodiscard]] inline bool ScratchEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_BINDMEMO_SCRATCH");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Vulkan's BindingKind: what a binding's resolution depends on besides its view and sampler.
namespace Kind {
constexpr u32 TextureBuffer = 1;
constexpr u32 ImageBuffer = 2;
constexpr u32 Texture = 3;
constexpr u32 Image = 4;
constexpr u32 ClassMask = 7;
constexpr u32 ComputeBit = 1U << 31;
} // namespace Kind

/// Appends the kind of each binding of a stage in the order its views are read: texel buffers,
/// image buffers, textures, images (Vulkan's AppendBindingKinds).
inline void AppendKinds(const Shader::Info& info, u32 compute_bit, std::vector<u32>& kinds) {
    for (const auto& desc : info.texture_buffer_descriptors) {
        kinds.insert(kinds.end(), desc.count, compute_bit | Kind::TextureBuffer);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        kinds.insert(kinds.end(), desc.count, compute_bit | Kind::ImageBuffer);
    }
    for (const auto& desc : info.texture_descriptors) {
        const u32 kind = compute_bit | Kind::Texture | (static_cast<u32>(desc.type) << 3) |
                         (desc.is_depth ? 1U << 7 : 0U) |
                         (desc.handle_program != ~0U ? 1U << 14 : 0U);
        kinds.insert(kinds.end(), desc.count, kind);
    }
    for (const auto& desc : info.image_descriptors) {
        const u32 kind = compute_bit | Kind::Image | (static_cast<u32>(desc.type) << 3) |
                         (static_cast<u32>(desc.format) << 8) |
                         (desc.is_byte_addressed ? 1U << 12 : 0U) |
                         (desc.is_written ? 1U << 13 : 0U);
        kinds.insert(kinds.end(), desc.count, kind);
    }
}

/// Hits and misses since the last call ("Metal binding memo:" line).
struct Counts {
    u64 hits{};
    u64 misses{};
    u64 full_resolves{}; ///< Draws whose misses had every binding resolved again
};

[[nodiscard]] constexpr u32 CanonicalKind(u32 kind) noexcept {
    constexpr u32 ComputeAndHandleMask = Kind::ComputeBit | (1U << 14);
    const u32 cls = kind & Kind::ClassMask;
    if (cls == Kind::Texture) {
        return Kind::Texture | (kind & ComputeAndHandleMask);
    }
    if (cls == Kind::Image) {
        return Kind::Image | (kind & (ComputeAndHandleMask | (1U << 13)));
    }
    return cls | (kind & ComputeAndHandleMask);
}

[[nodiscard]] inline u64 L2KeyHash(u32 tic_handle, u32 tsc_index, u32 kind) noexcept {
    u64 h = (u64{tic_handle} * 0x9E3779B97F4A7C15ULL) ^
            (u64{tsc_index} * 0xC2B2AE3D27D4EB4FULL) ^
            (u64{kind} * 0x165667B19E3779F9ULL);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return h;
}

template <typename Cache>
struct OverflowState {
    static constexpr size_t NumSets = 2048;
    static constexpr size_t NumWays = 8;
    static constexpr size_t SeenL1Slots = 512;

    using Memo = typename Cache::BindingMemo;
    struct Entry {
        Memo memo{};
        u64 last_l1_frame{};
    };
    struct SeenL1 {
        const Memo* ptr{};
        u64 epoch{};
        u64 image_serial{};
        u32 tic_handle{};
        u32 tsc_index{};
        u32 kind{};
    };
    struct TableBounds {
        const void* channel{};
        size_t tic_capacity{0x400};
        size_t tsc_capacity{0x400};
    };

    std::vector<Entry> entries;
    std::vector<u32> tags;
    std::vector<u8> used;
    std::vector<u8> clock_hand;
    std::array<SeenL1, SeenL1Slots> seen_l1{};
    u64 seen_l1_frame{~u64{0}};
    std::array<TableBounds, 2> bounds{};
    u64 last_seen_epoch{};

    [[nodiscard]] static size_t SeenSlotOf(const Memo& memo) noexcept {
        const u64 raw = static_cast<u64>(reinterpret_cast<std::uintptr_t>(&memo));
        return static_cast<size_t>(((raw >> 6) ^ (raw >> 15)) & (SeenL1Slots - 1));
    }

    void RememberL1Seen(const Memo& memo, u64 current_frame) noexcept {
        if (seen_l1_frame != current_frame) {
            seen_l1.fill({});
            seen_l1_frame = current_frame;
        }
        seen_l1[SeenSlotOf(memo)] = SeenL1{
            .ptr = &memo,
            .epoch = memo.epoch,
            .image_serial = memo.image_serial,
            .tic_handle = memo.tic_handle,
            .tsc_index = memo.tsc_index,
            .kind = memo.kind,
        };
    }

    void NoteLimits(bool compute, const void* channel, u32 tic_limit, u32 tsc_limit) noexcept {
        TableBounds& b = bounds[compute ? 1 : 0];
        if (b.channel != channel) {
            b.channel = channel;
            b.tic_capacity = 0x400;
            b.tsc_capacity = 0x400;
        }
        if (tic_limit < 0xfff80000U) {
            const size_t needed = Common::AlignUp(size_t{tic_limit} + 1, size_t{0x400});
            if (needed > b.tic_capacity) {
                b.tic_capacity = (std::max)(needed, b.tic_capacity * 2);
            }
        }
        if (tsc_limit < 0xfff80000U) {
            const size_t needed = Common::AlignUp(size_t{tsc_limit} + 1, size_t{0x400});
            if (needed > b.tsc_capacity) {
                b.tsc_capacity = (std::max)(needed, b.tsc_capacity * 2);
            }
        }
    }

    void EnsureAllocated() {
        if (tags.empty()) {
            entries.resize(NumSets * NumWays);
            tags.assign(NumSets * NumWays, 0);
            used.assign(NumSets * NumWays, 0);
            clock_hand.assign(NumSets, 0);
        }
    }

    [[nodiscard]] bool RefreshL1Hit(const Memo& memo, u64 current_frame) noexcept {
        last_seen_epoch = memo.epoch;
        if (tags.empty()) {
            return false;
        }
        if (seen_l1_frame != current_frame) {
            seen_l1.fill({});
            seen_l1_frame = current_frame;
        } else {
            const SeenL1& s = seen_l1[SeenSlotOf(memo)];
            if (s.ptr == &memo && s.epoch == memo.epoch && s.image_serial == memo.image_serial &&
                s.tic_handle == memo.tic_handle && s.tsc_index == memo.tsc_index &&
                s.kind == memo.kind) {
                return true;
            }
        }
        const u64 h = L2KeyHash(memo.tic_handle, memo.tsc_index, memo.kind);
        const size_t first = static_cast<size_t>(h & (NumSets - 1)) * NumWays;
        const u32 tag = static_cast<u32>(h >> 32) | 1U;
        for (size_t way = 0; way < NumWays; ++way) {
            if (tags[first + way] != tag) {
                continue;
            }
            Entry& e = entries[first + way];
            if (e.memo.kind == memo.kind && e.memo.tic_handle == memo.tic_handle &&
                e.memo.tsc_index == memo.tsc_index && e.memo.channel == memo.channel &&
                e.memo.epoch == memo.epoch && e.memo.image_serial == memo.image_serial &&
                e.memo.view == memo.view && e.memo.sampler == memo.sampler) {
                if (e.last_l1_frame != current_frame) {
                    e.last_l1_frame = current_frame;
                }
                if (used[first + way] == 0) {
                    used[first + way] = 1;
                }
                RememberL1Seen(memo, current_frame);
                return true;
            }
        }
        return false;
    }

    void InsertOrUpdate(const Memo& memo, u64 current_frame, bool from_l1 = false) {
        if (memo.kind == 0) {
            return;
        }
        EnsureAllocated();
        const u64 h = L2KeyHash(memo.tic_handle, memo.tsc_index, memo.kind);
        const size_t set = static_cast<size_t>(h & (NumSets - 1));
        const size_t first = set * NumWays;
        const u32 tag = static_cast<u32>(h >> 32) | 1U;
        size_t target = NumWays;
        for (size_t way = 0; way < NumWays; ++way) {
            if (tags[first + way] != tag) {
                continue;
            }
            const Memo& existing = entries[first + way].memo;
            if (existing.kind == memo.kind && existing.tic_handle == memo.tic_handle &&
                existing.tsc_index == memo.tsc_index && existing.channel == memo.channel) {
                target = way;
                break;
            }
        }
        if (target == NumWays) {
            for (size_t way = 0; way < NumWays; ++way) {
                if (tags[first + way] == 0) {
                    target = way;
                    break;
                }
            }
        }
        if (target == NumWays) {
            u8 hand = clock_hand[set];
            for (size_t step = 0; step < NumWays * 2; ++step) {
                const size_t way = hand % NumWays;
                hand = static_cast<u8>((way + 1) % NumWays);
                if (used[first + way] == 0) {
                    target = way;
                    break;
                }
                used[first + way] = 0;
            }
            clock_hand[set] = hand;
            if (target == NumWays) {
                target = 0;
            }
        }
        tags[first + target] = tag;
        used[first + target] = 1;
        entries[first + target].memo = memo;
        entries[first + target].last_l1_frame = current_frame;
        if (from_l1) {
            RememberL1Seen(memo, current_frame);
        }
    }
};

template <typename Memory, typename Pages, typename T>
inline void ReadGuestEntry(const Memory& gpu_memory, Pages& pages, GPUVAddr gpu_addr,
                           T& out) noexcept {
    if (const u8* const host = pages.Get(gpu_memory, gpu_addr, sizeof(T), false)) {
        std::memcpy(&out, host, sizeof(T));
        return;
    }
    const bool one_page =
        (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
    if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
        std::memcpy(&out, host, sizeof(T));
        return;
    }
    gpu_memory.ReadBlockUnsafe(gpu_addr, &out, sizeof(T));
}

template <typename Cache>
[[nodiscard]] inline bool IsMemoStillValid(Cache& cache, const typename Cache::BindingMemo& memo) {
    if constexpr (requires {
                      { cache.MemoStillValid(memo) } -> std::convertible_to<bool>;
                  }) {
        return cache.MemoStillValid(memo);
    } else {
        (void)cache;
        (void)memo;
        return false;
    }
}

template <typename Cache, typename Memory>
[[nodiscard]] inline typename Cache::BindingMemo* FindInOverflow(
    Cache& cache, const Memory& gpu_memory, const typename Cache::BindingLookup& lookup,
    OverflowState<Cache>& overflow, u64 current_frame, u32 tic_handle, u32 tsc_index, u32 kind,
    bool skip_cross_frame_guard) {
    if (overflow.tags.empty()) {
        return nullptr;
    }
    const u64 h = L2KeyHash(tic_handle, tsc_index, kind);
    const size_t first = static_cast<size_t>(h & (OverflowState<Cache>::NumSets - 1)) *
                         OverflowState<Cache>::NumWays;
    const u32 tag = static_cast<u32>(h >> 32) | 1U;
    Tegra::Texture::TICEntry guest_tic{};
    Tegra::Texture::TSCEntry guest_tsc{};
    u32 read_tic_entry = ~0U;
    u32 read_tsc_entry = ~0U;
    for (size_t way = 0; way < OverflowState<Cache>::NumWays; ++way) {
        if (overflow.tags[first + way] != tag) {
            continue;
        }
        auto& entry = overflow.entries[first + way];
        auto& m = entry.memo;
        if (m.kind != kind || m.tic_handle != tic_handle || m.tsc_index != tsc_index ||
            m.channel != lookup.channel || m.tic_pool != lookup.tic_pool ||
            m.tic_limit != lookup.tic_limit ||
            (tsc_index != ~0U &&
             (m.tsc_pool != lookup.tsc_pool || m.tsc_limit != lookup.tsc_limit))) {
            continue;
        }
        if (!IsMemoStillValid(cache, m)) {
            continue;
        }
        const bool has_image = m.image && m.image != VideoCommon::NULL_IMAGE_ID;
        if (has_image && entry.last_l1_frame != current_frame) {
            const bool skip_guard = skip_cross_frame_guard || (kind & (1U << 14)) != 0;
            const auto& view_obj = cache.GetImageView(m.view);
            const auto* const backing = view_obj.BackingImage();
            if (!skip_guard && backing != nullptr && backing->last_write_frame != ~u64{0}) {
                return nullptr;
            }
            if (current_frame > entry.last_l1_frame &&
                current_frame - entry.last_l1_frame >= 64) {
                return nullptr;
            }
        }
        if (m.tic_entry != ~0U) {
            if (read_tic_entry != m.tic_entry) {
                ReadGuestEntry(gpu_memory, lookup.pages,
                               lookup.tic_pool +
                                   u64{m.tic_entry} * sizeof(Tegra::Texture::TICEntry),
                               guest_tic);
                read_tic_entry = m.tic_entry;
            }
            if (std::memcmp(&guest_tic, &m.tic, sizeof(Tegra::Texture::TICEntry)) != 0) {
                continue;
            }
        }
        if (tsc_index != ~0U && m.tsc_entry != ~0U) {
            if (read_tsc_entry != m.tsc_entry) {
                ReadGuestEntry(gpu_memory, lookup.pages,
                               lookup.tsc_pool +
                                   u64{m.tsc_entry} * sizeof(Tegra::Texture::TSCEntry),
                               guest_tsc);
                read_tsc_entry = m.tsc_entry;
            }
            if (std::memcmp(&guest_tsc, &m.tsc, sizeof(Tegra::Texture::TSCEntry)) != 0) {
                continue;
            }
        }
        overflow.used[first + way] = 1;
        return &m;
    }
    return nullptr;
}

template <typename Cache, typename Memory>
void Resolve(Cache& cache, const Memory* gpu_memory, u64 current_frame,
             OverflowState<Cache>* overflow, bool compute, bool blacklist,
             std::span<const u32> kinds, std::span<VideoCommon::ImageViewInOut> views,
             std::span<const u32> sampler_handles, std::span<VideoCommon::SamplerId> samplers,
             Counts& counts) {
    if (views.empty()) {
        return;
    }
    using Memo = typename Cache::BindingMemo;
    const auto lookup = cache.BeginBindingLookup(compute);
    if (overflow != nullptr) {
        overflow->NoteLimits(compute, lookup.channel, lookup.tic_limit, lookup.tsc_limit);
    }
    struct MemoSwitches {
        bool scratch;
        bool batch;
        bool partial;
    };
    static const MemoSwitches switches{
        .scratch = ScratchEnabled(),
        .batch = [] {
            const char* const value = VulkanSwitch("EDEN_NX2_BINDMEMO_BATCH");
            return value == nullptr || value[0] != '0';
        }(),
        .partial = [] {
            const char* const value = VulkanSwitch("EDEN_NX2_BINDMEMO_PARTIAL");
            return value == nullptr || value[0] != '0';
        }(),
    };
    struct Miss {
        u32 view;
        u32 sampler;
    };
    struct Scratch {
        std::vector<Memo*> memos;
        std::vector<Miss> misses;
        std::vector<Miss> stale_hits;
        std::vector<VideoCommon::ImageViewInOut> subset;
        std::vector<typename Cache::BindingProbe> probes;
        std::vector<u32> tsc_indices;
        std::vector<u32> canon_kinds;
        std::vector<u32> l1_to_insert;
        bool taken = false;
    };
    static thread_local Scratch scratch;
    std::unique_ptr<Scratch> local;
    const bool use_scratch = switches.scratch && !scratch.taken;
    if (!use_scratch) {
        local = std::make_unique<Scratch>();
    }
    Scratch& lists = use_scratch ? scratch : *local;
    lists.taken = true;
    struct Release {
        Scratch& lists;
        ~Release() {
            lists.taken = false;
        }
    } release{lists};
    std::vector<Memo*>& memos = lists.memos;
    memos.resize(views.size());
    std::vector<Miss>& misses = lists.misses;
    misses.clear();
    bool aliased_hit = false;
    const bool do_batch = switches.batch && views.size() > 2;
    std::vector<typename Cache::BindingProbe>& probes = lists.probes;
    std::vector<u32>& tsc_indices = lists.tsc_indices;
    tsc_indices.resize(views.size());
    std::vector<u32>& canon_kinds = lists.canon_kinds;
    canon_kinds.resize(views.size());
    std::vector<u32>& l1_to_insert = lists.l1_to_insert;
    l1_to_insert.clear();
    u32 sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 ckind = CanonicalKind(kinds[index]);
        canon_kinds[index] = ckind;
        const bool is_texture = (ckind & Kind::ClassMask) == Kind::Texture;
        tsc_indices[index] = is_texture ? sampler_handles[sampler_index++] : ~0U;
    }
    if (do_batch) {
        probes.resize(views.size());
        for (size_t index = 0; index < views.size(); ++index) {
            if (index > 0 && views[index].index == views[index - 1].index &&
                tsc_indices[index] == tsc_indices[index - 1] &&
                canon_kinds[index] == canon_kinds[index - 1]) {
                probes[index] = probes[index - 1];
                continue;
            }
            probes[index] =
                cache.ProbeBinding(views[index].index, tsc_indices[index], canon_kinds[index]);
        }
        for (size_t index = 0; index < views.size(); ++index) {
            if (index > 0 && views[index].index == views[index - 1].index &&
                tsc_indices[index] == tsc_indices[index - 1] &&
                canon_kinds[index] == canon_kinds[index - 1]) {
                continue;
            }
            cache.PrefetchBinding(lookup, probes[index], views[index].index, tsc_indices[index]);
        }
    }
    sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 kind = canon_kinds[index];
        const bool is_texture = (kind & Kind::ClassMask) == Kind::Texture;
        if (index > 0 && memos[index - 1] != nullptr &&
            views[index].index == views[index - 1].index &&
            tsc_indices[index] == tsc_indices[index - 1] && kind == canon_kinds[index - 1] &&
            views[index].skip_cross_frame_guard == views[index - 1].skip_cross_frame_guard) {
            Memo* const prev_memo = memos[index - 1];
            memos[index] = prev_memo;
            views[index].id = prev_memo->view;
            if (views[index].blacklist) {
                cache.NoteBlacklistedBinding(prev_memo->image);
            }
            if (is_texture) {
                samplers[sampler_index++] = prev_memo->sampler;
            }
            ++counts.hits;
            continue;
        }
        Memo* memo =
            do_batch ? cache.FindBinding(lookup, views[index].index, tsc_indices[index], kind,
                                         probes[index], views[index].skip_cross_frame_guard)
                     : cache.FindBinding(lookup, views[index].index, tsc_indices[index], kind,
                                         views[index].skip_cross_frame_guard);
        if (memo != nullptr) {
            if (overflow != nullptr && !overflow->RefreshL1Hit(*memo, current_frame)) {
                l1_to_insert.push_back(static_cast<u32>(index));
            }
        } else if (overflow != nullptr && gpu_memory != nullptr) {
            memo = FindInOverflow(cache, *gpu_memory, lookup, *overflow, current_frame,
                                  views[index].index, tsc_indices[index], kind,
                                  views[index].skip_cross_frame_guard);
        }
        memos[index] = memo;
        if (memo != nullptr) {
            views[index].id = memo->view;
            aliased_hit |= memo->aliased != 0;
            if (views[index].blacklist) {
                cache.NoteBlacklistedBinding(memo->image);
            }
            if (is_texture) {
                samplers[sampler_index] = memo->sampler;
            }
            ++counts.hits;
        } else {
            misses.push_back({static_cast<u32>(index), is_texture ? sampler_index : ~0U});
            ++counts.misses;
        }
        if (is_texture) {
            ++sampler_index;
        }
    }
    if (misses.empty()) {
        if (overflow != nullptr && !l1_to_insert.empty()) {
            for (const u32 idx : l1_to_insert) {
                if (memos[idx] != nullptr) {
                    overflow->InsertOrUpdate(*memos[idx], current_frame, true);
                }
            }
        }
        return;
    }
    const bool had_hits = misses.size() != views.size();
    bool resolved = false;
    if (had_hits && switches.partial) {
        u32 prev_sampler_handle = ~0U;
        VideoCommon::SamplerId prev_sampler_id{};
        bool have_prev_sampler = false;
        for (const Miss& miss : misses) {
            if (miss.sampler != ~0U) {
                const u32 sh = sampler_handles[miss.sampler];
                if (have_prev_sampler && sh == prev_sampler_handle) {
                    samplers[miss.sampler] = prev_sampler_id;
                } else {
                    prev_sampler_handle = sh;
                    prev_sampler_id = cache.GetSamplerId(sh, compute);
                    have_prev_sampler = true;
                    samplers[miss.sampler] = prev_sampler_id;
                }
            }
        }
        std::vector<VideoCommon::ImageViewInOut>& subset = lists.subset;
        subset.clear();
        subset.reserve(misses.size());
        for (const Miss& miss : misses) {
            subset.push_back(views[miss.view]);
        }
        cache.FillImageViews(subset, compute, blacklist);
        for (size_t index = 0; index < misses.size(); ++index) {
            views[misses[index].view].id = subset[index].id;
        }
        if (cache.BindingClock() == lookup.clock && !aliased_hit) {
            resolved = true;
        } else {
            size_t total_resolved = misses.size();
            std::vector<Miss>& stale_hits = lists.stale_hits;
            for (u32 pass = 0; pass < 3; ++pass) {
                stale_hits.clear();
                u32 hit_sampler_index = 0;
                for (size_t index = 0; index < views.size(); ++index) {
                    const bool is_texture = (canon_kinds[index] & Kind::ClassMask) == Kind::Texture;
                    if (Memo* const memo = memos[index]) {
                        if (!IsMemoStillValid(cache, *memo)) {
                            memos[index] = nullptr;
                            stale_hits.push_back(
                                {static_cast<u32>(index), is_texture ? hit_sampler_index : ~0U});
                        }
                    }
                    if (is_texture) {
                        ++hit_sampler_index;
                    }
                }
                if (stale_hits.empty()) {
                    resolved = true;
                    break;
                }
                if (total_resolved + stale_hits.size() > views.size()) {
                    break;
                }
                const u64 retry_clock = cache.BindingClock();
                for (const Miss& stale : stale_hits) {
                    if (stale.sampler != ~0U) {
                        samplers[stale.sampler] =
                            cache.GetSamplerId(sampler_handles[stale.sampler], compute);
                    }
                }
                subset.clear();
                subset.reserve(stale_hits.size());
                for (const Miss& stale : stale_hits) {
                    subset.push_back(views[stale.view]);
                }
                cache.FillImageViews(subset, compute, blacklist);
                for (size_t index = 0; index < stale_hits.size(); ++index) {
                    views[stale_hits[index].view].id = subset[index].id;
                }
                total_resolved += stale_hits.size();
                if (cache.BindingClock() == retry_clock && !aliased_hit) {
                    resolved = true;
                    break;
                }
            }
        }
    }
    if (!resolved) {
        for (size_t index = 0; index < sampler_handles.size(); ++index) {
            samplers[index] = cache.GetSamplerId(sampler_handles[index], compute);
        }
        cache.FillImageViews(views, compute, blacklist);
        std::fill(memos.begin(), memos.end(), nullptr);
        l1_to_insert.clear();
        if (had_hits) {
            ++counts.full_resolves;
        }
    }
    if (overflow != nullptr && !l1_to_insert.empty()) {
        for (const u32 idx : l1_to_insert) {
            if (memos[idx] != nullptr) {
                overflow->InsertOrUpdate(*memos[idx], current_frame, true);
            }
        }
    }
    std::optional<typename Cache::BindingLookup> refreshed_lookup;
    if (cache.BindingClock() != lookup.clock) {
        refreshed_lookup.emplace(cache.BeginBindingLookup(compute));
    }
    const auto& rec_lookup = refreshed_lookup ? *refreshed_lookup : lookup;
    const u64 post_resolve_clock = rec_lookup.clock;
    sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 kind = canon_kinds[index];
        const u32 kind_class = kind & Kind::ClassMask;
        const bool is_texture = kind_class == Kind::Texture;
        const size_t texture_index = sampler_index;
        if (is_texture) {
            ++sampler_index;
        }
        if (memos[index] != nullptr || !views[index].id) {
            continue;
        }
        const VideoCommon::SamplerId samp_id =
            is_texture ? samplers[texture_index] : VideoCommon::SamplerId{};
        cache.RecordBinding(rec_lookup, views[index].index, tsc_indices[index], kind,
                            views[index].id, samp_id, nullptr, 0, false);
        if (overflow != nullptr) {
            Memo* const recorded =
                cache.FindBinding(rec_lookup, views[index].index, tsc_indices[index], kind, true);
            if (recorded != nullptr) {
                overflow->last_seen_epoch = recorded->epoch;
                overflow->InsertOrUpdate(*recorded, current_frame, true);
            } else if (gpu_memory != nullptr && overflow->last_seen_epoch != 0 &&
                       views[index].id == VideoCommon::NULL_IMAGE_VIEW_ID &&
                       cache.BindingClock() == post_resolve_clock) {
                const auto& tb = overflow->bounds[compute ? 1 : 0];
                const u32 header_index = Tegra::Texture::TextureHeaderIndex(
                    views[index].index, rec_lookup.tic_limit,
                    Tegra::Texture::UsesHeaderByteOffsetHandles(gpu_memory->UsesNativeTiling()));
                const bool tic_unbounded_out =
                    rec_lookup.tic_limit >= 0xfff80000U && header_index >= tb.tic_capacity;
                if (tic_unbounded_out) {
                    const u32 tsc = tsc_indices[index];
                    const bool tsc_none_or_out =
                        tsc == ~0U || tsc > rec_lookup.tsc_limit ||
                        (rec_lookup.tsc_limit >= 0xfff80000U && tsc >= tb.tsc_capacity);
                    Memo synth{};
                    synth.tic_handle = views[index].index;
                    synth.tsc_index = tsc;
                    synth.kind = kind;
                    synth.view = VideoCommon::NULL_IMAGE_VIEW_ID;
                    synth.sampler = samp_id;
                    synth.image = VideoCommon::NULL_IMAGE_ID;
                    synth.tic_limit = rec_lookup.tic_limit;
                    synth.tsc_limit = rec_lookup.tsc_limit;
                    synth.tic_pool = rec_lookup.tic_pool;
                    synth.tsc_pool = rec_lookup.tsc_pool;
                    synth.channel = rec_lookup.channel;
                    synth.tic_entry = ~0U;
                    synth.tsc_entry = tsc_none_or_out ? ~0U : tsc;
                    synth.epoch = overflow->last_seen_epoch;
                    synth.map_serial = rec_lookup.map_serial;
                    if (!tsc_none_or_out) {
                        ReadGuestEntry(*gpu_memory, rec_lookup.pages,
                                       rec_lookup.tsc_pool +
                                           u64{tsc} * sizeof(Tegra::Texture::TSCEntry),
                                       synth.tsc);
                    }
                    if (IsMemoStillValid(cache, synth)) {
                        overflow->InsertOrUpdate(synth, current_frame, false);
                    }
                }
            }
        }
    }
}

template <typename Cache>
void Resolve(Cache& cache, bool compute, bool blacklist, std::span<const u32> kinds,
             std::span<VideoCommon::ImageViewInOut> views, std::span<const u32> sampler_handles,
             std::span<VideoCommon::SamplerId> samplers, Counts& counts) {
    struct DummyMemory {
        const u8* GetPointer(GPUVAddr) const noexcept {
            return nullptr;
        }
        void ReadBlockUnsafe(GPUVAddr, void*, size_t) const noexcept {}
        bool UsesNativeTiling() const noexcept {
            return false;
        }
    };
    Resolve<Cache, DummyMemory>(cache, nullptr, 0, nullptr, compute, blacklist, kinds, views,
                                sampler_handles, samplers, counts);
}

} // namespace Metal::BindMemo
