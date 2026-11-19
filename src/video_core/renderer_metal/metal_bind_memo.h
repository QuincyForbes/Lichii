// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Vulkan's binding memo (EDEN_NX2_BINDMEMO, renderer_vulkan/pipeline_helper.h) for the Metal
// rasterizer: a draw's or dispatch's texture and image bindings looked up in the texture cache's
// memo (TextureCache::FindBinding) instead of resolving every view and sampler again, the ones
// missed resolved as FillImageViews and GetSamplerId do and recorded. Metal binds the views it
// gets directly, so nothing is kept in a memo's payload.

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/texture_cache/texture_cache_base.h"

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

/// Resolves `views` (TIC handles in, view ids out) and `samplers` (one per texture binding, from
/// `sampler_handles`) through the memo, as FindImageBindings, ResolveMissedBindings and
/// RecordImageBindings do on Vulkan.
template <typename Cache>
void Resolve(Cache& cache, bool compute, bool blacklist, std::span<const u32> kinds,
             std::span<VideoCommon::ImageViewInOut> views, std::span<const u32> sampler_handles,
             std::span<VideoCommon::SamplerId> samplers, Counts& counts) {
    using Memo = typename Cache::BindingMemo;
    const auto lookup = cache.BeginBindingLookup(compute);
    struct Miss {
        u32 view;
        u32 sampler;
    };
    // EDEN_NX2_METAL_BINDMEMO_SCRATCH: the lists keep their storage (per thread) between calls;
    // a nested call (none is made) would find them taken and allocate its own.
    struct Scratch {
        std::vector<Memo*> memos;
        std::vector<Miss> misses;
        std::vector<Miss> stale_hits;
        std::vector<VideoCommon::ImageViewInOut> subset;
        std::vector<typename Cache::BindingProbe> probes;
        std::vector<u32> tsc_indices;
        bool taken = false;
    };
    static thread_local Scratch scratch;
    Scratch local;
    const bool use_scratch = ScratchEnabled() && !scratch.taken;
    Scratch& lists = use_scratch ? scratch : local;
    lists.taken = true;
    struct Release {
        Scratch& lists;
        ~Release() {
            lists.taken = false;
        }
    } release{lists};
    std::vector<Memo*>& memos = lists.memos;
    memos.assign(views.size(), nullptr);
    std::vector<Miss>& misses = lists.misses;
    misses.clear();
    bool aliased_hit = false;
    static const bool batch = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_BINDMEMO_BATCH");
        return value == nullptr || value[0] != '0';
    }();
    std::vector<typename Cache::BindingProbe>& probes = lists.probes;
    probes.clear();
    std::vector<u32>& tsc_indices = lists.tsc_indices;
    tsc_indices.assign(views.size(), 0U);
    u32 sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const bool is_texture = (kinds[index] & Kind::ClassMask) == Kind::Texture;
        tsc_indices[index] = is_texture ? sampler_handles[sampler_index++] : ~0U;
    }
    if (batch) {
        probes.resize(views.size());
        for (size_t index = 0; index < views.size(); ++index) {
            probes[index] = cache.ProbeBinding(views[index].index, tsc_indices[index], kinds[index]);
        }
        for (size_t index = 0; index < views.size(); ++index) {
            cache.PrefetchBinding(lookup, probes[index], views[index].index, tsc_indices[index]);
        }
    }
    sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 kind = kinds[index];
        const bool is_texture = (kind & Kind::ClassMask) == Kind::Texture;
        Memo* const memo =
            batch ? cache.FindBinding(lookup, views[index].index, tsc_indices[index], kind,
                                      probes[index])
                  : cache.FindBinding(lookup, views[index].index, tsc_indices[index], kind);
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
        return;
    }
    // EDEN_NX2_BINDMEMO_PARTIAL: only the missed bindings are resolved, unless one hit's image had
    // aliases or the texture cache changed while they were.
    static const bool partial = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_BINDMEMO_PARTIAL");
        return value == nullptr || value[0] != '0';
    }();
    bool resolved = false;
    if (misses.size() != views.size() && partial) {
        for (const Miss& miss : misses) {
            if (miss.sampler != ~0U) {
                samplers[miss.sampler] = cache.GetSamplerId(sampler_handles[miss.sampler], compute);
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
            std::vector<Miss>& stale_hits = lists.stale_hits;
            stale_hits.clear();
            u32 hit_sampler_index = 0;
            for (size_t index = 0; index < views.size(); ++index) {
                const bool is_texture = (kinds[index] & Kind::ClassMask) == Kind::Texture;
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
            } else if (stale_hits.size() + misses.size() < views.size()) {
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
                bool all_remaining_valid = true;
                if (cache.BindingClock() != retry_clock || aliased_hit) {
                    for (size_t index = 0; index < views.size(); ++index) {
                        if (memos[index] != nullptr && !IsMemoStillValid(cache, *memos[index])) {
                            all_remaining_valid = false;
                            break;
                        }
                    }
                }
                resolved = all_remaining_valid;
            }
        }
    }
    if (!resolved) {
        for (size_t index = 0; index < sampler_handles.size(); ++index) {
            samplers[index] = cache.GetSamplerId(sampler_handles[index], compute);
        }
        cache.FillImageViews(views, compute, blacklist);
        std::fill(memos.begin(), memos.end(), nullptr);
        ++counts.full_resolves;
    }
    // What was resolved is remembered (the texture cache refuses what changed since the lookup).
    sampler_index = 0;
    for (size_t index = 0; index < views.size(); ++index) {
        const u32 kind = kinds[index];
        const u32 kind_class = kind & Kind::ClassMask;
        const bool is_texture = kind_class == Kind::Texture;
        const size_t texture_index = sampler_index;
        if (is_texture) {
            ++sampler_index;
        }
        if (memos[index] != nullptr || !views[index].id) {
            continue;
        }
        const bool has_image = is_texture || kind_class == Kind::Image;
        const bool rescaled =
            has_image && cache.IsRescaling(cache.GetImageView(views[index].id));
        cache.RecordBinding(lookup, views[index].index, tsc_indices[index], kind, views[index].id,
                            is_texture ? samplers[texture_index] : VideoCommon::SamplerId{},
                            nullptr, 0, rescaled);
    }
}

} // namespace Metal::BindMemo
