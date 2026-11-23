// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/range_sets.h"

namespace VideoCommon {

/// Expands `[start, end)` forward within `[start, buffer_end)` across page-contiguous
/// GPU-modified intervals in `gpu_modified_ranges`, capped at `max_coalesce_bytes` from `start`.
/// Used on reactive synchronous buffer downloads so sequential CPU reads across a GPU-written
/// buffer pay one GPU pipeline drain per batch instead of one per 4 KiB page.
[[nodiscard]] inline DAddr CoalesceContiguousDownloadEnd(
    DAddr start, DAddr end, DAddr buffer_end, u64 page_size, u64 max_coalesce_bytes,
    const Common::RangeSet<DAddr>& gpu_modified_ranges) {
    if (start >= end || end >= buffer_end || page_size == 0 || max_coalesce_bytes == 0) {
        return (std::min)(end, buffer_end);
    }
    const u64 requested = end - start;
    if (requested >= max_coalesce_bytes) {
        return (std::min)(end, buffer_end);
    }
    const DAddr limit = (std::min)(buffer_end, start + max_coalesce_bytes);
    DAddr coalesced_end = (std::min)(Common::AlignUp(end, page_size), limit);
    gpu_modified_ranges.ForEachInRange(
        start, limit - start, [&](DAddr mod_begin, DAddr mod_end) {
            const DAddr mod_page_begin = Common::AlignDown(mod_begin, page_size);
            if (mod_page_begin > coalesced_end) {
                return;
            }
            const DAddr mod_page_end = (std::min)(Common::AlignUp(mod_end, page_size), limit);
            if (mod_page_end > coalesced_end) {
                coalesced_end = mod_page_end;
            }
        });
    return (std::min)(coalesced_end, buffer_end);
}

} // namespace VideoCommon
