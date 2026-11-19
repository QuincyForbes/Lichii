// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include "common/common_types.h"
#include "common/settings_enums.h"

namespace VideoCommon::DlssMode {

using Quality = Settings::DlssQuality;

struct Extent {
    u32 width;
    u32 height;
};

struct Range {
    Extent min;
    Extent max;
};

[[nodiscard]] constexpr bool Fits(Extent render, Extent output, Quality mode, Range range) {
    return render.width != 0 && render.height != 0 && output.width != 0 && output.height != 0 &&
           render.width >= range.min.width && render.width <= range.max.width &&
           render.height >= range.min.height && render.height <= range.max.height &&
           (mode != Quality::Dlaa ||
            (render.width == output.width && render.height == output.height));
}

[[nodiscard]] constexpr Quality Preferred(Extent render, Extent output) {
    const float ratio =
        output.width ? static_cast<float>(render.width) / static_cast<float>(output.width) : 0.0f;
    if (ratio >= 0.99f) {
        return Quality::Dlaa;
    }
    if (ratio >= 0.65f) {
        return Quality::Quality;
    }
    if (ratio >= 0.57f) {
        return Quality::Balanced;
    }
    if (ratio >= 0.49f) {
        return Quality::Performance;
    }
    return Quality::UltraPerformance;
}

struct Selection {
    std::optional<Quality> mode;
    std::optional<Range> requested_range;
    bool fallback{};
};

template <typename Query>
[[nodiscard]] constexpr Selection Select(Quality requested, Extent render, Extent output,
                                         Query&& query) {
    std::optional<Range> requested_range;
    if (requested != Quality::Auto) {
        requested_range = query(requested);
        if (requested_range && Fits(render, output, requested, *requested_range)) {
            return {requested, requested_range, false};
        }
    }
    const auto preferred = Preferred(render, output);
    bool tried_preferred = false;
    for (const auto mode : {preferred, Quality::Quality, Quality::Balanced, Quality::Performance,
                            Quality::UltraPerformance, Quality::Dlaa}) {
        if (mode == requested || (tried_preferred && mode == preferred)) {
            continue;
        }
        if (mode == preferred) {
            tried_preferred = true;
        }
        if (const auto range = query(mode); range && Fits(render, output, mode, *range)) {
            return {mode, requested_range, requested != Quality::Auto};
        }
    }
    return {std::nullopt, requested_range, requested != Quality::Auto};
}

} // namespace VideoCommon::DlssMode
