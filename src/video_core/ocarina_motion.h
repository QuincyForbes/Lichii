// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ocarina of Time's camera motion, shared by the Vulkan and Metal temporal upscale paths.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>

#include "common/common_types.h"

namespace VideoCommon {

using OcarinaMatrix = std::array<f32, 16>;

/// Reproject depth from the current clip space directly into the previous camera's clip space.
inline std::optional<OcarinaMatrix> OcarinaClipToPreviousClip(const OcarinaMatrix& current,
                                                              const OcarinaMatrix& previous) {
    std::array<std::array<f64, 8>, 4> rows{};
    for (size_t i = 0; i < 4; ++i) {
        std::copy_n(current.begin() + i * 4, 4, rows[i].begin());
        rows[i][i + 4] = 1.0;
    }
    for (size_t col = 0; col < 4; ++col) {
        const auto pivot = std::max_element(rows.begin() + col, rows.end(),
            [col](const auto& a, const auto& b) { return std::abs(a[col]) < std::abs(b[col]); });
        std::swap(rows[col], *pivot);
        const f64 divisor = rows[col][col];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-10) {
            return std::nullopt;
        }
        for (f64& value : rows[col]) {
            value /= divisor;
        }
        for (size_t row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const f64 scale = rows[row][col];
            for (size_t i = 0; i < 8; ++i) {
                rows[row][i] -= scale * rows[col][i];
            }
        }
    }
    OcarinaMatrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            f64 value{};
            for (size_t i = 0; i < 4; ++i) {
                value += previous[row * 4 + i] * rows[i][col + 4];
            }
            result[row * 4 + col] = static_cast<f32>(value);
        }
    }
    if (!std::ranges::all_of(result, [](f32 value) { return std::isfinite(value); })) {
        return std::nullopt;
    }
    return result;
}

} // namespace VideoCommon
