// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

/// Images larger than this are never backed (title 1's sparse texture arrays reach 4.5G texels,
/// ~23 GB): VMA would quietly place them in system memory once VRAM is full.
constexpr u64 MaxBackedImageTexels = u64{1} << 28;

/// Whether an image no memory is left for gets an unbacked sparse image (reads return zero,
/// writes are dropped) instead of failing. Only one too large to back, or one whose caller cannot
/// do without an image: the texture cache can (TryCreateImage), and leaves the image out until
/// the collector has freed memory. An unbacked render target is drawn into, which AMD's drivers
/// do not survive, and the texture cache would take it for an image over unwritten sparse memory.
constexpr bool UsesUnbackedFallback(u64 texels, bool may_fail) {
    return texels > MaxBackedImageTexels || !may_fail;
}

/// Whether a rescaled image of `scaled_width` x `scaled_height` x `depth` x `layers` fits within
/// the device's 2D/framebuffer dimension limit and MaxBackedImageTexels. Above MaxBackedImageTexels,
/// ScaleUp would create an unbacked sparse image that drops writes and reads zero (e.g. Ocarina of
/// Time's 12288x12288 sun shadow map at 2x -> 24576x24576).
constexpr bool CanRescaleImageExtent(u64 scaled_width, u64 scaled_height, u64 depth, u64 layers,
                                     u64 max_dimension) {
    if (scaled_width == 0 || scaled_height == 0 || depth == 0 || layers == 0) {
        return false;
    }
    if (scaled_width > max_dimension || scaled_height > max_dimension) {
        return false;
    }
    const u64 plane = scaled_width * scaled_height;
    if (plane > MaxBackedImageTexels / depth) {
        return false;
    }
    return plane * depth <= MaxBackedImageTexels / layers;
}

} // namespace Vulkan
