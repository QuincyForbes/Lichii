// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>

#include "common/common_types.h"

namespace Vulkan {

struct FramebufferSize {
    u32 width;
    u32 height;
    u32 layers;
};

// Draws without attachments take their size from the surface clip, which games leave at
// 32768x32768. Apple GPUs allow 16384, and Metal loses the device on a larger framebuffer.
constexpr FramebufferSize ClampFramebufferSize(FramebufferSize size, FramebufferSize max) {
    return {
        (std::max)(1U, (std::min)(size.width, max.width)),
        (std::max)(1U, (std::min)(size.height, max.height)),
        (std::max)(1U, (std::min)(size.layers, max.layers)),
    };
}

// A draw reaches no pixel past the end of its scissor, so a framebuffer without attachments
// need not be larger: a tile-based GPU prepares all the tiles of the size it is given. Along one
// axis, rounded up to a power of two so the scissors a game goes through share framebuffers.
constexpr u32 ScissorReach(s32 offset, u32 extent, u32 size) {
    const u32 limit = (std::max)(1U, size);
    const s64 end = std::clamp<s64>(s64{offset} + s64{extent}, 1, limit);
    return (std::min)(std::bit_ceil(static_cast<u32>(end)), limit);
}

} // namespace Vulkan
