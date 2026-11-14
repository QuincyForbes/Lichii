// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Objective-C++ only (renderer_metal/*.mm): Object <-> Metal protocol types.

#pragma once

#import <Metal/Metal.h>

#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

/// Retains `object` for C++ (nil gives an empty Object).
[[nodiscard]] inline Object Retain(id object) {
    return Object{object != nil ? (__bridge_retained void*)object : nullptr};
}

[[nodiscard]] inline id<MTLTexture> AsTexture(const void* handle) {
    return (__bridge id<MTLTexture>)handle;
}

[[nodiscard]] inline id<MTLBuffer> AsBuffer(const void* handle) {
    return (__bridge id<MTLBuffer>)handle;
}

[[nodiscard]] inline id<MTLSamplerState> AsSampler(const void* handle) {
    return (__bridge id<MTLSamplerState>)handle;
}

} // namespace Metal
