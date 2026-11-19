// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// MTLFXFrameInterpolator (iOS 26, macOS 26) on the renderer's own textures and command buffer,
// with the FSR 3 flow as its motion, as Vulkan's MetalFxFrameGen (present/metalfx_frame_gen.mm)
// does through VK_EXT_metal_objects. Plain C++ header; the implementation is Objective-C++.

#pragma once

#include <array>
#include <memory>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

class Device;
class Scheduler;

class FrameInterpolator {
public:
    FrameInterpolator();
    ~FrameInterpolator();

    /// Whether this GPU and OS run MetalFX frame interpolation (iOS 26, macOS 26).
    [[nodiscard]] static bool IsSupported(const Device& device);

    bool Prepare(const Device& device, TextureHandle color, TextureHandle destination,
                 std::array<u32, 2> size);

    /// Writes the frame halfway between `previous` and `current` (`size`, the same format) into
    /// `destination`. `motion` (RG16F at `motion_size`) holds per current pixel the offset in
    /// motion texels to where it was in the previous frame. False when it can't (logged once);
    /// the caller keeps FSR 3's own interpolation.
    bool Generate(const Device& device, Scheduler& scheduler, TextureHandle previous,
                  TextureHandle current, std::array<u32, 2> size, TextureHandle motion,
                  std::array<u32, 2> motion_size, TextureHandle destination, f32 delta_time,
                  bool reset);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
