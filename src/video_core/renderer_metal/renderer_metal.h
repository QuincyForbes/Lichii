// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Plain C++: the Metal objects live in renderer_metal.mm (docs/nx2/metal-renderer.md).

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"

namespace Metal {

class RendererMetal final : public VideoCore::RendererBase {
public:
    explicit RendererMetal(Core::Frontend::EmuWindow& emu_window,
                           Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu,
                           std::unique_ptr<Core::Frontend::GraphicsContext> context);
    ~RendererMetal() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override;

    [[nodiscard]] std::string GetDeviceVendor() const override;

private:
    struct Impl;

    Tegra::GPU& gpu;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
