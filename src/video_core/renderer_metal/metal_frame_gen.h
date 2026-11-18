// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Frame generation on Metal, as Vulkan's FrameGen runs FSR 3 and MetalFX (present/frame_gen.cpp,
// present/fsr3_frame_gen.cpp): the presented frame is copied, a flow compute pass finds the
// motion from it to the one before (fsr3_optical_flow.comp through the SPIR-V -> MSL
// translator), and the frame halfway between (or at 1/n, 2/n ...) is made by
// fsr3_frame_interpolation.comp, or by MTLFXFrameInterpolator from the same flow. The pacer is
// Vulkan's (frame_gen_pacer.cpp). LSFG and NVIDIA's optical flow are not ported. Plain C++.

#pragma once

#ifdef HAS_LSFG

#include <array>
#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/frame_gen/lossless_dll.h"
#include "video_core/renderer_metal/metal_lsfg.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/renderer_vulkan/present/frame_gen_pacer.h"

namespace Metal {

class Device;
class FrameInterpolator;

class FrameGen {
public:
    explicit FrameGen(const Device& device, Scheduler& scheduler);
    ~FrameGen();

    /// Whether the settings ask for frame generation (FSR 3, MetalFX or LSFG).
    [[nodiscard]] static bool Wanted();

    /// The real frame was drawn into `frame` (`size` and `format` of the window's drawable):
    /// copies it, runs the flow, and plans the generated frames to show before it. Their number;
    /// 0 while the chain warms up, the pacer holds back, or a pass can't be made.
    [[nodiscard]] size_t Process(TextureHandle frame, std::array<u32, 2> size, u32 format,
                                 std::array<u32, 2> guest_size);

    /// Draws generated frame `generation` (0 based, of the count Process returned) into the
    /// texture returned, which stays until the next call; null when it can't be made.
    [[nodiscard]] TextureHandle Generate(size_t generation);

    /// The smoothed time between real frames in seconds, 0 while unknown.
    [[nodiscard]] f32 FrameInterval() const noexcept {
        return frame_interval;
    }

    /// Frames generated since the last call.
    [[nodiscard]] u64 TakeGenerated() noexcept;

    /// Which maker ran last, for the log and smoke test: "MetalFX", "FSR 3" or empty.
    [[nodiscard]] const char* LastMaker() const noexcept {
        return last_maker;
    }

private:
    struct Program {
        Object library;
        Object function;
        Object state;
        std::vector<ShaderTranslate::ResourceSlot> slots;
    };

    bool Prepare();
    /// LSFG: the DLL's shaders loaded; false (logged once) when it is not installed.
    bool PrepareLsfg();
    bool BuildProgram(Program& program, std::span<const u32> code);
    void Rebuild(std::array<u32, 2> size, u32 format, f32 flow_scale, bool lsfg_type);
    /// One of the two compute programs: `textures` by shader binding, the push constants, and the
    /// grid in 16 x 16 groups.
    bool Dispatch(const Program& program, std::span<const TextureHandle> textures,
                  std::span<const u8> constants, std::array<u32, 2> extent);

    const Device& device;
    Scheduler& scheduler;
    std::unique_ptr<FrameInterpolator> interpolator;
    std::unique_ptr<LsfgChain> lsfg;
    VideoCore::FrameGen::ShaderModules lsfg_modules;
    bool lsfg_loaded{};
    bool lsfg_failed{};
    bool built_lsfg{};

    Program flow;
    Program interpolation;
    Object sampler;
    bool prepared{};
    bool failed{};

    std::array<Object, 2> frames;
    Object motion;
    Object generated;
    std::array<u32, 2> built_size{};
    std::array<u32, 2> flow_size{};
    u32 built_format{};
    u32 parity{};
    bool has_history{};
    bool flow_valid{};

    std::chrono::steady_clock::time_point last_process{};
    f32 frame_interval{};
    Vulkan::FrameGenPacer pacer;
    Vulkan::FrameGenPlan plan{};
    std::array<u32, 2> peak_guest{};
    f32 flow_scale{1.0f};
    u64 frame_count{};
    u64 last_count{};
    u64 interpolator_count{};
    size_t last_generations{};
    u32 warm_streak{};
    bool unavailable{};
    bool interpolator_unsupported{};
    const char* last_maker{""};
    u64 generated_frames{};
};

} // namespace Metal

#endif // HAS_LSFG
