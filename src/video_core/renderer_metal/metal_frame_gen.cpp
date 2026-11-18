// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef HAS_LSFG

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host_shaders/fsr3_frame_interpolation_comp_spv.h"
#include "video_core/host_shaders/fsr3_optical_flow_comp_spv.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_frame_gen.h"
#include "video_core/renderer_metal/metal_frame_gen_metalfx.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"

namespace Metal {

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

constexpr u64 REQUIRED_FRAMES = 2;
constexpr u32 RECURRENCE_FRAMES = 2;
constexpr u32 GroupSize = 16;

/// fsr3_optical_flow.comp's push constants (Vulkan's FlowPushConstants).
struct FlowConstants {
    f32 inv_render_w;
    f32 inv_render_h;
    f32 render_w;
    f32 render_h;
    f32 search_radius;
    f32 block_size;
};

/// fsr3_frame_interpolation.comp's push constants (Vulkan's InterpPushConstants), padded as MSL
/// lays the block out (a float2 member after the scalars is 8-aligned: 36 bytes become 40).
struct InterpConstants {
    f32 inv_render_w;
    f32 inv_render_h;
    f32 render_w;
    f32 render_h;
    f32 blend_factor;
    f32 flow_scale;
    f32 motion_scale_x;
    f32 motion_scale_y;
    f32 has_backward;
    f32 padding;
};
static_assert(sizeof(InterpConstants) == 40);

/// Vulkan's ConfiguredFlowScale: the flow scale the interpolation shader gets.
f32 ConfiguredFlowScale(std::array<u32, 2> guest, std::array<u32, 2> presented) {
    if (!Settings::values.frame_gen_flow_scale_auto.GetValue()) {
        return static_cast<f32>(Settings::values.frame_gen_flow_scale.GetValue()) / 100.0f;
    }
    if (guest[0] == 0 || presented[0] == 0) {
        return 1.0f;
    }
    const f32 rendered_width =
        static_cast<f32>(guest[0]) * Settings::values.resolution_info.up_factor;
    const f32 ratio = rendered_width / static_cast<f32>(presented[0]);
    constexpr f32 FLOW_SCALE_STEPS = 20.0f;
    const f32 stepped = std::ceil(ratio * FLOW_SCALE_STEPS) / FLOW_SCALE_STEPS;
    return std::clamp(stepped, 0.25f, 1.0f);
}

} // Anonymous namespace

FrameGen::FrameGen(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, interpolator{std::make_unique<FrameInterpolator>()} {}

FrameGen::~FrameGen() = default;

bool FrameGen::Wanted() {
    return Settings::values.frame_gen.GetValue();
}

bool FrameGen::PrepareLsfg() {
    if (lsfg_loaded || lsfg_failed) {
        return lsfg_loaded;
    }
    const auto status = VideoCore::FrameGen::LoadShaderModules(lsfg_modules);
    if (status != VideoCore::FrameGen::LosslessStatus::Ok) {
        lsfg_failed = true;
        LOG_WARNING(Render_Metal, "LSFG unavailable: Lossless.dll status {} (no frames are "
                                  "generated)",
                    static_cast<u32>(status));
        return false;
    }
    LOG_INFO(Render_Metal, "LSFG shaders loaded: {} modules", lsfg_modules.size());
    lsfg_loaded = true;
    return true;
}

u64 FrameGen::TakeGenerated() noexcept {
    return std::exchange(generated_frames, 0);
}

bool FrameGen::BuildProgram(Program& program, std::span<const u32> code) {
    ShaderTranslate::Options options{};
    options.ios = TranslateForIos;
    options.apple_family = DeviceAppleFamily(device);
    const ShaderTranslate::Result msl =
        ShaderTranslate::SpirvToMsl(code, ShaderTranslate::Stage::Compute, options);
    std::string error = msl.error;
    if (!msl.source.empty()) {
        program.library = CreateLibrary(device, msl.source, error);
        program.function =
            program.library ? CreateFunction(program.library.Get(), msl.entry_point) : Object{};
        program.state = program.function
                            ? CreateComputePipelineState(device, program.function.Get(), error)
                            : Object{};
    }
    program.slots = msl.slots;
    if (!program.state) {
        LOG_WARNING(Render_Metal, "Metal: frame generation shader not built: {}", error);
        return false;
    }
    return true;
}

bool FrameGen::Prepare() {
    if (prepared || failed) {
        return !failed;
    }
    sampler = CreateLinearClampSampler(device);
    failed = !sampler || !BuildProgram(flow, FSR3_OPTICAL_FLOW_COMP_SPV) ||
             !BuildProgram(interpolation, FSR3_FRAME_INTERPOLATION_COMP_SPV);
    prepared = !failed;
    return prepared;
}

void FrameGen::Rebuild(std::array<u32, 2> size, u32 format, f32 scale, bool lsfg_type) {
    scheduler.Finish();
    built_size = size;
    built_format = format;
    flow_scale = scale;
    flow_size = {(std::max)(1U, size[0] / 2), (std::max)(1U, size[1] / 2)};
    lsfg.reset();
    built_lsfg = lsfg_type;
    for (Object& frame : frames) {
        if (lsfg_type) {
            break;
        }
        frame = CreateTexture(device, {.kind = TextureKind::e2D,
                                       .format = format,
                                       .width = size[0],
                                       .height = size[1]});
    }
    // R16G16F: per current pixel, the offset in flow texels to the previous frame.
    if (!lsfg_type) {
        motion = CreateTexture(device, {.kind = TextureKind::e2D,
                                        .format = Fmt::RG16Float,
                                        .width = flow_size[0],
                                        .height = flow_size[1],
                                        .shader_write = true});
    }
    generated = CreateTexture(device, {.kind = TextureKind::e2D,
                                       .format = format,
                                       .width = size[0],
                                       .height = size[1],
                                       .render_target = true,
                                       .shader_write = true});
    parity = 0;
    has_history = false;
    flow_valid = false;
    frame_count = 0;
    last_count = 0;
    warm_streak = 0;
    interpolator_count = 0;
    if (lsfg_type) {
        lsfg = std::make_unique<LsfgChain>(device, scheduler, lsfg_modules, size, format, scale);
        if (!lsfg->Valid()) {
            LOG_WARNING(Render_Metal, "LSFG: a program or image could not be made; no frames "
                                      "are generated");
            lsfg.reset();
            failed = true;
        }
        return;
    }
    if (!frames[0] || !frames[1] || !motion || !generated) {
        failed = true;
        LOG_WARNING(Render_Metal, "Metal: frame generation textures not made for {}x{}", size[0],
                    size[1]);
    }
}

bool FrameGen::Dispatch(const Program& program, std::span<const TextureHandle> textures,
                        std::span<const u8> constants, std::array<u32, 2> extent) {
    scheduler.BeginCompute(program.state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.set == ShaderTranslate::PushConstantSet) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, constants.data(),
                                    static_cast<u32>(constants.size()));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  slot.binding < textures.size() ? textures[slot.binding]
                                                                 : nullptr);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, sampler.Get());
            break;
        }
    }
    scheduler.Dispatch({(extent[0] + GroupSize - 1) / GroupSize,
                        (extent[1] + GroupSize - 1) / GroupSize, 1},
                       {GroupSize, GroupSize, 1});
    return true;
}

size_t FrameGen::Process(TextureHandle frame, std::array<u32, 2> size, u32 format,
                         std::array<u32, 2> guest_size) {
    if (!Wanted() || unavailable || frame == nullptr || size[0] == 0 || size[1] == 0) {
        if (has_history) {
            scheduler.Finish();
            has_history = false;
            flow_valid = false;
        }
        warm_streak = 0;
        plan = {};
        return 0;
    }
    const bool lsfg_type = Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::Lsfg;
    if (lsfg_type ? !PrepareLsfg() : !Prepare()) {
        unavailable = true;
        return 0;
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_process != std::chrono::steady_clock::time_point{}) {
        const f32 interval = std::chrono::duration<f32>(now - last_process).count();
        // A load or a pause would take seconds to forget: start again after one.
        frame_interval = interval > 0.25f || frame_interval == 0.0f
                             ? (std::min)(interval, 0.1f)
                             : frame_interval + (interval - frame_interval) * 0.25f;
    }
    last_process = now;
    peak_guest = {(std::max)(peak_guest[0], guest_size[0]), (std::max)(peak_guest[1], guest_size[1])};

    const f32 scale = ConfiguredFlowScale(peak_guest, size);
    // Only LSFG uses the flow scale (the chain's size); the other chains keep their history.
    if (!generated || built_size != size || built_format != format || built_lsfg != lsfg_type ||
        (lsfg_type && flow_scale != scale)) {
        Rebuild(size, format, scale, lsfg_type);
        if (failed) {
            unavailable = true;
            return 0;
        }
    }
    flow_scale = scale;

    // The pacer's plan for this frame (Vulkan's FrameGen::WantedGenerations).
    plan = pacer.Plan(Settings::FrameGenMaxGenerations());
    const u64 count = frame_count++;
    const bool warm = plan.warm && count + 1 >= REQUIRED_FRAMES;
    warm_streak = warm ? warm_streak + 1 : 0;
    const bool generating = warm && warm_streak >= RECURRENCE_FRAMES && plan.generations > 0;
    last_generations = plan.generations;

    if (lsfg_type) {
        // Vulkan's FrameGen::Process for LSFG: every frame is copied in, the shared passes run
        // once the chain is warm.
        scheduler.CopyTexture(frame, 0, 0, {}, {size[0], size[1], 1}, lsfg->Input(count), 0, 0,
                              {});
        if (warm) {
            lsfg->DispatchShared(count);
        }
        last_count = count;
        has_history = true;
        flow_valid = generating;
        return generating ? last_generations : 0;
    }
    // Vulkan's Fsr3FrameGen::Process without the NVIDIA hardware.
    const std::array<u32, 3> extent{size[0], size[1], 1};
    const auto copy_to = [&](Object& destination) {
        scheduler.CopyTexture(frame, 0, 0, {}, {extent[0], extent[1], 1}, destination.Get(), 0, 0,
                              {});
    };
    if (!has_history) {
        copy_to(frames[0]);
        copy_to(frames[1]);
        has_history = true;
        parity = 0;
        last_count = count;
        flow_valid = false;
        return 0;
    }
    if (!generating) {
        parity ^= 1U;
        copy_to(frames[parity]);
        last_count = count;
        flow_valid = false;
        return 0;
    }
    if (count == last_count) {
        return flow_valid ? last_generations : 0;
    }
    last_count = count;
    parity ^= 1U;
    copy_to(frames[parity]);
    const FlowConstants constants{
        .inv_render_w = 1.0f / static_cast<f32>(flow_size[0]),
        .inv_render_h = 1.0f / static_cast<f32>(flow_size[1]),
        .render_w = static_cast<f32>(flow_size[0]),
        .render_h = static_cast<f32>(flow_size[1]),
        .search_radius = 2.0f,
        .block_size = 2.0f,
    };
    const std::array<TextureHandle, 3> textures{frames[parity].Get(), frames[parity ^ 1].Get(),
                                                motion.Get()};
    flow_valid = Dispatch(flow, textures,
                          {reinterpret_cast<const u8*>(&constants), sizeof(constants)}, flow_size);
    return flow_valid ? last_generations : 0;
}

TextureHandle FrameGen::Generate(size_t generation) {
    if (!has_history || !flow_valid || failed || !generated) {
        return nullptr;
    }
    if (lsfg) {
        lsfg->DispatchGeneration(last_count, last_generations, generation, generated.Get(),
                                 built_size);
        last_maker = "LSFG";
        ++generated_frames;
        return generated.Get();
    }
    // MetalFX makes only the frame halfway between two: 3x and 4x stay with FSR 3.
    if (last_generations == 1 &&
        Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::MetalFx &&
        !interpolator_unsupported) {
        if (!FrameInterpolator::IsSupported(device)) {
            LOG_WARNING(Render_Metal, "MetalFX frame interpolation needs iOS 26 or macOS 26 and "
                                      "an Apple GPU that has it; using FSR 3 frame generation");
            interpolator_unsupported = true;
        } else {
            // The interpolator keeps the last pair: a skipped frame starts it again.
            const bool reset = last_count != interpolator_count + 1;
            interpolator_count = last_count;
            const f32 delta_time = frame_interval > 0.0f ? frame_interval : 1.0f / 30.0f;
            if (interpolator->Generate(device, scheduler, frames[parity ^ 1].Get(),
                                       frames[parity].Get(), built_size, motion.Get(), flow_size,
                                       generated.Get(), delta_time, reset)) {
                last_maker = "MetalFX";
                ++generated_frames;
                return generated.Get();
            }
            interpolator_unsupported = true;
        }
    }
    const f32 blend = static_cast<f32>(generation + 1) / static_cast<f32>(last_generations + 1);
    const InterpConstants constants{
        .inv_render_w = 1.0f / static_cast<f32>(built_size[0]),
        .inv_render_h = 1.0f / static_cast<f32>(built_size[1]),
        .render_w = static_cast<f32>(built_size[0]),
        .render_h = static_cast<f32>(built_size[1]),
        .blend_factor = blend,
        .flow_scale = flow_scale,
        // The flow is in its own texels (half the frame): to texture coordinates.
        .motion_scale_x = 1.0f / static_cast<f32>(flow_size[0]),
        .motion_scale_y = 1.0f / static_cast<f32>(flow_size[1]),
        .has_backward = 0.0f,
        .padding = 0.0f,
    };
    // Bindings: 0 current, 1 previous, 2 the flow, 3 the generated frame (storage), 4 the
    // backward flow (the same image: there is no hardware flow here).
    const std::array<TextureHandle, 5> textures{frames[parity].Get(), frames[parity ^ 1].Get(),
                                                motion.Get(), generated.Get(), motion.Get()};
    if (!Dispatch(interpolation, textures,
                  {reinterpret_cast<const u8*>(&constants), sizeof(constants)}, built_size)) {
        return nullptr;
    }
    last_maker = "FSR 3";
    ++generated_frames;
    return generated.Get();
}

} // namespace Metal

#endif // HAS_LSFG
