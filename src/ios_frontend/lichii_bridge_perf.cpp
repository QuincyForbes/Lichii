// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include "common/settings.h"
#include "core/core.h"
#include "core/perf_stats.h"
#include "ios_frontend/bridge_host.h"
#include "ios_frontend/lichii_bridge_perf.h"
#include "video_core/gpu.h"
#include "video_core/render_probe.h"
#include "video_core/shader_notify.h"

namespace {

double Finite(double value) {
    return std::isfinite(value) ? value : 0.0;
}

} // namespace

bool lichii_perf_snapshot(LichiiPerfSnapshot* out) {
    if (out == nullptr) {
        return false;
    }
    const auto state = IosFrontend::GetState();
    auto& system = IosFrontend::System();
    if ((state != LICHII_EMU_RUNNING && state != LICHII_EMU_PAUSED) || !system.IsPoweredOn()) {
        return false;
    }

    const auto results = system.GetAndResetPerfStats();
    *out = {};
    out->real_fps = Finite(results.average_game_fps);
    out->generated_fps = Finite(results.generated_fps);
    const double total = Finite(results.total_fps);
    out->displayed_fps = total > 0.0 ? total : out->real_fps + out->generated_fps;
    out->frametime_ms = Finite(results.frametime) * 1000.0;
    out->speed = Finite(results.emulation_speed);

    out->paused = state == LICHII_EMU_PAUSED;
    out->frame_gen_enabled = Settings::values.frame_gen.GetValue();
    out->game_dlss_network = Settings::values.game_dlss_network.GetValue();
    out->nvidia_dlss = Settings::values.nvidia_dlss.GetValue();
    out->frame_gen_type = static_cast<int32_t>(Settings::values.frame_gen_type.GetValue());
    out->frame_gen_multiplier = Settings::FrameGenMultiplier();
    out->scaling_filter = static_cast<int32_t>(Settings::values.scaling_filter.GetValue());
    out->gpu_accuracy = static_cast<int32_t>(Settings::values.current_gpu_accuracy);
    const auto& scaling = Settings::values.resolution_info;
    out->resolution_scale = scaling.up_factor;

    const u64 extents = VideoCore::Probe::frame_extents.load(std::memory_order_relaxed);
    out->guest_width = static_cast<u32>(extents & 0xFFFF);
    out->guest_height = static_cast<u32>((extents >> 16) & 0xFFFF);
    out->output_width = static_cast<u32>((extents >> 32) & 0xFFFF);
    out->output_height = static_cast<u32>((extents >> 48) & 0xFFFF);
    out->render_width = scaling.ScaleUp(out->guest_width);
    out->render_height = scaling.ScaleUp(out->guest_height);

    out->texture_cache_bytes = VideoCore::Probe::texture_cache_bytes.load(std::memory_order_relaxed);
    out->texture_cache_expected =
        VideoCore::Probe::texture_cache_expected.load(std::memory_order_relaxed);
    out->shaders_building = system.GPU().ShaderNotify().ShadersBuilding();
    return true;
}
