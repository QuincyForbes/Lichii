// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>

#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
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

struct Rates {
    double real_fps;
    double generated_fps;
    double frametime_ms;
    double speed;
};

// Rates from the difference to the previous reading of the never-reset counters. The SDL window's
// title timer resets the shared counters (GetAndResetPerfStats) every 2 s, which made every other
// reader's interval arbitrary.
Rates ReadRates(Core::System& system, bool paused) {
    using namespace std::chrono;
    constexpr auto MinInterval = milliseconds{250};
    static std::mutex mutex;
    static const Core::PerfStats* stats{};
    static std::optional<Core::PerfCounters> last;
    static Rates rates{};

    auto& perf = system.GetPerfStats();
    const auto now = perf.GetCounters(system.CoreTiming().GetGlobalTimeUs());
    std::scoped_lock lock{mutex};
    // A new game has new counters; time spent paused is not part of the rate.
    if (paused || stats != &perf || !last || now.game_frames < last->game_frames ||
        now.system_frames < last->system_frames) {
        stats = &perf;
        last = now;
        if (!paused) {
            rates = {};
        }
        return rates;
    }
    if (now.time - last->time < MinInterval) {
        return rates;
    }
    const double seconds = duration<double>(now.time - last->time).count();
    const u64 system_frames = now.system_frames - last->system_frames;
    rates.real_fps = static_cast<double>(now.game_frames - last->game_frames) / seconds;
    rates.generated_fps =
        static_cast<double>(now.generated_frames - last->generated_frames) / seconds;
    rates.frametime_ms =
        system_frames > 0
            ? duration<double, std::milli>(now.system_frametime - last->system_frametime).count() /
                  static_cast<double>(system_frames)
            : (rates.real_fps > 0.0 ? 1000.0 / rates.real_fps : 0.0);
    rates.speed = duration<double>(now.system_time_us - last->system_time_us).count() / seconds;
    last = now;
    return rates;
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

    *out = {};
    out->paused = state == LICHII_EMU_PAUSED;
    const Rates rates = ReadRates(system, out->paused);
    out->real_fps = Finite(rates.real_fps);
    out->generated_fps = Finite(rates.generated_fps);
    out->displayed_fps = out->real_fps + out->generated_fps;
    out->frametime_ms = Finite(rates.frametime_ms);
    out->speed = Finite(rates.speed);

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
