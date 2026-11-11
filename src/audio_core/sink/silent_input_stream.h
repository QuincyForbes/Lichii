// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>
#include <span>
#include <vector>

#include "audio_core/common/common.h"
#include "audio_core/sink/sink_stream.h"
#include "core/core.h"
#include "core/core_timing.h"

namespace AudioCore::Sink {

/**
 * Guest capture stream that hears silence, used when the host microphone cannot be opened or
 * started. Console hardware always has a microphone, so games do not handle a failed capture
 * open. Silence is synthesized on demand from elapsed CoreTiming without scheduling a periodic
 * timer or copying zeros through the sample ring buffer.
 */
class SilentInputStream final : public SinkStream {
public:
    SilentInputStream(Core::System& system_, u32 system_channels_)
        : SinkStream{system_, StreamType::In} {
        system_channels = system_channels_;
    }

    ~SilentInputStream() override {
        Stop();
    }

    bool IsReady() const override {
        return !failed;
    }

    void Start(bool = false) override {
        if (!IsReady() || !paused) {
            return;
        }
        std::scoped_lock lk{state_mutex};
        paused = false;
        last_time_ns = system.CoreTiming().GetGlobalTimeNs();
        remainder_ns = {};
    }

    void Stop() override {
        if (paused) {
            return;
        }
        {
            std::scoped_lock lk{state_mutex};
            AdvanceSilenceLocked();
        }
        SignalPause();
    }

    std::size_t DrainInputSamples(std::span<s16> out_samples, u64 max_frames) override {
        if (system_channels == 0 || max_frames == 0 || out_samples.empty()) {
            return 0;
        }
        std::scoped_lock lk{state_mutex};
        AdvanceSilenceLocked();
        const u64 fit_frames = out_samples.size() / system_channels;
        const u64 frames = std::min<u64>({buffered_frames, max_frames, fit_frames});
        buffered_frames -= frames;
        std::fill_n(out_samples.data(), static_cast<std::size_t>(frames * system_channels), s16{0});
        return static_cast<std::size_t>(frames);
    }

    void DiscardInputSamples() override {
        std::scoped_lock lk{state_mutex};
        AdvanceSilenceLocked();
        buffered_frames = 0;
    }

    std::vector<s16> ReleaseBuffer(u64 num_samples) override {
        {
            std::scoped_lock lk{state_mutex};
            AdvanceSilenceLocked();
            if (system_channels != 0) {
                const u64 frames = num_samples / system_channels;
                buffered_frames -= std::min(buffered_frames, frames);
            }
        }
        return std::vector<s16>(static_cast<std::size_t>(num_samples), 0);
    }

    u64 GetExpectedPlayedSampleCount() override {
        std::scoped_lock lk{state_mutex};
        AdvanceSilenceLocked();
        return total_frames + TargetSampleCount * 5;
    }

private:
    static constexpr u64 MaxBufferedFrames = 0x8000;

    void AdvanceSilenceLocked() {
        if (paused || system.IsPaused() || system.IsShuttingDown()) {
            return;
        }
        const auto now = system.CoreTiming().GetGlobalTimeNs();
        if (now > last_time_ns) {
            const auto elapsed_ns = (now - last_time_ns) + remainder_ns;
            const u64 generated =
                (static_cast<u64>(elapsed_ns.count()) * TargetSampleRate) / 1'000'000'000ULL;
            remainder_ns = elapsed_ns - std::chrono::nanoseconds{
                                            (generated * 1'000'000'000ULL) / TargetSampleRate};
            last_time_ns = now;
            buffered_frames = std::min<u64>(buffered_frames + generated, MaxBufferedFrames);
            total_frames += generated;
        } else {
            last_time_ns = now;
        }
    }

    std::mutex state_mutex;
    std::chrono::nanoseconds last_time_ns{};
    std::chrono::nanoseconds remainder_ns{};
    u64 buffered_frames{};
    u64 total_frames{};
};

} // namespace AudioCore::Sink
