// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"

namespace Core {
class System;
} // namespace Core

namespace AudioCore::Sink {

class NullSinkStreamImpl final : public SinkStream {
public:
    explicit NullSinkStreamImpl(Core::System& system_, StreamType type_, u32 system_channels_,
                                std::string_view name_)
        : SinkStream{system_, type_} {
        system_channels = system_channels_;
        name = name_;
    }

    ~NullSinkStreamImpl() override = default;

    bool IsReady() const override {
        return true;
    }

    bool DiscardsSamples() const noexcept override {
        return true;
    }

    void Start(bool /*resume*/ = false) override {
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lk{pace_mutex};
        paused = false;
        if (next_deadline < now) {
            next_deadline = now;
        }
    }

    void Stop() override {
        {
            std::scoped_lock lk{pace_mutex};
            paused = true;
        }
        pace_cv.notify_all();
        SignalPause();
    }

    void AppendBuffer(SinkBuffer& buffer, std::span<s16>) override {
        if (type == StreamType::In) {
            return;
        }
        const u64 frames = buffer.frames != 0 ? buffer.frames : TargetSampleCount;
        AdvancePlayedSampleCount(frames);
        const auto duration =
            std::chrono::nanoseconds{(frames * 1'000'000'000ULL) / TargetSampleRate};
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lk{pace_mutex};
        if (next_deadline < now) {
            next_deadline = now;
        }
        next_deadline += duration;
    }

    bool TryAppendOutputBlock(SinkBuffer& buffer, std::span<s16> samples) override {
        return buffer.frames == TargetSampleCount && buffer.frames_played == 0 &&
               !buffer.consumed && samples.size() == TargetSampleCount * 2;
    }

    void WaitFreeSpace(std::stop_token stop_token) override {
        constexpr auto max_lead = std::chrono::nanoseconds{
            (u64{TargetSampleCount} * 3 * 1'000'000'000ULL) / TargetSampleRate};
        std::unique_lock lk{pace_mutex};
        if (paused.load(std::memory_order_relaxed)) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (next_deadline > now + max_lead) {
            pace_cv.wait_until(lk, stop_token, next_deadline - max_lead,
                               [this] { return paused.load(std::memory_order_relaxed); });
        }
    }

    std::vector<s16> ReleaseBuffer(u64) override {
        return {};
    }

private:
    std::mutex pace_mutex;
    std::condition_variable_any pace_cv;
    std::chrono::steady_clock::time_point next_deadline{};
};

/**
 * A no-op sink for when no audio out is wanted. Paces output streams to real time (from
 * Djihads80/Eglantine#910, per-stream) so AudioRenderer does not spin unthrottled.
 */
class NullSink final : public Sink {
public:
    explicit NullSink(std::string_view) {}
    ~NullSink() override = default;

    SinkStream* AcquireSinkStream(Core::System& system, u32 system_channels_,
                                  const std::string& name, StreamType type) override {
        std::scoped_lock lk{streams_mutex};
        system_channels = system_channels_;
        auto& stream = null_streams.emplace_back(
            std::make_unique<NullSinkStreamImpl>(system, type, system_channels_, name));
        stream->SetDeviceVolume(device_volume);
        return stream.get();
    }

    void CloseStream(SinkStream* stream) override {
        if (stream != nullptr) {
            stream->Stop();
        }
    }

    void CloseStreams() override {
        std::scoped_lock lk{streams_mutex};
        for (auto& stream : null_streams) {
            if (stream != nullptr) {
                stream->Stop();
            }
        }
    }

    f32 GetDeviceVolume() const override {
        return device_volume;
    }

    void SetDeviceVolume(f32 volume) override {
        std::scoped_lock lk{streams_mutex};
        device_volume = volume;
        for (auto& stream : null_streams) {
            stream->SetDeviceVolume(volume);
        }
    }

    void SetSystemVolume(f32 volume) override {
        std::scoped_lock lk{streams_mutex};
        for (auto& stream : null_streams) {
            stream->SetSystemVolume(volume);
        }
    }

private:
    mutable std::mutex streams_mutex;
    std::vector<SinkStreamPtr> null_streams{};
};

} // namespace AudioCore::Sink
