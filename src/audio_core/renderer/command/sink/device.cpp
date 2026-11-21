// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/sink/device.h"
#include "audio_core/sink/sink.h"

namespace AudioCore::Renderer {

void DeviceSinkCommand::Dump([[maybe_unused]] const AudioRenderer::CommandListProcessor& processor,
                             std::string& string) {
    string += fmt::format("DeviceSinkCommand\n\t{} session {} input_count {}\n\tinputs: ",
                          std::string_view(name), session_id, input_count);
    for (u32 i = 0; i < input_count; i++) {
        string += fmt::format("{:02X}, ", inputs[i]);
    }
    string += "\n";
}

void DeviceSinkCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    constexpr s32 min = (std::numeric_limits<s16>::min)();
    constexpr s32 max = (std::numeric_limits<s16>::max)();

    auto stream{processor.GetOutputSinkStream()};
    stream->SetSystemChannels(input_count);

    Sink::SinkBuffer out_buffer{
        .frames{TargetSampleCount},
        .frames_played{0},
        .tag{0},
        .consumed{false},
    };

    if (stream->DiscardsSamples()) {
        stream->AppendBuffer(out_buffer, {});
        if (stream->IsPaused()) {
            stream->Start();
        }
        return;
    }

    std::array<s16, TargetSampleCount * MaxChannels> samples;
    const u32 frames = static_cast<u32>(out_buffer.frames);
    if (input_count == 2) {
        const s32* left = &sample_buffer[inputs[0] * frames];
        const s32* right = &sample_buffer[inputs[1] * frames];
        DspSimd::SaturateInterleaveStereoS32ToS16(samples.data(), left, right, frames);
    } else if (input_count == 1) {
        DspSimd::SaturateS32ToS16(samples.data(), &sample_buffer[inputs[0] * frames], frames);
    } else {
        for (u32 channel = 0; channel < input_count; channel++) {
            const auto offset{inputs[channel] * frames};
            const s32* __restrict src = &sample_buffer[offset];
            for (u32 index = 0; index < frames; index++) {
                samples[index * input_count + channel] =
                    static_cast<s16>(std::clamp(src[index], min, max));
            }
        }
    }

    out_buffer.tag = reinterpret_cast<u64>(samples.data());
    stream->AppendBuffer(out_buffer, {samples.data(), out_buffer.frames * input_count});

    if (stream->IsPaused()) {
        stream->Start();
    }
}

bool DeviceSinkCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
