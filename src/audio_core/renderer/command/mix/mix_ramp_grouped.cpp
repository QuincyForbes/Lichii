// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/mix/mix_ramp.h"
#include "audio_core/renderer/command/mix/mix_ramp_grouped.h"

namespace AudioCore::Renderer {

void MixRampGroupedCommand::Dump(const AudioRenderer::CommandListProcessor& processor,
                                 std::string& string) {
    string += "MixRampGroupedCommand";
    for (u32 i = 0; i < buffer_count; i++) {
        const auto ramp{(volumes[i] - prev_volumes[i]) / static_cast<f32>(processor.sample_count)};
        string += fmt::format(
            "\n\t{}\n\t\tinput {:02X}\n\t\toutput {:02X}\n\t\tvolume {:.8f}"
            "\n\t\tprev_volume {:.8f}\n\t\tramp {:.8f}\n",
            i, inputs[i], outputs[i], volumes[i], prev_volumes[i], ramp);
    }
}

void MixRampGroupedCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    std::span<s32> prev_samples = {reinterpret_cast<s32*>(previous_samples), MaxMixBuffers};
    const f32 inv_sample_count =
        processor.sample_count > 0 ? 1.0f / static_cast<f32>(processor.sample_count) : 0.0f;

    s32 checked_input = -1;
    bool checked_input_is_zero = false;

    for (u32 i = 0; i < buffer_count; i++) {
        if (prev_volumes[i] == 0.0f && volumes[i] == 0.0f) {
            prev_samples[i] = 0;
            continue;
        }

        const s16 in_idx = inputs[i];
        const auto input{
            processor.mix_buffers.subspan(in_idx * processor.sample_count, processor.sample_count)};
        if (in_idx != checked_input) {
            checked_input = in_idx;
            checked_input_is_zero = DspSimd::IsAllZeroS32(input.data(), processor.sample_count);
        }
        if (checked_input_is_zero) {
            prev_samples[i] = 0;
            continue;
        }

        const auto ramp{(volumes[i] - prev_volumes[i]) * inv_sample_count};
        if (prev_volumes[i] == 0.0f && ramp == 0.0f) {
            prev_samples[i] = 0;
            continue;
        }

        const auto output{processor.mix_buffers.subspan(outputs[i] * processor.sample_count,
                                                        processor.sample_count)};
        s32 last_sample = 0;
        switch (precision) {
        case 15:
            last_sample =
                ApplyMixRamp<15>(output, input, prev_volumes[i], ramp, processor.sample_count);
            break;
        case 23:
            last_sample =
                ApplyMixRamp<23>(output, input, prev_volumes[i], ramp, processor.sample_count);
            break;
        default:
            LOG_ERROR(Service_Audio, "Invalid precision {}", precision);
            break;
        }
        prev_samples[i] = last_sample;
    }
}

bool MixRampGroupedCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
