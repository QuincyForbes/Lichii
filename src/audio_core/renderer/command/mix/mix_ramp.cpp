// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/mix/mix_ramp.h"
#include "common/logging.h"

namespace AudioCore::Renderer {

template <size_t Q>
s32 ApplyMixRamp(std::span<s32> output, std::span<const s32> input, const f32 volume_,
                 const f32 ramp_, const u32 sample_count) {
    return DspSimd::ApplyGainRampMix<Q>(output.data(), input.data(), volume_, ramp_, sample_count);
}

template s32 ApplyMixRamp<15>(std::span<s32>, std::span<const s32>, f32, f32, u32);
template s32 ApplyMixRamp<23>(std::span<s32>, std::span<const s32>, f32, f32, u32);

void MixRampCommand::Dump(const AudioRenderer::CommandListProcessor& processor,
                          std::string& string) {
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
    string += fmt::format(
        "MixRampCommand\n\tinput {:02X}\n\toutput {:02X}\n\tvolume {:.8f}\n\tprev_volume {:.8f}"
        "\n\tramp {:.8f}\n",
        input_index, output_index, volume, prev_volume, ramp);
}

void MixRampCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    auto prev_sample_ptr{reinterpret_cast<s32*>(previous_sample)};
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};

    if (prev_volume == 0.0f && ramp == 0.0f) {
        *prev_sample_ptr = 0;
        return;
    }

    auto output{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffers.subspan(input_index * processor.sample_count,
                                             processor.sample_count)};

    switch (precision) {
    case 15:
        *prev_sample_ptr =
            ApplyMixRamp<15>(output, input, prev_volume, ramp, processor.sample_count);
        break;

    case 23:
        *prev_sample_ptr =
            ApplyMixRamp<23>(output, input, prev_volume, ramp, processor.sample_count);
        break;

    default:
        LOG_ERROR(Service_Audio, "Invalid precision {}", precision);
        break;
    }
}

bool MixRampCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
