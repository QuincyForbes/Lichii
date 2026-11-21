// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/mix/volume_ramp.h"

namespace AudioCore::Renderer {

void VolumeRampCommand::Dump(const AudioRenderer::CommandListProcessor& processor,
                             std::string& string) {
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
    string += fmt::format(
        "VolumeRampCommand\n\tinput {:02X}\n\toutput {:02X}\n\tvolume {:.8f}"
        "\n\tprev_volume {:.8f}\n\tramp {:.8f}\n",
        input_index, output_index, volume, prev_volume, ramp);
}

void VolumeRampCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
    if (input_index == output_index && prev_volume == 1.0f && ramp == 0.0f) {
        return;
    }

    auto output{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffers.subspan(input_index * processor.sample_count,
                                             processor.sample_count)};

    switch (precision) {
    case 15:
        DspSimd::ApplyGainRampReplace<15>(output.data(), input.data(), prev_volume, ramp,
                                          processor.sample_count);
        break;

    case 23:
        DspSimd::ApplyGainRampReplace<23>(output.data(), input.data(), prev_volume, ramp,
                                          processor.sample_count);
        break;

    default:
        LOG_ERROR(Service_Audio, "Invalid precision {}", precision);
        break;
    }
}

bool VolumeRampCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
