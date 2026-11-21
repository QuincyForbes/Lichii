// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <span>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/mix/mix.h"

namespace AudioCore::Renderer {

void MixCommand::Dump([[maybe_unused]] const AudioRenderer::CommandListProcessor& processor,
                      std::string& string) {
    string += fmt::format("MixCommand\n\tinput {:02X}\n\toutput {:02X}\n\tvolume {:.8f}\n",
                          input_index, output_index, volume);
}

void MixCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    if (volume == 0.0f) {
        return;
    }

    auto output{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffers.subspan(input_index * processor.sample_count,
                                             processor.sample_count)};

    switch (precision) {
    case 15:
        DspSimd::ApplyGainMix<15>(output.data(), input.data(), volume, processor.sample_count);
        break;

    case 23:
        DspSimd::ApplyGainMix<23>(output.data(), input.data(), volume, processor.sample_count);
        break;

    default:
        LOG_ERROR(Service_Audio, "Invalid precision {}", precision);
        break;
    }
}

bool MixCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
