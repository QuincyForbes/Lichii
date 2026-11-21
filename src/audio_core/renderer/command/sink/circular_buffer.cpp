// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/sink/circular_buffer.h"
#include "core/memory.h"

namespace AudioCore::Renderer {

void CircularBufferSinkCommand::Dump(
    [[maybe_unused]] const AudioRenderer::CommandListProcessor& processor, std::string& string) {
    string += fmt::format(
        "CircularBufferSinkCommand\n\tinput_count {} ring size {:04X} ring pos {:04X}\n\tinputs: ",
        input_count, size, pos);
    for (u32 i = 0; i < input_count; i++) {
        string += fmt::format("{:02X}, ", inputs[i]);
    }
    string += "\n";
}

void CircularBufferSinkCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    std::array<s16, TargetSampleCount * MaxChannels> output;
    for (u32 channel = 0; channel < input_count; channel++) {
        auto input{processor.mix_buffers.subspan(inputs[channel] * processor.sample_count,
                                                 processor.sample_count)};
        DspSimd::SaturateS32ToS16(output.data(), input.data(), processor.sample_count);

        processor.memory->WriteBlockUnsafe(address + pos, output.data(),
                                           processor.sample_count * sizeof(s16));
        pos += static_cast<u32>(processor.sample_count * sizeof(s16));
        if (pos >= size) {
            pos = 0;
        }
    }
}

bool CircularBufferSinkCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
