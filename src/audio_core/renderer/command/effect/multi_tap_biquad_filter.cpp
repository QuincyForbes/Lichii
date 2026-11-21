// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/renderer/command/effect/biquad_filter.h"
#include "audio_core/renderer/command/effect/multi_tap_biquad_filter.h"

namespace AudioCore::Renderer {

void MultiTapBiquadFilterCommand::Dump(
    [[maybe_unused]] const AudioRenderer::CommandListProcessor& processor, std::string& string) {
    string += fmt::format(
        "MultiTapBiquadFilterCommand\n\tinput {:02X}\n\toutput {:02X}\n\tneeds_init ({}, {})\n",
        input, output, needs_init[0], needs_init[1]);
}

void MultiTapBiquadFilterCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    if (filter_tap_count > MaxBiquadFilters) {
        LOG_ERROR(Service_Audio, "Too many filter taps! {}", filter_tap_count);
        filter_tap_count = MaxBiquadFilters;
    }

    auto input_buffer{
        processor.mix_buffers.subspan(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{
        processor.mix_buffers.subspan(output * processor.sample_count, processor.sample_count)};

    if (filter_tap_count == 2) {
        auto state0{reinterpret_cast<VoiceState::BiquadFilterState*>(states[0])};
        auto state1{reinterpret_cast<VoiceState::BiquadFilterState*>(states[1])};
        if (needs_init[0]) {
            *state0 = {};
        }
        if (needs_init[1]) {
            *state1 = {};
        }
        if (use_float_coefficients) {
            ApplyDoubleBiquadFilterFloat2(output_buffer, input_buffer, biquads_float, *state0,
                                          *state1, processor.sample_count);
        } else {
            ApplyDoubleBiquadFilterFloat(output_buffer, input_buffer, biquads, *state0, *state1,
                                         processor.sample_count);
        }
        return;
    }

    for (u32 i = 0; i < filter_tap_count; i++) {
        auto state{reinterpret_cast<VoiceState::BiquadFilterState*>(states[i])};
        if (needs_init[i]) {
            *state = {};
        }

        const std::span<const s32> tap_input = i == 0 ? input_buffer : output_buffer;
        if (use_float_coefficients) {
            ApplyBiquadFilterFloat2(output_buffer, tap_input, biquads_float[i].numerator,
                                    biquads_float[i].denominator, *state, processor.sample_count);
        } else {
            ApplyBiquadFilterFloat(output_buffer, tap_input, biquads[i].b, biquads[i].a, *state,
                                   processor.sample_count);
        }
    }
}

bool MultiTapBiquadFilterCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
