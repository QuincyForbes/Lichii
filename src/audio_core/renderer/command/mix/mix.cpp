// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <span>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/renderer/command/mix/mix.h"
#include "common/fixed_point.h"

namespace AudioCore::Renderer {
/**
 * Mix input mix buffer into output mix buffer, with volume applied to the input.
 *
 * @tparam Q           - Number of bits for fixed point operations.
 * @param output       - Output mix buffer.
 * @param input        - Input mix buffer.
 * @param volume       - Volume applied to the input.
 * @param sample_count - Number of samples to process.
 */
template <size_t Q>
static void ApplyMix(std::span<s32> output, std::span<const s32> input, const f32 volume_,
                     const u32 sample_count) {
    const Common::FixedPoint<64 - Q, Q> volume{volume_};
    const s64 gain_raw = volume.to_raw();
    constexpr s64 frac_mask = Common::FixedPoint<64 - Q, Q>::fractional_mask;
    s32* const out = output.data();
    const s32* const in = input.data();
    if (out != in) {
        s32* __restrict dst = out;
        const s32* __restrict src = in;
        for (u32 i = 0; i < sample_count; i++) {
            const s64 scaled = s64{src[i]} * gain_raw;
            dst[i] = static_cast<s32>(s64{dst[i]} + ((scaled + ((scaled & frac_mask) >> 1)) >> Q));
        }
    } else {
        for (u32 i = 0; i < sample_count; i++) {
            const s64 scaled = s64{out[i]} * gain_raw;
            out[i] = static_cast<s32>(s64{out[i]} + ((scaled + ((scaled & frac_mask) >> 1)) >> Q));
        }
    }
}

void MixCommand::Dump([[maybe_unused]] const AudioRenderer::CommandListProcessor& processor,
                      std::string& string) {
    string += fmt::format("MixCommand");
    string += fmt::format("\n\tinput {:02X}", input_index);
    string += fmt::format("\n\toutput {:02X}", output_index);
    string += fmt::format("\n\tvolume {:.8f}", volume);
    string += "\n";
}

void MixCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    auto output{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffers.subspan(input_index * processor.sample_count,
                                             processor.sample_count)};

    // If volume is 0, nothing will be added to the output, so just skip.
    if (volume == 0.0f) {
        return;
    }

    switch (precision) {
    case 15:
        ApplyMix<15>(output, input, volume, processor.sample_count);
        break;

    case 23:
        ApplyMix<23>(output, input, volume, processor.sample_count);
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
