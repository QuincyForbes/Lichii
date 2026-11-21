// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/resample/downmix_6ch_to_2ch.h"

namespace AudioCore::Renderer {

void DownMix6chTo2chCommand::Dump(
    [[maybe_unused]] const AudioRenderer::CommandListProcessor& processor, std::string& string) {
    string += fmt::format("DownMix6chTo2chCommand\n\tinputs:  ");
    for (u32 i = 0; i < MaxChannels; i++) {
        string += fmt::format("{:02X}, ", inputs[i]);
    }
    string += "\n\toutputs: ";
    for (u32 i = 0; i < MaxChannels; i++) {
        string += fmt::format("{:02X}, ", outputs[i]);
    }
    string += "\n";
}

void DownMix6chTo2chCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    const u32 count = processor.sample_count;
    const s32* in_fl = &processor.mix_buffers[inputs[0] * count];
    const s32* in_fr = &processor.mix_buffers[inputs[1] * count];
    const s32* in_c = &processor.mix_buffers[inputs[2] * count];
    const s32* in_lfe = &processor.mix_buffers[inputs[3] * count];
    const s32* in_bl = &processor.mix_buffers[inputs[4] * count];
    const s32* in_br = &processor.mix_buffers[inputs[5] * count];

    s32* out_fl = &processor.mix_buffers[outputs[0] * count];
    s32* out_fr = &processor.mix_buffers[outputs[1] * count];
    auto out_center{processor.mix_buffers.subspan(outputs[2] * count, count)};
    auto out_lfe{processor.mix_buffers.subspan(outputs[3] * count, count)};
    auto out_back_left{processor.mix_buffers.subspan(outputs[4] * count, count)};
    auto out_back_right{processor.mix_buffers.subspan(outputs[5] * count, count)};

    const s64 c0 = down_mix_coeff[0].to_raw();
    const s64 c1 = down_mix_coeff[1].to_raw();
    const s64 c2 = down_mix_coeff[2].to_raw();
    const s64 c3 = down_mix_coeff[3].to_raw();

    for (u32 i = 0; i < count; i++) {
        const s64 center_lfe = s64{in_c[i]} * c1 + s64{in_lfe[i]} * c2;
        const s64 left_raw = s64{in_fl[i]} * c0 + center_lfe + s64{in_bl[i]} * c3;
        const s64 right_raw = s64{in_fr[i]} * c0 + center_lfe + s64{in_br[i]} * c3;

        out_fl[i] = DspSimd::RoundFpToInt<16>(left_raw);
        out_fr[i] = DspSimd::RoundFpToInt<16>(right_raw);
    }

    std::memset(out_center.data(), 0, out_center.size_bytes());
    std::memset(out_lfe.data(), 0, out_lfe.size_bytes());
    std::memset(out_back_left.data(), 0, out_back_left.size_bytes());
    std::memset(out_back_right.data(), 0, out_back_right.size_bytes());
}

bool DownMix6chTo2chCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
