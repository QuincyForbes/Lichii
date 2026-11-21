// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/effect/delay.h"

namespace AudioCore::Renderer {

static void SetDelayEffectParameter(const DelayInfo::ParameterVersion1& params,
                                    DelayInfo::State& state) {
    auto channel_spread{params.channel_spread};
    state.feedback_gain = Common::FixedPoint<18, 14>::from_base(static_cast<s32>(DspSimd::MulFp<14>(
        params.feedback_gain.to_raw(), Common::FixedPoint<18, 14>{0.97998046875f}.to_raw())));
    state.delay_feedback_gain = Common::FixedPoint<18, 14>::from_base(static_cast<s32>(
        DspSimd::MulFp<14>(state.feedback_gain.to_raw(),
                           (Common::FixedPoint<18, 14>{1.0f} - channel_spread).to_raw())));
    if (params.channel_count == 4 || params.channel_count == 6) {
        channel_spread >>= 1;
    }
    state.delay_feedback_cross_gain = Common::FixedPoint<18, 14>::from_base(
        static_cast<s32>(DspSimd::MulFp<14>(channel_spread.to_raw(), state.feedback_gain.to_raw())));
    state.lowpass_feedback_gain =
        Common::FixedPoint<18, 14>::from_base(static_cast<s32>(DspSimd::MulFp<14>(
            params.lowpass_amount.to_raw(), Common::FixedPoint<18, 14>{0.949951171875f}.to_raw())));
    state.lowpass_gain = Common::FixedPoint<18, 14>{1.0f} - state.lowpass_feedback_gain;
}

static void InitializeDelayEffect(const DelayInfo::ParameterVersion1& params,
                                  DelayInfo::State& state,
                                  [[maybe_unused]] const CpuAddr workbuffer) {
    state = {};

    for (u32 channel = 0; channel < params.channel_count; channel++) {
        Common::FixedPoint<32, 32> sample_count_max{0.064f};
        sample_count_max *= params.sample_rate.to_int_floor() * params.delay_time_max;

        Common::FixedPoint<18, 14> delay_time{params.delay_time};
        delay_time *= params.sample_rate / 1000;
        Common::FixedPoint<32, 32> sample_count{delay_time};

        if (sample_count > sample_count_max) {
            sample_count = sample_count_max;
        }

        state.delay_lines[channel].sample_count_max = sample_count_max.to_int_floor();
        state.delay_lines[channel].sample_count = sample_count.to_int_floor();
        state.delay_lines[channel].buffer.resize(state.delay_lines[channel].sample_count, 0);
        if (state.delay_lines[channel].sample_count == 0) {
            state.delay_lines[channel].buffer.push_back(0);
        }
        state.delay_lines[channel].buffer_pos = 0;
        state.delay_lines[channel].decay_rate = 1.0f;
    }

    SetDelayEffectParameter(params, state);
}

template <size_t NumChannels>
static void ApplyDelay(const DelayInfo::ParameterVersion1& params, DelayInfo::State& state,
                       std::span<std::span<const s32>> inputs, std::span<std::span<s32>> outputs,
                       const u32 sample_count) {
    std::array<const s32*, NumChannels> in_ptrs{};
    std::array<s32*, NumChannels> out_ptrs{};
    for (size_t c = 0; c < NumChannels; ++c) {
        in_ptrs[c] = inputs[c].data();
        out_ptrs[c] = outputs[c].data();
    }

    const s64 fb_gain = state.feedback_gain.to_raw();
    const s64 dfb_gain = state.delay_feedback_gain.to_raw();
    const s64 cross_gain = state.delay_feedback_cross_gain.to_raw();
    const s64 param_fb_gain = params.feedback_gain.to_raw();
    const s64 in_gain = params.in_gain.to_raw();
    const s64 lp_gain = state.lowpass_gain.to_raw();
    const s64 lp_fb_gain = state.lowpass_feedback_gain.to_raw();
    const s64 dry_gain = params.dry_gain.to_raw();
    const s64 wet_gain = params.wet_gain.to_raw();

    for (u32 sample_index = 0; sample_index < sample_count; sample_index++) {
        std::array<s64, NumChannels> in_raw{};
        std::array<s64, NumChannels> d_raw{};
        for (size_t c = 0; c < NumChannels; ++c) {
            const s32 scaled_in =
                static_cast<s32>(static_cast<u32>(in_ptrs[c][sample_index]) * 64U);
            in_raw[c] = s64{scaled_in} << 14;
            d_raw[c] = state.delay_lines[c].Read().to_raw();
        }

        std::array<s64, NumChannels> fb{};
        if constexpr (NumChannels == 1) {
            fb[0] = DspSimd::MulFp<14>(d_raw[0], fb_gain);
        } else if constexpr (NumChannels == 2) {
            fb[0] = DspSimd::MulFp<14>(d_raw[0], dfb_gain) +
                    DspSimd::MulFp<14>(d_raw[1], cross_gain);
            fb[1] = DspSimd::MulFp<14>(d_raw[0], cross_gain) +
                    DspSimd::MulFp<14>(d_raw[1], dfb_gain);
        } else if constexpr (NumChannels == 4) {
            const s64 c0 = DspSimd::MulFp<14>(d_raw[0], cross_gain);
            const s64 c1 = DspSimd::MulFp<14>(d_raw[1], cross_gain);
            const s64 c2 = DspSimd::MulFp<14>(d_raw[2], cross_gain);
            const s64 c3 = DspSimd::MulFp<14>(d_raw[3], cross_gain);
            fb[0] = DspSimd::MulFp<14>(d_raw[0], dfb_gain) + c1 + c2;
            fb[1] = c0 + DspSimd::MulFp<14>(d_raw[1], dfb_gain) + c3;
            fb[2] = c0 + DspSimd::MulFp<14>(d_raw[2], dfb_gain) + c3;
            fb[3] = c1 + c2 + DspSimd::MulFp<14>(d_raw[3], dfb_gain);
        } else if constexpr (NumChannels == 6) {
            const s64 c0 = DspSimd::MulFp<14>(d_raw[0], cross_gain);
            const s64 c1 = DspSimd::MulFp<14>(d_raw[1], cross_gain);
            const s64 c2 = DspSimd::MulFp<14>(d_raw[2], cross_gain);
            const s64 c4 = DspSimd::MulFp<14>(d_raw[4], cross_gain);
            const s64 c5 = DspSimd::MulFp<14>(d_raw[5], cross_gain);
            fb[0] = DspSimd::MulFp<14>(d_raw[0], dfb_gain) + c2 + c4;
            fb[1] = DspSimd::MulFp<14>(d_raw[1], dfb_gain) + c2 + c5;
            fb[2] = c0 + c1 + DspSimd::MulFp<14>(d_raw[2], dfb_gain);
            fb[3] = DspSimd::MulFp<14>(d_raw[3], param_fb_gain);
            fb[4] = c0 + DspSimd::MulFp<14>(d_raw[4], dfb_gain) + c5;
            fb[5] = c1 + c4 + DspSimd::MulFp<14>(d_raw[5], dfb_gain);
        }

        for (size_t c = 0; c < NumChannels; ++c) {
            const s64 gained = DspSimd::MulFp<14>(in_raw[c], in_gain) + fb[c];
            const s64 lp = DspSimd::MulFp<14>(gained, lp_gain) +
                           DspSimd::MulFp<14>(state.lowpass_z[c].to_raw(), lp_fb_gain);
            state.lowpass_z[c] = Common::FixedPoint<50, 14>::from_base(lp);
            state.delay_lines[c].Write(state.lowpass_z[c]);

            const s64 out_fp =
                DspSimd::MulFp<14>(in_raw[c], dry_gain) + DspSimd::MulFp<14>(d_raw[c], wet_gain);
            out_ptrs[c][sample_index] = static_cast<s32>(out_fp >> 14) / 64;
        }
    }
}

static void ApplyDelayEffect(const DelayInfo::ParameterVersion1& params, DelayInfo::State& state,
                             const bool enabled, std::span<std::span<const s32>> inputs,
                             std::span<std::span<s32>> outputs, const u32 sample_count) {
    if (!IsChannelCountValid(params.channel_count)) {
        LOG_ERROR(Service_Audio, "Invalid delay channels {}", params.channel_count);
        return;
    }

    if (enabled) {
        switch (params.channel_count) {
        case 1:
            ApplyDelay<1>(params, state, inputs, outputs, sample_count);
            break;
        case 2:
            ApplyDelay<2>(params, state, inputs, outputs, sample_count);
            break;
        case 4:
            ApplyDelay<4>(params, state, inputs, outputs, sample_count);
            break;
        case 6:
            ApplyDelay<6>(params, state, inputs, outputs, sample_count);
            break;
        default:
            for (u32 channel = 0; channel < params.channel_count; channel++) {
                if (inputs[channel].data() != outputs[channel].data()) {
                    std::memcpy(outputs[channel].data(), inputs[channel].data(),
                                sample_count * sizeof(s32));
                }
            }
            break;
        }
    } else {
        for (u32 channel = 0; channel < params.channel_count; channel++) {
            if (inputs[channel].data() != outputs[channel].data()) {
                std::memcpy(outputs[channel].data(), inputs[channel].data(),
                            sample_count * sizeof(s32));
            }
        }
    }
}

void DelayCommand::Dump([[maybe_unused]] const AudioRenderer::CommandListProcessor& processor,
                        std::string& string) {
    string += fmt::format("DelayCommand\n\tenabled {} \n\tinputs: ", effect_enabled);
    for (u32 i = 0; i < MaxChannels; i++) {
        string += fmt::format("{:02X}, ", inputs[i]);
    }
    string += "\n\toutputs: ";
    for (u32 i = 0; i < MaxChannels; i++) {
        string += fmt::format("{:02X}, ", outputs[i]);
    }
    string += "\n";
}

void DelayCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    std::array<std::span<const s32>, MaxChannels> input_buffers{};
    std::array<std::span<s32>, MaxChannels> output_buffers{};

    for (u16 i = 0; i < parameter.channel_count; i++) {
        input_buffers[i] = processor.mix_buffers.subspan(inputs[i] * processor.sample_count,
                                                         processor.sample_count);
        output_buffers[i] = processor.mix_buffers.subspan(outputs[i] * processor.sample_count,
                                                          processor.sample_count);
    }

    auto state_{reinterpret_cast<DelayInfo::State*>(state)};

    if (effect_enabled) {
        if (parameter.state == DelayInfo::ParameterState::Updating) {
            SetDelayEffectParameter(parameter, *state_);
        } else if (parameter.state == DelayInfo::ParameterState::Initialized) {
            InitializeDelayEffect(parameter, *state_, workbuffer);
        }
    }
    ApplyDelayEffect(parameter, *state_, effect_enabled, input_buffers, output_buffers,
                     processor.sample_count);
}

bool DelayCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
