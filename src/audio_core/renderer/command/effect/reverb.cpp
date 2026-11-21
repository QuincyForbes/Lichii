// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <numbers>
#include <ranges>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/effect/reverb.h"

namespace AudioCore::Renderer {

constexpr std::array<f32, ReverbInfo::MaxDelayLines> FdnMaxDelayLineTimes = {
    53.9532470703125f,
    79.19256591796875f,
    116.23876953125f,
    170.61529541015625f,
};

constexpr std::array<f32, ReverbInfo::MaxDelayLines> DecayMaxDelayLineTimes = {
    7.0f,
    9.0f,
    13.0f,
    17.0f,
};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayTaps + 1>, ReverbInfo::NumEarlyModes>
    EarlyDelayTimes = {
        {{0.000000f, 3.500000f, 2.799988f, 3.899963f, 2.699951f, 13.399963f, 7.899963f, 8.399963f,
          9.899963f, 12.000000f, 12.500000f},
         {0.000000f, 11.799988f, 5.500000f, 11.199951f, 10.399963f, 38.099976f, 22.199951f,
          29.599976f, 21.199951f, 24.799988f, 40.000000f},
         {0.000000f, 41.500000f, 20.500000f, 41.299988f, 0.000000f, 29.500000f, 33.799988f,
          45.199951f, 46.799988f, 0.000000f, 50.000000f},
         {33.099976f, 43.299988f, 22.799988f, 37.899963f, 14.899963f, 35.299988f, 17.899963f,
          34.199951f, 0.000000f, 43.299988f, 50.000000f},
         {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
          0.000000f, 0.000000f, 0.000000f}},
};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayTaps>, ReverbInfo::NumEarlyModes>
    EarlyDelayGains = {{
        {0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.699951f, 0.679993f,
         0.679993f, 0.679993f},
        {0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.699951f, 0.679993f, 0.679993f, 0.679993f,
         0.679993f, 0.679993f},
        {0.500000f, 0.699951f, 0.699951f, 0.679993f, 0.500000f, 0.679993f, 0.679993f, 0.699951f,
         0.679993f, 0.000000f},
        {0.929993f, 0.919983f, 0.869995f, 0.859985f, 0.939941f, 0.809998f, 0.799988f, 0.769958f,
         0.759949f, 0.649963f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
         0.000000f, 0.000000f},
    }};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayLines>, ReverbInfo::NumLateModes>
    FdnDelayTimes = {{
        {53.953247f, 79.192566f, 116.238770f, 130.615295f},
        {53.953247f, 79.192566f, 116.238770f, 170.615295f},
        {5.000000f, 10.000000f, 5.000000f, 10.000000f},
        {47.029968f, 71.000000f, 103.000000f, 170.000000f},
        {53.953247f, 79.192566f, 116.238770f, 170.615295f},
    }};

constexpr std::array<std::array<f32, ReverbInfo::MaxDelayLines>, ReverbInfo::NumLateModes>
    DecayDelayTimes = {{
        {7.000000f, 9.000000f, 13.000000f, 17.000000f},
        {7.000000f, 9.000000f, 13.000000f, 17.000000f},
        {1.000000f, 1.000000f, 1.000000f, 1.000000f},
        {7.000000f, 7.000000f, 13.000000f, 9.000000f},
        {7.000000f, 9.000000f, 13.000000f, 17.000000f},
    }};

/**
 * Update the ReverbInfo state according to the given parameters.
 *
 * @param params - Input parameters to update the state.
 * @param state  - State to be updated.
 */
static void UpdateReverbEffectParameter(const ReverbInfo::ParameterVersion2& params,
                                        ReverbInfo::State& state) {
    const auto pow_10 = [](f32 val) -> f32 {
        return (val >= 0.0f) ? 1.0f : (val <= -5.3f) ? 0.0f : std::pow(10.0f, val);
    };

    const auto cos = [](f32 degrees) -> f32 {
        return std::cos(degrees * std::numbers::pi_v<f32> / 180.0f);
    };

    static bool unk_initialized{false};
    static Common::FixedPoint<50, 14> unk_value{};

    auto sample_rate = Common::FixedPoint<50, 14>::from_base(params.sample_rate);
    if (sample_rate.to_float() < 1.0f) sample_rate = Common::FixedPoint<50, 14>(1.0f);

    auto pre_delay_time = Common::FixedPoint<50, 14>::from_base(params.pre_delay);
    if (pre_delay_time.to_float() < 0.0f) pre_delay_time = Common::FixedPoint<50, 14>(0.0f);

    for (u32 i = 0; i < ReverbInfo::MaxDelayTaps; i++) {
        int target_delay = ((pre_delay_time + EarlyDelayTimes[params.early_mode][i]) * sample_rate).to_int();
        if (target_delay < 0) target_delay = 0;
        if (target_delay > state.pre_delay_line.sample_count_max) target_delay = state.pre_delay_line.sample_count_max;

        int old_delay = state.early_delay_times[i] - 1;
        int smooth_delay = old_delay + (target_delay - old_delay) / 4;
        state.early_delay_times[i] = smooth_delay + 1;

        auto target_gain = Common::FixedPoint<50, 14>::from_base(params.early_gain) * EarlyDelayGains[params.early_mode][i];
        if (target_gain.to_float() < 0.0f) target_gain = Common::FixedPoint<50, 14>(0.0f);
        if (target_gain.to_float() > 1.0f) target_gain = Common::FixedPoint<50, 14>(1.0f);

        state.early_gains[i] = state.early_gains[i] + (target_gain - state.early_gains[i]) / 4;
    }

    if (params.channel_count == 2) {
        state.early_gains[4] *= Common::FixedPoint<50, 14>(0.5f);
        state.early_gains[5] *= Common::FixedPoint<50, 14>(0.5f);
    }

    int target_pre_time = ((pre_delay_time + EarlyDelayTimes[params.early_mode][10]) * sample_rate).to_int();
    if (target_pre_time < 0) target_pre_time = 0;
    if (target_pre_time > state.pre_delay_line.sample_count_max) target_pre_time = state.pre_delay_line.sample_count_max;

    int old_pre_time = state.pre_delay_time;
    state.pre_delay_time = old_pre_time + (target_pre_time - old_pre_time) / 4;

    if (!unk_initialized) {
        unk_value = cos((1280.0f / sample_rate).to_float());
        unk_initialized = true;
    }

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
        int target_fdn_delay = (FdnDelayTimes[params.late_mode][i] * sample_rate).to_int();
        if (target_fdn_delay < 0) target_fdn_delay = 0;
        if (target_fdn_delay > state.fdn_delay_lines[i].sample_count_max) target_fdn_delay = state.fdn_delay_lines[i].sample_count_max;

        int old_fdn = state.fdn_delay_lines[i].sample_count;
        state.fdn_delay_lines[i].sample_count = old_fdn + (target_fdn_delay - old_fdn) / 4;
        state.fdn_delay_lines[i].buffer_end = &state.fdn_delay_lines[i].buffer[state.fdn_delay_lines[i].sample_count - 1];

        int target_decay_delay = (DecayDelayTimes[params.late_mode][i] * sample_rate).to_int();
        if (target_decay_delay < 0) target_decay_delay = 0;
        if (target_decay_delay > state.decay_delay_lines[i].sample_count_max) target_decay_delay = state.decay_delay_lines[i].sample_count_max;

        int old_decay = state.decay_delay_lines[i].sample_count;
        state.decay_delay_lines[i].sample_count = old_decay + (target_decay_delay - old_decay) / 4;
        state.decay_delay_lines[i].buffer_end = &state.decay_delay_lines[i].buffer[state.decay_delay_lines[i].sample_count - 1];

        auto colouration = Common::FixedPoint<50, 14>::from_base(params.colouration);
        if (colouration.to_float() < 0.0f) colouration = Common::FixedPoint<50, 14>(0.0f);
        if (colouration.to_float() > 1.0f) colouration = Common::FixedPoint<50, 14>(1.0f);

        state.decay_delay_lines[i].decay = state.decay_delay_lines[i].decay + (0.5999755859375f * (1.0f - colouration) - state.decay_delay_lines[i].decay) / 4;

        auto decay_time_fp = Common::FixedPoint<50, 14>::from_base(params.decay_time);
        if (decay_time_fp.to_float() <= 0.0f) decay_time_fp = Common::FixedPoint<50, 14>(0.0001f);

        auto a = (Common::FixedPoint<50, 14>(state.fdn_delay_lines[i].sample_count_max) + state.decay_delay_lines[i].sample_count_max) * -3;
        auto b = a / (decay_time_fp * sample_rate);

        auto hf_decay_ratio = Common::FixedPoint<50, 14>::from_base(params.high_freq_decay_ratio);
        if (hf_decay_ratio.to_float() < 0.001f) hf_decay_ratio = Common::FixedPoint<50, 14>(0.001f);
        if (hf_decay_ratio.to_float() > 0.999f) hf_decay_ratio = Common::FixedPoint<50, 14>(0.999f);

        Common::FixedPoint<50, 14> c{0.0f}, d{0.0f};
        if (hf_decay_ratio.to_float() > 0.9949f) {
            c = 0.0f;
            d = 1.0f;
        } else {
            auto e = pow_10(((((1.0f / hf_decay_ratio.to_float()) - 1.0f) * 2) / 100 * (b / 10)).to_float());
            if (e < 0.0001f) e = 0.0001f;
            if (e > 1.0f) e = 1.0f;

            auto f = 1.0f - e;
            if (f < 0.0001f) f = 0.0001f;

            auto g = 2.0f - (unk_value.to_float() * e * 2);

            auto h_sq = g * g - f * f * 4;
            if (h_sq < 0.0f) h_sq = 0.0f;

            auto h = std::sqrt(h_sq);

            c = (g - h) / (f * 2.0f);
            d = 1.0f - c;
        }

        state.hf_decay_prev_gain[i] = state.hf_decay_prev_gain[i] + (c - state.hf_decay_prev_gain[i]) / 4;
        state.hf_decay_gain[i] = state.hf_decay_gain[i] + (pow_10((b / 1000).to_float()) * d * 0.70709228515625f - state.hf_decay_gain[i]) / 4;

        state.prev_feedback_output[i] = 0;
    }
}

/**
 * Initialize a new ReverbInfo state according to the given parameters.
 *
 * @param params                        - Input parameters to update the state.
 * @param state                         - State to be updated.
 * @param workbuffer                    - Game-supplied memory for the state. (Unused)
 * @param long_size_pre_delay_supported - Use a longer pre-delay time before reverb begins.
 */
static void InitializeReverbEffect(const ReverbInfo::ParameterVersion2& params,
                                   ReverbInfo::State& state, const CpuAddr workbuffer,
                                   const bool long_size_pre_delay_supported) {
    state = {};

    auto delay{Common::FixedPoint<50, 14>::from_base(params.sample_rate)};

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
        auto fdn_delay_time{(FdnMaxDelayLineTimes[i] * delay).to_uint_floor()};
        state.fdn_delay_lines[i].Initialize(fdn_delay_time, 1.0f);

        auto decay_delay_time{(DecayMaxDelayLineTimes[i] * delay).to_uint_floor()};
        state.decay_delay_lines[i].Initialize(decay_delay_time, 0.0f);
    }

    const auto pre_delay{long_size_pre_delay_supported ? 350.0f : 150.0f};
    const auto pre_delay_line{(pre_delay * delay).to_uint_floor()};
    state.pre_delay_line.Initialize(pre_delay_line, 1.0f);

    const auto center_delay_time{(5 * delay).to_uint_floor()};
    state.center_delay_line.Initialize(center_delay_time, 1.0f);

    UpdateReverbEffectParameter(params, state);

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
        std::ranges::fill(state.fdn_delay_lines[i].buffer, 0);
        std::ranges::fill(state.decay_delay_lines[i].buffer, 0);
    }
    std::ranges::fill(state.center_delay_line.buffer, 0);
    std::ranges::fill(state.pre_delay_line.buffer, 0);
}

/**
 * Pass-through the effect, copying input to output directly, with no reverb applied.
 *
 * @param inputs        - Array of input mix buffers to copy.
 * @param outputs       - Array of output mix buffers to receive copy.
 * @param channel_count - Number of channels in inputs and outputs.
 * @param sample_count  - Number of samples within each channel.
 */
static void ApplyReverbEffectBypass(std::span<std::span<const s32>> inputs,
                                    std::span<std::span<s32>> outputs, const u32 channel_count,
                                    const u32 sample_count) {
    for (u32 i = 0; i < channel_count; i++) {
        if (inputs[i].data() != outputs[i].data()) {
            std::memcpy(outputs[i].data(), inputs[i].data(), outputs[i].size_bytes());
        }
    }
}

/**
 * Tick the delay lines, reading and returning their current output, and writing a new decaying
 * sample (mix).
 *
 * @param decay  - The decay line.
 * @param fdn    - Feedback delay network.
 * @param mix    - The new calculated sample to be written and decayed.
 * @return The next delayed and decayed sample.
 */
static s64 Axfx2AllPassTickRaw(ReverbInfo::ReverbDelayLine& decay, ReverbInfo::ReverbDelayLine& fdn,
                               const s64 mix_raw) {
    const s64 decay_raw = decay.decay.to_raw();
    const s64 val_raw = decay.Read().to_raw();
    const s64 mixed_raw = mix_raw - DspSimd::MulFp<14>(val_raw, decay_raw);
    const s64 out_raw = decay.Tick(Common::FixedPoint<50, 14>::from_base(mixed_raw)).to_raw() +
                        DspSimd::MulFp<14>(mixed_raw, decay_raw);

    fdn.Tick(Common::FixedPoint<50, 14>::from_base(out_raw));
    return out_raw;
}

template <size_t NumChannels>
static void ApplyReverbEffect(const ReverbInfo::ParameterVersion2& params, ReverbInfo::State& state,
                              std::span<std::span<const s32>> inputs,
                              std::span<std::span<s32>> outputs, const u32 sample_count) {
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes1Ch{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes2Ch{
        0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
    };
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes4Ch{
        0, 0, 1, 1, 0, 1, 2, 2, 3, 3,
    };
    static constexpr std::array<u8, ReverbInfo::MaxDelayTaps> OutTapIndexes6Ch{
        0, 0, 1, 1, 2, 2, 4, 4, 5, 5,
    };

    std::span<const u8> tap_indexes{};
    if constexpr (NumChannels == 1) {
        tap_indexes = OutTapIndexes1Ch;
    } else if constexpr (NumChannels == 2) {
        tap_indexes = OutTapIndexes2Ch;
    } else if constexpr (NumChannels == 4) {
        tap_indexes = OutTapIndexes4Ch;
    } else if constexpr (NumChannels == 6) {
        tap_indexes = OutTapIndexes6Ch;
    }

    std::array<const s32*, NumChannels> in_ptrs{};
    std::array<s32*, NumChannels> out_ptrs{};
    for (size_t c = 0; c < NumChannels; ++c) {
        in_ptrs[c] = inputs[c].data();
        out_ptrs[c] = outputs[c].data();
    }

    const s64 base_gain_raw = params.base_gain;
    const s64 late_gain_raw = params.late_gain;
    const s64 dry_gain_raw = params.dry_gain;
    const s64 wet_gain_raw = params.wet_gain;
    constexpr s64 lfe_scale_raw = Common::FixedPoint<50, 14>{0.2f}.to_raw();
    constexpr s64 half_raw = Common::FixedPoint<50, 14>{0.5f}.to_raw();

    std::array<s64, ReverbInfo::MaxDelayTaps> early_gains_raw{};
    for (u32 t = 0; t < ReverbInfo::MaxDelayTaps; ++t) {
        early_gains_raw[t] = state.early_gains[t].to_raw();
    }
    std::array<s64, ReverbInfo::MaxDelayLines> hf_prev_raw{};
    std::array<s64, ReverbInfo::MaxDelayLines> hf_gain_raw{};
    std::array<s64, ReverbInfo::MaxDelayLines> prev_fb_raw{};
    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; ++i) {
        hf_prev_raw[i] = state.hf_decay_prev_gain[i].to_raw();
        hf_gain_raw[i] = state.hf_decay_gain[i].to_raw();
        prev_fb_raw[i] = state.prev_feedback_output[i].to_raw();
    }

    for (u32 sample_index = 0; sample_index < sample_count; sample_index++) {
        std::array<s64, NumChannels> output_samples_raw{};

        for (u32 early_tap = 0; early_tap < ReverbInfo::MaxDelayTaps; early_tap++) {
            const s64 sample_raw = DspSimd::MulFp<14>(
                state.pre_delay_line.TapOut(state.early_delay_times[early_tap]).to_raw(),
                early_gains_raw[early_tap]);
            output_samples_raw[tap_indexes[early_tap]] += sample_raw;
            if constexpr (NumChannels == 6) {
                output_samples_raw[static_cast<u32>(Channels::LFE)] += sample_raw;
            }
        }

        if constexpr (NumChannels == 6) {
            output_samples_raw[static_cast<u32>(Channels::LFE)] = DspSimd::MulFp<14>(
                output_samples_raw[static_cast<u32>(Channels::LFE)], lfe_scale_raw);
        }

        s64 sum_in = 0;
        for (size_t channel = 0; channel < NumChannels; channel++) {
            sum_in += in_ptrs[channel][sample_index];
        }

        const s64 input_sample_raw = DspSimd::MulFp<14>(sum_in << 20, base_gain_raw);
        state.pre_delay_line.Write(Common::FixedPoint<50, 14>::from_base(input_sample_raw));

        for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
            prev_fb_raw[i] =
                DspSimd::MulFp<14>(prev_fb_raw[i], hf_prev_raw[i]) +
                DspSimd::MulFp<14>(state.fdn_delay_lines[i].Read().to_raw(), hf_gain_raw[i]);
        }

        const s64 pre_delay_sample_raw =
            DspSimd::MulFp<14>(state.pre_delay_line.TapOut(state.pre_delay_time).to_raw(),
                               late_gain_raw);

        const std::array<s64, ReverbInfo::MaxDelayLines> mix_matrix_raw{
            prev_fb_raw[2] + prev_fb_raw[1] + pre_delay_sample_raw,
            -prev_fb_raw[0] - prev_fb_raw[3] + pre_delay_sample_raw,
            prev_fb_raw[0] - prev_fb_raw[3] + pre_delay_sample_raw,
            prev_fb_raw[1] - prev_fb_raw[2] + pre_delay_sample_raw,
        };

        std::array<s64, ReverbInfo::MaxDelayLines> allpass_raw{};
        for (u32 i = 0; i < ReverbInfo::MaxDelayLines; i++) {
            allpass_raw[i] = Axfx2AllPassTickRaw(state.decay_delay_lines[i],
                                                 state.fdn_delay_lines[i], mix_matrix_raw[i]);
        }

        if constexpr (NumChannels == 6) {
            const std::array<s64, MaxChannels> allpass_outputs_raw{
                allpass_raw[0], allpass_raw[1], allpass_raw[2] - allpass_raw[3],
                allpass_raw[3], allpass_raw[2], allpass_raw[3],
            };

            for (size_t channel = 0; channel < NumChannels; channel++) {
                const s64 in_sample_raw = s64{in_ptrs[channel][sample_index]} * dry_gain_raw;
                s64 ap_raw{};
                if (channel == static_cast<size_t>(Channels::Center)) {
                    const s64 center_in = DspSimd::MulFp<14>(allpass_outputs_raw[channel], half_raw);
                    ap_raw = state.center_delay_line
                                 .Tick(Common::FixedPoint<50, 14>::from_base(center_in))
                                 .to_raw();
                } else {
                    ap_raw = allpass_outputs_raw[channel];
                }

                const s64 out_sample_raw =
                    DspSimd::MulFp<14>(output_samples_raw[channel] + ap_raw, wet_gain_raw) / 64;
                out_ptrs[channel][sample_index] =
                    DspSimd::RoundFpToInt<14>(in_sample_raw + out_sample_raw);
            }
        } else {
            for (size_t channel = 0; channel < NumChannels; channel++) {
                const s64 in_sample_raw = s64{in_ptrs[channel][sample_index]} * dry_gain_raw;
                const s64 out_sample_raw =
                    DspSimd::MulFp<14>(output_samples_raw[channel] + allpass_raw[channel],
                                       wet_gain_raw) /
                    64;
                out_ptrs[channel][sample_index] =
                    DspSimd::RoundFpToInt<14>(in_sample_raw + out_sample_raw);
            }
        }
    }

    for (u32 i = 0; i < ReverbInfo::MaxDelayLines; ++i) {
        state.prev_feedback_output[i] = Common::FixedPoint<50, 14>::from_base(prev_fb_raw[i]);
    }
}

/**
 * Apply a Reverb if enabled, according to the current state, on the input mix buffers,
 * saving the results to the output mix buffers.
 *
 * @param params       - Input parameters to use.
 * @param state        - State to use, must be initialized (see InitializeReverbEffect).
 * @param enabled      - If enabled, delay will be applied, otherwise input is copied to output.
 * @param inputs       - Input mix buffers to performan the reverb on.
 * @param outputs      - Output mix buffers to receive the reverbed samples.
 * @param sample_count - Number of samples to process.
 */
static void ApplyReverbEffect(const ReverbInfo::ParameterVersion2& params, ReverbInfo::State& state,
                              const bool enabled, std::span<std::span<const s32>> inputs,
                              std::span<std::span<s32>> outputs, const u32 sample_count) {
    if (enabled) {
        switch (params.channel_count) {
        case 0:
            return;
        case 1:
            ApplyReverbEffect<1>(params, state, inputs, outputs, sample_count);
            break;
        case 2:
            ApplyReverbEffect<2>(params, state, inputs, outputs, sample_count);
            break;
        case 4:
            ApplyReverbEffect<4>(params, state, inputs, outputs, sample_count);
            break;
        case 6:
            ApplyReverbEffect<6>(params, state, inputs, outputs, sample_count);
            break;
        default:
            ApplyReverbEffectBypass(inputs, outputs, params.channel_count, sample_count);
            break;
        }
    } else {
        ApplyReverbEffectBypass(inputs, outputs, params.channel_count, sample_count);
    }
}

void ReverbCommand::Dump([[maybe_unused]] const AudioRenderer::CommandListProcessor& processor,
                         std::string& string) {
    string += fmt::format(
        "ReverbCommand\n\tenabled {} long_size_pre_delay_supported {}\n\tinputs: ", effect_enabled,
        long_size_pre_delay_supported);
    for (u32 i = 0; i < MaxChannels; i++) {
        string += fmt::format("{:02X}, ", inputs[i]);
    }
    string += "\n\toutputs: ";
    for (u32 i = 0; i < MaxChannels; i++) {
        string += fmt::format("{:02X}, ", outputs[i]);
    }
    string += "\n";
}

void ReverbCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    std::array<std::span<const s32>, MaxChannels> input_buffers{};
    std::array<std::span<s32>, MaxChannels> output_buffers{};

    for (u32 i = 0; i < parameter.channel_count; i++) {
        input_buffers[i] = processor.mix_buffers.subspan(inputs[i] * processor.sample_count,
                                                         processor.sample_count);
        output_buffers[i] = processor.mix_buffers.subspan(outputs[i] * processor.sample_count,
                                                          processor.sample_count);
    }

    auto state_{reinterpret_cast<ReverbInfo::State*>(state)};

    if (effect_enabled) {
        if (parameter.state == ReverbInfo::ParameterState::Updating) {
            UpdateReverbEffectParameter(parameter, *state_);
        } else if (parameter.state == ReverbInfo::ParameterState::Initialized) {
            InitializeReverbEffect(parameter, *state_, workbuffer, long_size_pre_delay_supported);
        }
    }
    ApplyReverbEffect(parameter, *state_, effect_enabled, input_buffers, output_buffers,
                      processor.sample_count);
}

bool ReverbCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
