// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/effect/biquad_filter.h"
#include "audio_core/renderer/voice/voice_state.h"

namespace AudioCore::Renderer {

namespace {

constexpr f64 MinS32F64 = static_cast<f64>((std::numeric_limits<s32>::min)());
constexpr f64 MaxS32F64 = static_cast<f64>((std::numeric_limits<s32>::max)());

[[nodiscard]] constexpr bool IsStateZero(const VoiceState::BiquadFilterState& state) noexcept {
    return (state.s0 | state.s1 | state.s2 | state.s3) == 0;
}

[[nodiscard]] constexpr bool IsMemorylessBiquad(const std::array<f64, 3>& b,
                                                const std::array<f64, 2>& a) noexcept {
    return b[1] == 0.0 && b[2] == 0.0 && a[0] == 0.0 && a[1] == 0.0;
}

void ApplyBiquadCore(std::span<s32> output, std::span<const s32> input,
                     const std::array<f64, 3>& b, const std::array<f64, 2>& a,
                     VoiceState::BiquadFilterState& state, const u32 sample_count) noexcept {
    if (sample_count == 0) {
        return;
    }

    s32* out_ptr = output.data();
    const s32* in_ptr = input.data();

    // Fastpath 1: b1 = b2 = a0 = a1 = 0 (most commonly unity pass-through {1, 0, 0}, {0, 0} on
    // unoccluded voices). No feedback or history terms participate; only the final two samples are
    // needed to leave `state` bit-identical for future frames.
    if (sample_count >= 2 && IsMemorylessBiquad(b, a)) {
        const s32 last0 = in_ptr[sample_count - 1];
        const s32 last1 = in_ptr[sample_count - 2];
        const f64 b0 = b[0];

        if (b0 == 1.0) {
            if (out_ptr != in_ptr) {
                std::memcpy(out_ptr, in_ptr, static_cast<size_t>(sample_count) * sizeof(s32));
            }
            state.s0 = std::bit_cast<s64>(static_cast<f64>(last0));
            state.s1 = std::bit_cast<s64>(static_cast<f64>(last1));
            state.s2 = std::bit_cast<s64>(static_cast<f64>(last0) + 0.0);
            state.s3 = std::bit_cast<s64>(static_cast<f64>(last1) + 0.0);
            return;
        }
        if (b0 == 0.0) {
            std::memset(out_ptr, 0, static_cast<size_t>(sample_count) * sizeof(s32));
            state.s0 = std::bit_cast<s64>(static_cast<f64>(last0));
            state.s1 = std::bit_cast<s64>(static_cast<f64>(last1));
            state.s2 = 0;
            state.s3 = 0;
            return;
        }

        for (u32 i = 0; i < sample_count; ++i) {
            const f64 sample = static_cast<f64>(in_ptr[i]) * b0 + 0.0;
            out_ptr[i] = static_cast<s32>(std::clamp(sample, MinS32F64, MaxS32F64));
        }
        state.s0 = std::bit_cast<s64>(static_cast<f64>(last0));
        state.s1 = std::bit_cast<s64>(static_cast<f64>(last1));
        state.s2 = std::bit_cast<s64>(static_cast<f64>(last0) * b0 + 0.0);
        state.s3 = std::bit_cast<s64>(static_cast<f64>(last1) * b0 + 0.0);
        return;
    }

    // Fastpath 2: silent input buffer with zero filter state stays zero.
    if (IsStateZero(state) && DspSimd::IsAllZeroS32(in_ptr, sample_count)) {
        if (out_ptr != in_ptr) {
            std::memset(out_ptr, 0, static_cast<size_t>(sample_count) * sizeof(s32));
        }
        return;
    }

    const f64 b0 = b[0];
    const f64 b1 = b[1];
    const f64 b2 = b[2];
    const f64 a0 = a[0];
    const f64 a1 = a[1];

    f64 s0 = std::bit_cast<f64>(state.s0);
    f64 s1 = std::bit_cast<f64>(state.s1);
    f64 s2 = std::bit_cast<f64>(state.s2);
    f64 s3 = std::bit_cast<f64>(state.s3);

    // Precompute the non-recursive feedforward FIR terms 4 samples ahead so out-of-order FP units
    // overlap feedforward multiplies with the serial AR(2) feedback dependency, while preserving
    // exact IEEE-754 left-to-right addition order.
    u32 i = 0;
    for (; i + 4 <= sample_count; i += 4) {
        const f64 x0 = static_cast<f64>(in_ptr[i + 0]);
        const f64 x1 = static_cast<f64>(in_ptr[i + 1]);
        const f64 x2 = static_cast<f64>(in_ptr[i + 2]);
        const f64 x3 = static_cast<f64>(in_ptr[i + 3]);

        const f64 ff0 = (x0 * b0 + s0 * b1) + s1 * b2;
        const f64 ff1 = (x1 * b0 + x0 * b1) + s0 * b2;
        const f64 ff2 = (x2 * b0 + x1 * b1) + x0 * b2;
        const f64 ff3 = (x3 * b0 + x2 * b1) + x1 * b2;

        const f64 y0 = (ff0 + s2 * a0) + s3 * a1;
        out_ptr[i + 0] = static_cast<s32>(std::clamp(y0, MinS32F64, MaxS32F64));

        const f64 y1 = (ff1 + y0 * a0) + s2 * a1;
        out_ptr[i + 1] = static_cast<s32>(std::clamp(y1, MinS32F64, MaxS32F64));

        const f64 y2 = (ff2 + y1 * a0) + y0 * a1;
        out_ptr[i + 2] = static_cast<s32>(std::clamp(y2, MinS32F64, MaxS32F64));

        const f64 y3 = (ff3 + y2 * a0) + y1 * a1;
        out_ptr[i + 3] = static_cast<s32>(std::clamp(y3, MinS32F64, MaxS32F64));

        s1 = x2;
        s0 = x3;
        s3 = y2;
        s2 = y3;
    }

    for (; i < sample_count; ++i) {
        const f64 x = static_cast<f64>(in_ptr[i]);
        const f64 y = (((x * b0 + s0 * b1) + s1 * b2) + s2 * a0) + s3 * a1;
        out_ptr[i] = static_cast<s32>(std::clamp(y, MinS32F64, MaxS32F64));
        s1 = s0;
        s0 = x;
        s3 = s2;
        s2 = y;
    }

    state.s0 = std::bit_cast<s64>(s0);
    state.s1 = std::bit_cast<s64>(s1);
    state.s2 = std::bit_cast<s64>(s2);
    state.s3 = std::bit_cast<s64>(s3);
}

void ApplyDoubleBiquadCore(std::span<s32> output, std::span<const s32> input,
                           const std::array<f64, 3>& b0_arr, const std::array<f64, 2>& a0_arr,
                           const std::array<f64, 3>& b1_arr, const std::array<f64, 2>& a1_arr,
                           VoiceState::BiquadFilterState& state0,
                           VoiceState::BiquadFilterState& state1,
                           const u32 sample_count) noexcept {
    if (sample_count == 0) {
        return;
    }

    // Delegate to single-tap fastpaths if either stage is memoryless (e.g. unity {1, 0, 0}) or
    // both states and input are silent.
    if (IsMemorylessBiquad(b0_arr, a0_arr) || IsMemorylessBiquad(b1_arr, a1_arr)) {
        ApplyBiquadCore(output, input, b0_arr, a0_arr, state0, sample_count);
        ApplyBiquadCore(output, output, b1_arr, a1_arr, state1, sample_count);
        return;
    }

    s32* out_ptr = output.data();
    const s32* in_ptr = input.data();

    if (IsStateZero(state0) && IsStateZero(state1) && DspSimd::IsAllZeroS32(in_ptr, sample_count)) {
        if (out_ptr != in_ptr) {
            std::memset(out_ptr, 0, static_cast<size_t>(sample_count) * sizeof(s32));
        }
        return;
    }

    const f64 b00 = b0_arr[0], b01 = b0_arr[1], b02 = b0_arr[2];
    const f64 a00 = a0_arr[0], a01 = a0_arr[1];
    const f64 b10 = b1_arr[0], b11 = b1_arr[1], b12 = b1_arr[2];
    const f64 a10 = a1_arr[0], a11 = a1_arr[1];

    f64 s0_0 = std::bit_cast<f64>(state0.s0);
    f64 s0_1 = std::bit_cast<f64>(state0.s1);
    f64 s0_2 = std::bit_cast<f64>(state0.s2);
    f64 s0_3 = std::bit_cast<f64>(state0.s3);

    f64 s1_0 = std::bit_cast<f64>(state1.s0);
    f64 s1_1 = std::bit_cast<f64>(state1.s1);
    f64 s1_2 = std::bit_cast<f64>(state1.s2);
    f64 s1_3 = std::bit_cast<f64>(state1.s3);

    // Fused single-pass cascade: keeps intermediate clamped s32 samples in registers and lets the
    // CPU overlap stage 0's feedback on sample i+1 with stage 1's feedback on sample i.
    for (u32 i = 0; i < sample_count; ++i) {
        const f64 x0 = static_cast<f64>(in_ptr[i]);
        const f64 y0 = (((x0 * b00 + s0_0 * b01) + s0_1 * b02) + s0_2 * a00) + s0_3 * a01;
        const s32 mid = static_cast<s32>(std::clamp(y0, MinS32F64, MaxS32F64));
        s0_1 = s0_0;
        s0_0 = x0;
        s0_3 = s0_2;
        s0_2 = y0;

        const f64 x1 = static_cast<f64>(mid);
        const f64 y1 = (((x1 * b10 + s1_0 * b11) + s1_1 * b12) + s1_2 * a10) + s1_3 * a11;
        out_ptr[i] = static_cast<s32>(std::clamp(y1, MinS32F64, MaxS32F64));
        s1_1 = s1_0;
        s1_0 = x1;
        s1_3 = s1_2;
        s1_2 = y1;
    }

    state0.s0 = std::bit_cast<s64>(s0_0);
    state0.s1 = std::bit_cast<s64>(s0_1);
    state0.s2 = std::bit_cast<s64>(s0_2);
    state0.s3 = std::bit_cast<s64>(s0_3);

    state1.s0 = std::bit_cast<s64>(s1_0);
    state1.s1 = std::bit_cast<s64>(s1_1);
    state1.s2 = std::bit_cast<s64>(s1_2);
    state1.s3 = std::bit_cast<s64>(s1_3);
}

[[nodiscard]] constexpr std::array<f64, 3> Q14NumToDouble(const std::array<s16, 3>& b) noexcept {
    return {Common::FixedPoint<50, 14>::from_base(b[0]).to_double(),
            Common::FixedPoint<50, 14>::from_base(b[1]).to_double(),
            Common::FixedPoint<50, 14>::from_base(b[2]).to_double()};
}

[[nodiscard]] constexpr std::array<f64, 2> Q14DenToDouble(const std::array<s16, 2>& a) noexcept {
    return {Common::FixedPoint<50, 14>::from_base(a[0]).to_double(),
            Common::FixedPoint<50, 14>::from_base(a[1]).to_double()};
}

[[nodiscard]] constexpr std::array<f64, 3> F32NumToDouble(const std::array<f32, 3>& b) noexcept {
    return {static_cast<f64>(b[0]), static_cast<f64>(b[1]), static_cast<f64>(b[2])};
}

[[nodiscard]] constexpr std::array<f64, 2> F32DenToDouble(const std::array<f32, 2>& a) noexcept {
    return {static_cast<f64>(a[0]), static_cast<f64>(a[1])};
}

} // namespace

void ApplyBiquadFilterFloat(std::span<s32> output, std::span<const s32> input,
                            std::array<s16, 3>& b_, std::array<s16, 2>& a_,
                            VoiceState::BiquadFilterState& state, const u32 sample_count) {
    ApplyBiquadCore(output, input, Q14NumToDouble(b_), Q14DenToDouble(a_), state, sample_count);
}

void ApplyBiquadFilterFloat2(std::span<s32> output, std::span<const s32> input,
                             std::array<f32, 3>& b, std::array<f32, 2>& a,
                             VoiceState::BiquadFilterState& state, const u32 sample_count) {
    ApplyBiquadCore(output, input, F32NumToDouble(b), F32DenToDouble(a), state, sample_count);
}

void ApplyDoubleBiquadFilterFloat(
    std::span<s32> output, std::span<const s32> input,
    const std::array<VoiceInfo::BiquadFilterParameter, MaxBiquadFilters>& biquads,
    VoiceState::BiquadFilterState& state0, VoiceState::BiquadFilterState& state1,
    const u32 sample_count) {
    ApplyDoubleBiquadCore(output, input, Q14NumToDouble(biquads[0].b), Q14DenToDouble(biquads[0].a),
                          Q14NumToDouble(biquads[1].b), Q14DenToDouble(biquads[1].a), state0,
                          state1, sample_count);
}

void ApplyDoubleBiquadFilterFloat2(
    std::span<s32> output, std::span<const s32> input,
    const std::array<VoiceInfo::BiquadFilterParameter2, MaxBiquadFilters>& biquads,
    VoiceState::BiquadFilterState& state0, VoiceState::BiquadFilterState& state1,
    const u32 sample_count) {
    ApplyDoubleBiquadCore(output, input, F32NumToDouble(biquads[0].numerator),
                          F32DenToDouble(biquads[0].denominator),
                          F32NumToDouble(biquads[1].numerator),
                          F32DenToDouble(biquads[1].denominator), state0, state1, sample_count);
}

static void ApplyBiquadFilterInt(std::span<s32> output, std::span<const s32> input,
                                 const std::array<s16, 3>& b, const std::array<s16, 2>& a,
                                 VoiceState::BiquadFilterState& state, const u32 sample_count) {
    if (sample_count == 0) {
        return;
    }

    constexpr s64 min{(std::numeric_limits<s32>::min)()};
    constexpr s64 max{(std::numeric_limits<s32>::max)()};

    s32* out_ptr = output.data();
    const s32* in_ptr = input.data();

    const s64 b0 = b[0];
    const s64 b1 = b[1];
    const s64 b2 = b[2];
    const s64 a0 = a[0];
    const s64 a1 = a[1];

    s64 s0 = state.s0;
    s64 s1 = state.s1;

    if (b1 == 0 && b2 == 0 && a0 == 0 && a1 == 0 && s0 == 0 && s1 == 0) {
        if (b0 == 16384) {
            if (out_ptr != in_ptr) {
                std::memcpy(out_ptr, in_ptr, static_cast<size_t>(sample_count) * sizeof(s32));
            }
            return;
        }
        if (b0 == 0) {
            std::memset(out_ptr, 0, static_cast<size_t>(sample_count) * sizeof(s32));
            return;
        }
    }

    if ((s0 | s1) == 0 && DspSimd::IsAllZeroS32(in_ptr, sample_count)) {
        if (out_ptr != in_ptr) {
            std::memset(out_ptr, 0, static_cast<size_t>(sample_count) * sizeof(s32));
        }
        return;
    }

    for (u32 i = 0; i < sample_count; i++) {
        const s64 in_sample{in_ptr[i]};
        const s64 sample{in_sample * b0 + s0};
        const s64 out_sample{std::clamp<s64>((sample + (1 << 13)) >> 14, min, max)};

        out_ptr[i] = static_cast<s32>(out_sample);

        s0 = s1 + b1 * in_sample + a0 * out_sample;
        s1 = b2 * in_sample + a1 * out_sample;
    }

    state.s0 = s0;
    state.s1 = s1;
}

void BiquadFilterCommand::Dump(
    [[maybe_unused]] const AudioRenderer::CommandListProcessor& processor, std::string& string) {
    string += fmt::format(
        "BiquadFilterCommand\n\tinput {:02X} output {:02X} needs_init {} use_float_processing {}\n",
        input, output, needs_init, use_float_processing);
}

void BiquadFilterCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    auto state_{reinterpret_cast<VoiceState::BiquadFilterState*>(state)};
    if (needs_init) {
        *state_ = {};
    }

    auto input_buffer{
        processor.mix_buffers.subspan(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{
        processor.mix_buffers.subspan(output * processor.sample_count, processor.sample_count)};

    if (use_float_processing) {
        if (use_float_coefficients) {
            ApplyBiquadFilterFloat2(output_buffer, input_buffer, biquad_float.numerator,
                                    biquad_float.denominator, *state_, processor.sample_count);
        } else {
            ApplyBiquadFilterFloat(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                                   processor.sample_count);
        }
    } else {
        ApplyBiquadFilterInt(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                             processor.sample_count);
    }
}

bool BiquadFilterCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
