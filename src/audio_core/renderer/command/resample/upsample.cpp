// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "audio_core/adsp/apps/audio_renderer/command_list_processor.h"
#include "audio_core/renderer/command/resample/upsample.h"
#include "audio_core/renderer/upsampler/upsampler_info.h"

namespace AudioCore::Renderer {

namespace {
constexpr u32 WindowSize = 10;
constexpr u32 FullTapCount = WindowSize * 2;

using SincHalf = std::array<Common::FixedPoint<17, 15>, WindowSize>;
using SincKernel = std::array<s32, FullTapCount>;

consteval SincKernel MakeLinearKernel(const SincHalf& back, const SincHalf& fwd) {
    SincKernel kernel{};
    for (u32 i = 0; i < WindowSize; ++i) {
        kernel[i] = back[WindowSize - 1 - i].to_raw();
        kernel[WindowSize + i] = fwd[i].to_raw();
    }
    return kernel;
}

constexpr SincHalf WindowedSinc1{
    0.95376587f,   -0.12872314f, 0.060028076f,  -0.032470703f, 0.017669678f,
    -0.009124756f, 0.004272461f, -0.001739502f, 0.000579834f,  -0.000091552734f,
};
constexpr SincHalf WindowedSinc2{
    0.8230896f,    -0.19161987f,  0.093444824f,  -0.05090332f,   0.027557373f,
    -0.014038086f, 0.0064697266f, -0.002532959f, 0.00079345703f, -0.00012207031f,
};
constexpr SincHalf WindowedSinc3{
    0.6298828f,    -0.19274902f, 0.09725952f,    -0.05319214f,  0.028625488f,
    -0.014373779f, 0.006500244f, -0.0024719238f, 0.0007324219f, -0.000091552734f,
};
constexpr SincHalf WindowedSinc4{
    0.4057312f,    -0.1468811f,  0.07601929f,    -0.041656494f,  0.022216797f,
    -0.011016846f, 0.004852295f, -0.0017700195f, 0.00048828125f, -0.000030517578f,
};
constexpr SincHalf WindowedSinc5{
    0.1854248f,    -0.075164795f, 0.03967285f,    -0.021728516f,  0.011474609f,
    -0.005584717f, 0.0024108887f, -0.0008239746f, 0.00021362305f, 0.0f,
};

constexpr SincKernel Kernel15 = MakeLinearKernel(WindowedSinc1, WindowedSinc5);
constexpr SincKernel Kernel24 = MakeLinearKernel(WindowedSinc2, WindowedSinc4);
constexpr SincKernel Kernel33 = MakeLinearKernel(WindowedSinc3, WindowedSinc3);
constexpr SincKernel Kernel42 = MakeLinearKernel(WindowedSinc4, WindowedSinc2);
constexpr SincKernel Kernel51 = MakeLinearKernel(WindowedSinc5, WindowedSinc1);

[[nodiscard]] inline s32 ConvolveLinear20(const s32* __restrict hist,
                                          const SincKernel& kernel) noexcept {
    u64 result = 0;
    for (u32 k = 0; k < FullTapCount; ++k) {
        result += static_cast<u64>(static_cast<s64>(hist[k])) *
                  static_cast<u64>(static_cast<s64>(kernel[k]));
    }
    return static_cast<s32>(result >> (8 + 15));
}
} // namespace

static void SrcProcessFrame(std::span<s32> output, std::span<const s32> input,
                            const u32 target_sample_count, const u32 source_sample_count,
                            UpsamplerState* state) {
    if (!state->initialized) {
        switch (source_sample_count) {
        case 40:
            state->window_size = WindowSize;
            state->ratio = 6.0f;
            state->history.fill(0);
            break;

        case 80:
            state->window_size = WindowSize;
            state->ratio = 3.0f;
            state->history.fill(0);
            break;

        case 160:
            state->window_size = WindowSize;
            state->ratio = 1.5f;
            state->history.fill(0);
            break;

        default:
            LOG_ERROR(Service_Audio, "Invalid upsampling source count {}!", source_sample_count);
            state->window_size = WindowSize;
            state->ratio = 1.5f;
            state->history.fill(0);
            break;
        }

        state->history_input_index = 0;
        state->history_output_index = 9;
        state->history_start_index = 0;
        state->history_end_index = UpsamplerState::HistorySize - 1;
        state->initialized = true;
    }

    if (target_sample_count == 0) {
        return;
    }

    // Fast linear-window path when the 20-element ring uses standard indices and fits on stack.
    constexpr u32 MaxStackInputs = 256;
    if (target_sample_count <= MaxStackInputs && state->history_start_index == 0 &&
        state->history_end_index == UpsamplerState::HistorySize - 1 &&
        ((state->history_input_index + 9U) % UpsamplerState::HistorySize) ==
            state->history_output_index) {
        std::array<s32, UpsamplerState::HistorySize + MaxStackInputs> linear;
        const u32 in_idx0 = state->history_input_index;
        for (u32 i = 0; i < UpsamplerState::HistorySize; ++i) {
            linear[i] = state->history[(in_idx0 + i) % UpsamplerState::HistorySize].to_raw();
        }

        u32 pos = UpsamplerState::HistorySize;
        u32 read_index = 0;
        s32* __restrict out_ptr = output.data();
        const s32* __restrict in_ptr = input.data();
        u8 sample_idx = state->sample_index;

        const auto push_input = [&]() noexcept {
            linear[pos++] = Common::FixedPoint<24, 8>{in_ptr[read_index++]}.to_raw();
        };

        switch (state->ratio.to_int_floor()) {
        case 6:
            for (u32 write_index = 0; write_index < target_sample_count; write_index++) {
                switch (sample_idx) {
                case 0:
                    push_input();
                    out_ptr[write_index] = linear[pos - 11] >> 8;
                    break;
                case 1:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel15);
                    break;
                case 2:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel24);
                    break;
                case 3:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel33);
                    break;
                case 4:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel42);
                    break;
                case 5:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel51);
                    break;
                }
                sample_idx = static_cast<u8>(sample_idx == 5 ? 0 : sample_idx + 1);
            }
            break;

        case 3:
            for (u32 write_index = 0; write_index < target_sample_count; write_index++) {
                switch (sample_idx) {
                case 0:
                    push_input();
                    out_ptr[write_index] = linear[pos - 11] >> 8;
                    break;
                case 1:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel24);
                    break;
                case 2:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel42);
                    break;
                }
                sample_idx = static_cast<u8>(sample_idx == 2 ? 0 : sample_idx + 1);
            }
            break;

        default:
            for (u32 write_index = 0; write_index < target_sample_count; write_index++) {
                switch (sample_idx) {
                case 0:
                    push_input();
                    out_ptr[write_index] = linear[pos - 11] >> 8;
                    break;
                case 1:
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel42);
                    break;
                case 2:
                    push_input();
                    out_ptr[write_index] = ConvolveLinear20(&linear[pos - 20], Kernel24);
                    break;
                }
                sample_idx = static_cast<u8>(sample_idx == 2 ? 0 : sample_idx + 1);
            }
            break;
        }

        state->sample_index = sample_idx;
        const u32 new_in_idx = (in_idx0 + read_index) % UpsamplerState::HistorySize;
        state->history_input_index = static_cast<u16>(new_in_idx);
        state->history_output_index =
            static_cast<u16>((new_in_idx + 9U) % UpsamplerState::HistorySize);
        const s32* tail = &linear[pos - UpsamplerState::HistorySize];
        for (u32 i = 0; i < UpsamplerState::HistorySize; ++i) {
            state->history[(new_in_idx + i) % UpsamplerState::HistorySize] =
                Common::FixedPoint<24, 8>::from_base(tail[i]);
        }
        return;
    }

    u32 read_index{0};
    auto increment = [&]() -> void {
        state->history[state->history_input_index] = input[read_index++];
        state->history_input_index =
            static_cast<u16>((state->history_input_index + 1) % UpsamplerState::HistorySize);
        state->history_output_index =
            static_cast<u16>((state->history_output_index + 1) % UpsamplerState::HistorySize);
    };

    auto calculate_sample = [&state](std::span<const Common::FixedPoint<17, 15>> coeffs1,
                                     std::span<const Common::FixedPoint<17, 15>> coeffs2) -> s32 {
        auto output_index{state->history_output_index};
        u64 result{0};

        for (u32 coeff_index = 0; coeff_index < 10; coeff_index++) {
            result += static_cast<u64>(state->history[output_index].to_raw()) *
                      coeffs1[coeff_index].to_raw();
            output_index = output_index == state->history_start_index ? state->history_end_index
                                                                      : output_index - 1;
        }

        output_index =
            static_cast<u16>((state->history_output_index + 1) % UpsamplerState::HistorySize);

        for (u32 coeff_index = 0; coeff_index < 10; coeff_index++) {
            result += static_cast<u64>(state->history[output_index].to_raw()) *
                      coeffs2[coeff_index].to_raw();
            output_index = output_index == state->history_end_index ? state->history_start_index
                                                                    : output_index + 1;
        }

        return static_cast<s32>(result >> (8 + 15));
    };

    switch (state->ratio.to_int_floor()) {
    case 6:
        for (u32 write_index = 0; write_index < target_sample_count; write_index++) {
            switch (state->sample_index) {
            case 0:
                increment();
                output[write_index] = state->history[state->history_output_index].to_int_floor();
                break;
            case 1:
                output[write_index] = calculate_sample(WindowedSinc1, WindowedSinc5);
                break;
            case 2:
                output[write_index] = calculate_sample(WindowedSinc2, WindowedSinc4);
                break;
            case 3:
                output[write_index] = calculate_sample(WindowedSinc3, WindowedSinc3);
                break;
            case 4:
                output[write_index] = calculate_sample(WindowedSinc4, WindowedSinc2);
                break;
            case 5:
                output[write_index] = calculate_sample(WindowedSinc5, WindowedSinc1);
                break;
            }
            state->sample_index = static_cast<u8>((state->sample_index + 1) % 6);
        }
        break;

    case 3:
        for (u32 write_index = 0; write_index < target_sample_count; write_index++) {
            switch (state->sample_index) {
            case 0:
                increment();
                output[write_index] = state->history[state->history_output_index].to_int_floor();
                break;
            case 1:
                output[write_index] = calculate_sample(WindowedSinc2, WindowedSinc4);
                break;
            case 2:
                output[write_index] = calculate_sample(WindowedSinc4, WindowedSinc2);
                break;
            }
            state->sample_index = static_cast<u8>((state->sample_index + 1) % 3);
        }
        break;

    default:
        for (u32 write_index = 0; write_index < target_sample_count; write_index++) {
            switch (state->sample_index) {
            case 0:
                increment();
                output[write_index] = state->history[state->history_output_index].to_int_floor();
                break;
            case 1:
                output[write_index] = calculate_sample(WindowedSinc4, WindowedSinc2);
                break;
            case 2:
                increment();
                output[write_index] = calculate_sample(WindowedSinc2, WindowedSinc4);
                break;
            }
            state->sample_index = static_cast<u8>((state->sample_index + 1) % 3);
        }
        break;
    }
}

auto UpsampleCommand::Dump([[maybe_unused]] const AudioRenderer::CommandListProcessor& processor,
                           std::string& string) -> void {
    string += fmt::format("UpsampleCommand\n\tsource_sample_count {} source_sample_rate {}",
                          source_sample_count, source_sample_rate);
    const auto upsampler{reinterpret_cast<UpsamplerInfo*>(upsampler_info)};
    if (upsampler != nullptr) {
        string += fmt::format("\n\tUpsampler\n\t\tenabled {} sample count {}\n\tinputs: ",
                              upsampler->enabled, upsampler->sample_count);
        for (u32 i = 0; i < upsampler->input_count; i++) {
            string += fmt::format("{:02X}, ", upsampler->inputs[i]);
        }
    }
    string += "\n";
}

void UpsampleCommand::Process(const AudioRenderer::CommandListProcessor& processor) {
    const auto info{reinterpret_cast<UpsamplerInfo*>(upsampler_info)};
    const auto input_count{(std::min)(info->input_count, buffer_count)};
    const std::span<const s16> inputs_{reinterpret_cast<const s16*>(inputs), input_count};

    for (u32 i = 0; i < input_count; i++) {
        const auto channel{inputs_[i]};

        if (channel >= 0 && channel < static_cast<s16>(processor.buffer_count)) {
            auto state{&info->states[i]};
            std::span<s32> output{
                reinterpret_cast<s32*>(samples_buffer + info->sample_count * channel * sizeof(s32)),
                info->sample_count};
            auto input{processor.mix_buffers.subspan(channel * processor.sample_count,
                                                     processor.sample_count)};

            SrcProcessFrame(output, input, info->sample_count, source_sample_count, state);
        }
    }
}

bool UpsampleCommand::Verify(const AudioRenderer::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::Renderer
