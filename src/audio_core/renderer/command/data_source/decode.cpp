// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "audio_core/common/dsp_simd.h"
#include "audio_core/renderer/command/data_source/decode.h"
#include "audio_core/renderer/command/resample/resample.h"
#include "common/fixed_point.h"
#include "common/logging.h"
#include "core/guest_memory.h"
#include "core/memory.h"

namespace AudioCore::Renderer {

constexpr u32 TempBufferSize = 0x3F00;

template <typename T>
static u32 DecodePcm(Core::Memory::Memory& memory, std::span<s16> out_buffer,
                     const DecodeArg& req) {
    constexpr s32 min{(std::numeric_limits<s16>::min)()};
    constexpr s32 max{(std::numeric_limits<s16>::max)()};

    if (req.buffer == 0 || req.buffer_size == 0) {
        return 0;
    }

    if (req.start_offset >= req.end_offset) {
        return 0;
    }

    auto samples_to_decode{
        (std::min)(req.samples_to_read, req.end_offset - req.start_offset - req.offset)};
    const u32 channel_count{static_cast<u32>(req.channel_count)};
    s16* __restrict dst = out_buffer.data();

    switch (req.channel_count) {
    default: {
        const VAddr source{req.buffer +
                           (((req.start_offset + req.offset) * channel_count) * sizeof(T))};
        const u64 size{channel_count * samples_to_decode};

        Core::Memory::CpuGuestMemory<T, Core::Memory::GuestMemoryFlags::UnsafeRead> samples(
            memory, source, size);
        const T* __restrict src = samples.data() + req.target_channel;
        if constexpr (std::is_floating_point_v<T>) {
            for (u32 i = 0; i < samples_to_decode; i++) {
                const auto sample{static_cast<s32>(src[i * channel_count] *
                                                   static_cast<f32>((std::numeric_limits<s16>::max)()))};
                dst[i] = static_cast<s16>(std::clamp(sample, min, max));
            }
        } else {
            for (u32 i = 0; i < samples_to_decode; i++) {
                dst[i] = src[i * channel_count];
            }
        }
    } break;

    case 1: {
        if (req.target_channel != 0) {
            LOG_ERROR(Service_Audio, "Invalid target channel, expected 0, got {}",
                      req.target_channel);
            return 0;
        }

        const VAddr source{req.buffer + ((req.start_offset + req.offset) * sizeof(T))};
        Core::Memory::CpuGuestMemory<T, Core::Memory::GuestMemoryFlags::UnsafeRead> samples(
            memory, source, samples_to_decode);
        const T* __restrict src = samples.data();

        if constexpr (std::is_floating_point_v<T>) {
            u32 i = 0;
#if defined(ARCHITECTURE_arm64)
            const float32x4_t vscale = vdupq_n_f32(32767.0f);
            const float32x4_t vmin = vdupq_n_f32(-32768.0f);
            const float32x4_t vmax = vdupq_n_f32(32767.0f);
            for (; i + 8 <= samples_to_decode; i += 8) {
                const float32x4_t f0 =
                    vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src + i), vscale), vmin), vmax);
                const float32x4_t f1 =
                    vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src + i + 4), vscale), vmin), vmax);
                const int16x4_t s0 = vqmovn_s32(vcvtq_s32_f32(f0));
                const int16x8_t s01 = vqmovn_high_s32(s0, vcvtq_s32_f32(f1));
                vst1q_s16(dst + i, s01);
            }
#elif defined(ARCHITECTURE_x86_64)
            const __m128 vscale = _mm_set1_ps(32767.0f);
            const __m128 vmin = _mm_set1_ps(-32768.0f);
            const __m128 vmax = _mm_set1_ps(32767.0f);
            for (; i + 8 <= samples_to_decode; i += 8) {
                const __m128 p0 = _mm_mul_ps(_mm_loadu_ps(src + i), vscale);
                const __m128 p1 = _mm_mul_ps(_mm_loadu_ps(src + i + 4), vscale);
                // NaN checks: keep scalar static_cast<s32> behavior if non-finite values appear.
                if (_mm_movemask_ps(_mm_cmpord_ps(p0, p1)) == 0xF) {
                    const __m128 c0 = _mm_min_ps(_mm_max_ps(p0, vmin), vmax);
                    const __m128 c1 = _mm_min_ps(_mm_max_ps(p1, vmin), vmax);
                    const __m128i i0 = _mm_cvttps_epi32(c0);
                    const __m128i i1 = _mm_cvttps_epi32(c1);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_packs_epi32(i0, i1));
                } else {
                    for (u32 k = 0; k < 8; ++k) {
                        const auto sample = static_cast<s32>(
                            src[i + k] * static_cast<f32>((std::numeric_limits<s16>::max)()));
                        dst[i + k] = static_cast<s16>(std::clamp(sample, min, max));
                    }
                }
            }
#endif
            for (; i < samples_to_decode; i++) {
                const auto sample{
                    static_cast<s32>(src[i] * static_cast<f32>((std::numeric_limits<s16>::max)()))};
                dst[i] = static_cast<s16>(std::clamp(sample, min, max));
            }
        } else {
            std::memcpy(dst, src, samples_to_decode * sizeof(s16));
        }
    } break;
    }

    return samples_to_decode;
}

static u32 DecodeAdpcm(Core::Memory::Memory& memory, std::span<s16> out_buffer,
                       const DecodeArg& req) {
    constexpr u32 SamplesPerFrame{14};
    constexpr u32 NibblesPerFrame{16};

    if (req.buffer == 0 || req.buffer_size == 0) {
        return 0;
    }

    if (req.end_offset < req.start_offset) {
        return 0;
    }

    auto end{(req.end_offset % SamplesPerFrame) +
             NibblesPerFrame * (req.end_offset / SamplesPerFrame)};
    if (req.end_offset % SamplesPerFrame) {
        end += 3;
    } else {
        end += 1;
    }

    if (req.buffer_size < end / 2) {
        return 0;
    }

    auto start_pos{req.start_offset + req.offset};
    auto samples_to_process{(std::min)(req.end_offset - start_pos, req.samples_to_read)};
    if (samples_to_process == 0) {
        return 0;
    }

    auto samples_to_read = samples_to_process;
    auto samples_remaining_in_frame = start_pos % SamplesPerFrame;
    auto position_in_frame =
        (start_pos / SamplesPerFrame) * NibblesPerFrame + samples_remaining_in_frame;
    if (samples_remaining_in_frame) {
        position_in_frame += 2;
    }

    const auto size{(std::max)((samples_to_process / 8U) * SamplesPerFrame, 8U)};
    Core::Memory::CpuGuestMemory<u8, Core::Memory::GuestMemoryFlags::UnsafeRead> wavebuffer(
        memory, req.buffer + position_in_frame / 2, size);
    const u8* __restrict wave_ptr = wavebuffer.data();
    s16* __restrict out_ptr = out_buffer.data();
    const u32 out_limit = static_cast<u32>(out_buffer.size());

    auto context = req.adpcm_context;
    auto header = context->header;
    u32 scale_shift = (header & 0xFU) + 11U;
    u8 coeff_index = static_cast<u8>((header >> 4U) & 0x7U);
    s32 coeff0 = req.coefficients[coeff_index * 2 + 0];
    s32 coeff1 = req.coefficients[coeff_index * 2 + 1];

    s32 yn0 = context->yn0;
    s32 yn1 = context->yn1;

    // 4-bit signed nibble sign extension: 0..7 -> 0..7, 8..15 -> -8..-1.
    const auto high_nibble = [](const u8 byte) noexcept -> s32 {
        return static_cast<s32>(static_cast<s8>(byte)) >> 4;
    };
    const auto low_nibble = [](const u8 byte) noexcept -> s32 {
        return static_cast<s32>(static_cast<s8>(static_cast<u8>(byte << 4))) >> 4;
    };
    const auto decode_sample = [&](const s32 code) noexcept -> s16 {
        const s32 xn_shifted = static_cast<s32>(static_cast<u32>(code) << scale_shift);
        const s32 prediction = coeff0 * yn0 + coeff1 * yn1;
        const s32 sample = (xn_shifted + 0x400 + prediction) >> 11;
        const s32 saturated = std::clamp<s32>(sample, -0x8000, 0x7FFF);
        yn1 = yn0;
        yn0 = saturated;
        return static_cast<s16>(saturated);
    };

    u32 read_index = 0;
    for (u32 write_index = 0; samples_to_read > 0 && write_index < out_limit;) {
        if ((position_in_frame % NibblesPerFrame) == 0) {
            header = wave_ptr[read_index++];
            scale_shift = (header & 0xFU) + 11U;
            coeff_index = static_cast<u8>((header >> 4U) & 0x7U);
            coeff0 = req.coefficients[coeff_index * 2 + 0];
            coeff1 = req.coefficients[coeff_index * 2 + 1];
            position_in_frame += 2;

            if (samples_to_read >= SamplesPerFrame && write_index + SamplesPerFrame <= out_limit) {
                for (u32 i = 0; i < SamplesPerFrame / 2; i++) {
                    const u8 byte = wave_ptr[read_index + i];
                    out_ptr[write_index++] = decode_sample(high_nibble(byte));
                    out_ptr[write_index++] = decode_sample(low_nibble(byte));
                }
                read_index += SamplesPerFrame / 2;
                position_in_frame += SamplesPerFrame;
                samples_to_read -= SamplesPerFrame;
                continue;
            }
        }

        const u8 byte = wave_ptr[read_index];
        s32 step{};
        if (position_in_frame & 1) {
            step = low_nibble(byte);
            read_index++;
        } else {
            step = high_nibble(byte);
        }
        out_ptr[write_index++] = decode_sample(step);

        position_in_frame++;
        samples_to_read--;
    }

    context->header = header;
    context->yn0 = static_cast<s16>(yn0);
    context->yn1 = static_cast<s16>(yn1);
    return samples_to_process;
}

void DecodeFromWaveBuffers(Core::Memory::Memory& memory, const DecodeFromWaveBuffersArgs& args) {
    constexpr auto EndWaveBuffer = [](auto& voice_state, auto& wavebuffer, auto& index,
                                      auto& played_samples, auto& consumed) -> void {
        voice_state.wave_buffer_valid[index] = false;
        voice_state.loop_count = 0;
        if (wavebuffer.stream_ended) {
            played_samples = 0;
        }
        index = (index + 1) % MaxWaveBuffers;
        consumed++;
    };
    auto& voice_state{*args.voice_state};
    auto remaining_sample_count{args.sample_count};
    auto fraction{voice_state.fraction};

    const auto sample_rate_ratio{Common::FixedPoint<49, 15>(
        static_cast<f32>(args.source_sample_rate) / static_cast<f32>(args.target_sample_rate) *
        args.pitch)};
    const auto size_required{fraction + remaining_sample_count * sample_rate_ratio};

    if (size_required < 0) {
        return;
    }

    // 0 -> 4, 1 -> 8, 2 -> 4
    const auto pitch = u32((0x040804ul >> (u32(args.src_quality) * 8)) & 0xfful);
    if (u32(pitch + size_required.to_int_floor()) > TempBufferSize) {
        return;
    }

    auto max_remaining_sample_count{
        ((Common::FixedPoint<17, 15>(TempBufferSize) - fraction) / sample_rate_ratio)
            .to_uint_floor()};
    max_remaining_sample_count = (std::min)(max_remaining_sample_count, remaining_sample_count);

    auto wavebuffers_consumed{voice_state.wave_buffers_consumed};
    auto wavebuffer_index{voice_state.wave_buffer_index};
    auto played_sample_count{voice_state.played_sample_count};

    bool is_buffer_starved{false};
    u32 offset{voice_state.offset};

    std::array<s16, 16> adpcm_coeffs{};
    if (args.sample_format == SampleFormat::Adpcm && args.data_address != 0 && args.data_size > 0) {
        memory.ReadBlockUnsafe(args.data_address, adpcm_coeffs.data(),
                               (std::min<u64>)(args.data_size, sizeof(adpcm_coeffs)));
    }

    auto output_buffer = args.output;
    // Uninitialized scratch buffer: only `[0 .. pitch + samples_to_read)` is read, avoiding a
    // 32 KiB memset on every voice channel per 5 ms frame.
    std::array<s16, TempBufferSize> temp_buffer;

    while (remaining_sample_count > 0) {
        const auto samples_to_write{(std::min)(remaining_sample_count, max_remaining_sample_count)};
        const auto samples_to_read{
            (fraction + samples_to_write * sample_rate_ratio).to_uint_floor()};

        u32 temp_buffer_pos{0};

        if (!args.IsVoicePitchAndSrcSkippedSupported) {
            std::memcpy(temp_buffer.data(), voice_state.sample_history.data(), pitch * sizeof(s16));
            temp_buffer_pos = pitch;
        }

        u32 samples_read{0};
        while (samples_read < samples_to_read) {
            if (wavebuffer_index >= MaxWaveBuffers) {
                LOG_ERROR(Service_Audio, "Invalid wavebuffer index! {}", wavebuffer_index);
                wavebuffer_index = 0;
                wavebuffers_consumed = MaxWaveBuffers;
                voice_state.wave_buffer_valid.fill(false);
            }

            if (!voice_state.wave_buffer_valid[wavebuffer_index]) {
                is_buffer_starved = true;
                break;
            }

            auto& wavebuffer = args.wave_buffers[wavebuffer_index];
            if (offset == 0 && args.sample_format == SampleFormat::Adpcm &&
                wavebuffer.context != 0) {
                memory.ReadBlockUnsafe(wavebuffer.context, &voice_state.adpcm_context,
                                       wavebuffer.context_size);
            }

            auto start_offset{wavebuffer.start_offset};
            auto end_offset{wavebuffer.end_offset};

            if (wavebuffer.loop && voice_state.loop_count > 0 &&
                wavebuffer.loop_start_offset <= wavebuffer.loop_end_offset) {
                start_offset = wavebuffer.loop_start_offset;
                end_offset = wavebuffer.loop_end_offset;
            }

            DecodeArg decode_arg{
                .buffer{wavebuffer.buffer},
                .buffer_size{wavebuffer.buffer_size},
                .start_offset{start_offset},
                .end_offset{end_offset},
                .channel_count{args.channel_count},
                .coefficients{adpcm_coeffs},
                .adpcm_context{nullptr},
                .target_channel{args.channel},
                .offset{offset},
                .samples_to_read{samples_to_read - samples_read},
            };

            u32 samples_decoded{0};

            switch (args.sample_format) {
            case SampleFormat::PcmInt16:
                samples_decoded = DecodePcm<s16>(
                    memory, {&temp_buffer[temp_buffer_pos], TempBufferSize - temp_buffer_pos},
                    decode_arg);
                break;

            case SampleFormat::PcmFloat:
                samples_decoded = DecodePcm<f32>(
                    memory, {&temp_buffer[temp_buffer_pos], TempBufferSize - temp_buffer_pos},
                    decode_arg);
                break;

            case SampleFormat::Adpcm:
                decode_arg.adpcm_context = &voice_state.adpcm_context;
                samples_decoded = DecodeAdpcm(
                    memory, {&temp_buffer[temp_buffer_pos], TempBufferSize - temp_buffer_pos},
                    decode_arg);
                break;

            default:
                LOG_ERROR(Service_Audio, "Invalid sample format to decode {}",
                          static_cast<u32>(args.sample_format));
                samples_decoded = 0;
                break;
            }

            played_sample_count += samples_decoded;
            samples_read += samples_decoded;
            temp_buffer_pos += samples_decoded;
            offset += samples_decoded;

            if (samples_decoded && offset < end_offset - start_offset) {
                continue;
            }

            offset = 0;
            if (wavebuffer.loop) {
                voice_state.loop_count++;
                if (wavebuffer.loop_count >= 0 &&
                    (voice_state.loop_count > wavebuffer.loop_count || samples_decoded == 0)) {
                    EndWaveBuffer(voice_state, wavebuffer, wavebuffer_index, played_sample_count,
                                  wavebuffers_consumed);
                }

                if (samples_decoded == 0) {
                    is_buffer_starved = true;
                    break;
                }

                if (args.IsVoicePlayedSampleCountResetAtLoopPointSupported) {
                    played_sample_count = 0;
                }
            } else {
                EndWaveBuffer(voice_state, wavebuffer, wavebuffer_index, played_sample_count,
                              wavebuffers_consumed);
            }
        }

        if (args.IsVoicePitchAndSrcSkippedSupported) {
            const u32 to_copy =
                (std::min)(samples_read, static_cast<u32>(output_buffer.size()));
            if (samples_read > output_buffer.size()) {
                LOG_ERROR(Service_Audio, "Attempting to write past the end of output buffer!");
            }
            s32* __restrict dst = output_buffer.data();
            const s16* __restrict src = temp_buffer.data();
            for (u32 i = 0; i < to_copy; i++) {
                dst[i] = src[i];
            }
        } else {
            if (samples_to_read > samples_read) {
                std::memset(&temp_buffer[temp_buffer_pos], 0,
                            (samples_to_read - samples_read) * sizeof(s16));
            }

            Resample(output_buffer, temp_buffer, sample_rate_ratio, fraction, samples_to_write,
                     args.src_quality);

            std::memcpy(voice_state.sample_history.data(), &temp_buffer[samples_to_read],
                        pitch * sizeof(s16));
        }

        remaining_sample_count -= samples_to_write;
        if (remaining_sample_count != 0 && is_buffer_starved) {
            LOG_ERROR(Service_Audio, "Samples remaining but buffer is starving??");
            break;
        }

        output_buffer = output_buffer.subspan(samples_to_write);
    }

    voice_state.wave_buffers_consumed = wavebuffers_consumed;
    voice_state.played_sample_count = played_sample_count;
    voice_state.wave_buffer_index = wavebuffer_index;
    voice_state.offset = offset;
    voice_state.fraction = fraction;
}

} // namespace AudioCore::Renderer
