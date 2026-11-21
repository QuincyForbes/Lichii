// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>

#include "common/common_types.h"
#include "common/fixed_point.h"

#if defined(ARCHITECTURE_arm64)
#include <arm_neon.h>
#elif defined(ARCHITECTURE_x86_64)
#include <emmintrin.h>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace AudioCore::DspSimd {

/// Exact 64x64->128 fixed-point multiply matching Common::FixedPoint<64 - Q, Q>::operator*
/// without triggering the 4-multiply software fallback on ARM64 or MSVC.
template <size_t Q>
[[nodiscard]] inline s64 MulFp(const s64 lhs, const s64 rhs) noexcept {
    static_assert(Q > 0 && Q < 64);
#if defined(__SIZEOF_INT128__)
    return static_cast<s64>((static_cast<__int128>(lhs) * static_cast<__int128>(rhs)) >> Q);
#elif defined(_MSC_VER) && defined(_M_X64)
    __int64 hi = 0;
    const __int64 lo = _mul128(lhs, rhs, &hi);
    return static_cast<s64>((static_cast<u64>(lo) >> Q) | (static_cast<u64>(hi) << (64 - Q)));
#elif defined(_MSC_VER) && defined(_M_ARM64)
    const u64 lo = static_cast<u64>(lhs) * static_cast<u64>(rhs);
    const s64 hi = __mulh(lhs, rhs);
    return static_cast<s64>((lo >> Q) | (static_cast<u64>(hi) << (64 - Q)));
#else
    return (Common::FixedPoint<64 - Q, Q>::from_base(lhs) *
            Common::FixedPoint<64 - Q, Q>::from_base(rhs))
        .to_raw();
#endif
}

/// Exact fixed-point rounding matching Common::FixedPoint<64 - Q, Q>::to_int().
template <size_t Q>
[[nodiscard]] constexpr s32 RoundFpToInt(const s64 raw) noexcept {
    static_assert(Q > 0 && Q <= 32);
    constexpr s64 frac_mask = (s64{1} << Q) - 1;
    return static_cast<s32>((raw + ((raw & frac_mask) >> 1)) >> Q);
}

/// Fast SIMD check whether `data[0..count)` contains only zeros. Exits on first non-zero vector.
[[nodiscard]] inline bool IsAllZeroS32(const s32* data, const u32 count) noexcept {
    u32 i = 0;
#if defined(ARCHITECTURE_arm64)
    for (; i + 8 <= count; i += 8) {
        const uint32x4_t v0 = vreinterpretq_u32_s32(vld1q_s32(data + i));
        const uint32x4_t v1 = vreinterpretq_u32_s32(vld1q_s32(data + i + 4));
        if (vmaxvq_u32(vorrq_u32(v0, v1)) != 0) {
            return false;
        }
    }
#elif defined(ARCHITECTURE_x86_64)
    const __m128i zero = _mm_setzero_si128();
    for (; i + 8 <= count; i += 8) {
        const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i + 4));
        const __m128i eq = _mm_cmpeq_epi32(_mm_or_si128(v0, v1), zero);
        if (_mm_movemask_epi8(eq) != 0xFFFF) {
            return false;
        }
    }
#endif
    for (; i < count; ++i) {
        if (data[i] != 0) {
            return false;
        }
    }
    return true;
}

/// Vectorized `dst[i] += src[i]` for unity-gain mixing (`volume == 1.0f`).
inline void AddBuffersS32(s32* dst, const s32* src, const u32 count) noexcept {
    u32 i = 0;
    if (dst != src) {
        s32* __restrict d = dst;
        const s32* __restrict s = src;
#if defined(ARCHITECTURE_arm64)
        for (; i + 8 <= count; i += 8) {
            vst1q_s32(d + i, vaddq_s32(vld1q_s32(d + i), vld1q_s32(s + i)));
            vst1q_s32(d + i + 4, vaddq_s32(vld1q_s32(d + i + 4), vld1q_s32(s + i + 4)));
        }
#elif defined(ARCHITECTURE_x86_64)
        for (; i + 8 <= count; i += 8) {
            const __m128i d0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
            const __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i));
            const __m128i d1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i + 4));
            const __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i + 4));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), _mm_add_epi32(d0, s0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i + 4), _mm_add_epi32(d1, s1));
        }
#endif
        for (; i < count; ++i) {
            d[i] = static_cast<s32>(static_cast<u32>(d[i]) + static_cast<u32>(s[i]));
        }
    } else {
        for (; i < count; ++i) {
            dst[i] = static_cast<s32>(static_cast<u32>(dst[i]) << 1);
        }
    }
}

namespace detail {

[[nodiscard]] constexpr bool FitsInS32(const s64 v) noexcept {
    return v >= static_cast<s64>((std::numeric_limits<s32>::min)()) &&
           v <= static_cast<s64>((std::numeric_limits<s32>::max)());
}

#if defined(ARCHITECTURE_arm64)
template <size_t Q>
[[nodiscard]] inline int32x4_t ScaleSamplesNeon(const int32x4_t samples,
                                                const int32x4_t gains) noexcept {
    static_assert(Q > 0 && Q <= 32);
    constexpr u64 frac_mask = (u64{1} << Q) - 1;
    const uint64x2_t v_mask = vdupq_n_u64(frac_mask);

    const uint64x2_t p_lo =
        vreinterpretq_u64_s64(vmull_s32(vget_low_s32(samples), vget_low_s32(gains)));
    const uint64x2_t p_hi = vreinterpretq_u64_s64(vmull_high_s32(samples, gains));

    const uint64x2_t r_lo = vshrq_n_u64(vandq_u64(p_lo, v_mask), 1);
    const uint64x2_t r_hi = vshrq_n_u64(vandq_u64(p_hi, v_mask), 1);

    // Q <= 32 guarantees bits [Q .. Q + 31] lie entirely within [0 .. 63], so logical right shift
    // yields the exact same low 32 bits as arithmetic right shift.
    const uint64x2_t s_lo = vshrq_n_u64(vaddq_u64(p_lo, r_lo), Q);
    const uint64x2_t s_hi = vshrq_n_u64(vaddq_u64(p_hi, r_hi), Q);

    return vreinterpretq_s32_u32(
        vuzp1q_u32(vreinterpretq_u32_u64(s_lo), vreinterpretq_u32_u64(s_hi)));
}
#elif defined(ARCHITECTURE_x86_64) && defined(__AVX2__)
template <size_t Q>
[[nodiscard]] inline __m256i ScaleSamplesAvx2(const __m256i samples, const __m256i gains) noexcept {
    static_assert(Q > 0 && Q <= 32);
    constexpr s64 frac_mask = (s64{1} << Q) - 1;
    const __m256i v_mask = _mm256_set1_epi64x(frac_mask);

    const __m256i p_even = _mm256_mul_epi32(samples, gains);
    const __m256i p_odd =
        _mm256_mul_epi32(_mm256_srli_si256(samples, 4), _mm256_srli_si256(gains, 4));

    const __m256i r_even = _mm256_srli_epi64(_mm256_and_si256(p_even, v_mask), 1);
    const __m256i r_odd = _mm256_srli_epi64(_mm256_and_si256(p_odd, v_mask), 1);

    const __m256i s_even = _mm256_srli_epi64(_mm256_add_epi64(p_even, r_even), Q);
    const __m256i s_odd =
        _mm256_slli_epi64(_mm256_srli_epi64(_mm256_add_epi64(p_odd, r_odd), Q), 32);

    return _mm256_blend_epi32(s_even, s_odd, 0xAA);
}
#elif defined(ARCHITECTURE_x86_64) && defined(__SSE4_1__)
template <size_t Q>
[[nodiscard]] inline __m128i ScaleSamplesSse41(const __m128i samples,
                                               const __m128i gains) noexcept {
    static_assert(Q > 0 && Q <= 32);
    constexpr s64 frac_mask = (s64{1} << Q) - 1;
    const __m128i v_mask = _mm_set1_epi64x(frac_mask);

    const __m128i p_even = _mm_mul_epi32(samples, gains);
    const __m128i p_odd = _mm_mul_epi32(_mm_srli_si128(samples, 4), _mm_srli_si128(gains, 4));

    const __m128i r_even = _mm_srli_epi64(_mm_and_si128(p_even, v_mask), 1);
    const __m128i r_odd = _mm_srli_epi64(_mm_and_si128(p_odd, v_mask), 1);

    const __m128i s_even = _mm_srli_epi64(_mm_add_epi64(p_even, r_even), Q);
    const __m128i s_odd = _mm_slli_epi64(_mm_srli_epi64(_mm_add_epi64(p_odd, r_odd), Q), 32);

    return _mm_blend_epi16(s_even, s_odd, 0xCC);
}
#endif

} // namespace detail

/// Mix `src * volume` into `dst` (`dst[i] += RoundFpToInt<Q>(src[i] * gain)`).
template <size_t Q>
inline void ApplyGainMix(s32* dst, const s32* src, const f32 volume_f, const u32 count) noexcept {
    if (count == 0) {
        return;
    }
    const s64 gain_raw = Common::FixedPoint<64 - Q, Q>{volume_f}.to_raw();
    if (gain_raw == 0 || IsAllZeroS32(src, count)) {
        return;
    }
    if (gain_raw == (s64{1} << Q)) {
        AddBuffersS32(dst, src, count);
        return;
    }

    u32 i = 0;
    if (detail::FitsInS32(gain_raw)) {
        const s32 g32 = static_cast<s32>(gain_raw);
#if defined(ARCHITECTURE_arm64)
        const int32x4_t vg = vdupq_n_s32(g32);
        for (; i + 8 <= count; i += 8) {
            const int32x4_t s0 = detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i), vg);
            const int32x4_t s1 = detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i + 4), vg);
            vst1q_s32(dst + i, vaddq_s32(vld1q_s32(dst + i), s0));
            vst1q_s32(dst + i + 4, vaddq_s32(vld1q_s32(dst + i + 4), s1));
        }
        for (; i + 4 <= count; i += 4) {
            const int32x4_t s0 = detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i), vg);
            vst1q_s32(dst + i, vaddq_s32(vld1q_s32(dst + i), s0));
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__AVX2__)
        const __m256i vg = _mm256_set1_epi32(g32);
        for (; i + 8 <= count; i += 8) {
            const __m256i in_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            const __m256i out_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
            const __m256i scaled = detail::ScaleSamplesAvx2<Q>(in_v, vg);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                                _mm256_add_epi32(out_v, scaled));
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__SSE4_1__)
        const __m128i vg = _mm_set1_epi32(g32);
        for (; i + 4 <= count; i += 4) {
            const __m128i in_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            const __m128i out_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
            const __m128i scaled = detail::ScaleSamplesSse41<Q>(in_v, vg);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_add_epi32(out_v, scaled));
        }
#else
        (void)g32;
#endif
    }

    if (dst != src) {
        s32* __restrict d = dst;
        const s32* __restrict s = src;
        for (; i < count; ++i) {
            const u32 scaled = static_cast<u32>(RoundFpToInt<Q>(s64{s[i]} * gain_raw));
            d[i] = static_cast<s32>(static_cast<u32>(d[i]) + scaled);
        }
    } else {
        for (; i < count; ++i) {
            const u32 scaled = static_cast<u32>(RoundFpToInt<Q>(s64{dst[i]} * gain_raw));
            dst[i] = static_cast<s32>(static_cast<u32>(dst[i]) + scaled);
        }
    }
}

/// Write `src * volume` into `dst` (`dst[i] = RoundFpToInt<Q>(src[i] * gain)`).
template <size_t Q>
inline void ApplyGainReplace(s32* dst, const s32* src, const f32 volume_f,
                             const u32 count) noexcept {
    if (count == 0) {
        return;
    }
    const s64 gain_raw = Common::FixedPoint<64 - Q, Q>{volume_f}.to_raw();
    if (gain_raw == 0) {
        std::memset(dst, 0, static_cast<size_t>(count) * sizeof(s32));
        return;
    }
    if (gain_raw == (s64{1} << Q)) {
        if (dst != src) {
            std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(s32));
        }
        return;
    }
    if (IsAllZeroS32(src, count)) {
        if (dst != src) {
            std::memset(dst, 0, static_cast<size_t>(count) * sizeof(s32));
        }
        return;
    }

    u32 i = 0;
    if (detail::FitsInS32(gain_raw)) {
        const s32 g32 = static_cast<s32>(gain_raw);
#if defined(ARCHITECTURE_arm64)
        const int32x4_t vg = vdupq_n_s32(g32);
        for (; i + 8 <= count; i += 8) {
            vst1q_s32(dst + i, detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i), vg));
            vst1q_s32(dst + i + 4, detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i + 4), vg));
        }
        for (; i + 4 <= count; i += 4) {
            vst1q_s32(dst + i, detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i), vg));
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__AVX2__)
        const __m256i vg = _mm256_set1_epi32(g32);
        for (; i + 8 <= count; i += 8) {
            const __m256i in_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                                detail::ScaleSamplesAvx2<Q>(in_v, vg));
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__SSE4_1__)
        const __m128i vg = _mm_set1_epi32(g32);
        for (; i + 4 <= count; i += 4) {
            const __m128i in_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                             detail::ScaleSamplesSse41<Q>(in_v, vg));
        }
#else
        (void)g32;
#endif
    }

    if (dst != src) {
        s32* __restrict d = dst;
        const s32* __restrict s = src;
        for (; i < count; ++i) {
            d[i] = RoundFpToInt<Q>(s64{s[i]} * gain_raw);
        }
    } else {
        for (; i < count; ++i) {
            dst[i] = RoundFpToInt<Q>(s64{dst[i]} * gain_raw);
        }
    }
}

/// Mix ramped `src * (volume + i * ramp)` into `dst` and return the final scaled sample.
template <size_t Q>
[[nodiscard]] inline s32 ApplyGainRampMix(s32* dst, const s32* src, const f32 volume_f,
                                          const f32 ramp_f, const u32 count) noexcept {
    if (count == 0) {
        return 0;
    }
    const s64 gain0_raw = Common::FixedPoint<64 - Q, Q>{volume_f}.to_raw();
    const s64 ramp_raw = ramp_f == 0.0f ? s64{0} : Common::FixedPoint<64 - Q, Q>{ramp_f}.to_raw();

    if (ramp_raw == 0) {
        if (gain0_raw == 0 || IsAllZeroS32(src, count)) {
            return 0;
        }
        const s32 last_in = src[count - 1];
        ApplyGainMix<Q>(dst, src, volume_f, count);
        return RoundFpToInt<Q>(s64{last_in} * gain0_raw);
    }

    if (IsAllZeroS32(src, count)) {
        return 0;
    }

    const s32 last_in = src[count - 1];
    const s64 last_gain_raw = gain0_raw + s64{count - 1} * ramp_raw;
    const s64 end_gain_raw = gain0_raw + s64{count} * ramp_raw;
    const s64 init8_end_raw = gain0_raw + ramp_raw * 8;
    const s64 step8_raw = ramp_raw * 8;

    u32 i = 0;
    if (detail::FitsInS32(gain0_raw) && detail::FitsInS32(end_gain_raw) &&
        detail::FitsInS32(init8_end_raw) && detail::FitsInS32(step8_raw)) {
        const auto lane_gain = [&](const s64 lane) noexcept -> s32 {
            return static_cast<s32>(gain0_raw + ramp_raw * lane);
        };
#if defined(ARCHITECTURE_arm64)
        const s32 init[4]{lane_gain(0), lane_gain(1), lane_gain(2), lane_gain(3)};
        int32x4_t vg = vld1q_s32(init);
        const int32x4_t vstep4 = vdupq_n_s32(static_cast<s32>(ramp_raw * 4));
        for (; i + 4 <= count; i += 4) {
            const int32x4_t scaled = detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i), vg);
            vst1q_s32(dst + i, vaddq_s32(vld1q_s32(dst + i), scaled));
            vg = vaddq_s32(vg, vstep4);
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__AVX2__)
        __m256i vg = _mm256_setr_epi32(lane_gain(0), lane_gain(1), lane_gain(2), lane_gain(3),
                                       lane_gain(4), lane_gain(5), lane_gain(6), lane_gain(7));
        const __m256i vstep8 = _mm256_set1_epi32(static_cast<s32>(step8_raw));
        for (; i + 8 <= count; i += 8) {
            const __m256i in_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            const __m256i out_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
            const __m256i scaled = detail::ScaleSamplesAvx2<Q>(in_v, vg);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                                _mm256_add_epi32(out_v, scaled));
            vg = _mm256_add_epi32(vg, vstep8);
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__SSE4_1__)
        __m128i vg = _mm_setr_epi32(lane_gain(0), lane_gain(1), lane_gain(2), lane_gain(3));
        const __m128i vstep4 = _mm_set1_epi32(static_cast<s32>(ramp_raw * 4));
        for (; i + 4 <= count; i += 4) {
            const __m128i in_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            const __m128i out_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
            const __m128i scaled = detail::ScaleSamplesSse41<Q>(in_v, vg);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_add_epi32(out_v, scaled));
            vg = _mm_add_epi32(vg, vstep4);
        }
#else
        (void)lane_gain;
#endif
    }

    s64 gain_cur = gain0_raw + s64{i} * ramp_raw;
    for (; i < count; ++i) {
        const u32 scaled = static_cast<u32>(RoundFpToInt<Q>(s64{src[i]} * gain_cur));
        dst[i] = static_cast<s32>(static_cast<u32>(dst[i]) + scaled);
        gain_cur += ramp_raw;
    }

    return RoundFpToInt<Q>(s64{last_in} * last_gain_raw);
}

/// Write ramped `src * (volume + i * ramp)` into `dst`.
template <size_t Q>
inline void ApplyGainRampReplace(s32* dst, const s32* src, const f32 volume_f, const f32 ramp_f,
                                 const u32 count) noexcept {
    if (count == 0) {
        return;
    }
    const s64 gain0_raw = Common::FixedPoint<64 - Q, Q>{volume_f}.to_raw();
    const s64 ramp_raw = ramp_f == 0.0f ? s64{0} : Common::FixedPoint<64 - Q, Q>{ramp_f}.to_raw();

    if (ramp_raw == 0) {
        ApplyGainReplace<Q>(dst, src, volume_f, count);
        return;
    }

    if (IsAllZeroS32(src, count)) {
        if (dst != src) {
            std::memset(dst, 0, static_cast<size_t>(count) * sizeof(s32));
        }
        return;
    }

    const s64 end_gain_raw = gain0_raw + s64{count} * ramp_raw;
    const s64 init8_end_raw = gain0_raw + ramp_raw * 8;
    const s64 step8_raw = ramp_raw * 8;

    u32 i = 0;
    if (detail::FitsInS32(gain0_raw) && detail::FitsInS32(end_gain_raw) &&
        detail::FitsInS32(init8_end_raw) && detail::FitsInS32(step8_raw)) {
        const auto lane_gain = [&](const s64 lane) noexcept -> s32 {
            return static_cast<s32>(gain0_raw + ramp_raw * lane);
        };
#if defined(ARCHITECTURE_arm64)
        const s32 init[4]{lane_gain(0), lane_gain(1), lane_gain(2), lane_gain(3)};
        int32x4_t vg = vld1q_s32(init);
        const int32x4_t vstep4 = vdupq_n_s32(static_cast<s32>(ramp_raw * 4));
        for (; i + 4 <= count; i += 4) {
            vst1q_s32(dst + i, detail::ScaleSamplesNeon<Q>(vld1q_s32(src + i), vg));
            vg = vaddq_s32(vg, vstep4);
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__AVX2__)
        __m256i vg = _mm256_setr_epi32(lane_gain(0), lane_gain(1), lane_gain(2), lane_gain(3),
                                       lane_gain(4), lane_gain(5), lane_gain(6), lane_gain(7));
        const __m256i vstep8 = _mm256_set1_epi32(static_cast<s32>(step8_raw));
        for (; i + 8 <= count; i += 8) {
            const __m256i in_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                                detail::ScaleSamplesAvx2<Q>(in_v, vg));
            vg = _mm256_add_epi32(vg, vstep8);
        }
#elif defined(ARCHITECTURE_x86_64) && defined(__SSE4_1__)
        __m128i vg = _mm_setr_epi32(lane_gain(0), lane_gain(1), lane_gain(2), lane_gain(3));
        const __m128i vstep4 = _mm_set1_epi32(static_cast<s32>(ramp_raw * 4));
        for (; i + 4 <= count; i += 4) {
            const __m128i in_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                             detail::ScaleSamplesSse41<Q>(in_v, vg));
            vg = _mm_add_epi32(vg, vstep4);
        }
#else
        (void)lane_gain;
#endif
    }

    s64 gain_cur = gain0_raw + s64{i} * ramp_raw;
    for (; i < count; ++i) {
        dst[i] = RoundFpToInt<Q>(s64{src[i]} * gain_cur);
        gain_cur += ramp_raw;
    }
}

/// Saturate `s32` samples to `[-32768, 32767]` `s16`.
inline void SaturateS32ToS16(s16* __restrict dst, const s32* __restrict src,
                             const u32 count) noexcept {
    constexpr s32 min_val = (std::numeric_limits<s16>::min)();
    constexpr s32 max_val = (std::numeric_limits<s16>::max)();
    u32 i = 0;
#if defined(ARCHITECTURE_arm64)
    for (; i + 8 <= count; i += 8) {
        const int16x4_t lo = vqmovn_s32(vld1q_s32(src + i));
        const int16x8_t both = vqmovn_high_s32(lo, vld1q_s32(src + i + 4));
        vst1q_s16(dst + i, both);
    }
#elif defined(ARCHITECTURE_x86_64)
    for (; i + 8 <= count; i += 8) {
        const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 4));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_packs_epi32(lo, hi));
    }
#endif
    for (; i < count; ++i) {
        dst[i] = static_cast<s16>(std::clamp(src[i], min_val, max_val));
    }
}

/// Saturate two planar `s32` channels and interleave into stereo `s16` (`L0, R0, L1, R1, ...`).
inline void SaturateInterleaveStereoS32ToS16(s16* __restrict dst, const s32* __restrict left,
                                             const s32* __restrict right,
                                             const u32 frames) noexcept {
    constexpr s32 min_val = (std::numeric_limits<s16>::min)();
    constexpr s32 max_val = (std::numeric_limits<s16>::max)();
    u32 i = 0;
#if defined(ARCHITECTURE_arm64)
    for (; i + 8 <= frames; i += 8) {
        int16x8x2_t pair;
        pair.val[0] = vqmovn_high_s32(vqmovn_s32(vld1q_s32(left + i)), vld1q_s32(left + i + 4));
        pair.val[1] = vqmovn_high_s32(vqmovn_s32(vld1q_s32(right + i)), vld1q_s32(right + i + 4));
        vst2q_s16(dst + i * 2, pair);
    }
#elif defined(ARCHITECTURE_x86_64)
    for (; i + 8 <= frames; i += 8) {
        const __m128i l8 =
            _mm_packs_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(left + i)),
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + i + 4)));
        const __m128i r8 =
            _mm_packs_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(right + i)),
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(right + i + 4)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * 2), _mm_unpacklo_epi16(l8, r8));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * 2 + 8), _mm_unpackhi_epi16(l8, r8));
    }
#endif
    for (; i < frames; ++i) {
        dst[i * 2] = static_cast<s16>(std::clamp(left[i], min_val, max_val));
        dst[i * 2 + 1] = static_cast<s16>(std::clamp(right[i], min_val, max_val));
    }
}

} // namespace AudioCore::DspSimd
