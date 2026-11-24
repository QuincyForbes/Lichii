// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstring>

#include "common/common_types.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace Common {

/// True when all `size` bytes at `data` are zero. Checks 4 KiB at a time (memory bound).
[[nodiscard]] inline bool IsAllZero(const u8* data, std::size_t size) {
#if defined(_M_X64) || defined(__x86_64__)
    for (; size >= 4096; data += 4096, size -= 4096) {
        __m128i acc0 = _mm_setzero_si128();
        __m128i acc1 = _mm_setzero_si128();
        __m128i acc2 = _mm_setzero_si128();
        __m128i acc3 = _mm_setzero_si128();
        for (std::size_t i = 0; i < 4096; i += 64) {
            const auto* p = reinterpret_cast<const __m128i*>(data + i);
            acc0 = _mm_or_si128(acc0, _mm_loadu_si128(p));
            acc1 = _mm_or_si128(acc1, _mm_loadu_si128(p + 1));
            acc2 = _mm_or_si128(acc2, _mm_loadu_si128(p + 2));
            acc3 = _mm_or_si128(acc3, _mm_loadu_si128(p + 3));
        }
        const __m128i acc = _mm_or_si128(_mm_or_si128(acc0, acc1), _mm_or_si128(acc2, acc3));
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF) {
            return false;
        }
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    for (; size >= 4096; data += 4096, size -= 4096) {
        uint64x2_t acc0 = vdupq_n_u64(0);
        uint64x2_t acc1 = vdupq_n_u64(0);
        uint64x2_t acc2 = vdupq_n_u64(0);
        uint64x2_t acc3 = vdupq_n_u64(0);
        for (std::size_t i = 0; i < 4096; i += 64) {
            acc0 = vorrq_u64(acc0, vreinterpretq_u64_u8(vld1q_u8(data + i)));
            acc1 = vorrq_u64(acc1, vreinterpretq_u64_u8(vld1q_u8(data + i + 16)));
            acc2 = vorrq_u64(acc2, vreinterpretq_u64_u8(vld1q_u8(data + i + 32)));
            acc3 = vorrq_u64(acc3, vreinterpretq_u64_u8(vld1q_u8(data + i + 48)));
        }
        const uint64x2_t acc = vorrq_u64(vorrq_u64(acc0, acc1), vorrq_u64(acc2, acc3));
        if (vmaxvq_u32(vreinterpretq_u32_u64(acc)) != 0) {
            return false;
        }
    }
#endif
    for (; size >= 8; data += 8, size -= 8) {
        u64 word;
        std::memcpy(&word, data, sizeof(word));
        if (word != 0) {
            return false;
        }
    }
    for (; size > 0; ++data, --size) {
        if (*data != 0) {
            return false;
        }
    }
    return true;
}

} // namespace Common
