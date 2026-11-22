// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if defined(_MSC_VER)
#define NX2_FRONT_INLINE __forceinline
#else
#define NX2_FRONT_INLINE inline __attribute__((always_inline))
#endif

#if defined(__clang__) || defined(__GNUC__)
#define NX2_FRONT_LAMBDA_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NX2_FRONT_LAMBDA_INLINE [[msvc::forceinline]]
#else
#define NX2_FRONT_LAMBDA_INLINE
#endif

namespace VideoCommon {

constexpr bool FrontLeanEnabled() {
    return true;
}

} // namespace VideoCommon
