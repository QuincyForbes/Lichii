// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <string_view>

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace VideoCommon {

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available,
                                                    std::string_view override) {
    if (override == "native") {
        return OcarinaUpscaleBackend::Native;
    }
    if (override == "fsr") {
        return OcarinaUpscaleBackend::Fsr;
    }
    if ((dlss_enabled || override == "dlss") && dlss_available) {
        return OcarinaUpscaleBackend::Dlss;
    }
    return fsr_enabled ? OcarinaUpscaleBackend::Fsr : OcarinaUpscaleBackend::Native;
}

/// Whether the "Ocarina of Time anti-aliasing" choice asks for AMD FSR 3.1.
constexpr bool OcarinaFsrChosen(Settings::OotAntiAliasing choice) {
    return choice == Settings::OotAntiAliasing::Fsr3;
}

/// The backend for the "Ocarina of Time anti-aliasing" choice and the DLSS (or MetalFX) setting.
/// An FSR choice wins; with Game or No sharpening, DLSS replaces the game's pass when it's on and
/// available (as before FSR came in).
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(Settings::OotAntiAliasing choice,
                                                    bool dlss_enabled, bool dlss_available,
                                                    std::string_view override) {
    const bool fsr = OcarinaFsrChosen(choice);
    return SelectOcarinaUpscaler(fsr, dlss_enabled && !fsr, dlss_available, override);
}

/// EDEN_NX2_DLSS_JITTER (boot, 0 = off): DLSS or MetalFX gets the camera jitter the FSR path uses.
/// 1 = the jitter as FSR gets it, 2 = with its vertical sign flipped.
inline int OcarinaDlssJitterMode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_DLSS_JITTER");
        return value != nullptr ? std::atoi(value) : 0;
    }();
    return mode;
}

/// Whether Ocarina's scene draws get a sub-pixel offset each frame for that backend: FSR unless
/// EDEN_NX2_OOT_JITTER=0 (boot), DLSS or MetalFX only with EDEN_NX2_DLSS_JITTER.
inline bool OcarinaJitterEnabled(OcarinaUpscaleBackend backend) {
    static const bool fsr_jitter = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_JITTER");
        return value == nullptr || std::string_view{value} != "0";
    }();
    switch (backend) {
    case OcarinaUpscaleBackend::Fsr:
        return fsr_jitter;
    case OcarinaUpscaleBackend::Dlss:
        return OcarinaDlssJitterMode() != 0;
    case OcarinaUpscaleBackend::Native:
        break;
    }
    return false;
}

/// AMD's jitter phase count for FSR: 8 * (output / render width)^2, so 18 for 1440p -> 4K and 8
/// for native anti-aliasing.
constexpr u32 OcarinaJitterPhases(u32 render_width, u32 output_width) {
    if (render_width == 0) {
        return 8;
    }
    const float ratio = static_cast<float>(output_width) / static_cast<float>(render_width);
    const u32 phases = static_cast<u32>(8.0f * ratio * ratio);
    return phases != 0 ? phases : 1;
}

} // namespace VideoCommon
