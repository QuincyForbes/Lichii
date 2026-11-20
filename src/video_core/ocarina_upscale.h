// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/textures/texture.h"

namespace Settings {
std::optional<DlssQuality> AppliedOcarinaRenderScale();
}

namespace VideoCommon::OcarinaUpscale {

[[nodiscard]] constexpr f32 Halton(u32 index, u32 base) noexcept {
    f32 result = 0.0f;
    f32 fraction = 1.0f;
    while (index > 0) {
        fraction /= static_cast<f32>(base);
        result += fraction * static_cast<f32>(index % base);
        index /= base;
    }
    return result;
}

} // namespace VideoCommon::OcarinaUpscale

namespace VideoCommon {

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

/// The game's output passes the temporal upscaler can replace.
enum class OcarinaUpscalePass {
    None,
    /// The 3/4 upscale (1080p to 1440p docked, 810p to 1080p handheld).
    ThreeQuarter,
    /// The output pass at 1:1: 2560x1440 docked and 1920x1080 handheld with the Max Resolution
    /// add-on.
    OneToOne,
    /// The upscale from a 16:9 scene the game's dynamic resolution or the DLSS render scale shrank.
    Scaled,
};

/// The compute pass whose depth texture is the scene depth the upscaler reprojects.
constexpr u64 OcarinaDepthReaderKey = 0x20ab4117f4db5bdbULL; // shader 1fc7a0423116281f
constexpr u64 OcarinaDepthReaderShader = 0x1fc7a0423116281fULL;

/// Passes are matched by native program key, which only the game's code decides: the shader hash
/// changes with every lowering change that touches the program.
constexpr OcarinaUpscalePass FindOcarinaUpscalePass(u64 native_key, u64 shader_hash) {
    using enum OcarinaUpscalePass;
    switch (native_key) {
    case 0x6f9367bd41dc0b58ULL: // shader e40f5efb5480e692
        return ThreeQuarter;
    case 0xda6df3e54c716df2ULL: // shader 6bd913e1f34e83eb
        return OneToOne;
    case 0xe4cda162ff6009e2ULL: // shader 9302bc0d6d1d9bc1, in the field (#903)
    case 0xd8efd9f0c95abd80ULL: // shader 03bf0ffc7d2d0bed, handheld (#1866)
        return Scaled;
    }
    switch (shader_hash) {
    case 0xe40f5efb5480e692ULL:
    case 0xef542df87cacfe9eULL:
        return ThreeQuarter;
    case 0x6bd913e1f34e83ebULL:
    case 0xc033c3ddc9f0897bULL:
        return OneToOne;
    case 0x4c1ae2fe58f30a97ULL: // in menus (#903)
    case 0x9302bc0d6d1d9bc1ULL:
    case 0x03bf0ffc7d2d0bedULL:
    case 0x31c0acb7c5e87c89ULL: // oot_resolution 40 % and 33 %
        return Scaled;
    }
    return None;
}

/// The FSR jitter for a phase (from 1) as a Halton (2, 3) offset in render pixels, applied to the
/// scene's viewport. AMD's own drivers (PAL) clip each viewport to an integer scissor that drops
/// the fraction of its right and bottom edges, so a negative offset loses the last column and row
/// of every jittered draw there; on those drivers the offsets stay in [0, 1) instead of
/// [-0.5, 0.5).
[[nodiscard]] constexpr std::array<f32, 2> OcarinaJitter(u32 index,
                                                         bool truncated_viewport_edges) noexcept {
    const f32 center = truncated_viewport_edges ? 0.0f : 0.5f;
    return {OcarinaUpscale::Halton(index, 2) - center, OcarinaUpscale::Halton(index, 3) - center};
}

/// render_scaled: a DLSS render-scale patch shrank the scene to a size FSR 3.1 takes, so FSR 3.1
/// reconstructs it when DLSS can't run.
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available, bool render_scaled,
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
    return fsr_enabled || render_scaled ? OcarinaUpscaleBackend::Fsr
                                        : OcarinaUpscaleBackend::Native;
}

constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available,
                                                    std::string_view override) {
    return SelectOcarinaUpscaler(fsr_enabled, dlss_enabled, dlss_available, false, override);
}

using Settings::AppliedOcarinaRenderScale;

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

/// NVIDIA's log2(render / output) mip bias for each 4K DLSS mode.
constexpr float OcarinaDlssLodBias(Settings::OcarinaDlssMode mode) {
    switch (mode) {
    case Settings::OcarinaDlssMode::Quality:
        return -0.58f;
    case Settings::OcarinaDlssMode::Balanced:
        return -0.75f;
    case Settings::OcarinaDlssMode::Performance:
        return -1.0f;
    case Settings::OcarinaDlssMode::UltraPerformance:
        return -1.58f;
    }
    return -0.58f;
}

/// Texture LOD bias added to Ocarina's mipmapped samplers while a jittered temporal upscaler runs.
/// AMD recommends log2(render / output) - 1: -1.58 for docked 1440p -> 4K. With
/// EDEN_NX2_DLSS_JITTER, DLSS / MetalFX Temporal gets log2(render / output) for its mode.
inline float OcarinaFsrLodBias(
    OcarinaUpscaleBackend backend,
    Settings::OcarinaDlssMode dlss_mode = Settings::OcarinaDlssMode::Quality) {
    if (!OcarinaJitterEnabled(backend)) {
        return 0.0f;
    }
    if (backend == OcarinaUpscaleBackend::Fsr) {
        return -1.58f;
    }
    if (backend == OcarinaUpscaleBackend::Dlss) {
        return OcarinaDlssLodBias(dlss_mode);
    }
    return 0.0f;
}

/// Whether a sampler with that bias keeps it for an image: only docked (handheld isn't measured),
/// not for images the GPU wrote (render targets would read a mip too sharp), and not once the
/// upscaler failed this session (the game's own pass runs again).
constexpr bool KeepOcarinaFsrLodBias(bool docked, bool upscaler_failed, bool gpu_written) {
    return docked && !upscaler_failed && !gpu_written;
}

} // namespace VideoCommon

namespace VideoCommon::OcarinaUpscale {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;

// The post-tonemap chain is four dispatches; the third samples the tonemapped 3D scene frame and
// the fourth writes the anti-aliased / upscaled result before HUD rendering.
// Match by either lowered ShaderInfo::unique_identifier (*Hash) or raw guest code key (*Key):
// - Pass 3 (input pass, cs=0x11483c1100):
constexpr u64 InputPassHash = 0xa2fcaa95c5a162f3ULL;
constexpr u64 InputPassKey = 0x42593194bb0b2b32ULL;
// - Pass 4 (output pass):
//   * Handheld 1080p->1080p: 0x6bd913e1f34e83eb
//   * Docked 1080p->1440p upscale (Max Resolution OFF, cs=0x11483d5300): key 0x6f9367bd41dc0b58,
//     hash 0xcf806c2846075a1f
//   * Docked 1440p->1440p DLAA (Max Resolution ON, cs=0x11483c8e00): key 0xda6df3e54c716df2,
//     hash 0x9dfd3ff9ed9cb72a
constexpr u64 OutputPassHash = 0x6bd913e1f34e83ebULL;
constexpr u64 OutputPassUpscaleHash = 0xcf806c2846075a1fULL;
constexpr u64 OutputPassUpscaleKey = 0x6f9367bd41dc0b58ULL;
constexpr u64 OutputPassDockedHash = 0x9dfd3ff9ed9cb72aULL;
constexpr u64 OutputPassDockedKey = 0xda6df3e54c716df2ULL;

[[nodiscard]] constexpr bool IsInputPass(u64 native_key, u64 shader_hash) noexcept {
    return shader_hash == InputPassHash || native_key == InputPassKey ||
           native_key == InputPassHash;
}

[[nodiscard]] constexpr bool IsOutputPass(u64 native_key, u64 shader_hash) noexcept {
    return shader_hash == OutputPassHash || shader_hash == OutputPassUpscaleHash ||
           shader_hash == OutputPassDockedHash || native_key == OutputPassUpscaleKey ||
           native_key == OutputPassDockedKey || native_key == OutputPassHash;
}

// Constant bank 0 offsets of their image handles (both B10G11R11).
constexpr u32 InputColorOffset = 0xa8;
constexpr u32 OutputColorOffset = 0xd0;

// The scene's draws bind the camera as vertex constant buffer 3: the view-projection matrix (rows,
// clip = M * world) at 0x30 and the projection at 0x70.
constexpr u32 CameraBuffer = 3;
constexpr u32 ViewProjectionOffset = 0x30;
constexpr u32 ProjectionOffset = 0x70;
constexpr u32 CameraBytes = 0xb0;

/// Frames in a cycle of the camera jitter.
constexpr u32 JitterPhases = 16;

using Matrix = std::array<f64, 16>; ///< Row-major 4x4.

[[nodiscard]] inline bool PlainImage(const Tegra::Texture::TICEntry& image) {
    using namespace Tegra::Texture;
    return (image.texture_type == TextureType::Texture2D ||
            image.texture_type == TextureType::Texture2DNoMipmap) &&
           image.IsBlockLinear() && image.Depth() == 1 && image.BaseLayer() == 0 &&
           image.max_mip_level == 0 && image.Address() != 0;
}

/// Whether a projection (rows) is a GL-style perspective one, as the scene's camera uses.
[[nodiscard]] inline bool IsPerspective(const std::array<f32, 16>& projection) {
    for (const f32 value : projection) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return projection[0] > 0.0f && projection[5] > 0.0f && projection[14] == -1.0f &&
           projection[15] == 0.0f && projection[12] == 0.0f && projection[13] == 0.0f;
}

/// The sample offset of a frame in render pixels (-0.5 to 0.5): Halton bases 2 and 3.
[[nodiscard]] constexpr std::array<f32, 2> Jitter(u32 frame) noexcept {
    return OcarinaJitter(frame % JitterPhases + 1, false);
}

/// Normalized device coordinates to render pixels and device depth: the viewport (x, y, width,
/// height in render pixels; y_sign -1 when the device negates y before the viewport) and the
/// depth range (device depth = depth_offset + depth_scale * NDC z).
[[nodiscard]] constexpr Matrix ViewportMatrix(f32 x, f32 y, f32 width, f32 height, f32 y_sign,
                                              f32 depth_offset, f32 depth_scale) noexcept {
    const f64 half_width = width * 0.5;
    const f64 half_height = height * 0.5;
    return {
        half_width, 0.0, 0.0, x + half_width,
        0.0, y_sign * half_height, 0.0, y + half_height,
        0.0, 0.0, depth_scale, depth_offset,
        0.0, 0.0, 0.0, 1.0,
    };
}

[[nodiscard]] constexpr Matrix Multiply(const Matrix& a, const Matrix& b) noexcept {
    Matrix result{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            f64 sum = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                sum += a[row * 4 + k] * b[k * 4 + column];
            }
            result[row * 4 + column] = sum;
        }
    }
    return result;
}

[[nodiscard]] inline std::optional<Matrix> Invert(Matrix m) {
    Matrix inverse{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    for (size_t column = 0; column < 4; ++column) {
        size_t pivot = column;
        for (size_t row = column + 1; row < 4; ++row) {
            if (std::abs(m[row * 4 + column]) > std::abs(m[pivot * 4 + column])) {
                pivot = row;
            }
        }
        if (std::abs(m[pivot * 4 + column]) < 1e-300) {
            return std::nullopt;
        }
        for (size_t k = 0; k < 4; ++k) {
            std::swap(m[column * 4 + k], m[pivot * 4 + k]);
            std::swap(inverse[column * 4 + k], inverse[pivot * 4 + k]);
        }
        const f64 scale = 1.0 / m[column * 4 + column];
        for (size_t k = 0; k < 4; ++k) {
            m[column * 4 + k] *= scale;
            inverse[column * 4 + k] *= scale;
        }
        for (size_t row = 0; row < 4; ++row) {
            const f64 factor = m[row * 4 + column];
            if (row == column || factor == 0.0) {
                continue;
            }
            for (size_t k = 0; k < 4; ++k) {
                m[row * 4 + k] -= factor * m[column * 4 + k];
                inverse[row * 4 + k] -= factor * inverse[column * 4 + k];
            }
        }
    }
    return inverse;
}

/// World to render pixels and device depth (homogeneous) for a camera's view-projection (rows).
[[nodiscard]] inline Matrix ScreenMatrix(const Matrix& viewport, const std::array<f32, 16>& camera) {
    Matrix view_projection;
    for (size_t i = 0; i < 16; ++i) {
        view_projection[i] = camera[i];
    }
    return Multiply(viewport, view_projection);
}

/// Takes (pixel x, pixel y, device depth, 1) of this frame to the same point's pixel in the
/// previous frame (homogeneous), as a column-major matrix for a shader.
[[nodiscard]] inline std::optional<std::array<f32, 16>> Reprojection(const Matrix& previous,
                                                                     const Matrix& current) {
    const std::optional<Matrix> inverse = Invert(current);
    if (!inverse) {
        return std::nullopt;
    }
    const Matrix reprojection = Multiply(previous, *inverse);
    std::array<f32, 16> columns;
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            const f64 value = reprojection[row * 4 + column];
            if (!std::isfinite(value)) {
                return std::nullopt;
            }
            columns[column * 4 + row] = static_cast<f32>(value);
        }
    }
    return columns;
}

} // namespace VideoCommon::OcarinaUpscale
