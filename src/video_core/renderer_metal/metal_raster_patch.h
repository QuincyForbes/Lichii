// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Raster state Metal has no setting for, done in the translated programs: logic operations in the
// pixel program (the framebuffer fetch of the target's own value, the operation on the bits the
// target stores), and viewport swizzles in the vertex program. Both edit the MSL SPIRV-Cross
// writes, so they take the shape of its output (an output struct, an entry point, one
// `return out;` per exit). The framebuffer fetch with an in-shader logic operation follows
// nxnative's dm/gfx/tools/dmsh_logic.py (MIT, github.com/seedds/nxnative). Plain C++, no Metal.

#pragma once

#include <array>
#include <string>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Metal::RasterPatch {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// What a colour target stores, as far as a logic operation is concerned: the operation is done
/// on the stored bits, so normalized values are rounded to the bits the format keeps.
enum class LogicFormat : u8 {
    None, ///< Float, sRGB or no target: no operation (as on Vulkan)
    Unorm8,
    Snorm8,
    Unorm16,
    Snorm16,
    Unorm10_2,
    Unorm565,
    Unorm5551,
    Unorm4444,
    Uint8,
    Uint16,
    Uint32,
    Uint10_2,
    Sint8,
    Sint16,
    Sint32,
};

[[nodiscard]] LogicFormat LogicFormatOf(Tegra::RenderTargetFormat format) noexcept;

/// The draw's logic operation: `formats` holds four bits (a LogicFormat) per colour target; zero
/// when the operation is off, the programs have no pixel program or no target takes it.
struct LogicState {
    u32 op{};      ///< Maxwell::LogicOp::Op - 0x1500 (VkLogicOp)
    u32 formats{}; ///< 4 bits per target

    [[nodiscard]] bool Active() const noexcept {
        return formats != 0;
    }
    [[nodiscard]] LogicFormat Format(size_t target) const noexcept {
        return static_cast<LogicFormat>((formats >> (target * 4)) & 15);
    }
};

/// EDEN_NX2_METAL_LOGIC_OP (default on; 0: the logic operation is ignored, blending applies).
[[nodiscard]] bool LogicOpEnabled() noexcept;

[[nodiscard]] LogicState LogicStateOf(const Maxwell& regs) noexcept;

/// The pixel program's MSL with the logic operation applied to every output of a target in
/// `state`: false (the source untouched) when its entry point or outputs are not as expected.
[[nodiscard]] bool ApplyLogicOp(std::string& msl, const LogicState& state);

/// Viewport swizzle bits as a register has them (x at 0, y at 4, z at 8, w at 12, three bits each).
inline constexpr u32 IdentitySwizzle = 0x6420;
/// The swizzle with y negated, as the viewport's flip does it.
inline constexpr u32 NegatedYSwizzle = 0x6430;

/// EDEN_NX2_METAL_VIEWPORT_SWIZZLE (default on; 0: swizzles are ignored but the negated y).
[[nodiscard]] bool ViewportSwizzleEnabled() noexcept;

/// Whether a swizzle is done by the vertex program: neither the identity nor the negated y the
/// viewport flip does.
[[nodiscard]] bool SwizzleInProgram(u32 swizzle) noexcept;

/// The vertex program's MSL with its clip position swizzled (before the y flip SPIRV-Cross adds
/// for Metal, so the swizzle is the guest's).
[[nodiscard]] bool ApplyViewportSwizzle(std::string& msl, u32 swizzle);

} // namespace Metal::RasterPatch
