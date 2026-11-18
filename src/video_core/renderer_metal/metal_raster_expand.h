// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Rasterization Metal has no setting for, done on the processed vertices: lines wider than a pixel
// (as quads of two triangles, a parallelogram w pixels across the minor axis, as OpenGL draws an
// aliased wide line, nxnative's dm_wl_offset in dm/gfx/tools/dmsh_pull.py, MIT), stippled and
// smooth lines (a coordinate along and across the line for the pixel program to discard or
// cover by), and conservative triangles (grown by half a pixel). The vertex program runs first as
// the stage-in kernel of the transform feedback path (metal_xfb.h) writing every vertex's outputs
// to a buffer; a vertex function generated here, from the outputs struct of the program's own
// vertex MSL, pulls the ends of each primitive from it and emits the expanded triangles. Plain
// C++, no Metal.

#pragma once

#include <string>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Metal::RasterExpand {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// Buffer slots of the generated vertex function (the source vertices, the draw's Params).
inline constexpr u32 SourceSlot = 28;
inline constexpr u32 ParamsSlot = 29;
inline constexpr const char* Entry = "raster_expand";

/// GraphicsPipelineKey::expand bits.
inline constexpr u32 Wide = 1;         ///< Lines drawn as quads
inline constexpr u32 Stipple = 2;      ///< The pixel program discards by the stipple pattern
inline constexpr u32 Smooth = 4;       ///< The pixel program scales alpha by the line's coverage
inline constexpr u32 Conservative = 8; ///< Triangles grown by half a pixel

/// The generated function's constants (a draw's), set with setVertexBytes at ParamsSlot.
struct Params {
    f32 half_viewport[2]; ///< Half the viewport's size in pixels
    f32 width;            ///< Line width in pixels
    u32 prim;             ///< Primitive: PrimLines.. (below)
    u32 count;            ///< Vertices per instance in the source buffer
    f32 factor;           ///< Line stipple factor
    u32 pattern;          ///< Line stipple pattern (16 bits)
    u32 pad;
};
static_assert(sizeof(Params) == 32);

inline constexpr u32 PrimLines = 0;
inline constexpr u32 PrimLineStrip = 1;
inline constexpr u32 PrimLineLoop = 2;
inline constexpr u32 PrimTriangles = 3;
inline constexpr u32 PrimTriangleStrip = 4;
inline constexpr u32 PrimTriangleFan = 5;

/// EDEN_NX2_METAL_RASTER_EXPAND (default on; 0: lines are 1 pixel wide and triangles as they are).
[[nodiscard]] bool Enabled() noexcept;

/// Params::prim of a topology, or ~0U when the expansion does not take it.
[[nodiscard]] u32 PrimitiveOf(Maxwell::PrimitiveTopology topology) noexcept;

/// The expansion a draw's state asks for (the bits above, zero for none).
[[nodiscard]] u32 BitsOf(const Maxwell& regs, Maxwell::PrimitiveTopology topology) noexcept;

/// The line width the draw takes: the register's, rounded unless smooth, at most 64 pixels.
[[nodiscard]] f32 LineWidth(const Maxwell& regs) noexcept;

/// Primitives the `count` vertices of an instance make.
[[nodiscard]] u32 PrimitivesOf(u32 prim, u32 count) noexcept;

/// Vertices the expanded draw has per instance: 6 a line, 3 a triangle.
[[nodiscard]] u32 VerticesOf(u32 prim, u32 count) noexcept;

/// `vertex_msl` (the program's vertex function and outputs struct) with the generated function
/// appended; `kernel_msl` is the same program as a capture kernel (its outputs struct is what the
/// buffer holds). `swizzle`: viewport 0's swizzle to apply to the position (0: none). Empty when
/// the structs are not as expected.
[[nodiscard]] std::string Generate(const std::string& vertex_msl, const std::string& kernel_msl,
                                   u32 bits, u32 prim, u32 swizzle);

/// The pixel program's MSL with the stipple discard and the smooth coverage (bits Stipple and
/// Smooth), reading the line coordinate Generate writes; false when it is not as expected.
[[nodiscard]] bool PatchFragment(std::string& msl, u32 bits);

} // namespace Metal::RasterExpand
