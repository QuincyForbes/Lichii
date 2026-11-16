// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Index lists Metal can draw: quads, quad strips, fans, polygons, line loops and adjacency
// topologies rewritten (Metal has none of them), and 8-bit indices widened (Metal draws 16- and
// 32-bit indices only). With `provoking_last`, every line and triangle topology is written so
// each primitive starts with the vertex Vulkan's last vertex convention takes flat inputs from
// (Metal has no provoking vertex setting and takes them from the first): triangles as lists
// rotated to keep their winding, lines swapped, line strips and loops backwards. Plain C++, no
// Metal.

#pragma once

#include <span>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Metal::IndexConvert {

using Topology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;

/// EDEN_NX2_METAL_ADJACENCY (default on; 0: adjacency topologies are not rewritten, their draws
/// refused as before).
[[nodiscard]] bool AdjacencyEnabled() noexcept;

/// Whether Metal can't draw the topology from the guest's own vertices or indices.
[[nodiscard]] bool NeedsConversion(Topology topology) noexcept;

/// Whether the provoking vertex rewrite takes the topology (triangle lists and strips).
[[nodiscard]] bool RotatesForProvokingVertex(Topology topology) noexcept;

/// What the rewritten list is drawn as: Triangles for quads, quad strips, fans, polygons and
/// triangle adjacency (and strips with `provoking_last`), LineStrip for line loops and line
/// strips with adjacency, Lines for lines with adjacency, the topology itself otherwise.
[[nodiscard]] Topology ConvertedTopology(Topology topology, bool provoking_last = false) noexcept;

/// Indices the rewritten list of `count` guest vertices holds at most.
[[nodiscard]] u32 ConvertedCount(Topology topology, u32 count,
                                 bool provoking_last = false) noexcept;

/// Writes ConvertedCount(topology, count) 32-bit indices to `out`. The guest's indices come
/// from `source` (`index_size` bytes each), or are first, first + 1, ... when it is empty. With
/// `restart`, an 8-bit 0xFF becomes 0xFFFFFFFF (Metal's restart value for 32-bit indices); a
/// strip rewritten for the provoking vertex starts again after it, the triangles across a
/// restart written as (0, 0, 0). Returns the indices written.
u32 Convert(Topology topology, std::span<const u8> source, u32 index_size, u32 first, u32 count,
            bool restart, std::span<u32> out, bool provoking_last = false) noexcept;

/// Constants of the GPU rewrite (RewriteKernelSource's RewriteParams), for indices the GPU wrote:
/// Convert's result for `count` indices of `index_size` bytes at byte `source_offset` of the
/// source buffer, one thread per index written.
struct RewriteParams {
    u32 mode; ///< RewriteMode
    u32 index_size; ///< 0: no source, the indices are source_offset, source_offset + 1, ...
    u32 count;
    u32 restart;
    u32 converted; ///< ConvertedCount
    u32 source_offset;
};

/// The mode of a triangle strip with restarts rewritten for the last vertex convention: kernel
/// `rewrite_restart_strip`, one threadgroup of StripRestartThreads threads.
constexpr u32 StripRestartMode = 7;
constexpr u32 StripRestartThreads = 256;

/// RewriteParams::mode of a topology: 0 as it is, 1 quads, 2 quad strip, 3 fan or polygon,
/// 4 line loop; with `provoking_last` 5 triangles, 6 triangle strip, StripRestartMode a strip
/// with `restart`, 8 lines, 9 line strip, 10 line loop, 11 quads, 12 quad strip, 13 fan; 14 lines,
/// 15 line strip, 16 triangles and 17 triangle strip with adjacency.
[[nodiscard]] u32 RewriteMode(Topology topology, bool provoking_last = false,
                              bool restart = false) noexcept;

/// MSL of the kernels `rewrite_indices` (one thread per index written) and
/// `rewrite_restart_strip`: source bytes at buffer 0, the u32 indices at buffer 1,
/// RewriteParams at buffer 2.
[[nodiscard]] const char* RewriteKernelSource() noexcept;

} // namespace Metal::IndexConvert
