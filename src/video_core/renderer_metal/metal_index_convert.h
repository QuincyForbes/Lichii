// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Index lists Metal can draw: quads, quad strips, fans, polygons and line loops rewritten, and
// 8-bit indices widened (Metal draws 16- and 32-bit indices only). With `provoking_last`,
// triangle lists and strips become lists whose triangles start with the vertex Vulkan's last
// vertex convention takes flat inputs from (Metal takes them from the first). Plain C++, no Metal.

#pragma once

#include <span>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace Metal::IndexConvert {

using Topology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;

/// Whether Metal can't draw the topology from the guest's own vertices or indices.
[[nodiscard]] bool NeedsConversion(Topology topology) noexcept;

/// Whether the provoking vertex rewrite takes the topology (triangle lists and strips).
[[nodiscard]] bool RotatesForProvokingVertex(Topology topology) noexcept;

/// What the rewritten list is drawn as: Triangles for quads, quad strips, fans and polygons
/// (and strips with `provoking_last`), LineStrip for line loops, the topology itself otherwise.
[[nodiscard]] Topology ConvertedTopology(Topology topology, bool provoking_last = false) noexcept;

/// Indices the rewritten list of `count` guest vertices holds at most.
[[nodiscard]] u32 ConvertedCount(Topology topology, u32 count,
                                 bool provoking_last = false) noexcept;

/// Writes ConvertedCount(topology, count) 32-bit indices to `out`. The guest's indices come
/// from `source` (`index_size` bytes each), or are first, first + 1, ... when it is empty. With
/// `restart`, an 8-bit 0xFF becomes 0xFFFFFFFF (Metal's restart value for 32-bit indices); a
/// strip rewritten for the provoking vertex starts again after it. Returns the indices written,
/// fewer than ConvertedCount for such a strip with restarts.
u32 Convert(Topology topology, std::span<const u8> source, u32 index_size, u32 first, u32 count,
            bool restart, std::span<u32> out, bool provoking_last = false) noexcept;

/// Constants of the GPU rewrite (RewriteKernelSource's RewriteParams), for indices the GPU wrote:
/// Convert's result for `count` indices of `index_size` bytes at byte `source_offset` of the
/// source buffer, one thread per index written.
struct RewriteParams {
    u32 mode; ///< RewriteMode
    u32 index_size;
    u32 count;
    u32 restart;
    u32 converted; ///< ConvertedCount
    u32 source_offset;
};

/// RewriteParams::mode of a topology: 0 as it is, 1 quads, 2 quad strip, 3 fan or polygon,
/// 4 line loop; with `provoking_last` 5 triangles, 6 triangle strip (without restarts: the GPU
/// rewrite does not find where a strip starts again).
[[nodiscard]] u32 RewriteMode(Topology topology, bool provoking_last = false) noexcept;

/// MSL of the kernel `rewrite_indices`: source bytes at buffer 0, the u32 indices at buffer 1,
/// RewriteParams at buffer 2.
[[nodiscard]] const char* RewriteKernelSource() noexcept;

} // namespace Metal::IndexConvert
