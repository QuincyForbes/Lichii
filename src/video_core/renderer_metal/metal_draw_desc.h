// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// What a draw's pipeline and depth-stencil descriptions take from the registers and the
// translated programs alone. The rasterizer adds what the caches know (targets, bound buffers);
// the offline replay (metal_replay.cpp) adds it from a census row.

#pragma once

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_pipeline_translate.h"

namespace Metal {

/// Write masks, blending, the raster switches, the topology class and the vertex attributes.
/// `mesh`: the mesh emulation's vertex shader reads its output buffer, no vertex streams.
void FillRegisterState(RenderPipelineDesc& desc, const Maxwell& regs,
                       Maxwell::PrimitiveTopology topology, bool mesh);

/// The topology a draw is drawn with: the guest's, or Points for triangles whose polygon mode is
/// point (Metal has no point fill mode: the vertices are drawn as points, each once per index).
/// EDEN_NX2_METAL_POINT_FILL=0: as the guest's.
[[nodiscard]] Maxwell::PrimitiveTopology DrawTopology(const Tegra::Engines::Maxwell3D& engine);

/// DrawTopology from the registers and the guest's topology.
[[nodiscard]] Maxwell::PrimitiveTopology PointFillTopology(const Maxwell& regs,
                                                           Maxwell::PrimitiveTopology topology);

/// A vertex stream's layout; `bound`: the stream has a buffer.
void FillStream(RenderPipelineDesc& desc, const Maxwell& regs, size_t index, bool bound,
                u32 stride);

/// SanitizeVertexLayout's view of a vertex program.
struct VertexProgramUse {
    u32 loaded{};         ///< Bit n: the program loads generic n
    u32 resource_slots{}; ///< Buffer slots from 0 up its resources take
};

[[nodiscard]] VertexProgramUse VertexUse(const StageProgram& program);

/// Post-FitAttachments target formats and sample count, with a precomputed hash.
struct AttachmentFormatsKey {
    u64 hash{};
    std::array<u32, Maxwell::NumRenderTargets> color_formats{};
    u32 depth_format{};
    u32 stencil_format{};
    u32 samples{};

    bool operator==(const AttachmentFormatsKey&) const noexcept = default;
};

[[nodiscard]] AttachmentFormatsKey MakeAttachmentFormatsKey(
    const std::array<u32, Maxwell::NumRenderTargets>& color_formats, u32 depth_format,
    u32 stencil_format, u32 samples) noexcept;

/// Compact key of everything RenderPipelineDesc + SanitizeVertexLayout read for a pipeline.
struct RenderStateMemoKey {
    AttachmentFormatsKey formats{};
    u32 write_masks{}; ///< 4 bits per colour target (R 1 .. A 8)
    u32 bound_streams_mask{};
    u8 blend_enable_mask{};
    u8 topology_class{};
    u8 flags{}; ///< 0 rasterize, 1..2 alpha_to_coverage, 3 per_target, 4 native, 5..6 reducing, 7 checked
    u8 used_attrib_count{};
    u8 used_stream_count{};
    std::array<u32, 6> common_blend{};
    std::array<std::array<u32, 6>, Maxwell::NumRenderTargets> target_blend{};
    std::array<u32, Maxwell::NumVertexAttributes> attrib_hex{};
    std::array<u32, Maxwell::NumVertexArrays> stream_strides{};
    std::array<u32, Maxwell::NumVertexArrays> stream_divisors{};

    [[nodiscard]] bool Matches(const RenderStateMemoKey& other,
                               bool mesh_pipeline) const noexcept;
};

/// Fills the non-vertex portion of `key` and `color_written` from the registers.
void FillRenderStateMemoRegisters(RenderStateMemoKey& key, u8& color_written, const Maxwell& regs,
                                  Maxwell::PrimitiveTopology topology,
                                  const AttachmentFormatsKey& formats, bool native, u32 reducing,
                                  bool checked) noexcept;

/// A tessellation pipeline's render state: no vertex layout (its post-tessellation vertex
/// function reads buffers), partition mode and winding.
void ApplyTessellation(RenderPipelineDesc& desc, const Tessellation& tessellation);

[[nodiscard]] DepthStencilDesc DepthStencilState(const Maxwell& regs, bool has_depth,
                                                 bool has_stencil);

[[nodiscard]] u64 PackDepthStencilKey(const DepthStencilDesc& desc) noexcept;

} // namespace Metal
