// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Draw state for plain C++: descriptions hold Maxwell's own values, metal_graphics.mm turns them
// into Metal pipeline and depth-stencil states. Descriptions are compared and hashed as bytes:
// build them from a zeroed object (Zeroed<T>()).

#pragma once

#include <array>
#include <cstring>
#include <optional>
#include <string>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

class PipelineArchive;

class Device;

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// Vertex stream i is bound at buffer slot 30 - i (below the translator's
/// first_vertex_buffer_slot), resources take the slots from 0 up.
[[nodiscard]] constexpr u32 VertexBufferSlot(u32 stream) noexcept {
    return 30 - stream;
}

template <typename T>
[[nodiscard]] T Zeroed() noexcept {
    T value;
    std::memset(&value, 0, sizeof(value));
    return value;
}

struct BlendTargetState {
    u32 enable;
    u32 write_mask; ///< Bit 0 R .. bit 3 A
    Maxwell::Blend::Equation color_op;
    Maxwell::Blend::Equation alpha_op;
    Maxwell::Blend::Factor color_source;
    Maxwell::Blend::Factor color_dest;
    Maxwell::Blend::Factor alpha_source;
    Maxwell::Blend::Factor alpha_dest;
};

struct VertexAttributeState {
    u32 enabled;
    u32 stream;
    u32 offset;
    Maxwell::VertexAttribute::Type type;
    Maxwell::VertexAttribute::Size size;
};

struct VertexStreamState {
    u32 enabled;
    u32 stride;
    u32 divisor; ///< 0: per vertex
};

struct RenderPipelineDesc {
    const void* vertex_function;
    const void* fragment_function;
    std::array<u32, Maxwell::NumRenderTargets> color_formats;
    u32 depth_format;
    u32 stencil_format;
    u32 samples;
    u32 topology_class; ///< 0 points, 1 lines, 2 triangles
    u32 rasterize;
    u32 alpha_to_coverage;
    std::array<BlendTargetState, Maxwell::NumRenderTargets> blend;
    std::array<VertexAttributeState, Maxwell::NumVertexAttributes> attributes;
    std::array<VertexStreamState, Maxwell::NumVertexArrays> streams;
    /// Tessellation pipelines (the vertex function is post-tessellation; no vertex layout):
    /// MTLTessellationPartitionMode and MTLWinding.
    u32 tessellation;
    u32 tess_partition;
    u32 tess_winding;
};

struct StencilFaceState {
    Maxwell::StencilOp::Op fail;
    Maxwell::StencilOp::Op depth_fail;
    Maxwell::StencilOp::Op pass;
    Maxwell::ComparisonOp func;
    u32 read_mask;
    u32 write_mask;
};

struct DepthStencilDesc {
    u32 depth_test;
    u32 depth_write;
    Maxwell::ComparisonOp depth_func;
    u32 stencil_test;
    StencilFaceState front;
    StencilFaceState back;
};

/// The vertex program of a tessellation pipeline as a compute kernel reading layout's vertex
/// attributes and streams as stage-in data (one thread per vertex or, with index_size 2 or 4, per
/// index of the buffer at index_slot; instances along y).
[[nodiscard]] Object CreateStageInComputePipelineState(const Device& device, const void* function,
                                                       const RenderPipelineDesc& layout,
                                                       u32 index_size, u32 index_slot,
                                                       std::string& error,
                                                       PipelineArchive* archive = nullptr);

/// With `archive`, the pipeline's GPU code is looked up there first and added on a miss.
[[nodiscard]] Object CreateRenderPipelineState(const Device& device, const RenderPipelineDesc& desc,
                                               std::string& error,
                                               PipelineArchive* archive = nullptr);

[[nodiscard]] Object CreateDepthStencilState(const Device& device, const DepthStencilDesc& desc);

/// MTLPrimitiveType for the topology, or nothing when Metal can't draw it directly (quads,
/// fans, loops, polygons, adjacency, patches: drawn by emulation or skipped).
[[nodiscard]] std::optional<u32> PrimitiveType(Maxwell::PrimitiveTopology topology);

/// 0 points, 1 lines, 2 triangles (MTLPrimitiveTopologyClass order).
[[nodiscard]] u32 TopologyClass(Maxwell::PrimitiveTopology topology);

} // namespace Metal
