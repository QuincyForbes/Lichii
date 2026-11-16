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

struct TextureDesc;
enum class TextureKind : u8;

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// Vertex stream i is bound at buffer slot 30 - i (below the translator's
/// first_vertex_buffer_slot), resources take the slots from 0 up.
[[nodiscard]] constexpr u32 VertexBufferSlot(u32 stream) noexcept {
    return 30 - stream;
}

/// Attributes and buffer layouts of a Metal vertex descriptor; asking the descriptor for one past
/// them aborts the process.
inline constexpr u32 MaxVertexSlots = 31;

/// VertexStreamState::divisor of a stream read at the same place for every vertex and instance.
inline constexpr u32 ConstantStep = ~0U;

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
    u32 divisor; ///< 0: per vertex, ConstantStep: one element for all
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
    u32 alpha_to_coverage; ///< Bit 0 alpha to coverage, bit 1 alpha to one
    std::array<BlendTargetState, Maxwell::NumRenderTargets> blend;
    std::array<VertexAttributeState, Maxwell::NumVertexAttributes> attributes;
    std::array<VertexStreamState, Maxwell::NumVertexArrays> streams;
    /// Tessellation pipelines (the vertex function is post-tessellation; no vertex layout):
    /// MTLTessellationPartitionMode and MTLWinding.
    u32 tessellation;
    u32 tess_partition;
    u32 tess_winding;
};

/// Where the buffer of a sanitised vertex layout's stream comes from (SanitizeVertexLayout).
struct VertexSource {
    static constexpr u32 Zeros = ~0U;
    static constexpr u32 Gathered = ~1U;

    /// The guest's vertex stream, Zeros: a buffer of zeros, or Gathered: VertexSources' gathered
    /// buffer.
    u32 stream{Zeros};
    u32 shift{};       ///< Added to the offset the stream's buffer is bound at
    /// Integer or scaled 10:10:10:2, which Metal has no vertex format for: the attribute at
    /// `shift` of each element of the stream, widened by a kernel to four 16-bit integers (8
    /// bytes each, the layout's stride); 1 unsigned, 2 signed (VertexUnpackKernelSource).
    u32 unpack{};

    bool operator==(const VertexSource&) const noexcept = default;
};

/// An attribute of a per-vertex stream copied into the gathered buffer (VertexSources::gather).
struct GatheredAttribute {
    u32 stream;
    u32 offset; ///< In the stream's element
    u32 bytes;
    u32 gathered_offset; ///< In the gathered buffer's element
};

struct VertexSources {
    std::array<VertexSource, Maxwell::NumVertexArrays> streams{};
    u32 count{};
    /// With more streams than buffer slots, the last slot (gather_slot) holds a buffer the
    /// attributes of the remaining per-vertex streams are copied into, gather_stride bytes a
    /// vertex (VertexGatherKernelSource).
    std::array<GatheredAttribute, Maxwell::NumVertexAttributes> gathered{};
    u32 gathered_count{};
    u32 gather_slot{~0U};
    u32 gather_stride{};
};

/// What the device takes in a render pipeline descriptor.
struct PipelineLimits {
    u32 sample_counts{1U << 1}; ///< Bit n: n samples per pixel
};

/// EDEN_NX2_METAL_PIPELINE_CHECK (on by default; 0: descriptors go to Metal as the guest's state
/// gives them, and one Metal rejects by assertion aborts the process).
[[nodiscard]] bool PipelineCheckEnabled();

/// EDEN_NX2_METAL_VERTEX_UNPACK (on by default; 0: integer and scaled 10:10:10:2 attributes are
/// read through Metal's normalized 10:10:10:2 formats, as before).
[[nodiscard]] bool VertexUnpackEnabled();

/// MSL of the kernel `unpack_1010102`: the stream's bytes at buffer 0, four 16-bit integers per
/// element at buffer 1, VertexUnpackParams at buffer 2.
[[nodiscard]] const char* VertexUnpackKernelSource() noexcept;

/// EDEN_NX2_METAL_VERTEX_GATHER (on by default; 0: a draw with more streams than buffer slots is
/// refused, as before).
[[nodiscard]] bool VertexGatherEnabled();

/// MSL of the kernel `gather_vertices`: one attribute's bytes of each element copied from the
/// stream (buffer 0) into the gathered buffer (buffer 1), VertexGatherParams at buffer 2.
[[nodiscard]] const char* VertexGatherKernelSource() noexcept;

struct VertexGatherParams {
    u32 count;  ///< Elements
    u32 stride; ///< Of the stream
    u32 offset; ///< Of the attribute in the first element, from the bound offset
    u32 bytes;
    u32 gathered_stride;
    u32 gathered_offset;
};

struct VertexUnpackParams {
    u32 count;     ///< Elements
    u32 stride;    ///< Of the stream, 0 for one element
    u32 is_signed;
    u32 offset;    ///< Bytes before the first element's attribute, from the bound offset
};

/// EDEN_NX2_METAL_ABORT_TRACE (on by default): the renderer's boot trace.
[[nodiscard]] bool BootTraceEnabled();

/// Bytes of a vertex attribute in the stream, 0 when Metal has no vertex format for it.
[[nodiscard]] u32 VertexFormatBytes(Maxwell::VertexAttribute::Type type,
                                    Maxwell::VertexAttribute::Size size);

/// Rewrites desc's vertex layout into one Metal takes, as Vulkan and MoltenVK present the same
/// guest state: only the attributes the vertex function reads (bit n of `loaded`: generic n), only
/// the streams those read, numbered from 0 in `sources`; an attribute of an unbound stream reads
/// zeros; an attribute past its stream's stride moves to a copy of the stream bound further on.
/// `resource_slots` buffer slots from 0 up belong to the function's resources. False (desc
/// untouched, `reason` set) when the layout doesn't fit Metal's vertex descriptor.
[[nodiscard]] bool SanitizeVertexLayout(RenderPipelineDesc& desc, u32 loaded, u32 resource_slots,
                                        VertexSources& sources, std::string& reason);

/// Why Metal would reject (by assertion, aborting the process) desc's vertex layout, or nothing.
[[nodiscard]] std::string CheckVertexLayout(const RenderPipelineDesc& desc);

/// Why Metal would reject desc by assertion, or nothing.
[[nodiscard]] std::string CheckPipeline(const RenderPipelineDesc& desc,
                                        const PipelineLimits& limits);

/// Why Metal would reject the texture by assertion, or nothing.
[[nodiscard]] std::string CheckTexture(const TextureDesc& desc, const PipelineLimits& limits);

/// The sample count a texture asking for `samples` gets: the largest the device takes up to it
/// (Apple GPUs take 4 at most, so the guest's 8 draw at 4 instead of being refused).
[[nodiscard]] inline u32 SupportedSamples(u32 samples, const PipelineLimits& limits) {
    for (u32 count = samples < 31 ? samples : 31; count > 1; --count) {
        if (((limits.sample_counts >> count) & 1) != 0) {
            return count;
        }
    }
    return 1;
}

/// Why Metal would reject by assertion a view of kind `view` with `layers` slices of a texture
/// of kind `base`, or nothing.
[[nodiscard]] std::string CheckTextureView(TextureKind base, TextureKind view, u32 layers);

struct RenderTargetSet;
struct TextureFacts;

/// The pass's attachments as Metal must get them (its validation layer checks these; release
/// builds have none): a level or slice past the texture becomes its last one, and an attachment
/// whose format or sample count the pipeline does not have takes the texture's format when it is
/// of the same kind and sample count, else is left out of the pass and the pipeline. Counted as
/// guards (Coverage::NoteGuard). `describe` is DescribeTexture outside tests.
void FitAttachments(RenderPipelineDesc& desc, RenderTargetSet& targets,
                    TextureFacts (*describe)(TextureHandle));

/// The 1x1 RGBA8 texture of zeros bound where a program reads a texture of `type` and its view
/// has none (RasterizerMetal::OrPlaceholder); 1D types as 2D, as the pipeline check makes them.
[[nodiscard]] TextureDesc PlaceholderTexture(u32 texture_type);

/// The whole description on one line, for the log.
[[nodiscard]] std::string Describe(const RenderPipelineDesc& desc);

[[nodiscard]] PipelineLimits DevicePipelineLimits(const Device& device);

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

/// The functions of a Metal 3 mesh pipeline (NativeMesh) and what else its descriptor takes
/// besides desc's targets and blending.
struct MeshFunctions {
    const void* object_function; ///< Null without a task program
    const void* mesh_function;
    const void* fragment_function;
    u32 payload_bytes;
    u32 object_threads;
    u32 mesh_threads;
};

/// An MTLRenderPipelineState from an MTLMeshRenderPipelineDescriptor: desc's targets, blending,
/// samples and rasterization, `functions` for the stages (desc's functions and vertex layout are
/// not read). With `archive` as CreateRenderPipelineState, where the OS has binary archives of
/// mesh pipelines (iOS 18, macOS 15).
[[nodiscard]] Object CreateMeshRenderPipelineState(const Device& device,
                                                   const RenderPipelineDesc& desc,
                                                   const MeshFunctions& functions,
                                                   std::string& error,
                                                   PipelineArchive* archive = nullptr);

[[nodiscard]] Object CreateDepthStencilState(const Device& device, const DepthStencilDesc& desc);

/// MTLPrimitiveType for the topology, or nothing when Metal can't draw it directly (quads,
/// fans, loops, polygons, adjacency, patches: drawn by emulation or skipped).
[[nodiscard]] std::optional<u32> PrimitiveType(Maxwell::PrimitiveTopology topology);

/// 0 points, 1 lines, 2 triangles (MTLPrimitiveTopologyClass order).
[[nodiscard]] u32 TopologyClass(Maxwell::PrimitiveTopology topology);

} // namespace Metal
