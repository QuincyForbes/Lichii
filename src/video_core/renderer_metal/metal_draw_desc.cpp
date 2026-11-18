// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>

#include "video_core/renderer_metal/metal_draw_desc.h"
#include "video_core/renderer_metal/metal_raster_patch.h"

namespace Metal {

u32 TopologyClass(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return 0;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return 1;
    default:
        return 2;
    }
}

Maxwell::PrimitiveTopology DrawTopology(const Tegra::Engines::Maxwell3D& engine) {
    return PointFillTopology(engine.regs, engine.draw_manager.draw_state.topology);
}

Maxwell::PrimitiveTopology PointFillTopology(const Maxwell& regs,
                                             Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
        break;
    default:
        return topology;
    }
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_POINT_FILL");
        return value == nullptr || value[0] != '0';
    }();
    // Both faces in point mode; a cull of both faces draws nothing and stays with the guest's.
    const bool culls_all = regs.gl_cull_test_enabled != 0 &&
                           regs.gl_cull_face == Maxwell::CullFace::FrontAndBack;
    return enabled && !culls_all && regs.polygon_mode_front == Maxwell::PolygonMode::Point &&
                   regs.polygon_mode_back == Maxwell::PolygonMode::Point
               ? Maxwell::PrimitiveTopology::Points
               : topology;
}

void FillRegisterState(RenderPipelineDesc& desc, const Maxwell& regs,
                       Maxwell::PrimitiveTopology topology, bool mesh) {
    // A logic operation replaces blending where the pixel program does it (RasterPatch).
    const RasterPatch::LogicState logic = RasterPatch::LogicStateOf(regs);
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : rt];
        BlendTargetState& blend = desc.blend[rt];
        blend.write_mask = (mask.R != 0 ? 1U : 0U) | (mask.G != 0 ? 2U : 0U) |
                           (mask.B != 0 ? 4U : 0U) | (mask.A != 0 ? 8U : 0U);
        const bool per_target = regs.blend_per_target_enabled != 0;
        blend.enable = regs.blend.enable[per_target ? rt : 0] != 0 &&
                               logic.Format(rt) == RasterPatch::LogicFormat::None
                           ? 1U
                           : 0U;
        if (per_target) {
            const auto& src = regs.blend_per_target[rt];
            blend.color_op = src.color_op;
            blend.alpha_op = src.alpha_op;
            blend.color_source = src.color_source;
            blend.color_dest = src.color_dest;
            blend.alpha_source = src.alpha_source;
            blend.alpha_dest = src.alpha_dest;
        } else {
            blend.color_op = regs.blend.color_op;
            blend.alpha_op = regs.blend.alpha_op;
            blend.color_source = regs.blend.color_source;
            blend.color_dest = regs.blend.color_dest;
            blend.alpha_source = regs.blend.alpha_source;
            blend.alpha_dest = regs.blend.alpha_dest;
        }
    }
    desc.topology_class = TopologyClass(topology);
    desc.rasterize = regs.rasterize_enable != 0 ? 1U : 0U;
    // Alpha to one in bit 1 (EDEN_NX2_METAL_ALPHA_TO_ONE=0: not set, as before).
    static const bool alpha_to_one = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ALPHA_TO_ONE");
        return value == nullptr || value[0] != '0';
    }();
    desc.alpha_to_coverage = (regs.anti_alias_alpha_control.alpha_to_coverage != 0 ? 1U : 0U) |
                             (alpha_to_one && regs.anti_alias_alpha_control.alpha_to_one != 0
                                  ? 2U
                                  : 0U);
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const auto& input = regs.vertex_attrib_format[index];
        VertexAttributeState& attribute = desc.attributes[index];
        attribute.enabled = input.constant != 0 || mesh ? 0U : 1U;
        attribute.stream = input.buffer;
        attribute.offset = input.offset;
        attribute.type = input.type;
        attribute.size = input.size;
    }
}

void FillStream(RenderPipelineDesc& desc, const Maxwell& regs, size_t index, bool bound,
                u32 stride) {
    VertexStreamState& stream = desc.streams[index];
    stream.enabled = bound ? 1U : 0U;
    stream.stride = stride;
    stream.divisor = regs.vertex_stream_instances.IsInstancingEnabled(index)
                         ? regs.vertex_streams[index].frequency
                         : 0U;
}

VertexProgramUse VertexUse(const StageProgram& program) {
    VertexProgramUse use{};
    for (u32 index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        use.loaded |= program.info.loads.Generic(index) ? 1U << index : 0U;
    }
    for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
        if (slot.kind == ShaderTranslate::ResourceKind::Buffer && slot.spill == ~0U) {
            use.resource_slots = (std::max)(use.resource_slots, slot.index + slot.count);
        }
    }
    const ShaderTranslate::TessellationSlots& slots = program.tessellation_slots;
    for (const u32 slot : {slots.output, slots.patch_output, slots.tess_factors, slots.input,
                           slots.patch_input, slots.indirect_params, slots.index}) {
        if (slot != ~0U) {
            use.resource_slots = (std::max)(use.resource_slots, slot + 1);
        }
    }
    return use;
}

void ApplyTessellation(RenderPipelineDesc& desc, const Tessellation& tessellation) {
    std::ranges::fill(desc.attributes, VertexAttributeState{});
    std::ranges::fill(desc.streams, VertexStreamState{});
    if (tessellation.domain) {
        // Drawn from the evaluated vertices as lines or points, not by Metal's tessellator.
        desc.tessellation = 0;
        desc.topology_class = tessellation.domain->emulation.lines ? 1U : 0U;
        return;
    }
    using Spacing = ShaderTranslate::TessellationModes::Spacing;
    // MTLTessellationPartitionModeInteger, FractionalOdd, FractionalEven; MTLWinding reversed, as
    // Vulkan's domain origin is the upper left (MoltenVK does the same).
    desc.tessellation = 1;
    desc.tess_partition = tessellation.modes.spacing == Spacing::FractionalOdd    ? 2U
                          : tessellation.modes.spacing == Spacing::FractionalEven ? 3U
                                                                                  : 1U;
    desc.tess_winding = tessellation.modes.clockwise ? 1U : 0U;
}

DepthStencilDesc DepthStencilState(const Maxwell& regs, bool has_depth, bool has_stencil) {
    auto desc = Zeroed<DepthStencilDesc>();
    desc.depth_test = regs.depth_test_enable != 0 && has_depth ? 1U : 0U;
    desc.depth_write = regs.depth_write_enabled != 0 ? 1U : 0U;
    desc.depth_func = regs.depth_test_func;
    desc.stencil_test = regs.stencil_enable != 0 && has_stencil ? 1U : 0U;
    desc.front = {regs.stencil_front_op.fail,  regs.stencil_front_op.zfail,
                  regs.stencil_front_op.zpass, regs.stencil_front_op.func,
                  regs.stencil_front_func_mask, regs.stencil_front_mask};
    desc.back = regs.stencil_two_side_enable != 0
                    ? StencilFaceState{regs.stencil_back_op.fail,  regs.stencil_back_op.zfail,
                                       regs.stencil_back_op.zpass, regs.stencil_back_op.func,
                                       regs.stencil_back_func_mask, regs.stencil_back_mask}
                    : desc.front;
    return desc;
}

} // namespace Metal
