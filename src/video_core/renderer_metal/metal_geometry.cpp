// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// See metal_geometry.h. Ported in design from nxnative's geometry-to-compute (MIT License,
// Copyright (c) 2026 seedds).

#include <algorithm>
#include <span>
#include <vector>

#include <fmt/format.h>

#include "common/div_ceil.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "video_core/renderer_metal/metal_geometry.h"

namespace Metal::Geometry {

namespace IR = Shader::IR;

namespace {

constexpr u32 PositionX = static_cast<u32>(IR::Attribute::PositionX);
constexpr u32 PointSizeWord = static_cast<u32>(IR::Attribute::PointSize);
constexpr u32 Generic0X = static_cast<u32>(IR::Attribute::Generic0X);
constexpr u32 PrimitiveIdWord = static_cast<u32>(IR::Attribute::PrimitiveId);

[[nodiscard]] bool RecordWord(u32 word) {
    return word == PointSizeWord || (word >= PositionX && word < PositionX + 4) ||
           (word >= Generic0X && word < Generic0X + 32 * 4);
}

constexpr u32 ClipDistance0Word = static_cast<u32>(IR::Attribute::ClipDistance0);

/// What a geometry program may write: the record words, the layer, the viewport index and the
/// clip distances.
[[nodiscard]] bool StoreWord(u32 word) {
    return RecordWord(word) || word == static_cast<u32>(IR::Attribute::Layer) ||
           word == static_cast<u32>(IR::Attribute::ViewportIndex) ||
           (word >= ClipDistance0Word && word < ClipDistance0Word + 8);
}

[[nodiscard]] std::optional<u32> SlotOf(const std::vector<u32>& words, u32 word) {
    const auto it = std::ranges::find(words, word);
    if (it == words.end()) {
        return std::nullopt;
    }
    return static_cast<u32>(it - words.begin());
}

/// An expression for float `word` of a capture kernel's output struct `v` (what GatherSource
/// does for transform feedback), or its default.
[[nodiscard]] std::string OutputExpression(
    u32 word, std::span<const ShaderTranslate::InterfaceVariable> outputs,
    const std::map<IR::Attribute, IR::Attribute>& legacy_mapping) {
    using Variable = ShaderTranslate::InterfaceVariable;
    constexpr u32 BuiltinPosition = 0;
    constexpr u32 BuiltinPointSize = 1;
    const auto mapped = legacy_mapping.find(static_cast<IR::Attribute>(word));
    if (mapped != legacy_mapping.end()) {
        word = static_cast<u32>(mapped->second);
    }
    const Variable* variable = nullptr;
    u32 element = 0;
    const auto find = [&](u32 location, u32 builtin) -> const Variable* {
        for (const Variable& candidate : outputs) {
            if (candidate.per_patch || candidate.name.empty()) {
                continue;
            }
            if (builtin == Variable::NoBuiltin ? candidate.builtin == Variable::NoBuiltin &&
                                                     candidate.location == location
                                               : candidate.builtin == builtin) {
                return &candidate;
            }
        }
        return nullptr;
    };
    if (word == PointSizeWord) {
        variable = find(0, BuiltinPointSize);
    } else if (word >= PositionX && word < PositionX + 4) {
        variable = find(0, BuiltinPosition);
        element = word - PositionX;
    } else if (word >= Generic0X && word < Generic0X + 32 * 4) {
        variable = find((word - Generic0X) / 4, Variable::NoBuiltin);
        element = (word - Generic0X) % 4;
    }
    if (variable == nullptr || element >= variable->vecsize) {
        return element == 3 ? "1.0f" : "0.0f";
    }
    static constexpr std::string_view Swizzle = "xyzw";
    return variable->vecsize == 1 ? fmt::format("float(v.{})", variable->name)
                                  : fmt::format("float(v.{}.{})", variable->name, Swizzle[element]);
}

} // Anonymous namespace

std::optional<Plan> Transform(IR::Program& program, u32 input_vertices, u32 input_step,
                              std::string& why) {
    if (program.stage != Shader::Stage::Geometry) {
        why = "not a geometry program";
        return std::nullopt;
    }
    if (program.invocations > 32) {
        why = "geometry program with more than 32 invocations";
        return std::nullopt;
    }
    if (program.output_vertices == 0 || program.output_vertices > 256) {
        why = "geometry program's output vertex count";
        return std::nullopt;
    }
    Plan plan;
    plan.input_vertices = input_vertices;
    plan.input_step = input_step;
    plan.invocations = (std::max)(program.invocations, 1U);
    plan.max_vertices = program.output_vertices;
    plan.index_slots = 2 * program.output_vertices + 2;
    switch (program.output_topology) {
    case Shader::OutputTopology::PointList:
        plan.output_topology = 1;
        break;
    case Shader::OutputTopology::LineStrip:
        plan.output_topology = 6;
        break;
    case Shader::OutputTopology::TriangleStrip:
        plan.output_topology = 3;
        break;
    default:
        why = "geometry output topology";
        return std::nullopt;
    }
    const Shader::Info& info = program.info;
    for (u32 word = 0; word < info.loads.mask.size(); ++word) {
        if (!info.loads.mask[word] || word == PrimitiveIdWord) {
            continue;
        }
        if (!RecordWord(word)) {
            why = fmt::format("geometry program reads attribute {}", word);
            return std::nullopt;
        }
        plan.input_words.push_back(word);
    }
    for (u32 word = 0; word < info.stores.mask.size(); ++word) {
        if (!info.stores.mask[word]) {
            continue;
        }
        if (!StoreWord(word)) {
            why = fmt::format("geometry program writes attribute {}", word);
            return std::nullopt;
        }
        plan.output_words.push_back(word);
    }
    if (plan.output_words.empty()) {
        why = "geometry program writes nothing";
        return std::nullopt;
    }
    // A program with no reads still takes a record slot for the indexing.
    if (plan.input_words.empty()) {
        plan.input_words.push_back(PositionX);
    }
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetAttributeIndexed:
            case IR::Opcode::SetAttributeIndexed:
                why = "geometry program with indexed attributes";
                return std::nullopt;
            case IR::Opcode::EmitVertex:
            case IR::Opcode::EndPrimitive:
                if (!inst.Arg(0).IsImmediate()) {
                    why = "geometry program with a computed output stream";
                    return std::nullopt;
                }
                break;
            default:
                break;
            }
        }
    }

    const u32 old_local = Common::DivCeil(program.local_memory_size, 4U);
    const u32 vertex_cursor = old_local;
    const u32 index_cursor = old_local + 1;
    const u32 output_base = old_local + 2;
    const u32 first_binding = static_cast<u32>(info.storage_buffers_descriptors.size());
    const u32 input_binding = first_binding;
    const u32 output_binding = first_binding + 1;
    const u32 index_binding = first_binding + 2;
    const u32 istride = plan.InputStride();
    const u32 ostride = plan.OutputStride();

    IR::Block* const entry = program.syntax_list.front().data.block;
    IR::IREmitter entry_ir{*entry, entry->begin()};
    const IR::U32 primitive{
        entry_ir.IAdd(entry_ir.IMul(entry_ir.WorkgroupIdX(), entry_ir.Imm32(GroupSize)),
                      entry_ir.LocalInvocationIdX())};
    // The invocation is the grid's second dimension; a thread's region is the primitive's
    // invocations after one another.
    const IR::U32 invocation{entry_ir.WorkgroupIdY()};
    const IR::U32 region{plan.invocations > 1
                             ? IR::U32{entry_ir.IAdd(
                                   entry_ir.IMul(primitive, entry_ir.Imm32(plan.invocations)),
                                   invocation)}
                             : primitive};
    entry_ir.WriteLocal(entry_ir.Imm32(vertex_cursor), entry_ir.Imm32(0U));
    entry_ir.WriteLocal(entry_ir.Imm32(index_cursor), entry_ir.Imm32(0U));
    for (u32 slot = 0; slot < ostride; ++slot) {
        entry_ir.WriteLocal(entry_ir.Imm32(output_base + slot), entry_ir.Imm32(0U));
    }

    const auto store = [](IR::IREmitter& ir, u32 binding, const IR::U32& word, const IR::U32& value,
                          IR::Block& block, IR::Block::iterator at) {
        const IR::U32 byte{ir.ShiftLeftLogical(word, ir.Imm32(2U))};
        block.PrependNewInst(at, IR::Opcode::WriteStorage32,
                             {IR::Value{binding}, IR::Value{byte}, IR::Value{value}});
    };
    const auto index_word = [&](IR::IREmitter& ir, const IR::U32& slot) {
        return IR::U32{ir.IAdd(ir.IMul(region, ir.Imm32(plan.IndexStride())),
                               ir.IAdd(ir.Imm32(1U), slot))};
    };

    for (IR::Block* const block : program.blocks) {
        for (auto it = block->begin(); it != block->end();) {
            IR::Inst& inst = *it;
            ++it;
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetAttribute:
            case IR::Opcode::GetAttributeU32: {
                const u32 word = static_cast<u32>(inst.Arg(0).Attribute());
                IR::U32 value;
                if (word == PrimitiveIdWord) {
                    value = primitive;
                } else {
                    const IR::U32 vertex{inst.Arg(1)};
                    const IR::U32 first{ir.IMul(primitive, ir.Imm32(plan.input_vertices))};
                    const IR::U32 record{ir.IAdd(first, vertex)};
                    const IR::U32 slot_word{ir.IAdd(ir.IMul(record, ir.Imm32(istride)),
                                                    ir.Imm32(*SlotOf(plan.input_words, word)))};
                    const IR::U32 byte{ir.ShiftLeftLogical(slot_word, ir.Imm32(2U))};
                    const auto loaded = block->PrependNewInst(
                        IR::Block::InstructionList::s_iterator_to(inst), IR::Opcode::LoadStorage32,
                        {IR::Value{input_binding}, IR::Value{byte}});
                    value = IR::U32{IR::Value{&*loaded}};
                }
                inst.ReplaceUsesWith(inst.GetOpcode() == IR::Opcode::GetAttribute
                                         ? IR::Value{ir.BitCast<IR::F32>(value)}
                                         : IR::Value{value});
                break;
            }
            case IR::Opcode::InvocationId:
                inst.ReplaceUsesWith(IR::Value{invocation});
                break;
            case IR::Opcode::SetAttribute: {
                const u32 word = static_cast<u32>(inst.Arg(0).Attribute());
                const auto slot = SlotOf(plan.output_words, word);
                if (slot) {
                    ir.WriteLocal(ir.Imm32(output_base + *slot),
                                  ir.BitCast<IR::U32>(IR::F32{inst.Arg(1)}));
                }
                inst.Invalidate();
                break;
            }
            case IR::Opcode::EmitVertex: {
                if (inst.Arg(0).U32() != 0) {
                    // Only stream 0 is rasterized; the others are for transform feedback.
                    inst.Invalidate();
                    break;
                }
                const IR::U32 vertex_count{ir.LoadLocal(ir.Imm32(vertex_cursor))};
                const IR::U32 index_count{ir.LoadLocal(ir.Imm32(index_cursor))};
                const IR::U32 slot{ir.UMin(vertex_count, ir.Imm32(plan.max_vertices - 1))};
                const IR::U32 vertex_id{
                    ir.IAdd(ir.IMul(region, ir.Imm32(plan.max_vertices)), slot)};
                const IR::U32 base{ir.IMul(vertex_id, ir.Imm32(ostride))};
                const auto at = IR::Block::InstructionList::s_iterator_to(inst);
                for (u32 s = 0; s < ostride; ++s) {
                    const IR::U32 value{ir.LoadLocal(ir.Imm32(output_base + s))};
                    store(ir, output_binding, IR::U32{ir.IAdd(base, ir.Imm32(s))}, value, *block,
                          at);
                }
                const IR::U32 index_slot{ir.UMin(index_count, ir.Imm32(plan.index_slots - 1))};
                store(ir, index_binding, index_word(ir, index_slot), vertex_id, *block, at);
                ir.WriteLocal(ir.Imm32(vertex_cursor), IR::U32{ir.IAdd(vertex_count, ir.Imm32(1U))});
                ir.WriteLocal(ir.Imm32(index_cursor), IR::U32{ir.IAdd(index_count, ir.Imm32(1U))});
                inst.Invalidate();
                break;
            }
            case IR::Opcode::EndPrimitive:
            case IR::Opcode::Epilogue: {
                if (inst.GetOpcode() == IR::Opcode::EndPrimitive && inst.Arg(0).U32() != 0) {
                    inst.Invalidate();
                    break;
                }
                const IR::U32 index_count{ir.LoadLocal(ir.Imm32(index_cursor))};
                const IR::U32 index_slot{ir.UMin(index_count, ir.Imm32(plan.index_slots - 1))};
                const auto at = IR::Block::InstructionList::s_iterator_to(inst);
                store(ir, index_binding, index_word(ir, index_slot), ir.Imm32(0xFFFFFFFFU), *block,
                      at);
                const IR::U32 next{ir.IAdd(index_count, ir.Imm32(1U))};
                ir.WriteLocal(ir.Imm32(index_cursor), next);
                if (inst.GetOpcode() == IR::Opcode::Epilogue) {
                    // The count of the thread's index region, what the compaction reads.
                    const IR::U32 count{ir.UMin(next, ir.Imm32(plan.index_slots))};
                    store(ir, index_binding,
                          IR::U32{ir.IMul(region, ir.Imm32(plan.IndexStride()))}, count, *block,
                          at);
                } else {
                    inst.Invalidate();
                }
                break;
            }
            default:
                break;
            }
        }
    }

    // The replaced reads are identities of what replaced them.
    Shader::Optimization::IdentityRemovalPass(program);
    program.stage = Shader::Stage::Compute;
    program.workgroup_size = {GroupSize, 1, 1};
    program.local_memory_size = (output_base + ostride) * 4;
    Shader::Info& out = program.info;
    out.loads.mask.reset();
    out.stores.mask.reset();
    out.passthrough.mask.reset();
    out.uses_workgroup_id = true;
    out.uses_local_invocation_id = true;
    out.uses_invocation_id = false;
    out.used_storage_buffer_types |= IR::Type::U32;
    for (u32 buffer = 0; buffer < ExtraBuffers; ++buffer) {
        out.storage_buffers_descriptors.push_back({.cbuf_index = 0,
                                                   .cbuf_offset = 0,
                                                   .count = 1,
                                                   .is_written = buffer != 0});
    }
    return plan;
}

std::string RecordsSource(const Plan& plan, std::string_view entry,
                          std::span<const ShaderTranslate::InterfaceVariable> outputs,
                          const std::map<IR::Attribute, IR::Attribute>& legacy_mapping) {
    std::string body;
    for (u32 slot = 0; slot < plan.InputStride(); ++slot) {
        body += fmt::format("    r[{}] = {};\n", slot,
                            OutputExpression(plan.input_words[slot], outputs, legacy_mapping));
    }
    return fmt::format(R"(
struct GeometryRecordsParams {{ uint records, vertices, step, count, primitives; }};
kernel void geo_records(device const {0}_out* src [[buffer({1})]], device float* dst [[buffer({2})]],
                        constant GeometryRecordsParams& j [[buffer({3})]],
                        uint n [[thread_position_in_grid]]) {{
    if (n >= j.records) {{
        return;
    }}
    const uint p = n / j.vertices;
    uint s = (p / j.primitives) * j.count + (p % j.primitives) * j.step + n % j.vertices;
    if (j.vertices == 6 && j.step == 2) {{
        // Triangle strips with adjacency: Vulkan's vertex order of primitive i of n.
        const uint i = p % j.primitives, c = n % 6, m = j.primitives;
        int v[6];
        const int b = int(2 * i);
        if (m == 1) {{
            const int t[6] = {{0, 1, 2, 5, 4, 3}};
            for (uint k = 0; k < 6; k++) v[k] = t[k];
        }} else if (i == 0) {{
            const int t[6] = {{0, 1, 2, 6, 4, 3}};
            for (uint k = 0; k < 6; k++) v[k] = t[k];
        }} else if (i < m - 1) {{
            if (i % 2 == 1) {{
                const int t[6] = {{0, 3, 4, 6, 2, -2}};
                for (uint k = 0; k < 6; k++) v[k] = b + t[k];
            }} else {{
                const int t[6] = {{0, -2, 2, 6, 4, 3}};
                for (uint k = 0; k < 6; k++) v[k] = b + t[k];
            }}
        }} else if (i % 2 == 1) {{
            const int t[6] = {{0, 3, 4, 5, 2, -2}};
            for (uint k = 0; k < 6; k++) v[k] = b + t[k];
        }} else {{
            const int t[6] = {{0, -2, 2, 5, 4, 3}};
            for (uint k = 0; k < 6; k++) v[k] = b + t[k];
        }}
        s = (p / j.primitives) * j.count + uint(v[c]);
    }}
    device const {0}_out& v = src[s];
    device float* r = dst + ulong(n) * {4}u;
{5}}}
)",
                       entry, RecordsSourceSlot, RecordsOutputSlot, RecordsCountSlot,
                       plan.InputStride(), body);
}

std::string XfbSource(const Plan& plan, const Xfb::Layout& layout,
                      const std::map<IR::Attribute, IR::Attribute>& legacy_mapping) {
    std::string body;
    for (const Xfb::Varying& varying : Xfb::Varyings(layout)) {
        for (u32 component = 0; component < varying.components; ++component) {
            u32 word = varying.attribute + component;
            const auto mapped = legacy_mapping.find(static_cast<IR::Attribute>(word));
            if (mapped != legacy_mapping.end()) {
                word = static_cast<u32>(mapped->second);
            }
            const auto slot = SlotOf(plan.output_words, word);
            body += fmt::format("    *(device float*)(dst{} + ulong(n) * {}u + {}u) = {};\n",
                                varying.buffer, layout.Stride(varying.buffer),
                                varying.offset + component * 4,
                                slot ? fmt::format("r[{}]", *slot) : std::string{"0.0f"});
        }
    }
    return fmt::format(R"(
struct GeometryXfbParams {{ uint capacity[4]; uint written[4]; uint vertices; }};
kernel void geo_xfb(device const float* rec [[buffer(0)]], device const uint* list [[buffer(1)]],
                    device uchar* dst0 [[buffer(2)]], device uchar* dst1 [[buffer(3)]],
                    device uchar* dst2 [[buffer(4)]], device uchar* dst3 [[buffer(5)]],
                    device const uint* args [[buffer(6)]], constant GeometryXfbParams& j [[buffer(7)]],
                    uint n [[thread_position_in_grid]]) {{
    const uint stride[4] = {{{0}u, {1}u, {2}u, {3}u}};
    uint count = args[0];
    for (uint b = 0; b < 4; b++) {{
        if (stride[b] != 0) {{
            const uint room = j.capacity[b] - min(j.capacity[b], j.written[b]);
            count = min(count, room / stride[b]);
        }}
    }}
    count -= count % j.vertices;
    if (n >= count) {{
        return;
    }}
    device const float* r = rec + ulong(list[n]) * {4}u;
{5}}}
)",
                       layout.Stride(0), layout.Stride(1), layout.Stride(2), layout.Stride(3),
                       plan.OutputStride(), body);
}

std::string ReaderSource(const Plan& plan, bool convert_depth, bool flip_y,
                         float default_point_size) {
    const auto word = [&](u32 attribute, const char* fallback) {
        const auto slot = SlotOf(plan.output_words, attribute);
        return slot ? fmt::format("r[{}]", *slot) : std::string{fallback};
    };
    const bool points = plan.output_topology == 1;
    std::string members;
    std::string assignments;
    std::vector<u32> locations;
    for (const u32 attribute : plan.output_words) {
        if (attribute >= Generic0X && attribute < Generic0X + 32 * 4) {
            const u32 location = (attribute - Generic0X) / 4;
            if (std::ranges::find(locations, location) == locations.end()) {
                locations.push_back(location);
            }
        }
    }
    const u32 layer_word = static_cast<u32>(IR::Attribute::Layer);
    const u32 viewport_word = static_cast<u32>(IR::Attribute::ViewportIndex);
    if (const auto slot = SlotOf(plan.output_words, layer_word)) {
        members += "    uint layer [[render_target_array_index]];\n";
        assignments += fmt::format("    o.layer = as_type<uint>(r[{}]);\n", *slot);
    }
    if (const auto slot = SlotOf(plan.output_words, viewport_word)) {
        members += "    uint viewport [[viewport_array_index]];\n";
        assignments += fmt::format("    o.viewport = as_type<uint>(r[{}]);\n", *slot);
    }
    u32 clip_count = 0;
    for (u32 index = 0; index < 8; ++index) {
        if (SlotOf(plan.output_words, ClipDistance0Word + index)) {
            clip_count = index + 1;
        }
    }
    if (clip_count != 0) {
        members += fmt::format("    float clip [[clip_distance]] [{}];\n", clip_count);
        for (u32 index = 0; index < clip_count; ++index) {
            assignments += fmt::format("    o.clip[{}] = {};\n", index,
                                       word(ClipDistance0Word + index, "1.0f"));
        }
    }
    for (const u32 location : locations) {
        members += fmt::format("    float4 loc{0} [[user(locn{0})]];\n", location);
        std::string components;
        for (u32 element = 0; element < 4; ++element) {
            components += (element != 0 ? ", " : "") +
                          word(Generic0X + location * 4 + element, element == 3 ? "1.0f" : "0.0f");
        }
        assignments += fmt::format("    o.loc{} = float4({});\n", location, components);
    }
    std::string position;
    for (u32 element = 0; element < 4; ++element) {
        position += (element != 0 ? ", " : "") + word(PositionX + element, element == 3 ? "1.0f" : "0.0f");
    }
    return fmt::format(R"(#include <metal_stdlib>
using namespace metal;
struct GeometryReaderOut {{
    float4 position [[position]];
{8}{0}}};
vertex GeometryReaderOut geo_reader(uint vid [[vertex_id]], device const float* rec [[buffer(0)]]) {{
    device const float* r = rec + ulong(vid) * {1}u;
    GeometryReaderOut o;
    o.position = float4({2});
{3}{4}{5}{6}{7}    return o;
}}
)",
                       members, plan.OutputStride(), position, assignments,
                       convert_depth ? "    o.position.z = (o.position.z + o.position.w) * 0.5f;\n" : "",
                       flip_y ? "    o.position.y = -o.position.y;\n" : "",
                       points ? fmt::format("    o.point_size = {};\n",
                                            word(PointSizeWord,
                                                 fmt::format("{:.9e}f", default_point_size).c_str()))
                              : std::string{},
                       "",
                       points ? "    float point_size [[point_size]];\n" : "");
}

} // namespace Metal::Geometry
