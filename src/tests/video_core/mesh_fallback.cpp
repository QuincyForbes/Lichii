// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"

namespace {
using namespace Shader;
namespace SPIRV = Shader::Backend::SPIRV;

constexpr u32 OUTPUT_BINDING = 4;
constexpr u32 PAYLOAD_BINDING = 5;
constexpr u32 OFFSETS_BINDING = 6;
constexpr u32 ARGS_BINDING = 7;

struct Instruction {
    spv::Op op;
    std::span<const u32> operands;
};

std::vector<Instruction> Instructions(const std::vector<u32>& code) {
    std::vector<Instruction> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 count{code[offset] >> 16};
        REQUIRE(count != 0);
        REQUIRE(offset + count <= code.size());
        result.push_back({static_cast<spv::Op>(code[offset] & 0xffff),
                          std::span<const u32>(code).subspan(offset + 1, count - 1)});
        offset += count;
    }
    return result;
}

spv::ExecutionModel Model(const std::vector<u32>& code) {
    for (const Instruction& inst : Instructions(code)) {
        if (inst.op == spv::Op::OpEntryPoint) {
            return static_cast<spv::ExecutionModel>(inst.operands[0]);
        }
    }
    FAIL("no entry point");
    return {};
}

bool HasCapability(const std::vector<u32>& code, spv::Capability capability) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpCapability &&
               inst.operands[0] == static_cast<u32>(capability);
    });
}

bool HasExtension(const std::vector<u32>& code, std::string_view name) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpExtension &&
               std::string_view{reinterpret_cast<const char*>(inst.operands.data())} == name;
    });
}

bool HasVariable(const std::vector<u32>& code, spv::StorageClass storage_class) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpVariable &&
               inst.operands[2] == static_cast<u32>(storage_class);
    });
}

bool UsesStorageClass(const std::vector<u32>& code, spv::StorageClass storage_class) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return (inst.op == spv::Op::OpTypePointer &&
                inst.operands[1] == static_cast<u32>(storage_class)) ||
               (inst.op == spv::Op::OpVariable &&
                inst.operands[2] == static_cast<u32>(storage_class));
    });
}

std::vector<u32> Bindings(const std::vector<u32>& code) {
    std::vector<u32> bindings;
    for (const Instruction& inst : Instructions(code)) {
        if (inst.op == spv::Op::OpDecorate &&
            inst.operands[1] == static_cast<u32>(spv::Decoration::Binding)) {
            bindings.push_back(inst.operands[2]);
        }
    }
    std::ranges::sort(bindings);
    return bindings;
}

// EDEN_MESH_FALLBACK_TEST_OUTPUT=<dir>: write the modules for spirv-val.
void Dump(const std::vector<u32>& code, std::string_view name) {
    const char* const directory{std::getenv("EDEN_MESH_FALLBACK_TEST_OUTPUT")};
    if (!directory) {
        return;
    }
    std::filesystem::create_directories(directory);
    std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                       std::ios::binary};
    file.write(reinterpret_cast<const char*>(code.data()),
               static_cast<std::streamsize>(code.size() * sizeof(u32)));
    REQUIRE(file.good());
}

MeshInfo TriangleMesh() {
    MeshInfo mesh;
    mesh.payload_words = 32;
    mesh.max_vertices = 3;
    mesh.max_primitives = 1;
    mesh.topology = 4;
    for (const IR::Attribute attribute : {IR::Attribute::PositionX, IR::Attribute::PositionY,
                                          IR::Attribute::PositionZ, IR::Attribute::PositionW}) {
        mesh.vertex_lines.push_back(static_cast<u32>(attribute));
    }
    return mesh;
}

Profile MakeProfile() {
    Profile profile;
    profile.supported_spirv = 0x00010500;
    return profile;
}

// A one-triangle mesh program (or the task program launching it), as the frontend lowers them.
// With lane_id the mesh program writes its lane as the vertex's first attribute word.
std::vector<u32> EmitProgram(Stage stage, bool as_compute, bool with_task,
                             Profile profile = MakeProfile(), bool lane_id = false,
                             bool direct_outputs = false, const MeshInfo& mesh = TriangleMesh()) {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = stage;
    program.workgroup_size = {32, 1, 1};
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    program.mesh = mesh;
    program.info.is_task = stage == Stage::Task;
    program.info.is_mesh = stage == Stage::Mesh;
    program.info.uses_workgroup_id = true;
    program.info.uses_local_invocation_id = true;
    IR::IREmitter ir{block};
    ir.Prologue();
    if (stage == Stage::Mesh) {
        const IR::U32 index{lane_id ? ir.LaneId() : ir.IsbeRead(ir.Imm32(0))};
        program.info.uses_subgroup_invocation_id = lane_id;
        ir.IsbeWriteAttribute(ir.Imm32(0), index);
        ir.IsbeWriteMapU8(ir.Imm32(4), ir.Imm32(0));
        ir.IsbeWriteMapU32(ir.Imm32(3), ir.Imm32(1));
    } else {
        ir.IsbeWriteAttribute(ir.Imm32(4), ir.Imm32(1)); // one mesh workgroup
    }
    ir.Epilogue();

    RuntimeInfo runtime_info;
    runtime_info.mesh_as_compute = as_compute;
    runtime_info.mesh_direct_outputs = direct_outputs;
    runtime_info.mesh_output_binding = OUTPUT_BINDING;
    if (with_task) {
        runtime_info.mesh_task_payload_binding = PAYLOAD_BINDING;
        runtime_info.mesh_task_offsets_binding = OFFSETS_BINDING;
        runtime_info.mesh_task_args_binding = ARGS_BINDING;
        runtime_info.mesh_task_payload_stride = program.mesh.payload_words;
    }
    Backend::Bindings bindings;
    return SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

void RequireNoMeshShading(const std::vector<u32>& code) {
    REQUIRE(Model(code) == spv::ExecutionModel::GLCompute);
    REQUIRE_FALSE(HasCapability(code, spv::Capability::MeshShadingEXT));
    REQUIRE_FALSE(HasExtension(code, "SPV_EXT_mesh_shader"));
    REQUIRE_FALSE(UsesStorageClass(code, spv::StorageClass::TaskPayloadWorkgroupEXT));
    REQUIRE_FALSE(HasVariable(code, spv::StorageClass::Output));
}
} // Anonymous namespace

TEST_CASE("Mesh fallback leaves native task and mesh programs as mesh shaders",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> mesh{EmitProgram(Stage::Mesh, false, false)};
    Dump(mesh, "native-mesh");
    REQUIRE(Model(mesh) == spv::ExecutionModel::MeshEXT);
    REQUIRE(HasCapability(mesh, spv::Capability::MeshShadingEXT));
    REQUIRE(Bindings(mesh).empty());

    const std::vector<u32> task{EmitProgram(Stage::Task, false, false)};
    Dump(task, "native-task");
    REQUIRE(Model(task) == spv::ExecutionModel::TaskEXT);
    REQUIRE(UsesStorageClass(task, spv::StorageClass::TaskPayloadWorkgroupEXT));
}

TEST_CASE("Native mesh programs with direct outputs keep no attribute region",
          "[video_core][spirv][mesh_fallback]") {
    const auto workgroup_variables = [](const std::vector<u32>& code) {
        return std::ranges::count_if(Instructions(code), [](const Instruction& inst) {
            return inst.op == spv::Op::OpVariable &&
                   inst.operands[2] == static_cast<u32>(spv::StorageClass::Workgroup);
        });
    };
    const auto calls = [](const std::vector<u32>& code) {
        return std::ranges::count_if(Instructions(code), [](const Instruction& inst) {
            return inst.op == spv::Op::OpFunctionCall;
        });
    };
    const std::vector<u32> copied{EmitProgram(Stage::Mesh, false, false)};
    const std::vector<u32> direct{
        EmitProgram(Stage::Mesh, false, false, MakeProfile(), false, true)};
    Dump(direct, "native-mesh-direct");
    REQUIRE(Model(direct) == spv::ExecutionModel::MeshEXT);
    // The map region and the primitive count stay; the attribute region goes.
    REQUIRE(workgroup_variables(copied) == 3);
    REQUIRE(workgroup_variables(direct) == 2);
    REQUIRE(calls(direct) == calls(copied) + 1);

    // The emulation keeps its region whatever the switch says.
    const std::vector<u32> emulated{
        EmitProgram(Stage::Mesh, true, false, MakeProfile(), false, true)};
    RequireNoMeshShading(emulated);
}

TEST_CASE("Mesh fallback emits mesh programs as compute shaders writing storage buffers",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> alone{EmitProgram(Stage::Mesh, true, false)};
    Dump(alone, "fallback-mesh");
    RequireNoMeshShading(alone);
    REQUIRE(Bindings(alone) == std::vector<u32>{OUTPUT_BINDING});

    const std::vector<u32> after_task{EmitProgram(Stage::Mesh, true, true)};
    Dump(after_task, "fallback-mesh-after-task");
    RequireNoMeshShading(after_task);
    REQUIRE(Bindings(after_task) ==
            std::vector<u32>{OUTPUT_BINDING, PAYLOAD_BINDING, OFFSETS_BINDING, ARGS_BINDING});
}

TEST_CASE("Mesh fallback programs use compute subgroup support",
          "[video_core][spirv][mesh_fallback]") {
    // A device without VK_EXT_mesh_shader reports subgroup support for compute, not mesh.
    Profile profile{MakeProfile()};
    profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
    const std::vector<u32> code{EmitProgram(Stage::Mesh, true, false, profile, true)};
    Dump(code, "fallback-mesh-lane-id");
    RequireNoMeshShading(code);
    REQUIRE(HasCapability(code, spv::Capability::GroupNonUniform));
}

TEST_CASE("Mesh fallback emits task programs as compute shaders writing their payload",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> task{EmitProgram(Stage::Task, true, true)};
    Dump(task, "fallback-task");
    RequireNoMeshShading(task);
    REQUIRE(Bindings(task) == std::vector<u32>{PAYLOAD_BINDING});
}

TEST_CASE("Mesh fallback vertex shader pulls the compute output",
          "[video_core][spirv][mesh_fallback]") {
    const MeshInfo mesh{TriangleMesh()};
    const std::vector<u32> alone{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false)};
    Dump(alone, "fallback-vertex");
    REQUIRE(Model(alone) == spv::ExecutionModel::Vertex);
    REQUIRE(Bindings(alone) == std::vector<u32>{OUTPUT_BINDING});

    const std::vector<u32> after_task{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ARGS_BINDING, true)};
    Dump(after_task, "fallback-vertex-after-task");
    REQUIRE(Model(after_task) == spv::ExecutionModel::Vertex);
    REQUIRE(Bindings(after_task) == std::vector<u32>{OUTPUT_BINDING, ARGS_BINDING});
}

namespace {
bool HasBuiltIn(const std::vector<u32>& code, spv::BuiltIn builtin) {
    return std::ranges::any_of(Instructions(code), [&](const Instruction& inst) {
        return inst.op == spv::Op::OpDecorate &&
               inst.operands[1] == static_cast<u32>(spv::Decoration::BuiltIn) &&
               inst.operands[2] == static_cast<u32>(builtin);
    });
}
} // Anonymous namespace

TEST_CASE("Mesh fallback blocks hold one index per point and two per line",
          "[video_core][spirv][mesh_fallback]") {
    MeshInfo mesh{TriangleMesh()};
    mesh.max_primitives = 8;
    const u32 vertex_words{mesh.max_vertices * static_cast<u32>(mesh.vertex_lines.size())};
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8 * 3);
    mesh.topology = 1;
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8 * 2);
    mesh.topology = 0;
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8);
}

namespace {
// TriangleMesh() with generic 1's x and generic 3's x and y written per primitive.
MeshInfo PerPrimitiveMesh() {
    MeshInfo mesh{TriangleMesh()};
    mesh.max_primitives = 8;
    for (const IR::Attribute attribute :
         {IR::Attribute::Generic1X, IR::Attribute::Generic3X, IR::Attribute::Generic3Y}) {
        mesh.primitive_lines.push_back(static_cast<u32>(attribute));
    }
    return mesh;
}

// The ids decorated with `decoration`.
std::vector<u32> Decorated(const std::vector<u32>& code, spv::Decoration decoration) {
    std::vector<u32> ids;
    for (const Instruction& inst : Instructions(code)) {
        if (inst.op == spv::Op::OpDecorate && inst.operands[1] == static_cast<u32>(decoration)) {
            ids.push_back(inst.operands[0]);
        }
    }
    std::ranges::sort(ids);
    return ids;
}

// The locations of the variables decorated with `decoration`.
std::vector<u32> LocationsDecorated(const std::vector<u32>& code, spv::Decoration decoration) {
    const std::vector<u32> ids{Decorated(code, decoration)};
    std::vector<u32> locations;
    for (const Instruction& inst : Instructions(code)) {
        if (inst.op == spv::Op::OpDecorate &&
            inst.operands[1] == static_cast<u32>(spv::Decoration::Location) &&
            std::ranges::binary_search(ids, inst.operands[0])) {
            locations.push_back(inst.operands[2]);
        }
    }
    std::ranges::sort(locations);
    return locations;
}

// A pixel program reading generics 1 and 3 after a mesh program that writes them per primitive.
std::vector<u32> EmitPerPrimitivePixel(bool emulated, bool flat) {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = Stage::Fragment;
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    RuntimeInfo runtime_info;
    for (const IR::Attribute attribute : {IR::Attribute::Generic1X, IR::Attribute::Generic3X}) {
        program.info.loads.Set(attribute);
        runtime_info.previous_stage_stores.Set(attribute);
        runtime_info.per_primitive_generics.set(IR::GenericAttributeIndex(attribute));
    }
    runtime_info.mesh_emulation_inputs = emulated;
    runtime_info.flat_per_primitive_inputs = flat;
    Backend::Bindings bindings;
    return SPIRV::EmitSPIRV(MakeProfile(), runtime_info, program, bindings);
}
} // Anonymous namespace

TEST_CASE("Mesh fallback blocks end with the primitives' own lines",
          "[video_core][spirv][mesh_fallback]") {
    const MeshInfo mesh{PerPrimitiveMesh()};
    const u32 vertex_words{mesh.max_vertices * static_cast<u32>(mesh.vertex_lines.size())};
    REQUIRE(SPIRV::MeshEmulationBlockWords(mesh) == vertex_words + 8 * 3 + 8 * 3);
}

TEST_CASE("Mesh fallback takes per-primitive generics only",
          "[video_core][spirv][mesh_fallback]") {
    REQUIRE(SPIRV::MeshEmulationTakesPrimitiveLines(TriangleMesh()));
    REQUIRE(SPIRV::MeshEmulationTakesPrimitiveLines(PerPrimitiveMesh()));
    for (const IR::Attribute attribute :
         {IR::Attribute::PrimitiveId, IR::Attribute::Layer, IR::Attribute::ViewportIndex}) {
        MeshInfo mesh{PerPrimitiveMesh()};
        mesh.primitive_lines.push_back(static_cast<u32>(attribute));
        REQUIRE_FALSE(SPIRV::MeshEmulationTakesPrimitiveLines(mesh));
    }
    // A location is Flat as a whole: not one written per vertex and per primitive.
    MeshInfo mixed{PerPrimitiveMesh()};
    mixed.vertex_lines.push_back(static_cast<u32>(IR::Attribute::Generic3Z));
    REQUIRE_FALSE(SPIRV::MeshEmulationTakesPrimitiveLines(mixed));
}

TEST_CASE("Mesh fallback draws per-primitive generics as Flat varyings",
          "[video_core][spirv][mesh_fallback]") {
    MeshInfo mesh{PerPrimitiveMesh()};
    mesh.vertex_lines.push_back(static_cast<u32>(IR::Attribute::Generic2X));

    const std::vector<u32> compute{
        EmitProgram(Stage::Mesh, true, false, MakeProfile(), false, false, mesh)};
    Dump(compute, "fallback-mesh-per-primitive");
    RequireNoMeshShading(compute);
    REQUIRE(Bindings(compute) == std::vector<u32>{OUTPUT_BINDING});

    const std::vector<u32> vertex{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false)};
    Dump(vertex, "fallback-vertex-per-primitive");
    REQUIRE(Model(vertex) == spv::ExecutionModel::Vertex);
    // Generic 2 is written per vertex and stays interpolated.
    REQUIRE(LocationsDecorated(vertex, spv::Decoration::Flat) == std::vector<u32>{1, 3});
    REQUIRE(Decorated(vertex, spv::Decoration::PerPrimitiveEXT).empty());

    // Without primitive lines the vertex shader is the one it was.
    REQUIRE(Decorated(SPIRV::EmitMeshEmulationVertex(MakeProfile(), TriangleMesh(),
                                                     OUTPUT_BINDING, ~0U, false),
                      spv::Decoration::Flat)
                .empty());
}

TEST_CASE("Pixel programs read per-primitive generics Flat after an emulated mesh program",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> native{EmitPerPrimitivePixel(false, false)};
    Dump(native, "native-pixel-per-primitive");
    REQUIRE(HasCapability(native, spv::Capability::MeshShadingEXT));
    REQUIRE(LocationsDecorated(native, spv::Decoration::PerPrimitiveEXT) ==
            std::vector<u32>{1, 3});
    REQUIRE(Decorated(native, spv::Decoration::Flat).empty());

    const std::vector<u32> native_flat{EmitPerPrimitivePixel(false, true)};
    REQUIRE(LocationsDecorated(native_flat, spv::Decoration::PerPrimitiveEXT) ==
            std::vector<u32>{1, 3});
    REQUIRE(LocationsDecorated(native_flat, spv::Decoration::Flat) == std::vector<u32>{1, 3});

    const std::vector<u32> emulated{EmitPerPrimitivePixel(true, true)};
    Dump(emulated, "fallback-pixel-per-primitive");
    REQUIRE(Model(emulated) == spv::ExecutionModel::Fragment);
    REQUIRE_FALSE(HasCapability(emulated, spv::Capability::MeshShadingEXT));
    REQUIRE_FALSE(HasExtension(emulated, "SPV_EXT_mesh_shader"));
    REQUIRE(Decorated(emulated, spv::Decoration::PerPrimitiveEXT).empty());
    REQUIRE(LocationsDecorated(emulated, spv::Decoration::Flat) == std::vector<u32>{1, 3});
}

TEST_CASE("Mesh fallback vertex shader gives points a size and culls unused points and lines",
          "[video_core][spirv][mesh_fallback]") {
    MeshInfo mesh{TriangleMesh()};
    REQUIRE_FALSE(HasBuiltIn(
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false),
        spv::BuiltIn::PointSize));
    mesh.topology = 1;
    const std::vector<u32> lines{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ~0U, false)};
    Dump(lines, "fallback-vertex-lines");
    REQUIRE_FALSE(HasBuiltIn(lines, spv::BuiltIn::PointSize));
    REQUIRE(HasBuiltIn(lines, spv::BuiltIn::ClipDistance));
    mesh.topology = 0;
    const std::vector<u32> points{
        SPIRV::EmitMeshEmulationVertex(MakeProfile(), mesh, OUTPUT_BINDING, ARGS_BINDING, false,
                                       2.0f)};
    Dump(points, "fallback-vertex-points");
    REQUIRE(Model(points) == spv::ExecutionModel::Vertex);
    REQUIRE(HasBuiltIn(points, spv::BuiltIn::PointSize));
    REQUIRE(HasBuiltIn(points, spv::BuiltIn::ClipDistance));
}

TEST_CASE("Mesh fallback distribution pass binds its buffers and constants",
          "[video_core][spirv][mesh_fallback]") {
    const std::vector<u32> code{SPIRV::EmitMeshEmulationDistribution(MakeProfile())};
    Dump(code, "fallback-distribution");
    REQUIRE(Model(code) == spv::ExecutionModel::GLCompute);
    REQUIRE(Bindings(code) == std::vector<u32>{0, 1, 2, 3});
    REQUIRE(UsesStorageClass(code, spv::StorageClass::PushConstant));
    const auto instructions{Instructions(code)};
    const bool has_constants{std::ranges::any_of(instructions, [](const Instruction& inst) {
        return inst.op == spv::Op::OpTypeStruct &&
               inst.operands.size() == 1 + sizeof(SPIRV::MeshEmulationDistributionConstants) / 4;
    })};
    REQUIRE(has_constants);
}

TEST_CASE("Mesh fallback reserves auxiliary storage buffers in the descriptor set limit",
          "[video_core][mesh_fallback]") {
    using Vulkan::FitsStorageBufferSetLimit;

    // Ordinary graphics pipelines do not add emulation descriptors.
    REQUIRE(FitsStorageBufferSetLimit(8, 0, 8));
    REQUIRE_FALSE(FitsStorageBufferSetLimit(9, 0, 8));

    // Mesh only: one output buffer. Task+mesh: four auxiliary bindings.
    REQUIRE(FitsStorageBufferSetLimit(7, 1, 8));
    REQUIRE_FALSE(FitsStorageBufferSetLimit(8, 1, 8));
    REQUIRE(FitsStorageBufferSetLimit(4, 4, 8));
    REQUIRE_FALSE(FitsStorageBufferSetLimit(5, 4, 8));

    // No limit is represented by zero. Reject huge inputs without addition overflow.
    REQUIRE(FitsStorageBufferSetLimit(200, 4, 0));
    REQUIRE_FALSE(FitsStorageBufferSetLimit((std::numeric_limits<size_t>::max)(), 4, 8));
}

TEST_CASE("Native mesh programs on MoltenVK count their threadgroup memory",
          "[video_core][mesh_fallback]") {
    using Vulkan::MetalMeshThreadgroupBytes;

    Vulkan::NativeMeshShape shape{
        .max_vertices = 64,
        .max_primitives = 126,
        .primitive_vertices = 3,
        .vertex_lines = 8,
        .primitive_lines = 0,
        .vertex_locations = 2,
        .primitive_locations = 0,
        .shared_memory_bytes = 0,
    };
    // Workgroup: (2 * 8 * 32 attribute words + 4 + 3 * 126 map words + 1) * 4 = 3580 B.
    // Metal mesh object: 64 * 2 * 16 + 126 * 16 + 8 = 4072 B.
    REQUIRE(MetalMeshThreadgroupBytes(shape) == 3580 + 4072);
    shape.shared_memory_bytes = 1024;
    REQUIRE(MetalMeshThreadgroupBytes(shape) == 3580 + 4072 + 1024);
}

TEST_CASE("Ocarina of Time's mesh programs fit Metal with direct outputs",
          "[video_core][mesh_fallback]") {
    using Vulkan::MetalMeshThreadgroupBytes;
    constexpr std::uint64_t metal_threadgroup_bytes = 32768;

    // The three shapes an iPhone 17 Pro Max emulated (build 62 log, "exceeds the native mesh
    // limits"): vertices, triangles, stage buffer lines (position + generics), vec4 outputs.
    struct Case {
        Vulkan::NativeMeshShape shape;
        std::uint64_t with_region;
        std::uint64_t direct;
    };
    const auto shape = [](std::uint32_t vertices, std::uint32_t primitives, std::uint32_t lines,
                          std::uint32_t locations) {
        return Vulkan::NativeMeshShape{
            .max_vertices = vertices,
            .max_primitives = primitives,
            .primitive_vertices = 3,
            .vertex_lines = lines,
            .primitive_lines = 0,
            .vertex_locations = locations,
            .primitive_locations = 0,
            .shared_memory_bytes = 0,
        };
    };
    // Direct: (4 + 3 * primitives map words + 1) * 4, plus the Metal mesh object.
    const std::array cases{
        Case{shape(128, 64, 25, 10), 35100, 788 + 21512},
        Case{shape(160, 128, 24, 9), 42012, 1556 + 25096},
        Case{shape(128, 64, 23, 10), 34076, 788 + 21512},
    };
    for (Case test : cases) {
        REQUIRE(MetalMeshThreadgroupBytes(test.shape) == test.with_region);
        REQUIRE(test.with_region > metal_threadgroup_bytes);
        test.shape.direct_outputs = true;
        REQUIRE(MetalMeshThreadgroupBytes(test.shape) == test.direct);
        REQUIRE(test.direct <= metal_threadgroup_bytes);
    }
}
