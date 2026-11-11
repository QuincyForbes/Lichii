// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/profile.h"

namespace {
using namespace Shader;

class StubEnvironment final : public Environment {
public:
    StubEnvironment() {
        stage = Stage::Compute;
    }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    TextureType ReadTextureType(u32) override { return {}; }
    TexturePixelFormat ReadTexturePixelFormat(u32) override { return {}; }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 0; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

// SPIR-V instructions with opcode `opcode`, as their operand words.
std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t i = 5; i < code.size();) {
        const u32 count = code[i] >> 16;
        REQUIRE(count > 0);
        REQUIRE(i + count <= code.size());
        if ((code[i] & 0xffff) == opcode) {
            result.emplace_back(code.begin() + i + 1, code.begin() + i + count);
        }
        i += count;
    }
    return result;
}

// Verifies every SPIR-V basic block has exactly one terminator (SPIRV-Cross throws in
// Parser::parse if a second terminator such as OpBranch follows OpReturn/OpReturnValue,
// which SIGSEGVs MoltenVK in SPIRVToMSLConverter::convert).
void RequireSingleBlockTerminators(const std::vector<u32>& code) {
    constexpr u32 OpFunction = 54;
    constexpr u32 OpFunctionParameter = 55;
    constexpr u32 OpFunctionEnd = 56;
    constexpr u32 OpLabel = 248;
    const auto is_terminator = [](u32 op) { return op >= 249 && op <= 255; };

    bool in_function = false;
    bool in_block = false;
    for (size_t i = 5; i < code.size();) {
        const u32 count = code[i] >> 16;
        const u32 op = code[i] & 0xffff;
        REQUIRE(count > 0);
        REQUIRE(i + count <= code.size());
        if (op == OpFunction) {
            in_function = true;
            in_block = false;
        } else if (op == OpFunctionParameter) {
            REQUIRE(in_function);
            REQUIRE_FALSE(in_block);
        } else if (op == OpFunctionEnd) {
            REQUIRE_FALSE(in_block);
            in_function = false;
        } else if (in_function) {
            if (op == OpLabel) {
                REQUIRE_FALSE(in_block);
                in_block = true;
            } else {
                REQUIRE(in_block);
                if (is_terminator(op)) {
                    in_block = false;
                }
            }
        }
        i += count;
    }
}

// A compute program whose atomic address comes from global memory, so the storage buffer pass
// tracks the load but not the atomic.
std::vector<u32> EmitUntrackedAtomic(IR::Opcode opcode) {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    const auto low = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x110));
    const auto high = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x114));
    const auto loaded = ir.LoadGlobal64(ir.PackUint2x32(ir.CompositeConstruct(low, high)));
    const IR::U64 address{ir.PackUint2x32(
        ir.CompositeConstruct(ir.CompositeExtract(loaded, 0), ir.CompositeExtract(loaded, 1)))};
    const IR::Value value{opcode == IR::Opcode::GlobalAtomicAddF32 ? IR::Value{ir.Imm32(1.0f)}
                                                                   : IR::Value{ir.Imm32(1)}};
    const IR::Value previous{&*block.PrependNewInst(block.end(), opcode, {address, value})};
    const IR::U32 result{opcode == IR::Opcode::GlobalAtomicAddF32
                             ? IR::U32{ir.BitCast<IR::U32>(IR::F32{previous})}
                             : IR::U32{previous}};
    ir.WriteGlobal32(address, result);

    IR::Program program;
    program.stage = Stage::Compute;
    program.workgroup_size = {1, 1, 1};
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    StubEnvironment env;
    Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
    REQUIRE(std::ranges::any_of(block.Instructions(), [&](const IR::Inst& inst) {
        return inst.GetOpcode() == opcode;
    }));
    Optimization::CollectShaderInfoPass(env, program);
    program.info.native_global_bounds = true;
    program.info.constant_buffer_descriptors = {{0, 1}};

    Profile profile{};
    profile.supported_spirv = 0x00010300;
    profile.support_int64 = true;
    profile.support_descriptor_aliasing = true;
    profile.unified_descriptor_binding = true;
    profile.min_ssbo_alignment = 16;
    return Backend::SPIRV::EmitSPIRV(profile, program);
}
} // namespace

TEST_CASE("Untracked 32-bit global atomics route to their storage buffer",
          "[shader_recompiler][global_memory]") {
    constexpr u32 OpFunctionCall = 57;
    constexpr u32 OpAtomicCompareExchange = 230;
    // OpAtomicExchange 229, IAdd 234, SMin 236, UMin 237, SMax 238, UMax 239, And 240, Or 241,
    // Xor 242. INC, DEC and F32 add go through the storage buffer CAS loops.
    const std::array<std::pair<IR::Opcode, u32>, 12> cases{{
        {IR::Opcode::GlobalAtomicIAdd32, 234},
        {IR::Opcode::GlobalAtomicSMin32, 236},
        {IR::Opcode::GlobalAtomicUMin32, 237},
        {IR::Opcode::GlobalAtomicSMax32, 238},
        {IR::Opcode::GlobalAtomicUMax32, 239},
        {IR::Opcode::GlobalAtomicAnd32, 240},
        {IR::Opcode::GlobalAtomicOr32, 241},
        {IR::Opcode::GlobalAtomicXor32, 242},
        {IR::Opcode::GlobalAtomicExchange32, 229},
        {IR::Opcode::GlobalAtomicInc32, OpAtomicCompareExchange},
        {IR::Opcode::GlobalAtomicDec32, OpAtomicCompareExchange},
        {IR::Opcode::GlobalAtomicAddF32, OpAtomicCompareExchange},
    }};
    for (const auto& [opcode, atomic] : cases) {
        CAPTURE(static_cast<u32>(opcode));
        std::vector<u32> code;
        REQUIRE_NOTHROW(code = EmitUntrackedAtomic(opcode));
        RequireSingleBlockTerminators(code);
        CHECK_FALSE(Instructions(code, atomic).empty());
        CHECK_FALSE(Instructions(code, OpFunctionCall).empty());
    }
}
