// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <map>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <spirv/unified1/spirv.hpp11>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/object_pool.h"

namespace {
namespace IR = Shader::IR;

static_assert(static_cast<u32>(spv::Op::OpVariable) == 59);
static_assert(static_cast<u32>(spv::StorageClass::PushConstant) == 9);
static_assert(Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET == 48);

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, spv::Op opcode) {
    REQUIRE(code.size() >= 5);
    REQUIRE(code[0] == spv::MagicNumber);
    std::vector<std::vector<u32>> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        REQUIRE(offset + size <= code.size());
        if ((code[offset] & 0xffffU) == static_cast<u32>(opcode)) {
            const auto words = std::span{code}.subspan(offset + 1, size - 1);
            result.emplace_back(words.begin(), words.end());
        }
        offset += size;
    }
    return result;
}

struct Fixture {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Shader::Profile profile{};

    explicit Fixture(Shader::Stage stage = Shader::Stage::Compute) {
        program.stage = stage;
        if (stage == Shader::Stage::Compute) {
            program.workgroup_size = {1, 1, 1};
        }
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        profile.supported_spirv = 0x00010400;
        profile.unified_descriptor_binding = true;
    }

    std::vector<u32> Emit() {
        return Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    }
};

std::vector<u32> PushConstantVariables(const std::vector<u32>& code) {
    std::vector<u32> vars;
    for (const auto& words : Instructions(code, spv::Op::OpVariable)) {
        REQUIRE(words.size() >= 3);
        if (words[2] == static_cast<u32>(spv::StorageClass::PushConstant)) {
            vars.push_back(words[1]);
        }
    }
    return vars;
}

u32 PushConstantStructType(const std::vector<u32>& code, u32 var_id) {
    u32 pointer_type = 0;
    for (const auto& words : Instructions(code, spv::Op::OpVariable)) {
        if (words.size() >= 3 && words[1] == var_id) {
            pointer_type = words[0];
            break;
        }
    }
    REQUIRE(pointer_type != 0);
    for (const auto& words : Instructions(code, spv::Op::OpTypePointer)) {
        REQUIRE(words.size() == 3);
        if (words[0] == pointer_type) {
            REQUIRE(words[1] == static_cast<u32>(spv::StorageClass::PushConstant));
            return words[2];
        }
    }
    FAIL("Missing OpTypePointer for PushConstant OpVariable");
    return 0;
}

std::map<u32, u32> MemberOffsets(const std::vector<u32>& code, u32 struct_id) {
    std::map<u32, u32> offsets;
    for (const auto& words : Instructions(code, spv::Op::OpMemberDecorate)) {
        if (words.size() == 4 && words[0] == struct_id &&
            words[2] == static_cast<u32>(spv::Decoration::Offset)) {
            const auto [it, inserted] = offsets.emplace(words[1], words[3]);
            REQUIRE(inserted);
        }
    }
    return offsets;
}

} // Anonymous namespace

TEST_CASE("SPIR-V rescaling and render area share one PushConstant block without offset collisions",
          "[video_core][spirv][push_constants]") {
    for (const Shader::Stage stage : {Shader::Stage::VertexB, Shader::Stage::Fragment}) {
        for (const u32 spirv_version : {0x00010300U, 0x00010400U}) {
            CAPTURE(stage, spirv_version);
            Fixture fixture{stage};
            fixture.profile.supported_spirv = spirv_version;
            fixture.program.info.uses_rescaling_uniform = true;
            fixture.program.info.uses_render_area = true;

            IR::IREmitter ir{fixture.block};
            const IR::U1 tex_scaled{ir.IsTextureScaled(ir.Imm32(0U))};
            const IR::U1 img_scaled{ir.IsImageScaled(ir.Imm32(0U))};
            const IR::F32 down_factor{ir.ResolutionDownFactor()};
            const IR::F32 area_w{ir.RenderAreaWidth()};
            const IR::F32 area_h{ir.RenderAreaHeight()};
            const IR::F32 selected{
                ir.Select(ir.LogicalOr(tex_scaled, img_scaled), area_h, ir.Imm32(0.0f))};
            const IR::F32 combined{ir.FPAdd(down_factor, ir.FPAdd(area_w, selected))};
            if (stage == Shader::Stage::Fragment) {
                fixture.program.info.stores_frag_color[0] = true;
                fixture.program.info.frag_color_components[0] = 1;
                ir.SetFragColor(0, 0, combined);
            }

            const std::vector<u32> code = fixture.Emit();
            const std::vector<u32> push_vars = PushConstantVariables(code);
            REQUIRE(push_vars.size() == 1);

            const u32 struct_id = PushConstantStructType(code, push_vars[0]);
            const std::map<u32, u32> offsets = MemberOffsets(code, struct_id);
            const std::map<u32, u32> expected{{0U, 0U}, {1U, 16U}, {2U, 24U}, {3U, 48U}};
            REQUIRE(offsets == expected);

            std::set<u32> unique_offsets;
            for (const auto& [member, byte_offset] : offsets) {
                unique_offsets.insert(byte_offset);
            }
            REQUIRE(unique_offsets.size() == 4);

            size_t push_access_chains = 0;
            for (const auto& words : Instructions(code, spv::Op::OpAccessChain)) {
                REQUIRE(words.size() >= 4);
                if (words[2] == push_vars[0]) {
                    ++push_access_chains;
                }
            }
            CHECK(push_access_chains >= 4);
        }
    }

    // Render area alone also sits at RENDERAREA_LAYOUT_OFFSET (48), never at 0.
    Fixture render_area_only{Shader::Stage::Fragment};
    render_area_only.program.info.uses_render_area = true;
    IR::IREmitter ir{render_area_only.block};
    std::ignore = ir.RenderAreaWidth();
    const std::vector<u32> code = render_area_only.Emit();
    const std::vector<u32> push_vars = PushConstantVariables(code);
    REQUIRE(push_vars.size() == 1);
    const u32 struct_id = PushConstantStructType(code, push_vars[0]);
    REQUIRE(MemberOffsets(code, struct_id) == std::map<u32, u32>{{0U, 48U}});
}

TEST_CASE("SPIR-V FPRecip64 emits a 64-bit 1.0 OpConstant rather than a 32-bit float literal",
          "[video_core][spirv][fp64]") {
    Fixture fixture{Shader::Stage::Compute};
    fixture.program.info.uses_fp64 = true;
    fixture.program.info.uses_local_invocation_id = true;

    IR::IREmitter ir{fixture.block};
    const IR::F64 input{ir.PackDouble2x32(
        ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.LocalInvocationIdY()))};
    const IR::F64 recip{ir.FPRecip(input)};
    REQUIRE(recip.Inst()->GetOpcode() == IR::Opcode::FPRecip64);

    const std::vector<u32> code = fixture.Emit();

    u32 f64_type = 0;
    for (const auto& words : Instructions(code, spv::Op::OpTypeFloat)) {
        REQUIRE(words.size() == 2);
        if (words[1] == 64) {
            f64_type = words[0];
        }
    }
    REQUIRE(f64_type != 0);

    u32 f64_one_id = 0;
    for (const auto& words : Instructions(code, spv::Op::OpConstant)) {
        REQUIRE(words.size() >= 3);
        if (words[0] == f64_type) {
            // Every 64-bit OpConstant must carry 2 literal words (low, high), never 1 32-bit word.
            REQUIRE(words.size() == 4);
            CHECK_FALSE(words[2] == 0x3f800000U);
            const u64 bits = (static_cast<u64>(words[3]) << 32) | words[2];
            if (bits == 0x3ff0000000000000ULL) {
                CHECK(words[2] == 0x00000000U);
                CHECK(words[3] == 0x3ff00000U);
                f64_one_id = words[1];
            }
        }
    }
    REQUIRE(f64_one_id != 0);

    bool found_fdiv = false;
    for (const auto& words : Instructions(code, spv::Op::OpFDiv)) {
        REQUIRE(words.size() == 4);
        if (words[0] == f64_type && words[2] == f64_one_id) {
            found_fdiv = true;
        }
    }
    REQUIRE(found_fdiv);
}

TEST_CASE("SPIR-V ImageAtomicU32 accepts an empty index Value for single-image descriptors",
          "[video_core][spirv][image_atomic]") {
    for (const bool is_buffer : {false, true}) {
        CAPTURE(is_buffer);
        Fixture fixture{Shader::Stage::Compute};
        fixture.program.info.uses_atomic_image_u32 = true;
        fixture.program.info.uses_local_invocation_id = true;
        if (is_buffer) {
            fixture.program.info.uses_image_buffers = true;
            fixture.program.info.image_buffer_descriptors.push_back({
                .format = Shader::ImageFormat::R32_UINT,
                .is_written = true,
                .is_read = true,
                .is_integer = true,
                .count = 1,
            });
        } else {
            fixture.program.info.image_descriptors.push_back({
                .type = Shader::TextureType::Color2D,
                .format = Shader::ImageFormat::R32_UINT,
                .is_written = true,
                .is_read = true,
                .is_integer = true,
                .count = 1,
            });
        }

        IR::IREmitter ir{fixture.block};
        const IR::Value coords =
            is_buffer ? IR::Value{ir.LocalInvocationIdX()}
                      : ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.LocalInvocationIdY());
        IR::TextureInstInfo info{};
        info.descriptor_index.Assign(0);
        info.type.Assign(is_buffer ? Shader::TextureType::Buffer : Shader::TextureType::Color2D);
        info.image_format.Assign(Shader::ImageFormat::R32_UINT);

        const IR::Value empty_index{};
        REQUIRE(empty_index.IsEmpty());
        fixture.block.PrependNewInst(fixture.block.end(), IR::Opcode::ImageAtomicIAdd32,
                                     {empty_index, coords, IR::Value{1U}}, info.raw);
        fixture.block.PrependNewInst(fixture.block.end(), IR::Opcode::ImageAtomicExchange32,
                                     {empty_index, coords, IR::Value{7U}}, info.raw);

        const std::vector<u32> code = fixture.Emit();
        CHECK(Instructions(code, spv::Op::OpImageTexelPointer).size() == 2);
        CHECK(Instructions(code, spv::Op::OpAtomicIAdd).size() == 1);
        CHECK(Instructions(code, spv::Op::OpAtomicExchange).size() == 1);
    }
}
