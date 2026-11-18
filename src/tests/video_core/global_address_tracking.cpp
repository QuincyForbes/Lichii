// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
class StubEnvironment final : public Shader::Environment {
public:
    StubEnvironment() { stage = Shader::Stage::Compute; }
    bool native{};
    bool IsNativeProgram() const override { return native; }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};
}

TEST_CASE("Global address tracking stops at a memory load", "[shader_recompiler][global_memory]") {
    using namespace Shader;
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    const auto low = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x110));
    const auto high = ir.GetCbuf(ir.Imm32(0), ir.Imm32(0x114));
    const auto source = ir.PackUint2x32(ir.CompositeConstruct(low, high));
    const auto loaded = ir.LoadGlobal64(source);
    const auto destination = ir.PackUint2x32(ir.CompositeConstruct(
        ir.CompositeExtract(loaded, 0), ir.CompositeExtract(loaded, 1)));
    ir.WriteGlobal32(destination, ir.Imm32(1));

    IR::Program program;
    program.post_order_blocks.push_back(&block);
    HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    StubEnvironment env;
    Optimization::GlobalMemoryToStorageBufferPass(env, program, host);

    const auto has = [&](IR::Opcode opcode) {
        return std::ranges::any_of(block.Instructions(), [=](const IR::Inst& inst) {
            return inst.GetOpcode() == opcode;
        });
    };
    REQUIRE(has(IR::Opcode::LoadStorage64));
    REQUIRE(has(IR::Opcode::WriteGlobal32));
    REQUIRE_FALSE(has(IR::Opcode::WriteStorage32));
}

TEST_CASE("A stage binds up to 64 storage buffers when the device allows them",
          "[shader_recompiler][global_memory]") {
    using namespace Shader;
    ObjectPool<IR::Inst> inst_pool;
    IR::Block block{inst_pool};
    IR::IREmitter ir{block};
    for (u32 slot = 0; slot < 40; ++slot) {
        const u32 offset{0x100 + slot * 8};
        const IR::U32 low{ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset))};
        const IR::U32 high{ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset + 4))};
        static_cast<void>(ir.LoadGlobal32(ir.PackUint2x32(ir.CompositeConstruct(low, high))));
    }

    IR::Program program;
    program.stage = Stage::Compute;
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    host.support_int64 = true;
    StubEnvironment env;
    env.native = true;
    Optimization::GlobalMemoryToStorageBufferPass(env, program, host);

    CHECK(program.info.storage_buffers_descriptors.size() == 40);
}
