// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "common/bit_field.h"
#include "common/common_types.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {
namespace {
void MOV(TranslatorVisitor& v, u64 insn, const IR::U32& src, bool is_mov32i = false) {
    union {
        u64 raw;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<39, 4, u64> mask;
        BitField<12, 4, u64> mov32i_mask;
    } const mov{insn};

    u64 mask = is_mov32i ? mov.mov32i_mask : mov.mask;
    if (mask == 0xf || mask == 0x1) {
        v.X(mov.dest_reg, src);
        return;
    }
    // EDEN_NX2_MASKED_MOV=0: drop masked moves as before (A/B switch).
    static const bool lane_masked = [] {
        const char* const value = std::getenv("EDEN_NX2_MASKED_MOV");
        return value == nullptr || value[0] != '0';
    }();
    if (mask == 0 || !lane_masked) {
        return;
    }
    // Quad lane mask: bit N enables the move for lane N of each quad (lane id & 3); the other
    // lanes keep the destination's previous value. Dropping the move entirely left the
    // destination stale in every lane, which broke shaders that build per-quad values this way
    // (derivative and lighting math).
    const IR::U32 quad_lane{v.ir.BitwiseAnd(v.ir.LaneId(), v.ir.Imm32(3))};
    const IR::U32 lane_bit{IR::U32{v.ir.ShiftLeftLogical(v.ir.Imm32(1), quad_lane)}};
    const IR::U1 enabled{v.ir.INotEqual(
        v.ir.BitwiseAnd(lane_bit, v.ir.Imm32(static_cast<u32>(mask))), v.ir.Imm32(0))};
    v.X(mov.dest_reg, IR::U32{v.ir.Select(enabled, src, v.X(mov.dest_reg))});
}
} // Anonymous namespace

void TranslatorVisitor::MOV_reg(u64 insn) {
    // Native register copy (lowered native programs only): a MOV with an empty write mask and
    // 0b1010 in bits 44-47, never emitted by Maxwell compilers. Bits 8-9 / 10-11 select the
    // destination / source bank of 256 registers.
    if (env.HasNativeInstructions() && ((insn >> 39) & 15) == 0 && ((insn >> 44) & 15) == 10) {
        const u32 destination_bank = (insn >> 8) & 3, source_bank = (insn >> 10) & 3;
        if (destination_bank > 2 || source_bank > 2)
            throw NotImplementedException("Invalid native register copy bank");
        const auto destination = static_cast<IR::Reg>((insn & 255) + destination_bank * 256);
        const auto source = static_cast<IR::Reg>(((insn >> 20) & 255) + source_bank * 256);
        X(destination, X(source));
        return;
    }
    MOV(*this, insn, GetReg20(insn));
}

void TranslatorVisitor::MOV_cbuf(u64 insn) {
    MOV(*this, insn, GetCbuf(insn));
}

void TranslatorVisitor::MOV_imm(u64 insn) {
    MOV(*this, insn, GetImm20(insn));
}

void TranslatorVisitor::MOV32I(u64 insn) {
    MOV(*this, insn, GetImm32(insn), true);
}

} // namespace Shader::Maxwell
