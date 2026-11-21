// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <bit>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

oaknut::Label EmitA64Cond(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Cond cond) {
    oaknut::Label pass;
    if (ctx.conf.enable_cycle_counting || !ReuseHostFlagsOn() || ctx.ebi.nzcv_live_pos == nullptr ||
        code.xptr<const void*>() != ctx.ebi.nzcv_live_pos) {
        code.LDR(Wscratch0, Xstate, offsetof(A64JitState, cpsr_nzcv));
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);
    }
    ctx.ebi.nzcv_live_pos = nullptr;
    code.B(static_cast<oaknut::Cond>(cond), pass);
    return pass;
}

void EmitA64LeafTerminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step);
void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step);

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::ReturnToDispatch, IR::LocationDescriptor, bool) {
    EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
}

static bool IsForwardLink(const EmitContext& ctx, const IR::LocationDescriptor& next, const IR::LocationDescriptor& initial_location) {
    return ctx.conf.halt_check_on_back_edges && !ctx.conf.enable_cycle_counting &&
           A64::LocationDescriptor{next}.PC() > A64::LocationDescriptor{initial_location}.PC();
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    oaknut::Label fail;

    if (ctx.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
        if (IsForwardLink(ctx, terminal.next, initial_location)) {
            EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
        } else if (ctx.conf.enable_cycle_counting) {
            code.CMP(Xticks, 0);
            code.B(LE, fail);
            EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
        } else {
            code.LDR(Wscratch0, Xhalt);
            code.CBNZ(Wscratch0, fail);
            EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
        }
    }

    code.l(fail);
    code.MOV(Xscratch0, A64::LocationDescriptor{terminal.next}.PC());
    code.STR(Xscratch0, Xstate, offsetof(A64JitState, pc));
    EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    oaknut::Label fail;

    if (ctx.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
        if (ctx.conf.halt_check_on_back_edges && !ctx.conf.enable_cycle_counting && !IsForwardLink(ctx, terminal.next, initial_location)) {
            code.LDR(Wscratch0, Xhalt);
            code.CBNZ(Wscratch0, fail);
        }
        EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
    }

    code.l(fail);
    code.MOV(Xscratch0, A64::LocationDescriptor{terminal.next}.PC());
    code.STR(Xscratch0, Xstate, offsetof(A64JitState, pc));
    EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::PopRSBHint, IR::LocationDescriptor, bool is_single_step) {
    if (ctx.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) && !is_single_step) {
        EmitRelocation(code, ctx, LinkTarget::PopRSBHint);
        return;
    }

    EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::FastDispatchHint, IR::LocationDescriptor, bool is_single_step) {
    if (ctx.conf.HasOptimization(OptimizationFlag::FastDispatch) && !is_single_step) {
        EmitRelocation(code, ctx, LinkTarget::FastDispatchHint);
        return;
    }

    EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    switch (terminal.if_) {
    case IR::Cond::AL:
    case IR::Cond::NV:
        EmitA64LeafTerminal(code, ctx, terminal.then_, initial_location, is_single_step);
        return;
    default:
        break;
    }
    oaknut::Label pass = EmitA64Cond(code, ctx, terminal.if_);
    EmitA64LeafTerminal(code, ctx, terminal.else_, initial_location, is_single_step);
    code.l(pass);
    EmitA64LeafTerminal(code, ctx, terminal.then_, initial_location, is_single_step);
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    oaknut::Label fail;
    switch (ctx.ebi.check_bit_kind) {
    case CheckBitKind::ConstTrue:
        ctx.ebi.check_bit_kind = CheckBitKind::Stack;
        EmitA64LeafTerminal(code, ctx, terminal.then_, initial_location, is_single_step);
        return;
    case CheckBitKind::ConstFalse:
        ctx.ebi.check_bit_kind = CheckBitKind::Stack;
        EmitA64LeafTerminal(code, ctx, terminal.else_, initial_location, is_single_step);
        return;
    case CheckBitKind::RegZero32:
        ctx.ebi.check_bit_kind = CheckBitKind::Stack;
        code.CBNZ(Wscratch2, fail);
        break;
    case CheckBitKind::RegZero64:
        ctx.ebi.check_bit_kind = CheckBitKind::Stack;
        code.CBNZ(Xscratch2, fail);
        break;
    case CheckBitKind::TestBit:
        ctx.ebi.check_bit_kind = CheckBitKind::Stack;
        code.TBZ(Xscratch2, ctx.ebi.check_bit_num, fail);
        break;
    case CheckBitKind::RegBit:
        ctx.ebi.check_bit_kind = CheckBitKind::Stack;
        code.CBZ(Wscratch2, fail);
        break;
    case CheckBitKind::Stack:
        code.LDRB(Wscratch0, SP, offsetof(StackLayout, check_bit));
        code.CBZ(Wscratch0, fail);
        break;
    }
    EmitA64LeafTerminal(code, ctx, terminal.then_, initial_location, is_single_step);
    code.l(fail);
    EmitA64LeafTerminal(code, ctx, terminal.else_, initial_location, is_single_step);
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    oaknut::Label fail;
    code.LDR(Wscratch0, Xhalt);
    code.CBNZ(Wscratch0, fail);
    EmitA64LeafTerminal(code, ctx, terminal.else_, initial_location, is_single_step);
    code.l(fail);
    EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
}

void EmitA64LeafTerminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (auto const x = std::get_if<IR::Term::ReturnToDispatch>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::LinkBlock>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::LinkBlockFast>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::PopRSBHint>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::FastDispatchHint>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    UNREACHABLE();
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (auto const x = std::get_if<IR::Term::LeafTerminal>(&terminal))
        return EmitA64LeafTerminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::If>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::CheckBit>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    if (auto const x = std::get_if<IR::Term::CheckHalt>(&terminal))
        return EmitA64Terminal(code, ctx, *x, initial_location, is_single_step);
    UNREACHABLE();
}

void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx) {
    const A64::LocationDescriptor location{ctx.block.Location()};
    EmitA64Terminal(code, ctx, ctx.block.GetTerminal(), location.SetSingleStepping(false), location.SingleStepping());
}

void EmitA64ConditionFailedTerminal(oaknut::CodeGenerator& code, EmitContext& ctx) {
    const A64::LocationDescriptor location{ctx.block.Location()};
    EmitA64Terminal(code, ctx, IR::Term::LinkBlock{ctx.block.ConditionFailedLocation()}, location.SetSingleStepping(false), location.SingleStepping());
}

void EmitA64CheckMemoryAbort(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, oaknut::Label& end) {
    if (!ctx.conf.check_halt_on_memory_access) {
        return;
    }

    const A64::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};

    code.LDAR(Xscratch0, Xhalt);
    code.TST(Xscratch0, static_cast<u32>(HaltReason::MemoryAbort));
    code.B(EQ, end);
    code.MOV(Xscratch0, current_location.PC());
    code.STR(Xscratch0, Xstate, offsetof(A64JitState, pc));
    EmitRelocation(code, ctx, LinkTarget::ReturnFromRunCode);
}

template<>
void EmitIR<IR::Opcode::A64SetCheckBit>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();

    if (inst == ctx.ebi.terminal_set_check_bit) {
        if (args[0].IsImmediate()) {
            ctx.ebi.check_bit_kind = args[0].GetImmediateU1() ? CheckBitKind::ConstTrue : CheckBitKind::ConstFalse;
        } else if (ctx.ebi.fused_check_bit_src != nullptr) {
            switch (ctx.ebi.fused_check_bit_src->GetOpcode()) {
            case IR::Opcode::IsZero32: {
                auto Wop = ctx.reg_alloc.ReadW(args[0]);
                RegAlloc::Realize(Wop);
                code.MOV(Wscratch2, Wop);
                ctx.ebi.check_bit_kind = CheckBitKind::RegZero32;
                break;
            }
            case IR::Opcode::IsZero64: {
                auto Xop = ctx.reg_alloc.ReadX(args[0]);
                RegAlloc::Realize(Xop);
                code.MOV(Xscratch2, Xop);
                ctx.ebi.check_bit_kind = CheckBitKind::RegZero64;
                break;
            }
            case IR::Opcode::TestBit: {
                auto Xop = ctx.reg_alloc.ReadX(args[0]);
                RegAlloc::Realize(Xop);
                code.MOV(Xscratch2, Xop);
                ctx.ebi.check_bit_kind = CheckBitKind::TestBit;
                ctx.ebi.check_bit_num = ctx.ebi.fused_check_bit_src->GetArg(1).GetU8();
                break;
            }
            default:
                UNREACHABLE();
            }
        } else {
            auto Wbit = ctx.reg_alloc.ReadW(args[0]);
            RegAlloc::Realize(Wbit);
            code.MOV(Wscratch2, Wbit);
            ctx.ebi.check_bit_kind = CheckBitKind::RegBit;
        }
        if (flags_live) {
            ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
        }
        return;
    }

    if (args[0].IsImmediate()) {
        if (args[0].GetImmediateU1()) {
            code.MOV(Wscratch0, 1);
            code.STRB(Wscratch0, SP, offsetof(StackLayout, check_bit));
        } else {
            code.STRB(WZR, SP, offsetof(StackLayout, check_bit));
        }
    } else {
        auto Wbit = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wbit);
        code.STRB(Wbit, SP, offsetof(StackLayout, check_bit));
    }
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64GetCFlag>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Wflag = ctx.reg_alloc.WriteW(inst);
    RegAlloc::Realize(Wflag);
    code.LDR(Wflag, Xstate, offsetof(A64JitState, cpsr_nzcv));
    code.AND(Wflag, Wflag, 1 << 29);
}

template<>
void EmitIR<IR::Opcode::A64GetNZCVRaw>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Wnzcv = ctx.reg_alloc.WriteW(inst);
    RegAlloc::Realize(Wnzcv);

    code.LDR(Wnzcv, Xstate, offsetof(A64JitState, cpsr_nzcv));
}

template<>
void EmitIR<IR::Opcode::A64SetNZCVRaw>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wnzcv);

    code.STR(Wnzcv, Xstate, offsetof(A64JitState, cpsr_nzcv));
    ctx.ebi.nzcv_live_pos = nullptr;
}

template<>
void EmitIR<IR::Opcode::A64SetNZCV>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool from_flags = !ctx.conf.enable_cycle_counting && ReuseHostFlagsOn() && !args[0].IsImmediate() &&
                            args[0].CurrentLocationKind(ctx.reg_alloc) == HostLoc::Kind::Flags;
    auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wnzcv);

    code.STR(Wnzcv, Xstate, offsetof(A64JitState, cpsr_nzcv));
    ctx.ebi.nzcv_live_pos = from_flags ? code.xptr<const void*>() : nullptr;
}

namespace {

template<std::size_t bitsize>
void EmitA64GetGpr(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(bitsize == 32 || bitsize == 64);
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();

    auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    RegAlloc::Realize(Rresult);

    code.LDR(Rresult, Xstate, offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<std::size_t bitsize>
void EmitA64SetGpr(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(bitsize == 32 || bitsize == 64);
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const std::size_t reg_off = offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    if (args[1].IsImmediate()) {
        const u64 imm = bitsize == 32 ? u64{args[1].GetImmediateU32()} : args[1].GetImmediateU64();
        if (imm == 0) {
            code.STR(XZR, Xstate, reg_off);
        } else {
            code.MOV(Xscratch0, imm);
            code.STR(Xscratch0, Xstate, reg_off);
        }
    } else if constexpr (bitsize == 32) {
        if (args[1].IsInFpr(ctx.reg_alloc)) {
            auto Svalue = ctx.reg_alloc.ReadS(args[1]);
            RegAlloc::Realize(Svalue);
            code.STR(Svalue, Xstate, reg_off);
            code.STR(WZR, Xstate, reg_off + 4);
        } else {
            auto Wvalue = ctx.reg_alloc.ReadW(args[1]);
            RegAlloc::Realize(Wvalue);
            if (ElideRedundantZextOn() && IsZeroExtendedU32Producer(args[1].GetInst())) {
                code.STR(Wvalue->toX(), Xstate, reg_off);
            } else {
                code.MOV(Wscratch0, Wvalue);
                code.STR(Xscratch0, Xstate, reg_off);
            }
        }
    } else if (args[1].IsInFpr(ctx.reg_alloc)) {
        auto Dvalue = ctx.reg_alloc.ReadD(args[1]);
        RegAlloc::Realize(Dvalue);
        code.STR(Dvalue, Xstate, reg_off);
    } else {
        auto Xvalue = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xvalue);
        code.STR(Xvalue, Xstate, reg_off);
    }
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<std::size_t bitsize>
void EmitA64SetScalarVec(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(bitsize == 32 || bitsize == 64);
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const std::size_t reg_off = offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec);

    if (args[1].IsImmediate()) {
        const u64 imm = bitsize == 32 ? u64{args[1].GetImmediateU32()} : args[1].GetImmediateU64();
        if (imm == 0) {
            code.STR(XZR, Xstate, reg_off);
        } else {
            code.MOV(Xscratch0, imm);
            code.STR(Xscratch0, Xstate, reg_off);
        }
    } else if (args[1].IsInGpr(ctx.reg_alloc)) {
        auto Rvalue = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        RegAlloc::Realize(Rvalue);
        code.STR(Rvalue, Xstate, reg_off);
        if constexpr (bitsize == 32) {
            code.STR(WZR, Xstate, reg_off + 4);
        }
    } else {
        auto Vvalue = ctx.reg_alloc.ReadVec<bitsize>(args[1]);
        RegAlloc::Realize(Vvalue);
        code.STR(Vvalue, Xstate, reg_off);
        if constexpr (bitsize == 32) {
            code.STR(WZR, Xstate, reg_off + 4);
        }
    }
    code.STR(XZR, Xstate, reg_off + 8);
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

void EmitFlushCycles(oaknut::CodeGenerator& code, EmitContext& ctx) {
    if (!ctx.conf.enable_cycle_counting) {
        return;
    }
    code.LDR(X1, SP, offsetof(StackLayout, cycles_to_run));
    code.SUB(X1, X1, Xticks);
    EmitRelocation(code, ctx, LinkTarget::AddTicks);
}

void EmitReloadCycles(oaknut::CodeGenerator& code, EmitContext& ctx) {
    if (!ctx.conf.enable_cycle_counting) {
        return;
    }
    EmitRelocation(code, ctx, LinkTarget::GetTicksRemaining);
    code.STR(X0, SP, offsetof(StackLayout, cycles_to_run));
    code.MOV(Xticks, X0);
}

}  // namespace

template<>
void EmitIR<IR::Opcode::A64GetW>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitA64GetGpr<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A64GetX>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitA64GetGpr<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A64GetS>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    auto Sresult = ctx.reg_alloc.WriteS(inst);
    RegAlloc::Realize(Sresult);
    code.LDR(Sresult, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64GetD>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    auto Dresult = ctx.reg_alloc.WriteD(inst);
    RegAlloc::Realize(Dresult);
    code.LDR(Dresult, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64GetQ>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const std::size_t vec_base = offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec);
    if (const auto* scalar = ctx.ebi.FindScalarGetQ(inst)) {
        const std::size_t elem_off = vec_base + (static_cast<std::size_t>(scalar->elem_bits) / 8) * scalar->lane;
        switch (scalar->elem_bits) {
        case 8: {
            auto Bresult = ctx.reg_alloc.WriteB(inst);
            RegAlloc::Realize(Bresult);
            code.LDR(Bresult, Xstate, static_cast<int>(elem_off));
            break;
        }
        case 16: {
            auto Hresult = ctx.reg_alloc.WriteH(inst);
            RegAlloc::Realize(Hresult);
            code.LDR(Hresult, Xstate, static_cast<int>(elem_off));
            break;
        }
        case 32: {
            auto Sresult = ctx.reg_alloc.WriteS(inst);
            RegAlloc::Realize(Sresult);
            code.LDR(Sresult, Xstate, static_cast<int>(elem_off));
            break;
        }
        case 64: {
            auto Dresult = ctx.reg_alloc.WriteD(inst);
            RegAlloc::Realize(Dresult);
            code.LDR(Dresult, Xstate, static_cast<int>(elem_off));
            break;
        }
        default:
            UNREACHABLE();
        }
        if (flags_live) {
            ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
        }
        return;
    }
    auto Qresult = ctx.reg_alloc.WriteQ(inst);
    RegAlloc::Realize(Qresult);
    code.LDR(Qresult, Xstate, vec_base);
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64GetSP>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    auto Xresult = ctx.reg_alloc.WriteX(inst);
    RegAlloc::Realize(Xresult);

    code.LDR(Xresult, Xstate, offsetof(A64JitState, sp));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64GetFPCR>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    RegAlloc::Realize(Wresult);

    code.LDR(Wresult, Xstate, offsetof(A64JitState, fpcr));
}

template<>
void EmitIR<IR::Opcode::A64GetFPSR>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    RegAlloc::Realize(Wresult);

    ctx.fpsr.GetFpsr(Wresult);
}

template<>
void EmitIR<IR::Opcode::A64SetW>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitA64SetGpr<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A64SetX>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitA64SetGpr<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A64SetS>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitA64SetScalarVec<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A64SetD>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitA64SetScalarVec<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A64SetQ>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    auto Qvalue = ctx.reg_alloc.ReadQ(args[1]);
    RegAlloc::Realize(Qvalue);
    code.STR(Qvalue, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64SetSP>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Xvalue);
    code.STR(Xvalue, Xstate, offsetof(A64JitState, sp));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64SetFPCR>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wvalue);
    code.STR(Wvalue, Xstate, offsetof(A64JitState, fpcr));
    code.MSR(oaknut::SystemReg::FPCR, Wvalue->toX());
}

template<>
void EmitIR<IR::Opcode::A64SetFPSR>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wvalue);
    code.STR(Wvalue, Xstate, offsetof(A64JitState, fpsr));
    code.MSR(oaknut::SystemReg::FPSR, Wvalue->toX());
}

template<>
void EmitIR<IR::Opcode::A64SetPC>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Xvalue);
    code.STR(Xvalue, Xstate, offsetof(A64JitState, pc));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::A64CallSupervisor>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall();

    EmitFlushCycles(code, ctx);
    code.MOV(W1, args[0].GetImmediateU32());
    EmitRelocation(code, ctx, LinkTarget::CallSVC);
    code.STRB(WZR, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitReloadCycles(code, ctx);
}

template<>
void EmitIR<IR::Opcode::A64ExceptionRaised>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall();

    EmitFlushCycles(code, ctx);
    code.MOV(X1, args[0].GetImmediateU64());
    code.MOV(X2, args[1].GetImmediateU64());
    EmitRelocation(code, ctx, LinkTarget::ExceptionRaised);
    EmitReloadCycles(code, ctx);
}

template<>
void EmitIR<IR::Opcode::A64DataCacheOperationRaised>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    EmitRelocation(code, ctx, LinkTarget::DataCacheOperationRaised);
}

template<>
void EmitIR<IR::Opcode::A64InstructionCacheOperationRaised>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[0], args[1]);
    EmitRelocation(code, ctx, LinkTarget::InstructionCacheOperationRaised);
}

template<>
void EmitIR<IR::Opcode::A64DataSynchronizationBarrier>(oaknut::CodeGenerator& code, EmitContext&, IR::Inst*) {
    code.DSB(oaknut::BarrierOp::ISH);
}

template<>
void EmitIR<IR::Opcode::A64DataMemoryBarrier>(oaknut::CodeGenerator& code, EmitContext&, IR::Inst* inst) {
    switch (inst->GetArg(0).GetU8() & 0b11) {
    case 0b01:
        code.DMB(oaknut::BarrierOp::ISHLD);
        break;
    case 0b10:
        code.DMB(oaknut::BarrierOp::ISHST);
        break;
    default:
        code.DMB(oaknut::BarrierOp::ISH);
        break;
    }
}

template<>
void EmitIR<IR::Opcode::A64InstructionSynchronizationBarrier>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst*) {
    if (!ctx.conf.hook_isb) {
        return;
    }

    ctx.reg_alloc.PrepareForCall();
    EmitRelocation(code, ctx, LinkTarget::InstructionSynchronizationBarrierRaised);
}

template<>
void EmitIR<IR::Opcode::A64GetCNTFRQ>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Xvalue = ctx.reg_alloc.WriteX(inst);
    RegAlloc::Realize(Xvalue);
    code.MOV(Xvalue, ctx.conf.cntfreq_el0);
}

template<>
void EmitIR<IR::Opcode::A64GetCNTPCT>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.wall_clock_cntpct && ctx.conf.inline_cntpct) {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Xresult);

        code.ISB();
        code.MRS(*Xresult, oaknut::SystemReg::CNTVCT_EL0);
        if (ctx.conf.cntpct_scale_lo == 0) {
            if (ctx.conf.cntpct_scale_hi == 0) {
                code.MOV(*Xresult, XZR);
            } else if (ctx.conf.cntpct_scale_hi != 1) {
                code.MOV(Xscratch0, ctx.conf.cntpct_scale_hi);
                code.MUL(*Xresult, *Xresult, Xscratch0);
            }
        } else if (ctx.conf.cntpct_scale_hi == 0) {
            code.MOV(Xscratch0, ctx.conf.cntpct_scale_lo);
            code.UMULH(*Xresult, *Xresult, Xscratch0);
        } else if (ctx.conf.cntpct_scale_hi == 1) {
            code.MOV(Xscratch0, ctx.conf.cntpct_scale_lo);
            code.UMULH(Xscratch1, *Xresult, Xscratch0);
            code.ADD(*Xresult, Xscratch1, *Xresult);
        } else {
            code.MOV(Xscratch0, ctx.conf.cntpct_scale_lo);
            code.UMULH(Xscratch1, *Xresult, Xscratch0);
            code.MOV(Xscratch0, ctx.conf.cntpct_scale_hi);
            code.MADD(*Xresult, *Xresult, Xscratch0, Xscratch1);
        }
        return;
    }

    ctx.reg_alloc.PrepareForCall();
    if (!ctx.conf.wall_clock_cntpct) {
        EmitFlushCycles(code, ctx);
        EmitReloadCycles(code, ctx);
    }
    EmitRelocation(code, ctx, LinkTarget::GetCNTPCT);
    ctx.reg_alloc.DefineAsRegister(inst, X0);
}

template<>
void EmitIR<IR::Opcode::A64GetCTR>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Wvalue = ctx.reg_alloc.WriteW(inst);
    RegAlloc::Realize(Wvalue);
    code.MOV(Wvalue, ctx.conf.ctr_el0);
}

template<>
void EmitIR<IR::Opcode::A64GetDCZID>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Wvalue = ctx.reg_alloc.WriteW(inst);
    RegAlloc::Realize(Wvalue);
    code.MOV(Wvalue, ctx.conf.dczid_el0);
}

template<>
void EmitIR<IR::Opcode::A64GetTPIDR>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Xvalue = ctx.reg_alloc.WriteX(inst);
    RegAlloc::Realize(Xvalue);
    code.MOV(Xscratch0, std::bit_cast<u64>(ctx.conf.tpidr_el0));
    code.LDR(Xvalue, Xscratch0);
}

template<>
void EmitIR<IR::Opcode::A64GetTPIDRRO>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto Xvalue = ctx.reg_alloc.WriteX(inst);
    RegAlloc::Realize(Xvalue);
    code.MOV(Xscratch0, std::bit_cast<u64>(ctx.conf.tpidrro_el0));
    code.LDR(Xvalue, Xscratch0);
}

template<>
void EmitIR<IR::Opcode::A64SetTPIDR>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Xvalue);
    code.MOV(Xscratch0, std::bit_cast<u64>(ctx.conf.tpidr_el0));
    code.STR(Xvalue, Xscratch0);
}

}  // namespace Dynarmic::Backend::Arm64
