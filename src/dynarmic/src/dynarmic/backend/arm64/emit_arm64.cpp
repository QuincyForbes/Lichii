// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/emit_arm64.h"

#include <cstdlib>
#include <span>
#include <variant>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/backend/arm64/verbose_debugging_output.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

template<>
void EmitIR<IR::Opcode::Void>(oaknut::CodeGenerator&, EmitContext&, IR::Inst*) {}

template<>
void EmitIR<IR::Opcode::Identity>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

template<>
void EmitIR<IR::Opcode::Breakpoint>(oaknut::CodeGenerator& code, EmitContext&, IR::Inst*) {
    code.BRK(0);
}

template<>
void EmitIR<IR::Opcode::CallHostFunction>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    ctx.reg_alloc.PrepareForCall(args[1], args[2], args[3]);
    code.MOV(Xscratch0, args[0].GetImmediateU64());
    code.BLR(Xscratch0);
}

template<>
void EmitIR<IR::Opcode::PushRSB>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (!ctx.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        return;
    }

    const bool flags_live = ctx.ebi.nzcv_live_pos != nullptr && ctx.ebi.nzcv_live_pos == code.xptr<const void*>();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const IR::LocationDescriptor target{args[0].GetImmediateU64()};

    if (ctx.conf.use_state_rsb) {
        code.LDR(Wscratch2, Xstate, ctx.conf.state_rsb_ptr_offset);
        code.ADD(Wscratch2, Wscratch2, sizeof(RSBEntry));
        code.AND(Wscratch2, Wscratch2, A64RSBIndexMask);
        code.STR(Wscratch2, Xstate, ctx.conf.state_rsb_ptr_offset);
        code.ADD(Xscratch2, Xstate, Xscratch2);

        code.MOV(Xscratch0, target.Value());
        EmitBlockLinkRelocation(code, ctx, target, BlockRelocationType::MoveToScratch1);
        code.STP(Xscratch0, Xscratch1, Xscratch2, 0);
        if (flags_live) {
            ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
        }
        return;
    }

    code.LDR(Wscratch2, SP, offsetof(StackLayout, rsb_ptr));
    code.ADD(Wscratch2, Wscratch2, sizeof(RSBEntry));
    code.AND(Wscratch2, Wscratch2, RSBIndexMask);
    code.STR(Wscratch2, SP, offsetof(StackLayout, rsb_ptr));
    code.ADD(Xscratch2, SP, Xscratch2);

    code.MOV(Xscratch0, target.Value());
    EmitBlockLinkRelocation(code, ctx, target, BlockRelocationType::MoveToScratch1);
    code.STP(Xscratch0, Xscratch1, Xscratch2, offsetof(StackLayout, rsb));
    if (flags_live) {
        ctx.ebi.nzcv_live_pos = code.xptr<const void*>();
    }
}

template<>
void EmitIR<IR::Opcode::GetCarryFromOp>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(ctx.reg_alloc.WasValueDefined(inst));
}

template<>
void EmitIR<IR::Opcode::GetOverflowFromOp>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(ctx.reg_alloc.WasValueDefined(inst));
}

template<>
void EmitIR<IR::Opcode::GetGEFromOp>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(ctx.reg_alloc.WasValueDefined(inst));
}

template<>
void EmitIR<IR::Opcode::GetNZCVFromOp>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (ctx.reg_alloc.WasValueDefined(inst)) {
        return;
    }

    switch (args[0].GetType()) {
    case IR::Type::U32: {
        auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
        auto flags = ctx.reg_alloc.WriteFlags(inst);
        RegAlloc::Realize(Wvalue, flags);

        code.TST(*Wvalue, Wvalue);
        break;
    }
    case IR::Type::U64: {
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        auto flags = ctx.reg_alloc.WriteFlags(inst);
        RegAlloc::Realize(Xvalue, flags);

        code.TST(*Xvalue, Xvalue);
        break;
    }
    default:
        UNREACHABLE();
    }
}

template<>
void EmitIR<IR::Opcode::GetNZFromOp>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (ctx.reg_alloc.WasValueDefined(inst)) {
        return;
    }

    switch (args[0].GetType()) {
    case IR::Type::U32: {
        auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
        auto flags = ctx.reg_alloc.WriteFlags(inst);
        RegAlloc::Realize(Wvalue, flags);

        code.TST(*Wvalue, *Wvalue);
        break;
    }
    case IR::Type::U64: {
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        auto flags = ctx.reg_alloc.WriteFlags(inst);
        RegAlloc::Realize(Xvalue, flags);

        code.TST(*Xvalue, *Xvalue);
        break;
    }
    default:
        UNREACHABLE();
    }
}

template<>
void EmitIR<IR::Opcode::GetUpperFromOp>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(ctx.reg_alloc.WasValueDefined(inst));
}

template<>
void EmitIR<IR::Opcode::GetLowerFromOp>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(ctx.reg_alloc.WasValueDefined(inst));
}

template<>
void EmitIR<IR::Opcode::GetCFlagFromNZCV>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wc = ctx.reg_alloc.WriteW(inst);
    auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wc, Wnzcv);

    code.AND(Wc, Wnzcv, 1 << 29);
}

template<>
void EmitIR<IR::Opcode::NZCVFromPackedFlags>(oaknut::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

static void EmitAddCycles(oaknut::CodeGenerator& code, EmitContext& ctx, std::size_t cycles_to_add) {
    if (!ctx.conf.enable_cycle_counting) {
        return;
    }
    if (cycles_to_add == 0) {
        return;
    }

    if (oaknut::AddSubImm::is_valid(cycles_to_add)) {
        code.SUB(Xticks, Xticks, cycles_to_add);
    } else {
        code.MOV(Xscratch1, cycles_to_add);
        code.SUB(Xticks, Xticks, Xscratch1);
    }
}

#ifdef DYNARMIC_EMIT_STATS
// Offline code size study only (built with -DDYNARMIC_EMIT_STATS): host bytes per IR opcode,
// inline (deferred = false) and in the block's out-of-line tail (deferred = true); opcode ~0u is
// the condition check, ~1u the terminal.
void (*g_emit_stats)(unsigned opcode, bool deferred, std::size_t bytes) = nullptr;
#endif

bool ReuseHostFlagsOn() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_FLAGS");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

bool ElideRedundantZextOn() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_ZEXT");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

bool FoldAddressOffsetsOn() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_FOLD_OFFSETS");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}

bool FprScalarForwardingOn() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_FPR_SCALAR");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

bool IsZeroExtendedU32Producer(const IR::Inst* inst) noexcept {
    switch (inst->GetOpcode()) {
    case IR::Opcode::Identity:
    case IR::Opcode::ZeroExtendByteToWord:
    case IR::Opcode::ZeroExtendHalfToWord:
        return !inst->GetArg(0).IsImmediate() && IsZeroExtendedU32Producer(inst->GetArg(0).GetInst());
    case IR::Opcode::A64GetW:
    case IR::Opcode::A64GetFPCR:
    case IR::Opcode::A64GetFPSR:
    case IR::Opcode::A64GetCTR:
    case IR::Opcode::A64GetDCZID:
    case IR::Opcode::A64GetNZCVRaw:
    case IR::Opcode::A64GetCFlag:
    case IR::Opcode::A64ReadMemory8:
    case IR::Opcode::A64ReadMemory16:
    case IR::Opcode::A64ReadMemory32:
    case IR::Opcode::LeastSignificantWord:
    case IR::Opcode::LeastSignificantHalf:
    case IR::Opcode::LeastSignificantByte:
    case IR::Opcode::MostSignificantWord:
    case IR::Opcode::MostSignificantBit:
    case IR::Opcode::IsZero32:
    case IR::Opcode::IsZero64:
    case IR::Opcode::Add32:
    case IR::Opcode::Sub32:
    case IR::Opcode::And32:
    case IR::Opcode::AndNot32:
    case IR::Opcode::Eor32:
    case IR::Opcode::Or32:
    case IR::Opcode::Not32:
    case IR::Opcode::Mul32:
    case IR::Opcode::UnsignedDiv32:
    case IR::Opcode::SignedDiv32:
    case IR::Opcode::LogicalShiftLeft32:
    case IR::Opcode::LogicalShiftRight32:
    case IR::Opcode::ArithmeticShiftRight32:
    case IR::Opcode::BitRotateRight32:
    case IR::Opcode::RotateRightExtended:
    case IR::Opcode::LogicalShiftLeftMasked32:
    case IR::Opcode::LogicalShiftRightMasked32:
    case IR::Opcode::ArithmeticShiftRightMasked32:
    case IR::Opcode::RotateRightMasked32:
    case IR::Opcode::SignExtendByteToWord:
    case IR::Opcode::SignExtendHalfToWord:
    case IR::Opcode::ByteReverseWord:
    case IR::Opcode::ByteReverseHalf:
    case IR::Opcode::CountLeadingZeros32:
    case IR::Opcode::ExtractRegister32:
    case IR::Opcode::ReplicateBit32:
    case IR::Opcode::MaxSigned32:
    case IR::Opcode::MinSigned32:
    case IR::Opcode::MaxUnsigned32:
    case IR::Opcode::MinUnsigned32:
    case IR::Opcode::ConditionalSelect32:
    case IR::Opcode::ConditionalSelectNZCV:
    case IR::Opcode::VectorGetElement8:
    case IR::Opcode::VectorGetElement16:
    case IR::Opcode::VectorGetElement32:
        return true;
    default:
        return false;
    }
}

bool CanEncodeMemDisp(std::size_t bitsize, s64 disp) noexcept {
    const s64 byte_size = static_cast<s64>(bitsize / 8);
    if (disp >= 0 && (disp & (byte_size - 1)) == 0 && (disp / byte_size) <= 4095) {
        return true;
    }
    return disp >= -256 && disp <= 255;
}

namespace {

bool IsOrderedAcc(IR::AccType acctype) noexcept {
    return acctype == IR::AccType::ORDERED || acctype == IR::AccType::ORDEREDRW || acctype == IR::AccType::LIMITEDORDERED;
}

bool MatchFoldableFastmemAccess(const IR::Inst& inst, const EmitConfig& conf, FastmemManager& fastmem_manager,
                                IR::LocationDescriptor block_loc, std::size_t& out_bitsize) {
    int acc_arg = -1;
    switch (inst.GetOpcode()) {
    case IR::Opcode::A64ReadMemory8:
        out_bitsize = 8;
        acc_arg = 2;
        break;
    case IR::Opcode::A64ReadMemory16:
        out_bitsize = 16;
        acc_arg = 2;
        break;
    case IR::Opcode::A64ReadMemory32:
        out_bitsize = 32;
        acc_arg = 2;
        break;
    case IR::Opcode::A64ReadMemory64:
        out_bitsize = 64;
        acc_arg = 2;
        break;
    case IR::Opcode::A64ReadMemory128:
        out_bitsize = 128;
        acc_arg = 2;
        break;
    case IR::Opcode::A64WriteMemory8:
        out_bitsize = 8;
        acc_arg = 3;
        break;
    case IR::Opcode::A64WriteMemory16:
        out_bitsize = 16;
        acc_arg = 3;
        break;
    case IR::Opcode::A64WriteMemory32:
        out_bitsize = 32;
        acc_arg = 3;
        break;
    case IR::Opcode::A64WriteMemory64:
        out_bitsize = 64;
        acc_arg = 3;
        break;
    case IR::Opcode::A64WriteMemory128:
        out_bitsize = 128;
        acc_arg = 3;
        break;
    default:
        return false;
    }
    if (!conf.fastmem_pointer || !fastmem_manager.SupportsFastmem()) {
        return false;
    }
    if (inst.GetArg(1).IsImmediate() || IsOrderedAcc(inst.GetArg(acc_arg).GetAccType())) {
        return false;
    }
    return fastmem_manager.ShouldFastmem(std::make_tuple(block_loc, inst.GetName()));
}

bool MatchConstAddrOffset(const IR::Inst* inst, const IR::Inst*& out_base, s64& out_disp) noexcept {
    if (inst->HasAssociatedPseudoOperation()) {
        return false;
    }
    const auto op = inst->GetOpcode();
    if (op != IR::Opcode::Add64 && op != IR::Opcode::Sub64) {
        return false;
    }
    const IR::Value base = inst->GetArg(0);
    const IR::Value imm = inst->GetArg(1);
    const IR::Value carry = inst->GetArg(2);
    if (base.IsImmediate() || !imm.IsImmediate() || !carry.IsImmediate()) {
        return false;
    }
    if (op == IR::Opcode::Add64) {
        if (carry.GetU1()) {
            return false;
        }
        out_disp = static_cast<s64>(imm.GetImmediateAsU64());
    } else {
        if (!carry.GetU1()) {
            return false;
        }
        out_disp = -static_cast<s64>(imm.GetImmediateAsU64());
    }
    out_base = base.GetInst();
    return true;
}

void PreAnalyzeBlock(IR::Block& block, const EmitConfig& conf, FastmemManager& fastmem_manager, EmittedBlockInfo& ebi) {
    if (!conf.very_verbose_debugging_output && ReuseHostFlagsOn() &&
        std::holds_alternative<IR::Term::CheckBit>(block.GetTerminal())) {
        const IR::Inst* last_non_void = nullptr;
        for (auto it = block.instructions.rbegin(); it != block.instructions.rend(); ++it) {
            if (it->GetOpcode() != IR::Opcode::Void) {
                last_non_void = &*it;
                break;
            }
        }
        if (last_non_void != nullptr && last_non_void->GetOpcode() == IR::Opcode::A64SetCheckBit) {
            ebi.terminal_set_check_bit = last_non_void;
            const IR::Value arg = last_non_void->GetArg(0);
            if (!arg.IsImmediate()) {
                const IR::Inst* const prod = arg.GetInst();
                const auto pop = prod->GetOpcode();
                if (prod->UseCount() == 1 && !prod->GetArg(0).IsImmediate() &&
                    (pop == IR::Opcode::IsZero32 || pop == IR::Opcode::IsZero64 || pop == IR::Opcode::TestBit)) {
                    ebi.fused_check_bit_src = prod;
                }
            }
        }
    }

    if (FprScalarForwardingOn()) {
        for (const auto& inst : block.instructions) {
            u8 elem_bits = 0;
            switch (inst.GetOpcode()) {
            case IR::Opcode::VectorGetElement8:
                elem_bits = 8;
                break;
            case IR::Opcode::VectorGetElement16:
                elem_bits = 16;
                break;
            case IR::Opcode::VectorGetElement32:
                elem_bits = 32;
                break;
            case IR::Opcode::VectorGetElement64:
                elem_bits = 64;
                break;
            default:
                continue;
            }
            if (!inst.GetArg(0).IsImmediate() && inst.GetArg(1).IsImmediate()) {
                const IR::Inst* const src = inst.GetArg(0).GetInst();
                if (src->GetOpcode() == IR::Opcode::A64GetQ && src->UseCount() == 1) {
                    ebi.scalar_get_q.push_back(EmittedBlockInfo::ScalarGetQInfo{src, elem_bits, inst.GetArg(1).GetU8()});
                }
            }
        }
    }

    if (FoldAddressOffsetsOn() && conf.fastmem_pointer && conf.fastmem_address_space_bits == 64 &&
        fastmem_manager.SupportsFastmem()) {
        struct CandidateUseCount {
            const IR::Inst* off_inst = nullptr;
            const IR::Inst* base_inst = nullptr;
            s64 disp = 0;
            std::size_t compatible_uses = 0;
        };
        boost::container::small_vector<CandidateUseCount, 16> candidates;

        for (const auto& inst : block.instructions) {
            std::size_t bitsize = 0;
            if (!MatchFoldableFastmemAccess(inst, conf, fastmem_manager, block.Location(), bitsize)) {
                continue;
            }
            const IR::Inst* const vaddr_inst = inst.GetArg(1).GetInst();
            const IR::Inst* base_inst = nullptr;
            s64 disp = 0;
            if (!MatchConstAddrOffset(vaddr_inst, base_inst, disp) || !CanEncodeMemDisp(bitsize, disp)) {
                continue;
            }
            auto it = std::find_if(candidates.begin(), candidates.end(),
                                   [vaddr_inst](const auto& c) { return c.off_inst == vaddr_inst; });
            if (it == candidates.end()) {
                candidates.push_back(CandidateUseCount{vaddr_inst, base_inst, disp, 1});
            } else {
                ++it->compatible_uses;
            }
        }

        for (const auto& inst : block.instructions) {
            const auto it = std::find_if(candidates.begin(), candidates.end(),
                                         [&inst](const auto& c) { return c.off_inst == &inst; });
            if (it != candidates.end() && it->compatible_uses == inst.UseCount() &&
                !ebi.IsFoldedOffset(it->base_inst)) {
                ebi.folded_offsets.push_back(EmittedBlockInfo::FoldedOffsetInfo{it->off_inst, it->base_inst, it->disp});
            }
        }

        for (const auto& inst : block.instructions) {
            std::size_t bitsize = 0;
            if (!MatchFoldableFastmemAccess(inst, conf, fastmem_manager, block.Location(), bitsize)) {
                continue;
            }
            const IR::Inst* const vaddr_inst = inst.GetArg(1).GetInst();
            const IR::Inst* resolved_base = vaddr_inst;
            s64 resolved_disp = 0;
            for (const auto& folded : ebi.folded_offsets) {
                if (folded.off_inst == vaddr_inst) {
                    resolved_base = folded.base_inst;
                    resolved_disp = folded.disp;
                    break;
                }
            }
            ebi.folded_accesses.push_back(EmittedBlockInfo::FoldedAccessInfo{&inst, resolved_base, resolved_disp});
            auto count_it = std::find_if(ebi.fastmem_base_counts.begin(), ebi.fastmem_base_counts.end(),
                                         [resolved_base](const auto& b) { return b.base_inst == resolved_base; });
            if (count_it == ebi.fastmem_base_counts.end()) {
                ebi.fastmem_base_counts.push_back(EmittedBlockInfo::BaseCountInfo{resolved_base, 1});
            } else {
                ++count_it->count;
            }
        }
    }
}

}  // namespace

EmittedBlockInfo EmitArm64(oaknut::CodeGenerator& code, IR::Block block, const EmitConfig& conf, FastmemManager& fastmem_manager) {
#ifdef DYNARMIC_EMIT_STATS
    const auto stat_offset = [&] { return reinterpret_cast<std::uintptr_t>(code.xptr<CodePtr>()); };
    std::size_t stat_mark = stat_offset();
    std::vector<unsigned> deferred_opcodes;
    const auto stat = [&](unsigned opcode, bool deferred) {
        const std::size_t now = stat_offset();
        if (g_emit_stats != nullptr) {
            g_emit_stats(opcode, deferred, now - stat_mark);
        }
        stat_mark = now;
    };
#endif
    if (conf.very_verbose_debugging_output) {
        std::puts(IR::DumpBlock(block).c_str());
    }

    EmittedBlockInfo ebi;
    ebi.entry_point = code.xptr<CodePtr>();
    ebi.nzcv_live_pos = nullptr;

    FpsrManager fpsr_manager{code, conf.state_fpsr_offset};
    thread_local RegAlloc reg_alloc;
    reg_alloc.Reset(code, fpsr_manager, std::span<const int>{GPR_ORDER.begin(), GPR_ORDER.size()},
                    std::span<const int>{FPR_ORDER.begin(), FPR_ORDER.size()});
    EmitContext ctx{block, reg_alloc, conf, ebi, fpsr_manager, fastmem_manager, {}};
    PreAnalyzeBlock(block, conf, fastmem_manager, ebi);

    if (ctx.block.GetCondition() == IR::Cond::AL) {
        ASSERT(!ctx.block.HasConditionFailedLocation());
    } else {
        ASSERT(ctx.block.HasConditionFailedLocation());
        oaknut::Label pass;

        pass = conf.emit_cond(code, ctx, ctx.block.GetCondition());
        EmitAddCycles(code, ctx, ctx.block.ConditionFailedCycleCount());
        conf.emit_condition_failed_terminal(code, ctx);

        code.l(pass);
        ebi.nzcv_live_pos = nullptr;
    }
#ifdef DYNARMIC_EMIT_STATS
    stat(~0u, false);
#endif

    for (auto iter = block.instructions.begin(); iter != block.instructions.end(); ++iter) {
        IR::Inst* inst = &*iter;

        switch (inst->GetOpcode()) {
#define OPCODE(name, type, ...)                    \
    case IR::Opcode::name:                         \
        EmitIR<IR::Opcode::name>(code, ctx, inst); \
        break;
#define A32OPC(name, type, ...)                         \
    case IR::Opcode::A32##name:                         \
        EmitIR<IR::Opcode::A32##name>(code, ctx, inst); \
        break;
#define A64OPC(name, type, ...)                         \
    case IR::Opcode::A64##name:                         \
        EmitIR<IR::Opcode::A64##name>(code, ctx, inst); \
        break;
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
        default:
            UNREACHABLE();
        }

        reg_alloc.UpdateAllUses();
        reg_alloc.AssertAllUnlocked();

        if (conf.very_verbose_debugging_output) {
            EmitVerboseDebuggingOutput(code, ctx);
            ebi.nzcv_live_pos = nullptr;
        }
#ifdef DYNARMIC_EMIT_STATS
        stat(static_cast<unsigned>(inst->GetOpcode()), false);
        deferred_opcodes.resize(ctx.deferred_emits.size(), static_cast<unsigned>(inst->GetOpcode()));
#endif
    }

    const bool flags_live = ebi.nzcv_live_pos != nullptr && ebi.nzcv_live_pos == code.xptr<const void*>();
    fpsr_manager.Spill();
    if (flags_live) {
        ebi.nzcv_live_pos = code.xptr<const void*>();
    }

    reg_alloc.AssertNoMoreUses();

    EmitAddCycles(code, ctx, block.CycleCount());
    if (conf.enable_cycle_counting) {
        ebi.nzcv_live_pos = nullptr;
    }
    conf.emit_terminal(code, ctx);
    ebi.nzcv_live_pos = nullptr;
    code.BRK(0);
#ifdef DYNARMIC_EMIT_STATS
    stat(~1u, false);
#endif

    for (std::size_t i = 0; i < ctx.deferred_emits.size(); ++i) {
        ctx.deferred_emits[i]();
#ifdef DYNARMIC_EMIT_STATS
        stat(i < deferred_opcodes.size() ? deferred_opcodes[i] : ~1u, true);
#endif
    }
    code.BRK(0);

    ebi.block_relocations.shrink_to_fit();
    ebi.fastmem_patch_info.shrink_to_fit();
    ebi.size = code.xptr<CodePtr>() - ebi.entry_point;
    return ebi;
}

void EmitRelocation(oaknut::CodeGenerator& code, EmitContext& ctx, LinkTarget link_target) {
    ctx.ebi.relocations.emplace_back(Relocation{code.xptr<CodePtr>() - ctx.ebi.entry_point, link_target});
    code.NOP();
}

void EmitBlockLinkRelocation(oaknut::CodeGenerator& code, EmitContext& ctx, const IR::LocationDescriptor& descriptor, BlockRelocationType type) {
    ctx.ebi.block_relocations[descriptor].emplace_back(BlockRelocation{code.xptr<CodePtr>() - ctx.ebi.entry_point, type});
    switch (type) {
    case BlockRelocationType::Branch:
        code.NOP();
        break;
    case BlockRelocationType::MoveToScratch1:
        code.BRK(0);
        code.NOP();
        break;
    default:
        UNREACHABLE();
    }
}

}  // namespace Dynarmic::Backend::Arm64
