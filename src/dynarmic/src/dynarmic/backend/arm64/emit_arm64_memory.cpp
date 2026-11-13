// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/emit_arm64_memory.h"

#include <optional>
#include <utility>
#include <cstddef>

#include <bit>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

namespace {

bool IsOrdered(IR::AccType acctype) {
    return acctype == IR::AccType::ORDERED || acctype == IR::AccType::ORDEREDRW || acctype == IR::AccType::LIMITEDORDERED;
}

LinkTarget ReadMemoryLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::ReadMemory8;
    case 16:
        return LinkTarget::ReadMemory16;
    case 32:
        return LinkTarget::ReadMemory32;
    case 64:
        return LinkTarget::ReadMemory64;
    case 128:
        return LinkTarget::ReadMemory128;
    }
    UNREACHABLE();
}

LinkTarget WriteMemoryLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WriteMemory8;
    case 16:
        return LinkTarget::WriteMemory16;
    case 32:
        return LinkTarget::WriteMemory32;
    case 64:
        return LinkTarget::WriteMemory64;
    case 128:
        return LinkTarget::WriteMemory128;
    }
    UNREACHABLE();
}

LinkTarget WrappedReadMemoryLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WrappedReadMemory8;
    case 16:
        return LinkTarget::WrappedReadMemory16;
    case 32:
        return LinkTarget::WrappedReadMemory32;
    case 64:
        return LinkTarget::WrappedReadMemory64;
    case 128:
        return LinkTarget::WrappedReadMemory128;
    }
    UNREACHABLE();
}

LinkTarget WrappedWriteMemoryLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WrappedWriteMemory8;
    case 16:
        return LinkTarget::WrappedWriteMemory16;
    case 32:
        return LinkTarget::WrappedWriteMemory32;
    case 64:
        return LinkTarget::WrappedWriteMemory64;
    case 128:
        return LinkTarget::WrappedWriteMemory128;
    }
    UNREACHABLE();
}

LinkTarget ExclusiveReadMemoryLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::ExclusiveReadMemory8;
    case 16:
        return LinkTarget::ExclusiveReadMemory16;
    case 32:
        return LinkTarget::ExclusiveReadMemory32;
    case 64:
        return LinkTarget::ExclusiveReadMemory64;
    case 128:
        return LinkTarget::ExclusiveReadMemory128;
    }
    UNREACHABLE();
}

LinkTarget ExclusiveWriteMemoryLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::ExclusiveWriteMemory8;
    case 16:
        return LinkTarget::ExclusiveWriteMemory16;
    case 32:
        return LinkTarget::ExclusiveWriteMemory32;
    case 64:
        return LinkTarget::ExclusiveWriteMemory64;
    case 128:
        return LinkTarget::ExclusiveWriteMemory128;
    }
    UNREACHABLE();
}

LinkTarget WrappedAtomicRmwLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WrappedAtomicRmw8;
    case 16:
        return LinkTarget::WrappedAtomicRmw16;
    case 32:
        return LinkTarget::WrappedAtomicRmw32;
    case 64:
        return LinkTarget::WrappedAtomicRmw64;
    }
    UNREACHABLE();
}

LinkTarget WrappedAtomicCasLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WrappedAtomicCas8;
    case 16:
        return LinkTarget::WrappedAtomicCas16;
    case 32:
        return LinkTarget::WrappedAtomicCas32;
    case 64:
        return LinkTarget::WrappedAtomicCas64;
    }
    UNREACHABLE();
}

template<std::size_t bitsize>
void CallbackOnlyEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    EmitRelocation(code, ctx, ReadMemoryLinkTarget(bitsize));
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }

    if constexpr (bitsize == 128) {
        code.MOV(Q8.B16(), Q0.B16());
        ctx.reg_alloc.DefineAsRegister(inst, Q8);
    } else {
        ctx.reg_alloc.DefineAsRegister(inst, X0);
    }
}

template<std::size_t bitsize>
void CallbackOnlyEmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    code.MOV(Wscratch0, 1);
    code.STRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitRelocation(code, ctx, ExclusiveReadMemoryLinkTarget(bitsize));
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }

    if constexpr (bitsize == 128) {
        code.MOV(Q8.B16(), Q0.B16());
        ctx.reg_alloc.DefineAsRegister(inst, Q8);
    } else {
        ctx.reg_alloc.DefineAsRegister(inst, X0);
    }
}

template<std::size_t bitsize>
void CallbackOnlyEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    EmitRelocation(code, ctx, WriteMemoryLinkTarget(bitsize));
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
}

template<std::size_t bitsize>
void CallbackOnlyEmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    oaknut::Label end;

    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    code.MOV(W0, 1);
    code.LDRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.CBZ(Wscratch0, end);
    code.STRB(WZR, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitRelocation(code, ctx, ExclusiveWriteMemoryLinkTarget(bitsize));
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    code.l(end);
    ctx.reg_alloc.DefineAsRegister(inst, X0);
}

constexpr std::size_t page_table_const_bits = 12;
constexpr std::size_t page_table_const_size = 1 << page_table_const_bits;
constexpr std::size_t page_table_const_mask = (1 << page_table_const_bits) - 1;

// This function may use Xscratch0 as a scratch register
// Trashes NZCV
template<std::size_t bitsize>
void EmitDetectMisalignedVAddr(oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr, const SharedLabel& fallback) {
    static_assert(bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64 || bitsize == 128);

    if (bitsize == 8 || (ctx.conf.detect_misaligned_access_via_page_table & bitsize) == 0) {
        return;
    }

    if (!ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary) {
        code.TST(Xaddr, u64(bitsize / 8 - 1));
        code.B(NE, *fallback);
    } else {
        // If (addr & page_table_const_mask) > page_table_const_size - byte_size, use fallback.
        code.AND(Wscratch0, Xaddr.toW(), u32(page_table_const_mask));
        code.CMP(Wscratch0, u32(page_table_const_size - bitsize / 8));
        code.B(HI, *fallback);
    }
}

// Outputs Xscratch0 = page_table[addr >> page_table_const_bits]
// May use Xscratch1 as scratch register
// Address to read/write = [ret0 + ret1], ret0 is always Xscratch0 and ret1 is either Xaddr or Xscratch1
// Trashes NZCV
template<std::size_t bitsize>
std::pair<oaknut::XReg, oaknut::XReg> InlinePageTableEmitVAddrLookup(
    oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr,
    const SharedLabel& fallback, bool near_fallback = false) {
    const std::size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_table_const_bits;
    const std::size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;

    EmitDetectMisalignedVAddr<bitsize>(code, ctx, Xaddr, fallback);

    if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
        code.UBFX(Xscratch0, Xaddr, page_table_const_bits, valid_page_index_bits);
    } else {
        code.LSR(Xscratch0, Xaddr, page_table_const_bits);
        code.TST(Xscratch0, u64(~u64(0)) << valid_page_index_bits);
        code.B(NE, *fallback);
    }

    if (ctx.conf.page_table_log2_stride == 3) {
        code.LDR(Xscratch0, Xpagetable, Xscratch0, oaknut::IndexExt::LSL, 3);
    } else {
        if (ctx.conf.page_table_log2_stride != 0) {
            code.LSL(Xscratch0, Xscratch0, ctx.conf.page_table_log2_stride);
        }
        code.LDR(Xscratch0, Xpagetable, Xscratch0);
    }

    if (ctx.conf.page_table_marked_bit) {
        if (near_fallback || ctx.block.instructions.size() < 512) {
            code.TBNZ(Xscratch0, *ctx.conf.page_table_marked_bit, *fallback);
        } else {
            code.TST(Xscratch0, 1ULL << *ctx.conf.page_table_marked_bit);
            code.B(NE, *fallback);
        }
    }

    if (ctx.conf.page_table_pointer_mask != 0) {
        code.AND(Xscratch0, Xscratch0, ctx.conf.page_table_pointer_mask);
    }

    if (ctx.conf.page_table_sign_extension) {
        code.SBFM(Xscratch0, Xscratch0, 0, *ctx.conf.page_table_sign_extension);
    }

    code.CBZ(Xscratch0, *fallback);

    if (ctx.conf.absolute_offset_page_table) {
        return std::make_pair(Xscratch0, Xaddr);
    }
    code.AND(Xscratch1, Xaddr, page_table_const_mask);
    return std::make_pair(Xscratch0, Xscratch1);
}

template<std::size_t bitsize>
CodePtr EmitMemoryLdr(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xbase, oaknut::XReg Xoffset, bool ordered, bool extend32 = false) {
    const auto index_ext = extend32 ? oaknut::IndexExt::UXTW : oaknut::IndexExt::LSL;
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};

    CodePtr fastmem_location = code.xptr<CodePtr>();

    if (ordered) {
        code.ADD(Xscratch0, Xbase, Roffset, add_ext);

        fastmem_location = code.xptr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.LDARB(oaknut::WReg{value_idx}, Xscratch0);
            break;
        case 16:
            code.LDARH(oaknut::WReg{value_idx}, Xscratch0);
            break;
        case 32:
            code.LDAR(oaknut::WReg{value_idx}, Xscratch0);
            break;
        case 64:
            code.LDAR(oaknut::XReg{value_idx}, Xscratch0);
            break;
        case 128:
            code.LDR(oaknut::QReg{value_idx}, Xscratch0);
            code.DMB(oaknut::BarrierOp::ISH);
            break;
        default:
            UNREACHABLE();
        }
    } else {
        fastmem_location = code.xptr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.LDRB(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 16:
            code.LDRH(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 32:
            code.LDR(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 64:
            code.LDR(oaknut::XReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 128:
            code.LDR(oaknut::QReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        default:
            UNREACHABLE();
        }
    }

    return fastmem_location;
}

template<std::size_t bitsize>
CodePtr EmitMemoryStr(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xbase, oaknut::XReg Xoffset, bool ordered, bool extend32 = false) {
    const auto index_ext = extend32 ? oaknut::IndexExt::UXTW : oaknut::IndexExt::LSL;
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};

    CodePtr fastmem_location;

    if (ordered) {
        code.ADD(Xscratch0, Xbase, Roffset, add_ext);

        fastmem_location = code.xptr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.STLRB(oaknut::WReg{value_idx}, Xscratch0);
            break;
        case 16:
            code.STLRH(oaknut::WReg{value_idx}, Xscratch0);
            break;
        case 32:
            code.STLR(oaknut::WReg{value_idx}, Xscratch0);
            break;
        case 64:
            code.STLR(oaknut::XReg{value_idx}, Xscratch0);
            break;
        case 128:
            code.DMB(oaknut::BarrierOp::ISH);
            code.STR(oaknut::QReg{value_idx}, Xscratch0);
            code.DMB(oaknut::BarrierOp::ISH);
            break;
        default:
            UNREACHABLE();
        }
    } else {
        fastmem_location = code.xptr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.STRB(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 16:
            code.STRH(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 32:
            code.STR(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 64:
            code.STR(oaknut::XReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 128:
            code.STR(oaknut::QReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        default:
            UNREACHABLE();
        }
    }

    return fastmem_location;
}

LinkTarget PageTableLookupLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::PageTableLookup8;
    case 16:
        return LinkTarget::PageTableLookup16;
    case 32:
        return LinkTarget::PageTableLookup32;
    case 64:
        return LinkTarget::PageTableLookup64;
    case 128:
        return LinkTarget::PageTableLookup128;
    }
    UNREACHABLE();
}

// [Xptr] with nothing added (the shared page table walk returned the host address).
template<std::size_t bitsize>
void EmitLoadAt(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xptr, bool ordered) {
    switch (bitsize) {
    case 8:
        ordered ? code.LDARB(oaknut::WReg{value_idx}, Xptr) : code.LDRB(oaknut::WReg{value_idx}, Xptr);
        break;
    case 16:
        ordered ? code.LDARH(oaknut::WReg{value_idx}, Xptr) : code.LDRH(oaknut::WReg{value_idx}, Xptr);
        break;
    case 32:
        ordered ? code.LDAR(oaknut::WReg{value_idx}, Xptr) : code.LDR(oaknut::WReg{value_idx}, Xptr);
        break;
    case 64:
        ordered ? code.LDAR(oaknut::XReg{value_idx}, Xptr) : code.LDR(oaknut::XReg{value_idx}, Xptr);
        break;
    case 128:
        code.LDR(oaknut::QReg{value_idx}, Xptr);
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        break;
    default:
        UNREACHABLE();
    }
}

template<std::size_t bitsize>
void EmitStoreAt(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xptr, bool ordered) {
    switch (bitsize) {
    case 8:
        ordered ? code.STLRB(oaknut::WReg{value_idx}, Xptr) : code.STRB(oaknut::WReg{value_idx}, Xptr);
        break;
    case 16:
        ordered ? code.STLRH(oaknut::WReg{value_idx}, Xptr) : code.STRH(oaknut::WReg{value_idx}, Xptr);
        break;
    case 32:
        ordered ? code.STLR(oaknut::WReg{value_idx}, Xptr) : code.STR(oaknut::WReg{value_idx}, Xptr);
        break;
    case 64:
        ordered ? code.STLR(oaknut::XReg{value_idx}, Xptr) : code.STR(oaknut::XReg{value_idx}, Xptr);
        break;
    case 128:
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        code.STR(oaknut::QReg{value_idx}, Xptr);
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        break;
    default:
        UNREACHABLE();
    }
}

template<std::size_t bitsize>
void InlinePageTableEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.WriteQ(inst);
        } else {
            return ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        }
    }();
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    if (ctx.conf.page_table_lookup_calls) {
        code.MOV(Xscratch0, *Xaddr);
        EmitRelocation(code, ctx, PageTableLookupLinkTarget(bitsize));
        code.CBZ(Xscratch0, *fallback);
        EmitLoadAt<bitsize>(code, Rvalue->index(), Xscratch0, ordered);
    } else {
        const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
        EmitMemoryLdr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered);
    }

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end] {
        code.l(*fallback);
        code.MOV(Xscratch0, Xaddr);
        EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        if constexpr (bitsize == 128) {
            code.MOV(Rvalue.B16(), Q0.B16());
        } else {
            code.MOV(Rvalue.toX(), Xscratch0);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

template<std::size_t bitsize>
void InlinePageTableEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.ReadQ(args[2]);
        } else {
            return ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        }
    }();
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    if (ctx.conf.page_table_lookup_calls) {
        code.MOV(Xscratch0, *Xaddr);
        EmitRelocation(code, ctx, PageTableLookupLinkTarget(bitsize));
        code.CBZ(Xscratch0, *fallback);
        EmitStoreAt<bitsize>(code, Rvalue->index(), Xscratch0, ordered);
    } else {
        const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
        EmitMemoryStr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered);
    }

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end] {
        code.l(*fallback);
        if constexpr (bitsize == 128) {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Q0.B16(), Rvalue.B16());
        } else {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Rvalue.toX());
        }
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

std::optional<DoNotFastmemMarker> ShouldFastmem(EmitContext& ctx, IR::Inst* inst) {
    if (!ctx.conf.fastmem_pointer || !ctx.fastmem.SupportsFastmem()) {
        return std::nullopt;
    }

    const auto marker = std::make_tuple(ctx.block.Location(), inst->GetName());
    if (ctx.fastmem.ShouldFastmem(marker)) {
        return marker;
    }
    return std::nullopt;
}

inline bool ShouldExt32(EmitContext& ctx) {
    return ctx.conf.fastmem_address_space_bits == 32 && ctx.conf.silently_mirror_fastmem;
}

inline bool CanInlinePageTableFallback(const EmitContext& ctx) {
    return ctx.conf.page_table_pointer != 0 &&
           ctx.conf.inline_page_table_fallback &&
           !ctx.conf.silently_mirror_fastmem &&
           ctx.conf.fastmem_address_space_bits < 64 &&
           ctx.conf.fastmem_address_space_bits < ctx.conf.page_table_address_space_bits;
}

// May use Xscratch0 as scratch register
// Address to read/write = [ret0 + ret1], ret0 is always Xfastmem and ret1 is either Xaddr or Xscratch0
// Trashes NZCV
template<std::size_t bitsize>
std::pair<oaknut::XReg, oaknut::XReg> FastmemEmitVAddrLookup(oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr, const SharedLabel& fallback) {
    if (ctx.conf.fastmem_address_space_bits == 64 || ShouldExt32(ctx)) {
        return std::make_pair(Xfastmem, Xaddr);
    }

    if (ctx.conf.silently_mirror_fastmem) {
        code.UBFX(Xscratch0, Xaddr, 0, ctx.conf.fastmem_address_space_bits);
        return std::make_pair(Xfastmem, Xscratch0);
    }

    code.LSR(Xscratch0, Xaddr, ctx.conf.fastmem_address_space_bits);
    code.CBNZ(Xscratch0, *fallback);
    return std::make_pair(Xfastmem, Xaddr);
}

template<std::size_t bitsize>
void FastmemEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.WriteQ(inst);
        } else {
            return ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        }
    }();
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();
    SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
    const auto fastmem_location = EmitMemoryLdr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
        if (use_pt_fallback) {
            code.l(*bounds_miss);
            const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback, true);
            EmitMemoryLdr<bitsize>(code, Rvalue.index(), Xpt_base, Xpt_offset, ordered);
            code.B(*end);
        }

        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = std::bit_cast<u64>(code.xptr<void*>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
            });

        code.l(*fallback);
        code.MOV(Xscratch0, Xaddr);
        EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        if constexpr (bitsize == 128) {
            code.MOV(Rvalue.B16(), Q0.B16());
        } else {
            code.MOV(Rvalue.toX(), Xscratch0);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

template<std::size_t bitsize>
void FastmemEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.ReadQ(args[2]);
        } else {
            return ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        }
    }();
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();
    SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
    const auto fastmem_location = EmitMemoryStr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
        if (use_pt_fallback) {
            code.l(*bounds_miss);
            const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback, true);
            EmitMemoryStr<bitsize>(code, Rvalue.index(), Xpt_base, Xpt_offset, ordered);
            code.B(*end);
        }

        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = std::bit_cast<u64>(code.xptr<void*>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
            });

        code.l(*fallback);
        if constexpr (bitsize == 128) {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Q0.B16(), Rvalue.B16());
        } else {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Rvalue.toX());
        }
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

#define EMIT_LSE_RMW_ORDER(BASE, SUFFIX, RS, RT, XN) \
    switch ((encoded_op >> 4) & 0b11) {              \
    case 0b00:                                       \
        code.BASE##SUFFIX(RS, RT, XN);               \
        break;                                       \
    case 0b01:                                       \
        code.BASE##L##SUFFIX(RS, RT, XN);            \
        break;                                       \
    case 0b10:                                       \
        code.BASE##A##SUFFIX(RS, RT, XN);            \
        break;                                       \
    default:                                         \
        code.BASE##AL##SUFFIX(RS, RT, XN);           \
        break;                                       \
    }

#define EMIT_LSE_RMW_OP(SUFFIX, RS, RT, XN)                  \
    switch (encoded_op & 0x0F) {                             \
    case 0:                                                  \
        EMIT_LSE_RMW_ORDER(LDADD, SUFFIX, RS, RT, XN);       \
        break;                                               \
    case 1:                                                  \
        EMIT_LSE_RMW_ORDER(LDCLR, SUFFIX, RS, RT, XN);       \
        break;                                               \
    case 2:                                                  \
        EMIT_LSE_RMW_ORDER(LDEOR, SUFFIX, RS, RT, XN);       \
        break;                                               \
    case 3:                                                  \
        EMIT_LSE_RMW_ORDER(LDSET, SUFFIX, RS, RT, XN);       \
        break;                                               \
    case 4:                                                  \
        EMIT_LSE_RMW_ORDER(LDSMAX, SUFFIX, RS, RT, XN);      \
        break;                                               \
    case 5:                                                  \
        EMIT_LSE_RMW_ORDER(LDSMIN, SUFFIX, RS, RT, XN);      \
        break;                                               \
    case 6:                                                  \
        EMIT_LSE_RMW_ORDER(LDUMAX, SUFFIX, RS, RT, XN);      \
        break;                                               \
    case 7:                                                  \
        EMIT_LSE_RMW_ORDER(LDUMIN, SUFFIX, RS, RT, XN);      \
        break;                                               \
    default:                                                 \
        EMIT_LSE_RMW_ORDER(SWP, SUFFIX, RS, RT, XN);         \
        break;                                               \
    }

template<std::size_t bitsize>
CodePtr EmitLseRmwAt(oaknut::CodeGenerator& code, u8 encoded_op, int operand_idx, int result_idx,
                     oaknut::XReg Xbase, oaknut::XReg Xoffset, bool extend32 = false) {
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};
    code.ADD(Xscratch0, Xbase, Roffset, add_ext);
    const CodePtr location = code.xptr<CodePtr>();
    if constexpr (bitsize == 8) {
        oaknut::WReg ws{operand_idx}, wt{result_idx};
        EMIT_LSE_RMW_OP(B, ws, wt, Xscratch0);
    } else if constexpr (bitsize == 16) {
        oaknut::WReg ws{operand_idx}, wt{result_idx};
        EMIT_LSE_RMW_OP(H, ws, wt, Xscratch0);
    } else if constexpr (bitsize == 32) {
        oaknut::WReg ws{operand_idx}, wt{result_idx};
        EMIT_LSE_RMW_OP(, ws, wt, Xscratch0);
    } else {
        oaknut::XReg xs{operand_idx}, xt{result_idx};
        EMIT_LSE_RMW_OP(, xs, xt, Xscratch0);
    }
    return location;
}

#undef EMIT_LSE_RMW_OP
#undef EMIT_LSE_RMW_ORDER

template<std::size_t bitsize>
CodePtr EmitLseCasAt(oaknut::CodeGenerator& code, int expected_idx, int desired_idx, int result_idx,
                     oaknut::XReg Xbase, oaknut::XReg Xoffset, bool extend32 = false) {
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};
    code.ADD(Xscratch0, Xbase, Roffset, add_ext);
    if constexpr (bitsize == 64) {
        code.MOV(oaknut::XReg{result_idx}, oaknut::XReg{expected_idx});
    } else {
        code.MOV(oaknut::WReg{result_idx}, oaknut::WReg{expected_idx});
    }
    const CodePtr location = code.xptr<CodePtr>();
    if constexpr (bitsize == 8) {
        code.CASALB(oaknut::WReg{result_idx}, oaknut::WReg{desired_idx}, Xscratch0);
    } else if constexpr (bitsize == 16) {
        code.CASALH(oaknut::WReg{result_idx}, oaknut::WReg{desired_idx}, Xscratch0);
    } else if constexpr (bitsize == 32) {
        code.CASAL(oaknut::WReg{result_idx}, oaknut::WReg{desired_idx}, Xscratch0);
    } else {
        code.CASAL(oaknut::XReg{result_idx}, oaknut::XReg{desired_idx}, Xscratch0);
    }
    return location;
}

}  // namespace

// The inline walk of InlinePageTableEmitVAddrLookup, once: it was ~12 instructions at every
// access that faulted out of fastmem (65 bytes per load or store against 29 with fastmem).
void* EmitPageTableLookupThunk(oaknut::CodeGenerator& code, const EmitConfig& conf, std::size_t bitsize) {
    void* const entry = code.xptr<void*>();
    oaknut::Label miss;
    const std::size_t valid_page_index_bits = conf.page_table_address_space_bits - page_table_const_bits;
    const std::size_t unused_top_bits = 64 - conf.page_table_address_space_bits;

    if (bitsize != 8 && (conf.detect_misaligned_access_via_page_table & bitsize) != 0) {
        if (!conf.only_detect_misalignment_via_page_table_on_page_boundary) {
            code.TST(Xscratch0, u64(bitsize / 8 - 1));
            code.B(NE, miss);
        } else {
            code.AND(Wscratch1, Wscratch0, u32(page_table_const_mask));
            code.CMP(Wscratch1, u32(page_table_const_size - bitsize / 8));
            code.B(HI, miss);
        }
    }
    if (conf.silently_mirror_page_table || unused_top_bits == 0) {
        code.UBFX(Xscratch1, Xscratch0, page_table_const_bits, valid_page_index_bits);
    } else {
        code.LSR(Xscratch1, Xscratch0, page_table_const_bits);
        code.TST(Xscratch1, u64(~u64(0)) << valid_page_index_bits);
        code.B(NE, miss);
    }
    if (conf.page_table_log2_stride == 3) {
        code.LDR(Xscratch1, Xpagetable, Xscratch1, oaknut::IndexExt::LSL, 3);
    } else {
        if (conf.page_table_log2_stride != 0) {
            code.LSL(Xscratch1, Xscratch1, conf.page_table_log2_stride);
        }
        code.LDR(Xscratch1, Xpagetable, Xscratch1);
    }
    if (conf.page_table_marked_bit) {
        code.TBNZ(Xscratch1, *conf.page_table_marked_bit, miss);
    }
    if (conf.page_table_pointer_mask != 0) {
        code.AND(Xscratch1, Xscratch1, conf.page_table_pointer_mask);
    }
    if (conf.page_table_sign_extension) {
        code.SBFM(Xscratch1, Xscratch1, 0, *conf.page_table_sign_extension);
    }
    code.CBZ(Xscratch1, miss);
    if (conf.absolute_offset_page_table) {
        code.ADD(Xscratch0, Xscratch1, Xscratch0);
    } else {
        code.AND(Xscratch0, Xscratch0, page_table_const_mask);
        code.ADD(Xscratch0, Xscratch1, Xscratch0);
    }
    code.RET();
    code.l(miss);
    code.MOV(Xscratch0, 0);
    code.RET();
    return entry;
}

template<std::size_t bitsize>
void EmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (const auto marker = ShouldFastmem(ctx, inst)) {
        FastmemEmitReadMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitReadMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitReadMemory<bitsize>(code, ctx, inst);
    }
}

template<std::size_t bitsize>
void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    CallbackOnlyEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
}

template<std::size_t bitsize>
void EmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (const auto marker = ShouldFastmem(ctx, inst)) {
        FastmemEmitWriteMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitWriteMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitWriteMemory<bitsize>(code, ctx, inst);
    }
}

template<std::size_t bitsize>
void EmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    CallbackOnlyEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
}

template<std::size_t bitsize>
void EmitAtomicRmwMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Roperand = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
    auto Rresult = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
    const u8 encoded_op = args[3].GetImmediateU8();
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Roperand, Rresult);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    if (const auto marker = ShouldFastmem(ctx, inst)) {
        const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
        SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        const CodePtr fastmem_location = EmitLseRmwAt<bitsize>(
            code, encoded_op, Roperand->index(), Rresult->index(), Xbase, Xoffset, ShouldExt32(ctx));

        ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker = *marker, Xaddr = *Xaddr, Roperand = *Roperand, Rresult = *Rresult, encoded_op, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
            if (use_pt_fallback) {
                code.l(*bounds_miss);
                if constexpr (bitsize > 8) {
                    code.TST(Xaddr, u64(bitsize / 8 - 1));
                    code.B(NE, *fallback);
                }
                const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback, true);
                EmitLseRmwAt<bitsize>(code, encoded_op, Roperand.index(), Rresult.index(), Xpt_base, Xpt_offset);
                code.B(*end);
            }

            ctx.ebi.fastmem_patch_info.emplace(
                fastmem_location - ctx.ebi.entry_point,
                FastmemPatchInfo{
                    .marker = marker,
                    .fc = FakeCall{
                        .call_pc = std::bit_cast<u64>(code.xptr<void*>()),
                    },
                    .recompile = ctx.conf.recompile_on_exclusive_fastmem_failure,
                });

            code.l(*fallback);
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Roperand.toX());
            code.MOV(Xscratch2, u64(encoded_op & 0x0F));
            code.STR(Xscratch2, Xstate, ctx.conf.state_atomic_scratch_offset);
            EmitRelocation(code, ctx, WrappedAtomicRmwLinkTarget(bitsize));
            code.DMB(oaknut::BarrierOp::ISH);
            code.MOV(Rresult.toX(), Xscratch0);
            ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
            code.B(*end);
        });
    } else if (ctx.conf.page_table_pointer != 0) {
        if constexpr (bitsize > 8) {
            code.TST(Xaddr, u64(bitsize / 8 - 1));
            code.B(NE, *fallback);
        }
        const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
        EmitLseRmwAt<bitsize>(code, encoded_op, Roperand->index(), Rresult->index(), Xpt_base, Xpt_offset);

        ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Roperand = *Roperand, Rresult = *Rresult, encoded_op, fallback, end] {
            code.l(*fallback);
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Roperand.toX());
            code.MOV(Xscratch2, u64(encoded_op & 0x0F));
            code.STR(Xscratch2, Xstate, ctx.conf.state_atomic_scratch_offset);
            EmitRelocation(code, ctx, WrappedAtomicRmwLinkTarget(bitsize));
            code.DMB(oaknut::BarrierOp::ISH);
            code.MOV(Rresult.toX(), Xscratch0);
            ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
            code.B(*end);
        });
    } else {
        code.MOV(Xscratch0, *Xaddr);
        code.MOV(Xscratch1, Roperand->toX());
        code.MOV(Xscratch2, u64(encoded_op & 0x0F));
        code.STR(Xscratch2, Xstate, ctx.conf.state_atomic_scratch_offset);
        EmitRelocation(code, ctx, WrappedAtomicRmwLinkTarget(bitsize));
        code.DMB(oaknut::BarrierOp::ISH);
        code.MOV(Rresult->toX(), Xscratch0);
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
    }

    code.l(*end);
}

template<std::size_t bitsize>
void EmitAtomicCasMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rexpected = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
    auto Rdesired = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[3]);
    auto Rresult = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rexpected, Rdesired, Rresult);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    if (const auto marker = ShouldFastmem(ctx, inst)) {
        const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
        SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        const CodePtr fastmem_location = EmitLseCasAt<bitsize>(
            code, Rexpected->index(), Rdesired->index(), Rresult->index(), Xbase, Xoffset, ShouldExt32(ctx));

        ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker = *marker, Xaddr = *Xaddr, Rexpected = *Rexpected, Rdesired = *Rdesired, Rresult = *Rresult, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
            if (use_pt_fallback) {
                code.l(*bounds_miss);
                if constexpr (bitsize > 8) {
                    code.TST(Xaddr, u64(bitsize / 8 - 1));
                    code.B(NE, *fallback);
                }
                const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback, true);
                EmitLseCasAt<bitsize>(code, Rexpected.index(), Rdesired.index(), Rresult.index(), Xpt_base, Xpt_offset);
                code.B(*end);
            }

            ctx.ebi.fastmem_patch_info.emplace(
                fastmem_location - ctx.ebi.entry_point,
                FastmemPatchInfo{
                    .marker = marker,
                    .fc = FakeCall{
                        .call_pc = std::bit_cast<u64>(code.xptr<void*>()),
                    },
                    .recompile = ctx.conf.recompile_on_exclusive_fastmem_failure,
                });

            code.l(*fallback);
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Rexpected.toX());
            code.STR(Rdesired.toX(), Xstate, ctx.conf.state_atomic_scratch_offset);
            EmitRelocation(code, ctx, WrappedAtomicCasLinkTarget(bitsize));
            code.DMB(oaknut::BarrierOp::ISH);
            code.MOV(Rresult.toX(), Xscratch0);
            ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
            code.B(*end);
        });
    } else if (ctx.conf.page_table_pointer != 0) {
        if constexpr (bitsize > 8) {
            code.TST(Xaddr, u64(bitsize / 8 - 1));
            code.B(NE, *fallback);
        }
        const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
        EmitLseCasAt<bitsize>(code, Rexpected->index(), Rdesired->index(), Rresult->index(), Xpt_base, Xpt_offset);

        ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rexpected = *Rexpected, Rdesired = *Rdesired, Rresult = *Rresult, fallback, end] {
            code.l(*fallback);
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Rexpected.toX());
            code.STR(Rdesired.toX(), Xstate, ctx.conf.state_atomic_scratch_offset);
            EmitRelocation(code, ctx, WrappedAtomicCasLinkTarget(bitsize));
            code.DMB(oaknut::BarrierOp::ISH);
            code.MOV(Rresult.toX(), Xscratch0);
            ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
            code.B(*end);
        });
    } else {
        code.MOV(Xscratch0, *Xaddr);
        code.MOV(Xscratch1, Rexpected->toX());
        code.STR(Rdesired->toX(), Xstate, ctx.conf.state_atomic_scratch_offset);
        EmitRelocation(code, ctx, WrappedAtomicCasLinkTarget(bitsize));
        code.DMB(oaknut::BarrierOp::ISH);
        code.MOV(Rresult->toX(), Xscratch0);
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
    }

    code.l(*end);
}

template void EmitReadMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicRmwMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicRmwMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicRmwMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicRmwMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicCasMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicCasMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicCasMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitAtomicCasMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

}  // namespace Dynarmic::Backend::Arm64
