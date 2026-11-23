// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/emit_arm64_memory.h"

#include <array>
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

struct JitHotWriteCallbacks {
    std::uintptr_t ctx_ptr = 0;
    std::uintptr_t writes_base = 0;
    std::array<std::uintptr_t, 5> write_fns{};
    bool preserve_all_abi = false;
};

std::optional<JitHotWriteCallbacks> FindJitHotWrite(const u64* tpidrro_el0) noexcept;

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

struct UntrappedCachedReadLabels {
    SharedLabel check_cached;
    SharedLabel resume_unmarked;
};

void EmitUntrappedCachedReadCheck(oaknut::CodeGenerator& code, const EmitContext& ctx,
                                  oaknut::XReg Xaddr, const UntrappedCachedReadLabels& labels,
                                  const SharedLabel& fallback) {
    if (!labels.check_cached) {
        return;
    }
    const std::size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_table_const_bits;
    const std::size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;

    code.l(*labels.check_cached);
    // Bit 0 is already 1; RasterizerCachedMemory (0b111) requires bits 1 and 2 to also be 1.
    code.TBZ(Xscratch0, 1, *fallback);
    code.TBZ(Xscratch0, 2, *fallback);
    code.MOV(Xscratch1, ctx.conf.untrapped_read_bitmap_ptr);
    code.LDR(Xscratch1, Xscratch1);
    code.CBZ(Xscratch1, *fallback);
    code.CMP(Xscratch1, 1);
    code.B(EQ, *labels.resume_unmarked);
    if (valid_page_index_bits <= 27) {
        if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
            if (valid_page_index_bits > 6) {
                code.UBFX(Xscratch2, Xaddr, page_table_const_bits + 6, valid_page_index_bits - 6);
            } else {
                code.MOV(Xscratch2, XZR);
            }
        } else {
            code.LSR(Xscratch2, Xaddr, page_table_const_bits + 6);
        }
    } else {
        if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
            code.UBFX(Xscratch2, Xaddr, page_table_const_bits, valid_page_index_bits);
        } else {
            code.LSR(Xscratch2, Xaddr, page_table_const_bits);
        }
        code.TST(Xscratch2, ~((u64{1} << 27) - 1));
        code.B(NE, *labels.resume_unmarked);
        code.LSR(Xscratch2, Xscratch2, 6);
    }
    code.LDR(Xscratch1, Xscratch1, Xscratch2, oaknut::IndexExt::LSL, 3);
    code.UBFX(Xscratch2, Xaddr, page_table_const_bits, 6);
    code.LSR(Xscratch1, Xscratch1, Xscratch2);
    code.TBNZ(Xscratch1, 0, *fallback);
    code.B(*labels.resume_unmarked);
}

template<std::size_t bitsize>
bool CanInlineCrossPage(const EmitContext& ctx, bool ordered) noexcept {
    return bitsize > 8 &&
           (!ordered || bitsize == 128) &&
           !ctx.conf.page_table_lookup_calls &&
           (ctx.conf.detect_misaligned_access_via_page_table & bitsize) != 0 &&
           ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary &&
           ctx.conf.absolute_offset_page_table &&
           ctx.conf.page_table_log2_stride == 3 &&
           !ctx.conf.silently_mirror_page_table &&
           ctx.conf.page_table_address_space_bits > page_table_const_bits &&
           ctx.conf.page_table_address_space_bits < 64;
}

// Checks whether a cross-4K-page access spans two adjacent unmarked PageType::Memory pages that
// are contiguous in host backing memory (identical masked deltas in absolute_offset_page_table).
// Outputs Xscratch0 = shared host delta; branches to *fallback on any miss without modifying Xaddr.
template<std::size_t bitsize>
void EmitCrossPageContiguousLookup(oaknut::CodeGenerator& code, const EmitContext& ctx,
                                   oaknut::XReg Xaddr, const SharedLabel& fallback) {
    if constexpr (bitsize > 8) {
        code.ADDS(Xscratch1, Xaddr, u64(bitsize / 8 - 1));
        code.B(CS, *fallback);
        code.LSR(Xscratch1, Xscratch1, ctx.conf.page_table_address_space_bits);
        code.CBNZ(Xscratch1, *fallback);

        code.LSR(Xscratch0, Xaddr, page_table_const_bits);
        code.ADD(Xscratch0, Xpagetable, Xscratch0, LSL, 3);
        code.LDP(Xscratch0, Xscratch1, Xscratch0, 0);
        if (ctx.conf.page_table_marked_bit) {
            code.ORR(Xscratch2, Xscratch0, Xscratch1);
            code.TBNZ(Xscratch2, *ctx.conf.page_table_marked_bit, *fallback);
        }
        if (ctx.conf.page_table_pointer_mask != 0) {
            code.AND(Xscratch0, Xscratch0, ctx.conf.page_table_pointer_mask);
            code.AND(Xscratch1, Xscratch1, ctx.conf.page_table_pointer_mask);
        }
        code.CBZ(Xscratch0, *fallback);
        code.CMP(Xscratch0, Xscratch1);
        code.B(NE, *fallback);
        if (ctx.conf.page_table_sign_extension) {
            code.SBFM(Xscratch0, Xscratch0, 0, *ctx.conf.page_table_sign_extension);
        }
    }
}

// Outputs Xscratch0 = page_table[addr >> page_table_const_bits]
// May use Xscratch1 as scratch register
// Address to read/write = [ret0 + ret1], ret0 is always Xscratch0 and ret1 is either Xaddr or Xscratch1
// Trashes NZCV
template<std::size_t bitsize>
std::pair<oaknut::XReg, oaknut::XReg> InlinePageTableEmitVAddrLookup(
    oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr,
    const SharedLabel& fallback, bool near_fallback = false,
    UntrappedCachedReadLabels* out_cached = nullptr,
    const SharedLabel* misaligned_target = nullptr,
    bool check_misaligned = true,
    SharedLabel* out_marked_write = nullptr) {
    const std::size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_table_const_bits;
    const std::size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;

    if (check_misaligned) {
        EmitDetectMisalignedVAddr<bitsize>(code, ctx, Xaddr, misaligned_target ? *misaligned_target : fallback);
    }

    if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
        code.UBFX(Xscratch0, Xaddr, page_table_const_bits, valid_page_index_bits);
    } else {
        code.LSR(Xscratch0, Xaddr, page_table_const_bits);
        code.TST(Xaddr, u64(~u64(0)) << ctx.conf.page_table_address_space_bits);
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
        SharedLabel marked_target = fallback;
        if (out_cached != nullptr && ctx.conf.untrapped_read_bitmap_ptr != 0 &&
            *ctx.conf.page_table_marked_bit == 0) {
            out_cached->check_cached = GenSharedLabel();
            out_cached->resume_unmarked = GenSharedLabel();
            marked_target = out_cached->check_cached;
        } else if (out_marked_write != nullptr && *ctx.conf.page_table_marked_bit == 0) {
            *out_marked_write = GenSharedLabel();
            marked_target = *out_marked_write;
        }
        if (near_fallback || (ctx.block.instructions.size() < 256 && ctx.deferred_emits.size() < 128)) {
            code.TBNZ(Xscratch0, *ctx.conf.page_table_marked_bit, *marked_target);
        } else {
            code.TST(Xscratch0, 1ULL << *ctx.conf.page_table_marked_bit);
            code.B(NE, *marked_target);
        }
        if (out_cached != nullptr && out_cached->resume_unmarked) {
            code.l(*out_cached->resume_unmarked);
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
CodePtr EmitMemoryLdr(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xbase, oaknut::XReg Xoffset, bool ordered, bool extend32 = false, bool is_fpr = bitsize == 128) {
    const auto index_ext = extend32 ? oaknut::IndexExt::UXTW : oaknut::IndexExt::LSL;
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};

    CodePtr fastmem_location = code.xptr<CodePtr>();

    if (ordered) {
        ASSERT(!is_fpr || bitsize == 128);
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
    } else if (is_fpr) {
        fastmem_location = code.xptr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.LDR(oaknut::BReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 16:
            code.LDR(oaknut::HReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 32:
            code.LDR(oaknut::SReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 64:
            code.LDR(oaknut::DReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 128:
            code.LDR(oaknut::QReg{value_idx}, Xbase, Roffset, index_ext);
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
CodePtr EmitMemoryStr(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xbase, oaknut::XReg Xoffset, bool ordered, bool extend32 = false, bool is_fpr = bitsize == 128) {
    const auto index_ext = extend32 ? oaknut::IndexExt::UXTW : oaknut::IndexExt::LSL;
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};

    CodePtr fastmem_location;

    if (ordered) {
        ASSERT(!is_fpr || bitsize == 128);
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
            // The fault happens on the STR, not on the barrier before it.
            fastmem_location = code.xptr<CodePtr>();
            code.STR(oaknut::QReg{value_idx}, Xscratch0);
            code.DMB(oaknut::BarrierOp::ISH);
            break;
        default:
            UNREACHABLE();
        }
    } else if (is_fpr) {
        fastmem_location = code.xptr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.STR(oaknut::BReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 16:
            code.STR(oaknut::HReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 32:
            code.STR(oaknut::SReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 64:
            code.STR(oaknut::DReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 128:
            code.STR(oaknut::QReg{value_idx}, Xbase, Roffset, index_ext);
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

template<std::size_t bitsize>
CodePtr EmitMemoryLdrDisp(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xhost, s64 disp, bool is_fpr) {
    constexpr s64 byte_size = static_cast<s64>(bitsize / 8);
    const bool scaled = disp >= 0 && (disp & (byte_size - 1)) == 0 && (disp / byte_size) <= 4095;
    const int imm = static_cast<int>(disp);
    const CodePtr fastmem_location = code.xptr<CodePtr>();
    if (scaled) {
        if (is_fpr) {
            switch (bitsize) {
            case 8:
                code.LDR(oaknut::BReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.LDR(oaknut::HReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.LDR(oaknut::SReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.LDR(oaknut::DReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.LDR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        } else {
            switch (bitsize) {
            case 8:
                code.LDRB(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.LDRH(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.LDR(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.LDR(oaknut::XReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.LDR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        }
    } else {
        ASSERT(disp >= -256 && disp <= 255);
        if (is_fpr) {
            switch (bitsize) {
            case 8:
                code.LDUR(oaknut::BReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.LDUR(oaknut::HReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.LDUR(oaknut::SReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.LDUR(oaknut::DReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.LDUR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        } else {
            switch (bitsize) {
            case 8:
                code.LDURB(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.LDURH(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.LDUR(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.LDUR(oaknut::XReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.LDUR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        }
    }
    return fastmem_location;
}

template<std::size_t bitsize>
CodePtr EmitMemoryStrDisp(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xhost, s64 disp, bool is_fpr) {
    constexpr s64 byte_size = static_cast<s64>(bitsize / 8);
    const bool scaled = disp >= 0 && (disp & (byte_size - 1)) == 0 && (disp / byte_size) <= 4095;
    const int imm = static_cast<int>(disp);
    const CodePtr fastmem_location = code.xptr<CodePtr>();
    if (scaled) {
        if (is_fpr) {
            switch (bitsize) {
            case 8:
                code.STR(oaknut::BReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.STR(oaknut::HReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.STR(oaknut::SReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.STR(oaknut::DReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.STR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        } else {
            switch (bitsize) {
            case 8:
                code.STRB(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.STRH(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.STR(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.STR(oaknut::XReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.STR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        }
    } else {
        ASSERT(disp >= -256 && disp <= 255);
        if (is_fpr) {
            switch (bitsize) {
            case 8:
                code.STUR(oaknut::BReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.STUR(oaknut::HReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.STUR(oaknut::SReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.STUR(oaknut::DReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.STUR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
        } else {
            switch (bitsize) {
            case 8:
                code.STURB(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 16:
                code.STURH(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 32:
                code.STUR(oaknut::WReg{value_idx}, Xhost, imm);
                break;
            case 64:
                code.STUR(oaknut::XReg{value_idx}, Xhost, imm);
                break;
            case 128:
                code.STUR(oaknut::QReg{value_idx}, Xhost, imm);
                break;
            default:
                UNREACHABLE();
            }
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

LinkTarget PageTableReadLookupLinkTarget(std::size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::PageTableReadLookup8;
    case 16:
        return LinkTarget::PageTableReadLookup16;
    case 32:
        return LinkTarget::PageTableReadLookup32;
    case 64:
        return LinkTarget::PageTableReadLookup64;
    case 128:
        return LinkTarget::PageTableReadLookup128;
    }
    UNREACHABLE();
}

// [Xptr] with nothing added (the shared page table walk returned the host address).
template<std::size_t bitsize>
void EmitLoadAt(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xptr, bool ordered, bool is_fpr = bitsize == 128) {
    if (is_fpr && bitsize < 128) {
        ASSERT(!ordered);
        switch (bitsize) {
        case 8:
            code.LDR(oaknut::BReg{value_idx}, Xptr);
            return;
        case 16:
            code.LDR(oaknut::HReg{value_idx}, Xptr);
            return;
        case 32:
            code.LDR(oaknut::SReg{value_idx}, Xptr);
            return;
        case 64:
            code.LDR(oaknut::DReg{value_idx}, Xptr);
            return;
        default:
            UNREACHABLE();
        }
    }
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
void EmitStoreAt(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xptr, bool ordered, bool is_fpr = bitsize == 128) {
    if (is_fpr && bitsize < 128) {
        ASSERT(!ordered);
        switch (bitsize) {
        case 8:
            code.STR(oaknut::BReg{value_idx}, Xptr);
            return;
        case 16:
            code.STR(oaknut::HReg{value_idx}, Xptr);
            return;
        case 32:
            code.STR(oaknut::SReg{value_idx}, Xptr);
            return;
        case 64:
            code.STR(oaknut::DReg{value_idx}, Xptr);
            return;
        default:
            UNREACHABLE();
        }
    }
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

void EmitFallbackVAddrFromBase(oaknut::CodeGenerator& code, oaknut::XReg Xbase, s64 disp) {
    if (disp == 0) {
        code.MOV(Xscratch0, Xbase);
    } else if (disp > 0) {
        const u64 udisp = static_cast<u64>(disp);
        if (oaknut::AddSubImm::is_valid(udisp)) {
            code.ADD(Xscratch0, Xbase, udisp);
        } else {
            code.MOV(Xscratch0, udisp);
            code.ADD(Xscratch0, Xbase, Xscratch0);
        }
    } else {
        const u64 udisp = static_cast<u64>(-disp);
        ASSERT(oaknut::AddSubImm::is_valid(udisp));
        code.SUB(Xscratch0, Xbase, udisp);
    }
}

void EmitFallbackVAddrFromHost(oaknut::CodeGenerator& code, oaknut::XReg Xhost, s64 disp) {
    code.SUB(Xscratch0, Xhost, Xfastmem);
    if (disp > 0) {
        const u64 udisp = static_cast<u64>(disp);
        if (oaknut::AddSubImm::is_valid(udisp)) {
            code.ADD(Xscratch0, Xscratch0, udisp);
        } else {
            code.MOV(Xscratch2, udisp);
            code.ADD(Xscratch0, Xscratch0, Xscratch2);
        }
    } else if (disp < 0) {
        const u64 udisp = static_cast<u64>(-disp);
        ASSERT(oaknut::AddSubImm::is_valid(udisp));
        code.SUB(Xscratch0, Xscratch0, udisp);
    }
}

constexpr std::size_t kMaxInlinePageTableSaveOps = 12;
constexpr std::size_t kMaxCompactPageTableSaveOps = 8;
constexpr std::size_t kMaxFastmemFallbackSaveOps = 4;

constexpr std::size_t MaxPageTableSaveOps(const EmitContext& ctx) noexcept {
    return ctx.conf.page_table_lookup_calls ? kMaxCompactPageTableSaveOps : kMaxInlinePageTableSaveOps;
}

constexpr std::size_t SaveOpCount(RegisterList rl) noexcept {
    const std::size_t num_gprs = static_cast<std::size_t>(std::popcount(static_cast<u32>(rl)));
    const std::size_t num_fprs = static_cast<std::size_t>(std::popcount(static_cast<u32>(rl >> 32)));
    return (num_gprs + 1) / 2 + (num_fprs + 1) / 2;
}

template<std::size_t bitsize>
void EmitReadFallback(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst,
                      oaknut::XReg Xvaddr, int value_idx, [[maybe_unused]] bool is_fpr, bool ordered,
                      RegisterList live_caller_save, bool inline_save, oaknut::Label& end) {
    if (inline_save) {
        ASSERT((live_caller_save & (is_fpr ? ToRegList(oaknut::QReg{value_idx})
                                           : ToRegList(oaknut::XReg{value_idx}))) == 0);
        if (live_caller_save != 0) {
            ABI_PushRegisters(code, live_caller_save, 0);
        }
        if (Xvaddr.index() != 1) {
            code.MOV(X1, Xvaddr);
        }
        EmitRelocation(code, ctx, ReadMemoryLinkTarget(bitsize));
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        if constexpr (bitsize == 128) {
            if (value_idx != 0) {
                code.MOV(oaknut::QReg{value_idx}.B16(), Q0.B16());
            }
        } else if (is_fpr) {
            if constexpr (bitsize <= 32) {
                code.FMOV(oaknut::SReg{value_idx}, W0);
            } else {
                code.FMOV(oaknut::DReg{value_idx}, X0);
            }
        } else if (value_idx != 0) {
            code.MOV(oaknut::XReg{value_idx}, X0);
        }
        if (live_caller_save != 0) {
            ABI_PopRegisters(code, live_caller_save, 0);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
        code.B(end);
        return;
    }

    if (Xvaddr.index() != Xscratch0.index()) {
        code.MOV(Xscratch0, Xvaddr);
    }
    EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    if constexpr (bitsize == 128) {
        code.MOV(oaknut::QReg{value_idx}.B16(), Q0.B16());
    } else if (is_fpr) {
        if constexpr (bitsize <= 32) {
            code.FMOV(oaknut::SReg{value_idx}, Wscratch0);
        } else {
            code.FMOV(oaknut::DReg{value_idx}, Xscratch0);
        }
    } else {
        code.MOV(oaknut::XReg{value_idx}, Xscratch0);
    }
    ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
    code.B(end);
}

template<std::size_t bitsize>
void EmitWriteFallback(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst,
                       oaknut::XReg Xvaddr, int value_idx, [[maybe_unused]] bool is_fpr, bool ordered,
                       RegisterList live_caller_save, bool inline_save, oaknut::Label& end) {
    if (inline_save) {
        if constexpr (bitsize == 128) {
            ASSERT((live_caller_save & ToRegList(Q0)) == 0);
            if (value_idx != 0) {
                code.MOV(Q0.B16(), oaknut::QReg{value_idx}.B16());
            }
            if (live_caller_save != 0) {
                ABI_PushRegisters(code, live_caller_save, 0);
            }
            if (Xvaddr.index() != 1) {
                code.MOV(X1, Xvaddr);
            }
        } else if (is_fpr) {
            if (live_caller_save != 0) {
                ABI_PushRegisters(code, live_caller_save, 0);
            }
            if (Xvaddr.index() != 1) {
                code.MOV(X1, Xvaddr);
            }
            if constexpr (bitsize <= 32) {
                code.FMOV(W2, oaknut::SReg{value_idx});
            } else {
                code.FMOV(X2, oaknut::DReg{value_idx});
            }
        } else {
            if (live_caller_save != 0) {
                ABI_PushRegisters(code, live_caller_save, 0);
            }
            const int addr_idx = Xvaddr.index();
            if (addr_idx == 2 && value_idx == 1) {
                code.MOV(Xscratch0, X2);
                code.MOV(X2, X1);
                code.MOV(X1, Xscratch0);
            } else if (value_idx == 1) {
                code.MOV(X2, X1);
                if (addr_idx != 1) {
                    code.MOV(X1, Xvaddr);
                }
            } else {
                if (addr_idx != 1) {
                    code.MOV(X1, Xvaddr);
                }
                if (value_idx != 2) {
                    code.MOV(X2, oaknut::XReg{value_idx});
                }
            }
        }
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        EmitRelocation(code, ctx, WriteMemoryLinkTarget(bitsize));
        if (ordered) {
            code.DMB(oaknut::BarrierOp::ISH);
        }
        if (live_caller_save != 0) {
            ABI_PopRegisters(code, live_caller_save, 0);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
        code.B(end);
        return;
    }

    if (Xvaddr.index() != Xscratch0.index()) {
        code.MOV(Xscratch0, Xvaddr);
    }
    if constexpr (bitsize == 128) {
        code.MOV(Q0.B16(), oaknut::QReg{value_idx}.B16());
    } else if (is_fpr) {
        if constexpr (bitsize <= 32) {
            code.FMOV(Wscratch1, oaknut::SReg{value_idx});
        } else {
            code.FMOV(Xscratch1, oaknut::DReg{value_idx});
        }
    } else {
        code.MOV(Xscratch1, oaknut::XReg{value_idx});
    }
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
    code.B(end);
}

template<std::size_t bitsize>
void InlinePageTableEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const IR::AccType acc_type = args[2].GetImmediateAccType();
    const bool ordered = IsOrdered(acc_type);
    bool use_fpr = bitsize == 128 || (FprScalarForwardingOn() && !ordered && acc_type == IR::AccType::VEC);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();

    const bool use_cross_page = CanInlineCrossPage<bitsize>(ctx, ordered);
    int value_idx = 0;
    RegisterList live_caller_save = 0;
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();
    SharedLabel cross_page = use_cross_page ? GenSharedLabel() : SharedLabel{};
    UntrappedCachedReadLabels pt_cached;

    const auto emit_body = [&](int idx) {
        value_idx = idx;
        if (ctx.conf.page_table_lookup_calls) {
            code.MOV(Xscratch0, *Xaddr);
            EmitRelocation(code, ctx, PageTableReadLookupLinkTarget(bitsize));
            code.CBZ(Xscratch0, *fallback);
            EmitLoadAt<bitsize>(code, idx, Xscratch0, ordered, use_fpr);
        } else {
            const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(
                code, ctx, Xaddr, fallback, false, &pt_cached, use_cross_page ? &cross_page : nullptr);
            EmitMemoryLdr<bitsize>(code, idx, Xbase, Xoffset, ordered, false, use_fpr);
        }
    };

    if constexpr (bitsize == 128) {
        auto Rvalue = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
        emit_body(Rvalue->index());
    } else if (use_fpr) {
        auto Rvalue = ctx.reg_alloc.WriteVec<bitsize>(inst);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
        emit_body(Rvalue->index());
    } else {
        auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
        emit_body(Rvalue->index());
    }
    const bool inline_save = SaveOpCount(live_caller_save) <= MaxPageTableSaveOps(ctx);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, use_cross_page, cross_page, pt_cached, fallback, end] {
        EmitUntrappedCachedReadCheck(code, ctx, Xaddr, pt_cached, fallback);
        if (use_cross_page) {
            code.l(*cross_page);
            EmitCrossPageContiguousLookup<bitsize>(code, ctx, Xaddr, fallback);
            EmitMemoryLdr<bitsize>(code, value_idx, Xscratch0, Xaddr, ordered, false, use_fpr);
            code.B(*end);
        }
        code.l(*fallback);
        EmitReadFallback<bitsize>(code, ctx, inst, Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, *end);
    });

    code.l(*end);
}

constexpr std::size_t HotWriteSizeIndex(std::size_t bitsize) noexcept {
    switch (bitsize) {
    case 8:
        return 0;
    case 16:
        return 1;
    case 32:
        return 2;
    case 64:
        return 3;
    case 128:
        return 4;
    }
    UNREACHABLE();
}

template<std::size_t bitsize>
bool CanEmitHotCachedWrite(const EmitContext& ctx, bool ordered) noexcept {
    return !ordered &&
           ctx.conf.page_table_marked_bit &&
           *ctx.conf.page_table_marked_bit == 0 &&
           ctx.conf.page_table_address_space_bits > page_table_const_bits &&
           (bitsize == 8 || (ctx.conf.detect_misaligned_access_via_page_table & bitsize) != 0);
}

template<std::size_t bitsize>
void EmitHotCachedWriteCheck(oaknut::CodeGenerator& code, const EmitContext& ctx,
                             oaknut::XReg Xaddr, int value_idx, [[maybe_unused]] bool use_fpr,
                             RegisterList live_caller_save, const JitHotWriteCallbacks& hot_write,
                             const SharedLabel& check_label, const SharedLabel& lookup_miss_label,
                             const SharedLabel& fallback, const SharedLabel& end,
                             bool fallthrough_to_fallback) {
    const std::uintptr_t fn_ptr = hot_write.write_fns[HotWriteSizeIndex(bitsize)];
    if (fn_ptr == 0 || (!check_label && !lookup_miss_label)) {
        return;
    }

    if (lookup_miss_label) {
        const std::size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_table_const_bits;
        const std::size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;
        code.l(*lookup_miss_label);
        if constexpr (bitsize > 8) {
            if (!ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary) {
                code.TST(Xaddr, u64(bitsize / 8 - 1));
                code.B(NE, *fallback);
            } else {
                code.AND(Wscratch0, Xaddr.toW(), u32(page_table_const_mask));
                code.CMP(Wscratch0, u32(page_table_const_size - bitsize / 8));
                code.B(HI, *fallback);
            }
        }
        if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
            code.UBFX(Xscratch0, Xaddr, page_table_const_bits, valid_page_index_bits);
        } else {
            code.LSR(Xscratch0, Xaddr, page_table_const_bits);
            code.TST(Xaddr, u64(~u64(0)) << ctx.conf.page_table_address_space_bits);
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
        code.TBZ(Xscratch0, 0, *fallback);
    }

    if (check_label) {
        code.l(*check_label);
    }
    code.TBZ(Xscratch0, 1, *fallback);
    code.TBZ(Xscratch0, 2, *fallback);

    code.UBFX(Xscratch2, Xaddr, page_table_const_bits, 48 - page_table_const_bits);
    code.EOR(Xscratch0, Xscratch2, Xscratch2, LSR, 6);
    code.AND(Xscratch0, Xscratch0, 63);
    code.MOV(Xscratch1, hot_write.writes_base);
    code.ADD(Xscratch1, Xscratch1, Xscratch0, LSL, 7);
    code.LDR(Xscratch0, Xscratch1, 0);
    code.LSL(Xscratch2, Xscratch2, page_table_const_bits);
    code.CMP(Xscratch0, Xscratch2);
    code.B(NE, *fallback);
    code.LDRB(Wscratch0, Xscratch1, 64);
    code.CBZ(Wscratch0, *fallback);

    if constexpr (bitsize == 128) {
        ASSERT((live_caller_save & ToRegList(Q0)) == 0);
    }
    const RegisterList value_reg = (bitsize == 128 || use_fpr)
                                       ? ToRegList(oaknut::QReg{value_idx})
                                       : ToRegList(oaknut::XReg{value_idx});
    const RegisterList caller_mask = hot_write.preserve_all_abi
                                         ? RegisterList{0x01FFULL}
                                         : (ABI_CALLER_SAVE & ~(RegisterList{1} << 30));
    const RegisterList save_regs = (live_caller_save | ToRegList(Xaddr) | value_reg) & caller_mask;
    if (save_regs != 0) {
        ABI_PushRegisters(code, save_regs, 0);
    }

    if constexpr (bitsize == 128) {
        if (value_idx != 0) {
            code.MOV(Q0.B16(), oaknut::QReg{value_idx}.B16());
        }
        if (Xaddr.index() != 2) {
            code.MOV(X2, Xaddr);
        }
    } else if (use_fpr) {
        if (Xaddr.index() != 2) {
            code.MOV(X2, Xaddr);
        }
        if constexpr (bitsize <= 32) {
            code.FMOV(W3, oaknut::SReg{value_idx});
        } else {
            code.FMOV(X3, oaknut::DReg{value_idx});
        }
    } else {
        const int addr_idx = Xaddr.index();
        if (addr_idx == 3 && value_idx == 2) {
            code.MOV(Xscratch0, X3);
            code.MOV(X3, X2);
            code.MOV(X2, Xscratch0);
        } else if (value_idx == 2) {
            code.MOV(X3, X2);
            if (addr_idx != 2) {
                code.MOV(X2, Xaddr);
            }
        } else {
            if (addr_idx != 2) {
                code.MOV(X2, Xaddr);
            }
            if (value_idx != 3) {
                code.MOV(X3, oaknut::XReg{value_idx});
            }
        }
    }
    code.MOV(X1, Xscratch1);
    code.MOV(X0, hot_write.ctx_ptr);
    code.MOV(Xscratch0, fn_ptr);
    code.BLR(Xscratch0);
    code.MOV(Wscratch0, W0);
    if (save_regs != 0) {
        ABI_PopRegisters(code, save_regs, 0);
    }
    code.CBNZ(Wscratch0, *end);
    if (!fallthrough_to_fallback) {
        code.B(*fallback);
    }
}

template<std::size_t bitsize>
void InlinePageTableEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    bool use_fpr = bitsize == 128 || (FprScalarForwardingOn() && !ordered && args[2].IsInFpr(ctx.reg_alloc));
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();

    const bool use_cross_page = CanInlineCrossPage<bitsize>(ctx, ordered);
    const auto hot_write = CanEmitHotCachedWrite<bitsize>(ctx, ordered)
                               ? FindJitHotWrite(ctx.conf.tpidrro_el0)
                               : std::nullopt;
    int value_idx = 0;
    RegisterList live_caller_save = 0;
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();
    SharedLabel cross_page = use_cross_page ? GenSharedLabel() : SharedLabel{};
    SharedLabel marked_write;
    SharedLabel lookup_miss = (hot_write && ctx.conf.page_table_lookup_calls) ? GenSharedLabel() : SharedLabel{};

    const auto emit_body = [&](int idx) {
        value_idx = idx;
        if (ctx.conf.page_table_lookup_calls) {
            code.MOV(Xscratch0, *Xaddr);
            EmitRelocation(code, ctx, PageTableLookupLinkTarget(bitsize));
            code.CBZ(Xscratch0, lookup_miss ? *lookup_miss : *fallback);
            EmitStoreAt<bitsize>(code, idx, Xscratch0, ordered, use_fpr);
        } else {
            const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(
                code, ctx, Xaddr, fallback, false, nullptr, use_cross_page ? &cross_page : nullptr,
                true, hot_write ? &marked_write : nullptr);
            EmitMemoryStr<bitsize>(code, idx, Xbase, Xoffset, ordered, false, use_fpr);
        }
    };

    if constexpr (bitsize == 128) {
        auto Rvalue = ctx.reg_alloc.ReadQ(args[2]);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
        emit_body(Rvalue->index());
    } else if (use_fpr) {
        auto Rvalue = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
        emit_body(Rvalue->index());
    } else {
        auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
        emit_body(Rvalue->index());
    }
    const bool inline_save = SaveOpCount(live_caller_save) <= MaxPageTableSaveOps(ctx);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, use_cross_page, cross_page, hot_write, marked_write, lookup_miss, fallback, end] {
        if (use_cross_page) {
            code.l(*cross_page);
            EmitCrossPageContiguousLookup<bitsize>(code, ctx, Xaddr, fallback);
            EmitMemoryStr<bitsize>(code, value_idx, Xscratch0, Xaddr, ordered, false, use_fpr);
            code.B(*end);
        }
        if (hot_write) {
            EmitHotCachedWriteCheck<bitsize>(code, ctx, Xaddr, value_idx, use_fpr, live_caller_save,
                                             *hot_write, marked_write, lookup_miss, fallback, end, true);
        }
        code.l(*fallback);
        EmitWriteFallback<bitsize>(code, ctx, inst, Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, *end);
    });

    code.l(*end);
}

std::optional<DoNotFastmemMarker> ShouldFastmem(EmitContext& ctx, IR::Inst* inst, bool is_load = false) {
    if (!ctx.conf.fastmem_pointer || !ctx.fastmem.SupportsFastmem()) {
        return std::nullopt;
    }

    const auto marker = std::make_tuple(ctx.block.Location(), inst->GetName());
    if (ctx.fastmem.ShouldFastmem(marker, is_load)) {
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
    const IR::AccType acc_type = args[2].GetImmediateAccType();
    const bool ordered = IsOrdered(acc_type);
    bool use_fpr = bitsize == 128 || (FprScalarForwardingOn() && !ordered && acc_type == IR::AccType::VEC);
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    if (const auto* const folded = ctx.ebi.FindFoldedAccess(inst)) {
        const std::size_t base_uses = ctx.ebi.BaseAccessCount(folded->base_inst);
        const s64 disp = folded->disp;
        if (base_uses >= 2) {
            ctx.fpsr.Spill();
            ctx.reg_alloc.SpillFlags();
            oaknut::XReg Xhost{0};
            int value_idx = 0;
            RegisterList live_caller_save = 0;
            CodePtr fastmem_location = nullptr;
            if (ctx.reg_alloc.HasFastmemBase(folded->base_inst)) {
                auto host_handle = ctx.reg_alloc.AcquireExistingFastmemBase(folded->base_inst);
                if constexpr (bitsize == 128) {
                    auto Rvalue = ctx.reg_alloc.WriteQ(inst);
                    RegAlloc::Realize(Rvalue);
                    Xhost = ctx.reg_alloc.RealizeExistingFastmemBase(host_handle);
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else if (use_fpr) {
                    auto Rvalue = ctx.reg_alloc.WriteVec<bitsize>(inst);
                    RegAlloc::Realize(Rvalue);
                    Xhost = ctx.reg_alloc.RealizeExistingFastmemBase(host_handle);
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else {
                    auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
                    RegAlloc::Realize(Rvalue);
                    Xhost = ctx.reg_alloc.RealizeExistingFastmemBase(host_handle);
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xhost, disp, false);
                }
            } else {
                auto Xbase = ctx.reg_alloc.ReadX(args[1]);
                if constexpr (bitsize == 128) {
                    auto Rvalue = ctx.reg_alloc.WriteQ(inst);
                    RegAlloc::Realize(Xbase, Rvalue);
                    auto host_handle = ctx.reg_alloc.CreateFastmemBase(folded->base_inst, base_uses, Xfastmem, *Xbase);
                    Xhost = host_handle.Reg();
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else if (use_fpr) {
                    auto Rvalue = ctx.reg_alloc.WriteVec<bitsize>(inst);
                    RegAlloc::Realize(Xbase, Rvalue);
                    auto host_handle = ctx.reg_alloc.CreateFastmemBase(folded->base_inst, base_uses, Xfastmem, *Xbase);
                    Xhost = host_handle.Reg();
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else {
                    auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
                    RegAlloc::Realize(Xbase, Rvalue);
                    auto host_handle = ctx.reg_alloc.CreateFastmemBase(folded->base_inst, base_uses, Xfastmem, *Xbase);
                    Xhost = host_handle.Reg();
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xhost, disp, false);
                }
            }
            const bool inline_save = SaveOpCount(live_caller_save) <= kMaxFastmemFallbackSaveOps;
            ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xhost, disp, value_idx, use_fpr, live_caller_save, inline_save, fallback, end, fastmem_location] {
                ctx.ebi.fastmem_patch_info.emplace(
                    fastmem_location - ctx.ebi.entry_point,
                    FastmemPatchInfo{
                        .marker = marker,
                        .fc = FakeCall{.call_pc = std::bit_cast<u64>(code.xptr<void*>())},
                        .recompile = ctx.conf.recompile_on_fastmem_failure,
                        .load = true,
                    });
                code.l(*fallback);
                EmitFallbackVAddrFromHost(code, Xhost, disp);
                EmitReadFallback<bitsize>(code, ctx, inst, Xscratch0, value_idx, use_fpr, false, live_caller_save, inline_save, *end);
            });
            code.l(*end);
            return;
        }
        if (disp != 0) {
            auto Xbase = ctx.reg_alloc.ReadX(args[1]);
            ctx.fpsr.Spill();
            ctx.reg_alloc.SpillFlags();
            int value_idx = 0;
            RegisterList live_caller_save = 0;
            CodePtr fastmem_location = nullptr;
            if constexpr (bitsize == 128) {
                auto Rvalue = ctx.reg_alloc.WriteQ(inst);
                RegAlloc::Realize(Xbase, Rvalue);
                live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                value_idx = Rvalue->index();
                code.ADD(Xscratch0, Xfastmem, *Xbase);
                fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xscratch0, disp, true);
            } else if (use_fpr) {
                auto Rvalue = ctx.reg_alloc.WriteVec<bitsize>(inst);
                RegAlloc::Realize(Xbase, Rvalue);
                live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                value_idx = Rvalue->index();
                code.ADD(Xscratch0, Xfastmem, *Xbase);
                fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xscratch0, disp, true);
            } else {
                auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
                RegAlloc::Realize(Xbase, Rvalue);
                live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
                value_idx = Rvalue->index();
                code.ADD(Xscratch0, Xfastmem, *Xbase);
                fastmem_location = EmitMemoryLdrDisp<bitsize>(code, value_idx, Xscratch0, disp, false);
            }
            const bool inline_save = SaveOpCount(live_caller_save) <= kMaxFastmemFallbackSaveOps;
            ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xbase = *Xbase, disp, value_idx, use_fpr, live_caller_save, inline_save, fallback, end, fastmem_location] {
                ctx.ebi.fastmem_patch_info.emplace(
                    fastmem_location - ctx.ebi.entry_point,
                    FastmemPatchInfo{
                        .marker = marker,
                        .fc = FakeCall{.call_pc = std::bit_cast<u64>(code.xptr<void*>())},
                        .recompile = ctx.conf.recompile_on_fastmem_failure,
                        .load = true,
                    });
                code.l(*fallback);
                EmitFallbackVAddrFromBase(code, Xbase, disp);
                EmitReadFallback<bitsize>(code, ctx, inst, Xscratch0, value_idx, use_fpr, false, live_caller_save, inline_save, *end);
            });
            code.l(*end);
            return;
        }
    }

    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();

    const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
    SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;
    int value_idx = 0;
    RegisterList live_caller_save = 0;
    CodePtr fastmem_location = nullptr;

    if constexpr (bitsize == 128) {
        auto Rvalue = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
        value_idx = Rvalue->index();
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        fastmem_location = EmitMemoryLdr<bitsize>(code, value_idx, Xbase, Xoffset, ordered, ShouldExt32(ctx), true);
    } else if (use_fpr) {
        auto Rvalue = ctx.reg_alloc.WriteVec<bitsize>(inst);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
        value_idx = Rvalue->index();
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        fastmem_location = EmitMemoryLdr<bitsize>(code, value_idx, Xbase, Xoffset, ordered, ShouldExt32(ctx), true);
    } else {
        auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue));
        value_idx = Rvalue->index();
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        fastmem_location = EmitMemoryLdr<bitsize>(code, value_idx, Xbase, Xoffset, ordered, ShouldExt32(ctx), false);
    }
    const bool inline_save = SaveOpCount(live_caller_save) <=
                             (use_pt_fallback ? MaxPageTableSaveOps(ctx) : kMaxFastmemFallbackSaveOps);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
        if (use_pt_fallback) {
            code.l(*bounds_miss);
            const bool use_cross_page = CanInlineCrossPage<bitsize>(ctx, ordered);
            SharedLabel cross_page = use_cross_page ? GenSharedLabel() : SharedLabel{};
            UntrappedCachedReadLabels pt_cached;
            const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
                code, ctx, Xaddr, fallback, true, &pt_cached, use_cross_page ? &cross_page : nullptr);
            EmitMemoryLdr<bitsize>(code, value_idx, Xpt_base, Xpt_offset, ordered, false, use_fpr);
            code.B(*end);
            EmitUntrappedCachedReadCheck(code, ctx, Xaddr, pt_cached, fallback);
            if (use_cross_page) {
                code.l(*cross_page);
                EmitCrossPageContiguousLookup<bitsize>(code, ctx, Xaddr, fallback);
                EmitMemoryLdr<bitsize>(code, value_idx, Xscratch0, Xaddr, ordered, false, use_fpr);
                code.B(*end);
            }
        }

        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = std::bit_cast<u64>(code.xptr<void*>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
                .load = true,
            });

        code.l(*fallback);
        EmitReadFallback<bitsize>(code, ctx, inst, Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, *end);
    });

    code.l(*end);
}

template<std::size_t bitsize>
void FastmemEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    bool use_fpr = bitsize == 128 || (FprScalarForwardingOn() && !ordered && args[2].IsInFpr(ctx.reg_alloc));
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    if (const auto* const folded = ctx.ebi.FindFoldedAccess(inst)) {
        const std::size_t base_uses = ctx.ebi.BaseAccessCount(folded->base_inst);
        const s64 disp = folded->disp;
        if (base_uses >= 2) {
            ctx.fpsr.Spill();
            ctx.reg_alloc.SpillFlags();
            oaknut::XReg Xhost{0};
            int value_idx = 0;
            RegisterList live_caller_save = 0;
            CodePtr fastmem_location = nullptr;
            if (ctx.reg_alloc.HasFastmemBase(folded->base_inst)) {
                auto host_handle = ctx.reg_alloc.AcquireExistingFastmemBase(folded->base_inst);
                if constexpr (bitsize == 128) {
                    auto Rvalue = ctx.reg_alloc.ReadQ(args[2]);
                    RegAlloc::Realize(Rvalue);
                    Xhost = ctx.reg_alloc.RealizeExistingFastmemBase(host_handle);
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else if (use_fpr) {
                    auto Rvalue = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
                    RegAlloc::Realize(Rvalue);
                    Xhost = ctx.reg_alloc.RealizeExistingFastmemBase(host_handle);
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else {
                    auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
                    RegAlloc::Realize(Rvalue);
                    Xhost = ctx.reg_alloc.RealizeExistingFastmemBase(host_handle);
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xhost, disp, false);
                }
            } else {
                auto Xbase = ctx.reg_alloc.ReadX(args[1]);
                if constexpr (bitsize == 128) {
                    auto Rvalue = ctx.reg_alloc.ReadQ(args[2]);
                    RegAlloc::Realize(Xbase, Rvalue);
                    auto host_handle = ctx.reg_alloc.CreateFastmemBase(folded->base_inst, base_uses, Xfastmem, *Xbase);
                    Xhost = host_handle.Reg();
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else if (use_fpr) {
                    auto Rvalue = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
                    RegAlloc::Realize(Xbase, Rvalue);
                    auto host_handle = ctx.reg_alloc.CreateFastmemBase(folded->base_inst, base_uses, Xfastmem, *Xbase);
                    Xhost = host_handle.Reg();
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xhost, disp, true);
                } else {
                    auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
                    RegAlloc::Realize(Xbase, Rvalue);
                    auto host_handle = ctx.reg_alloc.CreateFastmemBase(folded->base_inst, base_uses, Xfastmem, *Xbase);
                    Xhost = host_handle.Reg();
                    live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                    value_idx = Rvalue->index();
                    fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xhost, disp, false);
                }
            }
            const bool inline_save = SaveOpCount(live_caller_save) <= kMaxFastmemFallbackSaveOps;
            ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xhost, disp, value_idx, use_fpr, live_caller_save, inline_save, fallback, end, fastmem_location] {
                ctx.ebi.fastmem_patch_info.emplace(
                    fastmem_location - ctx.ebi.entry_point,
                    FastmemPatchInfo{
                        .marker = marker,
                        .fc = FakeCall{.call_pc = std::bit_cast<u64>(code.xptr<void*>())},
                        .recompile = ctx.conf.recompile_on_fastmem_failure,
                    });
                code.l(*fallback);
                EmitFallbackVAddrFromHost(code, Xhost, disp);
                EmitWriteFallback<bitsize>(code, ctx, inst, Xscratch0, value_idx, use_fpr, false, live_caller_save, inline_save, *end);
            });
            code.l(*end);
            return;
        }
        if (disp != 0) {
            auto Xbase = ctx.reg_alloc.ReadX(args[1]);
            ctx.fpsr.Spill();
            ctx.reg_alloc.SpillFlags();
            int value_idx = 0;
            RegisterList live_caller_save = 0;
            CodePtr fastmem_location = nullptr;
            if constexpr (bitsize == 128) {
                auto Rvalue = ctx.reg_alloc.ReadQ(args[2]);
                RegAlloc::Realize(Xbase, Rvalue);
                live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                value_idx = Rvalue->index();
                code.ADD(Xscratch0, Xfastmem, *Xbase);
                fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xscratch0, disp, true);
            } else if (use_fpr) {
                auto Rvalue = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
                RegAlloc::Realize(Xbase, Rvalue);
                live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                value_idx = Rvalue->index();
                code.ADD(Xscratch0, Xfastmem, *Xbase);
                fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xscratch0, disp, true);
            } else {
                auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
                RegAlloc::Realize(Xbase, Rvalue);
                live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
                value_idx = Rvalue->index();
                code.ADD(Xscratch0, Xfastmem, *Xbase);
                fastmem_location = EmitMemoryStrDisp<bitsize>(code, value_idx, Xscratch0, disp, false);
            }
            const bool inline_save = SaveOpCount(live_caller_save) <= kMaxFastmemFallbackSaveOps;
            ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xbase = *Xbase, disp, value_idx, use_fpr, live_caller_save, inline_save, fallback, end, fastmem_location] {
                ctx.ebi.fastmem_patch_info.emplace(
                    fastmem_location - ctx.ebi.entry_point,
                    FastmemPatchInfo{
                        .marker = marker,
                        .fc = FakeCall{.call_pc = std::bit_cast<u64>(code.xptr<void*>())},
                        .recompile = ctx.conf.recompile_on_fastmem_failure,
                    });
                code.l(*fallback);
                EmitFallbackVAddrFromBase(code, Xbase, disp);
                EmitWriteFallback<bitsize>(code, ctx, inst, Xscratch0, value_idx, use_fpr, false, live_caller_save, inline_save, *end);
            });
            code.l(*end);
            return;
        }
    }

    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();

    const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
    SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;
    int value_idx = 0;
    RegisterList live_caller_save = 0;
    CodePtr fastmem_location = nullptr;

    if constexpr (bitsize == 128) {
        auto Rvalue = ctx.reg_alloc.ReadQ(args[2]);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
        value_idx = Rvalue->index();
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        fastmem_location = EmitMemoryStr<bitsize>(code, value_idx, Xbase, Xoffset, ordered, ShouldExt32(ctx), true);
    } else if (use_fpr) {
        auto Rvalue = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
        value_idx = Rvalue->index();
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        fastmem_location = EmitMemoryStr<bitsize>(code, value_idx, Xbase, Xoffset, ordered, ShouldExt32(ctx), true);
    } else {
        auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        RegAlloc::Realize(Xaddr, Rvalue);
        live_caller_save = ctx.reg_alloc.LiveCallerSaveRegisters();
        value_idx = Rvalue->index();
        const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
        fastmem_location = EmitMemoryStr<bitsize>(code, value_idx, Xbase, Xoffset, ordered, ShouldExt32(ctx), false);
    }
    const bool inline_save = SaveOpCount(live_caller_save) <=
                             (use_pt_fallback ? MaxPageTableSaveOps(ctx) : kMaxFastmemFallbackSaveOps);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
        if (use_pt_fallback) {
            code.l(*bounds_miss);
            const bool use_cross_page = CanInlineCrossPage<bitsize>(ctx, ordered);
            const auto hot_write = CanEmitHotCachedWrite<bitsize>(ctx, ordered)
                                       ? FindJitHotWrite(ctx.conf.tpidrro_el0)
                                       : std::nullopt;
            SharedLabel cross_page = use_cross_page ? GenSharedLabel() : SharedLabel{};
            SharedLabel marked_write;
            const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
                code, ctx, Xaddr, fallback, true, nullptr, use_cross_page ? &cross_page : nullptr,
                true, hot_write ? &marked_write : nullptr);
            EmitMemoryStr<bitsize>(code, value_idx, Xpt_base, Xpt_offset, ordered, false, use_fpr);
            code.B(*end);
            if (hot_write) {
                EmitHotCachedWriteCheck<bitsize>(code, ctx, Xaddr, value_idx, use_fpr, live_caller_save,
                                                 *hot_write, marked_write, SharedLabel{}, fallback, end, false);
            }
            if (use_cross_page) {
                code.l(*cross_page);
                EmitCrossPageContiguousLookup<bitsize>(code, ctx, Xaddr, fallback);
                EmitMemoryStr<bitsize>(code, value_idx, Xscratch0, Xaddr, ordered, false, use_fpr);
                code.B(*end);
            }
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
        EmitWriteFallback<bitsize>(code, ctx, inst, Xaddr, value_idx, use_fpr, ordered, live_caller_save, inline_save, *end);
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
void* EmitPageTableLookupThunk(oaknut::CodeGenerator& code, const EmitConfig& conf, std::size_t bitsize,
                               bool allow_untrapped_cached_read) {
    void* const entry = code.xptr<void*>();
    oaknut::Label miss, cross_page, check_cached, resume_unmarked, pop_and_resume, pop_and_miss;
    const std::size_t valid_page_index_bits = conf.page_table_address_space_bits - page_table_const_bits;
    const std::size_t unused_top_bits = 64 - conf.page_table_address_space_bits;
    const bool use_cached_read = allow_untrapped_cached_read && conf.untrapped_read_bitmap_ptr != 0 &&
                                 conf.page_table_marked_bit && *conf.page_table_marked_bit == 0;
    const bool use_cross_page =
        bitsize > 8 &&
        (conf.detect_misaligned_access_via_page_table & bitsize) != 0 &&
        conf.only_detect_misalignment_via_page_table_on_page_boundary &&
        conf.absolute_offset_page_table &&
        conf.page_table_log2_stride == 3 &&
        !conf.silently_mirror_page_table &&
        conf.page_table_address_space_bits > page_table_const_bits &&
        conf.page_table_address_space_bits < 64;

    if (bitsize != 8 && (conf.detect_misaligned_access_via_page_table & bitsize) != 0) {
        if (!conf.only_detect_misalignment_via_page_table_on_page_boundary) {
            code.TST(Xscratch0, u64(bitsize / 8 - 1));
            code.B(NE, miss);
        } else {
            code.AND(Wscratch1, Wscratch0, u32(page_table_const_mask));
            code.CMP(Wscratch1, u32(page_table_const_size - bitsize / 8));
            code.B(HI, use_cross_page ? cross_page : miss);
        }
    }
    if (conf.silently_mirror_page_table || unused_top_bits == 0) {
        code.UBFX(Xscratch1, Xscratch0, page_table_const_bits, valid_page_index_bits);
    } else {
        code.LSR(Xscratch1, Xscratch0, page_table_const_bits);
        code.TST(Xscratch0, u64(~u64(0)) << conf.page_table_address_space_bits);
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
        if (use_cached_read) {
            code.TBNZ(Xscratch1, *conf.page_table_marked_bit, check_cached);
            code.l(resume_unmarked);
        } else {
            code.TBNZ(Xscratch1, *conf.page_table_marked_bit, miss);
        }
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

    if (use_cached_read) {
        code.l(check_cached);
        code.TBZ(Xscratch1, 1, miss);
        code.TBZ(Xscratch1, 2, miss);
        code.STP(Xscratch2, Xscratch1, SP, PRE_INDEXED, -16);
        code.MOV(Xscratch2, conf.untrapped_read_bitmap_ptr);
        code.LDR(Xscratch2, Xscratch2);
        code.CBZ(Xscratch2, pop_and_miss);
        code.CMP(Xscratch2, 1);
        code.B(EQ, pop_and_resume);
        if (valid_page_index_bits <= 27) {
            if (conf.silently_mirror_page_table || unused_top_bits == 0) {
                if (valid_page_index_bits > 6) {
                    code.UBFX(Xscratch1, Xscratch0, page_table_const_bits + 6, valid_page_index_bits - 6);
                } else {
                    code.MOV(Xscratch1, XZR);
                }
            } else {
                code.LSR(Xscratch1, Xscratch0, page_table_const_bits + 6);
            }
        } else {
            if (conf.silently_mirror_page_table || unused_top_bits == 0) {
                code.UBFX(Xscratch1, Xscratch0, page_table_const_bits, valid_page_index_bits);
            } else {
                code.LSR(Xscratch1, Xscratch0, page_table_const_bits);
            }
            code.TST(Xscratch1, ~((u64{1} << 27) - 1));
            code.B(NE, pop_and_resume);
            code.LSR(Xscratch1, Xscratch1, 6);
        }
        code.LDR(Xscratch2, Xscratch2, Xscratch1, oaknut::IndexExt::LSL, 3);
        code.UBFX(Xscratch1, Xscratch0, page_table_const_bits, 6);
        code.LSR(Xscratch2, Xscratch2, Xscratch1);
        code.TBNZ(Xscratch2, 0, pop_and_miss);
        code.l(pop_and_resume);
        code.LDP(Xscratch2, Xscratch1, SP, POST_INDEXED, 16);
        code.B(resume_unmarked);
        code.l(pop_and_miss);
        code.LDP(Xscratch2, Xscratch1, SP, POST_INDEXED, 16);
    }

    if (use_cross_page) {
        oaknut::Label cross_pop_miss;
        code.l(cross_page);
        code.ADDS(Xscratch1, Xscratch0, u64(bitsize / 8 - 1));
        code.B(CS, miss);
        code.LSR(Xscratch1, Xscratch1, conf.page_table_address_space_bits);
        code.CBNZ(Xscratch1, miss);

        code.STR(Xscratch2, SP, PRE_INDEXED, -16);
        code.LSR(Xscratch1, Xscratch0, page_table_const_bits);
        code.ADD(Xscratch1, Xpagetable, Xscratch1, LSL, 3);
        code.LDP(Xscratch1, Xscratch2, Xscratch1, 0);
        if (conf.page_table_marked_bit) {
            code.TBNZ(Xscratch1, *conf.page_table_marked_bit, cross_pop_miss);
            code.TBNZ(Xscratch2, *conf.page_table_marked_bit, cross_pop_miss);
        }
        if (conf.page_table_pointer_mask != 0) {
            code.AND(Xscratch1, Xscratch1, conf.page_table_pointer_mask);
            code.AND(Xscratch2, Xscratch2, conf.page_table_pointer_mask);
        }
        code.CBZ(Xscratch1, cross_pop_miss);
        code.CMP(Xscratch1, Xscratch2);
        code.B(NE, cross_pop_miss);
        if (conf.page_table_sign_extension) {
            code.SBFM(Xscratch1, Xscratch1, 0, *conf.page_table_sign_extension);
        }
        code.LDR(Xscratch2, SP, POST_INDEXED, 16);
        code.ADD(Xscratch0, Xscratch1, Xscratch0);
        code.RET();

        code.l(cross_pop_miss);
        code.LDR(Xscratch2, SP, POST_INDEXED, 16);
    }

    code.l(miss);
    code.MOV(Xscratch0, 0);
    code.RET();
    return entry;
}

template<std::size_t bitsize>
void EmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (const auto marker = ShouldFastmem(ctx, inst, true)) {
        FastmemEmitReadMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitReadMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitReadMemory<bitsize>(code, ctx, inst);
    }
}

static void EmitStoreExclusiveReservation(oaknut::CodeGenerator& code, const EmitContext& ctx,
                                          oaknut::XReg Xaddr, oaknut::XReg Xvalue) {
    code.MOV(Xscratch0, ctx.conf.exclusive_address_pointer);
    const s64 value_disp = static_cast<s64>(ctx.conf.exclusive_value_pointer - ctx.conf.exclusive_address_pointer);
    if (value_disp == 8) {
        code.STP(Xaddr, Xvalue, Xscratch0, 0);
    } else if (value_disp >= 0 && (value_disp % 8) == 0 && value_disp <= 4095 * 8) {
        code.STR(Xvalue, Xscratch0, static_cast<int>(value_disp));
        code.STR(Xaddr, Xscratch0);
    } else {
        code.STR(Xaddr, Xscratch0);
        code.MOV(Xscratch0, ctx.conf.exclusive_value_pointer);
        code.STR(Xvalue, Xscratch0);
    }
    code.MOV(Wscratch0, 1);
    code.STRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
}

// LDXR in fastmem with the global monitor ignored (EmitConfig::exclusive_address_pointer), as the
// x64 backend does it: load, then note address and value in this core's monitor slots. A fault or
// an address past the arena reads through the page table or plain read callback and notes the same.
template<std::size_t bitsize>
void FastmemEmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);
    const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
    constexpr RegisterList caller_gpr_mask = static_cast<u32>(ABI_CALLER_SAVE) & ~(u32{1} << 30);
    const RegisterList live_caller_save =
        ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue)) |
        (ToRegList(*Xaddr) & caller_gpr_mask);
    const bool inline_save = SaveOpCount(live_caller_save) <=
                             (use_pt_fallback ? MaxPageTableSaveOps(ctx) : kMaxFastmemFallbackSaveOps);

    SharedLabel fallback = GenSharedLabel(), loaded = GenSharedLabel();
    SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
    const auto fastmem_location = EmitMemoryLdr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    code.l(*loaded);
    EmitStoreExclusiveReservation(code, ctx, *Xaddr, Rvalue->toX());

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, live_caller_save, inline_save, use_pt_fallback, bounds_miss, fallback, loaded, fastmem_location] {
        if (use_pt_fallback) {
            code.l(*bounds_miss);
            if constexpr (bitsize > 8) {
                code.TST(Xaddr, u64(bitsize / 8 - 1));
                code.B(NE, *fallback);
            }
            UntrappedCachedReadLabels pt_cached;
            const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
                code, ctx, Xaddr, fallback, true, &pt_cached, nullptr, false);
            EmitMemoryLdr<bitsize>(code, Rvalue.index(), Xpt_base, Xpt_offset, ordered);
            code.B(*loaded);
            EmitUntrappedCachedReadCheck(code, ctx, Xaddr, pt_cached, fallback);
        }

        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = std::bit_cast<u64>(code.xptr<void*>()),
                },
                .recompile = ctx.conf.recompile_on_exclusive_fastmem_failure,
                .load = true,
            });

        code.l(*fallback);
        EmitReadFallback<bitsize>(code, ctx, inst, Xaddr, Rvalue.index(), false, ordered, live_caller_save, inline_save, *loaded);
    });
}

template<std::size_t bitsize>
void InlinePageTableEmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);
    constexpr RegisterList caller_gpr_mask = static_cast<u32>(ABI_CALLER_SAVE) & ~(u32{1} << 30);
    const RegisterList live_caller_save =
        ctx.reg_alloc.LiveCallerSaveRegisters(ToRegList(*Rvalue)) |
        (ToRegList(*Xaddr) & caller_gpr_mask);
    const bool inline_save = SaveOpCount(live_caller_save) <= MaxPageTableSaveOps(ctx);

    SharedLabel fallback = GenSharedLabel(), loaded = GenSharedLabel();
    UntrappedCachedReadLabels pt_cached;

    if constexpr (bitsize > 8) {
        code.TST(*Xaddr, u64(bitsize / 8 - 1));
        code.B(NE, *fallback);
    }
    const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
        code, ctx, Xaddr, fallback, false, &pt_cached, nullptr, false);
    EmitMemoryLdr<bitsize>(code, Rvalue->index(), Xpt_base, Xpt_offset, ordered);

    code.l(*loaded);
    EmitStoreExclusiveReservation(code, ctx, *Xaddr, Rvalue->toX());

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, live_caller_save, inline_save, pt_cached, fallback, loaded] {
        EmitUntrappedCachedReadCheck(code, ctx, Xaddr, pt_cached, fallback);
        code.l(*fallback);
        EmitReadFallback<bitsize>(code, ctx, inst, Xaddr, Rvalue.index(), false, ordered, live_caller_save, inline_save, *loaded);
    });
}

template<std::size_t bitsize, typename RegT>
static void EmitExclusiveWriteCasFallback(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst,
                                          oaknut::XReg Xaddr, RegT Rvalue, oaknut::WReg Wstatus,
                                          u64 value_pointer, oaknut::Label& end) {
    code.MOV(Xscratch0, Xaddr);
    code.MOV(Xscratch1, Xscratch2);
    code.STR(Rvalue.toX(), Xstate, ctx.conf.state_atomic_scratch_offset);
    EmitRelocation(code, ctx, WrappedAtomicCasLinkTarget(bitsize));
    code.DMB(oaknut::BarrierOp::ISH);
    // Xscratch0 is the old value; the call kept neither scratch register, so the expected
    // value is read from the monitor slot again (the callbacks do not touch it).
    code.MOV(Xscratch1, value_pointer);
    if constexpr (bitsize == 8) {
        code.LDRB(Wscratch1, Xscratch1, 0);
    } else if constexpr (bitsize == 16) {
        code.LDRH(Wscratch1, Xscratch1, 0);
    } else if constexpr (bitsize == 32) {
        code.LDR(Wscratch1, Xscratch1, 0);
    } else {
        code.LDR(Xscratch1, Xscratch1, 0);
    }
    if constexpr (bitsize == 64) {
        code.CMP(Xscratch0, Xscratch1);
    } else {
        code.CMP(Wscratch0, Wscratch1);
    }
    code.CSET(Wstatus, NE);
    ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
    code.B(end);
}

// STXR in fastmem with the global monitor ignored: if this core holds the reservation for the
// address, a CAS from the value its LDXR saw (CASAL: release and acquire). A fault or an address
// past the arena does the CAS through the page table or the callbacks (the LSE atomics' fallback).
template<std::size_t bitsize>
void FastmemEmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
    auto Wstatus = ctx.reg_alloc.WriteW(inst);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue, Wstatus);

    const bool use_pt_fallback = CanInlinePageTableFallback(ctx);
    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();
    SharedLabel bounds_miss = use_pt_fallback ? GenSharedLabel() : fallback;

    const auto compare = [&code](oaknut::XReg a, oaknut::XReg b) {
        if constexpr (bitsize == 64) {
            code.CMP(a, b);
        } else {
            code.CMP(a.toW(), b.toW());
        }
    };
    // Xscratch2 = the value this core's LDXR saw, kept untouched across FastmemEmitVAddrLookup
    // and InlinePageTableEmitVAddrLookup for the post-CAS compare and fallback.
    const u64 value_pointer = ctx.conf.exclusive_value_pointer;
    const s64 value_disp = static_cast<s64>(value_pointer - ctx.conf.exclusive_address_pointer);
    const bool near_value = value_disp >= 0 && (value_disp % 8) == 0 && value_disp <= 4095;
    const auto load_expected_at = [&code](oaknut::XReg Xptr, int offset) {
        if constexpr (bitsize == 8) {
            code.LDRB(Wscratch2, Xptr, offset);
        } else if constexpr (bitsize == 16) {
            code.LDRH(Wscratch2, Xptr, offset);
        } else if constexpr (bitsize == 32) {
            code.LDR(Wscratch2, Xptr, offset);
        } else {
            code.LDR(Xscratch2, Xptr, offset);
        }
    };

    code.MOV(Wstatus, 1);
    code.LDRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.CBZ(Wscratch0, *end);
    code.STRB(WZR, Xstate, ctx.conf.state_exclusive_state_offset);
    code.MOV(Xscratch0, ctx.conf.exclusive_address_pointer);
    code.LDR(Xscratch1, Xscratch0);
    code.CMP(Xscratch1, *Xaddr);
    code.B(NE, *end);
    if (near_value) {
        load_expected_at(Xscratch0, static_cast<int>(value_disp));
    } else {
        code.MOV(Xscratch2, value_pointer);
        load_expected_at(Xscratch2, 0);
    }
    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, bounds_miss);
    const CodePtr fastmem_location = EmitLseCasAt<bitsize>(
        code, Xscratch2.index(), Rvalue->index(), Xscratch1.index(), Xbase, Xoffset, ShouldExt32(ctx));
    compare(Xscratch1, Xscratch2);
    code.CSET(Wstatus, NE);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, Wstatus = *Wstatus, value_pointer, compare, use_pt_fallback, bounds_miss, fallback, end, fastmem_location] {
        if (use_pt_fallback) {
            code.l(*bounds_miss);
            if constexpr (bitsize > 8) {
                code.TST(Xaddr, u64(bitsize / 8 - 1));
                code.B(NE, *fallback);
            }
            const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
                code, ctx, Xaddr, fallback, true, nullptr, nullptr, false);
            EmitLseCasAt<bitsize>(code, Xscratch2.index(), Rvalue.index(), Xscratch1.index(), Xpt_base, Xpt_offset);
            compare(Xscratch1, Xscratch2);
            code.CSET(Wstatus, NE);
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
        EmitExclusiveWriteCasFallback<bitsize>(code, ctx, inst, Xaddr, Rvalue, Wstatus, value_pointer, *end);
    });

    code.l(*end);
}

template<std::size_t bitsize>
void InlinePageTableEmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
    auto Wstatus = ctx.reg_alloc.WriteW(inst);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue, Wstatus);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const u64 value_pointer = ctx.conf.exclusive_value_pointer;
    const s64 value_disp = static_cast<s64>(value_pointer - ctx.conf.exclusive_address_pointer);
    const bool near_value = value_disp >= 0 && (value_disp % 8) == 0 && value_disp <= 4095;
    const auto load_expected_at = [&code](oaknut::XReg Xptr, int offset) {
        if constexpr (bitsize == 8) {
            code.LDRB(Wscratch2, Xptr, offset);
        } else if constexpr (bitsize == 16) {
            code.LDRH(Wscratch2, Xptr, offset);
        } else if constexpr (bitsize == 32) {
            code.LDR(Wscratch2, Xptr, offset);
        } else {
            code.LDR(Xscratch2, Xptr, offset);
        }
    };

    code.MOV(Wstatus, 1);
    code.LDRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.CBZ(Wscratch0, *end);
    code.STRB(WZR, Xstate, ctx.conf.state_exclusive_state_offset);
    code.MOV(Xscratch0, ctx.conf.exclusive_address_pointer);
    code.LDR(Xscratch1, Xscratch0);
    code.CMP(Xscratch1, *Xaddr);
    code.B(NE, *end);
    if (near_value) {
        load_expected_at(Xscratch0, static_cast<int>(value_disp));
    } else {
        code.MOV(Xscratch2, value_pointer);
        load_expected_at(Xscratch2, 0);
    }
    if constexpr (bitsize > 8) {
        code.TST(*Xaddr, u64(bitsize / 8 - 1));
        code.B(NE, *fallback);
    }
    const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
        code, ctx, Xaddr, fallback, false, nullptr, nullptr, false);
    EmitLseCasAt<bitsize>(code, Xscratch2.index(), Rvalue->index(), Xscratch1.index(), Xpt_base, Xpt_offset);
    if constexpr (bitsize == 64) {
        code.CMP(Xscratch1, Xscratch2);
    } else {
        code.CMP(Wscratch1, Wscratch2);
    }
    code.CSET(Wstatus, NE);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, Wstatus = *Wstatus, value_pointer, fallback, end] {
        code.l(*fallback);
        EmitExclusiveWriteCasFallback<bitsize>(code, ctx, inst, Xaddr, Rvalue, Wstatus, value_pointer, *end);
    });

    code.l(*end);
}

template<std::size_t bitsize>
void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if constexpr (bitsize != 128) {
        if (ctx.conf.exclusive_address_pointer != 0) {
            if (const auto marker = ShouldFastmem(ctx, inst, true)) {
                FastmemEmitExclusiveReadMemory<bitsize>(code, ctx, inst, *marker);
                return;
            }
            if (ctx.conf.page_table_pointer != 0) {
                InlinePageTableEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
                return;
            }
        }
    }
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
    if constexpr (bitsize != 128) {
        if (ctx.conf.exclusive_address_pointer != 0) {
            if (const auto marker = ShouldFastmem(ctx, inst)) {
                FastmemEmitExclusiveWriteMemory<bitsize>(code, ctx, inst, *marker);
                return;
            }
            if (ctx.conf.page_table_pointer != 0) {
                InlinePageTableEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
                return;
            }
        }
    }
    CallbackOnlyEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
}

template<std::size_t bitsize, typename RegT>
static void EmitAtomicRmwCallback(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst,
                                  oaknut::XReg Xaddr, RegT Roperand, RegT Rresult, u8 encoded_op, oaknut::Label& end) {
    code.MOV(Xscratch0, Xaddr);
    code.MOV(Xscratch1, Roperand.toX());
    code.MOV(Xscratch2, u64(encoded_op & 0x0F));
    code.STR(Xscratch2, Xstate, ctx.conf.state_atomic_scratch_offset);
    EmitRelocation(code, ctx, WrappedAtomicRmwLinkTarget(bitsize));
    code.DMB(oaknut::BarrierOp::ISH);
    code.MOV(Rresult.toX(), Xscratch0);
    ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
}

template<std::size_t bitsize, typename RegT>
static void EmitAtomicCasCallback(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst,
                                  oaknut::XReg Xaddr, RegT Rexpected, RegT Rdesired, RegT Rresult, oaknut::Label& end) {
    code.MOV(Xscratch0, Xaddr);
    code.MOV(Xscratch1, Rexpected.toX());
    code.STR(Rdesired.toX(), Xstate, ctx.conf.state_atomic_scratch_offset);
    EmitRelocation(code, ctx, WrappedAtomicCasLinkTarget(bitsize));
    code.DMB(oaknut::BarrierOp::ISH);
    code.MOV(Rresult.toX(), Xscratch0);
    ctx.conf.emit_check_memory_abort(code, ctx, inst, end);
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
                const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
                    code, ctx, Xaddr, fallback, true, nullptr, nullptr, false);
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
            EmitAtomicRmwCallback<bitsize>(code, ctx, inst, Xaddr, Roperand, Rresult, encoded_op, *end);
            code.B(*end);
        });
    } else if (ctx.conf.page_table_pointer != 0) {
        if constexpr (bitsize > 8) {
            code.TST(Xaddr, u64(bitsize / 8 - 1));
            code.B(NE, *fallback);
        }
        const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
            code, ctx, Xaddr, fallback, false, nullptr, nullptr, false);
        EmitLseRmwAt<bitsize>(code, encoded_op, Roperand->index(), Rresult->index(), Xpt_base, Xpt_offset);

        ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Roperand = *Roperand, Rresult = *Rresult, encoded_op, fallback, end] {
            code.l(*fallback);
            EmitAtomicRmwCallback<bitsize>(code, ctx, inst, Xaddr, Roperand, Rresult, encoded_op, *end);
            code.B(*end);
        });
    } else {
        EmitAtomicRmwCallback<bitsize>(code, ctx, inst, *Xaddr, *Roperand, *Rresult, encoded_op, *end);
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
                const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
                    code, ctx, Xaddr, fallback, true, nullptr, nullptr, false);
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
            EmitAtomicCasCallback<bitsize>(code, ctx, inst, Xaddr, Rexpected, Rdesired, Rresult, *end);
            code.B(*end);
        });
    } else if (ctx.conf.page_table_pointer != 0) {
        if constexpr (bitsize > 8) {
            code.TST(Xaddr, u64(bitsize / 8 - 1));
            code.B(NE, *fallback);
        }
        const auto [Xpt_base, Xpt_offset] = InlinePageTableEmitVAddrLookup<bitsize>(
            code, ctx, Xaddr, fallback, false, nullptr, nullptr, false);
        EmitLseCasAt<bitsize>(code, Rexpected->index(), Rdesired->index(), Rresult->index(), Xpt_base, Xpt_offset);

        ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rexpected = *Rexpected, Rdesired = *Rdesired, Rresult = *Rresult, fallback, end] {
            code.l(*fallback);
            EmitAtomicCasCallback<bitsize>(code, ctx, inst, Xaddr, Rexpected, Rdesired, Rresult, *end);
            code.B(*end);
        });
    } else {
        EmitAtomicCasCallback<bitsize>(code, ctx, inst, *Xaddr, *Rexpected, *Rdesired, *Rresult, *end);
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
