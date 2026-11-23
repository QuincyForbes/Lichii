// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/reg_alloc.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iterator>

#include "common/assert.h"
#include "dynarmic/mcl/bit.hpp"
#include <bit>
#include "common/common_types.h"

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/verbose_debugging_output.h"
#include "dynarmic/common/always_false.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

constexpr std::size_t spill_offset = offsetof(StackLayout, spill);
constexpr std::size_t spill_slot_size = sizeof(decltype(StackLayout::spill)::value_type);
constexpr u32 callee_saved_allocatable_gprs = 0x00F8'0000U; // X19..X23

static bool MoveEliminationOn() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_MOV_ELIM");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

static bool SpillToFreeGprOn() noexcept {
    static const bool on = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_SPILL_GPR");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

static bool IsNarrowScalarType(IR::Type type) noexcept {
    return type == IR::Type::U8 || type == IR::Type::U16 || type == IR::Type::U32;
}

static bool IsValuelessType(IR::Type type) {
    switch (type) {
    case IR::Type::Table:
        return true;
    default:
        return false;
    }
}

static void BuildOrderMasks(std::span<const int> order, RegAlloc::OrderMasks& out) noexcept {
    out.clear();
    int prev = 32;
    for (const int reg : order) {
        if (reg <= prev) {
            out.push_back(u32{1} << reg);
        } else {
            out.back() |= u32{1} << reg;
        }
        prev = reg;
    }
}

IR::Type Argument::GetType() const {
    return value.GetType();
}

bool Argument::IsImmediate() const {
    return value.IsImmediate();
}

bool Argument::GetImmediateU1() const {
    return value.GetU1();
}

u8 Argument::GetImmediateU8() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100);
    return u8(imm);
}

u16 Argument::GetImmediateU16() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x10000);
    return u16(imm);
}

u32 Argument::GetImmediateU32() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100000000);
    return u32(imm);
}

u64 Argument::GetImmediateU64() const {
    return value.GetImmediateAsU64();
}

IR::Cond Argument::GetImmediateCond() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::Cond);
    return value.GetCond();
}

IR::AccType Argument::GetImmediateAccType() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::AccType);
    return value.GetAccType();
}

HostLoc::Kind Argument::CurrentLocationKind(RegAlloc& reg_alloc) const {
    return reg_alloc.ValueLocation(value.GetInst())->kind;
}

bool HostLocInfo::Contains(const IR::Inst* value) const {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void HostLocInfo::SetupScratchLocation() {
    ASSERT(IsCompletelyEmpty());
    realized = true;
}

void HostLocInfo::SetupLocation(const IR::Inst* value) {
    ASSERT(IsCompletelyEmpty());
    values.clear();
    fastmem_base = nullptr;
    values.push_back(value);
    realized = true;
    uses_this_inst = 0;
    accumulated_uses = 0;
    expected_uses = value->UseCount();
}

bool HostLocInfo::IsCompletelyEmpty() const {
    return values.empty() && fastmem_base == nullptr && !locked && !realized && !accumulated_uses && !expected_uses && !uses_this_inst;
}

bool HostLocInfo::MaybeAllocatable() const {
    return !locked && !realized;
}

bool HostLocInfo::IsOneRemainingUse() const {
    return accumulated_uses + 1 == expected_uses && uses_this_inst == 1;
}

void HostLocInfo::UpdateUses() {
    accumulated_uses += uses_this_inst;
    uses_this_inst = 0;

    if (accumulated_uses == expected_uses) {
        values.clear();
        fastmem_base = nullptr;
        accumulated_uses = 0;
        expected_uses = 0;
    }
}

RAFastmemBase::~RAFastmemBase() {
    if (!base_inst) {
        return;
    }
    const auto loc = reg_alloc.FastmemBaseLocation(base_inst);
    ASSERT(loc);
    auto& info = reg_alloc.ValueInfo(*loc);
    ASSERT(info.locked > 0);
    info.locked--;
    info.realized = false;
    if (info.IsCompletelyEmpty()) {
        reg_alloc.ClearActive(*loc);
    }
}

void RegAlloc::Reset() noexcept {
    for (u64 mask = active_gprs; mask != 0; mask &= mask - 1) {
        gprs[std::countr_zero(mask)] = {};
    }
    for (u64 mask = active_fprs; mask != 0; mask &= mask - 1) {
        fprs[std::countr_zero(mask)] = {};
    }
    if (active_flags) {
        flags = {};
    }
    for (u64 mask = active_spills; mask != 0; mask &= mask - 1) {
        spills[std::countr_zero(mask)] = {};
    }
    active_gprs = 0;
    active_fprs = 0;
    active_spills = 0;
    active_flags = false;
    alloc_candidate_index = 0;
    defined_insts.clear();
}

void RegAlloc::Reset(oaknut::CodeGenerator& code_, FpsrManager& fpsr_manager_, std::span<const int> gpr_order_, std::span<const int> fpr_order_) noexcept {
    code_ptr = &code_;
    fpsr_manager_ptr = &fpsr_manager_;
    if (gpr_order.data() != gpr_order_.data() || gpr_order.size() != gpr_order_.size()) {
        gpr_order = gpr_order_;
        BuildOrderMasks(gpr_order, gpr_order_masks);
    }
    if (fpr_order.data() != fpr_order_.data() || fpr_order.size() != fpr_order_.size()) {
        fpr_order = fpr_order_;
        BuildOrderMasks(fpr_order, fpr_order_masks);
    }
    Reset();
}

void RegAlloc::RecordDefined(const IR::Inst* inst) {
    if (inst != nullptr && IR::IsAPseudoOperation(inst->GetOpcode())) {
        defined_insts.push_back(inst);
    }
}

RegAlloc::ArgumentInfo RegAlloc::GetArgumentInfo(IR::Inst* inst) {
    ArgumentInfo ret = {Argument{}, Argument{}, Argument{}, Argument{}};
    for (std::size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->GetArg(i);
        ret[i].value = arg;
        if (!arg.IsImmediate() && !IsValuelessType(arg.GetType())) {
            const auto loc = ValueLocation(arg.GetInst());
            ASSERT(loc && "argument must already been defined");
            ValueInfo(*loc).uses_this_inst++;
        }
    }
    return ret;
}

bool RegAlloc::WasValueDefined(IR::Inst* inst) const {
    return std::find(defined_insts.begin(), defined_insts.end(), inst) != defined_insts.end();
}

void RegAlloc::PrepareForCall(std::optional<Argument::copyable_reference> arg0, std::optional<Argument::copyable_reference> arg1, std::optional<Argument::copyable_reference> arg2, std::optional<Argument::copyable_reference> arg3) {
    Fpsr().Spill();
    SpillFlags();

    // TODO: Spill into callee-save registers

    for (u64 mask = active_gprs & static_cast<u32>(ABI_CALLER_SAVE); mask != 0; mask &= mask - 1) {
        SpillGpr(std::countr_zero(mask));
    }

    for (u64 mask = active_fprs & static_cast<u32>(ABI_CALLER_SAVE >> 32); mask != 0; mask &= mask - 1) {
        SpillFpr(std::countr_zero(mask));
    }

    const std::array<std::optional<Argument::copyable_reference>, 4> args{arg0, arg1, arg2, arg3};

    // AAPCS64 Next General-purpose Register Number
    int ngrn = 0;
    // AAPCS64 Next SIMD and Floating-point Register Number
    int nsrn = 0;

    for (int i = 0; i < 4; i++) {
        if (args[i]) {
            if (args[i]->get().GetType() == IR::Type::U128) {
                ASSERT(fprs[nsrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, oaknut::QReg{nsrn});
                nsrn++;
            } else {
                ASSERT(gprs[ngrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, oaknut::XReg{ngrn});
                ngrn++;
            }
        } else {
            // Gaps are assumed to be in general-purpose registers
            // TODO: should there be a separate list passed for FPRs instead?
            ngrn++;
        }
    }
}

void RegAlloc::DefineAsExisting(IR::Inst* inst, Argument& arg) {
    RecordDefined(inst);
    DEBUG_ASSERT(!ValueLocation(inst));

    if (arg.value.IsImmediate()) {
        inst->ReplaceUsesWith(arg.value);
        return;
    }

    auto& info = ValueInfo(arg.value.GetInst());
    info.values.push_back(inst);
    info.expected_uses += inst->UseCount();
}

void RegAlloc::DefineAsRegister(IR::Inst* inst, oaknut::Reg reg) {
    RecordDefined(inst);
    DEBUG_ASSERT(!ValueLocation(inst));
    const HostLoc loc{reg.is_vector() ? HostLoc::Kind::Fpr : HostLoc::Kind::Gpr, reg.index()};
    auto& info = ValueInfo(loc);
    ASSERT(info.IsCompletelyEmpty());
    info.values.push_back(inst);
    info.expected_uses += inst->UseCount();
    MarkActive(loc);
}

void RegAlloc::UpdateAllUses() {
    for (u64 mask = active_gprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        gprs[i].UpdateUses();
        if (gprs[i].IsCompletelyEmpty()) {
            active_gprs &= ~(u64{1} << i);
        }
    }
    for (u64 mask = active_fprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        fprs[i].UpdateUses();
        if (fprs[i].IsCompletelyEmpty()) {
            active_fprs &= ~(u64{1} << i);
        }
    }
    if (active_flags) {
        flags.UpdateUses();
        if (flags.IsCompletelyEmpty()) {
            active_flags = false;
        }
    }
    for (u64 mask = active_spills; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        spills[i].UpdateUses();
        if (spills[i].IsCompletelyEmpty()) {
            active_spills &= ~(u64{1} << i);
        }
    }
}

void RegAlloc::AssertAllUnlocked() const {
    const auto is_unlocked = [](const auto& i) { return !i.locked && !i.realized; };
    for (u64 mask = active_gprs; mask != 0; mask &= mask - 1) {
        ASSERT(is_unlocked(gprs[std::countr_zero(mask)]));
    }
    for (u64 mask = active_fprs; mask != 0; mask &= mask - 1) {
        ASSERT(is_unlocked(fprs[std::countr_zero(mask)]));
    }
    if (active_flags) {
        ASSERT(is_unlocked(flags));
    }
    for (u64 mask = active_spills; mask != 0; mask &= mask - 1) {
        ASSERT(is_unlocked(spills[std::countr_zero(mask)]));
    }
#ifdef _DEBUG
    for (std::size_t i = 0; i < gprs.size(); ++i) {
        DEBUG_ASSERT(((active_gprs >> i) & 1) == u64{!gprs[i].IsCompletelyEmpty()});
    }
    for (std::size_t i = 0; i < fprs.size(); ++i) {
        DEBUG_ASSERT(((active_fprs >> i) & 1) == u64{!fprs[i].IsCompletelyEmpty()});
    }
    DEBUG_ASSERT(active_flags == !flags.IsCompletelyEmpty());
    for (std::size_t i = 0; i < spills.size(); ++i) {
        DEBUG_ASSERT(((active_spills >> i) & 1) == u64{!spills[i].IsCompletelyEmpty()});
    }
#endif
}

void RegAlloc::AssertNoMoreUses() const {
    ASSERT(active_gprs == 0 && active_fprs == 0 && !active_flags && active_spills == 0);
#ifdef _DEBUG
    const auto is_empty = [](const auto& i) { return i.IsCompletelyEmpty(); };
    DEBUG_ASSERT(std::all_of(gprs.begin(), gprs.end(), is_empty));
    DEBUG_ASSERT(std::all_of(fprs.begin(), fprs.end(), is_empty));
    DEBUG_ASSERT(is_empty(flags));
    DEBUG_ASSERT(std::all_of(spills.begin(), spills.end(), is_empty));
#endif
}

void RegAlloc::EmitVerboseDebuggingOutput() {
    auto& code = Code();
    code.MOV(X19, std::bit_cast<u64>(&PrintVerboseDebuggingOutputLine));  // Non-volatile register

    const auto do_location = [&](HostLocInfo& info, HostLocType type, std::size_t index) {
        using namespace oaknut::util;
        for (const IR::Inst* value : info.values) {
            code.MOV(X0, SP);
            code.MOV(X1, static_cast<u64>(type));
            code.MOV(X2, index);
            code.MOV(X3, value->GetName());
            code.MOV(X4, static_cast<u64>(value->GetType()));
            code.BLR(X19);
        }
    };

    for (std::size_t i = 0; i < gprs.size(); i++) {
        do_location(gprs[i], HostLocType::X, i);
    }
    for (std::size_t i = 0; i < fprs.size(); i++) {
        do_location(fprs[i], HostLocType::Q, i);
    }
    do_location(flags, HostLocType::Nzcv, 0);
    for (std::size_t i = 0; i < spills.size(); i++) {
        do_location(spills[i], HostLocType::Spill, i);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::GenerateImmediate(const IR::Value& value) {
    ASSERT(value.GetType() != IR::Type::U1);
    auto& code = Code();
    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order, gpr_order_masks, active_gprs);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupScratchLocation();
        active_gprs |= u64{1} << new_location_index;

        code.MOV(oaknut::XReg{new_location_index}, value.GetImmediateAsU64());

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order, fpr_order_masks, active_fprs);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupScratchLocation();
        active_fprs |= u64{1} << new_location_index;

        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.FMOV(oaknut::DReg{new_location_index}, Xscratch0);

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupScratchLocation();
        active_flags = true;

        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);

        return 0;
    } else {
        UNREACHABLE();
    }
}

template<HostLoc::Kind required_kind>
int RegAlloc::RealizeReadImpl(const IR::Value& value) {
    if (value.IsImmediate()) {
        return GenerateImmediate<required_kind>(value);
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == required_kind) {
        ValueInfo(*current_location).realized = true;
        return current_location->index;
    }

    ASSERT(!bool(ValueInfo(*current_location).realized));
    ASSERT(bool(ValueInfo(*current_location).locked));

    auto& code = Code();
    if constexpr (required_kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order, gpr_order_masks, active_gprs);
        SpillGpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            UNREACHABLE(); //logic error
        case HostLoc::Kind::Fpr:
            code.FMOV(oaknut::XReg{new_location_index}, oaknut::DReg{current_location->index});
            // ASSERT size fits
            break;
        case HostLoc::Kind::Spill:
            code.LDR(oaknut::XReg{new_location_index}, SP, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            code.MRS(oaknut::XReg{new_location_index}, oaknut::SystemReg::NZCV);
            break;
        }

        gprs[new_location_index] = std::exchange(ValueInfo(*current_location), {});
        gprs[new_location_index].realized = true;
        ClearActive(*current_location);
        active_gprs |= u64{1} << new_location_index;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order, fpr_order_masks, active_fprs);
        SpillFpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            if (IsNarrowScalarType(value.GetType())) {
                code.FMOV(oaknut::SReg{new_location_index}, oaknut::WReg{current_location->index});
            } else {
                code.FMOV(oaknut::DReg{new_location_index}, oaknut::XReg{current_location->index});
            }
            break;
        case HostLoc::Kind::Fpr:
            UNREACHABLE(); //logic error
        case HostLoc::Kind::Spill:
            code.LDR(oaknut::QReg{new_location_index}, SP, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            ASSERT(false && "Moving from flags into fprs is not currently supported");
            break;
        }

        fprs[new_location_index] = std::exchange(ValueInfo(*current_location), {});
        fprs[new_location_index].realized = true;
        ClearActive(*current_location);
        active_fprs |= u64{1} << new_location_index;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Flags) {
        UNREACHABLE(); //A simple read from flags is likely a logic error
    } else {
        UNREACHABLE();
    }
}

template<HostLoc::Kind kind>
int RegAlloc::RealizeWriteImpl(const IR::Inst* value) {
    RecordDefined(value);
    DEBUG_ASSERT(!ValueLocation(value));

    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order, gpr_order_masks, active_gprs);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupLocation(value);
        active_gprs |= u64{1} << new_location_index;
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order, fpr_order_masks, active_fprs);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupLocation(value);
        active_fprs |= u64{1} << new_location_index;
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupLocation(value);
        active_flags = true;
        return 0;
    } else {
        UNREACHABLE();
    }
}

template<HostLoc::Kind kind>
std::optional<int> RegAlloc::TryRealizeReadWriteInPlace(const IR::Value& read_value, const IR::Inst* write_value) {
    if constexpr (kind != HostLoc::Kind::Gpr && kind != HostLoc::Kind::Fpr) {
        return std::nullopt;
    }
    if (!MoveEliminationOn() || read_value.IsImmediate()) {
        return std::nullopt;
    }
    const auto current_location = ValueLocation(read_value.GetInst());
    ASSERT(current_location);
    if (current_location->kind != kind) {
        return std::nullopt;
    }
    auto& info = ValueInfo(*current_location);
    if (info.locked != 1 || info.realized || !info.IsOneRemainingUse()) {
        return std::nullopt;
    }
    RecordDefined(write_value);
    DEBUG_ASSERT(!ValueLocation(write_value));
    info.locked = 0;
    info = {};
    info.SetupLocation(write_value);
    MarkActive(*current_location);
    return current_location->index;
}

template<HostLoc::Kind kind>
int RegAlloc::RealizeReadWriteImpl(const IR::Value& read_value, const IR::Inst* write_value) {
    const int write_loc = RealizeWriteImpl<kind>(write_value);

    if constexpr (kind == HostLoc::Kind::Gpr) {
        LoadCopyInto(read_value, oaknut::XReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        LoadCopyInto(read_value, oaknut::QReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        ASSERT(false && "Incorrect function for ReadWrite of flags");
        UNREACHABLE();
    } else {
        UNREACHABLE();
    }
}

template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Gpr>(const IR::Value& value);
template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Fpr>(const IR::Value& value);
template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Flags>(const IR::Value& value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Gpr>(const IR::Inst* value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Fpr>(const IR::Inst* value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Flags>(const IR::Inst* value);
template std::optional<int> RegAlloc::TryRealizeReadWriteInPlace<HostLoc::Kind::Gpr>(const IR::Value&, const IR::Inst*);
template std::optional<int> RegAlloc::TryRealizeReadWriteInPlace<HostLoc::Kind::Fpr>(const IR::Value&, const IR::Inst*);
template std::optional<int> RegAlloc::TryRealizeReadWriteInPlace<HostLoc::Kind::Flags>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Gpr>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Fpr>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Flags>(const IR::Value&, const IR::Inst*);

int RegAlloc::AllocateRegister(const std::array<HostLocInfo, 32>& regs, std::span<const int> order, const OrderMasks& order_masks, u64 active_mask) const {
    const u32 free_mask = ~static_cast<u32>(active_mask);
    for (const u32 run_mask : order_masks) {
        if (const u32 avail = free_mask & run_mask; avail != 0) {
            const int reg = std::countr_zero(avail);
            DEBUG_ASSERT(regs[reg].IsCompletelyEmpty());
            return reg;
        }
    }

    boost::container::static_vector<int, 32> candidates;
    for (const int i : order) {
        if (regs[i].MaybeAllocatable()) {
            candidates.push_back(i);
        }
    }

    // TODO: The candidate was chosen randomly before, and a LRU was
    // suggested as an improvement. However, using an incrementing index
    // seems to be close enough. Determine if an LRU is still needed.
    return candidates[alloc_candidate_index++ % candidates.size()];
}

void RegAlloc::SpillGpr(int index) {
    ASSERT(!gprs[index].locked && !gprs[index].realized);
    if (!gprs[index].HasLiveValue()) {
        return;
    }
    if (SpillToFreeGprOn() && ((callee_saved_allocatable_gprs >> index) & 1U) == 0) {
        const u32 free_callee = (~static_cast<u32>(active_gprs)) & callee_saved_allocatable_gprs;
        if (free_callee != 0) {
            const int dst = std::countr_zero(free_callee);
            DEBUG_ASSERT(gprs[dst].IsCompletelyEmpty());
            Code().MOV(oaknut::XReg{dst}, oaknut::XReg{index});
            gprs[dst] = std::exchange(gprs[index], {});
            active_gprs = (active_gprs & ~(u64{1} << index)) | (u64{1} << dst);
            return;
        }
    }
    const int new_location_index = FindFreeSpill();
    Code().STR(oaknut::XReg{index}, SP, spill_offset + new_location_index * spill_slot_size);
    spills[new_location_index] = std::exchange(gprs[index], {});
    active_gprs &= ~(u64{1} << index);
    active_spills |= u64{1} << new_location_index;
}

bool RegAlloc::HasFastmemBase(const IR::Inst* base_inst) const {
    return FastmemBaseLocation(base_inst).has_value();
}

std::optional<HostLoc> RegAlloc::FastmemBaseLocation(const IR::Inst* base_inst) const {
    for (u64 mask = active_gprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        if (gprs[i].fastmem_base == base_inst) {
            return HostLoc{HostLoc::Kind::Gpr, i};
        }
    }
    for (u64 mask = active_spills; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        if (spills[i].fastmem_base == base_inst) {
            return HostLoc{HostLoc::Kind::Spill, i};
        }
    }
    return std::nullopt;
}

RAFastmemBase RegAlloc::AcquireExistingFastmemBase(const IR::Inst* base_inst) {
    const auto loc = FastmemBaseLocation(base_inst);
    ASSERT(loc);
    auto& info = ValueInfo(*loc);
    info.uses_this_inst++;
    info.locked++;
    return RAFastmemBase{*this, base_inst};
}

oaknut::XReg RegAlloc::RealizeExistingFastmemBase(RAFastmemBase& handle) {
    const auto loc = FastmemBaseLocation(handle.base_inst);
    ASSERT(loc);
    if (loc->kind == HostLoc::Kind::Gpr) {
        gprs[loc->index].realized = true;
        handle.reg = oaknut::XReg{loc->index};
        return *handle.reg;
    }
    ASSERT(loc->kind == HostLoc::Kind::Spill);
    const int new_idx = AllocateRegister(gprs, gpr_order, gpr_order_masks, active_gprs);
    SpillGpr(new_idx);
    Code().LDR(oaknut::XReg{new_idx}, SP, spill_offset + loc->index * spill_slot_size);
    gprs[new_idx] = std::exchange(ValueInfo(*loc), {});
    gprs[new_idx].realized = true;
    ClearActive(*loc);
    active_gprs |= u64{1} << new_idx;
    handle.reg = oaknut::XReg{new_idx};
    return *handle.reg;
}

RAFastmemBase RegAlloc::CreateFastmemBase(const IR::Inst* base_inst, std::size_t total_uses, oaknut::XReg Xfastmem_reg, oaknut::XReg Xbase_reg) {
    ASSERT(total_uses >= 2);
    DEBUG_ASSERT(!FastmemBaseLocation(base_inst));
    const int host_idx = AllocateRegister(gprs, gpr_order, gpr_order_masks, active_gprs);
    SpillGpr(host_idx);
    auto& info = gprs[host_idx];
    ASSERT(info.IsCompletelyEmpty());
    info.fastmem_base = base_inst;
    info.locked = 1;
    info.realized = true;
    info.uses_this_inst = 1;
    info.accumulated_uses = 0;
    info.expected_uses = total_uses;
    active_gprs |= u64{1} << host_idx;
    Code().ADD(oaknut::XReg{host_idx}, Xfastmem_reg, Xbase_reg);
    return RAFastmemBase{*this, base_inst, oaknut::XReg{host_idx}};
}

void RegAlloc::SpillFpr(int index) {
    ASSERT(!fprs[index].locked && !fprs[index].realized);
    if (fprs[index].values.empty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    Code().STR(oaknut::QReg{index}, SP, spill_offset + new_location_index * spill_slot_size);
    spills[new_location_index] = std::exchange(fprs[index], {});
    active_fprs &= ~(u64{1} << index);
    active_spills |= u64{1} << new_location_index;
}

void RegAlloc::ReadWriteFlags(Argument& read, IR::Inst* write) {
    RecordDefined(write);
    const auto current_location = ValueLocation(read.value.GetInst());
    ASSERT(current_location);

    auto& code = Code();
    if (current_location->kind == HostLoc::Kind::Flags) {
        if (!flags.IsOneRemainingUse()) {
            SpillFlags();
        } else if (write) {
            flags = {};
            active_flags = false;
        }
    } else if (current_location->kind == HostLoc::Kind::Gpr) {
        if (!flags.values.empty()) {
            SpillFlags();
        }
        code.MSR(oaknut::SystemReg::NZCV, oaknut::XReg{current_location->index});
    } else if (current_location->kind == HostLoc::Kind::Spill) {
        if (!flags.values.empty()) {
            SpillFlags();
        }
        code.LDR(Wscratch0, SP, spill_offset + current_location->index * spill_slot_size);
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);
    } else {
        UNREACHABLE(); //ASSERT(false && "Invalid current location for flags");
    }

    if (write) {
        flags.SetupLocation(write);
        flags.realized = false;
        active_flags = true;
    }
}

void RegAlloc::SpillFlags() {
    ASSERT(!flags.locked && !flags.realized);
    if (flags.values.empty()) {
        return;
    }
    const int new_location_index = AllocateRegister(gprs, gpr_order, gpr_order_masks, active_gprs);
    SpillGpr(new_location_index);
    Code().MRS(oaknut::XReg{new_location_index}, oaknut::SystemReg::NZCV);
    gprs[new_location_index] = std::exchange(flags, {});
    active_flags = false;
    active_gprs |= u64{1} << new_location_index;
}

RegisterList RegAlloc::LiveCallerSaveRegisters(RegisterList exclude) const noexcept {
    constexpr u32 caller_gprs = static_cast<u32>(ABI_CALLER_SAVE) & ~(u32{1} << 30);
    constexpr u32 caller_fprs = static_cast<u32>(ABI_CALLER_SAVE >> 32);
    RegisterList live = 0;

    for (u64 mask = active_gprs & caller_gprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        const auto& info = gprs[static_cast<std::size_t>(i)];
        if (info.HasLiveValue() && info.accumulated_uses + info.uses_this_inst < info.expected_uses) {
            live |= RegisterList{1} << i;
        }
    }
    for (u64 mask = active_fprs & caller_fprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        const auto& info = fprs[static_cast<std::size_t>(i)];
        if (info.HasLiveValue() && info.accumulated_uses + info.uses_this_inst < info.expected_uses) {
            live |= RegisterList{1} << (i + 32);
        }
    }
    return live & ~exclude;
}

int RegAlloc::FindFreeSpill() const {
    const int index = std::countr_one(active_spills);
    ASSERT(static_cast<std::size_t>(index) < SpillCount && "All spill locations are full");
    DEBUG_ASSERT(spills[index].values.empty());
    return index;
}

void RegAlloc::LoadCopyInto(const IR::Value& value, oaknut::XReg reg) {
    auto& code = Code();
    if (value.IsImmediate()) {
        code.MOV(reg, value.GetImmediateAsU64());
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        code.MOV(reg, oaknut::XReg{current_location->index});
        break;
    case HostLoc::Kind::Fpr:
        code.FMOV(reg, oaknut::DReg{current_location->index});
        // ASSERT size fits
        break;
    case HostLoc::Kind::Spill:
        code.LDR(reg, SP, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        code.MRS(reg, oaknut::SystemReg::NZCV);
        break;
    }
}

void RegAlloc::LoadCopyInto(const IR::Value& value, oaknut::QReg reg) {
    auto& code = Code();
    if (value.IsImmediate()) {
        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.FMOV(reg.toD(), Xscratch0);
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        if (IsNarrowScalarType(value.GetType())) {
            code.FMOV(reg.toS(), oaknut::WReg{current_location->index});
        } else {
            code.FMOV(reg.toD(), oaknut::XReg{current_location->index});
        }
        break;
    case HostLoc::Kind::Fpr:
        code.MOV(reg.B16(), oaknut::QReg{current_location->index}.B16());
        break;
    case HostLoc::Kind::Spill:
        // TODO: Minimize move size to max value width
        code.LDR(reg, SP, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        UNREACHABLE(); //ASSERT(false && "Moving from flags into fprs is not currently supported");
    }
}

std::optional<HostLoc> RegAlloc::ValueLocation(const IR::Inst* value) const {
    for (u64 mask = active_gprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        if (gprs[i].Contains(value)) {
            return HostLoc{HostLoc::Kind::Gpr, i};
        }
    }
    for (u64 mask = active_fprs; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        if (fprs[i].Contains(value)) {
            return HostLoc{HostLoc::Kind::Fpr, i};
        }
    }
    if (active_flags && flags.Contains(value)) {
        return HostLoc{HostLoc::Kind::Flags, 0};
    }
    for (u64 mask = active_spills; mask != 0; mask &= mask - 1) {
        const int i = std::countr_zero(mask);
        if (spills[i].Contains(value)) {
            return HostLoc{HostLoc::Kind::Spill, i};
        }
    }
    return std::nullopt;
}

HostLocInfo& RegAlloc::ValueInfo(HostLoc host_loc) {
    switch (host_loc.kind) {
    case HostLoc::Kind::Gpr:
        return gprs[static_cast<std::size_t>(host_loc.index)];
    case HostLoc::Kind::Fpr:
        return fprs[static_cast<std::size_t>(host_loc.index)];
    case HostLoc::Kind::Flags:
        return flags;
    case HostLoc::Kind::Spill:
        return spills[static_cast<std::size_t>(host_loc.index)];
    }
    UNREACHABLE();
}

HostLocInfo& RegAlloc::ValueInfo(const IR::Inst* value) {
    const auto loc = ValueLocation(value);
    ASSERT(loc);
    return ValueInfo(*loc);
}

}  // namespace Dynarmic::Backend::Arm64
