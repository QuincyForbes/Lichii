// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/cached_interpreter/cached_block_engine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace Core::CachedInterpreter {

namespace {

constexpr u32 ByteSwap32(u32 v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(v);
#else
    return ((v & 0x0000'00FFU) << 24) | ((v & 0x0000'FF00U) << 8) | ((v & 0x00FF'0000U) >> 8) |
           ((v & 0xFF00'0000U) >> 24);
#endif
}

constexpr u64 ByteSwap64(u64 v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#else
    return (static_cast<u64>(ByteSwap32(static_cast<u32>(v))) << 32) |
           ByteSwap32(static_cast<u32>(v >> 32));
#endif
}

u64 MulHighU64(u64 a, u64 b) noexcept {
#if defined(__SIZEOF_INT128__)
    return static_cast<u64>((static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b)) >>
                            64);
#elif defined(_MSC_VER) && defined(_M_X64)
    u64 hi = 0;
    (void)_umul128(a, b, &hi);
    return hi;
#else
    const u64 a_lo = a & 0xFFFF'FFFFULL, a_hi = a >> 32;
    const u64 b_lo = b & 0xFFFF'FFFFULL, b_hi = b >> 32;
    const u64 p0 = a_lo * b_lo;
    const u64 p1 = a_lo * b_hi;
    const u64 p2 = a_hi * b_lo;
    const u64 p3 = a_hi * b_hi;
    const u64 cy = ((p0 >> 32) + (p1 & 0xFFFF'FFFFULL) + (p2 & 0xFFFF'FFFFULL)) >> 32;
    return p3 + (p1 >> 32) + (p2 >> 32) + cy;
#endif
}

u64 MulHighS64(s64 a, s64 b) noexcept {
#if defined(__SIZEOF_INT128__)
    return static_cast<u64>((static_cast<__int128>(a) * static_cast<__int128>(b)) >> 64);
#else
    const u64 uhi = MulHighU64(static_cast<u64>(a), static_cast<u64>(b));
    const u64 sub_a = (a < 0) ? static_cast<u64>(b) : 0ULL;
    const u64 sub_b = (b < 0) ? static_cast<u64>(a) : 0ULL;
    return uhi - sub_a - sub_b;
#endif
}

constexpr u32 Flag64 = 1U << 0;
constexpr u32 FlagSetFlags = 1U << 1;

constexpr s64 SignExtend(u64 val, unsigned bits) noexcept {
    const unsigned shift = 64U - bits;
    return static_cast<s64>(val << shift) >> shift;
}

bool DecodeBitMasks(u32 immN, u32 imms, u32 immr, bool immediate, unsigned reg_bits, u64& wmask,
                    u64& tmask) noexcept {
    const u32 combined = (immN << 6) | (~imms & 0x3FU);
    if (combined == 0) {
        return false;
    }
    const unsigned len = 31U - static_cast<unsigned>(std::countl_zero(combined));
    if (len < 1) {
        return false;
    }
    const unsigned esize = 1U << len;
    if (esize > reg_bits) {
        return false;
    }
    const u32 levels = esize - 1U;
    const u32 s = imms & levels;
    const u32 r = immr & levels;
    if (immediate && s == levels) {
        return false;
    }
    const u64 welem = (s + 1U == 64U) ? ~u64{0} : ((u64{1} << (s + 1U)) - 1U);
    const u32 d = (s - r) & levels;
    const u64 telem = (d + 1U == 64U) ? ~u64{0} : ((u64{1} << (d + 1U)) - 1U);
    const u64 ror_w = (r == 0) ? welem
                               : (((welem >> r) | (welem << (esize - r))) &
                                  ((esize == 64U) ? ~u64{0} : ((u64{1} << esize) - 1U)));
    wmask = 0;
    tmask = 0;
    for (unsigned pos = 0; pos < reg_bits; pos += esize) {
        wmask |= ror_w << pos;
        tmask |= telem << pos;
    }
    return true;
}

u64 ShiftValue(u64 val, u8 shift_type, unsigned amount, bool is64) noexcept {
    const unsigned width = is64 ? 64U : 32U;
    if (!is64) {
        val &= 0xFFFF'FFFFULL;
    }
    amount &= (width - 1U);
    if (amount == 0) {
        return val;
    }
    switch (shift_type & 3U) {
    case 0:
        return is64 ? (val << amount) : ((val << amount) & 0xFFFF'FFFFULL);
    case 1:
        return val >> amount;
    case 2:
        return is64 ? static_cast<u64>(static_cast<s64>(val) >> amount)
                    : (static_cast<u64>(static_cast<s32>(val) >> amount) & 0xFFFF'FFFFULL);
    default:
        return is64 ? std::rotr(val, amount) : std::rotr(static_cast<u32>(val), amount);
    }
}

u64 ExtendValue(u64 val, u8 option, unsigned shift, bool is64) noexcept {
    u64 out = 0;
    switch (option & 7U) {
    case 0:
        out = val & 0xFFULL;
        break;
    case 1:
        out = val & 0xFFFFULL;
        break;
    case 2:
        out = val & 0xFFFF'FFFFULL;
        break;
    case 3:
        out = val;
        break;
    case 4:
        out = static_cast<u64>(static_cast<s64>(static_cast<s8>(val)));
        break;
    case 5:
        out = static_cast<u64>(static_cast<s64>(static_cast<s16>(val)));
        break;
    case 6:
        out = static_cast<u64>(static_cast<s64>(static_cast<s32>(val)));
        break;
    default:
        out = val;
        break;
    }
    out <<= (shift & 7U);
    return is64 ? out : (out & 0xFFFF'FFFFULL);
}

u64 AddWithCarry(u64 x, u64 y, bool carry_in, bool is64, bool& n, bool& z, bool& c,
                 bool& v) noexcept {
    if (is64) {
        const u64 cin = carry_in ? 1ULL : 0ULL;
        const u64 s1 = x + y;
        const bool c1 = s1 < x;
        const u64 res = s1 + cin;
        const bool c2 = res < s1;
        n = (res >> 63) != 0;
        z = res == 0;
        c = c1 || c2;
        v = (((x ^ ~y) & (x ^ res)) >> 63) != 0;
        if (carry_in && y == 0x7FFF'FFFF'FFFF'FFFFULL) {
            // Recompute signed overflow cleanly when y + cin wraps signed boundary
            const bool sx = (x >> 63) != 0;
            v = !sx && n;
        } else if (carry_in) {
            const u64 y_eff = y + 1ULL;
            if (y_eff != 0) {
                v = (((x ^ ~y_eff) & (x ^ res)) >> 63) != 0;
            } else {
                v = false;
            }
        }
        return res;
    }
    const u64 x32 = x & 0xFFFF'FFFFULL;
    const u64 y32 = y & 0xFFFF'FFFFULL;
    const u64 usum = x32 + y32 + (carry_in ? 1ULL : 0ULL);
    const s64 ssum = static_cast<s64>(static_cast<s32>(x32)) +
                     static_cast<s64>(static_cast<s32>(y32)) + (carry_in ? 1LL : 0LL);
    const u32 res = static_cast<u32>(usum);
    n = (res >> 31) != 0;
    z = res == 0;
    c = (usum >> 32) != 0;
    v = static_cast<s64>(static_cast<s32>(res)) != ssum;
    return res;
}

double DecodeFp8Imm(u8 imm8) noexcept {
    const double sign = (imm8 & 0x80U) ? -1.0 : 1.0;
    const int exp = (((imm8 & 0x40U) ? 0 : 4) | ((imm8 >> 4) & 3)) - 3;
    const double mant = 1.0 + static_cast<double>(imm8 & 0xFU) / 16.0;
    return std::ldexp(sign * mant, exp);
}

bool IsBlockTerminator(OpKind kind) noexcept {
    switch (kind) {
    case OpKind::BranchImm:
    case OpKind::BranchLinkImm:
    case OpKind::BranchReg:
    case OpKind::BranchCond:
    case OpKind::CompareBranchZero:
    case OpKind::TestBitBranch:
    case OpKind::Svc:
    case OpKind::Brk:
    case OpKind::FusedSubImmCbnz:
    case OpKind::FusedSubImmBcond:
    case OpKind::Unsupported:
        return true;
    default:
        return false;
    }
}

} // namespace

CachedBlockEngine::CachedBlockEngine(MemoryBus& bus_, bool enable_cache, bool enable_fusion)
    : bus{bus_}, cache(CacheEntries), cache_enabled{enable_cache}, fusion_enabled{enable_fusion} {}

void CachedBlockEngine::ClearCache() {
    ++generation;
    if (generation == 0) {
        for (auto& entry : cache) {
            entry.generation = 0;
            entry.op_count = 0;
        }
        generation = 1;
    }
    code_page_bitmap.fill(0);
    ++stats.invalidations;
}

void CachedBlockEngine::SetCacheEnabled(bool enabled) {
    if (cache_enabled != enabled) {
        cache_enabled = enabled;
        ClearCache();
    }
}

void CachedBlockEngine::MarkCodePage(u64 start_pc, u64 end_pc) noexcept {
    const u64 last = end_pc > start_pc ? (end_pc - 1) : start_pc;
    for (u64 page = start_pc & ~u64{0xFFF}; page <= (last & ~u64{0xFFF}); page += 0x1000) {
        const std::size_t bit = PageHash(page);
        code_page_bitmap[bit / 64] |= (u64{1} << (bit % 64));
    }
}

bool CachedBlockEngine::PageMayContainCode(u64 addr, std::size_t size) const noexcept {
    if (size == 0) {
        return false;
    }
    const u64 end = addr + size - 1;
    for (u64 page = addr & ~u64{0xFFF}; page <= (end & ~u64{0xFFF}); page += 0x1000) {
        const std::size_t bit = PageHash(page);
        if ((code_page_bitmap[bit / 64] & (u64{1} << (bit % 64))) != 0) {
            return true;
        }
    }
    return false;
}

void CachedBlockEngine::InvalidateRange(u64 vaddr, std::size_t size) {
    if (size == 0) {
        return;
    }
    const u64 end = vaddr + size;
    bool invalidated_any = false;
    for (auto& b : cache) {
        if (b.generation == generation && b.op_count != 0 && vaddr < b.end_pc && b.start_pc < end) {
            b.op_count = 0;
            invalidated_any = true;
        }
    }
    if (invalidated_any) {
        ++stats.invalidations;
    }
    bus.OnCodeRangeInvalidated(vaddr, size);
}

void CachedBlockEngine::NotifyGuestWrite(u64 addr, std::size_t size) {
    if (cache_enabled && PageMayContainCode(addr, size)) {
        InvalidateRange(addr, size);
    }
}

u64 CachedBlockEngine::ReadRegOrZero(u8 reg, bool is64) const noexcept {
    if (reg >= 31) {
        return 0;
    }
    return is64 ? state.x[reg] : (state.x[reg] & 0xFFFF'FFFFULL);
}

u64 CachedBlockEngine::ReadRegOrSp(u8 reg, bool is64) const noexcept {
    const u64 v = (reg >= 31) ? state.sp : state.x[reg];
    return is64 ? v : (v & 0xFFFF'FFFFULL);
}

void CachedBlockEngine::WriteRegOrZero(u8 reg, u64 value, bool is64) noexcept {
    if (reg < 31) {
        state.x[reg] = is64 ? value : (value & 0xFFFF'FFFFULL);
    }
}

void CachedBlockEngine::WriteRegOrSp(u8 reg, u64 value, bool is64) noexcept {
    const u64 v = is64 ? value : (value & 0xFFFF'FFFFULL);
    if (reg >= 31) {
        state.sp = v;
    } else {
        state.x[reg] = v;
    }
}

void CachedBlockEngine::SetNzcv(bool n, bool z, bool c, bool v) noexcept {
    state.pstate &= ~0xF000'0000U;
    state.pstate |= (u32{n} << 31) | (u32{z} << 30) | (u32{c} << 29) | (u32{v} << 28);
}

bool CachedBlockEngine::ConditionHolds(u8 cond) const noexcept {
    const bool n = (state.pstate >> 31) & 1U;
    const bool z = (state.pstate >> 30) & 1U;
    const bool c = (state.pstate >> 29) & 1U;
    const bool v = (state.pstate >> 28) & 1U;
    bool res = false;
    switch ((cond >> 1) & 7U) {
    case 0:
        res = z;
        break;
    case 1:
        res = c;
        break;
    case 2:
        res = n;
        break;
    case 3:
        res = v;
        break;
    case 4:
        res = c && !z;
        break;
    case 5:
        res = (n == v);
        break;
    case 6:
        res = (n == v) && !z;
        break;
    default:
        return true;
    }
    return ((cond & 1U) != 0 && (cond & 0xFU) != 0xFU) ? !res : res;
}

DecodedOp CachedBlockEngine::DecodeInstruction(u32 w, u64 pc) const {
    DecodedOp d{};
    d.raw = w;
    d.rd = static_cast<u8>(w & 31U);
    d.rn = static_cast<u8>((w >> 5) & 31U);
    d.rm = static_cast<u8>((w >> 16) & 31U);
    d.ra = static_cast<u8>((w >> 10) & 31U);

    // 1. Hints, barriers, CLREX, SVC, BRK, MRS, MSR, SYS
    if ((w & 0xFFFF'F01FU) == 0xD503'201FU) {
        d.kind = OpKind::Nop;
        return d;
    }
    if ((w & 0xFFFF'F01FU) == 0xD503'301FU) {
        d.kind = OpKind::Nop;
        d.aux = ((w >> 8) & 0xFU) == 0 ? 1 : 0; // bit0: CLREX when CRm..op2 == 0x3F5F
        if ((w & 0xFFFF'F0FFU) == 0xD503'305FU) {
            d.aux = 1; // CLREX
        }
        return d;
    }
    if ((w & 0xFFE0'001FU) == 0xD400'0001U) {
        d.kind = OpKind::Svc;
        d.imm = (w >> 5) & 0xFFFFU;
        return d;
    }
    if ((w & 0xFFE0'001FU) == 0xD420'0000U) {
        d.kind = OpKind::Brk;
        d.imm = (w >> 5) & 0xFFFFU;
        return d;
    }
    // DC ZVA: 0xD50B7420 | Rt
    if ((w & 0xFFFF'FFE0U) == 0xD50B'7420U) {
        d.kind = OpKind::Msr;
        d.aux = 0xFEU;
        return d;
    }
    // IC IVAU / DC CVAU / cache maintenance -> NOP/icache hint
    if ((w & 0xFFF8'0000U) == 0xD508'0000U) {
        d.kind = OpKind::Nop;
        return d;
    }
    // MSR (register) / MRS
    if ((w & 0xFFF0'0000U) == 0xD510'0000U || (w & 0xFFF0'0000U) == 0xD530'0000U) {
        const bool is_read = ((w >> 21) & 1U) != 0;
        const u32 sysreg = (w >> 5) & 0x7FFFU;
        switch (sysreg) {
        case 0x5A10U: // NZCV (op0=3,op1=3,CRn=4,CRm=2,op2=0)
        case 0x5A20U: // FPCR (op0=3,op1=3,CRn=4,CRm=4,op2=0)
        case 0x5A21U: // FPSR (op0=3,op1=3,CRn=4,CRm=4,op2=1)
        case 0x5E82U: // TPIDR_EL0
        case 0x5E83U: // TPIDRRO_EL0
        case 0x5F00U: // CNTFRQ_EL0
        case 0x5F01U: // CNTPCT_EL0
        case 0x5F02U: // CNTVCT_EL0
        case 0x5801U: // CTR_EL0
        case 0x5807U: // DCZID_EL0
            if (!is_read && (sysreg == 0x5E83U || sysreg >= 0x5F00U || sysreg < 0x5A00U)) {
                break;
            }
            d.kind = is_read ? OpKind::Mrs : OpKind::Msr;
            d.imm = sysreg;
            return d;
        default:
            break;
        }
    }

    const u32 op0 = (w >> 25) & 0xFU;

    // 2. Data processing - immediate (op0 = 100x)
    if ((op0 & 0xEU) == 0x8U) {
        const bool sf = ((w >> 31) & 1U) != 0;
        const u8 f64 = sf ? Flag64 : 0;
        const u32 subgrp = (w >> 23) & 7U;
        if ((subgrp & 6U) == 0) {
            // ADR / ADRP
            const s64 imm = SignExtend((((w >> 5) & 0x7FFFFULL) << 2) | ((w >> 29) & 3ULL), 21);
            if ((w >> 31) & 1U) {
                d.kind = OpKind::Adrp;
                d.imm = (pc & ~u64{0xFFF}) + static_cast<u64>(imm << 12);
            } else {
                d.kind = OpKind::Adr;
                d.imm = pc + static_cast<u64>(imm);
            }
            d.flags = Flag64;
            return d;
        }
        if (subgrp == 2 || subgrp == 3) {
            // ADD/ADDS/SUB/SUBS (immediate)
            const u32 sh = (w >> 22) & 1U;
            const u64 imm12 = (w >> 10) & 0xFFFULL;
            d.kind = ((w >> 30) & 1U) ? OpKind::SubImm : OpKind::AddImm;
            d.flags = f64 | (((w >> 29) & 1U) ? FlagSetFlags : 0U);
            d.imm = imm12 << (sh * 12U);
            return d;
        }
        if (subgrp == 4) {
            // Logical (immediate)
            const u32 opc = (w >> 29) & 3U;
            const u32 n_bit = (w >> 22) & 1U;
            if (!sf && n_bit != 0) {
                return d;
            }
            u64 wmask = 0, tmask = 0;
            if (!DecodeBitMasks(n_bit, (w >> 10) & 0x3FU, (w >> 16) & 0x3FU, true, sf ? 64U : 32U,
                                wmask, tmask)) {
                return d;
            }
            d.kind = OpKind::LogicalImm;
            d.flags = f64 | (opc == 3U ? FlagSetFlags : 0U);
            d.aux = static_cast<u8>(opc);
            d.imm = wmask;
            return d;
        }
        if (subgrp == 5) {
            // MOVN / MOVZ / MOVK
            const u32 opc = (w >> 29) & 3U;
            const u32 hw = (w >> 21) & 3U;
            if (opc == 1U || (!sf && hw >= 2U)) {
                return d;
            }
            d.kind = OpKind::MovWide;
            d.flags = f64;
            d.aux = static_cast<u8>((opc << 2) | hw);
            d.imm = (w >> 5) & 0xFFFFULL;
            return d;
        }
        if (subgrp == 6) {
            // SBFM / BFM / UBFM
            const u32 opc = (w >> 29) & 3U;
            const u32 n_bit = (w >> 22) & 1U;
            if (opc == 3U || (n_bit != (sf ? 1U : 0U))) {
                return d;
            }
            u64 wmask = 0, tmask = 0;
            const u32 imms = (w >> 10) & 0x3FU;
            const u32 immr = (w >> 16) & 0x3FU;
            if ((!sf && ((imms & 0x20U) || (immr & 0x20U))) ||
                !DecodeBitMasks(n_bit, imms, immr, false, sf ? 64U : 32U, wmask, tmask)) {
                return d;
            }
            d.kind = OpKind::Bitfield;
            d.flags = f64;
            d.aux = static_cast<u8>(opc);
            d.ra = static_cast<u8>(immr);
            d.rm = static_cast<u8>(imms);
            d.imm = wmask;
            d.imm2 = tmask;
            return d;
        }
        if (subgrp == 7) {
            // EXTR
            const u32 opc = (w >> 29) & 3U;
            const u32 n_bit = (w >> 22) & 1U;
            const u32 o0 = (w >> 21) & 1U;
            const u32 imms = (w >> 10) & 0x3FU;
            if (opc != 0U || o0 != 0U || n_bit != (sf ? 1U : 0U) || (!sf && imms >= 32U)) {
                return d;
            }
            d.kind = OpKind::Extr;
            d.flags = f64;
            d.aux = static_cast<u8>(imms);
            return d;
        }
    }

    // 3. Branches (op0 = 101x)
    if ((op0 & 0xEU) == 0xAU) {
        if ((w & 0x7C00'0000U) == 0x1400'0000U) {
            // B / BL imm26
            const s64 off = SignExtend(w & 0x03FF'FFFFULL, 26) * 4;
            d.kind = ((w >> 31) & 1U) ? OpKind::BranchLinkImm : OpKind::BranchImm;
            d.imm = pc + static_cast<u64>(off);
            return d;
        }
        if ((w & 0xFF00'0010U) == 0x5400'0000U) {
            // B.cond
            const s64 off = SignExtend((w >> 5) & 0x7FFFFULL, 19) * 4;
            d.kind = OpKind::BranchCond;
            d.ra = static_cast<u8>(w & 0xFU);
            d.imm = pc + static_cast<u64>(off);
            return d;
        }
        if ((w & 0x7E00'0000U) == 0x3400'0000U) {
            // CBZ / CBNZ
            const s64 off = SignExtend((w >> 5) & 0x7FFFFULL, 19) * 4;
            d.kind = OpKind::CompareBranchZero;
            d.flags = (((w >> 31) & 1U) ? Flag64 : 0U) | (((w >> 24) & 1U) ? FlagSetFlags : 0U);
            d.imm = pc + static_cast<u64>(off);
            return d;
        }
        if ((w & 0x7E00'0000U) == 0x3600'0000U) {
            // TBZ / TBNZ
            const s64 off = SignExtend((w >> 5) & 0x3FFFULL, 14) * 4;
            d.kind = OpKind::TestBitBranch;
            d.flags = Flag64 | (((w >> 24) & 1U) ? FlagSetFlags : 0U);
            d.aux = static_cast<u8>((((w >> 31) & 1U) << 5) | ((w >> 19) & 0x1FU));
            d.imm = pc + static_cast<u64>(off);
            return d;
        }
        if ((w & 0xFE1F'FC1FU) == 0xD61F'0000U) {
            // BR / BLR / RET
            const u32 opc = (w >> 21) & 3U;
            if (opc <= 2U) {
                d.kind = OpKind::BranchReg;
                d.aux = static_cast<u8>(opc); // 0=BR, 1=BLR, 2=RET
                return d;
            }
        }
        // RETAA / RETAB (0xD65F0BFF / 0xD65F0FFF)
        if (w == 0xD65F'0BFFU || w == 0xD65F'0FFFU) {
            d.kind = OpKind::BranchReg;
            d.rn = 30;
            d.aux = 2;
            return d;
        }
    }

    // 4. Data processing - register (op0 = x101 -> (w & 0x0E000000) == 0x0A000000 or 0x1A000000)
    if ((w & 0x0E00'0000U) == 0x0A00'0000U) {
        const bool sf = ((w >> 31) & 1U) != 0;
        const u8 f64 = sf ? Flag64 : 0U;
        const u32 opc = (w >> 29) & 3U;
        const u32 shift = (w >> 22) & 3U;
        const u32 imm6 = (w >> 10) & 0x3FU;
        if (!sf && imm6 >= 32U) {
            return d;
        }
        if (((w >> 24) & 1U) == 0U) {
            // Logical (shifted register)
            d.kind = OpKind::LogicalReg;
            d.flags = f64 | (opc == 3U ? FlagSetFlags : 0U) | ((((w >> 21) & 1U) != 0) ? 4U : 0U);
            d.aux = static_cast<u8>((opc << 2) | shift);
            d.imm = imm6;
            return d;
        }
        if (((w >> 21) & 1U) == 0U) {
            // ADD/SUB (shifted register)
            if (shift == 3U) {
                return d;
            }
            d.kind = OpKind::AddSubReg;
            d.flags = f64 | ((opc & 1U) ? FlagSetFlags : 0U) | ((opc & 2U) ? 4U : 0U);
            d.aux = static_cast<u8>(shift);
            d.imm = imm6;
            return d;
        }
        // ADD/SUB (extended register)
        if (shift != 0U || imm6 > 4U) {
            return d;
        }
        d.kind = OpKind::AddSubExtReg;
        d.flags = f64 | ((opc & 1U) ? FlagSetFlags : 0U) | ((opc & 2U) ? 4U : 0U);
        d.aux = static_cast<u8>((((w >> 13) & 7U) << 3) | (imm6 & 7U));
        return d;
    }

    if ((w & 0x1E00'0000U) == 0x1A00'0000U) {
        const bool sf = ((w >> 31) & 1U) != 0;
        const u8 f64 = sf ? Flag64 : 0U;
        const u32 op2 = (w >> 21) & 0xFU;
        if (op2 == 0U && ((w >> 10) & 0x3FU) == 0U) {
            // ADC / ADCS / SBC / SBCS
            const u32 opc = (w >> 29) & 3U;
            d.kind = OpKind::AddSubCarry;
            d.flags = f64 | ((opc & 1U) ? FlagSetFlags : 0U) | ((opc & 2U) ? 4U : 0U);
            return d;
        }
        if (op2 == 2U && (((w >> 10) & 2U) == 0U)) {
            // CCMN / CCMP (reg or imm)
            const u32 opc = (w >> 29) & 3U;
            if ((opc & 1U) == 0U) {
                return d;
            }
            d.kind = OpKind::CondCompare;
            d.flags = f64 | ((opc & 2U) ? 4U : 0U) | ((((w >> 11) & 1U) != 0) ? 8U : 0U);
            d.ra = static_cast<u8>((w >> 12) & 0xFU); // cond
            d.aux = static_cast<u8>(w & 0xFU);        // nzcv
            d.imm = (w >> 16) & 0x1FU;
            return d;
        }
        if (op2 == 4U && (((w >> 29) & 1U) == 0U) && (((w >> 11) & 1U) == 0U)) {
            // CSEL / CSINC / CSINV / CSNEG
            d.kind = OpKind::CondSelect;
            d.flags = f64;
            d.ra = static_cast<u8>((w >> 12) & 0xFU); // cond
            d.aux = static_cast<u8>((((w >> 30) & 1U) << 1) | ((w >> 10) & 1U));
            return d;
        }
        if (op2 == 6U && ((w >> 29) & 3U) == 0U) {
            // Data-processing 2-source (UDIV, SDIV, LSLV, LSRV, ASRV, RORV)
            const u32 opcode = (w >> 10) & 0x3FU;
            if (opcode == 2U || opcode == 3U) {
                d.kind = OpKind::MulDiv;
                d.flags = f64;
                d.aux = static_cast<u8>(opcode == 3U ? 10U : 9U); // 9=UDIV, 10=SDIV
                return d;
            }
            if (opcode >= 8U && opcode <= 11U) {
                d.kind = OpKind::ShiftReg;
                d.flags = f64;
                d.aux = static_cast<u8>(opcode - 8U);
                return d;
            }
        }
        if (op2 == 6U && ((w >> 29) & 3U) == 2U && ((w >> 16) & 0x1FU) == 0U) {
            // Data-processing 1-source (RBIT, REV16, REV32, REV64, CLZ, CLS)
            const u32 opcode = (w >> 10) & 0x3FU;
            if (opcode <= 5U && (sf || opcode != 3U)) {
                d.kind = OpKind::BitManip;
                d.flags = f64;
                d.aux = static_cast<u8>(opcode);
                return d;
            }
        }
        if (op2 >= 8U && ((w >> 29) & 3U) == 0U) {
            // Data-processing 3-source (MADD, MSUB, SMADDL, SMSUBL, SMULH, UMADDL, UMSUBL, UMULH)
            const u32 op31 = (w >> 21) & 7U;
            const u32 o0 = (w >> 15) & 1U;
            const u32 code = (op31 << 1) | o0;
            if (code == 0U || code == 1U ||
                (sf && (code == 2U || code == 3U || code == 4U || code == 10U || code == 11U ||
                        code == 12U))) {
                d.kind = OpKind::MulDiv;
                d.flags = f64;
                d.aux = static_cast<u8>(code);
                return d;
            }
        }
    }

    // 5. Loads and stores (op0 = x1x0 -> (w & 0x0A000000) == 0x08000000)
    if ((w & 0x0A00'0000U) == 0x0800'0000U) {
        const u32 size = (w >> 30) & 3U;
        const bool is_fp = ((w >> 26) & 1U) != 0;

        // Exclusive / Ordered loads & stores: (w & 0x3F000000) == 0x08000000
        if (!is_fp && (w & 0x3F00'0000U) == 0x0800'0000U) {
            const bool o2 = ((w >> 23) & 1U) != 0;
            const bool l_bit = ((w >> 22) & 1U) != 0;
            const bool o1 = ((w >> 21) & 1U) != 0;
            const bool o0 = ((w >> 15) & 1U) != 0;
            if (o1) {
                // LDXP / LDAXP / STXP / STLXP (size = 10 -> 32-bit pair, 11 -> 64-bit pair)
                if ((size & 2U) != 0 && !o2) {
                    d.kind = OpKind::LoadStoreExclusivePair;
                    d.flags = (size == 3U ? Flag64 : 0U) | (l_bit ? 4U : 0U);
                    return d;
                }
            } else if (o2) {
                // LDAR / STLR / LDLAR / STLLR -> ordered load/store
                d.kind = OpKind::LoadStoreImm;
                d.flags = static_cast<u8>(size); // aux encodes opc
                d.aux = l_bit ? 1U : 0U;         // 0=store, 1=zero-ext load
                d.imm = 0;
                return d;
            } else {
                // LDXR / LDAXR / STXR / STLXR
                d.kind = OpKind::LoadStoreExclusive;
                d.flags = static_cast<u8>(size) | (l_bit ? 4U : 0U) | (o0 ? 8U : 0U);
                return d;
            }
        }

        // CAS (LSE compare-and-swap): (w & 0x3FA00000) == 0x08A00000
        if (!is_fp && (w & 0x3FA0'7C00U) == 0x08A0'7C00U) {
            d.kind = OpKind::AtomicRmw;
            d.flags = static_cast<u8>(size);
            d.aux = 0xFFU; // CAS
            return d;
        }

        // Load literal (PC-relative): (w & 0x3B000000) == 0x18000000
        if ((w & 0x3B00'0000U) == 0x1800'0000U) {
            const s64 off = SignExtend((w >> 5) & 0x7FFFFULL, 19) * 4;
            d.imm = pc + static_cast<u64>(off);
            if (is_fp) {
                if (size <= 2U) {
                    d.kind = OpKind::LoadStoreFpLiteral;
                    d.aux = static_cast<u8>(size); // 0=S(4B), 1=D(8B), 2=Q(16B)
                }
            } else if (size <= 2U) {
                d.kind = OpKind::LoadLiteral;
                d.aux = static_cast<u8>(size); // 0=32-bit, 1=64-bit, 2=LDRSW
            }
            return d;
        }

        // Load/store pair (offset, post-index, pre-index): (w & 0x3A000000) == 0x28000000
        if ((w & 0x3A00'0000U) == 0x2800'0000U) {
            const u32 mode = (w >> 23) & 3U; // 1=post, 2=offset, 3=pre (0=no-allocate)
            const bool l_bit = ((w >> 22) & 1U) != 0;
            const s64 imm7 = SignExtend((w >> 15) & 0x7FULL, 7);
            if (is_fp) {
                if (size <= 2U && (mode != 0U || size <= 2U)) {
                    const unsigned scale = 2U + size; // 4, 8, 16 bytes
                    d.kind = OpKind::LoadStoreFpPair;
                    d.aux = static_cast<u8>((size << 3) | (mode << 1) | (l_bit ? 1U : 0U));
                    d.imm = static_cast<u64>(imm7 * (1LL << scale));
                }
            } else {
                if (size == 0U || (size == 1U && !l_bit) || size == 2U) {
                    const unsigned scale = (size == 2U) ? 3U : 2U;
                    d.kind = OpKind::LoadStorePair;
                    d.aux = static_cast<u8>((size << 3) | (mode << 1) | (l_bit ? 1U : 0U));
                    d.imm = static_cast<u64>(imm7 * (1LL << scale));
                } else if (size == 1U && l_bit) {
                    // LDPSW
                    d.kind = OpKind::LoadStorePair;
                    d.aux = static_cast<u8>((1U << 3) | (mode << 1) | 1U);
                    d.imm = static_cast<u64>(imm7 * 4LL);
                }
            }
            return d;
        }

        // Load/store register (unsigned immediate): (w & 0x3B000000) == 0x39000000
        if ((w & 0x3B00'0000U) == 0x3900'0000U) {
            const u32 opc = (w >> 22) & 3U;
            const u64 imm12 = (w >> 10) & 0xFFFULL;
            if (is_fp) {
                const unsigned scale = ((opc & 2U) != 0) ? (4U + size) : size;
                if (scale <= 4U && !(scale == 4U && size != 0U)) {
                    d.kind = OpKind::LoadStoreFpImm;
                    d.aux = static_cast<u8>((scale << 2) | (opc & 1U)); // bit0: load
                    d.imm = imm12 << scale;
                }
            } else {
                if (!(opc == 2U && size == 3U) && !(opc == 3U && size >= 2U)) {
                    d.kind = OpKind::LoadStoreImm;
                    d.flags = static_cast<u8>(size);
                    d.aux = static_cast<u8>(opc); // 0=STR,1=LDR,2=LDRS64,3=LDRS32
                    d.imm = imm12 << size;
                }
            }
            return d;
        }

        // Load/store unscaled / pre / post / reg / LSE atomic: (w & 0x3B000000) == 0x38000000
        if ((w & 0x3B00'0000U) == 0x3800'0000U) {
            const u32 opc = (w >> 22) & 3U;
            const u32 top21 = (w >> 21) & 1U;
            const u32 low2 = (w >> 10) & 3U;
            if (top21 == 0U && (low2 == 0U || low2 == 1U || low2 == 3U)) {
                // Unscaled (0), Post-index (1), Pre-index (3)
                const s64 simm9 = SignExtend((w >> 12) & 0x1FFULL, 9);
                if (is_fp) {
                    const unsigned scale = ((opc & 2U) != 0) ? (4U + size) : size;
                    if (scale <= 4U && !(scale == 4U && size != 0U)) {
                        d.kind = OpKind::LoadStoreFpImm;
                        d.aux = static_cast<u8>((low2 << 5) | (scale << 2) | (opc & 1U));
                        d.imm = static_cast<u64>(simm9);
                    }
                } else if (!(opc == 2U && size == 3U) && !(opc == 3U && size >= 2U)) {
                    d.kind = OpKind::LoadStoreImm;
                    d.flags = static_cast<u8>(size | (low2 << 2));
                    d.aux = static_cast<u8>(opc);
                    d.imm = static_cast<u64>(simm9);
                }
                return d;
            }
            if (top21 == 1U && low2 == 2U) {
                // Register offset
                const u32 option = (w >> 13) & 7U;
                const u32 s_bit = (w >> 12) & 1U;
                if ((option & 2U) == 0U) {
                    return d;
                }
                if (is_fp) {
                    const unsigned scale = ((opc & 2U) != 0) ? (4U + size) : size;
                    if (scale <= 4U && !(scale == 4U && size != 0U)) {
                        d.kind = OpKind::LoadStoreFpReg;
                        d.aux = static_cast<u8>((scale << 4) | (option << 1) | (opc & 1U));
                        d.flags = s_bit ? 1U : 0U;
                    }
                } else if (!(opc == 2U && size == 3U) && !(opc == 3U && size >= 2U)) {
                    d.kind = OpKind::LoadStoreReg;
                    d.flags = static_cast<u8>(size | (s_bit ? 8U : 0U));
                    d.aux = static_cast<u8>((option << 2) | opc);
                }
                return d;
            }
            if (!is_fp && top21 == 1U && low2 == 0U) {
                // LSE Atomics: LDADD, LDCLR, LDEOR, LDSET, SWP
                const u32 a_opc = (w >> 12) & 7U;
                if (a_opc == 0U || a_opc == 1U || a_opc == 2U || a_opc == 3U ||
                    ((w >> 12) & 0xFU) == 8U) {
                    d.kind = OpKind::AtomicRmw;
                    d.flags = static_cast<u8>(size);
                    d.aux = static_cast<u8>((w >> 12) & 0xFU);
                    return d;
                }
            }
        }
    }

    // 6. Scalar FP & basic SIMD zero/copy (op0 = x111 -> (w & 0x0E000000) == 0x0E000000)
    if ((w & 0x0E00'0000U) == 0x0E00'0000U) {
        // Common SIMD zero/copy idioms: MOVI Vd.2D/4S/8H/16B, #0 or EOR/ORR Vd.16B, Vn.16B, Vm.16B
        if (w == (0x4F00'0400U | (w & 0x1FU)) || w == (0x6F00'E400U | (w & 0x1FU)) ||
            w == (0x0F00'0400U | (w & 0x1FU))) {
            d.kind = OpKind::FpMoveImm;
            d.aux = 0xFEU; // zero 128-bit vector
            return d;
        }
        if ((w & 0xFFE0'FC00U) == 0x6E20'1C00U && d.rn == d.rm) {
            // EOR Vd.16B, Vn.16B, Vn.16B -> zero vector
            d.kind = OpKind::FpMoveImm;
            d.aux = 0xFEU;
            return d;
        }
        if ((w & 0xFFE0'FC00U) == 0x4EA0'1C00U && d.rn == d.rm) {
            // MOV Vd.16B, Vn.16B (ORR Vd.16B, Vn.16B, Vn.16B)
            d.kind = OpKind::FpMoveReg;
            d.aux = 2U; // 128-bit copy
            return d;
        }

        // Scalar FP (M=0, bit28=1 -> (w & 0x5F200000) == 0x1E200000)
        if ((w & 0x5F20'0000U) == 0x1E20'0000U) {
            const u32 ftype = (w >> 22) & 3U; // 0=float, 1=double
            if (ftype <= 1U) {
                const u8 is_dbl = (ftype == 1U) ? Flag64 : 0U;
                // FP <-> Int conversions & FMOV GPR<->FP: bit21=1, bits[29:30]=00, bits[15:10]=000000
                if (((w >> 29) & 3U) == 0U && ((w >> 21) & 1U) == 1U &&
                    ((w >> 10) & 0x3FU) == 0U) {
                    const bool sf = ((w >> 31) & 1U) != 0;
                    const u32 rmode = (w >> 19) & 3U;
                    const u32 opcode = (w >> 16) & 7U;
                    if (opcode == 6U || opcode == 7U) {
                        if (rmode == 0U && ((sf && ftype == 1U) || (!sf && ftype == 0U))) {
                            d.kind = OpKind::FpMoveGpr;
                            d.flags = (sf ? Flag64 : 0U) | ((opcode == 7U) ? 4U : 0U);
                            return d;
                        }
                    } else if (opcode <= 5U) {
                        d.kind = OpKind::FpConvert;
                        d.flags = (sf ? Flag64 : 0U) | (is_dbl ? 4U : 0U);
                        d.aux = static_cast<u8>((rmode << 3) | opcode);
                        return d;
                    }
                }
                if (((w >> 31) & 1U) == 0U && ((w >> 29) & 3U) == 0U && ((w >> 21) & 1U) == 1U) {
                    const u32 low = (w >> 10) & 0x3FU;
                    if ((low & 0x1FU) == 0x10U) {
                        // 1-source: FMOV, FABS, FNEG, FSQRT, FCVT, FRINT*
                        const u32 opc = (w >> 15) & 0x3FU;
                        d.kind = (opc == 0U) ? OpKind::FpMoveReg : OpKind::FpUnary;
                        d.flags = is_dbl;
                        d.aux = static_cast<u8>(opc);
                        return d;
                    }
                    if ((low & 0x1FU) == 0x08U) {
                        // FCMP / FCMPE
                        d.kind = OpKind::FpCompare;
                        d.flags = is_dbl;
                        d.aux = static_cast<u8>((w >> 3) & 3U); // bit0: zero, bit1: quiet/signal
                        return d;
                    }
                    if ((w & 0x0000'1FF0U) == 0x0000'1000U) {
                        // FMOV (scalar, immediate)
                        d.kind = OpKind::FpMoveImm;
                        d.flags = is_dbl;
                        d.aux = static_cast<u8>((w >> 13) & 0xFFU);
                        return d;
                    }
                    if ((low & 0x3U) == 0x2U && ((low >> 2) <= 8U)) {
                        // 2-source: FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM, FMINNM, FNMUL
                        d.kind = OpKind::FpBinary;
                        d.flags = is_dbl;
                        d.aux = static_cast<u8>((w >> 12) & 0xFU);
                        return d;
                    }
                    if ((low & 0x3U) == 0x3U) {
                        // FCSEL
                        d.kind = OpKind::FpCondSelect;
                        d.flags = is_dbl;
                        d.ra = static_cast<u8>((w >> 12) & 0xFU);
                        return d;
                    }
                }
            }
        }
        // Scalar FP 3-source (FMADD, FMSUB, FNMADD, FNMSUB): (w & 0xFF200000) == 0x1F000000
        if ((w & 0xFF20'0000U) == 0x1F00'0000U) {
            const u32 ftype = (w >> 22) & 3U;
            if (ftype <= 1U) {
                d.kind = OpKind::FpFma;
                d.flags = (ftype == 1U) ? Flag64 : 0U;
                d.aux = static_cast<u8>((((w >> 21) & 1U) << 1) | ((w >> 15) & 1U));
                return d;
            }
        }
    }

    return d;
}

bool CachedBlockEngine::DecodeBlock(CachedBlock& block, u64 start_pc, bool allow_fusion) {
    block.start_pc = start_pc;
    block.generation = generation;
    block.op_count = 0;
    block.guest_insns = 0;

    u64 pc = start_pc;
    while (block.op_count < CachedBlock::MaxOps) {
        if ((pc & 3ULL) != 0) {
            break;
        }
        const auto word = bus.ReadCode32(pc);
        if (!word.has_value()) {
            break;
        }
        DecodedOp op = DecodeInstruction(*word, pc);

        // Live block-level fusion for hot AArch64 sequences
        if (allow_fusion && block.op_count > 0) {
            DecodedOp& prev = block.ops[block.op_count - 1];
            // Pattern 1: ADRP Xd, page + ADD Xd/Rd, Xd, #imm -> FusedAdrpAdd
            if (prev.kind == OpKind::Adrp && op.kind == OpKind::AddImm &&
                (op.flags & FlagSetFlags) == 0 && (op.flags & Flag64) != 0 &&
                prev.rd < 31 && op.rn == prev.rd) {
                prev.kind = OpKind::FusedAdrpAdd;
                prev.rd = op.rd; // may write SP if op.rd == 31, handled in ExecuteOp via aux=1
                prev.aux = (op.rd == 31) ? 1 : 0;
                prev.imm = prev.imm + op.imm;
                prev.insn_count = 2;
                ++block.guest_insns;
                ++stats.ops_fused;
                pc += 4;
                continue;
            }
            // Pattern 2: SUB Xd/Wd, Xd/Wd, #imm (no flags) + CBNZ/CBZ Xd/Wd, target -> FusedSubImmCbnz
            if (prev.kind == OpKind::SubImm && (prev.flags & FlagSetFlags) == 0 &&
                op.kind == OpKind::CompareBranchZero && prev.rd < 31 && prev.rn == prev.rd &&
                op.rd == prev.rd && ((prev.flags & Flag64) == (op.flags & Flag64))) {
                prev.kind = OpKind::FusedSubImmCbnz;
                prev.flags = op.flags; // preserves Flag64 and FlagSetFlags(is_cbnz)
                prev.imm2 = op.imm;    // branch target
                prev.insn_count = 2;
                ++block.guest_insns;
                ++stats.ops_fused;
                pc += 4;
                break;
            }
            // Pattern 3: SUBS Xd/Wd, Xn/Wn, #imm + B.cond target -> FusedSubImmBcond
            if (prev.kind == OpKind::SubImm && (prev.flags & FlagSetFlags) != 0 &&
                op.kind == OpKind::BranchCond) {
                prev.kind = OpKind::FusedSubImmBcond;
                prev.ra = op.ra;    // condition code
                prev.imm2 = op.imm; // branch target
                prev.insn_count = 2;
                ++block.guest_insns;
                ++stats.ops_fused;
                pc += 4;
                break;
            }
        }

        block.ops[block.op_count++] = op;
        ++block.guest_insns;
        pc += 4;
        if (IsBlockTerminator(op.kind)) {
            break;
        }
    }

    block.end_pc = pc;
    if (block.op_count == 0) {
        return false;
    }
    if (cache_enabled) {
        MarkCodePage(block.start_pc, block.end_pc);
    }
    ++stats.blocks_decoded;
    return true;
}

StopReason CachedBlockEngine::ExecuteOp(const DecodedOp& op, u64& next_pc) {
    const bool is64 = (op.flags & Flag64) != 0;
    const bool set_flags = (op.flags & FlagSetFlags) != 0;

    switch (op.kind) {
    case OpKind::Nop:
        if (op.aux == 1) {
            bus.ClearExclusive();
        }
        return StopReason::None;

    case OpKind::AddImm:
    case OpKind::SubImm: {
        const u64 lhs = ReadRegOrSp(op.rn, is64);
        const u64 rhs = (op.kind == OpKind::SubImm) ? ~op.imm : op.imm;
        bool n = false, z = false, c = false, v = false;
        const u64 res = AddWithCarry(lhs, rhs, op.kind == OpKind::SubImm, is64, n, z, c, v);
        if (set_flags) {
            SetNzcv(n, z, c, v);
            WriteRegOrZero(op.rd, res, is64);
        } else {
            WriteRegOrSp(op.rd, res, is64);
        }
        return StopReason::None;
    }

    case OpKind::LogicalImm: {
        const u64 lhs = ReadRegOrZero(op.rn, is64);
        u64 res = 0;
        switch (op.aux) {
        case 0:
        case 3:
            res = lhs & op.imm;
            break;
        case 1:
            res = lhs | op.imm;
            break;
        default:
            res = lhs ^ op.imm;
            break;
        }
        if (!is64) {
            res &= 0xFFFF'FFFFULL;
        }
        if (set_flags) {
            SetNzcv((is64 ? (res >> 63) : (res >> 31)) != 0, res == 0, false, false);
            WriteRegOrZero(op.rd, res, is64);
        } else {
            WriteRegOrSp(op.rd, res, is64);
        }
        return StopReason::None;
    }

    case OpKind::MovWide: {
        const u32 opc = op.aux >> 2;
        const unsigned shift = (op.aux & 3U) * 16U;
        u64 res = 0;
        if (opc == 0U) {
            res = ~(op.imm << shift);
        } else if (opc == 2U) {
            res = op.imm << shift;
        } else {
            res = (ReadRegOrZero(op.rd, is64) & ~(u64{0xFFFF} << shift)) | (op.imm << shift);
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::Adr:
    case OpKind::Adrp:
        WriteRegOrZero(op.rd, op.imm, true);
        return StopReason::None;

    case OpKind::FusedAdrpAdd:
        if (op.aux != 0) {
            WriteRegOrSp(op.rd, op.imm, true);
        } else {
            WriteRegOrZero(op.rd, op.imm, true);
        }
        return StopReason::None;

    case OpKind::Bitfield: {
        const u64 src = ReadRegOrZero(op.rn, is64);
        const u64 dst = (op.aux == 1U) ? ReadRegOrZero(op.rd, is64) : 0ULL;
        const u64 ror = ShiftValue(src, 3, op.ra, is64);
        u64 bot = (dst & ~op.imm) | (ror & op.imm);
        u64 top = 0;
        if (op.aux == 0U) {
            const bool sign = ((src >> op.rm) & 1ULL) != 0;
            top = sign ? ~u64{0} : 0ULL;
        } else if (op.aux == 1U) {
            top = dst;
        }
        const u64 res = (top & ~op.imm2) | (bot & op.imm2);
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::Extr: {
        const u64 hi = ReadRegOrZero(op.rn, is64);
        const u64 lo = ReadRegOrZero(op.rm, is64);
        const unsigned lsb = op.aux;
        u64 res = 0;
        if (lsb == 0) {
            res = lo;
        } else if (is64) {
            res = (lo >> lsb) | (hi << (64U - lsb));
        } else {
            res = ((lo >> lsb) | (hi << (32U - lsb))) & 0xFFFF'FFFFULL;
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::AddSubReg: {
        const u64 lhs = ReadRegOrZero(op.rn, is64);
        u64 rhs = ShiftValue(ReadRegOrZero(op.rm, is64), op.aux, static_cast<unsigned>(op.imm),
                             is64);
        const bool is_sub = (op.flags & 4U) != 0;
        if (is_sub) {
            rhs = ~rhs;
        }
        bool n = false, z = false, c = false, v = false;
        const u64 res = AddWithCarry(lhs, rhs, is_sub, is64, n, z, c, v);
        if (set_flags) {
            SetNzcv(n, z, c, v);
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::AddSubExtReg: {
        const u64 lhs = ReadRegOrSp(op.rn, is64);
        u64 rhs = ExtendValue(ReadRegOrZero(op.rm, true), op.aux >> 3, op.aux & 7U, is64);
        const bool is_sub = (op.flags & 4U) != 0;
        if (is_sub) {
            rhs = ~rhs;
        }
        bool n = false, z = false, c = false, v = false;
        const u64 res = AddWithCarry(lhs, rhs, is_sub, is64, n, z, c, v);
        if (set_flags) {
            SetNzcv(n, z, c, v);
            WriteRegOrZero(op.rd, res, is64);
        } else {
            WriteRegOrSp(op.rd, res, is64);
        }
        return StopReason::None;
    }

    case OpKind::AddSubCarry: {
        const u64 lhs = ReadRegOrZero(op.rn, is64);
        u64 rhs = ReadRegOrZero(op.rm, is64);
        const bool is_sub = (op.flags & 4U) != 0;
        if (is_sub) {
            rhs = ~rhs;
        }
        const bool carry_in = ((state.pstate >> 29) & 1U) != 0;
        bool n = false, z = false, c = false, v = false;
        const u64 res = AddWithCarry(lhs, rhs, carry_in, is64, n, z, c, v);
        if (set_flags) {
            SetNzcv(n, z, c, v);
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::LogicalReg: {
        const u64 lhs = ReadRegOrZero(op.rn, is64);
        u64 rhs = ShiftValue(ReadRegOrZero(op.rm, is64), op.aux & 3U,
                             static_cast<unsigned>(op.imm), is64);
        if ((op.flags & 4U) != 0) {
            rhs = ~rhs;
        }
        u64 res = 0;
        switch ((op.aux >> 2) & 3U) {
        case 0:
        case 3:
            res = lhs & rhs;
            break;
        case 1:
            res = lhs | rhs;
            break;
        default:
            res = lhs ^ rhs;
            break;
        }
        if (!is64) {
            res &= 0xFFFF'FFFFULL;
        }
        if (set_flags) {
            SetNzcv((is64 ? (res >> 63) : (res >> 31)) != 0, res == 0, false, false);
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::ShiftReg: {
        const u64 lhs = ReadRegOrZero(op.rn, is64);
        const unsigned amt = static_cast<unsigned>(ReadRegOrZero(op.rm, is64));
        WriteRegOrZero(op.rd, ShiftValue(lhs, op.aux, amt, is64), is64);
        return StopReason::None;
    }

    case OpKind::MulDiv: {
        const u64 n = ReadRegOrZero(op.rn, is64);
        const u64 m = ReadRegOrZero(op.rm, is64);
        const u64 a = ReadRegOrZero(op.ra, true);
        u64 res = 0;
        switch (op.aux) {
        case 0:
            res = (is64 ? a : (a & 0xFFFF'FFFFULL)) + n * m;
            break;
        case 1:
            res = (is64 ? a : (a & 0xFFFF'FFFFULL)) - n * m;
            break;
        case 2:
            res = a + static_cast<u64>(static_cast<s64>(static_cast<s32>(n)) *
                                       static_cast<s64>(static_cast<s32>(m)));
            break;
        case 3:
            res = a - static_cast<u64>(static_cast<s64>(static_cast<s32>(n)) *
                                       static_cast<s64>(static_cast<s32>(m)));
            break;
        case 4:
            res = MulHighS64(static_cast<s64>(n), static_cast<s64>(m));
            break;
        case 9:
            res = (m == 0) ? 0ULL : (is64 ? (n / m) : ((n & 0xFFFF'FFFFULL) / (m & 0xFFFF'FFFFULL)));
            break;
        case 10:
            if (op.flags & Flag64) {
                if ((op.raw & 0x1F00'0000U) == 0x1B00'0000U) {
                    res = a + (n & 0xFFFF'FFFFULL) * (m & 0xFFFF'FFFFULL);
                } else {
                    const s64 sn = static_cast<s64>(n);
                    const s64 sm = static_cast<s64>(m);
                    res = (sm == 0) ? 0ULL
                                    : ((sn == std::numeric_limits<s64>::min() && sm == -1)
                                           ? static_cast<u64>(sn)
                                           : static_cast<u64>(sn / sm));
                }
            } else {
                const s32 sn = static_cast<s32>(n);
                const s32 sm = static_cast<s32>(m);
                res = (sm == 0) ? 0ULL
                                : ((sn == std::numeric_limits<s32>::min() && sm == -1)
                                       ? static_cast<u32>(sn)
                                       : static_cast<u32>(sn / sm));
            }
            break;
        case 11:
            res = a - (n & 0xFFFF'FFFFULL) * (m & 0xFFFF'FFFFULL);
            break;
        case 12:
            res = MulHighU64(n, m);
            break;
        default:
            break;
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::CondSelect: {
        u64 res = 0;
        if (ConditionHolds(op.ra)) {
            res = ReadRegOrZero(op.rn, is64);
        } else {
            res = ReadRegOrZero(op.rm, is64);
            if ((op.aux & 2U) != 0) {
                res = ~res;
            }
            if ((op.aux & 1U) != 0) {
                ++res;
            }
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::CondCompare: {
        if (ConditionHolds(op.ra)) {
            const u64 lhs = ReadRegOrZero(op.rn, is64);
            u64 rhs = ((op.flags & 8U) != 0) ? op.imm : ReadRegOrZero(op.rm, is64);
            const bool is_sub = (op.flags & 4U) != 0;
            if (is_sub) {
                rhs = ~rhs;
            }
            bool n = false, z = false, c = false, v = false;
            AddWithCarry(lhs, rhs, is_sub, is64, n, z, c, v);
            SetNzcv(n, z, c, v);
        } else {
            SetNzcv((op.aux >> 3) & 1U, (op.aux >> 2) & 1U, (op.aux >> 1) & 1U, op.aux & 1U);
        }
        return StopReason::None;
    }

    case OpKind::BitManip: {
        const u64 src = ReadRegOrZero(op.rn, is64);
        u64 res = 0;
        switch (op.aux) {
        case 0: {
            u64 v = src;
            v = ((v & 0x5555'5555'5555'5555ULL) << 1) | ((v >> 1) & 0x5555'5555'5555'5555ULL);
            v = ((v & 0x3333'3333'3333'3333ULL) << 2) | ((v >> 2) & 0x3333'3333'3333'3333ULL);
            v = ((v & 0x0F0F'0F0F'0F0F'0F0FULL) << 4) | ((v >> 4) & 0x0F0F'0F0F'0F0F'0F0FULL);
            res = is64 ? ByteSwap64(v) : ByteSwap32(static_cast<u32>(v));
            break;
        }
        case 1:
            res = ((src & 0x00FF'00FF'00FF'00FFULL) << 8) | ((src >> 8) & 0x00FF'00FF'00FF'00FFULL);
            break;
        case 2:
            res = is64 ? ((static_cast<u64>(ByteSwap32(static_cast<u32>(src >> 32))) << 32) |
                          ByteSwap32(static_cast<u32>(src)))
                       : ByteSwap32(static_cast<u32>(src));
            break;
        case 3:
            res = ByteSwap64(src);
            break;
        case 4:
            res = is64 ? std::countl_zero(src) : std::countl_zero(static_cast<u32>(src));
            break;
        case 5: {
            if (is64) {
                const u64 x = (src >> 63) ? ~src : src;
                res = static_cast<unsigned>(std::countl_zero(x)) - 1U;
            } else {
                const u32 x = (src >> 31) ? ~static_cast<u32>(src) : static_cast<u32>(src);
                res = static_cast<unsigned>(std::countl_zero(x)) - 1U;
            }
            break;
        }
        default:
            break;
        }
        WriteRegOrZero(op.rd, res, is64);
        return StopReason::None;
    }

    case OpKind::BranchImm:
        next_pc = op.imm;
        return StopReason::None;

    case OpKind::BranchLinkImm:
        state.x[30] = next_pc;
        next_pc = op.imm;
        return StopReason::None;

    case OpKind::BranchReg: {
        const u64 target = ReadRegOrZero(op.rn, true);
        if (op.aux == 1U) {
            state.x[30] = next_pc;
        }
        next_pc = target;
        return StopReason::None;
    }

    case OpKind::BranchCond:
        if (ConditionHolds(op.ra)) {
            next_pc = op.imm;
        }
        return StopReason::None;

    case OpKind::CompareBranchZero: {
        const u64 val = ReadRegOrZero(op.rd, is64);
        const bool is_nz = set_flags;
        if (is_nz ? (val != 0) : (val == 0)) {
            next_pc = op.imm;
        }
        return StopReason::None;
    }

    case OpKind::FusedSubImmCbnz: {
        const u64 res = (ReadRegOrZero(op.rd, is64) - op.imm) &
                        (is64 ? ~u64{0} : 0xFFFF'FFFFULL);
        WriteRegOrZero(op.rd, res, is64);
        if (set_flags ? (res != 0) : (res == 0)) {
            next_pc = op.imm2;
        }
        return StopReason::None;
    }

    case OpKind::FusedSubImmBcond: {
        const u64 lhs = ReadRegOrSp(op.rn, is64);
        bool n = false, z = false, c = false, v = false;
        const u64 res = AddWithCarry(lhs, ~op.imm, true, is64, n, z, c, v);
        SetNzcv(n, z, c, v);
        WriteRegOrZero(op.rd, res, is64);
        if (ConditionHolds(op.ra)) {
            next_pc = op.imm2;
        }
        return StopReason::None;
    }

    case OpKind::TestBitBranch: {
        const u64 val = ReadRegOrZero(op.rd, true);
        const bool bit_set = ((val >> op.aux) & 1ULL) != 0;
        if (set_flags ? bit_set : !bit_set) {
            next_pc = op.imm;
        }
        return StopReason::None;
    }

    case OpKind::Svc:
        last_svc = static_cast<u32>(op.imm);
        state.pc = next_pc;
        return StopReason::SupervisorCall;

    case OpKind::Brk:
        return StopReason::Breakpoint;

    case OpKind::Mrs: {
        u64 val = 0;
        switch (op.imm) {
        case 0x5A10U:
            val = state.pstate & 0xF000'0000ULL;
            break;
        case 0x5A20U:
            val = state.fpcr;
            break;
        case 0x5A21U:
            val = state.fpsr;
            break;
        case 0x5E82U:
            val = state.tpidr_el0;
            break;
        case 0x5E83U:
            val = state.tpidrro_el0;
            break;
        case 0x5F00U:
            val = bus.GetCntfrq();
            break;
        case 0x5F01U:
        case 0x5F02U:
            val = bus.GetCntvct();
            break;
        case 0x5801U:
            val = 0x8444'C004ULL; // 64B I/D cache lines
            break;
        case 0x5807U:
            val = 4ULL; // 64-byte DC ZVA block
            break;
        default:
            break;
        }
        WriteRegOrZero(op.rd, val, true);
        return StopReason::None;
    }

    case OpKind::Msr:
        if (op.aux == 0xFEU) {
            const u64 base = ReadRegOrZero(op.rd, true) & ~u64{63};
            for (u64 off = 0; off < 64; off += 8) {
                if (!bus.Write64(base + off, 0)) {
                    last_fault_addr = base + off;
                    return StopReason::DataAbort;
                }
            }
            NotifyGuestWrite(base, 64);
            return StopReason::None;
        }
        switch (op.imm) {
        case 0x5A10U:
            state.pstate = (state.pstate & ~0xF000'0000U) |
                           (static_cast<u32>(ReadRegOrZero(op.rd, true)) & 0xF000'0000U);
            break;
        case 0x5A20U:
            state.fpcr = static_cast<u32>(ReadRegOrZero(op.rd, true));
            break;
        case 0x5A21U:
            state.fpsr = static_cast<u32>(ReadRegOrZero(op.rd, true));
            break;
        case 0x5E82U:
            state.tpidr_el0 = ReadRegOrZero(op.rd, true);
            break;
        default:
            break;
        }
        return StopReason::None;

    case OpKind::LoadStoreImm: {
        const unsigned size = op.flags & 3U;
        const unsigned mode = (op.flags >> 2) & 3U; // 0=offset, 1=post, 3=pre
        const u64 base = ReadRegOrSp(op.rn, true);
        const u64 addr = (mode == 1U) ? base : (base + op.imm);
        if (op.aux == 0U) {
            const u64 val = ReadRegOrZero(op.rd, true);
            bool ok = false;
            switch (size) {
            case 0: ok = bus.Write8(addr, static_cast<u8>(val)); break;
            case 1: ok = bus.Write16(addr, static_cast<u16>(val)); break;
            case 2: ok = bus.Write32(addr, static_cast<u32>(val)); break;
            default: ok = bus.Write64(addr, val); break;
            }
            if (!ok) {
                last_fault_addr = addr;
                return StopReason::DataAbort;
            }
            NotifyGuestWrite(addr, std::size_t{1} << size);
        } else {
            u64 val = 0;
            bool ok = false;
            if (size == 0) {
                u8 b = 0;
                ok = bus.Read8(addr, b);
                val = (op.aux >= 2U) ? static_cast<u64>(static_cast<s64>(static_cast<s8>(b))) : b;
            } else if (size == 1) {
                u16 h = 0;
                ok = bus.Read16(addr, h);
                val = (op.aux >= 2U) ? static_cast<u64>(static_cast<s64>(static_cast<s16>(h))) : h;
            } else if (size == 2) {
                u32 w = 0;
                ok = bus.Read32(addr, w);
                val = (op.aux == 2U) ? static_cast<u64>(static_cast<s64>(static_cast<s32>(w))) : w;
            } else {
                ok = bus.Read64(addr, val);
            }
            if (!ok) {
                last_fault_addr = addr;
                return StopReason::DataAbort;
            }
            WriteRegOrZero(op.rd, val, op.aux != 3U && (size == 3U || op.aux == 2U));
        }
        if (mode == 1U || mode == 3U) {
            WriteRegOrSp(op.rn, base + op.imm, true);
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreReg: {
        const unsigned size = op.flags & 3U;
        const unsigned shift = ((op.flags & 8U) != 0) ? size : 0U;
        const u8 option = op.aux >> 2;
        const u8 opc = op.aux & 3U;
        const u64 addr = ReadRegOrSp(op.rn, true) +
                         ExtendValue(ReadRegOrZero(op.rm, true), option, shift, true);
        if (opc == 0U) {
            const u64 val = ReadRegOrZero(op.rd, true);
            bool ok = false;
            switch (size) {
            case 0: ok = bus.Write8(addr, static_cast<u8>(val)); break;
            case 1: ok = bus.Write16(addr, static_cast<u16>(val)); break;
            case 2: ok = bus.Write32(addr, static_cast<u32>(val)); break;
            default: ok = bus.Write64(addr, val); break;
            }
            if (!ok) {
                last_fault_addr = addr;
                return StopReason::DataAbort;
            }
            NotifyGuestWrite(addr, std::size_t{1} << size);
        } else {
            u64 val = 0;
            bool ok = false;
            if (size == 0) {
                u8 b = 0;
                ok = bus.Read8(addr, b);
                val = (opc >= 2U) ? static_cast<u64>(static_cast<s64>(static_cast<s8>(b))) : b;
            } else if (size == 1) {
                u16 h = 0;
                ok = bus.Read16(addr, h);
                val = (opc >= 2U) ? static_cast<u64>(static_cast<s64>(static_cast<s16>(h))) : h;
            } else if (size == 2) {
                u32 w = 0;
                ok = bus.Read32(addr, w);
                val = (opc == 2U) ? static_cast<u64>(static_cast<s64>(static_cast<s32>(w))) : w;
            } else {
                ok = bus.Read64(addr, val);
            }
            if (!ok) {
                last_fault_addr = addr;
                return StopReason::DataAbort;
            }
            WriteRegOrZero(op.rd, val, opc != 3U && (size == 3U || opc == 2U));
        }
        return StopReason::None;
    }

    case OpKind::LoadLiteral: {
        if (op.aux == 1U) {
            u64 v = 0;
            if (!bus.Read64(op.imm, v)) {
                last_fault_addr = op.imm;
                return StopReason::DataAbort;
            }
            WriteRegOrZero(op.rd, v, true);
        } else {
            u32 v = 0;
            if (!bus.Read32(op.imm, v)) {
                last_fault_addr = op.imm;
                return StopReason::DataAbort;
            }
            if (op.aux == 2U) {
                WriteRegOrZero(op.rd, static_cast<u64>(static_cast<s64>(static_cast<s32>(v))), true);
            } else {
                WriteRegOrZero(op.rd, v, false);
            }
        }
        return StopReason::None;
    }

    case OpKind::LoadStorePair: {
        const unsigned size = (op.aux >> 3) & 3U; // 0=32b, 1=LDPSW, 2=64b
        const unsigned mode = (op.aux >> 1) & 3U; // 1=post, 2=offset, 3=pre
        const bool is_load = (op.aux & 1U) != 0;
        const u64 base = ReadRegOrSp(op.rn, true);
        const u64 addr = (mode == 1U) ? base : (base + op.imm);
        const u64 step = (size == 2U) ? 8ULL : 4ULL;
        if (is_load) {
            if (size == 2U) {
                u64 v1 = 0, v2 = 0;
                if (!bus.Read64(addr, v1) || !bus.Read64(addr + 8, v2)) {
                    last_fault_addr = addr;
                    return StopReason::DataAbort;
                }
                WriteRegOrZero(op.rd, v1, true);
                WriteRegOrZero(op.ra, v2, true);
            } else {
                u32 v1 = 0, v2 = 0;
                if (!bus.Read32(addr, v1) || !bus.Read32(addr + 4, v2)) {
                    last_fault_addr = addr;
                    return StopReason::DataAbort;
                }
                if (size == 1U) {
                    WriteRegOrZero(op.rd, static_cast<u64>(static_cast<s64>(static_cast<s32>(v1))),
                                   true);
                    WriteRegOrZero(op.ra, static_cast<u64>(static_cast<s64>(static_cast<s32>(v2))),
                                   true);
                } else {
                    WriteRegOrZero(op.rd, v1, false);
                    WriteRegOrZero(op.ra, v2, false);
                }
            }
        } else {
            if (size == 2U) {
                if (!bus.Write64(addr, ReadRegOrZero(op.rd, true)) ||
                    !bus.Write64(addr + 8, ReadRegOrZero(op.ra, true))) {
                    last_fault_addr = addr;
                    return StopReason::DataAbort;
                }
                NotifyGuestWrite(addr, 16);
            } else {
                if (!bus.Write32(addr, static_cast<u32>(ReadRegOrZero(op.rd, false))) ||
                    !bus.Write32(addr + 4, static_cast<u32>(ReadRegOrZero(op.ra, false)))) {
                    last_fault_addr = addr;
                    return StopReason::DataAbort;
                }
                NotifyGuestWrite(addr, 8);
            }
            (void)step;
        }
        if (mode == 1U || mode == 3U) {
            WriteRegOrSp(op.rn, base + op.imm, true);
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreExclusive: {
        const unsigned size = op.flags & 3U;
        const bool is_load = (op.flags & 4U) != 0;
        const u64 addr = ReadRegOrSp(op.rn, true);
        if (is_load) {
            u64 val = 0;
            bool ok = false;
            switch (size) {
            case 0: { u8 v = 0; ok = bus.ExclusiveRead8(addr, v); val = v; break; }
            case 1: { u16 v = 0; ok = bus.ExclusiveRead16(addr, v); val = v; break; }
            case 2: { u32 v = 0; ok = bus.ExclusiveRead32(addr, v); val = v; break; }
            default: ok = bus.ExclusiveRead64(addr, val); break;
            }
            if (!ok) {
                last_fault_addr = addr;
                return StopReason::DataAbort;
            }
            WriteRegOrZero(op.rd, val, size == 3U);
        } else {
            const u64 val = ReadRegOrZero(op.rd, true);
            bool success = false;
            switch (size) {
            case 0: success = bus.ExclusiveWrite8(addr, static_cast<u8>(val)); break;
            case 1: success = bus.ExclusiveWrite16(addr, static_cast<u16>(val)); break;
            case 2: success = bus.ExclusiveWrite32(addr, static_cast<u32>(val)); break;
            default: success = bus.ExclusiveWrite64(addr, val); break;
            }
            WriteRegOrZero(op.rm, success ? 0ULL : 1ULL, false);
            if (success) {
                NotifyGuestWrite(addr, std::size_t{1} << size);
            }
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreExclusivePair: {
        const bool is_load = (op.flags & 4U) != 0;
        const u64 addr = ReadRegOrSp(op.rn, true);
        if (is_load) {
            if (is64) {
                u128 v{};
                if (!bus.ExclusiveRead128(addr, v)) {
                    last_fault_addr = addr;
                    return StopReason::DataAbort;
                }
                WriteRegOrZero(op.rd, v[0], true);
                WriteRegOrZero(op.ra, v[1], true);
            } else {
                u64 v = 0;
                if (!bus.ExclusiveRead64(addr, v)) {
                    last_fault_addr = addr;
                    return StopReason::DataAbort;
                }
                WriteRegOrZero(op.rd, static_cast<u32>(v), false);
                WriteRegOrZero(op.ra, static_cast<u32>(v >> 32), false);
            }
        } else {
            bool success = false;
            if (is64) {
                const u128 v{ReadRegOrZero(op.rd, true), ReadRegOrZero(op.ra, true)};
                success = bus.ExclusiveWrite128(addr, v);
                if (success) {
                    NotifyGuestWrite(addr, 16);
                }
            } else {
                const u64 v = ReadRegOrZero(op.rd, false) | (ReadRegOrZero(op.ra, false) << 32);
                success = bus.ExclusiveWrite64(addr, v);
                if (success) {
                    NotifyGuestWrite(addr, 8);
                }
            }
            WriteRegOrZero(op.rm, success ? 0ULL : 1ULL, false);
        }
        return StopReason::None;
    }

    case OpKind::AtomicRmw: {
        const unsigned size = op.flags & 3U;
        const u64 addr = ReadRegOrSp(op.rn, true);
        for (int attempt = 0; attempt < 64; ++attempt) {
            u64 old_val = 0;
            bool ok = false;
            switch (size) {
            case 0: { u8 v = 0; ok = bus.ExclusiveRead8(addr, v); old_val = v; break; }
            case 1: { u16 v = 0; ok = bus.ExclusiveRead16(addr, v); old_val = v; break; }
            case 2: { u32 v = 0; ok = bus.ExclusiveRead32(addr, v); old_val = v; break; }
            default: ok = bus.ExclusiveRead64(addr, old_val); break;
            }
            if (!ok) {
                last_fault_addr = addr;
                return StopReason::DataAbort;
            }
            if (op.aux == 0xFFU) {
                const u64 mask = (size == 3U) ? ~u64{0} : ((u64{1} << ((1U << size) * 8U)) - 1U);
                const u64 expected = ReadRegOrZero(op.rm, true) & mask;
                if ((old_val & mask) != expected) {
                    bus.ClearExclusive();
                    WriteRegOrZero(op.rm, old_val, size == 3U);
                    return StopReason::None;
                }
                const u64 desired = ReadRegOrZero(op.rd, true);
                bool wrote = false;
                switch (size) {
                case 0: wrote = bus.ExclusiveWrite8(addr, static_cast<u8>(desired)); break;
                case 1: wrote = bus.ExclusiveWrite16(addr, static_cast<u16>(desired)); break;
                case 2: wrote = bus.ExclusiveWrite32(addr, static_cast<u32>(desired)); break;
                default: wrote = bus.ExclusiveWrite64(addr, desired); break;
                }
                if (wrote) {
                    WriteRegOrZero(op.rm, old_val, size == 3U);
                    NotifyGuestWrite(addr, std::size_t{1} << size);
                    return StopReason::None;
                }
            } else {
                const u64 arg = ReadRegOrZero(op.rm, true);
                u64 next_val = old_val;
                switch (op.aux) {
                case 0: next_val = old_val + arg; break;
                case 1: next_val = old_val & ~arg; break;
                case 2: next_val = old_val ^ arg; break;
                case 3: next_val = old_val | arg; break;
                default: next_val = arg; break; // SWP
                }
                bool wrote = false;
                switch (size) {
                case 0: wrote = bus.ExclusiveWrite8(addr, static_cast<u8>(next_val)); break;
                case 1: wrote = bus.ExclusiveWrite16(addr, static_cast<u16>(next_val)); break;
                case 2: wrote = bus.ExclusiveWrite32(addr, static_cast<u32>(next_val)); break;
                default: wrote = bus.ExclusiveWrite64(addr, next_val); break;
                }
                if (wrote) {
                    WriteRegOrZero(op.rd, old_val, size == 3U);
                    NotifyGuestWrite(addr, std::size_t{1} << size);
                    return StopReason::None;
                }
            }
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreFpImm: {
        const unsigned mode = (op.aux >> 5) & 3U;
        const unsigned scale = (op.aux >> 2) & 7U;
        const bool is_load = (op.aux & 1U) != 0;
        const u64 base = ReadRegOrSp(op.rn, true);
        const u64 addr = (mode == 1U) ? base : (base + op.imm);
        bool ok = false;
        if (is_load) {
            state.v[op.rd] = {0ULL, 0ULL};
            switch (scale) {
            case 0: { u8 v = 0; ok = bus.Read8(addr, v); state.v[op.rd][0] = v; break; }
            case 1: { u16 v = 0; ok = bus.Read16(addr, v); state.v[op.rd][0] = v; break; }
            case 2: { u32 v = 0; ok = bus.Read32(addr, v); state.v[op.rd][0] = v; break; }
            case 3: ok = bus.Read64(addr, state.v[op.rd][0]); break;
            default: ok = bus.Read128(addr, state.v[op.rd]); break;
            }
        } else {
            switch (scale) {
            case 0: ok = bus.Write8(addr, static_cast<u8>(state.v[op.rd][0])); break;
            case 1: ok = bus.Write16(addr, static_cast<u16>(state.v[op.rd][0])); break;
            case 2: ok = bus.Write32(addr, static_cast<u32>(state.v[op.rd][0])); break;
            case 3: ok = bus.Write64(addr, state.v[op.rd][0]); break;
            default: ok = bus.Write128(addr, state.v[op.rd]); break;
            }
            if (ok) {
                NotifyGuestWrite(addr, std::size_t{1} << scale);
            }
        }
        if (!ok) {
            last_fault_addr = addr;
            return StopReason::DataAbort;
        }
        if (mode == 1U || mode == 3U) {
            WriteRegOrSp(op.rn, base + op.imm, true);
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreFpReg: {
        const unsigned scale = (op.aux >> 4) & 7U;
        const u8 option = (op.aux >> 1) & 7U;
        const bool is_load = (op.aux & 1U) != 0;
        const unsigned shift = (op.flags != 0) ? scale : 0U;
        const u64 addr = ReadRegOrSp(op.rn, true) +
                         ExtendValue(ReadRegOrZero(op.rm, true), option, shift, true);
        bool ok = false;
        if (is_load) {
            state.v[op.rd] = {0ULL, 0ULL};
            switch (scale) {
            case 0: { u8 v = 0; ok = bus.Read8(addr, v); state.v[op.rd][0] = v; break; }
            case 1: { u16 v = 0; ok = bus.Read16(addr, v); state.v[op.rd][0] = v; break; }
            case 2: { u32 v = 0; ok = bus.Read32(addr, v); state.v[op.rd][0] = v; break; }
            case 3: ok = bus.Read64(addr, state.v[op.rd][0]); break;
            default: ok = bus.Read128(addr, state.v[op.rd]); break;
            }
        } else {
            switch (scale) {
            case 0: ok = bus.Write8(addr, static_cast<u8>(state.v[op.rd][0])); break;
            case 1: ok = bus.Write16(addr, static_cast<u16>(state.v[op.rd][0])); break;
            case 2: ok = bus.Write32(addr, static_cast<u32>(state.v[op.rd][0])); break;
            case 3: ok = bus.Write64(addr, state.v[op.rd][0]); break;
            default: ok = bus.Write128(addr, state.v[op.rd]); break;
            }
            if (ok) {
                NotifyGuestWrite(addr, std::size_t{1} << scale);
            }
        }
        if (!ok) {
            last_fault_addr = addr;
            return StopReason::DataAbort;
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreFpLiteral: {
        state.v[op.rd] = {0ULL, 0ULL};
        bool ok = false;
        if (op.aux == 0U) {
            u32 v = 0;
            ok = bus.Read32(op.imm, v);
            state.v[op.rd][0] = v;
        } else if (op.aux == 1U) {
            ok = bus.Read64(op.imm, state.v[op.rd][0]);
        } else {
            ok = bus.Read128(op.imm, state.v[op.rd]);
        }
        if (!ok) {
            last_fault_addr = op.imm;
            return StopReason::DataAbort;
        }
        return StopReason::None;
    }

    case OpKind::LoadStoreFpPair: {
        const unsigned size = (op.aux >> 3) & 3U; // 0=4B, 1=8B, 2=16B
        const unsigned mode = (op.aux >> 1) & 3U;
        const bool is_load = (op.aux & 1U) != 0;
        const u64 base = ReadRegOrSp(op.rn, true);
        const u64 addr = (mode == 1U) ? base : (base + op.imm);
        bool ok = false;
        if (is_load) {
            state.v[op.rd] = {0ULL, 0ULL};
            state.v[op.ra] = {0ULL, 0ULL};
            if (size == 0U) {
                u32 a = 0, b = 0;
                ok = bus.Read32(addr, a) && bus.Read32(addr + 4, b);
                state.v[op.rd][0] = a;
                state.v[op.ra][0] = b;
            } else if (size == 1U) {
                ok = bus.Read64(addr, state.v[op.rd][0]) && bus.Read64(addr + 8, state.v[op.ra][0]);
            } else {
                ok = bus.Read128(addr, state.v[op.rd]) && bus.Read128(addr + 16, state.v[op.ra]);
            }
        } else {
            if (size == 0U) {
                ok = bus.Write32(addr, static_cast<u32>(state.v[op.rd][0])) &&
                     bus.Write32(addr + 4, static_cast<u32>(state.v[op.ra][0]));
                if (ok) NotifyGuestWrite(addr, 8);
            } else if (size == 1U) {
                ok = bus.Write64(addr, state.v[op.rd][0]) &&
                     bus.Write64(addr + 8, state.v[op.ra][0]);
                if (ok) NotifyGuestWrite(addr, 16);
            } else {
                ok = bus.Write128(addr, state.v[op.rd]) &&
                     bus.Write128(addr + 16, state.v[op.ra]);
                if (ok) NotifyGuestWrite(addr, 32);
            }
        }
        if (!ok) {
            last_fault_addr = addr;
            return StopReason::DataAbort;
        }
        if (mode == 1U || mode == 3U) {
            WriteRegOrSp(op.rn, base + op.imm, true);
        }
        return StopReason::None;
    }

    case OpKind::FpMoveReg:
        if (op.aux == 2U) {
            state.v[op.rd] = state.v[op.rn];
        } else if (is64) {
            state.v[op.rd] = {state.v[op.rn][0], 0ULL};
        } else {
            state.v[op.rd] = {state.v[op.rn][0] & 0xFFFF'FFFFULL, 0ULL};
        }
        return StopReason::None;

    case OpKind::FpMoveGpr:
        if ((op.flags & 4U) != 0) {
            state.v[op.rd] = {ReadRegOrZero(op.rn, is64), 0ULL};
        } else {
            WriteRegOrZero(op.rd, state.v[op.rn][0], is64);
        }
        return StopReason::None;

    case OpKind::FpMoveImm:
        if (op.aux == 0xFEU) {
            state.v[op.rd] = {0ULL, 0ULL};
        } else if (is64) {
            const double val = DecodeFp8Imm(op.aux);
            state.v[op.rd] = {std::bit_cast<u64>(val), 0ULL};
        } else {
            const float val = static_cast<float>(DecodeFp8Imm(op.aux));
            state.v[op.rd] = {std::bit_cast<u32>(val), 0ULL};
        }
        return StopReason::None;

    case OpKind::FpUnary: {
        if (is64) {
            const double x = std::bit_cast<double>(state.v[op.rn][0]);
            switch (op.aux) {
            case 1: state.v[op.rd] = {std::bit_cast<u64>(std::fabs(x)), 0ULL}; break;
            case 2: state.v[op.rd] = {std::bit_cast<u64>(-x), 0ULL}; break;
            case 3: state.v[op.rd] = {std::bit_cast<u64>(std::sqrt(x)), 0ULL}; break;
            case 4: state.v[op.rd] = {std::bit_cast<u32>(static_cast<float>(x)), 0ULL}; break; // FCVT S, D
            case 8: state.v[op.rd] = {std::bit_cast<u64>(std::nearbyint(x)), 0ULL}; break;
            case 9: state.v[op.rd] = {std::bit_cast<u64>(std::ceil(x)), 0ULL}; break;
            case 10: state.v[op.rd] = {std::bit_cast<u64>(std::floor(x)), 0ULL}; break;
            case 11: state.v[op.rd] = {std::bit_cast<u64>(std::trunc(x)), 0ULL}; break;
            case 12: state.v[op.rd] = {std::bit_cast<u64>(std::round(x)), 0ULL}; break;
            default: return StopReason::UnsupportedInstruction;
            }
        } else {
            const float x = std::bit_cast<float>(static_cast<u32>(state.v[op.rn][0]));
            switch (op.aux) {
            case 1: state.v[op.rd] = {std::bit_cast<u32>(std::fabs(x)), 0ULL}; break;
            case 2: state.v[op.rd] = {std::bit_cast<u32>(-x), 0ULL}; break;
            case 3: state.v[op.rd] = {std::bit_cast<u32>(std::sqrt(x)), 0ULL}; break;
            case 5: state.v[op.rd] = {std::bit_cast<u64>(static_cast<double>(x)), 0ULL}; break; // FCVT D, S
            case 8: state.v[op.rd] = {std::bit_cast<u32>(std::nearbyint(x)), 0ULL}; break;
            case 9: state.v[op.rd] = {std::bit_cast<u32>(std::ceil(x)), 0ULL}; break;
            case 10: state.v[op.rd] = {std::bit_cast<u32>(std::floor(x)), 0ULL}; break;
            case 11: state.v[op.rd] = {std::bit_cast<u32>(std::trunc(x)), 0ULL}; break;
            case 12: state.v[op.rd] = {std::bit_cast<u32>(std::round(x)), 0ULL}; break;
            default: return StopReason::UnsupportedInstruction;
            }
        }
        return StopReason::None;
    }

    case OpKind::FpBinary: {
        if (is64) {
            const double a = std::bit_cast<double>(state.v[op.rn][0]);
            const double b = std::bit_cast<double>(state.v[op.rm][0]);
            double r = 0.0;
            switch (op.aux) {
            case 0: r = a * b; break;
            case 1: r = a / b; break;
            case 2: r = a + b; break;
            case 3: r = a - b; break;
            case 4: r = (std::isnan(a) || std::isnan(b)) ? (a + b) : std::fmax(a, b); break;
            case 5: r = (std::isnan(a) || std::isnan(b)) ? (a + b) : std::fmin(a, b); break;
            case 6: r = std::fmax(a, b); break;
            case 7: r = std::fmin(a, b); break;
            case 8: r = -(a * b); break;
            default: return StopReason::UnsupportedInstruction;
            }
            state.v[op.rd] = {std::bit_cast<u64>(r), 0ULL};
        } else {
            const float a = std::bit_cast<float>(static_cast<u32>(state.v[op.rn][0]));
            const float b = std::bit_cast<float>(static_cast<u32>(state.v[op.rm][0]));
            float r = 0.0f;
            switch (op.aux) {
            case 0: r = a * b; break;
            case 1: r = a / b; break;
            case 2: r = a + b; break;
            case 3: r = a - b; break;
            case 4: r = (std::isnan(a) || std::isnan(b)) ? (a + b) : std::fmax(a, b); break;
            case 5: r = (std::isnan(a) || std::isnan(b)) ? (a + b) : std::fmin(a, b); break;
            case 6: r = std::fmax(a, b); break;
            case 7: r = std::fmin(a, b); break;
            case 8: r = -(a * b); break;
            default: return StopReason::UnsupportedInstruction;
            }
            state.v[op.rd] = {std::bit_cast<u32>(r), 0ULL};
        }
        return StopReason::None;
    }

    case OpKind::FpFma: {
        if (is64) {
            const double n = std::bit_cast<double>(state.v[op.rn][0]);
            const double m = std::bit_cast<double>(state.v[op.rm][0]);
            const double a = std::bit_cast<double>(state.v[op.ra][0]);
            double r = 0.0;
            switch (op.aux) {
            case 0: r = std::fma(n, m, a); break;
            case 1: r = std::fma(-n, m, a); break;
            case 2: r = -std::fma(n, m, a); break;
            default: r = std::fma(n, m, -a); break;
            }
            state.v[op.rd] = {std::bit_cast<u64>(r), 0ULL};
        } else {
            const float n = std::bit_cast<float>(static_cast<u32>(state.v[op.rn][0]));
            const float m = std::bit_cast<float>(static_cast<u32>(state.v[op.rm][0]));
            const float a = std::bit_cast<float>(static_cast<u32>(state.v[op.ra][0]));
            float r = 0.0f;
            switch (op.aux) {
            case 0: r = std::fma(n, m, a); break;
            case 1: r = std::fma(-n, m, a); break;
            case 2: r = -std::fma(n, m, a); break;
            default: r = std::fma(n, m, -a); break;
            }
            state.v[op.rd] = {std::bit_cast<u32>(r), 0ULL};
        }
        return StopReason::None;
    }

    case OpKind::FpCompare: {
        const bool cmp_zero = (op.aux & 1U) != 0;
        const double a = is64 ? std::bit_cast<double>(state.v[op.rn][0])
                              : static_cast<double>(
                                    std::bit_cast<float>(static_cast<u32>(state.v[op.rn][0])));
        const double b = cmp_zero
                             ? 0.0
                             : (is64 ? std::bit_cast<double>(state.v[op.rm][0])
                                     : static_cast<double>(std::bit_cast<float>(
                                           static_cast<u32>(state.v[op.rm][0]))));
        if (std::isnan(a) || std::isnan(b)) {
            SetNzcv(false, false, true, true);
        } else if (a == b) {
            SetNzcv(false, true, true, false);
        } else if (a < b) {
            SetNzcv(true, false, false, false);
        } else {
            SetNzcv(false, false, true, false);
        }
        return StopReason::None;
    }

    case OpKind::FpCondSelect: {
        const u8 src = ConditionHolds(op.ra) ? op.rn : op.rm;
        if (is64) {
            state.v[op.rd] = {state.v[src][0], 0ULL};
        } else {
            state.v[op.rd] = {state.v[src][0] & 0xFFFF'FFFFULL, 0ULL};
        }
        return StopReason::None;
    }

    case OpKind::FpConvert: {
        const bool is_dbl = (op.flags & 4U) != 0;
        const u32 rmode = (op.aux >> 3) & 3U;
        const u32 opcode = op.aux & 7U;
        if (opcode == 2U || opcode == 3U) {
            // SCVTF / UCVTF
            if (opcode == 2U) {
                const s64 iv = is64 ? static_cast<s64>(ReadRegOrZero(op.rn, true))
                                    : static_cast<s64>(static_cast<s32>(ReadRegOrZero(op.rn, false)));
                if (is_dbl) {
                    state.v[op.rd] = {std::bit_cast<u64>(static_cast<double>(iv)), 0ULL};
                } else {
                    state.v[op.rd] = {std::bit_cast<u32>(static_cast<float>(iv)), 0ULL};
                }
            } else {
                const u64 uv = ReadRegOrZero(op.rn, is64);
                if (is_dbl) {
                    state.v[op.rd] = {std::bit_cast<u64>(static_cast<double>(uv)), 0ULL};
                } else {
                    state.v[op.rd] = {std::bit_cast<u32>(static_cast<float>(uv)), 0ULL};
                }
            }
            return StopReason::None;
        }
        double fv = is_dbl ? std::bit_cast<double>(state.v[op.rn][0])
                           : static_cast<double>(
                                 std::bit_cast<float>(static_cast<u32>(state.v[op.rn][0])));
        if (std::isnan(fv)) {
            WriteRegOrZero(op.rd, 0, is64);
            return StopReason::None;
        }
        if (opcode >= 4U) {
            fv = std::round(fv); // FCVTAS / FCVTAU
        } else {
            switch (rmode) {
            case 0: fv = std::nearbyint(fv); break; // N
            case 1: fv = std::ceil(fv); break;      // P
            case 2: fv = std::floor(fv); break;     // M
            default: fv = std::trunc(fv); break;    // Z
            }
        }
        if ((opcode & 1U) == 0U) {
            // Signed
            if (is64) {
                s64 out = 0;
                if (fv >= static_cast<double>(std::numeric_limits<s64>::max())) {
                    out = std::numeric_limits<s64>::max();
                } else if (fv <= static_cast<double>(std::numeric_limits<s64>::min())) {
                    out = std::numeric_limits<s64>::min();
                } else {
                    out = static_cast<s64>(fv);
                }
                WriteRegOrZero(op.rd, static_cast<u64>(out), true);
            } else {
                s32 out = 0;
                if (fv >= static_cast<double>(std::numeric_limits<s32>::max())) {
                    out = std::numeric_limits<s32>::max();
                } else if (fv <= static_cast<double>(std::numeric_limits<s32>::min())) {
                    out = std::numeric_limits<s32>::min();
                } else {
                    out = static_cast<s32>(fv);
                }
                WriteRegOrZero(op.rd, static_cast<u32>(out), false);
            }
        } else {
            // Unsigned
            if (fv <= 0.0) {
                WriteRegOrZero(op.rd, 0, is64);
            } else if (is64) {
                const u64 out = (fv >= 18446744073709551616.0)
                                    ? std::numeric_limits<u64>::max()
                                    : static_cast<u64>(fv);
                WriteRegOrZero(op.rd, out, true);
            } else {
                const u32 out = (fv >= 4294967296.0)
                                    ? std::numeric_limits<u32>::max()
                                    : static_cast<u32>(fv);
                WriteRegOrZero(op.rd, out, false);
            }
        }
        return StopReason::None;
    }

    case OpKind::Unsupported:
    default:
        last_unsupported_pc = state.pc;
        last_unsupported_insn = op.raw;
        return StopReason::UnsupportedInstruction;
    }
}

StopReason CachedBlockEngine::Step() {
    if ((state.pc & 3ULL) != 0) {
        return StopReason::PrefetchAbort;
    }
    const auto word = bus.ReadCode32(state.pc);
    if (!word.has_value()) {
        return StopReason::PrefetchAbort;
    }
    const DecodedOp op = DecodeInstruction(*word, state.pc);
    u64 next_pc = state.pc + 4;
    const StopReason stop = ExecuteOp(op, next_pc);
    if (stop == StopReason::None) {
        state.pc = next_pc;
        ++stats.instructions_retired;
        return StopReason::StepCompleted;
    }
    if (stop == StopReason::SupervisorCall) {
        ++stats.instructions_retired;
    }
    return stop;
}

StopReason CachedBlockEngine::Run(u64 max_instructions) {
    const bool bounded = max_instructions != 0;
    u64 remaining = max_instructions;
    CachedBlock scratch{};

    while (!bounded || remaining != 0) {
        if (interrupt_requested.load(std::memory_order_acquire)) {
            interrupt_requested.store(false, std::memory_order_relaxed);
            return StopReason::Interrupted;
        }
        if ((state.pc & 3ULL) != 0) {
            return StopReason::PrefetchAbort;
        }

        CachedBlock* block = &scratch;
        if (cache_enabled) {
            block = &cache[CacheIndex(state.pc)];
            if (block->generation == generation && block->op_count != 0 &&
                block->start_pc == state.pc) {
                ++stats.cache_hits;
            } else {
                ++stats.cache_misses;
                if (!DecodeBlock(*block, state.pc, fusion_enabled)) {
                    return StopReason::PrefetchAbort;
                }
            }
        } else {
            if (!DecodeBlock(scratch, state.pc, fusion_enabled)) {
                return StopReason::PrefetchAbort;
            }
        }

        // If a tight instruction budget would expire mid-block, step exact instructions.
        if (bounded && remaining < block->guest_insns) {
            while (remaining != 0) {
                if (interrupt_requested.load(std::memory_order_acquire)) {
                    interrupt_requested.store(false, std::memory_order_relaxed);
                    return StopReason::Interrupted;
                }
                const StopReason sr = Step();
                if (sr != StopReason::StepCompleted) {
                    return sr;
                }
                --remaining;
            }
            return StopReason::StepCompleted;
        }

        const u16 count = block->op_count;
        const u32 block_gen = block->generation;
        for (u16 i = 0; i < count; ++i) {
            // If a prior store invalidated this cached block mid-flight, re-enter outer loop cleanly.
            if (cache_enabled && (block->op_count == 0 || block->generation != block_gen)) {
                break;
            }
            const DecodedOp op = block->ops[i];
            u64 next_pc = state.pc + 4ULL * op.insn_count;
            const StopReason stop = ExecuteOp(op, next_pc);
            if (stop != StopReason::None) {
                if (stop == StopReason::SupervisorCall) {
                    stats.instructions_retired += op.insn_count;
                }
                return stop;
            }
            state.pc = next_pc;
            stats.instructions_retired += op.insn_count;
            if (bounded) {
                remaining -= op.insn_count;
            }
        }
    }

    return StopReason::StepCompleted;
}

} // namespace Core::CachedInterpreter
