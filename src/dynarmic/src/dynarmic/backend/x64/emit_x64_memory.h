// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <bit>
#include <cstdlib>
#include <utility>
#include "dynarmic/backend/x64/xbyak.h"

#include "dynarmic/backend/x64/a32_emit_x64.h"
#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/common/spin_lock_x64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/acc_type.h"

namespace Dynarmic::Backend::X64 {

namespace {

using namespace Xbyak::util;

constexpr size_t page_table_const_bits = 12;
constexpr size_t page_table_const_size = 1 << page_table_const_bits;
constexpr size_t page_table_const_mask = (1 << page_table_const_bits) - 1;

/// Where cold deferred code goes: the far region when there is one.
template<typename Ctx>
std::vector<std::function<void()>>& ColdEmits(BlockOfCode& code, Ctx& ctx) {
    return code.HasFarCode() ? ctx.far_deferred_emits : ctx.deferred_emits;
}

template<typename EmitContext>
void EmitDetectMisalignedVAddr(BlockOfCode& code, EmitContext& ctx, size_t bitsize, Xbyak::Label& abort, Xbyak::Reg64 vaddr, Xbyak::Reg64 tmp) {
    if (bitsize == 8 || (ctx.conf.detect_misaligned_access_via_page_table & bitsize) == 0) {
        return;
    }

    const u32 align_mask = [bitsize]() -> u32 {
        switch (bitsize) {
        case 16:
            return 0b1;
        case 32:
            return 0b11;
        case 64:
            return 0b111;
        case 128:
            return 0b1111;
        default:
            UNREACHABLE();
        }
    }();

    code.test(vaddr, align_mask);

    if (ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary) {
        const u32 page_align_mask = static_cast<u32>(page_table_const_size - 1) & ~align_mask;

        SharedLabel detect_boundary = ctx.GenSharedLabel(), resume = ctx.GenSharedLabel();

        code.jnz(*detect_boundary, code.T_NEAR);
        code.L(*resume);

        ColdEmits(code, ctx).emplace_back([=, &code] {
            code.L(*detect_boundary);
            code.mov(tmp, vaddr);
            code.and_(tmp, page_align_mask);
            code.cmp(tmp, page_align_mask);
            code.jne(*resume, code.T_NEAR);
            // NOTE: We expect to fallthrough into abort code here.
        });
    } else {
        code.jnz(abort, code.T_NEAR);
    }
}

struct UntrappedCachedReadX64 {
    SharedLabel check_cached{};
    SharedLabel resume_unmarked{};
    Xbyak::Reg64 page{};
    Xbyak::Reg64 tmp{};
    Xbyak::Reg64 tmp2{};
};

template<typename EmitContext>
void EmitUntrappedCachedReadCheckX64(BlockOfCode&, EmitContext&, Xbyak::Reg64, const UntrappedCachedReadX64&, Xbyak::Label&) {}

template<>
[[maybe_unused]] void EmitUntrappedCachedReadCheckX64<A64EmitContext>(
    BlockOfCode& code, A64EmitContext& ctx, Xbyak::Reg64 vaddr,
    const UntrappedCachedReadX64& cached, Xbyak::Label& abort) {
    if (!cached.check_cached) {
        return;
    }
    const size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_table_const_bits;
    const size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;

    code.L(*cached.check_cached);
    code.mov(cached.tmp.cvt32(), cached.page.cvt32());
    code.and_(cached.tmp.cvt32(), 7);
    code.cmp(cached.tmp.cvt32(), 7);
    code.jne(abort, code.T_NEAR);

    code.mov(cached.tmp, ctx.conf.untrapped_read_bitmap_ptr);
    code.mov(cached.tmp, qword[cached.tmp]);
    code.test(cached.tmp, cached.tmp);
    code.jz(abort, code.T_NEAR);
    code.cmp(cached.tmp, 1);
    code.je(*cached.resume_unmarked, code.T_NEAR);

    code.mov(cached.tmp2, vaddr);
    if (unused_top_bits == 0) {
        code.shr(cached.tmp2, int(page_table_const_bits));
    } else if (ctx.conf.silently_mirror_page_table) {
        code.shl(cached.tmp2, int(unused_top_bits));
        code.shr(cached.tmp2, int(unused_top_bits + page_table_const_bits));
    } else {
        code.shr(cached.tmp2, int(page_table_const_bits));
    }

    if (valid_page_index_bits > 27) {
        code.shr(cached.tmp2, 27);
        code.jnz(*cached.resume_unmarked, code.T_NEAR);
        code.mov(cached.tmp2, vaddr);
        if (ctx.conf.silently_mirror_page_table && unused_top_bits != 0) {
            code.shl(cached.tmp2, int(unused_top_bits));
            code.shr(cached.tmp2, int(unused_top_bits + page_table_const_bits + 6));
        } else {
            code.shr(cached.tmp2, int(page_table_const_bits + 6));
        }
    } else {
        code.shr(cached.tmp2, 6);
    }

    code.mov(cached.tmp, qword[cached.tmp + cached.tmp2 * 8]);
    code.mov(cached.tmp2, vaddr);
    code.shr(cached.tmp2, int(page_table_const_bits));
    code.bt(cached.tmp, cached.tmp2);
    code.jc(abort, code.T_NEAR);
    code.jmp(*cached.resume_unmarked, code.T_NEAR);
}

template<typename EmitContext>
Xbyak::RegExp EmitVAddrLookup(BlockOfCode& code, EmitContext& ctx, size_t bitsize, Xbyak::Label& abort, Xbyak::Reg64 vaddr, UntrappedCachedReadX64* out_cached = nullptr);

template<>
[[maybe_unused]] Xbyak::RegExp EmitVAddrLookup<A32EmitContext>(BlockOfCode& code, A32EmitContext& ctx, size_t bitsize, Xbyak::Label& abort, Xbyak::Reg64 vaddr, UntrappedCachedReadX64*) {
    const Xbyak::Reg64 page = ctx.reg_alloc.ScratchGpr(code);
    const Xbyak::Reg64 tmp = ctx.conf.absolute_offset_page_table && ctx.conf.page_table_pointer_mask == 0 ? page : ctx.reg_alloc.ScratchGpr(code);

    EmitDetectMisalignedVAddr(code, ctx, bitsize, abort, vaddr, tmp);

    code.mov(tmp, vaddr);
    code.shr(tmp, int(page_table_const_bits));

    if (ctx.conf.page_table_log2_stride > 3) {
        code.shl(tmp, int(ctx.conf.page_table_log2_stride));
        code.mov(page, qword[r14 + tmp.cvt64()]);
    } else {
        code.mov(page, qword[r14 + tmp.cvt64() * int(1 << ctx.conf.page_table_log2_stride)]);
    }

    // check for marked bit, use as unmapped if marked
	if (ctx.conf.page_table_marked_bit) {
        code.bt(page, *ctx.conf.page_table_marked_bit);
        code.jc(abort, code.T_NEAR);
	}
    // mask away attributes
    if (ctx.conf.page_table_pointer_mask == 0) {
        code.test(page, page);
    } else if (std::in_range<s32>(ctx.conf.page_table_pointer_mask)) {
        code.and_(page, ctx.conf.page_table_pointer_mask);
    } else {
        code.mov(tmp, ctx.conf.page_table_pointer_mask);
        code.and_(page, tmp);
    }
    // check for sign bit, apply sign extension as needed
    if (ctx.conf.page_table_sign_extension) {
        if (const int shift = 63 - int(*ctx.conf.page_table_sign_extension); shift > 0) {
            code.shl(page, shift);
            code.sar(page, shift);
        }
    }

    code.jz(abort, code.T_NEAR);
    if (ctx.conf.absolute_offset_page_table) {
        return page + vaddr;
    }
    code.mov(tmp, vaddr);
    code.and_(tmp, u32(page_table_const_mask));
    return page + tmp.cvt64();
}

template<>
[[maybe_unused]] Xbyak::RegExp EmitVAddrLookup<A64EmitContext>(BlockOfCode& code, A64EmitContext& ctx, size_t bitsize, Xbyak::Label& abort, Xbyak::Reg64 vaddr, UntrappedCachedReadX64* out_cached) {
    const size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_table_const_bits;
    const size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;
    const bool use_cached_read = out_cached != nullptr && ctx.conf.untrapped_read_bitmap_ptr != 0 &&
                                 ctx.conf.page_table_marked_bit && *ctx.conf.page_table_marked_bit == 0;

    const Xbyak::Reg64 page = ctx.reg_alloc.ScratchGpr(code);
    const Xbyak::Reg64 tmp = (ctx.conf.absolute_offset_page_table && ctx.conf.page_table_pointer_mask == 0 && !use_cached_read)
                                 ? page
                                 : ctx.reg_alloc.ScratchGpr(code);
    std::optional<Xbyak::Reg64> tmp2;
    if (use_cached_read) {
        tmp2 = ctx.reg_alloc.ScratchGpr(code);
    }

    EmitDetectMisalignedVAddr(code, ctx, bitsize, abort, vaddr, tmp);

    if (unused_top_bits == 0) {
        code.mov(tmp, vaddr);
        code.shr(tmp, int(page_table_const_bits));
    } else if (ctx.conf.silently_mirror_page_table) {
        if (valid_page_index_bits >= 32) {
            if (code.HasHostFeature(HostFeature::BMI2)) {
                const Xbyak::Reg64 bit_count = ctx.reg_alloc.ScratchGpr(code);
                code.mov(bit_count, unused_top_bits);
                code.bzhi(tmp, vaddr, bit_count);
                code.shr(tmp, int(page_table_const_bits));
                ctx.reg_alloc.Release(bit_count);
            } else {
                code.mov(tmp, vaddr);
                code.shl(tmp, int(unused_top_bits));
                code.shr(tmp, int(unused_top_bits + page_table_const_bits));
            }
        } else {
            code.mov(tmp, vaddr);
            code.shr(tmp, int(page_table_const_bits));
            code.and_(tmp, u32((1 << valid_page_index_bits) - 1));
        }
    } else {
        ASSERT(valid_page_index_bits < 32);
        code.mov(tmp, vaddr);
        code.shr(tmp, int(page_table_const_bits));
        code.test(tmp, u32(-(1 << valid_page_index_bits)));
        code.jnz(abort, code.T_NEAR);
    }

    if (ctx.conf.page_table_log2_stride > 3) {
        code.shl(tmp, int(ctx.conf.page_table_log2_stride));
        code.mov(page, qword[r14 + tmp.cvt64()]);
    } else {
        code.mov(page, qword[r14 + tmp.cvt64() * int(1 << ctx.conf.page_table_log2_stride)]);
    }

    // check for marked bit, use as unmapped if marked
    if (ctx.conf.page_table_marked_bit) {
        if (use_cached_read) {
            out_cached->check_cached = ctx.GenSharedLabel();
            out_cached->resume_unmarked = ctx.GenSharedLabel();
            out_cached->page = page;
            out_cached->tmp = tmp;
            out_cached->tmp2 = *tmp2;
            code.bt(page, *ctx.conf.page_table_marked_bit);
            code.jc(*out_cached->check_cached, code.T_NEAR);
            code.L(*out_cached->resume_unmarked);
        } else {
            code.bt(page, *ctx.conf.page_table_marked_bit);
            code.jc(abort, code.T_NEAR);
        }
    }
    if (tmp2) {
        ctx.reg_alloc.Release(*tmp2);
    }
    // mask away attributes
    if (ctx.conf.page_table_pointer_mask == 0) {
        code.test(page, page);
    } else if (std::in_range<s32>(ctx.conf.page_table_pointer_mask)) {
        code.and_(page, ctx.conf.page_table_pointer_mask);
    } else {
        code.mov(tmp, ctx.conf.page_table_pointer_mask);
        code.and_(page, tmp);
    }
    // check for sign bit, apply sign extension as needed
    if (ctx.conf.page_table_sign_extension) {
        if (const int shift = 63 - int(*ctx.conf.page_table_sign_extension); shift > 0) {
            code.shl(page, shift);
            code.sar(page, shift);
        }
    }

    code.jz(abort, code.T_NEAR);
    if (ctx.conf.absolute_offset_page_table) {
        return page + vaddr;
    }
    code.mov(tmp, vaddr);
    code.and_(tmp, static_cast<u32>(page_table_const_mask));
    return page + tmp;
}

template<typename EmitContext>
Xbyak::RegExp EmitFastmemVAddr(BlockOfCode& code, EmitContext& ctx, Xbyak::Label& abort, Xbyak::Reg64 vaddr, bool& require_abort_handling, std::optional<Xbyak::Reg64> tmp = std::nullopt);

template<>
[[maybe_unused]] Xbyak::RegExp EmitFastmemVAddr<A32EmitContext>(BlockOfCode&, A32EmitContext&, Xbyak::Label&, Xbyak::Reg64 vaddr, bool&, std::optional<Xbyak::Reg64>) {
    return r13 + vaddr;
}

template<>
[[maybe_unused]] Xbyak::RegExp EmitFastmemVAddr<A64EmitContext>(BlockOfCode& code, A64EmitContext& ctx, Xbyak::Label& abort, Xbyak::Reg64 vaddr, bool& require_abort_handling, std::optional<Xbyak::Reg64> tmp) {
    auto const unused_top_bits = 64 - ctx.conf.fastmem_address_space_bits;
    if (unused_top_bits == 0) {
        return r13 + vaddr;
    } else if (ctx.conf.silently_mirror_fastmem) {
        if (!tmp) {
            tmp = ctx.reg_alloc.ScratchGpr(code);
        }
        if (unused_top_bits < 32) {
            code.mov(*tmp, vaddr);
            code.shl(*tmp, int(unused_top_bits));
            code.shr(*tmp, int(unused_top_bits));
        } else if (unused_top_bits == 32) {
            code.mov(tmp->cvt32(), vaddr.cvt32());
        } else {
            code.mov(tmp->cvt32(), vaddr.cvt32());
            code.and_(*tmp, u32((1 << ctx.conf.fastmem_address_space_bits) - 1));
        }
        return r13 + *tmp;
    } else {
        const u64 pt_mask = ctx.conf.page_table_pointer_mask;
        const int pt_top_zeros = pt_mask == 0 ? 0 : std::countl_zero(pt_mask);
        const u64 sign_filled_pt_mask = pt_top_zeros == 0 ? pt_mask : (pt_mask | (~u64{0} << (64 - pt_top_zeros)));
        const size_t valid_pt_index_bits = ctx.conf.page_table_address_space_bits > page_table_const_bits
                                             ? ctx.conf.page_table_address_space_bits - page_table_const_bits
                                             : 0;
        const bool can_pt_fallback =
            ctx.conf.page_table != nullptr && ctx.conf.absolute_offset_page_table &&
            ctx.conf.inline_page_table_fallback &&
            ctx.conf.fastmem_address_space_bits < ctx.conf.page_table_address_space_bits &&
            valid_pt_index_bits > 0 &&
            (pt_mask == 0 || std::in_range<s32>(static_cast<s64>(sign_filled_pt_mask))) &&
            (!ctx.conf.page_table_sign_extension || int(*ctx.conf.page_table_sign_extension) <= 63 - pt_top_zeros);

        if (can_pt_fallback) {
            const Xbyak::Reg64 base = tmp ? *tmp : ctx.reg_alloc.ScratchGpr(code);
            SharedLabel pt_lookup = ctx.GenSharedLabel(), pt_done = ctx.GenSharedLabel();
            Xbyak::Label* const abort_ptr = &abort;
            if (ctx.conf.fastmem_address_space_bits < 32) {
                code.test(vaddr, u32(-(1 << ctx.conf.fastmem_address_space_bits)));
            } else {
                code.mov(base, vaddr);
                code.shr(base, int(ctx.conf.fastmem_address_space_bits));
            }
            code.mov(base, r13);
            code.jnz(*pt_lookup, code.T_NEAR);
            code.L(*pt_done);
            require_abort_handling = true;

            ColdEmits(code, ctx).emplace_back([=, &code, &ctx] {
                code.L(*pt_lookup);
                if (ctx.conf.detect_misaligned_access_via_page_table != 0) {
                    if (!ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary) {
                        code.test(vaddr, 0xF);
                        code.jnz(*abort_ptr, code.T_NEAR);
                    } else {
                        code.mov(base.cvt32(), vaddr.cvt32());
                        code.and_(base.cvt32(), static_cast<u32>(page_table_const_mask));
                        code.cmp(base.cvt32(), static_cast<u32>(page_table_const_size - 16));
                        code.ja(*abort_ptr, code.T_NEAR);
                    }
                }

                const size_t unused_pt_top_bits = 64 - ctx.conf.page_table_address_space_bits;
                code.mov(base, vaddr);
                if (unused_pt_top_bits == 0) {
                    code.shr(base, int(page_table_const_bits));
                } else if (ctx.conf.silently_mirror_page_table) {
                    code.shl(base, int(unused_pt_top_bits));
                    code.shr(base, int(unused_pt_top_bits + page_table_const_bits));
                } else {
                    code.shr(base, int(ctx.conf.page_table_address_space_bits));
                    code.jnz(*abort_ptr, code.T_NEAR);
                    code.mov(base, vaddr);
                    code.shr(base, int(page_table_const_bits));
                }

                if (ctx.conf.page_table_log2_stride > 3) {
                    code.shl(base, int(ctx.conf.page_table_log2_stride));
                    code.mov(base, qword[r14 + base]);
                } else {
                    code.mov(base, qword[r14 + base * int(1 << ctx.conf.page_table_log2_stride)]);
                }

                if (ctx.conf.page_table_marked_bit) {
                    code.bt(base, *ctx.conf.page_table_marked_bit);
                    code.jc(*abort_ptr, code.T_NEAR);
                }

                if (pt_mask == 0) {
                    code.test(base, base);
                } else {
                    code.and_(base, static_cast<s32>(static_cast<s64>(sign_filled_pt_mask)));
                }

                if (ctx.conf.page_table_sign_extension) {
                    if (const int shift = 63 - int(*ctx.conf.page_table_sign_extension); shift > 0) {
                        code.shl(base, shift);
                        code.sar(base, shift);
                    }
                } else if (pt_top_zeros > 0) {
                    code.shl(base, pt_top_zeros);
                    code.shr(base, pt_top_zeros);
                }

                code.jz(*abort_ptr, code.T_NEAR);
                code.jmp(*pt_done, code.T_NEAR);
            });
            return base + vaddr;
        }

        if (ctx.conf.fastmem_address_space_bits < 32) {
            code.test(vaddr, u32(-(1 << ctx.conf.fastmem_address_space_bits)));
            code.jnz(abort, code.T_NEAR);
            require_abort_handling = true;
        } else {
            // TODO: Consider having TEST as above but coalesce 64-bit constant in register allocator
            if (!tmp) {
                tmp = ctx.reg_alloc.ScratchGpr(code);
            }
            code.mov(*tmp, vaddr);
            code.shr(*tmp, int(ctx.conf.fastmem_address_space_bits));
            code.jnz(abort, code.T_NEAR);
            require_abort_handling = true;
        }
        return r13 + vaddr;
    }
}

template<std::size_t bitsize>
const void* EmitReadMemoryMov(BlockOfCode& code, int value_idx, const Xbyak::RegExp& addr, bool ordered) {
    if constexpr (bitsize == 128) {
        if (ordered) {
            code.xor_(eax, eax);
            code.xor_(ebx, ebx);
            code.xor_(ecx, ecx);
            code.xor_(edx, edx);

            const void* fastmem_location = code.getCurr();
            code.lock();
            code.cmpxchg16b(xword[addr]);
            if (code.HasHostFeature(HostFeature::SSE41)) {
                code.movq(Xbyak::Xmm(value_idx), rax);
                code.pinsrq(Xbyak::Xmm(value_idx), rdx, 1);
            } else {
                code.movq(Xbyak::Xmm(value_idx), rax);
                code.movq(xmm0, rdx);
                code.punpcklqdq(Xbyak::Xmm(value_idx), xmm0);
            }
            return fastmem_location;
        }
    }

    {
        const void* fastmem_location = code.getCurr();
        switch (bitsize) {
        case 8:
            code.movzx(Xbyak::Reg64(value_idx).cvt32(), code.byte[addr]);
            break;
        case 16:
            code.movzx(Xbyak::Reg64(value_idx).cvt32(), word[addr]);
            break;
        case 32:
            code.mov(Xbyak::Reg64(value_idx).cvt32(), dword[addr]);
            break;
        case 64:
            code.mov(Xbyak::Reg64(value_idx), qword[addr]);
            break;
        case 128:
            code.movups(Xbyak::Xmm(value_idx), xword[addr]);
            break;
        default:
            UNREACHABLE();
        }
        return fastmem_location;
    }
}

template<std::size_t bitsize>
const void* EmitWriteMemoryMov(BlockOfCode& code, const Xbyak::RegExp& addr, int value_idx, bool ordered) {
    if (ordered) {
        if constexpr (bitsize == 128) {
            code.xor_(eax, eax);
            code.xor_(edx, edx);
            if (code.HasHostFeature(HostFeature::SSE41)) {
                code.movq(rbx, Xbyak::Xmm(value_idx));
                code.pextrq(rcx, Xbyak::Xmm(value_idx), 1);
            } else {
                code.movaps(xmm0, Xbyak::Xmm(value_idx));
                code.movq(rbx, xmm0);
                code.punpckhqdq(xmm0, xmm0);
                code.movq(rcx, xmm0);
            }
        }

        const void* fastmem_location = code.getCurr();
        switch (bitsize) {
        case 8:
            code.xchg(code.byte[addr], Xbyak::Reg64(value_idx).cvt8());
            break;
        case 16:
            code.xchg(word[addr], Xbyak::Reg64(value_idx).cvt16());
            break;
        case 32:
            code.xchg(dword[addr], Xbyak::Reg64(value_idx).cvt32());
            break;
        case 64:
            code.xchg(qword[addr], Xbyak::Reg64(value_idx));
            break;
        case 128: {
            Xbyak::Label loop;
            code.L(loop);
            code.lock();
            code.cmpxchg16b(xword[addr]);
            code.jnz(loop, code.T_NEAR);
            break;
        }
        default:
            UNREACHABLE();
        }
        return fastmem_location;
    } else {
        const void* fastmem_location = code.getCurr();
        switch (bitsize) {
        case 8:
            code.mov(code.byte[addr], Xbyak::Reg64(value_idx).cvt8());
            break;
        case 16:
            code.mov(word[addr], Xbyak::Reg64(value_idx).cvt16());
            break;
        case 32:
            code.mov(dword[addr], Xbyak::Reg64(value_idx).cvt32());
            break;
        case 64:
            code.mov(qword[addr], Xbyak::Reg64(value_idx));
            break;
        case 128:
            code.movups(xword[addr], Xbyak::Xmm(value_idx));
            break;
        default:
            UNREACHABLE();
        }
        return fastmem_location;
    }
}

/// EDEN_NX2_LOCKED_MONITOR=1 restores the global monitor spinlock around the inline exclusive
/// fast paths, to compare against the lock-free ordering.
inline bool UseLockedExclusiveMonitor() {
    static const bool locked = [] {
        const char* const value = std::getenv("EDEN_NX2_LOCKED_MONITOR");
        return value != nullptr && *value != '\0' && *value != '0';
    }();
    return locked;
}

template<typename UserConfig>
void EmitExclusiveLock(BlockOfCode& code, const UserConfig& conf, Xbyak::Reg64 pointer, Xbyak::Reg32 tmp) {
    if (conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor)) {
        return;
    }

    code.mov(pointer, std::bit_cast<u64>(GetExclusiveMonitorLockPointer(conf.global_monitor)));
    EmitSpinLockLock(code, pointer, tmp, code.HasHostFeature(HostFeature::WAITPKG));
}

template<typename UserConfig>
void EmitExclusiveUnlock(BlockOfCode& code, const UserConfig& conf, Xbyak::Reg64 pointer, Xbyak::Reg32 tmp) {
    if (conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor)) {
        return;
    }

    code.mov(pointer, std::bit_cast<u64>(GetExclusiveMonitorLockPointer(conf.global_monitor)));
    EmitSpinLockUnlock(code, pointer, tmp);
}

/// Atomic cross-core reservation clear. Each slot has one writer (its owner core) and is a
/// naturally atomic aligned u64 on x86, so the clear is a LOCK CMPXCHG that cannot tear
/// against the owner's write, and the caller needs no global monitor lock. A reservation
/// lost to the read-then-cmpxchg window only ever surfaces as a failed guest-memory compare
/// in the owner's next exclusive store: a spurious LL/SC failure, which the architecture
/// permits and every retry loop handles.
template<typename UserConfig>
void EmitExclusiveTestAndClear(BlockOfCode& code, const UserConfig& conf, Xbyak::Reg64 vaddr,
                               Xbyak::Reg64 pointer, Xbyak::Reg64 dead, Xbyak::Reg64 expected) {
    if (conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor)) {
        return;
    }

    code.mov(dead, u64(0xDEAD'DEAD'DEAD'DEAD));
    const size_t processor_count = GetExclusiveMonitorProcessorCount(conf.global_monitor);
    for (size_t processor_index = 0; processor_index < processor_count; processor_index++) {
        if (processor_index == conf.processor_id) {
            continue;
        }
        Xbyak::Label ok;
        code.mov(pointer, std::bit_cast<u64>(GetExclusiveMonitorAddressPointer(conf.global_monitor, processor_index)));
        code.cmp(qword[pointer], vaddr);
        code.jne(ok, code.T_NEAR);
        code.mov(expected, vaddr);
        code.lock();
        code.cmpxchg(qword[pointer], dead);
        code.L(ok);
    }
}

/// A folded address displacement as an Xbyak displacement (sign-extended).
inline size_t FoldedDisp(s32 disp) {
    return static_cast<size_t>(static_cast<s64>(disp));
}

inline bool IsOrdered(IR::AccType acctype) {
    return acctype == IR::AccType::ORDERED || acctype == IR::AccType::ORDEREDRW || acctype == IR::AccType::LIMITEDORDERED;
}

}  // namespace

}  // namespace Dynarmic::Backend::X64
