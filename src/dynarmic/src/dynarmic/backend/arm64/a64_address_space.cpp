// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/a64_address_space.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::Backend::Arm64 {

template<auto mfp, typename T>
static void* EmitCallReadTrampoline(oaknut::CodeGenerator& code, T* this_, size_t bitsize) {
    using namespace oaknut::util;
    const auto info = Devirtualize<mfp>(this_);
    oaknut::Label l_addr, l_this;
    void* target = code.xptr<void*>();
    // params = { this, vaddr, bitsize }
    code.LDR(X0, l_this);
    // X1 = vaddr
    code.MOV(X2, bitsize);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);
    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);
    return target;
}

template<auto mfp, typename T>
static void* EmitCallWriteTrampoline(oaknut::CodeGenerator& code, T* this_, size_t bitsize) {
    using namespace oaknut::util;
    const auto info = Devirtualize<mfp>(this_);
    oaknut::Label l_addr, l_this;
    void* target = code.xptr<void*>();
    // params = { this, vaddr, value, bitsize }
    code.LDR(X0, l_this);
    // X1 = vaddr
    // X2 = value
    code.MOV(X3, bitsize);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);
    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);
    return target;
}

template<auto mfp, typename T>
static void* EmitCallTrampoline(oaknut::CodeGenerator& code, T* this_) {
    using namespace oaknut::util;
    const auto info = Devirtualize<mfp>(this_);
    oaknut::Label l_addr, l_this;
    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);
    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);
    return target;
}

template<auto mfp, typename T>
static void* EmitWrappedReadCallTrampoline(oaknut::CodeGenerator& code, T* this_, size_t bitsize) {
    using namespace oaknut::util;

    const auto info = Devirtualize<mfp>(this_);

    oaknut::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Xscratch0);

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDR(X0, l_this);
    code.MOV(X1, Xscratch0);
    code.MOV(X2, bitsize);
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    code.MOV(Xscratch0, X0);
    ABI_PopRegisters(code, save_regs, 0);
    code.RET();

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto callback, typename T>
static void* EmitExclusiveReadCallTrampoline(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr) -> T {
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
            return (conf.callbacks->*callback)(vaddr, sizeof(T));
        });
    };

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(std::bit_cast<u64>(&conf));
    code.l(l_addr);
    code.dx(std::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

template<auto mfp, typename T>
static void* EmitWrappedWriteCallTrampoline(oaknut::CodeGenerator& code, T* this_, size_t bitsize) {
    using namespace oaknut::util;

    const auto info = Devirtualize<mfp>(this_);

    oaknut::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE;

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDR(X0, l_this);
    code.MOV(X1, Xscratch0);
    code.MOV(X2, Xscratch1);
    code.MOV(X3, bitsize);
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    ABI_PopRegisters(code, save_regs, 0);
    code.RET();

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto callback, typename T>
static void* EmitExclusiveWriteCallTrampoline(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_this;
    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr, T value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr, [&](T expected) -> bool {
            return (conf.callbacks->*callback)(vaddr, value, expected);
        }) ? 0 : 1;
    };

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(std::bit_cast<u64>(&conf));
    code.l(l_addr);
    code.dx(std::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

static void* EmitRead128CallTrampoline(oaknut::CodeGenerator& code, A64::UserCallbacks* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryRead128>(this_);

    oaknut::Label l_addr, l_this;

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    code.FMOV(D0, X0);
    code.FMOV(V0.D()[1], X1);
    ABI_PopRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.RET();

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitWrappedRead128CallTrampoline(oaknut::CodeGenerator& code, A64::UserCallbacks* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryRead128>(this_);

    oaknut::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Q0);

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDR(X0, l_this);
    code.MOV(X1, Xscratch0);
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    code.FMOV(D0, X0);
    code.FMOV(V0.D()[1], X1);
    ABI_PopRegisters(code, save_regs, 0);
    code.RET();

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitExclusiveRead128CallTrampoline(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr) -> Vector {
        return conf.global_monitor->ReadAndMark<Vector>(conf.processor_id, vaddr, [&]() -> Vector {
            return conf.callbacks->MemoryRead128(vaddr);
        });
    };

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    code.FMOV(D0, X0);
    code.FMOV(V0.D()[1], X1);
    ABI_PopRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.RET();

    code.align(8);
    code.l(l_this);
    code.dx(std::bit_cast<u64>(&conf));
    code.l(l_addr);
    code.dx(std::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

static void* EmitWrite128CallTrampoline(oaknut::CodeGenerator& code, A64::UserCallbacks* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryWrite128>(this_);

    oaknut::Label l_addr, l_this;

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.FMOV(X2, D0);
    code.FMOV(X3, V0.D()[1]);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitWrappedWrite128CallTrampoline(oaknut::CodeGenerator& code, A64::UserCallbacks* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryWrite128>(this_);

    oaknut::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE;

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDR(X0, l_this);
    code.MOV(X1, Xscratch0);
    code.FMOV(X2, D0);
    code.FMOV(X3, V0.D()[1]);
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    ABI_PopRegisters(code, save_regs, 0);
    code.RET();

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitExclusiveWrite128CallTrampoline(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr, Vector value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<Vector>(conf.processor_id, vaddr,
                                                                 [&](Vector expected) -> bool {
                                                                     return conf.callbacks->MemoryWriteExclusive128(vaddr, value, expected);
                                                                 })
                 ? 0
                 : 1;
    };

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.FMOV(X2, D0);
    code.FMOV(X3, V0.D()[1]);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(std::bit_cast<u64>(&conf));
    code.l(l_addr);
    code.dx(std::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

namespace {

bool Arm64EnvOn(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

inline u32 HashFastDispatch(u64 value) noexcept {
#if defined(__GNUC__) && defined(__ARM_FEATURE_CRC32)
    return __builtin_arm_crc32cd(0, value);
#else
    u32 crc = 0;
    u64 data = value;
    for (int i = 0; i < 64; ++i) {
        crc = (crc >> 1) ^ ((crc ^ static_cast<u32>(data)) & 1U ? 0x82F63B78U : 0U);
        data >>= 1;
    }
    return crc;
#endif
}

template<std::size_t bitsize>
using AtomicUInt = std::conditional_t<
    bitsize == 8, u8,
    std::conditional_t<bitsize == 16, u16, std::conditional_t<bitsize == 32, u32, u64>>>;

template<std::size_t bitsize>
bool AtomicWriteExclusive(A64::UserCallbacks* cb, u64 vaddr, AtomicUInt<bitsize> value, AtomicUInt<bitsize> expected) {
    if constexpr (bitsize == 8) {
        return cb->MemoryWriteExclusive8(vaddr, value, expected);
    } else if constexpr (bitsize == 16) {
        return cb->MemoryWriteExclusive16(vaddr, value, expected);
    } else if constexpr (bitsize == 32) {
        return cb->MemoryWriteExclusive32(vaddr, value, expected);
    } else {
        return cb->MemoryWriteExclusive64(vaddr, value, expected);
    }
}

template<std::size_t bitsize>
AtomicUInt<bitsize> AtomicRmwCompute(AtomicUInt<bitsize> a, AtomicUInt<bitsize> b, u64 op) {
    using T = AtomicUInt<bitsize>;
    using S = std::make_signed_t<T>;
    switch (op & 0x0F) {
    case 0:
        return static_cast<T>(a + b);
    case 1:
        return static_cast<T>(a & ~b);
    case 2:
        return static_cast<T>(a ^ b);
    case 3:
        return static_cast<T>(a | b);
    case 4:
        return static_cast<S>(a) > static_cast<S>(b) ? a : b;
    case 5:
        return static_cast<S>(a) < static_cast<S>(b) ? a : b;
    case 6:
        return a > b ? a : b;
    case 7:
        return a < b ? a : b;
    default:
        return b;
    }
}

template<std::size_t bitsize>
u64 AtomicRmwFallback(const A64AddressSpace::AtomicCallbackCtx* ctx, A64::VAddr vaddr, u64 operand, u64 op) {
    using T = AtomicUInt<bitsize>;
    const A64::UserConfig& conf = *ctx->conf;
    for (u32 attempts = 0;; ++attempts) {
        const T old = static_cast<T>(conf.callbacks->MemoryRead(vaddr, bitsize / 8));
        if (AtomicWriteExclusive<bitsize>(conf.callbacks, vaddr, AtomicRmwCompute<bitsize>(old, static_cast<T>(operand), op), old)) {
            return static_cast<u64>(old);
        }
        if ((ctx->halt_reason != nullptr && *ctx->halt_reason != 0) || attempts >= 4096) {
            return static_cast<u64>(old);
        }
    }
}

template<std::size_t bitsize>
u64 AtomicCasFallback(const A64AddressSpace::AtomicCallbackCtx* ctx, A64::VAddr vaddr, u64 expected_u64, u64 desired_u64) {
    using T = AtomicUInt<bitsize>;
    const A64::UserConfig& conf = *ctx->conf;
    const T expected = static_cast<T>(expected_u64);
    const T desired = static_cast<T>(desired_u64);
    for (u32 attempts = 0;; ++attempts) {
        const T old = static_cast<T>(conf.callbacks->MemoryRead(vaddr, bitsize / 8));
        if (old != expected) {
            return static_cast<u64>(old);
        }
        if (AtomicWriteExclusive<bitsize>(conf.callbacks, vaddr, desired, old)) {
            return static_cast<u64>(old);
        }
        if ((ctx->halt_reason != nullptr && *ctx->halt_reason != 0) || attempts >= 4096) {
            return static_cast<u64>(old);
        }
    }
}

void* EmitWrappedAtomicCallTrampoline(
    oaknut::CodeGenerator& code,
    const A64AddressSpace::AtomicCallbackCtx* ctx,
    u64 (*fn)(const A64AddressSpace::AtomicCallbackCtx*, A64::VAddr, u64, u64)) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_ctx;
    constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Xscratch0);

    void* target = code.xptr<void*>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDR(X0, l_ctx);
    code.MOV(X1, Xscratch0);
    code.MOV(X2, Xscratch1);
    code.LDR(X3, Xstate, offsetof(A64JitState, atomic_scratch));
    code.LDR(Xscratch0, l_addr);
    code.BLR(Xscratch0);
    code.MOV(Xscratch0, X0);
    ABI_PopRegisters(code, save_regs, 0);
    code.RET();

    code.align(8);
    code.l(l_ctx);
    code.dx(std::bit_cast<u64>(ctx));
    code.l(l_addr);
    code.dx(std::bit_cast<u64>(fn));

    return target;
}

} // namespace

A64AddressSpace::A64AddressSpace(const A64::UserConfig& conf)
        : AddressSpace(conf.code_cache_size)
        , conf(conf) {
    jit_stats = conf.jit_stats;
    code_cache_used = conf.code_cache_used;
    atomic_ctx.conf = &this->conf;
    if (this->conf.HasOptimization(OptimizationFlag::FastDispatch) && Arm64EnvOn("EDEN_NX2_JIT_DISPATCH")) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
        constexpr std::size_t default_bits = 15;
#else
        constexpr std::size_t default_bits = 16;
#endif
        std::size_t bits = this->conf.fast_dispatch_table_bits != 0 ? this->conf.fast_dispatch_table_bits : default_bits;
        bits = std::clamp<std::size_t>(bits, 8, 22);
        fast_dispatch_table.resize(std::size_t{1} << bits);
        fast_dispatch_set_mask = static_cast<u32>(((std::size_t{1} << bits) - fast_dispatch_ways) * sizeof(FastDispatchEntry));
    }
    EmitPrelude();
}

void A64AddressSpace::GenerateIR(IR::Block& ir_block, IR::LocationDescriptor descriptor) const {
    ir_block.Reset(descriptor);
    const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
#if defined(__APPLE__) || defined(__ARM_FEATURE_ATOMICS)
    constexpr bool host_has_lse = true;
#else
    constexpr bool host_has_lse = false;
#endif
    static const bool env_atomics = Arm64EnvOn("EDEN_NX2_JIT_ATOMICS");
    const bool use_lse_atomics =
        conf.native_lse_atomics ||
        (host_has_lse && env_atomics && conf.fastmem_exclusive_access &&
         conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor));
    A64::TranslationOptions trans_opts{};
    trans_opts.define_unpredictable_behaviour = conf.define_unpredictable_behaviour;
    trans_opts.wall_clock_cntpct = conf.wall_clock_cntpct;
    trans_opts.hook_hint_instructions = true;
    trans_opts.native_lse_atomics = use_lse_atomics;
    A64::Translate(ir_block, A64::LocationDescriptor{descriptor}, get_code, trans_opts);
    Optimization::Optimize(ir_block, conf, {});
}

void A64AddressSpace::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    const auto blocks = block_ranges.InvalidateRanges(ranges);
    A64::AddJitStat(jit_stats, A64::JitStat::BlocksInvalidated, blocks.size());
    InvalidateBasicBlocks(blocks);
}

void A64AddressSpace::ClearFastDispatchTable() {
    if (!fast_dispatch_table.empty()) {
        std::fill(fast_dispatch_table.begin(), fast_dispatch_table.end(), FastDispatchEntry{});
    }
}

void A64AddressSpace::ClearCache() {
    block_ranges.ClearCache();
    ClearFastDispatchTable();
    if (jit_state != nullptr) {
        jit_state->ResetRSB();
    }
    AddressSpace::ClearCache();
}

void A64AddressSpace::OnInvalidateBasicBlock(IR::LocationDescriptor descriptor) {
    const u64 desc_val = descriptor.Value();
    if (!fast_dispatch_table.empty()) {
        const u32 byte_offset = HashFastDispatch(desc_val) & fast_dispatch_set_mask;
        auto* const set = reinterpret_cast<FastDispatchEntry*>(
            reinterpret_cast<std::byte*>(fast_dispatch_table.data()) + byte_offset);
        for (std::size_t way = 0; way < fast_dispatch_ways; ++way) {
            if (set[way].location_descriptor == desc_val) {
                set[way] = FastDispatchEntry{};
            }
        }
    }
    if (jit_state != nullptr) {
        jit_state->InvalidateRSBTarget(desc_val);
    }
}

void A64AddressSpace::EmitPrelude() {
    using namespace oaknut::util;

    UnprotectCodeMemory();

    prelude_info.read_memory_8 = EmitCallReadTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u8));
    prelude_info.read_memory_16 = EmitCallReadTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u16));
    prelude_info.read_memory_32 = EmitCallReadTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u32));
    prelude_info.read_memory_64 = EmitCallReadTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u64));
    prelude_info.read_memory_128 = EmitRead128CallTrampoline(code, conf.callbacks);
    prelude_info.wrapped_read_memory_8 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u8));
    prelude_info.wrapped_read_memory_16 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u16));
    prelude_info.wrapped_read_memory_32 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u32));
    prelude_info.wrapped_read_memory_64 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead>(code, conf.callbacks, sizeof(u64));
    prelude_info.wrapped_read_memory_128 = EmitWrappedRead128CallTrampoline(code, conf.callbacks);
    prelude_info.exclusive_read_memory_8 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead, u8>(code, conf);
    prelude_info.exclusive_read_memory_16 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead, u16>(code, conf);
    prelude_info.exclusive_read_memory_32 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead, u32>(code, conf);
    prelude_info.exclusive_read_memory_64 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead, u64>(code, conf);
    prelude_info.exclusive_read_memory_128 = EmitExclusiveRead128CallTrampoline(code, conf);
    prelude_info.write_memory_8 = EmitCallWriteTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u8));
    prelude_info.write_memory_16 = EmitCallWriteTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u16));
    prelude_info.write_memory_32 = EmitCallWriteTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u32));
    prelude_info.write_memory_64 = EmitCallWriteTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u64));
    prelude_info.write_memory_128 = EmitWrite128CallTrampoline(code, conf.callbacks);
    prelude_info.wrapped_write_memory_8 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u8));
    prelude_info.wrapped_write_memory_16 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u16));
    prelude_info.wrapped_write_memory_32 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u32));
    prelude_info.wrapped_write_memory_64 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite>(code, conf.callbacks, sizeof(u64));
    prelude_info.wrapped_write_memory_128 = EmitWrappedWrite128CallTrampoline(code, conf.callbacks);
    prelude_info.exclusive_write_memory_8 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive8, u8>(code, conf);
    prelude_info.exclusive_write_memory_16 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive16, u16>(code, conf);
    prelude_info.exclusive_write_memory_32 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive32, u32>(code, conf);
    prelude_info.exclusive_write_memory_64 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive64, u64>(code, conf);
    prelude_info.exclusive_write_memory_128 = EmitExclusiveWrite128CallTrampoline(code, conf);
    prelude_info.wrapped_atomic_rmw_8 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicRmwFallback<8>);
    prelude_info.wrapped_atomic_rmw_16 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicRmwFallback<16>);
    prelude_info.wrapped_atomic_rmw_32 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicRmwFallback<32>);
    prelude_info.wrapped_atomic_rmw_64 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicRmwFallback<64>);
    prelude_info.wrapped_atomic_cas_8 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicCasFallback<8>);
    prelude_info.wrapped_atomic_cas_16 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicCasFallback<16>);
    prelude_info.wrapped_atomic_cas_32 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicCasFallback<32>);
    prelude_info.wrapped_atomic_cas_64 = EmitWrappedAtomicCallTrampoline(code, &atomic_ctx, &AtomicCasFallback<64>);
    prelude_info.call_svc = EmitCallTrampoline<&A64::UserCallbacks::CallSVC>(code, conf.callbacks);
    prelude_info.exception_raised = EmitCallTrampoline<&A64::UserCallbacks::ExceptionRaised>(code, conf.callbacks);
    prelude_info.isb_raised = EmitCallTrampoline<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(code, conf.callbacks);
    prelude_info.ic_raised = EmitCallTrampoline<&A64::UserCallbacks::InstructionCacheOperationRaised>(code, conf.callbacks);
    prelude_info.dc_raised = EmitCallTrampoline<&A64::UserCallbacks::DataCacheOperationRaised>(code, conf.callbacks);
    prelude_info.get_cntpct = EmitCallTrampoline<&A64::UserCallbacks::GetCNTPCT>(code, conf.callbacks);
    prelude_info.add_ticks = EmitCallTrampoline<&A64::UserCallbacks::AddTicks>(code, conf.callbacks);
    prelude_info.get_ticks_remaining = EmitCallTrampoline<&A64::UserCallbacks::GetTicksRemaining>(code, conf.callbacks);

    oaknut::Label return_from_run_code, return_to_dispatcher_label, rsb_miss;
    const bool use_fast_dispatch = !fast_dispatch_table.empty();

    prelude_info.run_code = code.xptr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));

        code.MOV(X19, X0);
        code.MOV(Xstate, X1);
        code.MOV(Xhalt, X2);
        if (conf.page_table) {
            code.MOV(Xpagetable, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.MOV(Xfastmem, *conf.fastmem_pointer);
        }

        if (conf.enable_cycle_counting) {
            code.BL(prelude_info.get_ticks_remaining);
            code.MOV(Xticks, X0);
            code.STR(Xticks, SP, offsetof(StackLayout, cycles_to_run));
        }

        code.MRS(Xscratch1, oaknut::SystemReg::FPCR);
        code.STR(Wscratch1, SP, offsetof(StackLayout, save_host_fpcr));
        code.LDR(Wscratch0, Xstate, offsetof(A64JitState, fpcr));
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        code.LDR(Wscratch0, Xhalt);
        code.CBNZ(Wscratch0, return_from_run_code);

        code.BR(X19);
    }

    prelude_info.step_code = code.xptr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));

        code.MOV(X19, X0);
        code.MOV(Xstate, X1);
        code.MOV(Xhalt, X2);
        if (conf.page_table) {
            code.MOV(Xpagetable, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.MOV(Xfastmem, *conf.fastmem_pointer);
        }

        if (conf.enable_cycle_counting) {
            code.MOV(Xticks, 1);
            code.STR(Xticks, SP, offsetof(StackLayout, cycles_to_run));
        }

        code.MRS(Xscratch1, oaknut::SystemReg::FPCR);
        code.STR(Wscratch1, SP, offsetof(StackLayout, save_host_fpcr));
        code.LDR(Wscratch0, Xstate, offsetof(A64JitState, fpcr));
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        oaknut::Label step_hr_loop;
        code.l(step_hr_loop);
        code.LDAXR(Wscratch0, Xhalt);
        code.CBNZ(Wscratch0, return_from_run_code);
        code.ORR(Wscratch0, Wscratch0, static_cast<u32>(HaltReason::Step));
        code.STLXR(Wscratch1, Wscratch0, Xhalt);
        code.CBNZ(Wscratch1, step_hr_loop);

        code.BR(X19);
    }

    const auto emit_indirect_halt_check = [&] {
        code.LDR(Wscratch0, Xhalt);
        code.CBNZ(Wscratch0, return_from_run_code);
        if (conf.enable_cycle_counting) {
            code.CMP(Xticks, 0);
            code.B(LE, return_from_run_code);
        }
    };

    const auto emit_calc_location_descriptor = [&] {
        code.LDR(W0, Xstate, offsetof(A64JitState, fpcr));
        code.LDR(X1, Xstate, offsetof(A64JitState, pc));
        code.MOV(Wscratch0, A64::LocationDescriptor::fpcr_mask);
        code.AND(W0, W0, Wscratch0);
        code.AND(X1, X1, A64::LocationDescriptor::pc_mask);
        code.ORR(X0, X1, X0, LSL, A64::LocationDescriptor::fpcr_shift);
    };

    prelude_info.pop_rsb_hint = code.xptr<void*>();
    {
        emit_indirect_halt_check();
        emit_calc_location_descriptor();

        code.LDR(Wscratch2, Xstate, offsetof(A64JitState, rsb_ptr));
        code.AND(Wscratch2, Wscratch2, A64RSBIndexMask);
        code.ADD(X2, Xstate, Xscratch2);
        code.SUB(Wscratch2, Wscratch2, sizeof(RSBEntry));
        code.AND(Wscratch2, Wscratch2, A64RSBIndexMask);
        code.STR(Wscratch2, Xstate, offsetof(A64JitState, rsb_ptr));

        code.LDP(Xscratch0, Xscratch1, X2, 0);
        code.CMP(X0, Xscratch0);
        if (use_fast_dispatch) {
            code.B(NE, rsb_miss);
        } else {
            code.B(NE, return_to_dispatcher_label);
        }
        code.BR(Xscratch1);
    }

    if (use_fast_dispatch) {
        prelude_info.fast_dispatch_hint = code.xptr<void*>();
        oaknut::Label fd_miss, l_table, l_this, l_lookup_fn;

        emit_indirect_halt_check();
        emit_calc_location_descriptor();

        code.l(rsb_miss);
        code.LDR(X1, l_table);
        code.CRC32CX(W2, WZR, X0);
        code.AND(W2, W2, fast_dispatch_set_mask);
        code.ADD(X1, X1, X2);

        for (std::size_t way = 0; way < fast_dispatch_ways; ++way) {
            oaknut::Label next_way;
            code.LDP(Xscratch0, Xscratch1, X1, static_cast<int>(way * sizeof(FastDispatchEntry)));
            code.CMP(X0, Xscratch0);
            code.B(NE, way + 1 < fast_dispatch_ways ? next_way : fd_miss);
            code.BR(Xscratch1);
            code.l(next_way);
        }

        code.l(fd_miss);
        code.MOV(X2, X0);
        code.MOV(X3, X1);
        code.LDR(X0, l_this);
        code.MOV(X1, Xstate);
        code.LDR(Xscratch0, l_lookup_fn);
        code.BLR(Xscratch0);
        code.BR(X0);

        const auto lookup_fn = [](A64AddressSpace& self, A64JitState& context, u64 desc, FastDispatchEntry* set) -> CodePtr {
            const u64 generation = self.cache_generation;
            const CodePtr entry_point = self.GetOrEmit(IR::LocationDescriptor{desc});
            if (generation != self.cache_generation) {
                context.ResetRSB();
            }
            for (std::size_t way = fast_dispatch_ways - 1; way > 0; --way) {
                set[way] = set[way - 1];
            }
            set[0] = FastDispatchEntry{desc, entry_point};
            return entry_point;
        };

        code.align(8);
        code.l(l_table);
        code.dx(std::bit_cast<u64>(fast_dispatch_table.data()));
        code.l(l_this);
        code.dx(std::bit_cast<u64>(this));
        code.l(l_lookup_fn);
        code.dx(std::bit_cast<u64>(Common::FptrCast(lookup_fn)));
    } else {
        prelude_info.fast_dispatch_hint = code.xptr<void*>();
    }

    prelude_info.return_to_dispatcher = code.xptr<void*>();
    {
        oaknut::Label l_this, l_addr;
        code.l(return_to_dispatcher_label);

        emit_indirect_halt_check();

        code.LDR(X0, l_this);
        code.MOV(X1, Xstate);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        code.BR(X0);

        const auto fn = [](A64AddressSpace& self, A64JitState& context) -> CodePtr {
            const u64 generation = self.cache_generation;
            const CodePtr entry_point = self.GetOrEmit(context.GetLocationDescriptor());
            if (generation != self.cache_generation) {
                context.ResetRSB();
            }
            return entry_point;
        };

        code.align(8);
        code.l(l_this);
        code.dx(std::bit_cast<u64>(this));
        code.l(l_addr);
        code.dx(std::bit_cast<u64>(Common::FptrCast(fn)));
    }

    prelude_info.return_from_run_code = code.xptr<void*>();
    {
        code.l(return_from_run_code);

        if (conf.enable_cycle_counting) {
            code.LDR(X1, SP, offsetof(StackLayout, cycles_to_run));
            code.SUB(X1, X1, Xticks);
            code.BL(prelude_info.add_ticks);
        }

        code.LDR(Wscratch0, SP, offsetof(StackLayout, save_host_fpcr));
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        oaknut::Label exit_hr_loop;
        code.l(exit_hr_loop);
        code.LDAXR(W0, Xhalt);
        code.STLXR(Wscratch0, WZR, Xhalt);
        code.CBNZ(Wscratch0, exit_hr_loop);

        ABI_PopRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));
        code.RET();
    }

    prelude_info.end_of_prelude = code.offset();

    mem.invalidate_all();
    ProtectCodeMemory();
}

EmitConfig A64AddressSpace::GetEmitConfig() {
    OptimizationFlag active_opts = conf.unsafe_optimizations ? conf.optimizations : conf.optimizations & all_safe_optimizations;
    if (fast_dispatch_table.empty()) {
        active_opts &= ~OptimizationFlag::FastDispatch;
    }
    static const bool env_backedge_halt = Arm64EnvOn("EDEN_NX2_JIT_BACKEDGE_HALT");
    const bool backedge_halt =
        !conf.enable_cycle_counting && !conf.check_halt_on_memory_access &&
        (conf.halt_check_on_back_edges || (conf.unsafe_optimizations && env_backedge_halt));

    return EmitConfig{
        .optimizations = active_opts,

        .hook_isb = conf.hook_isb,

        .cntfreq_el0 = conf.cntfrq_el0,
        .ctr_el0 = conf.ctr_el0,
        .dczid_el0 = conf.dczid_el0,
        .tpidrro_el0 = conf.tpidrro_el0,
        .tpidr_el0 = conf.tpidr_el0,

        .check_halt_on_memory_access = conf.check_halt_on_memory_access,

        .page_table_pointer = std::bit_cast<u64>(conf.page_table),
        .page_table_address_space_bits = conf.page_table_address_space_bits,
        .page_table_pointer_mask = conf.page_table_pointer_mask,
        .page_table_log2_stride = conf.page_table_log2_stride,
        .page_table_marked_bit = conf.page_table_marked_bit,
        .page_table_sign_extension = conf.page_table_sign_extension,
        .silently_mirror_page_table = conf.silently_mirror_page_table,
        .absolute_offset_page_table = conf.absolute_offset_page_table,
        .detect_misaligned_access_via_page_table = conf.detect_misaligned_access_via_page_table,
        .only_detect_misalignment_via_page_table_on_page_boundary = conf.only_detect_misalignment_via_page_table_on_page_boundary,

        .fastmem_pointer = conf.fastmem_pointer,
        .recompile_on_fastmem_failure = conf.recompile_on_fastmem_failure,
        .recompile_on_exclusive_fastmem_failure = conf.recompile_on_exclusive_fastmem_failure,
        .fastmem_address_space_bits = conf.fastmem_address_space_bits,
        .silently_mirror_fastmem = conf.silently_mirror_fastmem,
        .inline_page_table_fallback = conf.inline_page_table_fallback,

        .wall_clock_cntpct = conf.wall_clock_cntpct,
        .enable_cycle_counting = conf.enable_cycle_counting,
        .halt_check_on_back_edges = backedge_halt,
        .use_state_rsb = true,

        .always_little_endian = true,

        .descriptor_to_fpcr = [](const IR::LocationDescriptor& location) { return A64::LocationDescriptor{location}.FPCR(); },
        .emit_cond = EmitA64Cond,
        .emit_condition_failed_terminal = EmitA64ConditionFailedTerminal,
        .emit_terminal = EmitA64Terminal,
        .emit_check_memory_abort = EmitA64CheckMemoryAbort,

        .state_nzcv_offset = offsetof(A64JitState, cpsr_nzcv),
        .state_fpsr_offset = offsetof(A64JitState, fpsr),
        .state_exclusive_state_offset = offsetof(A64JitState, exclusive_state),
        .state_rsb_ptr_offset = offsetof(A64JitState, rsb_ptr),
        .state_atomic_scratch_offset = offsetof(A64JitState, atomic_scratch),

        .coprocessors{},

        .very_verbose_debugging_output = conf.very_verbose_debugging_output,
    };
}

void A64AddressSpace::RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) {
    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};
    static std::FILE* const block_map = [] {
        const char* path = std::getenv("DYNARMIC_BLOCK_MAP");
        std::FILE* file = path != nullptr ? std::fopen(path, "w") : nullptr;
        if (file != nullptr) {
            std::setvbuf(file, nullptr, _IOLBF, 0);
        }
        return file;
    }();
    if (block_map != nullptr) {
        std::fprintf(block_map, "%llx %zx %llx %llx\n",
                     static_cast<unsigned long long>(std::bit_cast<u64>(block_info.entry_point)),
                     block_info.size, static_cast<unsigned long long>(descriptor.PC()),
                     static_cast<unsigned long long>(end_location.PC()));
    }
    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);
}

}  // namespace Dynarmic::Backend::Arm64
