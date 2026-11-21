// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Core::CachedInterpreter {

enum class StopReason : u8 {
    None,
    StepCompleted,
    Interrupted,
    SupervisorCall,
    Breakpoint,
    DataAbort,
    PrefetchAbort,
    UnsupportedInstruction,
};

struct CpuState {
    std::array<u64, 31> x{};
    u64 sp{};
    u64 pc{};
    u32 pstate{};
    std::array<u128, 32> v{};
    u32 fpcr{};
    u32 fpsr{};
    u64 tpidr_el0{};
    u64 tpidrro_el0{};
};

struct Stats {
    u64 blocks_decoded{};
    u64 cache_hits{};
    u64 cache_misses{};
    u64 ops_fused{};
    u64 instructions_retired{};
    u64 invalidations{};
};

/// Abstract guest memory and platform callback interface so the cached engine
/// runs identically against Core::Memory::Memory and standalone unit-test harnesses.
class MemoryBus {
public:
    virtual ~MemoryBus() = default;

    virtual std::optional<u32> ReadCode32(u64 vaddr) = 0;

    virtual bool Read8(u64 vaddr, u8& out) = 0;
    virtual bool Read16(u64 vaddr, u16& out) = 0;
    virtual bool Read32(u64 vaddr, u32& out) = 0;
    virtual bool Read64(u64 vaddr, u64& out) = 0;
    virtual bool Read128(u64 vaddr, u128& out) = 0;

    virtual bool Write8(u64 vaddr, u8 value) = 0;
    virtual bool Write16(u64 vaddr, u16 value) = 0;
    virtual bool Write32(u64 vaddr, u32 value) = 0;
    virtual bool Write64(u64 vaddr, u64 value) = 0;
    virtual bool Write128(u64 vaddr, u128 value) = 0;

    virtual bool ExclusiveRead8(u64 vaddr, u8& out) = 0;
    virtual bool ExclusiveRead16(u64 vaddr, u16& out) = 0;
    virtual bool ExclusiveRead32(u64 vaddr, u32& out) = 0;
    virtual bool ExclusiveRead64(u64 vaddr, u64& out) = 0;
    virtual bool ExclusiveRead128(u64 vaddr, u128& out) = 0;

    virtual bool ExclusiveWrite8(u64 vaddr, u8 value) = 0;
    virtual bool ExclusiveWrite16(u64 vaddr, u16 value) = 0;
    virtual bool ExclusiveWrite32(u64 vaddr, u32 value) = 0;
    virtual bool ExclusiveWrite64(u64 vaddr, u64 value) = 0;
    virtual bool ExclusiveWrite128(u64 vaddr, u128 value) = 0;

    virtual void ClearExclusive() = 0;

    virtual u64 GetCntvct() const = 0;
    virtual u64 GetCntfrq() const {
        return 19'200'000ULL;
    }
    virtual void OnCodeRangeInvalidated(u64 /*vaddr*/, std::size_t /*size*/) {}
};

enum class OpKind : u8 {
    Nop,
    // Immediate ALU
    AddImm,
    SubImm,
    LogicalImm,
    MovWide,
    Adr,
    Adrp,
    Bitfield,
    Extr,
    // Register ALU
    AddSubReg,
    AddSubExtReg,
    AddSubCarry,
    LogicalReg,
    ShiftReg,
    MulDiv,
    CondSelect,
    CondCompare,
    BitManip,
    // Branches & System
    BranchImm,
    BranchLinkImm,
    BranchReg,
    BranchCond,
    CompareBranchZero,
    TestBitBranch,
    Svc,
    Brk,
    Mrs,
    Msr,
    // Memory
    LoadStoreImm,
    LoadStoreReg,
    LoadLiteral,
    LoadStorePair,
    LoadStoreExclusive,
    LoadStoreExclusivePair,
    AtomicRmw,
    LoadStoreFpImm,
    LoadStoreFpReg,
    LoadStoreFpLiteral,
    LoadStoreFpPair,
    // Scalar FP
    FpMoveReg,
    FpMoveGpr,
    FpMoveImm,
    FpUnary,
    FpBinary,
    FpFma,
    FpCompare,
    FpCondSelect,
    FpConvert,
    // Fused hot sequences (built during cached block recode)
    FusedAdrpAdd,
    FusedSubImmCbnz,
    FusedSubImmBcond,
    // Fail-closed sentinel
    Unsupported,
};

struct DecodedOp {
    OpKind kind{OpKind::Unsupported};
    u8 rd{};
    u8 rn{};
    u8 rm{};
    u8 ra{};      // ra, rt2, cond, or sub-opcode
    u8 flags{};   // bit0: 64-bit, bit1: set_flags/signed, bit2..7: op-specific
    u8 aux{};     // shift type, option, size, or sub-mode
    u8 insn_count{1};
    u32 raw{};
    u64 imm{};
    u64 imm2{};   // secondary immediate / branch target for fused ops
};

struct CachedBlock {
    static constexpr std::size_t MaxOps = 32;

    u64 start_pc{};
    u64 end_pc{};
    u32 generation{};
    u16 op_count{};
    u16 guest_insns{};
    std::array<DecodedOp, MaxOps> ops{};
};

/// Portable AArch64 live-recode and decoded-block cache interpreter.
/// Uses ordinary data memory only (no executable memory, no JIT, works on all platforms).
class CachedBlockEngine {
public:
    static constexpr std::size_t CacheEntries = 4096;
    static constexpr std::size_t PageBitmapBits = 4096;

    explicit CachedBlockEngine(MemoryBus& bus, bool enable_cache = true, bool enable_fusion = true);

    CpuState& State() noexcept {
        return state;
    }
    const CpuState& State() const noexcept {
        return state;
    }

    StopReason Run(u64 max_instructions = 0);
    StopReason Step();

    void RequestInterrupt() noexcept {
        interrupt_requested.store(true, std::memory_order_release);
    }
    void ClearInterrupt() noexcept {
        interrupt_requested.store(false, std::memory_order_release);
    }

    void ClearCache();
    void InvalidateRange(u64 vaddr, std::size_t size);

    void SetCacheEnabled(bool enabled);
    void SetFusionEnabled(bool enabled) noexcept {
        fusion_enabled = enabled;
    }

    u32 LastSvcNumber() const noexcept {
        return last_svc;
    }
    u64 LastFaultAddress() const noexcept {
        return last_fault_addr;
    }
    u64 LastUnsupportedPc() const noexcept {
        return last_unsupported_pc;
    }
    u32 LastUnsupportedInsn() const noexcept {
        return last_unsupported_insn;
    }
    const Stats& GetStats() const noexcept {
        return stats;
    }
    void ResetStats() noexcept {
        stats = {};
    }

private:
    static constexpr std::size_t CacheIndex(u64 pc) noexcept {
        const u64 word = pc >> 2;
        return static_cast<std::size_t>((word ^ (word >> 12)) & (CacheEntries - 1));
    }

    static constexpr std::size_t PageHash(u64 addr) noexcept {
        const u64 page = addr >> 12;
        return static_cast<std::size_t>((page ^ (page >> 12)) & (PageBitmapBits - 1));
    }

    void MarkCodePage(u64 start_pc, u64 end_pc) noexcept;
    bool PageMayContainCode(u64 addr, std::size_t size) const noexcept;
    void NotifyGuestWrite(u64 addr, std::size_t size);

    DecodedOp DecodeInstruction(u32 word, u64 pc) const;
    bool DecodeBlock(CachedBlock& block, u64 start_pc, bool allow_fusion);
    StopReason ExecuteOp(const DecodedOp& op, u64& next_pc);

    u64 ReadRegOrZero(u8 reg, bool is64) const noexcept;
    u64 ReadRegOrSp(u8 reg, bool is64) const noexcept;
    void WriteRegOrZero(u8 reg, u64 value, bool is64) noexcept;
    void WriteRegOrSp(u8 reg, u64 value, bool is64) noexcept;
    bool ConditionHolds(u8 cond) const noexcept;
    void SetNzcv(bool n, bool z, bool c, bool v) noexcept;

    MemoryBus& bus;
    CpuState state{};
    std::vector<CachedBlock> cache;
    std::array<u64, PageBitmapBits / 64> code_page_bitmap{};
    u32 generation{1};
    bool cache_enabled{true};
    bool fusion_enabled{true};
    std::atomic<bool> interrupt_requested{false};

    u32 last_svc{};
    u64 last_fault_addr{};
    u64 last_unsupported_pc{};
    u32 last_unsupported_insn{};
    Stats stats{};
};

} // namespace Core::CachedInterpreter
