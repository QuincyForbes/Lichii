// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <cstring>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/arm/cached_interpreter/cached_block_engine.h"

namespace {

using namespace Core::CachedInterpreter;

class FlatMemoryBus final : public MemoryBus {
public:
    explicit FlatMemoryBus(std::size_t size = 0x20000, u64 base_ = 0x10000)
        : base{base_}, mem(size, 0) {}

    u64 Base() const noexcept {
        return base;
    }

    void LoadProgram(u64 addr, std::span<const u32> words) {
        const std::size_t off = static_cast<std::size_t>(addr - base);
        std::memcpy(mem.data() + off, words.data(), words.size_bytes());
    }

    std::optional<u32> ReadCode32(u64 vaddr) override {
        u32 v = 0;
        if (!Read32(vaddr, v)) {
            return std::nullopt;
        }
        return v;
    }

    bool Read8(u64 vaddr, u8& out) override {
        if (!Valid(vaddr, 1)) return false;
        out = mem[vaddr - base];
        return true;
    }
    bool Read16(u64 vaddr, u16& out) override {
        if (!Valid(vaddr, 2)) return false;
        std::memcpy(&out, &mem[vaddr - base], 2);
        return true;
    }
    bool Read32(u64 vaddr, u32& out) override {
        if (!Valid(vaddr, 4)) return false;
        std::memcpy(&out, &mem[vaddr - base], 4);
        return true;
    }
    bool Read64(u64 vaddr, u64& out) override {
        if (!Valid(vaddr, 8)) return false;
        std::memcpy(&out, &mem[vaddr - base], 8);
        return true;
    }
    bool Read128(u64 vaddr, u128& out) override {
        if (!Valid(vaddr, 16)) return false;
        std::memcpy(&out[0], &mem[vaddr - base], 8);
        std::memcpy(&out[1], &mem[vaddr - base + 8], 8);
        return true;
    }

    bool Write8(u64 vaddr, u8 value) override {
        if (!Valid(vaddr, 1)) return false;
        mem[vaddr - base] = value;
        return true;
    }
    bool Write16(u64 vaddr, u16 value) override {
        if (!Valid(vaddr, 2)) return false;
        std::memcpy(&mem[vaddr - base], &value, 2);
        return true;
    }
    bool Write32(u64 vaddr, u32 value) override {
        if (!Valid(vaddr, 4)) return false;
        std::memcpy(&mem[vaddr - base], &value, 4);
        return true;
    }
    bool Write64(u64 vaddr, u64 value) override {
        if (!Valid(vaddr, 8)) return false;
        std::memcpy(&mem[vaddr - base], &value, 8);
        return true;
    }
    bool Write128(u64 vaddr, u128 value) override {
        if (!Valid(vaddr, 16)) return false;
        std::memcpy(&mem[vaddr - base], &value[0], 8);
        std::memcpy(&mem[vaddr - base + 8], &value[1], 8);
        return true;
    }

    bool ExclusiveRead8(u64 vaddr, u8& out) override {
        if (!Read8(vaddr, out)) return false;
        res_addr = vaddr; res_val = {out, 0}; res_ok = true;
        return true;
    }
    bool ExclusiveRead16(u64 vaddr, u16& out) override {
        if (!Read16(vaddr, out)) return false;
        res_addr = vaddr; res_val = {out, 0}; res_ok = true;
        return true;
    }
    bool ExclusiveRead32(u64 vaddr, u32& out) override {
        if (!Read32(vaddr, out)) return false;
        res_addr = vaddr; res_val = {out, 0}; res_ok = true;
        return true;
    }
    bool ExclusiveRead64(u64 vaddr, u64& out) override {
        if (!Read64(vaddr, out)) return false;
        res_addr = vaddr; res_val = {out, 0}; res_ok = true;
        return true;
    }
    bool ExclusiveRead128(u64 vaddr, u128& out) override {
        if (!Read128(vaddr, out)) return false;
        res_addr = vaddr; res_val = out; res_ok = true;
        return true;
    }

    bool ExclusiveWrite8(u64 vaddr, u8 value) override {
        u8 cur = 0;
        if (!res_ok || res_addr != vaddr || !Read8(vaddr, cur) || cur != static_cast<u8>(res_val[0])) {
            res_ok = false; return false;
        }
        res_ok = false; return Write8(vaddr, value);
    }
    bool ExclusiveWrite16(u64 vaddr, u16 value) override {
        u16 cur = 0;
        if (!res_ok || res_addr != vaddr || !Read16(vaddr, cur) || cur != static_cast<u16>(res_val[0])) {
            res_ok = false; return false;
        }
        res_ok = false; return Write16(vaddr, value);
    }
    bool ExclusiveWrite32(u64 vaddr, u32 value) override {
        u32 cur = 0;
        if (!res_ok || res_addr != vaddr || !Read32(vaddr, cur) || cur != static_cast<u32>(res_val[0])) {
            res_ok = false; return false;
        }
        res_ok = false; return Write32(vaddr, value);
    }
    bool ExclusiveWrite64(u64 vaddr, u64 value) override {
        u64 cur = 0;
        if (!res_ok || res_addr != vaddr || !Read64(vaddr, cur) || cur != res_val[0]) {
            res_ok = false; return false;
        }
        res_ok = false; return Write64(vaddr, value);
    }
    bool ExclusiveWrite128(u64 vaddr, u128 value) override {
        u128 cur{};
        if (!res_ok || res_addr != vaddr || !Read128(vaddr, cur) || cur != res_val) {
            res_ok = false; return false;
        }
        res_ok = false; return Write128(vaddr, value);
    }

    void ClearExclusive() override {
        res_ok = false;
    }

    u64 GetCntvct() const override {
        return 0x1234'5678ULL;
    }

private:
    bool Valid(u64 vaddr, std::size_t sz) const noexcept {
        return vaddr >= base && (vaddr - base) + sz <= mem.size();
    }

    u64 base{};
    std::vector<u8> mem;
    u64 res_addr{};
    u128 res_val{};
    bool res_ok{false};
};

} // namespace

TEST_CASE("CachedBlockEngine executes fused loops and ADRP+ADD identically across modes", "[core][arm]") {
    // Program at 0x10000:
    //   0x10000: adrp x2, +0x10000       (0x90000082 -> x2 = 0x20000)
    //   0x10004: add  x2, x2, #0x100     (0x91040042 -> x2 = 0x20100, fused!)
    //   0x10008: movz x0, #0             (0xD2800000)
    //   0x1000C: movz x1, #100           (0xD2800C81)
    // loop (0x10010):
    //   0x10010: add  x0, x0, x1         (0x8B010000)
    //   0x10014: sub  x1, x1, #1         (0xD1000421)
    //   0x10018: cbnz x1, -8 (0x10010)   (0xB5FFFFC1, fused with sub!)
    //   0x1001C: str  x0, [x2]           (0xF9000040)
    //   0x10020: svc  #0x29              (0xD4000521)
    static constexpr u32 kProg[] = {
        0x90000082U,
        0x91040042U,
        0xD2800000U,
        0xD2800C81U,
        0x8B010000U,
        0xD1000421U,
        0xB5FFFFC1U,
        0xF9000040U,
        0xD4000521U,
    };

    for (int mode = 0; mode < 3; ++mode) {
        FlatMemoryBus bus;
        bus.LoadProgram(0x10000, kProg);
        const bool cached = mode >= 1;
        const bool fused = mode == 2;
        CachedBlockEngine engine{bus, cached, fused};
        engine.State().pc = 0x10000;

        const StopReason stop = engine.Run(0);
        CHECK(stop == StopReason::SupervisorCall);
        CHECK(engine.LastSvcNumber() == 0x29U);
        CHECK(engine.State().pc == 0x10024ULL);
        CHECK(engine.State().x[0] == 5050ULL);
        CHECK(engine.State().x[1] == 0ULL);
        CHECK(engine.State().x[2] == 0x20100ULL);
        u64 stored = 0;
        REQUIRE(bus.Read64(0x20100, stored));
        CHECK(stored == 5050ULL);
        CHECK(engine.GetStats().instructions_retired == 306ULL);

        if (cached) {
            CHECK(engine.GetStats().blocks_decoded == 3ULL);
            CHECK(engine.GetStats().cache_hits == 98ULL);
        }
        if (fused) {
            CHECK(engine.GetStats().ops_fused == 3ULL);
        }
    }
}

TEST_CASE("CachedBlockEngine invalidates cached block on self-modifying store", "[core][arm]") {
    // 0x10000: movz x0, #0           (0xD2800000)
    // 0x10004: movz x1, #2           (0xD2800041)
    // patch_target (0x10008):
    // 0x10008: add  x0, x0, #2       (0x91000800)  <-- overwritten on pass 1 with add x0, x0, #10!
    // 0x1000C: sub  x1, x1, #1       (0xD1000421)
    // 0x10010: cbz  x1, +20 (0x10024)(0xB40000A1)
    // 0x10014: adr  x2, -12 (0x10008)(0x10FFFFA2)
    // 0x10018: movz w3, #0x2800      (0x52850003)
    // 0x1001C: movk w3, #0x9100,lsl16(0x72B22003) -> 0x91002800 = add x0, x0, #10
    // 0x10020: str  w3, [x2]         (0xB9000043)
    // 0x10024: cbnz x1, -28 (0x10008)(0xB5FFFF21)
    // 0x10028: svc  #1               (0xD4000021)
    static constexpr u32 kSelfMod[] = {
        0xD2800000U,
        0xD2800041U,
        0x91000800U,
        0xD1000421U,
        0xB40000A1U,
        0x10FFFFA2U,
        0x52850003U,
        0x72B22003U,
        0xB9000043U,
        0xB5FFFF21U,
        0xD4000021U,
    };

    FlatMemoryBus bus;
    bus.LoadProgram(0x10000, kSelfMod);
    CachedBlockEngine engine{bus, true, true};
    engine.State().pc = 0x10000;

    CHECK(engine.Run(0) == StopReason::SupervisorCall);
    // Pass 1 added 2, then patched 0x10008 to add 10, pass 2 added 10 => x0 == 12 (not stale 4).
    CHECK(engine.State().x[0] == 12ULL);
    CHECK(engine.GetStats().invalidations >= 1ULL);
}

TEST_CASE("CachedBlockEngine handles atomics, exclusives, scalar FP, and unsupported fail-closed",
          "[core][arm]") {
    FlatMemoryBus bus;
    REQUIRE(bus.Write32(0x18000, 40U));
    REQUIRE(bus.Write32(0x18004, 100U));

    // 0x10000: movz x1, #0x8000, lsl #0 + movk x1, #1, lsl #16 -> x1 = 0x18000
    // 0x10008: movz w2, #5
    // 0x1000C: ldadd w2, w3, [x1]        (0xB8220023 -> mem[0x18000]=45, w3=40)
    // 0x10010: ldxr  w4, [x1, #0]        (add x5, x1, #4 = 0x91001025; ldxr w4, [x5] = 0x885F7CA4)
    // 0x10018: add   w4, w4, #7          (0x11001C84)
    // 0x1001C: stxr  w6, w4, [x5]        (0x88067CA4 -> w6=0, mem[0x18004]=107)
    // 0x10020: scvtf s0, w3              (0x1E220060 -> s0 = 40.0f)
    // 0x10024: fmov  s1, #2.0f           (0x1E201001 -> imm8=0x00 -> 2.0f)
    // 0x10028: fmov  s2, #1.5f           (0x1E3F1002 -> wait: use scvtf s2 from w2=5 -> 5.0f: 0x1E220042)
    // 0x1002C: fmadd s3, s0, s1, s2      (0x1F010803 -> 40*2 + 5 = 85.0f)
    // 0x10030: fcvtzs x7, s3             (0x9E380067 -> x7 = 85)
    // 0x10034: svc   #7                  (0xD40000E1)
    // 0x10038: udf   #0                  (0x00000000 -> UnsupportedInstruction)
    static constexpr u32 kOps[] = {
        0xD2900001U, // movz x1, #0x8000
        0xF2A00021U, // movk x1, #0x1, lsl #16 -> 0x18000
        0x528000A2U, // movz w2, #5
        0xB8220023U, // ldadd w2, w3, [x1]
        0x91001025U, // add x5, x1, #4
        0x885F7CA4U, // ldxr w4, [x5]
        0x11001C84U, // add w4, w4, #7
        0x88067CA4U, // stxr w6, w4, [x5]
        0x1E220060U, // scvtf s0, w3
        0x1E201001U, // fmov s1, #2.0
        0x1E220042U, // scvtf s2, w2
        0x1F010803U, // fmadd s3, s0, s1, s2
        0x9E380067U, // fcvtzs x7, s3
        0xD40000E1U, // svc #7
        0x00000000U, // udf #0
    };

    bus.LoadProgram(0x10000, kOps);
    CachedBlockEngine engine{bus, true, true};
    engine.State().pc = 0x10000;

    // Budgeted run of first 5 instructions stops cleanly mid-stream
    CHECK(engine.Run(5) == StopReason::StepCompleted);
    CHECK(engine.State().pc == 0x10014ULL);

    // Continue to SVC
    CHECK(engine.Run(0) == StopReason::SupervisorCall);
    CHECK(engine.LastSvcNumber() == 7U);
    CHECK(engine.State().x[3] == 40ULL);
    CHECK(engine.State().x[6] == 0ULL);
    CHECK(engine.State().x[7] == 85ULL);

    u32 m0 = 0, m1 = 0;
    REQUIRE(bus.Read32(0x18000, m0));
    REQUIRE(bus.Read32(0x18004, m1));
    CHECK(m0 == 45U);
    CHECK(m1 == 107U);

    // Next instruction at 0x10038 is UDF #0 -> fail-closed UnsupportedInstruction
    CHECK(engine.Run(0) == StopReason::UnsupportedInstruction);
    CHECK(engine.LastUnsupportedPc() == 0x10038ULL);
    CHECK(engine.LastUnsupportedInsn() == 0U);
}
