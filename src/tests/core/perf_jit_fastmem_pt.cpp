// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace {

class ShortArenaEnv final : public Dynarmic::A64::UserCallbacks {
public:
    static constexpr std::size_t PageSize = 4096;
    static constexpr u32 AddressSpaceBits = 47;
    static constexpr u32 FastmemBits = 14; // 16 KiB fastmem [0, 0x4000)

    alignas(PageSize) std::array<u8, 4 * PageSize> fastmem{};
    alignas(PageSize) std::array<u8, 4 * PageSize> high_backing{};

    Common::PageTable page_table;
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::Jit* jit = nullptr;
    std::map<u64, u32> code;

    u64 read_callbacks = 0;
    u64 write_callbacks = 0;
    u64 svc_calls = 0;
    bool halt_on_svc = true;

    ShortArenaEnv() {
        page_table.Resize(AddressSpaceBits, 12);
        page_table.fastmem_arena = fastmem.data();
    }

    void MapPage(u64 vaddr, u8* host_page, bool marked = false,
                 Common::PageType type = Common::PageType::Memory) {
        const u64 page = vaddr >> 12;
        page_table.entries.CommitRegion(page, page + 1);
        const uintptr_t delta = reinterpret_cast<uintptr_t>(host_page) - static_cast<uintptr_t>(page << 12);
        page_table.entries.GetUnchecked(page).Store(marked, type, 1, delta);
    }

    std::optional<u32> MemoryReadCode(u64 vaddr) override {
        if (const auto it = code.find(vaddr); it != code.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    u64 MemoryRead(u64 vaddr, std::size_t size) override {
        ++read_callbacks;
        u64 out = 0;
        for (std::size_t i = 0; i < size; ++i) {
            const u64 byte_vaddr = vaddr + i;
            const u64 page = byte_vaddr >> 12;
            u8 byte_val = 0xAA;
            if (page < page_table.entries.size()) {
                if (const uintptr_t delta = page_table.entries[page].Pointer(true); delta != 0) {
                    byte_val = *reinterpret_cast<const u8*>(delta + static_cast<uintptr_t>(byte_vaddr));
                }
            }
            out |= static_cast<u64>(byte_val) << (i * 8);
        }
        return out;
    }

    Dynarmic::A64::Vector MemoryRead128(u64 vaddr) override {
        return {MemoryRead(vaddr, 8), MemoryRead(vaddr + 8, 8)};
    }

    void MemoryWrite(u64 vaddr, u64 value, std::size_t size) override {
        ++write_callbacks;
        for (std::size_t i = 0; i < size; ++i) {
            const u64 byte_vaddr = vaddr + i;
            const u64 page = byte_vaddr >> 12;
            if (page < page_table.entries.size()) {
                if (const uintptr_t delta = page_table.entries[page].Pointer(true); delta != 0) {
                    *reinterpret_cast<u8*>(delta + static_cast<uintptr_t>(byte_vaddr)) =
                        static_cast<u8>(value >> (i * 8));
                }
            }
        }
    }

    void MemoryWrite128(u64 vaddr, Dynarmic::A64::Vector value) override {
        MemoryWrite(vaddr, value[0], 8);
        MemoryWrite(vaddr + 8, value[1], 8);
    }

    bool MemoryWriteExclusive32(u64 vaddr, u32 value, u32 expected) override {
        const u32 cur = static_cast<u32>(MemoryRead(vaddr, 4));
        if (cur != expected) {
            return false;
        }
        MemoryWrite(vaddr, value, 4);
        return true;
    }

    bool MemoryWriteExclusive64(u64 vaddr, u64 value, u64 expected) override {
        const u64 cur = MemoryRead(vaddr, 8);
        if (cur != expected) {
            return false;
        }
        MemoryWrite(vaddr, value, 8);
        return true;
    }

    void CallSVC(u32 swi) override {
        ++svc_calls;
        if (halt_on_svc || swi == 0) {
            jit->HaltExecution();
        }
    }

    void ExceptionRaised(u64, Dynarmic::A64::Exception) override {
        jit->HaltExecution();
    }

    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 10000; }
    u64 GetCNTPCT() override { return 123456; }

    Dynarmic::A64::UserConfig MakeConfig() {
        Dynarmic::A64::UserConfig conf{};
        conf.callbacks = this;
        conf.global_monitor = &monitor;
        conf.processor_id = 0;
        conf.code_cache_size = 8 * 1024 * 1024;
        conf.enable_cycle_counting = false;
        conf.unsafe_optimizations = true;
        conf.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;

        conf.page_table = reinterpret_cast<void**>(
            const_cast<Common::PageTable::PageEntryData*>(page_table.entries.data()));
        conf.page_table_address_space_bits = AddressSpaceBits;
        conf.page_table_pointer_mask = Common::PageTable::ATTRIBUTE_MASK;
        conf.page_table_marked_bit = u8{0};
        conf.page_table_sign_extension = u8(Common::PageTable::SIGN_BIT);
        conf.silently_mirror_page_table = false;
        conf.absolute_offset_page_table = true;
        conf.detect_misaligned_access_via_page_table = 16 | 32 | 64 | 128;
        conf.only_detect_misalignment_via_page_table_on_page_boundary = true;

        conf.fastmem_pointer = reinterpret_cast<uintptr_t>(fastmem.data());
        conf.fastmem_address_space_bits = FastmemBits;
        conf.silently_mirror_fastmem = false;
        conf.fastmem_exclusive_access = true;
        conf.native_lse_atomics = true;
        conf.halt_check_on_back_edges = true;
        return conf;
    }
};

} // namespace

TEST_CASE("Short fastmem arena routes out-of-arena accesses through inline page table with negative SIGN_BIT deltas", "[core][jit]") {
    auto env_ptr = std::make_unique<ShortArenaEnv>();
    auto& env = *env_ptr;

    // Map fastmem page 0x1000
    env.MapPage(0x1000, env.fastmem.data() + 0x1000);

    // Map high guest pages at 0x7F00'0000'0000..0x7F00'0000'2000 (> all 47-bit user heap pointers,
    // guaranteeing negative absolute-offset page table deltas on every 64-bit OS!)
    constexpr u64 HighBase = 0x7F00'0000'0000ULL;
    env.MapPage(HighBase, env.high_backing.data());
    env.MapPage(HighBase + 0x1000, env.high_backing.data() + 0x1000);
    // Marked page (e.g. RasterizerCachedMemory) at HighBase + 0x2000
    env.MapPage(HighBase + 0x2000, env.high_backing.data() + 0x2000, true,
                Common::PageType::RasterizerCachedMemory);

    REQUIRE(static_cast<s64>(env.page_table.entries[HighBase >> 12].Pointer()) < 0);

    constexpr u64 CodeBase = 0x1000'0000ULL;
    // STR X2, [X1]; LDR X3, [X1]; STR X4, [X0]; LDR X5, [X0]; SVC #0
    env.code[CodeBase + 0x00] = 0xF9000022; // str x2, [x1]
    env.code[CodeBase + 0x04] = 0xF9400023; // ldr x3, [x1]
    env.code[CodeBase + 0x08] = 0xF9000004; // str x4, [x0]
    env.code[CodeBase + 0x0C] = 0xF9400005; // ldr x5, [x0]
    env.code[CodeBase + 0x10] = 0xD4000001; // svc #0

    Dynarmic::A64::Jit jit{env.MakeConfig()};
    env.jit = &jit;

    // 1. Mapped fastmem (< 16 KiB) + mapped high page (>= 16 KiB, negative PT delta): 0 callbacks!
    jit.SetPC(CodeBase);
    jit.SetRegister(0, 0x1008);
    jit.SetRegister(1, HighBase + 0x18);
    jit.SetRegister(2, 0xCAFEBABE'12345678ULL);
    jit.SetRegister(4, 0xDEADBEEF'87654321ULL);
    jit.Run();

    REQUIRE(jit.GetRegister(3) == 0xCAFEBABE'12345678ULL);
    REQUIRE(jit.GetRegister(5) == 0xDEADBEEF'87654321ULL);
    REQUIRE(env.read_callbacks == 0);
    REQUIRE(env.write_callbacks == 0);

    // 2. Intra-page unaligned access at HighBase + 3 stays on inline PT (0 callbacks)
    jit.SetPC(CodeBase);
    jit.SetRegister(0, 0x1010);
    jit.SetRegister(1, HighBase + 3);
    jit.SetRegister(2, 0x11223344'55667788ULL);
    jit.Run();
    REQUIRE(jit.GetRegister(3) == 0x11223344'55667788ULL);
    REQUIRE(env.read_callbacks == 0);
    REQUIRE(env.write_callbacks == 0);

    // 3. Cross-page boundary access at HighBase + 0xFFD falls back cleanly to callbacks
    jit.SetPC(CodeBase);
    jit.SetRegister(0, 0x1020);
    jit.SetRegister(1, HighBase + 0xFFD);
    jit.SetRegister(2, 0xAABBCCDD'EEFF0011ULL);
    jit.Run();
    REQUIRE(jit.GetRegister(3) == 0xAABBCCDD'EEFF0011ULL);
    REQUIRE(env.read_callbacks == 1);
    REQUIRE(env.write_callbacks == 1);

    // 4. Marked page (RasterizerCachedMemory) at HighBase + 0x2000 falls back to callbacks
    env.read_callbacks = 0;
    env.write_callbacks = 0;
    jit.SetPC(CodeBase);
    jit.SetRegister(0, 0x1030);
    jit.SetRegister(1, HighBase + 0x2010);
    jit.SetRegister(2, 0x01020304'05060708ULL);
    jit.Run();
    REQUIRE(jit.GetRegister(3) == 0x01020304'05060708ULL);
    REQUIRE(env.read_callbacks == 1);
    REQUIRE(env.write_callbacks == 1);

    // 5. Unmapped page at HighBase + 0x3000 falls back to callbacks without crashing
    env.read_callbacks = 0;
    env.write_callbacks = 0;
    jit.SetPC(CodeBase);
    jit.SetRegister(0, 0x1040);
    jit.SetRegister(1, HighBase + 0x3000);
    jit.SetRegister(2, 0x99999999'99999999ULL);
    jit.Run();
    REQUIRE(jit.GetRegister(3) == 0xAAAAAAAA'AAAAAAAAULL);
    REQUIRE(env.read_callbacks == 1);
    REQUIRE(env.write_callbacks == 1);
}

TEST_CASE("JIT preserves deep RSB across SVC exits and handles barriers, indirect jumps, and LSE atomics", "[core][jit]") {
    auto env_ptr = std::make_unique<ShortArenaEnv>();
    auto& env = *env_ptr;
    constexpr u64 HighMem = 0x7F00'0000'0000ULL;
    env.MapPage(0x1000, env.fastmem.data() + 0x1000);
    env.MapPage(HighMem, env.high_backing.data());

    // Build a 20-deep recursive function chain + indirect BLR + barriers + LSE LDADDAL & CASAL:
    // 0x2000'0000 (main):
    //   blr x8               ; indirect call to recurse(x0 = 20)
    //   dmb ishld
    //   dmb ishst
    //   dmb ish
    //   dsb ish
    //   ldaddal x2, x3, [x1] ; atomic add on HighMem
    //   casal   x4, x5, [x1] ; atomic cas on HighMem
    //   svc #0               ; halt
    //
    // 0x2000'0100 (recurse):
    //   cbz x0, base_case (+0x18 -> 0x2000'0118)
    //   str x30, [sp, #-16]!
    //   sub x0, x0, #1
    //   svc #1               ; exit to host and resume (tests persistent RSB across Jit::Run!)
    //   bl  0x2000'0100
    //   ldr x30, [sp], #16
    // base_case (0x2000'0118):
    //   add x6, x6, #1
    //   ret
    constexpr u64 MainPC = 0x2000'0000ULL;
    constexpr u64 RecursePC = 0x2000'0100ULL;

    env.code[MainPC + 0x00] = 0xD63F0100; // blr x8
    env.code[MainPC + 0x04] = 0xD50339BF; // dmb ishld
    env.code[MainPC + 0x08] = 0xD5033ABF; // dmb ishst
    env.code[MainPC + 0x0C] = 0xD5033BBF; // dmb ish
    env.code[MainPC + 0x10] = 0xD5033B9F; // dsb ish
    env.code[MainPC + 0x14] = 0xF8E20023; // ldaddal x2, x3, [x1]
    env.code[MainPC + 0x18] = 0xC8E4FC25; // casal x4, x5, [x1]
    env.code[MainPC + 0x1C] = 0xD4000001; // svc #0

    env.code[RecursePC + 0x00] = 0xB40000C0; // cbz x0, +0x18 (0x2000'0118)
    env.code[RecursePC + 0x04] = 0xF81F0FFE; // str x30, [sp, #-16]!
    env.code[RecursePC + 0x08] = 0xD1000400; // sub x0, x0, #1
    env.code[RecursePC + 0x0C] = 0xD4000021; // svc #1
    env.code[RecursePC + 0x10] = 0x97FFFFFC; // bl 0x2000'0100 (-16 bytes)
    env.code[RecursePC + 0x14] = 0xF84107FE; // ldr x30, [sp], #16
    env.code[RecursePC + 0x18] = 0x910004C6; // add x6, x6, #1
    env.code[RecursePC + 0x1C] = 0xD65F03C0; // ret

    *reinterpret_cast<u64*>(env.high_backing.data() + 0x40) = 100;

    Dynarmic::A64::Jit jit{env.MakeConfig()};
    env.jit = &jit;
    env.halt_on_svc = true;

    jit.SetPC(MainPC);
    jit.SetSP(0x1F00);
    jit.SetRegister(0, 20);
    jit.SetRegister(1, HighMem + 0x40);
    jit.SetRegister(2, 50);  // ldaddal +50 -> 150 (old = 100 in x3)
    jit.SetRegister(4, 150); // casal expected = 150 -> desired = 999 (old = 150 in x4)
    jit.SetRegister(5, 999);
    jit.SetRegister(6, 0);
    jit.SetRegister(8, RecursePC);

    // Run across all 20 SVC #1 exits plus the final SVC #0
    for (int i = 0; i < 25 && jit.GetPC() != MainPC + 0x20; ++i) {
        jit.Run();
    }

    REQUIRE(env.svc_calls == 21);
    REQUIRE(jit.GetRegister(6) == 21);
    REQUIRE(jit.GetRegister(3) == 100);
    REQUIRE(jit.GetRegister(4) == 150);
    REQUIRE(*reinterpret_cast<u64*>(env.high_backing.data() + 0x40) == 999);
    REQUIRE(env.read_callbacks == 0);
    REQUIRE(env.write_callbacks == 0);

    // ClearCache and InvalidateCacheRange must scrub metadata/RSB/FastDispatch cleanly and re-run
    jit.ClearCache();
    jit.InvalidateCacheRange(MainPC, 0x200);
    jit.SetPC(MainPC);
    jit.SetRegister(0, 5);
    jit.SetRegister(2, 1);
    jit.SetRegister(4, 1000);
    jit.SetRegister(5, 2026);
    jit.SetRegister(6, 0);
    for (int i = 0; i < 10 && jit.GetPC() != MainPC + 0x20; ++i) {
        jit.Run();
    }
    REQUIRE(jit.GetRegister(6) == 6);
    REQUIRE(jit.GetRegister(3) == 999);
    REQUIRE(jit.GetRegister(4) == 1000);
    REQUIRE(*reinterpret_cast<u64*>(env.high_backing.data() + 0x40) == 2026);
}
