// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef HAS_HYPERVISOR

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/host_memory.h"
#include "common/page_table.h"
#include "core/arm/hypervisor/arm_hypervisor.h"
#include "core/arm/hypervisor/hv_vm.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"

namespace {

struct HypervisorTestEnv {
    ScopedSystemTestContext context;
    Core::System system;
    Kernel::KProcess process;
    std::unique_ptr<Core::HvVm> vm;
    Core::ArmHypervisor hv;

    static constexpr u64 CodeVa = 0x10000;
    static constexpr u64 DataVa = 0x20000;

    HypervisorTestEnv()
        : process{system.Kernel()},
          vm{Core::HvVm::CreateForTest(system.DeviceMemory().buffer.BackingBasePointer(),
                                       0x100000)},
          hv{system, *vm, 0} {
        system.Initialize();
        auto& table = process.GetPageTable().GetImpl();
        table.Resize(39, 12);
        u8* const backing = system.DeviceMemory().buffer.BackingBasePointer() + 0x20000;
        std::memset(backing, 0, 32 * 0x1000);
        for (std::size_t i = 0; i < 32; ++i) {
            const u64 vpage = (CodeVa >> 12) + i;
            u8* const host = backing + i * 0x1000;
            table.entries.GetAndFault(vpage).Store(
                false, Common::PageType::Memory, 1,
                reinterpret_cast<uintptr_t>(host) - (vpage << 12));
        }
        process.GetMemory().SetCurrentPageTable(process);
    }

    ~HypervisorTestEnv() {
        system.Kernel().UnregisterKernelObject(&process);
    }

    bool RunOne(u32 insn, Kernel::Svc::ThreadContext& ctx) {
        auto& mem = process.GetMemory();
        mem.Write32(CodeVa, insn);
        hv.SetContext(ctx);
        const bool ok = hv.EmulateLoadStore(mem, CodeVa);
        hv.GetContext(ctx);
        return ok;
    }
};

} // namespace

TEST_CASE("ArmHypervisor emulates STNP/LDNP, SIMD STR Q reg-offset, STLR/STLXR, CASP, and LD2/ST2",
          "[arm_hypervisor]") {
    HypervisorTestEnv env;
    auto& mem = env.process.GetMemory();

    // 1. Integer 64-bit / 32-bit STNP & LDNP + 128-bit Q STNP & LDNP across page boundary.
    {
        Kernel::Svc::ThreadContext ctx{};
        ctx.r[0] = 0x1122334455667788ULL;
        ctx.r[1] = 0x99AABBCCDDEEFF00ULL;
        ctx.r[2] = HypervisorTestEnv::DataVa;
        REQUIRE(env.RunOne(0xA8010440U, ctx)); // STNP X0, X1, [X2, #16]
        REQUIRE(mem.Read64(HypervisorTestEnv::DataVa + 16) == 0x1122334455667788ULL);
        REQUIRE(mem.Read64(HypervisorTestEnv::DataVa + 24) == 0x99AABBCCDDEEFF00ULL);
        REQUIRE(env.RunOne(0xA8411043U, ctx)); // LDNP X3, X4, [X2, #16]
        REQUIRE(ctx.r[3] == ctx.r[0]);
        REQUIRE(ctx.r[4] == ctx.r[1]);

        ctx.r[2] = HypervisorTestEnv::DataVa + 64;
        REQUIRE(env.RunOne(0x283F0440U, ctx)); // STNP W0, W1, [X2, #-8]
        ctx.r[5] = ~u64{0};
        ctx.r[6] = ~u64{0};
        REQUIRE(env.RunOne(0x287F1845U, ctx)); // LDNP W5, W6, [X2, #-8]
        REQUIRE(ctx.r[5] == 0x55667788ULL);
        REQUIRE(ctx.r[6] == 0xDDEEFF00ULL);

        ctx.v[0] = u128{0x0102030405060708ULL, 0x1112131415161718ULL};
        ctx.v[1] = u128{0x2122232425262728ULL, 0x3132333435363738ULL};
        ctx.r[2] = HypervisorTestEnv::DataVa + 0xFD0;
        REQUIRE(env.RunOne(0xAC010440U, ctx)); // STNP Q0, Q1, [X2, #32] -> straddles 0x20FF0..0x21010
        REQUIRE(mem.Read128(HypervisorTestEnv::DataVa + 0xFF0) == ctx.v[0]);
        REQUIRE(mem.Read128(HypervisorTestEnv::DataVa + 0x1000) == ctx.v[1]);
        REQUIRE(env.RunOne(0xAC411444U, ctx)); // LDNP Q4, Q5, [X2, #32]
        REQUIRE(ctx.v[4] == ctx.v[0]);
        REQUIRE(ctx.v[5] == ctx.v[1]);
    }

    // 2. Register-offset SIMD STR Q / LDR Q and AdvSIMD ST2/LD2/LD1R/ST1-lane.
    {
        Kernel::Svc::ThreadContext ctx{};
        ctx.v[3] = u128{0xCAFEBABEDEADBEEFULL, 0x0F1E2D3C4B5A6978ULL};
        ctx.r[1] = HypervisorTestEnv::DataVa + 0x100;
        ctx.r[2] = 5;
        REQUIRE(env.RunOne(0x3CA27823U, ctx)); // STR Q3, [X1, X2, LSL #4]
        REQUIRE(mem.Read128(HypervisorTestEnv::DataVa + 0x150) == ctx.v[3]);
        REQUIRE(env.RunOne(0x3CE27827U, ctx)); // LDR Q7, [X1, X2, LSL #4]
        REQUIRE(ctx.v[7] == ctx.v[3]);

        ctx.v[0] = u128{0x00000002'00000001ULL, 0x00000004'00000003ULL};
        ctx.v[1] = u128{0x00000020'00000010ULL, 0x00000040'00000030ULL};
        ctx.r[1] = HypervisorTestEnv::DataVa + 0x200;
        REQUIRE(env.RunOne(0x4C9F8820U, ctx)); // ST2 {V0.4S, V1.4S}, [X1], #32
        REQUIRE(ctx.r[1] == HypervisorTestEnv::DataVa + 0x220);
        ctx.r[5] = HypervisorTestEnv::DataVa + 0x200;
        REQUIRE(env.RunOne(0x4C4088A2U, ctx)); // LD2 {V2.4S, V3.4S}, [X5]
        REQUIRE(ctx.v[2] == ctx.v[0]);
        REQUIRE(ctx.v[3] == ctx.v[1]);
        REQUIRE(env.RunOne(0x4D40C8A6U, ctx)); // LD1R {V6.4S}, [X5]
        REQUIRE(ctx.v[6] == u128{0x00000001'00000001ULL, 0x00000001'00000001ULL});
    }

    // 3. STLR / LDAR, LDAXR / STLXR, and CASPAL.
    {
        const u64 addr = HypervisorTestEnv::DataVa + 0x300;
        Kernel::Svc::ThreadContext ctx{};
        ctx.r[0] = 0xFEEDFACE12345678ULL;
        ctx.r[1] = addr;
        REQUIRE(env.RunOne(0xC89FFC20U, ctx)); // STLR X0, [X1]
        REQUIRE(env.RunOne(0xC8DFFC22U, ctx)); // LDAR X2, [X1]
        REQUIRE(ctx.r[2] == 0xFEEDFACE12345678ULL);

        mem.Write64(addr, 41);
        ctx.r[0] = 99;
        ctx.r[3] = 77;
        REQUIRE(env.RunOne(0xC803FC20U, ctx)); // Unpaired STLXR W3, X0, [X1] -> W3 = 1
        REQUIRE(ctx.r[3] == 1);
        REQUIRE(mem.Read64(addr) == 41);

        mem.Write32(HypervisorTestEnv::CodeVa, 0xC85FFC22U);     // LDAXR X2, [X1]
        mem.Write32(HypervisorTestEnv::CodeVa + 4, 0xC803FC20U); // STLXR W3, X0, [X1]
        env.hv.SetContext(ctx);
        REQUIRE(env.hv.EmulateLoadStore(mem, HypervisorTestEnv::CodeVa));
        REQUIRE(env.hv.EmulateLoadStore(mem, HypervisorTestEnv::CodeVa + 4));
        env.hv.GetContext(ctx);
        REQUIRE(ctx.r[2] == 41);
        REQUIRE(ctx.r[3] == 0);
        REQUIRE(mem.Read64(addr) == 99);

        const u64 pair_addr = HypervisorTestEnv::DataVa + 0x340;
        mem.Write128(pair_addr, u128{10, 20});
        ctx.r[0] = 10;
        ctx.r[1] = 20;
        ctx.r[2] = 30;
        ctx.r[3] = 40;
        ctx.r[4] = pair_addr;
        REQUIRE(env.RunOne(0x4860FC82U, ctx)); // CASPAL X0, X1, X2, X3, [X4]
        REQUIRE(mem.Read128(pair_addr) == u128{30, 40});
    }
}

TEST_CASE("HvVm defers write revocation during concurrent TrackedStep and repairs uncached PTEs",
          "[arm_hypervisor]") {
    std::vector<u8> dram(0x40000, 0);
    auto vm = Core::HvVm::CreateForTest(dram.data(), dram.size());
    REQUIRE(vm != nullptr);

    constexpr u64 BaseVa = 0x100000;
    vm->Map(BaseVa, 0, 0x10000,
            Common::MemoryPermission::ReadWrite | Common::MemoryPermission::Execute);

    const u64 page = BaseVa + 0x1000;
    vm->ProtectAccess(BaseVa, 0x4000, Common::MemoryPermission::Read);
    REQUIRE(vm->BeginStepAccess(page, true));
    REQUIRE(vm->BeginStepAccess(page, true));
    REQUIRE(vm->EffectivePermits(page, true));
    REQUIRE_FALSE(vm->Permits(page, true));

    vm->ProtectAccess(BaseVa, 0x4000, Common::MemoryPermission::Read);
    REQUIRE(vm->EffectivePermits(page, true));
    vm->EndStepAccess(page, true);
    REQUIRE(vm->EffectivePermits(page, true));
    vm->EndStepAccess(page, true);
    REQUIRE_FALSE(vm->EffectivePermits(page, true));
    REQUIRE(vm->EffectivePermits(page, false));

    std::atomic<bool> stop{false};
    std::atomic<u32> violations{0};
    std::jthread protector{[&] {
        while (!stop.load(std::memory_order_relaxed)) {
            vm->ProtectAccess(BaseVa, 0x10000, Common::MemoryPermission::Read);
            vm->ProtectAccess(BaseVa, 0x4000, Common::MemoryPermission{});
        }
    }};
    std::vector<std::jthread> steppers;
    for (int c = 0; c < 6; ++c) {
        steppers.emplace_back([&, c] {
            for (u32 iter = 0; iter < 500; ++iter) {
                const u64 p = BaseVa + ((static_cast<u64>(iter + c) & 15ULL) << 12);
                if (!vm->BeginStepAccess(p, true) || !vm->EffectivePermits(p, true)) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                vm->EndStepAccess(p, true);
            }
        });
    }
    steppers.clear();
    stop.store(true, std::memory_order_relaxed);
    protector.join();
    REQUIRE(violations.load() == 0);

    Common::PageTable pt;
    pt.Resize(39, 12);
    const u64 rw_page = BaseVa + 0x8000;
    const u64 ro_page = BaseVa + 0x9000;
    vm->Protect(ro_page, 0x1000, Common::MemoryPermission::Read);
    vm->ProtectAccess(rw_page, 0x1000, Common::MemoryPermission::Read);
    pt.entries.GetAndFault(rw_page >> 12)
        .Store(false, Common::PageType::Memory, 1,
               reinterpret_cast<uintptr_t>(dram.data()) - rw_page);
    pt.entries.GetAndFault(ro_page >> 12)
        .Store(false, Common::PageType::Memory, 1,
               reinterpret_cast<uintptr_t>(dram.data() + 0x1000) - ro_page);
    REQUIRE(vm->RestoreGuestAccess(pt, rw_page));
    REQUIRE(vm->Permits(rw_page, true));
    REQUIRE(vm->RestoreGuestAccess(pt, ro_page));
    REQUIRE_FALSE(vm->Permits(ro_page, true));
}

#endif // HAS_HYPERVISOR
