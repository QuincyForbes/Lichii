// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
#include <zstd.h>
#include <catch2/catch_test_macros.hpp>
#include "common/page_table.h"
#include "common/platform_events.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc/svc_nx2_zstd.h"
#include "core/hle/service/nx2_gsv.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"

TEST_CASE("Memory acquire reads and exclusive publication use mapped backing", "[hwacod_memory]") {
    ScopedSystemTestContext context;
    Core::System system;
    Core::Memory::Memory unbound{system};
    u32 value = 99;
    REQUIRE_FALSE(unbound.Read32Acquire(0x4000, value));
    REQUIRE(value == 0);

    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT { system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* page = system.DeviceMemory().buffer.BackingBasePointer() + 0x10000;
    std::memset(page, 0, 0x1000);
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(page) - 0x4000);
    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);
    REQUIRE(memory.Read32Acquire(0x4000, value));
    REQUIRE(value == 0);
    REQUIRE(memory.WriteExclusive32(0x4000, 0x12345678, 0));
    REQUIRE_FALSE(memory.WriteExclusive32(0x4000, 99, 0));
    REQUIRE(memory.Read32Acquire(0x4000, value));
    REQUIRE(value == 0x12345678);
    REQUIRE(memory.WriteExclusive32(0x4ffc, 0xffffffff, 0));
    REQUIRE(memory.Read32Acquire(0x4ffc, value));
    REQUIRE(value == 0xffffffff);
    for (const u64 address : {0x4001ULL, 0x5000ULL, 0x100000ULL}) {
        value = 99;
        REQUIRE_FALSE(memory.Read32Acquire(address, value));
        REQUIRE(value == 0);
    }

    for (u32 iteration = 0; iteration < 32; ++iteration) {
        CAPTURE(iteration);
        const u32 zero{};
        REQUIRE(memory.WriteBlock(0x4000, &zero, sizeof(zero)));
        std::array<u32, 16> expected{};
        expected.fill(iteration + 1);
        bool written{};
        std::jthread writer{[&] {
            written = memory.WriteBlock(0x4080, expected.data(), sizeof(expected)) &&
                      memory.WriteExclusive32(0x4000, 1, 0);
        }};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        bool acquired{};
        do {
            if (!memory.Read32Acquire(0x4000, value)) {
                break;
            }
            acquired = value == 1;
            if (!acquired) {
                std::this_thread::yield();
            }
        } while (!acquired && std::chrono::steady_clock::now() < deadline);
        std::array<u32, 16> observed{};
        const bool read = acquired && memory.ReadBlock(0x4080, observed.data(), sizeof(observed));
        writer.join();
        REQUIRE(written);
        REQUIRE(acquired);
        REQUIRE(read);
        REQUIRE(observed == expected);
    }
}

TEST_CASE("Virtual address ranges that wrap past the address space end are invalid",
          "[hwacod_memory]") {
    ScopedSystemTestContext context;
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT { system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* page = system.DeviceMemory().buffer.BackingBasePointer() + 0x10000;
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(page) - 0x4000);
    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);
    REQUIRE(memory.IsValidVirtualAddressRange(0x4000, 0x1000));
    REQUIRE(memory.IsValidVirtualAddressRange(0x4ff8, 8));
    REQUIRE_FALSE(memory.IsValidVirtualAddressRange(0x4ffc, 8));
    REQUIRE_FALSE(memory.IsValidVirtualAddressRange(~u64{0} - 3, 8));
    REQUIRE_FALSE(memory.IsValidVirtualAddressRange(0x4000, ~u64{0}));
}

TEST_CASE("Contiguous safe spans and zero-copy native zstd match split-page fallback",
          "[hwacod_memory]") {
    ScopedSystemTestContext context;
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT {
        Kernel::Svc::ResetNx2OotZstd();
        system.Kernel().UnregisterKernelObject(&process);
    };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    u8* const backing = system.DeviceMemory().buffer.BackingBasePointer() + 0x20000;
    std::memset(backing, 0, 0x10000);

    // Contiguous 4-page region at 0x10000..0x14000 (src) and 0x14000..0x18000 (dst).
    for (u64 p = 0; p < 8; ++p) {
        const u64 vpage = 0x10 + p;
        u8* const host = backing + p * 0x1000;
        table.entries.GetAndFault(vpage).Store(
            false, Common::PageType::Memory, 2,
            reinterpret_cast<uintptr_t>(host) - (vpage << 12));
    }
    // Non-contiguous 2-page destination at 0x20000..0x22000 (backed by reversed physical pages).
    table.entries.GetAndFault(0x20).Store(
        false, Common::PageType::Memory, 3,
        reinterpret_cast<uintptr_t>(backing + 0x9000) - 0x20000);
    table.entries.GetAndFault(0x21).Store(
        false, Common::PageType::Memory, 4,
        reinterpret_cast<uintptr_t>(backing + 0x8000) - 0x21000);
    // Debug page at 0x30000 must reject GetSafeReadSpan/GetSafeWriteSpan.
    table.entries.GetAndFault(0x30).Store(
        true, Common::PageType::DebugMemory, 5,
        reinterpret_cast<uintptr_t>(backing + 0xA000) - 0x30000);

    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);

    REQUIRE(memory.GetSafeReadSpan(0x10100, 0x3800) == backing + 0x100);
    REQUIRE(memory.GetSafeWriteSpan(0x14200, 0x3000) == backing + 0x4200);
    REQUIRE(memory.GetSafeReadSpan(0x20800, 0x1000) == nullptr);
    REQUIRE(memory.GetSafeWriteSpan(0x20800, 0x1000) == nullptr);
    REQUIRE(memory.GetSafeReadSpan(0x17800, 0x1000) == nullptr);
    REQUIRE(memory.GetSafeReadSpan(0x30000, 0x100) == nullptr);
    REQUIRE(memory.GetSafeWriteSpan(0x30000, 0x100) == nullptr);

    // Verify multi-page payload compresses and decompresses identically via zero-copy and fallback.
    std::vector<u8> payload(6000);
    for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<u8>((i * 37 + (i >> 5)) & 0xFF);
    }
    std::vector<u8> compressed(ZSTD_compressBound(payload.size()));
    const size_t csize =
        ZSTD_compress(compressed.data(), compressed.size(), payload.data(), payload.size(), 1);
    REQUIRE_FALSE(ZSTD_isError(csize));
    REQUIRE(memory.WriteBlock(0x10200, compressed.data(), csize));

    std::array<u64, 8> args_zero_copy{0, 0x14100, payload.size(), 0x10200, csize, 0, 0, 0};
    REQUIRE(Kernel::Svc::Nx2OotZstdDecompressForTest(memory, args_zero_copy) == nullptr);
    REQUIRE(args_zero_copy[0] == payload.size());

    std::array<u64, 8> args_split{0, 0x20100, payload.size(), 0x10200, csize, 0, 0, 0};
    REQUIRE(Kernel::Svc::Nx2OotZstdDecompressForTest(memory, args_split) == nullptr);
    REQUIRE(args_split[0] == payload.size());

    std::vector<u8> out_zero_copy(payload.size());
    std::vector<u8> out_split(payload.size());
    REQUIRE(memory.ReadBlock(0x14100, out_zero_copy.data(), out_zero_copy.size()));
    REQUIRE(memory.ReadBlock(0x20100, out_split.data(), out_split.size()));
    REQUIRE(out_zero_copy == payload);
    REQUIRE(out_split == payload);
}

TEST_CASE("Auto-Cool thermal FPS cap clamps Nx2VsyncRateHz without mutating Settings",
          "[hwacod_memory]") {
    const u16 saved_limit = Settings::values.frame_rate_limit.GetValue();
    SCOPE_EXIT {
        Common::PlatformEvents::SetThermalMaxFps(0);
        Settings::values.frame_rate_limit.SetValue(saved_limit);
    };
    Settings::values.frame_rate_limit.SetValue(60);
    Common::PlatformEvents::SetThermalMaxFps(0);
    REQUIRE(Service::Nx2VsyncRateHz() == 60);

    Common::PlatformEvents::SetThermalMaxFps(45);
    REQUIRE(Service::Nx2VsyncRateHz() == 45);
    REQUIRE(Settings::values.frame_rate_limit.GetValue() == 60);

    Settings::values.frame_rate_limit.SetValue(40);
    REQUIRE(Service::Nx2VsyncRateHz() == 40);

    Common::PlatformEvents::SetThermalMaxFps(30);
    REQUIRE(Service::Nx2VsyncRateHz() == 30);
    REQUIRE(Settings::values.frame_rate_limit.GetValue() == 40);
}
