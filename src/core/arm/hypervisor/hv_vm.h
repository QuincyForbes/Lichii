// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from upstream Djihads80/Eglantine#405 with Switch 2 DRAM sizing.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <Hypervisor/Hypervisor.h>

#include "common/common_types.h"
#include "common/host_memory.h"
#include "core/hardware_properties.h"

namespace Common {
struct PageTable;
}

namespace Core {

class System;

/// The process-wide Hypervisor.framework VM that application code runs in on Apple Silicon.
///
/// Emulated DRAM is mapped at IPA 0, so a guest physical address is its offset into DeviceMemory.
/// Guest virtual memory is described by 4K-granule stage-1 tables that this class keeps in sync
/// with the emulated kernel's mappings; the host's 16K page size only applies to stage 2.
class HvVm {
public:
    ~HvVm();

    /// Returns the active VM, or nullptr when the hypervisor backend is not in use.
    static HvVm* Get();

    /// Creates the VM (or resets the existing one for a new process). Returns nullptr on failure.
    static HvVm* Create(System& system);

    static std::unique_ptr<HvVm> CreateForTest(u8* dram_base, u64 dram_size);

    void Map(u64 vaddr, u64 phys_offset, u64 size, Common::MemoryPermission perms);
    void Unmap(u64 vaddr, u64 size);
    void Protect(u64 vaddr, u64 size, Common::MemoryPermission perms);
    /// Changes read/write access only, keeping each page's execute permission (GPU tracking).
    void ProtectAccess(u64 vaddr, u64 size, Common::MemoryPermission perms);
    /// Temporarily grants read or read/write access on a 4K page while a vCPU single-steps it.
    bool BeginStepAccess(u64 page, bool write);
    /// Releases a single-step grant and restores the underlying tracked permission once the last
    /// stepping vCPU on that page finishes.
    void EndStepAccess(u64 page, bool write);
    /// Stops routing mappings to the VM until the next Create.
    static void Deactivate();

    /// Installs a mapping for vaddr from the emulated page table if it exists there but not in
    /// the stage-1 tables. Returns true if a mapping that allows access was added.
    bool MapFromPageTable(const Common::PageTable& page_table, u64 vaddr,
                          bool read_trap_armed = false);

    bool RestoreGuestAccess(const Common::PageTable& page_table, u64 vaddr);

    /// Returns whether the underlying (non-step) stage-1 state for vaddr allows the access.
    bool Permits(u64 vaddr, bool write);

    /// Returns whether the live stage-1 descriptor (including active step grants) allows access.
    bool EffectivePermits(u64 vaddr, bool write);

    /// Returns whether the guest kernel's mapped permission for vaddr allows the access.
    bool GuestPermits(u64 vaddr, bool write);

    u32 ActiveStepCount(u64 page, bool write);

    /// Makes every vCPU flush its TLB (and optionally instruction cache) before it next runs
    /// guest code.
    void RequestFlush(bool icache = true);

    u64 FlushGeneration() const {
        return m_flush_generation.load(std::memory_order_acquire);
    }

    u64 IcacheGeneration() const {
        return m_icache_generation.load(std::memory_order_acquire);
    }

    void SetRunning(size_t core, hv_vcpu_t vcpu, bool running);

    static constexpr u64 VectorsVa = 0x1000;
    /// Offset of the "hvc" in each vector slot; the guest's x0 is in TPIDR_EL1 once it is reached.
    static constexpr u64 VectorHvcOffset = 8;
    static constexpr u64 FlushStubVa = VectorsVa + 0x800;
    static constexpr u64 FlushTlbStubVa = VectorsVa + 0x820;
    static constexpr u64 FlushPageStubVa = VectorsVa + 0x840;
    static constexpr u64 TableBaseIpa = 1ULL << 39;

    u64 RootTableIpa() const {
        return TableBaseIpa + RootTableOffset;
    }

private:
    struct StepGrant {
        u64 page{};
        u64 desired_access{};
        u32 read_refs{};
        u32 write_refs{};
    };

    HvVm() = default;

    bool Initialize(System& system);
    bool InitializeForTest(u8* dram_base, u64 dram_size);
    bool MapDram(System& system);
    void Reset();

    u64* EntryFor(u64 vaddr, bool allocate);
    u64* TableAt(u64 table_ipa);
    void SetPage(u64 vaddr, u64 entry);
    StepGrant* FindStepGrant(u64 page) noexcept;

    static constexpr u64 VaBits = 39;
    static constexpr u64 TableRegionSize = 1ULL << 30;
    static constexpr u64 TestTableRegionSize = 16ULL << 20;
    static constexpr u64 RootTableOffset = 0x1000;

    u8* m_dram_base{};
    u64 m_dram_size{};
    u8* m_tables{};
    u64 m_tables_size{TableRegionSize};
    u64 m_next_table_offset{};
    bool m_vm_created{};
    std::mutex m_table_mutex;
    std::vector<StepGrant> m_step_grants;

    std::atomic<u64> m_flush_generation{1};
    std::atomic<u64> m_icache_generation{1};
    std::mutex m_running_mutex;
    std::array<hv_vcpu_t, Hardware::NUM_CPU_CORES> m_vcpus{};
    std::array<bool, Hardware::NUM_CPU_CORES> m_running{};
};

} // namespace Core
