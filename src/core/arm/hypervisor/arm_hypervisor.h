// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from upstream Djihads80/Eglantine#405.

#pragma once

#include <array>
#include <atomic>

#include <Hypervisor/Hypervisor.h>
#include <pthread.h>

#include "core/arm/arm_interface.h"
#include "core/hle/kernel/svc_types.h"

namespace Core::Memory {
class Memory;
}

namespace Core {

class HvVm;
class System;

/// Runs AArch64 application code natively in a Hypervisor.framework vCPU (Apple Silicon).
/// Guest code executes at EL0; every exception it raises is routed to the host through EL1
/// vectors that issue "hvc", so no code patching is needed.
class ArmHypervisor final : public ArmInterface {
public:
    ArmHypervisor(System& system, HvVm& vm, std::size_t core_index);
    ~ArmHypervisor() override;

    Architecture GetArchitecture() const override {
        return Architecture::AArch64;
    }

    HaltReason RunThread(Kernel::KThread* thread) override;
    HaltReason StepThread(Kernel::KThread* thread) override;

    void GetContext(Kernel::Svc::ThreadContext& ctx) const override;
    void SetContext(const Kernel::Svc::ThreadContext& ctx) override;
    void SetTpidrroEl0(u64 value) override;

    void GetSvcArguments(std::span<uint64_t, 8> args) const override;
    void SetSvcArguments(std::span<const uint64_t, 8> args) override;
    u32 GetSvcNumber() const override;

    void SignalInterrupt(Kernel::KThread* thread) override;
    void ClearInstructionCache() override;
    void InvalidateCacheRange(u64 addr, std::size_t size) override;

    bool EmulateLoadStore(Memory::Memory& memory, u64 pc);

protected:
    const Kernel::DebugWatchpoint* HaltedWatchpoint() const override {
        return nullptr;
    }
    void RewindBreakpointInstruction() override {}

private:
    bool EnsureVcpu();
    void LoadVcpuContext();
    void ReadVcpuContext(Kernel::Svc::ThreadContext& ctx) const;
    bool VcpuHoldsContext() const;
    bool RunFlushStub(u64 stub_va, u64 arg_x0 = 0);
    void ArmHeartbeat();
    bool EmulateSystemRegister(u64 esr);
    HaltReason HandleGuestException(Kernel::KThread* thread);
    bool BeginTrackedStep(u64 pc, u64 page, bool write);
    void FinishTrackedAccess();
    void ClearExclusiveMonitor(Kernel::KProcess* process = nullptr);
    bool EmulateTrackedAccess(Kernel::KProcess* process, u64 pc);
    bool EmulateMemoryTagging(Kernel::KProcess* process, u64 pc);
    u64 GetBaseRegister(u32 index);
    void SetBaseRegister(u32 index, u64 value);
    void GetVectorRegister(u32 index, u8* out);
    void SetVectorRegister(u32 index, const u8* in);

    System& m_system;
    HvVm& m_vm;
    std::size_t m_core_index;

    hv_vcpu_t m_vcpu{};
    hv_vcpu_exit_t* m_exit{};
    bool m_vcpu_created{};
    pthread_t m_owner_thread{};
    u64 m_flushed_generation{};
    u64 m_flushed_icache_generation{};
    u64 m_heartbeat_ticks{};

    Kernel::Svc::ThreadContext m_ctx{};
    u64 m_tpidrro_el0{};
    bool m_vcpu_has_context{};
    bool m_tpidrro_dirty{};
    u32 m_svc{};

    std::atomic<u64> m_halt{};
    std::atomic<bool> m_running{};

    u64 m_last_fault_pc{};
    u64 m_last_fault_address{};
    u32 m_fault_repeat{};

    u64 m_step_pc{};
    std::array<u64, 4> m_step_pages{};
    std::array<bool, 4> m_step_writes{};
    u32 m_step_count{};
    bool m_stepping{};

    u64 m_excl_addr{};
    u128 m_excl_val{};
    u64 m_excl_rearm_page{};
    u32 m_excl_size{};
    bool m_excl_valid{};
};

} // namespace Core
