// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>

#include "common/common_types.h"
#include "core/arm/arm_interface.h"
#include "core/arm/cached_interpreter/cached_block_engine.h"

namespace Core {

class ExclusiveMonitor;
class System;

namespace Memory {
class Memory;
}

class ArmCachedInterpreter final : public ArmInterface {
public:
    ArmCachedInterpreter(System& system, bool uses_wall_clock, Kernel::KProcess* process,
                         ExclusiveMonitor* exclusive_monitor, std::size_t core_index);
    ~ArmCachedInterpreter() override;

    HaltReason RunThread(Kernel::KThread* thread) override;
    HaltReason StepThread(Kernel::KThread* thread) override;

    void ClearInstructionCache() override;
    void InvalidateCacheRange(u64 addr, std::size_t size) override;

    Architecture GetArchitecture() const override {
        return Architecture::AArch64;
    }

    void GetContext(Kernel::Svc::ThreadContext& ctx) const override;
    void SetContext(const Kernel::Svc::ThreadContext& ctx) override;
    void SetTpidrroEl0(u64 value) override;

    void GetSvcArguments(std::span<uint64_t, 8> args) const override;
    void SetSvcArguments(std::span<const uint64_t, 8> args) override;
    u32 GetSvcNumber() const override;

    void SignalInterrupt(Kernel::KThread* thread) override;

    const Kernel::DebugWatchpoint* HaltedWatchpoint() const override;
    void RewindBreakpointInstruction() override;

    const CachedInterpreter::Stats& GetStats() const noexcept {
        return m_engine->GetStats();
    }

private:
    class ProcessMemoryBus;

    HaltReason TranslateStopReason(CachedInterpreter::StopReason stop);

    System& m_system;
    std::size_t m_core_index{};
    std::unique_ptr<ProcessMemoryBus> m_bus;
    std::unique_ptr<CachedInterpreter::CachedBlockEngine> m_engine;
    const Kernel::DebugWatchpoint* m_halted_watchpoint{};
    Kernel::Svc::ThreadContext m_breakpoint_context{};
};

} // namespace Core
