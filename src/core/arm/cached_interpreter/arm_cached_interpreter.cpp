// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/cached_interpreter/arm_cached_interpreter.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "common/logging.h"
#include "common/settings.h"
#include "core/arm/exclusive_monitor.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_process.h"
#include "core/memory.h"

namespace Core {

namespace {

std::mutex g_fallback_exclusive_mutex;
struct FallbackReservation {
    u64 addr{~u64{0}};
    u128 value{};
    bool active{false};
};
std::array<FallbackReservation, Core::Hardware::NUM_CPU_CORES> g_fallback_reservations{};

} // namespace

class ArmCachedInterpreter::ProcessMemoryBus final : public CachedInterpreter::MemoryBus {
public:
    ProcessMemoryBus(ArmCachedInterpreter& parent, System& system, Kernel::KProcess* process,
                     ExclusiveMonitor* monitor, std::size_t core_index)
        : m_parent{parent}, m_system{system}, m_memory{process->GetMemory()}, m_monitor{monitor},
          m_core_index{core_index}, m_debugger_enabled{system.DebuggerEnabled()} {}

    std::optional<u32> ReadCode32(u64 vaddr) override {
        if (!m_memory.IsValidVirtualAddressRange(vaddr, sizeof(u32))) {
            return std::nullopt;
        }
        const u64 page_base = vaddr & ~u64{0xFFF};
        if (page_base != m_code_page_addr) {
            if (m_memory.IsValidVirtualAddressRange(page_base, m_code_page.size()) &&
                m_memory.ReadBlock(page_base, m_code_page.data(), m_code_page.size())) {
                m_code_page_addr = page_base;
            } else {
                m_code_page_addr = ~u64{0};
                u32 inst = 0;
                if (!m_memory.ReadBlock(vaddr, &inst, sizeof(inst))) {
                    return std::nullopt;
                }
                return inst;
            }
        }
        u32 inst = 0;
        std::memcpy(&inst, m_code_page.data() + (vaddr & 0xFFFU), sizeof(inst));
        return inst;
    }

    bool CheckAccess(u64 vaddr, std::size_t size, Kernel::DebugWatchpointType type) {
        if (!m_memory.IsValidVirtualAddressRange(vaddr, size)) {
            LOG_CRITICAL(Core_ARM, "CachedInterpreter: unmapped memory access at {:#x} (size={})",
                         vaddr, size);
            return false;
        }
        if (m_debugger_enabled) {
            if (const auto* match = m_parent.MatchingWatchpoint(vaddr, size, type)) {
                m_parent.m_halted_watchpoint = match;
                return false;
            }
        }
        return true;
    }

    bool Read8(u64 vaddr, u8& out) override {
        if (!CheckAccess(vaddr, 1, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        out = m_memory.Read8(vaddr);
        return true;
    }

    bool Read16(u64 vaddr, u16& out) override {
        if (!CheckAccess(vaddr, 2, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        out = m_memory.Read16(vaddr);
        return true;
    }

    bool Read32(u64 vaddr, u32& out) override {
        if (!CheckAccess(vaddr, 4, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        out = m_memory.Read32(vaddr);
        return true;
    }

    bool Read64(u64 vaddr, u64& out) override {
        if (!CheckAccess(vaddr, 8, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        out = m_memory.Read64(vaddr);
        return true;
    }

    bool Read128(u64 vaddr, u128& out) override {
        if (!CheckAccess(vaddr, 16, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        out = {m_memory.Read64(vaddr), m_memory.Read64(vaddr + 8)};
        return true;
    }

    bool Write8(u64 vaddr, u8 value) override {
        if (!CheckAccess(vaddr, 1, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 1);
        m_memory.Write8(vaddr, value);
        return true;
    }

    bool Write16(u64 vaddr, u16 value) override {
        if (!CheckAccess(vaddr, 2, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 2);
        m_memory.Write16(vaddr, value);
        return true;
    }

    bool Write32(u64 vaddr, u32 value) override {
        if (!CheckAccess(vaddr, 4, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 4);
        m_memory.Write32(vaddr, value);
        return true;
    }

    bool Write64(u64 vaddr, u64 value) override {
        if (!CheckAccess(vaddr, 8, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 8);
        m_memory.Write64(vaddr, value);
        return true;
    }

    bool Write128(u64 vaddr, u128 value) override {
        if (!CheckAccess(vaddr, 16, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 16);
        m_memory.Write64(vaddr, value[0]);
        m_memory.Write64(vaddr + 8, value[1]);
        return true;
    }

    bool ExclusiveRead8(u64 vaddr, u8& out) override {
        if (!CheckAccess(vaddr, 1, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        if (m_monitor != nullptr) {
            out = m_monitor->ExclusiveRead8(m_core_index, vaddr);
            return true;
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        out = m_memory.Read8(vaddr);
        g_fallback_reservations[m_core_index] = {vaddr, {out, 0}, true};
        return true;
    }

    bool ExclusiveRead16(u64 vaddr, u16& out) override {
        if (!CheckAccess(vaddr, 2, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        if (m_monitor != nullptr) {
            out = m_monitor->ExclusiveRead16(m_core_index, vaddr);
            return true;
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        out = m_memory.Read16(vaddr);
        g_fallback_reservations[m_core_index] = {vaddr, {out, 0}, true};
        return true;
    }

    bool ExclusiveRead32(u64 vaddr, u32& out) override {
        if (!CheckAccess(vaddr, 4, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        if (m_monitor != nullptr) {
            out = m_monitor->ExclusiveRead32(m_core_index, vaddr);
            return true;
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        out = m_memory.Read32(vaddr);
        g_fallback_reservations[m_core_index] = {vaddr, {out, 0}, true};
        return true;
    }

    bool ExclusiveRead64(u64 vaddr, u64& out) override {
        if (!CheckAccess(vaddr, 8, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        if (m_monitor != nullptr) {
            out = m_monitor->ExclusiveRead64(m_core_index, vaddr);
            return true;
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        out = m_memory.Read64(vaddr);
        g_fallback_reservations[m_core_index] = {vaddr, {out, 0}, true};
        return true;
    }

    bool ExclusiveRead128(u64 vaddr, u128& out) override {
        if (!CheckAccess(vaddr, 16, Kernel::DebugWatchpointType::Read)) {
            return false;
        }
        if (m_monitor != nullptr) {
            out = m_monitor->ExclusiveRead128(m_core_index, vaddr);
            return true;
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        out = {m_memory.Read64(vaddr), m_memory.Read64(vaddr + 8)};
        g_fallback_reservations[m_core_index] = {vaddr, out, true};
        return true;
    }

    bool ExclusiveWrite8(u64 vaddr, u8 value) override {
        if (!CheckAccess(vaddr, 1, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 1);
        if (m_monitor != nullptr) {
            return m_monitor->ExclusiveWrite8(m_core_index, vaddr, value);
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        auto& res = g_fallback_reservations[m_core_index];
        if (!res.active || res.addr != vaddr || m_memory.Read8(vaddr) != static_cast<u8>(res.value[0])) {
            res.active = false;
            return false;
        }
        m_memory.Write8(vaddr, value);
        res.active = false;
        return true;
    }

    bool ExclusiveWrite16(u64 vaddr, u16 value) override {
        if (!CheckAccess(vaddr, 2, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 2);
        if (m_monitor != nullptr) {
            return m_monitor->ExclusiveWrite16(m_core_index, vaddr, value);
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        auto& res = g_fallback_reservations[m_core_index];
        if (!res.active || res.addr != vaddr || m_memory.Read16(vaddr) != static_cast<u16>(res.value[0])) {
            res.active = false;
            return false;
        }
        m_memory.Write16(vaddr, value);
        res.active = false;
        return true;
    }

    bool ExclusiveWrite32(u64 vaddr, u32 value) override {
        if (!CheckAccess(vaddr, 4, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 4);
        if (m_monitor != nullptr) {
            return m_monitor->ExclusiveWrite32(m_core_index, vaddr, value);
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        auto& res = g_fallback_reservations[m_core_index];
        if (!res.active || res.addr != vaddr || m_memory.Read32(vaddr) != static_cast<u32>(res.value[0])) {
            res.active = false;
            return false;
        }
        m_memory.Write32(vaddr, value);
        res.active = false;
        return true;
    }

    bool ExclusiveWrite64(u64 vaddr, u64 value) override {
        if (!CheckAccess(vaddr, 8, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 8);
        if (m_monitor != nullptr) {
            return m_monitor->ExclusiveWrite64(m_core_index, vaddr, value);
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        auto& res = g_fallback_reservations[m_core_index];
        if (!res.active || res.addr != vaddr || m_memory.Read64(vaddr) != res.value[0]) {
            res.active = false;
            return false;
        }
        m_memory.Write64(vaddr, value);
        res.active = false;
        return true;
    }

    bool ExclusiveWrite128(u64 vaddr, u128 value) override {
        if (!CheckAccess(vaddr, 16, Kernel::DebugWatchpointType::Write)) {
            return false;
        }
        InvalidateCodePageIfHit(vaddr, 16);
        if (m_monitor != nullptr) {
            return m_monitor->ExclusiveWrite128(m_core_index, vaddr, value);
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        auto& res = g_fallback_reservations[m_core_index];
        if (!res.active || res.addr != vaddr || m_memory.Read64(vaddr) != res.value[0] ||
            m_memory.Read64(vaddr + 8) != res.value[1]) {
            res.active = false;
            return false;
        }
        m_memory.Write64(vaddr, value[0]);
        m_memory.Write64(vaddr + 8, value[1]);
        res.active = false;
        return true;
    }

    void ClearExclusive() override {
        if (m_monitor != nullptr) {
            m_monitor->ClearExclusive(m_core_index);
            return;
        }
        std::scoped_lock lk{g_fallback_exclusive_mutex};
        g_fallback_reservations[m_core_index].active = false;
    }

    u64 GetCntvct() const override {
        return m_system.CoreTiming().GetClockTicks();
    }

    void DropCodePage() noexcept {
        m_code_page_addr = ~u64{0};
    }

    void OnCodeRangeInvalidated(u64 vaddr, std::size_t size) override {
        InvalidateCodePageIfHit(vaddr, size);
    }

private:
    void InvalidateCodePageIfHit(u64 vaddr, std::size_t size) noexcept {
        if (m_code_page_addr != ~u64{0} && vaddr < m_code_page_addr + m_code_page.size() &&
            m_code_page_addr < vaddr + size) {
            m_code_page_addr = ~u64{0};
        }
    }

    ArmCachedInterpreter& m_parent;
    System& m_system;
    Memory::Memory& m_memory;
    ExclusiveMonitor* m_monitor{};
    std::size_t m_core_index{};
    bool m_debugger_enabled{};
    u64 m_code_page_addr{~u64{0}};
    std::array<u8, 4096> m_code_page{};
};

ArmCachedInterpreter::ArmCachedInterpreter(System& system, bool uses_wall_clock,
                                           Kernel::KProcess* process,
                                           ExclusiveMonitor* exclusive_monitor,
                                           std::size_t core_index)
    : ArmInterface{uses_wall_clock}, m_system{system}, m_core_index{core_index},
      m_bus{std::make_unique<ProcessMemoryBus>(*this, system, process, exclusive_monitor,
                                               core_index)} {
    const bool enable_cache = [] {
        if (const char* env = std::getenv("EDEN_NX2_INTERP_CACHE")) {
            return env[0] != '0';
        }
        return true;
    }();
    const bool enable_fusion = [] {
        if (const char* env = std::getenv("EDEN_NX2_INTERP_FUSION")) {
            return env[0] != '0';
        }
        return true;
    }();
    m_engine = std::make_unique<CachedInterpreter::CachedBlockEngine>(*m_bus, enable_cache,
                                                                      enable_fusion);
}

ArmCachedInterpreter::~ArmCachedInterpreter() = default;

HaltReason ArmCachedInterpreter::TranslateStopReason(CachedInterpreter::StopReason stop) {
    using CachedInterpreter::StopReason;
    switch (stop) {
    case StopReason::StepCompleted:
        return HaltReason::StepThread;
    case StopReason::Interrupted:
        return HaltReason::BreakLoop;
    case StopReason::SupervisorCall:
        return HaltReason::SupervisorCall;
    case StopReason::Breakpoint:
        this->GetContext(m_breakpoint_context);
        return HaltReason::InstructionBreakpoint;
    case StopReason::DataAbort:
        return m_halted_watchpoint != nullptr ? HaltReason::DataAbort : HaltReason::PrefetchAbort;
    case StopReason::UnsupportedInstruction:
        LOG_CRITICAL(Core_ARM,
                     "CachedInterpreter: unsupported AArch64 instruction {:#010x} at pc={:#x}",
                     m_engine->LastUnsupportedInsn(), m_engine->LastUnsupportedPc());
        return HaltReason::PrefetchAbort;
    case StopReason::PrefetchAbort:
    case StopReason::None:
    default:
        return HaltReason::PrefetchAbort;
    }
}

HaltReason ArmCachedInterpreter::RunThread(Kernel::KThread* /*thread*/) {
    m_halted_watchpoint = nullptr;
    m_bus->ClearExclusive();

    if (m_uses_wall_clock) {
        return TranslateStopReason(m_engine->Run(0));
    }

    const u64 budget =
        static_cast<u64>(std::max<s64>(m_system.CoreTiming().downcount, 1));
    const u64 before = m_engine->GetStats().instructions_retired;
    const auto stop = m_engine->Run(budget);
    const u64 ran = std::max<u64>(1, m_engine->GetStats().instructions_retired - before);
    const u64 amortized = std::max<u64>(1, ran / Core::Hardware::NUM_CPU_CORES);
    m_system.CoreTiming().AddTicks(amortized);
    if (stop == CachedInterpreter::StopReason::StepCompleted) {
        return HaltReason::BreakLoop;
    }
    return TranslateStopReason(stop);
}

HaltReason ArmCachedInterpreter::StepThread(Kernel::KThread* /*thread*/) {
    m_halted_watchpoint = nullptr;
    m_bus->ClearExclusive();
    const auto stop = m_engine->Step();
    if (!m_uses_wall_clock) {
        m_system.CoreTiming().AddTicks(1);
    }
    return TranslateStopReason(stop);
}

void ArmCachedInterpreter::ClearInstructionCache() {
    m_bus->DropCodePage();
    m_engine->ClearCache();
}

void ArmCachedInterpreter::InvalidateCacheRange(u64 addr, std::size_t size) {
    m_bus->OnCodeRangeInvalidated(addr, size);
    m_engine->InvalidateRange(addr, size);
}

void ArmCachedInterpreter::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    const auto& s = m_engine->State();
    std::memcpy(ctx.r.data(), s.x.data(), 29 * sizeof(u64));
    ctx.fp = s.x[29];
    ctx.lr = s.x[30];
    ctx.sp = s.sp;
    ctx.pc = s.pc;
    ctx.pstate = s.pstate;
    ctx.padding = 0;
    ctx.v = s.v;
    ctx.fpcr = s.fpcr;
    ctx.fpsr = s.fpsr;
    ctx.tpidr = s.tpidr_el0;
}

void ArmCachedInterpreter::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    auto& s = m_engine->State();
    std::memcpy(s.x.data(), ctx.r.data(), 29 * sizeof(u64));
    s.x[29] = ctx.fp;
    s.x[30] = ctx.lr;
    s.sp = ctx.sp;
    s.pc = ctx.pc;
    s.pstate = ctx.pstate;
    s.v = ctx.v;
    s.fpcr = ctx.fpcr;
    s.fpsr = ctx.fpsr;
    s.tpidr_el0 = ctx.tpidr;
}

void ArmCachedInterpreter::SetTpidrroEl0(u64 value) {
    m_engine->State().tpidrro_el0 = value;
}

void ArmCachedInterpreter::GetSvcArguments(std::span<uint64_t, 8> args) const {
    const auto& s = m_engine->State();
    for (std::size_t i = 0; i < 8; ++i) {
        args[i] = s.x[i];
    }
}

void ArmCachedInterpreter::SetSvcArguments(std::span<const uint64_t, 8> args) {
    auto& s = m_engine->State();
    for (std::size_t i = 0; i < 8; ++i) {
        s.x[i] = args[i];
    }
}

u32 ArmCachedInterpreter::GetSvcNumber() const {
    return m_engine->LastSvcNumber();
}

void ArmCachedInterpreter::SignalInterrupt(Kernel::KThread* /*thread*/) {
    m_engine->RequestInterrupt();
}

const Kernel::DebugWatchpoint* ArmCachedInterpreter::HaltedWatchpoint() const {
    return m_halted_watchpoint;
}

void ArmCachedInterpreter::RewindBreakpointInstruction() {
    this->SetContext(m_breakpoint_context);
}

} // namespace Core
