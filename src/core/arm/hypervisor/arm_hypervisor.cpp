// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from upstream Djihads80/Eglantine#405.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "common/cpu_features.h"
#include "common/logging.h"
#include "common/page_table.h"
#include "common/settings.h"
#include "core/arm/hypervisor/arm_hypervisor.h"
#include "core/arm/hypervisor/hv_vm.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/svc/svc_thread_yield.h"
#include "core/memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"

namespace Core {

namespace {

constexpr u32 ExceptionClassUnknown = 0x00;
constexpr u32 ExceptionClassWfx = 0x01;
constexpr u32 ExceptionClassSvc64 = 0x15;
constexpr u32 ExceptionClassHvc64 = 0x16;
constexpr u32 ExceptionClassSysReg = 0x18;
constexpr u32 ExceptionClassInstructionAbortLower = 0x20;
constexpr u32 ExceptionClassDataAbortLower = 0x24;
constexpr u32 ExceptionClassSoftwareStepLower = 0x32;
constexpr u32 ExceptionClassBrk64 = 0x3C;

constexpr u64 PstateSoftwareStep = 1ULL << 21;
constexpr u64 MdscrSoftwareStep = 1ULL << 0;

constexpr u64 PstateEl1h = 0x3C5; // EL1h with DAIF masked
constexpr u64 PstateNzcvMask = 0xF0000000;

// SCTLR_EL1: MMU, caches, and EL0 access to DC ZVA, CTR_EL0, WFI/WFE and cache maintenance.
constexpr u64 SctlrEl1 = 0x30D00800ULL | (1ULL << 0) | (1ULL << 2) | (1ULL << 12) | (1ULL << 14) |
                         (1ULL << 15) | (1ULL << 16) | (1ULL << 18) | (1ULL << 26);
// TCR_EL1: T0SZ=25 (39-bit), write-back inner shareable walks, 4K granule, no TTBR1, 40-bit IPA.
constexpr u64 TcrEl1 =
    25ULL | (1ULL << 8) | (1ULL << 10) | (3ULL << 12) | (1ULL << 23) | (2ULL << 32);
constexpr u64 MairEl1 = 0xFF;        // Attr0: normal memory, write-back
constexpr u64 CpacrEl1 = 3ULL << 20; // FP/SIMD at EL0 and EL1

constexpr u32 ExceptionClass(u64 esr) {
    return static_cast<u32>(esr >> 26) & 0x3F;
}

u64 HostCounter() {
    u64 value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
}

u64 HostCounterFrequency() {
    u64 value;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
}

u64 DcZvaBlockSize() {
    static const u64 block_size = []() -> u64 {
        uint64_t dczid = 4;
        if (hv_vcpu_config_t cfg = hv_vcpu_config_create()) {
            if (hv_vcpu_config_get_feature_reg(cfg, HV_FEATURE_REG_DCZID_EL0, &dczid) !=
                    HV_SUCCESS ||
                (dczid & (1ULL << 4)) != 0) {
                dczid = 4;
            }
            os_release(cfg);
        }
        return 4ULL << (dczid & 0xFU);
    }();
    return block_size;
}

u64 CacheMaintenanceBlockSize() {
    static const u64 line_size = []() -> u64 {
        uint64_t ctr = 0x8444C004ULL;
        if (hv_vcpu_config_t cfg = hv_vcpu_config_create()) {
            if (hv_vcpu_config_get_feature_reg(cfg, HV_FEATURE_REG_CTR_EL0, &ctr) != HV_SUCCESS) {
                ctr = 0x8444C004ULL;
            }
            os_release(cfg);
        }
        return 4ULL << ((ctr >> 16) & 0xFU);
    }();
    return line_size;
}

struct VcpuTimerState {
    hv_vcpu_t vcpu{};
    u64 offset{};
    u64 armed_cval{};
    bool offset_valid{};
    bool armed{};
};

struct CachedSvcArgs {
    const ArmHypervisor* owner{};
    std::array<u64, 8> regs{};
    bool valid{};
};
thread_local CachedSvcArgs t_svc_args{};

struct ThreadVcpus {
    std::vector<hv_vcpu_t> vcpus;
    std::vector<VcpuTimerState> timers;

    VcpuTimerState& TimerFor(hv_vcpu_t vcpu) {
        for (auto& timer : timers) {
            if (timer.vcpu == vcpu) {
                return timer;
            }
        }
        timers.push_back(VcpuTimerState{.vcpu = vcpu});
        return timers.back();
    }

    ~ThreadVcpus() {
        for (const hv_vcpu_t vcpu : vcpus) {
            hv_vcpu_destroy(vcpu);
        }
    }
};
thread_local ThreadVcpus t_thread_vcpus;

u64 GranuleMaskInPage(u64 offset, std::size_t size) noexcept {
    const u64 first_g = offset >> 6;
    const u64 last_g = (offset + size - 1) >> 6;
    const u64 count_g = last_g - first_g + 1;
    return count_g >= 64 ? ~u64{0} : (((u64{1} << count_g) - 1) << first_g);
}

u32 FastReadInsn(Memory::Memory& memory, std::size_t core_index, bool allow_hot, u64 pc) {
    if (allow_hot && (pc & 3ULL) == 0) {
        const u64 masked = pc & 0xffffffffffffULL;
        const u64 page = masked & ~u64{Memory::YUZU_PAGEMASK};
        const u64 offset = masked & u64{Memory::YUZU_PAGEMASK};
        if (auto* const hot_core = memory.HotCore(core_index)) {
            auto& hot = hot_core->ReadSlot(page);
            if (page == hot.vaddr_page && memory.FastReadGenerationRelaxed() == hot.mem_gen &&
                (!hot.uses_map_gen ||
                 (hot_core->dev_mem != nullptr &&
                  hot_core->dev_mem->MappingGeneration() == hot.map_gen))) {
                u32 insn{};
                std::memcpy(&insn, hot.host_page + offset, sizeof(insn));
                return insn;
            }
            const u32 insn = memory.Read32(pc);
            if (insn != 0) {
                static_cast<void>(memory.TryCacheHotReadPage(core_index, pc, hot));
            }
            return insn;
        }
    }
    return memory.Read32(pc);
}

bool FastReadBytes(Memory::Memory& memory, std::size_t core_index, bool allow_hot, u64 vaddr,
                   void* dst, std::size_t size) {
    if (size == 0) {
        return true;
    }
    const u64 masked = vaddr & 0xffffffffffffULL;
    const u64 page = masked & ~u64{Memory::YUZU_PAGEMASK};
    const u64 offset = masked & u64{Memory::YUZU_PAGEMASK};
    const bool in_page = offset + size <= Memory::YUZU_PAGESIZE;
    if (allow_hot && in_page) {
        if (auto* const hot_core = memory.HotCore(core_index)) {
            const auto& hot = hot_core->ReadSlot(page);
            if (page == hot.vaddr_page && memory.FastReadGenerationRelaxed() == hot.mem_gen &&
                (!hot.uses_map_gen ||
                 (hot_core->dev_mem != nullptr &&
                  hot_core->dev_mem->MappingGeneration() == hot.map_gen))) {
                std::memcpy(dst, hot.host_page + offset, size);
                return true;
            }
        }
    }
    if (!memory.IsValidVirtualAddressRange(vaddr, size)) {
        return false;
    }
    memory.ReadBlock(vaddr, dst, size);
    if (allow_hot && in_page) {
        if (auto* const hot_core = memory.HotCore(core_index)) {
            static_cast<void>(
                memory.TryCacheHotReadPage(core_index, vaddr, hot_core->ReadSlot(page)));
        }
    }
    return true;
}

bool FastWriteBytes(Memory::Memory& memory, std::size_t core_index, bool allow_hot, u64 vaddr,
                    const void* src, std::size_t size) {
    if (size == 0) {
        return true;
    }
    const u64 masked = vaddr & 0xffffffffffffULL;
    const u64 page = masked & ~u64{Memory::YUZU_PAGEMASK};
    const u64 offset = masked & u64{Memory::YUZU_PAGEMASK};
    const bool in_page = offset + size <= Memory::YUZU_PAGESIZE;
    const u64 granule_mask = in_page ? GranuleMaskInPage(offset, size) : 0;
    if (allow_hot && in_page) {
        if (auto* const hot_core = memory.HotCore(core_index)) {
            auto& hot = hot_core->WriteSlot(page);
            if (page == hot.vaddr_page) {
                if (!hot.rasterizer_cached) {
                    if (memory.FastWriteGenerationRelaxed() == hot.mem_gen) {
                        std::memcpy(hot.host_page + offset, src, size);
                        return true;
                    }
                } else {
                    if ((hot.collected_granules & granule_mask) == granule_mask &&
                        memory.RasterizerWriteGenerationRelaxed() == hot.mem_gen &&
                        Core::g_write_answer_epoch.load(std::memory_order_relaxed) ==
                            hot.answer_epoch &&
                        hot_core->dev_mem != nullptr &&
                        hot_core->dev_mem->MappingGeneration() == hot.map_gen) {
                        const u64 gather_ep = hot_core->GatherEpoch();
                        if (gather_ep == hot.gather_epoch) {
                            Core::g_collected_write_pages.Mark(hot.daddr_page, 1);
                            std::memcpy(hot.host_page + offset, src, size);
                            std::atomic_signal_fence(std::memory_order_seq_cst);
                            if (hot_core->GatherEpoch() == gather_ep) {
                                return true;
                            }
                        }
                    }
                    if (memory.TryCollectHotWrite(core_index, offset, size, granule_mask, hot)) {
                        std::memcpy(hot.host_page + offset, src, size);
                        return true;
                    }
                }
            }
        }
    }
    if (!memory.IsValidVirtualAddressRange(vaddr, size)) {
        return false;
    }
    memory.WriteBlock(vaddr, src, size);
    if (allow_hot && in_page) {
        if (auto* const hot_core = memory.HotCore(core_index)) {
            static_cast<void>(memory.TryCacheHotWritePage(core_index, vaddr, granule_mask,
                                                          hot_core->WriteSlot(page)));
        }
    }
    return true;
}

bool FastZeroBytes(Memory::Memory& memory, std::size_t core_index, bool allow_hot, u64 vaddr,
                   std::size_t size) {
    if (size == 0) {
        return true;
    }
    const u64 masked = vaddr & 0xffffffffffffULL;
    const u64 page = masked & ~u64{Memory::YUZU_PAGEMASK};
    const u64 offset = masked & u64{Memory::YUZU_PAGEMASK};
    const bool in_page = offset + size <= Memory::YUZU_PAGESIZE;
    const u64 granule_mask = in_page ? GranuleMaskInPage(offset, size) : 0;
    if (allow_hot && in_page) {
        if (auto* const hot_core = memory.HotCore(core_index)) {
            auto& hot = hot_core->WriteSlot(page);
            if (page == hot.vaddr_page) {
                if (!hot.rasterizer_cached) {
                    if (memory.FastWriteGenerationRelaxed() == hot.mem_gen) {
                        std::memset(hot.host_page + offset, 0, size);
                        return true;
                    }
                } else {
                    if ((hot.collected_granules & granule_mask) == granule_mask &&
                        memory.RasterizerWriteGenerationRelaxed() == hot.mem_gen &&
                        Core::g_write_answer_epoch.load(std::memory_order_relaxed) ==
                            hot.answer_epoch &&
                        hot_core->dev_mem != nullptr &&
                        hot_core->dev_mem->MappingGeneration() == hot.map_gen) {
                        const u64 gather_ep = hot_core->GatherEpoch();
                        if (gather_ep == hot.gather_epoch) {
                            Core::g_collected_write_pages.Mark(hot.daddr_page, 1);
                            std::memset(hot.host_page + offset, 0, size);
                            std::atomic_signal_fence(std::memory_order_seq_cst);
                            if (hot_core->GatherEpoch() == gather_ep) {
                                return true;
                            }
                        }
                    }
                    if (memory.TryCollectHotWrite(core_index, offset, size, granule_mask, hot)) {
                        std::memset(hot.host_page + offset, 0, size);
                        return true;
                    }
                }
            }
        }
    }
    if (!memory.IsValidVirtualAddressRange(vaddr, size)) {
        return false;
    }
    memory.ZeroBlock(vaddr, size);
    if (allow_hot && in_page) {
        if (auto* const hot_core = memory.HotCore(core_index)) {
            static_cast<void>(memory.TryCacheHotWritePage(core_index, vaddr, granule_mask,
                                                          hot_core->WriteSlot(page)));
        }
    }
    return true;
}

#define HV_CHECK(call)                                                                             \
    do {                                                                                           \
        if (const hv_return_t r_ = (call); r_ != HV_SUCCESS) {                                     \
            LOG_CRITICAL(Core_ARM, "Hypervisor: {} failed ({:#x})", #call, static_cast<u32>(r_));  \
        }                                                                                          \
    } while (0)

} // Anonymous namespace

ArmHypervisor::ArmHypervisor(System& system, HvVm& vm, std::size_t core_index)
    : ArmInterface{true}, m_system{system}, m_vm{vm}, m_core_index{core_index} {
    m_heartbeat_ticks = HostCounterFrequency() / 1000;
}

ArmHypervisor::~ArmHypervisor() {
    m_vm.SetRunning(m_core_index, m_vcpu, false);
}

bool ArmHypervisor::EnsureVcpu() {
    if (m_vcpu_created) {
        if (!pthread_equal(m_owner_thread, pthread_self())) {
            LOG_CRITICAL(Core_ARM, "Hypervisor: core {} vCPU used from a foreign thread",
                         m_core_index);
            return false;
        }
        return true;
    }

    if (hv_vcpu_create(&m_vcpu, &m_exit, nullptr) != HV_SUCCESS) {
        LOG_CRITICAL(Core_ARM, "Hypervisor: failed to create vCPU for core {}", m_core_index);
        return false;
    }
    m_vcpu_created = true;
    m_owner_thread = pthread_self();
    t_thread_vcpus.vcpus.push_back(m_vcpu);

    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_MAIR_EL1, MairEl1));
    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_TCR_EL1, TcrEl1));
    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_TTBR0_EL1, m_vm.RootTableIpa()));
    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_VBAR_EL1, HvVm::VectorsVa));
    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_CPACR_EL1, CpacrEl1));
    static const bool native_timer = [] {
        const char* env = std::getenv("EDEN_NX2_HV_NATIVE_TIMER");
        return env != nullptr && env[0] == '1';
    }();
    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_CNTKCTL_EL1, native_timer ? 3ULL : 0ULL));
    HV_CHECK(hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_SCTLR_EL1, SctlrEl1));
    HV_CHECK(hv_vcpu_set_trap_debug_exceptions(m_vcpu, true));

    m_flushed_generation = 0;
    m_flushed_icache_generation = 0;
    m_vm.SetRunning(m_core_index, m_vcpu, true);
    LOG_INFO(Core_ARM, "Hypervisor: created vCPU for core {} (native_timer={})", m_core_index,
             native_timer);
    return true;
}

void ArmHypervisor::LoadVcpuContext() {
    for (int i = 0; i < 29; i++) {
        hv_vcpu_set_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + i), m_ctx.r[i]);
    }
    hv_vcpu_set_reg(m_vcpu, HV_REG_FP, m_ctx.fp);
    hv_vcpu_set_reg(m_vcpu, HV_REG_LR, m_ctx.lr);
    hv_vcpu_set_reg(m_vcpu, HV_REG_PC, m_ctx.pc);
    hv_vcpu_set_reg(m_vcpu, HV_REG_CPSR, m_ctx.pstate & PstateNzcvMask);
    hv_vcpu_set_reg(m_vcpu, HV_REG_FPCR, m_ctx.fpcr);
    hv_vcpu_set_reg(m_vcpu, HV_REG_FPSR, m_ctx.fpsr);
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_SP_EL0, m_ctx.sp);
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_TPIDR_EL0, m_ctx.tpidr);
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_TPIDRRO_EL0, m_tpidrro_el0);
    for (int i = 0; i < 32; i++) {
        hv_simd_fp_uchar16_t value;
        std::memcpy(&value, &m_ctx.v[i], sizeof(value));
        hv_vcpu_set_simd_fp_reg(m_vcpu, static_cast<hv_simd_fp_reg_t>(HV_SIMD_FP_REG_Q0 + i),
                                value);
    }
    m_excl_valid = false;
    t_svc_args.valid = false;
}

void ArmHypervisor::ReadVcpuContext(Kernel::Svc::ThreadContext& ctx) const {
    for (int i = 0; i < 29; i++) {
        hv_vcpu_get_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + i), &ctx.r[i]);
    }
    u64 value{};
    hv_vcpu_get_reg(m_vcpu, HV_REG_FP, &ctx.fp);
    hv_vcpu_get_reg(m_vcpu, HV_REG_LR, &ctx.lr);
    hv_vcpu_get_reg(m_vcpu, HV_REG_PC, &ctx.pc);
    hv_vcpu_get_reg(m_vcpu, HV_REG_CPSR, &value);
    ctx.pstate = static_cast<u32>(value & PstateNzcvMask);
    hv_vcpu_get_reg(m_vcpu, HV_REG_FPCR, &value);
    ctx.fpcr = static_cast<u32>(value);
    hv_vcpu_get_reg(m_vcpu, HV_REG_FPSR, &value);
    ctx.fpsr = static_cast<u32>(value);
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_SP_EL0, &ctx.sp);
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_TPIDR_EL0, &ctx.tpidr);
    for (int i = 0; i < 32; i++) {
        hv_simd_fp_uchar16_t q;
        hv_vcpu_get_simd_fp_reg(m_vcpu, static_cast<hv_simd_fp_reg_t>(HV_SIMD_FP_REG_Q0 + i), &q);
        std::memcpy(&ctx.v[i], &q, sizeof(q));
    }
}

bool ArmHypervisor::VcpuHoldsContext() const {
    if (!m_vcpu_has_context) {
        return false;
    }
    if (!pthread_equal(m_owner_thread, pthread_self())) {
        LOG_ERROR(Core_ARM, "Hypervisor: core {} context accessed from a foreign thread",
                  m_core_index);
        return false;
    }
    return true;
}

void ArmHypervisor::ArmHeartbeat() {
    auto& timer = t_thread_vcpus.TimerFor(m_vcpu);
    if (!timer.offset_valid) {
        hv_vcpu_get_vtimer_offset(m_vcpu, &timer.offset);
        timer.offset_valid = true;
    }
    const u64 now = HostCounter() - timer.offset;
    if (timer.armed && now + (m_heartbeat_ticks >> 2) < timer.armed_cval) {
        return;
    }
    const u64 cval = now + m_heartbeat_ticks;
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_CNTV_CVAL_EL0, cval);
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_CNTV_CTL_EL0, 1);
    hv_vcpu_set_vtimer_mask(m_vcpu, false);
    timer.armed_cval = cval;
    timer.armed = true;
}

bool ArmHypervisor::RunFlushStub(u64 stub_va, u64 arg_x0) {
    u64 pc{}, cpsr{}, x0{}, mdscr{};
    hv_vcpu_get_reg(m_vcpu, HV_REG_PC, &pc);
    hv_vcpu_get_reg(m_vcpu, HV_REG_CPSR, &cpsr);
    hv_vcpu_get_reg(m_vcpu, HV_REG_X0, &x0);
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_MDSCR_EL1, &mdscr);
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_MDSCR_EL1, 0);
    hv_vcpu_set_reg(m_vcpu, HV_REG_X0, arg_x0);
    hv_vcpu_set_reg(m_vcpu, HV_REG_PC, stub_va);
    hv_vcpu_set_reg(m_vcpu, HV_REG_CPSR, PstateEl1h);

    bool done = false;
    bool vtimer_fired = false;
    u32 canceled_streak = 0;
    while (!done) {
        const hv_return_t run_ret = hv_vcpu_run(m_vcpu);
        if (run_ret != HV_SUCCESS) {
            LOG_CRITICAL(Core_ARM, "Hypervisor: hv_vcpu_run failed in flush stub ({:#x})",
                         static_cast<u32>(run_ret));
            break;
        }
        if (m_exit->reason == HV_EXIT_REASON_CANCELED) {
            if (++canceled_streak >= 16) {
                std::this_thread::yield();
            }
            continue;
        }
        canceled_streak = 0;
        if (m_exit->reason == HV_EXIT_REASON_VTIMER_ACTIVATED) {
            vtimer_fired = true;
            continue;
        }
        if (m_exit->reason == HV_EXIT_REASON_EXCEPTION) {
            const u64 syndrome = m_exit->exception.syndrome;
            done = ExceptionClass(syndrome) == ExceptionClassHvc64 && (syndrome & 0xFFFF) == 1;
            if (!done) {
                LOG_CRITICAL(Core_ARM, "Hypervisor: unexpected exit {:#x} in flush stub", syndrome);
                break;
            }
        } else {
            LOG_CRITICAL(Core_ARM, "Hypervisor: unexpected exit reason {} in flush stub",
                         static_cast<u32>(m_exit->reason));
            break;
        }
    }

    hv_vcpu_set_reg(m_vcpu, HV_REG_PC, pc);
    hv_vcpu_set_reg(m_vcpu, HV_REG_CPSR, cpsr);
    hv_vcpu_set_reg(m_vcpu, HV_REG_X0, x0);
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_MDSCR_EL1, mdscr);
    if (vtimer_fired) {
        t_thread_vcpus.TimerFor(m_vcpu).armed = false;
        ArmHeartbeat();
    }
    return done;
}

bool ArmHypervisor::EmulateSystemRegister(u64 esr) {
    const bool is_read = (esr & 1) != 0;
    const u32 rt = (esr >> 5) & 0x1F;
    const u32 crm = (esr >> 1) & 0xF;
    const u32 crn = (esr >> 10) & 0xF;
    const u32 op1 = (esr >> 14) & 0x7;
    const u32 op2 = (esr >> 17) & 0x7;
    const u32 op0 = (esr >> 20) & 0x3;

    if (!is_read || op0 != 3 || op1 != 3 || crn != 14 || crm != 0) {
        LOG_WARNING(Core_ARM, "Hypervisor: unhandled system register access {:#x}", esr);
        return false;
    }

    u64 value{};
    switch (op2) {
    case 0: // CNTFRQ_EL0
        value = Common::g_wall_clock.GetGuestCNTFRQ();
        break;
    case 1: // CNTPCT_EL0
    case 2: // CNTVCT_EL0
    case 5: // CNTPCTSS_EL0
    case 6: // CNTVCTSS_EL0
        value = m_system.CoreTiming().GetClockTicks();
        break;
    default:
        LOG_WARNING(Core_ARM, "Hypervisor: unhandled counter register access {:#x}", esr);
        return false;
    }
    if (rt != 31) {
        hv_vcpu_set_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + rt), value);
    }
    return true;
}

static Common::PageType GetPageType(Kernel::KProcess* process, u64 vaddr) {
    const auto& page_table = process->GetPageTable().GetImpl();
    if ((vaddr >> page_table.GetAddressSpaceBits()) != 0) {
        return Common::PageType::Unmapped;
    }
    return page_table.entries[vaddr >> Memory::YUZU_PAGEBITS].Type();
}

static bool IsMapped(Kernel::KProcess* process, u64 vaddr) {
    return GetPageType(process, vaddr) != Common::PageType::Unmapped;
}

static bool IsNormalMemory(Kernel::KProcess* process, u64 vaddr) {
    return GetPageType(process, vaddr) == Common::PageType::Memory;
}

static bool IsDebugMemory(Kernel::KProcess* process, u64 vaddr) {
    return GetPageType(process, vaddr) == Common::PageType::DebugMemory;
}

static bool IsRasterizerCached(Kernel::KProcess* process, u64 vaddr) {
    return GetPageType(process, vaddr) == Common::PageType::RasterizerCachedMemory;
}

static bool IsReadTrapArmed(const Memory::Memory& memory, u64 vaddr) {
    const auto* const state_ptr = memory.JitReadTrapStatePtr();
    if (state_ptr == nullptr) {
        return false;
    }
    const std::uintptr_t state = state_ptr->load(std::memory_order_acquire);
    if (state <= 1) {
        return false;
    }
    const u64 page = vaddr >> Memory::YUZU_PAGEBITS;
    if (page >= (1ULL << (39 - Memory::YUZU_PAGEBITS))) {
        return false;
    }
    const auto* const words = reinterpret_cast<const std::atomic<u64>*>(state);
    return (words[page >> 6].load(std::memory_order_acquire) & (1ULL << (page & 63))) != 0;
}

bool ArmHypervisor::BeginTrackedStep(u64 pc, u64 page, bool write) {
    page &= ~u64{0xFFF};
    if (m_stepping && pc != m_step_pc) {
        FinishTrackedAccess();
    }
    for (u32 i = 0; i < m_step_count; ++i) {
        if (m_step_pages[i] == page) {
            if (write && !m_step_writes[i]) {
                if (!m_vm.BeginStepAccess(page, true)) {
                    return false;
                }
                m_vm.EndStepAccess(page, false);
                m_step_writes[i] = true;
            }
            return true;
        }
    }
    if (m_step_count >= m_step_pages.size()) {
        return false;
    }
    if (!m_vm.BeginStepAccess(page, write)) {
        return false;
    }
    m_step_pc = pc;
    m_step_pages[m_step_count] = page;
    m_step_writes[m_step_count] = write;
    ++m_step_count;
    m_stepping = true;
    hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_MDSCR_EL1, MdscrSoftwareStep);
    return true;
}

void ArmHypervisor::FinishTrackedAccess() {
    if (!m_stepping) {
        return;
    }
    m_stepping = false;
    if (m_vcpu_created) {
        hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_MDSCR_EL1, 0);
        u64 cpsr{};
        hv_vcpu_get_reg(m_vcpu, HV_REG_CPSR, &cpsr);
        if ((cpsr & PstateSoftwareStep) != 0) {
            hv_vcpu_set_reg(m_vcpu, HV_REG_CPSR, cpsr & ~PstateSoftwareStep);
        }
    }
    for (u32 i = 0; i < m_step_count; ++i) {
        m_vm.EndStepAccess(m_step_pages[i], m_step_writes[i]);
    }
    m_step_count = 0;
}

void ArmHypervisor::ClearExclusiveMonitor(Kernel::KProcess* process) {
    if (!m_excl_valid && m_excl_rearm_page == 0) {
        return;
    }
    m_excl_valid = false;
    if (m_excl_rearm_page != 0) {
        const u64 page = std::exchange(m_excl_rearm_page, 0);
        if (process != nullptr && IsRasterizerCached(process, page) &&
            !Settings::values.use_reactive_flushing.GetValue() &&
            !IsReadTrapArmed(process->GetMemory(), page)) {
            m_vm.ProtectAccess(page, Memory::YUZU_PAGESIZE, Common::MemoryPermission::Read);
        }
    }
}

bool ArmHypervisor::EmulateMemoryTagging(Kernel::KProcess* process, u64 pc) {
    auto& memory = process->GetMemory();
    const bool allow_hot = !m_system.DebuggerEnabled();
    const u32 insn = FastReadInsn(memory, m_core_index, allow_hot, pc);
    const u32 rt = insn & 0x1F;
    const u32 rn = (insn >> 5) & 0x1F;
    const u64 base = GetBaseRegister(rn);
    const auto get_data = [&](u32 index) { return index == 31 ? 0 : GetBaseRegister(index); };
    const auto finish = [&](u32 mode, u64 offset) {
        if (mode != 2) {
            SetBaseRegister(rn, base + offset);
        }
        return mode == 1 ? base : base + offset;
    };

    if ((insn & 0xFF200000) == 0xD9200000) {
        const u32 opc = (insn >> 22) & 3;
        const u32 mode = (insn >> 10) & 3;
        const u64 offset = static_cast<u64>(static_cast<s64>(static_cast<u64>(insn) << 43) >> 55)
                           << 4;
        if (mode == 0) {
            if (opc != 1) {
                return false;
            }
            if (rt != 31) {
                SetBaseRegister(rt, get_data(rt) & ~(u64{0xF} << 56));
            }
            return true;
        }
        const u64 address = finish(mode, offset);
        if ((opc & 1) != 0) {
            const std::size_t zero_bytes = opc == 3 ? 32 : 16;
            if (!FastZeroBytes(memory, m_core_index, allow_hot, address, zero_bytes)) {
                memory.ZeroBlock(address, zero_bytes);
            }
        }
        return true;
    }
    if ((insn & 0xFE400000U) == 0x68000000U && ((insn >> 23) & 3) != 0) {
        const u32 rt2 = (insn >> 10) & 0x1F;
        const u64 offset = static_cast<u64>(static_cast<s64>(static_cast<u64>(insn) << 42) >> 57)
                           << 4;
        const u64 address = finish((insn >> 23) & 3, offset);
        const std::array<u64, 2> pair{get_data(rt), get_data(rt2)};
        if (!FastWriteBytes(memory, m_core_index, allow_hot, address, pair.data(), sizeof(pair))) {
            memory.Write64(address, pair[0]);
            memory.Write64(address + 8, pair[1]);
        }
        return true;
    }
    // IRG Xd, Xn, Xm (allocation tags are always 0; Rd/Rn == 31 is SP).
    if ((insn & 0xFFE0FC00U) == 0x9AC01000U) {
        SetBaseRegister(rt, base & ~(u64{0xF} << 56));
        return true;
    }
    // GMI Xd, Xn, Xm (tag of Xn is 0 -> sets bit 0 in Xm; Rd == 31 is XZR).
    if ((insn & 0xFFE0FC00U) == 0x9AC01400U) {
        if (rt != 31) {
            const u32 rm = (insn >> 16) & 0x1F;
            SetBaseRegister(rt, get_data(rm) | 1ULL);
        }
        return true;
    }
    // ADDG / SUBG Xd, Xn, #uimm6<<4, #uimm4 (Rd/Rn == 31 is SP).
    if ((insn & 0xFFC0C000U) == 0x91800000U || (insn & 0xFFC0C000U) == 0xD1800000U) {
        const u64 uimm = static_cast<u64>((insn >> 16) & 0x3F) << 4;
        const u64 result = ((insn & 0x40000000U) != 0) ? (base - uimm) : (base + uimm);
        SetBaseRegister(rt, result & ~(u64{0xF} << 56));
        return true;
    }
    return false;
}

u64 ArmHypervisor::GetBaseRegister(u32 index) {
    if (!m_vcpu_has_context) {
        if (index < 29) {
            return m_ctx.r[index];
        }
        return index == 29 ? m_ctx.fp : (index == 30 ? m_ctx.lr : m_ctx.sp);
    }
    u64 value{};
    if (index == 31) {
        hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_SP_EL0, &value);
    } else {
        hv_vcpu_get_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + index), &value);
    }
    return value;
}

void ArmHypervisor::SetBaseRegister(u32 index, u64 value) {
    if (!m_vcpu_has_context) {
        if (index < 29) {
            m_ctx.r[index] = value;
        } else if (index == 29) {
            m_ctx.fp = value;
        } else if (index == 30) {
            m_ctx.lr = value;
        } else {
            m_ctx.sp = value;
        }
        return;
    }
    if (index == 31) {
        hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_SP_EL0, value);
    } else {
        if (index < 8 && t_svc_args.owner == this) {
            t_svc_args.valid = false;
        }
        hv_vcpu_set_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + index), value);
    }
}

void ArmHypervisor::GetVectorRegister(u32 index, u8* out) {
    if (!m_vcpu_has_context) {
        std::memcpy(out, &m_ctx.v[index & 31U], 16);
        return;
    }
    hv_simd_fp_uchar16_t value;
    hv_vcpu_get_simd_fp_reg(m_vcpu, static_cast<hv_simd_fp_reg_t>(HV_SIMD_FP_REG_Q0 + (index & 31U)),
                            &value);
    std::memcpy(out, &value, sizeof(value));
}

void ArmHypervisor::SetVectorRegister(u32 index, const u8* in) {
    if (!m_vcpu_has_context) {
        std::memcpy(&m_ctx.v[index & 31U], in, 16);
        return;
    }
    hv_simd_fp_uchar16_t value;
    std::memcpy(&value, in, sizeof(value));
    hv_vcpu_set_simd_fp_reg(m_vcpu, static_cast<hv_simd_fp_reg_t>(HV_SIMD_FP_REG_Q0 + (index & 31U)),
                            value);
}

bool ArmHypervisor::EmulateTrackedAccess(Kernel::KProcess* process, u64 pc) {
    return EmulateLoadStore(process->GetMemory(), pc);
}

bool ArmHypervisor::EmulateLoadStore(Memory::Memory& memory, u64 pc) {
    const bool allow_hot = !m_system.DebuggerEnabled();
    const u32 insn = FastReadInsn(memory, m_core_index, allow_hot, pc);
    const u32 rt = insn & 0x1FU;
    const u32 rn = (insn >> 5) & 0x1FU;

    const auto get_x_or_zero = [&](u32 index) -> u64 {
        return index == 31U ? 0 : GetBaseRegister(index);
    };
    const auto sign_extend_imm = [](u32 value, u32 bits) -> u64 {
        return static_cast<u64>(static_cast<s64>(static_cast<u64>(value) << (64 - bits)) >>
                                (64 - bits));
    };

    if ((insn & 0xFFFFFFE0U) == 0xD50B7420U) { // DC ZVA, Xt
        const u64 block_size = DcZvaBlockSize();
        const u64 aligned = get_x_or_zero(rt) & ~(block_size - 1);
        return FastZeroBytes(memory, m_core_index, allow_hot, aligned,
                             static_cast<std::size_t>(block_size));
    }
    if ((insn & 0xFFFFFFE0U) == 0xD50B7E20U) { // DC CIVAC, Xt
        const u64 line_size = CacheMaintenanceBlockSize();
        memory.FlushDataCache(get_x_or_zero(rt) & ~(line_size - 1),
                              static_cast<std::size_t>(line_size));
        return true;
    }
    if ((insn & 0xFFFFFFE0U) == 0xD50B7A20U || // DC CVAC, Xt
        (insn & 0xFFFFFFE0U) == 0xD50B7C20U || // DC CVAP, Xt
        (insn & 0xFFFFFFE0U) == 0xD50B7D20U) { // DC CVADP, Xt
        const u64 line_size = CacheMaintenanceBlockSize();
        memory.StoreDataCache(get_x_or_zero(rt) & ~(line_size - 1),
                              static_cast<std::size_t>(line_size));
        return true;
    }
    if ((insn & 0xFFFFFFE0U) == 0xD50B7B20U) { // DC CVAU, Xt
        return true;
    }
    if ((insn & 0xFFFFFFE0U) == 0xD50B7520U) { // IC IVAU, Xt
        m_vm.RequestFlush(true);
        return true;
    }
    // Prefetch hints (PRFM immediate, PRFUM, PRFM register, PRFM literal).
    if ((insn & 0xFFC00000U) == 0xF9800000U || (insn & 0xFFE00C00U) == 0xF8800000U ||
        ((insn & 0xFFE00C00U) == 0xF8A00800U && (((insn >> 13) & 2U) != 0)) ||
        (insn & 0xFF000000U) == 0xD8000000U) {
        return true;
    }

    // AdvSIMD LD1..LD4 / ST1..ST4 multiple structures (1..4 registers, optional post-index).
    if ((insn & 0xBF200000U) == 0x0C000000U) {
        const bool post_index = ((insn >> 23) & 1U) != 0;
        const u32 rm = (insn >> 16) & 0x1FU;
        if (!post_index && rm != 0) {
            return false;
        }
        u32 rpt = 0;
        u32 selem = 0;
        switch ((insn >> 12) & 0xFU) {
        case 0: rpt = 1; selem = 4; break;
        case 2: rpt = 4; selem = 1; break;
        case 4: rpt = 1; selem = 3; break;
        case 6: rpt = 3; selem = 1; break;
        case 7: rpt = 1; selem = 1; break;
        case 8: rpt = 1; selem = 2; break;
        case 10: rpt = 2; selem = 1; break;
        default: return false;
        }
        const bool q = ((insn >> 30) & 1U) != 0;
        const u32 elem_size = (insn >> 10) & 3U;
        if (elem_size == 3U && !q && selem != 1U) {
            return false;
        }
        const std::size_t reg_bytes = q ? 16 : 8;
        const std::size_t esize = std::size_t{1} << elem_size;
        const std::size_t elements = reg_bytes / esize;
        const u32 num_regs = rpt * selem;
        const std::size_t total_bytes = reg_bytes * num_regs;
        const u64 base = GetBaseRegister(rn);
        const bool load = ((insn >> 22) & 1U) != 0;
        std::array<u8, 64> buf{};
        std::array<std::array<u8, 16>, 4> regs{};
        if (load) {
            if (!FastReadBytes(memory, m_core_index, allow_hot, base, buf.data(), total_bytes)) {
                return false;
            }
            std::size_t offs = 0;
            for (u32 r = 0; r < rpt; ++r) {
                for (std::size_t e = 0; e < elements; ++e) {
                    for (u32 s = 0; s < selem; ++s) {
                        std::memcpy(regs[r + s].data() + e * esize, buf.data() + offs, esize);
                        offs += esize;
                    }
                }
            }
            for (u32 i = 0; i < num_regs; ++i) {
                SetVectorRegister((rt + i) & 31U, regs[i].data());
            }
        } else {
            for (u32 i = 0; i < num_regs; ++i) {
                GetVectorRegister((rt + i) & 31U, regs[i].data());
            }
            std::size_t offs = 0;
            for (u32 r = 0; r < rpt; ++r) {
                for (std::size_t e = 0; e < elements; ++e) {
                    for (u32 s = 0; s < selem; ++s) {
                        std::memcpy(buf.data() + offs, regs[r + s].data() + e * esize, esize);
                        offs += esize;
                    }
                }
            }
            if (!FastWriteBytes(memory, m_core_index, allow_hot, base, buf.data(), total_bytes)) {
                return false;
            }
        }
        if (post_index) {
            SetBaseRegister(rn, base + ((rm == 31U) ? static_cast<u64>(total_bytes)
                                                    : GetBaseRegister(rm)));
        }
        return true;
    }

    // AdvSIMD LD1..LD4 / ST1..ST4 single structure (lane) and LD1R..LD4R (replicate).
    if ((insn & 0xBF000000U) == 0x0D000000U) {
        const bool post_index = ((insn >> 23) & 1U) != 0;
        const u32 rm = (insn >> 16) & 0x1FU;
        if (!post_index && rm != 0) {
            return false;
        }
        const bool q = ((insn >> 30) & 1U) != 0;
        const bool load = ((insn >> 22) & 1U) != 0;
        const u32 opcode = (insn >> 13) & 7U;
        const u32 s_bit = (insn >> 12) & 1U;
        const u32 size = (insn >> 10) & 3U;
        const u32 selem = (((opcode & 1U) << 1) | ((insn >> 21) & 1U)) + 1U;
        u32 scale = 0;
        u32 lane = 0;
        bool replicate = false;
        switch (opcode >> 1) {
        case 0:
            scale = 0;
            lane = ((q ? 1U : 0U) << 3) | (s_bit << 2) | size;
            break;
        case 1:
            if ((size & 1U) != 0) {
                return false;
            }
            scale = 1;
            lane = ((q ? 1U : 0U) << 2) | (s_bit << 1) | (size >> 1);
            break;
        case 2:
            if (size == 0) {
                scale = 2;
                lane = ((q ? 1U : 0U) << 1) | s_bit;
            } else if (size == 1U && s_bit == 0) {
                scale = 3;
                lane = q ? 1U : 0U;
            } else {
                return false;
            }
            break;
        case 3:
            if (!load || s_bit != 0 || (size == 3U && !q)) {
                return false;
            }
            scale = size;
            replicate = true;
            break;
        }
        const std::size_t esize = std::size_t{1} << scale;
        const std::size_t total_bytes = esize * selem;
        const u64 base = GetBaseRegister(rn);
        std::array<u8, 32> buf{};
        if (replicate) {
            const std::size_t reg_bytes = q ? 16 : 8;
            if (!FastReadBytes(memory, m_core_index, allow_hot, base, buf.data(), total_bytes)) {
                return false;
            }
            for (u32 s = 0; s < selem; ++s) {
                std::array<u8, 16> vbytes{};
                for (std::size_t off = 0; off < reg_bytes; off += esize) {
                    std::memcpy(vbytes.data() + off, buf.data() + s * esize, esize);
                }
                SetVectorRegister((rt + s) & 31U, vbytes.data());
            }
        } else if (load) {
            if (!FastReadBytes(memory, m_core_index, allow_hot, base, buf.data(), total_bytes)) {
                return false;
            }
            for (u32 s = 0; s < selem; ++s) {
                std::array<u8, 16> vbytes{};
                GetVectorRegister((rt + s) & 31U, vbytes.data());
                std::memcpy(vbytes.data() + lane * esize, buf.data() + s * esize, esize);
                SetVectorRegister((rt + s) & 31U, vbytes.data());
            }
        } else {
            for (u32 s = 0; s < selem; ++s) {
                std::array<u8, 16> vbytes{};
                GetVectorRegister((rt + s) & 31U, vbytes.data());
                std::memcpy(buf.data() + s * esize, vbytes.data() + lane * esize, esize);
            }
            if (!FastWriteBytes(memory, m_core_index, allow_hot, base, buf.data(), total_bytes)) {
                return false;
            }
        }
        if (post_index) {
            SetBaseRegister(rn, base + ((rm == 31U) ? static_cast<u64>(total_bytes)
                                                    : GetBaseRegister(rm)));
        }
        return true;
    }

    const auto atomic_read = [&](u64 addr, u32 sz) -> u64 {
        switch (sz) {
        case 0: return memory.Read8(addr);
        case 1: return memory.Read16(addr);
        case 2: return memory.Read32(addr);
        default: return memory.Read64(addr);
        }
    };
    const auto atomic_cas = [&](u64 addr, u32 sz, u64 desired, u64 expected) -> bool {
        switch (sz) {
        case 0:
            return memory.WriteExclusive8(addr, static_cast<u8>(desired),
                                          static_cast<u8>(expected));
        case 1:
            return memory.WriteExclusive16(addr, static_cast<u16>(desired),
                                           static_cast<u16>(expected));
        case 2:
            return memory.WriteExclusive32(addr, static_cast<u32>(desired),
                                           static_cast<u32>(expected));
        default:
            return memory.WriteExclusive64(addr, desired, expected);
        }
    };
    const auto atomic_mask = [](u32 sz) -> u64 {
        return sz == 3U ? ~u64{0} : ((u64{1} << (8U << sz)) - 1);
    };
    const auto finish_excl_load = [&](u64 addr) {
        const u64 page = addr & ~u64{0xFFF};
        if (m_excl_rearm_page == page) {
            m_excl_rearm_page = 0;
            if (!Settings::values.use_reactive_flushing.GetValue() &&
                !IsReadTrapArmed(memory, page)) {
                m_vm.ProtectAccess(page, Memory::YUZU_PAGESIZE, Common::MemoryPermission::Read);
            }
        }
    };
    const auto arm_excl_retry = [&](u64 addr) {
        if (m_vcpu_has_context && m_vm.Permits(addr, false)) {
            const u64 page = addr & ~u64{0xFFF};
            m_excl_rearm_page = page;
            m_vm.ProtectAccess(page, Memory::YUZU_PAGESIZE, Common::MemoryPermission{});
        }
    };

    // LSE Compare-and-Swap: CAS/CASA/CASL/CASAL/CASB/CASH (8/16/32/64-bit).
    if ((insn & 0x3FA07C00U) == 0x08A07C00U) {
        const u32 sz = insn >> 30;
        const u64 bytes = u64{1} << sz;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (bytes - 1)) != 0 || !memory.IsValidVirtualAddressRange(addr, bytes)) {
            return false;
        }
        const u32 rs = (insn >> 16) & 0x1FU;
        const bool ordered = (((insn >> 22) & 1U) | ((insn >> 15) & 1U)) != 0;
        const u64 mask = atomic_mask(sz);
        const u64 expected = get_x_or_zero(rs) & mask;
        const u64 desired = get_x_or_zero(rt) & mask;
        if (ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        u64 current = atomic_read(addr, sz);
        while (current == expected) {
            if (atomic_cas(addr, sz, desired, current)) {
                break;
            }
            current = atomic_read(addr, sz);
        }
        if (ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        if (rs != 31U) {
            SetBaseRegister(rs, current);
        }
        return true;
    }

    // LSE Compare-and-Swap Pair: CASP/CASPA/CASPL/CASPAL (32-bit & 64-bit pairs).
    if ((insn & 0xBFA07C00U) == 0x08207C00U) {
        const u32 rs = (insn >> 16) & 0x1FU;
        if ((rs & 1U) != 0 || (rt & 1U) != 0) {
            return false;
        }
        const bool is_64_pair = ((insn >> 30) & 1U) != 0;
        const u64 total_bytes = is_64_pair ? 16 : 8;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (total_bytes - 1)) != 0 ||
            !memory.IsValidVirtualAddressRange(addr, total_bytes)) {
            return false;
        }
        const bool ordered = (((insn >> 22) & 1U) | ((insn >> 15) & 1U)) != 0;
        if (ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        if (is_64_pair) {
            const u128 expected{GetBaseRegister(rs), GetBaseRegister(rs + 1)};
            const u128 desired{GetBaseRegister(rt), GetBaseRegister(rt + 1)};
            u128 current = memory.Read128(addr);
            while (current == expected) {
                if (memory.WriteExclusive128(addr, desired, current)) {
                    break;
                }
                current = memory.Read128(addr);
            }
            SetBaseRegister(rs, current[0]);
            SetBaseRegister(rs + 1, current[1]);
        } else {
            const u64 expected = (static_cast<u64>(static_cast<u32>(GetBaseRegister(rs + 1))) << 32) |
                                 static_cast<u32>(GetBaseRegister(rs));
            const u64 desired = (static_cast<u64>(static_cast<u32>(GetBaseRegister(rt + 1))) << 32) |
                                static_cast<u32>(GetBaseRegister(rt));
            u64 current = memory.Read64(addr);
            while (current == expected) {
                if (memory.WriteExclusive64(addr, desired, current)) {
                    break;
                }
                current = memory.Read64(addr);
            }
            SetBaseRegister(rs, static_cast<u32>(current));
            SetBaseRegister(rs + 1, static_cast<u32>(current >> 32));
        }
        if (ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        return true;
    }

    // Exclusive Load single: LDXR / LDAXR (8/16/32/64-bit).
    if ((insn & 0x3FFF7C00U) == 0x085F7C00U) {
        const u32 sz = insn >> 30;
        const u64 bytes = u64{1} << sz;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (bytes - 1)) != 0 || !memory.IsValidVirtualAddressRange(addr, bytes)) {
            return false;
        }
        const u64 val = atomic_read(addr, sz);
        if (rt != 31U) {
            SetBaseRegister(rt, val);
        }
        if (((insn >> 15) & 1U) != 0) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        m_excl_addr = addr;
        m_excl_val = {val, 0};
        m_excl_size = static_cast<u32>(bytes);
        m_excl_valid = true;
        finish_excl_load(addr);
        return true;
    }

    // Exclusive Store single: STXR / STLXR (8/16/32/64-bit).
    if ((insn & 0x3FE07C00U) == 0x08007C00U) {
        const u32 sz = insn >> 30;
        const u64 bytes = u64{1} << sz;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (bytes - 1)) != 0 || !memory.IsValidVirtualAddressRange(addr, bytes)) {
            return false;
        }
        const u32 rs = (insn >> 16) & 0x1FU;
        const bool release = ((insn >> 15) & 1U) != 0;
        const u64 desired = get_x_or_zero(rt) & atomic_mask(sz);
        bool ok = false;
        if (m_excl_valid && m_excl_addr == addr && m_excl_size == bytes) {
            if (release) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
            ok = atomic_cas(addr, sz, desired, m_excl_val[0]);
            if (release) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
            m_excl_valid = false;
        } else {
            m_excl_valid = false;
            arm_excl_retry(addr);
        }
        if (rs != 31U) {
            SetBaseRegister(rs, ok ? 0 : 1);
        }
        return true;
    }

    // Exclusive Load pair: LDXP / LDAXP (32-bit & 64-bit pairs).
    if ((insn & 0xBFFF0000U) == 0x887F0000U) {
        const bool is_64_pair = ((insn >> 30) & 1U) != 0;
        const u64 total_bytes = is_64_pair ? 16 : 8;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (total_bytes - 1)) != 0 ||
            !memory.IsValidVirtualAddressRange(addr, total_bytes)) {
            return false;
        }
        const u32 rt2_excl = (insn >> 10) & 0x1FU;
        if (is_64_pair) {
            const u128 val = memory.Read128(addr);
            if (rt != 31U) {
                SetBaseRegister(rt, val[0]);
            }
            if (rt2_excl != 31U) {
                SetBaseRegister(rt2_excl, val[1]);
            }
            m_excl_val = val;
        } else {
            const u64 val = memory.Read64(addr);
            if (rt != 31U) {
                SetBaseRegister(rt, static_cast<u32>(val));
            }
            if (rt2_excl != 31U) {
                SetBaseRegister(rt2_excl, static_cast<u32>(val >> 32));
            }
            m_excl_val = {val, 0};
        }
        if (((insn >> 15) & 1U) != 0) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        m_excl_addr = addr;
        m_excl_size = static_cast<u32>(total_bytes);
        m_excl_valid = true;
        finish_excl_load(addr);
        return true;
    }

    // Exclusive Store pair: STXP / STLXP (32-bit & 64-bit pairs).
    if ((insn & 0xBFE00000U) == 0x88200000U) {
        const bool is_64_pair = ((insn >> 30) & 1U) != 0;
        const u64 total_bytes = is_64_pair ? 16 : 8;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (total_bytes - 1)) != 0 ||
            !memory.IsValidVirtualAddressRange(addr, total_bytes)) {
            return false;
        }
        const u32 rs = (insn >> 16) & 0x1FU;
        const u32 rt2_excl = (insn >> 10) & 0x1FU;
        const bool release = ((insn >> 15) & 1U) != 0;
        const u64 lo = get_x_or_zero(rt);
        const u64 hi = get_x_or_zero(rt2_excl);
        bool ok = false;
        if (m_excl_valid && m_excl_addr == addr && m_excl_size == total_bytes) {
            if (release) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
            if (is_64_pair) {
                ok = memory.WriteExclusive128(addr, u128{lo, hi}, m_excl_val);
            } else {
                const u64 desired =
                    (static_cast<u64>(static_cast<u32>(hi)) << 32) | static_cast<u32>(lo);
                ok = memory.WriteExclusive64(addr, desired, m_excl_val[0]);
            }
            if (release) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
            m_excl_valid = false;
        } else {
            m_excl_valid = false;
            arm_excl_retry(addr);
        }
        if (rs != 31U) {
            SetBaseRegister(rs, ok ? 0 : 1);
        }
        return true;
    }

    u32 size_log2{};
    bool is_load{};
    bool is_vector{};
    bool sign_extend{};
    bool sign_to_64{};
    bool is_pair{};
    bool is_ordered{};
    u32 rt2{};
    u64 address{};
    std::optional<u64> writeback;

    const auto decode_single = [&]() {
        const u32 size = insn >> 30;
        const u32 opc = (insn >> 22) & 3U;
        is_vector = ((insn >> 26) & 1U) != 0;
        if (is_vector) {
            size_log2 = ((opc & 2U) << 1) | size;
            is_load = (opc & 1U) != 0;
            return size_log2 <= 4U;
        }
        size_log2 = size;
        is_load = opc != 0;
        sign_extend = opc >= 2U;
        sign_to_64 = opc == 2U;
        return !(size == 3U && opc >= 2U) && !(size == 2U && opc == 3U);
    };

    if ((insn & 0x3B000000U) == 0x39000000U) {
        if (!decode_single()) {
            return false;
        }
        address = GetBaseRegister(rn) + (static_cast<u64>((insn >> 10) & 0xFFFU) << size_log2);
    } else if ((insn & 0x3B200000U) == 0x38000000U) {
        if (!decode_single()) {
            return false;
        }
        const u32 mode = (insn >> 10) & 3U;
        if (is_vector && mode == 2U) {
            return false;
        }
        const u64 base = GetBaseRegister(rn);
        const u64 offset = sign_extend_imm((insn >> 12) & 0x1FFU, 9);
        switch (mode) {
        case 1:
            address = base;
            writeback = base + offset;
            break;
        case 3:
            address = base + offset;
            writeback = address;
            break;
        default:
            address = base + offset;
            break;
        }
    } else if ((insn & 0x3B200C00U) == 0x38200800U) {
        if (!decode_single()) {
            return false;
        }
        const u32 rm = (insn >> 16) & 0x1FU;
        const u32 option = (insn >> 13) & 7U;
        u64 offset = rm == 31U ? 0 : GetBaseRegister(rm);
        switch (option) {
        case 2:
            offset = static_cast<u32>(offset);
            break;
        case 6:
            offset = static_cast<u64>(static_cast<s64>(static_cast<s32>(offset)));
            break;
        case 3:
        case 7:
            break;
        default:
            return false;
        }
        if (((insn >> 12) & 1U) != 0) {
            offset <<= size_log2;
        }
        address = GetBaseRegister(rn) + offset;
    } else if ((insn & 0x3A000000U) == 0x28000000U) {
        const u32 opc = insn >> 30;
        const u32 mode = (insn >> 23) & 3U;
        is_pair = true;
        is_vector = ((insn >> 26) & 1U) != 0;
        is_load = ((insn >> 22) & 1U) != 0;
        rt2 = (insn >> 10) & 0x1FU;
        if (is_vector) {
            if (opc == 3U) {
                return false;
            }
            size_log2 = 2U + opc;
        } else {
            if (opc == 3U || (opc == 1U && (!is_load || mode == 0U))) {
                return false;
            }
            size_log2 = opc == 2U ? 3U : 2U;
            sign_extend = opc == 1U;
            sign_to_64 = sign_extend;
        }
        const u64 base = GetBaseRegister(rn);
        const u64 offset = sign_extend_imm((insn >> 15) & 0x7FU, 7) << size_log2;
        switch (mode) {
        case 1:
            address = base;
            writeback = base + offset;
            break;
        case 3:
            address = base + offset;
            writeback = address;
            break;
        default:
            address = base + offset;
            break;
        }
    } else if ((insn & 0x3FFF7C00U) == 0x08DF7C00U || (insn & 0x3FFFFC00U) == 0x38BFC000U ||
               (insn & 0x3FFF7C00U) == 0x089F7C00U) {
        size_log2 = insn >> 30;
        is_load = (insn & 0x3FFF7C00U) != 0x089F7C00U;
        is_ordered = true;
        address = GetBaseRegister(rn);
    } else if ((insn & 0x3F200C00U) == 0x38200000U) {
        const u32 op = (insn >> 12) & 0xFU;
        if (op > 8) {
            return false;
        }
        const u32 sz = insn >> 30;
        const u64 bytes = u64{1} << sz;
        const u64 addr = GetBaseRegister(rn);
        if ((addr & (bytes - 1)) != 0) {
            return false;
        }
        const u32 rs = (insn >> 16) & 0x1FU;
        const u64 mask = atomic_mask(sz);
        const u32 bits = 8U << sz;
        const u64 src = get_x_or_zero(rs) & mask;
        u64 old_val = atomic_read(addr, sz);
        while (true) {
            u64 new_val = src;
            switch (op) {
            case 0:
                new_val = (old_val + src) & mask;
                break;
            case 1:
                new_val = old_val & ~src;
                break;
            case 2:
                new_val = old_val ^ src;
                break;
            case 3:
                new_val = old_val | src;
                break;
            case 4: {
                const s64 so = sz == 3 ? static_cast<s64>(old_val)
                                       : static_cast<s64>(sign_extend_imm(static_cast<u32>(old_val), bits));
                const s64 ss = sz == 3 ? static_cast<s64>(src)
                                       : static_cast<s64>(sign_extend_imm(static_cast<u32>(src), bits));
                new_val = static_cast<u64>(std::max<s64>(so, ss)) & mask;
                break;
            }
            case 5: {
                const s64 so = sz == 3 ? static_cast<s64>(old_val)
                                       : static_cast<s64>(sign_extend_imm(static_cast<u32>(old_val), bits));
                const s64 ss = sz == 3 ? static_cast<s64>(src)
                                       : static_cast<s64>(sign_extend_imm(static_cast<u32>(src), bits));
                new_val = static_cast<u64>(std::min<s64>(so, ss)) & mask;
                break;
            }
            case 6:
                new_val = std::max<u64>(old_val, src);
                break;
            case 7:
                new_val = std::min<u64>(old_val, src);
                break;
            default:
                new_val = src;
                break;
            }
            if (atomic_cas(addr, sz, new_val, old_val)) {
                break;
            }
            old_val = atomic_read(addr, sz);
        }
        if (rt != 31) {
            SetBaseRegister(rt, old_val);
        }
        return true;
    } else if ((insn & 0x3F200C00U) == 0x19000000U) {
        if (!decode_single()) {
            return false;
        }
        is_ordered = true;
        address = GetBaseRegister(rn) + sign_extend_imm((insn >> 12) & 0x1FF, 9);
    } else if ((insn & 0x3B000000U) == 0x18000000U) {
        const u32 opc = insn >> 30;
        is_vector = ((insn >> 26) & 1) != 0;
        is_load = true;
        if (opc == 3) {
            return !is_vector;
        }
        if (is_vector) {
            size_log2 = 2 + opc;
        } else {
            size_log2 = opc == 1 ? 3 : 2;
            sign_extend = opc == 2;
            sign_to_64 = sign_extend;
        }
        address = pc + (sign_extend_imm((insn >> 5) & 0x7FFFF, 19) << 2);
    } else if ((insn & 0xFF200400U) == 0xF8200400U) {
        size_log2 = 3;
        is_load = true;
        const u32 imm10 = (((insn >> 22) & 1U) << 9) | ((insn >> 12) & 0x1FFU);
        address = GetBaseRegister(rn) + (sign_extend_imm(imm10, 10) << 3);
        if (((insn >> 11) & 1U) != 0) {
            writeback = address;
        }
    } else {
        return false;
    }

    const size_t size = size_t{1} << size_log2;
    const size_t total = is_pair ? size * 2 : size;
    const auto read_register = [&](u32 index, u8* out) {
        if (is_vector) {
            std::array<u8, 16> vbytes{};
            GetVectorRegister(index, vbytes.data());
            std::memcpy(out, vbytes.data(), size);
        } else {
            const u64 value = index == 31U ? 0 : GetBaseRegister(index);
            std::memcpy(out, &value, size);
        }
    };
    const auto write_register = [&](u32 index, const u8* in) {
        if (is_vector) {
            std::array<u8, 16> bytes{};
            std::memcpy(bytes.data(), in, size);
            SetVectorRegister(index, bytes.data());
            return;
        }
        if (index == 31U) {
            return;
        }
        u64 value{};
        std::memcpy(&value, in, size);
        if (sign_extend && size < 8) {
            value = sign_extend_imm(static_cast<u32>(value), static_cast<u32>(size * 8));
            if (!sign_to_64) {
                value = static_cast<u32>(value);
            }
        }
        SetBaseRegister(index, value);
    };

    std::array<u8, 32> data{};
    if (is_load) {
        if (!FastReadBytes(memory, m_core_index, allow_hot, address, data.data(), total)) {
            return false;
        }
        write_register(rt, data.data());
        if (is_pair) {
            write_register(rt2, data.data() + size);
        }
        if (is_ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
    } else {
        if (is_ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        read_register(rt, data.data());
        if (is_pair) {
            read_register(rt2, data.data() + size);
        }
        if (!FastWriteBytes(memory, m_core_index, allow_hot, address, data.data(), total)) {
            return false;
        }
        if (is_ordered) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
    }
    if (writeback) {
        SetBaseRegister(rn, *writeback);
    }
    return true;
}

HaltReason ArmHypervisor::HandleGuestException(Kernel::KThread* thread) {
    u64 guest_x0{};
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_TPIDR_EL1, &guest_x0);

    u64 esr{}, elr{}, spsr{};
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_ESR_EL1, &esr);
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_ELR_EL1, &elr);
    hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_SPSR_EL1, &spsr);

    const auto return_to = [&](u64 pc) {
        if (!m_stepping) {
            spsr &= ~PstateSoftwareStep;
        }
        HV_CHECK(hv_vcpu_set_reg(m_vcpu, HV_REG_PC, pc));
        HV_CHECK(hv_vcpu_set_reg(m_vcpu, HV_REG_CPSR, spsr));
    };

    const u32 ec = ExceptionClass(esr);
    const bool sysreg_writes_x0 =
        ec == ExceptionClassSysReg && (esr & 1U) != 0 && ((esr >> 5) & 0x1FU) == 0U;
    if (ec != ExceptionClassSvc64 && !sysreg_writes_x0) {
        hv_vcpu_set_reg(m_vcpu, HV_REG_X0, guest_x0);
    }

    switch (ec) {
    case ExceptionClassSvc64: {
        FinishTrackedAccess();
        if (m_excl_valid || m_excl_rearm_page != 0) {
            ClearExclusiveMonitor(thread->GetOwnerProcess());
        }
        const u32 svc = static_cast<u32>(esr & 0xFFFF);
        return_to(elr);
        if (!m_system.DebuggerEnabled()) {
            bool handled = false;
            u64 fast_x0 = 0;
            bool writes_x0 = false;
            switch (svc) {
            case 0x0BU: {
                const s64 ns = static_cast<s64>(guest_x0);
                if (ns <= 0 && ns >= -2) {
                    u64 spin_hash = 0;
                    if (Kernel::Svc::YieldBackoffEnabled()) {
                        spin_hash = Kernel::Svc::HashYieldSpinWord(0, elr);
                        spin_hash = Kernel::Svc::HashYieldSpinWord(spin_hash, GetBaseRegister(31));
                        for (u32 i = 19; i <= 29; ++i) {
                            spin_hash =
                                Kernel::Svc::HashYieldSpinWord(spin_hash, GetBaseRegister(i));
                        }
                    }
                    handled = Kernel::Svc::TryFastFutileYield64(m_system, ns, spin_hash);
                }
                break;
            }
            case 0x10U:
                handled = Kernel::Svc::TryFastGetCurrentProcessorNumber64(m_system, &fast_x0);
                writes_x0 = true;
                break;
            case 0x1AU:
                handled = Kernel::Svc::TryFastArbitrateLock64(
                    m_system, static_cast<u32>(guest_x0), GetBaseRegister(1),
                    static_cast<u32>(GetBaseRegister(2)), &fast_x0);
                writes_x0 = true;
                break;
            case 0x34U:
                handled = Kernel::Svc::TryFastWaitForAddress64(
                    m_system, guest_x0, static_cast<u32>(GetBaseRegister(1)),
                    static_cast<s64>(GetBaseRegister(2)), static_cast<s64>(GetBaseRegister(3)),
                    &fast_x0);
                writes_x0 = true;
                break;
            case 0x35U:
                handled = Kernel::Svc::TryFastSignalToAddress64(
                    m_system, guest_x0, static_cast<u32>(GetBaseRegister(1)),
                    static_cast<s32>(GetBaseRegister(2)), static_cast<s32>(GetBaseRegister(3)),
                    &fast_x0);
                writes_x0 = true;
                break;
            default:
                break;
            }
            if (handled) {
                hv_vcpu_set_reg(m_vcpu, HV_REG_X0, writes_x0 ? fast_x0 : guest_x0);
                return {};
            }
        }
        hv_vcpu_set_reg(m_vcpu, HV_REG_X0, guest_x0);
        m_svc = svc;
        return HaltReason::SupervisorCall;
    }

    case ExceptionClassSysReg:
        FinishTrackedAccess();
        return_to(elr + 4);
        if (!EmulateSystemRegister(esr)) {
            const u32 rt = (esr >> 5) & 0x1F;
            if ((esr & 1) != 0 && rt != 31) {
                hv_vcpu_set_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + rt), 0);
            }
        }
        return {};

    case ExceptionClassDataAbortLower:
    case ExceptionClassInstructionAbortLower: {
        u64 far{};
        hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_FAR_EL1, &far);
        const bool is_prefetch = ExceptionClass(esr) == ExceptionClassInstructionAbortLower;
        auto* process = thread->GetOwnerProcess();
        auto& memory = process->GetMemory();
        const auto& page_table = process->GetPageTable().GetImpl();
        const u64 page = far & ~u64{0xFFF};

        const bool repeated = elr == m_last_fault_pc && (far == m_last_fault_address || m_stepping);
        m_fault_repeat = repeated ? m_fault_repeat + 1 : 0;
        m_last_fault_pc = elr;
        m_last_fault_address = far;

        const bool is_write = !is_prefetch && (esr & (1ULL << 6)) != 0;
        // Stage-1 permission faults (FSC 0b0011xx) only fire when a valid stage-1 leaf descriptor
        // with AccessFlag is already installed, so MapFromPageTable cannot add a mapping.
        const bool is_permission_fault = (esr & 0x3CU) == 0x0CU;
        bool retry = false;
        if (m_fault_repeat < 64) {
            if (!is_permission_fault) {
                const bool read_trap_armed = IsReadTrapArmed(memory, page);
                retry = m_vm.MapFromPageTable(page_table, page, read_trap_armed);
            }
            if (!retry) {
                retry = m_vm.Permits(page, is_write);
            }
        }
        if (!retry && !is_prefetch &&
            (IsRasterizerCached(process, page) || IsDebugMemory(process, page))) {
            if (!m_stepping && EmulateTrackedAccess(process, elr)) {
                m_fault_repeat = 0;
                return_to(elr + 4);
                return {};
            }
        }
        if (!retry && !is_prefetch && m_fault_repeat < 64 && IsRasterizerCached(process, page)) {
            if (m_fault_repeat >= 4) {
                std::this_thread::yield();
            }
            if (is_write) {
                memory.InvalidateNCE(page, Memory::YUZU_PAGESIZE);
            } else {
                memory.DownloadRasterizerRegion(page, Memory::YUZU_PAGESIZE);
            }
            if (!IsRasterizerCached(process, page)) {
                m_vm.RestoreGuestAccess(page_table, page);
            }
            if (!IsRasterizerCached(process, page) || m_vm.Permits(page, is_write)) {
                if (m_stepping && elr != m_step_pc) {
                    FinishTrackedAccess();
                }
                if (m_stepping) {
                    spsr |= PstateSoftwareStep;
                }
                retry = true;
            } else if (BeginTrackedStep(elr, page, is_write)) {
                spsr |= PstateSoftwareStep;
                retry = true;
            }
        }
        if (!retry && !is_prefetch && m_fault_repeat < 64 &&
            IsNormalMemory(process, page) && m_vm.GuestPermits(page, is_write)) {
            std::this_thread::yield();
            if (IsRasterizerCached(process, page)) {
                if (!m_stepping && EmulateTrackedAccess(process, elr)) {
                    m_fault_repeat = 0;
                    return_to(elr + 4);
                    return {};
                }
                if (BeginTrackedStep(elr, page, is_write)) {
                    spsr |= PstateSoftwareStep;
                    retry = true;
                }
            } else if (m_fault_repeat >= 1 && m_vm.RestoreGuestAccess(page_table, page)) {
                retry = true;
            } else {
                retry = true;
            }
        }
        if (!retry && m_fault_repeat < 4 && IsMapped(process, page)) {
            retry = true;
        }
        if (retry) {
            return_to(elr);
            const u64 stub_va = is_prefetch ? HvVm::FlushStubVa : HvVm::FlushPageStubVa;
            const u64 stub_arg = is_prefetch ? 0 : (page >> Memory::YUZU_PAGEBITS);
            if (!RunFlushStub(stub_va, stub_arg)) {
                return HaltReason::PrefetchAbort;
            }
            return {};
        }

        FinishTrackedAccess();
        if (is_prefetch) {
            LOG_ERROR(Core_ARM, "Hypervisor: instruction abort at {:#x} (esr={:#x})", elr, esr);
            return_to(elr);
            return HaltReason::PrefetchAbort;
        }

        const u32 page_type = (page >> page_table.GetAddressSpaceBits()) == 0
                                  ? static_cast<u32>(page_table.entries[page >> 12].Type())
                                  : ~0U;
        LOG_ERROR(Core_ARM,
                  "Hypervisor: unhandled data abort at pc={:#x} address={:#x} "
                  "(esr={:#x}, page type {}, attempt {}), skipping",
                  elr, far, esr, page_type, m_fault_repeat);
        return_to(elr + 4);
        return {};
    }

    case ExceptionClassBrk64:
        FinishTrackedAccess();
        if (!m_system.DebuggerEnabled()) {
            LOG_ERROR(Core_ARM, "Hypervisor: guest BRK #{:#x} at {:#x}", esr & 0xFFFF, elr);
        }
        return_to(elr);
        return HaltReason::InstructionBreakpoint;

    case ExceptionClassUnknown:
        FinishTrackedAccess();
        if (EmulateMemoryTagging(thread->GetOwnerProcess(), elr)) {
            return_to(elr + 4);
            return {};
        }
        [[fallthrough]];
    default: {
        FinishTrackedAccess();
        u64 far{};
        hv_vcpu_get_sys_reg(m_vcpu, HV_SYS_REG_FAR_EL1, &far);
        LOG_ERROR(Core_ARM,
                  "Hypervisor: unhandled guest exception at {:#x} (esr={:#x}, far={:#x})", elr,
                  esr, far);
        return_to(elr);
        return HaltReason::PrefetchAbort;
    }
    }
}

HaltReason ArmHypervisor::RunThread(Kernel::KThread* thread) {
    if (const auto hr = static_cast<HaltReason>(m_halt.exchange(0)); True(hr)) {
        return hr;
    }
    if (!EnsureVcpu()) {
        return HaltReason::PrefetchAbort;
    }

    if (thread != nullptr) {
        if (auto* const process = thread->GetOwnerProcess();
            process != nullptr && process->GetMemory().HotCore(m_core_index) == nullptr) {
            process->GetMemory().PrepareHotCores();
        }
    }
    if (!m_vcpu_has_context) {
        LoadVcpuContext();
        m_vcpu_has_context = true;
        m_tpidrro_dirty = false;
    } else if (m_tpidrro_dirty) {
        hv_vcpu_set_sys_reg(m_vcpu, HV_SYS_REG_TPIDRRO_EL0, m_tpidrro_el0);
        m_tpidrro_dirty = false;
    }
    ArmHeartbeat();
    t_svc_args.valid = false;
    m_running.store(true, std::memory_order_release);
    m_vm.SetActive(m_core_index, true);

    HaltReason hr{};
    bool in_vector = false;
    while (!True(hr)) {
        if (!std::exchange(in_vector, false)) {
            if (const u64 pending = m_halt.exchange(0); pending != 0) {
                hr |= static_cast<HaltReason>(pending);
                break;
            }
            if (const u64 generation = m_vm.FlushGeneration(); generation != m_flushed_generation) {
                const u64 icache_gen = m_vm.IcacheGeneration();
                const u64 stub_va = icache_gen != m_flushed_icache_generation
                                        ? HvVm::FlushStubVa
                                        : HvVm::FlushTlbStubVa;
                m_flushed_generation = generation;
                m_flushed_icache_generation = icache_gen;
                if (!RunFlushStub(stub_va)) {
                    hr |= HaltReason::PrefetchAbort;
                    break;
                }
            }
        }

        if (hv_vcpu_run(m_vcpu) != HV_SUCCESS) {
            LOG_CRITICAL(Core_ARM, "Hypervisor: hv_vcpu_run failed on core {}", m_core_index);
            hr |= HaltReason::PrefetchAbort;
            break;
        }

        const auto check_in_vector = [&] {
            u64 cpsr{};
            hv_vcpu_get_reg(m_vcpu, HV_REG_CPSR, &cpsr);
            in_vector = (cpsr & 0xF) != 0;
            if (!in_vector) {
                return;
            }
            u64 pc{};
            hv_vcpu_get_reg(m_vcpu, HV_REG_PC, &pc);
            if (pc >= HvVm::VectorsVa && pc < HvVm::FlushStubVa &&
                (pc & 0x7F) > HvVm::VectorHvcOffset) {
                hv_vcpu_set_reg(m_vcpu, HV_REG_PC, (pc & ~u64{0x7F}) + HvVm::VectorHvcOffset);
            }
        };

        switch (m_exit->reason) {
        case HV_EXIT_REASON_CANCELED:
            check_in_vector();
            break;
        case HV_EXIT_REASON_VTIMER_ACTIVATED:
            t_thread_vcpus.TimerFor(m_vcpu).armed = false;
            ArmHeartbeat();
            check_in_vector();
            break;
        case HV_EXIT_REASON_EXCEPTION: {
            const u64 syndrome = m_exit->exception.syndrome;
            const u32 ec = ExceptionClass(syndrome);
            if (ec == ExceptionClassHvc64 && (syndrome & 0xFFFF) == 0) {
                hr |= HandleGuestException(thread);
            } else if (ec == ExceptionClassSoftwareStepLower) {
                FinishTrackedAccess();
                check_in_vector();
            } else if (ec == ExceptionClassSysReg) {
                u64 pc{};
                hv_vcpu_get_reg(m_vcpu, HV_REG_PC, &pc);
                hv_vcpu_set_reg(m_vcpu, HV_REG_PC, pc + 4);
                if (!EmulateSystemRegister(syndrome)) {
                    const u32 rt = (syndrome >> 5) & 0x1F;
                    if ((syndrome & 1) != 0 && rt != 31) {
                        hv_vcpu_set_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + rt), 0);
                    }
                }
            } else if (ec == ExceptionClassBrk64) {
                FinishTrackedAccess();
                if (!m_system.DebuggerEnabled()) {
                    u64 pc{};
                    hv_vcpu_get_reg(m_vcpu, HV_REG_PC, &pc);
                    LOG_ERROR(Core_ARM, "Hypervisor: guest BRK #{:#x} at {:#x}", syndrome & 0xFFFF,
                              pc);
                }
                hr |= HaltReason::InstructionBreakpoint;
            } else if (ec == ExceptionClassWfx) {
                u64 pc{};
                hv_vcpu_get_reg(m_vcpu, HV_REG_PC, &pc);
                hv_vcpu_set_reg(m_vcpu, HV_REG_PC, pc + 4);
            } else {
                LOG_CRITICAL(Core_ARM, "Hypervisor: unexpected exit syndrome {:#x} (ipa={:#x})",
                             syndrome, m_exit->exception.physical_address);
                hr |= HaltReason::PrefetchAbort;
            }
            break;
        }
        default:
            LOG_CRITICAL(Core_ARM, "Hypervisor: unexpected exit reason {}",
                         static_cast<u32>(m_exit->reason));
            hr |= HaltReason::PrefetchAbort;
            break;
        }
    }

    m_vm.SetActive(m_core_index, false);
    m_running.store(false, std::memory_order_release);
    FinishTrackedAccess();
    if (m_excl_valid || m_excl_rearm_page != 0) {
        ClearExclusiveMonitor(thread != nullptr ? thread->GetOwnerProcess() : nullptr);
    }
    m_fault_repeat = 0;
    return hr | static_cast<HaltReason>(m_halt.exchange(0));
}

HaltReason ArmHypervisor::StepThread(Kernel::KThread* /*thread*/) {
    return HaltReason::StepThread;
}

void ArmHypervisor::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    if (VcpuHoldsContext()) {
        ReadVcpuContext(ctx);
        return;
    }
    ctx = m_ctx;
}

void ArmHypervisor::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    m_ctx = ctx;
    m_vcpu_has_context = false;
    m_excl_valid = false;
    t_svc_args.valid = false;
}

void ArmHypervisor::SetTpidrroEl0(u64 value) {
    m_tpidrro_el0 = value;
    m_tpidrro_dirty = true;
}

void ArmHypervisor::GetSvcArguments(std::span<uint64_t, 8> args) const {
    if (VcpuHoldsContext()) {
        for (size_t i = 0; i < 8; i++) {
            hv_vcpu_get_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + i), &args[i]);
            t_svc_args.regs[i] = args[i];
        }
        t_svc_args.owner = this;
        t_svc_args.valid = true;
        return;
    }
    for (size_t i = 0; i < 8; i++) {
        args[i] = m_ctx.r[i];
    }
}

void ArmHypervisor::SetSvcArguments(std::span<const uint64_t, 8> args) {
    if (VcpuHoldsContext()) {
        const bool have_cached = t_svc_args.valid && t_svc_args.owner == this;
        for (size_t i = 0; i < 8; i++) {
            if (!have_cached || args[i] != t_svc_args.regs[i]) {
                hv_vcpu_set_reg(m_vcpu, static_cast<hv_reg_t>(HV_REG_X0 + i), args[i]);
                if (have_cached) {
                    t_svc_args.regs[i] = args[i];
                }
            }
        }
        return;
    }
    for (size_t i = 0; i < 8; i++) {
        m_ctx.r[i] = args[i];
    }
}

u32 ArmHypervisor::GetSvcNumber() const {
    return m_svc;
}

void ArmHypervisor::SignalInterrupt(Kernel::KThread* /*thread*/) {
    m_halt.fetch_or(static_cast<u64>(HaltReason::BreakLoop));
    if (m_running.load(std::memory_order_acquire)) {
        hv_vcpus_exit(&m_vcpu, 1);
    }
}

void ArmHypervisor::ClearInstructionCache() {
    m_vm.RequestFlush(true);
}

void ArmHypervisor::InvalidateCacheRange(u64 /*addr*/, std::size_t /*size*/) {
    m_vm.RequestFlush(true);
}

} // namespace Core
