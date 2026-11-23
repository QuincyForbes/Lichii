// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Ported from upstream Djihads80/Eglantine#405 with Switch 2 DRAM sizing.

#include <algorithm>
#include <cstring>
#include <memory>

#include <sys/mman.h>

#include "common/alignment.h"
#include "common/logging.h"
#include "common/page_table.h"
#include "common/settings.h"
#include "core/arm/hypervisor/hv_vm.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/board/nintendo/nx/k_system_control.h"

namespace Core {

namespace {

constexpr u64 HostPageSize = 16 * 1024;

constexpr u64 DescValid = 1ULL << 0;
constexpr u64 DescTableOrPage = 1ULL << 1;
constexpr u64 AttrIndexNormal = 0ULL << 2;
constexpr u64 ApEl0ReadWrite = 1ULL << 6;
constexpr u64 ApEl1ReadOnly = 2ULL << 6;
constexpr u64 ApEl0ReadOnly = 3ULL << 6;
constexpr u64 ShareInner = 3ULL << 8;
constexpr u64 AccessFlag = 1ULL << 10;
constexpr u64 PrivilegedExecuteNever = 1ULL << 53;
constexpr u64 UserExecuteNever = 1ULL << 54;
constexpr u64 SwGuestRead = 1ULL << 55;
constexpr u64 SwGuestWrite = 1ULL << 56;
constexpr u64 SwGuestAccessMask = SwGuestRead | SwGuestWrite;
constexpr u64 AddressMask = 0x000000FFFFFFF000ULL;
constexpr u64 AccessBitsMask = DescValid | ApEl0ReadOnly;
constexpr u64 PermissionMask = ApEl0ReadOnly | UserExecuteNever | DescValid | SwGuestAccessMask;

// Every exception vector traps to the host with "hvc #0". Hypervisor.framework services hvc calls
// whose x0 looks like an SMCCC function ID itself (guest pointers such as 0x84xxxxxx do), so the
// guest's x0 is parked in TPIDR_EL1 and cleared first. The flush stubs at +0x800/+0x820/+0x840
// invalidate VM TLB+ICache, VM TLB only, or a single 4K VA page (x0 = vaddr >> 12) and return
// with "hvc #1"; the host preserves x0.
constexpr std::array<u32, 3> VectorSlot{
    0xD518D080, // msr tpidr_el1, x0
    0xD2800000, // mov x0, #0
    0xD4000002, // hvc #0
};
constexpr std::array<u32, 7> FlushStub{
    0xD2800000, // mov x0, #0
    0xD508831F, // tlbi vmalle1is
    0xD508711F, // ic ialluis
    0xD5033B9F, // dsb ish
    0xD5033FDF, // isb
    0xD4000022, // hvc #1
    0xD4000022,
};
constexpr std::array<u32, 6> FlushTlbStub{
    0xD2800000, // mov x0, #0
    0xD508831F, // tlbi vmalle1is
    0xD5033B9F, // dsb ish
    0xD5033FDF, // isb
    0xD4000022, // hvc #1
    0xD4000022,
};
constexpr std::array<u32, 6> FlushPageStub{
    0xD5088360, // tlbi vaae1is, x0
    0xD2800000, // mov x0, #0
    0xD5033B9F, // dsb ish
    0xD5033FDF, // isb
    0xD4000022, // hvc #1
    0xD4000022,
};

u64 GuestAccessBits(Common::MemoryPermission perms) noexcept {
    if (True(perms & Common::MemoryPermission::Write)) {
        return SwGuestRead | SwGuestWrite;
    }
    return True(perms & Common::MemoryPermission::Read) ? SwGuestRead : 0;
}

u64 DesiredAccessFromGuestBits(u64 guest_bits) noexcept {
    if ((guest_bits & SwGuestWrite) != 0) {
        return DescValid | ApEl0ReadWrite;
    }
    return (guest_bits & SwGuestRead) != 0 ? (DescValid | ApEl0ReadOnly) : 0;
}

u64 ClampToGuestAccess(u64 access_bits, u64 guest_bits) noexcept {
    if ((guest_bits & SwGuestRead) == 0) {
        return 0;
    }
    if (access_bits == (DescValid | ApEl0ReadWrite) && (guest_bits & SwGuestWrite) == 0) {
        return DescValid | ApEl0ReadOnly;
    }
    return access_bits;
}

u64 PermissionBits(Common::MemoryPermission perms) {
    if (perms == Common::MemoryPermission{}) {
        return PrivilegedExecuteNever | UserExecuteNever;
    }
    const u64 guest_bits = GuestAccessBits(perms);
    u64 bits = PrivilegedExecuteNever | guest_bits | DesiredAccessFromGuestBits(guest_bits);
    if (False(perms & Common::MemoryPermission::Execute)) {
        bits |= UserExecuteNever;
    }
    return bits;
}

bool IsAccessTightened(u64 old_pte, u64 new_pte) noexcept {
    return ((old_pte & DescValid) != 0 && (new_pte & DescValid) == 0) ||
           ((old_pte & AccessBitsMask) == (DescValid | ApEl0ReadWrite) &&
            (new_pte & AccessBitsMask) == (DescValid | ApEl0ReadOnly));
}

u64 EffectiveStepAccess(u64 desired_access, u32 read_refs, u32 write_refs) noexcept {
    if (read_refs == 0 && write_refs == 0) {
        return desired_access;
    }
    if (write_refs > 0 || desired_access == (DescValid | ApEl0ReadWrite)) {
        return DescValid | ApEl0ReadWrite;
    }
    return DescValid | ApEl0ReadOnly;
}

std::unique_ptr<HvVm> g_vm;
HvVm* g_active_vm{};

} // Anonymous namespace

HvVm::~HvVm() {
    if (!m_vm_created && m_tables != nullptr && m_tables != MAP_FAILED) {
        munmap(m_tables, m_tables_size);
        m_tables = nullptr;
    }
}

HvVm* HvVm::Get() {
    return g_active_vm;
}

void HvVm::Deactivate() {
    g_active_vm = nullptr;
}

std::unique_ptr<HvVm> HvVm::CreateForTest(u8* dram_base, u64 dram_size) {
    std::unique_ptr<HvVm> vm{new HvVm()};
    return vm->InitializeForTest(dram_base, dram_size) ? std::move(vm) : nullptr;
}

HvVm* HvVm::Create(System& system) {
    if (!g_vm) {
        g_vm.reset(new HvVm());
        if (!g_vm->Initialize(system)) {
            g_vm.reset();
            return nullptr;
        }
    } else if (!g_vm->MapDram(system)) {
        return nullptr;
    } else {
        g_vm->Reset();
    }
    g_active_vm = g_vm.get();
    return g_active_vm;
}

bool HvVm::Initialize(System& system) {
    uint32_t max_ipa_size{};
    hv_vm_config_t config = hv_vm_config_create();
    if (hv_vm_config_get_max_ipa_size(&max_ipa_size) != HV_SUCCESS || max_ipa_size < 40) {
        LOG_ERROR(Core_ARM, "Hypervisor: 40-bit IPA space is unavailable (max {} bits)",
                  max_ipa_size);
        os_release(config);
        return false;
    }
    hv_vm_config_set_ipa_size(config, 40);
    const hv_return_t created = hv_vm_create(config);
    os_release(config);
    if (created != HV_SUCCESS) {
        LOG_ERROR(Core_ARM,
                  "Hypervisor: hv_vm_create failed ({:#x}); is the app signed with "
                  "the com.apple.security.hypervisor entitlement?",
                  static_cast<u32>(created));
        return false;
    }

    if (!MapDram(system)) {
        return false;
    }

    m_tables_size = TableRegionSize;
    m_tables = static_cast<u8*>(
        mmap(nullptr, m_tables_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (m_tables == MAP_FAILED ||
        hv_vm_map(m_tables, TableBaseIpa, m_tables_size,
                  HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC) != HV_SUCCESS) {
        LOG_ERROR(Core_ARM, "Hypervisor: failed to map the page table region");
        return false;
    }

    m_vm_created = true;
    m_step_grants.reserve(64);
    Reset();
    LOG_INFO(Core_ARM, "Hypervisor: VM created, {:#x} bytes of DRAM mapped", m_dram_size);
    return true;
}

bool HvVm::InitializeForTest(u8* dram_base, u64 dram_size) {
    m_dram_base = dram_base;
    m_dram_size = dram_size;
    m_tables_size = TestTableRegionSize;
    m_tables = static_cast<u8*>(
        mmap(nullptr, m_tables_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (m_tables == MAP_FAILED) {
        m_tables = nullptr;
        return false;
    }
    m_vm_created = false;
    m_step_grants.reserve(64);
    Reset();
    return true;
}

bool HvVm::MapDram(System& system) {
    u8* const base = system.DeviceMemory().buffer.BackingBasePointer();
    const u64 backing_bytes = std::max<u64>(
        system.DeviceMemory().buffer.BackingSize(),
        Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize());
    const u64 size = Common::AlignUp(backing_bytes, HostPageSize);
    if (base == m_dram_base && size == m_dram_size) {
        return true;
    }
    if (m_dram_base != nullptr && hv_vm_unmap(0, m_dram_size) != HV_SUCCESS) {
        LOG_ERROR(Core_ARM, "Hypervisor: failed to unmap the previous DRAM");
        return false;
    }
    m_dram_base = nullptr;
    m_dram_size = 0;
    if (hv_vm_map(base, 0, size, HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC) != HV_SUCCESS) {
        LOG_ERROR(Core_ARM, "Hypervisor: failed to map {:#x} bytes of DRAM", size);
        return false;
    }
    m_dram_base = base;
    m_dram_size = size;
    return true;
}

void HvVm::Reset() {
    std::scoped_lock lk{m_table_mutex};

    m_step_grants.clear();
    madvise(m_tables, m_tables_size, MADV_FREE);
    std::memset(m_tables, 0, RootTableOffset + 0x1000);
    for (size_t i = 0; i < 16; i++) {
        std::memcpy(m_tables + i * 0x80, VectorSlot.data(), sizeof(VectorSlot));
    }
    std::memcpy(m_tables + 0x800, FlushStub.data(), sizeof(FlushStub));
    std::memcpy(m_tables + 0x820, FlushTlbStub.data(), sizeof(FlushTlbStub));
    std::memcpy(m_tables + 0x840, FlushPageStub.data(), sizeof(FlushPageStub));
    m_next_table_offset = RootTableOffset + 0x1000;

    u64* entry = EntryFor(VectorsVa, true);
    *entry = TableBaseIpa | DescValid | DescTableOrPage | AttrIndexNormal | ShareInner |
             AccessFlag | ApEl1ReadOnly | UserExecuteNever;

    m_icache_generation.fetch_add(1, std::memory_order_release);
    m_flush_generation.fetch_add(1, std::memory_order_release);
}

u64* HvVm::TableAt(u64 table_ipa) {
    return reinterpret_cast<u64*>(m_tables + (table_ipa - TableBaseIpa));
}

u64* HvVm::EntryFor(u64 vaddr, bool allocate) {
    if ((vaddr >> VaBits) != 0) {
        return nullptr;
    }
    u64* table = TableAt(RootTableIpa());
    for (const u32 shift : {30U, 21U}) {
        u64& descriptor = table[(vaddr >> shift) & 511];
        if ((descriptor & DescValid) == 0) {
            if (!allocate) {
                return nullptr;
            }
            if (m_next_table_offset + 0x1000 > m_tables_size) {
                LOG_CRITICAL(Core_ARM, "Hypervisor: out of page table memory");
                return nullptr;
            }
            const u64 table_ipa = TableBaseIpa + m_next_table_offset;
            m_next_table_offset += 0x1000;
            std::memset(TableAt(table_ipa), 0, 0x1000);
            std::atomic_ref<u64>{descriptor}.store(table_ipa | DescValid | DescTableOrPage,
                                                   std::memory_order_release);
        }
        table = TableAt(descriptor & AddressMask);
    }
    return &table[(vaddr >> 12) & 511];
}

void HvVm::SetPage(u64 vaddr, u64 entry) {
    if (u64* pte = EntryFor(vaddr, true)) {
        std::atomic_ref<u64>{*pte}.store(entry, std::memory_order_release);
    }
}

HvVm::StepGrant* HvVm::FindStepGrant(u64 page) noexcept {
    for (auto& grant : m_step_grants) {
        if ((grant.read_refs != 0 || grant.write_refs != 0) && grant.page == page) {
            return &grant;
        }
    }
    return nullptr;
}

void HvVm::Map(u64 vaddr, u64 phys_offset, u64 size, Common::MemoryPermission perms) {
    const u64 perm_bits = PermissionBits(perms);
    bool replaced = false;
    {
        std::scoped_lock lk{m_table_mutex};
        for (u64 offset = 0; offset < size; offset += 0x1000) {
            const u64 cur = vaddr + offset;
            u64 effective_perm_bits = perm_bits;
            if (StepGrant* grant = FindStepGrant(cur)) {
                grant->desired_access = perm_bits & AccessBitsMask;
                const u64 step_access = ClampToGuestAccess(
                    EffectiveStepAccess(grant->desired_access, grant->read_refs, grant->write_refs),
                    perm_bits & SwGuestAccessMask);
                effective_perm_bits = (perm_bits & ~AccessBitsMask) | step_access;
            }
            if (const u64* pte = EntryFor(cur, false); pte && (*pte & DescValid)) {
                replaced = true;
            }
            SetPage(cur, ((phys_offset + offset) & AddressMask) | effective_perm_bits |
                             DescTableOrPage | AttrIndexNormal | ShareInner | AccessFlag);
        }
    }
    if (replaced) {
        RequestFlush(true);
    }
}

void HvVm::Unmap(u64 vaddr, u64 size) {
    {
        std::scoped_lock lk{m_table_mutex};
        for (u64 offset = 0; offset < size; offset += 0x1000) {
            const u64 cur = vaddr + offset;
            if (StepGrant* grant = FindStepGrant(cur)) {
                *grant = {};
            }
            if (u64* pte = EntryFor(cur, false)) {
                std::atomic_ref<u64>{*pte}.store(0, std::memory_order_release);
            }
        }
    }
    RequestFlush(true);
}

void HvVm::Protect(u64 vaddr, u64 size, Common::MemoryPermission perms) {
    const u64 perm_bits = PermissionBits(perms);
    const u64 guest_bits = perm_bits & SwGuestAccessMask;
    bool changed = false;
    {
        std::scoped_lock lk{m_table_mutex};
        for (u64 offset = 0; offset < size; offset += 0x1000) {
            const u64 cur = vaddr + offset;
            u64* pte = EntryFor(cur, false);
            if (pte == nullptr || (*pte & AccessFlag) == 0) {
                continue;
            }
            u64 effective_perm_bits = perm_bits;
            if (StepGrant* grant = FindStepGrant(cur)) {
                grant->desired_access = perm_bits & AccessBitsMask;
                const u64 step_access = ClampToGuestAccess(
                    EffectiveStepAccess(grant->desired_access, grant->read_refs, grant->write_refs),
                    guest_bits);
                effective_perm_bits = (perm_bits & ~AccessBitsMask) | step_access;
            }
            const u64 old = *pte;
            const u64 entry = (old & ~PermissionMask) | effective_perm_bits;
            if (entry != old) {
                std::atomic_ref<u64>{*pte}.store(entry, std::memory_order_release);
                changed = true;
            }
        }
    }
    if (changed) {
        RequestFlush(true);
    }
}

void HvVm::ProtectAccess(u64 vaddr, u64 size, Common::MemoryPermission perms) {
    u64 requested_access = 0;
    if (True(perms & Common::MemoryPermission::Write)) {
        requested_access = DescValid | ApEl0ReadWrite;
    } else if (True(perms & Common::MemoryPermission::Read)) {
        requested_access = DescValid | ApEl0ReadOnly;
    }
    bool tightened = false;
    {
        std::scoped_lock lk{m_table_mutex};
        for (u64 offset = 0; offset < size; offset += 0x1000) {
            const u64 cur = vaddr + offset;
            u64* pte = EntryFor(cur, false);
            if (pte == nullptr || (*pte & AccessFlag) == 0) {
                continue;
            }
            const u64 old = *pte;
            const u64 guest_bits = old & SwGuestAccessMask;
            const u64 clamped_access = ClampToGuestAccess(requested_access, guest_bits);
            u64 target_access = clamped_access;
            if (StepGrant* grant = FindStepGrant(cur)) {
                grant->desired_access = clamped_access;
                target_access = ClampToGuestAccess(
                    EffectiveStepAccess(clamped_access, grant->read_refs, grant->write_refs),
                    guest_bits);
            }
            const u64 entry = (old & ~AccessBitsMask) | target_access;
            if (entry != old) {
                std::atomic_ref<u64>{*pte}.store(entry, std::memory_order_release);
                if (IsAccessTightened(old, entry)) {
                    tightened = true;
                }
            }
        }
    }
    if (tightened) {
        RequestFlush(false);
    }
}

bool HvVm::BeginStepAccess(u64 page, bool write) {
    page &= ~u64{0xFFF};
    std::scoped_lock lk{m_table_mutex};
    u64* pte = EntryFor(page, false);
    if (pte == nullptr || (*pte & AccessFlag) == 0) {
        return false;
    }
    const u64 guest_bits = *pte & SwGuestAccessMask;
    if ((guest_bits & (write ? SwGuestWrite : SwGuestRead)) == 0) {
        return false;
    }
    StepGrant* grant = FindStepGrant(page);
    if (grant == nullptr) {
        for (auto& candidate : m_step_grants) {
            if (candidate.read_refs == 0 && candidate.write_refs == 0) {
                grant = &candidate;
                break;
            }
        }
        if (grant == nullptr) {
            m_step_grants.push_back(StepGrant{});
            grant = &m_step_grants.back();
        }
        grant->page = page;
        grant->desired_access = *pte & AccessBitsMask;
    }
    if (write) {
        ++grant->write_refs;
    } else {
        ++grant->read_refs;
    }
    const u64 step_access = ClampToGuestAccess(
        EffectiveStepAccess(grant->desired_access, grant->read_refs, grant->write_refs),
        guest_bits);
    const u64 entry = (*pte & ~AccessBitsMask) | step_access;
    if (entry != *pte) {
        std::atomic_ref<u64>{*pte}.store(entry, std::memory_order_release);
    }
    return true;
}

void HvVm::EndStepAccess(u64 page, bool write) {
    page &= ~u64{0xFFF};
    bool tightened = false;
    {
        std::scoped_lock lk{m_table_mutex};
        StepGrant* grant = FindStepGrant(page);
        if (grant == nullptr) {
            return;
        }
        if (write) {
            if (grant->write_refs > 0) {
                --grant->write_refs;
            }
        } else {
            if (grant->read_refs > 0) {
                --grant->read_refs;
            }
        }
        u64* pte = EntryFor(page, false);
        if (pte != nullptr && (*pte & AccessFlag) != 0) {
            const u64 old = *pte;
            const u64 guest_bits = old & SwGuestAccessMask;
            const u64 step_access = ClampToGuestAccess(
                EffectiveStepAccess(grant->desired_access, grant->read_refs, grant->write_refs),
                guest_bits);
            const u64 entry = (old & ~AccessBitsMask) | step_access;
            if (entry != old) {
                std::atomic_ref<u64>{*pte}.store(entry, std::memory_order_release);
                tightened = IsAccessTightened(old, entry);
            }
        }
        if (grant->read_refs == 0 && grant->write_refs == 0) {
            *grant = {};
        }
    }
    if (tightened) {
        RequestFlush(false);
    }
}

bool HvVm::MapFromPageTable(const Common::PageTable& page_table, u64 vaddr, bool read_trap_armed) {
    vaddr &= ~u64{0xFFF};
    if ((vaddr >> page_table.GetAddressSpaceBits()) != 0 || (vaddr >> VaBits) != 0) {
        return false;
    }
    const auto [pointer, type, block] = page_table.entries[vaddr >> 12].PointerTypeBlock(true);
    if (type == Common::PageType::Unmapped || type == Common::PageType::DebugMemory ||
        pointer == 0) {
        return false;
    }
    const u8* host = reinterpret_cast<const u8*>(pointer + vaddr);
    if (host < m_dram_base || host >= m_dram_base + m_dram_size) {
        return false;
    }

    const bool cached = type == Common::PageType::RasterizerCachedMemory;
    std::scoped_lock lk{m_table_mutex};
    if (const u64* pte = EntryFor(vaddr, false); pte != nullptr && (*pte & AccessFlag) != 0) {
        return false;
    }
    u64 perm_bits =
        PermissionBits(Common::MemoryPermission::ReadWrite | Common::MemoryPermission::Execute);
    if (cached) {
        perm_bits = (perm_bits & ~AccessBitsMask) | (DescValid | ApEl0ReadOnly);
        if (Settings::values.use_reactive_flushing.GetValue() || read_trap_armed) {
            perm_bits &= ~AccessBitsMask;
        }
    }
    SetPage(vaddr, (static_cast<u64>(host - m_dram_base) & AddressMask) | perm_bits |
                       DescTableOrPage | AttrIndexNormal | ShareInner | AccessFlag);
    return (perm_bits & DescValid) != 0;
}

bool HvVm::RestoreGuestAccess(const Common::PageTable& page_table, u64 vaddr) {
    vaddr &= ~u64{0xFFF};
    if ((vaddr >> page_table.GetAddressSpaceBits()) != 0 || (vaddr >> VaBits) != 0 ||
        page_table.entries[vaddr >> 12].Type() != Common::PageType::Memory) {
        return false;
    }
    std::scoped_lock lk{m_table_mutex};
    u64* pte = EntryFor(vaddr, false);
    if (pte == nullptr || (*pte & AccessFlag) == 0) {
        return false;
    }
    const u64 old = *pte;
    const u64 guest_bits = old & SwGuestAccessMask;
    const u64 desired = DesiredAccessFromGuestBits(guest_bits);
    if (desired == 0) {
        return false;
    }
    u64 target = desired;
    if (StepGrant* grant = FindStepGrant(vaddr)) {
        grant->desired_access = desired;
        target = ClampToGuestAccess(
            EffectiveStepAccess(desired, grant->read_refs, grant->write_refs), guest_bits);
    }
    const u64 entry = (old & ~AccessBitsMask) | target;
    if (entry != old) {
        std::atomic_ref<u64>{*pte}.store(entry, std::memory_order_release);
    }
    return true;
}

bool HvVm::Permits(u64 vaddr, bool write) {
    vaddr &= ~u64{0xFFF};
    std::scoped_lock lk{m_table_mutex};
    const u64* pte = EntryFor(vaddr, false);
    if (pte == nullptr || (*pte & AccessFlag) == 0) {
        return false;
    }
    const StepGrant* grant = FindStepGrant(vaddr);
    const u64 access = grant != nullptr ? grant->desired_access : (*pte & AccessBitsMask);
    if ((access & DescValid) == 0) {
        return false;
    }
    return !write || (access & ApEl0ReadOnly) == ApEl0ReadWrite;
}

bool HvVm::EffectivePermits(u64 vaddr, bool write) {
    vaddr &= ~u64{0xFFF};
    std::scoped_lock lk{m_table_mutex};
    const u64* pte = EntryFor(vaddr, false);
    if (pte == nullptr || (*pte & DescValid) == 0) {
        return false;
    }
    return !write || (*pte & ApEl0ReadOnly) == ApEl0ReadWrite;
}

bool HvVm::GuestPermits(u64 vaddr, bool write) {
    vaddr &= ~u64{0xFFF};
    std::scoped_lock lk{m_table_mutex};
    const u64* pte = EntryFor(vaddr, false);
    if (pte == nullptr || (*pte & AccessFlag) == 0) {
        return false;
    }
    return (*pte & (write ? SwGuestWrite : SwGuestRead)) != 0;
}

u32 HvVm::ActiveStepCount(u64 page, bool write) {
    page &= ~u64{0xFFF};
    std::scoped_lock lk{m_table_mutex};
    const StepGrant* grant = FindStepGrant(page);
    if (grant == nullptr) {
        return 0;
    }
    return write ? grant->write_refs : (grant->read_refs + grant->write_refs);
}

void HvVm::RequestFlush(bool icache) {
    if (icache) {
        m_icache_generation.fetch_add(1, std::memory_order_acq_rel);
    }
    m_flush_generation.fetch_add(1, std::memory_order_acq_rel);
    if (!m_vm_created) {
        return;
    }
    std::scoped_lock lk{m_running_mutex};
    for (size_t i = 0; i < m_vcpus.size(); i++) {
        if (m_running[i]) {
            hv_vcpus_exit(&m_vcpus[i], 1);
        }
    }
}

void HvVm::SetRunning(size_t core, hv_vcpu_t vcpu, bool running) {
    std::scoped_lock lk{m_running_mutex};
    m_vcpus[core] = vcpu;
    m_running[core] = running;
}

} // namespace Core
