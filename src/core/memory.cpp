// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2015 Citra Emulator Project
// SPDX-FileCopyrightText: 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(ARCHITECTURE_arm64)
#include <arm_neon.h>
#endif

#include <fmt/format.h>

#include "common/assert.h"
#include "common/atomic_ops.h"
#include "common/common_types.h"
#ifdef __ANDROID__
#include "common/heap_tracker.h"
#endif
#include "common/logging.h"
#include "common/page_table.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/gpu_dirty_memory_manager.h"
#include "video_core/nx2_hazard.h"
#include "video_core/nx2_spikes.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/memory.h"
#ifdef HAS_HYPERVISOR
#include "core/arm/hypervisor/hv_vm.h"
#endif
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/host1x.h"
#include "video_core/rasterizer_download_area.h"

namespace Core::Memory {

u32& UnmappedReadFlag() {
    thread_local u32 flag{};
    return flag;
}

namespace {

// Unmapped guest accesses: Ocarina of Time dereferences null about six times a frame, an Error
// line each. The first ones of each kind and address in full, then one count line every 10 s.
class UnmappedAccessLog {
public:
    /// kind: bits per access, + 0x100 for a write, + 0x200 for an exclusive write.
    bool Note(u32 kind, u64 addr) {
        constexpr u64 FullLines = 4;
        const auto now = std::chrono::steady_clock::now();
        std::string summary;
        bool full{};
        {
            std::scoped_lock lock{mutex};
            auto it = std::ranges::find_if(sites, [&](const Site& site) {
                return site.kind == kind && site.addr == addr;
            });
            if (it == sites.end() && sites.size() < 64) {
                it = sites.insert(sites.end(), Site{kind, addr});
            }
            if (it == sites.end()) {
                ++others;
            } else {
                full = it->total++ < FullLines;
                if (!full) {
                    ++it->window;
                }
            }
            if (now - last_report >= std::chrono::seconds{10}) {
                last_report = now;
                if (others != 0) {
                    summary = fmt::format("{} at other addresses", std::exchange(others, 0));
                }
                for (Site& site : sites) {
                    if (site.window == 0) {
                        continue;
                    }
                    summary += fmt::format("{}{}{} @ {:#x} x{}", summary.empty() ? "" : ", ",
                                           (site.kind & 0x200) != 0   ? "WriteExclusive"
                                           : (site.kind & 0x100) != 0 ? "Write"
                                                                      : "Read",
                                           site.kind & 0xff, site.addr, site.window);
                    site.window = 0;
                }
            }
        }
        if (!summary.empty()) {
            LOG_ERROR(HW_Memory, "Unmapped accesses in the last 10 s, past the first {} of each: {}",
                      FullLines, summary);
        }
        return full;
    }

private:
    struct Site {
        u32 kind;
        u64 addr;
        u64 total{};
        u64 window{};
    };
    std::mutex mutex;
    std::vector<Site> sites;
    u64 others{};
    std::chrono::steady_clock::time_point last_report{};
};

UnmappedAccessLog& UnmappedAccesses() {
    static UnmappedAccessLog log;
    return log;
}

} // Anonymous namespace

static bool TrapFastpathEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_TRAP_FASTPATH");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

static bool SysSlotEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SYS_SLOT");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

static inline bool AddressSpaceContains(const Common::PageTable& table, const Common::ProcessAddress addr, const std::size_t size) {
    const Common::ProcessAddress max_addr = 1ULL << table.GetAddressSpaceBits();
    return addr + size >= addr && addr + size <= max_addr;
}

// Implementation class used to keep the specifics of the memory subsystem hidden
// from outside classes. This also allows modification to the internals of the memory
// subsystem without needing to rebuild all files that make use of the memory interface.
struct Memory::Impl {
    explicit Impl(Core::System& system_) : system{system_} {}

    void SetCurrentPageTable(Kernel::KProcess& process) {
        current_page_table = &process.GetPageTable().GetImpl();

        if (process.IsApplication() && Settings::IsFastmemEnabled()) {
            current_page_table->fastmem_arena = system.DeviceMemory().buffer.VirtualBasePointer();
        } else {
            current_page_table->fastmem_arena = nullptr;
        }

#ifdef __ANDROID__
        heap_tracker.emplace(system.DeviceMemory().buffer);
        host_buffer = std::addressof(*heap_tracker);
#else
        host_buffer = std::addressof(system.DeviceMemory().buffer);
#endif

#ifdef HAS_HYPERVISOR
        if (process.IsApplication()) {
            hv_vm = nullptr;
            if (Settings::IsHypervisorEnabled()) {
                hv_vm = HvVm::Create(system);
                if (hv_vm != nullptr) {
                    // Hypervisor translates through its own 4K stage-1 guest tables; skip host
                    // mach_vm_protect calls on the unused Dynarmic fastmem arena.
                    current_page_table->fastmem_arena = nullptr;
                } else {
                    LOG_WARNING(HW_Memory, "Hypervisor unavailable, falling back to Dynarmic");
                }
            }
            if (hv_vm == nullptr) {
                HvVm::Deactivate();
            }
        }
#endif
        RefreshJitReadTrapState();
    }

    bool HasHypervisor() const noexcept {
#ifdef HAS_HYPERVISOR
        return hv_vm != nullptr;
#else
        return false;
#endif
    }

    void MapMemoryRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                         Common::PhysicalAddress target, Common::MemoryPermission perms,
                         bool separate_heap) {
        ASSERT_MSG((size & YUZU_PAGEMASK) == 0, "non-page aligned size: {:016X}", size);
        ASSERT_MSG((base & YUZU_PAGEMASK) == 0, "non-page aligned base: {:016X}", GetInteger(base));
        ASSERT_MSG(target >= DramMemoryMap::Base, "Out of bounds target: {:016X}",
                   GetInteger(target));
        MapPages(page_table, base / YUZU_PAGESIZE, size / YUZU_PAGESIZE, target,
                 Common::PageType::Memory);

        if (current_page_table->fastmem_arena) {
            host_buffer->Map(GetInteger(base), GetInteger(target) - DramMemoryMap::Base, size, perms, separate_heap);
        }
#ifdef HAS_HYPERVISOR
        if (hv_vm) {
            hv_vm->Map(GetInteger(base), GetInteger(target) - DramMemoryMap::Base, size, perms);
        }
#endif
    }

    void UnmapRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                     bool separate_heap) {
        ASSERT_MSG((size & YUZU_PAGEMASK) == 0, "non-page aligned size: {:016X}", size);
        ASSERT_MSG((base & YUZU_PAGEMASK) == 0, "non-page aligned base: {:016X}", GetInteger(base));
        MapPages(page_table, base / YUZU_PAGESIZE, size / YUZU_PAGESIZE, 0,
                 Common::PageType::Unmapped);

        if (current_page_table->fastmem_arena) {
            host_buffer->Unmap(GetInteger(base), size, separate_heap);
        }
#ifdef HAS_HYPERVISOR
        if (hv_vm) {
            hv_vm->Unmap(GetInteger(base), size);
        }
#endif
    }

    void ProtectRegion(Common::PageTable& page_table, VAddr vaddr, u64 size,
                       Common::MemoryPermission perms) {
        ASSERT_MSG((size & YUZU_PAGEMASK) == 0, "non-page aligned size: {:016X}", size);
        ASSERT_MSG((vaddr & YUZU_PAGEMASK) == 0, "non-page aligned base: {:016X}", vaddr);
        static_cast<void>(page_table);

        if (!current_page_table->fastmem_arena && !HasHypervisor()) {
            return;
        }

        const auto protect = [&](u64 begin, u64 bytes) {
            if (current_page_table->fastmem_arena) {
                host_buffer->Protect(begin, bytes, perms);
            }
#ifdef HAS_HYPERVISOR
            if (hv_vm) {
                hv_vm->Protect(begin, bytes, perms);
            }
#endif
        };

        u64 protect_bytes = 0;
        u64 protect_begin = 0;

        const auto& entries = std::as_const(current_page_table->entries);
        for (u64 addr = vaddr; addr < vaddr + size; addr += YUZU_PAGESIZE) {
            const Common::PageType page_type = entries[addr >> YUZU_PAGEBITS].Type();
            switch (page_type) {
            case Common::PageType::RasterizerCachedMemory:
                if (protect_bytes > 0) {
                    protect(protect_begin, protect_bytes);
                    protect_bytes = 0;
                }
                break;
            default:
                if (protect_bytes == 0) {
                    protect_begin = addr;
                }
                protect_bytes += YUZU_PAGESIZE;
                break;
            }
        }

        if (protect_bytes > 0) {
            protect(protect_begin, protect_bytes);
        }
    }

    struct u128_le {
        u64_le lo;
        u64_le hi;
        constexpr explicit operator u64() const noexcept {
            return lo;
        }
    };

    template <std::size_t N>
    void ReadCrossPageBytes(const Common::ProcessAddress vaddr, u8 (&dst)[N]) noexcept {
        const u64 addr0 = GetInteger(vaddr);
        const std::size_t first =
            static_cast<std::size_t>(YUZU_PAGESIZE - (addr0 & YUZU_PAGEMASK));
        const std::size_t second = N - first;
        const u64 addr1 = addr0 + first;
        const bool check_traps = ReadTrapsEnabled();

        u8* const ptr0 = GetPointerImpl(
            addr0,
            [addr0, first]() {
                if (UnmappedAccesses().Note(static_cast<u32>(first * 8), addr0)) {
                    LOG_ERROR(HW_Memory, "Unmapped Read{} @ {:#016x}", first * 8, addr0);
                }
                ++UnmappedReadFlag();
            },
            [&](u8* host_ptr) {
                if (!check_traps || ReadTrapArmed(addr0 & 0xffffffffffffULL, first)) {
                    HandleRasterizerDownload(addr0, first, host_ptr);
                }
            });
        if (ptr0 != nullptr) {
            std::memcpy(dst, ptr0, first);
        } else {
            std::memset(dst, 0, first);
        }

        u8* const ptr1 = GetPointerImpl(
            addr1,
            [addr1, second]() {
                if (UnmappedAccesses().Note(static_cast<u32>(second * 8), addr1)) {
                    LOG_ERROR(HW_Memory, "Unmapped Read{} @ {:#016x}", second * 8, addr1);
                }
                ++UnmappedReadFlag();
            },
            [&](u8* host_ptr) {
                if (!check_traps || ReadTrapArmed(addr1 & 0xffffffffffffULL, second)) {
                    HandleRasterizerDownload(addr1, second, host_ptr);
                }
            });
        if (ptr1 != nullptr) {
            std::memcpy(dst + first, ptr1, second);
        } else {
            std::memset(dst + first, 0, second);
        }
    }

    template <std::size_t N>
    void WriteCrossPageBytes(const Common::ProcessAddress vaddr, const u8 (&src)[N],
                             const std::size_t known_core = ~std::size_t{0}) noexcept {
        const u64 addr0 = GetInteger(vaddr);
        const std::size_t first =
            static_cast<std::size_t>(YUZU_PAGESIZE - (addr0 & YUZU_PAGEMASK));
        const std::size_t second = N - first;
        const u64 addr1 = addr0 + first;

        if (u8* const ptr0 = GetPointerImpl(
                addr0,
                [addr0, first]() {
                    if (UnmappedAccesses().Note(0x100U | static_cast<u32>(first * 8), addr0)) {
                        LOG_ERROR(HW_Memory, "Unmapped Write{} @ {:#016x}", first * 8, addr0);
                    }
                },
                [&](u8* host_ptr) {
                    HandleRasterizerWrite(addr0, first, host_ptr, known_core);
                });
            ptr0 != nullptr) {
            std::memcpy(ptr0, src, first);
        }

        if (u8* const ptr1 = GetPointerImpl(
                addr1,
                [addr1, second]() {
                    if (UnmappedAccesses().Note(0x100U | static_cast<u32>(second * 8), addr1)) {
                        LOG_ERROR(HW_Memory, "Unmapped Write{} @ {:#016x}", second * 8, addr1);
                    }
                },
                [&](u8* host_ptr) {
                    HandleRasterizerWrite(addr1, second, host_ptr, known_core);
                });
            ptr1 != nullptr) {
            std::memcpy(ptr1, src + first, second);
        }
    }

    u8 Read8(const Common::ProcessAddress addr) {
        return Read<u8>(addr);
    }

    u16 Read16(const Common::ProcessAddress addr) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u16) <= YUZU_PAGESIZE) {
            return Read<u16_le>(addr);
        }
        u8 bytes[sizeof(u16_le)]{};
        ReadCrossPageBytes(addr, bytes);
        u16_le val{};
        std::memcpy(&val, bytes, sizeof(val));
        return val;
    }

    u32 Read32(const Common::ProcessAddress addr) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u32) <= YUZU_PAGESIZE) {
            return Read<u32_le>(addr);
        }
        u8 bytes[sizeof(u32_le)]{};
        ReadCrossPageBytes(addr, bytes);
        u32_le val{};
        std::memcpy(&val, bytes, sizeof(val));
        return val;
    }

    bool Read32Acquire(const Common::ProcessAddress addr, u32& value) {
        value = 0;
        if (!current_page_table || (addr & 3) != 0) {
            return false;
        }
        auto* pointer = GetPointerImpl(GetInteger(addr), [] {}, [&](u8* host_ptr) {
            HandleRasterizerDownload(GetInteger(addr), sizeof(u32), host_ptr);
        });
        if (!pointer) {
            return false;
        }
        u32_le actual{};
        static_cast<void>(Common::AtomicCompareAndSwap(reinterpret_cast<u32_le*>(pointer),
                                                     u32_le{0}, u32_le{0}, actual));
        value = actual;
        return true;
    }

    u64 Read64(const Common::ProcessAddress addr) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u64) <= YUZU_PAGESIZE) {
            return Read<u64_le>(addr);
        }
        u8 bytes[sizeof(u64_le)]{};
        ReadCrossPageBytes(addr, bytes);
        u64_le val{};
        std::memcpy(&val, bytes, sizeof(val));
        return val;
    }

    u128 Read128(const Common::ProcessAddress addr) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u128_le) <= YUZU_PAGESIZE) {
            const u128_le val = Read<u128_le>(addr);
            return {val.lo, val.hi};
        }
        u8 bytes[sizeof(u128_le)]{};
        ReadCrossPageBytes(addr, bytes);
        u128_le val{};
        std::memcpy(&val, bytes, sizeof(val));
        return {val.lo, val.hi};
    }

    void Write8(const Common::ProcessAddress addr, const u8 data) {
        Write<u8>(addr, data);
    }

    void Write16(const Common::ProcessAddress addr, const u16 data) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u16) <= YUZU_PAGESIZE) {
            Write<u16_le>(addr, data);
        } else {
            const u16_le val{data};
            u8 bytes[sizeof(val)]{};
            std::memcpy(bytes, &val, sizeof(val));
            WriteCrossPageBytes(addr, bytes);
        }
    }

    void Write32(const Common::ProcessAddress addr, const u32 data) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u32) <= YUZU_PAGESIZE) {
            Write<u32_le>(addr, data);
        } else {
            const u32_le val{data};
            u8 bytes[sizeof(val)]{};
            std::memcpy(bytes, &val, sizeof(val));
            WriteCrossPageBytes(addr, bytes);
        }
    }

    void Write64(const Common::ProcessAddress addr, const u64 data) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u64) <= YUZU_PAGESIZE) {
            Write<u64_le>(addr, data);
        } else {
            const u64_le val{data};
            u8 bytes[sizeof(val)]{};
            std::memcpy(bytes, &val, sizeof(val));
            WriteCrossPageBytes(addr, bytes);
        }
    }

    void Write128(const Common::ProcessAddress addr, const u128 data) {
        if ((GetInteger(addr) & YUZU_PAGEMASK) + sizeof(u128_le) <= YUZU_PAGESIZE) {
            Write<u128_le>(addr, u128_le{data[0], data[1]});
        } else {
            const u128_le val{data[0], data[1]};
            u8 bytes[sizeof(val)]{};
            std::memcpy(bytes, &val, sizeof(val));
            WriteCrossPageBytes(addr, bytes);
        }
    }

    bool WriteExclusive8(const Common::ProcessAddress addr, const u8 data, const u8 expected) {
        return WriteExclusive<u8>(addr, data, expected);
    }

    bool WriteExclusive16(const Common::ProcessAddress addr, const u16 data, const u16 expected) {
        return WriteExclusive<u16_le>(addr, data, expected);
    }

    bool WriteExclusive32(const Common::ProcessAddress addr, const u32 data, const u32 expected) {
        return WriteExclusive<u32_le>(addr, data, expected);
    }

    bool WriteExclusive64(const Common::ProcessAddress addr, const u64 data, const u64 expected) {
        return WriteExclusive<u64_le>(addr, data, expected);
    }

    std::string ReadCString(Common::ProcessAddress vaddr, std::size_t max_length) {
        std::string string;
        string.reserve(max_length);
        for (std::size_t i = 0; i < max_length; ++i) {
            const char c = Read<s8>(vaddr);
            if (c == '\0') {
                break;
            }
            string.push_back(c);
            ++vaddr;
        }
        string.shrink_to_fit();
        return string;
    }

    template<typename F1, typename F2, typename F3>
    inline bool WalkBlock(const Common::ProcessAddress addr, const std::size_t size, F1&& on_unmapped, F2&& on_memory, F3&& on_rasterizer) {
        if (size == 0) {
            return true;
        }
        std::size_t offset = 0;
        if (!AddressSpaceContains(*current_page_table, addr, size)) {
            on_unmapped(offset, size, addr);
            return false;
        }
        std::size_t remaining_size = size;
        std::size_t page_index = addr >> YUZU_PAGEBITS;
        std::size_t page_offset = addr & YUZU_PAGEMASK;
        bool user_accessible = true;

        std::size_t mem_run_offset = 0;
        std::size_t mem_run_size = 0;
        u8* mem_run_ptr = nullptr;
        const auto flush_mem_run = [&] {
            if (mem_run_size != 0) {
                on_memory(mem_run_offset, mem_run_size, mem_run_ptr);
                mem_run_size = 0;
            }
        };
        const auto append_mem_run = [&](std::size_t chunk_offset, std::size_t chunk_size,
                                        u8* ptr) {
            if (mem_run_size != 0 && mem_run_ptr != nullptr &&
                mem_run_ptr + mem_run_size == ptr) {
                mem_run_size += chunk_size;
            } else {
                flush_mem_run();
                mem_run_offset = chunk_offset;
                mem_run_size = chunk_size;
                mem_run_ptr = ptr;
            }
        };

        const std::size_t first_page = page_index;
        const std::size_t last_page = (GetInteger(addr) + size - 1) >> YUZU_PAGEBITS;
        const auto& const_entries = std::as_const(current_page_table->entries);
        const auto* const span = const_entries.CommittedSpan(first_page, last_page);
        while (remaining_size != 0) {
            const std::size_t copy_amount = (std::min)(std::size_t(YUZU_PAGESIZE) - page_offset, remaining_size);
            const auto current_vaddr = u64((page_index << YUZU_PAGEBITS) + page_offset);
            const auto& entry =
                span != nullptr ? span[page_index - first_page] : const_entries[page_index];
            const auto [pointer, type, _] = entry.PointerTypeBlock(true);
            if (type == Common::PageType::Unmapped || pointer == 0) {
                flush_mem_run();
                user_accessible = false;
                on_unmapped(offset, copy_amount, current_vaddr);
            } else {
                u8* const host_ptr = reinterpret_cast<u8*>(pointer + current_vaddr);
                switch (type) {
                case Common::PageType::Memory:
                case Common::PageType::DebugMemory:
                    append_mem_run(offset, copy_amount, host_ptr);
                    break;
                case Common::PageType::RasterizerCachedMemory:
                    on_rasterizer(current_vaddr, offset, copy_amount, host_ptr);
                    append_mem_run(offset, copy_amount, host_ptr);
                    break;
                default:
                    UNREACHABLE();
                }
            }
            page_index++;
            page_offset = 0;
            offset += copy_amount;
            remaining_size -= copy_amount;
        }
        flush_mem_run();
        return user_accessible;
    }

    bool ReadBlockImpl(const Common::ProcessAddress addr, void* buffer, const std::size_t size, bool unsafe) {
        if (size == 0) {
            return true;
        }
        const u64 raw_addr = GetInteger(addr);
        if (!AddressSpaceContains(*current_page_table, addr, size)) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})",
                      raw_addr, raw_addr, size);
            std::memset(buffer, 0, size);
            return false;
        }
        const bool check_traps = !unsafe && ReadTrapsEnabled();
        const bool need_download_check =
            !unsafe && (!check_traps || ReadTrapArmed(raw_addr & 0xffffffffffffULL, size));
        if (!need_download_check) {
            const std::size_t first_page = raw_addr >> YUZU_PAGEBITS;
            const std::size_t last_page = (raw_addr + size - 1) >> YUZU_PAGEBITS;
            if (const auto* const entries =
                    std::as_const(current_page_table->entries).CommittedSpan(first_page, last_page)) {
                const auto [first_ptr, first_type, _] = entries[0].PointerTypeBlock(true);
                if (first_type != Common::PageType::Unmapped && first_ptr != 0) {
                    const std::size_t page_count = last_page - first_page + 1;
                    std::size_t i = 1;
                    for (; i < page_count; ++i) {
                        const auto [ptr, type, block] = entries[i].PointerTypeBlock(true);
                        static_cast<void>(block);
                        if (type == Common::PageType::Unmapped || ptr != first_ptr) {
                            break;
                        }
                    }
                    if (i == page_count) {
                        std::memcpy(buffer, reinterpret_cast<const u8*>(first_ptr + raw_addr), size);
                        return true;
                    }
                }
            }
        }
        return WalkBlock(addr, size,
        [addr, size, &buffer](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(addr), size);
            std::memset(reinterpret_cast<u8*>(buffer) + offset, 0, copy_amount);
        },
        [&](const std::size_t offset, const std::size_t copy_amount, const u8* const ptr) {
            std::memcpy(reinterpret_cast<u8*>(buffer) + offset, ptr, copy_amount);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, const u8* const ptr) {
            static_cast<void>(offset);
            if (!need_download_check) {
                return;
            }
            const u64 raw_vaddr = GetInteger(current_vaddr);
            if (!check_traps || ReadTrapArmed(raw_vaddr & 0xffffffffffffULL, copy_amount)) {
                HandleRasterizerDownload(raw_vaddr, copy_amount, ptr);
            }
        });
    }

    /// Returns a host pointer to [addr, addr + size) if the whole range is backed by one linear
    /// host range, nullptr otherwise. Every page of such a range stores the same offset; block ids
    /// can't prove it, since equal ids at the ends say nothing about a page remapped in between.
    [[nodiscard]] u8* GetSpan(const VAddr addr, const std::size_t size) const noexcept {
        if (!AddressSpaceContains(*current_page_table, addr, size)) {
            return nullptr;
        }
        const std::size_t first_page = addr >> YUZU_PAGEBITS;
        const std::size_t last_page = (addr + (size != 0 ? size - 1 : 0)) >> YUZU_PAGEBITS;
        const auto* const entries =
            current_page_table->entries.CommittedSpan(first_page, last_page);
        if (entries == nullptr) {
            return nullptr;
        }
        const uintptr_t offset = entries[0].Pointer(true);
        if (offset == 0) {
            return nullptr;
        }
        const std::size_t page_count = last_page - first_page + 1;
        for (std::size_t i = 1; i < page_count; ++i) {
            if (entries[i].Pointer(true) != offset) {
                return nullptr;
            }
        }
        return reinterpret_cast<u8*>(offset + addr);
    }

    bool WriteBlockImpl(const Common::ProcessAddress addr, const void* buffer, const std::size_t size, bool unsafe) {
        if (size == 0) {
            return true;
        }
        const u64 raw_addr = GetInteger(addr);
        if (!AddressSpaceContains(*current_page_table, addr, size)) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})",
                      raw_addr, raw_addr, size);
            return false;
        }
        const std::size_t first_page = raw_addr >> YUZU_PAGEBITS;
        const std::size_t last_page = (raw_addr + size - 1) >> YUZU_PAGEBITS;
        if (const auto* const entries =
                std::as_const(current_page_table->entries).CommittedSpan(first_page, last_page)) {
            const auto [first_ptr, first_type, _] = entries[0].PointerTypeBlock(true);
            const bool first_ok =
                first_type != Common::PageType::Unmapped && first_ptr != 0 &&
                (unsafe || first_type != Common::PageType::RasterizerCachedMemory);
            if (first_ok) {
                const std::size_t page_count = last_page - first_page + 1;
                std::size_t i = 1;
                for (; i < page_count; ++i) {
                    const auto [ptr, type, block] = entries[i].PointerTypeBlock(true);
                    static_cast<void>(block);
                    if (type == Common::PageType::Unmapped || ptr != first_ptr ||
                        (!unsafe && type == Common::PageType::RasterizerCachedMemory)) {
                        break;
                    }
                }
                if (i == page_count) {
                    std::memcpy(reinterpret_cast<u8*>(first_ptr + raw_addr), buffer, size);
                    return true;
                }
            }
        }
        return WalkBlock(addr, size,
        [addr, size](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            static_cast<void>(offset);
            static_cast<void>(copy_amount);
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(addr), size);
        },
        [&](const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            std::memcpy(ptr, reinterpret_cast<u8 const*>(buffer) + offset, copy_amount);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            static_cast<void>(offset);
            if (!unsafe) {
                HandleRasterizerWrite(GetInteger(current_vaddr), copy_amount, ptr);
            }
        });
    }

    template <typename OnCachedPage>
    [[nodiscard]] u8* ResolveSafeSpan(const Common::ProcessAddress target_addr,
                                      const std::size_t size, OnCachedPage&& on_cached,
                                      const bool inspect_cached = true) {
        const u64 addr = GetInteger(target_addr) & 0xffffffffffffULL;
        if (size == 0 || !AddressSpaceContains(*current_page_table, addr, size)) {
            return nullptr;
        }
        const std::size_t first_page = addr >> YUZU_PAGEBITS;
        const std::size_t last_page = (addr + size - 1) >> YUZU_PAGEBITS;
        const auto* const entries =
            current_page_table->entries.CommittedSpan(first_page, last_page);
        if (entries == nullptr) {
            return nullptr;
        }

        const std::size_t page_count = last_page - first_page + 1;
        uintptr_t first_delta = 0;
        bool has_cached = false;
        for (std::size_t i = 0; i < page_count; ++i) {
            const auto [pointer, type, _] = entries[i].PointerTypeBlock(true);
            if (pointer == 0) {
                return nullptr;
            }
            if (type == Common::PageType::RasterizerCachedMemory) {
                has_cached = true;
            } else if (type != Common::PageType::Memory) {
                return nullptr;
            }
            if (i == 0) {
                first_delta = pointer;
            } else if (pointer != first_delta) {
                return nullptr;
            }
        }

        u8* const base_ptr = reinterpret_cast<u8*>(first_delta + addr);
        if (has_cached && inspect_cached) {
            const u64 end_addr = addr + size;
            for (std::size_t i = 0; i < page_count; ++i) {
                if (entries[i].Type() != Common::PageType::RasterizerCachedMemory) {
                    continue;
                }
                const u64 page = first_page + i;
                const u64 p_begin = (std::max<u64>)(addr, page << YUZU_PAGEBITS);
                const u64 p_end = (std::min<u64>)(end_addr, (page + 1) << YUZU_PAGEBITS);
                const std::size_t chunk = static_cast<std::size_t>(p_end - p_begin);
                on_cached(p_begin, chunk, base_ptr + (p_begin - addr));
            }
        }
        return base_ptr;
    }

    [[nodiscard]] const u8* GetSafeReadSpan(const Common::ProcessAddress src_addr,
                                            const std::size_t size) {
        const u64 masked_addr = GetInteger(src_addr) & 0xffffffffffffULL;
        if (size == 0 || !AddressSpaceContains(*current_page_table, masked_addr, size)) {
            return nullptr;
        }
        const bool check_traps = ReadTrapsEnabled();
        const bool inspect_cached = !check_traps || ReadTrapArmed(masked_addr, size);
        return ResolveSafeSpan(
            src_addr, size,
            [&](u64 p_begin, std::size_t chunk, u8* host_ptr) {
                if (!check_traps || ReadTrapArmed(p_begin, chunk)) {
                    HandleRasterizerDownload(p_begin, chunk, host_ptr);
                }
            },
            inspect_cached);
    }

    [[nodiscard]] u8* GetSafeWriteSpan(const Common::ProcessAddress dest_addr,
                                       const std::size_t size) {
        return ResolveSafeSpan(dest_addr, size,
                               [&](u64 p_begin, std::size_t chunk, u8* host_ptr) {
                                   HandleRasterizerWrite(p_begin, chunk, host_ptr);
                               });
    }

    bool ZeroBlock(const Common::ProcessAddress addr, const std::size_t size) {
        return WalkBlock(addr, size,
        [addr, size](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            static_cast<void>(offset);
            static_cast<void>(copy_amount);
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(addr), size);
        },
        [=](const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            static_cast<void>(offset);
            std::memset(ptr, 0, copy_amount);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            static_cast<void>(offset);
            HandleRasterizerWrite(GetInteger(current_vaddr), copy_amount, ptr);
        });
    }

    bool CopyBlock(Common::ProcessAddress dest_addr, Common::ProcessAddress src_addr, const std::size_t size) {
        const bool check_traps = ReadTrapsEnabled();
        return WalkBlock(src_addr, size,
        [&](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
            LOG_ERROR(HW_Memory, "Unmapped @ {:#016x} (start address = {:#016x}, size = {})", GetInteger(current_vaddr), GetInteger(src_addr), size);
            ZeroBlock(dest_addr + offset, copy_amount);
        },
        [&](const std::size_t offset, const std::size_t copy_amount, const u8* const ptr) {
            WriteBlockImpl(dest_addr + offset, ptr, copy_amount, false);
        },
        [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const ptr) {
            static_cast<void>(offset);
            const u64 raw_vaddr = GetInteger(current_vaddr);
            if (!check_traps || ReadTrapArmed(raw_vaddr & 0xffffffffffffULL, copy_amount)) {
                HandleRasterizerDownload(raw_vaddr, copy_amount, ptr);
            }
        });
    }

    template <typename Callback>
    Result PerformCacheOperation(Common::ProcessAddress dest_addr, std::size_t size, Callback&& cb) {
        bool had_unmapped = false;
        WalkBlock(dest_addr, size,
            [&](const std::size_t offset, const std::size_t copy_amount, const Common::ProcessAddress current_vaddr) {
                LOG_ERROR(HW_Memory, "Unmapped @ {:#018X}", GetInteger(current_vaddr));
                had_unmapped = true;
            },
            [](const std::size_t offset, const std::size_t copy_amount, u8* const host_ptr) {},
            [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t copy_amount, u8* const host_ptr) { cb(current_vaddr, offset, copy_amount); }
        );
        return had_unmapped ? Kernel::ResultInvalidCurrentMemory : ResultSuccess;
    }

    Result InvalidateDataCache(Common::ProcessAddress dest_addr, std::size_t size) {
        auto on_rasterizer = [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t block_size) {
            // dc ivac: Invalidate to point of coherency
            // GPU flush -> CPU invalidate
            HandleRasterizerDownload(GetInteger(current_vaddr), block_size);
        };
        return PerformCacheOperation(dest_addr, size, on_rasterizer);
    }

    Result StoreDataCache(Common::ProcessAddress dest_addr, std::size_t size) {
        auto on_rasterizer = [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t block_size) {
            // dc cvac: Store to point of coherency
            // CPU flush -> GPU invalidate
            HandleRasterizerWrite(GetInteger(current_vaddr), block_size);
        };
        return PerformCacheOperation(dest_addr, size, on_rasterizer);
    }

    Result FlushDataCache(Common::ProcessAddress dest_addr, std::size_t size) {
        auto on_rasterizer = [&](const Common::ProcessAddress current_vaddr, const std::size_t offset, const std::size_t block_size) {
            // dc civac: Store to point of coherency, and invalidate from cache
            // CPU flush -> GPU invalidate
            HandleRasterizerWrite(GetInteger(current_vaddr), block_size);
        };
        return PerformCacheOperation(dest_addr, size, on_rasterizer);
    }

    void MarkRegionDebug(u64 vaddr, u64 size, bool debug) {
        if (vaddr == 0 || !AddressSpaceContains(*current_page_table, vaddr, size)) {
            return;
        }

        if (current_page_table->fastmem_arena) {
            const auto perm{debug ? Common::MemoryPermission{}
                                  : Common::MemoryPermission::ReadWrite};
            host_buffer->Protect(vaddr, size, perm);
        }

        // Iterate over a contiguous CPU address space, marking/unmarking the region.
        // The region is at a granularity of CPU pages.

        const u64 num_pages = ((vaddr + size - 1) >> YUZU_PAGEBITS) - (vaddr >> YUZU_PAGEBITS) + 1;

        current_page_table->entries.CommitRegion(vaddr >> YUZU_PAGEBITS, (vaddr >> YUZU_PAGEBITS) + num_pages);
        for (u64 i = 0; i < num_pages; ++i, vaddr += YUZU_PAGESIZE) {
            auto& entry = current_page_table->entries.GetUnchecked(vaddr >> YUZU_PAGEBITS);
            const auto [pointer, type, block] = entry.PointerTypeBlock(true);
            if (debug) {
                // Switch page type to debug if now debug
                switch (type) {
                case Common::PageType::Unmapped:
                    ASSERT(false && "Attempted to mark unmapped pages as debug");
                    break;
                case Common::PageType::RasterizerCachedMemory:
                case Common::PageType::DebugMemory:
                    // Page is already marked.
                    break;
                case Common::PageType::Memory:
                    entry.MarkDebug(pointer, block);
                    break;
                default:
                    UNREACHABLE();
                }
            } else {
                // Switch page type to non-debug if now non-debug
                switch (type) {
                case Common::PageType::Unmapped:
                    ASSERT(false && "Attempted to mark unmapped pages as non-debug");
                    break;
                case Common::PageType::RasterizerCachedMemory:
                case Common::PageType::Memory:
                    // Don't mess with already non-debug or rasterizer memory.
                    break;
                case Common::PageType::DebugMemory: {
                    entry.Store(false, Common::PageType::Memory, block, pointer);
                    break;
                }
                default:
                    UNREACHABLE();
                }
            }
        }
    }

    // EDEN_NX2_MME_DMA_STATS: how often pages are protected again after an upload and
    // unprotected by a CPU write, and how many distinct pages that is.
    struct ProtectStats {
        std::mutex mutex;
        std::array<u64, 2> calls{};
        std::array<u64, 2> pages{};
        std::unordered_map<u64, u32> protected_pages;
        std::chrono::steady_clock::time_point last_log{std::chrono::steady_clock::now()};

        static ProtectStats* Get() {
            static const bool enabled = std::getenv("EDEN_NX2_MME_DMA_STATS") != nullptr;
            static ProtectStats stats;
            return enabled ? &stats : nullptr;
        }

        void Note(u64 vaddr, u64 size, bool cached) {
            std::scoped_lock lk{mutex};
            const u64 first = vaddr >> YUZU_PAGEBITS;
            const u64 last = (vaddr + size - 1) >> YUZU_PAGEBITS;
            ++calls[cached];
            pages[cached] += last - first + 1;
            if (cached) {
                for (u64 page = first; page <= last; ++page) {
                    ++protected_pages[page];
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - last_log < std::chrono::seconds{5}) {
                return;
            }
            last_log = now;
            size_t hot = 0;
            for (const auto& [page, count] : protected_pages) {
                hot += count >= 100 ? 1 : 0;
            }
            LOG_INFO(HW_Memory,
                     "Page protection (5 s): protect {} calls {} pages ({} distinct, {} protected "
                     "100+ times), unprotect {} calls {} pages",
                     calls[1], pages[1], protected_pages.size(), hot, calls[0], pages[0]);
            calls = {};
            pages = {};
            protected_pages.clear();
        }
    };

    void RasterizerMarkRegionCached(u64 vaddr, u64 size, bool cached) {
        if (vaddr == 0 || size == 0 || !AddressSpaceContains(*current_page_table, vaddr, size)) {
            return;
        }
        if (auto* stats = ProtectStats::Get()) {
            stats->Note(vaddr, size, cached);
        }

        if (current_page_table->fastmem_arena || HasHypervisor()) {
            Common::MemoryPermission perm{};
            // EDEN_NX2_NO_REACTIVE=1 drops the read traps on Switch 2 programs. Opt-in: without
            // them a CPU read of GPU-written memory can see stale data.
            static const bool nx2_no_reactive = [] {
                const char* value = std::getenv("EDEN_NX2_NO_REACTIVE");
                return value != nullptr && value[0] == '1';
            }();
            const bool reactive_flushing =
                Settings::values.use_reactive_flushing.GetValue() &&
                !(nx2_no_reactive && Settings::CurrentProgramIsSwitch2());
            if (!reactive_flushing || !cached) {
                perm |= Common::MemoryPermission::Read;
            }
            if (!cached) {
                perm |= Common::MemoryPermission::Write;
            }
            // EDEN_NX2_PROTECT_STATS=1: how many protection changes the tracker makes, and what
            // they cost, per 5 s (diagnostic).
            static const bool stats_enabled = std::getenv("EDEN_NX2_PROTECT_STATS") != nullptr;
            const bool spikes = VideoCommon::Nx2Spikes::Counting();
            const auto stats_start = stats_enabled || spikes
                                         ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
            if (current_page_table->fastmem_arena) {
                host_buffer->Protect(vaddr, size, perm);
            }
#ifdef HAS_HYPERVISOR
            const bool batch_hv = hv_vm != nullptr && cached && perm != Common::MemoryPermission{};
            if (batch_hv) {
                hv_vm->BeginProtectBatch();
            }
            if (hv_vm) {
                hv_vm->ProtectAccess(vaddr, size, perm);
            }
#endif
            if (spikes) {
                VideoCommon::Nx2Spikes::Note(VideoCommon::Nx2Spikes::Event::ProtectChange,
                                             std::chrono::steady_clock::now() - stats_start,
                                             size >> YUZU_PAGEBITS);
            }
            if (stats_enabled) [[unlikely]] {
                static std::atomic<u64> calls[2]{}, pages[2]{}, nanos[2]{};
                static std::atomic<u64> window_start{0};
                const auto now = std::chrono::steady_clock::now();
                const u64 now_ns = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch())
                        .count());
                const int side = cached ? 1 : 0;
                calls[side].fetch_add(1, std::memory_order_relaxed);
                pages[side].fetch_add(size >> YUZU_PAGEBITS, std::memory_order_relaxed);
                nanos[side].fetch_add(
                    static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(now - stats_start)
                            .count()),
                    std::memory_order_relaxed);
                u64 started = window_start.load(std::memory_order_relaxed);
                if (started == 0) {
                    window_start.compare_exchange_strong(started, now_ns);
                } else if (now_ns - started > 5'000'000'000ULL &&
                           window_start.compare_exchange_strong(started, now_ns)) {
                    const double seconds = static_cast<double>(now_ns - started) / 1e9;
                    LOG_INFO(HW_Memory,
                             "nx2 protect: to read-only (tracked) {:.0f}/s ({:.0f} pages/s, {:.0f} "
                             "ms/s), to writable (untracked) {:.0f}/s ({:.0f} pages/s, {:.0f} ms/s)",
                             static_cast<double>(calls[1].exchange(0)) / seconds,
                             static_cast<double>(pages[1].exchange(0)) / seconds,
                             static_cast<double>(nanos[1].exchange(0)) / 1e6 / seconds,
                             static_cast<double>(calls[0].exchange(0)) / seconds,
                             static_cast<double>(pages[0].exchange(0)) / seconds,
                             static_cast<double>(nanos[0].exchange(0)) / 1e6 / seconds);
                }
            }
            if (cached && perm != Common::MemoryPermission{}) {
                ProtectArmedPages(vaddr, size);
            }
#ifdef HAS_HYPERVISOR
            if (batch_hv) {
                hv_vm->EndProtectBatch();
            }
#endif
        }

        // Iterate over a contiguous CPU address space, which corresponds to the specified GPU
        // address space, marking the region as un/cached. The region is marked un/cached at a
        // granularity of CPU pages, hence why we iterate on a CPU page basis (note: GPU page size
        // is different). This assumes the specified GPU address region is contiguous as well.

        const std::size_t first_page = vaddr >> YUZU_PAGEBITS;
        const std::size_t last_page = (vaddr + size - 1) >> YUZU_PAGEBITS;
        const std::size_t num_pages = last_page - first_page + 1;
        const auto& const_entries = std::as_const(current_page_table->entries);
        const auto* const span = const_entries.CommittedSpan(first_page, last_page);
        if (cached) {
            if (span != nullptr) {
                for (std::size_t i = 0; i < num_pages; ++i) {
                    const Common::PageType page_type = span[i].Type();
                    if (page_type == Common::PageType::Memory ||
                        page_type == Common::PageType::DebugMemory) {
                        current_page_table->entries.GetUnchecked(first_page + i)
                            .MarkRasterizerCached();
                    }
                }
            } else {
                for (std::size_t i = 0; i < num_pages; ++i) {
                    const std::size_t page = first_page + i;
                    const Common::PageType page_type = const_entries[page].Type();
                    if (page_type == Common::PageType::Memory ||
                        page_type == Common::PageType::DebugMemory) {
                        current_page_table->entries.GetAndFault(page).MarkRasterizerCached();
                    }
                }
            }
        } else {
            const auto uncache_entry = [&](std::size_t page,
                                           Common::PageTable::PageEntryData::Data raw,
                                           bool committed) {
                if (static_cast<Common::PageType>(raw.type) !=
                    Common::PageType::RasterizerCachedMemory) {
                    return;
                }
                const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw, true);
                const u16 block = Common::PageTable::PageEntryData::ExtractBlock(raw);
                auto& entry = committed ? current_page_table->entries.GetUnchecked(page)
                                        : current_page_table->entries.GetAndFault(page);
                if (ptr == 0) {
                    // Underlying VMA was unmapped while updating the pagetable; leave blank.
                    entry.Store(false, Common::PageType::Unmapped, block, 0);
                } else {
                    entry.Store(false, Common::PageType::Memory, block, ptr);
                }
            };
            if (span != nullptr) {
                for (std::size_t i = 0; i < num_pages; ++i) {
                    uncache_entry(first_page + i, span[i].Raw(), true);
                }
            } else {
                for (std::size_t i = 0; i < num_pages; ++i) {
                    const std::size_t page = first_page + i;
                    uncache_entry(page, const_entries[page].Raw(), false);
                }
            }
        }
    }

    /// Read traps replace reactive flushing's blanket read protection of every cached page: only
    /// pages the GPU writes are protected, and each one only until its first read
    /// (EDEN_NX2_READ_TRAPS=0 turns them off). Under NCE every fault is taken as a write, so they
    /// are not used there.
    bool ReadTrapsEnabled() const {
        static const bool disabled = [] {
            const char* value = std::getenv("EDEN_NX2_READ_TRAPS");
            return value != nullptr && value[0] == '0';
        }();
        return !disabled && (current_page_table->fastmem_arena || HasHypervisor()) &&
               !Settings::values.use_reactive_flushing.GetValue() &&
               Settings::values.cpu_backend.GetValue() != Settings::CpuBackend::Nce &&
               !Settings::IsCachedInterpreterEnabled();
    }

    void RefreshJitReadTrapState() noexcept {
        if (current_page_table == nullptr || !ReadTrapsEnabled()) {
            jit_read_trap_state.store(0, std::memory_order_release);
            return;
        }
        const auto* const bits = armed_trap_pages.words.load(std::memory_order_acquire);
        jit_read_trap_state.store(bits != nullptr ? reinterpret_cast<std::uintptr_t>(bits)
                                                  : std::uintptr_t{1},
                                  std::memory_order_release);
    }

    void ProtectReadTrapPages(u64 vaddr, u64 size, Common::MemoryPermission perm) {
        if (current_page_table->fastmem_arena) {
            host_buffer->Protect(vaddr, size, perm);
        }
#ifdef HAS_HYPERVISOR
        if (hv_vm) {
            hv_vm->ProtectAccess(vaddr, size, perm);
        }
#endif
    }

    void RasterizerArmReadTraps(u64 vaddr, u64 size) {
        if (vaddr == 0 || size == 0 || !AddressSpaceContains(*current_page_table, vaddr, size)) {
            return;
        }
        if (!ReadTrapsEnabled()) {
            read_trap_generation.fetch_add(1, std::memory_order_release);
            jit_read_trap_state.store(0, std::memory_order_release);
            return;
        }
        const u64 page_end = ((vaddr + size - 1) >> YUZU_PAGEBITS) + 1;
        std::scoped_lock lk{read_trap_mutex};
#ifdef HAS_HYPERVISOR
        if (hv_vm) {
            hv_vm->BeginProtectBatch();
        }
#endif
        if (armed_trap_pages.words.load(std::memory_order_relaxed) == nullptr) {
            auto* const bits = armed_trap_pages.Words();
            jit_read_trap_state.store(reinterpret_cast<std::uintptr_t>(bits),
                                      std::memory_order_release);
        } else if (jit_read_trap_state.load(std::memory_order_relaxed) <= 1) {
            RefreshJitReadTrapState();
        }
        u64 run_begin = 0;
        u64 run_pages = 0;
        const auto protect_run = [&] {
            if (run_pages > 0) {
                read_trap_generation.fetch_add(1, std::memory_order_release);
                ProtectReadTrapPages(run_begin << YUZU_PAGEBITS, run_pages << YUZU_PAGEBITS,
                                     Common::MemoryPermission{});
                run_pages = 0;
            }
        };
        for (u64 page = vaddr >> YUZU_PAGEBITS; page != page_end; ++page) {
            const bool cached = std::as_const(current_page_table->entries)[page].Type() ==
                                Common::PageType::RasterizerCachedMemory;
            if (!cached || !armed_read_traps.insert(page).second) {
                protect_run();
                continue;
            }
            armed_trap_pages.Set(page);
            armed_read_trap_count.fetch_add(1, std::memory_order_relaxed);
            if (run_pages == 0) {
                run_begin = page;
            }
            ++run_pages;
        }
        protect_run();
#ifdef HAS_HYPERVISOR
        if (hv_vm) {
            hv_vm->EndProtectBatch();
        }
#endif
    }

    bool ReadTrapArmed(u64 vaddr, u64 size) {
        if (armed_read_trap_count.load(std::memory_order_relaxed) == 0) {
            return false;
        }
        return armed_trap_pages.Any(vaddr, size);
    }

    /// Takes read permission away again from armed pages that were just cached again.
    void ProtectArmedPages(u64 vaddr, u64 size) {
        if (armed_read_trap_count.load(std::memory_order_relaxed) == 0 ||
            !armed_trap_pages.Any(vaddr, size)) {
            return;
        }
        std::scoped_lock lk{read_trap_mutex};
        const u64 page_end = ((vaddr + size - 1) >> YUZU_PAGEBITS) + 1;
        u64 run_begin = 0;
        u64 run_pages = 0;
        for (auto it = armed_read_traps.lower_bound(vaddr >> YUZU_PAGEBITS);
             it != armed_read_traps.end() && *it < page_end; ++it) {
            if (run_pages > 0 && *it == run_begin + run_pages) {
                ++run_pages;
            } else {
                if (run_pages > 0) {
                    ProtectReadTrapPages(run_begin << YUZU_PAGEBITS, run_pages << YUZU_PAGEBITS,
                                         Common::MemoryPermission{});
                }
                run_begin = *it;
                run_pages = 1;
            }
        }
        if (run_pages > 0) {
            ProtectReadTrapPages(run_begin << YUZU_PAGEBITS, run_pages << YUZU_PAGEBITS,
                                 Common::MemoryPermission{});
        }
    }

    /// A slow-path read of a cached page: its trap, if any, has done its job.
    void DisarmReadTraps(u64 vaddr, size_t size) {
        if (armed_read_trap_count.load(std::memory_order_relaxed) == 0 ||
            !armed_trap_pages.Any(vaddr, size)) {
            return;
        }
        std::scoped_lock lk{read_trap_mutex};
        const u64 page_end = ((vaddr + size - 1) >> YUZU_PAGEBITS) + 1;
        u64 run_begin = 0;
        u64 run_pages = 0;
        const auto unprotect_run = [&] {
            if (run_pages > 0) {
                ProtectReadTrapPages(run_begin << YUZU_PAGEBITS, run_pages << YUZU_PAGEBITS,
                                     Common::MemoryPermission::Read);
                run_pages = 0;
            }
        };
        auto it = armed_read_traps.lower_bound(vaddr >> YUZU_PAGEBITS);
        while (it != armed_read_traps.end() && *it < page_end) {
            const u64 page = *it;
            // Still cached (this read came through the rasterizer path): writes stay trapped.
            if (run_pages > 0 && page == run_begin + run_pages) {
                ++run_pages;
            } else {
                unprotect_run();
                run_begin = page;
                run_pages = 1;
            }
            armed_trap_pages.Clear(page);
            it = armed_read_traps.erase(it);
            armed_read_trap_count.fetch_sub(1, std::memory_order_relaxed);
            VideoCommon::Nx2Spikes::Note(VideoCommon::Nx2Spikes::Event::ReadTrapHit);
        }
        unprotect_run();
    }

    /**
     * Maps a region of pages as a specific type.
     *
     * @param page_table The page table to use to perform the mapping.
     * @param base       The base address to begin mapping at.
     * @param size       The total size of the range in bytes.
     * @param target     The target address to begin mapping from.
     * @param type       The page type to map the memory as.
     */
    void MapPages(Common::PageTable& page_table, Common::ProcessAddress base_address, u64 size,
                  Common::PhysicalAddress target, Common::PageType type) {
        auto base = GetInteger(base_address);

        LOG_DEBUG(HW_Memory, "Mapping {:016X} onto {:016X}-{:016X}", GetInteger(target),
                  base * YUZU_PAGESIZE, (base + size) * YUZU_PAGESIZE);

        const auto end = base + size;
        ASSERT_MSG(end <= page_table.entries.size(), "out of range mapping at {:016X}", base + page_table.entries.size());

        if (!target) {
            ASSERT_MSG(type != Common::PageType::Memory,
                       "Mapping memory page without a pointer @ {:016x}", base * YUZU_PAGESIZE);

            last_mapped_table = nullptr;
            page_table.entries.ZeroRegion(base, end);
        } else if (base != end) {
            const u64 first_host_ptr =
                reinterpret_cast<u64>(system.DeviceMemory().GetPointer<u8>(target)) -
                (base << YUZU_PAGEBITS);
            u16 current_block = Common::PageTable::PageEntryData::INVALID_BLOCK_ID;
            if (last_mapped_table == &page_table && base == last_mapped_end &&
                first_host_ptr == last_mapped_host_ptr &&
                last_mapped_block != Common::PageTable::PageEntryData::INVALID_BLOCK_ID) {
                current_block = last_mapped_block;
                last_mapped_end = end;
            } else {
                u64 mapping_id = block_count.load(std::memory_order_relaxed);
                constexpr u64 exhausted_count =
                    u64{Common::PageTable::PageEntryData::INVALID_BLOCK_ID} + 1;
                while (mapping_id < exhausted_count &&
                       !block_count.compare_exchange_weak(mapping_id, mapping_id + 1,
                                                          std::memory_order_relaxed)) {
                }
                current_block =
                    Common::PageTable::PageEntryData::BlockIdForMapping(mapping_id);
                last_mapped_table = &page_table;
                last_mapped_end = end;
                last_mapped_host_ptr = first_host_ptr;
                last_mapped_block = current_block;
            }

            page_table.entries.CommitRegion(base, end);
            ASSERT_MSG(Common::PageTable::PageEntryData::ExtractPointer(
                           Common::PageTable::PageEntryData::Data{false, type, current_block,
                                                                  first_host_ptr}) != 0,
                       "memory mapping base yield a nullptr within the table");
            for (u64 page = base; page != end; ++page) {
                page_table.entries.GetUnchecked(page).Store(false, type, current_block,
                                                            first_host_ptr);
            }
        }
    }

    template<typename F, typename G>
    [[nodiscard]] u8* GetPointerImpl(u64 vaddr, F&& on_unmapped, G&& on_rasterizer) const {
        // AARCH64 masks the upper 16 bit of all memory accesses
        vaddr &= 0xffffffffffffULL;
        if (AddressSpaceContains(*current_page_table, vaddr, 1)) [[likely]] {
            // Avoid adding any extra logic to this fast-path block
            const auto raw = current_page_table->entries[vaddr >> YUZU_PAGEBITS].Raw();
            if (auto pointer = Common::PageTable::PageEntryData::ExtractPointer(raw); pointer) [[likely]] {
                return reinterpret_cast<u8*>(pointer + vaddr);
            } else {
                const auto marked_ptr = [&]() -> u8* {
                    const uintptr_t paddr =
                        Common::PageTable::PageEntryData::ExtractPointer(raw, true);
                    return paddr != 0 ? reinterpret_cast<u8*>(paddr + vaddr) : nullptr;
                };
                switch (static_cast<Common::PageType>(raw.type)) {
                case Common::PageType::Memory:
                    ASSERT_MSG(false, "Mapped memory page without a pointer @ {:#016x}", vaddr);
                    return nullptr;
                case Common::PageType::DebugMemory:
                    return marked_ptr();
                case Common::PageType::RasterizerCachedMemory: {
                    u8* const host_ptr{marked_ptr()};
                    if constexpr (std::is_invocable_v<G&, u8*>) {
                        on_rasterizer(host_ptr);
                    } else {
                        on_rasterizer();
                    }
                    return host_ptr;
                }
                case Common::PageType::Unmapped: [[unlikely]] {
                    on_unmapped();
                    return nullptr;
                }
                default:
                    UNREACHABLE();
                }
                return nullptr;
            }
        } else {
            on_unmapped();
            return nullptr;
        }
    }

    [[nodiscard]] u8* GetPointer(const Common::ProcessAddress vaddr) const {
        return GetPointerImpl(
            GetInteger(vaddr),
            [vaddr]() {
                LOG_ERROR(HW_Memory, "Unmapped GetPointer @ {:#016x}", GetInteger(vaddr));
            },
            []() {});
    }

    [[nodiscard]] u8* GetPointerSilent(const Common::ProcessAddress vaddr) const {
        return GetPointerImpl(
            GetInteger(vaddr), []() {}, []() {});
    }

    /// @brief Reads a particular data type out of memory at the given virtual address.
    /// @param vaddr The virtual address to read the data type from.
    /// @tparam T The data type to read out of memory.
    /// @returns The instance of T read from the specified virtual address.
    template <typename T>
    inline T Read(Common::ProcessAddress vaddr) noexcept requires(std::is_trivially_copyable_v<T>) {
        const u64 addr = GetInteger(vaddr);
        if (auto const ptr = GetPointerImpl(addr, [addr]() {
            if (UnmappedAccesses().Note(sizeof(T) * 8, addr)) {
                LOG_ERROR(HW_Memory, "Unmapped Read{} @ {:#016x}", sizeof(T) * 8, addr);
            }
            ++UnmappedReadFlag();
        }, [&](u8* host_ptr) {
            if (!ReadTrapsEnabled() || ReadTrapArmed(addr & 0xffffffffffffULL, sizeof(T))) {
                HandleRasterizerDownload(addr, sizeof(T), host_ptr);
            }
        }); ptr) [[likely]] {
            // It may be tempting to rewrite this particular section to use "reinterpret_cast";
            // afterall, it's trivially copyable so surely it can be copied ov- Alignment.
            // Remember, alignment. memcpy() will deal with all the alignment extremely fast.
            T result{};
            std::memcpy(&result, ptr, sizeof(T));
            return result;
        }
        return T{};
    }

    /// @brief Writes a particular data type to memory at the given virtual address.
    /// @param vaddr The virtual address to write the data type to.
    /// @tparam T The data type to write to memory.
    template <typename T>
    inline void Write(Common::ProcessAddress vaddr, const T data) noexcept requires(std::is_trivially_copyable_v<T>) {
        const u64 addr = GetInteger(vaddr);
        if (auto const ptr = GetPointerImpl(addr, [addr, data]() {
            if (UnmappedAccesses().Note(0x100 | (sizeof(T) * 8), addr)) {
                LOG_ERROR(HW_Memory, "Unmapped Write{} @ {:#016x} = {:#016x}", sizeof(T) * 8, addr,
                          u64(data));
            }
        }, [&](u8* host_ptr) { HandleRasterizerWrite(addr, sizeof(T), host_ptr); }); ptr) [[likely]]
            std::memcpy(ptr, &data, sizeof(T));
    }

    template <typename T>
    bool WriteExclusive(Common::ProcessAddress vaddr, const T data, const T expected) {
        u8* const ptr = GetPointerImpl(
            GetInteger(vaddr),
            [vaddr, data]() {
                if (UnmappedAccesses().Note(0x200 | (sizeof(T) * 8), GetInteger(vaddr))) {
                    LOG_ERROR(HW_Memory, "Unmapped WriteExclusive{} @ {:#016x} = {:#016x}",
                              sizeof(T) * 8, GetInteger(vaddr), static_cast<u64>(data));
                }
            },
            [&](u8* host_ptr) { HandleRasterizerWrite(GetInteger(vaddr), sizeof(T), host_ptr); });
        if (ptr) {
            return Common::AtomicCompareAndSwap(reinterpret_cast<T*>(ptr), data, expected);
        }
        return true;
    }

    bool WriteExclusive128(Common::ProcessAddress vaddr, const u128 data, const u128 expected) {
        u8* const ptr = GetPointerImpl(
            GetInteger(vaddr),
            [vaddr, data]() {
                LOG_ERROR(HW_Memory, "Unmapped WriteExclusive128 @ {:#016x} = {:#016x}{:016X}",
                          GetInteger(vaddr), static_cast<u64>(data[1]), static_cast<u64>(data[0]));
            },
            [&](u8* host_ptr) { HandleRasterizerWrite(GetInteger(vaddr), sizeof(u128), host_ptr); });
        if (ptr) {
            return Common::AtomicCompareAndSwap(reinterpret_cast<u64*>(ptr), data, expected);
        }
        return true;
    }

    /// A rasterizer-cached page whose host pointer is outside the emulated physical memory has no
    /// device address to flush or invalidate (its lookup used to read far out of bounds).
    void ReportForeignPointer(VAddr v_address, const u8* p) const {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1, std::memory_order_relaxed) >= 16) {
            return;
        }
        const auto raw =
            current_page_table->entries[(v_address & 0xffffffffffffULL) >> YUZU_PAGEBITS].Raw();
        LOG_ERROR(HW_Memory,
                  "Rasterizer access at {:#x}: host pointer {} is outside the emulated physical "
                  "memory at {} (page type {}, block {})",
                  v_address, fmt::ptr(p),
                  fmt::ptr(system.DeviceMemory().buffer.BackingBasePointer()),
                  static_cast<u32>(raw.type), static_cast<u32>(raw.block));
    }

    void HandleRasterizerDownload(VAddr v_address, size_t size, const u8* known_pointer = nullptr) {
        const bool fast = TrapFastpathEnabled();
        const u8* const p = fast && known_pointer != nullptr
                                ? known_pointer
                                : GetPointerImpl(
                                      v_address, []() {}, []() {});
        if (!gpu_device_memory) [[unlikely]] {
            gpu_device_memory = &system.Host1x().MemoryManager();
        }
        if (!gpu_device_memory->IsPhysicalPointer(p)) [[unlikely]] {
            ReportForeignPointer(v_address, p);
            return;
        }
        const size_t core = system.GetCurrentHostThreadID();
        if (core >= Core::Hardware::NUM_CPU_CORES) {
            // A service thread has no slot in the arrays below, and it does not use a slot of a
            // core. It keeps no area, so it asks on each read.
            if (static std::atomic_flag logged; !logged.test_and_set(std::memory_order_relaxed)) {
                LOG_INFO(HW_Memory, "Read of cached guest memory from a thread that is not a core");
            }
            Common::ScratchBuffer<u32> scratch;
            gpu_device_memory->ApplyOpOnPointer(p, scratch, [&](DAddr address) {
                DisarmReadTraps(v_address, size);
                (void)system.GPU().OnCPURead(address, size);
            });
            return;
        }
        auto& areas = rasterizer_read_areas[core];
        if (const u64 generation = read_trap_generation.load(std::memory_order_acquire);
            read_area_generations[core] != generation) {
            areas = {};
            read_area_generations[core] = generation;
        }
        const u64 masked_vaddr = v_address & 0xffffffffffffULL;
        const bool trap_armed = ReadTrapsEnabled() && ReadTrapArmed(masked_vaddr, size);
        if (fast && trap_armed && size <= YUZU_PAGESIZE) {
            if (const auto dpage0 = gpu_device_memory->TryGetSingleDevicePage(p)) {
                auto& gpu = system.GPU();
                if (gpu.IsCleanReadTrapPage(DAddr{*dpage0} << YUZU_PAGEBITS)) {
                    const u64 vpage0 = masked_vaddr >> YUZU_PAGEBITS;
                    const u64 as_pages =
                        (u64{1} << current_page_table->GetAddressSpaceBits()) >> YUZU_PAGEBITS;
                    const u64 max_pages =
                        (std::min<u64>)(16ULL - (u64{*dpage0} & 15ULL), as_pages - vpage0);
                    const auto* const entries = current_page_table->entries.CommittedSpan(
                        static_cast<size_t>(vpage0),
                        static_cast<size_t>(vpage0 + max_pages - 1));
                    const uintptr_t base_offset =
                        entries != nullptr ? entries[0].Pointer(true) : uintptr_t{0};
                    const u8* const page0_host = reinterpret_cast<const u8*>(
                        reinterpret_cast<uintptr_t>(p) & ~uintptr_t{YUZU_PAGEMASK});
                    u64 run_pages = 1;
                    if (base_offset != 0) {
                        for (; run_pages < max_pages; ++run_pages) {
                            const auto [ptr, type, _] =
                                entries[run_pages].PointerTypeBlock(true);
                            if (ptr != base_offset ||
                                type != Common::PageType::RasterizerCachedMemory) {
                                break;
                            }
                            const u64 next_vaddr = (vpage0 + run_pages) << YUZU_PAGEBITS;
                            if (!ReadTrapArmed(next_vaddr, YUZU_PAGESIZE)) {
                                break;
                            }
                            const auto next_dpage = gpu_device_memory->TryGetSingleDevicePage(
                                page0_host + run_pages * YUZU_PAGESIZE);
                            if (!next_dpage || u64{*next_dpage} != u64{*dpage0} + run_pages) {
                                break;
                            }
                            if (!gpu.IsCleanReadTrapPage(DAddr{*next_dpage} << YUZU_PAGEBITS)) {
                                break;
                            }
                        }
                    }
                    const u64 v_begin = vpage0 << YUZU_PAGEBITS;
                    const DAddr d_begin = DAddr{*dpage0} << YUZU_PAGEBITS;
                    const u64 run_bytes = run_pages << YUZU_PAGEBITS;
                    DisarmReadTraps(v_begin, static_cast<size_t>(run_bytes));
                    if (const auto fast_area = gpu.TryFastCleanRead(d_begin, run_bytes)) {
                        areas.Insert(*fast_area, true);
                        return;
                    }
                    areas.Insert(gpu.OnCPURead(d_begin | DAddr{masked_vaddr & YUZU_PAGEMASK}, size),
                                 true);
                    return;
                }
            }
        }
        gpu_device_memory->ApplyOpOnPointer(p, scratch_buffers[core], [&](DAddr address) {
            const DAddr end_address = address + size;
            if (!trap_armed && areas.Contains(address, end_address)) {
                return;
            }
            DisarmReadTraps(v_address, size);
            areas.Insert(system.GPU().OnCPURead(address, size), fast);
        });
    }

    std::pair<size_t, bool> WriteSlot(std::size_t known_core = ~std::size_t{0}) const {
        constexpr size_t num_cores = Core::Hardware::NUM_CPU_CORES;
        const size_t thread_id =
            known_core < num_cores ? known_core : system.GetCurrentHostThreadID();
        if (SysSlotEnabled() && gpu_dirty_managers.size() > num_cores) {
            return {(std::min<std::size_t>)(thread_id, num_cores), thread_id >= num_cores};
        }
        const size_t last_core = (std::min<std::size_t>)(thread_id, num_cores - 1);
        return {last_core, last_core == num_cores - 1};
    }

    void HandleRasterizerWrite(VAddr v_address, size_t size, const u8* known_pointer = nullptr,
                               std::size_t known_core = ~std::size_t{0}) {
        if (VideoCommon::Nx2Spikes::Counting()) {
            VideoCommon::Nx2Spikes::Note(VideoCommon::Nx2Spikes::Event::CpuWriteSlowPath, {}, size);
        }
        const bool fast = TrapFastpathEnabled();
        const u8* const p = fast && known_pointer != nullptr
                                ? known_pointer
                                : GetPointerImpl(
                                      v_address, []() {}, []() {});
        const std::pair<size_t, bool> slot = WriteSlot(known_core);
        const size_t core = slot.first;
        const bool shared = slot.second;
        if (!gpu_device_memory) {
            gpu_device_memory = &system.Host1x().MemoryManager();
        }
        if (shared) {
            sys_core_guard.lock();
        }
        SCOPE_EXIT {
            if (shared) {
                sys_core_guard.unlock();
            }
        };
        const auto collect_device_write = [&](DAddr address) {
            VideoCommon::Nx2Hazard::NoteGuestWrite(address, size);
            if (gpu_dirty_managers[core].Collect(address, size)) {
                const DAddr granule_begin = address & ~DAddr{63};
                const u64 granule_end = (address + size + 63ULL) & ~u64{63};
                VideoCore::DownloadLanding::Await(granule_begin, granule_end - granule_begin);
            }
            g_collected_write_pages.Mark(address, size);
        };
        const uintptr_t host_address = reinterpret_cast<uintptr_t>(p);
        const u64 page_offset = host_address & YUZU_PAGEMASK;
        const bool single_page = size != 0 && page_offset + size <= YUZU_PAGESIZE;
        u64 g_mask = 0;
        if (single_page) {
            const u64 first_g = page_offset >> 6;
            const u64 last_g = (page_offset + size - 1) >> 6;
            const u64 count_g = last_g - first_g + 1;
            g_mask = count_g >= 64 ? ~u64{0} : (((u64{1} << count_g) - 1) << first_g);
        }
        if (fast && single_page) {
            const uintptr_t host_page = host_address >> YUZU_PAGEBITS;
            auto& entry = write_caches[core].Slot(host_page);
            const u64 mapping_generation = gpu_device_memory->MappingGeneration();
            if (entry.host_page == host_page && entry.map_gen == mapping_generation) {
                const DAddr address =
                    (DAddr{entry.device_page} << YUZU_PAGEBITS) | DAddr{page_offset};
                const u64 epoch = Core::g_write_answer_epoch.load(std::memory_order_relaxed);
                if (!entry.collect || entry.answer_epoch != epoch) {
                    const bool do_collection =
                        address != 0 && system.GPU().OnCPUWrite(address, size);
                    entry.collect = do_collection;
                    entry.answer_epoch = epoch;
                    entry.collected_granules = 0;
                    entry.gather_epoch = 0;
                    if (!do_collection) {
                        return;
                    }
                }
                if (!shared && !VideoCommon::Nx2Hazard::Enabled()) {
                    const u64 gather_ep = gpu_dirty_managers[core].GatherEpoch();
                    if ((gather_ep & 1) == 0 && entry.gather_epoch == gather_ep &&
                        (entry.collected_granules & g_mask) == g_mask) {
                        g_collected_write_pages.Mark(address, 1);
                        return;
                    }
                }
                const u64 gather_before = !shared ? gpu_dirty_managers[core].GatherEpoch() : 0;
                collect_device_write(address);
                const u64 gather_after = !shared ? gpu_dirty_managers[core].GatherEpoch() : 0;
                if (!shared && !VideoCommon::Nx2Hazard::Enabled() && (gather_before & 1) == 0 &&
                    gather_before == gather_after) {
                    if (entry.gather_epoch == gather_before) {
                        entry.collected_granules |= g_mask;
                    } else {
                        entry.gather_epoch = gather_before;
                        entry.collected_granules = g_mask;
                    }
                } else {
                    entry.collected_granules = 0;
                    entry.gather_epoch = 0;
                }
                return;
            }
        }
        if (!gpu_device_memory->IsPhysicalPointer(p)) {
            ReportForeignPointer(v_address, p);
            return;
        }
        const auto on_device_address = [&](DAddr address, bool& out_collected, u64& out_epoch) {
            out_epoch = Core::g_write_answer_epoch.load(std::memory_order_relaxed);
            out_collected = (address != 0 && size != 0) && system.GPU().OnCPUWrite(address, size);
            if (out_collected) {
                collect_device_write(address);
            }
        };
        if (!fast || !single_page) {
            gpu_device_memory->ApplyOpOnPointer(p, scratch_buffers[core], [&](DAddr address) {
                bool collected = false;
                u64 epoch = 0;
                on_device_address(address, collected, epoch);
            });
            return;
        }
        const uintptr_t host_page = host_address >> YUZU_PAGEBITS;
        auto& entry = write_caches[core].Slot(host_page);
        const u64 mapping_generation = gpu_device_memory->MappingGeneration();
        u32 count = 0;
        DAddr last_address = 0;
        bool last_collected = false;
        u64 last_epoch = 0;
        const u64 gather_before = !shared ? gpu_dirty_managers[core].GatherEpoch() : 0;
        gpu_device_memory->ApplyOpOnPointer(p, scratch_buffers[core], [&](DAddr address) {
            ++count;
            last_address = address;
            on_device_address(address, last_collected, last_epoch);
        });
        const u64 gather_after = !shared ? gpu_dirty_managers[core].GatherEpoch() : 0;
        if (count == 1) {
            const bool seed = !shared && last_collected && !VideoCommon::Nx2Hazard::Enabled() &&
                              (gather_before & 1) == 0 && gather_before == gather_after;
            entry = CachedWriteEntry{
                .host_page = host_page,
                .device_page = last_address >> YUZU_PAGEBITS,
                .collected_granules = seed ? g_mask : 0,
                .gather_epoch = seed ? gather_before : 0,
                .map_gen = mapping_generation,
                .answer_epoch = last_epoch,
                .collect = last_collected,
            };
        } else {
            entry = CachedWriteEntry{};
        }
    }

    void InvalidateGPUMemory(u8* p, size_t size) {
        const std::pair<size_t, bool> slot = WriteSlot();
        const size_t core = slot.first;
        const bool shared = slot.second;
        if (!gpu_device_memory) [[unlikely]] {
            gpu_device_memory = &system.Host1x().MemoryManager();
        }
        if (shared) [[unlikely]] {
            sys_core_guard.lock();
        }
        SCOPE_EXIT {
            if (shared) [[unlikely]] {
                sys_core_guard.unlock();
            }
        };
        auto& gpu = system.GPU();
        gpu_device_memory->ApplyOpOnPointer(
            p, scratch_buffers[core], [&](DAddr address) { gpu.InvalidateRegion(address, size, true); });
    }

    Core::System& system;
    Tegra::MaxwellDeviceMemoryManager* gpu_device_memory{};
    Common::PageTable* current_page_table = nullptr;

    struct ReadAreaCache {
        std::array<VideoCore::RasterizerDownloadArea, 4> areas{};
        size_t next{};

        bool Contains(DAddr begin, DAddr end) const {
            for (const auto& area : areas) {
                if (area.start_address <= begin && end <= area.end_address) {
                    return true;
                }
            }
            return false;
        }

        void Insert(const VideoCore::RasterizerDownloadArea& area, bool many) {
            if (!many) {
                areas[0] = area;
                return;
            }
            areas[next] = area;
            next = (next + 1) % areas.size();
        }
    };

    struct CachedWriteEntry {
        uintptr_t host_page{~uintptr_t{0}};
        DAddr device_page{};
        u64 collected_granules{};
        u64 gather_epoch{};
        u64 map_gen{};
        u64 answer_epoch{};
        bool collect{};
    };

    struct WritePageCache {
        static constexpr size_t NumSlots = 256;
        std::array<CachedWriteEntry, NumSlots> slots{};

        CachedWriteEntry& Slot(uintptr_t host_page) noexcept {
            return slots[(host_page ^ (host_page >> 8)) & (NumSlots - 1)];
        }

        const CachedWriteEntry& Slot(uintptr_t host_page) const noexcept {
            return slots[(host_page ^ (host_page >> 8)) & (NumSlots - 1)];
        }
    };

    struct ArmedTrapBitmap {
        static constexpr u64 MaxAddressBits = 39;
        static constexpr u64 NumPages = 1ULL << (MaxAddressBits - YUZU_PAGEBITS);
        static constexpr u64 NumWords = NumPages / 64;
        // 16 MiB, zeroed, made at the first armed trap: every service process has a Memory too,
        // and ~60 zeroed bitmaps were ~1 GB of resident host memory.
        std::atomic<std::atomic<u64>*> words{};
        std::unique_ptr<std::atomic<u64>[]> storage;

        std::atomic<u64>* Words() noexcept {
            std::atomic<u64>* current = words.load(std::memory_order_acquire);
            if (current != nullptr) {
                return current;
            }
            auto fresh = std::make_unique<std::atomic<u64>[]>(NumWords);
            if (words.compare_exchange_strong(current, fresh.get(), std::memory_order_acq_rel)) {
                storage = std::move(fresh);
                return storage.get();
            }
            return current;
        }

        bool Any(u64 vaddr, u64 size) const noexcept {
            const std::atomic<u64>* const bits = words.load(std::memory_order_acquire);
            if (size == 0 || bits == nullptr) {
                return false;
            }
            const u64 first = vaddr >> YUZU_PAGEBITS;
            const u64 last = (vaddr + size - 1) >> YUZU_PAGEBITS;
            if (last >= NumPages) [[unlikely]] {
                return false;
            }
            for (u64 page = first; page <= last;) {
                const u64 bit = page & 63;
                const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
                const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
                if (bits[page >> 6].load(std::memory_order_acquire) & mask) {
                    return true;
                }
                page += count;
            }
            return false;
        }

        bool Set(u64 page) noexcept {
            if (page >= NumPages) [[unlikely]] {
                return false;
            }
            const u64 mask = 1ULL << (page & 63);
            auto& word = Words()[page >> 6];
            if (word.load(std::memory_order_relaxed) & mask) {
                return false;
            }
            return (word.fetch_or(mask, std::memory_order_release) & mask) == 0;
        }

        void Clear(u64 page) noexcept {
            std::atomic<u64>* const bits = words.load(std::memory_order_acquire);
            if (page >= NumPages || bits == nullptr) [[unlikely]] {
                return;
            }
            bits[page >> 6].fetch_and(~(1ULL << (page & 63)), std::memory_order_release);
        }
    };

    std::array<ReadAreaCache, Core::Hardware::NUM_CPU_CORES> rasterizer_read_areas{};
    std::array<u64, Core::Hardware::NUM_CPU_CORES> read_area_generations{};
    struct ScopedGenBump {
        std::atomic<u64>& generation;
        std::atomic<u32>& writers;

        ScopedGenBump(std::atomic<u64>& gen, std::atomic<u32>& active) noexcept
            : generation{gen}, writers{active} {
            writers.fetch_add(1, std::memory_order_acq_rel);
            generation.fetch_add(1, std::memory_order_acq_rel);
        }

        ~ScopedGenBump() {
            generation.fetch_add(1, std::memory_order_acq_rel);
            writers.fetch_sub(1, std::memory_order_release);
        }

        ScopedGenBump(const ScopedGenBump&) = delete;
        ScopedGenBump& operator=(const ScopedGenBump&) = delete;
    };

    std::atomic<u64> read_trap_generation{};
    std::atomic<u64> fast_read_generation{1};
    std::atomic<u64> fast_write_generation{1};
    std::atomic<u64> rasterizer_write_generation{1};
    std::atomic<u32> read_gen_writers{};
    std::atomic<u32> write_gen_writers{};
    std::atomic<u32> rasterizer_write_gen_writers{};
    std::atomic<std::uintptr_t> jit_read_trap_state{0};
    ArmedTrapBitmap armed_trap_pages{};
    std::set<u64> armed_read_traps;
    std::atomic<size_t> armed_read_trap_count{};
    std::mutex read_trap_mutex;
    std::array<WritePageCache, Core::Hardware::NUM_CPU_CORES + 1> write_caches{};
    std::array<Common::ScratchBuffer<u32>, Core::Hardware::NUM_CPU_CORES + 1> scratch_buffers{};
    std::span<Core::GPUDirtyMemoryManager> gpu_dirty_managers;
    std::unique_ptr<std::array<HotCoreState, MaxHotCores>> hot_cores;
    std::mutex hot_cores_mutex;
    std::mutex sys_core_guard;
#ifdef HAS_HYPERVISOR
    HvVm* hv_vm{};
#endif
#ifdef __ANDROID__
    std::optional<Common::HeapTracker> heap_tracker;
    Common::HeapTracker* host_buffer{};
#else
    Common::HostMemory* host_buffer{};
#endif
    std::atomic<u64> block_count = 0;
    const Common::PageTable* last_mapped_table = nullptr;
    size_t last_mapped_end = 0;
    u64 last_mapped_host_ptr = 0;
    u16 last_mapped_block = Common::PageTable::PageEntryData::INVALID_BLOCK_ID;
};

static_assert(sizeof(Memory::HotWritePage) == 128);
static_assert(offsetof(Memory::HotWritePage, vaddr_page) == 0);
static_assert(offsetof(Memory::HotWritePage, rasterizer_cached) == 64);
static_assert(Memory::HotCoreState::WriteSlots == 64);

Memory::Memory(Core::System& system_) : system{system_} {
    Reset();
}

Memory::~Memory() = default;

void Memory::Reset() {
    hot_cores_ptr = nullptr;
    impl = std::make_unique<Impl>(system);
    fast_read_gen_ptr = &impl->fast_read_generation;
    fast_write_gen_ptr = &impl->fast_write_generation;
    rasterizer_write_gen_ptr = &impl->rasterizer_write_generation;
    jit_read_trap_state_ptr = &impl->jit_read_trap_state;
}

void Memory::PrepareHotCores() {
    std::scoped_lock lk{impl->hot_cores_mutex};
    if (!impl->hot_cores) {
        impl->hot_cores = std::make_unique<std::array<HotCoreState, MaxHotCores>>();
    }
    for (std::size_t i = 0; i < MaxHotCores; ++i) {
        (*impl->hot_cores)[i].gather_epoch_ptr =
            i < impl->gpu_dirty_managers.size() ? impl->gpu_dirty_managers[i].GatherEpochPtr()
                                                : nullptr;
        (*impl->hot_cores)[i].dev_mem = impl->gpu_device_memory;
    }
    hot_cores_ptr = impl->hot_cores->data();
}

void Memory::SetCurrentPageTable(Kernel::KProcess& process) {
    Impl::ScopedGenBump r_bump{impl->fast_read_generation, impl->read_gen_writers};
    Impl::ScopedGenBump w_bump{impl->fast_write_generation, impl->write_gen_writers};
    Impl::ScopedGenBump rw_bump{impl->rasterizer_write_generation,
                                impl->rasterizer_write_gen_writers};
    impl->SetCurrentPageTable(process);
}

void Memory::MapMemoryRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                             Common::PhysicalAddress target, Common::MemoryPermission perms,
                             bool separate_heap) {
    Impl::ScopedGenBump r_bump{impl->fast_read_generation, impl->read_gen_writers};
    Impl::ScopedGenBump w_bump{impl->fast_write_generation, impl->write_gen_writers};
    Impl::ScopedGenBump rw_bump{impl->rasterizer_write_generation,
                                impl->rasterizer_write_gen_writers};
    impl->MapMemoryRegion(page_table, base, size, target, perms, separate_heap);
}

void Memory::UnmapRegion(Common::PageTable& page_table, Common::ProcessAddress base, u64 size,
                         bool separate_heap) {
    Impl::ScopedGenBump r_bump{impl->fast_read_generation, impl->read_gen_writers};
    Impl::ScopedGenBump w_bump{impl->fast_write_generation, impl->write_gen_writers};
    Impl::ScopedGenBump rw_bump{impl->rasterizer_write_generation,
                                impl->rasterizer_write_gen_writers};
    impl->UnmapRegion(page_table, base, size, separate_heap);
}

void Memory::ProtectRegion(Common::PageTable& page_table, Common::ProcessAddress vaddr, u64 size,
                           Common::MemoryPermission perms) {
    impl->ProtectRegion(page_table, GetInteger(vaddr), size, perms);
}

bool Memory::IsValidVirtualAddress(const Common::ProcessAddress vaddr) const {
    const auto& page_table = *impl->current_page_table;
    const size_t page = vaddr >> YUZU_PAGEBITS;
    if (page >= page_table.entries.size()) {
        return false;
    }
    const auto [pointer, type, _] = page_table.entries[page].PointerTypeBlock();
    return pointer != 0 || type == Common::PageType::RasterizerCachedMemory ||
           type == Common::PageType::DebugMemory;
}

bool Memory::IsValidVirtualAddressRange(Common::ProcessAddress base, u64 size) const {
    Common::ProcessAddress end = base + size;
    if (end < base) {
        return false;
    }
    Common::ProcessAddress page = Common::AlignDown(GetInteger(base), YUZU_PAGESIZE);

    for (; page < end; page += YUZU_PAGESIZE) {
        if (!IsValidVirtualAddress(page)) {
            return false;
        }
    }

    return true;
}

u8* Memory::GetPointer(Common::ProcessAddress vaddr) {
    return impl->GetPointer(vaddr);
}

u8* Memory::GetPointerSilent(Common::ProcessAddress vaddr) {
    return impl->GetPointerSilent(vaddr);
}

const u8* Memory::GetPointer(Common::ProcessAddress vaddr) const {
    return impl->GetPointer(vaddr);
}

u8 Memory::Read8(const Common::ProcessAddress addr) {
    return impl->Read8(addr);
}

u16 Memory::Read16(const Common::ProcessAddress addr) {
    return impl->Read16(addr);
}

u32 Memory::Read32(const Common::ProcessAddress addr) {
    return impl->Read32(addr);
}

bool Memory::Read32Acquire(const Common::ProcessAddress addr, u32& value) {
    return impl->Read32Acquire(addr, value);
}

u64 Memory::Read64(const Common::ProcessAddress addr) {
    return impl->Read64(addr);
}

u128 Memory::Read128(const Common::ProcessAddress addr) {
    return impl->Read128(addr);
}

void Memory::Write8(Common::ProcessAddress addr, u8 data) {
    impl->Write8(addr, data);
}

void Memory::Write16(Common::ProcessAddress addr, u16 data) {
    impl->Write16(addr, data);
}

void Memory::Write32(Common::ProcessAddress addr, u32 data) {
    impl->Write32(addr, data);
}

void Memory::Write64(Common::ProcessAddress addr, u64 data) {
    impl->Write64(addr, data);
}

void Memory::Write128(Common::ProcessAddress addr, u128 data) {
    impl->Write128(addr, data);
}

bool Memory::WriteExclusive8(Common::ProcessAddress addr, u8 data, u8 expected) {
    return impl->WriteExclusive8(addr, data, expected);
}

bool Memory::WriteExclusive16(Common::ProcessAddress addr, u16 data, u16 expected) {
    return impl->WriteExclusive16(addr, data, expected);
}

bool Memory::WriteExclusive32(Common::ProcessAddress addr, u32 data, u32 expected) {
    return impl->WriteExclusive32(addr, data, expected);
}

bool Memory::WriteExclusive64(Common::ProcessAddress addr, u64 data, u64 expected) {
    return impl->WriteExclusive64(addr, data, expected);
}

bool Memory::WriteExclusive128(Common::ProcessAddress addr, u128 data, u128 expected) {
    return impl->WriteExclusive128(addr, data, expected);
}

std::string Memory::ReadCString(Common::ProcessAddress vaddr, std::size_t max_length) {
    return impl->ReadCString(vaddr, max_length);
}

bool Memory::ReadBlock(const Common::ProcessAddress src_addr, void* dest_buffer,
                       const std::size_t size) {
    return impl->ReadBlockImpl(src_addr, dest_buffer, size, false);
}

bool Memory::ReadBlockUnsafe(const Common::ProcessAddress src_addr, void* dest_buffer,
                             const std::size_t size) {
    return impl->ReadBlockImpl(src_addr, dest_buffer, size, true);
}

const u8* Memory::GetSpan(const VAddr src_addr, const std::size_t size) const {
    return impl->GetSpan(src_addr, size);
}

u8* Memory::GetSpan(const VAddr src_addr, const std::size_t size) {
    return impl->GetSpan(src_addr, size);
}

const u8* Memory::GetSafeReadSpan(const Common::ProcessAddress src_addr, const std::size_t size) {
    return impl->GetSafeReadSpan(src_addr, size);
}

u8* Memory::GetSafeWriteSpan(const Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->GetSafeWriteSpan(dest_addr, size);
}

bool Memory::WriteBlock(const Common::ProcessAddress dest_addr, const void* src_buffer,
                        const std::size_t size) {
    return impl->WriteBlockImpl(dest_addr, src_buffer, size, false);
}

bool Memory::WriteBlockUnsafe(const Common::ProcessAddress dest_addr, const void* src_buffer,
                              const std::size_t size) {
    return impl->WriteBlockImpl(dest_addr, src_buffer, size, true);
}

bool Memory::CopyBlock(Common::ProcessAddress dest_addr, Common::ProcessAddress src_addr,
                       const std::size_t size) {
    return impl->CopyBlock(dest_addr, src_addr, size);
}

bool Memory::ZeroBlock(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->ZeroBlock(dest_addr, size);
}

void Memory::SetGPUDirtyManagers(std::span<Core::GPUDirtyMemoryManager> managers) {
    impl->gpu_dirty_managers = managers;
    std::scoped_lock lk{impl->hot_cores_mutex};
    if (impl->hot_cores) {
        for (std::size_t i = 0; i < MaxHotCores; ++i) {
            (*impl->hot_cores)[i].gather_epoch_ptr =
                i < managers.size() ? managers[i].GatherEpochPtr() : nullptr;
        }
    }
}

Result Memory::InvalidateDataCache(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->InvalidateDataCache(dest_addr, size);
}

Result Memory::StoreDataCache(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->StoreDataCache(dest_addr, size);
}

Result Memory::FlushDataCache(Common::ProcessAddress dest_addr, const std::size_t size) {
    return impl->FlushDataCache(dest_addr, size);
}

void Memory::RasterizerMarkRegionCached(Common::ProcessAddress vaddr, u64 size, bool cached) {
    Impl::ScopedGenBump w_bump{impl->fast_write_generation, impl->write_gen_writers};
    std::optional<Impl::ScopedGenBump> r_bump;
    std::optional<Impl::ScopedGenBump> rw_bump;
    if (cached && !impl->ReadTrapsEnabled()) {
        r_bump.emplace(impl->fast_read_generation, impl->read_gen_writers);
    }
    if (!cached) {
        rw_bump.emplace(impl->rasterizer_write_generation, impl->rasterizer_write_gen_writers);
    }
    impl->RefreshJitReadTrapState();
    impl->RasterizerMarkRegionCached(GetInteger(vaddr), size, cached);
}

void Memory::BeginProtectBatch() noexcept {
#ifdef HAS_HYPERVISOR
    if (HvVm* const vm = HvVm::Get()) {
        vm->BeginProtectBatch();
    }
#endif
}

void Memory::EndProtectBatch() {
#ifdef HAS_HYPERVISOR
    if (HvVm* const vm = HvVm::Get()) {
        vm->EndProtectBatch();
    }
#endif
}

void Memory::RasterizerArmReadTraps(Common::ProcessAddress vaddr, u64 size) {
    Impl::ScopedGenBump r_bump{impl->fast_read_generation, impl->read_gen_writers};
    impl->RasterizerArmReadTraps(GetInteger(vaddr), size);
}

void Memory::MarkRegionDebug(Common::ProcessAddress vaddr, u64 size, bool debug) {
    Impl::ScopedGenBump r_bump{impl->fast_read_generation, impl->read_gen_writers};
    Impl::ScopedGenBump w_bump{impl->fast_write_generation, impl->write_gen_writers};
    Impl::ScopedGenBump rw_bump{impl->rasterizer_write_generation,
                                impl->rasterizer_write_gen_writers};
    impl->MarkRegionDebug(GetInteger(vaddr), size, debug);
}

bool Memory::TryCollectHotWrite(const std::size_t core_index, const u64 offset,
                                const std::size_t size, const u64 granule_mask,
                                HotWritePage& hot) noexcept {
    const auto invalidate = [&hot]() noexcept {
        hot.vaddr_page = ~u64{0};
        hot.collected_granules = 0;
        return false;
    };
    const auto* const dev_mem = impl->gpu_device_memory;
    if (dev_mem == nullptr || dev_mem->MappingGeneration() != hot.map_gen ||
        Core::g_write_answer_epoch.load(std::memory_order_relaxed) != hot.answer_epoch) {
        return invalidate();
    }
    const u64 cur_gen = impl->rasterizer_write_generation.load(std::memory_order_acquire);
    if (cur_gen != hot.mem_gen) {
        if (impl->rasterizer_write_gen_writers.load(std::memory_order_acquire) != 0) {
            return invalidate();
        }
        const Common::PageTable* const table = impl->current_page_table;
        if (table == nullptr || !AddressSpaceContains(*table, hot.vaddr_page, YUZU_PAGESIZE)) {
            return invalidate();
        }
        const auto raw = table->entries[hot.vaddr_page >> YUZU_PAGEBITS].Raw();
        if (static_cast<Common::PageType>(raw.type) != Common::PageType::RasterizerCachedMemory) {
            return invalidate();
        }
        const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw, true);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (ptr == 0 || reinterpret_cast<u8*>(ptr + hot.vaddr_page) != hot.host_page ||
            impl->rasterizer_write_gen_writers.load(std::memory_order_acquire) != 0 ||
            impl->rasterizer_write_generation.load(std::memory_order_acquire) != cur_gen) {
            return invalidate();
        }
        hot.mem_gen = cur_gen;
    }
    auto& manager = impl->gpu_dirty_managers[core_index];
    const u64 gather_before = manager.GatherEpoch();
    const DAddr daddr = hot.daddr_page | offset;
    VideoCommon::Nx2Hazard::NoteGuestWrite(daddr, size);
    if (manager.Collect(daddr, size)) {
        const DAddr granule_begin = daddr & ~DAddr{63};
        const u64 granule_end = (daddr + size + 63ULL) & ~u64{63};
        VideoCore::DownloadLanding::Await(granule_begin, granule_end - granule_begin);
    }
    Core::g_collected_write_pages.Mark(hot.daddr_page, 1);
    const u64 gather_after = manager.GatherEpoch();
    if (!VideoCommon::Nx2Hazard::Enabled() && (gather_before & 1) == 0 &&
        gather_before == gather_after) {
        if (hot.gather_epoch == gather_before) {
            hot.collected_granules |= granule_mask;
        } else {
            hot.gather_epoch = gather_before;
            hot.collected_granules = granule_mask;
        }
    } else {
        hot.collected_granules = 0;
    }
    return true;
}

bool Memory::WriteCoreInPage(const std::size_t core_index, const u64 vaddr, const u8* const src,
                             const std::size_t size, const u64 granule_mask,
                             HotWritePage& out_hot) noexcept {
    u8* const ptr = impl->GetPointerImpl(
        vaddr,
        [vaddr, src, size]() {
            if (UnmappedAccesses().Note(0x100U | static_cast<u32>(size * 8), vaddr)) {
                u64 val = 0;
                std::memcpy(&val, src, (std::min<std::size_t>)(size, sizeof(u64)));
                LOG_ERROR(HW_Memory, "Unmapped Write{} @ {:#016x} = {:#016x}", size * 8, vaddr,
                          val);
            }
        },
        [&](u8* host_ptr) { impl->HandleRasterizerWrite(vaddr, size, host_ptr, core_index); });
    if (ptr == nullptr) {
        out_hot.vaddr_page = ~u64{0};
        out_hot.collected_granules = 0;
        return false;
    }
    std::memcpy(ptr, src, size);
    return TryCacheHotWritePage(core_index, vaddr, granule_mask, out_hot);
}

#if defined(ARCHITECTURE_arm64)
namespace {

#if defined(__clang__)
#define LICHII_JIT_HOT_WRITE_CC [[clang::preserve_all]]
#define LICHII_JIT_HOT_WRITE_COLD_CC [[clang::preserve_all, clang::noinline]]
constexpr bool kJitHotWritePreserveAll = true;
#else
#define LICHII_JIT_HOT_WRITE_CC
#define LICHII_JIT_HOT_WRITE_COLD_CC [[gnu::noinline]]
constexpr bool kJitHotWritePreserveAll = false;
#endif

template <std::size_t ByteSize>
inline void StoreHotScalarBytes(u8* dst, u64 value) noexcept {
    if constexpr (ByteSize == 8) {
        std::memcpy(dst, &value, 8);
    } else if constexpr (ByteSize == 4) {
        const u32 v32 = static_cast<u32>(value);
        std::memcpy(dst, &v32, 4);
    } else if constexpr (ByteSize == 2) {
        const u16 v16 = static_cast<u16>(value);
        std::memcpy(dst, &v16, 2);
    } else {
        dst[0] = static_cast<u8>(value);
    }
}

template <std::size_t ByteSize>
LICHII_JIT_HOT_WRITE_COLD_CC bool JitColdCollectScalar(
    const Memory::JitHotWriteContext* ctx, u64 offset, u64 gmask,
    Memory::HotWritePage* hot, u64 value) noexcept {
    if (ctx->memory->TryCollectHotWrite(ctx->core_index, offset, ByteSize, gmask, *hot)) {
        StoreHotScalarBytes<ByteSize>(hot->host_page + offset, value);
        return true;
    }
    return false;
}

template <std::size_t ByteSize>
LICHII_JIT_HOT_WRITE_CC bool JitTryHotWriteScalar(
    const Memory::JitHotWriteContext* ctx, Memory::HotWritePage* hot,
    u64 vaddr, u64 value) noexcept {
    const u64 offset = vaddr & YUZU_PAGEMASK;
    const u64 gmask = ByteSize == 1
                          ? (u64{1} << (offset >> 6))
                          : ((u64{1} << (offset >> 6)) |
                             (u64{1} << ((offset + ByteSize - 1) >> 6)));
    const auto* const hot_core = ctx->hot_core;
    const auto* const dev_mem = hot_core->dev_mem;
    const auto* const gather_ptr = hot_core->gather_epoch_ptr;
    if ((hot->collected_granules & gmask) == gmask &&
        ctx->rast_gen_ptr->load(std::memory_order_relaxed) == hot->mem_gen &&
        Core::g_write_answer_epoch.load(std::memory_order_relaxed) == hot->answer_epoch &&
        dev_mem != nullptr && dev_mem->MappingGeneration() == hot->map_gen &&
        gather_ptr != nullptr) {
        const u64 gather_ep = gather_ptr->load(std::memory_order_acquire);
        if (gather_ep == hot->gather_epoch) {
            Core::g_collected_write_pages.Mark(hot->daddr_page, 1);
            StoreHotScalarBytes<ByteSize>(hot->host_page + offset, value);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (gather_ptr->load(std::memory_order_acquire) == gather_ep) {
                return true;
            }
        }
    }
    return JitColdCollectScalar<ByteSize>(ctx, offset, gmask, hot, value);
}

LICHII_JIT_HOT_WRITE_COLD_CC bool JitColdCollect128(
    const Memory::JitHotWriteContext* ctx, u64 offset, u64 gmask,
    Memory::HotWritePage* hot, uint8x16_t value) noexcept {
    if (ctx->memory->TryCollectHotWrite(ctx->core_index, offset, 16, gmask, *hot)) {
        std::memcpy(hot->host_page + offset, &value, sizeof(value));
        return true;
    }
    return false;
}

LICHII_JIT_HOT_WRITE_CC bool JitTryHotWrite128(
    const Memory::JitHotWriteContext* ctx, Memory::HotWritePage* hot,
    u64 vaddr, uint8x16_t value) noexcept {
    const u64 offset = vaddr & YUZU_PAGEMASK;
    const u64 gmask = (u64{1} << (offset >> 6)) | (u64{1} << ((offset + 15) >> 6));
    const auto* const hot_core = ctx->hot_core;
    const auto* const dev_mem = hot_core->dev_mem;
    const auto* const gather_ptr = hot_core->gather_epoch_ptr;
    if ((hot->collected_granules & gmask) == gmask &&
        ctx->rast_gen_ptr->load(std::memory_order_relaxed) == hot->mem_gen &&
        Core::g_write_answer_epoch.load(std::memory_order_relaxed) == hot->answer_epoch &&
        dev_mem != nullptr && dev_mem->MappingGeneration() == hot->map_gen &&
        gather_ptr != nullptr) {
        const u64 gather_ep = gather_ptr->load(std::memory_order_acquire);
        if (gather_ep == hot->gather_epoch) {
            Core::g_collected_write_pages.Mark(hot->daddr_page, 1);
            std::memcpy(hot->host_page + offset, &value, sizeof(value));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (gather_ptr->load(std::memory_order_acquire) == gather_ep) {
                return true;
            }
        }
    }
    return JitColdCollect128(ctx, offset, gmask, hot, value);
}

#undef LICHII_JIT_HOT_WRITE_COLD_CC
#undef LICHII_JIT_HOT_WRITE_CC

} // namespace
#endif

bool Memory::PopulateJitHotWrite(const std::size_t core_index, JitHotWriteContext& out_ctx,
                                 std::array<std::uintptr_t, 5>& out_write_fns,
                                 bool& out_preserve_all_abi) noexcept {
#if defined(ARCHITECTURE_arm64)
    HotCoreState* const hot_core = HotCore(core_index);
    if (hot_core == nullptr || core_index >= Core::Hardware::NUM_CPU_CORES ||
        rasterizer_write_gen_ptr == nullptr) {
        return false;
    }
    out_ctx = JitHotWriteContext{
        .memory = this,
        .core_index = core_index,
        .hot_core = hot_core,
        .rast_gen_ptr = rasterizer_write_gen_ptr,
    };
    out_write_fns = {
        reinterpret_cast<std::uintptr_t>(&JitTryHotWriteScalar<1>),
        reinterpret_cast<std::uintptr_t>(&JitTryHotWriteScalar<2>),
        reinterpret_cast<std::uintptr_t>(&JitTryHotWriteScalar<4>),
        reinterpret_cast<std::uintptr_t>(&JitTryHotWriteScalar<8>),
        reinterpret_cast<std::uintptr_t>(&JitTryHotWrite128),
    };
    out_preserve_all_abi = kJitHotWritePreserveAll;
    return true;
#else
    static_cast<void>(core_index);
    static_cast<void>(out_ctx);
    static_cast<void>(out_write_fns);
    static_cast<void>(out_preserve_all_abi);
    return false;
#endif
}

bool Memory::TryCacheHotWritePage(const std::size_t core_index, const u64 vaddr,
                                  const u64 granule_mask, HotWritePage& out_hot) noexcept {
    const Common::PageTable* const table = impl->current_page_table;
    if (table == nullptr || core_index >= Core::Hardware::NUM_CPU_CORES || !SysSlotEnabled() ||
        impl->gpu_dirty_managers.size() <= Core::Hardware::NUM_CPU_CORES ||
        VideoCommon::Nx2Spikes::Counting()) {
        out_hot.vaddr_page = ~u64{0};
        return false;
    }
    const u64 page_vaddr = (vaddr & 0xffffffffffffULL) & ~u64{YUZU_PAGEMASK};
    if (!AddressSpaceContains(*table, page_vaddr, YUZU_PAGESIZE)) {
        out_hot.vaddr_page = ~u64{0};
        return false;
    }
    const u64 plain_gen = impl->fast_write_generation.load(std::memory_order_acquire);
    const u64 rast_gen = impl->rasterizer_write_generation.load(std::memory_order_acquire);
    if (impl->write_gen_writers.load(std::memory_order_acquire) != 0 ||
        impl->rasterizer_write_gen_writers.load(std::memory_order_acquire) != 0) {
        out_hot.vaddr_page = ~u64{0};
        return false;
    }
    const auto raw = table->entries[page_vaddr >> YUZU_PAGEBITS].Raw();
    if (const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw); ptr != 0) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (impl->write_gen_writers.load(std::memory_order_acquire) != 0 ||
            impl->fast_write_generation.load(std::memory_order_acquire) != plain_gen) {
            out_hot.vaddr_page = ~u64{0};
            return false;
        }
        out_hot = HotWritePage{
            .vaddr_page = page_vaddr,
            .host_page = reinterpret_cast<u8*>(ptr + page_vaddr),
            .daddr_page = 0,
            .collected_granules = 0,
            .gather_epoch = 0,
            .mem_gen = plain_gen,
            .map_gen = 0,
            .answer_epoch = 0,
            .rasterizer_cached = false,
        };
        return true;
    }
    if (static_cast<Common::PageType>(raw.type) == Common::PageType::RasterizerCachedMemory &&
        TrapFastpathEnabled() && impl->gpu_device_memory != nullptr) {
        const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw, true);
        if (ptr != 0) {
            u8* const host_page = reinterpret_cast<u8*>(ptr + page_vaddr);
            const uintptr_t host_page_idx =
                reinterpret_cast<uintptr_t>(host_page) >> YUZU_PAGEBITS;
            const u64 map_gen = impl->gpu_device_memory->MappingGeneration();
            const auto& entry = impl->write_caches[core_index].Slot(host_page_idx);
            const u64 epoch = Core::g_write_answer_epoch.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (entry.host_page == host_page_idx && entry.map_gen == map_gen &&
                entry.collect && entry.answer_epoch == epoch &&
                impl->rasterizer_write_gen_writers.load(std::memory_order_acquire) == 0 &&
                impl->rasterizer_write_generation.load(std::memory_order_acquire) == rast_gen) {
                if (hot_cores_ptr != nullptr && core_index < MaxHotCores) {
                    hot_cores_ptr[core_index].dev_mem = impl->gpu_device_memory;
                }
                const u64 gather_ep = impl->gpu_dirty_managers[core_index].GatherEpoch();
                const bool seed_granule = !VideoCommon::Nx2Hazard::Enabled() &&
                                          (gather_ep & 1) == 0 && entry.gather_epoch == gather_ep &&
                                          (entry.collected_granules & granule_mask) == granule_mask;
                out_hot = HotWritePage{
                    .vaddr_page = page_vaddr,
                    .host_page = host_page,
                    .daddr_page = DAddr{entry.device_page} << YUZU_PAGEBITS,
                    .collected_granules = seed_granule ? entry.collected_granules : 0,
                    .gather_epoch = seed_granule ? gather_ep : 0,
                    .mem_gen = rast_gen,
                    .map_gen = map_gen,
                    .answer_epoch = epoch,
                    .rasterizer_cached = true,
                };
                return true;
            }
        }
    }
    out_hot.vaddr_page = ~u64{0};
    return false;
}

bool Memory::TryCacheHotReadPage(const std::size_t core_index, const u64 vaddr,
                                 HotReadPage& out_hot) noexcept {
    const Common::PageTable* const table = impl->current_page_table;
    if (table == nullptr) {
        out_hot.vaddr_page = ~u64{0};
        return false;
    }
    const u64 page_vaddr = (vaddr & 0xffffffffffffULL) & ~u64{YUZU_PAGEMASK};
    if (!AddressSpaceContains(*table, page_vaddr, YUZU_PAGESIZE)) {
        out_hot.vaddr_page = ~u64{0};
        return false;
    }
    const u64 gen = impl->fast_read_generation.load(std::memory_order_acquire);
    if (impl->read_gen_writers.load(std::memory_order_acquire) != 0) {
        out_hot.vaddr_page = ~u64{0};
        return false;
    }
    const auto raw = table->entries[page_vaddr >> YUZU_PAGEBITS].Raw();
    if (const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw); ptr != 0) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (impl->read_gen_writers.load(std::memory_order_acquire) != 0 ||
            impl->fast_read_generation.load(std::memory_order_acquire) != gen) {
            out_hot.vaddr_page = ~u64{0};
            return false;
        }
        out_hot = HotReadPage{
            .vaddr_page = page_vaddr,
            .host_page = reinterpret_cast<const u8*>(ptr + page_vaddr),
            .mem_gen = gen,
            .map_gen = 0,
            .rasterizer_cached = false,
            .uses_map_gen = false,
        };
        return true;
    }
    if (static_cast<Common::PageType>(raw.type) == Common::PageType::RasterizerCachedMemory) {
        const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw, true);
        if (ptr == 0) {
            out_hot.vaddr_page = ~u64{0};
            return false;
        }
        const u8* const host_page = reinterpret_cast<const u8*>(ptr + page_vaddr);
        if (impl->ReadTrapsEnabled()) {
            if (!impl->ReadTrapArmed(page_vaddr, YUZU_PAGESIZE)) {
                std::atomic_thread_fence(std::memory_order_acquire);
                if (impl->read_gen_writers.load(std::memory_order_acquire) == 0 &&
                    impl->fast_read_generation.load(std::memory_order_acquire) == gen) {
                    out_hot = HotReadPage{
                        .vaddr_page = page_vaddr,
                        .host_page = host_page,
                        .mem_gen = gen,
                        .map_gen = 0,
                        .rasterizer_cached = true,
                        .uses_map_gen = false,
                    };
                    return true;
                }
            }
        } else if (core_index < Core::Hardware::NUM_CPU_CORES &&
                   impl->gpu_device_memory != nullptr) {
            if (const auto dpage = impl->gpu_device_memory->TryGetSingleDevicePage(host_page)) {
                const DAddr dpage_begin = DAddr{*dpage} << YUZU_PAGEBITS;
                const u64 map_gen = impl->gpu_device_memory->MappingGeneration();
                if (impl->read_area_generations[core_index] ==
                        impl->read_trap_generation.load(std::memory_order_acquire) &&
                    impl->rasterizer_read_areas[core_index].Contains(
                        dpage_begin, dpage_begin + YUZU_PAGESIZE)) {
                    std::atomic_thread_fence(std::memory_order_acquire);
                    if (impl->read_gen_writers.load(std::memory_order_acquire) == 0 &&
                        impl->fast_read_generation.load(std::memory_order_acquire) == gen &&
                        impl->gpu_device_memory->MappingGeneration() == map_gen) {
                        if (hot_cores_ptr != nullptr && core_index < MaxHotCores) {
                            hot_cores_ptr[core_index].dev_mem = impl->gpu_device_memory;
                        }
                        out_hot = HotReadPage{
                            .vaddr_page = page_vaddr,
                            .host_page = host_page,
                            .mem_gen = gen,
                            .map_gen = map_gen,
                            .rasterizer_cached = true,
                            .uses_map_gen = true,
                        };
                        return true;
                    }
                }
            }
        }
    }
    out_hot.vaddr_page = ~u64{0};
    return false;
}

bool Memory::TryCacheCleanReadPage(u64 vaddr, const u8*& out_host_page,
                                   u64& out_generation) const noexcept {
    const Common::PageTable* const table = impl->current_page_table;
    if (table == nullptr) {
        return false;
    }
    const u64 page_vaddr = (vaddr & 0xffffffffffffULL) & ~u64{YUZU_PAGEMASK};
    if (!AddressSpaceContains(*table, page_vaddr, YUZU_PAGESIZE)) {
        return false;
    }
    const u64 gen = impl->fast_read_generation.load(std::memory_order_acquire);
    if (impl->read_gen_writers.load(std::memory_order_acquire) != 0) {
        return false;
    }
    const auto raw = table->entries[page_vaddr >> YUZU_PAGEBITS].Raw();
    if (const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw); ptr != 0) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (impl->read_gen_writers.load(std::memory_order_acquire) != 0 ||
            impl->fast_read_generation.load(std::memory_order_acquire) != gen) {
            return false;
        }
        out_host_page = reinterpret_cast<const u8*>(ptr + page_vaddr);
        out_generation = gen;
        return true;
    }
    if (static_cast<Common::PageType>(raw.type) == Common::PageType::RasterizerCachedMemory &&
        impl->ReadTrapsEnabled() && !impl->ReadTrapArmed(page_vaddr, YUZU_PAGESIZE)) {
        const uintptr_t ptr = Common::PageTable::PageEntryData::ExtractPointer(raw, true);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (ptr != 0 && impl->read_gen_writers.load(std::memory_order_acquire) == 0 &&
            impl->fast_read_generation.load(std::memory_order_acquire) == gen) {
            out_host_page = reinterpret_cast<const u8*>(ptr + page_vaddr);
            out_generation = gen;
            return true;
        }
    }
    return false;
}

void Memory::DownloadRasterizerRegion(Common::ProcessAddress vaddr, size_t size) {
    impl->HandleRasterizerDownload(GetInteger(vaddr), size);
}

bool Memory::InvalidateNCE(Common::ProcessAddress vaddr, size_t size) {
    [[maybe_unused]] bool mapped = true;
    [[maybe_unused]] bool rasterizer = false;

    u8* const ptr = impl->GetPointerImpl(
        GetInteger(vaddr),
        [&] {
            LOG_ERROR(HW_Memory, "Unmapped InvalidateNCE for {} bytes @ {:#x}", size,
                      GetInteger(vaddr));
            mapped = false;
        },
        [&] { rasterizer = true; });
    if (rasterizer) {
        impl->InvalidateGPUMemory(ptr, size);
    }

#ifdef __ANDROID__
    if (!rasterizer && mapped) {
        impl->host_buffer->DeferredMapSeparateHeap(GetInteger(vaddr));
    }
#endif

    return mapped && ptr != nullptr;
}

bool Memory::InvalidateSeparateHeap(void* fault_address) {
#ifdef __ANDROID__
    return impl->host_buffer->DeferredMapSeparateHeap(static_cast<u8*>(fault_address));
#else
    return false;
#endif
}

} // namespace Core::Memory
