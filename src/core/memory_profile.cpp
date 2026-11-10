// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#include <fmt/format.h>

#include "common/host_memory.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_memory_block.h"
#include "core/hle/kernel/k_memory_manager.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "core/memory_profile.h"
#include "video_core/host1x/host1x.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace Core {
namespace {

constexpr size_t PageSize = 0x1000;
constexpr size_t GpuShared = 0x20;
constexpr u8 Unmapped = 0xFF;

constexpr std::array<const char*, 0x18> StateNames{
    "free",        "io",           "static",         "code",          "code data",
    "heap",        "shared",       "alias",          "alias code",    "alias code data",
    "ipc",         "stack",        "thread local",   "transferred",   "shared transferred",
    "shared code", "inaccessible", "non-secure ipc", "non-device ipc", "kernel",
    "generated code", "code out",  "coverage",       "insecure",
};

struct Tally {
    u64 mapped;          ///< Guest address space bytes in this state
    u64 pages;           ///< Distinct backing pages first seen in this state
    u64 resident_data;   ///< Resident, non-zero
    u64 resident_zero;   ///< Resident, all zeros
    u64 paged_data;      ///< Not resident, non-zero (in the page file or compressed)
    u64 sparse;          ///< Non-zero pages with at most 8 non-zero words
    u64 fill;            ///< Non-zero pages of one repeated word
};

bool IsZeroPage(const u8* page) {
    const auto* words = reinterpret_cast<const u64*>(page);
    u64 bits = 0;
    for (size_t i = 0; i < PageSize / sizeof(u64); ++i) {
        bits |= words[i];
    }
    return bits == 0;
}

/// Counts a non-zero page as sparse or a fill pattern when it is one.
void CountData(Tally& tally, const u8* page) {
    const auto* words = reinterpret_cast<const u64*>(page);
    size_t non_zero = 0;
    bool same = true;
    for (size_t i = 0; i < PageSize / sizeof(u64); ++i) {
        non_zero += words[i] != 0 ? 1 : 0;
        same = same && words[i] == words[0];
    }
    if (same) {
        tally.fill += PageSize;
    } else if (non_zero <= 8) {
        tally.sparse += PageSize;
    }
}

constexpr u64 MiB(u64 bytes) {
    return bytes >> 20;
}

std::string ProfileGuest(System& system) {
    Kernel::KProcess* const process = system.ApplicationProcess();
    auto& buffer = system.DeviceMemory().buffer;
    u8* const backing = buffer.BackingBasePointer();
    if (process == nullptr || backing == nullptr) {
        return "no application process";
    }
    const size_t num_pages = buffer.BackingSize() / PageSize;
    std::vector<u8> owner(num_pages, Unmapped);
    std::array<Tally, 2 * GpuShared> tallies{};
    auto& page_table = process->GetPageTable();
    auto& memory = process->GetMemory();
    const auto& device_memory = system.Host1x().MemoryManager();
    u64 address = 0;
    while (true) {
        Kernel::KMemoryInfo info{};
        Kernel::Svc::PageInfo page_info{};
        if (page_table.QueryInfo(&info, &page_info, address).IsError()) {
            break;
        }
        const auto state = static_cast<size_t>(info.m_state & Kernel::KMemoryState::Mask);
        const bool has_pages = state != static_cast<size_t>(Kernel::Svc::MemoryState::Free) &&
                               state != static_cast<size_t>(Kernel::Svc::MemoryState::Inaccessible) &&
                               state != static_cast<size_t>(Kernel::Svc::MemoryState::Io) &&
                               state < StateNames.size();
        if (has_pages) {
            const bool shared = True(info.m_attribute & Kernel::KMemoryAttribute::DeviceShared);
            for (u64 page = info.m_address; page < info.m_address + info.m_size; page += PageSize) {
                const u8* const pointer = memory.GetPointerSilent(page);
                if (pointer < backing) {
                    tallies[state].mapped += PageSize;
                    continue;
                }
                const size_t offset = static_cast<size_t>(pointer - backing);
                // GPU: mapped into the GPU's (SMMU) address space.
                const bool gpu = shared || device_memory.IsPhysicalPageMapped(offset);
                const size_t category = state + (gpu ? GpuShared : 0);
                tallies[category].mapped += PageSize;
                const size_t index = offset / PageSize;
                if (index < num_pages && owner[index] == Unmapped) {
                    owner[index] = static_cast<u8>(category);
                }
            }
        }
        const u64 next = info.m_address + info.m_size;
        if (next <= address) {
            break;
        }
        address = next;
    }

    const std::vector<bool> resident = buffer.ResidentPages();
    const bool known = resident.size() == num_pages;
    Tally other{};
    std::vector<std::pair<size_t, size_t>> brought_in;
    // Per 2 MiB of backing: the application's zero pages, and whether anything there holds data.
    // A backing made of separate host objects could only give back the ones without data.
    constexpr size_t ChunkPages = (size_t{2} << 20) / PageSize;
    struct Chunk {
        u32 zero;
        u32 resident_zero;
        bool data;
    };
    std::vector<Chunk> chunks((num_pages + ChunkPages - 1) / ChunkPages);
    // Where the resident pages outside the application are, per 64 MiB of backing.
    constexpr size_t WindowPages = (size_t{64} << 20) / PageSize;
    std::vector<u32> other_windows((num_pages + WindowPages - 1) / WindowPages);
    for (size_t index = 0; index < num_pages; ++index) {
        const bool is_resident = known && resident[index];
        Chunk& chunk = chunks[index / ChunkPages];
        if (owner[index] == Unmapped) {
            // Kernel, services and pages the game gave back: only what is resident.
            if (is_resident) {
                ++other.pages;
                ++other_windows[index / WindowPages];
                const bool zero = IsZeroPage(backing + index * PageSize);
                (zero ? other.resident_zero : other.resident_data) += PageSize;
                chunk.data = chunk.data || !zero;
            }
            continue;
        }
        Tally& tally = tallies[owner[index]];
        ++tally.pages;
        const bool zero = IsZeroPage(backing + index * PageSize);
        if (!zero) {
            CountData(tally, backing + index * PageSize);
            chunk.data = true;
        } else {
            ++chunk.zero;
            chunk.resident_zero += is_resident ? 1 : 0;
        }
        if (is_resident) {
            (zero ? tally.resident_zero : tally.resident_data) += PageSize;
            continue;
        }
        if (!zero) {
            tally.paged_data += PageSize;
        }
        if (!brought_in.empty() && brought_in.back().second == index) {
            brought_in.back().second = index + 1;
        } else {
            brought_in.emplace_back(index, index + 1);
        }
    }
    // Reading them made the host bring them in: hand them back.
    for (const auto& [first, end] : brought_in) {
        buffer.HintCold(first * PageSize, (end - first) * PageSize, std::nullopt);
    }

    Tally total{};
    std::string text;
    for (size_t category = 0; category < tallies.size(); ++category) {
        const Tally& tally = tallies[category];
        if (tally.mapped == 0) {
            continue;
        }
        total.mapped += tally.mapped;
        total.pages += tally.pages;
        total.resident_data += tally.resident_data;
        total.resident_zero += tally.resident_zero;
        total.paged_data += tally.paged_data;
        total.sparse += tally.sparse;
        total.fill += tally.fill;
        text += fmt::format(
            "\n  {}{}: mapped {} MiB, backing {} MiB, data {} MiB (resident {}, paged out {}, "
            "sparse {}, fill {}), resident zero {} MiB",
            StateNames[category % GpuShared], category >= GpuShared ? " (GPU)" : "",
            MiB(tally.mapped), MiB(tally.pages * PageSize),
            MiB(tally.resident_data + tally.paged_data), MiB(tally.resident_data),
            MiB(tally.paged_data), MiB(tally.sparse), MiB(tally.fill), MiB(tally.resident_zero));
    }
    std::string chunk_text;
    for (const size_t size : {size_t{1}, size_t{8}, size_t{32}}) {
        u64 zero = 0;
        u64 resident_zero = 0;
        for (size_t first = 0; first < chunks.size(); first += size) {
            const size_t end = (std::min)(first + size, chunks.size());
            bool data = false;
            u64 chunk_zero = 0;
            u64 chunk_resident_zero = 0;
            for (size_t i = first; i < end; ++i) {
                data = data || chunks[i].data;
                chunk_zero += chunks[i].zero;
                chunk_resident_zero += chunks[i].resident_zero;
            }
            if (!data) {
                zero += chunk_zero;
                resident_zero += chunk_resident_zero;
            }
        }
        chunk_text += fmt::format("{}{} MiB chunks {} MiB (resident {})", chunk_text.empty() ? "" : ", ",
                                  size * 2, MiB(zero * PageSize), MiB(resident_zero * PageSize));
    }
    text += fmt::format("\n  application zero pages in backing chunks without data: {}", chunk_text);
    // Runs of 64 MiB windows that each hold 8 MiB or more.
    std::string other_text;
    constexpr u32 WindowFloor = (u32{8} << 20) / PageSize;
    for (size_t first = 0; first < other_windows.size();) {
        if (other_windows[first] < WindowFloor) {
            ++first;
            continue;
        }
        size_t end = first;
        u64 pages = 0;
        for (; end < other_windows.size() && other_windows[end] >= WindowFloor; ++end) {
            pages += other_windows[end];
        }
        other_text += fmt::format("{}{}-{} MiB: {} MiB", other_text.empty() ? "" : ", ", first * 64,
                                  end * 64, MiB(pages * PageSize));
        first = end;
    }
    if (!other_text.empty()) {
        text += fmt::format("\n  resident outside the application, by backing offset: {}", other_text);
    }
    auto& manager = system.Kernel().MemoryManager();
    using Pool = Kernel::KMemoryManager::Pool;
    const auto used = [&manager](Pool pool) {
        return MiB(manager.GetSize(pool) - manager.GetFreeSize(pool));
    };
    return fmt::format(
        "guest: backing {} MiB, residency {}; application mapped {} MiB on {} MiB of backing: "
        "data {} MiB (resident {}, paged out {}, sparse {}, fill {}), resident zero {} MiB; "
        "outside the application: "
        "resident {} MiB (data {}); kernel pools in use: application {} / {} MiB, applet {}, "
        "system {}, non-secure {}; process: heap {} MiB, user physical {} / {} MiB; page table "
        "at {} committed {} MiB{}",
        MiB(buffer.BackingSize()), known ? "known" : "unknown", MiB(total.mapped),
        MiB(total.pages * PageSize), MiB(total.resident_data + total.paged_data),
        MiB(total.resident_data), MiB(total.paged_data), MiB(total.sparse), MiB(total.fill),
        MiB(total.resident_zero),
        MiB(other.resident_data + other.resident_zero), MiB(other.resident_data),
        used(Pool::Application), MiB(manager.GetSize(Pool::Application)), used(Pool::Applet),
        used(Pool::System), used(Pool::SystemNonSecure),
        MiB(process->GetPageTable().GetNormalMemorySize()),
        MiB(process->GetUsedUserPhysicalMemorySize(system.Kernel())),
        MiB(process->GetTotalUserPhysicalMemorySize(system.Kernel())),
        fmt::ptr(page_table.GetImpl().entries.data()),
        MiB(page_table.GetImpl().entries.CommittedBytes()), text);
}

#ifdef _WIN32
// What the host process has resident besides guest memory, and what malloc holds.
std::string ProfileHost(System& system) {
    auto& buffer = system.DeviceMemory().buffer;
    const auto backing = reinterpret_cast<uintptr_t>(buffer.BackingBasePointer());
    const auto view = reinterpret_cast<uintptr_t>(buffer.VirtualBasePointer());
    const size_t backing_size = buffer.BackingSize();
    const size_t view_size = buffer.VirtualSize();

    std::vector<ULONG_PTR> entries(size_t{1} << 22);
    for (int attempt = 0; attempt < 4; ++attempt) {
        if (K32QueryWorkingSet(GetCurrentProcess(), entries.data(),
                               static_cast<DWORD>(entries.size() * sizeof(ULONG_PTR)))) {
            break;
        }
        entries.assign((std::max)(entries.size() * 2, size_t{entries[0]} + (size_t{1} << 18)), 0);
    }
    const size_t count = (std::min)(size_t{entries[0]}, entries.size() - 1);
    std::vector<uintptr_t> pages;
    pages.reserve(count);
    u64 guest = 0;
    for (size_t i = 1; i <= count; ++i) {
        const uintptr_t page = entries[i] & ~uintptr_t{PageSize - 1};
        if (page - backing < backing_size || (view != 0 && page - view < view_size)) {
            guest += PageSize;
        } else {
            pages.push_back(page);
        }
    }
    std::sort(pages.begin(), pages.end());
    u64 image = 0;
    u64 mapped = 0;
    u64 private_data = 0;
    u64 private_code = 0;
    std::vector<std::pair<uintptr_t, u64>> allocations;
    MEMORY_BASIC_INFORMATION region{};
    uintptr_t region_end = 0;
    for (const uintptr_t page : pages) {
        if (page >= region_end || page < reinterpret_cast<uintptr_t>(region.BaseAddress)) {
            if (VirtualQuery(reinterpret_cast<void*>(page), &region, sizeof(region)) == 0) {
                continue;
            }
            region_end = reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
        }
        const bool code = (region.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
                                             PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
        if (region.Type == MEM_IMAGE) {
            image += PageSize;
        } else if (region.Type == MEM_MAPPED) {
            mapped += PageSize;
        } else {
            (code ? private_code : private_data) += PageSize;
            const auto base = reinterpret_cast<uintptr_t>(region.AllocationBase);
            if (allocations.empty() || allocations.back().first != base) {
                allocations.emplace_back(base, 0);
            }
            allocations.back().second += PageSize;
        }
    }
    // The private allocations with the most resident memory (heap segments, page tables, the
    // driver's), with their reserved size.
    std::sort(allocations.begin(), allocations.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string allocations_text;
    for (size_t i = 0; i < (std::min<size_t>)(allocations.size(), 16); ++i) {
        u64 reserved = 0;
        for (uintptr_t address = allocations[i].first;; address += region.RegionSize) {
            if (VirtualQuery(reinterpret_cast<void*>(address), &region, sizeof(region)) == 0 ||
                reinterpret_cast<uintptr_t>(region.AllocationBase) != allocations[i].first) {
                break;
            }
            reserved += region.RegionSize;
        }
        allocations_text += fmt::format("{}{} MiB of {} MiB at {:#x}", i == 0 ? "" : ", ",
                                        MiB(allocations[i].second), MiB(reserved),
                                        allocations[i].first);
    }

    // Private memory committed, resident or not (malloc, the JIT, driver allocations).
    u64 committed = 0;
    u64 committed_code = 0;
    for (uintptr_t address = 0;;) {
        if (VirtualQuery(reinterpret_cast<void*>(address), &region, sizeof(region)) == 0) {
            break;
        }
        if (region.State == MEM_COMMIT && region.Type == MEM_PRIVATE) {
            const bool code = (region.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
                                                 PAGE_EXECUTE_READWRITE)) != 0;
            (code ? committed_code : committed) += region.RegionSize;
        }
        const uintptr_t next = reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize;
        if (next <= address) {
            break;
        }
        address = next;
    }

    // Busy heap blocks (malloc is the process heap), by size.
    constexpr std::array<u64, 5> Limits{u64{4} << 10, u64{64} << 10, u64{1} << 20, u64{16} << 20,
                                        ~u64{0}};
    std::array<u64, Limits.size()> bucket_bytes{};
    std::array<u64, Limits.size()> bucket_blocks{};
    std::array<u64, 10> largest{};
    std::array<HANDLE, 64> heaps{};
    const DWORD num_heaps = (std::min)(GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data()),
                                       static_cast<DWORD>(heaps.size()));
    for (DWORD i = 0; i < num_heaps; ++i) {
        if (!HeapLock(heaps[i])) {
            continue;
        }
        PROCESS_HEAP_ENTRY entry{};
        while (HeapWalk(heaps[i], &entry)) {
            if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) == 0) {
                continue;
            }
            const u64 size = entry.cbData;
            size_t bucket = 0;
            while (size >= Limits[bucket]) {
                ++bucket;
            }
            bucket_bytes[bucket] += size;
            ++bucket_blocks[bucket];
            if (size > largest.back()) {
                largest.back() = size;
                std::sort(largest.rbegin(), largest.rend());
            }
        }
        HeapUnlock(heaps[i]);
    }
    std::string largest_text;
    for (const u64 size : largest) {
        if (size != 0) {
            largest_text += fmt::format("{}{:.1f}", largest_text.empty() ? "" : ", ",
                                        static_cast<double>(size) / (1 << 20));
        }
    }
    // PROCESS_MEMORY_COUNTERS_EX2, which MinGW's psapi.h does not declare.
    struct {
        DWORD cb;
        DWORD PageFaultCount;
        SIZE_T PeakWorkingSetSize;
        SIZE_T WorkingSetSize;
        SIZE_T QuotaPeakPagedPoolUsage;
        SIZE_T QuotaPagedPoolUsage;
        SIZE_T QuotaPeakNonPagedPoolUsage;
        SIZE_T QuotaNonPagedPoolUsage;
        SIZE_T PagefileUsage;
        SIZE_T PeakPagefileUsage;
        SIZE_T PrivateUsage;
        SIZE_T PrivateWorkingSetSize;
        ULONG64 SharedCommitUsage;
    } counters{};
    counters.cb = sizeof(counters);
    K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                            sizeof(counters));
    return fmt::format(
        "host: working set {} MiB (peak {}, private {}): guest {} MiB, images {} MiB, mapped {} MiB "
        "(driver), private data {} MiB, private code {} MiB (JIT); private committed {} MiB + "
        "code {} MiB; heap blocks <4K {} MiB ({}), <64K {} MiB ({}), <1M {} MiB ({}), <16M {} MiB "
        "({}), >=16M {} MiB ({}); largest MiB: {}; private allocations resident: {}",
        MiB(counters.WorkingSetSize), MiB(counters.PeakWorkingSetSize),
        MiB(counters.PrivateWorkingSetSize), MiB(guest), MiB(image), MiB(mapped),
        MiB(private_data), MiB(private_code), MiB(committed), MiB(committed_code),
        MiB(bucket_bytes[0]), bucket_blocks[0], MiB(bucket_bytes[1]), bucket_blocks[1],
        MiB(bucket_bytes[2]), bucket_blocks[2], MiB(bucket_bytes[3]), bucket_blocks[3],
        MiB(bucket_bytes[4]), bucket_blocks[4], largest_text, allocations_text);
}
#endif

} // namespace

std::string ProfileMemory(System& system) {
    const auto start = std::chrono::steady_clock::now();
    std::string text;
#ifdef _WIN32
    // First: the guest pass brings pages in.
    text = ProfileHost(system) + "; ";
#endif
    text += ProfileGuest(system);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    LOG_INFO(Core, "Memory profile ({} ms): {}", ms, text);
    return text;
}

} // namespace Core
