// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Measurement builds only (-DLICHII_ALLOC_TRACE=ON, MSVC): replaces the global operator new and
// delete to remember the call stack of live allocations, for the debug server's alloc_trace
// method. Every block of 4 KiB or more is recorded; smaller ones are sampled once per 32 KiB a
// thread allocates and stand for that many bytes.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <string>
#include <vector>

#include <windows.h>
#include <psapi.h>

#include <fmt/format.h>

#include "common/common_types.h"
#include "core/memory_profile.h"

namespace {

constexpr size_t Exact = 4096;
constexpr long long Interval = 32 << 10;
constexpr size_t ResidencyFloor = 16 << 10;
constexpr size_t Slots = size_t{1} << 18;
constexpr size_t Frames = 14;
constexpr uintptr_t Tombstone = 1;

struct Info {
    size_t size;
    size_t weight;
    bool aligned;
    void* stack[Frames];
};

std::atomic<uintptr_t> g_keys[Slots];
Info g_info[Slots];
std::atomic<size_t> g_max_probe{0};
SRWLOCK g_lock = SRWLOCK_INIT;
thread_local long long t_budget = Interval;

size_t Hash(uintptr_t p) {
    return static_cast<size_t>(((p >> 4) * 0x9E3779B97F4A7C15ULL) >> 46) & (Slots - 1);
}

void Add(void* pointer, size_t size, size_t weight, bool aligned) {
    Info info{size, weight, aligned, {}};
    RtlCaptureStackBackTrace(1, static_cast<DWORD>(Frames), info.stack, nullptr);
    const auto p = reinterpret_cast<uintptr_t>(pointer);
    AcquireSRWLockExclusive(&g_lock);
    const size_t limit = g_max_probe.load(std::memory_order_relaxed);
    size_t slot = Slots;
    size_t distance = 0;
    // A block freed by code outside this executable left its record behind: reuse it.
    for (size_t i = Hash(p), n = 0; n < Slots; ++n, i = (i + 1) & (Slots - 1)) {
        const uintptr_t key = g_keys[i].load(std::memory_order_relaxed);
        if (key == p) {
            slot = i;
            distance = n;
            break;
        }
        if (key <= Tombstone && slot == Slots) {
            slot = i;
            distance = n;
        }
        if (key == 0 || (n >= limit && slot != Slots)) {
            break;
        }
    }
    if (slot != Slots) {
        g_info[slot] = info;
        g_keys[slot].store(p, std::memory_order_release);
        if (distance + 1 > limit) {
            g_max_probe.store(distance + 1, std::memory_order_relaxed);
        }
    }
    ReleaseSRWLockExclusive(&g_lock);
}

void Remove(void* pointer) {
    const auto p = reinterpret_cast<uintptr_t>(pointer);
    const size_t limit = g_max_probe.load(std::memory_order_relaxed);
    for (size_t i = Hash(p), n = 0; n < limit; ++n, i = (i + 1) & (Slots - 1)) {
        const uintptr_t key = g_keys[i].load(std::memory_order_relaxed);
        if (key == 0) {
            return;
        }
        if (key == p) {
            AcquireSRWLockExclusive(&g_lock);
            if (g_keys[i].load(std::memory_order_relaxed) == p) {
                g_keys[i].store(Tombstone, std::memory_order_relaxed);
            }
            ReleaseSRWLockExclusive(&g_lock);
            return;
        }
    }
}

void Note(void* pointer, size_t size, bool aligned) {
    if (size >= Exact) {
        Add(pointer, size, size, aligned);
        return;
    }
    t_budget -= static_cast<long long>(size);
    if (t_budget <= 0) {
        t_budget += Interval;
        Add(pointer, size, static_cast<size_t>(Interval), aligned);
    }
}

// Address space the executable reserves itself (page tables, the JIT's arenas), through its
// import of VirtualAlloc.
struct Region {
    std::atomic<uintptr_t> base;
    void* stack[Frames];
};
constexpr size_t MaxRegions = 8192;
Region g_regions[MaxRegions];
std::atomic<size_t> g_region_count{0};
using VirtualAllocFn = LPVOID(WINAPI*)(LPVOID, SIZE_T, DWORD, DWORD);
VirtualAllocFn g_virtual_alloc = nullptr;

LPVOID WINAPI TracedVirtualAlloc(LPVOID address, SIZE_T size, DWORD type, DWORD protect) {
    void* const result = g_virtual_alloc(address, size, type, protect);
    if (result != nullptr && address == nullptr) {
        const size_t index = g_region_count.fetch_add(1, std::memory_order_relaxed);
        if (index < MaxRegions) {
            RtlCaptureStackBackTrace(1, static_cast<DWORD>(Frames), g_regions[index].stack,
                                     nullptr);
            g_regions[index].base.store(reinterpret_cast<uintptr_t>(result),
                                        std::memory_order_release);
        }
    }
    return result;
}

struct ImportHook {
    ImportHook() {
        auto* const image = reinterpret_cast<u8*>(GetModuleHandleW(nullptr));
        const auto* const dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
        const auto* const nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(image + dos->e_lfanew);
        const IMAGE_DATA_DIRECTORY& directory =
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        if (directory.VirtualAddress == 0) {
            return;
        }
        for (auto* module = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(
                 image + directory.VirtualAddress);
             module->Name != 0; ++module) {
            if (module->OriginalFirstThunk == 0) {
                continue;
            }
            const auto* name = reinterpret_cast<const IMAGE_THUNK_DATA*>(
                image + module->OriginalFirstThunk);
            auto* slot = reinterpret_cast<IMAGE_THUNK_DATA*>(image + module->FirstThunk);
            for (; name->u1.AddressOfData != 0; ++name, ++slot) {
                if (IMAGE_SNAP_BY_ORDINAL(name->u1.Ordinal)) {
                    continue;
                }
                const auto* const by_name =
                    reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(image + name->u1.AddressOfData);
                if (std::strcmp(by_name->Name, "VirtualAlloc") != 0) {
                    continue;
                }
                DWORD old_protect = 0;
                if (!VirtualProtect(&slot->u1.Function, sizeof(slot->u1.Function), PAGE_READWRITE,
                                    &old_protect)) {
                    continue;
                }
                if (g_virtual_alloc == nullptr) {
                    g_virtual_alloc = reinterpret_cast<VirtualAllocFn>(slot->u1.Function);
                }
                slot->u1.Function = reinterpret_cast<ULONG_PTR>(&TracedVirtualAlloc);
                VirtualProtect(&slot->u1.Function, sizeof(slot->u1.Function), old_protect,
                               &old_protect);
            }
        }
    }
};
const ImportHook g_import_hook;

void* Allocate(size_t size) {
    void* const pointer = std::malloc(size == 0 ? 1 : size);
    if (pointer == nullptr) {
        throw std::bad_alloc{};
    }
    Note(pointer, size, false);
    return pointer;
}

void* AllocateAligned(size_t size, std::align_val_t alignment) {
    void* const pointer = _aligned_malloc(size == 0 ? 1 : size, static_cast<size_t>(alignment));
    if (pointer == nullptr) {
        throw std::bad_alloc{};
    }
    Note(pointer, size, true);
    return pointer;
}

void Release(void* pointer) {
    if (pointer != nullptr) {
        Remove(pointer);
        std::free(pointer);
    }
}

void ReleaseAligned(void* pointer) {
    if (pointer != nullptr) {
        Remove(pointer);
        _aligned_free(pointer);
    }
}

u64 ResidentBytes(uintptr_t begin, size_t size) {
    constexpr size_t PageSize = 0x1000;
    constexpr size_t Batch = 4096;
    static PSAPI_WORKING_SET_EX_INFORMATION pages[Batch];
    const uintptr_t first = begin & ~uintptr_t{PageSize - 1};
    const uintptr_t end = begin + size;
    u64 resident = 0;
    for (uintptr_t page = first; page < end;) {
        size_t count = 0;
        for (; count < Batch && page < end; ++count, page += PageSize) {
            pages[count].VirtualAddress = reinterpret_cast<void*>(page);
        }
        if (!K32QueryWorkingSetEx(GetCurrentProcess(), pages,
                                  static_cast<DWORD>(count * sizeof(pages[0])))) {
            break;
        }
        for (size_t i = 0; i < count; ++i) {
            if (pages[i].VirtualAttributes.Valid) {
                const auto address = reinterpret_cast<uintptr_t>(pages[i].VirtualAddress);
                resident += (std::min)(address + PageSize, end) - (std::max)(address, begin);
            }
        }
    }
    return resident;
}

} // namespace

void* operator new(size_t size) {
    return Allocate(size);
}
void* operator new[](size_t size) {
    return Allocate(size);
}
void* operator new(size_t size, const std::nothrow_t&) noexcept {
    void* const pointer = std::malloc(size == 0 ? 1 : size);
    if (pointer != nullptr) {
        Note(pointer, size, false);
    }
    return pointer;
}
void* operator new[](size_t size, const std::nothrow_t&) noexcept {
    void* const pointer = std::malloc(size == 0 ? 1 : size);
    if (pointer != nullptr) {
        Note(pointer, size, false);
    }
    return pointer;
}
void* operator new(size_t size, std::align_val_t alignment) {
    return AllocateAligned(size, alignment);
}
void* operator new[](size_t size, std::align_val_t alignment) {
    return AllocateAligned(size, alignment);
}
void operator delete(void* pointer) noexcept {
    Release(pointer);
}
void operator delete[](void* pointer) noexcept {
    Release(pointer);
}
void operator delete(void* pointer, size_t) noexcept {
    Release(pointer);
}
void operator delete[](void* pointer, size_t) noexcept {
    Release(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
    Release(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
    Release(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
    ReleaseAligned(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
    ReleaseAligned(pointer);
}
void operator delete(void* pointer, size_t, std::align_val_t) noexcept {
    ReleaseAligned(pointer);
}
void operator delete[](void* pointer, size_t, std::align_val_t) noexcept {
    ReleaseAligned(pointer);
}

namespace Core {

std::string AllocTraceReport() {
    struct Live {
        uintptr_t pointer;
        Info info;
    };
    // Not from operator new: the lock is held.
    size_t capacity = 1 << 16;
    auto* live = static_cast<Live*>(std::malloc(capacity * sizeof(Live)));
    size_t count = 0;
    AcquireSRWLockShared(&g_lock);
    for (size_t i = 0; i < Slots && live != nullptr; ++i) {
        const uintptr_t key = g_keys[i].load(std::memory_order_relaxed);
        if (key <= Tombstone) {
            continue;
        }
        if (count == capacity) {
            capacity *= 2;
            live = static_cast<Live*>(std::realloc(live, capacity * sizeof(Live)));
            if (live == nullptr) {
                break;
            }
        }
        live[count++] = Live{key, g_info[i]};
    }
    ReleaseSRWLockShared(&g_lock);
    if (live == nullptr) {
        return "{}";
    }

    struct Group {
        u64 count;
        u64 bytes;
        u64 weight;
        u64 resident;
    };
    std::map<std::vector<void*>, Group> groups;
    const HANDLE heap = GetProcessHeap();
    u64 stale = 0;
    for (size_t i = 0; i < count; ++i) {
        const Info& info = live[i].info;
        const auto* const pointer = reinterpret_cast<const void*>(live[i].pointer);
        // Freed by a delete outside this executable (a Qt parent, the CRT's).
        if (!info.aligned &&
            (!HeapValidate(heap, 0, pointer) || HeapSize(heap, 0, pointer) < info.size)) {
            ++stale;
            continue;
        }
        Group& group = groups[std::vector<void*>(info.stack, info.stack + Frames)];
        ++group.count;
        group.bytes += info.size;
        group.weight += info.weight;
        group.resident += info.size >= ResidencyFloor ? ResidentBytes(live[i].pointer, info.size)
                                                      : info.weight;
    }
    std::free(live);

    std::string text = fmt::format(
        "{{\"module_base\":{},\"stale\":{},\"regions\":[",
        reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr)), stale);
    // Newest first: an address can be handed out again after a release.
    std::vector<uintptr_t> seen;
    const size_t num_regions =
        (std::min)(g_region_count.load(std::memory_order_relaxed), MaxRegions);
    for (size_t i = num_regions; i-- > 0;) {
        const uintptr_t base = g_regions[i].base.load(std::memory_order_acquire);
        if (base == 0 || std::find(seen.begin(), seen.end(), base) != seen.end()) {
            continue;
        }
        seen.push_back(base);
        u64 reserved = 0;
        u64 committed = 0;
        u64 resident = 0;
        bool code = false;
        MEMORY_BASIC_INFORMATION region{};
        for (uintptr_t address = base;; address += region.RegionSize) {
            if (VirtualQuery(reinterpret_cast<void*>(address), &region, sizeof(region)) == 0 ||
                reinterpret_cast<uintptr_t>(region.AllocationBase) != base ||
                region.State == MEM_FREE) {
                break;
            }
            reserved += region.RegionSize;
            if (region.State == MEM_COMMIT) {
                committed += region.RegionSize;
                resident += ResidentBytes(address, region.RegionSize);
                code = code || (region.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
                                                  PAGE_EXECUTE_READWRITE)) != 0;
            }
        }
        if (reserved == 0) {
            continue;
        }
        text += fmt::format("{}{{\"base\":{},\"reserved\":{},\"committed\":{},\"resident\":{},"
                            "\"code\":{},\"stack\":[",
                            seen.size() == 1 ? "" : ",", base, reserved, committed, resident,
                            code);
        for (size_t frame = 0; frame < Frames && g_regions[i].stack[frame] != nullptr; ++frame) {
            text += fmt::format("{}{}", frame == 0 ? "" : ",",
                                reinterpret_cast<uintptr_t>(g_regions[i].stack[frame]));
        }
        text += "]}";
    }
    text += "],\"groups\":[";
    bool first = true;
    for (const auto& [stack, group] : groups) {
        text += fmt::format("{}{{\"count\":{},\"bytes\":{},\"weight\":{},\"resident\":{},\"stack\":[",
                            first ? "" : ",", group.count, group.bytes, group.weight,
                            group.resident);
        first = false;
        for (size_t i = 0; i < stack.size() && stack[i] != nullptr; ++i) {
            text += fmt::format("{}{}", i == 0 ? "" : ",", reinterpret_cast<uintptr_t>(stack[i]));
        }
        text += "]}";
    }
    text += "]}";
    return text;
}

} // namespace Core
