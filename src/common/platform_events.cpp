// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include "common/host_memory.h"
#if defined(__APPLE__) && TARGET_OS_IPHONE
#include "common/ios_jit.h"
#endif
#include "common/logging.h"
#include "common/platform_events.h"

namespace Common::PlatformEvents {
namespace {
std::atomic<u32> g_pressure_generation{};
std::atomic<u64> g_low_water{};
std::atomic<long long> g_next_poll_ns{};
std::atomic<u32> g_display_max_fps{};
std::atomic<u32> g_thermal_max_fps{};

// Pressure signals from the low-water mark: 1 s apart at first, up to 30 s.
constexpr long long SignalWaitMin = 1'000'000'000LL;
constexpr long long SignalWaitMax = 30'000'000'000LL;
constexpr u64 SignalStep = u64{64} << 20;
std::atomic<long long> g_signal_wait_ns{SignalWaitMin};
std::atomic<long long> g_next_signal_ns{};
std::atomic<u64> g_signal_available{~u64{0}};

std::mutex g_listeners_mutex;
std::vector<std::pair<std::size_t, std::function<void()>>> g_listeners;
std::size_t g_next_listener_id = 1;

std::mutex g_pressure_listeners_mutex;
std::vector<std::pair<std::size_t, std::function<void()>>> g_pressure_listeners;
std::size_t g_next_pressure_listener_id = 1;
} // Anonymous namespace

void SignalMemoryPressure() {
    g_pressure_generation.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::function<void()>> listeners;
    {
        std::scoped_lock lock{g_pressure_listeners_mutex};
        listeners.reserve(g_pressure_listeners.size());
        for (const auto& entry : g_pressure_listeners) {
            listeners.push_back(entry.second);
        }
    }
    for (const auto& listener : listeners) {
        listener();
    }
}

std::size_t AddMemoryPressureListener(std::function<void()> listener) {
    std::scoped_lock lock{g_pressure_listeners_mutex};
    const std::size_t id = g_next_pressure_listener_id++;
    g_pressure_listeners.emplace_back(id, std::move(listener));
    return id;
}

void RemoveMemoryPressureListener(std::size_t id) {
    if (id == 0) {
        return;
    }
    std::scoped_lock lock{g_pressure_listeners_mutex};
    std::erase_if(g_pressure_listeners, [id](const auto& entry) { return entry.first == id; });
}

u32 MemoryPressureGeneration() {
    return g_pressure_generation.load(std::memory_order_relaxed);
}

u64 AvailableProcessMemory() {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return static_cast<u64>(os_proc_available_memory());
#else
    return 0;
#endif
}

void SetMemoryLowWater(u64 bytes) {
    g_low_water.store(bytes, std::memory_order_relaxed);
}

void PollMemoryPressure() {
    const u64 low_water = g_low_water.load(std::memory_order_relaxed);
    if (low_water == 0) {
        return;
    }
    const long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    long long next = g_next_poll_ns.load(std::memory_order_relaxed);
    if (now < next || !g_next_poll_ns.compare_exchange_strong(next, now + 250'000'000LL,
                                                              std::memory_order_relaxed)) {
        return;
    }
    const u64 available = AvailableProcessMemory();
    if (available == 0 || available >= low_water) {
        g_signal_wait_ns.store(SignalWaitMin, std::memory_order_relaxed);
        g_signal_available.store(~u64{0}, std::memory_order_relaxed);
        return;
    }
    // A signal a second freed nothing on a phone that stayed below the mark (the caches were
    // at their floor already) and cost a collection each: while it gets no worse, every signal
    // waits twice as long as the one before.
    const bool worse = available + SignalStep < g_signal_available.load(std::memory_order_relaxed);
    if (!worse && now < g_next_signal_ns.load(std::memory_order_relaxed)) {
        return;
    }
    const long long wait =
        worse ? SignalWaitMin : g_signal_wait_ns.load(std::memory_order_relaxed);
    LOG_WARNING(Common_Memory,
                "{} MiB left to this process (low water {} MiB): collecting (again in {} s, or "
                "with {} MiB less)",
                available >> 20, low_water >> 20, wait / 1'000'000'000LL, SignalStep >> 20);
    SignalMemoryPressure();
    g_signal_available.store(available, std::memory_order_relaxed);
    g_next_signal_ns.store(now + wait, std::memory_order_relaxed);
    g_signal_wait_ns.store((std::min)(wait * 2, SignalWaitMax), std::memory_order_relaxed);
}

ProcessMemory QueryProcessMemory() {
    ProcessMemory memory{};
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS) {
        memory.footprint = info.phys_footprint;
        memory.footprint_peak = info.ledger_phys_footprint_peak;
        memory.compressed = info.compressed;
        memory.reusable = info.reusable;
        memory.internal = info.internal;
        memory.purgeable_nonvolatile = static_cast<u64>(info.ledger_purgeable_nonvolatile);
        memory.external = info.external;
        if (count >= TASK_VM_INFO_REV3_COUNT) {
            memory.graphics = static_cast<u64>(info.ledger_tag_graphics_footprint +
                                               info.ledger_tag_graphics_footprint_compressed);
        }
    }
#elif defined(_WIN32)
    // The closest Windows has: everything resident, guest memory included.
    PROCESS_MEMORY_COUNTERS counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        memory.footprint = counters.WorkingSetSize;
        memory.footprint_peak = counters.PeakWorkingSetSize;
    }
#endif
    return memory;
}

std::string FootprintSplit([[maybe_unused]] const ProcessMemory& memory,
                           [[maybe_unused]] u64 left) {
#ifdef __APPLE__
    // Ledger-tagged guest memory is in the footprint once per resident or compressed page and
    // in none of the ledgers task_info reports; a file's pages are in neither.
    const u64 resident = HostMemory::ScannedResidentBytes();
    // What the compressor holds of it: the rest of "compressed" is the emulator's own memory.
    const u64 compressed = HostMemory::ScannedCompressedBytes();
    u64 guest = resident;
    std::string guest_text;
    switch (HostMemory::ScannedAccounting()) {
    case HostMemory::GuestAccounting::Once:
        guest_text = fmt::format("{} MiB (counted once, {} MiB of it compressed)", resident >> 20,
                                 compressed >> 20);
        break;
    case HostMemory::GuestAccounting::PerMapping:
        guest_text = fmt::format("{} MiB or more (counted per mapping, {} MiB of it compressed)",
                                 resident >> 20, compressed >> 20);
        break;
    case HostMemory::GuestAccounting::File:
        guest = 0;
        guest_text = fmt::format("0 MiB (in a file, {} MiB of it resident)", resident >> 20);
        break;
    case HostMemory::GuestAccounting::Uncounted:
        guest = 0;
        guest_text = "not counted yet";
        break;
    }
    u64 jit = 0;
    std::string jit_text = "none";
#if TARGET_OS_IPHONE
    // An estimate: the pool is ordinary memory, counted per page and per view that touched it.
    // Code is written through one view and run through the other, so a page can count twice
    // (that part is in the rest) or, never touched by this process, not at all (then the rest
    // is that much larger than shown). The "Memory regions" line has what each view holds.
    jit = IosJit::PreparedBytes();
    jit_text = fmt::format("{} MiB assumed ({} handed out)", jit >> 20,
                           IosJit::HandedOutBytes() >> 20);
#endif
    const u64 known = guest + jit + memory.graphics;
    return fmt::format("limit {} MiB | guest {}, JIT pool {}, graphics {} MiB, rest {} MiB",
                       left != 0 ? fmt::format("{}", (memory.footprint + left) >> 20) : "unknown",
                       guest_text, jit_text, memory.graphics >> 20,
                       (memory.footprint - (std::min)(memory.footprint, known)) >> 20);
#else
    return "not known on this OS";
#endif
}

u64 RelieveMallocPressure() {
#ifdef __APPLE__
    return static_cast<u64>(malloc_zone_pressure_relief(nullptr, 0));
#else
    return 0;
#endif
}

#ifdef __APPLE__
namespace {
struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};
std::mutex g_guest_ranges_mutex;
AddressRange g_guest_backing{};
AddressRange g_guest_arena{};

// What a region was allocated for (<mach/vm_statistics.h>, VM_MEMORY_*), for the tags an app has.
const char* RegionTagName(u32 tag) {
    switch (tag) {
    case 0:
        return "untagged";
    case 1:
        return "MALLOC";
    case 2:
        return "MALLOC_SMALL";
    case 3:
        return "MALLOC_LARGE";
    case 4:
        return "MALLOC_HUGE";
    case 6:
        return "REALLOC";
    case 7:
        return "MALLOC_TINY";
    case 8:
        return "MALLOC_LARGE_REUSABLE";
    case 9:
        return "MALLOC_LARGE_REUSED";
    case 11:
        return "MALLOC_NANO";
    case 12:
        return "MALLOC_MEDIUM";
    case 13:
        return "MALLOC_PROB_GUARD";
    case 20:
        return "MACH_MSG";
    case 21:
        return "IOKIT";
    case 30:
        return "STACK";
    case 31:
        return "GUARD";
    case 32:
        return "SHARED_PMAP";
    case 33:
        return "DYLIB";
    case 35:
        return "UNSHARED_PMAP";
    case 41:
        return "FOUNDATION";
    case 42:
        return "COREGRAPHICS";
    case 51:
        return "COREANIMATION";
    case 52:
        return "CGIMAGE";
    case 54:
        return "COREGRAPHICS_DATA";
    case 60:
        return "DYLD";
    case 61:
        return "DYLD_MALLOC";
    case 62:
        return "SQLITE";
    case 66:
        return "GLSL";
    case 70:
        return "IMAGEIO";
    case 73:
        return "OS_ALLOC_ONCE";
    case 74:
        return "LIBDISPATCH";
    case 75:
        return "ACCELERATE";
    case 76:
        return "COREUI";
    case 78:
        return "ACTIVITY_TRACING";
    case 82:
        return "SWIFT_RUNTIME";
    case 83:
        return "SWIFT_METADATA";
    case 88:
        return "IOSURFACE";
    case 89:
        return "LIBNETWORK";
    case 90:
        return "AUDIO";
    case 91:
        return "VIDEOBITSTREAM";
    case 99:
        return "SANITIZER";
    case 100:
        return "IOACCELERATOR";
    default:
        return nullptr;
    }
}

struct RegionUse {
    u64 counted{};  ///< Dirty or compressed: what the footprint is charged
    u64 resident{};
    u64 mapped{};
};

// The 256 tags, then what is told apart by other means.
constexpr std::size_t KindFiles = 256;
constexpr std::size_t KindJitCode = 257;
constexpr std::size_t KindJitWrite = 258;
constexpr std::size_t NumKinds = 259;

std::string KindName(std::size_t kind) {
    if (kind == KindFiles) {
        return "mapped files";
    }
    const char* const name = RegionTagName(static_cast<u32>(kind));
    return name != nullptr ? std::string{name} : fmt::format("tag {}", kind);
}

u32 HostThreadCount() {
    thread_act_array_t threads = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) {
        return 0;
    }
    for (mach_msg_type_number_t i = 0; i < count; ++i) {
        mach_port_deallocate(mach_task_self(), threads[i]);
    }
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                  count * sizeof(thread_act_t));
    return count;
}

#if TARGET_OS_IPHONE
// What Metal has allocated for this process on the GPU MoltenVK renders with (a phone has one).
// Looked up at run time: not everything that links this file links Metal.
u64 MetalAllocatedBytes() {
    using CreateDevice = void* (*)();
    using RegisterName = void* (*)(const char*);
    using Getter = unsigned long (*)(void*, void*);
    static void* const device = [] {
        const auto create = reinterpret_cast<CreateDevice>(
            dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice"));
        return create != nullptr ? create() : nullptr;
    }();
    static const auto send = reinterpret_cast<Getter>(dlsym(RTLD_DEFAULT, "objc_msgSend"));
    static const auto register_name =
        reinterpret_cast<RegisterName>(dlsym(RTLD_DEFAULT, "sel_registerName"));
    if (device == nullptr || send == nullptr || register_name == nullptr) {
        return 0;
    }
    return send(device, register_name("currentAllocatedSize"));
}
#endif

void LogRegions(const char* reason, const ProcessMemory& memory) {
    const auto start = std::chrono::steady_clock::now();
    AddressRange backing;
    AddressRange arena;
    {
        std::scoped_lock lock{g_guest_ranges_mutex};
        backing = g_guest_backing;
        arena = g_guest_arena;
    }
    // With vm.self_region_footprint on, the kernel answers from the page tables with what the
    // footprint is charged (xnu vm_map_region_walk) and adds a last region for the memory billed
    // to the task as its owner: ledger-tagged guest memory, Metal, IOSurface. The app sandbox
    // may refuse the switch: the numbers are then the objects' dirty and compressed pages, which
    // count a page once per mapping of it.
    int enable = 1;
    const bool footprint_view =
        sysctlbyname("vm.self_region_footprint", nullptr, nullptr, &enable, sizeof(enable)) == 0;
    const int view_error = footprint_view ? 0 : errno;

    constexpr natural_t MaxDepth = 8;
    constexpr u32 MaxRegions = 100'000;
    // Every page costs a lookup under its object's lock: guest memory is left out by address,
    // and nothing else this large holds data.
    constexpr vm_size_t MaxWalk = vm_size_t{8} << 30;
    constexpr u32 OwnedTag = ~u32{0};
    const u64 page = vm_page_size;
    std::array<RegionUse, NumKinds> kinds{};
    struct Large {
        u64 counted;
        std::size_t kind;
    };
    std::array<Large, 4> largest{};
    u64 owned = 0;
    u64 skipped = 0;
    u32 regions = 0;
    vm_address_t address = 0;
    for (; regions < MaxRegions; ++regions) {
        // First without the pages, which is cheap.
        vm_address_t at = address;
        vm_size_t size = 0;
        natural_t depth = MaxDepth;
        vm_region_submap_short_info_data_64_t brief{};
        mach_msg_type_number_t count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
        if (vm_region_recurse_64(mach_task_self(), &at, &size, &depth,
                                 reinterpret_cast<vm_region_recurse_info_t>(&brief),
                                 &count) != KERN_SUCCESS) {
            break;
        }
        const std::uintptr_t end = at + size;
        const auto within = [at, end](const AddressRange& range) {
            return at < range.end && end > range.begin;
        };
        if (within(backing)) {
            address = (std::max)(end, backing.end);
            continue;
        }
        if (within(arena)) {
            address = (std::max)(end, arena.end);
            continue;
        }
        if (brief.user_tag == OwnedTag) {
            owned = size;
            address = end;
            continue;
        }
        if (brief.is_submap || size > MaxWalk) {
            skipped += size;
            address = end;
            continue;
        }
        vm_address_t full_at = at;
        vm_size_t length = 0;
        depth = MaxDepth;
        vm_region_submap_info_data_64_t info{};
        count = VM_REGION_SUBMAP_INFO_COUNT_64;
        if (vm_region_recurse_64(mach_task_self(), &full_at, &length, &depth,
                                 reinterpret_cast<vm_region_recurse_info_t>(&info),
                                 &count) != KERN_SUCCESS) {
            break;
        }
        if (full_at != at) {
            // The map changed in between.
            address = (std::max)(full_at + length, end);
            continue;
        }
        address = at + length;
        if (info.user_tag == OwnedTag) {
            owned = length;
            continue;
        }
        std::size_t kind = (std::min<std::size_t>)(info.user_tag, 255);
        if (info.external_pager != 0) {
            kind = KindFiles;
        }
#if TARGET_OS_IPHONE
        if (const auto pool = IosJit::Locate(reinterpret_cast<const void*>(at))) {
            kind = pool->writable_view ? KindJitWrite : KindJitCode;
        }
#endif
        const u64 counted = (u64{info.pages_dirtied} + info.pages_swapped_out) * page;
        kinds[kind].counted += counted;
        kinds[kind].resident += u64{info.pages_resident} * page;
        kinds[kind].mapped += length;
        if (kind < KindJitCode && counted > largest.back().counted) {
            largest.back() = Large{counted, kind};
            std::sort(largest.begin(), largest.end(),
                      [](const Large& a, const Large& b) { return a.counted > b.counted; });
        }
    }
    if (footprint_view) {
        int disable = 0;
        sysctlbyname("vm.self_region_footprint", nullptr, nullptr, &disable, sizeof(disable));
    }

    u64 listed = owned;
    std::vector<std::size_t> order;
    for (std::size_t kind = 0; kind < NumKinds; ++kind) {
        listed += kinds[kind].counted;
        if (kind < KindJitCode && kinds[kind].counted + kinds[kind].resident != 0) {
            order.push_back(kind);
        }
    }
    std::sort(order.begin(), order.end(), [&kinds](std::size_t a, std::size_t b) {
        return kinds[a].counted > kinds[b].counted;
    });
    constexpr std::size_t Shown = 8;
    std::string by_kind;
    RegionUse others;
    for (std::size_t i = 0; i < order.size(); ++i) {
        const RegionUse& use = kinds[order[i]];
        if (i < Shown) {
            by_kind += fmt::format("{}{} {}/{}", i != 0 ? ", " : "", KindName(order[i]),
                                   use.counted >> 20, use.resident >> 20);
        } else {
            others.counted += use.counted;
            others.resident += use.resident;
        }
    }
    std::string large;
    for (const Large& region : largest) {
        if (region.counted != 0) {
            large += fmt::format("{}{} {}", large.empty() ? "" : ", ", KindName(region.kind),
                                 region.counted >> 20);
        }
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    LOG_INFO(Common_Memory,
             "Memory regions ({}; {}), MiB counted/resident: {}, others {}/{}", reason,
             footprint_view ? std::string{"footprint view"}
                            : fmt::format("object view, the footprint view was refused: {}",
                                          std::strerror(view_error)),
             by_kind.empty() ? "none" : by_kind.c_str(), others.counted >> 20,
             others.resident >> 20);
    LOG_INFO(Common_Memory,
             "Memory regions: JIT code {}/{} MiB, JIT write view {}/{} MiB, billed as owner {} "
             "(guest memory, Metal, IOSurface), listed {} of footprint {} MiB | largest: {} | "
             "{} regions in {} ms, guest memory left out, {} MiB too large to look at",
             kinds[KindJitCode].counted >> 20, kinds[KindJitCode].resident >> 20,
             kinds[KindJitWrite].counted >> 20, kinds[KindJitWrite].resident >> 20,
             footprint_view ? fmt::format("{} MiB", owned >> 20) : std::string{"not in this view"},
             listed >> 20, memory.footprint >> 20, large.empty() ? "none" : large.c_str(), regions,
             ms, skipped >> 20);
}
} // Anonymous namespace
#endif

void SetGuestMemoryRanges([[maybe_unused]] const void* backing,
                          [[maybe_unused]] std::size_t backing_size,
                          [[maybe_unused]] const void* arena,
                          [[maybe_unused]] std::size_t arena_size) {
#ifdef __APPLE__
    const auto range = [](const void* base, std::size_t size) {
        const auto begin = reinterpret_cast<std::uintptr_t>(base);
        return base != nullptr && size != 0 ? AddressRange{begin, begin + size} : AddressRange{};
    };
    std::scoped_lock lock{g_guest_ranges_mutex};
    g_guest_backing = range(backing, backing_size);
    g_guest_arena = range(arena, arena_size);
#endif
}

void LogMemoryBreakdown([[maybe_unused]] const char* reason) {
#ifdef __APPLE__
    const ProcessMemory memory = QueryProcessMemory();
    // Walking 34,000+ Mach VM regions takes 1,100-2,550 ms holding the task's vm_map lock, causing
    // multi-second freezes during gameplay and memory-warning bursts. QueryProcessMemory(),
    // malloc_zone_statistics(), and MetalAllocatedBytes() below already report exact footprint,
    // malloc, and Metal totals in < 0.1 ms, so only walk individual VM regions when explicitly
    // requested via EDEN_IOS_MEM_REGIONS=1.
    static const bool force_regions = [] {
        const char* const env = std::getenv("EDEN_IOS_MEM_REGIONS");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    if (force_regions) {
        LogRegions(reason, memory);
    }
    // In use is what the emulator holds; the rest of what the zones took from the OS is free
    // inside them (malloc_zone_pressure_relief gives back the whole pages of it).
    malloc_statistics_t zones{};
    malloc_zone_statistics(nullptr, &zones);
    u64 metal = 0;
#if TARGET_OS_IPHONE
    metal = MetalAllocatedBytes();
#endif
    LOG_INFO(Common_Memory,
             "Memory malloc: {} MiB in use, {} MiB taken from the OS ({} MiB free inside), {} "
             "host threads | Metal has {} MiB allocated, graphics in the footprint {} MiB",
             zones.size_in_use >> 20, zones.size_allocated >> 20,
             (zones.size_allocated - (std::min)(zones.size_allocated, zones.size_in_use)) >> 20,
             HostThreadCount(), metal >> 20, memory.graphics >> 20);
    LogStorageWrites(reason);
#endif
}

void LogStorageWrites([[maybe_unused]] const char* reason) {
#ifdef __APPLE__
    // Declared in <libproc.h>, which the iOS SDK lacks; the function is in libSystem all the same.
    using ProcessUsage = int (*)(int pid, int flavor, rusage_info_t* buffer);
    static const auto process_usage =
        reinterpret_cast<ProcessUsage>(dlsym(RTLD_DEFAULT, "proc_pid_rusage"));
    if (process_usage == nullptr) {
        LOG_INFO(Common_Memory, "Storage ({}): not known: no proc_pid_rusage", reason);
        return;
    }
    rusage_info_v4 usage{};
    if (process_usage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&usage)) != 0) {
        LOG_INFO(Common_Memory, "Storage ({}): not known: {}", reason, std::strerror(errno));
        return;
    }
    // ri_logical_writes also counts every page of a mapped file at its first write, long before
    // (and whether or not) the page reaches storage.
    const bool file = HostMemory::ScannedAccounting() == HostMemory::GuestAccounting::File;
    LOG_INFO(Common_Memory,
             "Storage ({}): {} MiB written by this process since it started, {} MiB of writes "
             "asked for (file pages made dirty included) | guest memory {}",
             reason, usage.ri_diskio_byteswritten >> 20, usage.ri_logical_writes >> 20,
             file ? fmt::format("file: {} MiB of it in RAM",
                                HostMemory::ScannedResidentBytes() >> 20)
                  : std::string{"is not in a file"});
#endif
}

std::size_t AddBackgroundListener(std::function<void()> listener) {
    std::scoped_lock lock{g_listeners_mutex};
    const std::size_t id = g_next_listener_id++;
    g_listeners.emplace_back(id, std::move(listener));
    return id;
}

void RemoveBackgroundListener(std::size_t id) {
    std::scoped_lock lock{g_listeners_mutex};
    std::erase_if(g_listeners, [id](const auto& entry) { return entry.first == id; });
}

void NotifyEnteringBackground() {
    std::scoped_lock lock{g_listeners_mutex};
    for (const auto& entry : g_listeners) {
        entry.second();
    }
}

void SetDisplayMaxFps(u32 fps) {
    g_display_max_fps.store(fps, std::memory_order_relaxed);
}

u32 DisplayMaxFps() {
    return g_display_max_fps.load(std::memory_order_relaxed);
}

void SetThermalMaxFps(u32 fps) {
    const u32 prev = g_thermal_max_fps.exchange(fps, std::memory_order_relaxed);
    if (prev != fps) {
        if (fps != 0) {
            LOG_INFO(Common, "Auto-Cool: capping live frame rate to {} FPS", fps);
        } else {
            LOG_INFO(Common, "Auto-Cool: thermal state normal, restoring full frame rate");
        }
    }
}

u32 ThermalMaxFps() {
    return g_thermal_max_fps.load(std::memory_order_relaxed);
}

} // namespace Common::PlatformEvents
