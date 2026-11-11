// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32

#include <iterator>
#include <shared_mutex>
#include "common/container/unordered_map.h"
#include <boost/icl/separate_interval_set.hpp>
#include <windows.h>
#include <psapi.h>
#include <vector>
#include "common/dynamic_library.h"

#else // ^^^ Windows ^^^ vvv POSIX vvv

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/icl/interval_set.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common/scope_exit.h"
#include "common/settings.h"

#if defined(__linux__)
#include <sys/random.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE
#include <sys/random.h>
#endif
#include <mach/vm_map.h>
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <mach/memory_object_types.h>
#include <mach/vm_statistics.h>
#if TARGET_OS_IPHONE
#include <mach/vm_purgable.h>
#include <sys/mount.h>
#include <sys/param.h>
// iOS forbids POSIX shared memory for third-party apps; the guest is backed by an anonymous Mach
// memory object and every fastmem view is a vm_map of it.
#define LICHII_IOS_MACH_MEMORY 1
#endif
#elif defined(__FreeBSD__)
#include <sys/shm.h>
#elif defined(__OPENORBIS__)
#include <orbis/libkernel.h>
#endif

// FreeBSD
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
// Solaris 11 and illumos
#ifndef MAP_ALIGNED_SUPER
#define MAP_ALIGNED_SUPER 0
#endif
// macOS
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#endif // ^^^ POSIX ^^^

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/free_region_manager.h"
#include "common/guest_chunks.h"
#include "common/host_memory.h"
#include <fmt/format.h>
#include "common/logging.h"

#if defined(__ANDROID__) && __ANDROID_API__ < 30
#include <sys/syscall.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
static int memfd_create(const char* name, unsigned int flags) {
    return syscall(__NR_memfd_create, name, flags);
}
#endif

namespace Common {

[[maybe_unused]] constexpr size_t PageAlignment = 0x1000;
[[maybe_unused]] constexpr int BackingAttempts = 5; // fastmem backing allocation tries (Windows)
[[maybe_unused]] constexpr size_t HugePageSize = 0x200000;

#ifdef _WIN32

// Manually imported for MinGW compatibility
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

using PFN_CreateFileMapping2 = _Ret_maybenull_ HANDLE(WINAPI*)(
    _In_ HANDLE File, _In_opt_ SECURITY_ATTRIBUTES* SecurityAttributes, _In_ ULONG DesiredAccess,
    _In_ ULONG PageProtection, _In_ ULONG AllocationAttributes, _In_ ULONG64 MaximumSize,
    _In_opt_ PCWSTR Name,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_VirtualAlloc2 = _Ret_maybenull_ PVOID(WINAPI*)(
    _In_opt_ HANDLE Process, _In_opt_ PVOID BaseAddress, _In_ SIZE_T Size,
    _In_ ULONG AllocationType, _In_ ULONG PageProtection,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_MapViewOfFile3 = _Ret_maybenull_ PVOID(WINAPI*)(
    _In_ HANDLE FileMapping, _In_opt_ HANDLE Process, _In_opt_ PVOID BaseAddress,
    _In_ ULONG64 Offset, _In_ SIZE_T ViewSize, _In_ ULONG AllocationType, _In_ ULONG PageProtection,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_UnmapViewOfFile2 = BOOL(WINAPI*)(_In_ HANDLE Process, _In_ PVOID BaseAddress,
                                           _In_ ULONG UnmapFlags);

template <typename T>
static void GetFuncAddress(Common::DynamicLibrary& dll, const char* name, T& pfn) {
    if (!dll.GetSymbol(name, &pfn)) {
        LOG_CRITICAL(HW_Memory, "Failed to load {}", name);
    }
}

class HostMemory::Impl {
public:
    explicit Impl(size_t backing_size_, size_t virtual_size_)
        : backing_size{backing_size_}
        , virtual_size{virtual_size_}
        , process{GetCurrentProcess()}
        , kernelbase_dll("Kernelbase")
    {}

    bool Init() {
        if (!kernelbase_dll.IsOpen()) {
            LOG_CRITICAL(HW_Memory, "Failed to load Kernelbase.dll");
            return false;
        }
        GetFuncAddress(kernelbase_dll, "CreateFileMapping2", pfn_CreateFileMapping2);
        GetFuncAddress(kernelbase_dll, "VirtualAlloc2", pfn_VirtualAlloc2);
        GetFuncAddress(kernelbase_dll, "MapViewOfFile3", pfn_MapViewOfFile3);
        GetFuncAddress(kernelbase_dll, "UnmapViewOfFile2", pfn_UnmapViewOfFile2);

        if (!pfn_CreateFileMapping2 || !pfn_VirtualAlloc2 || !pfn_MapViewOfFile3 || !pfn_UnmapViewOfFile2) {
            LOG_CRITICAL(HW_Memory, "Failed to find functions for virtual allocs");
            return false;
        }

        // Allocate backing file map. SEC_COMMIT charges all of it to the commit limit up front, and
        // when the free commit is only just above its size the call fails with
        // ERROR_NO_SYSTEM_RESOURCES without growing a system-managed page file. The emulator then
        // runs the whole session without fastmem (Ocarina of Time: 20-25% fewer fps). A private
        // commit of the same size does grow the page file, so make one, release it and try again.
        for (int attempt = 1; attempt <= BackingAttempts && !backing_handle; ++attempt) {
            backing_handle = pfn_CreateFileMapping2(INVALID_HANDLE_VALUE, nullptr, FILE_MAP_WRITE | FILE_MAP_READ, PAGE_READWRITE, SEC_COMMIT, backing_size, nullptr, nullptr, 0);
            if (backing_handle) {
                if (attempt > 1) {
                    LOG_INFO(HW_Memory, "Allocated {} MiB of backing memory on attempt {}", backing_size >> 20, attempt);
                }
                break;
            }
            const DWORD error = GetLastError();
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            LOG_WARNING(HW_Memory, "Failed to allocate {} MiB of backing memory (attempt {}, error {}, {} MiB of commit free)",
                        backing_size >> 20, attempt, error, status.ullAvailPageFile >> 20);
            if (void* const probe = VirtualAlloc(nullptr, backing_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
                VirtualFree(probe, 0, MEM_RELEASE);
            }
            if (attempt < BackingAttempts) {
                Sleep(50);
            }
        }
        if (!backing_handle) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate {} MiB of backing memory", backing_size >> 20);
            return false;
        }
        // Allocate a virtual memory for the backing file map as placeholder
        backing_base = static_cast<u8*>(pfn_VirtualAlloc2(process, nullptr, backing_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
        if (!backing_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to reserve {} MiB of virtual memory", backing_size >> 20);
            return false;
        }
        // Map backing placeholder
        void* const ret = pfn_MapViewOfFile3(backing_handle, process, backing_base, 0, backing_size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
        if (ret != backing_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to map {} MiB of virtual memory", backing_size >> 20);
            return false;
        }
        // Allocate virtual address placeholder
        for (int attempt = 0; attempt < 8 && !virtual_base; ++attempt) {
            if (attempt > 0) {
                Sleep(10);
            }
            virtual_base = static_cast<u8*>(pfn_VirtualAlloc2(
                process, nullptr, virtual_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS, nullptr, 0));
        }
        if (!virtual_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to reserve {} GiB of virtual memory", virtual_size >> 30);
            return false;
        }
        return true;
    }

    ~Impl() {
        Release();
    }

    void* Allocate(size_t size) {
        auto* ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (ptr == nullptr) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate fallback buffer with size {:#x}, error {}", size, GetLastError());
        }
        return ptr;
    }

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms) {
        std::unique_lock lock{placeholder_mutex};
        if (!IsNiechePlaceholder(virtual_offset, length)) {
            Split(virtual_offset, length);
        }
        ASSERT(placeholders.find({virtual_offset, virtual_offset + length}) == placeholders.end());
        TrackPlaceholder(virtual_offset, host_offset, length);

        MapView(virtual_offset, host_offset, length);
    }

    void Unmap(size_t virtual_offset, size_t length) {
        std::scoped_lock lock{placeholder_mutex};

        // Unmap until there are no more placeholders
        while (UnmapOnePlaceholder(virtual_offset, length)) {
        }
    }

    void Protect(size_t virtual_offset, size_t length, bool read, bool write, bool execute) {
        DWORD new_flags{};
        if (read && write) {
            new_flags = PAGE_READWRITE;
        } else if (read && !write) {
            new_flags = PAGE_READONLY;
        } else if (!read && !write) {
            new_flags = PAGE_NOACCESS;
        } else {
            UNIMPLEMENTED_MSG("Protection flag combination read={} write={}", read, write);
        }
        const size_t virtual_end = virtual_offset + length;

        const auto protect = [&](auto it, auto end) {
            while (it != end) {
                const size_t offset = (std::max)(it->lower(), virtual_offset);
                const size_t protect_length = (std::min)(it->upper(), virtual_end) - offset;
                DWORD old_flags{};
                if (!VirtualProtect(virtual_base + offset, protect_length, new_flags, &old_flags)) {
                    LOG_CRITICAL(HW_Memory, "Failed to change virtual memory protect rules");
                }
                ++it;
            }
        };
        {
            std::shared_lock lock{placeholder_mutex};
            const auto [it, end] = placeholders.equal_range({virtual_offset, virtual_end});
            if (it == end || std::next(it) == end) {
                protect(it, end);
                return;
            }
        }
        // A range over several placeholders takes a call for each: exclusively, so that other
        // threads see one change, as they see one call.
        std::scoped_lock lock{placeholder_mutex};
        const auto [it, end] = placeholders.equal_range({virtual_offset, virtual_end});
        protect(it, end);
    }

    /// Backing offset the view page at virtual_offset maps, or nullopt when it is not mapped.
    std::optional<size_t> BackingOffset(size_t virtual_offset) {
        std::shared_lock lock{placeholder_mutex};
        const auto it = placeholders.find(virtual_offset);
        if (it == placeholders.end()) {
            return std::nullopt;
        }
        const auto host_pointer_it = placeholder_host_pointers.find(it->lower());
        if (host_pointer_it == placeholder_host_pointers.end()) {
            return std::nullopt;
        }
        return host_pointer_it->second + virtual_offset - it->lower();
    }

    void EnableDirectMappedAddress() {
        // TODO
        UNREACHABLE();
    }

    const size_t backing_size; ///< Size of the backing memory in bytes
    const size_t virtual_size; ///< Size of the virtual address placeholder in bytes

    u8* backing_base{};
    u8* virtual_base{};

private:
    /// Release all resources in the object
    void Release() {
        if (!placeholders.empty()) {
            for (const auto& placeholder : placeholders) {
                if (!pfn_UnmapViewOfFile2(process, virtual_base + placeholder.lower(),
                                          MEM_PRESERVE_PLACEHOLDER)) {
                    LOG_CRITICAL(HW_Memory, "Failed to unmap virtual memory placeholder");
                }
            }
            Coalesce(0, virtual_size);
        }
        if (virtual_base) {
            if (!VirtualFree(virtual_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(HW_Memory, "Failed to free virtual memory");
            }
        }
        if (backing_base) {
            if (!pfn_UnmapViewOfFile2(process, backing_base, MEM_PRESERVE_PLACEHOLDER)) {
                LOG_CRITICAL(HW_Memory, "Failed to unmap backing memory placeholder");
            }
            if (!VirtualFreeEx(process, backing_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(HW_Memory, "Failed to free backing memory");
            }
        }
        if (!CloseHandle(backing_handle)) {
            LOG_CRITICAL(HW_Memory, "Failed to free backing memory file handle");
        }
    }

    /// Unmap one placeholder in the given range (partial unmaps are supported)
    /// Return true when there are no more placeholders to unmap
    bool UnmapOnePlaceholder(size_t virtual_offset, size_t length) {
        const auto it = placeholders.find({virtual_offset, virtual_offset + length});
        const auto begin = placeholders.begin();
        const auto end = placeholders.end();
        if (it == end) {
            return false;
        }
        const size_t placeholder_begin = it->lower();
        const size_t placeholder_end = it->upper();
        const size_t unmap_begin = (std::max)(virtual_offset, placeholder_begin);
        const size_t unmap_end = (std::min)(virtual_offset + length, placeholder_end);
        ASSERT(unmap_begin >= placeholder_begin && unmap_begin < placeholder_end);
        ASSERT(unmap_end <= placeholder_end && unmap_end > placeholder_begin);

        const auto host_pointer_it = placeholder_host_pointers.find(placeholder_begin);
        ASSERT(host_pointer_it != placeholder_host_pointers.end());
        const size_t host_offset = host_pointer_it->second;

        const bool split_left = unmap_begin > placeholder_begin;
        const bool split_right = unmap_end < placeholder_end;

        if (!pfn_UnmapViewOfFile2(process, virtual_base + placeholder_begin,
                                  MEM_PRESERVE_PLACEHOLDER)) {
            LOG_CRITICAL(HW_Memory, "Failed to unmap placeholder");
        }
        // If we have to remap memory regions due to partial unmaps, we are in a data race as
        // Windows doesn't support remapping memory without unmapping first. Avoid adding any extra
        // logic within the panic region described below.

        // Panic region, we are in a data race right now
        if (split_left || split_right) {
            Split(unmap_begin, unmap_end - unmap_begin);
        }
        if (split_left) {
            MapView(placeholder_begin, host_offset, unmap_begin - placeholder_begin);
        }
        if (split_right) {
            MapView(unmap_end, host_offset + unmap_end - placeholder_begin,
                    placeholder_end - unmap_end);
        }
        // End panic region

        size_t coalesce_begin = unmap_begin;
        if (!split_left) {
            // Try to coalesce pages to the left
            coalesce_begin = it == begin ? 0 : std::prev(it)->upper();
            if (coalesce_begin != placeholder_begin) {
                Coalesce(coalesce_begin, unmap_end - coalesce_begin);
            }
        }
        if (!split_right) {
            // Try to coalesce pages to the right
            const auto next = std::next(it);
            const size_t next_begin = next == end ? virtual_size : next->lower();
            if (placeholder_end != next_begin) {
                // We can coalesce to the right
                Coalesce(coalesce_begin, next_begin - coalesce_begin);
            }
        }
        // Remove and reinsert placeholder trackers
        UntrackPlaceholder(it);
        if (split_left) {
            TrackPlaceholder(placeholder_begin, host_offset, unmap_begin - placeholder_begin);
        }
        if (split_right) {
            TrackPlaceholder(unmap_end, host_offset + unmap_end - placeholder_begin,
                             placeholder_end - unmap_end);
        }
        return true;
    }

    void MapView(size_t virtual_offset, size_t host_offset, size_t length) {
        if (!pfn_MapViewOfFile3(backing_handle, process, virtual_base + virtual_offset, host_offset,
                                length, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) {
            LOG_CRITICAL(HW_Memory, "Failed to map placeholder");
        }
    }

    void Split(size_t virtual_offset, size_t length) {
        if (!VirtualFreeEx(process, reinterpret_cast<LPVOID>(virtual_base + virtual_offset), length,
                           MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            LOG_CRITICAL(HW_Memory, "Failed to split placeholder");
        }
    }

    void Coalesce(size_t virtual_offset, size_t length) {
        if (!VirtualFreeEx(process, reinterpret_cast<LPVOID>(virtual_base + virtual_offset), length,
                           MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
            LOG_CRITICAL(HW_Memory, "Failed to coalesce placeholders");
        }
    }

    void TrackPlaceholder(size_t virtual_offset, size_t host_offset, size_t length) {
        placeholders.insert({virtual_offset, virtual_offset + length});
        placeholder_host_pointers.emplace(virtual_offset, host_offset);
    }

    void UntrackPlaceholder(boost::icl::separate_interval_set<size_t>::iterator it) {
        placeholder_host_pointers.erase(it->lower());
        placeholders.erase(it);
    }

    /// Return true when a given memory region is a "nieche" and the placeholders don't have to be
    /// split.
    bool IsNiechePlaceholder(size_t virtual_offset, size_t length) const {
        const auto it = placeholders.upper_bound({virtual_offset, virtual_offset + length});
        if (it != placeholders.end() && it->lower() == virtual_offset + length) {
            return it == placeholders.begin() ? virtual_offset == 0
                                              : std::prev(it)->upper() == virtual_offset;
        }
        return false;
    }

    HANDLE process{};        ///< Current process handle
    HANDLE backing_handle{}; ///< File based backing memory

    DynamicLibrary kernelbase_dll;
    PFN_CreateFileMapping2 pfn_CreateFileMapping2{};
    PFN_VirtualAlloc2 pfn_VirtualAlloc2{};
    PFN_MapViewOfFile3 pfn_MapViewOfFile3{};
    PFN_UnmapViewOfFile2 pfn_UnmapViewOfFile2{};

    std::shared_mutex placeholder_mutex;                          ///< Mutex for placeholders
    boost::icl::separate_interval_set<size_t> placeholders;       ///< Mapped placeholders
    ::Common::unordered_map<size_t, size_t> placeholder_host_pointers; ///< Placeholder backing offset
};

#elif defined(__OPENORBIS__) || defined(__managarm__)
// None of the luxuries of POSIX, all of the suffering
// For managarm: see https://github.com/managarm/managarm/issues/1370
#else // ^^^ Windows ^^^ vvv POSIX vvv

#ifndef MAP_NOCORE
#define MAP_NOCORE 0
#endif

#if defined(ARCHITECTURE_arm64) && !defined(__APPLE__)

static void* ChooseVirtualBase(size_t virtual_size) {
    constexpr uintptr_t Map39BitSize = (1ULL << 39);
    constexpr uintptr_t Map36BitSize = (1ULL << 36);

    // This is not a cryptographic application, we just want something random.
    std::mt19937_64 rng;

    // We want to ensure we are allocating at an address aligned to the L2 block size.
    // For Qualcomm devices, we must also allocate memory above 36 bits.
    const size_t lower = Map36BitSize / HugePageSize;
    const size_t upper = (Map39BitSize - virtual_size) / HugePageSize;
    const size_t range = upper - lower;

    // Try up to 64 times to allocate memory at random addresses in the range.
    for (int i = 0; i < 64; i++) {
        // Calculate a possible location.
        uintptr_t hint_address = ((rng() % range) + lower) * HugePageSize;

        // Try to map.
        // Note: we may be able to take advantage of MAP_FIXED_NOREPLACE here.
        void* map_pointer =
            mmap(reinterpret_cast<void*>(hint_address), virtual_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_NOCORE, -1, 0);

        // If we successfully mapped, we're done.
        if (reinterpret_cast<uintptr_t>(map_pointer) == hint_address) {
            return map_pointer;
        }

        // Unmap if necessary, and try again.
        if (map_pointer != MAP_FAILED) {
            munmap(map_pointer, virtual_size);
        }
    }

    return MAP_FAILED;
}

#else

static void* ChooseVirtualBase(size_t virtual_size) {
#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__OpenBSD__) || defined(__sun__) || defined(__HAIKU__) || defined(__managarm__) || defined(__AIX__)
    void* virtual_base = mmap(nullptr, virtual_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_ALIGNED_SUPER | MAP_NOCORE, -1, 0);
    if (virtual_base != MAP_FAILED)
        return virtual_base;
#endif
    return mmap(nullptr, virtual_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_NOCORE, -1, 0);
}

#endif

#if defined(__sun__) || defined(__HAIKU__) || defined(__NetBSD__) || defined(__DragonFly__)
/// Most Unices don't have a portable shm_open (AIX, OpenBSD, NetBSD, Solaris 11, OpenIndiana)
/// Portable implementation of shm_open(SHM_ANON, ...) - roughly equivalent but without
/// OS support - may fail sporadically, beware!
static int shm_open_anon(int flags, mode_t mode) {
    char name[16] = "/shm-";
    char *const limit = name + sizeof(name) - 1;
    *limit = '\0';
    char *start = name + strlen(name);
    for (int tries = 0; tries < 4; tries++) {
        struct timespec tv;
        clock_gettime(CLOCK_REALTIME, &tv);
        unsigned long r = (unsigned long)tv.tv_sec + (unsigned long)tv.tv_nsec;
        for (char *fill = start; fill < limit; r /= 8)
            *fill++ = '0' + (r % 8);
        int fd = shm_open(name, flags, mode);
        if (fd != -1) {
            if (shm_unlink(name) == -1) {
                int tmp = errno;
                close(fd);
                errno = tmp;
                return -1;
            }
            return fd;
        }
        if (errno != EEXIST)
            break;
    }
    return -1;
}
#elif defined(__OpenBSD__)
/// Except OpenBSD which explicitly uses shm_mkstemp instead (as a more secure alternative)
static int shm_open_anon(int flags, mode_t mode) {
    char name[16] = "/shm-XXXXXXXXXX";
    int fd;
    if ((fd = shm_mkstemp(name)) == -1)
        return -1;
    if (shm_unlink(name) == -1) {
        int tmp = errno;
        close(fd);
        errno = tmp;
        return -1;
    }
    return fd;
}
#endif

class HostMemory::Impl {
public:
    explicit Impl(size_t backing_size_, size_t virtual_size_)
        : backing_size{backing_size_}
        , virtual_size{virtual_size_}
    {}

#ifdef __linux__
    bool PunchHole(size_t offset, size_t length) {
        return fd >= 0 &&
               fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(offset),
                         static_cast<off_t>(length)) == 0;
    }
#endif

    bool Init() {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size < static_cast<long>(PageAlignment) ||
            page_size % static_cast<long>(PageAlignment) != 0) {
            LOG_WARNING(Common_Memory, "Host page size {:#x} does not support fastmem", page_size);
            return false;
        }
#ifdef HAS_NCE
        if (page_size != static_cast<long>(PageAlignment) && Settings::IsNceEnabled()) {
            LOG_WARNING(Common_Memory, "Host page size {:#x} does not support fastmem with NCE", page_size);
            return false;
        }
#endif
        host_page_size = static_cast<size_t>(page_size);
        // Backing memory initialization
#ifdef LICHII_IOS_MACH_MEMORY
        {
            // XNU charges an untagged object's page to the footprint once per page table entry
            // (arm64 pmap_enter), and the guest memory has two views: the backing (emulator) and
            // the fastmem arena (guest). A ledger-tagged object is billed to its owner once per
            // resident or compressed page instead and its mappings are left out of the pmap
            // ledgers (vm_map_enter: use_pmap = FALSE). Tagging for our own task needs no
            // entitlement. MADV_FREE_REUSABLE does nothing on tagged memory, so zero trimming
            // only applies untagged: LICHII_IOS_UNTAGGED_GUEST=1 (experiment) goes back to that.
            const char* const untagged_env = std::getenv("LICHII_IOS_UNTAGGED_GUEST");
            ledger_tagged = untagged_env == nullptr || untagged_env[0] != '1';
            const char* const file_env = std::getenv("LICHII_IOS_FILE_GUEST");
            const bool file_wanted = file_env != nullptr && file_env[0] == '1';
            const bool file = file_wanted && InitFile();
            if (file) {
                ledger_tagged = false;
            }
            const char* const chunked_env = std::getenv("LICHII_IOS_CHUNKED_GUEST");
            const bool chunked_wanted = chunked_env != nullptr && chunked_env[0] == '1';
            const bool chunked = !file && ledger_tagged && chunked_wanted && InitChunks();
            memory_object_size_t entry_size = backing_size;
            kern_return_t kr = KERN_SUCCESS;
            if (!chunked && !file) {
                kr = mach_make_memory_entry_64(
                    mach_task_self(), &entry_size, 0,
                    MAP_MEM_NAMED_CREATE | VM_PROT_READ | VM_PROT_WRITE |
                        (ledger_tagged ? MapMemLedgerTagged : 0),
                    &backing_entry, MACH_PORT_NULL);
            }
            if (kr != KERN_SUCCESS && ledger_tagged) {
                LOG_WARNING(HW_Memory, "Ledger-tagged guest memory refused ({}): untagged",
                            mach_error_string(kr));
                ledger_tagged = false;
                entry_size = backing_size;
                kr = mach_make_memory_entry_64(mach_task_self(), &entry_size, 0,
                                               MAP_MEM_NAMED_CREATE | VM_PROT_READ | VM_PROT_WRITE,
                                               &backing_entry, MACH_PORT_NULL);
            }
            if (kr != KERN_SUCCESS || entry_size < backing_size) {
                LOG_CRITICAL(HW_Memory, "mach_make_memory_entry_64({:#x}) failed: {}", backing_size,
                             mach_error_string(kr));
                backing_entry = MACH_PORT_NULL;
                return false;
            }
            LOG_INFO(HW_Memory, "Guest memory: {} MiB, {}", backing_size >> 20,
                     file ? "in a file (experiment: outside the footprint; iOS writes it to "
                            "storage when it needs the memory)"
                     : chunked ? "ledger-tagged in purgeable 2 MiB chunks (each page counted once; "
                                 "chunks of zeros are given back)"
                     : ledger_tagged ? "ledger-tagged (each page counted once)"
                                     : "untagged (counted per mapping; zero trimming on)");
            if (file_wanted && !file) {
                LOG_WARNING(HW_Memory, "Guest memory: the file experiment is on but was refused "
                                       "(reason in the line above): memory as without it");
            } else if (chunked_wanted && !chunked) {
                LOG_WARNING(HW_Memory, "Guest memory: the chunk experiment is on but not in use ({})",
                            file ? "the file experiment wins"
                            : ledger_tagged ? "refused, reason in the line above"
                                            : "it needs ledger-tagged memory");
            }
            if (!chunked && !file) {
                vm_address_t backing_address = 0;
                kr = vm_map(mach_task_self(), &backing_address, backing_size, 0, VM_FLAGS_ANYWHERE,
                            backing_entry, 0, FALSE, VM_PROT_READ | VM_PROT_WRITE,
                            VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
                if (kr != KERN_SUCCESS) {
                    LOG_CRITICAL(HW_Memory, "vm_map of the {:#x} byte backing object failed: {}",
                                 backing_size, mach_error_string(kr));
                    return false;
                }
                backing_base = reinterpret_cast<u8*>(backing_address);
            }
            // Even with extended-virtual-addressing iOS hands out far less than the 512 GiB a 39-bit
            // guest spans, so take the largest power of two it grants: guest addresses above it
            // take dynarmic's page table path.
            constexpr size_t MinGuestSpace = size_t{1} << 32;
            vm_address_t virtual_address = 0;
            size_t guest_space = virtual_size - HugePageSize;
            while (true) {
                kr = vm_allocate(mach_task_self(), &virtual_address, guest_space + HugePageSize,
                                 VM_FLAGS_ANYWHERE);
                if (kr == KERN_SUCCESS) {
                    break;
                }
                LOG_INFO(HW_Memory, "fastmem: reserving {} GiB of guest address space failed: {}",
                         guest_space >> 30, mach_error_string(kr));
                if (guest_space / 2 < MinGuestSpace) {
                    LOG_CRITICAL(HW_Memory,
                                 "vm_allocate of the guest address space failed down to {} GiB (is "
                                 "the extended-virtual-addressing entitlement present?)",
                                 guest_space >> 30);
                    return false;
                }
                guest_space /= 2;
            }
            virtual_size = guest_space + HugePageSize;
            vm_protect(mach_task_self(), virtual_address, virtual_size, FALSE, VM_PROT_NONE);
            virtual_base = virtual_map_base = reinterpret_cast<u8*>(virtual_address);
        }
#else
#if defined(__sun__) || defined(__HAIKU__) || defined(__NetBSD__) || defined(__DragonFly__)
        fd = shm_open_anon(O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
#elif defined(__OpenBSD__)
        fd = shm_open_anon(O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
#elif defined(__FreeBSD__)
        fd = shm_open(SHM_ANON, O_RDWR, 0600);
#elif defined(__APPLE__)
        char shm_name[32];
        std::snprintf(shm_name, sizeof(shm_name), "/eden_mem_%d", int(getpid()));
        fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) {
            shm_unlink(shm_name);
        } else {
            char template_path[] = "/tmp/eden_mem_XXXXXX";
            fd = mkstemp(template_path);
            if (fd >= 0) {
                unlink(template_path);
            }
        }
#elif defined(__managarm__)
        // macOS doesn't have memfd_create, use anonymous temporary file
        char template_path[] = "/tmp/eden_mem_XXXXXX";
        fd = mkstemp(template_path);
        if (fd >= 0) {
            unlink(template_path);
        }
#else
        fd = memfd_create("HostMemory", 0);
#endif
        bool use_anon = false;
        if (fd <= 0) {
            LOG_WARNING(Common_Memory, "memfd_create: {}", strerror(errno));
            use_anon = true;
        }
        if (!use_anon) {
            // Defined to extend the file with zeros
            int ret = ftruncate(fd, backing_size);
            if (ret != 0) {
                LOG_WARNING(Common_Memory, "ftruncate: {} (likely out-of-emory)", strerror(errno));
                use_anon = true;
            }
        }
        if (use_anon) {
            LOG_WARNING(Common_Memory, "Using private mappings instead of shared ones");
            backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NOCORE, -1, 0));
            if (fd > 0) {
                fd = -1;
                close(fd);
            }
        } else {
            backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NOCORE, fd, 0));
        }
        if (backing_base == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
            return false;
        }

        // Virtual memory initialization
        virtual_base = virtual_map_base = static_cast<u8*>(ChooseVirtualBase(virtual_size));
        if (virtual_base == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
            return false;
        }
#if defined(__linux__)
        madvise(virtual_base, virtual_size, MADV_HUGEPAGE);
#endif
#endif
        free_manager.SetAddressSpace(virtual_base, virtual_size);

        if (host_page_size != PageAlignment) {
            if ((fd < 0 && !HasBackingObject()) || virtual_size % host_page_size != 0) {
                LOG_WARNING(Common_Memory, "Can't emulate {:#x} byte pages with host page size {:#x}",
                            PageAlignment, host_page_size);
                return false;
            }
            page_states_size = (virtual_size / PageAlignment) * sizeof(u32);
            void* const states = mmap(nullptr, page_states_size, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (states == MAP_FAILED) {
                LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
                return false;
            }
            page_states = static_cast<u32*>(states);
            LOG_INFO(Common_Memory, "Fastmem maps {:#x} byte host pages whose guest pages are contiguous",
                     host_page_size);
        }
        return true;
    }

    ~Impl() {
        Release();
    }

    void* Allocate(size_t size) {
        auto* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (ptr == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate fallback buffer with size {:#x}, {}", size, strerror(errno));
            return nullptr;
        }
        return ptr;
    }

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms) {
        // Intersect the range with our address space.
        const size_t intended_offset = virtual_offset;
        AdjustMap(&virtual_offset, &length);
        if (length != 0) {
            host_offset += virtual_offset - intended_offset;
        }

        if (page_states) {
            if (length == 0) {
                return;
            }
            std::scoped_lock lk{page_states_mutex};
            const size_t first = StateIndex(virtual_offset);
            const u32 state = PageMapped | PermissionBits(perms);
            for (size_t i = 0; i < length / PageAlignment; ++i) {
                const u32 host_page = static_cast<u32>(host_offset / PageAlignment + i);
                page_states[first + i] = state | (host_page << PageOffsetShift);
            }
            UpdateHostPages(virtual_offset, length, true);
            return;
        }

        // We are removing a placeholder.
        free_manager.AllocateBlock(virtual_base + virtual_offset, length);

        // Deduce mapping protection flags.
        int prot_flags = PROT_NONE;
        if (True(perms & MemoryPermission::Read))
            prot_flags |= PROT_READ;
        if (True(perms & MemoryPermission::Write))
            prot_flags |= PROT_WRITE;
#ifdef ARCHITECTURE_arm64
        if (True(perms & MemoryPermission::Execute))
            prot_flags |= PROT_EXEC;
#endif
#ifdef LICHII_IOS_MACH_MEMORY
        ASSERT_MSG(MapBackingView(virtual_base + virtual_offset, length, host_offset, prot_flags),
                   "vm_map failed for {:#x} bytes at offset {:#x}", length, virtual_offset);
#else
        int flags = (fd >= 0 ? MAP_SHARED : MAP_PRIVATE) | MAP_FIXED;
        void* ret = mmap(virtual_base + virtual_offset, length, prot_flags, flags, fd, host_offset);
        ASSERT_MSG(ret != MAP_FAILED, "mmap: {} {}", strerror(errno), fd);
#endif
    }

    void Unmap(size_t virtual_offset, size_t length) {
        // The method name is wrong. We're still talking about the virtual range.
        // We don't want to unmap, we want to reserve this memory.

        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        if (page_states) {
            if (length == 0) {
                return;
            }
            std::scoped_lock lk{page_states_mutex};
            std::fill_n(page_states + StateIndex(virtual_offset), length / PageAlignment, 0u);
            UpdateHostPages(virtual_offset, length, true);
            return;
        }

        // Merge with any adjacent placeholder mappings.
        auto [merged_pointer, merged_size] =
            free_manager.FreeBlock(virtual_base + virtual_offset, length);

#ifdef LICHII_IOS_MACH_MEMORY
        ASSERT_MSG(ReserveRange(merged_pointer, merged_size), "vm_allocate failed for {:#x} bytes",
                   merged_size);
#else
        void* ret = mmap(merged_pointer, merged_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
#endif
    }

    void Protect(size_t virtual_offset, size_t length, bool read, bool write, bool execute) {
        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        if (page_states) {
            if (length == 0) {
                return;
            }
            const u32 bits = (read ? PageRead : 0u) | (write ? PageWrite : 0u) |
                             (execute ? PageExecute : 0u);
            std::scoped_lock lk{page_states_mutex};
            u32* const states = page_states + StateIndex(virtual_offset);
            for (size_t i = 0; i < length / PageAlignment; ++i) {
                states[i] = (states[i] & ~PagePermissionMask) | bits;
            }
            UpdateHostPages(virtual_offset, length, false);
            return;
        }

        int flags = PROT_NONE;
        if (read) {
            flags |= PROT_READ;
        }
        if (write) {
            flags |= PROT_WRITE;
        }
#ifdef HAS_NCE
        if (execute) {
            flags |= PROT_EXEC;
        }
#endif
        int ret = mprotect(virtual_base + virtual_offset, length, flags);
        ASSERT_MSG(ret == 0, "mprotect failed: {}", strerror(errno));
    }

    void EnableDirectMappedAddress() {
        virtual_base = nullptr;
    }

#ifdef LICHII_IOS_MACH_MEMORY
    /// Untagged guest memory: maps [offset, offset + size) of the backing view again. Its pages
    /// stay in the memory object; only the view's page table entries go, each of which XNU
    /// charges to the footprint on top of the fastmem view's. The swap is atomic for other
    /// threads (the map lock is held throughout) and a later access soft-faults the page back.
    bool DropBackingMappings(size_t offset, size_t size) {
        if (backing_entry == MACH_PORT_NULL || ledger_tagged) {
            return false;
        }
        vm_address_t target = reinterpret_cast<vm_address_t>(backing_base + offset);
        return vm_map(mach_task_self(), &target, size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                      backing_entry, offset, FALSE, VM_PROT_READ | VM_PROT_WRITE,
                      VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE) == KERN_SUCCESS;
    }

    bool Chunked() const noexcept {
        return !chunk_entries.empty();
    }

    /// Chunked guest memory: gives back the chunks that held memory but no data on consecutive
    /// passes. residency is mincore's answer for the whole backing and counted the bits of a page
    /// that is charged. Returns the bytes given back.
    ///
    /// Nothing may write to a chunk between the check that it is all zeros and the purge, so both
    /// happen with every other thread of the process suspended (thread_suspend returns once the
    /// thread is off its core and it runs no user instruction until thread_resume). Whatever a
    /// stopped thread wrote before is seen by the check; whatever it reads or writes after finds
    /// zeros either way, as fresh zero-filled pages. The mappings and their protections are not
    /// touched. A thread stopped inside a system call still finishes it in the kernel: a batch is
    /// skipped while one sits in a read into guest memory, and the purge leaves wired and busy
    /// pages (I/O in flight) alone.
    size_t ReleaseZeroChunks(const std::vector<char>& residency, size_t page_size, int counted,
                             const std::function<bool()>& pause) {
        if (!Chunked() || release_broken) {
            return 0;
        }
        const size_t chunk_pages = ChunkSize / page_size;
        due.clear();
        for (size_t chunk = 0; chunk < chunk_entries.size(); ++chunk) {
            const char* const pages = residency.data() + chunk * chunk_pages;
            const u8* const base = backing_base + chunk * ChunkSize;
            bool has_memory = false;
            bool has_data = false;
            for (size_t page = 0; page < chunk_pages && !has_data; ++page) {
                if ((pages[page] & counted) != 0) {
                    has_memory = true;
                    has_data = !IsZeroRange(base + page * page_size, page_size);
                }
            }
            if (candidates.Observe(chunk, has_memory, has_data)) {
                due.push_back(static_cast<u32>(chunk));
            }
        }
        size_t released = 0;
        size_t chunks = 0;
        u32 stops = 0;
        u32 skipped = 0;
        s64 longest_us = 0;
        for (size_t first = 0; first < due.size(); first += BatchChunks) {
            const size_t batch = (std::min)(BatchChunks, due.size() - first);
            const auto start = std::chrono::steady_clock::now();
            const BatchResult result = PurgeBatch(due.data() + first, batch, page_size, counted);
            longest_us = (std::max<s64>)(
                longest_us, std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
            ++stops;
            released += result.bytes;
            chunks += result.chunks;
            if (!result.stopped) {
                // Not this pass: the same threads are likely where they were.
                ++skipped;
                break;
            }
            if (pause && !pause()) {
                break;
            }
        }
        released_total += released;
        if (release_broken) {
            LOG_ERROR(HW_Memory, "Guest chunks: a purge failed; zero chunks stay from now on");
        }
        if (released != 0 || (skipped != 0 && skip_reports++ < 8)) {
            LOG_INFO(HW_Memory,
                     "Guest chunks: {} MiB of zero pages given back in {} of {} due chunks, {} "
                     "stops (longest {} us){}; {} MiB so far",
                     released >> 20, chunks, due.size(), stops, longest_us,
                     skipped != 0 ? ", stopped early: threads not stopped or a read in flight" : "",
                     released_total >> 20);
        }
        return released;
    }

    bool ledger_tagged{}; ///< The guest memory object is billed per page, not per mapping
    bool file_backed{};   ///< The guest memory is a mapped file (InitFile)
#endif

    const size_t backing_size; ///< Size of the backing memory in bytes
    size_t virtual_size; ///< Size of the virtual address placeholder in bytes (iOS: as granted)

    u8* backing_base{reinterpret_cast<u8*>(MAP_FAILED)};
    u8* virtual_base{reinterpret_cast<u8*>(MAP_FAILED)};
    u8* virtual_map_base{reinterpret_cast<u8*>(MAP_FAILED)};

private:
    /// Release all resources in the object
    void Release() {
        if (page_states != nullptr) {
            munmap(page_states, page_states_size);
        }

        if (virtual_map_base != MAP_FAILED) {
            int ret = munmap(virtual_map_base, virtual_size);
            ASSERT_MSG(ret == 0, "munmap failed: {}", strerror(errno));
        }

        if (backing_base != MAP_FAILED) {
            int ret = munmap(backing_base, backing_size);
            ASSERT_MSG(ret == 0, "munmap failed: {}", strerror(errno));
        }

        if (fd != -1) {
            int ret = close(fd);
            ASSERT_MSG(ret == 0, "close failed: {}", strerror(errno));
        }
#ifdef LICHII_IOS_MACH_MEMORY
        if (backing_entry != MACH_PORT_NULL) {
            mach_port_deallocate(mach_task_self(), backing_entry);
            backing_entry = MACH_PORT_NULL;
        }
        for (const mach_port_t entry : chunk_entries) {
            if (entry != MACH_PORT_NULL) {
                mach_port_deallocate(mach_task_self(), entry);
            }
        }
        chunk_entries.clear();
#endif
    }

    bool HasBackingObject() const {
#ifdef LICHII_IOS_MACH_MEMORY
        return backing_entry != MACH_PORT_NULL || Chunked() || file_backed;
#else
        return false;
#endif
    }

#ifdef LICHII_IOS_MACH_MEMORY
    static vm_prot_t ToVmProt(int prot) {
        vm_prot_t out = VM_PROT_NONE;
        if (prot & PROT_READ) {
            out |= VM_PROT_READ;
        }
        if (prot & PROT_WRITE) {
            out |= VM_PROT_WRITE;
        }
        return out;
    }

    bool MapBackingView(void* address, size_t size, size_t host_offset, int prot) {
        if (file_backed) {
            // MAP_FIXED replaces the reservation in one step (mmap: VM_FLAGS_OVERWRITE).
            void* const mapped = mmap(address, size, prot & (PROT_READ | PROT_WRITE),
                                      MAP_SHARED | MAP_FIXED, fd, static_cast<off_t>(host_offset));
            if (mapped == MAP_FAILED) {
                LOG_CRITICAL(HW_Memory, "mmap({}, {:#x}, offset {:#x}) of the guest file failed: {}",
                             fmt::ptr(address), size, host_offset, strerror(errno));
            } else if (!std::exchange(file_view_logged, true)) {
                LOG_INFO(HW_Memory, "Guest file: first view mapped ({:#x} bytes at {}, file offset "
                                    "{:#x}, protection {})",
                         size, fmt::ptr(address), host_offset, prot);
            }
            return mapped != MAP_FAILED;
        }
        if (Chunked()) {
            // One mapping per chunk the range touches; offsets into the backing stay linear.
            return ForEachChunkPiece(
                host_offset, size, ChunkSize,
                [&](size_t chunk, size_t offset, size_t piece, size_t done) {
                    return MapEntry(chunk_entries[chunk], static_cast<u8*>(address) + done, piece,
                                    offset, prot);
                });
        }
        return MapEntry(backing_entry, address, size, host_offset, prot);
    }

    static bool MapEntry(mach_port_t entry, void* address, size_t size, size_t offset, int prot) {
        vm_address_t target = reinterpret_cast<vm_address_t>(address);
        const kern_return_t kr = vm_map(mach_task_self(), &target, size, 0,
                                        VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, entry, offset, FALSE,
                                        ToVmProt(prot), VM_PROT_READ | VM_PROT_WRITE,
                                        VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) {
            LOG_CRITICAL(HW_Memory, "vm_map({}, {:#x}, offset {:#x}) failed: {}", fmt::ptr(address), size,
                         offset, mach_error_string(kr));
        }
        return kr == KERN_SUCCESS;
    }

    /// LICHII_IOS_FILE_GUEST=1 (experiment): the backing as a file in the app's tmp directory,
    /// mapped shared. XNU bills a page of a file to the task's external ledger, which is not part
    /// of the footprint (pmap_enter: only PMAP_OPTIONS_INTERNAL mappings reach phys_footprint, and
    /// vm_fault sets that for internal objects only), counts it as available memory for the
    /// system-wide jetsam and can write it out and drop it, which it never does to anonymous
    /// memory on a phone. The price: a dirty page has to reach storage before its RAM is free, and
    /// a page that left is read back on its next use.
    ///
    /// The file is unlinked while open, so it is gone when the process ends however it ends, and
    /// its blocks are reserved up front: a write-out can't run out of space later. False (and a
    /// log line) when storage is short; the caller then uses the usual memory object.
    bool InitFile() {
        const char* const directory = std::getenv("TMPDIR");
        if (directory == nullptr || directory[0] == '\0') {
            LOG_WARNING(HW_Memory, "Guest memory in a file refused: no TMPDIR");
            return false;
        }
        std::string path{directory};
        if (path.back() != '/') {
            path += '/';
        }
        path += "lichii-guest-memory";
        // Left by a run that died between open and unlink.
        unlink(path.c_str());
        struct statfs volume {};
        if (statfs(directory, &volume) != 0) {
            LOG_WARNING(HW_Memory, "Guest memory in a file refused: statfs: {}", strerror(errno));
            return false;
        }
        constexpr u64 Margin = u64{2} << 30;
        const u64 free_bytes = static_cast<u64>(volume.f_bavail) * volume.f_bsize;
        if (free_bytes < backing_size + Margin) {
            LOG_WARNING(HW_Memory,
                        "Guest memory in a file refused: {} MiB of storage free, {} MiB needed",
                        free_bytes >> 20, (backing_size + Margin) >> 20);
            return false;
        }
        const int file =
            open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (file < 0) {
            LOG_WARNING(HW_Memory, "Guest memory in a file refused: open: {}", strerror(errno));
            return false;
        }
        unlink(path.c_str());
#ifdef F_SETPROTECTIONCLASS
        // Class D (NSFileProtectionNone): readable and writable while the phone is locked.
        constexpr int ProtectionClassD = 4;
        const bool unprotected = fcntl(file, F_SETPROTECTIONCLASS, ProtectionClassD) == 0;
#else
        const bool unprotected = false;
#endif
        fstore_t store{};
        store.fst_flags = F_ALLOCATEALL;
        store.fst_posmode = F_PEOFPOSMODE;
        store.fst_length = static_cast<off_t>(backing_size);
        const bool reserved = fcntl(file, F_PREALLOCATE, &store) == 0;
        const int reserve_error = errno;
        if (ftruncate(file, static_cast<off_t>(backing_size)) != 0) {
            LOG_WARNING(HW_Memory, "Guest memory in a file refused: ftruncate: {}", strerror(errno));
            close(file);
            return false;
        }
        void* const base =
            mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
        if (base == MAP_FAILED) {
            LOG_WARNING(HW_Memory, "Guest memory in a file refused: mmap: {}", strerror(errno));
            close(file);
            return false;
        }
        LOG_WARNING(HW_Memory,
                    "Guest memory file: {} MiB in {} ({} MiB of storage were free), storage {}, "
                    "data protection {}",
                    backing_size >> 20, directory, free_bytes >> 20,
                    reserved ? "reserved" : fmt::format("not reserved ({})", strerror(reserve_error)),
                    unprotected ? "off" : "left at the default");
        backing_base = static_cast<u8*>(base);
        fd = file;
        file_backed = true;
        return true;
    }

    /// LICHII_IOS_CHUNKED_GUEST=1 (experiment): the backing as one ledger-tagged, purgeable memory
    /// object per 2 MiB, mapped back to back. Emptying a purgeable object frees its pages and
    /// takes them off the footprint at once while every mapping of it stays, and the next access
    /// zero-fills (vm_object_purgable_control: vm_purgeable_accounting, vm_object_purge): the way
    /// a chunk of zeros leaves the footprint, which tagged memory otherwise never does. The
    /// objects stay non-volatile, so the kernel never empties one by itself. 2 MiB frees the most
    /// (zero pages sit between data) and 6144 objects, ports and map entries are cheap in XNU.
    bool InitChunks() {
        if (backing_size % ChunkSize != 0) {
            return false;
        }
        const size_t count = backing_size / ChunkSize;
        vm_address_t address = 0;
        kern_return_t kr = vm_map(mach_task_self(), &address, backing_size, 0, VM_FLAGS_ANYWHERE,
                                  MACH_PORT_NULL, 0, FALSE, VM_PROT_NONE,
                                  VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) {
            LOG_WARNING(HW_Memory, "Chunked guest memory: no address range ({}): one object",
                        mach_error_string(kr));
            return false;
        }
        chunk_entries.assign(count, MACH_PORT_NULL);
        const char* step = "mach_make_memory_entry_64";
        for (size_t chunk = 0; chunk < count; ++chunk) {
            memory_object_size_t entry_size = ChunkSize;
            kr = mach_make_memory_entry_64(mach_task_self(), &entry_size, 0,
                                           MAP_MEM_NAMED_CREATE | MapMemPurgable |
                                               MapMemLedgerTagged | VM_PROT_READ | VM_PROT_WRITE,
                                           &chunk_entries[chunk], MACH_PORT_NULL);
            if (kr == KERN_SUCCESS && entry_size < ChunkSize) {
                kr = KERN_NO_SPACE;
            }
            if (kr != KERN_SUCCESS) {
                break;
            }
            vm_address_t target = address + chunk * ChunkSize;
            kr = vm_map(mach_task_self(), &target, ChunkSize, 0,
                        VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, chunk_entries[chunk], 0, FALSE,
                        VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE,
                        VM_INHERIT_NONE);
            if (kr != KERN_SUCCESS) {
                step = "vm_map";
                break;
            }
        }
        backing_base = reinterpret_cast<u8*>(address);
        stopped.assign(MaxStopped, MACH_PORT_NULL);
        if (kr == KERN_SUCCESS) {
            // Everything a release needs, once, while nothing is at stake: a refusal here means
            // one object instead, and no call binds or allocates for the first time later, with
            // the other threads stopped.
            step = "stopping the threads";
            kr = KERN_FAILURE;
            if (StopWorld()) {
                static_cast<void>(KernelMayWriteGuest());
                std::array<char, ChunkSize / PageAlignment> pages;
                const bool queried = mincore(reinterpret_cast<char*>(backing_base), ChunkSize,
                                             pages.data()) == 0;
                step = "vm_purgable_control";
                kr = Purge(0);
                ResumeWorld();
                if (kr == KERN_SUCCESS && !queried) {
                    step = "mincore";
                    kr = KERN_FAILURE;
                }
            }
        }
        if (kr != KERN_SUCCESS) {
            LOG_WARNING(HW_Memory, "Chunked guest memory: {} failed ({}): one object", step,
                        mach_error_string(kr));
            for (const mach_port_t entry : chunk_entries) {
                if (entry != MACH_PORT_NULL) {
                    mach_port_deallocate(mach_task_self(), entry);
                }
            }
            chunk_entries.clear();
            vm_deallocate(mach_task_self(), address, backing_size);
            backing_base = reinterpret_cast<u8*>(MAP_FAILED);
            return false;
        }
        candidates = ChunkCandidates{count};
        due.reserve(count);
        return true;
    }

    static bool IsZeroRange(const u8* data, size_t size) {
        const auto* words = reinterpret_cast<const u64*>(data);
        for (size_t i = 0; i < size / sizeof(u64); i += 8) {
            const u64 bits = words[i] | words[i + 1] | words[i + 2] | words[i + 3] | words[i + 4] |
                             words[i + 5] | words[i + 6] | words[i + 7];
            if (bits != 0) {
                return false;
            }
        }
        return true;
    }

    /// Empties a chunk and makes it non-volatile again.
    kern_return_t Purge(size_t chunk) {
        const vm_address_t address = reinterpret_cast<vm_address_t>(backing_base + chunk * ChunkSize);
        int purgeable = VM_PURGABLE_EMPTY;
        kern_return_t kr = vm_purgable_control(mach_task_self(), address, VM_PURGABLE_SET_STATE,
                                               &purgeable);
        if (kr != KERN_SUCCESS) {
            return kr;
        }
        purgeable = VM_PURGABLE_NONVOLATILE;
        kr = vm_purgable_control(mach_task_self(), address, VM_PURGABLE_SET_STATE, &purgeable);
        if (kr != KERN_SUCCESS) {
            purgeable = VM_PURGABLE_NONVOLATILE;
            kr = vm_purgable_control(mach_task_self(), address, VM_PURGABLE_SET_STATE, &purgeable);
        }
        return kr;
    }

    // Between StopWorld and ResumeWorld another thread may hold any user-space lock (malloc, the
    // log, the loader): only system calls, no allocation and no logging in there.

    /// Suspends every other thread of the process, listing them again until a round finds no new
    /// one. False (and nothing left suspended) when that fails.
    bool StopWorld() {
        const thread_act_t self = mach_thread_self();
        num_stopped = 0;
        bool complete = false;
        bool failed = false;
        for (int round = 0; round < 8 && !complete && !failed; ++round) {
            thread_act_array_t list = nullptr;
            mach_msg_type_number_t listed = 0;
            if (task_threads(mach_task_self(), &list, &listed) != KERN_SUCCESS) {
                failed = true;
                break;
            }
            size_t added = 0;
            for (mach_msg_type_number_t i = 0; i < listed; ++i) {
                const thread_act_t thread = list[i];
                const thread_act_t* const begin = stopped.data();
                const thread_act_t* const end = begin + num_stopped;
                bool kept = false;
                if (thread != self && std::find(begin, end, thread) == end) {
                    if (num_stopped == stopped.size()) {
                        failed = true;
                    } else if (thread_suspend(thread) == KERN_SUCCESS) {
                        // Anything else is a thread that ended: it runs no more user code.
                        stopped[num_stopped++] = thread;
                        ++added;
                        kept = true;
                    }
                }
                if (!kept) {
                    mach_port_deallocate(mach_task_self(), thread);
                }
            }
            vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(list),
                          listed * sizeof(thread_act_t));
            complete = added == 0;
        }
        mach_port_deallocate(mach_task_self(), self);
        if (failed || !complete) {
            ResumeWorld();
            return false;
        }
        return true;
    }

    void ResumeWorld() {
        for (size_t i = 0; i < num_stopped; ++i) {
            thread_resume(stopped[i]);
            mach_port_deallocate(mach_task_self(), stopped[i]);
        }
        num_stopped = 0;
    }

    /// Whether a stopped thread sits in a system call that reads into guest memory: the kernel
    /// would still copy into its buffer. The saved registers of a thread inside a system call
    /// are the call's: x16 the number, x1 the buffer of the read family.
    bool KernelMayWriteGuest() const {
#if defined(__arm64__) || defined(__aarch64__)
        constexpr std::array<u64, 6> ReadCalls{3, 29, 153, 396, 403, 414};
        const auto backing = reinterpret_cast<u64>(backing_base);
        const auto arena = reinterpret_cast<u64>(virtual_map_base);
        for (size_t i = 0; i < num_stopped; ++i) {
            arm_thread_state64_t registers;
            mach_msg_type_number_t words = ARM_THREAD_STATE64_COUNT;
            if (thread_get_state(stopped[i], ARM_THREAD_STATE64,
                                 reinterpret_cast<thread_state_t>(&registers),
                                 &words) != KERN_SUCCESS) {
                continue;
            }
            const u64 number = registers.__x[16];
            if (std::find(ReadCalls.begin(), ReadCalls.end(), number) == ReadCalls.end()) {
                continue;
            }
            const u64 buffer = registers.__x[1];
            if (buffer - backing < backing_size || buffer - arena < virtual_size) {
                return true;
            }
        }
#endif
        return false;
    }

    struct BatchResult {
        size_t bytes;
        size_t chunks;
        bool stopped;
    };

    /// Gives back the chunks of the list that hold only zeros, with the other threads stopped.
    BatchResult PurgeBatch(const u32* list, size_t count, size_t page_size, int counted) {
        BatchResult result{};
        if (!StopWorld()) {
            return result;
        }
        if (KernelMayWriteGuest()) {
            ResumeWorld();
            return result;
        }
        result.stopped = true;
        const size_t chunk_pages = ChunkSize / page_size;
        for (size_t i = 0; i < count; ++i) {
            const u8* const base = backing_base + list[i] * ChunkSize;
            std::array<char, ChunkSize / PageAlignment> pages;
            if (mincore(reinterpret_cast<char*>(const_cast<u8*>(base)), ChunkSize,
                        pages.data()) != 0) {
                continue;
            }
            size_t held = 0;
            bool zero = true;
            for (size_t page = 0; page < chunk_pages && zero; ++page) {
                if ((pages[page] & counted) != 0) {
                    held += page_size;
                    zero = IsZeroRange(base + page * page_size, page_size);
                }
            }
            if (!zero || held == 0) {
                continue;
            }
            if (Purge(list[i]) != KERN_SUCCESS) {
                release_broken = true;
                break;
            }
            candidates.Released(list[i]);
            result.bytes += held;
            ++result.chunks;
        }
        ResumeWorld();
        return result;
    }

    bool ReserveRange(void* address, size_t size) {
        // One call that maps the range without access. vm_allocate followed by vm_protect left it
        // readable and writable zero-filled memory in between: a guest thread whose 4 KiB page
        // shares a 16 KiB host page with the range (UpdateHostPages reserves whole host pages)
        // read zeros or lost a write there instead of faulting into the page table path.
        vm_address_t target = reinterpret_cast<vm_address_t>(address);
        const kern_return_t kr = vm_map(mach_task_self(), &target, size, 0,
                                        VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, MACH_PORT_NULL, 0,
                                        FALSE, VM_PROT_NONE, VM_PROT_READ | VM_PROT_WRITE,
                                        VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) {
            LOG_CRITICAL(HW_Memory, "vm_map({}, {:#x}) of a reserved range failed: {}",
                         fmt::ptr(address), size, mach_error_string(kr));
        } else if (file_backed && file_view_logged && !std::exchange(file_unmap_logged, true)) {
            LOG_INFO(HW_Memory, "Guest file: first view unmapped ({:#x} bytes at {})", size,
                     fmt::ptr(address));
        }
        return kr == KERN_SUCCESS;
    }
#endif

    void AdjustMap(size_t* virtual_offset, size_t* length) {
        if (virtual_base != nullptr) {
            return;
        }

        // If we are direct mapped, we want to make sure we are operating on a region
        // that is in range of our virtual mapping.
        size_t intended_start = *virtual_offset;
        size_t intended_end = intended_start + *length;
        size_t address_space_start = reinterpret_cast<size_t>(virtual_map_base);
        size_t address_space_end = address_space_start + virtual_size;

        if (address_space_start > intended_end || intended_start > address_space_end) {
            *virtual_offset = 0;
            *length = 0;
        } else {
            *virtual_offset = (std::max)(intended_start, address_space_start);
            *length = (std::min)(intended_end, address_space_end) - *virtual_offset;
        }
    }

    static constexpr u32 PageRead = 1u << 0;
    static constexpr u32 PageWrite = 1u << 1;
    static constexpr u32 PageExecute = 1u << 2;
    static constexpr u32 PagePermissionMask = PageRead | PageWrite | PageExecute;
    static constexpr u32 PageMapped = 1u << 3;
    static constexpr u32 PageOffsetShift = 4;

    static u32 PermissionBits(MemoryPermission perms) {
        return (True(perms & MemoryPermission::Read) ? PageRead : 0u) |
               (True(perms & MemoryPermission::Write) ? PageWrite : 0u) |
               (True(perms & MemoryPermission::Execute) ? PageExecute : 0u);
    }

    static int ProtectionFlags(u32 bits) {
        int prot = PROT_NONE;
        if ((bits & PageRead) != 0) {
            prot |= PROT_READ;
        }
        if ((bits & PageWrite) != 0) {
            prot |= PROT_WRITE;
        }
#ifdef HAS_NCE
        if ((bits & PageExecute) != 0) {
            prot |= PROT_EXEC;
        }
#endif
        return prot;
    }

    struct HostPage {
        bool mapped;
        size_t host_offset;
        int prot;
    };

    size_t StateIndex(size_t virtual_offset) const {
        const size_t base = virtual_base != nullptr ? 0 : reinterpret_cast<size_t>(virtual_map_base);
        return (virtual_offset - base) / PageAlignment;
    }

    HostPage ResolveHostPage(size_t virtual_offset) const {
        const u32* const states = page_states + StateIndex(virtual_offset);
        const u32 first = states[0];
        const size_t host_offset = size_t(first >> PageOffsetShift) * PageAlignment;
        if ((first & PageMapped) == 0 || host_offset % host_page_size != 0) {
            return {};
        }
        u32 permissions = first & PagePermissionMask;
        for (size_t i = 1; i < host_page_size / PageAlignment; ++i) {
            const u32 state = states[i];
            if ((state & PageMapped) == 0 || (state >> PageOffsetShift) != (first >> PageOffsetShift) + i) {
                return {};
            }
            permissions &= state;
        }
        return {true, host_offset, ProtectionFlags(permissions)};
    }

    void UpdateHostPages(size_t virtual_offset, size_t length, bool remap) {
        const size_t begin = AlignDown(virtual_offset, host_page_size);
        const size_t end = AlignUp(virtual_offset + length, host_page_size);
        const auto apply = [&](size_t run_begin, size_t run_end, const HostPage& page) {
            u8* const address = virtual_base + run_begin;
            const size_t size = run_end - run_begin;
            if (!page.mapped) {
                if (remap) {
#ifdef LICHII_IOS_MACH_MEMORY
                    ASSERT_MSG(ReserveRange(address, size), "vm_allocate failed for {:#x} bytes", size);
#else
                    void* ret = mmap(address, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
                    ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
#endif
                }
            } else if (remap) {
#ifdef LICHII_IOS_MACH_MEMORY
                ASSERT_MSG(MapBackingView(address, size, page.host_offset, page.prot),
                           "vm_map failed for {:#x} bytes", size);
#else
                void* ret = mmap(address, size, page.prot, MAP_SHARED | MAP_FIXED, fd, off_t(page.host_offset));
                ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
#endif
            } else {
                int ret = mprotect(address, size, page.prot);
#ifdef LICHII_IOS_MACH_MEMORY
                if (ret != 0) {
                    const int error = errno;
                    LOG_CRITICAL(HW_Memory, "mprotect({}, {:#x}, {}) of a guest view failed: {}{}",
                                 fmt::ptr(address), size, page.prot, strerror(error),
                                 file_backed ? " (guest memory in a file)" : "");
                    errno = error;
                } else if (file_backed && !std::exchange(file_protect_logged, true)) {
                    LOG_INFO(HW_Memory, "Guest file: first protection change ({:#x} bytes at {}, "
                                        "protection {})",
                             size, fmt::ptr(address), page.prot);
                }
#endif
                ASSERT_MSG(ret == 0, "mprotect failed: {}", strerror(errno));
            }
        };
        size_t run_begin = begin;
        HostPage run = ResolveHostPage(begin);
        for (size_t offset = begin + host_page_size; offset < end; offset += host_page_size) {
            const HostPage page = ResolveHostPage(offset);
            const bool continues = page.mapped == run.mapped &&
                                   (!page.mapped || (page.prot == run.prot &&
                                                     page.host_offset == run.host_offset + (offset - run_begin)));
            if (!continues) {
                apply(run_begin, offset, run);
                run_begin = offset;
                run = page;
            }
        }
        apply(run_begin, end, run);
    }

    int fd{-1}; // memfd file descriptor, -1 is the error value of memfd_create
#ifdef LICHII_IOS_MACH_MEMORY
    static constexpr int MapMemLedgerTagged = 0x002000;
    static constexpr int MapMemPurgable = 0x040000;
    static constexpr size_t ChunkSize = HugePageSize;
    /// Chunks checked per stop of the other threads: 16 MiB to read at most, about a millisecond.
    static constexpr size_t BatchChunks = 8;
    static constexpr size_t MaxStopped = 2048;

    mach_port_t backing_entry{MACH_PORT_NULL};
    std::vector<mach_port_t> chunk_entries; ///< Chunked: one memory object per ChunkSize
    ChunkCandidates candidates;
    std::vector<u32> due;
    std::vector<thread_act_t> stopped;
    size_t num_stopped{};
    u64 released_total{};
    u32 skip_reports{};
    bool release_broken{};
    // File mode, under page_states_mutex: the first of each step is logged, so a boot that stops
    // shows where.
    bool file_view_logged{};
    bool file_unmap_logged{};
    bool file_protect_logged{};
#endif
    FreeRegionManager free_manager{};
    size_t host_page_size{PageAlignment};
    u32* page_states{};
    size_t page_states_size{};
    std::mutex page_states_mutex;
};

#endif // ^^^ POSIX ^^^

HostMemory::HostMemory(size_t backing_size_, size_t virtual_size_)
    : backing_size(backing_size_)
    , virtual_size(virtual_size_)
    , num_granules(Common::DivCeil(backing_size_, size_t{1} << GranuleShift))
{
    granules = std::make_unique<std::atomic<u8>[]>(num_granules);
#if defined(__OPENORBIS__) || defined(__managarm__)
    LOG_WARNING(HW_Memory, "Platform doesn't support fastmem");
    backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    virtual_base = nullptr;
#else
    // Try to allocate a fastmem arena.
    impl = std::make_unique<HostMemory::Impl>(AlignUp(backing_size, PageAlignment), AlignUp(virtual_size, PageAlignment) + HugePageSize);
    if (impl->Init()) {
        backing_base = impl->backing_base;
        virtual_base = impl->virtual_base;
        if (virtual_base) {
            // Ensure the virtual base is aligned to the L2 block size.
            virtual_base = reinterpret_cast<u8*>(Common::AlignUp(uintptr_t(virtual_base), HugePageSize));
            virtual_base_offset = virtual_base - impl->virtual_base;
        }
        // iOS may grant less than asked; mappings above it are left to the page table.
        virtual_size = (std::min)(virtual_size, impl->virtual_size - HugePageSize);
        LOG_INFO(HW_Memory,
                 "fastmem: reserved {} GiB guest address space (requested {} GiB), covering guest "
                 "{:#x}-{:#x}",
                 virtual_size >> 30, virtual_size_ >> 30, 0, virtual_size - 1);
    } else {
        LOG_WARNING(HW_Memory, "Platform can support fastmem, but can't create it");
        fallback_buffer = true;
        backing_base = static_cast<u8*>(impl->Allocate(backing_size));
        virtual_base = nullptr;
        impl.reset();
    }
#endif
}

HostMemory::~HostMemory() {
#ifdef _WIN32
    if (fallback_buffer) {
        VirtualFree(backing_base, backing_size, MEM_RELEASE);
    }
#else
    if (fallback_buffer && backing_base != nullptr) {
        munmap(backing_base, backing_size);
    }
#endif
}

HostMemory::HostMemory(HostMemory&&) noexcept = default;

HostMemory& HostMemory::operator=(HostMemory&&) noexcept = default;

void HostMemory::Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms, bool separate_heap) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(host_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(host_offset + length <= backing_size);
    length = ClipToVirtualSize(virtual_offset, length);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    impl->Map(virtual_offset + virtual_base_offset, host_offset, length, perms);
#endif
}

void HostMemory::Unmap(size_t virtual_offset, size_t length, bool separate_heap) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    length = ClipToVirtualSize(virtual_offset, length);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    impl->Unmap(virtual_offset + virtual_base_offset, length);
#endif
}

void HostMemory::Protect(size_t virtual_offset, size_t length, MemoryPermission perm) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    length = ClipToVirtualSize(virtual_offset, length);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    const bool read = True(perm & MemoryPermission::Read);
    const bool write = True(perm & MemoryPermission::Write);
    const bool execute = True(perm & MemoryPermission::Execute);
    impl->Protect(virtual_offset + virtual_base_offset, length, read, write, execute);
#endif
}

namespace {
/// ResetTouched is measuring: nothing of ours may take pages out of the working set or read them.
std::atomic<bool> g_touch_window{};
} // namespace

void HostMemory::HintCold(size_t physical_offset, size_t length,
                          std::optional<size_t> virtual_offset) const {
    if (length == 0 || backing_base == nullptr ||
        g_touch_window.load(std::memory_order_relaxed)) {
        return;
    }
    const auto hint = [length](u8* address) {
#ifdef _WIN32
        // Unlocking pages that are not locked takes them out of the working set.
        VirtualUnlock(address, length);
#elif defined(__linux__) && defined(MADV_COLD)
        // MADV_COLD only deactivates the pages, like VirtualUnlock. MADV_PAGEOUT writes them to
        // swap right away while holding the mmap lock, which stalled every guest core for 3-15 s.
        madvise(address, length, MADV_COLD);
#else
        static_cast<void>(address);
#endif
    };
    hint(backing_base + physical_offset);
    if (virtual_offset && virtual_base != nullptr && *virtual_offset + length <= virtual_size) {
        hint(virtual_base + *virtual_offset);
    }
}

namespace {
std::atomic<u64> g_zero_filled_bytes{};
[[maybe_unused]] std::atomic<u64> g_zero_advised_bytes{};
[[maybe_unused]] std::atomic<u64> g_scanned_resident_bytes{};
std::atomic<HostMemory::GuestAccounting> g_scanned_accounting{};
[[maybe_unused]] std::atomic<u64> g_trimmed_zero_bytes{};
[[maybe_unused]] std::atomic<u64> g_scanned_handed_out_bytes{};

[[maybe_unused]] bool IsZeroPage(const u8* page) {
    // A cache line at a time: most pages that are not zero say so in their first line.
    const auto* words = reinterpret_cast<const u64*>(page);
    for (size_t i = 0; i < PageAlignment / sizeof(u64); i += 8) {
        const u64 bits = words[i] | words[i + 1] | words[i + 2] | words[i + 3] | words[i + 4] |
                         words[i + 5] | words[i + 6] | words[i + 7];
        if (bits != 0) {
            return false;
        }
    }
    return true;
}

#ifdef __APPLE__
/// Zeroes a range of a mapped file, writing only to the pages that hold something: a write makes
/// a page dirty, and a dirty page of a file has to reach storage before its RAM is free. Returns
/// the bytes written.
size_t ZeroNonZeroPages(u8* data, size_t length) {
    size_t written = 0;
    const auto zero = [&written](u8* at, size_t size) {
        if (size != 0) {
            std::memset(at, 0, size);
            written += size;
        }
    };
    u8* const end = data + length;
    u8* page = (std::min)(end, reinterpret_cast<u8*>(Common::AlignUp(
                                   reinterpret_cast<uintptr_t>(data), PageAlignment)));
    zero(data, static_cast<size_t>(page - data));
    for (; static_cast<size_t>(end - page) >= PageAlignment; page += PageAlignment) {
        if (!IsZeroPage(page)) {
            zero(page, PageAlignment);
        }
    }
    zero(page, static_cast<size_t>(end - page));
    return written;
}
#endif

#ifdef _WIN32
/// Addresses of the pages in this process's working set, entries[0] being the count; empty on
/// failure.
std::vector<ULONG_PTR> QueryWorkingSetPages() {
    std::vector<ULONG_PTR> entries(size_t{1} << 22);
    for (int attempt = 0;; ++attempt) {
        if (K32QueryWorkingSet(GetCurrentProcess(), entries.data(),
                               static_cast<DWORD>(entries.size() * sizeof(ULONG_PTR)))) {
            break;
        }
        if (GetLastError() != ERROR_BAD_LENGTH || attempt == 3) {
            return {};
        }
        entries.assign((std::max)(entries.size() * 2, size_t{entries[0]} + (size_t{1} << 18)), 0);
    }
    entries.resize((std::min)(size_t{entries[0]}, entries.size() - 1) + 1);
    return entries;
}
#endif
} // namespace

std::vector<bool> HostMemory::ResidentPages() const {
    if (backing_base == nullptr) {
        return {};
    }
    std::vector<bool> resident(backing_size / PageAlignment);
#ifdef _WIN32
    const std::vector<ULONG_PTR> entries = QueryWorkingSetPages();
    if (entries.empty()) {
        return {};
    }
    const auto backing_begin = reinterpret_cast<uintptr_t>(backing_base);
    const auto view_begin = reinterpret_cast<uintptr_t>(impl ? impl->virtual_base : nullptr);
    const size_t view_size = impl ? impl->virtual_size : 0;
    for (size_t i = 1; i < entries.size(); ++i) {
        const uintptr_t page = entries[i] & ~uintptr_t{PageAlignment - 1};
        if (page - backing_begin < backing_size) {
            resident[(page - backing_begin) / PageAlignment] = true;
        } else if (view_begin != 0 && page - view_begin < view_size) {
            if (const auto offset = impl->BackingOffset(page - view_begin);
                offset && *offset < backing_size) {
                resident[*offset / PageAlignment] = true;
            }
        }
    }
#elif defined(__APPLE__) || defined(__linux__)
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t num_pages = backing_size / page_size;
#ifdef __APPLE__
    std::vector<char> residency(num_pages);
#else
    std::vector<unsigned char> residency(num_pages);
#endif
    if (mincore(backing_base, num_pages * page_size, residency.data()) != 0) {
        return {};
    }
    int counted = 1;
#ifdef MINCORE_PAGED_OUT
    counted |= MINCORE_PAGED_OUT;
#endif
    const size_t per_page = page_size / PageAlignment;
    for (size_t index = 0; index < num_pages; ++index) {
        if ((residency[index] & counted) != 0) {
            for (size_t i = 0; i < per_page; ++i) {
                resident[index * per_page + i] = true;
            }
        }
    }
#else
    return {};
#endif
    return resident;
}

bool HostMemory::ResetTouched() const {
    if (backing_base == nullptr) {
        return false;
    }
#ifdef _WIN32
    g_touch_window.store(true, std::memory_order_relaxed);
    // Not VirtualUnlock per range: this is the documented call, and it keeps the contents too.
    return K32EmptyWorkingSet(GetCurrentProcess()) != 0;
#elif defined(__APPLE__) && defined(MINCORE_REFERENCED)
    g_touch_window.store(true, std::memory_order_relaxed);
    // Deactivates the pages and clears their reference bits (per physical page, so a use
    // through any view sets it again); nothing is discarded.
    return madvise(backing_base, backing_size, MADV_DONTNEED) == 0;
#else
    return false;
#endif
}

void HostMemory::EndTouched() const {
    g_touch_window.store(false, std::memory_order_relaxed);
}

std::vector<bool> HostMemory::TouchedPages() const {
    if (backing_base == nullptr || !g_touch_window.load(std::memory_order_relaxed)) {
        return {};
    }
#if defined(__APPLE__) && defined(MINCORE_REFERENCED)
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t num_pages = backing_size / page_size;
    std::vector<char> residency(num_pages);
    if (mincore(backing_base, num_pages * page_size, residency.data()) != 0) {
        return {};
    }
    std::vector<bool> touched(backing_size / PageAlignment);
    const size_t per_page = page_size / PageAlignment;
    for (size_t index = 0; index < num_pages; ++index) {
        if ((residency[index] & MINCORE_REFERENCED) != 0) {
            for (size_t i = 0; i < per_page; ++i) {
                touched[index * per_page + i] = true;
            }
        }
    }
    return touched;
#else
    // Windows: the pages that came back into the working set.
    return ResidentPages();
#endif
}

u64 HostMemory::ZeroFilledBytes() noexcept {
    return g_zero_filled_bytes.load(std::memory_order_relaxed);
}

HostMemory::GuestAccounting HostMemory::ScannedAccounting() noexcept {
    return g_scanned_accounting.load(std::memory_order_relaxed);
}

u64 HostMemory::ScannedResidentBytes() noexcept {
    return g_scanned_resident_bytes.load(std::memory_order_relaxed);
}

u64 HostMemory::TrimmedZeroBytes() noexcept {
    return g_trimmed_zero_bytes.load(std::memory_order_relaxed);
}

u64 HostMemory::ScannedHandedOutBytes() noexcept {
    return g_scanned_handed_out_bytes.load(std::memory_order_relaxed);
}

std::pair<size_t, size_t> HostMemory::TrimZeroPages(
    [[maybe_unused]] ZeroTrimState* state,
    [[maybe_unused]] const std::function<bool()>& pause) const {
    if (g_touch_window.load(std::memory_order_relaxed)) {
        return {0, 0};
    }
#ifdef _WIN32
    if (backing_base == nullptr) {
        return {0, 0};
    }
    ZeroTrimState local;
    ZeroTrimState& s = state != nullptr ? *state : local;
    auto& entries = s.entries;
    if (entries.size() < (size_t{1} << 20)) {
        entries.resize(size_t{1} << 20);
    }
    for (int attempt = 0;; ++attempt) {
        if (K32QueryWorkingSet(GetCurrentProcess(), entries.data(),
                               static_cast<DWORD>(entries.size() * sizeof(ULONG_PTR)))) {
            break;
        }
        if (GetLastError() != ERROR_BAD_LENGTH || attempt == 3) {
            return {0, 0};
        }
        entries.resize((std::max)(entries.size() * 2, size_t{entries[0]} + (size_t{1} << 18)));
    }
    const size_t count = (std::min)(size_t{entries[0]}, entries.size() - 1);
    const auto backing_begin = reinterpret_cast<uintptr_t>(backing_base);
    const auto view_begin = reinterpret_cast<uintptr_t>(impl ? impl->virtual_base : nullptr);
    const size_t view_size = impl ? impl->virtual_size : 0;
    auto& backing_pages = s.backing_pages;
    auto& view_pages = s.view_pages;
    backing_pages.clear();
    view_pages.clear();
    for (size_t i = 1; i <= count; ++i) {
        const uintptr_t page = entries[i] & ~uintptr_t{PageAlignment - 1};
        if (page - backing_begin < backing_size) {
            backing_pages.push_back(page);
        } else if (view_begin != 0 && page - view_begin < view_size) {
            view_pages.push_back(page);
        }
    }
    std::sort(backing_pages.begin(), backing_pages.end());
    std::sort(view_pages.begin(), view_pages.end());

    // Pages are only ever read through the backing, which stays mapped and writable: a view page
    // can be unmapped or protected by a guest thread meanwhile. Trimming keeps the contents, so a
    // write that lands after the check costs a soft fault and nothing else.
    const bool confirm = state != nullptr;
    auto& trim = s.trim;
    auto& zero_now = s.zero_now;
    trim.clear();
    zero_now.clear();
    size_t zero_bytes = 0;
    const auto zero = [&](uintptr_t page) {
        zero_now.push_back(page);
        if (!confirm || std::binary_search(s.zero_before.begin(), s.zero_before.end(), page)) {
            trim.push_back(page);
            zero_bytes += PageAlignment;
        }
    };
    // A run can span two views, which VirtualUnlock refuses as a whole: then page by page.
    const auto unlock = [](uintptr_t begin, uintptr_t end) {
        if (VirtualUnlock(reinterpret_cast<void*>(begin), end - begin) ||
            GetLastError() == ERROR_NOT_LOCKED) {
            return;
        }
        for (uintptr_t page = begin; page < end; page += PageAlignment) {
            VirtualUnlock(reinterpret_cast<void*>(page), PageAlignment);
        }
    };
    // Each slice reads at most 4 MiB and unlocks what it found, then the caller may sleep: one
    // pass over a 9 GB working set held a core and the working-set lock for up to 4.7 s.
    constexpr size_t SlicePages = 1024;
    const auto end_slice = [&] {
        std::sort(trim.begin(), trim.end());
        trim.erase(std::unique(trim.begin(), trim.end()), trim.end());
        for (size_t i = 0; i < trim.size();) {
            size_t j = i + 1;
            while (j < trim.size() && trim[j] == trim[j - 1] + PageAlignment) {
                ++j;
            }
            unlock(trim[i], trim[j - 1] + PageAlignment);
            i = j;
        }
        trim.clear();
        return !pause || pause();
    };
    bool finished = true;
    for (size_t i = 0; i < backing_pages.size() && finished; ++i) {
        const uintptr_t page = backing_pages[i];
        if (IsZeroPage(reinterpret_cast<const u8*>(page))) {
            zero(page);
        }
        if ((i + 1) % SlicePages == 0) {
            finished = end_slice();
        }
    }
    for (size_t i = 0; i < view_pages.size() && finished; ++i) {
        const uintptr_t page = view_pages[i];
        const auto offset = impl->BackingOffset(page - view_begin);
        if (offset && *offset < backing_size) {
            const uintptr_t backing_page = backing_begin + *offset;
            if (IsZeroPage(reinterpret_cast<const u8*>(backing_page))) {
                zero(page);
            }
            // Reading it put the backing page in the working set: take it out again.
            if (!std::binary_search(backing_pages.begin(), backing_pages.end(), backing_page)) {
                trim.push_back(backing_page);
            }
        }
        if ((i + 1) % SlicePages == 0) {
            finished = end_slice();
        }
    }
    if (finished) {
        end_slice();
    }
    std::sort(zero_now.begin(), zero_now.end());
    s.zero_before.swap(zero_now);
    return {(backing_pages.size() + view_pages.size()) * PageAlignment, zero_bytes};
#elif defined(__APPLE__)
    // Untagged guest memory (macOS, iOS experiment): XNU counts every mapping of a resident or
    // compressed page in the footprint iOS kills the app over. Zero pages are made reusable:
    // they leave it for every mapping (until MADV_FREE_REUSE; writes leave them dirty, so the OS
    // compresses rather than drops them) and read back as zeros if the OS takes them.
    // Ledger-tagged guest memory (iOS default) is counted per page and ignores MADV_FREE_REUSABLE:
    // the pass only counts what is resident, without reading it. EDEN_NX2_TRIM_ZERO=0 turns it off.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_TRIM_ZERO");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled || backing_base == nullptr) {
        return {0, 0};
    }
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t num_pages = backing_size / page_size;
    std::vector<char> residency(num_pages);
    if (mincore(reinterpret_cast<char*>(backing_base), num_pages * page_size,
                residency.data()) != 0) {
        return {0, 0};
    }
    if (trimmed_pages.size() != num_pages) {
        trimmed_pages.assign(num_pages, false);
    }
    int counted = MINCORE_INCORE;
#ifdef MINCORE_PAGED_OUT
    counted |= MINCORE_PAGED_OUT;
#endif
    const auto is_zero = [page_size](const u8* page) {
        for (size_t offset = 0; offset < page_size; offset += PageAlignment) {
            if (!IsZeroPage(page + offset)) {
                return false;
            }
        }
        return true;
    };
    size_t resident = 0;
    size_t trimmed = 0;
    // A guest write between the check and madvise is the one thing a reclaim could lose: each run
    // is checked again right after, and a page that changed is written to (unchanged) and reused.
    const auto trim = [&](size_t first, size_t end) {
        u8* const begin = backing_base + first * page_size;
        if (madvise(begin, (end - first) * page_size, MADV_FREE_REUSABLE) != 0) {
            return;
        }
        for (size_t index = first; index < end; ++index) {
            u8* const page = backing_base + index * page_size;
            if (!is_zero(page)) {
                // A compare-exchange of a word with itself stores it (an idempotent RMW can be
                // compiled to a load), so the page counts as written again.
                auto* const word = reinterpret_cast<u64*>(page);
                u64 value = __atomic_load_n(word, __ATOMIC_RELAXED);
                while (!__atomic_compare_exchange_n(word, &value, value, false, __ATOMIC_RELAXED,
                                                    __ATOMIC_RELAXED)) {
                }
                madvise(page, page_size, MADV_FREE_REUSE);
                trimmed_pages[index] = false;
                continue;
            }
            if (!trimmed_pages[index]) {
                trimmed_pages[index] = true;
                trimmed += page_size;
            }
        }
    };
    const bool scan = !PagesCountedOnce();
#ifdef LICHII_IOS_MACH_MEMORY
    // Untagged, a page the guest uses through the fastmem view and the emulator (or this scan)
    // read through the backing counts twice. Every 64 MiB the backing mappings of what was
    // scanned go again, so a pass holds at most that much extra. EDEN_NX2_TRIM_UNMAP=0 keeps them.
    static const bool unmap = [] {
        const char* value = std::getenv("EDEN_NX2_TRIM_UNMAP");
        return value == nullptr || value[0] != '0';
    }();
    constexpr size_t UnmapBytes = size_t{64} << 20;
    const size_t unmap_pages = (std::max)(size_t{1}, UnmapBytes / page_size);
    size_t unmap_begin = 0;
    bool unmap_touched = false;
    const auto drop_mappings = [&](size_t end) {
        if (unmap && unmap_touched && scan && impl) {
            impl->DropBackingMappings(unmap_begin * page_size, (end - unmap_begin) * page_size);
        }
        unmap_begin = end;
        unmap_touched = false;
    };
#endif
    const size_t slice_pages = (std::max)(size_t{1}, (size_t{4} << 20) / page_size);
    size_t run_begin = num_pages;
    size_t index = 0;
    for (; index < num_pages; ++index) {
        bool zero = false;
        if ((residency[index] & counted) != 0) {
            resident += page_size;
            if (!scan) {
                continue;
            }
#ifdef LICHII_IOS_MACH_MEMORY
            unmap_touched = true;
#endif
            zero = is_zero(backing_base + index * page_size);
            if (!zero) {
                trimmed_pages[index] = false;
            }
        }
        if (!scan) {
            continue;
        }
        if (zero && run_begin == num_pages) {
            run_begin = index;
        } else if (!zero && run_begin != num_pages) {
            trim(run_begin, index);
            run_begin = num_pages;
        }
        const bool slice_end = pause && (index + 1) % slice_pages == 0;
#ifdef LICHII_IOS_MACH_MEMORY
        const bool unmap_end = (index + 1) % unmap_pages == 0;
#else
        constexpr bool unmap_end = false;
#endif
        if ((slice_end || unmap_end) && run_begin != num_pages) {
            trim(run_begin, index + 1);
            run_begin = num_pages;
        }
#ifdef LICHII_IOS_MACH_MEMORY
        if (unmap_end) {
            drop_mappings(index + 1);
        }
#endif
        // Slices of 4 MiB, like Windows: the caller may sleep between them.
        if (slice_end && !pause()) {
            ++index;
            break;
        }
    }
    if (run_begin != num_pages) {
        trim(run_begin, (std::min)(index, num_pages));
    }
#ifdef LICHII_IOS_MACH_MEMORY
    drop_mappings((std::min)(index, num_pages));
    if (!scan && impl && impl->Chunked()) {
        trimmed = impl->ReleaseZeroChunks(residency, page_size, counted, pause);
        resident -= (std::min)(resident, trimmed);
    }
#endif
    u64 handed_out = 0;
    for (size_t granule = 0; granule < num_granules; ++granule) {
        if ((granules[granule].load(std::memory_order_relaxed) & GranuleUsed) != 0) {
            handed_out += u64{1} << GranuleShift;
        }
    }
    g_scanned_handed_out_bytes.store(handed_out, std::memory_order_relaxed);
    g_scanned_resident_bytes.store(resident, std::memory_order_relaxed);
    g_scanned_accounting.store(FileBacked()         ? GuestAccounting::File
                               : PagesCountedOnce() ? GuestAccounting::Once
                                                    : GuestAccounting::PerMapping,
                               std::memory_order_relaxed);
    g_trimmed_zero_bytes.fetch_add(trimmed, std::memory_order_relaxed);
    return {resident, trimmed};
#else
    return {0, 0};
#endif
}

bool HostMemory::PagesCountedOnce() const noexcept {
#ifdef LICHII_IOS_MACH_MEMORY
    return impl && (impl->ledger_tagged || impl->file_backed);
#else
    return false;
#endif
}

bool HostMemory::FileBacked() const noexcept {
#ifdef LICHII_IOS_MACH_MEMORY
    return impl && impl->file_backed;
#else
    return false;
#endif
}

bool HostMemory::TrimOnlyCounts() const noexcept {
#ifdef LICHII_IOS_MACH_MEMORY
    return impl && (impl->ledger_tagged || impl->file_backed) && !impl->Chunked();
#else
    return false;
#endif
}

u64 HostMemory::ZeroAdvisedBytes() noexcept {
    return g_zero_advised_bytes.load(std::memory_order_relaxed);
}

#ifdef __APPLE__
#ifndef MADV_ZERO
#define MADV_ZERO 11
#endif
bool HostMemory::ZeroWithoutFaulting(size_t offset, size_t length) {
    // MADV_ZERO (iOS 17, macOS 14) zeroes resident pages in place and drops compressed ones
    // without faulting the rest in, where a memset gives every page RAM (and, untagged, a second
    // charged mapping). Untagged, the cleared whole pages are also made reusable.
    static std::atomic<bool> unsupported{};
    static std::atomic<bool> logged{};
    static const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    // A file's pages can't be zeroed this way (vm_object_zero_preflight: anonymous memory only).
    if (unsupported.load(std::memory_order_relaxed) || FileBacked()) {
        return false;
    }
    const size_t begin = Common::AlignUp(offset, page_size);
    const size_t end = Common::AlignDown(offset + length, page_size);
    if (begin >= end) {
        return false;
    }
    if (madvise(backing_base + begin, end - begin, MADV_ZERO) != 0) {
        const int error = errno;
        if (error == EINVAL || error == ENOTSUP) {
            unsupported.store(true, std::memory_order_relaxed);
            LOG_INFO(Common_Memory, "MADV_ZERO unavailable ({}): guest memory is cleared with memset",
                     strerror(error));
        }
        return false;
    }
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        LOG_INFO(Common_Memory, "Guest memory is cleared with MADV_ZERO (no pages faulted in)");
    }
    std::memset(backing_base + offset, 0, begin - offset);
    std::memset(backing_base + end, 0, offset + length - end);
    if (!PagesCountedOnce()) {
        madvise(backing_base + begin, end - begin, MADV_FREE_REUSABLE);
    }
    g_zero_filled_bytes.fetch_add(length - (end - begin), std::memory_order_relaxed);
    g_zero_advised_bytes.fetch_add(end - begin, std::memory_order_relaxed);
    return true;
}
#endif

void HostMemory::ClearBackingRegion(size_t physical_offset, size_t length, u32 fill_value) {
    if (length == 0) {
        return;
    }
    MarkUsed(physical_offset, length);
    if (fill_value != 0) {
        std::memset(backing_base + physical_offset, fill_value, length);
        MarkDirty(physical_offset, length);
        return;
    }
    // Fresh backing memory is zero and the host gives it no RAM until it's touched: clear only
    // the granules that may hold data (title 1's heap made the host commit every page of it).
    // A granule cleared in full is zero again.
#ifdef __linux__
    // Handing a whole granule back to the kernel frees its RAM; a memset would commit it.
    static const bool punch = [] {
        const char* value = std::getenv("EDEN_PUNCH_HOLE");
        return value == nullptr || value[0] != '0';
    }();
#endif
    const size_t end = physical_offset + length;
    for (size_t offset = physical_offset; offset < end;) {
        const size_t index = offset >> GranuleShift;
        const size_t granule_begin = index << GranuleShift;
        const size_t granule_end = granule_begin + (size_t{1} << GranuleShift);
        const size_t clear_end = (std::min)(granule_end, end);
        if ((granules[index].load(std::memory_order_acquire) & GranuleDirty) != 0) {
            const bool whole = offset == granule_begin && clear_end == granule_end;
            bool cleared = false;
#ifdef __APPLE__
            const bool released = (granules[index].fetch_and(static_cast<u8>(~GranuleReleased),
                                                             std::memory_order_acq_rel) &
                                   GranuleReleased) != 0;
            cleared = ZeroWithoutFaulting(offset, clear_end - offset);
            if (!cleared && FileBacked()) {
                const size_t written = ZeroNonZeroPages(backing_base + offset, clear_end - offset);
                g_zero_filled_bytes.fetch_add(written, std::memory_order_relaxed);
                cleared = true;
                static std::atomic<bool> logged{};
                if (!logged.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(HW_Memory, "Guest file: first clear ({:#x} bytes at file offset {:#x}, "
                                        "{:#x} of them held data and were rewritten)",
                             clear_end - offset, offset, written);
                }
            }
            if (!cleared && released) {
                madvise(backing_base + granule_begin,
                        (std::min)(granule_end, backing_size) - granule_begin, MADV_FREE_REUSE);
            }
#endif
#ifdef __linux__
            cleared = whole && punch && impl && impl->PunchHole(granule_begin, granule_end - granule_begin);
#endif
            if (!cleared) {
                std::memset(backing_base + offset, 0, clear_end - offset);
                g_zero_filled_bytes.fetch_add(clear_end - offset, std::memory_order_relaxed);
            }
            if (whole) {
                granules[index].fetch_and(static_cast<u8>(~GranuleDirty), std::memory_order_release);
            }
        }
        offset = clear_end;
    }
}

void HostMemory::MarkUsed(size_t physical_offset, size_t length) {
    if (length == 0) {
        return;
    }
    const size_t last = (physical_offset + length - 1) >> GranuleShift;
    for (size_t index = physical_offset >> GranuleShift; index <= last; ++index) {
        if ((granules[index].load(std::memory_order_relaxed) & GranuleUsed) == 0) {
            granules[index].fetch_or(GranuleUsed, std::memory_order_release);
        }
    }
}

void HostMemory::MarkDirty(size_t physical_offset, size_t length) {
    if (length == 0) {
        return;
    }
    const size_t last = (physical_offset + length - 1) >> GranuleShift;
    for (size_t index = physical_offset >> GranuleShift; index <= last; ++index) {
        granules[index].fetch_or(GranuleDirty, std::memory_order_release);
    }
}

void HostMemory::ReleaseFreed(size_t physical_offset, size_t length) {
#ifdef __APPLE__
    // Nothing else hands freed guest memory back on Apple: the footprint jetsam watches only grew
    // to the session's peak. EDEN_NX2_RELEASE_FREED=0 turns it off.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RELEASE_FREED");
        return value == nullptr || value[0] != '0';
    }();
    // Ledger-tagged guest memory ignores MADV_FREE_REUSABLE.
    if (!enabled || length == 0 || backing_base == nullptr || PagesCountedOnce()) {
        return;
    }
    // Whole granules only: the rest of a partial one may still be in use. They stay Dirty, so the
    // next allocation clears them whatever the OS left in them.
    constexpr size_t GranuleSize = size_t{1} << GranuleShift;
    const size_t first = Common::AlignUp(physical_offset, GranuleSize);
    const size_t end = (std::min)(Common::AlignDown(physical_offset + length, GranuleSize),
                                  Common::AlignDown(backing_size, GranuleSize));
    if (first >= end) {
        return;
    }
    for (size_t offset = first; offset < end; offset += GranuleSize) {
        granules[offset >> GranuleShift].fetch_or(GranuleReleased, std::memory_order_release);
    }
    if (madvise(backing_base + first, end - first, MADV_FREE_REUSABLE) != 0) {
        return;
    }
    static std::atomic<u64> released{};
    constexpr u64 LogEvery = u64{1} << 30;
    const u64 before = released.fetch_add(end - first, std::memory_order_relaxed);
    if (before / LogEvery != (before + (end - first)) / LogEvery) {
        task_vm_info_data_t info{};
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        const bool have_info = task_info(mach_task_self(), TASK_VM_INFO,
                                         reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS;
        LOG_INFO(Common_Memory,
                 "Freed guest memory handed back to the OS: {} MiB so far (footprint {} MiB, "
                 "reusable {} MiB)",
                 (before + (end - first)) >> 20, have_info ? info.phys_footprint >> 20 : 0,
                 have_info ? info.reusable >> 20 : 0);
    }
#else
    static_cast<void>(physical_offset);
    static_cast<void>(length);
#endif
}

void HostMemory::ForgetContents() {
    for (size_t index = 0; index < num_granules; ++index) {
        if ((granules[index].load(std::memory_order_relaxed) & GranuleUsed) != 0) {
            granules[index].fetch_or(GranuleDirty, std::memory_order_release);
        }
    }
}

void HostMemory::EnableDirectMappedAddress() {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    if (impl) {
        impl->EnableDirectMappedAddress();
        virtual_size += reinterpret_cast<uintptr_t>(virtual_base);
    }
#endif
}

} // namespace Common
