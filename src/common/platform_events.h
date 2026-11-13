// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "common/common_types.h"

// Host platform signals the engine reacts to: memory pressure (iOS memory warnings and the
// per-process limit), the app leaving the foreground and the display's refresh rate. A frontend
// feeds them; nothing here changes behaviour until it does.
namespace Common::PlatformEvents {

/// A low-memory warning from the OS: the caches collect aggressively for a while. Signals
/// within 2 s of the last reaction are folded into it.
void SignalMemoryPressure();
/// Bumped by every reaction of SignalMemoryPressure, including those PollMemoryPressure raises.
u32 MemoryPressureGeneration();
/// Listeners run inline during SignalMemoryPressure (must be fast and lock-safe).
std::size_t AddMemoryPressureListener(std::function<void()> listener);
void RemoveMemoryPressureListener(std::size_t id);

/// Bytes the OS still lets this process allocate before it is killed (iOS), 0 where unknown.
u64 AvailableProcessMemory();
/// Signal pressure whenever AvailableProcessMemory falls below `bytes` (0, the default: never).
void SetMemoryLowWater(u64 bytes);
/// The mark SetMemoryLowWater set (0: none).
u64 MemoryLowWater();
/// Checks the low-water mark at most every 250 ms. Called once per frame by the caches.
void PollMemoryPressure();

/// Guest memory is in a file (iOS): outside the footprint, so AvailableProcessMemory does not see
/// it, but iOS takes RAM for the rest of the process from its pages.
void SetGuestInFile(bool in_file);
bool GuestInFile();
/// What the caches leave to a guest memory file on top of their own reserve: 0 unless the guest
/// is in a file. It grows while the process keeps paging the file back in (the caches crowd it
/// out), signalling memory pressure, and shrinks back when that stops. EDEN_NX2_GUEST_FILE_RESERVE_MIB
/// (base, default 1024 on iOS; 0 off), _MAX_MIB (default 3072), EDEN_NX2_PAGEIN_PRESSURE (page
/// -ins a second that count as thrashing, default 1500; 0 off); the rest in GuestReserveTracker.
u64 GuestFileReserve();
/// The part of `reserve` the caches give up with `left` bytes left to the process and `own` of
/// it kept for themselves: at most EDEN_NX2_GUEST_RESERVE_SHARE percent (default 50, 100 = all)
/// of the room past `own`. An iPhone whose reserve had grown to 3072 MiB saw "1 MiB left" with
/// 1.4-4 GiB left and collected textures and buffers every frame.
u64 BoundedGuestFileReserve(u64 reserve, u64 left, u64 own);

/// When the guest memory file is being thrashed, from the process's page-in counter and the
/// file pages it holds. Loading pages the file in too (boot and the title screen: 1,500-21,000
/// page-ins a second on an iPhone, which grew the reserve to its maximum in 10 s and kept it
/// there all session), but what it pages in stays: the file's resident pages grow. Thrashing
/// pages in what was taken away, with the resident pages flat or falling. The reserve grows one
/// step per `sustain` seconds of that, and gives a step back per `decay` seconds under half the
/// threshold (EDEN_NX2_PAGEIN_SUSTAIN_S, default 5; EDEN_NX2_RESERVE_DECAY_S, default 10;
/// EDEN_NX2_RESERVE_RATCHET=1: the earlier rule, a step per second over the threshold and back
/// after 30 s under a quarter of it).
class GuestReserveTracker {
public:
    struct Config {
        u64 base;              ///< Bytes
        u64 max;               ///< Bytes
        u64 threshold;         ///< Page-ins a second
        u32 sustain_seconds;
        u32 decay_seconds;
        u64 step;              ///< Bytes a change moves the reserve
        /// The file's resident pages growing this fast (bytes a second) is loading, not thrashing
        u64 loading_growth;
        bool ratchet;          ///< The earlier rule
    };
    enum class Change { None, Grew, Shrank };

    explicit GuestReserveTracker(const Config& config_) : config{config_} {}

    /// One sample of the counters (only ever growing, but for `file_resident`), at least a
    /// second after the last; the first only sets the start.
    Change Sample(long long now_ns, u64 pageins, u64 file_resident);
    void Reset();

    u64 Extra() const {
        return extra;
    }
    double LastRate() const {
        return last_rate;
    }

private:
    Config config;
    u64 extra{};
    u64 last_pageins{};
    u64 last_resident{};
    long long last_ns{};
    long long thrash_since_ns{};
    long long quiet_since_ns{};
    double last_rate{};
};

struct ProcessMemory {
    u64 footprint;      ///< What jetsam compares with the limit (phys_footprint)
    u64 footprint_peak;
    u64 compressed;
    u64 reusable;       ///< Marked reusable (MADV_FREE_REUSABLE), outside the footprint
    u64 internal;       ///< Anonymous memory mapped by the process, per mapping (pmap ledger)
    u64 purgeable_nonvolatile;
    u64 external;       ///< Resident pages of mapped files, outside the footprint
    u64 graphics;       ///< Graphics ledger (Metal, IOSurface), part of the footprint
};
/// Apple: the process footprint and parts of it. Windows: the working set and its peak. Zeros
/// elsewhere.
ProcessMemory QueryProcessMemory();
/// Counters that only grow, for rates: how often the process waits on memory and storage.
struct ProcessEvents {
    bool known;
    u64 faults;         ///< Page faults of every kind (Apple: 32-bit, wraps)
    u64 pageins;        ///< Of them, pages read in from a file or swap (Linux: major faults)
    u64 cow_faults;     ///< Copy-on-write faults (Apple)
    u64 decompressions; ///< Pages taken back from the memory compressor (Apple)
    u64 storage_read;   ///< Bytes read from storage (Windows: every read, cached ones too)
    u64 storage_written;
    u64 file_resident;  ///< Resident pages of mapped files (Apple), not a counter
};
ProcessEvents QueryProcessEvents();

/// Apple: the footprint's limit (`left` from AvailableProcessMemory) and an estimate of its
/// split into guest memory (as of the last zero trimming pass), the JIT pool, graphics and the
/// rest, for the log.
std::string FootprintSplit(const ProcessMemory& memory, u64 left);
/// macOS: hands the memory malloc holds free back to the OS. Returns the bytes released (0 on
/// iOS, where libmalloc has nothing to do here).
u64 RelieveMallocPressure();

/// Where guest memory is mapped (its backing and the fastmem arena), so that LogMemoryBreakdown
/// lists it under its own name and leaves its pages uncounted. Sizes of 0 forget a range.
void SetGuestMemoryRanges(const void* backing, std::size_t backing_size, const void* arena,
                          std::size_t arena_size);
/// Apple: logs what the footprint is made of beyond FootprintSplit: the process's memory regions
/// by kind, malloc's zones, host threads, Metal and the bytes written to storage. It asks the
/// kernel about every page outside guest memory, which takes tens of milliseconds: not for a
/// thread the game waits for. Nothing elsewhere.
void LogMemoryBreakdown(const char* reason);
/// A line LogMemoryBreakdown adds after its own (the renderer's memory by use); empty: none.
void SetGraphicsMemoryReport(std::function<std::string()> report);
/// Apple: logs the bytes this process wrote to storage since it started (guest memory kept in a
/// file is written there when iOS needs its RAM).
void LogStorageWrites(const char* reason);

/// Work to finish before the app is suspended (write caches out). Listeners run on the thread
/// calling NotifyEnteringBackground; they must not wait for the emulated CPU or the GPU thread.
std::size_t AddBackgroundListener(std::function<void()> listener);
void RemoveBackgroundListener(std::size_t id);
void NotifyEnteringBackground();

/// The display's highest refresh rate (UIScreen.maximumFramesPerSecond), 0 when unknown.
void SetDisplayMaxFps(u32 fps);
u32 DisplayMaxFps();

/// Live thermal / low-power FPS cap (0 = uncapped; never modifies saved user settings).
void SetThermalMaxFps(u32 fps);
u32 ThermalMaxFps();

} // namespace Common::PlatformEvents
