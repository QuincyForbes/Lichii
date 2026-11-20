// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The pipelines a run builds, kept per game in an MTLBinaryArchive and loaded at the next boot,
// so Metal finds their GPU code instead of compiling it again (milestone 5, part one; the MSL
// is still compiled at run time). Plain C++ header.

#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "common/common_types.h"

namespace Metal {

class Device;

class PipelineArchive {
public:
    struct Counts {
        u64 hits;      ///< Pipelines found in the archive
        u64 misses;    ///< Built without it
        u64 hit_ns;    ///< Time the hits took
        u64 miss_ns;   ///< Time the misses took
        u64 added;     ///< Misses added to the archive
        u64 add_failures;
    };

    PipelineArchive();
    ~PipelineArchive();

    PipelineArchive(const PipelineArchive&) = delete;
    PipelineArchive& operator=(const PipelineArchive&) = delete;

    /// Opens the archive file (an empty archive when it is missing or Metal can't read it).
    /// Closed when EDEN_NX2_METAL_ARCHIVE=0. With `read_only`, misses do not recompile into the
    /// archive (avoiding exclusive writer lock stalls during live gameplay).
    bool Open(const Device& device, const std::filesystem::path& path, bool read_only = false);

    [[nodiscard]] bool IsOpen() const noexcept {
        return is_open.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool HasEntries() const noexcept {
        return has_entries.load(std::memory_order_relaxed);
    }

    void SetReadOnly(bool read_only_) noexcept {
        read_only.store(read_only_, std::memory_order_relaxed);
    }

    [[nodiscard]] bool IsReadOnly() const noexcept {
        return read_only.load(std::memory_order_relaxed);
    }

    /// Waits for all queued DeferAdd jobs to finish.
    void FlushAdds();

    /// Writes the archive when pipelines were added since it was opened or last saved.
    void Save();

    /// The active writable MTLBinaryArchive (id<MTLBinaryArchive>), or null when closed.
    [[nodiscard]] void* Handle() const noexcept;

    /// NSArray<id<MTLBinaryArchive>>* of all populated archive shards for fast multi-shard lookup,
    /// or null when closed / empty. Must be accessed under shared BuildMutex.
    [[nodiscard]] void* LookupHandles() const noexcept;

    /// Held shared while a pipeline is built with the archive attached, and alone while one is
    /// added to it or the file is written (boot builds on several threads).
    [[nodiscard]] std::shared_mutex& BuildMutex() noexcept {
        return build_mutex;
    }

    /// A pipeline built with the archive attached: found in it or not, and how long it took.
    void NoteBuild(bool hit, u64 nanoseconds) noexcept;
    void NoteAdded(bool added) noexcept;

    /// Queues an archive addition onto the current active shard (rolling over to a new <= 128-PSO
    /// shard automatically so FlatBuffers serialization stays < 150 ms). Runs alone under BuildMutex.
    void DeferAdd(std::function<bool(void* archive_handle)> add);
    void DeferAdd(std::function<bool()> add);

    /// Counts since the last call (the renderer's 10 s report).
    [[nodiscard]] Counts TakeCounts() noexcept;

    /// Counts since the archive was opened.
    [[nodiscard]] Counts Totals() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    mutable std::mutex mutex;
    std::shared_mutex build_mutex;
    Counts recent{};
    Counts total{};
    bool dirty{};
    std::atomic_bool is_open{false};
    std::atomic_bool read_only{false};
    std::atomic_bool has_entries{false};
};

} // namespace Metal
