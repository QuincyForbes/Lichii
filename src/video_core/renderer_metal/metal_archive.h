// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The pipelines a run builds, kept per game in an MTLBinaryArchive and loaded at the next boot,
// so Metal finds their GPU code instead of compiling it again (milestone 5, part one; the MSL
// is still compiled at run time). Plain C++ header.

#pragma once

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
    /// Closed unless EDEN_NX2_METAL_ARCHIVE=1.
    bool Open(const Device& device, const std::filesystem::path& path);

    [[nodiscard]] bool IsOpen() const noexcept;

    /// Writes the archive when pipelines were added since it was opened or last saved.
    void Save();

    /// The MTLBinaryArchive (id<MTLBinaryArchive>), or null when closed.
    [[nodiscard]] void* Handle() const noexcept;

    /// Held shared while a pipeline is built with the archive attached, and alone while one is
    /// added to it or the file is written (boot builds on several threads).
    [[nodiscard]] std::shared_mutex& BuildMutex() noexcept {
        return build_mutex;
    }

    /// A pipeline built with the archive attached: found in it or not, and how long it took.
    void NoteBuild(bool hit, u64 nanoseconds) noexcept;
    void NoteAdded(bool added) noexcept;

    /// Queues an archive addition (it compiles the pipeline again into the archive) for one
    /// background thread, so builders never wait on it or block one another; the function
    /// returns whether the addition succeeded. It runs alone against BuildMutex.
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
};

} // namespace Metal
