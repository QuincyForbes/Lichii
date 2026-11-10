// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "common/common_types.h"

namespace Common {

/// Calls func(chunk, offset_in_chunk, size, done) for each piece of [offset, offset + size) of a
/// linear backing made of chunk_size chunks, in order; done is the bytes before the piece. Stops
/// and returns false when func does.
template <typename Func>
bool ForEachChunkPiece(size_t offset, size_t size, size_t chunk_size, Func&& func) {
    size_t done = 0;
    while (done < size) {
        const size_t position = offset + done;
        const size_t in_chunk = position % chunk_size;
        const size_t piece = (std::min)(size - done, chunk_size - in_chunk);
        if (!func(position / chunk_size, in_chunk, piece, done)) {
            return false;
        }
        done += piece;
    }
    return true;
}

/// Which chunks of the guest backing may be given back to the host: one that held memory but no
/// data on consecutive passes. A chunk the guest keeps filling with zeros again (scratch memory)
/// needs twice as many passes after each release, up to MaxPasses.
class ChunkCandidates {
public:
    static constexpr u8 FirstPasses = 2;
    static constexpr u8 MaxPasses = 16;

    explicit ChunkCandidates(size_t num_chunks = 0) : chunks(num_chunks) {}

    size_t Size() const noexcept {
        return chunks.size();
    }

    /// What a pass saw of a chunk. True when it is due for release.
    bool Observe(size_t chunk, bool has_memory, bool has_data) noexcept {
        Entry& entry = chunks[chunk];
        if (!has_memory || has_data) {
            entry.passes = 0;
            return false;
        }
        entry.passes = static_cast<u8>((std::min)(entry.passes + 1, int{MaxPasses}));
        return entry.passes >= Required(entry);
    }

    /// The chunk was given back.
    void Released(size_t chunk) noexcept {
        Entry& entry = chunks[chunk];
        entry.passes = 0;
        entry.releases = static_cast<u8>((std::min)(entry.releases + 1, 3));
    }

    /// Passes without data a chunk needs before it is due.
    u8 RequiredPasses(size_t chunk) const noexcept {
        return Required(chunks[chunk]);
    }

private:
    struct Entry {
        u8 passes;
        u8 releases;
    };

    static u8 Required(const Entry& entry) noexcept {
        return static_cast<u8>((std::min)(int{FirstPasses} << entry.releases, int{MaxPasses}));
    }

    std::vector<Entry> chunks;
};

} // namespace Common
