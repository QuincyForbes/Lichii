// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/guest_chunks.h"

namespace {

constexpr size_t ChunkSize = 0x200000;

struct Piece {
    size_t chunk;
    size_t offset;
    size_t size;
    size_t done;

    bool operator==(const Piece&) const = default;
};

std::vector<Piece> Split(size_t offset, size_t size) {
    std::vector<Piece> pieces;
    const bool all = Common::ForEachChunkPiece(
        offset, size, ChunkSize, [&](size_t chunk, size_t in_chunk, size_t piece, size_t done) {
            pieces.push_back({chunk, in_chunk, piece, done});
            return true;
        });
    REQUIRE(all);
    return pieces;
}

} // namespace

TEST_CASE("Guest chunks: a range inside one chunk is one piece", "[common]") {
    REQUIRE(Split(0, 0).empty());
    REQUIRE(Split(0, ChunkSize) == std::vector<Piece>{{0, 0, ChunkSize, 0}});
    REQUIRE(Split(5 * ChunkSize + 0x4000, 0x8000) ==
            std::vector<Piece>{{5, 0x4000, 0x8000, 0}});
    REQUIRE(Split(3 * ChunkSize - 0x1000, 0x1000) ==
            std::vector<Piece>{{2, ChunkSize - 0x1000, 0x1000, 0}});
}

TEST_CASE("Guest chunks: a range over chunk boundaries splits at each", "[common]") {
    REQUIRE(Split(ChunkSize - 0x4000, 0x8000) ==
            std::vector<Piece>{{0, ChunkSize - 0x4000, 0x4000, 0}, {1, 0, 0x4000, 0x4000}});
    const std::vector<Piece> pieces = Split(7 * ChunkSize + 0x10000, 3 * ChunkSize);
    REQUIRE(pieces == std::vector<Piece>{
                          {7, 0x10000, ChunkSize - 0x10000, 0},
                          {8, 0, ChunkSize, ChunkSize - 0x10000},
                          {9, 0, ChunkSize, 2 * ChunkSize - 0x10000},
                          {10, 0, 0x10000, 3 * ChunkSize - 0x10000},
                      });
    size_t total = 0;
    for (const Piece& piece : pieces) {
        REQUIRE(piece.done == total);
        REQUIRE(piece.offset + piece.size <= ChunkSize);
        total += piece.size;
    }
    REQUIRE(total == 3 * ChunkSize);
}

TEST_CASE("Guest chunks: the split stops when the callback fails", "[common]") {
    size_t calls = 0;
    const bool all = Common::ForEachChunkPiece(0, 4 * ChunkSize, ChunkSize,
                                               [&](size_t chunk, size_t, size_t, size_t) {
                                                   ++calls;
                                                   return chunk != 1;
                                               });
    REQUIRE_FALSE(all);
    REQUIRE(calls == size_t{2});
}

TEST_CASE("Guest chunks: a chunk is due after consecutive passes without data", "[common]") {
    Common::ChunkCandidates candidates{4};
    REQUIRE(candidates.Size() == size_t{4});
    // Never held memory: nothing to give back.
    REQUIRE_FALSE(candidates.Observe(0, false, false));
    REQUIRE_FALSE(candidates.Observe(0, false, false));
    // Zeros once is not enough.
    REQUIRE_FALSE(candidates.Observe(1, true, false));
    REQUIRE(candidates.Observe(1, true, false));
    // Still due while it stays that way and is not released.
    REQUIRE(candidates.Observe(1, true, false));
    // Data in between starts over.
    REQUIRE_FALSE(candidates.Observe(2, true, false));
    REQUIRE_FALSE(candidates.Observe(2, true, true));
    REQUIRE_FALSE(candidates.Observe(2, true, false));
    REQUIRE(candidates.Observe(2, true, false));
    // So does a pass that found the chunk empty.
    REQUIRE_FALSE(candidates.Observe(3, true, false));
    REQUIRE_FALSE(candidates.Observe(3, false, false));
    REQUIRE_FALSE(candidates.Observe(3, true, false));
    REQUIRE(candidates.Observe(3, true, false));
}

TEST_CASE("Guest chunks: a chunk released again and again waits longer", "[common]") {
    Common::ChunkCandidates candidates{1};
    const std::array<int, 6> expected{2, 4, 8, 16, 16, 16};
    for (const int passes : expected) {
        REQUIRE(int{candidates.RequiredPasses(0)} == passes);
        for (int pass = 1; pass < passes; ++pass) {
            REQUIRE_FALSE(candidates.Observe(0, true, false));
        }
        REQUIRE(candidates.Observe(0, true, false));
        candidates.Released(0);
        // Given back: no memory on the next pass.
        REQUIRE_FALSE(candidates.Observe(0, false, false));
    }
}
