// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <functional>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/gpu_dirty_memory_manager.h"

TEST_CASE("GPUDirtyMemoryManager reports collected writes until they are gathered",
          "[core][gpu_dirty_memory]") {
    Core::GPUDirtyMemoryManager manager;
    manager.Collect(0x10040, 0x10);
    manager.Collect(0x20000, 0x40);

    REQUIRE(manager.Overlaps(0x10040, 1));
    REQUIRE(manager.Overlaps(0x10000, 0x1000));
    REQUIRE(manager.Overlaps(0x20000, 4));
    REQUIRE_FALSE(manager.Overlaps(0x10000, 0x40));
    REQUIRE_FALSE(manager.Overlaps(0x10080, 0x40));
    REQUIRE_FALSE(manager.Overlaps(0x30000, 0x1000));

    std::vector<std::pair<PAddr, size_t>> gathered;
    std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
        gathered.emplace_back(address, size);
    };
    manager.Gather(callback);

    REQUIRE(gathered.size() == 2);
    REQUIRE_FALSE(manager.Overlaps(0x10000, 0x1000));
    REQUIRE_FALSE(manager.Overlaps(0x20000, 4));
}

TEST_CASE("GPUDirtyMemoryManager collects a write crossing a 2 KiB page whole",
          "[core][gpu_dirty_memory]") {
    // Ported from zortos293/Lichii#1586 (nx2_backx): a transform covers 2 KiB in 64-byte bits, and
    // a write past its page's 32 bits must still be gathered to its last byte.
    Core::GPUDirtyMemoryManager manager;
    std::vector<std::pair<PAddr, size_t>> gathered;
    std::function<void(PAddr, size_t)> callback = [&](PAddr address, size_t size) {
        gathered.emplace_back(address, size);
    };
    for (const auto& [offset, size] :
         {std::pair<u64, u64>{600, 1960}, {0, 4096}, {1000, 3000}, {2047, 2}, {64, 2048},
          {2044, 8}}) {
        CAPTURE(offset, size);
        manager.Collect(0x7000, 0x40); // Something already collected on another page
        manager.Collect(0x10000 + offset, size);
        gathered.clear();
        manager.Gather(callback);
        std::set<u64> blocks;
        for (const auto& [address, length] : gathered) {
            for (u64 block = address >> 6; block <= (address + length - 1) >> 6; ++block) {
                blocks.insert(block);
            }
        }
        for (u64 block = (0x10000 + offset) >> 6; block <= (0x10000 + offset + size - 1) >> 6;
             ++block) {
            REQUIRE(blocks.contains(block));
        }
        gathered.clear();
        manager.Gather(callback);
        REQUIRE(gathered.empty());
    }
}