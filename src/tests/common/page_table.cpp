// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"

TEST_CASE("Page table block ids do not wrap", "[common]") {
    using Entry = Common::PageTable::PageEntryData;

    REQUIRE(Entry::BlockIdForMapping(0) == 0);
    REQUIRE(Entry::BlockIdForMapping(65534) == 65534);
    REQUIRE(Entry::BlockIdForMapping(65535) == Entry::INVALID_BLOCK_ID);
    REQUIRE(Entry::BlockIdForMapping(65536) == Entry::INVALID_BLOCK_ID);
    REQUIRE(Entry::BlockIdForMapping((std::numeric_limits<u64>::max)()) ==
            Entry::INVALID_BLOCK_ID);
}

TEST_CASE("Page table fast spans reject exhausted block ids", "[common]") {
    using Entry = Common::PageTable::PageEntryData;

    REQUIRE(Entry::CanFastSpan(7, 7));
    REQUIRE_FALSE(Entry::CanFastSpan(7, 8));
    REQUIRE_FALSE(Entry::CanFastSpan(Entry::INVALID_BLOCK_ID, Entry::INVALID_BLOCK_ID));
}

TEST_CASE("Page table SIGN_BIT matches ATTRIBUTE_MASK and sign-extends negative deltas", "[common]") {
    using Entry = Common::PageTable::PageEntryData;

    STATIC_REQUIRE(Common::PageTable::SIGN_BIT == 55);
    STATIC_REQUIRE((1ULL << Common::PageTable::SIGN_BIT) ==
                   (Common::PageTable::ATTRIBUTE_MASK & ~(Common::PageTable::ATTRIBUTE_MASK >> 1)));

    Entry entry{};

    // Positive 48-bit user-space pointer
    constexpr uintptr_t pos_ptr = 0x0000'7FF6'1234'5000ULL;
    entry.Store(false, Common::PageType::Memory, 0xBEEF, pos_ptr);
    REQUIRE(entry.Pointer() == pos_ptr);
    REQUIRE(entry.Type() == Common::PageType::Memory);
    REQUIRE(entry.Block() == 0xBEEF);

    // Negative delta (e.g. iOS 12 GiB backing minus 64 GiB guest vaddr = 0xFFFF'FFF3'0000'0000)
    constexpr uintptr_t host_backing = 0x0000'0003'0000'0000ULL;
    constexpr u64 guest_vaddr = 0x0000'0010'0000'0000ULL;
    const uintptr_t neg_delta = static_cast<uintptr_t>(host_backing - guest_vaddr);
    entry.Store(false, Common::PageType::Memory, 0x4321, neg_delta);
    REQUIRE(entry.Pointer() == neg_delta);
    REQUIRE(static_cast<u64>(entry.Pointer()) + guest_vaddr == host_backing);
    REQUIRE(entry.Block() == 0x4321);

    // Marked entry returns 0 unless ignore_marked is true
    entry.MarkRasterizerCached();
    REQUIRE(entry.Pointer() == 0);
    REQUIRE(entry.Pointer(true) == neg_delta);

    // Unmapped entry with non-zero block id returns 0
    entry.Store(false, Common::PageType::Unmapped, Entry::INVALID_BLOCK_ID, 0);
    REQUIRE(entry.Pointer() == 0);
    REQUIRE(entry.Block() == Entry::INVALID_BLOCK_ID);
}
