// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "video_core/texture_cache/eviction_age.h"

using namespace Common::Literals;

namespace {

using VideoCommon::EvictionAge;
using VideoCommon::MinEvictionAge;
using VideoCommon::RetentionBudget;
using VideoCommon::RetentionBudgetRise;

// Compile-time verification of core invariants.
static_assert(EvictionAge(10, 600, 4_GiB, 6_GiB, true) == 10);
static_assert(EvictionAge(10, 600, 4_GiB, 0, false) == 10);
static_assert(EvictionAge(10, 600, 6_GiB, 6_GiB, false) == 10);
static_assert(EvictionAge(10, 0, 4_GiB, 6_GiB, false) == MinEvictionAge);
static_assert(EvictionAge(10, 400, 4_GiB, 6_GiB, false) == 400);
static_assert(RetentionBudget(0, 12_GiB) == 12_GiB);
static_assert(RetentionBudget(12_GiB, 10_GiB) == 10_GiB);
static_assert(RetentionBudget(10_GiB, 12_GiB) == 10_GiB + RetentionBudgetRise);

} // namespace

TEST_CASE("EvictionAge enforces MinEvictionAge and override only while within budget",
          "[video_core][texture_cache][eviction_age]") {
    constexpr u64 limit = 6_GiB;

    SECTION("Headroom raises short mode ages to MinEvictionAge and respects larger ages/overrides") {
        CHECK(EvictionAge(10, 0, 3_GiB, limit, false) == MinEvictionAge);
        CHECK(EvictionAge(25, 100, 5_GiB, limit, false) == MinEvictionAge);
        CHECK(EvictionAge(240, 0, 3_GiB, limit, false) == 240);
        CHECK(EvictionAge(25, 500, 3_GiB, limit, false) == 500);
        CHECK(EvictionAge(600, 500, 3_GiB, limit, false) == 600);
    }

    SECTION("Out-of-memory, unknown limit, or usage at/above limit bypasses floors and overrides") {
        for (const u64 override_age : {u64{0}, MinEvictionAge, u64{900}, ~u64{0}}) {
            CAPTURE(override_age);
            CHECK(EvictionAge(10, override_age, 2_GiB, limit, true) == 10);
            CHECK(EvictionAge(25, override_age, 2_GiB, 0, false) == 25);
            CHECK(EvictionAge(10, override_age, limit, limit, false) == 10);
            CHECK(EvictionAge(10, override_age, limit + 1, limit, false) == 10);
        }
    }
}

TEST_CASE("RetentionBudget drops immediately and recovers gradually by RetentionBudgetRise",
          "[video_core][texture_cache][eviction_age]") {
    // First check adopts the driver's reported budget directly.
    u64 held = RetentionBudget(0, 15_GiB);
    CHECK(held == 15_GiB);

    // An external VRAM spike drops the reported budget: held follows immediately on that check.
    held = RetentionBudget(held, 14_GiB);
    CHECK(held == 14_GiB);

    // Equal budget stays unchanged.
    CHECK(RetentionBudget(held, 14_GiB) == 14_GiB);

    // When the reported budget rebounds to 15 GiB, held climbs by RetentionBudgetRise (16 MiB)
    // per check and clamps cleanly once it reaches 15 GiB.
    const u64 steps = (15_GiB - 14_GiB) / RetentionBudgetRise;
    for (u64 step = 1; step < steps; ++step) {
        held = RetentionBudget(held, 15_GiB);
        REQUIRE(held == 14_GiB + step * RetentionBudgetRise);
    }

    // Partial final step clamps to target budget rather than overshooting.
    const u64 partial_target = held + RetentionBudgetRise / 2;
    CHECK(RetentionBudget(held, partial_target) == partial_target);
    held = RetentionBudget(held, 15_GiB);
    CHECK(held == 15_GiB);
    CHECK(RetentionBudget(held, 15_GiB) == 15_GiB);
}
