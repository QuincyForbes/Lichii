// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include "common/literals.h"
#include "video_core/texture_cache/memory_budget.h"

using namespace Common::Literals;

TEST_CASE("Idle texture retention never delays pressure-driven collection",
          "[video_core][texture_cache_budget]") {
    using VideoCommon::TextureCacheCollectionAge;
    for (const u64 idle_age : {u64{0}, u64{50}, u64{600}, u64{3600}, ~u64{0}}) {
        CAPTURE(idle_age);
        CHECK(TextureCacheCollectionAge(true, false, idle_age) == 25);
        CHECK(TextureCacheCollectionAge(true, true, idle_age) == 10);
    }
    CHECK(TextureCacheCollectionAge(false, false, 50) == 50);
    CHECK(TextureCacheCollectionAge(false, false, 600) == 600);
    CHECK(TextureCacheCollectionAge(false, false, 0) == 50);
    CHECK(TextureCacheCollectionAge(false, false, ~u64{0}) == 3600);
}
