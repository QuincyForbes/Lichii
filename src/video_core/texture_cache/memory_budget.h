// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include "common/common_types.h"

namespace VideoCommon {

[[nodiscard]] constexpr u64 TextureCacheCollectionAge(bool high_priority, bool aggressive,
                                                      u64 idle_age) {
    return aggressive ? 10 : high_priority ? 25 : (std::clamp<u64>)(idle_age, 50, 3600);
}

[[nodiscard]] inline u64 NativeTextureCacheIdleAge() {
    static const u64 age = [] {
        const char* value = std::getenv("EDEN_NX2_TC_IDLE_AGE");
        return value != nullptr ? static_cast<u64>(std::strtoull(value, nullptr, 10)) : u64{50};
    }();
    return age;
}

} // namespace VideoCommon
