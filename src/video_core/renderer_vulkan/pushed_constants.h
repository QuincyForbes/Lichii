// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "common/common_types.h"

namespace Vulkan {

/// The bytes the command buffer holds in one part of the push constants, as far as the recording
/// thread knows them, so a draw whose values are already there does not push them again.
template <size_t Size>
class PushedConstants {
public:
    /// Returns true when the value has to be pushed. An uncertain push (lands = false: its
    /// command may return before it, as for a pipeline that is still building) is not remembered.
    template <typename T>
    [[nodiscard]] bool Update(const T& value, bool lands) noexcept {
        static_assert(sizeof(T) == Size && std::is_trivially_copyable_v<T>);
        if (known && std::memcmp(bytes.data(), &value, Size) == 0) {
            return false;
        }
        std::memcpy(bytes.data(), &value, Size);
        known = lands;
        return true;
    }

    void Forget() noexcept {
        known = false;
    }

private:
    std::array<u8, Size> bytes{};
    bool known{};
};

} // namespace Vulkan
