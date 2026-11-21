// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"

namespace Common {

struct MemoryInfo {
    u64 TotalPhysicalMemory{};
    u64 TotalSwapMemory{};
};

/**
 * Gets the memory info of the host system
 * @return Reference to a MemoryInfo struct with the physical and swap memory sizes in bytes
 */
[[nodiscard]] const MemoryInfo& GetMemInfo();

/// Page faults of this process so far, soft and hard (0 where unknown).
[[nodiscard]] u64 GetProcessPageFaults();

/// Commit this process can still take, in bytes: the system's remaining commit, or less under
/// a job object's commit limit. No value if the query fails or the host is not Windows.
[[nodiscard]] std::optional<u64> GetProcessCommitRoom();

} // namespace Common
