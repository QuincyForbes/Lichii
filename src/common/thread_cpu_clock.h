// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <optional>

#include "common/common_types.h"

namespace Common {

std::optional<u64> CurrentThreadCpuTimeNs();

class ThreadCpuClock {
public:
    explicit ThreadCpuClock(bool enabled = true);
    ~ThreadCpuClock();
    ThreadCpuClock(const ThreadCpuClock&) = delete;
    ThreadCpuClock& operator=(const ThreadCpuClock&) = delete;

    std::optional<u64> Read() const;
    /// Of it, the time in the kernel (not on Linux).
    std::optional<u64> ReadSystem() const;

private:
    std::intptr_t native_clock{};
    bool available{};
};

} // namespace Common
