// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <tuple>
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"

#include "dynarmic/mcl/bit.hpp"
#include "common/common_types.h"
#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;

constexpr std::size_t xmrx(std::size_t x) noexcept {
    x ^= x >> 32;
    x *= 0xff51afd7ed558ccd;
    x ^= std::rotr(x, 47) ^ std::rotr(x, 23);
    return x;
}

struct DoNotFastmemMarkerHash {
    [[nodiscard]] std::size_t operator()(const DoNotFastmemMarker& value) const noexcept {
        return xmrx(std::get<0>(value).Value() ^ u64(std::get<1>(value)));
    }
};

struct FastmemPatchInfo {
    DoNotFastmemMarker marker;
    FakeCall fc;
    bool recompile;
};

class FastmemManager {
public:
    explicit FastmemManager(ExceptionHandler& eh)
            : exception_handler(eh) {}

    /// shared: markers any guest core sets apply to every core's translations (they all run the
    /// same code). block_after_faults != 0: once that many accesses of one block faulted, the
    /// whole block translates without fastmem, so it is translated again at most that many times.
    void Configure(bool shared, unsigned block_after_faults) {
        shared_markers = shared;
        faults_per_block_limit = block_after_faults;
    }

    bool SupportsFastmem() const {
        return exception_handler.SupportsFastmem();
    }

    bool ShouldFastmem(DoNotFastmemMarker marker) const;
    void MarkDoNotFastmem(DoNotFastmemMarker marker);

private:
    ExceptionHandler& exception_handler;
    ::Common::unordered_set<DoNotFastmemMarker, DoNotFastmemMarkerHash> do_not_fastmem;
    ::Common::unordered_set<IR::LocationDescriptor> do_not_fastmem_blocks;
    ::Common::unordered_map<IR::LocationDescriptor, unsigned> faults_per_block;
    bool shared_markers = false;
    unsigned faults_per_block_limit = 0;
};

}  // namespace Dynarmic::Backend::Arm64
