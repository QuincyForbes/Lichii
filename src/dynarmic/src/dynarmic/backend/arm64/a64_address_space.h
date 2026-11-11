// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <vector>

#include "dynarmic/backend/arm64/address_space.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/interface/A64/config.h"

namespace Dynarmic::Backend::Arm64 {

struct A64JitState;
struct EmittedBlockInfo;

class A64AddressSpace final : public AddressSpace {
public:
    struct AtomicCallbackCtx {
        const A64::UserConfig* conf = nullptr;
        const volatile u32* halt_reason = nullptr;
    };

    struct alignas(16) FastDispatchEntry {
        u64 location_descriptor = ~u64{0};
        const void* code_ptr = nullptr;
    };
    static_assert(sizeof(FastDispatchEntry) == 16);
    static constexpr std::size_t fast_dispatch_ways = 4;

    explicit A64AddressSpace(const A64::UserConfig& conf);

    void SetJitState(A64JitState* state, const volatile u32* halt_ptr) noexcept {
        jit_state = state;
        atomic_ctx.halt_reason = halt_ptr;
    }

    void GenerateIR(IR::Block& ir_block, IR::LocationDescriptor) const override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);
    void ClearCache() override;

protected:
    friend class A64Core;

    void EmitPrelude();
    void ClearFastDispatchTable();
    EmitConfig GetEmitConfig() override;
    void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) override;
    void OnInvalidateBasicBlock(IR::LocationDescriptor descriptor) override;

    const A64::UserConfig conf;
    BlockRangeInformation<u64> block_ranges;
    AtomicCallbackCtx atomic_ctx{};
    A64JitState* jit_state = nullptr;
    std::vector<FastDispatchEntry> fast_dispatch_table;
    u32 fast_dispatch_set_mask = 0;
};

}  // namespace Dynarmic::Backend::Arm64
