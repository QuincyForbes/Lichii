// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"

#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

struct StackLayout;

class AddressSpace {
public:
    /// evict_slice != 0: a full cache drops its oldest evict_slice bytes of blocks and goes on
    /// writing there (a ring), instead of being emptied. processor_id: whose cache (Perf report).
    AddressSpace(std::size_t code_cache_size, std::size_t evict_slice, std::size_t processor_id);
    virtual ~AddressSpace();

    virtual void GenerateIR(IR::Block& ir_block, IR::LocationDescriptor) const = 0;

    CodePtr Get(IR::LocationDescriptor descriptor);

    // Returns "most likely" LocationDescriptor assocated with the emitted code at that location
    std::optional<IR::LocationDescriptor> ReverseGetLocation(CodePtr host_pc);

    // Returns "most likely" entry_point associated with the emitted code at that location
    CodePtr ReverseGetEntryPoint(CodePtr host_pc);

    CodePtr GetOrEmit(IR::LocationDescriptor descriptor);
    CodePtr GetOrEmitFromDispatcher(IR::LocationDescriptor descriptor, StackLayout& stack);

    void InvalidateBasicBlocks(const ::Common::unordered_set<IR::LocationDescriptor>& descriptors);

    virtual void ClearCache();
protected:
    struct BlockReferences {
        std::array<CodePtr, 2> first{};
        std::unique_ptr<std::vector<CodePtr>> rest;

        void Add(CodePtr ptr);
        void Remove(CodePtr ptr);
        bool Empty() const;
    };

    virtual EmitConfig GetEmitConfig() = 0;
    virtual void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) = 0;
    virtual void OnInvalidateBasicBlock(IR::LocationDescriptor) {}

    void ProtectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.protect();
#endif
    }

    void UnprotectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.unprotect();
#endif
    }

    std::size_t GetRemainingSize();
    void MakeRoomForBlock();
    void WrapRing();
    void EvictOldest(std::size_t end_offset);
    void EvictBlock(CodePtr entry_point, IR::LocationDescriptor descriptor);
    std::size_t LiveBytes() const;
    CodePtr Emit(IR::Block ir_block);
    void Link(EmittedBlockInfo& block);
    void LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list);
    void RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr);

    FakeCall FastmemCallback(u64 host_pc);

    IR::Block ir_block;
    const std::size_t code_cache_size;
    const std::size_t evict_slice;
    const std::size_t processor_id;
    u64 cache_generation = 0;
    std::size_t dropped_writable_offset = 0;
    /// Ring mode: [code.offset(), free_end) may be written; blocks of the previous lap start at
    /// free_end and above.
    std::size_t free_end;
    oaknut::CodeBlock mem;
    oaknut::CodeGenerator code;

    // CodePtr entries are appended monotonically between ClearCache() calls (ring mode: within
    // a lap; the previous lap's survivors are old_reverse_entries[old_reverse_head...]).
    std::vector<std::pair<CodePtr, IR::LocationDescriptor>> reverse_block_entries;
    std::vector<std::pair<CodePtr, IR::LocationDescriptor>> old_reverse_entries;
    std::size_t old_reverse_head = 0;
    ::Common::unordered_map<IR::LocationDescriptor, CodePtr> block_entries;
    ::Common::unordered_map<CodePtr, EmittedBlockInfo> block_infos;
    ::Common::unordered_map<IR::LocationDescriptor, BlockReferences> block_references;

    ExceptionHandler exception_handler;
    FastmemManager fastmem_manager;

    struct PreludeInfo {
        std::ptrdiff_t end_of_prelude;

        using RunCodeFuncType = HaltReason (*)(CodePtr entry_point, void* jit_state, volatile u32* halt_reason);
        RunCodeFuncType run_code;
        RunCodeFuncType step_code;
        void* return_to_dispatcher;
        void* return_from_run_code;
        void* pop_rsb_hint = nullptr;
        void* fast_dispatch_hint = nullptr;

        void* read_memory_8;
        void* read_memory_16;
        void* read_memory_32;
        void* read_memory_64;
        void* read_memory_128;
        void* wrapped_read_memory_8;
        void* wrapped_read_memory_16;
        void* wrapped_read_memory_32;
        void* wrapped_read_memory_64;
        void* wrapped_read_memory_128;
        void* exclusive_read_memory_8;
        void* exclusive_read_memory_16;
        void* exclusive_read_memory_32;
        void* exclusive_read_memory_64;
        void* exclusive_read_memory_128;
        void* write_memory_8;
        void* write_memory_16;
        void* write_memory_32;
        void* write_memory_64;
        void* write_memory_128;
        void* wrapped_write_memory_8;
        void* wrapped_write_memory_16;
        void* wrapped_write_memory_32;
        void* wrapped_write_memory_64;
        void* wrapped_write_memory_128;
        void* exclusive_write_memory_8;
        void* exclusive_write_memory_16;
        void* exclusive_write_memory_32;
        void* exclusive_write_memory_64;
        void* exclusive_write_memory_128;
        void* wrapped_atomic_rmw_8 = nullptr;
        void* wrapped_atomic_rmw_16 = nullptr;
        void* wrapped_atomic_rmw_32 = nullptr;
        void* wrapped_atomic_rmw_64 = nullptr;
        void* wrapped_atomic_cas_8 = nullptr;
        void* wrapped_atomic_cas_16 = nullptr;
        void* wrapped_atomic_cas_32 = nullptr;
        void* wrapped_atomic_cas_64 = nullptr;

        void* call_svc;
        void* exception_raised;
        void* dc_raised;
        void* ic_raised;
        void* isb_raised;

        void* get_cntpct;
        void* add_ticks;
        void* get_ticks_remaining;
    } prelude_info;
};

}  // namespace Dynarmic::Backend::Arm64
