// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include "common/jit_block_lists.h"

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
    // What a block keeps after it is linked, behind one pointer (a 16-byte table slot): its
    // size, link lists and fastmem patch points in one packed heap block
    // (Common::JitBlockLists), or the lists as the emitter made them (EDEN_NX2_JIT_PACKED_LISTS
    // off, and blocks the packed layout cannot hold). EmittedBlockInfo in block_infos was 96
    // bytes a slot (45.5 MiB tables per guest core in the iPhone malloc census), then 32 bytes
    // plus four to six small heap blocks a block (15.5 MiB tables per core and 237 MiB of
    // 16-64 byte blocks).
    class StoredBlockInfo {
    public:
        struct Lists {
            std::size_t size;
            decltype(EmittedBlockInfo::block_relocations) block_relocations;
            decltype(EmittedBlockInfo::fastmem_patch_info) fastmem_patch_info;
        };

        StoredBlockInfo() = default;
        /// Takes block_info's lists when it keeps them unpacked.
        StoredBlockInfo(EmittedBlockInfo& block_info, bool packed);
        StoredBlockInfo(StoredBlockInfo&& other) noexcept
                : value{std::exchange(other.value, 0)} {}
        StoredBlockInfo& operator=(StoredBlockInfo&& other) noexcept {
            if (this != &other) {
                Reset();
                value = std::exchange(other.value, 0);
            }
            return *this;
        }
        ~StoredBlockInfo() {
            Reset();
        }

        std::size_t Size() const;
        /// Heap bytes behind the pointer (an estimate for unpacked lists).
        std::size_t Bytes() const;
        /// func(IR::LocationDescriptor) for each block this one branches to.
        template<typename Func>
        void ForEachTarget(Func&& func) const;
        /// func(BlockRelocation) for each place in this block that branches to target.
        template<typename Func>
        void ForEachRelocationTo(IR::LocationDescriptor target, Func&& func) const;
        std::optional<FastmemPatchInfo> FindPatch(std::ptrdiff_t code_offset) const;

    private:
        void Reset();
        const Lists* Unpacked() const {
            return (value & 1) != 0 ? reinterpret_cast<const Lists*>(value - 1) : nullptr;
        }
        ::Common::JitBlockLists Packed() const {
            return ::Common::JitBlockLists{reinterpret_cast<const u64*>(value)};
        }

        /// A JitBlockLists heap block, or with the low bit set a Lists.
        std::uintptr_t value = 0;
    };

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
    void LinkBlockLink(const CodePtr entry_point, const CodePtr target_ptr, BlockRelocation relocation);
    void LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list);
    std::size_t TableBytes() const;
    /// For RegisterNewBasicBlock, after it added a block to its range list (BlockRangeInformation).
    template<typename Ranges>
    void RangeAdded(Ranges& ranges) {
        if (compact_ranges) {
            ranges.CompactIfGrown([this](IR::LocationDescriptor location) { return block_entries.contains(location); });
        }
        range_table_bytes = ranges.Bytes();
    }
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
    ::Common::unordered_map<CodePtr, StoredBlockInfo> block_infos;
    ::Common::unordered_map<IR::LocationDescriptor, BlockReferences> block_references;
    const bool packed_lists;
    /// EDEN_NX2_JIT_RANGE_COMPACT (on unless 0): the range list forgets blocks that left the cache.
    const bool compact_ranges;
    /// Heap bytes behind block_infos' pointers.
    std::size_t list_bytes = 0;
    /// What the derived class keeps for each block (its guest ranges), for the Perf report.
    std::size_t range_table_bytes = 0;

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

        void* page_table_lookup_8 = nullptr;
        void* page_table_lookup_16 = nullptr;
        void* page_table_lookup_32 = nullptr;
        void* page_table_lookup_64 = nullptr;
        void* page_table_lookup_128 = nullptr;
    } prelude_info;
};

template<typename Func>
void AddressSpace::StoredBlockInfo::ForEachTarget(Func&& func) const {
    if (const Lists* const lists = Unpacked()) {
        for (const auto& [target, list] : lists->block_relocations) {
            func(target);
        }
        return;
    }
    const ::Common::JitBlockLists packed = Packed();
    for (std::size_t index = 0; index < packed.Targets(); ++index) {
        func(IR::LocationDescriptor{packed.TargetLocation(index)});
    }
}

template<typename Func>
void AddressSpace::StoredBlockInfo::ForEachRelocationTo(IR::LocationDescriptor target, Func&& func) const {
    if (const Lists* const lists = Unpacked()) {
        if (const auto iter = lists->block_relocations.find(target); iter != lists->block_relocations.end()) {
            for (const BlockRelocation& relocation : iter->second) {
                func(relocation);
            }
        }
        return;
    }
    const ::Common::JitBlockLists packed = Packed();
    if (const std::size_t index = packed.FindTarget(target.Value()); index != packed.Targets()) {
        packed.ForEachRelocation(index, [&func](::Common::JitBlockLists::Relocation relocation) {
            func(BlockRelocation{static_cast<std::ptrdiff_t>(relocation.code_offset),
                                 static_cast<BlockRelocationType>(relocation.type)});
        });
    }
}

}  // namespace Dynarmic::Backend::Arm64
