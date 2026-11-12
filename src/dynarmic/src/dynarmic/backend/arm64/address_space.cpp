// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include "common/ios_jit.h"
#endif
#endif

#include "dynarmic/backend/arm64/a64_address_space.h"
#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

AddressSpace::AddressSpace(std::size_t code_cache_size)
    : ir_block{IR::LocationDescriptor{0}}
    , code_cache_size(code_cache_size)
    , mem(code_cache_size)
    , code(mem.wptr(), mem.ptr())
    , fastmem_manager(exception_handler)
{
    ASSERT(code_cache_size <= 128 * 1024 * 1024 && "code_cache_size > 128 MiB not currently supported");

    exception_handler.Register(mem, code_cache_size);
    exception_handler.SetFastmemCallback([this](u64 host_pc) {
        return FastmemCallback(host_pc);
    });
}

AddressSpace::~AddressSpace() = default;

void AddressSpace::BlockReferences::Add(CodePtr ptr) {
    for (CodePtr& slot : first) {
        if (slot == ptr) {
            return;
        }
        if (slot == nullptr) {
            slot = ptr;
            return;
        }
    }
    if (!rest) {
        rest = std::make_unique<std::vector<CodePtr>>();
    } else if (std::find(rest->begin(), rest->end(), ptr) != rest->end()) {
        return;
    }
    rest->push_back(ptr);
}

CodePtr AddressSpace::Get(IR::LocationDescriptor descriptor) {
    if (const auto iter = block_entries.find(descriptor); iter != block_entries.end()) {
        return iter->second;
    }
    return nullptr;
}

std::optional<IR::LocationDescriptor> AddressSpace::ReverseGetLocation(CodePtr host_pc) {
    const auto cmp = [](CodePtr pc, const auto& entry) { return pc < entry.first; };
    if (auto iter = std::upper_bound(reverse_block_entries.begin(), reverse_block_entries.end(), host_pc, cmp);
        iter != reverse_block_entries.begin()) {
        --iter;
        return iter->second;
    }
    return std::nullopt;
}

CodePtr AddressSpace::ReverseGetEntryPoint(CodePtr host_pc) {
    const auto cmp = [](CodePtr pc, const auto& entry) { return pc < entry.first; };
    if (auto iter = std::upper_bound(reverse_block_entries.begin(), reverse_block_entries.end(), host_pc, cmp);
        iter != reverse_block_entries.begin()) {
        --iter;
        return iter->first;
    }
    return nullptr;
}

CodePtr AddressSpace::GetOrEmit(IR::LocationDescriptor descriptor) {
    if (CodePtr block_entry = Get(descriptor)) {
        return block_entry;
    }
    if (jit_stats == nullptr) {
        GenerateIR(ir_block, descriptor);
        return Emit(std::move(ir_block));
    }
    const auto start = std::chrono::steady_clock::now();
    GenerateIR(ir_block, descriptor);
    const CodePtr entry_point = Emit(std::move(ir_block));
    A64::AddJitStat(jit_stats, A64::JitStat::BlocksCompiled);
    A64::AddJitStat(jit_stats, A64::JitStat::CompileNs,
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - start)
                                         .count()));
    return entry_point;
}

CodePtr AddressSpace::GetOrEmitFromDispatcher(IR::LocationDescriptor descriptor, StackLayout& stack) {
    const u64 generation = cache_generation;
    const CodePtr entry_point = GetOrEmit(descriptor);
    if (generation != cache_generation) {
        for (RSBEntry& entry : stack.rsb) {
            entry.code_ptr = std::bit_cast<u64>(prelude_info.return_to_dispatcher);
        }
    }
    return entry_point;
}

void AddressSpace::InvalidateBasicBlocks(const ::Common::unordered_set<IR::LocationDescriptor>& descriptors) {
    UnprotectCodeMemory();

    bool any_erased = false;
    for (const auto& descriptor : descriptors) {
        const auto iter = block_entries.find(descriptor);
        if (iter == block_entries.end()) {
            continue;
        }

        // Unlink before removal because InvalidateBasicBlocks can be called within a fastmem callback,
        // and the currently executing block may have references to itself which need to be unlinked.
        RelinkForDescriptor(descriptor, nullptr);
        OnInvalidateBasicBlock(descriptor);

        block_entries.erase(iter);
        any_erased = true;
    }
    if (any_erased) {
        ++cache_generation;
    }

    ProtectCodeMemory();
}

void AddressSpace::ClearCache() {
    block_entries.clear();
    reverse_block_entries.clear();
    block_infos.clear();
    block_references.clear();
    code.set_offset(prelude_info.end_of_prelude);
#if defined(__APPLE__) && TARGET_OS_IPHONE
    ::Common::IosJit::DropWritablePtes(mem.wptr(), code_cache_size);
    dropped_writable_offset = 0;
#endif
    ++cache_generation;
}

std::size_t AddressSpace::GetRemainingSize() {
    return code_cache_size - static_cast<std::size_t>(code.offset());
}

CodePtr AddressSpace::Emit(IR::Block block) {
    if (GetRemainingSize() < 1024 * 1024) {
#ifdef __APPLE__
        // The iOS caches are small (debugger-prepared pool): every full flush translates the hot
        // code again, so count them (iOS: Documents/console.txt).
        static std::atomic<unsigned> full_flushes{};
        std::fprintf(stderr, "dynarmic: code cache full (%zu MiB), flushed; %u full flushes so far\n",
                     code_cache_size >> 20, full_flushes.fetch_add(1, std::memory_order_relaxed) + 1);
#endif
        ClearCache();
        A64::AddJitStat(jit_stats, A64::JitStat::CacheFlushes);
    }

    UnprotectCodeMemory();

    EmittedBlockInfo block_info = EmitArm64(code, std::move(block), GetEmitConfig(), fastmem_manager);
    const CodePtr entry_point = block_info.entry_point;

    ASSERT(block_entries.insert({block.Location(), entry_point}).second);
    ASSERT(reverse_block_entries.empty() || reverse_block_entries.back().first < entry_point);
    reverse_block_entries.emplace_back(entry_point, block.Location());
    const auto [info_iter, inserted] = block_infos.emplace(entry_point, std::move(block_info));
    ASSERT(inserted);
    EmittedBlockInfo& stored_info = info_iter->second;

    Link(stored_info);
    std::vector<Relocation>{}.swap(stored_info.relocations);
    RelinkForDescriptor(block.Location(), entry_point);

    mem.invalidate(reinterpret_cast<u32*>(entry_point), stored_info.size);
    ProtectCodeMemory();

#if defined(__APPLE__) && TARGET_OS_IPHONE
    constexpr std::size_t kDropGranularity = 2 * 1024 * 1024;
    constexpr std::size_t kIosPageSize = 16 * 1024;
    const auto current_offset = static_cast<std::size_t>(code.offset());
    if (current_offset - dropped_writable_offset >= kDropGranularity) {
        const std::size_t drop_end = current_offset & ~(kIosPageSize - 1);
        if (drop_end > dropped_writable_offset) {
            ::Common::IosJit::DropWritablePtes(
                reinterpret_cast<std::uint8_t*>(mem.wptr()) + dropped_writable_offset,
                drop_end - dropped_writable_offset);
            dropped_writable_offset = drop_end;
        }
    }
#endif

    RegisterNewBasicBlock(block, stored_info);
    if (code_cache_used != nullptr) {
        code_cache_used->store(static_cast<std::uint64_t>(code.offset()), std::memory_order_relaxed);
    }

    return entry_point;
}

void AddressSpace::Link(EmittedBlockInfo& block_info) {
    using namespace oaknut;
    using namespace oaknut::util;

    for (auto [ptr_offset, target] : block_info.relocations) {
        CodeGenerator c{mem.wptr(), mem.ptr()};
        c.set_xptr(reinterpret_cast<u32*>(block_info.entry_point + ptr_offset));

        switch (target) {
        case LinkTarget::ReturnToDispatcher:
            c.B(prelude_info.return_to_dispatcher);
            break;
        case LinkTarget::ReturnFromRunCode:
            c.B(prelude_info.return_from_run_code);
            break;
        case LinkTarget::PopRSBHint:
            c.B(prelude_info.pop_rsb_hint);
            break;
        case LinkTarget::FastDispatchHint:
            c.B(prelude_info.fast_dispatch_hint);
            break;
        // { this, vaddr, size }
        case LinkTarget::ReadMemory8:
            c.BL(prelude_info.read_memory_8);
            break;
        case LinkTarget::ReadMemory16:
            c.BL(prelude_info.read_memory_16);
            break;
        case LinkTarget::ReadMemory32:
            c.BL(prelude_info.read_memory_32);
            break;
        case LinkTarget::ReadMemory64:
            c.BL(prelude_info.read_memory_64);
            break;
        case LinkTarget::ReadMemory128:
            c.BL(prelude_info.read_memory_128);
            break;
        // { this, vaddr, size }
        case LinkTarget::WrappedReadMemory8:
            c.BL(prelude_info.wrapped_read_memory_8);
            break;
        case LinkTarget::WrappedReadMemory16:
            c.BL(prelude_info.wrapped_read_memory_16);
            break;
        case LinkTarget::WrappedReadMemory32:
            c.BL(prelude_info.wrapped_read_memory_32);
            break;
        case LinkTarget::WrappedReadMemory64:
            c.BL(prelude_info.wrapped_read_memory_64);
            break;
        case LinkTarget::WrappedReadMemory128:
            c.BL(prelude_info.wrapped_read_memory_128);
            break;
        //
        case LinkTarget::ExclusiveReadMemory8:
            c.BL(prelude_info.exclusive_read_memory_8);
            break;
        case LinkTarget::ExclusiveReadMemory16:
            c.BL(prelude_info.exclusive_read_memory_16);
            break;
        case LinkTarget::ExclusiveReadMemory32:
            c.BL(prelude_info.exclusive_read_memory_32);
            break;
        case LinkTarget::ExclusiveReadMemory64:
            c.BL(prelude_info.exclusive_read_memory_64);
            break;
        case LinkTarget::ExclusiveReadMemory128:
            c.BL(prelude_info.exclusive_read_memory_128);
            break;
        // { this, vaddr, value, size }
        case LinkTarget::WriteMemory8:
            c.BL(prelude_info.write_memory_8);
            break;
        case LinkTarget::WriteMemory16:
            c.BL(prelude_info.write_memory_16);
            break;
        case LinkTarget::WriteMemory32:
            c.BL(prelude_info.write_memory_32);
            break;
        case LinkTarget::WriteMemory64:
            c.BL(prelude_info.write_memory_64);
            break;
        case LinkTarget::WriteMemory128:
            c.BL(prelude_info.write_memory_128);
            break;
        //
        case LinkTarget::WrappedWriteMemory8:
            c.BL(prelude_info.wrapped_write_memory_8);
            break;
        case LinkTarget::WrappedWriteMemory16:
            c.BL(prelude_info.wrapped_write_memory_16);
            break;
        case LinkTarget::WrappedWriteMemory32:
            c.BL(prelude_info.wrapped_write_memory_32);
            break;
        case LinkTarget::WrappedWriteMemory64:
            c.BL(prelude_info.wrapped_write_memory_64);
            break;
        case LinkTarget::WrappedWriteMemory128:
            c.BL(prelude_info.wrapped_write_memory_128);
            break;
        //
        case LinkTarget::ExclusiveWriteMemory8:
            c.BL(prelude_info.exclusive_write_memory_8);
            break;
        case LinkTarget::ExclusiveWriteMemory16:
            c.BL(prelude_info.exclusive_write_memory_16);
            break;
        case LinkTarget::ExclusiveWriteMemory32:
            c.BL(prelude_info.exclusive_write_memory_32);
            break;
        case LinkTarget::ExclusiveWriteMemory64:
            c.BL(prelude_info.exclusive_write_memory_64);
            break;
        case LinkTarget::ExclusiveWriteMemory128:
            c.BL(prelude_info.exclusive_write_memory_128);
            break;
        case LinkTarget::WrappedAtomicRmw8:
            c.BL(prelude_info.wrapped_atomic_rmw_8);
            break;
        case LinkTarget::WrappedAtomicRmw16:
            c.BL(prelude_info.wrapped_atomic_rmw_16);
            break;
        case LinkTarget::WrappedAtomicRmw32:
            c.BL(prelude_info.wrapped_atomic_rmw_32);
            break;
        case LinkTarget::WrappedAtomicRmw64:
            c.BL(prelude_info.wrapped_atomic_rmw_64);
            break;
        case LinkTarget::WrappedAtomicCas8:
            c.BL(prelude_info.wrapped_atomic_cas_8);
            break;
        case LinkTarget::WrappedAtomicCas16:
            c.BL(prelude_info.wrapped_atomic_cas_16);
            break;
        case LinkTarget::WrappedAtomicCas32:
            c.BL(prelude_info.wrapped_atomic_cas_32);
            break;
        case LinkTarget::WrappedAtomicCas64:
            c.BL(prelude_info.wrapped_atomic_cas_64);
            break;
        case LinkTarget::CallSVC:
            c.BL(prelude_info.call_svc);
            break;
        case LinkTarget::ExceptionRaised:
            c.BL(prelude_info.exception_raised);
            break;
        case LinkTarget::InstructionSynchronizationBarrierRaised:
            c.BL(prelude_info.isb_raised);
            break;
        case LinkTarget::InstructionCacheOperationRaised:
            c.BL(prelude_info.ic_raised);
            break;
        case LinkTarget::DataCacheOperationRaised:
            c.BL(prelude_info.dc_raised);
            break;
        case LinkTarget::GetCNTPCT:
            c.BL(prelude_info.get_cntpct);
            break;
        case LinkTarget::AddTicks:
            c.BL(prelude_info.add_ticks);
            break;
        case LinkTarget::GetTicksRemaining:
            c.BL(prelude_info.get_ticks_remaining);
            break;
        default:
            UNREACHABLE();
        }
    }

    for (const auto& [target_descriptor, list] : block_info.block_relocations) {
        block_references[target_descriptor].Add(block_info.entry_point);
        LinkBlockLinks(block_info.entry_point, Get(target_descriptor), list);
    }
}

void AddressSpace::LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list) {
    using namespace oaknut;
    using namespace oaknut::util;

    for (auto [ptr_offset, type] : block_relocations_list) {
        CodeGenerator c{mem.wptr(), mem.ptr()};
        c.set_xptr(reinterpret_cast<u32*>(entry_point + ptr_offset));

        switch (type) {
        case BlockRelocationType::Branch:
            if (target_ptr) {
                c.B((void*)target_ptr);
            } else {
                c.NOP();
            }
            break;
        case BlockRelocationType::MoveToScratch1:
            if (target_ptr) {
                c.ADRL(Xscratch1, (void*)target_ptr);
            } else {
                c.ADRL(Xscratch1, prelude_info.return_to_dispatcher);
            }
            break;
        default:
            UNREACHABLE();
        }
    }
}

void AddressSpace::RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr) {
    const auto refs_iter = block_references.find(target_descriptor);
    if (refs_iter == block_references.end()) {
        return;
    }

    const auto relink_one = [&](CodePtr code_ptr) {
        if (auto block_iter = block_infos.find(code_ptr); block_iter != block_infos.end()) {
            const EmittedBlockInfo& block_info = block_iter->second;

            if (auto relocation_iter = block_info.block_relocations.find(target_descriptor);
                relocation_iter != block_info.block_relocations.end()) {
                LinkBlockLinks(block_info.entry_point, target_ptr, relocation_iter->second);
            }

            mem.invalidate(reinterpret_cast<u32*>(block_info.entry_point), block_info.size);
        }
    };

    for (CodePtr code_ptr : refs_iter->second.first) {
        if (code_ptr != nullptr) {
            relink_one(code_ptr);
        }
    }
    if (refs_iter->second.rest) {
        for (CodePtr code_ptr : *refs_iter->second.rest) {
            relink_one(code_ptr);
        }
    }
}

FakeCall AddressSpace::FastmemCallback(u64 host_pc) {
    A64::AddJitStat(jit_stats, A64::JitStat::FastmemFaults);
    {
        const auto host_ptr = std::bit_cast<CodePtr>(host_pc);

        const auto entry_point = ReverseGetEntryPoint(host_ptr);
        if (!entry_point) {
            goto fail;
        }

        const auto block_info = block_infos.find(entry_point);
        if (block_info == block_infos.end()) {
            goto fail;
        }

        const auto patch_entry = block_info->second.fastmem_patch_info.find(host_ptr - entry_point);
        if (patch_entry == block_info->second.fastmem_patch_info.end()) {
            goto fail;
        }

        const auto fc = patch_entry->second.fc;

        if (patch_entry->second.recompile) {
            const auto marker = patch_entry->second.marker;
            fastmem_manager.MarkDoNotFastmem(marker);
            InvalidateBasicBlocks({std::get<0>(marker)});
        }

        return fc;
    }

fail:
    fmt::print("dynarmic: Segfault happened within JITted code at host_pc = {:016x}\n"
        "Segfault wasn't at a fastmem patch location!\n", host_pc);
    UNREACHABLE();
}

}  // namespace Dynarmic::Backend::Arm64
