// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/memory_manager.h"

namespace Shader {
using HandleProgram = std::vector<u32>;
} // namespace Shader

namespace VideoCommon {

/// Per-draw/dispatch 4-slot 4 KiB guest page cache for constant and storage buffer handle reads.
/// Valid while GPU page mappings do not change (within a single draw or dispatch setup).
struct DescriptorPageCache {
    static constexpr size_t NUM_SLOTS = 4;
    static constexpr GPUVAddr PAGE_BITS = 12;
    static constexpr GPUVAddr PAGE_BYTES = GPUVAddr{1} << PAGE_BITS;
    static constexpr GPUVAddr PAGE_MASK = PAGE_BYTES - 1;
    static constexpr GPUVAddr INVALID_PAGE = ~GPUVAddr{0};

    explicit DescriptorPageCache(const Tegra::MemoryManager& gpu_memory_) noexcept
        : gpu_memory{gpu_memory_} {}

    void Reset() noexcept {
        pages.fill(INVALID_PAGE);
    }

    [[nodiscard]] u32 ReadHandleWord(GPUVAddr addr) noexcept {
        const GPUVAddr page_offset = addr & PAGE_MASK;
        if (page_offset <= PAGE_BYTES - sizeof(u32)) {
            if (const u8* const host = LookupPage(addr)) {
                u32 value;
                std::memcpy(&value, host + page_offset, sizeof(u32));
                return value;
            }
        }
        return gpu_memory.ReadHandleWord(addr);
    }

    [[nodiscard]] u64 ReadHandleU64(GPUVAddr addr) noexcept {
        const GPUVAddr page_offset = addr & PAGE_MASK;
        if (page_offset <= PAGE_BYTES - sizeof(u64)) {
            if (const u8* const host = LookupPage(addr)) {
                u64 value;
                std::memcpy(&value, host + page_offset, sizeof(u64));
                return value;
            }
        }
        return u64{ReadHandleWord(addr)} | (u64{ReadHandleWord(addr + sizeof(u32))} << 32);
    }

    [[nodiscard]] u32 operator()(GPUVAddr addr) noexcept {
        return ReadHandleWord(addr);
    }

private:
    [[nodiscard]] const u8* LookupPage(GPUVAddr addr) noexcept {
        const GPUVAddr page = addr >> PAGE_BITS;
        const size_t slot = static_cast<size_t>(page ^ (page >> 4)) & (NUM_SLOTS - 1);
        const u8* host = hosts[slot];
        if (pages[slot] != page) {
            const GPUVAddr page_addr = addr & ~PAGE_MASK;
            host = gpu_memory.OverlapsNativeSyncpointShim(page_addr, PAGE_BYTES)
                       ? nullptr
                       : gpu_memory.GetPointerForDescriptor(page_addr);
            pages[slot] = host != nullptr ? page : INVALID_PAGE;
            hosts[slot] = host;
        }
        return host;
    }

    const Tegra::MemoryManager& gpu_memory;
    std::array<GPUVAddr, NUM_SLOTS> pages{INVALID_PAGE, INVALID_PAGE, INVALID_PAGE, INVALID_PAGE};
    std::array<const u8*, NUM_SLOTS> hosts{};
};

/// The texture handle a shader reads from a storage buffer, computed from the current constant
/// buffer and storage buffer words (Shader::Info::handle_programs). cbuf_address(index) is the
/// bound constant buffer's address, 0 when none is bound. storage_alignment is the host storage
/// buffer offset alignment: storage offsets in the program are relative to the buffer address
/// aligned down to it (StorageOffset), which is also where the buffer cache binds the buffer.
/// element is the descriptor array element, for a program computing one handle per element (a
/// handle table the shader indexes per invocation, HandleOp::Index).
template <typename Memory, typename CbufAddress>
u32 EvaluateTextureHandle(const Shader::Info& info, u32 program, Memory& gpu_memory,
                          CbufAddress&& cbuf_address, u64 storage_alignment, u32 element = 0) {
    const GPUVAddr storage_mask{~(GPUVAddr{(std::max)(storage_alignment, u64{1})} - 1)};
    return Shader::EvaluateHandleProgram(
               info.handle_programs[program],
               [&](u32 index, u32 offset) -> u32 {
                   const GPUVAddr base{cbuf_address(index)};
                   return base != 0 ? gpu_memory.ReadHandleWord(base + offset) : 0U;
               },
               [&](u32 binding, u32 offset) -> u32 {
                   if (binding >= info.storage_buffers_descriptors.size()) {
                       return 0U;
                   }
                   const auto& descriptor{info.storage_buffers_descriptors[binding]};
                   const GPUVAddr cbuf{cbuf_address(descriptor.cbuf_index)};
                   if (cbuf == 0) {
                       return 0U;
                   }
                   const auto read_u64 = [&](GPUVAddr addr) -> u64 {
                       if constexpr (requires(GPUVAddr a) { gpu_memory.ReadHandleU64(a); }) {
                           return gpu_memory.ReadHandleU64(addr);
                       } else {
                           return u64{gpu_memory.ReadHandleWord(addr)} |
                                  (u64{gpu_memory.ReadHandleWord(addr + sizeof(u32))} << 32);
                       }
                   };
                   GPUVAddr base{read_u64(cbuf + descriptor.cbuf_offset)};
                   if (base != 0 && descriptor.indirect_offset != ~0U) {
                       // The table entry is read at its exact address. Only the buffer address
                       // it holds is aligned down, as the shader does.
                       base = read_u64(base + descriptor.indirect_offset);
                   }
                   base &= storage_mask;
                   return base != 0 ? gpu_memory.ReadHandleWord(base + offset) : 0U;
               },
               element)
        .value_or(0U);
}

/// Evaluates elements 0..count-1 of a handle program into `out`, caching constant buffer words and
/// up to 64 SSBO base pointers across all elements of the array.
template <typename MemoryReader>
void EvaluateTextureHandles(std::span<const u32> program, u32 count,
                            std::span<const GPUVAddr> cbuf_addrs,
                            std::span<const GPUVAddr> ssbo_addrs, MemoryReader& memory,
                            std::span<u32> out) {
    const auto read_word = [&](GPUVAddr addr) -> u32 {
        if constexpr (requires(GPUVAddr a) { memory.ReadHandleWord(a); }) {
            return memory.ReadHandleWord(addr);
        } else {
            return memory.template Read<u32>(addr);
        }
    };
    struct CbufWord {
        u32 index;
        u32 offset;
        u32 value;
    };
    std::array<CbufWord, 16> words{};
    size_t num_words = 0;
    std::array<GPUVAddr, Shader::Info::MAX_SSBOS> bases{};
    static_assert(Shader::Info::MAX_SSBOS <= 64);
    u64 known_bases = 0;
    const auto read_cbuf = [&](u32 index, u32 offset) -> u32 {
        for (size_t i = 0; i < num_words; ++i) {
            if (words[i].index == index && words[i].offset == offset) {
                return words[i].value;
            }
        }
        const GPUVAddr base = index < cbuf_addrs.size() ? cbuf_addrs[index] : GPUVAddr{};
        const u32 value = base != 0 ? read_word(base + offset) : 0U;
        if (num_words < words.size()) {
            words[num_words++] = {index, offset, value};
        }
        return value;
    };
    const auto read_storage = [&](u32 binding, u32 offset) -> u32 {
        if (binding >= bases.size()) {
            return 0U;
        }
        const u64 bit = u64{1} << binding;
        if ((known_bases & bit) == 0) {
            bases[binding] = binding < ssbo_addrs.size() ? ssbo_addrs[binding] : GPUVAddr{};
            known_bases |= bit;
        }
        const GPUVAddr base = bases[binding];
        return base != 0 ? read_word(base + offset) : 0U;
    };
    const u32 n = static_cast<u32>((std::min<size_t>)(count, out.size()));
    for (u32 element = 0; element < n; ++element) {
        out[element] =
            Shader::EvaluateHandleProgram(program, read_cbuf, read_storage, element).value_or(0U);
    }
}

template <typename MemoryReader>
void EvaluateTextureHandles(const Shader::HandleProgram& program, u32 count,
                            std::span<const GPUVAddr> cbuf_addrs,
                            std::span<const GPUVAddr> ssbo_addrs, MemoryReader& memory,
                            std::span<u32> out) {
    EvaluateTextureHandles(std::span<const u32>{program}, count, cbuf_addrs, ssbo_addrs, memory,
                           out);
}

/// Whether texture descriptor `desc`'s handle word (element `index`) is where a storage buffer of
/// the same program keeps its address. The translation falls back to such a word when the shader
/// loads the handle per invocation, which no draw-time evaluation can give (a light's texture
/// read in a clustered light loop, Ocarina of Time's waterfall). Its low bits then name an
/// unrelated texture, a different one whenever the game moves the buffer: every frame for a
/// double-buffered one, so the surface flickers. Bind no texture for it instead.
template <typename Descriptor>
[[nodiscard]] bool HandleIsStoragePointer(const Shader::Info& info, const Descriptor& desc,
                                          u32 index) {
    if (desc.handle_program != ~0U || desc.has_secondary) {
        return false;
    }
    const u32 offset = desc.cbuf_offset + (index << desc.size_shift);
    return std::ranges::any_of(info.storage_buffers_descriptors, [&](const auto& storage) {
        return storage.cbuf_index == desc.cbuf_index && offset >= storage.cbuf_offset &&
               offset - storage.cbuf_offset < sizeof(u64);
    });
}

/// A texture handle no table holds: the texture cache binds its null view and sampler.
inline constexpr std::pair<u32, u32> NULL_TEXTURE_HANDLE{~0U, ~0U};

} // namespace VideoCommon
