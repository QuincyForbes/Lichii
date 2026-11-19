// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <map>
#include <span>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "video_core/shader_handle_program.h"

namespace {

using Shader::HandleOp;

constexpr u32 Op(HandleOp op) {
    return static_cast<u32>(op);
}

struct CountingHandleMemory {
    std::map<GPUVAddr, u32> words;
    mutable std::map<GPUVAddr, u32> reads;

    void WriteU64(GPUVAddr addr, u64 value) {
        words[addr] = static_cast<u32>(value);
        words[addr + sizeof(u32)] = static_cast<u32>(value >> 32);
    }

    u32 ReadHandleWord(GPUVAddr addr) const {
        ++reads[addr];
        const auto it = words.find(addr);
        return it != words.end() ? it->second : 0U;
    }
};

struct TypedReadMemory {
    std::map<GPUVAddr, u32> words;

    template <typename T>
    T Read(GPUVAddr addr) const {
        static_assert(sizeof(T) == sizeof(u32));
        const auto it = words.find(addr);
        return it != words.end() ? static_cast<T>(it->second) : T{};
    }
};

std::vector<GPUVAddr> ResolveSsboAddrs(const Shader::Info& info,
                                       std::span<const GPUVAddr> cbuf_addrs,
                                       const CountingHandleMemory& memory,
                                       u64 storage_alignment) {
    const GPUVAddr storage_mask{~(GPUVAddr{(std::max)(storage_alignment, u64{1})} - 1)};
    const auto read_u64 = [&](GPUVAddr addr) -> u64 {
        const auto lo = memory.words.find(addr);
        const auto hi = memory.words.find(addr + sizeof(u32));
        const u32 lo_word = lo != memory.words.end() ? lo->second : 0U;
        const u32 hi_word = hi != memory.words.end() ? hi->second : 0U;
        return u64{lo_word} | (u64{hi_word} << 32);
    };
    std::vector<GPUVAddr> addrs(info.storage_buffers_descriptors.size(), GPUVAddr{});
    for (size_t i = 0; i < info.storage_buffers_descriptors.size(); ++i) {
        const auto& desc = info.storage_buffers_descriptors[i];
        const GPUVAddr cbuf = desc.cbuf_index < cbuf_addrs.size() ? cbuf_addrs[desc.cbuf_index] : 0;
        if (cbuf == 0) {
            continue;
        }
        GPUVAddr base = read_u64(cbuf + desc.cbuf_offset);
        if (base != 0 && desc.indirect_offset != ~0U) {
            base = read_u64(base + desc.indirect_offset);
        }
        addrs[i] = base & storage_mask;
    }
    return addrs;
}

void CheckBatchMatchesSingle(const Shader::Info& info, u32 program_idx, u32 count,
                             std::span<const GPUVAddr> cbuf_addrs, u64 storage_alignment,
                             const CountingHandleMemory& memory) {
    const std::vector<GPUVAddr> ssbo_addrs =
        ResolveSsboAddrs(info, cbuf_addrs, memory, storage_alignment);
    std::vector<u32> batch(count, 0xDEADBEEFU);
    VideoCommon::EvaluateTextureHandles(info.handle_programs[program_idx], count, cbuf_addrs,
                                        ssbo_addrs, memory, batch);
    const auto cbuf_lookup = [&](u32 index) -> GPUVAddr {
        return index < cbuf_addrs.size() ? cbuf_addrs[index] : GPUVAddr{};
    };
    for (u32 element = 0; element < count; ++element) {
        CAPTURE(element);
        const u32 single = VideoCommon::EvaluateTextureHandle(
            info, program_idx, memory, cbuf_lookup, storage_alignment, element);
        CHECK(batch[element] == single);
    }
}

} // namespace

TEST_CASE("EvaluateTextureHandles matches single evaluation on static, dynamic, and >16 cbuf words",
          "[video_core][texture_handles]") {
    CountingHandleMemory memory;
    std::array<GPUVAddr, Shader::Info::MAX_CBUFS> cbuf_addrs{};
    cbuf_addrs[0] = 0x10000;
    cbuf_addrs[2] = 0x20000;
    // cbuf_addrs[1] stays 0 (unbound).

    // Populate 20 distinct static words in cbuf 0 to exceed the 16-entry CbufWord cache, plus
    // dynamic element-indexed entries in cbuf 2.
    for (u32 i = 0; i < 20; ++i) {
        memory.words[cbuf_addrs[0] + i * 4] = 0x100U + i * 3U;
    }
    for (u32 elem = 0; elem < 12; ++elem) {
        memory.words[cbuf_addrs[2] + 0x40 + elem * 4] = 0x24000000U | (elem * 0x11U);
    }

    Shader::Info info;
    Shader::HandleProgram prog;
    // Sum 20 distinct cbuf[0] words (overflows 16-word cache), XOR unbound cbuf[1] and OOB cbuf[99],
    // then OR with cbuf_dynamic[2](0x40 + element * 4).
    prog.push_back(Op(HandleOp::Cbuf));
    prog.push_back(0U);
    prog.push_back(0U);
    for (u32 i = 1; i < 20; ++i) {
        prog.push_back(Op(HandleOp::Cbuf));
        prog.push_back(0U);
        prog.push_back(i * 4U);
        prog.push_back(Op(HandleOp::Add));
    }
    prog.push_back(Op(HandleOp::Cbuf));
    prog.push_back(1U);
    prog.push_back(0U);
    prog.push_back(Op(HandleOp::Xor));
    prog.push_back(Op(HandleOp::Cbuf));
    prog.push_back(99U);
    prog.push_back(0U);
    prog.push_back(Op(HandleOp::Xor));

    // Dynamic offset: (Index << 2) + 0x40 -> CbufDynamic(2)
    prog.push_back(Op(HandleOp::Index));
    prog.push_back(Op(HandleOp::Immediate));
    prog.push_back(2U);
    prog.push_back(Op(HandleOp::Shl));
    prog.push_back(Op(HandleOp::Immediate));
    prog.push_back(0x40U);
    prog.push_back(Op(HandleOp::Add));
    prog.push_back(Op(HandleOp::CbufDynamic));
    prog.push_back(2U);
    prog.push_back(Op(HandleOp::Or));

    info.handle_programs.push_back(prog);

    memory.reads.clear();
    std::array<u32, 12> batch{};
    VideoCommon::EvaluateTextureHandles(info.handle_programs[0], 12, cbuf_addrs, {}, memory, batch);
    // First 16 static cbuf words are cached across all 12 elements (read once each); words 16..19
    // bypass the full cache and still produce identical handles.
    for (u32 i = 0; i < 16; ++i) {
        CHECK(memory.reads[cbuf_addrs[0] + i * 4] == 1U);
    }
    for (u32 i = 16; i < 20; ++i) {
        CHECK(memory.reads[cbuf_addrs[0] + i * 4] == 12U);
    }

    CheckBatchMatchesSingle(info, 0, 12, cbuf_addrs, 16, memory);
}

TEST_CASE("EvaluateTextureHandles matches single evaluation across SSBO slots 0..63 and indirect pointers",
          "[video_core][texture_handles]") {
    CountingHandleMemory memory;
    std::array<GPUVAddr, Shader::Info::MAX_CBUFS> cbuf_addrs{};
    cbuf_addrs[0] = 0x8000;
    constexpr u64 storage_alignment = 32;

    Shader::Info info;
    info.storage_buffers_descriptors.resize(Shader::Info::MAX_SSBOS);
    for (u32 slot = 0; slot < Shader::Info::MAX_SSBOS; ++slot) {
        const u32 cbuf_off = slot * 8U;
        const bool indirect = (slot % 2) == 1;
        info.storage_buffers_descriptors[slot] = {
            .cbuf_index = 0,
            .cbuf_offset = cbuf_off,
            .count = 1,
            .is_written = false,
            .indirect_offset = indirect ? 0x0CU : ~0U,
        };
        if (slot == 7) {
            // Null SSBO pointer in slot 7.
            memory.WriteU64(cbuf_addrs[0] + cbuf_off, 0);
            continue;
        }
        const GPUVAddr raw_ssbo = 0x100000ULL + u64{slot} * 0x1000ULL + 0x14ULL; // +0x14 past 32B
        const GPUVAddr aligned_ssbo = raw_ssbo & ~(storage_alignment - 1);
        if (indirect) {
            const GPUVAddr table_addr = 0x50004ULL + u64{slot} * 0x40ULL; // unaligned table
            memory.WriteU64(cbuf_addrs[0] + cbuf_off, table_addr);
            memory.WriteU64(table_addr + 0x0CU, raw_ssbo);
        } else {
            memory.WriteU64(cbuf_addrs[0] + cbuf_off, raw_ssbo);
        }
        for (u32 elem = 0; elem < 8; ++elem) {
            memory.words[aligned_ssbo + 0x14ULL + elem * 4U] =
                (slot << 16) ^ (elem * 0x101U) ^ 0x5A00U;
        }
    }

    // Program reads from slot 0, slot 7 (null), slot 31 (indirect), slot 63 (indirect MSB of 64-bit
    // known_bases mask!), and OOB slot 64.
    Shader::HandleProgram prog = {
        // offset = 0x14 + (Index << 2), saved in Tee(3)
        Op(HandleOp::Index),
        Op(HandleOp::Immediate), 2U,
        Op(HandleOp::Shl),
        Op(HandleOp::Immediate), 0x14U,
        Op(HandleOp::Add),
        Op(HandleOp::Tee), 3U,
        Op(HandleOp::Storage), 0U,

        Op(HandleOp::Load), 3U,
        Op(HandleOp::Storage), 31U,
        Op(HandleOp::Xor),

        Op(HandleOp::Load), 3U,
        Op(HandleOp::Storage), 63U,
        Op(HandleOp::Add),

        Op(HandleOp::Load), 3U,
        Op(HandleOp::Storage), 7U, // null SSBO -> 0
        Op(HandleOp::Add),

        Op(HandleOp::Load), 3U,
        Op(HandleOp::Storage), 64U, // >= MAX_SSBOS -> 0
        Op(HandleOp::Add),
    };
    info.handle_programs.push_back(prog);

    CheckBatchMatchesSingle(info, 0, 8, cbuf_addrs, storage_alignment, memory);

    SECTION("Generic MemoryReader::Read<u32> path matches ReadHandleWord") {
        TypedReadMemory typed_memory{.words = memory.words};
        const std::vector<GPUVAddr> ssbo_addrs =
            ResolveSsboAddrs(info, cbuf_addrs, memory, storage_alignment);
        std::array<u32, 8> via_handle_word{};
        std::array<u32, 8> via_typed_read{};
        VideoCommon::EvaluateTextureHandles(std::span<const u32>{prog}, 8, cbuf_addrs, ssbo_addrs,
                                            memory, via_handle_word);
        VideoCommon::EvaluateTextureHandles(std::span<const u32>{prog}, 8, cbuf_addrs, ssbo_addrs,
                                            typed_memory, via_typed_read);
        CHECK(via_typed_read == via_handle_word);
    }
}

TEST_CASE("EvaluateTextureHandles matches single evaluation on arithmetic, shifts, selects, and clamping",
          "[video_core][texture_handles]") {
    CountingHandleMemory memory;
    std::array<GPUVAddr, 2> cbuf_addrs{0x4000, 0x5000};
    memory.words[0x4000] = 0x80000010U; // negative as s32
    memory.words[0x4004] = 0x00000025U; // shift >= 32 test operand
    memory.words[0x5000] = 0x00ABCDEFU;

    Shader::Info info;
    // Exercises Select, ULt/SLt, Shl/Shr/Sar (both normal and >= 32 clamping), UMin/SMax, Sub/Mul,
    // Neg/Not/LNot, and Tee/Load.
    Shader::HandleProgram prog = {
        // cond = (Index & 1) != 0
        Op(HandleOp::Index),
        Op(HandleOp::Immediate), 1U,
        Op(HandleOp::And),

        // true_val: ((cbuf[0][0] >>s 4) ^ (cbuf[0][0] << 35) ^ (cbuf[0][0] >>u 40)) + Index * 5
        Op(HandleOp::Cbuf), 0U, 0U,
        Op(HandleOp::Tee), 0U,
        Op(HandleOp::Immediate), 4U,
        Op(HandleOp::Sar),
        Op(HandleOp::Load), 0U,
        Op(HandleOp::Cbuf), 0U, 4U, // 37 >= 32 -> Shl clamps to 0
        Op(HandleOp::Shl),
        Op(HandleOp::Xor),
        Op(HandleOp::Load), 0U,
        Op(HandleOp::Immediate), 40U, // >= 32 -> Shr clamps to 0
        Op(HandleOp::Shr),
        Op(HandleOp::Xor),
        Op(HandleOp::Index),
        Op(HandleOp::Immediate), 5U,
        Op(HandleOp::Mul),
        Op(HandleOp::Add),

        // false_val: UMin(cbuf[1][0], 0x00FFFF00) - SMax(cbuf[0][0], -8) + Sar(cbuf[0][0], 35)
        Op(HandleOp::Cbuf), 1U, 0U,
        Op(HandleOp::Immediate), 0x00FFFF00U,
        Op(HandleOp::UMin),
        Op(HandleOp::Load), 0U,
        Op(HandleOp::Immediate), 8U,
        Op(HandleOp::Neg),
        Op(HandleOp::SMax),
        Op(HandleOp::Sub),
        Op(HandleOp::Load), 0U,
        Op(HandleOp::Immediate), 35U, // >= 32 -> Sar clamps shift to 31 -> 0xFFFFFFFF
        Op(HandleOp::Sar),
        Op(HandleOp::Add),

        Op(HandleOp::Select),
    };
    info.handle_programs.push_back(prog);

    CheckBatchMatchesSingle(info, 0, 16, cbuf_addrs, 16, memory);

    SECTION("Output span smaller than count clamps cleanly without overrunning") {
        std::array<u32, 10> guarded{};
        guarded.fill(0xCAFEBABEU);
        VideoCommon::EvaluateTextureHandles(info.handle_programs[0], 16, cbuf_addrs, {}, memory,
                                            std::span<u32>{guarded.data(), 6});
        const auto cbuf_lookup = [&](u32 idx) -> GPUVAddr {
            return idx < cbuf_addrs.size() ? cbuf_addrs[idx] : 0;
        };
        for (u32 i = 0; i < 6; ++i) {
            CHECK(guarded[i] ==
                  VideoCommon::EvaluateTextureHandle(info, 0, memory, cbuf_lookup, 16, i));
        }
        for (size_t i = 6; i < guarded.size(); ++i) {
            CHECK(guarded[i] == 0xCAFEBABEU);
        }
    }

    SECTION("Malformed handle program evaluates to 0 for both single and batch") {
        Shader::Info bad_info;
        bad_info.handle_programs.push_back({999U});
        CheckBatchMatchesSingle(bad_info, 0, 4, cbuf_addrs, 16, memory);
        std::array<u32, 4> out{1, 2, 3, 4};
        VideoCommon::EvaluateTextureHandles(bad_info.handle_programs[0], 4, cbuf_addrs, {}, memory,
                                            out);
        CHECK(out == std::array<u32, 4>{0, 0, 0, 0});
    }
}
