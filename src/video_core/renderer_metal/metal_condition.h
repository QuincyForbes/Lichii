// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Conditional rendering decided on the GPU. Metal has no predicated draws (Vulkan's
// VK_EXT_conditional_rendering): while the condition reads an occlusion report the GPU has not
// finished, the draws go through indirect arguments in a region of the stream ring, and a kernel
// recorded ahead of them sets the instance count of each to 0 when the condition fails. The
// kernel runs after the render passes that counted the samples and reads their visibility
// results; it reads the arguments the CPU wrote for the region's draws after it was recorded,
// which are all there by the time the command buffer is committed. Plain C++, no Metal.

#pragma once

#include <array>

#include "common/common_types.h"

namespace Metal::Condition {

/// Visibility slots summed into one report at most; a report from more passes is resolved on
/// the CPU.
constexpr u32 MaxSlots = 16;
/// Draws one region's kernel takes; the next draw starts a new region.
constexpr u32 RegionDraws = 64;
/// Words of a region before the first draw's arguments (the first holds the draw count).
constexpr u32 HeaderWords = 4;
/// Words per draw's arguments (MTLDrawIndexedPrimitivesIndirectArguments; the instance count is
/// the second word for indexed and plain draws alike).
constexpr u32 ArgumentWords = 5;

enum class Mode : u32 {
    /// Maxwell's Conditional: both words of the first value non-zero.
    BothWordsNonZero = 0,
    Equal = 1,
    NotEqual = 2,
};

/// What the kernel compares: two 64-bit values, each the guest memory's value with the bits a
/// pending report writes (write_mask) taken from the sum of its visibility slots.
struct Params {
    u32 mode;
    u32 pad;
    std::array<u32, 2> slot_count;
    std::array<std::array<u32, 2>, 2> memory;
    std::array<std::array<u32, 2>, 2> write_mask;
    std::array<std::array<u32, MaxSlots>, 2> slots;
};

/// The kernel's decision on the CPU, from the visibility results.
[[nodiscard]] inline bool Evaluate(const Params& params, const u64* visibility) noexcept {
    std::array<u64, 2> values{};
    for (size_t value = 0; value < 2; ++value) {
        u64 sum = 0;
        for (u32 slot = 0; slot < params.slot_count[value]; ++slot) {
            sum += visibility[params.slots[value][slot]];
        }
        const u64 memory = u64{params.memory[value][0]} | (u64{params.memory[value][1]} << 32);
        const u64 mask =
            u64{params.write_mask[value][0]} | (u64{params.write_mask[value][1]} << 32);
        values[value] = (memory & ~mask) | (sum & mask);
    }
    switch (static_cast<Mode>(params.mode)) {
    case Mode::BothWordsNonZero:
        return static_cast<u32>(values[0]) != 0 && static_cast<u32>(values[0] >> 32) != 0;
    case Mode::Equal:
        return values[0] == values[1];
    default:
        return values[0] != values[1];
    }
}

/// MSL of the kernel `condition_arguments`: visibility results at buffer 0, Params at buffer 1,
/// the region at buffer 2; one thread per draw of the region (RegionDraws).
[[nodiscard]] inline const char* KernelSource() noexcept {
    return R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint mode;
    uint pad;
    uint slot_count[2];
    uint memory[2][2];
    uint write_mask[2][2];
    uint slots[2][16];
};

kernel void condition_arguments(device const ulong* visibility [[buffer(0)]],
                                constant Params& params [[buffer(1)]],
                                device uint* region [[buffer(2)]],
                                uint draw [[thread_position_in_grid]]) {
    if (draw >= region[0]) {
        return;
    }
    ulong values[2];
    for (uint value = 0; value < 2; ++value) {
        ulong sum = 0;
        for (uint slot = 0; slot < params.slot_count[value]; ++slot) {
            sum += visibility[params.slots[value][slot]];
        }
        const ulong memory = ulong(params.memory[value][0]) | (ulong(params.memory[value][1]) << 32);
        const ulong mask =
            ulong(params.write_mask[value][0]) | (ulong(params.write_mask[value][1]) << 32);
        values[value] = (memory & ~mask) | (sum & mask);
    }
    bool passes;
    if (params.mode == 0) {
        passes = uint(values[0]) != 0 && uint(values[0] >> 32) != 0;
    } else if (params.mode == 1) {
        passes = values[0] == values[1];
    } else {
        passes = values[0] != values[1];
    }
    if (!passes) {
        region[4 + draw * 5 + 1] = 0;
    }
}
)";
}

} // namespace Metal::Condition
