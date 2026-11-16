// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Indirect draws whose arguments Metal can't take as the guest wrote them, prepared on the GPU so
// the CPU never reads what the GPU wrote: a draw count from a buffer (Metal's indirect draws have
// none: every draw past the count gets an instance count of 0), indices that must be rewritten
// (8-bit, quads, fans, loops, the provoking vertex: the arguments say how many only on the GPU,
// so a kernel sizes the rewrite and writes the arguments of the rewritten draw) and draws whose
// vertex count GPU work wrote (EDEN_NX2_INDIRECT_DRAWS). Plain C++, no Metal.

#pragma once

#include "common/common_types.h"

namespace Metal::Indirect {

/// Words of the guest's indexed (and Metal's) draw arguments; plain draws use the first 4.
constexpr u32 IndexedWords = 5;
constexpr u32 PlainWords = 4;

/// What `indirect_arguments` copies (buffer 3).
struct CopyParams {
    u32 stride_words; ///< Between the guest's draws
    u32 max_draws;
    u32 has_count;    ///< The count is buffer 1's first word (else max_draws)
    u32 words;        ///< IndexedWords or PlainWords
};

/// What `indirect_rewrite_setup` reads (buffer 2): one guest draw's arguments become a rewrite
/// of its indices and the arguments of the rewritten draw.
struct SetupParams {
    u32 draw;         ///< Which of the guest's draws
    u32 stride_words;
    u32 has_count;
    u32 indexed;
    u32 mode;         ///< IndexConvert::RewriteMode
    u32 index_size;   ///< Guest index bytes (0: a plain draw, its vertices are the indices)
    u32 restart;
    u32 capacity;     ///< Indices the output holds; a longer rewrite is cut there
    u32 source_offset; ///< Byte offset of the index buffer binding in buffer 3
    u32 strip_threads; ///< Threads of the rewrite dispatch of StripRestartMode, else 0
};

/// Words of a draw's control block (buffer 4 of the setup, read by the rewrite and the draw):
/// IndexConvert::RewriteParams first (constant buffers want 256-byte aligned offsets on Macs),
/// the rewrite's threadgroups at ControlGroupsWord, the rewritten draw's arguments at
/// ControlDrawWord; 256 bytes each.
constexpr u32 ControlWords = 64;
constexpr u32 ControlGroupsWord = 8;
constexpr u32 ControlDrawWord = 12;

/// What `grid_arguments` builds: a plain draw's arguments from the two words GPU work wrote.
struct GridParams {
    u32 instances;
    u32 first;          ///< Used when the GPU did not write the first vertex
    u32 first_from_gpu; ///< The second word is the first vertex
    u32 base_instance;
};

[[nodiscard]] inline const char* KernelSource() noexcept {
    return R"(
#include <metal_stdlib>
using namespace metal;

struct CopyParams {
    uint stride_words;
    uint max_draws;
    uint has_count;
    uint words;
};

// Copies the guest's draw arguments packed for Metal; a draw past the GPU's count draws nothing.
kernel void indirect_arguments(device const uint* source [[buffer(0)]],
                               device const uint* count [[buffer(1)]],
                               device uint* out [[buffer(2)]],
                               constant CopyParams& params [[buffer(3)]],
                               uint draw [[thread_position_in_grid]]) {
    if (draw >= params.max_draws) {
        return;
    }
    const uint draws = params.has_count != 0 ? min(count[0], params.max_draws) : params.max_draws;
    for (uint word = 0; word < params.words; ++word) {
        out[draw * params.words + word] = source[draw * params.stride_words + word];
    }
    if (draw >= draws) {
        out[draw * params.words + 1] = 0;
    }
}

struct SetupParams {
    uint draw;
    uint stride_words;
    uint has_count;
    uint indexed;
    uint mode;
    uint index_size;
    uint restart;
    uint capacity;
    uint source_offset;
    uint strip_threads;
};

static uint converted_count(uint mode, uint count) {
    switch (mode) {
    case 1:
    case 11:
        return count / 4 * 6;
    case 2:
    case 12:
        return count >= 4 ? (count - 2) / 2 * 6 : 0;
    case 3:
    case 13:
    case 6:
    case 7:
        return count >= 3 ? (count - 2) * 3 : 0;
    case 4:
    case 10:
        return count >= 2 ? count + 1 : count;
    case 5:
        return count / 3 * 3;
    case 8:
        return count / 2 * 2;
    default:
        return count;
    }
}

// One guest draw: the rewrite's size and constants, and the arguments of the rewritten draw.
kernel void indirect_rewrite_setup(device const uint* source [[buffer(0)]],
                                   device const uint* count [[buffer(1)]],
                                   constant SetupParams& params [[buffer(2)]],
                                   device uint* control [[buffer(4)]],
                                   uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    device const uint* arguments = source + params.draw * params.stride_words;
    const bool drawn = params.has_count == 0 || params.draw < count[0];
    const uint guest_count = drawn ? arguments[0] : 0;
    const uint instances = arguments[1];
    const uint first = arguments[2];
    uint converted = converted_count(params.mode, guest_count);
    if (converted > params.capacity) {
        // Cut at a whole primitive of the rewritten list.
        converted = params.capacity / 6 * 6;
    }
    const uint threads = params.strip_threads != 0 ? 1 : (converted + 63) / 64;
    // IndexConvert::RewriteParams; a plain draw's "indices" are its vertices from `first`.
    control[0] = params.mode;
    control[1] = params.index_size;
    control[2] = guest_count;
    control[3] = params.restart;
    control[4] = converted;
    control[5] = params.index_size != 0 ? params.source_offset + first * params.index_size : first;
    control[8] = converted != 0 ? threads : 0;
    control[9] = 1;
    control[10] = 1;
    control[12] = converted;
    control[13] = drawn ? instances : 0;
    control[14] = 0;
    control[15] = params.indexed != 0 ? arguments[3] : 0;
    control[16] = params.indexed != 0 ? arguments[4] : arguments[3];
}

struct GridParams {
    uint instances;
    uint first;
    uint first_from_gpu;
    uint base_instance;
};

// A plain draw whose vertex count (and first vertex) GPU work wrote.
kernel void grid_arguments(device const uint* words [[buffer(0)]],
                           device uint* out [[buffer(1)]],
                           constant GridParams& params [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    out[0] = words[0];
    out[1] = params.instances;
    out[2] = params.first_from_gpu != 0 ? words[1] : params.first;
    out[3] = params.base_instance;
}
)";
}

} // namespace Metal::Indirect
