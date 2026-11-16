// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The kernel that turns the guest's indirect mesh draw records into Metal's indirect mesh
// arguments (RasterizerMetal::ConvertMeshIndirect, the smoke test). Plain C++.

#pragma once

#include <string_view>

#include "common/common_types.h"

namespace Metal {

/// Guest draw records (their first word: the mesh threadgroup count) to Metal's indirect mesh
/// arguments (MTLDispatchThreadgroupsIndirectArguments), x = that count, y = z = 1; records past
/// a count from a buffer draw nothing. Buffers: 0 records, 1 count, 2 arguments, 3 parameters.
inline constexpr std::string_view MeshIndirectKernelSource = R"(#include <metal_stdlib>
using namespace metal;
struct Params {
    uint draws;
    uint stride_words;
    uint has_count;
};
kernel void mesh_indirect_args(device const uint* records [[buffer(0)]],
                               device const uint* count [[buffer(1)]],
                               device uint* args [[buffer(2)]],
                               constant Params& params [[buffer(3)]],
                               uint draw [[thread_position_in_grid]]) {
    if (draw >= params.draws) {
        return;
    }
    uint groups = records[draw * params.stride_words];
    if (params.has_count != 0 && draw >= count[0]) {
        groups = 0;
    }
    args[draw * 3 + 0] = groups;
    args[draw * 3 + 1] = 1;
    args[draw * 3 + 2] = 1;
}
)";

inline constexpr std::string_view MeshIndirectKernelName = "mesh_indirect_args";

/// The kernel's parameters (buffer 3).
struct MeshIndirectParams {
    u32 draws;
    u32 stride_words;
    u32 has_count;
};

/// Threads per threadgroup of its dispatch.
inline constexpr u32 MeshIndirectGroupSize = 64;

} // namespace Metal
