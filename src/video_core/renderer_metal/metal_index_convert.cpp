// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "video_core/renderer_metal/metal_index_convert.h"

namespace Metal::IndexConvert {

bool NeedsConversion(Topology topology) noexcept {
    switch (topology) {
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::TriangleFan:
    case Topology::Polygon:
    case Topology::LineLoop:
        return true;
    default:
        return false;
    }
}

Topology ConvertedTopology(Topology topology) noexcept {
    switch (topology) {
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::TriangleFan:
    case Topology::Polygon:
        return Topology::Triangles;
    case Topology::LineLoop:
        return Topology::LineStrip;
    default:
        return topology;
    }
}

u32 ConvertedCount(Topology topology, u32 count) noexcept {
    switch (topology) {
    case Topology::Quads:
        return count / 4 * 6;
    case Topology::QuadStrip:
        return count >= 4 ? (count - 2) / 2 * 6 : 0;
    case Topology::TriangleFan:
    case Topology::Polygon:
        return count >= 3 ? (count - 2) * 3 : 0;
    case Topology::LineLoop:
        return count >= 2 ? count + 1 : count;
    default:
        return count;
    }
}

u32 Convert(Topology topology, std::span<const u8> source, u32 index_size, u32 first, u32 count,
            bool restart, std::span<u32> out) noexcept {
    const auto index = [&](u32 vertex) -> u32 {
        if (source.empty()) {
            return first + vertex;
        }
        const size_t offset = size_t{vertex} * index_size;
        if (offset + index_size > source.size()) {
            return 0;
        }
        switch (index_size) {
        case 1: {
            const u8 value = source[offset];
            return restart && value == 0xFF ? 0xFFFFFFFFU : value;
        }
        case 2: {
            u16 value;
            std::memcpy(&value, source.data() + offset, sizeof(value));
            return restart && value == 0xFFFF ? 0xFFFFFFFFU : value;
        }
        default: {
            u32 value;
            std::memcpy(&value, source.data() + offset, sizeof(value));
            return value;
        }
        }
    };
    const u32 total = ConvertedCount(topology, count);
    if (out.size() < total) {
        return 0;
    }
    u32 written = 0;
    const auto emit = [&](u32 vertex) { out[written++] = index(vertex); };
    switch (topology) {
    case Topology::Quads:
        for (u32 quad = 0; quad + 4 <= count; quad += 4) {
            emit(quad);
            emit(quad + 1);
            emit(quad + 2);
            emit(quad);
            emit(quad + 2);
            emit(quad + 3);
        }
        break;
    case Topology::QuadStrip:
        // Each new pair closes a quad with the pair before it, in Vulkan's order
        // (QuadStripIndexBuffer: 0 3 1, 0 2 3).
        for (u32 pair = 0; pair + 4 <= count; pair += 2) {
            emit(pair);
            emit(pair + 3);
            emit(pair + 1);
            emit(pair);
            emit(pair + 2);
            emit(pair + 3);
        }
        break;
    case Topology::TriangleFan:
    case Topology::Polygon:
        for (u32 vertex = 1; vertex + 1 < count; ++vertex) {
            emit(0);
            emit(vertex);
            emit(vertex + 1);
        }
        break;
    case Topology::LineLoop:
        for (u32 vertex = 0; vertex < count; ++vertex) {
            emit(vertex);
        }
        if (count >= 2) {
            emit(0);
        }
        break;
    default:
        for (u32 vertex = 0; vertex < count; ++vertex) {
            emit(vertex);
        }
        break;
    }
    return written;
}

u32 RewriteMode(Topology topology) noexcept {
    switch (topology) {
    case Topology::Quads:
        return 1;
    case Topology::QuadStrip:
        return 2;
    case Topology::TriangleFan:
    case Topology::Polygon:
        return 3;
    case Topology::LineLoop:
        return 4;
    default:
        return 0;
    }
}

const char* RewriteKernelSource() noexcept {
    // Convert's orders, per index written.
    return R"(
#include <metal_stdlib>
using namespace metal;

struct RewriteParams {
    uint mode;
    uint index_size;
    uint count;
    uint restart;
    uint converted;
    uint source_offset;
};

constant uint quad_order[6] = {0, 1, 2, 0, 2, 3};
constant uint quad_strip_order[6] = {0, 3, 1, 0, 2, 3};

kernel void rewrite_indices(device const uchar* source [[buffer(0)]],
                            device uint* indices [[buffer(1)]],
                            constant RewriteParams& params [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= params.converted) {
        return;
    }
    uint element;
    switch (params.mode) {
    case 1:
        element = index / 6 * 4 + quad_order[index % 6];
        break;
    case 2:
        element = index / 6 * 2 + quad_strip_order[index % 6];
        break;
    case 3:
        element = index % 3 == 0 ? 0 : index / 3 + index % 3;
        break;
    case 4:
        element = index < params.count ? index : 0;
        break;
    default:
        element = index;
        break;
    }
    uint value = 0;
    if (element < params.count) {
        const uint at = params.source_offset + element * params.index_size;
        switch (params.index_size) {
        case 1:
            value = source[at];
            value = params.restart != 0 && value == 0xFFu ? 0xFFFFFFFFu : value;
            break;
        case 2:
            value = uint(source[at]) | (uint(source[at + 1]) << 8);
            value = params.restart != 0 && value == 0xFFFFu ? 0xFFFFFFFFu : value;
            break;
        default:
            value = uint(source[at]) | (uint(source[at + 1]) << 8) |
                    (uint(source[at + 2]) << 16) | (uint(source[at + 3]) << 24);
            break;
        }
    }
    indices[index] = value;
}
)";
}

} // namespace Metal::IndexConvert
