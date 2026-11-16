// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <cstring>

#include "video_core/renderer_metal/metal_index_convert.h"

namespace Metal::IndexConvert {

bool AdjacencyEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ADJACENCY");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

namespace {

[[nodiscard]] bool IsAdjacency(Topology topology) noexcept {
    return topology == Topology::LinesAdjacency || topology == Topology::LineStripAdjacency ||
           topology == Topology::TrianglesAdjacency ||
           topology == Topology::TriangleStripAdjacency;
}

} // Anonymous namespace

bool NeedsConversion(Topology topology) noexcept {
    if (IsAdjacency(topology) && !AdjacencyEnabled()) {
        return false;
    }
    switch (topology) {
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::TriangleFan:
    case Topology::Polygon:
    case Topology::LineLoop:
    case Topology::LinesAdjacency:
    case Topology::LineStripAdjacency:
    case Topology::TrianglesAdjacency:
    case Topology::TriangleStripAdjacency:
        return true;
    default:
        return false;
    }
}

bool RotatesForProvokingVertex(Topology topology) noexcept {
    return topology == Topology::Triangles || topology == Topology::TriangleStrip;
}

Topology ConvertedTopology(Topology topology, bool provoking_last) noexcept {
    if (provoking_last && RotatesForProvokingVertex(topology)) {
        return Topology::Triangles;
    }
    if (IsAdjacency(topology) && !AdjacencyEnabled()) {
        return topology;
    }
    switch (topology) {
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::TriangleFan:
    case Topology::Polygon:
    case Topology::TrianglesAdjacency:
    case Topology::TriangleStripAdjacency:
        return Topology::Triangles;
    case Topology::LineLoop:
    case Topology::LineStripAdjacency:
        return Topology::LineStrip;
    case Topology::LinesAdjacency:
        return Topology::Lines;
    default:
        return topology;
    }
}

u32 ConvertedCount(Topology topology, u32 count, bool provoking_last) noexcept {
    if (provoking_last && topology == Topology::Triangles) {
        return count / 3 * 3;
    }
    if (provoking_last && topology == Topology::TriangleStrip) {
        return count >= 3 ? (count - 2) * 3 : 0;
    }
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
    case Topology::LinesAdjacency:
        return count / 4 * 2;
    case Topology::LineStripAdjacency:
        return count >= 4 ? count - 2 : 0;
    case Topology::TrianglesAdjacency:
        return count / 6 * 3;
    case Topology::TriangleStripAdjacency:
        return count >= 6 ? (count - 4) / 2 * 3 : 0;
    default:
        return count;
    }
}

u32 Convert(Topology topology, std::span<const u8> source, u32 index_size, u32 first, u32 count,
            bool restart, std::span<u32> out, bool provoking_last) noexcept {
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
    const u32 total = ConvertedCount(topology, count, provoking_last);
    if (out.size() < total) {
        return 0;
    }
    u32 written = 0;
    const auto emit = [&](u32 vertex) { out[written++] = index(vertex); };
    if (provoking_last && topology == Topology::Triangles) {
        // The last vertex first; a rotation keeps the winding.
        for (u32 vertex = 0; vertex + 3 <= count; vertex += 3) {
            emit(vertex + 2);
            emit(vertex);
            emit(vertex + 1);
        }
        return written;
    }
    if (provoking_last && topology == Topology::TriangleStrip) {
        // Triangle i of a strip is i, i+1, i+2 (even) or i, i+2, i+1 (odd), its last vertex
        // i+2; each is written from that one on. A restart index begins a new strip.
        u32 start = 0;
        for (u32 vertex = 0; vertex < count; ++vertex) {
            if (restart && index(vertex) == 0xFFFFFFFFU) {
                start = vertex + 1;
                continue;
            }
            if (vertex < start + 2) {
                continue;
            }
            const u32 triangle = vertex - 2;
            emit(vertex);
            if ((triangle - start) % 2 == 0) {
                emit(triangle);
                emit(triangle + 1);
            } else {
                emit(triangle + 1);
                emit(triangle);
            }
        }
        return written;
    }
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
    // Without a geometry program Vulkan draws adjacency primitives without their adjacent
    // vertices: the line of 1 and 2 of each four, the strip of all but the first and last
    // vertex, the triangle of 0, 2 and 4 of each six, and strip triangle i of 2i, 2i + 2 and
    // 2i + 4 (2i + 2 first when i is odd, keeping the winding).
    case Topology::LinesAdjacency:
        for (u32 line = 0; line + 4 <= count; line += 4) {
            emit(line + 1);
            emit(line + 2);
        }
        break;
    case Topology::LineStripAdjacency:
        for (u32 vertex = 1; count >= 4 && vertex + 1 < count; ++vertex) {
            emit(vertex);
        }
        break;
    case Topology::TrianglesAdjacency:
        for (u32 triangle = 0; triangle + 6 <= count; triangle += 6) {
            emit(triangle);
            emit(triangle + 2);
            emit(triangle + 4);
        }
        break;
    case Topology::TriangleStripAdjacency:
        for (u32 triangle = 0; count >= 6 && triangle < (count - 4) / 2; ++triangle) {
            const bool odd = triangle % 2 != 0;
            emit(odd ? 2 * triangle + 2 : 2 * triangle);
            emit(odd ? 2 * triangle : 2 * triangle + 2);
            emit(2 * triangle + 4);
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

u32 RewriteMode(Topology topology, bool provoking_last) noexcept {
    if (provoking_last && topology == Topology::Triangles) {
        return 5;
    }
    if (provoking_last && topology == Topology::TriangleStrip) {
        return 6;
    }
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
    case Topology::LinesAdjacency:
        return 7;
    case Topology::LineStripAdjacency:
        return 8;
    case Topology::TrianglesAdjacency:
        return 9;
    case Topology::TriangleStripAdjacency:
        return 10;
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
    case 5:
        element = index / 3 * 3 + (index % 3 == 0 ? 2 : index % 3 - 1);
        break;
    case 6: {
        const uint triangle = index / 3;
        const uint corner = index % 3;
        element = corner == 0 ? triangle + 2
                  : (triangle % 2 == 0) == (corner == 1) ? triangle
                                                         : triangle + 1;
        break;
    }
    case 7:
        element = index / 2 * 4 + 1 + index % 2;
        break;
    case 8:
        element = index + 1;
        break;
    case 9:
        element = index / 3 * 6 + index % 3 * 2;
        break;
    case 10: {
        const uint triangle = index / 3;
        const uint corner = index % 3;
        element = corner == 2               ? 2 * triangle + 4
                  : (triangle % 2 == 0) == (corner == 0) ? 2 * triangle
                                                         : 2 * triangle + 2;
        break;
    }
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
