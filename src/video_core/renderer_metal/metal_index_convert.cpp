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
    switch (topology) {
    case Topology::Triangles:
    case Topology::TriangleStrip:
    case Topology::Quads:
    case Topology::QuadStrip:
    case Topology::TriangleFan:
    case Topology::Polygon:
    case Topology::Lines:
    case Topology::LineStrip:
    case Topology::LineLoop:
        return true;
    default:
        return false;
    }
}

bool RestartsStrips(Topology topology) noexcept {
    return topology == Topology::TriangleStrip || topology == Topology::LineStrip;
}

Topology ConvertedTopology(Topology topology, bool provoking_last) noexcept {
    if (provoking_last && RotatesForProvokingVertex(topology)) {
        switch (topology) {
        case Topology::Lines:
            return Topology::Lines;
        case Topology::LineStrip:
        case Topology::LineLoop:
            return Topology::LineStrip;
        default:
            return Topology::Triangles;
        }
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
    if (provoking_last) {
        switch (topology) {
        case Topology::Triangles:
            return count / 3 * 3;
        case Topology::TriangleStrip:
            return count >= 3 ? (count - 2) * 3 : 0;
        case Topology::Lines:
            return count / 2 * 2;
        case Topology::LineStrip:
            return count;
        default:
            break;
        }
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
    provoking_last = provoking_last && RotatesForProvokingVertex(topology);
    const u32 total = ConvertedCount(topology, count, provoking_last);
    if (out.size() < total) {
        return 0;
    }
    u32 written = 0;
    const auto emit = [&](u32 vertex) { out[written++] = index(vertex); };
    // Vulkan's last vertex convention, where Metal takes flat inputs from the first: a triangle
    // (a, b, c) is written (c, a, b), which keeps its winding.
    const auto triangle = [&](u32 a, u32 b, u32 c) {
        if (provoking_last) {
            emit(c);
            emit(a);
            emit(b);
        } else {
            emit(a);
            emit(b);
            emit(c);
        }
    };
    switch (topology) {
    case Topology::Triangles:
        if (!provoking_last) {
            break;
        }
        if (!source.empty() && source.size() >= size_t{count} * index_size) {
            const u32 tris = count / 3 * 3;
            if (index_size == 2) {
                const u8* ptr = source.data();
                for (u32 vertex = 0; vertex < tris; vertex += 3) {
                    u16 a, b, c;
                    std::memcpy(&a, ptr + size_t{vertex} * 2, 2);
                    std::memcpy(&b, ptr + size_t{vertex + 1} * 2, 2);
                    std::memcpy(&c, ptr + size_t{vertex + 2} * 2, 2);
                    out[written++] = restart && c == 0xFFFF ? 0xFFFFFFFFU : u32{c};
                    out[written++] = restart && a == 0xFFFF ? 0xFFFFFFFFU : u32{a};
                    out[written++] = restart && b == 0xFFFF ? 0xFFFFFFFFU : u32{b};
                }
                return written;
            }
            if (index_size == 4) {
                const u8* ptr = source.data();
                for (u32 vertex = 0; vertex < tris; vertex += 3) {
                    u32 a, b, c;
                    std::memcpy(&a, ptr + size_t{vertex} * 4, 4);
                    std::memcpy(&b, ptr + size_t{vertex + 1} * 4, 4);
                    std::memcpy(&c, ptr + size_t{vertex + 2} * 4, 4);
                    out[written++] = c;
                    out[written++] = a;
                    out[written++] = b;
                }
                return written;
            }
        }
        for (u32 vertex = 0; vertex + 3 <= count; vertex += 3) {
            triangle(vertex, vertex + 1, vertex + 2);
        }
        return written;
    case Topology::TriangleStrip: {
        if (!provoking_last) {
            break;
        }
        // Triangle i of a strip is i, i+1, i+2 (even) or i+1, i, i+2 (odd), its last vertex i+2,
        // each at slot i. A restart index begins a new strip; the triangles across it are
        // degenerate (0, 0, 0), as the GPU rewrite writes them.
        u32 start = 0;
        for (u32 vertex = 0; vertex < count; ++vertex) {
            const bool is_restart = restart && index(vertex) == 0xFFFFFFFFU;
            if (is_restart) {
                start = vertex + 1;
            }
            if (vertex < 2) {
                continue;
            }
            const u32 first_vertex = vertex - 2;
            if (is_restart || vertex < start + 2) {
                out[written++] = 0;
                out[written++] = 0;
                out[written++] = 0;
            } else if ((first_vertex - start) % 2 == 0) {
                triangle(first_vertex, first_vertex + 1, vertex);
            } else {
                triangle(first_vertex + 1, first_vertex, vertex);
            }
        }
        return written;
    }
    case Topology::Lines:
        if (!provoking_last) {
            break;
        }
        // Lines keep no winding: each pair swapped.
        for (u32 vertex = 0; vertex + 2 <= count; vertex += 2) {
            emit(vertex + 1);
            emit(vertex);
        }
        return written;
    case Topology::LineStrip:
        if (!provoking_last) {
            break;
        }
        // The strip backwards: line i starts at its last vertex, restarts stay where they split.
        for (u32 vertex = 0; vertex < count; ++vertex) {
            emit(count - 1 - vertex);
        }
        return written;
    case Topology::Quads:
        for (u32 quad = 0; quad + 4 <= count; quad += 4) {
            triangle(quad, quad + 1, quad + 2);
            triangle(quad, quad + 2, quad + 3);
        }
        return written;
    case Topology::QuadStrip:
        // Each new pair closes a quad with the pair before it, in Vulkan's order
        // (QuadStripIndexBuffer: 0 3 1, 0 2 3).
        for (u32 pair = 0; pair + 4 <= count; pair += 2) {
            triangle(pair, pair + 3, pair + 1);
            triangle(pair, pair + 2, pair + 3);
        }
        return written;
    case Topology::TriangleFan:
    case Topology::Polygon:
        // Vulkan's fan triangle i is i+1, i+2, 0: the first convention takes i+1, the last i+2.
        for (u32 vertex = 1; vertex + 1 < count; ++vertex) {
            if (provoking_last) {
                emit(vertex + 1);
                emit(0);
                emit(vertex);
            } else {
                emit(vertex);
                emit(vertex + 1);
                emit(0);
            }
        }
        return written;
    case Topology::LineLoop:
        if (provoking_last) {
            // The closed loop backwards (0, n-1, ..., 1, 0).
            for (u32 vertex = 0; vertex <= count && count >= 2; ++vertex) {
                emit(vertex == 0 ? 0 : count - vertex);
            }
            if (count < 2) {
                for (u32 vertex = 0; vertex < count; ++vertex) {
                    emit(vertex);
                }
            }
            return written;
        }
        for (u32 vertex = 0; vertex < count; ++vertex) {
            emit(vertex);
        }
        if (count >= 2) {
            emit(0);
        }
        return written;
    // Without a geometry program Vulkan draws adjacency primitives without their adjacent
    // vertices: the line of 1 and 2 of each four, the strip of all but the first and last
    // vertex, the triangle of 0, 2 and 4 of each six, and strip triangle i of 2i, 2i + 2 and
    // 2i + 4 (2i + 2 first when i is odd, keeping the winding).
    case Topology::LinesAdjacency:
        for (u32 line = 0; line + 4 <= count; line += 4) {
            emit(line + 1);
            emit(line + 2);
        }
        return written;
    case Topology::LineStripAdjacency:
        for (u32 vertex = 1; count >= 4 && vertex + 1 < count; ++vertex) {
            emit(vertex);
        }
        return written;
    case Topology::TrianglesAdjacency:
        for (u32 primitive = 0; primitive + 6 <= count; primitive += 6) {
            emit(primitive);
            emit(primitive + 2);
            emit(primitive + 4);
        }
        return written;
    case Topology::TriangleStripAdjacency:
        for (u32 primitive = 0; count >= 6 && primitive < (count - 4) / 2; ++primitive) {
            const bool odd = primitive % 2 != 0;
            emit(odd ? 2 * primitive + 2 : 2 * primitive);
            emit(odd ? 2 * primitive : 2 * primitive + 2);
            emit(2 * primitive + 4);
        }
        return written;
    default:
        break;
    }
    if (!source.empty() && source.size() >= size_t{count} * index_size && index_size == 1) {
        const u8* ptr = source.data();
        if (restart) {
            for (u32 vertex = 0; vertex < count; ++vertex) {
                const u8 v = ptr[vertex];
                out[vertex] = v == 0xFF ? 0xFFFFFFFFU : u32{v};
            }
        } else {
            for (u32 vertex = 0; vertex < count; ++vertex) {
                out[vertex] = u32{ptr[vertex]};
            }
        }
        return count;
    }
    for (u32 vertex = 0; vertex < count; ++vertex) {
        emit(vertex);
    }
    return written;
}

u32 RewriteMode(Topology topology, bool provoking_last, bool restart) noexcept {
    provoking_last = provoking_last && RotatesForProvokingVertex(topology);
    switch (topology) {
    case Topology::Quads:
        return provoking_last ? 11 : 1;
    case Topology::QuadStrip:
        return provoking_last ? 12 : 2;
    case Topology::TriangleFan:
    case Topology::Polygon:
        return provoking_last ? 13 : 3;
    case Topology::LineLoop:
        return provoking_last ? 10 : 4;
    case Topology::Triangles:
        return provoking_last ? 5 : 0;
    case Topology::TriangleStrip:
        return !provoking_last ? 0 : restart ? StripRestartMode : 6;
    case Topology::Lines:
        return provoking_last ? 8 : 0;
    case Topology::LineStrip:
        return provoking_last ? 9 : 0;
    case Topology::LinesAdjacency:
        return 14;
    case Topology::LineStripAdjacency:
        return 15;
    case Topology::TrianglesAdjacency:
        return 16;
    case Topology::TriangleStripAdjacency:
        return 17;
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

static uint read_index(device const uchar* source, constant RewriteParams& params, uint element) {
    if (element >= params.count) {
        return 0;
    }
    if (params.index_size == 0) {
        // A plain draw: its vertices from the first (source_offset) are the indices.
        return params.source_offset + element;
    }
    const uint at = params.source_offset + element * params.index_size;
    uint value;
    switch (params.index_size) {
    case 1:
        value = source[at];
        return params.restart != 0 && value == 0xFFu ? 0xFFFFFFFFu : value;
    case 2:
        value = uint(source[at]) | (uint(source[at + 1]) << 8);
        return params.restart != 0 && value == 0xFFFFu ? 0xFFFFFFFFu : value;
    default:
        return uint(source[at]) | (uint(source[at + 1]) << 8) | (uint(source[at + 2]) << 16) |
               (uint(source[at + 3]) << 24);
    }
}

kernel void rewrite_indices(device const uchar* source [[buffer(0)]],
                            device uint* indices [[buffer(1)]],
                            constant RewriteParams& params [[buffer(2)]],
                            uint slot [[thread_position_in_grid]]) {
    if (slot >= params.converted) {
        return;
    }
    // Which index of the list is read; the slot written stays where it is.
    uint index = slot;
    uint mode = params.mode;
    // 11-13: quads, quad strips and fans with the last vertex convention. The list's triangle
    // (a, b, c) becomes (c, a, b) (fans: their own order).
    if (mode == 11 || mode == 12) {
        index = index / 3 * 3 + (index % 3 + 2) % 3;
        mode -= 10;
    }
    const uint triangle = index / 3;
    const uint corner = index % 3;
    uint element;
    switch (mode) {
    case 1:
        element = index / 6 * 4 + quad_order[index % 6];
        break;
    case 2:
        element = index / 6 * 2 + quad_strip_order[index % 6];
        break;
    case 3:
        element = corner == 2 ? 0 : triangle + 1 + corner;
        break;
    case 13:
        element = corner == 0 ? triangle + 2 : corner == 1 ? 0 : triangle + 1;
        break;
    case 4:
        element = index < params.count ? index : 0;
        break;
    case 5:
        element = triangle * 3 + (corner == 0 ? 2 : corner - 1);
        break;
    case 6:
        element = corner == 0 ? triangle + 2
                  : (triangle % 2 == 0) == (corner == 1) ? triangle
                                                         : triangle + 1;
        break;
    case 8:
        element = index ^ 1u;
        break;
    case 9:
        element = params.count - 1 - index;
        break;
    case 10:
        element = index == 0 ? 0 : params.count - index;
        break;
    // 14-17: adjacency lists and strips without their adjacent vertices.
    case 14:
        element = index / 2 * 4 + 1 + index % 2;
        break;
    case 15:
        element = index + 1;
        break;
    case 16:
        element = index / 3 * 6 + index % 3 * 2;
        break;
    case 17:
        element = corner == 2               ? 2 * triangle + 4
                  : (triangle % 2 == 0) == (corner == 0) ? 2 * triangle
                                                         : 2 * triangle + 2;
        break;
    default:
        element = index;
        break;
    }
    indices[slot] = read_index(source, params, element);
}

// A triangle strip with restarts for the last vertex convention (mode 7), in one threadgroup:
// each thread takes a run of the strip, finds where the strip last started before its run from
// the other runs' last restarts, and writes triangle i at slot i, degenerate across a restart.
kernel void rewrite_restart_strip(device const uchar* source [[buffer(0)]],
                                  device uint* indices [[buffer(1)]],
                                  constant RewriteParams& params [[buffer(2)]],
                                  uint lane [[thread_position_in_threadgroup]],
                                  uint lanes [[threads_per_threadgroup]]) {
    threadgroup uint run_starts[256];
    const uint count = params.count;
    const uint run = (count + lanes - 1) / lanes;
    const uint begin = min(lane * run, count);
    const uint end = min(begin + run, count);
    uint found = 0;
    for (uint element = begin; element < end; ++element) {
        if (read_index(source, params, element) == 0xFFFFFFFFu) {
            found = element + 1;
        }
    }
    run_starts[lane] = found;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = 0;
    for (uint other = 0; other < lane; ++other) {
        start = max(start, run_starts[other]);
    }
    for (uint element = begin; element < end; ++element) {
        const uint value = read_index(source, params, element);
        const bool is_restart = value == 0xFFFFFFFFu;
        if (is_restart) {
            start = element + 1;
        }
        if (element < 2 || (element - 2) * 3 + 3 > params.converted) {
            continue;
        }
        const uint at = (element - 2) * 3;
        const uint first = element - 2;
        if (is_restart || element < start + 2) {
            indices[at] = 0;
            indices[at + 1] = 0;
            indices[at + 2] = 0;
            continue;
        }
        const bool even = (first - start) % 2 == 0;
        indices[at] = value;
        indices[at + 1] = read_index(source, params, even ? first : first + 1);
        indices[at + 2] = read_index(source, params, even ? first + 1 : first);
    }
}
)";
}

} // namespace Metal::IndexConvert
