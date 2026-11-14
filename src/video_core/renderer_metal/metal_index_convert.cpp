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

} // namespace Metal::IndexConvert
