// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_metal/metal_index_convert.h"

namespace {

using Metal::IndexConvert::Topology;

std::vector<u32> Convert(Topology topology, std::span<const u8> source, u32 index_size,
                         u32 first, u32 count, bool restart = false) {
    std::vector<u32> out(Metal::IndexConvert::ConvertedCount(topology, count));
    out.resize(Metal::IndexConvert::Convert(topology, source, index_size, first, count, restart,
                                            out));
    return out;
}

} // Anonymous namespace

TEST_CASE("MetalIndexConvert[QuadsFromVertices]", "[video_core]") {
    REQUIRE(Metal::IndexConvert::NeedsConversion(Topology::Quads));
    REQUIRE(Metal::IndexConvert::ConvertedTopology(Topology::Quads) == Topology::Triangles);
    // Two quads from vertex 10 (Vulkan's QuadArrayIndexBuffer order 0 1 2, 0 2 3).
    REQUIRE(Convert(Topology::Quads, {}, 0, 10, 8) ==
            std::vector<u32>{10, 11, 12, 10, 12, 13, 14, 15, 16, 14, 16, 17});
    // A trailing partial quad is dropped.
    REQUIRE(Convert(Topology::Quads, {}, 0, 0, 6).size() == 6);
}

TEST_CASE("MetalIndexConvert[QuadStrip]", "[video_core]") {
    REQUIRE(Convert(Topology::QuadStrip, {}, 0, 0, 6) ==
            std::vector<u32>{0, 3, 1, 0, 2, 3, 2, 5, 3, 2, 4, 5});
    REQUIRE(Convert(Topology::QuadStrip, {}, 0, 0, 3).empty());
}

TEST_CASE("MetalIndexConvert[FanAndPolygon]", "[video_core]") {
    const std::array<u8, 5> indices{7, 8, 9, 10, 11};
    // Vulkan's fan triangle i is (i+1, i+2, 0): Metal's first vertex is then Vulkan's.
    REQUIRE(Convert(Topology::TriangleFan, indices, 1, 0, 5) ==
            std::vector<u32>{8, 9, 7, 9, 10, 7, 10, 11, 7});
    REQUIRE(Convert(Topology::Polygon, {}, 0, 3, 4) == std::vector<u32>{4, 5, 3, 5, 6, 3});
}

TEST_CASE("MetalIndexConvert[ProvokingLast]", "[video_core]") {
    using Metal::IndexConvert::ConvertedCount;
    const auto last = [](Topology topology, u32 count, bool restart = false,
                         std::span<const u8> source = {}, u32 index_size = 0) {
        std::vector<u32> out(ConvertedCount(topology, count, true));
        out.resize(Metal::IndexConvert::Convert(topology, source, index_size, 0, count, restart,
                                                out, true));
        return out;
    };
    // Each triangle starts with its last vertex and keeps its winding: (a, b, c) -> (c, a, b).
    REQUIRE(last(Topology::Triangles, 7) == std::vector<u32>{2, 0, 1, 5, 3, 4});
    REQUIRE(last(Topology::TriangleStrip, 5) == std::vector<u32>{2, 0, 1, 3, 2, 1, 4, 2, 3});
    REQUIRE(last(Topology::Quads, 4) == std::vector<u32>{2, 0, 1, 3, 0, 2});
    REQUIRE(last(Topology::QuadStrip, 4) == std::vector<u32>{1, 0, 3, 3, 0, 2});
    REQUIRE(last(Topology::TriangleFan, 4) == std::vector<u32>{2, 0, 1, 3, 0, 2});
    REQUIRE(last(Topology::Lines, 5) == std::vector<u32>{1, 0, 3, 2});
    REQUIRE(last(Topology::LineStrip, 3) == std::vector<u32>{2, 1, 0});
    REQUIRE(last(Topology::LineLoop, 3) == std::vector<u32>{0, 2, 1, 0});
    // Strips with a restart: each part on its own, the triangles across it degenerate.
    const std::array<u8, 6> strip{0, 1, 2, 0xFF, 3, 4};
    REQUIRE(last(Topology::TriangleStrip, 6, true, strip, 1) ==
            std::vector<u32>{2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    REQUIRE(Metal::IndexConvert::RewriteMode(Topology::TriangleStrip, true, true) ==
            Metal::IndexConvert::StripRestartMode);
}

TEST_CASE("MetalIndexConvert[LineLoop]", "[video_core]") {
    REQUIRE(Metal::IndexConvert::RewriteMode(Topology::LineLoop) == 4);
    REQUIRE(Metal::IndexConvert::RewriteMode(Topology::Polygon) == 3);
    REQUIRE(Metal::IndexConvert::RewriteMode(Topology::Triangles) == 0);
    REQUIRE(Metal::IndexConvert::ConvertedTopology(Topology::LineLoop) == Topology::LineStrip);
    REQUIRE(Convert(Topology::LineLoop, {}, 0, 2, 3) == std::vector<u32>{2, 3, 4, 2});
}

TEST_CASE("MetalIndexConvert[WidenedIndices]", "[video_core]") {
    REQUIRE_FALSE(Metal::IndexConvert::NeedsConversion(Topology::Triangles));
    // 8-bit indices of a strip keep their restarts as Metal's 32-bit restart value.
    const std::array<u8, 5> indices{1, 2, 0xFF, 3, 4};
    REQUIRE(Convert(Topology::TriangleStrip, indices, 1, 0, 5, true) ==
            std::vector<u32>{1, 2, 0xFFFFFFFFU, 3, 4});
    REQUIRE(Convert(Topology::TriangleStrip, indices, 1, 0, 5, false) ==
            std::vector<u32>{1, 2, 0xFF, 3, 4});
    // 16-bit indices, little endian.
    const std::array<u8, 4> shorts{0x34, 0x12, 0xFF, 0xFF};
    REQUIRE(Convert(Topology::Triangles, shorts, 2, 0, 2, true) ==
            std::vector<u32>{0x1234, 0xFFFFFFFFU});
}
