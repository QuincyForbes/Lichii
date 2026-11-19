// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

namespace {
using Regs = Tegra::Engines::Maxwell3D::Regs;
using Topology = Regs::PrimitiveTopology;

template <typename R>
constexpr Topology QueryRasterTopology(const R& regs, Topology draw_topology) {
    if constexpr (requires { regs.RasterTopology(draw_topology); }) {
        return regs.RasterTopology(draw_topology);
    } else {
        return Vulkan::RasterTopologyOf(regs, draw_topology);
    }
}

TEST_CASE("Mesh draws rasterize SET_MESH_SHADER_A's topology", "[video_core]") {
    const auto regs = std::make_unique<Regs>();
    REQUIRE(QueryRasterTopology(*regs, Topology::Points) == Topology::Points);
    REQUIRE(QueryRasterTopology(*regs, Topology::TriangleStrip) == Topology::TriangleStrip);

    regs->reg_array[Regs::MeshControl] = 1;
    regs->reg_array[Regs::MeshShaderA] = (64u << 13) | (64u << 4) | 4;
    REQUIRE(QueryRasterTopology(*regs, Topology::Points) == Topology::Triangles);
    regs->reg_array[Regs::MeshShaderA] = (64u << 4) | 1;
    REQUIRE(QueryRasterTopology(*regs, Topology::Points) == Topology::Lines);
    regs->reg_array[Regs::MeshShaderA] = 0;
    REQUIRE(QueryRasterTopology(*regs, Topology::Triangles) == Topology::Points);
}

} // Anonymous namespace
