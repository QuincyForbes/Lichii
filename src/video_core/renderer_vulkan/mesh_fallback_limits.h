// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

namespace Vulkan {

// The mesh fallback adds storage-buffer bindings for its output (mesh-only), or
// output, task payload, offsets and indirect arguments (task + mesh). Check
// against the final descriptor set limit without overflowing guest + auxiliary.
[[nodiscard]] constexpr bool FitsStorageBufferSetLimit(std::size_t guest_buffers,
                                                       std::size_t auxiliary_buffers,
                                                       std::size_t set_limit) noexcept {
    return set_limit == 0 ||
           (guest_buffers <= set_limit && auxiliary_buffers <= set_limit - guest_buffers);
}

struct NativeMeshShape {
    std::uint32_t max_vertices;
    std::uint32_t max_primitives;
    std::uint32_t primitive_vertices; ///< 1, 2 or 3
    std::uint32_t vertex_lines;       ///< Scalar output attributes per vertex
    std::uint32_t primitive_lines;
    std::uint32_t vertex_locations; ///< vec4 outputs per vertex (generics, then built-ins)
    std::uint32_t primitive_locations;
    std::uint32_t shared_memory_bytes; ///< The program's own workgroup memory
    bool direct_outputs{};             ///< RuntimeInfo::mesh_direct_outputs: no attribute region
};

// Threadgroup memory of a native mesh program on MoltenVK: the attribute and map regions of
// emit_spirv_mesh.cpp and the program's shared memory, plus what SPIRV-Cross stages for the Metal
// mesh object (outputs, triangle indices as 16-byte uint3 and the two counts).
[[nodiscard]] constexpr std::uint64_t MetalMeshThreadgroupBytes(const NativeMeshShape& s) noexcept {
    const auto groups = [](std::uint64_t count) { return (count + 31) / 32; };
    const std::uint64_t attribute_words{
        s.direct_outputs ? 0
                         : groups(s.max_vertices) * s.vertex_lines * 32 +
                               groups(s.max_primitives) * s.primitive_lines * 32};
    const std::uint64_t map_words{4 + std::uint64_t{s.primitive_vertices} * s.max_primitives};
    const std::uint64_t workgroup{s.shared_memory_bytes + (attribute_words + map_words + 1) * 4};
    const std::uint64_t outputs{std::uint64_t{s.max_vertices} * s.vertex_locations * 16 +
                                std::uint64_t{s.max_primitives} * (s.primitive_locations + 1) * 16 +
                                8};
    return workgroup + outputs;
}

} // namespace Vulkan
