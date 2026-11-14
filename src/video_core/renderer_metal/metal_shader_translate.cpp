// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <exception>

#include <spirv_msl.hpp>

#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal::ShaderTranslate {

namespace {

[[nodiscard]] spv::ExecutionModel Model(Stage stage) {
    switch (stage) {
    case Stage::Vertex:
        return spv::ExecutionModelVertex;
    case Stage::Fragment:
        return spv::ExecutionModelFragment;
    case Stage::Compute:
        return spv::ExecutionModelGLCompute;
    }
    return spv::ExecutionModelGLCompute;
}

[[nodiscard]] u32 ArrayCount(const spirv_cross::CompilerMSL& compiler,
                           const spirv_cross::Resource& resource) {
    const spirv_cross::SPIRType& type = compiler.get_type(resource.type_id);
    u32 count = 1;
    for (const u32 size : type.array) {
        count *= (std::max)(size, 1U);
    }
    return count;
}

struct Declared {
    u32 set;
    u32 binding;
    bool combined; ///< Takes a texture and a sampler
    ResourceKind kind;
    u32 count; ///< Array elements, each a slot
};

} // Anonymous namespace

Result SpirvToMsl(std::span<const u32> spirv, Stage stage, const Options& options) {
    Result result;
    try {
        spirv_cross::CompilerMSL compiler(std::vector<u32>(spirv.begin(), spirv.end()));
        auto msl = compiler.get_msl_options();
        msl.platform = options.ios ? spirv_cross::CompilerMSL::Options::iOS
                                   : spirv_cross::CompilerMSL::Options::macOS;
        msl.set_msl_version(options.msl_major, options.msl_minor);
        msl.argument_buffers = false;
        msl.pad_fragment_output_components = true;
        msl.enable_decoration_binding = false;
        // Texel buffers as MSL texture_buffer (MSL 2.1+), not emulated with 2D textures.
        msl.texture_buffer_native = true;
        compiler.set_msl_options(msl);
        auto common = compiler.get_common_options();
        common.vertex.flip_vert_y = stage == Stage::Vertex && options.flip_vertex_y;
        compiler.set_common_options(common);

        const spv::ExecutionModel model = Model(stage);
        const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
        std::vector<Declared> declared;
        const auto add = [&](const auto& list, bool combined, ResourceKind kind) {
            for (const spirv_cross::Resource& resource : list) {
                declared.push_back(Declared{
                    .set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet),
                    .binding = compiler.get_decoration(resource.id, spv::DecorationBinding),
                    .combined = combined,
                    .kind = kind,
                    .count = ArrayCount(compiler, resource),
                });
            }
        };
        add(resources.uniform_buffers, false, ResourceKind::Buffer);
        add(resources.storage_buffers, false, ResourceKind::Buffer);
        add(resources.sampled_images, true, ResourceKind::Texture);
        add(resources.separate_images, false, ResourceKind::Texture);
        add(resources.storage_images, false, ResourceKind::Texture);
        add(resources.separate_samplers, false, ResourceKind::Sampler);
        std::ranges::sort(declared, [](const Declared& lhs, const Declared& rhs) {
            return lhs.set != rhs.set ? lhs.set < rhs.set : lhs.binding < rhs.binding;
        });

        u32 buffers = 0;
        u32 textures = 0;
        u32 samplers = 0;
        for (const Declared& resource : declared) {
            spirv_cross::MSLResourceBinding binding{};
            binding.stage = model;
            binding.desc_set = resource.set;
            binding.binding = resource.binding;
            binding.count = resource.count;
            switch (resource.kind) {
            case ResourceKind::Buffer:
                binding.msl_buffer = buffers;
                result.slots.push_back(
                    {resource.set, resource.binding, resource.kind, buffers, resource.count});
                buffers += resource.count;
                break;
            case ResourceKind::Texture:
                binding.msl_texture = textures;
                result.slots.push_back(
                    {resource.set, resource.binding, resource.kind, textures, resource.count});
                textures += resource.count;
                if (resource.combined) {
                    binding.msl_sampler = samplers;
                    result.slots.push_back({resource.set, resource.binding, ResourceKind::Sampler,
                                            samplers, resource.count});
                    samplers += resource.count;
                }
                break;
            case ResourceKind::Sampler:
                binding.msl_sampler = samplers;
                result.slots.push_back(
                    {resource.set, resource.binding, resource.kind, samplers, resource.count});
                samplers += resource.count;
                break;
            }
            compiler.add_msl_resource_binding(binding);
        }
        // The recompiler's push constants (rescaling, render area): one buffer after the rest.
        if (!resources.push_constant_buffers.empty()) {
            spirv_cross::MSLResourceBinding binding{};
            binding.stage = model;
            binding.desc_set = spirv_cross::kPushConstDescSet;
            binding.binding = spirv_cross::kPushConstBinding;
            binding.msl_buffer = buffers;
            compiler.add_msl_resource_binding(binding);
            result.slots.push_back({PushConstantSet, 0, ResourceKind::Buffer, buffers++, 1});
        }
        if (buffers > options.first_vertex_buffer_slot && stage == Stage::Vertex) {
            result.error = "too many buffers for the vertex buffer slots";
            return result;
        }

        result.source = compiler.compile();
        const auto entry_points = compiler.get_entry_points_and_stages();
        if (!entry_points.empty()) {
            result.entry_point =
                compiler.get_cleansed_entry_point_name(entry_points.front().name, model);
        }
    } catch (const std::exception& exception) {
        result.source.clear();
        result.slots.clear();
        result.error = exception.what();
    }
    return result;
}

} // namespace Metal::ShaderTranslate
