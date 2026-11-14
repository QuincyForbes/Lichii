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

struct Declared {
    u32 set;
    u32 binding;
    bool combined; ///< Takes a texture and a sampler
    ResourceKind kind;
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
        compiler.set_msl_options(msl);

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
            switch (resource.kind) {
            case ResourceKind::Buffer:
                binding.msl_buffer = buffers;
                result.slots.push_back({resource.set, resource.binding, resource.kind, buffers++});
                break;
            case ResourceKind::Texture:
                binding.msl_texture = textures;
                result.slots.push_back({resource.set, resource.binding, resource.kind, textures++});
                if (resource.combined) {
                    binding.msl_sampler = samplers;
                    result.slots.push_back(
                        {resource.set, resource.binding, ResourceKind::Sampler, samplers++});
                }
                break;
            case ResourceKind::Sampler:
                binding.msl_sampler = samplers;
                result.slots.push_back({resource.set, resource.binding, resource.kind, samplers++});
                break;
            }
            compiler.add_msl_resource_binding(binding);
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
