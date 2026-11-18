// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <optional>
#include <unordered_map>
#include <utility>

#include <spirv_msl.hpp>

#include "video_core/renderer_metal/metal_graphics.h"
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
    case Stage::VertexForTessellation:
        return spv::ExecutionModelVertex;
    case Stage::TessellationControl:
        return spv::ExecutionModelTessellationControl;
    case Stage::TessellationEvaluation:
        return spv::ExecutionModelTessellationEvaluation;
    case Stage::Object:
        return spv::ExecutionModelTaskEXT;
    case Stage::Mesh:
        return spv::ExecutionModelMeshEXT;
    }
    return spv::ExecutionModelGLCompute;
}

/// The builtins a tessellation stage passes on in its buffer (tessellation levels go to the
/// factor buffer instead).
[[nodiscard]] bool PassedBuiltin(spv::BuiltIn builtin) {
    switch (builtin) {
    case spv::BuiltInPosition:
    case spv::BuiltInPointSize:
    case spv::BuiltInClipDistance:
    case spv::BuiltInCullDistance:
    case spv::BuiltInLayer:
    case spv::BuiltInViewportIndex:
        return true;
    default:
        return false;
    }
}

/// The member of an output struct the MSL gives a built-in (SPIRV-Cross's builtin_to_glsl name).
[[nodiscard]] std::string BuiltinMember(spv::BuiltIn builtin) {
    switch (builtin) {
    case spv::BuiltInPosition:
        return "gl_Position";
    case spv::BuiltInPointSize:
        return "gl_PointSize";
    case spv::BuiltInClipDistance:
        return "gl_ClipDistance";
    case spv::BuiltInCullDistance:
        return "gl_CullDistance";
    case spv::BuiltInLayer:
        return "gl_Layer";
    case spv::BuiltInViewportIndex:
        return "gl_ViewportIndex";
    default:
        return {};
    }
}

/// What a vertex or tessellation control stage writes for the next stage.
[[nodiscard]] std::vector<InterfaceVariable> StageOutputs(spirv_cross::CompilerMSL& compiler,
                                                          const spirv_cross::ShaderResources& resources) {
    std::vector<InterfaceVariable> outputs;
    for (const spirv_cross::Resource& resource : resources.stage_outputs) {
        const spirv_cross::SPIRType& type = compiler.get_type(resource.base_type_id);
        outputs.push_back(InterfaceVariable{
            .location = compiler.get_decoration(resource.id, spv::DecorationLocation),
            .vecsize = (std::max)(type.vecsize, 1U),
            .per_patch = compiler.has_decoration(resource.id, spv::DecorationPatch),
            .name = resource.name,
        });
    }
    for (const spirv_cross::BuiltInResource& builtin : resources.builtin_outputs) {
        if (!PassedBuiltin(builtin.builtin) ||
            !compiler.has_active_builtin(builtin.builtin, spv::StorageClassOutput)) {
            continue;
        }
        const spirv_cross::SPIRType& type = compiler.get_type(builtin.value_type_id);
        u32 elements = 1;
        for (const u32 size : type.array) {
            elements *= (std::max)(size, 1U);
        }
        outputs.push_back(InterfaceVariable{
            .builtin = static_cast<u32>(builtin.builtin),
            .vecsize = (std::max)(type.vecsize, 1U) * elements,
            .per_patch = compiler.has_decoration(builtin.resource.id, spv::DecorationPatch),
            .name = BuiltinMember(builtin.builtin),
        });
    }
    return outputs;
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

/// Sets the Depth operand of the image types of the sampled images at `bindings` (set, binding),
/// through their variable, pointer and sampled image types; arrays of them are left as they are.
void MarkDepthImages(std::vector<u32>& words, std::span<const std::pair<u32, u32>> bindings) {
    constexpr size_t HeaderWords = 5;
    std::unordered_map<u32, u32> sets;
    std::unordered_map<u32, u32> binding_of;
    std::unordered_map<u32, u32> variable_type;
    std::unordered_map<u32, u32> pointee;
    std::unordered_map<u32, u32> sampled_image;
    std::unordered_map<u32, size_t> image_at;
    for (size_t at = HeaderWords; at < words.size();) {
        const u32 count = words[at] >> 16;
        const u32 opcode = words[at] & 0xFFFF;
        if (count == 0 || at + count > words.size()) {
            return;
        }
        if (opcode == spv::OpDecorate && count >= 4) {
            if (words[at + 2] == spv::DecorationDescriptorSet) {
                sets[words[at + 1]] = words[at + 3];
            } else if (words[at + 2] == spv::DecorationBinding) {
                binding_of[words[at + 1]] = words[at + 3];
            }
        } else if (opcode == spv::OpVariable && count >= 4) {
            variable_type[words[at + 2]] = words[at + 1];
        } else if (opcode == spv::OpTypePointer && count >= 4) {
            pointee[words[at + 1]] = words[at + 3];
        } else if (opcode == spv::OpTypeSampledImage && count >= 3) {
            sampled_image[words[at + 1]] = words[at + 2];
        } else if (opcode == spv::OpTypeImage && count >= 9) {
            image_at[words[at + 1]] = at;
        }
        at += count;
    }
    const auto find = [](const std::unordered_map<u32, u32>& map, u32 key) -> std::optional<u32> {
        const auto it = map.find(key);
        return it != map.end() ? std::optional{it->second} : std::nullopt;
    };
    for (const auto& [variable, binding] : binding_of) {
        const u32 set = find(sets, variable).value_or(0);
        if (std::ranges::find(bindings, std::pair{set, binding}) == bindings.end()) {
            continue;
        }
        std::optional<u32> type = find(variable_type, variable);
        type = type ? find(pointee, *type) : std::nullopt;
        if (type) {
            // A combined image sampler, or a separate image.
            type = find(sampled_image, *type).value_or(*type);
        }
        if (const auto image = type ? image_at.find(*type) : image_at.end();
            image != image_at.end()) {
            words[image->second + 4] = 1;
        }
    }
}

struct Declared {
    u32 set;
    u32 binding;
    bool combined; ///< Takes a texture and a sampler
    ResourceKind kind;
    u32 count; ///< Array elements, each a slot
};

} // Anonymous namespace

bool DeviceMslEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MSL_DEVICE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

Result SpirvToMsl(std::span<const u32> spirv, Stage stage, const Options& options) {
    Result result;
    try {
        std::vector<u32> words(spirv.begin(), spirv.end());
        if (!options.depth_images.empty()) {
            MarkDepthImages(words, options.depth_images);
        }
        spirv_cross::CompilerMSL compiler(std::move(words));
        auto msl = compiler.get_msl_options();
        msl.platform = options.ios ? spirv_cross::CompilerMSL::Options::iOS
                                   : spirv_cross::CompilerMSL::Options::macOS;
        msl.set_msl_version(options.msl_major, options.msl_minor);
        msl.argument_buffers = false;
        msl.pad_fragment_output_components = true;
        msl.enable_decoration_binding = false;
        // Texel buffers as MSL texture_buffer (MSL 2.1+), not emulated with 2D textures.
        msl.texture_buffer_native = true;
        // Metal's 1D textures take no levels and no views as 2D ones: with the pipeline check
        // the texture cache keeps 1D images as 2D ones a texel high, as MoltenVK does.
        msl.texture_1D_as_2D = PipelineCheckEnabled();
        if (options.apple_family != 0 && DeviceMslEnabled()) {
            // What MoltenVK tells SPIRV-Cross about Apple GPUs. Without the first an iOS program
            // that reads the base instance (every one that reads the instance index) does not
            // translate; without the second its subgroups are quads of four.
            msl.ios_support_base_vertex_instance = options.apple_family >= 3;
            msl.ios_use_simdgroup_functions = options.apple_family >= 6;
            msl.agx_manual_cube_grad_fixup = true;
            msl.check_discarded_frag_stores = stage == Stage::Fragment;
            msl.replace_recursive_inputs = true;
        }
        const bool tess_vertex = stage == Stage::VertexForTessellation;
        const bool tess_control = stage == Stage::TessellationControl;
        const bool tess_evaluation = stage == Stage::TessellationEvaluation;
        msl.vertex_for_tessellation = tess_vertex;
        msl.capture_output_to_buffer = tess_vertex || tess_control;
        msl.multi_patch_workgroup = tess_vertex || tess_control || tess_evaluation;
        msl.raw_buffer_tese_input = tess_evaluation;
        switch (options.vertex_index_size) {
        case 2:
            msl.vertex_index_type = spirv_cross::CompilerMSL::Options::IndexType::UInt16;
            break;
        case 4:
            msl.vertex_index_type = spirv_cross::CompilerMSL::Options::IndexType::UInt32;
            break;
        default:
            break;
        }
        compiler.set_msl_options(msl);
        auto common = compiler.get_common_options();
        common.vertex.flip_vert_y =
            (stage == Stage::Vertex || stage == Stage::Mesh || tess_evaluation) &&
            options.flip_vertex_y;
        compiler.set_common_options(common);
        if (tess_control || tess_evaluation) {
            // Each program declares part of these (the domain, spacing and winding are the
            // evaluation program's, the control points the control program's).
            using Primitive = TessellationModes::Primitive;
            using Spacing = TessellationModes::Spacing;
            const TessellationModes& modes = options.tessellation;
            compiler.set_execution_mode(modes.primitive == Primitive::Quads ? spv::ExecutionModeQuads
                                        : modes.primitive == Primitive::Isolines
                                            ? spv::ExecutionModeIsolines
                                            : spv::ExecutionModeTriangles);
            compiler.set_execution_mode(modes.spacing == Spacing::FractionalEven
                                            ? spv::ExecutionModeSpacingFractionalEven
                                        : modes.spacing == Spacing::FractionalOdd
                                            ? spv::ExecutionModeSpacingFractionalOdd
                                            : spv::ExecutionModeSpacingEqual);
            compiler.set_execution_mode(modes.clockwise ? spv::ExecutionModeVertexOrderCw
                                                        : spv::ExecutionModeVertexOrderCcw);
            if (modes.point_mode) {
                compiler.set_execution_mode(spv::ExecutionModePointMode);
            }
            if (modes.output_vertices != 0) {
                compiler.set_execution_mode(spv::ExecutionModeOutputVertices,
                                            modes.output_vertices);
            }
        }
        for (const InterfaceVariable& input : options.inputs) {
            spirv_cross::MSLShaderInterfaceVariable variable{};
            variable.location = input.location;
            variable.format = spirv_cross::MSL_SHADER_VARIABLE_FORMAT_ANY32;
            variable.builtin = input.builtin == InterfaceVariable::NoBuiltin
                                   ? spv::BuiltInMax
                                   : static_cast<spv::BuiltIn>(input.builtin);
            variable.vecsize = input.vecsize;
            variable.rate = input.per_patch ? spirv_cross::MSL_SHADER_VARIABLE_RATE_PER_PATCH
                                            : spirv_cross::MSL_SHADER_VARIABLE_RATE_PER_VERTEX;
            compiler.add_msl_shader_input(variable);
        }

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
        // The tessellation stages' own buffers, after their resources (the vertex kernel's
        // vertex buffers take slots from first_vertex_buffer_slot down, as a vertex stage's).
        TessellationSlots& tess_slots = result.tessellation_slots;
        if (tess_vertex) {
            tess_slots.output = buffers++;
            tess_slots.index = buffers++;
            msl.shader_output_buffer_index = tess_slots.output;
            msl.shader_index_buffer_index = tess_slots.index;
        } else if (tess_control) {
            tess_slots.input = buffers++;
            tess_slots.output = buffers++;
            tess_slots.patch_output = buffers++;
            tess_slots.tess_factors = buffers++;
            tess_slots.indirect_params = buffers++;
            msl.shader_input_buffer_index = tess_slots.input;
            msl.shader_output_buffer_index = tess_slots.output;
            msl.shader_patch_output_buffer_index = tess_slots.patch_output;
            msl.shader_tess_factor_buffer_index = tess_slots.tess_factors;
            msl.indirect_params_buffer_index = tess_slots.indirect_params;
        } else if (tess_evaluation) {
            tess_slots.input = buffers++;
            tess_slots.patch_input = buffers++;
            msl.shader_input_buffer_index = tess_slots.input;
            msl.shader_patch_input_buffer_index = tess_slots.patch_input;
        }
        compiler.set_msl_options(msl);
        if (buffers > options.first_vertex_buffer_slot &&
            (stage == Stage::Vertex || tess_vertex)) {
            result.error = "too many buffers for the vertex buffer slots";
            return result;
        }

        result.source = compiler.compile();
        if (tess_vertex || tess_control) {
            result.outputs = StageOutputs(compiler, resources);
        }
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
