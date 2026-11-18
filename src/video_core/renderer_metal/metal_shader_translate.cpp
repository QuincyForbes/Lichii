// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <spirv_msl.hpp>

#include "common/large_stack.h"

#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal::ShaderTranslate {

namespace {

/// Routes the 2D samples of an entry function through spvReducedSample (see
/// Options::sampler_reduction); the buffer slot of the mask, or ~0 when nothing was rewritten.
/// Moves the entry function's spilled buffers (MSL buffer indices from SpillBase) to reads
/// through a table of addresses at `table_slot`; false when a parameter is not understood.
[[nodiscard]] bool SpillBuffers(std::string& source, const std::string& entry_point, u32 spilled,
                                u32 table_slot);
constexpr u32 SpillBase = 1000;
/// Moves the entry function's spilled samplers (MSL sampler indices from SamplerSpillBase) to
/// reads from a table of their resource IDs at `table_slot` (Metal 3 argument buffer).
[[nodiscard]] bool SpillSamplers(std::string& source, const std::string& entry_point,
                                 u32 spilled, u32 table_slot);
constexpr u32 SamplerSpillBase = 2000;
/// Spilled texture and sampler arrays: MSL indices from ArraySpillBase + their first table entry.
[[nodiscard]] bool SpillArrays(std::string& source, const std::string& entry_point, u32 table_slot);
constexpr u32 ArraySpillBase = 100000;
/// Metal's texture slots per stage (0-127).
constexpr u32 MetalTextureSlots = 128;
/// Metal's sampler slots per stage (0-15).
constexpr u32 MetalSamplerSlots = 16;

[[nodiscard]] u32 RewriteSamplerReduction(std::string& source, const std::string& entry_point,
                                          bool fragment, u32 buffer_slot);

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

bool SamplerReductionEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SAMPLER_REDUCTION");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

bool DeviceMslEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MSL_DEVICE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

namespace {
Result SpirvToMslOnThisStack(std::span<const u32> spirv, Stage stage, const Options& options);

/// The deepest nesting of (), [] and {} in `source` (what Metal's parser limits to 256).
[[nodiscard]] u32 MaxBracketNesting(std::string_view source) {
    u32 depth = 0;
    u32 deepest = 0;
    for (const char c : source) {
        if (c == '(' || c == '[' || c == '{') {
            deepest = (std::max)(deepest, ++depth);
        } else if ((c == ')' || c == ']' || c == '}') && depth != 0) {
            --depth;
        }
    }
    return deepest;
}
} // Anonymous namespace

Result SpirvToMsl(std::span<const u32> spirv, Stage stage, const Options& options) {
    // SPIRV-Cross recurses through a program's control flow: large game shaders overflowed the
    // 512 KiB stack of the worker threads.
    Result result;
    Common::RunOnLargeStack([&] {
        result = SpirvToMslOnThisStack(spirv, stage, options);
        // SPIRV-Cross inlines single-use expressions, and a long chain of them nests one call in
        // the next: Metal's parser stops at 256 brackets ("bracket nesting level exceeded"), and
        // the program was refused. Such a program is translated again with temporaries.
        constexpr u32 SafeNesting = 192;
        if (!options.force_temporary && !result.source.empty() &&
            MaxBracketNesting(result.source) > SafeNesting) {
            Options flat = options;
            flat.force_temporary = true;
            result = SpirvToMslOnThisStack(spirv, stage, flat);
        }
    });
    return result;
}

namespace {
Result SpirvToMslOnThisStack(std::span<const u32> spirv, Stage stage, const Options& options) {
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
            // The rest of what MoltenVK sets (MVKPipeline.mm): every Apple GPU has SIMD groups
            // of 32 threads, and before iOS 17 (MSL 3.0) a shadow sample with a LOD in an array
            // reads a biased LOD, so it goes through gradients.
            msl.fixed_subgroup_size = 32;
            msl.sample_dref_lod_array_as_grad = options.ios && options.msl_major < 3;
        }
        msl.enable_point_size_builtin = options.point_size_builtin && !options.drop_point_size;
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
        common.force_temporary = common.force_temporary || options.force_temporary;
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
            // SPIRV-Cross throws on isolines: they are translated as quads (control factors in
            // the quad layout, evaluation of a (u, v) point), and EmulateDomain draws the lines.
            compiler.unset_execution_mode(spv::ExecutionModeIsolines);
            compiler.set_execution_mode(modes.primitive == Primitive::Triangles
                                            ? spv::ExecutionModeTriangles
                                            : spv::ExecutionModeQuads);
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
        u32 spilled = 0;
        u32 total_buffers = 0;
        for (const Declared& resource : declared) {
            total_buffers += resource.kind == ResourceKind::Buffer ? resource.count : 0;
        }
        // Metal has 31 buffer slots (0-30). A vertex program spills early to leave slots for the
        // vertex streams. Any other program spills once its buffers plus the slots kept after
        // them (push constants, the spill table, tessellation buffers) would not fit: a compute
        // program with more buffers failed to build ("'buffer' attribute parameter is out of
        // bounds"), its dispatch was skipped, and the command list it writes was read as garbage.
        constexpr u32 MetalBufferSlots = 31;
        constexpr u32 ReservedSlots = 8;
        // A program with more samplers than Metal's 16 slots keeps the first 16 there and reads
        // the others from a table of their resource IDs (an argument buffer, one more buffer
        // slot). Without, its MSL did not build ("'sampler' attribute parameter is out of bounds")
        // and its draws were skipped: Ocarina's lighting passes sample more than 16 textures.
        u32 textures = 0;
        u32 samplers = 0;
        u32 total_samplers = 0;
        for (const Declared& resource : declared) {
            const bool takes_sampler = resource.kind == ResourceKind::Sampler ||
                                       (resource.kind == ResourceKind::Texture && resource.combined);
            total_samplers += takes_sampler ? resource.count : 0;
        }
        u32 total_textures = 0;
        for (const Declared& resource : declared) {
            total_textures += resource.kind == ResourceKind::Texture ? resource.count : 0;
        }
        // Arrays (bindless tables, Ocarina's volume lookup reads 1020 3D textures) go to a table
        // of resource IDs when the program has more than Metal's slots of their kind.
        const bool spill_texture_arrays = total_textures > MetalTextureSlots;
        const bool spill_sampler_arrays = total_samplers > MetalSamplerSlots;
        u32 array_entries = 0;
        const auto spill_array = [&](const Declared& resource, ResourceKind kind) {
            const u32 first = array_entries;
            array_entries += resource.count;
            result.slots.push_back(
                {resource.set, resource.binding, kind, 0, resource.count, first, true});
            return ArraySpillBase + first;
        };
        const bool spill_samplers = total_samplers > MetalSamplerSlots;
        u32 spilled_samplers = 0;
        const auto assign_sampler = [&](spirv_cross::MSLResourceBinding& binding,
                                        const Declared& resource) {
            if (spill_sampler_arrays && resource.count > 1) {
                binding.msl_sampler = spill_array(resource, ResourceKind::Sampler);
                return;
            }
            if (spill_samplers && resource.count == 1 && samplers >= MetalSamplerSlots) {
                binding.msl_sampler = SamplerSpillBase + spilled_samplers;
                result.slots.push_back({resource.set, resource.binding, ResourceKind::Sampler, 0,
                                        1, spilled_samplers});
                ++spilled_samplers;
                return;
            }
            binding.msl_sampler = samplers;
            result.slots.push_back(
                {resource.set, resource.binding, ResourceKind::Sampler, samplers, resource.count});
            samplers += resource.count;
        };
        const bool vertex = stage == Stage::Vertex;
        const bool spill =
            options.spill_buffers && (vertex ? total_buffers > SpillThreshold
                                             : total_buffers + ReservedSlots > MetalBufferSlots);
        const u32 spill_cutoff = vertex ? SpillCutoff : MetalBufferSlots - ReservedSlots;
        for (const Declared& resource : declared) {
            spirv_cross::MSLResourceBinding binding{};
            binding.stage = model;
            binding.desc_set = resource.set;
            binding.binding = resource.binding;
            binding.count = resource.count;
            switch (resource.kind) {
            case ResourceKind::Buffer:
                if (spill && resource.count == 1 && buffers >= spill_cutoff) {
                    binding.msl_buffer = SpillBase + spilled;
                    result.slots.push_back(
                        {resource.set, resource.binding, resource.kind, 0, 1, spilled});
                    ++spilled;
                    break;
                }
                binding.msl_buffer = buffers;
                result.slots.push_back(
                    {resource.set, resource.binding, resource.kind, buffers, resource.count});
                buffers += resource.count;
                break;
            case ResourceKind::Texture:
                if (spill_texture_arrays && resource.count > 1) {
                    binding.msl_texture = spill_array(resource, ResourceKind::Texture);
                } else {
                    binding.msl_texture = textures;
                    result.slots.push_back(
                        {resource.set, resource.binding, resource.kind, textures, resource.count});
                    textures += resource.count;
                }
                if (resource.combined) {
                    assign_sampler(binding, resource);
                }
                break;
            case ResourceKind::Sampler:
                assign_sampler(binding, resource);
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
        if (spilled != 0) {
            result.spill_slot = buffers++;
        }
        if (spilled_samplers != 0) {
            result.sampler_spill_slot = buffers++;
        }
        if (array_entries != 0) {
            result.array_spill_slot = buffers++;
            result.array_spill_entries = array_entries;
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
        if (stage == Stage::Object) {
            // SPIRV-Cross gives an object function with no stage outputs an empty main0_out
            // return type, and Metal accepts only void ("invalid return type 'main0_out' for
            // object function"): every task-shader pipeline fell back to the compute emulation.
            constexpr std::string_view empty_output = "struct main0_out\n{\n};";
            if (result.source.find(empty_output) != std::string::npos) {
                const auto replace_all = [&](std::string_view from, std::string_view to) {
                    for (size_t at = result.source.find(from); at != std::string::npos;
                         at = result.source.find(from, at + to.size())) {
                        result.source.replace(at, from.size(), to);
                    }
                };
                replace_all("[[object]] main0_out ", "[[object]] void ");
                replace_all("    main0_out out = {};\n", "");
                replace_all("return out;", "return;");
            }
        }
        if (spilled != 0) {
            const auto points = compiler.get_entry_points_and_stages();
            if (points.empty() ||
                !SpillBuffers(result.source,
                              compiler.get_cleansed_entry_point_name(points.front().name, model),
                              spilled, result.spill_slot)) {
                result.source.clear();
                result.error = "buffers spilled to a table not understood";
                return result;
            }
        }
        if (array_entries != 0) {
            const auto points = compiler.get_entry_points_and_stages();
            if (points.empty() ||
                !SpillArrays(result.source,
                             compiler.get_cleansed_entry_point_name(points.front().name, model),
                             result.array_spill_slot)) {
                result.source.clear();
                result.error = "texture or sampler arrays spilled to a table not understood";
                return result;
            }
        }
        if (spilled_samplers != 0) {
            const auto points = compiler.get_entry_points_and_stages();
            if (points.empty() ||
                !SpillSamplers(result.source,
                               compiler.get_cleansed_entry_point_name(points.front().name, model),
                               spilled_samplers, result.sampler_spill_slot)) {
                result.source.clear();
                result.error = "samplers spilled to a table not understood";
                return result;
            }
        }
        const bool reducible = stage == Stage::Vertex || stage == Stage::Fragment ||
                               stage == Stage::Compute;
        // The reduction rewrite reads the entry's [[sampler(n)]] parameters: not with spills.
        if (reducible && options.sampler_reduction && !resources.sampled_images.empty() &&
            spilled_samplers == 0 && array_entries == 0 &&
            !(stage == Stage::Vertex && buffers + 1 > options.first_vertex_buffer_slot)) {
            const auto points = compiler.get_entry_points_and_stages();
            if (!points.empty()) {
                std::string plain = result.source;
                result.reduction_slot = RewriteSamplerReduction(
                    result.source, compiler.get_cleansed_entry_point_name(points.front().name, model),
                    stage == Stage::Fragment, buffers);
                if (result.reduction_slot != ~0U) {
                    result.source_without_reduction = std::move(plain);
                }
            }
        }
        if (tess_vertex || tess_control || tess_evaluation) {
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
} // Anonymous namespace

namespace {

/// The domain points of a patch from its factors, and the kernels around them. The spacing
/// splits an edge as the OpenGL and Vulkan specifications describe (equal segments; odd and even
/// spacing: the level rounded up to an odd or an even count, shorter segments at both ends); the
/// points of a triangle or quad patch are its boundary (each edge by its outer level) and the
/// interior grid of its inner levels, each point once.
constexpr std::string_view DomainGenerator = R"msl(
#define SPV_TRIANGLES 0
#define SPV_QUADS 1
#define SPV_ISOLINES 2

static uint spvSegments(float level) {
#if SPV_SPACING == 1
    level = clamp(level, 1.0f, 63.0f);
    const uint n = uint(ceil(level));
    return (n & 1u) != 0u ? n : n + 1u;
#elif SPV_SPACING == 2
    level = clamp(level, 2.0f, 64.0f);
    const uint n = uint(ceil(level));
    return (n & 1u) == 0u ? n : n + 1u;
#else
    return uint(ceil(clamp(level, 1.0f, 64.0f)));
#endif
}

static float spvSplit(uint i, uint n, float level) {
#if SPV_SPACING == 0
    return float(i) / float(n);
#else
    if (n <= 1u) {
        return float(i);
    }
    level = clamp(level, SPV_SPACING == 1 ? 1.0f : 2.0f, SPV_SPACING == 1 ? 63.0f : 64.0f);
    if (i == 0u) {
        return 0.0f;
    }
    if (i >= n) {
        return 1.0f;
    }
    const float end = (level - float(n - 2u)) * 0.5f;
    return (end + float(i - 1u)) / level;
#endif
}

static uint spvPatchPoints(spvFactors_t f) {
#if SPV_DOMAIN == SPV_TRIANGLES
    const float e0 = float(f.edgeTessellationFactor[0]);
    const float e1 = float(f.edgeTessellationFactor[1]);
    const float e2 = float(f.edgeTessellationFactor[2]);
    if (!(e0 > 0.0f && e1 > 0.0f && e2 > 0.0f)) {
        return 0u;
    }
    const uint ni = spvSegments(float(f.insideTessellationFactor));
    return spvSegments(e0) + spvSegments(e1) + spvSegments(e2) +
           (ni >= 3u ? (ni - 1u) * (ni - 2u) / 2u : 0u);
#elif SPV_DOMAIN == SPV_QUADS
    const float e0 = float(f.edgeTessellationFactor[0]);
    const float e1 = float(f.edgeTessellationFactor[1]);
    const float e2 = float(f.edgeTessellationFactor[2]);
    const float e3 = float(f.edgeTessellationFactor[3]);
    if (!(e0 > 0.0f && e1 > 0.0f && e2 > 0.0f && e3 > 0.0f)) {
        return 0u;
    }
    const uint nu = spvSegments(float(f.insideTessellationFactor[0]));
    const uint nv = spvSegments(float(f.insideTessellationFactor[1]));
    return spvSegments(e0) + spvSegments(e1) + spvSegments(e2) + spvSegments(e3) +
           (nu - 1u) * (nv - 1u);
#else
    const float lines = float(f.edgeTessellationFactor[0]);
    const float detail = float(f.edgeTessellationFactor[1]);
    if (!(lines > 0.0f && detail > 0.0f)) {
        return 0u;
    }
    const uint line_count = uint(ceil(clamp(lines, 1.0f, 64.0f)));
    const uint segments = spvSegments(detail);
#if SPV_LINES
    return line_count * segments * 2u;
#else
    return line_count * (segments + 1u);
#endif
#endif
}

static float3 spvDomainPoint(spvFactors_t f, uint k) {
#if SPV_DOMAIN == SPV_TRIANGLES
    const uint n0 = spvSegments(float(f.edgeTessellationFactor[0]));
    const uint n1 = spvSegments(float(f.edgeTessellationFactor[1]));
    const uint n2 = spvSegments(float(f.edgeTessellationFactor[2]));
    if (k < n0) {
        const float t = spvSplit(k, n0, float(f.edgeTessellationFactor[0]));
        return float3(0.0f, 1.0f - t, t);
    }
    k -= n0;
    if (k < n1) {
        const float t = spvSplit(k, n1, float(f.edgeTessellationFactor[1]));
        return float3(t, 0.0f, 1.0f - t);
    }
    k -= n1;
    if (k < n2) {
        const float t = spvSplit(k, n2, float(f.edgeTessellationFactor[2]));
        return float3(1.0f - t, t, 0.0f);
    }
    k -= n2;
    const uint ni = spvSegments(float(f.insideTessellationFactor));
    uint i = 1u;
    while (i + 2u < ni && k >= ni - 1u - i) {
        k -= ni - 1u - i;
        ++i;
    }
    const float a = float(i) / float(ni);
    const float b = float(1u + k) / float(ni);
    return float3(a, b, 1.0f - a - b);
#elif SPV_DOMAIN == SPV_QUADS
    const uint n0 = spvSegments(float(f.edgeTessellationFactor[0]));
    const uint n1 = spvSegments(float(f.edgeTessellationFactor[1]));
    const uint n2 = spvSegments(float(f.edgeTessellationFactor[2]));
    const uint n3 = spvSegments(float(f.edgeTessellationFactor[3]));
    if (k < n1) {
        return float3(spvSplit(k, n1, float(f.edgeTessellationFactor[1])), 0.0f, 0.0f);
    }
    k -= n1;
    if (k < n2) {
        return float3(1.0f, spvSplit(k, n2, float(f.edgeTessellationFactor[2])), 0.0f);
    }
    k -= n2;
    if (k < n3) {
        return float3(1.0f - spvSplit(k, n3, float(f.edgeTessellationFactor[3])), 1.0f, 0.0f);
    }
    k -= n3;
    if (k < n0) {
        return float3(0.0f, 1.0f - spvSplit(k, n0, float(f.edgeTessellationFactor[0])), 0.0f);
    }
    k -= n0;
    const float in_u = float(f.insideTessellationFactor[0]);
    const float in_v = float(f.insideTessellationFactor[1]);
    const uint nu = spvSegments(in_u);
    const uint nv = spvSegments(in_v);
    return float3(spvSplit(1u + k % (nu - 1u), nu, in_u), spvSplit(1u + k / (nu - 1u), nv, in_v),
                  0.0f);
#else
    const float lines = float(f.edgeTessellationFactor[0]);
    const float detail = float(f.edgeTessellationFactor[1]);
    const uint line_count = uint(ceil(clamp(lines, 1.0f, 64.0f)));
    const uint segments = spvSegments(detail);
#if SPV_LINES
    const uint segment = k >> 1u;
    const uint line = segment / segments;
    const uint s = segment % segments + (k & 1u);
#else
    const uint line = k / (segments + 1u);
    const uint s = k % (segments + 1u);
#endif
    return float3(spvSplit(s, segments, detail), float(line) / float(line_count), 0.0f);
#endif
}

#if SPV_TRIANGLE_LIST
// The patch as a list of triangles (what a geometry program reads), from the same points: the
// loop of the outer edges' points and the loop of the inside lattice's outermost nodes, side by
// side (the sides of the two loops correspond, corner to corner), the strip between each pair of
// sides stitched by walking both chains in step, and the inside lattice's own triangles. Triangle
// t of the patch is found without walking: the side by the counts, the strip's events by a
// search, the lattice's by arithmetic.
#if SPV_DOMAIN == SPV_TRIANGLES
#define SPV_SIDES 3u
#else
#define SPV_SIDES 4u
#endif

// Segments of the outer loop's side s (in the order spvDomainPoint walks it).
static uint spvSideOuter(spvFactors_t f, uint s) {
#if SPV_DOMAIN == SPV_TRIANGLES
    return spvSegments(float(f.edgeTessellationFactor[s]));
#else
    const uint edges[4] = {1u, 2u, 3u, 0u};
    return spvSegments(float(f.edgeTessellationFactor[edges[s]]));
#endif
}

static uint spvOuterCount(spvFactors_t f) {
    uint count = 0u;
    for (uint s = 0u; s < SPV_SIDES; ++s) {
        count += spvSideOuter(f, s);
    }
    return count;
}

#if SPV_DOMAIN == SPV_TRIANGLES
static uint spvInnerSegments(spvFactors_t f) {
    return spvSegments(float(f.insideTessellationFactor));
}

// Segments of the inner loop's sides: the lattice's outermost ring, a single node (or the centre)
// when the lattice has one or none.
static uint spvSideInner(spvFactors_t f, uint s) {
    const uint n = spvInnerSegments(f);
    return n >= 3u ? n - 3u : 0u;
}

static uint spvInteriorTriangles(spvFactors_t f) {
    const uint n = spvInnerSegments(f);
    return n >= 4u ? (n - 3u) * (n - 3u) : 0u;
}

static float3 spvLattice(uint n, uint a, uint b) {
    return float3(float(a), float(b), float(n - a - b)) / float(n);
}

static float3 spvSideInnerNode(spvFactors_t f, uint s, uint t) {
    const uint n = spvInnerSegments(f);
    if (n < 3u) {
        return float3(1.0f / 3.0f);
    }
    if (s == 0u) {
        return spvLattice(n, 1u, n - 2u - t);
    }
    if (s == 1u) {
        return spvLattice(n, 1u + t, 1u);
    }
    return spvLattice(n, n - 2u - t, 1u + t);
}

static void spvInterior(spvFactors_t f, uint k, thread float3* c) {
    const uint n = spvInnerSegments(f);
    const uint s = n - 3u;
    uint row = 0u;
    while (row + 1u < s && k >= 2u * (s - row) - 1u) {
        k -= 2u * (s - row) - 1u;
        ++row;
    }
    const uint b = 1u + row;
    if ((k & 1u) == 0u) {
        const uint a = 1u + k / 2u;
        c[0] = spvLattice(n, a, b);
        c[1] = spvLattice(n, a + 1u, b);
        c[2] = spvLattice(n, a, b + 1u);
    } else {
        const uint a = 1u + k / 2u;
        c[0] = spvLattice(n, a + 1u, b);
        c[1] = spvLattice(n, a + 1u, b + 1u);
        c[2] = spvLattice(n, a, b + 1u);
    }
}
#else
static float3 spvQuadNode(spvFactors_t f, uint a, uint b) {
    const float in_u = float(f.insideTessellationFactor[0]);
    const float in_v = float(f.insideTessellationFactor[1]);
    return float3(spvSplit(a, spvSegments(in_u), in_u), spvSplit(b, spvSegments(in_v), in_v), 0.0f);
}

static uint spvInnerWidth(spvFactors_t f) {
    return spvSegments(float(f.insideTessellationFactor[0])) - 1u;
}

static uint spvInnerHeight(spvFactors_t f) {
    return spvSegments(float(f.insideTessellationFactor[1])) - 1u;
}

static uint spvSideInner(spvFactors_t f, uint s) {
    const uint w = spvInnerWidth(f);
    const uint h = spvInnerHeight(f);
    if (w < 1u || h < 1u) {
        return 0u;
    }
    return (s & 1u) == 0u ? w - 1u : h - 1u;
}

static uint spvInteriorTriangles(spvFactors_t f) {
    const uint w = spvInnerWidth(f);
    const uint h = spvInnerHeight(f);
    return w >= 2u && h >= 2u ? 2u * (w - 1u) * (h - 1u) : 0u;
}

static float3 spvSideInnerNode(spvFactors_t f, uint s, uint t) {
    const uint w = spvInnerWidth(f);
    const uint h = spvInnerHeight(f);
    if (w < 1u || h < 1u) {
        return float3(0.5f, 0.5f, 0.0f);
    }
    if (s == 0u) {
        return spvQuadNode(f, 1u + t, 1u);
    }
    if (s == 1u) {
        return spvQuadNode(f, w, 1u + t);
    }
    if (s == 2u) {
        return spvQuadNode(f, w - t, h);
    }
    return spvQuadNode(f, 1u, h - t);
}

static void spvInterior(spvFactors_t f, uint k, thread float3* c) {
    const uint w = spvInnerWidth(f);
    const uint q = k >> 1u;
    const uint a = 1u + q % (w - 1u);
    const uint b = 1u + q / (w - 1u);
    if ((k & 1u) == 0u) {
        c[0] = spvQuadNode(f, a, b);
        c[1] = spvQuadNode(f, a + 1u, b);
        c[2] = spvQuadNode(f, a + 1u, b + 1u);
    } else {
        c[0] = spvQuadNode(f, a, b);
        c[1] = spvQuadNode(f, a + 1u, b + 1u);
        c[2] = spvQuadNode(f, a, b + 1u);
    }
}
#endif

static uint spvInnerCount(spvFactors_t f) {
    uint count = 0u;
    for (uint s = 0u; s < SPV_SIDES; ++s) {
        count += spvSideInner(f, s);
    }
    return count;
}

// Position of the outer event k (1..n) of a strip of n outer and m inner segments among the
// strip's events, the two chains walked in step (the outer first when they meet).
static uint spvStitchPosition(uint k, uint n, uint m) {
    return k - 2u + (k * m + n - 1u) / n;
}

static void spvStitch(spvFactors_t f, uint t, thread float3* c) {
    const uint outer = spvOuterCount(f);
    uint offset = 0u;
    uint s = 0u;
    uint n = spvSideOuter(f, 0u);
    uint m = spvSideInner(f, 0u);
    while (t >= n + m && s + 1u < SPV_SIDES) {
        t -= n + m;
        offset += n;
        ++s;
        n = spvSideOuter(f, s);
        m = spvSideInner(f, s);
    }
    uint i;
    uint j;
    bool outer_event;
    if (m == 0u) {
        i = t;
        j = 0u;
        outer_event = true;
    } else {
        uint lo = 0u;
        uint hi = n;
        while (lo < hi) {
            const uint mid = (lo + hi + 1u) >> 1u;
            if (spvStitchPosition(mid, n, m) <= t) {
                lo = mid;
            } else {
                hi = mid - 1u;
            }
        }
        outer_event = lo >= 1u && spvStitchPosition(lo, n, m) == t;
        i = outer_event ? lo - 1u : lo;
        j = t - i;
    }
    c[0] = spvDomainPoint(f, (offset + i) % outer);
    c[2] = spvSideInnerNode(f, s, j);
    c[1] = outer_event ? spvDomainPoint(f, (offset + i + 1u) % outer)
                       : spvSideInnerNode(f, s, j + 1u);
}

static uint spvPatchTriangles(spvFactors_t f) {
    if (spvPatchPoints(f) == 0u) {
        return 0u;
    }
    return spvOuterCount(f) + spvInnerCount(f) + spvInteriorTriangles(f);
}

static float3 spvTriangleCorner(spvFactors_t f, uint tri, uint corner) {
    float3 c[3];
    const uint ring = spvOuterCount(f) + spvInnerCount(f);
    if (tri < ring) {
        spvStitch(f, tri, c);
    } else {
        spvInterior(f, tri - ring, c);
    }
    // Counter-clockwise in (u, v) when the domain's winding is clockwise (Metal's domain origin
    // is the lower left, Vulkan's the upper left), else the other way round.
    const float2 ab = c[1].xy - c[0].xy;
    const float2 ac = c[2].xy - c[0].xy;
    const float area = ab.x * ac.y - ab.y * ac.x;
    if ((SPV_CLOCKWISE != 0 && area < 0.0f) || (SPV_CLOCKWISE == 0 && area > 0.0f)) {
        const float3 swap = c[1];
        c[1] = c[2];
        c[2] = swap;
    }
    return c[corner];
}
#endif
)msl";

[[nodiscard]] std::string_view Trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] bool IsIdentifier(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

/// The index of the bracket closing the one at `open` ((), {} or []), npos when there is none.
[[nodiscard]] size_t MatchingBracket(std::string_view text, size_t open) {
    const char opener = text[open];
    const char closer = opener == '(' ? ')' : opener == '{' ? '}' : ']';
    int depth = 0;
    for (size_t at = open; at < text.size(); ++at) {
        if (text[at] == opener) {
            ++depth;
        } else if (text[at] == closer && --depth == 0) {
            return at;
        }
    }
    return std::string_view::npos;
}

/// Splits at the commas outside brackets and template arguments.
[[nodiscard]] std::vector<std::string_view> SplitParameters(std::string_view text) {
    std::vector<std::string_view> parts;
    int depth = 0;
    // Template brackets apart: '<' opens one only right after an identifier (as_type<float>,
    // texture2d<float, access::write>), not in a comparison or shift (a <= b, a < b, a << b),
    // whose '<' and '>' counted as brackets split an argument at a comma inside float2(...).
    int angles = 0;
    size_t begin = 0;
    const auto identifier = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    };
    for (size_t at = 0; at < text.size(); ++at) {
        const char c = text[at];
        const char before = at != 0 ? text[at - 1] : ' ';
        const char after = at + 1 < text.size() ? text[at + 1] : ' ';
        if (c == '<') {
            if (identifier(before) && after != '=' && after != '<') {
                ++angles;
            }
        } else if (c == '>') {
            if (angles > 0 && before != '-' && before != ' ' && after != '=') {
                --angles;
            }
        } else if (c == '(' || c == '[' || c == '{') {
            ++depth;
        } else if (c == ')' || c == ']' || c == '}') {
            --depth;
        } else if (c == ',' && depth == 0 && angles == 0) {
            parts.push_back(Trim(text.substr(begin, at - begin)));
            begin = at + 1;
        }
    }
    if (!Trim(text.substr(begin)).empty()) {
        parts.push_back(Trim(text.substr(begin)));
    }
    return parts;
}

struct EntryParameter {
    std::string declaration; ///< Type and name, no attribute
    std::string type;
    std::string name;
    std::string attribute;   ///< The text in [[ ]]
};

[[nodiscard]] std::optional<EntryParameter> ParseParameter(std::string_view text) {
    const size_t attribute_begin = text.find("[[");
    const size_t attribute_end = text.rfind("]]");
    if (attribute_begin == std::string_view::npos || attribute_end == std::string_view::npos) {
        return std::nullopt;
    }
    EntryParameter parameter;
    parameter.attribute = std::string(Trim(text.substr(attribute_begin + 2,
                                                       attribute_end - attribute_begin - 2)));
    const std::string_view declaration = Trim(text.substr(0, attribute_begin));
    size_t name_begin = declaration.size();
    while (name_begin > 0 && IsIdentifier(declaration[name_begin - 1])) {
        --name_begin;
    }
    if (name_begin == declaration.size() || name_begin == 0) {
        return std::nullopt;
    }
    parameter.declaration = std::string(declaration);
    parameter.name = std::string(declaration.substr(name_begin));
    parameter.type = std::string(Trim(declaration.substr(0, name_begin)));
    return parameter;
}

struct MslMember {
    std::string type;
    std::string name;
    std::string array; ///< "[3]" or empty
};

/// Size and alignment of a scalar or vector type of the stage output structs.
[[nodiscard]] bool TypeLayout(const std::string& type, u32& size, u32& alignment) {
    size_t digits = type.size();
    while (digits > 0 && std::isdigit(static_cast<unsigned char>(type[digits - 1]))) {
        --digits;
    }
    const std::string base = type.substr(0, digits);
    const u32 lanes = digits < type.size() ? static_cast<u32>(type[digits] - '0') : 1;
    if (digits + 1 < type.size() || lanes == 0 || lanes > 4) {
        return false;
    }
    u32 scalar = 0;
    if (base == "float" || base == "int" || base == "uint") {
        scalar = 4;
    } else if (base == "half" || base == "short" || base == "ushort") {
        scalar = 2;
    } else if (base == "char" || base == "uchar" || base == "bool") {
        scalar = 1;
    } else {
        return false;
    }
    alignment = scalar * (lanes == 3 ? 4 : lanes);
    size = scalar * lanes;
    return true;
}

} // Anonymous namespace


namespace {

constexpr std::string_view ReductionPrelude = R"msl(
// Sampler reduction (min and max of the texels a linear sample weighs), four bits per sampler
// slot: bit 0 min, bit 1 max, bit 2 linear mip filter, bit 3 nearest mip filter.
static bool spvReducing(ulong modes, uint slot) {
    return ((modes >> (slot * 4u)) & 3ul) != 0ul;
}

static float4 spvReduceTaps(texture2d<float> t, sampler s, float2 uv, uint mip, bool use_max) {
    const float2 size = float2(float(t.get_width(mip)), float(t.get_height(mip)));
    const float2 base = floor(uv * size - 0.5f);
    const float2 inverse = 1.0f / size;
    const float mip_level = float(mip);
    // A linear sample at a texel's centre is that texel, whatever the addressing does outside.
    const float4 a = t.sample(s, (base + 0.5f) * inverse, level(mip_level));
    const float4 b = t.sample(s, (base + float2(1.5f, 0.5f)) * inverse, level(mip_level));
    const float4 c = t.sample(s, (base + float2(0.5f, 1.5f)) * inverse, level(mip_level));
    const float4 d = t.sample(s, (base + 1.5f) * inverse, level(mip_level));
    return use_max ? max(max(a, b), max(c, d)) : min(min(a, b), min(c, d));
}

static float4 spvReducedSample(texture2d<float> t, sampler s, float2 uv, float lod, ulong modes,
                               uint slot) {
    const uint bits = uint((modes >> (slot * 4u)) & 15ul);
    const bool use_max = (bits & 2u) != 0u;
    const uint last = t.get_num_mip_levels() - 1u;
    if ((bits & 12u) == 0u) {
        return spvReduceTaps(t, s, uv, 0u, use_max);
    }
    lod = clamp(lod, 0.0f, float(last));
    if ((bits & 4u) != 0u) {
        const uint low = uint(floor(lod));
        const float4 a = spvReduceTaps(t, s, uv, low, use_max);
        if (low >= last || lod == float(low)) {
            return a;
        }
        const float4 b = spvReduceTaps(t, s, uv, low + 1u, use_max);
        return use_max ? max(a, b) : min(a, b);
    }
    return spvReduceTaps(t, s, uv, min(uint(floor(lod + 0.5f)), last), use_max);
}
)msl";


bool SpillBuffers(std::string& source, const std::string& entry_point, u32 spilled,
                  u32 table_slot) {
    const std::string needle = " " + entry_point + "(";
    const size_t at = source.find(needle);
    if (at == std::string::npos) {
        return false;
    }
    const size_t open = at + needle.size() - 1;
    const size_t close = MatchingBracket(source, open);
    const size_t body_open = close == std::string::npos ? close : source.find('{', close);
    if (close == std::string::npos || body_open == std::string::npos) {
        return false;
    }
    std::string parameters;
    std::string declarations;
    u32 found = 0;
    for (const std::string_view text :
         SplitParameters(std::string_view(source).substr(open + 1, close - open - 1))) {
        const auto parameter = ParseParameter(text);
        if (!parameter) {
            return false;
        }
        if (parameter->attribute.starts_with("buffer(")) {
            const u32 index = static_cast<u32>(std::atoi(parameter->attribute.c_str() + 7));
            if (index >= SpillBase && index < SpillBase + spilled) {
                // constant U& / const device S& / device S* ...: the same name and shape over an
                // address read from the table.
                std::string base;
                bool is_const = false;
                size_t cursor = 0;
                const std::string& type = parameter->type;
                while (cursor < type.size()) {
                    const size_t end = type.find(' ', cursor);
                    const std::string word =
                        type.substr(cursor, end == std::string::npos ? end : end - cursor);
                    if (word == "constant" || word == "const") {
                        is_const = true;
                    } else if (word != "device") {
                        base += (base.empty() ? "" : " ") + word;
                    }
                    cursor = end == std::string::npos ? type.size() : end + 1;
                }
                if (base.empty() || (base.back() != '&' && base.back() != '*')) {
                    return false;
                }
                const bool is_reference = base.back() == '&';
                base.pop_back();
                const std::string qualifier = is_const ? "const device " : "device ";
                const std::string cast = "reinterpret_cast<" + qualifier + base + "*>(spvSpill[" +
                                         std::to_string(index - SpillBase) + "])";
                declarations += "    " + qualifier + base + (is_reference ? "& " : "* ") +
                                parameter->name + " = " + (is_reference ? "*" : "") + cast + ";\n";
                ++found;
                continue;
            }
        }
        parameters += (parameters.empty() ? "" : ", ") + std::string(text);
    }
    if (found != spilled) {
        return false;
    }
    parameters += (parameters.empty() ? "" : ", ") + std::string("constant ulong* spvSpill [[buffer(") +
                  std::to_string(table_slot) + ")]]";
    source.replace(open + 1, close - open - 1, parameters);
    const size_t brace = source.find('{', open + 1 + parameters.size());
    source.insert(source.find('\n', brace) + 1, declarations);
    return true;
}

bool SpillArrays(std::string& source, const std::string& entry_point, u32 table_slot) {
    const std::string needle = " " + entry_point + "(";
    const size_t at = source.find(needle);
    if (at == std::string::npos) {
        return false;
    }
    const size_t open = at + needle.size() - 1;
    const size_t close = MatchingBracket(source, open);
    if (close == std::string::npos || source.find('{', close) == std::string::npos) {
        return false;
    }
    std::string parameters;
    std::string declarations;
    std::string members;
    for (const std::string_view text :
         SplitParameters(std::string_view(source).substr(open + 1, close - open - 1))) {
        const auto parameter = ParseParameter(text);
        if (!parameter) {
            return false;
        }
        const bool texture = parameter->attribute.starts_with("texture(");
        const bool sampler = parameter->attribute.starts_with("sampler(");
        if (texture || sampler) {
            const u32 index = static_cast<u32>(
                std::atoi(parameter->attribute.c_str() + (texture ? 8 : 8)));
            if (index >= ArraySpillBase) {
                if (!parameter->type.starts_with("array<")) {
                    return false;
                }
                // In the table (Metal 3: the resource IDs written by the CPU), and a reference
                // to it under the parameter's name.
                members += "    " + parameter->type + " " + parameter->name + " [[id(" +
                           std::to_string(index - ArraySpillBase) + ")]];\n";
                declarations += "    constant " + parameter->type + "& " + parameter->name +
                                " = spvArrayTable." + parameter->name + ";\n";
                continue;
            }
        }
        parameters += (parameters.empty() ? "" : ", ") + std::string(text);
    }
    if (members.empty()) {
        return false;
    }
    parameters += (parameters.empty() ? "" : ", ") +
                  std::string("constant spvArraySpill& spvArrayTable [[buffer(") +
                  std::to_string(table_slot) + ")]]";
    source.replace(open + 1, close - open - 1, parameters);
    const size_t brace = source.find('{', open + 1 + parameters.size());
    source.insert(source.find('\n', brace) + 1, declarations);
    constexpr std::string_view using_line = "using namespace metal;\n";
    const size_t using_at = source.find(using_line);
    const size_t insert_at =
        using_at != std::string::npos ? using_at + using_line.size() : source.rfind('\n', at) + 1;
    source.insert(insert_at, "\nstruct spvArraySpill\n{\n" + members + "};\n\n");
    return true;
}

bool SpillSamplers(std::string& source, const std::string& entry_point, u32 spilled,
                   u32 table_slot) {
    const std::string needle = " " + entry_point + "(";
    const size_t at = source.find(needle);
    if (at == std::string::npos) {
        return false;
    }
    const size_t open = at + needle.size() - 1;
    const size_t close = MatchingBracket(source, open);
    if (close == std::string::npos || source.find('{', close) == std::string::npos) {
        return false;
    }
    std::string parameters;
    std::string declarations;
    u32 found = 0;
    for (const std::string_view text :
         SplitParameters(std::string_view(source).substr(open + 1, close - open - 1))) {
        const auto parameter = ParseParameter(text);
        if (!parameter) {
            return false;
        }
        if (parameter->attribute.starts_with("sampler(")) {
            const u32 index = static_cast<u32>(std::atoi(parameter->attribute.c_str() + 8));
            if (index >= SamplerSpillBase && index < SamplerSpillBase + spilled) {
                if (parameter->type != "sampler") {
                    return false;
                }
                // A local copy of the table's sampler under the parameter's name (Metal 3:
                // samplers made with supportArgumentBuffers, their gpuResourceID in the table).
                declarations += "    sampler " + parameter->name + " = spvSamplerTable.s[" +
                                std::to_string(index - SamplerSpillBase) + "];\n";
                ++found;
                continue;
            }
        }
        parameters += (parameters.empty() ? "" : ", ") + std::string(text);
    }
    if (found != spilled) {
        return false;
    }
    parameters += (parameters.empty() ? "" : ", ") +
                  std::string("constant spvSamplerSpill& spvSamplerTable [[buffer(") +
                  std::to_string(table_slot) + ")]]";
    source.replace(open + 1, close - open - 1, parameters);
    const size_t brace = source.find('{', open + 1 + parameters.size());
    source.insert(source.find('\n', brace) + 1, declarations);
    // The table's type, before any function.
    const std::string table_type = "struct spvSamplerSpill\n{\n    array<sampler, " +
                                   std::to_string(spilled) + "> s [[id(0)]];\n};\n\n";
    constexpr std::string_view using_line = "using namespace metal;\n";
    const size_t using_at = source.find(using_line);
    const size_t insert_at =
        using_at != std::string::npos ? using_at + using_line.size() : source.rfind('\n', at) + 1;
    source.insert(insert_at, "\n" + table_type);
    return true;
}

u32 RewriteSamplerReduction(std::string& source, const std::string& entry_point, bool fragment,
                            u32 buffer_slot) {
    const std::string needle = " " + entry_point + "(";
    const size_t at = source.find(needle);
    if (at == std::string::npos) {
        return ~0U;
    }
    const size_t open = at + needle.size() - 1;
    const size_t close = MatchingBracket(source, open);
    const size_t body_close = close == std::string::npos ? close : source.find('{', close);
    if (close == std::string::npos || body_close == std::string::npos) {
        return ~0U;
    }
    std::unordered_map<std::string, bool> textures; // 2D float textures by name
    std::unordered_map<std::string, u32> samplers;  // sampler slot by name
    for (const std::string_view text :
         SplitParameters(std::string_view(source).substr(open + 1, close - open - 1))) {
        const auto parameter = ParseParameter(text);
        if (!parameter) {
            continue;
        }
        if (parameter->type == "texture2d<float>" && parameter->attribute.starts_with("texture(")) {
            textures[parameter->name] = true;
        } else if (parameter->type == "sampler" && parameter->attribute.starts_with("sampler(")) {
            samplers[parameter->name] = static_cast<u32>(std::atoi(parameter->attribute.c_str() + 8));
        }
    }
    if (textures.empty() || samplers.empty()) {
        return ~0U;
    }
    // The samples in the body, from the end so the positions before them hold.
    std::vector<std::pair<size_t, size_t>> calls; // begin of the object, end of the call
    const std::string call = ".sample(";
    for (size_t found = source.find(call, body_close); found != std::string::npos;
         found = source.find(call, found + call.size())) {
        size_t begin = found;
        while (begin > body_close && IsIdentifier(source[begin - 1])) {
            --begin;
        }
        if (begin == found || !textures.contains(source.substr(begin, found - begin))) {
            continue;
        }
        const size_t end = MatchingBracket(source, found + call.size() - 1);
        if (end != std::string::npos) {
            calls.emplace_back(begin, end + 1);
        }
    }
    bool rewritten = false;
    // From the last call back: a call nested in another's arguments (a sample whose coordinate
    // is another sample) comes after it, and rewriting it lengthens the outer call, whose end
    // is moved along below; with the end left where it was, the outer rewrite cut the source in
    // the wrong place and Metal refused the program.
    for (size_t index = calls.size(); index-- != 0;) {
        const auto [begin, end] = calls[index];
        const size_t dot = source.find('.', begin);
        const std::string texture = source.substr(begin, dot - begin);
        const std::string_view inner = std::string_view(source).substr(
            dot + call.size(), end - 1 - (dot + call.size()));
        const std::vector<std::string_view> arguments = SplitParameters(inner);
        if (arguments.size() < 2 || arguments.size() > 3 ||
            !samplers.contains(std::string(arguments[0]))) {
            continue;
        }
        const std::string sampler(arguments[0]);
        const std::string coordinate(arguments[1]);
        const std::string implicit = fragment ? texture + ".calculate_unclamped_lod(" + sampler +
                                                    ", " + coordinate + ")"
                                              : "0.0f";
        std::string lod;
        if (arguments.size() == 2) {
            lod = implicit;
        } else if (arguments[2].starts_with("level(") && arguments[2].ends_with(")")) {
            lod = "(" + std::string(arguments[2].substr(6, arguments[2].size() - 7)) + ")";
        } else if (arguments[2].starts_with("bias(") && arguments[2].ends_with(")")) {
            lod = implicit + " + (" + std::string(arguments[2].substr(5, arguments[2].size() - 6)) + ")";
        } else {
            continue;
        }
        const std::string slot = std::to_string(samplers[sampler]);
        const std::string original = source.substr(begin, end - begin);
        const std::string replacement =
            "(spvReducing(spvReduction, " + slot + "u) ? spvReducedSample(" + texture + ", " +
            sampler + ", " + coordinate + ", " + lod + ", spvReduction, " + slot + "u) : " +
            original + ")";
        source.replace(begin, end - begin, replacement);
        const size_t added = replacement.size() - (end - begin);
        for (size_t outer = 0; outer < index; ++outer) {
            if (calls[outer].second >= end) {
                calls[outer].second += added;
            }
        }
        rewritten = true;
    }
    if (!rewritten) {
        return ~0U;
    }
    source.insert(close, ", constant ulong& spvReduction [[buffer(" + std::to_string(buffer_slot) +
                             ")]]");
    const size_t using_at = source.find("using namespace metal;");
    source.insert(using_at == std::string::npos ? 0 : using_at + 22,
                  "\n" + std::string(ReductionPrelude));
    return buffer_slot;
}

} // Anonymous namespace

DomainEmulation EmulateDomain(const std::string& evaluation_msl, const std::string& entry_point,
                              const TessellationModes& modes, bool triangle_list) {
    DomainEmulation result;
    const auto fail = [&](std::string error) {
        result.error = std::move(error);
        return result;
    };
    // The entry function: [[ patch(...) ]] vertex R entry(parameters) { body }.
    const std::string needle = " " + entry_point + "(";
    size_t at = 0;
    size_t vertex_at = std::string::npos;
    while ((at = evaluation_msl.find(needle, at)) != std::string::npos) {
        const size_t line_begin = evaluation_msl.rfind('\n', at) + 1;
        const size_t found = evaluation_msl.find("vertex ", line_begin);
        if (found != std::string::npos && found < at) {
            vertex_at = found;
            break;
        }
        at += needle.size();
    }
    if (vertex_at == std::string::npos) {
        return fail("evaluation function not found");
    }
    const size_t line_begin = evaluation_msl.rfind('\n', vertex_at) + 1;
    const std::string return_type(
        Trim(std::string_view(evaluation_msl).substr(vertex_at + 7, at - vertex_at - 7)));
    const size_t open = at + needle.size() - 1;
    const size_t close = MatchingBracket(evaluation_msl, open);
    const size_t body_open = close == std::string::npos ? close : evaluation_msl.find('{', close);
    const size_t body_close =
        body_open == std::string::npos ? body_open : MatchingBracket(evaluation_msl, body_open);
    if (close == std::string::npos || body_open == std::string::npos ||
        body_close == std::string::npos || return_type.empty()) {
        return fail("evaluation function not understood");
    }

    std::vector<EntryParameter> parameters;
    for (const std::string_view text :
         SplitParameters(std::string_view(evaluation_msl).substr(open + 1, close - open - 1))) {
        auto parameter = ParseParameter(text);
        if (!parameter) {
            return fail("evaluation parameter without an attribute: " + std::string(text));
        }
        parameters.push_back(std::move(*parameter));
    }

    // Slots after the program's own buffers.
    u32 base = 0;
    for (const EntryParameter& parameter : parameters) {
        if (parameter.attribute.starts_with("buffer(") && parameter.name != "spvTessLevel") {
            base = (std::max)(base, static_cast<u32>(std::atoi(parameter.attribute.c_str() + 7)) + 1);
        }
    }
    if (base + 5 > 30) {
        return fail("too many buffers next to the domain emulation's");
    }
    result.factors_slot = base;
    result.params_slot = base + 1;
    result.offsets_slot = base + 2;
    result.args_slot = base + 3;
    result.output_slot = base + 4;

    // Its parameters: those the kernel passes on as they are, and the three it computes.
    std::string kernel_parameters;
    std::string helper_parameters;
    std::string call_arguments;
    for (const EntryParameter& parameter : parameters) {
        const std::string& attribute = parameter.attribute;
        std::string argument = parameter.name;
        if (attribute == "position_in_patch") {
            argument = parameter.type == "float2" ? "spvCoordinate.xy" : "spvCoordinate";
        } else if (attribute == "patch_id") {
            argument = "spvPatch";
        } else if (attribute == "instance_id" || attribute == "base_instance") {
            argument = parameter.type + "(0)";
        } else if (attribute.starts_with("buffer(") || attribute.starts_with("texture(") ||
                   attribute.starts_with("sampler(")) {
            std::string text = parameter.declaration + " [[" + attribute + "]]";
            if (parameter.name == "spvTessLevel") {
                result.level_slot = base + 5;
                text = parameter.declaration + " [[buffer(" + std::to_string(result.level_slot) + ")]]";
            }
            kernel_parameters += text + ", ";
        } else {
            return fail("evaluation parameter not understood: " + attribute);
        }
        helper_parameters += (helper_parameters.empty() ? "" : ", ") + parameter.declaration;
        call_arguments += (call_arguments.empty() ? "" : ", ") + argument;
    }

    // The output struct, and a copy of it without attributes to keep in a buffer.
    const std::string struct_head = "struct " + return_type;
    const size_t struct_at = evaluation_msl.find(struct_head + "\n{");
    const size_t struct_end =
        struct_at == std::string::npos ? struct_at : evaluation_msl.find("};", struct_at);
    if (struct_at == std::string::npos || struct_end == std::string::npos) {
        return fail("evaluation output struct not found");
    }
    std::vector<MslMember> members;
    {
        std::string_view text = std::string_view(evaluation_msl)
                                    .substr(struct_at + struct_head.size() + 2,
                                            struct_end - struct_at - struct_head.size() - 2);
        size_t begin = 0;
        while (begin < text.size()) {
            size_t end = text.find(';', begin);
            if (end == std::string_view::npos) {
                break;
            }
            std::string member(text.substr(begin, end - begin));
            begin = end + 1;
            for (size_t attribute = member.find("[["); attribute != std::string::npos;
                 attribute = member.find("[[")) {
                const size_t stop = member.find("]]", attribute);
                if (stop == std::string::npos) {
                    return fail("evaluation output member not understood");
                }
                member.erase(attribute, stop + 2 - attribute);
            }
            std::string_view declaration = Trim(member);
            if (declaration.empty()) {
                continue;
            }
            MslMember parsed;
            if (declaration.back() == ']') {
                const size_t bracket = declaration.rfind('[');
                parsed.array = std::string(declaration.substr(bracket));
                declaration = Trim(declaration.substr(0, bracket));
            }
            size_t name_begin = declaration.size();
            while (name_begin > 0 && IsIdentifier(declaration[name_begin - 1])) {
                --name_begin;
            }
            parsed.name = std::string(declaration.substr(name_begin));
            parsed.type = std::string(Trim(declaration.substr(0, name_begin)));
            if (parsed.name.empty() || parsed.type.empty()) {
                return fail("evaluation output member not understood");
            }
            members.push_back(std::move(parsed));
        }
    }
    u32 offset = 0;
    u32 widest = 4;
    std::string vertex_struct = "struct spvDomainVertex\n{\n";
    std::string store = "static spvDomainVertex spvDomainStore(" + return_type +
                        " v)\n{\n    spvDomainVertex o;\n";
    std::string load = "static " + return_type +
                       " spvDomainLoad(const device spvDomainVertex& v)\n{\n    " + return_type +
                       " o = {};\n";
    for (const MslMember& member : members) {
        u32 size = 0;
        u32 alignment = 0;
        if (!TypeLayout(member.type, size, alignment)) {
            return fail("evaluation output type not understood: " + member.type);
        }
        u32 elements = 1;
        if (!member.array.empty()) {
            elements = static_cast<u32>(std::atoi(member.array.c_str() + 1));
        }
        offset = (offset + alignment - 1) / alignment * alignment + size * elements;
        widest = (std::max)(widest, alignment);
        vertex_struct += "    " + member.type + " " + member.name + member.array + ";\n";
        if (member.array.empty()) {
            store += "    o." + member.name + " = v." + member.name + ";\n";
            load += "    o." + member.name + " = v." + member.name + ";\n";
        } else {
            const std::string loop = "    for (uint i = 0; i < " + std::to_string(elements) +
                                     "; ++i) { ";
            store += loop + "o." + member.name + "[i] = v." + member.name + "[i]; }\n";
            load += loop + "o." + member.name + "[i] = v." + member.name + "[i]; }\n";
        }
    }
    result.vertex_stride = (offset + widest - 1) / widest * widest;
    vertex_struct += "};\nstatic_assert(sizeof(spvDomainVertex) == " +
                     std::to_string(result.vertex_stride) + ", \"vertex layout\");\n";
    store += "    return o;\n}\n";
    load += "    return o;\n}\n";

    using Primitive = TessellationModes::Primitive;
    const u32 domain = modes.primitive == Primitive::Triangles ? 0 : modes.primitive == Primitive::Quads ? 1 : 2;
    result.lines = domain == 2 && !modes.point_mode;
    result.triangles = triangle_list && domain != 2 && !modes.point_mode;
    result.max_vertices_per_patch = domain == 0   ? 3 * 64 + 63 * 62 / 2
                                    : domain == 1 ? 4 * 64 + 63 * 63
                                    : result.lines ? 64 * 64 * 2
                                                   : 64 * 65;
    if (result.triangles) {
        // Three corners a triangle: a ring of outer and inner loops, then the inside lattice.
        result.max_vertices_per_patch = domain == 0 ? 3 * (192 + 183 + 61 * 61)
                                                    : 3 * (256 + 248 + 2 * 62 * 62);  // ring and lattice
    }
    const u32 spacing = modes.spacing == TessellationModes::Spacing::FractionalOdd    ? 1
                        : modes.spacing == TessellationModes::Spacing::FractionalEven ? 2
                                                                                      : 0;
    const std::string factors_type = domain == 0 ? "MTLTriangleTessellationFactorsHalf"
                                                 : "MTLQuadTessellationFactorsHalf";
    result.count_entry = "spvDomainCount";
    result.evaluate_entry = "spvDomainEvaluate";
    result.vertex_entry = "spvDomainVertexMain";

    std::string source = evaluation_msl.substr(0, line_begin);
    source += "static " + return_type + " spvEvaluate(" + helper_parameters + ")\n";
    source += evaluation_msl.substr(body_open, body_close + 1 - body_open);
    source += evaluation_msl.substr(body_close + 1);
    source += "\n#define SPV_DOMAIN " + std::to_string(domain) + "\n#define SPV_SPACING " +
              std::to_string(spacing) + "\n#define SPV_LINES " + (result.lines ? "1" : "0") +
              "\n#define SPV_TRIANGLE_LIST " + (result.triangles ? "1" : "0") +
              "\n#define SPV_CLOCKWISE " + (modes.clockwise ? "1" : "0") + "\n";
    source += "typedef " + factors_type + " spvFactors_t;\n";
    source += "struct spvDomainParams\n{\n    uint patches;\n    uint capacity;\n    uint reserved0;\n"
              "    uint reserved1;\n};\n";
    source += vertex_struct + store + load;
    source += DomainGenerator;
    const auto slot = [](u32 value) { return std::to_string(value); };
    source += "\nkernel void spvDomainCount(const device spvFactors_t* spvFactors [[buffer(" +
              slot(result.factors_slot) + ")]], constant spvDomainParams& spvParams [[buffer(" +
              slot(result.params_slot) + ")]], device uint* spvOffsets [[buffer(" +
              slot(result.offsets_slot) + ")]], device uint* spvArgs [[buffer(" +
              slot(result.args_slot) + ")]], uint spvThread [[thread_position_in_grid]])\n{\n"
              "    if (spvThread != 0u) {\n        return;\n    }\n    uint total = 0u;\n"
              "    for (uint patch = 0u; patch < spvParams.patches; ++patch) {\n"
              "        spvOffsets[patch] = total;\n"
              "#if SPV_TRIANGLE_LIST\n"
              "        uint count = min(3u * spvPatchTriangles(spvFactors[patch]), spvParams.capacity - total);\n"
              "        count -= count % 3u;\n"
              "#else\n"
              "        uint count = min(spvPatchPoints(spvFactors[patch]), spvParams.capacity - total);\n"
              "#endif\n"
              "#if SPV_LINES\n        count &= ~1u;\n#endif\n        total += count;\n    }\n"
              "    spvOffsets[spvParams.patches] = total;\n    spvArgs[0] = total;\n"
              "    spvArgs[1] = 1u;\n    spvArgs[2] = 0u;\n    spvArgs[3] = 0u;\n}\n";
    source += "\nkernel void spvDomainEvaluate(" + kernel_parameters +
              "const device spvFactors_t* spvFactors [[buffer(" + slot(result.factors_slot) +
              ")]], constant spvDomainParams& spvParams [[buffer(" + slot(result.params_slot) +
              ")]], const device uint* spvOffsets [[buffer(" + slot(result.offsets_slot) +
              ")]], device spvDomainVertex* spvOut [[buffer(" + slot(result.output_slot) +
              ")]], uint spvPatch [[threadgroup_position_in_grid]], uint spvLane "
              "[[thread_index_in_threadgroup]])\n{\n    if (spvPatch >= spvParams.patches) {\n"
              "        return;\n    }\n    const spvFactors_t factors = spvFactors[spvPatch];\n"
              "    const uint first = spvOffsets[spvPatch];\n"
              "    const uint count = spvOffsets[spvPatch + 1u] - first;\n"
              "    for (uint k = spvLane; k < count; k += " + std::to_string(DomainThreads) +
              "u) {\n"
              "#if SPV_TRIANGLE_LIST\n"
              "        const float3 spvCoordinate = spvTriangleCorner(factors, k / 3u, k % 3u);\n"
              "#else\n"
              "        const float3 spvCoordinate = spvDomainPoint(factors, k);\n"
              "#endif\n"
              "        spvOut[first + k] = spvDomainStore(spvEvaluate(" + call_arguments + "));\n"
              "    }\n}\n";
    source += "\nvertex " + return_type +
              " spvDomainVertexMain(uint spvVertex [[vertex_id]], const device spvDomainVertex* "
              "spvIn [[buffer(0)]])\n{\n    return spvDomainLoad(spvIn[spvVertex]);\n}\n";
    result.source = std::move(source);
    return result;
}

} // namespace Metal::ShaderTranslate
