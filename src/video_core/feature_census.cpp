// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/feature_census.h"

namespace VideoCommon::FeatureCensus {

namespace Detail {
const bool enabled = [] {
    const char* const value = std::getenv("EDEN_NX2_FEATURE_CENSUS");
    return value != nullptr && value[0] == '1';
}();
} // namespace Detail

namespace {

using Maxwell = Tegra::Engines::Maxwell3D;

constexpr size_t MAX_ROWS = 1U << 18;

struct Category {
    const char* name;
    std::vector<std::string> columns;
    std::map<std::vector<u32>, u64> rows;
    u64 total{};
    u64 dropped{};
};

struct State {
    std::mutex mutex;
    Category draws{"draws"};
    Category clears{"clears"};
    Category dispatches{"dispatches"};
    Category queries{"queries"};
    Category programs{"programs"};
    Category images{"images"};
    Category image_views{"image_views"};
    Category samplers{"samplers"};
    Category copies{"copies"};
    Category misc{"misc"};
    std::chrono::steady_clock::time_point last_write{std::chrono::steady_clock::now()};
    u64 writes{};

    [[nodiscard]] std::array<Category*, 10> All() noexcept {
        return {&draws,  &clears,      &dispatches, &queries, &programs,
                &images, &image_views, &samplers,   &copies,  &misc};
    }
};

State& Census() {
    static State state;
    return state;
}

thread_local u32 draw_kind{};

/// One record: the first of a category names its columns. Made with the census locked.
class Row {
public:
    explicit Row(Category& category_) : category{category_}, naming{category_.columns.empty()} {
        words.clear();
    }

    void Put(const char* name, u32 value) {
        if (naming) {
            category.columns.emplace_back(name);
        }
        words.push_back(value);
    }

    void Put(const char* name, u32 index, u32 value) {
        if (naming) {
            category.columns.push_back(fmt::format("{}[{}]", name, index));
        }
        words.push_back(value);
    }

    void PutHash(const char* name, u64 hash) {
        if (naming) {
            category.columns.push_back(fmt::format("{}.hi", name));
            category.columns.push_back(fmt::format("{}.lo", name));
        }
        words.push_back(static_cast<u32>(hash >> 32));
        words.push_back(static_cast<u32>(hash));
    }

    void Commit() {
        ++category.total;
        if (const auto it = category.rows.find(words); it != category.rows.end()) {
            ++it->second;
        } else if (category.rows.size() < MAX_ROWS) {
            category.rows.emplace(words, 1);
        } else {
            ++category.dropped;
        }
    }

private:
    static inline std::vector<u32> words;

    Category& category;
    bool naming;
};

#define REG(field) static_cast<u32>(offsetof(Maxwell::Regs, field) / sizeof(u32))

static_assert(offsetof(Maxwell::Regs::RenderTargetConfig, format) == 0x10);
static_assert(sizeof(Maxwell::Regs::RenderTargetConfig) == 0x40);
static_assert(offsetof(Maxwell::Regs::Zeta, format) == 0x8);
static_assert(sizeof(Maxwell::Regs::ScissorTest) == 0x10);

using Range = DrawRange;

const std::array DRAW_RANGES{
    Range{"rt.format", REG(rt) + 4, 8, 16},
    Range{"rt_control", REG(rt_control)},
    Range{"zeta.format", REG(zeta) + 2},
    Range{"zeta_enable", REG(zeta_enable)},
    Range{"anti_alias_samples_mode", REG(anti_alias_samples_mode)},
    Range{"anti_alias_alpha_control", REG(anti_alias_alpha_control)},
    Range{"framebuffer_srgb", REG(framebuffer_srgb)},
    Range{"rasterize_enable", REG(rasterize_enable)},
    Range{"mandated_early_z", REG(mandated_early_z)},
    Range{"transform_feedback_enabled", REG(transform_feedback_enabled)},
    Range{"tessellation.params", REG(tessellation)},
    Range{"patch_vertices", REG(patch_vertices)},
    Range{"mesh", static_cast<u32>(Maxwell::Regs::MeshControl), 4},
    Range{"depth_mode", REG(depth_mode)},
    Range{"viewport_clip_control", REG(viewport_clip_control)},
    Range{"viewport_scale_offset_enabled", REG(viewport_scale_offset_enabled)},
    Range{"window_origin", REG(window_origin)},
    Range{"scissor_test.enable", REG(scissor_test), 16, 4},
    Range{"polygon_mode_front", REG(polygon_mode_front)},
    Range{"polygon_mode_back", REG(polygon_mode_back)},
    Range{"fill_via_triangle_mode", REG(fill_via_triangle_mode)},
    Range{"polygon_offset_point_enable", REG(polygon_offset_point_enable)},
    Range{"polygon_offset_line_enable", REG(polygon_offset_line_enable)},
    Range{"polygon_offset_fill_enable", REG(polygon_offset_fill_enable)},
    Range{"gl_cull_test_enabled", REG(gl_cull_test_enabled)},
    Range{"gl_front_face", REG(gl_front_face)},
    Range{"gl_cull_face", REG(gl_cull_face)},
    Range{"provoking_vertex", REG(provoking_vertex)},
    Range{"shade_mode", REG(shade_mode)},
    Range{"conservative_raster_enable", REG(conservative_raster_enable)},
    Range{"line_width_smooth", REG(line_width_smooth)},
    Range{"line_width_aliased", REG(line_width_aliased)},
    Range{"aliased_line_width_enabled", REG(aliased_line_width_enabled)},
    Range{"line_anti_alias_enable", REG(line_anti_alias_enable)},
    Range{"point_size", REG(point_size)},
    Range{"point_sprite_enable", REG(point_sprite_enable)},
    Range{"point_size_attribute", REG(point_size_attribute)},
    Range{"user_clip_enable", REG(user_clip_enable)},
    Range{"depth_test_enable", REG(depth_test_enable)},
    Range{"depth_write_enabled", REG(depth_write_enabled)},
    Range{"depth_test_func", REG(depth_test_func)},
    Range{"depth_bounds_enable", REG(depth_bounds_enable)},
    Range{"stencil_enable", REG(stencil_enable)},
    Range{"stencil_two_side_enable", REG(stencil_two_side_enable)},
    Range{"stencil_front_op", REG(stencil_front_op), 4},
    Range{"stencil_front_func_mask", REG(stencil_front_func_mask)},
    Range{"stencil_front_mask", REG(stencil_front_mask)},
    Range{"stencil_back_op", REG(stencil_back_op), 4},
    Range{"stencil_back_func_mask", REG(stencil_back_func_mask)},
    Range{"stencil_back_mask", REG(stencil_back_mask)},
    Range{"alpha_test_enabled", REG(alpha_test_enabled)},
    Range{"alpha_test_func", REG(alpha_test_func)},
    Range{"alpha_test_ref", REG(alpha_test_ref)},
    Range{"blend", REG(blend), 17},
    Range{"blend_per_target_enabled", REG(blend_per_target_enabled)},
    Range{"blend_per_target", REG(blend_per_target), 64},
    Range{"color_mask_common", REG(color_mask_common)},
    Range{"color_mask", REG(color_mask), 8},
    Range{"color_target_mrt_enable", REG(color_target_mrt_enable)},
    Range{"frag_color_clamp", REG(frag_color_clamp)},
    Range{"logic_op", REG(logic_op), 2},
    Range{"primitive_restart.enabled", REG(primitive_restart)},
    Range{"render_enable.mode", REG(render_enable) + 2},
    Range{"render_enable_override", REG(render_enable_override)},
    Range{"zpass_pixel_count_enable", REG(zpass_pixel_count_enable)},
    Range{"vertex_attrib_format", REG(vertex_attrib_format), 32},
    Range{"vertex_streams.stride", REG(vertex_streams), 32, 4},
    Range{"vertex_streams.frequency", REG(vertex_streams) + 3, 32, 4},
    Range{"vertex_stream_instances", REG(vertex_stream_instances), 32},
};

#undef REG

void PutRanges(Row& row, const Maxwell::Regs& regs) {
    const bool per_target = regs.blend_per_target_enabled != 0;
    for (const Range& range : DRAW_RANGES) {
        // Registers that do not apply would only multiply the records.
        const bool stale = !per_target && std::string_view{range.name} == "blend_per_target";
        for (u32 index = 0; index < range.count; ++index) {
            const u32 value = stale ? 0 : regs.reg_array[range.first + index * range.step];
            if (range.count == 1) {
                row.Put(range.name, value);
            } else {
                row.Put(range.name, index, value);
            }
        }
    }
}

template <typename Func>
void VisitProgramFlags(const Shader::Info& info, Func&& func) {
    func("is_task", info.is_task);
    func("is_mesh", info.is_mesh);
    func("uses_workgroup_id", info.uses_workgroup_id);
    func("uses_local_invocation_id", info.uses_local_invocation_id);
    func("uses_invocation_id", info.uses_invocation_id);
    func("uses_invocation_info", info.uses_invocation_info);
    func("uses_sample_id", info.uses_sample_id);
    func("uses_is_helper_invocation", info.uses_is_helper_invocation);
    func("uses_subgroup_invocation_id", info.uses_subgroup_invocation_id);
    func("uses_subgroup_shuffles", info.uses_subgroup_shuffles);
    func("uses_quad_shuffles", info.uses_quad_shuffles);
    func("uses_subgroup_vote", info.uses_subgroup_vote);
    func("uses_subgroup_mask", info.uses_subgroup_mask);
    func("uses_fswzadd", info.uses_fswzadd);
    func("removed_conditional_barrier", info.removed_conditional_barrier);
    func("loads_indexed_attributes", info.loads_indexed_attributes);
    func("stores_indexed_attributes", info.stores_indexed_attributes);
    func("stores_sample_mask", info.stores_sample_mask);
    func("stores_frag_depth", info.stores_frag_depth);
    func("stores_tess_level_outer", info.stores_tess_level_outer);
    func("stores_tess_level_inner", info.stores_tess_level_inner);
    func("stores_global_memory", info.stores_global_memory);
    func("uses_local_memory", info.uses_local_memory);
    func("uses_fp16", info.uses_fp16);
    func("uses_fp64", info.uses_fp64);
    func("uses_int8", info.uses_int8);
    func("uses_int16", info.uses_int16);
    func("uses_int64", info.uses_int64);
    func("uses_image_1d", info.uses_image_1d);
    func("uses_sampled_1d", info.uses_sampled_1d);
    func("uses_sparse_residency", info.uses_sparse_residency);
    func("uses_demote_to_helper_invocation", info.uses_demote_to_helper_invocation);
    func("uses_derivatives", info.uses_derivatives);
    func("uses_typeless_image_reads", info.uses_typeless_image_reads);
    func("uses_typeless_image_writes", info.uses_typeless_image_writes);
    func("uses_image_buffers", info.uses_image_buffers);
    func("uses_shared_atomics", info.uses_shared_increment || info.uses_shared_decrement);
    func("uses_global_atomics", info.uses_global_increment || info.uses_global_decrement);
    func("uses_float_atomics",
         info.uses_atomic_f32_add || info.uses_atomic_f16x2_add || info.uses_atomic_f16x2_min ||
             info.uses_atomic_f16x2_max || info.uses_atomic_f32x2_add ||
             info.uses_atomic_f32x2_min || info.uses_atomic_f32x2_max);
    func("uses_s32_min_max_atomics", info.uses_atomic_s32_min || info.uses_atomic_s32_max);
    func("uses_int64_bit_atomics", info.uses_int64_bit_atomics);
    func("uses_atomic_image_u32", info.uses_atomic_image_u32);
    func("uses_global_memory", info.uses_global_memory);
    func("uses_cooperative_matrix", info.uses_cooperative_matrix);
    func("uses_shadow_lod", info.uses_shadow_lod);
    func("uses_rescaling_uniform", info.uses_rescaling_uniform);
    func("uses_cbuf_indirect", info.uses_cbuf_indirect);
    func("uses_render_area", info.uses_render_area);
    func("requires_layer_emulation", info.requires_layer_emulation);
    using Shader::IR::Attribute;
    func("loads_primitive_id", info.loads[Attribute::PrimitiveId]);
    func("loads_front_face", info.loads[Attribute::FrontFace]);
    func("loads_point_sprite", info.loads.AnyComponent(Attribute::PointSpriteS));
    func("loads_instance_id", info.loads[Attribute::InstanceId]);
    func("loads_vertex_id", info.loads[Attribute::VertexId]);
    func("loads_base_instance", info.loads[Attribute::BaseInstance]);
    func("loads_base_vertex", info.loads[Attribute::BaseVertex]);
    func("loads_draw_id", info.loads[Attribute::DrawID]);
    func("loads_legacy", info.loads.Legacy());
    func("stores_point_size", info.stores[Attribute::PointSize]);
    func("stores_layer", info.stores[Attribute::Layer]);
    func("stores_viewport_index", info.stores[Attribute::ViewportIndex]);
    func("stores_viewport_mask", info.stores[Attribute::ViewportMask]);
    func("stores_primitive_id", info.stores[Attribute::PrimitiveId]);
    func("stores_clip_distances", info.stores.ClipDistances());
    func("stores_legacy", info.stores.Legacy());
}

void WriteCategory(std::ofstream& file, const Category& category, bool last) {
    file << "\"" << category.name << "\":{\"total\":" << category.total
         << ",\"dropped\":" << category.dropped << ",\"columns\":[\"count\"";
    for (const std::string& column : category.columns) {
        file << ",\"" << column << "\"";
    }
    file << "],\"rows\":[";
    bool first = true;
    for (const auto& [words, count] : category.rows) {
        file << (first ? "\n[" : ",\n[") << count;
        first = false;
        for (const u32 word : words) {
            file << ',' << word;
        }
        file << ']';
    }
    file << (last ? "]}\n" : "]},\n");
}

void Write(State& state) {
    const std::filesystem::path directory = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir);
    const std::filesystem::path target = directory / "nx2_feature_census.json";
    const std::filesystem::path temporary = directory / "nx2_feature_census.json.tmp";
    {
        std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
        if (!file) {
            return;
        }
        file << "{\"version\":1,\n";
        const auto categories = state.All();
        for (size_t index = 0; index < categories.size(); ++index) {
            WriteCategory(file, *categories[index], index + 1 == categories.size());
        }
        file << "}\n";
    }
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(target, error);
        std::filesystem::rename(temporary, target, error);
    }
    if (state.writes++ == 0) {
        LOG_INFO(HW_GPU, "Feature census: writing {}", target.string());
    }
    LOG_INFO(HW_GPU,
             "Feature census: {} draws ({} distinct), {} clears ({}), {} dispatches ({}), {} "
             "queries ({}), {} programs, {} images ({}), {} views ({}), {} samplers ({}), {} "
             "copies ({})",
             state.draws.total, state.draws.rows.size(), state.clears.total,
             state.clears.rows.size(), state.dispatches.total, state.dispatches.rows.size(),
             state.queries.total, state.queries.rows.size(), state.programs.rows.size(),
             state.images.total, state.images.rows.size(), state.image_views.total,
             state.image_views.rows.size(), state.samplers.total, state.samplers.rows.size(),
             state.copies.total, state.copies.rows.size());
}

} // Anonymous namespace

std::span<const DrawRange> DrawRanges() noexcept {
    return DRAW_RANGES;
}

void SetDrawKind(u32 flags, u32 instance_count) {
    draw_kind = flags | (instance_count > 1 ? DrawInstanced : 0U);
}

void NoteDraw(const Maxwell& maxwell3d, std::span<const u64, 6> unique_hashes) {
    const Maxwell::Regs& regs = maxwell3d.regs;
    const auto& draw_state = maxwell3d.draw_manager.draw_state;
    const bool is_indexed = (draw_kind & DrawIndexed) != 0;
    u32 flags = draw_kind;
    flags |= regs.IsMeshEnabled() ? DrawMesh : 0U;
    flags |= !regs.IsMeshEnabled() && unique_hashes[2] != 0 ? DrawTessellated : 0U;
    flags |= !regs.IsMeshEnabled() && unique_hashes[4] != 0 ? DrawGeometry : 0U;
    flags |= draw_state.draw_mode == Maxwell::DrawManager::DrawMode::InlineIndex
                 ? DrawInlineIndex
                 : 0U;
    u32 color_targets = 0;
    for (const auto& target : regs.rt) {
        color_targets += target.Address() != 0 && static_cast<u32>(target.format) != 0 ? 1U : 0U;
    }
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.draws};
    row.Put("flags", flags);
    row.Put("topology", static_cast<u32>(draw_state.topology));
    row.Put("index_format", is_indexed ? static_cast<u32>(draw_state.index_buffer.format) : ~0U);
    static constexpr std::array names{"vertex_a", "vertex_b",  "tess_control",
                                      "tess_eval", "geometry", "fragment"};
    for (size_t index = 0; index < names.size(); ++index) {
        row.PutHash(names[index], unique_hashes[index]);
    }
    row.Put("color_targets", color_targets);
    row.Put("zeta_target", regs.zeta_enable != 0 && regs.zeta.Address() != 0 ? 1U : 0U);
    PutRanges(row, regs);
    row.Commit();
}

void NoteClear(const Maxwell& maxwell3d, u32 layer_count) {
    const Maxwell::Regs& regs = maxwell3d.regs;
    const u32 target = regs.clear_surface.RT;
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.clears};
    row.Put("clear_surface", regs.clear_surface.raw & 0x3ffU);
    row.Put("layer", regs.clear_surface.layer);
    row.Put("layer_count", layer_count);
    row.Put("clear_control", maxwell3d.regs.reg_array[offsetof(Maxwell::Regs, clear_control) / 4]);
    row.Put("rt.format", target < regs.rt.size() ? static_cast<u32>(regs.rt[target].format) : 0U);
    row.Put("zeta.format", static_cast<u32>(regs.zeta.format));
    row.Put("anti_alias_samples_mode", static_cast<u32>(regs.anti_alias_samples_mode));
    row.Put("color_mask",
            maxwell3d.regs.reg_array[offsetof(Maxwell::Regs, color_mask) / 4 +
                                     (regs.color_mask_common != 0 ? 0U : target & 7U)]);
    row.Put("scissor_test.enable",
            maxwell3d.regs.reg_array[offsetof(Maxwell::Regs, scissor_test) / 4]);
    row.Put("stencil_front_mask", regs.stencil_front_mask);
    row.Put("framebuffer_srgb", regs.framebuffer_srgb);
    row.Commit();
}

void NoteDispatch(u64 unique_hash, bool is_indirect) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.dispatches};
    row.PutHash("compute", unique_hash);
    row.Put("indirect", is_indirect ? 1U : 0U);
    row.Commit();
}

void NoteQuery(u32 type, u32 raw_query, bool has_address) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.queries};
    row.Put("report", type);
    row.Put("query", raw_query);
    row.Put("has_address", has_address ? 1U : 0U);
    row.Commit();
}

void NoteProgram(u64 unique_hash, const Shader::IR::Program& program) {
    const Shader::Info& info = program.info;
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.programs};
    row.PutHash("program", unique_hash);
    row.Put("stage", static_cast<u32>(program.stage));
    VisitProgramFlags(info, [&row](const char* name, bool value) {
        row.Put(name, value ? 1U : 0U);
    });
    u32 frag_colors = 0;
    for (size_t index = 0; index < info.stores_frag_color.size(); ++index) {
        frag_colors |= info.stores_frag_color[index] ? 1U << index : 0U;
    }
    row.Put("stores_frag_color", frag_colors);
    row.Put("used_clip_distances", info.used_clip_distances);
    row.Put("constant_buffers", static_cast<u32>(info.constant_buffer_descriptors.size()));
    row.Put("storage_buffers", static_cast<u32>(info.storage_buffers_descriptors.size()));
    row.Put("texture_buffers", static_cast<u32>(info.texture_buffer_descriptors.size()));
    row.Put("image_buffers", static_cast<u32>(info.image_buffer_descriptors.size()));
    row.Put("textures", static_cast<u32>(info.texture_descriptors.size()));
    row.Put("images", static_cast<u32>(info.image_descriptors.size()));
    row.Put("handle_programs", static_cast<u32>(info.handle_programs.size()));
    u32 texture_types = 0;
    u32 depth_textures = 0;
    u32 multisample_textures = 0;
    u32 integer_textures = 0;
    u32 texture_arrays = 0;
    for (const auto& descriptor : info.texture_descriptors) {
        texture_types |= 1U << static_cast<u32>(descriptor.type);
        depth_textures += descriptor.is_depth ? 1U : 0U;
        multisample_textures += descriptor.is_multisample ? 1U : 0U;
        integer_textures += descriptor.is_integer ? 1U : 0U;
        texture_arrays += descriptor.count > 1 ? 1U : 0U;
    }
    row.Put("texture_types", texture_types);
    row.Put("depth_textures", depth_textures);
    row.Put("multisample_textures", multisample_textures);
    row.Put("integer_textures", integer_textures);
    row.Put("texture_arrays", texture_arrays);
    u32 image_types = 0;
    u32 image_formats = 0;
    u32 images_written = 0;
    u32 images_read = 0;
    for (const auto& descriptor : info.image_descriptors) {
        image_types |= 1U << static_cast<u32>(descriptor.type);
        image_formats |= 1U << static_cast<u32>(descriptor.format);
        images_written += descriptor.is_written ? 1U : 0U;
        images_read += descriptor.is_read ? 1U : 0U;
    }
    row.Put("image_types", image_types);
    row.Put("image_formats", image_formats);
    row.Put("images_written", images_written);
    row.Put("images_read", images_read);
    row.Put("workgroup_size", 0, program.workgroup_size[0]);
    row.Put("workgroup_size", 1, program.workgroup_size[1]);
    row.Put("workgroup_size", 2, program.workgroup_size[2]);
    row.Put("output_topology", static_cast<u32>(program.output_topology));
    row.Put("output_vertices", program.output_vertices);
    row.Put("invocations", program.invocations);
    row.Put("local_memory_size", program.local_memory_size);
    row.Put("shared_memory_size", program.shared_memory_size);
    row.Put("is_geometry_passthrough", program.is_geometry_passthrough ? 1U : 0U);
    row.Put("mesh.payload_words", program.mesh.payload_words);
    row.Put("mesh.max_vertices", program.mesh.max_vertices);
    row.Put("mesh.max_primitives", program.mesh.max_primitives);
    row.Put("mesh.topology", program.mesh.topology);
    row.Put("mesh.vertex_lines", static_cast<u32>(program.mesh.vertex_lines.size()));
    row.Put("mesh.primitive_lines", static_cast<u32>(program.mesh.primitive_lines.size()));
    row.Commit();
}

void NoteImage(u32 format, u32 type, u32 samples, u32 levels, u32 layers, u32 flags) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.images};
    row.Put("format", format);
    row.Put("type", type);
    row.Put("samples", samples);
    row.Put("levels", levels);
    row.Put("layers", layers);
    row.Put("flags", flags);
    row.Commit();
}

void NoteImageView(u32 type, u32 format, u32 image_format, u32 image_type, u32 swizzle,
                   u32 levels, u32 layers) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.image_views};
    row.Put("type", type);
    row.Put("format", format);
    row.Put("image_format", image_format);
    row.Put("image_type", image_type);
    row.Put("swizzle", swizzle);
    row.Put("levels", levels);
    row.Put("layers", layers);
    row.Commit();
}

void NoteSampler(std::span<const u64> raw) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.samplers};
    for (u32 index = 0; index < raw.size(); ++index) {
        row.Put("raw", index * 2, static_cast<u32>(raw[index]));
        row.Put("raw", index * 2 + 1, static_cast<u32>(raw[index] >> 32));
    }
    row.Commit();
}

void NoteCopy(Copy kind, u32 src_format, u32 dst_format, u32 src_samples, u32 dst_samples,
              u32 flags) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.copies};
    row.Put("kind", static_cast<u32>(kind));
    row.Put("src_format", src_format);
    row.Put("dst_format", dst_format);
    row.Put("src_samples", src_samples);
    row.Put("dst_samples", dst_samples);
    row.Put("flags", flags);
    row.Commit();
}

void NoteMisc(Misc kind, u32 value) {
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    Row row{state.misc};
    row.Put("kind", static_cast<u32>(kind));
    row.Put("value", value);
    row.Commit();
}

void Tick() {
    if (!Enabled()) {
        return;
    }
    State& state = Census();
    std::scoped_lock lock{state.mutex};
    const auto now = std::chrono::steady_clock::now();
    if (now - state.last_write < std::chrono::seconds{10}) {
        return;
    }
    state.last_write = now;
    Write(state);
}

} // namespace VideoCommon::FeatureCensus
