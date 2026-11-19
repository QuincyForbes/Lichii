// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <system_error>

#include <boost/container/small_vector.hpp>

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/oot_visual_workaround.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/feature_census.h"
#include "video_core/renderer_metal/metal_pipeline_translate.h"
#include "video_core/renderer_metal/metal_raster_expand.h"
#include "video_core/renderer_metal/metal_raster_patch.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"
#include "video_core/surface.h"

namespace Metal {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;

namespace {

[[nodiscard]] Shader::InputTopology InputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::InputTopology::Points;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::InputTopology::Lines;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return Shader::InputTopology::LinesAdjacency;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        return Shader::InputTopology::TrianglesAdjacency;
    default:
        return Shader::InputTopology::Triangles;
    }
}

// GraphicsPipelineKey::state.
constexpr u32 StateDepthMinusOneToOne = 1U << 0;
constexpr u32 StateLowerLeftOrigin = 1U << 1;
constexpr u32 StateDualSourceBlend = 1U << 2;
constexpr u32 StateAlphaTestShift = 4;
constexpr u32 StateColorTypesShift = 8;

[[nodiscard]] bool FixedStateEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FIXED_STATE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_METAL_ATTRIBUTE_TYPES=0: the key takes Vulkan's FixedPipelineState::
/// DynamicAttributeTypes as it is.
[[nodiscard]] bool AttributeTypesEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ATTRIBUTE_TYPES");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_METAL_SHUFFLE_RELATIVE=0: guest SHFL.UP/DOWN/BFLY through the absolute shuffle again.
[[nodiscard]] bool ShuffleRelativeEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SHUFFLE_RELATIVE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Two bits per generic attribute: 0 disabled (or constant), 1 float, 2 signed, 3 unsigned
/// integer, by Maxwell's VertexAttribute::Type. Normalized and scaled types read as floats. This
/// is the table FixedPipelineState::DynamicAttributeTypes had before b8d5405b1a packed it into the
/// constant 0x55dc, which gives SNORM and SINT attributes the unsigned type and UINT ones float.
[[nodiscard]] u64 AttributeTypes(const Maxwell& regs) {
    if (!AttributeTypesEnabled()) {
        return Vulkan::FixedPipelineState::DynamicAttributeTypes(regs);
    }
    static constexpr std::array<u32, 8> Types{
        0, // Invalid
        1, // SNorm
        1, // UNorm
        2, // SInt
        3, // UInt
        1, // UScaled
        1, // SScaled
        1, // Float
    };
    u64 types = 0;
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant != 0) {
            continue;
        }
        types |= u64{Types[static_cast<u32>(attribute.type.Value()) & 7]} << (index * 2);
    }
    return types;
}

/// Zero for a test that passes everything, else Shader::CompareFunction + 1.
[[nodiscard]] u32 AlphaTestState(const Maxwell& regs) {
    if (regs.alpha_test_enabled == 0) {
        return 0;
    }
    switch (regs.alpha_test_func) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return static_cast<u32>(Shader::CompareFunction::Never) + 1;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return static_cast<u32>(Shader::CompareFunction::Less) + 1;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return static_cast<u32>(Shader::CompareFunction::Equal) + 1;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return static_cast<u32>(Shader::CompareFunction::LessThanEqual) + 1;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return static_cast<u32>(Shader::CompareFunction::Greater) + 1;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return static_cast<u32>(Shader::CompareFunction::NotEqual) + 1;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return static_cast<u32>(Shader::CompareFunction::GreaterThanEqual) + 1;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return 0;
    }
    return 0;
}

[[nodiscard]] bool IsSecondSourceFactor(Maxwell::Blend::Factor factor) {
    using Factor = Maxwell::Blend::Factor;
    switch (factor) {
    case Factor::Source1Color_D3D:
    case Factor::OneMinusSource1Color_D3D:
    case Factor::Source1Alpha_D3D:
    case Factor::OneMinusSource1Alpha_D3D:
    case Factor::Source1Color_GL:
    case Factor::OneMinusSource1Color_GL:
    case Factor::Source1Alpha_GL:
    case Factor::OneMinusSource1Alpha_GL:
        return true;
    default:
        return false;
    }
}

/// Fills the key's fixed state from the registers (FixedPipelineState::Refresh's derivations).
void RefreshFixedState(GraphicsPipelineKey& key, const Maxwell& regs) {
    key.state = 0;
    key.alpha_test_ref = 0;
    key.point_size = 0;
    key.raster = 0;
    key.logic_formats = 0;
    key.expand = 0;
    key.expand_unused = 0;
    if (!FixedStateEnabled()) {
        return;
    }
    if (regs.depth_mode == Maxwell::DepthMode::MinusOneToOne) {
        key.state |= StateDepthMinusOneToOne;
    }
    if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
        key.state |= StateLowerLeftOrigin;
    }
    if (regs.blend.enable[0] != 0) {
        const auto second_source = [](const auto& blend) {
            return IsSecondSourceFactor(blend.color_source) ||
                   IsSecondSourceFactor(blend.color_dest) ||
                   IsSecondSourceFactor(blend.alpha_source) ||
                   IsSecondSourceFactor(blend.alpha_dest);
        };
        if (regs.blend_per_target_enabled != 0 ? second_source(regs.blend_per_target[0])
                                               : second_source(regs.blend)) {
            key.state |= StateDualSourceBlend;
        }
    }
    const u32 alpha_test = AlphaTestState(regs);
    key.state |= alpha_test << StateAlphaTestShift;
    if (alpha_test > static_cast<u32>(Shader::CompareFunction::Never) + 1) {
        // Rounded to a step of 8-bit alpha, as Vulkan keys it.
        key.alpha_test_ref = std::bit_cast<u32>(std::round(regs.alpha_test_ref * 255.0f) / 255.0f);
    }
    if (key.unique_hashes[5] != 0) {
        for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
            const Tegra::RenderTargetFormat format = regs.rt[rt].format;
            if (format == Tegra::RenderTargetFormat::NONE) {
                continue;
            }
            const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
            if (!VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                continue;
            }
            const Shader::AttributeType type =
                VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)
                    ? Shader::AttributeType::SignedInt
                    : Shader::AttributeType::UnsignedInt;
            key.state |= static_cast<u32>(type) << (StateColorTypesShift + rt * 2);
        }
    }
    const bool tessellation_points =
        key.unique_hashes[3] != 0 &&
        ((key.tessellation >> 4) & 3) ==
            static_cast<u32>(Maxwell::Tessellation::OutputPrimitives::Points);
    if (static_cast<Maxwell::PrimitiveTopology>(key.topology) ==
            Maxwell::PrimitiveTopology::Points ||
        tessellation_points || regs.IsMeshEnabled()) {
        key.point_size = std::bit_cast<u32>(regs.point_size);
    }
    if (key.unique_hashes[5] != 0) {
        const RasterPatch::LogicState logic = RasterPatch::LogicStateOf(regs);
        key.raster |= logic.op;
        key.logic_formats = logic.formats;
    }
    if (!regs.IsMeshEnabled() && key.unique_hashes[2] == 0 && key.unique_hashes[3] == 0 &&
        key.unique_hashes[4] == 0) {
        key.expand = RasterExpand::BitsOf(regs, static_cast<Maxwell::PrimitiveTopology>(key.topology));
    }
    if (RasterPatch::ViewportSwizzleEnabled() && regs.viewport_scale_offset_enabled != 0) {
        const u32 swizzle = regs.viewport_transform[0].swizzle.raw & 0x7777;
        if (RasterPatch::SwizzleInProgram(swizzle)) {
            key.raster |= swizzle << 8;
        }
    }
}

/// The vertex program's share of it (also before tessellation programs: Vulkan converts the
/// depth there too).
void ApplyVertexState(Shader::RuntimeInfo& info, const GraphicsPipelineKey& key) {
    info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
    info.convert_depth_mode = (key.state & StateDepthMinusOneToOne) != 0;
    if (FixedStateEnabled() && static_cast<Maxwell::PrimitiveTopology>(key.topology) ==
                                   Maxwell::PrimitiveTopology::Points) {
        info.fixed_state_point_size = std::bit_cast<float>(key.point_size);
    }
}

void ApplyFragmentState(Shader::RuntimeInfo& info, const GraphicsPipelineKey& key) {
    info.force_early_z = key.early_z != 0;
    info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
    info.dual_source_blend = (key.state & StateDualSourceBlend) != 0;
    if (const u32 alpha_test = (key.state >> StateAlphaTestShift) & 15; alpha_test != 0) {
        info.alpha_test_func = static_cast<Shader::CompareFunction>(alpha_test - 1);
        info.alpha_test_reference = std::bit_cast<float>(key.alpha_test_ref);
    }
    for (size_t rt = 0; rt < info.color_output_types.size(); ++rt) {
        info.color_output_types[rt] =
            static_cast<Shader::AttributeType>((key.state >> (StateColorTypesShift + rt * 2)) & 3);
    }
}

std::atomic<u64> visual_workaround_edits{};

bool VisualWorkaroundEnabled() {
    static const bool enabled = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_OOT_VISUAL_WORKAROUND");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

/// Vulkan's EDEN_NX2_OOT_VISUAL_WORKAROUND=1 (read under EDEN_NX2_METAL_VULKAN_SWITCHES): its
/// opt-in edit of two of Ocarina of Time's programs (the fog dispatch, the exposure pass).
void ApplyVisualWorkaround(Shader::IR::Program& program, u64 hash) {
    if (const auto edits = Shader::Optimization::ApplyOotVisualWorkaround(
            program, Settings::GetCurrentProgramID(), hash, VisualWorkaroundEnabled())) {
        visual_workaround_edits.fetch_add(1, std::memory_order_relaxed);
        LOG_WARNING(Render_Metal, "Metal: Ocarina visual workaround edited {} instructions of {:016x}",
                    *edits, hash);
    }
}

constexpr u64 StageMslMagic = 0x3156534754534C4DULL; // "MLSTGSV1"

[[nodiscard]] bool StageMslCacheEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_STAGE_CACHE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] std::optional<u128> MakeStageMslKey(std::span<const u32> spirv,
                                                  ShaderTranslate::Stage stage,
                                                  const ShaderTranslate::Options& options) {
    if (!options.inputs.empty() || !options.depth_images.empty() ||
        options.tessellation.output_vertices != 0) {
        return std::nullopt;
    }
    static const bool env_invariant = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_INVARIANT_POSITION");
        return value != nullptr && value[0] == '1';
    }();
    const u128 spirv_hash = Common::CityHash128(
        reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(u32));
    struct OptionKey {
        u64 spirv_lo;
        u64 spirv_hi;
        u32 apple_family;
        u32 msl_major;
        u32 msl_minor;
        u32 first_vertex_buffer_slot;
        u32 vertex_index_size;
        u32 flags;
    } const key{
        .spirv_lo = spirv_hash[0],
        .spirv_hi = spirv_hash[1],
        .apple_family = options.apple_family,
        .msl_major = options.msl_major,
        .msl_minor = options.msl_minor,
        .first_vertex_buffer_slot = options.first_vertex_buffer_slot,
        .vertex_index_size = options.vertex_index_size,
        .flags = static_cast<u32>(stage) | (options.ios ? (1U << 8) : 0U) |
                 (options.flip_vertex_y ? (1U << 9) : 0U) |
                 ((options.invariant_float_math || env_invariant) ? (1U << 10) : 0U) |
                 (options.point_size_builtin ? (1U << 11) : 0U) |
                 (options.sampler_reduction ? (1U << 12) : 0U) |
                 (options.spill_buffers ? (1U << 13) : 0U) |
                 (options.drop_point_size ? (1U << 14) : 0U) |
                 (options.force_temporary ? (1U << 15) : 0U),
    };
    return Common::CityHash128(reinterpret_cast<const char*>(&key), sizeof(key));
}

} // Anonymous namespace

void StageMslCache::Open(const std::filesystem::path& file_path, u64 build_stamp) {
    if (!StageMslCacheEnabled()) {
        return;
    }
    std::unique_lock lock{mutex};
    path = file_path;
    stamp = build_stamp;
    open = true;
    dirty = false;
    entries.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    u64 magic{}, stored_stamp{}, count{};
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        !in.read(reinterpret_cast<char*>(&stored_stamp), sizeof(stored_stamp)) ||
        !in.read(reinterpret_cast<char*>(&count), sizeof(count)) || magic != StageMslMagic ||
        (stamp != 0 && stored_stamp != stamp) || count > 262144) {
        return;
    }
    const auto read_str = [&](std::string& out, u32 max_len) {
        u32 len{};
        if (!in.read(reinterpret_cast<char*>(&len), sizeof(len)) || len > max_len) {
            return false;
        }
        out.resize(len);
        return len == 0 || static_cast<bool>(in.read(out.data(), len));
    };
    entries.reserve(static_cast<size_t>(count));
    for (u64 i = 0; i < count; ++i) {
        u128 key{};
        ShaderTranslate::Result res{};
        u32 num_slots{}, num_outputs{};
        if (!in.read(reinterpret_cast<char*>(&key), sizeof(key)) ||
            !read_str(res.source, 16U << 20) ||
            !read_str(res.source_without_reduction, 16U << 20) ||
            !read_str(res.entry_point, 4096) ||
            !in.read(reinterpret_cast<char*>(&num_slots), sizeof(num_slots)) || num_slots > 4096) {
            entries.clear();
            return;
        }
        res.slots.resize(num_slots);
        if (num_slots != 0 &&
            !in.read(reinterpret_cast<char*>(res.slots.data()),
                     static_cast<std::streamsize>(num_slots * sizeof(ShaderTranslate::ResourceSlot)))) {
            entries.clear();
            return;
        }
        if (!in.read(reinterpret_cast<char*>(&num_outputs), sizeof(num_outputs)) ||
            num_outputs > 4096) {
            entries.clear();
            return;
        }
        res.outputs.resize(num_outputs);
        if (num_outputs != 0 &&
            !in.read(reinterpret_cast<char*>(res.outputs.data()),
                     static_cast<std::streamsize>(num_outputs *
                                                  sizeof(ShaderTranslate::InterfaceVariable)))) {
            entries.clear();
            return;
        }
        if (!in.read(reinterpret_cast<char*>(&res.tessellation_slots),
                     sizeof(res.tessellation_slots)) ||
            !in.read(reinterpret_cast<char*>(&res.reduction_slot), sizeof(u32)) ||
            !in.read(reinterpret_cast<char*>(&res.spill_slot), sizeof(u32)) ||
            !in.read(reinterpret_cast<char*>(&res.sampler_spill_slot), sizeof(u32)) ||
            !in.read(reinterpret_cast<char*>(&res.array_spill_slot), sizeof(u32)) ||
            !in.read(reinterpret_cast<char*>(&res.array_spill_entries), sizeof(u32)) ||
            res.source.empty()) {
            entries.clear();
            return;
        }
        entries.emplace(key, std::move(res));
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    LOG_INFO(Render_Metal, "Metal: stage MSL cache {} loaded ({} stages, {:.1f} ms)",
             path.string(), entries.size(), ms);
}

void StageMslCache::Save() {
    std::shared_lock lock{mutex};
    if (!open || !dirty || path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return;
        }
        const u64 magic = StageMslMagic;
        const u64 count = entries.size();
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&stamp), sizeof(stamp));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        const auto write_str = [&](const std::string& str) {
            const u32 len = static_cast<u32>(str.size());
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len != 0) {
                out.write(str.data(), len);
            }
        };
        for (const auto& [key, res] : entries) {
            out.write(reinterpret_cast<const char*>(&key), sizeof(key));
            write_str(res.source);
            write_str(res.source_without_reduction);
            write_str(res.entry_point);
            const u32 num_slots = static_cast<u32>(res.slots.size());
            out.write(reinterpret_cast<const char*>(&num_slots), sizeof(num_slots));
            if (num_slots != 0) {
                out.write(reinterpret_cast<const char*>(res.slots.data()),
                          static_cast<std::streamsize>(num_slots *
                                                       sizeof(ShaderTranslate::ResourceSlot)));
            }
            const u32 num_outputs = static_cast<u32>(res.outputs.size());
            out.write(reinterpret_cast<const char*>(&num_outputs), sizeof(num_outputs));
            if (num_outputs != 0) {
                out.write(reinterpret_cast<const char*>(res.outputs.data()),
                          static_cast<std::streamsize>(
                              num_outputs * sizeof(ShaderTranslate::InterfaceVariable)));
            }
            out.write(reinterpret_cast<const char*>(&res.tessellation_slots),
                      sizeof(res.tessellation_slots));
            out.write(reinterpret_cast<const char*>(&res.reduction_slot), sizeof(u32));
            out.write(reinterpret_cast<const char*>(&res.spill_slot), sizeof(u32));
            out.write(reinterpret_cast<const char*>(&res.sampler_spill_slot), sizeof(u32));
            out.write(reinterpret_cast<const char*>(&res.array_spill_slot), sizeof(u32));
            out.write(reinterpret_cast<const char*>(&res.array_spill_entries), sizeof(u32));
        }
        if (!out.good()) {
            lock.unlock();
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    const size_t saved_count = entries.size();
    const std::filesystem::path dst_path = path;
    lock.unlock();
    std::unique_lock write_lock{mutex};
    std::filesystem::rename(tmp, dst_path, ec);
    if (!ec) {
        dirty = false;
        LOG_INFO(Render_Metal, "Metal: stage MSL cache {} written ({} stages)", dst_path.string(),
                 saved_count);
    }
}

std::optional<ShaderTranslate::Result> StageMslCache::Find(const u128& key) const {
    if (!StageMslCacheEnabled()) {
        return std::nullopt;
    }
    std::shared_lock lock{mutex};
    const auto it = entries.find(key);
    if (it == entries.end()) {
        return std::nullopt;
    }
    return it->second;
}

void StageMslCache::Insert(const u128& key, const ShaderTranslate::Result& result) {
    if (!StageMslCacheEnabled() || result.source.empty()) {
        return;
    }
    std::unique_lock lock{mutex};
    if (entries.emplace(key, result).second) {
        dirty = true;
    }
}

size_t StageMslCache::Size() const {
    std::shared_lock lock{mutex};
    return entries.size();
}

u64 PipelineTranslator::VisualWorkaroundEdits() noexcept {
    return visual_workaround_edits.load(std::memory_order_relaxed);
}

void PipelineTranslator::Refuse(std::atomic<u32>& count, u64 hash, std::string_view cause_,
                                std::string_view detail_) {
    cause = cause_;
    detail = detail_;
    if (count.fetch_add(1, std::memory_order_relaxed) >= 16) {
        return;
    }
    if (detail_.empty()) {
        LOG_WARNING(Render_Metal, "Metal: pipeline {:016x} not made: {}", hash, cause_);
    } else {
        LOG_WARNING(Render_Metal, "Metal: pipeline {:016x} not made: {}: {}", hash, cause_,
                    detail_);
    }
}

void PipelineTranslator::RefreshKey(GraphicsPipelineKey& key, const Maxwell& regs,
                                    Maxwell::PrimitiveTopology topology) {
    key.topology = static_cast<u32>(topology);
    key.early_z = regs.mandated_early_z != 0 ? 1 : 0;
    key.attribute_types = AttributeTypes(regs);
    const auto& tessellation = regs.tessellation.params;
    key.tessellation = key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0
                           ? static_cast<u32>(tessellation.domain_type.Value()) |
                                 (static_cast<u32>(tessellation.spacing.Value()) << 2) |
                                 (static_cast<u32>(tessellation.output_primitives.Value()) << 4)
                           : 0U;
    if (key.unique_hashes[2] == 0 && key.unique_hashes[3] != 0) {
        // Without a control program the patch's control points are the vertex outputs: the
        // evaluation program is translated for their count (Vulkan's patch control points).
        key.tessellation |= (std::clamp(regs.patch_vertices, 1U, 32U) - 1) << TessPatchShift;
    }
    RefreshFixedState(key, regs);
    key.xfb = Xfb::MakeLayout(regs, topology, key.unique_hashes[4] != 0);
}

size_t GraphicsPipelineKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this));
}

size_t ComputePipelineKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this));
}

PipelineTranslator::PipelineTranslator(bool ios, u32 apple_family, bool native_mesh_,
                                       VideoCommon::StageCache* stage_cache_,
                                       StageMslCache* msl_cache_)
    : stage_cache{stage_cache_}, msl_cache{msl_cache_},
      native_mesh{native_mesh_ && NativeMeshEnabled()}, native_mesh_capable{native_mesh_} {
    enum class PreciseMode : u8 { All, Mixed, Fma, Strip };
    static const PreciseMode precise_mode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_PRECISE");
        const std::string_view mode{value != nullptr ? value : ""};
        if (mode == "all" || mode == "0") {
            return PreciseMode::All;
        }
        if (mode == "fma") {
            return PreciseMode::Fma;
        }
        if (mode == "strip") {
            return PreciseMode::Strip;
        }
        return PreciseMode::Mixed;
    }();
    // What Apple GPUs run through SPIRV-Cross (MoltenVK's defaults for Apple7+).
    profile = Shader::Profile{
        .supported_spirv = 0x00010300,
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = false,
        .support_int8 = true,
        .support_uniform_and_storage_buffer_8bit = true,
        .support_storage_buffer_8bit = true,
        .support_int16 = true,
        .support_uniform_and_storage_buffer_16bit = true,
        .support_storage_buffer_16bit = true,
        .support_int64 = true,
        .support_vertex_instance_id = false,
        .support_float_controls = false,
        .support_vote = true,
        // Metal has simd_shuffle_up/down/xor (SPIRV-Cross's ShuffleUp/Down/Xor), as MoltenVK
        // reports VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT: without, every relative shuffle
        // computed its source lane and went through the absolute shuffle.
        .support_shuffle_relative = ShuffleRelativeEnabled(),
        .supported_subgroup_stages = ~0U,
        .support_viewport_index_layer_non_geometry = true,
        .support_viewport_mask = false,
        // MSL reads a storage texture of any format (the view is the image's own, StorageHandle):
        // with false, every format-less image load was replaced by zero (compute lighting and
        // post-process passes read black).
        .support_typeless_image_loads = true,
        .support_demote_to_helper_invocation = true,
        .support_int64_atomics = false,
        .support_float32_atomic_add = true,
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = false,
        .support_native_ndc = false,
        .support_scaled_attributes = false,
        .support_multi_viewport = true,
        .support_geometry_streams = false,
        .support_sampled_image_array_nonuniform_indexing = true,
        .support_storage_image_array_nonuniform_indexing = true,
        .support_uniform_texel_buffer_array_nonuniform_indexing = true,
        .support_storage_texel_buffer_array_nonuniform_indexing = true,
        .warp_size_potentially_larger_than_guest = false,
        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .has_broken_spirv_clamp = false,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .ignore_nan_fp_comparisons = false,
        .max_subgroup_size = 32,
        .min_ssbo_alignment = 16,
        .max_user_clip_distances = 8,
        .fp_no_contraction_as_fma =
            precise_mode == PreciseMode::Mixed || precise_mode == PreciseMode::Fma,
        .fp_contraction_in_fragment_compute =
            precise_mode == PreciseMode::Mixed || precise_mode == PreciseMode::Strip,
    };
    constexpr u32 Limit = Shader::HostTranslateInfo::DEFAULT_DESCRIPTOR_LIMIT;
    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = 16,
        .max_per_stage_descriptor_sampled_images = Limit,
        .max_per_stage_resources = Limit,
        .max_per_stage_descriptor_storage_buffers = Limit,
        .max_descriptor_set_samplers = Limit,
        .max_descriptor_set_uniform_buffers = Limit,
        .max_descriptor_set_uniform_buffers_dynamic = Limit,
        .max_descriptor_set_storage_buffers = Limit,
        .max_descriptor_set_storage_buffers_dynamic = Limit,
        .max_descriptor_set_sampled_images = Limit,
        .max_descriptor_set_storage_images = Limit,
        .max_descriptor_set_input_attachements = Limit,
        .support_float64 = false,
        .support_float16 = true,
        .support_int64 = true,
        .needs_demote_reorder = false,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = true,
        // A guest geometry passthrough program stays as it is (its stores are all the fold into
        // the vertex program reads); one that can't fold is refused with the geometry stage.
        .support_geometry_shader_passthrough = true,
        // Kept, as Vulkan keeps them (MoltenVK included): with false, ConditionalBarrierPass deleted
        // every barrier inside an if or a loop, and compute passes sharing threadgroup memory
        // across loop steps (tiled lighting, reductions, blurs) raced.
        .support_conditional_barrier = true,
    };
    host_info.ApplyDescriptorLimitPolicy();
    // EDEN_NX2_DYNAMIC_HANDLE_ARRAYS: a handle read from a table at a per-invocation index becomes
    // a texture array the shader indexes, which goes to the resource ID table (spvArrayTable)
    // past Metal's sampler slots.
    host_info.handle_table_size = Shader::HostTranslateInfo::DynamicHandleArraySize();
    static std::once_flag logged_handle_arrays;
    std::call_once(logged_handle_arrays, [size = host_info.handle_table_size] {
        LOG_INFO(Render_Metal, "Dynamic handle arrays (EDEN_NX2_DYNAMIC_HANDLE_ARRAYS): {} elements{}",
                 size, size == 0 ? ", off" : "");
    });
    translate_options.ios = ios;
    translate_options.apple_family = apple_family;
    translate_options.sampler_reduction = ShaderTranslate::SamplerReductionEnabled();
    translate_options.spill_buffers = true;
}

void PipelineTranslator::PatchRasterState(std::string& msl, ShaderTranslate::Stage stage,
                                          const GraphicsPipelineKey& key) {
    static std::atomic<u32> failed{};
    const auto note = [&](const char* what) {
        if (failed.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Metal, "Metal: {} not applied to the program of pipeline {:016x}",
                        what, key.Hash());
        }
    };
    if (stage == ShaderTranslate::Stage::Fragment) {
        if (key.logic_formats != 0) {
            const RasterPatch::LogicState logic{.op = key.raster & 15, .formats = key.logic_formats};
            if (!RasterPatch::ApplyLogicOp(msl, logic)) {
                note("the logic operation");
            }
        }
        if (expand_active && !RasterExpand::PatchFragment(msl, key.expand)) {
            note("the line coordinate");
        }
    } else if ((stage == ShaderTranslate::Stage::Vertex ||
                stage == ShaderTranslate::Stage::TessellationEvaluation) &&
               ((key.raster >> 8) & 0xFFF) != 0) {
        if (!RasterPatch::ApplyViewportSwizzle(msl, (key.raster >> 8) & 0xFFF)) {
            note("the viewport swizzle");
        }
    }
}

std::optional<StageProgram> PipelineTranslator::Translate(
    Shader::IR::Program& program, ShaderTranslate::Stage stage,
    Shader::Backend::Bindings& bindings, const Shader::RuntimeInfo& runtime_info,
    const ShaderTranslate::Options* options) {
    ConvertLegacyToGeneric(program, runtime_info);
    const std::vector<u32> spirv = EmitSPIRV(profile, runtime_info, program, bindings);
    return FromSpirv(spirv, program, stage, options != nullptr ? *options : translate_options);
}

std::optional<StageProgram> PipelineTranslator::TranslateCached(
    size_t slot, Shader::IR::Program& program, ShaderTranslate::Stage stage,
    Shader::Backend::Bindings& bindings, const Shader::RuntimeInfo& runtime_info,
    const ShaderTranslate::Options* options) {
    using Cache = VideoCommon::StageCache;
    const ShaderTranslate::Options& eff_options =
        options != nullptr ? *options : translate_options;
    const Cache::Translation* const translation =
        stage_cache == nullptr || !stage_cache->Enabled() || slot >= stage_reuse.reused.size()
            ? nullptr
            : (stage_reuse.reused[slot] ? stage_reuse.reused[slot].get()
                                        : stage_reuse.recorded[slot].get());
    if (translation == nullptr) {
        return Translate(program, stage, bindings, runtime_info, &eff_options);
    }
    std::vector<u8> runtime_key = Cache::RuntimeKey(runtime_info, bindings);
    std::shared_ptr<const Cache::Emitted> emitted;
    if (stage_reuse.reused[slot]) {
        emitted = stage_cache->FindEmitted(*stage_reuse.reused[slot], runtime_key);
    }
    if (emitted && !stage_cache->Verify()) {
        program = emitted->program;
        bindings = emitted->bindings_out;
        stage_cache->CountReuse(true);
        return FromSpirv(emitted->spirv, program, stage, eff_options);
    }
    ConvertLegacyToGeneric(program, runtime_info);
    const std::vector<u32> spirv = EmitSPIRV(profile, runtime_info, program, bindings);
    if (emitted) {
        stage_cache->CountReuse(true);
    } else {
        auto entry = std::make_shared<Cache::Emitted>();
        entry->runtime_key = std::move(runtime_key);
        entry->spirv = spirv;
        entry->program = Cache::Summary(program);
        entry->bindings_out = bindings;
        if (stage_reuse.reused[slot]) {
            stage_cache->AddEmitted(*stage_reuse.reused[slot], std::move(entry));
            stage_cache->CountReuse(false);
        } else if (stage_reuse.recorded[slot]) {
            stage_reuse.new_translations.emplace_back(stage_reuse.recorded[slot],
                                                      std::move(entry));
        }
    }
    return FromSpirv(spirv, program, stage, eff_options);
}

std::optional<StageProgram> PipelineTranslator::FromSpirv(std::span<const u32> spirv,
                                                          const Shader::IR::Program& program,
                                                          ShaderTranslate::Stage stage,
                                                          const ShaderTranslate::Options& options) {
    const std::optional<u128> cache_key =
        msl_cache != nullptr ? MakeStageMslKey(spirv, stage, options) : std::nullopt;
    std::optional<ShaderTranslate::Result> cached =
        cache_key ? msl_cache->Find(*cache_key) : std::nullopt;
    ShaderTranslate::Result result =
        cached ? std::move(*cached) : ShaderTranslate::SpirvToMsl(spirv, stage, options);
    if (!cached && cache_key && !result.source.empty()) {
        msl_cache->Insert(*cache_key, result);
    }
    if (!result.source.empty() && patch_key != nullptr) {
        PatchRasterState(result.source, stage, *patch_key);
        if (!result.source_without_reduction.empty()) {
            PatchRasterState(result.source_without_reduction, stage, *patch_key);
        }
    }
    if (result.source.empty()) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Metal, "Metal: SPIR-V -> MSL failed: {}", result.error);
        }
        cause = "SPIR-V to MSL failed";
        detail = result.error;
        return std::nullopt;
    }
    return StageProgram{
        .stage = stage,
        .msl = std::move(result.source),
        .msl_without_reduction = std::move(result.source_without_reduction),
        .entry_point = std::move(result.entry_point),
        .slots = std::move(result.slots),
        .info = program.info,
        .outputs = std::move(result.outputs),
        .tessellation_slots = result.tessellation_slots,
        .reduction_slot = result.reduction_slot,
        .spill_slot = result.spill_slot,
        .sampler_spill_slot = result.sampler_spill_slot,
        .array_spill_slot = result.array_spill_slot,
        .array_spill_entries = result.array_spill_entries,
    };
}

bool PipelineTranslator::TranslatePrograms(const GraphicsPipelineKey& key,
                                           std::span<Shader::Environment* const> envs,
                                           std::span<Shader::IR::Program> programs,
                                           size_t& mesh_index) {
    const u64 hash = key.Hash();
    static std::atomic<u32> unsupported_logged{};
    pools.ReleaseContents();
    stage_reuse = {};
    const bool uses_vertex_a = key.unique_hashes[0] != 0;
    const bool use_stage_cache =
        stage_cache != nullptr && stage_cache->Enabled() && !uses_vertex_a &&
        !VisualWorkaroundEnabled() && key.unique_hashes[4] == 0;
    size_t env_index = 0;
    bool task = false;
    mesh_index = 0;
    folded_geometry = false;
    Shader::Environment* geometry_env = nullptr;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        if (env_index >= envs.size()) {
            Refuse(unsupported_logged, hash, "program without an environment");
            return false;
        }
        Shader::Environment& env = *envs[env_index++];
        const Shader::Stage stage = env.ShaderStage();
        if (stage == Shader::Stage::Task && index == 1) {
            task = true;
        } else if (stage == Shader::Stage::Mesh && (index == 1 || task) && mesh_index == 0) {
            mesh_index = index;
        } else if (stage == Shader::Stage::Task || stage == Shader::Stage::Mesh) {
            Refuse(unsupported_logged, hash, "task or mesh program out of place");
            return false;
        } else if (index == 4 && stage != Shader::Stage::Geometry) {
            Refuse(unsupported_logged, hash, "unexpected stage in the geometry slot");
            return false;
        } else if ((index == 2 && stage != Shader::Stage::TessellationControl) ||
                   (index == 3 && stage != Shader::Stage::TessellationEval)) {
            Refuse(unsupported_logged, hash, "unexpected stage in a tessellation slot");
            return false;
        }
        if (index == 4) {
            geometry_env = &env;
        }
        const u32 cfg_offset = static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
        if (use_stage_cache && stage != Shader::Stage::Task && stage != Shader::Stage::Mesh) {
            using Cache = VideoCommon::StageCache;
            std::vector<u8> signature = Cache::Signature(env, key.unique_hashes[index], index);
            stage_reuse.reused[index] = stage_cache->Find(signature, env);
            if (stage_reuse.reused[index]) {
                // Standard vertex + fragment without XFB/expand/tessellation can skip IR
                // translation completely when FindEmitted hits; if FindEmitted misses, we
                // translate lazily only then!
                const bool can_skip_ir = !task && mesh_index == 0 && key.unique_hashes[2] == 0 &&
                                         key.unique_hashes[3] == 0 && !key.xfb.Active() &&
                                         key.expand == 0 && !stage_cache->Verify();
                if (can_skip_ir) {
                    programs[index] = stage_reuse.reused[index]->program;
                    continue;
                }
                Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
                programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
            } else {
                Cache::Recorder recorder{env};
                Shader::Maxwell::Flow::CFG cfg(recorder, pools.flow_block, cfg_offset, index == 0);
                programs[index] =
                    TranslateProgram(pools.inst, pools.block, recorder, cfg, host_info);
                stage_cache->CountTranslated();
                auto translation = std::make_shared<Cache::Translation>();
                translation->signature = std::move(signature);
                translation->reads = recorder.TakeReads();
                translation->program = Cache::Summary(programs[index]);
                stage_reuse.recorded[index] = std::move(translation);
            }
        } else {
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            if (!uses_vertex_a || index != 1) {
                programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
            } else {
                auto program_vb = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
                programs[index] = MergeDualVertexPrograms(programs[0], program_vb, env, host_info);
            }
        }
        ApplyVisualWorkaround(programs[index], key.unique_hashes[index]);
        if (VideoCommon::FeatureCensus::Enabled()) {
            VideoCommon::FeatureCensus::NoteProgram(key.unique_hashes[index], programs[index]);
        }
    }
    if (key.unique_hashes[1] == 0) {
        Refuse(unsupported_logged, hash, "no vertex program");
        return false;
    }
    if (key.unique_hashes[4] != 0) {
        // A mesh pipeline has no vertex, tessellation or geometry stages to follow (Vulkan's
        // VK_EXT_mesh_shader pipelines refuse them too); a control program needs its evaluation.
        if (mesh_index != 0 || task ||
            (key.unique_hashes[2] != 0 && key.unique_hashes[3] == 0)) {
            Refuse(unsupported_logged, hash,
                   "geometry program with mesh programs or a control program alone");
            return false;
        }
        // The passthrough fold edits the vertex program: not after tessellation.
        const bool after_tess = key.unique_hashes[3] != 0;
        folded_geometry = !after_tess && programs[4].is_geometry_passthrough &&
                          FoldGeometryPassthrough(programs[1], programs[4]);
        if (!folded_geometry && programs[4].is_geometry_passthrough) {
            // The passthrough as a geometry program that emits its input primitive.
            Shader::HostTranslateInfo lowered = host_info;
            lowered.support_geometry_shader_passthrough = false;
            const u32 cfg_offset =
                static_cast<u32>(geometry_env->StartAddress() + sizeof(Shader::ProgramHeader));
            Shader::Maxwell::Flow::CFG cfg(*geometry_env, pools.flow_block, cfg_offset, false);
            programs[4] = TranslateProgram(pools.inst, pools.block, *geometry_env, cfg, lowered);
        }
    }
    if (task && mesh_index == 0) {
        Refuse(unsupported_logged, hash, "task program without a mesh program");
        return false;
    }
    return true;
}

std::optional<TranslatedPipeline> PipelineTranslator::Graphics(
    const GraphicsPipelineKey& key, std::span<Shader::Environment* const> envs) try {
    cause.clear();
    detail.clear();
    native_cause.clear();
    patch_key = &key;
    struct ClearKey {
        const GraphicsPipelineKey*& key;
        ~ClearKey() {
            key = nullptr;
        }
    } clear_key{patch_key};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    size_t mesh_index = 0;
    if (!TranslatePrograms(key, envs, programs, mesh_index)) {
        return std::nullopt;
    }
    if (key.unique_hashes[4] != 0 && !folded_geometry) {
        return GeometryPipeline(key, programs);
    }
    if (mesh_index != 0) {
        static const bool eager_fallback = [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_MESH_EAGER_FALLBACK");
            return value != nullptr && value[0] == '1';
        }();
        std::optional<NativeMesh> native;
        if (native_mesh) {
            native = NativeMeshPipeline(key, programs, mesh_index);
            if (native && !eager_fallback) {
                cause.clear();
                detail.clear();
                return TranslatedPipeline{.native_mesh = std::move(native)};
            }
            // The emulation (the fallback) from programs of its own: translating one IR program
            // to SPIR-V twice is not something the recompiler promises.
            programs = {};
            if (!TranslatePrograms(key, envs, programs, mesh_index)) {
                if (native) {
                    cause.clear();
                    detail.clear();
                    return TranslatedPipeline{.native_mesh = std::move(native)};
                }
                return std::nullopt;
            }
        } else {
            native_cause = "the device has no Metal 3 mesh pipelines (or EDEN_NX2_METAL_MESH_NATIVE=0)";
        }
        std::optional<TranslatedPipeline> emulated = Mesh(key, programs, mesh_index);
        if (!native && !emulated && !native_mesh && native_mesh_capable) {
            // The emulation is the default path; what it can't take still gets Metal's mesh
            // stage when the device has one (else the pipeline's draws were all refused).
            std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> native_programs;
            size_t native_index = 0;
            if (TranslatePrograms(key, envs, native_programs, native_index)) {
                native = NativeMeshPipeline(key, native_programs, native_index);
            }
            if (native) {
                cause.clear();
                detail.clear();
                emulated.emplace();
            }
        }
        if (!native) {
            return emulated;
        }
        if (!emulated) {
            // Metal's mesh stage takes what the emulation can't: no fallback.
            cause.clear();
            detail.clear();
            emulated.emplace();
        }
        emulated->native_mesh = std::move(native);
        return emulated;
    }
    if (key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0) {
        if (key.unique_hashes[3] == 0) {
            static std::atomic<u32> unsupported_logged{};
            Refuse(unsupported_logged, key.Hash(), "a tessellation control program without an evaluation program");
            return std::nullopt;
        }
        return Tessellated(key, programs);
    }

    Shader::Backend::Bindings bindings;
    Shader::RuntimeInfo vertex_info;
    vertex_info.previous_stage_stores.mask.set();
    vertex_info.input_topology = InputTopology(static_cast<Maxwell::PrimitiveTopology>(key.topology));
    for (size_t attribute = 0; attribute < vertex_info.generic_input_types.size(); ++attribute) {
        constexpr std::array types{Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                                   Shader::AttributeType::SignedInt,
                                   Shader::AttributeType::UnsignedInt};
        vertex_info.generic_input_types[attribute] = types[(key.attribute_types >> (attribute * 2)) & 3];
    }
    ApplyVertexState(vertex_info, key);
    // Only a draw of points takes a point size.
    ShaderTranslate::Options vertex_options = translate_options;
    vertex_options.point_size_builtin = static_cast<Maxwell::PrimitiveTopology>(key.topology) ==
                                        Maxwell::PrimitiveTopology::Points;
    std::optional<StageProgram> vertex;
    std::optional<XfbCapture> xfb;
    expand_active = false;
    const bool expands = key.expand != 0 && !key.xfb.Active();
    if (key.xfb.Active() || expands) {
        // The vertex program once more as capture kernels, from the same SPIR-V.
        ConvertLegacyToGeneric(programs[1], vertex_info);
        const std::vector<u32> spirv = EmitSPIRV(profile, vertex_info, programs[1], bindings);
        vertex = FromSpirv(spirv, programs[1], ShaderTranslate::Stage::Vertex, vertex_options);
        xfb = XfbPrograms(key, spirv, programs[1]);
        if (!xfb) {
            if (!expands) {
                vertex.reset();
            }
        } else if (expands) {
            // The generated vertex function; without one the draws stay plain.
            const std::string generated = RasterExpand::Generate(
                vertex->msl, xfb->capture[0].msl, key.expand,
                RasterExpand::PrimitiveOf(static_cast<Maxwell::PrimitiveTopology>(key.topology)),
                (key.raster >> 8) & 0xFFF);
            if (generated.empty()) {
                static std::atomic<u32> failed{};
                if (failed.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_WARNING(Render_Metal,
                                "Metal: the line and triangle expansion was not made for {:016x}",
                                key.Hash());
                }
                xfb.reset();
            } else {
                xfb->expand = true;
                xfb->render = StageProgram{.stage = ShaderTranslate::Stage::Vertex,
                                           .msl = generated,
                                           .entry_point = RasterExpand::Entry,
                                           .info = vertex->info};
                expand_active = true;
            }
        }
    } else {
        if (stage_reuse.reused[1] && programs[1].blocks.empty()) {
            const std::vector<u8> rt_key =
                VideoCommon::StageCache::RuntimeKey(vertex_info, bindings);
            if (!stage_cache->FindEmitted(*stage_reuse.reused[1], rt_key)) {
                Shader::Environment& v_env = *envs[0];
                const u32 cfg_offset =
                    static_cast<u32>(v_env.StartAddress() + sizeof(Shader::ProgramHeader));
                Shader::Maxwell::Flow::CFG cfg(v_env, pools.flow_block, cfg_offset, false);
                programs[1] = TranslateProgram(pools.inst, pools.block, v_env, cfg, host_info);
            }
        }
        vertex = TranslateCached(1, programs[1], ShaderTranslate::Stage::Vertex, bindings,
                                 vertex_info, &vertex_options);
    }
    if (!vertex) {
        return std::nullopt;
    }
    StageProgram fragment{};
    if (key.unique_hashes[5] != 0) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = programs[1].info.stores;
        fragment_info.previous_stage_legacy_stores_mapping =
            programs[1].info.legacy_stores_mapping;
        ApplyFragmentState(fragment_info, key);
        if (stage_reuse.reused[5] && programs[5].blocks.empty()) {
            const std::vector<u8> rt_key =
                VideoCommon::StageCache::RuntimeKey(fragment_info, bindings);
            if (!stage_cache->FindEmitted(*stage_reuse.reused[5], rt_key)) {
                Shader::Environment& f_env = *envs.back();
                const u32 cfg_offset =
                    static_cast<u32>(f_env.StartAddress() + sizeof(Shader::ProgramHeader));
                Shader::Maxwell::Flow::CFG cfg(f_env, pools.flow_block, cfg_offset, false);
                programs[5] = TranslateProgram(pools.inst, pools.block, f_env, cfg, host_info);
            }
        }
        auto translated =
            TranslateCached(5, programs[5], ShaderTranslate::Stage::Fragment, bindings,
                            fragment_info);
        if (!translated) {
            return std::nullopt;
        }
        fragment = std::move(*translated);
    }
    if (stage_cache != nullptr) {
        for (auto& [translation, emitted] : stage_reuse.new_translations) {
            stage_cache->Insert(std::move(translation), std::move(emitted));
        }
    }
    stage_reuse = {};
    return TranslatedPipeline{
        .vertex = std::move(*vertex),
        .fragment = std::move(fragment),
        .xfb = std::move(xfb),
    };
} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Metal, "Metal: {}", exception.what());
    cause = "shader recompiler exception";
    detail = exception.what();
    return std::nullopt;
}

namespace {

/// Bytes per element a buffer between tessellation stages needs at most for these variables
/// (MSL lays them out in structs of 16-byte aligned vectors).
[[nodiscard]] u32 OutputStride(std::span<const ShaderTranslate::InterfaceVariable> variables) {
    u32 locations = 0;
    u32 builtin_bytes = 0;
    for (const ShaderTranslate::InterfaceVariable& variable : variables) {
        if (variable.builtin == ShaderTranslate::InterfaceVariable::NoBuiltin) {
            locations = (std::max)(locations, variable.location + 1);
        } else {
            builtin_bytes += (variable.vecsize * 4 + 15) / 16 * 16;
        }
    }
    return locations * 16 + builtin_bytes + 16;
}

} // Anonymous namespace

std::optional<XfbCapture> PipelineTranslator::XfbPrograms(const GraphicsPipelineKey& key,
                                                          std::span<const u32> spirv,
                                                          const Shader::IR::Program& vertex) {
    XfbCapture xfb;
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        xfb.stride[buffer] = key.xfb.Stride(buffer);
    }
    for (size_t variant = 0; variant < XfbCapture::IndexSizes.size(); ++variant) {
        ShaderTranslate::Options options = translate_options;
        options.vertex_index_size = XfbCapture::IndexSizes[variant];
        auto program = FromSpirv(spirv, vertex, ShaderTranslate::Stage::VertexForTessellation, options);
        if (!program) {
            return std::nullopt;
        }
        program->msl += Xfb::GatherSource(key.xfb, program->entry_point, program->outputs,
                                          vertex.info.legacy_stores_mapping);
        xfb.vertex_stride = (std::max)(xfb.vertex_stride, OutputStride(program->outputs));
        xfb.capture[variant] = std::move(*program);
    }
    return xfb;
}

std::optional<TranslatedPipeline> PipelineTranslator::GeometryPipeline(
    const GraphicsPipelineKey& key, std::span<Shader::IR::Program> programs) {
    const u64 hash = key.Hash();
    static std::atomic<u32> refused_logged{};
    // After tessellation the geometry program reads the evaluation program's vertices, which the
    // tessellation's domain emulation lists as triangles, lines or points.
    const bool after_tess = key.unique_hashes[3] != 0;
    u32 input_vertices = 0;
    u32 input_step = 0;
    std::optional<Tessellation> tessellation;
    std::vector<ShaderTranslate::InterfaceVariable> feed_outputs;
    Shader::Backend::Bindings bindings;
    if (after_tess) {
        using Domain = Maxwell::Tessellation::DomainType;
        using OutputPrimitives = Maxwell::Tessellation::OutputPrimitives;
        std::optional<TranslatedPipeline> tessellated = Tessellated(key, programs, true);
        if (!tessellated) {
            return std::nullopt;
        }
        tessellation = std::move(tessellated->tessellation);
        feed_outputs = std::move(tessellated->vertex.outputs);
        bindings = feed_bindings;
        const auto domain = static_cast<Domain>(key.tessellation & 3);
        const auto output = static_cast<OutputPrimitives>((key.tessellation >> 4) & 3);
        input_vertices = output == OutputPrimitives::Points ? 1 : domain == Domain::Isolines ? 2 : 3;
        input_step = input_vertices;
    } else if (!Geometry::InputTopology(static_cast<Maxwell::PrimitiveTopology>(key.topology),
                                        input_vertices, input_step)) {
        Refuse(refused_logged, hash,
               "geometry program with an input topology the compute path can't take");
        return std::nullopt;
    }
    // The program before the geometry stage: the vertex program, or the evaluation program.
    Shader::IR::Program& vertex_program = programs[after_tess ? 3 : 1];
    Shader::IR::Program& geometry_program = programs[4];
    // The vertex program's legacy outputs become generics, which the geometry program's legacy
    // inputs read; its own legacy outputs become generics for the pixel program.
    Shader::RuntimeInfo vertex_info;
    vertex_info.previous_stage_stores.mask.set();
    for (size_t attribute = 0; attribute < vertex_info.generic_input_types.size(); ++attribute) {
        constexpr std::array types{Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                                   Shader::AttributeType::SignedInt,
                                   Shader::AttributeType::UnsignedInt};
        vertex_info.generic_input_types[attribute] =
            types[(key.attribute_types >> (attribute * 2)) & 3];
    }
    if (!after_tess) {
        ConvertLegacyToGeneric(vertex_program, vertex_info);
    }
    Shader::RuntimeInfo geometry_info;
    geometry_info.previous_stage_stores = vertex_program.info.stores;
    geometry_info.previous_stage_legacy_stores_mapping = vertex_program.info.legacy_stores_mapping;
    geometry_info.generic_input_types.fill(Shader::AttributeType::Float);
    ConvertLegacyToGeneric(geometry_program, geometry_info);
    const Shader::VaryingState geometry_stores = geometry_program.info.stores;
    const auto geometry_legacy = geometry_program.info.legacy_stores_mapping;
    std::string why;
    const std::optional<Geometry::Plan> plan =
        Geometry::Transform(geometry_program, input_vertices, input_step, why);
    if (!plan) {
        Refuse(refused_logged, hash, "geometry program the compute path can't take", why);
        return std::nullopt;
    }
    GeometryCompute geometry;
    geometry.plan = *plan;
    if (after_tess) {
        // The records kernel joins the domain emulation's kernels (its MSL defines the evaluated
        // vertex struct); the count kernel's MSL is that source.
        geometry.from_tess = true;
        const ShaderTranslate::DomainEmulation& emulation = tessellation->domain->emulation;
        geometry.tess_records = MeshKernel{
            .program = {.stage = ShaderTranslate::Stage::Compute,
                        .msl = tessellation->domain->count.program.msl +
                               Geometry::RecordsSource(*plan, "spvDomain", feed_outputs,
                                                       vertex_program.info.legacy_stores_mapping,
                                                       "spvDomainVertex"),
                        .entry_point = std::string{Geometry::RecordsEntry}},
        };
        static_cast<void>(emulation);
    } else {
        // The vertex program as capture kernels only: its depth conversion and Y flip belong to
        // the reader, which draws what the geometry program emits.
        const std::vector<u32> vertex_spirv =
            EmitSPIRV(profile, vertex_info, vertex_program, bindings);
        for (size_t variant = 0; variant < GeometryCompute::IndexSizes.size(); ++variant) {
            ShaderTranslate::Options options = translate_options;
            options.vertex_index_size = GeometryCompute::IndexSizes[variant];
            auto capture = FromSpirv(vertex_spirv, vertex_program,
                                     ShaderTranslate::Stage::VertexForTessellation, options);
            if (!capture) {
                return std::nullopt;
            }
            capture->msl += Geometry::RecordsSource(*plan, capture->entry_point, capture->outputs,
                                                    vertex_program.info.legacy_stores_mapping);
            geometry.vertex_stride =
                (std::max)(geometry.vertex_stride, OutputStride(capture->outputs));
            geometry.capture[variant] = std::move(*capture);
        }
    }
    Shader::RuntimeInfo kernel_info;
    auto kernel = Translate(geometry_program, ShaderTranslate::Stage::Compute, bindings, kernel_info);
    if (!kernel) {
        return std::nullopt;
    }
    // The extra buffers are not guest storage buffers: out of the descriptors the draw binds.
    auto& descriptors = kernel->info.storage_buffers_descriptors;
    descriptors.resize(descriptors.size() - Geometry::ExtraBuffers);
    {
        std::vector<ShaderTranslate::ResourceSlot> buffers;
        for (const ShaderTranslate::ResourceSlot& slot : kernel->slots) {
            if (slot.kind == ShaderTranslate::ResourceKind::Buffer &&
                slot.set != ShaderTranslate::PushConstantSet) {
                buffers.push_back(slot);
            }
        }
        std::ranges::sort(buffers, {}, &ShaderTranslate::ResourceSlot::binding);
        if (buffers.size() < Geometry::ExtraBuffers) {
            Refuse(refused_logged, hash, "geometry kernel without its buffers");
            return std::nullopt;
        }
        for (u32 buffer = 0; buffer < Geometry::ExtraBuffers; ++buffer) {
            geometry.extra_slots[buffer] =
                buffers[buffers.size() - Geometry::ExtraBuffers + buffer].index;
        }
    }
    if (key.xfb.Active()) {
        // The capture of the geometry output: the compacted list's vertices into the buffers.
        for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
            geometry.xfb_stride[buffer] = key.xfb.Stride(buffer);
            if (key.xfb.Stride(buffer) != 0) {
                geometry.xfb_streams |= 1U << (std::min)(key.xfb.State().layouts[buffer].stream,
                                                         Geometry::MaxStreams - 1);
            }
        }
        for (u32 stream = 0; stream < Geometry::MaxStreams; ++stream) {
            if ((geometry.xfb_streams >> stream) & 1) {
                // A stream the program never emits to captures nothing: its list is empty.
                kernel->msl += Geometry::XfbSource(*plan, key.xfb, geometry_legacy, stream);
            }
        }
    }
    geometry.kernel = MeshKernel{.program = std::move(*kernel),
                                 .workgroup_size = {Geometry::GroupSize, 1, 1}};

    StageProgram fragment{};
    if (key.unique_hashes[5] != 0) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = geometry_stores;
        fragment_info.previous_stage_legacy_stores_mapping = geometry_legacy;
        ApplyFragmentState(fragment_info, key);
        auto translated =
            Translate(programs[5], ShaderTranslate::Stage::Fragment, bindings, fragment_info);
        if (!translated) {
            return std::nullopt;
        }
        fragment = std::move(*translated);
    }
    const float point_size = FixedStateEnabled() && static_cast<Maxwell::PrimitiveTopology>(key.topology) ==
                                                         Maxwell::PrimitiveTopology::Points
                                 ? std::bit_cast<float>(key.point_size)
                                 : 1.0f;
    StageProgram reader{
        .stage = ShaderTranslate::Stage::Vertex,
        .msl = Geometry::ReaderSource(*plan, (key.state & StateDepthMinusOneToOne) != 0,
                                      translate_options.flip_vertex_y, point_size),
        .entry_point = std::string{Geometry::ReaderEntry},
    };
    reader.info.stores = geometry_stores;
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_INFO(Render_Metal, "Metal: geometry pipeline {:016x} (geometry {:016x}) through the compute "
                        "path: {} in, {} out vertices, {} input and {} output words",
                 hash, key.unique_hashes[4], input_vertices, plan->max_vertices, plan->InputStride(),
                 plan->OutputStride());
    }
    return TranslatedPipeline{
        .vertex = std::move(reader),
        .fragment = std::move(fragment),
        .tessellation = std::move(tessellation),
        .geometry = std::move(geometry),
    };
}

std::optional<StageProgram> PipelineTranslator::GeometryKernel(Shader::IR::Program& program) {
    Shader::Backend::Bindings bindings;
    try {
        return Translate(program, ShaderTranslate::Stage::Compute, bindings, Shader::RuntimeInfo{});
    } catch (const Shader::Exception& exception) {
        cause = "shader recompiler exception";
        detail = exception.what();
        return std::nullopt;
    }
}

bool PipelineTranslator::FoldGeometryPassthrough(Shader::IR::Program& vertex,
                                                 const Shader::IR::Program& geometry) {
    namespace IR = Shader::IR;
    if (!geometry.is_geometry_passthrough || vertex.info.requires_layer_emulation) {
        return false;
    }
    // What the passthrough program sets (the layer, the viewport index), each from a vertex
    // output it reads back: the value (bits) the vertex program stored there.
    struct Fold {
        IR::Attribute target;
        IR::Attribute source;
    };
    boost::container::small_vector<Fold, 2> folds;
    for (IR::Block* const block : geometry.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::Prologue:
            case IR::Opcode::Epilogue:
            case IR::Opcode::Identity:
            case IR::Opcode::BitCastF32U32:
            case IR::Opcode::BitCastU32F32:
            case IR::Opcode::GetAttribute:
            case IR::Opcode::GetAttributeU32:
                break;
            case IR::Opcode::SetAttribute: {
                const IR::Attribute target = inst.Arg(0).Attribute();
                if (target != IR::Attribute::Layer && target != IR::Attribute::ViewportIndex) {
                    return false;
                }
                // Bits only: casts down to the attribute read.
                IR::Value value = inst.Arg(1);
                const IR::Inst* leaf = nullptr;
                while (!value.IsImmediate()) {
                    leaf = value.InstRecursive();
                    if (leaf->GetOpcode() != IR::Opcode::BitCastF32U32 &&
                        leaf->GetOpcode() != IR::Opcode::BitCastU32F32) {
                        break;
                    }
                    value = leaf->Arg(0);
                }
                if (leaf == nullptr || value.IsImmediate() ||
                    (leaf->GetOpcode() != IR::Opcode::GetAttribute &&
                     leaf->GetOpcode() != IR::Opcode::GetAttributeU32)) {
                    return false;
                }
                folds.push_back({target, leaf->Arg(0).Attribute()});
                break;
            }
            default:
                // Anything else reads state the geometry stage has its own copy of (constant
                // buffers are bound per stage) or builds a value from it.
                return false;
            }
        }
    }
    if (folds.empty()) {
        return false;
    }
    // Each source must be written by exactly one store of the vertex program: the layer is set
    // there, from the same value.
    struct Store {
        IR::Block* block;
        IR::Inst* inst;
        IR::Attribute target;
    };
    boost::container::small_vector<Store, 2> stores;
    for (const Fold& fold : folds) {
        Store found{nullptr, nullptr, fold.target};
        for (IR::Block* const block : vertex.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::SetAttribute &&
                    inst.Arg(0).Attribute() == fold.source) {
                    if (found.inst != nullptr) {
                        return false;
                    }
                    found.block = block;
                    found.inst = &inst;
                }
            }
        }
        if (found.inst == nullptr) {
            return false;
        }
        stores.push_back(found);
    }
    for (const Store& store : stores) {
        IR::IREmitter ir{*store.block,
                         std::next(IR::Block::InstructionList::s_iterator_to(*store.inst))};
        ir.SetAttribute(store.target, IR::F32{store.inst->Arg(1)}, ir.Imm32(0));
        vertex.info.stores.Set(store.target, true);
    }
    return true;
}

std::optional<TranslatedPipeline> PipelineTranslator::Mesh(
    const GraphicsPipelineKey& key, std::span<Shader::IR::Program> programs, size_t mesh_index) {
    const u64 hash = key.Hash();
    static std::atomic<u32> refused_logged{};
    const bool has_task = mesh_index != 1;
    Shader::IR::Program& mesh = programs[mesh_index];
    // As Vulkan's emulation takes them: points, lines or triangles, before a pixel program only.
    const u32 topology = mesh.mesh.topology;
    if ((topology != 0 && topology != 1 && topology != 4) ||
        !Shader::Backend::SPIRV::MeshEmulationTakesPrimitiveLines(mesh.mesh) ||
        mesh.mesh.vertex_lines.empty() || mesh.mesh.max_vertices == 0 ||
        mesh.mesh.max_primitives == 0) {
        Refuse(refused_logged, hash, "mesh program the emulation can't take");
        return std::nullopt;
    }
    // The output buffer's binding follows every program's (unified bindings).
    u32 output_binding = 0;
    for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const Shader::Info& info = programs[index].info;
        output_binding += Shader::NumDescriptors(info.constant_buffer_descriptors) +
                          Shader::NumDescriptors(info.storage_buffers_descriptors) +
                          Shader::NumDescriptors(info.texture_buffer_descriptors) +
                          Shader::NumDescriptors(info.image_buffer_descriptors) +
                          Shader::NumDescriptors(info.texture_descriptors) +
                          Shader::NumDescriptors(info.image_descriptors);
    }
    const u32 payload_stride = has_task ? (std::max)(programs[1].mesh.payload_words, 2U) : 0U;
    Shader::Backend::Bindings bindings;
    Shader::RuntimeInfo kernel_info;
    kernel_info.previous_stage_stores.mask.set();
    kernel_info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
    kernel_info.mesh_as_compute = true;
    kernel_info.mesh_output_binding = output_binding;
    if (has_task) {
        kernel_info.mesh_task_payload_binding = output_binding + 1;
        kernel_info.mesh_task_offsets_binding = output_binding + 2;
        kernel_info.mesh_task_args_binding = output_binding + 3;
        kernel_info.mesh_task_payload_stride = payload_stride;
    }
    MeshEmulation emulation{
        .has_task = has_task,
        .mesh_stage = static_cast<u32>(mesh_index - 1),
        .payload_stride = payload_stride,
        .output_binding = output_binding,
        .block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh),
        .vertices_per_group = mesh.mesh.max_primitives * mesh.mesh.PrimitiveVertices(),
        .topology = topology,
    };
    // Task, then mesh, then pixel program: the bindings count up in that order.
    static const bool invariant_mesh_kernel = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_INVARIANT_POSITION");
        return value != nullptr && value[0] == '1';
    }();
    const std::array<size_t, 2> kernel_programs{1, mesh_index};
    for (size_t kernel = 0; kernel < (has_task ? 2U : 1U); ++kernel) {
        const size_t index = kernel_programs[kernel];
        ShaderTranslate::Options kernel_options = translate_options;
        kernel_options.invariant_float_math = invariant_mesh_kernel && index == mesh_index;
        auto translated = Translate(programs[index], ShaderTranslate::Stage::Compute, bindings,
                                    kernel_info, &kernel_options);
        if (!translated) {
            return std::nullopt;
        }
        MeshKernel& target = kernel == 0 ? emulation.first : emulation.mesh;
        target.program = std::move(*translated);
        target.workgroup_size = programs[index].workgroup_size;
    }
    if (has_task) {
        ShaderTranslate::Result distribution = ShaderTranslate::SpirvToMsl(
            Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile),
            ShaderTranslate::Stage::Compute, translate_options);
        if (distribution.source.empty()) {
            Refuse(refused_logged, hash, "SPIR-V to MSL failed", distribution.error);
            return std::nullopt;
        }
        emulation.distribution.program = StageProgram{
            .stage = ShaderTranslate::Stage::Compute,
            .msl = std::move(distribution.source),
            .entry_point = std::move(distribution.entry_point),
            .slots = std::move(distribution.slots),
        };
        emulation.distribution.workgroup_size = {1, 1, 1};
    }
    // The emulation's vertex shader converts the depth and sizes the points.
    const std::vector<u32> vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
        profile, mesh.mesh, output_binding, has_task ? output_binding + 3 : ~0U,
        (key.state & StateDepthMinusOneToOne) != 0,
        FixedStateEnabled() && topology == 0 ? std::bit_cast<float>(key.point_size) : 1.0f);
    ShaderTranslate::Options vertex_options = translate_options;
    vertex_options.point_size_builtin = topology == 0;
    ShaderTranslate::Result vertex =
        ShaderTranslate::SpirvToMsl(vertex_spirv, ShaderTranslate::Stage::Vertex, vertex_options);
    if (vertex.source.empty()) {
        Refuse(refused_logged, hash, "SPIR-V to MSL failed", vertex.error);
        return std::nullopt;
    }
    StageProgram fragment{};
    if (key.unique_hashes[5] != 0) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = mesh.info.stores;
        fragment_info.previous_stage_legacy_stores_mapping = mesh.info.legacy_stores_mapping;
        for (const u32 line : mesh.mesh.primitive_lines) {
            const auto attribute = static_cast<Shader::IR::Attribute>(line);
            if (Shader::IR::IsGeneric(attribute)) {
                fragment_info.per_primitive_generics.set(
                    Shader::IR::GenericAttributeIndex(attribute));
            }
        }
        fragment_info.mesh_emulation_inputs = true;
        ApplyFragmentState(fragment_info, key);
        auto translated =
            Translate(programs[5], ShaderTranslate::Stage::Fragment, bindings, fragment_info);
        if (!translated) {
            return std::nullopt;
        }
        fragment = std::move(*translated);
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_INFO(Render_Metal,
                 "Metal: mesh pipeline {:016x} (mesh {:016x}, task {}) emulated: {} vertices x {} "
                 "lines, {} primitives per workgroup, {} output words",
                 hash, key.unique_hashes[mesh_index], has_task, mesh.mesh.max_vertices,
                 mesh.mesh.vertex_lines.size(), mesh.mesh.max_primitives, emulation.block_words);
    }
    StageProgram vertex_program{
        .stage = ShaderTranslate::Stage::Vertex,
        .msl = std::move(vertex.source),
        .entry_point = std::move(vertex.entry_point),
        .slots = std::move(vertex.slots),
    };
    return TranslatedPipeline{
        .vertex = std::move(vertex_program),
        .fragment = std::move(fragment),
        .mesh = std::move(emulation),
    };
}

bool NativeMeshEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_METAL_MESH_NATIVE");
        if (value == nullptr) {
            value = std::getenv("EDEN_NX2_METAL_NATIVE_MESH");
        }
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

bool NativeMeshPerPrimitive() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MESH_PER_PRIMITIVE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

namespace {

/// Generic vec4 outputs, plus one for the built-ins, of a mesh program's output lines.
template <typename Lines>
[[nodiscard]] u32 OutputLocations(const Lines& lines) {
    std::bitset<Shader::IR::NUM_GENERICS> generics;
    bool builtins = false;
    for (const u32 line : lines) {
        const auto attribute = static_cast<Shader::IR::Attribute>(line);
        if (Shader::IR::IsGeneric(attribute)) {
            generics.set(Shader::IR::GenericAttributeIndex(attribute));
        } else {
            builtins = true;
        }
    }
    return static_cast<u32>(generics.count()) + (builtins ? 1U : 0U);
}

[[nodiscard]] u32 MeshThreadgroupBytes(const Shader::IR::Program& mesh) {
    const Shader::MeshInfo& info = mesh.mesh;
    return static_cast<u32>((std::min)(
        Vulkan::MetalMeshThreadgroupBytes({
            .max_vertices = info.max_vertices,
            .max_primitives = info.max_primitives,
            .primitive_vertices = info.PrimitiveVertices(),
            .vertex_lines = static_cast<u32>(info.vertex_lines.size()),
            .primitive_lines = static_cast<u32>(info.primitive_lines.size()),
            .vertex_locations = OutputLocations(info.vertex_lines),
            .primitive_locations = OutputLocations(info.primitive_lines),
            .shared_memory_bytes = mesh.shared_memory_size,
            .direct_outputs = true,
        }),
        u64{~0U}));
}

[[nodiscard]] bool WritesPerPrimitiveGenerics(const Shader::IR::Program& mesh) {
    return std::ranges::any_of(mesh.mesh.primitive_lines, [](u32 line) {
        return Shader::IR::IsGeneric(static_cast<Shader::IR::Attribute>(line));
    });
}

} // Anonymous namespace

std::optional<std::string> NativeMeshLimit(const Shader::IR::Program* task,
                                           const Shader::IR::Program& mesh) {
    using Limits = NativeMeshLimits;
    const Shader::MeshInfo& info = mesh.mesh;
    if (info.topology != 0 && info.topology != 1 && info.topology != 4) {
        return fmt::format("output topology {}", info.topology);
    }
    if (info.max_vertices == 0 || info.max_primitives == 0) {
        return std::string{"no vertices or primitives"};
    }
    if (info.max_vertices > Limits::MaxVertices || info.max_primitives > Limits::MaxPrimitives) {
        return fmt::format("{} vertices / {} primitives, Metal {} / {}", info.max_vertices,
                           info.max_primitives, Limits::MaxVertices, Limits::MaxPrimitives);
    }
    if (mesh.workgroup_size[0] > Limits::MaxThreads ||
        (task != nullptr && task->workgroup_size[0] > Limits::MaxThreads)) {
        return std::string{"threads per threadgroup"};
    }
    if (task != nullptr && task->mesh.payload_words * 4 > Limits::MaxPayloadBytes) {
        return fmt::format("payload {} B, Metal {} B", task->mesh.payload_words * 4,
                           Limits::MaxPayloadBytes);
    }
    if (task != nullptr && task->shared_memory_size > Limits::MaxThreadgroupBytes) {
        return fmt::format("object threadgroup memory {} B, Metal {} B", task->shared_memory_size,
                           Limits::MaxThreadgroupBytes);
    }
    if (OutputLocations(info.vertex_lines) * 4 > Limits::MaxVertexComponents) {
        return fmt::format("{} output components per vertex, Metal {}",
                           OutputLocations(info.vertex_lines) * 4, Limits::MaxVertexComponents);
    }
    if (const u32 bytes = MeshThreadgroupBytes(mesh); bytes > Limits::MaxThreadgroupBytes) {
        return fmt::format("mesh threadgroup memory {} B, Metal {} B", bytes,
                           Limits::MaxThreadgroupBytes);
    }
    return std::nullopt;
}

std::optional<NativeMesh> PipelineTranslator::NativeMeshPipeline(
    const GraphicsPipelineKey& key, std::span<Shader::IR::Program> programs,
    size_t mesh_index) try {
    const bool has_task = mesh_index != 1;
    Shader::IR::Program& mesh = programs[mesh_index];
    Shader::IR::Program* const task = has_task ? &programs[1] : nullptr;
    for (size_t index = 2; index < 5; ++index) {
        if (key.unique_hashes[index] != 0 && index != mesh_index) {
            native_cause = "stages besides task, mesh and pixel programs";
            return std::nullopt;
        }
    }
    if (auto limit = NativeMeshLimit(task, mesh)) {
        native_cause = std::move(*limit);
        return std::nullopt;
    }
    if (WritesPerPrimitiveGenerics(mesh) && !NativeMeshPerPrimitive()) {
        native_cause = "writes per-primitive attributes (emulated, as Vulkan on MoltenVK; "
                       "EDEN_NX2_METAL_MESH_PER_PRIMITIVE=1 builds it natively)";
        return std::nullopt;
    }
    const auto keep_cause = [this, saved_cause = cause, saved_detail = detail] {
        native_cause = cause.empty() ? "translation failed" : cause;
        if (!detail.empty()) {
            native_cause += ": " + detail;
        }
        cause = saved_cause;
        detail = saved_detail;
    };
    Shader::Backend::Bindings bindings;
    NativeMesh native{
        .mesh_stage = static_cast<u32>(mesh_index - 1),
        .payload_bytes = has_task ? task->mesh.payload_words * 4 : 0U,
        .object_threads = has_task ? (std::max)(task->workgroup_size[0], 1U) : 1U,
        .mesh_threads = (std::max)(mesh.workgroup_size[0], 1U),
        .topology = mesh.mesh.topology,
        .threadgroup_bytes = MeshThreadgroupBytes(mesh),
    };
    // As Vulkan's MakeRuntimeInfo for native task and mesh programs; the mesh program writes its
    // outputs directly (RuntimeInfo::mesh_direct_outputs), as on MoltenVK.
    if (has_task) {
        Shader::RuntimeInfo task_info;
        task_info.previous_stage_stores.mask.set();
        task_info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
        auto object = Translate(*task, ShaderTranslate::Stage::Object, bindings, task_info);
        if (!object) {
            keep_cause();
            return std::nullopt;
        }
        native.object = std::move(*object);
    }
    Shader::RuntimeInfo mesh_info;
    mesh_info.previous_stage_stores.mask.set();
    mesh_info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
    mesh_info.convert_depth_mode = (key.state & StateDepthMinusOneToOne) != 0;
    if (FixedStateEnabled() && mesh.mesh.topology == 0) {
        mesh_info.fixed_state_point_size = std::bit_cast<float>(key.point_size);
    }
    mesh_info.mesh_direct_outputs = true;
    auto mesh_function = Translate(mesh, ShaderTranslate::Stage::Mesh, bindings, mesh_info);
    if (!mesh_function) {
        keep_cause();
        return std::nullopt;
    }
    native.mesh = std::move(*mesh_function);
    if (key.unique_hashes[5] != 0) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = mesh.info.stores;
        fragment_info.previous_stage_legacy_stores_mapping = mesh.info.legacy_stores_mapping;
        for (const u32 line : mesh.mesh.primitive_lines) {
            const auto attribute = static_cast<Shader::IR::Attribute>(line);
            if (Shader::IR::IsGeneric(attribute)) {
                fragment_info.per_primitive_generics.set(
                    Shader::IR::GenericAttributeIndex(attribute));
            }
        }
        // Constant over the primitive and compared bit for bit by some pixel programs (Vulkan's
        // FlatPerPrimitiveInputs on MoltenVK).
        fragment_info.flat_per_primitive_inputs = true;
        ApplyFragmentState(fragment_info, key);
        auto fragment =
            Translate(programs[5], ShaderTranslate::Stage::Fragment, bindings, fragment_info);
        if (!fragment) {
            keep_cause();
            return std::nullopt;
        }
        native.fragment = std::move(*fragment);
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_INFO(Render_Metal,
                 "Metal: mesh pipeline {:016x} (mesh {:016x}, task {}) native: {} vertices, {} "
                 "primitives, {} + {} threads, payload {} B, ~{} B threadgroup memory",
                 key.Hash(), key.unique_hashes[mesh_index], has_task, mesh.mesh.max_vertices,
                 mesh.mesh.max_primitives, native.object_threads, native.mesh_threads,
                 native.payload_bytes, native.threadgroup_bytes);
    }
    return native;
} catch (const Shader::Exception& exception) {
    native_cause = fmt::format("shader recompiler exception: {}", exception.what());
    return std::nullopt;
}

namespace {

/// Bytes per element a buffer between tessellation stages needs at most for these variables
/// (MSL lays them out in structs of 16-byte aligned vectors).
[[nodiscard]] u32 InterfaceStride(std::span<const ShaderTranslate::InterfaceVariable> variables,
                                  bool per_patch) {
    u32 locations = 0;
    u32 builtin_bytes = 0;
    for (const ShaderTranslate::InterfaceVariable& variable : variables) {
        if (variable.per_patch != per_patch) {
            continue;
        }
        if (variable.builtin == ShaderTranslate::InterfaceVariable::NoBuiltin) {
            locations = (std::max)(locations, variable.location + 1);
        } else {
            builtin_bytes += (variable.vecsize * 4 + 15) / 16 * 16;
        }
    }
    return locations * 16 + builtin_bytes + 16;
}

} // Anonymous namespace

std::optional<TranslatedPipeline> PipelineTranslator::Tessellated(
    const GraphicsPipelineKey& key, std::span<Shader::IR::Program> programs,
    bool feeds_geometry) {
    const u64 hash = key.Hash();
    static std::atomic<u32> refused_logged{};
    using Domain = Maxwell::Tessellation::DomainType;
    using Spacing = Maxwell::Tessellation::Spacing;
    using OutputPrimitives = Maxwell::Tessellation::OutputPrimitives;
    const auto domain = static_cast<Domain>(key.tessellation & 3);
    const auto spacing = static_cast<Spacing>((key.tessellation >> 2) & 3);
    const auto output = static_cast<OutputPrimitives>((key.tessellation >> 4) & 3);
    Shader::IR::Program& control = programs[2];
    // Without a control program the hardware takes the factors from registers and hands the
    // vertex outputs to the evaluation program as the patch's control points.
    const bool has_control = key.unique_hashes[2] != 0;
    const u32 control_points = has_control ? control.invocations
                                           : ((key.tessellation >> TessPatchShift) & 31U) + 1;
    if (control_points == 0 || control_points > 32) {
        Refuse(refused_logged, hash, "tessellation control program's output vertex count");
        return std::nullopt;
    }
    using Modes = ShaderTranslate::TessellationModes;
    Tessellation tessellation{
        .modes =
            {
                .primitive = domain == Domain::Quads      ? Modes::Primitive::Quads
                             : domain == Domain::Isolines ? Modes::Primitive::Isolines
                                                          : Modes::Primitive::Triangles,
                .spacing = spacing == Spacing::FractionalOdd    ? Modes::Spacing::FractionalOdd
                           : spacing == Spacing::FractionalEven ? Modes::Spacing::FractionalEven
                                                                : Modes::Spacing::Equal,
                .clockwise = output == OutputPrimitives::Triangles_CW,
                .point_mode = output == OutputPrimitives::Points,
                .output_vertices = control_points,
            },
    };
    Shader::Backend::Bindings bindings;

    // The vertex program as a kernel, once per index size (the same SPIR-V and bindings).
    Shader::RuntimeInfo vertex_info;
    vertex_info.previous_stage_stores.mask.set();
    for (size_t attribute = 0; attribute < vertex_info.generic_input_types.size(); ++attribute) {
        constexpr std::array types{Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                                   Shader::AttributeType::SignedInt,
                                   Shader::AttributeType::UnsignedInt};
        vertex_info.generic_input_types[attribute] =
            types[(key.attribute_types >> (attribute * 2)) & 3];
    }
    ApplyVertexState(vertex_info, key);
    vertex_info.convert_depth_mode = false;
    ConvertLegacyToGeneric(programs[1], vertex_info);
    const std::vector<u32> vertex_spirv = EmitSPIRV(profile, vertex_info, programs[1], bindings);
    for (size_t variant = 0; variant < Tessellation::IndexSizes.size(); ++variant) {
        ShaderTranslate::Options options = translate_options;
        options.vertex_index_size = Tessellation::IndexSizes[variant];
        ShaderTranslate::Result result = ShaderTranslate::SpirvToMsl(
            vertex_spirv, ShaderTranslate::Stage::VertexForTessellation, options);
        if (result.source.empty()) {
            Refuse(refused_logged, hash, "SPIR-V to MSL failed", result.error);
            return std::nullopt;
        }
        tessellation.vertex[variant] = StageProgram{
            .stage = ShaderTranslate::Stage::VertexForTessellation,
            .msl = std::move(result.source),
            .msl_without_reduction = std::move(result.source_without_reduction),
            .entry_point = std::move(result.entry_point),
            .slots = std::move(result.slots),
            .info = programs[1].info,
            .outputs = std::move(result.outputs),
            .tessellation_slots = result.tessellation_slots,
            .reduction_slot = result.reduction_slot,
            .spill_slot = result.spill_slot,
            .sampler_spill_slot = result.sampler_spill_slot,
            .array_spill_slot = result.array_spill_slot,
            .array_spill_entries = result.array_spill_entries,
        };
    }
    const StageProgram& vertex = tessellation.vertex[0];

    std::optional<StageProgram> control_program;
    if (has_control) {
        ShaderTranslate::Options control_options = translate_options;
        control_options.inputs = vertex.outputs;
        control_options.tessellation = tessellation.modes;
        Shader::RuntimeInfo control_info;
        control_info.previous_stage_stores = programs[1].info.stores;
        control_info.previous_stage_legacy_stores_mapping = programs[1].info.legacy_stores_mapping;
        control_info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
        control_program = Translate(control, ShaderTranslate::Stage::TessellationControl, bindings,
                                    control_info, &control_options);
        if (!control_program) {
            return std::nullopt;
        }
    }
    const Shader::IR::Program& before_evaluation = has_control ? control : programs[1];

    ShaderTranslate::Options evaluation_options = translate_options;
    evaluation_options.inputs = has_control ? control_program->outputs : vertex.outputs;
    evaluation_options.tessellation = tessellation.modes;
    const bool emulated_domain =
        domain == Domain::Isolines || tessellation.modes.point_mode || feeds_geometry;
    // Only points take a point size (the emulated point mode writes the fixed state's); a
    // geometry program may read it whatever follows.
    evaluation_options.point_size_builtin = tessellation.modes.point_mode || feeds_geometry;
    // A geometry program reads the evaluation's own outputs: the reader flips and converts.
    evaluation_options.flip_vertex_y = !feeds_geometry && evaluation_options.flip_vertex_y;
    Shader::RuntimeInfo evaluation_info;
    if (FixedStateEnabled() && tessellation.modes.point_mode) {
        evaluation_info.fixed_state_point_size = std::bit_cast<float>(key.point_size);
    }
    evaluation_info.previous_stage_stores = before_evaluation.info.stores;
    evaluation_info.previous_stage_legacy_stores_mapping =
        before_evaluation.info.legacy_stores_mapping;
    evaluation_info.tess_primitive = domain == Domain::Quads      ? Shader::TessPrimitive::Quads
                                     : domain == Domain::Isolines ? Shader::TessPrimitive::Isolines
                                                                  : Shader::TessPrimitive::Triangles;
    evaluation_info.tess_spacing = spacing == Spacing::FractionalOdd ? Shader::TessSpacing::FractionalOdd
                                   : spacing == Spacing::FractionalEven
                                       ? Shader::TessSpacing::FractionalEven
                                       : Shader::TessSpacing::Equal;
    evaluation_info.tess_clockwise = tessellation.modes.clockwise;
    evaluation_info.y_negate = !feeds_geometry && (key.state & StateLowerLeftOrigin) != 0;
    evaluation_info.convert_depth_mode =
        !feeds_geometry && (key.state & StateDepthMinusOneToOne) != 0;
    // The raster state patches are for the stage the rasterizer sees.
    const GraphicsPipelineKey* const saved_patch_key = patch_key;
    if (feeds_geometry) {
        patch_key = nullptr;
    }
    auto evaluation = Translate(programs[3], ShaderTranslate::Stage::TessellationEvaluation,
                                bindings, evaluation_info, &evaluation_options);
    patch_key = saved_patch_key;
    if (!evaluation) {
        return std::nullopt;
    }

    StageProgram fragment{};
    // After a geometry program the pixel program is translated with it.
    if (key.unique_hashes[5] != 0 && !feeds_geometry) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = programs[3].info.stores;
        fragment_info.previous_stage_legacy_stores_mapping = programs[3].info.legacy_stores_mapping;
        ApplyFragmentState(fragment_info, key);
        auto translated =
            Translate(programs[5], ShaderTranslate::Stage::Fragment, bindings, fragment_info);
        if (!translated) {
            return std::nullopt;
        }
        fragment = std::move(*translated);
    }
    tessellation.vertex_stride = InterfaceStride(vertex.outputs, false);
    if (has_control) {
        tessellation.control_point_stride = InterfaceStride(control_program->outputs, false);
        tessellation.patch_stride = InterfaceStride(control_program->outputs, true);
        tessellation.control = MeshKernel{
            .program = std::move(*control_program),
            .workgroup_size = {control.invocations, 1, 1},
        };
    } else {
        tessellation.control_point_stride = tessellation.vertex_stride;
        tessellation.factors_from_registers = true;
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_INFO(Render_Metal,
                 "Metal: tessellation pipeline {:016x} (control {:016x}, evaluation {:016x}): "
                 "domain {}, spacing {}, output primitives {}, {} control points {}",
                 hash, key.unique_hashes[2], key.unique_hashes[3], static_cast<u32>(domain),
                 static_cast<u32>(spacing), static_cast<u32>(output), control_points,
                 has_control ? "out" : "in, factors from registers");
    }
    if (emulated_domain) {
        ShaderTranslate::DomainEmulation emulation = ShaderTranslate::EmulateDomain(
            evaluation->msl, evaluation->entry_point, tessellation.modes, feeds_geometry);
        if (!emulation.error.empty()) {
            Refuse(refused_logged, hash, "tessellation domain emulation", emulation.error);
            return std::nullopt;
        }
        const auto kernel = [&](std::string entry) {
            return MeshKernel{.program = {.stage = ShaderTranslate::Stage::Compute,
                                          .msl = emulation.source,
                                          .entry_point = std::move(entry)}};
        };
        Tessellation::Domain domain_kernels{
            .count = kernel(emulation.count_entry),
            .evaluate = kernel(emulation.evaluate_entry),
        };
        // The kernel is the evaluation program: its resources, buffers and information.
        domain_kernels.evaluate.program.slots = evaluation->slots;
        domain_kernels.evaluate.program.info = evaluation->info;
        domain_kernels.evaluate.program.tessellation_slots = evaluation->tessellation_slots;
        domain_kernels.evaluate.program.reduction_slot = evaluation->reduction_slot;
        domain_kernels.evaluate.program.spill_slot = evaluation->spill_slot;
        domain_kernels.evaluate.program.sampler_spill_slot = evaluation->sampler_spill_slot;
        domain_kernels.evaluate.program.array_spill_slot = evaluation->array_spill_slot;
        domain_kernels.evaluate.program.array_spill_entries = evaluation->array_spill_entries;
        evaluation->msl = std::move(emulation.source);
        evaluation->entry_point = emulation.vertex_entry;
        evaluation->slots.clear();
        evaluation->tessellation_slots = {};
        evaluation->reduction_slot = ~0U;
        evaluation->spill_slot = ~0U;
        evaluation->sampler_spill_slot = ~0U;
        evaluation->array_spill_slot = ~0U;
        evaluation->array_spill_entries = 0;
        emulation.source.clear();
        emulation.vertex_entry.clear();
        domain_kernels.emulation = std::move(emulation);
        tessellation.domain = std::move(domain_kernels);
    }
    feed_bindings = bindings;
    return TranslatedPipeline{
        .vertex = std::move(*evaluation),
        .fragment = std::move(fragment),
        .tessellation = std::move(tessellation),
    };
}

std::optional<StageProgram> PipelineTranslator::Compute(u64 unique_hash,
                                                        Shader::Environment& env) try {
    cause.clear();
    detail.clear();
    pools.ReleaseContents();
    using Cache = VideoCommon::StageCache;
    const bool use_stage_cache =
        stage_cache != nullptr && stage_cache->Enabled() && !VisualWorkaroundEnabled();
    if (use_stage_cache) {
        std::vector<u8> signature = Cache::Signature(env, unique_hash, 5);
        std::shared_ptr<const Cache::Translation> reused = stage_cache->Find(signature, env);
        const Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings bindings{};
        std::vector<u8> runtime_key = Cache::RuntimeKey(runtime_info, bindings);
        if (reused && !stage_cache->Verify()) {
            if (const auto emitted = stage_cache->FindEmitted(*reused, runtime_key)) {
                stage_cache->CountReuse(true);
                return FromSpirv(emitted->spirv, emitted->program, ShaderTranslate::Stage::Compute,
                                 translate_options);
            }
        }
        Cache::Recorder recorder{env};
        Shader::Environment& active_env = reused ? env : static_cast<Shader::Environment&>(recorder);
        Shader::Maxwell::Flow::CFG cfg{active_env, pools.flow_block, env.StartAddress()};
        auto program = TranslateProgram(pools.inst, pools.block, active_env, cfg, host_info);
        if (!reused) {
            stage_cache->CountTranslated();
        }
        if (VideoCommon::FeatureCensus::Enabled()) {
            VideoCommon::FeatureCensus::NoteProgram(unique_hash, program);
        }
        const std::vector<u32> spirv = EmitSPIRV(profile, program);
        auto entry = std::make_shared<Cache::Emitted>();
        entry->runtime_key = std::move(runtime_key);
        entry->spirv = spirv;
        entry->program = Cache::Summary(program);
        entry->bindings_out = bindings;
        if (reused) {
            stage_cache->AddEmitted(*reused, std::move(entry));
            stage_cache->CountReuse(false);
        } else {
            auto translation = std::make_shared<Cache::Translation>();
            translation->signature = std::move(signature);
            translation->reads = recorder.TakeReads();
            translation->program = entry->program;
            stage_cache->Insert(std::move(translation), std::move(entry));
        }
        return FromSpirv(spirv, program, ShaderTranslate::Stage::Compute, translate_options);
    }

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};
    auto program = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
    ApplyVisualWorkaround(program, unique_hash);
    if (VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteProgram(unique_hash, program);
    }
    const std::vector<u32> spirv = EmitSPIRV(profile, program);
    return FromSpirv(spirv, program, ShaderTranslate::Stage::Compute, translate_options);
} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Metal, "Metal: {}", exception.what());
    cause = "shader recompiler exception";
    detail = exception.what();
    return std::nullopt;
}

std::optional<StageProgram> PipelineTranslator::WarmSingleStage(const GraphicsPipelineKey& key,
                                                                size_t slot,
                                                                Shader::Environment& env) try {
    if (slot != 1 && slot != 5) {
        return std::nullopt;
    }
    cause.clear();
    detail.clear();
    pools.ReleaseContents();
    stage_reuse = {};
    patch_key = &key;
    expand_active = false;
    struct ClearKey {
        const GraphicsPipelineKey*& key;
        ~ClearKey() {
            key = nullptr;
        }
    } clear_key{patch_key};

    const u64 unique_hash = key.unique_hashes[slot];
    const Shader::Stage guest_stage = env.ShaderStage();
    if ((slot == 1 && guest_stage != Shader::Stage::VertexB) ||
        (slot == 5 && guest_stage != Shader::Stage::Fragment)) {
        return std::nullopt;
    }

    Shader::RuntimeInfo runtime_info;
    ShaderTranslate::Options options = translate_options;
    const ShaderTranslate::Stage msl_stage =
        slot == 1 ? ShaderTranslate::Stage::Vertex : ShaderTranslate::Stage::Fragment;
    if (slot == 1) {
        runtime_info.previous_stage_stores.mask.set();
        runtime_info.input_topology =
            InputTopology(static_cast<Maxwell::PrimitiveTopology>(key.topology));
        for (size_t attr = 0; attr < runtime_info.generic_input_types.size(); ++attr) {
            constexpr std::array types{Shader::AttributeType::Disabled,
                                       Shader::AttributeType::Float,
                                       Shader::AttributeType::SignedInt,
                                       Shader::AttributeType::UnsignedInt};
            runtime_info.generic_input_types[attr] = types[(key.attribute_types >> (attr * 2)) & 3];
        }
        ApplyVertexState(runtime_info, key);
        options.point_size_builtin =
            static_cast<Maxwell::PrimitiveTopology>(key.topology) ==
            Maxwell::PrimitiveTopology::Points;
    } else {
        runtime_info.previous_stage_stores.mask.set();
        ApplyFragmentState(runtime_info, key);
    }

    Shader::Backend::Bindings bindings{};
    const u32 cfg_offset = static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
    using Cache = VideoCommon::StageCache;
    if (stage_cache != nullptr && stage_cache->Enabled() && !VisualWorkaroundEnabled()) {
        std::vector<u8> signature = Cache::Signature(env, unique_hash, slot);
        stage_reuse.reused[slot] = stage_cache->Find(signature, env);
        const std::vector<u8> rt_key = Cache::RuntimeKey(runtime_info, bindings);
        if (stage_reuse.reused[slot] && !stage_cache->Verify() &&
            stage_cache->FindEmitted(*stage_reuse.reused[slot], rt_key)) {
            Shader::IR::Program summary = stage_reuse.reused[slot]->program;
            return TranslateCached(slot, summary, msl_stage, bindings, runtime_info, &options);
        }
        Cache::Recorder recorder{env};
        Shader::Environment& active_env =
            stage_reuse.reused[slot] ? env : static_cast<Shader::Environment&>(recorder);
        Shader::Maxwell::Flow::CFG cfg(active_env, pools.flow_block, cfg_offset, false);
        Shader::IR::Program program =
            TranslateProgram(pools.inst, pools.block, active_env, cfg, host_info);
        if (!stage_reuse.reused[slot]) {
            stage_cache->CountTranslated();
            auto translation = std::make_shared<Cache::Translation>();
            translation->signature = std::move(signature);
            translation->reads = recorder.TakeReads();
            translation->program = Cache::Summary(program);
            stage_reuse.recorded[slot] = std::move(translation);
        }
        auto out = TranslateCached(slot, program, msl_stage, bindings, runtime_info, &options);
        for (auto& [translation, emitted] : stage_reuse.new_translations) {
            stage_cache->Insert(std::move(translation), std::move(emitted));
        }
        stage_reuse = {};
        return out;
    }

    Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, false);
    Shader::IR::Program program = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
    ApplyVisualWorkaround(program, unique_hash);
    return Translate(program, msl_stage, bindings, runtime_info, &options);
} catch (const Shader::Exception&) {
    return std::nullopt;
}

} // namespace Metal
