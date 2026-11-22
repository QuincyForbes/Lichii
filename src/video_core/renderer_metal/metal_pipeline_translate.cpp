// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <boost/container/small_vector.hpp>

#include "common/cityhash.h"
#include "common/fs/file.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/zstd_compression.h"
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

static_assert(std::is_trivially_copyable_v<ShaderTranslate::ResourceSlot>);
static_assert(std::is_trivially_copyable_v<ShaderTranslate::TessellationSlots>);

constexpr std::array<char, 8> StageMslMagic{'M', 'T', 'L', 'S', 'T', 'G', '0', '2'};
constexpr u32 StageMslVersion = 4;
constexpr u32 MaxMslSourceBytes = 16U << 20;
constexpr u32 MaxStageRecordBytes = 40U << 20;

struct StageFileHeader {
    std::array<char, 8> magic;
    u32 version;
    u32 reserved;
};
static_assert(sizeof(StageFileHeader) == 16);

struct StageRecordHeader {
    u128 key;
    u32 uncompressed_size;
    u32 compressed_size;
};
static_assert(sizeof(StageRecordHeader) == 24);

[[nodiscard]] u32 StageStampTag(u64 build_stamp) noexcept {
    if (build_stamp == 0) {
        return 0;
    }
    const u32 hashed = static_cast<u32>(
        Common::CityHash64(reinterpret_cast<const char*>(&build_stamp), sizeof(build_stamp)));
    return hashed != 0 ? hashed : 1U;
}

[[nodiscard]] std::vector<u8> SerializeStageResult(const ShaderTranslate::Result& res) {
    std::vector<u8> buf;
    buf.reserve(res.source.size() + res.source_without_reduction.size() + res.entry_point.size() +
                res.slots.size() * sizeof(ShaderTranslate::ResourceSlot) +
                res.outputs.size() * 32 + 64);
    const auto put_bytes = [&](const void* src, size_t len) {
        if (len != 0) {
            const auto* const ptr = static_cast<const u8*>(src);
            buf.insert(buf.end(), ptr, ptr + len);
        }
    };
    const auto put_u32 = [&](u32 val) { put_bytes(&val, sizeof(val)); };
    const auto put_str = [&](std::string_view str) {
        put_u32(static_cast<u32>(str.size()));
        put_bytes(str.data(), str.size());
    };

    put_str(res.source);
    put_str(res.source_without_reduction);
    put_str(res.entry_point);

    put_u32(static_cast<u32>(res.slots.size()));
    put_bytes(res.slots.data(), res.slots.size() * sizeof(ShaderTranslate::ResourceSlot));

    put_u32(static_cast<u32>(res.outputs.size()));
    for (const ShaderTranslate::InterfaceVariable& var : res.outputs) {
        put_u32(var.location);
        put_u32(var.builtin);
        put_u32(var.vecsize);
        const u8 per_patch = var.per_patch ? 1U : 0U;
        put_bytes(&per_patch, sizeof(per_patch));
        put_str(var.name);
    }

    put_bytes(&res.tessellation_slots, sizeof(res.tessellation_slots));
    put_u32(res.reduction_slot);
    put_u32(res.spill_slot);
    put_u32(res.sampler_spill_slot);
    put_u32(res.array_spill_slot);
    put_u32(res.array_spill_entries);
    put_u32(res.immutable_buffer_mask);
    return buf;
}

[[nodiscard]] std::optional<ShaderTranslate::Result> DeserializeStageResult(
    std::span<const u8> data) {
    size_t cursor = 0;
    const auto get_bytes = [&](void* dst, size_t len) -> bool {
        if (len > data.size() - cursor) {
            return false;
        }
        if (len != 0) {
            std::memcpy(dst, data.data() + cursor, len);
            cursor += len;
        }
        return true;
    };
    const auto get_u32 = [&](u32& out) -> bool { return get_bytes(&out, sizeof(out)); };
    const auto get_str = [&](std::string& out, u32 max_len) -> bool {
        u32 len{};
        if (!get_u32(len) || len > max_len || len > data.size() - cursor) {
            return false;
        }
        out.assign(reinterpret_cast<const char*>(data.data() + cursor), len);
        cursor += len;
        return true;
    };

    ShaderTranslate::Result res{};
    u32 num_slots{};
    u32 num_outputs{};
    if (!get_str(res.source, MaxMslSourceBytes) ||
        !get_str(res.source_without_reduction, MaxMslSourceBytes) ||
        !get_str(res.entry_point, 4096) || !get_u32(num_slots) || num_slots > 4096) {
        return std::nullopt;
    }
    res.slots.resize(num_slots);
    if (!get_bytes(res.slots.data(), num_slots * sizeof(ShaderTranslate::ResourceSlot))) {
        return std::nullopt;
    }
    if (!get_u32(num_outputs) || num_outputs > 4096) {
        return std::nullopt;
    }
    res.outputs.resize(num_outputs);
    for (ShaderTranslate::InterfaceVariable& var : res.outputs) {
        u8 per_patch{};
        if (!get_u32(var.location) || !get_u32(var.builtin) || !get_u32(var.vecsize) ||
            !get_bytes(&per_patch, sizeof(per_patch)) || !get_str(var.name, 4096)) {
            return std::nullopt;
        }
        var.per_patch = per_patch != 0;
    }
    if (!get_bytes(&res.tessellation_slots, sizeof(res.tessellation_slots)) ||
        !get_u32(res.reduction_slot) || !get_u32(res.spill_slot) ||
        !get_u32(res.sampler_spill_slot) || !get_u32(res.array_spill_slot) ||
        !get_u32(res.array_spill_entries) || !get_u32(res.immutable_buffer_mask) ||
        cursor != data.size() || res.source.empty()) {
        return std::nullopt;
    }
    return res;
}

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

bool StageMslCache::HasValidEntries(const std::filesystem::path& file_path, u64 build_stamp) {
    if (!StageMslCacheEnabled()) {
        return true;
    }
    Common::FS::IOFile file{file_path, Common::FS::FileAccessMode::Read,
                            Common::FS::FileType::BinaryFile,
                            Common::FS::FileShareFlag::ShareReadWrite};
    if (!file.IsOpen()) {
        return false;
    }
    const u64 file_size = file.GetSize();
    if (file_size <= sizeof(StageFileHeader) + sizeof(StageRecordHeader)) {
        return false;
    }
    StageFileHeader hdr{};
    if (!file.ReadObject(hdr) || hdr.magic != StageMslMagic || hdr.version != StageMslVersion) {
        return false;
    }
    return build_stamp == 0 || hdr.reserved == StageStampTag(build_stamp);
}

void StageMslCache::Open(const std::filesystem::path& file_path, u64 build_stamp) {
    if (!StageMslCacheEnabled()) {
        return;
    }
    std::scoped_lock save_lock{save_mutex};
    std::unique_lock lock{mutex};
    path = file_path;
    stamp = build_stamp;
    valid_file_size = 0;
    open = true;
    read_file.reset();
    disk_index.clear();
    pending_entries.clear();

    if (path.filename() == "metal_stages.v2.bin") {
        std::error_code remove_ec;
        std::filesystem::remove(path.parent_path() / "metal_stages.v1.bin", remove_ec);
    }

    auto scan_file = std::make_shared<Common::FS::IOFile>(
        path, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile,
        Common::FS::FileShareFlag::ShareReadWrite);
    if (!scan_file->IsOpen()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    StageFileHeader hdr{};
    if (!scan_file->ReadObject(hdr) || hdr.magic != StageMslMagic ||
        hdr.version != StageMslVersion ||
        (stamp != 0 && hdr.reserved != StageStampTag(stamp))) {
        return;
    }
    const u64 file_size = scan_file->GetSize();
    u64 pos = sizeof(StageFileHeader);
    while (pos + sizeof(StageRecordHeader) <= file_size) {
        if (!scan_file->Seek(static_cast<s64>(pos))) {
            break;
        }
        StageRecordHeader rec{};
        if (!scan_file->ReadObject(rec)) {
            break;
        }
        const u64 payload_offset = pos + sizeof(StageRecordHeader);
        if (rec.uncompressed_size == 0 || rec.uncompressed_size > MaxStageRecordBytes ||
            rec.compressed_size == 0 || rec.compressed_size > MaxStageRecordBytes ||
            rec.compressed_size > file_size - payload_offset) {
            break;
        }
        disk_index.insert_or_assign(rec.key, DiskSlice{
                                                 .offset = payload_offset,
                                                 .comp_size = rec.compressed_size,
                                                 .uncomp_size = rec.uncompressed_size,
                                             });
        pos = payload_offset + rec.compressed_size;
    }
    if (pos < file_size) {
        scan_file.reset();
        std::error_code resize_ec;
        std::filesystem::resize_file(path, pos, resize_ec);
        scan_file = std::make_shared<Common::FS::IOFile>(
            path, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile,
            Common::FS::FileShareFlag::ShareReadWrite);
        if (!scan_file->IsOpen()) {
            disk_index.clear();
            return;
        }
    }
    valid_file_size = pos;
    read_file = std::move(scan_file);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    LOG_INFO(Render_Metal, "Metal: stage MSL cache {} indexed ({} stages, {} KiB, {:.1f} ms)",
             path.string(), disk_index.size(), valid_file_size >> 10, ms);
}

void StageMslCache::Flush() {
    if (!StageMslCacheEnabled()) {
        return;
    }
    std::scoped_lock save_lock{save_mutex};
    std::filesystem::path target_path;
    u64 target_stamp{};
    u64 start_offset{};
    bool need_reader = false;
    std::vector<std::pair<u128, std::shared_ptr<const ShaderTranslate::Result>>> batch;
    {
        std::shared_lock lock{mutex};
        if (!open || path.empty() || pending_entries.empty()) {
            return;
        }
        target_path = path;
        target_stamp = stamp;
        start_offset = valid_file_size;
        need_reader = !read_file || !read_file->IsOpen();
        batch.reserve(pending_entries.size());
        for (const auto& [key, res_ptr] : pending_entries) {
            if (!disk_index.contains(key) && res_ptr) {
                batch.emplace_back(key, res_ptr);
            }
        }
    }
    if (batch.empty()) {
        std::unique_lock lock{mutex};
        for (auto it = pending_entries.begin(); it != pending_entries.end();) {
            if (disk_index.contains(it->first)) {
                it = pending_entries.erase(it);
            } else {
                ++it;
            }
        }
        return;
    }

    struct PreparedRecord {
        u128 key;
        u32 uncomp_size;
        std::vector<u8> compressed;
    };
    std::vector<PreparedRecord> prepared;
    prepared.reserve(batch.size());
    size_t total_append_bytes = 0;
    for (const auto& [key, res_ptr] : batch) {
        const std::vector<u8> raw = SerializeStageResult(*res_ptr);
        if (raw.empty() || raw.size() > MaxStageRecordBytes) {
            continue;
        }
        std::vector<u8> comp = Common::Compression::CompressDataZSTD(raw.data(), raw.size(), 1);
        if (comp.empty() || comp.size() > MaxStageRecordBytes) {
            continue;
        }
        total_append_bytes += sizeof(StageRecordHeader) + comp.size();
        prepared.push_back(PreparedRecord{
            .key = key,
            .uncomp_size = static_cast<u32>(raw.size()),
            .compressed = std::move(comp),
        });
    }
    if (prepared.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(target_path.parent_path(), ec);

    std::vector<u8> write_buf;
    const u64 base_offset = start_offset == 0 ? sizeof(StageFileHeader) : start_offset;
    write_buf.reserve((start_offset == 0 ? sizeof(StageFileHeader) : 0) + total_append_bytes);
    if (start_offset == 0) {
        const StageFileHeader hdr{
            .magic = StageMslMagic,
            .version = StageMslVersion,
            .reserved = StageStampTag(target_stamp),
        };
        const auto* const hdr_bytes = reinterpret_cast<const u8*>(&hdr);
        write_buf.insert(write_buf.end(), hdr_bytes, hdr_bytes + sizeof(hdr));
    }

    std::vector<std::pair<u128, DiskSlice>> new_slices;
    new_slices.reserve(prepared.size());
    u64 running_offset = base_offset;
    for (const PreparedRecord& item : prepared) {
        const StageRecordHeader rec{
            .key = item.key,
            .uncompressed_size = item.uncomp_size,
            .compressed_size = static_cast<u32>(item.compressed.size()),
        };
        const auto* const rec_bytes = reinterpret_cast<const u8*>(&rec);
        write_buf.insert(write_buf.end(), rec_bytes, rec_bytes + sizeof(rec));
        write_buf.insert(write_buf.end(), item.compressed.begin(), item.compressed.end());
        const u64 payload_offset = running_offset + sizeof(StageRecordHeader);
        new_slices.emplace_back(item.key, DiskSlice{
                                              .offset = payload_offset,
                                              .comp_size = rec.compressed_size,
                                              .uncomp_size = rec.uncompressed_size,
                                          });
        running_offset = payload_offset + rec.compressed_size;
    }

    std::shared_ptr<Common::FS::IOFile> opened_reader;
    if (start_offset == 0) {
        std::filesystem::path tmp = target_path;
        tmp += ".tmp";
        {
            Common::FS::IOFile out{tmp, Common::FS::FileAccessMode::Write,
                                   Common::FS::FileType::BinaryFile,
                                   Common::FS::FileShareFlag::ShareReadWrite};
            if (!out.IsOpen() || out.WriteSpan<u8>(write_buf) != write_buf.size() || !out.Flush()) {
                out.Close();
                std::filesystem::remove(tmp, ec);
                return;
            }
        }
        std::filesystem::rename(tmp, target_path, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            return;
        }
        need_reader = true;
    } else {
        Common::FS::IOFile out{target_path, Common::FS::FileAccessMode::ReadWrite,
                               Common::FS::FileType::BinaryFile,
                               Common::FS::FileShareFlag::ShareReadWrite};
        if (!out.IsOpen() || !out.Seek(static_cast<s64>(start_offset))) {
            return;
        }
        if (out.WriteSpan<u8>(write_buf) != write_buf.size() || !out.Flush()) {
            static_cast<void>(out.SetSize(start_offset));
            return;
        }
    }

    if (need_reader) {
        opened_reader = std::make_shared<Common::FS::IOFile>(
            target_path, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile,
            Common::FS::FileShareFlag::ShareReadWrite);
        if (!opened_reader->IsOpen()) {
            return;
        }
    }

    size_t total_indexed = 0;
    {
        std::unique_lock lock{mutex};
        if (!open || path != target_path || stamp != target_stamp) {
            return;
        }
        if (opened_reader) {
            read_file = std::move(opened_reader);
        }
        valid_file_size = running_offset;
        for (const auto& [key, slice] : new_slices) {
            disk_index.insert_or_assign(key, slice);
            pending_entries.erase(key);
        }
        if (pending_entries.empty() && pending_entries.bucket_count() > 256) {
            std::unordered_map<u128, std::shared_ptr<const ShaderTranslate::Result>, KeyHasher>{}
                .swap(pending_entries);
        }
        total_indexed = disk_index.size();
    }
    LOG_DEBUG(Render_Metal, "Metal: stage MSL cache {} flushed +{} stages ({} total, {} KiB)",
              target_path.string(), new_slices.size(), total_indexed, running_offset >> 10);
}

std::optional<ShaderTranslate::Result> StageMslCache::Find(const u128& key) const {
    if (!StageMslCacheEnabled()) {
        return std::nullopt;
    }
    DiskSlice slice{};
    std::shared_ptr<Common::FS::IOFile> reader;
    {
        std::shared_lock lock{mutex};
        if (!open) {
            return std::nullopt;
        }
        if (const auto it = pending_entries.find(key);
            it != pending_entries.end() && it->second != nullptr) {
            return *it->second;
        }
        const auto disk_it = disk_index.find(key);
        if (disk_it == disk_index.end()) {
            return std::nullopt;
        }
        slice = disk_it->second;
        reader = read_file;
    }
    if (!reader || !reader->IsOpen() || slice.comp_size == 0 || slice.uncomp_size == 0 ||
        slice.comp_size > MaxStageRecordBytes || slice.uncomp_size > MaxStageRecordBytes) {
        return std::nullopt;
    }
    std::vector<u8> compressed(slice.comp_size);
    if (reader->ReadBytesAt(compressed, slice.offset) != slice.comp_size) {
        return std::nullopt;
    }
    std::vector<u8> uncompressed(slice.uncomp_size);
    if (Common::Compression::DecompressDataZSTD(compressed, uncompressed) != slice.uncomp_size) {
        return std::nullopt;
    }
    return DeserializeStageResult(uncompressed);
}

void StageMslCache::Insert(const u128& key, const ShaderTranslate::Result& result) {
    if (!StageMslCacheEnabled() || result.source.empty()) {
        return;
    }
    std::unique_lock lock{mutex};
    if (!open || disk_index.contains(key) || pending_entries.contains(key)) {
        return;
    }
    pending_entries.emplace(key, std::make_shared<const ShaderTranslate::Result>(result));
}

size_t StageMslCache::Size() const {
    std::shared_lock lock{mutex};
    return disk_index.size() + pending_entries.size();
}

size_t StageMslCache::PendingSize() const {
    std::shared_lock lock{mutex};
    return pending_entries.size();
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
        .support_float_controls = true,
        .support_fp16_signed_zero_nan_preserve = true,
        .support_fp32_signed_zero_nan_preserve = true,
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
        .max_per_stage_descriptor_sampled_images = 500000U,
        .max_per_stage_resources = 500000U,
        .max_per_stage_descriptor_storage_buffers = Limit,
        .max_descriptor_set_samplers = 2048U,
        .max_descriptor_set_uniform_buffers = Limit,
        .max_descriptor_set_uniform_buffers_dynamic = Limit,
        .max_descriptor_set_storage_buffers = Limit,
        .max_descriptor_set_storage_buffers_dynamic = Limit,
        .max_descriptor_set_sampled_images = 500000U,
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
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 170000)
    if (apple_family >= 8) {
        // MSL 3.1 enables native hardware texture atomics on Apple8+ (A15/M2/M3/M4+) instead of
        // SPIRV-Cross synthesizing auxiliary atomic buffers.
        translate_options.msl_minor = 1;
    }
#endif
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
        .immutable_buffer_mask = result.immutable_buffer_mask,
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

    Shader::Backend::Bindings vertex_bindings;
    Shader::RuntimeInfo vertex_info;
    vertex_info.previous_stage_stores.mask.set();
    vertex_info.input_topology = InputTopology(static_cast<Maxwell::PrimitiveTopology>(key.topology));
    for (size_t attribute = 0; attribute < vertex_info.generic_input_types.size(); ++attribute) {
        constexpr std::array types{Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                                   Shader::AttributeType::SignedInt,
                                   Shader::AttributeType::UnsignedInt};
        vertex_info.generic_input_types[attribute] = types[(key.attribute_types >> (attribute * 2)) & 3];
    }
    if (!programs[1].info.loads.Legacy() && !programs[1].info.passthrough.mask.any()) {
        for (size_t attribute = 0; attribute < vertex_info.generic_input_types.size(); ++attribute) {
            if (!programs[1].info.loads.Generic(attribute)) {
                vertex_info.generic_input_types[attribute] = Shader::AttributeType::Disabled;
            }
        }
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
        const std::vector<u32> spirv = EmitSPIRV(profile, vertex_info, programs[1], vertex_bindings);
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
                VideoCommon::StageCache::RuntimeKey(vertex_info, vertex_bindings);
            if (!stage_cache->FindEmitted(*stage_reuse.reused[1], rt_key)) {
                Shader::Environment& v_env = *envs[0];
                const u32 cfg_offset =
                    static_cast<u32>(v_env.StartAddress() + sizeof(Shader::ProgramHeader));
                Shader::Maxwell::Flow::CFG cfg(v_env, pools.flow_block, cfg_offset, false);
                programs[1] = TranslateProgram(pools.inst, pools.block, v_env, cfg, host_info);
            }
        }
        vertex = TranslateCached(1, programs[1], ShaderTranslate::Stage::Vertex, vertex_bindings,
                                 vertex_info, &vertex_options);
    }
    if (!vertex) {
        return std::nullopt;
    }
    StageProgram fragment{};
    if (key.unique_hashes[5] != 0) {
        // Metal binds each stage's descriptors independently starting from slot 0, so reset
        // Bindings and canonicalize previous_stage_stores when all loaded varyings are written.
        // That lets StageMslCache and macOS's MTLLibrary cache hit WarmSingleStage across all
        // vertex shaders.
        Shader::Backend::Bindings fragment_bindings;
        Shader::RuntimeInfo fragment_info;
        if (!programs[5].info.loads.Legacy() && !programs[5].info.passthrough.mask.any() &&
            (programs[5].info.loads.mask & ~programs[1].info.stores.mask).none()) {
            fragment_info.previous_stage_stores.mask.set();
        } else {
            fragment_info.previous_stage_stores = programs[1].info.stores;
            fragment_info.previous_stage_legacy_stores_mapping =
                programs[1].info.legacy_stores_mapping;
        }
        ApplyFragmentState(fragment_info, key);
        if (stage_reuse.reused[5] && programs[5].blocks.empty()) {
            const std::vector<u8> rt_key =
                VideoCommon::StageCache::RuntimeKey(fragment_info, fragment_bindings);
            if (!stage_cache->FindEmitted(*stage_reuse.reused[5], rt_key)) {
                Shader::Environment& f_env = *envs.back();
                const u32 cfg_offset =
                    static_cast<u32>(f_env.StartAddress() + sizeof(Shader::ProgramHeader));
                Shader::Maxwell::Flow::CFG cfg(f_env, pools.flow_block, cfg_offset, false);
                programs[5] = TranslateProgram(pools.inst, pools.block, f_env, cfg, host_info);
            }
        }
        auto translated =
            TranslateCached(5, programs[5], ShaderTranslate::Stage::Fragment, fragment_bindings,
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
        Shader::Backend::Bindings object_bindings;
        Shader::RuntimeInfo task_info;
        task_info.previous_stage_stores.mask.set();
        task_info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
        auto object = Translate(*task, ShaderTranslate::Stage::Object, object_bindings, task_info);
        if (!object) {
            keep_cause();
            return std::nullopt;
        }
        native.object = std::move(*object);
    }
    Shader::Backend::Bindings mesh_bindings;
    Shader::RuntimeInfo mesh_info;
    mesh_info.previous_stage_stores.mask.set();
    mesh_info.y_negate = (key.state & StateLowerLeftOrigin) != 0;
    mesh_info.convert_depth_mode = (key.state & StateDepthMinusOneToOne) != 0;
    if (FixedStateEnabled() && mesh.mesh.topology == 0) {
        mesh_info.fixed_state_point_size = std::bit_cast<float>(key.point_size);
    }
    mesh_info.mesh_direct_outputs = true;
    auto mesh_function = Translate(mesh, ShaderTranslate::Stage::Mesh, mesh_bindings, mesh_info);
    if (!mesh_function) {
        keep_cause();
        return std::nullopt;
    }
    native.mesh = std::move(*mesh_function);
    if (key.unique_hashes[5] != 0) {
        Shader::Backend::Bindings fragment_bindings;
        Shader::RuntimeInfo fragment_info;
        if (!programs[5].info.loads.Legacy() && !programs[5].info.passthrough.mask.any() &&
            (programs[5].info.loads.mask & ~mesh.info.stores.mask).none()) {
            fragment_info.previous_stage_stores.mask.set();
        } else {
            fragment_info.previous_stage_stores = mesh.info.stores;
            fragment_info.previous_stage_legacy_stores_mapping = mesh.info.legacy_stores_mapping;
        }
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
            Translate(programs[5], ShaderTranslate::Stage::Fragment, fragment_bindings, fragment_info);
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

    const auto mask_unused_vertex_inputs = [&](const Shader::Info& info) {
        if (slot != 1 || info.loads.Legacy() || info.passthrough.mask.any()) {
            return;
        }
        for (size_t attr = 0; attr < runtime_info.generic_input_types.size(); ++attr) {
            if (!info.loads.Generic(attr)) {
                runtime_info.generic_input_types[attr] = Shader::AttributeType::Disabled;
            }
        }
    };

    Shader::Backend::Bindings bindings{};
    const u32 cfg_offset = static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
    using Cache = VideoCommon::StageCache;
    if (stage_cache != nullptr && stage_cache->Enabled() && !VisualWorkaroundEnabled()) {
        std::vector<u8> signature = Cache::Signature(env, unique_hash, slot);
        stage_reuse.reused[slot] = stage_cache->Find(signature, env);
        if (stage_reuse.reused[slot] && !stage_cache->Verify()) {
            mask_unused_vertex_inputs(stage_reuse.reused[slot]->program.info);
            const std::vector<u8> rt_key = Cache::RuntimeKey(runtime_info, bindings);
            if (stage_cache->FindEmitted(*stage_reuse.reused[slot], rt_key)) {
                Shader::IR::Program summary = stage_reuse.reused[slot]->program;
                return TranslateCached(slot, summary, msl_stage, bindings, runtime_info, &options);
            }
        }
        Cache::Recorder recorder{env};
        Shader::Environment& active_env =
            stage_reuse.reused[slot] ? env : static_cast<Shader::Environment&>(recorder);
        Shader::Maxwell::Flow::CFG cfg(active_env, pools.flow_block, cfg_offset, false);
        Shader::IR::Program program =
            TranslateProgram(pools.inst, pools.block, active_env, cfg, host_info);
        mask_unused_vertex_inputs(program.info);
        if (!stage_reuse.reused[slot]) {
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
    mask_unused_vertex_inputs(program.info);
    ApplyVisualWorkaround(program, unique_hash);
    return Translate(program, msl_stage, bindings, runtime_info, &options);
} catch (const Shader::Exception&) {
    return std::nullopt;
}

namespace {

struct Taint {
    bool uses_vertex_id{};
    u32 attr_mask{};

    constexpr Taint operator|(const Taint& other) const noexcept {
        return {uses_vertex_id || other.uses_vertex_id, attr_mask | other.attr_mask};
    }

    constexpr Taint& operator|=(const Taint& other) noexcept {
        uses_vertex_id = uses_vertex_id || other.uses_vertex_id;
        attr_mask |= other.attr_mask;
        return *this;
    }

    static constexpr Taint Varying() noexcept {
        return {true, 0};
    }
};

[[nodiscard]] bool IsIdentStartChar(char c) noexcept {
    return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}

[[nodiscard]] bool IsIdentContChar(char c) noexcept {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

[[nodiscard]] std::string_view TrimWs(std::string_view s) noexcept {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
        s.remove_suffix(1);
    }
    return s;
}

[[nodiscard]] size_t SkipMslCommentOrLit(std::string_view src, size_t i) noexcept {
    const size_t n = src.size();
    if (i + 1 < n && src[i] == '/' && src[i + 1] == '/') {
        i += 2;
        while (i < n && src[i] != '\n') {
            ++i;
        }
        return i;
    }
    if (i + 1 < n && src[i] == '/' && src[i + 1] == '*') {
        i += 2;
        while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) {
            ++i;
        }
        return i + 1 < n ? i + 2 : n;
    }
    if (src[i] == '"' || src[i] == '\'') {
        const char q = src[i++];
        while (i < n && src[i] != q) {
            i += (src[i] == '\\' && i + 1 < n) ? 2 : 1;
        }
        return i < n ? i + 1 : n;
    }
    return i;
}

[[nodiscard]] size_t SkipMslTrivia(std::string_view src, size_t i, size_t end) noexcept {
    while (i < end) {
        if (std::isspace(static_cast<unsigned char>(src[i])) != 0) {
            ++i;
            continue;
        }
        if (i + 1 < end && src[i] == '/' && (src[i + 1] == '/' || src[i + 1] == '*')) {
            i = (std::min)(SkipMslCommentOrLit(src, i), end);
            continue;
        }
        break;
    }
    return i;
}

[[nodiscard]] std::vector<u32> BuildMslBracketTable(std::string_view src) {
    if (src.size() >= ~0U) {
        return {};
    }
    std::vector<u32> match(src.size(), ~0U);
    std::vector<std::pair<char, u32>> stack;
    stack.reserve(128);
    for (size_t i = 0; i < src.size();) {
        const size_t next = SkipMslCommentOrLit(src, i);
        if (next != i) {
            i = next;
            continue;
        }
        const char c = src[i];
        if (c == '(' || c == '[' || c == '{') {
            stack.emplace_back(c, static_cast<u32>(i));
        } else if (c == ')' || c == ']' || c == '}') {
            const char want = c == ')' ? '(' : c == ']' ? '[' : '{';
            if (stack.empty() || stack.back().first != want) {
                return {};
            }
            const u32 open = stack.back().second;
            stack.pop_back();
            match[open] = static_cast<u32>(i);
            match[i] = open;
        }
        ++i;
    }
    if (!stack.empty()) {
        return {};
    }
    return match;
}

[[nodiscard]] bool ParseUnsignedUint(std::string_view s, u32& out) noexcept {
    s = TrimWs(s);
    if (s.empty()) {
        return false;
    }
    size_t pos = 0;
    u64 val = 0;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos])) != 0) {
        val = val * 10 + static_cast<u64>(s[pos] - '0');
        if (val > 0xFFFFFFFFULL) {
            return false;
        }
        ++pos;
    }
    if (pos == 0) {
        return false;
    }
    while (pos < s.size() && (s[pos] == 'u' || s[pos] == 'U' || s[pos] == 'l' || s[pos] == 'L')) {
        ++pos;
    }
    if (pos != s.size()) {
        return false;
    }
    out = static_cast<u32>(val);
    return true;
}

[[nodiscard]] std::vector<std::string_view> SplitTopCommas(std::string_view text) {
    std::vector<std::string_view> parts;
    int depth = 0;
    int angles = 0;
    size_t begin = 0;
    for (size_t i = 0; i < text.size();) {
        const size_t skipped = SkipMslCommentOrLit(text, i);
        if (skipped != i) {
            i = skipped;
            continue;
        }
        const char c = text[i];
        const char before = i != 0 ? text[i - 1] : ' ';
        const char after = i + 1 < text.size() ? text[i + 1] : ' ';
        if (c == '<') {
            if (IsIdentContChar(before) && after != '=' && after != '<') {
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
            parts.push_back(TrimWs(text.substr(begin, i - begin)));
            begin = i + 1;
        }
        ++i;
    }
    if (!TrimWs(text.substr(begin)).empty()) {
        parts.push_back(TrimWs(text.substr(begin)));
    }
    return parts;
}

[[nodiscard]] bool ContainsStandaloneIdent(std::string_view text, std::string_view want) noexcept {
    if (want.empty()) {
        return false;
    }
    for (size_t i = 0; i < text.size();) {
        const size_t skipped = SkipMslCommentOrLit(text, i);
        if (skipped != i) {
            i = skipped;
            continue;
        }
        if (IsIdentStartChar(text[i])) {
            size_t end = i + 1;
            while (end < text.size() && IsIdentContChar(text[end])) {
                ++end;
            }
            if (text.substr(i, end - i) == want) {
                size_t back = i;
                while (back > 0 && std::isspace(static_cast<unsigned char>(text[back - 1])) != 0) {
                    --back;
                }
                const bool member = back > 0 && (text[back - 1] == '.' ||
                                                 (back >= 2 && text[back - 2] == '-' &&
                                                  text[back - 1] == '>'));
                if (!member) {
                    return true;
                }
            }
            i = end;
            continue;
        }
        ++i;
    }
    return false;
}

[[nodiscard]] bool IsTemplateCallName(std::string_view id) noexcept {
    return id == "as_type" || id == "static_cast" || id == "reinterpret_cast" ||
           id == "const_cast" || id == "vec" || id == "matrix" || id == "array" ||
           id == "spvUnsafeArray" || id == "spvFAdd" || id == "spvFSub" || id == "spvFMul" ||
           id == "spvFMulVectorMatrix" || id == "spvFMulMatrixVector" ||
           id == "spvFMulMatrixMatrix";
}

[[nodiscard]] bool IsTypeOrQualifier(std::string_view id) noexcept {
    static const std::unordered_set<std::string_view> kTypes{
        "const",    "constant", "device",  "thread",      "threadgroup", "constexpr", "inline",
        "signed",   "unsigned", "void",    "bool",        "bool2",       "bool3",     "bool4",
        "char",     "char2",    "char3",   "char4",       "uchar",       "uchar2",    "uchar3",
        "uchar4",   "short",    "short2",  "short3",      "short4",      "ushort",    "ushort2",
        "ushort3",  "ushort4",  "int",     "int2",        "int3",        "int4",      "uint",
        "uint2",    "uint3",    "uint4",   "long",        "long2",       "long3",     "long4",
        "ulong",    "ulong2",   "ulong3",  "ulong4",      "half",        "half2",     "half3",
        "half4",    "float",    "float2",  "float3",      "float4",      "half2x2",   "half2x3",
        "half2x4",  "half3x2",  "half3x3", "half3x4",     "half4x2",     "half4x3",   "half4x4",
        "float2x2", "float2x3", "float2x4","float3x2",    "float3x3",    "float3x4",  "float4x2",
        "float4x3", "float4x4", "size_t",  "ptrdiff_t",   "intptr_t",    "uintptr_t", "sampler",
        "packed_char2", "packed_char3", "packed_char4", "packed_uchar2", "packed_uchar3",
        "packed_uchar4", "packed_short2", "packed_short3", "packed_short4", "packed_ushort2",
        "packed_ushort3", "packed_ushort4", "packed_int2", "packed_int3", "packed_int4",
        "packed_uint2", "packed_uint3", "packed_uint4", "packed_half2", "packed_half3",
        "packed_half4", "packed_float2", "packed_float3", "packed_float4",
    };
    return kTypes.contains(id);
}

[[nodiscard]] bool IsPureBuiltinOrCtor(std::string_view id) noexcept {
    if (IsTypeOrQualifier(id) || IsTemplateCallName(id)) {
        return true;
    }
    static const std::unordered_set<std::string_view> kPureBuiltins{
        "sizeof", "alignof", "level", "bias", "gradient2d", "gradient3d", "gradientcube",
        "min_lod_clamp", "abs", "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
        "ceil", "clamp", "copysign", "cos", "cosh", "cospi", "degrees", "exp", "exp2", "exp10",
        "fabs", "fdim", "floor", "fma", "fmax", "fmax3", "fmed3", "fmin", "fmin3", "fmod",
        "fract", "ilogb", "ldexp", "log", "log2", "log10", "mad", "max", "max3", "median3",
        "min", "min3", "mix", "pow", "powr", "pown", "radians", "rint", "rootn", "round",
        "rsqrt", "saturate", "sign", "sin", "sinh", "sinpi", "smoothstep", "sqrt", "step",
        "tan", "tanh", "tanpi", "trunc", "cross", "distance", "dot", "fast_distance",
        "fast_length", "fast_normalize", "faceforward", "length", "length_squared",
        "distance_squared", "normalize", "reflect", "refract", "determinant", "transpose",
        "all", "any", "isfinite", "isinf", "isnan", "isnormal", "isordered", "isunordered",
        "not", "select", "signbit", "abs_diff", "add_sat", "clz", "ctz", "extract_bits", "hadd",
        "insert_bits", "mad24", "mad_hi", "mad_sat", "mul24", "mul_hi", "popcount",
        "reverse_bits", "rhadd", "rotate", "sub_sat", "pack_float_to_snorm2x16",
        "pack_float_to_unorm2x16", "pack_float_to_snorm4x8", "pack_float_to_unorm4x8",
        "unpack_snorm2x16_to_float", "unpack_unorm2x16_to_float", "unpack_snorm4x8_to_float",
        "unpack_unorm4x8_to_float", "as_float", "as_half", "as_int", "as_uint", "as_short",
        "as_ushort", "as_char", "as_uchar", "as_long", "as_ulong",
        // Pure SPIRV-Cross & Lichii per-lane MSL helpers (no out-params, no cross-lane state):
        "spvReducing", "spvReduceTaps", "spvReducedSample", "spvSMod", "spvUMod",
        "spvInverse4x4", "spvInverse3x3", "spvInverse2x2", "spvPackFloat2x16",
        "spvUnpackFloat2x16", "spvPackUnorm2x16", "spvUnpackUnorm2x16", "spvPackSnorm2x16",
        "spvUnpackSnorm2x16", "spvPackUnorm4x8", "spvUnpackUnorm4x8", "spvPackSnorm4x8",
        "spvUnpackSnorm4x8", "spvFindLSB", "spvFindSMSB", "spvFindUMSB", "spvBitCount",
        "spvBitFieldInsert", "spvBitFieldSExtract", "spvBitFieldUExtract", "spvBitReverse",
        "spvTextureSwizzle", "spvGatherSwizzle", "spvGatherCompareSwizzle",
        "spvCubemapTo2DArrayFace",
    };
    return kPureBuiltins.contains(id);
}

[[nodiscard]] bool IsPureMemberMethod(std::string_view id) noexcept {
    static const std::unordered_set<std::string_view> kPureMethods{
        "sample", "sample_compare", "read", "gather", "gather_compare",
        "calculate_clamped_lod", "calculate_unclamped_lod", "get_width", "get_height",
        "get_depth", "get_array_size", "get_num_mip_levels", "get_num_samples",
        "is_null_texture",
    };
    return kPureMethods.contains(id);
}

struct ParsedStructMember {
    std::string name;
    std::string type;
    u32 array_count{1};
    std::vector<std::string> attrs;
};

[[nodiscard]] bool ParseMslStruct(std::string_view msl, const std::vector<u32>& match,
                                  std::string_view struct_name,
                                  std::vector<ParsedStructMember>& out_members) {
    out_members.clear();
    for (size_t pos = 0; pos < msl.size();) {
        const size_t skipped = SkipMslCommentOrLit(msl, pos);
        if (skipped != pos) {
            pos = skipped;
            continue;
        }
        if (IsIdentStartChar(msl[pos])) {
            size_t end = pos + 1;
            while (end < msl.size() && IsIdentContChar(msl[end])) {
                ++end;
            }
            if (msl.substr(pos, end - pos) == "struct") {
                const size_t name_start = SkipMslTrivia(msl, end, msl.size());
                size_t name_end = name_start;
                while (name_end < msl.size() && IsIdentContChar(msl[name_end])) {
                    ++name_end;
                }
                if (msl.substr(name_start, name_end - name_start) == struct_name) {
                    const size_t brace = SkipMslTrivia(msl, name_end, msl.size());
                    if (brace < msl.size() && msl[brace] == '{' && match[brace] != ~0U) {
                        const size_t close = match[brace];
                        size_t cur = brace + 1;
                        while (cur < close) {
                            cur = SkipMslTrivia(msl, cur, close);
                            if (cur >= close) {
                                break;
                            }
                            size_t semi = cur;
                            while (semi < close && msl[semi] != ';') {
                                const size_t s2 = SkipMslCommentOrLit(msl, semi);
                                if (s2 != semi) {
                                    semi = s2;
                                    continue;
                                }
                                if ((msl[semi] == '(' || msl[semi] == '[' || msl[semi] == '{') &&
                                    match[semi] < close) {
                                    semi = size_t{match[semi]} + 1;
                                    continue;
                                }
                                ++semi;
                            }
                            if (semi >= close) {
                                return false;
                            }
                            std::string decl{msl.substr(cur, semi - cur)};
                            cur = semi + 1;
                            ParsedStructMember member{};
                            for (size_t attr_at = decl.find("[["); attr_at != std::string::npos;
                                 attr_at = decl.find("[[")) {
                                const size_t attr_end = decl.find("]]", attr_at + 2);
                                if (attr_end == std::string::npos) {
                                    return false;
                                }
                                member.attrs.emplace_back(
                                    TrimWs(std::string_view{decl}.substr(attr_at + 2,
                                                                         attr_end - attr_at - 2)));
                                decl.erase(attr_at, attr_end + 2 - attr_at);
                            }
                            std::string_view clean = TrimWs(decl);
                            if (clean.empty()) {
                                continue;
                            }
                            while (!clean.empty() && clean.back() == ']') {
                                const size_t open_b = clean.rfind('[');
                                if (open_b == std::string_view::npos) {
                                    return false;
                                }
                                u32 count = 0;
                                if (!ParseUnsignedUint(
                                        clean.substr(open_b + 1, clean.size() - open_b - 2),
                                        count) ||
                                    count == 0) {
                                    return false;
                                }
                                member.array_count *= count;
                                clean = TrimWs(clean.substr(0, open_b));
                            }
                            size_t n_start = clean.size();
                            while (n_start > 0 && IsIdentContChar(clean[n_start - 1])) {
                                --n_start;
                            }
                            if (n_start == clean.size() || n_start == 0) {
                                return false;
                            }
                            member.name = std::string{clean.substr(n_start)};
                            member.type = std::string{TrimWs(clean.substr(0, n_start))};
                            if (member.type.empty()) {
                                return false;
                            }
                            out_members.push_back(std::move(member));
                        }
                        return true;
                    }
                }
            }
            pos = end;
            continue;
        }
        ++pos;
    }
    return false;
}

[[nodiscard]] bool IsMatrixOrArrayType(std::string_view type) noexcept {
    if (type.find("matrix<") != std::string_view::npos ||
        type.find("array<") != std::string_view::npos) {
        return true;
    }
    return type.find("2x") != std::string_view::npos ||
           type.find("3x") != std::string_view::npos ||
           type.find("4x") != std::string_view::npos;
}

[[nodiscard]] bool FindTopLevelAssign(std::string_view stmt, size_t& eq_begin,
                                      size_t& eq_end) noexcept {
    int depth = 0;
    int angles = 0;
    for (size_t i = 0; i < stmt.size();) {
        const size_t skipped = SkipMslCommentOrLit(stmt, i);
        if (skipped != i) {
            i = skipped;
            continue;
        }
        const char c = stmt[i];
        const char before = i != 0 ? stmt[i - 1] : ' ';
        const char after = i + 1 < stmt.size() ? stmt[i + 1] : ' ';
        if (c == '<') {
            if (depth == 0 && IsIdentContChar(before) && after != '=' && after != '<') {
                ++angles;
            }
        } else if (c == '>') {
            if (depth == 0 && angles > 0 && before != '-' && before != ' ' && after != '=') {
                --angles;
            }
        } else if (c == '(' || c == '[' || c == '{') {
            ++depth;
        } else if (c == ')' || c == ']' || c == '}') {
            --depth;
        } else if (c == '=' && depth == 0 && angles == 0) {
            if (after == '=') {
                i += 2;
                continue;
            }
            if (before == '!') {
                ++i;
                continue;
            }
            if (before == '<' || before == '>') {
                if (i >= 2 && stmt[i - 2] == before) {
                    eq_begin = i - 2;
                    eq_end = i + 1;
                    return true;
                }
                ++i;
                continue;
            }
            if (before == '+' || before == '-' || before == '*' || before == '/' ||
                before == '%' || before == '&' || before == '|' || before == '^') {
                eq_begin = i - 1;
                eq_end = i + 1;
                return true;
            }
            eq_begin = i;
            eq_end = i + 1;
            return true;
        }
        ++i;
    }
    return false;
}

class VertexMslAnalyzer {
public:
    VertexMslAnalyzer(std::string_view msl_, const std::vector<u32>& match_)
        : msl{msl_}, match{match_} {}

    std::string stage_in_var;
    std::string out_struct_name;
    std::string out_var;
    std::unordered_map<std::string_view, Taint> in_field_taint;
    std::unordered_map<std::string_view, u32> out_field_locations;
    std::unordered_map<std::string_view, Taint> out_builtin_taint;
    std::unordered_map<std::string_view, Taint> var_taint;
    std::array<Taint, 32> generic_taint{};
    Taint post_return_taint{};
    bool seen_return{};
    bool alias_hazard{};

    [[nodiscard]] Taint EvalExpr(std::string_view expr) {
        Taint res{};
        bool prev_was_value = false;
        for (size_t i = 0; i < expr.size() && !alias_hazard;) {
            const size_t ws = SkipMslTrivia(expr, i, expr.size());
            if (ws != i) {
                i = ws;
                continue;
            }
            const char c = expr[i];
            if (c == '"' || c == '\'') {
                i = SkipMslCommentOrLit(expr, i);
                prev_was_value = true;
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(c)) != 0 ||
                (c == '.' && i + 1 < expr.size() &&
                 std::isdigit(static_cast<unsigned char>(expr[i + 1])) != 0)) {
                ++i;
                while (i < expr.size() &&
                       (IsIdentContChar(expr[i]) || expr[i] == '.' ||
                        ((expr[i] == '+' || expr[i] == '-') && i > 0 &&
                         (expr[i - 1] == 'e' || expr[i - 1] == 'E' || expr[i - 1] == 'p' ||
                          expr[i - 1] == 'P')))) {
                    ++i;
                }
                prev_was_value = true;
                continue;
            }
            if (c == '&') {
                if (i + 1 < expr.size() && expr[i + 1] == '&') {
                    prev_was_value = false;
                    i += 2;
                    continue;
                }
                if (!prev_was_value) {
                    alias_hazard = true;
                    return Taint::Varying();
                }
                prev_was_value = false;
                ++i;
                continue;
            }
            if ((c == '+' && i + 1 < expr.size() && expr[i + 1] == '+') ||
                (c == '-' && i + 1 < expr.size() && expr[i + 1] == '-')) {
                // Side-effecting increment / decrement inside an expression.
                alias_hazard = true;
                return Taint::Varying();
            }
            if (c == '=') {
                const char prev = i > 0 ? expr[i - 1] : ' ';
                const char next = i + 1 < expr.size() ? expr[i + 1] : ' ';
                if (next == '=') {
                    prev_was_value = false;
                    i += 2;
                    continue;
                }
                if (prev == '!' || prev == '<' || prev == '>') {
                    if (i >= 2 && expr[i - 2] == prev) {
                        alias_hazard = true;
                        return Taint::Varying();
                    }
                    prev_was_value = false;
                    ++i;
                    continue;
                }
                alias_hazard = true;
                return Taint::Varying();
            }
            if (c == '(' || c == '[' || c == '{') {
                const size_t abs_open = static_cast<size_t>(expr.data() - msl.data()) + i;
                if (abs_open >= match.size() || match[abs_open] == ~0U) {
                    alias_hazard = true;
                    return Taint::Varying();
                }
                const size_t close = match[abs_open] - static_cast<size_t>(expr.data() - msl.data());
                if (close >= expr.size()) {
                    alias_hazard = true;
                    return Taint::Varying();
                }
                res |= EvalExpr(expr.substr(i + 1, close - i - 1));
                prev_was_value = (c == '(' || c == ']');
                if (c == '[' || c == '(') {
                    prev_was_value = true;
                }
                i = close + 1;
                continue;
            }
            if (IsIdentStartChar(c)) {
                size_t back = i;
                while (back > 0 && std::isspace(static_cast<unsigned char>(expr[back - 1])) != 0) {
                    --back;
                }
                const bool after_member =
                    back > 0 && (expr[back - 1] == '.' ||
                                 (back >= 2 && expr[back - 2] == '-' && expr[back - 1] == '>'));
                size_t id_end = i + 1;
                while (id_end < expr.size() && IsIdentContChar(expr[id_end])) {
                    ++id_end;
                }
                std::string_view id = expr.substr(i, id_end - i);
                size_t cursor = id_end;
                if (!after_member && (id == "fast" || id == "precise" || id == "metal")) {
                    const size_t ns_pos = SkipMslTrivia(expr, cursor, expr.size());
                    if (ns_pos + 1 < expr.size() && expr[ns_pos] == ':' &&
                        expr[ns_pos + 1] == ':') {
                        const size_t fn_start = SkipMslTrivia(expr, ns_pos + 2, expr.size());
                        if (fn_start < expr.size() && IsIdentStartChar(expr[fn_start])) {
                            size_t fn_end = fn_start + 1;
                            while (fn_end < expr.size() && IsIdentContChar(expr[fn_end])) {
                                ++fn_end;
                            }
                            id = expr.substr(fn_start, fn_end - fn_start);
                            cursor = fn_end;
                        }
                    }
                }
                size_t after_id = SkipMslTrivia(expr, cursor, expr.size());
                if (!after_member && IsTemplateCallName(id) && after_id < expr.size() &&
                    expr[after_id] == '<') {
                    int angles = 0;
                    size_t t = after_id;
                    while (t < expr.size()) {
                        const size_t sk = SkipMslCommentOrLit(expr, t);
                        if (sk != t) {
                            t = sk;
                            continue;
                        }
                        if (expr[t] == '(' || expr[t] == '[' || expr[t] == '{') {
                            const size_t ao = static_cast<size_t>(expr.data() - msl.data()) + t;
                            if (ao >= match.size() || match[ao] == ~0U) {
                                alias_hazard = true;
                                return Taint::Varying();
                            }
                            t = match[ao] - static_cast<size_t>(expr.data() - msl.data()) + 1;
                            continue;
                        }
                        if (expr[t] == '<') {
                            ++angles;
                        } else if (expr[t] == '>') {
                            if (--angles == 0) {
                                ++t;
                                break;
                            }
                        }
                        ++t;
                    }
                    if (angles != 0) {
                        alias_hazard = true;
                        return Taint::Varying();
                    }
                    after_id = SkipMslTrivia(expr, t, expr.size());
                }
                if (after_id < expr.size() && expr[after_id] == '(') {
                    const size_t abs_open =
                        static_cast<size_t>(expr.data() - msl.data()) + after_id;
                    if (abs_open >= match.size() || match[abs_open] == ~0U) {
                        alias_hazard = true;
                        return Taint::Varying();
                    }
                    const size_t close =
                        match[abs_open] - static_cast<size_t>(expr.data() - msl.data());
                    if (close >= expr.size()) {
                        alias_hazard = true;
                        return Taint::Varying();
                    }
                    const std::string_view args_slice =
                        expr.substr(after_id + 1, close - after_id - 1);
                    const Taint args_taint = EvalExpr(args_slice);
                    if (after_member) {
                        if (IsPureMemberMethod(id)) {
                            res |= args_taint;
                        } else {
                            res |= Taint::Varying();
                        }
                    } else if (IsPureBuiltinOrCtor(id)) {
                        res |= args_taint;
                    } else {
                        res |= Taint::Varying();
                        TaintCallArguments(args_slice);
                    }
                    prev_was_value = true;
                    i = close + 1;
                    continue;
                }
                if (after_member) {
                    prev_was_value = true;
                    i = cursor;
                    continue;
                }
                if (!stage_in_var.empty() && id == stage_in_var) {
                    if (after_id < expr.size() && expr[after_id] == '.') {
                        const size_t f_start = SkipMslTrivia(expr, after_id + 1, expr.size());
                        if (f_start < expr.size() && IsIdentStartChar(expr[f_start])) {
                            size_t f_end = f_start + 1;
                            while (f_end < expr.size() && IsIdentContChar(expr[f_end])) {
                                ++f_end;
                            }
                            const std::string_view field = expr.substr(f_start, f_end - f_start);
                            const auto it = in_field_taint.find(field);
                            res |= it != in_field_taint.end() ? it->second : Taint::Varying();
                            prev_was_value = true;
                            i = f_end;
                            continue;
                        }
                    }
                    res |= Taint::Varying();
                    prev_was_value = true;
                    i = cursor;
                    continue;
                }
                if (!out_var.empty() && id == out_var) {
                    if (after_id < expr.size() && expr[after_id] == '.') {
                        const size_t f_start = SkipMslTrivia(expr, after_id + 1, expr.size());
                        if (f_start < expr.size() && IsIdentStartChar(expr[f_start])) {
                            size_t f_end = f_start + 1;
                            while (f_end < expr.size() && IsIdentContChar(expr[f_end])) {
                                ++f_end;
                            }
                            const std::string_view field = expr.substr(f_start, f_end - f_start);
                            const auto it = out_field_locations.find(field);
                            if (it == out_field_locations.end()) {
                                alias_hazard = true;
                                return Taint::Varying();
                            }
                            u32 loc_mask = it->second;
                            if (loc_mask == 0) {
                                if (const auto bit_it = out_builtin_taint.find(field);
                                    bit_it != out_builtin_taint.end()) {
                                    res |= bit_it->second;
                                }
                            } else {
                                while (loc_mask != 0) {
                                    const u32 g = static_cast<u32>(std::countr_zero(loc_mask));
                                    loc_mask &= loc_mask - 1U;
                                    res |= generic_taint[g];
                                }
                            }
                            prev_was_value = true;
                            i = f_end;
                            continue;
                        }
                    }
                    alias_hazard = true;
                    return Taint::Varying();
                }
                if (const auto it = var_taint.find(id); it != var_taint.end()) {
                    res |= it->second;
                    prev_was_value = true;
                } else {
                    prev_was_value = !IsTypeOrQualifier(id);
                }
                i = cursor;
                continue;
            }
            prev_was_value = false;
            ++i;
        }
        return res;
    }

    void TaintCallArguments(std::string_view args) {
        for (size_t i = 0; i < args.size();) {
            const size_t sk = SkipMslCommentOrLit(args, i);
            if (sk != i) {
                i = sk;
                continue;
            }
            if (IsIdentStartChar(args[i])) {
                size_t back = i;
                while (back > 0 && std::isspace(static_cast<unsigned char>(args[back - 1])) != 0) {
                    --back;
                }
                const bool member =
                    back > 0 && (args[back - 1] == '.' ||
                                 (back >= 2 && args[back - 2] == '-' && args[back - 1] == '>'));
                size_t end = i + 1;
                while (end < args.size() && IsIdentContChar(args[end])) {
                    ++end;
                }
                if (!member) {
                    const std::string_view tok = args.substr(i, end - i);
                    if ((!out_var.empty() && tok == out_var) ||
                        (!stage_in_var.empty() && tok == stage_in_var)) {
                        alias_hazard = true;
                        return;
                    }
                    if (const auto it = var_taint.find(tok); it != var_taint.end()) {
                        it->second = Taint::Varying();
                    }
                }
                i = end;
                continue;
            }
            ++i;
        }
    }

    void PreTaintLoopRange(size_t begin, size_t end) {
        const std::string_view span = msl.substr(begin, end - begin);
        if (ContainsStandaloneIdent(span, "return")) {
            post_return_taint = Taint::Varying();
        }
        for (size_t i = begin; i < end && !alias_hazard;) {
            const size_t sk = SkipMslCommentOrLit(msl, i);
            if (sk != i) {
                i = sk;
                continue;
            }
            if ((msl[i] == '+' && i + 1 < end && msl[i + 1] == '+') ||
                (msl[i] == '-' && i + 1 < end && msl[i + 1] == '-')) {
                const size_t after = SkipMslTrivia(msl, i + 2, end);
                if (after < end && IsIdentStartChar(msl[after])) {
                    size_t id_end = after + 1;
                    while (id_end < end && IsIdentContChar(msl[id_end])) {
                        ++id_end;
                    }
                    const std::string_view id = msl.substr(after, id_end - after);
                    if (!out_var.empty() && id == out_var) {
                        alias_hazard = true;
                        return;
                    }
                    var_taint[id] = Taint::Varying();
                }
                i += 2;
                continue;
            }
            if (IsIdentStartChar(msl[i])) {
                size_t back = i;
                while (back > begin &&
                       std::isspace(static_cast<unsigned char>(msl[back - 1])) != 0) {
                    --back;
                }
                const bool member =
                    back > begin && (msl[back - 1] == '.' ||
                                     (back >= begin + 2 && msl[back - 2] == '-' &&
                                      msl[back - 1] == '>'));
                size_t id_end = i + 1;
                while (id_end < end && IsIdentContChar(msl[id_end])) {
                    ++id_end;
                }
                const std::string_view id = msl.substr(i, id_end - i);
                if (!member && id != "if" && id != "for" && id != "while" && id != "switch" &&
                    id != "return" && id != "break" && id != "continue" &&
                    !IsTypeOrQualifier(id)) {
                    size_t cur = id_end;
                    while (cur < end) {
                        cur = SkipMslTrivia(msl, cur, end);
                        if (cur < end && msl[cur] == '[' && match[cur] < end) {
                            cur = size_t{match[cur]} + 1;
                            continue;
                        }
                        if (cur < end && msl[cur] == '.') {
                            cur = SkipMslTrivia(msl, cur + 1, end);
                            while (cur < end && IsIdentContChar(msl[cur])) {
                                ++cur;
                            }
                            continue;
                        }
                        break;
                    }
                    bool mutated = false;
                    if (cur + 1 < end &&
                        ((msl[cur] == '+' && msl[cur + 1] == '+') ||
                         (msl[cur] == '-' && msl[cur + 1] == '-'))) {
                        mutated = true;
                    } else if (cur < end && msl[cur] == '=' &&
                               (cur + 1 >= end || msl[cur + 1] != '=')) {
                        mutated = true;
                    } else if (cur + 1 < end && msl[cur + 1] == '=' &&
                               (msl[cur] == '+' || msl[cur] == '-' || msl[cur] == '*' ||
                                msl[cur] == '/' || msl[cur] == '%' || msl[cur] == '&' ||
                                msl[cur] == '|' || msl[cur] == '^')) {
                        mutated = true;
                    } else if (cur + 2 < end && msl[cur + 2] == '=' &&
                               ((msl[cur] == '<' && msl[cur + 1] == '<') ||
                                (msl[cur] == '>' && msl[cur + 1] == '>'))) {
                        mutated = true;
                    } else if (cur < end && msl[cur] == ';') {
                        // Potentially a declaration `Type id;` inside the loop.
                        if (back > begin && IsIdentContChar(msl[back - 1])) {
                            mutated = true;
                        }
                    }
                    if (mutated && (out_var.empty() || id != out_var)) {
                        var_taint[id] = Taint::Varying();
                    }
                }
                i = id_end;
                continue;
            }
            ++i;
        }
    }

    bool AnalyzeSingleStmt(size_t& pos, size_t limit, Taint control_taint, bool is_top_level,
                           Taint* switch_break_taint) {
        pos = SkipMslTrivia(msl, pos, limit);
        if (pos >= limit) {
            return true;
        }
        if (msl[pos] == ';') {
            ++pos;
            return true;
        }
        if (msl[pos] == '#') {
            if (msl.substr(pos).starts_with("#pragma")) {
                const size_t nl = msl.find('\n', pos);
                pos = nl == std::string_view::npos ? limit : (std::min)(nl + 1, limit);
                return true;
            }
            alias_hazard = true;
            return false;
        }
        if (msl[pos] == '{') {
            if (match[pos] >= limit) {
                alias_hazard = true;
                return false;
            }
            const size_t close = match[pos];
            if (!AnalyzeBlock(pos + 1, close, control_taint, false, switch_break_taint)) {
                return false;
            }
            pos = close + 1;
            return true;
        }
        if (IsIdentStartChar(msl[pos])) {
            size_t kw_end = pos + 1;
            while (kw_end < limit && IsIdentContChar(msl[kw_end])) {
                ++kw_end;
            }
            const std::string_view kw = msl.substr(pos, kw_end - pos);
            if (kw == "if") {
                const size_t p_open = SkipMslTrivia(msl, kw_end, limit);
                if (p_open >= limit || msl[p_open] != '(' || match[p_open] >= limit) {
                    alias_hazard = true;
                    return false;
                }
                const size_t p_close = match[p_open];
                const Taint eff_ctrl =
                    control_taint | post_return_taint |
                    (switch_break_taint != nullptr ? *switch_break_taint : Taint{});
                const Taint cond_t =
                    EvalExpr(msl.substr(p_open + 1, p_close - p_open - 1)) | eff_ctrl;
                if (alias_hazard) {
                    return false;
                }
                pos = p_close + 1;
                if (!AnalyzeSingleStmt(pos, limit, cond_t, false, switch_break_taint)) {
                    return false;
                }
                const size_t after_then = SkipMslTrivia(msl, pos, limit);
                if (after_then + 4 <= limit && msl.substr(after_then, 4) == "else" &&
                    (after_then + 4 == limit || !IsIdentContChar(msl[after_then + 4]))) {
                    pos = after_then + 4;
                    if (!AnalyzeSingleStmt(pos, limit, cond_t, false, switch_break_taint)) {
                        return false;
                    }
                }
                return true;
            }
            if (kw == "switch") {
                const size_t p_open = SkipMslTrivia(msl, kw_end, limit);
                if (p_open >= limit || msl[p_open] != '(' || match[p_open] >= limit) {
                    alias_hazard = true;
                    return false;
                }
                const size_t p_close = match[p_open];
                const size_t b_open = SkipMslTrivia(msl, p_close + 1, limit);
                if (b_open >= limit || msl[b_open] != '{' || match[b_open] >= limit) {
                    alias_hazard = true;
                    return false;
                }
                const size_t b_close = match[b_open];
                const Taint eff_ctrl =
                    control_taint | post_return_taint |
                    (switch_break_taint != nullptr ? *switch_break_taint : Taint{});
                const Taint sw_t =
                    EvalExpr(msl.substr(p_open + 1, p_close - p_open - 1)) | eff_ctrl;
                Taint local_break{};
                if (!AnalyzeBlock(b_open + 1, b_close, sw_t, false, &local_break)) {
                    return false;
                }
                pos = b_close + 1;
                return true;
            }
            if (kw == "for" || kw == "while") {
                const size_t p_open = SkipMslTrivia(msl, kw_end, limit);
                if (p_open >= limit || msl[p_open] != '(' || match[p_open] >= limit) {
                    alias_hazard = true;
                    return false;
                }
                const size_t p_close = match[p_open];
                size_t body_pos = SkipMslTrivia(msl, p_close + 1, limit);
                if (body_pos >= limit) {
                    alias_hazard = true;
                    return false;
                }
                size_t loop_end = 0;
                if (msl[body_pos] == '{') {
                    if (match[body_pos] >= limit) {
                        alias_hazard = true;
                        return false;
                    }
                    loop_end = size_t{match[body_pos]} + 1;
                } else {
                    size_t scan = body_pos;
                    while (scan < limit && msl[scan] != ';') {
                        const size_t sk = SkipMslCommentOrLit(msl, scan);
                        if (sk != scan) {
                            scan = sk;
                            continue;
                        }
                        if ((msl[scan] == '(' || msl[scan] == '[' || msl[scan] == '{') &&
                            match[scan] < limit) {
                            scan = size_t{match[scan]} + 1;
                            continue;
                        }
                        ++scan;
                    }
                    if (scan >= limit) {
                        alias_hazard = true;
                        return false;
                    }
                    loop_end = scan + 1;
                }
                PreTaintLoopRange(pos, loop_end);
                if (alias_hazard) {
                    return false;
                }
                pos = body_pos;
                if (!AnalyzeSingleStmt(pos, loop_end, Taint::Varying(), false, nullptr)) {
                    return false;
                }
                pos = loop_end;
                return true;
            }
            if (kw == "do") {
                const size_t b_open = SkipMslTrivia(msl, kw_end, limit);
                if (b_open >= limit || msl[b_open] != '{' || match[b_open] >= limit) {
                    alias_hazard = true;
                    return false;
                }
                const size_t b_close = match[b_open];
                const size_t while_kw = SkipMslTrivia(msl, b_close + 1, limit);
                if (while_kw + 5 > limit || msl.substr(while_kw, 5) != "while" ||
                    (while_kw + 5 < limit && IsIdentContChar(msl[while_kw + 5]))) {
                    alias_hazard = true;
                    return false;
                }
                const size_t p_open = SkipMslTrivia(msl, while_kw + 5, limit);
                if (p_open >= limit || msl[p_open] != '(' || match[p_open] >= limit) {
                    alias_hazard = true;
                    return false;
                }
                const size_t semi = SkipMslTrivia(msl, size_t{match[p_open]} + 1, limit);
                if (semi >= limit || msl[semi] != ';') {
                    alias_hazard = true;
                    return false;
                }
                const size_t loop_end = semi + 1;
                PreTaintLoopRange(pos, loop_end);
                if (alias_hazard) {
                    return false;
                }
                if (!AnalyzeBlock(b_open + 1, b_close, Taint::Varying(), false, nullptr)) {
                    return false;
                }
                static_cast<void>(EvalExpr(msl.substr(p_open + 1, match[p_open] - p_open - 1)));
                pos = loop_end;
                return !alias_hazard;
            }
            if ((kw == "case" || kw == "default") && switch_break_taint != nullptr) {
                size_t colon = kw_end;
                while (colon < limit && msl[colon] != ':') {
                    const size_t sk = SkipMslCommentOrLit(msl, colon);
                    if (sk != colon) {
                        colon = sk;
                        continue;
                    }
                    if ((msl[colon] == '(' || msl[colon] == '[' || msl[colon] == '{') &&
                        match[colon] < limit) {
                        colon = size_t{match[colon]} + 1;
                        continue;
                    }
                    ++colon;
                }
                if (colon >= limit) {
                    alias_hazard = true;
                    return false;
                }
                if (kw == "case") {
                    *switch_break_taint |= EvalExpr(msl.substr(kw_end, colon - kw_end));
                }
                pos = colon + 1;
                return !alias_hazard;
            }
        }

        // Scan to terminating `;` at current bracket depth.
        size_t semi = pos;
        while (semi < limit && msl[semi] != ';') {
            const size_t sk = SkipMslCommentOrLit(msl, semi);
            if (sk != semi) {
                semi = sk;
                continue;
            }
            if ((msl[semi] == '(' || msl[semi] == '[' || msl[semi] == '{') &&
                match[semi] < limit) {
                semi = size_t{match[semi]} + 1;
                continue;
            }
            ++semi;
        }
        if (semi >= limit) {
            alias_hazard = true;
            return false;
        }
        const std::string_view stmt = TrimWs(msl.substr(pos, semi - pos));
        pos = semi + 1;
        if (stmt.empty()) {
            return true;
        }

        const Taint eff_ctrl =
            control_taint | post_return_taint |
            (switch_break_taint != nullptr ? *switch_break_taint : Taint{});

        if (stmt == "break") {
            if (switch_break_taint != nullptr) {
                *switch_break_taint |= eff_ctrl;
            }
            return true;
        }
        if (stmt == "continue") {
            return true;
        }
        if (stmt.starts_with("return") &&
            (stmt.size() == 6 || !IsIdentContChar(stmt[6]))) {
            const std::string_view ret_expr = TrimWs(stmt.substr(6));
            if (out_var.empty() || ret_expr != out_var) {
                alias_hazard = true;
                return false;
            }
            seen_return = true;
            post_return_taint |= eff_ctrl;
            return true;
        }

        // Output struct variable declaration: `<out_struct> <out_var> = {};`
        if (stmt.starts_with(out_struct_name) && stmt.size() > out_struct_name.size() &&
            !IsIdentContChar(stmt[out_struct_name.size()])) {
            size_t eq_b = 0;
            size_t eq_e = 0;
            if (!out_var.empty() || !FindTopLevelAssign(stmt, eq_b, eq_e) ||
                stmt.substr(eq_b, eq_e - eq_b) != "=") {
                alias_hazard = true;
                return false;
            }
            const std::string_view decl_name =
                TrimWs(stmt.substr(out_struct_name.size(), eq_b - out_struct_name.size()));
            const std::string_view init_rhs = TrimWs(stmt.substr(eq_e));
            if (decl_name.empty() || init_rhs != "{}") {
                alias_hazard = true;
                return false;
            }
            for (const char ch : decl_name) {
                if (!IsIdentContChar(ch)) {
                    alias_hazard = true;
                    return false;
                }
            }
            out_var = std::string{decl_name};
            return true;
        }

        // Spilled read-only resource binding at entry prologue (`constant ...` / `const device ...`).
        if (stmt.starts_with("constant ") || stmt.starts_with("const device ")) {
            size_t eq_b = 0;
            size_t eq_e = 0;
            if (!FindTopLevelAssign(stmt, eq_b, eq_e) ||
                stmt.substr(eq_b, eq_e - eq_b) != "=") {
                alias_hazard = true;
                return false;
            }
            const std::string_view lhs = TrimWs(stmt.substr(0, eq_b));
            const std::string_view rhs = TrimWs(stmt.substr(eq_e));
            size_t ns = lhs.size();
            while (ns > 0 && IsIdentContChar(lhs[ns - 1])) {
                --ns;
            }
            if (ns == lhs.size() || ns == 0) {
                alias_hazard = true;
                return false;
            }
            const std::string_view vname = lhs.substr(ns);
            var_taint[vname] = EvalExpr(rhs) | eff_ctrl;
            return !alias_hazard;
        }

        size_t eq_b = 0;
        size_t eq_e = 0;
        if (FindTopLevelAssign(stmt, eq_b, eq_e)) {
            const std::string_view lhs = TrimWs(stmt.substr(0, eq_b));
            const std::string_view op = stmt.substr(eq_b, eq_e - eq_b);
            const std::string_view rhs = TrimWs(stmt.substr(eq_e));
            size_t rhs_eq_b = 0;
            size_t rhs_eq_e = 0;
            if (FindTopLevelAssign(rhs, rhs_eq_b, rhs_eq_e)) {
                alias_hazard = true;
                return false;
            }
            // Write to `out.<field>[...].swizzle <op> <rhs>`
            if (!out_var.empty() && lhs.starts_with(out_var) &&
                (lhs.size() == out_var.size() || !IsIdentContChar(lhs[out_var.size()]))) {
                size_t cur = SkipMslTrivia(lhs, out_var.size(), lhs.size());
                if (cur >= lhs.size() || lhs[cur] != '.') {
                    alias_hazard = true;
                    return false;
                }
                cur = SkipMslTrivia(lhs, cur + 1, lhs.size());
                size_t f_end = cur;
                while (f_end < lhs.size() && IsIdentContChar(lhs[f_end])) {
                    ++f_end;
                }
                if (f_end == cur) {
                    alias_hazard = true;
                    return false;
                }
                const std::string_view field = lhs.substr(cur, f_end - cur);
                const auto loc_it = out_field_locations.find(field);
                if (loc_it == out_field_locations.end()) {
                    alias_hazard = true;
                    return false;
                }
                Taint idx_taint{};
                cur = f_end;
                while (cur < lhs.size()) {
                    cur = SkipMslTrivia(lhs, cur, lhs.size());
                    if (cur >= lhs.size()) {
                        break;
                    }
                    if (lhs[cur] == '[') {
                        const size_t ao = static_cast<size_t>(lhs.data() - msl.data()) + cur;
                        if (ao >= match.size() || match[ao] == ~0U) {
                            alias_hazard = true;
                            return false;
                        }
                        const size_t close =
                            match[ao] - static_cast<size_t>(lhs.data() - msl.data());
                        if (close >= lhs.size()) {
                            alias_hazard = true;
                            return false;
                        }
                        idx_taint |= EvalExpr(lhs.substr(cur + 1, close - cur - 1));
                        cur = close + 1;
                        continue;
                    }
                    if (lhs[cur] == '.') {
                        cur = SkipMslTrivia(lhs, cur + 1, lhs.size());
                        size_t sw_end = cur;
                        while (sw_end < lhs.size() && IsIdentContChar(lhs[sw_end])) {
                            ++sw_end;
                        }
                        if (sw_end == cur) {
                            alias_hazard = true;
                            return false;
                        }
                        cur = sw_end;
                        continue;
                    }
                    alias_hazard = true;
                    return false;
                }
                const Taint val_taint = EvalExpr(rhs) | idx_taint | eff_ctrl;
                if (alias_hazard) {
                    return false;
                }
                u32 loc_mask = loc_it->second;
                if (loc_mask == 0) {
                    out_builtin_taint[field] |= val_taint;
                } else {
                    while (loc_mask != 0) {
                        const u32 g = static_cast<u32>(std::countr_zero(loc_mask));
                        loc_mask &= loc_mask - 1U;
                        generic_taint[g] |= val_taint;
                    }
                }
                return true;
            }
            if (!out_var.empty() && ContainsStandaloneIdent(lhs, out_var)) {
                alias_hazard = true;
                return false;
            }
            // Check for multi-declarator `Type a = e1, b = e2` vs single LHS.
            const std::vector<std::string_view> comma_parts = SplitTopCommas(stmt);
            if (comma_parts.size() > 1) {
                if (!out_var.empty() && ContainsStandaloneIdent(stmt, out_var)) {
                    alias_hazard = true;
                    return false;
                }
                for (const std::string_view part : comma_parts) {
                    if (part.find('&') != std::string_view::npos ||
                        part.find('*') != std::string_view::npos) {
                        alias_hazard = true;
                        return false;
                    }
                    size_t peq_b = 0;
                    size_t peq_e = 0;
                    if (FindTopLevelAssign(part, peq_b, peq_e)) {
                        std::string_view plhs = TrimWs(part.substr(0, peq_b));
                        while (!plhs.empty() && plhs.back() == ']') {
                            const size_t ob = plhs.rfind('[');
                            if (ob == std::string_view::npos) {
                                alias_hazard = true;
                                return false;
                            }
                            plhs = TrimWs(plhs.substr(0, ob));
                        }
                        size_t ns = plhs.size();
                        while (ns > 0 && IsIdentContChar(plhs[ns - 1])) {
                            --ns;
                        }
                        if (ns == plhs.size()) {
                            alias_hazard = true;
                            return false;
                        }
                        var_taint[plhs.substr(ns)] |= EvalExpr(part.substr(peq_e)) | eff_ctrl;
                    } else {
                        std::string_view plhs = TrimWs(part);
                        while (!plhs.empty() && plhs.back() == ']') {
                            const size_t ob = plhs.rfind('[');
                            if (ob == std::string_view::npos) {
                                alias_hazard = true;
                                return false;
                            }
                            plhs = TrimWs(plhs.substr(0, ob));
                        }
                        size_t ns = plhs.size();
                        while (ns > 0 && IsIdentContChar(plhs[ns - 1])) {
                            --ns;
                        }
                        if (ns < plhs.size()) {
                            var_taint[plhs.substr(ns)] |= eff_ctrl;
                        }
                    }
                }
                return !alias_hazard;
            }

            // Distinguish local declaration `Type var[N] = rhs` from assignment `var[idx].swiz = rhs`.
            std::string_view stripped_lhs = lhs;
            Taint lhs_idx_taint{};
            while (!stripped_lhs.empty() && stripped_lhs.back() == ']') {
                const size_t close_pos =
                    static_cast<size_t>(stripped_lhs.data() - msl.data()) + stripped_lhs.size() - 1;
                if (close_pos >= match.size() || match[close_pos] == ~0U) {
                    alias_hazard = true;
                    return false;
                }
                const size_t open_off =
                    match[close_pos] - static_cast<size_t>(stripped_lhs.data() - msl.data());
                if (open_off >= stripped_lhs.size()) {
                    alias_hazard = true;
                    return false;
                }
                lhs_idx_taint |= EvalExpr(
                    stripped_lhs.substr(open_off + 1, stripped_lhs.size() - open_off - 2));
                stripped_lhs = TrimWs(stripped_lhs.substr(0, open_off));
            }
            if (stripped_lhs.find('.') == std::string_view::npos &&
                stripped_lhs.find("->") == std::string_view::npos && op == "=") {
                size_t ns = stripped_lhs.size();
                while (ns > 0 && IsIdentContChar(stripped_lhs[ns - 1])) {
                    --ns;
                }
                if (ns > 0 && ns < stripped_lhs.size()) {
                    const std::string_view prefix = TrimWs(stripped_lhs.substr(0, ns));
                    if (!prefix.empty()) {
                        if (prefix.find('&') != std::string_view::npos ||
                            prefix.find('*') != std::string_view::npos) {
                            alias_hazard = true;
                            return false;
                        }
                        const std::string_view vname = stripped_lhs.substr(ns);
                        var_taint[vname] = EvalExpr(rhs) | lhs_idx_taint | eff_ctrl;
                        return !alias_hazard;
                    }
                }
            }

            // Assignment to existing variable `root_var[...].swiz <op> rhs`.
            if (!IsIdentStartChar(lhs[0])) {
                alias_hazard = true;
                return false;
            }
            size_t r_end = 1;
            while (r_end < lhs.size() && IsIdentContChar(lhs[r_end])) {
                ++r_end;
            }
            const std::string_view root_var = lhs.substr(0, r_end);
            Taint idx_taint{};
            size_t cur = r_end;
            while (cur < lhs.size()) {
                cur = SkipMslTrivia(lhs, cur, lhs.size());
                if (cur >= lhs.size()) {
                    break;
                }
                if (lhs[cur] == '[') {
                    const size_t ao = static_cast<size_t>(lhs.data() - msl.data()) + cur;
                    if (ao >= match.size() || match[ao] == ~0U) {
                        alias_hazard = true;
                        return false;
                    }
                    const size_t close =
                        match[ao] - static_cast<size_t>(lhs.data() - msl.data());
                    if (close >= lhs.size()) {
                        alias_hazard = true;
                        return false;
                    }
                    idx_taint |= EvalExpr(lhs.substr(cur + 1, close - cur - 1));
                    cur = close + 1;
                    continue;
                }
                if (lhs[cur] == '.') {
                    cur = SkipMslTrivia(lhs, cur + 1, lhs.size());
                    size_t f_end = cur;
                    while (f_end < lhs.size() && IsIdentContChar(lhs[f_end])) {
                        ++f_end;
                    }
                    if (f_end == cur) {
                        alias_hazard = true;
                        return false;
                    }
                    cur = f_end;
                    continue;
                }
                alias_hazard = true;
                return false;
            }
            const Taint val_taint = EvalExpr(rhs) | idx_taint | eff_ctrl;
            if (alias_hazard) {
                return false;
            }
            const bool full_overwrite = is_top_level && !eff_ctrl.uses_vertex_id &&
                                        eff_ctrl.attr_mask == 0 && op == "=" && lhs == root_var;
            if (full_overwrite) {
                var_taint[root_var] = val_taint;
            } else {
                var_taint[root_var] |= val_taint;
            }
            return true;
        }

        // Non-assignment statement (`Type x;`, `++i;`, `foo();`, etc.).
        if (!out_var.empty() && ContainsStandaloneIdent(stmt, out_var)) {
            alias_hazard = true;
            return false;
        }
        // Check for standalone `++v` / `v++` / `--v` / `v--`.
        if (stmt.find("++") != std::string_view::npos ||
            stmt.find("--") != std::string_view::npos) {
            for (size_t i = 0; i < stmt.size();) {
                if (IsIdentStartChar(stmt[i])) {
                    size_t e = i + 1;
                    while (e < stmt.size() && IsIdentContChar(stmt[e])) {
                        ++e;
                    }
                    var_taint[stmt.substr(i, e - i)] |= eff_ctrl;
                    i = e;
                } else {
                    ++i;
                }
            }
            return true;
        }
        // Uninitialized declaration `Type var[N];` (no top-level `(`).
        bool has_paren = false;
        for (size_t i = 0; i < stmt.size(); ++i) {
            if (stmt[i] == '(') {
                has_paren = true;
                break;
            }
        }
        if (!has_paren) {
            if (stmt.find('&') != std::string_view::npos ||
                stmt.find('*') != std::string_view::npos) {
                alias_hazard = true;
                return false;
            }
            for (const std::string_view part : SplitTopCommas(stmt)) {
                std::string_view p = part;
                while (!p.empty() && p.back() == ']') {
                    const size_t ob = p.rfind('[');
                    if (ob == std::string_view::npos) {
                        alias_hazard = true;
                        return false;
                    }
                    p = TrimWs(p.substr(0, ob));
                }
                size_t ns = p.size();
                while (ns > 0 && IsIdentContChar(p[ns - 1])) {
                    --ns;
                }
                if (ns < p.size()) {
                    var_taint[p.substr(ns)] |= eff_ctrl;
                }
            }
            return true;
        }
        static_cast<void>(EvalExpr(stmt));
        return !alias_hazard;
    }

    bool AnalyzeBlock(size_t begin, size_t end, Taint control_taint, bool is_top_level,
                      Taint* switch_break_taint) {
        size_t pos = begin;
        while (pos < end) {
            pos = SkipMslTrivia(msl, pos, end);
            if (pos >= end) {
                break;
            }
            if (!AnalyzeSingleStmt(pos, end, control_taint, is_top_level, switch_break_taint)) {
                return false;
            }
        }
        return !alias_hazard;
    }

private:
    std::string_view msl;
    const std::vector<u32>& match;
};

} // Anonymous namespace

VertexOutputInvariance AnalyzeVertexOutputInvariance(std::string_view msl,
                                                     std::string_view entry_point) {
    VertexOutputInvariance conservative{};
    conservative.uses_vertex_id_mask = ~0U;
    if (msl.empty() || entry_point.empty()) {
        return conservative;
    }
    const std::vector<u32> match = BuildMslBracketTable(msl);
    if (match.empty()) {
        return conservative;
    }

    // Locate `vertex <out_struct> <entry_point>(<params>) { <body> }`.
    size_t params_open = ~size_t{0};
    size_t params_close = ~size_t{0};
    size_t body_open = ~size_t{0};
    size_t body_close = ~size_t{0};
    std::string out_struct;
    for (size_t pos = 0; pos < msl.size();) {
        const size_t sk = SkipMslCommentOrLit(msl, pos);
        if (sk != pos) {
            pos = sk;
            continue;
        }
        if (IsIdentStartChar(msl[pos])) {
            size_t end = pos + 1;
            while (end < msl.size() && IsIdentContChar(msl[end])) {
                ++end;
            }
            if (msl.substr(pos, end - pos) == "vertex") {
                const size_t ret_start = SkipMslTrivia(msl, end, msl.size());
                size_t ret_end = ret_start;
                while (ret_end < msl.size() && IsIdentContChar(msl[ret_end])) {
                    ++ret_end;
                }
                const size_t fn_start = SkipMslTrivia(msl, ret_end, msl.size());
                size_t fn_end = fn_start;
                while (fn_end < msl.size() && IsIdentContChar(msl[fn_end])) {
                    ++fn_end;
                }
                const size_t p_open = SkipMslTrivia(msl, fn_end, msl.size());
                if (ret_end > ret_start && msl.substr(fn_start, fn_end - fn_start) == entry_point &&
                    p_open < msl.size() && msl[p_open] == '(' && match[p_open] != ~0U) {
                    const size_t p_close = match[p_open];
                    size_t after_p = SkipMslTrivia(msl, p_close + 1, msl.size());
                    while (after_p + 1 < msl.size() && msl[after_p] == '[' &&
                           msl[after_p + 1] == '[' && match[after_p] != ~0U) {
                        after_p = SkipMslTrivia(msl, size_t{match[after_p]} + 1, msl.size());
                    }
                    if (after_p < msl.size() && msl[after_p] == '{' && match[after_p] != ~0U) {
                        out_struct = std::string{msl.substr(ret_start, ret_end - ret_start)};
                        params_open = p_open;
                        params_close = p_close;
                        body_open = after_p;
                        body_close = match[after_p];
                        break;
                    }
                }
            }
            pos = end;
            continue;
        }
        ++pos;
    }
    if (body_open == ~size_t{0} || out_struct.empty() || out_struct == "void") {
        return conservative;
    }

    VertexMslAnalyzer analyzer{msl, match};
    analyzer.out_struct_name = out_struct;

    // Parse entry parameters and their `[[...]]` attributes.
    std::string stage_in_struct;
    for (const std::string_view raw_param :
         SplitTopCommas(msl.substr(params_open + 1, params_close - params_open - 1))) {
        std::string param_copy{raw_param};
        std::vector<std::string> attrs;
        for (size_t a = param_copy.find("[["); a != std::string::npos;
             a = param_copy.find("[[")) {
            const size_t ae = param_copy.find("]]", a + 2);
            if (ae == std::string::npos) {
                return conservative;
            }
            attrs.emplace_back(
                TrimWs(std::string_view{param_copy}.substr(a + 2, ae - a - 2)));
            param_copy.erase(a, ae + 2 - a);
        }
        if (attrs.empty()) {
            return conservative;
        }
        const std::string_view decl = TrimWs(param_copy);
        size_t ns = decl.size();
        while (ns > 0 && IsIdentContChar(decl[ns - 1])) {
            --ns;
        }
        if (ns == decl.size() || ns == 0) {
            return conservative;
        }
        const std::string_view param_name = decl.substr(ns);
        const std::string_view param_type = TrimWs(decl.substr(0, ns));
        const std::string_view raw_name =
            raw_param.substr(raw_param.rfind(param_name), param_name.size());

        bool is_stage_in = false;
        bool is_uniform_or_inst = false;
        for (const std::string& attr : attrs) {
            if (attr == "stage_in") {
                is_stage_in = true;
            } else if (attr == "instance_id" || attr == "base_instance" || attr == "base_vertex" ||
                       attr.starts_with("buffer(") || attr.starts_with("texture(") ||
                       attr.starts_with("sampler(")) {
                is_uniform_or_inst = true;
            } else {
                is_uniform_or_inst = false;
                break;
            }
        }
        if (is_stage_in) {
            if (!analyzer.stage_in_var.empty()) {
                return conservative;
            }
            analyzer.stage_in_var = std::string{param_name};
            std::string_view st = param_type;
            if (!st.empty() && st.back() == '&') {
                st = TrimWs(st.substr(0, st.size() - 1));
            }
            while (st.starts_with("const ") || st.starts_with("thread ")) {
                const size_t sp = st.find(' ');
                st = TrimWs(st.substr(sp + 1));
            }
            stage_in_struct = std::string{st};
        } else if (is_uniform_or_inst) {
            analyzer.var_taint[raw_name] = Taint{false, 0};
        } else {
            analyzer.var_taint[raw_name] = Taint::Varying();
        }
    }

    // Parse `main0_in` (`stage_in` struct) if present.
    std::vector<ParsedStructMember> in_members;
    if (!stage_in_struct.empty()) {
        if (!ParseMslStruct(msl, match, stage_in_struct, in_members)) {
            return conservative;
        }
        for (const ParsedStructMember& m : in_members) {
            Taint field_t = Taint::Varying();
            if (m.array_count == 1 && !IsMatrixOrArrayType(m.type)) {
                for (const std::string& attr : m.attrs) {
                    if (attr.starts_with("attribute(") && attr.back() == ')') {
                        u32 idx = 0;
                        if (ParseUnsignedUint(
                                std::string_view{attr}.substr(10, attr.size() - 11), idx) &&
                            idx < 32) {
                            field_t = Taint{false, 1U << idx};
                        }
                    }
                }
            }
            // Key by string_view into `in_members` (kept alive until function return).
            analyzer.in_field_taint[m.name] = field_t;
        }
    }

    // Parse `main0_out` (`out_struct`).
    std::vector<ParsedStructMember> out_members;
    if (!ParseMslStruct(msl, match, out_struct, out_members)) {
        return conservative;
    }
    for (const ParsedStructMember& m : out_members) {
        if (IsMatrixOrArrayType(m.type)) {
            return conservative;
        }
        bool recognized = false;
        u32 loc_mask = 0;
        for (const std::string& attr : m.attrs) {
            for (const std::string_view sub : SplitTopCommas(attr)) {
                if (sub.starts_with("user(locn") && sub.back() == ')') {
                    u32 loc = 0;
                    if (!ParseUnsignedUint(sub.substr(9, sub.size() - 10), loc)) {
                        return conservative;
                    }
                    recognized = true;
                    for (u32 k = 0; k < m.array_count; ++k) {
                        if (loc + k < 32) {
                            loc_mask |= 1U << (loc + k);
                        }
                    }
                } else if (sub == "position" || sub == "point_size" || sub == "clip_distance" ||
                           sub == "cull_distance" || sub == "render_target_array_index" ||
                           sub == "viewport_array_index" || sub == "invariant" || sub == "flat" ||
                           sub == "center_no_perspective" || sub == "centroid_perspective" ||
                           sub == "centroid_no_perspective" || sub == "sample_perspective" ||
                           sub == "sample_no_perspective") {
                    recognized = true;
                }
            }
        }
        if (!recognized) {
            return conservative;
        }
        analyzer.out_field_locations[m.name] = loc_mask;
    }

    if (!analyzer.AnalyzeBlock(body_open + 1, body_close, Taint{}, true, nullptr) ||
        analyzer.alias_hazard || analyzer.out_var.empty() || !analyzer.seen_return) {
        return conservative;
    }

    VertexOutputInvariance out{};
    out.uses_vertex_id_mask = 0;
    for (u32 g = 0; g < 32; ++g) {
        out.attr_deps[g] = analyzer.generic_taint[g].attr_mask;
        if (analyzer.generic_taint[g].uses_vertex_id) {
            out.uses_vertex_id_mask |= 1U << g;
        }
    }
    return out;
}

} // namespace Metal
