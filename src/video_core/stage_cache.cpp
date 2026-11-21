// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>
#include <type_traits>

#include "common/assert.h"
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/zstd_compression.h"
#include "shader_recompiler/exception.h"
#include "video_core/stage_cache.h"

namespace VideoCommon {
namespace {

enum ReadKind : u8 {
    CbufValue,
    StoragePointer,
    IndirectStoragePointer,
    TextureType,
    TexturePixelFormat,
    TextureInteger,
    ViewportTransform,
    Replacement,
};

// Past this, the cache starts over (translations hold program metadata and SPIR-V).
constexpr size_t MaxBytes = 512ULL << 20;

class Writer {
public:
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void Put(const T& value) {
        const auto* const begin = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    void PutBytes(const void* data, size_t len) {
        if (len != 0) {
            const auto* const begin = static_cast<const u8*>(data);
            bytes.insert(bytes.end(), begin, begin + len);
        }
    }

    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void PutVec(std::span<const T> span) {
        Put(static_cast<u32>(span.size()));
        PutBytes(span.data(), span.size() * sizeof(T));
    }

    std::vector<u8> bytes;
};

class Reader {
public:
    explicit Reader(std::span<const u8> data_) : data{data_} {}

    bool GetBytes(void* dst, size_t len) {
        if (len > data.size() - cursor) {
            return false;
        }
        if (len != 0) {
            std::memcpy(dst, data.data() + cursor, len);
            cursor += len;
        }
        return true;
    }

    template <typename T>
        requires std::is_trivially_copyable_v<T>
    bool Get(T& out) {
        return GetBytes(&out, sizeof(T));
    }

    template <typename Vec>
    bool GetVec(Vec& out, u32 max_elems) {
        using Elem = typename Vec::value_type;
        static_assert(std::is_trivially_copyable_v<Elem>);
        u32 count{};
        if (!Get(count) || count > max_elems ||
            static_cast<size_t>(count) * sizeof(Elem) > data.size() - cursor) {
            return false;
        }
        out.resize(count);
        return GetBytes(out.data(), static_cast<size_t>(count) * sizeof(Elem));
    }

    [[nodiscard]] bool Done() const noexcept {
        return cursor == data.size();
    }

private:
    std::span<const u8> data;
    size_t cursor{};
};

static_assert(std::is_trivially_copyable_v<Shader::ConstantBufferDescriptor>);
static_assert(std::is_trivially_copyable_v<Shader::StorageBufferDescriptor>);
static_assert(std::is_trivially_copyable_v<Shader::TextureBufferDescriptor>);
static_assert(std::is_trivially_copyable_v<Shader::ImageBufferDescriptor>);
static_assert(std::is_trivially_copyable_v<Shader::TextureDescriptor>);
static_assert(std::is_trivially_copyable_v<Shader::ImageDescriptor>);
static_assert(std::is_trivially_copyable_v<StageCache::Translation::Read>);

void PutVaryingState(Writer& w, const Shader::VaryingState& v) {
    for (size_t index = 0; index < v.mask.size(); index += 64) {
        u64 word{};
        for (size_t bit = 0; bit < 64; ++bit) {
            word |= u64{v.mask[index + bit]} << bit;
        }
        w.Put(word);
    }
}

bool GetVaryingState(Reader& r, Shader::VaryingState& v) {
    v.mask.reset();
    for (size_t index = 0; index < v.mask.size(); index += 64) {
        u64 word{};
        if (!r.Get(word)) {
            return false;
        }
        for (size_t bit = 0; bit < 64; ++bit) {
            if (((word >> bit) & 1U) != 0) {
                v.mask.set(index + bit);
            }
        }
    }
    return true;
}

void WriteSummary(Writer& w, const Shader::IR::Program& p) {
    w.Put(p.stage);
    w.Put(p.workgroup_size);
    w.Put(p.output_topology);
    w.Put(p.output_vertices);
    w.Put(p.invocations);
    w.Put(p.local_memory_size);
    w.Put(p.shared_memory_size);
    w.Put(p.is_geometry_passthrough);

    w.Put(p.mesh.payload_words);
    w.Put(p.mesh.max_vertices);
    w.Put(p.mesh.max_primitives);
    w.Put(p.mesh.topology);
    w.PutVec<u32>(p.mesh.vertex_lines);
    w.PutVec<u32>(p.mesh.primitive_lines);

    const Shader::Info& info = p.info;
    w.Put(info.is_task);
    w.Put(info.is_mesh);
    w.Put(info.uses_workgroup_id);
    w.Put(info.uses_local_invocation_id);
    w.Put(info.uses_invocation_id);
    w.Put(info.uses_invocation_info);
    w.Put(info.uses_sample_id);
    w.Put(info.uses_is_helper_invocation);
    w.Put(info.uses_subgroup_invocation_id);
    w.Put(info.uses_subgroup_shuffles);
    w.Put(info.uses_quad_shuffles);
    w.Put(info.uses_patches);
    w.Put(info.interpolation);
    PutVaryingState(w, info.loads);
    PutVaryingState(w, info.stores);
    PutVaryingState(w, info.passthrough);

    w.Put(static_cast<u32>(info.legacy_stores_mapping.size()));
    for (const auto& [from, to] : info.legacy_stores_mapping) {
        w.Put(from);
        w.Put(to);
    }

    w.Put(info.loads_indexed_attributes);
    w.Put(info.stores_frag_color);
    w.Put(info.frag_color_components);
    w.Put(info.stores_sample_mask);
    w.Put(info.stores_frag_depth);
    w.Put(info.stores_tess_level_outer);
    w.Put(info.stores_tess_level_inner);
    w.Put(info.stores_indexed_attributes);
    w.Put(info.stores_global_memory);
    w.Put(info.uses_local_memory);

    w.Put(info.uses_fp16);
    w.Put(info.uses_fp64);
    w.Put(info.uses_fp16_denorms_flush);
    w.Put(info.uses_fp16_denorms_preserve);
    w.Put(info.uses_fp32_denorms_flush);
    w.Put(info.uses_fp32_denorms_preserve);
    w.Put(info.uses_int8);
    w.Put(info.uses_int16);
    w.Put(info.uses_int64);
    w.Put(info.uses_image_1d);
    w.Put(info.uses_sampled_1d);
    w.Put(info.uses_sparse_residency);
    w.Put(info.uses_demote_to_helper_invocation);
    w.Put(info.uses_subgroup_vote);
    w.Put(info.demoted_conditional_barrier);
    w.Put(info.removed_conditional_barrier);
    w.Put(info.uses_subgroup_mask);
    w.Put(info.uses_fswzadd);
    w.Put(info.uses_derivatives);
    w.Put(info.uses_typeless_image_reads);
    w.Put(info.uses_typeless_image_writes);
    w.Put(info.uses_image_buffers);
    w.Put(info.uses_shared_increment);
    w.Put(info.uses_shared_decrement);
    w.Put(info.uses_global_increment);
    w.Put(info.uses_global_decrement);
    w.Put(info.uses_atomic_f32_add);
    w.Put(info.uses_atomic_f16x2_add);
    w.Put(info.uses_atomic_f16x2_min);
    w.Put(info.uses_atomic_f16x2_max);
    w.Put(info.uses_atomic_f32x2_add);
    w.Put(info.uses_atomic_f32x2_min);
    w.Put(info.uses_atomic_f32x2_max);
    w.Put(info.uses_atomic_s32_min);
    w.Put(info.uses_atomic_s32_max);
    w.Put(info.uses_int64_bit_atomics);
    w.Put(info.uses_global_memory);
    w.Put(info.requires_coherent_global_memory);
    w.Put(info.requires_coherent_image_memory);
    w.Put(info.uses_cooperative_matrix);
    w.Put(info.native_global_bounds);
    w.Put(info.uses_atomic_image_u32);
    w.Put(info.uses_shadow_lod);
    w.Put(info.uses_rescaling_uniform);
    w.Put(info.uses_cbuf_indirect);
    w.Put(info.uses_render_area);

    w.Put(info.used_constant_buffer_types);
    w.Put(info.used_storage_buffer_types);
    w.Put(info.used_indirect_cbuf_types);
    w.Put(info.constant_buffer_mask);
    w.Put(info.constant_buffer_used_sizes);
    w.Put(info.nvn_buffer_base);
    w.Put(static_cast<u16>(info.nvn_buffer_used.to_ulong()));
    w.Put(info.requires_layer_emulation);
    w.Put(info.emulated_layer);
    w.Put(info.used_clip_distances);

    w.PutVec<Shader::ConstantBufferDescriptor>(info.constant_buffer_descriptors);
    w.PutVec<Shader::StorageBufferDescriptor>(info.storage_buffers_descriptors);
    w.PutVec<Shader::TextureBufferDescriptor>(info.texture_buffer_descriptors);
    w.PutVec<Shader::ImageBufferDescriptor>(info.image_buffer_descriptors);
    w.PutVec<Shader::TextureDescriptor>(info.texture_descriptors);
    w.Put(static_cast<u32>(info.handle_programs.size()));
    for (const auto& hp : info.handle_programs) {
        w.PutVec<u32>(hp);
    }
    w.PutVec<Shader::ImageDescriptor>(info.image_descriptors);
}

bool ReadSummary(Reader& r, Shader::IR::Program& p) {
    if (!r.Get(p.stage) || !r.Get(p.workgroup_size) || !r.Get(p.output_topology) ||
        !r.Get(p.output_vertices) || !r.Get(p.invocations) || !r.Get(p.local_memory_size) ||
        !r.Get(p.shared_memory_size) || !r.Get(p.is_geometry_passthrough)) {
        return false;
    }
    if (!r.Get(p.mesh.payload_words) || !r.Get(p.mesh.max_vertices) ||
        !r.Get(p.mesh.max_primitives) || !r.Get(p.mesh.topology) ||
        !r.GetVec(p.mesh.vertex_lines, 136) || !r.GetVec(p.mesh.primitive_lines, 136)) {
        return false;
    }

    Shader::Info& info = p.info;
    if (!r.Get(info.is_task) || !r.Get(info.is_mesh) || !r.Get(info.uses_workgroup_id) ||
        !r.Get(info.uses_local_invocation_id) || !r.Get(info.uses_invocation_id) ||
        !r.Get(info.uses_invocation_info) || !r.Get(info.uses_sample_id) ||
        !r.Get(info.uses_is_helper_invocation) || !r.Get(info.uses_subgroup_invocation_id) ||
        !r.Get(info.uses_subgroup_shuffles) || !r.Get(info.uses_quad_shuffles) ||
        !r.Get(info.uses_patches) || !r.Get(info.interpolation) ||
        !GetVaryingState(r, info.loads) || !GetVaryingState(r, info.stores) ||
        !GetVaryingState(r, info.passthrough)) {
        return false;
    }

    u32 map_count{};
    if (!r.Get(map_count) || map_count > 512) {
        return false;
    }
    info.legacy_stores_mapping.clear();
    for (u32 i = 0; i < map_count; ++i) {
        Shader::IR::Attribute from{}, to{};
        if (!r.Get(from) || !r.Get(to)) {
            return false;
        }
        info.legacy_stores_mapping.emplace(from, to);
    }

    if (!r.Get(info.loads_indexed_attributes) || !r.Get(info.stores_frag_color) ||
        !r.Get(info.frag_color_components) || !r.Get(info.stores_sample_mask) ||
        !r.Get(info.stores_frag_depth) || !r.Get(info.stores_tess_level_outer) ||
        !r.Get(info.stores_tess_level_inner) || !r.Get(info.stores_indexed_attributes) ||
        !r.Get(info.stores_global_memory) || !r.Get(info.uses_local_memory)) {
        return false;
    }

    if (!r.Get(info.uses_fp16) || !r.Get(info.uses_fp64) ||
        !r.Get(info.uses_fp16_denorms_flush) || !r.Get(info.uses_fp16_denorms_preserve) ||
        !r.Get(info.uses_fp32_denorms_flush) || !r.Get(info.uses_fp32_denorms_preserve) ||
        !r.Get(info.uses_int8) || !r.Get(info.uses_int16) || !r.Get(info.uses_int64) ||
        !r.Get(info.uses_image_1d) || !r.Get(info.uses_sampled_1d) ||
        !r.Get(info.uses_sparse_residency) || !r.Get(info.uses_demote_to_helper_invocation) ||
        !r.Get(info.uses_subgroup_vote) || !r.Get(info.demoted_conditional_barrier) ||
        !r.Get(info.removed_conditional_barrier) || !r.Get(info.uses_subgroup_mask) ||
        !r.Get(info.uses_fswzadd) || !r.Get(info.uses_derivatives) ||
        !r.Get(info.uses_typeless_image_reads) || !r.Get(info.uses_typeless_image_writes) ||
        !r.Get(info.uses_image_buffers) || !r.Get(info.uses_shared_increment) ||
        !r.Get(info.uses_shared_decrement) || !r.Get(info.uses_global_increment) ||
        !r.Get(info.uses_global_decrement) || !r.Get(info.uses_atomic_f32_add) ||
        !r.Get(info.uses_atomic_f16x2_add) || !r.Get(info.uses_atomic_f16x2_min) ||
        !r.Get(info.uses_atomic_f16x2_max) || !r.Get(info.uses_atomic_f32x2_add) ||
        !r.Get(info.uses_atomic_f32x2_min) || !r.Get(info.uses_atomic_f32x2_max) ||
        !r.Get(info.uses_atomic_s32_min) || !r.Get(info.uses_atomic_s32_max) ||
        !r.Get(info.uses_int64_bit_atomics) || !r.Get(info.uses_global_memory) ||
        !r.Get(info.requires_coherent_global_memory) || !r.Get(info.requires_coherent_image_memory) ||
        !r.Get(info.uses_cooperative_matrix) || !r.Get(info.native_global_bounds) ||
        !r.Get(info.uses_atomic_image_u32) || !r.Get(info.uses_shadow_lod) ||
        !r.Get(info.uses_rescaling_uniform) || !r.Get(info.uses_cbuf_indirect) ||
        !r.Get(info.uses_render_area)) {
        return false;
    }

    u16 nvn_used{};
    if (!r.Get(info.used_constant_buffer_types) || !r.Get(info.used_storage_buffer_types) ||
        !r.Get(info.used_indirect_cbuf_types) || !r.Get(info.constant_buffer_mask) ||
        !r.Get(info.constant_buffer_used_sizes) || !r.Get(info.nvn_buffer_base) ||
        !r.Get(nvn_used) || !r.Get(info.requires_layer_emulation) ||
        !r.Get(info.emulated_layer) || !r.Get(info.used_clip_distances)) {
        return false;
    }
    info.nvn_buffer_used = std::bitset<16>{nvn_used};

    if (!r.GetVec(info.constant_buffer_descriptors, Shader::Info::MAX_CBUFS) ||
        !r.GetVec(info.storage_buffers_descriptors, Shader::Info::MAX_SSBOS) ||
        !r.GetVec(info.texture_buffer_descriptors, 256) ||
        !r.GetVec(info.image_buffer_descriptors, 256) ||
        !r.GetVec(info.texture_descriptors, 1024)) {
        return false;
    }
    u32 hp_count{};
    if (!r.Get(hp_count) || hp_count > 1024) {
        return false;
    }
    info.handle_programs.resize(hp_count);
    for (u32 i = 0; i < hp_count; ++i) {
        if (!r.GetVec(info.handle_programs[i], 65536)) {
            return false;
        }
    }
    return r.GetVec(info.image_descriptors, 256);
}

constexpr std::array<char, 8> StageCacheDiskMagic{'L', 'C', 'H', 'S', 'T', 'G', '0', '1'};
constexpr u32 StageCacheDiskVersion = 1;

struct StageCacheDiskHeader {
    std::array<char, 8> magic;
    u32 version;
    u32 entry_count;
    u64 variant;
    u64 uncompressed_bytes;
};
static_assert(sizeof(StageCacheDiskHeader) == 32);

bool RebaseSpirvBindings(std::vector<u32>& spirv, const Shader::Backend::Bindings& src_in,
                         const Shader::Backend::Bindings& req_in) {
    if (spirv.size() < 6 || spirv[0] != 0x07230203U) {
        return false;
    }
    constexpr u32 OpTypeVoid = 19;
    constexpr u32 OpFunction = 54;
    constexpr u32 OpDecorate = 71;
    constexpr u32 DecorationBinding = 33;
    constexpr u32 DecorationDescriptorSet = 34;

    size_t pos = 5;
    while (pos < spirv.size()) {
        const u32 word0 = spirv[pos];
        const u32 word_count = word0 >> 16;
        const u32 opcode = word0 & 0xffffU;
        if (word_count == 0 || pos + word_count > spirv.size()) {
            return false;
        }
        if (opcode == OpTypeVoid || opcode == OpFunction) {
            break;
        }
        if (opcode == OpDecorate && word_count == 4) {
            if (spirv[pos + 2] == DecorationBinding) {
                if (spirv[pos + 3] < src_in.unified) {
                    return false;
                }
                spirv[pos + 3] = spirv[pos + 3] - src_in.unified + req_in.unified;
            } else if (spirv[pos + 2] == DecorationDescriptorSet) {
                spirv[pos + 3] = req_in.descriptor_set;
            }
        }
        pos += word_count;
    }
    return true;
}

u64 EncodeReplacement(const std::optional<Shader::ReplaceConstant>& value) {
    return value ? (u64{1} << 32) | static_cast<u32>(*value) : 0;
}

struct PrecountedState {
    const StageCache* cache{};
    u32 count{};
};

thread_local PrecountedState tls_precounted{};

} // Anonymous namespace

StageCache::Recorder::Recorder(Shader::Environment& inner_) : inner{inner_} {
    sph = inner.SPH();
    gp_passthrough_mask = inner.GpPassthroughMask();
    mesh_registers = inner.MeshRegisters();
    mesh_primitive_header = inner.MeshPrimitiveHeader();
    stage = inner.ShaderStage();
    start_address = inner.StartAddress();
    is_proprietary_driver = inner.IsProprietaryDriver();
}

u64 StageCache::Recorder::Log(u8 kind, u32 a, u32 b, u32 c, u64 value) {
    // Only first reads: within one translation a repeated read gives the same answer.
    const u64 id = Common::CityHash64(reinterpret_cast<const char*>(std::array<u32, 4>{kind, a, b, c}.data()),
                                      sizeof(u32) * 4);
    if (logged.insert(id).second) {
        reads.push_back({kind, a, b, c, value});
    }
    return value;
}

u64 StageCache::Recorder::ReadInstruction(u32 address) {
    return inner.ReadInstruction(address);
}

u32 StageCache::Recorder::TranslateProgramAddress(u32 address) const {
    return inner.TranslateProgramAddress(address);
}

bool StageCache::Recorder::IsNativeProgram() const {
    return inner.IsNativeProgram();
}

bool StageCache::Recorder::IsCbufStorageBufferPointer(u32 index, u32 offset) {
    return Log(StoragePointer, index, offset, 0, inner.IsCbufStorageBufferPointer(index, offset)) != 0;
}

bool StageCache::Recorder::IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) {
    return Log(IndirectStoragePointer, index, offset, indirect,
               inner.IsCbufIndirectStorageBufferPointer(index, offset, indirect)) != 0;
}

std::vector<std::pair<u32, u32>> StageCache::Recorder::NullStoragePointerSlots() const {
    return inner.NullStoragePointerSlots();
}

std::vector<u64> StageCache::Recorder::MappedStoragePointerKeys() const {
    return inner.MappedStoragePointerKeys();
}

u32 StageCache::Recorder::ReadCbufValue(u32 index, u32 offset) {
    return static_cast<u32>(Log(CbufValue, index, offset, 0, inner.ReadCbufValue(index, offset)));
}

Shader::TextureType StageCache::Recorder::ReadTextureType(u32 handle) {
    return static_cast<Shader::TextureType>(
        Log(TextureType, handle, 0, 0, static_cast<u64>(inner.ReadTextureType(handle))));
}

Shader::TexturePixelFormat StageCache::Recorder::ReadTexturePixelFormat(u32 handle) {
    return static_cast<Shader::TexturePixelFormat>(
        Log(TexturePixelFormat, handle, 0, 0, static_cast<u64>(inner.ReadTexturePixelFormat(handle))));
}

bool StageCache::Recorder::IsTexturePixelFormatInteger(u32 handle) {
    return Log(TextureInteger, handle, 0, 0, inner.IsTexturePixelFormatInteger(handle)) != 0;
}

u32 StageCache::Recorder::ReadViewportTransformState() {
    return static_cast<u32>(Log(ViewportTransform, 0, 0, 0, inner.ReadViewportTransformState()));
}

u32 StageCache::Recorder::TextureBoundBuffer() const {
    return inner.TextureBoundBuffer();
}

u32 StageCache::Recorder::LocalMemorySize() const {
    return inner.LocalMemorySize();
}

u32 StageCache::Recorder::SharedMemorySize() const {
    return inner.SharedMemorySize();
}

std::array<u32, 3> StageCache::Recorder::WorkgroupSize() const {
    return inner.WorkgroupSize();
}

bool StageCache::Recorder::HasHLEMacroState() const {
    return inner.HasHLEMacroState();
}

bool StageCache::Recorder::HasNativeInstructions() const {
    return inner.HasNativeInstructions();
}

std::optional<Shader::ReplaceConstant> StageCache::Recorder::GetReplaceConstBuffer(u32 bank,
                                                                                   u32 offset) {
    const auto value = inner.GetReplaceConstBuffer(bank, offset);
    Log(Replacement, bank, offset, 0, EncodeReplacement(value));
    return value;
}

void StageCache::Recorder::Dump(u64 pipeline_hash, u64 shader_hash) {
    inner.Dump(pipeline_hash, shader_hash);
}

StageCache::StageCache() {
    // EDEN_NX2_STAGE_CACHE=0 translates every stage again; =verify also builds every reused stage
    // and compares the SPIR-V.
    const char* const value = std::getenv("EDEN_NX2_STAGE_CACHE");
    enabled = value == nullptr || value[0] != '0';
    verify = enabled && value != nullptr && std::string_view{value} == "verify";
    if (verify) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_STAGE_CACHE=verify: reused stages are rebuilt and compared");
    }
}

std::vector<u8> StageCache::Signature(Shader::Environment& env, u64 unique_hash, size_t slot) {
    Writer writer;
    writer.Put(unique_hash);
    writer.Put(static_cast<u32>(slot));
    writer.Put(env.ShaderStage());
    writer.Put(env.StartAddress());
    writer.Put(env.IsProprietaryDriver());
    writer.Put(env.IsNativeProgram());
    writer.Put(env.HasNativeInstructions());
    writer.Put(env.SPH());
    writer.Put(env.GpPassthroughMask());
    writer.Put(env.MeshRegisters());
    writer.Put(env.MeshPrimitiveHeader());
    writer.Put(env.TextureBoundBuffer());
    writer.Put(env.LocalMemorySize());
    writer.Put(env.SharedMemorySize());
    writer.Put(env.WorkgroupSize());
    writer.Put(env.HasHLEMacroState());
    // What the recompiler reads from the settings, which can change during a session.
    const auto& resolution = Settings::values.resolution_info;
    writer.Put(resolution.up_scale);
    writer.Put(resolution.down_shift);
    writer.Put(resolution.up_factor);
    writer.Put(resolution.down_factor);
    writer.Put(resolution.active);
    writer.Put(Settings::values.renderer_debug.GetValue());
    writer.Put(Settings::values.disable_shader_loop_safety_checks.GetValue());
    writer.Put(Settings::values.fix_bloom_effects.GetValue());
    return std::move(writer.bytes);
}

std::shared_ptr<const StageCache::Translation> StageCache::Find(const std::vector<u8>& signature,
                                                                Shader::Environment& env) {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(signature.data()),
                                        signature.size());
    std::vector<std::shared_ptr<const Translation>> candidates;
    {
        std::scoped_lock lock{mutex};
        const auto it = translations.find(hash);
        if (it == translations.end()) {
            return nullptr;
        }
        candidates = it->second;
    }
    for (const auto& candidate : candidates) {
        if (candidate->signature != signature) {
            continue;
        }
        bool same = true;
        try {
            // In the order translation read them: up to the first difference, every read is one
            // translating here would make too.
            for (const Translation::Read& read : candidate->reads) {
                u64 value{};
                switch (read.kind) {
                case CbufValue:
                    value = env.ReadCbufValue(read.a, read.b);
                    break;
                case StoragePointer:
                    value = env.IsCbufStorageBufferPointer(read.a, read.b);
                    break;
                case IndirectStoragePointer:
                    value = env.IsCbufIndirectStorageBufferPointer(read.a, read.b, read.c);
                    break;
                case TextureType:
                    value = static_cast<u64>(env.ReadTextureType(read.a));
                    break;
                case TexturePixelFormat:
                    value = static_cast<u64>(env.ReadTexturePixelFormat(read.a));
                    break;
                case TextureInteger:
                    value = env.IsTexturePixelFormatInteger(read.a);
                    break;
                case ViewportTransform:
                    value = env.ReadViewportTransformState();
                    break;
                case Replacement:
                    value = EncodeReplacement(env.GetReplaceConstBuffer(read.a, read.b));
                    break;
                }
                if (value != read.value) {
                    same = false;
                    break;
                }
            }
        } catch (const Shader::Exception&) {
            // A disk cache environment that never recorded this read.
            same = false;
        }
        if (same) {
            return candidate;
        }
    }
    return nullptr;
}

std::shared_ptr<const StageCache::Emitted> StageCache::FindEmitted(
    const Translation& translation, const std::vector<u8>& runtime_key, bool allow_spirv_rebase) {
    {
        std::scoped_lock lock{translation.emitted_mutex};
        for (const auto& emitted : translation.emitted) {
            if (emitted->runtime_key == runtime_key) {
                return emitted;
            }
        }
        if (!allow_spirv_rebase || runtime_key.size() <= sizeof(Shader::Backend::Bindings)) {
            return nullptr;
        }
        Shader::Backend::Bindings req_in{};
        std::memcpy(&req_in, runtime_key.data(), sizeof(req_in));
        const size_t tail_offset = sizeof(Shader::Backend::Bindings);
        for (const auto& emitted : translation.emitted) {
            if (emitted->runtime_key.size() != runtime_key.size() ||
                std::memcmp(emitted->runtime_key.data() + tail_offset,
                            runtime_key.data() + tail_offset,
                            runtime_key.size() - tail_offset) != 0) {
                continue;
            }
            Shader::Backend::Bindings src_in{};
            std::memcpy(&src_in, emitted->runtime_key.data(), sizeof(src_in));
            if (src_in.texture_scaling_index != req_in.texture_scaling_index ||
                src_in.image_scaling_index != req_in.image_scaling_index) {
                continue;
            }
            std::vector<u32> rebased_spirv = emitted->spirv;
            if (!RebaseSpirvBindings(rebased_spirv, src_in, req_in)) {
                continue;
            }
            auto rebased = std::make_shared<Emitted>();
            rebased->runtime_key = runtime_key;
            rebased->spirv = std::move(rebased_spirv);
            rebased->program = emitted->program;
            rebased->bindings_out = req_in;
            rebased->bindings_out.unified =
                req_in.unified + (emitted->bindings_out.unified - src_in.unified);
            rebased->bindings_out.texture_scaling_index =
                req_in.texture_scaling_index +
                (emitted->bindings_out.texture_scaling_index - src_in.texture_scaling_index);
            rebased->bindings_out.image_scaling_index =
                req_in.image_scaling_index +
                (emitted->bindings_out.image_scaling_index - src_in.image_scaling_index);
            return rebased;
        }
    }
    return nullptr;
}

void StageCache::Insert(std::shared_ptr<Translation> translation,
                        std::shared_ptr<const Emitted> emitted) {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(translation->signature.data()),
                                        translation->signature.size());
    const size_t size = translation->reads.size() * sizeof(Translation::Read) +
                        translation->signature.size() + sizeof(Translation) +
                        emitted->spirv.size() * sizeof(u32) + sizeof(Emitted);
    translation->emitted.push_back(std::move(emitted));
    {
        std::scoped_lock lock{mutex};
        translations[hash].push_back(std::move(translation));
    }
    dirty.store(true, std::memory_order_relaxed);
    Account(size);
    if (tls_precounted.cache == this && tls_precounted.count > 0) {
        --tls_precounted.count;
    } else {
        translated.fetch_add(1, std::memory_order_relaxed);
        Log();
    }
}

void StageCache::AddEmitted(const Translation& translation, std::shared_ptr<const Emitted> emitted) {
    const size_t size = emitted->spirv.size() * sizeof(u32) + sizeof(Emitted);
    {
        std::scoped_lock lock{translation.emitted_mutex};
        translation.emitted.push_back(std::move(emitted));
    }
    dirty.store(true, std::memory_order_relaxed);
    Account(size);
}

void StageCache::Account(size_t added) {
    std::scoped_lock lock{mutex};
    bytes += added;
    if (bytes > MaxBytes) {
        // Translations in use elsewhere stay alive through their shared pointers.
        translations.clear();
        bytes = 0;
    }
}

std::vector<u8> StageCache::RuntimeKey(const Shader::RuntimeInfo& info,
                                       const Shader::Backend::Bindings& bindings,
                                       bool support_descriptor_aliasing) {
    // Every field of RuntimeInfo: one left out would let different SPIR-V share a key.
    Writer writer;
    writer.Put(bindings);
    writer.Put(support_descriptor_aliasing);
    writer.Put(info.generic_input_types);
    for (size_t index = 0; index < info.previous_stage_stores.mask.size(); index += 64) {
        u64 word{};
        for (size_t bit = 0; bit < 64; ++bit) {
            word |= u64{info.previous_stage_stores.mask[index + bit]} << bit;
        }
        writer.Put(word);
    }
    writer.Put(static_cast<u32>(info.per_primitive_generics.to_ulong()));
    writer.Put(static_cast<u32>(info.previous_stage_legacy_stores_mapping.size()));
    for (const auto& [from, to] : info.previous_stage_legacy_stores_mapping) {
        writer.Put(from);
        writer.Put(to);
    }
    writer.Put(info.convert_depth_mode);
    writer.Put(info.force_early_z);
    writer.Put(info.tess_primitive);
    writer.Put(info.tess_spacing);
    writer.Put(info.tess_clockwise);
    writer.Put(info.input_topology);
    writer.Put(info.fixed_state_point_size.has_value());
    writer.Put(info.fixed_state_point_size.value_or(0.0f));
    writer.Put(info.alpha_test_func.has_value());
    writer.Put(info.alpha_test_func.value_or(Shader::CompareFunction{}));
    writer.Put(info.alpha_test_reference);
    writer.Put(info.y_negate);
    writer.Put(info.glasm_use_storage_buffers);
    writer.Put(info.xfb_count);
    for (u32 index = 0; index < info.xfb_count && index < info.xfb_varyings.size(); ++index) {
        const auto& varying = info.xfb_varyings[index];
        writer.Put(varying.buffer);
        writer.Put(varying.stream);
        writer.Put(varying.stride);
        writer.Put(varying.offset);
        writer.Put(varying.components);
    }
    writer.Put(info.color_output_types);
    writer.Put(info.dual_source_blend);
    return std::move(writer.bytes);
}

Shader::IR::Program StageCache::Summary(const Shader::IR::Program& program) {
    Shader::IR::Program summary;
    summary.info = program.info;
    summary.stage = program.stage;
    summary.workgroup_size = program.workgroup_size;
    summary.output_topology = program.output_topology;
    summary.output_vertices = program.output_vertices;
    summary.invocations = program.invocations;
    summary.local_memory_size = program.local_memory_size;
    summary.shared_memory_size = program.shared_memory_size;
    summary.is_geometry_passthrough = program.is_geometry_passthrough;
    summary.mesh = program.mesh;
    return summary;
}

bool StageCache::SameSummary(const Shader::IR::Program& a, const Shader::IR::Program& b) {
    return a.info == b.info && a.stage == b.stage && a.workgroup_size == b.workgroup_size &&
           a.output_topology == b.output_topology && a.output_vertices == b.output_vertices &&
           a.invocations == b.invocations && a.local_memory_size == b.local_memory_size &&
           a.shared_memory_size == b.shared_memory_size &&
           a.is_geometry_passthrough == b.is_geometry_passthrough && a.mesh == b.mesh;
}

void StageCache::CountReuse(bool emitted) {
    (emitted ? reused_emitted : reused_translation).fetch_add(1, std::memory_order_relaxed);
    Log();
}

void StageCache::CountTranslated() {
    if (tls_precounted.cache != this) {
        tls_precounted = {this, 1};
    } else {
        ++tls_precounted.count;
    }
    translated.fetch_add(1, std::memory_order_relaxed);
    Log();
}

void StageCache::CountVerifyMismatch(u64 unique_hash) {
    verify_mismatches.fetch_add(1, std::memory_order_relaxed);
    LOG_ERROR(Render_Vulkan, "Stage cache: reused stage {:016x} differs from a fresh build",
              unique_hash);
}

void StageCache::Log() {
    const s64 now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = last_log.load(std::memory_order_relaxed);
    if (now - last < 10 || !last_log.compare_exchange_strong(last, now)) {
        return;
    }
    if (last == 0) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Stage cache: {} stages reused whole, {} translations reused, {} translated{}",
             reused_emitted.load(), reused_translation.load(), translated.load(),
             verify ? fmt::format(", {} verify mismatches", verify_mismatches.load())
                    : std::string{});
}

bool StageCache::HasValidFile(const std::filesystem::path& path, u64 expected_variant) {
    std::error_code ec;
    if (path.empty() || !std::filesystem::exists(path, ec)) {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    StageCacheDiskHeader hdr{};
    if (!file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) {
        return false;
    }
    return hdr.magic == StageCacheDiskMagic && hdr.version == StageCacheDiskVersion &&
           hdr.variant == expected_variant && hdr.entry_count > 0;
}

bool StageCache::LoadFromDisk(const std::filesystem::path& path, u64 expected_variant) {
    if (!enabled || path.empty()) {
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const auto file_end = file.tellg();
    if (file_end < static_cast<std::streamoff>(sizeof(StageCacheDiskHeader))) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    StageCacheDiskHeader hdr{};
    if (!file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) {
        return false;
    }
    if (hdr.magic != StageCacheDiskMagic || hdr.version != StageCacheDiskVersion ||
        hdr.variant != expected_variant || hdr.uncompressed_bytes == 0 ||
        hdr.uncompressed_bytes > MaxBytes) {
        return false;
    }
    const size_t comp_size = static_cast<size_t>(file_end) - sizeof(StageCacheDiskHeader);
    std::vector<u8> compressed(comp_size);
    if (comp_size == 0 || !file.read(reinterpret_cast<char*>(compressed.data()),
                                     static_cast<std::streamsize>(comp_size))) {
        return false;
    }
    const std::vector<u8> raw = Common::Compression::DecompressDataZSTD(compressed);
    if (raw.size() != hdr.uncompressed_bytes) {
        return false;
    }
    Reader r{raw};
    std::unordered_map<u64, std::vector<std::shared_ptr<const Translation>>> loaded;
    loaded.reserve(hdr.entry_count);
    size_t loaded_bytes = 0;
    size_t loaded_emitted = 0;
    for (u32 i = 0; i < hdr.entry_count; ++i) {
        auto tr = std::make_shared<Translation>();
        if (!r.GetVec(tr->signature, 4096) || !r.GetVec(tr->reads, 65536) ||
            !ReadSummary(r, tr->program)) {
            return false;
        }
        u32 num_emitted{};
        if (!r.Get(num_emitted) || num_emitted > 256) {
            return false;
        }
        tr->emitted.reserve(num_emitted);
        size_t tr_bytes =
            tr->reads.size() * sizeof(Translation::Read) + tr->signature.size() + sizeof(Translation);
        for (u32 e = 0; e < num_emitted; ++e) {
            auto em = std::make_shared<Emitted>();
            if (!r.GetVec(em->runtime_key, 4096) || !r.GetVec(em->spirv, 250'000) ||
                !ReadSummary(r, em->program) || !r.Get(em->bindings_out)) {
                return false;
            }
            tr_bytes += em->spirv.size() * sizeof(u32) + sizeof(Emitted);
            tr->emitted.push_back(std::move(em));
            ++loaded_emitted;
        }
        const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(tr->signature.data()),
                                            tr->signature.size());
        loaded[hash].push_back(std::move(tr));
        loaded_bytes += tr_bytes;
    }
    if (!r.Done()) {
        return false;
    }
    {
        std::scoped_lock lock{mutex};
        translations = std::move(loaded);
        bytes = loaded_bytes;
        dirty.store(false, std::memory_order_relaxed);
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    LOG_INFO(Render_Vulkan,
             "Stage cache: loaded {} translations ({} SPIR-V modules, {} MiB) from {} in {:.1f} ms",
             hdr.entry_count, loaded_emitted, loaded_bytes >> 20,
             Common::FS::PathToUTF8String(path), ms);
    return true;
}

bool StageCache::SaveToDisk(const std::filesystem::path& path, u64 variant) {
    if (!enabled || path.empty() || !dirty.load(std::memory_order_relaxed)) {
        return false;
    }
    std::vector<std::shared_ptr<const Translation>> snapshot;
    {
        std::scoped_lock lock{mutex};
        if (translations.empty()) {
            return false;
        }
        for (const auto& [hash, list] : translations) {
            for (const auto& tr : list) {
                snapshot.push_back(tr);
            }
        }
    }
    Writer w;
    w.bytes.reserve(snapshot.size() * 2048);
    bool checked_roundtrip = false;
    u32 written_entries = 0;
    size_t written_emitted = 0;
    for (const auto& tr : snapshot) {
        std::vector<std::shared_ptr<const Emitted>> em_list;
        {
            std::scoped_lock lock{tr->emitted_mutex};
            em_list = tr->emitted;
        }
        if (em_list.empty()) {
            continue;
        }
        if (!checked_roundtrip) {
            Writer test_w;
            WriteSummary(test_w, tr->program);
            Reader test_r{test_w.bytes};
            Shader::IR::Program roundtrip;
            ASSERT(ReadSummary(test_r, roundtrip) && test_r.Done() &&
                   SameSummary(tr->program, roundtrip));
            checked_roundtrip = true;
        }
        w.PutVec<u8>(tr->signature);
        w.PutVec<Translation::Read>(tr->reads);
        WriteSummary(w, tr->program);
        w.Put(static_cast<u32>(em_list.size()));
        for (const auto& em : em_list) {
            w.PutVec<u8>(em->runtime_key);
            w.PutVec<u32>(em->spirv);
            WriteSummary(w, em->program);
            w.Put(em->bindings_out);
            ++written_emitted;
        }
        ++written_entries;
    }
    if (written_entries == 0) {
        return false;
    }
    const std::vector<u8> compressed =
        Common::Compression::CompressDataZSTD(w.bytes.data(), w.bytes.size(), 1);
    if (compressed.empty()) {
        return false;
    }
    const StageCacheDiskHeader hdr{
        .magic = StageCacheDiskMagic,
        .version = StageCacheDiskVersion,
        .entry_count = written_entries,
        .variant = variant,
        .uncompressed_bytes = w.bytes.size(),
    };
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr)) ||
            !out.write(reinterpret_cast<const char*>(compressed.data()),
                       static_cast<std::streamsize>(compressed.size())) ||
            !out.flush()) {
            out.close();
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    dirty.store(false, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan,
             "Stage cache: saved {} translations ({} SPIR-V modules, {} KiB compressed) to {}",
             written_entries, written_emitted, compressed.size() >> 10,
             Common::FS::PathToUTF8String(path));
    return true;
}

} // namespace VideoCommon
