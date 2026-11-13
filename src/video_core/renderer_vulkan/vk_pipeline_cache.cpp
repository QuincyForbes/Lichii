// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>
#include <bit>
#include <cstring>
#include <numeric>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include "common/cityhash.h"
#include "common/platform_events.h"
#include "common/scope_exit.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/memory_detect.h"
#include "common/shutdown_progress.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/ir_opt/oot_visual_workaround.h"
#include "shader_recompiler/program_header.h"
#include "video_core/async_translate.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/pipeline_cache_policy.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_cooperative_hmma_test.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_translate_queue.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GuessMissing;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = VideoCommon::VulkanPipelineCacheVersion;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

std::array<std::atomic<std::chrono::nanoseconds::rep>, 2> runtime_spirv_ns{};
/// Set while a translation thread builds (EDEN_NX2_ASYNC_TRANSLATE).
thread_local bool t_translate_worker{};

void AddRuntimeSpirvTime(size_t thread, std::chrono::steady_clock::time_point begin) {
    if (t_translate_worker) {
        // SPIR-V a translation thread emits itself is the workers' time, not the GPU thread's.
        thread = 1;
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    runtime_spirv_ns[thread].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order::relaxed);
}

/// Pipeline lookups skipped for unbuilt pipelines this session (BuiltPipeline), and for
/// pipelines still translating (EDEN_NX2_ASYNC_TRANSLATE).
std::atomic<u64> skipped_lookups{};

void NoteSkippedLookup() {
    if ((skipped_lookups.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
        // Counts lookups, not draws: a draw state can look its pipeline up more than once.
        LOG_INFO(Render_Vulkan, "Async shaders: {} pipeline lookups skipped for unbuilt pipelines",
                 skipped_lookups.load(std::memory_order_relaxed));
    }
}

/// EDEN_NX2_ASYNC_TRANSLATE: translations in flight at most (each holds its constant buffer
/// copies until translated), how long a draw that must not be skipped waits for one before
/// building its pipeline itself, unusable translations of a key before it is built on the GPU
/// thread for good, and the age at which a translation is taken for wedged.
constexpr size_t MaxTranslationsInFlight = 64;
constexpr std::chrono::milliseconds TranslateWaitCap{50};
constexpr u32 MaxTranslateFailures = 2;
constexpr std::chrono::seconds TranslateSweepAge{5};

/// EDEN_NX2_ASYNC_TRANSLATE_THREADS: translation threads (2 by default, 1 to 8).
size_t TranslateThreads() {
    const char* const value = std::getenv("EDEN_NX2_ASYNC_TRANSLATE_THREADS");
    const int threads = value != nullptr ? std::atoi(value) : 2;
    return static_cast<size_t>(std::clamp(threads, 1, 8));
}

vk::ShaderModule EmitShaderModule(const Device& device, const Shader::Profile& profile,
                                 const Shader::RuntimeInfo& runtime_info,
                                 Shader::IR::Program& program, Shader::Backend::Bindings& binding,
                                 u64 shader_hash, size_t stage_index,
                                 std::span<const u32> reference = {},
                                 u128* spirv_hash = nullptr) {
    const auto code =
        EmitPipelineShader(profile, runtime_info, program, binding, shader_hash, reference);
    if (spirv_hash) {
        // Graphics pipeline libraries are keyed by the SPIR-V the stage was emitted to.
        *spirv_hash = Common::CityHash128(reinterpret_cast<const char*>(code.data()),
                                          code.size() * sizeof(u32));
    }
    device.SaveShader(code);
    vk::ShaderModule module{BuildShader(device, code)};

    // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
    // is independent and gated only by gpu_log_shader_dumps.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry",
                                               "fragment"};
        const std::string shader_name =
            fmt::format("shader_{:016x}_{}", shader_hash, stage_names[stage_index]);
        if (should_log) {
            const std::string shader_info = fmt::format(
                "SPIR-V size: {} bytes, hash: {:016x}", code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                         std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const std::string name{fmt::format("Shader {:016x}", shader_hash)};
        module.SetObjectNameEXT(name.c_str());
    }
    return module;
}

// Graphics pipeline libraries (EDEN_NX2_GPL): each stage's resources go to its own descriptor
// set (set index = stage index) with bindings from 0, and the fragment stage's rescaling bits
// start at fixed indices, so a stage's SPIR-V does not depend on the other stages' resources.
void PrepareStageBindings(Shader::Backend::Bindings& binding, size_t stage_index, bool libraries) {
    if (!libraries) {
        return;
    }
    binding.unified = 0;
    binding.uniform_buffer = 0;
    binding.storage_buffer = 0;
    binding.texture = 0;
    binding.image = 0;
    binding.descriptor_set = static_cast<u32>(stage_index);
    if (stage_index + 1 == Maxwell::MaxShaderStage) {
        binding.texture_scaling_index = GPL_FRAGMENT_TEXTURE_SCALING_BASE;
        binding.image_scaling_index = GPL_FRAGMENT_IMAGE_SCALING_BASE;
    }
}

struct DeferredStages {
    std::unique_ptr<EmissionBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
};

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
#ifdef __AVX2__
    const auto* a = reinterpret_cast<const __m256i*>(this);
    const auto* b = reinterpret_cast<const __m256i*>(&rhs);
    const __m256i d0 = _mm256_xor_si256(_mm256_loadu_si256(a + 0), _mm256_loadu_si256(b + 0));
    const __m256i d1 = _mm256_xor_si256(_mm256_loadu_si256(a + 1), _mm256_loadu_si256(b + 1));
    const __m256i d2 = _mm256_xor_si256(_mm256_loadu_si256(a + 2), _mm256_loadu_si256(b + 2));
    const __m256i d3 = _mm256_xor_si256(_mm256_loadu_si256(a + 3), _mm256_loadu_si256(b + 3));
    __m256i diff = _mm256_or_si256(_mm256_or_si256(d0, d1), _mm256_or_si256(d2, d3));
    const size_t size = Size();
    if (size == 160) [[likely]] {
        diff = _mm256_or_si256(diff, _mm256_xor_si256(_mm256_loadu_si256(a + 4), _mm256_loadu_si256(b + 4)));
        return _mm256_testz_si256(diff, diff) != 0;
    }
    if (size == 128) {
        return _mm256_testz_si256(diff, diff) != 0;
    }
    if (_mm256_testz_si256(diff, diff) == 0) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + 128,
                       reinterpret_cast<const char*>(&rhs) + 128, size - 128) == 0;
#else
    if (unique_hashes != rhs.unique_hashes) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + sizeof(unique_hashes),
                       reinterpret_cast<const char*>(&rhs) + sizeof(unique_hashes),
                       state.Size()) == 0;
#endif
}

namespace {
// NVIDIA's and AMD's drivers keep a disk cache of built pipelines of their own. Eden's copy on top
// only kept every pipeline the title ever built in host memory (title 1: a 245 MB file, ~0.35 GB
// of RAM for the session) and copied it whole at each save, which stalled frames; pipelines built
// no slower without it.
bool DriverCachesPipelines(const Device& device) {
    switch (device.GetDriverID()) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return true;
    default:
        break;
    }
    // MoltenVK converts SPIR-V to MSL and compiles it to a MTLLibrary while it holds its pipeline
    // cache's one mutex, so every build that names the cache runs one at a time (build 62: 1,773
    // pipelines in ~175 s on six cores). Creating the cache from a file compiles every library in
    // it again on the calling thread before any pipeline is built, and the file keeps only MSL;
    // Metal's own shader cache keeps the compiled code. EDEN_NX2_MVK_PIPELINE_CACHE=1 uses it.
    if (device.IsMoltenVK()) {
        const char* const value = std::getenv("EDEN_NX2_MVK_PIPELINE_CACHE");
        return value == nullptr || value[0] == '0';
    }
    return false;
}
} // Anonymous namespace

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_},
      render_pass_cache{render_pass_cache_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue() &&
                                !DriverCachesPipelines(device_)},
      oot_visual_workaround{[] {
          const char* value = std::getenv("EDEN_NX2_OOT_VISUAL_WORKAROUND");
          return value != nullptr && std::string_view{value} == "1";
      }()},
      num_pipeline_workers{device.HasBrokenParallelShaderCompiling() ? 1ULL
                                                                   : GetTotalPipelineWorkers()},
      emission_budget{num_pipeline_workers},
      workers(num_pipeline_workers,
              "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {},
                           Common::ThreadPlacement::Background) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_float32_atomic_add = device.IsFloat32AtomicAddSupported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .max_compute_shared_memory_size = device.GetMaxComputeSharedMemorySize(),
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_spirv_fragment_shuffle = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };
    profile.support_cooperative_hmma =
        device.IsCooperativeHmmaSupported() && TestCooperativeHmma(device);
    profile.support_cooperative_hmma_16x16x16 =
        !profile.support_cooperative_hmma && device.IsCooperativeHmma16x16x16Supported();
    if (profile.support_cooperative_hmma_16x16x16) {
        LOG_INFO(Render_Vulkan, "Guest tensor operations run on 16x16x16 cooperative matrices");
    }

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = device.GetMaxPerStageDescriptorStorageBuffers(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_byte_addressed_r16 = true,
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));
    LOG_INFO(Render_Vulkan, "Float32 storage atomic adds: {}",
             profile.support_float32_atomic_add && profile.support_descriptor_aliasing
                 ? "native"
                 : "compare-exchange loop");

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();
    dynamic_features.has_static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();

    // A suspended app may be killed without notice (iOS): write the driver's cache out when it
    // leaves the foreground, and the used list at the next draw.
    background_listener = Common::PlatformEvents::AddBackgroundListener([this] {
        save_used_requested.store(true, std::memory_order_relaxed);
        if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
            return;
        }
        serialization_thread.QueueWork([this] {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        });
        serialization_thread.WaitForRequests();
    });
    // EDEN_NX2_ASYNC_TRANSLATE=1/2: new pipelines of draws that may be skipped are translated on
    // threads of their own. It relies on the asynchronous shaders' draw skipping.
    if (VideoCommon::AsyncTranslateWorkers()) {
        if (use_asynchronous_shaders) {
            const size_t threads = TranslateThreads();
            translate_queue = std::make_unique<TranslateQueue>(
                threads, MaxTranslationsInFlight,
                [this](TranslationJob& job) { TranslateOnWorker(job); },
                [this](std::unique_ptr<GraphicsPipeline> pipeline) {
                    RetireTranslatedPipeline(std::move(pipeline));
                });
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_ASYNC_TRANSLATE=1: new pipelines of draws that may be skipped are "
                     "translated on {} threads",
                     threads);
        } else {
            LOG_WARNING(Render_Vulkan, "EDEN_NX2_ASYNC_TRANSLATE needs asynchronous shaders; "
                                       "translating on the GPU thread");
        }
    } else if (VideoCommon::AsyncTranslateMode() == 3) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_TRANSLATE=3: GPU-thread translations use "
                                "environment snapshots, checked against the live state");
    }
}

namespace {
// <title>/vulkan_used.bin: the graphics pipelines recent sessions used, as key hash + sessions
// since the last use.
constexpr u32 UsedPipelinesMagic = 0x31505545; // "EUP1"
// Pipelines the last two sessions used are built at boot. Each holds ~0.45 MB of driver memory:
// title 1 built 4,800 from eight sessions, 1.5 GB more than building them on first use (but that
// stalled frames 4x as often), while one session of its title screen uses ~1,500.
// MoltenVK keeps the previous eight sessions: a pipeline built while playing goes through
// SPIR-V -> MSL -> Metal, far more than the 2-3 ms the shorter window assumes, so each one
// built at boot is a stutter avoided.
// Ocarina still built pipelines for 1-2 s of every 10 s early in play with two sessions kept:
// keep sixteen (#161).
constexpr u8 UsedPipelinesMaxAge = 2;
constexpr u8 UsedPipelinesMaxAgePerf = 16;
constexpr u8 UsedPipelinesMaxAgeMoltenVK = 8;

u8 MaxUsedPipelineAge(const Device& device) {
    if (device.IsMoltenVK()) {
        return UsedPipelinesMaxAgeMoltenVK;
    }
    return VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::PipelineRetention) ? UsedPipelinesMaxAgePerf : UsedPipelinesMaxAge;
}

std::unordered_map<u64, u8> LoadUsedPipelines(const std::filesystem::path& path, u8 max_age) {
    std::unordered_map<u64, u8> used;
    std::ifstream file{path, std::ios::binary};
    u32 magic{}, count{};
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != UsedPipelinesMagic ||
        !file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return used;
    }
    used.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 age{};
        if (!file.read(reinterpret_cast<char*>(&hash), sizeof(hash)) ||
            !file.read(reinterpret_cast<char*>(&age), sizeof(age))) {
            break;
        }
        if (age < max_age) { // lists saved with a longer window
            used.emplace(hash, age);
        }
    }
    return used;
}

/// Keeps only the vertex attribute types the programs of `envs` read (dynamic vertex input keys).
void MaskKeyedAttributeTypes(GraphicsPipelineCacheKey& key, std::span<const FileEnvironment> envs) {
    std::array<u64, Maxwell::MaxShaderProgram> attribute_masks;
    attribute_masks.fill(~u64{0});
    size_t env_index = 0;
    for (size_t stage = 0; stage < Maxwell::MaxShaderProgram; ++stage) {
        if (key.unique_hashes[stage] != 0 && env_index < envs.size()) {
            attribute_masks[stage] = VideoCommon::VertexAttributeMaskOf(envs[env_index++]);
        }
    }
    key.state.attribute_types = VideoCommon::KeyedAttributeTypes(
        key.state.attribute_types, key.unique_hashes, attribute_masks);
}

// EDEN_NX2_REKEY_FOREIGN_PIPELINES=0: cached graphics pipelines another GPU driver (or driver
// version) stored are skipped at boot instead of loaded under this driver (see load_graphics).
bool RekeyForeignPipelines() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_REKEY_FOREIGN_PIPELINES");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Reads one record as VideoCommon::LoadPipelines does and returns the key it is loaded under
/// (graphics keys masked as load_graphics masks them, compared over GraphicsPipelineCacheKey::Size
/// bytes), for merging another build's cache. Counts and sizes are checked against the bytes
/// left first, so a corrupt record cannot make an environment allocate more than the file holds.
std::optional<std::string> ReadPipelineRecordKey(std::ifstream& file, std::streamoff end) {
    u32 num_envs{};
    file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
    if (num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
        return std::nullopt;
    }
    std::vector<FileEnvironment> envs(num_envs);
    for (FileEnvironment& env : envs) {
        const auto begin = static_cast<std::streamoff>(file.tellg());
        std::array<u64, 6> counts{}; // code size, then the entry count of each table
        file.read(reinterpret_cast<char*>(counts.data()), sizeof(counts));
        file.seekg(begin);
        const u64 left = static_cast<u64>(end - begin);
        if (std::ranges::any_of(counts, [left](u64 count) { return count > left; })) {
            return std::nullopt;
        }
        env.Deserialize(file);
        if (static_cast<u32>(env.ShaderStage()) > static_cast<u32>(Shader::Stage::Mesh)) {
            return std::nullopt;
        }
    }
    const bool compute = envs.front().ShaderStage() == Shader::Stage::Compute;
    const auto compute_envs = std::ranges::count_if(envs, [](const FileEnvironment& env) {
        return env.ShaderStage() == Shader::Stage::Compute;
    });
    if (compute ? num_envs != 1 : compute_envs != 0) {
        return std::nullopt;
    }
    std::string bytes(1, compute ? 'c' : 'g');
    if (compute) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        bytes.append(reinterpret_cast<const char*>(&key), sizeof(key));
        return bytes;
    }
    GraphicsPipelineCacheKey key;
    file.read(reinterpret_cast<char*>(&key), sizeof(key));
    if (key.state.dynamic_vertex_input != 0) {
        MaskKeyedAttributeTypes(key, envs);
    }
    // Loading puts every record under this driver (see load_graphics): a pipeline cached by
    // another driver is the same pipeline.
    key.state.driver_id = 0;
    key.state.driver_version = 0;
    bytes.append(reinterpret_cast<const char*>(&key), (std::min)(key.Size(), sizeof(key)));
    return bytes;
}

/// Pipeline caches of zortos293/Lichii builds whose records this build reads, newest first.
/// From #194 on (cache version 53) they write the same records as this build under another
/// version number; version 54 (#408) only masks the vertex attribute types in its keys, as
/// loading does here. Community packs often keep the legacy names, so those are taken when
/// their header says 53 or 54.
void MergeCompatiblePipelineCaches(const std::filesystem::path& base_dir,
                                   const std::filesystem::path& own_cache,
                                   const std::filesystem::path& own_used) {
    constexpr std::array<char, 8> PipelineCacheMagic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
    constexpr std::array<u32, 2> CompatibleVersions{54, 53};
    struct Candidate {
        std::filesystem::path cache;
        std::filesystem::path used;
        std::optional<u32> version; // nullopt: legacy name, read from the header
    };
    std::vector<Candidate> candidates;
    for (const u32 version : CompatibleVersions) {
        candidates.push_back({VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan", version),
                              VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used",
                                                                      version),
                              version});
    }
    candidates.push_back({base_dir / "vulkan.bin", base_dir / "vulkan_used.bin", std::nullopt});
    candidates.push_back({base_dir / "vulkan_pipelines.bin", {}, std::nullopt});

    auto log_path = own_cache;
    log_path += ".merged";
    std::error_code error;
    // Merge everything again into a cache that was deleted.
    auto merge_log = std::filesystem::exists(own_cache, error)
                         ? VideoCommon::ReadPipelineMergeLog(log_path)
                         : VideoCommon::PipelineMergeLog{};
    bool log_changed = false;
    std::optional<VideoCommon::PipelineKeySet> own_keys;
    using MergeStatus = VideoCommon::PipelineCacheMergeResult::MergeStatus;
    for (const Candidate& candidate : candidates) {
        const auto stamp = VideoCommon::StampPipelineCacheFile(candidate.cache);
        if (!stamp) {
            continue;
        }
        const std::string name = Common::FS::PathToUTF8String(candidate.cache.filename());
        const std::string path = Common::FS::PathToUTF8String(candidate.cache);
        std::optional<u32> version = candidate.version;
        if (!version) {
            for (const u32 compatible : CompatibleVersions) {
                if (VideoCommon::PipelineCacheHeaderMatches(candidate.cache, PipelineCacheMagic,
                                                            compatible)) {
                    version = compatible;
                    break;
                }
            }
            if (!version) {
                continue; // another build's or a driver cache
            }
        }
        const auto merged = merge_log.find(name);
        bool cache_merged = merged != merge_log.end() && merged->second == *stamp;
        if (!cache_merged) {
            const auto result = VideoCommon::MergePipelineCache(
                candidate.cache, own_cache, PipelineCacheMagic, *version, CACHE_VERSION,
                ReadPipelineRecordKey, own_keys);
            switch (result.status) {
            case MergeStatus::Merged:
                LOG_INFO(Render_Vulkan,
                         "Merged {} new pipelines ({} records) from {} (cache version {}); {} "
                         "were already cached",
                         result.added_pipelines, result.added_records, path, *version,
                         result.known_pipelines);
                cache_merged = true;
                break;
            case MergeStatus::Missing:
                continue;
            case MergeStatus::WrongHeader:
                LOG_WARNING(Render_Vulkan,
                            "Skipped pipeline cache {}: it is not a cache version {} file", path,
                            *version);
                break;
            case MergeStatus::BadSource:
                LOG_WARNING(Render_Vulkan,
                            "Skipped pipeline cache {} (cache version {}): {}; nothing was "
                            "merged, it is tried again when it changes",
                            path, *version, result.detail);
                break;
            case MergeStatus::BadDestination:
                LOG_WARNING(Render_Vulkan, "Not merging pipeline cache {} into {}: {}", path,
                            Common::FS::PathToUTF8String(own_cache), result.detail);
                return;
            case MergeStatus::NoSpace:
            case MergeStatus::WriteFailed:
                LOG_WARNING(Render_Vulkan,
                            "Not merging pipeline cache {}: {}; the cache is unchanged", path,
                            result.detail);
                continue; // tried again next boot
            }
            // A source that does not parse is recorded too, so it is not scanned every boot.
            merge_log.insert_or_assign(name, *stamp);
            log_changed = true;
        }
        if (!cache_merged || candidate.used.empty()) {
            continue;
        }
        const auto used_stamp = VideoCommon::StampPipelineCacheFile(candidate.used);
        const std::string used_name = Common::FS::PathToUTF8String(candidate.used.filename());
        const auto used_merged = merge_log.find(used_name);
        if (!used_stamp || (used_merged != merge_log.end() && used_merged->second == *used_stamp)) {
            continue;
        }
        if (const auto added =
                VideoCommon::MergeUsedPipelineLists(candidate.used, own_used, UsedPipelinesMagic)) {
            LOG_INFO(Render_Vulkan, "Merged {} used pipelines from {}", *added,
                     Common::FS::PathToUTF8String(candidate.used));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Skipped used pipelines list {}: it or {} does not parse",
                        Common::FS::PathToUTF8String(candidate.used),
                        Common::FS::PathToUTF8String(own_used));
        }
        merge_log.insert_or_assign(used_name, *used_stamp);
        log_changed = true;
    }
    if (log_changed && !VideoCommon::WritePipelineMergeLog(log_path, merge_log)) {
        LOG_WARNING(Render_Vulkan, "Failed to write {}", Common::FS::PathToUTF8String(log_path));
    }
}

// Without a list of the pipelines recent sessions used (the first boot with this cache, or a session
// that ended before saving it), this many cached graphics pipelines at most are built at boot and the
// rest when first needed. A cache that grew over many builds held 48k pipelines, ~21 GB of driver
// memory built before the game started; the list is only saved at exit or after two minutes, so a
// boot that ran out of memory never saved one and the next boot did the same.
constexpr size_t MaxPipelinesWithoutUsedList = 3000;

// Captured environments the shader precache holds for skipped graphics pipelines; the rest wait
// for a later precache.
constexpr size_t PrecacheCaptureLimit = size_t{1} << 30;

// EDEN_NX2_BACKGROUND_PIPELINES=0: cached pipelines boot skips are built when first needed, on
// the GPU thread, as before. EDEN_NX2_BACKGROUND_PIPELINE_THREADS sets how many threads build
// them (default a quarter of the host's, 1 to 6).
bool BackgroundPipelinesEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_BACKGROUND_PIPELINES");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

size_t BackgroundPipelineThreads() {
    if (const char* value = std::getenv("EDEN_NX2_BACKGROUND_PIPELINE_THREADS")) {
        return std::clamp<size_t>(static_cast<size_t>(std::strtoull(value, nullptr, 10)), 1, 16);
    }
    return std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 1, 6);
}

/// Built graphics pipelines past which EvictUnusedPipelines destroys unused ones
/// (EDEN_NX2_PIPELINE_EVICT_ABOVE, default 8,000: ~3.6 GB of driver memory at most).
size_t EvictAbovePipelines() {
    static const size_t evict_above = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_EVICT_ABOVE");
        return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10))
                                : size_t{8000};
    }();
    return evict_above;
}

/// MoltenVK: pipelines are evicted on memory pressure only with less than this left to the
/// process (EDEN_NX2_PIPELINE_EVICT_LEFT_MIB; 0 or unset: on every pressure signal; the iOS
/// device profile sets it). An iPhone evicted 985 of them with 1.3 GiB left, ~60 draws were
/// skipped right after, and 1,041 evicted earlier gave back at most ~28 MiB of Metal memory.
u64 PressureEvictLeftBytes() {
    static const u64 bytes = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_EVICT_LEFT_MIB");
        return value != nullptr ? u64{std::strtoull(value, nullptr, 10)} << 20 : u64{0};
    }();
    return bytes;
}

/// With that limit set, the most evicted per pressure signal (EDEN_NX2_PIPELINE_EVICT_MAX).
size_t PressureEvictMax() {
    static const size_t max = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_EVICT_MAX");
        return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10))
                                : size_t{256};
    }();
    return max;
}

/// How many of the cached graphics pipelines boot skipped the background builds, newest first;
/// the rest are built when a draw first needs them. A cache merged from a community pack held
/// 60k pipelines: building all of them took 15 minutes on 6 threads and, at ~0.45 MB each, held
/// ~25 GB of driver memory in pipelines no draw had asked for (none is evicted before a draw
/// takes it), which shutting down then had to destroy one by one.
/// EDEN_NX2_BACKGROUND_PIPELINE_LIMIT sets it.
size_t BackgroundPipelineLimit(size_t built_at_boot) {
    if (const char* value = std::getenv("EDEN_NX2_BACKGROUND_PIPELINE_LIMIT")) {
        return static_cast<size_t>(std::strtoull(value, nullptr, 10));
    }
    // ~0.45 MB of driver memory each: at most a sixteenth of the RAM (16 GB: ~2,200), and no more
    // than the eviction threshold leaves past what boot built.
    constexpr u64 PipelineBytes = u64{460} << 10;
    const u64 ram = Common::GetMemInfo().TotalPhysicalMemory;
    const size_t by_ram = static_cast<size_t>(ram / 16 / PipelineBytes);
    const size_t evict_above = EvictAbovePipelines();
    const size_t by_eviction = evict_above > built_at_boot ? evict_above - built_at_boot : 0;
    constexpr size_t MinPipelines = 500;
    constexpr size_t MaxPipelines = 6000;
    return std::clamp<size_t>((std::min)(by_ram, by_eviction), MinPipelines, MaxPipelines);
}

// EXPERIMENT EDEN_NX2_EXP5 (live, read by each background build): background pipeline builds
// competed with the game's few busy threads (Ocarina of Time 49 instead of 60 fps for the 45-80 s
// they took). 0 = as before (below normal priority, any core). 1 = idle priority, so every game
// thread and the GPU thread's own pipeline builders run first, and off the cores pinned to the GPU
// threads (and off their L3 where the CPU has two), on efficiency cores on hybrid CPUs. 2..16 = as
// 1, with at most that many building at once.
int QuietPrebuildMode() {
    return VideoCore::Nx2Debug::Experiment(5);
}

void PlacePrebuildThread(bool quiet) {
    thread_local bool placed_quiet = false;
    thread_local u64 placed_mask = 0;
    // The GPU threads pin themselves when they start, which can be after the first builds.
    const u64 mask = quiet ? Common::BackgroundWorkCpuMask() : 0;
    if (quiet == placed_quiet && mask == placed_mask) {
        return;
    }
    if (quiet) {
        Common::SetCurrentThreadIdlePriority();
        Common::SetCurrentThreadToEfficiencyCores();
    } else {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        Common::SetCurrentThreadToBackgroundWork();
    }
    Common::SetCurrentThreadCpuMask(mask);
    placed_quiet = quiet;
    placed_mask = mask;
    static std::atomic<u64> logged{~u64{0}};
    const u64 state = quiet ? mask | (u64{1} << 63) : 0;
    if (logged.exchange(state, std::memory_order_relaxed) != state) {
        LOG_INFO(Render_Vulkan, "Background pipeline threads: {} priority, CPU mask {:#x}",
                 quiet ? "idle" : "below normal", mask);
    }
}

/// Apple: Metal compiles in MTLCompilerService at the QoS of the thread asking, so a pipeline the
/// boot waits for is built at user-initiated rather than the workers' utility class.
struct BootBuildQos {
#ifdef __APPLE__
    BootBuildQos() {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
    }
    ~BootBuildQos() {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    }
#endif
};

/// "pipeline=<key hash> s<stage>:<shader hash> ...", as the runtime pipeline lines print them
std::string PipelineShaders(const GraphicsPipelineCacheKey& key) {
    std::string shaders = fmt::format("pipeline={:016x}", key.Hash());
    for (size_t stage = 0; stage < key.unique_hashes.size(); ++stage) {
        if (key.unique_hashes[stage] != 0) {
            shaders += fmt::format(" s{}:{:016x}", stage, key.unique_hashes[stage]);
        }
    }
    return shaders;
}

/// A shader precache build running this long is logged with its pipeline. One past PrecacheGiveUp
/// is left to its thread and counted failed (EDEN_NX2_PRECACHE_TIMEOUT=<seconds>, default 90);
/// MoltenVK's Metal compile timeout (vulkan_library.cpp, 30 s) ends most stalls before that.
constexpr auto PrecacheSlowBuild = std::chrono::seconds{20};

std::chrono::seconds PrecacheGiveUp() {
    static const std::chrono::seconds seconds{[] {
        const char* const value = std::getenv("EDEN_NX2_PRECACHE_TIMEOUT");
        return value != nullptr ? (std::max)(1LL, std::atoll(value)) : 90LL;
    }()};
    return seconds;
}

/// Cached graphics pipelines boot skips that a Build Shaders launch builds as well, newest first:
/// as many as the background builders would take after the next boot
/// (EDEN_NX2_BUILD_SHADERS_LIMIT=<count> sets it).
size_t BuildShadersDeferredLimit(size_t built_at_boot) {
    if (const char* value = std::getenv("EDEN_NX2_BUILD_SHADERS_LIMIT")) {
        return static_cast<size_t>(std::strtoull(value, nullptr, 10));
    }
    return BackgroundPipelineLimit(built_at_boot);
}

// Preload every cached pipeline at boot across all worker threads unless EDEN_NX2_PRELOAD_ALL_PIPELINES=0.
// Off by default on MoltenVK, as upstream: each one goes through SPIR-V -> MSL -> Metal and none
// is ever evicted there, so Ocarina's ~23,000 took minutes and more memory than an iPhone has.
bool PreloadAllPipelines(const Device& device) {
    static const int all = [] {
        const char* value = std::getenv("EDEN_NX2_PRELOAD_ALL_PIPELINES");
        return value == nullptr ? -1 : (value[0] != '0' ? 1 : 0);
    }();
    return all < 0 ? !device.IsMoltenVK() : all != 0;
}

/// Native mesh programs write their outputs directly (RuntimeInfo::mesh_direct_outputs): on
/// MoltenVK with Settings mesh_direct_outputs (on by default), EDEN_NX2_MESH_DIRECT_OUTPUTS=0
/// keeps the workgroup attribute region, =1 uses direct outputs on every driver.
bool MeshDirectOutputs(const Device& device) {
    static const int direct = [] {
        const char* value = std::getenv("EDEN_NX2_MESH_DIRECT_OUTPUTS");
        return value == nullptr ? -1 : (value[0] != '0' ? 1 : 0);
    }();
    if (direct >= 0) {
        return direct != 0;
    }
    return device.IsMoltenVK() && Settings::values.mesh_direct_outputs.GetValue();
}

/// Destroys the pipelines of `pipelines` (a map or a deque of pairs holding unique_ptrs), telling
/// the frontend every few hundred that the teardown still moves: tens of thousands of them took
/// longer than the frontend waited for the emulation thread.
template <typename Container>
void DestroyPipelinesWithProgress(Container& pipelines) {
    constexpr size_t ProgressEvery = 256;
    size_t destroyed = 0;
    for (auto& entry : pipelines) {
        entry.second.reset();
        if (++destroyed % ProgressEvery == 0) {
            Common::NoteShutdownProgress();
        }
    }
    pipelines.clear();
    Common::NoteShutdownProgress();
}
} // Anonymous namespace

PipelineCache::~PipelineCache() {
    Common::PlatformEvents::RemoveBackgroundListener(background_listener);
    // First: the stop callback writes background_stop.
    background_stop_callback.reset();
    // Its threads build with the members below (a translation in progress finishes).
    translate_queue.reset();
    StopBackgroundPipelines();
    // A precache build left to its thread uses the cache: what returns soon is waited for.
    for (int wait = 0; *precache_threads_left > 0 && wait < 100; ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    if (const int left = *precache_threads_left; left > 0) {
        LOG_ERROR(Render_Vulkan, "{} shader precache threads are still building at shutdown", left);
    }
    Common::NoteShutdownProgress();
    workers.WaitForRequests();
    serialization_thread.WaitForRequests();
    Common::NoteShutdownProgress();
    // The caches are written before anything slow is torn down, so a teardown cut short still
    // leaves them on disk.
    SaveUsedPipelines();
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
    Common::NoteShutdownProgress();
    DestroyPipelinesWithProgress(background_ready);
    DestroyPipelinesWithProgress(graphics_cache);
    DestroyPipelinesWithProgress(evicted_pipelines);
    DestroyPipelinesWithProgress(retired_translations);
}

void PipelineCache::SaveUsedPipelines() {
    used_pipelines_saved = std::chrono::steady_clock::now();
    // A session too short to say what the game uses (a boot to the menu) keeps the old list.
    constexpr size_t MinimumUsed = 64;
    if (used_pipelines_filename.empty() ||
        used_this_session.size() + used_and_evicted.size() < MinimumUsed) {
        return;
    }
    std::unordered_map<u64, u8> next;
    next.reserve(used_pipelines.size() + used_this_session.size());
    for (const auto& [key, pipeline] : graphics_cache) {
        if (used_this_session.contains(pipeline.get())) {
            next.emplace(key.Hash(), u8{0});
        }
    }
    for (const u64 hash : used_and_evicted) {
        next.emplace(hash, u8{0});
    }
    for (const auto& [hash, age] : used_pipelines) {
        if (age + 1 < MaxUsedPipelineAge(device)) {
            next.try_emplace(hash, static_cast<u8>(age + 1));
        }
    }
    std::ofstream file{used_pipelines_filename, std::ios::binary | std::ios::trunc};
    const u32 count = static_cast<u32>(next.size());
    file.write(reinterpret_cast<const char*>(&UsedPipelinesMagic), sizeof(UsedPipelinesMagic));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [hash, age] : next) {
        file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        file.write(reinterpret_cast<const char*>(&age), sizeof(age));
    }
}

namespace {
// EDEN_NX2_PIPELINE_CLEAN=1 or EDEN_NX2_EXP19=1 (live; default off): see
// CurrentGraphicsPipelineClean. EXP19=2: with the patches of zortos293/Lichii#641
// (EDEN_NX2_PIPE_SKIP, EDEN_NX2_PIPE_PATCH).
int PipelineCleanMode() {
    static const bool env = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_CLEAN");
        return value != nullptr && value[0] != '0';
    }();
    const int experiment = VideoCore::Nx2Debug::Experiment(19);
    return experiment != 0 ? experiment : (env ? 1 : 0);
}
} // Anonymous namespace

void PipelineCache::BuildCleanWatch(bool patch) {
    using Range = CleanLookup::Range;
    const Maxwell& regs = maxwell3d->regs;
    const DynamicFeatures& features = dynamic_features;
    const auto watch_into = [&regs](std::vector<Range>& ranges, const auto& field) {
        const auto offset = static_cast<size_t>(reinterpret_cast<const char*>(&field) -
                                                reinterpret_cast<const char*>(&regs));
        const size_t first = offset / sizeof(u32);
        const size_t last = (offset + sizeof(field) + sizeof(u32) - 1) / sizeof(u32);
        ranges.push_back({static_cast<u32>(first), static_cast<u32>(last - first)});
    };
    std::vector<Range> ranges;
    const auto watch = [&](const auto& field) { watch_into(ranges, field); };

    // Every register FixedPipelineState::Refresh (and what it calls) reads with these features.
    // With `patch`, the blend registers are left out (the dual-source bit is compared instead,
    // and the attachments follow Dirty::Blending), and so are the vertex attribute formats with
    // dynamic vertex input (the types follow Dirty::VertexInput), as Refresh itself does.
    // Refresh itself, always:
    watch(regs.transform_feedback_enabled);
    watch(regs.depth_mode);
    watch(regs.polygon_mode_front);
    watch(regs.tessellation);
    watch(regs.patch_vertices);
    watch(regs.anti_alias_samples_mode);
    if (!patch) {
        watch(regs.blend);                    // ComputeAttachment0DualSourceBlend
        watch(regs.blend_per_target_enabled); // ComputeAttachment0DualSourceBlend
        watch(regs.blend_per_target[0]);      // ComputeAttachment0DualSourceBlend
    }
    watch(regs.alpha_test_enabled);
    watch(regs.alpha_test_func);
    watch(regs.alpha_test_ref);
    watch(regs.mandated_early_z);
    watch(regs.zeta_enable);
    watch(regs.zeta.format);
    watch(regs.window_origin);
    watch(regs.provoking_vertex);
    watch(regs.conservative_raster_enable);
    watch(regs.line_anti_alias_enable);
    watch(regs.anti_alias_alpha_control);
    watch(regs.depth_bounds);
    watch(regs.line_stipple_params);
    for (const auto& rt : regs.rt) {
        watch(rt.format);
    }
    watch(regs.point_size);
    // Dirty::VertexInput (both branches) and the static vertex repack:
    if (!patch || !features.has_dynamic_vertex_input) {
        watch(regs.vertex_attrib_format);
    }
    const bool repack = !features.has_dynamic_vertex_input && features.has_static_vertex_repack;
    if (!features.has_dynamic_vertex_input) {
        watch(regs.vertex_stream_instances);
        for (const auto& stream : regs.vertex_streams) {
            watch(stream.frequency);
        }
    }
    if (!features.has_extended_dynamic_state || repack) {
        // Strides (DynamicState::Refresh's vertex_strides, the repack): the first word of each
        // stream, VertexStream's leading union of stride and enable.
        for (const auto& stream : regs.vertex_streams) {
            const auto offset = static_cast<size_t>(reinterpret_cast<const char*>(&stream) -
                                                    reinterpret_cast<const char*>(&regs));
            ranges.push_back({static_cast<u32>(offset / sizeof(u32)), 1});
        }
    }
    // DynamicState::Refresh (no extended dynamic state):
    if (!features.has_extended_dynamic_state) {
        watch(regs.gl_front_face);
        watch(regs.stencil_front_op);
        watch(regs.stencil_two_side_enable);
        watch(regs.stencil_back_op);
        watch(regs.stencil_enable);
        watch(regs.depth_write_enabled);
        watch(regs.depth_bounds_enable);
        watch(regs.depth_test_enable);
        watch(regs.depth_test_func);
        watch(regs.gl_cull_face);
        watch(regs.gl_cull_test_enabled);
    }
    // DynamicState::Refresh2:
    if (!features.has_extended_dynamic_state_2_logic_op) {
        watch(regs.logic_op);
        if (!features.has_extended_dynamic_state_2) {
            watch(regs.polygon_offset_point_enable);
            watch(regs.polygon_offset_line_enable);
            watch(regs.polygon_offset_fill_enable);
            watch(regs.rasterize_enable);
            watch(regs.primitive_restart);
        }
    }
    // Dirty::Blending, BlendingAttachment::Refresh (no extended dynamic state 3 blending):
    if (!features.has_extended_dynamic_state_3_blend) {
        if (!patch) {
            watch(regs.color_mask_common);
            watch(regs.color_mask);
            watch(regs.blend_per_target);
        }
        if (Settings::values.use_squashed_iterated_blend) {
            watch(regs.iterated_blend.enable);
        }
    }
    // DynamicState::Refresh3:
    if (!features.has_dynamic_state3_logic_op_enable) {
        watch(regs.logic_op);
    }
    if (!features.has_dynamic_state3_depth_clamp_enable) {
        watch(regs.viewport_clip_control);
    }
    if (!features.has_dynamic_state3_line_stipple_enable) {
        watch(regs.line_stipple_enable);
    }

    // Sorted and merged: contiguous words compare as one block.
    const auto merge = [](std::vector<Range> input) {
        std::ranges::sort(input, {}, &Range::first);
        std::vector<Range> merged;
        for (const Range& range : input) {
            if (!merged.empty() && range.first <= merged.back().first + merged.back().count) {
                const u32 end = (std::max)(merged.back().first + merged.back().count,
                                           range.first + range.count);
                merged.back().count = end - merged.back().first;
            } else {
                merged.push_back(range);
            }
        }
        return merged;
    };
    clean.ranges = merge(std::move(ranges));
    // RefreshXfbState, while transform_feedback_enabled (watched above) is set:
    std::vector<Range> xfb_ranges;
    watch_into(xfb_ranges, regs.transform_feedback.controls);
    watch_into(xfb_ranges, regs.stream_out_layout);
    clean.xfb_ranges = merge(std::move(xfb_ranges));

    size_t words = 0;
    for (const Range& range : clean.ranges) {
        words += range.count;
    }
    clean.snapshot.assign(words, 0);
    size_t xfb_words = 0;
    for (const Range& range : clean.xfb_ranges) {
        xfb_words += range.count;
    }
    clean.xfb_snapshot.assign(xfb_words, 0);
    clean.built = true;
    clean.patch = patch;
    clean.pipeline = nullptr;
    LOG_INFO(Render_Vulkan,
             "EDEN_NX2_PIPELINE_CLEAN: watching {} register words in {} ranges{}", words,
             clean.ranges.size(), patch ? " (with patches)" : "");
}

bool PipelineCache::CleanInputsUnchanged() const {
    const u32* const words = maxwell3d->regs.reg_array.data();
    const u32* saved = clean.snapshot.data();
    const auto ranges_equal = [words](const std::vector<CleanLookup::Range>& ranges,
                                      const u32*& snap) {
        for (const CleanLookup::Range& range : ranges) {
            const u32* const reg = words + range.first;
            if (range.count == 1) {
                if (reg[0] != snap[0]) {
                    return false;
                }
            } else if (range.count == 2) {
                if ((reg[0] ^ snap[0]) | (reg[1] ^ snap[1])) {
                    return false;
                }
            } else if (std::memcmp(reg, snap, range.count * sizeof(u32)) != 0) {
                return false;
            }
            snap += range.count;
        }
        return true;
    };
    if (!ranges_equal(clean.ranges, saved)) {
        return false;
    }
    if (maxwell3d->regs.transform_feedback_enabled != 0) {
        saved = clean.xfb_snapshot.data();
        if (!ranges_equal(clean.xfb_ranges, saved)) {
            return false;
        }
    }
    return true;
}

void PipelineCache::TakeCleanSnapshot() {
    const u32* const words = maxwell3d->regs.reg_array.data();
    u32* saved = clean.snapshot.data();
    for (const CleanLookup::Range& range : clean.ranges) {
        std::memcpy(saved, words + range.first, range.count * sizeof(u32));
        saved += range.count;
    }
    if (maxwell3d->regs.transform_feedback_enabled != 0) {
        saved = clean.xfb_snapshot.data();
        for (const CleanLookup::Range& range : clean.xfb_ranges) {
            std::memcpy(saved, words + range.first, range.count * sizeof(u32));
            saved += range.count;
        }
    }
    clean.hashes = graphics_key.unique_hashes;
    clean.maxwell3d = maxwell3d;
    clean.topology = maxwell3d->draw_manager.draw_state.topology;
    clean.engine_state = maxwell3d->engine_state;
}

// The key is (unique_hashes, FixedPipelineState). Refresh's output depends only on: the
// registers in clean.ranges (and clean.xfb_ranges while transform feedback is on), the dirty
// flags VertexInput, ViewportSwizzles and Blending (it recomputes those groups only while set,
// and clears some), dynamic_features (set once at construction), the draw's topology and the
// engine hint. When all of these equal their values at the last Refresh and the flags are
// clear, Refresh would rewrite the same key and Next(graphics_key) would return
// current_pipeline (whose key it is): the lookup returns it as that fast path does.
// With `patch` (zortos293/Lichii#641): the dual-source blend bit is compared instead of the blend
// registers, and a raised VertexInput (dynamic vertex input) or Blending flag redoes only that
// part of the key, which Next then looks up.
GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineClean(bool patch) {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        clean.pipeline = nullptr;
        return nullptr;
    }
    if (!clean.built || clean.patch != patch) {
        BuildCleanWatch(patch);
    }
    const auto& flags = maxwell3d->dirty.flags;
    const auto draw_topology = maxwell3d->draw_manager.draw_state.topology;
    const bool blending_dirty =
        flags[Dirty::Blending] && (patch || !dynamic_features.has_extended_dynamic_state_3_blend);
    const bool vertex_input_dirty = flags[Dirty::VertexInput];
    const bool viewport_swizzles_dirty = flags[Dirty::ViewportSwizzles];
    const bool hashes_changed = clean.hashes != graphics_key.unique_hashes;
    const bool topology_changed = clean.topology != draw_topology;
    const bool engine_state_changed = clean.engine_state != maxwell3d->engine_state;
    const bool patch_blending = patch && blending_dirty;
    const bool patch_vertex_input =
        patch && vertex_input_dirty && dynamic_features.has_dynamic_vertex_input;
    const bool patch_viewport_swizzles = patch && viewport_swizzles_dirty;
    const bool patch_hashes = patch && hashes_changed;
    const bool patch_topology = patch && topology_changed;
    const bool patch_engine_state = patch && engine_state_changed;
    if (clean.pipeline != nullptr && clean.pipeline == current_pipeline &&
        clean.maxwell3d == maxwell3d && (!vertex_input_dirty || patch_vertex_input) &&
        (!viewport_swizzles_dirty || patch_viewport_swizzles) &&
        (!blending_dirty || patch_blending) && (!hashes_changed || patch_hashes) &&
        (!topology_changed || patch_topology) && (!engine_state_changed || patch_engine_state) &&
        CleanInputsUnchanged()) {
        ++clean.hits;
    } else {
        ++clean.misses;
        clean.pipeline = nullptr;
    }
    if (VideoCommon::Nx2Profile::Enabled() && ((clean.hits + clean.misses) & 0xfff) == 0) [[unlikely]] {
        const auto now = std::chrono::steady_clock::now();
        if (clean.last_log == std::chrono::steady_clock::time_point{}) {
            clean.last_log = now;
        } else if (now - clean.last_log >= std::chrono::seconds{10}) {
            const double seconds = std::chrono::duration<double>(now - clean.last_log).count();
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_PIPELINE_CLEAN per second: lookups without Refresh {} (patched "
                     "{}), with Refresh {}",
                     static_cast<u64>(static_cast<double>(clean.hits) / seconds),
                     static_cast<u64>(static_cast<double>(clean.patches) / seconds),
                     static_cast<u64>(static_cast<double>(clean.misses) / seconds));
            clean.last_log = now;
            clean.hits = 0;
            clean.patches = 0;
            clean.misses = 0;
        }
    }
    if (clean.pipeline != nullptr) {
        if (!patch_blending && !patch_vertex_input && !patch_viewport_swizzles && !patch_hashes &&
            !patch_topology && !patch_engine_state) {
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
        // Every other Refresh input is unchanged, so the snapshot stands. As RefreshGraphicsKey
        // under the raised flags; VertexInput stays raised for the command buffer update.
        ++clean.patches;
        bool key_changed = false;
        if (patch_hashes) {
            clean.hashes = graphics_key.unique_hashes;
            key_changed = true;
        }
        if (patch_vertex_input) {
            vertex_attribute_types = FixedPipelineState::DynamicAttributeTypes(maxwell3d->regs);
            const u64 new_types = VideoCommon::KeyedAttributeTypes(
                vertex_attribute_types, graphics_key.unique_hashes, vertex_attribute_masks);
            if (graphics_key.state.attribute_types != new_types) {
                graphics_key.state.attribute_types = new_types;
                key_changed = true;
            }
        } else if (patch_hashes && dynamic_features.has_dynamic_vertex_input) {
            graphics_key.state.attribute_types = VideoCommon::KeyedAttributeTypes(
                vertex_attribute_types, graphics_key.unique_hashes, vertex_attribute_masks);
        }
        if (patch_viewport_swizzles) {
            const auto old_swizzles = graphics_key.state.viewport_swizzles;
            graphics_key.state.RefreshViewportSwizzles(*maxwell3d);
            if (graphics_key.state.viewport_swizzles != old_swizzles) {
                key_changed = true;
            }
        }
        if (patch_topology) {
            clean.topology = draw_topology;
            const u32 old_raw1 = graphics_key.state.raw1;
            const u32 old_dyn_raw1 = graphics_key.state.dynamic_state.raw1;
            graphics_key.state.RefreshTopology(maxwell3d->regs, draw_topology, dynamic_features);
            if (graphics_key.state.raw1 != old_raw1 ||
                graphics_key.state.dynamic_state.raw1 != old_dyn_raw1) {
                key_changed = true;
            }
        }
        if (patch_engine_state) {
            clean.engine_state = maxwell3d->engine_state;
            const u32 old_raw2 = graphics_key.state.raw2;
            graphics_key.state.app_stage.Assign(maxwell3d->engine_state);
            if (graphics_key.state.raw2 != old_raw2) {
                key_changed = true;
            }
        }
        if (patch_blending) {
            const u32 old_raw1 = graphics_key.state.raw1;
            graphics_key.state.attachment0_dual_source_blend.Assign(
                FixedPipelineState::DualSourceBlend(maxwell3d->regs) ? 1 : 0);
            if (graphics_key.state.raw1 != old_raw1) {
                key_changed = true;
            }
            if (!dynamic_features.has_extended_dynamic_state_3_blend) {
                const auto old_attachments = graphics_key.state.attachments;
                const u32 old_raw2 = graphics_key.state.raw2;
                graphics_key.state.RefreshBlending(*maxwell3d);
                if (graphics_key.state.raw2 != old_raw2 ||
                    std::memcmp(graphics_key.state.attachments.data(), old_attachments.data(),
                                sizeof(old_attachments)) != 0) {
                    key_changed = true;
                }
            }
        }
        if (!key_changed) {
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
        clean.pipeline = nullptr;
    } else {
        TakeCleanSnapshot();
        RefreshGraphicsKey();
    }
    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            clean.pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    // The slow path makes graphics_cache[graphics_key] current, unless it has no pipeline.
    GraphicsPipeline* const previous = current_pipeline;
    GraphicsPipeline* const result = CurrentGraphicsPipelineSlowPath();
    if (current_pipeline != previous) {
        clean.pipeline = current_pipeline;
    }
    return result;
}

void PipelineCache::RefreshGraphicsKey() {
    if (!dynamic_features.has_dynamic_vertex_input) {
        graphics_key.state.Refresh(*maxwell3d, dynamic_features);
        return;
    }
    // Refresh only rewrites the types when the vertex input changes, so the masked copy in the
    // key must not carry over to the next programs. Types of attributes the programs don't read
    // would split their pipelines by whatever the game left in those registers (Ocarina of Time:
    // up to 20 keys for the same shaders, each built again on the GPU thread).
    graphics_key.state.attribute_types = vertex_attribute_types;
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    vertex_attribute_types = graphics_key.state.attribute_types;
    graphics_key.state.attribute_types = VideoCommon::KeyedAttributeTypes(
        vertex_attribute_types, graphics_key.unique_hashes, vertex_attribute_masks);
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    ++lookup_serial;
    if (const int clean_mode = PipelineCleanMode(); clean_mode != 0) {
        return CurrentGraphicsPipelineClean(clean_mode == 2);
    }
    // Switched off live: the next switch on starts from a full Refresh.
    clean.pipeline = nullptr;

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    RefreshGraphicsKey();

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* ComputePipelineCacheEntry::Get(bool native, GPUVAddr address,
                                                const ComputePipelineCacheKey& key) const {
    if (!pipeline && native) {
        throw Shader::RuntimeError(
            "Native compute pipeline unavailable: address={:#x} shader={:#016x} pipeline={:#016x}: {}",
            address, key.unique_hash, key.Hash(),
            failure.empty() ? "No compiler diagnostic was recorded" : failure);
    }
    return pipeline.get();
}

constexpr u32 MaxComputeRebuilds = 4;

namespace {
/// Which pointer slots a native program's arguments fill depends on the area (Ocarina of Time's
/// foliage culling fills different list slots in Hyrule Field and Castle Town), so a translation
/// rebuilt in one area may lack pointers the first cached one binds. Try first the translations
/// that bind every pointer the first one binds, those binding the most first and the newest of
/// those binding as many (an older one can be from a translator that asked about fewer slots, and
/// fail), then the others in cache order.
void OrderComputeTranslations(std::vector<FileEnvironment>& candidates) {
    if (candidates.size() < 2) {
        return;
    }
    std::vector<std::vector<u64>> keys;
    keys.reserve(candidates.size());
    for (const FileEnvironment& env : candidates) {
        keys.push_back(env.MappedStoragePointerKeys());
    }
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const auto keeps_first = [&](size_t index) {
        return std::ranges::includes(keys[index], keys.front());
    };
    std::ranges::stable_sort(order, [&](size_t lhs, size_t rhs) {
        const bool lhs_keeps{keeps_first(lhs)};
        if (lhs_keeps != keeps_first(rhs)) {
            return lhs_keeps;
        }
        if (!lhs_keeps) {
            return false;
        }
        if (keys[lhs].size() != keys[rhs].size()) {
            return keys[lhs].size() > keys[rhs].size();
        }
        return lhs > rhs;
    });
    std::vector<FileEnvironment> ordered;
    ordered.reserve(candidates.size());
    for (const size_t index : order) {
        ordered.push_back(std::move(candidates[index]));
    }
    candidates = std::move(ordered);
}
} // Anonymous namespace

bool PipelineCache::StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const {
    const auto& qmd{kepler_compute->launch_description};
    for (const auto& [index, offset] : entry.null_pointer_slots) {
        if (index >= qmd.const_buffer_config.size() ||
            ((qmd.const_buffer_enable_mask >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        if (offset + sizeof(u64) > cbuf.size) {
            continue; // The translator reads past the bound size as zero, too
        }
        const GPUVAddr slot{cbuf.Address() + offset};
        if (!gpu_memory->IsFullyMappedRange(slot, sizeof(u64))) {
            continue;
        }
        const GPUVAddr address{gpu_memory->Read<u64>(slot)};
        if (address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
            gpu_memory->IsFullyMappedRange(address, 1)) {
            LOG_INFO(Render_Vulkan,
                     "Rebuilding compute program {:#x}: c{}[{:#x}] was null at translation and "
                     "now points to {:#x}",
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, index, offset, address);
            return true;
        }
    }
    return false;
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& entry{pair->second};
    if (is_new) {
        entry = CreateComputePipeline(key, shader);
    } else {
        if (entry.pending_rebuild && entry.pending_rebuild->IsBuilt()) {
            if (!entry.pending_rebuild->BuildFailed()) {
                if (entry.pipeline) {
                    retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                           std::move(entry.pipeline));
                }
                entry.pipeline = std::move(entry.pending_rebuild);
                entry.null_pointer_slots = std::move(entry.pending_null_pointer_slots);
                entry.mapped_pointers = std::move(entry.pending_mapped_pointers);
                ++entry.rebuilds;
            } else {
                LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed",
                            key.unique_hash);
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(entry.pending_rebuild));
                entry.pending_null_pointer_slots.clear();
                entry.pending_mapped_pointers.clear();
                entry.rebuilds = MaxComputeRebuilds;
            }
        }
        if (!entry.pending_rebuild && !entry.null_pointer_slots.empty() &&
            entry.rebuilds < MaxComputeRebuilds && StaleNullPointerSlot(entry)) {
            // The translation dropped accesses through a pointer that was null back then. Which
            // slots the game fills depends on the area, so the new one keeps binding every pointer
            // this one binds, null now or not: a null pointer binds nothing. Built on a worker
            // while dispatches keep using the current pipeline (#969 / #995).
            auto rebuilt{CreateComputePipeline(key, shader, entry.mapped_pointers)};
            if (rebuilt.pipeline) {
                if (entry.pipeline && !entry.pipeline->BuildFailed()) {
                    if (rebuilt.pipeline->IsBuilt()) {
                        if (!rebuilt.pipeline->BuildFailed()) {
                            retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                                   std::move(entry.pipeline));
                            rebuilt.rebuilds = entry.rebuilds + 1;
                            entry = std::move(rebuilt);
                        } else {
                            LOG_WARNING(Render_Vulkan,
                                        "Keeping compute shader {:016x}: rebuild failed",
                                        key.unique_hash);
                            retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                                   std::move(rebuilt.pipeline));
                            entry.rebuilds = MaxComputeRebuilds;
                        }
                    } else {
                        entry.pending_rebuild = std::move(rebuilt.pipeline);
                        entry.pending_null_pointer_slots = std::move(rebuilt.null_pointer_slots);
                        entry.pending_mapped_pointers = std::move(rebuilt.mapped_pointers);
                    }
                } else {
                    if (entry.pipeline) {
                        retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                               std::move(entry.pipeline));
                    }
                    rebuilt.rebuilds = entry.rebuilds + 1;
                    entry = std::move(rebuilt);
                }
            } else {
                // Keep the working translation when the new one cannot be built (for example a
                // narrow access that now stays global across two buffers), and stop retrying.
                LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed: {}",
                            key.unique_hash, rebuilt.failure);
                entry.rebuilds = MaxComputeRebuilds;
            }
        }
    }
    return entry.Get(kepler_compute->IsNativeMode(),
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, key);
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    visual_workaround_title = title_id;
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        // The earlier load's background builds too: its pipelines no draw took are dropped
        // (never bound, so no GPU work uses them).
        background_stop_callback.reset();
        StopBackgroundPipelines();
        {
            std::scoped_lock lock{background_mutex};
            background_ready.clear();
            background_claimed.clear();
        }
        background_active.store(false, std::memory_order_release);
        background_stop.store(false, std::memory_order_relaxed);
        background_slots.store(0, std::memory_order_relaxed);
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
    }

    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    // This fork's files: its records carry task/mesh state other builds of the same cache version
    // cannot read, and their loader deletes a file it fails to read. The shared files of earlier
    // builds are copied once.
    pipeline_cache_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan-frizti", CACHE_VERSION);
    used_pipelines_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used-frizti", CACHE_VERSION);
    for (const auto& [shared, own] :
         {std::pair{VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan", CACHE_VERSION),
                    pipeline_cache_filename},
          std::pair{VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used", CACHE_VERSION),
                    used_pipelines_filename}}) {
        std::error_code error;
        if (!std::filesystem::exists(own, error) && std::filesystem::exists(shared, error)) {
            std::filesystem::copy_file(shared, own, error);
            LOG_INFO(Render_Vulkan, "Copied pipeline cache {} to {}{}",
                     Common::FS::PathToUTF8String(shared), Common::FS::PathToUTF8String(own),
                     error ? fmt::format(" (failed: {})", error.message()) : std::string{});
        }
    }
    // Pipelines of zortos293/Lichii caches this cache lacks are appended each time such a cache
    // changes, and their used lists merged, so they are built like this build's own.
    MergeCompatiblePipelineCaches(base_dir, pipeline_cache_filename, used_pipelines_filename);
    used_pipelines = LoadUsedPipelines(used_pipelines_filename, MaxUsedPipelineAge(device));
    used_pipelines_saved = std::chrono::steady_clock::now();
    size_t skipped_pipelines = 0;
    size_t queued_graphics = 0;
    using GraphicsRecords = std::vector<std::vector<FileEnvironment>>;
    std::vector<std::pair<GraphicsPipelineCacheKey, GraphicsRecords>> deferred_graphics;
    // Keys of graphics pipelines only another driver stored (see load_graphics).
    std::unordered_set<u64> rekeyed_pipelines;
    std::unordered_set<u64> native_pipelines;
    size_t foreign_records = 0;

    // With the shader precache this boot, the cached graphics pipelines skipped below are built
    // on the loading screen too, unless an earlier precache built them for this driver.
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    // A Build Shaders launch (the game does not start after the load) keeps no pipeline: those
    // boot would build, then the newest of the rest, go to the precache, built and dropped one by
    // one, with what an earlier precache built for this driver left out.
    const bool build_only = ShaderWarm::BuildOnlyRequested();
    const bool precache = ShaderWarm::PrecacheBeforeStart(
                              Settings::values.precache_shaders.GetValue(),
                              ShaderWarm::TakeWarmBeforeStart()) ||
                          build_only;
    std::unordered_set<u64> precached_pipelines;
    if (precache) {
        precached_pipelines = ShaderWarm::ReadPrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION});
    }
    std::vector<PrecacheJob> precache_graphics;
    size_t precache_capture_bytes = 0;
    size_t already_precached = 0;
    const auto precache_graphics_job = [&](const GraphicsPipelineCacheKey& key,
                                           GraphicsRecords& records) {
        const u64 hash = key.Hash();
        if (precached_pipelines.contains(hash)) {
            ++already_precached;
            return;
        }
        precache_graphics.push_back({
            .hash = hash,
            .record = true,
            .label = PipelineShaders(key),
            .build = [this, key_ = key, records_ = std::move(records)]() mutable {
                const auto pipeline = CreateCachedGraphicsPipeline(key_, records_, nullptr);
                return pipeline != nullptr && !pipeline->Failed();
            },
        });
    };

    if (use_vulkan_pipeline_cache) {
        // MoltenVK with mesh shaders (Settings moltenvk_driver) has its own file: each MoltenVK
        // rejects the other's (another git revision in the cache UUID) and would overwrite it.
        const bool mvk_mesh{device.IsMoltenVK() && device.IsExtMeshShaderSupported()};
        vulkan_pipeline_cache_filename = VideoCommon::VersionedPipelineCachePath(
            base_dir, mvk_mesh ? "vulkan_pipelines_mvk_mesh" : "vulkan_pipelines", CACHE_VERSION);
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        std::chrono::steady_clock::time_point last_save{std::chrono::steady_clock::now()};
    } state;
    // A rebuilt translation is cached beside the stale one. Choose between them before building, so
    // loading only ever adds entries (see OrderComputeTranslations), and take the next when a
    // translation fails.
    ::Common::unordered_map<ComputePipelineCacheKey, std::vector<FileEnvironment>> compute_envs;
    // The same for graphics: a pipeline whose record failed to load is stored again when the GPU
    // thread builds it, and its records are tried newest first.
    std::vector<std::pair<GraphicsPipelineCacheKey, GraphicsRecords>> graphics_records;
    ::Common::unordered_map<GraphicsPipelineCacheKey, size_t> graphics_record_index;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        compute_envs[key].push_back(std::move(env));
    }};
    const auto queue_compute{[&](const ComputePipelineCacheKey& key,
                                 std::vector<FileEnvironment> candidates) {
        if (build_only) {
            if (precached_pipelines.contains(key.Hash())) {
                ++already_precached;
                return;
            }
            precache_graphics.push_back({
                .hash = key.Hash(),
                .record = true,
                .label = fmt::format("compute pipeline={:016x} s5:{:016x}", key.Hash(),
                                     key.unique_hash),
                .build = [this, key, envs_ = std::move(candidates)]() mutable {
                    ShaderPools pools;
                    for (auto& env : envs_) {
                        pools.ReleaseContents();
                        if (CreateComputePipeline(pools, key, env, nullptr, false).pipeline) {
                            return true;
                        }
                    }
                    return false;
                },
            });
            return;
        }
        workers.QueueWork([this, key, envs_ = std::move(candidates), &state,
                           &callback]() mutable {
            [[maybe_unused]] BootBuildQos qos;
            ShaderPools pools;
            ComputePipelineCacheEntry entry;
            for (auto& env : envs_) {
                pools.ReleaseContents();
                entry = CreateComputePipeline(pools, key, env, state.statistics.get(), false);
                if (entry.pipeline) {
                    break;
                }
            }
            std::scoped_lock lock{state.mutex};
            if (entry.pipeline) {
                compute_cache.emplace(key, std::move(entry));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        // The key holds the driver that stored it, so a driver update (or a cache shared by
        // someone on another driver) left every record a key no draw makes: built at boot or in
        // the background all the same, never used, while the game translated each pipeline
        // again. Nothing builds from those two fields, so the record is the same pipeline under
        // this driver; its used-list age moves to that key.
        const bool foreign = key.state.driver_id != dynamic_features.driver_id ||
                             key.state.driver_version != dynamic_features.driver_version;
        if (foreign) {
            ++foreign_records;
            if (!RekeyForeignPipelines()) {
                return;
            }
            const u64 stored_hash = key.Hash();
            key.state.driver_id = dynamic_features.driver_id;
            key.state.driver_version = dynamic_features.driver_version;
            if (const auto used = used_pipelines.find(stored_hash); used != used_pipelines.end()) {
                const u8 age = used->second;
                const auto [entry, inserted] = used_pipelines.try_emplace(key.Hash(), age);
                entry->second = (std::min)(entry->second, age);
            }
        }

        // Keys stored before they kept only the attribute types the programs read (and imported
        // cache version 53 ones) are looked up masked. The pipeline is the same: a type is only
        // dropped for an attribute no program reads. Records that now share a key are tried
        // newest first, like those of a pipeline stored again.
        if (key.state.dynamic_vertex_input != 0) {
            const u64 stored_hash = key.Hash();
            MaskKeyedAttributeTypes(key, envs);
            if (const auto used = used_pipelines.find(stored_hash); used != used_pipelines.end()) {
                const u8 age = used->second;
                const auto [entry, inserted] = used_pipelines.try_emplace(key.Hash(), age);
                entry->second = (std::min)(entry->second, age);
            }
        }

        // Keys asking for a last-vertex mode the device lacks are loaded as they are: draws look
        // them up with the same bit, and the pipeline falls back to the first vertex when built.
        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        (foreign ? rekeyed_pipelines : native_pipelines).insert(key.Hash());
        const auto [it, inserted] = graphics_record_index.try_emplace(key, graphics_records.size());
        if (inserted) {
            graphics_records.emplace_back(key, GraphicsRecords{});
        }
        graphics_records[it->second].second.push_back(std::move(envs));
    }};
    const auto queue_graphics{[&](const GraphicsPipelineCacheKey& key, GraphicsRecords records) {
        const u64 hash = key.Hash();
        disk_pipelines.insert(hash);
        if (!PreloadAllPipelines(device) &&
            (used_pipelines.empty() ? queued_graphics >= MaxPipelinesWithoutUsedList
                                    : !used_pipelines.contains(hash))) {
            ++skipped_pipelines;
            if (BackgroundPipelinesEnabled() || precache) {
                deferred_graphics.emplace_back(key, std::move(records));
            }
            return;
        }
        ++queued_graphics;
        if (build_only) {
            precache_graphics_job(key, records);
            return;
        }

        workers.QueueWork([this, key, records_ = std::move(records), &state, &callback]() mutable {
            [[maybe_unused]] BootBuildQos qos;
            auto pipeline{CreateCachedGraphicsPipeline(key, records_, state.statistics.get())};

            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                graphics_cache.emplace(key, std::move(pipeline));
            }
            // A boot that builds tens of thousands of pipelines takes minutes: what it built so
            // far is written now and then, so a boot that ends early (a crash, the app killed)
            // does not leave the next one to compile everything again.
            if (VulkanPipelineCacheSaveDue(state.last_save)) {
                QueueVulkanPipelineCacheSave(false);
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto boot_build_begin = std::chrono::steady_clock::now();
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);
    for (auto& [key, records] : graphics_records) {
        std::ranges::reverse(records);
        queue_graphics(key, std::move(records));
    }
    graphics_records.clear();
    graphics_record_index.clear();
    for (auto& [key, candidates] : compute_envs) {
        OrderComputeTranslations(candidates);
        queue_compute(key, std::move(candidates));
    }
    compute_envs.clear();

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (PreloadAllPipelines(device) && queued_graphics > 10000) {
        LOG_WARNING(Render_Vulkan,
                    "EDEN_NX2_PRELOAD_ALL_PIPELINES: building and keeping all {} cached graphics "
                    "pipelines holds ~{} GB of driver memory all session",
                    queued_graphics, queued_graphics * 460 / (1024 * 1024));
    }
    if (foreign_records != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipeline records were stored by another GPU driver{}",
                 foreign_records,
                 RekeyForeignPipelines() ? ", loaded under this one" : " and are skipped");
    }
    if (skipped_pipelines != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipelines are built when first needed",
                 skipped_pipelines);
    }

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);
    LOG_INFO(Render_Vulkan,
             "Boot pipelines: {} built in {:.1f} s on {} threads ({} driver pipeline cache)",
             state.total,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - boot_build_begin)
                 .count(),
             num_pipeline_workers, use_vulkan_pipeline_cache ? "with the" : "without a");

    // Built newest first (from the end): this driver's own records before those only another
    // driver stored, which a merged pack appends after them.
    std::ranges::stable_partition(deferred_graphics, [&](const auto& deferred) {
        const u64 hash = deferred.first.Hash();
        return rekeyed_pipelines.contains(hash) && !native_pipelines.contains(hash);
    });
    if (build_only) {
        // The newest, as many as the background builders take after the next boot (an earlier
        // run built the oldest first, up to 1 GiB of captures: 490 of Ocarina's ~21,000, mostly
        // records other drivers stored). The rest are dropped now.
        const size_t boot_pipelines = precache_graphics.size() + already_precached;
        const size_t limit = BuildShadersDeferredLimit(queued_graphics);
        for (auto& [key, records] : std::views::reverse(deferred_graphics) | std::views::take(limit)) {
            precache_graphics_job(key, records);
        }
        LOG_INFO(Common_Memory,
                 "Build shaders: the {} pipelines boot builds and {} of the {} it leaves to later, "
                 "{} of them already built for this driver",
                 boot_pipelines, (std::min)(limit, deferred_graphics.size()),
                 deferred_graphics.size(), already_precached);
        decltype(deferred_graphics){}.swap(deferred_graphics);
    }
    const size_t background_count =
        BackgroundPipelinesEnabled() && !build_only
            ? (std::min)(deferred_graphics.size(), BackgroundPipelineLimit(queued_graphics))
            : 0;
    // The shader precache builds the rest, those otherwise built when a draw first needs them.
    for (size_t index = 0; precache && index < deferred_graphics.size() - background_count;
         ++index) {
        auto& [key, records] = deferred_graphics[index];
        size_t capture_bytes = 0;
        for (const auto& envs : records) {
            for (const auto& env : envs) {
                capture_bytes += env.MemorySize();
            }
        }
        if (precached_pipelines.contains(key.Hash())) {
            ++already_precached;
            continue;
        }
        if (capture_bytes > PrecacheCaptureLimit - precache_capture_bytes) {
            continue;
        }
        precache_capture_bytes += capture_bytes;
        precache_graphics_job(key, records);
    }
    if (background_count != 0 && !stop_loading.stop_requested()) {
        const size_t threads = BackgroundPipelineThreads();
        LOG_INFO(Render_Vulkan,
                 "Building {} of {} cached graphics pipelines in the background ({} threads), the "
                 "rest when first needed",
                 background_count, deferred_graphics.size(), threads);
        background_left.store(background_count, std::memory_order_relaxed);
        background_start = std::chrono::steady_clock::now();
        background_last_save = background_start;
        background_active.store(true, std::memory_order_release);
        background_workers.reset(new Common::ThreadWorker(threads, "VkPipelinePrebuild", {},
                                                          Common::ThreadPlacement::Background));
        // Stopping emulation stops them at once: they would otherwise compile on (minutes for a
        // large cache) until the cache is destroyed, competing with the teardown for the CPU.
        background_stop_callback = std::make_unique<std::stop_callback<std::function<void()>>>(
            stop_loading, std::function<void()>{[this] {
                background_stop.store(true, std::memory_order_relaxed);
            }});
        // Newest first: the cache file appends, so the end holds what the latest sessions
        // found, which the game is likelier to need soon than its oldest entries.
        for (auto& deferred :
             std::views::reverse(deferred_graphics) | std::views::take(background_count)) {
            background_workers->QueueWork([this, key = deferred.first,
                                           records_ = std::move(deferred.second)]() mutable {
                thread_local const size_t slot =
                    background_slots.fetch_add(1, std::memory_order_relaxed);
                for (;;) {
                    const int mode = QuietPrebuildMode();
                    PlacePrebuildThread(mode != 0);
                    if (mode < 2 || slot < static_cast<size_t>(mode) ||
                        background_stop.load(std::memory_order_relaxed)) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                }
                bool build = !background_stop.load(std::memory_order_relaxed);
                if (build) {
                    std::scoped_lock claimed_lock{background_mutex};
                    build = !background_claimed.contains(key);
                }
                std::unique_ptr<GraphicsPipeline> pipeline;
                if (build) {
                    pipeline = CreateCachedGraphicsPipeline(key, records_, nullptr);
                }
                std::unique_lock ready_lock{background_mutex};
                if (pipeline && !background_claimed.contains(key)) {
                    background_ready.emplace(key, std::move(pipeline));
                }
                // These builds fill the driver cache too, but only GPU thread builds queued its
                // writes: a session stopped before its exit save kept none of them.
                if (build && VulkanPipelineCacheSaveDue(background_last_save)) {
                    QueueVulkanPipelineCacheSave(false);
                }
                if (background_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    const auto seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - background_start).count();
                    LOG_INFO(Render_Vulkan,
                             "Background pipelines done in {:.1f} s: {} ready, {} taken by draws, "
                             "{} built by the GPU thread first",
                             seconds, background_ready.size() + background_taken, background_taken,
                             background_claimed.size());
                    background_claimed.clear();
                    background_active.store(false, std::memory_order_release);
                    if (!background_stop.load(std::memory_order_relaxed)) {
                        QueueVulkanPipelineCacheSave(true);
                    }
                }
                ready_lock.unlock();
                // A pipeline dropped here was never bound, so no GPU work can still use it.
                pipeline.reset();
            });
        }
    }

    if (use_vulkan_pipeline_cache) {
        // On the serialization thread: for a cache of tens of thousands of pipelines, reading
        // the driver data (gigabytes) and writing it can hold the boot at 100% long enough to
        // look frozen. It follows any write the boot build queued.
        QueueVulkanPipelineCacheSave(true);
        last_flush = std::chrono::steady_clock::now();
    }

    if (state.statistics) {
        state.statistics->Report();
    }
    if (precache && !stop_loading.stop_requested()) {
        if (PrecacheShaders(title_id, std::move(precache_graphics), std::move(precached_pipelines),
                            already_precached, stop_loading, callback)) {
            PrecacheGraphicsLibraries(title_id, stop_loading, callback);
        }
    }
}

bool PipelineCache::PrecacheShaders(u64 title_id, std::vector<PrecacheJob> jobs,
                                    std::unordered_set<u64> precached, size_t already_built,
                                    std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const size_t graphics_jobs = jobs.size();
    const std::filesystem::path warm_file = ShaderWarm::ComputeWarmPath(title_id);
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    const auto stamp = [&](u64 entries) {
        return ShaderWarm::CurrentWarmStamp(warm_file, entries, device.GetDriverVersion(),
                                            device.GetDeviceID(), CACHE_VERSION);
    };
    // Compute programs of the precache file, unless a matching marker says they were built.
    size_t file_entries = 0;
    bool compute_pending = false;
    std::error_code ec;
    const auto current = stamp(0);
    const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
    if (!std::filesystem::exists(warm_file, ec)) {
        LOG_INFO(Render_Vulkan, "Shader precache: no precache file at {}", path);
    } else if (current && marker && marker->Matches(*current)) {
        LOG_INFO(Render_Vulkan, "Shader precache: compute programs already done for this driver");
    } else {
        // Read the whole file here (its own reader: the disk cache loader deletes files it rejects).
        try {
            std::ifstream file(warm_file, std::ios::binary | std::ios::ate);
            file.exceptions(std::ifstream::failbit);
            const auto end{file.tellg()};
            file.seekg(0, std::ios::beg);
            std::array<char, 8> magic;
            u32 version{};
            file.read(magic.data(), magic.size())
                .read(reinterpret_cast<char*>(&version), sizeof(version));
            constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
            if (magic != expected_magic || version != CACHE_VERSION) {
                LOG_WARNING(Render_Vulkan, "Shader precache: {} is not a version {} file", path,
                            CACHE_VERSION);
                throw std::runtime_error("wrong version");
            }
            while (file.tellg() != end) {
                u32 num_envs{};
                file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                FileEnvironment env;
                env.Deserialize(file);
                if (num_envs != 1 || env.ShaderStage() != Shader::Stage::Compute) {
                    LOG_WARNING(Render_Vulkan, "Shader precache: unexpected entry in {}", path);
                    throw std::runtime_error("unexpected entry");
                }
                ComputePipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                ++file_entries;
                // Programs already in the disk cache were built above; leave them out.
                if (compute_cache.contains(key) || precached.contains(key.Hash())) {
                    continue;
                }
                jobs.push_back({
                    .label = fmt::format("precache file compute s5:{:016x}", key.unique_hash),
                    .build = [this, key, env = std::move(env)]() mutable {
                        static const GuessMissing guess{};
                        env.SetGuessMissing(&guess);
                        ShaderPools pools;
                        return CreateComputePipeline(pools, key, env, nullptr, false).pipeline !=
                               nullptr;
                    },
                });
            }
            compute_pending = true;
        } catch (const std::exception& e) {
            LOG_WARNING(Render_Vulkan, "Shader precache: reading {} failed: {}", path, e.what());
            jobs.resize(graphics_jobs);
            file_entries = 0;
        }
    }
    const bool build_only = ShaderWarm::BuildOnlyRequested();
    const char* const what = build_only ? "Build shaders" : "Shader precache";
    const size_t num_threads = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 64);
    if (jobs.empty() && !compute_pending) {
        LOG_INFO(Render_Vulkan,
                 "{}: 0 built, {} skipped (already built), 0 failed in 0.0 s on {} threads", what,
                 already_built, num_threads);
        ShaderWarm::SetBuildStatus({.skipped = already_built, .threads = num_threads});
        if (callback) {
            callback(VideoCore::LoadCallbackStage::Warm, 0, 0);
        }
        return true;
    }

    // Shared with the threads: one left to a build that never returns keeps it alive.
    struct Run {
        enum : u8 { Waiting, Building, Built, Failed, GivenUp };
        std::vector<PrecacheJob> jobs;
        std::unique_ptr<std::atomic<u8>[]> state;
        std::atomic<size_t> next{};
        std::atomic<size_t> done{};
        std::atomic<size_t> built{};
        std::atomic<size_t> failed{};
        std::atomic<bool> cancel{};
    };
    // One per thread: the job it builds and since when. Running -> Exited by the thread, or
    // Running -> Left when its build is given up (the thread is detached then).
    struct Slot {
        enum : u8 { Running, Exited, Left };
        std::atomic<u8> state{Running};
        std::atomic<size_t> job{SIZE_MAX};
        std::atomic<s64> since{};
        bool reported{};
    };
    auto run = std::make_shared<Run>();
    run->jobs = std::move(jobs);
    const size_t total = run->jobs.size();
    run->state = std::make_unique<std::atomic<u8>[]>(total);
    const auto now_ns = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };

    const auto begin = std::chrono::steady_clock::now();
    bool skipped = false;
    size_t given_up = 0;
    LOG_INFO(Render_Vulkan,
             "{}: {} cached pipelines ({} skipped, already built) and {} of {} precache file "
             "compute programs to build on {} threads",
             what, graphics_jobs, already_built, total - graphics_jobs, file_entries, num_threads);
    std::vector<std::pair<std::thread, std::shared_ptr<Slot>>> threads;
    const auto spawn = [&] {
        auto slot = std::make_shared<Slot>();
        threads.emplace_back(
            std::thread{[run, slot, left = precache_threads_left, now_ns] {
                Common::SetCurrentThreadName("VkShaderPrecache");
                // Someone waits for these (a loading screen or Build Shaders).
                [[maybe_unused]] BootBuildQos qos;
                while (!run->cancel.load(std::memory_order_relaxed)) {
                    const size_t index = run->next.fetch_add(1);
                    if (index >= run->jobs.size()) {
                        break;
                    }
                    run->state[index].store(Run::Building, std::memory_order_relaxed);
                    slot->since.store(now_ns(), std::memory_order_relaxed);
                    slot->job.store(index, std::memory_order_release);
                    // EDEN_NX2_PRECACHE_STALL=<n>: every n-th build hangs for ten minutes first
                    // (a driver compile that never returns, to test the give-up path).
                    static const size_t stall = [] {
                        const char* const value = std::getenv("EDEN_NX2_PRECACHE_STALL");
                        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
                    }();
                    if (stall != 0 && index % stall == 0) {
                        std::this_thread::sleep_for(std::chrono::minutes{10});
                    }
                    bool ok = false;
                    try {
                        // Built and destroyed here: only the driver's cache keeps it.
                        ok = run->jobs[index].build();
                    } catch (const std::exception& e) {
                        LOG_WARNING(Render_Vulkan, "Shader precache: {}: {}",
                                    run->jobs[index].label, e.what());
                    }
                    // Its captured records go now: memory falls as the run goes on.
                    run->jobs[index].build = {};
                    slot->job.store(SIZE_MAX, std::memory_order_release);
                    u8 building = Run::Building;
                    if (run->state[index].compare_exchange_strong(building,
                                                                  ok ? Run::Built : Run::Failed)) {
                        (ok ? run->built : run->failed).fetch_add(1);
                        run->done.fetch_add(1);
                    }
                    if (slot->state.load() == Slot::Left) {
                        break;
                    }
                }
                u8 running = Slot::Running;
                if (!slot->state.compare_exchange_strong(running, Slot::Exited)) {
                    left->fetch_sub(1);
                }
            }},
            slot);
    };
    for (size_t thread = 0; thread < (std::min)(num_threads, total); ++thread) {
        spawn();
    }
    const auto record_built = [&] {
        bool added = false;
        for (size_t index = 0; index < total; ++index) {
            const PrecacheJob& job = run->jobs[index];
            if (job.record && run->state[index].load(std::memory_order_relaxed) == Run::Built) {
                added |= precached.insert(job.hash).second;
            }
        }
        if (added) {
            ShaderWarm::WritePrecachedPipelines(
                ShaderWarm::PrecachedPipelinesPath(title_id),
                {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION}, precached);
        }
    };
    const auto give_up = [&](Slot& slot, size_t index, s64 seconds) {
        u8 running = Slot::Running;
        precache_threads_left->fetch_add(1);
        if (!slot.state.compare_exchange_strong(running, Slot::Left)) {
            precache_threads_left->fetch_sub(1);
            return false;
        }
        u8 building = Run::Building;
        if (run->state[index].compare_exchange_strong(building, Run::GivenUp)) {
            run->failed.fetch_add(1);
            run->done.fetch_add(1);
        }
        ++given_up;
        LOG_ERROR(Render_Vulkan, "{}: gave up on {} after {} s, its thread is left to it", what,
                  run->jobs[index].label, seconds);
        return true;
    };
    const auto status = [&](bool running) {
        s64 slowest = 0;
        const s64 now = now_ns();
        for (const auto& [thread, slot] : threads) {
            if (slot->state.load() == Slot::Running && slot->job.load() != SIZE_MAX) {
                slowest = (std::max)(slowest, now - slot->since.load(std::memory_order_relaxed));
            }
        }
        ShaderWarm::SetBuildStatus({
            .total = total,
            .done = run->done.load(),
            .built = run->built.load(),
            .failed = run->failed.load(),
            .skipped = already_built,
            .threads = num_threads,
            .elapsed_ms = static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - begin)
                                               .count()),
            .slowest_ms = static_cast<u64>(slowest / 1'000'000),
            .running = running,
        });
        if (callback) {
            callback(VideoCore::LoadCallbackStage::Warm, run->done.load(), total);
        }
    };
    auto last_record = std::chrono::steady_clock::now();
    for (;;) {
        status(true);
        if (stop_loading.stop_requested() || ShaderWarm::TakeSkipWarm()) {
            skipped = true;
            break;
        }
        const s64 now = now_ns();
        size_t active = 0;
        for (size_t thread = 0; thread < threads.size(); ++thread) {
            Slot& slot = *threads[thread].second;
            if (slot.state.load() != Slot::Running) {
                continue;
            }
            ++active;
            const size_t index = slot.job.load(std::memory_order_acquire);
            if (index == SIZE_MAX) {
                continue;
            }
            const auto busy = std::chrono::nanoseconds{now - slot.since.load()};
            const s64 seconds = std::chrono::duration_cast<std::chrono::seconds>(busy).count();
            if (busy > PrecacheSlowBuild && !slot.reported) {
                slot.reported = true;
                LOG_WARNING(Render_Vulkan, "{}: {} still building after {} s", what,
                            run->jobs[index].label, seconds);
            }
            if (busy > PrecacheGiveUp() && give_up(slot, index, seconds)) {
                --active;
                // Another thread takes its place, unless every thread was lost this way: then
                // the driver is stuck and the rest is not reached.
                if (given_up < num_threads) {
                    spawn();
                    ++active;
                } else {
                    run->cancel = true;
                }
            }
        }
        if (active == 0) {
            break;
        }
        if (std::chrono::steady_clock::now() - last_record > std::chrono::seconds{30}) {
            // An app closed during a long run keeps what it built.
            last_record = std::chrono::steady_clock::now();
            record_built();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    run->cancel = true;
    // Cancel: the threads stop after their current build, which is waited for briefly; one that
    // does not return is left to it.
    for (int wait = 0; wait < 300; ++wait) {
        if (std::ranges::none_of(threads, [](const auto& entry) {
                return entry.second->state.load() == Slot::Running;
            })) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    for (auto& [thread, slot] : threads) {
        const size_t index = slot->job.load();
        if (slot->state.load() == Slot::Running && index != SIZE_MAX) {
            give_up(*slot, index,
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::nanoseconds{now_ns() - slot->since.load()})
                        .count());
        }
        if (slot->state.load() == Slot::Left) {
            thread.detach();
        } else {
            thread.join();
        }
    }
    status(false);
    const size_t total_built = run->built.load();
    const size_t done = run->done.load();
    if (compute_pending && !skipped && done == total) {
        if (const auto stamp_done = stamp(file_entries)) {
            ShaderWarm::WriteWarmMarker(warm_file, *stamp_done);
        }
    }
    record_built();
    if (!stop_loading.stop_requested() && total_built != 0) {
        // Without this, the builds reach the file only at the next boot, game change or clean
        // close, and are lost if the app ends otherwise.
        QueueVulkanPipelineCacheSave(true);
    }
    LOG_INFO(Render_Vulkan,
             "{}: {} built, {} skipped (already built), {} failed in {:.1f} s on {} threads ({}, "
             "{} given up, {} not reached)",
             what, total_built, already_built, done - total_built,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
             num_threads,
             stop_loading.stop_requested() ? "stopped" : (skipped ? "skipped" : "finished"),
             given_up, total - done);
    return !skipped && !stop_loading.stop_requested();
}

static_assert(sizeof(GraphicsPipelineCacheKey) == VideoCommon::ShaderWarm::GraphicsKeySize);
static_assert(offsetof(GraphicsPipelineCacheKey, state) +
                  offsetof(FixedPipelineState, attribute_types) ==
              VideoCommon::ShaderWarm::AttributeTypesOffset);

void PipelineCache::PrecacheGraphicsLibraries(u64 title_id, std::stop_token stop_loading,
                                              const VideoCore::DiskResourceLoadCallback& callback) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    if (!ShaderWarm::GraphicsPrecacheRequested() ||
        !device.IsExtGraphicsPipelineLibrarySupported()) {
        return;
    }
    const std::filesystem::path warm_file = ShaderWarm::GraphicsWarmPath(title_id);
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    std::error_code ec;
    if (!std::filesystem::exists(warm_file, ec)) {
        LOG_INFO(Render_Vulkan, "Shader precache: no graphics file at {}", path);
        return;
    }
    const auto stamp = [&](u64 entries) {
        return ShaderWarm::CurrentWarmStamp(warm_file, entries, device.GetDriverVersion(),
                                            device.GetDeviceID(), CACHE_VERSION);
    };
    const auto current = stamp(0);
    const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
    if (current && marker && marker->Matches(*current)) {
        LOG_INFO(Render_Vulkan, "Shader precache: graphics programs already done for this driver");
        return;
    }
    // Streamed: the file can be over a gigabyte; threads take the next entry under the lock.
    struct Reader {
        std::mutex mutex;
        std::ifstream file;
        std::streampos end{};
        bool failed{};
        size_t read{};
        std::streamoff position{};
    };
    Reader reader;
    try {
        reader.file.open(warm_file, std::ios::binary | std::ios::ate);
        reader.file.exceptions(std::ifstream::failbit);
        reader.end = reader.file.tellg();
        reader.file.seekg(0, std::ios::beg);
        std::array<char, 8> magic{};
        u32 version{};
        reader.file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
        if (magic != expected_magic || version != CACHE_VERSION) {
            LOG_WARNING(Render_Vulkan, "Shader precache: {} is not a version {} file", path,
                        CACHE_VERSION);
            return;
        }
        reader.position = reader.file.tellg();
    } catch (const std::exception& e) {
        LOG_WARNING(Render_Vulkan, "Shader precache: opening {} failed: {}", path, e.what());
        return;
    }
    const size_t known_total = static_cast<size_t>(ShaderWarm::ReadWarmCount(warm_file));
    const auto begin = std::chrono::steady_clock::now();
    const size_t num_threads = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 64);
    std::atomic<size_t> built{};
    std::atomic<size_t> failed{};
    std::atomic<bool> cancel{false};
    std::atomic<size_t> running{num_threads};
    bool skipped = false;
    LOG_INFO(Render_Vulkan,
             "Shader precache: {} graphics programs ({} MB) to build as libraries on {} threads",
             known_total, static_cast<long long>(reader.end) / (1024 * 1024), num_threads);
    {
        std::vector<std::jthread> threads;
        for (size_t thread = 0; thread < num_threads; ++thread) {
            threads.emplace_back([this, &reader, &built, &failed, &cancel, &running] {
                Common::SetCurrentThreadName("VkShaderPrecache");
                Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
                static const GuessMissing guess{};
                GraphicsPipelineLibraryCache::SetNoRetainOnThisThread(true);
                ShaderPools pools;
                while (!cancel.load(std::memory_order_relaxed)) {
                    FileEnvironment env;
                    GraphicsPipelineCacheKey key{};
                    {
                        std::scoped_lock lock{reader.mutex};
                        if (reader.failed || reader.file.tellg() == reader.end) {
                            break;
                        }
                        try {
                            u32 num_envs{};
                            reader.file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                            if (num_envs != 1) {
                                throw std::runtime_error("unexpected entry");
                            }
                            env.Deserialize(reader.file);
                            reader.file.read(reinterpret_cast<char*>(&key), sizeof(key));
                            ++reader.read;
                            reader.position = reader.file.tellg();
                        } catch (const std::exception& e) {
                            LOG_WARNING(Render_Vulkan,
                                        "Shader precache: graphics entry {} unreadable ({}), "
                                        "skipping the rest",
                                        reader.read + 1, e.what());
                            reader.failed = true;
                            break;
                        }
                    }
                    env.SetGuessMissing(&guess);
                    bool ok = false;
                    try {
                        Shader::Environment* const env_ptr = &env;
                        ok = WarmGraphicsLibraries(pools, key, std::span{&env_ptr, 1});
                    } catch (const std::exception& e) {
                        LOG_DEBUG(Render_Vulkan, "Shader precache: {}", e.what());
                    }
                    (ok ? built : failed).fetch_add(1);
                }
                running.fetch_sub(1);
            });
        }
        while (running != 0) {
            if (stop_loading.stop_requested() || ShaderWarm::TakeSkipWarm()) {
                cancel = true;
                skipped = true;
                break;
            }
            if (callback) {
                const size_t done = built + failed;
                // The builder's recorded count, or else an estimate from the bytes read.
                size_t total = known_total;
                if (total == 0) {
                    std::scoped_lock lock{reader.mutex};
                    const std::streamoff header = 12;
                    const std::streamoff end = reader.end;
                    if (reader.position > header) {
                        total = static_cast<size_t>(static_cast<double>(reader.read) *
                                                    static_cast<double>(end - header) /
                                                    static_cast<double>(reader.position - header));
                    }
                }
                callback(VideoCore::LoadCallbackStage::Warm, done, std::max(total, done));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
    }
    if (!skipped && !reader.failed && reader.end == std::streampos{reader.position}) {
        if (const auto stamp_done = stamp(reader.read)) {
            ShaderWarm::WriteWarmMarker(warm_file, *stamp_done);
        }
    }
    LOG_INFO(Render_Vulkan,
             "Shader precache: graphics libraries {}, {} built, {} failed, {:.1f} s",
             stop_loading.stop_requested() ? "stopped" : (skipped ? "skipped" : "finished"),
             built.load(), failed.load(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.load(std::memory_order_acquire)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    QueueVulkanPipelineCacheSave(false);
}

bool PipelineCache::VulkanPipelineCacheSaveDue(std::chrono::steady_clock::time_point& last) const {
    // As QueueVulkanPipelineCacheFlush: at most once a second per MiB the cache holds, as each
    // write copies all of it.
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (now - last < interval) {
        return false;
    }
    last = now;
    return true;
}

void PipelineCache::QueueVulkanPipelineCacheSave(bool again_if_busy) {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        // The write under way may have read the driver data already: one more follows it.
        if (again_if_busy) {
            save_again.store(true, std::memory_order_release);
        }
        return;
    }
    serialization_thread.QueueWork([this] {
        do {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
            size_t size = 0;
            vulkan_pipeline_cache.Read(&size, nullptr);
            last_cache_size.store(size, std::memory_order_relaxed);
        } while (save_again.exchange(false, std::memory_order_acq_rel));
        flush_in_flight.store(false, std::memory_order_release);
    });
}

namespace {
// EDEN_NX2_PIPELINE_DIFF=1: for a new graphics pipeline whose shaders an earlier one uses too,
// log which parts of the fixed state differ from that one (finds state that splits the same
// draws into many pipelines, each translated on the GPU thread).
void ReportNewPipelineKey(const GraphicsPipelineCacheKey& key) {
    static const bool enabled = std::getenv("EDEN_NX2_PIPELINE_DIFF") != nullptr;
    if (!enabled) {
        return;
    }
    static std::unordered_map<u64, GraphicsPipelineCacheKey> last_by_shaders;
    static u32 logged{};
    const u64 shaders = Common::CityHash64(reinterpret_cast<const char*>(key.unique_hashes.data()),
                                           sizeof(key.unique_hashes));
    const auto [it, first] = last_by_shaders.try_emplace(shaders, key);
    if (first) {
        return;
    }
    const FixedPipelineState& now = key.state;
    const FixedPipelineState& before = it->second.state;
    std::string diffs;
    if (now.raw1 != before.raw1) diffs += fmt::format(" raw1^{:#x}", now.raw1 ^ before.raw1);
    if (now.raw2 != before.raw2) diffs += fmt::format(" raw2^{:#x}", now.raw2 ^ before.raw2);
    if (now.color_formats != before.color_formats) {
        diffs += fmt::format(" color_formats {:02x} -> {:02x}", fmt::join(before.color_formats, ""),
                             fmt::join(now.color_formats, ""));
    }
    if (now.alpha_test_ref != before.alpha_test_ref) {
        diffs += fmt::format(" alpha_test_ref {:#x} -> {:#x} (func {})", before.alpha_test_ref,
                             now.alpha_test_ref, now.alpha_test_func.Value());
    }
    if (now.point_size != before.point_size) {
        diffs += fmt::format(" point_size {:#x} -> {:#x}", before.point_size, now.point_size);
    }
    if (now.viewport_swizzles != before.viewport_swizzles) diffs += " viewport_swizzles";
    if (now.attribute_types != before.attribute_types) {
        diffs += fmt::format(" attribute_types^{:#x}", now.attribute_types ^ before.attribute_types);
    }
    if (std::memcmp(&now.dynamic_state, &before.dynamic_state, sizeof(now.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    if (now.driver_id != before.driver_id || now.driver_version != before.driver_version) {
        diffs += " driver";
    }
    const size_t size = now.Size();
    if (diffs.empty() && size > offsetof(FixedPipelineState, attachments)) {
        diffs = fmt::format(" past attachments (key size {})", size);
    }
    it->second = key;
    if (logged++ < 400) {
        LOG_INFO(Render_Vulkan, "New pipeline for known shaders {:016x} differs in:{}", shaders,
                 diffs.empty() ? std::string{" nothing in the key prefix"} : diffs);
    }
}
} // Anonymous namespace

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        ReportNewPipelineKey(graphics_key);
        pipeline = TakeBackgroundPipeline(graphics_key);
        if (!pipeline) {
            // EDEN_NX2_ASYNC_TRANSLATE=1/2: a draw that may be skipped has its new pipeline
            // translated on a translation thread, and is skipped (as for an unbuilt pipeline)
            // until it is ready. The entry stays empty meanwhile.
            if (translate_queue && EnqueueTranslation()) {
                NoteSkippedLookup();
                ++translate_stats.skipped;
                return nullptr;
            }
            pipeline = CreateGraphicsPipeline();
        }
    } else if (!pipeline && translate_queue) {
        bool pending{};
        pipeline = AdoptTranslation(pending);
        if (pending) {
            return nullptr;
        }
    }
    if (!pipeline) {
        return nullptr;
    }
    const auto save_interval = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::PipelineRetention)
                                   ? std::chrono::steady_clock::duration{std::chrono::seconds{15}}
                                   : std::chrono::steady_clock::duration{std::chrono::minutes{2}};
    if ((used_this_session.insert(pipeline.get()).second &&
         std::chrono::steady_clock::now() - used_pipelines_saved > save_interval) ||
        save_used_requested.exchange(false, std::memory_order_relaxed)) {
        SaveUsedPipelines();
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    current_pipeline->MarkUsed(frame_count);
    return BuiltPipeline(current_pipeline);
}

bool PipelineCache::WarmGraphicsLibraries(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                          std::span<Shader::Environment* const> envs) {
    if (!device.IsExtGraphicsPipelineLibrarySupported()) {
        return false;
    }
    pools.ReleaseContents();
    const auto pipeline{
        CreateGraphicsPipeline(pools, key, envs, nullptr, false, nullptr, nullptr, true)};
    return pipeline && !pipeline->Failed();
}

void PipelineCache::TickFrame() {
    ++frame_count;
    if (translate_queue) {
        TickTranslations();
    }
    if (device.IsExtGraphicsPipelineLibrarySupported()) {
        graphics_libraries.Report();
    }
    while (!evicted_pipelines.empty() && scheduler.IsFree(evicted_pipelines.front().first)) {
        evicted_pipelines.pop_front();
    }
    // A retired translation's build may still be queued on the pipeline workers: destroying it
    // now would wait for that build, or run it here on the GPU thread.
    std::erase_if(retired_translations, [this](const auto& entry) {
        return scheduler.IsFree(entry.first) && entry.second->IsBuilt();
    });
    while (!retired_compute_pipelines.empty() &&
           scheduler.IsFree(retired_compute_pipelines.front().first) &&
           retired_compute_pipelines.front().second->IsBuildTaskDone()) {
        retired_compute_pipelines.pop_front();
    }
    constexpr u64 EvictCheckFrames = 600;
    if (frame_count % EvictCheckFrames == 0 ||
        (device.IsMoltenVK() &&
         Common::PlatformEvents::MemoryPressureGeneration() != evict_pressure_generation)) {
        EvictUnusedPipelines();
    }
}

void PipelineCache::RetireTranslatedPipeline(std::unique_ptr<GraphicsPipeline> pipeline) {
    if (pipeline) {
        retired_translations.emplace_back(scheduler.CurrentTick(), std::move(pipeline));
    }
}

void PipelineCache::EvictUnusedPipelines() {
    // Building a pipeline again on MoltenVK means a Metal compile mid-game, far dearer than the
    // 2-3 ms below, so none are evicted there unless the OS reports memory pressure: then every
    // one unused for ~10 s goes, ahead of a jetsam kill.
    bool under_pressure = false;
    // Only near the limit (PressureEvictLeftBytes): fewer, older ones, boot-built ones kept.
    bool critical_only = false;
    if (device.IsMoltenVK()) {
        const u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
        if (pressure == evict_pressure_generation) {
            return;
        }
        evict_pressure_generation = pressure;
        under_pressure = true;
        if (const u64 limit = PressureEvictLeftBytes(); limit != 0) {
            const u64 left = Common::PlatformEvents::AvailableProcessMemory();
            if (left != 0 && left >= limit) {
                ++pressure_evictions_skipped;
                return;
            }
            critical_only = true;
        }
    }
    // EDEN_NX2_PRELOAD_ALL_PIPELINES asked for every cached pipeline built: evicting the ones not
    // used yet threw the preload away after 9000 frames (Ocarina: 22,510 of 24,285, then rebuilt
    // one by one in new places).
    if (!under_pressure && PreloadAllPipelines(device)) {
        return;
    }
    // Each built pipeline holds ~0.45 MB of driver memory and none was ever destroyed, so it grew
    // with every place a session went (title 1: 1,500 on its title screen alone). One unused for
    // this long is destroyed; it takes 2-3 ms to build again from the driver's pipeline cache.
    const u64 evict_after_frames = critical_only ? 3600 : under_pressure ? 600 : 9000;
    if (frame_count <= evict_after_frames) {
        return;
    }
    // Only while the cache is large: each one evicted is translated again on the GPU thread when
    // its place is revisited (~10 ms). Ocarina of Time kept ~1,900 of them and evicted 3,400 in
    // 25 minutes of play, then rebuilt 1,850 (14 s of stalls, 3.8 s in one 10 s window). 8,000
    // pipelines are ~3.6 GB of driver memory at most. EDEN_NX2_PIPELINE_EVICT_ABOVE sets it.
    // Past it only the excess goes, least recently used first (from zortos293/Lichii#408), so
    // crossing it doesn't drop every stale pipeline at once.
    // Under pressure the 256 most recently used stay, and fewer than 128 stale ones are kept too:
    // an iPhone's 1,041 evicted at once gave back at most ~28 MiB of Metal memory and ~130 MiB of
    // malloc with other caches freeing at the same time, and the later evictions (27 -> 7 -> 4
    // -> 3) next to nothing, each one rebuilt with a Metal compile when its place came back.
    constexpr size_t PressureKeptPipelines = 256;
    constexpr size_t PressureMinEvicted = 128;
    const size_t evict_above = critical_only    ? PressureKeptPipelines * 2
                               : under_pressure ? PressureKeptPipelines
                                                : EvictAbovePipelines();
    if (graphics_cache.size() <= evict_above) {
        return;
    }
    const u64 cutoff = frame_count - evict_after_frames;
    // A pipeline built from the disk cache at boot that no draw has used yet stays, up to a budget
    // (from zortos293/Lichii#550): evicting it throws the prebuild away, and the game translates it
    // again on the GPU thread when it reaches that content. Past the budget (~0.9 GB of driver
    // memory) they go like the rest, so a large cache can't hold their memory all session.
    const size_t max_unused_prebuilt = under_pressure && !critical_only ? 0 : 2000;
    size_t kept_prebuilt = 0;
    std::vector<std::pair<u64, GraphicsPipelineCacheKey>> candidates;
    for (const auto& [key, pipeline] : graphics_cache) {
        if (pipeline == nullptr || pipeline.get() == current_pipeline || !pipeline->IsBuilt() ||
            pipeline->LastUse() >= cutoff) {
            continue;
        }
        if (!pipeline->WasUsed() && kept_prebuilt < max_unused_prebuilt) {
            ++kept_prebuilt;
            continue;
        }
        candidates.emplace_back(pipeline->LastUse(), key);
    }
    size_t evicted =
        under_pressure
            ? VideoCommon::PressurePipelinesToEvict(graphics_cache.size(), candidates.size(),
                                                    evict_above, PressureMinEvicted)
            : VideoCommon::GraphicsPipelinesToEvict(graphics_cache.size(), candidates.size(),
                                                    evict_above);
    if (critical_only) {
        evicted = (std::min)(evicted, PressureEvictMax());
    }
    if (evicted == 0) {
        return;
    }
    std::nth_element(candidates.begin(), candidates.begin() + (evicted - 1), candidates.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    const u64 tick = scheduler.CurrentTick();
    for (size_t i = 0; i < evicted; ++i) {
        const auto it = graphics_cache.find(candidates[i].second);
        if (used_this_session.erase(it->second.get()) != 0) {
            used_and_evicted.insert(it->first.Hash());
        }
        evicted_pipelines.emplace_back(tick, std::move(it->second));
        graphics_cache.erase(it);
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    // Common.Memory: the iOS log filter drops Render.Vulkan below Warning.
    LOG_INFO(Common_Memory,
             "Evicted {} graphics pipelines unused for {} frames, {} left{}", evicted,
             evict_after_frames, graphics_cache.size(),
             critical_only
                 ? fmt::format(" ({} MiB left to the process; {} pressure signals before passed "
                               "with more than {} MiB left)",
                               Common::PlatformEvents::AvailableProcessMemory() >> 20,
                               pressure_evictions_skipped, PressureEvictLeftBytes() >> 20)
                 : std::string{});
}

std::unique_ptr<GraphicsPipeline> PipelineCache::TakeBackgroundPipeline(
    const GraphicsPipelineCacheKey& key) {
    std::unique_lock lock{background_mutex, std::defer_lock};
    if (QuietPrebuildMode() == 0) {
        lock.lock();
    } else {
        // The background threads hold this briefly, but at idle priority one can be preempted
        // while holding it: the GPU thread then builds the pipeline itself rather than wait. Not
        // claimed, its background build (if any) stays in background_ready, unused.
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::microseconds{200};
        while (!lock.try_lock()) {
            if (std::chrono::steady_clock::now() > give_up) {
                return nullptr;
            }
            std::this_thread::yield();
        }
    }
    if (const auto it = background_ready.find(key); it != background_ready.end()) {
        auto pipeline = std::move(it->second);
        background_ready.erase(it);
        ++background_taken;
        return pipeline;
    }
    if (background_active.load(std::memory_order_acquire)) {
        background_claimed.insert(key);
    }
    return nullptr;
}

void PipelineCache::StopBackgroundPipelines() {
    background_stop.store(true, std::memory_order_relaxed);
    // Joins the threads: a build in progress finishes, queued ones are dropped.
    background_workers.reset();
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    const bool discard = DrawMayBeSkipped();
    if (discard) {
        NoteSkippedLookup();
    }
    return discard ? nullptr : pipeline;
}

int AsyncDiscardMode() {
    // Read on first use (the first missing pipeline), after nx2_switches.txt was applied.
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_DISCARD");
        // 2 by default: with an empty shader cache, mode 1 dropped draws into render-once layers
        // for good (title text, NPC chat bubbles, LUT/irradiance probes); mode 2 skips only into
        // targets cleared every frame and waits for the rest. Warm caches are unchanged.
        const int parsed = value == nullptr ? 2 : value[0] == '0' ? 0 : value[0] == '1' ? 1 : 2;
        static constexpr std::array<const char*, 3> what{
            "wait for every unbuilt pipeline",
            "skip unbuilt pipelines by the legacy heuristic",
            "skip unbuilt pipelines only into render targets cleared every frame",
        };
        LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_DISCARD={}: native draws {}", parsed,
                 what[parsed]);
        return parsed;
    }();
    return mode;
}

bool PipelineCache::DrawMayBeSkipped() const {
    if (discard_decision.serial == lookup_serial) {
        return discard_decision.skip;
    }
    const int mode = AsyncDiscardMode();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    bool recurring = true;
    if (mode == 2 && native) {
        recurring = texture_cache.ClassifyDrawTargets().kind ==
                    VideoCommon::DrawTargetClass::Recurring;
    }
    const bool skip = CanDiscardPendingDraw(native, maxwell3d->regs.zeta_enable != 0,
                                            draw_state.draw_element_count,
                                            draw_state.draw_element_count, mode != 0, recurring);
    discard_decision = DiscardDecision{.serial = lookup_serial, .skip = skip};
    return skip;
}

namespace {
// Without VK_EXT_mesh_shader: logs, once per shader set, what a skipped task/mesh pipeline would
// need from an emulation (task stage, per-primitive outputs, topology, output sizes).
template <typename Programs>
void ReportUnsupportedMeshPipeline(const GraphicsPipelineCacheKey& key, const Programs& programs) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    static u32 with_task{}, per_primitive{}, non_triangle{}, total{};
    u64 shaders{};
    for (const u64 hash : key.unique_hashes) {
        shaders = shaders * 31 + hash;
    }
    std::scoped_lock lock{mutex};
    if (!seen.insert(shaders).second) {
        return;
    }
    const Shader::IR::Program* task{};
    const Shader::IR::Program* mesh{};
    bool has_pixel{};
    for (size_t index = 0; index < programs.size(); ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const auto& program{programs[index]};
        if (program.stage == Shader::Stage::Task) {
            task = &program;
        } else if (program.stage == Shader::Stage::Mesh) {
            mesh = &program;
        } else if (program.stage == Shader::Stage::Fragment) {
            has_pixel = true;
        }
    }
    ++total;
    if (mesh == nullptr) {
        LOG_WARNING(Render_Vulkan, "Mesh emulation survey: pipeline {:016x} without a mesh stage",
                    shaders);
        return;
    }
    const auto& info{mesh->mesh};
    u32 position{}, generic{}, other{};
    std::string others;
    for (const u32 line : info.vertex_lines) {
        const auto attribute{static_cast<Shader::IR::Attribute>(line)};
        if (attribute >= Shader::IR::Attribute::PositionX &&
            attribute <= Shader::IR::Attribute::PositionW) {
            ++position;
        } else if (Shader::IR::IsGeneric(attribute)) {
            ++generic;
        } else {
            ++other;
            others += fmt::format(" {}", line);
        }
    }
    with_task += task != nullptr;
    per_primitive += !info.primitive_lines.empty();
    non_triangle += info.topology != 4;
    LOG_WARNING(Render_Vulkan,
                "Mesh emulation survey #{}: task={} (payload {} words, wg {}x{}x{}) mesh wg "
                "{}x{}x{} max_vertices={} max_primitives={} topology={} vertex_lines={} "
                "(position {} generic {} other {}:{}) primitive_lines={} pixel={} | totals: "
                "task {} per-primitive {} non-triangle {}",
                total, task != nullptr, info.payload_words, task ? task->workgroup_size[0] : 0,
                task ? task->workgroup_size[1] : 0, task ? task->workgroup_size[2] : 0,
                mesh->workgroup_size[0], mesh->workgroup_size[1], mesh->workgroup_size[2],
                info.max_vertices, info.max_primitives, info.topology, info.vertex_lines.size(),
                position, generic, other, others, info.primitive_lines.size(), has_pixel,
                with_task, per_primitive, non_triangle);
}

// With VK_EXT_mesh_shader: why a task/mesh pipeline cannot be built natively, if it cannot. Such a
// pipeline takes the compute emulation instead. MoltenVK (KhronosGroup/MoltenVK#2842) builds the
// mesh program as a Metal mesh function: SPIRV-Cross keeps its outputs, their counts and triangle
// indices in threadgroup memory next to the program's own workgroup memory (the attribute and map
// regions of emit_spirv_mesh.cpp), and Metal refuses a function that needs more than the device has.
template <typename Programs>
std::optional<std::string> NativeMeshLimitExceeded(const Device& device,
                                                   const GraphicsPipelineCacheKey& key,
                                                   const Programs& programs) {
    const Shader::IR::Program* task{};
    const Shader::IR::Program* mesh{};
    for (size_t index = 1; index < programs.size(); ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        if (programs[index].stage == Shader::Stage::Task) {
            task = &programs[index];
        } else if (programs[index].stage == Shader::Stage::Mesh) {
            mesh = &programs[index];
        }
    }
    if (mesh == nullptr) {
        return std::nullopt;
    }
    const auto& limits{device.GetMeshShaderProperties()};
    const auto& info{mesh->mesh};
    if (info.max_vertices > limits.maxMeshOutputVertices ||
        info.max_primitives > limits.maxMeshOutputPrimitives) {
        return fmt::format("{} vertices / {} primitives, device {} / {}", info.max_vertices,
                           info.max_primitives, limits.maxMeshOutputVertices,
                           limits.maxMeshOutputPrimitives);
    }
    if (task != nullptr && info.payload_words * 4 > limits.maxTaskPayloadSize) {
        return fmt::format("task payload {} B, device {} B", info.payload_words * 4,
                           limits.maxTaskPayloadSize);
    }
    if (!device.IsMoltenVK()) {
        return std::nullopt;
    }
    // Output locations: one vec4 per generic, plus position and the other built-ins.
    const auto locations = [](const auto& lines) {
        std::bitset<Shader::IR::NUM_GENERICS> generics;
        bool builtins{};
        for (const u32 line : lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                generics.set(Shader::IR::GenericAttributeIndex(attribute));
            } else {
                builtins = true;
            }
        }
        return static_cast<u32>(generics.count()) + (builtins ? 1U : 0U);
    };
    const u32 vertex_locations{locations(info.vertex_lines)};
    const u32 primitive_locations{locations(info.primitive_lines)};
    if (vertex_locations * 4 > limits.maxMeshOutputComponents) {
        return fmt::format("{} output components, device {}", vertex_locations * 4,
                           limits.maxMeshOutputComponents);
    }
    const u64 bytes{MetalMeshThreadgroupBytes({
        .max_vertices = info.max_vertices,
        .max_primitives = info.max_primitives,
        .primitive_vertices = info.PrimitiveVertices(),
        .vertex_lines = static_cast<u32>(info.vertex_lines.size()),
        .primitive_lines = static_cast<u32>(info.primitive_lines.size()),
        .vertex_locations = vertex_locations,
        .primitive_locations = primitive_locations,
        .shared_memory_bytes = mesh->shared_memory_size,
        .direct_outputs = MeshDirectOutputs(device),
    })};
    const u64 available{device.GetMaxComputeSharedMemorySize()};
    if (bytes > available) {
        return fmt::format("threadgroup memory {} B, device {} B", bytes, available);
    }
    return std::nullopt;
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateCachedGraphicsPipeline(
    const GraphicsPipelineCacheKey& key, std::vector<std::vector<FileEnvironment>>& records,
    PipelineStatistics* statistics) {
    ShaderPools pools;
    for (auto& envs : records) {
        pools.ReleaseContents();
        boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
        for (auto& env : envs) {
            env_ptrs.push_back(&env);
        }
        if (auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), statistics, false)}) {
            return pipeline;
        }
    }
    // A pipeline that keeps failing however often it is stored again is not appended every session.
    constexpr size_t MaxRecords = 3;
    if (records.size() < MaxRecords) {
        std::scoped_lock lock{failed_disk_pipelines_mutex};
        failed_disk_pipelines.insert(key.Hash());
    }
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools,
    std::unique_ptr<EmissionBudget::Reservation> reservation, bool libraries_only) try {
    auto hash = key.Hash();
    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);
    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};
    bool mesh_unsupported{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
            !device.IsGeometryShaderSupported() && maxwell3d != nullptr &&
            maxwell3d->GetNativeMacroEngine().native_turing) {
            // A native (Switch 2) geometry program on a device without geometry shaders: the
            // pipeline cannot run it, so its draws are skipped.
            static std::atomic<bool> logged{};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "Skipping native geometry shader pipelines: the device has no geometry shaders");
            }
            return nullptr;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline: eligible ones
        // are emulated below (compute plus a vertex pipeline), the others skipped.
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "{} has no VK_EXT_mesh_shader: task/mesh pipelines are emulated with "
                            "compute where they can be (triangles, points or lines without "
                            "per-primitive outputs) and skipped otherwise; indirect mesh draws "
                            "are skipped",
                            device.GetModelName());
            }
            mesh_unsupported = true;
        }

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env, host_info);
        }

        if (const auto edits = Shader::Optimization::ApplyOotVisualWorkaround(
                programs[index], program_id != 0 ? program_id : visual_workaround_title,
                key.unique_hashes[index],
                oot_visual_workaround)) {
            LOG_WARNING(Render_Vulkan,
                        "TEMPORARY OoT visual workaround: shader {:016x}, {} scalar edits",
                        key.unique_hashes[index], *edits);
        }
        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return nullptr;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
        if (key.state.dynamic_vertex_input != 0 &&
            programs[index].stage == Shader::Stage::VertexB) {
            // The key dropped the types of attributes outside the input map: such a read now gets
            // the default value (EDEN_NX2_KEY_ALL_ATTRIBUTES=1 keeps every type).
            const u64 attribute_mask = VideoCommon::VertexAttributeMaskOf(env);
            for (size_t attribute = 0; attribute < Shader::IR::NUM_GENERICS; ++attribute) {
                if (programs[index].info.loads.Generic(attribute) &&
                    ((attribute_mask >> (attribute * 2)) & 3) == 0) {
                    static std::atomic<u32> warned{};
                    if (warned.fetch_add(1, std::memory_order_relaxed) < 16) {
                        LOG_WARNING(Render_Vulkan,
                                    "Vertex program {:016x} reads attribute {} outside its input "
                                    "map",
                                    key.unique_hashes[index], attribute);
                    }
                    break;
                }
            }
        }
    }
    if (!mesh_unsupported && device.IsExtMeshShaderSupported()) {
        if (const auto reason{NativeMeshLimitExceeded(device, key, programs)}) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_WARNING(Render_Vulkan,
                            "Mesh pipeline {:016x} exceeds the native mesh limits ({}): emulated",
                            hash, *reason);
            }
            mesh_unsupported = true;
            emulated_native_mesh.store(true, std::memory_order_relaxed);
        }
    }
    // Mesh emulation without VK_EXT_mesh_shader (prototype): a mesh program, after an optional
    // task program, drawing triangles without per-primitive outputs, before a pixel program only.
    std::optional<MeshEmulationSetup> mesh_emulation;
    size_t mesh_index{};
    if (mesh_unsupported) {
        ReportUnsupportedMeshPipeline(key, programs);
        const bool has_task{key.unique_hashes[1] != 0 &&
                            programs[1].stage == Shader::Stage::Task};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0 && programs[index].stage == Shader::Stage::Mesh) {
                mesh_index = index;
                break;
            }
        }
        bool eligible{mesh_index != 0 && (mesh_index == 1 || has_task) &&
                      layer_source_program == nullptr};
        const Shader::IR::Program& mesh{programs[mesh_index]};
        // Points (0), lines (1) and triangles (4).
        const u32 topology{mesh.mesh.topology};
        eligible = eligible && (topology == 0 || topology == 1 || topology == 4) &&
                   mesh.mesh.primitive_lines.empty() &&
                   !mesh.mesh.vertex_lines.empty() && mesh.mesh.max_vertices != 0 &&
                   mesh.mesh.max_primitives != 0;
        for (size_t index = 1; index < Maxwell::MaxShaderProgram && eligible; ++index) {
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            eligible = emulated || key.unique_hashes[index] == 0 ||
                       programs[index].stage == Shader::Stage::Fragment;
        }
        // EDEN_NX2_NO_MESH_EMULATION=1: skip them all, as before mesh emulation.
        static const bool no_emulation{std::getenv("EDEN_NX2_NO_MESH_EMULATION") != nullptr};
        if (no_emulation) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "Mesh emulation off (EDEN_NX2_NO_MESH_EMULATION): task/mesh draws "
                            "are skipped");
            }
            return nullptr;
        }
        if (!eligible) {
            return nullptr;
        }
        // The output buffer's binding follows every program's (unified bindings, one per
        // descriptor element, as the SPIR-V backend numbers them). The task and mesh programs
        // share the compute stage, so their descriptors and the emulation's own buffers count
        // against one set of per-stage limits.
        u32 output_binding{};
        u32 compute_storage_buffers{has_task ? 4U : 1U};
        u32 compute_sampled_images{};
        u32 compute_resources{compute_storage_buffers};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            const Shader::Info& info{programs[index].info};
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            const auto add = [&](const auto& descriptors, u32* category) {
                const u32 count{Shader::NumDescriptors(descriptors)};
                output_binding += count;
                if (emulated) {
                    compute_resources += count;
                    if (category) {
                        *category += count;
                    }
                }
            };
            add(info.constant_buffer_descriptors, nullptr);
            add(info.storage_buffers_descriptors, &compute_storage_buffers);
            add(info.texture_buffer_descriptors, &compute_sampled_images);
            add(info.image_buffer_descriptors, nullptr);
            add(info.texture_descriptors, &compute_sampled_images);
            add(info.image_descriptors, nullptr);
        }
        if (compute_storage_buffers > device.GetMaxPerStageDescriptorStorageBuffers() ||
            compute_sampled_images > device.GetMaxPerStageDescriptorSampledImages() ||
            compute_resources > device.GetMaxPerStageResources()) {
            LOG_WARNING(Render_Vulkan,
                        "Mesh emulation skipped: {} storage buffers, {} sampled images and {} "
                        "resources exceed the compute stage limits",
                        compute_storage_buffers, compute_sampled_images, compute_resources);
            return nullptr;
        }
        mesh_emulation.emplace();
        mesh_emulation->output_binding = output_binding;
        if (has_task) {
            // After the output: the task payload, offset and argument buffers.
            mesh_emulation->has_task = true;
            mesh_emulation->mesh_stage = static_cast<u32>(mesh_index - 1);
            mesh_emulation->payload_stride = (std::max)(programs[1].mesh.payload_words, 2U);
            mesh_emulation->payload_binding = output_binding + 1;
            mesh_emulation->offsets_binding = output_binding + 2;
            mesh_emulation->args_binding = output_binding + 3;
            mesh_emulation->distribution_spirv =
                Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile);
        }
        mesh_emulation->block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh);
        mesh_emulation->max_primitives = mesh.mesh.max_primitives;
        mesh_emulation->primitive_vertices = mesh.mesh.PrimitiveVertices();
        mesh_emulation->max_vertices = mesh.mesh.max_vertices;
        mesh_emulation->vertex_lines = static_cast<u32>(mesh.mesh.vertex_lines.size());
        for (u32 line = 0; line < mesh.mesh.vertex_lines.size(); ++line) {
            const u32 attribute{mesh.mesh.vertex_lines[line]};
            const u32 position_x{static_cast<u32>(Shader::IR::Attribute::PositionX)};
            if (attribute >= position_x && attribute < position_x + 4) {
                mesh_emulation->position_lines[attribute - position_x] = line;
            }
        }
        static std::atomic<u32> emulated{};
        if (const u32 count{++emulated}; count <= 64) {
            // The input region addresses the mesh program reads (0 is its workgroup index).
            std::string reads;
            for (const Shader::IR::Block* block : mesh.blocks) {
                for (const Shader::IR::Inst& inst : block->Instructions()) {
                    if (inst.GetOpcode() != Shader::IR::Opcode::IsbeRead) {
                        continue;
                    }
                    const Shader::IR::Value address{inst.Arg(0)};
                    reads += address.IsImmediate() ? fmt::format(" {:#x}", address.U32())
                                                   : std::string{" dynamic"};
                }
            }
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation #{}: pipeline {:016x}, {} vertices x {} lines, {} triangles "
                     "per workgroup, output binding {}, task {} (payload {} words), input "
                     "reads:{}",
                     count, hash, mesh.mesh.max_vertices, mesh.mesh.vertex_lines.size(),
                     mesh.mesh.max_primitives, output_binding, has_task,
                     mesh_emulation->payload_stride, reads.empty() ? " none" : reads);
        }
    }
    // A tessellation pipeline the driver would refuse is skipped like a missing mesh pipeline:
    // the draw is dropped (see the pipeline check in the rasterizer) instead of failing the build.
    {
        const Shader::IR::Program* control{};
        const Shader::IR::Program* evaluation{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            if (programs[index].stage == Shader::Stage::TessellationControl) {
                control = &programs[index];
            } else if (programs[index].stage == Shader::Stage::TessellationEval) {
                evaluation = &programs[index];
            }
        }
        if (control != nullptr || evaluation != nullptr) {
            const u32 max_patch_size = device.GetMaxTessellationPatchSize();
            const char* reason = nullptr;
            if (!device.IsTessellationSupported()) {
                reason = "the device has no tessellation shaders";
            } else if (control == nullptr || evaluation == nullptr) {
                reason = "it has only one of the control and evaluation stages";
            } else if (control->invocations == 0 || control->invocations > max_patch_size) {
                reason = "the control shader's output vertex count is out of range";
            }
            if (reason != nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping tessellation pipeline: {} (tcs={} tes={}, tcs output "
                            "vertices {}, patch control points {}, device max patch size {})",
                            reason, control != nullptr, evaluation != nullptr,
                            control != nullptr ? control->invocations : 0,
                            key.state.patch_control_points_minus_one.Value() + 1, max_patch_size);
                return nullptr;
            }
        }
    }
    // The mesh fallback binds its output buffer (and, for task draws, payload, offsets and
    // indirect arguments) into the descriptor set every stage shares: with the stages' own storage
    // buffers they must fit the set's limit (zortos293/Lichii#642; only fallback pipelines).
    if (const u32 set_limit = host_info.max_descriptor_set_storage_buffers;
        mesh_emulation && set_limit != 0) {
        size_t storage_buffers{};
        for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
             index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                storage_buffers +=
                    Shader::NumDescriptors(programs[index].info.storage_buffers_descriptors);
            }
        }
        const size_t auxiliary_buffers{mesh_emulation->has_task ? 4U : 1U};
        if (!FitsStorageBufferSetLimit(storage_buffers, auxiliary_buffers, set_limit)) {
            throw Shader::NotImplementedException(
                "Binding {} storage buffers ({} for mesh fallback) to a descriptor set limited to {}",
                storage_buffers + auxiliary_buffers, auxiliary_buffers, set_limit);
        }
    }
    // Graphics pipeline libraries (EDEN_NX2_GPL). Left on the monolithic route: task and mesh
    // pipelines, transform feedback, a static rasterizer discard, and stages whose texture or
    // image counts do not fit the split rescaling bits.
    const bool use_libraries = [&] {
        if (!device.IsExtGraphicsPipelineLibrarySupported() || mesh_emulation ||
            key.state.xfb_enabled != 0) {
            return false;
        }
        if (key.state.extended_dynamic_state_2 == 0 &&
            key.state.dynamic_state.rasterize_enable == 0) {
            return false;
        }
        size_t pre_textures{}, pre_images{}, fragment_textures{}, fragment_images{};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            const bool is_emulated_stage =
                layer_source_program != nullptr &&
                index == static_cast<u32>(Maxwell::ShaderType::Geometry);
            if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
                continue;
            }
            const Shader::IR::Program& program{programs[index]};
            if (program.stage == Shader::Stage::Task || program.stage == Shader::Stage::Mesh) {
                return false;
            }
            const bool fragment = index + 1 == Maxwell::MaxShaderProgram;
            (fragment ? fragment_textures : pre_textures) += program.info.texture_descriptors.size();
            (fragment ? fragment_images : pre_images) += program.info.image_descriptors.size();
        }
        return pre_textures <= GPL_FRAGMENT_TEXTURE_SCALING_BASE &&
               fragment_textures <= GPL_FRAGMENT_TEXTURE_SCALING_BASE &&
               pre_images <= GPL_FRAGMENT_IMAGE_SCALING_BASE &&
               fragment_images <= GPL_FRAGMENT_IMAGE_SCALING_BASE;
    }();
    if (libraries_only && !use_libraries) {
        return nullptr;
    }
    std::array<u128, Maxwell::MaxShaderStage> spirv_hashes{};

    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> deferred_runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        if (mesh_emulation && (program.stage == Shader::Stage::Mesh ||
                               program.stage == Shader::Stage::Task)) {
            runtime_info.mesh_as_compute = true;
            runtime_info.mesh_output_binding = mesh_emulation->output_binding;
            if (mesh_emulation->has_task) {
                runtime_info.mesh_task_payload_binding = mesh_emulation->payload_binding;
                runtime_info.mesh_task_offsets_binding = mesh_emulation->offsets_binding;
                runtime_info.mesh_task_args_binding = mesh_emulation->args_binding;
                runtime_info.mesh_task_payload_stride = mesh_emulation->payload_stride;
            }
            if (program.stage == Shader::Stage::Mesh) {
                mesh_emulation->vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
                    profile, program.mesh, mesh_emulation->output_binding,
                    mesh_emulation->has_task ? mesh_emulation->args_binding : ~0U,
                    runtime_info.convert_depth_mode && !profile.support_native_ndc,
                    runtime_info.fixed_state_point_size.value_or(1.0f));
            }
        } else if (program.stage == Shader::Stage::Mesh && MeshDirectOutputs(device)) {
            runtime_info.mesh_direct_outputs = true;
        }
        ConvertLegacyToGeneric(program, runtime_info);
        // EDEN_NX2_MESH_EMULATION_DUMP=<dir>: write the emulation's vertex shader for spirv-val.
        // (Emitting the mesh program a second time here to dump its compute shader crashed the
        // build that followed, so only the vertex shader is written.)
        if (mesh_emulation && program.stage == Shader::Stage::Mesh) {
            if (const char* const dump{std::getenv("EDEN_NX2_MESH_EMULATION_DUMP")}) {
                const auto& code{mesh_emulation->vertex_spirv};
                std::ofstream file{fmt::format("{}/{:016x}.vert.spv", dump, hash),
                                   std::ios::binary};
                file.write(reinterpret_cast<const char*>(code.data()),
                           static_cast<std::streamsize>(code.size() * sizeof(u32)));
            }
        }
        PrepareStageBindings(binding, stage_index, use_libraries);
        if (emit_pools) {
            deferred_runtime_infos[stage_index] = runtime_info;
            if (VideoCommon::Nx2Perf::Verify()) {
                reference_spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
            }
        } else {
            const auto begin = std::chrono::steady_clock::now();
            modules[stage_index] = EmitShaderModule(
                device, profile, runtime_info, program, binding, key.unique_hashes[index],
                stage_index, {}, use_libraries ? &spirv_hashes[stage_index] : nullptr);
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
        }
        previous_stage = &program;
    }
    GraphicsPipeline::EmitStagesFunc emit_stages;
    if (emit_pools) {
        auto job = std::make_shared<DeferredStages>();
        job->reservation = std::move(reservation);
        job->pools = emit_pools;
        job->programs = std::move(programs);
        job->runtime_infos = std::move(deferred_runtime_infos);
        job->reference_spirv = std::move(reference_spirv);
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                infos[stage] = &job->programs[stage + 1].info;
            }
        }
        emit_stages = [this, job = std::move(job), hashes = key.unique_hashes, use_libraries](
                          std::array<u128, Maxwell::MaxShaderStage>& stage_hashes) mutable {
            const auto begin = std::chrono::steady_clock::now();
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> stage_modules;
            Shader::Backend::Bindings stage_binding;
            for (size_t stage = 0; stage < stage_modules.size(); ++stage) {
                if (job->runtime_infos[stage]) {
                    PrepareStageBindings(stage_binding, stage, use_libraries);
                    stage_modules[stage] = EmitShaderModule(
                        device, profile, *job->runtime_infos[stage], job->programs[stage + 1],
                        stage_binding, hashes[stage + 1], stage, job->reference_spirv[stage],
                        use_libraries ? &stage_hashes[stage] : nullptr);
                }
            }
            auto job_pools = std::move(job->pools);
            job.reset();
            job_pools->ReleaseContents();
            {
                std::scoped_lock lock{spare_pools_mutex};
                if (spare_pools.size() < 4) {
                    spare_pools.push_back(std::move(job_pools));
                }
            }
            AddRuntimeSpirvTime(1, begin);
            return stage_modules;
        };
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel && !libraries_only ? &workers
                                                                                   : nullptr};
    GraphicsPipeline::LibraryParams library_params{
        .library_cache = use_libraries ? &graphics_libraries : nullptr,
        .spirv_hashes = spirv_hashes,
        .libraries_only = libraries_only,
    };
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache,
        libraries_only ? nullptr : &shader_notify, device, descriptor_pool,
        guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(emit_stages),
        std::move(mesh_emulation), library_params);

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        try {
            // Walks the program for the dump; the one that failed may fail here again, and an
            // exception leaving this handler would end the process.
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        } catch (const Shader::Exception&) {
        }
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "Graphics pipeline compilation failed {}{}: {}", PipelineShaders(key),
              build_in_parallel ? "" : " (disk cache)", exception.what());
    return nullptr;
}

namespace {
// Pipelines built on the GPU thread at run time (shader translation, and the SPIR-V unless a worker
// emits it; the driver compile is on workers): the first 400 over 2 ms and every one over 100 ms
// are logged, and the totals every 10 s, to tell how much of the stutter on new content is
// translation.
void ReportRuntimePipeline(const char* kind, std::chrono::steady_clock::time_point begin,
                           const std::string& shaders) {
    static std::chrono::steady_clock::duration total{};
    static u32 count{}, logged{};
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - begin;
    total += elapsed;
    ++count;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    // Slow builds are always logged with their shaders, to find the programs that are slow to
    // translate.
    if (elapsed > std::chrono::milliseconds{100} ||
        (elapsed > std::chrono::milliseconds{2} && logged < 400)) {
        ++logged;
        LOG_INFO(Render_Vulkan, "{} pipeline built on the GPU thread in {:.1f} ms: {}", kind,
                 ms(elapsed), shaders);
    }
    if (now - window > std::chrono::seconds{10}) {
        const auto take = [](auto& counter) {
            return std::chrono::nanoseconds{counter.exchange(0, std::memory_order::relaxed)};
        };
        LOG_INFO(Render_Vulkan,
                 "Runtime pipelines in the last 10 s: {} built, {:.1f} ms on the GPU thread "
                 "({:.1f} ms of it graphics SPIR-V), {:.1f} ms graphics SPIR-V on workers, "
                 "{:.1f} ms of draws held for unbuilt pipelines",
                 count, ms(total), ms(take(runtime_spirv_ns[0])), ms(take(runtime_spirv_ns[1])),
                 ms(GraphicsPipeline::TakeHoldTime()));
        total = {};
        count = 0;
        window = now;
    }
}

// EDEN_NX2_ASYNC_TRANSLATE=3: GPU-thread builds translate from environment snapshots, and every
// read they recorded is answered again from the live state right after. Any difference is a
// snapshot gap (or a texture descriptor that needed a cache flush): the first 16 are logged with
// their pipeline, and the counts every 10 s. GPU thread only.
void NoteSnapshotCheck(const GraphicsPipelineCacheKey& key, size_t bytes,
                       std::span<Shader::Environment* const> envs) {
    static u32 pipelines{}, logged{};
    static size_t copied{}, reads{}, differing{}, gave_up{};
    static auto window = std::chrono::steady_clock::now();
    ++pipelines;
    copied += bytes;
    for (Shader::Environment* const env : envs) {
        auto& graphics_env = static_cast<GraphicsEnvironment&>(*env);
        const auto check = graphics_env.CheckAgainstLive();
        reads += check.reads;
        differing += check.differing;
        if (graphics_env.GuardedReadFailed()) {
            ++gave_up;
        }
        if (check.differing != 0 && logged < 16) {
            ++logged;
            LOG_WARNING(Render_Vulkan,
                        "Environment snapshot differs from the live state: {} reads of {} in {} "
                        "stage {}, first: {}",
                        check.differing, check.reads, PipelineShaders(key),
                        static_cast<u32>(env->ShaderStage()), check.first);
        }
        graphics_env.ReleaseSnapshotBytes();
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - window > std::chrono::seconds{10}) {
        LOG_INFO(Render_Vulkan,
                 "Environment snapshots in the last 10 s: {} pipelines, {} KB copied, {} reads "
                 "checked, {} differ, {} guarded reads gave up",
                 pipelines, copied / 1024, reads, differing, gave_up);
        pipelines = 0;
        copied = reads = differing = gave_up = 0;
        window = now;
    }
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Graphics", begin, PipelineShaders(graphics_key));
    };
    GraphicsEnvironments environments;
    const bool snapshot_check = VideoCommon::AsyncTranslateMode() == 3;
    const size_t snapshot_bytes =
        GetGraphicsEnvironments(environments, graphics_key.unique_hashes, snapshot_check);
    SCOPE_EXIT {
        // Right after the translation (the SPIR-V and the driver compile may still be queued),
        // before the GPU thread changes any state.
        if (snapshot_check) {
            NoteSnapshotCheck(graphics_key, snapshot_bytes, environments.Span());
        }
    };

    std::shared_ptr<ShaderPools> emit_pools;
    auto reservation = emission_budget.TryAcquire(
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::AsyncPipelines),
        maxwell3d->GetNativeMacroEngine().native_turing);
    if (reservation) {
        {
            std::scoped_lock lock{spare_pools_mutex};
            if (!spare_pools.empty()) {
                emit_pools = std::move(spare_pools.back());
                spare_pools.pop_back();
            }
        }
        if (!emit_pools) {
            emit_pools = std::make_shared<ShaderPools>();
        }
    } else {
        main_pools.ReleaseContents();
    }
    ShaderPools& pools = emit_pools ? *emit_pools : main_pools;
    auto pipeline{CreateGraphicsPipeline(pools, graphics_key, environments.Span(), nullptr, true,
                                         std::move(emit_pools), std::move(reservation))};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    StoreRuntimeGraphicsPipeline(graphics_key, environments.envs);
    return pipeline;
}

void PipelineCache::StoreRuntimeGraphicsPipeline(
    const GraphicsPipelineCacheKey& key,
    const std::array<GraphicsEnvironment, Maxwell::MaxShaderProgram>& environments) {
    if (disk_pipelines.contains(key.Hash())) {
        bool failed_to_load{};
        {
            std::scoped_lock lock{failed_disk_pipelines_mutex};
            failed_to_load = failed_disk_pipelines.erase(key.Hash()) != 0;
        }
        if (!failed_to_load) {
            // In the disk cache already, only not built at boot.
            QueueVulkanPipelineCacheFlush();
            return;
        }
        // Its records failed to load (see CreateCachedGraphicsPipeline): store this build after
        // them, to be tried first from the next boot on.
    }
    disk_pipelines.insert(key.Hash());
    serialization_thread.QueueWork([this, key, envs = environments] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
}

bool PipelineCache::EnqueueTranslation() {
    if (!DrawMayBeSkipped() || translate_sync_only.contains(graphics_key)) {
        return false;
    }
    if (translate_queue->InFlight() >= MaxTranslationsInFlight) {
        ++translate_stats.full;
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    auto job = std::make_unique<TranslationJob>();
    job->key = graphics_key;
    job->native = maxwell3d->GetNativeMacroEngine().native_turing;
    job->enqueued = begin;
    GraphicsEnvironments environments;
    translate_stats.snapshot_bytes +=
        GetGraphicsEnvironments(environments, graphics_key.unique_hashes, true);
    // A copy (environments declare a destructor, so they have no moves); the snapshot is shared.
    job->envs = environments.envs;
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    translate_stats.enqueue_time += elapsed;
    if (!translate_queue->Enqueue(std::move(job))) {
        ++translate_stats.full;
        return false;
    }
    ++translate_stats.queued;
    return true;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::AdoptTranslation(bool& pending) {
    pending = false;
    if (!translate_queue->Contains(graphics_key)) {
        // Not translating: a key that made no pipeline, which stays empty as before.
        return nullptr;
    }
    auto job = translate_queue->Take(graphics_key);
    if (!job) {
        if (DrawMayBeSkipped()) {
            NoteSkippedLookup();
            ++translate_stats.skipped;
            pending = true;
            return nullptr;
        }
        // A draw that must not be skipped (a full-screen pass) needs a pipeline a skippable draw
        // queued: wait a little for it, then build it here. Never a near match.
        const auto begin = std::chrono::steady_clock::now();
        job = translate_queue->Wait(graphics_key, TranslateWaitCap);
        ++translate_stats.waits;
        translate_stats.wait_time += std::chrono::steady_clock::now() - begin;
        if (!job) {
            translate_queue->Abandon(graphics_key);
            ++translate_stats.fallbacks;
            return CreateGraphicsPipeline();
        }
    }
    bool retry{};
    auto pipeline = AdoptJob(std::move(job), retry);
    if (!retry) {
        return pipeline;
    }
    // Not usable: translate again from a fresh snapshot of this draw, or build it here.
    if (!translate_sync_only.contains(graphics_key) && DrawMayBeSkipped() &&
        EnqueueTranslation()) {
        ++translate_stats.requeued;
        NoteSkippedLookup();
        ++translate_stats.skipped;
        pending = true;
        return nullptr;
    }
    ++translate_stats.fallbacks;
    return CreateGraphicsPipeline();
}

std::unique_ptr<GraphicsPipeline> PipelineCache::AdoptJob(std::unique_ptr<TranslationJob> job,
                                                          bool& retry) {
    retry = false;
    switch (job->outcome) {
    case TranslationJob::Outcome::Built:
        if (std::ranges::all_of(job->envs, [](GraphicsEnvironment& env) {
                return env.TextureReadsStillMatch();
            })) {
            const auto ready = std::chrono::steady_clock::now() - job->enqueued;
            ++translate_stats.adopted;
            translate_stats.translation_time += job->translation;
            translate_stats.ready_total += ready;
            translate_stats.ready_max = (std::max)(translate_stats.ready_max, ready);
            translate_failures.erase(job->key);
            auto pipeline = std::move(job->pipeline);
            if (!pipeline_cache_filename.empty()) {
                StoreRuntimeGraphicsPipeline(job->key, job->envs);
            }
            return pipeline;
        }
        ++translate_stats.stale;
        break;
    case TranslationJob::Outcome::Retry:
        ++translate_stats.gave_up;
        break;
    case TranslationJob::Outcome::NoPipeline:
    case TranslationJob::Outcome::Pending:
        ++translate_stats.no_pipeline;
        // A pipeline made before a failure on its thread is retired, not destroyed here.
        translate_queue->Discard(std::move(job));
        return nullptr;
    }
    retry = true;
    if (++translate_failures[job->key] >= MaxTranslateFailures) {
        translate_sync_only.insert(job->key);
    }
    translate_queue->Discard(std::move(job));
    return nullptr;
}

void PipelineCache::TranslateOnWorker(TranslationJob& job) {
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (job.key.unique_hashes[index] != 0) {
            env_ptrs.push_back(&job.envs[index]);
        }
    }
    // As the GPU thread's build: SPIR-V emission and the driver compile on the pipeline workers
    // when the emission budget has room, otherwise the SPIR-V here.
    std::shared_ptr<ShaderPools> emit_pools;
    auto reservation = emission_budget.TryAcquire(
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::AsyncPipelines), job.native);
    if (reservation) {
        {
            std::scoped_lock lock{spare_pools_mutex};
            if (!spare_pools.empty()) {
                emit_pools = std::move(spare_pools.back());
                spare_pools.pop_back();
            }
        }
        if (!emit_pools) {
            emit_pools = std::make_shared<ShaderPools>();
        }
    }
    thread_local ShaderPools thread_pools;
    if (!emit_pools) {
        thread_pools.ReleaseContents();
    }
    ShaderPools& pools = emit_pools ? *emit_pools : thread_pools;
    t_translate_worker = true;
    SCOPE_EXIT {
        t_translate_worker = false;
    };
    job.pipeline = CreateGraphicsPipeline(pools, job.key, MakeSpan(env_ptrs), nullptr, true,
                                          std::move(emit_pools), std::move(reservation));
    if (std::ranges::any_of(job.envs, [](const GraphicsEnvironment& env) {
            return env.GuardedReadFailed();
        })) {
        // Translated with zeros where a read gave up: never used, translated again from a fresh
        // snapshot. The pipeline stays in the job: the GPU thread retires it (AdoptJob, Discard)
        // rather than destroying it on this thread.
        job.outcome = TranslationJob::Outcome::Retry;
    }
    for (GraphicsEnvironment& env : job.envs) {
        env.ReleaseSnapshotBytes();
    }
}

void PipelineCache::TickTranslations() {
    translate_queue->RetireAbandoned();
    // Translations whose draws did not come back yet: adopted now, so the next draw finds them.
    for (auto& job : translate_queue->TakeFinished()) {
        const auto it = graphics_cache.find(job->key);
        if (it == graphics_cache.end() || it->second) {
            translate_queue->Discard(std::move(job));
            continue;
        }
        bool retry{};
        if (auto pipeline = AdoptJob(std::move(job), retry)) {
            it->second = std::move(pipeline);
            ++translate_stats.adopted_between_frames;
        } else if (retry) {
            // Not usable, and no draw to take a fresh snapshot from: the next lookup of the key
            // is a new one (queued again, or built on the GPU thread once it failed too often).
            graphics_cache.erase(it);
        }
    }
    if (frame_count % 60 == 0) {
        for (const auto& key : translate_queue->Sweep(TranslateSweepAge)) {
            ++translate_stats.swept;
            translate_sync_only.insert(key);
            if (const auto it = graphics_cache.find(key);
                it != graphics_cache.end() && !it->second) {
                graphics_cache.erase(it);
            }
            LOG_WARNING(Render_Vulkan,
                        "Async translate: {} still translating after {} s; built on the GPU "
                        "thread at its next draw instead",
                        PipelineShaders(key), TranslateSweepAge.count());
        }
    }
    auto& stats = translate_stats;
    const auto now = std::chrono::steady_clock::now();
    if (stats.window == std::chrono::steady_clock::time_point{}) {
        stats.window = now;
        return;
    }
    if (now - stats.window < std::chrono::seconds{10}) {
        return;
    }
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    const u32 adopted = stats.adopted;
    LOG_INFO(Render_Vulkan,
             "Async translate in the last 10 s: {} queued ({:.1f} ms of GPU thread for "
             "environments and snapshots, {} KB copied, {} not queued: full), {} adopted ({} "
             "between frames; first lookup to ready {:.1f} ms average, {:.1f} ms worst; {:.1f} ms "
             "translating on its threads), {} lookups skipped while translating, {} waits "
             "({:.1f} ms), {} built on the GPU thread after all, {} queued again ({} stale "
             "texture descriptors, {} guarded reads gave up), {} made no pipeline, {} swept, {} "
             "in flight",
             stats.queued, ms(stats.enqueue_time), stats.snapshot_bytes / 1024, stats.full,
             adopted, stats.adopted_between_frames,
             adopted != 0 ? ms(stats.ready_total) / adopted : 0.0, ms(stats.ready_max),
             ms(stats.translation_time), stats.skipped, stats.waits, ms(stats.wait_time),
             stats.fallbacks, stats.requeued, stats.stale, stats.gave_up, stats.no_pipeline,
             stats.swept, translate_queue->InFlight());
    stats = {};
    stats.window = now;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                               const ShaderInfo* shader,
                                                               std::span<const u64> keep_pointers) {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Compute", begin,
                              fmt::format("pipeline={:016x} s6:{:016x} address={:#x}", key.Hash(),
                                          key.unique_hash,
                                          kepler_compute->regs.code_loc.Address() +
                                              kepler_compute->launch_description.program_start));
    };
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    env.KeepStoragePointers(keep_pointers);

    main_pools.ReleaseContents();
    auto entry{CreateComputePipeline(main_pools, key, env, nullptr, true)};
    if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        directory && !entry.pipeline && env.CanBeSerialized()) {
        const auto path = std::filesystem::path{directory} /
                          fmt::format("failed-compute-{:016x}-v{}.env", key.unique_hash, CACHE_VERSION);
        std::ofstream file{path, std::ios::binary};
        if (file) {
            env.Serialize(file);
            LOG_INFO(Render_Vulkan, "Captured failed compute environment: {} written={}",
                     path.string(), file.good());
        }
    }
    if (!entry.pipeline || pipeline_cache_filename.empty()) {
        return entry;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
    return entry;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return {.failure = "Device reports broken compute support"};
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    if (const auto edits = Shader::Optimization::ApplyOotVisualWorkaround(
            program, program_id != 0 ? program_id : visual_workaround_title, key.unique_hash,
            oot_visual_workaround)) {
        LOG_WARNING(Render_Vulkan,
                    "TEMPORARY OoT visual workaround: shader {:016x}, {} scalar edits",
                    key.unique_hash, *edits);
    }
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY || driver_id == VK_DRIVER_ID_MOLTENVK;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    std::vector<u32> code{EmitSPIRV(profile, program)};
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    const Shader::Info info{std::move(program.info)};
    if (!thread_worker) {
        // The driver compiles on this thread. Precache and boot workers would otherwise hold their
        // IR through the compile, up to hundreds of MiB for one large shader.
        program = {};
        code = std::vector<u32>{};
        pools.block.Free();
        pools.inst.Free();
    }
    return {.pipeline = std::make_unique<ComputePipeline>(
                device, scheduler, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue,
                descriptor_buffer_ring, thread_worker, statistics, &shader_notify, info,
                std::move(spv_module), key.unique_hash),
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Compute pipeline compilation failed shader={:#016x} pipeline={:#016x}: {}",
              key.unique_hash, key.Hash(), exception.what());
    return {.failure = exception.what()};
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) try {
    size_t cache_size = 0;
    if (pipeline_cache) {
        pipeline_cache.Read(&cache_size, nullptr);
    }
    std::error_code exists_ec;
    if (cache_size > 0 && cache_size == last_cache_size.load(std::memory_order_relaxed) &&
        std::filesystem::exists(filename, exists_ec)) {
        return;
    }
    // Written beside the cache and renamed over it: the emulator stopped mid-write left the cache
    // empty, and every later boot failed to read it.
    auto temporary = filename;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
        return;
    }
    file.exceptions(std::ifstream::failbit);
    file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
        .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

    std::vector<char> cache_data;
    if (pipeline_cache && cache_size > 0) {
        cache_data.resize(cache_size);
        pipeline_cache.Read(&cache_size, cache_data.data());
    }
    Common::NoteShutdownProgress();
    // In pieces: a cache of tens of thousands of pipelines is gigabytes, and a shutdown waiting
    // on it is told the write still moves.
    constexpr size_t Chunk = size_t{64} << 20;
    for (size_t offset = 0; offset < cache_size; offset += Chunk) {
        file.write(cache_data.data() + offset,
                   static_cast<std::streamsize>((std::min)(Chunk, cache_size - offset)));
        Common::NoteShutdownProgress();
    }
    file.close();
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), error.message());
        std::filesystem::remove(temporary, error);
        return;
    }
    last_cache_size.store(cache_size, std::memory_order_relaxed);

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {} ({} KiB)",
             Common::FS::PathToUTF8String(filename), cache_size >> 10);

} catch (const std::ios_base::failure& e) {
    // Only the half-written copy goes: the cache on disk is still whole (deleting it on a failed
    // write, e.g. a full disk, made the next boot compile every pipeline again).
    LOG_ERROR(Common_Filesystem, "Writing the Vulkan driver pipeline cache failed: {}", e.what());
    auto temporary = filename;
    temporary += ".tmp";
    std::error_code error;
    std::filesystem::remove(temporary, error);
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        // A file cut short (the emulator stopped while writing it) would throw from the reads
        // below, and macOS's libc++ throws an ios_base::failure the handler doesn't match.
        if (static_cast<size_t>(end) < magic_number.size() + sizeof(cache_version)) {
            file.close();
            LOG_ERROR(Common_Filesystem, "Truncated Vulkan driver pipeline cache file, deleting it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan, "Loaded Vulkan driver pipeline cache: {} ({} KiB)",
                 Common::FS::PathToUTF8String(filename), cache_size >> 10);

        try {
            auto loaded = create_pipeline_cache(cache_size, cache_data.data());
            size_t driver_size = 0;
            if (loaded && loaded.Read(&driver_size, nullptr) == VK_SUCCESS) {
                last_cache_size.store(driver_size, std::memory_order_relaxed);
            }
            return loaded;
        } catch (const vk::Exception& e) {
            // The blob came from another driver or driver version (e.g. switching between
            // MoltenVK and KosmicKrisp on macOS): the driver rejects it. Start empty instead of
            // aborting, and drop the stale file so the next run doesn't retry it.
            LOG_ERROR(Render_Vulkan,
                      "Driver rejected the Vulkan pipeline cache file ({}), starting with an empty "
                      "cache and deleting it",
                      e.what());
            file.close();
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
