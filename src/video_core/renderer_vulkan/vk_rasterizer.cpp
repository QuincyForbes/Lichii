// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <map>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <tuple>
#include <set>
#include <span>
#include <fstream>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/assert.h"
#include "common/dkb_frame_trace.h"
#include "common/frame_deadline.h"
#include "common/host_memory.h"
#include "common/perf_sched.h"
#include "common/platform_events.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu_split.h"
#include "video_core/host_budget.h"
#include "video_core/native_query_trace.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/render_probe.h"
#include "video_core/render_trace.h"
#include "video_core/indirect_reset.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/feature_census.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/framebuffer_clear_layers.h"
#include "video_core/renderer_vulkan/framebuffer_limits.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_fsr3_upscaler.h"
#ifdef LICHII_METALFX
#include "video_core/renderer_vulkan/metal_gpu_time.h"
#include "video_core/renderer_vulkan/present/metalfx_temporal.h"
#endif
#ifdef EGLANTINE_STREAMLINE
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#endif
#include "video_core/renderer_vulkan/submission_policy.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/nx2_draw_stats.h"
#include "video_core/nx2_spikes.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_diag.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {
/// EDEN_NX2_DIAG_SAMPLES: the occlusion queries that count the samples each captured draw passes.
vk::QueryPool* g_diag_pool{};

// EXPERIMENT EDEN_NX2_INVALIDATE_BEFORE_DISPATCH=1: apply collected guest CPU writes before every
// dispatch and draw (=list does it once per command list, in DmaPusher).
bool InvalidateBeforeUse() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INVALIDATE_BEFORE_DISPATCH");
        return value != nullptr && value[0] != '0' && std::string_view{value} != "list";
    }();
    return enabled;
}

/// read_gpu_memory: the first address space (the bound channel's first) that maps the whole range,
/// read as the GPU reads it, without flushing GPU caches.
VideoCore::Nx2Debug::GpuReadResult ReadGpuMemory(std::span<Tegra::MemoryManager* const> spaces,
                                                 GPUVAddr address, u64 size) {
    VideoCore::Nx2Debug::GpuReadResult result{};
    if (spaces.empty()) {
        result.error = "no GPU address space is bound yet";
        return result;
    }
    for (Tegra::MemoryManager* const space : spaces) {
        if (space->IsFullyMappedRange(address, size)) {
            result.bytes.resize(size);
            space->ReadBlockUnsafe(address, result.bytes.data(), size);
            result.address_space = space->GetID();
            return result;
        }
    }
    constexpr u64 page_mask = 0xfff;
    const GPUVAddr end = address + size;
    GPUVAddr first = address;
    while (first < end) {
        const GPUVAddr next = (std::min)((first | page_mask) + 1, end);
        if (!spaces.front()->IsFullyMappedRange(first, next - first)) {
            break;
        }
        first = next;
    }
    result.error = fmt::format("GPU address {:#x} is not mapped (the first unmapped byte in the "
                               "bound channel's address space; {} address spaces checked)",
                               first, spaces.size());
    return result;
}

struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}

/// Before a dispatch: earlier writes visible to it, its indirect arguments and WAW/WAR included.
void RecordDispatchBarrier(vk::CommandBuffer cmdbuf) {
    static constexpr VkMemoryBarrier READ_WRITE_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT |
                         VK_ACCESS_MEMORY_WRITE_BIT,
    };
    cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, READ_WRITE_BARRIER);
}

/// EDEN_NX2_DISPATCH_CHAIN: a guest dispatch recorded right after another one with only bindings
/// between them (Scheduler::FollowsDispatch) records no barrier. Valid on MoltenVK only: both
/// land in one serial Metal compute encoder, which runs its dispatches in order with the writes of
/// each visible to the next, while any barrier ends that encoder (argument buffers with a
/// residency set, MVKCmdPipelineBarrier::encode). =1 on MoltenVK, =force on every driver (tests
/// only, not valid Vulkan), unset or 0 off.
bool DispatchChainEnabled(const Device& device) {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_DISPATCH_CHAIN");
        if (value == nullptr || value[0] == '0') {
            return 0;
        }
        return std::string_view{value} == "force" ? 2 : 1;
    }();
    return mode == 2 || (mode == 1 && device.IsMoltenVK());
}

/// EDEN_NX2_MVK_POST_DISPATCH_BARRIER: on MoltenVK, Scheduler flushes compute_writes_pending with
/// a full barrier when leaving compute (BeginRenderPass, SplitBeforeNextPass, and outside-pass
/// operations), while consecutive hazard-free dispatches share one Metal compute encoder through
/// ComputeDispatchChain. =every or =force emits a barrier after every individual dispatch (A/B);
/// =0 disables the extra MoltenVK post-compute barrier.
bool ImmediatePostDispatchBarrierEnabled(const Device& device) {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_MVK_POST_DISPATCH_BARRIER");
        if (value == nullptr) {
            return 1;
        }
        if (value[0] == '0') {
            return 0;
        }
        const std::string_view sv{value};
        if (sv == "force") {
            return 3;
        }
        if (sv == "every") {
            return 2;
        }
        return 1;
    }();
    const bool immediate = mode == 3 || (mode == 2 && device.IsMoltenVK());
    const bool deferred = mode == 1 && device.IsMoltenVK();
    static const bool logged = [immediate, deferred] {
        LOG_INFO(Render_Vulkan,
                 "Post-dispatch compute barrier (EDEN_NX2_MVK_POST_DISPATCH_BARRIER) {}",
                 immediate ? "every dispatch" : (deferred ? "deferred" : "off"));
        return true;
    }();
    (void)logged;
    return immediate;
}

bool DispatchBeforePassEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_DISPATCH_BEFORE_PASS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

void RecordPostDispatchBarrier(vk::CommandBuffer cmdbuf, bool mesh_supported = false) {
    static constexpr VkMemoryBarrier POST_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    // ALL_COMMANDS plus the explicit stages, so a translation that maps only individual
    // stage bits still waits in the vertex, fragment, indirect and transfer stages.
    VkPipelineStageFlags dst_stages =
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (mesh_supported) {
        dst_stages |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
    }
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                           dst_stages, 0, POST_BARRIER);
}

bool DynamicStateCacheEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_DYNSTATE_CACHE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

bool VertexInputCacheEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_VERTEX_INPUT_CACHE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

template <typename T>
void AppendWords(u32*& out, const T& value) {
    static_assert(sizeof(T) % sizeof(u32) == 0);
    std::memcpy(out, &value, sizeof(T));
    out += sizeof(T) / sizeof(u32);
}

bool HasAppliedOcarinaPerformanceScale() {
    return Settings::AppliedOcarinaRenderScale() == Settings::DlssQuality::Performance;
}

template <typename Params>
constexpr bool IsIndirectFromRecord(const Params& p) noexcept {
    if constexpr (requires { p.from_record; }) {
        return p.from_record;
    } else {
        return true;
    }
}

template <typename CounterEnum = VideoCommon::Nx2Profile::Counter>
void CountIndirectRecordUnordered() noexcept {
    if constexpr (requires { CounterEnum::IndirectRecordsUnordered; }) {
        VideoCommon::Nx2Profile::Add(CounterEnum::IndirectRecordsUnordered, 1);
    }
}
} // Anonymous namespace

VideoCommon::Extent2D TargetlessDrawReach(const Maxwell& regs, bool all_viewports,
                                          bool is_rescaling) {
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (is_rescaling) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const VideoCommon::Extent2D size{
        (regs.surface_clip.width * up_scale) >> down_shift,
        (regs.surface_clip.height * up_scale) >> down_shift,
    };
    if (!regs.viewport_scale_offset_enabled) {
        // The scissor is the surface clip then (UpdateScissorsState)
        return size;
    }
    VideoCommon::Extent2D reach{};
    const size_t num_scissors = all_viewports ? Maxwell::NumViewports : 1;
    for (size_t index = 0; index < num_scissors; ++index) {
        const VkRect2D scissor = GetScissorState(regs, index, up_scale, down_shift);
        reach.width = (std::max)(
            reach.width, ScissorReach(scissor.offset.x, scissor.extent.width, size.width));
        reach.height = (std::max)(
            reach.height, ScissorReach(scissor.offset.y, scissor.extent.height, size.height));
    }
    return reach;
}

void RasterizerVulkan::ReadMeshEmulationProbe() {
    const MeshEmulationProbe probe{mesh_probe};
    mesh_probe.pending = false;
    const size_t num_words{probe.data.size() / sizeof(u32)};
    const auto word = [&](u64 index) -> u32 {
        if (index >= num_words) {
            return 0;
        }
        u32 value;
        std::memcpy(&value, probe.data.data() + index * sizeof(u32), sizeof(value));
        return value;
    };
    const auto position = [&](u64 base, u32 vertex) {
        std::array<f32, 4> result{};
        for (u32 element = 0; element < 4; ++element) {
            const u32 line{probe.position_lines[element]};
            if (line != ~0U) {
                result[element] = std::bit_cast<f32>(word(base + u64{vertex} * probe.vertex_lines +
                                                          line));
            }
        }
        return result;
    };
    u64 triangles{}, degenerate{}, outside{}, bad_index{};
    std::string sample;
    const u32 corners{probe.primitive_vertices};
    for (u32 group = 0; group < probe.groups; ++group) {
        const u64 base{u64{group} * probe.block_words};
        const u64 index_base{base + u64{probe.max_vertices} * probe.vertex_lines};
        for (u32 primitive = 0; primitive < probe.max_primitives; ++primitive) {
            std::array<u32, 3> storage{};
            for (u32 corner = 0; corner < corners; ++corner) {
                storage[corner] = word(index_base + u64{primitive} * corners + corner);
            }
            const std::span<const u32> indices{storage.data(), corners};
            // Slots past the primitive count: 0 0 0 for triangles, the unused marker otherwise.
            const bool unused{
                corners == 3
                    ? indices[0] == indices[1] && indices[1] == indices[2]
                    : indices[0] == Shader::Backend::SPIRV::MESH_EMULATION_UNUSED_VERTEX};
            if (unused) {
                ++degenerate;
                continue;
            }
            ++triangles;
            bool any_bad{};
            bool any_inside{};
            for (const u32 vertex : indices) {
                if (vertex >= probe.max_vertices) {
                    any_bad = true;
                    continue;
                }
                const auto p{position(base, vertex)};
                const f32 w{std::abs(p[3])};
                any_inside |= w > 0.0f && std::abs(p[0]) <= w && std::abs(p[1]) <= w;
            }
            bad_index += any_bad;
            outside += !any_inside;
            if (sample.size() < 400 && triangles <= 2) {
                for (const u32 vertex : indices) {
                    const auto p{position(base, std::min(vertex, probe.max_vertices - 1))};
                    sample += fmt::format(" g{} v{} ({:.3g} {:.3g} {:.3g} {:.3g})", group, vertex,
                                          p[0], p[1], p[2], p[3]);
                }
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "Mesh emulation probe frame {}: {} groups, {} primitives of {} vertices ({} unused "
             "slots), {} with every vertex off screen, {} with a bad index; first primitives:{}",
             probe.frame, probe.groups, triangles, corners, degenerate, outside, bad_index, sample);
}

const RasterizerVulkan::MeshArena& RasterizerVulkan::AcquireMeshArena() {
    const u64 tick{scheduler.CurrentTick()};
    if (mesh_arena_current < mesh_arenas.size() && mesh_arenas[mesh_arena_current].tick == tick) {
        return mesh_arenas[mesh_arena_current];
    }
    // At most MaxMeshArenas submissions in flight hold one: past that, wait for the oldest rather
    // than make another (early staging submits during a scene load made one per submission).
    // Two more with over 1.5 GiB left to the process (an iPhone in Kokiri with 2 GiB left waited
    // for one 16 times in 3600 submissions).
    constexpr size_t MaxMeshArenas = 4;
    const auto busy = [this] {
        return std::ranges::none_of(mesh_arenas, [this](const MeshArena& arena) {
            return arena.counted || scheduler.IsFree(arena.tick);
        });
    };
    if (mesh_arenas.size() >= MaxMeshArenas && busy() &&
        (mesh_arenas.size() >= MaxMeshArenas + 2 ||
         Common::PlatformEvents::AvailableProcessMemory() < (u64{3} << 29) + mesh_arena_size)) {
        const auto oldest{std::ranges::min_element(mesh_arenas, {}, &MeshArena::tick)};
        scheduler.Wait(oldest->tick);
        ++mesh_arena_waits;
        Common::PerfReport::Add(Common::PerfReport::MeshArenaWaits);
    }
    // A finished submission's counter holds the words its draws asked for, including those
    // that did not fit: grow later arenas to hold them.
    const u64 max_size{(std::min)(u64{1} << 30, u64{device.GetMaxStorageBufferRange()})};
    for (MeshArena& arena : mesh_arenas) {
        if (arena.counted || !scheduler.IsFree(arena.tick)) {
            continue;
        }
        arena.counted = true;
        arena.counter.Invalidate();
        u32 words;
        std::memcpy(&words, arena.counter.Mapped().data(), sizeof(words));
        const u64 needed{u64{words} * 4};
        mesh_need_peak = (std::max)(mesh_need_peak, needed);
        if (++mesh_need_counted % 3600 == 0) {
            LOG_INFO(Common_Memory,
                     "Mesh emulation arenas: {} of {} MiB; the last 3600 submissions asked for at "
                     "most {:.1f} MiB; {} waits for a submission's arena",
                     mesh_arenas.size(), mesh_arena_size >> 20,
                     static_cast<double>(mesh_need_peak) / (1 << 20), mesh_arena_waits);
            mesh_need_previous_peak = mesh_need_peak;
            mesh_need_peak = 0;
            mesh_arena_waits = 0;
        }
        if (needed > arena.size && mesh_arena_size < max_size) {
            mesh_arena_size = (std::min)(std::bit_ceil(needed + needed / 4), max_size);
            LOG_INFO(Common_Memory, "Mesh emulation: a submission needed {} MiB of output, arenas "
                                    "grow to {} MiB",
                     needed >> 20, mesh_arena_size >> 20);
        }
    }
    // Host memory pressure: arenas shrink to what recent submissions asked for, with half of it
    // again as margin (they grow back from the counter if a submission needs more).
    if (const u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
        pressure != mesh_arena_pressure) {
        mesh_arena_pressure = pressure;
        constexpr u64 MinArenaSize = 8ULL << 20;
        const u64 recent{(std::max)(mesh_need_peak, mesh_need_previous_peak)};
        const u64 target{(std::max)(std::bit_ceil(recent + recent / 2), MinArenaSize)};
        if (target < mesh_arena_size) {
            LOG_INFO(Common_Memory,
                     "Mesh emulation: host memory pressure, arenas shrink from {} to {} MiB",
                     mesh_arena_size >> 20, target >> 20);
            mesh_arena_size = target;
        }
    }
    std::erase_if(mesh_arenas, [&](const MeshArena& arena) {
        return arena.counted && arena.size != mesh_arena_size;
    });
    auto it{std::ranges::find_if(mesh_arenas, [](const MeshArena& arena) { return arena.counted; })};
    if (it == mesh_arenas.end()) {
        const MemoryTagScope memory_tag{MemoryTag::MeshEmulation};
        const auto make_buffer = [&](u64 size, VkBufferUsageFlags usage, MemoryUsage memory) {
            return memory_allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = size,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                memory);
        };
        it = mesh_arenas.insert(
            mesh_arenas.end(),
            MeshArena{
                .buffer = make_buffer(mesh_arena_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      MemoryUsage::DeviceLocal),
                .counter = make_buffer(sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       MemoryUsage::Download),
                .size = mesh_arena_size,
            });
    }
    // The GPU is done with this arena, and this submission's commands have not been sent yet.
    const u32 zero{};
    std::memcpy(it->counter.Mapped().data(), &zero, sizeof(zero));
    it->counter.Flush();
    it->tick = tick;
    it->counted = false;
    mesh_arena_current = static_cast<size_t>(std::distance(mesh_arenas.begin(), it));
    return *it;
}

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler), descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler, staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      frame_dump(device, memory_allocator, scheduler),
      flash_watch(device, memory_allocator, scheduler) {
    scheduler.SetQueryCache(query_cache);
    VideoCore::Probe::StoreScene(0, 0, 0, 0, nullptr);
#ifdef LICHII_METALFX
    if (Common::PerfReport::Enabled()) {
        MetalGpuTime::Start();
    }
#endif
    buffer_cache.SetPendingCpuWrites(
        [this](const std::function<void(DAddr, size_t)>& callback) {
            gpu.ForEachPendingCpuWrite(callback);
        });
    VideoCommon::GpuSplit::Get().SetIdleHook(
        [](void* argument, bool requested) {
            return static_cast<RasterizerVulkan*>(argument)->FlushSplitIdle(requested);
        },
        [](void* argument) {
            return static_cast<RasterizerVulkan*>(argument)->scheduler.DeferredFlushMayWait();
        },
        this);
    VideoCommon::GpuSplit::Get().SetIdleBusy(
        [](void* argument) { return static_cast<RasterizerVulkan*>(argument)->scheduler.GpuBehind(); });
        VideoCommon::GpuSplit::Get().SetFenceQuietCheck(
            [](void* argument, bool strict, u64* epoch) {
                return static_cast<RasterizerVulkan*>(argument)->FencesQuiet(strict, *epoch);
            },
            this);
        buffer_cache.mutex.SetOwnerHook(
            [](void* argument) {
                auto& self = *static_cast<RasterizerVulkan*>(argument);
                self.write_handoff.Drain([&self](DAddr addr, u64 size) {
                    if (!self.buffer_cache.OnCPUWrite(addr, size)) {
                        return;
                    }
                    constexpr u64 page_size = VideoCommon::Nx2WriteHandoff::PageSize;
                    for (DAddr page = addr; page < addr + size;) {
                        const DAddr page_end =
                            (std::min)(addr + size, Common::AlignDown(page, page_size) + page_size);
                        if (self.buffer_cache.OnCPUWrite(page, page_end - page)) {
                            self.gpu.CollectHandoffWrite(page, page_end - page);
                        }
                        page = page_end;
                    }
                });
            },
            this);
    Common::PlatformEvents::SetGraphicsMemoryReport([this] { return GraphicsMemoryReport(); });
    submit_while_waited =
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::SubmitWhileWaited);
    if (submit_while_waited) {
        const auto draws = [](const char* name, u32 fallback) {
            const char* value = std::getenv(name);
            if (value == nullptr || value[0] < '0' || value[0] > '9') {
                return fallback;
            }
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);
            return *end == '\0' ? static_cast<u32>(std::clamp(parsed, 1UL, 1'048'576UL))
                                : fallback;
        };
        submit_waited_draws = draws("EDEN_NX2_SUBMIT_WAITED_DRAWS", 64);
        submit_idle_draws = draws("EDEN_NX2_SUBMIT_IDLE_DRAWS", 128);
        LOG_INFO(Render_Vulkan, "Nx2Perf submit_while_waited: {} draws while the front waits, {} "
                                "once the GPU is idle",
                 submit_waited_draws, submit_idle_draws);
    }
    oot_performance_scale = HasAppliedOcarinaPerformanceScale();
}

RasterizerVulkan::~RasterizerVulkan() {
    delete std::exchange(g_diag_pool, nullptr);
    Common::PlatformEvents::SetGraphicsMemoryReport({});
    VideoCommon::GpuSplit::Get().SetIdleHook(nullptr, nullptr, nullptr);
    VideoCommon::GpuSplit::Get().SetIdleBusy(nullptr);
    VideoCommon::GpuSplit::Get().SetFenceQuietCheck(nullptr, nullptr);
    scheduler.WaitWorker();
    scheduler.Finish();
}

void RasterizerVulkan::Nx2ProveNonWriterDraw(const GraphicsPipeline& pipeline) {
    // The programs as RefreshStages resolved them, in the epoch read before it did: a program
    // dropped or rewritten since moved the epoch on, and the proof is stale from the start.
    const u64 epoch = pipeline_cache.Nx2ResolvedEpoch();
    const u64 map = pipeline_cache.Nx2ResolvedMap();
    if (epoch == 0 || map != gpu_memory->GetMapSerial()) {
        return;
    }
    Tegra::Nx2DrawPrograms programs;
    if (!Tegra::Engines::Maxwell3D::Nx2DrawProgramsOf(
            maxwell3d->regs, maxwell3d->GetNativeMacroEngine().native_turing, gpu_memory->GetID(),
            map, programs) ||
        Tegra::Nx2IsNonWriterDraw(programs, epoch)) {
        return;
    }
    // The indirect reset program is drawn by writing guest memory on the host.
    if (pipeline.IsMeshPipeline() || pipeline.Nx2MayWriteMemory() ||
        pipeline_cache.IsIndirectResetShader()) {
        return;
    }
    const auto& ranges = pipeline_cache.Nx2CodeRanges();
    for (const auto& [address, size] : ranges) {
        // A CPU write to GPU-modified buffer pages reaches the shader cache only later
        // (OnCPUWrite): code there could change under a proof without moving the epoch in time.
        if (size != 0 && buffer_cache.IsRegionGpuModified(address, size)) {
            return;
        }
    }
    for (const auto& [address, size] : ranges) {
        Tegra::Nx2MarkProvenCode(address, size);
    }
    // An invalidation on another thread meanwhile (CPU writes) makes the proof stale anyway.
    if (Tegra::g_nx2_code_epoch.load(std::memory_order_acquire) != epoch) {
        return;
    }
    Tegra::Nx2InsertNonWriterDraw(programs, epoch);
}

// Experimental adaptation of PokeDinner commit 76df4591a747: execute the reset shader's
// indirect record writes on the host until general GPU-address translation is available.
bool RasterizerVulkan::IsNx2IndirectReset(bool is_indexed, u32 instance_count) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RESETFIX");
        return !value || std::string_view{value} != "0";
    }();
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    if (!enabled || is_indexed || instance_count != 1 || regs.IsMeshEnabled() ||
        draw.topology != Maxwell::PrimitiveTopology::Points ||
        !maxwell3d->GetNativeMacroEngine().native_turing) {
        return false;
    }
    const GPUVAddr program =
        (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    std::array<u32, VideoCommon::NativeIndirectReset::Program.size()> code{};
    if (!gpu_memory->IsFullyMappedRange(program + 128, sizeof(code))) {
        return false;
    }
    gpu_memory->ReadBlock(program + 128, code.data(), sizeof(code));
    return VideoCommon::NativeIndirectReset::Matches(code);
}

bool RasterizerVulkan::TryNx2IndirectReset(bool is_indexed, u32 instance_count) {
    if (!IsNx2IndirectReset(is_indexed, instance_count)) {
        return false;
    }
    const auto& draw = maxwell3d->draw_manager.draw_state;
    FlushWork();
    gpu_memory->FlushCaching();
    const auto reject = [](std::string_view reason) {
        static u32 reported_rejections{};
        if (reported_rejections++ < 8) {
            LOG_WARNING(HW_GPU, "Experimental indirect reset skipped: {}", reason);
        }
        return false;
    };
    const auto in_gpu_range = [this](u64 address, size_t size) {
        return size != 0 && address <= (std::numeric_limits<u64>::max)() - size &&
               gpu_memory->IsWithinGPUAddressRange(address) &&
               gpu_memory->IsWithinGPUAddressRange(address + size - 1);
    };
    const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[0];
    if (!cbuf.enabled || cbuf.size < 0x120 || !in_gpu_range(cbuf.address, 0x120) ||
        !gpu_memory->IsFullyMappedRange(cbuf.address + 0x110, 16)) {
        return reject("descriptor unavailable");
    }
    std::array<u64, 2> descriptor{};
    gpu_memory->ReadBlock(cbuf.address + 0x110, descriptor.data(), sizeof(descriptor));
    const u64 list = descriptor[0];
    const u64 first = draw.vertex_buffer.first;
    const u64 count = draw.vertex_buffer.count;
    const auto range = VideoCommon::IndirectResetRange(list, static_cast<u32>(descriptor[1]),
                                                        first, count);
    if (!range || !in_gpu_range(range->begin, range->size) ||
        !gpu_memory->IsFullyMappedRange(range->begin, range->size)) {
        return reject("list outside mapped descriptor");
    }
    std::vector<u64> pointers(count);
    gpu_memory->ReadBlock(range->begin, pointers.data(), range->size);
    static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
    boost::container::small_vector<VideoCommon::Nx2ResetRun, 16> runs;

    for (const u64 pointer : pointers) {
        if (!pointer) {
            return reject("record is null");
        }
        if (!in_gpu_range(pointer, sizeof(record)) ||
            !gpu_memory->IsFullyMappedRange(pointer, sizeof(record))) {
            return reject("record not mapped");
        }
        if (pointer < range->begin + range->size && range->begin < pointer + sizeof(record)) {
            return reject("record overlaps pointer list");
        }
        const auto segments = gpu_memory->GetSubmappedRange(pointer, sizeof(record));
        if (segments.size() != 1 || segments[0].first != pointer ||
            segments[0].second != sizeof(record)) {
            return reject("record crosses a mapping boundary");
        }
        const auto cpu_addr = gpu_memory->GpuToCpuAddress(pointer);
        if (!cpu_addr) {
            return reject("record not mapped to CPU address");
        }
        if (!runs.empty()) {
            auto& last = runs.back();
            const u64 expected_gpu = last.gpu_addr + last.count * sizeof(record);
            const u64 expected_cpu = last.cpu_addr + last.count * sizeof(record);
            if (pointer == expected_gpu && *cpu_addr == expected_cpu) {
                last.count++;
                continue;
            }
        }
        runs.push_back(VideoCommon::Nx2ResetRun{
            .gpu_addr = pointer,
            .cpu_addr = *cpu_addr,
            .count = 1,
        });
    }
    const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
    for (const u64 pointer : pointers) {
        AccelerateInlineToMemory(pointer, bytes.size(), bytes);
    }
    static u32 reported{};
    if (reported++ < 5) {
        LOG_INFO(HW_GPU, "Experimental indirect reset: {} records", count);
    }
    gpu.TickWork();
    return true;
}

namespace {
// EDEN_NX2_DIAG*: what the draw being prepared is (set by Draw/DrawIndirect, read by PrepareDraw).
struct DiagDraw {
    const char* kind{"draw"};
    u32 vertices{};
    u32 instances{};
};
thread_local DiagDraw g_diag_draw;
thread_local int g_diag_query{-1};

u64 DiagShader(const Tegra::Engines::Maxwell3D& maxwell3d, size_t stage) {
    const auto& regs = maxwell3d.regs;
    return (u64{regs.pipelines[stage].reserved[0] & 0xff} << 32) | regs.pipelines[stage].reserved[1];
}

// diag_draw_lite (EDEN_NX2_DIAG only): the back's draw bookkeeping without a lock per draw.
bool DiagLite() {
    static const bool on = VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::DiagDrawLite);
    return on;
}

// Verify only, back thread: differences found and the error lines left to log.
u64 g_lite_differences{};
u32 g_lite_lines{64};

VideoCommon::Nx2Diag::LiteTable& DiagLiteTable() {
    auto& table = VideoCommon::Nx2Diag::ThreadLite();
    if (!table) {
        table = std::make_unique<VideoCommon::Nx2Diag::LiteTable>();
        VideoCommon::Nx2Diag::Diagnostics::Get().EnableLite();
    }
    return *table;
}

// The cached answers for a draw's programs. Verify: the synchronous answers, compared with the
// cached ones when no generation change came between.
VideoCommon::Nx2Diag::LiteAnswer DiagLiteAnswer(VideoCommon::Nx2Diag::Diagnostics& diag,
                                                VideoCommon::Nx2Diag::LiteTable& table, u64 vs,
                                                u64 ps) {
    const auto& cached = diag.LiteLookup(table, vs, ps);
    if (!VideoCommon::Nx2Perf::Verify()) {
        return cached;
    }
    // Polled like the synchronous Ablated/Inspecting calls.
    const auto fresh = diag.Resolve(vs, ps, true);
    if (fresh.generation == cached.generation &&
        (fresh.vkey != cached.vkey || fresh.pkey != cached.pkey ||
         fresh.ablated_v != cached.ablated_v || fresh.ablated_p != cached.ablated_p ||
         fresh.inspect_v != cached.inspect_v || fresh.inspect_p != cached.inspect_p)) {
        ++g_lite_differences;
        if (g_lite_lines > 0) {
            --g_lite_lines;
            LOG_ERROR(HW_GPU, "Nx2Perf diag_draw_lite: answer differs for vs={:#x} ps={:#x}", vs, ps);
        }
    }
    return fresh;
}

// Verify, after the present's fold: the folded state against the shadow's synchronous one.
void DiagLiteVerifyFrame() {
    auto& diag = VideoCommon::Nx2Diag::Diagnostics::Get();
    if (const auto& table = VideoCommon::Nx2Diag::ThreadLite(); table && table->used != 0) {
        diag.FoldLite(*table);
    }
    g_lite_differences += diag.DrawDifferences(VideoCommon::Nx2Diag::Diagnostics::Shadow(), g_lite_lines);
    static std::chrono::steady_clock::time_point last_log{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log >= std::chrono::seconds{1}) {
        last_log = now;
        if (g_lite_differences != 0) {
            LOG_ERROR(HW_GPU, "Nx2Perf diag_draw_lite: {} differences", g_lite_differences);
        } else {
            LOG_INFO(HW_GPU, "Nx2Perf diag_draw_lite: 0 differences");
        }
    }
}

bool DiagAblated(const Tegra::Engines::Maxwell3D& maxwell3d) {
    auto& diag = VideoCommon::Nx2Diag::Diagnostics::Get();
    if (DiagLite()) {
        const auto answer = DiagLiteAnswer(diag, DiagLiteTable(), DiagShader(maxwell3d, 1),
                                           DiagShader(maxwell3d, 5));
        return answer.ablated_v || answer.ablated_p;
    }
    return diag.Ablated('v', DiagShader(maxwell3d, 1)) || diag.Ablated('p', DiagShader(maxwell3d, 5));
}

// Shows the constant buffers bound to a stage to the inspector (inspect.txt).
void DiagObserveStage(Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& gpu_memory,
                      size_t stage, u64 key) {
    auto& diag = VideoCommon::Nx2Diag::Diagnostics::Get();
    const auto& info = maxwell3d.state.shader_stages[stage];
    for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
        const auto& buffer = info.const_buffers[slot];
        if (!buffer.enabled || buffer.size == 0) {
            continue;
        }
        std::vector<u32> words(std::min<u32>(buffer.size, 0x2000) / 4);
        if (words.empty() || !gpu_memory.IsFullyMappedRange(buffer.address, words.size() * 4)) {
            continue;
        }
        gpu_memory.ReadBlockUnsafe(buffer.address, words.data(), words.size() * 4);
        diag.Observe(key, slot, words);
    }
}

void DiagRecordDraw(Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& gpu_memory,
                    const char* outcome, bool mesh) {
    auto& diag = VideoCommon::Nx2Diag::Diagnostics::Get();
    const u64 vs = DiagShader(maxwell3d, 1);
    const u64 ps = DiagShader(maxwell3d, 5);
    if (DiagLite()) {
        auto& table = DiagLiteTable();
        const auto answer = DiagLiteAnswer(diag, table, vs, ps);
        const auto& rt = maxwell3d.regs.rt[0];
        if (VideoCommon::Nx2Perf::Verify()) {
            VideoCommon::Nx2Diag::Diagnostics::Shadow().ShadowDraw(
                diag.CurrentFrame(), answer, outcome, g_diag_draw.instances, g_diag_draw.vertices,
                rt.width, rt.height, VideoCommon::Nx2Diag::CountersEnabled());
        }
        if (!diag.Capturing()) {
            diag.LiteAdd(table, answer, outcome, g_diag_draw.instances, g_diag_draw.vertices,
                         rt.width, rt.height);
            if (answer.inspect_v) {
                DiagObserveStage(maxwell3d, gpu_memory, 0, answer.vkey);
            }
            if (answer.inspect_p) {
                DiagObserveStage(maxwell3d, gpu_memory, 4, answer.pkey);
            }
            g_diag_query = -1;
            return;
        }
        // A capture: the synchronous path, after this thread's earlier draws.
        if (table.used != 0) {
            diag.FoldLite(table);
        }
    }
    {
        const auto& rt = maxwell3d.regs.rt[0];
        diag.Touch('v', vs, outcome, g_diag_draw.instances, g_diag_draw.vertices, rt.width, rt.height);
        diag.Touch('p', ps, outcome, g_diag_draw.instances, g_diag_draw.vertices, rt.width, rt.height);
        if (const u64 key = diag.ProgramKey(vs); diag.Inspecting(key)) {
            DiagObserveStage(maxwell3d, gpu_memory, 0, key);
        }
        if (const u64 key = diag.ProgramKey(ps); diag.Inspecting(key)) {
            DiagObserveStage(maxwell3d, gpu_memory, 4, key);
        }
    }
    if (VideoCommon::Nx2Diag::CountersEnabled()) {
        diag.Note(false, vs, ps, outcome, g_diag_draw.instances, g_diag_draw.vertices);
    }
    if (!diag.Capturing()) {
        g_diag_query = -1;
    } else {
        const auto& regs = maxwell3d.regs;
        diag.Row(fmt::format(
            "{}\t{}\t{:#x}\t{:#x}\t{}\t{}\t{}\t{}\t{:#x}\t{}\t{}\t{:#x}\t{}\t{}\t{:#x}\t{}\t{:016x}\t{:016x}",
            diag.NextSeq(), g_diag_draw.kind, vs, ps, outcome, g_diag_draw.vertices,
            g_diag_draw.instances, static_cast<u32>(maxwell3d.draw_manager.draw_state.topology),
            regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
            static_cast<u32>(regs.rt[0].format), regs.zeta_enable, regs.blend.enable[0],
            regs.color_mask[0].raw, mesh ? "mesh" : "", diag.ProgramKey(vs), diag.ProgramKey(ps)),
            std::exchange(g_diag_query, -1), vs, ps);
    }
}
} // namespace

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func, bool is_indirect) {

    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_draws;
    }
    Common::PerfReport::Add(Common::PerfReport::Draws);

    SCOPE_EXIT {
        gpu.TickWork();
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            native.native_execution_failed = true;
            const auto& regs = maxwell3d->regs;
            const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                           regs.pipelines[1].reserved[1];
            const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                           regs.pipelines[5].reserved[1];
            LOG_DEBUG(HW_GPU, "Native graphics draw not recorded: frame={} vs={:#x} ps={:#x}",
                      render_trace_frame, vs, ps);
        }
    };
    namespace Profile = VideoCommon::Nx2Profile;
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }
    if (frame_dump.Capturing()) {
        frame_dump.DrainIfFull();
    }

    const char* diag_outcome = "drawn";
    const bool diag_active = VideoCommon::Nx2Diag::Active();
    SCOPE_EXIT {
        if (diag_active) {
            DiagRecordDraw(*maxwell3d, *gpu_memory, diag_outcome, mesh_draw);
        }
    };
    const auto report_failure = [&](std::string_view reason) {
        diag_outcome = reason == "pipeline still building" ? "async-build"
                       : reason == "no pipeline"           ? "no-pipeline"
                                                           : "skipped";
        TraceDrawFailure(reason);
        static std::set<std::tuple<u64, u64, std::string_view>> seen;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        if (frame_dump.Capturing()) {
            frame_dump.NoteSkipped(false, ps != 0 ? ps : vs, regs.rt[0].Address(), reason);
        }
        if (seen.size() < 64 && seen.emplace(vs, ps, reason).second) {
            LOG_WARNING(HW_GPU, "Draw skipped ({}): vs={:#x} ps={:#x} rt0={:#x} {}x{}", reason, vs, ps,
                        regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height);
        }
    };
    const u64 skipped_lookups = PipelineCache::SkippedLookups();
    GraphicsPipeline* const pipeline{[&] {
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
    if (VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteDraw(
            *maxwell3d, pipeline ? pipeline->ShaderHashes() : pipeline_cache.LookupHashes());
    }
    if (!pipeline) {
        Common::PerfReport::Add(Common::PerfReport::DrawsSkipped);
        // A pipeline still being built is skipped only into targets cleared every frame (the
        // draw comes back next frame); one that could not be made at all is skipped for good.
        report_failure(PipelineCache::SkippedLookups() != skipped_lookups
                           ? "pipeline still building"
                           : "no pipeline");
        TraceRenderState(is_indexed, "no-pipeline", nullptr);
        // Mesh emulation survey: how many task/mesh draws a device without
        // VK_EXT_mesh_shader drops, how many are indirect and how many groups they launch.
        if (maxwell3d->regs.IsMeshEnabled() && !device.IsExtMeshShaderSupported()) {
            static u64 draws{}, indirect{}, groups{}, max_groups{};
            ++draws;
            if (is_indirect) {
                ++indirect;
            } else {
                const u64 count{maxwell3d->draw_manager.draw_state.vertex_buffer.count};
                groups += count;
                max_groups = std::max(max_groups, count);
            }
            if (draws == 1 || draws % 5000 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "Mesh emulation survey: {} mesh draws dropped, {} indirect, {} groups "
                            "in direct ones (max {} per draw), frame {}",
                            draws, indirect, groups, max_groups, render_trace_frame);
            }
        }
        return;
    }
    if (VideoCommon::OotNetwork::BypassEnabled(Settings::GetCurrentProgramID())) {
        TrackOotTonemappedImage(*pipeline);
    }
    // Task/mesh pipelines: the draw's vertex count is the number of task workgroups.
    mesh_draw = pipeline->IsMeshPipeline();
    // EDEN_NX2_NO_MESH=1: skip task/mesh draws as before they were supported (A/B comparisons).
    // EDEN_NX2_NO_MESH_FILE=<path>: skip them while that file exists (checked every 1024 draws),
    // to compare the same scene with and without them in one run.
    static const bool no_mesh_always{std::getenv("EDEN_NX2_NO_MESH") != nullptr};
    static const char* const no_mesh_file{std::getenv("EDEN_NX2_NO_MESH_FILE")};
    static bool no_mesh{no_mesh_always};
    static u32 no_mesh_check{};
    if (no_mesh_file && (no_mesh_check++ % 1024) == 0) {
        no_mesh = std::filesystem::exists(no_mesh_file);
    }
    const bool emulated_mesh{mesh_draw && pipeline->IsEmulatedMesh()};
    if (mesh_draw &&
        (no_mesh || (!device.IsExtMeshShaderSupported() && !emulated_mesh) ||
         (emulated_mesh && is_indirect))) {
        report_failure(no_mesh         ? "EDEN_NX2_NO_MESH"
                       : emulated_mesh ? "indirect emulated mesh draw"
                                       : "no VK_EXT_mesh_shader");
        TraceRenderState(is_indexed, "mesh-skipped", nullptr);
        return;
    }
    // Mesh emulation: one output block per mesh workgroup, written by the pipeline's compute
    // dispatch and drawn as groups * vertices-per-group triangle list vertices.
    mesh_emulation_groups = 0;
    mesh_emulation_vertices_per_group = 0;
    if (emulated_mesh) {
        // The same group count the native path hands vkCmdDrawMeshTasksEXT (MakeDrawParams:
        // the index count for indexed draws), within the device's dispatch limit.
        const auto& draw_state{maxwell3d->draw_manager.draw_state};
        const u32 groups{(std::min)(MakeDrawParams(draw_state, 1, is_indexed).num_vertices,
                                    device.GetMaxComputeWorkGroupCount()[0])};
        // Mesh emulation statistics: draws with and without groups, and the frames with any
        // groups (the game culls on the GPU and sets the counts itself).
        {
            static u64 draws{}, empty{}, total_groups{}, frames{}, frames_drawn{};
            static u64 last_frame{~0ULL};
            static bool frame_drawn{};
            if (render_trace_frame != last_frame) {
                if (last_frame != ~0ULL) {
                    ++frames;
                    frames_drawn += frame_drawn;
                }
                last_frame = render_trace_frame;
                frame_drawn = false;
            }
            ++draws;
            empty += groups == 0;
            total_groups += groups;
            frame_drawn |= groups != 0;
            if (draws % 2000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "Mesh emulation draws: {} ({} with no groups), {} groups; frames with "
                         "emulated draws: {} of {}",
                         draws, empty, total_groups, frames_drawn, frames);
            }
        }
        // Read the pending probe before Request can hand its buffer out again.
        if (mesh_probe.pending && scheduler.IsFree(mesh_probe.tick)) {
            ReadMeshEmulationProbe();
        }
        // EDEN_NX2_MESH_EMULATION_PROBE=1: every 60 frames, read one draw's output back (from a
        // host-visible download buffer, as device-local ones are not mapped on every GPU).
        static const bool probe{std::getenv("EDEN_NX2_MESH_EMULATION_PROBE") != nullptr};
        const bool has_task{pipeline->MeshEmulationHasTask()};
        const bool probe_draw{probe && !has_task && !mesh_probe.pending && groups != 0 &&
                              render_trace_frame % 60 == 0};
        // With a task program the draw's groups are task workgroups. The mesh workgroups they ask
        // for are only known on the GPU, so their output goes to the submission's arena.
        const u64 block_bytes{pipeline->MeshEmulationBlockBytes()};
        StagingBufferRef output{};
        u64 bytes{};
        if (has_task) {
            const MeshArena& arena{AcquireMeshArena()};
            bytes = arena.size;
            pipeline->SetMeshEmulationOutput(*arena.buffer, 0, bytes, groups);
        } else {
            bytes = (std::max)(u64{groups} * block_bytes, u64{4});
            output = staging_pool.Request(bytes, probe_draw ? MemoryUsage::Download
                                                            : MemoryUsage::DeviceLocal);
            pipeline->SetMeshEmulationOutput(output.buffer, output.offset, bytes, groups);
        }
        mesh_emulation_indirect = {};
        if (mesh_task_probe.pending && scheduler.IsFree(mesh_task_probe.tick)) {
            mesh_task_probe.pending = false;
            std::array<u32, Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS> words{};
            std::memcpy(words.data(), mesh_task_probe.args.data(), sizeof(words));
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation task probe: {} tasks asked for {} mesh workgroups, {} "
                     "dispatched at arena word {} (arena {} words)",
                     mesh_task_probe.tasks,
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_REQUESTED], words[0],
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_BASE],
                     mesh_task_probe.capacity);
        }
        if (has_task && groups != 0) {
            const bool task_probe{probe && !mesh_task_probe.pending &&
                                  render_trace_frame % 60 == 0};
            const auto request = [&](u64 size, MemoryUsage usage = MemoryUsage::DeviceLocal) {
                size = (std::max)(size, u64{16});
                const StagingBufferRef ref{staging_pool.Request(size, usage)};
                return std::make_pair(GraphicsPipeline::MeshEmulationBuffer{ref.buffer, ref.offset,
                                                                            size},
                                      ref.mapped_span.subspan(
                                          (std::min)(ref.offset, u64{ref.mapped_span.size()})));
            };
            const auto payload{
                request(u64{groups} * pipeline->MeshEmulationPayloadStride() * 4).first};
            const auto offsets{request(u64{groups} * 4).first};
            constexpr u64 args_bytes{Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS * 4};
            const auto [args, args_span] =
                request(args_bytes, task_probe ? MemoryUsage::Download : MemoryUsage::DeviceLocal);
            const u32 capacity_words{static_cast<u32>(
                (std::min)(bytes / 4, u64{(std::numeric_limits<u32>::max)()}))};
            if (task_probe && args_span.size() >= args_bytes) {
                mesh_task_probe = {
                    .args = args_span,
                    .tick = scheduler.CurrentTick(),
                    .tasks = groups,
                    .capacity = capacity_words,
                    .pending = true,
                };
            }
            const MeshArena& arena{mesh_arenas[mesh_arena_current]};
            pipeline->SetMeshEmulationTaskBuffers(
                payload, offsets, args, {*arena.counter, 0, sizeof(u32)}, capacity_words);
            mesh_emulation_indirect = {args.buffer, args.offset};
        }
        if (probe_draw && output.mapped_span.size() >= output.offset + bytes) {
            const MeshEmulationSetup& setup{*pipeline->MeshEmulation()};
            mesh_probe = {
                .data = output.mapped_span.subspan(output.offset, bytes),
                .tick = scheduler.CurrentTick(),
                .frame = render_trace_frame,
                .groups = groups,
                .block_words = setup.block_words,
                .vertex_lines = setup.vertex_lines,
                .max_vertices = setup.max_vertices,
                .max_primitives = setup.max_primitives,
                .primitive_vertices = setup.primitive_vertices,
                .position_lines = setup.position_lines,
                .pending = true,
            };
        } else if (probe_draw && output.mapped_span.empty()) {
            static bool warned{};
            if (!warned) {
                warned = true;
                LOG_WARNING(Render_Vulkan, "Mesh emulation probe: output buffer is not mapped");
            }
        }
        mesh_emulation_groups = groups;
        mesh_emulation_vertices_per_group = pipeline->MeshEmulationVerticesPerGroup();
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::MacroReadWriterSet)) {
        Nx2ProveNonWriterDraw(*pipeline);
    }
    if (buffer_cache.HasPendingNx2ListCopies() &&
        !Tegra::Engines::Maxwell3D::FrontIsWriterDraw(maxwell3d->regs)) {
        SubmitNx2ListCopies();
    }
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    const u32 repack_dirty = UpdateVertexRepackLayout(pipeline->HasDynamicVertexInput());
    if (repack_dirty) {
        auto& flags = maxwell3d->dirty.flags;
        flags[VideoCommon::Dirty::VertexBuffers] = true;
        flags[Dirty::VertexInput] = true;
        for (u32 binding = 0; binding < VideoCommon::NUM_VERTEX_BUFFERS; ++binding) {
            if (repack_dirty & (1U << binding)) {
                flags[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
    }
    SCOPE_EXIT {
        staging_pool.HoldSubmits(false);
    };
    if (maxwell3d->dirty.flags[VideoCommon::Dirty::RenderTargets]) {
        auto& dirty_flags = maxwell3d->dirty.flags;
        dirty_flags[Dirty::LogicOpEnable] = true;
        dirty_flags[Dirty::Blending] = true;
        dirty_flags[Dirty::BlendEnable] = true;
        if (!maxwell3d->regs.viewport_scale_offset_enabled) {
            dirty_flags[Dirty::Viewports] = true;
            dirty_flags[Dirty::Scissors] = true;
        }
    }
    if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Configure(is_indexed)) {
        report_failure("configure failed");
        TraceRenderState(is_indexed, "configure-failed", nullptr);
        return;
    }

    TrackOcarinaCamera();
    TrackOcarinaActor(pipeline->ShaderHashes()[1], is_indexed, is_indirect);
    TracePendingCpuWrites(false, pipeline->ShaderHashes());
    TraceRenderState(is_indexed, "configured", texture_cache.GetFramebuffer());
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        FrameDumpPass(texture_cache.GetFramebuffer());
    }
    if (flash_watch.WatchesPasses()) {
        flash_watch.NoteDraw(pipeline->ShaderHashes());
    }
    if (frame_dump.Capturing()) {
        frame_dump.NoteDraw(pipeline->ShaderHashes());
        if (frame_dump.CapturingSampled()) {
            boost::container::small_vector<FrameDump::Sampled, 16> sampled;
            for (const auto& image : pipeline->SampledImages()) {
                sampled.push_back({image.stage, image.tic, &texture_cache.GetImageView(image.id),
                                   &texture_cache.GetSampler(image.sampler)});
            }
            bool recorded = frame_dump.NoteSampled(sampled);
            recorded |= FrameDumpBuffers(false, pipeline->ShaderHashes());
            if (recorded) {
                scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
            }
        }
    }
    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace pipeline: frame={} draw={} hashes={:016x} dynamic_vertex_input={}",
                  render_trace_frame, render_trace_draws, fmt::join(pipeline->ShaderHashes(), ","),
                  pipeline->HasDynamicVertexInput());
    }

    if (is_indirect && !mesh_draw && Common::DkbFrameTrace::Enabled()) {
        static const bool readback = [] {
            const char* value = std::getenv("EDEN_DKB_INDIRECT_READBACK");
            return value && std::string_view{value} == "1";
        }();
        if (readback) {
            CaptureIndirectTrace();
        }
    }

    NoteCameraMotionDraw();
    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        UpdateOcarinaJitter();
        UpdateDynamicStates(pipeline);
    }

    query_cache.NotifySegment(true);
    HandleTransformFeedback();
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const std::optional<u32> probe =
        frame_dump.CapturingSampled() ? frame_dump.BeginProbe() : std::nullopt;
    // EDEN_NX2_DIAG_SAMPLES: count the samples this draw passes while a draw list is captured. The
    // game's own occlusion counting is left alone (one occlusion query can be active at a time).
    int diag_query = -1;
    if (diag_active && VideoCommon::Nx2Diag::SamplesEnabled() && g_diag_pool != nullptr &&
        !maxwell3d->regs.zpass_pixel_count_enable) {
        diag_query = VideoCommon::Nx2Diag::Diagnostics::Get().NextQuery();
    }
    if (diag_query >= 0) {
        scheduler.Record([pool = **g_diag_pool, diag_query](vk::CommandBuffer cmdbuf) {
            cmdbuf.BeginQuery(pool, static_cast<u32>(diag_query), 0);
        });
    }
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        scheduler.TraceGraphicsPipeline(pipeline->ShaderHashes());
        trace_draw_hashes = {pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]};
        draw_func();
    }
    if (diag_query >= 0) {
        scheduler.Record([pool = **g_diag_pool, diag_query](vk::CommandBuffer cmdbuf) {
            cmdbuf.EndQuery(pool, static_cast<u32>(diag_query));
        });
        g_diag_query = diag_query;
    }
    if (probe) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const auto& regs = maxwell3d->regs;
        const bool alpha_to_coverage = regs.anti_alias_alpha_control.alpha_to_coverage != 0;
        const bool dynamic_alpha_to_coverage =
            device.IsExtExtendedDynamicState3EnablesSupported() &&
            device.SupportsDynamicState3AlphaToCoverageEnable() &&
            pipeline->UsesExtendedDynamicState();
        frame_dump.EndProbe(
            *probe,
            {
                .mesh = mesh_draw,
                .indirect = is_indirect,
                .first = is_indexed ? draw_state.index_buffer.first : draw_state.vertex_buffer.first,
                .elements = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
                .instances = draw_state.instance_count,
                .alpha_test = regs.alpha_test_enabled ? static_cast<u32>(regs.alpha_test_func) : 0,
                .alpha_ref = regs.alpha_test_ref,
                .alpha_to_coverage = alpha_to_coverage,
                .host_alpha_to_coverage = dynamic_alpha_to_coverage
                                              ? alpha_to_coverage_enable
                                              : alpha_to_coverage && pipeline->SupportsAlphaToCoverage(),
                .color0 = pipeline->SupportsAlphaToCoverage(),
                .kill = pipeline->FragmentCanDiscard(),
                .zero_handles = pipeline->ZeroTextureHandles(),
            });
    }
    submitted = true;
    const auto& hashes = pipeline->ShaderHashes();
    if (oot_4k_scene_frame == render_trace_frame) {
        constexpr u64 final_vertex = 0x16aff33d999f993fULL;
        constexpr u64 final_fragment = 0x52402d8e8ab55263ULL;
        if (hashes[5] == final_fragment &&
            std::ranges::find(hashes, final_vertex) != hashes.end()) {
            const auto target_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
            if (target_id) {
                const auto& target = texture_cache.GetImageView(target_id);
                if (target.size.width == 2560 && target.size.height == 1440 &&
                    target.format == VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB) {
                    oot_4k_guest_present = target.ImageHandle();
                }
            }
        }
    }
    // The game draws Navi's glow, fireflies and dust right after the fog. They write no depth and
    // move on their own, so the upscaler must not take them from its history.
    constexpr u64 fog_fragment = 0x3ddfb1d7a854c849ULL;
    if (hashes[5] == fog_fragment && oot_live_dlss && OcarinaTemporalEnabled() &&
        oot_camera_sample_frame == render_trace_frame && !frame_dump.Capturing()) {
        const auto scene_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
        if (scene_id) {
            const auto& scene = texture_cache.GetImageView(scene_id);
            if (scene.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                oot_motion.SceneSized(scene.size.width, scene.size.height) && !scene.IsRescaled() &&
                scene.range == VideoCommon::SubresourceRange{}) {
                oot_effects_scene = scene.Handle(Shader::TextureType::Color2D);
                oot_effects_address = scene.gpu_addr;
                oot_effects_frame = render_trace_frame;
                oot_live_dlss->SaveEffectsBase(oot_effects_scene, oot_motion.extent,
                                               render_trace_frame);
            }
        }
    }
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        native_capture_count < 16 && [] {
            static const bool capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return capture;
        }()) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}


namespace {
// EDEN_NX2_WINDOW_TRACE: describe clears and draws into screen-sized colour targets (debug aid).
void TraceWindowTarget(const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                       const char* kind, u32 count) {
    static const bool trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    if (!trace) {
        return;
    }
    static u32 traced{};
    const auto& regs = maxwell3d.regs;
    static const u64 skip = [] {
        const char* value = std::getenv("EDEN_NX2_WINDOW_TRACE_SKIP");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    static u64 seen_events{};
    static const u64 only_ps = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_PS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    const u64 pixel_shader =
        (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    if (regs.rt[0].width != 3840 || traced >= 400 || seen_events++ < skip ||
        (only_ps != 0 && pixel_shader != only_ps)) {
        return;
    }
    ++traced;
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    LOG_INFO(HW_GPU,
             "nx2 window {} rt0={:#x} count={} clear={:#x} color=({},{},{},{}) mask_common={} "
             "mask0={:#x} rt_count={} vp scale=({},{}) translate=({},{}) scissor en={} x={}..{} "
             "y={}..{} blend0={} vs={:#x} ps={:#x} rasterize={}",
             kind, regs.rt[0].Address(), count, regs.clear_surface.raw, regs.clear_color[0],
             regs.clear_color[1], regs.clear_color[2], regs.clear_color[3], regs.color_mask_common,
             regs.color_mask[0].raw, regs.rt_control.count.Value(), vp.scale_x, vp.scale_y,
             vp.translate_x, vp.translate_y, sc.enable, sc.min_x.Value(), sc.max_x.Value(),
             sc.min_y.Value(), sc.max_y.Value(), regs.blend.enable[0],
             (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
             (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
             regs.rasterize_enable);
    if (kind[0] != 'd') {
        return;
    }
    // Pixel-stage constant buffer 3: colour constants of the loading-screen passes.
    if (const auto& cb3 = maxwell3d.state.shader_stages[4].const_buffers[3]; cb3.enabled) {
        std::array<f32, 8> values{};
        memory.ReadBlockUnsafe(cb3.address, values.data(), sizeof(values));
        LOG_INFO(HW_GPU, "nx2 window ps c[3]@{:#x} size={}: {}", cb3.address, cb3.size,
                 fmt::join(values, " "));
    }
    static u32 texture_traces{};
    if (texture_traces++ > 200) {
        return;
    }
    // Texture used by the pixel shader's first bound TEX (handle in c[2][0x20]) and its TIC.
    const auto& cb = maxwell3d.state.shader_stages[4].const_buffers[2];
    const u32 handle = cb.enabled ? memory.Read<u32>(cb.address + 0x20) : 0;
    const u32 tic_index = handle & 0xfffff;
    const GPUVAddr tic_address = regs.tex_header.Address() + u64{tic_index} * 32;
    std::array<u32, 8> tic{};
    memory.ReadBlockUnsafe(tic_address, tic.data(), sizeof(tic));
    LOG_INFO(HW_GPU,
             "nx2 window texture: cb2={:#x} en={} handle={:#x} tic pool={:#x} limit={} "
             "sampler pool={:#x} tic[{}]={:08x}",
             cb.address, cb.enabled, handle, regs.tex_header.Address(), regs.tex_header.limit,
             regs.tex_sampler.Address(), tic_index, fmt::join(tic, " "));
    // First bytes of the texture's memory (block-linear, so just a sample).
    const GPUVAddr texture = (GPUVAddr{tic[2] & 0xffff} << 32) | tic[1];
    std::array<u32, 16> texels{};
    memory.ReadBlockUnsafe(texture, texels.data(), sizeof(texels));
    LOG_INFO(HW_GPU, "nx2 window texture {:#x} {}x{}: {:08x}", texture, (tic[4] & 0xffff) + 1,
             (tic[5] & 0xffff) + 1, fmt::join(texels, " "));
    // EDEN_NX2_TEXTURE_DUMP=<file>: save the first 4 MiB of the last traced texture's memory.
    if (const char* path = std::getenv("EDEN_NX2_TEXTURE_DUMP"); path != nullptr) {
        std::vector<u8> bytes(2048u * 10240u);
        memory.ReadBlockUnsafe(texture, bytes.data(), bytes.size());
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}
} // Anonymous namespace

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    if (VideoCommon::Nx2Diag::Active()) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        g_diag_draw = {is_indexed ? "indexed" : "draw",
                       is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
                       instance_count};
        if (VideoCommon::Nx2Diag::Ablates() && DiagAblated(*maxwell3d)) {
            DiagRecordDraw(*maxwell3d, *gpu_memory, "ablated", false);
            return;
        }
    }
    // A vertex count GPU work wrote (nx2_indirect_grid.h): drawn with that count, read on the GPU.
    // Draws that need the count here (the indirect reset, quads, a first vertex) read the words,
    // which waits for the GPU on this thread.
    std::optional<std::pair<VkBuffer, u32>> gpu_count;
    if (!is_indexed && Tegra::Nx2IndirectGrid::AnyLive()) {
        auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 count = draw_state.vertex_buffer.count;
        if (Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(count)) {
            auto& grid = Tegra::Nx2IndirectGrid::Get();
            Tegra::Nx2IndirectGrid::LogCounts();
            const auto source = Tegra::Nx2IndirectGrid::MatchDrawCount(count);
            if (!source) {
                grid.skipped.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1) < 8) {
                    LOG_ERROR(Render_Vulkan, "GPU-counted draw {:#x} lost its source: skipped", count);
                }
                return;
            }
            const u32 first = draw_state.vertex_buffer.first;
            const bool first_placeholder =
                first == Tegra::Nx2IndirectGrid::DrawPlaceholder(
                             count & (Tegra::Nx2IndirectGrid::Slots - 1), 1);
            const bool on_gpu = (first == 0 || first_placeholder) &&
                                draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
                                draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
                                device.IsExtTransformFeedbackSupported() &&
                                !IsNx2IndirectReset(is_indexed, instance_count);
            if (!on_gpu) {
                std::array<u32, 2> words{};
                gpu_memory->ReadBlock(*source, words.data(), sizeof(words));
                draw_state.vertex_buffer.count = words[0];
                if (first_placeholder) {
                    draw_state.vertex_buffer.first = words[1];
                }
                grid.draws_refused.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::scoped_lock lock{buffer_cache.mutex};
                const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                    *source, sizeof(u32), VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                    VideoCommon::ObtainBufferOperation::DoNothing);
                buffer->MarkUsage(offset, sizeof(u32));
                gpu_count.emplace(buffer->Handle(), offset);
                // The count comes from a dispatch before: make its writes visible to the draw's
                // indirect read when no pass-opening COMPUTE_TO_GRAPHICS barrier has run yet.
                if (std::exchange(compute_since_count_barrier, false) &&
                    !scheduler.IsRenderPassActive()) {
                    static constexpr VkMemoryBarrier COUNT_BARRIER{
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                         VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT |
                                         VK_ACCESS_MEMORY_READ_BIT,
                    };
                    scheduler.RequestOutsideRenderPassKeepClear();
                    scheduler.Record([](vk::CommandBuffer cmdbuf) {
                        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                               0, COUNT_BARRIER);
                    });
                }
                static const bool log_counts = [] {
                    const char* value = std::getenv("EDEN_NX2_INDIRECT_GRID");
                    return value != nullptr && value[0] == '2';
                }();
                if (log_counts) {
                    std::array<u32, 2> words{};
                    gpu_memory->ReadBlock(*source, words.data(), sizeof(words));
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1) < 40) {
                        LOG_INFO(Render_Vulkan,
                                 "GPU-counted draw from {:#x}: count {} first {} instances {} topology {}",
                                 *source, words[0], words[1], instance_count,
                                 static_cast<u32>(draw_state.topology));
                    }
                }
                draw_state.vertex_buffer.first = 0;
                grid.draws.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    {
        // A draw's first and count can come from memory a skipped (rejected) compute program
        // should have written. Past 16M vertices they are garbage: building the quad index
        // buffer for one threw out of the GPU thread and ended the process (Hyrule Warriors:
        // Age of Imprisonment, entering a new area), and indexed ones read far out of bounds.
        const auto& state = maxwell3d->draw_manager.draw_state;
        const u64 first = is_indexed ? state.index_buffer.first : state.vertex_buffer.first;
        const u64 count = is_indexed ? state.index_buffer.count : state.vertex_buffer.count;
        constexpr u64 MaxVertices = u64{1} << 24;
        if (!gpu_count && first + count > MaxVertices) {
            static std::atomic<u32> skipped{};
            if (skipped.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(HW_GPU, "Draw skipped: {} {} from {} (vertex/index count past {})",
                            count, is_indexed ? "indices" : "vertices", first, MaxVertices);
            }
            return;
        }
    }
    if (!gpu_count && TryNx2IndirectReset(is_indexed, instance_count)) {
        return;
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, is_indexed ? "draw-indexed" : "draw",
                      is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                 : maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    // EDEN_NX2_CB24_TRACE: for draws whose vertex shader reads bank 24, show the inline
    // constant buffer upload target (CB_ADDRESS) and its first words.
    if (static const bool trace = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr; trace) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if ((vs == 0x11471c9a00 || vs == 0x11471ca600 || vs == 0x11471cd300) && logged++ < 3) {
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(regs.const_buffer.Address(), words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "bank24 draw vs={:#x}: CB_ADDRESS {:#x} size {:#x} offset {:#x}: {:08x}", vs,
                     regs.const_buffer.Address(), regs.const_buffer.size, regs.const_buffer.offset,
                     fmt::join(words, " "));
            // Search every bound vertex-stage buffer for render-target-size values.
            const auto& stage = maxwell3d->state.shader_stages[0];
            for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& buffer = stage.const_buffers[slot];
                if (!buffer.enabled || buffer.size == 0) continue;
                std::vector<u32> data(std::min<u32>(buffer.size, 0x10000) / 4);
                gpu_memory->ReadBlockUnsafe(buffer.address, data.data(), data.size() * 4);
                for (size_t i = 0; i < data.size(); ++i) {
                    const u32 w = data[i];
                    if (w == 0x45200000 || w == 0x44b40000 || w == 0x39cccccd || w == 0x3a360b61 ||
                        w == 0x44a00000 || w == 0x44340000 || w == 2559 || w == 1439) {
                        LOG_INFO(HW_GPU, "  slot {} +{:#x} = {:#x}", slot, i * 4, w);
                    }
                }
            }
        }
    }
    // EDEN_NX2_TRACE_VS=<address>: describe the next few draws using that vertex shader like the
    // after-upscale trace does (render target, blend, fragment textures).
    static const u64 trace_vs = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    if (trace_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 traced{};
        if (vs == trace_vs && traced++ % 200 == 0 && traced < 1000) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        --Tegra::Engines::g_nx2_trace_draws_after_upscale;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU, "after upscale: draw rt0 {:#x} {}x{} fmt {:#x} block {}/{}/{} pitch_linear {} depth {} array_pitch {:#x} ctrl {:#x} vs {:#x} ps {:#x}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].tile_mode.block_width.Value(),
                 regs.rt[0].tile_mode.block_height.Value(), regs.rt[0].tile_mode.block_depth.Value(),
                 regs.rt[0].tile_mode.is_pitch_linear.Value(), regs.rt[0].depth.Value(),
                 regs.rt[0].array_pitch, regs.rt_control.count.Value(), vs, ps);
        // Fragment-stage texture handles (bound buffer 2) and the images they name.
        const auto& handles = maxwell3d->state.shader_stages[4].const_buffers[2];
        if (handles.enabled) {
            for (u32 i = 0; i < 24; ++i) {
                const u32 handle = gpu_memory->Read<u32>(handles.address + i * 4);
                if (handle == 0 || handle == 0xffffffff) continue;
                Tegra::Texture::TICEntry tic{};
                gpu_memory->ReadBlockUnsafe(regs.tex_header.Address() + u64{handle & 0xfffff} * sizeof(tic),
                                            &tic, sizeof(tic));
                LOG_INFO(HW_GPU, "  fs handle[{}] = {:#x}: {:#x} {}x{} format {:#x}", i, handle,
                         tic.Address(), tic.Width(), tic.Height(), static_cast<u32>(tic.format.Value()));
            }
        }
        const auto& bt = regs.blend_per_target[0];
        LOG_INFO(HW_GPU, "  blend enable {} per-target {} global op {} src {:#x} dst {:#x} | rt0 op {} src {:#x} dst {:#x} alpha {:#x}/{:#x} mask {:#x}",
                 regs.blend.enable[0], regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend.color_op), static_cast<u32>(regs.blend.color_source),
                 static_cast<u32>(regs.blend.color_dest), static_cast<u32>(bt.color_op),
                 static_cast<u32>(bt.color_source), static_cast<u32>(bt.color_dest),
                 static_cast<u32>(bt.alpha_source), static_cast<u32>(bt.alpha_dest),
                 regs.color_mask[0].raw);
    }
    // EDEN_NX2_SKIP_VS=<address>[,<address>...]: drop draws using those vertex shaders (debug aid).
    static const std::vector<u64> skip_list = [] {
        std::vector<u64> list;
        if (const char* value = std::getenv("EDEN_NX2_SKIP_VS")) {
            for (char* end = nullptr; *value; value = *end ? end + 1 : end) {
                list.push_back(std::strtoull(value, &end, 16));
                if (end == value) break;
            }
        }
        return list;
    }();
    if (!skip_list.empty()) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        if (std::find(skip_list.begin(), skip_list.end(), vs) != skip_list.end()) return;
    }
    // The scene image: a depth-tested draw to a floating-point (HDR) colour target. Tonemapped
    // post-processing targets are 8- or 10-bit.
    if (const auto& regs = maxwell3d->regs;
        regs.zeta_enable != 0 && regs.depth_test_enable != 0 && regs.rt_control.count > 0 &&
        (regs.rt[0].format == Tegra::RenderTargetFormat::B10G11R11_FLOAT ||
         regs.rt[0].format == Tegra::RenderTargetFormat::R16G16B16A16_FLOAT)) {
        Tegra::Engines::g_nx2_scene_target.store(regs.rt[0].Address(), std::memory_order_relaxed);
        Tegra::Engines::g_nx2_scene_target_size.store(
            (u64{regs.rt[0].width} << 32) | regs.rt[0].height, std::memory_order_relaxed);
    }
    // EDEN_NX2_RT_TRACE: log the first draw to every distinct colour target (debug aid).
    if (static const bool trace = std::getenv("EDEN_NX2_RT_TRACE") != nullptr; trace) {
        static std::set<std::tuple<GPUVAddr, u32, u32, u64, u64>> seen;
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        const auto key = std::make_tuple(rt.Address(), rt.width, static_cast<u32>(rt.format), vs, ps);
        if (rt.width >= 640 && seen.size() < 600 && seen.insert(key).second) {
            LOG_INFO(HW_GPU, "nx2 draw target {:#x} {}x{} format={:#x} vs={:#x} ps={:#x} blend={} (pairs so far {})",
                     rt.Address(), rt.width, rt.height, static_cast<u32>(rt.format), vs, ps,
                     regs.blend.enable[0], seen.size());
        }
    }
    if (!gpu_count && maxwell3d->regs.transform_feedback_enabled == 0 &&
        (MakeDrawParams(maxwell3d->draw_manager.draw_state, instance_count, is_indexed)
                 .num_vertices == 0 ||
         (instance_count == 0 && [this] {
             const GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
             return pipeline != nullptr && !pipeline->IsMeshPipeline();
         }()))) {
        if (Common::DkbFrameTrace::Enabled()) {
            ++trace_draws;
            ++trace_zero_direct;
        }
        gpu.TickWork();
        return;
    }
    PrepareDraw(is_indexed, [this, is_indexed, instance_count, gpu_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        if (gpu_count) {
            scheduler.Record([this, counter = *gpu_count, num_instances,
                              base_instance = draw_state.base_instance](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                // Stride 1: the vertex count is the counter's value.
                cmdbuf.DrawIndirectByteCountEXT(num_instances, base_instance, counter.first,
                                                counter.second, 0, 1);
            });
            return;
        }
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};
        if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace direct: frame={} draw={} count={} instances={} first_index={} "
                      "base_vertex={} base_instance={} mesh={}",
                      render_trace_frame, render_trace_draws, draw_params.num_vertices,
                      draw_params.num_instances, draw_params.first_index, draw_params.base_vertex,
                      draw_params.base_instance, mesh_draw);
        }

        if (Common::DkbFrameTrace::Enabled() &&
            (draw_params.num_vertices == 0 || (!mesh_draw && draw_params.num_instances == 0))) {
            ++trace_zero_direct;
        }

        if (mesh_draw && mesh_emulation_vertices_per_group != 0) {
            const u32 vertices{mesh_emulation_groups * mesh_emulation_vertices_per_group};
            if (vertices == 0) {
                return;
            }
            if (mesh_emulation_indirect.buffer) {
                // With a task program the distribution pass wrote the vertex count.
                scheduler.Record([this, indirect = mesh_emulation_indirect](vk::CommandBuffer cmdbuf) {
                    if (scheduler.GraphicsPipelineMissing()) {
                        return;
                    }
                    cmdbuf.DrawIndirect(indirect.buffer,
                                        indirect.offset +
                                            Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET,
                                        1, 0);
                });
                return;
            }
            scheduler.Record([this, vertices](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.Draw(vertices, 1, 0, 0);
            });
            return;
        }
        if (mesh_draw) {
            scheduler.Record([this, groups = draw_params.num_vertices](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
            });
            return;
        }
        scheduler.Record([this, draw_params, hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            scheduler.TraceGpuWork(
                cmdbuf, hashes.first, hashes.second,
                {draw_params.num_vertices, draw_params.num_instances, 0}, [&] {
                    if (draw_params.is_indexed) {
                        if (!scheduler.CheckIndexBufferBound(hashes.first, hashes.second)) {
                            return;
                        }
                        cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                           draw_params.first_index, draw_params.base_vertex,
                                           draw_params.base_instance);
                    } else {
                        cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                                    draw_params.base_vertex, draw_params.base_instance);
                    }
                });
        });

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    });
}

bool RasterizerVulkan::DrawMeshTasksFromGpu(DAddr count_addr) {
    if (!device.IsExtMeshShaderSupported() || !maxwell3d->regs.IsMeshEnabled()) {
        return false;
    }
    // A pipeline over the native limits is emulated: it needs the count on the CPU.
    if (pipeline_cache.HasEmulatedNativeMeshPipelines()) {
        const GraphicsPipeline* const pipeline{pipeline_cache.CurrentGraphicsPipeline()};
        if (pipeline != nullptr && pipeline->IsEmulatedMesh()) {
            return false;
        }
    }
    std::optional<u32> slot;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        slot = buffer_cache.Nx2MeshCountSlot(count_addr);
    }
    if (!slot) {
        return false;
    }
    bool drawn = false;
    PrepareDraw(
        false,
        [this, &drawn, offset = VkDeviceSize{*slot} * BufferCacheRuntime::Nx2MeshMirrorSlotSize] {
            // Only a native task/mesh pipeline takes the count as is; anything else needs it on
            // the CPU (the caller reads it and draws again).
            if (!mesh_draw) {
                return;
            }
            scheduler.Record([this, mirror = buffer_cache_runtime.Nx2MeshMirror(),
                              offset](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawMeshTasksIndirectEXT(mirror, offset, 1,
                                                BufferCacheRuntime::Nx2MeshMirrorSlotSize);
            });
            drawn = true;
        },
        true);
    return drawn;
}

void RasterizerVulkan::DrawIndirect() {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    if (VideoCommon::Nx2Diag::Active()) {
        g_diag_draw = {"indirect", static_cast<u32>(params.max_draw_counts), 0};
        if (VideoCommon::Nx2Diag::Ablates() && DiagAblated(*maxwell3d)) {
            DiagRecordDraw(*maxwell3d, *gpu_memory, "ablated", false);
            return;
        }
    }
    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_indirect;
    }
    if (IsIndirectFromRecord(params) && scheduler.StorageWritesUnordered()) {
        // The record may come from the stores of earlier draws (DK Bananza's terrain culling
        // geometry shader, its reset vertex shader): nothing orders the host's argument fetch
        // after them unless the guest's own wait for idle recorded a barrier since.
        // EDEN_NX2_INDIRECT_BARRIER=0: draw without it.
        static const bool barrier = [] {
            const char* value = std::getenv("EDEN_NX2_INDIRECT_BARRIER");
            return value == nullptr || value[0] != '0';
        }();
        CountIndirectRecordUnordered();
        if (barrier) {
            scheduler.RequestOutsideRenderPassOperationContext();
            scheduler.Record([](vk::CommandBuffer cmdbuf) {
                static constexpr VkMemoryBarrier STORES_TO_ARGUMENTS{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                     VK_ACCESS_INDEX_READ_BIT |
                                     VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                     VK_ACCESS_SHADER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                       VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0,
                                       STORES_TO_ARGUMENTS);
            });
            scheduler.NoteStorageWritesOrdered();
        }
    }
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace indirect: frame={} draw={} address={:#x} count_address={:#x} "
                      "bytes={} max_draws={} stride={} include_count={} byte_count={} mesh={}",
                      render_trace_frame, render_trace_draws, params.indirect_start_address,
                      params.count_start_address, params.buffer_size, params.max_draw_counts,
                      params.stride, params.include_count, params.is_byte_count, mesh_draw);
        }
        if (mesh_draw) {
            TraceDrawFailure("indirect task/mesh unsupported");
            // Indirect task/mesh draws would need vkCmdDrawMeshTasksIndirectEXT's layout.
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "Indirect task/mesh draw skipped");
            }
            return;
        }
        const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([this, buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            return;
        }
        if (params.include_count) {
            const auto count = buffer_cache.GetDrawIndirectCount();
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([this, draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                if (params.is_indexed) {
                    if (!scheduler.CheckIndexBufferBound(0, 0)) {
                        return;
                    }
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            return;
        }
        scheduler.Record([this, buffer_obj = buffer->Handle(), offset, params,
                          hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, hashes.first, hashes.second, {}, [&] {
                if (params.is_indexed) {
                    if (!scheduler.CheckIndexBufferBound(hashes.first, hashes.second)) {
                        return;
                    }
                    cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                               static_cast<u32>(params.max_draw_counts),
                                               static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirect(buffer_obj, offset,
                                        static_cast<u32>(params.max_draw_counts),
                                        static_cast<u32>(params.stride));
                }
            });
        });

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    }, true);
    buffer_cache.SetDrawIndirect(nullptr);
}

VideoCommon::OotNetwork::BypassState* RasterizerVulkan::OotNetworkBypass() {
    // Temporal upscaling replaces the same output pass and takes precedence until it fails.
    if (OcarinaTemporalEnabled() && !(oot_live_dlss && oot_live_dlss->EvaluationFailed())) {
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina of Time: No sharpening is off while temporal upscaling is on");
        }
        return nullptr;
    }
    return &oot_network_bypass;
}

void RasterizerVulkan::TrackOotTonemappedImage(const GraphicsPipeline& pipeline) {
    const auto& hashes = pipeline.ShaderHashes();
    if (std::none_of(hashes.begin(), hashes.end(), VideoCommon::OotNetwork::IsTonemapShader)) {
        return;
    }
    const auto& target = maxwell3d->regs.rt[0];
    if (target.Address() == 0) {
        return;
    }
    std::scoped_lock lock{oot_network_bypass.mutex};
    if (oot_network_bypass.tonemapped.Address() != target.Address()) {
        LOG_INFO(Render_Vulkan, "Ocarina of Time: tonemapped image at {:#x} {}x{}", target.Address(),
                 target.width, target.height);
    }
    oot_network_bypass.tonemapped = target;
}

void RasterizerVulkan::TraceDrawFailure(std::string_view reason) {
    if (!Common::DkbFrameTrace::Enabled()) {
        return;
    }
    ++trace_skipped;
    if (trace_failure_details++ >= 4) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    std::array<GPUVAddr, 6> stages{};
    for (size_t i = 0; i < stages.size(); ++i) {
        if (regs.IsShaderConfigEnabled(i)) {
            stages[i] = maxwell3d->GetNativeMacroEngine().native_turing
                            ? (u64{regs.pipelines[i].reserved[0] & 0xff} << 32) |
                                  regs.pipelines[i].reserved[1]
                            : regs.program_region.Address() + regs.pipelines[i].offset;
        }
    }
    LOG_INFO(Render_Vulkan,
             "DKB draw skipped frame={} reason={} stages[0..5]={:#x} rt0={:#x} {}x{}",
             trace_frame, reason, fmt::join(stages, ","), regs.rt[0].Address(),
             regs.rt[0].width, regs.rt[0].height);
}

void RasterizerVulkan::CaptureIndirectTrace() {
    if (!trace_sample_frame || trace_captures >= 4) {
        return;
    }
    ++trace_captures;
    const auto slot = std::find_if(indirect_traces.begin(), indirect_traces.end(),
                                   [](const auto& trace) { return !trace.pending; });
    if (slot == indirect_traces.end()) {
        ++trace_capture_dropped;
        return;
    }
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const auto [commands, command_offset] = buffer_cache.GetDrawIndirectBuffer();
    const u32 max_draws = static_cast<u32>(params.max_draw_counts);
    std::array<VkBufferCopy, 8> copies{};
    u32 sampled{};
    const u64 command_bytes = params.is_byte_count ? 4 : 8;
    const u64 available = command_offset <= commands->SizeBytes()
                              ? std::min<u64>(params.buffer_size,
                                              commands->SizeBytes() - command_offset)
                              : 0;
    const u32 limit = params.is_byte_count ? 1 : std::min(max_draws, 8U);
    for (u32 i = 0; i < limit; ++i) {
        if (available < command_bytes || (i && params.stride > (available - command_bytes) / i)) {
            break;
        }
        const u64 offset = command_offset + i * params.stride;
        copies[sampled++] = VkBufferCopy{offset, i * 8ULL, command_bytes};
    }
    VkBuffer count_buffer{};
    VkBufferCopy count_copy{};
    if (params.include_count && !params.is_byte_count) {
        const auto [count, offset] = buffer_cache.GetDrawIndirectCount();
        if (offset > count->SizeBytes() || sizeof(u32) > count->SizeBytes() - offset) {
            ++trace_capture_dropped;
            return;
        }
        count_buffer = count->Handle();
        count_copy = VkBufferCopy{offset, 64, sizeof(u32)};
    }
    if (!sampled && !count_buffer) {
        LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} max_draws={} sampled=0",
                 trace_frame, params.indirect_start_address, max_draws);
        return;
    }
    if (!slot->readback) {
        const MemoryTagScope memory_tag{MemoryTag::Diagnostics};
        slot->readback = memory_allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = 68,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::Download);
    }
    slot->frame = trace_frame;
    slot->address = params.indirect_start_address;
    slot->count_address = params.count_start_address;
    slot->max_draws = max_draws;
    slot->sampled_commands = sampled;
    slot->include_count = count_buffer != VK_NULL_HANDLE;
    slot->byte_count = params.is_byte_count;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = commands->Handle(), destination = *slot->readback, copies, sampled,
                      count_buffer, count_copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        if (sampled) {
            cmdbuf.CopyBuffer(source, destination, vk::Span(copies.data(), sampled));
        }
        if (count_buffer) {
            cmdbuf.CopyBuffer(count_buffer, destination, count_copy);
        }
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, after);
    });
    slot->tick = scheduler.CurrentTick();
    slot->pending = true;
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    scheduler.NoteRenderPassStores();
}

void RasterizerVulkan::CollectIndirectTrace() {
    for (auto& trace : indirect_traces) {
        if (!trace.pending || !scheduler.IsFree(trace.tick)) {
            continue;
        }
        trace.readback.Invalidate();
        std::array<u32, 17> words{};
        std::memcpy(words.data(), trace.readback.Mapped().data(), sizeof(words));
        if (trace.byte_count) {
            LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} gpu_byte_count={}",
                     trace.frame, trace.address, words[0]);
        } else {
            const u32 count = trace.include_count ? words[16] : trace.max_draws;
            const u32 requested_draws = std::min(count, trace.max_draws);
            const u32 sampled = std::min(requested_draws, trace.sampled_commands);
            u32 zero_commands{};
            std::string commands;
            for (u32 i = 0; i < sampled; ++i) {
                zero_commands += words[2 * i] == 0 || words[2 * i + 1] == 0;
                commands += fmt::format("{}{}x{}", i ? "," : "", words[2 * i], words[2 * i + 1]);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB indirect frame={} address={:#x} count_address={:#x} draw_count={} "
                     "count_source={} max_draws={} requested_draws={} sampled={} zero_commands={} "
                     "vertices_or_indices_x_instances=[{}] conditional_execution=unobserved",
                     trace.frame, trace.address, trace.count_address, count,
                     trace.include_count ? "gpu" : "command-parameter",
                     trace.max_draws, requested_draws, sampled, zero_commands, commands);
        }
        trace.pending = false;
    }
}

void RasterizerVulkan::DrawTexture() {

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    if (maxwell3d->dirty.flags[VideoCommon::Dirty::RenderTargets]) {
        auto& dirty_flags = maxwell3d->dirty.flags;
        dirty_flags[Dirty::LogicOpEnable] = true;
        dirty_flags[Dirty::Blending] = true;
        dirty_flags[Dirty::BlendEnable] = true;
        if (!maxwell3d->regs.viewport_scale_offset_enabled) {
            dirty_flags[Dirty::Viewports] = true;
            dirty_flags[Dirty::Scissors] = true;
        }
    }
    texture_cache.UpdateRenderTargets(false);

    UpdateDynamicStates(nullptr);

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();
    // No render target (OoT's register state while entering Kokiri Forest): nothing to draw to.
    if (framebuffer == nullptr) {
        LOG_WARNING(Render_Vulkan, "DrawTexture skipped: no render target");
        return;
    }
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        FrameDumpPass(framebuffer);
    }
    if (flash_watch.WatchesPasses()) {
        flash_watch.NoteDraw({});
    }

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

void RasterizerVulkan::Clear(u32 layer_count) {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU, "after upscale: clear rt0 {:#x} {}x{} fmt {:#x} surface {:#x} colour {} {} {} {}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.clear_surface.raw, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]);
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, "clear", layer_count);
    FlushWork();
    gpu_memory->FlushCaching();

    auto& regs = maxwell3d->regs;
    const u32 color_attachment = regs.clear_surface.RT;
    const bool use_color =
        (regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
         regs.clear_surface.A) &&
        color_attachment < regs.rt_control.count &&
        regs.rt[color_attachment].Address() != 0 &&
        regs.rt[color_attachment].format != Tegra::RenderTargetFormat::NONE;
    const bool use_depth = regs.clear_surface.Z && regs.zeta_enable && regs.zeta.Address() != 0;
    const bool use_stencil = regs.clear_surface.S && regs.zeta_enable && regs.zeta.Address() != 0;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }
    if (use_depth) {
        NoteCameraMotionClear();
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    const bool clear_color = use_color && framebuffer->HasAspectColorBit(color_attachment);
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (!clear_color && aspect_flags == 0) {
        return;
    }
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        FrameDumpPass(framebuffer);
    }
    const VkExtent2D render_area = framebuffer->RenderArea();
    const u32 base_array_layer = regs.clear_surface.layer;
    const auto clear_layers =
        GetFramebufferClearLayers(base_array_layer, layer_count, framebuffer->Layers());
    if (clear_layers.count != layer_count) {
        LOG_DEBUG(Render_Vulkan,
                  "Framebuffer clear layers: base={}, requested={}, clipped={}, framebuffer={}",
                  base_array_layer, layer_count, clear_layers.count, framebuffer->Layers());
    }
    if (clear_layers.count == 0) {
        return;
    }

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = base_array_layer,
        .layerCount = clear_layers.count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_clears++ < 2048) {
        const u32 target = regs.clear_surface.RT;
        LOG_DEBUG(Render_Vulkan,
                  "Render trace clear: frame={} docked={} color={} depth={} stencil={} "
                  "color_target={} color_address={:#x} depth_address={:#x} depth_value={} base_layer={} layers={} "
                  "framebuffer_layers={} rect=({},{} {}x{})",
                  render_trace_frame, Settings::IsDockedMode(), use_color, use_depth, use_stencil,
                  target, target < regs.rt.size() ? regs.rt[target].Address() : 0,
                  regs.zeta.Address(), regs.clear_depth,
                  clear_rect.baseArrayLayer, clear_rect.layerCount, framebuffer->Layers(),
                  clear_rect.rect.offset.x, clear_rect.rect.offset.y,
                  clear_rect.rect.extent.width, clear_rect.rect.extent.height);
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 clear_layers.covers_all_layers &&
                                 !scheduler.RenderPassMatches(framebuffer) &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (submit_while_waited && scheduler.IsRenderPassActive() &&
        !scheduler.RenderPassMatches(framebuffer) &&
        scheduler.DrawsSinceSubmit() >= submit_waited_draws &&
        Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        VideoCommon::GpuSplit::OnBackThread() &&
        (VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
         (scheduler.DrawsSinceSubmit() >= submit_idle_draws && scheduler.GpuCaughtUp()))) {
        scheduler.RequestOutsideRenderPassOperationContext();
        if (SubmitWhileWaited()) {
            draw_counter = 0;
            pending_native_work = false;
        }
    }
    if (!scheduler.RenderPassMatches(framebuffer)) {
        scheduler.FlushExpiredDeferred();
    }
    const u64 target_key =
        ((regs.rt_control.count != 0 ? regs.rt[0].Address() : 0ULL) ^
         ((regs.zeta_enable ? regs.zeta.Address() : 0ULL) * 0x9e3779b97f4a7c15ULL)) |
        1ULL;
    scheduler.NoteTargetKey(target_key);
    if (!can_defer_clear) {
        if (!scheduler.RenderPassMatches(framebuffer)) {
            if (!scheduler.SplitBeforeTargetSwitch(framebuffer) &&
                !scheduler.IsRenderPassActive()) {
                (void)scheduler.SplitBeforeNextPass(
                    u64{render_area.width} * render_area.height, framebuffer->Handle(), target_key);
            }
        }
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    // Deferred load-op clears have no render pass yet. Starting a query here
    // would begin it outside the pass and end it inside the next draw's pass.
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                              !can_defer_clear && regs.zpass_pixel_count_enable);

    if (clear_color) {
        const auto format =
            VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        VkClearValue clear_value{};
        if (!IsPixelFormatInteger(format)) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(),
                        regs.clear_color.size() * sizeof(f32));
        } else if (!IsPixelFormatSignedInteger(format)) {
            for (size_t i = 0; i < 4; ++i) {
                clear_value.color.uint32[i] = std::bit_cast<u32>(regs.clear_color[i]);
            }
        } else {
            for (size_t i = 0; i < 4; ++i) {
                clear_value.color.int32[i] = std::bit_cast<s32>(regs.clear_color[i]);
            }
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region,
                                  regs.clear_surface.RT);
        }
    }

    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.NoteDepthStencilWrite();
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

void RasterizerVulkan::NoteDispatchDeps(const ComputePipeline& pipeline) {
    if (!DispatchDepsTrace() || !scheduler.IsRenderPassActive()) {
        return;
    }
    struct Deps {
        u64 hash{};
        u32 ends{};          ///< Open passes it ended
        u32 split{};         ///< Of them, passes that could take work before them
        u32 targets{};       ///< Bound one of the pass's attachments
        u32 stores{};        ///< Came after storage writes in the pass
        u32 writes{};        ///< Writes storage buffers, global memory or images
    };
    static std::vector<Deps> deps;
    static auto window_start = std::chrono::steady_clock::now();
    const Shader::Info& info = pipeline.Info();
    const auto written = [](const auto& descriptors) {
        return std::ranges::any_of(descriptors, [](const auto& desc) { return desc.is_written; });
    };
    auto it = std::ranges::find(deps, pipeline.ShaderHash(), &Deps::hash);
    if (it == deps.end()) {
        it = deps.insert(deps.end(), Deps{.hash = pipeline.ShaderHash()});
    }
    ++it->ends;
    it->split += scheduler.CanRecordBeforePass() ? 1 : 0;
    it->targets += texture_cache.SamplesRenderTarget(pipeline.BoundImages()) ? 1 : 0;
    it->stores += scheduler.RenderPassHasStores() ? 1 : 0;
    it->writes += info.stores_global_memory || written(info.storage_buffers_descriptors) ||
                          written(info.image_descriptors) || written(info.image_buffer_descriptors)
                      ? 1
                      : 0;
    const auto now = std::chrono::steady_clock::now();
    if (now - window_start < std::chrono::seconds{10}) {
        return;
    }
    window_start = now;
    std::ranges::sort(deps, std::greater{}, &Deps::ends);
    std::string text;
    for (const Deps& d : deps) {
        text += fmt::format("{}{:016x}: {} ends ({} after a split, {} bind its targets, {} after "
                            "storage writes, {} write storage)",
                            text.empty() ? "" : " | ", d.hash, d.ends, d.split, d.targets, d.stores,
                            d.writes);
    }
    LOG_INFO(Common_Memory, "Dispatch deps (10 s): {}", text);
    deps.clear();
}

bool RasterizerVulkan::ShouldSkipDispatchBarrier(
    const ComputePipeline& pipeline, bool may_chain,
    std::optional<ComputeAccessRange> indirect_range) {
    Common::PerfReport::Add(Common::PerfReport::Dispatches);
    const bool follows = may_chain && scheduler.FollowsDispatch();
    if (follows) {
        Common::PerfReport::Add(Common::PerfReport::DispatchesChained);
    }
    if (follows && DispatchChainEnabled(device)) {
        Common::PerfReport::Add(Common::PerfReport::DispatchBarriersSkipped);
        return true;
    }
    if (!may_chain || ImmediatePostDispatchBarrierEnabled(device) ||
        !VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::ComputeBarrierElide)) {
        dispatch_chain.Reset();
        return false;
    }

    const bool cur_unbounded = pipeline.HasUnboundedAccess();
    const auto base_reads = pipeline.ReadRanges();
    const auto cur_writes = pipeline.WriteRanges();
    boost::container::small_vector<ComputeAccessRange, 32> indirect_reads;
    std::span<const ComputeAccessRange> cur_reads = base_reads;
    if (indirect_range && !cur_unbounded) {
        indirect_reads.assign(base_reads.begin(), base_reads.end());
        AppendComputeRange(indirect_reads, indirect_range->begin, indirect_range->end);
        cur_reads = indirect_reads;
    }

    const auto cur_read_imgs = pipeline.UnbackedReadImages();
    const auto cur_write_imgs = pipeline.UnbackedWriteImages();
    if (follows &&
        dispatch_chain.CanElide(cur_reads, cur_writes, cur_read_imgs, cur_write_imgs,
                                cur_unbounded)) {
        dispatch_chain.Append(cur_reads, cur_writes, cur_read_imgs, cur_write_imgs);
        Common::PerfReport::Add(Common::PerfReport::DispatchBarriersSkipped);
        return true;
    }
    dispatch_chain.Start(cur_reads, cur_writes, cur_read_imgs, cur_write_imgs, cur_unbounded);
    return false;
}

void RasterizerVulkan::DispatchCompute() {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Compute};
    ++nx2_dispatch_serial;
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    scheduler.FlushExpiredDeferred();
    FlushWork();
    gpu_memory->FlushCaching();
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.DrainIfFull();
    }
    if (flash_watch.WatchesPasses()) {
        flash_watch.EndPass();
    }

    u64 native_key{};
    ComputePipeline* const pipeline{[&] {
        try {
            return pipeline_cache.CurrentComputePipeline(native_key);
        } catch (const Shader::Exception& error) {
            if (frame_dump.Capturing()) {
                frame_dump.NoteSkipped(true,
                                       kepler_compute->regs.code_loc.Address() +
                                           kepler_compute->launch_description.program_start,
                                       0, error.what());
            }
            throw;
        }
    }()};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) {
            // A program rejected before returns no pipeline without throwing (ComputeShader).
            if (frame_dump.Capturing()) {
                frame_dump.NoteSkipped(true,
                                       kepler_compute->regs.code_loc.Address() +
                                           kepler_compute->launch_description.program_start,
                                       0, "rejected (cached)");
            }
            throw Shader::RuntimeError("Native compute pipeline unavailable");
        }
        return;
    }
    if (kepler_compute->IsNativeMode() &&
        kepler_compute->ProcessOcarinaUpscale(Settings::GetCurrentProgramID(), native_key,
                                              pipeline->ShaderHash())) {
        return;
    }
    const Shader::Info& shader_info = pipeline->Info();
    const bool may_hoist_before_pass =
        DispatchBeforePassEnabled() && scheduler.IsRenderPassActive() &&
        !scheduler.HasDeferredWaitForIdle() && !scheduler.RenderPassHasStores() &&
        !scheduler.StorageWritesUnordered() &&
        (maxwell3d == nullptr || maxwell3d->regs.transform_feedback_enabled == 0) &&
        !frame_dump.Capturing() && !flash_watch.WatchesPasses() &&
        shader_info.image_buffer_descriptors.empty() &&
        std::ranges::all_of(shader_info.image_descriptors, [](const auto& desc) {
            return desc.is_written && !desc.is_byte_addressed;
        });
    if (submit_while_waited && scheduler.IsRenderPassActive() &&
        scheduler.DrawsSinceSubmit() >= submit_waited_draws &&
        Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        VideoCommon::GpuSplit::OnBackThread() &&
        (VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
         (scheduler.DrawsSinceSubmit() >= submit_idle_draws && scheduler.GpuCaughtUp()))) {
        if (may_hoist_before_pass) {
            scheduler.DispatchWork(true);
        } else {
            scheduler.RequestOutsideRenderPassOperationContext();
            if (SubmitWhileWaited()) {
                draw_counter = 0;
                pending_native_work = false;
            }
        }
    }
    if (VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteDispatch(
            pipeline->ShaderHash(), kepler_compute->GetIndirectComputeAddress().has_value());
    }
    std::scoped_lock buffer_lock{buffer_cache.mutex};
    std::scoped_lock texture_lock{texture_cache.mutex};
    std::optional<ByteAddressedShadow> byte_addressed_shadow;
    const auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    const auto oot_pass = VideoCommon::FindOcarinaUpscalePass(native_key, pipeline->ShaderHash());
    const bool can_replace =
        OcarinaTemporalEnabled() &&
        oot_pass != VideoCommon::OcarinaUpscalePass::None &&
        pipeline->CanReplaceOcarinaUpscale() && !indirect_address &&
        !frame_dump.Capturing();
    const bool can_bilinear =
        OcarinaBilinearUpscaleEnabled() &&
        (oot_pass == VideoCommon::OcarinaUpscalePass::Scaled ||
         oot_pass == VideoCommon::OcarinaUpscalePass::ThreeQuarter) &&
        pipeline->CanReplaceOcarinaUpscale() && !indirect_address &&
        program_id == VideoCommon::OotNetwork::ProgramId &&
        !frame_dump.Capturing();
    const bool marking_effects = oot_effects_frame == render_trace_frame &&
                                 OcarinaTemporalEnabled() && !indirect_address &&
                                 !frame_dump.Capturing();
    bool try_before_pass = may_hoist_before_pass && !can_replace && !can_bilinear &&
                           !marking_effects && scheduler.IsRenderPassActive();
    if (try_before_pass && !shader_info.storage_buffers_descriptors.empty() &&
        (shader_info.stores_global_memory ||
         std::ranges::any_of(shader_info.storage_buffers_descriptors,
                             &Shader::StorageBufferDescriptor::is_written))) {
        buffer_cache.SetComputeUniformBufferState(0, nullptr);
        buffer_cache.UnbindComputeStorageBuffers();
        buffer_cache.UnbindComputeTextureBuffers();
        size_t ssbo_idx = 0;
        for (const auto& desc : shader_info.storage_buffers_descriptors) {
            buffer_cache.BindComputeStorageBuffer(ssbo_idx++, desc.cbuf_index, desc.cbuf_offset,
                                                  desc.is_written, desc.indirect_offset,
                                                  desc.write_extent);
        }
        buffer_cache.UpdateComputeBuffers();
        buffer_cache.ForEachBoundBuffer(
            true, [&](u32, bool storage, u32, bool written, const auto&, u32 bound_size,
                      const Buffer* host, u32 host_offset) {
                if (!try_before_pass || !storage || bound_size == 0 ||
                    (!written && !shader_info.stores_global_memory)) {
                    return;
                }
                if (host == nullptr || host->IsRegionUsed(host_offset, bound_size)) {
                    try_before_pass = false;
                }
            });
        buffer_cache.UnbindComputeStorageBuffers();
        if (!scheduler.IsRenderPassActive()) {
            try_before_pass = false;
        }
    }
    // Command list pages the dispatch writes are copied for the split's front right after it.
    buffer_cache.Nx2BeginWriterTrace(pipeline->ShaderHash());
    buffer_cache.Nx2CollectListCopies(true);
    SCOPE_EXIT {
        buffer_cache.Nx2CollectListCopies(false);
        buffer_cache.Nx2EndWriterTrace();
        if (byte_addressed_shadow) {
            texture_cache_runtime.RetireByteAddressedShadow(*byte_addressed_shadow);
        }
    };
    if (buffer_cache.Nx2TracingBufferDependencies() && indirect_address) {
        if (const auto address = gpu_memory->GpuToCpuAddress(*indirect_address)) {
            buffer_cache.Nx2TraceBufferInput(*address, 12,
                                             VideoCommon::BufferDependencyTrace::Input::Indirect);
        }
    }
    const u64 tick_before_cfg = scheduler.CurrentTick();
    const u64 records_before_cfg = scheduler.WorkRecords();
    const bool want_deferred_bind =
        can_replace || can_bilinear || marking_effects || try_before_pass;
    ComputePipeline::BindCommand deferred_bind;
    if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                             texture_cache, texture_cache_runtime, byte_addressed_shadow,
                             want_deferred_bind ? &deferred_bind : nullptr)) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    }
    const u64 tick_after_cfg = scheduler.CurrentTick();
    const u64 records_after_cfg = scheduler.WorkRecords();
    if (VideoCommon::GpuSplit::RecordingDispatch()) {
        // EXP10: the buffer cache recorded the buffers the dispatch marks written; add its
        // images and the indirect arguments ObtainBuffer drops the GPU-written state of.
        auto& split = VideoCommon::GpuSplit::Get();
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            const ImageView& view = texture_cache.GetImageView(id);
            const Image* const image = view.BackingImage();
            if (image == nullptr || view.image_id == VideoCommon::NULL_IMAGE_ID) {
                continue;
            }
            if (image->cpu_addr_end > image->cpu_addr) {
                split.RecordDispatchWrite(image->cpu_addr, image->cpu_addr_end - image->cpu_addr);
            } else {
                split.RecordDispatchUnbounded(VideoCommon::GpuSplit::DispatchUnbounded::Image);
            }
        }
        if (const auto indirect = kepler_compute->GetIndirectComputeAddress()) {
            if (const auto device_addr = gpu_memory->GpuToCpuAddress(*indirect)) {
                split.RecordDispatchWrite(*device_addr, 12);
            }
        }
        split.ConfirmDispatch();
    }
    SCOPE_EXIT {
        if (!frame_dump.Capturing() && !flash_watch.WatchesPasses()) {
            return;
        }
        boost::container::small_vector<ImageView*, 8> written;
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            written.push_back(&texture_cache.GetImageView(id));
        }
        if (frame_dump.Capturing()) {
            frame_dump.CaptureDispatch(pipeline->ShaderHash(), written);
        }
        if (flash_watch.WatchesPasses()) {
            flash_watch.NoteDispatch(pipeline->ShaderHash(), written);
        }
    };
    TracePendingCpuWrites(true, std::array{pipeline->ShaderHash()});
    if (frame_dump.CapturingSampled()) {
        boost::container::small_vector<FrameDump::Sampled, 16> sampled;
        for (const auto& image : pipeline->SampledImages()) {
            sampled.push_back({5, image.tic, &texture_cache.GetImageView(image.id),
                               &texture_cache.GetSampler(image.sampler)});
        }
        frame_dump.NoteDispatchSampled(pipeline->ShaderHash(), sampled);
        FrameDumpBuffers(true, std::array{pipeline->ShaderHash()});
    }

    const auto& qmd{kepler_compute->launch_description};
    // Earlier writes must be visible to the dispatch, its indirect arguments included
    // (RecordDispatchBarrier, recorded at the start of the dispatch's own command), and what it
    // writes to the next render pass (NoteComputeWrites).
    if (frame_dump.CapturingSampled()) {
        std::optional<std::array<u32, 3>> indirect_grid;
        if (indirect_address) {
            indirect_grid = std::array{gpu_memory->Read<u32>(*indirect_address),
                                       gpu_memory->Read<u32>(*indirect_address + 4),
                                       gpu_memory->Read<u32>(*indirect_address + 8)};
        }
        frame_dump.NoteLaunch(pipeline->ShaderHash(),
                              {qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z},
                              {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
                              indirect_address, indirect_grid);
    }
    const auto can_hoist_images = [&]() {
        if (!try_before_pass || !scheduler.IsRenderPassActive() ||
            byte_addressed_shadow.has_value() || buffer_cache.HasPendingNx2ListCopies()) {
            return false;
        }
        for (const auto& entry : pipeline->SampledImages()) {
            if (!entry.id || entry.id == VideoCommon::NULL_IMAGE_VIEW_ID) {
                continue;
            }
            const ImageView& view = texture_cache.GetImageView(entry.id);
            if (!view.IsBuffer() && view.BackingImage() != nullptr &&
                scheduler.RenderPassTouches(view.ImageHandle())) {
                return false;
            }
        }
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            if (!id || id == VideoCommon::NULL_IMAGE_VIEW_ID) {
                continue;
            }
            const ImageView& view = texture_cache.GetImageView(id);
            const Image* const backing = view.BackingImage();
            const VkImage image_handle = view.ImageHandle();
            if (backing == nullptr || scheduler.RenderPassTouches(image_handle)) {
                return false;
            }
            bool freshly_transitioned = false;
            if (tick_before_cfg == tick_after_cfg && records_after_cfg >= records_before_cfg) {
                const u64 max_scan = std::min<u64>(records_after_cfg - records_before_cfg, 256);
                const u64 scan_begin = records_after_cfg - max_scan;
                for (u64 r = scan_begin; r <= records_after_cfg; ++r) {
                    if (backing->JustTransitioned(tick_after_cfg, r)) {
                        freshly_transitioned = true;
                        break;
                    }
                }
                if (!freshly_transitioned && scan_begin > records_before_cfg &&
                    backing->JustTransitioned(tick_after_cfg, records_before_cfg)) {
                    freshly_transitioned = true;
                }
            }
            if (!freshly_transitioned) {
                return false;
            }
        }
        return true;
    };
    const auto leave_render_pass_keep_clear = [&] {
        if (scheduler.DeferredClearFramebufferHandle() == VK_NULL_HANDLE) {
            if (scheduler.IsRenderPassActive()) {
                scheduler.RequestOutsideRenderPassOperationContext();
            }
            return;
        }
        boost::container::small_vector<VkImage, 16> touched;
        for (const auto& entry : pipeline->BoundImages()) {
            if (!entry.id || entry.id == VideoCommon::NULL_IMAGE_VIEW_ID) {
                continue;
            }
            const ImageView& view = texture_cache.GetImageView(entry.id);
            if (!view.IsBuffer() && view.BackingImage() != nullptr) {
                touched.push_back(view.ImageHandle());
            }
        }
        if (byte_addressed_shadow) {
            touched.push_back(byte_addressed_shadow->original);
        }
        scheduler.RequestOutsideRenderPassKeepClear(touched);
    };
    const bool mesh_supported = device.IsExtMeshShaderSupported();
    if (indirect_address) {
        // DispatchIndirect
        static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
        const auto post_op = VideoCommon::ObtainBufferOperation::DoNothing;
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
        buffer->MarkUsage(offset, 12);
        if (can_hoist_images()) {
            if (scheduler.CanRecordBeforePass()) {
                Common::PerfReport::Add(Common::PerfReport::Dispatches);
                scheduler.RecordBeforePass(
                    [pipeline, bind = std::move(deferred_bind), indirect_buffer = buffer->Handle(),
                     indirect_offset = offset, mesh_supported](vk::CommandBuffer cmdbuf) {
                        RecordDispatchBarrier(cmdbuf);
                        if (bind) {
                            bind(cmdbuf);
                        }
                        if (!pipeline->IsBound()) {
                            return;
                        }
                        cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
                        RecordPostDispatchBarrier(cmdbuf, mesh_supported);
                    });
                scheduler.TraceComputePipeline(pipeline->ShaderHash());
                dispatch_chain.Reset();
                NoteWriterDeps(*pipeline);
                return;
            }
            if (scheduler.IsRenderPassActive() && !scheduler.HasDeferredWaitForIdle()) {
                scheduler.WantSplitBeforeCurrentPass();
            }
        }
        NoteDispatchDeps(*pipeline);
        leave_render_pass_keep_clear();
        scheduler.NoteComputeWrites();
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
        const ComputeAccessRange indirect_range{
            buffer->CpuAddr() + offset,
            buffer->CpuAddr() + offset + 12,
        };
        const bool plain_indirect = !byte_addressed_shadow && !want_deferred_bind;
        const bool skip_barrier{
            ShouldSkipDispatchBarrier(*pipeline, plain_indirect, indirect_range)};
        scheduler.Record([this, pipeline, byte_addressed_shadow, skip_barrier,
                          bind = std::move(deferred_bind), mesh_supported,
                          post_barrier = ImmediatePostDispatchBarrierEnabled(device),
                          indirect_buffer = buffer->Handle(),
                          indirect_offset = offset](vk::CommandBuffer cmdbuf) {
            if (!skip_barrier) {
                RecordDispatchBarrier(cmdbuf);
            }
            if (bind) {
                bind(cmdbuf);
            }
            if (!pipeline->IsBound()) {
                return;
            }
            if (byte_addressed_shadow) {
                TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, false);
            }
            scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, {}, [&] {
                cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            });
            if (byte_addressed_shadow) {
                TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, true);
            }
            if (post_barrier) {
                RecordPostDispatchBarrier(cmdbuf, mesh_supported);
            }
        });
        if (want_deferred_bind) {
            scheduler.InvalidateState();
        }
        if (plain_indirect) {
            scheduler.NoteDispatchRecorded(pipeline);
        }
        compute_since_count_barrier = true;
        NoteWriterDeps(*pipeline);
        SubmitNx2ListCopies(pipeline->ShaderHash());
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return;
    }
    if (can_hoist_images()) {
        if (scheduler.CanRecordBeforePass()) {
            Common::PerfReport::Add(Common::PerfReport::Dispatches);
            scheduler.RecordBeforePass(
                [pipeline, dim, bind = std::move(deferred_bind),
                 mesh_supported](vk::CommandBuffer cmdbuf) {
                    RecordDispatchBarrier(cmdbuf);
                    if (bind) {
                        bind(cmdbuf);
                    }
                    if (!pipeline->IsBound()) {
                        return;
                    }
                    cmdbuf.Dispatch(dim[0], dim[1], dim[2]);
                    RecordPostDispatchBarrier(cmdbuf, mesh_supported);
                });
            scheduler.TraceComputePipeline(pipeline->ShaderHash());
            dispatch_chain.Reset();
            NoteWriterDeps(*pipeline);
            TryOcarinaUpscale(*pipeline, native_key, false);
            return;
        }
        if (scheduler.IsRenderPassActive() && !scheduler.HasDeferredWaitForIdle()) {
            scheduler.WantSplitBeforeCurrentPass();
        }
    }
    NoteDispatchDeps(*pipeline);
    leave_render_pass_keep_clear();
    scheduler.NoteComputeWrites();
    if (can_replace || can_bilinear) {
        // Upscaler evaluation may be recorded next, before the dispatch: the barrier goes first.
        scheduler.Record([](vk::CommandBuffer cmdbuf) { RecordDispatchBarrier(cmdbuf); });
    }
    if (marking_effects) {
        const auto written = pipeline->WrittenImages();
        if (std::ranges::any_of(written, [this](VideoCommon::ImageViewId id) {
                return id && texture_cache.GetImageView(id).gpu_addr == oot_effects_address;
            })) {
            oot_live_dlss->MarkEffects(oot_effects_scene, render_trace_frame);
            oot_effects_frame.reset();
        }
    }
    const auto* const replacement =
        can_replace && !byte_addressed_shadow && TryOcarinaUpscale(*pipeline, native_key, true)
            ? oot_live_dlss.get()
            : nullptr;
    if (can_bilinear && !can_replace) {
        CountOcarinaUpscale(false, OcarinaTemporalEnabled() ? "unsupported upscale pass"
                                                            : "no temporal upscaler chosen");
    }
    const bool bilinear =
        !replacement && can_bilinear && !byte_addressed_shadow &&
        TryOcarinaBilinear(*pipeline, native_key);
    if (!replacement && !bilinear) {
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
    } else if (replacement) {
        // Once per pass: which of the game's upscale passes the upscaler took, and with what.
        static std::array<u64, 8> logged{};
        const u64 hash = pipeline->ShaderHash();
        if (std::ranges::find(logged, hash) == logged.end()) {
            if (const auto slot = std::ranges::find(logged, 0ULL); slot != logged.end()) {
                *slot = hash;
            }
            const char* const name =
                oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr
                    ? (replacement->UsedFsr4() ? "FSR 4" : "FSR 3.1")
                    : "the live upscaler";
            LOG_INFO(Render_Vulkan, "Ocarina temporal: {} replaces the game's pass {:016x}", name,
                     hash);
        }
    }
    if (program_id == VideoCommon::OotNetwork::ProgramId &&
        oot_pass != VideoCommon::OcarinaUpscalePass::None &&
        pipeline->CanReplaceOcarinaUpscale()) {
        NoteOcarinaScene(*pipeline, replacement != nullptr, bilinear);
    }
    const bool plain_dispatch{!want_deferred_bind && !byte_addressed_shadow};
    const bool skip_barrier{ShouldSkipDispatchBarrier(*pipeline, plain_dispatch)};
    scheduler.Record([this, pipeline, dim, byte_addressed_shadow, bind = std::move(deferred_bind),
                      barrier_recorded = can_replace || can_bilinear || skip_barrier, replacement,
                      bilinear, mesh_supported,
                      post_barrier =
                          ImmediatePostDispatchBarrierEnabled(device)](vk::CommandBuffer cmdbuf) {
        if (!barrier_recorded) {
            RecordDispatchBarrier(cmdbuf);
        }
        if (bilinear || (replacement && !replacement->EvaluationFailed())) {
            return;
        }
        if (replacement) {
            LOG_WARNING(Render_Vulkan, "Ocarina DLSS evaluation failed; dispatching native upscaler");
        }
        if (bind) {
            bind(cmdbuf);
        }
        if (!pipeline->IsBound()) {
            return;
        }
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, false);
        }
        scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, dim,
                               [&] { cmdbuf.Dispatch(dim[0], dim[1], dim[2]); });
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, true);
        }
        if (post_barrier) {
            RecordPostDispatchBarrier(cmdbuf, mesh_supported);
        }
    });
    if (want_deferred_bind) {
        // The deferred native binding may follow DLSS's command buffer bindings.
        scheduler.InvalidateState();
    }
    if (plain_dispatch) {
        scheduler.NoteDispatchRecorded(pipeline);
    }
    compute_since_count_barrier = true;
    NoteWriterDeps(*pipeline);
    SubmitNx2ListCopies(pipeline->ShaderHash());

    if (!can_replace) {
        TryOcarinaUpscale(*pipeline, native_key, false);
    }

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::NoteWriterDeps(const ComputePipeline& pipeline) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_WRITER_DEPS");
        return value != nullptr && value[0] == '1';
    }();
    if (!enabled || !buffer_cache.Nx2WriterTraceHasWrites()) {
        return;
    }
    struct Stats {
        u64 dispatches{}, sampled{}, gpu_written{}, this_frame{}, with_frame_input{};
        u64 storage_buffers{}, storage_images{}, without_sampled{};
    };
    static std::map<u64, Stats> per_shader;
    static u64 total{};
    auto& stats = per_shader[pipeline.ShaderHash()];
    ++stats.dispatches;
    stats.storage_buffers += pipeline.StorageBufferCount();
    stats.storage_images += pipeline.StorageImageCount();
    stats.without_sampled += pipeline.SampledImages().empty();
    bool frame_input = false;
    for (const auto& image : pipeline.SampledImages()) {
        if (!image.id) {
            continue;
        }
        ++stats.sampled;
        stats.gpu_written += image.gpu_modified;
        stats.this_frame += image.recent_gpu_modified;
        frame_input |= image.recent_gpu_modified;
    }
    stats.with_frame_input += frame_input;
    buffer_cache.Nx2WriterTraceInput(frame_input);
    buffer_cache.Nx2CommitBufferDependencies();
    if ((++total % 2000) == 0) {
        for (const auto& [hash, s] : per_shader) {
            LOG_INFO(Render_Vulkan,
                     "nx2 writer deps: shader {:016x}: {} list-writing dispatches, {} sampled "
                     "bindings ({} GPU-dirty, {} marked after frame boundary); {} dispatches "
                     "sample recent GPU-dirty input; {} with no sampled bindings; {} storage "
                     "buffer / {} storage image bindings (dependencies not classified)",
                     hash, s.dispatches, s.sampled, s.gpu_written, s.this_frame,
                     s.with_frame_input, s.without_sampled, s.storage_buffers, s.storage_images);
        }
    }
}


void RasterizerVulkan::SubmitNx2ListCopies(u64 writer) {
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick(), writer) &&
        (VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
         VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush) ||
         !scheduler.DeferFlush())) {
        VideoCommon::Nx2Profile::FlushOriginTag tag{VideoCommon::Nx2Profile::FlushOrigin::ListCopy};
        scheduler.Flush();
    }
}

bool RasterizerVulkan::FlushSplitIdle(bool requested) {
    bool submitted = false;
    if (buffer_cache.HasPendingNx2ListCopies()) {
        std::scoped_lock lock{buffer_cache.mutex};
        SubmitNx2ListCopies();
        submitted = true;
    }
    VideoCommon::Nx2Profile::FlushOriginTag tag{VideoCommon::Nx2Profile::FlushOrigin::IdleHook};
    return scheduler.FlushDeferredFromIdle(requested) || submitted;
}

VideoCore::RasterizerInterface::GpuWrittenList RasterizerVulkan::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync) {
    if (!VideoCommon::GpuSplit::Get().Active() && buffer_cache.HasPendingNx2ListCopies()) {
        std::scoped_lock lock{buffer_cache.mutex};
        SubmitNx2ListCopies();
    }
    // No cache lock: the front only reads the copies, under their own lock.
    switch (buffer_cache.Nx2ReadListCopy(addr, out, past_sync)) {
    case BufferCache::Nx2ListRead::Copied:
        return GpuWrittenList::Copied;
    case BufferCache::Nx2ListRead::Synced:
        return GpuWrittenList::Synced;
    case BufferCache::Nx2ListRead::NeedsSync:
        return GpuWrittenList::NeedsSync;
    default:
        return GpuWrittenList::Unknown;
    }
}

void RasterizerVulkan::NoteGpuWrittenCommandListSynced(DAddr page) {
    buffer_cache.Nx2NoteListSynced(page);
}

void RasterizerVulkan::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    gpu.InvalidateGPUCache();
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2FlushListPages(pages);
}

void RasterizerVulkan::FlushQueryResults(DAddr addr, u64 size) {
    if (addr == 0 || size == 0 || !query_cache.SemiFlushRegion(addr, size)) {
        return;
    }
    WaitQueryResultsFence(addr);
}

bool RasterizerVulkan::WaitQueryResultsFence(DAddr addr) {
    // Below High GPU accuracy a fence runs its callback at once, so wait for an operation queued
    // behind the pending report writes instead. On the split's back the fence's submission may be
    // deferred, and the fencing thread would wait for this thread to submit it.
    // The state outlives this call: if the wait gives up, the operation may still run later.
    struct WaitState {
        std::mutex mutex;
        std::condition_variable cv;
        bool written{};
    };
    const auto state = std::make_shared<WaitState>();
    fence_manager.SyncOperation([state] {
        std::scoped_lock lock{state->mutex};
        state->written = true;
        state->cv.notify_all();
    });
    fence_manager.SignalFence([] {});
    scheduler.FlushDeferred();
    // Never hang the GPU thread on it: wait in slices, submitting deferred work again, and give up
    // after 2 s (the read may then see older report values).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    std::unique_lock lock{state->mutex};
    while (!state->cv.wait_for(lock, std::chrono::milliseconds(5),
                               [&state] { return state->written; })) {
        if (std::chrono::steady_clock::now() >= deadline) {
            LOG_WARNING(Render_Vulkan, "FlushQueryResults: gave up waiting for query results "
                                       "at {:#x} after 2 s",
                        addr);
            return false;
        }
        lock.unlock();
        scheduler.FlushDeferred();
        lock.lock();
    }
    return true;
}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & (VideoCommon::CacheType::TextureCache | VideoCommon::CacheType::BufferCache))) {
        // CPU writes into GPU-written memory are only collected when they happen. Apply them
        // first, or the download writes the stale GPU copy over them: the Yoshi demo writes a
        // native program over memory a compute pass had bound as a written texel buffer, and its
        // title menu or background read as zeros once the shader cache's read flushed it.
        gpu.InvalidateGPUCache();
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

namespace {
/// EDEN_NX2_TEXTURE_READ_FILTER: the texture cache only flushes or reports GPU-modified images, so
/// queries over pages no image was ever GPU-modified on skip its lock (gpu_modified_filter).
bool TextureReadFilter() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_TEXTURE_READ_FILTER");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}
} // Anonymous namespace

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if ((True(which & VideoCommon::CacheType::BufferCache)) && MayBeGpuModified(addr, size)) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!VideoCore::Nx2Debug::AccuratePart(4)) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        if (TextureReadFilter() && !texture_cache.MayHaveGpuModifiedImages(addr, size)) {
            return false;
        }
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

bool RasterizerVulkan::MayBeGpuModified(DAddr addr, u64 size) {
    // Every range the buffer cache records as GPU-modified also marks its pages GPU-written
    // (bits that never clear), so pages without the bit need no look under its mutex. The split's
    // front asks this before each macro DMA read; the back holds the mutex through each draw.
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return true;
    }
    const DAddr last = addr + size - 1;
    for (DAddr page = addr >> Tegra::Nx2WrittenPageBits; page <= (last >> Tegra::Nx2WrittenPageBits);
         ++page) {
        if (Tegra::Nx2IsGpuWritten(page << Tegra::Nx2WrittenPageBits)) {
            return true;
        }
    }
    return false;
}

bool RasterizerVulkan::HasPendingQueries(DAddr addr, u64 size) {
    return query_cache.IsRegionGpuModified(addr, size);
}

bool RasterizerVulkan::HasCachedReports(DAddr addr, u64 size) {
    return query_cache.HasCachedReports(addr, size);
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    static const bool fast_path = [] {
        const char* value = std::getenv("EDEN_NX2_READ_FASTPATH");
        return value == nullptr || value[0] != '0';
    }();
    // Nor where no image was ever GPU-modified (EDEN_NX2_TEXTURE_READ_FILTER).
    const bool skip_textures =
        fast_path && (!texture_cache.page_filter.Any(addr, size) ||
                      (TextureReadFilter() && !texture_cache.MayHaveGpuModifiedImages(addr, size)));
    const bool skip_buffers = fast_path && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
                              !device_memory.MayHaveArmedReadTraps(addr, size) &&
                              !Tegra::Nx2MayBeGpuWritten(addr, size);
    namespace ReadStats = VideoCommon::CpuReadStats;
    if (!skip_textures) {
        const u64 lock_begin = ReadStats::NowNs();
        std::scoped_lock lock{texture_cache.mutex};
        ReadStats::Add(ReadStats::g_counters.texture_lock_ns, ReadStats::NowNs() - lock_begin);
        ReadStats::Add(ReadStats::g_counters.texture_asks, 1);
        auto area = texture_cache.GetFlushArea(addr, size);
        if (area) {
            area->cause = static_cast<u8>(ReadStats::Cause::Image);
            return *area;
        }
    }
    if (!skip_buffers && buffer_cache.FlushesOnCpuRead()) {
        if (fast_path) {
            if (const auto fast_area = buffer_cache.TryFastGetFlushArea(addr, size)) {
                return *fast_area;
            }
        }
        const u64 lock_begin = ReadStats::NowNs();
        std::scoped_lock lock{buffer_cache.mutex};
        ReadStats::Add(ReadStats::g_counters.buffer_lock_ns, ReadStats::NowNs() - lock_begin);
        ReadStats::Add(ReadStats::g_counters.buffer_asks, 1);
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

bool RasterizerVulkan::IsCleanReadTrapPage(DAddr page_addr) {
    if (texture_cache.page_filter.Any(page_addr, Core::DEVICE_PAGESIZE) &&
        (!TextureReadFilter() ||
         texture_cache.MayHaveGpuModifiedImages(page_addr, Core::DEVICE_PAGESIZE))) {
        return false;
    }
    return buffer_cache.IsPageCleanForFastRead(page_addr);
}

std::optional<VideoCore::RasterizerDownloadArea> RasterizerVulkan::TryFastCleanRead(
    DAddr page_begin, u64 page_bytes) {
    if (texture_cache.page_filter.Any(page_begin, page_bytes) &&
        (!TextureReadFilter() ||
         texture_cache.MayHaveGpuModifiedImages(page_begin, page_bytes))) {
        return std::nullopt;
    }
    return buffer_cache.TryFastGetFlushAreaForDisarmed(page_begin, page_bytes);
}

bool RasterizerVulkan::LandDownloadForCpuRead(DAddr addr, u64 size, u64 tick) {
    // Only the thread that records may submit: a download still to be submitted goes to it.
    if (!scheduler.IsSubmitted(tick)) {
        return false;
    }
    // As the fencing thread: wait for the submission, then land under the cache's lock.
    scheduler.GetMasterSemaphore().Wait(tick);
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.LandInFlightForCpuRead(addr, size, false);
    // The GPU may have written the range again meanwhile.
    return !buffer_cache.IsRegionGpuModified(addr, size);
}

void RasterizerVulkan::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0 || False(which & VideoCommon::CacheType::BufferCache)) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.ClearDownload(addr, size);
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    texture_cache.NoteGuestWrite();
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    // The GPU's own cached writes (constant buffer data, every draw): most land where no image,
    // query or shader was ever registered, which page_filter tells without the caches' locks.
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        for (const auto& [addr, size] : sequences) {
            if (filter.Any(addr, size)) {
                return true;
            }
        }
        return false;
    };
    if (any(texture_cache.page_filter)) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    // Cached textures must observe GPU-side copies even when the remaining invalidations are skipped.
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    {
        VideoCommon::Nx2DrawStats::LockTimer lock_probe{VideoCommon::Nx2DrawStats::Site::LockInvalidation};
        std::scoped_lock lock{buffer_cache.mutex};
        lock_probe.Acquired();
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    if (any(query_cache.page_filter)) {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

namespace {
/// Guest writes of cached memory collected without the caches' mutexes (EDEN_NX2_EXP35) and those
/// that took them, logged every 5 s with EDEN_NX2_GPU_PROFILE.
struct GuestWriteCounters {
    std::atomic<u64> collected{};
    std::atomic<u64> locked{};
    std::atomic<s64> last_log{};
};
GuestWriteCounters g_guest_writes;

void CountGuestWrite(std::atomic<u64>& counter) {
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return;
    }
    const u64 count = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((count & 1023) != 0) {
        return;
    }
    auto& c = g_guest_writes;
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = c.last_log.load(std::memory_order_relaxed);
    if (now - last < 5000 || !c.last_log.compare_exchange_strong(last, now)) {
        return;
    }
    LOG_INFO(Render_Vulkan, "Guest writes of cached memory (5 s): {} collected unasked, {} locked",
             c.collected.exchange(0), c.locked.exchange(0));
}
} // Anonymous namespace

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    texture_cache.NoteGuestWrite();
    DEBUG_ASSERT(addr != 0 || size != 0);
    // EDEN_NX2_EXP35 (live; from Fritchii v5.5, a34e72fec1/e423d1bd17): collect the write for the
    // next InvalidateGPUCache without either cache's mutex, which the split's back holds through
    // each draw (guest cores spent 16% of their busy time spinning on them at Castle Town).
    // 1 = pages no GPU write ever marked, 2 = every page. A GPU-modified range's write is
    // collected anyway; a clean one's WriteMemory becomes the same marks when the collected writes
    // are applied (fences, cache invalidations, every flush), and downloads keep its bytes
    // (DownloadLanding). Not pages of shader code or images (applied at once, as before): image
    // downloads don't look at collected writes, and an image rendered to before they are applied
    // would be uploaded again from the older guest bytes.
    if (const int collect = VideoCore::Nx2Debug::Experiment(35);
        collect != 0 && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        (collect == 2 || !Tegra::Nx2MayBeGpuWritten(addr, size)) &&
        !pipeline_cache.page_filter.Any(addr, size) && !texture_cache.page_filter.Any(addr, size)) {
        CountGuestWrite(g_guest_writes.collected);
        return true;
    }
    CountGuestWrite(g_guest_writes.locked);
    static const bool fast_path = [] {
        const char* value = std::getenv("EDEN_NX2_WRITE_FASTPATH");
        return value == nullptr || value[0] != '0';
    }();
    if (fast_path && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        !texture_cache.page_filter.Any(addr, size) &&
        !pipeline_cache.page_filter.Any(addr, size) &&
        buffer_cache.TryFastOnCpuWrite(addr, size)) {
        Tegra::Nx2NoteDeferredCodeWrite(addr, size);
        return true;
    }
    const bool handoff = fast_path && VideoCommon::Nx2WriteHandoff::Enabled() &&
                         Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
                         !Tegra::Nx2MayBeGpuWritten(addr, size) &&
                         !texture_cache.page_filter.Any(addr, size) &&
                         !pipeline_cache.page_filter.Any(addr, size);
    if (handoff && !buffer_cache.mutex.try_lock()) {
        if (write_handoff.Push(addr, size)) {
            if (Tegra::Nx2MayBeGpuWritten(addr, size)) {
                std::scoped_lock lock{buffer_cache.mutex};
            }
            return false;
        }
        buffer_cache.mutex.lock();
    } else if (!handoff) {
        buffer_cache.mutex.lock();
    }
    // EDEN_NX2_PROTECT_UNLOCKED (zortos293/Lichii#1518): the pages the write untracks get their
    // protection changed (a system call each run) after the lock is dropped, not while every
    // other drawing or reading thread waits for it.
    static thread_local VideoCommon::DeferredPageCaching deferred;
    const bool defer = VideoCommon::ProtectUnlockedEnabled() &&
                       VideoCommon::t_deferred_page_caching == nullptr &&
                       !VideoCommon::GpuSplit::OwnsScheduler();
    bool gpu_modified;
    {
        std::unique_lock lock{buffer_cache.mutex, std::adopt_lock};
        if (defer) {
            VideoCommon::t_deferred_page_caching = &deferred;
        }
        gpu_modified = buffer_cache.OnCPUWrite(addr, size);
        if (defer) {
            VideoCommon::t_deferred_page_caching = nullptr;
            if (!deferred.Empty()) {
                device_memory.BeginDeferredCachedUpdates();
            }
        }
    }
    if (defer && !deferred.Empty()) {
        deferred.ApplyAll();
        device_memory.EndDeferredCachedUpdates();
    }
    if (gpu_modified) {
        // The shader cache sees this write later (InvalidateGPUCache): proofs of programs there
        // must stop now, before the guest submits work that uses the new code.
        Tegra::Nx2NoteDeferredCodeWrite(addr, size);
        return true;
    }
    if (!fast_path || texture_cache.page_filter.Any(addr, size)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (!fast_path || pipeline_cache.page_filter.Any(addr, size)) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
    return false;
}

void RasterizerVulkan::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    texture_cache.NoteGuestWrite();
    // Out of the GPU-written ranges first: applying one write downloads the GPU's bytes elsewhere
    // on its pages (WriteMemory), which would write over the next collected write.
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ClaimCpuWrites(ranges);
    }
    // Each lock once for the batch, and the texture and shader caches only where they ever held
    // something (page_filter): with EDEN_NX2_EXP35 most guest writes of cached memory come here.
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        return std::ranges::any_of(ranges, [&](const std::pair<DAddr, u64>& range) {
            return range.first != 0 && filter.Any(range.first, range.second);
        });
    };
    if (any(texture_cache.page_filter)) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : ranges) {
            if (addr != 0 && size != 0) {
                texture_cache.WriteMemory(addr, size);
            }
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : ranges) {
            if (addr != 0 && size != 0) {
                buffer_cache.WriteMemory(addr, size);
            }
        }
    }
    if (any(pipeline_cache.page_filter)) {
        for (const auto& [addr, size] : ranges) {
            if (addr != 0 && size != 0) {
                pipeline_cache.InvalidateRegion(addr, size);
            }
        }
    }
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    texture_cache.NoteGuestWrite();
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        // WriteMemory downloads GPU-written bytes elsewhere on the written pages, which only the
        // GPU thread may do: leave those pages to the fallback.
        const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func, bool force_completion) {
    fence_manager.SignalFence(std::move(func), force_completion);
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

bool RasterizerVulkan::FencesQuiet(bool strict, u64& epoch) {
    return fence_manager.FencesQuiet(strict, epoch);
}

u64 RasterizerVulkan::FenceWorkEpoch() const {
    return fence_manager.FenceWorkEpoch();
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (VideoCore::Nx2Debug::AccuratePart(8)) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (!device.IsGeometryShaderSupported()) {
        flags &= ~VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    }
    if (!device.IsTessellationSupported()) {
        flags &= ~(VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
                   VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT);
    }
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }
    if (device.IsExtMeshShaderSupported()) {
        flags |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
    }

    query_cache.NotifyWFI();

    static const bool wfi_defer = [] {
        const char* value = std::getenv("EDEN_NX2_WFI_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    // Inside a pure rasterization render pass with no unordered storage writes or transform
    // feedback, EndRenderPass and CheckFeedbackLoop already order attachment writes (as Metal's
    // WaitForIdle does); keeping deferred_wfi_stages clear leaves before_pass_cmdbuf open for
    // subsequent buffer and image uploads instead of breaking and reopening the pass.
    if (wfi_defer && scheduler.IsRenderPassActive() && !scheduler.StorageWritesUnordered() &&
        !scheduler.RenderPassHasStores() && maxwell3d &&
        maxwell3d->regs.transform_feedback_enabled == 0) {
        nx2_wfi_dispatch_serial = nx2_dispatch_serial;
        scheduler.DeferWaitForIdle(flags);
        fence_manager.SignalOrdering();
        return;
    }
    nx2_wfi_dispatch_serial = nx2_dispatch_serial;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.NoteStorageWritesOrdered();
    scheduler.Record([flags](vk::CommandBuffer cmdbuf) {
        // Later work waits for the earlier work and sees its writes.
        static constexpr VkMemoryBarrier WFI_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(flags, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WFI_BARRIER);
    });
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::SignalOrdering};
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    // The next draw breaks the render pass only if it reads what this one wrote (#161): Ocarina
    // ended ~5,900 render passes a second here.
    scheduler.DeferFragmentBarrier();
}

void RasterizerVulkan::TiledCacheBarrier() {
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    if ((draw_counter == 0 && !pending_native_work) || !scheduler.HasPendingCommands()) {
        draw_counter = 0;
        pending_native_work = false;
        return;
    }
    draw_counter = 0;
    pending_native_work = false;
    // Fences with downloads or queries come here right after queueing: ~9,400 a second in
    // Hyrule Warriors: Age of Imprisonment fights. Batched like the fences themselves.
    if (!scheduler.DeferFlush()) {
        VideoCommon::Nx2Profile::FlushOriginTag tag{VideoCommon::Nx2Profile::FlushOrigin::Fence};
        scheduler.Flush();
    }
}

void RasterizerVulkan::TraceRenderState(bool is_indexed, const char* outcome,
                                        const Framebuffer* framebuffer) {
    if (!VideoCommon::RenderTrace::SampleFrame(render_trace_frame)) {
        return;
    }
    if (render_trace_draws++ >= 2048) {
        if (render_trace_draws == 2049) {
            LOG_DEBUG(Render_Vulkan, "Render trace frame={} truncated after 2048 draws",
                      render_trace_frame);
        }
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    std::array<u64, 6> shaders{};
    u32 stages{};
    for (size_t stage = 0; stage < shaders.size(); ++stage) {
        shaders[stage] = (u64{regs.pipelines[stage].reserved[0] & 0xff} << 32) |
                         regs.pipelines[stage].reserved[1];
        stages |= static_cast<u32>(regs.IsShaderConfigEnabled(stage)) << stage;
    }
    const auto extent = framebuffer ? framebuffer->RenderArea() : VkExtent2D{};
    LOG_DEBUG(Render_Vulkan,
              "Render trace frame={} draw={} docked={} outcome={} indexed={} stages={:#x} "
              "shaders={:x} guest_count={} guest_instances={} framebuffer={}x{} layers={}",
              render_trace_frame, render_trace_draws, Settings::IsDockedMode(), outcome, is_indexed,
              stages, fmt::join(shaders, ","),
              is_indexed ? draw.index_buffer.count : draw.vertex_buffer.count, draw.instance_count,
              extent.width, extent.height, framebuffer ? framebuffer->Layers() : 0);
    LOG_DEBUG(Render_Vulkan,
              "Render trace targets: color={:#x} {}x{} layers={} format={:#x} depth={:#x} "
              "{}x{} layers={} format={:#x} enabled={} clip={}x{}",
              regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height, regs.rt[0].depth.Value(),
              static_cast<u32>(regs.rt[0].format), regs.zeta.Address(), regs.zeta_size.width,
              regs.zeta_size.height, regs.zeta_size.depth.Value(), static_cast<u32>(regs.zeta.format),
              regs.zeta_enable, regs.surface_clip.width.Value(), regs.surface_clip.height.Value());
    const auto& viewport = regs.viewport_transform[0];
    const auto& scissor = regs.scissor_test[0];
    LOG_DEBUG(Render_Vulkan,
              "Render trace tests: raster={} depth_test={} depth_write={} depth_func={} "
              "cull={} face={} alpha={} func={} ref={} color_mask={:#x} "
              "viewport_scale=({},{},{}) translate=({},{},{}) scissor={} {}..{} {}..{}",
              regs.rasterize_enable, regs.depth_test_enable, regs.depth_write_enabled,
              static_cast<u32>(regs.depth_test_func), regs.gl_cull_test_enabled,
              static_cast<u32>(regs.gl_cull_face), regs.alpha_test_enabled,
              static_cast<u32>(regs.alpha_test_func), regs.alpha_test_ref, regs.color_mask[0].raw,
              viewport.scale_x, viewport.scale_y, viewport.scale_z, viewport.translate_x,
              viewport.translate_y, viewport.translate_z, scissor.enable, scissor.min_x.Value(),
              scissor.max_x.Value(), scissor.min_y.Value(), scissor.max_y.Value());
    LOG_DEBUG(Render_Vulkan,
              "Render trace bias: topology={} raster_topology={} offset_point={} offset_line={} "
              "offset_fill={} units={} slope={} clamp={} control={:#x}",
              static_cast<u32>(draw.topology),
              static_cast<u32>(RasterTopologyOf(regs, draw.topology)),
              regs.polygon_offset_point_enable, regs.polygon_offset_line_enable,
              regs.polygon_offset_fill_enable, regs.depth_bias, regs.slope_scale_depth_bias,
              regs.depth_bias_clamp, regs.depth_bias_control);
    for (size_t binding = 0; binding < regs.vertex_streams.size(); ++binding) {
        const auto& stream = regs.vertex_streams[binding];
        if (stream.IsEnabled()) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace vertex: binding={} address={:#x} limit={:#x} stride={} "
                      "frequency={} repacked={}",
                      binding, stream.Address(), regs.vertex_stream_limits[binding].Address(),
                      stream.stride.Value(), stream.frequency,
                      buffer_cache_runtime.IsVertexRepacked(static_cast<u32>(binding)));
        }
    }
    if (is_indexed) {
        LOG_DEBUG(Render_Vulkan, "Render trace index: address={:#x} limit={:#x} format={} first={}",
                  regs.index_buffer.StartAddress(), regs.index_buffer.EndAddress(),
                  static_cast<u32>(regs.index_buffer.format), draw.index_buffer.first);
    }
    LOG_DEBUG(Render_Vulkan, "Render trace condition: address={:#x} mode={} override={}",
              regs.render_enable.Address(), static_cast<u32>(regs.render_enable.mode),
              static_cast<u32>(regs.render_enable_override));
}

void RasterizerVulkan::TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes) {
    // Guest CPU writes into GPU-written buffer memory are collected, and applied to the host
    // buffers only at the next fence or cache invalidation. A hit here reads the older bytes.
    static const bool enabled = std::getenv("EDEN_NX2_PENDING_WRITE_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto*, u32) {
        if (lines > max_lines || !gpu.HasPendingCPUWrites(binding.device_addr, bound_size)) {
            return;
        }
        if (++lines > max_lines) {
            LOG_INFO(HW_GPU, "Pending write trace stopped after {} lines", max_lines);
            return;
        }
        const size_t program = compute ? 0 : stage + 1;
        LOG_INFO(HW_GPU,
                 "Pending CPU write frame={} {} shader={:016x} stage={} {}[{}]{} addr={:#x} "
                 "device={:#x} size={:#x}",
                 render_trace_frame, compute ? "dispatch" : "draw",
                 program < shader_hashes.size() ? shader_hashes[program] : 0, stage,
                 storage ? "ssbo" : "cbuf", index, written ? " written" : "", binding.gpu_addr,
                 binding.device_addr, bound_size);
    });
}

bool RasterizerVulkan::OcarinaTemporalEnabled() const {
    constexpr u64 ocarina_program = 0x0400486024DEC000ULL;
    if (program_id != ocarina_program || oot_live_failed) {
        return false;
    }
    if (!oot_backend) {
        oot_backend = ResolveOcarinaUpscaler(device);
        if (*oot_backend == VideoCommon::OcarinaUpscaleBackend::Native &&
            Settings::OcarinaFsrChosen() &&
            !Fsr3Upscaler::IsSupported(device)) {
            LOG_WARNING(Render_Vulkan,
                        "Ocarina FSR 3.1 unsupported on this device; using native upscale");
        }
    }
    return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
}

/// Capture the two camera matrices from a draw targeting the scene color and depth. Terrain and
/// materials draw with several vertex shaders, so any draw into a scene-sized target qualifies;
/// the camera buffer's contents are checked below.
void RasterizerVulkan::TrackOcarinaCamera() {
    if ((oot_camera_sample_frame == render_trace_frame && oot_motion.extent.width >= 2560 &&
         oot_motion.extent.height >= 1440) ||
        !OcarinaTemporalEnabled()) {
        return;
    }

    const auto& targets = texture_cache.GetRenderTargets();
    if (!targets.color_buffer_ids[0] || !targets.depth_buffer_id) {
        return;
    }
    const auto& color = texture_cache.GetImageView(targets.color_buffer_ids[0]);
    if (color.size.width < 640 || color.size.height < 360 ||
        color.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
        return;
    }
    // Reflections and half-size passes share the camera layout; the upscale reads the largest.
    if (oot_camera_sample_frame == render_trace_frame &&
        u64{color.size.width} * color.size.height <=
            u64{oot_motion.image_extent.width} * oot_motion.image_extent.height) {
        return;
    }

    const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[3];
    constexpr size_t camera_bytes = 0x230;
    if (!cbuf.enabled || cbuf.size < camera_bytes) {
        return;
    }
    const auto device_addr = gpu_memory->GpuToCpuAddress(cbuf.address);
    if (!device_addr || buffer_cache.IsRegionGpuModified(*device_addr, camera_bytes)) {
        return;
    }
    std::array<f32, 2> extent_check{};
    std::array<f32, 2> proj_check{};
    device_memory.ReadBlockUnsafe(*device_addr + 0x120, extent_check.data(), sizeof(extent_check));
    device_memory.ReadBlockUnsafe(*device_addr + 0xa8, proj_check.data(), sizeof(proj_check));
    if (proj_check[0] != -1.0f || proj_check[1] != 0.0f || !(extent_check[0] >= 640.0f) ||
        !(extent_check[1] >= 360.0f) || extent_check[0] > 16384.0f || extent_check[1] > 16384.0f) {
        return;
    }
    // The camera's extent is the scene's. The images are that size, or up to 15 pixels larger at
    // a render scale the game pads (oot_resolution 87 %); only NVIDIA DLSS takes those.
    const VkExtent2D scene{static_cast<u32>(extent_check[0]), static_cast<u32>(extent_check[1])};
    const bool padded = scene.width != color.size.width || scene.height != color.size.height;
    if (static_cast<f32>(scene.width) != extent_check[0] ||
        static_cast<f32>(scene.height) != extent_check[1] || color.size.width < scene.width ||
        color.size.height < scene.height || color.size.width - scene.width >= 16 ||
        color.size.height - scene.height >= 16 ||
        (padded && oot_backend != VideoCommon::OcarinaUpscaleBackend::Dlss)) {
        return;
    }
    oot_camera_sample_frame = render_trace_frame;
    oot_motion.camera_channel = gpu_memory;
    oot_motion.extent = scene;
    oot_motion.image_extent = {color.size.width, color.size.height};
    device_memory.ReadBlockUnsafe(*device_addr + 0x30, oot_motion.current.data(),
                                  sizeof(oot_motion.current));
    device_memory.ReadBlockUnsafe(*device_addr + 0x180, oot_motion.previous.data(),
                                  sizeof(oot_motion.previous));
}

void RasterizerVulkan::UpdateOcarinaJitter() {
    if (!oot_dlss_frame) {
        return;
    }
    std::array<f32, 2> want{};
    // Only while the upscaler replaced the previous frame's upscale: a frame the game upscales
    // itself would show the offset as a shake. Only draws that depth-test against the scene depth
    // at the scene size, so a depth pre-pass and the colour passes move together; shadow maps,
    // half-size passes and full-screen passes (which keep the depth bound without using it, some
    // with depth writes on: with the test off nothing writes depth) stay put.
    const auto& regs = maxwell3d->regs;
    if (*oot_dlss_frame + 1 >= render_trace_frame && regs.depth_test_enable &&
        oot_motion.extent.width != 0 && oot_backend &&
        VideoCommon::OcarinaJitterEnabled(*oot_backend)) {
        const auto& targets = texture_cache.GetRenderTargets();
        if (targets.depth_buffer_id) {
            const auto& depth = texture_cache.GetImageView(targets.depth_buffer_id);
            if (oot_motion.SceneSized(depth.size.width, depth.size.height) && !depth.IsRescaled()) {
                // AMD's phase count for the last upscaled frame's scale.
                const u32 index = static_cast<u32>(render_trace_frame % oot_jitter_phases) + 1;
                const auto driver = device.GetDriverID();
                want = VideoCommon::OcarinaJitter(
                    index, driver == VK_DRIVER_ID_AMD_PROPRIETARY ||
                               driver == VK_DRIVER_ID_AMD_OPEN_SOURCE);
                oot_jitter_frame = render_trace_frame;
                oot_jitter_offset = want;
            }
        }
    }
    if (want != oot_jitter_applied) {
        oot_jitter_applied = want;
        state_tracker.InvalidateViewports();
    }
}

/// Capture the skinning pose of the recognized Link draw for frame-to-frame motion.
void RasterizerVulkan::TrackOcarinaActor(u64 shader_hash, bool is_indexed, bool is_indirect) {
    if (shader_hash != 0x28c6e527634f814aULL || !is_indexed || is_indirect ||
        !OcarinaTemporalEnabled() ||
        oot_actor_poses[oot_actor_pose_index].frame == render_trace_frame) {
        return;
    }
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const auto& regs = maxwell3d->regs;
    const auto& stream = regs.vertex_streams[0];
    if (draw.index_buffer.count != 9702 || draw.index_buffer.FormatSizeInBytes() != 2 ||
        !stream.IsEnabled() || stream.stride != 48 || regs.vertex_stream_instances.IsInstancingEnabled(0)) {
        return;
    }
    const auto attribute = [&](u32 location, VkFormat format, u32 offset) {
        const auto& attrib = regs.vertex_attrib_format[location];
        return attrib.buffer == 0 && !attrib.constant && attrib.offset == offset &&
            MaxwellToVK::VertexFormat(device, attrib.type, attrib.size) == format;
    };
    if (!attribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0) ||
        !attribute(4, VK_FORMAT_R8G8B8A8_UNORM, 40) || !attribute(5, VK_FORMAT_R8G8B8A8_UNORM, 44) ||
        !attribute(6, VK_FORMAT_R16G16B16A16_UINT, 12) || !attribute(7, VK_FORMAT_R16G16B16A16_UINT, 20)) {
        return;
    }

    const auto& cbufs = maxwell3d->state.shader_stages[0].const_buffers;
    if (!cbufs[4].enabled || cbufs[4].size < 0x40 + 3 * sizeof(f32) ||
        !cbufs[6].enabled || cbufs[6].size != 13568) {
        return;
    }
    const auto origin_dev = gpu_memory->GpuToCpuAddress(cbufs[4].address);
    const auto bones_dev = gpu_memory->GpuToCpuAddress(cbufs[6].address);
    if (!origin_dev || !bones_dev ||
        buffer_cache.IsRegionGpuModified(*origin_dev, cbufs[4].size) ||
        buffer_cache.IsRegionGpuModified(*bones_dev, cbufs[6].size)) {
        return;
    }
    oot_actor_pose_index ^= 1U;
    auto& pose = oot_actor_poses[oot_actor_pose_index];
    pose.frame = render_trace_frame;
    pose.channel = gpu_memory;
    pose.vertex = stream.Address();
    pose.indices = draw.index_buffer.IndexStart();
    pose.base_vertex = static_cast<s32>(draw.base_index);
    pose.origin = {};
    device_memory.ReadBlockUnsafe(*origin_dev + 0x40, pose.origin.data(), 3 * sizeof(f32));
    device_memory.ReadBlockUnsafe(*bones_dev, pose.bones.data(), sizeof(pose.bones));
}

/// Read the shared mesh only when both poses refer to the same vertex and index buffers.
void RasterizerVulkan::ReadOcarinaActor(OcarinaActorMesh& mesh) {
    const auto& now = oot_actor_poses[oot_actor_pose_index];
    const auto& before = oot_actor_poses[oot_actor_pose_index ^ 1U];
    if (now.frame != render_trace_frame || before.frame + 1 != now.frame ||
        now.channel != gpu_memory || before.channel != now.channel || now.vertex != before.vertex ||
        now.indices != before.indices || now.base_vertex != before.base_vertex) {
        return;
    }
    auto& cache = oot_actor_mesh_cache;
    const auto can_stay_cached = [&](GPUVAddr address, size_t size) {
        const auto device_address = gpu_memory->GpuToCpuAddress(address);
        return device_address && !buffer_cache.IsRegionGpuModified(*device_address, size) &&
               !gpu.HasPendingCPUWrites(*device_address, size);
    };
    constexpr size_t index_count = 9702;
    constexpr size_t stride = 48;
    if (cache.channel != now.channel || cache.vertex != now.vertex ||
        cache.indices != now.indices || cache.base_vertex != now.base_vertex ||
        cache.index_data.size() != index_count || cache.vertices.empty() ||
        !can_stay_cached(now.indices, index_count * sizeof(u16))) {
        const auto read = [&](GPUVAddr address, void* dst, size_t size) {
            if (!gpu_memory->IsContinuousRange(address, size) ||
                !gpu_memory->IsFullyMappedRange(address, size) ||
                !can_stay_cached(address, size)) {
                return false;
            }
            gpu_memory->ReadBlockUnsafe(address, dst, size);
            return true;
        };
        cache.index_data.resize(index_count);
        if (!read(now.indices, cache.index_data.data(), cache.index_data.size() * sizeof(u16))) {
            cache.vertices.clear();
            return;
        }
        const auto [minimum, maximum] = std::ranges::minmax(cache.index_data);
        const s64 first = static_cast<s64>(now.base_vertex) + minimum;
        const u32 vertex_count = static_cast<u32>(maximum) - minimum + 1;
        if (first < 0 || first > (1 << 20) || vertex_count > 16384) {
            cache.vertices.clear();
            return;
        }
        cache.vertices.resize(vertex_count * stride);
        if (!read(now.vertex + first * stride, cache.vertices.data(), cache.vertices.size())) {
            cache.vertices.clear();
            return;
        }
        for (u16& index : cache.index_data) {
            index -= minimum;
        }
        cache.channel = now.channel;
        cache.vertex = now.vertex;
        cache.indices = now.indices;
        cache.base_vertex = now.base_vertex;
    }

    mesh = {
        .vertices = cache.vertices,
        .indices = cache.index_data,
        .vertex_address = now.vertex,
        .index_address = now.indices,
        .base_vertex = now.base_vertex,
        .current_origin = now.origin,
        .previous_origin = before.origin,
        .current_bones = now.bones,
        .previous_bones = before.bones,
    };
}

// A frame the game's pass draws between upscaled ones shows its own anti-aliasing and the
// upscaler's history starts again: the log shows how often that happens.
void RasterizerVulkan::CountOcarinaUpscale(bool upscaled, const char* reason) {
    if (oot_upscaled_passes + oot_game_passes != 0 && upscaled != oot_last_upscaled) {
        ++oot_switches;
    }
    oot_last_upscaled = upscaled;
    ++(upscaled ? oot_upscaled_passes : oot_game_passes);
    if (reason) {
        for (auto& [name, count] : oot_decline_reasons) {
            if (name == nullptr || name == reason) {
                name = reason;
                ++count;
                break;
            }
        }
    }
    if (oot_upscaled_passes + oot_game_passes < 600) {
        return;
    }
    std::string reasons;
    for (const auto& [name, count] : oot_decline_reasons) {
        if (name != nullptr) {
            reasons += fmt::format("{}{} {}", reasons.empty() ? "; declined: " : ", ", name, count);
        }
    }
    if (OcarinaBilinearUpscaleEnabled()) {
        reasons += fmt::format("; {} of the declined drawn bilinear instead of the game's pass",
                               oot_bilinear_passes);
    }
    const VkExtent2D scene = oot_motion.extent.width != 0 ? oot_motion.extent : oot_bilinear_extent;
    LOG_INFO(Common_Memory,
             "Ocarina upscaler: {} of {} upscale passes replaced, {} switches to or from the "
             "game's own; scene {}x{}{}",
             oot_upscaled_passes, oot_upscaled_passes + oot_game_passes, oot_switches,
             scene.width, scene.height, reasons);
    oot_upscaled_passes = oot_game_passes = oot_switches = oot_bilinear_passes = 0;
    oot_decline_reasons = {};
}

/// Track the scene depth on its reader dispatch, or try to replace a recognized upscale dispatch.
bool RasterizerVulkan::TryOcarinaUpscale(const ComputePipeline& pipeline, u64 native_key,
                                         bool replace) {
    const u64 upscale_hash = pipeline.ShaderHash();
    const bool depth_reader = native_key == VideoCommon::OcarinaDepthReaderKey ||
                              upscale_hash == VideoCommon::OcarinaDepthReaderShader;
    if ((!replace && !depth_reader) || !OcarinaTemporalEnabled()) {
        return false;
    }
    const bool native_resolution =
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss &&
        Settings::values.nvidia_dlss_quality.GetValue() == Settings::DlssQuality::Dlaa;
    const auto read_tic = [this](u32 index) -> std::optional<Tegra::Texture::TICEntry> {
        const auto& table = kepler_compute->regs.tic;
        using Entry = Tegra::Texture::TICEntry;
        const GPUVAddr address = table.Address() + u64{index} * sizeof(Entry);
        if (index > table.limit || !gpu_memory->IsFullyMappedRange(address, sizeof(Entry))) {
            return std::nullopt;
        }
        Entry entry{};
        gpu_memory->ReadBlockUnsafe(address, &entry, sizeof(entry));
        return entry;
    };
    if (!replace) {
        constexpr u32 depth_handle_offset = 0x128;
        const auto& qmd = kepler_compute->launch_description;
        const auto& cbuf = qmd.const_buffer_config[0];
        if (!(qmd.const_buffer_enable_mask & 1) || cbuf.size < depth_handle_offset + sizeof(u32) ||
            !gpu_memory->IsFullyMappedRange(cbuf.Address(), depth_handle_offset + sizeof(u32))) {
            return false;
        }
        const u32 index = gpu_memory->Read<u32>(cbuf.Address() + depth_handle_offset) & 0xfffff;
        const auto tic = read_tic(index);
        const bool scene_size = oot_camera_sample_frame != render_trace_frame ||
                                (tic && oot_motion.SceneSized(tic->Width(), tic->Height()));
        if (tic && tic->Address() && scene_size) {
            oot_motion.depth_channel = gpu_memory;
            oot_motion.depth_frame = render_trace_frame;
            oot_motion.depth_tic = index;
            oot_motion.depth_address = tic->Address();
        }
        return false;
    }
    // Logs why the game's own upscale draws, when the reason changes and every 120th repeat.
    const auto fail = [this](const char* reason) {
        CountOcarinaUpscale(false, reason);
        if (reason != oot_fallback_reason || ++oot_fallback_repeats % 120 == 0) {
            LOG_INFO(Render_Vulkan, "Ocarina DLSS: the game's upscale draws frame {} ({})",
                     render_trace_frame, reason);
        }
        if (reason != oot_fallback_reason) {
            oot_fallback_reason = reason;
            oot_fallback_repeats = 0;
        }
        return false;
    };
    if (oot_camera_sample_frame != render_trace_frame) {
        return fail("no scene camera this frame");
    }
    if (oot_motion.depth_frame != render_trace_frame) {
        return fail("no scene depth this frame");
    }
    if (oot_motion.camera_channel != gpu_memory || oot_motion.depth_channel != gpu_memory) {
        return fail("camera and depth from another channel");
    }
    const auto tic = read_tic(oot_motion.depth_tic);
    if (!tic || tic->Address() != oot_motion.depth_address ||
        !oot_motion.SceneSized(tic->Width(), tic->Height())) {
        return fail("scene depth descriptor changed");
    }

    texture_cache.SynchronizeDescriptors(true);
    std::array images{VideoCommon::ImageViewInOut{.index = oot_motion.depth_tic}};
    texture_cache.FillImageViews(images, true);
    if (!images[0].id) {
        return fail("no scene depth image");
    }
    auto& depth = texture_cache.GetImageView(images[0].id);
    const auto* backing = depth.BackingImage();
    if (!backing || depth.IsRescaled() || depth.format != VideoCore::Surface::PixelFormat::D32_FLOAT ||
        depth.range != VideoCommon::SubresourceRange{} || backing->info.num_samples != 1) {
        return fail("unsupported scene depth");
    }

    const auto sampled = pipeline.SampledImages();
    const auto written = pipeline.WrittenImages();
    if (sampled.size() != 1 || written.size() != 1 || !sampled[0].id || !written[0]) {
        return fail("unexpected upscale bindings");
    }
    const auto& color = texture_cache.GetImageView(sampled[0].id);
    auto& target = texture_cache.GetImageView(written[0]);
    using VideoCore::Surface::PixelFormat;
    const auto extent = oot_motion.extent;
    if (extent.width != oot_logged_extent.width || extent.height != oot_logged_extent.height ||
        target.size.width != oot_logged_target.width) {
        if (++oot_extent_changes <= 20 || oot_extent_changes % 100 == 0) {
            LOG_INFO(Render_Vulkan, "Ocarina scene {}x{} -> {}x{} by pass {:016x} (change {})",
                     extent.width, extent.height, target.size.width, target.size.height,
                     upscale_hash, oot_extent_changes);
        }
        oot_logged_extent = extent;
        oot_logged_target = {target.size.width, target.size.height};
    }
    using enum VideoCommon::OcarinaUpscalePass;
    const auto pass = VideoCommon::FindOcarinaUpscalePass(native_key, upscale_hash);
    const bool target_1440p = target.size.width == 2560 && target.size.height == 1440;
    // At 75% (1920x1080) or without Max Resolution, docked upscales to 2560x1440 via ThreeQuarter.
    // FSR 3.1, FSR 4 and DLSS Performance all take it directly to 4K with HUD compositing.
    const bool performance_scale =
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr || oot_performance_scale ||
        Settings::OcarinaDlssModeFor(Settings::values.oot_resolution.GetValue()) ==
            Settings::OcarinaDlssMode::Performance;
    const bool pass_1440p = target_1440p && (pass == OneToOne || pass == Scaled ||
                                             (pass == ThreeQuarter && performance_scale));
    // Handheld 100% (1920x1080 -> 1920x1080): all temporal upscalers (DLSS, FSR 4, FSR 3.1,
    // MetalFX) run as native temporal AA.
    const bool native_size = pass == OneToOne && !target_1440p;
    // Docked 40% / 33% (e.g. 1024x576 or 854x480) exceeds Streamline DLSS's 3.0x maximum ratio
    // against 3840x2160; cap DLSS output to the 2560x1440 target (<= 3.0x) so DLSS never rejects.
    const bool dlss_exceeds_3x =
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss &&
        (extent.width * 3 < 3840 || extent.height * 3 < 2160);
    const bool upscale_4k = pass_1440p && !native_resolution && !dlss_exceeds_3x;
    // The 1440p and scaled passes take any 16:9 scene up to their target: the DLSS render scales
    // shrink it docked, and the game's dynamic resolution handheld (854x480 at its floor).
    const bool scaled_scene = extent.width <= target.size.width &&
                              extent.height <= target.size.height &&
                              std::abs(static_cast<int>(extent.width * 9) -
                                       static_cast<int>(extent.height * 16)) <= 32;
    const bool valid_extents =
        pass_1440p || pass == Scaled ? scaled_scene
        : native_size ? target.size.width == extent.width && target.size.height == extent.height
                      : target.size.width == extent.width * 4 / 3 &&
                            target.size.height == extent.height * 4 / 3;
    if (!valid_extents || !color.BackingImage() || !target.BackingImage() ||
        color.IsRescaled() || target.IsRescaled() ||
        color.Samples() != VK_SAMPLE_COUNT_1_BIT || target.Samples() != VK_SAMPLE_COUNT_1_BIT ||
        color.range != VideoCommon::SubresourceRange{} || target.range != VideoCommon::SubresourceRange{} ||
        color.format != PixelFormat::E5B9G9R9_FLOAT || target.format != PixelFormat::B10G11R11_FLOAT ||
        !oot_motion.SceneSized(color.size.width, color.size.height) ||
        color.ImageHandle() == target.ImageHandle() || depth.ImageHandle() == target.ImageHandle()) {
        static constexpr const char* unexpected = "unexpected upscale images";
        if (oot_fallback_reason != unexpected || (oot_fallback_repeats + 1) % 120 == 0) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina DLSS: scene {}x{} in {}x{} images, colour {}x{} {}, target {}x{} {}, "
                     "pass {:016x}",
                     extent.width, extent.height, oot_motion.image_extent.width,
                     oot_motion.image_extent.height, color.size.width, color.size.height,
                     static_cast<int>(color.format), target.size.width, target.size.height,
                     static_cast<int>(target.format), upscale_hash);
        }
        return fail(unexpected);
    }

    constexpr VkFormat output_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const auto color_view = color.Handle(Shader::TextureType::Color2D);
    if (!color_view || !device.IsStorageImageExtendedFormatsSupported() ||
        !device.IsFormatSupported(output_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, FormatType::Optimal) ||
        (target.BackingImage()->UsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) == 0 ||
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, target.format).format != output_format) {
        return fail("unsupported storage format");
    }
    // Without Link's pose the mesh stays empty and the motion comes from the camera alone.
    OcarinaActorMesh mesh{};
    ReadOcarinaActor(mesh);
    try {
        const auto target_view =
            target.StorageView(Shader::TextureType::Color2D, Shader::ImageFormat::Typeless, false);
        const auto scene_reference = upscale_4k ? target.Handle(Shader::TextureType::Color2D)
                                                : VK_NULL_HANDLE;
        if (!target_view || (upscale_4k && !scene_reference)) {
            return fail("no upscale target view");
        }
        if (!oot_live_dlss) {
            oot_live_dlss = std::make_unique<OcarinaUpscaler>(
                device, scheduler, descriptor_pool, compute_pass_descriptor_queue, memory_allocator,
                staging_pool, *oot_backend);
        }
        const VkExtent2D target_extent{target.size.width, target.size.height};
#ifndef __APPLE__
        const bool present_4k = upscale_4k;
#else
        // A 3840x2160 image only to be scaled down to a phone or laptop screen: MetalFX and FSR
        // write the game's 1440p target instead.
        constexpr bool present_4k = false;
#endif
        const VkExtent2D output_extent = native_resolution ? extent
                                         : present_4k      ? VkExtent2D{3840, 2160}
                                                           : target_extent;
        // The offset this frame's scene was drawn with (none when no scene draw was offset).
        std::array<f32, 2> jitter{};
        if (oot_jitter_frame == render_trace_frame) {
            jitter = oot_jitter_offset;
        }
        const bool recorded = oot_live_dlss->Record(color_view, depth.DepthView(), target_view,
                                                   extent, output_extent, target_extent, oot_motion.current,
                                                   oot_motion.previous, mesh, depth.ImageHandle(),
                                                   render_trace_frame, jitter,
                                                   {depth.size.width, depth.size.height});
        if (!recorded) {
            if (oot_live_dlss->EvaluationFailed()) {
                texture_cache_runtime.ocarina_fsr_failed = true;
            }
            return fail("DLSS declined the frame");
        }
        if (present_4k) {
            oot_live_dlss->SaveSceneReference(scene_reference);
            oot_4k_scene_frame = render_trace_frame;
            oot_4k_guest_present = VK_NULL_HANDLE;
        }
        oot_dlss_frame = render_trace_frame;
        CountOcarinaUpscale(true);
        oot_jitter_phases = VideoCommon::OcarinaJitterPhases(extent.width, output_extent.width);
        oot_fallback_reason = nullptr;
        return true;
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Ocarina DLSS unavailable: {}", error.what());
        oot_live_failed = true;
        texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        return false;
    }
}

/// The game's scaled upscale passes cost a phone GPU ~30 ms a frame (IPA 150); where no temporal
/// upscaler takes the frame, one bilinear tap of the same scene writes the same target instead.
bool RasterizerVulkan::TryOcarinaBilinear(const ComputePipeline& pipeline, u64 native_key) {
    const auto pass = VideoCommon::FindOcarinaUpscalePass(native_key, pipeline.ShaderHash());
    if (oot_bilinear_failed ||
        (pass != VideoCommon::OcarinaUpscalePass::Scaled &&
         pass != VideoCommon::OcarinaUpscalePass::ThreeQuarter)) {
        return false;
    }
    const auto sampled = pipeline.SampledImages();
    const auto written = pipeline.WrittenImages();
    if (sampled.size() != 1 || written.size() != 1 || !sampled[0].id || !written[0]) {
        return false;
    }
    using VideoCore::Surface::PixelFormat;
    const auto& color = texture_cache.GetImageView(sampled[0].id);
    auto& target = texture_cache.GetImageView(written[0]);
    // A padded scene (oot_resolution 87 %) is only known from this frame's camera.
    const bool camera = oot_camera_sample_frame == render_trace_frame &&
                        oot_motion.SceneSized(color.size.width, color.size.height);
    const VkExtent2D extent = camera ? oot_motion.extent
                                     : VkExtent2D{color.size.width, color.size.height};
    if (!color.BackingImage() || !target.BackingImage() || color.IsRescaled() ||
        target.IsRescaled() || color.Samples() != VK_SAMPLE_COUNT_1_BIT ||
        target.Samples() != VK_SAMPLE_COUNT_1_BIT || color.range != VideoCommon::SubresourceRange{} ||
        target.range != VideoCommon::SubresourceRange{} ||
        color.format != PixelFormat::E5B9G9R9_FLOAT || target.format != PixelFormat::B10G11R11_FLOAT ||
        target.size.width <= extent.width || target.size.height <= extent.height ||
        color.ImageHandle() == target.ImageHandle()) {
        return false;
    }
    constexpr VkFormat output_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const auto color_view = color.Handle(Shader::TextureType::Color2D);
    if (!color_view || !device.IsStorageImageExtendedFormatsSupported() ||
        !device.IsFormatSupported(output_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                  FormatType::Optimal) ||
        (target.BackingImage()->UsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) == 0 ||
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, target.format).format !=
            output_format) {
        return false;
    }
    try {
        const auto target_view =
            target.StorageView(Shader::TextureType::Color2D, Shader::ImageFormat::Typeless, false);
        if (!target_view) {
            return false;
        }
        if (!oot_bilinear) {
            oot_bilinear = std::make_unique<OcarinaBilinearUpscale>(
                device, scheduler, descriptor_pool, compute_pass_descriptor_queue);
        }
        oot_bilinear->Record(color_view, {color.size.width, color.size.height}, extent,
                             target_view, {target.size.width, target.size.height});
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Ocarina bilinear upscale unavailable: {}", error.what());
        oot_bilinear_failed = true;
        return false;
    }
    ++oot_bilinear_passes;
    oot_bilinear_extent = extent;
    return true;
}

/// For the overlay and the frame chain line: the scene an upscale pass reads, the size it writes
/// and what drew it this frame.
void RasterizerVulkan::NoteOcarinaScene(const ComputePipeline& pipeline, bool replaced,
                                        bool bilinear) {
    const auto sampled = pipeline.SampledImages();
    const auto written = pipeline.WrittenImages();
    if (sampled.size() != 1 || written.size() != 1 || !sampled[0].id || !written[0]) {
        return;
    }
    const auto& color = texture_cache.GetImageView(sampled[0].id);
    const auto& target = texture_cache.GetImageView(written[0]);
    const bool camera = oot_camera_sample_frame == render_trace_frame &&
                        oot_motion.SceneSized(color.size.width, color.size.height);
    const VkExtent2D scene = camera ? oot_motion.extent
                                    : VkExtent2D{color.size.width, color.size.height};
#ifdef EGLANTINE_STREAMLINE
    constexpr const char* live = "DLSS";
#else
    constexpr const char* live = "MetalFX Temporal";
#endif
    const char* const upscaler =
        replaced ? oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr
                       ? (oot_live_dlss && oot_live_dlss->UsedFsr4() ? "FSR 4" : "FSR 3.1")
                       : live
                 : bilinear ? "bilinear" : "the game's pass";
    VideoCore::Probe::StoreScene(scene.width, scene.height, target.size.width, target.size.height,
                                 upscaler);
}

bool RasterizerVulkan::FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes) {
    static constexpr u32 BUFFER_CAP = 4U << 20;
    std::vector<FrameDump::BoundBuffer> buffers;
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto* host, u32 host_offset) {
        const u32 size = std::min(bound_size, BUFFER_CAP);
        const bool gpu_modified = buffer_cache.IsRegionGpuModified(binding.device_addr, size);
        FrameDump::BoundBuffer& buffer = buffers.emplace_back(FrameDump::BoundBuffer{
            .stage = stage,
            .storage = storage,
            .index = index,
            .written = written,
            .gpu_addr = binding.gpu_addr,
            .device_addr = binding.device_addr,
            .bound_size = bound_size,
            .size = size,
            .gpu_modified = gpu_modified,
            .host = host ? host->Handle() : VK_NULL_HANDLE,
            .host_offset = host_offset,
            .guest = {},
        });
        if (!gpu_modified || !host) {
            buffer.guest.resize(size);
            device_memory.ReadBlockUnsafe(binding.device_addr, buffer.guest.data(), size);
        }
        // EDEN_NX2_FRAME_DUMP_HOST: the host copy of guest-sourced ranges read from the buffer
        // cache (storage, and uniforms too large for the stream path), to find stale uploads.
        static const bool host_too = std::getenv("EDEN_NX2_FRAME_DUMP_HOST") != nullptr;
        buffer.also_host = host_too && host && !gpu_modified && (storage || bound_size > 4096);
    });
    return frame_dump.NoteBuffers(compute, shader_hashes, buffers);
}

void RasterizerVulkan::FrameDumpPass(const Framebuffer* framebuffer) {
    std::array<ImageView*, VideoCommon::NUM_RT> colors{};
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot]) {
            colors[slot] = &texture_cache.GetImageView(ids[slot]);
        }
    }
    bool recorded = false;
    if (frame_dump.Capturing()) {
        recorded |= frame_dump.BeginPass(framebuffer, colors);
    }
    if (flash_watch.WatchesPasses()) {
        recorded |= flash_watch.BeginPass(framebuffer, colors);
    }
    if (recorded) {
        scheduler.RequestRenderpass(framebuffer);
    }
}

void RasterizerVulkan::TickFrame() {
    VideoCommon::FeatureCensus::Tick();
    NoteDrawReuseProbeFrame();
    if (VideoCore::Nx2Debug::HasRequests()) {
        if (auto capture = VideoCore::Nx2Debug::TakeCapture()) {
            // This build keeps the EDEN_NX2_FRAME_DUMP capture (file or time trigger) only.
            VideoCore::Nx2Debug::FinishCapture(
                capture->id, {.error = "capture_frame is not available in this build; use "
                                       "EDEN_NX2_FRAME_DUMP"});
        }
        if (const auto change = VideoCore::Nx2Debug::TakeRejectChange()) {
            pipeline_cache.RefreshDebugRejections(change->unrejected);
            if (change->unrejected) {
                VideoCore::Nx2Debug::NoteUnrejectApplied();
            }
            VideoCore::Nx2Debug::NoteRejectApplied(change->generation);
        }
        while (const auto read = VideoCore::Nx2Debug::TakeGpuRead()) {
            std::vector<Tegra::MemoryManager*> spaces;
            if (gpu_memory != nullptr) {
                spaces.push_back(gpu_memory);
            }
            {
                std::scoped_lock lock{config_mutex};
                for (const auto& [id, space] : address_spaces) {
                    if (space.gpu_memory != gpu_memory) {
                        spaces.push_back(space.gpu_memory);
                    }
                }
            }
            VideoCore::Nx2Debug::FinishGpuRead(read->id,
                                               ReadGpuMemory(spaces, read->gpu_address, read->size));
        }
    }
    frame_dump.TickFrame();
    if (VideoCommon::Nx2Diag::Active()) {
        auto& diag = VideoCommon::Nx2Diag::Diagnostics::Get();
        if (const u32 count = diag.QueryCount();
            diag.Capturing() && count > 0 && g_diag_pool != nullptr) {
            // The capture ends here: wait for its draws and read what the probes counted.
            scheduler.Finish();
            std::vector<u64> samples(count);
            const VkResult result = device.GetLogical().GetQueryResults(
                **g_diag_pool, 0, count, sizeof(u64) * count, samples.data(), sizeof(u64),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (result == VK_SUCCESS) {
                diag.SetSamples(std::move(samples));
            } else {
                LOG_ERROR(Render_Vulkan, "diag: reading the draw probe failed ({})",
                          static_cast<int>(result));
            }
        }
        diag.TickFrame();
        if (DiagLite() && VideoCommon::Nx2Perf::Verify()) {
            DiagLiteVerifyFrame();
        }
        if (diag.Capturing() && VideoCommon::Nx2Diag::SamplesEnabled()) {
            // A capture just began: (re)create and reset the probe pool.
            if (g_diag_pool == nullptr) {
                g_diag_pool = new vk::QueryPool(device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_OCCLUSION,
                    .queryCount = VideoCommon::Nx2Diag::MaxProbeQueries,
                    .pipelineStatistics = 0,
                }));
            }
            if (device.IsHostQueryResetSupported()) {
                device.GetLogical().ResetQueryPool(**g_diag_pool, 0,
                                                   VideoCommon::Nx2Diag::MaxProbeQueries);
            } else {
                scheduler.RequestOutsideRenderPassOperationContext();
                scheduler.Record([pool = **g_diag_pool](vk::CommandBuffer cmdbuf) {
                    cmdbuf.ResetQueryPool(pool, 0, VideoCommon::Nx2Diag::MaxProbeQueries);
                });
            }
        }
    }
    flash_watch.TickFrame();
    ++render_trace_frame;
    render_trace_draws = 0;
    render_trace_clears = 0;
    if (Common::DkbFrameTrace::Enabled()) {
        CollectIndirectTrace();
        const auto now = std::chrono::steady_clock::now();
        if (trace_cpu_rate.Ready(now)) {
            auto snapshot = Common::DkbCpuTrace::ReadSnapshot();
            if (trace_cpu_snapshot && trace_frame > trace_cpu_frame) {
                const u64 frames = trace_frame - trace_cpu_frame;
                std::string roles;
                for (size_t i = 0; i < Common::DkbCpuTrace::RoleCount; ++i) {
                    const auto name = Common::DkbCpuTrace::RoleNames[i];
                    const auto delta = Common::DkbCpuTrace::CpuTimeDelta(
                        trace_cpu_snapshot->threads[i], snapshot.threads[i]);
                    if (delta) {
                        const double cpu_ms = static_cast<double>(*delta) / 1e6;
                        roles += fmt::format(" {}_cpu_ms={:.3f} {}_cpu_ms_per_frame={:.3f}",
                                             name, cpu_ms, name, cpu_ms / frames);
                    } else {
                        roles += fmt::format(" {}={}", name,
                                             snapshot.threads[i].active ? "unavailable" : "inactive");
                    }
                }
                LOG_INFO(Render_Vulkan,
                         "DKB cpu through_frame={} frames={} window_wall_ms={:.3f}{}",
                         trace_frame, frames,
                         std::chrono::duration<double, std::milli>(
                             snapshot.wall_time - trace_cpu_snapshot->wall_time).count(),
                         roles);
            }
            trace_cpu_snapshot = std::move(snapshot);
            trace_cpu_frame = trace_frame;
        }
        ++trace_window_frames;
        if (trace_sample_frame) {
            LOG_INFO(Render_Vulkan,
                     "DKB draws through_frame={} frames={} attempts={} skipped={} indirect={} "
                     "zero_direct={} capture_dropped={} attempts_per_frame={:.1f} "
                     "zero_direct_per_frame={:.1f} (totals since previous line)",
                     trace_frame, trace_window_frames, trace_draws, trace_skipped, trace_indirect,
                     trace_zero_direct, trace_capture_dropped,
                     static_cast<double>(trace_draws) / trace_window_frames,
                     static_cast<double>(trace_zero_direct) / trace_window_frames);
            trace_draws = trace_skipped = trace_indirect = trace_zero_direct = 0;
            trace_window_frames = 0;
            trace_capture_dropped = trace_failure_details = 0;
            trace_rate.next = now + std::chrono::seconds{1};
        }
        ++trace_frame;
        trace_captures = 0;
        trace_sample_frame = trace_frame < 4 || trace_rate.Ready(now);
    }
    ComputePipeline::TickUpscalerTrace();
    draw_counter = 0;
    pending_native_work = false;
    pipeline_cache.TickFrame();
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
    TickMemoryReport();
    TickPerfReport();
    Common::PerfSched::TickFrame();
    scheduler.MarkPassTimesFrame();
    Common::TickFrameWorkgroup();
    Common::FrameDeadline::SetPeriodNs(static_cast<u64>((std::max)(
        VideoCommon::Nx2Perf::g_guest_frame_interval_ns.load(std::memory_order_relaxed), 0LL)));
    Common::FrameDeadline::NoteFrame();
}

void RasterizerVulkan::ReportSlowAccesses(double seconds) {
    namespace PR = Common::PerfReport;
    auto& counts = PR::g_slow_access;
    std::array<std::array<std::array<u64, PR::SlowAccessCount>, 2>, 3> now{};
    for (size_t store = 0; store < 2; ++store) {
        for (size_t reason = 0; reason < PR::SlowAccessCount; ++reason) {
            now[0][store][reason] = counts.calls[store][reason].load(std::memory_order_relaxed);
            now[1][store][reason] = counts.timed[store][reason].load(std::memory_order_relaxed);
            now[2][store][reason] = counts.timed_ns[store][reason].load(std::memory_order_relaxed);
        }
    }
    const auto then = perf_report_slow;
    perf_report_slow = now;
    const std::array<u64, 2> runs_now{counts.tracked_page_runs[0].load(std::memory_order_relaxed),
                                      counts.tracked_page_runs[1].load(std::memory_order_relaxed)};
    const auto runs_then = perf_report_slow_runs;
    perf_report_slow_runs = runs_now;
    if (seconds <= 0.0) {
        PR::g_slow_site_epoch.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::array<std::string, 2> text;
    std::array<double, 2> total_us{};
    for (size_t store = 0; store < 2; ++store) {
        u64 calls = 0;
        for (size_t reason = 0; reason < PR::SlowAccessCount; ++reason) {
            const u64 n = now[0][store][reason] - then[0][store][reason];
            if (n == 0) {
                continue;
            }
            calls += n;
            const u64 timed = now[1][store][reason] - then[1][store][reason];
            const double ns = timed == 0 ? 0.0
                                         : static_cast<double>(now[2][store][reason] -
                                                               then[2][store][reason]) /
                                               static_cast<double>(timed);
            total_us[store] += ns * static_cast<double>(n) / 1e3;
            text[store] += fmt::format(", {} {:.0f} ({:.0f} ns)",
                                       PR::SlowAccessName(static_cast<PR::SlowAccess>(reason)),
                                       static_cast<double>(n) / seconds, ns);
        }
        const auto tracked = static_cast<size_t>(PR::SlowAccess::GpuTracked);
        const u64 tracked_calls = now[0][store][tracked] - then[0][store][tracked];
        const u64 runs = runs_now[store] - runs_then[store];
        if (tracked_calls != 0 && runs != 0) {
            text[store] += fmt::format(" (GPU-tracked: {:.0f} page changes/s, {:.1f} in a row)",
                                       static_cast<double>(runs) / seconds,
                                       static_cast<double>(tracked_calls) / static_cast<double>(runs));
        }
        text[store] = fmt::format("{:.0f}/s{}", static_cast<double>(calls) / seconds, text[store]);
    }
    // The busiest sampled guest blocks over all cores.
    struct Site {
        u64 key;
        u64 count;
        u64 vaddr;
    };
    std::vector<Site> sites;
    u64 sampled = 0;
    for (const auto& core_sites : PR::g_slow_sites) {
        for (const auto& site : core_sites) {
            const u64 key = site.key.load(std::memory_order_relaxed);
            const u64 count = site.count.load(std::memory_order_relaxed);
            if (key == 0 || count == 0) {
                continue;
            }
            sampled += count;
            const auto it = std::ranges::find(sites, key, &Site::key);
            if (it != sites.end()) {
                it->count += count;
            } else {
                sites.push_back({key, count, site.vaddr.load(std::memory_order_relaxed)});
            }
        }
    }
    PR::g_slow_site_epoch.fetch_add(1, std::memory_order_relaxed);
    std::ranges::sort(sites, [](const Site& a, const Site& b) { return a.count > b.count; });
    std::string site_text;
    for (size_t i = 0; i < (std::min<size_t>)(sites.size(), 8); ++i) {
        const Site& site = sites[i];
        site_text += fmt::format(
            "{}block {:#x} {} {} {:.0f}% (at {:#x})", i == 0 ? "" : ", ", site.key >> 8,
            (site.key & 1) != 0 ? "store" : "load",
            PR::SlowAccessName(static_cast<PR::SlowAccess>((site.key >> 1) & 0x7F)),
            100.0 * static_cast<double>(site.count) / static_cast<double>((std::max)(sampled, u64{1})),
            site.vaddr);
    }
    LOG_INFO(Common_Memory,
             "Perf memory callbacks: loads {} | stores {} | ~{:.1f} ms/s of guest core time | "
             "sampled blocks (1 in {}): {}",
             text[0], text[1], (total_us[0] + total_us[1]) / 1e3 / seconds, PR::SlowSampleEvery,
             site_text.empty() ? std::string{"none"} : site_text);
    ReportCpuWrites(seconds);
    ReportCpuReads(seconds);
}

void RasterizerVulkan::ReportCpuWrites(double seconds) {
    namespace Stats = VideoCommon::CpuWriteStats;
    static std::array<u64, static_cast<size_t>(Stats::Outcome::Count)> outcomes_then{};
    static std::array<u64, 4> age_then{};
    static std::array<u64, 3> source_then{};
    const auto delta = [&](auto& counters, auto& then, size_t index) {
        const u64 now = counters[index].load(std::memory_order_relaxed);
        return static_cast<double>(now - std::exchange(then[index], now)) / seconds;
    };
    using O = Stats::Outcome;
    const auto outcome = [&](O o) {
        return delta(Stats::g_outcomes, outcomes_then, static_cast<size_t>(o));
    };
    const double unregistered = outcome(O::Unregistered);
    const double gpu_page = outcome(O::GpuPage);
    const double gpu_bytes = outcome(O::GpuBytes);
    const double no_gpu_bytes = outcome(O::PageNoGpuBytes);
    const double cpu_read = outcome(O::PageCpuRead);
    const double preflushable = outcome(O::PagePreflushable);
    const double already = outcome(O::AlreadyCpu);
    const double marked = outcome(O::Marked);
    const double still = outcome(O::StillProtected);
    const double reclaimed = outcome(O::Reclaimed);
    const double queued = outcome(O::DownloadQueued);
    std::string detail;
    if (Stats::Detailed()) {
        detail = fmt::format(" | of those, pages without GPU-written bytes {:.0f}, read by the CPU "
                             "{:.0f}, preflushable {:.0f}",
                             no_gpu_bytes, cpu_read, preflushable);
        detail += fmt::format(" | GPU marks of those: age 0 {:.0f}, 1 {:.0f}, 2-15 {:.0f}, 16+ {:.0f} "
                             "frames; binding {:.0f}, DMA {:.0f}, unknown {:.0f} | pages:",
                             delta(Stats::g_gpu_mark_age, age_then, 0),
                             delta(Stats::g_gpu_mark_age, age_then, 1),
                             delta(Stats::g_gpu_mark_age, age_then, 2),
                             delta(Stats::g_gpu_mark_age, age_then, 3),
                             delta(Stats::g_gpu_mark_source, source_then, 0),
                             delta(Stats::g_gpu_mark_source, source_then, 1),
                             delta(Stats::g_gpu_mark_source, source_then, 2));
        auto samples = Stats::TakeSamples();
        std::ranges::sort(samples, [](const auto& a, const auto& b) { return a.count > b.count; });
        for (const auto& sample : samples) {
            if (sample.count == 0) {
                break;
            }
            detail += fmt::format(" {:#x} x{} (mark {:#x}+{:#x} {} frames ago {}, GPU bytes "
                                  "{:#x}-{:#x})",
                                  sample.addr, sample.count, sample.mark.begin, sample.mark.size,
                                  sample.age, sample.mark.source == 0 ? "binding" : "DMA",
                                  sample.gpu_first, sample.gpu_end);
        }
    }
    namespace Hot = VideoCommon::HotPages;
    static std::array<u64, 13> hot_then{};
    const auto hot_delta = [&](const std::atomic<u64>& counter, size_t index) {
        const u64 now = counter.load(std::memory_order_relaxed);
        return static_cast<double>(now - std::exchange(hot_then[index], now)) / seconds;
    };
    const auto& hc = Hot::g_counters;
    // Tracker protection changes are counted with hot pages off too, for the comparison.
    std::string hot_text = fmt::format(
        " | tracker protects {:.0f}/s ({:.0f} pages), unprotects {:.0f}/s ({:.0f} pages)",
        hot_delta(hc.protect_ranges, 0), hot_delta(hc.protect_pages, 1),
        hot_delta(hc.unprotect_ranges, 2), hot_delta(hc.unprotect_pages, 3));
    if (Hot::g_mode != 0) {
        hot_text += fmt::format(
            " | hot pages (mode {}) {} now: promoted {:.0f}/s, cooled {:.0f}/s, GPU-written "
            "{:.0f}/s, uploads left unprotected {:.0f}/s, epochs {:.0f}/s",
            Hot::g_mode, hc.hot.load(std::memory_order_relaxed), hot_delta(hc.promoted, 4),
            hot_delta(hc.cooled, 5), hot_delta(hc.gpu_written, 6), hot_delta(hc.kept, 7),
            hot_delta(hc.epochs, 8));
    }
    if (Hot::g_verify) {
        hot_text += fmt::format(" | hot verify: uses {:.0f}/s, stale {:.0f}/s, uploads {:.0f}/s, "
                                "same bytes as the last upload {:.0f}/s",
                                hot_delta(hc.verify_uses, 9), hot_delta(hc.verify_stale, 10),
                                hot_delta(hc.verify_uploads, 11), hot_delta(hc.verify_same, 12));
    }
    LOG_INFO(Common_Memory,
             "Perf CPU writes: buffer cache asked {:.0f}/s: no buffer {:.0f}, GPU-modified page "
             "{:.0f} (written bytes GPU-written {:.0f}; reclaimed as CPU pages {:.0f}, downloads "
             "queued {:.0f}), page already CPU-modified {:.0f}, marked {:.0f}, still protected "
             "after {:.0f}{}{}",
             unregistered + gpu_page + already + marked - reclaimed, unregistered, gpu_page, gpu_bytes,
             reclaimed, queued, already, marked, still, hot_text, detail);
}

void RasterizerVulkan::ReportCpuReads(double seconds) {
    namespace Stats = VideoCommon::CpuReadStats;
    auto& counters = Stats::g_counters;
    static std::array<u64, 2> ask_then{};
    static std::array<std::array<u64, Stats::CauseCount>, 3> cause_then{};
    static std::array<u64, 7> download_then{};
    const auto delta = [](const std::atomic<u64>& counter, u64& then) {
        const u64 now = counter.load(std::memory_order_relaxed);
        return static_cast<double>(now - std::exchange(then, now));
    };
    const double asks = delta(counters.asks, ask_then[0]);
    const double ask_ns = delta(counters.ask_ns, ask_then[1]);
    static std::array<u64, 4> lock_then{};
    const double texture_asks = delta(counters.texture_asks, lock_then[0]);
    const double texture_lock_ns = delta(counters.texture_lock_ns, lock_then[1]);
    const double buffer_asks = delta(counters.buffer_asks, lock_then[2]);
    const double buffer_lock_ns = delta(counters.buffer_lock_ns, lock_then[3]);
    std::string causes;
    double waits = 0.0;
    double wait_ns = 0.0;
    for (size_t cause = 0; cause < Stats::CauseCount; ++cause) {
        const double n = delta(counters.waits[cause], cause_then[0][cause]);
        const double queue = delta(counters.queue_ns[cause], cause_then[1][cause]);
        const double flush = delta(counters.flush_ns[cause], cause_then[2][cause]);
        if (n == 0.0) {
            continue;
        }
        waits += n;
        wait_ns += queue + flush;
        causes += fmt::format("{}{} {:.0f} ({:.0f} us queued, {:.0f} us flush)",
                              causes.empty() ? "" : ", ", Stats::CauseName(cause), n / seconds,
                              queue / n / 1e3, flush / n / 1e3);
    }
    const double staged = delta(counters.staged, download_then[0]);
    const double staged_ns = delta(counters.staged_ns, download_then[1]);
    const double mapped = delta(counters.mapped, download_then[2]);
    const double mapped_ns = delta(counters.mapped_ns, download_then[3]);
    const double bytes = delta(counters.bytes, download_then[4]);
    const double core_landed = delta(counters.core_landed, download_then[5]);
    const double core_ns = delta(counters.core_ns, download_then[6]);
    static std::array<u64, 4> land_then{};
    const double core_refused = delta(counters.core_refused, land_then[0]);
    const double landed = delta(counters.landed, land_then[1]);
    const double landed_ns = delta(counters.landed_ns, land_then[2]);
    const double landed_bytes = delta(counters.landed_bytes, land_then[3]);
    LOG_INFO(Common_Memory,
             "Perf CPU reads: asked {:.0f}/s ({:.1f} us each; texture cache lock {:.0f}/s "
             "{:.1f} us, buffer cache lock {:.0f}/s {:.1f} us), waited for the GPU {:.0f}/s, "
             "{:.1f} ms/s ({}) | downloads: staged {:.0f}/s ({:.2f} ms each), mapped {:.0f}/s "
             "({:.2f} ms each), {:.0f} KiB/s | in-flight downloads landed {:.0f}/s ({:.2f} ms "
             "each, {:.0f} KiB/s), by the reading core {:.0f}/s ({:.2f} ms each, {:.0f}/s went "
             "to the GPU thread after all) (land mode {})",
             asks / seconds, asks == 0.0 ? 0.0 : ask_ns / asks / 1e3, texture_asks / seconds,
             texture_asks == 0.0 ? 0.0 : texture_lock_ns / texture_asks / 1e3,
             buffer_asks / seconds, buffer_asks == 0.0 ? 0.0 : buffer_lock_ns / buffer_asks / 1e3,
             waits / seconds,
             wait_ns / 1e6 / seconds, causes.empty() ? std::string{"none"} : causes,
             staged / seconds, staged == 0.0 ? 0.0 : staged_ns / staged / 1e6, mapped / seconds,
             mapped == 0.0 ? 0.0 : mapped_ns / mapped / 1e6, bytes / 1024.0 / seconds,
             landed / seconds, landed == 0.0 ? 0.0 : landed_ns / landed / 1e6,
             landed_bytes / 1024.0 / seconds, core_landed / seconds,
             core_landed == 0.0 ? 0.0 : core_ns / core_landed / 1e6, core_refused / seconds,
             Stats::LandMode());
}

void RasterizerVulkan::TickPerfReport() {
    if (!Common::PerfReport::Enabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (perf_report_frames != 0) {
        perf_frame_max_ms = (std::max)(
            perf_frame_max_ms,
            std::chrono::duration<double, std::milli>(now - perf_frame_last).count());
    }
    perf_frame_last = now;
    if (perf_report_frames++ == 0) {
        perf_report_start = now;
        perf_report_counters = Common::PerfReport::Read();
        perf_report_churn = VideoCommon::Churn::Read();
        perf_report_cpu = Common::DkbCpuTrace::ReadSnapshot();
        perf_report_process_cpu = Common::PerfReport::ReadProcessCpu();
        perf_report_placement = Common::PerfReport::ReadPlacement();
        perf_report_events = Common::PlatformEvents::QueryProcessEvents();
        static_cast<void>(Common::PerfReport::TakeJitPeak());
        perf_report_jit_cores = Common::PerfReport::ReadJitCores();
        ReportSlowAccesses(0.0);
        Scheduler::ReportPerfSites(0);
        perf_frame_max_ms = 0.0;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - perf_report_start).count();
    if (seconds < 10.0) {
        return;
    }
    const u64 frames = perf_report_frames - 1;
    const auto counters = Common::PerfReport::Read();
    const auto churn = VideoCommon::Churn::Read();
    const auto cpu = Common::DkbCpuTrace::ReadSnapshot();
    const auto delta = [&](Common::PerfReport::Counter counter) {
        return counters[counter] - perf_report_counters[counter];
    };
    const auto churned = [&](VideoCommon::Churn::Counter counter) {
        return churn[counter] - perf_report_churn[counter];
    };
    const auto per_second = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto mib_per_second = [seconds](u64 bytes) {
        return static_cast<double>(bytes) / static_cast<double>(1 << 20) / seconds;
    };
    // Busy % of each thread's wall time; "-" when the OS has no thread clock.
    const auto busy = [&](Common::DkbCpuTrace::Role role) -> std::optional<double> {
        const size_t index = static_cast<size_t>(role);
        if (!perf_report_cpu) {
            return std::nullopt;
        }
        const auto ns = Common::DkbCpuTrace::CpuTimeDelta(perf_report_cpu->threads[index],
                                                          cpu.threads[index]);
        if (!ns) {
            return std::nullopt;
        }
        return static_cast<double>(*ns) / 1e7 / seconds;
    };
    const auto percent = [&](Common::DkbCpuTrace::Role role) {
        const auto value = busy(role);
        return value ? fmt::format("{:.0f}%", *value) : std::string{"-"};
    };
    double guest_sum = 0.0;
    double guest_max = 0.0;
    bool guest_known = false;
    double guest_system = 0.0;
    bool guest_system_known = false;
    for (size_t core = 0; core <= Common::DkbCpuTrace::GuestCoreCount; ++core) {
        if (const auto value = busy(static_cast<Common::DkbCpuTrace::Role>(core))) {
            guest_known = true;
            guest_sum += *value;
            guest_max = (std::max)(guest_max, *value);
        }
        if (perf_report_cpu) {
            if (const auto ns = Common::DkbCpuTrace::SystemTimeDelta(perf_report_cpu->threads[core],
                                                                     cpu.threads[core])) {
                guest_system_known = true;
                guest_system += static_cast<double>(*ns) / 1e7 / seconds;
            }
        }
    }
    // Page faults and storage: a guest memory file that iOS pages out shows as page-ins here and
    // as guest cores' time in the kernel (system) or off the CPU (guest code time over CPU time).
    const auto events = Common::PlatformEvents::QueryProcessEvents();
    std::string memory_events = "not known";
    if (events.known && perf_report_events.known) {
        const auto rate32 = [&](u64 after, u64 before) {
            return per_second(static_cast<u32>(after - before));
        };
        memory_events = fmt::format(
            "faults {:.0f}/s, page-ins {:.0f}/s, decompressions {:.0f}/s, copy-on-write {:.0f}/s "
            "| storage read {:.1f} MiB/s, written {:.1f} MiB/s{}",
            rate32(events.faults, perf_report_events.faults),
            per_second(events.pageins - perf_report_events.pageins),
            rate32(events.decompressions, perf_report_events.decompressions),
            rate32(events.cow_faults, perf_report_events.cow_faults),
            mib_per_second(events.storage_read - perf_report_events.storage_read),
            mib_per_second(events.storage_written - perf_report_events.storage_written),
            events.file_resident != 0
                ? fmt::format(", mapped files resident {} MiB", events.file_resident >> 20)
                : std::string{});
    }
    if (const u64 reserve = Common::PlatformEvents::GuestFileReserve(); reserve != 0) {
        // Of the reserve the caches give up at most half their room (BoundedGuestFileReserve).
        const u64 process_left = Common::PlatformEvents::AvailableProcessMemory();
        memory_events += fmt::format(
            ", caches leave the guest file {} of {} MiB",
            process_left != 0
                ? Common::PlatformEvents::BoundedGuestFileReserve(
                      reserve, process_left, VideoCommon::HostBudget::Reserve) >>
                      20
                : reserve >> 20,
            reserve >> 20);
    }
    const u64 jit_peak = Common::PerfReport::TakeJitPeak();
    const std::string jit_text = fmt::format(
        "JIT {:.0f} blocks/s ({:.2f} MiB/s), compiling {:.0f}%, {} "
        "full-cache flushes, {} evictions, {} clears, {} ranges dropped, fullest code cache "
        "{:.1f} of {} MiB",
        per_second(delta(Common::PerfReport::JitBlocks)),
        mib_per_second(delta(Common::PerfReport::JitBytes)),
        static_cast<double>(delta(Common::PerfReport::JitCompileNs)) / 1e7 / seconds,
        delta(Common::PerfReport::JitCacheFullFlushes),
        delta(Common::PerfReport::JitCacheEvictions), delta(Common::PerfReport::JitCacheClears),
        delta(Common::PerfReport::JitRangeInvalidations),
        static_cast<double>(jit_peak) / static_cast<double>(1 << 20),
        Common::PerfReport::g_jit_cache_size.load(std::memory_order_relaxed) >> 20);
    const u64 texture_bytes = delta(Common::PerfReport::TextureUploadBytes);
    const u64 buffer_bytes = delta(Common::PerfReport::BufferUploadBytes);
    const u64 staging_bytes = delta(Common::PerfReport::StagingUploadBytes);
    const u64 other_bytes = staging_bytes - (std::min)(staging_bytes, texture_bytes + buffer_bytes);
    std::array<u64, 3> collector;
    {
        std::scoped_lock lock{texture_cache.mutex};
        collector = texture_cache.CollectorMemory();
    }
    u64 buffers;
    u64 buffers_collect;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffers = buffer_cache.BufferBytes();
        buffers_collect = buffer_cache.CollectBytes();
    }
    const u64 available = VideoCommon::HostBudget::Available(
        VideoCommon::HostBudget::Simulated() ? device.GetDeviceMemoryUsage() : 0);
    const double frame_count = static_cast<double>(frames);
    const auto per_frame = [&](Common::PerfReport::Counter counter) {
        return static_cast<double>(delta(counter)) / frame_count;
    };
    // GPU: busy time of the finished command buffers (overlaps once), per frame and of the wall.
    const u64 gpu_buffers = delta(Common::PerfReport::GpuTimedBuffers);
    const double gpu_ms = static_cast<double>(delta(Common::PerfReport::GpuBusyNs)) / 1e6;
    const std::string gpu_text =
        gpu_buffers == 0
            ? std::string{"GPU - (no timings)"}
            : fmt::format("GPU {:.1f} ms/frame, {:.0f}% busy, {:.1f} command buffers/frame",
                          gpu_ms / frame_count, gpu_ms / 10.0 / seconds,
                          static_cast<double>(gpu_buffers) / frame_count);
    // Guest cores by wall time: guest code, kernel (SVCs, scheduling, fiber switches), idle.
    const auto wall_percent = [&](Common::PerfReport::Counter counter) {
        return static_cast<double>(delta(counter)) / 1e7 / seconds;
    };
    LOG_INFO(Common_Memory,
             "Perf: {:.1f} fps, frame {:.1f} ms avg {:.0f} max | {} | busy: GPU thread {}, GPU "
             "back {}, Vulkan worker {}, guest cores {} (max {}, in the host kernel {}) | guest "
             "cores' time: guest code {:.0f}%, kernel {:.0f}%, idle {:.0f}% ({:.0f} waits/s) | {} "
             "| memory: {} | draws {:.0f}/frame, {:.1f} "
             "skipped (no pipeline; {:.1f} while it was built, held {:.0f} ms/s for builds), {:.1f} "
             "render passes/frame ({:.1f} reopened the target the "
             "last one left, {:.1f} its views through another framebuffer, {:.1f} its images "
             "through other views, {:.1f} shared some of its images; attachments load {:.0f}, store {:.0f} MiB/frame; {:.1f} copies "
             "before their pass, {:.1f} clears kept), {:.1f} mesh "
             "emulated/frame "
             "({:.1f} ended a render pass, {:.1f} left one for another target, {:.1f} ran "
             "before their pass), {:.1f} dispatches/frame ({:.1f} right after another, {:.1f} "
             "without a barrier) | uploads MiB/s: textures {:.1f}, buffers {:.1f}, "
             "other {:.1f}, past the stream buffer {:.1f} | other: uniforms {:.1f} (padding "
             "{:.1f}, {:.1f} of it for indexed reads; {:.1f} indexed binds/frame), 8-bit indices "
             "{:.1f} | GPU scratch (mesh emulation) {:.1f} MiB/s | {} "
             "early submits, {} waits at the cap, {} mesh "
             "arena waits | textures {:.1f} collected/s ({} downloaded, {} redrawn not "
             "downloaded), {:.1f} made/s, {} MiB "
             "(collect {} / {} MiB) | buffers {:.1f} collected/s, {:.1f} made/s, {} MiB (collect "
             "{} MiB) | {} MiB left",
             frame_count / seconds, seconds * 1000.0 / frame_count, perf_frame_max_ms, gpu_text,
             percent(Common::DkbCpuTrace::Role::GpuCommand),
             percent(Common::DkbCpuTrace::Role::GpuBack),
             percent(Common::DkbCpuTrace::Role::VulkanWorker),
             guest_known ? fmt::format("{:.0f}%", guest_sum) : std::string{"-"},
             guest_known ? fmt::format("{:.0f}%", guest_max) : std::string{"-"},
             guest_system_known ? fmt::format("{:.0f}%", guest_system) : std::string{"-"},
             wall_percent(Common::PerfReport::GuestRunNs),
             wall_percent(Common::PerfReport::GuestKernelNs),
             wall_percent(Common::PerfReport::GuestIdleNs),
             per_second(delta(Common::PerfReport::GuestIdles)), jit_text, memory_events,
             per_frame(Common::PerfReport::Draws), per_frame(Common::PerfReport::DrawsSkipped),
             per_frame(Common::PerfReport::DrawsSkippedBuilding),
             static_cast<double>(delta(Common::PerfReport::DrawHoldNs)) / 1e6 / seconds,
             per_frame(Common::PerfReport::RenderPasses),
             per_frame(Common::PerfReport::RenderPassReopens),
             per_frame(Common::PerfReport::RenderPassSameViewReopens),
             per_frame(Common::PerfReport::RenderPassSameImageReopens),
             per_frame(Common::PerfReport::RenderPassSharedImageSwitches),
             per_frame(Common::PerfReport::AttachmentLoadBytes) / (1 << 20),
             per_frame(Common::PerfReport::AttachmentStoreBytes) / (1 << 20),
             per_frame(Common::PerfReport::BeforePassCommands),
             per_frame(Common::PerfReport::DeferredClearsKept),
             per_frame(Common::PerfReport::MeshEmulatedDraws),
             per_frame(Common::PerfReport::MeshPassBreaks),
             per_frame(Common::PerfReport::MeshPassLeaves),
             per_frame(Common::PerfReport::MeshBeforePass),
             per_frame(Common::PerfReport::Dispatches),
             per_frame(Common::PerfReport::DispatchesChained),
             per_frame(Common::PerfReport::DispatchBarriersSkipped), mib_per_second(texture_bytes),
             mib_per_second(buffer_bytes), mib_per_second(other_bytes),
             mib_per_second(delta(Common::PerfReport::StagingPooledBytes)),
             mib_per_second(delta(Common::PerfReport::UniformStreamBytes)),
             mib_per_second(delta(Common::PerfReport::UniformPadBytes)),
             mib_per_second(delta(Common::PerfReport::UniformPadIndexedBytes)),
             per_frame(Common::PerfReport::UniformIndexedBinds),
             mib_per_second(delta(Common::PerfReport::IndexStreamBytes)),
             mib_per_second(delta(Common::PerfReport::StagingScratchBytes)),
             delta(Common::PerfReport::EarlySubmits), delta(Common::PerfReport::CapWaits),
             delta(Common::PerfReport::MeshArenaWaits),
             per_second(delta(Common::PerfReport::TextureCollected)),
             delta(Common::PerfReport::TextureCollectorDownloads),
             delta(Common::PerfReport::TextureCollectorRedrawnDrops),
             per_second(churned(VideoCommon::Churn::ImagesCreated)), collector[0] >> 20,
             collector[1] >> 20, collector[2] >> 20,
             per_second(delta(Common::PerfReport::BufferCollected)),
             per_second(churned(VideoCommon::Churn::BuffersCreated)), buffers >> 20,
             buffers_collect >> 20,
             available != 0 ? fmt::format("{}", available >> 20) : std::string{"?"});
    // The guest CPU's own costs: the JIT's compiles and faults, and what guest code asks of the
    // host per second (all guest cores together).
    const auto process_cpu = Common::PerfReport::ReadProcessCpu();
    std::string host_cpu_text;
    if (process_cpu && perf_report_process_cpu &&
        process_cpu->cpu_ns > perf_report_process_cpu->cpu_ns) {
        const u64 cpu_ns = process_cpu->cpu_ns - perf_report_process_cpu->cpu_ns;
        const u64 performance_ns =
            process_cpu->performance_ns - perf_report_process_cpu->performance_ns;
        const u64 energy_nj = process_cpu->energy_nj - perf_report_process_cpu->energy_nj;
        host_cpu_text = fmt::format(
            " | process CPU {:.0f}%, {:.0f}% of it on performance cores, {:.2f} W",
            static_cast<double>(cpu_ns) / 1e7 / seconds,
            100.0 * static_cast<double>(performance_ns) / static_cast<double>(cpu_ns),
            static_cast<double>(energy_nj) / 1e9 / seconds);
        // Waiting for a CPU (runnable, not run) against blocked: what iOS withholds shows here.
        // Cycles per CPU nanosecond is the average clock while running.
        const auto since = [&](u64 now_value, u64 then_value) {
            return now_value > then_value ? now_value - then_value : u64{0};
        };
        const u64 runnable_ns = since(process_cpu->runnable_ns, perf_report_process_cpu->runnable_ns);
        const u64 cycles = since(process_cpu->cycles, perf_report_process_cpu->cycles);
        const u64 p_cycles =
            since(process_cpu->performance_cycles, perf_report_process_cpu->performance_cycles);
        const u64 instructions =
            since(process_cpu->instructions, perf_report_process_cpu->instructions);
        const u64 p_instructions = since(process_cpu->performance_instructions,
                                         perf_report_process_cpu->performance_instructions);
        const u64 e_ns = cpu_ns - (std::min)(cpu_ns, performance_ns);
        const u64 e_cycles = cycles - (std::min)(cycles, p_cycles);
        const auto ghz = [](u64 count, u64 ns) {
            return ns == 0 ? 0.0 : static_cast<double>(count) / static_cast<double>(ns);
        };
        const auto ipc = [](u64 count, u64 clocks) {
            return clocks == 0 ? 0.0 : static_cast<double>(count) / static_cast<double>(clocks);
        };
        if (cycles != 0) {
            host_cpu_text += fmt::format(
                ", runnable but not running {:.0f}% (of one core), clock: performance cores "
                "{:.2f} GHz ({:.2f} IPC), efficiency cores {:.2f} GHz ({:.2f} IPC)",
                static_cast<double>(runnable_ns) / 1e7 / seconds, ghz(p_cycles, performance_ns),
                ipc(p_instructions, p_cycles), ghz(e_cycles, e_ns),
                ipc(instructions - (std::min)(instructions, p_instructions), e_cycles));
        }
    }
    const auto placement = Common::PerfReport::ReadPlacement();
    if (const u32 high_first = Common::PerfReport::HighCpuFirst(); high_first != 0) {
        std::string cores;
        for (size_t core = 0; core < Common::PerfReport::PlacementCores; ++core) {
            const u64 samples = placement.samples[core] - perf_report_placement.samples[core];
            const u64 high = placement.high[core] - perf_report_placement.high[core];
            cores += samples == 0 ? std::string{" -"}
                                  : fmt::format(" {:.0f}%", 100.0 * static_cast<double>(high) /
                                                                static_cast<double>(samples));
        }
        host_cpu_text += fmt::format(" | guest cores on CPU {}+ (performance):{}", high_first, cores);
    }
    if (const int thermal = Common::PerfReport::ThermalPressure(); thermal >= 0) {
        // Who paces the game: Auto-Cool's cap and the frame workgroup (Experiments).
        static const char* const workgroup = std::getenv("EDEN_NX2_APPLE_WORKGROUP");
        const u32 cool_fps = Common::PlatformEvents::ThermalMaxFps();
        const int workgroup_threads = Common::FrameWorkgroupThreads();
        host_cpu_text += fmt::format(
            " | thermal pressure {}, Auto-Cool {}, frame workgroup {}", thermal,
            cool_fps != 0 ? fmt::format("caps at {} fps", cool_fps) : std::string{"no cap"},
            workgroup_threads >= 0 ? fmt::format("{} fps deadlines, {} threads joined",
                                                 workgroup != nullptr ? workgroup : "?",
                                                 workgroup_threads)
                                   : std::string{"off"});
    }
    host_cpu_text += Common::FrameDeadline::TakeStatusText();
    std::array<u64, 256> svcs{};
    std::array<std::pair<u64, size_t>, 256> svc_rank{};
    for (size_t i = 0; i < svcs.size(); ++i) {
        svcs[i] = Common::PerfReport::g_svc_counts[i].load(std::memory_order_relaxed);
        svc_rank[i] = {svcs[i] - perf_report_svcs[i], i};
    }
    std::partial_sort(svc_rank.begin(), svc_rank.begin() + 6, svc_rank.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string svc_text;
    for (size_t i = 0; i < 6 && svc_rank[i].first != 0; ++i) {
        svc_text += fmt::format("{}{:#04x} {:.0f}", i == 0 ? " (" : ", ", svc_rank[i].second,
                                per_second(svc_rank[i].first));
    }
    if (!svc_text.empty()) {
        svc_text += ")";
    }
    perf_report_svcs = svcs;
    LOG_INFO(Common_Memory,
             "Perf guest: {:.1f} fastmem faults/s | SVCs {:.0f}/s{} | memory callbacks/s: reads "
             "{:.0f}, writes {:.0f} | clock reads {:.0f}/s (callback), WFE/YIELD {:.0f}/s | yields "
             "to nobody {:.0f}/s | guest_wait_backoff sleeps: yields {:.0f}/s ({:.1f} ms/s), clock "
             "polls {:.0f}/s ({:.1f} ms/s){}",
             per_second(delta(Common::PerfReport::JitFastmemFaults)),
             per_second(delta(Common::PerfReport::GuestSvcs)), svc_text,
             per_second(delta(Common::PerfReport::GuestSlowReads)),
             per_second(delta(Common::PerfReport::GuestSlowWrites)),
             per_second(delta(Common::PerfReport::GuestClockReads)),
             per_second(delta(Common::PerfReport::GuestSpinHints)),
             per_second(delta(Common::PerfReport::GuestFutileYields)),
             per_second(delta(Common::PerfReport::GuestYieldSleeps)),
             static_cast<double>(delta(Common::PerfReport::GuestYieldSleepNs)) / 1e6 / seconds,
             per_second(delta(Common::PerfReport::GuestClockWaits)),
             static_cast<double>(delta(Common::PerfReport::GuestClockWaitNs)) / 1e6 / seconds,
             host_cpu_text);
    ReportSlowAccesses(seconds);
    // Each guest core's code cache: which cores need how much, and what refills them.
    const auto jit_cores = Common::PerfReport::ReadJitCores();
    std::string jit_core_text;
    for (size_t core = 0; core < Common::PerfReport::JitCores; ++core) {
        const auto& now_counts = jit_cores[core];
        const auto& then = perf_report_jit_cores[core];
        const auto [live, peak] = Common::PerfReport::TakeJitCoreUse(core);
        if (now_counts.blocks == 0 && live == 0) {
            continue;
        }
        const u64 evictions = now_counts.evictions - then.evictions;
        jit_core_text += fmt::format(
            "{}core {}: {:.1f} MiB (most {:.1f}), {:.0f} blocks/s, compiling {:.0f}%, {} flushes, "
            "{} evictions ({:.1f} MiB), {:.0f} faults/s, {:.2f} MiB/s dropped in place{}",
            jit_core_text.empty() ? "" : " | ", core, static_cast<double>(live) / (1 << 20),
            static_cast<double>(peak) / (1 << 20), per_second(now_counts.blocks - then.blocks),
            static_cast<double>(now_counts.compile_ns - then.compile_ns) / 1e7 / seconds,
            now_counts.flushes - then.flushes, evictions,
            static_cast<double>(now_counts.evicted_bytes - then.evicted_bytes) / (1 << 20),
            per_second(now_counts.faults - then.faults),
            mib_per_second(now_counts.dead_bytes - then.dead_bytes),
            now_counts.live_blocks == 0
                ? std::string{}
                : fmt::format(", {} blocks with {:.1f} MiB of tables + {:.1f} MiB of lists",
                              now_counts.live_blocks,
                              static_cast<double>(now_counts.table_bytes) / (1 << 20),
                              static_cast<double>(now_counts.list_bytes) / (1 << 20)));
    }
    Scheduler::ReportPerfSites(frames);
    pipeline_cache.ReportBuilds(seconds);
    perf_report_jit_cores = jit_cores;
    if (!jit_core_text.empty()) {
        LOG_INFO(Common_Memory, "Perf JIT ({} MiB per core, {} shared fault markers + {} blocks): {}",
                 Common::PerfReport::g_jit_cache_size.load(std::memory_order_relaxed) >> 20,
                 Common::PerfReport::g_jit_markers.load(std::memory_order_relaxed),
                 Common::PerfReport::g_jit_marker_blocks.load(std::memory_order_relaxed),
                 jit_core_text);
    }
    perf_report_process_cpu = process_cpu;
    perf_report_placement = placement;
    perf_report_start = now;
    perf_report_counters = counters;
    perf_report_churn = churn;
    perf_report_cpu = cpu;
    perf_report_events = events;
    perf_report_frames = 1;
    perf_frame_max_ms = 0.0;
}

void RasterizerVulkan::TickMemoryReport() {
    // After host memory pressure the caches collect for 30 frames: then malloc hands what they
    // freed back to the OS.
    bool pressured = false;
    if (const u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
        pressure != memory_report_pressure) {
        memory_report_pressure = pressure;
        malloc_relief_frames = 31;
        // And where the footprint went, now.
        memory_report_next = {};
        pressured = true;
    }
    if (malloc_relief_frames != 0 && --malloc_relief_frames == 0) {
        const u64 released = Common::PlatformEvents::RelieveMallocPressure();
        LOG_INFO(Common_Memory, "Host memory pressure: malloc released {} MiB", released >> 20);
    }
    // Where the footprint goes, at the first frame, on memory pressure and every 30 s (5 s with
    // under 1 GiB left to the process). On by default on iOS; EDEN_NX2_MEMORY_REPORT=1
    // elsewhere, =0 turns it off.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_MEMORY_REPORT");
#if defined(__APPLE__) && TARGET_OS_IPHONE
        return value == nullptr || value[0] != '0';
#else
        return value != nullptr && value[0] != '0';
#endif
    }();
    const auto now = std::chrono::steady_clock::now();
    const u32 requests = VideoCore::Probe::memory_report_requests.load(std::memory_order_relaxed);
    const bool requested = requests != memory_report_requests;
    memory_report_requests = requests;
    if (!requested && (!enabled || now < memory_report_next)) {
        return;
    }
    const u64 available = Common::PlatformEvents::AvailableProcessMemory();
    const bool tight = available != 0 && available < (u64{1} << 30);
    memory_report_next = now + std::chrono::seconds{tight ? 5 : 30};
    const auto process = Common::PlatformEvents::QueryProcessMemory();
    std::pair<size_t, u64> images;
    std::array<u64, 3> collector;
    std::array<u64, 6> image_kinds;
    {
        std::scoped_lock lock{texture_cache.mutex};
        images = texture_cache.ImageMemory();
        collector = texture_cache.CollectorMemory();
        image_kinds = texture_cache.ImageMemoryKinds();
    }
    u64 buffer_bytes;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_bytes = buffer_cache.BufferBytes();
    }
    const auto [graphics_pipelines, compute_pipelines] = pipeline_cache.PipelineCounts();
    // Logged as Common.Memory: the iOS log filter drops Render.Vulkan below Warning, which kept
    // these lines out of every log from a phone.
    LOG_INFO(Common_Memory,
             "Memory: footprint {} MiB (peak {}, internal {}, compressed {}, reusable {}, "
             "purgeable {}), {} MiB left, mapped files {} MiB resident (not in the footprint) | "
             "{} | guest RAM handed out {} MiB, resident {} MiB, zero "
             "pages trimmed {} MiB (last scan) | Vulkan {} | textures {} images {} MiB (collector "
             "{} / {} / {} MiB) | buffers {} MiB | staging stream {} MiB + pooled {} MiB | "
             "pipelines {} graphics, {} compute | ASTC {}, BCn {}",
             process.footprint >> 20, process.footprint_peak >> 20, process.internal >> 20,
             process.compressed >> 20, process.reusable >> 20,
             process.purgeable_nonvolatile >> 20, available >> 20, process.external >> 20,
             Common::PlatformEvents::FootprintSplit(process, available),
             Common::HostMemory::ScannedHandedOutBytes() >> 20,
             Common::HostMemory::ScannedResidentBytes() >> 20,
             Common::HostMemory::TrimmedZeroBytes() >> 20, device.MemoryHeapReport(),
             images.first, images.second >> 20, collector[0] >> 20, collector[1] >> 20,
             collector[2] >> 20, buffer_bytes >> 20, staging_pool.StreamBufferSize() >> 20,
             staging_pool.CachedBytes() >> 20, graphics_pipelines, compute_pipelines,
             device.IsOptimalAstcSupported() ? "native" : "decoded",
             device.IsOptimalBcnSupported() ? "native" : "decoded");
    if (const auto commit = Common::HostMemory::GetCommitStats(); commit.reserved) {
        // EDEN_NX2_SEC_RESERVE. A fault commit is a page touched before the kernel handed it out.
        LOG_INFO(Render_Vulkan,
                 "Guest memory section: {} of {} MiB committed, {} fault commits (first at {} "
                 "offset {:#x})",
                 commit.committed_bytes >> 20, commit.section_bytes >> 20, commit.net_commits,
                 (commit.first_net_offset >> 63) != 0 ? "virtual" : "backing",
                 commit.first_net_offset & ~(u64{1} << 63));
    }
    // Where the Vulkan memory goes, for fitting a game into a limit: on request, and under memory
    // pressure (which repeats every second while little is left) at most every 10 s.
    const bool pressure_detail = pressured && now >= memory_detail_next;
    if (pressure_detail) {
        memory_detail_next = now + std::chrono::seconds{10};
    }
    if (requested || pressure_detail || std::getenv("EDEN_NX2_MEMORY_REPORT") != nullptr) {
        LOG_INFO(Common_Memory,
                 "Memory detail: images (host bytes) GPU-written {} MiB, converted {} MiB, "
                 "rescaled {} MiB, other {} MiB, deleted {} MiB, pooled {} MiB | {} | driver "
                 "pipeline cache {} KiB",
                 image_kinds[0] >> 20, image_kinds[1] >> 20, image_kinds[2] >> 20,
                 image_kinds[3] >> 20, image_kinds[4] >> 20, image_kinds[5] >> 20,
                 device.MemoryTypeReport(),
                 pipeline_cache.DriverPipelineCacheBytes() >> 10);
        std::string image_census;
        {
            std::scoped_lock lock{texture_cache.mutex};
            image_census = texture_cache.ImageCensus();
        }
        std::string buffer_census;
        {
            std::scoped_lock lock{buffer_cache.mutex};
            buffer_census = buffer_cache.BufferCensus();
        }
        LOG_INFO(Common_Memory, "Memory census: images: {} | buffers: {}", image_census,
                 buffer_census);
    }
    // Every Vulkan allocation by what it is for: once, on request and with the detail line.
    if (!graphics_report_logged || requested || pressure_detail) {
        graphics_report_logged = true;
        LOG_INFO(Common_Memory, "Graphics memory (renderer): {}", GraphicsMemoryReport());
    }
}

std::string RasterizerVulkan::GraphicsMemoryReport() const {
    const DescriptorPoolStats pools = GetDescriptorPoolStats();
    return fmt::format("by use {} | {} | descriptor pools {} ({} descriptors, {} sets)",
                       MemoryTags::Report(), device.MemoryTypeReport(), pools.pools,
                       pools.descriptors, pools.sets);
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    const auto& regs = maxwell3d->regs;
    const bool reads_memory =
        regs.render_enable_override == Maxwell::RenderEnable::Override::UseRenderEnable &&
        regs.render_enable.mode != Maxwell::RenderEnable::Mode::False &&
        regs.render_enable.mode != Maxwell::RenderEnable::Mode::True;
    if (reads_memory &&
        (device.IsExtConditionalRendering() ||
         gpu_memory->IsMemoryDirty(regs.render_enable.Address(), 24,
                                   VideoCommon::CacheType::BufferCache))) {
        gpu_memory->FlushCaching();
    }
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerVulkan::UsesNvidiaDlss() const {
    return texture_cache_runtime.UsesNvidiaDlss();
}

bool RasterizerVulkan::UsesCameraMotionUpscale() const {
    // Ocarina temporal upscaling on Vulkan runs through TryOcarinaUpscale (OcarinaUpscaler with
    // actor motion, reactive mask and 4K HUD composite).
    return false;
}

bool RasterizerVulkan::SupportsGameDlssNetwork() const {
    // Compute pipelines are created 32 threads wide where the device can be told to.
    return device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
}

std::optional<VideoCore::GameDlssProbe> RasterizerVulkan::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    std::optional<std::array<u32, 2>> src_region) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.ProbeGameDlssOutput(dst, src, src_region);
}

bool RasterizerVulkan::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history,
    std::optional<std::array<u32, 2>> render_region) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale,
                                          use_history, render_region);
}

// NVIDIA DLSS (DLAA and Super Resolution) in place of Ocarina of Time's post-tonemap anti-aliasing
// and upscaling (ocarina_upscale.h). The game draws no jitter and no motion vectors, so the host
// offsets the viewport of every draw of the scene's camera at the scene's size by a sub-pixel
// jitter (the depth prepass and the colour pass alike, or the colour pass's equal depth test
// fails), and DLSS gets motion vectors from the camera alone: each pixel's depth taken back
// through this frame's camera and forward through the last one's.
void RasterizerVulkan::NoteCameraMotionDraw() {
    namespace Ocarina = VideoCommon::OcarinaUpscale;
    auto& motion = camera_motion;
    if (!motion.armed) {
        return;
    }
    if (!UsesNvidiaDlss()) {
        motion.armed = false;
        motion.jitter_active = false;
        motion.jitter_ready = false;
        if (motion.applied != std::array<f32, 2>{}) {
            motion.applied = {};
            state_tracker.InvalidateViewports();
        }
        return;
    }
    const auto& regs = maxwell3d->regs;
    // Exclude occlusion-query proxy draws (sample-count query enabled with no depth, stencil, or
    // color writes) and point topology draws (D3): Ocarina of Time's fog and sun bounding-box /
    // sphere visibility queries must not be jittered.
    const bool sample_query_only =
        regs.zpass_pixel_count_enable != 0 && !regs.depth_write_enabled && !regs.stencil_enable &&
        (regs.rt_control.count == 0 || regs.rt[0].format == Tegra::RenderTargetFormat::NONE ||
         (regs.color_mask[0].R == 0 && regs.color_mask[0].G == 0 && regs.color_mask[0].B == 0 &&
          regs.color_mask[0].A == 0));
    const bool excluded_draw =
        sample_query_only ||
        maxwell3d->draw_manager.draw_state.topology == Maxwell::PrimitiveTopology::Points;
    const GPUVAddr depth = !excluded_draw && regs.zeta_enable &&
                                   regs.viewport_scale_offset_enabled &&
                                   regs.zeta.format == Tegra::DepthFormat::Z32_FLOAT
                               ? regs.zeta.Address()
                               : 0;
    const std::array<u32, 2> depth_size{regs.zeta_size.width, regs.zeta_size.height};
    // The scene's geometry binds its camera. Full-screen passes on the same depth buffer (the
    // tonemap among them) don't, and stay where they are.
    const auto& camera = maxwell3d->state.shader_stages[0].const_buffers[Ocarina::CameraBuffer];
    const bool has_camera = depth != 0 && camera.enabled && camera.size >= Ocarina::CameraBytes;
    auto& cached = motion.camera;
    if (has_camera && camera.address != cached.address) {
        cached.address = camera.address;
        cached.perspective = false;
        if (gpu_memory->IsFullyMappedRange(camera.address, Ocarina::CameraBytes)) {
            std::array<f32, 16> projection;
            gpu_memory->ReadBlockUnsafe(camera.address + Ocarina::ProjectionOffset,
                                        projection.data(), sizeof(projection));
            cached.perspective = Ocarina::IsPerspective(projection);
            gpu_memory->ReadBlockUnsafe(camera.address + Ocarina::ViewProjectionOffset,
                                        cached.view_projection.data(),
                                        sizeof(cached.view_projection));
        }
    }
    const bool perspective = has_camera && cached.perspective;
    // Every depth buffer the scene's camera draws at the scene's size, not only the scene's: the
    // game shades its opaque geometry with an EQUAL test against a copy of the prepass depth, so
    // both must have the same offset.
    std::array<f32, 2> jitter{};
    if (perspective && motion.jitter_active && depth_size == motion.jitter_depth_size) {
        jitter = motion.jitter;
    }
    if (perspective) {
        CameraMotionScene* scene = nullptr;
        for (size_t i = 0; i < motion.num_scenes; ++i) {
            CameraMotionScene& candidate = motion.scenes[i];
            if (candidate.depth == depth && candidate.depth_info.size.width == depth_size[0] &&
                candidate.depth_info.size.height == depth_size[1]) {
                scene = &candidate;
            }
        }
        if (!scene) {
            // Shadow and effect passes draw smaller depth buffers before the scene's: when the
            // records are full, a larger buffer takes the smallest one's.
            const auto area = [](const CameraMotionScene& record) {
                return u64{record.depth_info.size.width} * record.depth_info.size.height;
            };
            if (motion.num_scenes < motion.scenes.size()) {
                scene = &motion.scenes[motion.num_scenes++];
            } else if (CameraMotionScene& smallest =
                           *std::ranges::min_element(motion.scenes, {}, area);
                       area(smallest) < u64{depth_size[0]} * depth_size[1]) {
                scene = &smallest;
            }
            if (scene) {
                *scene = CameraMotionScene{};
                scene->depth = depth;
                scene->depth_info = VideoCommon::ImageInfo(regs.zeta, regs.zeta_size,
                                                           regs.anti_alias_samples_mode);
            }
        }
        if (scene) {
            const bool rescaled = texture_cache.IsRescaling();
            const float scale = rescaled ? Settings::values.resolution_info.up_factor : 1.0f;
            scene->camera = camera.address;
            scene->view_projection = cached.view_projection;
            scene->viewport = GetViewportState(device, regs, 0, scale);
            scene->y_sign = device.IsNvViewportSwizzleSupported() &&
                                    regs.viewport_transform[0].swizzle.y ==
                                        Maxwell::ViewportSwizzle::NegativeY
                                ? -1.0f
                                : 1.0f;
            scene->depth_minus_one = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne;
            scene->rescaled = rescaled;
            scene->jitter = jitter;
            ++scene->draws;
        }
    }
    if (jitter != motion.applied) {
        motion.applied = jitter;
        state_tracker.InvalidateViewports();
    }
}

void RasterizerVulkan::NoteCameraMotionClear() {
    auto& motion = camera_motion;
    const auto& regs = maxwell3d->regs;
    if (!motion.armed || !regs.zeta_enable || regs.zeta.Address() != motion.jitter_depth ||
        std::array<u32, 2>{regs.zeta_size.width, regs.zeta_size.height} !=
            motion.jitter_depth_size) {
        return;
    }
    if (motion.jitter_ready) {
        motion.jitter_ready = false;
        motion.jitter_active = true;
        return;
    }
    // Cleared again with no evaluation since: another clear in the same frame, or the game's
    // anti-aliasing stopped running (a menu, a load), so its draws mustn't keep the jitter.
    if (motion.jitter_active && motion.last_evaluation &&
        std::chrono::steady_clock::now() - *motion.last_evaluation >
            std::chrono::milliseconds{100}) {
        motion.jitter_active = false;
    }
}

bool RasterizerVulkan::AccelerateCameraMotionUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                                     const Tegra::Engines::Fermi2D::Surface& dst) {
    namespace Ocarina = VideoCommon::OcarinaUpscale;
    auto& motion = camera_motion;
    motion.armed = true;
    // The scene: the depth buffer at the colour's size with the most perspective draws.
    const CameraMotionScene* scene = nullptr;
    for (size_t i = 0; i < motion.num_scenes; ++i) {
        const CameraMotionScene& candidate = motion.scenes[i];
        if (candidate.draws > 0 && candidate.depth_info.size.width == src.width &&
            candidate.depth_info.size.height == src.height &&
            (!scene || candidate.draws > scene->draws)) {
            scene = &candidate;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const auto finish = [&](bool done) {
        motion.num_scenes = 0;
        // Camera buffers are rewritten each frame, often at the same addresses.
        motion.camera = {};
        if (!done) {
            // The game's own anti-aliasing draws this frame: no jitter, and no history to keep.
            motion.jitter_active = false;
            motion.jitter_ready = false;
            motion.jitter_depth = 0;
            motion.jitter_depth_size = {};
            motion.previous_screen.reset();
        }
        return done;
    };
    if (!scene) {
        static u32 no_scene_logged{};
        if (no_scene_logged < 4) {
            ++no_scene_logged;
            LOG_INFO(Render_Vulkan,
                     "Ocarina of Time anti-aliasing: no matching scene for {}x{} -> {}x{} "
                     "({} recorded scenes)",
                     src.width, src.height, dst.width, dst.height, motion.num_scenes);
            for (size_t i = 0; i < motion.num_scenes; ++i) {
                const auto& candidate = motion.scenes[i];
                LOG_INFO(Render_Vulkan, "  scene[{}]: depth={:#x} {}x{} draws={} camera={:#x}", i,
                         candidate.depth, candidate.depth_info.size.width,
                         candidate.depth_info.size.height, candidate.draws, candidate.camera);
            }
        }
        return finish(false);
    }
    // The scene's viewport must cover the image, so the matrix maps to its pixels.
    const VkViewport& viewport = scene->viewport;
    const f32 width = static_cast<f32>(
        scene->rescaled ? Settings::values.resolution_info.ScaleUp(src.width) : src.width);
    const f32 height = static_cast<f32>(
        scene->rescaled ? Settings::values.resolution_info.ScaleUp(src.height) : src.height);
    if (std::abs(viewport.x) > 0.5f || std::abs(viewport.width - width) > 0.5f ||
        std::abs(std::min(viewport.y, viewport.y + viewport.height)) > 0.5f ||
        std::abs(std::abs(viewport.height) - height) > 0.5f) {
        if (motion.evaluations < 4) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina of Time anti-aliasing: viewport ({}, {}, {}x{}) does not cover "
                     "{}x{}",
                     viewport.x, viewport.y, viewport.width, viewport.height, width, height);
        }
        return finish(false);
    }
    // Device depth = offset + scale * NDC z, as the viewport's depth range gives it.
    const f32 depth_offset = scene->depth_minus_one
                                 ? (viewport.minDepth + viewport.maxDepth) * 0.5f
                                 : viewport.minDepth;
    const f32 depth_scale = scene->depth_minus_one
                                ? (viewport.maxDepth - viewport.minDepth) * 0.5f
                                : viewport.maxDepth - viewport.minDepth;
    const Ocarina::Matrix screen = Ocarina::ScreenMatrix(
        Ocarina::ViewportMatrix(viewport.x, viewport.y, viewport.width, viewport.height,
                                scene->y_sign, depth_offset, depth_scale),
        scene->view_projection);
    const std::array<u32, 2> size{src.width, src.height};
    // A gap in the evaluations (a menu, a load) or another scene starts the history again.
    bool reset = !motion.previous_screen || motion.previous_depth != scene->depth ||
                 motion.previous_size != size || motion.previous_rescaled != scene->rescaled ||
                 !motion.last_evaluation ||
                 now - *motion.last_evaluation > std::chrono::milliseconds{250};
    std::optional<std::array<f32, 16>> reprojection;
    if (!reset) {
        reprojection = Ocarina::Reprojection(*motion.previous_screen, screen);
    }
    if (!reprojection) {
        reset = true;
        reprojection = std::array<f32, 16>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    }
    const std::array<f32, 2> jitter = scene->jitter;
    bool done;
    {
        std::scoped_lock lock{texture_cache.mutex};
        done = texture_cache.CameraMotionUpscale(
            dst, src, scene->depth_info, scene->depth,
            {static_cast<u32>(width), static_cast<u32>(height)}, *reprojection, jitter, reset);
    }
    if (motion.evaluations < 4 || (!done && motion.evaluations < 64)) {
        LOG_INFO(Render_Vulkan,
                 "Ocarina of Time anti-aliasing: NVIDIA DLSS {} (scene depth {:#x} {}x{} -> "
                 "{}x{}, {} draws, camera {:#x}, jitter {},{}, reset {})",
                 done ? "applied" : "unavailable; the game's own runs", scene->depth, src.width,
                 src.height, dst.width, dst.height, scene->draws, scene->camera, jitter[0],
                 jitter[1], reset);
    }
    ++motion.evaluations;
    if (!done) {
        return finish(false);
    }
    motion.jittered |= scene->jitter != std::array<f32, 2>{};
    if (motion.evaluations == 120 && !motion.jittered) {
        LOG_WARNING(Render_Vulkan,
                    "Ocarina of Time anti-aliasing: the scene's depth buffer at {:#x} is never "
                    "cleared, so its draws get no jitter",
                    scene->depth);
    }
    motion.previous_screen = screen;
    motion.previous_depth = scene->depth;
    motion.previous_size = size;
    motion.previous_rescaled = scene->rescaled;
    motion.last_evaluation = now;
    // The next frame's jitter, from when its scene depth is cleared; the rest of this frame (its
    // interface) is drawn without it.
    motion.jitter_depth = scene->depth;
    motion.jitter_depth_size = size;
    motion.jitter = Ocarina::Jitter(++motion.phase);
    motion.jitter_ready = true;
    motion.jitter_active = false;
    return finish(true);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    // Grid placeholders among the words (a compute macro's constant buffer of its dispatch's
    // size): the GPU copies the words they stand for over them once the write is done.
    if (Tegra::Nx2IndirectGrid::AnyLive() && !Tegra::Nx2IndirectGrid::g_writing_qmd &&
        copy_size >= 4) {
        // (source, destination, words): x alone, or x, y and z when all three are placeholders.
        boost::container::small_vector<std::tuple<GPUVAddr, GPUVAddr, u32>, 2> grids;
        const size_t words = (std::min)(copy_size, memory.size()) / sizeof(u32);
        const auto word = [&](size_t i) {
            u32 value;
            std::memcpy(&value, memory.data() + i * sizeof(u32), sizeof(value));
            return value;
        };
        for (size_t i = 0; i < words; ++i) {
            const auto source = Tegra::Nx2IndirectGrid::MatchWord(word(i));
            if (!source) {
                continue;
            }
            u32 count = 1;
            while (i + count < words && count < 3 &&
                   Tegra::Nx2IndirectGrid::MatchWord(word(i + count)) ==
                       *source + count * sizeof(u32)) {
                ++count;
            }
            grids.emplace_back(*source, address + i * sizeof(u32), count);
            i += count - 1;
        }
        if (!grids.empty()) {
            AccelerateInlineToMemoryImpl(address, copy_size, memory);
            // The write's own invalidation first: it would upload the placeholders over the copy.
            gpu_memory->FlushCaching();
            std::scoped_lock lock{buffer_cache.mutex};
            for (const auto& [source, destination, count] : grids) {
                Tegra::Nx2IndirectGrid::Get().copied.fetch_add(1, std::memory_order_relaxed);
                if (!buffer_cache.DMACopy(source, destination, count * sizeof(u32), true)) {
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1) < 8) {
                        LOG_WARNING(Render_Vulkan,
                                    "GPU-sized dispatch: no host buffer to copy its size from "
                                    "{:#x} to {:#x}", source, destination);
                    }
                }
            }
            return;
        }
    }
    AccelerateInlineToMemoryImpl(address, copy_size, memory);
}

void RasterizerVulkan::AccelerateInlineToMemoryImpl(GPUVAddr address, size_t copy_size,
                                                    std::span<const u8> memory) {
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
        static const GPUVAddr watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
        }();
        if (watch != 0 && address < watch + 16 && address + copy_size > watch) {
            LOG_INFO(Render_Vulkan, "Semaphore watch: inline write {:#x} size {} first word {:#x}",
                     address, copy_size, first_word);
        }
    }
    if (!cpu_addr) {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    static const bool inline_fast = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_FAST");
        return value == nullptr || value[0] != '0';
    }();
    bool written = false;
    if (inline_fast && copy_size != 0 &&
        (address & Core::DEVICE_PAGEMASK) + copy_size <= Core::DEVICE_PAGESIZE &&
        (address + copy_size <= Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) ||
         address >= Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed))) {
        struct MappedPage {
            const Tegra::MemoryManager* memory{};
            size_t id{};
            GPUVAddr page{~GPUVAddr{0}};
            u64 serial{};
            bool mapped{};
        };
        thread_local MappedPage last{};
        const GPUVAddr page = address & ~static_cast<GPUVAddr>(Core::DEVICE_PAGEMASK);
        const u64 serial = gpu_memory->MapSerial();
        if (last.memory != gpu_memory || last.id != gpu_memory->GetID() || last.page != page ||
            last.serial != serial) {
            last = MappedPage{
                .memory = gpu_memory,
                .id = gpu_memory->GetID(),
                .page = page,
                .serial = serial,
                .mapped = gpu_memory->MaxContinuousRange(page, Core::DEVICE_PAGESIZE) >=
                          Core::DEVICE_PAGESIZE,
            };
        }
        if (last.mapped) {
            if (u8* const pointer = device_memory.GetPointer<u8>(*cpu_addr)) {
                std::memcpy(pointer, memory.data(), copy_size);
                written = true;
            }
        }
    }
    if (!written) {
        gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    }
    // EDEN_NX2_INLINE_DEFER=0: invalidate every cache at once (for comparisons).
    static const bool defer = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    static const bool i2m_coalesce = [] {
        const char* value = std::getenv("EDEN_NX2_I2M_COALESCE");
        return value == nullptr || value[0] != '0';
    }();
    static const bool per_page_list = [] {
        const char* value = std::getenv("EDEN_NX2_I2M_LIST_PAGE");
        return value == nullptr || value[0] != '0';
    }();
    // One cache at a time: holding both while waiting for either (std::lock) waited longer, the
    // fencing thread and CPU write traps holding one of them most of the time.
    bool uploaded = false;
    struct CleanPage {
        const void* cache{};
        DAddr page{~DAddr{0}};
        u64 serial{};
    };
    thread_local CleanPage clean{};
    const DAddr device_page = *cpu_addr & ~static_cast<DAddr>(Core::DEVICE_PAGEMASK);
    const bool one_page = copy_size != 0 && (*cpu_addr & Core::DEVICE_PAGEMASK) + copy_size <=
                                                Core::DEVICE_PAGESIZE;
    const bool known_clean =
        i2m_coalesce && defer && one_page &&
        clean.cache == &buffer_cache && clean.page == device_page &&
        clean.serial == buffer_cache.gpu_modified_serial.load(std::memory_order_relaxed) &&
        (per_page_list ? !Tegra::Nx2IsCommandListPage(device_page)
                       : !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed));
    if (!known_clean) {
        std::scoped_lock lock{buffer_cache.mutex};
        // Guest memory has the words now: a command list copy of the page no longer does.
        buffer_cache.Nx2NoteListHostWrite(*cpu_addr, copy_size, memory);
        bool page_clean = false;
        uploaded = buffer_cache.InlineMemory(*cpu_addr, copy_size, memory,
                                             per_page_list ? &page_clean : nullptr);
        if (!uploaded && !defer) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
        clean.cache = nullptr;
        if (i2m_coalesce && !uploaded && one_page &&
            (per_page_list ? page_clean
                           : !buffer_cache.IsRegionGpuModified(device_page,
                                                               Core::DEVICE_PAGESIZE))) {
            clean = CleanPage{
                .cache = &buffer_cache,
                .page = device_page,
                .serial = buffer_cache.gpu_modified_serial.load(std::memory_order_relaxed),
            };
        }
    }
    // Pending query results must not land over the new words later: drop them now.
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
    if (!uploaded && defer) {
        // Title 1 makes thousands of small inline writes between draws (~10% of the GPU
        // thread invalidating every cache for each). Guest memory is written; the buffer,
        // texture and shader caches see the merged ranges at the next FlushCaching, which runs
        // before anything of theirs is used. Writes into GPU-written buffer data were uploaded
        // in place above, as before.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    static const bool frame_capture = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    if (frame_capture && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.CapturePresent(*image_view);
    }
    if (flash_watch.Enabled()) {
        flash_watch.NotePresent(*image_view);
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    info.anti_aliased = oot_live_dlss && oot_dlss_frame == render_trace_frame &&
                        !oot_live_dlss->EvaluationFailed();
    if (oot_live_dlss && oot_4k_scene_frame == render_trace_frame && info.image &&
        info.image == oot_4k_guest_present && info.image_view &&
        info.width == 2560 && info.height == 1440 && !scaled &&
        image_view->format == VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM &&
        !oot_live_dlss->EvaluationFailed() && !frame_dump.Capturing()) {
        try {
            oot_live_dlss->Composite4k(info.image_view, render_trace_frame);
            info.image = oot_live_dlss->CompositedImage();
            info.image_view = oot_live_dlss->CompositedView();
            info.scaled_width = 3840;
            info.scaled_height = 2160;
            info.already_upscaled = true;
        } catch (const vk::Exception& error) {
            LOG_WARNING(Render_Vulkan, "Ocarina DLSS 4K presentation unavailable: {}", error.what());
            oot_live_failed = true;
            texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        }
    }
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr bool android = true;
#else
    static constexpr bool android = false;
#endif // __ANDROID__
    const bool native = gpu_memory->UsesNativeTiling();
    const u32 default_native_budget = SubmissionPolicy::DefaultNativeBudget;
    u32 budget = default_native_budget;
    if (native && !android) {
        static const std::optional<u32> configured_budget = []() -> std::optional<u32> {
            if (const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS")) {
                if (const auto parsed = SubmissionPolicy::ParseBudget(value)) {
                    return parsed;
                }
                LOG_WARNING(Render_Vulkan,
                            "Invalid EDEN_NX2_FLUSH_DRAWS; expected an integer from 1 to {}",
                            (std::numeric_limits<u32>::max)());
            }
            return std::nullopt;
        }();
        budget = configured_budget.value_or(default_native_budget);
    }
    u32 legacy_budget = 4096;
    if (!native && !android) {
        static const u32 forced_draws = [] {
            const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS");
            return value != nullptr
                       ? static_cast<u32>(std::max(32L, std::strtol(value, nullptr, 10)) & ~31L)
                       : 0U;
        }();
        legacy_budget = forced_draws != 0 ? forced_draws : (device.IsMoltenVK() ? 128U : 4096U);
    }
    if (native && !android) {
        pending_native_work = true;
    }
    if (staging_pool.BoundUploads()) {
        draw_counter = 0;
        pending_native_work = false;
        return;
    }
    scheduler.FlushExpiredDeferred();
    if (submit_while_waited) {
        scheduler.NoteDraw();
        // Every 32 draws: the draws a submit-while-waited check is worth.
        if ((draw_counter & 31U) == 31U && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
            VideoCommon::GpuSplit::OnBackThread() && SubmitWhileWaited()) {
            draw_counter = 0;
            pending_native_work = false;
            return;
        }
    }
    switch (SubmissionPolicy::Advance(draw_counter, native, android, budget, legacy_budget)) {
    case SubmissionPolicy::Action::None:
        break;
    case SubmissionPolicy::Action::Dispatch:
        scheduler.DispatchWork(true);
        break;
    case SubmissionPolicy::Action::Flush:
        if (!scheduler.IsRenderPassActive() || draw_counter >= 512U) {
            scheduler.Flush();
            pending_native_work = false;
        } else {
            scheduler.DispatchWork(true);
        }
        break;
    }
}

bool RasterizerVulkan::SubmitWhileWaited() {
    // A command list copy or a fence submitted later waits behind everything recorded since the
    // last submission: the host GPU starts on it now while the front can't use the time, or the
    // GPU would sit idle. The same commands in the same order; only the submissions move, at a
    // draw boundary on the back, as fences and list copies already submit.
    namespace Profile = VideoCommon::Nx2Profile;
    const u32 draws = scheduler.DrawsSinceSubmit();
    if (scheduler.IsRenderPassActive() && draws < 512U) {
        if (draws >= submit_waited_draws &&
            (VideoCommon::GpuSplit::Get().FrontWaitsForBack() ||
             (draws >= submit_idle_draws && scheduler.GpuCaughtUp()))) {
            scheduler.DispatchWork(true);
        }
        return false;
    }
    Profile::FlushOrigin origin;
    if (draws >= submit_waited_draws && VideoCommon::GpuSplit::Get().FrontWaitsForBack()) {
        origin = Profile::FlushOrigin::Waited;
    } else if (draws >= submit_idle_draws && scheduler.GpuCaughtUp()) {
        origin = Profile::FlushOrigin::GpuIdle;
    } else {
        return false;
    }
    Profile::FlushOriginTag tag{origin};
    if (!VideoCommon::Nx2Perf::Verify()) {
        scheduler.Flush();
        return true;
    }
    // No answer to compare: the submission must end the pass and advance one tick, as any.
    const u64 tick = scheduler.CurrentTick();
    scheduler.Flush();
    if (scheduler.CurrentTick() != tick + 1 || scheduler.IsRenderPassActive() ||
        scheduler.DrawsSinceSubmit() != 0) {
        static std::atomic<u64> mismatches{};
        if (const u64 count = mismatches.fetch_add(1, std::memory_order_relaxed) + 1; count <= 64) {
            LOG_ERROR(HW_GPU,
                      "Nx2Perf submit_while_waited: tick {} -> {}, render pass {}, {} draws left "
                      "after the submit ({} so far)",
                      tick, scheduler.CurrentTick(), scheduler.IsRenderPassActive(),
                      scheduler.DrawsSinceSubmit(), count);
        }
    }
    return true;
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    buffer_cache.Nx2CollectListCopies(true);
    boost::container::small_vector<std::pair<DAddr, size_t>, 4> gpu_written_dest;
    const bool copied =
        buffer_cache.DMACopy(src_address, dest_address, amount, false, &gpu_written_dest);
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick()) && !scheduler.DeferFlush()) {
        VideoCommon::Nx2Profile::FlushOriginTag tag{VideoCommon::Nx2Profile::FlushOrigin::ListCopy};
        scheduler.Flush();
    }
    buffer_cache.Nx2CollectListCopies(false);
    // DMACopy copies guest memory without telling the texture cache. Images over the destination
    // (pitch linear data textures such as Kirby Air Riders' vertex lighting) would keep their old
    // contents, and stale pages would skip the upload of images made there later.
    if (copied) {
        texture_cache.WriteDmaCopy(src_address, dest_address, amount, gpu_written_dest);
    }
    return copied;
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

template <typename T>
bool RasterizerVulkan::DynamicStateChanged(DynSlot slot, T& cached, const T& value) {
    if (!DynamicStateCacheEnabled()) {
        return true;
    }
    const u64 epoch = state_tracker.CommandBufferEpoch();
    if (dyn_cache.epoch != epoch) {
        dyn_cache.epoch = epoch;
        dyn_cache.valid = 0;
    }
    const u64 bit = u64{1} << static_cast<u32>(slot);
    if ((dyn_cache.valid & bit) != 0 && std::memcmp(&cached, &value, sizeof(T)) == 0) {
        return false;
    }
    cached = value;
    dyn_cache.valid |= bit;
    return true;
}

void RasterizerVulkan::BuildDynamicStateMask() {
    dyn_state_mask_built = true;
    auto& mask = dyn_state_mask;
    mask.reset();
    const auto set = [&mask](std::initializer_list<int> ids) {
        for (const int id : ids) {
            mask[static_cast<size_t>(id)] = true;
        }
    };
    set({Dirty::Viewports, Dirty::Scissors, Dirty::DepthBias, Dirty::BlendConstants,
         Dirty::DepthBounds, Dirty::StencilProperties, Dirty::LineWidth,
         VideoCommon::Dirty::RescaleViewports, VideoCommon::Dirty::RescaleScissors,
         VideoCommon::Dirty::DepthBiasGlobal});
    if (device.IsExtExtendedDynamicStateSupported()) {
        set({Dirty::CullMode, Dirty::DepthCompareOp, Dirty::FrontFace, Dirty::StencilOp,
             Dirty::StateEnable});
    }
    if (device.IsExtExtendedDynamicState2Supported()) {
        set({Dirty::PrimitiveRestartEnable, Dirty::RasterizerDiscardEnable,
             Dirty::DepthBiasEnable});
    }
    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        set({Dirty::LogicOp});
    }
    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        set({Dirty::LogicOpEnable, Dirty::DepthClampEnable, Dirty::LineStippleEnable,
             Dirty::ConservativeRasterizationMode, Dirty::AlphaToCoverageEnable,
             Dirty::AlphaToOneEnable});
        if (device.IsExtLineRasterizationSupported()) {
            set({Dirty::LineRasterizationMode});
        }
    }
    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        set({Dirty::Blending});
    } else if (device.IsExtColorWriteEnableSupported()) {
        set({Dirty::ColorMask});
    }
    if (device.IsExtVertexInputDynamicStateSupported()) {
        set({Dirty::VertexInput, VideoCommon::Dirty::VertexBuffers});
    }
}

void RasterizerVulkan::UpdateDynamicStates(GraphicsPipeline* draw_pipeline) {
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    // A mesh emulation pipeline has static input assembly and vertex input. Binding it replaces
    // that dynamic state, so leave it to the next regular draw to set again.
    const bool emulated_mesh{draw_pipeline != nullptr && draw_pipeline->IsEmulatedMesh()};
    if (emulated_mesh) {
        state_tracker.InvalidateInputState();
    }
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = !emulated_mesh && state_tracker.ChangePrimitiveTopology(topology);
    if (state_tracker.ChangeRasterTopology(RasterTopologyOf(regs, topology))) {
        flags[Dirty::DepthBiasEnable] = true;
    }
    const u64 cb_epoch = state_tracker.CommandBufferEpoch();
    const bool color_output_changed =
        device.IsExtExtendedDynamicState3BlendingSupported() &&
        ((draw_pipeline ? draw_pipeline->ColorOutputMask() : ~0U) != last_color_output_mask ||
         cb_epoch != last_blend_epoch);
    const bool a2c_changed =
        device.IsExtExtendedDynamicState3EnablesSupported() &&
        device.SupportsDynamicState3AlphaToCoverageEnable() &&
        ((draw_pipeline != nullptr && draw_pipeline->SupportsAlphaToCoverage()) !=
             last_a2c_supported ||
         cb_epoch != last_a2c_epoch);
    const bool a2o_changed =
        device.IsExtExtendedDynamicState3EnablesSupported() &&
        device.SupportsDynamicState3AlphaToOneEnable() &&
        ((draw_pipeline != nullptr && draw_pipeline->SupportsAlphaToOne()) != last_a2o_supported ||
         cb_epoch != last_a2o_epoch);
    if (DynamicStateCacheEnabled()) {
        if (!dyn_state_mask_built) {
            BuildDynamicStateMask();
        }
        if (!topology_changed && !(flags & dyn_state_mask).any()) {
            if (a2c_changed) {
                UpdateAlphaToCoverageEnable(regs, draw_pipeline);
            }
            if (a2o_changed) {
                UpdateAlphaToOneEnable(regs, draw_pipeline);
            }
            if (color_output_changed) {
                UpdateBlending(regs, draw_pipeline);
            }
            return;
        }
    }
    if (topology_changed) {
        flags[Dirty::PrimitiveRestartEnable] = true;
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (topology_changed) {
            scheduler.Record([topology_vk = MaxwellToVK::PrimitiveTopology(device, topology)](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        if (!emulated_mesh) {
            UpdatePrimitiveRestartEnable(regs);
        }
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs, draw_pipeline);
        UpdateAlphaToOneEnable(regs, draw_pipeline);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs, draw_pipeline);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported() && !emulated_mesh) {
        // A draw passes the pipeline it looked up: a second lookup was ~2.5% of the back thread.
        GraphicsPipeline* const gp =
            draw_pipeline != nullptr ? draw_pipeline : pipeline_cache.CurrentGraphicsPipeline();
        if (gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = maxwell3d->regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        scheduler.NoteRenderPassStores();
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

namespace {
/// Active guest render target extent when surface_clip is unset (0x0), or {0, 0} if targetless.
VkExtent2D FallbackGuestSurfaceExtent(const Tegra::Engines::Maxwell3D::Regs& regs) {
    const u32 count = (std::min<u32>)(regs.rt_control.count, Tegra::Engines::Maxwell3D::Regs::NumRenderTargets);
    for (u32 i = 0; i < count; ++i) {
        const auto& rt = regs.rt[regs.rt_control.Map(i)];
        if (rt.Address() != 0 && rt.format != Tegra::RenderTargetFormat::NONE && rt.width != 0 &&
            rt.height != 0) {
            return {rt.width, rt.height};
        }
    }
    if (regs.zeta_enable != 0 && regs.zeta.Address() != 0 && regs.zeta_size.width != 0 &&
        regs.zeta_size.height != 0) {
        return {regs.zeta_size.width, regs.zeta_size.height};
    }
    return {};
}
} // Anonymous namespace

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    maxwell3d->dirty.flags[Dirty::Scissors] = true;

    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    VkExtent2D fallback_extent{};
    if (!regs.viewport_scale_offset_enabled &&
        (regs.surface_clip.width == 0 || regs.surface_clip.height == 0)) {
        fallback_extent = FallbackGuestSurfaceExtent(regs);
        if (fallback_extent.width == 0 || fallback_extent.height == 0) {
            if (const Framebuffer* const fb = texture_cache.GetFramebuffer()) {
                fallback_extent = fb->RenderArea();
            }
        } else {
            fallback_extent.width =
                static_cast<u32>((std::max)(1.0f, static_cast<float>(fallback_extent.width) * scale));
            fallback_extent.height =
                static_cast<u32>((std::max)(1.0f, static_cast<float>(fallback_extent.height) * scale));
        }
    }
    if (DynamicStateCacheEnabled()) {
        static_assert(sizeof(Maxwell::WindowOrigin) == 4 && sizeof(Maxwell::SurfaceClip) == 8 &&
                      sizeof(Maxwell::DepthMode) == 4 && Maxwell::NumViewports == 16);
        std::array<u32, std::tuple_size_v<decltype(dyn_cache.viewports)>> key;
        u32* out = key.data();
        AppendWords(out, regs.viewport_scale_offset_enabled);
        AppendWords(out, regs.window_origin);
        AppendWords(out, regs.surface_clip);
        AppendWords(out, fallback_extent.width);
        AppendWords(out, fallback_extent.height);
        AppendWords(out, regs.depth_mode);
        AppendWords(out, scale);
        AppendWords(out, oot_jitter_applied[0]);
        AppendWords(out, oot_jitter_applied[1]);
        AppendWords(out, camera_motion.applied[0]);
        AppendWords(out, camera_motion.applied[1]);
        for (const auto& transform : regs.viewport_transform) {
            AppendWords(out, transform.scale_x);
            AppendWords(out, transform.scale_y);
            AppendWords(out, transform.scale_z);
            AppendWords(out, transform.translate_x);
            AppendWords(out, transform.translate_y);
            AppendWords(out, transform.translate_z);
            AppendWords(out, transform.swizzle.raw);
        }
        if (!DynamicStateChanged(DynSlot::Viewports, dyn_cache.viewports, key)) {
            return;
        }
    }

    if (!regs.viewport_scale_offset_enabled) {
        const bool use_clip = regs.surface_clip.width != 0 && regs.surface_clip.height != 0;
        float x = use_clip ? static_cast<float>(regs.surface_clip.x) * scale : 0.0f;
        float y = use_clip ? static_cast<float>(regs.surface_clip.y) * scale : 0.0f;
        float width = use_clip
                          ? (std::max)(1.0f, static_cast<float>(regs.surface_clip.width) * scale)
                          : (std::max)(1.0f, static_cast<float>(fallback_extent.width));
        float height = use_clip
                           ? (std::max)(1.0f, static_cast<float>(regs.surface_clip.height) * scale)
                           : (std::max)(1.0f, static_cast<float>(fallback_extent.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            const float clip_h =
                use_clip ? static_cast<float>(regs.surface_clip.height) * scale : height;
            y = clip_h - y;
            height = -height;
        }
        VkViewport viewport{
            .x = x + oot_jitter_applied[0],
            .y = y + oot_jitter_applied[1],
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scheduler.Record([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    std::array viewport_list{
        GetViewportState(device, regs, 0, scale),  GetViewportState(device, regs, 1, scale),
        GetViewportState(device, regs, 2, scale),  GetViewportState(device, regs, 3, scale),
        GetViewportState(device, regs, 4, scale),  GetViewportState(device, regs, 5, scale),
        GetViewportState(device, regs, 6, scale),  GetViewportState(device, regs, 7, scale),
        GetViewportState(device, regs, 8, scale),  GetViewportState(device, regs, 9, scale),
        GetViewportState(device, regs, 10, scale), GetViewportState(device, regs, 11, scale),
        GetViewportState(device, regs, 12, scale), GetViewportState(device, regs, 13, scale),
        GetViewportState(device, regs, 14, scale), GetViewportState(device, regs, 15, scale),
    };
    if (oot_jitter_applied != std::array<f32, 2>{}) {
        for (VkViewport& viewport : viewport_list) {
            viewport.x += oot_jitter_applied[0];
            viewport.y += oot_jitter_applied[1];
        }
    }
    // The host's jitter for NVIDIA DLSS (NoteCameraMotionDraw), in host pixels; added after the
    // scaling so it isn't rounded away.
    if (camera_motion.applied[0] != 0.0f || camera_motion.applied[1] != 0.0f) {
        for (VkViewport& viewport : viewport_list) {
            viewport.x += camera_motion.applied[0];
            viewport.y += camera_motion.applied[1];
        }
    }
    scheduler.Record([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    const bool rescaling = texture_cache.IsRescaling();
    const u32 up_scale = rescaling ? Settings::values.resolution_info.up_scale : 1U;
    const u32 down_shift = rescaling ? Settings::values.resolution_info.down_shift : 0U;
    VkExtent2D fallback_extent{};
    if (!regs.viewport_scale_offset_enabled &&
        (regs.surface_clip.width == 0 || regs.surface_clip.height == 0)) {
        fallback_extent = FallbackGuestSurfaceExtent(regs);
        if (fallback_extent.width == 0 || fallback_extent.height == 0) {
            if (const Framebuffer* const fb = texture_cache.GetFramebuffer()) {
                fallback_extent = fb->RenderArea();
            }
        } else if (rescaling) {
            fallback_extent.width = Settings::values.resolution_info.ScaleUp(fallback_extent.width);
            fallback_extent.height =
                Settings::values.resolution_info.ScaleUp(fallback_extent.height);
        }
    }
    if (DynamicStateCacheEnabled()) {
        static_assert(sizeof(Maxwell::ScissorTest) == 16 && Maxwell::NumViewports == 16);
        std::array<u32, std::tuple_size_v<decltype(dyn_cache.scissors)>> key;
        u32* out = key.data();
        AppendWords(out, regs.viewport_scale_offset_enabled);
        AppendWords(out, regs.window_origin);
        AppendWords(out, regs.surface_clip);
        AppendWords(out, fallback_extent.width);
        AppendWords(out, fallback_extent.height);
        AppendWords(out, up_scale);
        AppendWords(out, down_shift);
        for (const auto& scissor : regs.scissor_test) {
            std::memcpy(out, &scissor, 3 * sizeof(u32));
            out += 3;
        }
        if (!DynamicStateChanged(DynSlot::Scissors, dyn_cache.scissors, key)) {
            return;
        }
    }
    if (!regs.viewport_scale_offset_enabled) {
        const bool use_clip = regs.surface_clip.width != 0 && regs.surface_clip.height != 0;
        const auto scale_val = [&](u32 val) -> u32 {
            return rescaling ? Settings::values.resolution_info.ScaleUp(val) : val;
        };
        const u32 x = use_clip ? scale_val(regs.surface_clip.x) : 0U;
        u32 y = use_clip ? scale_val(regs.surface_clip.y) : 0U;
        const u32 width = (std::max)(1u, use_clip ? scale_val(regs.surface_clip.width)
                                                  : fallback_extent.width);
        const u32 height = (std::max)(1u, use_clip ? scale_val(regs.surface_clip.height)
                                                   : fallback_extent.height);
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            const u32 clip_h = use_clip ? scale_val(regs.surface_clip.height) : height;
            y = clip_h > y + height ? clip_h - (y + height) : 0U;
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width = width;
        scissor.extent.height = height;
        scheduler.Record([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    scheduler.Record([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    // The slope factor depends on whether the target is rescaled, so a change of that
    // re-records the bias even when the guest registers did not change.
    const bool bias_rescaling = texture_cache.IsRescaling();
    if (!state_tracker.TouchDepthBias() && bias_rescaling == depth_bias_rescaling) {
        return;
    }
    depth_bias_rescaling = bias_rescaling;
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }
    float factor = regs.slope_scale_depth_bias;
    if (texture_cache.IsRescaling()) {
        // The slope term is per pixel, so a rescaled target needs it scaled back to guest pixels
        // (zortos293/Lichii#1925): shadow maps showed self-shadow stripes above 1x.
        factor *= Settings::values.resolution_info.up_factor;
    }

    scheduler.Record([constant = units, clamp = regs.depth_bias_clamp, factor,
                      this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    scheduler.Record(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    scheduler.Record([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    if (!DynamicStateChanged(DynSlot::LineWidth, dyn_cache.line_width, width)) {
        return;
    }
    scheduler.Record([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    const u32 cull_mode = regs.gl_cull_test_enabled
                              ? static_cast<u32>(MaxwellToVK::CullFace(regs.gl_cull_face))
                              : static_cast<u32>(VK_CULL_MODE_NONE);
    if (!DynamicStateChanged(DynSlot::CullMode, dyn_cache.cull_mode, cull_mode)) {
        return;
    }
    scheduler.Record([cull_mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(static_cast<VkCullModeFlags>(cull_mode));
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    if (!DynamicStateChanged(DynSlot::DepthBoundsTestEnable, dyn_cache.depth_bounds_test,
                             static_cast<u32>(enabled))) {
        return;
    }
    scheduler.Record([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    if (!DynamicStateChanged(DynSlot::DepthTestEnable, dyn_cache.depth_test,
                             static_cast<u32>(regs.depth_test_enable))) {
        return;
    }
    scheduler.Record([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    if (!DynamicStateChanged(DynSlot::DepthWriteEnable, dyn_cache.depth_write,
                             static_cast<u32>(regs.depth_write_enabled))) {
        return;
    }
    scheduler.Record([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        const auto topology = MaxwellToVK::PrimitiveTopology(device, maxwell3d->draw_manager.draw_state.topology);
        enable = IsPrimitiveRestartSupported(device, topology);
    }
    if (!DynamicStateChanged(DynSlot::PrimitiveRestartEnable, dyn_cache.primitive_restart,
                             static_cast<u32>(enable))) {
        return;
    }

    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    if (!DynamicStateChanged(DynSlot::RasterizerDiscardEnable, dyn_cache.rasterizer_discard,
                             static_cast<u32>(regs.rasterize_enable == 0))) {
        return;
    }
    scheduler.Record([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    scheduler.Record([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    scheduler.Record([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    scheduler.Record([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index =
        u32(RasterTopologyOf(regs, maxwell3d->draw_manager.draw_state.topology));
    const u32 enable = enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]];
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable != 0); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    bool enable = regs.logic_op.enable != 0;
    if (enable) {
        using VideoCore::Surface::IsPixelFormatSRGB;
        using VideoCore::Surface::PixelFormat;
        using VideoCore::Surface::TryPixelFormatFromRenderTargetFormat;
        const auto is_float_or_srgb = [](PixelFormat pf) {
            if (IsPixelFormatSRGB(pf)) {
                return true;
            }
            switch (pf) {
            case PixelFormat::R16G16B16A16_FLOAT:
            case PixelFormat::R16G16B16X16_FLOAT:
            case PixelFormat::B10G11R11_FLOAT:
            case PixelFormat::E5B9G9R9_FLOAT:
            case PixelFormat::R32G32B32A32_FLOAT:
            case PixelFormat::R32G32B32_FLOAT:
            case PixelFormat::R32G32_FLOAT:
            case PixelFormat::R32_FLOAT:
            case PixelFormat::R16G16_FLOAT:
            case PixelFormat::R16_FLOAT:
            case PixelFormat::BC6H_UFLOAT:
            case PixelFormat::BC6H_SFLOAT:
                return true;
            default:
                return false;
            }
        };
        const u32 rt_count =
            (std::min<u32>)(regs.rt_control.count, Maxwell::NumRenderTargets);
        for (u32 i = 0; i < rt_count; ++i) {
            const auto& rt = regs.rt[regs.rt_control.Map(i)];
            if (rt.Address() == 0 || rt.format == Tegra::RenderTargetFormat::NONE) {
                continue;
            }
            const auto pf = TryPixelFormatFromRenderTargetFormat(rt.format);
            if (!pf || is_float_or_srgb(*pf)) {
                enable = false;
                break;
            }
        }
    }
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    scheduler.Record(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs,
                                                   const GraphicsPipeline* draw_pipeline) {
    // The enable also depends on the pipeline writing colour 0, which the dirty flag does not
    // track. Track last_a2c_supported so switching from depth-only to color refreshes A2C.
    static const bool refresh{std::getenv("EDEN_NX2_A2C_REFRESH") != nullptr};
    const bool supported = draw_pipeline != nullptr && draw_pipeline->SupportsAlphaToCoverage();
    const u64 epoch = state_tracker.CommandBufferEpoch();
    const bool support_changed = supported != last_a2c_supported || epoch != last_a2c_epoch;
    if (!state_tracker.TouchAlphaToCoverageEnable() && !support_changed && !refresh) {
        return;
    }
    last_a2c_supported = supported;
    last_a2c_epoch = epoch;
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    const bool enable = supported && regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    alpha_to_coverage_enable = enable;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs,
                                              const GraphicsPipeline* draw_pipeline) {
    const bool supported = draw_pipeline != nullptr && draw_pipeline->SupportsAlphaToOne();
    const u64 epoch = state_tracker.CommandBufferEpoch();
    const bool support_changed = supported != last_a2o_supported || epoch != last_a2o_epoch;
    if (!state_tracker.TouchAlphaToOneEnable() && !support_changed) {
        return;
    }
    last_a2o_supported = supported;
    last_a2o_epoch = epoch;
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    const bool enable = supported && regs.anti_alias_alpha_control.alpha_to_one != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    scheduler.Record([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::ComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    scheduler.Record(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        scheduler.Record([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        scheduler.Record([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    scheduler.Record([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs,
                                      const GraphicsPipeline* pipeline) {
    // The write mask also excludes components the fragment shader doesn't output, so it has to be
    // recorded again whenever the bound pipeline outputs a different set of components.
    const u64 epoch = state_tracker.CommandBufferEpoch();
    const bool epoch_changed = epoch != last_blend_epoch;
    const u32 color_output_mask = pipeline ? pipeline->ColorOutputMask() : ~0U;
    const bool color_output_changed = color_output_mask != last_color_output_mask || epoch_changed;

    const auto is_src1 = [](Maxwell::Blend::Factor factor) {
        const VkBlendFactor vk = MaxwellToVK::BlendFactor(factor);
        return vk == VK_BLEND_FACTOR_SRC1_COLOR ||
               vk == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ||
               vk == VK_BLEND_FACTOR_SRC1_ALPHA ||
               vk == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    };
    const auto uses_dual_src = [&](const auto& guest_blend) {
        return is_src1(guest_blend.color_source) || is_src1(guest_blend.color_dest) ||
               is_src1(guest_blend.alpha_source) || is_src1(guest_blend.alpha_dest);
    };

    u32 integer_rt_mask = 0;
    u32 dual_src_mask = 0;
    const u32 rt_count = (std::min<u32>)(regs.rt_control.count, Maxwell::NumRenderTargets);
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const u32 slot = index < rt_count ? regs.rt_control.Map(index) : static_cast<u32>(index);
        if (slot < Maxwell::NumRenderTargets &&
            regs.rt[slot].format != Tegra::RenderTargetFormat::NONE) {
            const auto rt_format =
                VideoCore::Surface::TryPixelFormatFromRenderTargetFormat(regs.rt[slot].format);
            if (!rt_format || VideoCore::Surface::IsPixelFormatInteger(*rt_format)) {
                integer_rt_mask |= 1U << index;
            }
        }
        if (index != 0) {
            const bool src1 = regs.blend_per_target_enabled
                                  ? uses_dual_src(regs.blend_per_target[index])
                                  : (!(regs.iterated_blend.enable &&
                                       Settings::values.use_squashed_iterated_blend) &&
                                     uses_dual_src(regs.blend));
            if (src1) {
                dual_src_mask |= 1U << index;
            }
        }
    }
    const bool blend_enable_override_changed =
        integer_rt_mask != last_integer_rt_mask || dual_src_mask != last_dual_src_mask ||
        epoch_changed;

    if (!state_tracker.TouchBlending() && !color_output_changed &&
        !blend_enable_override_changed) {
        return;
    }
    last_blend_epoch = epoch;

    if (state_tracker.TouchColorMask() || color_output_changed) {
        last_color_output_mask = color_output_mask;
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
            current &= (color_output_mask >> (index * 4)) & 0xF;
        }
        scheduler.Record([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable() || blend_enable_override_changed) {
        last_integer_rt_mask = integer_rt_mask;
        last_dual_src_mask = dual_src_mask;
        const u32 disable_mask = integer_rt_mask | dual_src_mask;
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const bool blocked = ((disable_mask >> index) & 1U) != 0;
            setup_enables[index] =
                (!blocked && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        scheduler.Record([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

size_t RasterizerVulkan::VertexInputKeyHash::operator()(const VertexInputKey& key) const noexcept {
    static_assert((sizeof(VertexInputKey) % sizeof(u64)) == 0);
    constexpr size_t NumWords = sizeof(VertexInputKey) / sizeof(u64);
    u64 h0 = 0x9e3779b97f4a7c15ULL;
    u64 h1 = 0xbf58476d1ce4e5b9ULL;
    u64 h2 = 0x94d049bb133111ebULL;
    u64 h3 = 0xcbf29ce484222325ULL;
    const u8* const bytes = reinterpret_cast<const u8*>(key.data());
    size_t i = 0;
    for (; i + 4 <= NumWords; i += 4) {
        u64 w0, w1, w2, w3;
        std::memcpy(&w0, bytes + (i + 0) * sizeof(u64), sizeof(u64));
        std::memcpy(&w1, bytes + (i + 1) * sizeof(u64), sizeof(u64));
        std::memcpy(&w2, bytes + (i + 2) * sizeof(u64), sizeof(u64));
        std::memcpy(&w3, bytes + (i + 3) * sizeof(u64), sizeof(u64));
        h0 = (h0 ^ w0) * 0x100000001b3ULL;
        h1 = (h1 ^ w1) * 0xbf58476d1ce4e5b9ULL;
        h2 = (h2 ^ w2) * 0x94d049bb133111ebULL;
        h3 = (h3 ^ w3) * 0x9e3779b97f4a7c15ULL;
    }
    for (; i < NumWords; ++i) {
        u64 w;
        std::memcpy(&w, bytes + i * sizeof(u64), sizeof(u64));
        h0 = (h0 ^ w) * 0x100000001b3ULL;
    }
    u64 hash = h0 ^ (h1 * 0x9e3779b97f4a7c15ULL) ^ (h2 * 0xbf58476d1ce4e5b9ULL) ^ h3;
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    return static_cast<size_t>(hash);
}

u32 RasterizerVulkan::UpdateVertexRepackLayout(bool dynamic_input) {
    const auto& regs = maxwell3d->regs;
    if (VertexInputCacheEnabled()) {
        if (repack_key_valid && repack_key_dynamic == dynamic_input &&
            !maxwell3d->dirty.flags[Dirty::VertexInput] &&
            !maxwell3d->dirty.flags[VideoCommon::Dirty::VertexBuffers]) {
            return 0;
        }
        static_assert(Maxwell::NumVertexAttributes == 32 && Maxwell::NumVertexArrays == 32);
        std::array<u32, 64> key;
        for (size_t index = 0; index < 32; ++index) {
            key[index] = regs.vertex_attrib_format[index].hex;
            key[32 + index] = regs.vertex_streams[index].stride.Value();
        }
        if (repack_key_valid && repack_key_dynamic == dynamic_input && key == repack_key) {
            return 0;
        }
        repack_key = key;
        repack_key_dynamic = dynamic_input;
        repack_key_valid = true;
    }
    return buffer_cache_runtime.SetVertexRepackLayout(regs, dynamic_input);
}

void RasterizerVulkan::BuildVertexInputLayout(const Tegra::Engines::Maxwell3D::Regs& regs,
                                              u32 max_attributes, u32 max_bindings,
                                              VertexInputLayout& layout) const {
    auto& bindings = layout.bindings;
    auto& attributes = layout.attributes;
    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        const VkFormat format =
            attribute.type == Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway
                ? VK_FORMAT_UNDEFINED
                : MaxwellToVK::VertexFormat(device, attribute.type, attribute.size);
        if (format == VK_FORMAT_UNDEFINED) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan,
                            "Dynamic vertex attribute {} left out (type {}, size {}, stream {})",
                            index, static_cast<u32>(attribute.type.Value()),
                            static_cast<u32>(attribute.size.Value()), binding);
            }
            continue;
        }
        attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = format,
            .offset = attribute.offset +
                      (buffer_cache_runtime.IsVertexRepacked(binding) && attribute.offset >= 65 ? 3 : 0),
        });
    }

    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = buffer_cache_runtime.IsVertexRepacked(binding) ? 72U : input_binding.stride.Value(),
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    }
}

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{maxwell3d->dirty.flags};
    const bool vertex_input_dirty = dirty[Dirty::VertexInput];
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    if (!vertex_input_dirty && !vertex_buffers_dirty) {
        return;
    }
    dirty[Dirty::VertexInput] = false;

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));

    if (!vertex_input_keep_mask_built) {
        vertex_input_keep_mask.set();
        vertex_input_keep_mask[Dirty::VertexInput] = false;
        for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
            vertex_input_keep_mask[Dirty::VertexAttribute0 + index] = false;
        }
        for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
            vertex_input_keep_mask[Dirty::VertexBinding0 + index] = false;
        }
        vertex_input_keep_mask_built = true;
    }
    dirty &= vertex_input_keep_mask;

    if (VertexInputCacheEnabled()) {
        static_assert(Maxwell::NumVertexAttributes <= 32 && Maxwell::NumVertexArrays <= 32);
        VertexInputKey key{};
        for (u32 index = 0; index < max_attributes; ++index) {
            key[index] = regs.vertex_attrib_format[index].hex;
        }
        u32 repacked = 0;
        for (u32 binding = 0; binding < max_bindings; ++binding) {
            const auto& stream = regs.vertex_streams[binding];
            const bool is_instanced = regs.vertex_stream_instances.IsInstancingEnabled(binding);
            key[32 + binding * 2] =
                (stream.stride.Value() & 0xffffU) | (is_instanced ? 0x80000000U : 0U);
            key[32 + binding * 2 + 1] = is_instanced ? stream.frequency : 0U;
            repacked |= buffer_cache_runtime.IsVertexRepacked(binding) ? 1U << binding : 0U;
        }
        key[32 + 32 * 2] = repacked;

        const VertexInputLayout* layout = nullptr;
        if (last_vertex_input_layout != nullptr && key == last_vertex_input_key) {
            layout = last_vertex_input_layout;
        } else {
            const size_t hash = VertexInputKeyHash{}(key);
            const size_t fast_slot = hash & (VertexInputFastSlots - 1);
            if (const auto* const fast_entry = vertex_input_fast_cache[fast_slot];
                fast_entry != nullptr && fast_entry->first == key) {
                layout = fast_entry->second.get();
            } else if (const auto it = vertex_input_layouts.find(key);
                       it != vertex_input_layouts.end()) {
                vertex_input_fast_cache[fast_slot] = &*it;
                layout = it->second.get();
            } else if (vertex_input_layouts.size() < 4096) {
                auto built = std::make_unique<VertexInputLayout>();
                BuildVertexInputLayout(regs, max_attributes, max_bindings, *built);
                const auto [ins_it, _] = vertex_input_layouts.emplace(key, std::move(built));
                vertex_input_fast_cache[fast_slot] = &*ins_it;
                layout = ins_it->second.get();
            }
        }
        if (layout != nullptr) {
            last_vertex_input_key = key;
            last_vertex_input_layout = layout;
            if (DynamicStateChanged(DynSlot::VertexInput, dyn_cache.vertex_input,
                                    static_cast<const void*>(layout))) {
                scheduler.Record([layout](vk::CommandBuffer cmdbuf) {
                    cmdbuf.SetVertexInputEXT(layout->bindings, layout->attributes);
                });
            }
            return;
        }
    }

    VertexInputLayout layout;
    BuildVertexInputLayout(regs, max_attributes, max_bindings, layout);
    ForgetDynamicState(DynSlot::VertexInput);
    scheduler.Record([bindings = layout.bindings, attributes = layout.attributes](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetVertexInputEXT(bindings, attributes); });
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
