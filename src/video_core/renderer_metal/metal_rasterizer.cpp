// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/perf_report.h"
#include "common/perf_sched.h"
#include "common/alignment.h"
#include "common/cityhash.h"
#include "common/frame_deadline.h"
#include "common/logging.h"
#include "common/platform_events.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "video_core/buffer_cache/cpu_read_stats.h"
#include "video_core/buffer_cache/deferred_page_caching.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/feature_census.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/host1x/host1x.h"
#include "video_core/indirect_reset.h"
#include "video_core/memory_manager.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/nx2_perf.h"
#include "video_core/dirty_flags.h"
#include "video_core/query_cache/query_cache.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/oot_network.h"
#include "video_core/render_probe.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_draw_desc.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_mesh_indirect.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_bind_memo.h"
#include "video_core/renderer_metal/metal_rasterizer.h"
#include "video_core/renderer_metal/metal_raster_expand.h"
#include "video_core/renderer_metal/metal_raster_patch.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/renderer_metal/metal_work_policy.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/shader_handle_program.h"
#include "video_core/textures/texture.h"

namespace Metal {

namespace {

// EDEN_NX2_METAL_DRAW_SCRATCH=0: PrepareDraw allocates its per-draw lists afresh each draw.
bool DrawScratchEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DRAW_SCRATCH");
        const bool on = value == nullptr || value[0] != '0';
        LOG_INFO(Render_Metal, "Metal: per-draw lists reuse their storage: {} "
                               "(EDEN_NX2_METAL_DRAW_SCRATCH)",
                 on ? "on" : "off");
        return on;
    }();
    return enabled;
}

// EDEN_NX2_METAL_BIND_SCRATCH=0: BindStageResources allocates its slot lists and tables afresh
// for each stage of each draw and dispatch.
bool BindScratchEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_BIND_SCRATCH");
        const bool on = value == nullptr || value[0] != '0';
        LOG_INFO(Render_Metal, "Metal: stage binding lists reuse their storage: {} "
                               "(EDEN_NX2_METAL_BIND_SCRATCH)",
                 on ? "on" : "off");
        return on;
    }();
    return enabled;
}

template <typename T>
void ClearScratch(std::vector<T>& list) {
    list.clear();
}

template <typename T, size_t N>
void ClearScratch(std::array<std::vector<T>, N>& lists) {
    for (std::vector<T>& list : lists) {
        list.clear();
    }
}

/// A list taken from `pool` for one PrepareDraw (empty, with the pool's storage) and given
/// back, emptied, when it ends. A nested one finds the pool taken and starts with nothing.
template <typename T>
class ScratchList {
public:
    explicit ScratchList(T& pool_, bool enabled = DrawScratchEnabled())
        : pool{enabled ? &pool_ : nullptr} {
        if (pool != nullptr) {
            std::swap(value, *pool);
            ClearScratch(value);
        }
    }
    ~ScratchList() {
        if (pool != nullptr) {
            ClearScratch(value);
            std::swap(value, *pool);
        }
    }
    ScratchList(const ScratchList&) = delete;
    ScratchList& operator=(const ScratchList&) = delete;

    T value{};

private:
    T* pool;
};

/// IEEE half precision, rounded to nearest (tessellation factors are halves; 0..64 here).
[[nodiscard]] u16 FloatToHalf(float value) {
    if (!(value > 0.0f)) {
        return 0;
    }
    value = (std::min)(value, 65504.0f);
    const u32 bits = std::bit_cast<u32>(value);
    const int exponent = static_cast<int>((bits >> 23) & 0xFF) - 127;
    if (exponent < -14) {
        // Subnormal halves: multiples of 2^-24.
        return static_cast<u16>(std::lround(value * 16777216.0f));
    }
    u32 half = (static_cast<u32>(exponent + 15) << 10) | ((bits & 0x7FFFFF) >> 13);
    // Round to nearest even; a carry out of the mantissa moves the exponent, as it should.
    const u32 rest = bits & 0x1FFF;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1) != 0)) {
        ++half;
    }
    return static_cast<u16>(half);
}
/// Logs a message the first time `flag` is seen unset.
#define LOG_ONCE(flag, ...)                                                                        \
    do {                                                                                           \
        static std::atomic_bool flag{false};                                                       \
        if (!flag.exchange(true)) {                                                                \
            LOG_WARNING(Render_Metal, __VA_ARGS__);                                                \
        }                                                                                          \
    } while (false)

/// Apple GPUs' largest render target side (MTLRenderPassDescriptor.renderTargetWidth/Height).
constexpr u32 MaxRenderTargetSize = 16384;

// A draw without render targets gets a framebuffer of the surface clip's size, which Ocarina of
// Time leaves at 32768x32768 for its occlusion draws: past Metal's render target limit, and a
// tile-based GPU pays for every tile. Sized by scissor 0 instead, as Vulkan's
// EDEN_NX2_TARGETLESS_REACH does on Apple (the same switch, on by default).
bool TargetlessReachEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_TARGETLESS_REACH");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

VideoCommon::Extent2D TargetlessReach(const Maxwell& regs, bool is_rescaling) {
    const auto& resolution = Settings::values.resolution_info;
    const auto scale = [&](u32 value) { return is_rescaling ? resolution.ScaleUp(value) : value; };
    u32 width = regs.surface_clip.width;
    u32 height = regs.surface_clip.height;
    const auto& scissor = regs.scissor_test[0];
    if (regs.viewport_scale_offset_enabled && scissor.enable) {
        const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
        const u32 min_y = static_cast<u32>(scissor.min_y.Value());
        const u32 max_y = lower_left ? (height > min_y ? height - min_y : 0U)
                                     : static_cast<u32>(scissor.max_y.Value());
        width = (std::min)(width, static_cast<u32>(scissor.max_x.Value()));
        height = (std::min)(height, max_y);
    }
    if (scale(width) > MaxRenderTargetSize || scale(height) > MaxRenderTargetSize) {
        LOG_ONCE(targetless_clamped,
                 "Metal: draw without render targets reaches {}x{}, clamped to {} (its scissor "
                 "may fall outside)",
                 scale(width), scale(height), MaxRenderTargetSize);
    }
    return {std::clamp(scale(width), 1U, MaxRenderTargetSize),
            std::clamp(scale(height), 1U, MaxRenderTargetSize)};
}

bool EnvironmentOn(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

// EDEN_NX2_METAL_CACHE_PATHS=0: guest reads, guest writes and inline writes take the plain
// paths this backend started with (every question under the cache locks, collected writes
// applied one at a time, reads landed by the GPU thread, no fence quiet check, every inline
// write invalidating every cache at once). On, they are the Vulkan rasterizer's.
bool CachePathsEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_CACHE_PATHS");
    return enabled;
}

// EDEN_NX2_METAL_CLEAR_DRAWS=0: a clear of part of a target, of some colour channels or of some
// stencil bits clears what a load action can instead, and integer targets take the register's
// floats, as this backend started.
bool ClearDrawsEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_CLEAR_DRAWS");
    return enabled;
}

/// The rectangle (x, y, width, height) a clear covers of a target of the given size: scissor 0
/// when the clear uses it, as Vulkan's Clear.
std::array<u32, 4> ClearArea(const Maxwell& regs, bool is_rescaling, u32 width, u32 height) {
    const auto& scissor = regs.scissor_test[0];
    if (regs.clear_control.use_scissor == 0 || scissor.enable == 0) {
        return {0, 0, width, height};
    }
    const auto& resolution = Settings::values.resolution_info;
    const auto scale = [&](s32 value) {
        const u32 positive = static_cast<u32>((std::max)(value, 0));
        return static_cast<s32>(is_rescaling ? resolution.ScaleUp(positive) : positive);
    };
    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = static_cast<s32>(regs.surface_clip.height);
    const s32 min_y = lower_left ? clip_height - static_cast<s32>(scissor.max_y)
                                 : static_cast<s32>(scissor.min_y.Value());
    const s32 max_y = lower_left ? clip_height - static_cast<s32>(scissor.min_y)
                                 : static_cast<s32>(scissor.max_y.Value());
    const s32 limit_x = static_cast<s32>(width);
    const s32 limit_y = static_cast<s32>(height);
    const s32 x0 = std::clamp(scale(static_cast<s32>(scissor.min_x.Value())), 0, limit_x);
    const s32 y0 = std::clamp(scale(min_y), 0, limit_y);
    const s32 x1 = std::clamp(scale(static_cast<s32>(scissor.max_x.Value())), x0, limit_x);
    const s32 y1 = std::clamp(scale(max_y), y0, limit_y);
    return {static_cast<u32>(x0), static_cast<u32>(y0), static_cast<u32>(x1 - x0),
            static_cast<u32>(y1 - y0)};
}

// EDEN_NX2_METAL_FRAGMENT_BARRIER=0: the guest's fragment barrier does nothing, as this backend
// started (a draw sampling the target of its own pass reads what the pass before it stored).
bool FragmentBarrierEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_FRAGMENT_BARRIER");
    return enabled;
}

bool TextureReadFilter() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_TEXTURE_READ_FILTER");
    return enabled;
}

// Vulkan's experiment EDEN_NX2_INVALIDATE_BEFORE_DISPATCH=1: the collected guest CPU writes are
// applied before every dispatch and draw (=list does it once per command list, in DmaPusher, for
// both backends).
bool InvalidateBeforeUse() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INVALIDATE_BEFORE_DISPATCH");
        return value != nullptr && value[0] != '0' && std::string_view{value} != "list";
    }();
    return enabled;
}

// EDEN_NX2_METAL_LIST_COPIES=0: command lists the GPU wrote are read after a full sync and flush
// of their page, as this backend started; on, as Vulkan reads them (from copies made after the
// dispatch or DMA copy that wrote them).
bool ListCopiesEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_LIST_COPIES");
    return enabled;
}

/// False when no page of the range was ever written by the GPU (no lock needed to know).
bool MayBeGpuModified(DAddr addr, u64 size) {
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return true;
    }
    const DAddr last = addr + size - 1;
    for (DAddr page = addr >> Tegra::Nx2WrittenPageBits;
         page <= (last >> Tegra::Nx2WrittenPageBits); ++page) {
        if (Tegra::Nx2IsGpuWritten(page << Tegra::Nx2WrittenPageBits)) {
            return true;
        }
    }
    return false;
}
} // Anonymous namespace

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    if (!ListCopiesEnabled()) {
        return buffer_cache.DMACopy(src_address, dest_address, amount);
    }
    // As Vulkan's: command list pages the copy writes are copied for the split's front.
    buffer_cache.Nx2CollectListCopies(true);
    const bool copied = buffer_cache.DMACopy(src_address, dest_address, amount);
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick())) {
        scheduler.Flush(CommitReason::ListCopy);
    }
    buffer_cache.Nx2CollectListCopies(false);
    return copied;
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
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
        if (offset % VideoCore::Surface::BytesPerBlock(image->info.format)) {
            return false;
        }
        buffer_cache.runtime.NotifyWrite(buffer->Handle(), offset, buffer_size);
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

RasterizerMetal::RasterizerMetal(Tegra::GPU& gpu_,
                                 Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                 const Device& device_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_}, scheduler{scheduler_},
      staging_pool{device_, scheduler_},
      texture_cache_runtime{device_, scheduler_, staging_pool},
      texture_cache{texture_cache_runtime, device_memory_},
      buffer_cache_runtime{device_, scheduler_, staging_pool},
      buffer_cache{device_memory_, buffer_cache_runtime}, pipeline_cache{device_memory_, device_},
      ocarina{device_, scheduler_}, ocarina_temporal{device_, scheduler_},
      accelerate_dma{buffer_cache, texture_cache, scheduler_},
      fence_manager{*this, gpu_, texture_cache, buffer_cache, query_cache, scheduler_},
      frame_dump{device_, scheduler_}, flash_watch{device_, scheduler_} {
    pipeline_cache.SetSkipCheck([this] { return DrawMayBeSkipped(); });
    buffer_cache_runtime.SetWriteHook(
        [](void* user, BufferHandle buf, u64 offset, u64 size) {
            static_cast<RasterizerMetal*>(user)->InvalidateConvertedIndices(buf, offset, size);
        },
        this);
    Common::PlatformEvents::SetGraphicsMemoryReport([this] { return GraphicsMemoryReport(); });
    // Fence work left pending (EDEN_NX2_FENCE_FLUSH_US) is committed when the split's back goes
    // idle, as Vulkan's deferred flushes; EDEN_NX2_IDLE_GRACE_US and EDEN_NX2_IDLE_HOLD_US work
    // through the same hooks.
    VideoCommon::GpuSplit::Get().SetIdleHook(
        [](void* argument, bool requested) {
            return static_cast<Scheduler*>(argument)->FlushDeferredFromIdle(requested);
        },
        [](void* argument) { return static_cast<Scheduler*>(argument)->DeferredFlushMayWait(); },
        &scheduler);
    VideoCommon::GpuSplit::Get().SetIdleBusy(
        [](void* argument) { return static_cast<Scheduler*>(argument)->GpuBehind(); });
    if (!CachePathsEnabled()) {
        LOG_WARNING(Render_Metal, "Metal: plain cache paths (EDEN_NX2_METAL_CACHE_PATHS=0)");
        return;
    }
    // Guest writes collected but not applied yet, for downloads that must not land over them.
    buffer_cache.SetPendingCpuWrites(
        [this](const std::function<void(DAddr, size_t)>& callback) {
            gpu.ForEachPendingCpuWrite(callback);
        });
    VideoCommon::GpuSplit::Get().SetFenceQuietCheck(
        [](void* argument, bool strict, u64* epoch) {
            return static_cast<RasterizerMetal*>(argument)->FencesQuiet(strict, *epoch);
        },
        this);
    // Whoever takes the buffer cache's lock applies the guest writes handed off meanwhile.
    buffer_cache.mutex.SetOwnerHook(
        [](void* argument) {
            auto& self = *static_cast<RasterizerMetal*>(argument);
            self.write_handoff.Drain([&self](DAddr addr, u64 size) {
                self.InvalidateConvertedIndicesAddr(addr, size);
                if (!self.buffer_cache.OnCPUWrite(addr, size)) {
                    return;
                }
                constexpr u64 page_size = VideoCommon::Nx2WriteHandoff::PageSize;
                for (DAddr page = addr; page < addr + size;) {
                    const DAddr page_end =
                        (std::min)(addr + size, Common::AlignDown(page, page_size) + page_size);
                    if (self.buffer_cache.OnCPUWrite(page, page_end - page)) {
                        Tegra::Nx2NoteDeferredCodeWrite(page, page_end - page);
                        self.gpu.CollectHandoffWrite(page, page_end - page);
                    }
                    page = page_end;
                }
            });
        },
        this);
}

RasterizerMetal::~RasterizerMetal() {
    buffer_cache_runtime.SetWriteHook(nullptr, nullptr);
    Common::PlatformEvents::SetGraphicsMemoryReport({});
    VideoCommon::GpuSplit::Get().SetIdleHook(nullptr, nullptr, nullptr);
    VideoCommon::GpuSplit::Get().SetIdleBusy(nullptr);
    if (CachePathsEnabled()) {
        VideoCommon::GpuSplit::Get().SetFenceQuietCheck(nullptr, nullptr);
    }
}

bool RasterizerMetal::BeginDraw(bool is_indexed, u32 instance_count) {
    auto& draw_state = maxwell3d->draw_manager.draw_state;
    grid_draw = {};
    // A vertex count GPU work wrote (nx2_indirect_grid.h, EDEN_NX2_INDIRECT_DRAWS=1). As Vulkan,
    // a plain draw takes it on the GPU (grid_arguments, EDEN_NX2_METAL_GPU_INDIRECT); a draw
    // whose vertices must be rewritten reads the words here, which waits for the GPU.
    if (!is_indexed && Tegra::Nx2IndirectGrid::AnyLive() &&
        Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(draw_state.vertex_buffer.count)) {
        const u32 count = draw_state.vertex_buffer.count;
        auto& grid = Tegra::Nx2IndirectGrid::Get();
        Tegra::Nx2IndirectGrid::LogCounts();
        const auto source = Tegra::Nx2IndirectGrid::MatchDrawCount(count);
        if (!source) {
            grid.skipped.fetch_add(1, std::memory_order_relaxed);
            RefuseDraw("GPU-counted draw that lost its source");
            return false;
        }
        const bool first_placeholder =
            draw_state.vertex_buffer.first ==
            Tegra::Nx2IndirectGrid::DrawPlaceholder(count & (Tegra::Nx2IndirectGrid::Slots - 1), 1);
        grid_draw = GridDraw{.wanted = true, .source = *source, .first_from_gpu = first_placeholder};
        static const bool on_gpu = EnvironmentOn("EDEN_NX2_METAL_GPU_INDIRECT");
        if (on_gpu && draw_state.topology != Maxwell::PrimitiveTopology::Points) {
            // Points are left to the CPU: Bananza's reset draw needs its count there.
            return true;
        }
        ReadGridWords();
    }
    if (PipelineCheckEnabled()) {
        // As Vulkan's Draw: past 16M vertices a first and count are what a rejected compute
        // program should have written, and indexed ones read far out of bounds.
        const u64 first = is_indexed ? draw_state.index_buffer.first
                                     : draw_state.vertex_buffer.first;
        const u64 count = is_indexed ? draw_state.index_buffer.count
                                     : draw_state.vertex_buffer.count;
        constexpr u64 MaxVertices = u64{1} << 24;
        if (first + count > MaxVertices) {
            RefuseDraw(is_indexed ? "index count past 16M" : "vertex count past 16M");
            return false;
        }
    }
    return !TryIndirectReset(is_indexed, instance_count);
}

void RasterizerMetal::ReadGridWords() {
    if (!grid_draw.wanted) {
        return;
    }
    auto& draw_state = maxwell3d->draw_manager.draw_state;
    std::array<u32, 2> words{};
    gpu_memory->ReadBlock(grid_draw.source, words.data(), sizeof(words));
    draw_state.vertex_buffer.count = words[0];
    if (grid_draw.first_from_gpu) {
        draw_state.vertex_buffer.first = words[1];
    }
    grid_draw = {};
    Tegra::Nx2IndirectGrid::Get().draws_refused.fetch_add(1, std::memory_order_relaxed);
}

bool RasterizerMetal::PrepareGridDraw(bool ahead) {
    const std::optional<DAddr> address = gpu_memory->GpuToCpuAddress(grid_draw.source);
    if (!address) {
        return false;
    }
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(
        grid_draw.source, 8, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
        VideoCommon::ObtainBufferOperation::DoNothing);
    if (buffer == nullptr || buffer->Handle() == nullptr) {
        return false;
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    grid_draw.arguments = staging_pool.Stream(Indirect::PlainWords * sizeof(u32), 16);
    const Indirect::GridParams params{
        .instances = grid_draw.instances,
        .first = draw_state.vertex_buffer.first,
        .first_from_gpu = grid_draw.first_from_gpu ? 1U : 0U,
        .base_instance = draw_state.base_instance,
    };
    scheduler.BeginCompute(grid_arguments_kernel.Get(), ahead);
    scheduler.BindBuffer(ShaderStage::Compute, 0, buffer->Handle(), offset);
    scheduler.BindBuffer(ShaderStage::Compute, 1, grid_draw.arguments.buffer,
                         static_cast<u32>(grid_draw.arguments.offset));
    scheduler.BindBytes(ShaderStage::Compute, 2, &params, static_cast<u32>(sizeof(params)));
    scheduler.Dispatch({1, 1, 1}, {1, 1, 1});
    return true;
}

bool RasterizerMetal::TryIndirectReset(bool is_indexed, u32 instance_count) {
    // Vulkan's TryNx2IndirectReset (EDEN_NX2_RESETFIX): Donkey Kong Bananza's pointer-list reset
    // draw, done as writes of the record to guest memory.
    static const bool enabled = CachePathsEnabled() && EnvironmentOn("EDEN_NX2_RESETFIX");
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
    if (!VideoCommon::NativeIndirectReset::Matches(code)) {
        return false;
    }
    gpu_memory->FlushCaching();
    const auto reject = [](std::string_view reason) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Metal, "Indirect reset left to the draw: {}", reason);
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
    const u64 count = draw.vertex_buffer.count;
    const auto range = VideoCommon::IndirectResetRange(
        descriptor[0], static_cast<u32>(descriptor[1]), draw.vertex_buffer.first, count);
    if (!range || !in_gpu_range(range->begin, range->size) ||
        !gpu_memory->IsFullyMappedRange(range->begin, range->size)) {
        return reject("list outside mapped descriptor");
    }
    std::vector<u64> pointers(count);
    gpu_memory->ReadBlock(range->begin, pointers.data(), range->size);
    static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
    for (const u64 pointer : pointers) {
        if (pointer == 0) {
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
    }
    const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
    for (const u64 pointer : pointers) {
        AccelerateInlineToMemory(pointer, bytes.size(), bytes);
    }
    LOG_ONCE(indirect_reset_done, "Metal: indirect reset done as {} record writes", count);
    Coverage::Note(Coverage::Work::Draw, Coverage::Path::Replaced);
    return true;
}

bool RasterizerMetal::SkipMeshDraws(bool emulated) {
    // Vulkan's EDEN_NX2_NO_MESH=1, EDEN_NX2_NO_MESH_EMULATION=1 (the draws through the compute
    // emulation only) and EDEN_NX2_NO_MESH_FILE=<path> (while it exists, checked every 1024 mesh
    // draws), read under EDEN_NX2_METAL_VULKAN_SWITCHES: mesh draws skipped for comparisons.
    static const char* const file = VulkanSwitch("EDEN_NX2_NO_MESH_FILE");
    static const bool no_emulation = VulkanSwitch("EDEN_NX2_NO_MESH_EMULATION") != nullptr;
    if (!mesh_skip) {
        mesh_skip = WorkPolicy::MeshSkip{
            .always = VulkanSwitch("EDEN_NX2_NO_MESH") != nullptr,
            .has_file = file != nullptr,
        };
    }
    const bool skip = mesh_skip->Skip([] { return std::filesystem::exists(file); });
    return skip || (emulated && no_emulation);
}

void RasterizerMetal::FlushWork() {
    // Vulkan's EDEN_NX2_FLUSH_DRAWS=<n> (read under EDEN_NX2_METAL_VULKAN_SWITCHES): the work
    // recorded is committed every n draws. Unset, Metal commits at fences, waits and frame ends.
    static const u32 flush_draws =
        WorkPolicy::ParseFlushDraws(VulkanSwitch("EDEN_NX2_FLUSH_DRAWS"));
    draw_flush.budget = flush_draws;
    if (draw_flush.Note()) {
        ++draw_flushes;
        scheduler.Flush(CommitReason::DrawBudget);
        return;
    }
    if (!staging_pool.WantsFlush()) {
        return;
    }
    ++staging_flushes;
    const u64 tick = scheduler.Flush(CommitReason::Staging);
    if (staging_pool.OverCap()) {
        ++staging_waits;
        scheduler.Wait(tick, CommitReason::Staging);
        staging_pool.Trim();
    }
}

namespace {

/// EDEN_NX2_METAL_DRAW_TRACE=1 (off by default): logs, once per pipeline, the draws into the
/// scene's targets that test depth EQUAL or LEQUAL without writing it (a colour pass over a depth
/// pre-pass), every task pipeline's draws past the first 24, and each draw of nothing with why.
bool DrawTraceEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DRAW_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Whether `key` is logged for the first time in `list` (64 keys at most).
bool DrawTraceFirst(int list, u64 key) {
    static std::mutex mutex;
    static std::array<std::vector<u64>, 2> keys;
    std::scoped_lock lock{mutex};
    std::vector<u64>& logged = keys[static_cast<size_t>(list)];
    if (logged.size() >= 64 || std::ranges::find(logged, key) != logged.end()) {
        return false;
    }
    logged.push_back(key);
    return true;
}

} // Anonymous namespace

void RasterizerMetal::NoteEmptyDraw(std::string_view why) {
    Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
    if (!DrawTraceEnabled() || maxwell3d == nullptr) {
        return;
    }
    const auto& hashes = pipeline_cache.LookupHashes();
    const u64 key = Common::CityHash64(reinterpret_cast<const char*>(hashes.data()),
                                       sizeof(hashes)) ^
                    Common::CityHash64(why.data(), why.size());
    if (!DrawTraceFirst(0, key)) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    LOG_WARNING(Render_Metal,
                "Metal: draw trace: empty draw ({}): {}; mesh {}, indexed {}, vertices {} from {}, "
                "indices {} from {}, base vertex {}, instances {} (state {}) from {}, depth test "
                "{} func {:#x} write {}, {}x{} target",
                why, DescribeDraw(), regs.IsMeshEnabled(), draw_state.draw_indexed,
                draw_state.vertex_buffer.count, draw_state.vertex_buffer.first,
                draw_state.index_buffer.count, draw_state.index_buffer.first,
                draw_state.base_index, trace_instances, draw_state.instance_count,
                draw_state.base_instance, regs.depth_test_enable,
                static_cast<u32>(regs.depth_test_func), regs.depth_write_enabled,
                regs.surface_clip.width, regs.surface_clip.height);
}

void RasterizerMetal::PollTaskTraces() {
    namespace SPIRV = Shader::Backend::SPIRV;
    std::erase_if(task_traces, [this](const TaskTrace& trace) {
        if (!scheduler.IsFree(trace.tick)) {
            return false;
        }
        // Best effort: the staging memory is read before this thread streams into it again.
        const u32* const args = trace.args;
        u64 asked = 0;
        u32 asking = 0;
        u32 most = 0;
        for (u32 task = 0; trace.payload_words >= 2 && task < trace.groups; ++task) {
            const u32 count = trace.payload[u64{task} * trace.payload_words + 1];
            asked += count;
            asking += count != 0 ? 1 : 0;
            most = (std::max)(most, count);
        }
        LOG_WARNING(Render_Metal,
                    "Metal: draw trace: task draw {:016x} (task {:016x}, mesh {:016x}, pixel "
                    "{:016x}, depth func {:#x} write {}, {}x{}): {} task groups, {} of them asked "
                    "for {} mesh groups (most {}); distribution: dispatch {} {} {}, requested {}, "
                    "draw {} vertices x {} from {} base {}, arena word {}, tasks {}",
                    trace.pipeline, trace.task, trace.mesh, trace.pixel, trace.depth_func,
                    trace.depth_write, trace.width, trace.height, trace.groups, asking, asked,
                    most, args[0], args[1], args[2], args[SPIRV::MESH_EMULATION_ARGS_REQUESTED],
                    args[SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET / 4],
                    args[SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET / 4 + 1],
                    args[SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET / 4 + 2],
                    args[SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET / 4 + 3],
                    args[SPIRV::MESH_EMULATION_ARGS_BASE], args[SPIRV::MESH_EMULATION_ARGS_TASKS]);
        return true;
    });
}

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    ++draws;
    skip_valid = false;
    trace_instances = instance_count;
    if (!task_traces.empty()) {
        PollTaskTraces();
    }
    // Cores waiting for a flush are served after each draw, as on Vulkan, not only when the
    // command list ends.
    SCOPE_EXIT {
        if (CachePathsEnabled()) {
            gpu.TickWork();
        }
    };
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    // Deferred inline writes reach the caches before anything of theirs is used.
    gpu_memory->FlushCaching();
    FlushWork();
    if (!BeginDraw(is_indexed, instance_count)) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteDraw(*maxwell3d, pipeline_cache.LookupHashes());
    }
    if (pipeline == nullptr) {
        RefuseDraw(pipeline_cache.Refusal());
        return;
    }
    if (!pipeline->BuildFunctions(device)) {
        RefuseDraw("program's MSL not compiled by Metal");
        return;
    }
    ++draws_with_pipeline;
    TrackOotTonemappedImage();
    // EDEN_NX2_METAL_DRAW=0: build pipelines but encode no draw (milestone 2 behaviour).
    static const bool encode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DRAW");
        return value == nullptr || value[0] != '0';
    }();
    if (!encode) {
        RefuseDraw("EDEN_NX2_METAL_DRAW=0");
        return;
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    index_rewrite = {};
    xfb_draw = {};
    if (grid_draw.wanted) {
        EnsureIndexKernels();
        // Only a plain draw takes its count on the GPU (a capture needs it on the CPU: its byte
        // offsets move by it).
        if (pipeline->IsMeshPipeline() || pipeline->Tess() != nullptr || gpu_condition.active ||
            indirect_kernels_failed || IndexConvert::NeedsConversion(DrawTopology(*maxwell3d)) ||
            ProvokingLastWanted(*pipeline) || RestartOffWidens(is_indexed) ||
            pipeline->Xfb() != nullptr || pipeline->Geo() != nullptr) {
            ReadGridWords();
        }
        grid_draw.instances = instance_count;
    }
    if ((pipeline->IsMeshPipeline() || pipeline->Tess() != nullptr) && !ConditionHoldsOnCpu()) {
        // Their kernels run whatever the condition: it is decided on the CPU.
        NoteEmptyDraw("mesh or tessellated draw under a condition false on the CPU");
        return;
    }
    if (pipeline->Geo() != nullptr) {
        DrawGeometry(*pipeline, is_indexed, instance_count);
        return;
    }
    if (pipeline->IsMeshPipeline()) {
        if (SkipMeshDraws(pipeline->Native() == nullptr)) {
            RefuseDraw("EDEN_NX2_NO_MESH");
            return;
        }
        // The draw's count is its mesh workgroups (the index count for indexed draws).
        const u32 groups = is_indexed ? draw_state.index_buffer.count
                                      : draw_state.vertex_buffer.count;
        if (groups == 0) {
            NoteEmptyDraw("mesh draw of 0 workgroups");
            return;
        }
        DrawMesh(*pipeline, is_indexed, MeshGroups{.count = groups});
        return;
    }
    if (pipeline->Tess() != nullptr) {
        // The vertex and control kernels before the render pass, then the patches.
        const auto format = draw_state.index_buffer.format;
        TessellationDraw tess_draw{
            .first = draw_state.vertex_buffer.first,
            .count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
            .first_instance = draw_state.base_instance,
            .instances = instance_count,
            .index_size = !is_indexed                                      ? 0U
                          : format == Maxwell::IndexFormat::UnsignedShort ? 2U
                                                                          : 4U,
            .widen_indices = is_indexed && format == Maxwell::IndexFormat::UnsignedByte,
            .first_index = draw_state.index_buffer.first,
            .index_buffer = nullptr,
            .index_offset = 0,
        };
        // The base vertex is added to the indices (EDEN_NX2_METAL_TESS_BASE_VERTEX=0: ignored,
        // as before).
        static const bool tess_base = EnvironmentOn("EDEN_NX2_METAL_TESS_BASE_VERTEX");
        if (is_indexed && tess_base) {
            tess_draw.base_vertex = static_cast<s32>(draw_state.base_index);
        } else if (is_indexed && draw_state.base_index != 0) {
            LOG_ONCE(warned_tess_base, "Metal: tessellated draws ignore a base vertex");
        }
        DrawTessellated(*pipeline, is_indexed, tess_draw);
        return;
    }
    const bool u8_indices =
        is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte;
    const u32 count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    const u32 index_size =
        is_indexed ? static_cast<u32>(draw_state.index_buffer.FormatSizeInBytes()) : 0U;
    const bool restart = maxwell3d->regs.primitive_restart.enabled != 0;
    const bool provoking_last = ProvokingLastWanted(*pipeline);
    CountDraw(count, instance_count);
    const bool convert = IndexConvert::NeedsConversion(DrawTopology(*maxwell3d)) || u8_indices ||
                         provoking_last || RestartOffWidens(is_indexed);
    const std::optional<u32> primitive =
        PrimitiveType(IndexConvert::ConvertedTopology(DrawTopology(*maxwell3d), provoking_last));
    if (!primitive) {
        RefuseDraw("topology Metal can't draw");
        return;
    }
    u32 converted =
        convert ? IndexConvert::ConvertedCount(DrawTopology(*maxwell3d), count, provoking_last) : 0U;
    if (convert && converted == 0) {
        NoteEmptyDraw("draw whose converted index count is 0");
        return;
    }
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
    };
    // Indices the GPU wrote are not in guest memory yet: a kernel rewrites them from the bound
    // index buffer before the render pass (EDEN_NX2_METAL_GPU_INDICES=1 for every indexed
    // rewrite, =0 never).
    static const char gpu_indices = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_GPU_INDICES");
        return value != nullptr ? value[0] : 'a';
    }();
    static const bool cache_converted_indices = EnvironmentOn("EDEN_NX2_METAL_INDEX_CACHE");
    if (convert && is_indexed && draw_state.inline_index_draw_indexes.empty()) {
        const GPUVAddr start = draw_state.index_buffer.StartAddress() +
                               GPUVAddr{draw_state.index_buffer.first} * index_size;
        const size_t src_bytes = size_t{count} * index_size;
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(start);
        const bool gpu_modified =
            device_addr && buffer_cache.IsRegionGpuModified(*device_addr, src_bytes);
        const bool cacheable = cache_converted_indices && device_addr.has_value() &&
                               src_bytes != 0 && gpu_memory->IsContinuousRange(start, src_bytes) &&
                               !gpu.HasPendingCPUWrites(*device_addr, src_bytes);
        if (gpu_indices != '0' && (gpu_indices == '1' || gpu_modified)) {
            index_rewrite.wanted = true;
            index_rewrite.only_ahead = false;
        } else if (!gpu_modified && (cacheable || (gpu_indices != '0' && count >= 256))) {
            index_rewrite.wanted = true;
            index_rewrite.only_ahead = true;
            index_rewrite.cacheable = cacheable;
            index_rewrite.device_addr = device_addr.value_or(0);
        }
        index_rewrite.params = {
            .mode = IndexConvert::RewriteMode(DrawTopology(*maxwell3d), provoking_last, restart),
            .index_size = index_size,
            .count = count,
            .restart = restart ? 1U : 0U,
            .converted = converted,
            .source_offset = draw_state.index_buffer.first * index_size,
        };
    }
    // Under a condition decided on the GPU, its kernel goes first and the draw through arguments
    // it can zero; else the CPU's answer.
    const bool under_condition = gpu_condition.active && PrepareConditionRegion();
    if (gpu_condition.active && !under_condition && !ConditionHoldsOnCpu()) {
        NoteEmptyDraw("draw under a condition false on the CPU");
        return;
    }
    PrepareXfbDraw(*pipeline, is_indexed, instance_count);
    SCOPE_EXIT {
        xfb_draw = {};
    };
    if (pipeline->Expand() != nullptr && (under_condition || !xfb_draw.wanted)) {
        RefuseDraw("expanded line or triangle draw that is conditional or too large");
        return;
    }
    if (!PrepareDraw(*pipeline, is_indexed)) {
        NoteUnprepared();
        return;
    }
    CountOcclusion();
    Coverage::Note(Coverage::Work::Draw, convert ? Coverage::Path::RewrittenIndices
                                                 : Coverage::Path::Native);
    if (xfb_draw.wanted && maxwell3d->regs.rasterize_enable == 0) {
        // Captured, and nothing to rasterize: the vertex program ran as the capture kernel.
        ++draws_encoded;
        return;
    }
    if (xfb_draw.wanted && xfb_draw.expand) {
        // Lines and triangles expanded by the generated vertex function from the kernel's outputs.
        const u32 vertices = RasterExpand::VerticesOf(xfb_draw.prim, xfb_draw.count);
        if (vertices != 0) {
            scheduler.Draw(*PrimitiveType(Maxwell::PrimitiveTopology::Triangles), 0, vertices,
                           instance_count, 0);
        }
        ++draws_encoded;
        ++expanded_draws;
        return;
    }
    // The draw through the condition's arguments; false to draw as usual (the condition holds).
    const auto conditional = [&](u32 draw_primitive, bool indexed, u32 draw_count,
                                 u32 draw_index_size, BufferHandle index_buffer, u64 index_offset,
                                 u32 first) {
        if (!under_condition) {
            return false;
        }
        if (DrawUnderCondition(draw_primitive, indexed, draw_count, draw_index_size,
                               index_buffer, index_offset, first, instance_count,
                               indexed && is_indexed ? static_cast<s32>(draw_state.base_index)
                                                     : 0,
                               draw_state.base_instance)) {
            return true;
        }
        // The region's command buffer was committed in between: decided on the CPU.
        return !ConditionHoldsOnCpu();
    };
    TrackOcarinaActor(pipeline->Key().unique_hashes[1], is_indexed);
    // After the draw is encoded: the fog's draw saves the scene for the effects mask.
    SCOPE_EXIT {
        NoteOcarinaEffectsDraw(pipeline->Key().unique_hashes[5]);
    };

    if (convert) {
        // 32-bit indices: from the converted-index cache, the GPU rewrite, or rewritten on the CPU
        // from the guest's indices (or the vertex range).
        BufferHandle draw_index_buffer = index_rewrite.cached_buffer;
        u64 draw_index_offset = 0;
        if (draw_index_buffer != nullptr) {
            converted = index_rewrite.cached_count;
        } else if (index_rewrite.output.buffer != nullptr) {
            ++indices_rewritten_on_gpu;
            draw_index_buffer = index_rewrite.output.buffer;
            draw_index_offset = index_rewrite.output.offset;
        } else {
            std::span<const u8> source;
            if (is_indexed) {
                if (!draw_state.inline_index_draw_indexes.empty()) {
                    source = draw_state.inline_index_draw_indexes;
                } else {
                    converted_source.resize(size_t{count} * index_size);
                    gpu_memory->ReadBlockUnsafe(
                        draw_state.index_buffer.StartAddress() +
                            GPUVAddr{draw_state.index_buffer.first} * index_size,
                        converted_source.data(), converted_source.size());
                    source = converted_source;
                }
            }
            ConvertedIndexEntry* promote_entry = nullptr;
            const u64 needed_bytes = u64{converted} * sizeof(u32);
            if (index_rewrite.wanted && index_rewrite.only_ahead && index_rewrite.cacheable &&
                needed_bytes != 0 && needed_bytes <= MaxSingleConvertedIndexBytes) {
                const BufferCacheRuntime::IndexState& idx = buffer_cache_runtime.Index();
                const auto found = converted_index_cache.find(idx.buffer);
                if (found != converted_index_cache.end()) {
                    const u64 src_offset = u64{idx.offset} + index_rewrite.params.source_offset;
                    const u64 src_size = u64{count} * index_size;
                    for (ConvertedIndexEntry& item : found->second) {
                        if (!item.buffer && item.hits >= 2 &&
                            item.device_addr == index_rewrite.device_addr &&
                            item.offset == src_offset && item.size == src_size &&
                            item.mode == index_rewrite.params.mode &&
                            item.index_size == index_size &&
                            item.restart == index_rewrite.params.restart &&
                            MakeRoomForConvertedIndexBytes(needed_bytes, &item)) {
                            promote_entry = &item;
                            break;
                        }
                    }
                }
            }
            u8* mapped = nullptr;
            Object promoted_buffer;
            if (promote_entry != nullptr) {
                promoted_buffer = CreateBuffer(device, needed_bytes, true, &mapped);
            }
            if (promoted_buffer && mapped != nullptr) {
                const std::span<u32> out(reinterpret_cast<u32*>(mapped), converted);
                converted = IndexConvert::Convert(DrawTopology(*maxwell3d), source, index_size,
                                                  is_indexed ? 0U : draw_state.vertex_buffer.first,
                                                  count, restart, out, provoking_last);
                if (converted != 0) {
                    promote_entry->converted = converted;
                    promote_entry->buffer_bytes = needed_bytes;
                    promote_entry->last_use = ++converted_index_clock;
                    promote_entry->last_frame = oot_frame;
                    promote_entry->buffer = std::move(promoted_buffer);
                    converted_index_bytes += needed_bytes;
                    draw_index_buffer = promote_entry->buffer.Get();
                }
            } else {
                const StagingBufferRef indices =
                    staging_pool.Stream(size_t{converted} * sizeof(u32), 4);
                const std::span<u32> out(reinterpret_cast<u32*>(indices.mapped_span.data()),
                                         converted);
                converted = IndexConvert::Convert(DrawTopology(*maxwell3d), source, index_size,
                                                  is_indexed ? 0U : draw_state.vertex_buffer.first,
                                                  count, restart, out, provoking_last);
                draw_index_buffer = indices.buffer;
                draw_index_offset = indices.offset;
            }
        }
        if (converted != 0 && draw_index_buffer != nullptr &&
            !conditional(*primitive, true, converted, 4, draw_index_buffer, draw_index_offset, 0)) {
            scheduler.DrawIndexed(*primitive, converted, 4, draw_index_buffer, draw_index_offset,
                                  instance_count,
                                  is_indexed ? static_cast<s32>(draw_state.base_index) : 0,
                                  draw_state.base_instance);
        }
        ++draws_converted;
    } else if (is_indexed) {
        const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
        if (!conditional(*primitive, true, draw_state.index_buffer.count, index_size, index.buffer,
                         index.offset, draw_state.index_buffer.first)) {
            scheduler.DrawIndexed(
                *primitive, draw_state.index_buffer.count, index_size, index.buffer,
                u64{index.offset} + u64{draw_state.index_buffer.first} * index_size,
                instance_count, static_cast<s32>(draw_state.base_index), draw_state.base_instance);
        }
    } else if (grid_draw.wanted) {
        scheduler.DrawIndirect(*primitive, grid_draw.arguments.buffer, grid_draw.arguments.offset);
        grid_draw = {};
        ++grid_draws_on_gpu;
    } else if (!conditional(*primitive, false, draw_state.vertex_buffer.count, 0, nullptr, 0,
                            draw_state.vertex_buffer.first)) {
        scheduler.Draw(*primitive, draw_state.vertex_buffer.first, draw_state.vertex_buffer.count,
                       instance_count, draw_state.base_instance);
    }
    ++draws_encoded;
}

std::vector<TextureHandle> RasterizerMetal::PrepareTexelBuffers(std::span<const Binding> bindings,
                                                                const Shader::Info& info) {
    if (info.texture_buffer_descriptors.empty() && info.image_buffer_descriptors.empty()) {
        return {};
    }
    // The buffer cache appends a stage's texture buffers, then its image buffers.
    const size_t read_only = Shader::NumDescriptors(info.texture_buffer_descriptors);
    std::vector<TextureHandle> result;
    for (const Binding& binding : bindings) {
        if (binding.kind != Binding::Kind::TexelBuffer) {
            continue;
        }
        const bool writable = result.size() >= read_only;
        const VideoCore::Surface::PixelFormat guest_format =
            binding.format == VideoCore::Surface::PixelFormat::Invalid
                ? VideoCore::Surface::PixelFormat::R8_UNORM
                : binding.format;
        const FormatInfo format = SurfaceFormat(device, guest_format, false);
        const u32 bytes_per_texel = VideoCore::Surface::BytesPerBlock(guest_format);
        const u32 elements = bytes_per_texel != 0 ? binding.size / bytes_per_texel : 0;
        if (format.format == 0 || elements == 0 || binding.buffer == nullptr) {
            result.push_back(nullptr);
            continue;
        }
        if (writable) {
            InvalidateConvertedIndices(binding.buffer, binding.offset, binding.size);
        }
        const u32 alignment = (std::max)(TextureBufferAlignment(device, format.format), 1U);
        BufferHandle buffer = binding.buffer;
        u64 offset = binding.offset;
        if (offset % alignment != 0) {
            // Through an aligned copy (ends an open render pass: counted as a texel copy). A
            // written one is copied back after the draw or dispatch (FinishTexelWritebacks).
            const StagingBufferRef copy = staging_pool.Stream(binding.size, alignment);
            scheduler.CopyBuffer(binding.buffer, binding.offset, copy.buffer, copy.offset,
                                 binding.size);
            if (writable) {
                texel_writebacks.push_back({copy.buffer, copy.offset, binding.buffer,
                                            binding.offset, binding.size});
            }
            buffer = copy.buffer;
            offset = copy.offset;
            ++texel_copies;
        }
        const TexelViewKey key{buffer, offset, elements, format.format, writable};
        Object& view = texel_view_cache[key];
        if (!view) {
            view = CreateTextureBufferView(buffer, offset, elements, bytes_per_texel, format.format,
                                           writable);
        }
        result.push_back(view.Get());
    }
    return result;
}

void RasterizerMetal::RefreshConvertedIndexAddrBounds() noexcept {
    converted_index_min_addr = ~DAddr{0};
    converted_index_max_addr = 0;
    for (const auto& [buf, entries] : converted_index_cache) {
        for (const ConvertedIndexEntry& entry : entries) {
            if (entry.size == 0) {
                continue;
            }
            converted_index_min_addr = (std::min)(converted_index_min_addr, entry.device_addr);
            const DAddr end = entry.size > ~DAddr{0} - entry.device_addr
                                  ? ~DAddr{0}
                                  : entry.device_addr + entry.size;
            converted_index_max_addr = (std::max)(converted_index_max_addr, end);
        }
    }
}

void RasterizerMetal::InvalidateConvertedIndices(BufferHandle buf, u64 offset, u64 size) {
    if (converted_index_count == 0) {
        return;
    }
    const auto it = converted_index_cache.find(buf);
    if (it == converted_index_cache.end()) {
        return;
    }
    if (size == ~u64{0}) {
        for (const ConvertedIndexEntry& entry : it->second) {
            converted_index_bytes -= entry.buffer_bytes;
        }
        converted_index_count -= it->second.size();
        converted_index_cache.erase(it);
        RefreshConvertedIndexAddrBounds();
        return;
    }
    const size_t removed = std::erase_if(it->second, [&](const ConvertedIndexEntry& entry) {
        const bool overlaps = entry.size != 0 && size != 0 &&
                              (entry.offset < offset ? offset - entry.offset < entry.size
                                                     : entry.offset - offset < size);
        if (overlaps) {
            converted_index_bytes -= entry.buffer_bytes;
        }
        return overlaps;
    });
    if (removed != 0) {
        converted_index_count -= removed;
        if (it->second.empty()) {
            converted_index_cache.erase(it);
        }
        RefreshConvertedIndexAddrBounds();
    }
}

void RasterizerMetal::InvalidateConvertedIndicesAddr(DAddr addr, u64 size) {
    if (converted_index_count == 0 || size == 0) {
        return;
    }
    const DAddr end = size > ~DAddr{0} - addr ? ~DAddr{0} : addr + size;
    if (end <= converted_index_min_addr || addr >= converted_index_max_addr) {
        return;
    }
    size_t total_removed = 0;
    std::erase_if(converted_index_cache, [&](auto& item) {
        const size_t removed = std::erase_if(item.second, [&](const ConvertedIndexEntry& entry) {
            const bool overlaps = entry.size != 0 &&
                                  (entry.device_addr < addr ? addr - entry.device_addr < entry.size
                                                            : entry.device_addr - addr < size);
            if (overlaps) {
                converted_index_bytes -= entry.buffer_bytes;
            }
            return overlaps;
        });
        total_removed += removed;
        return item.second.empty();
    });
    if (total_removed != 0) {
        converted_index_count -= total_removed;
        RefreshConvertedIndexAddrBounds();
    }
}

void RasterizerMetal::EvictConvertedIndexEntry() {
    while (converted_index_count >= MaxConvertedIndexEntries && !converted_index_cache.empty()) {
        auto victim_bucket = converted_index_cache.end();
        size_t victim_index = 0;
        u64 oldest_use = ~u64{0};
        for (auto it = converted_index_cache.begin(); it != converted_index_cache.end(); ++it) {
            for (size_t i = 0; i < it->second.size(); ++i) {
                if (it->second[i].last_use < oldest_use) {
                    oldest_use = it->second[i].last_use;
                    victim_bucket = it;
                    victim_index = i;
                }
            }
        }
        if (victim_bucket == converted_index_cache.end()) {
            break;
        }
        auto& vec = victim_bucket->second;
        converted_index_bytes -= vec[victim_index].buffer_bytes;
        vec.erase(vec.begin() + static_cast<std::ptrdiff_t>(victim_index));
        --converted_index_count;
        if (vec.empty()) {
            converted_index_cache.erase(victim_bucket);
        }
        RefreshConvertedIndexAddrBounds();
    }
}

bool RasterizerMetal::MakeRoomForConvertedIndexBytes(u64 needed_bytes,
                                                     const ConvertedIndexEntry* keep) noexcept {
    if (needed_bytes == 0 || needed_bytes > MaxSingleConvertedIndexBytes) {
        return false;
    }
    while (converted_index_bytes + needed_bytes > MaxConvertedIndexBytes) {
        ConvertedIndexEntry* victim = nullptr;
        u64 oldest_use = ~u64{0};
        for (auto& [buf, entries] : converted_index_cache) {
            for (ConvertedIndexEntry& entry : entries) {
                if (&entry != keep && entry.buffer && entry.last_use < oldest_use) {
                    oldest_use = entry.last_use;
                    victim = &entry;
                }
            }
        }
        if (victim == nullptr) {
            return false;
        }
        // Release the buffer in place so pointers into `entries` stay valid.
        converted_index_bytes -= victim->buffer_bytes;
        victim->buffer = {};
        victim->buffer_bytes = 0;
        victim->converted = 0;
        victim->hits = 0;
    }
    return true;
}

void RasterizerMetal::RewriteIndices(bool ahead) {
    const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
    EnsureIndexKernels();
    if (!index_kernel || index.buffer == nullptr) {
        // The CPU rewrites them from guest memory instead.
        return;
    }
    IndexConvert::RewriteParams params = index_rewrite.params;
    params.source_offset += index.offset;
    index_rewrite.output = staging_pool.Stream(size_t{params.converted} * sizeof(u32), 4);
    const bool strip = params.mode == IndexConvert::StripRestartMode;
    scheduler.BeginCompute(strip ? index_strip_kernel.Get() : index_kernel.Get(), ahead);
    scheduler.BindBuffer(ShaderStage::Compute, 0, index.buffer, 0);
    scheduler.BindBuffer(ShaderStage::Compute, 1, index_rewrite.output.buffer,
                         static_cast<u32>(index_rewrite.output.offset));
    scheduler.BindBytes(ShaderStage::Compute, 2, &params, static_cast<u32>(sizeof(params)));
    if (strip) {
        // One threadgroup finds where each part of the strip starts.
        scheduler.Dispatch({1, 1, 1}, {IndexConvert::StripRestartThreads, 1, 1});
        return;
    }
    constexpr u32 GroupSize = 64;
    scheduler.Dispatch({(params.converted + GroupSize - 1) / GroupSize, 1, 1}, {GroupSize, 1, 1});
}

namespace {

/// EDEN_NX2_METAL_MESH_FROM_GPU=0: mesh draws whose task count the GPU wrote are read back on
/// the CPU even when drawn natively (DrawMeshTasksFromGpu says no).
bool MeshFromGpuEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MESH_FROM_GPU");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

} // Anonymous namespace

bool RasterizerMetal::DrawMesh(GraphicsPipeline& pipeline, bool is_indexed,
                               const MeshGroups& groups) {
    const bool from_buffer = groups.buffer != nullptr || mesh_indirect.wanted;
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
        mesh_indirect = {};
    };
    for (;;) {
        NativeMesh* const native = pipeline.Native();
        if (native == nullptr && (from_buffer || pipeline.Mesh() == nullptr)) {
            if (!from_buffer) {
                RefuseDraw("mesh pipeline Metal refused, which the emulation can't take");
            }
            return false;
        }
        if (PrepareDraw(pipeline, is_indexed, from_buffer ? 0U : groups.count)) {
            break;
        }
        if (native != nullptr && pipeline.Native() == nullptr && pipeline.BuildFunctions(device)) {
            // Metal refused the mesh pipeline state just now: the emulation draws it.
            continue;
        }
        NoteUnprepared();
        return true;
    }
    CountOcclusion();
    if (const NativeMesh* const native = pipeline.Native()) {
        if (!from_buffer) {
            scheduler.DrawMeshThreadgroups(groups.count, native->object_threads,
                                           native->mesh_threads);
        } else {
            const BufferHandle buffer =
                mesh_indirect.wanted ? mesh_indirect.output.buffer : groups.buffer;
            const u64 offset = mesh_indirect.wanted ? mesh_indirect.output.offset : groups.offset;
            if (buffer == nullptr) {
                RefuseDraw("indirect mesh draw without its rewritten arguments");
                return true;
            }
            for (u32 draw = 0; draw < groups.draws; ++draw) {
                scheduler.DrawMeshThreadgroupsIndirect(buffer, offset + u64{draw} * groups.stride,
                                                       native->object_threads,
                                                       native->mesh_threads);
            }
        }
        ++draws_encoded;
        ++native_mesh_draws;
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Native);
        return true;
    }
    const MeshEmulation* const mesh = pipeline.Mesh();
    const auto topology = static_cast<Maxwell::PrimitiveTopology>(mesh->topology);
    if (mesh->has_task) {
        // The distribution pass wrote the vertex count.
        scheduler.DrawIndirect(*PrimitiveType(topology), mesh_buffers.args.buffer,
                               mesh_buffers.args.offset +
                                   Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET);
    } else {
        scheduler.Draw(*PrimitiveType(topology), 0, groups.count * mesh->vertices_per_group, 1, 0);
    }
    ++draws_encoded;
    ++mesh_draws;
    ++emulated_mesh_draws;
    mesh_groups += groups.count;
    Coverage::Note(Coverage::Work::Draw, Coverage::Path::MeshCompute);
    return true;
}

void RasterizerMetal::DrawTessellated(GraphicsPipeline& pipeline, bool is_indexed,
                                       TessellationDraw& tess_draw) {
    if (tess_draw.count == 0 || tess_draw.instances == 0) {
        NoteEmptyDraw("tessellated draw of 0 vertices or instances");
        return;
    }
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    SCOPE_EXIT {
        FinishTexelWritebacks();
    };
    if (!PrepareDraw(pipeline, is_indexed, 0, &tess_draw)) {
        NoteUnprepared();
        return;
    }
    CountOcclusion();
    if (const Tessellation* const tess = pipeline.Tess(); tess->domain) {
        // The lines or points the kernels evaluated, their count in the arguments the count
        // kernel wrote.
        const std::optional<u32> primitive = PrimitiveType(
            tess->domain->emulation.lines ? Maxwell::PrimitiveTopology::Lines
                                          : Maxwell::PrimitiveTopology::Points);
        scheduler.DrawIndirect(*primitive, tess_buffers.args.buffer, tess_buffers.args.offset);
    } else {
        scheduler.SetTessellationFactorBuffer(tess_buffers.factors.buffer,
                                              tess_buffers.factors.offset);
        scheduler.DrawPatches(tess_buffers.output_points, tess_buffers.patches);
    }
    ++draws_encoded;
    ++tessellated_draws;
    tessellated_patches += tess_buffers.patches;
    Coverage::Note(Coverage::Work::Draw, Coverage::Path::TessellationKernels);
}

void RasterizerMetal::DrawTessellatedIndirect(GraphicsPipeline& pipeline) {
    index_rewrite = {};
    indirect_plan = {};
    mesh_indirect = {};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const size_t stride = params.stride != 0 ? params.stride
                                             : (params.is_indexed ? 5 : 4) * sizeof(u32);
    // The kernels are sized from the arguments on the CPU, as the mesh emulation's: the records
    // (and their count) as guest memory has them once the caches wrote back what the GPU wrote
    // there, one tessellated draw each.
    const auto read_word = [this](GPUVAddr address) {
        if (const std::optional<DAddr> device_address = gpu_memory->GpuToCpuAddress(address)) {
            FlushRegion(*device_address, sizeof(u32), VideoCommon::CacheType::BufferCache);
        }
        return gpu_memory->Read<u32>(address);
    };
    size_t count = params.max_draw_counts;
    if (params.include_count) {
        count = (std::min)(count, size_t{read_word(params.count_start_address)});
        ++indirect_counts_read;
    }
    for (size_t draw = 0; draw < count; ++draw) {
        const GPUVAddr record = params.indirect_start_address + draw * stride;
        TessellationDraw tess_draw{};
        tess_draw.count = read_word(record);
        tess_draw.instances = read_word(record + 1 * sizeof(u32));
        if (params.is_indexed) {
            tess_draw.first_index = read_word(record + 2 * sizeof(u32));
            tess_draw.base_vertex = static_cast<s32>(read_word(record + 3 * sizeof(u32)));
            tess_draw.first_instance = read_word(record + 4 * sizeof(u32));
            // The indices of every record come from guest memory (the buffer cache binds the
            // range of the whole call), widened to 32 bits with the record's base vertex added.
            tess_draw.index_size = 4;
            tess_draw.widen_indices = true;
        } else {
            tess_draw.first = read_word(record + 2 * sizeof(u32));
            tess_draw.first_instance = read_word(record + 3 * sizeof(u32));
        }
        DrawTessellated(pipeline, params.is_indexed, tess_draw);
    }
}

void RasterizerMetal::DrawMeshIndirect(GraphicsPipeline& pipeline) {
    const auto& params = maxwell3d->draw_manager.indirect_state;
    if (params.is_byte_count) {
        RefuseDraw("transform feedback draw of a mesh pipeline");
        return;
    }
    const size_t stride = params.stride != 0 ? params.stride
                                             : (params.is_indexed ? 5 : 4) * sizeof(u32);
    if (pipeline.Native() != nullptr && stride % sizeof(u32) == 0 && params.max_draw_counts != 0) {
        // PrepareDraw finds the records (and the count) in the buffer cache and rewrites them.
        mesh_indirect = {
            .wanted = true,
            .has_count = params.include_count,
            .draws = static_cast<u32>(params.max_draw_counts),
            .stride_words = static_cast<u32>(stride / sizeof(u32)),
        };
        const bool drawn =
            DrawMesh(pipeline, params.is_indexed, MeshGroups{.draws = mesh_indirect.draws});
        if (drawn) {
            ++native_mesh_indirect;
            return;
        }
    }
    mesh_indirect = {};
    // The emulation needs the threadgroup counts on the CPU: the records (and their count) as
    // guest memory has them once the caches wrote back what the GPU wrote there.
    const auto read_word = [this](GPUVAddr address) {
        if (const std::optional<DAddr> device_address = gpu_memory->GpuToCpuAddress(address)) {
            FlushRegion(*device_address, sizeof(u32), VideoCommon::CacheType::BufferCache);
        }
        return gpu_memory->Read<u32>(address);
    };
    size_t count = params.max_draw_counts;
    if (params.include_count) {
        count = (std::min)(count, size_t{read_word(params.count_start_address)});
    }
    for (size_t draw = 0; draw < count; ++draw) {
        const u32 groups = read_word(params.indirect_start_address + draw * stride);
        if (groups == 0) {
            NoteEmptyDraw("indirect mesh record of 0 workgroups (read on the CPU)");
            continue;
        }
        DrawMesh(pipeline, params.is_indexed, MeshGroups{.count = groups});
    }
}

void RasterizerMetal::UnpackVertices(u32 slot, const VertexSource& source, bool ahead) {
    if (!vertex_unpack_kernel && !vertex_unpack_failed) {
        std::string error;
        vertex_unpack_library = CreateLibrary(device, VertexUnpackKernelSource(), error);
        const Object function = vertex_unpack_library
                                    ? CreateFunction(vertex_unpack_library.Get(), "unpack_1010102")
                                    : Object{};
        vertex_unpack_kernel =
            function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
        if (!vertex_unpack_kernel) {
            vertex_unpack_failed = true;
            LOG_ERROR(Render_Metal, "Metal: 10:10:10:2 vertex unpack kernel failed to build: {}",
                      error);
        }
    }
    const auto& vertex_states = buffer_cache_runtime.Vertices();
    if (!vertex_unpack_kernel || source.stream >= vertex_states.size()) {
        return;
    }
    const auto& stream = vertex_states[source.stream];
    if (stream.buffer == nullptr || u64{stream.size} < u64{source.shift} + 4) {
        return;
    }
    // Every element the bound range holds: an indexed draw may read any of them.
    const u32 count = stream.stride == 0 ? 1U : (stream.size - source.shift - 4) / stream.stride + 1;
    // The source is bound 4-byte aligned; the rest of its offset goes in the parameters.
    const VertexUnpackParams params{
        .count = count,
        .stride = stream.stride,
        .is_signed = source.unpack == 2 ? 1U : 0U,
        .offset = (stream.offset & 3U) + source.shift,
    };
    unpacked_vertices[slot] = staging_pool.Stream(size_t{count} * 8, 16);
    scheduler.BeginCompute(vertex_unpack_kernel.Get(), ahead);
    scheduler.BindBuffer(ShaderStage::Compute, 0, stream.buffer, stream.offset & ~3U);
    scheduler.BindBuffer(ShaderStage::Compute, 1, unpacked_vertices[slot].buffer,
                         static_cast<u32>(unpacked_vertices[slot].offset));
    scheduler.BindBytes(ShaderStage::Compute, 2, &params, static_cast<u32>(sizeof(params)));
    constexpr u32 GroupSize = 64;
    scheduler.Dispatch({(count + GroupSize - 1) / GroupSize, 1, 1}, {GroupSize, 1, 1});
}

void RasterizerMetal::GatherVertices(const VertexSources& sources, bool ahead) {
    if (!vertex_gather_kernel && !vertex_gather_failed) {
        std::string error;
        vertex_gather_library = CreateLibrary(device, VertexGatherKernelSource(), error);
        const Object function = vertex_gather_library
                                    ? CreateFunction(vertex_gather_library.Get(), "gather_vertices")
                                    : Object{};
        vertex_gather_kernel =
            function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
        if (!vertex_gather_kernel) {
            vertex_gather_failed = true;
            LOG_ERROR(Render_Metal, "Metal: vertex gather kernel failed to build: {}", error);
        }
    }
    if (!vertex_gather_kernel) {
        return;
    }
    const auto& vertex_states = buffer_cache_runtime.Vertices();
    const auto elements = [&](const GatheredAttribute& attribute) -> u32 {
        const auto& stream = vertex_states[attribute.stream];
        if (stream.buffer == nullptr || stream.stride == 0 ||
            u64{stream.size} < u64{attribute.offset} + attribute.bytes) {
            return 0;
        }
        return (stream.size - attribute.offset - attribute.bytes) / stream.stride + 1;
    };
    // As many vertices as the longest stream has: an indexed draw may read any of them.
    u32 count = 0;
    for (u32 index = 0; index < sources.gathered_count; ++index) {
        count = (std::max)(count, elements(sources.gathered[index]));
    }
    if (count == 0) {
        return;
    }
    gathered_vertices = staging_pool.Stream(size_t{count} * sources.gather_stride, 16);
    scheduler.BeginCompute(vertex_gather_kernel.Get(), ahead);
    for (u32 index = 0; index < sources.gathered_count; ++index) {
        const GatheredAttribute& attribute = sources.gathered[index];
        const auto& stream = vertex_states[attribute.stream];
        const u32 stream_elements = elements(attribute);
        if (stream_elements == 0) {
            continue;
        }
        const VertexGatherParams params{
            .count = stream_elements,
            .stride = stream.stride,
            .offset = (stream.offset & 3U) + attribute.offset,
            .bytes = attribute.bytes,
            .gathered_stride = sources.gather_stride,
            .gathered_offset = attribute.gathered_offset,
        };
        scheduler.BindBuffer(ShaderStage::Compute, 0, stream.buffer, stream.offset & ~3U);
        scheduler.BindBuffer(ShaderStage::Compute, 1, gathered_vertices.buffer,
                             static_cast<u32>(gathered_vertices.offset));
        scheduler.BindBytes(ShaderStage::Compute, 2, &params, static_cast<u32>(sizeof(params)));
        constexpr u32 GroupSize = 64;
        scheduler.Dispatch({(stream_elements + GroupSize - 1) / GroupSize, 1, 1},
                           {GroupSize, 1, 1});
    }
}

void RasterizerMetal::ConvertMeshIndirect(bool ahead) {
    if (!mesh_indirect_kernel && !mesh_indirect_failed) {
        std::string error;
        mesh_indirect_library = CreateLibrary(device, MeshIndirectKernelSource, error);
        const Object function =
            mesh_indirect_library
                ? CreateFunction(mesh_indirect_library.Get(), MeshIndirectKernelName)
                : Object{};
        mesh_indirect_kernel =
            function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
        if (!mesh_indirect_kernel) {
            mesh_indirect_failed = true;
            LOG_ERROR(Render_Metal, "Metal: indirect mesh argument kernel failed to build: {}",
                      error);
        }
    }
    if (!mesh_indirect_kernel || mesh_indirect.draws == 0) {
        return;
    }
    const MeshIndirectParams params{
        .draws = mesh_indirect.draws,
        .stride_words = mesh_indirect.stride_words,
        .has_count = mesh_indirect.has_count ? 1U : 0U,
    };
    if (mesh_indirect.records == nullptr ||
        (mesh_indirect.has_count && mesh_indirect.count == nullptr)) {
        return;
    }
    mesh_indirect.output = staging_pool.Stream(size_t{mesh_indirect.draws} * 3 * sizeof(u32), 16);
    scheduler.BeginCompute(mesh_indirect_kernel.Get(), ahead);
    scheduler.BindBuffer(ShaderStage::Compute, 0, mesh_indirect.records,
                         static_cast<u32>(mesh_indirect.records_offset));
    scheduler.BindBuffer(ShaderStage::Compute, 1,
                         mesh_indirect.has_count ? mesh_indirect.count : mesh_indirect.records,
                         static_cast<u32>(mesh_indirect.has_count ? mesh_indirect.count_offset
                                                                  : mesh_indirect.records_offset));
    scheduler.BindBuffer(ShaderStage::Compute, 2, mesh_indirect.output.buffer,
                         static_cast<u32>(mesh_indirect.output.offset));
    scheduler.BindBytes(ShaderStage::Compute, 3, &params, static_cast<u32>(sizeof(params)));
    scheduler.Dispatch({(params.draws + MeshIndirectGroupSize - 1) / MeshIndirectGroupSize, 1, 1},
                       {MeshIndirectGroupSize, 1, 1});
}

bool RasterizerMetal::DrawMeshTasksFromGpu(DAddr count_addr) {
    if (!MeshFromGpuEnabled() || !maxwell3d->regs.IsMeshEnabled()) {
        return false;
    }
    gpu_memory->FlushCaching();
    FlushWork();
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    // Only Metal's mesh functions take the count as it is; the emulation needs it on the CPU
    // (the caller reads it and draws again).
    if (pipeline == nullptr || pipeline->Native() == nullptr || !pipeline->BuildFunctions(device) ||
        pipeline->Native() == nullptr) {
        return false;
    }
    if (SkipMeshDraws(false)) {
        // Draw skips it once the count is read.
        return false;
    }
    if (!ConditionHoldsOnCpu()) {
        // As Draw: a mesh draw's condition is decided on the CPU.
        NoteEmptyDraw("mesh draw counted on the GPU under a condition false on the CPU");
        return true;
    }
    std::optional<u32> slot;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        slot = buffer_cache.Nx2MeshCountSlot(count_addr);
    }
    if (!slot) {
        return false;
    }
    SCOPE_EXIT {
        if (CachePathsEnabled()) {
            gpu.TickWork();
        }
    };
    TrackOotTonemappedImage();
    const MeshGroups groups{
        .buffer = buffer_cache_runtime.Nx2MeshMirror(),
        .offset = u64{*slot} * BufferCacheRuntime::Nx2MeshMirrorSlotSize,
    };
    if (!DrawMesh(*pipeline, false, groups)) {
        return false;
    }
    ++draws;
    ++draws_with_pipeline;
    ++native_mesh_from_gpu;
    if (VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteDraw(*maxwell3d, pipeline_cache.LookupHashes());
    }
    return true;
}

void RasterizerMetal::EnsureIndexKernels() {
    if (!index_kernel && !index_kernel_failed) {
        std::string error;
        index_kernel_library =
            CreateLibrary(device, IndexConvert::RewriteKernelSource(), error);
        const auto make = [&](const char* name) {
            const Object function = index_kernel_library
                                        ? CreateFunction(index_kernel_library.Get(), name)
                                        : Object{};
            return function ? CreateComputePipelineState(device, function.Get(), error)
                            : Object{};
        };
        index_kernel = make("rewrite_indices");
        index_strip_kernel = make("rewrite_restart_strip");
        if (!index_kernel || !index_strip_kernel) {
            index_kernel = {};
            index_kernel_failed = true;
            LOG_ERROR(Render_Metal, "Metal: index rewrite kernel failed to build: {}", error);
        }
    }
    if (!indirect_kernels_failed && !indirect_arguments_kernel) {
        std::string error;
        indirect_library = CreateLibrary(device, Indirect::KernelSource(), error);
        const auto make = [&](const char* name) {
            const Object function =
                indirect_library ? CreateFunction(indirect_library.Get(), name) : Object{};
            return function ? CreateComputePipelineState(device, function.Get(), error)
                            : Object{};
        };
        indirect_arguments_kernel = make("indirect_arguments");
        indirect_setup_kernel = make("indirect_rewrite_setup");
        grid_arguments_kernel = make("grid_arguments");
        if (!indirect_arguments_kernel || !indirect_setup_kernel || !grid_arguments_kernel) {
            indirect_arguments_kernel = {};
            indirect_kernels_failed = true;
            LOG_ERROR(Render_Metal, "Metal: indirect argument kernels failed to build: {}", error);
        }
    }
}

bool RasterizerMetal::PrepareIndirect(bool ahead) {
    IndirectPlan& plan = indirect_plan;
    const auto [indirect, indirect_offset] = buffer_cache.GetDrawIndirectBuffer();
    if (indirect == nullptr || indirect->Handle() == nullptr) {
        return false;
    }
    BufferHandle count_buffer = indirect->Handle();
    u32 count_offset = indirect_offset;
    if (plan.has_count) {
        const auto [count, offset] = buffer_cache.GetDrawIndirectCount();
        if (count == nullptr || count->Handle() == nullptr) {
            return false;
        }
        count_buffer = count->Handle();
        count_offset = offset;
    }
    const u32 words = plan.indexed ? Indirect::IndexedWords : Indirect::PlainWords;
    if (!plan.rewrite) {
        // Draws past the GPU's count draw no instances.
        plan.arguments = staging_pool.Stream(size_t{plan.max_draws} * words * sizeof(u32), 16);
        const Indirect::CopyParams params{
            .stride_words = plan.stride_words,
            .max_draws = plan.max_draws,
            .has_count = plan.has_count ? 1U : 0U,
            .words = words,
        };
        scheduler.BeginCompute(indirect_arguments_kernel.Get(), ahead);
        scheduler.BindBuffer(ShaderStage::Compute, 0, indirect->Handle(), indirect_offset);
        scheduler.BindBuffer(ShaderStage::Compute, 1, count_buffer, count_offset);
        scheduler.BindBuffer(ShaderStage::Compute, 2, plan.arguments.buffer,
                             static_cast<u32>(plan.arguments.offset));
        scheduler.BindBytes(ShaderStage::Compute, 3, &params, static_cast<u32>(sizeof(params)));
        constexpr u32 GroupSize = 64;
        scheduler.Dispatch({(plan.max_draws + GroupSize - 1) / GroupSize, 1, 1},
                           {GroupSize, 1, 1});
        return true;
    }
    // Each draw: a kernel reads its arguments and sizes the rewrite, the rewrite runs with the
    // threadgroups it wrote, and the draw takes the arguments it wrote.
    const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
    if (plan.indexed && index.buffer == nullptr) {
        return false;
    }
    plan.control = staging_pool.Stream(size_t{plan.max_draws} * Indirect::ControlWords * 4, 256);
    plan.indices = staging_pool.Request(size_t{plan.max_draws} * plan.capacity * sizeof(u32));
    const bool strip = plan.mode == IndexConvert::StripRestartMode;
    for (u32 draw = 0; draw < plan.max_draws; ++draw) {
        const u64 control = plan.control.offset + u64{draw} * Indirect::ControlWords * 4;
        const Indirect::SetupParams params{
            .draw = draw,
            .stride_words = plan.stride_words,
            .has_count = plan.has_count ? 1U : 0U,
            .indexed = plan.indexed ? 1U : 0U,
            .mode = plan.mode,
            .index_size = plan.indexed ? plan.index_size : 0U,
            .restart = plan.restart ? 1U : 0U,
            .capacity = plan.capacity,
            .source_offset = plan.indexed ? index.offset : 0U,
            .strip_threads = strip ? IndexConvert::StripRestartThreads : 0U,
        };
        scheduler.BeginCompute(indirect_setup_kernel.Get(), ahead);
        scheduler.BindBuffer(ShaderStage::Compute, 0, indirect->Handle(), indirect_offset);
        scheduler.BindBuffer(ShaderStage::Compute, 1, count_buffer, count_offset);
        scheduler.BindBytes(ShaderStage::Compute, 2, &params, static_cast<u32>(sizeof(params)));
        scheduler.BindBuffer(ShaderStage::Compute, 4, plan.control.buffer,
                             static_cast<u32>(control));
        scheduler.Dispatch({1, 1, 1}, {1, 1, 1});
        scheduler.BeginCompute(strip ? index_strip_kernel.Get() : index_kernel.Get(), ahead);
        scheduler.BindBuffer(ShaderStage::Compute, 0,
                             plan.indexed ? index.buffer : plan.control.buffer, 0);
        scheduler.BindBuffer(ShaderStage::Compute, 1, plan.indices.buffer,
                             static_cast<u32>(plan.indices.offset +
                                              u64{draw} * plan.capacity * sizeof(u32)));
        scheduler.BindBuffer(ShaderStage::Compute, 2, plan.control.buffer,
                             static_cast<u32>(control));
        scheduler.DispatchIndirect(plan.control.buffer,
                                   control + Indirect::ControlGroupsWord * sizeof(u32),
                                   {strip ? IndexConvert::StripRestartThreads : 64U, 1, 1});
    }
    return true;
}

void RasterizerMetal::BindMeshBuffers(const StageProgram& program, ShaderStage metal_stage,
                                      const MeshEmulation& mesh) {
    // The output buffer's binding, then (with a task program) payload, offsets and arguments.
    const std::array<const StagingBufferRef*, 4> buffers{
        &mesh_buffers.output, &mesh_buffers.payload, &mesh_buffers.offsets, &mesh_buffers.args};
    const size_t count = mesh.has_task ? buffers.size() : 1;
    for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
        if (slot.kind != ShaderTranslate::ResourceKind::Buffer ||
            slot.set == ShaderTranslate::PushConstantSet || slot.binding < mesh.output_binding ||
            slot.binding - mesh.output_binding >= count) {
            continue;
        }
        const StagingBufferRef& buffer = *buffers[slot.binding - mesh.output_binding];
        scheduler.BindBuffer(metal_stage, slot.index, buffer.buffer,
                             static_cast<u32>(buffer.offset));
    }
}

RasterizerMetal::MeshArena& RasterizerMetal::AcquireMeshArena() {
    const u64 tick = scheduler.CurrentTick();
    if (mesh_arena_current < mesh_arenas.size() && mesh_arenas[mesh_arena_current].tick == tick) {
        return mesh_arenas[mesh_arena_current];
    }
    // At most four command buffers in flight hold one: past that, wait for the oldest.
    constexpr size_t MaxMeshArenas = 4;
    const auto reusable = [this](const MeshArena& arena) {
        return arena.counted || scheduler.IsFree(arena.tick);
    };
    if (mesh_arenas.size() >= MaxMeshArenas && std::ranges::none_of(mesh_arenas, reusable)) {
        scheduler.Wait(std::ranges::min_element(mesh_arenas, {}, &MeshArena::tick)->tick);
    }
    // A finished command buffer's counter holds the words its draws asked for, including those
    // that did not fit: later arenas grow to hold them.
    constexpr u64 MaxArenaSize = u64{1} << 30;
    for (MeshArena& arena : mesh_arenas) {
        if (arena.counted || !scheduler.IsFree(arena.tick)) {
            continue;
        }
        arena.counted = true;
        u32 words;
        std::memcpy(&words, arena.counter_contents, sizeof(words));
        const u64 needed = u64{words} * sizeof(u32);
        {
            // As Vulkan's "Mesh emulation arenas" line: what the task draws of a command buffer
            // asked the distribution pass for. None at all means the task kernels launched no
            // mesh workgroups (grass, vine glow), whatever the draws' group counts.
            static u64 arenas_counted{};
            static u64 arenas_empty{};
            static u64 most_words{};
            ++arenas_counted;
            arenas_empty += words == 0 ? 1 : 0;
            most_words = (std::max)(most_words, u64{words});
            if (arenas_counted % 600 == 0) {
                LOG_DEBUG(Render_Metal,
                          "Metal: mesh emulation arenas: the last 600 command buffers with task "
                          "draws asked for at most {:.1f} MiB, {} of them for nothing ({} MiB "
                          "arenas)",
                          static_cast<double>(most_words * sizeof(u32)) / (1 << 20), arenas_empty,
                          mesh_arena_size >> 20);
                arenas_empty = 0;
                most_words = 0;
            }
        }
        if (needed > arena.size && mesh_arena_size < MaxArenaSize) {
            mesh_arena_size = (std::min)(std::bit_ceil(needed + needed / 4), MaxArenaSize);
            LOG_INFO(Render_Metal,
                     "Metal: mesh emulation output of a command buffer needed {} MiB, arenas grow "
                     "to {} MiB",
                     needed >> 20, mesh_arena_size >> 20);
        }
    }
    std::erase_if(mesh_arenas, [this](const MeshArena& arena) {
        return arena.counted && arena.size != mesh_arena_size;
    });
    auto it = std::ranges::find_if(mesh_arenas, &MeshArena::counted);
    if (it == mesh_arenas.end()) {
        MeshArena arena{.size = mesh_arena_size};
        arena.buffer = CreateBuffer(device, mesh_arena_size, false, nullptr);
        arena.counter = CreateBuffer(device, sizeof(u32), true, &arena.counter_contents);
        it = mesh_arenas.insert(mesh_arenas.end(), std::move(arena));
    }
    // The GPU is done with it, and this command buffer has not been committed.
    std::memset(it->counter_contents, 0, sizeof(u32));
    it->tick = tick;
    it->counted = false;
    mesh_arena_current = static_cast<size_t>(std::distance(mesh_arenas.begin(), it));
    return *it;
}

void RasterizerMetal::FinishTexelWritebacks() {
    for (const TexelWriteback& writeback : texel_writebacks) {
        scheduler.CopyBuffer(writeback.copy, writeback.copy_offset, writeback.buffer,
                             writeback.offset, writeback.size);
    }
    texel_writebacks.clear();
}

void RasterizerMetal::AddRescalingBits(const Shader::Info& info,
                                       std::span<const VideoCommon::ImageViewInOut> views,
                                       u32& texture_bit, u32& image_bit) {
    const auto rescaled = [&](size_t view, u32 count) {
        bool any = false;
        for (u32 element = 0; element < count && view + element < views.size(); ++element) {
            const ImageView& image_view = texture_cache.GetImageView(views[view + element].id);
            any = any || texture_cache.IsRescaling(image_view);
        }
        return any;
    };
    size_t view = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                  Shader::NumDescriptors(info.image_buffer_descriptors);
    for (const auto& desc : info.texture_descriptors) {
        if (rescaled(view, desc.count) && texture_bit < 32 * 4) {
            rescaling_words[texture_bit / 32] |= 1U << (texture_bit % 32);
        }
        view += desc.count;
        ++texture_bit;
    }
    for (const auto& desc : info.image_descriptors) {
        if (rescaled(view, desc.count) && image_bit < 32 * 2) {
            rescaling_words[4 + image_bit / 32] |= 1U << (image_bit % 32);
        }
        view += desc.count;
        ++image_bit;
    }
}

std::string RasterizerMetal::GraphicsMemoryReport() const {
    constexpr u64 MiB = u64{1} << 20;
    return fmt::format("textures {} MiB, buffers {} MiB, staging {} MiB; Metal device {} MiB "
                       "allocated, budget {} MiB",
                       texture_cache_runtime.allocated_bytes / MiB,
                       buffer_cache_runtime.GetDeviceMemoryUsage() / MiB,
                       staging_pool.HeldBytes() / MiB, DeviceAllocatedSize(device) / MiB,
                       DeviceMemoryBudget(device) / MiB);
}

TextureHandle RasterizerMetal::OrPlaceholder(const ImageView& view, TextureHandle handle,
                                             Shader::TextureType type) {
    if (handle != nullptr || !PipelineCheckEnabled() || type == Shader::TextureType::Buffer) {
        return handle;
    }
    // A texture or view Metal refused (or a null descriptor) reads zeros, as Vulkan's null
    // image, instead of a texture Metal leaves undefined.
    if (view.image_id != VideoCommon::NULL_IMAGE_ID) {
        Coverage::NoteGuard("refused texture read as zeros");
    }
    const size_t index = static_cast<size_t>(type);
    return index < placeholders.size() ? placeholders[index].Get() : nullptr;
}

void RasterizerMetal::MakePlaceholders() {
    if (placeholders_made || !PipelineCheckEnabled()) {
        return;
    }
    // Made before any render pass is open: filling them takes a blit encoder.
    placeholders_made = true;
    for (size_t index = 0; index < placeholders.size(); ++index) {
        const auto type = static_cast<Shader::TextureType>(index);
        if (type == Shader::TextureType::Buffer) {
            continue;
        }
        Object& placeholder = placeholders[index];
        const TextureDesc desc = PlaceholderTexture(static_cast<u32>(index));
        const u32 layers = desc.layers;
        placeholder = CreateTexture(device, desc);
        if (!placeholder_zeros) {
            u8* zeros = nullptr;
            placeholder_zeros = CreateBuffer(device, 256, true, &zeros);
            if (zeros != nullptr) {
                std::memset(zeros, 0, 256);
            }
        }
        for (u32 layer = 0; layer < layers && placeholder; ++layer) {
            scheduler.CopyBufferToTexture(placeholder_zeros.Get(), 0, 4, 4, placeholder.Get(), 0,
                                          layer, {}, {});
        }
    }
}

void RasterizerMetal::PrepareSampleGrids(const Shader::Info& info,
                                         std::span<const VideoCommon::ImageViewInOut> views) {
    // A multisampled image a program reads as a plain texture reads its samples as texels, as
    // the guest sees it: the grid is filled before the draw's render pass.
    size_t view = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                  Shader::NumDescriptors(info.image_buffer_descriptors);
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element, ++view) {
            if (view >= views.size()) {
                return;
            }
            ImageView& image_view = texture_cache.GetImageView(views[view].id);
            if (image_view.Samples() > 1 && !desc.is_multisample &&
                image_view.PrepareSampleGrid()) {
                Coverage::NoteGuard("multisampled image read as its samples");
            }
        }
    }
}

bool RasterizerMetal::NeedsReducingVariant(const StageProgram& program,
                                           std::span<const VideoCommon::ImageViewInOut> views,
                                           std::span<const VideoCommon::SamplerId> samplers) {
    if (program.reduction_slot == ~0U) {
        return false;
    }
    // Metal's sampler states reduce with linear min, mag and mip filters (CreateSampler; not with
    // EDEN_NX2_METAL_SAMPLERS=0, which makes plain ones): bit 2 of SamplerReductionBits, the
    // others only being set with linear min and mag filters.
    static const bool native_reduction = [this] {
        const char* const value = std::getenv("EDEN_NX2_METAL_SAMPLERS");
        return SamplerBiasSupported(device) && (value == nullptr || value[0] != '0');
    }();
    const Shader::Info& info = program.info;
    // As BindStageResources: texel buffers and image buffers first.
    size_t view = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                  Shader::NumDescriptors(info.image_buffer_descriptors);
    size_t sampler = 0;
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element, ++view, ++sampler) {
            if (sampler >= samplers.size()) {
                return false;
            }
            const u64 bits = texture_cache.GetSampler(samplers[sampler]).ReductionBits();
            if ((bits & 3) == 0) {
                continue;
            }
            // An integer or stencil view takes a nearest sampler variant (Sampler::HandleFor): no
            // reduction.
            if (native_reduction && (bits & 4) != 0 && view < views.size()) {
                const ImageView& image_view = texture_cache.GetImageView(views[view].id);
                if (!VideoCore::Surface::IsPixelFormatInteger(image_view.format) &&
                    image_view.HostClass() != FormatClass::Stencil) {
                    continue;
                }
            }
            return true;
        }
    }
    return false;
}

std::string RasterizerMetal::TakeReductionLine() {
    const ReductionCounts counts = std::exchange(reduction_counts, ReductionCounts{});
    const auto [queued, built, refused] = pipeline_cache.TakeReducingBuildCounts();
    if (counts.plain_draws + counts.reducing_draws + counts.skipped + counts.plain_dispatches +
            counts.reducing_dispatches + counts.plain_refused_dispatches + queued ==
        0) {
        return {};
    }
    return fmt::format(
        "draws plain {} reducing {} (vertex {}, pixel {}), skipped while a reducing function "
        "builds {}, stages drawn plain as Metal refused theirs {}; dispatches plain {} reducing {} "
        "(plain as refused {}); reducing functions queued {} built {} refused {}",
        counts.plain_draws, counts.reducing_draws, counts.reducing_vertex,
        counts.reducing_fragment, counts.skipped, counts.plain_refused, counts.plain_dispatches,
        counts.reducing_dispatches, counts.plain_refused_dispatches, queued, built, refused);
}

void RasterizerMetal::BindStageResources(StageProgram& program, ShaderStage metal_stage,
                                         std::span<const Binding> pending,
                                         std::span<const TextureHandle> texel_buffer_views,
                                         std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         u32 render_width, u32 render_height) {
    const Shader::Info& info = program.info;
    // EDEN_NX2_METAL_BIND_SCRATCH: the lists keep their storage between calls (a nested call
    // finds them taken and allocates its own).
    const bool scratch = BindScratchEnabled();
    ScratchList scratch_texture_slots{bind_scratch.texture_slots, scratch};
    ScratchList scratch_sampler_slots{bind_scratch.sampler_slots, scratch};
    ScratchList scratch_spill_table{bind_scratch.spill_table, scratch};
    ScratchList scratch_sampler_table{bind_scratch.sampler_table, scratch};
    ScratchList scratch_array_table{bind_scratch.array_table, scratch};
    std::vector<u32>& texture_slots = scratch_texture_slots.value;
    std::vector<u32>& sampler_slots = scratch_sampler_slots.value;
    std::vector<u64>& spill_table = scratch_spill_table.value;
    std::vector<u64>& sampler_table = scratch_sampler_table.value;
    /// Marks a sampler_slots entry as an entry of the spilled samplers' table.
    constexpr u32 SpilledSamplerFlag = 0x8000'0000U;
    /// Marks a texture_slots or sampler_slots entry as an entry of the array table.
    constexpr u32 ArrayEntryFlag = 0x4000'0000U;
    std::vector<u64>& array_table = scratch_array_table.value;
    array_table.assign(program.array_spill_slot != ~0U ? program.array_spill_entries : 0U, 0);
    // on_buffer=false: a texture cache view or placeholder, never a view of a buffer.
    const auto bind_texture = [&](u32 slot_value, TextureHandle texture, bool write,
                                  bool on_buffer) {
        if ((slot_value & ArrayEntryFlag) != 0) {
            const u32 entry = slot_value & ~ArrayEntryFlag;
            if (entry < array_table.size()) {
                array_table[entry] =
                    scheduler.UseTexture(metal_stage, texture, write, on_buffer);
            }
            return;
        }
        scheduler.BindTexture(metal_stage, slot_value, texture, on_buffer);
    };
    size_t next_buffer = 0;
    for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
        for (u32 element = 0; element < slot.count; ++element) {
            const u32 slot_index = slot.index + element;
            switch (slot.kind) {
            case ShaderTranslate::ResourceKind::Buffer:
                if (slot.spill != ~0U) {
                    // Read through the table of addresses bound after the loop.
                    while (next_buffer < pending.size() &&
                           pending[next_buffer].kind != Binding::Kind::Buffer) {
                        ++next_buffer;
                    }
                    if (spill_table.size() <= slot.spill) {
                        spill_table.resize(slot.spill + 1, 0);
                    }
                    if (next_buffer < pending.size()) {
                        spill_table[slot.spill] = scheduler.UseBuffer(
                            metal_stage, pending[next_buffer].buffer, pending[next_buffer].offset);
                        ++next_buffer;
                    }
                    break;
                }
                if (slot.set == ShaderTranslate::PushConstantSet) {
                    if (info.uses_render_area) {
                        std::array<u32, 16> words{};
                        std::copy(rescaling_words.begin(), rescaling_words.end(), words.begin());
                        words[12] = std::bit_cast<u32>(static_cast<f32>(render_width));
                        words[13] = std::bit_cast<u32>(static_cast<f32>(render_height));
                        scheduler.BindBytes(metal_stage, slot_index, words.data(),
                                            static_cast<u32>(sizeof(words)));
                    } else if (Shader::Backend::SPIRV::MeshFirstEnabled()) {
                        std::array<u32, 12> words{};
                        std::copy(rescaling_words.begin(), rescaling_words.end(), words.begin());
                        scheduler.BindBytes(metal_stage, slot_index, words.data(),
                                            static_cast<u32>(sizeof(words)));
                    } else {
                        scheduler.BindBytes(metal_stage, slot_index, rescaling_words.data(),
                                            static_cast<u32>(sizeof(rescaling_words)));
                    }
                    break;
                }
                while (next_buffer < pending.size() &&
                       pending[next_buffer].kind != Binding::Kind::Buffer) {
                    ++next_buffer;
                }
                if (next_buffer < pending.size()) {
                    scheduler.BindBuffer(metal_stage, slot_index, pending[next_buffer].buffer,
                                         pending[next_buffer].offset);
                    ++next_buffer;
                }
                break;
            case ShaderTranslate::ResourceKind::Texture:
                texture_slots.push_back(slot.in_array_table ? ArrayEntryFlag | (slot.spill + element)
                                                            : slot_index);
                break;
            case ShaderTranslate::ResourceKind::Sampler:
                // A spilled sampler is read from a table (an entry, flagged).
                sampler_slots.push_back(slot.in_array_table ? ArrayEntryFlag | (slot.spill + element)
                                        : slot.spill != ~0U ? SpilledSamplerFlag | slot.spill
                                                            : slot_index);
                break;
            }
        }
    }

    if (program.spill_slot != ~0U && !spill_table.empty()) {
        scheduler.BindBytes(metal_stage, program.spill_slot, spill_table.data(),
                            static_cast<u32>(spill_table.size() * sizeof(u64)));
    }
    // Texture slots in binding order: texel buffers, image buffers, textures, images.
    const size_t texel_buffers = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                                 Shader::NumDescriptors(info.image_buffer_descriptors);
    for (size_t element = 0; element < texel_buffers && element < texture_slots.size(); ++element) {
        bind_texture(texture_slots[element],
                     element < texel_buffer_views.size() ? texel_buffer_views[element] : nullptr,
                     false, true);
    }
    size_t texture_slot = texel_buffers;
    size_t view = texel_buffers;
    size_t sampler = 0;
    u64 reduction_modes = 0;
    for (const auto& desc : info.texture_descriptors) {
        VideoCommon::ImageViewId prev_view_id{};
        VideoCommon::SamplerId prev_sampler_id{};
        bool have_prev_tex = false;
        bool have_prev_samp = false;
        u64 prev_tex_res = 0;
        u64 prev_samp_res = 0;
        for (u32 element = 0; element < desc.count; ++element) {
            const VideoCommon::ImageViewId view_id = views[view++].id;
            const bool tex_in_array =
                texture_slot < texture_slots.size() &&
                (texture_slots[texture_slot] & ArrayEntryFlag) != 0;
            const bool has_sampler_slot = sampler < sampler_slots.size();
            const bool samp_in_array =
                has_sampler_slot && (sampler_slots[sampler] & ArrayEntryFlag) != 0;
            const VideoCommon::SamplerId samp_id =
                sampler < samplers.size() ? samplers[sampler] : VideoCommon::SamplerId{};
            if (tex_in_array && have_prev_tex && view_id == prev_view_id &&
                (!has_sampler_slot ||
                 (samp_in_array && have_prev_samp && samp_id == prev_sampler_id))) {
                const u32 tex_entry = texture_slots[texture_slot] & ~ArrayEntryFlag;
                if (tex_entry < array_table.size()) {
                    array_table[tex_entry] = prev_tex_res;
                }
                if (samp_in_array) {
                    const u32 samp_entry = sampler_slots[sampler] & ~ArrayEntryFlag;
                    if (samp_entry < array_table.size()) {
                        array_table[samp_entry] = prev_samp_res;
                    }
                }
                ++texture_slot;
                ++sampler;
                continue;
            }
            ImageView& image_view = texture_cache.GetImageView(view_id);
            if (texture_slot < texture_slots.size()) {
                const TextureHandle tex_handle = OrPlaceholder(
                    image_view, image_view.SampledHandle(desc.type, desc.is_multisample),
                    desc.type);
                if (tex_in_array) {
                    const u32 entry = texture_slots[texture_slot] & ~ArrayEntryFlag;
                    prev_tex_res = scheduler.UseTexture(metal_stage, tex_handle, false, false);
                    prev_view_id = view_id;
                    have_prev_tex = true;
                    if (entry < array_table.size()) {
                        array_table[entry] = prev_tex_res;
                    }
                } else {
                    scheduler.BindTexture(metal_stage, texture_slots[texture_slot], tex_handle,
                                          false);
                }
            }
            ++texture_slot;
            if (has_sampler_slot) {
                void* const handle =
                    texture_cache.GetSampler(samp_id).HandleFor(image_view, desc.is_depth);
                if (samp_in_array) {
                    const u32 entry = sampler_slots[sampler] & ~ArrayEntryFlag;
                    prev_samp_res = Scheduler::SamplerResourceId(handle);
                    prev_sampler_id = samp_id;
                    have_prev_samp = true;
                    if (entry < array_table.size()) {
                        array_table[entry] = prev_samp_res;
                    }
                } else if ((sampler_slots[sampler] & SpilledSamplerFlag) != 0) {
                    const u32 entry = sampler_slots[sampler] & ~SpilledSamplerFlag;
                    if (sampler_table.size() <= entry) {
                        sampler_table.resize(entry + 1, 0);
                    }
                    sampler_table[entry] = Scheduler::SamplerResourceId(handle);
                } else {
                    scheduler.BindSampler(metal_stage, sampler_slots[sampler], handle);
                }
            }
            if (has_sampler_slot && sampler_slots[sampler] < 16) {
                reduction_modes |= texture_cache.GetSampler(samp_id).ReductionBits()
                                   << (sampler_slots[sampler] * 4);
            }
            ++sampler;
        }
    }
    if (program.reduction_slot != ~0U) {
        scheduler.BindBytes(metal_stage, program.reduction_slot, &reduction_modes,
                            static_cast<u32>(sizeof(reduction_modes)));
    }
    if (program.sampler_spill_slot != ~0U && !sampler_table.empty()) {
        scheduler.BindBytes(metal_stage, program.sampler_spill_slot, sampler_table.data(),
                            static_cast<u32>(sampler_table.size() * sizeof(u64)));
    }
    for (const auto& desc : info.image_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            const size_t view_index = view++;
            ImageView& image_view = texture_cache.GetImageView(views[view_index].id);
            if (desc.is_written) {
                texture_cache.MarkModification(image_view.image_id);
            }
            if (texture_slot < texture_slots.size()) {
                bind_texture(texture_slots[texture_slot],
                             view_index == byte_shadow_view
                                 ? byte_shadow_texture
                                 : OrPlaceholder(image_view,
                                                 image_view.StorageHandle(desc.type, desc.format),
                                                 desc.type),
                             desc.is_written, false);
            }
            ++texture_slot;
        }
    }
    if (!array_table.empty()) {
        const size_t bytes = array_table.size() * sizeof(u64);
        if (bytes <= 4096) {
            scheduler.BindBytes(metal_stage, program.array_spill_slot, array_table.data(),
                                static_cast<u32>(bytes));
        } else {
            const StagingBufferRef table =
                staging_pool.Stream(bytes, buffer_cache_runtime.GetUniformBufferAlignment());
            if (table.mapped_span.size() >= bytes) {
                std::memcpy(table.mapped_span.data(), array_table.data(), bytes);
                scheduler.BindBuffer(metal_stage, program.array_spill_slot, table.buffer,
                                     static_cast<u32>(table.offset));
            }
        }
    }
}

template <typename CbufAddrFn>
void RasterizerMetal::CollectStageDescriptors(
    const Shader::Info& info, bool compute, bool via_header_index, bool use_bind_memo,
    CbufAddrFn&& cbuf_addr_of, DescriptorPageCache& page_cache, std::vector<u32>& batch_handles,
    std::vector<VideoCommon::ImageViewInOut>& views, std::vector<u32>& sampler_handles,
    std::vector<VideoCommon::SamplerId>& samplers) {
    const auto read_handle_word = [&](GPUVAddr addr) -> u32 {
        constexpr u64 page_mask = 0xfffULL;
        const u64 page_offset = addr & page_mask;
        if (page_offset <= 0x1000ULL - sizeof(u32)) {
            const u64 page = addr >> 12;
            const size_t page_slot = static_cast<size_t>(page ^ (page >> 4)) & 3U;
            const u8* host = page_cache.pages[page_slot].host;
            if (page_cache.pages[page_slot].page != page) {
                host = gpu_memory->GetPointerForDescriptor(addr & ~page_mask);
                page_cache.pages[page_slot] = {host != nullptr ? page : ~0ULL, host};
            }
            if (host != nullptr) {
                u32 value;
                std::memcpy(&value, host + page_offset, sizeof(u32));
                return value;
            }
        }
        return gpu_memory->ReadHandleWord(addr);
    };
    struct HandleWordReader {
        decltype(read_handle_word)& read_word;
        u32 ReadHandleWord(GPUVAddr addr) const {
            return read_word(addr);
        }
    } const handle_reader{read_handle_word};
    const u64 storage_alignment = buffer_cache_runtime.GetStorageBufferAlignment();
    const GPUVAddr storage_mask = ~(GPUVAddr{(std::max)(storage_alignment, u64{1})} - 1);
    std::array<GPUVAddr, Shader::Info::MAX_CBUFS> cbuf_addrs{};
    std::array<GPUVAddr, Shader::Info::MAX_SSBOS> ssbo_addrs{};
    const size_t num_ssbos =
        (std::min)(info.storage_buffers_descriptors.size(), ssbo_addrs.size());
    bool handle_addrs_ready = false;
    const auto ensure_handle_addrs = [&] {
        if (handle_addrs_ready) {
            return;
        }
        handle_addrs_ready = true;
        for (size_t cbuf = 0; cbuf < cbuf_addrs.size(); ++cbuf) {
            cbuf_addrs[cbuf] = cbuf_addr_of(static_cast<u32>(cbuf), true);
        }
        const auto read_u64 = [&](GPUVAddr addr) -> u64 {
            return u64{read_handle_word(addr)} |
                   (u64{read_handle_word(addr + sizeof(u32))} << 32);
        };
        for (size_t binding = 0; binding < num_ssbos; ++binding) {
            const auto& storage = info.storage_buffers_descriptors[binding];
            const GPUVAddr cbuf =
                storage.cbuf_index < cbuf_addrs.size() ? cbuf_addrs[storage.cbuf_index] : GPUVAddr{};
            if (cbuf == 0) {
                continue;
            }
            GPUVAddr base = read_u64(cbuf + storage.cbuf_offset);
            if (base != 0 && storage.indirect_offset != ~0U) {
                base = read_u64(base + storage.indirect_offset);
            }
            ssbo_addrs[binding] = base & storage_mask;
        }
    };
    const auto read_handle = [&](const auto& desc, u32 element) {
        if (desc.handle_program != ~0U) {
            return Tegra::Texture::TexturePair(
                VideoCommon::EvaluateTextureHandle(
                    info, desc.handle_program, handle_reader,
                    [&](u32 cbuf) -> GPUVAddr { return cbuf_addr_of(cbuf, true); },
                    storage_alignment, element),
                via_header_index);
        }
        const GPUVAddr addr =
            cbuf_addr_of(desc.cbuf_index, false) + desc.cbuf_offset + (element << desc.size_shift);
        if constexpr (requires { desc.has_secondary; }) {
            if (desc.has_secondary) {
                const GPUVAddr second =
                    cbuf_addr_of(desc.secondary_cbuf_index, false) + desc.secondary_cbuf_offset;
                const u32 raw = (read_handle_word(addr) << desc.shift_left) |
                                (read_handle_word(second) << desc.secondary_shift_left);
                return Tegra::Texture::TexturePair(raw, via_header_index);
            }
        }
        return Tegra::Texture::TexturePair(read_handle_word(addr), via_header_index);
    };
    const auto add_image_views = [&](const auto& desc, bool blacklist) {
        if (desc.handle_program != ~0U && desc.count > 1) {
            ensure_handle_addrs();
            batch_handles.resize(desc.count);
            VideoCommon::EvaluateTextureHandles(
                info.handle_programs[desc.handle_program], desc.count,
                std::span<const GPUVAddr>{cbuf_addrs},
                std::span<const GPUVAddr>{ssbo_addrs.data(), num_ssbos}, handle_reader,
                std::span<u32>{batch_handles.data(), desc.count});
            for (u32 element = 0; element < desc.count; ++element) {
                const auto handle =
                    Tegra::Texture::TexturePair(batch_handles[element], via_header_index);
                views.push_back({.index = handle.first, .blacklist = blacklist});
            }
            return;
        }
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back({.index = read_handle(desc, element).first, .blacklist = blacklist});
        }
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_image_views(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_image_views(desc, false);
    }
    for (const auto& desc : info.texture_descriptors) {
        if (desc.handle_program != ~0U && desc.count > 1) {
            ensure_handle_addrs();
            batch_handles.resize(desc.count);
            VideoCommon::EvaluateTextureHandles(
                info.handle_programs[desc.handle_program], desc.count,
                std::span<const GPUVAddr>{cbuf_addrs},
                std::span<const GPUVAddr>{ssbo_addrs.data(), num_ssbos}, handle_reader,
                std::span<u32>{batch_handles.data(), desc.count});
            for (u32 element = 0; element < desc.count; ++element) {
                const auto handle =
                    Tegra::Texture::TexturePair(batch_handles[element], via_header_index);
                views.push_back({.index = handle.first});
                if (use_bind_memo) {
                    sampler_handles.push_back(handle.second);
                    samplers.emplace_back();
                } else {
                    samplers.push_back(texture_cache.GetSamplerId(handle.second, compute));
                }
            }
            continue;
        }
        for (u32 element = 0; element < desc.count; ++element) {
            const auto handle = read_handle(desc, element);
            views.push_back({.index = handle.first});
            if (use_bind_memo) {
                sampler_handles.push_back(handle.second);
                samplers.emplace_back();
            } else {
                samplers.push_back(texture_cache.GetSamplerId(handle.second, compute));
            }
        }
    }
    for (const auto& desc : info.image_descriptors) {
        add_image_views(desc, desc.is_written);
    }
}

void RasterizerMetal::ResolveStageBindings(bool compute, bool blacklist,
                                           std::span<const u32> binding_kinds,
                                           std::span<VideoCommon::ImageViewInOut> views,
                                           std::span<const u32> sampler_handles,
                                           std::span<VideoCommon::SamplerId> samplers) {
    if (!binding_kinds.empty() && binding_kinds.size() == views.size()) {
        BindMemo::Resolve(texture_cache, compute, true, binding_kinds, views, sampler_handles,
                          samplers, bind_memo_counts);
        return;
    }
    for (size_t index = 0; index < sampler_handles.size(); ++index) {
        samplers[index] = texture_cache.GetSamplerId(sampler_handles[index], compute);
    }
    texture_cache.FillImageViews(views, compute, blacklist);
}

void RasterizerMetal::AppendSampledDump(u32 stage, const Shader::Info& info,
                                        std::span<const VideoCommon::ImageViewInOut> views,
                                        std::span<const VideoCommon::SamplerId> samplers,
                                        std::vector<FrameDump::Sampled>& out) {
    size_t view = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                  Shader::NumDescriptors(info.image_buffer_descriptors);
    size_t sampler = 0;
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element, ++view, ++sampler) {
            if (view < views.size() && sampler < samplers.size()) {
                out.push_back(DumpSampled(stage, views[view].index, views[view].id,
                                          samplers[sampler]));
            }
        }
    }
}

bool RasterizerMetal::PrepareDraw(GraphicsPipeline& pipeline, bool is_indexed, u32 mesh_workgroups,
                                  TessellationDraw* tess_draw) {
    // Buffer cache stage indices and the translated programs: the vertex program (or a mesh
    // pipeline's task or mesh kernel, or a tessellation pipeline's vertex kernel), a second kernel
    // (the mesh program after a task program, the tessellation control program), the render
    // vertex function where it is not the first (tessellation evaluation), and the pixel program.
    // Kernels are dispatched before the render pass. A natively drawn task/mesh pipeline has its
    // object (or mesh) function first, the mesh function second, on the render encoder.
    NativeMesh* const native = pipeline.Native();
    MeshEmulation* const mesh = native != nullptr ? nullptr : pipeline.Mesh();
    Tessellation* const tess = pipeline.Tess();
    GeometryCompute* const geo = pipeline.Geo();
    const bool native_task = native != nullptr && !native->object.entry_point.empty();
    const bool has_task = (mesh != nullptr && mesh->has_task) || native_task;
    const u32 mesh_stage = native != nullptr ? native->mesh_stage
                           : mesh != nullptr ? mesh->mesh_stage
                                             : 0U;
    const size_t vertex_variant =
        tess_draw == nullptr || tess_draw->index_size == 0 ? 0 : tess_draw->index_size == 2 ? 1 : 2;
    const size_t geo_variant =
        xfb_draw.index_size == 0 ? 0 : xfb_draw.index_size == 2 ? 1 : 2;
    // A geometry program after tessellation: the tessellation's stages first, then the geometry
    // kernel (stage 3), then the pixel program.
    const bool tess_geo = tess != nullptr && geo != nullptr;
    const std::array<size_t, 5> stages{
        0,
        has_task ? size_t{mesh_stage}
        : tess != nullptr ? size_t{1}
        : geo != nullptr  ? size_t{3}
                          : 0,
        tess != nullptr ? size_t{2} : 0, 3, 4};
    const std::array<StageProgram*, 5> programs{
        native != nullptr ? (native_task ? &native->object : &native->mesh)
        : mesh != nullptr ? &mesh->first.program
        : tess != nullptr ? &tess->vertex[vertex_variant]
        : geo != nullptr  ? &geo->capture[geo_variant]
                          : &pipeline.Vertex(),
        native_task       ? &native->mesh
        : has_task        ? &mesh->mesh.program
        : tess != nullptr ? &tess->control.program
        : geo != nullptr  ? &geo->kernel.program
                          : nullptr,
        tess != nullptr ? (tess->domain ? &tess->domain->evaluate.program : &pipeline.Vertex())
                        : nullptr,
        tess_geo ? &geo->kernel.program : nullptr,
        native != nullptr ? &native->fragment : &pipeline.Fragment()};
    prepare_refusal = {};
    prepare_empty_why = nullptr;
    if (tess != nullptr && tess_draw == nullptr) {
        prepare_refusal = "tessellated draw without its vertex count";
        return false;
    }
    const auto absent = [&](size_t index) {
        return programs[index] == nullptr || programs[index]->entry_point.empty();
    };
    const auto& regs = maxwell3d->regs;
    const Maxwell::PrimitiveTopology topology =
        native != nullptr ? static_cast<Maxwell::PrimitiveTopology>(native->topology)
        : mesh != nullptr ? static_cast<Maxwell::PrimitiveTopology>(mesh->topology)
        : geo != nullptr ? (geo->plan.output_topology == 1   ? Maxwell::PrimitiveTopology::Points
                            : geo->plan.output_topology == 6 ? Maxwell::PrimitiveTopology::Lines
                                                             : Maxwell::PrimitiveTopology::Triangles)
                         : DrawTopology(*maxwell3d);
    const bool mesh_pipeline = native != nullptr || mesh != nullptr;

    std::array<u32, VideoCommon::NUM_STAGES> uniform_masks{};
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        const Shader::Info& info = programs[index]->info;
        uniform_masks[stages[index]] = info.constant_buffer_mask;
        std::ranges::copy_n(info.constant_buffer_used_sizes.begin(),
                            uniform_buffer_sizes[stages[index]].size(),
                            uniform_buffer_sizes[stages[index]].begin());
    }
    MakePlaceholders();
    texture_cache.SynchronizeDescriptors(false);
    buffer_cache.SetUniformBuffersState(uniform_masks, &uniform_buffer_sizes);

    // Storage buffers, and every texture and image handle, as OpenGL's ConfigureImpl reads them.
    // EDEN_NX2_METAL_DRAW_SCRATCH: in the storage the draw before used.
    ScratchList scratch_views{draw_scratch.views};
    ScratchList scratch_samplers{draw_scratch.samplers};
    ScratchList scratch_sampler_handles{draw_scratch.sampler_handles};
    ScratchList scratch_binding_kinds{draw_scratch.binding_kinds};
    ScratchList scratch_batch_handles{draw_scratch.batch_handles};
    std::vector<VideoCommon::ImageViewInOut>& views = scratch_views.value;
    std::vector<VideoCommon::SamplerId>& samplers = scratch_samplers.value;
    std::array<size_t, 5> first_view{};
    std::array<size_t, 5> first_sampler{};
    // EDEN_NX2_BINDMEMO: the sampler handles and binding kinds, to look the bindings up in the
    // texture cache's memo.
    const bool use_bind_memo = BindMemo::Enabled();
    std::vector<u32>& sampler_handles = scratch_sampler_handles.value;
    std::vector<u32>& batch_handles = scratch_batch_handles.value;
    const bool dynamic_stage_kinds = tess != nullptr || geo != nullptr;
    std::vector<u32>& dynamic_kinds = scratch_binding_kinds.value;
    const bool via_header_index = regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding;
    DescriptorPageCache page_cache{};
    for (size_t index = 0; index < stages.size(); ++index) {
        first_view[index] = views.size();
        first_sampler[index] = samplers.size();
        if (absent(index)) {
            continue;
        }
        const size_t stage = stages[index];
        const Shader::Info& info = programs[index]->info;
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        size_t ssbo_index = 0;
        // A storage buffer whose pointer the program reads through another pointer (a list
        // picked from a pointer array: Ocarina's foliage lists, read by the grass and vine glow
        // task programs) is bound at the pointer read from memory, as Vulkan and compute
        // dispatches bind it. Without, the pointer array itself was bound, the task programs
        // read no work and launched no mesh workgroups (EDEN_NX2_METAL_SSBO_INDIRECT=0: as
        // before).
        static const bool indirect_ssbos = EnvironmentOn("EDEN_NX2_METAL_SSBO_INDIRECT");
        for (const auto& desc : info.storage_buffers_descriptors) {
            if (desc.indirect_offset != ~0U && DrawTraceEnabled() &&
                DrawTraceFirst(0, pipeline.Key().Hash() ^ (u64{0x55B0} << 48) ^ ssbo_index)) {
                LOG_WARNING(Render_Metal,
                            "Metal: draw trace: pipeline {:016x} stage {} storage buffer {} "
                            "through a pointer (cbuf {} +{:#x}, pointer at +{:#x}), bound {}",
                            pipeline.Key().Hash(), stage, ssbo_index, desc.cbuf_index,
                            desc.cbuf_offset, desc.indirect_offset,
                            indirect_ssbos ? "through it" : "at the pointer array (old)");
            }
            buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index++, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.is_written,
                                                   indirect_ssbos ? desc.indirect_offset : ~0U);
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        CollectStageDescriptors(
            info, false, via_header_index, use_bind_memo,
            [&](u32 cbuf, bool check_enabled) -> GPUVAddr {
                return cbuf < cbufs.size() && (!check_enabled || cbufs[cbuf].enabled)
                           ? cbufs[cbuf].address
                           : GPUVAddr{};
            },
            page_cache, batch_handles, views, sampler_handles, samplers);
        if (use_bind_memo && dynamic_stage_kinds) {
            if (programs[index]->binding_kinds.empty()) {
                BindMemo::AppendKinds(info, 0, programs[index]->binding_kinds);
            }
            dynamic_kinds.insert(dynamic_kinds.end(), programs[index]->binding_kinds.begin(),
                                 programs[index]->binding_kinds.end());
        }
    }
    const std::span<const u32> binding_kinds =
        !use_bind_memo ? std::span<const u32>{}
        : dynamic_stage_kinds ? std::span<const u32>{dynamic_kinds}
        : native != nullptr   ? pipeline.NativeBindingKinds()
        : mesh != nullptr     ? pipeline.MeshEmulationBindingKinds()
                              : pipeline.DefaultBindingKinds();
    ResolveStageBindings(false, true, binding_kinds, views, sampler_handles, samplers);
    for (size_t index = 0; index < stages.size(); ++index) {
        if (!absent(index)) {
            PrepareSampleGrids(programs[index]->info, std::span(views).subspan(first_view[index]));
        }
    }
    texture_cache.UpdateRenderTargets(false);
    TrackOcarinaCamera();
    NoteCameraMotionDraw();
    UpdateOcarinaJitter();
    if (TargetlessReachEnabled() && texture_cache.HasNoRenderTargets()) {
        texture_cache.BoundTargetlessFramebuffer(
            TargetlessReach(regs, texture_cache.IsRescaling()));
    }
    // A draw sampling the depth image it writes: the pass before it ends (BarrierFeedbackLoop),
    // as Vulkan's graphics pipeline asks at this point.
    texture_cache.CheckFeedbackLoop(views);
    const bool writes_storage = [&] {
        const auto any_written = [](const auto& descriptors) {
            return std::ranges::any_of(descriptors,
                                       [](const auto& desc) { return desc.is_written; });
        };
        for (size_t index = 0; index < stages.size(); ++index) {
            if (absent(index)) {
                continue;
            }
            const Shader::Info& info = programs[index]->info;
            if (info.stores_global_memory || any_written(info.storage_buffers_descriptors) ||
                any_written(info.image_descriptors) ||
                any_written(info.image_buffer_descriptors)) {
                return true;
            }
        }
        return false;
    }();
    if (fragment_barrier_pass != 0) {
        // A pending barrier: the encoder ends (storing its targets) when this draw samples one
        // of them or the pass wrote storage, and the draw's pass loads them again.
        const u64 serial = scheduler.RenderPassSerial();
        if (scheduler.IsRendering() && fragment_barrier_pass == serial &&
            (pass_stores == serial || texture_cache.SamplesRenderTarget(views))) {
            scheduler.EndRendering();
            ++barrier_pass_ends;
        }
        if (!scheduler.IsRendering() || fragment_barrier_pass != serial) {
            fragment_barrier_pass = 0;
        }
    }
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (framebuffer == nullptr) {
        prepare_refusal = "no framebuffer";
        return false;
    }
    RenderTargetSet targets{};
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        targets.color[rt] = framebuffer->Colors()[rt];
    }
    targets.depth = framebuffer->Depth();
    targets.has_depth = framebuffer->HasDepth();
    targets.has_stencil = framebuffer->HasStencil();
    targets.width = framebuffer->Width();
    targets.height = framebuffer->Height();
    targets.layers = framebuffer->Layers();
    // Before the buffer cache notes what this draw uses in the segment.
    scheduler.SplitBefore(targets);
    const u64 draw_segment = scheduler.SegmentSerial();
    // The stages' texture and image scaling indices count on through the pipeline, as their
    // bindings do.
    const bool is_rescaling = texture_cache.IsRescaling();
    const auto& resolution = Settings::values.resolution_info;
    rescaling_words = {};
    rescaling_words[6] = std::bit_cast<u32>(is_rescaling ? resolution.down_factor : 1.0f);
    {
        u32 texture_bit = 0;
        u32 image_bit = 0;
        for (size_t index = 0; index < stages.size(); ++index) {
            if (!absent(index)) {
                AddRescalingBits(programs[index]->info, std::span(views).subspan(first_view[index]),
                                 texture_bit, image_bit);
            }
        }
    }

    // Texel buffers come from the buffer cache.
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        const size_t stage = stages[index];
        const Shader::Info& info = programs[index]->info;
        buffer_cache.UnbindGraphicsTextureBuffers(stage);
        size_t view = first_view[index];
        size_t tbo_index = 0;
        const auto add = [&](const auto& desc, bool is_image) {
            for (u32 element = 0; element < desc.count; ++element) {
                ImageView& image_view = texture_cache.GetImageView(views[view++].id);
                bool is_written = false;
                if constexpr (requires { desc.is_written; }) {
                    is_written = desc.is_written;
                }
                buffer_cache.BindGraphicsTextureBuffer(stage, tbo_index++, image_view.GpuAddr(),
                                                       image_view.BufferSize(), image_view.format,
                                                       is_written, is_image);
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add(desc, false);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add(desc, true);
        }
    }
    if (mesh_pipeline) {
        auto& dirty_flags = maxwell3d->dirty.flags;
        const bool saved_vb_dirty = dirty_flags[VideoCommon::Dirty::VertexBuffers];
        dirty_flags[VideoCommon::Dirty::VertexBuffers] = false;
        buffer_cache.UpdateGraphicsBuffers(false);
        dirty_flags[VideoCommon::Dirty::VertexBuffers] =
            saved_vb_dirty || dirty_flags[VideoCommon::Dirty::VertexBuffers];
        if (mesh_indirect.wanted) {
            const auto& indirect_params = maxwell3d->draw_manager.indirect_state;
            const auto [records, records_offset] = buffer_cache.ObtainBuffer(
                indirect_params.indirect_start_address,
                static_cast<u32>(indirect_params.buffer_size),
                VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                VideoCommon::ObtainBufferOperation::DoNothing);
            mesh_indirect.records = records->Handle();
            mesh_indirect.records_offset = records_offset;
            if (mesh_indirect.has_count) {
                const auto [count, count_offset] = buffer_cache.ObtainBuffer(
                    indirect_params.count_start_address, sizeof(u32),
                    VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                    VideoCommon::ObtainBufferOperation::DoNothing);
                mesh_indirect.count = count->Handle();
                mesh_indirect.count_offset = count_offset;
            }
        }
    } else {
        buffer_cache.UpdateGraphicsBuffers(is_indexed);
        buffer_cache.BindHostGeometryBuffers(is_indexed);
    }

    // Each stage's buffer bindings, and texel buffer views (copies of misaligned ranges go
    // before the render pass).
    ScratchList scratch_stage_bindings{draw_scratch.stage_bindings};
    std::array<std::vector<Binding>, 5>& stage_bindings = scratch_stage_bindings.value;
    std::array<std::vector<TextureHandle>, 5> texel_views;
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        buffer_cache_runtime.PendingBindings().clear();
        buffer_cache.BindHostStageBuffers(stages[index]);
        stage_bindings[index] = buffer_cache_runtime.PendingBindings();
        const Shader::Info& info = programs[index]->info;
        if (!info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty()) {
            texel_views[index] = PrepareTexelBuffers(stage_bindings[index], info);
        }
    }
    // Until the state below is made.
    prepare_refusal = "render pipeline state refused by Metal";

    const bool checked = PipelineCheckEnabled();
    const auto& vertex_states = buffer_cache_runtime.Vertices();
    if (scheduler.IsRendering() && attachment_fit_memo.valid &&
        attachment_fit_memo.pass_serial == scheduler.RenderPassSerial() &&
        attachment_fit_memo.raw_samples == framebuffer->Samples() &&
        attachment_fit_memo.raw_depth_format == framebuffer->DepthFormat() &&
        attachment_fit_memo.raw_color_formats == framebuffer->ColorFormats() &&
        attachment_fit_memo.raw_targets == targets) {
        targets = attachment_fit_memo.fitted_targets;
    } else {
        const RenderTargetSet raw_targets = targets;
        std::array<u32, Maxwell::NumRenderTargets> fitted_colors{};
        for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
            fitted_colors[rt] = framebuffer->Colors()[rt].texture != nullptr
                                    ? framebuffer->ColorFormats()[rt]
                                    : 0U;
        }
        u32 fitted_depth = targets.has_depth ? framebuffer->DepthFormat() : 0U;
        u32 fitted_stencil = targets.has_stencil ? framebuffer->DepthFormat() : 0U;
        const u32 fitted_samples = framebuffer->Samples();
        if (checked) {
            auto fit_desc = Zeroed<RenderPipelineDesc>();
            fit_desc.color_formats = fitted_colors;
            fit_desc.depth_format = fitted_depth;
            fit_desc.stencil_format = fitted_stencil;
            fit_desc.samples = fitted_samples;
            FitAttachments(fit_desc, targets, DescribeTexture);
            fitted_colors = fit_desc.color_formats;
            fitted_depth = fit_desc.depth_format;
            fitted_stencil = fit_desc.stencil_format;
        }
        attachment_fit_memo = AttachmentFitMemo{
            .valid = true,
            .pass_serial = scheduler.IsRendering() ? scheduler.RenderPassSerial() : ~u64{0},
            .raw_targets = raw_targets,
            .raw_color_formats = framebuffer->ColorFormats(),
            .raw_depth_format = framebuffer->DepthFormat(),
            .raw_samples = framebuffer->Samples(),
            .fitted_targets = targets,
            .formats = MakeAttachmentFormatsKey(fitted_colors, fitted_depth, fitted_stencil,
                                                fitted_samples),
        };
    }

    // EDEN_NX2_METAL_REDUCTION_VARIANTS: the render vertex and pixel functions without the
    // sampler reduction rewrite, unless this draw's samplers need it.
    u32 reducing = 0;
    bool has_variants = false;
    const bool expanding = xfb_draw.wanted && xfb_draw.expand;
    if (ReductionVariantsEnabled() && pipeline.HasReductionVariants()) {
        for (size_t index = 0; index < programs.size(); ++index) {
            if (absent(index)) {
                continue;
            }
            StageProgram* const program = programs[index];
            const bool is_vertex =
                native == nullptr && program == &pipeline.Vertex() && !expanding;
            const bool is_fragment =
                program == (native != nullptr ? &native->fragment : &pipeline.Fragment());
            if ((!is_vertex && !is_fragment) || !program->reduction_variants) {
                continue;
            }
            has_variants = true;
            if (!NeedsReducingVariant(*program, std::span(views).subspan(first_view[index]),
                                      std::span(samplers).subspan(first_sampler[index]))) {
                continue;
            }
            bool variant_building = false;
            if (pipeline_cache.ReducingFunction(pipeline, *program, variant_building) != nullptr) {
                reducing |= is_vertex ? ReducingVertex : ReducingFragment;
            } else if (variant_building) {
                ++reduction_counts.skipped;
                prepare_refusal = "sampler reduction variant still building";
                return false;
            } else {
                ++reduction_counts.plain_refused;
            }
        }
    }

    static const bool use_state_memo = EnvironmentOn("EDEN_NX2_METAL_STATE_MEMO");
    const bool can_memo_state = use_state_memo && tess == nullptr && geo == nullptr &&
                                !xfb_draw.wanted && !DrawTraceEnabled();
    RenderStateMemoKey memo_key{};
    u8 color_written = 0;
    VertexSources vertex_sources{};
    RenderPipelineDesc layout_desc{};
    void* state = nullptr;
    bool memo_hit = false;
    if (can_memo_state) {
        FillRenderStateMemoRegisters(memo_key, color_written, regs, topology,
                                     attachment_fit_memo.formats, native != nullptr, reducing,
                                     checked);
        if (!mesh_pipeline) {
            const VertexProgramUse use = !absent(0) ? pipeline.GetVertexUse() : VertexProgramUse{};
            u32 attr_mask = checked ? use.loaded : ~0U;
            u32 stream_mask = 0;
            u32 attr_count = 0;
            while (attr_mask != 0) {
                const u32 attr = static_cast<u32>(std::countr_zero(attr_mask));
                attr_mask &= attr_mask - 1;
                const auto raw = regs.vertex_attrib_format[attr];
                memo_key.attrib_hex[attr_count++] = raw.hex;
                if (raw.constant == 0 && raw.buffer < Maxwell::NumVertexArrays) {
                    stream_mask |= 1U << raw.buffer;
                }
            }
            if (!checked) {
                stream_mask = ~0U;
            }
            u32 bound_mask = 0;
            u32 sm = stream_mask;
            u32 stream_count = 0;
            while (sm != 0) {
                const u32 s = static_cast<u32>(std::countr_zero(sm));
                sm &= sm - 1;
                if (vertex_states[s].buffer != nullptr) {
                    bound_mask |= 1U << s;
                    memo_key.stream_strides[stream_count] = vertex_states[s].stride;
                    memo_key.stream_divisors[stream_count] =
                        regs.vertex_stream_instances.IsInstancingEnabled(s)
                            ? regs.vertex_streams[s].frequency
                            : 0U;
                    ++stream_count;
                }
            }
            memo_key.used_attrib_count = static_cast<u8>(attr_count);
            memo_key.used_stream_count = static_cast<u8>(stream_count);
            memo_key.bound_streams_mask = bound_mask;
        }
        if (const auto* cached = pipeline.FindCachedRenderState(memo_key)) {
            state = cached->state;
            color_written = cached->color_written;
            vertex_sources = cached->vertex_sources;
            memo_hit = true;
        }
    }

    if (!memo_hit) {
        // The render pipeline state: targets, blending, vertex layout.
        auto state_desc = Zeroed<RenderPipelineDesc>();
        state_desc.color_formats = attachment_fit_memo.formats.color_formats;
        state_desc.depth_format = attachment_fit_memo.formats.depth_format;
        state_desc.stencil_format = attachment_fit_memo.formats.stencil_format;
        state_desc.samples = attachment_fit_memo.formats.samples;
        FillRegisterState(state_desc, regs, topology, mesh_pipeline);
        if (!mesh_pipeline) {
            for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
                FillStream(state_desc, regs, index, vertex_states[index].buffer != nullptr,
                           vertex_states[index].stride);
            }
            // The guest's layout as Metal takes it: Metal's vertex descriptor has 31 attributes
            // and 31 buffer layouts and aborts the process when asked for more (Ocarina of Time's
            // first draw has stream 31 bound), so only what the vertex function reads goes in,
            // numbered from 0.
            if (checked) {
                const VertexProgramUse use =
                    !absent(0)
                        ? (tess == nullptr && geo == nullptr ? pipeline.GetVertexUse()
                                                             : VertexUse(*programs[0]))
                        : VertexProgramUse{};
                std::string reason;
                if (!SanitizeVertexLayout(state_desc, use.loaded, use.resource_slots,
                                          vertex_sources, reason)) {
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                        LOG_WARNING(
                            Render_Metal,
                            "Metal: render pipeline {:016x} refused: {}; fragment {:016x}, {}",
                            pipeline.Key().unique_hashes[1], reason,
                            pipeline.Key().unique_hashes[5], Describe(state_desc));
                    }
                    return false;
                }
            }
        }
        // A tessellation pipeline's vertex kernel reads the vertex layout as stage-in data; its
        // post-tessellation vertex function reads buffers.
        layout_desc = state_desc;
        if (DrawTraceEnabled() && regs.depth_test_enable != 0 && regs.depth_write_enabled == 0 &&
            targets.has_depth && targets.width >= 1280 && targets.width > targets.height) {
            // A colour pass over a depth pre-pass into the scene's targets (EQUAL or LEQUAL, D3D
            // or GL numbering), once per pipeline.
            const u32 func = static_cast<u32>(regs.depth_test_func);
            if ((func == 3 || func == 4 || func == 0x202 || func == 0x203) &&
                DrawTraceFirst(1, pipeline.Key().Hash())) {
                const auto& draw_state = maxwell3d->draw_manager.draw_state;
                const auto& hashes = pipeline.Key().unique_hashes;
                std::string masks;
                for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
                    masks += fmt::format("{}{:04x}", rt == 0 ? "" : " ", regs.color_mask[rt].raw);
                }
                const auto& transform = regs.viewport_transform[0];
                LOG_WARNING(
                    Render_Metal,
                    "Metal: draw trace: scene draw {:016x} (vertex {:016x}, control {:016x}, eval "
                    "{:016x}, geometry/mesh {:016x}, pixel {:016x}; {}{}): topology {:#x}, indexed {}, "
                    "vertices {} from {}, indices {} from {}, base vertex {}, instances {} (state {}) "
                    "from {}, mesh groups {}, from GPU memory: indirect {} grid {} mesh {}; depth func "
                    "{:#x} write 0, bias {} {}/{}/{}, mode {:#x} clip {:#x}, viewport z {}+{}; cull {} "
                    "face {:#x} front {:#x}, rasterize {}, alpha test {} func {:#x}, masks (common "
                    "{}) {}, blend per target {} enable0 {}; {}x{}; {}",
                    pipeline.Key().Hash(), hashes[0] != 0 ? hashes[0] : hashes[1], hashes[2],
                    hashes[3], hashes[4], hashes[5],
                    native != nullptr ? "native mesh"
                    : mesh != nullptr ? "emulated mesh"
                    : tess != nullptr ? "tessellated"
                    : geo != nullptr  ? "geometry kernel"
                                      : "vertex",
                    has_task ? " with task" : "", static_cast<u32>(topology), is_indexed,
                    draw_state.vertex_buffer.count, draw_state.vertex_buffer.first,
                    draw_state.index_buffer.count, draw_state.index_buffer.first,
                    draw_state.base_index, trace_instances, draw_state.instance_count,
                    draw_state.base_instance, mesh_workgroups, indirect_plan.wanted,
                    grid_draw.wanted, mesh_indirect.wanted, func,
                    regs.polygon_offset_fill_enable, regs.depth_bias, regs.slope_scale_depth_bias,
                    regs.depth_bias_clamp, static_cast<u32>(regs.depth_mode),
                    static_cast<u32>(regs.viewport_clip_control.geometry_clip.Value()),
                    transform.translate_z, transform.scale_z, regs.gl_cull_test_enabled,
                    static_cast<u32>(regs.gl_cull_face), static_cast<u32>(regs.gl_front_face),
                    regs.rasterize_enable, regs.alpha_test_enabled,
                    static_cast<u32>(regs.alpha_test_func), regs.color_mask_common, masks,
                    regs.blend_per_target_enabled, regs.blend.enable[0], targets.width,
                    targets.height, Describe(layout_desc));
            }
        }
        if (geo != nullptr) {
            // The render pass draws from the reader's buffer: no vertex streams.
            std::ranges::fill(state_desc.attributes, VertexAttributeState{});
            std::ranges::fill(state_desc.streams, VertexStreamState{});
        }
        if (tess != nullptr && geo == nullptr) {
            ApplyTessellation(state_desc, *tess);
        }
        if (expanding) {
            // The expansion's vertex function reads the kernel's outputs: no vertex layout, drawn
            // as triangles.
            std::ranges::fill(state_desc.attributes, VertexAttributeState{});
            std::ranges::fill(state_desc.streams, VertexStreamState{});
            state_desc.topology_class = 2;
            state_desc.expand = 1;
        }
        bool state_building = false;
        bool used_fallback = false;
        state = pipeline_cache.RenderState(pipeline, state_desc, state_building, reducing);
        if (state == nullptr && !state_building && native == nullptr && checked) {
            // A state Metal refused is drawn without blending and alpha to coverage, what a target
            // format most often refuses, rather than leaving a hole.
            RenderPipelineDesc fallback = state_desc;
            for (BlendTargetState& blend : fallback.blend) {
                blend.enable = 0;
            }
            fallback.alpha_to_coverage = 0;
            state = pipeline_cache.RenderState(pipeline, fallback, state_building, reducing);
            if (state != nullptr) {
                used_fallback = true;
                Coverage::NoteGuard("refused pipeline drawn without blending");
            }
        }
        if (state == nullptr) {
            if (state_building) {
                prepare_refusal = "render pipeline state still building";
            } else if (native != nullptr && pipeline.Native() == nullptr) {
                prepare_refusal = "mesh render pipeline state refused by Metal";
            }
            return false;
        }
        color_written = 0;
        for (size_t index = 0; index < state_desc.blend.size(); ++index) {
            if (state_desc.blend[index].write_mask != 0) {
                color_written |= static_cast<u8>(1U << index);
            }
        }
        if (can_memo_state && !used_fallback) {
            pipeline.StoreCachedRenderState(memo_key, state, color_written, vertex_sources);
        }
    }

    // The buffers of the layout's streams, at their slots.
    const auto bind_vertex_streams = [&](const auto& bind) {
        if (!checked) {
            for (size_t index = 0; index < vertex_states.size(); ++index) {
                if (vertex_states[index].buffer != nullptr) {
                    bind(VertexBufferSlot(static_cast<u32>(index)), vertex_states[index].buffer,
                         vertex_states[index].offset);
                }
            }
            return;
        }
        for (u32 slot = 0; slot < vertex_sources.count; ++slot) {
            const VertexSource& source = vertex_sources.streams[slot];
            if (source.stream == VertexSource::Zeros) {
                if (!zero_vertices) {
                    // Metal hands out buffers zeroed.
                    zero_vertices = CreateBuffer(device, 64, false, nullptr);
                }
                bind(VertexBufferSlot(slot), zero_vertices.Get(), 0U);
            } else if (source.unpack != 0 || source.stream == VertexSource::Gathered) {
                const StagingBufferRef& unpacked = source.unpack != 0 ? unpacked_vertices[slot]
                                                                      : gathered_vertices;
                if (unpacked.buffer != nullptr) {
                    bind(VertexBufferSlot(slot), unpacked.buffer,
                         static_cast<u32>(unpacked.offset));
                } else {
                    if (!zero_vertices) {
                        zero_vertices = CreateBuffer(device, 64, false, nullptr);
                    }
                    bind(VertexBufferSlot(slot), zero_vertices.Get(), 0U);
                }
            } else {
                bind(VertexBufferSlot(slot), vertex_states[source.stream].buffer,
                     vertex_states[source.stream].offset + source.shift);
            }
        }
    };
    prepare_refusal = {};
    if (has_variants) {
        ++(reducing != 0 ? reduction_counts.reducing_draws : reduction_counts.plain_draws);
        reduction_counts.reducing_vertex += (reducing & ReducingVertex) != 0 ? 1 : 0;
        reduction_counts.reducing_fragment += (reducing & ReducingFragment) != 0 ? 1 : 0;
    }

    const DepthStencilDesc depth_stencil =
        DepthStencilState(regs, targets.has_depth, targets.has_stencil);
    const u64 depth_stencil_key = PackDepthStencilKey(depth_stencil);
    void* depth_stencil_handle = last_depth_stencil_state;
    if (depth_stencil_key != last_depth_stencil_key || depth_stencil_handle == nullptr) {
        Object& depth_stencil_state = depth_stencil_states[depth_stencil_key];
        if (!depth_stencil_state) {
            depth_stencil_state = CreateDepthStencilState(device, depth_stencil);
        }
        depth_stencil_handle = depth_stencil_state.Get();
        last_depth_stencil_key = depth_stencil_key;
        last_depth_stencil_state = depth_stencil_handle;
    }

    // A kernel goes ahead of the segment (leaving the render pass open) when it reads nothing
    // the segment writes: the pass is all the segment holds, and neither stored to what the
    // kernel reads nor draws into it.
    const bool pass_stored = scheduler.IsRendering() && pass_stores == scheduler.RenderPassSerial();
    const auto record_ahead = [&](bool reads_resources) {
        if (scheduler.SegmentSerial() != draw_segment || pass_stored) {
            return false;
        }
        if (reads_resources && (writes_storage || texture_cache.SamplesRenderTarget(views))) {
            return false;
        }
        if (scheduler.CanRecordAhead(targets)) {
            return true;
        }
        if (scheduler.CanUploadAhead() && scheduler.IsRenderingTo(targets)) {
            scheduler.WantSplitBefore(targets);
        }
        return false;
    };
    if (index_rewrite.wanted) {
        const BufferCacheRuntime::IndexState& idx = buffer_cache_runtime.Index();
        const IndexConvert::RewriteParams& params = index_rewrite.params;
        const u64 src_offset = u64{idx.offset} + params.source_offset;
        const u64 src_size = u64{params.count} * params.index_size;
        static const char gpu_indices = [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_GPU_INDICES");
            return value != nullptr ? value[0] : 'a';
        }();
        if (!index_rewrite.only_ahead) {
            InvalidateConvertedIndices(idx.buffer, src_offset, src_size);
            RewriteIndices(record_ahead(false));
        } else if (idx.buffer != nullptr) {
            ConvertedIndexEntry* matched = nullptr;
            if (index_rewrite.cacheable) {
                if (const auto found = converted_index_cache.find(idx.buffer);
                    found != converted_index_cache.end()) {
                    for (ConvertedIndexEntry& item : found->second) {
                        if (item.device_addr == index_rewrite.device_addr &&
                            item.offset == src_offset && item.size == src_size &&
                            item.mode == params.mode && item.index_size == params.index_size &&
                            item.restart == params.restart) {
                            matched = &item;
                            break;
                        }
                    }
                }
            }
            if (matched != nullptr && matched->buffer) {
                matched->last_use = ++converted_index_clock;
                matched->last_frame = oot_frame;
                index_rewrite.cached_buffer = matched->buffer.Get();
                index_rewrite.cached_count = matched->converted;
            } else if (matched != nullptr) {
                ++matched->hits;
                matched->last_use = ++converted_index_clock;
                matched->last_frame = oot_frame;
                if (gpu_indices != '0' && params.count >= 256 && record_ahead(false)) {
                    EnsureIndexKernels();
                    const u64 needed_bytes = u64{params.converted} * sizeof(u32);
                    Object promoted;
                    if (index_kernel && MakeRoomForConvertedIndexBytes(needed_bytes, matched)) {
                        promoted = CreateBuffer(device, needed_bytes, false, nullptr);
                    }
                    if (promoted) {
                        IndexConvert::RewriteParams gpu_params = params;
                        gpu_params.source_offset += idx.offset;
                        const bool strip = gpu_params.mode == IndexConvert::StripRestartMode;
                        scheduler.BeginCompute(
                            strip ? index_strip_kernel.Get() : index_kernel.Get(), true);
                        scheduler.BindBuffer(ShaderStage::Compute, 0, idx.buffer, 0);
                        scheduler.BindBuffer(ShaderStage::Compute, 1, promoted.Get(), 0);
                        scheduler.BindBytes(ShaderStage::Compute, 2, &gpu_params,
                                            static_cast<u32>(sizeof(gpu_params)));
                        if (strip) {
                            scheduler.Dispatch({1, 1, 1}, {IndexConvert::StripRestartThreads, 1, 1});
                        } else {
                            constexpr u32 GroupSize = 64;
                            scheduler.Dispatch(
                                {(gpu_params.converted + GroupSize - 1) / GroupSize, 1, 1},
                                {GroupSize, 1, 1});
                        }
                        matched->converted = params.converted;
                        matched->buffer_bytes = needed_bytes;
                        matched->buffer = std::move(promoted);
                        converted_index_bytes += needed_bytes;
                        index_rewrite.cached_buffer = matched->buffer.Get();
                        index_rewrite.cached_count = params.converted;
                        ++indices_rewritten_on_gpu;
                    } else {
                        RewriteIndices(true);
                    }
                }
            } else {
                const u64 needed_bytes = u64{params.converted} * sizeof(u32);
                if (index_rewrite.cacheable && needed_bytes != 0 &&
                    needed_bytes <= MaxSingleConvertedIndexBytes) {
                    if (converted_index_count >= MaxConvertedIndexEntries) {
                        EvictConvertedIndexEntry();
                    }
                    if (converted_index_count < MaxConvertedIndexEntries) {
                        auto& entries = converted_index_cache[idx.buffer];
                        entries.push_back(ConvertedIndexEntry{
                            .device_addr = index_rewrite.device_addr,
                            .offset = src_offset,
                            .size = src_size,
                            .buffer_bytes = 0,
                            .last_use = ++converted_index_clock,
                            .last_frame = oot_frame,
                            .mode = params.mode,
                            .index_size = params.index_size,
                            .restart = params.restart,
                            .converted = 0,
                            .hits = 1,
                        });
                        ++converted_index_count;
                        converted_index_min_addr =
                            (std::min)(converted_index_min_addr, index_rewrite.device_addr);
                        const DAddr end =
                            src_size > ~DAddr{0} - index_rewrite.device_addr
                                ? ~DAddr{0}
                                : index_rewrite.device_addr + src_size;
                        converted_index_max_addr = (std::max)(converted_index_max_addr, end);
                    }
                }
                if (gpu_indices != '0' && params.count >= 256 && record_ahead(false)) {
                    RewriteIndices(true);
                }
            }
        }
    }
    if (indirect_plan.wanted && !PrepareIndirect(record_ahead(false))) {
        prepare_refusal = "indirect draw arguments not bound";
        return false;
    }
    if (grid_draw.wanted && !PrepareGridDraw(record_ahead(false))) {
        prepare_refusal = "GPU-counted draw whose count is not in a buffer";
        return false;
    }
    if (mesh_indirect.wanted && native != nullptr) {
        ConvertMeshIndirect(record_ahead(false));
    }
    unpacked_vertices = {};
    for (u32 slot = 0; slot < vertex_sources.count; ++slot) {
        if (vertex_sources.streams[slot].unpack != 0) {
            UnpackVertices(slot, vertex_sources.streams[slot], record_ahead(false));
        }
    }
    gathered_vertices = {};
    if (vertex_sources.gathered_count != 0) {
        GatherVertices(vertex_sources, record_ahead(false));
    }
    if (mesh != nullptr) {
        void* const first = pipeline.KernelState(device, mesh->first);
        void* const mesh_kernel = has_task ? pipeline.KernelState(device, mesh->mesh) : nullptr;
        void* const distribution =
            has_task ? pipeline.KernelState(device, mesh->distribution) : nullptr;
        if (first == nullptr || mesh_workgroups == 0 ||
            (has_task && (mesh_kernel == nullptr || distribution == nullptr))) {
            if (mesh_workgroups != 0) {
                prepare_refusal = "mesh emulation kernel not built by Metal";
            }
            prepare_empty_why = "mesh draw of 0 workgroups or a kernel not built";
            return false;
        }
        namespace SPIRV = Shader::Backend::SPIRV;
        mesh_buffers = {};
        u32 capacity_words = 0;
        if (has_task) {
            // The mesh workgroups the tasks ask for are only known on the GPU: their output goes
            // to the command buffer's arena.
            MeshArena& arena = AcquireMeshArena();
            mesh_buffers.output = {.buffer = arena.buffer.Get()};
            mesh_buffers.counter = {.buffer = arena.counter.Get()};
            capacity_words = static_cast<u32>(arena.size / sizeof(u32));
            const auto stream = [&](u64 size) {
                return staging_pool.Stream((std::max)(size, u64{16}), 16);
            };
            mesh_buffers.payload = stream(u64{mesh_workgroups} * mesh->payload_stride * 4);
            mesh_buffers.offsets = stream(u64{mesh_workgroups} * 4);
            mesh_buffers.args = stream(SPIRV::MESH_EMULATION_ARGS_WORDS * 4);
            if (DrawTraceEnabled()) {
                // Each task pipeline's first 4 draws into a target, 256 at most.
                static std::vector<std::pair<u64, u32>> traced;
                static u32 total = 0;
                if (!task_traces.empty()) {
                    PollTaskTraces();
                }
                const u64 key = pipeline.Key().Hash() ^ (u64{targets.width} << 32) ^ targets.height;
                auto it = std::ranges::find(traced, key, &std::pair<u64, u32>::first);
                if (it == traced.end() && traced.size() < 256) {
                    it = traced.insert(traced.end(), {key, 0U});
                }
                if (it != traced.end() && it->second < 4 && total < 256 &&
                    !mesh_buffers.args.mapped_span.empty() &&
                    !mesh_buffers.payload.mapped_span.empty()) {
                    ++it->second;
                    ++total;
                    task_traces.push_back(TaskTrace{
                        .pipeline = pipeline.Key().Hash(),
                        .task = pipeline.Key().unique_hashes[1],
                        .mesh = pipeline.Key().unique_hashes[mesh->mesh_stage + 1],
                        .pixel = pipeline.Key().unique_hashes[5],
                        .tick = scheduler.CurrentTick(),
                        .args = reinterpret_cast<const u32*>(mesh_buffers.args.mapped_span.data()),
                        .payload =
                            reinterpret_cast<const u32*>(mesh_buffers.payload.mapped_span.data()),
                        .groups = mesh_workgroups,
                        .payload_words = mesh->payload_stride,
                        .depth_func = static_cast<u32>(regs.depth_test_func),
                        .depth_write = regs.depth_write_enabled,
                        .width = targets.width,
                        .height = targets.height,
                    });
                }
            }
        } else {
            mesh_buffers.output = staging_pool.Stream(
                (std::max)(u64{mesh_workgroups} * mesh->block_words * sizeof(u32), u64{16}), 16);
        }
        // After the arena: getting one may wait for a command buffer.
        // EDEN_NX2_METAL_MESH_TASK_IN_ORDER=1 (A/B, off: did not bring the grass back): the
        // kernels of a pipeline with a task program run in order, after the game's work so far,
        // as Vulkan's emulation runs them behind a full memory barrier.
        static const bool task_in_order = [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_MESH_TASK_IN_ORDER");
            return value != nullptr && value[0] == '1';
        }();
        const bool mesh_ahead = !(has_task && task_in_order) && record_ahead(true);
        if (has_task) {
            // The first draws of each task pipeline: what decides whether its pixels land.
            static std::atomic_bool logged_full{false};
            if (!logged_full.load(std::memory_order_relaxed)) {
                static std::mutex logged_mutex;
                static std::vector<u64> logged_pipelines;
                const u64 pipeline_hash = pipeline.Key().Hash();
                std::scoped_lock logged_lock{logged_mutex};
                const size_t limit = DrawTraceEnabled() ? 88U : 24U;
                if (logged_pipelines.size() >= limit) {
                    logged_full.store(true, std::memory_order_relaxed);
                } else if (std::ranges::find(logged_pipelines, pipeline_hash) ==
                           logged_pipelines.end()) {
                    logged_pipelines.push_back(pipeline_hash);
                    if (logged_pipelines.size() >= limit) {
                        logged_full.store(true, std::memory_order_relaxed);
                    }
                    LOG_WARNING(
                        Render_Metal,
                        "Metal: task mesh draw {:016x} (task {:016x}, mesh {:016x}, pixel "
                        "{:016x}): {} groups, kernels {}; depth test {} func {:#x} write {}, cull "
                        "{} face {:#x} front {:#x}, rasterize {}, alpha test {} func {:#x}, alpha "
                        "to coverage {}, blend0 {} mask0 {}{}{}{}, colour formats {:#x} {:#x} "
                        "{:#x} {:#x}, depth target {}, {}x{}",
                        pipeline_hash, pipeline.Key().unique_hashes[1],
                        pipeline.Key().unique_hashes[mesh->mesh_stage + 1],
                        pipeline.Key().unique_hashes[5], mesh_workgroups,
                        mesh_ahead ? "ahead" : "in order", regs.depth_test_enable,
                        static_cast<u32>(regs.depth_test_func), regs.depth_write_enabled,
                        regs.gl_cull_test_enabled, static_cast<u32>(regs.gl_cull_face),
                        static_cast<u32>(regs.gl_front_face), regs.rasterize_enable,
                        regs.alpha_test_enabled, static_cast<u32>(regs.alpha_test_func),
                        regs.anti_alias_alpha_control.alpha_to_coverage.Value(),
                        regs.blend.enable[0], regs.color_mask[0].R.Value(),
                        regs.color_mask[0].G.Value(), regs.color_mask[0].B.Value(),
                        regs.color_mask[0].A.Value(), static_cast<u32>(regs.rt[0].format),
                        static_cast<u32>(regs.rt[1].format), static_cast<u32>(regs.rt[2].format),
                        static_cast<u32>(regs.rt[3].format), targets.has_depth, targets.width,
                        targets.height);
                }
            }
        }
        const auto begin_kernel = [&](size_t index, MeshKernel& kernel, void* kernel_state) {
            scheduler.BeginCompute(kernel_state, mesh_ahead);
            BindStageResources(kernel.program, ShaderStage::Compute, stage_bindings[index],
                               texel_views[index], std::span(views).subspan(first_view[index]),
                               std::span(samplers).subspan(first_sampler[index]), targets.width,
                               targets.height);
            BindMeshBuffers(kernel.program, ShaderStage::Compute, *mesh);
        };
        begin_kernel(0, mesh->first, first);
        scheduler.Dispatch({mesh_workgroups, 1, 1}, mesh->first.workgroup_size);
        if (has_task) {
            // The distribution pass: payload, offsets, arguments and the arena counter at
            // bindings 0-3, and its constants.
            const SPIRV::MeshEmulationDistributionConstants constants{
                .tasks = mesh_workgroups,
                .payload_words = mesh->payload_stride,
                .capacity_words = capacity_words,
                .block_words = mesh->block_words,
                .vertices_per_group = mesh->vertices_per_group,
                // Vulkan's guaranteed maxComputeWorkGroupCount[0].
                .max_groups = 65535,
            };
            const std::array<const StagingBufferRef*, 4> distribution_buffers{
                &mesh_buffers.payload, &mesh_buffers.offsets, &mesh_buffers.args,
                &mesh_buffers.counter};
            scheduler.BeginCompute(distribution, mesh_ahead);
            for (const ShaderTranslate::ResourceSlot& slot : mesh->distribution.program.slots) {
                if (slot.kind != ShaderTranslate::ResourceKind::Buffer) {
                    continue;
                }
                if (slot.set == ShaderTranslate::PushConstantSet) {
                    scheduler.BindBytes(ShaderStage::Compute, slot.index, &constants,
                                        static_cast<u32>(sizeof(constants)));
                } else if (slot.binding < distribution_buffers.size()) {
                    const StagingBufferRef& buffer = *distribution_buffers[slot.binding];
                    scheduler.BindBuffer(ShaderStage::Compute, slot.index, buffer.buffer,
                                         static_cast<u32>(buffer.offset));
                }
            }
            scheduler.Dispatch({1, 1, 1}, {1, 1, 1});
            begin_kernel(1, mesh->mesh, mesh_kernel);
            scheduler.DispatchIndirect(mesh_buffers.args.buffer, mesh_buffers.args.offset,
                                       mesh->mesh.workgroup_size);
        }
    }

    std::array<u32, 4> domain_params_words{};
    if (tess != nullptr) {
        // 1. The vertex kernel: every vertex (or index) of every instance, in draw order.
        // 2. The control kernel: patches of input_points of them, several patches a workgroup.
        // The evaluation program then draws the patches from the control kernel's buffers.
        const u32 input_points = std::clamp(regs.patch_vertices, 1U, 32U);
        const u32 output_points = tess->modes.output_vertices;
        const u32 vertices = tess_draw->count * tess_draw->instances;
        const u32 patches = vertices / input_points;
        void* const vertex_kernel =
            pipeline.VertexKernelState(device, vertex_variant, layout_desc);
        // Without a control program the vertex outputs are the control points and the factors
        // come from the registers (and the same for every patch).
        const bool from_registers = tess->factors_from_registers;
        void* const control_kernel =
            from_registers ? nullptr : pipeline.KernelState(device, tess->control);
        void* const count_kernel =
            tess->domain ? pipeline.KernelState(device, tess->domain->count) : nullptr;
        void* const evaluate_kernel =
            tess->domain ? pipeline.KernelState(device, tess->domain->evaluate) : nullptr;
        if (vertex_kernel == nullptr || (!from_registers && control_kernel == nullptr) ||
            (tess->domain && (count_kernel == nullptr || evaluate_kernel == nullptr)) ||
            patches == 0) {
            if (patches != 0) {
                prepare_refusal = "tessellation kernel not built by Metal";
            }
            return false;
        }
        const auto stream = [&](u64 size) {
            return staging_pool.Stream((std::max)(size, u64{16}), 16);
        };
        if (tess_draw->index_size != 0) {
            if (tess_draw->widen_indices || tess_draw->base_vertex != 0) {
                // 8-bit indices (Metal reads 16- and 32-bit ones) widened from guest memory, and
                // the base vertex added: the stage-in fetch and VertexIndex both read the index.
                const auto& draw_state = maxwell3d->draw_manager.draw_state;
                const u32 source_size =
                    static_cast<u32>(draw_state.index_buffer.FormatSizeInBytes());
                converted_source.resize(size_t{tess_draw->count} * source_size);
                gpu_memory->ReadBlockUnsafe(draw_state.index_buffer.StartAddress() +
                                                u64{tess_draw->first_index} * source_size,
                                            converted_source.data(), converted_source.size());
                const StagingBufferRef widened = stream(u64{tess_draw->count} * sizeof(u32));
                const std::span<u32> out(reinterpret_cast<u32*>(widened.mapped_span.data()),
                                         tess_draw->count);
                const bool restart = regs.primitive_restart.enabled != 0;
                IndexConvert::Convert(Maxwell::PrimitiveTopology::Patches, converted_source,
                                      source_size, 0, tess_draw->count, restart, out);
                for (u32& index : out) {
                    if (!restart || index != 0xFFFFFFFFU) {
                        index = static_cast<u32>(static_cast<s32>(index) + tess_draw->base_vertex);
                    }
                }
                tess_draw->index_buffer = widened.buffer;
                tess_draw->index_offset = widened.offset;
            } else {
                const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
                tess_draw->index_buffer = index.buffer;
                tess_draw->index_offset =
                    u64{index.offset} + u64{tess_draw->first_index} * tess_draw->index_size;
            }
            if (tess_draw->index_buffer == nullptr) {
                prepare_refusal = "tessellated draw without an index buffer";
                return false;
            }
        }
        const u32 patches_per_group = (std::max)(32U / output_points, 1U);
        const u32 group_size = patches_per_group * output_points;
        const u32 groups = (patches + patches_per_group - 1) / patches_per_group;
        // The control kernel's threads past the last patch write past it too.
        const u32 point_slots = groups * group_size;
        const bool quads =
            tess->modes.primitive != ShaderTranslate::TessellationModes::Primitive::Triangles;
        const u32 factor_patches = from_registers ? patches : groups * patches_per_group;
        tess_buffers = {
            .vertices = stream(u64{vertices} * tess->vertex_stride),
            .control_points = from_registers
                                  ? StagingBufferRef{}
                                  : stream(u64{point_slots} * tess->control_point_stride),
            .patch_constants = stream(u64{groups * patches_per_group} * tess->patch_stride),
            // MTLQuadTessellationFactorsHalf (isolines too) or MTLTriangleTessellationFactorsHalf.
            .factors = stream(u64{factor_patches} * (quads ? 12U : 8U)),
            .patches = patches,
            .output_points = output_points,
        };
        if (from_registers) {
            tess_buffers.control_points = tess_buffers.vertices;
            const auto& lod = regs.tessellation.lod;
            // Edge factors 0-2 (0-3) and the inside ones, as the control kernel's factors are.
            const std::array<u16, 6> half_levels{
                FloatToHalf(lod.outer[0]), FloatToHalf(lod.outer[1]), FloatToHalf(lod.outer[2]),
                FloatToHalf(lod.outer[3]), FloatToHalf(lod.inner[0]), FloatToHalf(lod.inner[1])};
            std::array<u16, 6> one_patch{};
            if (quads) {
                one_patch = half_levels;
            } else {
                one_patch = {half_levels[0], half_levels[1], half_levels[2], half_levels[4], 0, 0};
            }
            const size_t patch_bytes = quads ? 12 : 8;
            u8* const out = tess_buffers.factors.mapped_span.data();
            for (u32 patch = 0; patch < patches; ++patch) {
                std::memcpy(out + patch * patch_bytes, one_patch.data(), patch_bytes);
            }
        }
        const auto bind = [&](ShaderStage metal_stage, u32 slot, const StagingBufferRef& buffer) {
            if (slot != ~0U) {
                scheduler.BindBuffer(metal_stage, slot, buffer.buffer,
                                     static_cast<u32>(buffer.offset));
            }
        };
        std::array<u32, 4>& domain_params = domain_params_words;
        if (tess->domain) {
            const ShaderTranslate::DomainEmulation& emulation = tess->domain->emulation;
            // As many evaluated vertices as the patches can have, within a bound of memory
            // (the count kernel truncates what does not fit).
            constexpr u64 MaxDomainBytes = u64{8} << 20;
            u64 capacity_limit = MaxDomainBytes / emulation.vertex_stride;
            if (tess_geo) {
                // The geometry program's scratch (records, output records, regions, lists) is
                // sized for the most primitives the evaluated vertices can make.
                const Geometry::Plan& plan = geo->plan;
                const u64 lists = 1 + std::popcount(geo->xfb_streams & ~1U);
                const u64 per_primitive =
                    4 * (u64{plan.input_vertices} * plan.InputStride() +
                         u64{plan.invocations} * plan.streams * plan.max_vertices *
                             plan.OutputStride() +
                         u64{plan.invocations} * plan.RegionStride() +
                         u64{plan.invocations} * plan.max_vertices * 3 * lists + plan.invocations);
                capacity_limit = (std::min)(
                    capacity_limit,
                    (std::max)(Geometry::MaxScratchBytes / per_primitive, u64{1}) *
                        plan.input_vertices);
            }
            const u32 capacity = static_cast<u32>((std::max)(
                (std::min)(u64{patches} * emulation.max_vertices_per_patch, capacity_limit),
                u64{2}));
            tess_buffers.captured = stream(u64{capacity} * emulation.vertex_stride);
            tess_buffers.offsets = stream((u64{patches} + 1) * sizeof(u32));
            tess_buffers.args = stream(4 * sizeof(u32));
            domain_params = {patches, capacity, 0, 0};
        }
        const bool tess_ahead = record_ahead(true);
        scheduler.BeginCompute(vertex_kernel, tess_ahead);
        BindStageResources(*programs[0], ShaderStage::Compute, stage_bindings[0], texel_views[0],
                           std::span(views).subspan(first_view[0]),
                           std::span(samplers).subspan(first_sampler[0]), targets.width,
                           targets.height);
        bind_vertex_streams([&](u32 slot, BufferHandle buffer, u32 offset) {
            scheduler.BindBuffer(ShaderStage::Compute, slot, buffer, offset);
        });
        const ShaderTranslate::TessellationSlots& vertex_slots = programs[0]->tessellation_slots;
        bind(ShaderStage::Compute, vertex_slots.output, tess_buffers.vertices);
        if (tess_draw->index_size != 0 && vertex_slots.index != ~0U) {
            // Read from the draw's first index (the stage-in fetch and VertexIndex both).
            scheduler.BindBuffer(ShaderStage::Compute, vertex_slots.index, tess_draw->index_buffer,
                                 static_cast<u32>(tess_draw->index_offset));
        }
        scheduler.DispatchStageIn(tess_draw->index_size != 0 ? 0U : tess_draw->first,
                                  tess_draw->count, tess_draw->first_instance,
                                  tess_draw->instances);

        if (!from_registers) {
            scheduler.BeginCompute(control_kernel, tess_ahead);
            BindStageResources(*programs[1], ShaderStage::Compute, stage_bindings[1], texel_views[1],
                               std::span(views).subspan(first_view[1]),
                               std::span(samplers).subspan(first_sampler[1]), targets.width,
                               targets.height);
            const ShaderTranslate::TessellationSlots& control_slots = programs[1]->tessellation_slots;
            bind(ShaderStage::Compute, control_slots.input, tess_buffers.vertices);
            bind(ShaderStage::Compute, control_slots.output, tess_buffers.control_points);
            bind(ShaderStage::Compute, control_slots.patch_output, tess_buffers.patch_constants);
            bind(ShaderStage::Compute, control_slots.tess_factors, tess_buffers.factors);
            if (control_slots.indirect_params != ~0U) {
                const std::array<u32, 2> params{input_points, patches};
                scheduler.BindBytes(ShaderStage::Compute, control_slots.indirect_params, params.data(),
                                    static_cast<u32>(sizeof(params)));
            }
            scheduler.Dispatch({groups, 1, 1}, {group_size, 1, 1});
        }
        if (tess->domain) {
            // The patches' domain points, evaluated into a buffer the draw reads as lines or
            // points: the count kernel sizes it, one threadgroup a patch evaluates.
            const ShaderTranslate::DomainEmulation& emulation = tess->domain->emulation;
            scheduler.BeginCompute(count_kernel, tess_ahead);
            bind(ShaderStage::Compute, emulation.factors_slot, tess_buffers.factors);
            scheduler.BindBytes(ShaderStage::Compute, emulation.params_slot, domain_params.data(),
                                static_cast<u32>(sizeof(domain_params)));
            bind(ShaderStage::Compute, emulation.offsets_slot, tess_buffers.offsets);
            bind(ShaderStage::Compute, emulation.args_slot, tess_buffers.args);
            scheduler.Dispatch({1, 1, 1}, {1, 1, 1});

            scheduler.BeginCompute(evaluate_kernel, tess_ahead);
            BindStageResources(*programs[2], ShaderStage::Compute, stage_bindings[2],
                               texel_views[2], std::span(views).subspan(first_view[2]),
                               std::span(samplers).subspan(first_sampler[2]), targets.width,
                               targets.height);
            const ShaderTranslate::TessellationSlots& evaluation_slots =
                programs[2]->tessellation_slots;
            bind(ShaderStage::Compute, evaluation_slots.input, tess_buffers.control_points);
            bind(ShaderStage::Compute, evaluation_slots.patch_input, tess_buffers.patch_constants);
            bind(ShaderStage::Compute, emulation.level_slot, tess_buffers.factors);
            bind(ShaderStage::Compute, emulation.factors_slot, tess_buffers.factors);
            scheduler.BindBytes(ShaderStage::Compute, emulation.params_slot, domain_params.data(),
                                static_cast<u32>(sizeof(domain_params)));
            bind(ShaderStage::Compute, emulation.offsets_slot, tess_buffers.offsets);
            bind(ShaderStage::Compute, emulation.output_slot, tess_buffers.captured);
            scheduler.Dispatch({patches, 1, 1}, {ShaderTranslate::DomainThreads, 1, 1});
        }
    }

    // The index buffer a capture or geometry kernel reads: the bound one, or a widened copy
    // (8-bit indices, a base vertex) read from guest memory.
    const auto resolve_capture_indices = [&](BufferHandle& index_buffer, u64& index_offset) {
        if (xfb_draw.widen_indices || xfb_draw.base_vertex != 0) {
            // 8-bit indices widened and the base vertex added, from guest memory.
            const auto& draw_state = maxwell3d->draw_manager.draw_state;
            const u32 source_size =
                static_cast<u32>(draw_state.index_buffer.FormatSizeInBytes());
            converted_source.resize(size_t{xfb_draw.count} * source_size);
            gpu_memory->ReadBlockUnsafe(draw_state.index_buffer.StartAddress() +
                                            u64{xfb_draw.first_index} * source_size,
                                        converted_source.data(), converted_source.size());
            const StagingBufferRef widened =
                staging_pool.Stream(u64{xfb_draw.count} * sizeof(u32), 16);
            const std::span<u32> out(reinterpret_cast<u32*>(widened.mapped_span.data()),
                                     xfb_draw.count);
            const bool restart = regs.primitive_restart.enabled != 0;
            IndexConvert::Convert(Maxwell::PrimitiveTopology::Patches, converted_source,
                                  source_size, 0, xfb_draw.count, restart, out);
            for (u32& index : out) {
                if (!restart || index != 0xFFFFFFFFU) {
                    index = static_cast<u32>(static_cast<s32>(index) + xfb_draw.base_vertex);
                }
            }
            index_buffer = widened.buffer;
            index_offset = widened.offset;
        } else {
            const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
            index_buffer = index.buffer;
            index_offset =
                u64{index.offset} + u64{xfb_draw.first_index} * xfb_draw.index_size;
        }
        return index_buffer != nullptr;
    };
    if (tess_geo) {
        // A geometry program after tessellation: the domain emulation's evaluated triangles,
        // lines or points are the input primitives, counted on the GPU.
        const Geometry::Plan& plan = geo->plan;
        void* const records_kernel = pipeline.KernelState(device, geo->tess_records);
        void* const geometry_kernel = pipeline.KernelState(device, geo->kernel);
        EnsureGeometryKernels();
        if (records_kernel == nullptr || geometry_kernel == nullptr ||
            geometry_feed_kernel.Get() == nullptr || geometry_count_kernel.Get() == nullptr) {
            prepare_refusal = "geometry kernel not built by Metal";
            return false;
        }
        const u32 capacity_vertices = domain_params_words[1];
        const u32 max_primitives = capacity_vertices / plan.input_vertices;
        if (max_primitives == 0) {
            prepare_refusal = {};
            prepare_empty_why = "tessellated geometry draw without room for a primitive";
            return false;
        }
        const u32 max_groups = max_primitives * plan.invocations;
        const u32 rounded = (max_primitives + Geometry::GroupSize - 1) / Geometry::GroupSize *
                            Geometry::GroupSize;
        const u64 list_words = u64{max_groups} * plan.max_vertices * 3;
        const auto stream = [&](u64 size, u64 alignment = 16) {
            return staging_pool.Stream((std::max)(size, u64{16}), alignment);
        };
        const StagingBufferRef records =
            stream(u64{rounded} * plan.input_vertices * plan.InputStride() * 4);
        geo_draw.output = stream(u64{rounded} * plan.invocations * plan.streams *
                                 plan.max_vertices * plan.OutputStride() * 4);
        const StagingBufferRef regions =
            stream(u64{rounded} * plan.invocations * plan.RegionStride() * 4);
        geo_draw.indices = stream(list_words * 4);
        const StagingBufferRef counts = stream(u64{max_groups} * 4);
        geo_draw.args = stream(32);
        const StagingBufferRef prepared = stream(Geometry::FeedBytes, 256);
        const auto bind = [&](u32 slot, const StagingBufferRef& buffer) {
            scheduler.BindBuffer(ShaderStage::Compute, slot, buffer.buffer,
                                 static_cast<u32>(buffer.offset));
        };
        const auto bind_words = [&](u32 slot, u32 word) {
            scheduler.BindBuffer(ShaderStage::Compute, slot, prepared.buffer,
                                 static_cast<u32>(prepared.offset) + word * 4);
        };
        const bool geo_ahead = record_ahead(true);
        // 1. The vertex count the domain kernels evaluated, as the sizes of what follows.
        const Geometry::GeometryFeed feed{
            .input_vertices = plan.input_vertices,
            .invocations = plan.invocations,
            .streams = plan.streams,
            .stride = plan.RegionStride(),
            .width = plan.IndexStride(),
            .topology = plan.output_topology,
            .capacity = static_cast<u32>(list_words),
            .max_primitives = max_primitives,
        };
        scheduler.BeginCompute(geometry_feed_kernel.Get(), geo_ahead);
        bind(0, tess_buffers.args);
        bind(1, prepared);
        scheduler.BindBytes(ShaderStage::Compute, 2, &feed, static_cast<u32>(sizeof(feed)));
        scheduler.Dispatch({1, 1, 1}, {1, 1, 1});
        // 2. The evaluated vertices as the geometry program's records.
        scheduler.BeginCompute(records_kernel, geo_ahead);
        bind(Geometry::RecordsSourceSlot, tess_buffers.captured);
        bind(Geometry::RecordsOutputSlot, records);
        bind_words(Geometry::RecordsCountSlot, Geometry::FeedRecordsWord);
        scheduler.DispatchIndirect(prepared.buffer, prepared.offset + Geometry::FeedRecordsDispatch * 4,
                                   {Geometry::GroupSize, 1, 1});
        // 3. The geometry program, a thread for each input primitive.
        scheduler.BeginCompute(geometry_kernel, geo_ahead);
        BindStageResources(geo->kernel.program, ShaderStage::Compute, stage_bindings[3],
                           texel_views[3], std::span(views).subspan(first_view[3]),
                           std::span(samplers).subspan(first_sampler[3]), targets.width,
                           targets.height);
        bind(geo->extra_slots[0], records);
        bind(geo->extra_slots[1], geo_draw.output);
        bind(geo->extra_slots[2], regions);
        scheduler.DispatchIndirect(prepared.buffer,
                                   prepared.offset + Geometry::FeedGeometryDispatch * 4,
                                   {Geometry::GroupSize, 1, 1});
        // 4. The strips of each stream, in order, as one list and the draw's arguments.
        const auto compact = [&](u32 stream_index, const StagingBufferRef& list,
                                 const StagingBufferRef& args) {
            const auto pass = [&](const Object& kernel, bool indirect) {
                scheduler.BeginCompute(kernel.Get(), geo_ahead);
                bind(0, regions);
                bind(1, list);
                bind(2, args);
                bind(3, counts);
                bind_words(4, Geometry::FeedStreamWord(stream_index));
                if (indirect) {
                    scheduler.DispatchIndirect(
                        prepared.buffer, prepared.offset + Geometry::FeedPackDispatch * 4,
                        {Geometry::GroupSize, 1, 1});
                } else {
                    scheduler.Dispatch({1, 1, 1}, {Geometry::GroupSize, 1, 1});
                }
            };
            pass(geometry_count_kernel, true);
            pass(geometry_scan_kernel, false);
            pass(geometry_write_kernel, true);
        };
        compact(0, geo_draw.indices, geo_draw.args);
        if (geo->xfb_stride != std::array<u32, Xfb::MaxBuffers>{}) {
            if ((geo->xfb_streams & 1U) != 0) {
                CaptureGeometryOutput(pipeline, *geo, geo_draw.output, geo_draw.indices,
                                      geo_draw.args, max_primitives, 0);
            }
            for (u32 stream_index = 1; stream_index < plan.streams; ++stream_index) {
                if (((geo->xfb_streams >> stream_index) & 1U) == 0) {
                    continue;
                }
                const StagingBufferRef list = stream(list_words * 4);
                const StagingBufferRef args = stream(32);
                compact(stream_index, list, args);
                CaptureGeometryOutput(pipeline, *geo, geo_draw.output, list, args, max_primitives,
                                      stream_index);
            }
        }
        ++geometry_draws;
        if (geometry_readbacks.size() >= 4096) {
            ResolveGeometryCounters();
        }
        geometry_readbacks.push_back({geo_draw.args, scheduler.CurrentTick(), plan.output_topology});
    } else if (geo != nullptr) {
        // A geometry program through metal_geometry.h: the vertex kernel and its records kernel,
        // the geometry kernel over the input primitives, and the compaction of its strips.
        const Geometry::Plan& plan = geo->plan;
        const auto [capture_kernel, records_kernel] =
            pipeline.GeometryKernelStates(device, geo_variant, layout_desc);
        void* const geometry_kernel = pipeline.KernelState(device, geo->kernel);
        EnsureGeometryKernels();
        if (!xfb_draw.wanted || capture_kernel == nullptr || records_kernel == nullptr ||
            geometry_kernel == nullptr || geometry_count_kernel.Get() == nullptr) {
            prepare_refusal = "geometry kernel not built by Metal";
            return false;
        }
        BufferHandle index_buffer{};
        u64 index_offset = 0;
        if (xfb_draw.index_size != 0 && !resolve_capture_indices(index_buffer, index_offset)) {
            prepare_refusal = "geometry draw without an index buffer";
            return false;
        }
        const u32 instances = xfb_draw.instances;
        const u32 vertices = xfb_draw.count * instances;
        const u32 per_instance = plan.Primitives(xfb_draw.count);
        const u32 input_primitives = per_instance * instances;
        const u32 groups = input_primitives * plan.invocations;
        const u32 rounded = (input_primitives + Geometry::GroupSize - 1) / Geometry::GroupSize *
                            Geometry::GroupSize;
        const u64 record_count = u64{rounded} * plan.input_vertices;
        const u64 record_words = record_count * plan.InputStride();
        // Every stream has its own vertex records and index region.
        const u64 output_words = u64{rounded} * plan.invocations * plan.streams *
                                 plan.max_vertices * plan.OutputStride();
        const u64 index_words = u64{rounded} * plan.invocations * plan.RegionStride();
        const u64 list_words = u64{groups} * plan.max_vertices * 3;
        // A list for the rasterized stream and one for each other stream with capture buffers.
        const u32 captured_streams = geo->xfb_streams & ~1U;
        const u64 lists = 1 + std::popcount(captured_streams);
        const u64 total = u64{vertices} * geo->vertex_stride + (record_words + output_words +
                                                                  index_words + list_words * lists +
                                                                  groups) * 4;
        if (input_primitives == 0) {
            prepare_refusal = {};
            prepare_empty_why = "geometry draw of 0 input primitives";
            return false;
        }
        if (total > Geometry::MaxScratchBytes) {
            prepare_refusal = "geometry draw past the scratch limit";
            return false;
        }
        const auto stream = [&](u64 size) {
            return staging_pool.Stream((std::max)(size, u64{16}), 16);
        };
        const StagingBufferRef captured = stream(u64{vertices} * geo->vertex_stride);
        const StagingBufferRef records = stream(record_words * 4);
        geo_draw.output = stream(output_words * 4);
        const StagingBufferRef regions = stream(index_words * 4);
        geo_draw.indices = stream(list_words * 4);
        const StagingBufferRef counts = stream(u64{groups} * 4);
        geo_draw.args = stream(32);
        const auto bind = [&](u32 slot, const StagingBufferRef& buffer) {
            scheduler.BindBuffer(ShaderStage::Compute, slot, buffer.buffer,
                                 static_cast<u32>(buffer.offset));
        };
        // 1. Every vertex of the draw through the vertex program.
        scheduler.BeginCompute(capture_kernel, false);
        BindStageResources(geo->capture[geo_variant], ShaderStage::Compute, stage_bindings[0],
                           texel_views[0], std::span(views).subspan(first_view[0]),
                           std::span(samplers).subspan(first_sampler[0]), targets.width,
                           targets.height);
        bind_vertex_streams([&](u32 slot, BufferHandle buffer, u32 offset) {
            scheduler.BindBuffer(ShaderStage::Compute, slot, buffer, offset);
        });
        const ShaderTranslate::TessellationSlots& slots =
            geo->capture[geo_variant].tessellation_slots;
        if (slots.output != ~0U) {
            bind(slots.output, captured);
        }
        if (xfb_draw.index_size != 0 && slots.index != ~0U) {
            scheduler.BindBuffer(ShaderStage::Compute, slots.index, index_buffer,
                                 static_cast<u32>(index_offset));
        }
        scheduler.DispatchStageIn(xfb_draw.index_size != 0 ? 0U : xfb_draw.first, xfb_draw.count,
                                  xfb_draw.first_instance, instances);
        // 2. The attributes the geometry program reads, as records.
        scheduler.BeginCompute(records_kernel, false);
        bind(Geometry::RecordsSourceSlot, captured);
        bind(Geometry::RecordsOutputSlot, records);
        const Geometry::RecordsParams records_params{
            .records = static_cast<u32>(u64{input_primitives} * plan.input_vertices),
            .vertices = plan.input_vertices,
            .step = plan.input_step,
            .count = xfb_draw.count,
            .primitives = per_instance,
        };
        scheduler.BindBytes(ShaderStage::Compute, Geometry::RecordsCountSlot, &records_params,
                            static_cast<u32>(sizeof(records_params)));
        scheduler.Dispatch(
            {(records_params.records + Geometry::GroupSize - 1) / Geometry::GroupSize, 1, 1},
            {Geometry::GroupSize, 1, 1});
        // 3. The geometry program, a thread for each input primitive.
        scheduler.BeginCompute(geometry_kernel, false);
        BindStageResources(geo->kernel.program, ShaderStage::Compute, stage_bindings[1],
                           texel_views[1], std::span(views).subspan(first_view[1]),
                           std::span(samplers).subspan(first_sampler[1]), targets.width,
                           targets.height);
        bind(geo->extra_slots[0], records);
        bind(geo->extra_slots[1], geo_draw.output);
        bind(geo->extra_slots[2], regions);
        scheduler.Dispatch({rounded / Geometry::GroupSize, plan.invocations, 1},
                           {Geometry::GroupSize, 1, 1});
        // 4. The strips of every primitive's region, in order, as one list and the draw's
        // arguments.
        const Geometry::GeometryParams params{
            .groups = groups,
            .stride = plan.RegionStride(),
            .topology = plan.output_topology,
            .capacity = static_cast<u32>(list_words),
            .base = 0,
            .width = plan.IndexStride(),
        };
        const auto pack = [&](const Geometry::GeometryParams& pack_params, const StagingBufferRef& list,
                              const StagingBufferRef& args, const Object& kernel, u32 threads) {
            scheduler.BeginCompute(kernel.Get(), false);
            bind(0, regions);
            bind(1, list);
            bind(2, args);
            bind(3, counts);
            scheduler.BindBytes(ShaderStage::Compute, 4, &pack_params,
                                static_cast<u32>(sizeof(pack_params)));
            scheduler.Dispatch({(threads + Geometry::GroupSize - 1) / Geometry::GroupSize, 1, 1},
                               {Geometry::GroupSize, 1, 1});
        };
        const auto compact = [&](u32 stream_index, const StagingBufferRef& list,
                                 const StagingBufferRef& args) {
            Geometry::GeometryParams stream_params = params;
            stream_params.base = stream_index * plan.IndexStride();
            pack(stream_params, list, args, geometry_count_kernel, groups);
            pack(stream_params, list, args, geometry_scan_kernel, 1);
            pack(stream_params, list, args, geometry_write_kernel, groups);
        };
        compact(0, geo_draw.indices, geo_draw.args);
        if (geo->xfb_stride != std::array<u32, Xfb::MaxBuffers>{}) {
            if ((geo->xfb_streams & 1U) != 0) {
                CaptureGeometryOutput(pipeline, *geo, geo_draw.output, geo_draw.indices,
                                      geo_draw.args, input_primitives, 0);
            }
            // Streams 1-3: their own compacted lists, into their own buffers.
            for (u32 stream_index = 1; stream_index < plan.streams; ++stream_index) {
                if (((geo->xfb_streams >> stream_index) & 1U) == 0) {
                    continue;
                }
                const StagingBufferRef list = stream(list_words * 4);
                const StagingBufferRef args = stream(32);
                compact(stream_index, list, args);
                CaptureGeometryOutput(pipeline, *geo, geo_draw.output, list, args, input_primitives,
                                      stream_index);
            }
        }
        ++geometry_draws;
        geometry_primitives += input_primitives;
        if (geometry_readbacks.size() >= 4096) {
            ResolveGeometryCounters();
        }
        geometry_readbacks.push_back({geo_draw.args, scheduler.CurrentTick(), plan.output_topology});
    }

    if (xfb_draw.wanted && (pipeline.Xfb() != nullptr || pipeline.Expand() != nullptr)) {
        // Transform feedback: the vertex program as a kernel writing every vertex's outputs, then
        // the gather kernel copying the captured varyings into the guest's buffers. An expanded
        // draw keeps the outputs for its vertex function and gathers nothing.
        XfbCapture& capture = xfb_draw.expand ? *pipeline.Expand() : *pipeline.Xfb();
        const size_t variant =
            xfb_draw.index_size == 0 ? 0 : xfb_draw.index_size == 2 ? 1 : 2;
        const auto [capture_kernel, gather_kernel] =
            pipeline.XfbKernelStates(device, variant, layout_desc);
        const auto& feedbacks = buffer_cache_runtime.TransformFeedbacks();
        bool captures = capture_kernel != nullptr && gather_kernel != nullptr;
        for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
            if (capture.stride[buffer] != 0 &&
                (feedbacks[buffer].buffer == nullptr || feedbacks[buffer].size == 0)) {
                captures = false;
            }
        }
        BufferHandle index_buffer{};
        u64 index_offset = 0;
        if (captures && xfb_draw.index_size != 0) {
            captures = resolve_capture_indices(index_buffer, index_offset);
        }
        if (!captures) {
            LOG_ONCE(warned_xfb_unbuilt,
                     "Metal: a transform feedback capture was skipped (kernel not built or "
                     "buffer unbound)");
            if (xfb_draw.expand) {
                prepare_refusal = "expansion kernel not built by Metal";
                return false;
            }
            xfb_draw.wanted = false;
        } else if (xfb_draw.captured != 0) {
            const u32 vertices = xfb_draw.count * xfb_draw.instances;
            const StagingBufferRef outputs = staging_pool.Stream(
                (std::max)(u64{vertices} * capture.vertex_stride, u64{16}), 16);
            scheduler.BeginCompute(capture_kernel, false);
            BindStageResources(capture.capture[variant], ShaderStage::Compute, stage_bindings[0],
                               texel_views[0], std::span(views).subspan(first_view[0]),
                               std::span(samplers).subspan(first_sampler[0]), targets.width,
                               targets.height);
            bind_vertex_streams([&](u32 slot, BufferHandle buffer, u32 offset) {
                scheduler.BindBuffer(ShaderStage::Compute, slot, buffer, offset);
            });
            const ShaderTranslate::TessellationSlots& slots =
                capture.capture[variant].tessellation_slots;
            if (slots.output != ~0U) {
                scheduler.BindBuffer(ShaderStage::Compute, slots.output, outputs.buffer,
                                     static_cast<u32>(outputs.offset));
            }
            if (xfb_draw.index_size != 0 && slots.index != ~0U) {
                scheduler.BindBuffer(ShaderStage::Compute, slots.index, index_buffer,
                                     static_cast<u32>(index_offset));
            }
            scheduler.DispatchStageIn(xfb_draw.index_size != 0 ? 0U : xfb_draw.first,
                                      xfb_draw.count, xfb_draw.first_instance,
                                      xfb_draw.instances);
            xfb_draw.outputs = outputs;
        }
        if (xfb_draw.wanted && !xfb_draw.expand && xfb_draw.captured != 0) {
            const StagingBufferRef& outputs = xfb_draw.outputs;
            scheduler.BeginCompute(gather_kernel, false);
            scheduler.BindBuffer(ShaderStage::Compute, Xfb::GatherSourceSlot, outputs.buffer,
                                 static_cast<u32>(outputs.offset));
            for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
                if (capture.stride[buffer] != 0) {
                    scheduler.BindBuffer(ShaderStage::Compute, Xfb::GatherBufferSlot + buffer,
                                         feedbacks[buffer].buffer,
                                         feedbacks[buffer].offset + xfb_draw.offsets[buffer]);
                }
            }
            scheduler.BindBytes(ShaderStage::Compute, Xfb::GatherCountSlot, &xfb_draw.captured,
                                static_cast<u32>(sizeof(u32)));
            scheduler.Dispatch(
                {(xfb_draw.captured + Xfb::GatherGroupSize - 1) / Xfb::GatherGroupSize, 1, 1},
                {Xfb::GatherGroupSize, 1, 1});
            ++xfb_captures;
            xfb_vertices_captured += xfb_draw.captured;
        }
    }

    if (scheduler.SegmentSerial() != draw_segment) {
        // Committed in the middle of the draw: what the buffer cache noted as used is gone.
        scheduler.BlockAhead();
    }
    DrawState draw{};
    draw.pipeline = state;
    draw.depth_stencil = depth_stencil_handle;
    draw.color_written = color_written;
    draw.depth_written = depth_stencil.depth_test != 0 && depth_stencil.depth_write != 0;
    draw.stencil_written =
        depth_stencil.stencil_test != 0 &&
        (depth_stencil.front.write_mask != 0 || depth_stencil.back.write_mask != 0);
    // Viewport i, as Vulkan's GetViewportState makes it (the shader flips y for Metal).
    const auto make_viewport = [&](size_t index) {
        Viewport viewport{};
        const auto& transform = regs.viewport_transform[index];
        if (regs.viewport_scale_offset_enabled) {
            viewport.x = transform.translate_x - transform.scale_x;
            viewport.width = transform.scale_x * 2.0f;
            viewport.y = transform.translate_y - transform.scale_y;
            viewport.height = transform.scale_y * 2.0f;
            if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
                viewport.y += static_cast<f32>(regs.surface_clip.height);
                viewport.height = -viewport.height;
            }
            // A swizzle the vertex program does (viewport 0's, RasterPatch) includes the y.
            const bool swizzled_in_program =
                index == 0 && RasterPatch::ViewportSwizzleEnabled() &&
                RasterPatch::SwizzleInProgram(transform.swizzle.raw & 0x7777);
            if (!swizzled_in_program &&
                transform.swizzle.y == Maxwell::ViewportSwizzle::NegativeY) {
                viewport.y += viewport.height;
                viewport.height = -viewport.height;
            }
            const f32 reduce_z =
                regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
            viewport.znear = transform.translate_z - transform.scale_z * reduce_z;
            viewport.zfar = transform.translate_z + transform.scale_z;
        } else {
            const bool use_clip = regs.surface_clip.width != 0 && regs.surface_clip.height != 0;
            const f32 conv_width = use_clip
                                       ? static_cast<f32>(regs.surface_clip.width)
                                       : (std::max)(1.0f, static_cast<f32>(targets.width) /
                                                              (is_rescaling ? resolution.up_factor : 1.0f));
            const f32 conv_height = use_clip
                                        ? static_cast<f32>(regs.surface_clip.height)
                                        : (std::max)(1.0f, static_cast<f32>(targets.height) /
                                                               (is_rescaling ? resolution.up_factor : 1.0f));
            viewport.x = use_clip ? static_cast<f32>(regs.surface_clip.x) : 0.0f;
            viewport.y = use_clip ? static_cast<f32>(regs.surface_clip.y) : 0.0f;
            viewport.width = conv_width;
            viewport.height = conv_height;
            if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
                viewport.y = conv_height - viewport.y;
                viewport.height = -conv_height;
            }
        }
        if (is_rescaling) {
            // As Vulkan's GetViewportState: the guest's viewport in scaled pixels.
            const f32 scale = resolution.up_factor;
            viewport.x *= scale;
            viewport.y *= scale;
            viewport.width *= scale;
            viewport.height *= scale;
        }
        // Ocarina's scene draws under EDEN_NX2_DLSS_JITTER (UpdateOcarinaJitter).
        viewport.x += oot_jitter_applied[0];
        viewport.y += oot_jitter_applied[1];
        if (viewport.width == 0.0f) {
            viewport.width = 1.0f;
        }
        if (viewport.height == 0.0f) {
            viewport.height = 1.0f;
        }
        return viewport;
    };
    const auto make_scissor = [&](size_t index) -> Scissor {
        const auto scale = [&](u32 value) {
            return is_rescaling ? resolution.ScaleUp(value) : value;
        };
        if (!regs.viewport_scale_offset_enabled) {
            if (regs.surface_clip.width == 0 || regs.surface_clip.height == 0) {
                return {0, 0, targets.width, targets.height};
            }
            const u32 x = regs.surface_clip.x;
            const u32 width = regs.surface_clip.width;
            const u32 height = regs.surface_clip.height;
            s32 y = static_cast<s32>(regs.surface_clip.y);
            if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
                y = static_cast<s32>(regs.surface_clip.height) -
                    static_cast<s32>(regs.surface_clip.y) - static_cast<s32>(height);
            }
            const u32 clamped_y = static_cast<u32>((std::max)(y, 0));
            return {scale(x), scale(clamped_y), scale(width), scale(height)};
        }
        const auto& scissor = regs.scissor_test[index];
        if (!scissor.enable) {
            return {0, 0, targets.width, targets.height};
        }
        const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
        const s32 clip_height = static_cast<s32>(regs.surface_clip.height);
        const s32 min_y = (std::max)(lower_left ? clip_height - static_cast<s32>(scissor.max_y)
                                                : static_cast<s32>(scissor.min_y.Value()),
                                     0);
        const s32 max_y = (std::max)(lower_left ? clip_height - static_cast<s32>(scissor.min_y)
                                                : static_cast<s32>(scissor.max_y.Value()),
                                     0);
        const u32 min_x = scale(static_cast<u32>(scissor.min_x.Value()));
        const u32 max_x = scale(static_cast<u32>(scissor.max_x.Value()));
        const u32 scaled_min_y = scale(static_cast<u32>(min_y));
        const u32 scaled_max_y = scale(static_cast<u32>((std::max)(max_y, min_y)));
        return {min_x, scaled_min_y, max_x > min_x ? max_x - min_x : 0U,
                scaled_max_y - scaled_min_y};
    };
    draw.viewport = make_viewport(0);
    draw.scissor = make_scissor(0);
    // A program that picks the viewport gets all 16 with their scissors, as Vulkan sets them
    // (EDEN_NX2_METAL_VIEWPORTS=0: viewport 0 only, as before).
    static const bool all_viewports = EnvironmentOn("EDEN_NX2_METAL_VIEWPORTS");
    const Shader::Info& pre_raster_info =
        native != nullptr                   ? native->mesh.info
        : (tess != nullptr && tess->domain) ? tess->domain->evaluate.program.info
                                            : pipeline.Vertex().info;
    const bool uses_viewport_index =
        pre_raster_info.stores[Shader::IR::Attribute::ViewportIndex] ||
        (geo != nullptr &&
         std::ranges::find(geo->plan.output_words,
                           static_cast<u32>(Shader::IR::Attribute::ViewportIndex)) !=
             geo->plan.output_words.end());
    if (all_viewports && uses_viewport_index) {
        draw.viewport_count = static_cast<u8>(Maxwell::NumViewports);
        for (size_t index = 1; index < Maxwell::NumViewports; ++index) {
            draw.more_viewports[index - 1] = make_viewport(index);
            draw.more_scissors[index - 1] = make_scissor(index);
        }
    }
    // Mesh positions are flipped in MSL (FlipMeshPositionY); EDEN_NX2_METAL_MESH_FLIP_Y=1
    // overrides by flipping the viewport instead.
    static const bool mesh_flip_y = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_MESH_FLIP_Y");
        return value != nullptr && value[0] == '1';
    }();
    if (mesh_flip_y && native != nullptr) {
        const auto flip = [](Viewport& viewport) {
            viewport.y += viewport.height;
            viewport.height = -viewport.height;
        };
        flip(draw.viewport);
        for (size_t index = 1; index < draw.viewport_count; ++index) {
            flip(draw.more_viewports[index - 1]);
        }
        LOG_ONCE(logged_mesh_flip_y, "Metal: native mesh draws flip the viewport's y "
                                     "(EDEN_NX2_METAL_MESH_FLIP_Y=0 turns it off)");
    }
    const u32 scissor_bound_w = (std::max)(targets.width, 1U);
    const u32 scissor_bound_h = (std::max)(targets.height, 1U);
    const auto scissor_empty = [&](const Scissor& s) {
        return s.width == 0 || s.height == 0 || s.x >= scissor_bound_w || s.y >= scissor_bound_h;
    };
    const bool xfb_active =
        (xfb_draw.wanted && !xfb_draw.expand) || (geo != nullptr && geo->xfb_streams != 0);
    bool any_scissor_visible = false;
    for (size_t index = 0; index < draw.viewport_count; ++index) {
        const Scissor& s = index == 0 ? draw.scissor : draw.more_scissors[index - 1];
        if (!scissor_empty(s)) {
            any_scissor_visible = true;
        }
    }
    if (!any_scissor_visible && !xfb_active) {
        prepare_empty_why = "empty scissor";
        return false;
    }
    if (draw.viewport_count > 1 || !any_scissor_visible) {
        for (size_t index = 0; index < draw.viewport_count; ++index) {
            const Scissor& s = index == 0 ? draw.scissor : draw.more_scissors[index - 1];
            if (scissor_empty(s)) {
                (index == 0 ? draw.viewport : draw.more_viewports[index - 1]) = {};
            }
        }
    }
    // Triangles drawn as lines when the faces the draw keeps are (Metal has one fill mode for
    // both; a point fill mode it has not).
    const bool front_lines = regs.polygon_mode_front == Maxwell::PolygonMode::Line;
    const bool back_lines = regs.polygon_mode_back == Maxwell::PolygonMode::Line;
    const bool culls = regs.gl_cull_test_enabled != 0;
    static const bool fill_lines = EnvironmentOn("EDEN_NX2_METAL_FILL_LINES");
    draw.fill_lines =
        fill_lines &&
        ((front_lines && back_lines) ||
         (culls && regs.gl_cull_face == Maxwell::CullFace::Back && front_lines) ||
         (culls && regs.gl_cull_face == Maxwell::CullFace::Front && back_lines));
    if (regs.gl_cull_test_enabled != 0) {
        switch (regs.gl_cull_face) {
        case Maxwell::CullFace::Front:
            draw.cull_mode = 1;
            break;
        case Maxwell::CullFace::Back:
            draw.cull_mode = 2;
            break;
        case Maxwell::CullFace::FrontAndBack:
            // Nothing is drawn but points and lines.
            if (TopologyClass(topology) == 2 && !xfb_active) {
                prepare_empty_why = "triangles culled front and back";
                return false;
            }
            break;
        }
    }
    bool front_ccw = regs.gl_front_face == Maxwell::FrontFace::CounterClockWise;
    if (regs.window_origin.flip_y != 0) {
        front_ccw = !front_ccw;
    }
    draw.front_ccw = front_ccw;
    draw.depth_clamp =
        regs.viewport_clip_control.geometry_clip !=
            Maxwell::ViewportClipControl::GeometryClip::Passthrough &&
        regs.viewport_clip_control.geometry_clip !=
            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ &&
        regs.viewport_clip_control.geometry_clip !=
            Maxwell::ViewportClipControl::GeometryClip::FrustumZ;
    u32 offset_mode = TopologyClass(topology);
    if (native != nullptr) {
        offset_mode = TopologyClass(static_cast<Maxwell::PrimitiveTopology>(native->topology));
    } else if (mesh != nullptr) {
        offset_mode = TopologyClass(static_cast<Maxwell::PrimitiveTopology>(mesh->topology));
    } else if (geo != nullptr) {
        offset_mode =
            geo->plan.output_topology == static_cast<u32>(Shader::OutputTopology::PointList) ? 0U
            : geo->plan.output_topology == static_cast<u32>(Shader::OutputTopology::LineStrip) ? 1U
                                                                                               : 2U;
    } else if (tess != nullptr && tess->domain) {
        offset_mode = tess->domain->emulation.lines ? 1U : 0U;
    }
    if (offset_mode == 2U && draw.fill_lines) {
        offset_mode = 1U;
    }
    const bool polygon_offset_enabled =
        offset_mode == 0U   ? regs.polygon_offset_point_enable != 0
        : offset_mode == 1U ? regs.polygon_offset_line_enable != 0
                            : regs.polygon_offset_fill_enable != 0;
    if (polygon_offset_enabled) {
        // The hardware halves the depth bias units (as OpenGL and Vulkan apply it).
        draw.depth_bias = regs.depth_bias / 2.0f;
        // Per-pixel slope term: scaled back to guest pixels on a rescaled target, or shadow
        // maps get self-shadow stripes above 1x (zortos293/Lichii#1925).
        draw.slope_scale =
            regs.slope_scale_depth_bias * (is_rescaling ? resolution.up_factor : 1.0f);
        draw.bias_clamp = regs.depth_bias_clamp;
    }
    draw.stencil_front_ref = regs.stencil_front_ref;
    draw.stencil_back_ref =
        regs.stencil_two_side_enable != 0 ? regs.stencil_back_ref : regs.stencil_front_ref;
    draw.blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                        regs.blend_color.a};
    if (xfb_draw.wanted && xfb_draw.expand && xfb_draw.prim <= RasterExpand::PrimLineLoop) {
        draw.cull_mode = 0; // A line has no faces
    }
    if (regs.depth_bounds_enable != 0 && targets.has_depth &&
        regs.depth_bounds[0] <= regs.depth_bounds[1]) {
        // Applied by GPUs that have the test (Scheduler::SetDrawState); Metal wants [0, 1].
        draw.depth_bounds_min = std::clamp(regs.depth_bounds[0], 0.0f, 1.0f);
        draw.depth_bounds_max = std::clamp(regs.depth_bounds[1], 0.0f, 1.0f);
    }

    if (flash_watch.WatchesPasses()) {
        FlashWatchPass(framebuffer, {&pipeline.Key().unique_hashes[1], 5});
    }
    if (frame_dump.Capturing()) {
        FrameDumpPass(framebuffer, {&pipeline.Key().unique_hashes[1], 5});
        if (frame_dump.CapturingSampled()) {
            if (frame_dump.CapturingTextures()) {
                std::vector<FrameDump::Sampled> sampled;
                for (size_t index = 0; index < stages.size(); ++index) {
                    if (!absent(index)) {
                        AppendSampledDump(static_cast<u32>(stages[index]), programs[index]->info,
                                          std::span(views).subspan(first_view[index]),
                                          std::span(samplers).subspan(first_sampler[index]),
                                          sampled);
                    }
                }
                frame_dump.NoteSampled(sampled);
            }
            FrameDumpBuffers(false, pipeline.Key().unique_hashes);
            const auto& draw_state = maxwell3d->draw_manager.draw_state;
            frame_dump.NoteDrawRecord({
                .mesh = mesh_workgroups != 0,
                .indirect = false,
                .first = is_indexed ? draw_state.index_buffer.first : draw_state.vertex_buffer.first,
                .elements =
                    is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
                .instances = draw_state.instance_count,
                .alpha_test = regs.alpha_test_enabled ? static_cast<u32>(regs.alpha_test_func) : 0,
                .alpha_ref = regs.alpha_test_ref,
                .alpha_to_coverage = regs.anti_alias_alpha_control.alpha_to_coverage != 0,
            });
        }
    }
    scheduler.BeginRendering(targets);
    if (attachment_fit_memo.valid && attachment_fit_memo.fitted_targets == targets) {
        attachment_fit_memo.pass_serial = scheduler.RenderPassSerial();
    }
    if (writes_storage) {
        pass_stores = scheduler.RenderPassSerial();
    }
    scheduler.SetDrawState(draw);

    if (xfb_draw.wanted && xfb_draw.expand) {
        // The generated vertex function's source and constants.
        RasterExpand::Params params{};
        params.half_viewport[0] = (std::max)(std::abs(draw.viewport.width) * 0.5f, 0.5f);
        params.half_viewport[1] = (std::max)(std::abs(draw.viewport.height) * 0.5f, 0.5f);
        params.width = RasterExpand::LineWidth(regs) * (is_rescaling ? resolution.up_factor : 1.0f);
        params.prim = xfb_draw.prim;
        params.count = xfb_draw.count;
        params.factor = static_cast<f32>(regs.line_stipple_params.factor.Value());
        params.pattern = regs.line_stipple_params.pattern.Value();
        scheduler.BindBuffer(ShaderStage::Vertex, RasterExpand::SourceSlot,
                             xfb_draw.outputs.buffer, static_cast<u32>(xfb_draw.outputs.offset));
        scheduler.BindBytes(ShaderStage::Vertex, RasterExpand::ParamsSlot, &params,
                            static_cast<u32>(sizeof(params)));
    } else if (!mesh_pipeline && tess == nullptr && geo == nullptr) {
        bind_vertex_streams([&](u32 slot, BufferHandle buffer, u32 offset) {
            scheduler.BindVertexBuffer(slot, buffer, offset);
        });
    }
    if (tess != nullptr && tess->domain && !tess_geo) {
        // The evaluated vertices the vertex function draws.
        scheduler.BindBuffer(ShaderStage::Vertex, 0, tess_buffers.captured.buffer,
                             static_cast<u32>(tess_buffers.captured.offset));
    }

    // Each stage's buffers (appended by the buffer cache in binding order), textures, samplers
    // and images, into the slots the translator gave them.
    for (size_t index = 0; index < stages.size(); ++index) {
        // Kernels were bound when they were dispatched.
        if (absent(index) || (index == 1 && native == nullptr) ||
            (index == 0 && (tess != nullptr || geo != nullptr)) || index == 3 ||
            (index == 2 && tess != nullptr && tess->domain)) {
            continue;
        }
        ShaderStage metal_stage = index == 4 ? ShaderStage::Fragment : ShaderStage::Vertex;
        if (native != nullptr && index < 2) {
            metal_stage = index == 0 && native_task ? ShaderStage::Object : ShaderStage::Mesh;
        }
        if (index == 0 && mesh != nullptr) {
            BindMeshBuffers(pipeline.Vertex(), metal_stage, *mesh);
            continue;
        }
        if (index == 2) {
            const ShaderTranslate::TessellationSlots& slots = programs[2]->tessellation_slots;
            if (slots.input != ~0U) {
                scheduler.BindBuffer(metal_stage, slots.input, tess_buffers.control_points.buffer,
                                     static_cast<u32>(tess_buffers.control_points.offset));
            }
            if (slots.patch_input != ~0U) {
                scheduler.BindBuffer(metal_stage, slots.patch_input,
                                     tess_buffers.patch_constants.buffer,
                                     static_cast<u32>(tess_buffers.patch_constants.offset));
            }
        }
        BindStageResources(*programs[index], metal_stage, stage_bindings[index], texel_views[index],
                           std::span(views).subspan(first_view[index]),
                           std::span(samplers).subspan(first_sampler[index]), targets.width,
                           targets.height);
    }
    return true;
}

void RasterizerMetal::DrawIndirect() {
    if (maxwell3d->draw_manager.indirect_state.is_byte_count) {
        DrawByteCount();
        return;
    }
    xfb_draw = {};
    ++draws;
    skip_valid = false;
    grid_draw = {};
    trace_instances = 0;
    SCOPE_EXIT {
        if (CachePathsEnabled()) {
            gpu.TickWork();
        }
    };
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    gpu_memory->FlushCaching();
    FlushWork();
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteDraw(*maxwell3d, pipeline_cache.LookupHashes());
    }
    if (pipeline == nullptr) {
        RefuseDraw(pipeline_cache.Refusal());
        return;
    }
    if (!pipeline->BuildFunctions(device)) {
        RefuseDraw("program's MSL not compiled by Metal");
        return;
    }
    ++draws_with_pipeline;
    TrackOotTonemappedImage();
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (!ConditionHoldsOnCpu()) {
        NoteEmptyDraw("indirect draw under a condition false on the CPU");
        return;
    }
    if (pipeline->Geo() != nullptr) {
        DrawGeometryIndirect(*pipeline);
        return;
    }
    if (pipeline->IsMeshPipeline()) {
        if (SkipMeshDraws(pipeline->Native() == nullptr)) {
            RefuseDraw("EDEN_NX2_NO_MESH");
            return;
        }
        DrawMeshIndirect(*pipeline);
        return;
    }
    if (pipeline->Tess() != nullptr) {
        DrawTessellatedIndirect(*pipeline);
        return;
    }
    const bool u8_indices = params.is_indexed && maxwell3d->regs.index_buffer.format ==
                                                     Maxwell::IndexFormat::UnsignedByte;
    const bool provoking_last = ProvokingLastWanted(*pipeline);
    const bool rewrite = u8_indices || provoking_last ||
                         IndexConvert::NeedsConversion(DrawTopology(*maxwell3d)) ||
                         RestartOffWidens(params.is_indexed);
    const std::optional<u32> primitive =
        PrimitiveType(IndexConvert::ConvertedTopology(DrawTopology(*maxwell3d), provoking_last));
    // EDEN_NX2_METAL_GPU_INDIRECT=0: a count is read from guest memory on the CPU and draws whose
    // indices need rewriting are refused, as before.
    static const bool on_gpu = EnvironmentOn("EDEN_NX2_METAL_GPU_INDIRECT");
    if (!primitive || (rewrite && !on_gpu)) {
        RefuseDraw(!primitive ? "indirect draw of a topology Metal can't draw"
                   : u8_indices ? "indirect draw with 8-bit indices"
                                : "indirect draw whose indices must be rewritten");
        return;
    }
    const u32 stride_bytes = static_cast<u32>(
        params.stride != 0 ? params.stride : (params.is_indexed ? 5 : 4) * sizeof(u32));
    indirect_plan = {};
    if (on_gpu && (rewrite || params.include_count)) {
        EnsureIndexKernels();
        const u32 index_size = static_cast<u32>(draw_state.index_buffer.FormatSizeInBytes());
        const bool restart = maxwell3d->regs.primitive_restart.enabled != 0;
        u32 capacity = 0;
        if (rewrite) {
            // Indexed: the rewrite of the whole bound index buffer at most; plain: up to 256K.
            const u32 elements = params.is_indexed
                                     ? static_cast<u32>(draw_state.index_buffer.EndAddress() -
                                                        draw_state.index_buffer.StartAddress()) /
                                           (std::max)(index_size, 1U)
                                     : 1U << 18;
            capacity = (std::max)(
                IndexConvert::ConvertedCount(DrawTopology(*maxwell3d), elements, provoking_last), 6U);
        }
        constexpr u64 MaxRewriteBytes = u64{64} << 20;
        if (indirect_kernels_failed || index_kernel_failed ||
            u64{capacity} * params.max_draw_counts * sizeof(u32) > MaxRewriteBytes ||
            stride_bytes % sizeof(u32) != 0) {
            RefuseDraw("indirect draw the GPU can't prepare");
            return;
        }
        indirect_plan = IndirectPlan{
            .wanted = true,
            .rewrite = rewrite,
            .indexed = params.is_indexed,
            .has_count = params.include_count,
            .restart = restart && params.is_indexed,
            .mode = rewrite ? IndexConvert::RewriteMode(DrawTopology(*maxwell3d), provoking_last,
                                                        restart && params.is_indexed)
                            : 0U,
            .index_size = params.is_indexed ? index_size : 0U,
            .max_draws = static_cast<u32>(params.max_draw_counts),
            .stride_words = stride_bytes / static_cast<u32>(sizeof(u32)),
            .capacity = capacity,
        };
    }
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
    };
    index_rewrite = {};
    buffer_cache.SetDrawIndirect(&params);
    SCOPE_EXIT {
        buffer_cache.SetDrawIndirect(nullptr);
        indirect_plan = {};
    };
    if (!PrepareDraw(*pipeline, params.is_indexed)) {
        NoteUnprepared();
        return;
    }
    CountOcclusion();
    if (indirect_plan.wanted) {
        Coverage::Note(Coverage::Work::Draw, rewrite ? Coverage::Path::RewrittenIndices
                                                     : Coverage::Path::IndirectLoop);
        const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
        const u32 index_size = index.format == Maxwell::IndexFormat::UnsignedShort ? 2U : 4U;
        const u32 words = params.is_indexed ? Indirect::IndexedWords : Indirect::PlainWords;
        for (u32 draw = 0; draw < indirect_plan.max_draws; ++draw) {
            if (rewrite) {
                const u64 control = indirect_plan.control.offset +
                                    (u64{draw} * Indirect::ControlWords +
                                     Indirect::ControlDrawWord) * sizeof(u32);
                scheduler.DrawIndexedIndirect(
                    *primitive, 4, indirect_plan.indices.buffer,
                    indirect_plan.indices.offset + u64{draw} * indirect_plan.capacity * sizeof(u32),
                    indirect_plan.control.buffer, control);
                continue;
            }
            const u64 offset = indirect_plan.arguments.offset + u64{draw} * words * sizeof(u32);
            if (params.is_indexed) {
                scheduler.DrawIndexedIndirect(*primitive, index_size, index.buffer, index.offset,
                                              indirect_plan.arguments.buffer, offset);
            } else {
                scheduler.DrawIndirect(*primitive, indirect_plan.arguments.buffer, offset);
            }
        }
        ++indirect_draws_on_gpu;
        ++draws_encoded;
        return;
    }
    const auto [indirect, indirect_offset] = buffer_cache.GetDrawIndirectBuffer();
    size_t count = params.max_draw_counts;
    // Metal has no multi draw: one call per draw, and a count read on the CPU.
    Coverage::Note(Coverage::Work::Draw, params.include_count || count > 1
                                             ? Coverage::Path::IndirectLoop
                                             : Coverage::Path::Native);
    if (params.include_count) {
        // Metal has no draw count from a buffer: the guest's count as it is in memory.
        count = (std::min)(count, size_t{gpu_memory->Read<u32>(params.count_start_address)});
        ++indirect_counts_read;
    }
    const size_t stride = params.stride != 0
                              ? params.stride
                              : (params.is_indexed ? 5 : 4) * sizeof(u32);
    const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
    const u32 index_size = index.format == Maxwell::IndexFormat::UnsignedShort ? 2U : 4U;
    for (size_t draw = 0; draw < count; ++draw) {
        const u64 offset = u64{indirect_offset} + draw * stride;
        if (params.is_indexed) {
            scheduler.DrawIndexedIndirect(*primitive, index_size, index.buffer, index.offset,
                                          indirect->Handle(), offset);
        } else {
            scheduler.DrawIndirect(*primitive, indirect->Handle(), offset);
        }
    }
    ++draws_encoded;
}

void RasterizerMetal::CaptureGeometryOutput(GraphicsPipeline& pipeline, GeometryCompute& geo,
                                            const StagingBufferRef& output,
                                            const StagingBufferRef& list,
                                            const StagingBufferRef& args, u32 primitives,
                                            u32 stream) {
    // The compaction's count is only on the GPU, and the byte offsets of the next draw depend on
    // it: the capture is dispatched with the clamp inside the kernel, and the CPU waits for the
    // count (one command buffer wait a draw that captures).
    void* const kernel = pipeline.GeometryXfbState(device, stream);
    const auto& feedbacks = buffer_cache_runtime.TransformFeedbacks();
    const auto& regs = maxwell3d->regs;
    // Only the buffers of this stream are captured from its list.
    std::array<u32, Xfb::MaxBuffers> strides{};
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        if (pipeline.Key().xfb.State().layouts[buffer].stream == stream) {
            strides[buffer] = geo.xfb_stride[buffer];
        }
    }
    Geometry::XfbParams params{};
    params.vertices = geo.plan.output_topology == 1 ? 1U : geo.plan.output_topology == 6 ? 2U : 3U;
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        if (strides[buffer] == 0) {
            continue;
        }
        if (feedbacks[buffer].buffer == nullptr || feedbacks[buffer].size == 0) {
            LOG_ONCE(warned_geometry_xfb, "Metal: a geometry transform feedback capture was skipped (buffer unbound)");
            return;
        }
        const auto& binding = regs.transform_feedback.buffers[buffer];
        const GPUVAddr base = binding.Address() + binding.start_offset;
        if (xfb_state.base[buffer] != base) {
            xfb_state.base[buffer] = base;
            xfb_state.written[buffer] = 0;
        }
        xfb_state.start[buffer] = static_cast<u32>(binding.start_offset);
        xfb_state.stride[buffer] = strides[buffer];
        params.capacity[buffer] = static_cast<u32>(binding.size);
        params.written[buffer] = xfb_state.written[buffer];
    }
    xfb_state.topology = geo.plan.output_topology == 1   ? Maxwell::PrimitiveTopology::Points
                         : geo.plan.output_topology == 6 ? Maxwell::PrimitiveTopology::Lines
                                                         : Maxwell::PrimitiveTopology::Triangles;
    if (kernel == nullptr) {
        return;
    }
    scheduler.BeginCompute(kernel, false);
    scheduler.BindBuffer(ShaderStage::Compute, Geometry::XfbRecordsSlot, output.buffer,
                         static_cast<u32>(output.offset));
    scheduler.BindBuffer(ShaderStage::Compute, Geometry::XfbListSlot, list.buffer,
                         static_cast<u32>(list.offset));
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        // A buffer the layout doesn't use still gets one bound.
        const u32 used = strides[buffer] != 0 ? buffer : 0;
        scheduler.BindBuffer(ShaderStage::Compute, Geometry::XfbBufferSlot + buffer,
                             feedbacks[used].buffer,
                             feedbacks[used].offset + (strides[used] != 0 ? xfb_state.written[used] : 0U));
    }
    scheduler.BindBuffer(ShaderStage::Compute, Geometry::XfbArgsSlot, args.buffer,
                         static_cast<u32>(args.offset));
    scheduler.BindBytes(ShaderStage::Compute, Geometry::XfbParamsSlot, &params,
                        static_cast<u32>(sizeof(params)));
    // One thread for each index the compaction could have listed.
    const u32 threads = primitives * geo.plan.invocations * geo.plan.max_vertices * 3;
    scheduler.Dispatch({(threads + Geometry::GroupSize - 1) / Geometry::GroupSize, 1, 1},
                       {Geometry::GroupSize, 1, 1});
    scheduler.Wait(scheduler.CurrentTick());
    u32 listed = 0;
    std::memcpy(&listed, args.mapped_span.data(), sizeof(listed));
    const u32 captured =
        Xfb::CapturedVertices(listed, params.vertices, strides, params.capacity, xfb_state.written);
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        xfb_state.written[buffer] += captured * strides[buffer];
    }
    ++xfb_captures;
    xfb_vertices_captured += captured;
}

void RasterizerMetal::EnsureGeometryKernels() {
    if (geometry_count_kernel || geometry_kernels_failed) {
        return;
    }
    std::string error;
    geometry_library = CreateLibrary(device, std::string{Geometry::PackSource}, error);
    const auto make = [&](const char* name) {
        const Object function =
            geometry_library ? CreateFunction(geometry_library.Get(), name) : Object{};
        return function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
    };
    geometry_feed_kernel = make("geo_feed");
    geometry_count_kernel = make("geo_count");
    geometry_scan_kernel = make("geo_scan");
    geometry_write_kernel = make("geo_write");
    if (!geometry_count_kernel || !geometry_scan_kernel || !geometry_write_kernel ||
        !geometry_feed_kernel) {
        geometry_count_kernel = {};
        geometry_kernels_failed = true;
        LOG_ERROR(Render_Metal, "Metal: geometry compaction kernels failed to build: {}", error);
    }
}

void RasterizerMetal::DrawGeometry(GraphicsPipeline& pipeline, bool is_indexed,
                                   u32 instance_count) {
    GeometryCompute& geo = *pipeline.Geo();
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const u32 count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    // After tessellation the draw's vertices are patch control points: the geometry program's
    // input primitives come from the evaluation.
    const bool after_tess = geo.from_tess;
    geometry_input_primitives = after_tess ? 0 : geo.plan.Primitives(count) * instance_count;
    CountDraw(count, instance_count);
    geometry_input_primitives = 0;
    if ((!after_tess && geo.plan.Primitives(count) == 0) || count == 0 || instance_count == 0) {
        NoteEmptyDraw("geometry draw of 0 primitives or instances");
        return;
    }
    const std::optional<u32> primitive = PrimitiveType(
        geo.plan.output_topology == 1   ? Maxwell::PrimitiveTopology::Points
        : geo.plan.output_topology == 6 ? Maxwell::PrimitiveTopology::Lines
                                        : Maxwell::PrimitiveTopology::Triangles);
    if (!primitive || !ConditionHoldsOnCpu()) {
        NoteEmptyDraw("geometry draw Metal can not draw or under a false condition");
        return;
    }
    xfb_draw = XfbDraw{
        .wanted = true,
        .first = draw_state.vertex_buffer.first,
        .count = count,
        .first_instance = draw_state.base_instance,
        .instances = instance_count,
        .index_size = !is_indexed ? 0U
                      : draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedShort ? 2U
                                                                                             : 4U,
        .widen_indices =
            is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte,
        .base_vertex = is_indexed ? static_cast<s32>(draw_state.base_index) : 0,
        .first_index = draw_state.index_buffer.first,
    };
    geo_draw = {};
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    SCOPE_EXIT {
        FinishTexelWritebacks();
        xfb_draw = {};
        geo_draw = {};
    };
    TessellationDraw tess_draw{
        .first = draw_state.vertex_buffer.first,
        .count = count,
        .first_instance = draw_state.base_instance,
        .instances = instance_count,
        .index_size = xfb_draw.index_size,
        .widen_indices = xfb_draw.widen_indices,
        .first_index = draw_state.index_buffer.first,
        .index_buffer = nullptr,
        .index_offset = 0,
    };
    if (after_tess && is_indexed) {
        static const bool tess_base = EnvironmentOn("EDEN_NX2_METAL_TESS_BASE_VERTEX");
        tess_draw.base_vertex = tess_base ? static_cast<s32>(draw_state.base_index) : 0;
    }
    if (!PrepareDraw(pipeline, is_indexed, 0, after_tess ? &tess_draw : nullptr)) {
        NoteUnprepared();
        return;
    }
    CountOcclusion();
    // The reader takes its vertices from the geometry kernel's records.
    scheduler.BindBuffer(ShaderStage::Vertex, 0, geo_draw.output.buffer,
                         static_cast<u32>(geo_draw.output.offset));
    scheduler.DrawIndexedIndirect(*primitive, 4, geo_draw.indices.buffer, geo_draw.indices.offset,
                                  geo_draw.args.buffer, geo_draw.args.offset);
    ++draws_encoded;
    Coverage::Note(Coverage::Work::Draw, Coverage::Path::GeometryKernels);
}

void RasterizerMetal::DrawGeometryIndirect(GraphicsPipeline& pipeline) {
    // The compute path sizes its buffers from the vertex count: the arguments are read on the CPU
    // (after the caches hand over what the GPU wrote), a draw each.
    const auto& params = maxwell3d->draw_manager.indirect_state;
    auto& draw_state = maxwell3d->draw_manager.draw_state;
    const bool indexed = params.is_indexed;
    const u32 words = indexed ? 5U : 4U;
    const u64 stride = params.stride != 0 ? params.stride : u64{words} * sizeof(u32);
    u32 num_draws = static_cast<u32>(params.max_draw_counts);
    if (params.include_count) {
        num_draws = (std::min)(num_draws, gpu_memory->Read<u32>(params.count_start_address));
    }
    for (u32 draw = 0; draw < num_draws; ++draw) {
        const GPUVAddr address = params.indirect_start_address + draw * stride;
        if (const std::optional<DAddr> device_address = gpu_memory->GpuToCpuAddress(address)) {
            FlushRegion(*device_address, words * sizeof(u32), VideoCommon::CacheType::BufferCache);
        }
        std::array<u32, 5> record{};
        gpu_memory->ReadBlockUnsafe(address, record.data(), words * sizeof(u32));
        if (indexed) {
            draw_state.index_buffer.count = record[0];
            draw_state.index_buffer.first = record[2];
            draw_state.base_index = record[3];
            draw_state.base_instance = record[4];
        } else {
            draw_state.vertex_buffer.count = record[0];
            draw_state.vertex_buffer.first = record[2];
            draw_state.base_instance = record[3];
        }
        ++geometry_indirect_draws;
        DrawGeometry(pipeline, indexed, record[1]);
    }
}

void RasterizerMetal::DrawByteCount() {
    const auto& params = maxwell3d->draw_manager.indirect_state;
    auto& draw_state = maxwell3d->draw_manager.draw_state;
    // The guest's counter holds the bytes the capture wrote (WriteQueryResult wrote them when
    // the report was made, on the CPU); the draw is that many bytes over the vertex stride.
    const u32 stride = static_cast<u32>(params.stride);
    if (stride == 0 || !gpu_memory) {
        RefuseDraw("transform feedback draw without a stride");
        return;
    }
    const u32 bytes = gpu_memory->Read<u32>(params.indirect_start_address);
    const u32 vertices = bytes / stride;
    ++xfb_byte_count_draws;
    if (vertices == 0) {
        NoteEmptyDraw("transform feedback draw of 0 vertices");
        return;
    }
    draw_state.vertex_buffer.first = 0;
    draw_state.vertex_buffer.count = vertices;
    draw_state.base_instance = 0;
    Draw(false, 1);
}

void RasterizerMetal::DrawTexture() {
    ++draws;
    SCOPE_EXIT {
        if (CachePathsEnabled()) {
            gpu.TickWork();
        }
    };
    // EDEN_NX2_METAL_DRAW_TEXTURE=0: refused, as this backend started.
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_DRAW_TEXTURE");
    if (!enabled) {
        RefuseDraw("DrawTexture");
        return;
    }
    if (!ConditionHoldsOnCpu()) {
        NoteEmptyDraw("texture draw under a condition false on the CPU");
        return;
    }
    gpu_memory->FlushCaching();
    FlushWork();
    // As Vulkan's: the texture's rectangle drawn scaled over the target's (BlitColor), filtered
    // as its sampler's magnification filter says.
    const auto& state = maxwell3d->draw_manager.draw_texture_state;
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (framebuffer == nullptr || framebuffer->Colors()[0].texture == nullptr) {
        RefuseDraw("DrawTexture without a colour target");
        return;
    }
    ImageView& texture = texture_cache.GetImageView(state.src_texture);
    Tegra::Texture::TSCEntry sampler{};
    const auto& regs = maxwell3d->regs;
    gpu_memory->ReadBlockUnsafe(regs.tex_sampler.Address() + u64{state.src_sampler} * sizeof(sampler),
                                &sampler, sizeof(sampler));
    const BlitPass pass{
        .src = texture.Handle(Shader::TextureType::Color2D),
        .src_rect = {static_cast<s32>(state.src_x0), static_cast<s32>(state.src_y0),
                     static_cast<s32>(state.src_x1), static_cast<s32>(state.src_y1)},
        .dst = framebuffer->Colors()[0],
        .dst_format = framebuffer->ColorFormats()[0],
        .dst_rect = {static_cast<s32>(state.dst_x0), static_cast<s32>(state.dst_y0),
                     static_cast<s32>(state.dst_x1), static_cast<s32>(state.dst_y1)},
        .dst_width = framebuffer->Width(),
        .dst_height = framebuffer->Height(),
        .linear = sampler.mag_filter == Tegra::Texture::TextureFilter::Linear,
        .keep_dst = true,
    };
    if (pass.src == nullptr || !scheduler.Blit(pass)) {
        RefuseDraw("DrawTexture into a format the blit can't draw");
        return;
    }
    ++draws_encoded;
    Coverage::Note(Coverage::Work::Draw, Coverage::Path::Native);
}

namespace {

/// EDEN_NX2_METAL_NETWORK_BYPASS=0: Ocarina of Time's anti-aliasing network runs whatever the
/// anti-aliasing setting says, as before the bypass reached this renderer.
bool NetworkBypassEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_NETWORK_BYPASS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

} // Anonymous namespace

VideoCommon::OotNetwork::BypassState* RasterizerMetal::OotNetworkBypass() {
    return NetworkBypassEnabled() ? &oot_network_bypass : nullptr;
}

void RasterizerMetal::TrackOotTonemappedImage() {
    if (!NetworkBypassEnabled() ||
        !VideoCommon::OotNetwork::BypassEnabled(Settings::GetCurrentProgramID())) {
        return;
    }
    const auto& hashes = pipeline_cache.LookupHashes();
    if (std::none_of(hashes.begin(), hashes.end(), VideoCommon::OotNetwork::IsTonemapShader)) {
        return;
    }
    const auto& target = maxwell3d->regs.rt[0];
    if (target.Address() == 0) {
        return;
    }
    std::scoped_lock lock{oot_network_bypass.mutex};
    if (oot_network_bypass.tonemapped.Address() != target.Address()) {
        LOG_INFO(Render_Metal, "Ocarina of Time: tonemapped image at {:#x} {}x{}", target.Address(),
                 target.width, target.height);
    }
    oot_network_bypass.tonemapped = target;
}

bool RasterizerMetal::OcarinaTemporalEnabled() {
    // The camera-motion path takes Ocarina instead when it is switched on.
    if (CameraMotionUpscaleEnabled() || !OcarinaTemporal::Enabled() ||
        Settings::GetCurrentProgramID() != VideoCommon::OotNetwork::ProgramId) {
        return false;
    }
    static const std::string replacement = [] {
        const char* const value = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return std::string{value != nullptr ? value : ""};
    }();
    // MetalFX takes Vulkan's Dlss backend on Apple; FSR 3.1 has no Metal path (the game's pass,
    // or the bilinear one, runs).
    return VideoCommon::SelectOcarinaUpscaler(Settings::OcarinaFsrChosen(),
                                              Settings::MetalFxTemporalChosen(), true,
                                              replacement) ==
               VideoCommon::OcarinaUpscaleBackend::Dlss &&
           ocarina_temporal.Available();
}

void RasterizerMetal::UpdateOcarinaJitter() {
    std::array<f32, 2> want{};
    // As Vulkan: only while MetalFX replaced the previous frame's upscale (a frame the game
    // upscales itself would show the offset as a shake), and only for draws that depth-test
    // against the scene depth at the scene size, so a depth pre-pass and the colour passes move
    // together and shadow maps, half-size and full-screen passes stay put.
    const auto& regs = maxwell3d->regs;
    if (oot_temporal_frame && *oot_temporal_frame + 1 >= oot_frame && regs.depth_test_enable &&
        oot_motion.extent[0] != 0 && VulkanSwitchesEnabled() &&
        VideoCommon::OcarinaJitterEnabled(VideoCommon::OcarinaUpscaleBackend::Dlss)) {
        const auto& targets = texture_cache.GetRenderTargets();
        if (targets.depth_buffer_id) {
            const ImageView& depth = texture_cache.GetImageView(targets.depth_buffer_id);
            if (oot_motion.SceneSized(depth.size.width, depth.size.height) &&
                !texture_cache.IsRescaling(depth)) {
                // Halton (2, 3) with AMD's phase count for the last upscaled frame's scale.
                want = WorkPolicy::OcarinaJitter(oot_frame, oot_jitter_phases);
                oot_jitter_frame = oot_frame;
                oot_jitter_offset = want;
            }
        }
    }
    oot_jitter_applied = want;
}

void RasterizerMetal::TrackOcarinaCamera() {
    if ((oot_motion.camera_frame == oot_frame && oot_motion.extent[0] >= 2560 &&
         oot_motion.extent[1] >= 1440) ||
        !OcarinaTemporalEnabled()) {
        return;
    }
    const auto& targets = texture_cache.GetRenderTargets();
    if (!targets.color_buffer_ids[0] || !targets.depth_buffer_id) {
        return;
    }
    const ImageView& color = texture_cache.GetImageView(targets.color_buffer_ids[0]);
    if (color.size.width < 640 || color.size.height < 360 ||
        color.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
        return;
    }
    // Reflections and half-size passes share the camera layout; the upscale reads the largest.
    if (oot_motion.camera_frame == oot_frame &&
        u64{color.size.width} * color.size.height <=
            u64{oot_motion.image_extent[0]} * oot_motion.image_extent[1]) {
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
    // The camera's extent is the scene's; the images are that size or up to 15 pixels larger at
    // a render scale the game pads (Vulkan takes those for its DLSS / MetalFX backend too).
    const std::array<u32, 2> scene{static_cast<u32>(extent_check[0]),
                                   static_cast<u32>(extent_check[1])};
    if (static_cast<f32>(scene[0]) != extent_check[0] ||
        static_cast<f32>(scene[1]) != extent_check[1] || color.size.width < scene[0] ||
        color.size.height < scene[1] || color.size.width - scene[0] >= 16 ||
        color.size.height - scene[1] >= 16) {
        return;
    }
    oot_motion.camera_frame = oot_frame;
    oot_motion.camera_channel = gpu_memory;
    oot_motion.extent = scene;
    oot_motion.image_extent = {color.size.width, color.size.height};
    device_memory.ReadBlockUnsafe(*device_addr + 0x30, oot_motion.current.data(),
                                  sizeof(oot_motion.current));
    device_memory.ReadBlockUnsafe(*device_addr + 0x180, oot_motion.previous.data(),
                                  sizeof(oot_motion.previous));
}

bool RasterizerMetal::CameraMotionUpscaleEnabled() {
    if (!camera_motion_enabled) {
        static const bool switched_on = [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_OOT_CAMERA_MOTION");
            return value != nullptr && value[0] == '1';
        }();
        camera_motion_enabled = switched_on &&
                                Settings::GetCurrentProgramID() ==
                                    VideoCommon::OotNetwork::ProgramId &&
                                TemporalUpscaler::IsSupported(device);
    }
    return *camera_motion_enabled;
}

bool RasterizerMetal::UsesCameraMotionUpscale() const {
    return camera_motion_enabled.value_or(false);
}

// Vulkan's NoteCameraMotionDraw without the jitter: notes the camera and depth buffer of a draw
// by the scene's perspective camera, for AccelerateCameraMotionUpscale.
void RasterizerMetal::NoteCameraMotionDraw() {
    namespace Ocarina = VideoCommon::OcarinaUpscale;
    auto& motion = camera_motion;
    if (!motion.armed) {
        return;
    }
    const auto& regs = maxwell3d->regs;
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
    const auto& camera = maxwell3d->state.shader_stages[0].const_buffers[Ocarina::CameraBuffer];
    if (depth == 0 || !camera.enabled || camera.size < Ocarina::CameraBytes) {
        return;
    }
    auto& cached = motion.camera;
    if (camera.address != cached.address) {
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
    if (!cached.perspective) {
        return;
    }
    CameraMotionScene* scene = nullptr;
    for (size_t i = 0; i < motion.num_scenes; ++i) {
        CameraMotionScene& candidate = motion.scenes[i];
        if (candidate.depth == depth && candidate.depth_info.size.width == depth_size[0] &&
            candidate.depth_info.size.height == depth_size[1]) {
            scene = &candidate;
        }
    }
    if (!scene) {
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
            scene->depth_info =
                VideoCommon::ImageInfo(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
        }
    }
    if (!scene) {
        return;
    }
    // Viewport 0 as the draw makes it (Vulkan's GetViewportState; no jitter).
    const bool rescaled = texture_cache.IsRescaling();
    const auto& transform = regs.viewport_transform[0];
    Viewport viewport{};
    viewport.x = transform.translate_x - transform.scale_x;
    viewport.width = transform.scale_x * 2.0f;
    viewport.y = transform.translate_y - transform.scale_y;
    viewport.height = transform.scale_y * 2.0f;
    if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
        viewport.y += static_cast<f32>(regs.surface_clip.height);
        viewport.height = -viewport.height;
    }
    if (transform.swizzle.y == Maxwell::ViewportSwizzle::NegativeY) {
        viewport.y += viewport.height;
        viewport.height = -viewport.height;
    }
    const f32 reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    viewport.znear = transform.translate_z - transform.scale_z * reduce_z;
    viewport.zfar = transform.translate_z + transform.scale_z;
    if (rescaled) {
        const f32 scale = Settings::values.resolution_info.up_factor;
        viewport.x *= scale;
        viewport.y *= scale;
        viewport.width *= scale;
        viewport.height *= scale;
    }
    scene->camera = camera.address;
    scene->view_projection = cached.view_projection;
    scene->viewport = viewport;
    scene->depth_minus_one = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne;
    scene->rescaled = rescaled;
    ++scene->draws;
}

// Vulkan's AccelerateCameraMotionUpscale with MetalFX (TextureCacheRuntime::
// UpscaleWithCameraMotion) as the upscaler and no jitter, which Metal's MetalFX path leaves off by
// default too.
bool RasterizerMetal::AccelerateCameraMotionUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                                    const Tegra::Engines::Fermi2D::Surface& dst) {
    namespace Ocarina = VideoCommon::OcarinaUpscale;
    auto& motion = camera_motion;
    motion.armed = true;
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
        motion.camera = {};
        if (!done) {
            motion.previous_screen.reset();
        }
        return done;
    };
    if (!scene) {
        return finish(false);
    }
    const Viewport& viewport = scene->viewport;
    const f32 width = static_cast<f32>(
        scene->rescaled ? Settings::values.resolution_info.ScaleUp(src.width) : src.width);
    const f32 height = static_cast<f32>(
        scene->rescaled ? Settings::values.resolution_info.ScaleUp(src.height) : src.height);
    if (std::abs(viewport.x) > 0.5f || std::abs(viewport.width - width) > 0.5f ||
        std::abs(std::min(viewport.y, viewport.y + viewport.height)) > 0.5f ||
        std::abs(std::abs(viewport.height) - height) > 0.5f) {
        return finish(false);
    }
    const f32 depth_offset = scene->depth_minus_one ? (viewport.znear + viewport.zfar) * 0.5f
                                                     : viewport.znear;
    const f32 depth_scale = scene->depth_minus_one ? (viewport.zfar - viewport.znear) * 0.5f
                                                    : viewport.zfar - viewport.znear;
    const Ocarina::Matrix screen = Ocarina::ScreenMatrix(
        Ocarina::ViewportMatrix(viewport.x, viewport.y, viewport.width, viewport.height, 1.0f,
                                depth_offset, depth_scale),
        scene->view_projection);
    const std::array<u32, 2> size{src.width, src.height};
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
    bool done;
    {
        std::scoped_lock lock{texture_cache.mutex};
        done = texture_cache.CameraMotionUpscale(
            dst, src, scene->depth_info, scene->depth,
            {static_cast<u32>(width), static_cast<u32>(height)}, *reprojection, {}, reset);
    }
    if (motion.evaluations < 4 || (!done && motion.evaluations < 64)) {
        LOG_INFO(Render_Metal,
                 "Ocarina of Time anti-aliasing: MetalFX camera-motion upscale {} ({}x{} -> {}x{}, "
                 "{} draws, reset {})",
                 done ? "applied" : "unavailable; the game's own runs", src.width, src.height,
                 dst.width, dst.height, scene->draws, reset);
    }
    ++motion.evaluations;
    if (!done) {
        return finish(false);
    }
    motion.previous_screen = screen;
    motion.previous_depth = scene->depth;
    motion.previous_size = size;
    motion.previous_rescaled = scene->rescaled;
    motion.last_evaluation = now;
    return finish(true);
}

void RasterizerMetal::TrackOcarinaDepth() {
    if (!OcarinaTemporalEnabled()) {
        return;
    }
    constexpr u32 depth_handle_offset = 0x128;
    const auto& qmd = kepler_compute->launch_description;
    const auto& cbuf = qmd.const_buffer_config[0];
    if (!(qmd.const_buffer_enable_mask & 1) || cbuf.size < depth_handle_offset + sizeof(u32) ||
        !gpu_memory->IsFullyMappedRange(cbuf.Address(), depth_handle_offset + sizeof(u32))) {
        return;
    }
    const u32 index = gpu_memory->Read<u32>(cbuf.Address() + depth_handle_offset) & 0xfffff;
    const auto tic = ReadComputeTic(index);
    const bool scene_size = oot_motion.camera_frame != oot_frame ||
                            (tic && oot_motion.SceneSized(tic->Width(), tic->Height()));
    if (tic && tic->Address() && scene_size) {
        oot_motion.depth_channel = gpu_memory;
        oot_motion.depth_frame = oot_frame;
        oot_motion.depth_tic = index;
        oot_motion.depth_address = tic->Address();
    }
}

std::optional<Tegra::Texture::TICEntry> RasterizerMetal::ReadComputeTic(u32 index) {
    const auto& table = kepler_compute->regs.tic;
    using Entry = Tegra::Texture::TICEntry;
    const GPUVAddr address = table.Address() + u64{index} * sizeof(Entry);
    if (index > table.limit || !gpu_memory->IsFullyMappedRange(address, sizeof(Entry))) {
        return std::nullopt;
    }
    Entry entry{};
    gpu_memory->ReadBlockUnsafe(address, &entry, sizeof(entry));
    return entry;
}

bool RasterizerMetal::TryOcarinaTemporal(u64 shader_hash,
                                         std::span<const VideoCommon::ImageViewInOut> views) {
    // Logs why the game's own upscale draws when the reason changes and every 120th repeat.
    const auto fail = [this](const char* reason) {
        CountOcarinaTemporal(false, reason);
        if (reason != oot_decline || ++oot_decline_repeats % 120 == 0) {
            LOG_INFO(Render_Metal, "Ocarina MetalFX: the game's upscale draws frame {} ({})",
                     oot_frame, reason);
        }
        if (reason != oot_decline) {
            oot_decline = reason;
            oot_decline_repeats = 0;
        }
        return false;
    };
    if (oot_motion.camera_frame != oot_frame) {
        return fail("no scene camera this frame");
    }
    if (oot_motion.depth_frame != oot_frame) {
        return fail("no scene depth this frame");
    }
    if (oot_motion.camera_channel != gpu_memory || oot_motion.depth_channel != gpu_memory) {
        return fail("camera and depth from another channel");
    }
    const auto tic = ReadComputeTic(oot_motion.depth_tic);
    if (!tic || tic->Address() != oot_motion.depth_address ||
        !oot_motion.SceneSized(tic->Width(), tic->Height())) {
        return fail("scene depth descriptor changed");
    }
    std::array depth_views{VideoCommon::ImageViewInOut{.index = oot_motion.depth_tic}};
    texture_cache.FillImageViews(depth_views, true);
    if (!depth_views[0].id || views.size() < 2 || !views[0].id || !views[1].id) {
        return fail("no scene depth image");
    }
    using VideoCore::Surface::PixelFormat;
    ImageView& depth = texture_cache.GetImageView(depth_views[0].id);
    if (texture_cache.IsRescaling(depth) || depth.format != PixelFormat::D32_FLOAT ||
        depth.range != VideoCommon::SubresourceRange{} || depth.Samples() != 1) {
        return fail("unsupported scene depth");
    }
    ImageView& color = texture_cache.GetImageView(views[0].id);
    ImageView& target = texture_cache.GetImageView(views[1].id);
    const auto extent = oot_motion.extent;
    bool upscale_4k = OcarinaTemporal::IsScaledPass(shader_hash);
    bool handheld_scaled = false;
    const bool sixteen_by_nine =
        std::abs(static_cast<int>(extent[0] * 9) - static_cast<int>(extent[1] * 16)) <= 32;
    // Handheld the scaled passes write 1920x1080 from the lower scales (50 %: 960x540, 2x).
    if (upscale_4k && OcarinaTemporal::HandheldEnabled() && target.size.width == 1920 &&
        target.size.height == 1080 && shader_hash != OcarinaTemporal::Ocarina1440pShader &&
        extent[0] < 1920 && sixteen_by_nine) {
        upscale_4k = false;
        handheld_scaled = true;
    }
    // 4K: any 16:9 scene up to 2560x1440; the 3/4 pass: a third larger.
    const bool valid_extents =
        handheld_scaled ? true
        : upscale_4k    ? extent[0] <= 2560 && sixteen_by_nine && target.size.width == 2560 &&
                           target.size.height == 1440
                        : target.size.width == extent[0] * 4 / 3 &&
                           target.size.height == extent[1] * 4 / 3;
    if (!valid_extents || texture_cache.IsRescaling(color) || texture_cache.IsRescaling(target) ||
        color.Samples() != 1 || target.Samples() != 1 ||
        color.range != VideoCommon::SubresourceRange{} ||
        target.range != VideoCommon::SubresourceRange{} ||
        color.format != PixelFormat::E5B9G9R9_FLOAT ||
        target.format != PixelFormat::B10G11R11_FLOAT ||
        !oot_motion.SceneSized(color.size.width, color.size.height) ||
        color.ImageHandle() == target.ImageHandle() || depth.ImageHandle() == target.ImageHandle()) {
        return fail("unexpected upscale images");
    }
    // Without Link's pose the motion comes from the camera alone.
    OcarinaActorMesh actor{};
    const bool has_actor = OcarinaTemporal::ActorEnabled() && ReadOcarinaActor(actor);
    const char* reason = nullptr;
    if (!ocarina_temporal.Record(
            {
                .color = color.Handle(Shader::TextureType::Color2D),
                .depth = depth.Handle(Shader::TextureType::Color2D),
                .target = target.Handle(Shader::TextureType::Color2D),
                .extent = extent,
                .target_size = {target.size.width, target.size.height},
                .current = oot_motion.current,
                .previous = oot_motion.previous,
                .frame = oot_frame,
                .actor = has_actor ? &actor : nullptr,
                .jitter = oot_jitter_frame == oot_frame ? oot_jitter_offset
                                                        : std::array<f32, 2>{},
            },
            reason)) {
        return fail(reason != nullptr ? reason : "MetalFX declined the frame");
    }
    oot_temporal_frame = oot_frame;
    oot_jitter_phases = VideoCommon::OcarinaJitterPhases(extent[0], target.size.width);
    CountOcarinaTemporal(true);
    oot_decline = nullptr;
    return true;
}

void RasterizerMetal::CountOcarinaTemporal(bool replaced, const char* reason) {
    ++(replaced ? oot_replaced : oot_declined);
    if (reason != nullptr) {
        for (auto& [name, count] : oot_decline_reasons) {
            if (name == nullptr || name == reason) {
                name = reason;
                ++count;
                break;
            }
        }
    }
    if (oot_replaced + oot_declined < 600) {
        return;
    }
    std::string reasons;
    for (const auto& [name, count] : oot_decline_reasons) {
        if (name != nullptr) {
            reasons += fmt::format("{}{} {}", reasons.empty() ? "; declined: " : ", ", name, count);
        }
    }
    const auto [actor_frames, effects_frames] = ocarina_temporal.TakeExtras();
    LOG_INFO(Render_Metal,
             "Ocarina MetalFX: {} of {} upscale passes replaced; scene {}x{}; Link's motion {}, "
             "effects mask {}{}",
             oot_replaced, oot_replaced + oot_declined, oot_motion.extent[0],
             oot_motion.extent[1], actor_frames, effects_frames, reasons);
    oot_replaced = oot_declined = 0;
    oot_decline_reasons = {};
}

void RasterizerMetal::TrackOcarinaActor(u64 vertex_hash, bool is_indexed) {
    if (vertex_hash != OcarinaTemporal::ActorVertexShader || !is_indexed ||
        !OcarinaTemporal::ActorEnabled() || !OcarinaTemporalEnabled() ||
        oot_actor_poses[oot_actor_pose_index].frame == oot_frame) {
        return;
    }
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const auto& regs = maxwell3d->regs;
    const auto& stream = regs.vertex_streams[0];
    if (draw.index_buffer.count != 9702 || draw.index_buffer.FormatSizeInBytes() != 2 ||
        !stream.IsEnabled() || stream.stride != OcarinaActorMesh::Stride ||
        regs.vertex_stream_instances.IsInstancingEnabled(0)) {
        return;
    }
    // The layout ocarina_actor_motion.vert reads (Vulkan compares the Vulkan formats).
    using Attribute = Maxwell::VertexAttribute;
    const auto attribute = [&](u32 location, Attribute::Type type, Attribute::Size size,
                               Attribute::Size same_size, u32 offset) {
        const auto& attrib = regs.vertex_attrib_format[location];
        const Attribute::Size attrib_size = attrib.size.Value();
        return attrib.buffer == 0 && attrib.constant == 0 && attrib.offset == offset &&
               attrib.type.Value() == type && (attrib_size == size || attrib_size == same_size);
    };
    constexpr Attribute::Size rgb32 = Attribute::Size::Size_R32_G32_B32;
    constexpr Attribute::Size rgba8 = Attribute::Size::Size_R8_G8_B8_A8;
    constexpr Attribute::Size xbgr8 = Attribute::Size::Size_X8_B8_G8_R8;
    constexpr Attribute::Size rgba16 = Attribute::Size::Size_R16_G16_B16_A16;
    if (!attribute(0, Attribute::Type::Float, rgb32, rgb32, 0) ||
        !attribute(4, Attribute::Type::UNorm, rgba8, xbgr8, 40) ||
        !attribute(5, Attribute::Type::UNorm, rgba8, xbgr8, 44) ||
        !attribute(6, Attribute::Type::UInt, rgba16, rgba16, 12) ||
        !attribute(7, Attribute::Type::UInt, rgba16, rgba16, 20)) {
        return;
    }
    const auto& cbufs = maxwell3d->state.shader_stages[0].const_buffers;
    if (!cbufs[4].enabled || cbufs[4].size < 0x40 + 3 * sizeof(f32) || !cbufs[6].enabled ||
        cbufs[6].size != 13568) {
        return;
    }
    const auto origin_dev = gpu_memory->GpuToCpuAddress(cbufs[4].address);
    const auto bones_dev = gpu_memory->GpuToCpuAddress(cbufs[6].address);
    if (!origin_dev || !bones_dev || buffer_cache.IsRegionGpuModified(*origin_dev, cbufs[4].size) ||
        buffer_cache.IsRegionGpuModified(*bones_dev, cbufs[6].size)) {
        return;
    }
    oot_actor_pose_index ^= 1U;
    auto& pose = oot_actor_poses[oot_actor_pose_index];
    pose.frame = oot_frame;
    pose.channel = gpu_memory;
    pose.vertex = stream.Address();
    pose.indices = draw.index_buffer.IndexStart();
    pose.base_vertex = static_cast<s32>(draw.base_index);
    pose.origin = {};
    device_memory.ReadBlockUnsafe(*origin_dev + 0x40, pose.origin.data(), 3 * sizeof(f32));
    device_memory.ReadBlockUnsafe(*bones_dev, pose.bones.data(), sizeof(pose.bones));
}

bool RasterizerMetal::ReadOcarinaActor(OcarinaActorMesh& mesh) {
    const auto& now = oot_actor_poses[oot_actor_pose_index];
    const auto& before = oot_actor_poses[oot_actor_pose_index ^ 1U];
    if (now.frame != oot_frame || before.frame + 1 != now.frame || now.channel != gpu_memory ||
        before.channel != now.channel || now.vertex != before.vertex ||
        now.indices != before.indices || now.base_vertex != before.base_vertex) {
        return false;
    }
    auto& cache = oot_actor_mesh_cache;
    const auto can_stay_cached = [&](GPUVAddr address, size_t size) {
        const auto device_address = gpu_memory->GpuToCpuAddress(address);
        return device_address && !buffer_cache.IsRegionGpuModified(*device_address, size) &&
               !gpu.HasPendingCPUWrites(*device_address, size);
    };
    constexpr size_t index_count = 9702;
    constexpr size_t stride = OcarinaActorMesh::Stride;
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
            return false;
        }
        const auto [minimum, maximum] = std::ranges::minmax(cache.index_data);
        const s64 first = static_cast<s64>(now.base_vertex) + minimum;
        const u32 vertex_count = static_cast<u32>(maximum) - minimum + 1;
        if (first < 0 || first > (1 << 20) || vertex_count > 16384) {
            cache.vertices.clear();
            return false;
        }
        cache.vertices.resize(vertex_count * stride);
        if (!read(now.vertex + first * stride, cache.vertices.data(), cache.vertices.size())) {
            cache.vertices.clear();
            return false;
        }
        for (u16& index : cache.index_data) {
            index = static_cast<u16>(index - minimum);
        }
        cache.channel = now.channel;
        cache.vertex = now.vertex;
        cache.indices = now.indices;
        cache.base_vertex = now.base_vertex;
        ++cache.key;
    }
    mesh = {
        .vertices = cache.vertices,
        .indices = cache.index_data,
        .mesh_key = cache.key,
        .current_origin = now.origin,
        .previous_origin = before.origin,
        .current_bones = now.bones,
        .previous_bones = before.bones,
    };
    return true;
}

void RasterizerMetal::NoteOcarinaEffectsDraw(u64 fragment_hash) {
    if (fragment_hash != OcarinaTemporal::FogFragmentShader || !OcarinaTemporal::EffectsEnabled() ||
        !OcarinaTemporalEnabled() || oot_motion.camera_frame != oot_frame) {
        return;
    }
    const auto scene_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
    if (!scene_id) {
        return;
    }
    ImageView& scene = texture_cache.GetImageView(scene_id);
    if (scene.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT ||
        !oot_motion.SceneSized(scene.size.width, scene.size.height) ||
        texture_cache.IsRescaling(scene) || scene.range != VideoCommon::SubresourceRange{}) {
        return;
    }
    oot_effects_scene = scene.Handle(Shader::TextureType::Color2D);
    oot_effects_address = scene.gpu_addr;
    oot_effects_frame = oot_frame;
    ocarina_temporal.SaveEffectsBase(oot_effects_scene, oot_motion.extent, oot_frame);
}

void RasterizerMetal::MarkOcarinaEffects(const Shader::Info& info,
                                         std::span<const VideoCommon::ImageViewInOut> views) {
    if (oot_effects_frame != oot_frame) {
        return;
    }
    // Views: texel buffers, image buffers, textures, then the images.
    size_t view = 0;
    for (const auto& desc : info.texture_buffer_descriptors) {
        view += desc.count;
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        view += desc.count;
    }
    for (const auto& desc : info.texture_descriptors) {
        view += desc.count;
    }
    for (const auto& desc : info.image_descriptors) {
        for (u32 element = 0; element < desc.count; ++element, ++view) {
            if (desc.is_written && view < views.size() && views[view].id &&
                texture_cache.GetImageView(views[view].id).gpu_addr == oot_effects_address) {
                ocarina_temporal.MarkEffects(oot_effects_scene, oot_frame);
                oot_effects_frame.reset();
                return;
            }
        }
    }
}

bool RasterizerMetal::ProvokingLastWanted(const GraphicsPipeline& pipeline) {
    // Metal has no provoking vertex setting (flat inputs take the first vertex), so the draws of
    // pipelines with flat inputs are rewritten to start each primitive with the vertex Vulkan's
    // last vertex convention takes them from, as Vulkan draws them on a desktop GPU.
    // EDEN_NX2_METAL_PROVOKING_LAST=0: they take the first vertex, as MoltenVK draws them.
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_PROVOKING_LAST");
    return enabled && maxwell3d->regs.provoking_vertex == Maxwell::ProvokingVertex::Last &&
           IndexConvert::RotatesForProvokingVertex(DrawTopology(*maxwell3d)) &&
           pipeline.HasFlatInputs();
}

bool RasterizerMetal::RestartOffWidens(bool is_indexed) {
    // Metal restarts a strip at the all-ones index whatever the guest says. With the restart off
    // that index is a vertex: the 16-bit indices are widened to 32 bits (Convert with the restart
    // off leaves 0xFFFF as it is), where all-ones is 0xFFFFFFFF. A 32-bit 0xFFFFFFFF vertex is
    // not told apart (no buffer Metal could address has that many vertices).
    return is_indexed && maxwell3d->regs.primitive_restart.enabled == 0 &&
           maxwell3d->draw_manager.draw_state.index_buffer.format ==
               Maxwell::IndexFormat::UnsignedShort &&
           IndexConvert::RestartsStrips(DrawTopology(*maxwell3d));
}

bool RasterizerMetal::DrawMayBeSkipped() {
    if (skip_valid) {
        return skip_decision;
    }
    skip_valid = true;
    // EDEN_NX2_ASYNC_DISCARD as Vulkan reads it: 2 (default) skips native draws only into
    // targets cleared every frame, 1 by the legacy heuristic, 0 never.
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_ASYNC_DISCARD");
        return value == nullptr ? 2 : value[0] == '0' ? 0 : value[0] == '1' ? 1 : 2;
    }();
    if (!Settings::values.use_asynchronous_shaders.GetValue()) {
        skip_decision = false;
        return false;
    }
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    bool recurring = true;
    if (mode == 2 && native) {
        recurring = texture_cache.ClassifyDrawTargets().kind ==
                    VideoCommon::DrawTargetClass::Recurring;
    }
    const u32 elements = maxwell3d->draw_manager.draw_state.draw_element_count;
    skip_decision = Vulkan::CanDiscardPendingDraw(native, maxwell3d->regs.zeta_enable != 0,
                                                  elements, elements, mode != 0, recurring);
    return skip_decision;
}

void RasterizerMetal::NoteUnprepared() {
    if (prepare_refusal.empty()) {
        NoteEmptyDraw(prepare_empty_why != nullptr ? prepare_empty_why : "PrepareDraw skipped it without a cause");
    } else {
        RefuseDraw(prepare_refusal);
    }
}

void RasterizerMetal::RefuseDraw(std::string_view cause) {
    // As Vulkan's PrepareDraw: a draw of the native macro engine that was not drawn ends the
    // front's GPU-counted draws.
    if (maxwell3d != nullptr && maxwell3d->GetNativeMacroEngine().native_turing) {
        maxwell3d->GetNativeMacroEngine().native_execution_failed = true;
    }
    if (frame_dump.Capturing()) {
        const auto& regs = maxwell3d->regs;
        const auto& hashes = pipeline_cache.LookupHashes();
        frame_dump.NoteSkipped(false, hashes[5] != 0 ? hashes[5] : hashes[1], regs.rt[0].Address(),
                               cause);
    }
    Coverage::Refuse(Coverage::Work::Draw, cause, [this] { return DescribeDraw(); });
}

std::string RasterizerMetal::DescribeDraw() const {
    const auto& regs = maxwell3d->regs;
    const auto& hashes = pipeline_cache.LookupHashes();
    std::string text = "programs";
    for (const u64 hash : hashes) {
        text += fmt::format(" {:016x}", hash);
    }
    text += fmt::format(", topology {}, colour formats",
                        static_cast<u32>(maxwell3d->draw_manager.draw_state.topology));
    for (u32 rt = 0; rt < regs.rt_control.count && rt < Maxwell::NumRenderTargets; ++rt) {
        text += fmt::format(" {:#x}", static_cast<u32>(regs.rt[rt].format));
    }
    text += fmt::format(", depth format {:#x}",
                        regs.zeta_enable != 0 ? static_cast<u32>(regs.zeta.format) : 0U);
    return text;
}

void RasterizerMetal::Clear(u32 layer_count) {
    if (!maxwell3d) {
        return;
    }
    gpu_memory->FlushCaching();
    FlushWork();
    if (!ConditionHoldsOnCpu()) {
        // A clear under a condition the GPU decides is decided on the CPU.
        return;
    }
    const auto& regs = maxwell3d->regs;
    const bool any_color = regs.clear_surface.R || regs.clear_surface.G ||
                           regs.clear_surface.B || regs.clear_surface.A;
    const bool all_color = regs.clear_surface.R && regs.clear_surface.G &&
                           regs.clear_surface.B && regs.clear_surface.A;
    if (!any_color && !regs.clear_surface.Z && !regs.clear_surface.S) {
        return;
    }
    const auto describe = [&] {
        return fmt::format("surface {:#x}, scissor {}, layers {} from {}, stencil mask {:#x}",
                           regs.clear_surface.raw, regs.clear_control.use_scissor != 0,
                           layer_count, static_cast<u32>(regs.clear_surface.layer),
                           regs.stencil_front_mask);
    };
    const bool stencil_partial = regs.clear_surface.S && regs.stencil_front_mask != 0xFF &&
                                 regs.stencil_front_mask != 0;
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (framebuffer == nullptr) {
        Coverage::Refuse(Coverage::Work::Clear, "no framebuffer", describe);
        return;
    }
    const u32 rt = regs.clear_surface.RT;
    const bool use_color =
        any_color && rt < NUM_RT && framebuffer->Colors()[rt].texture != nullptr;
    const bool use_depth = regs.clear_surface.Z && framebuffer->HasDepth();
    const bool use_stencil = regs.clear_surface.S && framebuffer->HasStencil();
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }
    const std::array<u32, 4> rect = ClearArea(regs, texture_cache.IsRescaling(),
                                              framebuffer->Width(), framebuffer->Height());
    if (rect[2] == 0 || rect[3] == 0) {
        return;
    }
    const bool covers = rect[0] == 0 && rect[1] == 0 && rect[2] >= framebuffer->Width() &&
                        rect[3] >= framebuffer->Height();
    const bool drawn_clears = ClearDrawsEnabled();
    // What a load action can't do (a rectangle, some colour channels, some stencil bits) is a
    // draw; EDEN_NX2_METAL_CLEAR_DRAWS=0 clears the whole of it instead and counts that.
    const bool draw_color = use_color && (!covers || !all_color);
    const bool draw_depth = use_depth && !covers;
    const bool draw_stencil = use_stencil && (!covers || stencil_partial);
    const bool needs_draw = draw_color || draw_depth || draw_stencil;
    std::array<f64, 4> color_value{};
    if (use_color) {
        const auto format =
            VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[rt].format);
        const bool is_integer = VideoCore::Surface::IsPixelFormatInteger(format);
        const bool is_signed = is_integer && VideoCore::Surface::IsPixelFormatSignedInteger(format);
        for (size_t component = 0; component < 4; ++component) {
            const f32 value = regs.clear_color[component];
            if (!is_integer) {
                color_value[component] = value;
            } else if (!is_signed) {
                color_value[component] = static_cast<f64>(std::bit_cast<u32>(value));
            } else {
                color_value[component] = static_cast<f64>(std::bit_cast<s32>(value));
            }
        }
    }
    if (!drawn_clears && needs_draw) {
        Coverage::Refuse(Coverage::Work::Clear,
                         !covers ? "clear of part of the target (scissor), whole target cleared"
                                 : draw_color ? "clear of some colour channels"
                                              : "clear of some stencil bits",
                         describe);
    }
    ClearPass pass{};
    pass.width = framebuffer->Width();
    pass.height = framebuffer->Height();
    bool whole = false;
    if (use_color && all_color && (!draw_color || !drawn_clears)) {
        pass.color[rt] = framebuffer->Colors()[rt];
        pass.clear_color[rt] = true;
        pass.color_value[rt] = color_value;
        whole = true;
    }
    if (use_depth && (!draw_depth || !drawn_clears)) {
        pass.depth = framebuffer->Depth();
        pass.clear_depth = true;
        pass.depth_value = regs.clear_depth;
        whole = true;
    }
    if (use_stencil && (!draw_stencil || !drawn_clears)) {
        pass.stencil = framebuffer->Depth();
        pass.clear_stencil = true;
        pass.stencil_value = static_cast<u32>(regs.clear_stencil);
        whole = true;
    }
    // The layers the clear names (clear_surface.layer, layer_count) of the targets' views, one
    // pass or draw each; EDEN_NX2_METAL_CLEAR_LAYERS=0: the bound layer only, as before.
    static const bool clear_layers = EnvironmentOn("EDEN_NX2_METAL_CLEAR_LAYERS");
    const u32 first_layer = clear_layers ? static_cast<u32>(regs.clear_surface.layer) : 0U;
    const u32 layers = clear_layers ? (std::max)(layer_count, 1U) : 1U;
    if (first_layer != 0 || layers > 1) {
        const u32 available = (std::max)(
            use_color ? framebuffer->ColorLayers()[rt] : 0U,
            use_depth || use_stencil ? framebuffer->DepthLayers() : 0U);
        // A view's layers follow its slice (a 3D slice target has one layer: only layer 0).
        const auto at_layer = [](Attachment attachment, u32 layer) {
            if (attachment.texture != nullptr) {
                attachment.slice += layer;
            }
            return attachment;
        };
        for (u32 layer = first_layer; layer < first_layer + layers && layer < available; ++layer) {
            ClearPass layer_pass = pass;
            for (Attachment& color : layer_pass.color) {
                color = at_layer(color, layer);
            }
            layer_pass.depth = at_layer(layer_pass.depth, layer);
            layer_pass.stencil = at_layer(layer_pass.stencil, layer);
            if (whole) {
                scheduler.Clear(layer_pass);
            }
            if (drawn_clears && needs_draw) {
                ClearRect draw{};
                if (draw_color) {
                    draw.color = at_layer(framebuffer->Colors()[rt], layer);
                    draw.color_mask = static_cast<u8>(
                        regs.clear_surface.R | regs.clear_surface.G << 1 |
                        regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
                    draw.color_value = color_value;
                }
                if (draw_depth || draw_stencil) {
                    draw.depth = at_layer(framebuffer->Depth(), layer);
                    draw.clear_depth = draw_depth;
                    draw.clear_stencil = draw_stencil;
                    draw.depth_value = regs.clear_depth;
                    draw.stencil_value = static_cast<u32>(regs.clear_stencil);
                    draw.stencil_mask = stencil_partial ? regs.stencil_front_mask : 0xFFU;
                }
                draw.rect = rect;
                draw.width = framebuffer->Width();
                draw.height = framebuffer->Height();
                static_cast<void>(scheduler.ClearRectangle(draw));
            }
        }
        Coverage::Note(Coverage::Work::Clear, Coverage::Path::ClearDraw);
        ++clears;
        return;
    }
    if (whole) {
        scheduler.Clear(pass);
    }
    if (drawn_clears && needs_draw) {
        ClearRect draw{};
        if (draw_color) {
            draw.color = framebuffer->Colors()[rt];
            draw.color_mask = static_cast<u8>(regs.clear_surface.R | regs.clear_surface.G << 1 |
                                              regs.clear_surface.B << 2 |
                                              regs.clear_surface.A << 3);
            draw.color_value = color_value;
        }
        if (draw_depth || draw_stencil) {
            draw.depth = framebuffer->Depth();
            draw.clear_depth = draw_depth;
            draw.clear_stencil = draw_stencil;
            draw.depth_value = regs.clear_depth;
            draw.stencil_value = static_cast<u32>(regs.clear_stencil);
            draw.stencil_mask = stencil_partial ? regs.stencil_front_mask : 0xFFU;
        }
        draw.rect = rect;
        draw.width = framebuffer->Width();
        draw.height = framebuffer->Height();
        if (scheduler.ClearRectangle(draw)) {
            Coverage::Note(Coverage::Work::Clear, Coverage::Path::ClearDraw);
        } else {
            Coverage::Refuse(Coverage::Work::Clear, "no pipeline draws this clear", describe);
        }
    } else if (!needs_draw) {
        Coverage::Note(Coverage::Work::Clear, Coverage::Path::Native);
    }
    ++clears;
}

namespace {

/// The host memory ranges under guest memory [address, address + size), merged where they join
/// (Vulkan's ByteAddressedMappedRanges).
std::vector<std::pair<DAddr, u64>> MappedRanges(Tegra::MemoryManager& memory, GPUVAddr address,
                                                u64 size) {
    if (size == 0 || address > (std::numeric_limits<u64>::max)() - size ||
        !memory.IsWithinGPUAddressRange(address + size - 1)) {
        throw Shader::RuntimeError("Unknown packed surface access range {:#x}+{:#x}", address,
                                   size);
    }
    std::vector<std::pair<DAddr, u64>> ranges;
    while (size != 0) {
        const u64 count = (std::min)(size, 0x1000 - (address & 0xfff));
        const auto mapped = memory.GpuToCpuAddress(address);
        if (!mapped || *mapped > (std::numeric_limits<u64>::max)() - count) {
            throw Shader::RuntimeError("Unmapped packed surface access range at {:#x}", address);
        }
        if (!ranges.empty() && ranges.back().first + ranges.back().second == *mapped) {
            ranges.back().second += count;
        } else {
            ranges.emplace_back(*mapped, count);
        }
        address += count;
        size -= count;
    }
    return ranges;
}

bool RangesOverlap(const std::pair<DAddr, u64>& a, const std::pair<DAddr, u64>& b) {
    return a.second != 0 && b.second != 0 &&
           (a.first < b.first ? b.first - a.first < a.second : a.first - b.first < b.second);
}

} // Anonymous namespace

void RasterizerMetal::DispatchCompute() {
    ++dispatches;
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    gpu_memory->FlushCaching();
    FlushWork();
    ComputePipeline* const pipeline = pipeline_cache.CurrentComputePipeline();
    if (pipeline != nullptr && VideoCommon::FeatureCensus::Enabled()) {
        VideoCommon::FeatureCensus::NoteDispatch(
            pipeline->Key().unique_hash, kepler_compute->GetIndirectComputeAddress().has_value());
    }
    void* const state = pipeline != nullptr ? pipeline->State(device) : nullptr;
    if (state == nullptr) {
        Coverage::Refuse(Coverage::Work::Dispatch,
                         pipeline == nullptr ? pipeline_cache.Refusal()
                                             : std::string_view{"compute kernel not built by Metal"},
                         [&] {
                             const auto& qmd = kepler_compute->launch_description;
                             return fmt::format(
                                 "program {:016x}, grid {}x{}x{}",
                                 pipeline != nullptr ? pipeline->Key().unique_hash : u64{0},
                                 qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                 qmd.grid_dim_z.Value());
                         });
        if (frame_dump.Capturing()) {
            frame_dump.NoteSkipped(true, pipeline != nullptr ? pipeline->Key().unique_hash : 0,
                                   0, pipeline == nullptr ? pipeline_cache.Refusal()
                                                          : std::string_view{"kernel not built"});
        }
        // As OpenGL and Vulkan report it: KeplerCompute::DispatchNativeCompute takes a silent
        // return for a successful dispatch.
        if (kepler_compute->IsNativeMode()) {
            throw Shader::RuntimeError("Native compute pipeline unavailable");
        }
        return;
    }
    ++dispatches_with_pipeline;
    static const bool encode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DISPATCH");
        return value == nullptr || value[0] != '0';
    }();
    if (!encode) {
        Coverage::Refuse(Coverage::Work::Dispatch, "EDEN_NX2_METAL_DISPATCH=0");
        return;
    }
    // Ocarina's anti-aliasing through the camera-motion upscale (EDEN_NX2_METAL_OOT_CAMERA_MOTION),
    // as Vulkan's DispatchCompute does with NVIDIA DLSS. The hash is the program's lowered one.
    if (kepler_compute->IsNativeMode() && CameraMotionUpscaleEnabled() &&
        kepler_compute->ProcessOcarinaUpscale(Settings::GetCurrentProgramID(), 0,
                                              pipeline->Key().unique_hash)) {
        Coverage::Note(Coverage::Work::Dispatch, Coverage::Path::Replaced);
        return;
    }
    StageProgram& program = pipeline->Program();
    const Shader::Info& info = program.info;
    const auto& qmd = kepler_compute->launch_description;
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Command list pages the dispatch writes are copied after it, for the split's front to read
    // without a full sync (EDEN_NX2_METAL_LIST_COPIES, as Vulkan's ReadGpuWrittenCommandList).
    const bool list_copies = ListCopiesEnabled();
    buffer_cache.Nx2CollectListCopies(list_copies);
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
        if (list_copies) {
            SubmitListCopies(pipeline->Key().unique_hash);
            buffer_cache.Nx2CollectListCopies(false);
        }
    };

    VideoCommon::ComputeUniformBufferSizes uniform_sizes{};
    std::ranges::copy_n(info.constant_buffer_used_sizes.begin(), uniform_sizes.size(),
                        uniform_sizes.begin());
    buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &uniform_sizes);
    buffer_cache.UnbindComputeStorageBuffers();
    size_t ssbo_index = 0;
    for (const auto& desc : info.storage_buffers_descriptors) {
        buffer_cache.BindComputeStorageBuffer(ssbo_index++, desc.cbuf_index, desc.cbuf_offset,
                                              desc.is_written, desc.indirect_offset,
                                              desc.write_extent);
    }
    texture_cache.SynchronizeDescriptors(true);

    const auto& cbufs = qmd.const_buffer_config;
    const bool via_header_index = qmd.linked_tsc != 0;
    ScratchList scratch_views{compute_scratch.views};
    ScratchList scratch_samplers{compute_scratch.samplers};
    ScratchList scratch_sampler_handles{compute_scratch.sampler_handles};
    ScratchList scratch_batch_handles{compute_scratch.batch_handles};
    std::vector<VideoCommon::ImageViewInOut>& views = scratch_views.value;
    std::vector<VideoCommon::SamplerId>& samplers = scratch_samplers.value;
    const bool use_bind_memo = BindMemo::Enabled();
    std::vector<u32>& sampler_handles = scratch_sampler_handles.value;
    std::vector<u32>& batch_handles = scratch_batch_handles.value;
    DescriptorPageCache page_cache{};
    CollectStageDescriptors(
        info, true, via_header_index, use_bind_memo,
        [&](u32 cbuf, bool check_enabled) -> GPUVAddr {
            return cbuf < cbufs.size() &&
                           (!check_enabled || ((qmd.const_buffer_enable_mask >> cbuf) & 1) != 0)
                       ? cbufs[cbuf].Address()
                       : GPUVAddr{};
        },
        page_cache, batch_handles, views, sampler_handles, samplers);
    if (use_bind_memo) {
        static_assert(BindMemo::Kind::ComputeBit == TextureCache::BINDING_KIND_COMPUTE);
        if (program.binding_kinds.empty() && !views.empty()) {
            BindMemo::AppendKinds(info, BindMemo::Kind::ComputeBit, program.binding_kinds);
        }
    }
    ResolveStageBindings(true, false,
                         use_bind_memo ? std::span<const u32>{program.binding_kinds}
                                       : std::span<const u32>{},
                         views, sampler_handles, samplers);
    MakePlaceholders();
    PrepareSampleGrids(info, views);
    if (pipeline->Key().unique_hash == OcarinaTemporal::DepthReader) {
        TrackOcarinaDepth();
    }
    if (!kepler_compute->GetIndirectComputeAddress()) {
        MarkOcarinaEffects(info, views);
    }
    // The upscale pass through MetalFX temporal upscaling when it is the chosen upscaler, as
    // Vulkan's TryOcarinaUpscale; declined, the bilinear pass or the game's own runs.
    if (!kepler_compute->GetIndirectComputeAddress() &&
        OcarinaTemporal::IsTemporalPass(pipeline->Key().unique_hash, info) &&
        OcarinaTemporalEnabled()) {
        if (TryOcarinaTemporal(pipeline->Key().unique_hash, views)) {
            const ImageView& color = texture_cache.GetImageView(views[0].id);
            const ImageView& target = texture_cache.GetImageView(views[1].id);
            VideoCore::Probe::StoreScene(color.size.width, color.size.height, target.size.width,
                                         target.size.height, "MetalFX");
            texture_cache.MarkModification(target.image_id);
            Coverage::Note(Coverage::Work::Dispatch, Coverage::Path::Replaced);
            return;
        }
    }
    // EDEN_NX2_OOT_BILINEAR_UPSCALE: Ocarina's upscale dispatch as one bilinear tap, as Vulkan's
    // TryOcarinaBilinear takes it (views: the sampled scene, then the written target).
    if (OcarinaUpscale::BilinearEnabled() &&
        Settings::GetCurrentProgramID() == VideoCommon::OotNetwork::ProgramId &&
        !kepler_compute->GetIndirectComputeAddress() &&
        OcarinaUpscale::IsBilinearPass(pipeline->Key().unique_hash, info)) {
        using VideoCore::Surface::PixelFormat;
        ImageView& color = texture_cache.GetImageView(views[0].id);
        ImageView& target = texture_cache.GetImageView(views[1].id);
        if (color.format == PixelFormat::E5B9G9R9_FLOAT &&
            target.format == PixelFormat::B10G11R11_FLOAT && color.Samples() <= 1 &&
            target.Samples() <= 1 && !texture_cache.IsRescaling(color) &&
            !texture_cache.IsRescaling(target) && color.range == VideoCommon::SubresourceRange{} &&
            target.range == VideoCommon::SubresourceRange{} &&
            target.size.width > color.size.width && target.size.height > color.size.height &&
            color.ImageHandle() != target.ImageHandle() &&
            ocarina.Bilinear(color.Handle(Shader::TextureType::Color2D),
                             {color.size.width, color.size.height},
                             {color.size.width, color.size.height},
                             target.Handle(Shader::TextureType::Color2D),
                             {target.size.width, target.size.height})) {
            VideoCore::Probe::StoreScene(color.size.width, color.size.height, target.size.width,
                                         target.size.height, "Bilinear");
            texture_cache.MarkModification(target.image_id);
            Coverage::Note(Coverage::Work::Dispatch, Coverage::Path::Replaced);
            return;
        }
    }

    buffer_cache.UnbindComputeTextureBuffers();
    size_t view = 0;
    size_t tbo_index = 0;
    const auto add_buffer = [&](const auto& desc, bool is_image) {
        for (u32 element = 0; element < desc.count; ++element) {
            ImageView& image_view = texture_cache.GetImageView(views[view++].id);
            bool is_written = false;
            if constexpr (requires { desc.is_written; }) {
                is_written = desc.is_written;
            }
            buffer_cache.BindComputeTextureBuffer(tbo_index++, image_view.GpuAddr(),
                                                  image_view.BufferSize(), image_view.format,
                                                  is_written, is_image);
        }
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_buffer(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_buffer(desc, true);
    }
    const u64 dispatch_segment = scheduler.SegmentSerial();
    buffer_cache.UpdateComputeBuffers();
    buffer_cache_runtime.PendingBindings().clear();
    buffer_cache.BindHostComputeBuffers();
    ScratchList scratch_bindings{compute_scratch.bindings};
    std::vector<Binding>& bindings = scratch_bindings.value;
    const auto& pending_bindings = buffer_cache_runtime.PendingBindings();
    bindings.assign(pending_bindings.begin(), pending_bindings.end());
    std::vector<TextureHandle> texel_views;
    if (!info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty()) {
        texel_views = PrepareTexelBuffers(bindings, info);
    }

    rescaling_words = {};
    rescaling_words[6] = std::bit_cast<u32>(1.0f);
    u32 texture_bit = 0;
    u32 image_bit = 0;
    AddRescalingBits(info, views, texture_bit, image_bit);
    // The group counts' buffer first: bringing it up to date may end the open encoder.
    const auto indirect = kepler_compute->GetIndirectComputeAddress();
    BufferHandle indirect_buffer{};
    u64 indirect_offset{};
    if (indirect) {
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(
            *indirect, 12, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
            VideoCommon::ObtainBufferOperation::DoNothing);
        indirect_buffer = buffer->Handle();
        indirect_offset = offset;
    }
    if (scheduler.SegmentSerial() != dispatch_segment) {
        // Committed in the middle of the dispatch: what the buffer cache noted as used is gone.
        scheduler.BlockAhead();
    }
    // G19: a program that addresses a packed R16 surface bytewise declares it as an R32 image of
    // half the width. Vulkan's shadow: the texels go through a buffer into an R32Uint texture
    // the program binds, and back after the dispatch.
    Object shadow_texture;
    Object shadow_bridge;
    TextureHandle shadow_original{};
    u32 shadow_width = 0;
    u32 shadow_height = 0;
    byte_shadow_view = ~size_t{0};
    byte_shadow_texture = nullptr;
    {
        size_t image_index = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                             Shader::NumDescriptors(info.image_buffer_descriptors) +
                             Shader::NumDescriptors(info.texture_descriptors);
        for (const auto& desc : info.image_descriptors) {
            for (u32 element = 0; element < desc.count; ++element, ++image_index) {
                if (!desc.is_byte_addressed ||
                    texture_cache.GetImageView(views[image_index].id).format !=
                        VideoCore::Surface::PixelFormat::R16_FLOAT) {
                    continue;
                }
                if (byte_shadow_view != ~size_t{0} || !kepler_compute->IsNativeMode() ||
                    indirect || desc.count != 1 || desc.is_read || !desc.is_written ||
                    desc.type != Shader::TextureType::Color2D ||
                    desc.format != Shader::ImageFormat::R32_UINT || info.uses_global_memory) {
                    throw Shader::RuntimeError(
                        "Unsupported packed R16 surface dispatch or unbounded global access");
                }
                byte_shadow_view = image_index;
            }
        }
    }
    if (byte_shadow_view != ~size_t{0}) {
        const ImageView& target_view = texture_cache.GetImageView(views[byte_shadow_view].id);
        const Image* const target = target_view.BackingImage();
        const VideoCommon::Extent3D size = target_view.size;
        if (target == nullptr || size.width < 2 || (size.width & 1) != 0 || size.height == 0 ||
            size.depth != 1 || target_view.Samples() != 1 ||
            texture_cache.IsRescaling(target_view) || target->info.type != VideoCommon::ImageType::e2D ||
            target->info.resources.levels != 1 || target->info.resources.layers != 1 ||
            target->info.size != size || target->info.format != target_view.format ||
            target_view.ImageHandle() != target->Handle() || target->Handle() == nullptr ||
            qmd.grid_dim_x == 0 || qmd.grid_dim_y == 0 || qmd.grid_dim_z == 0) {
            throw Shader::RuntimeError("Unsupported packed R16 byte-addressed image binding");
        }
        // The shadow replaces the surface for the dispatch: no other binding may alias it.
        const auto target_ranges =
            MappedRanges(*gpu_memory, target->gpu_addr, target->guest_size_bytes);
        for (size_t index = 0; index < views.size(); ++index) {
            if (index == byte_shadow_view) {
                continue;
            }
            const ImageView& peer = texture_cache.GetImageView(views[index].id);
            if (peer.IsBuffer()) {
                continue;
            }
            const Image* const image = peer.BackingImage();
            if (image == nullptr) {
                throw Shader::RuntimeError("Unknown packed R16 surface peer image at binding {}",
                                           index);
            }
            for (const auto& a : target_ranges) {
                for (const auto& b :
                     MappedRanges(*gpu_memory, image->gpu_addr, image->guest_size_bytes)) {
                    if (RangesOverlap(a, b)) {
                        throw Shader::RuntimeError(
                            "Aliased packed R16 surface peer image at binding {}", index);
                    }
                }
            }
        }
        shadow_width = size.width / 2;
        shadow_height = size.height;
        shadow_texture = CreateTexture(device, {.kind = TextureKind::e2D,
                                                .format = Fmt::R32Uint,
                                                .width = shadow_width,
                                                .height = shadow_height,
                                                .shader_write = true,
                                                .pixel_format_view = false});
        shadow_bridge = CreateBuffer(device, u64{size.width} * size.height * 2, false, nullptr);
        if (!shadow_texture || !shadow_bridge) {
            throw Shader::RuntimeError("Packed R16 surface shadow not made");
        }
        shadow_original = target->Handle();
        const u32 bytes_per_row = size.width * 2;
        scheduler.CopyTextureToBuffer(shadow_original, 0, 0, {}, {size.width, size.height, 1},
                                      shadow_bridge.Get(), 0, bytes_per_row, 0);
        scheduler.BreakBlits();
        scheduler.CopyBufferToTexture(shadow_bridge.Get(), 0, bytes_per_row, 0,
                                      shadow_texture.Get(), 0, 0, {},
                                      {shadow_width, shadow_height, 1});
        scheduler.BreakBlits();
        byte_shadow_texture = shadow_texture.Get();
    }
    void* dispatch_state = state;
    if (program.reduction_variants && ReductionVariantsEnabled()) {
        if (!NeedsReducingVariant(program, views, samplers)) {
            ++reduction_counts.plain_dispatches;
        } else if (void* const reducing_state = pipeline->ReducingState(device)) {
            dispatch_state = reducing_state;
            ++reduction_counts.reducing_dispatches;
        } else {
            ++reduction_counts.plain_refused_dispatches;
        }
    }
    scheduler.BeginCompute(dispatch_state);
    BindStageResources(program, ShaderStage::Compute, bindings, texel_views, views, samplers, 0,
                       0);
    byte_shadow_view = ~size_t{0};
    byte_shadow_texture = nullptr;
    const std::array<u32, 3> group_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z};
    if (indirect) {
        scheduler.DispatchIndirect(indirect_buffer, indirect_offset, group_size);
    } else {
        scheduler.Dispatch({qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z}, group_size);
    }
    if (shadow_texture) {
        const u32 bytes_per_row = shadow_width * 4;
        scheduler.CopyTextureToBuffer(shadow_texture.Get(), 0, 0, {},
                                      {shadow_width, shadow_height, 1}, shadow_bridge.Get(), 0,
                                      bytes_per_row, 0);
        scheduler.BreakBlits();
        scheduler.CopyBufferToTexture(shadow_bridge.Get(), 0, bytes_per_row, 0, shadow_original,
                                      0, 0, {}, {shadow_width * 2, shadow_height, 1});
        scheduler.BreakBlits();
    }
    if (frame_dump.CapturingSampled()) {
        if (frame_dump.CapturingTextures()) {
            std::vector<FrameDump::Sampled> sampled;
            AppendSampledDump(5, info, views, samplers, sampled);
            frame_dump.NoteDispatchSampled(pipeline->Key().unique_hash, sampled);
        }
        FrameDumpBuffers(true, std::array{pipeline->Key().unique_hash});
        std::optional<std::array<u32, 3>> indirect_grid;
        if (indirect) {
            indirect_grid = std::array{gpu_memory->Read<u32>(*indirect),
                                       gpu_memory->Read<u32>(*indirect + 4),
                                       gpu_memory->Read<u32>(*indirect + 8)};
        }
        frame_dump.NoteLaunch(pipeline->Key().unique_hash,
                              {qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z},
                              {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z}, indirect,
                              indirect_grid);
    }
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        // The images the program writes, read back after it (the storage images are the last
        // views, after the texel buffers and the sampled textures).
        size_t view_index = 0;
        for (const auto& desc : info.texture_buffer_descriptors) {
            view_index += desc.count;
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            view_index += desc.count;
        }
        for (const auto& desc : info.texture_descriptors) {
            view_index += desc.count;
        }
        std::vector<FrameDump::Source> written;
        std::vector<FlashWatch::Source> watched;
        u32 slot = 0;
        for (const auto& desc : info.image_descriptors) {
            for (u32 element = 0; element < desc.count; ++element, ++slot) {
                if (desc.is_written && view_index + element < views.size() &&
                    views[view_index + element].id) {
                    ImageView& written_view =
                        texture_cache.GetImageView(views[view_index + element].id);
                    if (frame_dump.Capturing()) {
                        if (const auto source = DumpSource(written_view, slot)) {
                            written.push_back(*source);
                        }
                    }
                    if (flash_watch.WatchesPasses()) {
                        if (const auto source = FlashSource(written_view, slot)) {
                            watched.push_back(*source);
                        }
                    }
                }
            }
            view_index += desc.count;
        }
        if (frame_dump.Capturing()) {
            frame_dump.CaptureDispatch(pipeline->Key().unique_hash, written);
        }
        if (flash_watch.WatchesPasses()) {
            flash_watch.NoteDispatch(pipeline->Key().unique_hash, watched);
        }
    }
    Coverage::Note(Coverage::Work::Dispatch, Coverage::Path::Native);
}

void RasterizerMetal::ResetCounter(VideoCommon::QueryType type) {
    if (type == VideoCommon::QueryType::ZPassPixelCount64 ||
        type == VideoCommon::QueryType::ZPassPixelCount) {
        occlusion_slots.clear();
        occlusion_pass = ~u64{0};
    }
    using Type = VideoCommon::QueryType;
    if (type == Type::StreamingByteCount || type == Type::StreamingPrimitivesSucceeded) {
        // A transform feedback begins: the buffers fill from their start offsets.
        xfb_state.written = {};
        counters[static_cast<size_t>(Type::StreamingPrimitivesNeeded)] = 0;
    }
    if (type == Type::PrimitivesGenerated || type == Type::VtgPrimitivesOut) {
        counters[static_cast<size_t>(type)] = 0;
    }
}

void RasterizerMetal::CountDraw(u32 vertices, u32 instances) {
    using Type = VideoCommon::QueryType;
    const auto& regs = maxwell3d->regs;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const u64 total = u64{vertices} * instances;
    const u64 primitives = Xfb::PrimitiveCount(topology, total, regs.patch_vertices);
    const auto add = [&](Type type, u64 value) { counters[static_cast<size_t>(type)] += value; };
    add(Type::VerticesGenerated, total);
    add(Type::VertexShaderInvocations, total);
    add(Type::PrimitivesGenerated, primitives);
    add(Type::VtgPrimitivesOut, primitives);
    add(Type::ClipperInvocations, primitives);
    add(Type::ClipperPrimitivesGenerated, primitives);
    add(Type::GeometryShaderInvocations, geometry_input_primitives);
    if (regs.transform_feedback_enabled != 0) {
        add(Type::StreamingPrimitivesNeeded, primitives);
    }
}

void RasterizerMetal::PrepareXfbDraw(GraphicsPipeline& pipeline, bool is_indexed,
                                     u32 instance_count) {
    xfb_draw = {};
    XfbCapture* const capture = pipeline.Xfb();
    XfbCapture* const expansion = pipeline.Expand();
    if (capture == nullptr && expansion == nullptr) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const u32 count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    const u64 total = u64{count} * instance_count;
    if (expansion != nullptr) {
        // Lines and triangles drawn from the vertex kernel's outputs.
        const u32 prim = RasterExpand::PrimitiveOf(DrawTopology(*maxwell3d));
        if (total == 0 || total > Xfb::MaxCapturedVertices || prim == ~0U) {
            LOG_ONCE(warned_expand_size, "Metal: an expanded draw of {} vertices was not made",
                     total);
            return;
        }
        xfb_draw = XfbDraw{
            .wanted = true,
            .first = draw_state.vertex_buffer.first,
            .count = count,
            .first_instance = draw_state.base_instance,
            .instances = instance_count,
            .index_size = !is_indexed ? 0U
                          : draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedShort
                              ? 2U
                              : 4U,
            .widen_indices =
                is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte,
            .base_vertex = is_indexed ? static_cast<s32>(draw_state.base_index) : 0,
            .first_index = draw_state.index_buffer.first,
            .captured = static_cast<u32>(total),
            .expand = true,
            .prim = prim,
        };
        return;
    }
    if (total == 0 || total > Xfb::MaxCapturedVertices) {
        LOG_ONCE(warned_xfb_size, "Metal: a transform feedback capture of {} vertices was skipped",
                 total);
        return;
    }
    const u32 primitive_vertices = draw_state.topology == Maxwell::PrimitiveTopology::Points  ? 1U
                                   : draw_state.topology == Maxwell::PrimitiveTopology::Lines ? 2U
                                                                                              : 3U;
    // Where each buffer's capture stands: a new address (a begin, or a resume at another start
    // offset) counts from zero.
    std::array<u32, Xfb::MaxBuffers> capacity{};
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        if (capture->stride[buffer] == 0) {
            continue;
        }
        const auto& binding = regs.transform_feedback.buffers[buffer];
        const GPUVAddr base = binding.Address() + binding.start_offset;
        if (xfb_state.base[buffer] != base) {
            xfb_state.base[buffer] = base;
            xfb_state.written[buffer] = 0;
        }
        xfb_state.start[buffer] = static_cast<u32>(binding.start_offset);
        xfb_state.stride[buffer] = capture->stride[buffer];
        capacity[buffer] = static_cast<u32>(binding.size);
    }
    xfb_state.topology = draw_state.topology;
    xfb_state.patch_vertices = (std::max)(regs.patch_vertices, 1U);
    const u32 captured = Xfb::CapturedVertices(static_cast<u32>(total), primitive_vertices,
                                               capture->stride, capacity, xfb_state.written);
    xfb_draw = XfbDraw{
        .wanted = true,
        .first = draw_state.vertex_buffer.first,
        .count = count,
        .first_instance = draw_state.base_instance,
        .instances = instance_count,
        .index_size = !is_indexed ? 0U
                      : draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedShort ? 2U
                                                                                             : 4U,
        .widen_indices =
            is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte,
        .base_vertex = is_indexed ? static_cast<s32>(draw_state.base_index) : 0,
        .first_index = draw_state.index_buffer.first,
        .captured = captured,
        .offsets = xfb_state.written,
    };
    for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
        xfb_state.written[buffer] += captured * capture->stride[buffer];
    }
}

std::optional<u64> RasterizerMetal::CounterValue(VideoCommon::QueryType type,
                                                 u32 subreport) const {
    using Type = VideoCommon::QueryType;
    // The buffer a stream's counters read: the first of the enabled ones that has the stream.
    const auto stream_buffer = [&]() -> std::optional<u32> {
        const auto& regs = maxwell3d->regs;
        for (u32 buffer = 0; buffer < Xfb::MaxBuffers; ++buffer) {
            if (regs.transform_feedback.buffers[buffer].enable != 0 &&
                regs.transform_feedback.controls[buffer].stream == subreport) {
                return buffer;
            }
        }
        return std::nullopt;
    };
    const auto bytes = [&](u32 buffer) {
        return u64{xfb_state.start[buffer]} + xfb_state.written[buffer];
    };
    const auto succeeded = [&]() -> u64 {
        const std::optional<u32> buffer = stream_buffer();
        if (!buffer || xfb_state.stride[*buffer] == 0) {
            return 0;
        }
        return Xfb::PrimitiveCount(xfb_state.topology,
                                   xfb_state.written[*buffer] / xfb_state.stride[*buffer],
                                   xfb_state.patch_vertices);
    };
    switch (type) {
    case Type::StreamingByteCount: {
        const std::optional<u32> buffer = stream_buffer();
        return buffer ? bytes(*buffer) : 0;
    }
    case Type::StreamingPrimitivesSucceeded:
        return succeeded();
    case Type::StreamingPrimitivesNeeded:
        return counters[static_cast<size_t>(type)];
    case Type::StreamingPrimitivesNeededMinusSucceeded: {
        const u64 needed = counters[static_cast<size_t>(Type::StreamingPrimitivesNeeded)];
        const u64 done = succeeded();
        return needed > done ? needed - done : 0;
    }
    case Type::VerticesGenerated:
    case Type::PrimitivesGenerated:
    case Type::VertexShaderInvocations:
    case Type::ClipperInvocations:
    case Type::ClipperPrimitivesGenerated:
    case Type::VtgPrimitivesOut:
    case Type::GeometryShaderInvocations:
    case Type::GeometryShaderPrimitivesGenerated:
    case Type::PixelShaderInvocations:
        return counters[static_cast<size_t>(type)];
    default:
        return std::nullopt;
    }
}

u32 RasterizerMetal::AllocateVisibilitySlot() {
    const u32 slot = next_visibility_slot;
    next_visibility_slot = (next_visibility_slot + 1) % Scheduler::VisibilitySlots;
    // A lap later: the GPU wrote the slot long ago, but make sure.
    if (!scheduler.IsFree(visibility_ticks[slot])) {
        scheduler.Wait(visibility_ticks[slot], CommitReason::Query);
    }
    visibility_ticks[slot] = scheduler.CurrentTick();
    scheduler.VisibilityResults()[slot] = 0;
    return slot;
}

void RasterizerMetal::CountOcclusion() {
    const bool zpass = maxwell3d->regs.zpass_pixel_count_enable != 0;
    if (!zpass && !count_pixels) {
        scheduler.SetVisibility(std::nullopt);
        return;
    }
    // A slot per render pass of the counting segment: results are summed at the query. The pixel
    // program invocation counter (once a report asks for it) reads the same slots.
    if (occlusion_pass != scheduler.RenderPassSerial()) {
        visibility_slot = AllocateVisibilitySlot();
        occlusion_pass = scheduler.RenderPassSerial();
        slot_in_occlusion = slot_in_pixels = false;
    }
    if (zpass && !slot_in_occlusion) {
        occlusion_slots.push_back(visibility_slot);
        slot_in_occlusion = true;
    }
    if (count_pixels && !slot_in_pixels) {
        if (pixel_slots.size() >= 4096) {
            ResolvePixelCounter();
        }
        pixel_slots.push_back(visibility_slot);
        slot_in_pixels = true;
    }
    if (const Framebuffer* const framebuffer = texture_cache.GetFramebuffer()) {
        pixel_samples = (std::max)(framebuffer->Samples(), 1U);
    }
    scheduler.SetVisibility(visibility_slot);
}

void RasterizerMetal::ResolvePixelCounter() {
    // The samples that passed, over the samples a pixel has: the invocations of the pixel
    // program as far as a visibility counter says (early depth and stencil kills are not
    // counted as invocations, a draw's helper invocations are).
    if (pixel_slots.empty()) {
        return;
    }
    scheduler.Wait(visibility_ticks[pixel_slots.back()], CommitReason::Query);
    const u64* const results = scheduler.VisibilityResults();
    u64 samples = 0;
    for (const u32 slot : pixel_slots) {
        samples += results[slot];
    }
    counters[static_cast<size_t>(VideoCommon::QueryType::PixelShaderInvocations)] +=
        samples / pixel_samples;
    pixel_slots.clear();
    // The slot the pass in progress still counts into: its samples so far were taken.
    slot_in_pixels = false;
    occlusion_pass = ~u64{0};
}

void RasterizerMetal::ResolveGeometryCounters() {
    // The primitives the geometry draws' compactions listed: read once the GPU is done with them.
    for (const GeometryReadback& readback : geometry_readbacks) {
        scheduler.Wait(readback.tick, CommitReason::Query);
        u32 indices = 0;
        std::memcpy(&indices, readback.args.mapped_span.data(), sizeof(indices));
        const u32 per_primitive = readback.topology == 1 ? 1U : readback.topology == 6 ? 2U : 3U;
        counters[static_cast<size_t>(VideoCommon::QueryType::GeometryShaderPrimitivesGenerated)] +=
            indices / per_primitive;
    }
    geometry_readbacks.clear();
}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (type != VideoCommon::QueryType::ZPassPixelCount64 &&
        type != VideoCommon::QueryType::ZPassPixelCount) {
        QueryFallback(gpu_addr, type, flags, payload, subreport);
        return;
    }
    if (type == VideoCommon::QueryType::ZPassPixelCount64 &&
        VideoCommon::OotOcclusionOff(program_id)) {
        // As Vulkan's query cache: Ocarina's occlusion reports read "visible"
        // (EDEN_NX2_OOT_OCCLUSION_OFF=0 counts them).
        QueryFallback(gpu_addr, VideoCommon::QueryType::Payload, flags, 1, subreport);
        return;
    }
    if (!gpu_memory) {
        return;
    }
    // The address is resolved now, as Vulkan's query cache does: a report to an unmapped address
    // is dropped there, and a later remap must not redirect a pending write.
    const std::optional<DAddr> address = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!address) {
        NoteUnmappedReport(gpu_addr, type);
        return;
    }
    ++occlusion_queries;
    const bool wide = True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
    // A condition that reads the report before it is written decides on the GPU from these slots.
    const u64 serial = ++report_serial;
    const DAddr report_address = *address;
    {
        std::scoped_lock lock{pending_reports_mutex};
        pending_reports.insert_or_assign(report_address,
                                         PendingReport{.slots = occlusion_slots,
                                                       .wide = wide,
                                                       .tick = scheduler.CurrentTick(),
                                                       .serial = serial});
    }
    // Later draws count into a new slot: the report must not see their samples (a render pass
    // goes on past the report when no fence ends its command buffer).
    occlusion_pass = ~u64{0};
    // The split's front treats the words as final only once the write ran (Nx2AsyncReportsSettled).
    const u64 async_slot = Tegra::Nx2AsyncReportSlot(report_address);
    Tegra::g_nx2_async_reports_issued[async_slot].fetch_add(1, std::memory_order_release);
    // Written behind the next fence, after waiting for the command buffer that holds the counted
    // draws: Metal's fences complete at once below High accuracy, and a callback summing the
    // slots right away read the zeros AllocateVisibilitySlot left there (every report was 0).
    // A read before then (macro DMA, conditional rendering) flushes it (FlushQueryResults).
    if (FenceManager::AsyncFences()) {
        // The operation runs on the fencing thread, which can't commit: the report's tick is
        // committed by the next commit point (EDEN_NX2_METAL_ASYNC_FENCES).
        scheduler.RequestCommit();
    }
    fence_manager.SyncOperation([this, report_address, serial, async_slot] {
        WritePendingReport(report_address, serial);
        Tegra::g_nx2_async_reports_completed[async_slot].fetch_add(1, std::memory_order_release);
    });
}

bool RasterizerMetal::WritePendingReport(DAddr address, u64 serial) {
    PendingReport report;
    {
        std::scoped_lock lock{pending_reports_mutex};
        const auto it = pending_reports.find(address);
        if (it == pending_reports.end() || (serial != 0 && it->second.serial != serial)) {
            return false;
        }
        report = it->second;
    }
    if (FenceManager::AsyncFences()) {
        // From the fencing thread (a fence operation) too: it waits without committing.
        scheduler.WaitAnyThread(report.tick, CommitReason::Query);
    } else {
        scheduler.Wait(report.tick, CommitReason::Query);
    }
    const u64* const results = scheduler.VisibilityResults();
    u64 samples = 0;
    for (const u32 slot : report.slots) {
        samples += results[slot];
    }
    WriteQueryResult(address, report.wide, samples);
    std::scoped_lock lock{pending_reports_mutex};
    if (const auto it = pending_reports.find(address);
        it != pending_reports.end() && it->second.serial == report.serial) {
        pending_reports.erase(it);
    }
    return true;
}

bool RasterizerMetal::HasPendingQueries(DAddr addr, u64 size) {
    std::scoped_lock lock{pending_reports_mutex};
    for (const auto& [address, report] : pending_reports) {
        if (address < addr + size && addr < address + (report.wide ? 16 : 4)) {
            return true;
        }
    }
    return false;
}

void RasterizerMetal::FlushQueryResults(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    boost::container::small_vector<DAddr, 8> addresses;
    {
        std::scoped_lock lock{pending_reports_mutex};
        for (const auto& [address, report] : pending_reports) {
            if (address < addr + size && addr < address + (report.wide ? 16 : 4)) {
                addresses.push_back(address);
            }
        }
    }
    for (const DAddr address : addresses) {
        WritePendingReport(address, 0);
    }
}

void RasterizerMetal::NoteUnmappedReport(GPUVAddr gpu_addr, VideoCommon::QueryType type) {
    if (unmapped_reports++ < 4) {
        LOG_INFO(Render_Metal, "Metal: query report {} to unmapped GPU address {:#x} dropped",
                 static_cast<u32>(type), gpu_addr);
    }
}

void RasterizerMetal::WriteQueryResult(DAddr address, bool wide, u64 value) {
    u8* const pointer = device_memory.GetPointer<u8>(address);
    if (pointer == nullptr) {
        return;
    }
    if (wide) {
        const u64 timestamp = gpu.GetTicks();
        if (u8* const timestamp_pointer = device_memory.GetPointer<u8>(address + 8)) {
            std::memcpy(timestamp_pointer, &timestamp, sizeof(timestamp));
        }
        std::memcpy(pointer, &value, sizeof(value));
    } else {
        const u32 word = static_cast<u32>(value);
        std::memcpy(pointer, &word, sizeof(word));
    }
    // A channel waiting on this semaphore resumes (the query cache does the same on Vulkan).
    Tegra::Nx2SemaphoreWaiters::Get().OnWrite(address, static_cast<u32>(value));
}

void RasterizerMetal::QueryFallback(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                                    VideoCommon::QueryPropertiesFlags flags, u32 payload,
                                    u32 subreport) {
    if (!gpu_memory) {
        return;
    }
    const std::optional<DAddr> address = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!address) {
        NoteUnmappedReport(gpu_addr, type);
        return;
    }
    // The counters of the draws made so far (the draws are counted on the CPU as they are made,
    // so the value is already final); the others read as 1 (something passed), as OpenGL's
    // fallback does.
    u64 value = payload;
    if (type == VideoCommon::QueryType::PixelShaderInvocations) {
        // Counting starts with the first report that asks.
        count_pixels = true;
        ResolvePixelCounter();
    } else if (type == VideoCommon::QueryType::GeometryShaderPrimitivesGenerated) {
        ResolveGeometryCounters();
    }
    if (type != VideoCommon::QueryType::Payload) {
        value = CounterValue(type, subreport).value_or(1U);
    }
    const bool wide = True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
    std::function<void()> func([this, report_addr = *address, wide, value] {
        WriteQueryResult(report_addr, wide, value);
    });
    if (True(flags & VideoCommon::QueryPropertiesFlags::IsAFence)) {
        // A counter's own fence waits too, but the draw of its byte count reads it on the CPU.
        if (type == VideoCommon::QueryType::StreamingByteCount) {
            WriteQueryResult(*address, wide, value);
        }
        SignalFence(std::move(func));
        return;
    }
    func();
}

void RasterizerMetal::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    // EDEN_NX2_METAL_UNIFORM_BIND_NOLOCK=0: the buffer cache's mutex taken for each constant
    // buffer bind. Otherwise none, as RasterizerVulkan::BindGraphicsUniformBuffer: it records the
    // channel's binding (resolved under the mutex at the draw) on the thread that draws, where
    // the profile found the lock contended with other threads' cache work.
    static const bool lock_bind = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_UNIFORM_BIND_NOLOCK");
        const bool on = value != nullptr && value[0] == '0';
        LOG_INFO(Render_Metal, "Metal: constant buffer binds take the buffer cache lock: {} "
                               "(EDEN_NX2_METAL_UNIFORM_BIND_NOLOCK={})",
                 on ? "on" : "off", on ? 0 : 1);
        return on;
    }();
    if (lock_bind) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
        return;
    }
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::SubmitListCopies(u64 writer) {
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick(), writer)) {
        scheduler.Flush(CommitReason::ListCopy);
    }
}

VideoCore::RasterizerInterface::GpuWrittenList RasterizerMetal::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync) {
    if (!ListCopiesEnabled()) {
        return GpuWrittenList::Unknown;
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

void RasterizerMetal::NoteGpuWrittenCommandListSynced(DAddr page) {
    if (ListCopiesEnabled()) {
        buffer_cache.Nx2NoteListSynced(page);
    }
}

void RasterizerMetal::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    if (!ListCopiesEnabled()) {
        RasterizerInterface::FlushGpuWrittenCommandListPages(pages);
        return;
    }
    gpu.InvalidateGPUCache();
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2FlushListPages(pages);
}

void RasterizerMetal::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (CachePathsEnabled() && True(which & (VideoCommon::CacheType::TextureCache |
                                             VideoCommon::CacheType::BufferCache))) {
        // Collected guest writes first: the download must not land over them.
        gpu.InvalidateGPUCache();
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::QueryCache)) {
        FlushQueryResults(addr, size);
    }
}

bool RasterizerMetal::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    const bool fast = CachePathsEnabled();
    if (True(which & VideoCommon::CacheType::BufferCache) &&
        (!fast || MayBeGpuModified(addr, size))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!VideoCore::Nx2Debug::AccuratePart(4)) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        if (fast && TextureReadFilter() && !texture_cache.MayHaveGpuModifiedImages(addr, size)) {
            return false;
        }
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerMetal::GetFlushArea(DAddr addr, u64 size) {
    // A guest core, on every trapped load: a cache that can hold nothing of the range is not
    // asked, so its lock (held by the GPU thread most of the time) is not waited for.
    static const bool read_fast_path = EnvironmentOn("EDEN_NX2_READ_FASTPATH");
    const bool fast = CachePathsEnabled() && read_fast_path;
    const bool skip_textures =
        fast && (!texture_cache.page_filter.Any(addr, size) ||
                 (TextureReadFilter() && !texture_cache.MayHaveGpuModifiedImages(addr, size)));
    const bool skip_buffers = fast && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
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
    if (!skip_buffers && (!CachePathsEnabled() || buffer_cache.FlushesOnCpuRead())) {
        const u64 lock_begin = ReadStats::NowNs();
        std::scoped_lock lock{buffer_cache.mutex};
        ReadStats::Add(ReadStats::g_counters.buffer_lock_ns, ReadStats::NowNs() - lock_begin);
        ReadStats::Add(ReadStats::g_counters.buffer_asks, 1);
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    return VideoCore::RasterizerDownloadArea{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
}

bool RasterizerMetal::LandDownloadForCpuRead(DAddr addr, u64 size, u64 tick) {
    // The copy was recorded on the GPU thread; the read can only wait for a committed one.
    if (!CachePathsEnabled() || !scheduler.IsSubmitted(tick)) {
        return false;
    }
    scheduler.WaitSubmitted(tick);
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.LandInFlightForCpuRead(addr, size, false);
    return !buffer_cache.IsRegionGpuModified(addr, size);
}

void RasterizerMetal::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (!CachePathsEnabled() || addr == 0 || size == 0 ||
        False(which & VideoCommon::CacheType::BufferCache)) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.ClearDownload(addr, size);
}

void RasterizerMetal::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    texture_cache.NoteGuestWrite();
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
        InvalidateConvertedIndicesAddr(addr, size);
    }
    if (True(which & VideoCommon::CacheType::ShaderCache)) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerMetal::InnerInvalidation(
    std::span<const std::pair<DAddr, std::size_t>> sequences) {
    if (!CachePathsEnabled()) {
        RasterizerInterface::InnerInvalidation(sequences);
        return;
    }
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    // Each lock once for the batch; the texture and shader caches only where they ever held
    // something (page_filter).
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
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
            InvalidateConvertedIndicesAddr(addr, size);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    texture_cache.NoteGuestWrite();
    if (!CachePathsEnabled()) {
        {
            std::scoped_lock lock{buffer_cache.mutex};
            InvalidateConvertedIndicesAddr(addr, size);
            if (buffer_cache.OnCPUWrite(addr, size)) {
                Tegra::Nx2NoteDeferredCodeWrite(addr, size);
                return true;
            }
        }
        {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.WriteMemory(addr, size);
        }
        pipeline_cache.InvalidateRegion(addr, size);
        return false;
    }
    // EDEN_NX2_EXP35: a write no cache can hold anything of is collected without the lock.
    if (const int collect = VideoCore::Nx2Debug::Experiment(35);
        collect != 0 && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        (collect == 2 || !Tegra::Nx2MayBeGpuWritten(addr, size)) &&
        !pipeline_cache.page_filter.Any(addr, size) &&
        !texture_cache.page_filter.Any(addr, size)) {
        return true;
    }
    static const bool fast_path = EnvironmentOn("EDEN_NX2_WRITE_FASTPATH");
    // A guest core never waits for the buffer cache's lock with a write of plain memory: the
    // lock's next owner applies it (the owner hook).
    const bool handoff = fast_path && VideoCommon::Nx2WriteHandoff::Enabled() &&
                         Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
                         !Tegra::Nx2MayBeGpuWritten(addr, size) &&
                         !texture_cache.page_filter.Any(addr, size) &&
                         !pipeline_cache.page_filter.Any(addr, size);
    if (handoff && !buffer_cache.mutex.try_lock()) {
        if (write_handoff.Push(addr, size)) {
            if (Tegra::Nx2MayBeGpuWritten(addr, size)) {
                // Marked GPU-written meanwhile: taking the lock applies the write now.
                std::scoped_lock lock{buffer_cache.mutex};
            }
            return false;
        }
        buffer_cache.mutex.lock();
    } else if (!handoff) {
        buffer_cache.mutex.lock();
    }
    static thread_local VideoCommon::DeferredPageCaching deferred;
    const bool defer = VideoCommon::ProtectUnlockedEnabled() &&
                       VideoCommon::t_deferred_page_caching == nullptr &&
                       !VideoCommon::GpuSplit::OwnsScheduler();
    bool gpu_modified;
    {
        std::unique_lock lock{buffer_cache.mutex, std::adopt_lock};
        InvalidateConvertedIndicesAddr(addr, size);
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

void RasterizerMetal::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    texture_cache.NoteGuestWrite();
    if (!CachePathsEnabled()) {
        RasterizerInterface::OnCacheInvalidations(ranges);
        return;
    }
    {
        // Out of the GPU-written ranges first: applying one write downloads the GPU's bytes
        // elsewhere on its pages, which would write over the next collected write.
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ClaimCpuWrites(ranges);
    }
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
                InvalidateConvertedIndicesAddr(addr, size);
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

bool RasterizerMetal::TryCacheInvalidation(DAddr addr, u64 size) {
    if (!CachePathsEnabled()) {
        return RasterizerInterface::TryCacheInvalidation(addr, size);
    }
    if (addr == 0 || size == 0) {
        return true;
    }
    {
        // WriteMemory downloads GPU-written bytes elsewhere on the written pages, which only
        // the GPU thread may do: those pages are left to the fallback.
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) {
            return false;
        }
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
        InvalidateConvertedIndicesAddr(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

bool RasterizerMetal::FencesQuiet(bool strict, u64& epoch) {
    return CachePathsEnabled() && fence_manager.FencesQuiet(strict, epoch);
}

u64 RasterizerMetal::FenceWorkEpoch() const {
    return CachePathsEnabled() ? fence_manager.FenceWorkEpoch() : 0;
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {
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
        InvalidateConvertedIndicesAddr(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

void RasterizerMetal::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
        InvalidateConvertedIndicesAddr(addr, size);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerMetal::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    // Nothing yet: the buffer cache keeps virtual ranges only for runtimes that bind them.
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerMetal::SignalFence(std::function<void()>&& func, bool force_completion) {
    fence_manager.SignalFence(std::move(func), force_completion);
}

void RasterizerMetal::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerMetal::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerMetal::SignalReference() {
    fence_manager.SignalOrdering();
}

void RasterizerMetal::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerMetal::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                               VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerMetal::WaitForIdle() {
    SignalReference();
}

void RasterizerMetal::FragmentBarrier() {
    // The next draw ends the render encoder only if it reads what the pass wrote, as Vulkan's.
    fragment_barrier_pass = FragmentBarrierEnabled() && scheduler.IsRendering()
                                ? scheduler.RenderPassSerial()
                                : 0;
}

void RasterizerMetal::TiledCacheBarrier() {
    // Vulkan barriers the attachments here. Apple GPUs keep a pass's attachments in tile memory,
    // so a later draw of the same pass that samples one sees it only from a new encoder: the same
    // treatment as FragmentBarrier (the next reading draw ends the encoder).
    FragmentBarrier();
}

bool RasterizerMetal::AccelerateConditionalRendering() {
    // Metal has no predicate rendering. A condition on values in guest memory is compared on the
    // CPU by the caller, as for Vulkan without VK_EXT_conditional_rendering (MoltenVK), after the
    // same flush. One that reads an occlusion report whose fence has not signalled (its value is
    // not in memory yet) is decided on the GPU (metal_condition.h): EDEN_NX2_METAL_GPU_CONDITION=0
    // compares what memory holds then, as before.
    static const bool on_gpu = EnvironmentOn("EDEN_NX2_METAL_GPU_CONDITION");
    gpu_condition.active = false;
    gpu_condition.region = {};
    const auto& regs = maxwell3d->regs;
    const bool reads_memory =
        regs.render_enable_override == Maxwell::RenderEnable::Override::UseRenderEnable &&
        regs.render_enable.mode != Maxwell::RenderEnable::Mode::False &&
        regs.render_enable.mode != Maxwell::RenderEnable::Mode::True;
    if (reads_memory && CachePathsEnabled()) {
        gpu_memory->FlushCaching();
    }
    if (!reads_memory || !on_gpu) {
        return false;
    }
    const GPUVAddr condition = regs.render_enable.Address();
    const std::optional<DAddr> first = gpu_memory->GpuToCpuAddress(condition);
    const std::optional<DAddr> second = gpu_memory->GpuToCpuAddress(condition + 16);
    if (!first || !second) {
        return false;
    }
    Condition::Params params{};
    params.mode = static_cast<u32>(
        regs.render_enable.mode == Maxwell::RenderEnable::Mode::Conditional ? Condition::Mode::BothWordsNonZero
        : regs.render_enable.mode == Maxwell::RenderEnable::Mode::IfEqual   ? Condition::Mode::Equal
                                                                            : Condition::Mode::NotEqual);
    const size_t values = params.mode == 0 ? 1 : 2;
    std::array<u64, 2> ticks{};
    std::array<std::vector<u32>, 2> long_reports;
    bool pending = false;
    {
        std::scoped_lock lock{pending_reports_mutex};
        for (size_t value = 0; value < values; ++value) {
            const DAddr address = value == 0 ? *first : *second;
            std::array<u32, 2> words{};
            if (const u8* const pointer = device_memory.GetPointer<u8>(address)) {
                std::memcpy(words.data(), pointer, sizeof(words));
            }
            params.memory[value] = words;
            const auto it = pending_reports.find(address);
            if (it == pending_reports.end()) {
                continue;
            }
            pending = true;
            ticks[value] = it->second.tick;
            params.write_mask[value] = {~0U, it->second.wide ? ~0U : 0U};
            if (it->second.slots.size() > Condition::MaxSlots) {
                long_reports[value] = it->second.slots;
                continue;
            }
            params.slot_count[value] = static_cast<u32>(it->second.slots.size());
            std::copy(it->second.slots.begin(), it->second.slots.end(),
                      params.slots[value].begin());
        }
    }
    if (!pending) {
        // Every value is in memory: the caller's comparison is exact.
        return false;
    }
    gpu_condition = GpuCondition{.active = true, .params = params, .ticks = ticks};
    if (!long_reports[0].empty() || !long_reports[1].empty()) {
        // More passes than the kernel sums: the CPU waits for them and decides.
        for (size_t value = 0; value < 2; ++value) {
            gpu_condition.params.slot_count[value] = 0;
            if (!long_reports[value].empty()) {
                gpu_condition.params.slots[value] = {};
            }
        }
        for (const u64 tick : ticks) {
            if (tick != 0) {
                scheduler.Wait(tick, CommitReason::Condition);
            }
        }
        const u64* const results = scheduler.VisibilityResults();
        for (size_t value = 0; value < values; ++value) {
            Condition::Params& resolved = gpu_condition.params;
            u64 sum = 0;
            const std::vector<u32>& slots =
                !long_reports[value].empty()
                    ? long_reports[value]
                    : std::vector<u32>(params.slots[value].begin(),
                                       params.slots[value].begin() + params.slot_count[value]);
            for (const u32 slot : slots) {
                sum += results[slot];
            }
            const u64 memory =
                u64{params.memory[value][0]} | (u64{params.memory[value][1]} << 32);
            const u64 mask =
                u64{params.write_mask[value][0]} | (u64{params.write_mask[value][1]} << 32);
            const u64 merged = (memory & ~mask) | (sum & mask);
            resolved.memory[value] = {static_cast<u32>(merged), static_cast<u32>(merged >> 32)};
            resolved.write_mask[value] = {};
        }
        gpu_condition.holds = Condition::Evaluate(gpu_condition.params, results);
        gpu_condition.resolved = true;
        ++conditions_on_cpu;
        return true;
    }
    ++conditions_on_gpu;
    return true;
}

std::string RasterizerMetal::TakeGpuDecisionLine() {
    const u64 on_gpu = std::exchange(conditions_on_gpu, 0);
    const u64 on_cpu = std::exchange(conditions_on_cpu, 0);
    const u64 under = std::exchange(draws_under_condition, 0);
    const u64 indirect = std::exchange(indirect_draws_on_gpu, 0);
    const u64 counts_read = std::exchange(indirect_counts_read, 0);
    const u64 grid = std::exchange(grid_draws_on_gpu, 0);
    if (on_gpu + on_cpu + under + indirect + counts_read + grid == 0) {
        return {};
    }
    return fmt::format("conditions on pending reports {} on the GPU ({} draws under them), {} "
                       "waited for on the CPU; indirect draws prepared on the GPU {}, counts "
                       "read on the CPU {}; GPU-counted draws drawn from GPU arguments {}",
                       on_gpu, under, on_cpu, indirect, counts_read, grid);
}

bool RasterizerMetal::ConditionHoldsOnCpu() {
    if (!gpu_condition.active) {
        return true;
    }
    if (!gpu_condition.resolved) {
        for (const u64 tick : gpu_condition.ticks) {
            if (tick != 0) {
                scheduler.Wait(tick, CommitReason::Condition);
            }
        }
        gpu_condition.holds =
            Condition::Evaluate(gpu_condition.params, scheduler.VisibilityResults());
        gpu_condition.resolved = true;
    }
    return gpu_condition.holds;
}

bool RasterizerMetal::PrepareConditionRegion() {
    if (gpu_condition.resolved) {
        // Waited for already: the CPU knows the answer.
        return false;
    }
    if (gpu_condition.region.buffer != nullptr && gpu_condition.draws < Condition::RegionDraws &&
        gpu_condition.region_tick == scheduler.CurrentTick()) {
        return true;
    }
    if (!condition_kernel && !condition_kernel_failed) {
        std::string error;
        const Object library = CreateLibrary(device, Condition::KernelSource(), error);
        const Object function =
            library ? CreateFunction(library.Get(), "condition_arguments") : Object{};
        condition_kernel =
            function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
        if (!condition_kernel) {
            condition_kernel_failed = true;
            LOG_ERROR(Render_Metal, "Metal: condition kernel failed to build: {}", error);
        }
    }
    if (!condition_kernel) {
        return false;
    }
    constexpr size_t RegionBytes =
        (Condition::HeaderWords + size_t{Condition::RegionDraws} * Condition::ArgumentWords) *
        sizeof(u32);
    gpu_condition.region = staging_pool.Stream(RegionBytes, 16);
    gpu_condition.region_words = reinterpret_cast<u32*>(gpu_condition.region.mapped_span.data());
    gpu_condition.region_words[0] = 0;
    gpu_condition.draws = 0;
    // After the render passes that counted the samples (not ahead of the segment), before the
    // draws: the draw count and their arguments are written before the commit.
    scheduler.BeginCompute(condition_kernel.Get(), false);
    scheduler.BindBuffer(ShaderStage::Compute, 0, scheduler.VisibilityBuffer(), 0);
    scheduler.BindBytes(ShaderStage::Compute, 1, &gpu_condition.params,
                        static_cast<u32>(sizeof(gpu_condition.params)));
    scheduler.BindBuffer(ShaderStage::Compute, 2, gpu_condition.region.buffer,
                         static_cast<u32>(gpu_condition.region.offset));
    constexpr u32 GroupSize = 64;
    scheduler.Dispatch({Condition::RegionDraws / GroupSize, 1, 1}, {GroupSize, 1, 1});
    gpu_condition.region_tick = scheduler.CurrentTick();
    return true;
}

bool RasterizerMetal::DrawUnderCondition(u32 primitive, bool indexed, u32 count, u32 index_size,
                                         BufferHandle index_buffer, u64 index_offset, u32 first,
                                         u32 instance_count, s32 base_vertex,
                                         u32 base_instance) {
    if (gpu_condition.region.buffer == nullptr ||
        gpu_condition.region_tick != scheduler.CurrentTick() ||
        gpu_condition.draws >= Condition::RegionDraws) {
        return false;
    }
    const u32 draw = gpu_condition.draws++;
    u32* const arguments =
        gpu_condition.region_words + Condition::HeaderWords + draw * Condition::ArgumentWords;
    if (indexed) {
        arguments[0] = count;
        arguments[1] = instance_count;
        arguments[2] = first;
        arguments[3] = static_cast<u32>(base_vertex);
        arguments[4] = base_instance;
    } else {
        arguments[0] = count;
        arguments[1] = instance_count;
        arguments[2] = first;
        arguments[3] = base_instance;
        arguments[4] = 0;
    }
    gpu_condition.region_words[0] = gpu_condition.draws;
    const u64 offset = gpu_condition.region.offset +
                       (Condition::HeaderWords + u64{draw} * Condition::ArgumentWords) * sizeof(u32);
    if (indexed) {
        scheduler.DrawIndexedIndirect(primitive, index_size, index_buffer, index_offset,
                                      gpu_condition.region.buffer, offset);
    } else {
        scheduler.DrawIndirect(primitive, gpu_condition.region.buffer, offset);
    }
    ++draws_under_condition;
    return true;
}

void RasterizerMetal::FlushCommands() {
    if (scheduler.HasPendingWork() && !scheduler.DeferListEnd()) {
        scheduler.Flush(CommitReason::ListEnd);
    }
}

std::optional<FrameDump::Source> RasterizerMetal::DumpSource(const ImageView& view,
                                                              u32 slot) {
    const Image* const image = view.BackingImage();
    if (image == nullptr || image->info.type == VideoCommon::ImageType::Buffer ||
        image->info.type == VideoCommon::ImageType::e3D) {
        return std::nullopt;
    }
    const auto& info = image->info;
    const TextureHandle texture = image->Handle();
    if (texture == nullptr) {
        return std::nullopt;
    }
    const bool rescaled = image->IsRescaled();
    const auto& resolution = Settings::values.resolution_info;
    const u32 width = rescaled ? resolution.ScaleUp(info.size.width) : info.size.width;
    const u32 height = rescaled ? resolution.ScaleUp(info.size.height) : info.size.height;
    const u32 level = static_cast<u32>(view.range.base.level);
    return FrameDump::Source{
        .texture = texture,
        .format = VideoCore::Surface::BytesPerBlock(view.format) ==
                          VideoCore::Surface::BytesPerBlock(info.format)
                      ? view.format
                      : info.format,
        .width = std::max(1U, width >> level),
        .height = std::max(1U, height >> level),
        .gpu_addr = image->gpu_addr,
        .slot = slot,
        .level = level,
        .layer = static_cast<u32>(view.range.base.layer),
        .msaa = info.num_samples > 1 || image->HostSamples() > 1,
        .rescaled = rescaled,
        .host_format = DescribeTexture(texture).format,
    };
}

FrameDump::Sampled RasterizerMetal::DumpSampled(u32 stage, u32 tic,
                                                VideoCommon::ImageViewId view_id,
                                                VideoCommon::SamplerId sampler_id) {
    FrameDump::Sampled out{.stage = stage, .tic = tic};
    if (!view_id) {
        out.null = true;
        return out;
    }
    const ImageView& view = texture_cache.GetImageView(view_id);
    const Image* const image = view.BackingImage();
    if (image == nullptr) {
        out.null = true;
        return out;
    }
    out.source = DumpSource(view, 0);
    out.key_addr = image->gpu_addr;
    out.key_extra = (static_cast<u32>(view.format) << 16) |
                    (static_cast<u32>(view.range.base.level) << 8) |
                    static_cast<u32>(view.range.base.layer);
    out.width = image->info.size.width;
    out.height = image->info.size.height;
    out.format = static_cast<u32>(image->info.format);
    out.block_compressed = VideoCore::Surface::DefaultBlockWidth(image->info.format) != 1;
    if (sampler_id) {
        const auto& tsc = texture_cache.GetSampler(sampler_id).Descriptor();
        std::array<u32, 8> words;
        static_assert(sizeof(words) == sizeof(tsc));
        std::memcpy(words.data(), &tsc, sizeof(words));
        std::string text;
        for (const u32 word : words) {
            text += fmt::format("{}\"{:08x}\"", text.empty() ? "" : ",", word);
        }
        out.sampler = fmt::format(
            R"(,"sampler":{{"mag":{},"min":{},"mip":{},"wrap":[{},{},{}],"compare":{},"tsc":[{}]}})",
            static_cast<u32>(tsc.mag_filter.Value()), static_cast<u32>(tsc.min_filter.Value()),
            static_cast<u32>(tsc.mipmap_filter.Value()), static_cast<u32>(tsc.wrap_u.Value()),
            static_cast<u32>(tsc.wrap_v.Value()), static_cast<u32>(tsc.wrap_p.Value()),
            tsc.depth_compare_enabled.Value() != 0, text);
    }
    return out;
}

bool RasterizerMetal::FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes) {
    static constexpr u32 BUFFER_CAP = 4U << 20;
    if (!frame_dump.CapturingBuffers()) {
        return false;
    }
    std::vector<FrameDump::BoundBuffer> bound;
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto* host, u32 host_offset) {
        const u32 size = (std::min)(bound_size, BUFFER_CAP);
        const bool gpu_modified = buffer_cache.IsRegionGpuModified(binding.device_addr, size);
        FrameDump::BoundBuffer& buffer = bound.emplace_back(FrameDump::BoundBuffer{
            .stage = stage,
            .storage = storage,
            .index = index,
            .written = written,
            .gpu_addr = binding.gpu_addr,
            .device_addr = binding.device_addr,
            .bound_size = bound_size,
            .size = size,
            .gpu_modified = gpu_modified,
            .host = host ? host->Handle() : nullptr,
            .host_offset = host_offset,
            .guest = {},
        });
        if (!gpu_modified || !host) {
            buffer.guest.resize(size);
            device_memory.ReadBlockUnsafe(binding.device_addr, buffer.guest.data(), size);
        }
        static const bool host_too = std::getenv("EDEN_NX2_FRAME_DUMP_HOST") != nullptr;
        buffer.also_host = host_too && host && !gpu_modified && (storage || bound_size > 4096);
    });
    return frame_dump.NoteBuffers(compute, shader_hashes, bound);
}

std::optional<FlashWatch::Source> RasterizerMetal::FlashSource(ImageView& view, u32 slot) {
    using namespace VideoCore::Surface;
    const Image* const image = view.BackingImage();
    if (image == nullptr) {
        return std::nullopt;
    }
    const auto& info = image->info;
    if (info.type == VideoCommon::ImageType::Buffer || info.num_samples > 1 ||
        GetFormatType(view.format) != SurfaceType::ColorTexture ||
        IsPixelFormatInteger(view.format) || DefaultBlockWidth(view.format) != 1) {
        return std::nullopt;
    }
    const TextureHandle texture = view.Handle(Shader::TextureType::Color2D);
    if (texture == nullptr) {
        return std::nullopt;
    }
    const bool rescaled = image->IsRescaled();
    const auto& resolution = Settings::values.resolution_info;
    const u32 width = rescaled ? resolution.ScaleUp(info.size.width) : info.size.width;
    const u32 height = rescaled ? resolution.ScaleUp(info.size.height) : info.size.height;
    const u32 level = static_cast<u32>(view.range.base.level);
    return FlashWatch::Source{
        .texture = texture,
        .format = view.format,
        .width = (std::max)(1U, width >> level),
        .height = (std::max)(1U, height >> level),
        .gpu_addr = image->gpu_addr,
        .slot = slot,
    };
}

void RasterizerMetal::FlashWatchPass(const Framebuffer* framebuffer,
                                     std::span<const u64> shader_hashes) {
    std::array<FlashWatch::Source, NUM_RT> colors;
    size_t count = 0;
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (u32 slot = 0; slot < ids.size(); ++slot) {
        if (!ids[slot]) {
            continue;
        }
        if (const auto source = FlashSource(texture_cache.GetImageView(ids[slot]), slot)) {
            colors[count++] = *source;
        }
    }
    flash_watch.BeginPass(framebuffer, std::span(colors).first(count));
    flash_watch.NoteDraw(shader_hashes);
}

void RasterizerMetal::FrameDumpPass(const Framebuffer* framebuffer,
                                    std::span<const u64> shader_hashes) {
    std::array<FrameDump::Source, NUM_RT> colors;
    size_t count = 0;
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (u32 slot = 0; slot < ids.size(); ++slot) {
        if (!ids[slot]) {
            continue;
        }
        if (const auto source = DumpSource(texture_cache.GetImageView(ids[slot]), slot)) {
            colors[count++] = *source;
        }
    }
    frame_dump.BeginPass(framebuffer, std::span(colors).first(count));
    frame_dump.NoteDraw(shader_hashes);
}

namespace {

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

} // Anonymous namespace

namespace {
// "Perf guest" figures as Vulkan's TickPerfReport prints them (EDEN_NX2_PERF_REPORT): system
// calls by number, slow-path memory callbacks, clock reads and yields, every 10 s.
void LogGuestReport() {
    if (!Common::PerfReport::Enabled()) {
        return;
    }
    static auto last = std::chrono::steady_clock::now();
    static std::array<u64, Common::PerfReport::Count> before = Common::PerfReport::Read();
    static std::array<u64, 256> svcs_before{};
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last).count();
    if (seconds < 10.0) {
        return;
    }
    const auto counters = Common::PerfReport::Read();
    const auto rate = [&](Common::PerfReport::Counter c) {
        return static_cast<double>(counters[c] - before[c]) / seconds;
    };
    std::array<std::pair<u64, size_t>, 256> rank{};
    std::array<u64, 256> svcs{};
    for (size_t i = 0; i < svcs.size(); ++i) {
        svcs[i] = Common::PerfReport::g_svc_counts[i].load(std::memory_order_relaxed);
        rank[i] = {svcs[i] - svcs_before[i], i};
    }
    std::partial_sort(rank.begin(), rank.begin() + 6, rank.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string svc_text;
    for (size_t i = 0; i < 6 && rank[i].first != 0; ++i) {
        svc_text += fmt::format("{}{:#04x} {:.0f}", i == 0 ? " (" : ", ", rank[i].second,
                                static_cast<double>(rank[i].first) / seconds);
    }
    if (!svc_text.empty()) {
        svc_text += ")";
    }
    LOG_INFO(Common_Memory,
             "Perf guest (Metal): {:.1f} fastmem faults/s | SVCs {:.0f}/s{} | memory callbacks/s: "
             "reads {:.0f}, writes {:.0f} | clock reads {:.0f}/s | yields to nobody {:.0f}/s",
             rate(Common::PerfReport::JitFastmemFaults), rate(Common::PerfReport::GuestSvcs),
             svc_text, rate(Common::PerfReport::GuestSlowReads),
             rate(Common::PerfReport::GuestSlowWrites), rate(Common::PerfReport::GuestClockReads),
             rate(Common::PerfReport::GuestFutileYields));
    before = counters;
    svcs_before = svcs;
    last = now;
}
} // Anonymous namespace

void RasterizerMetal::TickFrame() {
    VideoCommon::FeatureCensus::Tick();
    if (VideoCore::Nx2Debug::HasRequests()) {
        if (const auto capture = VideoCore::Nx2Debug::TakeCapture()) {
            frame_dump.Arm(capture->id, capture->directory, capture->sampled, capture->buffers);
        }
        if (const auto change = VideoCore::Nx2Debug::TakeRejectChange()) {
            pipeline_cache.RefreshDebugRejections(change->unrejected);
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
    flash_watch.TickFrame();
    ++oot_frame;
    if (!graphics_report_logged && oot_frame == 600) {
        graphics_report_logged = true;
        LOG_INFO(Common_Memory, "Graphics memory (renderer): {}", GraphicsMemoryReport());
    }
    attachment_fit_memo.valid = false;
    texel_view_cache.clear();
    fence_manager.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
        size_t stale_removed = 0;
        std::erase_if(converted_index_cache, [&](auto& item) {
            stale_removed += std::erase_if(item.second, [this](const ConvertedIndexEntry& entry) {
                const bool stale = oot_frame - entry.last_frame > 120;
                if (stale) {
                    converted_index_bytes -= entry.buffer_bytes;
                }
                return stale;
            });
            return item.second.empty();
        });
        if (stale_removed != 0) {
            converted_index_count -= stale_removed;
            RefreshConvertedIndexAddrBounds();
        }
    }
    staging_pool.TickFrame();
    pipeline_cache.TickFrame();
    // The frame threads' deadlines follow the frames presented (EDEN_NX2_APPLE_FRAME_DEADLINE).
    // As Vulkan's TickFrame: the guest-thread report ("Perf sched:", EDEN_NX2_PERF_REPORT)
    // and the Apple frame workgroup interval (frame-critical threads scheduled against the
    // frame deadline) were only driven by the Vulkan rasterizer.
    Common::PerfSched::TickFrame();
    Common::TickFrameWorkgroup();
    LogGuestReport();
    Common::FrameDeadline::SetPeriodNs(static_cast<u64>((std::max)(
        VideoCommon::Nx2Perf::g_guest_frame_interval_ns.load(std::memory_order_relaxed), 0LL)));
    Common::FrameDeadline::NoteFrame();
}

bool RasterizerMetal::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerMetal::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale,
                                          use_history);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerMetal::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerMetal::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    // Grid placeholders among the words (a compute macro's constant buffer of its dispatch's
    // size, EDEN_NX2_INDIRECT_GRID): the GPU copies the words they stand for over them once the
    // write is done.
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
            InlineToMemory(address, copy_size, memory);
            // The write's own invalidation first: it would upload the placeholders over the copy.
            gpu_memory->FlushCaching();
            std::scoped_lock lock{buffer_cache.mutex};
            for (const auto& [source, destination, count] : grids) {
                Tegra::Nx2IndirectGrid::Get().copied.fetch_add(1, std::memory_order_relaxed);
                if (!buffer_cache.DMACopy(source, destination, count * sizeof(u32), true)) {
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1) < 8) {
                        LOG_WARNING(Render_Metal,
                                    "GPU-sized dispatch: no host buffer to copy its size from "
                                    "{:#x} to {:#x}",
                                    source, destination);
                    }
                }
            }
            return;
        }
    }
    InlineToMemory(address, copy_size, memory);
}

void RasterizerMetal::InlineToMemory(GPUVAddr address, size_t copy_size,
                                     std::span<const u8> memory) {
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    if (!CachePathsEnabled()) {
        gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
        {
            std::unique_lock lock{buffer_cache.mutex};
            InvalidateConvertedIndicesAddr(*cpu_addr, copy_size);
            if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
                buffer_cache.WriteMemory(*cpu_addr, copy_size);
            }
        }
        {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.WriteMemory(*cpu_addr, copy_size);
        }
        pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
        return;
    }
    // A write within one page mapped as a whole goes straight to guest memory.
    static const bool inline_fast = EnvironmentOn("EDEN_NX2_INLINE_FAST");
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
    static const bool defer = EnvironmentOn("EDEN_NX2_INLINE_DEFER");
    static const bool i2m_coalesce = EnvironmentOn("EDEN_NX2_I2M_COALESCE");
    static const bool per_page_list = EnvironmentOn("EDEN_NX2_I2M_LIST_PAGE");
    bool uploaded = false;
    // The page of the last write, when the buffer cache held no GPU data of it: the next
    // writes into it need not ask again (gpu_modified_serial tells when that changed).
    struct CleanPage {
        const void* cache{};
        DAddr page{~DAddr{0}};
        u64 serial{};
    };
    thread_local CleanPage clean{};
    const DAddr device_page = *cpu_addr & ~static_cast<DAddr>(Core::DEVICE_PAGEMASK);
    const bool one_page =
        copy_size != 0 && (*cpu_addr & Core::DEVICE_PAGEMASK) + copy_size <= Core::DEVICE_PAGESIZE;
    const bool known_clean =
        i2m_coalesce && defer && one_page && clean.cache == &buffer_cache &&
        clean.page == device_page &&
        clean.serial == buffer_cache.gpu_modified_serial.load(std::memory_order_relaxed) &&
        (per_page_list ? !Tegra::Nx2IsCommandListPage(device_page)
                       : !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed));
    if (!known_clean) {
        std::scoped_lock lock{buffer_cache.mutex};
        InvalidateConvertedIndicesAddr(*cpu_addr, copy_size);
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
    if (!uploaded && defer) {
        // Guest memory is written; the buffer, texture and shader caches see the merged ranges
        // at the next FlushCaching, which runs before anything of theirs is used. Writes into
        // GPU-written buffer data were uploaded in place above.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

void RasterizerMetal::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                        const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.CreateChannel(channel);
    buffer_cache.CreateChannel(channel);
    pipeline_cache.CreateChannel(channel);
    // As Vulkan's and OpenGL's state trackers: the register writes that mark render targets,
    // vertex and index buffers, descriptors and shaders dirty. Without them the texture cache kept
    // the first render targets for every later draw (Ocarina's composite into its display buffers
    // drew into the scene image instead, so the screen stayed black).
    VideoCommon::Dirty::SetupDirtyFlags(channel.payload->maxwell_3d.dirty.tables);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.BindToChannel(channel_id);
    buffer_cache.BindToChannel(channel_id);
    pipeline_cache.BindToChannel(channel_id);
    // As Vulkan's StateTracker::InvalidateState on a channel switch: the render targets, buffers,
    // descriptors and shaders the caches hold are the previous channel's, so the new channel's
    // first draw must rebuild all of them even where its own registers did not change.
    channel.payload->maxwell_3d.dirty.flags.set();
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.EraseChannel(channel_id);
    buffer_cache.EraseChannel(channel_id);
    pipeline_cache.EraseChannel(channel_id);
}

std::optional<DisplayTexture> RasterizerMetal::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr) {
    if (framebuffer_addr == 0) {
        return std::nullopt;
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    if (image_view == nullptr) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
            LOG_INFO(Render_Metal,
                     "Metal: display layer at {:#x} ({}x{} stride {} format {} native_tiling {}) "
                     "has no cached image view; reading guest memory",
                     framebuffer_addr, config.width, config.height, config.stride,
                     static_cast<u32>(config.pixel_format), config.native_tiling);
        }
        return std::nullopt;
    }
    if (frame_dump.Capturing()) {
        if (const auto source = DumpSource(*image_view, 0)) {
            frame_dump.CapturePresent(*source);
        }
    }
    if (flash_watch.Enabled()) {
        if (const auto source = FlashSource(*image_view, 0)) {
            flash_watch.NotePresent(*source);
        }
    }
    const auto& resolution = Settings::values.resolution_info;
    DisplayTexture texture{
        .texture = image_view->Handle(Shader::TextureType::Color2D),
        .width = image_view->size.width,
        .height = image_view->size.height,
    };
    texture.scaled_width = scaled ? resolution.ScaleUp(texture.width) : texture.width;
    texture.scaled_height = scaled ? resolution.ScaleUp(texture.height) : texture.height;
    if (texture.texture == nullptr) {
        return std::nullopt;
    }
    return texture;
}

} // namespace Metal
