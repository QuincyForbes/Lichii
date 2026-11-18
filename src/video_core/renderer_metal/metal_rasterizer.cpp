// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
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
        scheduler.Flush();
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
}

RasterizerMetal::~RasterizerMetal() {
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
        scheduler.Flush();
        return;
    }
    if (!staging_pool.WantsFlush()) {
        return;
    }
    ++staging_flushes;
    const u64 tick = scheduler.Flush();
    if (staging_pool.OverCap()) {
        ++staging_waits;
        scheduler.Wait(tick);
        staging_pool.Trim();
    }
}

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    ++draws;
    skip_valid = false;
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
            Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
    if (convert && is_indexed && draw_state.inline_index_draw_indexes.empty() &&
        gpu_indices != '0') {
        const GPUVAddr start = draw_state.index_buffer.StartAddress() +
                               GPUVAddr{draw_state.index_buffer.first} * index_size;
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(start);
        index_rewrite.wanted =
            gpu_indices == '1' ||
            (device_addr &&
             buffer_cache.IsRegionGpuModified(*device_addr, size_t{count} * index_size));
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
        // 32-bit indices in the stream ring: from the GPU rewrite, or rewritten on the CPU from
        // the guest's indices (or the vertex range).
        StagingBufferRef indices = index_rewrite.output;
        if (indices.buffer != nullptr) {
            ++indices_rewritten_on_gpu;
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
            indices = staging_pool.Stream(size_t{converted} * sizeof(u32), 4);
            const std::span<u32> out(reinterpret_cast<u32*>(indices.mapped_span.data()),
                                     converted);
            converted = IndexConvert::Convert(DrawTopology(*maxwell3d), source, index_size,
                                              is_indexed ? 0U : draw_state.vertex_buffer.first,
                                              count, restart, out, provoking_last);
        }
        if (converted != 0 &&
            !conditional(*primitive, true, converted, 4, indices.buffer, indices.offset, 0)) {
            scheduler.DrawIndexed(*primitive, converted, 4, indices.buffer, indices.offset,
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
        buffer_cache.SetDrawIndirect(&params);
        const bool drawn =
            DrawMesh(pipeline, params.is_indexed, MeshGroups{.draws = mesh_indirect.draws});
        buffer_cache.SetDrawIndirect(nullptr);
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
            Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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

void RasterizerMetal::BindStageResources(StageProgram& program, ShaderStage metal_stage,
                                         std::span<const Binding> pending,
                                         std::span<const TextureHandle> texel_buffer_views,
                                         std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         u32 render_width, u32 render_height) {
    const Shader::Info& info = program.info;
    std::vector<u32> texture_slots;
    std::vector<u32> sampler_slots;
    std::vector<u64> spill_table;
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
                    // Shader::Backend::SPIRV::RescalingLayout (rescaling_words) or
                    // RenderAreaLayout.
                    std::array<u32, 8> words = rescaling_words;
                    if (info.uses_render_area && !info.uses_rescaling_uniform) {
                        words = {};
                        words[0] = std::bit_cast<u32>(static_cast<f32>(render_width));
                        words[1] = std::bit_cast<u32>(static_cast<f32>(render_height));
                    }
                    scheduler.BindBytes(metal_stage, slot_index, words.data(),
                                        static_cast<u32>(sizeof(words)));
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
                texture_slots.push_back(slot_index);
                break;
            case ShaderTranslate::ResourceKind::Sampler:
                sampler_slots.push_back(slot_index);
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
        scheduler.BindTexture(metal_stage, texture_slots[element],
                              element < texel_buffer_views.size() ? texel_buffer_views[element]
                                                                  : nullptr);
    }
    size_t texture_slot = texel_buffers;
    size_t view = texel_buffers;
    size_t sampler = 0;
    u64 reduction_modes = 0;
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            ImageView& image_view = texture_cache.GetImageView(views[view++].id);
            if (texture_slot < texture_slots.size()) {
                scheduler.BindTexture(
                    metal_stage, texture_slots[texture_slot],
                    OrPlaceholder(image_view, image_view.SampledHandle(desc.type,
                                                                       desc.is_multisample),
                                  desc.type));
            }
            ++texture_slot;
            if (sampler < sampler_slots.size()) {
                scheduler.BindSampler(
                    metal_stage, sampler_slots[sampler],
                    texture_cache.GetSampler(samplers[sampler])
                        .HandleFor(image_view, desc.is_depth));
            }
            if (sampler < sampler_slots.size() && sampler_slots[sampler] < 16) {
                reduction_modes |= texture_cache.GetSampler(samplers[sampler]).ReductionBits()
                                   << (sampler_slots[sampler] * 4);
            }
            ++sampler;
        }
    }
    if (program.reduction_slot != ~0U) {
        scheduler.BindBytes(metal_stage, program.reduction_slot, &reduction_modes,
                            static_cast<u32>(sizeof(reduction_modes)));
    }
    for (const auto& desc : info.image_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            const size_t view_index = view++;
            ImageView& image_view = texture_cache.GetImageView(views[view_index].id);
            if (desc.is_written) {
                texture_cache.MarkModification(image_view.image_id);
            }
            if (texture_slot < texture_slots.size()) {
                scheduler.BindTexture(
                    metal_stage, texture_slots[texture_slot],
                    view_index == byte_shadow_view
                        ? byte_shadow_texture
                        : OrPlaceholder(image_view,
                                        image_view.StorageHandle(desc.type, desc.format),
                                        desc.type));
            }
            ++texture_slot;
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
    const std::array<size_t, 4> stages{
        0, has_task ? size_t{mesh_stage} : tess != nullptr ? size_t{1} : geo != nullptr ? size_t{3} : 0,
        tess != nullptr ? size_t{2} : 0, 4};
    const std::array<StageProgram*, 4> programs{
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
        native != nullptr ? &native->fragment : &pipeline.Fragment()};
    prepare_refusal = {};
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
    std::vector<VideoCommon::ImageViewInOut> views;
    std::vector<VideoCommon::SamplerId> samplers;
    std::array<size_t, 4> first_view{};
    std::array<size_t, 4> first_sampler{};
    // EDEN_NX2_BINDMEMO: the sampler handles and binding kinds, to look the bindings up in the
    // texture cache's memo.
    const bool use_bind_memo = BindMemo::Enabled();
    std::vector<u32> sampler_handles;
    std::vector<u32> binding_kinds;
    const bool via_header_index = regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding;
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
        for (const auto& desc : info.storage_buffers_descriptors) {
            buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index++, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.is_written);
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        const auto read_handle = [&](const auto& desc, u32 element) {
            if (desc.handle_program != ~0U) {
                return Tegra::Texture::TexturePair(
                    VideoCommon::EvaluateTextureHandle(
                        info, desc.handle_program, *gpu_memory,
                        [&](u32 cbuf) -> GPUVAddr {
                            return cbuf < cbufs.size() && cbufs[cbuf].enabled
                                       ? cbufs[cbuf].address
                                       : GPUVAddr{};
                        },
                        buffer_cache_runtime.GetStorageBufferAlignment()),
                    via_header_index);
            }
            const u32 offset = desc.cbuf_offset + (element << desc.size_shift);
            const GPUVAddr addr = cbufs[desc.cbuf_index].address + offset;
            if constexpr (requires { desc.has_secondary; }) {
                if (desc.has_secondary) {
                    const GPUVAddr second = cbufs[desc.secondary_cbuf_index].address +
                                            desc.secondary_cbuf_offset;
                    const u32 raw = (gpu_memory->Read<u32>(addr) << desc.shift_left) |
                                    (gpu_memory->Read<u32>(second) << desc.secondary_shift_left);
                    return Tegra::Texture::TexturePair(raw, via_header_index);
                }
            }
            return Tegra::Texture::TexturePair(gpu_memory->Read<u32>(addr), via_header_index);
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                views.push_back({.index = read_handle(desc, element).first});
            }
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                views.push_back({.index = read_handle(desc, element).first});
            }
        }
        for (const auto& desc : info.texture_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                const auto handle = read_handle(desc, element);
                views.push_back({.index = handle.first});
                if (use_bind_memo) {
                    sampler_handles.push_back(handle.second);
                    samplers.emplace_back();
                } else {
                    samplers.push_back(texture_cache.GetSamplerId(handle.second, false));
                }
            }
        }
        for (const auto& desc : info.image_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                views.push_back(
                    {.index = read_handle(desc, element).first, .blacklist = desc.is_written});
            }
        }
        if (use_bind_memo) {
            BindMemo::AppendKinds(info, 0, binding_kinds);
        }
    }
    if (use_bind_memo && binding_kinds.size() == views.size()) {
        BindMemo::Resolve(texture_cache, false, true, binding_kinds, views, sampler_handles,
                          samplers, bind_memo_counts);
    } else if (use_bind_memo) {
        for (size_t index = 0; index < sampler_handles.size(); ++index) {
            samplers[index] = texture_cache.GetSamplerId(sampler_handles[index], false);
        }
        texture_cache.FillImageViews(views, false, true);
    } else {
        texture_cache.FillImageViews(views, false, true);
    }
    for (size_t index = 0; index < stages.size(); ++index) {
        if (!absent(index)) {
            PrepareSampleGrids(programs[index]->info, std::span(views).subspan(first_view[index]));
        }
    }
    texture_cache.UpdateRenderTargets(false);
    TrackOcarinaCamera();
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
    buffer_cache.UpdateGraphicsBuffers(is_indexed);
    buffer_cache.BindHostGeometryBuffers(is_indexed);
    if (mesh_indirect.wanted) {
        const auto [records, records_offset] = buffer_cache.GetDrawIndirectBuffer();
        mesh_indirect.records = records->Handle();
        mesh_indirect.records_offset = records_offset;
        if (mesh_indirect.has_count) {
            const auto [count, count_offset] = buffer_cache.GetDrawIndirectCount();
            mesh_indirect.count = count->Handle();
            mesh_indirect.count_offset = count_offset;
        }
    }

    // Each stage's buffer bindings, and texel buffer views (copies of misaligned ranges go
    // before the render pass).
    std::array<std::vector<Binding>, 4> stage_bindings;
    std::array<std::vector<TextureHandle>, 4> texel_views;
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        buffer_cache_runtime.PendingBindings().clear();
        buffer_cache.BindHostStageBuffers(stages[index]);
        stage_bindings[index] = buffer_cache_runtime.PendingBindings();
        texel_views[index] = PrepareTexelBuffers(stage_bindings[index], programs[index]->info);
    }
    // Until the state below is made.
    prepare_refusal = "render pipeline state refused by Metal";

    // The render pipeline state: targets, blending, vertex layout.
    auto state_desc = Zeroed<RenderPipelineDesc>();
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        state_desc.color_formats[rt] = framebuffer->Colors()[rt].texture != nullptr
                                     ? framebuffer->ColorFormats()[rt]
                                     : 0U;
    }
    state_desc.depth_format = targets.has_depth ? framebuffer->DepthFormat() : 0U;
    state_desc.stencil_format = targets.has_stencil ? framebuffer->DepthFormat() : 0U;
    state_desc.samples = framebuffer->Samples();
    if (PipelineCheckEnabled()) {
        FitAttachments(state_desc, targets, DescribeTexture);
    }
    FillRegisterState(state_desc, regs, topology, mesh_pipeline);
    const auto& vertex_states = buffer_cache_runtime.Vertices();
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        FillStream(state_desc, regs, index,
                   vertex_states[index].buffer != nullptr && !mesh_pipeline,
                   vertex_states[index].stride);
    }
    // The guest's layout as Metal takes it: Metal's vertex descriptor has 31 attributes and 31
    // buffer layouts and aborts the process when asked for more (Ocarina of Time's first draw has
    // stream 31 bound), so only what the vertex function reads goes in, numbered from 0.
    const bool checked = PipelineCheckEnabled();
    VertexSources vertex_sources{};
    if (checked) {
        const VertexProgramUse use =
            !mesh_pipeline && !absent(0) ? VertexUse(*programs[0]) : VertexProgramUse{};
        std::string reason;
        if (!SanitizeVertexLayout(state_desc, use.loaded, use.resource_slots, vertex_sources,
                                  reason)) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal,
                            "Metal: render pipeline {:016x} refused: {}; fragment {:016x}, {}",
                            pipeline.Key().unique_hashes[1], reason,
                            pipeline.Key().unique_hashes[5], Describe(state_desc));
            }
            return false;
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
    // A tessellation pipeline's vertex kernel reads the vertex layout as stage-in data; its
    // post-tessellation vertex function reads buffers.
    const RenderPipelineDesc layout_desc = state_desc;
    if (geo != nullptr) {
        // The render pass draws from the reader's buffer: no vertex streams.
        std::ranges::fill(state_desc.attributes, VertexAttributeState{});
        std::ranges::fill(state_desc.streams, VertexStreamState{});
    }
    if (tess != nullptr) {
        ApplyTessellation(state_desc, *tess);
    }
    if (xfb_draw.wanted && xfb_draw.expand) {
        // The expansion's vertex function reads the kernel's outputs: no vertex layout, drawn as
        // triangles.
        std::ranges::fill(state_desc.attributes, VertexAttributeState{});
        std::ranges::fill(state_desc.streams, VertexStreamState{});
        state_desc.topology_class = 2;
        state_desc.expand = 1;
    }
    bool state_building = false;
    void* state = pipeline_cache.RenderState(pipeline, state_desc, state_building);
    if (state == nullptr && !state_building && native == nullptr && PipelineCheckEnabled()) {
        // A state Metal refused is drawn without blending and alpha to coverage, what a target
        // format most often refuses, rather than leaving a hole.
        RenderPipelineDesc fallback = state_desc;
        for (BlendTargetState& blend : fallback.blend) {
            blend.enable = 0;
        }
        fallback.alpha_to_coverage = 0;
        state = pipeline_cache.RenderState(pipeline, fallback, state_building);
        if (state != nullptr) {
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
    prepare_refusal = {};

    const DepthStencilDesc depth_stencil =
        DepthStencilState(regs, targets.has_depth, targets.has_stencil);
    const u64 depth_stencil_hash = Common::CityHash64(
        reinterpret_cast<const char*>(&depth_stencil), sizeof(depth_stencil));
    Object& depth_stencil_state = depth_stencil_states[depth_stencil_hash];
    if (!depth_stencil_state) {
        depth_stencil_state = CreateDepthStencilState(device, depth_stencil);
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
        RewriteIndices(record_ahead(false));
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
        } else {
            mesh_buffers.output = staging_pool.Stream(
                (std::max)(u64{mesh_workgroups} * mesh->block_words * sizeof(u32), u64{16}), 16);
        }
        // After the arena: getting one may wait for a command buffer.
        const bool mesh_ahead = record_ahead(true);
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
        std::array<u32, 4> domain_params{};
        if (tess->domain) {
            const ShaderTranslate::DomainEmulation& emulation = tess->domain->emulation;
            // As many evaluated vertices as the patches can have, within a bound of memory
            // (the count kernel truncates what does not fit).
            constexpr u64 MaxDomainBytes = u64{8} << 20;
            const u32 capacity = static_cast<u32>((std::max)(
                (std::min)(u64{patches} * emulation.max_vertices_per_patch,
                           MaxDomainBytes / emulation.vertex_stride),
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
    if (geo != nullptr) {
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
        const u64 output_words = u64{rounded} * plan.invocations * plan.max_vertices * plan.OutputStride();
        const u64 index_words = u64{rounded} * plan.invocations * plan.IndexStride();
        const u64 list_words = u64{groups} * plan.max_vertices * 3;
        const u64 total = u64{vertices} * geo->vertex_stride + (record_words + output_words +
                                                                  index_words + list_words +
                                                                  groups) * 4;
        if (input_primitives == 0) {
            prepare_refusal = {};
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
            .stride = plan.IndexStride(),
            .topology = plan.output_topology,
            .capacity = static_cast<u32>(list_words),
        };
        const auto pack = [&](const Object& kernel, u32 threads) {
            scheduler.BeginCompute(kernel.Get(), false);
            bind(0, regions);
            bind(1, geo_draw.indices);
            bind(2, geo_draw.args);
            bind(3, counts);
            scheduler.BindBytes(ShaderStage::Compute, 4, &params, static_cast<u32>(sizeof(params)));
            scheduler.Dispatch({(threads + Geometry::GroupSize - 1) / Geometry::GroupSize, 1, 1},
                               {Geometry::GroupSize, 1, 1});
        };
        pack(geometry_count_kernel, groups);
        pack(geometry_scan_kernel, 1);
        pack(geometry_write_kernel, groups);
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
    if (flash_watch.WatchesPasses()) {
        FlashWatchPass(framebuffer, {&pipeline.Key().unique_hashes[1], 5});
    }
    if (frame_dump.Capturing()) {
        FrameDumpPass(framebuffer, {&pipeline.Key().unique_hashes[1], 5});
        if (frame_dump.CapturingSampled()) {
            if (frame_dump.CapturingTextures()) {
                std::vector<FrameDump::Sampled> sampled;
                for (size_t index = 0; index < stages.size(); ++index) {
                    if (absent(index)) {
                        continue;
                    }
                    const Shader::Info& info = programs[index]->info;
                    size_t view = first_view[index];
                    for (const auto& desc : info.texture_buffer_descriptors) {
                        view += desc.count;
                    }
                    for (const auto& desc : info.image_buffer_descriptors) {
                        view += desc.count;
                    }
                    size_t sampler = first_sampler[index];
                    for (const auto& desc : info.texture_descriptors) {
                        for (u32 element = 0; element < desc.count; ++element, ++view, ++sampler) {
                            if (view < views.size() && sampler < samplers.size()) {
                                sampled.push_back(DumpSampled(static_cast<u32>(stages[index]),
                                                              views[view].index, views[view].id,
                                                              samplers[sampler]));
                            }
                        }
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
    if (writes_storage) {
        pass_stores = scheduler.RenderPassSerial();
    }
    DrawState draw{};
    draw.pipeline = state;
    draw.depth_stencil = depth_stencil_state.Get();
    for (size_t index = 0; index < state_desc.blend.size(); ++index) {
        if (state_desc.blend[index].write_mask != 0) {
            draw.color_written |= static_cast<u8>(1U << index);
        } else {
            draw.color_written &= static_cast<u8>(~(1U << index));
        }
    }
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
            viewport.x = static_cast<f32>(regs.surface_clip.x);
            viewport.y = static_cast<f32>(regs.surface_clip.y);
            viewport.width = static_cast<f32>(regs.surface_clip.width);
            viewport.height = static_cast<f32>(regs.surface_clip.height);
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
        const auto scale = [&](u32 value) {
            return is_rescaling ? resolution.ScaleUp(value) : value;
        };
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
    if (all_viewports && pipeline.Vertex().info.stores[Shader::IR::Attribute::ViewportIndex]) {
        draw.viewport_count = static_cast<u8>(Maxwell::NumViewports);
        for (size_t index = 1; index < Maxwell::NumViewports; ++index) {
            draw.more_viewports[index - 1] = make_viewport(index);
            draw.more_scissors[index - 1] = make_scissor(index);
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
            if (TopologyClass(topology) == 2) {
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
    if (regs.polygon_offset_fill_enable != 0 || regs.polygon_offset_line_enable != 0 ||
        regs.polygon_offset_point_enable != 0) {
        // The hardware halves the depth bias units (as OpenGL and Vulkan apply it).
        draw.depth_bias = regs.depth_bias / 2.0f;
        draw.slope_scale = regs.slope_scale_depth_bias;
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
    if (tess != nullptr && tess->domain) {
        // The evaluated vertices the vertex function draws.
        scheduler.BindBuffer(ShaderStage::Vertex, 0, tess_buffers.captured.buffer,
                             static_cast<u32>(tess_buffers.captured.offset));
    }

    // Each stage's buffers (appended by the buffer cache in binding order), textures, samplers
    // and images, into the slots the translator gave them.
    for (size_t index = 0; index < stages.size(); ++index) {
        // Kernels were bound when they were dispatched.
        if (absent(index) || (index == 1 && native == nullptr) ||
            (index == 0 && (tess != nullptr || geo != nullptr)) ||
            (index == 2 && tess != nullptr && tess->domain)) {
            continue;
        }
        ShaderStage metal_stage = index == 3 ? ShaderStage::Fragment : ShaderStage::Vertex;
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
    geometry_count_kernel = make("geo_count");
    geometry_scan_kernel = make("geo_scan");
    geometry_write_kernel = make("geo_write");
    if (!geometry_count_kernel || !geometry_scan_kernel || !geometry_write_kernel) {
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
    geometry_input_primitives = geo.plan.Primitives(count) * instance_count;
    CountDraw(count, instance_count);
    geometry_input_primitives = 0;
    if (geo.plan.Primitives(count) == 0 || instance_count == 0) {
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
        return;
    }
    const std::optional<u32> primitive = PrimitiveType(
        geo.plan.output_topology == 1   ? Maxwell::PrimitiveTopology::Points
        : geo.plan.output_topology == 6 ? Maxwell::PrimitiveTopology::Lines
                                        : Maxwell::PrimitiveTopology::Triangles);
    if (!primitive || !ConditionHoldsOnCpu()) {
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
    if (!PrepareDraw(pipeline, is_indexed)) {
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
    u32 draws = static_cast<u32>(params.max_draw_counts);
    if (params.include_count) {
        draws = (std::min)(draws, gpu_memory->Read<u32>(params.count_start_address));
    }
    for (u32 draw = 0; draw < draws; ++draw) {
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
    if (!OcarinaTemporal::Enabled() ||
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
        Coverage::Note(Coverage::Work::Draw, Coverage::Path::Empty);
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
        const bool is_integer = drawn_clears && VideoCore::Surface::IsPixelFormatInteger(format);
        const f32 bits =
            static_cast<f32>(VideoCore::Surface::PixelComponentSizeBitsInteger(format));
        for (size_t component = 0; component < 4; ++component) {
            const f32 value = regs.clear_color[component];
            // Integer targets: the values Vulkan's Clear writes.
            if (!is_integer) {
                color_value[component] = value;
            } else if (!VideoCore::Surface::IsPixelFormatSignedInteger(format)) {
                color_value[component] = static_cast<u32>(bits * 2.0f * value);
            } else {
                color_value[component] = static_cast<s32>((bits - 1.0f) * 2.0f * (value - 0.5f));
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
    const auto read_handle = [&](const auto& desc, u32 element) {
        if (desc.handle_program != ~0U) {
            return Tegra::Texture::TexturePair(
                VideoCommon::EvaluateTextureHandle(
                    info, desc.handle_program, *gpu_memory,
                    [&](u32 cbuf) -> GPUVAddr {
                        return cbuf < cbufs.size() && ((qmd.const_buffer_enable_mask >> cbuf) & 1)
                                   ? cbufs[cbuf].Address()
                                   : GPUVAddr{};
                    },
                    buffer_cache_runtime.GetStorageBufferAlignment()),
                via_header_index);
        }
        const GPUVAddr addr =
            cbufs[desc.cbuf_index].Address() + desc.cbuf_offset + (element << desc.size_shift);
        if constexpr (requires { desc.has_secondary; }) {
            if (desc.has_secondary) {
                const GPUVAddr second =
                    cbufs[desc.secondary_cbuf_index].Address() + desc.secondary_cbuf_offset;
                const u32 raw = (gpu_memory->Read<u32>(addr) << desc.shift_left) |
                                (gpu_memory->Read<u32>(second) << desc.secondary_shift_left);
                return Tegra::Texture::TexturePair(raw, via_header_index);
            }
        }
        return Tegra::Texture::TexturePair(gpu_memory->Read<u32>(addr), via_header_index);
    };
    std::vector<VideoCommon::ImageViewInOut> views;
    std::vector<VideoCommon::SamplerId> samplers;
    const bool use_bind_memo = BindMemo::Enabled();
    std::vector<u32> sampler_handles;
    for (const auto& desc : info.texture_buffer_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back({.index = read_handle(desc, element).first});
        }
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back({.index = read_handle(desc, element).first});
        }
    }
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            const auto handle = read_handle(desc, element);
            views.push_back({.index = handle.first});
            if (use_bind_memo) {
                sampler_handles.push_back(handle.second);
                samplers.emplace_back();
            } else {
                samplers.push_back(texture_cache.GetSamplerId(handle.second, true));
            }
        }
    }
    for (const auto& desc : info.image_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back(
                {.index = read_handle(desc, element).first, .blacklist = desc.is_written});
        }
    }
    if (use_bind_memo) {
        std::vector<u32> binding_kinds;
        BindMemo::AppendKinds(info, TextureCache::BINDING_KIND_COMPUTE, binding_kinds);
        if (binding_kinds.size() == views.size()) {
            BindMemo::Resolve(texture_cache, true, true, binding_kinds, views, sampler_handles,
                              samplers, bind_memo_counts);
        } else {
            for (size_t index = 0; index < sampler_handles.size(); ++index) {
                samplers[index] = texture_cache.GetSamplerId(sampler_handles[index], true);
            }
            texture_cache.FillImageViews(views, true);
        }
    } else {
        texture_cache.FillImageViews(views, true);
    }
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
    const std::vector<Binding> bindings = buffer_cache_runtime.PendingBindings();
    const std::vector<TextureHandle> texel_views = PrepareTexelBuffers(bindings, info);

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
            VideoCommon::ObtainBufferOperation::DiscardWrite);
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
                                                .shader_write = true});
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
    scheduler.BeginCompute(state);
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
            size_t sampled_view = 0;
            for (const auto& desc : info.texture_buffer_descriptors) {
                sampled_view += desc.count;
            }
            for (const auto& desc : info.image_buffer_descriptors) {
                sampled_view += desc.count;
            }
            size_t sampled_sampler = 0;
            for (const auto& desc : info.texture_descriptors) {
                for (u32 element = 0; element < desc.count;
                     ++element, ++sampled_view, ++sampled_sampler) {
                    if (sampled_view < views.size() && sampled_sampler < samplers.size()) {
                        sampled.push_back(DumpSampled(5, views[sampled_view].index,
                                                      views[sampled_view].id,
                                                      samplers[sampled_sampler]));
                    }
                }
            }
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
        scheduler.Wait(visibility_ticks[slot]);
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
    scheduler.Wait(visibility_ticks[pixel_slots.back()]);
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
        scheduler.Wait(readback.tick);
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
    // A condition that reads the report before its fence decides on the GPU from these slots.
    const u64 serial = ++report_serial;
    {
        std::scoped_lock lock{pending_reports_mutex};
        pending_reports.insert_or_assign(*address, PendingReport{.slots = occlusion_slots,
                                                                 .wide = wide,
                                                                 .tick = scheduler.CurrentTick(),
                                                                 .serial = serial});
    }
    // Written once the GPU has finished the draws counted so far (the fence's command buffer
    // ends their render pass, so later draws count into new slots).
    SignalFence([this, address = *address, wide, serial, slots = occlusion_slots] {
        const u64* const results = scheduler.VisibilityResults();
        u64 samples = 0;
        for (const u32 slot : slots) {
            samples += results[slot];
        }
        WriteQueryResult(address, wide, samples);
        std::scoped_lock lock{pending_reports_mutex};
        if (const auto it = pending_reports.find(address);
            it != pending_reports.end() && it->second.serial == serial) {
            pending_reports.erase(it);
        }
    });
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
    std::function<void()> func([this, address = *address, wide, value] {
        WriteQueryResult(address, wide, value);
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
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::SubmitListCopies(u64 writer) {
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick(), writer)) {
        scheduler.Flush();
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
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    if (!CachePathsEnabled()) {
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (buffer_cache.OnCPUWrite(addr, size)) {
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
    {
        std::unique_lock lock{buffer_cache.mutex, std::adopt_lock};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
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
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

bool RasterizerMetal::FencesQuiet(bool, u64& epoch) {
    // Always strict: this backend's fences are only released by ReleaseFences.
    return CachePathsEnabled() && fence_manager.FencesQuiet(true, epoch);
}

u64 RasterizerMetal::FenceWorkEpoch() const {
    return CachePathsEnabled() ? fence_manager.FenceWorkEpoch() : 0;
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {
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

void RasterizerMetal::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
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

void RasterizerMetal::TiledCacheBarrier() {}

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
                scheduler.Wait(tick);
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
                scheduler.Wait(tick);
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
    if (scheduler.HasPendingWork()) {
        scheduler.Flush();
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
    texel_view_cache.clear();
    fence_manager.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
    staging_pool.TickFrame();
    pipeline_cache.TickFrame();
    // The frame threads' deadlines follow the frames presented (EDEN_NX2_APPLE_FRAME_DEADLINE).
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
        // Guest memory has the words now: a command list copy of the page no longer does.
        buffer_cache.Nx2NoteListHostWrite(*cpu_addr, copy_size);
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
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.BindToChannel(channel_id);
    buffer_cache.BindToChannel(channel_id);
    pipeline_cache.BindToChannel(channel_id);
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
