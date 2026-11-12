// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/perf_report.h"
#include "common/platform_events.h"
#include "video_core/churn_counters.h"
#include "video_core/gpu_split.h"
#include "video_core/host_budget.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
namespace {

using namespace Common::Literals;

// Maximum potential alignment of a Vulkan buffer
constexpr VkDeviceSize MAX_ALIGNMENT = 256;

// Stream buffer size in bytes
// *NIX drivers are more sensitive to increased buffers for streaming.
// Windows ones however, can intake bigger buffers and generally do not OOM.
// - GTX 960 on Windows will not OOM with 256mib
// - GT 1030 on ^NIX will OOM with 256mib
#if defined(__FreeBSD__)
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 128_MiB;
#elif defined(__APPLE__) && TARGET_OS_IPHONE
// iOS counts all of it in the footprint once the ring has gone round; uploads past a region (2 MiB)
// take pooled staging buffers instead. 32 MiB (and 16) cost no frame rate on desktop, where
// decoded ASTC uploads several times the bytes the phone does.
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 32_MiB;
#else
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 256_MiB;
#endif

// Unused space after the end of the stream buffer. Constant buffers are bound from it and the
// shaders declare each one at its full 64 KiB, so Metal checks 64 KiB from the binding offset
// against the end of the buffer ("argument c3[0] ... has space for 26624 bytes, but argument has a
// length(65536)"). The tail is never handed out, so every binding has the full 64 KiB behind it.
constexpr VkDeviceSize STREAM_BUFFER_TAIL = 64_KiB;

VkDeviceSize GetReBarHeapSize(const Device& device) {
    VkDeviceSize rebar_size = 0;
    const auto memory_props = device.GetPhysical().GetMemoryProperties().memoryProperties;
    constexpr VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (size_t i = 0; i < memory_props.memoryTypeCount; ++i) {
        const auto& type = memory_props.memoryTypes[i];
        if ((type.propertyFlags & required) == required) {
            rebar_size = (std::max)(rebar_size, memory_props.memoryHeaps[type.heapIndex].size);
        }
    }
    return rebar_size;
}

bool CheckReBarSupport(const Device& device) {
    return GetReBarHeapSize(device) >= 1_GiB;
}

bool IsRebarUploadEnabled(const Device& device) {
    const char* env = std::getenv("EDEN_NX2_REBAR_UPLOAD");
    if (env != nullptr) {
        return std::strcmp(env, "0") != 0;
    }
    return CheckReBarSupport(device);
}

size_t GetStreamBufferSize(const Device& device) {
    // EDEN_NX2_STREAM_BUFFER_MIB=<n>: another size, for measuring what it costs.
    if (const char* const env = std::getenv("EDEN_NX2_STREAM_BUFFER_MIB")) {
        if (const long mib = std::strtol(env, nullptr, 10); mib >= 4 && mib <= 1024) {
            return static_cast<size_t>(mib) << 20;
        }
    }
    if (!device.HasDebuggingToolAttached()) {
        if (device.IsIntegrated()) {
            // An integrated GPU's device-local host-visible heap can be no bigger than the stream
            // buffer (Radeon 780M: 256 MiB, 243 MiB budget), which then sat over that budget.
            // Leave a quarter of it free.
            VkDeviceSize heap_size = 0;
            ForEachDeviceLocalHostVisibleHeap(
                device, [&heap_size](size_t, VkMemoryHeap& heap) {
                    heap_size = (std::max)(heap_size, heap.size);
                });
            if (heap_size != 0 && heap_size <= MAX_STREAM_BUFFER_SIZE) {
                return Common::AlignUp(heap_size * 3 / 4, MAX_ALIGNMENT);
            }
        }
        return MAX_STREAM_BUFFER_SIZE;
    }

    VkDeviceSize size{0};
    bool has_device_local_host_visible_heap{};
    ForEachDeviceLocalHostVisibleHeap(device, [&size, &has_device_local_host_visible_heap](
                                                  size_t index, VkMemoryHeap& heap) {
        has_device_local_host_visible_heap = true;
        size = (std::max)(size, heap.size);
    });
    if (has_device_local_host_visible_heap) {
        // If rebar is not supported, cut the max heap size to 40%. This will allow 2 captures to be
        // loaded at the same time in RenderDoc. If rebar is supported, this shouldn't be an issue
        // as the heap will be much larger.
        if (size <= MAX_STREAM_BUFFER_SIZE) {
            size = size * 40 / 100;
        }
    } else {
        size = MAX_STREAM_BUFFER_SIZE;
    }
    return (std::min)(Common::AlignUp(size, MAX_ALIGNMENT), MAX_STREAM_BUFFER_SIZE);
}

// A scene load records thousands of uploads into one command buffer: the policy in
// RasterizerVulkan::FlushWork does not submit inside render passes, so the stream buffer's
// regions all belong to the command buffer being recorded and every upload past them took a
// pooled buffer of its own (iPhone, forest load: 8460 buffers, 1418 MiB, then out of memory).
// Submitting once a command buffer's pooled uploads pass the first mark lets the GPU free them;
// past the second (all pooled upload buffers, busy or idle), the GPU thread waits.
// With room to spare (VideoCommon::HostBudget) both marks rise, up to a quarter of the room past
// the reserve: every early submit is a submission of its own and ends the render pass.
constexpr u64 IOS_UPLOAD_SUBMIT_MIB = 24;
constexpr u64 IOS_UPLOAD_CAP_MIB = 96;
constexpr u64 IOS_UPLOAD_MAX_CAP = 256_MiB;
#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr u64 UPLOAD_SUBMIT_MIB = IOS_UPLOAD_SUBMIT_MIB;
constexpr u64 UPLOAD_CAP_MIB = IOS_UPLOAD_CAP_MIB;
#else
constexpr u64 UPLOAD_SUBMIT_MIB = 256;
constexpr u64 UPLOAD_CAP_MIB = 1024;
#endif

u64 UploadBoundBytes(const char* name, u64 default_mib) {
    // EDEN_NX2_STAGING_SUBMIT_MIB / EDEN_NX2_STAGING_CAP_MIB: other marks; SUBMIT 0 turns it off.
    if (const char* const env = std::getenv(name)) {
        const long mib = std::strtol(env, nullptr, 10);
        if (mib >= 0) {
            return static_cast<u64>(mib) << 20;
        }
    }
    return default_mib << 20;
}
} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, MemoryAllocator& memory_allocator_,
                                     Scheduler& scheduler_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      stream_buffer_size{GetStreamBufferSize(device)}, region_size{stream_buffer_size /
                                                                   StagingBufferPool::NUM_SYNCS} {
    use_rebar = IsRebarUploadEnabled(device);
    const VkDeviceSize rebar_heap = GetReBarHeapSize(device);
    LOG_INFO(Render_Vulkan,
             "PCIe ReBAR upload staging: {} (device-local host-visible coherent heap: {} MiB)",
             use_rebar ? "enabled" : "disabled", rebar_heap / 1_MiB);

    VkBufferCreateInfo stream_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = stream_buffer_size + STREAM_BUFFER_TAIL,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    {
        const MemoryTagScope memory_tag{MemoryTag::StreamBuffer};
        stream_buffer = memory_allocator.CreateBuffer(stream_ci, MemoryUsage::Stream);
    }
    if (device.HasDebuggingToolAttached()) {
        stream_buffer.SetObjectNameEXT("Stream Buffer");
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_buffer_address = device.GetLogical().GetBufferDeviceAddress(*stream_buffer);
    }
    stream_pointer = stream_buffer.Mapped();
    ASSERT_MSG(!stream_pointer.empty(), "Stream buffer must be host visible!");

    current_chunk.offset.store(0, std::memory_order_relaxed);
    current_chunk.limit.store(0, std::memory_order_relaxed);
    current_chunk.blocked_tick.store(0, std::memory_order_relaxed);

    const bool phone = VideoCommon::HostBudget::Simulated();
    upload_submit_bytes = UploadBoundBytes("EDEN_NX2_STAGING_SUBMIT_MIB",
                                           phone ? IOS_UPLOAD_SUBMIT_MIB : UPLOAD_SUBMIT_MIB);
    upload_cap_bytes =
        (std::max)(UploadBoundBytes("EDEN_NX2_STAGING_CAP_MIB",
                                    phone ? IOS_UPLOAD_CAP_MIB : UPLOAD_CAP_MIB),
                   upload_submit_bytes * 2);
    configured_submit_bytes = upload_submit_bytes;
    configured_cap_bytes = upload_cap_bytes;
    LOG_INFO(Common_Memory,
             "Staging: stream buffer {} MiB; uploads submit every {} MiB, pooled upload buffers "
             "capped at {} MiB",
             stream_buffer_size >> 20, upload_submit_bytes >> 20, upload_cap_bytes >> 20);
}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, MemoryUsage usage, bool deferred) {
    if (usage != MemoryUsage::Download) {
        Common::PerfReport::Add(Common::PerfReport::StagingUploadBytes, size);
    }
    if (!deferred && usage == MemoryUsage::Upload && size <= region_size) {
        return GetStreamBuffer(size);
    }
    return GetStagingBuffer(size, usage, deferred);
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    auto& cache_level = GetCache(ref.usage)[ref.log2_level];
    auto& entries = cache_level.entries;
    const auto is_this_one = [&ref](const StagingBuffer& entry) {
        return entry.index == ref.index;
    };
    auto it = std::find_if(entries.begin(), entries.end(), is_this_one);
    ASSERT(it != entries.end());
    ASSERT(it->deferred);
    it->tick = scheduler.CurrentTick();
    it->deferred = false;
    cache_level.known_busy_gpu_tick = ~u64{0};
}

void StagingBufferPool::TickFrame() {
    static constexpr size_t LARGE_BUFFER_LOG2 = 22; // 4 MiB
    current_delete_level = (current_delete_level + 1) % LARGE_BUFFER_LOG2;

    // Host memory pressure (iOS): every idle buffer goes now rather than a level every frame.
    if (const u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
        pressure != seen_host_pressure) {
        seen_host_pressure = pressure;
        const u64 before = CachedBytes();
        ReleaseIdleBuffers();
        LOG_INFO(Render_Vulkan, "Host memory pressure: staging buffers {} -> {} MiB", before >> 20,
                 CachedBytes() >> 20);
    }
    if (const u64 low_water = Common::PlatformEvents::MemoryLowWater(); low_water != 0) {
        const u64 available = Common::PlatformEvents::AvailableProcessMemory();
        low_memory = available != 0 && available < low_water;
    }
    if (configured_submit_bytes != 0 && VideoCommon::HostBudget::FollowsHeadroom() &&
        frame_count % 30 == 0) {
        const u64 available = VideoCommon::HostBudget::Available(
            VideoCommon::HostBudget::Simulated() ? device.GetDeviceMemoryUsage() : 0);
        const u64 room = available > VideoCommon::HostBudget::Reserve
                             ? available - VideoCommon::HostBudget::Reserve
                             : 0;
        const u64 cap = std::clamp(room / 4, configured_cap_bytes,
                                   (std::max)(configured_cap_bytes, IOS_UPLOAD_MAX_CAP));
        const u64 submit = (std::max)(configured_submit_bytes, cap / 4);
        if (cap != upload_cap_bytes && (cap > upload_cap_bytes + 32_MiB ||
                                        cap + 32_MiB < upload_cap_bytes ||
                                        cap == configured_cap_bytes)) {
            LOG_INFO(Common_Memory,
                     "Staging: {} MiB left to the process: uploads submit every {} MiB, pooled "
                     "upload buffers capped at {} MiB",
                     available >> 20, submit >> 20, cap >> 20);
            upload_cap_bytes = cap;
            upload_submit_bytes = submit;
        }
    }
    if (++frame_count % 60 == 0 && low_memory_waits != 0) {
        LOG_INFO(Common_Memory,
                 "Staging: waited for the GPU {} times under the low-water mark rather than "
                 "make more staging buffers",
                 low_memory_waits);
        low_memory_waits = 0;
    }
    if (frame_count % 60 == 0 && (early_submits != 0 || held_submits != 0)) {
        LOG_INFO(Common_Memory,
                 "Staging: {} command buffers submitted early for {} MiB of pooled uploads ({} "
                 "put off to the end of a draw), {} waits for the GPU at the {} MiB cap ({} of "
                 "the submits with under twice the low-water mark left); pooled upload buffers "
                 "{} MiB now, peak {} MiB",
                 early_submits, early_submit_bytes >> 20, held_submits, cap_waits,
                 upload_cap_bytes >> 20, low_memory_cap_waits, UploadPoolBytes() >> 20,
                 upload_pool_peak >> 20);
        early_submits = 0;
        held_submits = 0;
        early_submit_bytes = 0;
        cap_waits = 0;
        low_memory_cap_waits = 0;
        upload_pool_peak = 0;
    }

    ReleaseCache(MemoryUsage::DeviceLocal);
    ReleaseCache(MemoryUsage::Upload);
    ReleaseCache(MemoryUsage::Download);
}

void StagingBufferPool::ReleaseIdleBuffers() {
    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        for (StagingBuffers& staging : *cache) {
            std::erase_if(staging.entries, [this](const StagingBuffer& entry) {
                return !entry.deferred && scheduler.IsFree(entry.tick);
            });
            staging.delete_index = 0;
            staging.iterate_index = 0;
        }
    }
}

u64 StagingBufferPool::CachedBytes() const {
    u64 bytes = 0;
    for (const StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        for (size_t log2 = 0; log2 < NUM_LEVELS; ++log2) {
            bytes += u64{(*cache)[log2].entries.size()} << log2;
        }
    }
    return bytes;
}

u64 StagingBufferPool::AllocationBytes(MemoryUsage usage, size_t log2) const noexcept {
    const u64 logical_size = u64{1} << log2;
    return device.IsMoltenVK() && usage == MemoryUsage::Upload
               ? (std::max)(logical_size, u64{STREAM_BUFFER_TAIL})
               : logical_size;
}

u64 StagingBufferPool::OldestBusyUploadTick() const {
    u64 oldest = 0;
    for (const StagingBuffersCache* const cache : {&upload_cache, &device_local_cache}) {
        for (const StagingBuffers& staging : *cache) {
            for (const StagingBuffer& entry : staging.entries) {
                if (!entry.deferred && !scheduler.IsFree(entry.tick) &&
                    (oldest == 0 || entry.tick < oldest)) {
                    oldest = entry.tick;
                }
            }
        }
    }
    return oldest;
}

u64 StagingBufferPool::UploadPoolBytes() const {
    u64 bytes = 0;
    for (size_t log2 = 0; log2 < NUM_LEVELS; ++log2) {
        bytes += upload_cache[log2].entries.size() * AllocationBytes(MemoryUsage::Upload, log2);
        bytes += device_local_cache[log2].entries.size() << log2;
    }
    return bytes;
}

void StagingBufferPool::ReleaseIdleUploadsDownTo(u64 bytes) {
    u64 held = UploadPoolBytes();
    for (size_t level = NUM_LEVELS; level-- > 0 && held > bytes;) {
        for (StagingBuffersCache* const cache : {&upload_cache, &device_local_cache}) {
            StagingBuffers& staging = (*cache)[level];
            const u64 each = AllocationBytes(cache == &upload_cache ? MemoryUsage::Upload
                                                                    : MemoryUsage::DeviceLocal,
                                             level);
            std::erase_if(staging.entries, [&](const StagingBuffer& entry) {
                if (held <= bytes || entry.deferred || !scheduler.IsFree(entry.tick)) {
                    return false;
                }
                held -= each;
                return true;
            });
            staging.delete_index = 0;
            staging.iterate_index = 0;
        }
    }
}

bool StagingBufferPool::BoundUploads() {
    if (upload_submit_bytes == 0 || tick_upload_bytes < upload_submit_bytes ||
        upload_tick != scheduler.CurrentTick() || !VideoCommon::GpuSplit::OwnsScheduler()) {
        return false;
    }
    if (submits_held) {
        // The next BoundUploads between draws submits.
        ++held_submits;
        return false;
    }
    ++early_submits;
    Common::PerfReport::Add(Common::PerfReport::EarlySubmits);
    early_submit_bytes += tick_upload_bytes;
    tick_upload_bytes = 0;
    const u64 submitted = scheduler.Flush();
    u64 pooled = UploadPoolBytes();
    upload_pool_peak = (std::max)(upload_pool_peak, pooled);
    // iOS, with less than twice the low-water mark left to the process: only the command buffer
    // just submitted may hold upload buffers.
    u64 cap = upload_cap_bytes;
    if (const u64 low_water = Common::PlatformEvents::MemoryLowWater(); low_water != 0) {
        const u64 available = Common::PlatformEvents::AvailableProcessMemory();
        if (available != 0 && available < low_water * 2) {
            cap = upload_submit_bytes * 2;
            ++low_memory_cap_waits;
        }
    }
    if (pooled <= cap) {
        return true;
    }
    // Wait for the oldest submissions first, the one just made last, and free what they used
    // beyond the room the next command buffer's uploads need.
    ++cap_waits;
    Common::PerfReport::Add(Common::PerfReport::CapWaits);
    const u64 keep = cap - upload_submit_bytes;
    for (const u64 tick : {OldestBusyUploadTick(), submitted - 1, submitted}) {
        if (tick == 0 || tick > submitted) {
            continue;
        }
        scheduler.Wait(tick);
        ReleaseIdleUploadsDownTo(keep);
        pooled = UploadPoolBytes();
        if (pooled <= cap) {
            break;
        }
    }
    return true;
}

StagingBufferRef StagingBufferPool::GetStreamBuffer(size_t size) {
    // Checked before wrapping: wrapping first marked the start free (free_iterator = size), so the
    // next request was handed the start of the ring while the GPU could still read it.
    if (iterator + size >= stream_buffer_size && AreRegionsActive(0, Region(size) + 1)) {
        return GetStagingBuffer(size, MemoryUsage::Upload);
    }
    if (AreRegionsActive(Region(free_iterator) + 1,
                         (std::min)(Region(iterator + size) + 1, NUM_SYNCS))) {
        // Avoid waiting for the previous usages to be free
        return GetStagingBuffer(size, MemoryUsage::Upload);
    }
    const u64 current_tick = scheduler.CurrentTick();
    std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + Region(iterator),
              current_tick);
    used_iterator = iterator;
    free_iterator = (std::max)(free_iterator, iterator + size);

    if (iterator + size >= stream_buffer_size) {
        std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + NUM_SYNCS,
                  current_tick);
        used_iterator = 0;
        iterator = 0;
        free_iterator = size;
    }
    const size_t offset = iterator;
    iterator = Common::AlignUp(iterator + size, MAX_ALIGNMENT);
    return StagingBufferRef{
        .buffer = *stream_buffer,
        .device_address = stream_buffer_address,
        .offset = static_cast<VkDeviceSize>(offset),
        .mapped_span = stream_pointer.subspan(offset, size),
        .usage{},
        .log2_level{},
        .index{},
    };
}

bool StagingBufferPool::AreRegionsActive(size_t region_begin, size_t region_end) const {
    u64 gpu_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    const auto has_active = [&](u64 tick) {
        return std::any_of(sync_ticks.begin() + region_begin, sync_ticks.begin() + region_end,
                           [tick](u64 sync_tick) { return tick < sync_tick; });
    };
    if (!has_active(gpu_tick)) {
        return false;
    }
    scheduler.GetMasterSemaphore().Refresh();
    gpu_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    return has_active(gpu_tick);
}

StagingBufferRef StagingBufferPool::GetStagingBuffer(size_t size, MemoryUsage usage,
                                                     bool deferred) {
    if (usage != MemoryUsage::Download && !deferred) {
        const u64 tick = scheduler.CurrentTick();
        if (tick != upload_tick) {
            upload_tick = tick;
            tick_upload_bytes = 0;
        }
        tick_upload_bytes += AllocationBytes(usage, Common::Log2Ceil(size));
        Common::PerfReport::Add(Common::PerfReport::StagingPooledBytes,
                                AllocationBytes(usage, Common::Log2Ceil(size)));
    }
    if (const std::optional<StagingBufferRef> ref = TryGetReservedBuffer(size, usage, deferred)) {
        return *ref;
    }
    return CreateStagingBuffer(size, usage, deferred);
}

std::optional<StagingBufferRef> StagingBufferPool::TryGetReservedBuffer(size_t size,
                                                                        MemoryUsage usage,
                                                                        bool deferred) {
    StagingBuffers& cache_level = GetCache(usage)[Common::Log2Ceil(size)];
    auto& entries = cache_level.entries;
    if (entries.empty()) {
        return std::nullopt;
    }
    auto& semaphore = scheduler.GetMasterSemaphore();
    u64 gpu_tick = semaphore.KnownGpuTick();
    if (gpu_tick == cache_level.known_busy_gpu_tick) {
        semaphore.Refresh();
        gpu_tick = semaphore.KnownGpuTick();
        if (gpu_tick == cache_level.known_busy_gpu_tick) {
            return std::nullopt;
        }
    }
    if (cache_level.iterate_index >= entries.size()) {
        cache_level.iterate_index = 0;
    }
    const auto find_with_tick = [&](u64 tick) {
        const auto is_free = [tick](const StagingBuffer& entry) {
            return !entry.deferred && entry.tick <= tick;
        };
        const auto hint_it = entries.begin() + cache_level.iterate_index;
        auto it = std::find_if(hint_it, entries.end(), is_free);
        if (it == entries.end()) {
            it = std::find_if(entries.begin(), hint_it, is_free);
            if (it == hint_it) {
                return entries.end();
            }
        }
        return it;
    };
    auto it = find_with_tick(gpu_tick);
    if (it == entries.end()) {
        semaphore.Refresh();
        const u64 refreshed_tick = semaphore.KnownGpuTick();
        if (refreshed_tick != gpu_tick) {
            gpu_tick = refreshed_tick;
            it = find_with_tick(gpu_tick);
        }
        if (it == entries.end()) {
            cache_level.known_busy_gpu_tick = gpu_tick;
            return std::nullopt;
        }
    }
    cache_level.iterate_index =
        (static_cast<size_t>(std::distance(entries.begin(), it)) + 1) % entries.size();
    it->tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick();
    ASSERT(!it->deferred);
    it->deferred = deferred;
    return it->Ref();
}

StagingBufferRef StagingBufferPool::CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred) {
    // Under the low-water mark (iOS) a frame that uploads a new scene made a staging buffer for
    // each upload the stream buffer had no room for, before the GPU had read any: Metal went
    // 325 -> 549 MiB in the second before iOS closed the app. Past 32 MiB of them, wait for the
    // command buffers already submitted and reuse or free what they used. Only those: buffers
    // handed out for the current one may not have had their copies recorded yet.
    // The same past the cap of pooled upload buffers, on any device: BoundUploads holds it between
    // draws, this within one (a scene's first draws upload hundreds of MiB each on desktop).
    constexpr u64 LowMemoryStagingBytes = 32_MiB;
    // iOS: also with under 768 MiB left to the process after this buffer, read now rather than at
    // the last frame (the forest load went from 1743 to 936 MiB left within a frame).
    constexpr u64 MinRoomBytes = 768_MiB;
    if (!deferred && usage != MemoryUsage::Download && VideoCommon::GpuSplit::OwnsScheduler() &&
        scheduler.CurrentTick() > 1) {
        const u64 available = Common::PlatformEvents::AvailableProcessMemory();
        const bool room_low = available != 0 && available < size + MinRoomBytes;
        if ((low_memory || room_low) && CachedBytes() + size > LowMemoryStagingBytes) {
            ++low_memory_waits;
            scheduler.Wait(scheduler.CurrentTick() - 1);
            if (const std::optional<StagingBufferRef> ref =
                    TryGetReservedBuffer(size, usage, deferred)) {
                ReleaseIdleBuffers();
                return *ref;
            }
            ReleaseIdleBuffers();
        } else if (upload_submit_bytes != 0 && UploadPoolBytes() + size > upload_cap_bytes) {
            // The oldest submission first: waiting for all of them stalled area loads on desktop.
            ++cap_waits;
        Common::PerfReport::Add(Common::PerfReport::CapWaits);
            const u64 room = upload_cap_bytes - (std::min)(upload_cap_bytes, u64{size});
            const u64 oldest = OldestBusyUploadTick();
            for (const u64 tick : {oldest, scheduler.CurrentTick() - 1}) {
                if (tick == 0 || tick >= scheduler.CurrentTick()) {
                    continue;
                }
                scheduler.Wait(tick);
                if (const std::optional<StagingBufferRef> ref =
                        TryGetReservedBuffer(size, usage, deferred)) {
                    return *ref;
                }
                ReleaseIdleUploadsDownTo(room);
                if (UploadPoolBytes() + size <= upload_cap_bytes) {
                    break;
                }
            }
        }
    }
    auto const log2_size = Common::Log2Ceil<u32>(u32(size));
    const VkDeviceSize logical_size = 1ULL << log2_size;
    // Upload fallbacks can also back Metal's reflected 64 KiB constant-buffer arguments.
    const VkDeviceSize allocation_size = device.IsMoltenVK() && usage == MemoryUsage::Upload
                                            ? (std::max)(logical_size, STREAM_BUFFER_TAIL)
                                            : logical_size;
    VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = allocation_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    const MemoryUsage alloc_usage =
        (use_rebar && usage == MemoryUsage::Upload && (1ULL << log2_size) <= 1_MiB)
            ? MemoryUsage::Stream
            : usage;
    const MemoryTagScope memory_tag{usage == MemoryUsage::Upload        ? MemoryTag::StagingUpload
                                    : usage == MemoryUsage::DeviceLocal ? MemoryTag::StagingDeviceLocal
                                                                        : MemoryTag::StagingDownload};
    vk::Buffer buffer;
    try {
        buffer = memory_allocator.CreateBuffer(buffer_ci, alloc_usage);
    } catch (const vk::Exception& e) {
        // Out of memory: free every idle staging buffer and try once more.
        size_t freed = 0;
        for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
            for (StagingBuffers& level : *cache) {
                const size_t before = level.entries.size();
                std::erase_if(level.entries, [this](const StagingBuffer& entry) {
                    return !entry.deferred && scheduler.IsFree(entry.tick);
                });
                freed += before - level.entries.size();
                level.iterate_index = 0;
                level.delete_index = 0;
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Staging buffer of {} MiB could not be made ({}); freed {} idle staging "
                    "buffers and retrying",
                    buffer_ci.size >> 20, e.what(), freed);
        buffer = memory_allocator.CreateBuffer(buffer_ci, alloc_usage);
    }
    VideoCommon::Churn::Add(VideoCommon::Churn::StagingBuffers);
    VideoCommon::Churn::Add(VideoCommon::Churn::StagingBytes, buffer_ci.size);
    if (device.HasDebuggingToolAttached()) {
        ++buffer_index;
        buffer.SetObjectNameEXT(fmt::format("Staging Buffer {}", buffer_index).c_str());
    }
    const std::span<u8> mapped_span = buffer.Mapped().empty()
                                          ? std::span<u8>{}
                                          : buffer.Mapped().first(logical_size);
    const VkDeviceAddress buffer_address =
        device.IsBufferDeviceAddressSupported()
            ? device.GetLogical().GetBufferDeviceAddress(*buffer)
            : VkDeviceAddress{};
    auto& staging = GetCache(usage)[log2_size];
    StagingBuffer& entry = staging.entries.emplace_back(StagingBuffer{
        .buffer = std::move(buffer),
        .device_address = buffer_address,
        .mapped_span = mapped_span,
        .usage = usage,
        .log2_level = log2_size,
        .index = unique_ids++,
        .tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick(),
        .deferred = deferred,
    });
    staging.iterate_index = 0;
    return entry.Ref();
}

StagingBufferPool::StagingBuffersCache& StagingBufferPool::GetCache(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
        return device_local_cache;
    case MemoryUsage::Upload:
        return upload_cache;
    case MemoryUsage::Download:
        return download_cache;
    default:
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return upload_cache;
    }
}

void StagingBufferPool::ReleaseCache(MemoryUsage usage) {
    static constexpr size_t LARGE_BUFFER_LOG2 = 22; // >= 4 MiB swept every frame
    auto& cache = GetCache(usage);
    ReleaseLevel(cache, current_delete_level);
    for (size_t log2 = LARGE_BUFFER_LOG2; log2 < NUM_LEVELS; ++log2) {
        if (!cache[log2].entries.empty()) {
            ReleaseLevel(cache, log2);
        }
    }
}

void StagingBufferPool::ReleaseLevel(StagingBuffersCache& cache, size_t log2) {
    constexpr size_t deletions_per_tick = 16;
    auto& staging = cache[log2];
    auto& entries = staging.entries;
    if (entries.empty()) {
        return;
    }
    const size_t old_size = entries.size();

    // EDEN_NX2_EXP8 bit 1: keep two idle upload buffers per level up to 32 MiB (under 128 MiB in
    // all). Each level is swept every 64 frames, so after a pause each upload over 8 MiB (past
    // the stream buffer's regions) got a new dedicated allocation, written to for the first time
    // (16 MiB on an RTX 4080, idle system: 0.4 ms to create, 1.2 ms more for the first write).
    constexpr size_t KeepMaxLog2 = 25;
    constexpr size_t KeepPerLevel = 2;
    if (&cache == &upload_cache && log2 <= KeepMaxLog2 &&
        (VideoCore::Nx2Debug::Experiment(8) & 1) != 0) {
        size_t idle = 0;
        std::erase_if(entries, [this, &idle](const StagingBuffer& entry) {
            return !entry.deferred && scheduler.IsFree(entry.tick) && ++idle > KeepPerLevel;
        });
        staging.delete_index = 0;
        if (staging.iterate_index >= entries.size()) {
            staging.iterate_index = 0;
        }
        return;
    }

    const auto is_deletable = [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    };
    const size_t begin_offset = staging.delete_index;
    const size_t end_offset = (std::min)(begin_offset + deletions_per_tick, old_size);
    const auto begin = entries.begin() + begin_offset;
    const auto end = entries.begin() + end_offset;
    entries.erase(std::remove_if(begin, end, is_deletable), end);

    const size_t new_size = entries.size();
    staging.delete_index += deletions_per_tick;
    if (staging.delete_index >= new_size) {
        staging.delete_index = 0;
    }
    if (staging.iterate_index >= new_size) {
        staging.iterate_index = 0;
    }
}

} // namespace Vulkan
