// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <queue>
#include <span>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

class MasterSemaphore {
    using Waitable = std::pair<u64, vk::Fence>;

public:
    explicit MasterSemaphore(const Device& device);
    ~MasterSemaphore();

    /// Returns the current logical tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    /// Returns the last known GPU tick.
    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    /// The timeline semaphore, or null where fences stand in for it.
    [[nodiscard]] VkSemaphore Handle() const noexcept {
        return semaphore ? *semaphore : VK_NULL_HANDLE;
    }

    /// Returns true when a tick has been hit by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    /// Advance to the logical tick and return the old one
    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

    /// Submits the device graphics queue, updating the tick as necessary. The command buffers
    /// run in order (the upload command buffer first).
    VkResult SubmitQueue(std::span<const VkCommandBuffer> cmdbufs, VkSemaphore signal_semaphore,
                         VkSemaphore wait_semaphore, u64 host_tick);

private:
    VkResult SubmitQueueTimeline(std::span<const VkCommandBuffer> cmdbufs,
                                 VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                                 u64 host_tick);
    VkResult SubmitQueueFence(std::span<const VkCommandBuffer> cmdbufs,
                              VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                              u64 host_tick);

    void WaitThread(std::stop_token token);

    vk::Fence GetFreeFence();

private:
    const Device& device;             ///< Device.
    vk::Semaphore semaphore;          ///< Timeline semaphore.
    /// Separate 128-byte blocks (x86 spatial prefetcher pairs and Apple Silicon 128B L2 lines):
    /// GPUBack reads current_tick several times per draw while VulkanWorker/WaitThread update
    /// gpu_tick and wait_mutex (zortos293/Lichii#1161).
    alignas(128) std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    alignas(128) std::atomic<u64> current_tick{1}; ///< Current logical tick.
    alignas(128) std::mutex wait_mutex;
    std::mutex free_mutex;
    std::condition_variable_any wait_cv;
    std::queue<Waitable> wait_queue;  ///< Queue for the fences to be waited on by the wait thread.
    std::deque<vk::Fence> free_queue; ///< Holds available fences for submission.
    std::jthread debug_thread;        ///< Debug thread to workaround validation layer bugs.
    std::jthread wait_thread;         ///< Helper thread that waits for submitted fences.
};

} // namespace Vulkan
