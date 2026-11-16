// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <utility>

#include "common/logging.h"
#include "common/thread.h"

namespace Metal {

/// A build handed to builder threads, run with the runner's `Context` (a pipeline translator).
/// Whoever claims it first runs it: a builder, or a draw that may not be skipped and found it
/// unstarted, at that thread's own QoS. A build already running on a builder (background QoS) is
/// lent the waiter's QoS while the waiter waits, as EDEN_NX2_PIPELINE_HOLD_BOOST does for Vulkan
/// on Apple.
template <typename Context>
class BuildJob {
public:
    using Task = std::function<void(Context&)>;

    explicit BuildJob(Task task_) : task{std::move(task_)} {}

    /// Runs the task here; false when another thread had claimed it.
    bool Run(Context& context) noexcept {
        if (claimed.exchange(true, std::memory_order_acq_rel)) {
            return false;
        }
        {
            std::scoped_lock lock{mutex};
            builder_thread = Common::CurrentThreadHandle();
        }
        condition.notify_all();
        try {
            task(context);
        } catch (const std::exception& exception) {
            LOG_ERROR(Render_Metal, "Metal: pipeline build failed: {}", exception.what());
        } catch (...) {
            LOG_ERROR(Render_Metal, "Metal: pipeline build failed with an unknown exception");
        }
        task = {};
        {
            std::scoped_lock lock{mutex};
            builder_thread = nullptr;
            done.store(true, std::memory_order_release);
        }
        condition.notify_all();
        return true;
    }

    [[nodiscard]] bool IsDone() const noexcept {
        return done.load(std::memory_order_acquire);
    }

    /// Until the task is done: runs it with `context` when nobody has claimed it, else waits,
    /// lending this thread's QoS to the builder. True when it ran here; `boosted` when a builder
    /// was lent the QoS.
    bool Wait(Context& context, bool& boosted) {
        boosted = false;
        if (IsDone()) {
            return false;
        }
        if (Run(context)) {
            return true;
        }
        std::unique_lock lock{mutex};
        // The builder notes its thread right after claiming the task.
        condition.wait(lock, [this] { return builder_thread != nullptr || IsDone(); });
        if (!IsDone()) {
            const Common::ThreadQosBoost boost{builder_thread};
            boosted = boost.Active();
            condition.wait(lock, [this] { return IsDone(); });
        }
        return false;
    }

private:
    Task task;
    std::atomic_bool claimed{};
    std::atomic_bool done{};
    std::mutex mutex;
    std::condition_variable condition;
    /// Under the mutex: the thread running the task (Apple), null when none or done.
    void* builder_thread{};
};

} // namespace Metal
