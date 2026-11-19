// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"

namespace Vulkan {

class EmissionBudget {
public:
    class Reservation {
    public:
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        ~Reservation() {
            if (pending) {
                pending->fetch_sub(1, std::memory_order_relaxed);
            }
        }

    private:
        friend class EmissionBudget;
        Reservation() = default;
        std::shared_ptr<std::atomic<size_t>> pending;
    };

    explicit EmissionBudget(size_t capacity_) : capacity{capacity_} {}

    std::unique_ptr<Reservation> TryAcquire(bool enabled, bool native) {
        size_t count = pending->load(std::memory_order_relaxed);
        if (!DefersShaderEmission(enabled, native, count, capacity)) {
            return nullptr;
        }
        auto reservation = std::unique_ptr<Reservation>{new Reservation};
        while (DefersShaderEmission(enabled, native, count, capacity)) {
            if (pending->compare_exchange_weak(count, count + 1, std::memory_order_relaxed)) {
                reservation->pending = pending;
                return reservation;
            }
        }
        return nullptr;
    }

    size_t Pending() const {
        return pending->load(std::memory_order_relaxed);
    }

private:
    const size_t capacity;
    std::shared_ptr<std::atomic<size_t>> pending = std::make_shared<std::atomic<size_t>>(0);
};

enum class PipelineBuilder : u32 {
    Worker,    ///< "VkPipelineBuilder"
    Held,      ///< "VkPipelineHeld": builds a draw waits for
    GpuThread, ///< the thread that asked or waits for it
    Boot,
    Prebuild,
    Count,
};

/// Who builds what on this thread when no pool is told otherwise.
inline PipelineBuilder& CurrentPipelineBuilder() {
    thread_local PipelineBuilder builder = PipelineBuilder::GpuThread;
    return builder;
}

class PipelineBuilderScope {
public:
    explicit PipelineBuilderScope(PipelineBuilder builder)
        : previous{std::exchange(CurrentPipelineBuilder(), builder)} {}
    ~PipelineBuilderScope() {
        CurrentPipelineBuilder() = previous;
    }

    PipelineBuilderScope(const PipelineBuilderScope&) = delete;
    PipelineBuilderScope& operator=(const PipelineBuilderScope&) = delete;

private:
    PipelineBuilder previous;
};

/// A held pool thread builds at the class of the thread waiting for it (the Vulkan worker:
/// user-interactive on Apple). Above normal elsewhere: time critical would starve the rest.
inline void RaiseHeldBuilderThread() {
    thread_local bool raised{};
    if (std::exchange(raised, true)) {
        return;
    }
#ifdef __APPLE__
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
#else
    Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
#endif
}

/// Counters behind the "Perf pipelines:" line; the reporter takes them with exchange.
struct PipelineBuildStats {
    struct Builder {
        std::atomic<u64> built{};
        std::atomic<u64> queue_ns{};
        std::atomic<u64> queue_max_ns{};
        std::atomic<u64> build_ns{};
        std::atomic<u64> build_max_ns{};
        std::atomic<const char*> qos{""};
    };

    static PipelineBuildStats& Get() {
        static PipelineBuildStats stats;
        return stats;
    }

    static void StoreMax(std::atomic<u64>& max, u64 value) {
        u64 seen = max.load(std::memory_order_relaxed);
        while (seen < value &&
               !max.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
        }
    }

    void Add(PipelineBuilder who, std::chrono::nanoseconds queued,
             std::chrono::nanoseconds built) {
        Builder& builder = builders[static_cast<size_t>(who)];
        const u64 queue = static_cast<u64>(std::max<s64>(queued.count(), 0));
        const u64 build = static_cast<u64>(std::max<s64>(built.count(), 0));
        builder.built.fetch_add(1, std::memory_order_relaxed);
        builder.queue_ns.fetch_add(queue, std::memory_order_relaxed);
        builder.build_ns.fetch_add(build, std::memory_order_relaxed);
        StoreMax(builder.queue_max_ns, queue);
        StoreMax(builder.build_max_ns, build);
        builder.qos.store(Common::CurrentThreadQosName(), std::memory_order_relaxed);
    }

    std::array<Builder, static_cast<size_t>(PipelineBuilder::Count)> builders;
    /// SPIR-V emission inside the builds above.
    std::atomic<u64> emission_ns{};
    /// Builds a held draw found unclaimed and moved to the held pool.
    std::atomic<u64> handed_over{};
    /// Builds already running elsewhere that were lent the waiter's QoS.
    std::atomic<u64> boosted{};
    std::atomic<u32> holds_active{};
    std::atomic<u64> held_draws{};
    std::atomic<u64> held_ns{};
    std::atomic<u64> held_max_ns{};
};

class PipelineBuild {
    struct State {
        explicit State(std::function<void()> task_)
            : task{std::move(task_)}, created{std::chrono::steady_clock::now()} {}

        /// False when another thread had claimed the task.
        bool Run(PipelineBuilder who) noexcept {
            if (claimed.exchange(true)) {
                return false;
            }
            const auto begin = std::chrono::steady_clock::now();
            if (who == PipelineBuilder::Held) {
                RaiseHeldBuilderThread();
            }
            {
                std::scoped_lock lock{mutex};
                builder_thread = Common::CurrentThreadHandle();
                builder = who;
                running = true;
            }
            condition.notify_all();
            const PipelineBuilderScope scope{who};
            try {
                task();
            } catch (const std::exception& exception) {
                failed = true;
                LOG_ERROR(Render_Vulkan, "Graphics pipeline worker failed: {}", exception.what());
            } catch (...) {
                failed = true;
                LOG_ERROR(Render_Vulkan,
                          "Graphics pipeline worker failed with an unknown exception");
            }
            task = {};
            PipelineBuildStats::Get().Add(who, begin - created,
                                          std::chrono::steady_clock::now() - begin);
            std::scoped_lock lock{mutex};
            builder_thread = nullptr;
            complete.store(true, std::memory_order_release);
            condition.notify_all();
            return true;
        }

        std::function<void()> task;
        const std::chrono::steady_clock::time_point created;
        bool failed{};
        std::atomic_bool claimed{};
        std::atomic_bool complete{};
        std::mutex mutex;
        std::condition_variable condition;
        /// Under the mutex: the thread running the task, null once it is done.
        void* builder_thread{};
        PipelineBuilder builder{};
        bool running{};
    };

public:
    /// `held_workers`: where the build of a draw known to wait for it goes while a thread there
    /// is free; a burst beyond that queues on `workers` as the rest does (see WaitHeld).
    PipelineBuild(Common::ThreadWorker* workers, std::function<void()> task,
                  Common::ThreadWorker* held_workers = nullptr)
        : state{std::make_shared<State>(std::move(task))} {
        if (!workers) {
            (void)state->Run(CurrentPipelineBuilder());
        } else if (!held_workers || !held_workers->TryQueueWork([build_state = state] {
                       (void)build_state->Run(PipelineBuilder::Held);
                   })) {
            workers->QueueWork(
                [build_state = state] { (void)build_state->Run(PipelineBuilder::Worker); });
        }
    }

    ~PipelineBuild() {
        Wait(true);
    }

    bool IsBuilt() const {
        return state->complete.load(std::memory_order_acquire);
    }

    bool Failed() const {
        return IsBuilt() && state->failed;
    }

    /// Who ran the build, once one has.
    PipelineBuilder Builder() const {
        std::scoped_lock lock{state->mutex};
        return state->builder;
    }

    /// True when this call ran the build itself.
    bool Wait(bool may_build = false) const {
        const auto shared = state;
        if (may_build && shared->Run(CurrentPipelineBuilder())) {
            return true;
        }
        std::unique_lock lock{shared->mutex};
        shared->condition.wait(lock,
                               [&] { return shared->complete.load(std::memory_order_acquire); });
        return false;
    }

    /// Wait for a draw that is held. A build no thread has started goes to `held_workers`, ahead
    /// of whatever the builder pool has queued before it; one already running elsewhere is lent
    /// the waiter's QoS (Apple). With no `held_workers` this is Wait.
    bool WaitHeld(bool may_build, Common::ThreadWorker* held_workers) const {
        if (held_workers == nullptr || IsBuilt()) {
            return Wait(may_build);
        }
        const auto shared = state;
        if (may_build && shared->Run(CurrentPipelineBuilder())) {
            return true;
        }
        PipelineBuildStats& stats = PipelineBuildStats::Get();
        if (!shared->claimed.load(std::memory_order_acquire)) {
            held_workers->QueueWork([shared, &stats] {
                if (shared->Run(PipelineBuilder::Held)) {
                    stats.handed_over.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        stats.holds_active.fetch_add(1, std::memory_order_relaxed);
        {
            std::unique_lock lock{shared->mutex};
            const auto complete = [&] { return shared->complete.load(std::memory_order_acquire); };
            shared->condition.wait(lock, [&] { return shared->running || complete(); });
            if (!complete()) {
                const Common::ThreadQosBoost boost{
                    shared->builder == PipelineBuilder::Held ? nullptr : shared->builder_thread};
                if (boost.Active()) {
                    stats.boosted.fetch_add(1, std::memory_order_relaxed);
                }
                shared->condition.wait(lock, complete);
            }
        }
        stats.holds_active.fetch_sub(1, std::memory_order_relaxed);
        return false;
    }

    PipelineBuild(const PipelineBuild&) = delete;
    PipelineBuild& operator=(const PipelineBuild&) = delete;

private:
    std::shared_ptr<State> state;
};

} // namespace Vulkan
