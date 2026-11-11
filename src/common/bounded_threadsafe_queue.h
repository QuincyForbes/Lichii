// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <memory>
#include <new>
#include <utility>

#include "common/polyfill_thread.h"

namespace Common {

namespace detail {
constexpr size_t DefaultCapacity = 0x1000;
} // namespace detail

template <typename T, size_t Capacity = detail::DefaultCapacity>
class SPSCQueue {
    static_assert(Capacity != 0 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a nonzero power of two.");

public:
    SPSCQueue() = default;

    // Producers/consumers must be stopped before destruction. Only the queued
    // slots contain live T objects; popped slots were destroyed before reuse.
    ~SPSCQueue() {
        const auto end = producer.index.load(std::memory_order_relaxed);
        for (auto index = consumer.index.load(std::memory_order_relaxed); index != end; ++index) {
            std::destroy_at(std::addressof(m_data[index % Capacity].value));
        }
    }

    template <typename... Args>
    bool TryEmplace(Args&&... args) noexcept {
        return Emplace<PushMode::Try>(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void EmplaceWait(Args&&... args) noexcept {
        Emplace<PushMode::Wait>(std::forward<Args>(args)...);
    }

    bool TryPop(T& t) noexcept {
        return Pop<PopMode::Try>(t);
    }

    void PopWait(T& t) noexcept {
        Pop<PopMode::Wait>(t);
    }

    void PopWait(T& t, const std::stop_token stop_token) noexcept {
        Pop<PopMode::WaitWithStopToken>(t, stop_token);
    }

    // A bounded wait for consumers that also need to poll an external condition.
    // Use the same predicate/mutex as PopWait so a push between TryPop and this
    // call cannot be missed. Only the single consumer removes queue entries.
    template <typename Rep, typename Period>
    bool PopWaitFor(T& t, const std::chrono::duration<Rep, Period>& timeout,
                    const std::stop_token stop_token = {}) noexcept {
        if (stop_token.stop_requested()) {
            return false;
        }
        if (TryPop(t)) {
            return true;
        }
        if (timeout <= std::chrono::duration<Rep, Period>::zero()) {
            return false;
        }
        {
            std::unique_lock lock{consumer.cv_mutex};
            consumer.waiters.fetch_add(1, std::memory_order_seq_cst);
            const bool ready = consumer.cv.wait_for(lock, stop_token, timeout, [this] {
                return consumer.index.load(std::memory_order_relaxed) !=
                       producer.index.load(std::memory_order_seq_cst);
            });
            consumer.waiters.fetch_sub(1, std::memory_order_relaxed);
            if (!ready || stop_token.stop_requested()) {
                return false;
            }
        }
        return TryPop(t);
    }

    T PopWait() noexcept {
        T t{};
        Pop<PopMode::Wait>(t);
        return t;
    }

    T PopWait(const std::stop_token stop_token) noexcept {
        T t{};
        Pop<PopMode::WaitWithStopToken>(t, stop_token);
        return t;
    }

private:
    enum class PushMode {
        Try,
        Wait,
        Count,
    };

    enum class PopMode {
        Try,
        Wait,
        WaitWithStopToken,
        Count,
    };

    template <PushMode Mode, typename... Args>
    bool Emplace(Args&&... args) noexcept {
        const std::size_t write_index = producer.index.load(std::memory_order_relaxed);
        if ((write_index - consumer.index.load(std::memory_order_acquire)) == Capacity) [[unlikely]] {
            if constexpr (Mode == PushMode::Try) {
                return false;
            } else if constexpr (Mode == PushMode::Wait) {
                std::unique_lock lock{producer.cv_mutex};
                producer.waiters.fetch_add(1, std::memory_order_seq_cst);
                producer.cv.wait(lock, [this, write_index] {
                    return (write_index - consumer.index.load(std::memory_order_seq_cst)) <
                           Capacity;
                });
                producer.waiters.fetch_sub(1, std::memory_order_relaxed);
            } else {
                static_assert(Mode < PushMode::Count, "Invalid PushMode.");
            }
        }
        // Determine the position to write to.
        const std::size_t pos = write_index % Capacity;
        // Emplace into the queue.
        std::construct_at(std::addressof(m_data[pos].value), std::forward<Args>(args)...);
        // Publish the write index and wake a sleeping consumer only if one is registered.
        producer.index.store(write_index + 1, std::memory_order_seq_cst);
        if (consumer.waiters.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
            std::scoped_lock lock{consumer.cv_mutex};
            consumer.cv.notify_one();
        }
        return true;
    }

    template <PopMode Mode>
    bool Pop(T& t, [[maybe_unused]] std::stop_token stop_token = {}) noexcept {
        if constexpr (Mode == PopMode::WaitWithStopToken) {
            if (stop_token.stop_requested()) {
                return false;
            }
        }
        const std::size_t read_index = consumer.index.load(std::memory_order_relaxed);
        if (read_index == producer.index.load(std::memory_order_acquire)) [[unlikely]] {
            if constexpr (Mode == PopMode::Try) {
                return false;
            } else if constexpr (Mode == PopMode::Wait) {
                std::unique_lock lock{consumer.cv_mutex};
                consumer.waiters.fetch_add(1, std::memory_order_seq_cst);
                consumer.cv.wait(lock, [this, read_index] {
                    return read_index != producer.index.load(std::memory_order_seq_cst);
                });
                consumer.waiters.fetch_sub(1, std::memory_order_relaxed);
            } else if constexpr (Mode == PopMode::WaitWithStopToken) {
                std::unique_lock lock{consumer.cv_mutex};
                consumer.waiters.fetch_add(1, std::memory_order_seq_cst);
                const bool ready = consumer.cv.wait(lock, stop_token, [this, read_index] {
                    return read_index != producer.index.load(std::memory_order_seq_cst);
                });
                consumer.waiters.fetch_sub(1, std::memory_order_relaxed);
                if (!ready || stop_token.stop_requested()) {
                    return false;
                }
            } else {
                static_assert(Mode < PopMode::Count, "Invalid PopMode.");
            }
        }
        // Determine the position to read from.
        const std::size_t pos = read_index % Capacity;
        // Pop the data off the queue, moving it.
        t = std::move(m_data[pos].value);
        // Moving need not release every resource owned by the source. Destroy
        // it before publishing this slot as available to the producer.
        std::destroy_at(std::addressof(m_data[pos].value));
        // Publish the read index and wake a sleeping producer only if one is registered.
        consumer.index.store(read_index + 1, std::memory_order_seq_cst);
        if (producer.waiters.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
            std::scoped_lock lock{producer.cv_mutex};
            producer.cv.notify_one();
        }
        return true;
    }

    union Slot {
        Slot() noexcept {}
        ~Slot() {}
        T value;
    };
    std::array<Slot, Capacity> m_data;
    alignas(64) struct {
        std::atomic_size_t index{0};
        std::atomic<std::uint32_t> waiters{0};
        std::condition_variable_any cv;
        std::mutex cv_mutex;
    } producer;
    alignas(64) struct {
        std::atomic_size_t index{0};
        std::atomic<std::uint32_t> waiters{0};
        std::condition_variable_any cv;
        std::mutex cv_mutex;
    } consumer;
};

template <typename T, size_t Capacity = detail::DefaultCapacity>
class MPSCQueue {
public:
    template <typename... Args>
    bool TryEmplace(Args&&... args) {
        std::scoped_lock lock{write_mutex};
        return spsc_queue.TryEmplace(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void EmplaceWait(Args&&... args) {
        std::scoped_lock lock{write_mutex};
        spsc_queue.EmplaceWait(std::forward<Args>(args)...);
    }

    bool TryPop(T& t) {
        return spsc_queue.TryPop(t);
    }

    void PopWait(T& t) {
        spsc_queue.PopWait(t);
    }

    void PopWait(T& t, std::stop_token stop_token) {
        spsc_queue.PopWait(t, stop_token);
    }

    T PopWait() {
        return spsc_queue.PopWait();
    }

    T PopWait(std::stop_token stop_token) {
        return spsc_queue.PopWait(stop_token);
    }

private:
    SPSCQueue<T, Capacity> spsc_queue;
    std::mutex write_mutex;
};

template <typename T, size_t Capacity = detail::DefaultCapacity>
class MPMCQueue {
public:
    template <typename... Args>
    bool TryEmplace(Args&&... args) {
        std::scoped_lock lock{write_mutex};
        return spsc_queue.TryEmplace(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void EmplaceWait(Args&&... args) {
        std::scoped_lock lock{write_mutex};
        spsc_queue.EmplaceWait(std::forward<Args>(args)...);
    }

    bool TryPop(T& t) {
        std::scoped_lock lock{read_mutex};
        return spsc_queue.TryPop(t);
    }

    void PopWait(T& t) {
        std::scoped_lock lock{read_mutex};
        spsc_queue.PopWait(t);
    }

    void PopWait(T& t, std::stop_token stop_token) {
        std::scoped_lock lock{read_mutex};
        spsc_queue.PopWait(t, stop_token);
    }

    T PopWait() {
        std::scoped_lock lock{read_mutex};
        return spsc_queue.PopWait();
    }

    T PopWait(std::stop_token stop_token) {
        std::scoped_lock lock{read_mutex};
        return spsc_queue.PopWait(stop_token);
    }

private:
    SPSCQueue<T, Capacity> spsc_queue;
    alignas(64) std::mutex write_mutex;
    alignas(64) std::mutex read_mutex;
};

} // namespace Common
