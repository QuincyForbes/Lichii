// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include "audio_core/adsp/adsp.h"
#include "audio_core/audio_core.h"
#include "audio_core/common/common.h"
#include "audio_core/renderer/system_manager.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/core_timing.h"

namespace AudioCore::Renderer {

namespace {
constexpr auto FramePeriod = std::chrono::nanoseconds{
    (u64{TargetSampleCount} * 1'000'000'000ULL) / TargetSampleRate};
static_assert(FramePeriod == std::chrono::milliseconds{5});
} // namespace

SystemManager::SystemManager(Core::System& core_)
    : core{core_}, audio_renderer{core.AudioCore().ADSP().AudioRenderer()} {}

SystemManager::~SystemManager() {
    Stop();
}

void SystemManager::InitializeUnsafe() {
    if (!active) {
        active = true;
        audio_renderer.Start();
        thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("AudioRenderSystemManager");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::High);

            auto next_deadline = std::chrono::steady_clock::now();
            while (active && !stop_token.stop_requested()) {
                bool any_active = false;
                {
                    std::scoped_lock l{mutex1};
                    for (auto* system : systems) {
                        any_active |= system->IsActive();
                        system->SendCommandToDsp();
                    }
                }

                if (!any_active || core.IsPaused() || core.IsShuttingDown()) {
                    std::unique_lock lk{pace_mutex};
                    pace_cv.wait_for(lk, stop_token, FramePeriod,
                                     [this] { return !active.load(std::memory_order_relaxed); });
                    next_deadline = std::chrono::steady_clock::now();
                    continue;
                }

                const bool stream_paced = audio_renderer.Render(stop_token);
                if (!stream_paced) {
                    // Stream is paused, unavailable, or using CircularBufferSink without a host
                    // output callback (cf. Djihads80/Eglantine#910): pace to the 5 ms ADSP frame
                    // period regardless of CPU render cost so we never busy-spin or flood
                    // adsp_rendered_event.
                    const auto now = std::chrono::steady_clock::now();
                    if (next_deadline < now) {
                        next_deadline = now;
                    }
                    next_deadline += FramePeriod;
                    std::unique_lock lk{pace_mutex};
                    pace_cv.wait_until(lk, stop_token, next_deadline,
                                       [this] { return !active.load(std::memory_order_relaxed); });
                } else {
                    next_deadline = std::chrono::steady_clock::now();
                }
            }
        });
    }
}

void SystemManager::Stop() {
    if (active) {
        active = false;
        pace_cv.notify_all();
        thread.request_stop();
        thread.join();
        audio_renderer.Stop();
    }
}

bool SystemManager::Add(System& system_) {
    std::scoped_lock l2{mutex2};
    if (systems.size() + 1 > MaxRendererSessions) {
        LOG_ERROR(Service_Audio, "Maximum AudioRenderer Systems active, cannot add more!");
        return false;
    }
    {
        std::scoped_lock l{mutex1};
        if (systems.empty()) {
            InitializeUnsafe();
        }
        systems.push_back(&system_);
    }
    pace_cv.notify_one();
    return true;
}

bool SystemManager::Remove(System& system_) {
    std::scoped_lock l2{mutex2};
    {
        std::scoped_lock l{mutex1};
        if (systems.remove(&system_) == 0) {
            LOG_ERROR(Service_Audio,
                      "Failed to remove a render system, it was not found in the list!");
            return false;
        }
    }

    if (systems.empty()) {
        Stop();
    }
    return true;
}

} // namespace AudioCore::Renderer
