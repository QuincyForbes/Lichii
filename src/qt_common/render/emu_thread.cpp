// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <qdebug.h>
#ifdef _WIN32
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <fmt/format.h>
#include "common/logging.h"
#include "common/windows/timer_resolution.h"
#endif
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/cpu_manager.h"
#include "emu_thread.h"
#include "qt_common/qt_common.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"

EmuThread::EmuThread() {}

EmuThread::~EmuThread() = default;

void EmuThread::run() {
    Common::SetCurrentThreadName("EmuControlThread");
#ifdef _WIN32
    // The 0.5 ms system timer only while a game runs.
    QtCommon::system->CoreTiming().SetTimerResolutionNs(
        Common::Windows::SetCurrentTimerResolutionToMaximum());
    SCOPE_EXIT {
        Common::Windows::ReleaseTimerResolution();
    };
#endif

    auto& gpu = QtCommon::system->GPU();
    auto stop_token = m_stop_source.get_token();
#ifdef _WIN32
    // What a 1 ms sleep really takes in this process. Windows 11 may stop honouring the timer
    // request above, and the system figure cannot show that because another program's request
    // moves it too. Measure 20 short sleeps at start and once a minute for the first 10 minutes
    // while emulation runs.
    std::jthread timer_check_thread([stop_token] {
        using namespace std::chrono;
        for (int round = 0; round < 11 && !stop_token.stop_requested(); ++round) {
            const auto start = steady_clock::now();
            int completed = 0;
            for (; completed < 20 && !stop_token.stop_requested(); ++completed) {
                std::this_thread::sleep_for(milliseconds{1});
            }
            if (completed < 20) {
                break;
            }
            const double sleep_ms =
                duration<double, std::milli>(steady_clock::now() - start).count() / 20.0;
            LOG_INFO(Frontend, "Timer check: a 1 ms sleep takes {:.2f} ms (system timer {:.4f} ms)",
                     sleep_ms,
                     duration<double, std::milli>(Common::Windows::GetCurrentTimerResolution())
                         .count());
            if (round == 10 || !Common::StoppableTimedWait(stop_token, seconds{60})) {
                break;
            }
        }
    });
#endif

    QtCommon::system->RegisterHostThread();

    // Main process has been loaded. Make the context current to this thread and begin GPU and CPU
    // execution.
    gpu.ObtainContext();

    emit LoadProgress(VideoCore::LoadCallbackStage::Prepare, 0, 0);
    if (Settings::values.use_disk_shader_cache.GetValue()) {
        QtCommon::system->Renderer().ReadRasterizer()->LoadDiskResources(
            QtCommon::system->GetApplicationProcessProgramID(), stop_token,
            [this](VideoCore::LoadCallbackStage stage, std::size_t value, std::size_t total) {
                emit LoadProgress(stage, value, total);
            });
    }
    emit LoadProgress(VideoCore::LoadCallbackStage::Complete, 0, 0);

    gpu.ReleaseContext();
    gpu.Start();

    QtCommon::system->GetCpuManager().OnGpuReady();

    if (QtCommon::system->DebuggerEnabled()) {
        QtCommon::system->InitializeDebugger();
    }

    while (!stop_token.stop_requested()) {
        std::unique_lock lk{m_should_run_mutex};
        if (m_should_run) {
            QtCommon::system->Run();
            m_stopped.Reset();

            m_should_run_cv.wait(lk, stop_token, [&] {
                return !m_should_run || m_pending_shader_cache_title.has_value();
            });

            if (m_should_run && m_pending_shader_cache_title.has_value()) {
                const u64 program_id = *m_pending_shader_cache_title;
                m_pending_shader_cache_title.reset();

                lk.unlock();
                this->ReloadDiskShaderCache(program_id);
                lk.lock();
            }
        } else {
            QtCommon::system->Pause();
            m_stopped.Set();

            EmulationPaused(lk);
            m_should_run_cv.wait(lk, stop_token, [&] { return m_should_run; });
            EmulationResumed(lk);
        }
    }

    // Shutdown the main emulated process
    QtCommon::system->DetachDebugger();
    QtCommon::system->ShutdownMainProcess();
}

void EmuThread::ReloadDiskShaderCache(u64 program_id) {
    if (!Settings::values.use_disk_shader_cache.GetValue()) {
        return;
    }

    LOG_INFO(Frontend, "Reloading disk shader cache for {:016X}", program_id);

    auto& system = *QtCommon::system;
    auto& gpu = system.GPU();

    system.Pause();
    gpu.WaitForIdle();
    gpu.ObtainContext();

    emit ShaderCacheReloadStarted();
    emit LoadProgress(VideoCore::LoadCallbackStage::Prepare, 0, 0);
    system.Renderer().ReadRasterizer()->LoadDiskResources(program_id, m_stop_source.get_token(),
        [this](VideoCore::LoadCallbackStage stage, std::size_t value, std::size_t total) {
            emit LoadProgress(stage, value, total);
        });
    emit LoadProgress(VideoCore::LoadCallbackStage::Complete, 0, 0);
    emit ShaderCacheReloadFinished();

    gpu.ReleaseContext();
}

// Unlock while emitting signals so that the main thread can
// continue pumping events.

void EmuThread::EmulationPaused(std::unique_lock<std::mutex>& lk) {
    lk.unlock();
    emit DebugModeEntered();
    lk.lock();
}

void EmuThread::EmulationResumed(std::unique_lock<std::mutex>& lk) {
    lk.unlock();
    emit DebugModeLeft();
    lk.lock();
}
