// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

/// The platform layer (lichii_bridge_platform.h) in the app: pauses the game while the app is
/// inactive, keeps the screen awake while playing, follows the thermal state and the shader cache
/// loading progress, and reads the device facts.
@MainActor
final class PlatformCenter: ObservableObject {
    static let shared = PlatformCenter()

    @Published private(set) var thermalState: Int32 = 0
    @Published private(set) var lowPowerMode = false
    /// Thermal advice for a banner, empty when there is nothing to say.
    @Published private(set) var thermalMessage = ""
    @Published private(set) var shaderStage = Int32(LICHII_SHADER_IDLE.rawValue)
    @Published private(set) var shaderDone: UInt64 = 0
    @Published private(set) var shaderTotal: UInt64 = 0

    @Published var mixWithOtherAudio = UserDefaults.standard.bool(forKey: "mixWithOtherAudio") {
        didSet {
            UserDefaults.standard.set(mixWithOtherAudio, forKey: "mixWithOtherAudio")
            lichii_platform_configure_audio(mixWithOtherAudio ? 1 : 0, 0)
        }
    }

    /// Set when this layer paused the game for the app going inactive; it resumes the game when
    /// the app is active again unless the user paused it.
    private var pausedForLifecycle = false

    func start() {
        lichii_platform_init()
        lichii_platform_set_event_handler(platformEvent, nil)
        lichii_platform_set_shader_progress_handler(shaderProgress, nil)
        if mixWithOtherAudio {
            lichii_platform_configure_audio(1, 0)
        }
        refreshThermal()
    }

    /// Every LichiiEvent of the app bridge.
    func handle(appEvent event: Int32) {
        guard event == Int32(LICHII_EVENT_STATE_CHANGED.rawValue) else { return }
        switch Bridge.emuState {
        case .waitingForJit:
            lichii_platform_reset_shader_progress()
        case .running:
            lichii_platform_set_playing(1)
        case .idle, .failed:
            pausedForLifecycle = false
            lichii_platform_set_playing(0)
        case .booting, .paused:
            break
        }
    }

    fileprivate func handle(platformEvent event: Int32) {
        switch event {
        case Int32(LICHII_PLATFORM_EVENT_WILL_RESIGN_ACTIVE.rawValue),
             Int32(LICHII_PLATFORM_EVENT_DID_ENTER_BACKGROUND.rawValue):
            pauseForLifecycle()
        case Int32(LICHII_PLATFORM_EVENT_DID_BECOME_ACTIVE.rawValue):
            resumeAfterLifecycle()
        case Int32(LICHII_PLATFORM_EVENT_THERMAL_STATE.rawValue),
             Int32(LICHII_PLATFORM_EVENT_LOW_POWER_MODE.rawValue):
            let before = thermalState
            refreshThermal()
            if thermalState > before, !thermalMessage.isEmpty, ControlsCenter.shared.gameRunning {
                ControlsCenter.shared.show(toast: thermalMessage)
            }
        case Int32(LICHII_PLATFORM_EVENT_MEMORY_WARNING.rawValue):
            if ControlsCenter.shared.gameRunning {
                ControlsCenter.shared.show(toast: "Memory is low: caches trimmed")
            }
        default:
            break
        }
    }

    private func pauseForLifecycle() {
        guard Bridge.emuState == .running else { return }
        lichii_emu_pause()
        pausedForLifecycle = true
    }

    private func resumeAfterLifecycle() {
        guard pausedForLifecycle else { return }
        pausedForLifecycle = false
        guard Bridge.emuState == .paused, !ControlsCenter.shared.userPaused else { return }
        lichii_emu_resume()
    }

    private func refreshThermal() {
        var advice = LichiiThermalAdvice()
        lichii_platform_thermal_advice(&advice)
        thermalState = advice.thermal_state
        lowPowerMode = advice.low_power_mode != 0
        thermalMessage = Self.text(advice.message)
    }

    /// A short form of the thermal advice for the game overlay and the HUD.
    var thermalBadge: String {
        switch thermalState {
        case 3...: return "Very hot"
        case 2: return "Hot · 30 fps advised"
        default: return lowPowerMode ? "Low Power Mode" : ""
        }
    }

    fileprivate func shaderProgressChanged(_ stage: Int32, _ done: UInt64, _ total: UInt64) {
        shaderStage = stage
        shaderDone = done
        shaderTotal = total
    }

    // MARK: - Device

    static var deviceInfo: LichiiDeviceInfo? {
        var info = LichiiDeviceInfo()
        return lichii_platform_device_info(&info) != 0 ? info : nil
    }

    static var deviceDefaults: LichiiDeviceDefaults? {
        var defaults = LichiiDeviceDefaults()
        return lichii_platform_device_defaults(&defaults) != 0 ? defaults : nil
    }

    /// A fixed-size C string field (imported as a tuple).
    nonisolated static func text<T>(_ field: T) -> String {
        var copy = field
        return withUnsafeBytes(of: &copy) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    // MARK: - Shader caches

    /// Bytes in the shader caches of `programId` (0: every game), read off the main thread.
    nonisolated static func shaderCacheBytes(_ programId: UInt64) async -> UInt64 {
        await Task.detached(priority: .userInitiated) {
            lichii_platform_shader_cache_bytes(programId)
        }.value
    }

    /// Not while that game (or, for 0, any game) runs.
    static func canClearShaderCache(_ programId: UInt64) -> Bool {
        let running = lichii_emu_program_id()
        return programId == 0 ? !Bridge.emuState.inGame && Bridge.emuState != .booting
                              : running != programId
    }

    @discardableResult
    static func clearShaderCache(_ programId: UInt64) -> Int32 {
        guard canClearShaderCache(programId) else { return -1 }
        return lichii_platform_clear_shader_cache(programId)
    }
}

/// Called on the main thread.
private let platformEvent: LichiiPlatformEventFn = { event, _ in
    MainActor.assumeIsolated { PlatformCenter.shared.handle(platformEvent: event) }
}

/// Called on the thread loading the disk shader cache.
private let shaderProgress: LichiiShaderProgressFn = { stage, done, total, _ in
    DispatchQueue.main.async {
        MainActor.assumeIsolated { PlatformCenter.shared.shaderProgressChanged(stage, done, total) }
    }
}
