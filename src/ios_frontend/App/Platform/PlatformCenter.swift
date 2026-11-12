// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

/// The platform layer (lichii_bridge_platform.h) in the app: pauses the game while the app is
/// inactive or its audio is interrupted and brings it back paused, keeps the screen awake while
/// playing, follows the thermal state and the shader cache loading progress, notices a game that
/// ended without a clean stop, and reads the device facts.
@MainActor
final class PlatformCenter: ObservableObject {
    static let shared = PlatformCenter()

    @Published private(set) var thermalState: Int32 = 0
    @Published private(set) var lowPowerMode = false
    /// Thermal advice for a banner, empty when there is nothing to say.
    @Published private(set) var thermalMessage = ""
    /// The thermal badge over the game shows for a few seconds after it changes.
    @Published private(set) var thermalBadgeVisible = false
    @Published private(set) var shaderStage = Int32(LICHII_SHADER_IDLE.rawValue)
    @Published private(set) var shaderDone: UInt64 = 0
    @Published private(set) var shaderTotal: UInt64 = 0
    /// The previous game session ended without a clean stop (crash, or iOS closed Lichii).
    @Published var uncleanExit: GameSession?

    @Published var mixWithOtherAudio = UserDefaults.standard.bool(forKey: "mixWithOtherAudio") {
        didSet {
            UserDefaults.standard.set(mixWithOtherAudio, forKey: "mixWithOtherAudio")
            lichii_platform_configure_audio(mixWithOtherAudio ? 1 : 0, 0)
        }
    }

    /// Steps live frame-rate pacing down (45 FPS when hot / Low Power Mode, 30 FPS when critical)
    /// and restores full speed once cool, without modifying saved settings. Default on.
    @Published var autoCool: Bool = {
        if UserDefaults.standard.object(forKey: "autoCoolThermalPacing") == nil {
            return true
        }
        return UserDefaults.standard.bool(forKey: "autoCoolThermalPacing")
    }() {
        didSet {
            UserDefaults.standard.set(autoCool, forKey: "autoCoolThermalPacing")
            applyAutoCoolPacing()
        }
    }

    /// Set when this layer paused the game (app inactive, audio interrupted); the game comes back
    /// paused with the menu open, and runs again when the user closes it.
    private var pausedForLifecycle = false
    private var badgeTask: Task<Void, Never>?
    /// When the memory warning toast named Keep Guest Memory in a File in this game; the short
    /// toast of the warnings after it waits until it has been read.
    private var memoryHintShown: Date?

    func start() {
        uncleanExit = GameSession.takeUncleanExit()
        if let session = uncleanExit {
            let file = session.fileGuest.map { $0 ? "on" : "off" } ?? "unknown"
            lichii_platform_log_memory("the last session (\(session.title), started \(Int(-session.started.timeIntervalSinceNow)) s ago) ended without a clean stop; memory warning \(session.memoryWarning ? "yes" : "no"), Keep Guest Memory in a File \(file): shown the Last Session sheet")
        }
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
        if event == Int32(LICHII_EVENT_GAME_WINDOW_READY.rawValue) {
            GameSession.begin(AppController.shared.currentGame)
            memoryHintShown = nil
            flashThermalBadge()
            return
        }
        guard event == Int32(LICHII_EVENT_STATE_CHANGED.rawValue) else { return }
        switch Bridge.emuState {
        case .waitingForJit:
            lichii_platform_reset_shader_progress()
        case .running:
            lichii_platform_set_playing(1)
        case .paused:
            // The screen may lock while the game waits.
            lichii_platform_set_playing(0)
        case .idle, .failed:
            pausedForLifecycle = false
            lichii_platform_set_playing(0)
            GameSession.end()
        case .booting:
            break
        }
    }

    fileprivate func handle(platformEvent event: Int32) {
        switch event {
        case Int32(LICHII_PLATFORM_EVENT_WILL_RESIGN_ACTIVE.rawValue),
             Int32(LICHII_PLATFORM_EVENT_AUDIO_INTERRUPTION_BEGAN.rawValue):
            pauseForLifecycle()
        case Int32(LICHII_PLATFORM_EVENT_DID_ENTER_BACKGROUND.rawValue):
            pauseForLifecycle()
            GameSession.update { $0.background = true }
        case Int32(LICHII_PLATFORM_EVENT_WILL_ENTER_FOREGROUND.rawValue):
            GameSession.update { $0.background = false }
        case Int32(LICHII_PLATFORM_EVENT_DID_BECOME_ACTIVE.rawValue):
            returnAfterInterruption()
        case Int32(LICHII_PLATFORM_EVENT_AUDIO_INTERRUPTION_ENDED.rawValue):
            // A call banner or Siri over the active app; otherwise DID_BECOME_ACTIVE follows.
            if UIApplication.shared.applicationState == .active {
                returnAfterInterruption()
            }
        case Int32(LICHII_PLATFORM_EVENT_THERMAL_STATE.rawValue),
             Int32(LICHII_PLATFORM_EVENT_LOW_POWER_MODE.rawValue):
            let before = (thermalState, lowPowerMode, thermalBadge)
            refreshThermal()
            if ControlsCenter.shared.gameRunning {
                if thermalState > before.0, thermalState >= 2 {
                    ControlsCenter.shared.show(toast: thermalMessage)
                } else if lowPowerMode, !before.1 {
                    ControlsCenter.shared.show(toast: "Low Power Mode is on: the CPU and GPU run slower.")
                }
            }
            if thermalBadge != before.2 {
                flashThermalBadge()
            }
        case Int32(LICHII_PLATFORM_EVENT_MEMORY_WARNING.rawValue):
            GameSession.update { $0.memoryWarning = true }
            if ControlsCenter.shared.gameRunning {
                // The way out once per game; iOS sends several warnings in a row.
                if memoryHintShown == nil, !GuestMemoryFile.enabled {
                    memoryHintShown = Date()
                    ControlsCenter.shared.show(toast: "Memory is low: caches trimmed. If iOS closes the game, turn on Keep Guest Memory in a File in Settings > Memory & Experiments.")
                } else if memoryHintShown.map({ -$0.timeIntervalSinceNow > 6 }) ?? true {
                    ControlsCenter.shared.show(toast: "Memory is low: caches trimmed")
                }
            }
        case Int32(LICHII_PLATFORM_EVENT_WILL_TERMINATE.rawValue):
            GameSession.end()
        default:
            break
        }
    }

    private func pauseForLifecycle() {
        guard Bridge.emuState == .running else { return }
        lichii_emu_pause()
        pausedForLifecycle = true
    }

    /// After a call, the app switcher, Control Center or Siri the game waits paused under the
    /// in-game menu; closing the menu runs it again.
    private func returnAfterInterruption() {
        guard pausedForLifecycle else { return }
        pausedForLifecycle = false
        guard Bridge.emuState == .paused, !ControlsCenter.shared.userPaused else { return }
        ControlsCenter.shared.openMenu(resumeOnClose: true)
    }

    private func refreshThermal() {
        var advice = LichiiThermalAdvice()
        lichii_platform_thermal_advice(&advice)
        thermalState = advice.thermal_state
        lowPowerMode = advice.low_power_mode != 0
        thermalMessage = Self.text(advice.message)
        applyAutoCoolPacing()
    }

    private var autoCoolLogged: (Int32, Bool)?

    private func applyAutoCoolPacing() {
        guard autoCool else {
            lichii_platform_set_thermal_max_fps(0)
            return
        }
        // Which of the two set a cap ("Auto-Cool: capping ..." follows in the log).
        if autoCoolLogged.map({ $0 != (thermalState, lowPowerMode) }) ?? true {
            autoCoolLogged = (thermalState, lowPowerMode)
            lichii_platform_log_memory("Auto-Cool: thermal state \(thermalState) (2 = serious, 3 = critical), Low Power Mode \(lowPowerMode ? "on" : "off")")
        }
        if thermalState >= 3 {
            lichii_platform_set_thermal_max_fps(30)
        } else if thermalState == 2 || lowPowerMode {
            lichii_platform_set_thermal_max_fps(45)
        } else {
            lichii_platform_set_thermal_max_fps(0)
        }
    }

    /// A short form of the thermal state for the in-game menu and the HUD.
    var thermalBadge: String {
        switch thermalState {
        case 3...: return "Very hot"
        case 2: return "Hot"
        default: return lowPowerMode ? "Low Power Mode" : ""
        }
    }

    /// Shows the badge over the game for five seconds.
    func flashThermalBadge() {
        badgeTask?.cancel()
        thermalBadgeVisible = thermalState >= 2
        guard thermalBadgeVisible else { return }
        badgeTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            if !Task.isCancelled { PlatformCenter.shared.thermalBadgeVisible = false }
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

/// Experiments > Keep Guest Memory in a File, and whether storage has room for it.
@MainActor
enum GuestMemoryFile {
    enum Fit {
        /// The whole file is reserved up front.
        case full
        /// Enough to start with it; the rest of the file is reserved as the game uses it.
        case partial
        /// The game starts with its memory in RAM instead.
        case short
        case unknown
    }

    static var enabled: Bool {
        get { lichii_platform_file_guest() != 0 }
        set { lichii_platform_set_file_guest(newValue ? 1 : 0) }
    }

    static var storage: LichiiFileGuestStorage? {
        var storage = LichiiFileGuestStorage()
        return lichii_platform_file_guest_storage(&storage) != 0 ? storage : nil
    }

    static func fit(_ storage: LichiiFileGuestStorage?) -> Fit {
        guard let storage else { return .unknown }
        if storage.free_mib >= storage.full_mib { return .full }
        return storage.free_mib >= storage.needed_mib ? .partial : .short
    }

    /// "12.3 GB" from MiB.
    static func size(_ mib: UInt64) -> String {
        (mib << 20).byteCount
    }

    /// One line for the log before a game starts, so the next device log shows the decision.
    static func logLaunchCheck(title: String, storage: LichiiFileGuestStorage?, askedUser: Bool) {
        let line: String
        if !enabled {
            line = "launch check for \(title): Keep Guest Memory in a File off"
        } else if let storage {
            line = "launch check for \(title): Keep Guest Memory in a File on, \(storage.free_mib) MiB free, \(storage.needed_mib) MiB needed, \(storage.full_mib) MiB for the whole file" +
                (askedUser ? ": too little, asked whether to start anyway" : "")
        } else {
            line = "launch check for \(title): Keep Guest Memory in a File on, free storage unknown"
        }
        lichii_platform_log_memory(line)
    }
}

/// A game that is up, kept in UserDefaults until it stops cleanly. Still there at the next launch
/// means the app ended with it (a crash, or iOS closed Lichii, usually for memory).
struct GameSession: Codable, Identifiable {
    var title: String
    var programId: UInt64
    var resolution: String
    var started: Date
    var background = false
    var memoryWarning = false
    /// Keep Guest Memory in a File was on (nil: a session saved by an older build).
    var fileGuest: Bool?

    var id: Date { started }

    private static let key = "runningGameSession"

    @MainActor static func begin(_ game: Game?) {
        let session = GameSession(title: game?.title ?? "The game", programId: lichii_emu_program_id(),
                                  resolution: QuickSettings.string("resolution_setup"), started: Date(),
                                  fileGuest: GuestMemoryFile.enabled)
        save(session)
    }

    static func end() {
        UserDefaults.standard.removeObject(forKey: key)
    }

    static func update(_ change: (inout GameSession) -> Void) {
        guard var session = load() else { return }
        change(&session)
        save(session)
    }

    /// The session the last run left behind. One that ended in the background is dropped: the
    /// user left the game, and iOS (or the app switcher) closing a suspended app is expected.
    static func takeUncleanExit() -> GameSession? {
        defer { end() }
        guard let session = load(), !session.background else { return nil }
        return session
    }

    private static func load() -> GameSession? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(GameSession.self, from: data)
    }

    private static func save(_ session: GameSession) {
        if let data = try? JSONEncoder().encode(session) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
}

/// Called on the main thread.
private let platformEvent: LichiiPlatformEventFn = { event, _ in
    MainActor.assumeIsolated { PlatformCenter.shared.handle(platformEvent: event) }
}

/// Called on the thread loading the disk shader cache, at most ten times a second.
private let shaderProgress: LichiiShaderProgressFn = { stage, done, total, _ in
    DispatchQueue.main.async {
        MainActor.assumeIsolated { PlatformCenter.shared.shaderProgressChanged(stage, done, total) }
    }
}
