// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

/// Called by the SDL frontend once the core is initialised (SDL_AppInit).
@_cdecl("lichii_ui_main")
public func lichiiUIMain() {
    MainActor.assumeIsolated {
        AppController.shared.start()
    }
}

/// Owns the app window and follows the emulator: the library while idle, the game's render window
/// (created by the SDL frontend) while a game runs.
@MainActor
final class AppController: ObservableObject {
    static let shared = AppController()

    @Published private(set) var emuState: EmuState = .idle
    /// The game being launched or played.
    @Published var currentGame: Game?
    @Published var showLaunch = false
    /// Follows LICHII_EVENT_JIT_CHANGED; whether the JIT app is installed is checked again when
    /// Lichii comes to the foreground or the JIT app setting changes.
    @Published private(set) var jit = JitStatus()
    /// A boot was cancelled: the launch screen stays up until the game is shut down again (IDLE).
    @Published private(set) var cancellingBoot = false
    /// The launch is a Build Shaders run: the game does not start after it.
    @Published private(set) var buildingShaders = false
    /// False while the library window is hidden behind the game render window, allowing LibraryView
    /// to unmount off-screen icon cells so decoded UIImages are not pinned during gameplay.
    @Published private(set) var libraryVisible = true

    let library = GameLibrary()
    private var window: UIWindow?
    private var jitRequested = false

    func start() {
        lichii_set_event_handler { event in
            MainActor.assumeIsolated {
                AppController.shared.handle(event)
            }
        }
        let window = makeWindow()
        let root = RootView()
            .environmentObject(self)
            .environmentObject(library)
        let host = UIHostingController(rootView: AnyView(root))
        host.view.backgroundColor = .systemBackground
        window.rootViewController = host
        window.makeKeyAndVisible()
        self.window = window

        refreshJit(checkProvider: true)
        library.reload()
        NotificationCenter.default.addObserver(forName: UIApplication.willEnterForegroundNotification,
                                               object: nil, queue: .main) { _ in
            MainActor.assumeIsolated {
                AppController.shared.refreshJit(checkProvider: true)
                AppController.shared.requestJitIfNeeded(automatic: true)
            }
        }
        requestJitIfNeeded(automatic: true)
        AppExtensions.appStarted.forEach { $0() }
    }

    private func makeWindow() -> UIWindow {
        let scene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive } ??
            UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        if let scene {
            return UIWindow(windowScene: scene)
        }
        return UIWindow(frame: UIScreen.main.bounds)
    }

    /// Reads the JIT state again; checkProvider also asks iOS whether the JIT app is installed.
    func refreshJit(checkProvider: Bool = false) {
        let status = JitStatus.read(installed: checkProvider ? JitProvider.current.installed : jit.stikDebug)
        if status != jit {
            jit = status
        }
    }

    /// The JIT pool is prepared while the library is open, so a launch does not wait for it. The
    /// automatic request waits until onboarding has explained it.
    func requestJitIfNeeded(automatic: Bool) {
        let provider = JitProvider.current
        guard Bridge.jitState == .waiting, provider.installed else { return }
        if automatic {
            guard UserDefaults.standard.object(forKey: "autoRequestJit") as? Bool ?? true,
                  Onboarding.completed, !jitRequested else { return }
        }
        jitRequested = true
        provider.request()
    }

    /// Keep Guest Memory in a File is on but storage is short: asks before the game starts.
    @Published var storageWarning: StorageWarning?

    struct StorageWarning: Identifiable {
        let game: Game
        let storage: LichiiFileGuestStorage
        var buildShaders = false
        var id: String { game.id }
    }

    /// buildShaders: boot the game only to build all of its cached shaders (with progress on the
    /// launch screen), then return to the library.
    func launch(_ game: Game, storageChecked: Bool = false, buildShaders: Bool = false) {
        guard !emuState.inGame, emuState != .booting else { return }
        if !storageChecked {
            let storage = GuestMemoryFile.enabled ? GuestMemoryFile.storage : nil
            let short = storage.map { GuestMemoryFile.fit($0) == .short } ?? false
            GuestMemoryFile.logLaunchCheck(title: game.title, storage: storage, askedUser: short)
            if short, let storage {
                storageWarning = StorageWarning(game: game, storage: storage, buildShaders: buildShaders)
                return
            }
        }
        currentGame = game
        if !buildShaders {
            library.markPlayed(game)
        }
        let launched = game.path.withCString {
            buildShaders ? lichii_emu_build_shaders($0) : lichii_emu_launch($0)
        }
        guard launched else { return }
        buildingShaders = buildShaders
        showLaunch = true
        if Bridge.jitState == .waiting {
            jitRequested = false
            requestJitIfNeeded(automatic: false)
        }
    }

    func cancelLaunch() {
        lichii_emu_cancel_launch()
        if Bridge.emuState == .booting {
            cancellingBoot = true
        } else {
            showLaunch = false
        }
    }

    func stop() {
        lichii_platform_flush_caches()
        lichii_emu_stop()
    }

    private func handle(_ event: Int32) {
        switch event {
        case Int32(LICHII_EVENT_STATE_CHANGED.rawValue):
            stateChanged()
        case Int32(LICHII_EVENT_GAME_WINDOW_READY.rawValue):
            gameWindowReady()
        case Int32(LICHII_EVENT_MENU_REQUESTED.rawValue):
            presentInGameMenu()
        case Int32(LICHII_EVENT_OPEN_URL.rawValue):
            open(Bridge.string(lichii_take_open_url()))
        case Int32(LICHII_EVENT_JIT_CHANGED.rawValue):
            refreshJit()
        default:
            break
        }
        AppExtensions.eventHandlers.forEach { $0(event) }
    }

    /// lichii://game?id=<title id> launches a game from the library; a file opened with the app is
    /// imported.
    private func open(_ text: String) {
        guard !text.isEmpty else { return }
        if let url = URL(string: text), url.scheme == "lichii" {
            let id = URLComponents(url: url, resolvingAgainstBaseURL: false)?
                .queryItems?.first { $0.name == "id" }?.value?.uppercased()
            guard url.host == "game", let id else { return }
            // On a cold start the first scan is still running.
            library.reload { [weak self] in
                guard let self else { return }
                if let game = library.games.first(where: { $0.programIdHex == id }) {
                    launch(game)
                } else {
                    library.notice = LibraryNotice(title: "Game not found",
                                                   message: "No game with the title ID \(id) is in the library.")
                }
            }
            return
        }
        let file = text.hasPrefix("file://") ? URL(string: text) : URL(fileURLWithPath: text)
        if let file {
            library.importItems([file])
        }
    }

    private func stateChanged() {
        let state = Bridge.emuState
        let previous = emuState
        emuState = state
        if state != .booting {
            cancellingBoot = false
        }
        switch state {
        case .idle:
            showLibrary()
            showLaunch = false
            if previous.inGame {
                currentGame = nil
            }
            if buildingShaders {
                buildingShaders = false
                reportBuiltShaders(game: currentGame)
            }
        case .failed:
            buildingShaders = false
            showLibrary()
            showLaunch = true
        case .waitingForJit, .booting:
            showLaunch = true
        case .running, .paused:
            break
        }
    }

    /// After a Build Shaders run: what it built, as the log's "Build shaders:" line says it.
    private func reportBuiltShaders(game: Game?) {
        var status = LichiiBuildShadersStatus()
        lichii_emu_build_shaders_status(&status)
        // Cancelled before the shader cache loaded.
        guard status.total > 0 || status.skipped > 0 else { return }
        let seconds = Int(status.elapsed_ms / 1000)
        var lines = ["\(status.built) built, \(status.failed) failed in \(seconds / 60):"
                     + String(format: "%02d", seconds % 60) + " on \(status.threads) threads."]
        if status.skipped > 0 {
            lines.append("\(status.skipped) were already built and skipped.")
        }
        if status.done < status.total {
            lines.append("Stopped with \(status.total - status.done) not built; Build Shaders again continues.")
        }
        library.notice = LibraryNotice(title: "Shaders for \(game?.title ?? "the game")",
                                       message: lines.joined(separator: " "))
    }

    private func showLibrary() {
        libraryVisible = true
        guard let window, window.isHidden else { return }
        window.isHidden = false
        window.makeKeyAndVisible()
    }

    private func gameWindowReady() {
        showLaunch = false
        libraryVisible = false
        window?.isHidden = true
        IconCache.evictAll()
        guard let pointer = lichii_emu_window() else { return }
        let gameWindow = Unmanaged<UIWindow>.fromOpaque(pointer).takeUnretainedValue()
        // The full in-game menu brings its own menu button.
        if AppExtensions.presentInGameMenu == nil {
            EmulationOverlay.install(in: gameWindow)
        }
        AppExtensions.gameWindowReady.forEach { $0(gameWindow) }
    }

    /// The window the game renders into, if one is up.
    var gameWindow: UIWindow? {
        guard let pointer = lichii_emu_window() else { return nil }
        return Unmanaged<UIWindow>.fromOpaque(pointer).takeUnretainedValue()
    }

    private func presentInGameMenu() {
        guard emuState.inGame, let presenter = gameWindow?.rootViewController else { return }
        if let custom = AppExtensions.presentInGameMenu {
            custom(presenter)
        } else {
            EmulationOverlay.presentBasicMenu(from: presenter)
        }
    }
}
