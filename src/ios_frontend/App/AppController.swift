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

        library.reload()
        NotificationCenter.default.addObserver(forName: UIApplication.willEnterForegroundNotification,
                                               object: nil, queue: .main) { _ in
            MainActor.assumeIsolated {
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

    /// The JIT pool is prepared while the library is open, so a launch does not wait for it.
    func requestJitIfNeeded(automatic: Bool) {
        let provider = JitProvider.current
        guard Bridge.jitState == .waiting, provider.installed else { return }
        if automatic {
            guard UserDefaults.standard.object(forKey: "autoRequestJit") as? Bool ?? true,
                  !jitRequested else { return }
        }
        jitRequested = true
        provider.request()
    }

    func launch(_ game: Game) {
        guard !emuState.inGame, emuState != .booting else { return }
        currentGame = game
        library.markPlayed(game)
        guard game.path.withCString({ lichii_emu_launch($0) }) else { return }
        showLaunch = true
        if Bridge.jitState == .waiting {
            jitRequested = false
            requestJitIfNeeded(automatic: false)
        }
    }

    func cancelLaunch() {
        lichii_emu_cancel_launch()
        showLaunch = false
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
            if url.host == "game", let id, let game = library.games.first(where: { $0.programIdHex == id }) {
                launch(game)
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
        switch state {
        case .idle:
            showLibrary()
            showLaunch = false
            if previous.inGame {
                currentGame = nil
            }
        case .failed:
            showLibrary()
            showLaunch = true
        case .waitingForJit, .booting:
            showLaunch = true
        case .running, .paused:
            break
        }
    }

    private func showLibrary() {
        guard let window, window.isHidden else { return }
        window.isHidden = false
        window.makeKeyAndVisible()
    }

    private func gameWindowReady() {
        showLaunch = false
        window?.isHidden = true
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
