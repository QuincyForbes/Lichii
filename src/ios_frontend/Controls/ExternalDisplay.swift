// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import SwiftUI
import UIKit

/// External displays and AirPlay: while one is connected the game's render window moves onto it
/// and the phone shows the on-screen controller, HUD and menu on black.
@MainActor
final class ExternalDisplay: ObservableObject {
    static let shared = ExternalDisplay()

    @Published private(set) var connected = false
    /// The phone-side window while the game is on the external display.
    private(set) var controlsWindow: UIWindow?
    private weak var gameWindow: UIWindow?
    private var originalScene: UIWindowScene?
    private var externalScene: UIWindowScene?
    private var externalScreen: UIScreen?
    private var started = false

    var enabled: Bool {
        get { UserDefaults.standard.object(forKey: "useExternalDisplay") as? Bool ?? true }
        set {
            UserDefaults.standard.set(newValue, forKey: "useExternalDisplay")
            objectWillChange.send()
            newValue ? moveGameOut() : moveGameBack()
        }
    }

    func start() {
        guard !started else { return }
        started = true
        let center = NotificationCenter.default
        center.addObserver(forName: UIScene.willConnectNotification, object: nil, queue: .main) { note in
            guard let scene = note.object as? UIWindowScene else { return }
            MainActor.assumeIsolated { ExternalDisplay.shared.sceneConnected(scene) }
        }
        center.addObserver(forName: UIScene.didActivateNotification, object: nil, queue: .main) { note in
            guard let scene = note.object as? UIWindowScene else { return }
            MainActor.assumeIsolated { ExternalDisplay.shared.sceneConnected(scene) }
        }
        center.addObserver(forName: UIScene.didDisconnectNotification, object: nil, queue: .main) { note in
            guard let scene = note.object as? UIWindowScene else { return }
            MainActor.assumeIsolated { ExternalDisplay.shared.sceneDisconnected(scene) }
        }
        center.addObserver(forName: UIScreen.didConnectNotification, object: nil, queue: .main) { note in
            guard let screen = note.object as? UIScreen else { return }
            MainActor.assumeIsolated { ExternalDisplay.shared.screenConnected(screen) }
        }
        center.addObserver(forName: UIScreen.didDisconnectNotification, object: nil, queue: .main) { note in
            guard let screen = note.object as? UIScreen else { return }
            MainActor.assumeIsolated { ExternalDisplay.shared.screenDisconnected(screen) }
        }
        findExisting()
    }

    private static func isExternal(_ scene: UIWindowScene) -> Bool {
        scene.session.role == .windowExternalDisplayNonInteractive ||
            scene.screen != UIScreen.main && scene.screen.bounds.size != .zero && scene.session.role != .windowApplication
    }

    private func findExisting() {
        if let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first(where: Self.isExternal) {
            sceneConnected(scene)
        } else if let screen = UIScreen.screens.first(where: { $0 != UIScreen.main }) {
            screenConnected(screen)
        }
    }

    private func sceneConnected(_ scene: UIWindowScene) {
        guard Self.isExternal(scene), externalScene == nil else { return }
        externalScene = scene
        externalScreen = scene.screen
        connected = true
        moveGameOut()
    }

    private func sceneDisconnected(_ scene: UIWindowScene) {
        guard scene === externalScene else { return }
        moveGameBack(reattach: ControlsCenter.shared.gameRunning)
        externalScene = nil
        externalScreen = nil
        connected = false
    }

    /// Apps without a scene manifest get screens instead of scenes.
    private func screenConnected(_ screen: UIScreen) {
        guard screen != UIScreen.main, externalScene == nil, externalScreen == nil else { return }
        screen.overscanCompensation = .none
        externalScreen = screen
        connected = true
        // The scene for a new screen can arrive right after the screen itself.
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                let shared = ExternalDisplay.shared
                guard shared.externalScreen === screen else { return }
                if let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene })
                    .first(where: { $0.screen == screen }) {
                    shared.externalScene = scene
                }
                shared.moveGameOut()
            }
        }
    }

    private func screenDisconnected(_ screen: UIScreen) {
        guard screen == externalScreen else { return }
        moveGameBack(reattach: ControlsCenter.shared.gameRunning)
        externalScreen = nil
        externalScene = nil
        connected = false
    }

    /// The phone window currently showing the controls (or the game window when not on TV).
    var activePhoneWindow: UIWindow? {
        controlsWindow ?? gameWindow
    }

    func gameStarted(_ window: UIWindow) {
        gameWindow = window
        originalScene = window.windowScene
        moveGameOut()
    }

    func gameEnded() {
        moveGameBack(reattach: false)
        gameWindow = nil
        originalScene = nil
    }

    private func moveGameOut() {
        guard enabled, connected, let gameWindow, controlsWindow == nil else { return }
        ControlsCenter.shared.releaseTouch()
        if let externalScene {
            gameWindow.windowScene = externalScene
        } else if let externalScreen {
            gameWindow.screen = externalScreen
        } else {
            return
        }
        gameWindow.frame = gameWindow.screen.bounds
        gameWindow.isHidden = false

        let phone: UIWindow
        if let originalScene {
            phone = UIWindow(windowScene: originalScene)
        } else {
            phone = UIWindow(frame: UIScreen.main.bounds)
        }
        let root = ExternalControlsHostController()
        root.view.backgroundColor = .black
        phone.rootViewController = root
        phone.makeKeyAndVisible()
        controlsWindow = phone
        ControlsCenter.shared.attachOverlay(to: phone)
        ControlsCenter.shared.installThreeFingerTap(on: phone)
        OrientationManager.shared.apply()
    }

    private func moveGameBack(reattach: Bool = true) {
        guard let phone = controlsWindow else { return }
        controlsWindow = nil
        if let gameWindow {
            if let originalScene {
                gameWindow.windowScene = originalScene
            } else {
                gameWindow.screen = UIScreen.main
            }
            gameWindow.frame = gameWindow.screen.bounds
            if reattach {
                gameWindow.makeKeyAndVisible()
                ControlsCenter.shared.attachOverlay(to: gameWindow)
            }
        }
        phone.isHidden = true
        phone.rootViewController = nil
        phone.windowScene = nil
    }
}

/// Hosts the on-screen controls on the phone while the game renders to an external display,
/// respecting orientation lock and hiding the status bar / home indicator.
private final class ExternalControlsHostController: UIViewController {
    override var prefersStatusBarHidden: Bool { true }
    override var prefersHomeIndicatorAutoHidden: Bool { true }
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        OrientationManager.shared.mask
    }
}
