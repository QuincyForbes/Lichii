// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import Photos
import SwiftUI
import UIKit

/// Runs the controls while a game is up: the overlay on the render window (on-screen controller,
/// performance HUD, menu button, in-game menu), frontend actions from controllers, rumble, and
/// moving the game to an external display.
@MainActor
final class ControlsCenter: ObservableObject {
    static let shared = ControlsCenter()

    @Published private(set) var gameRunning = false
    @Published private(set) var menuOpen = false
    @Published var editingLayout = false
    @Published private(set) var touchControlsVisible = true
    @Published var toast: String?
    @Published var showHUD = UserDefaults.standard.bool(forKey: "showPerformanceHUD") {
        didSet { UserDefaults.standard.set(showHUD, forKey: "showPerformanceHUD") }
    }

    /// The on-screen controller hides while a controller drives player 1.
    var hideTouchWithController: Bool {
        get { UserDefaults.standard.object(forKey: "hideTouchWithController") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "hideTouchWithController"); updateTouchVisibility() }
    }

    var showTouchControls: Bool {
        get { UserDefaults.standard.object(forKey: "showTouchControls") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "showTouchControls"); updateTouchVisibility() }
    }

    private(set) weak var gameWindow: UIWindow?
    private var overlay: OverlayController?
    private var ticker: CADisplayLink?
    private var resumeAfterMenu = false
    private var toastTask: Task<Void, Never>?

    var currentGame: Game? { AppController.shared.currentGame }

    // MARK: - App hooks

    func appStarted() {
        PhysicalControllers.shared.start()
        ExternalDisplay.shared.start()
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(forName: UIDevice.orientationDidChangeNotification,
                                               object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { DeviceOrientation.update() }
        }
        NotificationCenter.default.addObserver(forName: UIApplication.willResignActiveNotification,
                                               object: nil, queue: .main) { _ in
            MainActor.assumeIsolated {
                lichii_controls_release_touch()
                PhysicalControllers.shared.stopRumble()
            }
        }
        DeviceOrientation.update()
    }

    func gameWindowReady(_ window: UIWindow) {
        gameWindow = window
        gameRunning = true
        editingLayout = false
        menuOpen = false
        let overlay = overlay ?? OverlayController()
        self.overlay = overlay
        attachOverlay(to: ExternalDisplay.shared.controlsWindow ?? window)
        let tap = UITapGestureRecognizer(target: overlay, action: #selector(OverlayController.threeFingerTap))
        tap.numberOfTouchesRequired = 3
        tap.cancelsTouchesInView = false
        window.gestureRecognizers?.filter { ($0 as? UITapGestureRecognizer)?.numberOfTouchesRequired == 3 }
            .forEach(window.removeGestureRecognizer)
        window.addGestureRecognizer(tap)
        OrientationManager.shared.apply()
        ExternalDisplay.shared.gameStarted(window)
        PhysicalControllers.shared.updateDeviceMotion()
        updateTouchVisibility()
        startTicker()
    }

    func handle(event: Int32) {
        switch event {
        case Int32(LICHII_EVENT_STATE_CHANGED.rawValue):
            if !Bridge.emuState.inGame, Bridge.emuState != .booting, gameRunning {
                gameEnded()
            }
        case Int32(LICHII_EVENT_CONTROLLERS_CHANGED.rawValue):
            updateTouchVisibility()
        default:
            break
        }
    }

    private func gameEnded() {
        gameRunning = false
        menuOpen = false
        editingLayout = false
        ticker?.invalidate()
        ticker = nil
        lichii_controls_release_touch()
        PhysicalControllers.shared.stopRumble()
        PhysicalControllers.shared.updateDeviceMotion()
        overlay?.detach()
        ExternalDisplay.shared.gameEnded()
        gameWindow = nil
    }

    /// Puts the overlay on `window` (the game window, or the phone's window while the game is on
    /// an external display).
    func attachOverlay(to window: UIWindow) {
        overlay?.attach(to: window)
    }

    // MARK: - Menu

    func openMenu() {
        guard gameRunning, !menuOpen else { return }
        lichii_controls_release_touch()
        resumeAfterMenu = Bridge.emuState == .running
        if resumeAfterMenu {
            lichii_emu_pause()
        }
        menuOpen = true
        overlay?.interactive = true
    }

    func closeMenu(resume: Bool = true) {
        guard menuOpen else { return }
        menuOpen = false
        overlay?.interactive = editingLayout
        if resume && resumeAfterMenu {
            lichii_emu_resume()
        }
        resumeAfterMenu = false
    }

    func togglePause() {
        if Bridge.emuState == .paused {
            lichii_emu_resume()
            resumeAfterMenu = false
        } else {
            lichii_emu_pause()
        }
        objectWillChange.send()
    }

    func quitToLibrary() {
        closeMenu(resume: false)
        AppController.shared.stop()
    }

    func editLayout() {
        closeMenu(resume: false)
        editingLayout = true
        overlay?.interactive = true
        touchControlsVisible = true
    }

    func layoutEditingChanged(_ editing: Bool) {
        guard !editing, editingLayout else { return }
        editingLayout = false
        overlay?.interactive = menuOpen
        updateTouchVisibility()
        if resumeAfterMenu {
            lichii_emu_resume()
            resumeAfterMenu = false
        }
    }

    // MARK: - Ticker

    private func startTicker() {
        ticker?.invalidate()
        let link = CADisplayLink(target: TickTarget.shared, selector: #selector(TickTarget.tick))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        link.add(to: .main, forMode: .common)
        ticker = link
    }

    fileprivate func tick() {
        let actions = lichii_controls_take_actions()
        if actions != 0 {
            run(actions)
        }
        if Bridge.emuState == .running {
            PhysicalControllers.shared.pollRumble()
        } else {
            PhysicalControllers.shared.stopRumble()
        }
    }

    private func run(_ actions: UInt32) {
        func has(_ action: LichiiAction) -> Bool {
            actions & (1 << UInt32(action.rawValue - LICHII_ACTION_MENU.rawValue)) != 0
        }
        if has(LICHII_ACTION_MENU) {
            menuOpen ? closeMenu() : openMenu()
        }
        if has(LICHII_ACTION_SCREENSHOT) {
            takeScreenshot()
        }
        if has(LICHII_ACTION_TOGGLE_HUD) {
            showHUD.toggle()
        }
        if has(LICHII_ACTION_TOGGLE_SPEED_LIMIT) {
            let enabled = QuickSettings.bool("use_speed_limit")
            QuickSettings.set("use_speed_limit", !enabled)
            show(toast: enabled ? "Speed limit off" : "Speed limit on")
        }
    }

    func updateTouchVisibility() {
        let controller = PhysicalControllers.shared.playerOneHasController
        let visible = editingLayout || (showTouchControls && !(hideTouchWithController && controller))
        if !visible {
            lichii_controls_release_touch()
        }
        touchControlsVisible = visible
    }

    // MARK: - Screenshot

    func takeScreenshot() {
        if !lichii_emu_screenshot(1080, screenshotDone, nil) {
            show(toast: "Screenshot unavailable right now")
        }
    }

    fileprivate func save(screenshot: UIImage?) {
        guard let screenshot else {
            show(toast: "Screenshot failed")
            return
        }
        Self.addToPhotos(screenshot)
    }

    /// Photos calls back on its own queues.
    private nonisolated static func addToPhotos(_ screenshot: UIImage) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    MainActor.assumeIsolated { ControlsCenter.shared.show(toast: "No access to Photos") }
                }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: screenshot)
            }) { ok, _ in
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        ControlsCenter.shared.show(toast: ok ? "Screenshot saved to Photos" : "Screenshot failed")
                    }
                }
            }
        }
    }

    func show(toast text: String) {
        toast = text
        toastTask?.cancel()
        toastTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            if !Task.isCancelled { ControlsCenter.shared.toast = nil }
        }
    }
}

/// Runs on a renderer thread, so it is a non-isolated constant rather than a closure in the class.
private let screenshotDone: LichiiScreenshotFn = { _, pixels, width, height, flipY in
    let image = pixels.flatMap { ScreenshotBox.image(bgra: $0, width: Int(width), height: Int(height), flipY: flipY) }
    DispatchQueue.main.async {
        MainActor.assumeIsolated { ControlsCenter.shared.save(screenshot: image) }
    }
}

/// CADisplayLink retains its target; this keeps ControlsCenter out of that cycle.
private final class TickTarget: NSObject {
    static let shared = TickTarget()

    @objc func tick() {
        MainActor.assumeIsolated { ControlsCenter.shared.tick() }
    }
}

private final class ScreenshotBox {
    /// BGRA rows (bottom-up when flipY) -> UIImage.
    static func image(bgra: UnsafePointer<UInt8>, width: Int, height: Int, flipY: Bool) -> UIImage? {
        guard width > 0, height > 0 else { return nil }
        let rowBytes = width * 4
        var data = Data(count: rowBytes * height)
        data.withUnsafeMutableBytes { target in
            guard let base = target.baseAddress else { return }
            for row in 0..<height {
                let source = bgra + (flipY ? height - 1 - row : row) * rowBytes
                memcpy(base + row * rowBytes, source, rowBytes)
            }
        }
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: rowBytes, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info,
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: image)
    }
}

/// Hosts ControlsOverlayView on a window. Touches that hit no control fall through to the game
/// (its touch screen) unless a menu or the layout editor is open.
@MainActor
final class OverlayController: NSObject {
    private let host = UIHostingController(rootView: ControlsOverlayView())
    private let container = PassthroughView()

    var interactive: Bool {
        get { container.interactive }
        set { container.interactive = newValue }
    }

    override init() {
        super.init()
        host.view.backgroundColor = .clear
        host.view.frame = container.bounds
        host.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        container.backgroundColor = .clear
        container.isMultipleTouchEnabled = true
        container.hostView = host.view
        container.addSubview(host.view)
    }

    func attach(to window: UIWindow) {
        guard let parent = window.rootViewController else { return }
        detach()
        parent.addChild(host)
        container.frame = parent.view.bounds
        container.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        parent.view.addSubview(container)
        host.didMove(toParent: parent)
    }

    func detach() {
        guard host.parent != nil else { return }
        host.willMove(toParent: nil)
        container.removeFromSuperview()
        host.removeFromParent()
    }

    @objc func threeFingerTap() {
        ControlsCenter.shared.openMenu()
    }
}

final class PassthroughView: UIView {
    var interactive = false
    weak var hostView: UIView?

    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        let hit = super.hitTest(point, with: event)
        if interactive {
            return hit
        }
        return hit === self || hit === hostView ? nil : hit
    }
}
