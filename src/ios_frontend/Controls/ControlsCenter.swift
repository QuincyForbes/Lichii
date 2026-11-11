// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Combine
import LichiiBridge
import Melo_Controller
import Photos
import SwiftUI
import UIKit
import UIKit.UIGestureRecognizerSubclass

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
    /// The user asked the game to stay paused after the menu closes.
    @Published private(set) var manuallyPaused = false
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

    /// Paused by the user (menu open, layout editor, or kept paused): the platform layer does not
    /// resume the game when the app becomes active again.
    var userPaused: Bool { manuallyPaused || menuOpen || editingLayout }

    // MARK: - App hooks

    func appStarted() {
        ControllerDiagnostics.sink = { lichii_controls_log($0) }
        PhysicalControllers.shared.start()
        ExternalDisplay.shared.start()
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(forName: UIDevice.orientationDidChangeNotification,
                                               object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { DeviceOrientation.update() }
            // The interface finishes rotating after the device reports the new orientation.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                MainActor.assumeIsolated { DeviceOrientation.update() }
            }
        }
        NotificationCenter.default.addObserver(forName: UIApplication.willResignActiveNotification,
                                               object: nil, queue: .main) { _ in
            MainActor.assumeIsolated {
                ControlsCenter.shared.releaseTouch()
                PhysicalControllers.shared.stopRumble()
            }
        }
        DeviceOrientation.update()
    }

    /// Releases everything the on-screen controller holds; toggled buttons show released again.
    func releaseTouch() {
        lichii_controls_release_touch()
        NotificationCenter.default.post(name: .lichiiTouchReleased, object: nil)
    }

    func gameWindowReady(_ window: UIWindow) {
        gameWindow = window
        gameRunning = true
        editingLayout = false
        menuOpen = false
        ControllerDiagnostics.reset()
        let overlay = overlay ?? OverlayController()
        self.overlay = overlay
        attachOverlay(to: ExternalDisplay.shared.controlsWindow ?? window)
        let tap = UITapGestureRecognizer(target: overlay, action: #selector(OverlayController.threeFingerTap))
        tap.numberOfTouchesRequired = 3
        tap.cancelsTouchesInView = false
        // Releases reach the controls at once instead of after the tap fails.
        tap.delaysTouchesEnded = false
        window.gestureRecognizers?.filter { ($0 as? UITapGestureRecognizer)?.numberOfTouchesRequired == 3 }
            .forEach(window.removeGestureRecognizer)
        window.addGestureRecognizer(tap)
        OrientationManager.shared.apply()
        ExternalDisplay.shared.gameStarted(window)
        PhysicalControllers.shared.updateDeviceMotion()
        updateTouchVisibility()
        startTicker()
        // Loaded while the app was inactive: the game waits paused under the menu.
        if Bridge.emuState == .paused {
            openMenu(resumeOnClose: true)
        }
    }

    func handle(event: Int32) {
        switch event {
        case Int32(LICHII_EVENT_STATE_CHANGED.rawValue):
            if !Bridge.emuState.inGame, Bridge.emuState != .booting, gameRunning {
                gameEnded()
            } else if gameRunning {
                updateIdleWork()
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
        manuallyPaused = false
        ticker?.invalidate()
        ticker = nil
        releaseTouch()
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

    /// `resumeOnClose`: the game was paused for it (an interruption, a controller that went away)
    /// and runs again when the menu closes.
    func openMenu(resumeOnClose: Bool = false) {
        guard gameRunning, !menuOpen else { return }
        releaseTouch()
        let running = Bridge.emuState == .running
        resumeAfterMenu = running || resumeOnClose
        if running {
            lichii_emu_pause()
        }
        menuOpen = true
        overlay?.interactive = true
        UIAccessibility.post(notification: .screenChanged, argument: nil)
    }

    func closeMenu(resume: Bool = true) {
        guard menuOpen else { return }
        menuOpen = false
        overlay?.interactive = editingLayout
        if resume && resumeAfterMenu && !manuallyPaused {
            lichii_emu_resume()
        }
        resumeAfterMenu = false
    }

    /// Closes the menu and runs the game, also when it was kept paused or paused by the system.
    func resumeGame() {
        manuallyPaused = false
        closeMenu(resume: false)
        lichii_emu_resume()
    }

    /// Whether the game stays paused once the menu closes.
    func toggleKeepPaused() {
        manuallyPaused.toggle()
    }

    func quitToLibrary() {
        closeMenu(resume: false)
        AppController.shared.stop()
    }

    func editLayout() {
        // The game stays paused while editing and resumes when the editor closes.
        let resume = resumeAfterMenu
        closeMenu(resume: false)
        resumeAfterMenu = resume
        editingLayout = true
        overlay?.interactive = true
        touchControlsVisible = true
    }

    func layoutEditingChanged(_ editing: Bool) {
        guard !editing, editingLayout else { return }
        editingLayout = false
        overlay?.interactive = menuOpen
        updateTouchVisibility()
        if resumeAfterMenu && !manuallyPaused {
            lichii_emu_resume()
        }
        resumeAfterMenu = false
    }

    // MARK: - Ticker

    private func startTicker() {
        ticker?.invalidate()
        let link = CADisplayLink(target: TickTarget.shared, selector: #selector(TickTarget.tick))
        link.add(to: .main, forMode: .common)
        ticker = link
        updateIdleWork()
    }

    /// While the game is paused only the controller actions (the menu chord) are polled, slowly;
    /// rumble and the phone's motion sensors stop.
    private func updateIdleWork() {
        let running = Bridge.emuState == .running
        ticker?.preferredFrameRateRange = running ? CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
                                                  : CAFrameRateRange(minimum: 10, maximum: 15, preferred: 15)
        if !running {
            PhysicalControllers.shared.stopRumble()
        }
        PhysicalControllers.shared.updateDeviceMotion()
    }

    fileprivate func tick() {
        let actions = lichii_controls_take_actions()
        if actions != 0 {
            run(actions)
        }
        if Bridge.emuState == .running {
            PhysicalControllers.shared.pollRumble()
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
            OverlayPreset.toggle()
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
            releaseTouch()
        }
        if touchControlsVisible != visible {
            touchControlsVisible = visible
        }
        overlay?.invalidateControlTargets()
    }

    // MARK: - Controllers

    func controllerConnected(_ name: String, player: Int32) {
        guard gameRunning else { return }
        show(toast: player >= 0 ? "\(name) connected · Player \(player + 1)" : "\(name) connected")
    }

    /// A controller that drove a player went away: the game pauses under the menu until the user
    /// closes it (reconnect, or carry on with the on-screen controller).
    func controllerDisconnected(_ name: String, drovePlayer: Bool) {
        guard gameRunning else { return }
        if drovePlayer, Bridge.emuState == .running, !editingLayout {
            openMenu(resumeOnClose: true)
        }
        show(toast: "\(name) disconnected")
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
        UIAccessibility.post(notification: .announcement, argument: text)
        toastTask?.cancel()
        // Longer toasts (thermal advice) stay longer.
        let seconds = text.count > 60 ? 5.0 : 2.5
        toastTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
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
    /// The game menu button, a plain UIKit button above the SwiftUI content: no SwiftUI hosting
    /// or gesture recognizer sits between it and its touches.
    private let menuButton = UIButton(type: .system)
    private var observers: Set<AnyCancellable> = []

    var interactive: Bool {
        get { container.interactive }
        set { container.interactive = newValue }
    }

    func invalidateControlTargets() {
        container.invalidateTargets()
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
        setUpMenuButton()
    }

    private func setUpMenuButton() {
        let config = UIImage.SymbolConfiguration(pointSize: 22, weight: .semibold)
        menuButton.setImage(UIImage(systemName: "gearshape.circle.fill", withConfiguration: config), for: .normal)
        menuButton.tintColor = .white
        menuButton.alpha = 0.6
        menuButton.accessibilityLabel = "Game Menu"
        menuButton.accessibilityHint = "Pauses the game"
        menuButton.addAction(UIAction { _ in
            MainActor.assumeIsolated {
                ControllerDiagnostics.once("menu-button", "menu button tapped, game state \(Bridge.emuState)")
                ControlsCenter.shared.openMenu()
            }
        }, for: .touchUpInside)
        container.addSubview(menuButton)
        container.menuButton = menuButton

        let center = ControlsCenter.shared
        let visibleSetting = NotificationCenter.default.publisher(for: UserDefaults.didChangeNotification)
            .map { _ in Self.menuButtonVisible }
            .prepend(Self.menuButtonVisible)
            .removeDuplicates()
        Publishers.CombineLatest3(center.$menuOpen, center.$editingLayout, visibleSetting)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] menuOpen, editing, visible in
                MainActor.assumeIsolated {
                    self?.menuButton.isHidden = !visible || menuOpen || editing
                }
            }
            .store(in: &observers)
    }

    private nonisolated static var menuButtonVisible: Bool {
        UserDefaults.standard.object(forKey: "menuButtonVisible") as? Bool ?? true
    }

    func attach(to window: UIWindow) {
        guard let parent = window.rootViewController else { return }
        detach()
        WindowBoundsFollower.install(in: window)
        parent.addChild(host)
        container.frame = parent.view.bounds
        container.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        parent.view.addSubview(container)
        host.didMove(toParent: parent)
        if !(window.gestureRecognizers ?? []).contains(where: { $0 is TouchProbe }) {
            window.addGestureRecognizer(TouchProbe(container: container))
        }
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

/// Takes the on-screen controller's touches itself and hands each to the button or stick under it,
/// so they never depend on how the hosting view hit-tests the SwiftUI content around them or on
/// SwiftUI's gesture recognizers on it. Other touches go to the overlay's own controls (menu
/// button), or fall through to the game.
final class PassthroughView: UIView {
    var interactive = false {
        didSet {
            if interactive != oldValue {
                invalidateTargets()
            }
        }
    }
    weak var hostView: UIView? {
        didSet { invalidateTargets() }
    }
    weak var menuButton: UIView?
    /// Touches handed to a control, by touch.
    private var owners: [ObjectIdentifier: UIView] = [:]

    private struct WeakControlTarget {
        weak var view: ControllerTouchTarget?
        let priority: Int
    }

    private struct HitCache {
        let point: CGPoint
        let time: CFTimeInterval
        weak var view: UIView?
    }

    private var cachedTargets: [WeakControlTarget] = []
    private var targetsDirty = true
    private var lastBoundsSize: CGSize = .zero
    private var lastHit: HitCache?
    private var defaultsObserver: NSObjectProtocol?

    override init(frame: CGRect) {
        super.init(frame: frame)
        defaultsObserver = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.invalidateTargets()
            }
        }
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }

    deinit {
        if let defaultsObserver {
            NotificationCenter.default.removeObserver(defaultsObserver)
        }
    }

    func invalidateTargets() {
        targetsDirty = true
        lastHit = nil
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        // The container follows the root view; the root view follows the window.
        if let window {
            WindowBoundsFollower.fit(window)
        }
        if let parent = superview, frame != parent.bounds {
            frame = parent.bounds
        }
        if bounds.size != lastBoundsSize {
            lastBoundsSize = bounds.size
            invalidateTargets()
        }
        if let menuButton {
            menuButton.frame = CGRect(x: safeAreaInsets.left + 8, y: safeAreaInsets.top + 8, width: 44, height: 44)
            bringSubviewToFront(menuButton)
        }
    }

    override func safeAreaInsetsDidChange() {
        super.safeAreaInsetsDidChange()
        setNeedsLayout()
    }

    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        if interactive {
            return super.hitTest(point, with: event)
        }
        // The menu button before anything under it.
        if let menuButton, !menuButton.isHidden, menuButton.frame.insetBy(dx: -6, dy: -6).contains(point) {
            ControllerDiagnostics.once("overlay-menu", "first touch on the menu button at \(Self.text(point))")
            return menuButton
        }
        if !ControlsCenter.shared.touchControlsVisible && !ControlsCenter.shared.editingLayout {
            return nil
        }
        if let target = control(at: point) {
            ControllerDiagnostics.once("overlay-control",
                                       "first touch on a control (\(type(of: target))) at \(Self.text(point)) in \(Self.text(bounds.size))")
            return self
        }
        ControllerDiagnostics.once("overlay-pass",
                                   "first touch on no control at \(Self.text(point)) in \(Self.text(bounds.size)): to the game, \(cachedTargets.count) controls")
        return nil
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            guard let target = control(at: touch.location(in: self)) else { continue }
            owners[ObjectIdentifier(touch)] = target
            target.touchesBegan([touch], with: event)
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            owners[ObjectIdentifier(touch)]?.touchesMoved([touch], with: event)
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            owners.removeValue(forKey: ObjectIdentifier(touch))?.touchesEnded([touch], with: event)
        }
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            owners.removeValue(forKey: ObjectIdentifier(touch))?.touchesCancelled([touch], with: event)
        }
    }

    private func rebuildTargetsIfNeeded(in root: UIView) {
        if !targetsDirty && !cachedTargets.isEmpty {
            let hasStale = cachedTargets.contains { $0.view == nil || $0.view?.window == nil }
            if !hasStale {
                return
            }
        }
        var collected: [WeakControlTarget] = []
        collected.reserveCapacity(20)
        func collect(_ view: UIView) {
            if let target = view as? ControllerTouchTarget {
                collected.append(WeakControlTarget(view: target, priority: target.touchPriority))
                return
            }
            let children = view.subviews
            var idx = children.count
            while idx > 0 {
                idx -= 1
                collect(children[idx])
            }
        }
        collect(root)
        collected.sort { $0.priority > $1.priority }
        cachedTargets = collected
        targetsDirty = false
    }

    /// Returns nil when `view` or an ancestor up to `root` is hidden/transparent; otherwise whether
    /// every ancestor has `isUserInteractionEnabled` on.
    private static func ancestorInteractiveState(of view: UIView, upTo root: UIView) -> Bool? {
        var interactive = true
        var current: UIView? = view
        while let v = current {
            if v.isHidden || v.alpha <= 0.01 {
                return nil
            }
            if !v.isUserInteractionEnabled {
                interactive = false
            }
            if v === root {
                return interactive
            }
            current = v.superview
        }
        return nil
    }

    /// The topmost visible button under `point` (in this view), else the topmost stick.
    func control(at point: CGPoint) -> UIView? {
        guard let hostView else { return nil }
        let now = CACurrentMediaTime()
        if let hit = lastHit,
           abs(hit.point.x - point.x) < 0.5,
           abs(hit.point.y - point.y) < 0.5,
           now - hit.time < 0.05 {
            if let view = hit.view {
                if view.window != nil {
                    return view
                }
            } else {
                return nil
            }
        }

        rebuildTargetsIfNeeded(in: hostView)

        var fallback: UIView?
        var fallbackPriority = Int.min
        for entry in cachedTargets {
            guard let view = entry.view else { continue }
            if fallback != nil && entry.priority < fallbackPriority {
                break
            }
            guard let ancestorsInteractive = Self.ancestorInteractiveState(of: view, upTo: hostView) else {
                continue
            }
            let local = view.convert(point, from: self)
            guard view.bounds.contains(local), view.point(inside: local, with: nil) else {
                continue
            }
            if ancestorsInteractive {
                lastHit = HitCache(point: point, time: now, view: view)
                return view
            }
            if entry.priority > fallbackPriority {
                fallback = view
                fallbackPriority = entry.priority
            }
        }

        if let fallback {
            ControllerDiagnostics.once("overlay-control-noninteractive",
                                       "control \(type(of: fallback)) under the touch sits in a view with user interaction off; taken anyway")
        }
        lastHit = HitCache(point: point, time: now, view: fallback)
        return fallback
    }

    static func text(_ point: CGPoint) -> String {
        String(format: "%.0f,%.0f", point.x, point.y)
    }

    static func text(_ size: CGSize) -> String {
        String(format: "%.0fx%.0f", size.width, size.height)
    }
}

/// Logs the first touch the game window sees: where it landed, which view UIKit hit-tests there
/// and whether a control is under it. Never recognizes, so it takes nothing from anyone.
final class TouchProbe: UIGestureRecognizer {
    private weak var container: PassthroughView?

    init(container: PassthroughView) {
        self.container = container
        super.init(target: nil, action: nil)
        cancelsTouchesInView = false
        delaysTouchesBegan = false
        delaysTouchesEnded = false
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        if let touch = touches.first, let window = view as? UIWindow {
            let point = touch.location(in: window)
            ControllerDiagnostics.once("window-touch", describe(point, in: window, event: event))
        }
        state = .failed
    }

    private func describe(_ point: CGPoint, in window: UIWindow, event: UIEvent) -> String {
        var chain: [String] = []
        var view = window.hitTest(point, with: event)
        while let current = view, chain.count < 5 {
            chain.append("\(type(of: current))")
            view = current.superview
        }
        let root = window.rootViewController?.view
        var text = "first touch in the game window at \(PassthroughView.text(point)), window \(PassthroughView.text(window.bounds.size))"
        text += ", root \(root.map { "\(type(of: $0)) \(PassthroughView.text($0.frame.size))" } ?? "none")"
        text += ", hit \(chain.isEmpty ? "nothing" : chain.joined(separator: " < "))"
        if let container {
            let local = container.convert(point, from: window)
            text += ", overlay \(container.window === window ? "attached" : "detached") \(PassthroughView.text(container.bounds.size))"
            text += " interactive \(container.interactive), control \(container.control(at: local).map { "\(type(of: $0))" } ?? "none")"
        } else {
            text += ", no overlay"
        }
        return text
    }
}

/// Keeps the window's root view (SDL's Metal view, which also carries the overlay) at the window's
/// bounds. SDL sets that view's frame itself when it shows a fullscreen window, from the window it
/// sees then, swapped to landscape for a landscape-only window. The game window is shown hidden
/// while the game loads (the library, in portrait, keeps the scene), so it got a 874 x 402 frame in
/// a 402 x 874 window, and the rotation's autoresizing then made it 1346 x -70: a black picture,
/// and the controls and menu off screen. Sits behind the root view and re-fits it whenever the
/// window's size changes.
final class WindowBoundsFollower: UIView {
    static func install(in window: UIWindow) {
        if let follower = window.subviews.first(where: { $0 is WindowBoundsFollower }) {
            follower.frame = window.bounds
        } else {
            let follower = WindowBoundsFollower(frame: window.bounds)
            follower.isUserInteractionEnabled = false
            follower.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            window.insertSubview(follower, at: 0)
        }
        fit(window)
    }

    static func fit(_ window: UIWindow) {
        if let root = window.rootViewController?.view, root.frame != window.bounds {
            root.frame = window.bounds
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        guard let window else { return }
        if frame != window.bounds {
            frame = window.bounds
        }
        Self.fit(window)
    }
}
