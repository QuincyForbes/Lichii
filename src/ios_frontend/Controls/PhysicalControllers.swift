// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import CoreMotion
import Foundation
import GameController
import LichiiBridge
import UIKit

/// Saved per controller model: mapping, sticks, triggers and the player it drives.
struct PadProfile: Codable, Equatable {
    var mapping: [Int32]
    var inner: [Float] = [0.10, 0.10]
    var outer: [Float] = [0.95, 0.95]
    var invertY: [Bool] = [false, false]
    var triggerThreshold: Float = 0.30
    var swapSticks = false
    var player: Int32?
    var rumble = true
    var motion = true
}

/// One connected GameController device and its bridge slot.
final class PhysicalPad: Identifiable, ObservableObject {
    let id = UUID()
    let controller: GCController
    let slot: Int32
    let kind: Int32
    var rumble: RumbleController?
    fileprivate var lastMotion: TimeInterval = 0

    var name: String { controller.vendorName ?? controller.productCategory }
    var profileKey: String { "\(controller.productCategory)|\(controller.vendorName ?? "")" }
    var hasMotion: Bool { controller.motion != nil }
    var hasHaptics: Bool { controller.haptics != nil }

    init(controller: GCController, slot: Int32, kind: Int32) {
        self.controller = controller
        self.slot = slot
        self.kind = kind
    }

    var player: Int32 {
        get { lichii_pad_get_player(slot) }
        set {
            lichii_pad_set_player(slot, newValue)
            objectWillChange.send()
        }
    }
}

/// Every GameController device (MFi, Xbox, DualSense, DualShock, Switch Pro, Joy-Con pairs),
/// routed through the controls bridge with a remappable layout per controller model, its motion
/// sensors, and the game's rumble on its haptic engines. The phone's own motion and Taptic Engine
/// stand in for player 1 while the on-screen controller is used.
@MainActor
final class PhysicalControllers: ObservableObject {
    static let shared = PhysicalControllers()

    @Published private(set) var pads: [PhysicalPad] = []

    private let inputQueue = DispatchQueue(label: "lichii.controllers", qos: .userInteractive)
    private let motionManager = CMMotionManager()
    private let motionOperations = OperationQueue()
    private var deviceRumble: RumbleController?
    private var deviceMotionTime: TimeInterval = 0
    private var started = false

    private static let profilesKey = "controllerProfiles"

    /// Rumble on the phone while the on-screen controller is in use.
    var deviceRumbleEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "deviceRumble") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "deviceRumble"); objectWillChange.send() }
    }

    /// The phone's gyroscope drives player 1 when no controller with motion sensors does.
    var deviceMotionEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "deviceMotion") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "deviceMotion"); objectWillChange.send() }
    }

    func start() {
        guard !started else { return }
        started = true
        lichii_pad_set_menu_chord(UserDefaults.standard.object(forKey: "menuChord") as? Bool ?? true)
        NotificationCenter.default.addObserver(forName: .GCControllerDidConnect, object: nil,
                                               queue: .main) { note in
            guard let controller = note.object as? GCController else { return }
            MainActor.assumeIsolated { PhysicalControllers.shared.connect(controller) }
        }
        NotificationCenter.default.addObserver(forName: .GCControllerDidDisconnect, object: nil,
                                               queue: .main) { note in
            guard let controller = note.object as? GCController else { return }
            MainActor.assumeIsolated { PhysicalControllers.shared.disconnect(controller) }
        }
        GCController.controllers().forEach(connect)
        GCController.startWirelessControllerDiscovery {}
    }

    /// Player 1 is driven by a controller (the on-screen controller can hide).
    var playerOneHasController: Bool {
        pads.contains { $0.player == 0 }
    }

    // MARK: - Connection

    private func connect(_ controller: GCController) {
        guard !pads.contains(where: { $0.controller === controller }),
              controller.extendedGamepad != nil else { return }
        let kind = Self.kind(of: controller)
        let slot = lichii_pad_connect(kind)
        guard slot >= 0 else { return }
        let pad = PhysicalPad(controller: controller, slot: slot, kind: kind)
        if let profile = profile(for: pad) {
            apply(profile, to: pad)
        }
        controller.playerIndex = GCControllerPlayerIndex(rawValue: Int(pad.player)) ?? .indexUnset
        controller.handlerQueue = inputQueue
        bind(pad)
        setUpMotion(pad, enabled: profile(for: pad)?.motion ?? true)
        if profile(for: pad)?.rumble ?? true {
            pad.rumble = RumbleController(controller: controller)
        }
        pads.append(pad)
        lichii_post_event(Int32(LICHII_EVENT_CONTROLLERS_CHANGED.rawValue))
        updateDeviceMotion()
    }

    private func disconnect(_ controller: GCController) {
        guard let index = pads.firstIndex(where: { $0.controller === controller }) else { return }
        let pad = pads.remove(at: index)
        pad.rumble?.stop()
        pad.rumble = nil
        controller.motion?.valueChangedHandler = nil
        lichii_pad_disconnect(pad.slot)
        lichii_post_event(Int32(LICHII_EVENT_CONTROLLERS_CHANGED.rawValue))
        updateDeviceMotion()
    }

    private static func kind(of controller: GCController) -> Int32 {
        let category = controller.productCategory
        if controller.extendedGamepad is GCDualSenseGamepad { return Int32(LICHII_PAD_KIND_DUALSENSE.rawValue) }
        if controller.extendedGamepad is GCDualShockGamepad { return Int32(LICHII_PAD_KIND_DUALSHOCK.rawValue) }
        if controller.extendedGamepad is GCXboxGamepad { return Int32(LICHII_PAD_KIND_XBOX.rawValue) }
        if category.contains("Joy-Con") { return Int32(LICHII_PAD_KIND_JOYCON_PAIR.rawValue) }
        if category.contains("Switch") || category.contains("Pro Controller") {
            return Int32(LICHII_PAD_KIND_SWITCH_PRO.rawValue)
        }
        if category == GCProductCategoryMFi { return Int32(LICHII_PAD_KIND_MFI.rawValue) }
        return Int32(LICHII_PAD_KIND_GENERIC.rawValue)
    }

    // MARK: - Input

    private nonisolated func bind(_ pad: PhysicalPad) {
        guard let gamepad = pad.controller.extendedGamepad else { return }
        let slot = pad.slot
        func button(_ input: GCControllerButtonInput?, _ element: LichiiPadElement) {
            guard let input else { return }
            input.preferredSystemGestureState = .disabled
            input.pressedChangedHandler = { _, _, pressed in
                lichii_pad_element(slot, Int32(element.rawValue), pressed)
            }
        }
        func stick(_ pad: GCControllerDirectionPad, _ x: LichiiPadAxis, _ y: LichiiPadAxis) {
            pad.valueChangedHandler = { _, xValue, yValue in
                lichii_pad_axis(slot, Int32(x.rawValue), xValue)
                lichii_pad_axis(slot, Int32(y.rawValue), yValue)
            }
        }
        func trigger(_ input: GCControllerButtonInput, _ axis: LichiiPadAxis) {
            input.valueChangedHandler = { _, value, _ in
                lichii_pad_axis(slot, Int32(axis.rawValue), value)
            }
        }

        button(gamepad.buttonA, LICHII_PAD_SOUTH)
        button(gamepad.buttonB, LICHII_PAD_EAST)
        button(gamepad.buttonX, LICHII_PAD_WEST)
        button(gamepad.buttonY, LICHII_PAD_NORTH)
        button(gamepad.leftShoulder, LICHII_PAD_L1)
        button(gamepad.rightShoulder, LICHII_PAD_R1)
        button(gamepad.leftThumbstickButton, LICHII_PAD_L3)
        button(gamepad.rightThumbstickButton, LICHII_PAD_R3)
        button(gamepad.buttonMenu, LICHII_PAD_MENU)
        button(gamepad.buttonOptions, LICHII_PAD_OPTIONS)
        button(gamepad.buttonHome, LICHII_PAD_HOME)
        button(gamepad.dpad.up, LICHII_PAD_DPAD_UP)
        button(gamepad.dpad.down, LICHII_PAD_DPAD_DOWN)
        button(gamepad.dpad.left, LICHII_PAD_DPAD_LEFT)
        button(gamepad.dpad.right, LICHII_PAD_DPAD_RIGHT)
        stick(gamepad.leftThumbstick, LICHII_PAD_AXIS_LX, LICHII_PAD_AXIS_LY)
        stick(gamepad.rightThumbstick, LICHII_PAD_AXIS_RX, LICHII_PAD_AXIS_RY)
        trigger(gamepad.leftTrigger, LICHII_PAD_AXIS_L2)
        trigger(gamepad.rightTrigger, LICHII_PAD_AXIS_R2)

        if let xbox = gamepad as? GCXboxGamepad {
            button(xbox.paddleButton1, LICHII_PAD_PADDLE1)
            button(xbox.paddleButton2, LICHII_PAD_PADDLE2)
            button(xbox.paddleButton3, LICHII_PAD_PADDLE3)
            button(xbox.paddleButton4, LICHII_PAD_PADDLE4)
            button(xbox.buttonShare, LICHII_PAD_SHARE)
        }
        if let dualSense = gamepad as? GCDualSenseGamepad {
            button(dualSense.touchpadButton, LICHII_PAD_TOUCHPAD)
        }
        if let dualShock = gamepad as? GCDualShockGamepad {
            button(dualShock.touchpadButton, LICHII_PAD_TOUCHPAD)
        }
        if let share = pad.controller.physicalInputProfile.buttons[GCInputButtonShare], pad.kind != Int32(LICHII_PAD_KIND_XBOX.rawValue) {
            button(share, LICHII_PAD_SHARE)
        }
    }

    // MARK: - Motion

    private nonisolated func setUpMotion(_ pad: PhysicalPad, enabled: Bool) {
        guard let motion = pad.controller.motion else { return }
        guard enabled else {
            motion.valueChangedHandler = nil
            return
        }
        if motion.sensorsRequireManualActivation {
            motion.sensorsActive = true
        }
        let slot = pad.slot
        motion.valueChangedHandler = { [weak pad] m in
            guard let pad else { return }
            let now = ProcessInfo.processInfo.systemUptime
            let delta = pad.lastMotion == 0 ? 1.0 / 60.0 : now - pad.lastMotion
            pad.lastMotion = now
            // Controller axes -> Switch axes (MeloNX's mapping); gyro in turns per second.
            let turns = 1.0 / (2.0 * Double.pi)
            lichii_pad_motion(slot, UInt64(max(delta, 0) * 1_000_000),
                              Float(m.rotationRate.x * turns), Float(m.rotationRate.z * turns),
                              Float(-m.rotationRate.y * turns),
                              Float(-m.acceleration.x), Float(-m.acceleration.z), Float(m.acceleration.y))
        }
    }

    /// Device motion for player 1 while no controller with sensors drives it.
    func updateDeviceMotion() {
        let wanted = deviceMotionEnabled && ControlsCenter.shared.gameRunning &&
            !pads.contains { $0.player == 0 && $0.hasMotion }
        if wanted == motionManager.isDeviceMotionActive { return }
        guard wanted else {
            motionManager.stopDeviceMotionUpdates()
            return
        }
        guard motionManager.isDeviceMotionAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        deviceMotionTime = 0
        motionManager.startDeviceMotionUpdates(to: motionOperations, withHandler: deviceMotionHandler)
    }

    fileprivate nonisolated static func submitDeviceMotion(_ m: CMDeviceMotion) {
        let accel = SIMD3<Double>(-(m.gravity.x + m.userAcceleration.x),
                                  -(m.gravity.y + m.userAcceleration.y),
                                  -(m.gravity.z + m.userAcceleration.z))
        let gyro = SIMD3<Double>(m.rotationRate.x, -m.rotationRate.y, -m.rotationRate.z)
        let (a, g) = remapToSwitch(accel: accel, gyro: gyro, orientation: DeviceOrientation.current)
        let turns = 1.0 / (2.0 * Double.pi)
        lichii_controls_motion(0, UInt64(1_000_000 / 60),
                               Float(g.x * turns), Float(g.y * turns), Float(g.z * turns),
                               Float(a.x), Float(a.y), Float(a.z))
    }

    /// Phone axes -> Switch axes for the way the phone is held (MeloNX's mapping).
    private nonisolated static func remapToSwitch(accel: SIMD3<Double>, gyro: SIMD3<Double>,
                                                  orientation: UIDeviceOrientation) -> (SIMD3<Double>, SIMD3<Double>) {
        switch orientation {
        case .landscapeLeft:
            return (SIMD3(accel.y, -accel.x, accel.z), SIMD3(gyro.y, -gyro.x, gyro.z))
        case .landscapeRight:
            return (SIMD3(-accel.y, accel.x, accel.z), SIMD3(-gyro.y, gyro.x, gyro.z))
        case .portraitUpsideDown:
            return (SIMD3(-accel.x, accel.z, accel.y), SIMD3(-gyro.x, gyro.z, gyro.y))
        default:
            return (SIMD3(accel.x, accel.z, -accel.y), SIMD3(gyro.x, gyro.z, -gyro.y))
        }
    }

    // MARK: - Rumble

    /// Polled by ControlsCenter while a game runs.
    func pollRumble() {
        var value = LichiiRumble()
        for pad in pads {
            guard let rumble = pad.rumble else { continue }
            if pad.player >= 0, lichii_controls_rumble(pad.player, &value) {
                rumble.update(value)
            } else {
                rumble.stop()
            }
        }
        if deviceRumbleEnabled, !playerOneHasController {
            if deviceRumble == nil {
                deviceRumble = RumbleController(deviceMultiplier: 2.0)
            }
            if lichii_controls_rumble(0, &value) {
                deviceRumble?.update(value)
            } else {
                deviceRumble?.stop()
            }
        } else if let deviceRumble {
            deviceRumble.stop()
            self.deviceRumble = nil
        }
    }

    func stopRumble() {
        pads.forEach { $0.rumble?.stop() }
        deviceRumble?.stop()
    }

    // MARK: - Profiles

    private func loadProfiles() -> [String: PadProfile] {
        guard let data = UserDefaults.standard.data(forKey: Self.profilesKey),
              let profiles = try? JSONDecoder().decode([String: PadProfile].self, from: data) else { return [:] }
        return profiles
    }

    func profile(for pad: PhysicalPad) -> PadProfile? {
        loadProfiles()[pad.profileKey]
    }

    /// Reads the slot's current configuration back from the bridge.
    func currentProfile(for pad: PhysicalPad) -> PadProfile {
        var profile = PadProfile(mapping: (0..<Int32(LICHII_PAD_ELEMENT_COUNT.rawValue)).map {
            lichii_pad_get_mapping(pad.slot, $0)
        })
        for stick in 0..<2 {
            var inner: Float = 0
            var outer: Float = 0
            lichii_pad_get_deadzone(pad.slot, Int32(stick), &inner, &outer)
            profile.inner[stick] = inner
            profile.outer[stick] = outer
            profile.invertY[stick] = lichii_pad_get_invert_y(pad.slot, Int32(stick))
        }
        profile.triggerThreshold = lichii_pad_get_trigger_threshold(pad.slot)
        profile.swapSticks = lichii_pad_get_swap_sticks(pad.slot)
        profile.player = pad.player
        let saved = self.profile(for: pad)
        profile.rumble = saved?.rumble ?? true
        profile.motion = saved?.motion ?? true
        return profile
    }

    func save(_ profile: PadProfile, for pad: PhysicalPad) {
        apply(profile, to: pad)
        var profiles = loadProfiles()
        profiles[pad.profileKey] = profile
        if let data = try? JSONEncoder().encode(profiles) {
            UserDefaults.standard.set(data, forKey: Self.profilesKey)
        }
        if profile.rumble, pad.rumble == nil {
            pad.rumble = RumbleController(controller: pad.controller)
        } else if !profile.rumble {
            pad.rumble?.stop()
            pad.rumble = nil
        }
        setUpMotion(pad, enabled: profile.motion)
        pad.controller.playerIndex = GCControllerPlayerIndex(rawValue: Int(pad.player)) ?? .indexUnset
        pad.objectWillChange.send()
        objectWillChange.send()
        updateDeviceMotion()
    }

    func forget(_ pad: PhysicalPad) {
        var profiles = loadProfiles()
        profiles[pad.profileKey] = nil
        if let data = try? JSONEncoder().encode(profiles) {
            UserDefaults.standard.set(data, forKey: Self.profilesKey)
        }
        lichii_pad_reset_mapping(pad.slot, false)
        for stick in 0..<2 {
            lichii_pad_set_deadzone(pad.slot, Int32(stick), 0.10, 0.95)
            lichii_pad_set_invert_y(pad.slot, Int32(stick), false)
        }
        lichii_pad_set_trigger_threshold(pad.slot, 0.30)
        lichii_pad_set_swap_sticks(pad.slot, false)
        pad.objectWillChange.send()
        objectWillChange.send()
    }

    private func apply(_ profile: PadProfile, to pad: PhysicalPad) {
        for (element, target) in profile.mapping.enumerated() where element < Int(LICHII_PAD_ELEMENT_COUNT.rawValue) {
            lichii_pad_set_mapping(pad.slot, Int32(element), target)
        }
        for stick in 0..<2 {
            lichii_pad_set_deadzone(pad.slot, Int32(stick), profile.inner[stick], profile.outer[stick])
            lichii_pad_set_invert_y(pad.slot, Int32(stick), profile.invertY[stick])
        }
        lichii_pad_set_trigger_threshold(pad.slot, profile.triggerThreshold)
        lichii_pad_set_swap_sticks(pad.slot, profile.swapSticks)
        if let player = profile.player {
            lichii_pad_set_player(pad.slot, player)
        }
    }
}

/// CoreMotion calls this on its operation queue.
private let deviceMotionHandler: CMDeviceMotionHandler = { data, _ in
    if let data {
        PhysicalControllers.submitDeviceMotion(data)
    }
}

/// The device orientation, readable from any thread (CoreMotion delivers on its own queue).
enum DeviceOrientation {
    private static let lock = NSLock()
    private static var value: UIDeviceOrientation = .landscapeLeft

    static var current: UIDeviceOrientation {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    @MainActor
    static func update() {
        let orientation = UIDevice.current.orientation
        guard orientation.isValidInterfaceOrientation else { return }
        lock.lock()
        value = orientation
        lock.unlock()
    }
}
