// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import CoreHaptics
import Foundation
import GameController
import LichiiBridge

/// Plays the game's rumble on a controller's haptic engines (one per handle when it has two) or
/// on the device: a low and a high continuous player per engine whose intensity and sharpness
/// follow the HD rumble amplitudes and frequencies.
final class RumbleController {
    private final class HapticChannel {
        private let engine: CHHapticEngine
        private let eventDuration: TimeInterval
        private var lowPlayer: CHHapticPatternPlayer?
        private var highPlayer: CHHapticPatternPlayer?

        init?(engine: CHHapticEngine?, eventDuration: TimeInterval) {
            guard let engine else { return nil }
            self.engine = engine
            self.eventDuration = eventDuration
            do {
                try engine.start()
                try createPlayers()
            } catch {
                return nil
            }
        }

        func restart() {
            do {
                try lowPlayer?.stop(atTime: CHHapticTimeImmediate)
                try highPlayer?.stop(atTime: CHHapticTimeImmediate)
                try engine.start()
                try createPlayers()
            } catch {
                lowPlayer = nil
                highPlayer = nil
            }
        }

        func stop() {
            try? lowPlayer?.stop(atTime: CHHapticTimeImmediate)
            try? highPlayer?.stop(atTime: CHHapticTimeImmediate)
            engine.stop()
        }

        func update(low: Float, lowHz: Float, high: Float, highHz: Float, multiplier: Float) {
            do {
                try send(low: low, lowHz: lowHz, high: high, highHz: highHz, multiplier: multiplier)
            } catch {
                restart()
            }
        }

        private func send(low: Float, lowHz: Float, high: Float, highHz: Float, multiplier: Float) throws {
            try lowPlayer?.sendParameters([
                CHHapticDynamicParameter(parameterID: .hapticIntensityControl,
                                         value: Self.clamp(low * multiplier), relativeTime: 0),
                CHHapticDynamicParameter(parameterID: .hapticSharpnessControl,
                                         value: Self.sharpness(for: lowHz, fallback: 0), relativeTime: 0),
            ], atTime: 0)
            try highPlayer?.sendParameters([
                CHHapticDynamicParameter(parameterID: .hapticIntensityControl,
                                         value: Self.clamp(high * multiplier), relativeTime: 0),
                CHHapticDynamicParameter(parameterID: .hapticSharpnessControl,
                                         value: Self.sharpness(for: highHz, fallback: 1), relativeTime: 0),
            ], atTime: 0)
        }

        private func createPlayers() throws {
            lowPlayer = try createPlayer(sharpness: 0)
            highPlayer = try createPlayer(sharpness: 1)
            try send(low: 0, lowHz: 160, high: 0, highHz: 320, multiplier: 1)
            try lowPlayer?.start(atTime: 0)
            try highPlayer?.start(atTime: 0)
        }

        private func createPlayer(sharpness: Float) throws -> CHHapticPatternPlayer {
            let event = CHHapticEvent(
                eventType: .hapticContinuous,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: 1),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness),
                ],
                relativeTime: 0,
                duration: eventDuration)
            return try engine.makePlayer(with: try CHHapticPattern(events: [event], parameters: []))
        }

        /// 40..1252 Hz (the HD rumble range) on a log scale -> sharpness 0..1.
        private static func sharpness(for frequency: Float, fallback: Float) -> Float {
            guard frequency.isFinite, frequency > 0 else { return fallback }
            let minHz: Float = 40
            let maxHz: Float = 1252
            let normalized = (log2(clamp(frequency, minHz, maxHz)) - log2(minHz)) / (log2(maxHz) - log2(minHz))
            return clamp(normalized)
        }

        private static func clamp(_ value: Float, _ lower: Float = 0, _ upper: Float = 1) -> Float {
            min(upper, max(lower, value))
        }
    }

    private var leftChannel: HapticChannel?
    private var rightChannel: HapticChannel?
    private var fallbackChannel: HapticChannel?
    private let multiplier: Float
    private let queue = DispatchQueue(label: "lichii.rumble", qos: .userInteractive)
    // Continuous events end after this long; the players restart shortly before.
    private let eventDuration: TimeInterval = 20
    private var restartTimer: Timer?
    private var last = LichiiRumble()

    /// The controller's engines (both handles when it has them), or nil when it has no haptics.
    init?(controller: GCController, multiplier: Float = 2.5) {
        guard let haptics = controller.haptics else { return nil }
        self.multiplier = multiplier
        let localities = haptics.supportedLocalities
        if localities.contains(.leftHandle), localities.contains(.rightHandle) {
            leftChannel = HapticChannel(engine: haptics.createEngine(withLocality: .leftHandle),
                                        eventDuration: eventDuration)
            rightChannel = HapticChannel(engine: haptics.createEngine(withLocality: .rightHandle),
                                         eventDuration: eventDuration)
        }
        if leftChannel == nil || rightChannel == nil {
            leftChannel = nil
            rightChannel = nil
            for locality in [GCHapticsLocality.handles, .default, .all] where localities.contains(locality) {
                fallbackChannel = HapticChannel(engine: haptics.createEngine(withLocality: locality),
                                                eventDuration: eventDuration)
                if fallbackChannel != nil { break }
            }
        }
        guard leftChannel != nil || fallbackChannel != nil else { return nil }
        scheduleRestarts()
    }

    /// The device's own Taptic Engine.
    init?(deviceMultiplier: Float = 2.0) {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics,
              let engine = try? CHHapticEngine() else { return nil }
        multiplier = deviceMultiplier
        fallbackChannel = HapticChannel(engine: engine, eventDuration: eventDuration)
        guard fallbackChannel != nil else { return nil }
        scheduleRestarts()
    }

    deinit {
        restartTimer?.invalidate()
        leftChannel?.stop()
        rightChannel?.stop()
        fallbackChannel?.stop()
    }

    private func scheduleRestarts() {
        restartTimer = Timer.scheduledTimer(withTimeInterval: eventDuration - 1, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.queue.async {
                self.leftChannel?.restart()
                self.rightChannel?.restart()
                self.fallbackChannel?.restart()
            }
        }
    }

    func stop() {
        update(LichiiRumble())
    }

    func update(_ value: LichiiRumble) {
        guard !Self.same(value, last) else { return }
        last = value
        queue.async { [self] in
            if let leftChannel, let rightChannel {
                leftChannel.update(low: value.left_low, lowHz: value.left_low_hz,
                                   high: value.left_high, highHz: value.left_high_hz, multiplier: multiplier)
                rightChannel.update(low: value.right_low, lowHz: value.right_low_hz,
                                    high: value.right_high, highHz: value.right_high_hz, multiplier: multiplier)
            } else {
                fallbackChannel?.update(
                    low: max(value.left_low, value.right_low),
                    lowHz: Self.weighted(value.left_low_hz, value.left_low, value.right_low_hz, value.right_low, 160),
                    high: max(value.left_high, value.right_high),
                    highHz: Self.weighted(value.left_high_hz, value.left_high, value.right_high_hz, value.right_high, 320),
                    multiplier: multiplier)
            }
        }
    }

    private static func same(_ a: LichiiRumble, _ b: LichiiRumble) -> Bool {
        a.left_low == b.left_low && a.left_high == b.left_high && a.right_low == b.right_low &&
            a.right_high == b.right_high && a.left_low_hz == b.left_low_hz &&
            a.left_high_hz == b.left_high_hz && a.right_low_hz == b.right_low_hz &&
            a.right_high_hz == b.right_high_hz
    }

    private static func weighted(_ f1: Float, _ a1: Float, _ f2: Float, _ a2: Float, _ fallback: Float) -> Float {
        let total = a1 + a2
        guard total > 0 else { return fallback }
        return (f1 * a1 + f2 * a2) / total
    }
}
