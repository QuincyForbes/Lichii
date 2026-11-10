// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import Foundation
import LichiiBridge
import Melo_Controller

/// Melo-Controller's on-screen controller -> the controls bridge (player 1, mirrored to Handheld).
final class TouchController: ObservableObject, Controller {
    static let shared = TouchController()

    var player: Int32 = 0

    /// Melo-Controller button ids -> LichiiButton values.
    private static let buttons: [String: Int32] = [
        "A": 0, "B": 1, "X": 2, "Y": 3,
        "leftStick": 4, "rightStick": 5,
        "leftShoulder": 6, "rightShoulder": 7, "leftTrigger": 8, "rightTrigger": 9,
        "start": 10, "back": 11,
        "dPadLeft": 12, "dPadUp": 13, "dPadRight": 14, "dPadDown": 15,
        "home": 18, "capture": 19,
    ]

    func buttonPressed(_ button: VirtualControllerButton) {
        if button.id == "guide" {
            Task { @MainActor in ControlsCenter.shared.openMenu() }
            return
        }
        guard let id = Self.buttons[button.id] else {
            ControllerDiagnostics.once("unmapped-\(button.id)", "on-screen button \(button.id) has no Switch button")
            return
        }
        ControllerDiagnostics.once("bridge-press", "first bridge call, \(button.id) -> button \(id) player \(player + 1)")
        lichii_controls_touch_button(player, id, true)
    }

    func buttonReleased(_ button: VirtualControllerButton) {
        guard let id = Self.buttons[button.id] else { return }
        lichii_controls_touch_button(player, id, false)
    }

    /// Melo-Controller reports y down; the bridge takes y up.
    func joystickMoved(position: CGPoint, right: Bool) {
        if position != .zero {
            ControllerDiagnostics.once("bridge-stick", "first bridge stick call, \(right ? "right" : "left") stick")
        }
        lichii_controls_touch_stick(player, right ? 1 : 0, Float(position.x), Float(-position.y))
    }
}
