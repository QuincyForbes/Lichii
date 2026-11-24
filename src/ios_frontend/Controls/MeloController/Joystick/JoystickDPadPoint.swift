// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  JoystickDPadPoint.swift
//  Melo-Controller
//
//  Created by Stossy11 on 26/1/2026.
//

import SwiftUI
import Foundation

public class JoystickDPadPoint: ObservableObject {
    public static var shared = JoystickDPadPoint()
    @Published var joystickDpadPoint: (x: Double, y: Double) = (0, 0)
    @Published var currentlyPressed: Set<VirtualControllerButton> = []
    @Published var isJoystickActive: Bool = false


    func pressed(_ button: VirtualControllerButton, controller: any Controller) {
        guard button.isDPad else { return }
        currentlyPressed.insert(button)
        recomputeDPad(controller: controller)
    }

    func released(_ button: VirtualControllerButton, controller: any Controller) {
        guard button.isDPad else { return }
        currentlyPressed.remove(button)
        recomputeDPad(controller: controller)
    }

    public func reset() {
        currentlyPressed.removeAll()
        joystickDpadPoint = (0, 0)
        isJoystickActive = false
    }

    private func recomputeDPad(controller: any Controller) {
        let up = currentlyPressed.contains(.dPadUp)
        let down = currentlyPressed.contains(.dPadDown)
        let left = currentlyPressed.contains(.dPadLeft)
        let right = currentlyPressed.contains(.dPadRight)

        var y: Double = (up == down) ? 0 : (up ? -1 : 1)
        var x: Double = (left == right) ? 0 : (left ? -1 : 1)

        if x != 0 && y != 0 {
            let invSqrt2 = 1.0 / 2.0.squareRoot()
            x *= invSqrt2
            y *= invSqrt2
        }

        joystickDpadPoint = (x, y)
        controller.joystickMoved(position: CGPoint(x: x, y: y), right: false)
    }
}
