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

class JoystickDPadPoint: ObservableObject {
    static var shared = JoystickDPadPoint()
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

    private func recomputeDPad(controller: any Controller) {
        // Y axis
        if currentlyPressed.contains(.dPadUp) {
            joystickDpadPoint.y = -1
        } else if currentlyPressed.contains(.dPadDown) {
            joystickDpadPoint.y = 1
        } else {
            joystickDpadPoint.y = 0
        }

        // X axis
        if currentlyPressed.contains(.dPadLeft) {
            joystickDpadPoint.x = -1
        } else if currentlyPressed.contains(.dPadRight) {
            joystickDpadPoint.x = 1
        } else {
            joystickDpadPoint.x = 0
        }

        controller.joystickMoved(position: CGPoint(x: joystickDpadPoint.x, y: joystickDpadPoint.y), right: false)
    }
}
