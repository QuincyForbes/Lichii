// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Controller.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//

import SwiftUI

public protocol Controller: ObservableObject {
    func buttonPressed(_ button: VirtualControllerButton)
    func buttonReleased(_ button: VirtualControllerButton)

    func joystickMoved(position: CGPoint, right: Bool)
}

public extension Notification.Name {
    /// Lichii: the app released every on-screen input (menu opened, app inactive, controls
    /// hidden); toggled buttons show released again.
    static let lichiiTouchReleased = Notification.Name("LichiiTouchReleased")
}
