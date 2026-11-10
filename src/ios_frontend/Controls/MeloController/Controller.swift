// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Controller.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//

import SwiftUI
import UIKit

/// Lichii: the UIKit views that take the on-screen controller's touches (buttons and sticks). The
/// host hands touches to them itself instead of relying on SwiftUI's hit testing.
@MainActor
public protocol ControllerTouchTarget: UIView {
    /// A button wins over a stick under it.
    var touchPriority: Int { get }
}

/// Lichii: one log line per kind of event and game, so a device log shows how far a touch got.
public enum ControllerDiagnostics {
    public static var sink: ((String) -> Void)?
    private static var logged: Set<String> = []

    public static func once(_ key: String, _ message: @autoclosure () -> String) {
        guard let sink, logged.insert(key).inserted else { return }
        sink(message())
    }

    public static func reset() {
        logged.removeAll()
    }
}

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
