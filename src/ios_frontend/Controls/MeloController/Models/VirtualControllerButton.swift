// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  VirtualControllerButton.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//

import Foundation

public struct VirtualContButtonConfig: Hashable, Codable, Identifiable {
    public var id = UUID.init()

    public var small: Bool = false
    public var trigger: Bool = false
}

public struct VirtualControllerButton: Hashable, Codable, Identifiable {
    static var registered: Set<VirtualControllerButton> = []

    public let id: String
    public var iconName: String
    public var small: Bool
    public var trigger: Bool

    public init(_ id: String, systemName: String, small: Bool? = nil, trigger: Bool? = nil) {
        self.id = id
        self.iconName = systemName
        self.small = small ?? false
        self.trigger = trigger ?? false

        Self.registered.insert(self)
    }
}

public extension VirtualControllerButton {
    var isTrigger: Bool {
        Self.registered.contains(where: { $0.id == id && $0.trigger })
    }

    var isDPad: Bool {
        ["dPadUp", "dPadDown", "dPadLeft", "dPadRight"].contains(id)
    }

    var isSmall: Bool {
        Self.registered.contains(where: { $0.id == id && $0.small })
    }
}

public extension VirtualControllerButton {
    static let A = Self("A", systemName: "a.circle.fill")
    static let B = Self("B", systemName: "b.circle.fill")
    static let X = Self("X", systemName: "x.circle.fill")
    static let Y = Self("Y", systemName: "y.circle.fill")
    static let back = Self("back", systemName: "minus.circle.fill", small: true)
    static let guide = Self("guide", systemName: "gearshape.fill", small: true)
    static let start = Self("start", systemName: "plus.circle.fill", small: true)
    static let leftStick = Self("leftStick", systemName: "l.joystick.press.down.fill")
    static let rightStick = Self("rightStick", systemName: "r.joystick.press.down.fill")
    static let leftShoulder = Self("leftShoulder", systemName: "l.rectangle.roundedbottom.fill", trigger: true)
    static let rightShoulder = Self("rightShoulder", systemName: "r.rectangle.roundedbottom.fill", trigger: true)
    static let dPadUp = Self("dPadUp", systemName: "arrowtriangle.up.circle.fill")
    static let dPadDown = Self("dPadDown", systemName: "arrowtriangle.down.circle.fill")
    static let dPadLeft = Self("dPadLeft", systemName: "arrowtriangle.left.circle.fill")
    static let dPadRight = Self("dPadRight", systemName: "arrowtriangle.right.circle.fill")
    static let leftTrigger = Self("leftTrigger", systemName: "zl.rectangle.roundedtop.fill", trigger: true)
    static let rightTrigger = Self("rightTrigger", systemName: "zr.rectangle.roundedtop.fill", trigger: true)
    // Lichii
    static let home = Self("home", systemName: "house.circle.fill", small: true)
    static let capture = Self("capture", systemName: "camera.circle.fill", small: true)
}
