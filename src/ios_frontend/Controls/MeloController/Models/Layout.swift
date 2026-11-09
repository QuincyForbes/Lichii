// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Layout.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//

import Foundation

public struct ButtonLayout: Codable, Equatable {
    var offset: CGSize = .zero
    var scale: CGFloat = 1.0
    var hidden: Bool = false
    var toggle: Bool = false
    var dpadToJoystickl: Bool = false
}

public struct JoystickLayout: Codable, Equatable {
    var offset: CGSize = .zero
    var scale: CGFloat = 1.0
    var hide: Bool = true
    var background: Bool = false
    var hidden: Bool = false
}

public struct LayoutConfig: Codable, Equatable {
    var buttons: [String: ButtonLayout] = [:]
    var joysticks: [String: JoystickLayout] = [:]

    public init() {}
}
