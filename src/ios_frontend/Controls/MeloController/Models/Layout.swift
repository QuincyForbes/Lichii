// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Layout.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//
//  Lichii: per-control opacity, and decoding that tolerates missing keys (Melo-Controller
//  layouts without opacity import unchanged).

import Foundation

public struct ButtonLayout: Codable, Equatable {
    var offset: CGSize = .zero
    var scale: CGFloat = 1.0
    var hidden: Bool = false
    var toggle: Bool = false
    var dpadToJoystickl: Bool = false
    var opacity: Double = 1.0

    init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        offset = try c.decodeIfPresent(CGSize.self, forKey: .offset) ?? .zero
        scale = try c.decodeIfPresent(CGFloat.self, forKey: .scale) ?? 1.0
        hidden = try c.decodeIfPresent(Bool.self, forKey: .hidden) ?? false
        toggle = try c.decodeIfPresent(Bool.self, forKey: .toggle) ?? false
        dpadToJoystickl = try c.decodeIfPresent(Bool.self, forKey: .dpadToJoystickl) ?? false
        opacity = try c.decodeIfPresent(Double.self, forKey: .opacity) ?? 1.0
    }
}

public struct JoystickLayout: Codable, Equatable {
    var offset: CGSize = .zero
    var scale: CGFloat = 1.0
    /// Lichii: off by default; fading the face buttons while the stick is held made a tap that
    /// slipped between them hide the very button it meant.
    var hide: Bool = false
    var background: Bool = false
    var hidden: Bool = false
    var opacity: Double = 1.0
    /// Lichii: centre where the thumb lands; nil follows the stick's default (left on, right off).
    var floating: Bool?

    init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        offset = try c.decodeIfPresent(CGSize.self, forKey: .offset) ?? .zero
        scale = try c.decodeIfPresent(CGFloat.self, forKey: .scale) ?? 1.0
        hide = try c.decodeIfPresent(Bool.self, forKey: .hide) ?? false
        background = try c.decodeIfPresent(Bool.self, forKey: .background) ?? false
        hidden = try c.decodeIfPresent(Bool.self, forKey: .hidden) ?? false
        opacity = try c.decodeIfPresent(Double.self, forKey: .opacity) ?? 1.0
        floating = try c.decodeIfPresent(Bool.self, forKey: .floating)
    }

    static func defaultFloating(right: Bool) -> Bool {
        !right
    }
}

public struct LayoutConfig: Codable, Equatable {
    var buttons: [String: ButtonLayout] = [:]
    var joysticks: [String: JoystickLayout] = [:]

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        buttons = try c.decodeIfPresent([String: ButtonLayout].self, forKey: .buttons) ?? [:]
        joysticks = try c.decodeIfPresent([String: JoystickLayout].self, forKey: .joysticks) ?? [:]
    }
}
