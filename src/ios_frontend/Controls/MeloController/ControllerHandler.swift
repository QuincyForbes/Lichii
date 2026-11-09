// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  ControllerHandler.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//

import SwiftUI

@MainActor
class ControllerHandler: ObservableObject {

    @Published var isPortrait: Bool = false

    func updateOrientation() {
        guard let window = Window.window else { return }
        self.isPortrait = window.bounds.size.height > window.bounds.size.width
    }
}
