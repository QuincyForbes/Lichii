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

    /// Lichii: the controller's own size once known. The key window can still be in portrait
    /// while the game window turns to landscape, which picked the portrait layout.
    private var size: CGSize = .zero

    func updateOrientation() {
        if size.width > 0, size.height > 0 {
            setPortrait(size.height > size.width)
            return
        }
        guard let window = Window.window else { return }
        setPortrait(window.bounds.size.height > window.bounds.size.width)
    }

    func update(size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        self.size = size
        setPortrait(size.height > size.width)
    }

    private func setPortrait(_ portrait: Bool) {
        if isPortrait != portrait {
            isPortrait = portrait
        }
    }
}
