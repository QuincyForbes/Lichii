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

    /// Lichii: the space the controller is offered, once known. The key window can still be in
    /// portrait while the game window turns to landscape, which picked the portrait layout.
    private var size: CGSize = .zero

    func updateOrientation() {
        if size.width > 0, size.height > 0 {
            setPortrait(for: size)
            return
        }
        guard let window = Window.window else { return }
        setPortrait(for: window.bounds.size)
    }

    func update(size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        self.size = size
        setPortrait(for: size)
    }

    /// Lichii: a clear majority decides; a near-square size in passing (a rotation, a window
    /// being resized) keeps the current layout instead of flipping it.
    private func setPortrait(for size: CGSize) {
        let portrait: Bool
        if size.height > size.width * 1.15 {
            portrait = true
        } else if size.width > size.height * 1.15 {
            portrait = false
        } else {
            return
        }
        if isPortrait != portrait {
            isPortrait = portrait
        }
    }
}
