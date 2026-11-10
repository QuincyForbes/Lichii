// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  JoystickViewRepresentable.swift
//  Melo-Controller
//
//  Created by Stossy11 on 28/2/2026.
//


import SwiftUI

struct JoystickViewRepresentable: UIViewRepresentable {
    var right: Bool
    @Binding var showBackground: Bool
    var controller: any Controller
    // Lichii
    var floating = false
    var deferTaps = false
    /// Only then is `showBackground` written: each write redraws the whole controller.
    var hidesButtons = true

    init(controller: any Controller, right: Bool, showBackground: Binding<Bool>,
         floating: Bool = false, deferTaps: Bool = false, hidesButtons: Bool = true) {
        self.right = right
        self._showBackground = showBackground
        self.controller = controller
        self.floating = floating
        self.deferTaps = deferTaps
        self.hidesButtons = hidesButtons
    }

    func makeUIView(context: Context) -> JoystickView {
        let view = JoystickView()
        apply(to: view)
        return view
    }

    func updateUIView(_ uiView: JoystickView, context: Context) {
        apply(to: uiView)
    }

    private func apply(to view: JoystickView) {
        let controller = controller
        let right = right
        let showBackground = _showBackground
        let hidesButtons = hidesButtons
        view.right = right
        view.floating = floating
        view.deferTaps = deferTaps

        view.onPositionChanged = { newPosition in
            controller.joystickMoved(position: newPosition, right: right)
        }

        view.onActiveChanged = { active in
            if hidesButtons || !active, showBackground.wrappedValue != active {
                showBackground.wrappedValue = active
            }
        }

        view.onFaceButton = { quadrant, pressed in
            let button: VirtualControllerButton
            switch quadrant {
            case .top: button = right ? .X : .dPadUp
            case .left: button = right ? .Y : .dPadLeft
            case .right: button = right ? .A : .dPadRight
            case .bottom: button = right ? .B : .dPadDown
            }
            if pressed {
                controller.buttonPressed(button)
                Haptics.shared.play(.soft)
            } else {
                controller.buttonReleased(button)
            }
        }
    }
}
