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

    init(controller: any Controller, right: Bool, showBackground: Binding<Bool>) {
        self.right = right
        self._showBackground = showBackground
        self.controller = controller
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
        view.right = right

        view.onPositionChanged = { newPosition in
            controller.joystickMoved(position: newPosition, right: right)
        }

        view.onActiveChanged = { active in
            showBackground.wrappedValue = active
        }
    }
}
