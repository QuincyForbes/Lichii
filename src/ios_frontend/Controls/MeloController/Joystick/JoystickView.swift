// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  JoystickView.swift
//  Melo-Controller
//
//  Created by Stossy11 on 26/1/2026.
//

import SwiftUI

struct EditableJoystickView: View {
    let id: String
    let iscool: Bool
    var controller: any Controller
    @Binding var showBackground: Bool
    @Binding var layout: LayoutConfig
    var isEditing: Bool
    @Binding var selectedJoystick: String?
    @Binding var selectedButton: String?
    @GestureState private var dragOffset = CGSize.zero
    @AppStorage("On-ScreenControllerScale") var controllerScale: Double = 1.0

    private var editorPlaceholder: some View {
        ZStack {
            Circle()
                .fill(Color.gray.opacity(0.22))

            Circle()
                .strokeBorder(
                    Color.white.opacity(0.3),
                    style: StrokeStyle(lineWidth: 2, dash: [7, 6])
                )

            VStack(spacing: 4) {
                Image(systemName: iscool ? "r.joystick.fill" : "l.joystick.fill")
                    .font(.system(size: 30, weight: .semibold))
                Text(iscool ? "Right Stick" : "Left Stick")
                    .font(.caption2.weight(.semibold))
            }
            .foregroundColor(.white.opacity(0.8))
        }
        .frame(width: 160, height: 160)
    }

    var body: some View {
        if isEditing {
            editorPlaceholder
                .opacity(layout.joysticks[id]?.hidden ?? false ? 0.35 : 1)
                .editorSelectionRing(isSelected: selectedJoystick == id, tint: .green)
                .scaleEffect(
                    (layout.joysticks[id]?.scale ?? 1.0)
                        * controllerScale
                        * (selectedJoystick == id ? 1.04 : 1.0)
                )
                .offset(
                    x: (layout.joysticks[id]?.offset.width ?? 0) + dragOffset.width,
                    y: (layout.joysticks[id]?.offset.height ?? 0) + dragOffset.height
                )
                .onTapGesture {
                    selectedJoystick = selectedJoystick == id ? nil : id
                    selectedButton = nil
                    Haptics.shared.play(.light)
                }
                .gesture(
                    DragGesture()
                        .updating($dragOffset) { value, state, _ in
                            state = value.translation
                            selectedJoystick = id
                            selectedButton = nil
                        }
                        .onEnded { value in
                            layout.joysticks[id, default: JoystickLayout()].offset.width += value.translation.width
                            layout.joysticks[id, default: JoystickLayout()].offset.height += value.translation.height
                        }
                )
        } else {
            JoystickViewRepresentable(controller: controller, right: iscool, showBackground: $showBackground)
                .frame(width: 160, height: 160)
                .scaleEffect((layout.joysticks[id]?.scale ?? 1.0) * controllerScale)
                .offset(layout.joysticks[id]?.offset ?? .zero)
        }
    }
}
