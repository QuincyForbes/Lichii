// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import Melo_Controller
import SwiftUI
import UIKit

/// Everything drawn over the game: the on-screen controller, the HUD, the menu button, toasts and
/// the in-game menu.
struct ControlsOverlayView: View {
    @ObservedObject private var center = ControlsCenter.shared
    @AppStorage("hudPosition") private var hudPosition = HUDPosition.topRight.rawValue
    @AppStorage("menuButtonVisible") private var menuButtonVisible = true

    private var gameId: String? { center.currentGame?.programIdHex }

    var body: some View {
        ZStack {
            if center.touchControlsVisible || center.editingLayout {
                ControllerView(
                    controller: TouchController.shared,
                    isEditing: center.editingLayout,
                    gameId: perGameLayouts ? gameId : nil,
                    gameName: perGameLayouts ? center.currentGame?.title : nil,
                    onEditingChanged: { center.layoutEditingChanged($0) }
                )
                .id("\(gameId ?? "")-\(perGameLayouts)")
            }

            if center.showHUD && !center.editingLayout {
                PerformanceHUD()
                    .padding(10)
                    .frame(maxWidth: .infinity, maxHeight: .infinity,
                           alignment: (HUDPosition(rawValue: hudPosition) ?? .topRight).alignment)
            }

            if menuButtonVisible && !center.editingLayout && !center.menuOpen {
                MenuButton { center.openMenu() }
                    .frame(width: 36, height: 36)
                    .padding(12)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }

            if let toast = center.toast {
                Text(toast)
                    .font(.footnote.weight(.semibold))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(.top, 16)
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }

            if center.menuOpen {
                InGameMenuView()
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: center.menuOpen)
        .animation(.easeInOut(duration: 0.2), value: center.toast)
        .ignoresSafeArea(.keyboard)
    }

    /// Layouts are saved per game unless turned off in the controls settings.
    private var perGameLayouts: Bool {
        UserDefaults.standard.object(forKey: "perGameLayouts") as? Bool ?? true
    }
}

/// A UIKit button, so it takes touches while the rest of the overlay lets them through.
private struct MenuButton: UIViewRepresentable {
    let action: () -> Void

    func makeUIView(context: Context) -> UIButton {
        let button = UIButton(type: .system)
        let config = UIImage.SymbolConfiguration(pointSize: 22, weight: .semibold)
        button.setImage(UIImage(systemName: "gearshape.circle.fill", withConfiguration: config), for: .normal)
        button.tintColor = .white
        button.alpha = 0.45
        button.addAction(UIAction { _ in context.coordinator.action() }, for: .touchUpInside)
        return button
    }

    func updateUIView(_ uiView: UIButton, context: Context) {
        context.coordinator.action = action
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(action: action)
    }

    final class Coordinator {
        var action: () -> Void

        init(action: @escaping () -> Void) {
            self.action = action
        }
    }
}
