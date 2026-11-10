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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

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

            if !center.editingLayout {
                PerformanceOverlayHost()
            }

            // The menu button is UIKit, above this view (OverlayController).

            if !center.editingLayout && !center.menuOpen {
                VStack(spacing: 6) {
                    if center.manuallyPaused {
                        Label("Paused", systemImage: "pause.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.black.opacity(0.6), in: Capsule())
                    }
                    ThermalBanner(compact: true, transient: true)
                }
                .padding(.bottom, 10)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                .allowsHitTesting(false)
            }

            if center.menuOpen {
                InGameMenuView()
                    .transition(.opacity)
            }

            // Above the menu: a controller that went away opens it with a toast.
            if let toast = center.toast {
                Text(toast)
                    .font(.footnote.weight(.semibold))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.horizontal, 60)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(.top, 16)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: center.menuOpen)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: center.toast)
        .ignoresSafeArea(.keyboard)
    }

    /// Layouts are saved per game unless turned off in the controls settings.
    private var perGameLayouts: Bool {
        UserDefaults.standard.object(forKey: "perGameLayouts") as? Bool ?? true
    }
}
