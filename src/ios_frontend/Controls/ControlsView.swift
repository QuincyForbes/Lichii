// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import Melo_Controller
import SwiftUI

/// Settings > Controllers: on-screen controller, physical controllers, feedback, HUD, orientation
/// and external display.
struct ControlsView: View {
    @ObservedObject private var controllers = PhysicalControllers.shared
    @ObservedObject private var center = ControlsCenter.shared
    @ObservedObject private var orientation = OrientationManager.shared
    @ObservedObject private var external = ExternalDisplay.shared
    @AppStorage("On-ScreenControllerScale") private var scale: Double = 1.0
    @AppStorage("On-ScreenControllerOpacity") private var opacity: Double = 1.0
    @AppStorage("On-ScreenControllerHaptics") private var haptics = true
    @AppStorage("buttonSlide") private var buttonSlide = true
    @AppStorage("stickButton") private var stickButtons = false
    @AppStorage("perGameLayouts") private var perGameLayouts = true
    @AppStorage("menuButtonVisible") private var menuButtonVisible = true
    @AppStorage("menuChord") private var menuChord = true
    @State private var editing = false
    @State private var confirmReset = false

    var body: some View {
        Form {
            Section {
                Toggle("Show On-Screen Controller", isOn: Binding(
                    get: { center.showTouchControls }, set: { center.showTouchControls = $0 }))
                Toggle("Hide While a Controller Is Connected", isOn: Binding(
                    get: { center.hideTouchWithController }, set: { center.hideTouchWithController = $0 }))
                LabeledSlider(title: "Size", value: $scale, range: 0.5...2.0, format: "%.1f×")
                LabeledSlider(title: "Opacity", value: $opacity, range: 0.1...1.0, format: "%.0f%%", factor: 100)
                Toggle("Haptic Feedback", isOn: $haptics)
                Toggle("Roll Between Buttons", isOn: $buttonSlide)
                Toggle("Stick Press Buttons", isOn: $stickButtons)
                Toggle("Separate Layout per Game", isOn: $perGameLayouts)
                Button("Edit Default Layout") { editing = true }
                Button("Reset All Layouts", role: .destructive) { confirmReset = true }
            } header: {
                Text("On-Screen Controller")
            } footer: {
                Text("Portrait and landscape keep their own layouts. Drag a control to move it; tap it to change its size, opacity or behaviour.")
            }

            Section("Controllers") {
                NavigationLink {
                    ControllersListView()
                } label: {
                    LabeledContent("Connected", value: "\(controllers.pads.count)")
                }
                Toggle("Rumble on This Device", isOn: Binding(
                    get: { controllers.deviceRumbleEnabled }, set: { controllers.deviceRumbleEnabled = $0 }))
                Toggle("Motion from This Device", isOn: Binding(
                    get: { controllers.deviceMotionEnabled }, set: {
                        controllers.deviceMotionEnabled = $0
                        controllers.updateDeviceMotion()
                    }))
                Toggle("Rumble (all players)", isOn: Binding(
                    get: { QuickSettings.bool("vibration_enabled") },
                    set: { QuickSettings.set("vibration_enabled", $0) }))
            }

            Section {
                Toggle("Menu Button", isOn: $menuButtonVisible)
                Toggle("+ and − Together Open the Menu", isOn: $menuChord)
                    .onChange(of: menuChord) { lichii_pad_set_menu_chord($0) }
            } header: {
                Text("In-Game Menu")
            } footer: {
                Text("A three-finger tap on the game also opens the menu. Controller buttons can be mapped to it as well.")
            }

            Section {
                NavigationLink("Performance Overlay") { PerformanceOverlaySettingsView() }
            }

            Section {
                Picker("Game Orientation", selection: $orientation.mode) {
                    ForEach(OrientationManager.Mode.allCases) { Text($0.title).tag($0) }
                }
                Toggle("Play on External Display", isOn: Binding(
                    get: { external.enabled }, set: { external.enabled = $0 }))
            } header: {
                Text("Display")
            } footer: {
                Text(external.connected ? "An external display is connected." :
                        "With a display or AirPlay connected, the game moves there and this device shows the controls.")
            }
        }
        .navigationTitle("Controls")
        .fullScreenCover(isPresented: $editing) {
            LayoutEditorScreen(gameId: nil, gameName: nil)
        }
        .alert("Reset every layout?", isPresented: $confirmReset) {
            Button("Reset", role: .destructive) { LayoutManager.shared.resetAll() }
            Button("Cancel", role: .cancel) {}
        }
    }
}

private struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    var factor: Double = 1

    var body: some View {
        HStack {
            Text(title)
            Slider(value: $value, in: range)
            Text(String(format: format, value * factor))
                .font(.caption.monospacedDigit())
                .frame(width: 48, alignment: .trailing)
        }
    }
}

/// The layout editor outside a game (settings, or a game's context menu), over a dark stand-in
/// for the game screen.
struct LayoutEditorScreen: View {
    let gameId: String?
    let gameName: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(white: 0.12), Color(white: 0.04)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            ControllerView(controller: EditorController(), isEditing: true, gameId: gameId, gameName: gameName,
                           onEditingChanged: { editing in
                               if !editing { dismiss() }
                           })
        }
        .statusBarHidden()
    }
}

/// Input from the editor's preview goes nowhere.
private final class EditorController: ObservableObject, Controller {
    func buttonPressed(_ button: VirtualControllerButton) {}
    func buttonReleased(_ button: VirtualControllerButton) {}
    func joystickMoved(position: CGPoint, right: Bool) {}
}
