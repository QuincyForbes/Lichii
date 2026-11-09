// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import GameController
import LichiiBridge
import SwiftUI

/// Connected controllers with the player each one drives.
struct ControllersListView: View {
    @ObservedObject private var controllers = PhysicalControllers.shared

    var body: some View {
        List {
            Section {
                if controllers.pads.isEmpty {
                    Text("No controller connected. Pair one in the Bluetooth settings (Xbox, DualSense, DualShock 4, Switch Pro, Joy-Con or MFi).")
                        .foregroundStyle(.secondary)
                }
                ForEach(controllers.pads) { pad in
                    NavigationLink {
                        PadSettingsView(pad: pad)
                    } label: {
                        PadRow(pad: pad)
                    }
                }
            } header: {
                Text("Controllers")
            } footer: {
                Text("Player 1 also plays as the handheld console. While a controller drives player 1 the on-screen controller can hide itself.")
            }
        }
        .navigationTitle("Controllers")
    }
}

private struct PadRow: View {
    @ObservedObject var pad: PhysicalPad

    var body: some View {
        HStack {
            Image(systemName: "gamecontroller.fill")
                .foregroundStyle(Theme.accent)
            VStack(alignment: .leading) {
                Text(pad.name)
                Text(pad.player >= 0 ? "Player \(pad.player + 1)" : "Not used")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if pad.hasMotion {
                Image(systemName: "gyroscope").foregroundStyle(.secondary)
            }
            if pad.hasHaptics {
                Image(systemName: "waveform").foregroundStyle(.secondary)
            }
        }
    }
}

/// Mapping screen for one controller; saved for its model.
struct PadSettingsView: View {
    let pad: PhysicalPad
    @State private var profile: PadProfile
    private let controllers = PhysicalControllers.shared

    init(pad: PhysicalPad) {
        self.pad = pad
        _profile = State(initialValue: PhysicalControllers.shared.currentProfile(for: pad))
    }

    var body: some View {
        Form {
            Section("Player") {
                Picker("Plays As", selection: playerBinding) {
                    Text("Not Used").tag(Int32(-1))
                    ForEach(0..<8, id: \.self) { Text("Player \($0 + 1)").tag(Int32($0)) }
                }
            }

            Section {
                ForEach(PadElement.all(for: pad)) { element in
                    Picker(element.name, selection: mappingBinding(element.id)) {
                        ForEach(MappingTarget.all) { Text($0.name).tag($0.id) }
                    }
                }
                Button("Default Layout (Nintendo positions)") { reset(byLabel: false) }
                Button("Match Button Labels (Xbox / PlayStation)") { reset(byLabel: true) }
            } header: {
                Text("Buttons")
            } footer: {
                Text("By position the bottom face button is B and the right one is A, like a Switch controller.")
            }

            Section("Sticks") {
                ForEach(0..<2, id: \.self) { stick in
                    VStack(alignment: .leading) {
                        Text(stick == 0 ? "Left Stick" : "Right Stick").font(.subheadline.weight(.semibold))
                        SliderRow(title: "Deadzone", value: floatBinding(\.inner, stick), range: 0...0.5)
                        SliderRow(title: "Full Tilt At", value: floatBinding(\.outer, stick), range: 0.5...1)
                        Toggle("Invert Vertical", isOn: invertBinding(stick))
                    }
                }
                Toggle("Swap Sticks", isOn: binding(\.swapSticks))
            }

            Section("Triggers") {
                SliderRow(title: "ZL / ZR Press Point",
                          value: Binding(get: { profile.triggerThreshold }, set: { profile.triggerThreshold = $0; save() }),
                          range: 0.05...0.95)
            }

            Section("Feedback") {
                Toggle("Rumble", isOn: binding(\.rumble))
                    .disabled(!pad.hasHaptics)
                Toggle("Motion Controls", isOn: binding(\.motion))
                    .disabled(!pad.hasMotion)
            }

            Section {
                Button("Forget Settings for This Controller", role: .destructive) {
                    controllers.forget(pad)
                    profile = controllers.currentProfile(for: pad)
                }
            }
        }
        .navigationTitle(pad.name)
    }

    private func save() {
        controllers.save(profile, for: pad)
    }

    private func reset(byLabel: Bool) {
        lichii_pad_reset_mapping(pad.slot, byLabel)
        profile.mapping = controllers.currentProfile(for: pad).mapping
        save()
    }

    private var playerBinding: Binding<Int32> {
        Binding(get: { profile.player ?? pad.player }, set: {
            profile.player = $0
            save()
        })
    }

    private func mappingBinding(_ element: Int) -> Binding<Int32> {
        Binding(get: { element < profile.mapping.count ? profile.mapping[element] : -1 }, set: {
            guard element < profile.mapping.count else { return }
            profile.mapping[element] = $0
            save()
        })
    }

    private func floatBinding(_ path: WritableKeyPath<PadProfile, [Float]>, _ stick: Int) -> Binding<Float> {
        Binding(get: { profile[keyPath: path][stick] }, set: {
            profile[keyPath: path][stick] = $0
            save()
        })
    }

    private func invertBinding(_ stick: Int) -> Binding<Bool> {
        Binding(get: { profile.invertY[stick] }, set: {
            profile.invertY[stick] = $0
            save()
        })
    }

    private func binding(_ path: WritableKeyPath<PadProfile, Bool>) -> Binding<Bool> {
        Binding(get: { profile[keyPath: path] }, set: {
            profile[keyPath: path] = $0
            save()
        })
    }
}

private struct SliderRow: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>

    var body: some View {
        HStack {
            Text(title).font(.caption)
            Slider(value: $value, in: range, step: 0.01)
            Text(String(format: "%.0f%%", value * 100))
                .font(.caption.monospacedDigit())
                .frame(width: 44, alignment: .trailing)
        }
    }
}

/// A physical element as the mapping screen names it.
struct PadElement: Identifiable {
    let id: Int
    let name: String

    static func all(for pad: PhysicalPad) -> [PadElement] {
        let playstation = pad.kind == Int32(LICHII_PAD_KIND_DUALSENSE.rawValue) ||
            pad.kind == Int32(LICHII_PAD_KIND_DUALSHOCK.rawValue)
        let xbox = pad.kind == Int32(LICHII_PAD_KIND_XBOX.rawValue)
        let names: [(LichiiPadElement, String)] = [
            (LICHII_PAD_SOUTH, playstation ? "Cross (bottom)" : xbox ? "A (bottom)" : "Bottom face button"),
            (LICHII_PAD_EAST, playstation ? "Circle (right)" : xbox ? "B (right)" : "Right face button"),
            (LICHII_PAD_WEST, playstation ? "Square (left)" : xbox ? "X (left)" : "Left face button"),
            (LICHII_PAD_NORTH, playstation ? "Triangle (top)" : xbox ? "Y (top)" : "Top face button"),
            (LICHII_PAD_L1, playstation ? "L1" : "Left bumper"),
            (LICHII_PAD_R1, playstation ? "R1" : "Right bumper"),
            (LICHII_PAD_L2, playstation ? "L2" : "Left trigger"),
            (LICHII_PAD_R2, playstation ? "R2" : "Right trigger"),
            (LICHII_PAD_L3, "Left stick press"),
            (LICHII_PAD_R3, "Right stick press"),
            (LICHII_PAD_MENU, playstation ? "Options" : "Menu"),
            (LICHII_PAD_OPTIONS, playstation ? "Create / Share" : "View"),
            (LICHII_PAD_HOME, "Home / Guide"),
            (LICHII_PAD_SHARE, "Share / Capture"),
            (LICHII_PAD_DPAD_UP, "D-Pad up"),
            (LICHII_PAD_DPAD_DOWN, "D-Pad down"),
            (LICHII_PAD_DPAD_LEFT, "D-Pad left"),
            (LICHII_PAD_DPAD_RIGHT, "D-Pad right"),
            (LICHII_PAD_PADDLE1, "Paddle 1"),
            (LICHII_PAD_PADDLE2, "Paddle 2"),
            (LICHII_PAD_PADDLE3, "Paddle 3"),
            (LICHII_PAD_PADDLE4, "Paddle 4"),
            (LICHII_PAD_TOUCHPAD, "Touchpad press"),
        ]
        return names.compactMap { element, name in
            let paddle = [LICHII_PAD_PADDLE1, LICHII_PAD_PADDLE2, LICHII_PAD_PADDLE3, LICHII_PAD_PADDLE4].contains(element)
            if paddle && !xbox { return nil }
            if element == LICHII_PAD_TOUCHPAD && !playstation { return nil }
            return PadElement(id: Int(element.rawValue), name: name)
        }
    }
}

/// What an element can do: a Switch button, a frontend action, or nothing.
struct MappingTarget: Identifiable {
    let id: Int32
    let name: String

    static let all: [MappingTarget] = [
        MappingTarget(id: -1, name: "Nothing"),
        MappingTarget(id: 0, name: "A"), MappingTarget(id: 1, name: "B"),
        MappingTarget(id: 2, name: "X"), MappingTarget(id: 3, name: "Y"),
        MappingTarget(id: 6, name: "L"), MappingTarget(id: 7, name: "R"),
        MappingTarget(id: 8, name: "ZL"), MappingTarget(id: 9, name: "ZR"),
        MappingTarget(id: 4, name: "Left Stick Press"), MappingTarget(id: 5, name: "Right Stick Press"),
        MappingTarget(id: 10, name: "+"), MappingTarget(id: 11, name: "-"),
        MappingTarget(id: 13, name: "D-Pad Up"), MappingTarget(id: 15, name: "D-Pad Down"),
        MappingTarget(id: 12, name: "D-Pad Left"), MappingTarget(id: 14, name: "D-Pad Right"),
        MappingTarget(id: 16, name: "SL"), MappingTarget(id: 17, name: "SR"),
        MappingTarget(id: 18, name: "Home"), MappingTarget(id: 19, name: "Capture"),
        MappingTarget(id: LICHII_ACTION_MENU.rawValue, name: "Lichii Menu"),
        MappingTarget(id: LICHII_ACTION_SCREENSHOT.rawValue, name: "Screenshot"),
        MappingTarget(id: LICHII_ACTION_TOGGLE_HUD.rawValue, name: "Toggle HUD"),
        MappingTarget(id: LICHII_ACTION_TOGGLE_SPEED_LIMIT.rawValue, name: "Toggle Speed Limit"),
    ]
}
