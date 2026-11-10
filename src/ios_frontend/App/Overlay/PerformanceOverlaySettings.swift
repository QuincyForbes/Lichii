// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// Settings > Performance Overlay.
struct PerformanceOverlaySettingsView: View {
    @AppStorage("overlayPreset") private var presetName = ""
    @AppStorage("hudPosition") private var positionName = HUDPosition.topRight.rawValue
    @AppStorage("overlayVoiceOver") private var voiceOver = false

    var body: some View {
        Form {
            Section {
                Picker("Performance Overlay", selection: Binding(
                    get: { OverlayPreset.resolve(presetName) },
                    set: { OverlayPreset.store($0) }
                )) {
                    ForEach(OverlayPreset.allCases) { Text($0.title).tag($0) }
                }
                Picker("Position", selection: $positionName) {
                    ForEach(HUDPosition.allCases) { Text($0.title).tag($0.rawValue) }
                }
            } footer: {
                Text("Shown over the game, with the same numbers as the desktop overlay. The in-game menu changes it too.")
            }

            Section("What Each Preset Shows") {
                ForEach(OverlayPreset.allCases.filter { $0 != .off }) { preset in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(preset.title)
                        Text(preset.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section {
                Toggle("Read Overlay with VoiceOver", isOn: $voiceOver)
            } footer: {
                Text("The overlay is hidden from VoiceOver by default. The in-game menu reads a summary instead.")
            }
        }
        .navigationTitle("Performance Overlay")
    }
}

/// The in-game menu's quick toggle: a tile that steps Off, Minimal, Standard, Detailed.
struct PerformanceOverlayTile: View {
    @AppStorage("overlayPreset") private var presetName = ""

    var body: some View {
        let preset = OverlayPreset.resolve(presetName)
        Button {
            OverlayPreset.store(preset.next)
        } label: {
            VStack(spacing: 6) {
                Image(systemName: "speedometer")
                    .font(.title3)
                Text("Overlay: \(preset.title)")
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .frame(maxWidth: .infinity, minHeight: 62)
            .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Performance overlay")
        .accessibilityValue(preset.title)
        .accessibilityHint("Cycles through Off, Minimal, Standard and Detailed")
    }
}

/// One line for the in-game menu that VoiceOver reads in full: the latest reading while an
/// overlay has been running, otherwise how to turn one on.
struct PerformanceSummaryRow: View {
    @ObservedObject private var monitor = PerformanceOverlayMonitor.shared
    @AppStorage("overlayPreset") private var presetName = ""

    var body: some View {
        let reading = monitor.reading
        let running = OverlayPreset.resolve(presetName) != .off && reading.active
        Text(running ? reading.spoken : "Performance: turn the overlay on to measure.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .accessibilityLabel(running ? reading.spoken : "Performance: turn the overlay on to measure.")
    }
}
