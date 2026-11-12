// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI

/// One value the device profile recommends (common/device_profile.cpp).
struct RecommendedValue: Identifiable {
    let key: String
    let value: String
    let display: String
    let reason: String
    let app: Bool

    var id: String { key }

    @MainActor
    var item: SettingItem? {
        app ? nil : SettingsCatalog.all.first { $0.key == key }
    }

    @MainActor
    var label: String {
        switch key {
        case "guest_memory_file": return "Keep Guest Memory in a File"
        case "auto_cool": return "Auto-Cool (Thermal Pacing)"
        default:
            guard let item else { return SettingLabels.title(for: key) }
            if item.game != 0 {
                return "\(SettingsCatalog.title(of: SettingsCatalog.gameCategory(item.game))): \(item.label)"
            }
            return item.label
        }
    }

    @MainActor
    var recommendedText: String {
        item?.displayName(of: value) ?? (value == "true" ? "On" : "Off")
    }

    /// The value now, in the same words.
    @MainActor
    var currentText: String {
        switch key {
        case "guest_memory_file": return GuestMemoryFile.enabled ? "On" : "Off"
        case "auto_cool": return PlatformCenter.shared.autoCool ? "On" : "Off"
        default:
            guard let item else { return "" }
            return item.displayName(of: item.value())
        }
    }

    @MainActor
    var differs: Bool {
        switch key {
        case "guest_memory_file": return GuestMemoryFile.enabled != (value == "true")
        case "auto_cool": return PlatformCenter.shared.autoCool != (value == "true")
        default: return item.map { $0.value() != value } ?? false
        }
    }
}

enum RecommendedSettings {
    static func load() -> (device: String, values: [RecommendedValue]) {
        let count = lichii_recommended_load()
        let values = (0..<count).compactMap { index -> RecommendedValue? in
            var raw = LichiiRecommended()
            guard lichii_recommended_get(index, &raw) else { return nil }
            return RecommendedValue(key: Bridge.constant(raw.key), value: Bridge.constant(raw.value),
                                    display: Bridge.constant(raw.display),
                                    reason: Bridge.constant(raw.reason), app: raw.app)
        }
        return (Bridge.string(lichii_recommended_device()), values)
    }

    /// Writes every recommended value; false while a game runs.
    @MainActor
    static func apply(_ values: [RecommendedValue]) -> Bool {
        guard lichii_recommended_apply() >= 0 else { return false }
        for value in values where value.app {
            let on = value.value == "true"
            switch value.key {
            case "guest_memory_file":
                if GuestMemoryFile.enabled != on {
                    GuestMemoryFile.enabled = on
                }
            case "auto_cool":
                PlatformCenter.shared.autoCool = on
            default:
                break
            }
        }
        lichii_platform_log_memory("recommended settings applied")
        return true
    }
}

/// Settings > Recommended Settings: what this device should use and why, and one button that sets
/// it all (after showing what changes).
struct RecommendedSettingsView: View {
    @State private var device = ""
    @State private var values: [RecommendedValue] = []
    @State private var confirm = false
    @State private var done = false
    @State private var generation = 0

    private var changes: [RecommendedValue] {
        values.filter(\.differs)
    }

    var body: some View {
        List {
            Section {
                Text(device).font(.headline)
                Button {
                    confirm = true
                } label: {
                    Label("Use Recommended Settings for This Device", systemImage: "wand.and.stars")
                }
                .disabled(Bridge.emuState.inGame || changes.isEmpty)
            } footer: {
                if Bridge.emuState.inGame {
                    Text("Quit the game first: it has its own values loaded.")
                } else if done {
                    Text("Applied. Games use them from their next start; a game's own overrides stay until you reset that game in its settings.")
                } else if changes.isEmpty {
                    Text("Everything is already set as recommended.")
                } else {
                    Text("\(changes.count) setting\(changes.count == 1 ? "" : "s") differ from the recommendation (marked below).")
                }
            }
            Section("Recommended") {
                ForEach(values) { value in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(value.label)
                            Spacer()
                            Text(value.recommendedText)
                                .foregroundStyle(.secondary)
                        }
                        if value.differs {
                            Label("Now: \(value.currentText)", systemImage: "arrow.triangle.2.circlepath")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                        Text(value.reason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .id(generation)
        }
        .navigationTitle("Recommended Settings")
        .onAppear(perform: reload)
        .confirmationDialog("Change these settings?", isPresented: $confirm, titleVisibility: .visible) {
            Button("Use Recommended Settings") {
                done = RecommendedSettings.apply(values)
                generation += 1
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(changes.map { "\($0.label): \($0.currentText) → \($0.recommendedText)" }.joined(separator: "\n"))
        }
    }

    private func reload() {
        let loaded = RecommendedSettings.load()
        device = loaded.device
        values = loaded.values
    }
}
