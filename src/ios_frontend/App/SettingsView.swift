// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

enum SettingsScope: Equatable {
    case global
    case game(Game)
}

/// Every setting the core saves, generated from the bridge's metadata. The per-game scope lists the
/// settings a game can override, each with a switch back to the global value.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    let scope: SettingsScope
    @State private var ready = false
    @State private var query = ""
    @State private var showReset = false
    @State private var generation = 0

    private var game: Game? {
        if case .game(let game) = scope { return game }
        return nil
    }

    var body: some View {
        List {
            if ready {
                if let game {
                    Section {
                        HStack(spacing: 12) {
                            GameIcon(game: game, size: 44)
                            VStack(alignment: .leading) {
                                Text(game.title).font(.headline)
                                Text("Overrides apply to this game only. The globe uses the global value.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } else {
                    StatusHeader()
                    if !AppExtensions.settingsLinks.isEmpty, query.isEmpty {
                        Section {
                            ForEach(AppExtensions.settingsLinks) { link in
                                NavigationLink {
                                    link.view()
                                } label: {
                                    Label(link.title, systemImage: link.systemImage)
                                }
                            }
                        }
                    }
                    if query.isEmpty {
                        ShaderCacheSettingsSection()
                    }
                }
                ForEach(SettingsCatalog.sections(perGame: game != nil, query: query)) { section in
                    Section(section.title) {
                        ForEach(section.items) { item in
                            SettingRow(item: item, perGame: game != nil)
                        }
                    }
                }
                .id(generation)
                if game != nil {
                    Section {
                        Button("Remove All Overrides", role: .destructive) { showReset = true }
                    }
                }
            }
        }
        .navigationTitle(game == nil ? "Settings" : "Game Settings")
        .searchable(text: $query, prompt: "Search settings")
        .toolbar {
            if game != nil {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .confirmationDialog("Use the global value for every setting of this game?", isPresented: $showReset,
                            titleVisibility: .visible) {
            Button("Remove Overrides", role: .destructive) {
                for item in SettingsCatalog.all where item.switchable {
                    item.setUsesGlobal(true)
                }
                generation += 1
            }
        }
        .onAppear(perform: open)
        .onDisappear(perform: close)
    }

    private func open() {
        guard !ready else { return }
        if let game {
            ready = lichii_settings_begin_game(game.programId)
        } else {
            ready = true
        }
    }

    private func close() {
        guard ready else { return }
        if game != nil {
            lichii_settings_end_game()
        } else {
            lichii_settings_save()
        }
        ready = false
    }
}

/// Build, JIT and device summary at the top of the global settings.
private struct StatusHeader: View {
    var body: some View {
        Section {
            LabeledContent("Version", value: Bridge.buildVersion)
            LabeledContent("JIT", value: jitText)
            DeviceInfoRows()
        }
    }

    private var jitText: String {
        switch Bridge.jitState {
        case .waiting: return lichii_jit_debugged() ? "Attached, waiting for the script" : "Not enabled"
        case .acquiring: return "Preparing"
        case .ready: return "Ready"
        case .failed: return "Failed"
        }
    }
}

struct SettingRow: View {
    let item: SettingItem
    let perGame: Bool
    @State private var value: String
    @State private var useGlobal: Bool

    init(item: SettingItem, perGame: Bool) {
        self.item = item
        self.perGame = perGame
        _value = State(initialValue: item.value())
        _useGlobal = State(initialValue: perGame ? item.usesGlobal : false)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            control
                .disabled(perGame && useGlobal)
                .opacity(perGame && useGlobal ? 0.55 : 1)
                .disabled(SettingLabels.unavailable[item.key] != nil)
                .help(SettingLabels.unavailable[item.key] ?? "")
            if perGame {
                Button {
                    useGlobal.toggle()
                    item.setUsesGlobal(useGlobal)
                    if !useGlobal {
                        item.set(value)
                    }
                    value = item.value()
                } label: {
                    Image(systemName: useGlobal ? "globe" : "pencil.circle.fill")
                        .foregroundStyle(useGlobal ? Color.secondary : Theme.accent)
                }
                .buttonStyle(.borderless)
                .accessibilityLabel(useGlobal ? "Using the global value" : "Overridden for this game")
            }
        }
    }

    @ViewBuilder
    private var control: some View {
        switch item.kind {
        case .bool:
            Toggle(item.label, isOn: Binding(get: { value == "true" },
                                             set: { commit($0 ? "true" : "false") }))
        case .enumeration:
            Picker(item.label, selection: Binding(get: { value }, set: { commit($0) })) {
                ForEach(item.choices, id: \.value) { choice in
                    Text(choice.name).tag(choice.value)
                }
            }
            .pickerStyle(.menu)
        case .int, .float:
            if item.ranged, item.maximum > item.minimum, item.maximum - item.minimum <= 5000 {
                NumberSlider(item: item, value: value, commit: commit)
            } else {
                TextRow(label: item.label, value: value, numeric: true, commit: commit)
            }
        case .string:
            TextRow(label: item.label, value: value, numeric: false, commit: commit)
        }
    }

    private func commit(_ newValue: String) {
        if perGame, useGlobal {
            return
        }
        item.set(newValue)
        value = item.value()
    }
}

private struct NumberSlider: View {
    let item: SettingItem
    let commit: (String) -> Void
    @State private var number: Double

    init(item: SettingItem, value: String, commit: @escaping (String) -> Void) {
        self.item = item
        self.commit = commit
        _number = State(initialValue: Double(value) ?? item.minimum)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(item.label)
                Spacer()
                Text(formatted)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: $number, in: item.minimum...item.maximum,
                   step: item.kind == .int ? 1 : (item.maximum - item.minimum) / 100,
                   onEditingChanged: { editing in
                if !editing {
                    commit(item.kind == .int ? String(Int(number.rounded())) : String(number))
                }
            })
        }
    }

    private var formatted: String {
        let text = item.kind == .int ? String(Int(number.rounded())) : String(format: "%.2f", number)
        return item.percentage ? text + "%" : text
    }
}

private struct TextRow: View {
    let label: String
    let numeric: Bool
    let commit: (String) -> Void
    @State private var text: String

    init(label: String, value: String, numeric: Bool, commit: @escaping (String) -> Void) {
        self.label = label
        self.numeric = numeric
        self.commit = commit
        _text = State(initialValue: value)
    }

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            TextField(label, text: $text)
                .multilineTextAlignment(.trailing)
                .keyboardType(numeric ? .numbersAndPunctuation : .default)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .onSubmit { commit(text) }
                .foregroundStyle(.secondary)
        }
    }
}
