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
/// settings a game can override, each with a switch back to the global value. Settings that do
/// nothing on iOS are hidden or greyed out (SettingsCatalog.availability); debug switches and hacks
/// are on the Advanced page.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    let scope: SettingsScope
    @State private var ready = false
    @State private var query = ""
    @State private var showReset = false
    @State private var showAdvanced = false
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
                                Text("Overrides apply to this game only. Tap the globe next to a setting to override it.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    SettingSectionsList(sections: SettingsCatalog.sections(perGame: true, query: query),
                                        perGame: true)
                        .id(generation)
                    if query.isEmpty {
                        // Inline rather than pushed: the game's settings scope ends when this view
                        // disappears.
                        Section {
                            Toggle("Show Advanced Settings", isOn: $showAdvanced)
                        } footer: {
                            Text("Debug switches, CPU and GPU hacks. Changing them can stop the game from starting.")
                        }
                        if showAdvanced {
                            SettingSectionsList(sections: SettingsCatalog.sections(perGame: true, query: "", advanced: true),
                                                perGame: true)
                                .id(generation)
                        }
                    }
                    Section {
                        Button("Remove All Overrides", role: .destructive) { showReset = true }
                    }
                } else if !query.isEmpty {
                    SettingSectionsList(sections: SettingsCatalog.sections(group: .graphics, query: query),
                                        perGame: false)
                } else {
                    GlobalSettingsRoot()
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
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            // Global changes survive the app being closed from the app switcher.
            if game == nil {
                lichii_settings_save()
            }
        }
    }

    private func open() {
        guard !ready else { return }
        if let game {
            ready = lichii_settings_begin_game(game.programId)
        } else {
            ready = true
        }
    }

    /// The global list stays up while a linked screen is pushed over it; it is saved each time it
    /// leaves the screen.
    private func close() {
        guard ready else { return }
        if game != nil {
            lichii_settings_end_game()
            ready = false
        } else {
            lichii_settings_save()
        }
    }
}

private struct SettingSectionsList: View {
    let sections: [SettingSection]
    let perGame: Bool

    var body: some View {
        ForEach(sections) { section in
            Section(section.title) {
                ForEach(section.items) { item in
                    SettingRow(item: item, perGame: perGame)
                }
            }
        }
    }
}

/// The global settings' first page: a compact header, the setting pages, then the screens for
/// game data and help. Links other areas add (AppExtensions.settingsLinks) are placed by id.
private struct GlobalSettingsRoot: View {
    private static let dataLinks = ["keys", "users"]
    private static let helpLinks = ["setup", "logs", "experiments"]

    var body: some View {
        let links = AppExtensions.settingsLinks
        SettingsHeader()
        Section {
            groupLink(.graphics)
            ForEach(links.filter { $0.id == "controls" }) { link(for: $0) }
            groupLink(.audio)
            groupLink(.system)
        }
        Section("Games & Data") {
            ForEach(links.filter { Self.dataLinks.contains($0.id) }) { link(for: $0) }
        }
        ShaderCacheSettingsSection()
        Section("Help") {
            ForEach(links.filter { Self.helpLinks.contains($0.id) }) { link(for: $0) }
            NavigationLink {
                DiagnosticsView()
            } label: {
                Label("About This Device", systemImage: "info.circle")
            }
        }
        let others = links.filter { !(["controls"] + Self.dataLinks + Self.helpLinks).contains($0.id) }
        if !others.isEmpty {
            Section {
                ForEach(others) { link(for: $0) }
            }
        }
        Section {
            groupLink(.advanced)
        } footer: {
            Text("Debug switches, CPU and GPU hacks. Changing them can stop games from starting.")
        }
    }

    private func groupLink(_ group: SettingsGroup) -> some View {
        NavigationLink {
            SettingsGroupView(group: group)
        } label: {
            Label(group.title, systemImage: group.systemImage)
        }
    }

    private func link(for link: SettingsLink) -> some View {
        NavigationLink {
            link.view()
        } label: {
            Label(link.title, systemImage: link.systemImage)
        }
    }
}

/// One page of the global settings (Graphics, Audio, System, Advanced).
private struct SettingsGroupView: View {
    let group: SettingsGroup

    var body: some View {
        List {
            if group == .advanced {
                Section {
                } footer: {
                    Label("These settings are for testing and troubleshooting. If a game stops starting after a change here, set it back.",
                          systemImage: "exclamationmark.triangle")
                }
            }
            SettingSectionsList(sections: SettingsCatalog.sections(group: group, query: ""), perGame: false)
            if group == .graphics {
                // App-side graphics options of other areas (the performance overlay, ...).
                ForEach(AppExtensions.graphicsSettingsSections.indices, id: \.self) { index in
                    AppExtensions.graphicsSettingsSections[index]()
                }
            }
        }
        .navigationTitle(group.title)
        .onDisappear { lichii_settings_save() }
    }
}

/// App version and JIT state; opens the device details.
private struct SettingsHeader: View {
    /// Follows LICHII_EVENT_JIT_CHANGED.
    @ObservedObject private var app = AppController.shared

    var body: some View {
        Section {
            NavigationLink {
                DiagnosticsView()
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "gamecontroller.fill")
                        .font(.title2)
                        .foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                        .background(Theme.accent, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Lichii").font(.headline)
                        Text(Bridge.buildVersion)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer(minLength: 8)
                    JitPill(status: app.jit)
                }
            }
            .accessibilityElement(children: .combine)
        }
    }
}

/// The JIT state in a few words.
struct JitPill: View {
    let status: JitStatus

    var body: some View {
        Label(text, systemImage: status.symbol)
            .font(.caption.weight(.semibold))
            .foregroundStyle(status.color)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(status.color.opacity(0.15), in: Capsule())
            .fixedSize()
    }

    private var text: String {
        switch status.state {
        case .waiting: return "JIT off"
        case .acquiring: return "JIT starting"
        case .ready: return "JIT ready"
        case .failed: return "JIT failed"
        }
    }
}

/// Build, JIT and device details, for bug reports and for checking the memory entitlements.
struct DiagnosticsView: View {
    @ObservedObject private var app = AppController.shared
    private let info = PlatformCenter.deviceInfo
    private let defaults = PlatformCenter.deviceDefaults

    var body: some View {
        List {
            Section {
                LabeledContent("Version", value: Bridge.buildVersion)
                LabeledContent("JIT", value: jitText)
            }
            if let info {
                Section {
                    LabeledContent("Memory", value: "\(gigabytes(info.launch_available_memory_mib)) available to games")
                    LabeledContent("Installed memory", value: gigabytes(info.physical_memory_mib))
                    LabeledContent("Increased Memory Limit", value: entitlement(info.increased_memory_limit))
                    LabeledContent("Extended Virtual Addressing", value: entitlement(info.extended_virtual_addressing))
                } header: {
                    Text("Memory")
                } footer: {
                    Text("Games need both entitlements. Without them iOS gives Lichii too little memory to start a game.")
                }
                Section("Device") {
                    LabeledContent("Model", value: "\(PlatformCenter.text(info.machine)), iOS \(info.os_major).\(info.os_minor)")
                    LabeledContent("GPU", value: PlatformCenter.text(info.gpu_name))
                    if let defaults {
                        LabeledContent("Performance tier", value: PlatformCenter.text(defaults.tier_name))
                    }
                }
                if let defaults, !PlatformCenter.text(defaults.note).isEmpty {
                    Section {
                        Text(PlatformCenter.text(defaults.note))
                            .font(.footnote)
                            .foregroundStyle(defaults.tier == Int32(LICHII_TIER_UNSUPPORTED.rawValue) ? Color.orange : Color.secondary)
                    }
                }
            } else {
                Section("Device") {
                    DeviceInfoRows()
                }
            }
        }
        .navigationTitle("About This Device")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var jitText: String {
        switch app.jit.state {
        case .waiting: return app.jit.debugged ? "Attached, waiting for the script" : "Not enabled"
        case .acquiring: return "Preparing"
        case .ready: return "Ready"
        case .failed: return "Failed"
        }
    }

    private func gigabytes(_ mib: UInt64) -> String {
        String(format: "%.1f GB", Double(mib) / 1024)
    }

    private func entitlement(_ value: Int32) -> String {
        value > 0 ? "Granted" : value == 0 ? "Missing" : "Unknown"
    }
}

struct SettingRow: View {
    let item: SettingItem
    let perGame: Bool
    @State private var value: String
    @State private var useGlobal: Bool
    /// Why the last change didn't stick.
    @State private var refusal: String?

    init(item: SettingItem, perGame: Bool) {
        self.item = item
        self.perGame = perGame
        _value = State(initialValue: item.value())
        _useGlobal = State(initialValue: perGame ? item.usesGlobal : false)
    }

    var body: some View {
        let reason = unavailableReason
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .center, spacing: 10) {
                control
                    .disabled((perGame && useGlobal) || reason != nil)
                    .opacity((perGame && useGlobal) || reason != nil ? 0.55 : 1)
                if perGame && reason == nil {
                    Button {
                        useGlobal.toggle()
                        item.setUsesGlobal(useGlobal)
                        if !useGlobal {
                            item.set(value)
                        }
                        value = item.value()
                        refusal = nil
                    } label: {
                        Image(systemName: useGlobal ? "globe" : "pencil.circle.fill")
                            .foregroundStyle(useGlobal ? Color.secondary : Theme.accent)
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Override for this game")
                    .accessibilityValue(useGlobal ? "Off, uses the global value" : "On")
                    .accessibilityAddTraits(.isToggle)
                }
            }
            if let reason {
                Caption(text: reason, systemImage: "nosign", color: .secondary)
            } else if let refusal {
                Caption(text: refusal, systemImage: "exclamationmark.circle", color: .orange)
            } else if let note = SettingsCatalog.note(for: item, value: value) {
                Caption(text: note, systemImage: "info.circle", color: .secondary)
            }
            if reason == nil, !item.runtimeModifiable, Bridge.emuState.inGame {
                Caption(text: "Applies the next time the game starts.", systemImage: "arrow.clockwise",
                        color: .secondary)
            }
        }
    }

    private var unavailableReason: String? {
        if case .unavailable(let reason) = SettingsCatalog.availability(of: item) {
            return reason
        }
        return nil
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
                    .id(value)
            } else {
                TextRow(label: item.label, value: value, numeric: true, commit: commit)
                    .id(value)
            }
        case .string:
            TextRow(label: item.label, value: value, numeric: false, commit: commit)
                .id(value)
        }
    }

    private func commit(_ newValue: String) {
        if perGame, useGlobal {
            return
        }
        let accepted = item.set(newValue)
        value = item.value()
        if !accepted {
            refusal = perGame ? "This setting can't be changed per game." : "Lichii didn't accept this value."
        } else if !matches(value, newValue) {
            refusal = rejectionText(for: newValue)
        } else {
            refusal = nil
        }
    }

    /// Numbers come back in the core's own format ("1.500000").
    private func matches(_ stored: String, _ requested: String) -> Bool {
        switch item.kind {
        case .int, .float:
            guard let a = Double(stored), let b = Double(requested.trimmingCharacters(in: .whitespaces)) else {
                return stored == requested
            }
            return abs(a - b) < 0.0001
        default:
            return stored == requested
        }
    }

    private func rejectionText(for requested: String) -> String {
        if (item.kind == .int || item.kind == .float), item.ranged, item.maximum > item.minimum {
            let format = { (number: Double) in
                item.kind == .int ? String(Int(number)) : String(format: "%g", number)
            }
            return "\"\(requested)\" isn't allowed: use \(format(item.minimum)) to \(format(item.maximum))."
        }
        return "\"\(requested)\" isn't a valid value; the previous one was kept."
    }
}

private struct Caption: View {
    let text: String
    let systemImage: String
    let color: Color

    var body: some View {
        Label(text, systemImage: systemImage)
            .font(.caption)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
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
            .accessibilityLabel(item.label)
            .accessibilityValue(formatted)
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
    @State private var original: String
    @FocusState private var focused: Bool

    init(label: String, value: String, numeric: Bool, commit: @escaping (String) -> Void) {
        self.label = label
        self.numeric = numeric
        self.commit = commit
        _text = State(initialValue: value)
        _original = State(initialValue: value)
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
                .focused($focused)
                .onSubmit(save)
                .onChange(of: focused) { _, isFocused in
                    // Leaving the field keeps the text, as Return does.
                    if !isFocused { save() }
                }
                .foregroundStyle(.secondary)
        }
    }

    private func save() {
        guard text != original else { return }
        original = text
        commit(text)
    }
}
