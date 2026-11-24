// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

enum SettingsScope: Equatable {
    case global
    case game(Game)
}

/// Top-level destinations in global Settings (pushed on iPhone, split-view sidebar on iPad).
enum SettingsRoute: Hashable {
    case recommended
    case group(SettingsGroup)
    case link(String)
    case about
}

/// Adaptive Settings container for the Settings tab: two-column sidebar + detail on regular-width
/// iPads, push stack on iPhone and compact Split View / Slide Over windows.
struct SettingsTabView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @ObservedObject private var platform = PlatformCenter.shared
    @State private var selectedRoute: SettingsRoute? = .group(.graphics)
    @State private var query = ""
    @State private var columnVisibility: NavigationSplitViewVisibility = .all

    var body: some View {
        let _ = platform.settingsRevision
        if Theme.isPad && horizontalSizeClass == .regular {
            NavigationSplitView(columnVisibility: $columnVisibility) {
                List(selection: $selectedRoute) {
                    GlobalSettingsRoot(splitSidebar: true)
                }
                .navigationTitle("Settings")
                .searchable(text: $query, prompt: UiText.text("search.settings"))
            } detail: {
                NavigationStack {
                    if !query.isEmpty {
                        List {
                            SettingSectionsList(
                                sections: SettingsCatalog.sections(group: .graphics, query: query),
                                perGame: false
                            )
                        }
                        .navigationTitle("Settings")
                    } else {
                        SettingsRouteDetail(route: selectedRoute ?? .group(.graphics))
                            .id(selectedRoute ?? .group(.graphics))
                    }
                }
            }
            .navigationSplitViewStyle(.balanced)
            .onDisappear { lichii_settings_save() }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
                lichii_settings_save()
            }
        } else {
            NavigationStack {
                SettingsView(scope: .global)
            }
        }
    }
}

/// Detail screen for a selected `SettingsRoute`.
private struct SettingsRouteDetail: View {
    let route: SettingsRoute

    var body: some View {
        switch route {
        case .recommended:
            RecommendedSettingsView()
        case .group(let group):
            SettingsGroupView(group: group)
        case .about:
            DiagnosticsView()
        case .link(let id):
            if let entry = AppExtensions.settingsLinks.first(where: { $0.id == id }) {
                entry.view()
            } else {
                SettingsGroupView(group: .graphics)
            }
        }
    }
}

/// Every setting the core saves, generated from the bridge's metadata. The per-game scope lists the
/// settings a game can override, each with a switch back to the global value. Settings that do
/// nothing on iOS are not listed (SettingsCatalog.availability); debug switches and hacks are on
/// the Advanced page.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var platform = PlatformCenter.shared
    let scope: SettingsScope
    @State private var ready = false
    @State private var query = ""
    @State private var showReset = false
    @State private var resetPreview = ""
    @State private var showAdvanced = false
    @State private var generation = 0

    private var game: Game? {
        if case .game(let game) = scope { return game }
        return nil
    }

    var body: some View {
        let _ = platform.settingsRevision
        List {
            if ready {
                if let game {
                    Section {
                        HStack(spacing: 12) {
                            GameIcon(game: game, size: 48)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(game.title).font(.headline)
                                Text("\(UiText.text("game.note")) \(UiText.text("game.note.ios"))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                    SettingSectionsList(sections: SettingsCatalog.sections(perGame: true, query: query,
                                                                           programId: game.programId),
                                        perGame: true)
                        .id(generation)
                    if query.isEmpty {
                        // Inline rather than pushed: the game's settings scope ends when this view
                        // disappears.
                        Section {
                            Toggle(UiText.text("advanced.show"), isOn: $showAdvanced)
                        } footer: {
                            Text(UiText.text("advanced.show.footer"))
                        }
                        if showAdvanced {
                            SettingSectionsList(sections: SettingsCatalog.sections(perGame: true, query: "", advanced: true,
                                                                                   programId: game.programId),
                                                perGame: true)
                                .id(generation)
                        }
                    }
                    Section {
                        Button(UiText.text("reset.button"), role: .destructive) {
                            resetPreview = overrideSummary
                            showReset = true
                        }
                    } footer: {
                        Text(UiText.text("reset.footer"))
                    }
                } else if !query.isEmpty {
                    SettingSectionsList(sections: SettingsCatalog.sections(group: .graphics, query: query),
                                        perGame: false)
                } else {
                    GlobalSettingsRoot(splitSidebar: false)
                }
            } else if game != nil {
                Section {
                    Text("Game settings can't be edited while another game is booting or running.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .navigationTitle(game == nil ? "Settings" : UiText.text("page.game_settings"))
        .searchable(text: $query, prompt: UiText.text("search.settings"))
        .toolbar {
            if game != nil {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .confirmationDialog(UiText.text("reset.question"), isPresented: $showReset,
                            titleVisibility: .visible) {
            Button(UiText.text("reset.confirm"), role: .destructive) {
                lichii_settings_reset_game()
                generation += 1
                PlatformCenter.shared.bumpSettings()
            }
        } message: {
            Text(resetPreview)
        }
        .onAppear(perform: open)
        .onDisappear(perform: close)
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            // Changes survive the app being closed from the app switcher.
            if let game, ready {
                lichii_settings_end_game()
                ready = lichii_settings_begin_game(game.programId)
            } else if game == nil {
                lichii_settings_save()
            }
        }
    }

    /// What the reset changes, one setting per line.
    private var overrideSummary: String {
        let lines = Bridge.string(lichii_settings_reset_game_preview())
            .split(separator: "\n")
            .compactMap { line -> String? in
                let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
                guard parts.count == 3 else { return nil }
                let item = SettingsCatalog.all.first { $0.key == parts[0] }
                let name = { (value: String) -> String in
                    value == "global" ? UiText.text("value.global") : (item?.displayName(of: value) ?? value)
                }
                return UiText.format("change.line", item?.label ?? parts[0], name(parts[1]), name(parts[2]))
            }
        if lines.isEmpty {
            return UiText.text("reset.already")
        }
        return lines.joined(separator: "\n")
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
            Section {
                ForEach(section.items) { item in
                    SettingRow(item: item, perGame: perGame)
                }
            } header: {
                Text(section.title)
            } footer: {
                if let footer = section.footer {
                    Text(footer)
                }
            }
        }
    }
}

/// The global settings' first page: a compact header, the setting pages, then the screens for
/// game data and help. Links other areas add (AppExtensions.settingsLinks) are placed by id.
private struct GlobalSettingsRoot: View {
    let splitSidebar: Bool
    private static let dataLinks = ["keys", "users"]
    private static let helpLinks = ["setup", "logs", "experiments"]

    var body: some View {
        let links = AppExtensions.settingsLinks
        SettingsHeader(splitSidebar: splitSidebar)
        Section {
            routeRow(
                route: .recommended,
                title: UiText.text("page.recommended"),
                systemImage: "wand.and.stars",
                colorId: "recommended"
            ) {
                RecommendedSettingsView()
            }
        } footer: {
            Text("What this device should use, and one button to set it.")
        }
        Section {
            groupLink(.graphics)
            ForEach(links.filter { $0.id == "controls" }) { link(for: $0) }
            groupLink(.audio)
            groupLink(.system)
        }
        Section(UiText.text("group.games_data")) {
            ForEach(links.filter { Self.dataLinks.contains($0.id) }) { link(for: $0) }
        }
        ShaderCacheSettingsSection()
        Section(UiText.text("group.help")) {
            ForEach(links.filter { Self.helpLinks.contains($0.id) }) { link(for: $0) }
            routeRow(
                route: .about,
                title: UiText.text("page.about_device"),
                systemImage: "info.circle.fill",
                colorId: "about"
            ) {
                DiagnosticsView()
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
            Text(UiText.text("advanced.footer"))
        }
    }

    @ViewBuilder
    private func routeRow<Destination: View>(
        route: SettingsRoute,
        title: String,
        systemImage: String,
        colorId: String,
        @ViewBuilder destination: () -> Destination
    ) -> some View {
        let rowLabel = SettingsRowLabel(
            title: title,
            systemImage: systemImage,
            color: Theme.badgeColor(for: colorId)
        )
        if splitSidebar {
            NavigationLink(value: route) { rowLabel }
        } else {
            NavigationLink(destination: destination) { rowLabel }
        }
    }

    private func groupLink(_ group: SettingsGroup) -> some View {
        routeRow(
            route: .group(group),
            title: group.title,
            systemImage: group.systemImage,
            colorId: group.rawValue
        ) {
            SettingsGroupView(group: group)
        }
    }

    private func link(for link: SettingsLink) -> some View {
        routeRow(
            route: .link(link.id),
            title: link.title,
            systemImage: link.systemImage,
            colorId: link.id
        ) {
            link.view()
        }
    }
}

/// One page of the global settings (Graphics, Audio, System, Advanced).
private struct SettingsGroupView: View {
    @ObservedObject private var platform = PlatformCenter.shared
    let group: SettingsGroup

    var body: some View {
        let _ = platform.settingsRevision
        List {
            if group == .advanced {
                Section {
                } footer: {
                    Label(UiText.text("advanced.warning"),
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

/// App version, device summary and JIT state; opens the device details.
private struct SettingsHeader: View {
    /// Follows LICHII_EVENT_JIT_CHANGED.
    @ObservedObject private var app = AppController.shared
    let splitSidebar: Bool

    private var cleanVersion: String {
        let raw = Bridge.buildVersion
        if raw.hasPrefix("Lichii | ") {
            return String(raw.dropFirst("Lichii | ".count))
        }
        return raw
    }

    private var deviceSummary: String? {
        guard let defaults = PlatformCenter.deviceDefaults else { return nil }
        let tier = PlatformCenter.text(defaults.tier_name)
        if let info = PlatformCenter.deviceInfo {
            let gpu = PlatformCenter.text(info.gpu_name).replacingOccurrences(of: " GPU", with: "")
            if !gpu.isEmpty, !tier.isEmpty {
                return "\(gpu) · \(tier) Tier"
            }
        }
        return tier.isEmpty ? nil : "\(tier) Tier"
    }

    private var content: some View {
        HStack(spacing: 12) {
            Image(systemName: "gamecontroller.fill")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 46, height: 46)
                .background(
                    LinearGradient(
                        colors: [
                            Color(red: 0.63, green: 0.44, blue: 0.99),
                            Color(red: 0.45, green: 0.27, blue: 0.90)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    in: RoundedRectangle(cornerRadius: 12, style: .continuous)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.22), lineWidth: 0.75)
                )
                .shadow(color: Theme.accent.opacity(0.28), radius: 5, y: 2)
            VStack(alignment: .leading, spacing: 2) {
                Text("Lichii")
                    .font(.headline)
                    .foregroundStyle(.primary)
                Text(cleanVersion)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let deviceSummary {
                    Text(deviceSummary)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
            JitPill(status: app.jit)
        }
        .padding(.vertical, 3)
    }

    var body: some View {
        Section {
            if splitSidebar {
                NavigationLink(value: SettingsRoute.about) {
                    content
                }
                .accessibilityElement(children: .combine)
            } else {
                NavigationLink {
                    DiagnosticsView()
                } label: {
                    content
                }
                .accessibilityElement(children: .combine)
            }
        }
    }
}

/// Compact horizontal pill showing JIT state. Uses explicit HStack instead of Label so SwiftUI
/// List never overrides its label style into an icon-only vertical cell.
struct JitPill: View {
    let status: JitStatus

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: status.symbol)
                .font(.system(size: 11, weight: .bold))
            Text(text)
                .lineLimit(1)
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(status.color)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(status.color.opacity(0.15), in: Capsule())
        .overlay(Capsule().strokeBorder(status.color.opacity(0.25), lineWidth: 0.75))
        .fixedSize(horizontal: true, vertical: true)
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

    private var osLabel: String {
        guard let info else {
            return "\(Theme.isPad ? "iPadOS" : "iOS") \(UIDevice.current.systemVersion)"
        }
        return "\(info.is_ipad != 0 ? "iPadOS" : "iOS") \(info.os_major).\(info.os_minor)"
    }

    private var modelSummary: String {
        guard let info else { return "\(UIDevice.current.model), \(osLabel)" }
        let machine = PlatformCenter.text(info.machine)
        let friendly = RecommendedSettings.load().device
        if !friendly.isEmpty, !friendly.hasPrefix(machine) {
            return "\(friendly) (\(machine)), \(osLabel)"
        }
        return "\(machine), \(osLabel)"
    }

    var body: some View {
        List {
            Section {
                LabeledContent("Version", value: Bridge.buildVersion)
                LabeledContent("JIT", value: jitText)
            }
            if let info {
                let memTitle = UiText.text("experiments.memory.title")
                Section {
                    LabeledContent(memTitle, value: "\(gigabytes(info.launch_available_memory_mib)) available to games")
                    LabeledContent("Installed memory", value: gigabytes(info.physical_memory_mib))
                    LabeledContent("Increased Memory Limit", value: entitlement(info.increased_memory_limit))
                    LabeledContent("Extended Virtual Addressing", value: entitlement(info.extended_virtual_addressing))
                } header: {
                    Text(memTitle)
                } footer: {
                    Text("Games need both entitlements. Without them \(info.is_ipad != 0 ? "iPadOS" : "iOS") gives Lichii too little memory to start a game.")
                }
                Section(UiText.text("section.device")) {
                    LabeledContent("Model", value: modelSummary)
                    LabeledContent("GPU", value: PlatformCenter.text(info.gpu_name))
                    if let defaults {
                        LabeledContent("Performance Tier", value: PlatformCenter.text(defaults.tier_name))
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
                Section(UiText.text("section.device")) {
                    DeviceInfoRows()
                }
            }
        }
        .navigationTitle(UiText.text("page.about_device"))
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
    @ObservedObject private var platform = PlatformCenter.shared
    let item: SettingItem
    let perGame: Bool
    @State private var value: String
    @State private var useGlobal: Bool
    /// Why the last change didn't stick.
    @State private var refusal: String?
    @State private var resetToken = 0

    init(item: SettingItem, perGame: Bool) {
        self.item = item
        self.perGame = perGame
        _value = State(initialValue: item.value())
        _useGlobal = State(initialValue: perGame ? item.usesGlobal : false)
    }

    var body: some View {
        let _ = platform.settingsRevision
        let disabledReason = SettingsCatalog.disabledReason(for: item)
        let controlDisabled = (perGame && useGlobal) || disabledReason != nil
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .center, spacing: 10) {
                control
                    .disabled(controlDisabled)
                    .opacity(controlDisabled ? 0.55 : 1)
                if perGame {
                    Button {
                        useGlobal.toggle()
                        item.setUsesGlobal(useGlobal)
                        if !useGlobal {
                            item.set(value)
                        }
                        value = item.value()
                        refusal = nil
                        PlatformCenter.shared.bumpSettings()
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
            if let disabledReason, !disabledReason.isEmpty {
                Caption(text: disabledReason, systemImage: "info.circle", color: .secondary)
            }
            if let interaction = SettingsCatalog.interactionNote(for: item, value: value) {
                Caption(text: interaction.text,
                        systemImage: interaction.highlight ? "exclamationmark.circle" : "info.circle",
                        color: interaction.highlight ? .orange : .secondary)
            }
            if let refusal {
                Caption(text: refusal, systemImage: "exclamationmark.circle", color: .orange)
            } else if let note = SettingsCatalog.note(for: item, value: value) {
                Caption(text: note, systemImage: "info.circle", color: .secondary)
            }
            if !item.runtimeModifiable, Bridge.emuState.inGame {
                Caption(text: UiText.text("note.applies_next_boot"), systemImage: "arrow.clockwise",
                        color: .secondary)
            }
        }
        .onChange(of: platform.settingsRevision) { _, _ in
            let latest = item.value()
            if value != latest {
                value = latest
            }
            if perGame {
                let latestGlobal = item.usesGlobal
                if useGlobal != latestGlobal {
                    useGlobal = latestGlobal
                }
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
                    .id("\(value)#\(resetToken)")
            } else {
                TextRow(label: item.label, value: value, numeric: true, commit: commit)
                    .id("\(value)#\(resetToken)")
            }
        case .string:
            TextRow(label: item.label, value: value, numeric: false, commit: commit)
                .id("\(value)#\(resetToken)")
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
            resetToken += 1
        } else if !matches(value, newValue) {
            refusal = rejectionText(for: newValue)
            resetToken += 1
        } else {
            refusal = nil
        }
        if !perGame {
            lichii_settings_save()
        }
        PlatformCenter.shared.bumpSettings()
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
