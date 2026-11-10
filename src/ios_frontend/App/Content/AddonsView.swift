// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI

/// Per-game updates, DLC and mods (with the built-in patches) in one screen.
struct AddonsView: View {
    let game: Game

    enum AddonsTab: String, CaseIterable, Identifiable {
        case updates = "Updates", dlc = "DLC", mods = "Mods"
        var id: String { rawValue }
    }

    @Environment(\.dismiss) private var dismiss
    @State private var tab: AddonsTab

    init(game: Game, initialTab: AddonsTab = .updates) {
        self.game = game
        _tab = State(initialValue: initialTab)
    }
    @State private var addons: [ContentBridge.Addon] = []
    @State private var dlc: [ContentBridge.DLC] = []
    @State private var importer: AddonsTab?
    @State private var progress: Double?
    @State private var busy = false
    @State private var alert: ContentAlert?
    @State private var removal: Removal?
    @State private var cancelFlag = CancelFlag()

    private enum Removal: Identifiable {
        case update, dlc(String), allDLC, mod(String)
        var id: String {
            switch self {
            case .update: return "update"
            case .dlc(let id): return "dlc/\(id)"
            case .allDLC: return "dlc"
            case .mod(let name): return "mod/\(name)"
            }
        }
    }

    var body: some View {
        List {
            Section {
                Picker("Content", selection: $tab) {
                    ForEach(AddonsTab.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .listRowBackground(Color.clear)
            }
            if let progress {
                Section {
                    ProgressView("Installing...", value: progress)
                    Button("Cancel", role: .cancel) { cancelFlag.set(true) }
                }
            }
            switch tab {
            case .updates: updates
            case .dlc: dlcSection
            case .mods: mods
            }
        }
        .navigationTitle(game.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    importer = tab
                } label: {
                    Image(systemName: "plus")
                }
                .disabled(busy)
            }
        }
        .onAppear(perform: reload)
        .refreshable { reload() }
        .fileImporter(isPresented: Binding(get: { importer != nil }, set: { if !$0 { importer = nil } }),
                      allowedContentTypes: importer == .mods ? ContentTypes.mods : ContentTypes.packages,
                      allowsMultipleSelection: true) { result in
            let target = importer
            importer = nil
            guard case .success(let urls) = result, !urls.isEmpty else { return }
            if target == .mods {
                installMods(urls)
            } else {
                installPackages(urls)
            }
        }
        .alert(item: $alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
        .confirmationDialog(removalTitle, isPresented: Binding(get: { removal != nil },
                                                              set: { if !$0 { removal = nil } }),
                            titleVisibility: .visible) {
            Button("Remove", role: .destructive) {
                if let removal { remove(removal) }
                removal = nil
            }
        }
    }

    // MARK: Sections

    private var updates: some View {
        let items = addons.filter(\.isUpdate)
        return Section {
            if items.isEmpty {
                Text("No update installed.").foregroundStyle(.secondary)
            }
            ForEach(items) { item in
                Toggle(isOn: binding(for: item)) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.version.isEmpty ? item.name : "Version \(item.version)")
                        Text(sourceLabel(item.source)).font(.caption).foregroundStyle(.secondary)
                    }
                }
                .swipeActions {
                    if item.removable {
                        Button("Remove", role: .destructive) { removal = .update }
                    }
                }
            }
        } header: {
            Text("Updates")
        } footer: {
            Text("Only one update applies at a time. Tap + to install an update (.nsp/.nsz); swipe to remove one installed to the NAND.")
        }
    }

    private var dlcSection: some View {
        let toggle = addons.first(where: \.isDLC)
        return Group {
            Section {
                if let toggle {
                    Toggle("Use DLC", isOn: binding(for: toggle))
                }
                if dlc.isEmpty {
                    Text("No DLC installed.").foregroundStyle(.secondary)
                }
                ForEach(dlc) { item in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("DLC \(item.index)")
                            Text("\(item.titleId) · \(sourceLabel(item.source))")
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        if item.removable {
                            Button("Remove", role: .destructive) { removal = .dlc(item.titleId) }
                        }
                    }
                }
            } header: {
                Text("DLC")
            } footer: {
                Text("Tap + to install DLC packages (.nsp/.nsz). Swipe to remove one.")
            }
            if dlc.contains(where: \.removable) {
                Section {
                    Button("Remove All DLC", role: .destructive) { removal = .allDLC }
                }
            }
        }
    }

    private var mods: some View {
        let items = addons.filter(\.isMod)
        let builtins = items.filter { $0.type == "builtin" }
        let user = items.filter { $0.type != "builtin" }
        return Group {
            Section {
                if user.isEmpty {
                    Text("No mods installed.").foregroundStyle(.secondary)
                }
                ForEach(user) { item in
                    Toggle(isOn: binding(for: item)) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.name)
                            Text(item.version).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        if item.removable {
                            Button("Delete", role: .destructive) { removal = .mod(item.name) }
                        }
                    }
                }
            } header: {
                Text("Mods")
            } footer: {
                Text("Tap + to add a mod folder or .zip containing exefs/, romfs/ or cheats/. Mods live in Documents/eden/load/\(game.programIdHex)/<mod name>/ (also reachable from the Files app). Changes apply on the next boot.")
            }
            if !builtins.isEmpty {
                Section {
                    ForEach(builtins) { item in
                        Toggle(item.name.replacingOccurrences(of: " (built-in)", with: ""),
                               isOn: binding(for: item))
                    }
                } header: {
                    Text("Built-in patches")
                } footer: {
                    Text("Patches that ship with Lichii for this game.")
                }
            }
        }
    }

    // MARK: Actions

    private func reload() {
        let current = game
        ContentBridge.run({
            (ContentBridge.addons(current), ContentBridge.dlc(current))
        }) { result in
            addons = result.0
            dlc = result.1
        }
    }

    private func binding(for item: ContentBridge.Addon) -> Binding<Bool> {
        Binding {
            addons.first { $0.id == item.id }?.enabled ?? item.enabled
        } set: { value in
            _ = item.toggleKey.withCString {
                lichii_content_set_addon_enabled(game.programId, $0, value)
            }
            reload()
        }
    }

    private func installPackages(_ urls: [URL]) {
        busy = true
        progress = 0
        let flag = CancelFlag()
        cancelFlag = flag
        let current = game
        let box = ContentBridge.Progress { value in
            DispatchQueue.main.async {
                MainActor.assumeIsolated { progress = value }
            }
            return flag.isCancelled
        }
        ContentBridge.run({
            urls.map { url in
                (url.lastPathComponent,
                 withSecurityScope(url) { ContentBridge.installPackage(url, for: current, progress: box) })
            }
        }) { results in
            busy = false
            progress = nil
            reload()
            let lines = results.map { "\($0.0): \(ContentBridge.describe($0.1))" }
            alert = ContentAlert(title: "Install", message: lines.joined(separator: "\n"))
        }
    }

    private func installMods(_ urls: [URL]) {
        busy = true
        let current = game
        ContentBridge.run({
            urls.map { url -> (String, LichiiModResult, Int32) in
                var count: Int32 = 0
                let result = withSecurityScope(url) {
                    url.path.withCString { lichii_content_install_mod(current.programId, $0, nil, &count) }
                }
                return (url.lastPathComponent, result, count)
            }
        }) { results in
            busy = false
            reload()
            let lines = results.map { name, result, count -> String in
                switch result {
                case LICHII_MOD_OK: return "\(name): \(count) mod\(count == 1 ? "" : "s") installed"
                case LICHII_MOD_NO_MOD: return "\(name): no exefs, romfs or cheats folder inside"
                case LICHII_MOD_BAD_ZIP: return "\(name): the archive could not be read"
                case LICHII_MOD_NOT_FOUND: return "\(name): not found"
                default: return "\(name): copying failed"
                }
            }
            alert = ContentAlert(title: "Mods", message: lines.joined(separator: "\n"))
        }
    }

    private var removalTitle: String {
        switch removal {
        case .update: return "Remove the installed update?"
        case .dlc: return "Remove this DLC?"
        case .allDLC: return "Remove all DLC of this game?"
        case .mod(let name): return "Delete the mod \(name)?"
        case nil: return ""
        }
    }

    private func remove(_ removal: Removal) {
        switch removal {
        case .update:
            _ = lichii_content_remove_update(game.programId)
        case .dlc(let titleId):
            _ = lichii_content_remove_dlc(UInt64(titleId, radix: 16) ?? 0)
        case .allDLC:
            _ = lichii_content_remove_all_dlc(game.programId)
        case .mod(let name):
            _ = name.withCString { lichii_content_remove_mod(game.programId, $0) }
        }
        reload()
    }

    private func sourceLabel(_ source: String) -> String {
        switch source {
        case "nand": return "Installed (NAND)"
        case "sdmc": return "SD card"
        case "external": return "Next to the game file"
        case "packed": return "Packed in the game file"
        default: return "Installed"
        }
    }
}
