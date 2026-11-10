// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI

/// Per-game cheats: every cheat file (in a mod's cheats/ folder or a load/<id>/cheat_*.txt) with a
/// switch for the file and one per cheat.
struct CheatsView: View {
    let game: Game
    @Environment(\.dismiss) private var dismiss
    @State private var files: [ContentBridge.CheatFile] = []
    @State private var showImporter = false
    @State private var alert: ContentAlert?
    @State private var query = ""
    @State private var busy = false

    var body: some View {
        List {
            if files.isEmpty && !busy {
                Section {
                    Text("No cheats for this game yet.").foregroundStyle(.secondary)
                } footer: {
                    Text("Tap + to add an Atmosphere cheat file (.txt with [Name] and code lines). Cheats can also come with a mod (load/\(game.programIdHex)/<mod>/cheats/<build id>.txt).")
                }
            }
            ForEach(files) { file in
                Section {
                    Toggle(isOn: sourceBinding(file)) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(file.source).font(.headline)
                            Text(file.file).font(.caption.monospaced()).foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        if file.source.hasPrefix("cheat_") {
                            Button("Delete", role: .destructive) { delete(file) }
                        }
                    }
                    // Names can repeat inside one file, so rows are told apart by position.
                    ForEach(Array(filtered(file.cheats).enumerated()), id: \.offset) { _, cheat in
                        Toggle(isOn: cheatBinding(file, cheat)) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(cheat.name)
                                Text("\(cheat.opcodes) opcode\(cheat.opcodes == 1 ? "" : "s")")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .disabled(!file.sourceEnabled)
                    }
                    if file.cheats.isEmpty {
                        Text("No named cheats in this file.").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            if !files.isEmpty {
                Section {
                } footer: {
                    Text("Cheats apply when the game boots. A cheat file only runs for the game build its name (the build id) matches.")
                }
            }
        }
        .overlay {
            if busy && files.isEmpty { ProgressView() }
        }
        .searchable(text: $query, prompt: "Search cheats")
        .navigationTitle("Cheats")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showImporter = true
                } label: {
                    Label("Add Cheat Files", systemImage: "plus")
                }
                .disabled(busy)
            }
        }
        .onAppear(perform: reload)
        .fileImporter(isPresented: $showImporter, allowedContentTypes: ContentTypes.cheats,
                      allowsMultipleSelection: true) { result in
            guard case .success(let urls) = result else { return }
            install(urls)
        }
        .alert(item: $alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
    }

    private func filtered(_ cheats: [ContentBridge.Cheat]) -> [ContentBridge.Cheat] {
        guard !query.isEmpty else { return cheats }
        return cheats.filter { $0.name.localizedCaseInsensitiveContains(query) }
    }

    private func reload() {
        busy = true
        let current = game
        ContentBridge.run({ ContentBridge.cheats(current) }) { result in
            files = result
            busy = false
        }
    }

    private func install(_ urls: [URL]) {
        busy = true
        let programId = game.programId
        ContentBridge.run({ () -> [String] in
            // Coordinated reads, so files in iCloud Drive or another file provider are downloaded.
            urls.compactMap { url -> String? in
                let (ok, error) = coordinatedRead(url) { readable in
                    readable.path.withCString { lichii_content_install_cheat_file(programId, $0) }
                }
                if ok == true { return nil }
                return error.map { "\(url.lastPathComponent) (\($0))" } ?? url.lastPathComponent
            }
        }) { failed in
            if !failed.isEmpty {
                alert = ContentAlert(title: "Cheats",
                                     message: "No cheats could be read from: \(failed.joined(separator: ", "))")
            }
            reload()
        }
    }

    private func sourceBinding(_ file: ContentBridge.CheatFile) -> Binding<Bool> {
        Binding {
            file.sourceEnabled
        } set: { value in
            _ = file.source.withCString { lichii_content_set_addon_enabled(game.programId, $0, value) }
            reload()
        }
    }

    private func cheatBinding(_ file: ContentBridge.CheatFile, _ cheat: ContentBridge.Cheat) -> Binding<Bool> {
        Binding {
            cheat.enabled
        } set: { value in
            _ = file.source.withCString { source in
                cheat.name.withCString { name in
                    lichii_content_set_cheat_enabled(game.programId, source, name, value)
                }
            }
            reload()
        }
    }

    private func delete(_ file: ContentBridge.CheatFile) {
        _ = file.source.withCString { lichii_content_remove_mod(game.programId, $0) }
        reload()
    }
}
