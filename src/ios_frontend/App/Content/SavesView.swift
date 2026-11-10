// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Save data of one game for the current user: export as .zip, import, delete.
struct SavesView: View {
    let game: Game
    @Environment(\.dismiss) private var dismiss
    @State private var info: ContentBridge.SaveInfo?
    @State private var share: ShareItem?
    @State private var showImporter = false
    @State private var pendingImport: URL?
    @State private var confirmDelete = false
    @State private var alert: ContentAlert?
    @State private var busy = false
    @State private var profile = ""

    var body: some View {
        List {
            Section {
                LabeledContent("User", value: profile.isEmpty ? "-" : profile)
                if let info {
                    LabeledContent("Save data", value: info.exists ? "\(info.files) files, \(info.size.byteCount)" : "None yet")
                    Text(info.path)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            } footer: {
                Text("Saves belong to the current user (Settings > Users). Exports are .zip files you can keep in Files or move to another device.")
            }
            Section {
                Button {
                    exportSave()
                } label: {
                    Label("Export Save...", systemImage: "square.and.arrow.up")
                }
                .disabled(info?.exists != true || busy)
                Button {
                    showImporter = true
                } label: {
                    Label("Import Save...", systemImage: "square.and.arrow.down")
                }
                .disabled(busy)
            }
            Section {
                Button("Delete Save Data", role: .destructive) { confirmDelete = true }
                    .disabled(info?.exists != true || busy)
            }
        }
        .navigationTitle("Save Data")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
        }
        .onAppear(perform: reload)
        .sheet(item: $share) { item in
            ShareSheet(items: [item.url])
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.zip]) { result in
            if case .success(let url) = result {
                pendingImport = url
            }
        }
        .confirmationDialog("Replace the current save with this archive?",
                            isPresented: Binding(get: { pendingImport != nil }, set: { if !$0 { pendingImport = nil } }),
                            titleVisibility: .visible) {
            Button("Import", role: .destructive) {
                if let url = pendingImport { importSave(url) }
                pendingImport = nil
            }
        } message: {
            Text("Export the current save first if you want to keep it.")
        }
        .confirmationDialog("Delete the save data of \(game.title)?", isPresented: $confirmDelete,
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                if !game.path.withCString({ lichii_content_delete_save(game.programId, $0) }) {
                    alert = ContentAlert(title: "Save Data", message: "Stop the game first.")
                }
                reload()
            }
        }
        .alert(item: $alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
    }

    private func reload() {
        let current = game
        ContentBridge.run({
            (ContentBridge.saveInfo(current), ContentBridge.profiles.first(where: \.current)?.name ?? "")
        }) { result in
            info = result.0
            profile = result.1
        }
    }

    private func exportSave() {
        busy = true
        let current = game
        let name = current.title.replacingOccurrences(of: "/", with: "-")
        let url = ContentBridge.exportURL("\(name) [\(current.programIdHex)] save.zip")
        ContentBridge.run({ () -> (Bool, String) in
            var raw: UnsafeMutablePointer<CChar>?
            let ok = current.path.withCString { game in
                url.path.withCString { zip in
                    lichii_content_export_save(current.programId, game, zip, &raw)
                }
            }
            return (ok, ContentBridge.string(raw) ?? "")
        }) { ok, message in
            busy = false
            if ok {
                share = ShareItem(url: url)
            } else {
                alert = ContentAlert(title: "Export failed", message: message)
            }
        }
    }

    private func importSave(_ url: URL) {
        busy = true
        let current = game
        ContentBridge.run({ () -> (Bool, String) in
            // A coordinated read, so an archive in iCloud Drive or another file provider is
            // downloaded first.
            let (result, error) = coordinatedRead(url) { readable -> (Bool, String) in
                var raw: UnsafeMutablePointer<CChar>?
                let ok = current.path.withCString { game in
                    readable.path.withCString { zip in
                        lichii_content_import_save(current.programId, game, zip, &raw)
                    }
                }
                return (ok, ContentBridge.string(raw) ?? "")
            }
            return result ?? (false, error ?? "")
        }) { ok, message in
            busy = false
            reload()
            alert = ContentAlert(title: ok ? "Save imported" : "Import failed", message: message)
        }
    }
}

struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

/// UIActivityViewController for files (Save to Files, AirDrop, Mail, ...).
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
