// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  LayoutOptionsView.swift
//  Melo-Controller
//
//  Created by Stossy11 on 26/1/2026.
//

import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    static var meloLayout: UTType { LayoutExporter.meloLayout }
}

struct LayoutOptionsView: View {
    let gameId: String?
    var gameName: String? = nil
    @Binding var layout: LayoutConfig
    @Environment(\.presentationMode) var presentationMode

    @State private var showingResetAlert = false
    @State private var showingCopySheet = false
    @State private var showingFileImporter = false

    @State private var exportItem: LayoutExportItem?

    @State private var alertTitle = ""
    @State private var alertMessage = ""
    @State private var showingAlert = false

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if let gameId = gameId {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Current Game")
                            .font(.headline)
                        HStack {
                            Image(systemName: "gamecontroller.fill")
                                .foregroundColor(.blue)
                            Text(gameName ?? gameId)
                                .font(.subheadline)
                            Spacer()
                            if LayoutManager.shared.hasCustomLayout(for: gameId) {
                                HStack {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                    Text("Custom Layout")
                                        .font(.caption)
                                        .foregroundColor(.green)
                                }
                            } else {
                                Text("Using Default")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                    }
                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Layout Actions")
                        .font(.headline)

                    Button(action: { showingCopySheet = true }) {
                        HStack {
                            Image(systemName: "doc.on.doc")
                            Text("Copy Layout From…")
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .foregroundColor(.blue)
                        .cornerRadius(8)
                    }

                    Button(action: exportCurrentLayout) {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            Text("Export Layout…")
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .padding()
                        .background(Color.purple.opacity(0.1))
                        .foregroundColor(.purple)
                        .cornerRadius(8)
                    }
                    .sheet(item: $exportItem) { item in
                        ShareSheet(activityItems: [item.url])
                    }

                    Button(action: { showingFileImporter = true }) {
                        HStack {
                            Image(systemName: "square.and.arrow.down")
                            Text("Import Layout…")
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .padding()
                        .background(Color.teal.opacity(0.1))
                        .foregroundColor(.teal)
                        .cornerRadius(8)
                    }
                    .fileImporter(
                        isPresented: $showingFileImporter,
                        allowedContentTypes: [.meloLayout, .json],
                        allowsMultipleSelection: false
                    ) { result in
                        handleImport(result: result)
                    }

                    Button(action: {
                        layout = LayoutManager.shared.load(for: nil)
                    }) {
                        HStack {
                            Image(systemName: "arrow.clockwise")
                            Text("Reset to Default Layout")
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .padding()
                        .background(Color.orange.opacity(0.1))
                        .foregroundColor(.orange)
                        .cornerRadius(8)
                    }

                    Button(action: { showingResetAlert = true }) {
                        HStack {
                            Image(systemName: "trash")
                            Text("Delete Custom Layout")
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .foregroundColor(.red)
                        .cornerRadius(8)
                    }
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Layout Options")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(trailing: Button("Done") {
                presentationMode.wrappedValue.dismiss()
            })
        }
        .alert("Delete Custom Layout", isPresented: $showingResetAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                LayoutManager.shared.reset(for: gameId)
                layout = LayoutManager.shared.load(for: gameId)
            }
        } message: {
            Text("This will delete the custom layout for this game and revert to the default layout.")
        }
        .alert(alertTitle, isPresented: $showingAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
        .sheet(isPresented: $showingCopySheet) {
            CopyLayoutView(targetGameId: gameId, layout: $layout)
        }
    }

    private func exportCurrentLayout() {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(layout)

            let fileName = LayoutManager.shared.exportFileName(for: gameId)
            let tmpURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try data.write(to: tmpURL, options: .atomic)
            exportItem = LayoutExportItem(url: tmpURL)
        } catch {
            alertTitle = "Export Failed"
            alertMessage = error.localizedDescription
            showingAlert = true
        }
    }

    private func handleImport(result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            alertTitle = "Import Failed"
            alertMessage = error.localizedDescription
            showingAlert = true

        case .success(let urls):
            guard let url = urls.first else { return }

            let accessing = url.startAccessingSecurityScopedResource()
            defer {
                if accessing { url.stopAccessingSecurityScopedResource() }
            }

            do {
                let data = try Data(contentsOf: url)
                let imported = try LayoutManager.shared.importLayout(from: data, for: gameId)
                layout = imported
                alertTitle = "Import Successful"
                alertMessage = "The layout has been applied\(gameId.map { " to \($0)" } ?? "")."
                showingAlert = true
            } catch {
                alertTitle = "Import Failed"
                alertMessage = error.localizedDescription
                showingAlert = true
            }
        }
    }
}

private struct LayoutExportItem: Identifiable {
    let id = UUID()
    let url: URL
}

private struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
