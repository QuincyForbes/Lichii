// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI

/// The emulator's logs (eden_log.txt, the previous run's .old, lichii-startup.txt, console.txt,
/// crash dumps) with a viewer and a share sheet for one file or everything as a .zip.
struct LogsView: View {
    @State private var logs: [ContentBridge.LogFile] = []
    @State private var share: ShareItem?
    @State private var busy = false
    @State private var alert: ContentAlert?

    var body: some View {
        List {
            Section {
                Button {
                    exportAll()
                } label: {
                    HStack {
                        Label("Share All Logs (.zip)", systemImage: "square.and.arrow.up")
                        if busy { Spacer(); ProgressView() }
                    }
                }
                .disabled(busy || logs.isEmpty)
            } footer: {
                Text("Attach the archive to bug reports. It holds the logs and your config files, no keys or games.")
            }
            Section("Files") {
                if logs.isEmpty {
                    Text("No logs yet.").foregroundStyle(.secondary)
                }
                ForEach(logs) { log in
                    NavigationLink {
                        LogViewer(log: log)
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(log.name)
                            Text("\(log.size.byteCount) · \(log.date.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        Button {
                            share = ShareItem(url: URL(fileURLWithPath: log.path))
                        } label: {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        .tint(Theme.accent)
                    }
                }
            }
        }
        .navigationTitle("Logs")
        .onAppear { logs = ContentBridge.logs }
        .refreshable { logs = ContentBridge.logs }
        .sheet(item: $share) { item in
            ShareSheet(items: [item.url])
        }
        .alert(item: $alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
    }

    private func exportAll() {
        busy = true
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = formatter.string(from: Date())
        let url = ContentBridge.exportURL("lichii-logs-\(stamp).zip")
        ContentBridge.run({ () -> (Bool, String) in
            var raw: UnsafeMutablePointer<CChar>?
            let ok = url.path.withCString { lichii_content_export_logs($0, &raw) }
            return (ok, ContentBridge.string(raw) ?? "")
        }) { ok, message in
            busy = false
            if ok {
                share = ShareItem(url: url)
            } else {
                alert = ContentAlert(title: "Logs", message: message)
            }
        }
    }
}

struct LogViewer: View {
    let log: ContentBridge.LogFile
    @State private var lines: [String] = []
    @State private var query = ""
    @State private var share: ShareItem?
    @State private var loading = true
    @State private var issuesOnly = false

    /// The viewer reads the tail; the share sheet always sends the whole file.
    private static let maxBytes: UInt64 = 2 << 20

    var body: some View {
        let shown = filtered
        ScrollViewReader { proxy in
            ScrollView([.vertical, .horizontal]) {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(shown.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(color(for: line))
                            .textSelection(.enabled)
                            .fixedSize()
                            .id(index)
                    }
                }
                .padding(8)
            }
            .overlay {
                if loading { ProgressView() }
            }
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    Menu {
                        Toggle("Warnings and errors only", isOn: $issuesOnly)
                        Button {
                            withAnimation { proxy.scrollTo(0, anchor: .top) }
                        } label: {
                            Label("Top", systemImage: "arrow.up.to.line")
                        }
                        Button {
                            withAnimation { proxy.scrollTo(max(0, shown.count - 1), anchor: .bottom) }
                        } label: {
                            Label("Bottom", systemImage: "arrow.down.to.line")
                        }
                        Button {
                            load()
                        } label: {
                            Label("Reload", systemImage: "arrow.clockwise")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                    Button {
                        share = ShareItem(url: URL(fileURLWithPath: log.path))
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
            .onChange(of: loading) { _, isLoading in
                if !isLoading {
                    proxy.scrollTo(max(0, filtered.count - 1), anchor: .bottom)
                }
            }
        }
        .searchable(text: $query, prompt: "Filter")
        .navigationTitle(log.name)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: load)
        .sheet(item: $share) { item in
            ShareSheet(items: [item.url])
        }
    }

    private var filtered: [String] {
        lines.filter { line in
            (!issuesOnly || isIssue(line)) && (query.isEmpty || line.localizedCaseInsensitiveContains(query))
        }
    }

    private func isIssue(_ line: String) -> Bool {
        line.contains("<Warning>") || line.contains("<Error>") || line.contains("<Critical>")
    }

    private func color(for line: String) -> Color {
        if line.contains("<Critical>") || line.contains("<Error>") { return .red }
        if line.contains("<Warning>") { return .orange }
        if line.contains("<Debug>") || line.contains("<Trace>") { return .secondary }
        return .primary
    }

    private func load() {
        loading = true
        let path = log.path
        let isDump = !path.hasSuffix(".txt") && !path.hasSuffix(".old")
        let maxBytes = Self.maxBytes
        ContentBridge.run({ () -> [String] in
            if isDump {
                return ["Binary file: use Share to send it."]
            }
            let text = path.withCString { ContentBridge.string(lichii_content_read_log($0, maxBytes)) } ?? ""
            return text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        }) { result in
            lines = result
            loading = false
        }
    }
}
