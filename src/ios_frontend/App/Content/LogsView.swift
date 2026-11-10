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
        .onAppear { ContentBridge.run({ ContentBridge.logs }) { logs = $0 } }
        .refreshable { ContentBridge.run({ ContentBridge.logs }) { logs = $0 } }
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

/// One line of a log with its level, worked out once when the file is read.
private struct LogLine {
    enum Level { case error, warning, debug, info }

    let text: String
    let level: Level

    init(_ text: String) {
        self.text = text
        if text.contains("<Critical>") || text.contains("<Error>") {
            level = .error
        } else if text.contains("<Warning>") {
            level = .warning
        } else if text.contains("<Debug>") || text.contains("<Trace>") {
            level = .debug
        } else {
            level = .info
        }
    }

    var isIssue: Bool { level == .error || level == .warning }

    var color: Color {
        switch level {
        case .error: return .red
        case .warning: return .orange
        case .debug: return .secondary
        case .info: return .primary
        }
    }
}

struct LogViewer: View {
    let log: ContentBridge.LogFile
    @State private var lines: [LogLine] = []
    @State private var shown: [LogLine] = []
    @State private var query = ""
    @State private var share: ShareItem?
    @State private var loading = true
    @State private var issuesOnly = false

    /// The viewer reads the tail; the share sheet always sends the whole file.
    private static let maxBytes: UInt64 = 2 << 20

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView([.vertical, .horizontal]) {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(shown.indices, id: \.self) { index in
                        Text(shown[index].text)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(shown[index].color)
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
                        Label("Options", systemImage: "ellipsis.circle")
                    }
                    Button {
                        share = ShareItem(url: URL(fileURLWithPath: log.path))
                    } label: {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                }
            }
            .onChange(of: loading) { _, isLoading in
                if !isLoading {
                    proxy.scrollTo(max(0, shown.count - 1), anchor: .bottom)
                }
            }
        }
        .searchable(text: $query, prompt: "Filter")
        .navigationTitle(log.name)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: load)
        .task(id: FilterKey(query: query, issuesOnly: issuesOnly, count: lines.count)) {
            // Typing waits for a pause; the filter runs off the main thread.
            if !query.isEmpty {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard !Task.isCancelled else { return }
            }
            shown = await Self.filter(lines, query: query, issuesOnly: issuesOnly)
        }
        .sheet(item: $share) { item in
            ShareSheet(items: [item.url])
        }
    }

    private struct FilterKey: Equatable {
        let query: String
        let issuesOnly: Bool
        let count: Int
    }

    private static func filter(_ lines: [LogLine], query: String, issuesOnly: Bool) async -> [LogLine] {
        if query.isEmpty && !issuesOnly {
            return lines
        }
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: lines.filter { line in
                    (!issuesOnly || line.isIssue) &&
                        (query.isEmpty || line.text.localizedCaseInsensitiveContains(query))
                })
            }
        }
    }

    private func load() {
        loading = true
        let path = log.path
        let isDump = !path.hasSuffix(".txt") && !path.hasSuffix(".old")
        let maxBytes = Self.maxBytes
        ContentBridge.run({ () -> [LogLine] in
            if isDump {
                return [LogLine("Binary file: use Share to send it.")]
            }
            let text = path.withCString { ContentBridge.string(lichii_content_read_log($0, maxBytes)) } ?? ""
            return text.split(separator: "\n", omittingEmptySubsequences: false).map { LogLine(String($0)) }
        }) { result in
            lines = result
            shown = result
            loading = false
        }
    }
}
