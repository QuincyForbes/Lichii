// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import SwiftUI
import UIKit

enum LibrarySort: String, CaseIterable, Identifiable {
    case title, recent, developer, size, fileType

    var id: String { rawValue }

    var label: String {
        switch self {
        case .title: return "Title"
        case .recent: return "Recently Played"
        case .developer: return "Developer"
        case .size: return "Size"
        case .fileType: return "File Type"
        }
    }
}

/// The import being copied: which item and how many bytes of all of them are done.
struct ImportProgress: Equatable {
    var names: [String]
    var index = 0
    var copied: UInt64 = 0
    var total: UInt64 = 0
    /// Imports waiting behind this one.
    var queued = 0

    var fraction: Double? {
        total == 0 ? nil : min(1, Double(copied) / Double(total))
    }
}

/// A message for the library's alert. `removable` are imported files the scan didn't recognise,
/// offered for removal.
struct LibraryNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    var removable: [URL] = []
}

/// Games found in Documents (and the folders below it), with play history.
@MainActor
final class GameLibrary: ObservableObject {
    @Published private(set) var games: [Game] = []
    @Published private(set) var isScanning = false
    @Published private(set) var lastPlayed: [String: Date] = [:]
    @Published private(set) var importProgress: ImportProgress?
    @Published var notice: LibraryNotice?
    /// Games of linked folders removed from the library; their files stay where they are.
    @Published private(set) var hiddenPaths: Set<String> = []

    /// Folders outside the app that are scanned in place (security-scoped bookmarks).
    @Published private(set) var linkedFolders: [URL] = []

    private let playedKey = "lastPlayed"
    private let foldersKey = "linkedFolders"
    private let hiddenKey = "hiddenGames"
    private var accessedFolders: Set<URL> = []
    private var rescanPending = false
    private var scanWaiters: [() -> Void] = []
    private var queuedImports: [[URL]] = []

    init() {
        if let stored = UserDefaults.standard.dictionary(forKey: playedKey) as? [String: Date] {
            lastPlayed = stored
        }
        hiddenPaths = Set(UserDefaults.standard.stringArray(forKey: hiddenKey) ?? [])
        resolveFolders()
        removePartialImports()
    }

    /// Scans again; a request during a scan runs once that scan ends, so nothing is missed.
    /// `completion` runs after the scan that includes every change made before this call.
    func reload(then completion: (() -> Void)? = nil) {
        if let completion {
            scanWaiters.append(completion)
        }
        guard !isScanning else {
            rescanPending = true
            return
        }
        isScanning = true
        let roots = [Bridge.documentsDir] + linkedFolders
        let hidden = hiddenPaths
        try? FileManager.default.createDirectory(at: Bridge.gamesDir, withIntermediateDirectories: true)
        Task.detached(priority: .userInitiated) {
            var found: [Game] = []
            var seen = Set<String>()
            for root in roots {
                for game in Bridge.scanGames(in: root) where seen.insert(game.path).inserted {
                    if !hidden.contains(game.path) {
                        found.append(game)
                    }
                }
            }
            let result = found
            await MainActor.run {
                self.games = result
                self.isScanning = false
                if self.rescanPending {
                    self.rescanPending = false
                    self.reload()
                    return
                }
                let waiters = self.scanWaiters
                self.scanWaiters = []
                waiters.forEach { $0() }
            }
        }
    }

    /// For pull to refresh: returns when the scan is done.
    func refresh() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            reload { continuation.resume() }
        }
    }

    /// Games in a linked folder stay where they are; access lasts for the app session, so a game
    /// from there can boot and stream its files.
    func linkFolder(_ url: URL) {
        guard url.startAccessingSecurityScopedResource() else {
            notice = LibraryNotice(title: "Link a Folder",
                                   message: "Lichii was not allowed to read \(url.lastPathComponent).")
            return
        }
        if linkedFolders.contains(where: { $0.standardizedFileURL == url.standardizedFileURL }) {
            url.stopAccessingSecurityScopedResource()
            reload()
            return
        }
        accessedFolders.insert(url)
        guard let bookmark = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil,
                                                   relativeTo: nil) else {
            notice = LibraryNotice(title: "Link a Folder",
                                   message: "Could not remember \(url.lastPathComponent).")
            return
        }
        var bookmarks = UserDefaults.standard.array(forKey: foldersKey) as? [Data] ?? []
        bookmarks.append(bookmark)
        UserDefaults.standard.set(bookmarks, forKey: foldersKey)
        linkedFolders.append(url)
        reload()
    }

    func unlinkFolder(_ url: URL) {
        let bookmarks = (UserDefaults.standard.array(forKey: foldersKey) as? [Data] ?? []).filter { data in
            var stale = false
            let resolved = try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil,
                                    bookmarkDataIsStale: &stale)
            return resolved?.standardizedFileURL != url.standardizedFileURL
        }
        UserDefaults.standard.set(bookmarks, forKey: foldersKey)
        if accessedFolders.remove(url) != nil {
            url.stopAccessingSecurityScopedResource()
        }
        linkedFolders.removeAll { $0 == url }
        reload()
    }

    private func resolveFolders() {
        var refreshed: [Data] = []
        for data in UserDefaults.standard.array(forKey: foldersKey) as? [Data] ?? [] {
            var stale = false
            guard let url = try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil,
                                     bookmarkDataIsStale: &stale),
                  url.startAccessingSecurityScopedResource() else { continue }
            accessedFolders.insert(url)
            linkedFolders.append(url)
            refreshed.append(stale ? (try? url.bookmarkData()) ?? data : data)
        }
        UserDefaults.standard.set(refreshed, forKey: foldersKey)
    }

    func markPlayed(_ game: Game) {
        lastPlayed[game.programIdHex] = Date()
        UserDefaults.standard.set(lastPlayed, forKey: playedKey)
    }

    func playedDate(_ game: Game) -> Date? {
        lastPlayed[game.programIdHex]
    }

    var recentlyPlayed: [Game] {
        games.compactMap { game in playedDate(game).map { (game, $0) } }
            .sorted { $0.1 > $1.1 }
            .prefix(10)
            .map(\.0)
    }

    func sorted(_ sort: LibrarySort, matching query: String) -> [Game] {
        let filtered = query.isEmpty ? games : games.filter {
            $0.title.localizedCaseInsensitiveContains(query) ||
                $0.developer.localizedCaseInsensitiveContains(query) ||
                $0.programIdHex.localizedCaseInsensitiveContains(query)
        }
        switch sort {
        case .title:
            return filtered.sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
        case .recent:
            return filtered.sorted {
                (playedDate($0) ?? .distantPast, $1.title) > (playedDate($1) ?? .distantPast, $0.title)
            }
        case .developer:
            return filtered.sorted { ($0.developer, $0.title) < ($1.developer, $1.title) }
        case .size:
            return filtered.sorted { $0.fileSize > $1.fileSize }
        case .fileType:
            return filtered.sorted { ($0.kind.label, $0.title) < ($1.kind.label, $1.title) }
        }
    }

    // MARK: - Removing games

    /// A game Lichii doesn't own: it lives in a linked folder (the user's own files) rather than
    /// in the app's Documents.
    func isLinked(_ game: Game) -> Bool {
        let documents = Bridge.documentsDir.standardizedFileURL.path + "/"
        return !game.containerURL.standardizedFileURL.path.hasPrefix(documents)
    }

    /// Deletes the game's file from the app's Documents. Linked games are only hidden (hide).
    func delete(_ game: Game) {
        guard !isLinked(game) else {
            hide(game)
            return
        }
        do {
            try FileManager.default.removeItem(at: game.containerURL)
            games.removeAll { $0.id == game.id }
        } catch {
            notice = LibraryNotice(title: "Could not delete \(game.title)", message: error.localizedDescription)
        }
    }

    /// Removes a game from the library without touching its file.
    func hide(_ game: Game) {
        hiddenPaths.insert(game.path)
        UserDefaults.standard.set(Array(hiddenPaths), forKey: hiddenKey)
        games.removeAll { $0.id == game.id }
    }

    func showHiddenGames() {
        hiddenPaths = []
        UserDefaults.standard.removeObject(forKey: hiddenKey)
        reload()
    }

    // MARK: - Importing

    nonisolated private static let partialSuffix = ".partial"
    private var importBackgroundTask = UIBackgroundTaskIdentifier.invalid

    /// Copies files or folders the user picked into Documents/Games. Imports run one after
    /// another; each item is copied under a .partial name and renamed once complete.
    func importItems(_ urls: [URL]) {
        // Keys picked in the game importer go to the keys installer.
        let keys = urls.filter { $0.pathExtension.lowercased() == "keys" }
        let items = urls.filter { $0.pathExtension.lowercased() != "keys" }
        if !keys.isEmpty {
            SystemFilesModel.shared.importKeys(keys)
            notice = LibraryNotice(title: "Keys",
                                   message: "\(keys.map(\.lastPathComponent).joined(separator: ", ")) went to the keys installer. Settings › Keys & Firmware shows the result.")
        }
        guard !items.isEmpty else { return }
        if importProgress != nil {
            queuedImports.append(items)
            importProgress?.queued = queuedImports.count
            return
        }
        runImport(items)
    }

    private func runImport(_ urls: [URL]) {
        let destination = Bridge.gamesDir
        let manager = FileManager.default
        try? manager.createDirectory(at: destination, withIntermediateDirectories: true)
        importProgress = ImportProgress(names: urls.map(\.lastPathComponent), queued: queuedImports.count)
        KeepAwake.set(true, for: "import")
        // Copies keep going for a short while if the app leaves the foreground.
        importBackgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Lichii import") { [weak self] in
            MainActor.assumeIsolated { self?.endImportBackgroundTask() }
        }
        let current = CurrentPartial()
        let poller = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 500_000_000)
                guard let self, let entry = current.get() else { continue }
                let size = Self.diskSize(URL(fileURLWithPath: entry.0))
                if let total = self.importProgress?.total {
                    self.importProgress?.copied = min(total, entry.1 + size)
                }
            }
        }
        Task.detached(priority: .userInitiated) {
            let sizes = urls.map { url in withSecurityScope(url) { Self.diskSize(url) } }
            let total = sizes.reduce(0, +)
            let free = (try? destination.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage ?? 0
            var failures: [String] = []
            var imported: [URL] = []
            if total > 0, free > 0, UInt64(free) < total {
                failures.append("Not enough free space: the import needs \(total.byteCount) and \(UInt64(free).byteCount) is free. Link a Folder plays games from Files without copying them.")
            } else {
                await MainActor.run { self.importProgress?.total = total }
                var done: UInt64 = 0
                for (index, url) in urls.enumerated() {
                    let target = destination.appendingPathComponent(url.lastPathComponent)
                    if target.standardizedFileURL.path == url.standardizedFileURL.path {
                        done += sizes[index]
                        continue
                    }
                    let partial = destination.appendingPathComponent(url.lastPathComponent + Self.partialSuffix)
                    let before = done
                    current.set(partial.path, before)
                    await MainActor.run {
                        self.importProgress?.index = index
                        self.importProgress?.copied = before
                    }
                    if let error = coordinatedCopy(url, to: partial) {
                        try? FileManager.default.removeItem(at: partial)
                        failures.append("\(url.lastPathComponent): \(error)")
                    } else {
                        do {
                            if FileManager.default.fileExists(atPath: target.path) {
                                try FileManager.default.removeItem(at: target)
                            }
                            try FileManager.default.moveItem(at: partial, to: target)
                            imported.append(target)
                        } catch {
                            try? FileManager.default.removeItem(at: partial)
                            failures.append("\(url.lastPathComponent): \(error.localizedDescription)")
                        }
                    }
                    done += sizes[index]
                    current.clear()
                }
            }
            let message = failures.isEmpty ? nil : failures.joined(separator: "\n")
            let copiedTargets = imported
            await MainActor.run {
                poller.cancel()
                self.importProgress = nil
                KeepAwake.set(false, for: "import")
                self.endImportBackgroundTask()
                if let message {
                    self.notice = LibraryNotice(title: copiedTargets.isEmpty ? "Import failed" : "Some items were not imported",
                                                message: message)
                }
                self.reload { self.reportUnreadable(copiedTargets) }
                if !self.queuedImports.isEmpty {
                    self.runImport(self.queuedImports.removeFirst())
                }
            }
        }
    }

    private func endImportBackgroundTask() {
        guard importBackgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(importBackgroundTask)
        importBackgroundTask = .invalid
    }

    /// Imported items the scan found no game in: encrypted dumps without keys, or other files.
    private func reportUnreadable(_ targets: [URL]) {
        let paths = games.map(\.path)
        let unreadable = targets.filter { target in
            let path = target.path
            return !paths.contains { $0 == path || $0.hasPrefix(path + "/") }
        }
        guard !unreadable.isEmpty else { return }
        let text = "\(unreadable.map(\.lastPathComponent).joined(separator: ", ")): Lichii found no game it can read. Encrypted Switch 1 dumps need your keys (Settings › Keys & Firmware); other files aren't games."
        if let existing = notice {
            notice = LibraryNotice(title: existing.title, message: existing.message + "\n\n" + text,
                                   removable: unreadable)
        } else {
            notice = LibraryNotice(title: "Not a game Lichii can read", message: text, removable: unreadable)
        }
    }

    func removeImported(_ urls: [URL]) {
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
        reload()
    }

    /// Copies left by an import that was interrupted (the app closed or was killed).
    private func removePartialImports() {
        let manager = FileManager.default
        guard let names = try? manager.contentsOfDirectory(atPath: Bridge.gamesDir.path) else { return }
        for name in names where name.hasSuffix(Self.partialSuffix) {
            try? manager.removeItem(at: Bridge.gamesDir.appendingPathComponent(name))
        }
    }

    /// Bytes in a file, or in every file below a folder.
    nonisolated static func diskSize(_ url: URL) -> UInt64 {
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .fileSizeKey]
        guard let values = try? url.resourceValues(forKeys: keys) else { return 0 }
        guard values.isDirectory == true else { return UInt64(values.fileSize ?? 0) }
        var total: UInt64 = 0
        let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: Array(keys))
        while let item = enumerator?.nextObject() as? URL {
            total += UInt64((try? item.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }
}

/// The partial file an import is writing and the bytes done before it, for the progress poller.
private final class CurrentPartial: @unchecked Sendable {
    private let lock = NSLock()
    private var value: (String, UInt64)?

    func set(_ path: String, _ before: UInt64) {
        lock.lock(); value = (path, before); lock.unlock()
    }

    func clear() {
        lock.lock(); value = nil; lock.unlock()
    }

    func get() -> (String, UInt64)? {
        lock.lock(); defer { lock.unlock() }
        return value
    }
}
