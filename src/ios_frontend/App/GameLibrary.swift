// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import LichiiBridge
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
        let documentsPrefix = Bridge.documentsDir.standardizedFileURL.path + "/"
        let userDir = Bridge.userDir
        let hidden = hiddenPaths
        try? FileManager.default.createDirectory(at: Bridge.gamesDir, withIntermediateDirectories: true)
        Task.detached(priority: .userInitiated) {
            let result = LibraryMetadataStore.scan(
                roots: roots,
                documentsPrefix: documentsPrefix,
                userDir: userDir,
                hidden: hidden
            )
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
                  url.startAccessingSecurityScopedResource() else {
                // A folder that is offline right now (a drive, a cloud volume) stays linked.
                refreshed.append(data)
                continue
            }
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
            IconCache.removeDiskIcon(forPath: game.path)
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

// MARK: - Persistent Library Metadata & Icon Cache

private struct CachedGameEntry: Codable {
    var path: String
    var stampedSize: UInt64
    var mtimeMs: Int64
    var envStamp: String
    var bootable: Bool
    var title: String
    var developer: String
    var version: String
    var programId: UInt64
    var fileSize: UInt64
    var kindRaw: Int32
    var isHomebrew: Bool
    var hasIcon: Bool
}

private struct LibraryDiskManifest: Codable {
    var version: Int = 1
    var entries: [String: CachedGameEntry] = [:]
}

private extension Game {
    init(cached entry: CachedGameEntry) {
        self = entry.path.withCString { pathPtr in
            entry.title.withCString { titlePtr in
                entry.developer.withCString { devPtr in
                    entry.version.withCString { verPtr in
                        var raw = LichiiGame()
                        raw.path = UnsafeMutablePointer(mutating: pathPtr)
                        raw.title = UnsafeMutablePointer(mutating: titlePtr)
                        raw.developer = UnsafeMutablePointer(mutating: devPtr)
                        raw.version = UnsafeMutablePointer(mutating: verPtr)
                        raw.program_id = entry.programId
                        raw.file_size = entry.fileSize
                        raw.file_type = entry.kindRaw
                        raw.is_homebrew = entry.isHomebrew
                        raw.icon = nil
                        raw.icon_size = 0
                        return Game(raw)
                    }
                }
            }
        }
    }
}

private enum LibraryMetadataStore {
    private static let gameExtensions: Set<String> = [
        "nup", "nsp", "nsz", "xci", "xcz", "nca", "nso", "nro",
    ]
    private static let resourceKeys: Set<URLResourceKey> = [
        .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
        .fileSizeKey, .contentModificationDateKey,
    ]

    private struct Candidate {
        let path: String
        let ext: String
        let stampedSize: UInt64
        let mtimeMs: Int64
    }

    private static var manifestURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return caches.appendingPathComponent("library-cache-v1.json")
    }

    private static func loadManifest() -> LibraryDiskManifest {
        guard let data = try? Data(contentsOf: manifestURL),
              let decoded = try? JSONDecoder().decode(LibraryDiskManifest.self, from: data),
              decoded.version == 1 else {
            return LibraryDiskManifest()
        }
        return decoded
    }

    private static func saveManifest(_ manifest: LibraryDiskManifest) {
        guard let data = try? JSONEncoder().encode(manifest) else { return }
        let dir = manifestURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? data.write(to: manifestURL, options: .atomic)
    }

    private static func fileStamp(_ url: URL) -> String {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]) else {
            return "0:0"
        }
        let size = values.fileSize ?? 0
        let mtime = Int64((values.contentModificationDate?.timeIntervalSince1970 ?? 0) * 1000)
        return "\(size):\(mtime)"
    }

    private static func directoryStamp(_ url: URL) -> String {
        let fm = FileManager.default
        let mtime = Int64(((try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate?.timeIntervalSince1970 ?? 0) * 1000)
        let count = (try? fm.contentsOfDirectory(atPath: url.path))?.count ?? 0
        return "\(count):\(mtime)"
    }

    private static func containerDirectoryMetrics(_ root: URL) -> (UInt64, Int64) {
        var total: UInt64 = 0
        var latestMs: Int64 = Int64(((try? root.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate?.timeIntervalSince1970 ?? 0) * 1000)
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
        let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys)
        while let item = enumerator?.nextObject() as? URL {
            guard let values = try? item.resourceValues(forKeys: Set(keys)),
                  values.isRegularFile == true else { continue }
            total += UInt64(values.fileSize ?? 0)
            let mtime = Int64((values.contentModificationDate?.timeIntervalSince1970 ?? 0) * 1000)
            if mtime > latestMs {
                latestMs = mtime
            }
        }
        return (total, latestMs)
    }

    private static func collectCandidates(in dir: URL, depth: Int, out: inout [Candidate]) {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: Array(resourceKeys),
            options: [.skipsHiddenFiles]
        ) else {
            return
        }
        for item in items {
            let name = item.lastPathComponent
            if name.isEmpty || name.hasPrefix(".") {
                continue
            }
            guard let values = try? item.resourceValues(forKeys: resourceKeys) else {
                continue
            }
            let ext = item.pathExtension.lowercased()
            if values.isDirectory == true {
                if depth == 0 && name == "eden" {
                    continue
                }
                let mainPath = item.appendingPathComponent("main").path
                let npdmPath = item.appendingPathComponent("main.npdm").path
                if fm.fileExists(atPath: mainPath) && fm.fileExists(atPath: npdmPath) {
                    let (size, mtime) = containerDirectoryMetrics(item)
                    out.append(Candidate(path: item.path, ext: "", stampedSize: size, mtimeMs: mtime))
                } else if gameExtensions.contains(ext),
                          fm.fileExists(atPath: item.appendingPathComponent("00").path) {
                    let (size, mtime) = containerDirectoryMetrics(item)
                    out.append(Candidate(path: item.appendingPathComponent("00").path,
                                         ext: ext, stampedSize: size, mtimeMs: mtime))
                } else if depth < 2 {
                    collectCandidates(in: item, depth: depth + 1, out: &out)
                }
            } else if (values.isRegularFile == true || values.isSymbolicLink == true),
                      gameExtensions.contains(ext) {
                let size = UInt64(values.fileSize ?? 0)
                let mtime = Int64((values.contentModificationDate?.timeIntervalSince1970 ?? 0) * 1000)
                out.append(Candidate(path: item.path, ext: ext, stampedSize: size, mtimeMs: mtime))
            }
        }
    }

    static func scan(
        roots: [URL],
        documentsPrefix: String,
        userDir: URL,
        hidden: Set<String>
    ) -> [Game] {
        let fm = FileManager.default
        var manifest = loadManifest()
        var dirty = false

        let keysDir = userDir.appendingPathComponent("keys", isDirectory: true)
        let keysStamp = "\(fileStamp(keysDir.appendingPathComponent("prod.keys")))|\(fileStamp(keysDir.appendingPathComponent("title.keys")))"
        let registeredDir = userDir
            .appendingPathComponent("nand/user/Contents/registered", isDirectory: true)
        let nandStamp = directoryStamp(registeredDir)

        var candidates: [Candidate] = []
        for root in roots {
            collectCandidates(in: root, depth: 0, out: &candidates)
        }

        var found: [Game] = []
        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate.path).inserted {
            let usesNandUpdates = candidate.ext != "nup" && candidate.ext != "nro" && candidate.ext != "nso"
            let envStamp = usesNandUpdates ? "\(keysStamp)|\(nandStamp)" : keysStamp

            if let cached = manifest.entries[candidate.path],
               cached.stampedSize == candidate.stampedSize,
               cached.mtimeMs == candidate.mtimeMs,
               cached.envStamp == envStamp,
               (!cached.hasIcon || fm.fileExists(atPath: IconCache.iconFileURL(forPath: candidate.path).path)) {
                if cached.bootable && !hidden.contains(candidate.path) {
                    found.append(Game(cached: cached))
                }
                continue
            }

            dirty = true
            if let scanned = Bridge.readGame(at: candidate.path) {
                let hasIcon = (scanned.icon?.isEmpty == false)
                IconCache.storeDiskIcon(scanned.icon, forPath: scanned.path)
                let entry = CachedGameEntry(
                    path: scanned.path,
                    stampedSize: candidate.stampedSize,
                    mtimeMs: candidate.mtimeMs,
                    envStamp: envStamp,
                    bootable: true,
                    title: scanned.title,
                    developer: scanned.developer,
                    version: scanned.version,
                    programId: scanned.programId,
                    fileSize: scanned.fileSize,
                    kindRaw: scanned.kind.rawValue,
                    isHomebrew: scanned.isHomebrew,
                    hasIcon: hasIcon
                )
                manifest.entries[candidate.path] = entry
                if !hidden.contains(scanned.path) {
                    found.append(Game(cached: entry))
                }
            } else {
                IconCache.removeDiskIcon(forPath: candidate.path)
                manifest.entries[candidate.path] = CachedGameEntry(
                    path: candidate.path,
                    stampedSize: candidate.stampedSize,
                    mtimeMs: candidate.mtimeMs,
                    envStamp: envStamp,
                    bootable: false,
                    title: "",
                    developer: "",
                    version: "",
                    programId: 0,
                    fileSize: candidate.stampedSize,
                    kindRaw: FileKind.unknown.rawValue,
                    isHomebrew: false,
                    hasIcon: false
                )
            }
        }

        // Remove stale cache entries for deleted files inside the app's own Documents folder.
        let staleKeys = manifest.entries.keys.filter { key in
            let stdPath = URL(fileURLWithPath: key).standardizedFileURL.path
            return stdPath.hasPrefix(documentsPrefix) && !seen.contains(key)
        }
        if !staleKeys.isEmpty {
            dirty = true
            for key in staleKeys {
                manifest.entries.removeValue(forKey: key)
                IconCache.removeDiskIcon(forPath: key)
            }
        }

        if dirty {
            saveManifest(manifest)
        }
        return found
    }
}
