// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import SwiftUI

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

/// Games found in Documents (and the folders below it), with play history.
@MainActor
final class GameLibrary: ObservableObject {
    @Published private(set) var games: [Game] = []
    @Published private(set) var isScanning = false
    @Published private(set) var lastPlayed: [String: Date] = [:]
    @Published var importing: [String] = []
    @Published var importError: String?

    private let playedKey = "lastPlayed"

    init() {
        if let stored = UserDefaults.standard.dictionary(forKey: playedKey) as? [String: Date] {
            lastPlayed = stored
        }
    }

    func reload() {
        guard !isScanning else { return }
        isScanning = true
        let documents = Bridge.documentsDir
        try? FileManager.default.createDirectory(at: Bridge.gamesDir, withIntermediateDirectories: true)
        Task.detached(priority: .userInitiated) {
            let found = Bridge.scanGames(in: documents)
            await MainActor.run {
                self.games = found
                self.isScanning = false
            }
        }
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

    /// Copies files or folders the user picked into Documents/Games.
    func importItems(_ urls: [URL]) {
        let destination = Bridge.gamesDir
        importing = urls.map(\.lastPathComponent)
        Task.detached(priority: .userInitiated) {
            var failures: [String] = []
            let manager = FileManager.default
            try? manager.createDirectory(at: destination, withIntermediateDirectories: true)
            for url in urls {
                let scoped = url.startAccessingSecurityScopedResource()
                defer {
                    if scoped { url.stopAccessingSecurityScopedResource() }
                }
                let target = destination.appendingPathComponent(url.lastPathComponent)
                do {
                    if manager.fileExists(atPath: target.path) {
                        try manager.removeItem(at: target)
                    }
                    try manager.copyItem(at: url, to: target)
                } catch {
                    failures.append("\(url.lastPathComponent): \(error.localizedDescription)")
                }
            }
            await MainActor.run {
                self.importing = []
                self.importError = failures.isEmpty ? nil : failures.joined(separator: "\n")
                self.reload()
            }
        }
    }

    func delete(_ game: Game) {
        try? FileManager.default.removeItem(at: game.containerURL)
        games.removeAll { $0.id == game.id }
    }
}
