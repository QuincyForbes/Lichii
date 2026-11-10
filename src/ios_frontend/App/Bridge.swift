// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import LichiiBridge

/// Swift wrappers over lichii_bridge.h.
enum Bridge {
    /// Takes ownership of a string the bridge allocated.
    static func string(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
        guard let pointer else { return "" }
        defer { lichii_free(pointer) }
        return String(cString: pointer)
    }

    static func constant(_ pointer: UnsafePointer<CChar>?) -> String {
        guard let pointer else { return "" }
        return String(cString: pointer)
    }

    static var documentsDir: URL { URL(fileURLWithPath: string(lichii_documents_dir())) }
    static var userDir: URL { URL(fileURLWithPath: string(lichii_user_dir())) }
    static var logFile: URL { URL(fileURLWithPath: string(lichii_log_file())) }
    static var buildVersion: String { string(lichii_build_version()) }

    /// Where imported games are copied.
    static var gamesDir: URL { documentsDir.appendingPathComponent("Games", isDirectory: true) }

    static func scanGames(in dir: URL) -> [Game] {
        var raw: UnsafeMutablePointer<LichiiGame>?
        let count = dir.path.withCString { lichii_games_scan($0, &raw) }
        guard let raw else { return [] }
        defer { lichii_games_free(raw, count) }
        return (0..<count).map { Game(raw[$0]) }
    }

    static func readGame(at path: String) -> Game? {
        var game = LichiiGame()
        guard path.withCString({ lichii_game_read($0, &game) }) else { return nil }
        defer { lichii_game_clear(&game) }
        return Game(game)
    }

    static var emuState: EmuState { EmuState(rawValue: lichii_emu_state()) ?? .idle }
    static var emuError: String { string(lichii_emu_error()) }

    static var jitState: JitState { JitState(rawValue: lichii_jit_state()) ?? .failed }
    static var jitError: String { string(lichii_jit_error()) }

    static func perfStats() -> LichiiPerfStats? {
        var stats = LichiiPerfStats()
        return lichii_emu_perf_stats(&stats) ? stats : nil
    }
}

enum EmuState: Int32 {
    case idle = 0, waitingForJit, booting, running, paused, failed

    var inGame: Bool { self == .running || self == .paused }
}

enum JitState: Int32 {
    case waiting = 0, acquiring, ready, failed
}

enum FileKind: Int32 {
    case unknown = 0, nso, nro, nca, nsp, nsz, xci, xcz, nup, directory

    var label: String {
        switch self {
        case .unknown: return "?"
        case .nso: return "NSO"
        case .nro: return "NRO"
        case .nca: return "NCA"
        case .nsp: return "NSP"
        case .nsz: return "NSZ"
        case .xci: return "XCI"
        case .xcz: return "XCZ"
        case .nup: return "NUP"
        case .directory: return "Folder"
        }
    }
}

struct Game: Identifiable, Hashable {
    var id: String { path }
    let path: String
    let title: String
    let developer: String
    let version: String
    let programId: UInt64
    let fileSize: UInt64
    let kind: FileKind
    let isHomebrew: Bool
    let icon: Data?

    var programIdHex: String { String(format: "%016llX", programId) }

    init(_ game: LichiiGame) {
        path = game.path.map { String(cString: $0) } ?? ""
        title = game.title.map { String(cString: $0) } ?? ""
        developer = game.developer.map { String(cString: $0) } ?? ""
        version = game.version.map { String(cString: $0) } ?? ""
        programId = game.program_id
        fileSize = game.file_size
        kind = FileKind(rawValue: game.file_type) ?? .unknown
        isHomebrew = game.is_homebrew
        if let icon = game.icon, game.icon_size > 0 {
            self.icon = Data(bytes: icon, count: game.icon_size)
        } else {
            self.icon = nil
        }
    }

    /// The file or folder the user sees for this game (split parts live in a "<name>.nsp" folder).
    var containerURL: URL {
        let url = URL(fileURLWithPath: path)
        return url.lastPathComponent == "00" ? url.deletingLastPathComponent() : url
    }

    // Without the icon bytes, which views compare on every update.
    static func == (lhs: Game, rhs: Game) -> Bool {
        lhs.path == rhs.path && lhs.title == rhs.title && lhs.developer == rhs.developer &&
            lhs.version == rhs.version && lhs.programId == rhs.programId && lhs.fileSize == rhs.fileSize &&
            lhs.icon?.count == rhs.icon?.count
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(path)
    }
}
