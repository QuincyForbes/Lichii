// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import LichiiBridge

/// Swift side of lichii_bridge_content.h: JSON results decoded, blocking calls run off the main
/// thread.
enum ContentBridge {
    static func string(_ pointer: UnsafeMutablePointer<CChar>?) -> String? {
        guard let pointer else { return nil }
        defer { lichii_content_free(pointer) }
        return String(cString: pointer)
    }

    static func decode<T: Decodable>(_ type: T.Type, _ pointer: UnsafeMutablePointer<CChar>?) -> T? {
        guard let text = string(pointer), let data = text.data(using: .utf8) else { return nil }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try? decoder.decode(type, from: data)
    }

    /// Runs `work` on a background queue and hands the result to the main queue.
    static func run<T>(_ work: @escaping @Sendable () -> T, then done: @escaping (T) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let result = work()
            DispatchQueue.main.async { done(result) }
        }
    }

    /// Bridges a Swift progress closure to LichiiContentProgress. The closure runs on the worker
    /// thread; return true to cancel.
    final class Progress: @unchecked Sendable {
        let handler: @Sendable (Double) -> Bool
        init(_ handler: @escaping @Sendable (Double) -> Bool) { self.handler = handler }

        private static let trampoline: LichiiContentProgress = { user, done, total in
            guard let user else { return false }
            let progress = Unmanaged<Progress>.fromOpaque(user).takeUnretainedValue()
            return progress.handler(total == 0 ? 0 : Double(done) / Double(total))
        }

        func call<T>(_ body: (LichiiContentProgress, UnsafeMutableRawPointer) -> T) -> T {
            let context = Unmanaged.passUnretained(self).toOpaque()
            return withExtendedLifetime(self) {
                body(Progress.trampoline, context)
            }
        }
    }

    static func message(_ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Void) -> String {
        var raw: UnsafeMutablePointer<CChar>?
        body(&raw)
        return string(raw) ?? ""
    }

    static func hex(_ value: UInt64) -> String { String(format: "%016llX", value) }

    // MARK: - Status

    struct Paths: Decodable {
        let userDir: String
        let documents: String
        let keysDir: String
        let loadDir: String
        let logDir: String
        let nandDir: String
        let sdmcDir: String
        let saveDir: String
        let cacheDir: String
    }

    struct KeyStatus: Decodable {
        let prodKeys: Bool
        let titleKeys: Bool
        let headerKey: Bool
        let masterKeys: Int
        let titleKeyCount: Int
        let usable: Bool
    }

    struct FirmwareStatus: Decodable {
        let present: Bool
        let version: String
        let title: String
        let corrupted: Bool
    }

    struct SetupStatus: Decodable {
        let keys: KeyStatus
        let firmware: FirmwareStatus
        let profiles: Int
        let ready: Bool
    }

    static var paths: Paths? { decode(Paths.self, lichii_content_paths_json()) }
    static var keyStatus: KeyStatus? { decode(KeyStatus.self, lichii_content_key_status_json()) }
    static var firmwareStatus: FirmwareStatus? {
        decode(FirmwareStatus.self, lichii_content_firmware_status_json())
    }
    static var setupStatus: SetupStatus? { decode(SetupStatus.self, lichii_content_setup_status_json()) }

    // MARK: - Keys and firmware

    static func installKeys(_ urls: [URL]) -> (LichiiKeyResult, String) {
        let paths = urls.map { strdup($0.path) }
        defer { paths.forEach { free($0) } }
        var raw: UnsafeMutablePointer<CChar>?
        let pointers: [UnsafePointer<CChar>?] = paths.map { $0.map { UnsafePointer($0) } }
        let result = pointers.withUnsafeBufferPointer { buffer in
            lichii_content_install_keys(buffer.baseAddress, Int32(buffer.count), &raw)
        }
        return (result, string(raw) ?? "")
    }

    static func installFirmware(_ url: URL, progress: Progress) -> (LichiiFirmwareResult, String) {
        var raw: UnsafeMutablePointer<CChar>?
        let result = progress.call { callback, context in
            url.path.withCString { lichii_content_install_firmware($0, callback, context, &raw) }
        }
        return (result, string(raw) ?? "")
    }

    // MARK: - Add-ons

    struct Addon: Decodable, Identifiable, Hashable {
        var id: String { "\(type)/\(toggleKey)/\(location)" }
        let name: String
        let toggleKey: String
        let version: String
        let type: String
        let enabled: Bool
        let source: String
        let titleId: String
        let numericVersion: UInt32
        let location: String
        let removable: Bool
        let builtin: BuiltinPatch?

        var isUpdate: Bool { type == "update" }
        var isDLC: Bool { type == "dlc" }
        var isMod: Bool { type == "mod" || type == "builtin" || type == "cheat" }
    }

    /// What a built-in patch does and what decided whether it applies.
    struct BuiltinPatch: Decodable, Hashable {
        let kind: String
        let game: String
        let description: String
        let effect: String
        /// "global" (follows the global setting of its kind), "on" or "off" (this game's choice).
        let choice: String
        /// What decided: "game", "global" or "default".
        let source: String
        let resolvedEnabled: Bool
        /// An enabled built-in patch that contains this one; empty when none.
        let includedIn: String

        var followsGlobal: Bool { choice == "global" }
    }

    /// 0 follows the global setting, 1 forces the patch on, 2 forces it off.
    static func setBuiltinChoice(_ game: Game, name: String, choice: Int32) {
        _ = name.withCString { lichii_content_set_builtin_patch_choice(game.programId, $0, choice) }
    }

    /// The games that ship built-in patches.
    static func builtinPatchGames() -> [String] {
        decode([String].self, lichii_content_builtin_patch_games_json()) ?? []
    }

    struct DLC: Decodable, Identifiable, Hashable {
        var id: String { titleId }
        let titleId: String
        let index: UInt64
        let source: String
        let removable: Bool
    }

    static func addons(_ game: Game) -> [Addon] {
        decode([Addon].self, game.path.withCString { lichii_content_list_addons_json(game.programId, $0) }) ?? []
    }

    static func dlc(_ game: Game) -> [DLC] {
        decode([DLC].self, lichii_content_list_dlc_json(game.programId)) ?? []
    }

    static func installPackage(_ url: URL, for game: Game, progress: Progress) -> LichiiInstallResult {
        progress.call { callback, context in
            url.path.withCString { lichii_content_install_nsp($0, game.programId, callback, context) }
        }
    }

    static func describe(_ result: LichiiInstallResult) -> String {
        switch result {
        case LICHII_INSTALL_OK: return "Installed."
        case LICHII_INSTALL_OVERWRITE: return "Installed (replaced the previous version)."
        case LICHII_INSTALL_BASE_GAME: return "This is a base game. Add games to the library instead; only updates and DLC are installed."
        case LICHII_INSTALL_NOT_FOUND: return "The file could not be read."
        case LICHII_INSTALL_WRONG_TYPE: return "Only .nsp and .nsz packages can be installed."
        case LICHII_INSTALL_NO_KEYS: return "Install your keys first."
        case LICHII_INSTALL_WRONG_GAME: return "This update or DLC belongs to another game."
        case LICHII_INSTALL_CANCELLED: return "Cancelled."
        default: return "The installation failed. The log has the details."
        }
    }

    // MARK: - Cheats

    struct Cheat: Decodable, Identifiable, Hashable {
        var id: String { name }
        let name: String
        let enabled: Bool
        let opcodes: Int
    }

    struct CheatFile: Decodable, Identifiable, Hashable {
        var id: String { path }
        let source: String
        let sourceEnabled: Bool
        let file: String
        let path: String
        let cheats: [Cheat]
    }

    static func cheats(_ game: Game) -> [CheatFile] {
        decode([CheatFile].self, lichii_content_list_cheats_json(game.programId)) ?? []
    }

    // MARK: - Saves

    struct SaveInfo: Decodable {
        let path: String
        let exists: Bool
        let size: UInt64
        let files: Int
    }

    static func saveInfo(_ game: Game) -> SaveInfo? {
        decode(SaveInfo.self, game.path.withCString { lichii_content_save_info_json(game.programId, $0) })
    }

    // MARK: - Logs

    struct LogFile: Decodable, Identifiable, Hashable {
        var id: String { path }
        let name: String
        let path: String
        let size: UInt64
        let modified: Int64

        var date: Date { Date(timeIntervalSince1970: TimeInterval(modified)) }
    }

    static var logs: [LogFile] { decode([LogFile].self, lichii_content_list_logs_json()) ?? [] }

    // MARK: - Profiles

    struct Profile: Decodable, Identifiable, Hashable {
        var id: String { uuid }
        let uuid: String
        let name: String
        let current: Bool
        let avatar: String
        let index: Int
    }

    static var profiles: [Profile] { decode([Profile].self, lichii_content_list_profiles_json()) ?? [] }

    private static var exportsDir: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("exports", isDirectory: true)
    }

    /// Exports of earlier sessions and import staging folders left by an interrupted install.
    static func removeTemporaryFiles() {
        let manager = FileManager.default
        try? manager.removeItem(at: exportsDir)
        let temporary = manager.temporaryDirectory
        for name in (try? manager.contentsOfDirectory(atPath: temporary.path)) ?? []
            where name.hasPrefix("lichii-import-") {
            try? manager.removeItem(at: temporary.appendingPathComponent(name))
        }
    }

    /// A temporary file for exports handed to the share sheet.
    static func exportURL(_ name: String) -> URL {
        let dir = exportsDir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: url)
        return url
    }
}

/// A cancel switch the UI flips and a worker thread reads.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    func set(_ value: Bool) {
        lock.lock(); cancelled = value; lock.unlock()
    }
}

/// Keeps a security-scoped URL from the document picker readable while `body` runs.
func withSecurityScope<T>(_ url: URL, _ body: () -> T) -> T {
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    return body()
}

/// Runs `body` with a readable URL for a picked file or folder, inside a coordinated read: that
/// downloads an iCloud Drive or file provider item first, where a plain read finds no data. The
/// URL is only valid during `body`. Returns nil and the error text when the item can't be read.
func coordinatedRead<T>(_ url: URL, _ body: (URL) -> T) -> (T?, String?) {
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    var coordinationError: NSError?
    var result: T?
    NSFileCoordinator().coordinate(readingItemAt: url, options: [], error: &coordinationError) { readable in
        result = body(readable)
    }
    if let result {
        return (result, nil)
    }
    return (nil, coordinationError?.localizedDescription ?? "The file could not be read.")
}

/// Copies a picked file or folder to `target` through a coordinated read (coordinatedRead).
/// Returns the error text, nil on success.
func coordinatedCopy(_ url: URL, to target: URL) -> String? {
    let (failure, error) = coordinatedRead(url) { readable -> String? in
        let manager = FileManager.default
        do {
            if manager.fileExists(atPath: target.path) {
                try manager.removeItem(at: target)
            }
            try manager.copyItem(at: readable, to: target)
            return nil
        } catch {
            return error.localizedDescription
        }
    }
    if let error {
        return error
    }
    return failure ?? nil
}

/// A new temporary folder for coordinated copies; the caller removes it.
func makeStagingDirectory() -> URL? {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("lichii-import-\(UUID().uuidString)", isDirectory: true)
    do {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    } catch {
        return nil
    }
}
