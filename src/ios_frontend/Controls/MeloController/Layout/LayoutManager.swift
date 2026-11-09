// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  LayoutManager.swift
//  Melo-Controller
//
//  Created by Stossy11 on 04/12/2025.
//

import Foundation
import UniformTypeIdentifiers

enum LayoutTransferError: LocalizedError {
    case encodingFailed
    case decodingFailed(underlying: Error)
    case fileWriteFailed(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .encodingFailed:
            return "Failed to encode the layout."
        case .decodingFailed(let e):
            return "The file doesn't appear to be a valid layout: \(e.localizedDescription)"
        case .fileWriteFailed(let e):
            return "Could not write the layout file: \(e.localizedDescription)"
        }
    }
}

/// App-Side editable types
public class LayoutExporter {
    public static var fileType = "meloLayout"

    public static var meloLayout = UTType(exportedAs: "com.stossy11.melocontroller.layout",
                                   conformingTo: .data)
}

class LayoutManager {
    static let shared = LayoutManager()
    private init() {}

    private var baseURL: URL {
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docDir.appendingPathComponent("controller_layouts")
    }

    private func fileURL(for gameId: String?) -> URL {
        let fileName = gameId?.isEmpty == false ? "\(gameId!).json" : "default.json"
        return baseURL.appendingPathComponent(fileName)
    }

    private func ensureDirectoryExists() {
        try? FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
    }

    func save(_ layout: LayoutConfig, for gameId: String? = nil) {
        ensureDirectoryExists()
        try? JSONEncoder().encode(layout).write(to: fileURL(for: gameId))
    }

    func load(for gameId: String? = nil) -> LayoutConfig {
        let url = fileURL(for: gameId)
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(LayoutConfig.self, from: data) else {
            if gameId != nil {
                return load(for: nil)
            }
            return LayoutConfig()
        }
        return config
    }

    func loadLegacy(for gameId: String? = nil) -> [String: ButtonLayout] {
        let url = fileURL(for: gameId)
        guard let data = try? Data(contentsOf: url),
              let legacyConfig = try? JSONDecoder().decode([String: ButtonLayout].self, from: data) else {
            return [:]
        }
        return legacyConfig
    }

    func reset(for gameId: String? = nil) {
        try? FileManager.default.removeItem(at: fileURL(for: gameId))
    }

    func resetAll() {
        try? FileManager.default.removeItem(at: baseURL)
    }

    func hasCustomLayout(for gameId: String) -> Bool {
        FileManager.default.fileExists(atPath: fileURL(for: gameId).path)
    }

    func copyLayout(from sourceGameId: String?, to targetGameId: String?) {
        save(load(for: sourceGameId), for: targetGameId)
    }

    func getAllGameLayouts() -> [String] {
        ensureDirectoryExists()
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: baseURL,
            includingPropertiesForKeys: nil
        ) else { return [] }
        return files.compactMap { url in
            let filename = url.lastPathComponent
            guard filename.hasSuffix(".json"), filename != "default.json" else { return nil }
            return String(filename.dropLast(5))
        }
    }

    func exportLayout(for gameId: String? = nil) throws -> Data {
        let layout = load(for: gameId)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(layout) else {
            throw LayoutTransferError.encodingFailed
        }
        return data
    }

    func exportFileName(for gameId: String? = nil) -> String {
        let base = gameId?.isEmpty == false ? gameId! : "default"
        return "\(base).\(LayoutExporter.fileType)"
    }

    @discardableResult
    func importLayout(from data: Data, for gameId: String? = nil) throws -> LayoutConfig {
        do {
            let layout = try JSONDecoder().decode(LayoutConfig.self, from: data)
            save(layout, for: gameId)
            return layout
        } catch {
            throw LayoutTransferError.decodingFailed(underlying: error)
        }
    }
}
