// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import ImageIO
import SwiftUI
import UIKit

enum Theme {
    static let accent = Color(red: 0.55, green: 0.36, blue: 0.96)
    static let cornerRadius: CGFloat = 14
    @MainActor static var isPad: Bool { UIDevice.current.userInterfaceIdiom == .pad }

    /// Badge tint for a settings link or page id, keeping a cohesive Apple-Settings visual rhythm.
    static func badgeColor(for id: String) -> Color {
        switch id {
        case "recommended", "graphics": return accent
        case "controls": return .indigo
        case "audio": return .pink
        case "system", "about": return .blue
        case "keys": return .orange
        case "users": return .teal
        case "shaders": return .cyan
        case "setup": return .green
        case "logs": return .secondary
        case "experiments": return .purple
        case "advanced": return .orange
        default: return accent
        }
    }
}

/// Uniform 29x29 rounded-square icon badge for Settings rows (avoids SwiftUI List's uneven symbol
/// widths and Label overrides).
struct SettingsIconBadge: View {
    @Environment(\.colorScheme) private var colorScheme
    let systemImage: String
    var color: Color = Theme.accent

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(color)
            .frame(width: 29, height: 29)
            .background(
                color.opacity(colorScheme == .dark ? 0.24 : 0.14),
                in: RoundedRectangle(cornerRadius: 7, style: .continuous)
            )
            .accessibilityHidden(true)
    }
}

/// Settings navigation row with a fixed-width icon badge so every row title aligns cleanly.
struct SettingsRowLabel: View {
    let title: String
    let systemImage: String
    var color: Color = Theme.accent

    var body: some View {
        HStack(spacing: 12) {
            SettingsIconBadge(systemImage: systemImage, color: color)
            Text(title)
                .foregroundStyle(.primary)
        }
    }
}

extension Notification.Name {
    static let lichiiIconsEvicted = Notification.Name("LichiiIconsEvicted")
}

/// Decoded game icon thumbnails (~360 px max), bounded in RAM and evicted on memory warnings or
/// when entering gameplay.
enum IconCache {
    static let maxPixelSize = 360

    private static let cache: NSCache<NSString, UIImage> = {
        let c = NSCache<NSString, UIImage>()
        c.totalCostLimit = 16 * 1024 * 1024
        c.countLimit = 96
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: nil
        ) { _ in
            evictAll()
        }
        return c
    }()

    static var iconsDir: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return caches.appendingPathComponent("icons", isDirectory: true)
    }

    static func iconFileURL(forPath path: String) -> URL {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in path.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return iconsDir.appendingPathComponent(String(format: "%016llx.img", hash))
    }

    private static func key(_ game: Game) -> NSString {
        "\(game.path)#\(game.fileSize)#\(game.icon?.count ?? 0)" as NSString
    }

    static func cached(_ game: Game) -> UIImage? {
        cache.object(forKey: key(game))
    }

    static func storeDiskIcon(_ data: Data?, forPath path: String) {
        let url = iconFileURL(forPath: path)
        if let data, !data.isEmpty {
            try? FileManager.default.createDirectory(at: iconsDir, withIntermediateDirectories: true)
            try? data.write(to: url, options: .atomic)
        } else {
            try? FileManager.default.removeItem(at: url)
        }
        evictAll()
    }

    static func removeDiskIcon(forPath path: String) {
        try? FileManager.default.removeItem(at: iconFileURL(forPath: path))
    }

    static func evictAll() {
        cache.removeAllObjects()
        if Thread.isMainThread {
            NotificationCenter.default.post(name: .lichiiIconsEvicted, object: nil)
        } else {
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: .lichiiIconsEvicted, object: nil)
            }
        }
    }

    private static func decodeThumbnail(from data: Data) -> UIImage? {
        let sourceOptions: [CFString: Any] = [
            kCGImageSourceShouldCache: false,
        ]
        guard let source = CGImageSourceCreateWithData(data as CFData, sourceOptions as CFDictionary) else {
            return nil
        }
        let thumbOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOptions as CFDictionary) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }

    static func load(_ game: Game) async -> UIImage? {
        let cacheKey = key(game)
        if let image = cache.object(forKey: cacheKey) {
            return image
        }
        let inlineData = game.icon
        let diskURL = iconFileURL(forPath: game.path)
        let image: UIImage? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let rawData = inlineData ?? (try? Data(contentsOf: diskURL))
                guard let rawData, !rawData.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: decodeThumbnail(from: rawData))
            }
        }
        if let image {
            let cost: Int
            if let cg = image.cgImage {
                cost = max(1, cg.bytesPerRow * cg.height)
            } else {
                cost = max(1, Int(image.size.width * image.scale * image.size.height * image.scale * 4))
            }
            cache.setObject(image, forKey: cacheKey, cost: cost)
        }
        return image
    }
}

/// A game's icon, or a placeholder with its initials. `fills`: a square as wide as the space
/// offered instead of `size`.
struct GameIcon: View {
    let game: Game
    var size: CGFloat = 120
    var fills = false
    @State private var loaded: UIImage?
    @State private var reloadToken = 0

    var body: some View {
        Group {
            if !fills {
                icon(size)
            } else {
                Color.clear
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        GeometryReader { proxy in
                            icon(proxy.size.width)
                        }
                    }
            }
        }
        .accessibilityHidden(true)
        .task(id: "\(game.id)#\(reloadToken)") {
            loaded = await IconCache.load(game)
        }
        .onDisappear {
            loaded = nil
        }
        .onReceive(NotificationCenter.default.publisher(for: .lichiiIconsEvicted)) { _ in
            loaded = nil
            if !Bridge.emuState.inGame {
                reloadToken += 1
            }
        }
    }

    private func icon(_ side: CGFloat) -> some View {
        Group {
            if let image = loaded ?? IconCache.cached(game) {
                Image(uiImage: image)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(1, contentMode: .fill)
            } else {
                ZStack {
                    LinearGradient(colors: [Theme.accent.opacity(0.85), .indigo],
                                   startPoint: .topLeading, endPoint: .bottomTrailing)
                    Text(initials)
                        .font(.system(size: side * 0.3, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
            }
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: side * 0.12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: side * 0.12, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.75)
        )
    }

    private var initials: String {
        let words = game.title.split(separator: " ").prefix(2)
        return words.compactMap { $0.first.map(String.init) }.joined().uppercased()
    }
}

extension UInt64 {
    var byteCount: String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: self), countStyle: .file)
    }
}
