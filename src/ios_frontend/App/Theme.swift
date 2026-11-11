// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import ImageIO
import SwiftUI
import UIKit

enum Theme {
    static let accent = Color(red: 0.55, green: 0.36, blue: 0.96)
    static let cornerRadius: CGFloat = 14
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
        .task(id: game) {
            loaded = await IconCache.load(game)
        }
        .onDisappear {
            loaded = nil
        }
        .onReceive(NotificationCenter.default.publisher(for: .lichiiIconsEvicted)) { _ in
            loaded = nil
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
