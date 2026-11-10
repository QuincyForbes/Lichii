// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UIKit

enum Theme {
    static let accent = Color(red: 0.55, green: 0.36, blue: 0.96)
    static let cornerRadius: CGFloat = 14
}

/// Decoded game icons. Decoding runs off the main thread, once per icon.
enum IconCache {
    private static let cache = NSCache<NSString, UIImage>()

    private static func key(_ game: Game) -> NSString {
        "\(game.path)#\(game.icon?.count ?? 0)" as NSString
    }

    static func cached(_ game: Game) -> UIImage? {
        cache.object(forKey: key(game))
    }

    static func load(_ game: Game) async -> UIImage? {
        if let image = cached(game) {
            return image
        }
        guard let data = game.icon else { return nil }
        let image: UIImage? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: UIImage(data: data)?.preparingForDisplay())
            }
        }
        if let image {
            cache.setObject(image, forKey: key(game))
        }
        return image
    }
}

/// A game's icon, or a placeholder with its initials.
struct GameIcon: View {
    let game: Game
    var size: CGFloat = 120
    @State private var loaded: UIImage?

    var body: some View {
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
                        .font(.system(size: size * 0.3, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.12, style: .continuous))
        .accessibilityHidden(true)
        .task(id: game) {
            loaded = await IconCache.load(game)
        }
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
