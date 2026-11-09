// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UIKit

enum Theme {
    static let accent = Color(red: 0.55, green: 0.36, blue: 0.96)
    static let cornerRadius: CGFloat = 14
}

/// A game's icon, or a placeholder with its initials.
struct GameIcon: View {
    let game: Game
    var size: CGFloat = 120

    var body: some View {
        Group {
            if let data = game.icon, let image = UIImage(data: data) {
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
