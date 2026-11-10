// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UIKit
import UniformTypeIdentifiers

enum ImportTypes {
    /// Game files have no registered types, so anything (and folders, for split or extracted
    /// dumps) can be picked; the scan keeps what the loader recognises.
    static let games: [UTType] = [.item, .folder]
}

/// Keeps the screen from locking while long copies or installs run (they stop when the app is
/// suspended). Several owners can hold it at once.
@MainActor
enum KeepAwake {
    private static var owners: Set<String> = []

    static func set(_ awake: Bool, for owner: String) {
        let before = !owners.isEmpty
        if awake {
            owners.insert(owner)
        } else {
            owners.remove(owner)
        }
        let after = !owners.isEmpty
        guard before != after else { return }
        // A running game keeps its own idle timer setting (lichii_platform_set_playing).
        if !Bridge.emuState.inGame {
            UIApplication.shared.isIdleTimerDisabled = after
        }
    }
}

struct ImportProgressBanner: View {
    let progress: ImportProgress

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                if progress.fraction == nil {
                    ProgressView()
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(progress.names.indices.contains(progress.index) ? progress.names[progress.index] : "")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
            }
            if let fraction = progress.fraction {
                ProgressView(value: fraction)
                    .tint(Theme.accent)
                Text("\(progress.copied.byteCount) of \(progress.total.byteCount). Keep Lichii open until the copy is done.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: Theme.cornerRadius))
        .accessibilityElement(children: .combine)
    }

    private var title: String {
        let count = progress.names.count
        var text = count == 1 ? "Importing 1 item" : "Importing item \(progress.index + 1) of \(count)"
        if progress.queued > 0 {
            text += " (\(progress.queued) more waiting)"
        }
        return text
    }
}
