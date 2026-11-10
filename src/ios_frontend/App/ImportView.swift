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
    let names: [String]

    var body: some View {
        HStack(spacing: 12) {
            ProgressView()
            VStack(alignment: .leading, spacing: 2) {
                Text("Importing \(names.count) item\(names.count == 1 ? "" : "s")...")
                    .font(.subheadline.weight(.semibold))
                Text(names.joined(separator: ", "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: Theme.cornerRadius))
    }
}
