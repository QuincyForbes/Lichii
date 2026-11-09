// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UniformTypeIdentifiers

enum ImportTypes {
    /// Game files have no registered types, so anything (and folders, for split or extracted
    /// dumps) can be picked; the scan keeps what the loader recognises.
    static let games: [UTType] = [.item, .folder]
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
