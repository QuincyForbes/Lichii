// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// The content area's entries in AppExtensions.
@MainActor
enum ContentEntries {
    static var settingsLinks: [SettingsLink] {
        [
            SettingsLink(id: "keys", title: "Keys & Firmware", systemImage: "key",
                         view: { AnyView(KeysFirmwareView()) }),
            SettingsLink(id: "users", title: "Users", systemImage: "person.2",
                         view: { AnyView(ProfilesView()) }),
            SettingsLink(id: "logs", title: "Logs", systemImage: "doc.text.magnifyingglass",
                         view: { AnyView(LogsView()) }),
            SettingsLink(id: "setup", title: "Setup Assistant", systemImage: "wand.and.stars",
                         view: { AnyView(SetupAssistantView()) }),
        ]
    }

    static var gameActions: [GameAction] {
        [
            GameAction(id: "updates", title: "Updates & DLC", systemImage: "square.stack.3d.up",
                       view: { AnyView(AddonsView(game: $0, initialTab: .updates)) }),
            GameAction(id: "mods", title: "Mods", systemImage: "puzzlepiece.extension",
                       view: { AnyView(AddonsView(game: $0, initialTab: .mods)) }),
            GameAction(id: "cheats", title: "Cheats", systemImage: "wand.and.rays",
                       view: { AnyView(CheatsView(game: $0)) }),
            GameAction(id: "saves", title: "Save Data", systemImage: "externaldrive",
                       view: { AnyView(SavesView(game: $0)) }),
        ]
    }

    static var appStarted: [() -> Void] {
        [
            { GuestDialogs.register() },
            { Onboarding.presentIfNeeded() },
        ]
    }
}

/// Setup status at a glance, and the onboarding again.
struct SetupAssistantView: View {
    @StateObject private var files = SystemFilesModel()

    var body: some View {
        List {
            Section {
                StatusRow(title: "Keys", ok: files.keysReady, optional: true)
                StatusRow(title: "Firmware", ok: files.firmwareReady,
                          detail: files.firmware.flatMap { $0.version.isEmpty ? nil : "Version \($0.version)" },
                          optional: true)
                StatusRow(title: "JIT", ok: Bridge.jitState == .ready, detail: "Ready")
            }
            Section {
                Button {
                    Onboarding.present()
                } label: {
                    Label("Run Setup Again", systemImage: "arrow.counterclockwise")
                }
            }
        }
        .navigationTitle("Setup Assistant")
        .onAppear { files.refresh() }
    }
}
