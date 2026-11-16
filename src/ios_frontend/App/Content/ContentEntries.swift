// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// The content area's entries in AppExtensions.
@MainActor
enum ContentEntries {
    static var settingsLinks: [SettingsLink] {
        [
            SettingsLink(id: "keys", title: UiText.text("page.keys_firmware"), systemImage: "key.fill",
                         view: { AnyView(KeysFirmwareView()) }),
            SettingsLink(id: "users", title: UiText.text("page.users"), systemImage: "person.2.fill",
                         view: { AnyView(ProfilesView()) }),
            SettingsLink(id: "logs", title: UiText.text("page.logs"), systemImage: "doc.text.magnifyingglass",
                         view: { AnyView(LogsView()) }),
            SettingsLink(id: "setup", title: UiText.text("page.setup_assistant"), systemImage: "checklist",
                         view: { AnyView(SetupAssistantView()) }),
        ]
    }

    static var gameActions: [GameAction] {
        [
            // One screen with Updates, DLC and Mods tabs.
            GameAction(id: "addons", title: UiText.text("game.addons"), systemImage: "square.stack.3d.up",
                       view: { AnyView(AddonsView(game: $0)) }),
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
            { DispatchQueue.global(qos: .utility).async { ContentBridge.removeTemporaryFiles() } },
        ]
    }
}

/// Setup status at a glance, and the onboarding again.
struct SetupAssistantView: View {
    @ObservedObject private var files = SystemFilesModel.shared
    /// Follows LICHII_EVENT_JIT_CHANGED.
    @ObservedObject private var app = AppController.shared

    private var jitState: JitState { app.jit.state }

    var body: some View {
        List {
            Section {
                StatusRow(title: "Keys", ok: files.keysReady, optional: true)
                StatusRow(title: "Firmware", ok: files.firmwareReady,
                          detail: files.firmware.flatMap { $0.version.isEmpty ? nil : "Version \($0.version)" },
                          optional: true)
                StatusRow(title: "JIT", ok: jitState == .ready, detail: "Ready",
                          pending: jitState == .waiting || jitState == .acquiring)
            }
            Section {
                Button {
                    Onboarding.present()
                } label: {
                    Label("Run Setup Again", systemImage: "arrow.counterclockwise")
                }
            }
        }
        .navigationTitle(UiText.text("page.setup_assistant"))
        .onAppear { files.refresh() }
    }
}
