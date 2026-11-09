// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import UIKit

struct AppTab: Identifiable {
    let id: String
    let title: String
    let systemImage: String
    let view: () -> AnyView
}

struct SettingsLink: Identifiable {
    let id: String
    let title: String
    let systemImage: String
    let view: () -> AnyView
}

/// A per-game screen offered in the library's context menu and the game info sheet.
struct GameAction: Identifiable {
    let id: String
    let title: String
    let systemImage: String
    let view: (Game) -> AnyView
}

/// Where the other app areas (controls, content, platform) plug in. Add one line per entry and keep
/// the views in your own folder.
@MainActor
enum AppExtensions {
    /// Tabs after Library, Settings and JIT.
    static var tabs: [AppTab] {
        [
            // ios-content: AppTab(id: "content", title: "Content", systemImage: "shippingbox", view: { AnyView(ContentHubView()) }),
        ]
    }

    /// Links shown at the top of the global settings (keys, firmware, controllers, logs, ...).
    static var settingsLinks: [SettingsLink] {
        [
            // ios-content: SettingsLink(id: "keys", title: "Keys & Firmware", systemImage: "key", view: { AnyView(KeysView()) }),
            // ios-controls: SettingsLink(id: "controls", title: "Controllers", systemImage: "gamecontroller", view: { AnyView(ControlsView()) }),
        ]
    }

    /// Per-game screens (updates, DLC, mods, cheats, controller layout, ...).
    static var gameActions: [GameAction] {
        [
            // ios-content: GameAction(id: "addons", title: "Updates & DLC", systemImage: "square.stack.3d.up", view: { AnyView(AddonsView(game: $0)) }),
        ]
    }

    /// Replaces the basic in-game menu (ios-controls). Gets the game window's view controller.
    static var presentInGameMenu: ((UIViewController) -> Void)? { nil }

    /// Called with the render window each time a game boots (HUD, touch controls, ...).
    static var gameWindowReady: [(UIWindow) -> Void] { [] }

    /// Called once after the library window is up (onboarding, ...).
    static var appStarted: [() -> Void] { [] }

    /// Every LichiiEvent the bridge posts.
    static var eventHandlers: [(Int32) -> Void] { [] }
}
