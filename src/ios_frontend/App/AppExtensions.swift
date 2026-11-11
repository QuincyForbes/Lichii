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
        ]
    }

    /// Links shown at the top of the global settings (keys, firmware, controllers, logs, ...).
    static var settingsLinks: [SettingsLink] {
        let links = ContentEntries.settingsLinks
        func content(_ id: String) -> [SettingsLink] { links.filter { $0.id == id } }
        return content("keys") + [
            SettingsLink(id: "controls", title: "Controls", systemImage: "gamecontroller", view: { AnyView(ControlsView()) }),
        ] + content("logs") + content("users") + PlatformEntries.settingsLinks + content("setup")
    }

    /// Sections added to the end of Settings > Graphics (app-side options such as the performance
    /// overlay). Each returns a `Section`.
    static var graphicsSettingsSections: [() -> AnyView] {
        [
            { AnyView(ThermalPacingSettingsSection()) },
            { AnyView(PerformanceOverlaySettingsSection()) },
        ]
    }

    /// Per-game screens after Game Settings (controller layout, updates, DLC, mods, cheats, saves, ...).
    static var gameActions: [GameAction] {
        [
            GameAction(id: "layout", title: "Controller Layout", systemImage: "hand.tap",
                       view: { AnyView(LayoutEditorScreen(gameId: $0.programIdHex, gameName: $0.title)) }),
        ] + ContentEntries.gameActions + PlatformEntries.gameActions
    }

    /// Replaces the basic in-game menu (ios-controls). Gets the game window's view controller.
    static var presentInGameMenu: ((UIViewController) -> Void)? { { _ in MainActor.assumeIsolated { ControlsCenter.shared.openMenu() } } }

    /// Called with the render window each time a game boots (HUD, touch controls, ...).
    static var gameWindowReady: [(UIWindow) -> Void] { [{ window in MainActor.assumeIsolated { ControlsCenter.shared.gameWindowReady(window) } }] }

    /// Called once after the library window is up (onboarding, ...).
    static var appStarted: [() -> Void] {
        [{ MainActor.assumeIsolated { PlatformCenter.shared.start() } }] + ContentEntries.appStarted +
            [{ MainActor.assumeIsolated { ControlsCenter.shared.appStarted() } }]
    }

    /// Every LichiiEvent the bridge posts.
    static var eventHandlers: [(Int32) -> Void] {
        [
            { event in MainActor.assumeIsolated { PlatformCenter.shared.handle(appEvent: event) } },
            { event in MainActor.assumeIsolated { ControlsCenter.shared.handle(event: event) } },
        ]
    }
}
