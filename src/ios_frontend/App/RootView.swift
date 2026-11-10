// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

struct RootView: View {
    @EnvironmentObject private var app: AppController
    @ObservedObject private var platform = PlatformCenter.shared

    var body: some View {
        TabView {
            NavigationStack {
                LibraryView()
            }
            .clearsTabBar()
            .tabItem { Label("Library", systemImage: "square.grid.2x2") }

            NavigationStack {
                SettingsView(scope: .global)
            }
            .clearsTabBar()
            .tabItem { Label("Settings", systemImage: "gearshape") }

            NavigationStack {
                JitView()
            }
            .clearsTabBar()
            .tabItem { Label("JIT", systemImage: "bolt") }

            ForEach(AppExtensions.tabs) { tab in
                NavigationStack {
                    tab.view()
                }
                .clearsTabBar()
                .tabItem { Label(tab.title, systemImage: tab.systemImage) }
            }
        }
        .tint(Theme.accent)
        .fullScreenCover(isPresented: $app.showLaunch) {
            LaunchView()
        }
        .sheet(item: $platform.uncleanExit) { session in
            UncleanExitView(session: session)
        }
    }
}

private extension View {
    /// The floating tab bar (iOS 26) draws over the bottom of a tab's lists; every scroll view in
    /// the tab, pushed screens included, gets room to scroll its last rows clear of it.
    func clearsTabBar() -> some View {
        contentMargins(.bottom, 72, for: .scrollContent)
    }
}
