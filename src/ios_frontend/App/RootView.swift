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
            .tabItem { Label("Library", systemImage: "square.grid.2x2") }

            NavigationStack {
                SettingsView(scope: .global)
            }
            .tabItem { Label("Settings", systemImage: "gearshape") }

            NavigationStack {
                JitView()
            }
            .tabItem { Label("JIT", systemImage: "bolt") }

            ForEach(AppExtensions.tabs) { tab in
                NavigationStack {
                    tab.view()
                }
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
