// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var app: AppController
    @ObservedObject private var platform = PlatformCenter.shared
    /// The device the "Use recommended settings?" prompt was last shown for: once per device.
    @AppStorage("recommendedPromptDevice") private var promptedDevice = ""
    @State private var recommendedPrompt: RecommendedPrompt?

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
        .alert("Not Enough Free Storage", isPresented: Binding(get: { app.storageWarning != nil },
                                                               set: { if !$0 { app.storageWarning = nil } }),
               presenting: app.storageWarning) { warning in
            Button("Start Anyway") {
                app.launch(warning.game, storageChecked: true, buildShaders: warning.buildShaders)
            }
            Button("Cancel", role: .cancel) {}
        } message: { warning in
            Text("Keep Guest Memory in a File needs \(GuestMemoryFile.size(warning.storage.needed_mib)) of free storage; \(GuestMemoryFile.size(warning.storage.free_mib)) is free. \(warning.game.title) would start with its memory in RAM, and iOS may close Lichii in a large game. Free up storage in the Settings app > General > iPhone Storage, or start anyway.")
        }
        .alert("Use Recommended Settings?", isPresented: Binding(get: { recommendedPrompt != nil },
                                                                 set: { if !$0 { recommendedPrompt = nil } }),
               presenting: recommendedPrompt) { prompt in
            Button("Use Recommended") {
                _ = RecommendedSettings.apply(prompt.values)
            }
            Button("Not Now", role: .cancel) {}
        } message: { prompt in
            Text("These settings differ from what \(prompt.device) should use:\n\n\(prompt.summary)\n\nSettings > Recommended Settings has the reasons and the button for later.")
        }
        .onAppear(perform: offerRecommended)
    }

    /// Once per device: an earlier install's settings (or a config migrated from them) can be far
    /// from what the phone needs; one iPhone ran Ocarina of Time at full textures and the game's
    /// resolution that way and reached critical memory pressure.
    private func offerRecommended() {
        guard recommendedPrompt == nil, !Bridge.emuState.inGame, lichii_recommended_differences() > 0 else { return }
        let loaded = RecommendedSettings.load()
        guard !loaded.device.isEmpty, promptedDevice != loaded.device else { return }
        let changes = loaded.values.filter(\.differs)
        guard !changes.isEmpty else { return }
        promptedDevice = loaded.device
        recommendedPrompt = RecommendedPrompt(
            device: loaded.device, values: loaded.values,
            summary: changes.map { "\($0.label): \($0.currentText) → \($0.recommendedText)" }.joined(separator: "\n"))
    }
}

private struct RecommendedPrompt: Identifiable {
    let device: String
    let values: [RecommendedValue]
    let summary: String
    var id: String { device }
}

private extension View {
    /// The floating tab bar (iOS 26) draws over the bottom of a tab's lists; every scroll view in
    /// the tab, pushed screens included, gets room to scroll its last rows clear of it.
    func clearsTabBar() -> some View {
        contentMargins(.bottom, 72, for: .scrollContent)
    }
}
