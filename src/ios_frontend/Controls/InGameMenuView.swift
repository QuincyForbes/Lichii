// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import LichiiBridge
import SwiftUI

/// The menu over a running game (menu button, three-finger tap, + and - together on a
/// controller, or a controller button mapped to it). The game pauses while it is open.
struct InGameMenuView: View {
    @ObservedObject private var center = ControlsCenter.shared
    @ObservedObject private var orientation = OrientationManager.shared
    @ObservedObject private var external = ExternalDisplay.shared
    @State private var confirmQuit = false
    @State private var showControllers = false
    @State private var resolution = QuickSettings.string("resolution_setup")
    @State private var frameGen = QuickSettings.bool("frame_gen")
    @State private var frameGenMultiplier = max(2, QuickSettings.int("frame_gen_multiplier"))
    @State private var speedLimit = QuickSettings.bool("use_speed_limit")
    @State private var speedPercent = QuickSettings.int("speed_limit")
    @State private var pendingRestart = false

    private let resolutions = QuickSettings.choices("resolution_setup")

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black.opacity(0.45)
                    .ignoresSafeArea()
                    .onTapGesture { center.closeMenu() }
                    .accessibilityHidden(true)

                // The header (Resume) and Quit stay in view; only the middle scrolls, so the
                // panel fits an iPhone in landscape.
                VStack(spacing: 0) {
                    header
                        .padding(.horizontal, 18)
                        .padding(.vertical, 12)
                    Divider()
                    ScrollView {
                        VStack(alignment: .leading, spacing: 14) {
                            actions
                            quickSettings
                        }
                        .padding(18)
                    }
                    Divider()
                    quit
                        .padding(.horizontal, 18)
                        .padding(.vertical, 12)
                }
                .frame(maxWidth: 480)
                .frame(height: min(560, max(200, proxy.size.height - 16)))
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                .padding(.horizontal, 20)
                .accessibilityElement(children: .contain)
                .accessibilityAction(.escape) { center.closeMenu() }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .sheet(isPresented: $showControllers) {
            NavigationStack {
                ControllersListView()
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showControllers = false }
                        }
                    }
            }
        }
        .confirmationDialog("Quit to the library?", isPresented: $confirmQuit, titleVisibility: .visible) {
            Button("Quit Game", role: .destructive) { center.quitToLibrary() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Unsaved progress is lost.")
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(center.currentGame?.title ?? "Lichii")
                    .font(.headline)
                    .lineLimit(1)
                    .accessibilityAddTraits(.isHeader)
                Text(Bridge.emuState == .paused ? "Paused" : "Running")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ThermalBanner(compact: true)
            }
            Spacer()
            Button {
                center.resumeGame()
            } label: {
                Label("Resume", systemImage: "play.fill")
                    .font(.subheadline.weight(.semibold))
            }
            .buttonStyle(.borderedProminent)
            .accessibilityHint("Closes the menu and continues the game")
        }
    }

    private var actions: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 132), spacing: 10)], spacing: 10) {
            MenuTile(title: center.manuallyPaused ? "Run on Close" : "Keep Paused",
                     systemImage: center.manuallyPaused ? "play.circle" : "pause.circle") {
                center.toggleKeepPaused()
            }
            MenuTile(title: "Screenshot", systemImage: "camera") {
                center.closeMenu()
                center.takeScreenshot()
            }
            MenuTile(title: center.showHUD ? "Hide HUD" : "Show HUD", systemImage: "speedometer") {
                center.showHUD.toggle()
            }
            MenuTile(title: "Edit Controls", systemImage: "hand.tap") {
                center.editLayout()
            }
            MenuTile(title: "Controllers", systemImage: "gamecontroller") {
                showControllers = true
            }
            MenuTile(title: orientation.mode == .locked ? "Unlock Rotation" : "Lock Rotation",
                     systemImage: orientation.mode == .locked ? "lock.rotation" : "lock.rotation.open") {
                orientation.toggleLock()
            }
            if external.connected {
                MenuTile(title: external.enabled ? "Game on This Screen" : "Game on TV",
                         systemImage: "tv") {
                    external.enabled.toggle()
                }
            }
        }
    }

    private var quickSettings: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Quick Settings")
                .font(.subheadline.weight(.semibold))

            if !resolutions.isEmpty {
                Picker("Resolution", selection: $resolution) {
                    ForEach(resolutions) { Text($0.name).tag($0.value) }
                }
                .onChange(of: resolution) { value in change("resolution_setup", value) }
            }

            Toggle("Frame Generation", isOn: $frameGen)
                .onChange(of: frameGen) { value in change("frame_gen", value ? "true" : "false") }
            if frameGen {
                Stepper("Multiplier: \(frameGenMultiplier)x", value: $frameGenMultiplier, in: 2...4)
                    .onChange(of: frameGenMultiplier) { value in change("frame_gen_multiplier", String(value)) }
            }

            Toggle("Limit Speed", isOn: $speedLimit)
                .onChange(of: speedLimit) { value in change("use_speed_limit", value ? "true" : "false") }
            if speedLimit {
                Stepper("Speed Limit: \(speedPercent)%", value: $speedPercent, in: 25...400, step: 25)
                    .onChange(of: speedPercent) { value in change("speed_limit", String(value)) }
            }

            Picker("Orientation", selection: $orientation.mode) {
                ForEach(OrientationManager.Mode.allCases) { Text($0.title).tag($0) }
            }

            if pendingRestart {
                Label("Some changes apply after the game restarts.", systemImage: "arrow.clockwise")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var quit: some View {
        Button(role: .destructive) {
            confirmQuit = true
        } label: {
            Label("Quit to Library", systemImage: "xmark.circle")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
    }

    private func change(_ key: String, _ value: String) {
        guard QuickSettings.string(key) != value else { return }
        QuickSettings.set(key, value)
        if !QuickSettings.appliesLive(key) {
            pendingRestart = true
        }
    }
}

private struct MenuTile: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: systemImage)
                    .font(.title3)
                Text(title)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .frame(maxWidth: .infinity, minHeight: 62)
            .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
    }
}
