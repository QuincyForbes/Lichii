// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// Shown from the moment a game is picked until its render window takes over: JIT wait, boot,
/// or the reason the launch failed.
struct LaunchView: View {
    @EnvironmentObject private var app: AppController
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    /// iPhone landscape: a smaller icon so everything fits without scrolling.
    private var compact: Bool { verticalSizeClass == .compact }

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(white: 0.06), Color(white: 0.12)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            ScrollView {
                VStack(spacing: compact ? 14 : 22) {
                    if let game = app.currentGame {
                        GameIcon(game: game, size: compact ? 72 : 140)
                            .shadow(color: .black.opacity(0.5), radius: 16, y: 6)
                            .accessibilityHidden(true)
                        Text(game.title)
                            .font(.title2.bold())
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                    }
                    content
                        .frame(maxWidth: 460)
                }
                .padding(compact ? 20 : 32)
                .frame(maxWidth: .infinity)
            }
            .scrollBounceBehavior(.basedOnSize)
        }
        .preferredColorScheme(.dark)
    }

    @ViewBuilder
    private var content: some View {
        switch app.emuState {
        case .failed:
            VStack(spacing: 14) {
                Label("Could not start", systemImage: "exclamationmark.triangle.fill")
                    .font(.headline)
                    .foregroundStyle(.red)
                Text(Bridge.emuError)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                HStack {
                    if FileManager.default.fileExists(atPath: Bridge.logFile.path) {
                        ShareLink(item: Bridge.logFile) {
                            Label("Share the Log", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                    }
                    Button("Back to Library") { app.cancelLaunch() }
                        .buttonStyle(.borderedProminent)
                }
            }
        case .booting, .running, .paused:
            VStack(spacing: 12) {
                ProgressView().controlSize(.large).tint(.white)
                Text(app.cancellingBoot ? "Cancelling..."
                     : app.buildingShaders ? "Building Shaders" : "Starting...")
                    .foregroundStyle(.secondary)
                if !app.cancellingBoot {
                    ShaderProgressView()
                    if app.emuState == .booting {
                        Button("Cancel", role: .cancel) { app.cancelLaunch() }
                            .buttonStyle(.bordered)
                    }
                }
            }
            .accessibilityElement(children: .contain)
        default:
            VStack(spacing: 14) {
                JitStatusCard(status: app.jit)
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: Theme.cornerRadius))
                HStack {
                    if app.jit.state == .waiting && app.jit.stikDebug {
                        Button("Open \(JitProvider.current.label)") { app.requestJitIfNeeded(automatic: false) }
                            .buttonStyle(.bordered)
                    }
                    Button("Cancel", role: .cancel) { app.cancelLaunch() }
                        .buttonStyle(.bordered)
                }
            }
        }
    }
}
