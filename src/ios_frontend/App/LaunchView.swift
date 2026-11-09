// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Combine
import SwiftUI

/// Shown from the moment a game is picked until its render window takes over: JIT wait, boot,
/// or the reason the launch failed.
struct LaunchView: View {
    @EnvironmentObject private var app: AppController
    @State private var jit = JitStatus.read()
    private let timer = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(white: 0.06), Color(white: 0.12)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            VStack(spacing: 22) {
                if let game = app.currentGame {
                    GameIcon(game: game, size: 140)
                        .shadow(color: .black.opacity(0.5), radius: 16, y: 6)
                    Text(game.title)
                        .font(.title2.bold())
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                }
                content
                    .frame(maxWidth: 460)
            }
            .padding(32)
        }
        .preferredColorScheme(.dark)
        .onReceive(timer) { _ in jit = JitStatus.read() }
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
                Button("Back to Library") { app.cancelLaunch() }
                    .buttonStyle(.borderedProminent)
            }
        case .booting, .running, .paused:
            VStack(spacing: 12) {
                ProgressView().controlSize(.large).tint(.white)
                Text("Starting...").foregroundStyle(.secondary)
            }
        default:
            VStack(spacing: 14) {
                JitStatusCard(status: jit)
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: Theme.cornerRadius))
                HStack {
                    if jit.state == .waiting && jit.stikDebug {
                        Button("Open StikDebug") { app.requestJitIfNeeded(automatic: false) }
                            .buttonStyle(.bordered)
                    }
                    Button("Cancel", role: .cancel) { app.cancelLaunch() }
                        .buttonStyle(.bordered)
                }
            }
        }
    }
}
