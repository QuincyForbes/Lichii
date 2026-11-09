// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Combine
import LichiiBridge
import SwiftUI

/// Snapshot of the JIT pool state, refreshed by the views that show it.
struct JitStatus {
    var state: JitState = .waiting
    var prepared: UInt64 = 0
    var total: UInt64 = 0
    var error = ""
    var debugged = false
    var needsScript = false
    var stikDebug = false

    static func read() -> JitStatus {
        JitStatus(state: Bridge.jitState, prepared: lichii_jit_prepared_bytes(),
                  total: lichii_jit_pool_bytes(), error: Bridge.jitError,
                  debugged: lichii_jit_debugged(), needsScript: lichii_jit_needs_script(),
                  stikDebug: lichii_jit_stikdebug_installed())
    }

    var progress: Double {
        total == 0 ? 0 : min(1, Double(prepared) / Double(total))
    }

    var title: String {
        switch state {
        case .waiting:
            return needsScript && debugged ? "Waiting for StikDebug's script" : "JIT not enabled"
        case .acquiring: return "Preparing JIT memory"
        case .ready: return "JIT ready"
        case .failed: return "JIT unavailable"
        }
    }

    var detail: String {
        switch state {
        case .waiting:
            if needsScript && debugged {
                return "JIT was enabled without its script. Enable it again from Lichii's request so StikDebug runs universal.js."
            }
            if stikDebug {
                return "Lichii asks StikDebug to enable JIT. Tap Enable JIT in StikDebug if it asks; Lichii continues by itself."
            }
            return "Install StikDebug 3.2 or newer and enable JIT for Lichii (script universal.js on iOS 26 and newer)."
        case .acquiring:
            return "StikDebug has mapped \(prepared >> 20) of \(total >> 20) MiB."
        case .ready:
            return "Games start right away."
        case .failed:
            return error.isEmpty ? "Close Lichii and launch it again." : error
        }
    }

    var symbol: String {
        switch state {
        case .waiting: return "bolt.slash"
        case .acquiring: return "bolt.badge.clock"
        case .ready: return "bolt.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    var color: Color {
        switch state {
        case .waiting: return .orange
        case .acquiring: return .blue
        case .ready: return .green
        case .failed: return .red
        }
    }
}

struct JitStatusCard: View {
    let status: JitStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(status.title, systemImage: status.symbol)
                .font(.headline)
                .foregroundStyle(status.color)
            Text(status.detail)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if status.state == .acquiring {
                ProgressView(value: status.progress)
            }
        }
    }
}

struct JitView: View {
    @EnvironmentObject private var app: AppController
    @State private var status = JitStatus.read()
    @AppStorage("autoRequestJit") private var autoRequest = true
    private let timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        List {
            Section {
                JitStatusCard(status: status)
                if status.state == .waiting || status.state == .failed {
                    Button {
                        app.requestJitIfNeeded(automatic: false)
                    } label: {
                        Label("Enable JIT with StikDebug", systemImage: "bolt")
                    }
                    .disabled(!status.stikDebug || status.state == .failed)
                }
            }
            Section("Details") {
                LabeledContent("Debugger attached once", value: status.debugged ? "Yes" : "No")
                LabeledContent("Needs StikDebug script", value: status.needsScript ? "Yes (TXM)" : "No")
                LabeledContent("StikDebug", value: status.stikDebug ? "Installed" : "Not found")
                LabeledContent("Pool", value: "\(status.prepared >> 20) / \(status.total >> 20) MiB")
            }
            Section {
                Toggle("Ask StikDebug at launch", isOn: $autoRequest)
            } footer: {
                Text("Lichii prepares its JIT memory while you browse the library, so games start without waiting.")
            }
        }
        .navigationTitle("JIT")
        .onReceive(timer) { _ in status = JitStatus.read() }
    }
}
