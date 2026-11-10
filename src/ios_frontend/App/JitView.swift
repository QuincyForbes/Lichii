// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Combine
import Darwin
import LichiiBridge
import SwiftUI
import UIKit

/// Apps that can attach a debugger to Lichii. Only StikDebug runs the script iOS 26+ (TXM) needs;
/// the others work where an attached debugger is enough.
enum JitProvider: String, CaseIterable, Identifiable {
    case stikDebug, stikDebugLiveContainer, stosDebug, trollStore

    var id: String { rawValue }

    var label: String {
        switch self {
        case .stikDebug: return "StikDebug"
        case .stikDebugLiveContainer: return "StikDebug (LiveContainer)"
        case .stosDebug: return "StosDebug"
        case .trollStore: return "TrollStore"
        }
    }

    static var current: JitProvider {
        JitProvider(rawValue: UserDefaults.standard.string(forKey: "jitProvider") ?? "") ?? .stikDebug
    }

    @MainActor var installed: Bool {
        switch self {
        case .stikDebug: return lichii_jit_stikdebug_installed()
        case .stikDebugLiveContainer: return canOpen("livecontainer://")
        case .stosDebug: return canOpen("stosdebug://")
        case .trollStore: return canOpen("apple-magnifier://")
        }
    }

    @MainActor @discardableResult
    func request() -> Bool {
        let bundle = Bundle.main.bundleIdentifier ?? ""
        let pid = getpid()
        switch self {
        case .stikDebug:
            return lichii_jit_request()
        case .stikDebugLiveContainer:
            // LiveContainer forwards the request to the StikDebug inside it.
            var inner = "stikdebug://enable-jit?pid=\(pid)"
            if lichii_jit_needs_script() {
                inner += "&script-name=universal.js"
            }
            let encoded = Data(inner.utf8).base64EncodedString()
                .replacingOccurrences(of: "+", with: "-")
                .replacingOccurrences(of: "/", with: "_")
                .trimmingCharacters(in: CharacterSet(charactersIn: "="))
            return open("livecontainer://open-web-page?url=\(encoded)")
        case .stosDebug:
            return open("stosdebug://enableJIT?bundleId=\(bundle)&appName=Lichii&pid=\(pid)")
        case .trollStore:
            return open("apple-magnifier://enable-jit?bundle-id=\(bundle)")
        }
    }

    @MainActor private func open(_ url: String) -> Bool {
        url.withCString { lichii_jit_request_url($0) }
    }

    @MainActor private func canOpen(_ url: String) -> Bool {
        guard let url = URL(string: url) else { return false }
        return UIApplication.shared.canOpenURL(url)
    }
}

/// Snapshot of the JIT pool state (AppController.jit keeps the current one).
struct JitStatus: Equatable {
    var state: JitState = .waiting
    var prepared: UInt64 = 0
    var total: UInt64 = 0
    var error = ""
    var debugged = false
    var needsScript = false
    /// The chosen JIT app (JitProvider.current) is installed.
    var providerInstalled = false
    var provider = JitProvider.stikDebug

    var stikDebug: Bool { providerInstalled }

    @MainActor static func read() -> JitStatus {
        read(installed: JitProvider.current.installed)
    }

    /// Without asking iOS whether the JIT app is installed (canOpenURL is an IPC).
    @MainActor static func read(installed: Bool) -> JitStatus {
        JitStatus(state: Bridge.jitState, prepared: lichii_jit_prepared_bytes(),
                  total: lichii_jit_pool_bytes(), error: Bridge.jitError,
                  debugged: lichii_jit_debugged(), needsScript: lichii_jit_needs_script(),
                  providerInstalled: installed, provider: JitProvider.current)
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
            if providerInstalled {
                return "Lichii asks \(provider.label) to enable JIT. Confirm in \(provider.label) if it asks; Lichii continues by itself."
            }
            if provider == .stikDebug || provider == .stikDebugLiveContainer {
                return "Install StikDebug 3.2 or newer (SideStore or AltStore), import your pairing file in it, then enable JIT for Lichii (script universal.js on iOS 26 and newer)."
            }
            return "\(provider.label) isn't installed. Install it or pick another JIT app below."
        case .acquiring:
            return "\(prepared >> 20) of \(total >> 20) MiB prepared."
        case .ready:
            return "Games start right away."
        case .failed:
            let restart = "Swipe Lichii away in the app switcher and open it again."
            return error.isEmpty ? restart : "\(error)\n\(restart)"
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
    @AppStorage("autoRequestJit") private var autoRequest = true
    @AppStorage("jitProvider") private var provider: JitProvider = .stikDebug

    private var status: JitStatus { app.jit }

    var body: some View {
        List {
            Section {
                JitStatusCard(status: status)
                if status.state == .waiting {
                    Button {
                        app.requestJitIfNeeded(automatic: false)
                    } label: {
                        Label("Enable JIT with \(provider.label)", systemImage: "bolt")
                    }
                    .disabled(!status.providerInstalled)
                }
                if status.state == .failed {
                    NavigationLink {
                        LogsView()
                    } label: {
                        Label("Share the Log", systemImage: "doc.text.magnifyingglass")
                    }
                }
            }
            Section {
                Picker("JIT app", selection: $provider) {
                    ForEach(JitProvider.allCases) { Text($0.label).tag($0) }
                }
            } footer: {
                if status.needsScript && provider != .stikDebug && provider != .stikDebugLiveContainer {
                    Text("This device needs StikDebug's universal.js script; other JIT apps cannot prepare Lichii's memory here.")
                }
            }
            Section {
                // Diagnostics, collapsed: for bug reports.
                DisclosureGroup("Diagnostics") {
                    LabeledContent(provider.label, value: status.providerInstalled ? "Installed" : "Not found")
                    LabeledContent("Debugger attached", value: status.debugged ? "Yes" : "No")
                    LabeledContent("Needs StikDebug's script", value: status.needsScript ? "Yes (TXM device)" : "No")
                    LabeledContent("JIT memory prepared", value: "\(status.prepared >> 20) of \(status.total >> 20) MiB")
                }
            }
            Section {
                Toggle("Enable JIT Automatically", isOn: $autoRequest)
            } footer: {
                Text("When Lichii opens without JIT, it switches to \(provider.label) once to enable it and comes back, so games start without waiting. Turn this off to enable JIT only with the button above or when you start a game.")
            }
        }
        .navigationTitle("JIT")
        .onChange(of: provider) { _ in app.refreshJit(checkProvider: true) }
    }
}
