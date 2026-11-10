// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import LichiiBridge
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// First-run setup: welcome, keys, firmware, JIT, games. Shown again when the steps change
/// (`version`) and from Settings.
@MainActor
enum Onboarding {
    static let version = 1
    private static let key = "onboardingVersion"

    static var completed: Bool { UserDefaults.standard.integer(forKey: key) >= version }

    static func presentIfNeeded() {
        guard !completed else { return }
        // The library window has to be on screen before something is presented over it.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            MainActor.assumeIsolated { present() }
        }
    }

    static func present() {
        guard let presenter = GuestDialogs.topViewController() else { return }
        let view = OnboardingView {
            UserDefaults.standard.set(version, forKey: key)
            presenter.dismiss(animated: true)
            AppController.shared.library.reload()
        }
        .environmentObject(AppController.shared.library)
        let host = UIHostingController(rootView: view)
        host.modalPresentationStyle = .fullScreen
        host.isModalInPresentation = true
        presenter.present(host, animated: true)
    }
}

private enum OnboardingPage: Int, CaseIterable {
    case welcome, keys, firmware, jit, games
}

struct OnboardingView: View {
    let finish: () -> Void
    @EnvironmentObject private var library: GameLibrary
    @StateObject private var files = SystemFilesModel()
    @State private var page: OnboardingPage = .welcome
    @State private var importer: OnboardingPage?
    @State private var jitState = Bridge.jitState
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 0) {
            ProgressView(value: Double(page.rawValue), total: Double(OnboardingPage.allCases.count - 1))
                .tint(Theme.accent)
                .padding(.horizontal)
                .padding(.top, 8)
            ScrollView {
                VStack(spacing: 16) {
                    switch page {
                    case .welcome: welcome
                    case .keys: keys
                    case .firmware: firmware
                    case .jit: jit
                    case .games: games
                    }
                }
                .frame(maxWidth: 560)
                .padding(24)
                .frame(maxWidth: .infinity)
                .id(page)
                .transition(.backslide)
            }
            footer
        }
        .onAppear { files.refresh() }
        .onReceive(timer) { _ in jitState = Bridge.jitState }
        .fileImporter(isPresented: Binding(get: { importer != nil }, set: { if !$0 { importer = nil } }),
                      allowedContentTypes: allowedTypes,
                      allowsMultipleSelection: importer != .firmware) { result in
            let target = importer
            importer = nil
            guard case .success(let urls) = result, !urls.isEmpty else { return }
            switch target {
            case .keys: files.importKeys(urls)
            case .firmware: files.installFirmware(urls[0])
            case .games: library.importItems(urls)
            default: break
            }
        }
        .alert(item: $files.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
    }

    private var allowedTypes: [UTType] {
        switch importer {
        case .firmware: return ContentTypes.firmware
        case .games: return ImportTypes.games
        default: return ContentTypes.keys
        }
    }

    // MARK: Pages

    private var welcome: some View {
        PageHeader(systemImage: "sparkles", title: "Welcome to Lichii",
                   text: "A few steps get you playing: JIT through StikDebug and your games. Keys and firmware from your own Switch are optional: Switch 2 games (.nup) run without them.")
    }

    private var keys: some View {
        VStack(spacing: 16) {
            PageHeader(systemImage: "key.fill", title: "Keys",
                       text: "Optional. Switch 2 games (.nup) don't need keys. Encrypted Switch 1 dumps (.nsp, .xci, .nca) and firmware do: select prod.keys and title.keys dumped from your Switch with Lockpick.")
            StatusRow(title: "prod.keys", ok: files.keys?.prodKeys == true && files.keysReady,
                      detail: files.keys.map { "\($0.masterKeys) master keys" }, optional: true)
            StatusRow(title: "title.keys", ok: files.keys?.titleKeys == true,
                      detail: files.keys.map { "\($0.titleKeyCount) title keys" }, optional: true)
            ActionButton(title: files.keysReady ? "Import Again" : "Import Keys", busy: files.busy) {
                importer = .keys
            }
        }
    }

    private var firmware: some View {
        VStack(spacing: 16) {
            PageHeader(systemImage: "cpu", title: "Firmware",
                       text: "Optional. Lichii has its own replacements for the system data games use; firmware is only needed for system applets such as Mii Maker. Pick a firmware .zip or folder dumped from your Switch.")
            StatusRow(title: "Firmware", ok: files.firmwareReady,
                      detail: files.firmware.flatMap { $0.version.isEmpty ? nil : "Version \($0.version)" },
                      optional: true)
            if let progress = files.progress {
                ProgressView(value: progress)
                Button("Cancel") { files.cancel() }
            } else {
                ActionButton(title: files.firmwareReady ? "Install Again" : "Install Firmware",
                             busy: files.busy) {
                    importer = .firmware
                }
                .disabled(!files.keysReady)
                if !files.keysReady {
                    Text("Install keys first.").font(.footnote).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var jit: some View {
        VStack(spacing: 16) {
            PageHeader(systemImage: "bolt.fill", title: "JIT",
                       text: "iOS only lets an app generate code while a debugger is attached. StikDebug (with its pairing file) does that on the device: open StikDebug, pick Lichii, and come back. On iOS 26 with TXM, StikDebug's script prepares Lichii's memory as well.")
            StatusRow(title: "JIT", ok: jitState == .ready, detail: "Ready")
            if jitState != .ready {
                Text(jitDetail).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            if lichii_jit_stikdebug_installed() {
                ActionButton(title: "Enable JIT with StikDebug", busy: jitState == .acquiring) {
                    _ = lichii_jit_request()
                }
            } else {
                Text("StikDebug is not installed. Install it with SideStore or AltStore, then enable JIT for Lichii from it.")
                    .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            Text("You can skip this: games wait for JIT when you start them.")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }

    private var jitDetail: String {
        switch jitState {
        case .waiting: return "Waiting for a debugger."
        case .acquiring: return "Preparing memory..."
        case .failed: return Bridge.jitError
        case .ready: return ""
        }
    }

    private var games: some View {
        VStack(spacing: 16) {
            PageHeader(systemImage: "square.grid.2x2.fill", title: "Games",
                       text: "Import your dumps (.nup for Switch 2; .nsp, .xci, .nsz, .xcz for Switch 1). They are copied into the app, or drop them into Documents/Games with the Files app or Finder.")
            if !library.importing.isEmpty {
                ImportProgressBanner(names: library.importing)
            }
            ActionButton(title: "Import Games", busy: false) {
                importer = .games
            }
            Text("\(library.games.count) game\(library.games.count == 1 ? "" : "s") in the library")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }

    // MARK: Navigation

    private var footer: some View {
        HStack {
            if page != .welcome {
                Button("Back") { move(-1) }
            }
            Spacer()
            if page != .games && page != .welcome {
                Button("Skip") { move(1) }
                    .foregroundStyle(.secondary)
                    .padding(.trailing, 8)
            }
            ContinueButton(text: page == .games ? "Done" : "Continue",
                           action: { page == .games ? finish() : move(1) },
                           success: stepDone, enabled: .constant(!files.busy), showCheckmark: false)
        }
        .padding()
        .background(.thinMaterial)
    }

    private var stepDone: Bool {
        switch page {
        case .keys: return files.keysReady
        case .firmware: return files.firmwareReady
        case .jit: return jitState == .ready
        default: return false
        }
    }

    private func move(_ delta: Int) {
        guard let next = OnboardingPage(rawValue: page.rawValue + delta) else { return }
        withAnimation { page = next }
    }
}

private struct PageHeader: View {
    let systemImage: String
    let title: String
    let text: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 52, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 112, height: 112)
                .background(
                    Circle().fill(LinearGradient(colors: [Theme.accent, .indigo],
                                                 startPoint: .topLeading, endPoint: .bottomTrailing))
                )
                .shadow(color: .black.opacity(0.12), radius: 14, y: 6)
                .padding(.top, 24)
            Text(title).font(.largeTitle.bold())
            Text(text)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }
}

private struct ActionButton: View {
    let title: String
    let busy: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                if busy { ProgressView().padding(.trailing, 4) }
                Text(title).bold()
            }
            .frame(maxWidth: 320)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .tint(Theme.accent)
        .disabled(busy)
    }
}

/// Bordered call-to-action that turns green once its step is done.
struct ContinueButton: View {
    var text: LocalizedStringKey
    var action: () -> Void
    var success: Bool = false
    @Binding var enabled: Bool
    var showCheckmark: Bool = true
    var baseColor: Color = .green
    @Environment(\.colorScheme) private var colorScheme

    private var borderColor: Color {
        guard !success else { return baseColor }
        return enabled ? Theme.accent : Color(.systemGray4)
    }

    private var textColor: Color {
        guard !success else { return baseColor }
        return enabled ? Color.primary : Color(.systemGray4)
    }

    private var backgroundColor: Color {
        guard !success else { return baseColor.opacity(0.15) }
        return enabled ? Theme.accent.opacity(0.15) : .clear
    }

    var body: some View {
        Button {
            if enabled { action() }
        } label: {
            Text(success && showCheckmark ? "\u{2713}" : text)
                .padding(.vertical, 12)
                .padding(.horizontal, 24)
                .foregroundStyle(textColor)
                .background {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(backgroundColor)
                        .overlay(RoundedRectangle(cornerRadius: 16).stroke(borderColor, lineWidth: 1))
                }
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}

extension AnyTransition {
    static var backslide: AnyTransition {
        .asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading))
    }
}
