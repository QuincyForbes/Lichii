// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UniformTypeIdentifiers

/// Keys and firmware state shared by the settings screen and onboarding.
@MainActor
final class SystemFilesModel: ObservableObject {
    @Published private(set) var keys: ContentBridge.KeyStatus?
    @Published private(set) var firmware: ContentBridge.FirmwareStatus?
    @Published private(set) var busy = false
    @Published var progress: Double?
    @Published var alert: ContentAlert?
    @Published private(set) var miiEditAvailable = false

    private let cancelFlag = CancelFlag()

    func refresh() {
        keys = ContentBridge.keyStatus
        firmware = ContentBridge.firmwareStatus
        miiEditAvailable = ContentBridge.string(lichii_content_mii_edit_path()) != nil
    }

    var keysReady: Bool { keys?.usable == true }
    var firmwareReady: Bool { firmware?.present == true && firmware?.corrupted == false }

    func importKeys(_ urls: [URL]) {
        busy = true
        ContentBridge.run({ () -> (LichiiKeyResult, String) in
            guard let staging = makeStagingDirectory() else {
                return (LICHII_KEYS_COPY_FAILED, "Could not create a temporary folder.")
            }
            defer { try? FileManager.default.removeItem(at: staging) }
            var staged: [URL] = []
            var errors: [String] = []
            for url in urls {
                let target = staging.appendingPathComponent(url.lastPathComponent)
                if let error = coordinatedCopy(url, to: target) {
                    errors.append("\(url.lastPathComponent): \(error)")
                } else {
                    staged.append(target)
                }
            }
            if staged.isEmpty {
                return (LICHII_KEYS_NO_FILES, errors.joined(separator: "\n"))
            }
            let (code, message) = ContentBridge.installKeys(staged)
            return (code, (errors + [message]).joined(separator: "\n"))
        }) { [weak self] result in
            guard let self else { return }
            busy = false
            refresh()
            let (code, message) = result
            switch code {
            case LICHII_KEYS_OK:
                if keys?.titleKeys == false {
                    alert = ContentAlert(title: "Keys installed",
                                         message: message + "\n\ntitle.keys is recommended as well; some games need it.")
                } else {
                    alert = ContentAlert(title: "Keys installed", message: message)
                }
            case LICHII_KEYS_INCOMPLETE:
                alert = ContentAlert(title: "Keys incomplete", message: message)
            default:
                alert = ContentAlert(title: "Keys not installed", message: message)
            }
        }
    }

    func installFirmware(_ url: URL) {
        busy = true
        progress = 0
        cancelFlag.set(false)
        let flag = cancelFlag
        let progressBox = ContentBridge.Progress { [weak self] value in
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self?.progress = value }
            }
            return flag.isCancelled
        }
        ContentBridge.run({ () -> (LichiiFirmwareResult, String) in
            guard let staging = makeStagingDirectory() else {
                return (LICHII_FIRMWARE_COPY_FAILED, "Could not create a temporary folder.")
            }
            defer { try? FileManager.default.removeItem(at: staging) }
            let target = staging.appendingPathComponent(url.lastPathComponent)
            if let error = coordinatedCopy(url, to: target) {
                return (LICHII_FIRMWARE_COPY_FAILED, "\(url.lastPathComponent): \(error)")
            }
            return ContentBridge.installFirmware(target, progress: progressBox)
        }) { [weak self] result in
            guard let self else { return }
            busy = false
            progress = nil
            refresh()
            let (code, message) = result
            alert = ContentAlert(title: code == LICHII_FIRMWARE_OK ? "Firmware installed" : "Firmware not installed",
                                 message: message)
        }
    }

    func cancel() {
        cancelFlag.set(true)
    }

    func removeKeys() {
        _ = lichii_content_remove_keys()
        refresh()
    }

    func removeFirmware() {
        if !lichii_content_remove_firmware() {
            alert = ContentAlert(title: "Firmware", message: "Stop the game before removing firmware.")
        }
        refresh()
    }

    func launchMiiMaker() {
        guard let path = ContentBridge.string(lichii_content_prepare_mii_edit()) else {
            alert = ContentAlert(title: "Mii Maker", message: "Install firmware first: Mii Maker is part of it.")
            return
        }
        let app = AppController.shared
        guard path.withCString({ lichii_emu_launch($0) }) else {
            alert = ContentAlert(title: "Mii Maker", message: "Stop the running game first.")
            return
        }
        app.showLaunch = true
        if Bridge.jitState == .waiting {
            app.requestJitIfNeeded(automatic: false)
        }
    }
}

struct ContentAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

enum ContentTypes {
    static let keys: [UTType] = [.item]
    static let firmware: [UTType] = [.zip, .folder]
    static let packages: [UTType] = [.item]
    static let mods: [UTType] = [.zip, .folder]
    static let cheats: [UTType] = [.plainText, .item]
}

struct KeysFirmwareView: View {
    @StateObject private var model = SystemFilesModel()
    @State private var importer: Importer?
    @State private var confirmRemoveKeys = false
    @State private var confirmRemoveFirmware = false

    private enum Importer: Identifiable {
        case keys, firmware
        var id: Self { self }
    }

    var body: some View {
        List {
            Section {
                StatusRow(title: "prod.keys", ok: model.keys?.prodKeys == true,
                          detail: model.keys.map { "\($0.masterKeys) master keys" + ($0.headerKey ? ", header key" : "") })
                StatusRow(title: "title.keys", ok: model.keys?.titleKeys == true,
                          detail: model.keys.map { "\($0.titleKeyCount) title keys" }, optional: true)
                Button {
                    importer = .keys
                } label: {
                    Label("Import Keys...", systemImage: "key")
                }
                .disabled(model.busy)
            } header: {
                Text("Keys")
            } footer: {
                Text("Select prod.keys and title.keys dumped from your own Switch (Lockpick). They are copied to Documents/eden/keys.")
            }

            Section {
                StatusRow(title: "Firmware", ok: model.firmwareReady,
                          detail: firmwareDetail)
                if let progress = model.progress {
                    VStack(alignment: .leading) {
                        ProgressView(value: progress)
                        Button("Cancel", role: .cancel) { model.cancel() }
                            .font(.caption)
                    }
                }
                Button {
                    importer = .firmware
                } label: {
                    Label("Install Firmware...", systemImage: "cpu")
                }
                .disabled(model.busy || !model.keysReady)
            } header: {
                Text("Firmware")
            } footer: {
                Text(model.keysReady
                     ? "Pick a firmware .zip or a folder of .nca files dumped from your own console. The current firmware is replaced."
                     : "Install keys before firmware.")
            }

            Section {
                Button {
                    model.launchMiiMaker()
                } label: {
                    Label("Open Mii Maker", systemImage: "person.crop.circle.badge.plus")
                }
                .disabled(!model.miiEditAvailable)
            } footer: {
                Text("Runs the firmware's Mii Maker applet.")
            }

            Section {
                Button("Remove Keys", role: .destructive) { confirmRemoveKeys = true }
                    .disabled(model.keys?.prodKeys != true)
                Button("Remove Firmware", role: .destructive) { confirmRemoveFirmware = true }
                    .disabled(model.firmware?.present != true)
            }
        }
        .navigationTitle("Keys & Firmware")
        .onAppear { model.refresh() }
        .fileImporter(isPresented: Binding(get: { importer != nil }, set: { if !$0 { importer = nil } }),
                      allowedContentTypes: importer == .firmware ? ContentTypes.firmware : ContentTypes.keys,
                      allowsMultipleSelection: importer == .keys) { result in
            let kind = importer
            importer = nil
            guard case .success(let urls) = result, !urls.isEmpty else { return }
            if kind == .firmware {
                model.installFirmware(urls[0])
            } else {
                model.importKeys(urls)
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message))
        }
        .confirmationDialog("Remove prod.keys and title.keys?", isPresented: $confirmRemoveKeys,
                            titleVisibility: .visible) {
            Button("Remove Keys", role: .destructive) { model.removeKeys() }
        }
        .confirmationDialog("Remove the installed firmware?", isPresented: $confirmRemoveFirmware,
                            titleVisibility: .visible) {
            Button("Remove Firmware", role: .destructive) { model.removeFirmware() }
        }
    }

    private var firmwareDetail: String? {
        guard let firmware = model.firmware else { return nil }
        if firmware.corrupted { return "Installed, but unreadable: install it again" }
        if !firmware.present { return nil }
        return firmware.version.isEmpty ? "Installed" : "Version \(firmware.version)"
    }
}

struct StatusRow: View {
    let title: String
    let ok: Bool
    var detail: String?
    var optional = false

    var body: some View {
        HStack {
            Image(systemName: ok ? "checkmark.circle.fill" : (optional ? "minus.circle" : "xmark.circle.fill"))
                .foregroundStyle(ok ? .green : (optional ? .secondary : .red))
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                if let detail, ok {
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                } else if !ok {
                    Text(optional ? "Optional" : "Missing").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}
