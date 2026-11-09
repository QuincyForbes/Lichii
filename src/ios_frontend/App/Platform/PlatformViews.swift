// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

/// The platform area's entries in AppExtensions.
@MainActor
enum PlatformEntries {
    static var settingsLinks: [SettingsLink] {
        [
            SettingsLink(id: "experiments", title: "Experiments", systemImage: "flask",
                         view: { AnyView(ExperimentsView()) }),
        ]
    }

    static var gameActions: [GameAction] {
        [
            GameAction(id: "shaders", title: "Shader Cache", systemImage: "cpu",
                       view: { AnyView(ShaderCacheView(game: $0)) }),
        ]
    }
}

private func mib(_ value: UInt64) -> String {
    (value << 20).byteCount
}

private func flag(_ value: Int32) -> String {
    value > 0 ? "Yes" : value == 0 ? "No" : "Unknown"
}

/// Device rows of the settings header: model, GPU, memory budget and the defaults tier.
struct DeviceInfoRows: View {
    var body: some View {
        if let info = PlatformCenter.deviceInfo {
            LabeledContent("Device", value: "\(PlatformCenter.text(info.machine)), iOS \(info.os_major).\(info.os_minor)")
            LabeledContent("GPU", value: PlatformCenter.text(info.gpu_name))
            LabeledContent("Memory", value: "\(mib(info.physical_memory_mib)), \(mib(info.available_memory_mib)) free for Lichii")
            LabeledContent("Increased Memory Limit", value: flag(info.increased_memory_limit))
            if let defaults = PlatformCenter.deviceDefaults {
                LabeledContent("Device Tier", value: PlatformCenter.text(defaults.tier_name))
                let note = PlatformCenter.text(defaults.note)
                if !note.isEmpty {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(defaults.tier == Int32(LICHII_TIER_UNSUPPORTED.rawValue) ? Color.orange : Color.secondary)
                }
            }
        } else {
            LabeledContent("Device", value: "\(UIDevice.current.model), iOS \(UIDevice.current.systemVersion)")
            LabeledContent("Memory", value: UInt64(ProcessInfo.processInfo.physicalMemory).byteCount)
        }
    }
}

/// The thermal advice as a banner (library) or a small badge (over the game).
struct ThermalBanner: View {
    @ObservedObject private var platform = PlatformCenter.shared
    let compact: Bool

    init(compact: Bool = false) {
        self.compact = compact
    }

    var body: some View {
        let text = compact ? platform.thermalBadge : platform.thermalMessage
        if !text.isEmpty {
            Label(text, systemImage: platform.thermalState >= 2 ? "thermometer.sun.fill" : "battery.25")
                .font(compact ? Font.caption2.weight(.semibold) : Font.footnote)
                .foregroundStyle(compact ? Color.white : Color.primary)
                .padding(.horizontal, compact ? 10 : 14)
                .padding(.vertical, compact ? 5 : 10)
                .frame(maxWidth: compact ? nil : CGFloat.infinity, alignment: .leading)
                .background(tint.opacity(compact ? 0.75 : 0.18),
                            in: RoundedRectangle(cornerRadius: compact ? 12 : Theme.cornerRadius, style: .continuous))
                .allowsHitTesting(false)
        }
    }

    private var tint: Color {
        switch platform.thermalState {
        case 3...: return .red
        case 2: return .orange
        default: return .yellow
        }
    }
}

/// Disk shader cache loading on the launch screen.
struct ShaderProgressView: View {
    @ObservedObject private var platform = PlatformCenter.shared

    var body: some View {
        switch platform.shaderStage {
        case Int32(LICHII_SHADER_PREPARE.rawValue):
            Text("Preparing the shader cache...")
                .font(.footnote)
                .foregroundStyle(.secondary)
        case Int32(LICHII_SHADER_BUILD.rawValue) where platform.shaderTotal > 0:
            VStack(spacing: 6) {
                ProgressView(value: Double(min(platform.shaderDone, platform.shaderTotal)),
                             total: Double(platform.shaderTotal))
                    .tint(Theme.accent)
                Text("Loading shaders \(platform.shaderDone) / \(platform.shaderTotal)")
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        default:
            EmptyView()
        }
    }
}

/// Size and clearing of one game's shader cache.
struct ShaderCacheView: View {
    @Environment(\.dismiss) private var dismiss
    let game: Game
    @State private var bytes: UInt64?
    @State private var confirm = false
    @State private var status: String?

    var body: some View {
        List {
            Section {
                LabeledContent("Size", value: bytes.map { $0.byteCount } ?? "...")
                Button("Clear Shader Cache", role: .destructive) { confirm = true }
                    .disabled(bytes == 0 || !PlatformCenter.canClearShaderCache(game.programId))
            } footer: {
                Text(PlatformCenter.canClearShaderCache(game.programId)
                     ? "The game builds its shaders again the next time it runs, with some stutter while it does."
                     : "Quit the game to clear its shader cache.")
            }
            if let status {
                Text(status).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Shader Cache")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            }
        }
        .confirmationDialog("Clear the shader cache of \(game.title)?", isPresented: $confirm,
                            titleVisibility: .visible) {
            Button("Clear", role: .destructive) {
                let removed = PlatformCenter.clearShaderCache(game.programId)
                status = removed < 0 ? "Could not clear the shader cache." : "Removed \(removed) files."
                Task { await refresh() }
            }
        }
        .task { await refresh() }
    }

    private func refresh() async {
        bytes = await PlatformCenter.shaderCacheBytes(game.programId)
    }
}

/// Global shader cache size and "clear all" in the settings.
struct ShaderCacheSettingsSection: View {
    @State private var bytes: UInt64?
    @State private var confirm = false

    var body: some View {
        Section {
            LabeledContent("Shader Caches", value: bytes.map { $0.byteCount } ?? "...")
            Button("Clear All Shader Caches", role: .destructive) { confirm = true }
                .disabled(bytes == 0 || !PlatformCenter.canClearShaderCache(0))
        } header: {
            Text("Storage")
        } footer: {
            Text("Games build their shaders again the next time they run.")
        }
        .confirmationDialog("Clear the shader caches of every game?", isPresented: $confirm,
                            titleVisibility: .visible) {
            Button("Clear All", role: .destructive) {
                PlatformCenter.clearShaderCache(0)
                Task { bytes = await PlatformCenter.shaderCacheBytes(0) }
            }
        }
        .task { bytes = await PlatformCenter.shaderCacheBytes(0) }
    }
}

/// Opt-in switches of the platform layer, audio mixing and the per-device defaults.
struct ExperimentsView: View {
    @ObservedObject private var platform = PlatformCenter.shared
    @State private var noFootprint = lichii_platform_experiment("no_footprint") != 0
    @State private var confirmReset = false
    @State private var resetDone = false

    var body: some View {
        List {
            Section {
                Toggle("Untracked Guest Memory", isOn: $noFootprint)
                    .onChange(of: noFootprint) { value in
                        lichii_platform_set_experiment("no_footprint", value ? 1 : 0)
                    }
            } header: {
                Text("Memory")
            } footer: {
                Text("Tags the guest memory out of the app's memory footprint so iOS may close the app later. Unverified; applies the next time Lichii starts.")
            }

            Section {
                Toggle("Mix with Other Audio", isOn: $platform.mixWithOtherAudio)
            } header: {
                Text("Audio")
            } footer: {
                Text("Music or podcasts from other apps keep playing under the game.")
            }

            if let defaults = PlatformCenter.deviceDefaults {
                Section {
                    LabeledContent("Tier", value: PlatformCenter.text(defaults.tier_name))
                    LabeledContent("Resolution", value: String(format: "%gx", defaults.resolution_scale))
                    LabeledContent("Frame Generation", value: defaults.frame_gen != 0 ? "On" : "Off")
                    LabeledContent("Asynchronous Shaders", value: defaults.async_shaders != 0 ? "On" : "Off")
                    LabeledContent("Docked Mode", value: defaults.docked != 0 ? "On" : "Off")
                    Button("Reset to Device Defaults", role: .destructive) { confirmReset = true }
                        .disabled(Bridge.emuState.inGame)
                } header: {
                    Text("Device Defaults")
                } footer: {
                    Text(resetDone ? "Device defaults applied." : PlatformCenter.text(defaults.note))
                }
            }
        }
        .navigationTitle("Experiments")
        .confirmationDialog("Overwrite resolution, frame generation, accuracy and docked mode with this device's defaults?",
                            isPresented: $confirmReset, titleVisibility: .visible) {
            Button("Reset", role: .destructive) {
                lichii_platform_apply_device_defaults(Int32(LICHII_DEFAULTS_FORCE.rawValue))
                lichii_settings_save()
                resetDone = true
            }
        }
    }
}
