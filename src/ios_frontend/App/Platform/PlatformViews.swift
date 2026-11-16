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
            SettingsLink(id: "experiments", title: UiText.text("page.experiments"), systemImage: "memorychip.fill",
                         view: { AnyView(ExperimentsView()) }),
        ]
    }

    static var gameActions: [GameAction] {
        [
            GameAction(id: "shaders", title: UiText.text("page.shader_cache"), systemImage: "archivebox",
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
        let memTitle = UiText.text("experiments.memory.title")
        if let info = PlatformCenter.deviceInfo {
            let osName = info.is_ipad != 0 ? "iPadOS" : "iOS"
            LabeledContent("Model", value: "\(PlatformCenter.text(info.machine)), \(osName) \(info.os_major).\(info.os_minor)")
            LabeledContent("GPU", value: PlatformCenter.text(info.gpu_name))
            LabeledContent(memTitle, value: "\(mib(info.physical_memory_mib)), \(mib(info.available_memory_mib)) free for Lichii")
            LabeledContent("Increased Memory Limit", value: flag(info.increased_memory_limit))
            LabeledContent("Extended Virtual Addressing", value: flag(info.extended_virtual_addressing))
            if let defaults = PlatformCenter.deviceDefaults {
                LabeledContent("Performance Tier", value: PlatformCenter.text(defaults.tier_name))
                let note = PlatformCenter.text(defaults.note)
                if !note.isEmpty {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(defaults.tier == Int32(LICHII_TIER_UNSUPPORTED.rawValue) ? Color.orange : Color.secondary)
                }
            }
        } else {
            let osName = Theme.isPad ? "iPadOS" : "iOS"
            LabeledContent("Model", value: "\(UIDevice.current.model), \(osName) \(UIDevice.current.systemVersion)")
            LabeledContent(memTitle, value: UInt64(ProcessInfo.processInfo.physicalMemory).byteCount)
        }
    }
}

/// Settings > Graphics section for live thermal / Low Power Mode frame-rate pacing.
struct ThermalPacingSettingsSection: View {
    @ObservedObject private var platform = PlatformCenter.shared

    var body: some View {
        Section {
            Toggle("Auto-Cool (Thermal Pacing)", isOn: $platform.autoCool)
        } header: {
            Text("Thermals & Battery")
        } footer: {
            Text("Smoothly steps the live frame rate down to 45 FPS when iOS reports high device heat or Low Power Mode (30 FPS if critical), and restores full speed automatically once cool.")
        }
    }
}

/// The thermal advice as a banner (library) or a small badge (in-game menu). `transient` (over
/// the game): only for a few seconds after the device gets hot, and not for Low Power Mode, which
/// gets a toast instead.
struct ThermalBanner: View {
    @ObservedObject private var platform = PlatformCenter.shared
    let compact: Bool
    let transient: Bool

    init(compact: Bool = false, transient: Bool = false) {
        self.compact = compact
        self.transient = transient
    }

    var body: some View {
        let hidden = transient && (!platform.thermalBadgeVisible || platform.thermalState < 2)
        let text = hidden ? "" : compact ? platform.thermalBadge : platform.thermalMessage
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
                .accessibilityElement(children: .combine)
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

/// Library banner when the signature lacks the memory entitlements: games can't start without
/// them. SideStore only keeps them when the App ID has both capabilities.
struct MemoryEntitlementBanner: View {
    private let missing: [String] = {
        guard let info = PlatformCenter.deviceInfo else { return [] }
        var names: [String] = []
        if info.extended_virtual_addressing == 0 { names.append("Extended Virtual Addressing") }
        if info.increased_memory_limit == 0 { names.append("Increased Memory Limit") }
        return names
    }()

    var body: some View {
        if !missing.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Label("Games can't start: memory entitlements missing", systemImage: "memorychip")
                    .font(.footnote.weight(.semibold))
                Text("The installed app is not signed with \(missing.joined(separator: " and ")). At developer.apple.com, open Certificates, Identifiers & Profiles > Identifiers, select Lichii's App ID, enable Extended Virtual Addressing and Increased Memory Limit, save, then delete Lichii and install it again with SideStore.")
                    .font(.footnote)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.red.opacity(0.15),
                        in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
            .accessibilityElement(children: .combine)
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
        case Int32(LICHII_SHADER_WARM.rawValue):
            // Read every second, so the time moves on while a slow shader builds.
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                BuildShadersProgress(status: Self.buildStatus)
            }
        default:
            EmptyView()
        }
    }

    private static var buildStatus: LichiiBuildShadersStatus {
        var status = LichiiBuildShadersStatus()
        lichii_emu_build_shaders_status(&status)
        return status
    }
}

/// The shader precache (Build Shaders): done / total, built, failed, skipped and the time taken.
private struct BuildShadersProgress: View {
    let status: LichiiBuildShadersStatus

    private static func time(_ milliseconds: UInt64) -> String {
        let seconds = milliseconds / 1000
        return "\(seconds / 60):" + String(format: "%02d", seconds % 60)
    }

    var body: some View {
        VStack(spacing: 6) {
            if status.total > 0 {
                ProgressView(value: Double(min(status.done, status.total)), total: Double(status.total))
                    .tint(Theme.accent)
                Text("Building shaders \(status.done) / \(status.total)")
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
                Text("\(status.built) built · \(status.failed) failed · \(Self.time(status.elapsed_ms))"
                     + (status.skipped > 0 ? " · \(status.skipped) already built" : ""))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                if status.slowest_ms >= 20_000 {
                    Text("One shader has been building for \(status.slowest_ms / 1000) s; it is given up after 90 s.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .multilineTextAlignment(.center)
                }
            } else if status.skipped > 0 {
                Text("All \(status.skipped) shaders are already built")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("Preparing the shader build...")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// Shown at launch when the previous game session ended without a clean stop.
struct UncleanExitView: View {
    @Environment(\.dismiss) private var dismiss
    let session: GameSession
    @State private var loweredTo: String?

    /// The log of the previous run (the logger renames it at start).
    private var logFiles: [URL] {
        let current = Bridge.logFile
        let previous = URL(fileURLWithPath: current.path + ".old.txt")
        let startup = Bridge.documentsDir.appendingPathComponent("lichii-startup.old.txt")
        return [previous, startup].filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Off when the sheet opened; stays in view once switched on here.
    @State private var fileGuestWasOff = !GuestMemoryFile.enabled

    /// The run was without file_guest: offered to switch it on here.
    private var offerFile: Bool {
        session.fileGuest != true && fileGuestWasOff
    }

    private var lowerResolution: QuickSettings.Choice? {
        let choices = QuickSettings.choices("resolution_setup")
        let current = QuickSettings.string("resolution_setup")
        guard let index = choices.firstIndex(where: { $0.value == current }), index > 0 else { return nil }
        return choices[index - 1]
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Label("\(session.title) closed unexpectedly", systemImage: "exclamationmark.triangle.fill")
                        .font(.headline)
                        .foregroundStyle(.orange)
                    Text(session.memoryWarning
                         ? "iOS warned that memory was running low just before, so it most likely closed Lichii to free memory. \(offerFile ? "Keeping the game's memory in a file helps most; a lower resolution also uses less." : "A lower resolution uses less.")"
                         : "Lichii stopped while the game was running: a crash, or iOS closed it, often for memory. The log of that run helps find out which.")
                        .fixedSize(horizontal: false, vertical: true)
                }
                if offerFile {
                    GuestMemoryFileSection()
                }
                Section {
                    if !logFiles.isEmpty {
                        ShareLink(items: logFiles) {
                            Label("Share the Log", systemImage: "square.and.arrow.up")
                        }
                    }
                    if let loweredTo {
                        Label("Resolution set to \(loweredTo)", systemImage: "checkmark")
                    } else if let lower = lowerResolution {
                        Button {
                            QuickSettings.set("resolution_setup", lower.value)
                            loweredTo = lower.name
                        } label: {
                            Label("Lower the Resolution to \(lower.name)",
                                  systemImage: "arrow.down.right.and.arrow.up.left")
                        }
                    }
                } footer: {
                    Text("The resolution applies to every game without its own setting.")
                }
            }
            .navigationTitle("Last Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
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
        .navigationTitle(UiText.text("page.shader_cache"))
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
            HStack(spacing: 12) {
                SettingsIconBadge(systemImage: "archivebox.fill", color: Theme.badgeColor(for: "shaders"))
                Text("Shader Caches")
                Spacer()
                Text(bytes.map { $0.byteCount } ?? "...")
                    .foregroundStyle(.secondary)
            }
            Button("Clear All Shader Caches", role: .destructive) { confirm = true }
                .disabled(bytes == 0 || !PlatformCenter.canClearShaderCache(0))
        } header: {
            Text(UiText.text("page.storage"))
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

/// Experiments > file_guest: what it does, and the storage it needs against what is free.
/// Also offered after a game was closed by iOS.
struct GuestMemoryFileSection: View {
    @State private var enabled = GuestMemoryFile.enabled
    @State private var storage = GuestMemoryFile.storage

    var body: some View {
        let fit = GuestMemoryFile.fit(storage)
        Section {
            Toggle(isOn: $enabled) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(UiText.experimentLabel("file_guest"))
                    Text(UiText.text("experiments.large_games.subtitle"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .onChange(of: enabled) { _, value in
                GuestMemoryFile.enabled = value
            }
            .disabled(Bridge.emuState.inGame || Bridge.emuState == .booting)
            if let storage {
                LabeledContent("Free storage", value: GuestMemoryFile.size(storage.free_mib))
                LabeledContent("Needed", value: "\(GuestMemoryFile.size(storage.needed_mib)) free")
            }
            if enabled {
                Label(fitText(fit), systemImage: fit == .short ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.footnote)
                    .foregroundStyle(fit == .short ? Color.orange : fit == .unknown ? Color.secondary : Color.green)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } header: {
            Text(UiText.text("experiments.large_games.title"))
        } footer: {
            Text(UiText.format(
                "experiments.large_games.footer",
                GuestMemoryFile.size(storage?.file_mib ?? 14336),
                GuestMemoryFile.size(storage?.needed_mib ?? 10240),
                Bridge.emuState.inGame ? UiText.text("experiments.large_games.quit_first") : ""
            ))
        }
        .onAppear { storage = GuestMemoryFile.storage }
    }

    private func fitText(_ fit: GuestMemoryFile.Fit) -> String {
        switch fit {
        case .full: return "Enough free storage: the whole file is reserved when a game starts."
        case .partial: return "Enough free storage to start. With less than \(GuestMemoryFile.size(storage?.full_mib ?? 16384)) free, part of the file is reserved as the game uses it."
        case .short:
            let storageSetting = Theme.isPad ? "iPad Storage" : "iPhone Storage"
            return "Not enough free storage: games start without the file, and iOS may close Lichii in large games. Free up storage in the Settings app > General > \(storageSetting)."
        case .unknown: return "Free storage couldn't be read; Lichii checks again when a game starts."
        }
    }
}

/// One toggle row for a `UiText.ExperimentEntry`.
private struct ExperimentToggleRow: View {
    let entry: UiText.ExperimentEntry
    @State private var enabled: Bool

    init(entry: UiText.ExperimentEntry) {
        self.entry = entry
        _enabled = State(initialValue: entry.id.withCString { lichii_platform_experiment($0) != 0 })
    }

    var body: some View {
        Toggle(entry.label, isOn: $enabled)
            .onChange(of: enabled) { _, value in
                entry.id.withCString { lichii_platform_set_experiment($0, value ? 1 : 0) }
            }
    }
}

/// Opt-in switches of the platform layer, audio mixing and the per-device defaults, generated
/// from `common/ui_catalog`.
struct ExperimentsView: View {
    @ObservedObject private var platform = PlatformCenter.shared
    @State private var query = ""

    private func matchingExperiments(in section: String) -> [UiText.ExperimentEntry] {
        let items = UiText.experiments(in: section)
        guard !query.isEmpty else { return items }
        return items.filter {
            $0.label.localizedCaseInsensitiveContains(query) ||
                $0.id.localizedCaseInsensitiveContains(query)
        }
    }

    var body: some View {
        List {
            if query.isEmpty {
                GuestMemoryFileSection()
            }

            ForEach(UiText.experimentSections, id: \.self) { section in
                let items = matchingExperiments(in: section)
                if !items.isEmpty {
                    Section {
                        ForEach(items) { entry in
                            ExperimentToggleRow(entry: entry)
                        }
                    } header: {
                        Text(UiText.text("experiments.\(section).title"))
                    } footer: {
                        if query.isEmpty {
                            Text(UiText.text("experiments.\(section).footer"))
                        }
                    }
                }
            }

            if query.isEmpty {
                Section {
                    Toggle(UiText.text("experiments.mix_audio"), isOn: $platform.mixWithOtherAudio)
                } header: {
                    Text(UiText.text("experiments.audio.title"))
                } footer: {
                    Text(UiText.text("experiments.audio.footer"))
                }

                if let defaults = PlatformCenter.deviceDefaults {
                    Section {
                        LabeledContent("Tier", value: PlatformCenter.text(defaults.tier_name))
                        NavigationLink {
                            RecommendedSettingsView()
                        } label: {
                            SettingsRowLabel(
                                title: UiText.text("page.recommended"),
                                systemImage: "wand.and.stars",
                                color: Theme.badgeColor(for: "recommended")
                            )
                        }
                    } header: {
                        Text(UiText.text("section.device"))
                    } footer: {
                        Text(PlatformCenter.text(defaults.note))
                    }
                }
            }
        }
        .navigationTitle(UiText.text("page.experiments"))
        .searchable(text: $query, prompt: UiText.text("search.settings"))
    }
}
