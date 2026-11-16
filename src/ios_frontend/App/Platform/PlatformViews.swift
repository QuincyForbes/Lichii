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
            SettingsLink(id: "experiments", title: "Memory & Experiments", systemImage: "memorychip",
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
            LabeledContent("Extended Virtual Addressing", value: flag(info.extended_virtual_addressing))
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

    /// The run was without Keep Guest Memory in a File: offered to switch it on here.
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

/// Experiments > Keep Guest Memory in a File: what it does, and the storage it needs against what
/// is free. Also offered after a game was closed by iOS.
struct GuestMemoryFileSection: View {
    @State private var enabled = GuestMemoryFile.enabled
    @State private var storage = GuestMemoryFile.storage

    var body: some View {
        let fit = GuestMemoryFile.fit(storage)
        Section {
            Toggle(isOn: $enabled) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Keep Guest Memory in a File")
                    Text("Recommended for large games")
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
            Text("Large Games")
        } footer: {
            Text("A game's own memory is most of what Lichii uses while it runs, and iOS closes Lichii when it uses too much. This moves that memory into a temporary file in storage, which iOS doesn't count: large games such as Ocarina of Time need it to get past their first areas.\n\nThe file can grow to \(GuestMemoryFile.size(storage?.file_mib ?? 14336)) and needs \(GuestMemoryFile.size(storage?.needed_mib ?? 10240)) of free storage when the game starts. Expect some stutter while a game loads, and a few GB written to storage per session (about 2.5 GB in Ocarina of Time). The file is deleted when the game stops. Takes effect the next time a game starts\(Bridge.emuState.inGame ? "; quit the game to change it" : "").")
        }
        .onAppear { storage = GuestMemoryFile.storage }
    }

    private func fitText(_ fit: GuestMemoryFile.Fit) -> String {
        switch fit {
        case .full: return "Enough free storage: the whole file is reserved when a game starts."
        case .partial: return "Enough free storage to start. With less than \(GuestMemoryFile.size(storage?.full_mib ?? 16384)) free, part of the file is reserved as the game uses it."
        case .short: return "Not enough free storage: games start without the file, and iOS may close Lichii in large games. Free up storage in the Settings app > General > iPhone Storage."
        case .unknown: return "Free storage couldn't be read; Lichii checks again when a game starts."
        }
    }
}

/// Opt-in switches of the platform layer, audio mixing and the per-device defaults.
struct ExperimentsView: View {
    @ObservedObject private var platform = PlatformCenter.shared
    @State private var untaggedGuest = lichii_platform_experiment("untagged_guest") != 0
    @State private var noZeroTrim = lichii_platform_experiment("no_zero_trim") != 0
    @State private var chunkedGuest = lichii_platform_experiment("chunked_guest") != 0
    @State private var nupInMemory = lichii_platform_experiment("nup_in_memory") != 0
    @State private var mallocCensus = lichii_platform_experiment("malloc_census") != 0
    @State private var noEarlySubmits = lichii_platform_experiment("no_early_submits") != 0
    @State private var noMeshEmulation = lichii_platform_experiment("no_mesh_emulation") != 0
    @State private var mvkUint8Indices = lichii_platform_experiment("mvk_uint8_indices") != 0
    @State private var nativePerPrimitive = lichii_platform_experiment("native_per_primitive") != 0
    @State private var mvkShaderDump = lichii_platform_experiment("mvk_shader_dump") != 0
    @State private var tcOldEviction = lichii_platform_experiment("tc_old_eviction") != 0
    @State private var helperQosOff = lichii_platform_experiment("helper_qos_off") != 0
    @State private var frameDeadlines = lichii_platform_experiment("frame_deadlines") != 0
    @State private var jitOldTables = lichii_platform_experiment("jit_old_tables") != 0
    @State private var fenceSubmitEach = lichii_platform_experiment("fence_submit_each") != 0
    @State private var renderPassesOld = lichii_platform_experiment("render_passes_old") != 0
    @State private var ootGameUpscale = lichii_platform_experiment("oot_game_upscale") != 0
    @State private var guestCoresFirst = lichii_platform_experiment("guest_cores_first") != 0
    @State private var ootNarrowAffinity = lichii_platform_experiment("oot_narrow_affinity") != 0
    @State private var ootIdealAffinity = lichii_platform_experiment("oot_ideal_affinity") != 0
    @State private var guestIdleSpin = lichii_platform_experiment("guest_idle_spin") != 0
    @State private var perfSchedOff = lichii_platform_experiment("perf_sched_off") != 0
    @State private var frameTimeOff = lichii_platform_experiment("frame_time_off") != 0
    @State private var frameTimeGuest = lichii_platform_experiment("frame_time_guest") != 0
    @State private var splitStdWaits = lichii_platform_experiment("split_std_waits") != 0
    @State private var pollYield = lichii_platform_experiment("poll_yield") != 0
    @State private var targetlessFull = lichii_platform_experiment("targetless_full") != 0
    @State private var jitInlineExclusives = lichii_platform_experiment("jit_inline_exclusives") != 0
    @State private var jitLoadFaults = lichii_platform_experiment("jit_load_faults") != 0
    @State private var cpuWritesOld = lichii_platform_experiment("cpu_writes_old") != 0
    @State private var gpuReadsOld = lichii_platform_experiment("gpu_reads_old") != 0
    @State private var uboZeroTails = lichii_platform_experiment("ubo_zero_tails") != 0
    @State private var releaseSubmit = lichii_platform_experiment("release_submit") != 0
    @State private var idleSubmitGrace = lichii_platform_experiment("idle_submit_grace") != 0
    @State private var idleHoldOff = lichii_platform_experiment("idle_hold_off") != 0
    @State private var meshComputeInPass = lichii_platform_experiment("mesh_compute_in_pass") != 0
    @State private var dispatchChain = lichii_platform_experiment("dispatch_chain") != 0
    @State private var emulateAllMesh = lichii_platform_experiment("emulate_all_mesh") != 0
    @State private var dispatchDeps = lichii_platform_experiment("dispatch_deps") != 0
    @State private var metalTraceOff = lichii_platform_experiment("metal_trace_off") != 0
    @State private var metalCheckOff = lichii_platform_experiment("metal_check_off") != 0
    @State private var metalFixedStateOff = lichii_platform_experiment("metal_fixed_state_off") != 0
    @State private var metalCachePlain = lichii_platform_experiment("metal_cache_plain") != 0
    @State private var metalMslPlain = lichii_platform_experiment("metal_msl_plain") != 0
    @State private var metalCopiesPlain = lichii_platform_experiment("metal_copies_plain") != 0
    @State private var metalClearsPlain = lichii_platform_experiment("metal_clears_plain") != 0
    @State private var metalBarrierOff = lichii_platform_experiment("metal_barrier_off") != 0
    @State private var pipelineHoldsOld = lichii_platform_experiment("pipeline_holds_old") != 0
    @State private var pipelineListOld = lichii_platform_experiment("pipeline_list_old") != 0
    @State private var mvkStatsOff = lichii_platform_experiment("mvk_stats_off") != 0

    var body: some View {
        List {
            GuestMemoryFileSection()

            Section {
                Toggle("Give Back Zeroed Guest Memory", isOn: $chunkedGuest)
                    .onChange(of: chunkedGuest) { value in
                        lichii_platform_set_experiment("chunked_guest", value ? 1 : 0)
                    }
            } header: {
                Text("Memory Savings")
            } footer: {
                Text("Untested on a device; applies the next time Lichii starts. Splits guest memory into 2 MB pieces and returns the ones holding only zeros to iOS (about 0.5 GB in Ocarina of Time). To do that safely Lichii stops all of its threads for about a millisecond at a time, which may show as a stutter. No effect while Keep Guest Memory in a File or Count Guest Memory per Mapping is on.")
            }

            Section {
                Toggle("Count Guest Memory per Mapping", isOn: $untaggedGuest)
                    .onChange(of: untaggedGuest) { value in
                        lichii_platform_set_experiment("untagged_guest", value ? 1 : 0)
                    }
                Toggle("Zero Page Trimming Off", isOn: $noZeroTrim)
                    .onChange(of: noZeroTrim) { value in
                        lichii_platform_set_experiment("no_zero_trim", value ? 1 : 0)
                    }
                Toggle("Keep Decompressed Game Files in Memory", isOn: $nupInMemory)
                    .onChange(of: nupInMemory) { value in
                        lichii_platform_set_experiment("nup_in_memory", value ? 1 : 0)
                    }
                Toggle("Count Memory Blocks in the Log", isOn: $mallocCensus)
                    .onChange(of: mallocCensus) { value in
                        lichii_platform_set_experiment("malloc_census", value ? 1 : 0)
                    }
            } header: {
                Text("Memory")
            } footer: {
                Text("Guest memory normally counts once toward the limit iOS closes Lichii at. Per mapping is the old accounting: pages both the game and the emulator touch count twice, and all-zero pages are trimmed.\n\nCompressed files of a .nup game are normally kept in memory up to 96 MB once read, and files over 48 MB (a movie, a texture archive) are unpacked to temporary files in storage instead. Keep Decompressed Game Files in Memory is the old behaviour: up to 512 MB in memory. Turn it on if a game that used to start no longer does.\n\nCount Memory Blocks in the Log writes what Lichii's own memory holds, by block size, to the log 60 s and 180 s into a game (always once when memory runs low). While it counts, threads that need memory wait.\n\nAll four apply the next time Lichii starts.")
            }

            Section {
                Toggle("Early Submits During Loads Off", isOn: $noEarlySubmits)
                    .onChange(of: noEarlySubmits) { value in
                        lichii_platform_set_experiment("no_early_submits", value ? 1 : 0)
                    }
                Toggle("Mesh Shader Emulation Off", isOn: $noMeshEmulation)
                    .onChange(of: noMeshEmulation) { value in
                        lichii_platform_set_experiment("no_mesh_emulation", value ? 1 : 0)
                    }
                Toggle("MoltenVK Converts 8-bit Indices", isOn: $mvkUint8Indices)
                    .onChange(of: mvkUint8Indices) { value in
                        lichii_platform_set_experiment("mvk_uint8_indices", value ? 1 : 0)
                    }
                Toggle("Native Per-Primitive Mesh Attributes", isOn: $nativePerPrimitive)
                    .onChange(of: nativePerPrimitive) { value in
                        lichii_platform_set_experiment("native_per_primitive", value ? 1 : 0)
                    }
                Toggle("MoltenVK Writes Its Shaders to Storage", isOn: $mvkShaderDump)
                    .onChange(of: mvkShaderDump) { value in
                        lichii_platform_set_experiment("mvk_shader_dump", value ? 1 : 0)
                    }
                Toggle("No Metal Crash Log Lines", isOn: $metalTraceOff)
                    .onChange(of: metalTraceOff) { value in
                        lichii_platform_set_experiment("metal_trace_off", value ? 1 : 0)
                    }
                Toggle("Metal Takes the Game's Draw State Unchecked", isOn: $metalCheckOff)
                    .onChange(of: metalCheckOff) { value in
                        lichii_platform_set_experiment("metal_check_off", value ? 1 : 0)
                    }
            } header: {
                Text("Graphics")
            } footer: {
                Text("For tracking down crashes. Early submits hand the GPU a scene load's texture uploads in parts, which keeps graphics memory down while a level loads; off, a large load can run out of memory. Mesh shader emulation draws some of Ocarina of Time's foliage and walls; off, they are missing. MoltenVK Converts 8-bit Indices is the old way of drawing models stored with 8-bit indices, which crashed Ocarina of Time entering Kokiri Forest; Lichii now converts them itself. Native Per-Primitive Mesh Attributes builds one of Ocarina of Time's mesh shaders the way it was built before, which the iPhone's shader compiler refused at every start; Lichii now draws it with the emulation. MoltenVK Writes Its Shaders to Storage saves every shader as MoltenVK hands it to Metal in dump/mvk-shaders (thousands of files, slower start), to find out why that one is refused; turn both on for that. No Metal Crash Log Lines is for the Metal renderer (Graphics API): normally Lichii writes a log line when the app aborts or an exception goes uncaught, and asks Metal which step of a failed command buffer failed; on, it does neither, as before. Metal Takes the Game's Draw State Unchecked is for the Metal renderer too: Metal closes the app when a draw's vertex layout, targets or a texture are outside what it takes (Ocarina of Time's first draw was), so Lichii now rewrites the vertex layout to what the shader reads, keeps 1D textures as 2D ones a pixel high and skips a draw or a texture view Metal would still reject, with a line in the log; on, Metal gets the state as the game set it, as before.\n\nAll seven apply the next time Lichii starts.")
            }

            Section {
                Toggle("Old Texture Eviction", isOn: $tcOldEviction)
                    .onChange(of: tcOldEviction) { value in
                        lichii_platform_set_experiment("tc_old_eviction", value ? 1 : 0)
                    }
                Toggle("Helper Threads at Full Priority", isOn: $helperQosOff)
                    .onChange(of: helperQosOff) { value in
                        lichii_platform_set_experiment("helper_qos_off", value ? 1 : 0)
                    }
                Toggle("Frame Deadlines for the CPU", isOn: $frameDeadlines)
                    .onChange(of: frameDeadlines) { value in
                        lichii_platform_set_experiment("frame_deadlines", value ? 1 : 0)
                    }
                Toggle("Old JIT Tables", isOn: $jitOldTables)
                    .onChange(of: jitOldTables) { value in
                        lichii_platform_set_experiment("jit_old_tables", value ? 1 : 0)
                    }
                Toggle("Submit at Every Fence", isOn: $fenceSubmitEach)
                    .onChange(of: fenceSubmitEach) { value in
                        lichii_platform_set_experiment("fence_submit_each", value ? 1 : 0)
                    }
                Toggle("Render Passes as Before", isOn: $renderPassesOld)
                    .onChange(of: renderPassesOld) { value in
                        lichii_platform_set_experiment("render_passes_old", value ? 1 : 0)
                    }
                Toggle("Game's Own Upscale Pass", isOn: $ootGameUpscale)
                    .onChange(of: ootGameUpscale) { value in
                        lichii_platform_set_experiment("oot_game_upscale", value ? 1 : 0)
                    }
                Toggle("Full-Size Passes Without Targets", isOn: $targetlessFull)
                    .onChange(of: targetlessFull) { value in
                        lichii_platform_set_experiment("targetless_full", value ? 1 : 0)
                    }
                Toggle("Zero Uniform Buffer Tails", isOn: $uboZeroTails)
                    .onChange(of: uboZeroTails) { value in
                        lichii_platform_set_experiment("ubo_zero_tails", value ? 1 : 0)
                    }
                Toggle("Submit at Every Fence Release", isOn: $releaseSubmit)
                    .onChange(of: releaseSubmit) { value in
                        lichii_platform_set_experiment("release_submit", value ? 1 : 0)
                    }
                Toggle("GPU Thread Waits for More Work", isOn: $idleSubmitGrace)
                    .onChange(of: idleSubmitGrace) { value in
                        lichii_platform_set_experiment("idle_submit_grace", value ? 1 : 0)
                    }
                Toggle("GPU Thread Submits When Idle", isOn: $idleHoldOff)
                    .onChange(of: idleHoldOff) { value in
                        lichii_platform_set_experiment("idle_hold_off", value ? 1 : 0)
                    }
                Toggle("Mesh Emulation Ends Its Pass", isOn: $meshComputeInPass)
                    .onChange(of: meshComputeInPass) { value in
                        lichii_platform_set_experiment("mesh_compute_in_pass", value ? 1 : 0)
                    }
                Toggle("Back-to-Back Dispatches Without Barriers", isOn: $dispatchChain)
                    .onChange(of: dispatchChain) { value in
                        lichii_platform_set_experiment("dispatch_chain", value ? 1 : 0)
                    }
                Toggle("Emulate Every Mesh Shader", isOn: $emulateAllMesh)
                    .onChange(of: emulateAllMesh) { value in
                        lichii_platform_set_experiment("emulate_all_mesh", value ? 1 : 0)
                    }
                Toggle("Log What Ties Dispatches to Render Passes", isOn: $dispatchDeps)
                    .onChange(of: dispatchDeps) { value in
                        lichii_platform_set_experiment("dispatch_deps", value ? 1 : 0)
                    }
            } header: {
                Text("Performance")
            } footer: {
                Text("For comparing builds. Old Texture Eviction reads textures the GPU drew back from the GPU before freeing them, a wait for the GPU each time, even when the game draws them again anyway. Helper Threads at Full Priority gives the timing and present threads the same priority as the game's. Frame Deadlines for the CPU tells iOS each frame's deadline, which may keep the game on the performance cores; untested on a device. Old JIT Tables keeps the translated code's bookkeeping as it was (about 0.3 GB more memory in Ocarina of Time); turn it on if a game that used to run crashes. Submit at Every Fence hands the GPU the game's work at every game fence, as before, instead of in batches of up to 2 ms (76-95 command buffers a frame in Ocarina of Time's Kokiri Forest on an iPhone); turn it on if the screen flashes white. Render Passes as Before ends the GPU's render pass for every buffer copy and parked clear, as before, instead of moving copies ahead of the pass (each ended pass stores and loads the whole screen again); turn it on if parts of the picture show stale or wrong data. Game's Own Upscale Pass runs Ocarina of Time's own upscale from its scene to the screen size, as before, instead of one bilinear tap whenever MetalFX Temporal doesn't take the frame (the game's pass was about 40 % of the phone's GPU time in Kokiri Forest); turn it on if the picture looks wrong. Full-Size Passes Without Targets gives a draw that writes no image a render pass of the game's full surface size, as before (32768x32768 in Ocarina of Time, about 6-7 % of the phone's GPU time), instead of only the pixels it can reach; turn it on if lens flares or glows behave differently. Zero Uniform Buffer Tails fills the unused rest of each uniform buffer a shader indexes with zeros, as before (about 75 % of the uniform bytes Lichii wrote in Kokiri Forest); turn it on if models or effects look wrong. Submit at Every Fence Release hands the GPU its work every time the game waits on a GPU value, as before, even when nothing waits for that work (about 11 command buffers a frame in Kokiri Forest); turn it on if the screen flashes or stutters. GPU Thread Waits for More Work lets the GPU thread, once it runs out of commands, wait up to 0.25 ms for more before it hands the GPU what it has (about 5 fewer command buffers a frame on a PC, but it keeps a CPU core busy meanwhile); untested on a device. GPU Thread Submits When Idle hands the GPU what the GPU thread has as soon as it runs out of commands, as before, instead of keeping it, asleep, up to 0.5 ms while the GPU still runs what it was handed earlier (about half of the phone's command buffers a frame in Kokiri Forest were such hand-overs; on a PC 2 of 3 of them went away); turn it on if the game stutters or runs slower than before. Mesh Emulation Ends Its Pass ends the render pass for each emulated mesh draw's compute, as before, instead of running that compute just before the pass; turn it on if foliage looks wrong. Back-to-Back Dispatches Without Barriers lets the game's consecutive compute dispatches share one Metal compute pass (the ~150 dispatches a frame in Ocarina of Time otherwise get a pass each); untested on a device, turn it on to compare and off if effects flicker. Emulate Every Mesh Shader draws all mesh shaders (most foliage) through the compute emulation instead of Metal mesh shaders, which hold their outputs in scarce GPU tile memory; untested on a device, for comparing the two. Log What Ties Dispatches to Render Passes writes a \"Dispatch deps\" line every 10 s naming each compute dispatch that ends the screen's render pass and whether it reads that pass's images; it changes nothing that is drawn.\n\nAll sixteen apply the next time Lichii starts.")
            }

            Section {
                Toggle("Two Guest Cores First", isOn: $guestCoresFirst)
                    .onChange(of: guestCoresFirst) { value in
                        lichii_platform_set_experiment("guest_cores_first", value ? 1 : 0)
                    }
                Toggle("Game's Own Thread Cores", isOn: $ootNarrowAffinity)
                    .onChange(of: ootNarrowAffinity) { value in
                        lichii_platform_set_experiment("oot_narrow_affinity", value ? 1 : 0)
                    }
                Toggle("One Core per Game Thread", isOn: $ootIdealAffinity)
                    .onChange(of: ootIdealAffinity) { value in
                        lichii_platform_set_experiment("oot_ideal_affinity", value ? 1 : 0)
                    }
                Toggle("Idle Guest Cores Wait Briefly", isOn: $guestIdleSpin)
                    .onChange(of: guestIdleSpin) { value in
                        lichii_platform_set_experiment("guest_idle_spin", value ? 1 : 0)
                    }
                Toggle("No CPU Scheduling Log Lines", isOn: $perfSchedOff)
                    .onChange(of: perfSchedOff) { value in
                        lichii_platform_set_experiment("perf_sched_off", value ? 1 : 0)
                    }
                Toggle("Graphics Threads Declare No Frame Time", isOn: $frameTimeOff)
                    .onChange(of: frameTimeOff) { value in
                        lichii_platform_set_experiment("frame_time_off", value ? 1 : 0)
                    }
                Toggle("Guest Cores Declare Frame Time Too", isOn: $frameTimeGuest)
                    .onChange(of: frameTimeGuest) { value in
                        lichii_platform_set_experiment("frame_time_guest", value ? 1 : 0)
                    }
                Toggle("Graphics Threads Poll Before Sleeping", isOn: $splitStdWaits)
                    .onChange(of: splitStdWaits) { value in
                        lichii_platform_set_experiment("split_std_waits", value ? 1 : 0)
                    }
                Toggle("Polling Threads Yield", isOn: $pollYield)
                    .onChange(of: pollYield) { value in
                        lichii_platform_set_experiment("poll_yield", value ? 1 : 0)
                    }
            } header: {
                Text("CPU Scheduling")
            } footer: {
                Text("For finding out why the game's CPU threads run on the efficiency cores. Two Guest Cores First leaves the top priority to two of the six emulated CPU cores and the graphics threads, and runs the other four one step lower, so iOS has fewer threads asking for its two performance cores (on a computer limited to four CPUs this gave Ocarina of Time 25 % more frames). Game's Own Thread Cores lets Ocarina of Time's main threads run only on the cores the game asks for, instead of on any core. One Core per Game Thread keeps each such thread on one core, so one host thread does its work; wins over Game's Own Thread Cores. Idle Guest Cores Wait Briefly keeps an emulated core that runs out of work polling for 20 µs before it sleeps, as on a computer, which saves a wake-up per hand-off but keeps a CPU busy. No CPU Scheduling Log Lines stops the \"Perf threads:\" and \"Perf sched:\" log lines and the counting behind them (thread listing every 10 s, a 20 Hz thread sampler, timestamps at each guest thread switch). Graphics Threads Declare No Frame Time stops Lichii's three graphics threads from telling iOS how much CPU time they need each frame, as before; with it iOS schedules them like audio threads, which on another emulator kept the performance cores faster and busier (\"frame deadlines:\" in the log's Perf line shows what each declared). Turn it on if the game stutters or runs slower than before. Guest Cores Declare Frame Time Too does the same for the emulated CPU cores; untested, and iOS may take it back from a core that never rests. Graphics Threads Poll Before Sleeping lets the two graphics threads poll for up to 64 µs before they sleep waiting on each other, as before. Polling Threads Yield lets the graphics thread that waits for the game to release a GPU value, and threads waiting for the graphics thread to send its work, give up the CPU between checks instead of sleeping 50 µs, as before; when other threads are waiting for a CPU, iOS then runs the yielding thread last for up to 10 ms.\n\nAll nine apply the next time Lichii starts.")
            }

            Section {
                Toggle("Exclusive Accesses in Fastmem", isOn: $jitInlineExclusives)
                    .onChange(of: jitInlineExclusives) { value in
                        lichii_platform_set_experiment("jit_inline_exclusives", value ? 1 : 0)
                    }
                Toggle("Loads Keep Fastmem After a Fault", isOn: $jitLoadFaults)
                    .onChange(of: jitLoadFaults) { value in
                        lichii_platform_set_experiment("jit_load_faults", value ? 1 : 0)
                    }
                Toggle("Writes Next to GPU Data as Before", isOn: $cpuWritesOld)
                    .onChange(of: cpuWritesOld) { value in
                        lichii_platform_set_experiment("cpu_writes_old", value ? 1 : 0)
                    }
                Toggle("Reads of GPU Data as Before", isOn: $gpuReadsOld)
                    .onChange(of: gpuReadsOld) { value in
                        lichii_platform_set_experiment("gpu_reads_old", value ? 1 : 0)
                    }
            } header: {
                Text("Guest Memory Access")
            } footer: {
                Text("Exclusive Accesses in Fastmem runs the game's exclusive loads and stores (LDXR/STXR, for locks and counters) directly on guest memory, as on a computer, instead of a call into the emulator each; Ocarina of Time uses none on a computer, other games may. Loads Keep Fastmem After a Fault lets a game load that hit protected memory (a page the GPU wrote) keep reading guest memory directly, and take the slow path only after its eighth such fault instead of the first; on a computer it cut Ocarina of Time's slow loads of GPU pages by 70 % but saved little time. Both are off by default. The \"Perf memory callbacks:\" log line counts both.\n\nWrites Next to GPU Data as Before keeps every game write to a memory page that also holds data the GPU wrote on the slow path until the next GPU sync, as before. Normally Lichii copies the GPU's data on such a page back first and then lets the game write the page directly (on a computer this halved Ocarina of Time's slow writes in Kokiri Forest); turn it on if objects flicker or show stale data. The \"Perf CPU writes:\" log line counts these pages.\n\nReads of GPU Data as Before: when the game reads data the GPU wrote and that data is still on its way back, the graphics thread copies that memory page again and waits until the GPU has finished everything, page after page, as before. Normally the game's thread waits only for the copy that is already under way and takes the data itself, so the graphics thread keeps working (on a computer this cut the time the graphics thread was held by such reads in Kokiri Forest by 93 %); turn it on if the picture's brightness jumps or objects show stale data. The \"Perf CPU reads:\" log line counts these reads and their waits.\n\nAll four apply the next time Lichii starts.")
            }

            Section {
                Toggle("Shaders Without the Draw's State", isOn: $metalFixedStateOff)
                    .onChange(of: metalFixedStateOff) { value in
                        lichii_platform_set_experiment("metal_fixed_state_off", value ? 1 : 0)
                    }
                Toggle("Game Memory Access as Before", isOn: $metalCachePlain)
                    .onChange(of: metalCachePlain) { value in
                        lichii_platform_set_experiment("metal_cache_plain", value ? 1 : 0)
                    }
                Toggle("Shaders Translated as Before", isOn: $metalMslPlain)
                    .onChange(of: metalMslPlain) { value in
                        lichii_platform_set_experiment("metal_msl_plain", value ? 1 : 0)
                    }
                Toggle("Image Copies as Before", isOn: $metalCopiesPlain)
                    .onChange(of: metalCopiesPlain) { value in
                        lichii_platform_set_experiment("metal_copies_plain", value ? 1 : 0)
                    }
                Toggle("Clears as Before", isOn: $metalClearsPlain)
                    .onChange(of: metalClearsPlain) { value in
                        lichii_platform_set_experiment("metal_clears_plain", value ? 1 : 0)
                    }
                Toggle("Ignore the Game's Draw Barriers", isOn: $metalBarrierOff)
                    .onChange(of: metalBarrierOff) { value in
                        lichii_platform_set_experiment("metal_barrier_off", value ? 1 : 0)
                    }
            } header: {
                Text("Metal Renderer")
            } footer: {
                Text("Only for the Metal renderer (Graphics API). Shaders Without the Draw's State builds the game's shaders as before, without what the draw sets outside the shader: the depth range the game uses, the size of points, the alpha test, blending with two colours and whole-number colour targets. Turn it on if the picture got worse with a new build.\n\nGame Memory Access as Before: when the game reads or writes memory the graphics side also uses, the Metal renderer now handles it the way the Vulkan renderer does (the game's threads wait less for the graphics thread, and the switches under Guest Memory Access apply to it too). On, it handles every such read and write the slow way it did before; turn it on if objects flicker, show stale data or the game hangs with a new build.\n\nShaders Translated as Before: the game's shaders are now turned into Metal shaders with what this iPhone's GPU can do, as the Vulkan renderer's MoltenVK does it. Before, every shader that reads which copy of a model it is drawing was refused (about a fifth of Ocarina of Time's, their draws skipped), and shaders that share values between neighbouring pixels did it in groups of four. On, they are translated as before; turn it on if shaders fail to build or the picture got worse with a new build.\n\nImage Copies as Before: the Metal renderer now copies one depth image into another when the game asks for it, and copies between a compressed image and a plain one whose pixels hold a compressed block each (Ocarina of Time does both). Before, both kinds of copy were skipped. On, they are skipped again; turn it on if the game stops with a new build.\n\nClears as Before: when the game clears only a rectangle of an image, only some of its colour channels or only some stencil bits, the Metal renderer now does exactly that (Ocarina of Time clears rectangles many times a frame). Before, it cleared the whole image. A clear of a whole image now also waits for the drawing that follows it and is done as that drawing starts, instead of as a step of its own. On, it clears the whole image again, each clear as its own step; turn it on if parts of the picture go missing with a new build.\n\nIgnore the Game's Draw Barriers: when the game draws into an image and then reads that same image in a later draw, it tells the graphics chip so first. The Metal renderer now finishes the drawing done so far at that point, so the later draw reads it. Before, such a draw read the image as it was before any of that drawing. On, the barriers are ignored again; turn it on if the game got slower with a new build.\n\nAll six apply the next time Lichii starts.")
            }

            Section {
                Toggle("Held Draws Wait in the Queue", isOn: $pipelineHoldsOld)
                    .onChange(of: pipelineHoldsOld) { value in
                        lichii_platform_set_experiment("pipeline_holds_old", value ? 1 : 0)
                    }
                Toggle("Used Shader List Saved as Before", isOn: $pipelineListOld)
                    .onChange(of: pipelineListOld) { value in
                        lichii_platform_set_experiment("pipeline_list_old", value ? 1 : 0)
                    }
                Toggle("No MoltenVK Compile Times", isOn: $mvkStatsOff)
                    .onChange(of: mvkStatsOff) { value in
                        lichii_platform_set_experiment("mvk_stats_off", value ? 1 : 0)
                    }
            } header: {
                Text("Shader Builds")
            } footer: {
                Text("A draw whose shaders are not built yet either is left out for a frame or waits. Normally the build a draw waits for runs at once on two threads of its own at the waiting thread's priority, and the background builds pause meanwhile. Held Draws Wait in the Queue leaves it in line behind the other builds on the low-priority build threads, as before (waits of 0.1-2.2 s each in Kokiri Forest on an iPhone); turn it on if the game stutters more than before.\n\nLichii keeps a list of the shaders a game really drew and builds those on the loading screen. Normally the list is saved within 10 s of a new shader's first use and when Lichii leaves the foreground. Used Shader List Saved as Before saves it only when a new shader is first drawn 15 s or more after the last save or the game is stopped from the menu, so the shaders first drawn since the last save are missing from it when Lichii is closed from the app switcher.\n\nNo MoltenVK Compile Times stops MoltenVK timing its work (a few timestamps per command buffer), which the \"Perf pipelines:\" log line reports per stage.\n\nAll three apply the next time Lichii starts.")
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
                    NavigationLink("Recommended Settings") {
                        RecommendedSettingsView()
                    }
                } header: {
                    Text("Device")
                } footer: {
                    Text(PlatformCenter.text(defaults.note))
                }
            }
        }
        .navigationTitle("Memory & Experiments")
    }
}
