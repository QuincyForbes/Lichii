// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import LichiiBridge

enum SettingKind: Int32 {
    case bool = 0, int, float, enumeration, string
}

struct SettingChoice: Hashable {
    let name: String
    let value: String
    /// The enum's own name ("Lsfg"), before SettingLabels.
    let canonical: String
}

/// One setting as the bridge describes it.
struct SettingItem: Identifiable, Hashable {
    let key: String
    let category: String
    let kind: SettingKind
    let switchable: Bool
    let runtimeModifiable: Bool
    let ranged: Bool
    let percentage: Bool
    let minimum: Double
    let maximum: Double
    let choices: [SettingChoice]
    /// The game this setting belongs to (Ocarina of Time's own...), 0 for every game.
    let game: UInt64

    var id: String { key }
    var label: String { SettingLabels.title(for: key) }

    init(_ info: LichiiSettingInfo) {
        let key = Bridge.constant(info.key)
        self.key = key
        let names = SettingsCatalog.categoryNames
        category = Int(info.category_id) < names.count ? names[Int(info.category_id)] : Bridge.constant(info.category)
        kind = SettingKind(rawValue: info.kind) ?? .string
        switchable = info.switchable
        runtimeModifiable = info.runtime_modifiable
        ranged = info.ranged
        percentage = info.percentage
        minimum = info.minimum
        maximum = info.maximum
        game = key.withCString { lichii_settings_game($0) }
        let hidden = SettingLabels.hiddenChoices[key] ?? []
        let all = (0..<info.enum_count).map { index in
            key.withCString { pointer in
                let name = Bridge.constant(lichii_settings_enum_name(pointer, index))
                return SettingChoice(name: SettingLabels.choice(name, for: key),
                                     value: String(lichii_settings_enum_value(pointer, index)),
                                     canonical: name)
            }
        }.filter { !hidden.contains($0.canonical) }
        // In the order SettingLabels.choiceOrder gives, where it has one.
        if let order = SettingLabels.choiceOrder[key] {
            choices = all.sorted { (order.firstIndex(of: $0.canonical) ?? Int.max) < (order.firstIndex(of: $1.canonical) ?? Int.max) }
        } else {
            choices = all
        }
    }

    /// The name of a value of this setting ("56%", "On").
    func displayName(of value: String) -> String {
        if kind == .bool {
            return value == "true" ? "On" : "Off"
        }
        return choices.first { $0.value == value }?.name ?? value
    }

    func value() -> String {
        key.withCString { Bridge.string(lichii_settings_get($0)) }
    }

    func defaultValue() -> String {
        key.withCString { Bridge.string(lichii_settings_get_default($0)) }
    }

    @discardableResult
    func set(_ value: String) -> Bool {
        key.withCString { k in value.withCString { lichii_settings_set(k, $0) } }
    }

    var usesGlobal: Bool {
        key.withCString { lichii_settings_uses_global($0) }
    }

    func setUsesGlobal(_ global: Bool) {
        key.withCString { lichii_settings_set_use_global($0, global) }
    }
}

struct SettingSection: Identifiable {
    let id: String
    let title: String
    let items: [SettingItem]
    var footer: String? = nil
}

/// Whether a setting does anything in this build on this device. Only available settings are
/// listed.
enum SettingAvailability: Equatable {
    case available
    /// Does nothing here; a search that finds it names the reason.
    case unavailable(String)
    /// Desktop or debug only.
    case hidden
}

/// Build and device facts the settings gate on.
@MainActor
struct SettingsPlatform {
    let frameGenBuilt: Bool
    let postShadersBuilt: Bool
    /// Apple GPU family 9 (A17 Pro) or an M-series chip; true when the platform layer can't tell.
    let frameGenCapable: Bool

    /// Read once the platform layer is up (lichii_platform_init), so after the app has started.
    static let current = SettingsPlatform()

    private init() {
        frameGenBuilt = lichii_settings_frame_gen_built()
        postShadersBuilt = lichii_settings_post_shaders_built()
        frameGenCapable = PlatformCenter.deviceDefaults.map { $0.frame_gen_capable != 0 } ?? true
    }

    /// MoltenVKMesh.framework, the MoltenVK with native mesh shaders (Settings moltenvk_driver).
    var meshMoltenVkBundled: Bool {
        FileManager.default.fileExists(atPath: Bundle.main.bundleURL
            .appendingPathComponent("Frameworks/MoltenVKMesh.framework/MoltenVKMesh").path)
    }

    /// Lossless Scaling's library, which LSFG frame generation loads; the user copies it in.
    var losslessInstalled: Bool {
        FileManager.default.fileExists(atPath: Bridge.userDir
            .appendingPathComponent("lossless/Lossless.dll").path)
    }
}

/// The pages of the global settings, each a list of setting categories in section order.
enum SettingsGroup: String, CaseIterable, Identifiable {
    case graphics, audio, system, advanced

    var id: String { rawValue }

    var title: String {
        switch self {
        case .graphics: return "Graphics"
        case .audio: return "Audio"
        case .system: return "System"
        case .advanced: return "Advanced"
        }
    }

    var systemImage: String {
        switch self {
        case .graphics: return "display"
        case .audio: return "speaker.wave.2"
        case .system: return "cpu"
        case .advanced: return "wrench.and.screwdriver"
        }
    }

    @MainActor
    var categories: [String] {
        switch self {
        case .graphics:
            return [SettingsCatalog.resolutionCategory, SettingsCatalog.upscalingCategory, "Renderer"] +
                SettingsCatalog.gameCategories + ["RendererAdvanced", SettingsCatalog.enhancementsCategory]
        case .audio: return ["Audio", "SystemAudio"]
        case .system: return ["Core", "Cpu", "System", "Controls", "LibraryApplet", "DataStorage", "Overlay"]
        case .advanced:
            return [SettingsCatalog.experimentsCategory] +
                SettingsCatalog.advancedCategories.subtracting([SettingsCatalog.experimentsCategory]).sorted()
        }
    }

    /// Section titles within a page.
    func sectionTitle(_ category: String) -> String {
        switch (self, category) {
        case (.graphics, "Renderer"): return "Image"
        case (.graphics, "RendererAdvanced"): return "Performance"
        case (.graphics, SettingsCatalog.enhancementsCategory): return "Game Enhancements"
        default: return SettingsCatalog.title(of: category)
        }
    }
}

enum SettingsCatalog {
    /// Every setting, read once (the list never changes while the app runs).
    @MainActor static let all: [SettingItem] = (0..<lichii_settings_count()).compactMap { index in
        var info = LichiiSettingInfo()
        return lichii_settings_info(index, &info) ? SettingItem(info) : nil
    }

    /// Settings::Category in declaration order (common/settings_common.h). The ini group name
    /// (info.category) folds the debug, hack and unsafe groups into their parents.
    static let categoryNames = [
        "Android", "Audio", "Core", "Cpu", "CpuDebug", "CpuUnsafe", "Overlay", "Renderer",
        "RendererAdvanced", "RendererHacks", "RendererExtensions", "RendererDebug", "System",
        "SystemAudio", "DataStorage", "Debugging", "DebuggingGraphics", "GpuDriver", "Miscellaneous",
        "Network", "WebService", "AddOns", "Controls", "Ui", "UiAudio", "UiGeneral", "UiLayout",
        "UiGameList", "Screenshots", "Shortcuts", "Multiplayer", "Services", "Paths", "LibraryApplet",
    ]

    /// Shown first, in this order.
    static let essentials = [
        "resolution_setup", "oot_graphics_preset", "oot_resolution", "texture_quality", "upscaler", "scaling_filter",
        "fsr_sharpening_slider", "anti_aliasing", "oot_frame_rate", "mkw_frame_rate", "frame_gen", "frame_gen_type", "frame_gen_multiplier", "use_vsync", "frame_pacing_mode",
        "use_speed_limit", "speed_limit", "use_docked_mode", "gpu_accuracy", "cpu_accuracy",
        "use_asynchronous_shaders", "use_disk_shader_cache", "aspect_ratio", "volume", "audio_muted",
    ]

    /// INI groups that only matter to desktop frontends or have no effect on iOS.
    static let hiddenCategories: Set<String> = [
        "Android", "Ui", "UiGeneral", "UiLayout", "UiGameList", "UiAudio", "Shortcuts", "Multiplayer",
        "WebService", "Paths", "Screenshots", "Network", "Services", "GpuDriver", "AddOns",
    ]

    /// Desktop input, debugger and bookkeeping keys with no use in the iOS app.
    static let hiddenKeys: Set<String> = [
        "vulkan_device", "fullscreen_mode", "output_device", "input_device", "dump_audio_commands",
        "frame_gen_dump_flow", "frame_gen_motion", "frame_gen_flow_quality",
        "oot_fps_asked", "frizti_settings_revision", "current_user",
        "current_speed_mode", "first_launch", "censor_username", "program_args", "gamecard_inserted",
        "gamecard_current_game", "gamecard_path", "nsight_aftermath", "renderdoc_hotkey", "use_gdbstub",
        "gdbstub_port", "perform_vulkan_check", "gpu_log_level", "gpu_log_vulkan_calls",
        "gpu_log_shader_dumps", "gpu_log_memory_tracking", "gpu_log_driver_debug",
        "gpu_log_ring_buffer_size", "disable_wgi_xinput", "enable_raw_input", "controller_navigation",
        "enable_joycon_driver", "enable_procon_driver", "udp_input_servers", "enable_udp_controller",
        "pause_tas_on_load", "tas_enable", "tas_loop", "tas_show_recording_dialog", "mouse_panning",
        "mouse_panning_sensitivity", "mouse_enabled", "mouse_panning_x_sensitivity",
        "mouse_panning_y_sensitivity", "mouse_panning_deadzone_counterweight",
        "mouse_panning_decay_strength", "mouse_panning_min_decay", "emulate_analog_keyboard",
        "keyboard_enabled", "debug_pad_enabled", "touch_device", "touch_from_button_map",
        "enable_ir_sensor", "ir_sensor_device",
    ]

    /// Settings that do nothing in the iOS app: not listed, and a search that finds one says why.
    static let unavailableReasons: [String: String] = [
        "nvidia_dlss_preset": "NVIDIA DLSS needs an NVIDIA RTX GPU on Windows.",
        "backend": "iOS has one graphics API: Vulkan through MoltenVK.",
        "cpu_backend": "iOS runs games with the Dynarmic JIT only.",
        "nvdec_emulation": "Lichii decodes game videos on the CPU on iOS and sets this at every start.",
        "force_max_clock": "Only works with AMD, NVIDIA, Intel and Qualcomm drivers.",
        "enable_compute_pipelines": "Only matters for some Intel drivers on Windows.",
        "patch_old_qcom_drivers": "Only for old Qualcomm drivers on Android.",
        "disable_macro_jit": "The macro JIT exists only on x86-64 PCs.",
        "accelerate_astc": "Apple GPUs read ASTC textures directly; this is for GPUs without ASTC.",
        "astc_recompression": "Apple GPUs read ASTC textures directly; this is for GPUs without ASTC.",
        "output_engine": "Lichii plays audio through SDL on iOS.",
        "log_filter": "Lichii sets its own log filter on iOS at every start.",
        "flush_line": "On iOS every log line is written at once, so a log survives iOS closing Lichii.",
    ]

    /// Kept off the main page: debug switches and hacks that can stop games from starting.
    static let advancedCategories: Set<String> = [
        "RendererHacks", "RendererExtensions", "RendererDebug", "CpuDebug", "CpuUnsafe", "Debugging",
        "DebuggingGraphics", "Miscellaneous", experimentsCategory,
    ]

    /// Switches to compare two ways of doing something on the device (Settings > Advanced).
    static let experimentsCategory = "Experiments"

    @MainActor
    static func availability(of item: SettingItem) -> SettingAvailability {
        if hiddenCategories.contains(item.category) || hiddenKeys.contains(item.key) {
            return .hidden
        }
        if let reason = unavailableReasons[item.key] {
            return .unavailable(reason)
        }
        let platform = SettingsPlatform.current
        if item.key.hasPrefix("frame_gen") {
            if !platform.frameGenBuilt {
                return .unavailable("Frame generation isn't part of this build.")
            }
            if !platform.frameGenCapable {
                return .unavailable("Frame generation needs an A17 Pro, M-series or newer chip.")
            }
        }
        if item.key.hasPrefix("post_shader"), !platform.postShadersBuilt {
            return .unavailable("ReShade post-processing isn't part of the iOS app.")
        }
        if item.key == "moltenvk_driver" || item.key == "mesh_direct_outputs", !platform.meshMoltenVkBundled {
            return .unavailable("This build has only the legacy MoltenVK.")
        }
        return .available
    }

    /// A hint under a usable setting: about its value where that matters, else what it does.
    @MainActor
    static func note(for item: SettingItem, value: String) -> String? {
        switch item.key {
        case "frame_gen_type":
            let canonical = item.choices.first { $0.value == value }?.canonical
            if canonical == "Lsfg", !SettingsPlatform.current.losslessInstalled {
                return "LSFG needs Lossless.dll from Lossless Scaling in Lichii's eden/lossless folder (Files app). AMD FSR 3 needs nothing."
            }
            if canonical == "MetalFx" {
                return "MetalFX makes the 2x frame on iOS 26. Before iOS 26, and at 3x or 4x, AMD FSR 3 does."
            }
            return nil
        default:
            return descriptions[item.key]
        }
    }

    /// What the everyday settings do, in a line.
    static let descriptions: [String: String] = [
        "scaling_filter": "The last step: how the finished picture is scaled to the screen.",
        "anti_aliasing": "Smooths jagged edges after the game has drawn its frame.",
        "upscaler": "What takes the place of a game's own upscaler. Automatic: Ocarina of Time's own, AMD FSR 3.1 in games that call DLSS. Game's Own also runs those games' translated DLSS network. AMD FSR 3.1 and MetalFX replace both and draw at 1x; in Ocarina of Time they need a fixed Resolution, so Dynamic draws at 75% handheld (100% docked) with them. Applies the next time the game starts.",
        "frame_gen": "Adds generated frames between the game's own: smoother motion, more GPU work and slightly more input lag.",
        "use_docked_mode": "Docked renders games at their TV resolution, which costs more GPU time and memory than Handheld.",
        "gpu_accuracy": "Higher fixes graphics in some games and is slower.",
        "use_asynchronous_shaders": "Builds new shaders in the background: less stutter, but an object can be missing for a moment.",
        "use_disk_shader_cache": "Keeps built shaders in storage, so the next start of a game stutters less.",
        "memory_layout_mode": "Switch 1 games only: Switch 2 games always get 12 GB.",
        "guest_wait_backoff": "A game thread that only waits (yields or reads the clock) for 2 ms sleeps briefly, leaving the performance cores to the rest of the game.",
        "cpuopt_compact_jit_code": "About half as much translated CPU code, so the code cache holds twice as much. Applies the next time the game starts.",
        "texture_quality": "Half or Quarter loads the game's large textures with less detail: softer up close, much less memory (about 1.2 GB less in Ocarina of Time). Applies the next time the game starts.",
        "resolution_setup": "The size games draw at. Games with their own Resolution (Ocarina of Time) use that instead. Higher is sharper and costs GPU time, memory and battery.",
        "oot_graphics_preset": "Shadows, reflections and effects as the game sets them. Phone Low: a small reflection probe, no bloom or depth of field, one small sun shadow, thinner grass, 50 % resolution and Quarter textures (about a third fewer render passes). Phone Medium: a small reflection probe and two small sun shadows, 56 % and Half. Console: the game as shipped. High: sharper sun shadows. Picking one sets Resolution and Texture Quality too; applies the next time the game starts.",
        "oot_resolution": "The size the game draws at, the one resolution setting for it. Dynamic: the game's own, lower when it thinks the GPU is busy. A percentage: a fixed share of full size (handheld 100% 1920x1080, 75% 1440x810, 56% 1080x608, 50% 960x540): less GPU work and memory, softer picture. Above 100% the emulator scales it up. With MetalFX Temporal or AMD FSR 3.1 as the Upscaler, Dynamic draws at 75% handheld (100% docked): they take only that size every frame. Applies the next time the game starts.",
        "oot_frame_rate": "30 FPS saves power and heat; Unlocked follows the frame rate limit. The game keeps its normal speed at each.",
        "mkw_frame_rate": "The game's native 30 FPS mode, or dynamic 30-60 FPS (experimental). Applies the next time the game starts.",
        "moltenvk_driver": "Which MoltenVK runs Vulkan on Metal. The new one draws mesh shaders (Ocarina of Time's foliage) with the GPU's own mesh shaders; the legacy one emulates them with compute. Lichii uses the legacy one when the new one doesn't load. Applies the next time Lichii starts.",
        "mesh_direct_outputs": "New MoltenVK only. On: mesh shaders write their vertices straight to Metal's outputs, which keeps Ocarina of Time's foliage within Metal's limits. Off: they copy them at the end, as builds before November 13 did (some foliage is then drawn with compute). Applies the next time the game starts.",
    ]

    /// The settings a search finds that do nothing here, as one footer naming each and why.
    @MainActor
    private static func unavailableSection(query: String, perGame: Bool) -> SettingSection? {
        guard !query.isEmpty else { return nil }
        let lines = all.compactMap { item -> String? in
            guard !perGame || item.switchable,
                  item.label.localizedCaseInsensitiveContains(query) || item.key.localizedCaseInsensitiveContains(query),
                  case .unavailable(let reason) = availability(of: item) else { return nil }
            return "\(item.label): \(reason)"
        }
        guard !lines.isEmpty else { return nil }
        return SettingSection(id: "unavailable", title: "Not Available on iOS", items: [],
                              footer: lines.joined(separator: "\n"))
    }

    static let categoryTitles: [String: String] = [
        "Renderer": "Graphics", "RendererAdvanced": "Graphics (Advanced)", "RendererHacks": "Graphics Hacks",
        "RendererExtensions": "Vulkan Extensions", "RendererDebug": "Graphics Debug", "Core": "Emulation",
        "Cpu": "CPU", "CpuDebug": "CPU Debug", "CpuUnsafe": "CPU Unsafe Optimizations", "Audio": "Audio",
        "System": "System", "SystemAudio": "System Audio", "DataStorage": "Storage", "Controls": "Input",
        "Debugging": "Debugging", "DebuggingGraphics": "Graphics Debugging", "Network": "Network",
        "Miscellaneous": "Miscellaneous", "Services": "Services", "LibraryApplet": "System Applets",
        "Overlay": "Overlay", "GpuDriver": "GPU Driver", "AddOns": "Add-ons",
        "GameEnhancements": "Game Enhancements", "Resolution": "Resolution & Textures",
        "Upscaling": "Upscaling", "Experiments": "Experiments",
    ]

    /// Settings shown with another group than their own: shader building is a hack in the core
    /// but an everyday switch on a phone. One place each for resolution, textures and upscaling.
    static let categoryOverrides: [String: String] = [
        "use_asynchronous_shaders": "RendererAdvanced",
        "patch_performance": enhancementsCategory,
        "patch_image": enhancementsCategory,
        "resolution_setup": resolutionCategory,
        "texture_quality": resolutionCategory,
        "oot_graphics_preset": resolutionCategory,
        "max_anisotropy": resolutionCategory,
        "upscaler": upscalingCategory,
        "scaling_filter": upscalingCategory,
        "fsr_sharpening_slider": upscalingCategory,
        "anti_aliasing": upscalingCategory,
        "mesh_direct_outputs": experimentsCategory,
    ]

    static let resolutionCategory = "Resolution"
    static let upscalingCategory = "Upscaling"

    /// Built-in game patches by kind (Settings > Graphics > Game Enhancements).
    static let enhancementsCategory = "GameEnhancements"

    /// A game's own settings, under its name (Settings > Graphics).
    static func gameCategory(_ programId: UInt64) -> String {
        "Game:" + String(programId, radix: 16, uppercase: true)
    }

    @MainActor
    static var gameCategories: [String] {
        var seen: [UInt64] = []
        for item in all where item.game != 0 && !seen.contains(item.game) {
            seen.append(item.game)
        }
        return seen.map(gameCategory)
    }

    static func title(of category: String) -> String {
        if category.hasPrefix("Game:"), let id = UInt64(category.dropFirst(5), radix: 16) {
            return Bridge.constant(lichii_settings_game_name(id))
        }
        return categoryTitles[category] ?? category
    }

    /// Explains the Game Enhancements rows, listing the games that ship such patches.
    static func enhancementsFooter() -> String {
        let games = ContentBridge.builtinPatchGames()
        let list = games.isEmpty ? "none yet" : games.joined(separator: ", ")
        return "Built-in patches by kind. Each applies to every game that ships such a patch, currently: \(list). Auto keeps each patch's default for this device. A game's own choice in its Updates, DLC & Mods sheet overrides these. Changes apply the next time a game starts."
    }

    static func displayCategory(of item: SettingItem) -> String {
        if item.game != 0 {
            return gameCategory(item.game)
        }
        return categoryOverrides[item.key] ?? item.category
    }

    /// One page of the global settings. A search covers every page.
    @MainActor
    static func sections(group: SettingsGroup, query: String) -> [SettingSection] {
        let rank = Dictionary(essentials.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        let visible = all.filter { item in
            availability(of: item) == .available &&
                (query.isEmpty ? group.categories.contains(displayCategory(of: item))
                    : item.label.localizedCaseInsensitiveContains(query) ||
                        item.key.localizedCaseInsensitiveContains(query))
        }
        var grouped: [String: [SettingItem]] = [:]
        var order: [String] = query.isEmpty ? group.categories : []
        for item in visible {
            let category = displayCategory(of: item)
            if !order.contains(category) {
                order.append(category)
            }
            grouped[category, default: []].append(item)
        }
        let sections = order.compactMap { category -> SettingSection? in
            guard let items = grouped[category], !items.isEmpty else { return nil }
            // The everyday settings first, then the core's order (a stable sort).
            let sorted = items.enumerated().sorted { a, b in
                (rank[a.element.key] ?? Int.max, a.offset) < (rank[b.element.key] ?? Int.max, b.offset)
            }.map(\.element)
            return SettingSection(id: category, title: group.sectionTitle(category), items: sorted,
                                  footer: footer(of: category))
        }
        return sections + [unavailableSection(query: query, perGame: false)].compactMap { $0 }
    }

    @MainActor
    private static func footer(of category: String) -> String? {
        switch category {
        case enhancementsCategory:
            return enhancementsFooter()
        case resolutionCategory:
            let games = gameCategories.filter { id in
                all.contains { displayCategory(of: $0) == id && $0.key.hasSuffix("_resolution") }
            }.map(title(of:))
            return games.isEmpty ? nil : "\(games.joined(separator: ", ")) has its own Resolution in its section below."
        case upscalingCategory:
            return "The game's picture goes through the upscaler, then the screen filter."
        case experimentsCategory:
            return "To compare upscalers: Graphics > Upscaling > Upscaler (MetalFX Temporal or Automatic) and Screen Filter (MetalFX Spatial). The log's \"Ocarina upscaler:\" lines count the frames MetalFX Temporal drew."
        default:
            return nil
        }
    }

    /// The per-game page (`advanced` false) or its advanced part; a search covers both. Settings
    /// of other games, and those the game's own replace, are not listed.
    @MainActor
    static func sections(perGame: Bool, query: String, advanced: Bool = false,
                         programId: UInt64 = 0) -> [SettingSection] {
        let visible = all.filter { item in
            availability(of: item) == .available && (!perGame || item.switchable) &&
                (!perGame || item.key.withCString { lichii_settings_applies($0, programId) }) &&
                (query.isEmpty ? advancedCategories.contains(displayCategory(of: item)) == advanced
                    : item.label.localizedCaseInsensitiveContains(query) ||
                        item.key.localizedCaseInsensitiveContains(query))
        }
        var sections: [SettingSection] = []
        let byKey = Dictionary(visible.map { ($0.key, $0) }, uniquingKeysWith: { first, _ in first })
        let top = essentials.compactMap { byKey[$0] }
        if !top.isEmpty {
            sections.append(SettingSection(id: "essentials", title: "Essentials", items: top))
        }
        let essentialKeys = Set(essentials)
        var order: [String] = []
        var grouped: [String: [SettingItem]] = [:]
        for item in visible where !essentialKeys.contains(item.key) {
            let category = displayCategory(of: item)
            if grouped[category] == nil {
                order.append(category)
            }
            grouped[category, default: []].append(item)
        }
        for category in order {
            sections.append(SettingSection(id: category, title: title(of: category),
                                           items: grouped[category] ?? []))
        }
        if !advanced, let unavailable = unavailableSection(query: query, perGame: perGame) {
            sections.append(unavailable)
        }
        return sections
    }
}

/// Readable names for keys and enum values; anything missing is derived from the key.
enum SettingLabels {
    static let titles: [String: String] = [
        "resolution_setup": "Resolution",
        "scaling_filter": "Screen Filter",
        "upscaler": "Upscaler",
        "fsr_sharpening_slider": "FSR Sharpness",
        "anti_aliasing": "Anti-Aliasing",
        "frame_gen": "Frame Generation",
        "frame_gen_type": "Frame Generation Type",
        "frame_gen_multiplier": "Frame Generation Multiplier",
        "frame_gen_target_rate": "Frame Generation Target",
        "use_vsync": "VSync Mode",
        "frame_pacing_mode": "Frame Rate Target",
        "use_speed_limit": "Limit Emulation Speed",
        "speed_limit": "Speed Limit",
        "use_docked_mode": "Console Mode",
        "gpu_accuracy": "GPU Accuracy",
        "cpu_accuracy": "CPU Accuracy",
        "use_asynchronous_shaders": "Asynchronous Shader Building",
        "use_disk_shader_cache": "Disk Shader Cache",
        "aspect_ratio": "Aspect Ratio",
        "volume": "Volume",
        "audio_muted": "Mute Audio",
        "max_anisotropy": "Anisotropic Filtering",
        "nvdec_emulation": "Video Decoding",
        "astc_decode_mode": "ASTC Decoding",
        "memory_layout_mode": "Memory Layout",
        "use_multi_core": "Multicore CPU Emulation",
        "language_index": "Language",
        "region_index": "Region",
        "time_zone_index": "Time Zone",
        "backend": "Graphics API",
        "cpu_backend": "CPU Backend",
        "guest_wait_backoff": "Sleep in Long Guest Waits",
        "cpuopt_compact_jit_code": "Compact JIT Code",
        "nvidia_dlss_preset": "DLSS Preset",
        "oot_anti_aliasing": "Sharpening",
        "oot_movie_upscale": "Movies",
        "texture_quality": "Texture Quality",
        "oot_graphics_preset": "Graphics Preset",
        "oot_resolution": "Resolution",
        "oot_frame_rate": "Frame Rate",
        "mkw_frame_rate": "Frame Rate",
        "moltenvk_driver": "Graphics Driver",
        "mesh_direct_outputs": "Mesh Shader Direct Outputs",
        "nx2_frame_rate_limit": "Frame Rate Limit (Switch 2)",
        "accelerate_astc": "ASTC Decoding",
        "astc_recompression": "ASTC Recompression",
        "output_engine": "Audio Output Engine",
        "force_max_clock": "Force Maximum GPU Clocks",
        "use_vulkan_driver_pipeline_cache": "Driver Pipeline Cache",
        "patch_performance": "Occlusion Off",
        "patch_image": "Disable Game Anti-Aliasing",
        "device_name": "Console Name",
        "log_filter": "Log Filter",
    ]

    static func title(for key: String) -> String {
        if let title = titles[key] {
            return title
        }
        var words = key.split(separator: "_").map(String.init)
        if words.first == "use" || words.first == "enable" {
            words.removeFirst()
        }
        let text = words.joined(separator: " ")
        return text.prefix(1).uppercased() + text.dropFirst()
    }

    static let choices: [String: [String: String]] = [
        "resolution_setup": [
            "Res1_4X": "0.25x", "Res1_2X": "0.5x", "Res3_4X": "0.75x", "Res1X": "1x (Native)",
            "Res5_4X": "1.25x", "Res3_2X": "1.5x", "Res2X": "2x", "Res3X": "3x", "Res4X": "4x",
            "Res5X": "5x", "Res6X": "6x", "Res7X": "7x", "Res8X": "8x",
        ],
        "use_docked_mode": ["Handheld": "Handheld", "Docked": "Docked"],
        "texture_quality": ["Full": "Full", "Half": "Half (less memory)", "Quarter": "Quarter (least memory)"],
        "oot_resolution": [
            "Game": "Dynamic (the game's own)", "Full": "100%", "Percent87": "87%", "Percent75": "75%",
            "Percent56": "56% (1080x608)", "Percent50": "50%", "Percent150": "150%", "Percent200": "200%",
            "Percent300": "300%",
        ],
        "upscaler": [
            "Auto": "Automatic", "Game": "Game's Own", "Fsr3": "AMD FSR 3.1", "Dlss": "NVIDIA DLSS",
            "MetalFx": "MetalFX Temporal",
        ],
        "oot_anti_aliasing": ["Game": "Game's own", "NoSharpening": "No sharpening"],
        "oot_graphics_preset": [
            "PhoneLow": "Phone Low", "PhoneMedium": "Phone Medium", "Console": "Console", "High": "High",
        ],
        "oot_frame_rate": ["Fps30": "30 FPS", "Fps60": "60 FPS", "Unlocked": "Unlocked (frame rate limit)"],
        "mkw_frame_rate": ["Game": "Game's own", "Fps30": "30 FPS", "Dynamic": "Dynamic 30-60 FPS"],
        "moltenvk_driver": ["Legacy": "Legacy MoltenVK", "Mesh": "New MoltenVK (native mesh shaders)"],
        "use_vsync": [
            "Immediate": "Off (Immediate)", "Mailbox": "Mailbox", "Fifo": "On (FIFO)",
            "FifoRelaxed": "Relaxed FIFO",
        ],
        "frame_pacing_mode": [
            "Target_Auto": "Auto", "Target_30": "30 FPS", "Target_60": "60 FPS", "Target_90": "90 FPS",
            "Target_120": "120 FPS",
        ],
        "scaling_filter": [
            "NearestNeighbor": "Nearest Neighbor", "Fsr": "AMD FSR 1", "Fsr3": "AMD FSR 3",
            "ScaleForce": "ScaleForce", "ZeroTangent": "Zero Tangent", "BSpline": "B-Spline",
            "Mmpx": "MMPX", "Sgsr": "Snapdragon GSR", "SgsrEdge": "Snapdragon GSR (Edge)",
            "MetalFxSpatial": "MetalFX Spatial",
        ],
        "frame_gen_type": [
            "Lsfg": "Lossless Scaling (LSFG)", "Fsr3": "AMD FSR 3", "MetalFx": "MetalFX (iOS 26, 2x)",
        ],
        "aspect_ratio": [
            "R16_9": "16:9", "R4_3": "4:3", "R21_9": "21:9", "R16_10": "16:10", "Stretch": "Stretch",
        ],
    ]

    /// Values that do nothing on iOS, by their enum name; not offered.
    static let hiddenChoices: [String: Set<String>] = [
        "upscaler": ["Dlss"],
    ]

    /// Lists shown in another order than the enum's.
    static let choiceOrder: [String: [String]] = [
        "oot_resolution": [
            "Game", "Percent50", "Percent56", "Percent75", "Percent87", "Full", "Percent150",
            "Percent200", "Percent300",
        ],
    ]

    static func choice(_ name: String, for key: String) -> String {
        if let mapped = choices[key]?[name] {
            return mapped
        }
        return name.replacingOccurrences(of: "_", with: " ")
    }
}
