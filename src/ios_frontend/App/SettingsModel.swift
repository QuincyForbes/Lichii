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
    /// The enum's own name ("Lsfg").
    let canonical: String
}

/// One setting as the bridge and `common/ui_catalog` describe it.
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
    let pairedKey: String?
    let choices: [SettingChoice]
    /// The game this setting belongs to (Ocarina of Time's own...), 0 for every game.
    let game: UInt64

    var id: String { key }
    var label: String { SettingLabels.title(for: key) }

    init(_ info: LichiiSettingInfo) {
        let key = Bridge.constant(info.key)
        self.key = key
        let enumName = Bridge.constant(info.category_enum_name)
        let names = SettingsCatalog.categoryNames
        category = !enumName.isEmpty ? enumName :
            (Int(info.category_id) < names.count ? names[Int(info.category_id)] : Bridge.constant(info.category))
        kind = SettingKind(rawValue: info.kind) ?? .string
        switchable = info.switchable
        runtimeModifiable = info.runtime_modifiable
        ranged = info.ranged
        percentage = info.percentage
        minimum = info.minimum
        maximum = info.maximum
        let paired = Bridge.constant(info.paired_key)
        pairedKey = paired.isEmpty ? nil : paired
        game = key.withCString { lichii_settings_game($0) }
        let catalogChoices = UiText.choices(for: key)
        if !catalogChoices.isEmpty {
            choices = catalogChoices
        } else {
            choices = (0..<info.enum_count).map { index in
                key.withCString { pointer in
                    let name = Bridge.constant(lichii_settings_enum_name(pointer, index))
                    return SettingChoice(
                        name: SettingLabels.choice(name, for: key),
                        value: String(lichii_settings_enum_value(pointer, index)),
                        canonical: name
                    )
                }
            }
        }
    }

    /// The name of a value of this setting ("56%", "On").
    func displayName(of value: String) -> String {
        if kind == .bool {
            return UiText.text(value == "true" ? "value.on" : "value.off")
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
        UiText.text("page.\(rawValue)")
    }

    var systemImage: String {
        switch self {
        case .graphics: return "display"
        case .audio: return "speaker.wave.2.fill"
        case .system: return "cpu"
        case .advanced: return "wrench.and.screwdriver.fill"
        }
    }

    @MainActor
    var categories: [String] {
        switch self {
        case .graphics:
            return [SettingsCatalog.resolutionCategory, SettingsCatalog.upscalingCategory, "Renderer"] + // catalog-check: ok
                SettingsCatalog.gameCategories + ["RendererAdvanced", "RendererMemory", SettingsCatalog.enhancementsCategory] // catalog-check: ok
        case .audio: return ["Audio", "SystemAudio"] // catalog-check: ok
        case .system: return ["Core", "Cpu", "System", "Controls", "LibraryApplet", "DataStorage", "Overlay"] // catalog-check: ok
        case .advanced:
            return [SettingsCatalog.experimentsCategory] +
                SettingsCatalog.advancedCategories.subtracting([SettingsCatalog.experimentsCategory]).sorted()
        }
    }

    /// Section titles within a page.
    func sectionTitle(_ category: String) -> String {
        SettingsCatalog.title(of: category)
    }
}

enum SettingsCatalog {
    /// Every setting, read once (the list never changes while the app runs).
    @MainActor static let all: [SettingItem] = (0..<lichii_settings_count()).compactMap { index in
        var info = LichiiSettingInfo()
        return lichii_settings_info(index, &info) ? SettingItem(info) : nil
    }

    @MainActor static let byKey: [String: SettingItem] =
        Dictionary(all.map { ($0.key, $0) }, uniquingKeysWith: { first, _ in first })

    /// Settings::Category in declaration order (common/settings_common.h).
    static let categoryNames = [
        "Android", "Audio", "Core", "Cpu", "CpuDebug", "CpuUnsafe", "Overlay", "Renderer", // catalog-check: ok
        "RendererAdvanced", "RendererHacks", "RendererExtensions", "RendererDebug", "System", // catalog-check: ok
        "SystemAudio", "DataStorage", "Debugging", "DebuggingGraphics", "GpuDriver", "Miscellaneous", // catalog-check: ok
        "Network", "WebService", "AddOns", "Controls", "Ui", "UiAudio", "UiGeneral", "UiLayout", // catalog-check: ok
        "UiGameList", "Screenshots", "Shortcuts", "Multiplayer", "Services", "Paths", "LibraryApplet", // catalog-check: ok
        "RendererMemory", // catalog-check: ok
    ]

    /// Shown first, in this order.
    static let essentials = [
        "resolution_setup", "oot_graphics_preset", "oot_resolution", "texture_quality", "upscaler", "scaling_filter",
        "fsr_sharpening_slider", "anti_aliasing", "oot_frame_rate", "mkw_frame_rate", "frame_gen", "frame_gen_type", "frame_gen_multiplier", "use_vsync", "frame_pacing_mode",
        "use_speed_limit", "speed_limit", "use_docked_mode", "gpu_accuracy", "cpu_backend", "cpu_accuracy",
        "use_asynchronous_shaders", "use_disk_shader_cache", "aspect_ratio", "volume", "audio_muted",
    ]

    /// INI groups that only matter to desktop frontends or have no effect on iOS.
    static let hiddenCategories: Set<String> = [
        "Android", "Ui", "UiGeneral", "UiLayout", "UiGameList", "UiAudio", "Shortcuts", "Multiplayer", // catalog-check: ok
        "WebService", "Paths", "Screenshots", "Network", "Services", "GpuDriver", "AddOns", // catalog-check: ok
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

    /// Kept off the main page: debug switches and hacks that can stop games from starting.
    static let advancedCategories: Set<String> = [
        "RendererHacks", "RendererExtensions", "RendererDebug", "CpuDebug", "CpuUnsafe", "Debugging", // catalog-check: ok
        "DebuggingGraphics", "Miscellaneous", experimentsCategory, // catalog-check: ok
    ]

    /// Switches to compare two ways of doing something on the device (Settings > Advanced).
    static let experimentsCategory = "Experiments" // catalog-check: ok

    static let vulkanOnlyKeys: Set<String> = [
        "moltenvk_driver", "mesh_direct_outputs", "use_vulkan_driver_pipeline_cache",
        "async_presentation",
    ]

    @MainActor
    static func currentValue(of key: String) -> String {
        key.withCString { Bridge.string(lichii_settings_get($0)) }
    }

    @MainActor
    static func currentCanonical(of key: String) -> String {
        let raw = currentValue(of: key)
        return byKey[key]?.choices.first { $0.value == raw }?.canonical ?? ""
    }

    @MainActor
    static func currentChoiceLabel(of key: String) -> String? {
        let raw = currentValue(of: key)
        return byKey[key]?.choices.first { $0.value == raw }?.name
    }

    @MainActor
    static func availability(of item: SettingItem) -> SettingAvailability {
        if !UiText.settingShown(item.key) {
            if let reason = UiText.settingReason(item.key), !reason.isEmpty {
                return .unavailable(reason)
            }
            return .hidden
        }
        if hiddenCategories.contains(item.category) || hiddenKeys.contains(item.key) {
            return .hidden
        }
        let platform = SettingsPlatform.current
        if item.key.hasPrefix("frame_gen") {
            if !platform.frameGenBuilt {
                return .unavailable(UiText.text("reason.frame_gen_build"))
            }
            if !platform.frameGenCapable {
                return .unavailable(UiText.text("reason.frame_gen_chip"))
            }
        }
        if item.key.hasPrefix("post_shader"), !platform.postShadersBuilt {
            return .unavailable(UiText.text("reason.reshade"))
        }
        if item.key == "moltenvk_driver" || item.key == "mesh_direct_outputs", !platform.meshMoltenVkBundled {
            return .unavailable(UiText.text("reason.legacy_moltenvk"))
        }
        if vulkanOnlyKeys.contains(item.key) || item.category == "RendererExtensions",
           currentCanonical(of: "backend") != "Vulkan" {
            return .unavailable(UiText.text("experiments.conflict.requires_vulkan"))
        }
        return .available
    }

    /// Whether a setting whose parent toggle or mode controls its visibility should be listed now.
    @MainActor
    static func isVisibleWithDependencies(_ item: SettingItem) -> Bool {
        if let pairedKey = item.pairedKey, UiText.settingShown(pairedKey),
           currentValue(of: pairedKey) != "true" {
            return false
        }
        if item.key == "frame_gen_flow_scale_auto" || item.key == "frame_gen_flow_scale" {
            if currentCanonical(of: "frame_gen_type") != "Lsfg" {
                return false
            }
            if item.key == "frame_gen_flow_scale", currentValue(of: "frame_gen_flow_scale_auto") == "true" {
                return false
            }
        }
        return true
    }

    /// Non-nil when the setting's control should be disabled (empty string when the help text
    /// already explains why, otherwise the reason to display under the control).
    @MainActor
    static func disabledReason(for item: SettingItem) -> String? {
        switch item.key {
        case "frame_gen_multiplier":
            return (Int(currentValue(of: "frame_gen_target_rate")) ?? 0) > 60 ? "" : nil
        case "mesh_direct_outputs":
            return currentCanonical(of: "moltenvk_driver") == "Legacy"
                ? UiText.text("experiments.conflict.requires_mesh_mvk") : nil
        default:
            return nil
        }
    }

    /// Dynamic interaction note when another active setting clamps or overrides this one.
    @MainActor
    static func interactionNote(for item: SettingItem, value: String) -> (text: String, highlight: Bool)? {
        switch item.key {
        case "resolution_setup":
            let upscaler = currentCanonical(of: "upscaler")
            guard ["MetalFx", "Fsr3", "Fsr4", "Dlss"].contains(upscaler) else { return nil }
            let canonical = item.choices.first { $0.value == value }?.canonical ?? ""
            return (UiText.text("note.upscaler_locks_scale"), canonical != "Res1X")
        case "oot_resolution":
            let upscaler = currentCanonical(of: "upscaler")
            let metalfxBlocked = currentCanonical(of: "backend") == "Metal" &&
                PlatformCenter.shared.isExperimentEnabled("metal_temporal_off")
            let temporalActive = ["Fsr3", "Fsr4", "Dlss"].contains(upscaler) ||
                (!metalfxBlocked && ["Auto", "MetalFx"].contains(upscaler))
            guard temporalActive else { return nil }
            let canonical = item.choices.first { $0.value == value }?.canonical ?? ""
            let clamped = ["Game", "Percent150", "Percent200", "Percent300"].contains(canonical)
            return (UiText.text("note.oot_upscaler_dynamic"), clamped)
        default:
            return nil
        }
    }

    /// A hint under a usable setting: about its value where that matters, else what it does.
    @MainActor
    static func note(for item: SettingItem, value: String) -> String? {
        switch item.key {
        case "frame_gen_type":
            let canonical = item.choices.first { $0.value == value }?.canonical
            if canonical == "Lsfg", !SettingsPlatform.current.losslessInstalled {
                return UiText.text("note.lsfg_missing")
            }
            if canonical == "MetalFx" {
                return UiText.text("note.metalfx_frame_gen")
            }
            return UiText.settingHelp(item.key)
        default:
            return UiText.settingHelp(item.key)
        }
    }

    /// The settings a search finds that do nothing here, as one footer naming each and why.
    @MainActor
    private static func unavailableSection(query: String, perGame: Bool) -> SettingSection? {
        guard !query.isEmpty else { return nil }
        let lines = all.compactMap { item -> String? in
            guard !perGame || item.switchable,
                  item.label.localizedCaseInsensitiveContains(query) || item.key.localizedCaseInsensitiveContains(query) else {
                return nil
            }
            switch availability(of: item) {
            case .unavailable(let reason):
                return "\(item.label): \(reason)"
            case .available where !isVisibleWithDependencies(item):
                return "\(item.label): \(UiText.text("search.hidden_now"))"
            default:
                return nil
            }
        }
        guard !lines.isEmpty else { return nil }
        let platformName = Theme.isPad ? "iPadOS" : "iOS"
        return SettingSection(id: "unavailable", title: UiText.format("search.unavailable", platformName),
                              items: [], footer: lines.joined(separator: "\n"))
    }

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

    static let resolutionCategory = "Resolution" // catalog-check: ok
    static let upscalingCategory = "Upscaling" // catalog-check: ok

    /// Built-in game patches by kind (Settings > Graphics > Game Enhancements).
    static let enhancementsCategory = "GameEnhancements" // catalog-check: ok

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
        if category == "RendererMemory" { // catalog-check: ok
            let key = "category.RendererMemory"
            let resolved = UiText.text(key)
            return resolved == key ? UiText.text("experiments.memory.title") : resolved
        }
        let key = "category.\(category)"
        let resolved = UiText.text(key)
        return resolved == key ? category : resolved
    }

    /// Explains the Game Enhancements rows, listing the games that ship such patches.
    static func enhancementsFooter() -> String {
        let games = ContentBridge.builtinPatchGames()
        let list = games.isEmpty ? UiText.text("list.none_yet") : games.joined(separator: ", ")
        return UiText.format("category.GameEnhancements.footer", list)
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
                isVisibleWithDependencies(item) &&
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
        if category.hasPrefix("Game:") {
            return UiText.text("category.Game.footer")
        }
        switch category {
        case enhancementsCategory:
            return enhancementsFooter()
        case resolutionCategory:
            let games = gameCategories.filter { id in
                all.contains { displayCategory(of: $0) == id && $0.key.hasSuffix("_resolution") }
            }.map(title(of:))
            return games.isEmpty ? nil : UiText.format("category.Resolution.footer", games.joined(separator: ", "))
        case upscalingCategory:
            return UiText.text("category.Upscaling.footer")
        case experimentsCategory:
            return UiText.text("category.Experiments.footer")
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
            availability(of: item) == .available &&
                isVisibleWithDependencies(item) &&
                (!perGame || item.switchable) &&
                (!perGame || item.key.withCString { lichii_settings_applies($0, programId) }) &&
                (query.isEmpty ? advancedCategories.contains(displayCategory(of: item)) == advanced
                    : item.label.localizedCaseInsensitiveContains(query) ||
                        item.key.localizedCaseInsensitiveContains(query))
        }
        var sections: [SettingSection] = []
        let byKey = Dictionary(visible.map { ($0.key, $0) }, uniquingKeysWith: { first, _ in first })
        let top = essentials.compactMap { byKey[$0] }
        if !top.isEmpty {
            sections.append(SettingSection(id: "essentials", title: UiText.text("category.Essentials"), items: top))
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

/// Setting and enum labels backed by `common/ui_catalog`, with fallback derivation for unknown keys.
enum SettingLabels {
    static func title(for key: String) -> String {
        if let label = UiText.settingLabel(key), !label.isEmpty {
            return label
        }
        var words = key.split(separator: "_").map(String.init)
        if words.first == "use" || words.first == "enable" {
            words.removeFirst()
        }
        let text = words.joined(separator: " ")
        return text.prefix(1).uppercased() + text.dropFirst()
    }

    static func choice(_ name: String, for key: String) -> String {
        if let mapped = UiText.choiceLabel(key, name), !mapped.isEmpty {
            return mapped
        }
        return name.replacingOccurrences(of: "_", with: " ")
    }
}
