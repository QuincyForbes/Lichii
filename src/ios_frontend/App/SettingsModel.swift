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

    var id: String { key }
    var label: String { SettingLabels.title(for: key) }

    init(_ info: LichiiSettingInfo) {
        let key = Bridge.constant(info.key)
        self.key = key
        category = Bridge.constant(info.category)
        kind = SettingKind(rawValue: info.kind) ?? .string
        switchable = info.switchable
        runtimeModifiable = info.runtime_modifiable
        ranged = info.ranged
        percentage = info.percentage
        minimum = info.minimum
        maximum = info.maximum
        choices = (0..<info.enum_count).map { index in
            key.withCString { pointer in
                let name = Bridge.constant(lichii_settings_enum_name(pointer, index))
                return SettingChoice(name: SettingLabels.choice(name, for: key),
                                     value: String(lichii_settings_enum_value(pointer, index)))
            }
        }
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
}

enum SettingsCatalog {
    /// Every setting, read once (the list never changes while the app runs).
    @MainActor static let all: [SettingItem] = (0..<lichii_settings_count()).compactMap { index in
        var info = LichiiSettingInfo()
        return lichii_settings_info(index, &info) ? SettingItem(info) : nil
    }

    /// Shown first, in this order.
    static let essentials = [
        "resolution_setup", "scaling_filter", "fsr_sharpening_slider", "anti_aliasing",
        "metalfx_temporal", "frame_gen", "frame_gen_type", "frame_gen_multiplier", "use_vsync", "frame_pacing_mode",
        "use_speed_limit", "speed_limit", "use_docked_mode", "gpu_accuracy", "cpu_accuracy",
        "use_asynchronous_shaders", "use_disk_shader_cache", "aspect_ratio", "volume", "audio_muted",
    ]

    /// INI groups that only matter to desktop frontends.
    static let hiddenCategories: Set<String> = [
        "Android", "UI", "UiGeneral", "UiLayout", "UiGameList", "UiAudio", "Shortcuts", "Multiplayer",
        "WebService", "Paths", "Screenshots",
    ]

    static let categoryTitles: [String: String] = [
        "Renderer": "Graphics", "RendererAdvanced": "Graphics (Advanced)", "RendererHacks": "Graphics Hacks",
        "RendererExtensions": "Vulkan Extensions", "RendererDebug": "Graphics Debug", "Core": "Emulation",
        "Cpu": "CPU", "CpuDebug": "CPU Debug", "CpuUnsafe": "CPU Unsafe Optimizations", "Audio": "Audio",
        "System": "System", "SystemAudio": "System Audio", "DataStorage": "Storage", "Controls": "Input",
        "Debugging": "Debugging", "DebuggingGraphics": "Graphics Debugging", "Network": "Network",
        "Miscellaneous": "Miscellaneous", "Services": "Services", "LibraryApplet": "System Applets",
        "Overlay": "Overlay", "GpuDriver": "GPU Driver", "AddOns": "Add-ons",
    ]

    @MainActor
    static func sections(perGame: Bool, query: String) -> [SettingSection] {
        let visible = all.filter { item in
            !hiddenCategories.contains(item.category) && (!perGame || item.switchable) &&
                (query.isEmpty || item.label.localizedCaseInsensitiveContains(query) ||
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
            if grouped[item.category] == nil {
                order.append(item.category)
            }
            grouped[item.category, default: []].append(item)
        }
        for category in order {
            sections.append(SettingSection(id: category, title: categoryTitles[category] ?? category,
                                           items: grouped[category] ?? []))
        }
        return sections
    }
}

/// Readable names for keys and enum values; anything missing is derived from the key.
enum SettingLabels {
    static let titles: [String: String] = [
        "resolution_setup": "Resolution Scale",
        "scaling_filter": "Upscaling Filter",
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
        "renderer_backend": "Graphics API",
        "use_vulkan_driver_pipeline_cache": "Driver Pipeline Cache",
        "nvidia_dlss": "DLSS",
        "metalfx_temporal": "MetalFX Upscaling (replaces the game's)",
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

    /// Settings that never apply on Apple devices (greyed out), with the reason.
    static let unavailable: [String: String] = [
        "nvidia_dlss": "Windows only (NVIDIA DLSS). MetalFX Upscaling does this here.",
        "nvidia_dlss_preset": "Windows only (NVIDIA DLSS).",
    ]

    static func choice(_ name: String, for key: String) -> String {
        if let mapped = choices[key]?[name] {
            return mapped
        }
        return name.replacingOccurrences(of: "_", with: " ")
    }
}
