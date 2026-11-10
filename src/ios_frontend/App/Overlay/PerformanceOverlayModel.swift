// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

/// How much the performance overlay shows. Stored by name in UserDefaults ("overlayPreset").
enum OverlayPreset: String, CaseIterable, Identifiable {
    case off, minimal, standard, detailed

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: return "Off"
        case .minimal: return "Minimal"
        case .standard: return "Standard"
        case .detailed: return "Detailed"
        }
    }

    var summary: String {
        switch self {
        case .off: return "Nothing is drawn over the game."
        case .minimal: return "Frames per second only."
        case .standard: return "Real, generated and displayed FPS, frame time, 10% low, frame generation, resolution scale, speed."
        case .detailed: return "Standard plus render and output resolution, GPU accuracy, CPU, memory, texture cache, shader builds, thermal state and battery."
        }
    }

    var next: OverlayPreset {
        let all = OverlayPreset.allCases
        return all[(all.firstIndex(of: self)! + 1) % all.count]
    }

    /// The old HUD switch ("showPerformanceHUD") counts as Standard until a preset is chosen.
    static func resolve(_ raw: String) -> OverlayPreset {
        if let preset = OverlayPreset(rawValue: raw) {
            return preset
        }
        return UserDefaults.standard.bool(forKey: "showPerformanceHUD") ? .standard : .off
    }

    static var current: OverlayPreset {
        resolve(UserDefaults.standard.string(forKey: "overlayPreset") ?? "")
    }

    static func store(_ preset: OverlayPreset) {
        UserDefaults.standard.set(preset.rawValue, forKey: "overlayPreset")
        if preset != .off {
            UserDefaults.standard.set(preset.rawValue, forKey: "overlayLastPreset")
        }
    }

    /// The controller shortcut: off, or back to the last preset in use.
    static func toggle() {
        if current != .off {
            store(.off)
        } else {
            store(OverlayPreset(rawValue: UserDefaults.standard.string(forKey: "overlayLastPreset") ?? "") ?? .standard)
        }
    }
}

extension HUDPosition {
    var horizontal: HorizontalAlignment {
        switch self {
        case .topLeft, .bottomLeft: return .leading
        case .topRight, .bottomRight: return .trailing
        }
    }
}

/// One refresh of everything the overlay can show. Values are rounded so an unchanged reading
/// compares equal and the view is not invalidated for nothing.
struct OverlayReading: Equatable {
    var active = false
    var paused = false

    var realFps = 0.0
    var generatedFps = 0.0
    var displayedFps = 0.0
    var frameTimeMs = 0.0
    var low10Fps = 0.0
    var speedPercent = 0.0

    var frameGen = "Off"
    var frameGenActive = false
    var frameGenMultiplier = 1.0
    var resolutionScale = 1.0
    var upscaler = ""
    var renderSize = ""
    var outputSize = ""
    var gpuAccuracy = ""

    var cpuPercent = -1.0
    var footprint: UInt64 = 0
    var memoryLimit: UInt64 = 0
    var textureBytes: UInt64 = 0
    var textureExpected: UInt64 = 0
    var shadersBuilding = 0

    var thermal = ProcessInfo.ThermalState.nominal
    var lowPower = false
    var battery: Float = -1
    var charging = false
}

/// Enum values of the settings bridge as short names: "frame_gen_type" 2 -> "MetalFX". A value a
/// later build adds shows its own name.
@MainActor
enum OverlayNames {
    private static var tables: [String: [Int32: String]] = [:]

    private static let short: [String: String] = [
        "Lsfg": "LSFG", "Fsr3": "FSR3", "MetalFx": "MetalFX", "MetalFxSpatial": "MetalFX Spatial",
        "Fsr": "FSR 1", "NearestNeighbor": "Nearest", "ScaleForce": "ScaleForce", "ZeroTangent": "Zero Tangent",
        "BSpline": "B-Spline", "Mmpx": "MMPX", "Sgsr": "Snapdragon GSR", "SgsrEdge": "Snapdragon GSR Edge",
    ]

    static func name(_ key: String, _ value: Int32) -> String {
        if tables[key] == nil {
            var table: [Int32: String] = [:]
            if let info = QuickSettings.info(key) {
                for index in 0..<info.enum_count {
                    let number = key.withCString { lichii_settings_enum_value($0, index) }
                    table[Int32(truncatingIfNeeded: number)] =
                        Bridge.constant(key.withCString { lichii_settings_enum_name($0, index) })
                }
            }
            tables[key] = table
        }
        guard let raw = tables[key]?[value] else { return "" }
        return short[raw] ?? raw.replacingOccurrences(of: "_", with: " ")
    }
}

/// Samples the engine and the device twice a second while an overlay is on screen. One timer,
/// one published value: the overlay's views redraw at most twice a second.
@MainActor
final class PerformanceOverlayMonitor: ObservableObject {
    static let shared = PerformanceOverlayMonitor()

    @Published private(set) var reading = OverlayReading()

    private static let historyLimit = 120
    private var timer: Timer?
    private var viewers = 0
    private var history: [Double] = []
    private let hasMetalFxTemporal = QuickSettings.info("metalfx_temporal") != nil

    func start() {
        viewers += 1
        guard timer == nil else { return }
        UIDevice.current.isBatteryMonitoringEnabled = true
        sample()
        let timer = Timer(timeInterval: 0.5, repeats: true) { _ in
            MainActor.assumeIsolated { PerformanceOverlayMonitor.shared.sample() }
        }
        timer.tolerance = 0.1
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        viewers = max(0, viewers - 1)
        guard viewers == 0 else { return }
        timer?.invalidate()
        timer = nil
        UIDevice.current.isBatteryMonitoringEnabled = false
    }

    private func sample() {
        var next = reading
        var snapshot = LichiiPerfSnapshot()
        next.active = lichii_perf_snapshot(&snapshot)
        if next.active {
            apply(snapshot, to: &next)
        }
        var system = LichiiSystemStats()
        if lichii_controls_system_stats(&system) {
            next.cpuPercent = system.cpu_percent
            next.footprint = system.memory_footprint
            next.memoryLimit = system.memory_available > 0 ? system.memory_footprint + system.memory_available : 0
        }
        next.thermal = ProcessInfo.processInfo.thermalState
        next.lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        next.battery = UIDevice.current.batteryLevel
        next.charging = UIDevice.current.batteryState == .charging || UIDevice.current.batteryState == .full
        if next != reading {
            reading = next
        }
    }

    private func apply(_ snapshot: LichiiPerfSnapshot, to next: inout OverlayReading) {
        next.paused = snapshot.paused
        // The counters reset on every read, so a paused game reads zero: keep the last numbers.
        if !snapshot.paused {
            next.realFps = Self.round(snapshot.real_fps)
            next.generatedFps = Self.round(snapshot.generated_fps)
            next.displayedFps = Self.round(snapshot.displayed_fps)
            next.frameTimeMs = Self.round(snapshot.frametime_ms > 0 ? snapshot.frametime_ms
                                          : (snapshot.real_fps > 0.1 ? 1000 / snapshot.real_fps : 0))
            next.speedPercent = (snapshot.speed * 100).rounded()
            if snapshot.real_fps > 3 {
                history.append(snapshot.real_fps)
                if history.count > Self.historyLimit {
                    history.removeFirst(history.count - Self.historyLimit)
                }
            }
            next.low10Fps = lowest10Percent().rounded()
        }

        let generating = snapshot.generated_fps > 0.5 && snapshot.real_fps > 0.5
        next.frameGenActive = generating
        next.frameGenMultiplier = generating ? snapshot.displayed_fps / snapshot.real_fps
                                             : Double(snapshot.frame_gen_multiplier)
        let type = OverlayNames.name("frame_gen_type", snapshot.frame_gen_type)
        if !snapshot.frame_gen_enabled {
            next.frameGen = "Off"
        } else if generating {
            next.frameGen = String(format: "%@ %.1fx", type, next.frameGenMultiplier)
        } else {
            next.frameGen = "\(type) warming up"
        }

        next.resolutionScale = Double(snapshot.resolution_scale)
        next.upscaler = upscaler(snapshot)
        next.renderSize = Self.size(snapshot.render_width, snapshot.render_height)
        next.outputSize = Self.size(snapshot.output_width, snapshot.output_height)
        next.gpuAccuracy = OverlayNames.name("gpu_accuracy", snapshot.gpu_accuracy)
        next.textureBytes = snapshot.texture_cache_bytes
        next.textureExpected = snapshot.texture_cache_expected
        next.shadersBuilding = Int(snapshot.shaders_building)
    }

    private func upscaler(_ snapshot: LichiiPerfSnapshot) -> String {
        if hasMetalFxTemporal, QuickSettings.bool("metalfx_temporal") {
            return "MetalFX Temporal"
        }
        if snapshot.game_dlss_network {
            return "Game DLSS"
        }
        if snapshot.nvidia_dlss {
            return "DLSS"
        }
        return OverlayNames.name("scaling_filter", snapshot.scaling_filter)
    }

    /// Average of the lowest tenth of the real frame rates in the last minute, as the desktop
    /// overlay's "10% low".
    private func lowest10Percent() -> Double {
        guard !history.isEmpty else { return 0 }
        let sorted = history.sorted()
        let count = max(1, sorted.count / 10)
        return sorted.prefix(count).reduce(0, +) / Double(count)
    }

    private static func round(_ value: Double) -> Double {
        (value * 10).rounded() / 10
    }

    private static func size(_ width: UInt32, _ height: UInt32) -> String {
        width > 0 && height > 0 ? "\(width)×\(height)" : ""
    }
}
