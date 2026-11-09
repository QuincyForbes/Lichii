// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import LichiiBridge
import SwiftUI
import UIKit

/// Samples the game's frame stats and the process's CPU, memory and thermal state twice a second
/// while the HUD is visible.
@MainActor
final class PerformanceMonitor: ObservableObject {
    @Published private(set) var fps: Double = 0
    @Published private(set) var gameFps: Double = 0
    @Published private(set) var generatedFps: Double = 0
    @Published private(set) var frameTime: Double = 0
    @Published private(set) var speed: Double = 0
    @Published private(set) var cpu: Double = -1
    @Published private(set) var memory: UInt64 = 0
    @Published private(set) var available: UInt64 = 0
    @Published private(set) var thermal: ProcessInfo.ThermalState = .nominal
    @Published private(set) var battery: Float = -1
    @Published private(set) var charging = false

    private var timer: Timer?

    func start() {
        guard timer == nil else { return }
        UIDevice.current.isBatteryMonitoringEnabled = true
        sample()
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            MainActor.assumeIsolated { HUDMonitor.shared.sample() }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func sample() {
        if let stats = Bridge.perfStats() {
            fps = stats.fps
            gameFps = stats.game_fps
            generatedFps = stats.generated_fps
            frameTime = stats.frametime_ms
            speed = stats.speed * 100
        }
        var system = LichiiSystemStats()
        if lichii_controls_system_stats(&system) {
            cpu = system.cpu_percent
            memory = system.memory_footprint
            available = system.memory_available
        }
        thermal = ProcessInfo.processInfo.thermalState
        battery = UIDevice.current.batteryLevel
        charging = UIDevice.current.batteryState == .charging || UIDevice.current.batteryState == .full
    }
}

@MainActor
enum HUDMonitor {
    static let shared = PerformanceMonitor()
}

enum HUDPosition: String, CaseIterable, Identifiable {
    case topLeft, topRight, bottomLeft, bottomRight

    var id: String { rawValue }

    var title: String {
        switch self {
        case .topLeft: return "Top Left"
        case .topRight: return "Top Right"
        case .bottomLeft: return "Bottom Left"
        case .bottomRight: return "Bottom Right"
        }
    }

    var alignment: Alignment {
        switch self {
        case .topLeft: return .topLeading
        case .topRight: return .topTrailing
        case .bottomLeft: return .bottomLeading
        case .bottomRight: return .bottomTrailing
        }
    }
}

struct PerformanceHUD: View {
    @ObservedObject private var monitor = HUDMonitor.shared
    @ObservedObject private var platform = PlatformCenter.shared
    @AppStorage("hudDetailed") private var detailed = true
    @AppStorage("hudFrameTime") private var showFrameTime = true
    @AppStorage("hudMemory") private var showMemory = true
    @AppStorage("hudCPU") private var showCPU = true
    @AppStorage("hudThermal") private var showThermal = true
    @AppStorage("hudBattery") private var showBattery = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(fpsLine)
            if detailed {
                if showFrameTime {
                    Text(String(format: "%.1f ms · %.0f%% speed", monitor.frameTime, monitor.speed))
                }
                if showCPU, monitor.cpu >= 0 {
                    Text(String(format: "CPU %.0f%%", monitor.cpu))
                }
                if showMemory, monitor.memory > 0 {
                    Text(memoryLine)
                }
                if showThermal {
                    Text("Thermal: \(thermalName)")
                        .foregroundStyle(thermalColor)
                    if !platform.thermalBadge.isEmpty {
                        Text(platform.thermalBadge)
                            .foregroundStyle(thermalColor)
                    }
                }
                if showBattery, monitor.battery >= 0 {
                    Text(String(format: "Battery %.0f%%%@", monitor.battery * 100, monitor.charging ? " ⚡︎" : ""))
                }
            }
        }
        .font(.system(size: 11, weight: .semibold, design: .monospaced))
        .foregroundStyle(.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .allowsHitTesting(false)
        .onAppear { monitor.start() }
        .onDisappear { monitor.stop() }
    }

    private var fpsLine: String {
        if monitor.generatedFps > 0.5 {
            return String(format: "%.0f FPS (%.0f + %.0f FG)", monitor.fps, monitor.gameFps, monitor.generatedFps)
        }
        return String(format: "%.0f FPS", monitor.fps)
    }

    private var memoryLine: String {
        let used = ByteCountFormatter.string(fromByteCount: Int64(monitor.memory), countStyle: .memory)
        guard monitor.available > 0 else { return "RAM \(used)" }
        let free = ByteCountFormatter.string(fromByteCount: Int64(monitor.available), countStyle: .memory)
        return "RAM \(used) · \(free) free"
    }

    private var thermalName: String {
        switch monitor.thermal {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "?"
        }
    }

    private var thermalColor: Color {
        switch monitor.thermal {
        case .nominal: return .white
        case .fair: return .yellow
        case .serious: return .orange
        default: return .red
        }
    }
}
