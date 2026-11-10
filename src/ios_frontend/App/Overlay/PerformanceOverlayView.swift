// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// The overlay as ControlsOverlayView places it: nothing at all while the preset is Off.
struct PerformanceOverlayHost: View {
    @AppStorage("overlayPreset") private var presetName = ""
    @AppStorage("hudPosition") private var positionName = HUDPosition.topRight.rawValue

    var body: some View {
        let preset = OverlayPreset.resolve(presetName)
        let position = HUDPosition(rawValue: positionName) ?? .topRight
        if preset != .off {
            PerformanceOverlayView(preset: preset, position: position)
                .padding(.horizontal, 6)
                .padding(.vertical, 6)
                // The menu button sits in the top left corner.
                .padding(.leading, position == .topLeft ? 56 : 0)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: position.alignment)
        }
    }
}

/// Small translucent cards. Touches pass through, VoiceOver skips them (the in-game menu carries a
/// spoken summary), and only the monitor's twice-a-second reading redraws them.
struct PerformanceOverlayView: View {
    let preset: OverlayPreset
    let position: HUDPosition

    @ObservedObject private var monitor = PerformanceOverlayMonitor.shared
    @AppStorage("overlayVoiceOver") private var voiceOver = false

    private static let realColor = Color(red: 56 / 255, green: 189 / 255, blue: 248 / 255)
    private static let displayedColor = Color(red: 74 / 255, green: 222 / 255, blue: 128 / 255)

    var body: some View {
        let r = monitor.reading
        VStack(alignment: position.horizontal, spacing: 4) {
            if r.active {
                if preset == .minimal {
                    card { Text(headline(r)).foregroundStyle(Self.displayedColor) }
                } else {
                    card { frameLines(r) }
                    card { infoLines(r) }
                    if preset == .detailed {
                        card { detailLines(r) }
                    }
                    if showsStatus(r) {
                        statusCard(r)
                    }
                }
            }
        }
        .font(.system(size: 11, weight: .semibold, design: .monospaced))
        .monospacedDigit()
        .foregroundStyle(.white)
        .allowsHitTesting(false)
        .accessibilityHidden(!voiceOver)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(r.spoken)
        .onAppear { monitor.start() }
        .onDisappear { monitor.stop() }
    }

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 2, content: content)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Color.white.opacity(0.12), lineWidth: 1))
    }

    private func headline(_ r: OverlayReading) -> String {
        if r.frameGenActive {
            return String(format: "%.0f → %.0f FPS", r.realFps, r.displayedFps)
        }
        return String(format: "%.0f FPS", r.realFps)
    }

    @ViewBuilder
    private func frameLines(_ r: OverlayReading) -> some View {
        Text(String(format: "Real %.1f FPS", r.realFps))
            .foregroundStyle(Self.realColor)
        if r.frameGenActive {
            Text(String(format: "Generated +%.1f", r.generatedFps))
            Text(String(format: "Displayed %.1f FPS", r.displayedFps))
                .foregroundStyle(Self.displayedColor)
        }
        Text(frameTimeLine(r))
        if r.low10Fps > 0 {
            Text(String(format: "10%% low %.0f", r.low10Fps))
        }
    }

    private func frameTimeLine(_ r: OverlayReading) -> String {
        if r.frameGenActive, r.displayedFps > 0.5 {
            return String(format: "Frame %.1f ms (%.1f out)", r.frameTimeMs, 1000 / r.displayedFps)
        }
        return String(format: "Frame %.1f ms", r.frameTimeMs)
    }

    @ViewBuilder
    private func infoLines(_ r: OverlayReading) -> some View {
        Text("Frame gen \(r.frameGen)")
        Text(scaleLine(r))
        Text(String(format: "Speed %.0f%%", r.speedPercent))
            .foregroundStyle(r.speedPercent > 0 && r.speedPercent < 90 ? Color.orange : Color.white)
        if r.shadersBuilding > 0 {
            Text("Shaders \(r.shadersBuilding) building")
                .foregroundStyle(.yellow)
        }
    }

    private func scaleLine(_ r: OverlayReading) -> String {
        let scale = String(format: "Scale %.2gx", r.resolutionScale)
        return r.upscaler.isEmpty ? scale : "\(scale) \(r.upscaler)"
    }

    @ViewBuilder
    private func detailLines(_ r: OverlayReading) -> some View {
        if !r.renderSize.isEmpty {
            Text("Render \(r.renderSize)")
            if !r.outputSize.isEmpty {
                Text("Output \(r.outputSize)")
            }
        }
        if !r.gpuAccuracy.isEmpty {
            Text("GPU accuracy \(r.gpuAccuracy)")
        }
        if r.cpuPercent >= 0 {
            Text(String(format: "CPU %.0f%%", r.cpuPercent))
        }
        if r.footprint > 0 {
            Text(r.memoryLimit > 0 ? "RAM \(Self.gigabytes(r.footprint)) / \(Self.gigabytes(r.memoryLimit)) GB"
                                   : "RAM \(Self.gigabytes(r.footprint)) GB")
        }
        if r.textureBytes > 0 {
            Text(r.textureExpected > 0 ? "Textures \(Self.megabytes(r.textureBytes)) / \(Self.megabytes(r.textureExpected)) MB"
                                       : "Textures \(Self.megabytes(r.textureBytes)) MB")
        }
    }

    /// Standard shows the icons only when something is worth knowing.
    private func showsStatus(_ r: OverlayReading) -> Bool {
        preset == .detailed || r.thermal != .nominal || r.lowPower
    }

    private func statusCard(_ r: OverlayReading) -> some View {
        HStack(spacing: 8) {
            Image(systemName: Self.thermalSymbol(r.thermal))
                .foregroundStyle(Self.thermalColor(r.thermal))
            if r.lowPower {
                Image(systemName: "leaf.fill")
                    .foregroundStyle(.yellow)
            }
            if preset == .detailed, r.battery >= 0 {
                Image(systemName: r.charging ? "battery.100percent.bolt" : Self.batterySymbol(r.battery))
                    .foregroundStyle(r.charging ? Color.green : (r.battery < 0.2 ? Color.red : Color.white))
                Text(String(format: "%.0f%%", r.battery * 100))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Color.white.opacity(0.12), lineWidth: 1))
    }

    private static func gigabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f", Double(bytes) / 1_073_741_824)
    }

    private static func megabytes(_ bytes: UInt64) -> String {
        String(bytes >> 20)
    }

    private static func thermalSymbol(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "thermometer.low"
        case .fair: return "thermometer.medium"
        default: return "thermometer.high"
        }
    }

    private static func thermalColor(_ state: ProcessInfo.ThermalState) -> Color {
        switch state {
        case .nominal: return .white
        case .fair: return .yellow
        case .serious: return .orange
        default: return .red
        }
    }

    private static func batterySymbol(_ level: Float) -> String {
        switch level {
        case ..<0.13: return "battery.0percent"
        case ..<0.38: return "battery.25percent"
        case ..<0.63: return "battery.50percent"
        case ..<0.88: return "battery.75percent"
        default: return "battery.100percent"
        }
    }
}

extension OverlayReading {
    /// What VoiceOver reads, for the overlay (when allowed) and the in-game menu.
    var spoken: String {
        guard active else { return "Performance: no game running." }
        var parts: [String] = []
        if frameGenActive {
            parts.append(String(format: "%.0f frames per second displayed, %.0f real", displayedFps, realFps))
        } else {
            parts.append(String(format: "%.0f frames per second", realFps))
        }
        parts.append(String(format: "frame time %.1f milliseconds", frameTimeMs))
        parts.append("frame generation \(frameGen)")
        parts.append(String(format: "resolution scale %.2g times", resolutionScale))
        if !upscaler.isEmpty {
            parts.append("upscaler \(upscaler)")
        }
        parts.append(String(format: "speed %.0f percent", speedPercent))
        if footprint > 0 {
            parts.append(String(format: "memory %.1f gigabytes", Double(footprint) / 1_073_741_824))
        }
        switch thermal {
        case .nominal: break
        case .fair: parts.append("device warm")
        case .serious: parts.append("device hot")
        default: parts.append("device critically hot")
        }
        if lowPower {
            parts.append("low power mode on")
        }
        return "Performance: " + parts.joined(separator: ", ") + "."
    }
}
