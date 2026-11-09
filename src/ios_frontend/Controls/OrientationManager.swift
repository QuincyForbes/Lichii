// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import LichiiBridge
import UIKit

/// Which way the game may turn: follow the device, landscape only, portrait only, or locked to
/// the orientation it had when the lock was set. The on-screen controller keeps a layout per
/// orientation.
@MainActor
final class OrientationManager: ObservableObject {
    static let shared = OrientationManager()

    enum Mode: String, CaseIterable, Identifiable {
        case auto, landscape, portrait, locked

        var id: String { rawValue }

        var title: String {
            switch self {
            case .auto: return "Automatic"
            case .landscape: return "Landscape"
            case .portrait: return "Portrait"
            case .locked: return "Locked"
            }
        }
    }

    @Published var mode: Mode = Mode(rawValue: UserDefaults.standard.string(forKey: "gameOrientation") ?? "") ?? .landscape {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: "gameOrientation")
            if mode == .locked {
                lockedMask = Self.mask(for: currentInterfaceOrientation)
            }
            apply()
        }
    }

    private var lockedMask: UIInterfaceOrientationMask = .landscape

    private var isPhone: Bool { UIDevice.current.userInterfaceIdiom == .phone }

    var mask: UIInterfaceOrientationMask {
        switch mode {
        case .auto: return isPhone ? .allButUpsideDown : .all
        case .landscape: return .landscape
        case .portrait: return .portrait
        case .locked: return lockedMask
        }
    }

    /// Locks to whatever the game shows now, or unlocks back to automatic.
    func toggleLock() {
        mode = mode == .locked ? .auto : .locked
    }

    func apply() {
        guard let window = ControlsCenter.shared.gameWindow else { return }
        lichii_controls_set_orientations(UInt32(mask.rawValue))
        window.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        window.windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: mask)) { _ in }
    }

    /// The library may take any orientation again.
    func restoreLibrary() {
        lichii_controls_set_orientations(0)
    }

    private var currentInterfaceOrientation: UIInterfaceOrientation {
        ControlsCenter.shared.gameWindow?.windowScene?.interfaceOrientation ?? .landscapeRight
    }

    private static func mask(for orientation: UIInterfaceOrientation) -> UIInterfaceOrientationMask {
        switch orientation {
        case .portrait: return .portrait
        case .portraitUpsideDown: return .portraitUpsideDown
        case .landscapeLeft: return .landscapeLeft
        case .landscapeRight: return .landscapeRight
        default: return .landscape
        }
    }
}
