// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Haptics.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2025.
//

import UIKit
import SwiftUI

class Haptics {
    static let shared = Haptics()

    private init() { }

    /// Lichii: "On-ScreenControllerHaptics" (default on) turns button feedback off.
    private var enabled: Bool {
        UserDefaults.standard.object(forKey: "On-ScreenControllerHaptics") as? Bool ?? true
    }

    private var generators: [UIImpactFeedbackGenerator.FeedbackStyle: UIImpactFeedbackGenerator] = [:]

    func play(_ feedbackStyle: UIImpactFeedbackGenerator.FeedbackStyle) {
        guard enabled else { return }
        let generator = generators[feedbackStyle] ?? UIImpactFeedbackGenerator(style: feedbackStyle)
        generators[feedbackStyle] = generator
        generator.impactOccurred()
        generator.prepare()
    }

    func notify(_ feedbackType: UINotificationFeedbackGenerator.FeedbackType) {
        UINotificationFeedbackGenerator().notificationOccurred(feedbackType)
    }
}
