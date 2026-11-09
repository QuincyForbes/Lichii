// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Window.swift
//  Melo-Controller
//
//  Created by Stossy11 on 8/3/2026.
//

import UIKit

@MainActor
struct Window {
    static var window: UIWindow? {
        UIApplication.shared.windows.filter {$0.isKeyWindow}.first
    }
}
