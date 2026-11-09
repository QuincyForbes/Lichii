// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

import LichiiBridge
import SwiftUI
import UIKit

/// The app's controls on top of the game's render window: a menu button, and the basic pause menu
/// used until the full in-game menu replaces it (AppExtensions.presentInGameMenu).
@MainActor
enum EmulationOverlay {
    private static let buttonTag = 0x4C49_4348

    static func install(in window: UIWindow) {
        guard let host = window.rootViewController?.view else { return }
        host.viewWithTag(buttonTag)?.removeFromSuperview()
        let button = UIButton(type: .system)
        button.tag = buttonTag
        button.setImage(UIImage(systemName: "line.3.horizontal.circle.fill"), for: .normal)
        button.tintColor = .white
        button.alpha = 0.55
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addAction(UIAction { _ in lichii_post_event(Int32(LICHII_EVENT_MENU_REQUESTED.rawValue)) },
                         for: .touchUpInside)
        host.addSubview(button)
        NSLayoutConstraint.activate([
            button.centerXAnchor.constraint(equalTo: host.safeAreaLayoutGuide.centerXAnchor),
            button.bottomAnchor.constraint(equalTo: host.safeAreaLayoutGuide.bottomAnchor, constant: -6),
            button.widthAnchor.constraint(equalToConstant: 40),
            button.heightAnchor.constraint(equalToConstant: 40),
        ])
    }

    static func presentBasicMenu(from presenter: UIViewController) {
        guard presenter.presentedViewController == nil else { return }
        lichii_emu_pause()
        let title = AppController.shared.currentGame?.title ?? "Lichii"
        let menu = UIAlertController(title: title, message: nil, preferredStyle: .actionSheet)
        menu.addAction(UIAlertAction(title: "Resume", style: .cancel) { _ in
            lichii_emu_resume()
        })
        menu.addAction(UIAlertAction(title: "Settings", style: .default) { _ in
            let settings = UIHostingController(rootView: NavigationStack {
                SettingsView(scope: .global)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") {
                                presenter.dismiss(animated: true) { lichii_emu_resume() }
                            }
                        }
                    }
            })
            presenter.present(settings, animated: true)
        })
        menu.addAction(UIAlertAction(title: "Quit to Library", style: .destructive) { _ in
            AppController.shared.stop()
        })
        if let popover = menu.popoverPresentationController {
            popover.sourceView = presenter.view
            popover.sourceRect = CGRect(x: presenter.view.bounds.midX, y: presenter.view.bounds.maxY - 30,
                                        width: 1, height: 1)
        }
        presenter.present(menu, animated: true)
    }
}
