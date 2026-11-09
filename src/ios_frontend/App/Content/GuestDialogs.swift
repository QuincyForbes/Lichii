// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

import LichiiBridge
import UIKit

/// The game's dialogs (software keyboard, errors, user picker, controller applet) as
/// UIAlertControllers over whichever window is on screen.
@MainActor
enum GuestDialogs {
    private struct Request: Decodable {
        let id: UInt64?
        let kind: Int
        let title: String?
        let message: String?
        let guideText: String?
        let okText: String?
        let initialText: String?
        let maxLength: Int?
        let minLength: Int?
        let password: Bool?
        let numeric: Bool?
        let cancel: Bool?
        let target: UInt64?
        let options: [Option]?
    }

    private struct Option: Decodable {
        let uuid: String
        let name: String
    }

    private static var presented: [UInt64: UIAlertController] = [:]
    private static var observers: [UInt64: NSObjectProtocol] = [:]

    static func register() {
        lichii_content_set_dialog_handler({ _, id, json in
            guard let json else { return }
            let data = Data(String(cString: json).utf8)
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    GuestDialogs.handle(id: id, data: data)
                }
            }
        }, nil)
    }

    private static func handle(id: UInt64, data: Data) {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let request = try? decoder.decode(Request.self, from: data) else {
            if id != 0 { respond(id, accepted: false, text: nil) }
            return
        }
        switch Int32(request.kind) {
        case LICHII_DIALOG_DISMISS.rawValue:
            if let target = request.target { dismiss(target) }
        case LICHII_DIALOG_KEYBOARD.rawValue:
            presentKeyboard(id: id, request)
        case LICHII_DIALOG_PROFILE.rawValue:
            presentProfiles(id: id, request)
        default:
            presentMessage(id: id, request)
        }
    }

    private static func respond(_ id: UInt64, accepted: Bool, text: String?) {
        presented[id] = nil
        if let observer = observers.removeValue(forKey: id) {
            NotificationCenter.default.removeObserver(observer)
        }
        if let text {
            text.withCString { lichii_content_dialog_respond(id, accepted, $0) }
        } else {
            lichii_content_dialog_respond(id, accepted, nil)
        }
    }

    private static func dismiss(_ id: UInt64) {
        guard let alert = presented.removeValue(forKey: id) else { return }
        if let observer = observers.removeValue(forKey: id) {
            NotificationCenter.default.removeObserver(observer)
        }
        alert.dismiss(animated: true)
    }

    private static func presentMessage(id: UInt64, _ request: Request) {
        let alert = UIAlertController(title: request.title, message: request.message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
            respond(id, accepted: true, text: nil)
        })
        if request.cancel == true {
            alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in
                respond(id, accepted: false, text: nil)
            })
        }
        present(alert, id: id)
    }

    private static func presentKeyboard(id: UInt64, _ request: Request) {
        let message = [request.message, request.guideText]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        let alert = UIAlertController(title: request.title, message: message.isEmpty ? nil : message,
                                      preferredStyle: .alert)
        let maxLength = max(0, request.maxLength ?? 0)
        let minLength = max(0, request.minLength ?? 0)
        alert.addTextField { field in
            field.text = request.initialText
            field.placeholder = request.guideText
            field.autocorrectionType = .no
            field.autocapitalizationType = .none
            field.isSecureTextEntry = request.password == true
            field.keyboardType = request.numeric == true ? .numberPad : .default
            field.clearButtonMode = .whileEditing
        }
        let okTitle = (request.okText?.isEmpty == false) ? request.okText! : "OK"
        let ok = UIAlertAction(title: okTitle, style: .default) { _ in
            var text = alert.textFields?.first?.text ?? ""
            if maxLength > 0, text.count > maxLength {
                text = String(text.prefix(maxLength))
            }
            respond(id, accepted: true, text: text)
        }
        alert.addAction(ok)
        if request.cancel != false {
            alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in
                respond(id, accepted: false, text: nil)
            })
        }
        let validate = {
            let count = alert.textFields?.first?.text?.count ?? 0
            ok.isEnabled = count >= minLength && (maxLength == 0 || count <= maxLength)
        }
        validate()
        if let field = alert.textFields?.first {
            observers[id] = NotificationCenter.default.addObserver(
                forName: UITextField.textDidChangeNotification, object: field, queue: .main) { _ in
                MainActor.assumeIsolated { validate() }
            }
        }
        present(alert, id: id)
    }

    private static func presentProfiles(id: UInt64, _ request: Request) {
        let alert = UIAlertController(title: request.title ?? "Select a user", message: nil,
                                      preferredStyle: .alert)
        for (index, option) in (request.options ?? []).enumerated() {
            alert.addAction(UIAlertAction(title: option.name, style: .default) { _ in
                respond(id, accepted: true, text: String(index))
            })
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in
            respond(id, accepted: false, text: nil)
        })
        present(alert, id: id)
    }

    private static func present(_ alert: UIAlertController, id: UInt64) {
        guard let presenter = topViewController() else {
            // Nothing on screen to present from: give the game its default answer.
            respond(id, accepted: true, text: alert.textFields?.first?.text)
            return
        }
        presented[id] = alert
        presenter.present(alert, animated: true)
    }

    /// The game window while a game runs, else the app's key window.
    static func topViewController() -> UIViewController? {
        let root = AppController.shared.gameWindow?.rootViewController ?? keyWindow()?.rootViewController
        guard var top = root else { return nil }
        while let next = top.presentedViewController, !next.isBeingDismissed {
            top = next
        }
        return top
    }

    static func keyWindow() -> UIWindow? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap(\.windows)
        return windows.first { $0.isKeyWindow && !$0.isHidden } ?? windows.first { !$0.isHidden }
    }
}
