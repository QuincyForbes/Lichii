// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  ControllerButton.swift
//  Melo-Controller
//
//  Created by Stossy11 on 10/1/2026.
//

import UIKit
import SwiftUI


final class ControllerTouchRouter {
    static let shared = ControllerTouchRouter()
    private init() {}

    private final class WeakTarget {
        weak var view: ControllerTouchView?
        init(_ view: ControllerTouchView) { self.view = view }
    }

    private var targets: [ObjectIdentifier: WeakTarget] = [:]

    private var owners: [ObjectIdentifier: ControllerTouchView] = [:]

    private var tracked: Set<ObjectIdentifier> = []

    // MARK: Registration

    func register(_ view: ControllerTouchView) {
        targets[ObjectIdentifier(view)] = WeakTarget(view)
    }

    func unregister(_ view: ControllerTouchView, cancellingPress: Bool = true) {
        targets[ObjectIdentifier(view)] = nil
        for (touch, owner) in owners where owner === view {
            owners[touch] = nil
        }
        if cancellingPress { view.cancelPress() }
    }

    // MARK: Touch handling

    func touchBegan(_ touch: UITouch, on view: ControllerTouchView) {
        let key = ObjectIdentifier(touch)
        owners[key]?.endPress()
        owners[key] = view
        tracked.insert(key)
        view.beginPress()
    }

    func touchMoved(_ touch: UITouch, on view: ControllerTouchView) {
        let key = ObjectIdentifier(touch)
        guard view.slideEnabled, tracked.contains(key) else { return }
        let current = owners[key]

        let next = target(at: touch.location(in: nil), in: touch.window)
        guard next !== current else { return }

        current?.endPress()
        owners[key] = next
        next?.beginPress()
    }

    func touchEnded(_ touch: UITouch, on view: ControllerTouchView) {
        let key = ObjectIdentifier(touch)
        tracked.remove(key)
        owners.removeValue(forKey: key)?.endPress()
    }

    private func target(at point: CGPoint, in window: UIWindow?) -> ControllerTouchView? {
        var best: ControllerTouchView?
        var bestScore = CGFloat.greatestFiniteMagnitude
        var stale: [ObjectIdentifier] = []

        for (key, box) in targets {
            guard let candidate = box.view else {
                stale.append(key)
                continue
            }

            guard let candidateWindow = candidate.window,
                  window == nil || candidateWindow === window,
                  !candidate.isHidden,
                  candidate.isUserInteractionEnabled,
                  candidate.acceptsSlide,
                  candidate.alpha > 0.01 else { continue }

            let frame = candidate.convert(candidate.bounds, to: nil)
            guard frame.width > 0, frame.height > 0 else { continue }


            let nx = (point.x - frame.midX) / (frame.width / 2)
            let ny = (point.y - frame.midY) / (frame.height / 2)

            let score = candidate.isCircular ? hypot(nx, ny) : max(abs(nx), abs(ny))

            guard score <= 1 else { continue }
            if score < bestScore {
                bestScore = score
                best = candidate
            }
        }

        for key in stale { targets[key] = nil }
        return best
    }
}

final class ControllerTouchView: UIView {
    var onPress: (() -> Void)?
    var onRelease: (() -> Void)?

    var isCircular = true

    var slideEnabled = true
    var acceptsSlide = true


    private var pressCount = 0

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        backgroundColor = .clear
        isUserInteractionEnabled = true
        isMultipleTouchEnabled = true
    }

    deinit {
        ControllerTouchRouter.shared.unregister(self, cancellingPress: false)
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window == nil {
            ControllerTouchRouter.shared.unregister(self)
        } else {
            ControllerTouchRouter.shared.register(self)
        }
    }

    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
        guard isCircular else { return super.point(inside: point, with: event) }
        guard bounds.width > 0, bounds.height > 0 else { return false }
        let nx = (point.x - bounds.midX) / (bounds.width / 2)
        let ny = (point.y - bounds.midY) / (bounds.height / 2)
        return hypot(nx, ny) <= 1
    }

    // MARK: Press state

    func beginPress() {
        pressCount += 1
        if pressCount == 1 { onPress?() }
    }

    func endPress() {
        guard pressCount > 0 else { return }
        pressCount -= 1
        if pressCount == 0 { onRelease?() }
    }

    func cancelPress() {
        guard pressCount > 0 else { return }
        pressCount = 0
        onRelease?()
    }

    // MARK: Touches

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            ControllerTouchRouter.shared.touchBegan(touch, on: self)
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            ControllerTouchRouter.shared.touchMoved(touch, on: self)
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            ControllerTouchRouter.shared.touchEnded(touch, on: self)
        }
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            ControllerTouchRouter.shared.touchEnded(touch, on: self)
        }
    }
}

struct ControllerUIButtonViewRepresentable: UIViewRepresentable {
    let onPress: () -> Void
    let onRelease: () -> Void
    var isCircular: Bool = true
    var slideEnabled: Bool = true
    var acceptsSlide: Bool = true

    func makeUIView(context: Context) -> ControllerTouchView {
        let view = ControllerTouchView(frame: .zero)
        apply(to: view)
        return view
    }

    func updateUIView(_ uiView: ControllerTouchView, context: Context) {
        apply(to: uiView)
    }

    private func apply(to view: ControllerTouchView) {
        view.onPress = onPress
        view.onRelease = onRelease
        view.isCircular = isCircular
        view.slideEnabled = slideEnabled
        view.acceptsSlide = acceptsSlide
    }
}
