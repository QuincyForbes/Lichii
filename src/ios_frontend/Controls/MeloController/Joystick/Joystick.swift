// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  Joystick.swift
//  Melo-Controller
//
//  Created by Stossy11 on 21/03/2025.
//


import UIKit
import SwiftUI

// i am never doing uikit again
// Lichii: ControllerTouchTarget and the first-touch log.
final class JoystickView: UIView, ControllerTouchTarget {
    var touchPriority: Int { 0 }

    var right: Bool = true
    var background: Bool = false {
        didSet { if activeTouch == nil { backgroundView.alpha = background ? 1 : 0 } }
    }
    var sensitivity: CGFloat = 1.2

    var deadZone: CGFloat = 0.08

    /// Lichii: the stick centres where the thumb lands (relative), instead of measuring from the
    /// middle of the control (absolute).
    var floating = false
    /// Lichii: under the ABXY buttons, a touch that starts in their area presses the face button
    /// of its quadrant unless it starts dragging, so a tap between two buttons is not lost to
    /// the camera.
    var deferTaps = false
    var onFaceButton: ((FaceQuadrant, Bool) -> Void)?

    enum FaceQuadrant {
        case top, left, right, bottom
    }

    private var origin: CGPoint?
    private var pendingStart: CGPoint?
    private var pendingHold: DispatchWorkItem?
    private var heldQuadrant: FaceQuadrant?

    private var stickCenter: CGPoint { origin ?? CGPoint(x: bounds.midX, y: bounds.midY) }

    private var dragDiameter: CGFloat { bounds.width > 0 ? bounds.width : 160 }

    private var joystickSize: CGFloat { dragDiameter * 0.2 }
    private var boundarySize: CGFloat { dragDiameter }

    var onPositionChanged: ((CGPoint) -> Void)?
    var onActiveChanged: ((Bool) -> Void)?

    private let boundaryView          = UIView()
    private let backgroundView        = UIView()
    private let joystickView          = UIView()
    private let joystickBackgroundView = UIView()

    private var offset: CGPoint = .zero

    private let edgeImpact   = UIImpactFeedbackGenerator(style: .light)
    private var didHitEdge   = false

    private var springAnimator: UIViewPropertyAnimator?

    private weak var activeTouch: UITouch?
    private var lastLayoutSize: CGSize = .zero

    private var extendedRadius: CGFloat {
        let maxRadius = (boundarySize - joystickSize) / 2
        return maxRadius + (joystickSize / 2)
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        clipsToBounds = false

        boundaryView.isUserInteractionEnabled = false
        backgroundView.isUserInteractionEnabled = false
        joystickBackgroundView.isUserInteractionEnabled = false
        joystickView.isUserInteractionEnabled = false

        addSubview(boundaryView)
        addSubview(backgroundView)
        addSubview(joystickBackgroundView)
        addSubview(joystickView)

        backgroundView.backgroundColor = UIColor.gray.withAlphaComponent(0.4)
        backgroundView.alpha = background ? 1 : 0

        joystickBackgroundView.backgroundColor = UIColor.gray.withAlphaComponent(0.3)
        joystickView.backgroundColor = UIColor.white.withAlphaComponent(0.5)

        joystickView.layer.shadowColor = UIColor.black.cgColor
        joystickView.layer.shadowOpacity = 0.18
        joystickView.layer.shadowOffset  = CGSize(width: 0, height: 2)
        joystickView.layer.shadowRadius  = 4

        isMultipleTouchEnabled = true

        if UserDefaults.standard.object(forKey: "On-ScreenControllerHaptics") as? Bool ?? true {
            edgeImpact.prepare()
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        if bounds.size != lastLayoutSize {
            lastLayoutSize = bounds.size
            let size = bounds.width
            boundaryView.frame              = CGRect(origin: .zero, size: bounds.size)
            boundaryView.layer.cornerRadius = size / 2
            layer.shadowPath = UIBezierPath(ovalIn: bounds).cgPath

            backgroundView.frame              = boundaryView.frame
            backgroundView.layer.cornerRadius = size / 2

            joystickBackgroundView.bounds = CGRect(
                x: 0, y: 0,
                width: joystickSize * 1.25,
                height: joystickSize * 1.25
            )
            joystickBackgroundView.layer.cornerRadius = joystickBackgroundView.bounds.width / 2

            joystickView.bounds = CGRect(
                x: 0, y: 0,
                width: joystickSize,
                height: joystickSize
            )
            joystickView.layer.cornerRadius = joystickSize / 2
            joystickView.layer.shadowPath = UIBezierPath(ovalIn: joystickView.bounds).cgPath
        }
        if let origin {
            backgroundView.center = origin
        }

        if activeTouch == nil && springAnimator == nil {
            placeAtCenter()
        } else {
            updateJoystickPosition()
        }
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()

        if window == nil, activeTouch != nil {
            finishTouch(cancelled: true)
        }
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard activeTouch == nil, let touch = touches.first else { return }
        ControllerDiagnostics.once("stick-touch",
                                   "first stick touch, \(right ? "right" : "left") stick, view \(frame.size), defers taps \(deferTaps)")
        activeTouch = touch
        let location = touch.location(in: self)
        if deferTaps, hypot(location.x - bounds.midX, location.y - bounds.midY) < bounds.width * 0.45 {
            pendingStart = location
            // Held still: the face button is held as well.
            let hold = DispatchWorkItem { [weak self] in self?.holdPendingButton() }
            pendingHold = hold
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3, execute: hold)
            return
        }
        beginTracking(at: location)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = activeTouch, touches.contains(touch), heldQuadrant == nil else { return }
        let location = touch.location(in: self)
        if let start = pendingStart {
            guard hypot(location.x - start.x, location.y - start.y) > 12 else { return }
            cancelPending()
            beginTracking(at: start)
        }
        processOffset(rawOffset(for: location))
        updateJoystickPosition()
        onPositionChanged?(normalizedPosition())
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = activeTouch, touches.contains(touch) else { return }
        finishTouch(cancelled: false)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = activeTouch, touches.contains(touch) else { return }
        finishTouch(cancelled: true)
    }

    private func finishTouch(cancelled: Bool) {
        if let quadrant = heldQuadrant {
            heldQuadrant = nil
            activeTouch = nil
            onFaceButton?(quadrant, false)
        } else if let start = pendingStart {
            cancelPending()
            activeTouch = nil
            if !cancelled, let quadrant = quadrant(for: start) {
                onFaceButton?(quadrant, true)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) { [weak self] in
                    self?.onFaceButton?(quadrant, false)
                }
            }
        } else {
            endTracking()
        }
    }

    private func cancelPending() {
        pendingHold?.cancel()
        pendingHold = nil
        pendingStart = nil
    }

    private func holdPendingButton() {
        guard let start = pendingStart else { return }
        cancelPending()
        if let quadrant = quadrant(for: start) {
            heldQuadrant = quadrant
            onFaceButton?(quadrant, true)
        } else {
            beginTracking(at: start)
        }
    }

    /// The face button whose quadrant `point` is in; nil right at the middle.
    private func quadrant(for point: CGPoint) -> FaceQuadrant? {
        let dx = point.x - bounds.midX
        let dy = point.y - bounds.midY
        guard hypot(dx, dy) > bounds.width * 0.06 else { return nil }
        if abs(dx) > abs(dy) {
            return dx > 0 ? .right : .left
        }
        return dy < 0 ? .top : .bottom
    }

    private func rawOffset(for location: CGPoint) -> CGPoint {
        let center = stickCenter
        return CGPoint(x: location.x - center.x, y: location.y - center.y)
    }

    private func beginTracking(at location: CGPoint) {
        springAnimator?.stopAnimation(true)
        springAnimator = nil
        if floating {
            origin = location
            backgroundView.center = location
        }
        let rawOffset = rawOffset(for: location)

        didHitEdge = false
        if UserDefaults.standard.object(forKey: "On-ScreenControllerHaptics") as? Bool ?? true {
            edgeImpact.prepare()
        }
        animateBackground(show: true)

        processOffset(rawOffset)
        updateJoystickPosition()
        onPositionChanged?(normalizedPosition())

        if !UIAccessibility.isReduceMotionEnabled {
            UIView.animate(withDuration: 0.12, delay: 0,
                           usingSpringWithDamping: 0.6, initialSpringVelocity: 1) {
                self.joystickView.transform = CGAffineTransform(scaleX: 1.12, y: 1.12)
                self.joystickBackgroundView.transform = CGAffineTransform(scaleX: 1.08, y: 1.08)
            }
        }
        onActiveChanged?(true)
    }

    private func endTracking() {
        activeTouch = nil
        UIView.animate(withDuration: 0.15) {
            self.joystickView.transform          = .identity
            self.joystickBackgroundView.transform = .identity
        }

        animateBackground(show: false)
        onPositionChanged?(.zero)
        if origin != nil {
            origin = nil
            backgroundView.center = CGPoint(x: bounds.midX, y: bounds.midY)
        }
        springReturnToCenter()
        onActiveChanged?(false)
    }

    private func normalizedPosition() -> CGPoint {
        let rx = offset.x / extendedRadius
        let ry = offset.y / extendedRadius
        let magnitude = hypot(rx, ry)

        guard magnitude > deadZone else { return .zero }

        let scale = (magnitude - deadZone) / (1 - deadZone) / magnitude

        return CGPoint(
            x: max(-1, min(1, rx * scale * sensitivity)),
            y: max(-1, min(1, ry * scale * sensitivity))
        )
    }

    private func processOffset(_ rawOffset: CGPoint) {
        let distance = hypot(rawOffset.x, rawOffset.y)
        let atEdge   = distance >= extendedRadius

        if atEdge {
            let angle = atan2(rawOffset.y, rawOffset.x)
            offset = CGPoint(
                x: cos(angle) * extendedRadius,
                y: sin(angle) * extendedRadius
            )

            if !didHitEdge {
                if UserDefaults.standard.object(forKey: "On-ScreenControllerHaptics") as? Bool ?? true {
                    edgeImpact.impactOccurred(intensity: 0.55)
                }
                didHitEdge = true
            }
        } else {
            // Lichii: the edge tap plays once per touch, not each time a circling thumb hits the
            // rim again.
            offset = rawOffset
        }
    }

    private func updateJoystickPosition() {
        let base = stickCenter
        let center = CGPoint(
            x: base.x + offset.x,
            y: base.y + offset.y
        )
        joystickView.center          = center
        joystickBackgroundView.center = center
    }

    private func placeAtCenter() {
        offset = .zero
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        joystickView.center          = center
        joystickBackgroundView.center = center
    }

    private func springReturnToCenter() {
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        if UIAccessibility.isReduceMotionEnabled {
            placeAtCenter()
            return
        }

        let params = UISpringTimingParameters(
            mass: 0.6,
            stiffness: 280,
            damping: 18,
            initialVelocity: .zero
        )

        let animator = UIViewPropertyAnimator(duration: 0, timingParameters: params)
        animator.addAnimations {
            self.joystickView.center          = center
            self.joystickBackgroundView.center = center
        }
        offset = .zero
        animator.addCompletion { _ in
            self.springAnimator = nil
        }
        animator.startAnimation()
        springAnimator = animator
    }

    private func animateBackground(show: Bool) {
        if !background {
            UIView.animate(withDuration: 0.15) {
                self.backgroundView.alpha = show ? 1 : 0
            }
        }
    }
}
