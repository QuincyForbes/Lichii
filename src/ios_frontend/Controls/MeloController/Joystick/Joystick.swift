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
final class JoystickView: UIView {
    var right: Bool = true
    var background: Bool = false {
        didSet { if activeTouch == nil { backgroundView.alpha = background ? 1 : 0 } }
    }
    var sensitivity: CGFloat = 1.2

    var deadZone: CGFloat = 0.08

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

        edgeImpact.prepare()
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        let size = bounds.width
        boundaryView.frame              = CGRect(origin: .zero, size: frame.size)
        boundaryView.layer.cornerRadius = size / 2

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

        if activeTouch == nil && springAnimator == nil {
            placeAtCenter()
        } else {
            updateJoystickPosition()
        }
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()

        if window == nil, activeTouch != nil {
            endTracking()
        }
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard activeTouch == nil, let touch = touches.first else { return }
        activeTouch = touch
        beginTracking(at: touch.location(in: self))
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = activeTouch, touches.contains(touch) else { return }
        processOffset(rawOffset(for: touch.location(in: self)))
        updateJoystickPosition()
        onPositionChanged?(normalizedPosition())
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = activeTouch, touches.contains(touch) else { return }
        endTracking()
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = activeTouch, touches.contains(touch) else { return }
        endTracking()
    }

    private func rawOffset(for location: CGPoint) -> CGPoint {
        CGPoint(x: location.x - bounds.midX, y: location.y - bounds.midY)
    }

    private func beginTracking(at location: CGPoint) {
        let rawOffset = rawOffset(for: location)
        springAnimator?.stopAnimation(true)
        springAnimator = nil

        didHitEdge = false
        edgeImpact.prepare()
        animateBackground(show: true)

        processOffset(rawOffset)
        updateJoystickPosition()
        onPositionChanged?(normalizedPosition())

        UIView.animate(withDuration: 0.12, delay: 0,
                       usingSpringWithDamping: 0.6, initialSpringVelocity: 1) {
            self.joystickView.transform = CGAffineTransform(scaleX: 1.12, y: 1.12)
            self.joystickBackgroundView.transform = CGAffineTransform(scaleX: 1.08, y: 1.08)
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
                edgeImpact.impactOccurred(intensity: 0.55)
                didHitEdge = true
            }
        } else {
            offset     = rawOffset
            didHitEdge = false
        }
    }

    private func updateJoystickPosition() {
        let center = CGPoint(
            x: bounds.midX + offset.x,
            y: bounds.midY + offset.y
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
