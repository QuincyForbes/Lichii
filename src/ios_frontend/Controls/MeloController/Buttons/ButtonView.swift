// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  ButtonView.swift
//  Melo-Controller
//
//  Created by Stossy11 on 26/1/2026.
//

import SwiftUI

public struct ButtonView: View {
    var disabled: Bool
    var controller: any Controller
    var button: VirtualControllerButton
    var opacity: Double
    @Binding var layout: LayoutConfig?
    var gameId: String?
    var joystickDpadPoint: JoystickDPadPoint = .shared

    @AppStorage("onscreenhandheld") var onscreenjoy: Bool = false
    @AppStorage("buttonSlide") private var buttonSlide = true
    @AppStorage("On-ScreenControllerScale") var controllerScale: Double = 1.0
    @Environment(\.presentationMode) var presentationMode

    @State private var isPressed = false
    @State private var toggleState = false
    @State private var size: CGSize = .zero

    public init(controller: any Controller, disabled: Bool = false, button: VirtualControllerButton, opacity: Double = 1.0, layout: Binding<LayoutConfig> = .constant(LayoutConfig()), gameId: String? = nil) {
        self.controller = controller
        self.gameId = gameId
        self.disabled = disabled
        self.button = button
        self.opacity = opacity
        if layout.wrappedValue == LayoutConfig() {
            _layout = .constant(nil)
        } else {
            _layout = Binding(
                get: { layout.wrappedValue },
                set: { layout.wrappedValue = $0 ?? LayoutConfig() }
            )
        }
    }

    public var body: some View {
        Circle()
            .foregroundStyle(.clear.opacity(0))
            .overlay {
                Image(systemName: buttonConfig.iconName)
                    .resizable()
                    .scaledToFit()
                    .frame(width: size.width, height: size.height)
                    .foregroundStyle(.white)
                    .opacity(isPressed ? 0.6 : 0.8)
                    .allowsHitTesting(false)
            }
            .frame(width: size.width, height: size.height)
            .background(buttonBackground)
            .onAppear {
                size = calculateButtonSize()
            }
            .onDisappear {
                if toggleState {
                    toggleState = false
                    isPressed = false
                    sendRelease()
                }
            }
            .onChange(of: controllerScale) { _ in
                size = calculateButtonSize()
            }
            .overlay {
                if !disabled {
                    ControllerUIButtonViewRepresentable(
                        onPress: handleButtonPress,
                        onRelease: handleButtonRelease,
                        isCircular: !button.isTrigger,
                        slideEnabled: buttonSlide && !istoggle,
                        acceptsSlide: buttonSlide && !istoggle
                    )
                    .allowsHitTesting(true)
                    .frame(width: size.width * 1.25, height: size.height * 1.25)
                }
            }
            .opacity(opacity)
    }

    private var buttonBackground: some View {
        Group {
            if !button.isTrigger && button != .leftStick && button != .rightStick {
                Circle()
                    .fill(Color.gray.opacity(0.4))
                    .frame(width: size.width * 1.25, height: size.height * 1.25)
            } else if button == .leftStick || button == .rightStick {
                Image(systemName: buttonConfig.iconName)
                    .resizable()
                    .scaledToFit()
                    .frame(width: size.width * 1.25, height: size.height * 1.25)
                    .foregroundColor(Color.gray.opacity(0.4))
            } else if button.isTrigger {
                Image(systemName: convertTriggerIconToButton(buttonConfig.iconName))
                    .resizable()
                    .scaledToFit()
                    .frame(width: size.width * 1.25, height: size.height * 1.25)
                    .foregroundColor(Color.gray.opacity(0.4))
            }
        }
    }

    private func convertTriggerIconToButton(_ iconName: String) -> String {
        if iconName.hasPrefix("zl") || iconName.hasPrefix("zr") {
            var converted = String(iconName.dropFirst(3))
            converted = converted.replacingOccurrences(of: "rectangle", with: "button")
            converted = converted.replacingOccurrences(of: ".fill", with: ".horizontal.fill")
            return converted
        } else {
            var converted = String(iconName.dropFirst(2))
            converted = converted.replacingOccurrences(of: "rectangle", with: "button")
            converted = converted.replacingOccurrences(of: ".fill", with: ".horizontal.fill")
            return converted
        }
    }


    private var istoggle: Bool {
        layout?.buttons[button.id]?.toggle ?? false
    }

    private var joystickDpad: Bool {
        let defaults = UserDefaults.standard
        let key = "joystickDpad-\(gameId ?? "global")"
        if defaults.object(forKey: key) != nil {
            return defaults.bool(forKey: key)
        }
        return defaults.bool(forKey: "joystickDpad")
    }

    private func sendPress() {
        if joystickDpad, button.isDPad {
            joystickDpadPoint.pressed(button, controller: controller)
        } else {
            controller.buttonPressed(button)
        }
    }

    private func sendRelease() {
        if button.isDPad, joystickDpadPoint.currentlyPressed.contains(button) {
            joystickDpadPoint.released(button, controller: controller)
        } else {
            controller.buttonReleased(button)
        }
    }

    private func handleButtonPress() {
        if istoggle || toggleState {
            toggleState.toggle()
            isPressed = toggleState
            toggleState ? sendPress() : sendRelease()
            Haptics.shared.play(.soft)
        } else {
            guard !isPressed else { return }
            isPressed = true
            sendPress()
            Haptics.shared.play(.soft)
        }
    }

    private func handleButtonRelease() {
        guard !toggleState, isPressed else { return }

        isPressed = false
        sendRelease()
    }

    private func calculateButtonSize() -> CGSize {
        let baseWidth: CGFloat
        let baseHeight: CGFloat

        if button.isTrigger {
            baseWidth = 70
            baseHeight = 40
        } else if button.isSmall {
            baseWidth = 35
            baseHeight = 35
        } else {
            baseWidth = 45
            baseHeight = 45
        }

        let deviceMultiplier = UIDevice.current.userInterfaceIdiom == .pad ? 1.2 : 1.0
        let scaleMultiplier = CGFloat(controllerScale)

        return CGSize(
            width: baseWidth * deviceMultiplier * scaleMultiplier,
            height: baseHeight * deviceMultiplier * scaleMultiplier
        )
    }

    private var buttonConfig: ButtonConfiguration {
        .init(iconName: button.iconName)
    }
}
