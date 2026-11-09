// SPDX-License-Identifier: GPL-3.0-only
// Portions adapted from Melo-Controller (https://github.com/stossy11/Melo-Controller @ b16a8e3), Copyright Stossy11 (GPLv3)
//
//  ControllerView.swift
//  Melo-Controller
//
//  Created by Stossy11 on 16/7/2024.
//

import SwiftUI
import GameController
import CoreMotion

/// A closure type used to build an individual button in a custom controller layout.
/// Receives a ``VirtualControllerButton`` and returns a type-erased view.
public typealias SubView = (VirtualControllerButton) -> AnyView

/// A closure type used to build a joystick in a custom controller layout.
///
/// Parameters:
/// - `String`: The joystick ID (e.g. `"leftJoystick"`, `"rightJoystick"`)
/// - `Bool`: `true` if this is the right stick (`iscool`)
/// - `Binding<Bool>`: Controls whether the overlapping D-pad / ABXY overlay is hidden
public typealias JoystickSubView = (String, Bool, Binding<Bool>) -> AnyView

/// A full on-screen nintendo-style game controller  as a SwiftUI view.
///
/// `ControllerView` handles orientation changes, layout persistence, and edit mode.
/// By default it renders a standard portrait or landscape layout automatically.
/// Pass a `customControllerLayout` closure to provide your own arrangement of buttons and sticks.
///
/// **Basic usage (default layout):**
/// ```swift
/// ControllerView(controller: myController, isEditing: false)
/// ```
///
/// **Custom layout:**
/// ```swift
/// ControllerView(controller: myController, isEditing: false) { button, joystick, isPortrait in
///     HStack {
///         joystick("leftJoystick", false, $hideDpad)
///         Spacer()
///         joystick("rightJoystick", true, $hideABXY)
///     }
/// }
/// ```
public struct ControllerView: View {
    var gameId: String?
    var gameName: String?
    var controller: any Controller
    @StateObject var controllerHandler = ControllerHandler()
    @AppStorage("On-ScreenControllerScale") private var controllerScale: Double = 1.0
    @AppStorage("On-ScreenControllerOpacity") private var controllerOpacity: Double = 1.0
    @AppStorage("stickButton") private var stickButton = false
    @State private var hideDpad = false
    @State private var hideABXY = false
    @State private var selectedButton: String?
    @State private var selectedJoystick: String?
    @Environment(\.verticalSizeClass) var verticalSizeClass
    @State private var showEditControls = true
    @State var isEditing: Bool = false
    private var isEditingInput: Bool

    /// An optional closure that replaces the built-in portrait/landscape layouts.
    ///
    /// The closure receives:
    /// - A ``SubView`` factory:  call it with any ``VirtualControllerButton`` to get a fully wired, editable button view.
    /// - A ``JoystickSubView`` factory: call it with an ID, `iscool`, and a `showBackground` binding to get a wired joystick.
    /// - A `Bool` indicating whether the device is currently in portrait orientation.
    ///
    /// Set to `nil` (the default) to use the automatic portrait/landscape layout.
    @State var customControllerLayout: ((@escaping SubView, @escaping JoystickSubView, Bool) -> AnyView)? = nil

    /// Creates a `ControllerView` with a custom button/joystick layout.
    ///
    /// - Parameters:
    ///   - controller: The ``Controller`` object that receives all button and stick input.
    ///   - isEditing: Pass `true` to enter edit mode, where buttons and sticks can be
    ///     repositioned and scaled. Pass `false` for normal play.
    ///   - gameId: An optional identifier used to load and save a per-game layout.
    ///     Pass `nil` to use the global layout.
    ///   - gameName: An optional display name for the game, shown in the layout editor
    ///     instead of `gameId`. The layout is still saved under `gameId`.
    ///   - customControllerLayout: A `@ViewBuilder` closure that composes your own
    ///     controller UI using the provided ``SubView`` and ``JoystickSubView`` factories.
    ///     The `Bool` argument is `true` when the device is in portrait orientation,
    ///     which you can use to switch between layouts.
    public init<V: View>(
        controller: any Controller,
        isEditing: Bool,
        gameId: String? = nil,
        gameName: String? = nil,
        @ViewBuilder customControllerLayout: @escaping (@escaping SubView, @escaping JoystickSubView, Bool) -> V
    ) {
        self.isEditingInput = isEditing
        self._isEditing = State(initialValue: isEditing)
        self.gameId = gameId
        self.gameName = gameName
        self.controller = controller
        self._customControllerLayout = State(initialValue: { s1, s2, s3 in AnyView(customControllerLayout(s1, s2, s3)) })
    }

    public init(
        controller: any Controller,
        isEditing: Bool,
        gameId: String? = nil,
        gameName: String? = nil,
        onEditingChanged: ((Bool) -> Void)? = nil
    ) {
        self.isEditingInput = isEditing
        self._isEditing = State(initialValue: isEditing)
        self.gameId = gameId
        self.gameName = gameName
        self.controller = controller
        self.onEditingChanged = onEditingChanged
    }

    /// Lichii: tells the host when the editor's Done button ends edit mode.
    private var onEditingChanged: ((Bool) -> Void)?


    @State private var layout: LayoutConfig = LayoutConfig()

    public var body: some View {
        ZStack {
            Group {
                let isPad = UIDevice.current.userInterfaceIdiom == .pad
                if let customControllerLayout {
                    customControllerLayout(editableButton, editableJoystick, controllerHandler.isPortrait)
                } else {
                    if controllerHandler.isPortrait && !isPad {
                        portraitLayout
                    } else {
                        landscapeLayout
                    }
                }
            }
            .padding()
            .opacity(isEditing ? 1 : controllerOpacity)
            .onChange(of: verticalSizeClass) { _ in controllerHandler.updateOrientation() }
            .onReceive(NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)) { _ in
                controllerHandler.updateOrientation()
            }
            .onChange(of: controllerHandler.isPortrait) { portrait in
                // Lichii: portrait and landscape keep separate layouts.
                if isEditing {
                    LayoutManager.shared.save(layout, for: gameId)
                }
                LayoutManager.shared.portrait = portrait
                loadLayout()
            }
            .onChange(of: isEditingInput) { isEditing = $0 }
            .onChange(of: isEditing) { onEditingChanged?($0) }
            .onAppear {
                controllerHandler.updateOrientation()
                LayoutManager.shared.portrait = controllerHandler.isPortrait
                loadLayout()
            }

            // Edit Controls
            if isEditing {
                if showEditControls {
                    LayoutEditorView(
                        hideDpad: $hideDpad,
                        hideABXY: $hideABXY,
                        isEditing: $isEditing,
                        showEditControls: $showEditControls,
                        layout: $layout,
                        selectedButton: $selectedButton,
                        selectedJoystick: $selectedJoystick,
                        gameId: gameId,
                        gameName: gameName
                    )
                    .zIndex(1)
                } else {
                    VStack {
                        HStack {
                            Button(action: {
                                withAnimation(.easeInOut(duration: 0.3)) {
                                    showEditControls = true
                                }
                            }) {
                                Image(systemName: "eye")
                                    .font(.system(size: 15, weight: .semibold))
                                    .foregroundColor(.primary)
                                    .padding(13)
                                    .background(.ultraThinMaterial, in: Circle())
                            }
                            .buttonStyle(.plain)
                            .padding(12)

                            Spacer()
                        }

                        Spacer()
                    }
                }
            }
        }
        .environmentObject(controllerHandler)
    }

    private func loadLayout() {
        layout = LayoutManager.shared.load(for: gameId)
    }

    private var portraitLayout: some View {
        VStack {
            Spacer()
            VStack(spacing: 20) {
                HStack(spacing: 30) {
                    VStack(spacing: 15) {
                        shoulderButtonsLeft
                        ZStack {
                            editableJoystick(id: "leftJoystick", showBackground: $hideDpad)

                            if layout.joysticks["leftJoystick"]?.hide ?? true {
                                dpadView
                                    .opacity(hideDpad ? 0 : 1)
                                    .allowsHitTesting(!hideDpad)
                                    .animation(.easeInOut(duration: 0.2), value: hideDpad)
                            } else {
                                dpadView
                            }
                        }
                    }

                    VStack(spacing: 15) {
                        shoulderButtonsRight
                        ZStack {
                            editableJoystick(id: "rightJoystick", iscool: true, showBackground: $hideABXY)
                            if layout.joysticks["rightJoystick"]?.hide ?? true {
                                abxyView
                                    .opacity(hideABXY ? 0 : 1)
                                    .allowsHitTesting(!hideABXY)
                                    .animation(.easeInOut(duration: 0.2), value: hideABXY)
                            } else {
                                abxyView
                            }
                        }
                    }
                }

                if stickButton {
                    HStack(spacing: 60) {
                        HStack {
                            editableButton(.leftStick).padding()
                            minusGroup
                        }
                        HStack {
                            plusGroup
                            editableButton(.rightStick).padding()
                        }
                    }
                } else {
                    HStack(spacing: 60) {
                        HStack {
                            minusGroup
                        }
                        HStack {
                            plusGroup
                        }
                    }
                }
            }
        }
    }

    private var landscapeLayout: some View {
        VStack {
            Spacer()
            HStack {
                VStack(spacing: 20) {
                    shoulderButtonsLeft
                    ZStack {
                        editableJoystick(id: "leftJoystick", showBackground: $hideDpad)

                        if layout.joysticks["leftJoystick"]?.hide ?? true {
                            dpadView
                                .opacity(hideDpad ? 0 : 1)
                                .allowsHitTesting(!hideDpad)
                                .animation(.easeInOut(duration: 0.2), value: hideDpad)
                        } else {
                            dpadView
                        }
                    }
                }

                Spacer()
                centerButtons
                Spacer()

                VStack(spacing: 20) {
                    shoulderButtonsRight
                    ZStack {
                        editableJoystick(id: "rightJoystick", iscool: true, showBackground: $hideABXY)
                        if layout.joysticks["rightJoystick"]?.hide ?? true {
                            abxyView
                                .opacity(hideABXY ? 0 : 1)
                                .allowsHitTesting(!hideABXY)
                                .animation(.easeInOut(duration: 0.2), value: hideABXY)
                        } else {
                            abxyView
                        }
                    }
                }
            }
        }
    }

    private var centerButtons: some View {
        Group {
            if stickButton {
                VStack {
                    HStack(spacing: 50) {
                        editableButton(.leftStick).padding()
                        Spacer()
                        editableButton(.rightStick).padding()
                    }
                    .padding(.top, 30)

                    HStack(spacing: 50) {
                        minusGroup
                        Spacer()
                        plusGroup
                    }
                }
                .padding(.bottom, 20)
            } else {
                HStack(spacing: 50) {
                    minusGroup
                    Spacer()
                    plusGroup
                }
                .padding(.bottom, 20)
            }
        }
    }

    // MARK: - Button Groups

    // Lichii: Capture next to -, Home next to +.
    private var minusGroup: some View {
        HStack(spacing: 14) {
            editableButton(.back)
            editableButton(.capture)
        }
    }

    private var plusGroup: some View {
        HStack(spacing: 14) {
            editableButton(.home)
            editableButton(.start)
        }
    }

    private var shoulderButtonsLeft: some View {
        HStack(spacing: 20) {
            editableButton(.leftTrigger)
            editableButton(.leftShoulder)
        }
        .frame(width: 160 * CGFloat(controllerScale), height: 20 * CGFloat(controllerScale))
    }

    private var shoulderButtonsRight: some View {
        HStack(spacing: 20) {
            editableButton(.rightShoulder)
            editableButton(.rightTrigger)
        }
        .frame(width: 160 * CGFloat(controllerScale), height: 20 * CGFloat(controllerScale))
    }

    private var dpadView: some View {
        VStack(spacing: 7) {
            editableButton(.dPadUp)
            HStack(spacing: 22) {
                editableButton(.dPadLeft)
                Spacer(minLength: 22)
                editableButton(.dPadRight)
            }
            editableButton(.dPadDown)
        }
        .frame(width: 145 * CGFloat(controllerScale), height: 145 * CGFloat(controllerScale))
    }

    private var abxyView: some View {
        VStack(spacing: 7) {
            editableButton(.X)
            HStack(spacing: 22) {
                editableButton(.Y)
                Spacer(minLength: 22)
                editableButton(.A)
            }
            editableButton(.B)
        }
        .frame(width: 145 * CGFloat(controllerScale), height: 145 * CGFloat(controllerScale))
    }

    // MARK: - Helper Methods

    private func editableButton(_ button: VirtualControllerButton) -> AnyView {
        .init(
            EditableButtonView(
                button: button,
                controller: controller,
                layout: $layout,
                gameId: gameId,
                isEditing: isEditing,
                selectedButton: $selectedButton,
                selectedJoystick: $selectedJoystick
            )
        )
    }

    private func editableJoystick(
        id: String,
        iscool: Bool = false,
        showBackground: Binding<Bool>,
    ) -> AnyView {
        .init(
            EditableJoystickView(
                id: id,
                iscool: iscool,
                controller: controller,
                showBackground: showBackground,
                layout: $layout,
                isEditing: isEditing,
                selectedJoystick: $selectedJoystick,
                selectedButton: $selectedButton,
            )
        )
    }
}
