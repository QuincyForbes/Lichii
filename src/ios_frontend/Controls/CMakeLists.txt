# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Controls: Melo-Controller (vendored, its own Swift module) and the controls screens, which join
# the app module.

set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "")
# No src/ include path for Swift (see App/CMakeLists.txt).
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")

add_library(lichii_melo_controller STATIC
    MeloController/Buttons/ButtonConfiguration.swift
    MeloController/Buttons/ButtonView.swift
    MeloController/Buttons/ControllerButton.swift
    MeloController/Controller.swift
    MeloController/ControllerHandler.swift
    MeloController/ControllerView.swift
    MeloController/Haptics/Haptics.swift
    MeloController/Joystick/Joystick.swift
    MeloController/Joystick/JoystickDPadPoint.swift
    MeloController/Joystick/JoystickView.swift
    MeloController/Joystick/JoystickViewRepresentable.swift
    MeloController/Layout/CopyLayoutView.swift
    MeloController/Layout/LayoutEditorView.swift
    MeloController/Layout/LayoutManager.swift
    MeloController/Layout/LayoutOptionsView.swift
    MeloController/Models/Layout.swift
    MeloController/Models/VirtualControllerButton.swift
    MeloController/Models/Window.swift
)
set_target_properties(lichii_melo_controller PROPERTIES
    Swift_MODULE_NAME Melo_Controller
    Swift_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
    Swift_LANGUAGE_VERSION 5)
target_compile_options(lichii_melo_controller PRIVATE
    -parse-as-library
    $<$<CONFIG:Release>:-O>)

target_sources(lichii_ios_app PRIVATE
    ControllerSettingsView.swift
    ControlsCenter.swift
    ControlsOverlayView.swift
    ControlsView.swift
    ExternalDisplay.swift
    InGameMenuView.swift
    OrientationManager.swift
    PerformanceHUD.swift
    PhysicalControllers.swift
    QuickSettings.swift
    RumbleController.swift
    TouchController.swift
)
target_link_libraries(lichii_ios_app PRIVATE lichii_melo_controller)
target_compile_options(lichii_ios_app PRIVATE "SHELL:-I ${CMAKE_CURRENT_BINARY_DIR}/modules")

target_sources(lichii_ios_bridge PRIVATE
    ../controls_host.h
    ../controls_router.cpp
    ../controls_router.h
    ../lichii_bridge_controls.cpp
    ../lichii_bridge_controls.h
)
target_link_libraries(lichii_ios_bridge PRIVATE SDL3::SDL3)
