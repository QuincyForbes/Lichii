# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# The flags set for C and C++ in parent directories (warnings, BUILD_ID="...") are not valid
# swiftc arguments, so the Swift app starts clean.
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "")
# src/ is an include directory of the C++ code; on the case-insensitive macOS filesystem its
# network/network.h hides the SDK's <Network/Network.h>, which Photos and AVFoundation import.
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")

# Other areas add their Swift files here (Controls/, Content/, Platform/ next to this folder's files).
add_library(lichii_ios_app STATIC
    AppController.swift
    AppExtensions.swift
    Bridge.swift
    EmulationOverlay.swift
    GameLibrary.swift
    ImportView.swift
    JitView.swift
    LaunchView.swift
    LibraryView.swift
    RootView.swift
    SettingsModel.swift
    SettingsView.swift
    Theme.swift
)
set_target_properties(lichii_ios_app PROPERTIES
    Swift_MODULE_NAME LichiiApp
    Swift_LANGUAGE_VERSION 5)
# ../module.modulemap exposes lichii_bridge*.h to Swift as the LichiiBridge module.
target_compile_options(lichii_ios_app PRIVATE
    "SHELL:-I ${CMAKE_CURRENT_SOURCE_DIR}/.."
    -parse-as-library
    $<$<CONFIG:Release>:-O>)
