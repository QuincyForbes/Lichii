# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Native iOS app: the C bridge (C++, here) and the SwiftUI app (App/), both linked into Lichii.app
# (yuzu-cmd). Other areas append their bridge sources to this list.

add_library(lichii_ios_bridge STATIC
    bridge_host.h
    lichii_bridge.cpp
    lichii_bridge.h
)
target_link_libraries(lichii_ios_bridge PRIVATE common core frontend_common input_common hid_core video_core)
target_include_directories(lichii_ios_bridge PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(lichii_ios_bridge PRIVATE
    -Wno-conversion -Wno-unused-parameter -Wno-missing-field-initializers)
create_target_directory_groups(lichii_ios_bridge)

add_subdirectory(App)
add_subdirectory(Controls)
