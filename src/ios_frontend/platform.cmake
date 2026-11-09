# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# iOS platform layer (lichii_bridge_platform.h): link lichii_platform into the app. Its
# lichii_platform_ios.mm uses IosSupport (yuzu_cmd/ios_support.mm), which the app provides.
enable_language(OBJCXX)

add_library(lichii_platform STATIC
    ${CMAKE_CURRENT_LIST_DIR}/lichii_bridge_platform.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lichii_bridge_platform.h
    ${CMAKE_CURRENT_LIST_DIR}/lichii_platform_internal.h
    ${CMAKE_CURRENT_LIST_DIR}/lichii_platform_ios.mm
)
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/lichii_platform_ios.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(lichii_platform PUBLIC common PRIVATE ${PLATFORM_LIBRARIES})
