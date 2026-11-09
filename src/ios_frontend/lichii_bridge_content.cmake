# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Content bridge of the iOS app (keys, firmware, add-ons, mods, cheats, saves, logs, profiles,
# guest dialogs). Plain C API in lichii_bridge_content.h for Swift.

add_library(lichii_bridge_content STATIC
    ${CMAKE_CURRENT_LIST_DIR}/lichii_bridge_content.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lichii_bridge_content.h
    ${CMAKE_CURRENT_LIST_DIR}/lichii_bridge_content_applets.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lichii_bridge_content_host.h
    ${CMAKE_CURRENT_LIST_DIR}/lichii_content_zip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lichii_content_zip.h
)

target_include_directories(lichii_bridge_content PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(lichii_bridge_content
    PUBLIC common core frontend_common
    PRIVATE ZLIB::ZLIB nlohmann_json::nlohmann_json fmt::fmt)

if (TARGET yuzu-cmd AND IOS)
    target_link_libraries(yuzu-cmd PRIVATE lichii_bridge_content)
endif()
