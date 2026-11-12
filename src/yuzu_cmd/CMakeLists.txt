# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# Credits to Samantas5855 and others for this function.
function(create_resource file output filename)
    # Read hex data from file
    file(READ ${file} filedata HEX)
    # Convert hex data for C compatibility
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
    # Write data to output file
    set(RESOURCES_DIR "${PROJECT_BINARY_DIR}/dist" PARENT_SCOPE)
    file(WRITE "${PROJECT_BINARY_DIR}/dist/${output}" "const unsigned char ${filename}[] = {${filedata}};\nconst unsigned ${filename}_size = sizeof(${filename});\n")
endfunction()

if (ENABLE_OPENGL)
    list(APPEND OPENGL_SOURCES
        emu_window/emu_window_sdl3_gl.cpp
        emu_window/emu_window_sdl3_gl.h
    )
else()
    set(OPENGL_SOURCES "")
endif()

add_executable(yuzu-cmd
    emu_window/emu_window_sdl3.cpp
    emu_window/emu_window_sdl3.h
    emu_window/emu_window_sdl3_null.cpp
    emu_window/emu_window_sdl3_null.h
    emu_window/emu_window_sdl3_vk.cpp
    emu_window/emu_window_sdl3_vk.h
    sdl_config.cpp
    sdl_config.h
    yuzu.cpp
    yuzu.rc
    ${OPENGL_SOURCES}
)

target_link_libraries(yuzu-cmd PRIVATE common core input_common frontend_common video_core)
if (ENABLE_OPENGL)
    target_link_libraries(yuzu-cmd PRIVATE glad)
endif()
target_link_libraries(yuzu-cmd PRIVATE ${PLATFORM_LIBRARIES} Threads::Threads)

create_resource("../../dist/eden.bmp" "yuzu_cmd/yuzu_icon.h" "yuzu_icon")
target_include_directories(yuzu-cmd PRIVATE ${RESOURCES_DIR})

target_link_libraries(yuzu-cmd PRIVATE SDL3::SDL3)

if(UNIX AND NOT APPLE)
    install(TARGETS yuzu-cmd)
endif()

if(WIN32)
    # compile as a win32 gui application instead of a console application
    if(MSVC)
        # SDL's main implementation provides wmain when UNICODE is defined, which src/CMakeLists.txt
        # does for cl.exe but not for clang-cl; forcing mainCRTStartup failed to link with cl.exe.
        if(CXX_CLANG)
            set(YUZU_CMD_ENTRY mainCRTStartup)
        else()
            set(YUZU_CMD_ENTRY wmainCRTStartup)
        endif()
        set_target_properties(yuzu-cmd PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS /ENTRY:${YUZU_CMD_ENTRY}")
    elseif(MINGW)
        set_target_properties(yuzu-cmd PROPERTIES LINK_FLAGS_RELEASE "-Wl,--subsystem,windows")
    endif()
endif()

create_target_directory_groups(yuzu-cmd)

if (IOS)
    enable_language(OBJCXX)
    target_sources(yuzu-cmd PRIVATE ios_support.mm ios_support.h)
    set_source_files_properties(ios_support.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(yuzu-cmd PRIVATE lichii_platform)
    set_target_properties(yuzu-cmd PROPERTIES
        OUTPUT_NAME "Lichii"
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST "${PROJECT_SOURCE_DIR}/dist/ios/Info.plist.in"
        MACOSX_BUNDLE_GUI_IDENTIFIER "io.github.zortos293.lichii"
        MACOSX_BUNDLE_BUNDLE_NAME "Lichii"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
        MACOSX_BUNDLE_BUNDLE_VERSION "1"
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
    add_custom_command(TARGET yuzu-cmd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:yuzu-cmd>/Frameworks"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${MOLTENVK_FRAMEWORK_DIR}" "$<TARGET_BUNDLE_DIR:yuzu-cmd>/Frameworks/MoltenVK.framework"
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/dist/ios/Lichii.entitlements" "$<TARGET_BUNDLE_DIR:yuzu-cmd>/../Lichii.entitlements"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/dist/ios/AppIcon" "$<TARGET_BUNDLE_DIR:yuzu-cmd>"
        COMMENT "Embedding MoltenVK.framework and the app icons into Lichii.app")
    if (MOLTENVK_MESH_FRAMEWORK_DIR)
        # The second driver (Settings moltenvk_driver, externals/CMakeLists.txt).
        add_custom_command(TARGET yuzu-cmd POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${MOLTENVK_MESH_FRAMEWORK_DIR}" "$<TARGET_BUNDLE_DIR:yuzu-cmd>/Frameworks/MoltenVKMesh.framework"
            COMMENT "Embedding MoltenVKMesh.framework into Lichii.app")
    endif()
    if (LICHII_IOS_UI)
        target_compile_definitions(yuzu-cmd PRIVATE LICHII_IOS_UI)
        target_link_libraries(yuzu-cmd PRIVATE lichii_ios_app lichii_ios_bridge)
        # Linked by clang++: the Swift runtime and overlays come from the SDK and the toolchain.
        execute_process(COMMAND xcrun --sdk iphoneos --show-sdk-path
                        OUTPUT_VARIABLE LICHII_IOS_SDK OUTPUT_STRIP_TRAILING_WHITESPACE)
        # CMAKE_Swift_COMPILER may be the /usr/bin shim; the runtime libraries sit next to the real one.
        execute_process(COMMAND xcrun --sdk iphoneos -f swiftc
                        OUTPUT_VARIABLE LICHII_SWIFTC OUTPUT_STRIP_TRAILING_WHITESPACE)
        get_filename_component(LICHII_SWIFT_BIN "${LICHII_SWIFTC}" DIRECTORY)
        set_target_properties(yuzu-cmd PROPERTIES LINKER_LANGUAGE CXX)
        target_link_options(yuzu-cmd PRIVATE
            "-L${LICHII_IOS_SDK}/usr/lib/swift"
            "-L${LICHII_SWIFT_BIN}/../lib/swift/iphoneos"
            "-Wl,-rpath,/usr/lib/swift")
    endif()
endif()

# needed for vma
if (NOT MSVC)
    target_compile_options(yuzu-cmd PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()
