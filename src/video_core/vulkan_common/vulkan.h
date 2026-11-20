// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#define VK_NO_PROTOTYPES
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#elif defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__HAIKU__)
#define VK_USE_PLATFORM_XCB_KHR
#else
#define VK_USE_PLATFORM_XLIB_KHR
#define VK_USE_PLATFORM_WAYLAND_KHR
#endif

#include <vulkan/vulkan.h>

// Sanitize X11 and Win32 macros pulled in by platform Vulkan headers
#undef Always
#undef BadAccess
#undef BadAlloc
#undef BadAtom
#undef BadColor
#undef BadCursor
#undef BadDrawable
#undef BadFont
#undef BadGC
#undef BadIDChoice
#undef BadImplementation
#undef BadLength
#undef BadMatch
#undef BadName
#undef BadPixmap
#undef BadRequest
#undef BadValue
#undef BadWindow
#undef Bool
#undef CreateEvent
#undef CreateSemaphore
#undef False
#undef LoadString
#undef None
#undef Status
#undef Success
#undef True

// Define maintenance 7-8 extension names (not yet in official Vulkan headers)
#ifndef VK_KHR_MAINTENANCE_7_EXTENSION_NAME
#define VK_KHR_MAINTENANCE_7_EXTENSION_NAME "VK_KHR_maintenance7"
#endif
#ifndef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
#define VK_KHR_MAINTENANCE_8_EXTENSION_NAME "VK_KHR_maintenance8"
#endif

// "Catch-all" handle for both Android and.. the rest of platforms
struct VkSurfaceKHR_T;
