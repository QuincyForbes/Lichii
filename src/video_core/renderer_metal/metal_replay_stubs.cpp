// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Symbols the frontends and the Objective-C++ part of the renderer normally provide
// (lichii-metal-replay links neither).

#define VMA_IMPLEMENTATION
#include "video_core/vulkan_common/vma.h"

#include "frontend_common/play_time_manager.h"
#include "video_core/renderer_metal/metal_object.h"

namespace PlayTime {

PlayTimeManager::PlayTimeManager() {}

PlayTimeManager::~PlayTimeManager() {}

u64 PlayTimeManager::GetPlayTime(u64 program_id) const {
    return 0;
}

} // namespace PlayTime

namespace Metal {

// The replay makes no Metal objects.
void ReleaseObject(void*) noexcept {}

} // namespace Metal
