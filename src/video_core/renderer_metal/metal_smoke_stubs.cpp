// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// What the smoke test's link reaches in core but no frontend provides (as the unit tests'
// undefined_fix.cpp): the play time manager lives in the Qt and SDL frontends.

#define VMA_IMPLEMENTATION
#include "video_core/vulkan_common/vma.h"

#include "frontend_common/play_time_manager.h"

namespace PlayTime {

PlayTimeManager::PlayTimeManager() {}
PlayTimeManager::~PlayTimeManager() {}
u64 PlayTimeManager::GetPlayTime(u64 program_id) const {
    return 0;
}

} // namespace PlayTime
