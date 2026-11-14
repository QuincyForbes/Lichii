// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <Foundation/Foundation.h>

#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

void ReleaseObject(void* object) noexcept {
    CFRelease(object);
}

} // namespace Metal
