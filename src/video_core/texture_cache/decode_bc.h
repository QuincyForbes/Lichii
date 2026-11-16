// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

[[nodiscard]] u32 ConvertedBytesPerBlock(VideoCore::Surface::PixelFormat pixel_format);

void DecompressBCn(std::span<const u8> input, std::span<u8> output, BufferImageCopy& copy,
                   VideoCore::Surface::PixelFormat pixel_format);

/// Plain formats a backend reads only as a wider one (Metal: R32G32B32_FLOAT as RGBA32F,
/// G4R4_UNORM as RG8), expanded by ExpandTexels when the image is Converted.
[[nodiscard]] bool IsExpandedFormat(VideoCore::Surface::PixelFormat pixel_format);

/// Copies `copy`'s texels from the guest's layout into the expanded format, tightly packed.
void ExpandTexels(std::span<const u8> input, std::span<u8> output, const BufferImageCopy& copy,
                  VideoCore::Surface::PixelFormat pixel_format);

} // namespace VideoCommon
