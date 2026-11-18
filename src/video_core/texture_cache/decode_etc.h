// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

/// ETC2 RGB, RGB with punch-through alpha, RGBA (EAC alpha) and EAC R11 / RG11 (unsigned and
/// signed), formats a device without them (Macs without an Apple GPU) reads decoded.
[[nodiscard]] bool IsEtcFormat(VideoCore::Surface::PixelFormat pixel_format);

/// Bytes of a decoded texel: 4 for the RGB(A) formats (RGBA8, sRGB ones as stored), 2 for R11
/// (R16), 4 for RG11 (RG16).
[[nodiscard]] u32 EtcDecodedBytes(VideoCore::Surface::PixelFormat pixel_format);

/// Decodes `copy`'s blocks into tightly packed texels (row = image width).
void DecodeEtc(std::span<const u8> input, std::span<u8> output, const BufferImageCopy& copy,
               VideoCore::Surface::PixelFormat pixel_format);

} // namespace VideoCommon
