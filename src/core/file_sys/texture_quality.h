// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

/// Textures whose larger edge is below this keep every mip level.
constexpr u32 TextureQualityMinSize = 512;

/// A smaller BNTX served for a `.bntx.zs` file: one zstd frame of raw blocks, so its size is known
/// from the start of the original file alone (the RomFS needs every size before the game starts).
struct ReducedTexture {
    std::vector<u8> file;
    u64 bntx_size{};
    u64 reduced_bntx_size{};
};

/// Size of the zstd frame MakeRawZstdFrame writes for `content_size` bytes.
u64 RawZstdFrameSize(u64 content_size);

/// `content` as one zstd frame of raw blocks with its content size.
std::vector<u8> MakeRawZstdFrame(std::span<const u8> content);

/// The size of the replacement of the `.bntx.zs` file `file` with `levels` top mip levels dropped,
/// read from its first zstd block only; nullopt when the file stays as it is (not one zstd frame
/// without a dictionary holding a single plain 2D texture of TextureQualityMinSize or more).
std::optional<u64> PlanZstdTexture(const VfsFile& file, u32 levels, bool arrays = false);

/// The replacement itself (`file` holds the whole original), the size PlanZstdTexture gave, or
/// nullopt when the file stays as it is.
std::optional<ReducedTexture> ReduceZstdTexture(std::span<const u8> file, u32 levels,
                                                bool arrays = false);

/// Settings texture_quality: `root` with every `*.bntx.zs` file PlanZstdTexture accepts served
/// through ReduceZstdTexture when the game reads it. Directories without such files are shared
/// with `root`. The planned sizes are kept in `plan_cache` (when given) for the next start.
/// `arrays`: 2D array textures are reduced too (Bntx::Reduce).
VirtualDir ApplyTextureQuality(const VirtualDir& root, u32 levels,
                               const std::filesystem::path& plan_cache = {}, bool arrays = false);

} // namespace FileSys
