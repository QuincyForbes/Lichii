// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace FileSys::Bntx {

/// The same BNTX with mip level 0 removed from its only texture, or nullopt when the file is not
/// a single plain 2D texture with at least two mip levels in a known block layout. Byte for byte
/// what tools/nx2/oot_texture_mod.py drop_top_mip writes.
std::optional<std::vector<u8>> DropTopMip(std::span<const u8> data);

/// Up to `levels` top mip levels dropped from a single-texture BNTX whose larger edge is at least
/// `min_size` (oot_texture_mod.py make --levels --min-size), or nullopt when none could be.
/// `lenient` drops a level even when the relocation table at the end of the file can't be
/// updated, so the result always has the size PlanReduce gave. `arrays` also reduces 2D arrays
/// with one-GOB blocks (each layer loses its top levels).
std::optional<std::vector<u8>> Reduce(std::span<const u8> data, u32 levels, u32 min_size,
                                      bool lenient = false, bool arrays = false);

/// The size Reduce would return, from the start of the file (`head`, at least up to its image
/// data) and the size of the whole file; nullopt when it would drop nothing.
std::optional<u64> PlanReduce(std::span<const u8> head, u64 size, u32 levels, u32 min_size,
                              bool arrays = false);

} // namespace FileSys::Bntx
