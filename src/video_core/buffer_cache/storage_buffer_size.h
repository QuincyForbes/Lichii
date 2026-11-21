// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include "common/common_types.h"
#include "common/literals.h"

namespace VideoCommon {

[[nodiscard]] constexpr bool IsNvnGraphicsStorageDescriptor(size_t stage, u32 cbuf_index,
                                                            u32 cbuf_offset, u32 indirect_offset) {
    const u32 base = 0x110 + static_cast<u32>(stage) * 0x100;
    return stage < 5 && cbuf_index == 0 && indirect_offset == ~0U && cbuf_offset >= base &&
           cbuf_offset < base + 0x100 && (cbuf_offset - base) % 0x10 == 0;
}

/// Validates an NVN compute root-table storage buffer descriptor in constant bank 0.
/// NVN's compute root table stores [end_offset, grid_x, grid_y, grid_z] at c[0][0x00] and
/// 16-byte [gpu_addr, byte_size, 0] descriptors at 16-byte offsets below end_offset, whereas
/// CUDA/NNGraph kernels store driver launch state in c[0] and bare 8-byte pointers in c[0][0x160+].
[[nodiscard]] constexpr u32 NvnComputeStorageDeclaredSize(u32 cbuf_index, u32 cbuf_offset,
                                                          u32 cbuf_size, u32 indirect_offset,
                                                          const u32 (&c0_header)[4], u32 grid_x,
                                                          u32 grid_y, u32 grid_z,
                                                          bool indirect_dispatch, u64 size_qword) {
    if (cbuf_index != 0 || indirect_offset != ~0U || cbuf_offset < 0x10 ||
        (cbuf_offset & 0xfU) != 0 || cbuf_offset + 0x10 > cbuf_size) {
        return 0;
    }
    const u32 root_end = c0_header[0];
    if (root_end < cbuf_offset + 0x10 || root_end > 0x1000 || (root_end & 0xfU) != 0 ||
        root_end > cbuf_size) {
        return 0;
    }
    if (!indirect_dispatch &&
        (c0_header[1] != grid_x || c0_header[2] != grid_y || c0_header[3] != grid_z)) {
        return 0;
    }
    if ((size_qword >> 32) != 0) {
        return 0;
    }
    return static_cast<u32>(size_qword);
}

[[nodiscard]] constexpr u32 StorageBufferSize(u32 memory_layout_size, u64 size_word,
                                              bool native_pointer, bool table_pointer,
                                              bool is_written,
                                              u32 read_only_table_limit = 64 * 1024 * 1024) {
    using namespace Common::Literals;
    if (table_pointer && !is_written) {
        return (std::min)(memory_layout_size, read_only_table_limit);
    }
    if (!native_pointer && static_cast<u32>(size_word >> 32) == 0 && size_word != 0 &&
        size_word <= memory_layout_size) {
        return static_cast<u32>(size_word);
    }
    return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
}

} // namespace VideoCommon
