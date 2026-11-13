// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/file_sys/bntx_reduce.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace FileSys::Bntx {

namespace {

struct BlockSize {
    u8 type;
    u8 width;
    u8 height;
    u8 bytes;
};

// BNTX/NVN format type (high byte of the format) -> block size and bytes per block, as in
// oot_texture_mod.py FORMATS.
constexpr std::array<BlockSize, 31> Formats{{
    {0x02, 1, 1, 1},    {0x07, 1, 1, 2},   {0x09, 1, 1, 2},   {0x0A, 1, 1, 2},
    {0x0B, 1, 1, 4},    {0x0C, 1, 1, 4},   {0x0E, 1, 1, 4},   {0x0F, 1, 1, 4},
    {0x12, 1, 1, 8},    {0x14, 1, 1, 4},   {0x1A, 4, 4, 8},   {0x1B, 4, 4, 16},
    {0x1C, 4, 4, 16},   {0x1D, 4, 4, 8},   {0x1E, 4, 4, 16},  {0x1F, 4, 4, 16},
    {0x20, 4, 4, 16},   {0x2D, 4, 4, 16},  {0x2E, 5, 4, 16},  {0x2F, 5, 5, 16},
    {0x30, 6, 5, 16},   {0x31, 6, 6, 16},  {0x32, 8, 5, 16},  {0x33, 8, 6, 16},
    {0x34, 8, 8, 16},   {0x35, 10, 5, 16}, {0x36, 10, 6, 16}, {0x37, 10, 8, 16},
    {0x38, 10, 10, 16}, {0x39, 12, 10, 16}, {0x3A, 12, 12, 16},
}};

constexpr u32 MaxMips = 32;

const BlockSize* FindFormat(u32 format) {
    const auto it = std::find_if(Formats.begin(), Formats.end(),
                                 [format](const BlockSize& f) { return f.type == (format >> 8); });
    return it != Formats.end() ? &*it : nullptr;
}

template <typename T>
bool Load(std::span<const u8> data, u64 offset, T& out) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) {
        return false;
    }
    std::memcpy(&out, data.data() + offset, sizeof(T));
    return true;
}

template <typename T>
bool Store(std::span<u8> data, u64 offset, T value) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) {
        return false;
    }
    std::memcpy(data.data() + offset, &value, sizeof(T));
    return true;
}

struct Texture {
    u64 brti{};
    u8 dim{};
    u16 tile{};
    u16 mips{};
    u32 format{};
    s32 width{};
    s32 height{};
    s32 depth{};
    s32 layers{};
    s32 block_height_log2{};
    u32 image_size{};
    u64 mip_ptr{};
    std::array<u64, MaxMips> mip_offsets{};

    bool Reducible(bool arrays) const {
        return mips >= 2 && dim == 2 && depth == 1 && (layers == 1 || (arrays && layers > 1)) &&
               tile == 0 && FindFormat(format) != nullptr;
    }
};

/// The only texture of a BNTX file (nullopt when the file holds another count of textures).
std::optional<Texture> ParseSingle(std::span<const u8> data) {
    if (data.size() < 0x40 || std::memcmp(data.data(), "BNTX", 4) != 0) {
        return std::nullopt;
    }
    u32 count{};
    u64 info_array{};
    Texture tex;
    if (!Load(data, 0x24, count) || !Load(data, 0x28, info_array) || count != 1 ||
        !Load(data, info_array, tex.brti)) {
        return std::nullopt;
    }
    const u64 brti = tex.brti;
    if (!Load(data, brti + 0x11, tex.dim) || !Load(data, brti + 0x12, tex.tile) ||
        !Load(data, brti + 0x16, tex.mips) || !Load(data, brti + 0x1C, tex.format) ||
        !Load(data, brti + 0x24, tex.width) || !Load(data, brti + 0x28, tex.height) ||
        !Load(data, brti + 0x2C, tex.depth) || !Load(data, brti + 0x30, tex.layers) ||
        !Load(data, brti + 0x34, tex.block_height_log2) ||
        !Load(data, brti + 0x50, tex.image_size) || !Load(data, brti + 0x70, tex.mip_ptr)) {
        return std::nullopt;
    }
    if (tex.mips > MaxMips) {
        return std::nullopt;
    }
    for (u32 i = 0; i < tex.mips; ++i) {
        if (!Load(data, tex.mip_ptr + 8 * u64{i}, tex.mip_offsets[i])) {
            return std::nullopt;
        }
    }
    return tex;
}

u32 BlockHeightFor(u64 height_in_gobs, u32 block_height) {
    // The Tegra block-linear rule each mip level follows: halve while the level fits in half.
    while (block_height > 1 && height_in_gobs <= block_height / 2) {
        block_height /= 2;
    }
    return block_height;
}

/// Bytes of mip level `level` of a block-linear 2D texture whose level 0 has blocks
/// 2^`block_height_log2` GOBs high.
u64 LevelBytes(const Texture& tex, const BlockSize& block, u32 level, s32 block_height_log2) {
    const u64 width = std::max<u64>(1, u64(tex.width) >> level);
    const u64 height = std::max<u64>(1, u64(tex.height) >> level);
    const u64 gobs_wide = ((width + block.width - 1) / block.width * block.bytes + 63) / 64;
    const u64 gobs_high = ((height + block.height - 1) / block.height + 7) / 8;
    const u32 block_height = BlockHeightFor(gobs_high, 1u << block_height_log2);
    return gobs_wide * ((gobs_high + block_height - 1) / block_height * block_height) * 512;
}

/// Bytes of one array layer from mip level `first` on. Arrays are only reduced with one-GOB
/// blocks (every OoT array texture), whose layers need no alignment past the GOB.
u64 LayerBytes(const Texture& tex, const BlockSize& block, u32 first) {
    u64 bytes = 0;
    for (u32 level = first; level < tex.mips; ++level) {
        bytes += LevelBytes(tex, block, level, tex.block_height_log2);
    }
    return bytes;
}

enum class Mode {
    /// oot_texture_mod.py drop_top_mip: nullopt for anything it would not write.
    Strict,
    /// As Strict, but a relocation table that can't be updated is left as it is.
    Lenient,
    /// Only the start of the file is given (up to its image data): the result is that start,
    /// patched, and `size` becomes the size of the whole result.
    Plan,
};

std::optional<std::vector<u8>> Drop(std::span<const u8> data, u64& size, Mode mode, bool arrays) {
    const auto parsed = ParseSingle(data);
    if (!parsed || !parsed->Reducible(arrays)) {
        return std::nullopt;
    }
    const Texture& tex = *parsed;
    const BlockSize& block = *FindFormat(tex.format);
    const u64 start = tex.mip_offsets[0];
    if (tex.mip_offsets[1] <= start || tex.mip_offsets[1] > size || start > data.size() ||
        tex.width <= 0 || tex.height <= 0 || tex.block_height_log2 < 0 ||
        tex.block_height_log2 > 5) {
        return std::nullopt;
    }
    // Level 0 of each layer goes; an array keeps its layers one after another.
    const u64 cut = tex.mip_offsets[1] - start;
    const u64 layers = static_cast<u64>(tex.layers);
    u64 old_layer = cut;
    u64 new_layer = 0;
    if (layers > 1) {
        // Only the layout recomputed here byte for byte: one-GOB blocks, level offsets and image
        // size as the file gives them.
        if (tex.block_height_log2 != 0) {
            return std::nullopt;
        }
        for (u32 i = 0; i + 1 < tex.mips; ++i) {
            if (tex.mip_offsets[i + 1] - tex.mip_offsets[i] !=
                LevelBytes(tex, block, i, tex.block_height_log2)) {
                return std::nullopt;
            }
        }
        old_layer = LayerBytes(tex, block, 0);
        new_layer = LayerBytes(tex, block, 1);
        if (old_layer * layers != tex.image_size || new_layer + cut != old_layer ||
            (mode != Mode::Plan && start + tex.image_size > data.size())) {
            return std::nullopt;
        }
    }
    const u64 removed = cut * layers;
    // The headers keep their offsets: everything rewritten below must sit before the cut.
    if (tex.brti + 0x78 > start || tex.mip_ptr + 8 * u64{tex.mips} > start ||
        tex.image_size < removed) {
        return std::nullopt;
    }
    const s32 width = std::max(1, tex.width / 2);
    const s32 height = std::max(1, tex.height / 2);
    const u64 rows = (u64(height) + block.height - 1) / block.height;
    const u32 new_block_height = BlockHeightFor((rows + 7) / 8, 1u << tex.block_height_log2);
    s32 new_block_height_log2 = 0;
    while ((1u << new_block_height_log2) < new_block_height) {
        ++new_block_height_log2;
    }

    std::vector<u8> out(mode == Mode::Plan ? start : data.size() - removed);
    std::memcpy(out.data(), data.data(), start);
    if (mode != Mode::Plan && layers > 1) {
        for (u64 layer = 0; layer < layers; ++layer) {
            std::memcpy(out.data() + start + layer * new_layer,
                        data.data() + start + layer * old_layer + cut, new_layer);
        }
        const u64 tail = start + tex.image_size;
        std::memcpy(out.data() + start + layers * new_layer, data.data() + tail,
                    data.size() - tail);
    } else if (mode != Mode::Plan) {
        std::memcpy(out.data() + start, data.data() + start + removed,
                    data.size() - start - removed);
    }
    std::span<u8> o{out};
    Store(o, tex.brti + 0x16, static_cast<u16>(tex.mips - 1));
    Store(o, tex.brti + 0x24, width);
    Store(o, tex.brti + 0x28, height);
    Store(o, tex.brti + 0x34, new_block_height_log2);
    Store(o, tex.brti + 0x50, static_cast<u32>(tex.image_size - removed));
    Store(o, tex.mip_ptr, start);
    for (u32 i = 2; i < tex.mips; ++i) {
        Store(o, tex.mip_ptr + 8 * u64{i - 1}, tex.mip_offsets[i] - cut);
    }
    Store(o, tex.mip_ptr + 8 * (u64{tex.mips} - 1), u64{0});

    u64 brtd{};
    u64 brtd_size{};
    if (!Load(data, 0x30, brtd) || brtd + 16 > start || !Load(data, brtd + 8, brtd_size) ||
        !(brtd < start && start < brtd + brtd_size) || brtd_size < removed) {
        return std::nullopt;
    }
    Store(o, brtd + 8, brtd_size - removed);

    u32 rlt{};
    u32 file_size{};
    if (!Load(data, 0x18, rlt) || !Load(data, 0x1C, file_size) ||
        rlt < start + (layers > 1 ? u64{tex.image_size} : removed) || file_size < removed) {
        return std::nullopt;
    }
    if (mode == Mode::Plan && (file_size != size || u64{rlt} + 16 > size)) {
        return std::nullopt;
    }
    rlt -= static_cast<u32>(removed);
    Store(o, 0x18, rlt);
    Store(o, 0x1C, static_cast<u32>(file_size - removed));
    size -= removed;
    if (mode == Mode::Plan) {
        return out;
    }
    const auto unpatched = [&]() -> std::optional<std::vector<u8>> {
        if (mode == Mode::Strict) {
            return std::nullopt;
        }
        return std::move(out);
    };
    u32 sections{};
    if (!Store(o, u64{rlt} + 4, rlt) || !Load<u32>(o, u64{rlt} + 8, sections) ||
        u64{rlt} + 16 + 0x18 * u64{sections} > out.size()) {
        return unpatched();
    }
    u64 entries = 0;
    for (u32 i = 0; i < sections; ++i) {
        const u64 pos = u64{rlt} + 16 + 0x18 * u64{i};
        std::array<u32, 4> section{};
        std::memcpy(section.data(), out.data() + pos + 8, sizeof(section));
        const u64 offset = section[0];
        const u64 section_size = section[1];
        if (offset <= start && start < offset + section_size) {
            if (section_size < removed) {
                return unpatched();
            }
            Store(o, pos + 12, static_cast<u32>(section_size - removed));
        }
        entries += section[3];
    }
    const u64 entry_base = u64{rlt} + 16 + 0x18 * u64{sections};
    if (entries > (out.size() - entry_base) / 8) {
        return unpatched();
    }
    for (u64 i = 0; i < entries; ++i) {
        const u64 pos = entry_base + 8 * i;
        u32 offset{};
        u16 structs{};
        u8 count{};
        Load<u32>(o, pos, offset);
        Load<u16>(o, pos + 4, structs);
        Load<u8>(o, pos + 6, count);
        if (offset == tex.mip_ptr && structs == 1 && count == tex.mips) {
            Store(o, pos + 6, static_cast<u8>(count - 1));
        }
    }
    return out;
}

bool LargeEnough(std::span<const u8> data, u32 min_size) {
    const auto tex = ParseSingle(data);
    return tex && std::max(tex->width, tex->height) >= static_cast<s64>(min_size);
}

} // namespace

std::optional<std::vector<u8>> DropTopMip(std::span<const u8> data) {
    u64 size = data.size();
    return Drop(data, size, Mode::Strict, false);
}

std::optional<std::vector<u8>> Reduce(std::span<const u8> data, u32 levels, u32 min_size,
                                      bool lenient, bool arrays) {
    if (!LargeEnough(data, min_size)) {
        return std::nullopt;
    }
    std::optional<std::vector<u8>> reduced;
    for (u32 i = 0; i < levels; ++i) {
        const std::span<const u8> current = reduced ? std::span<const u8>{*reduced} : data;
        u64 size = current.size();
        auto smaller = Drop(current, size, lenient ? Mode::Lenient : Mode::Strict, arrays);
        if (!smaller) {
            break;
        }
        reduced = std::move(smaller);
    }
    return reduced;
}

std::optional<u64> PlanReduce(std::span<const u8> head, u64 size, u32 levels, u32 min_size,
                             bool arrays) {
    if (!LargeEnough(head, min_size)) {
        return std::nullopt;
    }
    std::optional<std::vector<u8>> current;
    u64 current_size = size;
    for (u32 i = 0; i < levels; ++i) {
        u64 next_size = current_size;
        auto next = Drop(current ? std::span<const u8>{*current} : head, next_size, Mode::Plan,
                         arrays);
        if (!next) {
            break;
        }
        current = std::move(next);
        current_size = next_size;
    }
    if (!current) {
        return std::nullopt;
    }
    return current_size;
}

} // namespace FileSys::Bntx
