// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <zstd.h>

#include "common/fs/file.h"
#include "core/file_sys/bntx_reduce.h"
#include "core/file_sys/texture_quality.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace {

constexpr u64 InfoArray = 0x198;
constexpr u64 Brti = 0x1F8;
constexpr u64 MipPtr = 0x498;
constexpr u64 NamePtr = 0x1B8;
constexpr u64 Brtd = 0xFF0;

template <typename T>
void Put(std::vector<u8>& out, u64 offset, T value) {
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

struct Params {
    s32 width = 512;
    s32 height = 512;
    u16 mips = 10;
    u32 format = 0x1D01;
    s32 block_height_log2 = 4;
    bool random_data = false;
    /// Above 1: a BC4 array with one-GOB blocks, its levels sized as the hardware lays them out.
    s32 layers = 1;
};

/// Bytes of a BC4 level with one-GOB blocks: 64-byte GOB columns, 8-row GOBs of 4x4 blocks.
u64 ArrayLevelBytes(s64 width, s64 height) {
    const u64 gobs_wide = (u64((width + 3) / 4) * 8 + 63) / 64;
    const u64 gobs_high = (u64((height + 3) / 4) + 7) / 8;
    return gobs_wide * gobs_high * 512;
}

/// tools/nx2/test_oot_texture_mod.py make_bntx, byte for byte.
std::vector<u8> MakeBntx(const Params& p) {
    std::vector<u8> out(0x1000);
    std::memcpy(out.data(), "BNTX", 4);
    Put<u32>(out, 8, 0x00040101);
    Put<u16>(out, 12, 0xFEFF);
    Put<u8>(out, 14, 0x0C);
    Put<u8>(out, 15, 0x40);
    std::memcpy(out.data() + 0x20, "Ounc", 4);
    Put<u32>(out, 0x24, 1);
    Put<u64>(out, 0x28, InfoArray);
    Put<u64>(out, 0x30, Brtd);
    Put<u64>(out, InfoArray, Brti);
    Put<u16>(out, NamePtr, 4);
    std::memcpy(out.data() + NamePtr + 2, "Test", 4);
    std::memcpy(out.data() + Brti, "BRTI", 4);
    Put<u8>(out, Brti + 0x10, 9);
    Put<u8>(out, Brti + 0x11, 2);
    Put<u16>(out, Brti + 0x16, p.mips);
    Put<u32>(out, Brti + 0x18, 1);
    Put<u32>(out, Brti + 0x1C, p.format);
    Put<u32>(out, Brti + 0x20, 0x21);
    Put<s32>(out, Brti + 0x24, p.width);
    Put<s32>(out, Brti + 0x28, p.height);
    Put<s32>(out, Brti + 0x2C, 1);
    Put<s32>(out, Brti + 0x30, p.layers);
    Put<s32>(out, Brti + 0x34, p.block_height_log2);
    Put<s32>(out, Brti + 0x38, 0x100C9);
    std::vector<u64> sizes;
    s64 w = p.width;
    s64 h = p.height;
    for (u32 i = 0; i < p.mips; ++i) {
        sizes.push_back(p.layers > 1
                            ? ArrayLevelBytes(w, h)
                            : std::max<u64>(512, u64((w + 3) / 4) * u64((h + 3) / 4) * 8));
        w = std::max<s64>(1, w / 2);
        h = std::max<s64>(1, h / 2);
    }
    std::vector<u64> offsets{0x1000};
    for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
        offsets.push_back(offsets.back() + sizes[i]);
    }
    u64 image_size = 0;
    for (const u64 size : sizes) {
        image_size += size;
    }
    image_size *= static_cast<u64>(p.layers);
    Put<u32>(out, Brti + 0x50, static_cast<u32>(image_size));
    Put<u32>(out, Brti + 0x54, 512);
    Put<u64>(out, Brti + 0x60, NamePtr);
    Put<u64>(out, Brti + 0x70, MipPtr);
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        Put<u64>(out, MipPtr + 8 * i, offsets[i]);
    }
    std::memcpy(out.data() + Brtd, "BRTD", 4);
    Put<u64>(out, Brtd + 8, 0x10 + image_size);
    u32 seed = 12345;
    for (u64 i = 0; i < image_size; ++i) {
        seed = seed * 1103515245 + 12345;
        out.push_back(p.random_data ? static_cast<u8>(seed >> 16) : static_cast<u8>(i * 7));
    }
    const u64 rlt = out.size();
    out.resize(rlt + 16 + 2 * 0x18 + 2 * 8);
    std::memcpy(out.data() + rlt, "_RLT", 4);
    Put<u32>(out, rlt + 4, static_cast<u32>(rlt));
    Put<u32>(out, rlt + 8, 2);
    Put<u32>(out, rlt + 16 + 8, 0);
    Put<u32>(out, rlt + 16 + 12, static_cast<u32>(MipPtr + 8 * p.mips));
    Put<u32>(out, rlt + 16 + 20, 1);
    Put<u32>(out, rlt + 16 + 0x18 + 8, static_cast<u32>(Brtd));
    Put<u32>(out, rlt + 16 + 0x18 + 12, static_cast<u32>(0x10 + image_size));
    Put<u32>(out, rlt + 16 + 0x18 + 16, 1);
    Put<u32>(out, rlt + 16 + 0x18 + 20, 1);
    Put<u32>(out, rlt + 16 + 0x30, 0x30);
    Put<u16>(out, rlt + 16 + 0x34, 1);
    Put<u8>(out, rlt + 16 + 0x36, 1);
    Put<u32>(out, rlt + 16 + 0x38, static_cast<u32>(MipPtr));
    Put<u16>(out, rlt + 16 + 0x3C, 1);
    Put<u8>(out, rlt + 16 + 0x3E, static_cast<u8>(p.mips));
    Put<u32>(out, 0x18, static_cast<u32>(rlt));
    Put<u32>(out, 0x1C, static_cast<u32>(out.size()));
    return out;
}

u64 Fnv1a64(const std::vector<u8>& data) {
    u64 h = 0xCBF29CE484222325ULL;
    for (const u8 b : data) {
        h = (h ^ b) * 0x100000001B3ULL;
    }
    return h;
}

struct Case {
    const char* name;
    Params params;
    u64 patch_offset;
    u32 patch_size;
    u32 patch_value;
    u32 levels; // 0: DropTopMip once
    u64 input_fnv;
    // 0 when the tool leaves the file unchanged.
    std::size_t output_size;
    u64 output_fnv;
};

// python3 -I tools/nx2/texture_quality_vectors.py
const std::vector<Case>& Cases() {
    static const std::vector<Case> cases{
        {"bc4_512_l1", {}, 0, 0, 0, 1, 0x40d40e991fd68138, 50256, 0x1fc8fce7925c4b99},
        {"bc4_512_l2", {}, 0, 0, 0, 2, 0x40d40e991fd68138, 17488, 0x66d41a25ddaee30c},
        {"bc4_512_drop", {}, 0, 0, 0, 0, 0x40d40e991fd68138, 50256, 0x1fc8fce7925c4b99},
        {"astc5x5_1000x600_l1",
         {1000, 600, 8, 0x2F01},
         0,
         0,
         0,
         1,
         0xcb4e5f4669c011ab,
         106008,
         0xbd0db845cd200b94},
        {"astc5x5_1000x600_l2",
         {1000, 600, 8, 0x2F01},
         0,
         0,
         0,
         2,
         0xcb4e5f4669c011ab,
         31008,
         0x543da6d761f8dd72},
        {"astc8x8_2048x64_l2",
         {2048, 64, 6, 0x3401, 1},
         0,
         0,
         0,
         2,
         0xebdf338be6efe211,
         10320,
         0x719830902951593b},
        {"r8_640x16_l1",
         {640, 16, 4, 0x0201, 0},
         0,
         0,
         0,
         1,
         0x57aa483afc32f0ad,
         6480,
         0xe5cd448e1888b3dc},
        {"bc1_300x200_drop",
         {300, 200, 5, 0x1A01},
         0,
         0,
         0,
         0,
         0xe8f133b15badf456,
         14824,
         0x77564e5d1be75790},
        {"bc1_300x200_l1", {300, 200, 5, 0x1A01}, 0, 0, 0, 1, 0xe8f133b15badf456, 0, 0},
        {"two_mips_l2", {512, 512, 2}, 0, 0, 0, 2, 0x86b468ef0c70f592, 36944, 0x74ba091822ad8643},
        {"one_mip_l1", {512, 512, 1}, 0, 0, 0, 1, 0x7bd47328443fd3ac, 0, 0},
        {"cube_l1", {}, 0x30, 4, 6, 1, 0x3fef30b2d7c65513, 0, 0},
        {"dim3_l1", {}, 0x11, 1, 3, 1, 0x1ed5b045c7966ce3, 0, 0},
        {"tiled_l1", {}, 0x12, 2, 1, 1, 0xc3f28d946bed3419, 0, 0},
        {"unknown_format_l1", {512, 512, 10, 0x7701}, 0, 0, 0, 1, 0x01fa50f678068b3e, 0, 0},
    };
    return cases;
}

std::vector<u8> Build(const Case& c) {
    auto data = MakeBntx(c.params);
    if (c.patch_size != 0) {
        std::memcpy(data.data() + Brti + c.patch_offset, &c.patch_value, c.patch_size);
    }
    return data;
}

std::optional<std::vector<u8>> Run(const Case& c, const std::vector<u8>& data) {
    return c.levels == 0 ? FileSys::Bntx::DropTopMip(data)
                         : FileSys::Bntx::Reduce(data, c.levels, FileSys::TextureQualityMinSize);
}

std::vector<u8> Compress(const std::vector<u8>& data, int level) {
    std::vector<u8> out(ZSTD_compressBound(data.size()));
    out.resize(ZSTD_compress(out.data(), out.size(), data.data(), data.size(), level));
    return out;
}

std::vector<u8> Decompress(const std::vector<u8>& file) {
    std::vector<u8> out(64 * 1024 * 1024);
    const std::size_t size = ZSTD_decompress(out.data(), out.size(), file.data(), file.size());
    REQUIRE(!ZSTD_isError(size));
    out.resize(size);
    return out;
}

} // namespace

TEST_CASE("BNTX mip drop matches oot_texture_mod.py byte for byte", "[texture_quality]") {
    for (const auto& c : Cases()) {
        INFO(c.name);
        const auto data = Build(c);
        REQUIRE(Fnv1a64(data) == c.input_fnv);
        const auto out = Run(c, data);
        if (c.output_size == 0) {
            REQUIRE(!out.has_value());
            continue;
        }
        REQUIRE(out.has_value());
        REQUIRE(out->size() == c.output_size);
        REQUIRE(Fnv1a64(*out) == c.output_fnv);
    }
}

TEST_CASE("BNTX mip drop rejects truncated and malformed files", "[texture_quality]") {
    const auto data = MakeBntx({});
    for (std::size_t size : {std::size_t{0}, std::size_t{0x40}, std::size_t{0x500},
                             std::size_t{0x1000}, data.size() - 8}) {
        const std::vector<u8> cut(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(size));
        REQUIRE(!FileSys::Bntx::DropTopMip(cut).has_value());
    }
    auto bad_rlt = data;
    Put<u32>(bad_rlt, 0x18, 0x800);
    REQUIRE(!FileSys::Bntx::DropTopMip(bad_rlt).has_value());
    auto huge_sections = data;
    const u32 rlt = static_cast<u32>(data.size() - (16 + 2 * 0x18 + 2 * 8));
    Put<u32>(huge_sections, rlt + 8, 0x7FFFFFFF);
    REQUIRE(!FileSys::Bntx::DropTopMip(huge_sections).has_value());
    auto mips_backwards = data;
    Put<u64>(mips_backwards, MipPtr + 8, 0x800);
    REQUIRE(!FileSys::Bntx::DropTopMip(mips_backwards).has_value());
}

TEST_CASE("Planning from the header gives the size of a lenient drop", "[texture_quality]") {
    const auto data = MakeBntx({});
    const std::span<const u8> head{data.data(), 0x1000};
    for (u32 levels : {1u, 2u}) {
        const auto planned =
            FileSys::Bntx::PlanReduce(head, data.size(), levels, FileSys::TextureQualityMinSize);
        const auto strict = FileSys::Bntx::Reduce(data, levels, FileSys::TextureQualityMinSize);
        REQUIRE(planned.has_value());
        REQUIRE(strict.has_value());
        REQUIRE(*planned == strict->size());
    }
    // A relocation table the strict drop refuses: the lenient one keeps the planned size.
    auto huge_sections = data;
    const u32 rlt = static_cast<u32>(data.size() - (16 + 2 * 0x18 + 2 * 8));
    Put<u32>(huge_sections, rlt + 8, 0x7FFFFFFF);
    REQUIRE(!FileSys::Bntx::Reduce(huge_sections, 1, FileSys::TextureQualityMinSize));
    const auto lenient =
        FileSys::Bntx::Reduce(huge_sections, 1, FileSys::TextureQualityMinSize, true);
    REQUIRE(lenient.has_value());
    REQUIRE(lenient->size() ==
            *FileSys::Bntx::PlanReduce(head, data.size(), 1, FileSys::TextureQualityMinSize));
    REQUIRE(!FileSys::Bntx::PlanReduce(head, data.size(), 1, 1024).has_value());
    REQUIRE(!FileSys::Bntx::PlanReduce(head, data.size() + 1, 1, 512).has_value());
    REQUIRE(!FileSys::Bntx::PlanReduce({data.data(), 0x800}, data.size(), 1, 512).has_value());
}

TEST_CASE("BNTX arrays lose the top levels of every layer", "[texture_quality]") {
    Params params;
    params.width = 1365;
    params.height = 1365;
    params.mips = 11;
    params.block_height_log2 = 0;
    params.layers = 3;
    const auto data = MakeBntx(params);
    const std::span<const u8> head{data.data(), 0x1000};
    const u64 old_layer = (data.size() - 0x1000 - (16 + 2 * 0x18 + 2 * 8)) / 3;
    REQUIRE(!FileSys::Bntx::Reduce(data, 1, FileSys::TextureQualityMinSize).has_value());
    for (u32 levels : {1u, 2u}) {
        const auto reduced =
            FileSys::Bntx::Reduce(data, levels, FileSys::TextureQualityMinSize, false, true);
        REQUIRE(reduced.has_value());
        REQUIRE(FileSys::Bntx::PlanReduce(head, data.size(), levels,
                                          FileSys::TextureQualityMinSize, true) == reduced->size());
        u64 cut = 0;
        for (u32 level = 0; level < levels; ++level) {
            cut += ArrayLevelBytes(std::max(1, 1365 >> level), std::max(1, 1365 >> level));
        }
        const u64 new_layer = old_layer - cut;
        REQUIRE(reduced->size() == data.size() - 3 * cut);
        s32 width{};
        u16 mips{};
        u32 image_size{};
        std::memcpy(&width, reduced->data() + Brti + 0x24, sizeof(width));
        std::memcpy(&mips, reduced->data() + Brti + 0x16, sizeof(mips));
        std::memcpy(&image_size, reduced->data() + Brti + 0x50, sizeof(image_size));
        REQUIRE(width == (1365 >> levels));
        REQUIRE(mips == 11 - levels);
        REQUIRE(image_size == 3 * new_layer);
        for (u64 layer = 0; layer < 3; ++layer) {
            REQUIRE(std::memcmp(reduced->data() + 0x1000 + layer * new_layer,
                                data.data() + 0x1000 + layer * old_layer + cut, new_layer) == 0);
        }
        // The relocation table follows the image as before.
        REQUIRE(std::memcmp(reduced->data() + 0x1000 + 3 * new_layer, "_RLT", 4) == 0);
    }
    // A layout this does not recompute (blocks taller than a GOB) stays as it is.
    params.block_height_log2 = 4;
    REQUIRE(!FileSys::Bntx::Reduce(MakeBntx(params), 1, FileSys::TextureQualityMinSize, false,
                                   true)
                 .has_value());
}

TEST_CASE("Raw zstd frames have the size they promise", "[texture_quality]") {
    for (std::size_t size :
         {std::size_t{1}, std::size_t{255}, std::size_t{256}, std::size_t{65791},
          std::size_t{65792}, std::size_t{131072}, std::size_t{131073}, std::size_t{400000}}) {
        INFO(size);
        std::vector<u8> content(size);
        for (std::size_t i = 0; i < size; ++i) {
            content[i] = static_cast<u8>(i * 13 + 5);
        }
        const auto frame = FileSys::MakeRawZstdFrame(content);
        REQUIRE(frame.size() == FileSys::RawZstdFrameSize(size));
        REQUIRE(ZSTD_findFrameCompressedSize(frame.data(), frame.size()) == frame.size());
        REQUIRE(ZSTD_getFrameContentSize(frame.data(), frame.size()) == size);
        REQUIRE(Decompress(frame) == content);
    }
}

TEST_CASE("Reduced .bntx.zs is one frame of the planned size with the tool's BNTX",
          "[texture_quality]") {
    Params params;
    params.width = 1024;
    params.height = 1024;
    params.mips = 11;
    params.random_data = true;
    const auto bntx = MakeBntx(params);
    params.random_data = false;
    const auto pattern_bntx = MakeBntx(params);
    // A first block zstd compressed, and one stored raw.
    for (const auto& [file, source] : {std::pair{Compress(pattern_bntx, 3), &pattern_bntx},
                                       std::pair{FileSys::MakeRawZstdFrame(bntx), &bntx}}) {
        const FileSys::VectorVfsFile vfs_file{file, "Rock.bntx.zs"};
        for (u32 levels : {1u, 2u}) {
            INFO(levels);
            const auto reduced = FileSys::ReduceZstdTexture(file, levels);
            REQUIRE(reduced.has_value());
            REQUIRE(FileSys::PlanZstdTexture(vfs_file, levels) == reduced->file.size());
            REQUIRE(ZSTD_findFrameCompressedSize(reduced->file.data(), reduced->file.size()) ==
                    reduced->file.size());
            REQUIRE(reduced->bntx_size == source->size());
            const auto expected =
                FileSys::Bntx::Reduce(*source, levels, FileSys::TextureQualityMinSize);
            REQUIRE(expected.has_value());
            REQUIRE(reduced->reduced_bntx_size == expected->size());
            REQUIRE(Decompress(reduced->file) == *expected);
        }
    }

    // Left alone: not zstd, not a BNTX, a frame followed by more data, too small a texture.
    const auto file = Compress(bntx, 3);
    REQUIRE(!FileSys::ReduceZstdTexture(bntx, 1).has_value());
    const std::vector<u8> text(4096, 'a');
    REQUIRE(!FileSys::ReduceZstdTexture(Compress(text, 3), 1).has_value());
    REQUIRE(!FileSys::PlanZstdTexture(FileSys::VectorVfsFile{Compress(text, 3)}, 1).has_value());
    auto two_frames = file;
    const auto second = Compress(text, 3);
    two_frames.insert(two_frames.end(), second.begin(), second.end());
    REQUIRE(!FileSys::ReduceZstdTexture(two_frames, 1).has_value());
    const auto small = Compress(MakeBntx({300, 200, 5, 0x1A01}), 3);
    REQUIRE(!FileSys::ReduceZstdTexture(small, 1).has_value());
    REQUIRE(!FileSys::PlanZstdTexture(FileSys::VectorVfsFile{small}, 1).has_value());
    REQUIRE(!FileSys::ReduceZstdTexture(file, 0).has_value());
    REQUIRE(!FileSys::PlanZstdTexture(FileSys::VectorVfsFile{file}, 0).has_value());
}

TEST_CASE("Texture quality layer serves reduced textures and shares the rest",
          "[texture_quality]") {
    Params params;
    params.random_data = true;
    const auto file = Compress(MakeBntx(params), 3);
    auto texture = std::make_shared<FileSys::VectorVfsFile>(file, "Rock.bntx.zs");
    auto other = std::make_shared<FileSys::VectorVfsFile>(file, "Rock.bntx");
    auto small = std::make_shared<FileSys::VectorVfsFile>(
        Compress(MakeBntx({300, 200, 5, 0x1A01}), 3), "Small.bntx.zs");
    auto tex_dir = std::make_shared<FileSys::VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{texture, other, small},
        std::vector<FileSys::VirtualDir>{}, "Tex");
    auto model_dir = std::make_shared<FileSys::VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{other}, std::vector<FileSys::VirtualDir>{}, "Model");
    FileSys::VirtualDir root = std::make_shared<FileSys::VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{}, std::vector<FileSys::VirtualDir>{tex_dir, model_dir},
        "");

    REQUIRE(FileSys::ApplyTextureQuality(root, 0) == root);
    const auto layered = FileSys::ApplyTextureQuality(root, 1);
    REQUIRE(layered != root);
    REQUIRE(layered->GetSubdirectory("Model") == model_dir);
    const auto served = layered->GetSubdirectory("Tex")->GetFile("Rock.bntx.zs");
    REQUIRE(served != nullptr);
    REQUIRE(served != texture);
    REQUIRE(layered->GetSubdirectory("Tex")->GetFile("Rock.bntx") == other);
    REQUIRE(layered->GetSubdirectory("Tex")->GetFile("Small.bntx.zs") == small);

    const auto expected = FileSys::ReduceZstdTexture(file, 1);
    REQUIRE(expected.has_value());
    REQUIRE(served->GetSize() == expected->file.size());
    REQUIRE(served->GetSize() < file.size());
    const std::size_t size = served->GetSize();
    std::vector<u8> pieces(size);
    const std::size_t half = size / 2;
    REQUIRE(served->Read(pieces.data(), half, 0) == half);
    REQUIRE(served->Read(pieces.data() + half, size - half, half) == size - half);
    REQUIRE(pieces == expected->file);
    REQUIRE(served->ReadAllBytes() == expected->file);
    REQUIRE(served->Read(pieces.data(), 16, size) == 0);
}

TEST_CASE("Texture quality plans are kept for the next start", "[texture_quality]") {
    Params params;
    params.random_data = true;
    const auto file = Compress(MakeBntx(params), 3);
    const auto make_root = [&](const std::vector<u8>& content) -> FileSys::VirtualDir {
        auto texture = std::make_shared<FileSys::VectorVfsFile>(content, "Rock.bntx.zs");
        return std::make_shared<FileSys::VectorVfsDirectory>(
            std::vector<FileSys::VirtualFile>{texture}, std::vector<FileSys::VirtualDir>{}, "");
    };
    const auto cache = std::filesystem::temp_directory_path() / "lichii-texq-test" / "plan.bin";
    std::error_code ec;
    std::filesystem::remove_all(cache.parent_path(), ec);

    const auto first = FileSys::ApplyTextureQuality(make_root(file), 1, cache);
    REQUIRE(std::filesystem::exists(cache));
    const std::size_t planned = first->GetFile("Rock.bntx.zs")->GetSize();
    REQUIRE(planned < file.size());
    // The same sizes from the cache, with the file itself not readable as a texture any more.
    std::vector<u8> unreadable(file.size(), 0);
    REQUIRE(FileSys::ApplyTextureQuality(make_root(unreadable), 1, cache)
                ->GetFile("Rock.bntx.zs")
                ->GetSize() == planned);
    // Another level count or another original size plans again.
    REQUIRE(FileSys::ApplyTextureQuality(make_root(file), 2, cache)
                ->GetFile("Rock.bntx.zs")
                ->GetSize() < planned);
    unreadable.push_back(0);
    REQUIRE(FileSys::ApplyTextureQuality(make_root(unreadable), 1, cache)
                ->GetFile("Rock.bntx.zs")
                ->GetSize() == unreadable.size());
    std::filesystem::remove_all(cache.parent_path(), ec);
}

// Hidden: LICHII_TEXQ_IN=<dir of .bntx.zs files> [LICHII_TEXQ_OUT=<dir>] plans and reduces real
// files, checks each plan against the result, times both, and writes each decompressed original
// (<name>.bntx) and reduced BNTX (<name>.l1, .l2) for tools/nx2/texture_quality_vectors.py
// --compare. LICHII_TEXQ_ARRAYS=1 reduces 2D arrays too.
TEST_CASE("Texture quality on a directory of game files", "[.texture_quality_files]") {
    const char* in = std::getenv("LICHII_TEXQ_IN");
    const char* out = std::getenv("LICHII_TEXQ_OUT");
    REQUIRE(in != nullptr);
    const bool arrays = std::getenv("LICHII_TEXQ_ARRAYS") != nullptr;
    const auto write = [out](const std::filesystem::path& name, const std::vector<u8>& bytes) {
        if (out != nullptr) {
            Common::FS::IOFile dest{std::filesystem::path{out} / name,
                                    Common::FS::FileAccessMode::Write};
            REQUIRE(dest.WriteSpan(std::span<const u8>{bytes}) == bytes.size());
        }
    };
    for (u32 levels : {1u, 2u}) {
        std::size_t files = 0;
        std::size_t reduced_files = 0;
        std::size_t plan_mismatches = 0;
        u64 stored = 0;
        u64 served = 0;
        u64 bntx_before = 0;
        u64 bntx_after = 0;
        double plan_ms = 0;
        double total_ms = 0;
        double max_ms = 0;
        for (const auto& entry : std::filesystem::directory_iterator(in)) {
            const auto name = entry.path().filename().string();
            if (!name.ends_with(".bntx.zs")) {
                continue;
            }
            Common::FS::IOFile source{entry.path(), Common::FS::FileAccessMode::Read};
            std::vector<u8> file(static_cast<std::size_t>(source.GetSize()));
            REQUIRE(source.ReadSpan(std::span{file}) == file.size());
            ++files;
            if (levels == 1) {
                write(name + ".bntx", Decompress(file));
            }
            const FileSys::VectorVfsFile vfs_file{file, name};
            const auto plan_begin = std::chrono::steady_clock::now();
            const auto planned = FileSys::PlanZstdTexture(vfs_file, levels, arrays);
            const auto begin = std::chrono::steady_clock::now();
            const auto reduced = FileSys::ReduceZstdTexture(file, levels, arrays);
            const auto end = std::chrono::steady_clock::now();
            plan_ms += std::chrono::duration<double, std::milli>(begin - plan_begin).count();
            const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
            total_ms += ms;
            max_ms = std::max(max_ms, ms);
            if (planned && (!reduced || *planned != reduced->file.size())) {
                ++plan_mismatches;
                UNSCOPED_INFO("plan mismatch " << name);
            }
            if (!planned || !reduced) {
                continue;
            }
            ++reduced_files;
            stored += file.size();
            served += reduced->file.size();
            bntx_before += reduced->bntx_size;
            bntx_after += reduced->reduced_bntx_size;
            write(name + ".l" + std::to_string(levels), Decompress(reduced->file));
        }
        WARN("levels " << levels << ": " << reduced_files << " of " << files << " planned, "
                       << (bntx_before >> 20) << " -> " << (bntx_after >> 20)
                       << " MiB BNTX, stored " << (stored >> 20) << " -> " << (served >> 20)
                       << " MiB, plan " << plan_ms << " ms, reduce " << total_ms << " ms total, "
                       << max_ms << " ms max");
        CHECK(plan_mismatches == 0);
    }
}
