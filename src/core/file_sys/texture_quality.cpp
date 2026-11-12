// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/file_sys/texture_quality.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#define ZSTD_STATIC_LINKING_ONLY
#define ZSTD_DISABLE_DEPRECATE_WARNINGS
#include <zstd.h>

#include <fmt/format.h>

#include "common/fs/file.h"
#include "common/logging.h"
#include "common/platform_events.h"
#include "core/file_sys/bntx_reduce.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace FileSys {

namespace {

constexpr u32 ZstdMagic = 0xFD2FB528;
constexpr std::size_t RawBlockMax = ZSTD_BLOCKSIZE_MAX;
constexpr u64 MaxBntxSize = 512 * 1024 * 1024;
// Partially read files keep their replacement here until the rest is read.
constexpr std::size_t CacheCap = 32 * 1024 * 1024;

struct DCtxDeleter {
    void operator()(ZSTD_DCtx* dctx) const {
        ZSTD_freeDCtx(dctx);
    }
};

ZSTD_DCtx* ThreadDCtx() {
    thread_local std::unique_ptr<ZSTD_DCtx, DCtxDeleter> dctx{ZSTD_createDCtx()};
    return dctx.get();
}

/// The content size of `file` when it is one zstd frame without a dictionary that states it.
std::optional<u64> SingleFrameContentSize(std::span<const u8> file) {
    ZSTD_frameHeader header{};
    if (file.size() < 16 || ZSTD_getFrameHeader(&header, file.data(), file.size()) != 0 ||
        header.frameType != ZSTD_frame || header.dictID != 0 ||
        header.frameContentSize == ZSTD_CONTENTSIZE_UNKNOWN ||
        header.frameContentSize > MaxBntxSize) {
        return std::nullopt;
    }
    return header.frameContentSize;
}

} // namespace

u64 RawZstdFrameSize(u64 content_size) {
    const u64 fcs_bytes = content_size < 256           ? 1
                          : content_size < 65792       ? 2
                          : content_size <= 0xFFFFFFFF ? 4
                                                       : 8;
    const u64 blocks = std::max<u64>(1, (content_size + RawBlockMax - 1) / RawBlockMax);
    return 4 + 1 + fcs_bytes + 3 * blocks + content_size;
}

std::vector<u8> MakeRawZstdFrame(std::span<const u8> content) {
    const u64 size = content.size();
    std::vector<u8> out;
    out.reserve(RawZstdFrameSize(size));
    const auto put = [&out](u64 value, std::size_t bytes) {
        for (std::size_t i = 0; i < bytes; ++i) {
            out.push_back(static_cast<u8>(value >> (8 * i)));
        }
    };
    put(ZstdMagic, 4);
    // Single segment: the window is the content, whose size follows in 1, 2, 4 or 8 bytes.
    if (size < 256) {
        put(0x20, 1);
        put(size, 1);
    } else if (size < 65792) {
        put(0x60, 1);
        put(size - 256, 2);
    } else if (size <= 0xFFFFFFFF) {
        put(0xA0, 1);
        put(size, 4);
    } else {
        put(0xE0, 1);
        put(size, 8);
    }
    std::size_t offset = 0;
    do {
        const std::size_t block = std::min<std::size_t>(RawBlockMax, content.size() - offset);
        const bool last = offset + block == content.size();
        put((u64{block} << 3) | (last ? 1 : 0), 3);
        out.insert(out.end(), content.begin() + static_cast<std::ptrdiff_t>(offset),
                   content.begin() + static_cast<std::ptrdiff_t>(offset + block));
        offset += block;
    } while (offset < content.size());
    return out;
}

std::optional<u64> PlanZstdTexture(const VfsFile& file, u32 levels) {
    if (levels == 0) {
        return std::nullopt;
    }
    std::array<u8, 32> head{};
    const std::size_t head_size = file.Read(head.data(), head.size(), 0);
    ZSTD_frameHeader header{};
    if (head_size < head.size() || ZSTD_getFrameHeader(&header, head.data(), head_size) != 0 ||
        header.frameType != ZSTD_frame || header.dictID != 0 ||
        header.frameContentSize == ZSTD_CONTENTSIZE_UNKNOWN ||
        header.frameContentSize > MaxBntxSize || header.headerSize + 3 > head_size) {
        return std::nullopt;
    }
    const u8* block_header = head.data() + header.headerSize;
    const u32 block_info = block_header[0] | (block_header[1] << 8) | (block_header[2] << 16);
    const u32 block_type = (block_info >> 1) & 3;
    const std::size_t block_size = block_info >> 3;
    if (block_size > RawBlockMax || (block_type != 0 && block_type != 2)) {
        return std::nullopt;
    }
    std::vector<u8> block(block_size);
    if (file.Read(block.data(), block_size, header.headerSize + 3) != block_size) {
        return std::nullopt;
    }
    std::vector<u8> decoded;
    if (block_type == 0) {
        decoded = std::move(block);
    } else {
        ZSTD_DCtx* const dctx = ThreadDCtx();
        decoded.resize(RawBlockMax);
        if (dctx == nullptr || ZSTD_isError(ZSTD_decompressBegin(dctx))) {
            return std::nullopt;
        }
        const std::size_t result =
            ZSTD_decompressBlock(dctx, decoded.data(), decoded.size(), block.data(), block_size);
        if (ZSTD_isError(result)) {
            return std::nullopt;
        }
        decoded.resize(result);
    }
    const auto reduced =
        Bntx::PlanReduce(decoded, header.frameContentSize, levels, TextureQualityMinSize);
    if (!reduced) {
        return std::nullopt;
    }
    return RawZstdFrameSize(*reduced);
}

std::optional<ReducedTexture> ReduceZstdTexture(std::span<const u8> file, u32 levels) {
    const auto content_size = SingleFrameContentSize(file);
    if (levels == 0 || !content_size ||
        ZSTD_findFrameCompressedSize(file.data(), file.size()) != file.size()) {
        return std::nullopt;
    }
    ZSTD_DCtx* const dctx = ThreadDCtx();
    if (dctx == nullptr) {
        return std::nullopt;
    }
    std::vector<u8> bntx(static_cast<std::size_t>(*content_size));
    const std::size_t result =
        ZSTD_decompressDCtx(dctx, bntx.data(), bntx.size(), file.data(), file.size());
    if (ZSTD_isError(result) || result != bntx.size()) {
        return std::nullopt;
    }
    auto reduced = Bntx::Reduce(bntx, levels, TextureQualityMinSize);
    if (!reduced) {
        // PlanZstdTexture checks the start of the file only: drop the levels it planned for even
        // when the relocation table at the end can't follow.
        reduced = Bntx::Reduce(bntx, levels, TextureQualityMinSize, true);
        if (!reduced) {
            return std::nullopt;
        }
        LOG_WARNING(Common_Memory, "Texture quality: relocation table of a texture left as it is");
    }
    ReducedTexture out;
    out.file = MakeRawZstdFrame(*reduced);
    out.bntx_size = bntx.size();
    out.reduced_bntx_size = reduced->size();
    return out;
}

namespace {

struct TextureQualityState {
    explicit TextureQualityState(u32 levels_) : levels{levels_} {
        pressure_listener = Common::PlatformEvents::AddMemoryPressureListener([this] {
            std::scoped_lock lock{mutex};
            lru.clear();
            bytes = 0;
        });
    }

    ~TextureQualityState() {
        Common::PlatformEvents::RemoveMemoryPressureListener(pressure_listener);
    }

    const std::vector<u8>* Find(const void* key) {
        const auto it = std::find_if(lru.begin(), lru.end(),
                                     [key](const auto& entry) { return entry.first == key; });
        if (it == lru.end()) {
            return nullptr;
        }
        lru.splice(lru.begin(), lru, it);
        return &lru.front().second;
    }

    void Insert(const void* key, std::vector<u8> data) {
        if (data.size() > CacheCap || Find(key) != nullptr) {
            return;
        }
        while (bytes + data.size() > CacheCap && !lru.empty()) {
            bytes -= lru.back().second.size();
            lru.pop_back();
        }
        bytes += data.size();
        lru.emplace_front(key, std::move(data));
    }

    void Erase(const void* key) {
        std::scoped_lock lock{mutex};
        const auto it = std::find_if(lru.begin(), lru.end(),
                                     [key](const auto& entry) { return entry.first == key; });
        if (it != lru.end()) {
            bytes -= it->second.size();
            lru.erase(it);
        }
    }

    void Count(const ReducedTexture& reduced, std::chrono::steady_clock::duration time,
               bool partial) {
        work_us.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(time).count()));
        saved_bytes.fetch_add(reduced.bntx_size - reduced.reduced_bntx_size);
        if (partial) {
            partial_reads.fetch_add(1);
        }
        const u32 count = reduced_count.fetch_add(1) + 1;
        if (count == 1 || count % 1000 == 0) {
            LOG_INFO(Common_Memory,
                     "Texture quality: {} textures reduced so far, {} MiB less texture data, {} "
                     "ms spent, {} read in parts",
                     count, saved_bytes.load() >> 20, work_us.load() / 1000, partial_reads.load());
        }
    }

    const u32 levels;
    std::size_t pressure_listener{};
    std::mutex mutex;
    std::list<std::pair<const void*, std::vector<u8>>> lru;
    std::size_t bytes{};
    std::atomic<u32> reduced_count{};
    std::atomic<u32> partial_reads{};
    std::atomic<u64> saved_bytes{};
    std::atomic<u64> work_us{};
};

/// A `.bntx.zs` file as Settings texture_quality serves it, with the size PlanZstdTexture gave:
/// generated at each read that is not served from the cache, kept only while a read in parts
/// leaves the rest of it to read.
class TextureQualityFile final : public VfsFile {
public:
    TextureQualityFile(VirtualFile base_, std::shared_ptr<TextureQualityState> state_, u64 size_)
        : base{std::move(base_)}, state{std::move(state_)}, size{size_} {}

    ~TextureQualityFile() override {
        state->Erase(this);
    }

    std::string GetName() const override {
        return base->GetName();
    }

    std::size_t GetSize() const override {
        return static_cast<std::size_t>(size);
    }

    bool Resize(std::size_t) override {
        return false;
    }

    VirtualDir GetContainingDirectory() const override {
        return base->GetContainingDirectory();
    }

    bool IsWritable() const override {
        return false;
    }

    bool IsReadable() const override {
        return true;
    }

    std::size_t Read(u8* data, std::size_t length, std::size_t offset) const override {
        if (offset >= size) {
            return 0;
        }
        length = std::min<std::size_t>(length, size - offset);
        const bool whole = offset == 0 && length == size;
        if (!whole) {
            std::scoped_lock lock{state->mutex};
            if (const auto* cached = state->Find(this)) {
                std::memcpy(data, cached->data() + offset, length);
                return length;
            }
        }

        const auto begin = std::chrono::steady_clock::now();
        auto reduced = ReduceZstdTexture(base->ReadAllBytes(), state->levels);
        const auto time = std::chrono::steady_clock::now() - begin;
        if (!reduced || reduced->file.size() != size) {
            LOG_ERROR(Common_Memory, "Texture quality: {} could not be reduced as planned",
                      GetName());
            std::memset(data, 0, length);
            return length;
        }
        if (!counted.exchange(true)) {
            state->Count(*reduced, time, !whole);
        }
        std::memcpy(data, reduced->file.data() + offset, length);
        if (!whole) {
            std::scoped_lock lock{state->mutex};
            state->Insert(this, std::move(reduced->file));
        }
        return length;
    }

    std::size_t Write(const u8*, std::size_t, std::size_t) override {
        return 0;
    }

    bool Rename(std::string_view) override {
        return false;
    }

private:
    VirtualFile base;
    std::shared_ptr<TextureQualityState> state;
    u64 size;
    mutable std::atomic<bool> counted{};
};

bool IsTextureName(std::string_view name) {
    constexpr std::string_view Suffix = ".bntx.zs";
    return name.size() > Suffix.size() && name.substr(name.size() - Suffix.size()) == Suffix;
}

void Collect(const VirtualDir& dir, std::vector<VirtualFile>& out) {
    for (const auto& file : dir->GetFiles()) {
        if (IsTextureName(file->GetName())) {
            out.push_back(file);
        }
    }
    for (const auto& subdir : dir->GetSubdirectories()) {
        Collect(subdir, out);
    }
}

VirtualDir Wrap(const VirtualDir& dir, const std::shared_ptr<TextureQualityState>& state,
                const std::unordered_map<const VfsFile*, u64>& planned) {
    auto files = dir->GetFiles();
    auto subdirs = dir->GetSubdirectories();
    bool changed = false;
    for (auto& file : files) {
        if (const auto it = planned.find(file.get()); it != planned.end()) {
            file = std::make_shared<TextureQualityFile>(std::move(file), state, it->second);
            changed = true;
        }
    }
    for (auto& subdir : subdirs) {
        auto wrapped = Wrap(subdir, state, planned);
        if (wrapped != subdir) {
            subdir = std::move(wrapped);
            changed = true;
        }
    }
    if (!changed) {
        return dir;
    }
    return std::make_shared<VectorVfsDirectory>(std::move(files), std::move(subdirs),
                                                dir->GetName());
}

constexpr u32 PlanMagic = 0x31505154; // "TQP1"

/// Sizes planned at an earlier start for the same textures (same count and original sizes), or
/// empty.
std::vector<u64> LoadPlan(const std::filesystem::path& path, u32 levels,
                          const std::vector<u64>& original_sizes) {
    std::error_code ec;
    if (path.empty() || !std::filesystem::is_regular_file(path, ec)) {
        return {};
    }
    Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
    std::array<u32, 2> header{};
    u64 count{};
    if (!file.IsOpen() || file.ReadSpan(std::span<u32>{header}) != header.size() ||
        header[0] != PlanMagic || header[1] != levels || !file.ReadObject(count) ||
        count != original_sizes.size()) {
        return {};
    }
    std::vector<u64> records(count * 2);
    if (file.ReadSpan(std::span{records}) != records.size()) {
        return {};
    }
    std::vector<u64> sizes(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (records[2 * i] != original_sizes[i]) {
            return {};
        }
        sizes[i] = records[2 * i + 1];
    }
    return sizes;
}

void SavePlan(const std::filesystem::path& path, u32 levels, const std::vector<u64>& original_sizes,
              const std::vector<u64>& sizes) {
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    Common::FS::IOFile file{path, Common::FS::FileAccessMode::Write};
    const std::array<u32, 2> header{PlanMagic, levels};
    const u64 count = sizes.size();
    std::vector<u64> records(count * 2);
    for (std::size_t i = 0; i < count; ++i) {
        records[2 * i] = original_sizes[i];
        records[2 * i + 1] = sizes[i];
    }
    if (!file.IsOpen() || file.WriteSpan(std::span<const u32>{header}) != header.size() ||
        !file.WriteObject(count) ||
        file.WriteSpan(std::span<const u64>{records}) != records.size()) {
        file.Close();
        std::filesystem::remove(path, ec);
    }
}

} // namespace

VirtualDir ApplyTextureQuality(const VirtualDir& root, u32 levels,
                               const std::filesystem::path& plan_cache) {
    if (root == nullptr || levels == 0) {
        return root;
    }
    const auto begin = std::chrono::steady_clock::now();
    std::vector<VirtualFile> textures;
    Collect(root, textures);
    std::vector<u64> original_sizes(textures.size());
    for (std::size_t i = 0; i < textures.size(); ++i) {
        original_sizes[i] = textures[i]->GetSize();
    }

    // The RomFS needs each replacement's size now: from the last start, or by reading the start
    // of every texture.
    std::vector<u64> sizes = LoadPlan(plan_cache, levels, original_sizes);
    const bool cached = !sizes.empty();
    unsigned threads = 0;
    if (!cached) {
        sizes.resize(textures.size());
        std::atomic<std::size_t> next{};
        const auto work = [&] {
            for (std::size_t i = next.fetch_add(1); i < textures.size(); i = next.fetch_add(1)) {
                sizes[i] = PlanZstdTexture(*textures[i], levels).value_or(0);
            }
        };
        threads = std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
        std::vector<std::thread> workers;
        for (unsigned i = 1; i < threads; ++i) {
            workers.emplace_back(work);
        }
        work();
        for (auto& worker : workers) {
            worker.join();
        }
        SavePlan(plan_cache, levels, original_sizes, sizes);
    }

    std::unordered_map<const VfsFile*, u64> planned;
    u64 before = 0;
    u64 after = 0;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (sizes[i] != 0) {
            planned.emplace(textures[i].get(), sizes[i]);
            before += textures[i]->GetSize();
            after += sizes[i];
        }
    }
    const auto state = std::make_shared<TextureQualityState>(levels);
    auto wrapped = planned.empty() ? root : Wrap(root, state, planned);
    LOG_INFO(Common_Memory,
             "Texture quality {}: {} of {} .bntx.zs files get {} top mip level(s) dropped when "
             "the game reads them ({} -> {} MiB as stored; {} in {} ms)",
             levels == 1 ? "Half" : "Quarter", planned.size(), textures.size(), levels,
             before >> 20, after >> 20,
             cached ? std::string{"plan from the last start"}
                    : fmt::format("planned on {} threads", threads),
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - begin)
                 .count());
    return wrapped;
}

} // namespace FileSys
