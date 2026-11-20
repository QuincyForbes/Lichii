// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/loader/loader.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/shader_warm/emission_stamp.h"
#include "video_core/shader_warm/shader_archives.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"

#if defined(__APPLE__) && defined(LICHII_METAL_RENDERER)
namespace Metal {
struct OfflineWarmSummary {
    bool ok{};
    size_t built{};
    size_t skipped{};
    size_t failed{};
    std::string error;
};
OfflineWarmSummary CompileWarmShadersOffline(
    u64 title_id, std::stop_token stop,
    const std::function<void(size_t done, size_t total, std::string_view stage)>& on_progress = {});
bool MetalWarmPending(u64 title_id);
} // namespace Metal
#endif

namespace Vulkan {
struct OfflineWarmSummary {
    bool ok{};
    size_t built{};
    size_t skipped{};
    size_t failed{};
    std::string error;
};
OfflineWarmSummary CompileWarmShadersOffline(
    u64 title_id, std::stop_token stop,
    const std::function<void(size_t done, size_t total, std::string_view stage)>& on_progress = {});
bool VulkanWarmPending(u64 title_id);
} // namespace Vulkan

namespace VideoCommon::ShaderWarm {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view StageReading = "Reading game files";
constexpr std::string_view StageExtracting = "Extracting shaders";
constexpr std::string_view StageConverting = "Converting";
constexpr std::string_view StageWriting = "Writing";
/// Share of the progress bar for reading and extracting the archives; conversions run alongside
/// and only their tail after the last archive shows in the rest.
constexpr double ExtractionShare = 0.90;
/// Queued unique conversions in flight (bounded so peak RAM stays modest).
constexpr size_t MaxQueuedJobs = 4096;
constexpr u32 PrecachedPipelinesMagic = 0x31504750; // "PGP1"

fs::path TitleDir(u64 title_id) {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) /
           fmt::format("{:016x}", title_id);
}

constexpr u64 Switch1CounterpartBase(u64 base_id) noexcept {
    constexpr u64 platform_mask = 0xFF00000000000000ULL;
    constexpr u64 switch2_platform = 0x0400000000000000ULL;
    constexpr u64 switch1_platform = 0x0100000000000000ULL;
    if ((base_id & platform_mask) != switch2_platform) {
        return base_id;
    }
    return (base_id & ~platform_mask) | switch1_platform;
}

struct ArchiveFile {
    std::string path; ///< Inside the RomFS, '/' separated (the archive identity)
    FileSys::VirtualFile file;
    bool needs_vfs_lock{};
};

/// Appends every file under `dir` to `out`, recursively, with its '/' separated path.
void ListFiles(const FileSys::VirtualDir& dir, const std::string& prefix, bool needs_vfs_lock,
               std::vector<ArchiveFile>& out) {
    for (const auto& file : dir->GetFiles()) {
        out.push_back({prefix + file->GetName(), file, needs_vfs_lock});
    }
    for (const auto& sub : dir->GetSubdirectories()) {
        ListFiles(sub, prefix + sub->GetName() + "/", needs_vfs_lock, out);
    }
}

struct RomFsRoot {
    FileSys::VirtualDir dir;
    bool needs_vfs_lock{};
};

/// The game's RomFS trees: a .nup gives its base, newest update and newest add-on (DLC) packages;
/// other formats give the loader's RomFS plus any installed AOC RomFS for the title.
struct GameFiles {
    std::unique_ptr<FileSys::NupContainer> nup;
    std::unique_ptr<Loader::AppLoader> loader;
    std::vector<RomFsRoot> romfs;
};

/// Opens the game file and collects its RomFS trees into `game`. Returns the error text when the
/// file cannot be opened or has no RomFS.
std::optional<std::string> OpenGame(Core::System& system, const fs::path& game_path, u64 title_id,
                                    GameFiles& game, std::stop_token stop) {
    const bool has_system_provider = system.GetFilesystem() != nullptr;
    FileSys::VirtualFilesystem filesystem = system.GetFilesystem();
    if (filesystem == nullptr) {
        filesystem = std::make_shared<FileSys::RealVfsFilesystem>();
    }
    const auto file =
        filesystem->OpenFile(Common::FS::PathToUTF8String(game_path), FileSys::OpenMode::Read);
    if (file == nullptr) {
        return "Cannot open the game file";
    }
    const u64 own_base = FileSys::GetBaseTitleID(title_id);
    const u64 s1_base = Switch1CounterpartBase(own_base);
    std::unordered_set<u64> loaded_aoc_ids;
    if (FileSys::NupContainer::IsNupFile(file)) {
        // Every member is read once across producer threads.
        game.nup = std::make_unique<FileSys::NupContainer>(file, 128 * 1024 * 1024);
        if (game.nup->GetStatus() != Loader::ResultStatus::Success) {
            return "The game file is not a valid package";
        }
        std::vector<size_t> packages;
        if (const auto base = game.nup->FindBasePackage()) {
            packages.push_back(*base);
        }
        if (const auto update = game.nup->FindBestUpdatePackage()) {
            packages.push_back(*update);
        }
        const auto& all_packages = game.nup->GetPackages();
        std::map<u64, size_t> newest_addons;
        for (size_t i = 0; i < all_packages.size(); ++i) {
            const auto& pkg = all_packages[i];
            if (pkg.kind != FileSys::NupContainer::PackageKind::Addon) {
                continue;
            }
            const u64 pkg_base = FileSys::GetBaseTitleID(pkg.title_id);
            if (title_id != 0 && pkg_base != own_base && pkg_base != s1_base) {
                continue;
            }
            const auto [it, inserted] = newest_addons.try_emplace(pkg.title_id, i);
            if (!inserted && pkg.version > all_packages[it->second].version) {
                it->second = i;
            }
        }
        for (const auto& [addon_id, pkg_index] : newest_addons) {
            packages.push_back(pkg_index);
            loaded_aoc_ids.insert(addon_id);
        }
        std::sort(packages.begin(), packages.end());
        packages.erase(std::unique(packages.begin(), packages.end()), packages.end());
        for (const size_t package : packages) {
            if (stop.stop_requested()) {
                return "Stopped";
            }
            if (auto dir = game.nup->GetRomFSDir(package)) {
                game.romfs.push_back({std::move(dir), false});
            }
        }
    } else {
        game.loader = Loader::GetLoader(system, file);
        if (game.loader == nullptr) {
            return "Unsupported game file format";
        }
        FileSys::VirtualFile romfs;
        if (game.loader->ReadRomFS(romfs) != Loader::ResultStatus::Success || romfs == nullptr) {
            return "Cannot read the game's RomFS (are the keys installed?)";
        }
        if (stop.stop_requested()) {
            return "Stopped";
        }
        if (auto dir = FileSys::ExtractRomFS(romfs)) {
            game.romfs.push_back({std::move(dir), true});
        }
    }
    if (has_system_provider && title_id != 0) {
        const auto& provider = system.GetContentProvider();
        auto aoc_entries =
            provider.ListEntriesFilter(FileSys::TitleType::AOC, FileSys::ContentRecordType::Data);
        std::sort(aoc_entries.begin(), aoc_entries.end(),
                  [](const auto& a, const auto& b) { return a.title_id < b.title_id; });
        for (const auto& entry : aoc_entries) {
            if (stop.stop_requested()) {
                return "Stopped";
            }
            if (loaded_aoc_ids.contains(entry.title_id)) {
                continue;
            }
            const u64 entry_base = FileSys::GetBaseTitleID(entry.title_id);
            if (entry_base != own_base && entry_base != s1_base) {
                continue;
            }
            const auto nca = provider.GetEntry(entry.title_id, FileSys::ContentRecordType::Data);
            if (!nca || nca->GetStatus() != Loader::ResultStatus::Success) {
                continue;
            }
            if (const auto aoc_romfs = nca->GetRomFS()) {
                if (auto dir = FileSys::ExtractRomFS(aoc_romfs)) {
                    game.romfs.push_back({std::move(dir), true});
                    loaded_aoc_ids.insert(entry.title_id);
                }
            }
        }
    }
    if (game.romfs.empty()) {
        return "The game has no RomFS";
    }
    return std::nullopt;
}

u128 Hash128(std::span<const u8> bytes) {
    return Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

u128 HashGraphicsIdentity(u32 stage, bool maxwell, std::span<const u8> prefix) {
    std::array<u8, 88> head{};
    head[0] = static_cast<u8>(stage);
    head[1] = maxwell ? 1 : 0;
    if (maxwell) {
        std::memcpy(head.data() + 8, prefix.data() + 0x30, 80);
        const u128 seed = Hash128(head);
        const auto body = prefix.subspan(0x80);
        return Common::CityHash128WithSeed(reinterpret_cast<const char*>(body.data()), body.size(),
                                           seed);
    }
    std::memcpy(head.data() + 8, prefix.data(), 80);
    const u128 seed = Hash128(head);
    const auto body = prefix.subspan(128);
    return Common::CityHash128WithSeed(reinterpret_cast<const char*>(body.data()), body.size(),
                                       seed);
}

using Origin = std::pair<u32, u32>; ///< (archive_index, intra_archive_record_index)

struct ComputeDedupKey {
    bool maxwell{};
    u128 code_hash{};
    ComputeControl control{};
    bool operator==(const ComputeDedupKey&) const = default;
};

struct ComputeDedupHasher {
    size_t operator()(const ComputeDedupKey& k) const noexcept {
        u64 h = k.code_hash[0] ^ (k.code_hash[1] * 0x9E3779B97F4A7C15ULL);
        h ^= (u64{k.control.workgroup[0]} << 32) | k.control.workgroup[1];
        h ^= (u64{k.control.workgroup[2]} << 32) | k.control.shared_memory;
        h ^= u64{k.control.local_memory} * 0xBF58476D1CE4E5B9ULL;
        if (k.maxwell) {
            h ^= 0x94D049BB133111EBULL;
        }
        return static_cast<size_t>(h);
    }
};

struct U128Hasher {
    size_t operator()(const u128& h) const noexcept {
        return static_cast<size_t>(h[0] ^ (h[1] * 0x9E3779B97F4A7C15ULL));
    }
};

struct GraphicsRawEntry {
    u32 stage{};
    bool maxwell{};
    Origin best_origin{~0U, ~0U};
    std::array<u8, 80> sph{};
    std::vector<u8> best_bank1;
    std::vector<u32> archive_ids;
    LoweredCode lowered;
    bool lowered_ok{};
};

struct LowerJob {
    bool graphics{};
    bool maxwell{};
    size_t graphics_index{};
    ComputeDedupKey compute_id{};
    std::vector<u8> program;
};

struct FinishedCompute {
    ComputeKey key{};
    ComputeDedupKey id{};
    std::vector<u8> bytes;
};

struct FinishedGraphics {
    size_t index{};
    LoweredCode lowered;
};

/// Multi-core lowering pool that executes unique shader jobs and collects completed results
/// out-of-order into per-worker vectors with zero output lock contention.
class LowerPool {
public:
    LowerPool(size_t thread_count, std::stop_token stop_, std::atomic<size_t>& completed_,
              std::atomic<size_t>& skipped_)
        : stop{std::move(stop_)}, completed{completed_}, skipped{skipped_},
          worker_compute(thread_count), worker_graphics(thread_count),
          stop_callback{stop, StopWake{this}} {
        workers.reserve(thread_count);
        for (size_t i = 0; i < thread_count; ++i) {
            workers.emplace_back([this, i] { Work(i); });
        }
    }

    ~LowerPool() {
        CloseAndJoin();
    }

    bool Push(LowerJob job) {
        {
            std::unique_lock lock{mutex};
            not_full_cv.wait(lock, [this] {
                return closing || stop.stop_requested() || jobs.size() < MaxQueuedJobs;
            });
            if (closing || stop.stop_requested()) {
                return false;
            }
            jobs.push_back(std::move(job));
        }
        not_empty_cv.notify_one();
        return true;
    }

    void CloseAndJoin() {
        {
            std::scoped_lock lock{mutex};
            if (joined) {
                return;
            }
            closing = true;
        }
        not_empty_cv.notify_all();
        not_full_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        std::scoped_lock lock{mutex};
        joined = true;
    }

    std::vector<FinishedCompute> TakeCompute() {
        std::vector<FinishedCompute> merged;
        size_t total = 0;
        for (const auto& v : worker_compute) {
            total += v.size();
        }
        merged.reserve(total);
        for (auto& v : worker_compute) {
            for (auto& item : v) {
                merged.push_back(std::move(item));
            }
            v.clear();
        }
        return merged;
    }

    void ApplyGraphics(std::deque<GraphicsRawEntry>& entries) {
        for (auto& v : worker_graphics) {
            for (auto& item : v) {
                auto& target = entries[item.index];
                target.lowered = std::move(item.lowered);
                target.lowered_ok = true;
            }
            v.clear();
        }
    }

private:
    void Work(size_t worker_id) {
        Common::SetCurrentThreadName("ShaderWarmBuild");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        auto& out_compute = worker_compute[worker_id];
        auto& out_graphics = worker_graphics[worker_id];
        for (;;) {
            LowerJob job;
            {
                std::unique_lock lock{mutex};
                not_empty_cv.wait(
                    lock, [this] { return closing || stop.stop_requested() || !jobs.empty(); });
                if (stop.stop_requested() || jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            not_full_cv.notify_one();

            try {
                LoweredCode lowered;
                const auto error = job.maxwell
                                       ? PrepareMaxwellProgram(job.program, job.graphics, lowered)
                                       : LowerProgram(job.program, job.graphics, lowered);
                if (error) {
                    LOG_DEBUG(Render_Vulkan, "Shader precache: program not converted: {}", *error);
                    skipped.fetch_add(1, std::memory_order_relaxed);
                } else if (job.graphics) {
                    out_graphics.push_back({job.graphics_index, std::move(lowered)});
                } else {
                    FinishedCompute item;
                    item.id = job.compute_id;
                    item.bytes = ComputeEntry(lowered, job.compute_id.control, item.key);
                    out_compute.push_back(std::move(item));
                }
            } catch (const std::exception& e) {
                LOG_DEBUG(Render_Vulkan, "Shader precache: conversion failed: {}", e.what());
                skipped.fetch_add(1, std::memory_order_relaxed);
            }
            completed.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::stop_token stop;
    std::atomic<size_t>& completed;
    std::atomic<size_t>& skipped;
    std::mutex mutex;
    std::condition_variable not_empty_cv;
    std::condition_variable not_full_cv;
    std::deque<LowerJob> jobs;
    bool closing{};
    bool joined{};
    std::vector<std::vector<FinishedCompute>> worker_compute;
    std::vector<std::vector<FinishedGraphics>> worker_graphics;
    std::vector<std::thread> workers;
    struct StopWake {
        LowerPool* pool;
        void operator()() const noexcept {
            {
                std::scoped_lock lock{pool->mutex};
            }
            pool->not_empty_cv.notify_all();
            pool->not_full_cv.notify_all();
        }
    };
    std::stop_callback<StopWake> stop_callback;
};

template <typename T>
void Write(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void WriteHeader(std::ofstream& out) {
    out.write("yuzucach", 8);
    Write(out, WarmCacheVersion);
}

std::atomic<bool> warm_before_start{false};
std::atomic<bool> skip_warm{false};
std::atomic<bool> build_only{false};

/// Records `entries` for the finished precache file at `warm_file` (see WarmCountPath).
void WriteWarmCount(const fs::path& warm_file, u64 entries) {
    std::error_code ec;
    const auto size = fs::file_size(warm_file, ec);
    if (ec) {
        return;
    }
    std::ofstream out(WarmCountPath(warm_file), std::ios::trunc);
    out << entries << ' ' << size << ' ' << WarmBuilderVersion << '\n';
}

bool EndsWithIgnoreCase(std::string_view text, std::string_view suffix) {
    if (text.size() < suffix.size()) {
        return false;
    }
    const auto tail = text.substr(text.size() - suffix.size());
    for (size_t i = 0; i < suffix.size(); ++i) {
        const char a = tail[i] >= 'A' && tail[i] <= 'Z' ? static_cast<char>(tail[i] + 32) : tail[i];
        if (a != suffix[i]) {
            return false;
        }
    }
    return true;
}

} // Anonymous namespace

fs::path ComputeWarmPath(u64 title_id) {
    return TitleDir(title_id) / "warm-compute.bin";
}

fs::path GraphicsWarmPath(u64 title_id) {
    return TitleDir(title_id) / "warm-graphics.bin";
}

bool GraphicsPrecacheRequested() {
    const char* value = std::getenv("EDEN_NX2_GRAPHICS_PRECACHE");
    if (value && *value) {
        const std::string_view sv{value};
        return sv != "0" && sv != "false" && sv != "FALSE" && sv != "off" && sv != "OFF";
    }
    return true;
}

fs::path WarmMarkerPath(const fs::path& warm_file) {
    fs::path marker = warm_file;
    marker.replace_extension(".done");
    return marker;
}

fs::path WarmCountPath(const fs::path& warm_file) {
    fs::path count = warm_file;
    count.replace_extension(".count");
    return count;
}

u64 ReadWarmCount(const fs::path& warm_file) {
    std::ifstream in(WarmCountPath(warm_file));
    u64 entries{};
    u64 size{};
    if (!(in >> entries >> size)) {
        return 0;
    }
    std::error_code ec;
    const auto actual = fs::file_size(warm_file, ec);
    return !ec && actual == size ? entries : 0;
}

u64 EmissionStamp() {
    return GeneratedEmissionStamp;
}

std::optional<WarmStamp> CurrentWarmStamp(const fs::path& warm_file, u64 entries, u32 driver,
                                          u32 device_id, u32 cache_version, u64 variant) {
    std::error_code ec;
    const auto size = fs::file_size(warm_file, ec);
    if (ec) {
        return std::nullopt;
    }
    std::ifstream in(warm_file, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::vector<char> chunk(8U << 20);
    u64 hash = size;
    u64 read_total = 0;
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto got = static_cast<size_t>(in.gcount());
        if (got == 0) {
            break;
        }
        hash = Common::CityHash64WithSeed(chunk.data(), got, hash);
        read_total += got;
    }
    if (read_total != size) {
        return std::nullopt;
    }
    return WarmStamp{.size = size,
                     .content_hash = hash,
                     .entries = entries,
                     .driver = driver,
                     .device_id = device_id,
                     .cache_version = cache_version,
                     .variant = variant};
}

std::optional<WarmStamp> ReadWarmMarker(const fs::path& warm_file) {
    std::ifstream in(WarmMarkerPath(warm_file));
    WarmStamp stamp;
    // A marker of the older format (last-write time, no variant) fails here: the programs not in
    // the cache are built once more.
    std::string format;
    if (!(in >> format) || format != "v2" ||
        !(in >> stamp.size >> stamp.content_hash >> stamp.entries >> stamp.driver >>
          stamp.device_id >> stamp.cache_version >> stamp.variant)) {
        return std::nullopt;
    }
    return stamp;
}

bool WriteWarmMarker(const fs::path& warm_file, const WarmStamp& stamp) {
    std::ofstream out(WarmMarkerPath(warm_file), std::ios::trunc);
    out << "v2 " << stamp.size << ' ' << stamp.content_hash << ' ' << stamp.entries << ' '
        << stamp.driver << ' ' << stamp.device_id << ' ' << stamp.cache_version << ' '
        << stamp.variant << '\n';
    out.close();
    return static_cast<bool>(out);
}

void DropStaleWarmMarker(const fs::path& warm_file, const fs::path& new_file) {
    if (const auto marker = ReadWarmMarker(warm_file)) {
        const auto incoming = CurrentWarmStamp(new_file, 0, 0, 0, 0);
        if (incoming && incoming->size == marker->size &&
            incoming->content_hash == marker->content_hash) {
            return;
        }
    }
    std::error_code ignored;
    fs::remove(WarmMarkerPath(warm_file), ignored);
}

bool WarmPending(u64 title_id) {
    const auto file_pending = [](const fs::path& file) {
        if (WarmFilesStale(file).has_value()) {
            return true;
        }
        std::error_code ec;
        return !fs::exists(WarmMarkerPath(file), ec) || !ReadWarmMarker(file).has_value();
    };
    if (file_pending(ComputeWarmPath(title_id))) {
        return true;
    }
    if (GraphicsPrecacheRequested() && file_pending(GraphicsWarmPath(title_id))) {
        return true;
    }
    return false;
}

bool OfflineGpuWarmPending(u64 title_id) {
    const auto backend = Settings::values.renderer_backend.GetValue();
    if (backend == Settings::RendererBackend::Vulkan) {
        return ::Vulkan::VulkanWarmPending(title_id);
    }
#if defined(__APPLE__) && defined(LICHII_METAL_RENDERER)
    if (backend == Settings::RendererBackend::Metal) {
        return ::Metal::MetalWarmPending(title_id);
    }
#endif
    return false;
}

bool HasWarmFile(u64 title_id) {
    std::error_code ec;
    return fs::exists(ComputeWarmPath(title_id), ec) ||
           fs::exists(GraphicsWarmPath(title_id), ec);
}

std::optional<std::string> WarmFilesStale(const fs::path& warm_file) {
    std::error_code ec;
    if (!fs::exists(warm_file, ec)) {
        return "none built yet";
    }
    {
        std::ifstream in(warm_file, std::ios::binary);
        std::array<char, 8> magic{};
        u32 version{};
        in.read(magic.data(), magic.size()).read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!in || version != WarmCacheVersion) {
            return "built for another cache version";
        }
    }
    std::ifstream in(WarmCountPath(warm_file));
    u64 entries{};
    u64 size{};
    u32 builder{};
    if (!(in >> entries >> size >> builder) || builder != WarmBuilderVersion) {
        return "built by an older version";
    }
    if (fs::file_size(warm_file, ec) != size || ec) {
        return "changed since it was built";
    }
    return std::nullopt;
}

bool RefreshWarmFiles(Core::System& system, u64 title_id, std::stop_token stop) {
    if (title_id == 0 ||
        Settings::values.precache_shaders.GetValue() != Settings::PrecacheShaders::Always) {
        return false;
    }
    auto reason = WarmFilesStale(ComputeWarmPath(title_id));
    if (!reason && GraphicsPrecacheRequested()) {
        reason = WarmFilesStale(GraphicsWarmPath(title_id));
    }
    if (!reason) {
        return false;
    }
    const std::string& file_path = system.GetCurrentApplicationFilePath();
    if (file_path.empty()) {
        return false;
    }
    fs::path game_path = Common::FS::ToU8String(file_path);
    // Loose dumps run from their exefs folder; the loader opens its main.
    if (std::error_code ec; fs::is_directory(game_path, ec)) {
        game_path /= "main";
    }
    LOG_INFO(Render_Vulkan, "Shader precache for {:016x}: {}, building it", title_id, *reason);
    return BuildWarmFiles(system, game_path, title_id, {}, std::move(stop)).ok;
}

void RequestWarmBeforeStart(bool before_start) {
    skip_warm.store(false, std::memory_order_relaxed);
    warm_before_start.store(before_start, std::memory_order_relaxed);
}

bool TakeWarmBeforeStart() {
    return warm_before_start.exchange(false, std::memory_order_relaxed);
}

void RequestSkipWarm() {
    skip_warm.store(true, std::memory_order_relaxed);
}

bool TakeSkipWarm() {
    return skip_warm.exchange(false, std::memory_order_relaxed);
}

void RequestBuildOnly(bool only) {
    build_only.store(only, std::memory_order_relaxed);
    if (only) {
        skip_warm.store(false, std::memory_order_relaxed);
    }
}

bool BuildOnlyRequested() {
    return build_only.load(std::memory_order_relaxed);
}

bool TakeBuildOnly() {
    return build_only.exchange(false, std::memory_order_relaxed);
}

namespace {
std::mutex build_status_mutex;
BuildStatus build_status;
} // Anonymous namespace

void SetBuildStatus(const BuildStatus& status) {
    std::scoped_lock lock{build_status_mutex};
    build_status = status;
}

BuildStatus GetBuildStatus() {
    std::scoped_lock lock{build_status_mutex};
    return build_status;
}

fs::path PrecachedPipelinesPath(u64 title_id) {
    return TitleDir(title_id) / "precache-graphics.bin";
}

std::unordered_set<u64> ReadPrecachedPipelines(const fs::path& file, const DriverStamp& stamp) {
    std::unordered_set<u64> hashes;
    std::ifstream in(file, std::ios::binary);
    u32 magic{};
    DriverStamp saved;
    u64 count{};
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        magic != PrecachedPipelinesMagic ||
        !in.read(reinterpret_cast<char*>(&saved.driver), sizeof(saved.driver)) ||
        !in.read(reinterpret_cast<char*>(&saved.device_id), sizeof(saved.device_id)) ||
        !in.read(reinterpret_cast<char*>(&saved.cache_version), sizeof(saved.cache_version)) ||
        !in.read(reinterpret_cast<char*>(&count), sizeof(count)) || saved != stamp) {
        return hashes;
    }
    for (u64 i = 0; i < count; ++i) {
        u64 hash{};
        if (!in.read(reinterpret_cast<char*>(&hash), sizeof(hash))) {
            break;
        }
        hashes.insert(hash);
    }
    return hashes;
}

bool WritePrecachedPipelines(const fs::path& file, const DriverStamp& stamp,
                             const std::unordered_set<u64>& hashes) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    Write(out, PrecachedPipelinesMagic);
    Write(out, stamp.driver);
    Write(out, stamp.device_id);
    Write(out, stamp.cache_version);
    Write(out, static_cast<u64>(hashes.size()));
    for (const u64 hash : hashes) {
        Write(out, hash);
    }
    out.close();
    return static_cast<bool>(out);
}

bool DeleteWarmFiles(u64 title_id, std::string* error) {
    const auto dir = TitleDir(title_id);
    const auto warm = ComputeWarmPath(title_id);
    fs::path tmp = warm;
    tmp += ".tmp";
    const auto graphics = GraphicsWarmPath(title_id);
    fs::path graphics_tmp = graphics;
    graphics_tmp += ".tmp";
    const auto slow = dir / "precache-slow.bin";
    const auto metal_stages_v1 = dir / "metal_stages.v1.bin";
    fs::path metal_stages_v1_tmp = metal_stages_v1;
    metal_stages_v1_tmp += ".tmp";
    const auto metal_stages_v2 = dir / "metal_stages.v2.bin";
    fs::path metal_stages_v2_tmp = metal_stages_v2;
    metal_stages_v2_tmp += ".tmp";
    const auto metal_ledger = dir / "metal_warm_ledger.v1.bin";
    fs::path metal_ledger_tmp = metal_ledger;
    metal_ledger_tmp += ".tmp";
    const auto metal_pipelines = dir / "metal_pipelines.bin";
    fs::path metal_pipelines_tmp = metal_pipelines;
    metal_pipelines_tmp += ".tmp";
    bool ok = true;
    for (const auto& path :
         {warm, WarmMarkerPath(warm), WarmCountPath(warm), tmp, graphics, WarmMarkerPath(graphics),
          WarmCountPath(graphics), graphics_tmp, PrecachedPipelinesPath(title_id), slow,
          metal_stages_v1, metal_stages_v1_tmp, metal_stages_v2, metal_stages_v2_tmp,
          metal_ledger, metal_ledger_tmp, metal_pipelines, metal_pipelines_tmp}) {
        std::error_code ec;
        fs::remove(path, ec);
        if (ec) {
            ok = false;
            if (error != nullptr) {
                *error = ec.message();
            }
        }
    }
    for (size_t shard = 1; shard < 64; ++shard) {
        std::error_code ec;
        fs::remove(dir / fmt::format("metal_pipelines.s{:02d}.bin", shard), ec);
    }
    return ok;
}

GpuWarmSummary CompileWarmShadersOffline(u64 title_id, std::stop_token stop,
                                         const std::function<void(const Progress&)>& on_progress) {
    std::function<void(size_t, size_t, std::string_view)> gpu_progress;
    if (on_progress) {
        gpu_progress = [&on_progress](size_t done, size_t total, std::string_view stage) {
            on_progress(Progress{
                .stage = std::string{stage},
                .done = done,
                .total = total,
                .fraction =
                    total != 0 ? static_cast<double>(done) / static_cast<double>(total) : 0.0,
            });
        };
    }
    const auto backend = Settings::values.renderer_backend.GetValue();
    if (backend == Settings::RendererBackend::Vulkan) {
        const auto summary =
            ::Vulkan::CompileWarmShadersOffline(title_id, std::move(stop), gpu_progress);
        return GpuWarmSummary{
            .supported = true,
            .ok = summary.ok,
            .built = summary.built,
            .skipped = summary.skipped,
            .failed = summary.failed,
            .error = summary.error,
        };
    }
#if defined(__APPLE__) && defined(LICHII_METAL_RENDERER)
    if (backend == Settings::RendererBackend::Metal) {
        const auto summary =
            ::Metal::CompileWarmShadersOffline(title_id, std::move(stop), gpu_progress);
        return GpuWarmSummary{
            .supported = true,
            .ok = summary.ok,
            .built = summary.built,
            .skipped = summary.skipped,
            .failed = summary.failed,
            .error = summary.error,
        };
    }
#endif
    return GpuWarmSummary{.supported = false};
}

Result BuildWarmFiles(Core::System& system, const fs::path& game_path, u64 title_id,
                      const std::function<void(const Progress&)>& progress,
                      std::stop_token stop) {
    return BuildWarmFilesWith(system, game_path, title_id, BuildOptions{}, progress,
                              std::move(stop));
}

Result BuildWarmFilesWith(Core::System& system, const fs::path& game_path, u64 title_id,
                          const BuildOptions& options,
                          const std::function<void(const Progress&)>& progress,
                          std::stop_token stop) {
    Result result;
    u64 total_bytes = 0;
    std::atomic<u64> bytes_done{0};
    std::atomic<size_t> jobs_pushed{0};
    std::atomic<size_t> jobs_completed{0};
    std::atomic<size_t> skipped_count{0};
    double reported_fraction = 0.0;
    auto last_report = std::chrono::steady_clock::time_point{};
    const auto report = [&](std::string_view stage, size_t done, size_t total, bool force = false) {
        if (!progress) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const bool last = force || (total != 0 && done == total);
        if (!last && now - last_report < std::chrono::milliseconds{50}) {
            return;
        }
        last_report = now;
        const u64 b_done = bytes_done.load(std::memory_order_relaxed);
        const size_t pushed = jobs_pushed.load(std::memory_order_relaxed);
        const size_t completed = jobs_completed.load(std::memory_order_relaxed);
        const double extracted =
            total_bytes != 0 ? static_cast<double>(b_done) / static_cast<double>(total_bytes) : 0.0;
        double fraction = ExtractionShare * extracted;
        if (stage == StageConverting) {
            fraction = ExtractionShare + (1.0 - ExtractionShare) *
                                             (pushed != 0 ? static_cast<double>(completed) /
                                                                static_cast<double>(pushed)
                                                          : 1.0);
        } else if (stage == StageWriting) {
            fraction = (total != 0 && done == total) ? 1.0 : std::max(reported_fraction, 0.98);
        }
        reported_fraction = std::max(reported_fraction, std::min(fraction, 1.0));
        progress(Progress{std::string(stage), done, total, reported_fraction, pushed});
    };
    const auto begin = std::chrono::steady_clock::now();
    report(StageReading, 0, 0, true);

    const fs::path output_dir = options.output_dir.empty() ? TitleDir(title_id) : options.output_dir;
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    const fs::path compute_path = output_dir / "warm-compute.bin";
    const fs::path graphics_path = output_dir / "warm-graphics.bin";
    fs::path compute_tmp = compute_path;
    compute_tmp += ".tmp";
    fs::path graphics_tmp = graphics_path;
    graphics_tmp += ".tmp";
    std::ofstream compute_out;
    std::ofstream graphics_out;
    const auto fail = [&](std::string error) {
        compute_out.close();
        graphics_out.close();
        std::error_code ignored;
        fs::remove(compute_tmp, ignored);
        fs::remove(graphics_tmp, ignored);
        result.ok = false;
        result.error = std::move(error);
        LOG_WARNING(Render_Vulkan, "Shader precache build for {:016x} failed: {}", title_id,
                    result.error);
        return result;
    };

    const bool build_graphics = options.graphics.value_or(GraphicsPrecacheRequested());

    // Draw state guesses.
    std::vector<fs::path> caches;
    if (options.runtime_caches) {
        caches = *options.runtime_caches;
    } else if (build_graphics) {
        const fs::path title_dir = TitleDir(title_id);
        const fs::path vk_cache_dir = Common::FS::GetVulkanPipelineCacheDir(
            Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir), title_id);
        std::vector<fs::path> dirs{vk_cache_dir};
        if (vk_cache_dir != title_dir) {
            dirs.push_back(title_dir);
        }
        std::vector<fs::path> candidates;
        for (const auto& dir : dirs) {
            candidates.push_back(VideoCommon::VersionedPipelineCachePath(
                dir, "vulkan-frizti", VideoCommon::VulkanPipelineCacheVersion));
            candidates.push_back(VideoCommon::VersionedPipelineCachePath(
                dir, "vulkan", VideoCommon::VulkanPipelineCacheVersion));
            for (const u32 version : CompatibleRuntimeCacheVersions) {
                candidates.push_back(
                    VideoCommon::VersionedPipelineCachePath(dir, "vulkan", version));
            }
        }
        for (const auto& candidate : candidates) {
            if (fs::exists(candidate, ec)) {
                caches.push_back(candidate);
                break;
            }
        }
    }
    const RuntimeStats stats = ReadRuntimeStats(caches);
    const bool use_stats = stats.occurrences != 0;
    const GraphicsKey vertex_template = stats.templates[0] ? *stats.templates[0]
                                                           : DefaultTemplate(true);
    const GraphicsKey fragment_template = stats.templates[1] ? *stats.templates[1]
                                                             : DefaultTemplate(false);
    if (build_graphics) {
        LOG_INFO(Render_Vulkan,
                 "Shader precache build for {:016x}: {} runtime stage records from {} cache(s){}",
                 title_id, stats.occurrences, caches.size(),
                 use_stats ? "" : ", built-in draw state defaults");
    }

    GameFiles game;
    if (const auto error = OpenGame(system, game_path, title_id, game, stop)) {
        return fail(*error);
    }
    if (stop.stop_requested()) {
        return fail("Stopped");
    }

    // Archive list: every shader archive of each package, then each package's effects, sorted by
    // path inside a package (see ClassifyGameFile).
    std::vector<ArchiveFile> archives;
    std::vector<ArchiveFile> effects;
    ZstdDictionaries dictionaries;
    FileSys::VirtualFile dictionary_pack;
    for (const auto& romfs : game.romfs) {
        std::vector<ArchiveFile> files;
        ListFiles(romfs.dir, "", romfs.needs_vfs_lock, files);
        std::sort(files.begin(), files.end(),
                  [](const ArchiveFile& a, const ArchiveFile& b) { return a.path < b.path; });
        std::vector<ArchiveFile> package_effects;
        for (auto& file : files) {
            switch (ClassifyGameFile(file.path)) {
            case GameFileKind::ShaderArchive:
                archives.push_back(file);
                break;
            case GameFileKind::Effect:
                package_effects.push_back(file);
                break;
            case GameFileKind::ZstdDictionaries:
                dictionary_pack = file.file; // the newest package wins
                break;
            case GameFileKind::Other:
                break;
            }
        }
        effects.insert(effects.end(), package_effects.begin(), package_effects.end());
    }
    archives.insert(archives.end(), effects.begin(), effects.end());
    effects.clear();
    if (dictionary_pack) {
        try {
            dictionaries.Load(dictionary_pack->ReadAllBytes());
        } catch (const ArchiveError& e) {
            LOG_WARNING(Render_Vulkan, "Shader precache: zstd dictionary pack: {}", e.what());
        }
        dictionary_pack = nullptr;
    }

    result.archives = archives.size();
    std::vector<u32> archive_id_by_index(archives.size());
    {
        std::unordered_map<std::string, u32> path_to_id;
        path_to_id.reserve(archives.size());
        for (size_t i = 0; i < archives.size(); ++i) {
            total_bytes += archives[i].file->GetSize();
            const auto [it, _] =
                path_to_id.try_emplace(archives[i].path, static_cast<u32>(path_to_id.size()));
            archive_id_by_index[i] = it->second;
        }
    }
    if (stop.stop_requested()) {
        return fail("Stopped");
    }
    const size_t hw = std::max<size_t>(std::thread::hardware_concurrency(), 2);
    const size_t lower_threads =
        options.threads != 0 ? options.threads : std::clamp<size_t>(hw, 2, 16);
    const size_t producer_threads =
        archives.empty() ? 0 : std::min<size_t>(archives.size(), std::clamp<size_t>(hw / 4, 2, 4));
    LOG_INFO(Render_Vulkan,
             "Shader precache build: {} archives, {} zstd dictionaries, {} reader and {} lowering "
             "threads",
             archives.size(), dictionaries.Count(), producer_threads, lower_threads);

    std::mutex vfs_read_mutex;
    std::mutex compute_mutex;
    std::unordered_map<ComputeDedupKey, Origin, ComputeDedupHasher> compute_seen;
    std::mutex graphics_mutex;
    std::unordered_map<u128, size_t, U128Hasher> graphics_map;
    std::deque<GraphicsRawEntry> graphics_entries;
    std::atomic<size_t> archive_errors{0};
    std::atomic<size_t> member_errors{0};
    std::atomic<size_t> next_archive{0};
    std::atomic<size_t> archives_done{0};

    auto lower_pool =
        std::make_unique<LowerPool>(lower_threads, stop, jobs_completed, skipped_count);

    {
        std::vector<std::thread> producers;
        producers.reserve(producer_threads);
        for (size_t t = 0; t < producer_threads; ++t) {
            producers.emplace_back([&] {
                Common::SetCurrentThreadName("ShaderWarmRead");
                Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
                for (;;) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    const size_t archive_index =
                        next_archive.fetch_add(1, std::memory_order_relaxed);
                    if (archive_index >= archives.size()) {
                        return;
                    }
                    const auto& archive = archives[archive_index];
                    const u64 archive_bytes = archive.file->GetSize();
                    bytes_done.fetch_add(archive_bytes / 2, std::memory_order_relaxed);

                    std::vector<u8> data;
                    {
                        std::unique_lock<std::mutex> read_lock{vfs_read_mutex, std::defer_lock};
                        if (archive.needs_vfs_lock) {
                            read_lock.lock();
                        }
                        // Skip large uncompressed audio/asset .pck files (Wwise AKPK / Godot GDPC)
                        // after a 4-byte magic peek instead of reading hundreds of MiB into RAM.
                        bool skip_pck = false;
                        if (archive_bytes > (4ULL << 20) &&
                            EndsWithIgnoreCase(archive.path, ".pck")) {
                            std::array<u8, 4> head{};
                            if (archive.file->Read(head.data(), head.size(), 0) == head.size() &&
                                (std::memcmp(head.data(), "AKPK", 4) == 0 ||
                                 std::memcmp(head.data(), "GDPC", 4) == 0)) {
                                skip_pck = true;
                            }
                        }
                        if (!skip_pck) {
                            data = archive.file->ReadAllBytes();
                        }
                    }

                    if (!data.empty() && !stop.stop_requested()) {
                        const auto name = archive.path.substr(archive.path.rfind('/') + 1);
                        const u32 archive_id = archive_id_by_index[archive_index];
                        u32 record_index = 0;
                        const auto on_record = [&](const ShaderRecord& record) {
                            const Origin origin{static_cast<u32>(archive_index), record_index++};
                            const bool is_graphics = build_graphics &&
                                                     (record.stage == ProgramStage::Vertex ||
                                                      record.stage == ProgramStage::Fragment);
                            if (record.stage != ProgramStage::Compute && !is_graphics) {
                                return;
                            }
                            if (is_graphics) {
                                const auto prefix_size =
                                    CaptureProgramPrefixSize(record.code, true, record.maxwell);
                                if (!prefix_size) {
                                    skipped_count.fetch_add(1, std::memory_order_relaxed);
                                    return;
                                }
                                const u32 stage =
                                    record.stage == ProgramStage::Vertex ? 0U : 4U;
                                const auto prefix = record.code.first(*prefix_size);
                                const u128 gfx_id =
                                    HashGraphicsIdentity(stage, record.maxwell, prefix);
                                const auto bank1 = ExtractBank1(
                                    record.code, record.control, record.maxwell, *prefix_size);
                                size_t entry_index = 0;
                                bool need_lower = false;
                                {
                                    std::scoped_lock lock{graphics_mutex};
                                    const auto [it, inserted] =
                                        graphics_map.try_emplace(gfx_id, graphics_entries.size());
                                    entry_index = it->second;
                                    if (inserted) {
                                        auto& entry = graphics_entries.emplace_back();
                                        entry.stage = stage;
                                        entry.maxwell = record.maxwell;
                                        entry.best_origin = origin;
                                        const auto sph_src = record.maxwell
                                                                 ? record.code.subspan(0x30, 80)
                                                                 : record.code.first(80);
                                        std::memcpy(entry.sph.data(), sph_src.data(), 80);
                                        entry.best_bank1.assign(bank1.begin(), bank1.end());
                                        entry.archive_ids.push_back(archive_id);
                                        need_lower = true;
                                    } else {
                                        auto& entry = graphics_entries[entry_index];
                                        entry.archive_ids.push_back(archive_id);
                                        if (origin < entry.best_origin) {
                                            entry.best_origin = origin;
                                            entry.best_bank1.assign(bank1.begin(), bank1.end());
                                        }
                                    }
                                }
                                if (need_lower) {
                                    LowerJob job;
                                    job.graphics = true;
                                    job.maxwell = record.maxwell;
                                    job.graphics_index = entry_index;
                                    job.program.assign(prefix.begin(), prefix.end());
                                    if (lower_pool->Push(std::move(job))) {
                                        jobs_pushed.fetch_add(1, std::memory_order_relaxed);
                                    }
                                }
                                return;
                            }

                            const auto values =
                                ReadComputeControl(record.control, record.maxwell);
                            if (!values || !values->workgroup[0] || !values->workgroup[1] ||
                                !values->workgroup[2]) {
                                skipped_count.fetch_add(1, std::memory_order_relaxed);
                                return;
                            }
                            const auto prefix_size =
                                CaptureProgramPrefixSize(record.code, false, record.maxwell);
                            if (!prefix_size) {
                                skipped_count.fetch_add(1, std::memory_order_relaxed);
                                return;
                            }
                            const auto prefix = record.code.first(*prefix_size);
                            const ComputeDedupKey ckey{record.maxwell, Hash128(prefix), *values};
                            bool need_lower = false;
                            {
                                std::scoped_lock lock{compute_mutex};
                                const auto [it, inserted] =
                                    compute_seen.try_emplace(ckey, origin);
                                if (inserted) {
                                    need_lower = true;
                                } else if (origin < it->second) {
                                    it->second = origin;
                                }
                            }
                            if (need_lower) {
                                LowerJob job;
                                job.graphics = false;
                                job.maxwell = record.maxwell;
                                job.compute_id = ckey;
                                job.program.assign(prefix.begin(), prefix.end());
                                if (lower_pool->Push(std::move(job))) {
                                    jobs_pushed.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        };
                        const auto on_member_error = [&](const std::string& error) {
                            member_errors.fetch_add(1, std::memory_order_relaxed);
                            LOG_DEBUG(Render_Vulkan, "Shader precache: {}: {}", archive.path,
                                      error);
                        };
                        try {
                            WalkArchive(std::move(data), name, &dictionaries, on_record,
                                        on_member_error, stop);
                        } catch (const ArchiveError& e) {
                            archive_errors.fetch_add(1, std::memory_order_relaxed);
                            LOG_WARNING(Render_Vulkan, "Shader precache: {}: {}", archive.path,
                                        e.what());
                        }
                    }

                    bytes_done.fetch_add(archive_bytes - archive_bytes / 2,
                                         std::memory_order_relaxed);
                    archives_done.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }

        // Calling thread drives progress reporting while producers walk archives.
        while (archives_done.load(std::memory_order_relaxed) < archives.size() &&
               !stop.stop_requested()) {
            report(StageExtracting, archives_done.load(std::memory_order_relaxed), archives.size());
            std::this_thread::sleep_for(std::chrono::milliseconds{25});
        }
        for (auto& producer : producers) {
            producer.join();
        }
    }
    if (stop.stop_requested()) {
        lower_pool.reset();
        return fail("Stopped");
    }
    report(StageExtracting, archives.size(), archives.size(), true);

    const size_t total_jobs = jobs_pushed.load(std::memory_order_relaxed);
    while (jobs_completed.load(std::memory_order_relaxed) < total_jobs && !stop.stop_requested()) {
        report(StageConverting, jobs_completed.load(std::memory_order_relaxed), total_jobs);
        std::this_thread::sleep_for(std::chrono::milliseconds{25});
    }
    lower_pool->CloseAndJoin();
    if (stop.stop_requested()) {
        lower_pool.reset();
        return fail("Stopped");
    }
    report(StageConverting, total_jobs, total_jobs, true);

    auto finished_compute = lower_pool->TakeCompute();
    lower_pool->ApplyGraphics(graphics_entries);
    lower_pool.reset();
    result.skipped = skipped_count.load(std::memory_order_relaxed);

    for (auto& archive : archives) {
        archive.file = nullptr;
    }
    game.romfs.clear();
    game.loader.reset();
    game.nup.reset();

    // Sort & deduplicate compute entries deterministically by ComputeKey (earliest origin wins).
    struct OrderedCompute {
        ComputeKey key{};
        Origin origin{};
        size_t index{};
    };
    std::vector<OrderedCompute> ordered_compute;
    ordered_compute.reserve(finished_compute.size());
    for (size_t i = 0; i < finished_compute.size(); ++i) {
        ordered_compute.push_back(
            {finished_compute[i].key, compute_seen.at(finished_compute[i].id), i});
    }
    std::sort(ordered_compute.begin(), ordered_compute.end(),
              [](const OrderedCompute& a, const OrderedCompute& b) {
                  return a.key != b.key ? a.key < b.key : a.origin < b.origin;
              });

    // Deduplicate graphics strictly by (stage, lowered_code_hash), merging archive_ids and
    // selecting the canonical earliest-origin record so output is compact and 100% deterministic.
    struct DeduplicatedGraphics {
        u32 stage{};
        u64 hash{};
        Origin best_origin{~0U, ~0U};
        size_t winner_index{};
        std::vector<u32> archive_ids;
    };
    std::vector<DeduplicatedGraphics> ordered_graphics;
    if (build_graphics) {
        std::map<std::pair<u32, u64>, size_t> by_stage_hash;
        for (size_t i = 0; i < graphics_entries.size(); ++i) {
            auto& entry = graphics_entries[i];
            if (!entry.lowered_ok) {
                continue;
            }
            const u64 hash = entry.lowered.stage_hash != 0 ? entry.lowered.stage_hash
                                                           : StageHash(entry.lowered.code);
            const auto [it, inserted] =
                by_stage_hash.try_emplace({entry.stage, hash}, ordered_graphics.size());
            if (inserted) {
                ordered_graphics.push_back(
                    {entry.stage, hash, entry.best_origin, i, std::move(entry.archive_ids)});
            } else {
                auto& merged = ordered_graphics[it->second];
                merged.archive_ids.insert(merged.archive_ids.end(), entry.archive_ids.begin(),
                                          entry.archive_ids.end());
                if (entry.best_origin < merged.best_origin) {
                    merged.best_origin = entry.best_origin;
                    merged.winner_index = i;
                }
            }
        }
        for (auto& item : ordered_graphics) {
            std::sort(item.archive_ids.begin(), item.archive_ids.end());
            item.archive_ids.erase(std::unique(item.archive_ids.begin(), item.archive_ids.end()),
                                   item.archive_ids.end());
        }
    }

    // Per-archive sibling vertex attribute statistics (deterministic via sorted archive_ids).
    std::unordered_map<u32, std::set<u64>> by_archive;
    if (build_graphics && use_stats) {
        for (const auto& item : ordered_graphics) {
            if (item.stage == 0 && stats.vertex_attributes.contains(item.hash)) {
                for (const u32 aid : item.archive_ids) {
                    by_archive[aid].insert(item.hash);
                }
            }
        }
    }
    const auto attribute_types = [&](const DeduplicatedGraphics& item) -> u64 {
        if (!use_stats) {
            return DefaultAttributeTypes;
        }
        std::set<u64> others;
        for (const u32 aid : item.archive_ids) {
            if (const auto it = by_archive.find(aid); it != by_archive.end()) {
                others.insert(it->second.begin(), it->second.end());
            }
        }
        others.erase(item.hash);
        std::array<std::array<size_t, 4>, 32> counts{};
        std::array<std::array<size_t, 4>, 32> first{};
        size_t order = 0;
        for (const u64 hash : others) {
            for (const u64 value : stats.vertex_attributes.at(hash)) {
                for (size_t i = 0; i < 32; ++i) {
                    const auto x = static_cast<size_t>((value >> (2 * i)) & 3);
                    if (x != 0) {
                        if (counts[i][x]++ == 0) {
                            first[i][x] = order++;
                        }
                    }
                }
            }
        }
        u64 types = 0;
        for (size_t i = 0; i < 32; ++i) {
            u64 best = 1;
            size_t best_count = 0;
            size_t best_first = 0;
            for (size_t x = 1; x < 4; ++x) {
                if (counts[i][x] > best_count ||
                    (counts[i][x] == best_count && counts[i][x] != 0 && first[i][x] < best_first)) {
                    best = x;
                    best_count = counts[i][x];
                    best_first = first[i][x];
                }
            }
            types |= best << (2 * i);
        }
        return types;
    };

    const size_t total_writes = ordered_compute.size() + ordered_graphics.size();
    report(StageWriting, 0, total_writes, true);

    std::vector<char> compute_buf(1U << 20);
    std::vector<char> graphics_buf(build_graphics ? (1U << 20) : 0);
    compute_out.rdbuf()->pubsetbuf(compute_buf.data(),
                                   static_cast<std::streamsize>(compute_buf.size()));
    compute_out.open(compute_tmp, std::ios::binary | std::ios::trunc);
    if (build_graphics) {
        graphics_out.rdbuf()->pubsetbuf(graphics_buf.data(),
                                        static_cast<std::streamsize>(graphics_buf.size()));
        graphics_out.open(graphics_tmp, std::ios::binary | std::ios::trunc);
    }
    if (!compute_out || (build_graphics && !graphics_out)) {
        return fail("Cannot write to " + Common::FS::PathToUTF8String(output_dir));
    }
    WriteHeader(compute_out);
    if (build_graphics) {
        WriteHeader(graphics_out);
    }

    size_t written_items = 0;
    for (size_t i = 0; i < ordered_compute.size(); ++i) {
        if (stop.stop_requested()) {
            return fail("Stopped");
        }
        if (i == 0 || ordered_compute[i].key != ordered_compute[i - 1].key) {
            const auto& bytes = finished_compute[ordered_compute[i].index].bytes;
            compute_out.write(reinterpret_cast<const char*>(bytes.data()),
                              static_cast<std::streamsize>(bytes.size()));
            ++result.compute_programs;
        }
        if (++written_items % 2048 == 0) {
            report(StageWriting, written_items, total_writes);
        }
    }
    finished_compute.clear();

    if (build_graphics) {
        for (const auto& item : ordered_graphics) {
            if (stop.stop_requested()) {
                return fail("Stopped");
            }
            const auto& winner = graphics_entries[item.winner_index];
            const u64 types = item.stage == 0 ? attribute_types(item) : 0ULL;
            const auto env =
                GraphicsEnvFromParts(item.stage, winner.lowered, winner.sph, winner.best_bank1);
            const auto key = GraphicsEntryKey(
                item.stage, item.stage == 0 ? vertex_template : fragment_template, item.hash,
                types);
            Write(graphics_out, u32{1});
            graphics_out.write(reinterpret_cast<const char*>(env.data()),
                               static_cast<std::streamsize>(env.size()));
            graphics_out.write(reinterpret_cast<const char*>(key.data()),
                               static_cast<std::streamsize>(key.size()));
            if (item.stage == 0) {
                ++result.vertex_programs;
            } else {
                ++result.fragment_programs;
            }
            if (++written_items % 2048 == 0) {
                report(StageWriting, written_items, total_writes);
            }
        }
    }
    graphics_entries.clear();

    compute_out.close();
    if (build_graphics) {
        graphics_out.close();
    }
    if (!compute_out || (build_graphics && !graphics_out)) {
        return fail("Writing the precache files failed");
    }

    // Keep an existing .done marker when the rebuilt file has the exact same content hash; drop
    // it only when the contents changed.
    std::error_code ignored;
    DropStaleWarmMarker(compute_path, compute_tmp);
    fs::remove(WarmCountPath(compute_path), ignored);
    fs::rename(compute_tmp, compute_path, ec);
    if (ec) {
        return fail("Cannot replace " + Common::FS::PathToUTF8String(compute_path));
    }
    WriteWarmCount(compute_path, result.compute_programs);
    if (build_graphics) {
        DropStaleWarmMarker(graphics_path, graphics_tmp);
        fs::remove(WarmCountPath(graphics_path), ignored);
        fs::rename(graphics_tmp, graphics_path, ec);
        if (ec) {
            return fail("Cannot replace " + Common::FS::PathToUTF8String(graphics_path));
        }
        WriteWarmCount(graphics_path, result.vertex_programs + result.fragment_programs);
    }
    report(StageWriting, total_writes, total_writes, true);
    result.ok = true;
    LOG_INFO(Render_Vulkan,
             "Shader precache build for {:016x}: {} compute, {} vertex, {} fragment entries, {} "
             "programs skipped, {} archive and {} member errors, {:.1f} s",
             title_id, result.compute_programs, result.vertex_programs, result.fragment_programs,
             result.skipped, archive_errors.load(), member_errors.load(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return result;
}

} // namespace VideoCommon::ShaderWarm
