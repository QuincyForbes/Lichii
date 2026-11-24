// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_device.h"

namespace Metal {

namespace {
/// Maximum pipelines added to a single MTLBinaryArchive shard before rolling over to a new shard.
/// macOS 26/27's MTLBinaryArchive FlatBuffers serializer is O(N^2) per archive; capping each shard
/// at 128 pipelines (~5-12 MiB) keeps serializeToURL under ~150 ms and load under ~5 ms per shard.
constexpr u32 MaxPipelinesPerShard = 128;
constexpr u32 MaxArchiveShards = 64;
/// Any legacy monolithic archive larger than 64 MiB is discarded so Open() never hangs.
constexpr u64 MaxValidShardFileBytes = 64ULL << 20;
/// Pre-existing shard files >= 4 MiB are treated as sealed on load so new additions start a fresh shard.
constexpr u64 SealedShardFileBytes = 4ULL << 20;

std::filesystem::path ShardPath(const std::filesystem::path& base_path, size_t shard_index) {
    if (shard_index == 0) {
        return base_path;
    }
    std::filesystem::path p = base_path;
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();
    p.replace_filename(fmt::format("{}.s{:02d}{}", stem, shard_index, ext));
    return p;
}
} // Anonymous namespace

struct PipelineArchive::Impl {
    struct Shard {
        id<MTLBinaryArchive> archive = nil;
        NSURL* url = nil;
        NSData* mapped_data = nil;
        std::filesystem::path path;
        u32 entries = 0;
        bool has_data = false;
        bool sealed = false;
        bool dirty = false;
    };

    id<MTLDevice> mtl_device = nil;
    std::filesystem::path base_path;
    std::vector<Shard> shards;
    NSArray<id<MTLBinaryArchive>>* lookup_array = nil;

    std::mutex add_mutex;
    std::condition_variable add_cv;
    std::condition_variable idle_cv;
    std::deque<std::function<bool(void*)>> adds;
    std::thread add_thread;
    bool adding = false;
    bool stop = false;

    void RebuildLookupArrayLocked() {
        NSMutableArray<id<MTLBinaryArchive>>* arr = [NSMutableArray arrayWithCapacity:shards.size()];
        for (const Shard& s : shards) {
            if (s.archive != nil && s.has_data) {
                [arr addObject:s.archive];
            }
        }
        lookup_array = arr.count > 0 ? [arr copy] : nil;
    }

    Shard* EnsureWritableShardLocked() {
        if (mtl_device == nil || base_path.empty()) {
            return nullptr;
        }
        if (!shards.empty()) {
            Shard& last = shards.back();
            if (last.archive != nil && !last.sealed && last.entries < MaxPipelinesPerShard) {
                return &last;
            }
            last.sealed = true;
        }
        if (shards.size() >= MaxArchiveShards) {
            return shards.empty() ? nullptr : &shards.back();
        }
        NSError* error = nil;
        id<MTLBinaryArchive> created =
            [mtl_device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
                                                 error:&error];
        if (created == nil) {
            return nullptr;
        }
        const std::filesystem::path shard_path = ShardPath(base_path, shards.size());
        const std::string utf8 = shard_path.string();
        Shard shard;
        shard.archive = created;
        shard.path = shard_path;
        shard.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:utf8.c_str()]];
        shards.push_back(std::move(shard));
        return &shards.back();
    }
};

PipelineArchive::PipelineArchive() : impl{std::make_unique<Impl>()} {}

PipelineArchive::~PipelineArchive() {
    {
        std::scoped_lock lock{impl->add_mutex};
        impl->stop = true;
    }
    impl->add_cv.notify_all();
    if (impl->add_thread.joinable()) {
        impl->add_thread.join();
    }
    Save();
    is_open.store(false, std::memory_order_relaxed);
    @autoreleasepool {
        impl->adds.clear();
        impl->lookup_array = nil;
        impl->shards.clear();
        impl->mtl_device = nil;
    }
}

void PipelineArchive::DeferAdd(std::function<bool()> add) {
    if (!add) {
        return;
    }
    DeferAdd([fn = std::move(add)](void*) { return fn(); });
}

void PipelineArchive::DeferAdd(std::function<bool(void* archive_handle)> add) {
    if (read_only.load(std::memory_order_relaxed) || !add) {
        return;
    }
    {
        std::scoped_lock lock{impl->add_mutex};
        if (impl->stop) {
            return;
        }
        impl->adds.push_back(std::move(add));
        if (!impl->add_thread.joinable()) {
            impl->add_thread = std::thread([this] {
                pthread_setname_np("MetalArchive");
                pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
                for (;;) {
                    std::function<bool(void*)> next;
                    {
                        std::unique_lock thread_lock{impl->add_mutex};
                        impl->adding = false;
                        if (impl->adds.empty()) {
                            impl->idle_cv.notify_all();
                        }
                        impl->add_cv.wait(thread_lock,
                                          [this] { return impl->stop || !impl->adds.empty(); });
                        if (impl->adds.empty()) {
                            impl->idle_cv.notify_all();
                            return;
                        }
                        next = std::move(impl->adds.front());
                        impl->adds.pop_front();
                        impl->adding = true;
                    }
                    bool added = false;
                    @autoreleasepool {
                        {
                            std::unique_lock adding_lock{build_mutex};
                            Impl::Shard* const shard = impl->EnsureWritableShardLocked();
                            if (shard != nullptr && shard->archive != nil) {
                                added = next((__bridge void*)shard->archive);
                                if (added) {
                                    ++shard->entries;
                                    shard->dirty = true;
                                    if (!shard->has_data) {
                                        shard->has_data = true;
                                        impl->RebuildLookupArrayLocked();
                                    }
                                    if (shard->entries >= MaxPipelinesPerShard) {
                                        shard->sealed = true;
                                    }
                                }
                            }
                        }
                        next = {};
                    }
                    NoteAdded(added);
                }
            });
        }
    }
    impl->add_cv.notify_one();
}

void PipelineArchive::FlushAdds() {
    std::unique_lock lock{impl->add_mutex};
    impl->idle_cv.wait(lock, [this] { return impl->adds.empty() && !impl->adding; });
}

bool PipelineArchive::Open(const Device& device, const std::filesystem::path& path,
                           bool read_only_mode) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ARCHIVE");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled) {
        LOG_INFO(Render_Metal, "Metal: pipeline archive off (EDEN_NX2_METAL_ARCHIVE=0)");
        return false;
    }
    Save();
    read_only.store(read_only_mode, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    std::unique_lock build_lock{build_mutex};
    @autoreleasepool {
        impl->lookup_array = nil;
        impl->shards.clear();
        impl->mtl_device = device.GetDevice();
        impl->base_path = path;

        const auto start = std::chrono::steady_clock::now();
        u64 total_loaded_bytes = 0;
        size_t loaded_shards = 0;
        std::error_code ec;

        for (size_t idx = 0; idx < MaxArchiveShards; ++idx) {
            const std::filesystem::path shard_path = ShardPath(path, idx);
            if (!std::filesystem::exists(shard_path, ec)) {
                if (idx == 0) {
                    continue;
                }
                break;
            }
            const u64 fsize = std::filesystem::file_size(shard_path, ec);
            if (ec || fsize == 0 || fsize > MaxValidShardFileBytes) {
                if (fsize > MaxValidShardFileBytes) {
                    LOG_WARNING(Render_Metal,
                                "Metal: removing oversized legacy archive {} ({} MiB)",
                                shard_path.string(), fsize >> 20);
                }
                std::filesystem::remove(shard_path, ec);
                continue;
            }
            const std::string utf8 = shard_path.string();
            NSURL* const url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:utf8.c_str()]];
            NSError* error = nil;
            NSData* const mapped = [NSData dataWithContentsOfURL:url
                                                         options:NSDataReadingMappedAlways
                                                           error:&error];
            id<MTLBinaryArchive> archive = nil;
            if (mapped != nil && mapped.length > 0) {
                MTLBinaryArchiveDescriptor* descriptor = [MTLBinaryArchiveDescriptor new];
                descriptor.url = url;
                archive = [impl->mtl_device newBinaryArchiveWithDescriptor:descriptor error:&error];
            }
            if (archive == nil) {
                LOG_WARNING(Render_Metal, "Metal: archive shard {} unreadable ({}); removing",
                            utf8,
                            error != nil ? error.localizedDescription.UTF8String : "no reason");
                std::filesystem::remove(shard_path, ec);
                continue;
            }
            Impl::Shard shard;
            shard.archive = archive;
            shard.url = url;
            shard.mapped_data = mapped;
            shard.path = shard_path;
            shard.has_data = true;
            // Treat every pre-loaded shard as sealed so MTLBinaryArchive never overwrites a file
            // it opened with descriptor.url (in-place serializeToURL on an opened shard SIGSEGV'd
            // inside Metal on iOS 26 during shutdown). New pipelines always go into a fresh shard.
            shard.sealed = true;
            impl->shards.push_back(std::move(shard));
            total_loaded_bytes += fsize;
            ++loaded_shards;
        }

        if (impl->shards.empty() && !read_only_mode) {
            if (impl->EnsureWritableShardLocked() == nullptr) {
                is_open.store(false, std::memory_order_relaxed);
                LOG_WARNING(Render_Metal, "Metal: no pipeline archive available on this device");
                return false;
            }
        }

        impl->RebuildLookupArrayLocked();
        is_open.store(true, std::memory_order_relaxed);
        has_entries.store(loaded_shards > 0, std::memory_order_relaxed);
        recent = {};
        total = {};
        dirty = false;
        const auto ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        LOG_INFO(Render_Metal, "Metal: pipeline archive {} {} ({} shards, {} KiB, {:.1f} ms{})",
                 path.string(), loaded_shards > 0 ? "loaded" : "started", loaded_shards,
                 total_loaded_bytes >> 10, ms, read_only_mode ? ", read-only" : "");
    }
    return true;
}

void PipelineArchive::Save() {
    FlushAdds();
    std::scoped_lock lock{mutex};
    if (!dirty) {
        return;
    }
    @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        std::error_code ec;
        if (!impl->base_path.empty()) {
            std::filesystem::create_directories(impl->base_path.parent_path(), ec);
        }
        std::unique_lock build_lock{build_mutex};
        size_t written_shards = 0;
        for (Impl::Shard& shard : impl->shards) {
            if (!shard.dirty || shard.archive == nil || shard.url == nil) {
                continue;
            }
            NSError* error = nil;
            shard.mapped_data = nil;
            // Serialize to a temporary sibling file and atomically rename so Metal never truncates
            // or corrupts an existing archive file while another mapping/handle references it.
            std::filesystem::path tmp_path = shard.path;
            tmp_path += ".tmp";
            const std::string tmp_utf8 = tmp_path.string();
            NSURL* const tmp_url =
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:tmp_utf8.c_str()]];
            if (![shard.archive serializeToURL:tmp_url error:&error]) {
                std::filesystem::remove(tmp_path, ec);
                LOG_WARNING(Render_Metal, "Metal: archive shard {} not written: {}",
                            shard.path.string(),
                            error != nil ? error.localizedDescription.UTF8String : "no reason");
                continue;
            }
            std::error_code rename_ec;
            std::filesystem::rename(tmp_path, shard.path, rename_ec);
            if (rename_ec) {
                std::filesystem::remove(tmp_path, ec);
                LOG_WARNING(Render_Metal, "Metal: archive shard {} rename failed: {}",
                            shard.path.string(), rename_ec.message());
                continue;
            }
            shard.mapped_data = [NSData dataWithContentsOfURL:shard.url
                                                      options:NSDataReadingMappedAlways
                                                        error:nil];
            shard.dirty = false;
            ++written_shards;
        }
        dirty = false;
        if (written_shards > 0) {
            LOG_INFO(Render_Metal,
                     "Metal: pipeline archive written ({} shard(s) updated, {} total added, {:.1f} ms)",
                     written_shards, total.added,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               start)
                         .count());
        }
    }
}

void* PipelineArchive::Handle() const noexcept {
    if (!is_open.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    if (read_only.load(std::memory_order_relaxed) &&
        !has_entries.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    return impl->shards.empty() ? nullptr : (__bridge void*)impl->shards.back().archive;
}

void* PipelineArchive::LookupHandles() const noexcept {
    if (!is_open.load(std::memory_order_relaxed) ||
        !has_entries.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    return (__bridge void*)impl->lookup_array;
}

void PipelineArchive::NoteBuild(bool hit, u64 nanoseconds) noexcept {
    std::scoped_lock lock{mutex};
    for (Counts* const counts : {&recent, &total}) {
        (hit ? counts->hits : counts->misses) += 1;
        (hit ? counts->hit_ns : counts->miss_ns) += nanoseconds;
    }
}

void PipelineArchive::NoteAdded(bool added) noexcept {
    std::scoped_lock lock{mutex};
    for (Counts* const counts : {&recent, &total}) {
        (added ? counts->added : counts->add_failures) += 1;
    }
    if (added) {
        dirty = true;
        has_entries.store(true, std::memory_order_relaxed);
    }
}

PipelineArchive::Counts PipelineArchive::TakeCounts() noexcept {
    std::scoped_lock lock{mutex};
    return std::exchange(recent, Counts{});
}

PipelineArchive::Counts PipelineArchive::Totals() const noexcept {
    const_cast<PipelineArchive*>(this)->FlushAdds();
    std::scoped_lock lock{mutex};
    return total;
}

} // namespace Metal
