// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>
#include <cstdlib>
#include <string>
#include <utility>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_device.h"

namespace Metal {

struct PipelineArchive::Impl {
    id<MTLBinaryArchive> archive = nil;
    NSURL* url = nil;
    NSData* mapped_data = nil;
    std::string path;

    std::mutex add_mutex;
    std::condition_variable add_cv;
    std::condition_variable idle_cv;
    std::deque<std::function<bool()>> adds;
    std::thread add_thread;
    bool adding = false;
    bool stop = false;
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
        impl->archive = nil;
        impl->mapped_data = nil;
        impl->url = nil;
    }
}

void PipelineArchive::DeferAdd(std::function<bool()> add) {
    if (read_only.load(std::memory_order_relaxed)) {
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
                    std::function<bool()> next;
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
                            added = next();
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
    @autoreleasepool {
        impl->path = path.string();
        impl->url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:impl->path.c_str()]];
        std::error_code ec;
        const bool exists = std::filesystem::exists(path, ec);
        const auto start = std::chrono::steady_clock::now();
        NSError* error = nil;
        id<MTLBinaryArchive> archive = nil;
        impl->mapped_data = nil;
        if (exists) {
            NSData* const mapped = [NSData dataWithContentsOfURL:impl->url
                                                         options:NSDataReadingMappedAlways
                                                           error:&error];
            if (mapped != nil && mapped.length > 0) {
                impl->mapped_data = mapped;
                MTLBinaryArchiveDescriptor* descriptor = [MTLBinaryArchiveDescriptor new];
                descriptor.url = impl->url;
                archive =
                    [device.GetDevice() newBinaryArchiveWithDescriptor:descriptor error:&error];
            }
            if (archive == nil) {
                impl->mapped_data = nil;
                // Another OS or GPU, or a damaged file: start again.
                LOG_WARNING(Render_Metal, "Metal: pipeline archive {} not loaded ({}); a new one "
                                          "is made",
                            impl->path,
                            error != nil ? error.localizedDescription.UTF8String : "no reason");
            }
        }
        const bool loaded = archive != nil;
        if (archive == nil) {
            archive = [device.GetDevice() newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
                                                                   error:&error];
        }
        if (archive == nil) {
            is_open.store(false, std::memory_order_relaxed);
            LOG_WARNING(Render_Metal, "Metal: no pipeline archive: {}",
                        error != nil ? error.localizedDescription.UTF8String : "no reason");
            return false;
        }
        impl->archive = archive;
        is_open.store(true, std::memory_order_relaxed);
        has_entries.store(loaded, std::memory_order_relaxed);
        recent = {};
        total = {};
        dirty = false;
        const auto ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        std::uintmax_t size = loaded ? std::filesystem::file_size(path, ec) : 0;
        if (ec) {
            size = 0;
        }
        LOG_INFO(Render_Metal, "Metal: pipeline archive {} {} ({} KiB, {:.1f} ms{})", impl->path,
                 loaded ? "loaded" : "started", size >> 10, ms,
                 read_only_mode ? ", read-only" : "");
    }
    return true;
}

void PipelineArchive::Save() {
    FlushAdds();
    std::scoped_lock lock{mutex};
    if (impl->archive == nil || !dirty) {
        return;
    }
    @autoreleasepool {
        const auto start = std::chrono::steady_clock::now();
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path{impl->path}.parent_path(), ec);
        NSError* error = nil;
        std::unique_lock build_lock{build_mutex};
        impl->mapped_data = nil;
        if (![impl->archive serializeToURL:impl->url error:&error]) {
            LOG_WARNING(Render_Metal, "Metal: pipeline archive {} not written: {}", impl->path,
                        error != nil ? error.localizedDescription.UTF8String : "no reason");
            return;
        }
        impl->mapped_data = [NSData dataWithContentsOfURL:impl->url
                                                  options:NSDataReadingMappedAlways
                                                    error:nil];
        dirty = false;
        LOG_INFO(Render_Metal, "Metal: pipeline archive {} written with {} new pipelines ({:.1f} "
                               "ms)",
                 impl->path, total.added,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                           start)
                     .count());
    }
}

void* PipelineArchive::Handle() const noexcept {
    if (read_only.load(std::memory_order_relaxed) &&
        !has_entries.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    return (__bridge void*)impl->archive;
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
