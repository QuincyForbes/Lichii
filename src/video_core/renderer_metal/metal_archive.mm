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
    std::string path;

    std::mutex add_mutex;
    std::condition_variable add_cv;
    std::deque<std::function<bool()>> adds;
    std::thread add_thread;
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
}

void PipelineArchive::DeferAdd(std::function<bool()> add) {
    {
        std::scoped_lock lock{impl->add_mutex};
        if (impl->stop) {
            return;
        }
        impl->adds.push_back(std::move(add));
        if (!impl->add_thread.joinable()) {
            impl->add_thread = std::thread([this] {
                pthread_setname_np("MetalArchive");
                pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
                for (;;) {
                    std::function<bool()> next;
                    {
                        std::unique_lock lock{impl->add_mutex};
                        impl->add_cv.wait(lock, [this] { return impl->stop || !impl->adds.empty(); });
                        if (impl->adds.empty()) {
                            return;
                        }
                        next = std::move(impl->adds.front());
                        impl->adds.pop_front();
                    }
                    bool added = false;
                    @autoreleasepool {
                        std::unique_lock adding{build_mutex};
                        added = next();
                    }
                    NoteAdded(added);
                }
            });
        }
    }
    impl->add_cv.notify_one();
}

bool PipelineArchive::Open(const Device& device, const std::filesystem::path& path) {
    static const bool enabled = [] {
        // Off unless EDEN_NX2_METAL_ARCHIVE=1: adding a pipeline compiles it a second time under
        // the archive's exclusive lock, which every builder's lookup waits on. In an OoT lab run
        // with a cold cache that kept pipelines 20 s (up to 121 s) from ready at 0.2 fps, against
        // gameplay within seconds without the archive; the archive saved ~5 s a boot, and macOS
        // already caches compiled functions for the process.
        const char* const value = std::getenv("EDEN_NX2_METAL_ARCHIVE");
        return value != nullptr && value[0] == '1';
    }();
    if (!enabled) {
        LOG_INFO(Render_Metal, "Metal: pipeline archive off (EDEN_NX2_METAL_ARCHIVE=1 turns it on)");
        return false;
    }
    Save();
    std::scoped_lock lock{mutex};
    @autoreleasepool {
        impl->path = path.string();
        impl->url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:impl->path.c_str()]];
        std::error_code ec;
        const bool exists = std::filesystem::exists(path, ec);
        const auto start = std::chrono::steady_clock::now();
        NSError* error = nil;
        id<MTLBinaryArchive> archive = nil;
        if (exists) {
            MTLBinaryArchiveDescriptor* descriptor = [MTLBinaryArchiveDescriptor new];
            descriptor.url = impl->url;
            archive = [device.GetDevice() newBinaryArchiveWithDescriptor:descriptor error:&error];
            if (archive == nil) {
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
            LOG_WARNING(Render_Metal, "Metal: no pipeline archive: {}",
                        error != nil ? error.localizedDescription.UTF8String : "no reason");
            return false;
        }
        impl->archive = archive;
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
        LOG_INFO(Render_Metal, "Metal: pipeline archive {} {} ({} KiB, {:.1f} ms)", impl->path,
                 loaded ? "loaded" : "started", size >> 10, ms);
    }
    return true;
}

bool PipelineArchive::IsOpen() const noexcept {
    return impl->archive != nil;
}

void PipelineArchive::Save() {
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
        if (![impl->archive serializeToURL:impl->url error:&error]) {
            LOG_WARNING(Render_Metal, "Metal: pipeline archive {} not written: {}", impl->path,
                        error != nil ? error.localizedDescription.UTF8String : "no reason");
            return;
        }
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
    dirty = dirty || added;
}

PipelineArchive::Counts PipelineArchive::TakeCounts() noexcept {
    std::scoped_lock lock{mutex};
    return std::exchange(recent, Counts{});
}

PipelineArchive::Counts PipelineArchive::Totals() const noexcept {
    std::scoped_lock lock{mutex};
    return total;
}

} // namespace Metal
