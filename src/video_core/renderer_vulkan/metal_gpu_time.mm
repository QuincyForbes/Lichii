// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <string>

#include <mach/mach_time.h>
#import <Metal/Metal.h>
#include <objc/runtime.h>

#include "common/logging.h"
#include "common/perf_report.h"
#include "video_core/renderer_vulkan/metal_gpu_time.h"

namespace Vulkan::MetalGpuTime {

namespace {

std::mutex g_mutex;
CFTimeInterval g_last_end{};

void Record(id<MTLCommandBuffer> buffer) {
    const CFTimeInterval start = buffer.GPUStartTime;
    const CFTimeInterval end = buffer.GPUEndTime;
    if (start <= 0.0 || end <= start || end - start > 10.0) {
        return;
    }
    // Command buffers overlap on the GPU (and come from several queues): count each moment once.
    CFTimeInterval busy;
    {
        std::scoped_lock lock{g_mutex};
        const CFTimeInterval from = g_last_end > start ? (std::min)(g_last_end, end) : start;
        busy = end - from;
        g_last_end = (std::max)(g_last_end, end);
    }
    Common::PerfReport::Add(Common::PerfReport::GpuBusyNs, static_cast<u64>(busy * 1e9));
    Common::PerfReport::Add(Common::PerfReport::GpuTimedBuffers);
}

constexpr size_t MaxClasses = 3;
std::array<IMP, MaxClasses> g_commits{};

template <size_t Index>
void Commit(id self, SEL command) {
    [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        Record(buffer);
    }];
    reinterpret_cast<void (*)(id, SEL)>(g_commits[Index])(self, command);
}

const std::array<IMP, MaxClasses> Trampolines{
    reinterpret_cast<IMP>(&Commit<0>),
    reinterpret_cast<IMP>(&Commit<1>),
    reinterpret_cast<IMP>(&Commit<2>),
};

void Install() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            LOG_WARNING(Render_Vulkan, "Perf: no Metal device for GPU times");
            return;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        // The classes behind MoltenVK's command buffers (the driver's own, private ones).
        std::array<Class, MaxClasses> classes{};
        size_t count = 0;
        const auto add = [&](id<MTLCommandBuffer> buffer) {
            if (buffer == nil) {
                return;
            }
            Class cls = object_getClass(buffer);
            if (cls == nil || std::find(classes.begin(), classes.begin() + count, cls) !=
                                  classes.begin() + count) {
                return;
            }
            if (count < MaxClasses) {
                classes[count++] = cls;
            }
        };
        add([queue commandBuffer]);
        add([queue commandBufferWithUnretainedReferences]);
        if (@available(iOS 14.0, macOS 11.0, *)) {
            MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
            add([queue commandBufferWithDescriptor:descriptor]);
        }
        // Read every original first: a class may inherit `commit` from another one in the list.
        const SEL selector = @selector(commit);
        std::array<Method, MaxClasses> methods{};
        for (size_t i = 0; i < count; ++i) {
            methods[i] = class_getInstanceMethod(classes[i], selector);
            g_commits[i] = methods[i] != nullptr ? method_getImplementation(methods[i]) : nullptr;
        }
        std::string names;
        for (size_t i = 0; i < count; ++i) {
            if (g_commits[i] == nullptr) {
                continue;
            }
            class_replaceMethod(classes[i], selector, Trampolines[i],
                                method_getTypeEncoding(methods[i]));
            names += names.empty() ? "" : ", ";
            names += class_getName(classes[i]);
        }
        LOG_INFO(Common_Memory, "Perf: GPU time from Metal command buffers ({})",
                 names.empty() ? "none found" : names);
    }
}

} // Anonymous namespace

void Start() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_GPU_TIME");
        if (value != nullptr && value[0] == '0') {
            return;
        }
        Install();
    });
}

double TimestampNsPerTick() {
    static std::mutex mutex;
    static id<MTLDevice> device = nil;
    static MTLTimestamp first_cpu{};
    static MTLTimestamp first_gpu{};
    std::scoped_lock lock{mutex};
    if (device == nil) {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return 0.0;
        }
        [device sampleTimestamps:&first_cpu gpuTimestamp:&first_gpu];
        return 0.0;
    }
    MTLTimestamp cpu{};
    MTLTimestamp gpu{};
    [device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    mach_timebase_info_data_t timebase{};
    mach_timebase_info(&timebase);
    // The CPU value is in mach_absolute_time units.
    const double cpu_ns = static_cast<double>(cpu - first_cpu) * timebase.numer / timebase.denom;
    if (cpu_ns < 50e6 || gpu <= first_gpu) {
        return 0.0;
    }
    return cpu_ns / static_cast<double>(gpu - first_gpu);
}

} // namespace Vulkan::MetalGpuTime
