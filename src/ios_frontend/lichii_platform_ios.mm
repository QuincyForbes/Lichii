// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Portions adapted from MeloNX (https://git.ryujinx.app/projects/MeloNX @ 6a1c159), Copyright Stossy11 and MeloNX contributors (MIT/GPLv3)

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <mach/mach.h>
#include <mutex>
#include <os/proc.h>
#include <string>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/platform_events.h"
#include "ios_frontend/lichii_bridge_platform.h"
#include "ios_frontend/lichii_platform_internal.h"
#include "yuzu_cmd/ios_support.h"

namespace {

std::once_flag g_init_once;
std::atomic<bool> g_initialized{false};
LichiiDeviceInfo g_info{}; ///< Static facts, gathered once at init

std::mutex g_handler_mutex;
LichiiPlatformEventFn g_handler = nullptr;
void* g_handler_user = nullptr;

std::atomic<bool> g_audio_mix{false};
std::atomic<double> g_audio_buffer_ms{20.0};

void CopyString(char* out, std::size_t size, const char* text) {
    std::snprintf(out, size, "%s", text != nullptr ? text : "");
}

void Emit(LichiiPlatformEvent event) {
    LichiiPlatformEventFn handler;
    void* user;
    {
        std::scoped_lock lock{g_handler_mutex};
        handler = g_handler;
        user = g_handler_user;
    }
    if (handler != nullptr) {
        handler(static_cast<int>(event), user);
    }
}

// SecTask has no public header on iOS; its functions are in Security.framework.
int EntitlementValue(const char* name) {
    static void* const security =
        dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_LAZY);
    if (security == nullptr) {
        return -1;
    }
    using CreateFromSelf = CFTypeRef (*)(CFAllocatorRef);
    using CopyValue = CFTypeRef (*)(CFTypeRef, CFStringRef, CFErrorRef*);
    const auto create_from_self =
        reinterpret_cast<CreateFromSelf>(dlsym(security, "SecTaskCreateFromSelf"));
    const auto copy_value =
        reinterpret_cast<CopyValue>(dlsym(security, "SecTaskCopyValueForEntitlement"));
    if (create_from_self == nullptr || copy_value == nullptr) {
        return -1;
    }
    const CFTypeRef task = create_from_self(kCFAllocatorDefault);
    if (task == nullptr) {
        return -1;
    }
    const CFStringRef key = CFStringCreateWithCString(kCFAllocatorDefault, name,
                                                      kCFStringEncodingUTF8);
    const CFTypeRef value = copy_value(task, key, nullptr);
    CFRelease(key);
    CFRelease(task);
    if (value == nullptr) {
        return 0;
    }
    int result = 0;
    if (CFGetTypeID(value) == CFBooleanGetTypeID()) {
        result = CFBooleanGetValue(static_cast<CFBooleanRef>(value)) ? 1 : 0;
    } else if (CFGetTypeID(value) == CFNumberGetTypeID()) {
        int number = 0;
        CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType, &number);
        result = number != 0 ? 1 : 0;
    }
    CFRelease(value);
    return result;
}

bool CanReserve(std::size_t bytes) {
    void* const address = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (address == MAP_FAILED) {
        return false;
    }
    munmap(address, bytes);
    return true;
}

// Largest single reservation in GiB (the guest's fastmem arena needs one).
uint64_t LargestReservationGib() {
    uint64_t low = 0;
    uint64_t high = 1024;
    while (high - low > 1) {
        const uint64_t mid = (low + high) / 2;
        (CanReserve(std::size_t{mid} << 30) ? low : high) = mid;
    }
    return low;
}

uint64_t FootprintBytes() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return info.phys_footprint;
}

int ThermalState() {
    switch ([[NSProcessInfo processInfo] thermalState]) {
    case NSProcessInfoThermalStateNominal:
        return 0;
    case NSProcessInfoThermalStateFair:
        return 1;
    case NSProcessInfoThermalStateSerious:
        return 2;
    case NSProcessInfoThermalStateCritical:
        return 3;
    }
    return 0;
}

void GatherDeviceInfo() {
    LichiiDeviceInfo& info = g_info;
    utsname name{};
    uname(&name);
    CopyString(info.machine, sizeof(info.machine), name.machine);
    info.is_ipad = std::strncmp(name.machine, "iPad", 4) == 0 ? 1 : 0;

    const NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
    info.os_major = static_cast<int>(os.majorVersion);
    info.os_minor = static_cast<int>(os.minorVersion);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil) {
        CopyString(info.gpu_name, sizeof(info.gpu_name), device.name.UTF8String);
        // MTLGPUFamilyApple7 = 1007 ... Apple10 = 1010 (by value: older SDKs lack the newest).
        for (int family = 1010; family >= 1001; --family) {
            if ([device supportsFamily:static_cast<MTLGPUFamily>(family)]) {
                info.gpu_family = family - 1000;
                break;
            }
        }
        info.argument_buffers_tier = static_cast<int>(device.argumentBuffersSupport) + 1;
    }
    // "Apple A17 Pro GPU" -> "A17", "Apple M2 GPU" -> "M2".
    if (const char* chip = std::strstr(info.gpu_name, "Apple ");
        chip != nullptr && (chip[6] == 'A' || chip[6] == 'M')) {
        std::size_t length = 1;
        while (chip[6 + length] >= '0' && chip[6 + length] <= '9' && length < 4) {
            ++length;
        }
        std::snprintf(info.chip, sizeof(info.chip), "%.*s", static_cast<int>(length), chip + 6);
        info.is_m_series = chip[6] == 'M' ? 1 : 0;
    }

    uint64_t physical = 0;
    std::size_t physical_size = sizeof(physical);
    sysctlbyname("hw.memsize", &physical, &physical_size, nullptr, 0);
    info.physical_memory_mib = physical >> 20;
    info.launch_available_memory_mib = static_cast<uint64_t>(os_proc_available_memory()) >> 20;

    info.increased_memory_limit =
        EntitlementValue("com.apple.developer.kernel.increased-memory-limit");
    info.extended_virtual_addressing =
        EntitlementValue("com.apple.developer.kernel.extended-virtual-addressing");
    info.va_reserve_4g = CanReserve(std::size_t{4} << 30) ? 1 : 0;
    info.va_reserve_16g = CanReserve(std::size_t{16} << 30) ? 1 : 0;
    info.va_reserve_max_gib = LargestReservationGib();
    info.txm_from_device_tree = IosSupport::HasTxmIoKit() ? 1 : 0;
    info.txm = IosSupport::HasTxm() ? 1 : 0;
    info.max_fps = static_cast<int>(UIScreen.mainScreen.maximumFramesPerSecond);
}

void ConfigureMoltenVk() {
    // Argument buffers only where Metal has tier 2 support (A13+ / every Apple GPU we target),
    // as MeloNX does; MoltenVK's own check accepts tier 1 too.
    if (g_info.argument_buffers_tier > 0 && g_info.argument_buffers_tier < 2) {
        setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0", 0);
    }
}

void ExcludeCachesFromBackup() {
    const auto cache = Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir);
    std::error_code error;
    std::filesystem::create_directories(cache, error);
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cache.c_str()]
                            isDirectory:YES];
    NSError* ns_error = nil;
    if (![url setResourceValue:@YES forKey:NSURLIsExcludedFromBackupKey error:&ns_error]) {
        LOG_WARNING(Frontend, "iOS: could not exclude {} from backups: {}", cache.string(),
                    ns_error.localizedDescription.UTF8String);
    }
}

void ApplyAudioSession() {
    AVAudioSession* session = [AVAudioSession sharedInstance];
    NSError* error = nil;
    const AVAudioSessionCategoryOptions options =
        g_audio_mix.load() ? AVAudioSessionCategoryOptionMixWithOthers
                           : static_cast<AVAudioSessionCategoryOptions>(0);
    [session setCategory:AVAudioSessionCategoryPlayback
                    mode:AVAudioSessionModeDefault
                 options:options
                   error:&error];
    [session setPreferredSampleRate:48000 error:&error];
    if (const double ms = g_audio_buffer_ms.load(); ms > 0) {
        [session setPreferredIOBufferDuration:ms / 1000.0 error:&error];
    }
    [session setActive:YES error:&error];
    if (error != nil) {
        LOG_WARNING(Frontend, "iOS: audio session: {}", error.localizedDescription.UTF8String);
    }
}

void WriteCachesNow() {
    Common::PlatformEvents::NotifyEnteringBackground();
}

// iOS sends memory warnings in bursts (10 in 2 s on an iPhone18,2 while a game loaded),
// each one seen by the UIKit observer and by the dispatch source. The first of a burst starts the
// collections; the rest found nothing more to free and are only counted.
constexpr auto MemoryWarningBurst = std::chrono::seconds{2};
std::mutex g_warning_mutex;
std::chrono::steady_clock::time_point g_warning_burst_end{};
bool g_warning_burst_emitted = false;
unsigned g_warnings_folded = 0;

void OnMemoryWarning(const char* kind, bool from_uikit) {
    const auto now = std::chrono::steady_clock::now();
    bool first = false;
    bool emit = false;
    {
        std::scoped_lock lock{g_warning_mutex};
        if (now >= g_warning_burst_end) {
            first = true;
            g_warning_burst_end = now + MemoryWarningBurst;
            g_warning_burst_emitted = false;
            g_warnings_folded = 0;
        } else {
            ++g_warnings_folded;
        }
        // The app's own warning flag (the unclean-exit notice) still comes from UIKit only.
        if (from_uikit && !g_warning_burst_emitted) {
            g_warning_burst_emitted = true;
            emit = true;
        }
    }
    if (first) {
        LOG_WARNING(Frontend, "iOS: {} ({} MiB left, footprint {} MiB)", kind,
                    lichii_platform_available_memory_mib(), lichii_platform_footprint_mib());
        Common::PlatformEvents::SignalMemoryPressure();
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW,
                                     std::chrono::nanoseconds{MemoryWarningBurst}.count()),
                       dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                         unsigned folded;
                         {
                             std::scoped_lock lock{g_warning_mutex};
                             folded = g_warnings_folded;
                         }
                         if (folded != 0) {
                             LOG_INFO(Frontend,
                                      "iOS: {} more memory warnings in the same {} s folded into "
                                      "the one above ({} MiB left, footprint {} MiB)",
                                      folded, MemoryWarningBurst.count(),
                                      lichii_platform_available_memory_mib(),
                                      lichii_platform_footprint_mib());
                         }
                       });
    }
    if (emit) {
        Emit(LICHII_PLATFORM_EVENT_MEMORY_WARNING);
    }
}

void InstallObservers() {
    NSNotificationCenter* center = [NSNotificationCenter defaultCenter];
    NSOperationQueue* main = [NSOperationQueue mainQueue];
    const auto observe = [&](NSNotificationName name, void (^block)(NSNotification*)) {
        [center addObserverForName:name object:nil queue:main usingBlock:block];
    };
    observe(UIApplicationWillResignActiveNotification,
            ^(NSNotification* note) { Emit(LICHII_PLATFORM_EVENT_WILL_RESIGN_ACTIVE); });
    observe(UIApplicationDidEnterBackgroundNotification, ^(NSNotification* note) {
      LOG_INFO(Frontend, "iOS: in the background, writing caches");
      lichii_platform_flush_caches();
      Emit(LICHII_PLATFORM_EVENT_DID_ENTER_BACKGROUND);
    });
    observe(UIApplicationWillEnterForegroundNotification,
            ^(NSNotification* note) { Emit(LICHII_PLATFORM_EVENT_WILL_ENTER_FOREGROUND); });
    observe(UIApplicationDidBecomeActiveNotification,
            ^(NSNotification* note) { Emit(LICHII_PLATFORM_EVENT_DID_BECOME_ACTIVE); });
    observe(UIApplicationDidReceiveMemoryWarningNotification,
            ^(NSNotification* note) { OnMemoryWarning("memory warning", true); });
    observe(UIApplicationWillTerminateNotification, ^(NSNotification* note) {
      WriteCachesNow();
      Emit(LICHII_PLATFORM_EVENT_WILL_TERMINATE);
    });
    observe(NSProcessInfoThermalStateDidChangeNotification, ^(NSNotification* note) {
      LichiiThermalAdvice advice{};
      lichii_platform_thermal_advice(&advice);
      LOG_INFO(Frontend, "iOS: thermal state {}{}{}", advice.thermal_state,
               advice.message[0] != '\0' ? ": " : "", advice.message);
      Emit(LICHII_PLATFORM_EVENT_THERMAL_STATE);
    });
    observe(NSProcessInfoPowerStateDidChangeNotification, ^(NSNotification* note) {
      LOG_INFO(Frontend, "iOS: Low Power Mode {}", lichii_platform_low_power_mode() != 0);
      Emit(LICHII_PLATFORM_EVENT_LOW_POWER_MODE);
    });
    observe(AVAudioSessionInterruptionNotification, ^(NSNotification* note) {
      const auto type = static_cast<AVAudioSessionInterruptionType>(
          [note.userInfo[AVAudioSessionInterruptionTypeKey] unsignedIntegerValue]);
      if (type == AVAudioSessionInterruptionTypeBegan) {
          LOG_INFO(Frontend, "iOS: audio interrupted");
          Emit(LICHII_PLATFORM_EVENT_AUDIO_INTERRUPTION_BEGAN);
      } else {
          LOG_INFO(Frontend, "iOS: audio interruption ended");
          NSError* error = nil;
          [[AVAudioSession sharedInstance] setActive:YES error:&error];
          Emit(LICHII_PLATFORM_EVENT_AUDIO_INTERRUPTION_ENDED);
      }
    });
    observe(AVAudioSessionMediaServicesWereResetNotification, ^(NSNotification* note) {
      LOG_WARNING(Frontend, "iOS: media services were reset, configuring the audio session again");
      ApplyAudioSession();
      Emit(LICHII_PLATFORM_EVENT_AUDIO_SERVICES_RESET);
    });
    // The UIKit warning above arrives on the main queue, after whatever the main thread is busy
    // with; this one comes on its own queue as soon as the system's memory pressure rises.
    static dispatch_source_t pressure_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
    if (pressure_source != nullptr) {
        dispatch_source_set_event_handler(pressure_source, ^{
          const unsigned long level = dispatch_source_get_data(pressure_source);
          OnMemoryWarning((level & DISPATCH_MEMORYPRESSURE_CRITICAL) != 0
                              ? "system memory pressure critical"
                              : "system memory pressure warning",
                          false);
        });
        dispatch_resume(pressure_source);
    }
}

} // Anonymous namespace

extern "C" {

void lichii_platform_init(void) {
    std::call_once(g_init_once, [] {
        @autoreleasepool {
            GatherDeviceInfo();
            Settings::DeviceProfile::SetHostDetails(LichiiPlatform::DeviceOf(g_info));
            LOG_INFO(Frontend, "iOS platform: device profile switches: {}",
                     Settings::DeviceProfile::ApplyTuning(Settings::DeviceProfile::ForHost()));
            g_initialized.store(true);
            const LichiiDeviceDefaults defaults = LichiiPlatform::ComputeDefaults(g_info);
            LichiiPlatform::ApplyMemoryBudgets(defaults);
            LichiiPlatform::ApplyExperiments();
            ConfigureMoltenVk();
            Common::PlatformEvents::SetDisplayMaxFps(static_cast<u32>(g_info.max_fps));
            ExcludeCachesFromBackup();
            InstallObservers();
            LOG_INFO(Frontend,
                     "iOS platform: {} ({}, GPU family Apple{}, argument buffers tier {}), iOS "
                     "{}.{}, {} MiB RAM, {} MiB available at launch, {} Hz",
                     g_info.machine, g_info.gpu_name, g_info.gpu_family,
                     g_info.argument_buffers_tier, g_info.os_major, g_info.os_minor,
                     g_info.physical_memory_mib, g_info.launch_available_memory_mib,
                     g_info.max_fps);
            LOG_INFO(Frontend,
                     "iOS platform: increased memory limit {}, extended virtual addressing {}, "
                     "reserve 4 GiB {} / 16 GiB {} / largest {} GiB, TXM {} ({})",
                     g_info.increased_memory_limit, g_info.extended_virtual_addressing,
                     g_info.va_reserve_4g, g_info.va_reserve_16g, g_info.va_reserve_max_gib,
                     g_info.txm,
                     g_info.txm_from_device_tree ? "device tree" : "model table");
            LOG_INFO(Frontend, "iOS platform: increased debugging memory limit {}",
                     EntitlementValue("com.apple.developer.kernel.increased-debugging-memory-limit"));
            if (g_info.increased_memory_limit == 0) {
                LOG_WARNING(Frontend, "iOS: the increased-memory-limit entitlement is missing; "
                                      "games will likely be closed by iOS");
            }
        }
    });
}

int lichii_platform_device_info(LichiiDeviceInfo* out) {
    if (out == nullptr || !g_initialized.load()) {
        return 0;
    }
    *out = g_info;
    out->available_memory_mib = lichii_platform_available_memory_mib();
    out->footprint_mib = lichii_platform_footprint_mib();
    out->low_power_mode = lichii_platform_low_power_mode();
    out->thermal_state = lichii_platform_thermal_state();
    return 1;
}

void lichii_platform_set_event_handler(LichiiPlatformEventFn handler, void* user) {
    std::scoped_lock lock{g_handler_mutex};
    g_handler = handler;
    g_handler_user = user;
}

void lichii_platform_flush_caches(void) {
    const std::uint64_t task = IosSupport::BeginBackgroundTask("lichii-caches");
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
      WriteCachesNow();
      IosSupport::EndBackgroundTask(task);
    });
}

void lichii_platform_set_playing(int playing) {
    @autoreleasepool {
        [UIApplication sharedApplication].idleTimerDisabled = playing != 0 ? YES : NO;
        if (playing != 0) {
            NSError* error = nil;
            [[AVAudioSession sharedInstance] setActive:YES error:&error];
        }
    }
}

void lichii_platform_configure_audio(int mix_with_others, double io_buffer_ms) {
    g_audio_mix.store(mix_with_others != 0);
    if (io_buffer_ms > 0) {
        g_audio_buffer_ms.store(std::clamp(io_buffer_ms, 2.0, 100.0));
    }
    @autoreleasepool {
        ApplyAudioSession();
    }
}

int lichii_platform_thermal_state(void) {
    return ThermalState();
}

int lichii_platform_low_power_mode(void) {
    return [[NSProcessInfo processInfo] isLowPowerModeEnabled] ? 1 : 0;
}

uint64_t lichii_platform_available_memory_mib(void) {
    return static_cast<uint64_t>(os_proc_available_memory()) >> 20;
}

uint64_t lichii_platform_footprint_mib(void) {
    return FootprintBytes() >> 20;
}

int lichii_platform_max_fps(void) {
    return g_info.max_fps;
}

} // extern "C"
