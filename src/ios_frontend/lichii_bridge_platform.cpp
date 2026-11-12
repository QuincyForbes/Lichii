// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <sys/mount.h>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/host_memory.h"
#include "common/logging.h"
#include "common/platform_events.h"
#include "common/settings.h"
#include "ios_frontend/lichii_bridge_platform.h"
#include "ios_frontend/lichii_platform_internal.h"

namespace LichiiPlatform {
namespace {

/// Bump when the table changes: devices then get the new defaults once more.
constexpr int DefaultsVersion = 1;

struct TierRow {
    LichiiDeviceTier tier;
    const char* name;
    u64 min_memory_mib; ///< hw.memsize reports a little under the marketed size
    int min_gpu_family;
    Settings::ResolutionSetup resolution;
    float scale;
    const char* note;
};

// First match wins. Switch 2 titles get a 12 GB memory layout and iOS gives an app about 60-70%
// of the device's memory with the increased-memory-limit entitlement, so memory decides first.
constexpr TierRow Tiers[] = {
    {LICHII_TIER_ULTRA, "ultra", 14000, 8, Settings::ResolutionSetup::Res3_2X, 1.5f, ""},
    {LICHII_TIER_HIGH, "high", 10000, 9, Settings::ResolutionSetup::Res1X, 1.0f, ""},
    {LICHII_TIER_MID, "mid", 7000, 8, Settings::ResolutionSetup::Res1X, 1.0f, ""},
    {LICHII_TIER_LOW, "low", 5000, 8, Settings::ResolutionSetup::Res3_4X, 0.75f,
     "6 GB device: larger games may run out of memory."},
};

template <typename T>
T Clamp(T value, T low, T high) {
    return std::min(std::max(value, low), high);
}

void CopyString(char* out, std::size_t size, std::string_view text) {
    const std::size_t length = std::min(size - 1, text.size());
    std::memcpy(out, text.data(), length);
    out[length] = '\0';
}

std::filesystem::path ConfigPath(std::string_view name) {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / name;
}

bool ValidExperimentName(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    for (const char* c = name; *c != '\0'; ++c) {
        if (!((*c >= 'a' && *c <= 'z') || (*c >= '0' && *c <= '9') || *c == '_')) {
            return false;
        }
    }
    return true;
}

std::atomic<int> g_shader_stage{LICHII_SHADER_IDLE};
std::atomic<u64> g_shader_done{};
std::atomic<u64> g_shader_total{};
std::atomic<s64> g_shader_last_post_ms{};
std::mutex g_shader_handler_mutex;
LichiiShaderProgressFn g_shader_handler = nullptr;
void* g_shader_handler_user = nullptr;

std::filesystem::path ShaderCachePath(u64 title_id) {
    const auto dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    return title_id == 0 ? dir : dir / fmt::format("{:016x}", title_id);
}

/// The .nup loader keeps the files it decompressed in host memory (512 MiB by default, which a
/// phone doesn't have to spare) and writes those over half the cap to a temporary file instead.
constexpr u32 NupCacheMib = 96;

/// The loader removes its temporary files when the game closes; a game iOS closed leaves them.
void RemoveLeftoverNupFiles() {
    const auto dir = Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "nup";
    std::error_code error;
    std::vector<std::filesystem::path> leftovers;
    for (std::filesystem::directory_iterator it{dir, error}, end; !error && it != end;
         it.increment(error)) {
        if (it->path().extension() == ".tmp") {
            leftovers.push_back(it->path());
        }
    }
    u64 bytes = 0;
    for (const auto& path : leftovers) {
        const auto size = std::filesystem::file_size(path, error);
        if (!error && std::filesystem::remove(path, error)) {
            bytes += size;
        }
    }
    if (!leftovers.empty()) {
        LOG_INFO(Frontend, "iOS: removed {} temporary .nup files of an earlier run ({} MiB)",
                 leftovers.size(), bytes >> 20);
    }
}

} // Anonymous namespace

LichiiDeviceDefaults ComputeDefaults(const LichiiDeviceInfo& info) {
    LichiiDeviceDefaults out{};
    const TierRow* row = nullptr;
    for (const TierRow& candidate : Tiers) {
        if (info.physical_memory_mib >= candidate.min_memory_mib &&
            info.gpu_family >= candidate.min_gpu_family) {
            row = &candidate;
            break;
        }
    }
    if (row != nullptr) {
        out.tier = row->tier;
        CopyString(out.tier_name, sizeof(out.tier_name), row->name);
        out.resolution_setup = static_cast<int>(row->resolution);
        out.resolution_scale = row->scale;
        CopyString(out.note, sizeof(out.note), row->note);
    } else {
        out.tier = LICHII_TIER_UNSUPPORTED;
        CopyString(out.tier_name, sizeof(out.tier_name), "unsupported");
        out.resolution_setup = static_cast<int>(Settings::ResolutionSetup::Res1_2X);
        out.resolution_scale = 0.5f;
        CopyString(out.note, sizeof(out.note),
                   "Under 6 GB of memory or older than A15: Switch 2 games will likely be "
                   "closed by iOS.");
    }
    out.frame_gen_capable = info.gpu_family >= 9 || info.is_m_series ? 1 : 0;
    // Only where the GPU has room to spare and the display can show the extra frames.
    out.frame_gen = out.tier == LICHII_TIER_ULTRA && info.max_fps >= 120 ? 1 : 0;
    out.gpu_accuracy = static_cast<int>(Settings::GpuAccuracy::Low);
    out.async_shaders = 1;
    out.docked = 0;
    if (info.increased_memory_limit == 0) {
        CopyString(out.note, sizeof(out.note),
                   "The increased memory limit entitlement is missing: enable it for Lichii's "
                   "App ID at developer.apple.com and reinstall, or games will be closed by iOS.");
    }

    // Budgets from what this process may allocate. Note that EDEN_NX2_TC_*_MIB compares against
    // total_used_memory = runtime.GetDeviceMemoryUsage() (the entire unified Metal/Vulkan heap:
    // render targets + textures + buffers + staging + VMA block slack, ~1.8-2.2 GiB in Ocarina's
    // Kokiri Village), NOT cached textures alone: setting critical to 18% (1,195 MiB) locked the
    // texture collector in permanent aggressive mode doing dozens of synchronous runtime.Finish()
    // GPU readback stalls every frame (2-4 FPS). Runtime host_memory_cap in texture_cache.h still
    // clamps dynamically whenever remaining process headroom gets tight.
    u64 available = info.launch_available_memory_mib;
    if (available == 0) {
        available = info.physical_memory_mib * 6 / 10;
    }
    const u64 critical = Clamp<u64>(available * 36 / 100, 1280, 2560);
    out.texture_cache_critical_mib = static_cast<u32>(critical);
    out.texture_cache_expected_mib = static_cast<u32>(critical * 3 / 4);
    out.buffer_cache_collect_mib = static_cast<u32>(Clamp<u64>(available / 32, 160, 256));
    out.memory_low_water_mib = static_cast<u32>(Clamp<u64>(available / 12, 384, 640));
    return out;
}

void ApplyMemoryBudgets(const LichiiDeviceDefaults& defaults) {
    // Kept when already set, so EDEN_NX2_* overrides from a launch environment still win.
    setenv("EDEN_NX2_TC_CRITICAL_MIB", std::to_string(defaults.texture_cache_critical_mib).c_str(),
           0);
    setenv("EDEN_NX2_TC_EXPECTED_MIB", std::to_string(defaults.texture_cache_expected_mib).c_str(),
           0);
    setenv("EDEN_NX2_BC_COLLECT_MIB", std::to_string(defaults.buffer_cache_collect_mib).c_str(), 0);
    if (lichii_platform_experiment("nup_in_memory")) {
        LOG_WARNING(Frontend, "iOS: experiment nup_in_memory on");
    } else {
        setenv("EDEN_NUP_CACHE_MIB", std::to_string(NupCacheMib).c_str(), 0);
    }
    Common::PlatformEvents::SetMemoryLowWater(u64{defaults.memory_low_water_mib} << 20);
    const char* const nup_cache = std::getenv("EDEN_NUP_CACHE_MIB");
    LOG_INFO(Frontend,
             "iOS: tier {}, texture cache {} / {} MiB, buffer cache collects at {} MiB, caches "
             "collect below {} MiB left, decompressed .nup files {} MiB",
             defaults.tier_name, defaults.texture_cache_expected_mib,
             defaults.texture_cache_critical_mib, defaults.buffer_cache_collect_mib,
             defaults.memory_low_water_mib, nup_cache != nullptr ? nup_cache : "512");
    RemoveLeftoverNupFiles();
}

void ApplyExperiments() {
    if (lichii_platform_experiment("untagged_guest")) {
        setenv("LICHII_IOS_UNTAGGED_GUEST", "1", 0);
        LOG_WARNING(Frontend, "iOS: experiment untagged_guest on");
    }
    if (lichii_platform_experiment("chunked_guest")) {
        setenv("LICHII_IOS_CHUNKED_GUEST", "1", 0);
        LOG_WARNING(Frontend, "iOS: experiment chunked_guest on");
    }
    // Set in Experiments either way, or neither: guest memory's own default.
    if (lichii_platform_experiment("file_guest")) {
        setenv("LICHII_IOS_FILE_GUEST", "1", 0);
        LOG_WARNING(Frontend, "iOS: experiment file_guest on");
    } else if (lichii_platform_experiment("file_guest_off")) {
        setenv("LICHII_IOS_FILE_GUEST", "0", 0);
        LOG_WARNING(Frontend, "iOS: experiment file_guest off");
    } else {
        // Set either way, so a switch in Experiments later replaces the value instead of growing
        // the environment while other threads read it.
        setenv("LICHII_IOS_FILE_GUEST", Common::HostMemory::FileGuestByDefault() ? "1" : "0", 0);
    }
    if (lichii_platform_experiment("no_zero_trim")) {
        setenv("EDEN_NX2_TRIM_ZERO", "0", 0);
        LOG_WARNING(Frontend, "iOS: experiment no_zero_trim on");
    }
    if (lichii_platform_experiment("no_early_submits")) {
        setenv("EDEN_NX2_STAGING_SUBMIT_MIB", "0", 0);
        LOG_WARNING(Frontend, "iOS: experiment no_early_submits on");
    }
    if (lichii_platform_experiment("no_mesh_emulation")) {
        setenv("EDEN_NX2_NO_MESH_EMULATION", "1", 0);
        LOG_WARNING(Frontend, "iOS: experiment no_mesh_emulation on");
    }
    if (lichii_platform_experiment("mvk_uint8_indices")) {
        setenv("EDEN_NX2_UINT8_INDICES", "1", 0);
        LOG_WARNING(Frontend, "iOS: experiment mvk_uint8_indices on");
    }
}

LichiiThermalAdvice ComputeThermalAdvice(int thermal_state, int low_power_mode) {
    LichiiThermalAdvice out{};
    out.thermal_state = thermal_state;
    out.low_power_mode = low_power_mode;
    if (thermal_state >= 3) {
        out.disable_frame_gen = 1;
        out.max_fps = 30;
        out.lower_resolution = 1;
        out.suggest_pause = 1;
        CopyString(out.message, sizeof(out.message),
                   "Device is very hot and iOS is slowing Lichii down hard. Pause for a while, or "
                   "turn frame generation off and lower the resolution.");
    } else if (thermal_state == 2) {
        out.disable_frame_gen = 1;
        out.max_fps = 30;
        out.lower_resolution = 1;
        CopyString(out.message, sizeof(out.message),
                   "Device is hot and iOS may slow Lichii down. Turning frame generation off or "
                   "lowering the resolution helps.");
    } else if (low_power_mode != 0) {
        out.disable_frame_gen = 1;
        CopyString(out.message, sizeof(out.message),
                   "Low Power Mode is on: the CPU and GPU run slower.");
    }
    return out;
}

} // namespace LichiiPlatform

extern "C" {

int lichii_platform_device_defaults(LichiiDeviceDefaults* out) {
    LichiiDeviceInfo info{};
    if (out == nullptr || !lichii_platform_device_info(&info)) {
        return 0;
    }
    *out = LichiiPlatform::ComputeDefaults(info);
    return 1;
}

int lichii_platform_apply_device_defaults(int mode) {
    LichiiDeviceDefaults defaults{};
    if (!lichii_platform_device_defaults(&defaults)) {
        return 0;
    }
    const auto marker = LichiiPlatform::ConfigPath("ios_device_defaults");
    const std::string stamp = fmt::format("{} {}", LichiiPlatform::DefaultsVersion, defaults.tier);
    if (mode == LICHII_DEFAULTS_FIRST_LAUNCH) {
        std::ifstream in{marker};
        std::string line;
        if (std::getline(in, line) && line == stamp) {
            return 0;
        }
    }
    const bool only_unchanged = mode == LICHII_DEFAULTS_UNCHANGED;
    bool applied = false;
    const auto apply = [&](auto& setting, auto value) {
        using Value = std::remove_cvref_t<decltype(setting.GetValue())>;
        const Value typed = static_cast<Value>(value);
        if (only_unchanged && setting.GetValue() != setting.GetDefault()) {
            return;
        }
        if (setting.GetValue() != typed) {
            setting.SetValue(typed);
            applied = true;
        }
    };
    auto& values = Settings::values;
    apply(values.resolution_setup, defaults.resolution_setup);
    apply(values.frame_gen, defaults.frame_gen != 0);
    apply(values.gpu_accuracy, defaults.gpu_accuracy);
    apply(values.use_asynchronous_shaders, defaults.async_shaders != 0);
    apply(values.use_docked_mode,
          defaults.docked != 0 ? Settings::ConsoleMode::Docked : Settings::ConsoleMode::Handheld);
    Settings::UpdateRescalingInfo();
    if (mode != LICHII_DEFAULTS_UNCHANGED) {
        std::error_code error;
        std::filesystem::create_directories(marker.parent_path(), error);
        std::ofstream out{marker, std::ios::trunc};
        out << stamp << '\n';
    }
    LOG_INFO(Frontend, "iOS: device defaults ({}, mode {}): {}x resolution, frame generation {}{}",
             defaults.tier_name, mode, defaults.resolution_scale, defaults.frame_gen != 0,
             applied ? "" : " (nothing changed)");
    return applied ? 1 : 0;
}

void lichii_platform_thermal_advice(LichiiThermalAdvice* out) {
    if (out != nullptr) {
        *out = LichiiPlatform::ComputeThermalAdvice(lichii_platform_thermal_state(),
                                                   lichii_platform_low_power_mode());
    }
}

void lichii_platform_set_thermal_max_fps(int fps) {
    Common::PlatformEvents::SetThermalMaxFps(fps > 0 ? static_cast<u32>(fps) : 0U);
}

void lichii_platform_trim_memory(void) {
    Common::PlatformEvents::SignalMemoryPressure();
}

void lichii_platform_note_shader_progress(int stage, uint64_t done, uint64_t total) {
    using namespace std::chrono;
    const int previous = LichiiPlatform::g_shader_stage.exchange(stage, std::memory_order_relaxed);
    LichiiPlatform::g_shader_done.store(done, std::memory_order_relaxed);
    LichiiPlatform::g_shader_total.store(total, std::memory_order_relaxed);
    // The handler hops to the main queue: at most ten calls a second, plus stage changes and the
    // last shader.
    const s64 now =
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    s64 last = LichiiPlatform::g_shader_last_post_ms.load(std::memory_order_relaxed);
    if (previous == stage && done < total) {
        if (now - last < 100 ||
            !LichiiPlatform::g_shader_last_post_ms.compare_exchange_strong(last, now)) {
            return;
        }
    } else {
        LichiiPlatform::g_shader_last_post_ms.store(now, std::memory_order_relaxed);
    }
    std::scoped_lock lock{LichiiPlatform::g_shader_handler_mutex};
    if (LichiiPlatform::g_shader_handler != nullptr) {
        LichiiPlatform::g_shader_handler(stage, done, total, LichiiPlatform::g_shader_handler_user);
    }
}

int lichii_platform_shader_progress(uint64_t* done, uint64_t* total) {
    if (done != nullptr) {
        *done = LichiiPlatform::g_shader_done.load(std::memory_order_relaxed);
    }
    if (total != nullptr) {
        *total = LichiiPlatform::g_shader_total.load(std::memory_order_relaxed);
    }
    return LichiiPlatform::g_shader_stage.load(std::memory_order_relaxed);
}

void lichii_platform_reset_shader_progress(void) {
    lichii_platform_note_shader_progress(LICHII_SHADER_IDLE, 0, 0);
}

void lichii_platform_set_shader_progress_handler(LichiiShaderProgressFn handler, void* user) {
    std::scoped_lock lock{LichiiPlatform::g_shader_handler_mutex};
    LichiiPlatform::g_shader_handler = handler;
    LichiiPlatform::g_shader_handler_user = user;
}

uint64_t lichii_platform_shader_cache_bytes(uint64_t title_id) {
    std::error_code error;
    u64 bytes = 0;
    const auto root = LichiiPlatform::ShaderCachePath(title_id);
    for (auto it = std::filesystem::recursive_directory_iterator(root, error);
         !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
        if (it->is_regular_file(error)) {
            bytes += it->file_size(error);
        }
    }
    return bytes;
}

int lichii_platform_clear_shader_cache(uint64_t title_id) {
    const auto root = LichiiPlatform::ShaderCachePath(title_id);
    std::error_code error;
    if (!std::filesystem::exists(root, error)) {
        return 0;
    }
    int removed = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(root, error);
         !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
        if (it->is_regular_file(error)) {
            ++removed;
        }
    }
    if (title_id == 0) {
        // Keep the shader directory itself.
        for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
            std::filesystem::remove_all(entry.path(), error);
            if (error) {
                break;
            }
        }
    } else {
        std::filesystem::remove_all(root, error);
    }
    if (error) {
        LOG_ERROR(Frontend, "iOS: clearing the shader cache {} failed: {}",
                  Common::FS::PathToUTF8String(root), error.message());
        return -1;
    }
    LOG_INFO(Frontend, "iOS: cleared the shader cache {} ({} files)",
             Common::FS::PathToUTF8String(root), removed);
    return removed;
}

void lichii_platform_set_experiment(const char* name, int enabled) {
    if (!LichiiPlatform::ValidExperimentName(name)) {
        return;
    }
    const auto path = LichiiPlatform::ConfigPath("experiments") / name;
    std::error_code error;
    if (enabled != 0) {
        std::filesystem::create_directories(path.parent_path(), error);
        std::ofstream{path} << "1\n";
    } else {
        std::filesystem::remove(path, error);
    }
}

int lichii_platform_experiment(const char* name) {
    if (!LichiiPlatform::ValidExperimentName(name)) {
        return 0;
    }
    std::error_code error;
    return std::filesystem::exists(LichiiPlatform::ConfigPath("experiments") / name, error) ? 1
                                                                                            : 0;
}

int lichii_platform_file_guest(void) {
    if (lichii_platform_experiment("file_guest")) {
        return 1;
    }
    if (lichii_platform_experiment("file_guest_off")) {
        return 0;
    }
    return Common::HostMemory::FileGuestByDefault() ? 1 : 0;
}

void lichii_platform_set_file_guest(int enabled) {
    lichii_platform_set_experiment("file_guest", enabled);
    lichii_platform_set_experiment("file_guest_off", enabled == 0 ? 1 : 0);
    // HostMemory reads it each time a game boots; the user's choice replaces the launch value.
    setenv("LICHII_IOS_FILE_GUEST", enabled != 0 ? "1" : "0", 1);
    LOG_INFO(HW_Memory, "iOS app: Keep Guest Memory in a File turned {} (from the next game start)",
             enabled != 0 ? "on" : "off");
}

int lichii_platform_file_guest_storage(LichiiFileGuestStorage* out) {
    if (out == nullptr) {
        return 0;
    }
    *out = {};
    const char* const directory = std::getenv("TMPDIR");
    struct statfs volume {};
    if (directory == nullptr || directory[0] == '\0' || statfs(directory, &volume) != 0) {
        return 0;
    }
    // The figures of HostMemory InitFile: a 14 GiB file, refused under min(file, 8 GiB) + 2 GiB
    // free, reserved in full from file + 2 GiB free.
    constexpr u64 FileMib = 14336;
    constexpr u64 MarginMib = 2048;
    constexpr u64 MinReservedMib = 8192;
    out->free_mib = (static_cast<u64>(volume.f_bavail) * volume.f_bsize) >> 20;
    out->file_mib = FileMib;
    out->needed_mib = std::min(FileMib, MinReservedMib) + MarginMib;
    out->full_mib = FileMib + MarginMib;
    return 1;
}

void lichii_platform_log_memory(const char* message) {
    if (message != nullptr) {
        LOG_INFO(HW_Memory, "iOS app: {}", message);
    }
}

} // extern "C"
