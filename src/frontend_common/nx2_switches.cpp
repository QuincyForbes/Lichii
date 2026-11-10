// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "frontend_common/nx2_switches.h"

namespace FrontendCommon {

namespace {

// The NX2 switches are read from the environment, which a GUI or iOS launch has no way to set.
// nx2_switches.txt holds "NAME=VALUE" lines, one per line, '#' comments. Between files the later
// one wins, so the copy next to the executable (shipped with the build) can be overridden by the
// user's copy without editing the one that gets replaced.
void ParseSwitchFile(const std::filesystem::path& path,
                     std::map<std::string, std::string>& entries) {
    std::ifstream file{path};
    if (!file) {
        return;
    }
    constexpr const char* blank = " \t\r";
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(blank));
        line.erase(line.find_last_not_of(blank) + 1);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string name = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        name.erase(name.find_last_not_of(blank) + 1);
        value.erase(0, value.find_first_not_of(blank));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }
        if (!name.empty()) {
            entries.insert_or_assign(std::move(name), std::move(value));
        }
    }
}

std::filesystem::path ExeDirectory(const char* argv0) {
#ifdef _WIN32
    return Common::FS::GetExeDirectory();
#else
    std::error_code ec;
#ifdef __linux__
    if (auto exe = std::filesystem::read_symlink("/proc/self/exe", ec); !ec) {
        return exe.parent_path();
    }
#endif
    if (argv0 == nullptr) {
        return {};
    }
    const auto exe = std::filesystem::absolute(argv0, ec);
    return ec ? std::filesystem::path{} : exe.parent_path();
#endif
}

std::filesystem::path UserDirectory() {
#ifdef _WIN32
    return Common::FS::GetAppDataRoamingDirectory() / EDEN_DIR;
#elif defined(__APPLE__) && TARGET_OS_IPHONE
    // Documents/eden, which the Files app reaches.
    return Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir);
#else
    return Common::FS::GetDataDirectory("XDG_CONFIG_HOME") / EDEN_DIR;
#endif
}

// Built in so players don't have to set anything: the switches tuned for Ocarina of Time. Either
// nx2_switches.txt overrides any of them (=0 turns one off).
constexpr std::pair<const char*, const char*> BuiltinSwitches[] = {
    {"EDEN_NX2_GRAPHICS_SSBO_BARE", "0"}, {"EDEN_NX2_WRITTEN_SSBO_MARK_KIB", "64"},
    {"EDEN_NX2_ACQUIRE_PRECHECK", "2"},   {"EDEN_NX2_MACRO_SINK", "1"},
    {"EDEN_NX2_COMPUTE_SINK", "1"},       {"EDEN_NX2_RING_PAIRS", "1"},
    {"EDEN_NX2_UBO_INPLACE", "1"},        {"EDEN_NX2_PIPELINE_CLEAN", "1"},
    {"EDEN_NX2_PREFLUSH", "1"},           {"EDEN_NX2_SEM_FAST", "2"},
    // NVIDIA's driver shader cache must hold OoT's ~120,000 prepared pipelines. Its size is a
    // Control Panel setting newer drivers no longer show; the driver also reads these variables,
    // for this process only (from Fritchii v5.5, 4012e5271e).
    {"__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1"},
    {"__GL_SHADER_DISK_CACHE_SIZE", "10737418240"},
#ifdef __APPLE__
    // Ocarina's anti-aliasing network needs cooperative matrices, which MoltenVK lacks: emulated,
    // it costs more GPU time per frame than it saves (an RTX 5090 with them gains 41 -> 50 fps
    // without it). This game-side patch also drops its heavy dispatch.
    {"EDEN_NX2_OOT_NO_NDSR", "1"},
#endif
};

} // Anonymous namespace

void LoadNx2Switches(const char* argv0) {
    std::map<std::string, std::string> entries;
    for (const auto& [name, value] : BuiltinSwitches) {
        entries.emplace(name, value);
    }
    if (const auto exe_dir = ExeDirectory(argv0); !exe_dir.empty()) {
        ParseSwitchFile(exe_dir / "nx2_switches.txt", entries);
    }
    ParseSwitchFile(UserDirectory() / "nx2_switches.txt", entries);
    for (const auto& [name, value] : entries) {
        if (std::getenv(name.c_str()) != nullptr) {
            continue;
        }
#ifdef _WIN32
        _putenv_s(name.c_str(), value.c_str());
#else
        setenv(name.c_str(), value.c_str(), 1);
#endif
    }
}

} // namespace FrontendCommon
