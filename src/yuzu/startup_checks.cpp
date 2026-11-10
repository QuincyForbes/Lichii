// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Don't move this! put it below and MSVC will get angry!
#include "video_core/vulkan_common/vulkan_wrapper.h"

#ifdef _WIN32
#include <cstring>
#include <processthreadsapi.h>
#include <windows.h>
#else
#include <cstring>
#include <errno.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <string>

#include <fmt/core.h>
#include "common/common_types.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/string_util.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "yuzu/startup_checks.h"

void CheckVulkan() {
    // Just start the Vulkan loader, this will crash if something is wrong
    try {
        Vulkan::vk::InstanceDispatch dld;
        const auto library = Vulkan::OpenLibrary();
        const Vulkan::vk::Instance instance =
            Vulkan::CreateInstance(*library, dld, VK_API_VERSION_1_1);

    } catch (const Vulkan::vk::Exception& exception) {
        fmt::print(stderr, "Failed to initialize Vulkan: {}\n", exception.what());
    }
}

#ifdef _WIN32
namespace {

void AddFileStamp(std::string& out, const std::wstring& path) {
    WIN32_FILE_ATTRIBUTE_DATA data{};
    out += Common::UTF16ToUTF8(path);
    if (GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data)) {
        out += fmt::format("|{}|{}|{}|{};", data.nFileSizeHigh, data.nFileSizeLow,
                           data.ftLastWriteTime.dwHighDateTime,
                           data.ftLastWriteTime.dwLowDateTime);
    } else {
        out += "|missing;";
    }
}

/// The loader's driver and layer lists: the value names are manifest paths, stamped like files.
void AddManifests(std::string& out, HKEY root, const wchar_t* subkey) {
    HKEY key{};
    if (RegOpenKeyExW(root, subkey, 0, KEY_READ, &key) != ERROR_SUCCESS) {
        return;
    }
    out += Common::UTF16ToUTF8(subkey);
    out += ':';
    for (DWORD index = 0;; ++index) {
        wchar_t name[1024];
        DWORD name_length = static_cast<DWORD>(std::size(name));
        DWORD enabled = 0;
        DWORD size = sizeof(enabled);
        const LONG result = RegEnumValueW(key, index, name, &name_length, nullptr, nullptr,
                                          reinterpret_cast<BYTE*>(&enabled), &size);
        if (result == ERROR_NO_MORE_ITEMS) {
            break;
        }
        if (result == ERROR_SUCCESS) {
            AddFileStamp(out, std::wstring{name, name_length});
            out += std::to_string(enabled);
        }
    }
    RegCloseKey(key);
}

/// What a crash of the Vulkan check depends on: this executable, the Vulkan loader, the display
/// drivers' versions, the drivers and layers registered with the loader and the loader's
/// environment. The check runs again whenever one of them changes.
std::string VulkanCheckFingerprint() {
    std::string out;
    wchar_t path[MAX_PATH];
    if (const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
        length != 0 && length < MAX_PATH) {
        AddFileStamp(out, path);
    }
    if (const UINT length = GetSystemDirectoryW(path, MAX_PATH);
        length != 0 && length < MAX_PATH) {
        AddFileStamp(out, std::wstring{path, length} + L"\\vulkan-1.dll");
    }

    constexpr const wchar_t* display_class =
        L"SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";
    HKEY adapters{};
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, display_class, 0, KEY_READ, &adapters) ==
        ERROR_SUCCESS) {
        for (DWORD index = 0;; ++index) {
            wchar_t name[256];
            DWORD name_length = static_cast<DWORD>(std::size(name));
            const LONG result = RegEnumKeyExW(adapters, index, name, &name_length, nullptr,
                                              nullptr, nullptr, nullptr);
            if (result == ERROR_NO_MORE_ITEMS) {
                break;
            }
            if (result != ERROR_SUCCESS) {
                continue;
            }
            for (const wchar_t* value_name :
                 {L"DriverVersion", L"DriverDate", L"VulkanDriverName"}) {
                wchar_t value[1024];
                DWORD size = sizeof(value);
                if (RegGetValueW(adapters, name, value_name, RRF_RT_REG_SZ | RRF_RT_REG_MULTI_SZ,
                                 nullptr, value, &size) == ERROR_SUCCESS) {
                    out += Common::UTF16ToUTF8(std::wstring_view{value, size / sizeof(wchar_t)});
                    out += ';';
                }
            }
        }
        RegCloseKey(adapters);
    }

    for (const HKEY root : {HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER}) {
        for (const wchar_t* subkey : {L"SOFTWARE\\Khronos\\Vulkan\\Drivers",
                                      L"SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers",
                                      L"SOFTWARE\\Khronos\\Vulkan\\ExplicitLayers"}) {
            AddManifests(out, root, subkey);
        }
    }
    for (const char* variable :
         {"VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES", "VK_LAYER_PATH",
          "VK_ADD_LAYER_PATH", "VK_IMPLICIT_LAYER_PATH", "VK_ADD_IMPLICIT_LAYER_PATH",
          "VK_INSTANCE_LAYERS", "VK_LOADER_LAYERS_ENABLE", "VK_LOADER_LAYERS_DISABLE",
          "VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE"}) {
        if (const char* value = std::getenv(variable)) {
            out += fmt::format("{}={};", variable, value);
        }
    }

    // FNV-1a: the file keeps a hash, not the paths.
    u64 hash = 0xcbf29ce484222325ULL;
    for (const char c : out) {
        hash = (hash ^ static_cast<u8>(c)) * 0x100000001b3ULL;
    }
    return fmt::format("vulkan-check-ok {:016x}", hash);
}

std::filesystem::path VulkanCheckCachePath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "vulkan_check.txt";
}

} // Anonymous namespace
#endif
bool CheckEnvVars(bool* is_child) {
#ifdef _WIN32
    // Check environment variable to see if we are the child
    char variable_contents[8];
    const DWORD startup_check_var =
        GetEnvironmentVariableA(STARTUP_CHECK_ENV_VAR, variable_contents, 8);
    if (startup_check_var > 0 && std::strncmp(variable_contents, ENV_VAR_ENABLED_TEXT, 8) == 0) {
        CheckVulkan();
        return true;
    }

    // Don't perform startup checks if we are a child process
    char is_child_s[8];
    const DWORD is_child_len = GetEnvironmentVariableA(IS_CHILD_ENV_VAR, is_child_s, 8);
    if (is_child_len > 0 && std::strncmp(is_child_s, ENV_VAR_ENABLED_TEXT, 8) == 0) {
        *is_child = true;
        return false;
    } else if (!SetEnvironmentVariableA(IS_CHILD_ENV_VAR, ENV_VAR_ENABLED_TEXT)) {
        fmt::print(stderr, "SetEnvironmentVariableA failed to set {} with error {}\n",
                   IS_CHILD_ENV_VAR, GetLastError());
        return true;
    }
#else
    const char* startup_check_var = getenv(STARTUP_CHECK_ENV_VAR);
    if (startup_check_var != nullptr &&
        std::strncmp(startup_check_var, ENV_VAR_ENABLED_TEXT, 8) == 0) {
        CheckVulkan();
        return true;
    }
#endif
    return false;
}

bool StartupChecks(const char* arg0, bool* has_broken_vulkan, bool perform_vulkan_check) {
#ifdef _WIN32
    // Set the startup variable for child processes
    const bool env_var_set = SetEnvironmentVariableA(STARTUP_CHECK_ENV_VAR, ENV_VAR_ENABLED_TEXT);
    if (!env_var_set) {
        fmt::print(stderr, "SetEnvironmentVariableA failed to set {} with error {}\n",
                   STARTUP_CHECK_ENV_VAR, GetLastError());
        return false;
    }

    // The child is spawned only when something the check depends on changed since it last
    // passed: it cost 0.3-0.8 s before the window on every start.
    const std::string fingerprint = perform_vulkan_check ? VulkanCheckFingerprint() : std::string{};
    if (perform_vulkan_check &&
        Common::FS::ReadStringFromFile(VulkanCheckCachePath(), Common::FS::FileType::TextFile) ==
            fingerprint) {
        perform_vulkan_check = false;
    }

    if (perform_vulkan_check) {
        // Spawn child process that performs Vulkan check
        PROCESS_INFORMATION process_info;
        std::memset(&process_info, '\0', sizeof(process_info));

        if (!SpawnChild(arg0, &process_info, 0)) {
            return false;
        }

        // Wait until the process exits and get exit code from it
        WaitForSingleObject(process_info.hProcess, INFINITE);
        DWORD exit_code = STILL_ACTIVE;
        const int err = GetExitCodeProcess(process_info.hProcess, &exit_code);
        if (err == 0) {
            fmt::print(stderr, "GetExitCodeProcess failed with error {}\n", GetLastError());
        }

        // Vulkan is broken if the child crashed (return value is not zero)
        *has_broken_vulkan = (exit_code != 0);
        const auto cache_path = VulkanCheckCachePath();
        if (*has_broken_vulkan) {
            Common::FS::RemoveFile(cache_path);
        } else if (Common::FS::CreateParentDirs(cache_path)) {
            void(Common::FS::WriteStringToFile(cache_path, Common::FS::FileType::TextFile,
                                               fingerprint));
        }

        if (CloseHandle(process_info.hProcess) == 0) {
            fmt::print(stderr, "CloseHandle failed with error {}\n", GetLastError());
        }
        if (CloseHandle(process_info.hThread) == 0) {
            fmt::print(stderr, "CloseHandle failed with error {}\n", GetLastError());
        }
    }

    if (!SetEnvironmentVariableA(STARTUP_CHECK_ENV_VAR, nullptr)) {
        fmt::print(stderr, "SetEnvironmentVariableA failed to clear {} with error {}\n",
                   STARTUP_CHECK_ENV_VAR, GetLastError());
    }

#else
    const int env_var_set = setenv(STARTUP_CHECK_ENV_VAR, ENV_VAR_ENABLED_TEXT, 1);
    if (env_var_set == -1) {
        const int err = errno;
        fmt::print(stderr, "setenv failed to set {} with error {}\n", STARTUP_CHECK_ENV_VAR, err);
        return false;
    }

    if (perform_vulkan_check) {
        const pid_t pid = SpawnChild(arg0);
        if (pid == -1) {
            return false;
        }

        // Get exit code from child process
        int status;
        const int r_val = waitpid(pid, &status, 0);
        if (r_val == -1) {
            const int err = errno;
            fmt::print(stderr, "wait failed with error {}\n", err);
            return false;
        }
        // Vulkan is broken if the child crashed (return value is not zero)
        *has_broken_vulkan = (status != 0);
    }

    const int env_var_cleared = unsetenv(STARTUP_CHECK_ENV_VAR);
    if (env_var_cleared == -1) {
        const int err = errno;
        fmt::print(stderr, "unsetenv failed to clear {} with error {}\n", STARTUP_CHECK_ENV_VAR,
                   err);
    }
#endif
    return false;
}

#ifdef _WIN32
bool SpawnChild(const char* arg0, PROCESS_INFORMATION* pi, int flags) {
    STARTUPINFOA startup_info;

    std::memset(&startup_info, '\0', sizeof(startup_info));
    startup_info.cb = sizeof(startup_info);

    char p_name[255];
    std::strncpy(p_name, arg0, 254);
    p_name[254] = '\0';

    const bool process_created = CreateProcessA(nullptr,       // lpApplicationName
                                                p_name,        // lpCommandLine
                                                nullptr,       // lpProcessAttributes
                                                nullptr,       // lpThreadAttributes
                                                false,         // bInheritHandles
                                                flags,         // dwCreationFlags
                                                nullptr,       // lpEnvironment
                                                nullptr,       // lpCurrentDirectory
                                                &startup_info, // lpStartupInfo
                                                pi             // lpProcessInformation
    );
    if (!process_created) {
        fmt::print(stderr, "CreateProcessA failed with error {}\n", GetLastError());
        return false;
    }

    return true;
}
#else
pid_t SpawnChild(const char* arg0) {
    const pid_t pid = fork();

    if (pid == -1) {
        // error
        const int err = errno;
        fmt::print(stderr, "fork failed with error {}\n", err);
        return pid;
    } else if (pid == 0) {
        // child
        execlp(arg0, arg0, nullptr);
        const int err = errno;
        fmt::print(stderr, "execl failed with error {}\n", err);
        _exit(0);
    }

    return pid;
}
#endif
