// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <dlfcn.h>
#include <cstdlib>
#include <string_view>
#endif
#include <string>

#ifdef _WIN32
#include <windows.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

#include "video_core/vulkan_common/vulkan_overlay_filter.h"
#endif

#include "common/dynamic_library.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#ifdef __APPLE__
#include "common/settings.h"
#endif
#ifdef _WIN32
#include "common/scope_exit.h"
#endif
#include "video_core/vulkan_common/vulkan_library.h"

namespace Vulkan {

#ifdef _WIN32
namespace {

void DisableOverwolfLayers() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::set<std::string> names;
        std::map<std::string, std::string> environment;
        for (const auto hive : {HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER}) {
            for (const auto view : {KEY_WOW64_64KEY, KEY_WOW64_32KEY}) {
                for (const auto subkey : {L"SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers",
                                          L"SOFTWARE\\Khronos\\Vulkan\\ExplicitLayers"}) {
                    HKEY key{};
                    if (RegOpenKeyExW(hive, subkey, 0, KEY_QUERY_VALUE | view, &key) !=
                        ERROR_SUCCESS) {
                        continue;
                    }
                    SCOPE_EXIT {
                        RegCloseKey(key);
                    };
                    for (DWORD index = 0;; ++index) {
                        std::wstring path(32768, L'\0');
                        DWORD length = static_cast<DWORD>(path.size());
                        DWORD type{}, enabled{}, size = sizeof(enabled);
                        const auto result =
                            RegEnumValueW(key, index, path.data(), &length, nullptr, &type,
                                          reinterpret_cast<BYTE*>(&enabled), &size);
                        if (result == ERROR_NO_MORE_ITEMS) {
                            break;
                        }
                        if (result != ERROR_SUCCESS || type != REG_DWORD ||
                            size != sizeof(enabled) || enabled != 0) {
                            continue;
                        }
                        path.resize(length);
                        try {
                            const std::filesystem::path manifest_path{path};
                            std::error_code error;
                            const auto bytes = std::filesystem::file_size(manifest_path, error);
                            if (error || bytes > 1024 * 1024) {
                                continue;
                            }
                            std::ifstream file{manifest_path};
                            if (!file) {
                                continue;
                            }
                            const auto manifest = nlohmann::json::parse(file, nullptr, false);
                            const auto utf8 = manifest_path.u8string();
                            Detail::CollectOverwolfLayerControls(
                                manifest, std::string{utf8.begin(), utf8.end()}, names,
                                environment);
                        } catch (const std::exception& exception) {
                            LOG_WARNING(Render_Vulkan,
                                        "Skipping unreadable Vulkan layer manifest: {}",
                                        exception.what());
                        }
                    }
                }
            }
        }
        if (!names.empty()) {
            std::string disabled;
            if (const auto* current = std::getenv("VK_LOADER_LAYERS_DISABLE")) {
                disabled = current;
            }
            for (const auto& name : names) {
                if (!disabled.empty()) {
                    disabled += ',';
                }
                disabled += name;
            }
            environment.insert_or_assign("VK_LOADER_LAYERS_DISABLE", std::move(disabled));
        }
        for (const auto& [name, value] : environment) {
            if (_putenv_s(name.c_str(), value.c_str()) != 0) {
                LOG_WARNING(Render_Vulkan, "Could not set Overwolf layer disable control {}", name);
            }
        }
        if (!names.empty()) {
            LOG_INFO(Render_Vulkan,
                     "Disabled {} installed Overwolf Vulkan layer(s) for this process",
                     names.size());
        }
    });
}

} // namespace
#endif

std::shared_ptr<Common::DynamicLibrary> OpenLibrary(
    [[maybe_unused]] Core::Frontend::GraphicsContext* context) {
#ifdef _WIN32
    DisableOverwolfLayers();
#endif
    LOG_DEBUG(Render_Vulkan, "Looking for a Vulkan library");
#if defined(__ANDROID__) && defined(ARCHITECTURE_arm64)
    // Android manages its Vulkan driver from the frontend.
    return context->GetDriverLibrary();
#else
    auto library = std::make_shared<Common::DynamicLibrary>();
#ifdef __APPLE__
    static const bool mvk_tuning = [] {
        const char* const env = std::getenv("EDEN_NX2_MVK_TUNING");
        if (env != nullptr && env[0] == '0') {
            return false;
        }
        setenv("MVK_CONFIG_SHOULD_MAXIMIZE_CONCURRENT_COMPILATION", "1", 0);
        setenv("MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS", "0", 0);
        // MoltenVK 1.4.2 MVK_CONFIG_FAST_MATH_ON_DEMAND (2): fast math enabled in Metal shaders,
        // while honoring SPIR-V float controls (SignedZeroInfNanPreserve / ContractionOff).
        setenv("MVK_CONFIG_FAST_MATH_ENABLED", "2", 0);
        // Recover from transient command buffer failures (thermal/background/sleep) instead of
        // aborting with VK_ERROR_DEVICE_LOST, LZFSE-compress cached MSL strings, and suppress
        // MoltenVK info logs and per-command performance tracking overhead on macOS and iOS.
        setenv("MVK_CONFIG_RESUME_LOST_DEVICE", "1", 0);
        setenv("MVK_CONFIG_SHADER_COMPRESSION_ALGORITHM", "1", 0);
        setenv("MVK_CONFIG_LOG_LEVEL", "2", 0);
        setenv("MVK_CONFIG_PERFORMANCE_TRACKING", "0", 0);
#if TARGET_OS_IPHONE
        setenv("MVK_CONFIG_MAX_ACTIVE_METAL_COMMAND_BUFFERS_PER_QUEUE", "32", 0);
#endif
        return true;
    }();
    (void)mvk_tuning;
    // Two MoltenVKs are bundled (Settings moltenvk_driver): the release Lichii always shipped and
    // one with native mesh shaders (KhronosGroup/MoltenVK#2842). The first start of this process
    // picks one, so a process never loads both (they define the same Objective-C classes).
    // EDEN_NX2_MOLTENVK=mesh|legacy overrides the setting.
    static const bool want_mesh = [] {
        if (const char* const env = std::getenv("EDEN_NX2_MOLTENVK")) {
            return std::string_view{env} == "mesh";
        }
        return Settings::values.moltenvk_driver.GetValue() == Settings::MoltenVkDriver::Mesh;
    }();
#if TARGET_OS_IPHONE
    // iOS: the frameworks are embedded in the flat app bundle (no Contents/ directory).
    const auto frameworks = Common::FS::GetBundleDirectory() / "Frameworks";
    const auto mesh_filename = frameworks / "MoltenVKMesh.framework/MoltenVKMesh";
    const auto libvulkan_filename = frameworks / "MoltenVK.framework/MoltenVK";
    const auto libmoltenvk_filename = libvulkan_filename;
#else
    const auto frameworks = Common::FS::GetBundleDirectory() / "Contents/Frameworks";
    const auto mesh_filename = frameworks / "libMoltenVKMesh.dylib";
    const auto libvulkan_filename = frameworks / "libvulkan.1.dylib";
    const auto libmoltenvk_filename = frameworks / "libMoltenVK.dylib";
#endif
    const char* const env_path = std::getenv("LIBVULKAN_PATH");
    const char* library_paths[] = {env_path, want_mesh ? mesh_filename.c_str() : nullptr,
                                   libvulkan_filename.c_str(), libmoltenvk_filename.c_str()};
    std::string mesh_error;
    // Check if a path to a specific Vulkan library has been specified.
    for (const auto& library_path : library_paths) {
        if (library_path == nullptr) {
            continue;
        }
        const bool mesh = library_path == mesh_filename.c_str();
        if (!library->Open(library_path)) {
            if (mesh) {
                const char* const error = dlerror();
                mesh_error = error != nullptr ? error : "not found";
            }
            continue;
        }
        if (library->GetSymbolAddress("vkGetInstanceProcAddr") == nullptr) {
            LOG_ERROR(Render_Vulkan, "{} has no vkGetInstanceProcAddr", library_path);
            library->Close();
            continue;
        }
        // Warning level: iOS filters Render.Vulkan below it.
        if (want_mesh && !mesh && library_path != env_path) {
            LOG_WARNING(Render_Vulkan,
                        "MoltenVK with mesh shaders did not load ({}): legacy MoltenVK at {}",
                        mesh_error.empty() ? mesh_filename.string() : mesh_error, library_path);
        } else {
            LOG_WARNING(Render_Vulkan, "Vulkan library: {} ({} MoltenVK)", library_path,
                        library_path == env_path ? "LIBVULKAN_PATH" : (mesh ? "mesh" : "legacy"));
        }
        break;
    }
#else
    std::string filename = Common::DynamicLibrary::GetVersionedFilename("vulkan", 1);
    LOG_DEBUG(Render_Vulkan, "Trying Vulkan library: {}", filename);
    if (!library->Open(filename.c_str())) {
        // Android devices may not have libvulkan.so.1, only libvulkan.so.
        filename = Common::DynamicLibrary::GetVersionedFilename("vulkan");
        LOG_DEBUG(Render_Vulkan, "Trying Vulkan library (second attempt): {}", filename);
        void(library->Open(filename.c_str()));
    }
#endif
    return library;
#endif
}

} // namespace Vulkan
