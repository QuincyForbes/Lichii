// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <mutex>
#include <system_error>
#include <utility>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/vulkan_common/ffx_runtime.h"

#ifdef _WIN32
#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <windows.h>
#include <bcrypt.h>
#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>
#elif defined(__linux__) && !defined(__ANDROID__)
#include <dlfcn.h>
#endif

namespace Vulkan::FfxRuntime {

namespace {

struct Session {
    std::mutex mutex;
    std::atomic<bool> started{};
    std::atomic<bool> available{};
    std::atomic<u64> frames{};
    std::string version;
    std::string problem;
};

Session& GetSession() {
    static Session session;
    return session;
}

} // Anonymous namespace

SessionStatus GetSessionStatus() {
    Session& session = GetSession();
    SessionStatus status{
        .started = session.started.load(std::memory_order_relaxed),
        .available = session.available.load(std::memory_order_relaxed),
        .frames = session.frames.load(std::memory_order_relaxed),
    };
    std::scoped_lock lock{session.mutex};
    status.version = session.version;
    status.problem = session.problem;
    return status;
}

void SetProblem(std::string problem) {
    Session& session = GetSession();
    session.started.store(true, std::memory_order_relaxed);
    session.available.store(false, std::memory_order_relaxed);
    std::scoped_lock lock{session.mutex};
    session.problem = std::move(problem);
}

void SetAvailable(bool available, std::string version) {
    Session& session = GetSession();
    session.started.store(true, std::memory_order_relaxed);
    session.available.store(available, std::memory_order_relaxed);
    std::scoped_lock lock{session.mutex};
    if (!version.empty()) {
        session.version = std::move(version);
    }
    if (available) {
        session.problem.clear();
    }
}

void NoteEvaluation() {
    Session& session = GetSession();
    session.available.store(true, std::memory_order_relaxed);
    session.frames.fetch_add(1, std::memory_order_relaxed);
}

void ResetSessionStatus() {
    Session& session = GetSession();
    session.started.store(false, std::memory_order_relaxed);
    session.available.store(false, std::memory_order_relaxed);
    session.frames.store(0, std::memory_order_relaxed);
    std::scoped_lock lock{session.mutex};
    session.version.clear();
    session.problem.clear();
}

void StopSession() {
    Session& session = GetSession();
    session.started.store(false, std::memory_order_relaxed);
    session.available.store(false, std::memory_order_relaxed);
}

std::filesystem::path Folder() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::FidelityFxDir);
}

bool FilesPresent() {
    const std::filesystem::path folder = Folder();
    std::error_code ec;
    for (const File& file : FILES) {
        if (std::filesystem::file_size(folder / file.name, ec) != file.size || ec) {
            return false;
        }
    }
    return true;
}

#ifdef _WIN32

std::filesystem::path PortLibraryPath() {
    return {};
}

bool IsSignedByAmd(const std::filesystem::path& path) {
    const std::wstring file_path = std::filesystem::absolute(path).wstring();
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof(file);
    file.pcwszFilePath = file_path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    bool valid = WinVerifyTrust(nullptr, &action, &data) == ERROR_SUCCESS;
    if (valid) {
        CRYPT_PROVIDER_DATA* const provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        CRYPT_PROVIDER_SGNR* const signer =
            provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        CRYPT_PROVIDER_CERT* const cert = signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert &&
                CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name,
                                   static_cast<DWORD>(std::size(name))) > 1 &&
                std::wstring_view{name} == L"Advanced Micro Devices";
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

bool HasSha256(const std::filesystem::path& path, std::string_view sha256) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    BCRYPT_ALG_HANDLE algorithm{};
    if (!BCRYPT_SUCCESS(
            BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0))) {
        return false;
    }
    BCRYPT_HASH_HANDLE hash{};
    if (!BCRYPT_SUCCESS(BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0))) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return false;
    }
    std::vector<char> buffer(1 << 20);
    bool ok = true;
    while (ok && stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto read = static_cast<ULONG>(stream.gcount());
        ok = read == 0 || BCRYPT_SUCCESS(BCryptHashData(
                              hash, reinterpret_cast<PUCHAR>(buffer.data()), read, 0));
    }
    std::array<UCHAR, 32> digest{};
    ok = ok && stream.eof() &&
         BCRYPT_SUCCESS(BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0));
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    if (!ok || sha256.size() != digest.size() * 2) {
        return false;
    }
    std::array<char, 3> hex{};
    for (size_t i = 0; i < digest.size(); ++i) {
        std::snprintf(hex.data(), hex.size(), "%02x", digest[i]);
        if (sha256.substr(i * 2, 2) != std::string_view{hex.data(), 2}) {
            return false;
        }
    }
    return true;
}

namespace {

struct Loaded : Api {
    HMODULE module{};
};

std::mutex mutex;
std::weak_ptr<const Loaded> current;

/// Once every file checks out it isn't hashed again this session; a failed check is retried, so
/// files downloaded after a game ran are used by the next one.
bool FilesVerified() {
    static bool verified = false;
    if (verified) {
        return true;
    }
    const std::filesystem::path folder = Folder();
    for (const File& file : FILES) {
        const std::filesystem::path path = folder / file.name;
        std::error_code ec;
        const char* problem = nullptr;
        if (!std::filesystem::is_regular_file(path, ec)) {
            problem = "missing";
        } else if (std::filesystem::file_size(path, ec) != file.size ||
                   !HasSha256(path, file.sha256)) {
            problem = "not the pinned AMD FidelityFX SDK file";
        } else if (!IsSignedByAmd(path)) {
            problem = "not signed by AMD";
        }
        if (problem) {
            LOG_INFO(Render_Vulkan, "AMD FSR 4: {} is {}; using FSR 3.1",
                     Common::FS::PathToUTF8String(path), problem);
            SetProblem(fmt::format("{} is {}; using FSR 3.1.", file.name, problem));
            return false;
        }
    }
    verified = true;
    return true;
}

} // Anonymous namespace

std::shared_ptr<const Api> Acquire() {
    GetSession().started.store(true, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (auto existing = current.lock()) {
        return existing;
    }
    if (!FilesVerified()) {
        return nullptr;
    }
    // The loader finds the upscaler next to itself.
    const std::wstring loader = (Folder() / FILES[0].name).wstring();
    const HMODULE module = LoadLibraryExW(loader.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        const DWORD error = GetLastError();
        LOG_WARNING(Render_Vulkan, "AMD FSR 4: the FidelityFX loader doesn't load (error {})",
                    error);
        SetProblem(fmt::format("Windows could not load amd_fidelityfx_loader_vk.dll (error {}).",
                               error));
        return nullptr;
    }
    auto loaded = std::make_unique<Loaded>();
    loaded->module = module;
    loaded->create_context = reinterpret_cast<PfnFfxCreateContext>(
        reinterpret_cast<void*>(GetProcAddress(module, "ffxCreateContext")));
    loaded->destroy_context = reinterpret_cast<PfnFfxDestroyContext>(
        reinterpret_cast<void*>(GetProcAddress(module, "ffxDestroyContext")));
    loaded->query =
        reinterpret_cast<PfnFfxQuery>(reinterpret_cast<void*>(GetProcAddress(module, "ffxQuery")));
    loaded->dispatch = reinterpret_cast<PfnFfxDispatch>(
        reinterpret_cast<void*>(GetProcAddress(module, "ffxDispatch")));
    if (!loaded->create_context || !loaded->destroy_context || !loaded->query ||
        !loaded->dispatch) {
        LOG_WARNING(Render_Vulkan, "AMD FSR 4: the FidelityFX loader lacks its exports");
        SetProblem("The FidelityFX loader lacks its FFX API exports; using FSR 3.1.");
        FreeLibrary(module);
        return nullptr;
    }
    LOG_INFO(Render_Vulkan, "AMD FSR 4: FidelityFX SDK {} runtime loaded", SDK_VERSION);
    std::shared_ptr<const Loaded> shared{loaded.release(), [](const Loaded* l) {
                                             FreeLibrary(l->module);
                                             delete l;
                                         }};
    current = shared;
    return shared;
}

#else

bool IsSignedByAmd(const std::filesystem::path&) {
    return false;
}

bool HasSha256(const std::filesystem::path&, std::string_view) {
    return false;
}

#if defined(__linux__) && !defined(__ANDROID__)

std::filesystem::path PortLibraryPath() {
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        const std::filesystem::path bundled =
            exe.parent_path().parent_path() / "lib" / PORT_LIBRARY;
        if (std::filesystem::is_regular_file(bundled, ec)) {
            return bundled;
        }
    }
    const std::filesystem::path user = Folder() / PORT_LIBRARY;
    return std::filesystem::is_regular_file(user, ec) ? user : std::filesystem::path{};
}

namespace {

struct Loaded : Api {
    void* module{};
};

std::mutex mutex;
std::weak_ptr<const Loaded> current;

} // Anonymous namespace

std::shared_ptr<const Api> Acquire() {
    GetSession().started.store(true, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    if (auto existing = current.lock()) {
        return existing;
    }
    const std::filesystem::path path = PortLibraryPath();
    if (path.empty()) {
        LOG_INFO(Render_Vulkan, "FSR 4: {} is not in this build or in {}; using FSR 3.1",
                 PORT_LIBRARY, Common::FS::PathToUTF8String(Folder()));
        SetProblem(fmt::format("{} is missing; using FSR 3.1.", PORT_LIBRARY));
        return nullptr;
    }
    void* const module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        const char* const err = dlerror();
        LOG_WARNING(Render_Vulkan, "FSR 4: {} doesn't load ({}); using FSR 3.1",
                    Common::FS::PathToUTF8String(path), err ? err : "unknown error");
        SetProblem(fmt::format("Could not load {} ({}); using FSR 3.1.", PORT_LIBRARY,
                               err ? err : "unknown error"));
        return nullptr;
    }
    auto loaded = std::make_unique<Loaded>();
    loaded->module = module;
    loaded->create_context =
        reinterpret_cast<PfnFfxCreateContext>(dlsym(module, "ffxCreateContext"));
    loaded->destroy_context =
        reinterpret_cast<PfnFfxDestroyContext>(dlsym(module, "ffxDestroyContext"));
    loaded->query = reinterpret_cast<PfnFfxQuery>(dlsym(module, "ffxQuery"));
    loaded->dispatch = reinterpret_cast<PfnFfxDispatch>(dlsym(module, "ffxDispatch"));
    if (!loaded->create_context || !loaded->destroy_context || !loaded->query ||
        !loaded->dispatch) {
        LOG_WARNING(Render_Vulkan, "FSR 4: {} lacks the FFX API exports; using FSR 3.1",
                    Common::FS::PathToUTF8String(path));
        SetProblem(fmt::format("{} lacks the FFX API exports; using FSR 3.1.", PORT_LIBRARY));
        dlclose(module);
        return nullptr;
    }
    LOG_INFO(Render_Vulkan, "FSR 4: loaded the fsr4-vk community Vulkan port from {}",
             Common::FS::PathToUTF8String(path));
    std::shared_ptr<const Loaded> shared{loaded.release(), [](const Loaded* l) {
                                             dlclose(l->module);
                                             delete l;
                                         }};
    current = shared;
    return shared;
}

#else

std::filesystem::path PortLibraryPath() {
    return {};
}

std::shared_ptr<const Api> Acquire() {
    return nullptr;
}

#endif

#endif

} // namespace Vulkan::FfxRuntime
