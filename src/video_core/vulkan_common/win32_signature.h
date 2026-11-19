// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifdef _WIN32

#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

#include <windows.h>
#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>

namespace Vulkan {

/// True when `path` carries a valid Win32 Authenticode signature whose signer certificate display
/// name matches `expected_signer`.
[[nodiscard]] inline bool HasValidAuthenticodeSigner(const std::filesystem::path& path,
                                                     std::wstring_view expected_signer) {
    std::error_code ec;
    const std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
    const std::wstring file_path = (ec ? path : abs_path).wstring();
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
        CRYPT_PROVIDER_CERT* const cert =
            signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert &&
                CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name,
                                   static_cast<DWORD>(std::size(name))) > 1 &&
                std::wstring_view{name} == expected_signer;
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

} // namespace Vulkan

#endif
