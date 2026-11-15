// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#include "common/console_capture.h"

namespace Common {

#ifdef _WIN32

void CaptureConsole(const std::string&) {}

std::string ConsoleCaptureTail(std::size_t) {
    return {};
}

#else

namespace {

// Fixed storage: read by a crash handler.
std::array<char, 1024> capture_path{};
long capture_start{};

} // Anonymous namespace

void CaptureConsole(const std::string& path) {
    if (path.empty() || path.size() >= capture_path.size()) {
        return;
    }
    // What earlier launches left in the file is not this process's.
    if (std::FILE* const file = std::fopen(path.c_str(), "rb")) {
        if (std::fseek(file, 0, SEEK_END) == 0) {
            capture_start = (std::max)(std::ftell(file), 0L);
        }
        std::fclose(file);
    }
    const bool has_stdout = std::freopen(path.c_str(), "a", stdout) != nullptr;
    const bool has_stderr = std::freopen(path.c_str(), "a", stderr) != nullptr;
    if (!has_stdout && !has_stderr) {
        return;
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::memcpy(capture_path.data(), path.data(), path.size());
}

std::string ConsoleCaptureTail(std::size_t max_bytes) {
    if (capture_path[0] == '\0' || max_bytes == 0) {
        return {};
    }
    // stderr is unbuffered; stdout is left alone, its lock may be held by the thread that died.
    std::fflush(stderr);
    std::FILE* const file = std::fopen(capture_path.data(), "rb");
    if (file == nullptr) {
        return {};
    }
    std::string text;
    if (std::fseek(file, 0, SEEK_END) == 0) {
        const long end = std::ftell(file);
        const long wanted = static_cast<long>((std::min)(max_bytes, std::size_t{1} << 20));
        const long first = (std::max)(capture_start, end - wanted);
        if (end > first && std::fseek(file, first, SEEK_SET) == 0) {
            text.resize(static_cast<std::size_t>(end - first));
            text.resize(std::fread(text.data(), 1, text.size(), file));
        }
    }
    std::fclose(file);
    return text;
}

#endif

} // namespace Common
