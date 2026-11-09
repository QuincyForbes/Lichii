// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Minimal ZIP reader/writer on zlib for the iOS content screens (firmware, mods, save and log
// archives). Reads stored and deflated entries, with ZIP64 sizes and offsets; writes deflated
// entries without ZIP64 (each file and the archive must stay below 4 GiB).

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace LichiiZip {

/// Called with (done, total) bytes; return true to cancel.
using ProgressFn = std::function<bool(std::uint64_t, std::uint64_t)>;

enum class Result {
    Success,
    OpenFailed,
    NotAZip,
    Unsupported, // encrypted or an unknown compression method
    Corrupt,     // bad header, inflate error or CRC mismatch
    WriteFailed,
    TooLarge,
    Cancelled,
};

const char* ResultString(Result result);

/// Extracts every entry under `dest` (created when missing). Entries that would leave `dest`
/// (absolute paths, "..") are skipped.
Result Extract(const std::filesystem::path& zip, const std::filesystem::path& dest,
               const ProgressFn& progress = {});

struct Input {
    std::filesystem::path source; // a file or a directory (added recursively)
    std::string name;             // path inside the archive; a directory's entries go below it
};

/// Writes `zip` from the inputs. Missing inputs are skipped.
Result Create(const std::filesystem::path& zip, const std::vector<Input>& inputs,
              const ProgressFn& progress = {});

/// Names of the entries in the archive (directories end with '/'). Empty when unreadable.
std::vector<std::string> List(const std::filesystem::path& zip);

} // namespace LichiiZip
