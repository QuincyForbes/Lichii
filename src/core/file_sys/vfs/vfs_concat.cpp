// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <utility>

#include "common/assert.h"
#include "core/file_sys/vfs/vfs_concat.h"

namespace FileSys {

ConcatenatedVfsFile::ConcatenatedVfsFile(std::string&& name_,
                                         ConcatenationMap&& concatenation_map_, u8 filler_byte_)
    : concatenation_map(std::move(concatenation_map_)), name(std::move(name_)),
      filler_byte(filler_byte_) {
    DEBUG_ASSERT(this->VerifyContinuity());
}

bool ConcatenatedVfsFile::VerifyContinuity() const {
    u64 last_offset = 0;
    for (const auto& entry : concatenation_map) {
        if (entry.offset < last_offset) {
            return false;
        }

        last_offset = entry.offset + entry.size;
    }

    return true;
}

ConcatenatedVfsFile::~ConcatenatedVfsFile() = default;

VirtualFile ConcatenatedVfsFile::MakeConcatenatedFile(std::string&& name,
                                                      std::vector<VirtualFile>&& files) {
    // Fold trivial cases.
    if (files.empty()) {
        return nullptr;
    }
    if (files.size() == 1) {
        return files.front();
    }

    // Make the concatenation map from the input.
    std::vector<ConcatenationEntry> concatenation_map;
    concatenation_map.reserve(files.size());
    u64 last_offset = 0;

    for (auto& file : files) {
        const u64 size = file->GetSize();

        concatenation_map.emplace_back(ConcatenationEntry{
            .offset = last_offset,
            .size = size,
            .file = std::move(file),
        });

        last_offset += size;
    }

    return VirtualFile(new ConcatenatedVfsFile(std::move(name), std::move(concatenation_map)));
}

VirtualFile ConcatenatedVfsFile::MakeConcatenatedFile(
    u8 filler_byte, std::string&& name, std::vector<std::pair<u64, VirtualFile>>&& files) {
    // Fold trivial cases.
    if (files.empty()) {
        return nullptr;
    }
    if (files.size() == 1 && files.begin()->first == 0) {
        return files.begin()->second;
    }

    // Make the concatenation map from the input without allocating gap StaticVfsFile objects.
    std::vector<ConcatenationEntry> concatenation_map;
    concatenation_map.reserve(files.size());

    for (auto& [offset, file] : files) {
        const u64 size = file->GetSize();

        concatenation_map.emplace_back(ConcatenationEntry{
            .offset = offset,
            .size = size,
            .file = std::move(file),
        });
    }

    return VirtualFile(
        new ConcatenatedVfsFile(std::move(name), std::move(concatenation_map), filler_byte));
}

std::string ConcatenatedVfsFile::GetName() const {
    if (concatenation_map.empty()) {
        return "";
    }
    if (!name.empty()) {
        return name;
    }
    return concatenation_map.front().file->GetName();
}

std::size_t ConcatenatedVfsFile::GetSize() const {
    if (concatenation_map.empty()) {
        return 0;
    }
    return concatenation_map.back().offset + concatenation_map.back().size;
}

bool ConcatenatedVfsFile::Resize(std::size_t new_size) {
    return false;
}

VirtualDir ConcatenatedVfsFile::GetContainingDirectory() const {
    if (concatenation_map.empty()) {
        return nullptr;
    }
    return concatenation_map.front().file->GetContainingDirectory();
}

bool ConcatenatedVfsFile::IsWritable() const {
    return false;
}

bool ConcatenatedVfsFile::IsReadable() const {
    return true;
}

std::size_t ConcatenatedVfsFile::Read(u8* data, std::size_t length, std::size_t offset) const {
    if (length == 0 || concatenation_map.empty()) {
        return 0;
    }

    const u64 total_size = concatenation_map.back().offset + concatenation_map.back().size;
    if (offset >= total_size) {
        return 0;
    }

    u64 cur_offset = offset;
    u64 cur_length = std::min<u64>(length, total_size - offset);

    const ConcatenationEntry key{
        .offset = cur_offset,
        .size = 0,
        .file = nullptr,
    };

    auto it = std::upper_bound(concatenation_map.begin(), concatenation_map.end(), key);
    if (it != concatenation_map.begin()) {
        const auto prev = std::prev(it);
        if (cur_offset < prev->offset + prev->size) {
            it = prev;
        }
    }

    while (cur_length > 0 && it != concatenation_map.end()) {
        if (cur_offset < it->offset) {
            const u64 gap_size = std::min<u64>(cur_length, it->offset - cur_offset);
            std::memset(data + (cur_offset - offset), filler_byte, gap_size);
            cur_offset += gap_size;
            cur_length -= gap_size;
            if (cur_length == 0) {
                break;
            }
        }

        if (it->size == 0) {
            ++it;
            continue;
        }

        // Read the file at this position.
        const u64 file_seek = cur_offset - it->offset;
        const u64 intended_read_size = std::min<u64>(cur_length, it->size - file_seek);
        const u64 actual_read_size =
            it->file->Read(data + (cur_offset - offset), intended_read_size, file_seek);

        // Update tracking.
        cur_offset += actual_read_size;
        cur_length -= actual_read_size;
        ++it;

        // If we encountered a short read, we're done.
        if (actual_read_size < intended_read_size) {
            break;
        }
    }

    return cur_offset - offset;
}

std::size_t ConcatenatedVfsFile::Write(const u8* data, std::size_t length, std::size_t offset) {
    return 0;
}

bool ConcatenatedVfsFile::Rename(std::string_view new_name) {
    return false;
}

} // namespace FileSys
