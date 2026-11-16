// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_metal/metal_pipeline_files.h"

namespace Metal {

namespace {

constexpr u32 StatesMagic = 0x53544c4d; // "MLTS"
constexpr u32 UsedMagic = 0x55544c4d;   // "MLTU"

struct StatesHeader {
    u32 magic;
    u32 version;
    u32 key_size;
    u32 desc_size;

    bool operator==(const StatesHeader&) const noexcept = default;
};

constexpr StatesHeader ExpectedHeader{
    .magic = StatesMagic,
    .version = PipelineFiles::StatesVersion,
    .key_size = sizeof(GraphicsPipelineKey),
    .desc_size = sizeof(RenderPipelineDesc),
};

/// A state as the file holds it: the record, then its hash.
constexpr size_t StoredSize = sizeof(PipelineFiles::StateRecord) + sizeof(u64);

[[nodiscard]] u64 StateHash(const PipelineFiles::StateRecord& record) {
    return Common::CityHash64(reinterpret_cast<const char*>(&record), sizeof(record));
}

void AppendState(const std::filesystem::path& path, const PipelineFiles::StateRecord& record,
                 u64 hash) {
    std::ofstream file(path, std::ios::binary | std::ios::ate | std::ios::app);
    if (!file.is_open()) {
        return;
    }
    if (file.tellp() == 0) {
        file.write(reinterpret_cast<const char*>(&ExpectedHeader), sizeof(ExpectedHeader));
    }
    file.write(reinterpret_cast<const char*>(&record), sizeof(record))
        .write(reinterpret_cast<const char*>(&hash), sizeof(hash));
}

void WriteUsed(const std::filesystem::path& path,
               const std::vector<std::pair<u64, u8>>& entries) {
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return;
        }
        const u32 count = static_cast<u32>(entries.size());
        file.write(reinterpret_cast<const char*>(&UsedMagic), sizeof(UsedMagic))
            .write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [hash, age] : entries) {
            file.write(reinterpret_cast<const char*>(&hash), sizeof(hash))
                .write(reinterpret_cast<const char*>(&age), sizeof(age));
        }
        if (!file.good()) {
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
}

} // Anonymous namespace

u32 PipelineFiles::Mode() {
    static const u32 mode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_PIPELINE_FILE");
        if (value == nullptr) {
            return 1U;
        }
        return value[0] == '0' ? 0U : value[0] == '2' ? 2U : 1U;
    }();
    return mode;
}

PipelineFiles::PipelineFiles() = default;

PipelineFiles::~PipelineFiles() {
    Wait();
}

void PipelineFiles::Open(const std::filesystem::path& directory) {
    if (IsOpen() || Mode() == 0) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    records_path = VideoCommon::VersionedPipelineCachePath(directory, "metal", RecordsVersion);
    states_path = directory / "metal_states.bin";
    used_path = directory / "metal_used.bin";
    boot_path = directory / "metal_boot.pending";

    std::ifstream file(used_path, std::ios::binary);
    u32 magic{};
    u32 count{};
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic))
        .read(reinterpret_cast<char*>(&count), sizeof(count));
    if (file.good() && magic == UsedMagic) {
        has_list = true;
        for (u32 index = 0; index < count; ++index) {
            u64 hash{};
            u8 age{};
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash))
                .read(reinterpret_cast<char*>(&age), sizeof(age));
            if (!file.good()) {
                break;
            }
            listed.insert_or_assign(hash, age);
        }
    }
    writer.emplace(size_t{1}, "MetalPipelineFiles");
}

bool PipelineFiles::BeginBoot() {
    std::error_code ec;
    if (std::filesystem::exists(boot_path, ec)) {
        std::filesystem::remove(boot_path, ec);
        return false;
    }
    std::ofstream marker{boot_path, std::ios::binary | std::ios::trunc};
    return true;
}

void PipelineFiles::EndBoot() {
    std::error_code ec;
    std::filesystem::remove(boot_path, ec);
}

std::vector<PipelineFiles::StateRecord> PipelineFiles::LoadStates() {
    std::vector<StateRecord> records;
    std::ifstream file(states_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return records;
    }
    const u64 size = static_cast<u64>(file.tellg());
    file.seekg(0, std::ios::beg);
    StatesHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    std::error_code ec;
    if (!file.good() || header != ExpectedHeader) {
        file.close();
        std::filesystem::remove(states_path, ec);
        LOG_INFO(Render_Metal, "Metal: render states file of another layout removed");
        return records;
    }
    const u64 count = (size - sizeof(header)) / StoredSize;
    records.reserve(count);
    size_t damaged = 0;
    for (u64 index = 0; index < count; ++index) {
        StateRecord record;
        u64 hash{};
        file.read(reinterpret_cast<char*>(&record), sizeof(record))
            .read(reinterpret_cast<char*>(&hash), sizeof(hash));
        if (!file.good()) {
            break;
        }
        if (hash != StateHash(record)) {
            ++damaged;
            continue;
        }
        if (known_states.insert(hash).second) {
            records.push_back(record);
        }
    }
    file.close();
    // A record cut short by a kill would misplace everything appended after it.
    const u64 whole = sizeof(header) + count * StoredSize;
    if (whole != size) {
        std::filesystem::resize_file(states_path, whole, ec);
    }
    if (damaged != 0) {
        LOG_WARNING(Render_Metal, "Metal: {} damaged render state records skipped", damaged);
    }
    return records;
}

void PipelineFiles::RecordState(const GraphicsPipelineKey& key, const RenderPipelineDesc& desc) {
    if (!IsOpen()) {
        return;
    }
    StateRecord record;
    std::memset(static_cast<void*>(&record), 0, sizeof(record));
    record.key = key;
    std::memcpy(static_cast<void*>(&record.desc), &desc, sizeof(desc));
    record.desc.vertex_function = nullptr;
    record.desc.fragment_function = nullptr;
    const u64 hash = StateHash(record);
    if (!known_states.insert(hash).second) {
        return;
    }
    Queue([path = states_path, record, hash] { AppendState(path, record, hash); });
}

bool PipelineFiles::BuildAtBoot(u64 key_hash) const {
    if (Mode() == 2 || !has_list) {
        return true;
    }
    const auto it = listed.find(key_hash);
    return it != listed.end() && it->second < MaxBootAge;
}

void PipelineFiles::NoteUsed(u64 key_hash) {
    used_changed = used.insert(key_hash).second || used_changed;
}

void PipelineFiles::SaveUsed() {
    if (!IsOpen() || !used_changed || used.size() < MinimumUsed) {
        return;
    }
    used_changed = false;
    std::vector<std::pair<u64, u8>> entries;
    entries.reserve(used.size() + listed.size());
    for (const u64 hash : used) {
        entries.emplace_back(hash, u8{0});
    }
    for (const auto& [hash, age] : listed) {
        if (!used.contains(hash) && age + 1 < MaxListedAge) {
            entries.emplace_back(hash, static_cast<u8>(age + 1));
        }
    }
    Queue([path = used_path, entries = std::move(entries)] { WriteUsed(path, entries); });
}

void PipelineFiles::Wait() {
    if (writer) {
        writer->WaitForRequests();
    }
}

} // namespace Metal
