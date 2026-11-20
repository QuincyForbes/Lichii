// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_set>

#include "common/cityhash.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "video_core/shader_warm/warm_records.h"

namespace VideoCommon::ShaderWarm {
namespace {

constexpr u64 AddressMapMagic = 0x50414d4e4544454EULL;
constexpr u64 HashXor = 0x4F554E4345000002ULL;
constexpr size_t EnvHeaderSize = 6 * 8 + 7 * 4;
constexpr u32 ComputeStage = 5;

template <typename T>
void Put(std::vector<u8>& out, T value) {
    const size_t at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &value, sizeof(T));
}

void PutBytes(std::vector<u8>& out, std::span<const u8> bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

template <typename T>
T Get(std::span<const u8> data, size_t offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

/// The GenericEnvironment::Serialize header: six u64 counts and seven u32 fields.
void PutEnvHeader(std::vector<u8>& out, u64 code_size, u64 cbuf_count, u32 local_memory,
                  u32 texture_bound, u32 start_address, u32 stage, u32 read_lowest = 0) {
    Put<u64>(out, code_size);
    Put<u64>(out, 0); // texture types
    Put<u64>(out, 0); // texture pixel formats
    Put<u64>(out, cbuf_count);
    Put<u64>(out, 0); // storage pointer types
    Put<u64>(out, 0); // cbuf replacements
    Put<u32>(out, local_memory);
    Put<u32>(out, texture_bound);
    Put<u32>(out, start_address);
    Put<u32>(out, read_lowest);
    Put<u32>(out, read_lowest + static_cast<u32>(code_size - 8)); // read highest
    Put<u32>(out, 1);                                             // viewport transform state
    Put<u32>(out, stage);
}

struct PairHash {
    u128 a{};
    u128 b{};
    bool operator==(const PairHash&) const = default;
};

struct PairHasher {
    size_t operator()(const PairHash& h) const noexcept {
        return static_cast<size_t>(h.a[0] ^ (h.b[0] * 0x9E3779B97F4A7C15ULL));
    }
};

struct U128Hasher {
    size_t operator()(const u128& h) const noexcept {
        return static_cast<size_t>(h[0]);
    }
};

u128 Hash128(std::span<const u8> bytes) {
    return Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

size_t TailSize(u32 stage) {
    if (stage == 5) {
        return 16;
    }
    return 80 + (stage == 3 ? 32 : 0) + (stage == 7 || stage == 8 ? 92 : 0);
}

constexpr u64 MaxwellSelfBranchA = 0xE2400FFFFF87000FULL;
constexpr u64 MaxwellSelfBranchB = 0xE2400FFFFF07000FULL;

std::optional<size_t> FindMaxwellEnd(std::span<const u8> bytes, size_t inst_offset,
                                     size_t align_bias) {
    if (bytes.size() <= inst_offset || (bytes.size() - inst_offset) % sizeof(u64) != 0) {
        return std::nullopt;
    }
    const size_t limit = std::min<size_t>(bytes.size(), inst_offset + 0x100000);
    for (size_t offset = inst_offset; offset + sizeof(u64) <= limit; offset += sizeof(u64)) {
        if ((align_bias + offset) % 32 == 0) {
            continue; // Maxwell scheduling word
        }
        const u64 inst = Get<u64>(bytes, offset);
        if (inst == MaxwellSelfBranchA || inst == MaxwellSelfBranchB) {
            return offset + sizeof(u64);
        }
    }
    return std::nullopt;
}

} // Anonymous namespace

std::optional<size_t> CaptureProgramPrefixSize(std::span<const u8> program, bool graphics,
                                               bool maxwell) {
    if (maxwell) {
        if (graphics) {
            if (program.size() <= 0x80 || Get<u32>(program, 0) != 0x12345678) {
                return std::nullopt;
            }
            const u32 type = (Get<u32>(program, 0x30) >> 10) & 15;
            if (type < 1 || type > 5) {
                return std::nullopt;
            }
            const auto end = FindMaxwellEnd(program.subspan(0x30), 0x50, 0x30);
            return end ? std::optional<size_t>{0x30 + *end} : std::nullopt;
        }
        const size_t start =
            program.size() > 0x100 && Get<u32>(program, 0) == 0x12345679 ? 0x100 : 0;
        if (program.size() <= start) {
            return std::nullopt;
        }
        const auto end = FindMaxwellEnd(program.subspan(start), 0, 0);
        return end ? std::optional<size_t>{start + *end} : std::nullopt;
    }
    using Shader::Ampere::Instruction;
    const size_t header_size = graphics ? 128 : 0;
    if (program.size() <= header_size || (program.size() - header_size) % sizeof(Instruction)) {
        return std::nullopt;
    }
    if (graphics) {
        const u32 type = (Get<u32>(program, 0) >> 10) & 15;
        if (type < 1 || type > 5) {
            return std::nullopt;
        }
    }
    const size_t count =
        std::min<size_t>((program.size() - header_size) / sizeof(Instruction),
                         Shader::Ampere::ProgramCaptureState::MaximumInstructions);
    Shader::Ampere::ProgramCaptureState capture;
    size_t captured_count = 0;
    for (size_t index = 0; index < count; ++index) {
        Instruction inst;
        std::memcpy(&inst, program.data() + header_size + index * sizeof(Instruction),
                    sizeof(Instruction));
        if (inst.Bits(0, 12) == 0x95c) {
            return std::nullopt;
        }
        const auto result = capture.Observe(inst, static_cast<u32>(index));
        if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            return std::nullopt;
        }
        if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            captured_count = index + 1;
            break;
        }
    }
    if (captured_count == 0) {
        return std::nullopt;
    }
    return header_size + captured_count * sizeof(Instruction);
}

std::optional<std::string> LowerProgram(std::span<const u8> program, bool graphics,
                                        LoweredCode& out) {
    using Shader::Ampere::Instruction;
    const size_t header_size = graphics ? 128 : 0;
    if (program.size() <= header_size || (program.size() - header_size) % sizeof(Instruction)) {
        return "Native input must contain 1..65536 complete 16-byte instructions";
    }
    if (graphics) {
        const u32 type = (Get<u32>(program, 0) >> 10) & 15;
        if (type < 1 || type > 5) {
            return "Invalid native graphics header stage";
        }
    }
    const size_t max_count =
        std::min<size_t>((program.size() - header_size) / sizeof(Instruction),
                         Shader::Ampere::ProgramCaptureState::MaximumInstructions);
    std::vector<Instruction> native(max_count);
    std::memcpy(native.data(), program.data() + header_size, native.size() * sizeof(Instruction));
    Shader::Ampere::ProgramCaptureState capture;
    bool complete = false;
    for (u32 index = 0; index < native.size(); ++index) {
        const auto result = capture.Observe(native[index], index);
        if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            return "Native allocation has an out-of-range branch target";
        }
        if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            native.resize(index + 1);
            complete = true;
            break;
        }
    }
    if (!complete) {
        return "Native allocation lacks a complete runtime capture";
    }
    for (const auto& instruction : native) {
        if (instruction.Bits(0, 12) == 0x95c) {
            return "Trap guards require captured constant buffers";
        }
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(native, false, true, {},
                                                        Shader::Ampere::TessellationStage::None);
    if (!lowered) {
        return std::string(lowered.error);
    }
    const auto& offsets = lowered.source_instruction_offsets;
    if (lowered.source_size != native.size() * sizeof(Instruction) ||
        offsets.size() != native.size()) {
        return "Cannot encode the lowering as an override record";
    }
    std::vector<u64> words(10, 0);
    if (graphics) {
        std::memcpy(words.data(), program.data(), 80);
    }
    words.reserve(10 + lowered.instructions.size() * 5 / 4 + 4 + offsets.size());
    for (const u64 instruction : lowered.instructions) {
        if (words.size() % 4 == 0) {
            words.push_back(0); // Maxwell scheduling word
        }
        words.push_back(instruction);
    }
    words.push_back(0);
    if (!offsets.empty()) {
        // AppendNativeAddressMap: lowered index -> byte address inside the code.
        for (const u32 i : offsets) {
            words.push_back(0x50 + u64{i} * 8 + ((u64{i} + 1) / 3) * 8);
        }
        words.push_back(offsets.size());
        words.push_back(AddressMapMagic);
    }
    out.code.resize(words.size() * sizeof(u64));
    std::memcpy(out.code.data(), words.data(), out.code.size());
    out.texture_buffer = lowered.required_texture_buffer;
    out.stage_hash = StageHash(out.code);
    out.maxwell = false;
    return std::nullopt;
}

std::optional<std::string> PrepareMaxwellProgram(std::span<const u8> program, bool graphics,
                                                 LoweredCode& out) {
    if (graphics) {
        if (program.size() <= 0x80 || Get<u32>(program, 0) != 0x12345678) {
            return "Invalid Maxwell graphics header";
        }
        const u32 type = (Get<u32>(program, 0x30) >> 10) & 15;
        if (type < 1 || type > 5) {
            return "Invalid Maxwell SPH stage";
        }
        const auto sph_and_code = program.subspan(0x30);
        const auto end = FindMaxwellEnd(sph_and_code, 0x50, 0x30);
        if (!end || *end <= sizeof(u64)) {
            return "Maxwell graphics shader lacks a self-branch terminator";
        }
        out.code.assign(sph_and_code.begin(), sph_and_code.begin() + static_cast<ptrdiff_t>(*end));
        out.stage_hash =
            Common::CityHash64(reinterpret_cast<const char*>(out.code.data()), *end - sizeof(u64));
        out.texture_buffer = 2;
        out.maxwell = true;
        return std::nullopt;
    }
    const size_t start =
        program.size() > 0x100 && Get<u32>(program, 0) == 0x12345679 ? 0x100 : 0;
    if (program.size() <= start) {
        return "Invalid Maxwell compute code";
    }
    const auto inst_bytes = program.subspan(start);
    const auto end = FindMaxwellEnd(inst_bytes, 0, 0);
    if (!end || *end <= sizeof(u64)) {
        return "Maxwell compute shader lacks a self-branch terminator";
    }
    out.code.assign(inst_bytes.begin(), inst_bytes.begin() + static_cast<ptrdiff_t>(*end));
    out.stage_hash =
        Common::CityHash64(reinterpret_cast<const char*>(out.code.data()), *end - sizeof(u64));
    out.texture_buffer = 2;
    out.maxwell = true;
    return std::nullopt;
}

u64 StageHash(std::span<const u8> code) {
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()), code.size()) ^ HashXor;
}

std::optional<ComputeControl> ReadComputeControl(std::span<const u8> control, bool maxwell) {
    if (maxwell) {
        if (control.size() < 0x760 || Get<u32>(control, 0) != 0x98761234) {
            return std::nullopt;
        }
        ComputeControl result;
        for (size_t i = 0; i < 3; ++i) {
            result.workgroup[i] = Get<u32>(control, 0x748 + i * 4);
        }
        result.shared_memory =
            static_cast<u32>((u64{Get<u32>(control, 0x754)} + 255) / 256 * 256);
        const u32 local_pos = Get<u32>(control, 0x758);
        const u32 local_crs = Get<u32>(control, 0x75c);
        if (local_pos <= (1U << 20) && local_crs <= (1U << 20)) {
            result.local_memory = local_pos + local_crs;
        }
        return result;
    }
    if (control.size() < 108) {
        return std::nullopt;
    }
    ComputeControl result;
    for (size_t i = 0; i < 3; ++i) {
        result.workgroup[i] = Get<u32>(control, 88 + i * 4);
    }
    result.shared_memory =
        static_cast<u32>((u64{Get<u32>(control, 100)} + 255) / 256 * 256);
    result.local_memory = Get<u32>(control, 104);
    return result;
}

std::span<const u8> ExtractBank1(std::span<const u8> program, std::span<const u8> control,
                                 bool maxwell, size_t prefix_size) {
    if (maxwell) {
        return {};
    }
    if (control.empty()) {
        if (program.size() < prefix_size + 256) {
            return {};
        }
        return program.last(256);
    }
    if (control.size() < 72 || Get<u32>(control, 0) != 0x98761234 ||
        Get<u32>(control, 4) != 0xc9) {
        if (program.size() < prefix_size + 256) {
            return {};
        }
        return program.last(256);
    }
    const size_t inst_alloc = Get<u32>(control, 64);
    const size_t total_alloc = Get<u32>(control, 68);
    if (total_alloc <= inst_alloc || inst_alloc >= program.size()) {
        return {};
    }
    const size_t end = std::min<size_t>(total_alloc, program.size());
    const size_t min_start = std::max<size_t>(inst_alloc, prefix_size);
    if (end <= min_start) {
        return {};
    }
    const size_t bank1_size = std::min<size_t>(256, end - min_start);
    return program.subspan(end - bank1_size, bank1_size);
}

std::vector<u8> ComputeEntry(const LoweredCode& lowered, const ComputeControl& control,
                             ComputeKey& key) {
    std::vector<u8> out;
    out.reserve(4 + EnvHeaderSize + lowered.code.size() + 16 + key.size());
    Put<u32>(out, 1);
    // Like InitializeNativeCode: the program's own bank when it names one.
    const u32 texture_bound = lowered.texture_buffer != ~0U ? lowered.texture_buffer : 0;
    const u32 start_address = lowered.maxwell ? 0U : 80U;
    PutEnvHeader(out, lowered.code.size(), 0, control.local_memory, texture_bound, start_address,
                 ComputeStage, 0);
    PutBytes(out, lowered.code);
    for (const u32 size : control.workgroup) {
        Put<u32>(out, size);
    }
    Put<u32>(out, control.shared_memory);
    const u64 unique_hash = lowered.stage_hash != 0 ? lowered.stage_hash : StageHash(lowered.code);
    std::vector<u8> key_bytes;
    key_bytes.reserve(key.size());
    Put<u64>(key_bytes, unique_hash);
    Put<u32>(key_bytes, control.shared_memory);
    for (const u32 size : control.workgroup) {
        Put<u32>(key_bytes, size);
    }
    std::memcpy(key.data(), key_bytes.data(), key.size());
    PutBytes(out, key_bytes);
    return out;
}

std::vector<u8> GraphicsEnvFromParts(u32 stage, const LoweredCode& lowered,
                                     std::span<const u8> sph, std::span<const u8> bank1) {
    const u32 bound = lowered.texture_buffer != ~0U ? lowered.texture_buffer
                                                     : (stage == 0 ? 0U : 2U);
    const size_t total_words = (bank1.size() & ~size_t{3}) / 4;
    // GuessMissing returns 0 for missing bank 1 words, so only non-zero words need to be stored.
    size_t cbuf_count = 0;
    for (size_t i = 0; i < total_words; ++i) {
        if (Get<u32>(bank1, i * 4) != 0) {
            ++cbuf_count;
        }
    }
    const u32 w1 = Get<u32>(sph, 4), w2 = Get<u32>(sph, 8), w3 = Get<u32>(sph, 12);
    const u64 local_memory =
        ((u64{w1} & 0xFFFFFF) | ((u64{w2} & 0xFFFFFF) << 24)) + (u64{w3} & 0xFFFFFF);
    const u32 start_address = lowered.maxwell ? 0x30U : 0U;
    std::vector<u8> out;
    out.reserve(EnvHeaderSize + lowered.code.size() + cbuf_count * 12 + 80);
    PutEnvHeader(out, lowered.code.size(), cbuf_count, static_cast<u32>(local_memory), bound,
                 start_address, stage, start_address);
    PutBytes(out, lowered.code);
    for (size_t i = 0; i < total_words; ++i) {
        const u32 word = Get<u32>(bank1, i * 4);
        if (word != 0) {
            Put<u64>(out, (u64{1} << 32) | (i * 4));
            Put<u32>(out, word);
        }
    }
    PutBytes(out, sph.first(80));
    return out;
}

std::vector<u8> GraphicsEnv(u32 stage, const LoweredCode& lowered, std::span<const u8> program) {
    const auto sph = lowered.maxwell ? std::span<const u8>{lowered.code}.first(80)
                                     : program.first(80);
    const auto prefix_size = CaptureProgramPrefixSize(program, true, lowered.maxwell);
    const auto bank1 = prefix_size ? ExtractBank1(program, {}, lowered.maxwell, *prefix_size)
                                   : std::span<const u8>{};
    return GraphicsEnvFromParts(stage, lowered, sph, bank1);
}

GraphicsKey GraphicsEntryKey(u32 stage, const GraphicsKey& template_key, u64 stage_hash,
                             u64 attribute_types) {
    GraphicsKey key = template_key;
    const size_t slot = stage == 0 ? 1 : 5;
    std::memcpy(key.data() + slot * 8, &stage_hash, sizeof(stage_hash));
    if (stage == 0) {
        std::memcpy(key.data() + AttributeTypesOffset, &attribute_types, sizeof(u64));
    }
    return key;
}

RuntimeStats ReadRuntimeStats(const std::vector<std::filesystem::path>& caches) {
    struct Occurrence {
        u32 stage;
        u64 hash;
        const u8* key;
    };
    std::vector<std::vector<u8>> files;
    std::vector<Occurrence> occurrences;
    std::unordered_set<PairHash, PairHasher> seen;
    for (const auto& path : caches) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            continue;
        }
        const auto end_pos = file.tellg();
        if (end_pos < 12) {
            continue;
        }
        if constexpr (sizeof(u64) > sizeof(size_t)) {
            if (static_cast<u64>(end_pos) > static_cast<u64>(std::numeric_limits<size_t>::max())) {
                continue;
            }
        }
        const auto size = static_cast<size_t>(end_pos);
        file.seekg(0);
        std::vector<u8> data(size);
        if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
            continue;
        }
        if (std::memcmp(data.data(), "yuzucach", 8) != 0 ||
            !IsReadableRuntimeCacheVersion(Get<u32>(data, 8))) {
            continue;
        }
        const std::span<const u8> bytes{data};
        size_t cur = 12;
        struct Env {
            u32 stage;
            size_t begin;
            size_t end;
        };
        std::vector<Env> envs;
        while (cur + 4 <= size) {
            const u32 count = Get<u32>(bytes, cur);
            cur += 4;
            envs.clear();
            bool truncated = count == 0;
            for (u32 i = 0; i < count && !truncated; ++i) {
                if (cur + EnvHeaderSize > size) {
                    truncated = true;
                    break;
                }
                std::array<u64, 6> counts;
                std::memcpy(counts.data(), data.data() + cur, sizeof(counts));
                const u32 stage = Get<u32>(bytes, cur + 48 + 6 * 4);
                const u64 remaining = size - cur - EnvHeaderSize;
                constexpr std::array<u64, 6> Strides{1, 8, 8, 12, 9, 12};
                u64 body = TailSize(stage);
                if (body > remaining) {
                    truncated = true;
                    break;
                }
                for (size_t k = 0; k < counts.size(); ++k) {
                    if (counts[k] > (remaining - body) / Strides[k]) {
                        truncated = true;
                        break;
                    }
                    body += counts[k] * Strides[k];
                }
                if (truncated) {
                    break;
                }
                const size_t begin = cur;
                cur += EnvHeaderSize + static_cast<size_t>(body);
                envs.push_back({stage, begin, cur});
            }
            if (truncated) {
                break;
            }
            const size_t key_size = envs[0].stage == ComputeStage ? 24 : GraphicsKeySize;
            if (key_size > size - cur) {
                break;
            }
            const u8* const key = data.data() + cur;
            cur += key_size;
            if (envs[0].stage == ComputeStage) {
                continue;
            }
            std::array<u64, 6> hashes;
            std::memcpy(hashes.data(), key, sizeof(hashes));
            size_t env_index = 0;
            for (size_t slot = 0; slot < 6 && env_index < envs.size(); ++slot) {
                if (hashes[slot] == 0) {
                    continue;
                }
                const Env& env = envs[env_index++];
                if (env.stage != 0 && env.stage != 4) {
                    continue;
                }
                const PairHash id{Hash128(bytes.subspan(env.begin, env.end - env.begin)),
                                  Hash128({key, key_size})};
                if (seen.insert(id).second) {
                    occurrences.push_back({env.stage, hashes[slot], key});
                }
            }
        }
        files.push_back(std::move(data));
    }

    RuntimeStats stats;
    stats.occurrences = occurrences.size();
    for (size_t index = 0; index < 2; ++index) {
        const u32 stage = index == 0 ? 0 : 4;
        struct Count {
            size_t count;
            size_t first;
        };
        std::unordered_map<u128, Count, U128Hasher> counts;
        GraphicsKey normalized;
        for (size_t i = 0; i < occurrences.size(); ++i) {
            if (occurrences[i].stage != stage) {
                continue;
            }
            std::memcpy(normalized.data(), occurrences[i].key, normalized.size());
            std::fill(normalized.begin() + 56, normalized.begin() + 64, u8{0});
            ++counts.try_emplace(Hash128(normalized), Count{0, i}).first->second.count;
        }
        // Counter.most_common: highest count, the first seen among equals.
        const Count* best = nullptr;
        for (const auto& [hash, count] : counts) {
            if (best == nullptr || count.count > best->count ||
                (count.count == best->count && count.first < best->first)) {
                best = &count;
            }
        }
        if (best != nullptr) {
            GraphicsKey key;
            std::memcpy(key.data(), occurrences[best->first].key, key.size());
            std::fill(key.begin() + 56, key.begin() + 64, u8{0});
            std::fill(key.begin(), key.begin() + 48, u8{0});
            stats.templates[index] = key;
        }
    }
    for (const auto& occurrence : occurrences) {
        if (occurrence.stage == 0) {
            u64 attributes;
            std::memcpy(&attributes, occurrence.key + AttributeTypesOffset, sizeof(attributes));
            stats.vertex_attributes[occurrence.hash].push_back(attributes);
        }
    }
    return stats;
}

} // namespace VideoCommon::ShaderWarm
