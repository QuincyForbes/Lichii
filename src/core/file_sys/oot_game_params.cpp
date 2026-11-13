// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/file_sys/oot_game_params.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include <fmt/ranges.h>
#include <zstd.h>

#include "common/logging.h"
#include "common/settings.h"
#include "core/file_sys/texture_quality.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace FileSys {

namespace {

constexpr u8 BymlArray = 0xC0;
constexpr u8 BymlDict = 0xC1;
constexpr u8 BymlKeyTable = 0xC2;
constexpr u8 BymlBool = 0xD0;
constexpr u8 BymlInt = 0xD1;
constexpr u8 BymlFloat = 0xD2;
constexpr u8 BymlUint = 0xD3;

constexpr u64 MaxPackSize = 64 * 1024 * 1024;
constexpr u64 OcarinaTitleId = 0x0400486024DEC000;

constexpr std::string_view FoliagePack = "Foliage.pack.zs";
constexpr std::string_view DictionaryPack = "ZsDic.pack.zs";

u32 Load32(std::span<const u8> data, std::size_t offset) {
    u32 value{};
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

u16 Load16(std::span<const u8> data, std::size_t offset) {
    u16 value{};
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

bool EndsWith(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

} // namespace

BymlEditor::BymlEditor(std::span<u8> data_) : data{data_} {
    if (data.size() < 16 || data[0] != 'Y' || data[1] != 'B') {
        return;
    }
    const u16 version = Load16(data, 2);
    key_table = Load32(data, 4);
    root = Load32(data, 12);
    valid = version >= 2 && key_table != 0 && key_table + 4 <= data.size() &&
            data[key_table] == BymlKeyTable && root != 0 && root + 4 <= data.size() &&
            (data[root] == BymlDict || data[root] == BymlArray);
}

u32 BymlEditor::Read32(std::size_t offset) const {
    return offset + 4 <= data.size() ? Load32(data, offset) : 0;
}

u32 BymlEditor::Read24(std::size_t offset) const {
    if (offset + 3 > data.size()) {
        return 0;
    }
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16);
}

std::optional<u32> BymlEditor::KeyIndex(std::string_view key) const {
    const u32 count = Read24(key_table + 1);
    for (u32 i = 0; i < count; ++i) {
        const std::size_t start = key_table + std::size_t{Read32(key_table + 4 + 4 * i)};
        if (start + key.size() >= data.size()) {
            continue;
        }
        if (std::memcmp(data.data() + start, key.data(), key.size()) == 0 &&
            data[start + key.size()] == 0) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<BymlEditor::Slot> BymlEditor::Find(std::span<const std::string_view> path) const {
    if (!valid || path.size() == 0) {
        return std::nullopt;
    }
    std::size_t node = root;
    std::optional<Slot> slot;
    for (const std::string_view element : path) {
        if (slot) {
            if (slot->type != BymlDict && slot->type != BymlArray) {
                return std::nullopt;
            }
            node = Read32(slot->offset);
            if (node == 0 || node + 4 > data.size() || data[node] != slot->type) {
                return std::nullopt;
            }
        }
        const u32 count = Read24(node + 1);
        if (data[node] == BymlDict) {
            const auto key = KeyIndex(element);
            if (!key) {
                return std::nullopt;
            }
            slot.reset();
            for (u32 i = 0; i < count; ++i) {
                const std::size_t entry = node + 4 + 8 * std::size_t{i};
                if (entry + 8 > data.size()) {
                    return std::nullopt;
                }
                if (Read24(entry) == *key) {
                    slot = Slot{data[entry + 3], entry + 4};
                    break;
                }
            }
            if (!slot) {
                return std::nullopt;
            }
        } else {
            u32 index{};
            const auto [end, error] =
                std::from_chars(element.data(), element.data() + element.size(), index);
            if (error != std::errc{} || end != element.data() + element.size() || index >= count) {
                return std::nullopt;
            }
            const std::size_t values = node + 4 + ((std::size_t{count} + 3) & ~std::size_t{3});
            if (values + 4 * std::size_t{count} > data.size()) {
                return std::nullopt;
            }
            slot = Slot{data[node + 4 + index], values + 4 * std::size_t{index}};
        }
    }
    return slot;
}

bool BymlEditor::Claim(std::size_t offset) {
    if (std::ranges::find(written, offset) != written.end()) {
        return false;
    }
    written.push_back(offset);
    return true;
}

std::optional<float> BymlEditor::GetFloat(Path path) const {
    const auto slot = Find(path);
    if (!slot || slot->type != BymlFloat) {
        return std::nullopt;
    }
    float value{};
    const u32 bits = Read32(slot->offset);
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::optional<s64> BymlEditor::GetInt(Path path) const {
    const auto slot = Find(path);
    if (!slot) {
        return std::nullopt;
    }
    const u32 bits = Read32(slot->offset);
    if (slot->type == BymlInt) {
        return static_cast<s32>(bits);
    }
    if (slot->type == BymlUint) {
        return bits;
    }
    return std::nullopt;
}

std::optional<bool> BymlEditor::GetBool(Path path) const {
    const auto slot = Find(path);
    if (!slot || slot->type != BymlBool) {
        return std::nullopt;
    }
    return Read32(slot->offset) != 0;
}

bool BymlEditor::SetFloat(Path path, float value) {
    const auto slot = Find(path);
    if (!slot || slot->type != BymlFloat || !Claim(slot->offset)) {
        return false;
    }
    std::memcpy(data.data() + slot->offset, &value, sizeof(value));
    return true;
}

bool BymlEditor::SetInt(Path path, s64 value) {
    const auto slot = Find(path);
    if (!slot || (slot->type != BymlInt && slot->type != BymlUint)) {
        return false;
    }
    if (slot->type == BymlUint ? (value < 0 || value > 0xFFFFFFFF)
                               : (value < std::numeric_limits<s32>::min() ||
                                  value > std::numeric_limits<s32>::max())) {
        return false;
    }
    if (!Claim(slot->offset)) {
        return false;
    }
    const u32 bits = static_cast<u32>(value);
    std::memcpy(data.data() + slot->offset, &bits, sizeof(bits));
    return true;
}

bool BymlEditor::SetBool(Path path, bool value) {
    const auto slot = Find(path);
    if (!slot || slot->type != BymlBool || !Claim(slot->offset)) {
        return false;
    }
    const u32 bits = value ? 1 : 0;
    std::memcpy(data.data() + slot->offset, &bits, sizeof(bits));
    return true;
}

bool BymlEditor::SetFromText(std::span<const std::string> path, std::string_view text) {
    std::vector<std::string_view> keys(path.begin(), path.end());
    const auto slot = Find(std::span<const std::string_view>{keys});
    if (!slot) {
        return false;
    }
    const char* const first = text.data();
    const char* const last = text.data() + text.size();
    u32 bits{};
    if (slot->type == BymlBool) {
        if (text != "true" && text != "false") {
            return false;
        }
        bits = text == "true" ? 1 : 0;
    } else if (slot->type == BymlFloat) {
        // Floating-point from_chars needs iOS 26.
        const std::string copy{text};
        char* end{};
        const float value = std::strtof(copy.c_str(), &end);
        if (copy.empty() || end != copy.c_str() + copy.size()) {
            return false;
        }
        std::memcpy(&bits, &value, sizeof(bits));
    } else if (slot->type == BymlInt || slot->type == BymlUint) {
        s64 value{};
        const auto [end, error] = std::from_chars(first, last, value);
        if (error != std::errc{} || end != last ||
            (slot->type == BymlUint ? (value < 0 || value > 0xFFFFFFFF)
                                    : (value < std::numeric_limits<s32>::min() ||
                                       value > std::numeric_limits<s32>::max()))) {
            return false;
        }
        bits = static_cast<u32>(value);
    } else {
        return false;
    }
    if (!Claim(slot->offset)) {
        return false;
    }
    std::memcpy(data.data() + slot->offset, &bits, sizeof(bits));
    return true;
}

bool ForEachSarcMember(std::span<u8> data,
                       const std::function<void(std::string_view, std::span<u8>)>& visit) {
    if (data.size() < 0x14 || std::memcmp(data.data(), "SARC", 4) != 0 ||
        Load16(data, 6) != 0xFEFF) {
        return false;
    }
    const std::size_t header = Load16(data, 4);
    const std::size_t data_start = Load32(data, 0xC);
    if (header + 12 > data.size() || std::memcmp(data.data() + header, "SFAT", 4) != 0) {
        return false;
    }
    const std::size_t count = Load16(data, header + 6);
    const std::size_t nodes = header + 12;
    const std::size_t names_header = nodes + 16 * count;
    if (names_header + 8 > data.size() ||
        std::memcmp(data.data() + names_header, "SFNT", 4) != 0) {
        return false;
    }
    const std::size_t names = names_header + 8;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t node = nodes + 16 * i;
        const u32 attributes = Load32(data, node + 4);
        const std::size_t begin = data_start + Load32(data, node + 8);
        const std::size_t end = data_start + Load32(data, node + 12);
        if ((attributes >> 24) == 0 || begin > end || end > data.size()) {
            continue;
        }
        const std::size_t name = names + std::size_t{attributes & 0xFFFF} * 4;
        if (name >= data.size()) {
            continue;
        }
        const auto* first = reinterpret_cast<const char*>(data.data() + name);
        const std::size_t length =
            static_cast<std::size_t>(std::find(first, first + (data.size() - name), '\0') - first);
        visit(std::string_view{first, length}, data.subspan(begin, end - begin));
    }
    return true;
}

OotGameParamEdits EditFoliagePack(std::span<u8> sarc, u32 level) {
    OotGameParamEdits edits;
    if (level == 0) {
        return edits;
    }
    const float density_scale = level == 1 ? 0.5f : 0.33f;
    constexpr float distance_scale = 0.5f;
    ForEachSarcMember(sarc, [&](std::string_view name, std::span<u8> member) {
        if (name.find("Foliage/ResFoliageScene/") == std::string_view::npos ||
            !EndsWith(name, ".bgyml")) {
            return;
        }
        BymlEditor byml{member};
        if (!byml.IsValid()) {
            return;
        }
        u32 values = 0;
        const auto count = [&values](bool written) { values += written ? 1 : 0; };
        if (const auto density = byml.GetFloat({"TurfDensity"}); density && *density > 1.0f) {
            count(byml.SetFloat({"TurfDensity"}, std::max(1.0f, *density * density_scale)));
        }
        if (const auto blades = byml.GetInt({"TurfMinBlades"}); blades && *blades > 1) {
            const auto scaled = static_cast<s64>(std::lround(static_cast<float>(*blades) *
                                                             density_scale));
            count(byml.SetInt({"TurfMinBlades"}, std::max<s64>(1, scaled)));
        }
        for (const std::string_view lod : {"LodSetting", "LodSetting_hh"}) {
            for (const std::string_view key :
                 {"Dist_LowTree", "Dist_LowTreeShadow", "Dist_LowTreeShootLeaf", "Dist_Ornament",
                  "Dist_Undergrowth0", "Dist_Undergrowth1"}) {
                if (const auto distance = byml.GetFloat({lod, key})) {
                    count(byml.SetFloat({lod, key}, *distance * distance_scale));
                }
            }
        }
        if (level >= 2) {
            for (const std::string_view key : {"IsEnableUndergrowth8", "IsEnableMoss"}) {
                if (byml.GetBool({key}).value_or(false)) {
                    count(byml.SetBool({key}, false));
                }
            }
            if (byml.GetBool({"DisableTurfShdw"}) == false) {
                count(byml.SetBool({"DisableTurfShdw"}, true));
            }
        }
        if (values != 0) {
            ++edits.foliage_scenes;
            edits.foliage_values += values;
        }
    });
    return edits;
}


namespace {

struct DCtxDeleter {
    void operator()(ZSTD_DCtx* dctx) const {
        ZSTD_freeDCtx(dctx);
    }
};

std::optional<std::vector<u8>> Decompress(std::span<const u8> file, std::span<const u8> dict) {
    const unsigned long long size = ZSTD_getFrameContentSize(file.data(), file.size());
    if (size == ZSTD_CONTENTSIZE_UNKNOWN || size == ZSTD_CONTENTSIZE_ERROR || size > MaxPackSize) {
        return std::nullopt;
    }
    const std::unique_ptr<ZSTD_DCtx, DCtxDeleter> dctx{ZSTD_createDCtx()};
    if (!dctx) {
        return std::nullopt;
    }
    std::vector<u8> out(static_cast<std::size_t>(size));
    const std::size_t result = ZSTD_decompress_usingDict(dctx.get(), out.data(), out.size(),
                                                         file.data(), file.size(), dict.data(),
                                                         dict.size());
    if (ZSTD_isError(result) || result != out.size()) {
        return std::nullopt;
    }
    return out;
}

/// The dictionary the game compresses its packs with, from Pack/ZsDic.pack.zs.
std::vector<u8> PackDictionary(const VirtualDir& pack_dir) {
    const auto file = pack_dir->GetFile(DictionaryPack);
    if (file == nullptr) {
        return {};
    }
    auto sarc = Decompress(file->ReadAllBytes(), {});
    if (!sarc) {
        return {};
    }
    std::vector<u8> dict;
    ForEachSarcMember(*sarc, [&dict](std::string_view name, std::span<u8> member) {
        if (name == "pack.zsdic") {
            dict.assign(member.begin(), member.end());
        }
    });
    return dict;
}

std::vector<std::string> Split(std::string_view text, char separator) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = std::min(text.find(separator, start), text.size());
        if (end > start) {
            parts.emplace_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    return parts;
}

/// `dir` with `file` in place of the file of the same name under the subdirectories `dirs`.
VirtualDir WithFile(const VirtualDir& dir, std::span<const std::string> dirs,
                    const VirtualFile& file) {
    auto files = dir->GetFiles();
    auto subdirs = dir->GetSubdirectories();
    if (dirs.empty()) {
        for (auto& existing : files) {
            if (existing->GetName() == file->GetName()) {
                existing = file;
            }
        }
    } else {
        for (auto& subdir : subdirs) {
            if (subdir->GetName() == dirs.front()) {
                subdir = WithFile(subdir, dirs.subspan(1), file);
            }
        }
    }
    return std::make_shared<VectorVfsDirectory>(std::move(files), std::move(subdirs),
                                                dir->GetName());
}

// The presets as edits of the game's parameter files (the EDEN_NX2_OOT_GFX syntax). Measured in
// Kokiri Forest on the PC phone simulation; docs/nx2/oot-graphics-presets.md.
struct Preset {
    std::string_view name;
    std::string_view edits;
    u32 foliage;
};

// Phone Low: one 64x64 reflection probe instead of three of 128 (~19 fewer passes a frame; off
// saves no more passes and makes the texture cache assert on a zero layer stride), no bloom
// (~20 passes), no depth of field or translucent depth prepass, one 512x512 sun shadow cascade
// instead of three of 1024, a 1024 static shadow map instead of 4096, thinner grass. Depth of
// field is turned off in its agl parameters: without the object (CreateDof=false) the game reads
// and writes through a null pointer every frame.
// Phone Medium: one 64x64 probe, bloom kept, two 512x512 cascades, a 2048 static shadow map.
// High: sun shadow cascades of 2048x2048.
constexpr std::array<Preset, 4> Presets{{
    {"Phone Low",
     "gsys/ModelSceneConfigParam/Main:CubeMapMaxTexNum=1,CubeMapMaxTexWidth=64,"
     "CreateXluZPrePass=false,DepthShadowCascadeNum0=1,DepthShadowTexWidth=512,"
     "DepthShadowTexHeight=512,StaticSdwWidth=1024;agl/BloomParam/MainField:Enable=false;"
     "agl/DepthOfFieldParam/MainField:Enable=false",
     1},
    {"Phone Medium",
     "gsys/ModelSceneConfigParam/Main:CubeMapMaxTexNum=1,CubeMapMaxTexWidth=64,"
     "CreateXluZPrePass=false,DepthShadowCascadeNum0=2,DepthShadowTexWidth=512,"
     "DepthShadowTexHeight=512,StaticSdwWidth=2048;agl/DepthOfFieldParam/MainField:Enable=false",
     0},
    {"Console", "", 0},
    {"High", "gsys/ModelSceneConfigParam/Main:DepthShadowTexWidth=2048,DepthShadowTexHeight=2048",
     0},
}};

} // namespace

std::vector<OotGraphicsEdit> ParseOotGraphicsEdits(std::string_view text) {
    std::vector<OotGraphicsEdit> edits;
    for (const std::string& group : Split(text, ';')) {
        const std::size_t colon = group.find(':');
        if (colon == std::string::npos || colon == 0) {
            continue;
        }
        const std::string file = group.substr(0, colon);
        for (const std::string& entry : Split(std::string_view{group}.substr(colon + 1), ',')) {
            const std::size_t equals = entry.find('=');
            if (equals == std::string::npos || equals == 0 || equals + 1 == entry.size()) {
                continue;
            }
            edits.push_back({file, Split(std::string_view{entry}.substr(0, equals), '.'),
                             entry.substr(equals + 1)});
        }
    }
    return edits;
}

OotGameParams OotGameParamsFor(u64 title_id) {
    if (title_id != OcarinaTitleId) {
        return {};
    }
    u32 index = static_cast<u32>(Settings::values.oot_graphics_preset.GetValue());
    if (const char* text = std::getenv("EDEN_NX2_OOT_GRAPHICS_PRESET");
        text != nullptr && text[0] >= '0' && text[0] <= '3' && text[1] == '\0') {
        index = static_cast<u32>(text[0] - '0');
    }
    const Preset& preset = Presets[std::min<std::size_t>(index, Presets.size() - 1)];
    OotGameParams params{
        .preset = std::string{preset.name},
        .edits = ParseOotGraphicsEdits(preset.edits),
        .foliage = preset.foliage,
    };
    if (const char* text = std::getenv("EDEN_NX2_OOT_FOLIAGE");
        text != nullptr && text[0] >= '0' && text[0] <= '2' && text[1] == '\0') {
        params.foliage = static_cast<u32>(text[0] - '0');
    }
    if (const char* text = std::getenv("EDEN_NX2_OOT_GFX_SKIP")) {
        for (const std::string& key : Split(text, ',')) {
            std::erase_if(params.edits, [&key](const OotGraphicsEdit& edit) {
                return !edit.path.empty() && edit.path.back() == key;
            });
        }
    }
    if (const char* text = std::getenv("EDEN_NX2_OOT_GFX")) {
        for (OotGraphicsEdit& extra : ParseOotGraphicsEdits(text)) {
            std::erase_if(params.edits, [&extra](const OotGraphicsEdit& edit) {
                return edit.file == extra.file && edit.path == extra.path;
            });
            params.edits.push_back(std::move(extra));
        }
    }
    return params;
}

u32 EditBymlFile(std::span<u8> data, std::string_view name,
                 std::span<const OotGraphicsEdit> edits) {
    BymlEditor byml{data};
    if (!byml.IsValid()) {
        return 0;
    }
    u32 written = 0;
    for (const OotGraphicsEdit& edit : edits) {
        if (!name.starts_with(edit.file)) {
            continue;
        }
        if (byml.SetFromText(edit.path, edit.value)) {
            ++written;
        } else {
            LOG_WARNING(Common_Memory, "Ocarina game data: {} {} = {} not applied", name,
                        fmt::join(edit.path, "."), edit.value);
        }
    }
    return written;
}

VirtualDir ApplyOotGameParams(const VirtualDir& root, const OotGameParams& params) {
    if (root == nullptr || !params.Any()) {
        return root;
    }
    const auto begin = std::chrono::steady_clock::now();
    VirtualDir result = root;
    OotGameParamEdits total;

    // Loose files: the edits grouped by the file they name.
    std::vector<OotGraphicsEdit> loose;
    std::vector<OotGraphicsEdit> packed;
    for (const OotGraphicsEdit& edit : params.edits) {
        (edit.file.find('!') == std::string::npos ? loose : packed).push_back(edit);
    }
    std::vector<std::string> done;
    for (const OotGraphicsEdit& edit : loose) {
        auto parts = Split(edit.file, '/');
        if (parts.empty() || std::ranges::find(done, edit.file) != done.end()) {
            continue;
        }
        done.push_back(edit.file);
        const std::string prefix = parts.back();
        parts.pop_back();
        VirtualDir dir = result;
        for (const std::string& part : parts) {
            dir = dir != nullptr ? dir->GetSubdirectory(part) : nullptr;
        }
        if (dir == nullptr) {
            LOG_WARNING(Common_Memory, "Ocarina game data: no {}", edit.file);
            continue;
        }
        for (const VirtualFile& file : dir->GetFiles()) {
            const std::string name = file->GetName();
            if (!name.starts_with(prefix)) {
                continue;
            }
            auto bytes = file->ReadAllBytes();
            std::vector<OotGraphicsEdit> mine;
            for (const OotGraphicsEdit& other : loose) {
                if (other.file == edit.file) {
                    mine.push_back(other);
                    mine.back().file = prefix;
                }
            }
            const u32 written = EditBymlFile(bytes, name, mine);
            if (written != 0) {
                total.values += written;
                result = WithFile(result, parts,
                                  std::make_shared<VectorVfsFile>(std::move(bytes), name));
            }
        }
    }

    // Packs: Foliage for the foliage level, and any pack an edit names.
    const auto pack_dir = result->GetSubdirectory("Pack");
    if (pack_dir != nullptr && (params.foliage != 0 || !packed.empty())) {
        const auto dict = PackDictionary(pack_dir);
        for (const VirtualFile& file : pack_dir->GetFiles()) {
            const std::string name = file->GetName();
            const std::string pack_path = "Pack/" + name;
            const bool foliage = name == FoliagePack && params.foliage != 0;
            std::vector<OotGraphicsEdit> mine;
            for (const OotGraphicsEdit& edit : packed) {
                const std::size_t bang = edit.file.find('!');
                if (pack_path.starts_with(edit.file.substr(0, bang))) {
                    mine.push_back(edit);
                    mine.back().file = edit.file.substr(bang + 1);
                }
            }
            if (!foliage && mine.empty()) {
                continue;
            }
            auto content = dict.empty() ? std::nullopt : Decompress(file->ReadAllBytes(), dict);
            if (!content) {
                LOG_WARNING(Common_Memory, "Ocarina game data: {} could not be read", name);
                continue;
            }
            OotGameParamEdits edits;
            if (foliage) {
                edits = EditFoliagePack(*content, params.foliage);
            }
            if (!mine.empty()) {
                ForEachSarcMember(*content, [&](std::string_view member, std::span<u8> bytes) {
                    edits.values += EditBymlFile(bytes, member, mine);
                });
            }
            if (edits.values + edits.foliage_values == 0) {
                continue;
            }
            total.foliage_scenes += edits.foliage_scenes;
            total.foliage_values += edits.foliage_values;
            total.values += edits.values;
            // Raw blocks: the game decompresses packs with its dictionary, which compressed
            // blocks made without it could not be decoded with.
            result = WithFile(result, std::array<std::string, 1>{"Pack"},
                              std::make_shared<VectorVfsFile>(MakeRawZstdFrame(*content), name));
        }
    }

    if (total.values == 0 && total.foliage_values == 0) {
        LOG_WARNING(Common_Memory, "Ocarina graphics preset {}: nothing changed", params.preset);
        return root;
    }
    LOG_INFO(Common_Memory,
             "Ocarina graphics preset {}: {} values in the game's parameter files, foliage level {} "
             "({} values in {} scenes), made in {} ms",
             params.preset, total.values, params.foliage, total.foliage_values,
             total.foliage_scenes,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - begin)
                 .count());
    return result;
}

} // namespace FileSys
