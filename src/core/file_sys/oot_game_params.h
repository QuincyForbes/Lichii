// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

/// Scalar values of a little-endian BYML (v2 or later) changed in place: the file keeps its size
/// and layout. Containers BYML shares between parents are reached once per path.
class BymlEditor {
public:
    explicit BymlEditor(std::span<u8> data);

    bool IsValid() const {
        return valid;
    }

    /// Dictionary keys and array indices (as decimal text) from the root.
    using Path = std::initializer_list<std::string_view>;

    std::optional<float> GetFloat(Path path) const;
    std::optional<s64> GetInt(Path path) const;
    std::optional<bool> GetBool(Path path) const;

    /// False when the value is missing or of another type. The offset of each value written is
    /// kept, and a second write to the same value is refused.
    bool SetFloat(Path path, float value);
    bool SetInt(Path path, s64 value);
    bool SetBool(Path path, bool value);

    /// Writes `text` ("true"/"false", or a number) as the type the value already has.
    bool SetFromText(std::span<const std::string> path, std::string_view text);

private:
    struct Slot {
        u8 type;
        std::size_t offset;
    };

    std::optional<Slot> Find(std::span<const std::string_view> path) const;
    std::optional<Slot> Find(Path path) const {
        return Find(std::span<const std::string_view>{path.begin(), path.size()});
    }
    bool Claim(std::size_t offset);
    u32 Read32(std::size_t offset) const;
    u32 Read24(std::size_t offset) const;
    std::optional<u32> KeyIndex(std::string_view key) const;

    std::span<u8> data;
    u32 key_table{};
    u32 root{};
    bool valid{};
    std::vector<std::size_t> written;
};

/// Calls `visit` with the name and bytes of each member of a SARC archive; false when `data` is
/// not one.
bool ForEachSarcMember(std::span<u8> data,
                       const std::function<void(std::string_view, std::span<u8>)>& visit);

/// One value of the game's parameter files. `file` is a RomFS path, or `<pack>!<member>` for a
/// file inside one of the game's packs; both match by prefix, so the type suffix
/// (".gsys__ModelSceneConfigParam.bgyml") can be left out.
struct OotGraphicsEdit {
    std::string file;
    std::vector<std::string> path;
    std::string value;
};

/// "file:Key.Sub=value,Key2=value;file2:Key=value" (the EDEN_NX2_OOT_GFX syntax). Entries that do
/// not parse are skipped.
std::vector<OotGraphicsEdit> ParseOotGraphicsEdits(std::string_view text);

/// What Settings oot_graphics_preset asks of the game's files: its edits and the foliage level
/// (0 the game's, 1 half the grass and the plants' distances, 2 also no moss, second undergrowth
/// layer or grass shadows).
struct OotGameParams {
    std::string preset;
    std::vector<OotGraphicsEdit> edits;
    u32 foliage{};

    bool Any() const {
        return !edits.empty() || foliage != 0;
    }
};

/// The edits of Settings oot_graphics_preset for `title_id` (nothing for other games).
/// EDEN_NX2_OOT_GRAPHICS_PRESET=0..3 overrides the setting, EDEN_NX2_OOT_FOLIAGE=0..2 the foliage
/// level, and EDEN_NX2_OOT_GFX adds edits (replacing the preset's for the same value); an entry
/// of EDEN_NX2_OOT_GFX_SKIP (a key name) leaves the preset's edits of that key out.
OotGameParams OotGameParamsFor(u64 title_id);

/// Counts of the values ApplyOotGameParams changed.
struct OotGameParamEdits {
    u32 foliage_scenes{};
    u32 foliage_values{};
    u32 values{};
};

/// Edits the foliage scenes (ResFoliageScene) of a decompressed Foliage pack.
OotGameParamEdits EditFoliagePack(std::span<u8> sarc, u32 level);

/// Applies `edits` whose file matches `name` to one BYML file; returns how many values it wrote.
u32 EditBymlFile(std::span<u8> byml, std::string_view name,
                 std::span<const OotGraphicsEdit> edits);

/// Ocarina of Time's RomFS with the files `params` names served edited (made here, once, from the
/// game's own files; nothing is written). `root` when nothing is asked or nothing matched.
VirtualDir ApplyOotGameParams(const VirtualDir& root, const OotGameParams& params);

} // namespace FileSys
