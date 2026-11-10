// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "common/common_types.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/vfs/vfs_types.h"
#include "core/memory/dmnt_cheat_types.h"

namespace Core {
class System;
}

namespace Service::FileSystem {
class FileSystemController;
}

namespace FileSys {

class ContentProvider;
class NCA;
class NACP;

enum class PatchType { Update, DLC, Mod };

enum class PatchSource {
    Unknown,
    NAND,
    SDMC,
    External,
    Packed,
};

/// What a built-in patch changes; each kind has a global setting (Settings::patch_*).
enum class BuiltinPatchKind : u8 {
    FrameRate,
    Resolution,
    Performance,
    Image,
};

/// A game's own choice for a built-in patch.
enum class BuiltinPatchChoice : u8 {
    FollowGlobal,
    On,
    Off,
};

/// Which setting decided whether a built-in patch applies.
enum class BuiltinPatchSource : u8 {
    Game,
    Global,
    Default,
};

struct BuiltinPatchInfo {
    BuiltinPatchKind kind;
    std::string game;
    std::string description;
    /// What turning it on costs and gains.
    std::string effect;
    BuiltinPatchChoice choice;
    BuiltinPatchSource source;
    /// The resolved choice, before another patch that contains this one is considered.
    bool resolved_enabled;
    /// Name of an enabled built-in patch that contains this one (it then does nothing here).
    std::string included_in;
};

struct Patch {
    bool enabled;
    std::string name;
    std::string version;
    PatchType type;
    u64 program_id;
    u64 title_id;
    PatchSource source;
    std::string location;
    u32 numeric_version{0};
    std::optional<BuiltinPatchInfo> builtin{};
};

[[nodiscard]] std::string_view BuiltinPatchKindName(BuiltinPatchKind kind);
[[nodiscard]] std::string_view BuiltinPatchSourceName(BuiltinPatchSource source);
/// Games that ship at least one built-in patch, in table order.
[[nodiscard]] std::vector<std::string> BuiltinPatchGames();
/// Sets a built-in patch's per-game choice in a disabled add-ons list.
void ApplyBuiltinPatchChoice(std::vector<std::string>& disabled, std::string_view name,
                             BuiltinPatchChoice choice);
/// Whether a built-in patch is on when its game follows the global setting.
[[nodiscard]] bool BuiltinPatchEnabledWhenFollowing(std::string_view name);
[[nodiscard]] BuiltinPatchChoice GetBuiltinPatchChoice(const std::vector<std::string>& disabled,
                                                       std::string_view name);

// A centralized class to manage patches to games.
class PatchManager {
public:
    using BuildID = std::array<u8, 0x20>;
    using Metadata = std::pair<std::unique_ptr<NACP>, VirtualFile>;

    explicit PatchManager(u64 title_id_,
                          const Service::FileSystem::FileSystemController& fs_controller_,
                          const ContentProvider& content_provider_);
    ~PatchManager();

    [[nodiscard]] u64 GetTitleID() const;

    // Currently tracked ExeFS patches:
    // - Game Updates
    [[nodiscard]] VirtualDir PatchExeFS(VirtualDir exefs) const;

    // Currently tracked NSO patches:
    // - IPS
    // - IPSwitch
    // - Built-in patches, keyed by title and build id
    [[nodiscard]] std::vector<u8> PatchNSO(const std::vector<u8>& nso,
                                           const std::string& name) const;

    // Checks to see if PatchNSO() will have any effect given the NSO's build ID.
    // Used to prevent expensive copies in NSO loader.
    [[nodiscard]] bool HasNSOPatch(const BuildID& build_id, std::string_view name) const;

    // Creates a CheatList object with all
    [[nodiscard]] std::vector<Core::Memory::CheatEntry> CreateCheatList(
        const BuildID& build_id) const;

    // Currently tracked RomFS patches:
    // - Game Updates
    // - LayeredFS
    [[nodiscard]] VirtualFile PatchRomFS(const NCA* base_nca, VirtualFile base_romfs,
                                         ContentRecordType type = ContentRecordType::Program,
                                         VirtualFile packed_update_raw = nullptr,
                                         bool apply_layeredfs = true) const;

    // Returns a vector of patches
    [[nodiscard]] std::vector<Patch> GetPatches(VirtualFile update_raw = nullptr) const;

    // If the game update exists, returns the u32 version field in its Meta-type NCA. If that fails,
    // it will fallback to the Meta-type NCA of the base game. If that fails, the result will be
    // std::nullopt
    [[nodiscard]] std::optional<u32> GetGameVersion() const;

    // Given title_id of the program, attempts to get the control data of the update and parse
    // it, falling back to the base control data.
    [[nodiscard]] Metadata GetControlMetadata() const;

    // Version of GetControlMetadata that takes an arbitrary NCA
    [[nodiscard]] Metadata ParseControlNCA(const NCA& nca) const;

    /// @brief Gets NACP metadata (accounting for any patches or updates)
    [[nodiscard]] static PatchManager::Metadata GetMetadataFromBaseOrUpdate(Core::System& system, u64 application_id) noexcept;

private:
    [[nodiscard]] VirtualDir GetModificationLoadRoot(bool sdmc = false) const;
    [[nodiscard]] std::vector<std::string> GetDisabledAddons() const;
    void ApplyLayeredFS(VirtualFile& romfs, ContentRecordType type) const;
    [[nodiscard]] std::vector<VirtualFile> CollectPatches(const std::vector<VirtualDir>& patch_dirs,
                                                          const std::string& build_id) const;

    u64 title_id;
    std::optional<u64> parent_title_id;
    const Service::FileSystem::FileSystemController& fs_controller;
    const ContentProvider& content_provider;
};

} // namespace FileSys
