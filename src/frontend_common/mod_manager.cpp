// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include "common/fs/fs.h"
#include "common/fs/fs_types.h"
#include "common/fs/fs_util.h"
#include "common/logging.h"
#include "frontend_common/data_manager.h"
#include "mod_manager.h"

namespace FrontendCommon {

// TODO: Handle cases where the folder appears to contain multiple mods.
std::vector<std::filesystem::path> GetModFolder(const std::string& root) {
    std::vector<std::filesystem::path> paths;

    auto callback = [&paths](const std::filesystem::directory_entry& entry) -> bool {
        const auto name = Common::FS::PathToUTF8String(entry.path().filename());
        static const std::array<std::string, 5> valid_names = {"exefs",
                                                               "romfs",
                                                               "romfs_ext",
                                                               "cheats",
                                                               "romfslite"};

        if (std::ranges::find(valid_names, name) != valid_names.end()) {
            const auto parent = entry.path().parent_path();
            if (std::ranges::find(paths, parent) == paths.end()) {
                paths.emplace_back(parent);
            }
        }

        return true;
    };

    Common::FS::IterateDirEntriesRecursively(root, callback, Common::FS::DirEntryFilter::Directory);

    return paths;
}

ModInstallResult InstallMod(const std::filesystem::path& path, const u64 program_id, const bool copy) {
    const auto program_id_string = fmt::format("{:016X}", program_id);
    std::filesystem::path mod_dir;

    try {
        auto source = std::filesystem::absolute(path);
        while (source.has_relative_path() &&
               (source.filename().empty() || source.filename() == ".")) {
            source = source.parent_path();
        }
        const auto mod_name = source.filename();
        if (mod_name.empty() || mod_name == "..") {
            return Failed;
        }
        mod_dir = DataManager::GetDataDir(DataManager::DataDir::Mods) / program_id_string / mod_name;

        // A mod selected from its installed location must not delete its own source.
        if (std::filesystem::exists(mod_dir) && std::filesystem::equivalent(source, mod_dir)) {
            return Success;
        }

        std::filesystem::remove_all(mod_dir);
        std::filesystem::create_directories(mod_dir.parent_path());
        std::filesystem::copy(source, mod_dir, std::filesystem::copy_options::recursive);
        if (!copy) {
            std::filesystem::remove_all(source);
        }
    } catch (std::exception& e) {
        LOG_ERROR(Frontend, "Mod install failed with message {}", e.what());
        return Failed;
    }

    LOG_INFO(Frontend, "Copied mod from {} to {}", Common::FS::PathToUTF8String(path),
             Common::FS::PathToUTF8String(mod_dir));

    return Success;
}

ModInstallResult InstallModFromUTF8(const std::string& path, const u64 program_id, const bool copy) {
    return InstallMod(std::filesystem::path{Common::FS::ToU8String(path)}, program_id, copy);
}

} // namespace FrontendCommon
