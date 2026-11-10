// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "common/uuid.h"
#include "core/core.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/savedata_factory.h"
#include "core/file_sys/submission_package.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/service/acc/profile_manager.h"
#include "core/hle/service/am/am_types.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/loader.h"
#include "core/memory/cheat_engine.h"
#include "frontend_common/content_manager.h"
#include "frontend_common/firmware_manager.h"
#include "frontend_common/mod_manager.h"
#include "ios_frontend/bridge_host.h"
#include "ios_frontend/lichii_bridge_content.h"
#include "ios_frontend/lichii_bridge_content_host.h"
#include "ios_frontend/lichii_content_zip.h"

namespace LichiiContent {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using Common::FS::EdenPath;

struct State {
    std::recursive_mutex mutex;
    std::mutex applet_mutex;
    std::string applet_path; // boots as MiiEdit
};

State& GetState() {
    static State state;
    return state;
}

// Installs change the content the library scan reads: both hold the bridge's loader lock.
class Lock {
public:
    explicit Lock(std::recursive_mutex& mutex) : lock{IosFrontend::LoaderMutex(), mutex} {}

private:
    std::scoped_lock<std::recursive_mutex, std::recursive_mutex> lock;
};

// The SDL frontend attaches its System (filesystem and content provider set) before the app UI
// shows, so it is always there when Swift calls in.
Core::System& System() {
    return IosFrontend::System();
}

// Only the integrity check reads it; the app registers no loose game files.
FileSys::ManualContentProvider& ManualProvider() {
    static FileSys::ManualContentProvider empty;
    return empty;
}

bool IsRunning() {
    const auto state = IosFrontend::GetState();
    return state != LICHII_EMU_IDLE && state != LICHII_EMU_FAILED;
}

void SaveConfig() {
    IosFrontend::SaveGlobalConfig();
}

void RefreshFactories() {
    auto& system = System();
    system.GetFileSystemController().CreateFactories(*system.GetFilesystem());
}

char* Dup(std::string_view text) {
    auto* out = static_cast<char*>(std::malloc(text.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, text.data(), text.size());
        out[text.size()] = '\0';
    }
    return out;
}

char* DupJson(const json& value) {
    return Dup(value.dump(-1, ' ', false, json::error_handler_t::replace));
}

void SetMessage(char** out, std::string_view text) {
    if (out != nullptr) {
        *out = Dup(text);
    }
}

std::string PathString(const fs::path& path) {
    return Common::FS::PathToUTF8String(path);
}

fs::path PathFrom(const char* text) {
    return text != nullptr ? fs::path{std::string{text}} : fs::path{};
}

std::string Hex16(u64 value) {
    return fmt::format("{:016X}", value);
}

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

fs::path Documents() {
    return Common::FS::GetEdenPath(EdenPath::EdenDir).parent_path();
}

fs::path ScratchDir(std::string_view what) {
    std::random_device random;
    const auto dir = Common::FS::GetEdenPath(EdenPath::CacheDir) / "import" /
                     fmt::format("{}-{:08x}", what, random());
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir;
}

void RemoveQuietly(const fs::path& path) {
    std::error_code ec;
    fs::remove_all(path, ec);
}

LichiiZip::ProgressFn ScaledProgress(LichiiContentProgress progress, void* user, u64 from, u64 to,
                                     u64 scale) {
    if (progress == nullptr) {
        return {};
    }
    return [=](std::uint64_t done, std::uint64_t total) {
        const u64 span = to - from;
        const u64 value = total == 0 ? to : from + (span * done) / total;
        return progress(user, value, scale);
    };
}

std::string ReadTextFile(const fs::path& path, std::size_t limit = 16u << 20) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::string text;
    text.resize(limit);
    in.read(text.data(), static_cast<std::streamsize>(limit));
    text.resize(static_cast<std::size_t>(in.gcount()));
    return text;
}

bool CopyFileOver(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::create_directories(to.parent_path(), ec);
    fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        LOG_ERROR(Frontend, "Copying {} to {} failed: {}", PathString(from), PathString(to),
                  ec.message());
    }
    return !ec;
}

std::vector<std::string>& DisabledAddons(u64 program_id) {
    return Settings::values.disabled_addons[program_id];
}

void InvalidateGameListCache(u64 program_id) {
    std::error_code ec;
    fs::remove(Common::FS::GetEdenPath(EdenPath::CacheDir) / "game_list" /
                   fmt::format("{:016X}.pv.txt", program_id),
               ec);
}

void SetDisabled(u64 program_id, const std::string& key, bool disabled) {
    auto& list = DisabledAddons(program_id);
    const auto it = std::find(list.begin(), list.end(), key);
    if (disabled && it == list.end()) {
        list.push_back(key);
    } else if (!disabled && it != list.end()) {
        list.erase(it);
    }
}

bool ValidName(std::string_view name) {
    return !name.empty() && name != "." && name != ".." &&
           name.find_first_of("/\\") == std::string_view::npos;
}

// ---- Keys --------------------------------------------------------------------------------

enum class KeyFileKind { Invalid, Prod, Title, RetailBin };

bool IsHex(std::string_view text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
    });
}

std::string_view Trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

// A key file is "name = hex" lines (comments with ';' or '#'). Title keys use a 32 hex digit
// rights id as the name. Lines that are neither are counted in `skipped`, not fatal: dumping tools
// add their own lines, and the key manager ignores what it doesn't know as well.
KeyFileKind ClassifyKeyFile(const fs::path& path, int& entries, int& skipped) {
    entries = 0;
    skipped = 0;
    if (Lower(PathString(path.filename())) == "key_retail.bin") {
        return KeyFileKind::RetailBin;
    }
    auto text = ReadTextFile(path, 4u << 20);
    if (text.starts_with("\xEF\xBB\xBF")) {
        text.erase(0, 3);
    }
    int title_entries = 0;
    std::size_t start = 0;
    while (start < text.size()) {
        auto end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const auto line = Trim(std::string_view{text}.substr(start, end - start));
        start = end + 1;
        if (line.empty() || line.front() == ';' || line.front() == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string_view::npos) {
            ++skipped;
            continue;
        }
        const auto name = Trim(line.substr(0, eq));
        const auto value = Trim(line.substr(eq + 1));
        const bool name_ok = !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
            return std::isalnum(c) != 0 || c == '_';
        });
        if (!name_ok || !IsHex(value)) {
            ++skipped;
            continue;
        }
        ++entries;
        if (name.size() == 32 && IsHex(name) && value.size() == 32) {
            ++title_entries;
        }
    }
    if (entries == 0) {
        return KeyFileKind::Invalid;
    }
    return title_entries == entries ? KeyFileKind::Title : KeyFileKind::Prod;
}

int CountKeyLines(const fs::path& path) {
    int entries = 0;
    int skipped = 0;
    if (!Common::FS::Exists(path) ||
        ClassifyKeyFile(path, entries, skipped) == KeyFileKind::Invalid) {
        return 0;
    }
    return entries;
}

json KeyStatus() {
    const auto keys_dir = Common::FS::GetEdenPath(EdenPath::KeysDir);
    auto& keys = Core::Crypto::KeyManager::Instance();
    int master_keys = 0;
    for (u64 i = 0; i < 0x20; ++i) {
        if (keys.HasKey(Core::Crypto::S128KeyType::Master, i)) {
            ++master_keys;
        }
    }
    return {
        {"prod_keys", Common::FS::Exists(keys_dir / "prod.keys")},
        {"title_keys", Common::FS::Exists(keys_dir / "title.keys")},
        {"header_key", keys.HasKey(Core::Crypto::S256KeyType::Header)},
        {"master_keys", master_keys},
        {"title_key_count", CountKeyLines(keys_dir / "title.keys")},
        {"usable", ContentManager::AreKeysPresent()},
    };
}

// ---- Firmware ------------------------------------------------------------------------------

json FirmwareStatus() {
    auto& system = System();
    const bool present = FirmwareManager::CheckFirmwarePresence(system);
    json out{{"present", present}, {"version", ""}, {"title", ""}, {"corrupted", false}};
    if (!present) {
        return out;
    }
    const auto [data, result] = FirmwareManager::GetFirmwareVersion(system);
    if (result.IsError()) {
        out["corrupted"] = true;
        return out;
    }
    out["version"] = std::string{data.display_version.data()};
    out["title"] = std::string{data.display_title.data()};
    return out;
}

std::vector<fs::path> FindNcas(const fs::path& root) {
    std::vector<fs::path> out;
    Common::FS::IterateDirEntriesRecursively(
        root,
        [&out](const fs::directory_entry& entry) {
            if (Lower(PathString(entry.path().extension())) == ".nca") {
                out.push_back(entry.path());
            }
            return true;
        },
        Common::FS::DirEntryFilter::File);
    std::sort(out.begin(), out.end());
    return out;
}

// ---- Add-ons -------------------------------------------------------------------------------

const char* SourceName(FileSys::PatchSource source) {
    switch (source) {
    case FileSys::PatchSource::NAND:
        return "nand";
    case FileSys::PatchSource::SDMC:
        return "sdmc";
    case FileSys::PatchSource::External:
        return "external";
    case FileSys::PatchSource::Packed:
        return "packed";
    default:
        return "unknown";
    }
}

const char* SlotName(FileSys::ContentProviderUnionSlot slot) {
    switch (slot) {
    case FileSys::ContentProviderUnionSlot::SysNAND:
    case FileSys::ContentProviderUnionSlot::UserNAND:
        return "nand";
    case FileSys::ContentProviderUnionSlot::SDMC:
        return "sdmc";
    default:
        return "external";
    }
}

std::string ToggleKey(const FileSys::Patch& patch) {
    if (patch.type == FileSys::PatchType::Update &&
        patch.source == FileSys::PatchSource::External && patch.numeric_version != 0) {
        return fmt::format("Update@{}", patch.numeric_version);
    }
    return patch.name;
}

std::vector<FileSys::Patch> Patches(u64 program_id, const char* game_path) {
    auto& system = System();
    const FileSys::PatchManager pm{program_id, system.GetFileSystemController(),
                                   system.GetContentProvider()};
    FileSys::VirtualFile update_raw;
    if (game_path != nullptr && game_path[0] != '\0') {
        const auto file = Core::GetGameFileFromPath(system.GetFilesystem(), game_path);
        if (file != nullptr) {
            if (const auto loader = Loader::GetLoader(system, file); loader != nullptr) {
                loader->ReadUpdateRaw(update_raw);
            }
        }
    }
    return pm.GetPatches(update_raw);
}

// ---- Cheats --------------------------------------------------------------------------------

std::string CheatKey(std::string_view source, std::string_view name) {
    return fmt::format("cheat:{}/{}", source, name);
}

json ParseCheats(const fs::path& file, const std::vector<std::string>& disabled,
                 const std::string& source) {
    const auto text = ReadTextFile(file);
    const Core::Memory::TextCheatParser parser;
    const auto entries = parser.Parse(text);
    json cheats = json::array();
    for (const auto& entry : entries) {
        const std::string name{entry.definition.readable_name.data()};
        if (name.empty() || entry.definition.num_opcodes == 0) {
            continue;
        }
        const bool enabled = std::find(disabled.begin(), disabled.end(),
                                       CheatKey(source, name)) == disabled.end();
        cheats.push_back({{"name", name},
                          {"enabled", enabled},
                          {"opcodes", entry.definition.num_opcodes}});
    }
    return cheats;
}

fs::path ModRoot(u64 program_id) {
    return Common::FS::GetEdenPath(EdenPath::LoadDir) / Hex16(program_id);
}

std::optional<fs::path> FindCaseless(const fs::path& dir, std::string_view name) {
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_directory(ec) && Lower(PathString(entry.path().filename())) == name) {
            return entry.path();
        }
    }
    return std::nullopt;
}

// ---- Saves ---------------------------------------------------------------------------------

fs::path SaveDirFor(u64 program_id, const char* game_path) {
    auto& system = System();
    u64 user_size = 0;
    u64 device_size = 0;
    const FileSys::PatchManager pm{program_id, system.GetFileSystemController(),
                                   system.GetContentProvider()};
    if (const auto control = pm.GetControlMetadata().first; control != nullptr) {
        user_size = control->GetDefaultNormalSaveSize();
        device_size = control->GetDeviceSaveDataSize();
    } else if (game_path != nullptr && game_path[0] != '\0') {
        const auto file = Core::GetGameFileFromPath(system.GetFilesystem(), game_path);
        if (file != nullptr) {
            if (const auto loader = Loader::GetLoader(system, file); loader != nullptr) {
                FileSys::NACP nacp{};
                if (loader->ReadControlData(nacp) == Loader::ResultStatus::Success) {
                    user_size = nacp.GetDefaultNormalSaveSize();
                    device_size = nacp.GetDeviceSaveDataSize();
                }
            }
        }
    }
    // Same rule as the Qt frontend: NX2 games report no sizes yet use account saves.
    const bool use_device_save = device_size > 0 && user_size == 0;

    u128 user_id{};
    if (!use_device_save) {
        const auto& manager = system.GetProfileManager();
        const auto index = static_cast<std::size_t>(
            std::max(0, Settings::values.current_user.GetValue()));
        if (const auto uuid = manager.GetUser(index); uuid) {
            user_id = uuid->AsU128();
        } else if (const auto first = manager.GetUser(0); first) {
            user_id = first->AsU128();
        }
    }

    const auto save_dir = Common::FS::GetEdenPath(EdenPath::SaveDir);
    std::error_code ec;
    fs::create_directories(save_dir, ec);
    const auto vfs_save_dir =
        system.GetFilesystem()->OpenDirectory(PathString(save_dir), FileSys::OpenMode::Read);
    const auto relative = FileSys::SaveDataFactory::GetFullPath(
        {}, vfs_save_dir, FileSys::SaveDataSpaceId::User, FileSys::SaveDataType::Account,
        program_id, user_id, 0);
    return Common::FS::ConcatPathSafe(save_dir, relative);
}

std::pair<u64, u64> DirStats(const fs::path& dir) {
    u64 size = 0;
    u64 files = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            size += it->file_size(ec);
            ++files;
        }
    }
    return {size, files};
}

// A save archive made elsewhere often wraps everything in one folder; unwrap it.
fs::path SingleChildRoot(const fs::path& dir) {
    std::error_code ec;
    std::vector<fs::path> children;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        const auto name = PathString(entry.path().filename());
        if (name == "__MACOSX" || name == ".DS_Store") {
            continue;
        }
        children.push_back(entry.path());
    }
    if (children.size() == 1 && fs::is_directory(children.front(), ec)) {
        return children.front();
    }
    return dir;
}

// ---- Logs ----------------------------------------------------------------------------------

std::vector<fs::path> LogFiles() {
    std::vector<fs::path> out;
    std::error_code ec;
    const auto add_dir = [&](const fs::path& dir) {
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (entry.is_regular_file(ec)) {
                out.push_back(entry.path());
            }
        }
        ec.clear();
    };
    add_dir(Common::FS::GetEdenPath(EdenPath::LogDir));
    for (const char* name : {"lichii-startup.txt", "console.txt"}) {
        const auto path = Documents() / name;
        if (fs::is_regular_file(path, ec)) {
            out.push_back(path);
        }
    }
    add_dir(Common::FS::GetEdenPath(EdenPath::CrashDumpsDir));
    return out;
}

// ---- Profiles ------------------------------------------------------------------------------

std::string ProfileName(const Service::Account::ProfileManager& manager, const Common::UUID& uuid) {
    Service::Account::ProfileBase base{};
    if (!manager.GetProfileBase(uuid, base)) {
        return {};
    }
    return Common::StringFromFixedZeroTerminatedBuffer(
        {reinterpret_cast<const char*>(base.username.data()), base.username.size()},
        base.username.size());
}

fs::path AvatarPath(const Common::UUID& uuid) {
    return Common::FS::GetEdenPath(EdenPath::NANDDir) /
           fmt::format("system/save/8000000000000010/su/avators/{}.jpg", uuid.FormattedString());
}

std::optional<Common::UUID> ParseUuid(const char* text) {
    if (text == nullptr) {
        return std::nullopt;
    }
    const std::string_view view{text};
    if (view.size() != 32 && view.size() != 36) {
        return std::nullopt;
    }
    const Common::UUID uuid{view};
    if (!uuid.IsValid()) {
        return std::nullopt;
    }
    return uuid;
}

Service::Account::ProfileUsername ToUsername(std::string_view name) {
    Service::Account::ProfileUsername out{};
    // Cut at a UTF-8 character boundary.
    std::size_t length = std::min(name.size(), out.size());
    while (length > 0 && length < name.size() &&
           (static_cast<unsigned char>(name[length]) & 0xC0) == 0x80) {
        --length;
    }
    std::memcpy(out.data(), name.data(), length);
    return out;
}

std::string MiiEditPath() {
    const auto bis_system = System().GetFileSystemController().GetSystemNANDContents();
    if (bis_system == nullptr) {
        return {};
    }
    const auto nca = bis_system->GetEntry(static_cast<u64>(Service::AM::AppletProgramId::MiiEdit),
                                          FileSys::ContentRecordType::Program);
    return nca != nullptr ? nca->GetFullPath() : std::string{};
}

} // Anonymous namespace

u32 TakePendingAppletId(const std::string& path) {
    auto& state = GetState();
    std::scoped_lock lock{state.applet_mutex};
    const bool mii_edit = !state.applet_path.empty() && state.applet_path == path;
    state.applet_path.clear();
    return static_cast<u32>(mii_edit ? Service::AM::AppletId::MiiEdit
                                     : Service::AM::AppletId::Application);
}

} // namespace LichiiContent

using namespace LichiiContent;

extern "C" {

void lichii_content_free(void* p) {
    std::free(p);
}

char* lichii_content_paths_json(void) {
    return DupJson(json{
        {"user_dir", PathString(Common::FS::GetEdenPath(EdenPath::EdenDir))},
        {"documents", PathString(Documents())},
        {"keys_dir", PathString(Common::FS::GetEdenPath(EdenPath::KeysDir))},
        {"load_dir", PathString(Common::FS::GetEdenPath(EdenPath::LoadDir))},
        {"log_dir", PathString(Common::FS::GetEdenPath(EdenPath::LogDir))},
        {"nand_dir", PathString(Common::FS::GetEdenPath(EdenPath::NANDDir))},
        {"sdmc_dir", PathString(Common::FS::GetEdenPath(EdenPath::SDMCDir))},
        {"save_dir", PathString(Common::FS::GetEdenPath(EdenPath::SaveDir))},
        {"cache_dir", PathString(Common::FS::GetEdenPath(EdenPath::CacheDir))},
    });
}

char* lichii_content_setup_status_json(void) {
    Lock lock{GetState().mutex};
    const auto keys = KeyStatus();
    const auto firmware = FirmwareStatus();
    const auto profiles = System().GetProfileManager().GetUserCount();
    return DupJson(json{
        {"keys", keys},
        {"firmware", firmware},
        {"profiles", profiles},
        {"ready", keys.value("usable", false)},
    });
}

// ---- Keys ----------------------------------------------------------------------------------

LichiiKeyResult lichii_content_install_keys(const char* const* paths, int count,
                                            char** out_message) {
    Lock lock{GetState().mutex};
    const auto keys_dir = Common::FS::GetEdenPath(EdenPath::KeysDir);
    std::vector<std::string> lines;
    std::vector<std::pair<fs::path, fs::path>> copies;
    bool invalid = false;

    for (int i = 0; i < count && paths != nullptr; ++i) {
        const auto path = PathFrom(paths[i]);
        const auto file_name = PathString(path.filename());
        int entries = 0;
        int skipped = 0;
        if (!Common::FS::IsFile(path)) {
            lines.push_back(fmt::format("{}: not found", file_name));
            continue;
        }
        const auto skipped_note = [&] {
            return skipped != 0 ? fmt::format(" ({} unrecognised lines ignored)", skipped)
                                : std::string{};
        };
        switch (ClassifyKeyFile(path, entries, skipped)) {
        case KeyFileKind::Prod:
            copies.emplace_back(path, keys_dir / "prod.keys");
            lines.push_back(fmt::format("{}: {} keys{}", file_name, entries, skipped_note()));
            break;
        case KeyFileKind::Title:
            copies.emplace_back(path, keys_dir / "title.keys");
            lines.push_back(
                fmt::format("{}: {} title keys{}", file_name, entries, skipped_note()));
            break;
        case KeyFileKind::RetailBin:
            copies.emplace_back(path, keys_dir / "key_retail.bin");
            lines.push_back(fmt::format("{}: copied", file_name));
            break;
        case KeyFileKind::Invalid:
            invalid = true;
            lines.push_back(fmt::format("{}: not a key file (expected \"name = hex\" lines)",
                                        file_name));
            break;
        }
    }

    const auto finish = [&](LichiiKeyResult result) {
        SetMessage(out_message, fmt::format("{}", fmt::join(lines, "\n")));
        return result;
    };

    if (copies.empty()) {
        return finish(invalid ? LICHII_KEYS_INVALID_FORMAT : LICHII_KEYS_NO_FILES);
    }
    for (const auto& [from, to] : copies) {
        if (!CopyFileOver(from, to)) {
            lines.push_back(fmt::format("Copying {} failed", PathString(to.filename())));
            return finish(LICHII_KEYS_COPY_FAILED);
        }
    }

    Core::Crypto::KeyManager::Instance().ReloadKeys();
    RefreshFactories();
    LOG_INFO(Frontend, "Installed {} key file(s)", copies.size());

    if (!ContentManager::AreKeysPresent()) {
        lines.push_back("The header key or master keys are still missing: dump prod.keys again "
                        "with a current Lockpick.");
        return finish(LICHII_KEYS_INCOMPLETE);
    }
    if (invalid) {
        return finish(LICHII_KEYS_INVALID_FORMAT);
    }
    return finish(LICHII_KEYS_OK);
}

char* lichii_content_key_status_json(void) {
    Lock lock{GetState().mutex};
    return DupJson(KeyStatus());
}

bool lichii_content_remove_keys(void) {
    Lock lock{GetState().mutex};
    const auto keys_dir = Common::FS::GetEdenPath(EdenPath::KeysDir);
    std::error_code ec;
    bool ok = true;
    for (const char* name : {"prod.keys", "title.keys", "key_retail.bin"}) {
        fs::remove(keys_dir / name, ec);
        ok = ok && !ec;
    }
    Core::Crypto::KeyManager::Instance().ReloadKeys();
    return ok;
}

// ---- Firmware ------------------------------------------------------------------------------

LichiiFirmwareResult lichii_content_install_firmware(const char* path_text,
                                                     LichiiContentProgress progress, void* user,
                                                     char** out_message) {
    Lock lock{GetState().mutex};
    constexpr u64 Scale = 1000;
    const auto report = [&](u64 value) {
        return progress != nullptr && progress(user, value, Scale);
    };
    const auto fail = [&](LichiiFirmwareResult result, std::string_view message) {
        LOG_ERROR(Frontend, "Firmware install failed: {}", message);
        SetMessage(out_message, message);
        return result;
    };

    if (IsRunning()) {
        return fail(LICHII_FIRMWARE_BUSY, "Stop the game before installing firmware.");
    }
    if (!ContentManager::AreKeysPresent()) {
        return fail(LICHII_FIRMWARE_NO_KEYS, "Install prod.keys before installing firmware.");
    }

    const auto source = PathFrom(path_text);
    fs::path root = source;
    fs::path scratch;
    if (Common::FS::IsFile(source)) {
        if (Lower(PathString(source.extension())) != ".zip") {
            return fail(LICHII_FIRMWARE_BAD_ZIP, "Pick a firmware .zip or a folder of .nca files.");
        }
        scratch = ScratchDir("firmware");
        const auto result =
            LichiiZip::Extract(source, scratch, ScaledProgress(progress, user, 0, 300, Scale));
        if (result != LichiiZip::Result::Success) {
            RemoveQuietly(scratch);
            if (result == LichiiZip::Result::Cancelled) {
                return fail(LICHII_FIRMWARE_CANCELLED, "Cancelled.");
            }
            return fail(LICHII_FIRMWARE_BAD_ZIP, LichiiZip::ResultString(result));
        }
        root = scratch;
    } else if (!Common::FS::IsDir(source)) {
        return fail(LICHII_FIRMWARE_NO_NCAS, "The firmware file or folder was not found.");
    }

    const auto cleanup = [&] {
        if (!scratch.empty()) {
            RemoveQuietly(scratch);
        }
    };

    const auto ncas = FindNcas(root);
    if (ncas.empty()) {
        cleanup();
        return fail(LICHII_FIRMWARE_NO_NCAS, "No .nca files were found in the firmware.");
    }
    LOG_INFO(Frontend, "Installing firmware from {} ({} NCAs)", PathString(source), ncas.size());

    auto& system = System();
    auto sysnand = system.GetFileSystemController().GetSystemNANDContentDirectory();
    if (sysnand == nullptr ||
        (sysnand->IsWritable() && !sysnand->CleanSubdirectoryRecursive("registered"))) {
        cleanup();
        return fail(LICHII_FIRMWARE_DELETE_FAILED,
                    "The old firmware could not be removed from nand/system.");
    }
    auto registered = sysnand->GetDirectoryRelative("registered");
    if (registered == nullptr) {
        registered = sysnand->CreateDirectoryRelative("registered");
    }
    if (registered == nullptr) {
        cleanup();
        return fail(LICHII_FIRMWARE_COPY_FAILED, "nand/system/Contents/registered is missing.");
    }

    const auto& vfs = system.GetFilesystem();
    for (std::size_t i = 0; i < ncas.size(); ++i) {
        const auto src = vfs->OpenFile(PathString(ncas[i]), FileSys::OpenMode::Read);
        const auto dst = registered->CreateFileRelative(PathString(ncas[i].filename()));
        if (!FileSys::VfsRawCopy(src, dst)) {
            cleanup();
            RefreshFactories();
            return fail(LICHII_FIRMWARE_COPY_FAILED,
                        fmt::format("Copying {} failed (is the storage full?)",
                                    PathString(ncas[i].filename())));
        }
        if (report(300 + (500 * (i + 1)) / ncas.size())) {
            cleanup();
            RefreshFactories();
            return fail(LICHII_FIRMWARE_CANCELLED,
                        "Cancelled: the firmware is incomplete, install it again.");
        }
    }
    cleanup();
    RefreshFactories();

    bool cancelled = false;
    const auto failed = ContentManager::VerifyInstalledContents(
        system, ManualProvider(),
        [&](size_t total, size_t processed) {
            cancelled = report(800 + (total == 0 ? 200 : (200 * processed) / total));
            return cancelled;
        },
        true);
    if (!failed.empty()) {
        return fail(LICHII_FIRMWARE_VERIFY_FAILED,
                    fmt::format("Verification failed for:\n{}", fmt::join(failed, "\n")));
    }

    const auto status = FirmwareStatus();
    const auto version = status.value("version", std::string{});
    LOG_INFO(Frontend, "Firmware {} installed", version);
    report(Scale);
    SetMessage(out_message, version.empty() ? std::string{"Firmware installed."}
                                            : fmt::format("Firmware {} installed.", version));
    return LICHII_FIRMWARE_OK;
}

char* lichii_content_firmware_status_json(void) {
    Lock lock{GetState().mutex};
    return DupJson(FirmwareStatus());
}

bool lichii_content_remove_firmware(void) {
    Lock lock{GetState().mutex};
    if (IsRunning()) {
        return false;
    }
    auto sysnand = System().GetFileSystemController().GetSystemNANDContentDirectory();
    const bool ok = sysnand != nullptr && sysnand->CleanSubdirectoryRecursive("registered");
    RefreshFactories();
    return ok;
}

// ---- Updates and DLC -----------------------------------------------------------------------

uint64_t lichii_content_nsp_program_id(const char* path) {
    Lock lock{GetState().mutex};
    auto& system = System();
    const auto file = system.GetFilesystem()->OpenFile(PathString(PathFrom(path)),
                                                       FileSys::OpenMode::Read);
    if (file == nullptr) {
        return 0;
    }
    const FileSys::NSP nsp{file};
    if (nsp.GetStatus() != Loader::ResultStatus::Success) {
        return 0;
    }
    for (const auto& [title_id, ncas] : nsp.GetNCAs()) {
        if (!ncas.empty()) {
            return FileSys::GetBaseTitleID(title_id);
        }
    }
    return 0;
}

LichiiInstallResult lichii_content_install_nsp(const char* path_text, uint64_t program_id,
                                               LichiiContentProgress progress, void* user) {
    const auto path = PathFrom(path_text);
    if (!Common::FS::IsFile(path)) {
        return LICHII_INSTALL_NOT_FOUND;
    }
    const auto ext = Lower(PathString(path.extension()));
    if (ext != ".nsp" && ext != ".nsz") {
        return LICHII_INSTALL_WRONG_TYPE;
    }
    if (!ContentManager::AreKeysPresent()) {
        return LICHII_INSTALL_NO_KEYS;
    }
    if (program_id != 0) {
        const auto owner = lichii_content_nsp_program_id(path_text);
        if (owner != 0 && owner != FileSys::GetBaseTitleID(program_id)) {
            LOG_WARNING(Frontend, "{} belongs to {:016X}, not {:016X}", PathString(path), owner,
                        program_id);
            return LICHII_INSTALL_WRONG_GAME;
        }
    }

    Lock lock{GetState().mutex};
    auto& system = System();
    bool cancelled = false;
    const auto result = ContentManager::InstallNSP(
        system, *system.GetFilesystem(), PathString(path), [&](size_t total, size_t done) {
            cancelled = progress != nullptr && progress(user, done, total);
            return cancelled;
        });
    if (cancelled) {
        return LICHII_INSTALL_CANCELLED;
    }
    if (program_id != 0) {
        InvalidateGameListCache(FileSys::GetBaseTitleID(program_id));
    }
    switch (result) {
    case ContentManager::InstallResult::Success:
        return LICHII_INSTALL_OK;
    case ContentManager::InstallResult::Overwrite:
        return LICHII_INSTALL_OVERWRITE;
    case ContentManager::InstallResult::BaseInstallAttempted:
        return LICHII_INSTALL_BASE_GAME;
    case ContentManager::InstallResult::Failure:
        break;
    }
    return LICHII_INSTALL_FAILED;
}

char* lichii_content_list_addons_json(uint64_t program_id, const char* game_path) {
    Lock lock{GetState().mutex};
    json out = json::array();
    for (const auto& patch : Patches(program_id, game_path)) {
        std::string type;
        bool removable = false;
        switch (patch.type) {
        case FileSys::PatchType::Update:
            type = "update";
            removable = patch.source == FileSys::PatchSource::NAND ||
                        patch.source == FileSys::PatchSource::SDMC;
            break;
        case FileSys::PatchType::DLC:
            type = "dlc";
            removable = patch.source != FileSys::PatchSource::External;
            break;
        case FileSys::PatchType::Mod:
            if (patch.name.ends_with("(built-in)")) {
                type = "builtin";
            } else if (patch.name.starts_with("cheat_")) {
                type = "cheat";
                removable = true;
            } else {
                type = "mod";
                removable = !patch.location.empty();
            }
            break;
        }
        json entry{
            {"name", patch.name},
            {"toggle_key", ToggleKey(patch)},
            {"version", patch.version},
            {"type", type},
            {"enabled", patch.enabled},
            {"source", SourceName(patch.source)},
            {"title_id", Hex16(patch.title_id)},
            {"numeric_version", patch.numeric_version},
            {"location", patch.location},
            {"removable", removable},
        };
        if (patch.builtin) {
            const auto& info = *patch.builtin;
            const char* choice = info.choice == FileSys::BuiltinPatchChoice::On    ? "on"
                                 : info.choice == FileSys::BuiltinPatchChoice::Off ? "off"
                                                                                   : "global";
            entry["builtin"] = {
                {"kind", FileSys::BuiltinPatchKindName(info.kind)},
                {"game", info.game},
                {"description", info.description},
                {"effect", info.effect},
                {"choice", choice},
                {"source", FileSys::BuiltinPatchSourceName(info.source)},
                {"resolved_enabled", info.resolved_enabled},
                {"included_in", info.included_in},
            };
        }
        out.push_back(std::move(entry));
    }
    return DupJson(out);
}

char* lichii_content_builtin_patch_games_json(void) {
    return DupJson(json(FileSys::BuiltinPatchGames()));
}

bool lichii_content_set_builtin_patch_choice(uint64_t program_id, const char* name, int choice) {
    if (name == nullptr || name[0] == '\0' || choice < 0 || choice > 2) {
        return false;
    }
    Lock lock{GetState().mutex};
    FileSys::ApplyBuiltinPatchChoice(DisabledAddons(program_id), name,
                                     static_cast<FileSys::BuiltinPatchChoice>(choice));
    InvalidateGameListCache(program_id);
    SaveConfig();
    return true;
}

char* lichii_content_list_dlc_json(uint64_t program_id) {
    Lock lock{GetState().mutex};
    const auto& provider = System().GetContentProviderUnion();
    const auto base = FileSys::GetBaseTitleID(program_id);
    auto entries = provider.ListEntriesFilterOrigin(std::nullopt, FileSys::TitleType::AOC,
                                                    FileSys::ContentRecordType::Data);
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.second.title_id < b.second.title_id; });
    json out = json::array();
    for (const auto& [slot, entry] : entries) {
        if (FileSys::GetBaseTitleID(entry.title_id) != base) {
            continue;
        }
        const bool removable = slot == FileSys::ContentProviderUnionSlot::UserNAND ||
                               slot == FileSys::ContentProviderUnionSlot::SDMC;
        out.push_back({
            {"title_id", Hex16(entry.title_id)},
            {"index", entry.title_id & FileSys::AOC_TITLE_ID_MASK},
            {"source", SlotName(slot)},
            {"removable", removable},
        });
    }
    return DupJson(out);
}

bool lichii_content_set_addon_enabled(uint64_t program_id, const char* toggle_key, bool enabled) {
    if (toggle_key == nullptr || toggle_key[0] == '\0') {
        return false;
    }
    Lock lock{GetState().mutex};
    const std::string key{toggle_key};
    // Only one update can apply: enabling one disables the others.
    if (enabled && (key == "Update" || key.starts_with("Update@") || key.starts_with("Update ("))) {
        for (const auto& patch : Patches(program_id, nullptr)) {
            if (patch.type == FileSys::PatchType::Update) {
                SetDisabled(program_id, ToggleKey(patch), true);
            }
        }
        SetDisabled(program_id, "Update", false);
    }
    SetDisabled(program_id, key, !enabled);
    InvalidateGameListCache(program_id);
    SaveConfig();
    return true;
}

bool lichii_content_remove_update(uint64_t program_id) {
    Lock lock{GetState().mutex};
    const bool ok = ContentManager::RemoveUpdate(System().GetFileSystemController(), program_id);
    InvalidateGameListCache(program_id);
    return ok;
}

bool lichii_content_remove_dlc(uint64_t dlc_title_id) {
    Lock lock{GetState().mutex};
    const bool ok = ContentManager::RemoveDLC(System().GetFileSystemController(), dlc_title_id);
    InvalidateGameListCache(FileSys::GetBaseTitleID(dlc_title_id));
    return ok;
}

int lichii_content_remove_all_dlc(uint64_t program_id) {
    Lock lock{GetState().mutex};
    const auto removed = ContentManager::RemoveAllDLC(System(), program_id);
    InvalidateGameListCache(program_id);
    return static_cast<int>(removed);
}

// ---- Mods ----------------------------------------------------------------------------------

char* lichii_content_mod_dir(uint64_t program_id) {
    const auto dir = ModRoot(program_id);
    std::error_code ec;
    fs::create_directories(dir, ec);
    return Dup(PathString(dir));
}

LichiiModResult lichii_content_install_mod(uint64_t program_id, const char* path_text,
                                           const char* name, int* out_count) {
    if (out_count != nullptr) {
        *out_count = 0;
    }
    const auto source = PathFrom(path_text);
    fs::path root = source;
    fs::path scratch;
    std::string fallback_name = PathString(source.stem());

    if (Common::FS::IsFile(source)) {
        if (Lower(PathString(source.extension())) != ".zip") {
            return LICHII_MOD_NO_MOD;
        }
        scratch = ScratchDir("mod");
        if (LichiiZip::Extract(source, scratch) != LichiiZip::Result::Success) {
            RemoveQuietly(scratch);
            return LICHII_MOD_BAD_ZIP;
        }
        root = scratch;
    } else if (!Common::FS::IsDir(source)) {
        return LICHII_MOD_NOT_FOUND;
    } else {
        fallback_name = PathString(source.filename());
    }

    auto mods = FrontendCommon::GetModFolder(PathString(root));
    if (Common::FS::IsDir(root)) {
        for (const char* part : {"exefs", "romfs", "romfs_ext", "cheats", "romfslite"}) {
            if (FindCaseless(root, part) &&
                std::find(mods.begin(), mods.end(), root) == mods.end()) {
                mods.push_back(root);
            }
        }
    }
    // A mod's romfs can itself hold folders called "romfs" or "cheats"; keep the outermost.
    std::sort(mods.begin(), mods.end());
    std::vector<fs::path> outer;
    for (const auto& mod : mods) {
        const bool nested = std::any_of(outer.begin(), outer.end(), [&](const fs::path& parent) {
            const auto rel = mod.lexically_relative(parent);
            return !rel.empty() && *rel.begin() != "..";
        });
        if (!nested) {
            outer.push_back(mod);
        }
    }
    if (outer.empty()) {
        RemoveQuietly(scratch);
        return LICHII_MOD_NO_MOD;
    }

    const std::string requested = name != nullptr ? std::string{name} : std::string{};
    const auto dest_root = ModRoot(program_id);
    int installed = 0;
    LichiiModResult result = LICHII_MOD_OK;
    for (const auto& mod : outer) {
        std::string mod_name = PathString(mod.filename());
        if (mod == root || mod_name.empty()) {
            mod_name = fallback_name;
        }
        if (outer.size() == 1 && ValidName(requested)) {
            mod_name = requested;
        }
        if (!ValidName(mod_name)) {
            mod_name = fmt::format("Mod {}", installed + 1);
        }
        const auto dest = dest_root / mod_name;
        std::error_code ec;
        fs::remove_all(dest, ec);
        fs::create_directories(dest, ec);
        fs::copy(mod, dest, fs::copy_options::recursive | fs::copy_options::overwrite_existing,
                 ec);
        if (ec) {
            LOG_ERROR(Frontend, "Installing mod {} failed: {}", mod_name, ec.message());
            result = LICHII_MOD_FAILED;
            continue;
        }
        LOG_INFO(Frontend, "Installed mod {} for {:016X}", mod_name, program_id);
        ++installed;
    }
    RemoveQuietly(scratch);
    InvalidateGameListCache(program_id);
    if (out_count != nullptr) {
        *out_count = installed;
    }
    return installed > 0 ? LICHII_MOD_OK : result;
}

bool lichii_content_remove_mod(uint64_t program_id, const char* name) {
    if (name == nullptr || !ValidName(name)) {
        return false;
    }
    Lock lock{GetState().mutex};
    const auto path = ModRoot(program_id) / std::string{name};
    std::error_code ec;
    bool ok = false;
    if (fs::is_regular_file(path, ec)) {
        ok = fs::remove(path, ec);
    } else {
        ok = ContentManager::RemoveMod(System().GetFileSystemController(), program_id, name);
    }
    if (ok) {
        // Forget the toggles that belonged to it.
        auto& disabled = DisabledAddons(program_id);
        const std::string prefix = CheatKey(name, "");
        std::erase_if(disabled, [&](const std::string& key) {
            return key == name || key.starts_with(prefix);
        });
        InvalidateGameListCache(program_id);
        SaveConfig();
    }
    return ok;
}

// ---- Cheats --------------------------------------------------------------------------------

char* lichii_content_list_cheats_json(uint64_t program_id) {
    Lock lock{GetState().mutex};
    const auto root = ModRoot(program_id);
    const auto disabled = DisabledAddons(program_id);
    const auto is_disabled = [&](const std::string& key) {
        return std::find(disabled.begin(), disabled.end(), key) != disabled.end();
    };

    std::vector<fs::path> dirs;
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        const auto file_name = PathString(entry.path().filename());
        if (entry.is_directory(ec)) {
            dirs.push_back(entry.path());
        } else if (file_name.starts_with("cheat_")) {
            files.push_back(entry.path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    std::sort(files.begin(), files.end());

    json out = json::array();
    for (const auto& dir : dirs) {
        const auto cheats_dir = FindCaseless(dir, "cheats");
        if (!cheats_dir) {
            continue;
        }
        const auto source = PathString(dir.filename());
        std::vector<fs::path> txts;
        for (const auto& entry : fs::directory_iterator(*cheats_dir, ec)) {
            if (entry.is_regular_file(ec) && Lower(PathString(entry.path().extension())) == ".txt") {
                txts.push_back(entry.path());
            }
        }
        std::sort(txts.begin(), txts.end());
        for (const auto& txt : txts) {
            out.push_back({{"source", source},
                           {"source_enabled", !is_disabled(source)},
                           {"file", PathString(txt.filename())},
                           {"path", PathString(txt)},
                           {"cheats", ParseCheats(txt, disabled, source)}});
        }
    }
    for (const auto& file : files) {
        const auto source = PathString(file.filename());
        out.push_back({{"source", source},
                       {"source_enabled", !is_disabled(source)},
                       {"file", source},
                       {"path", PathString(file)},
                       {"cheats", ParseCheats(file, disabled, source)}});
    }
    return DupJson(out);
}

bool lichii_content_set_cheat_enabled(uint64_t program_id, const char* source, const char* name,
                                      bool enabled) {
    if (source == nullptr || name == nullptr || name[0] == '\0') {
        return false;
    }
    Lock lock{GetState().mutex};
    SetDisabled(program_id, CheatKey(source, name), !enabled);
    SaveConfig();
    return true;
}

bool lichii_content_install_cheat_file(uint64_t program_id, const char* path_text) {
    const auto path = PathFrom(path_text);
    const auto text = ReadTextFile(path);
    const Core::Memory::TextCheatParser parser;
    const auto parsed = parser.Parse(text);
    const bool any = std::any_of(parsed.begin(), parsed.end(), [](const auto& entry) {
        return entry.definition.num_opcodes > 0;
    });
    if (!any) {
        LOG_WARNING(Frontend, "{} holds no cheats", PathString(path));
        return false;
    }
    std::string stem = PathString(path.stem());
    if (!stem.starts_with("cheat_")) {
        stem = "cheat_" + stem;
    }
    if (!ValidName(stem)) {
        return false;
    }
    Lock lock{GetState().mutex};
    return CopyFileOver(path, ModRoot(program_id) / (stem + ".txt"));
}

// ---- Saves ---------------------------------------------------------------------------------

char* lichii_content_save_dir(uint64_t program_id, const char* game_path) {
    Lock lock{GetState().mutex};
    return Dup(PathString(SaveDirFor(program_id, game_path)));
}

char* lichii_content_save_info_json(uint64_t program_id, const char* game_path) {
    Lock lock{GetState().mutex};
    const auto dir = SaveDirFor(program_id, game_path);
    const bool exists = Common::FS::IsDir(dir);
    const auto [size, files] = exists ? DirStats(dir) : std::pair<u64, u64>{};
    return DupJson(json{{"path", PathString(dir)},
                    {"exists", exists && files > 0},
                    {"size", size},
                    {"files", files}});
}

bool lichii_content_export_save(uint64_t program_id, const char* game_path, const char* zip_path,
                                char** out_message) {
    Lock lock{GetState().mutex};
    const auto dir = SaveDirFor(program_id, game_path);
    if (!Common::FS::IsDir(dir) || DirStats(dir).second == 0) {
        SetMessage(out_message, "This game has no save data yet.");
        return false;
    }
    const auto result = LichiiZip::Create(PathFrom(zip_path), {{dir, ""}});
    SetMessage(out_message, result == LichiiZip::Result::Success
                                ? std::string{"Save exported."}
                                : std::string{LichiiZip::ResultString(result)});
    return result == LichiiZip::Result::Success;
}

bool lichii_content_import_save(uint64_t program_id, const char* game_path, const char* zip_path,
                                char** out_message) {
    Lock lock{GetState().mutex};
    if (IsRunning()) {
        SetMessage(out_message, "Stop the game before importing a save.");
        return false;
    }
    const auto dir = SaveDirFor(program_id, game_path);
    const auto scratch = ScratchDir("save");
    const auto result = LichiiZip::Extract(PathFrom(zip_path), scratch);
    if (result != LichiiZip::Result::Success) {
        RemoveQuietly(scratch);
        SetMessage(out_message, LichiiZip::ResultString(result));
        return false;
    }
    const auto root = SingleChildRoot(scratch);
    if (DirStats(root).second == 0) {
        RemoveQuietly(scratch);
        SetMessage(out_message, "The archive holds no files.");
        return false;
    }

    std::error_code ec;
    auto backup = dir;
    backup += ".bak";
    fs::remove_all(backup, ec);
    const bool had_save = fs::exists(dir, ec);
    if (had_save) {
        fs::rename(dir, backup, ec);
        if (ec) {
            RemoveQuietly(scratch);
            SetMessage(out_message, fmt::format("Moving the old save aside failed: {}",
                                                ec.message()));
            return false;
        }
    }
    fs::create_directories(dir, ec);
    fs::copy(root, dir, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    RemoveQuietly(scratch);
    if (ec) {
        const auto message = ec.message();
        RemoveQuietly(dir);
        if (had_save) {
            fs::rename(backup, dir, ec);
        }
        SetMessage(out_message, fmt::format("Importing failed: {}", message));
        return false;
    }
    RemoveQuietly(backup);
    LOG_INFO(Frontend, "Imported save for {:016X} into {}", program_id, PathString(dir));
    SetMessage(out_message, "Save imported.");
    return true;
}

bool lichii_content_delete_save(uint64_t program_id, const char* game_path) {
    Lock lock{GetState().mutex};
    if (IsRunning()) {
        return false;
    }
    const auto dir = SaveDirFor(program_id, game_path);
    std::error_code ec;
    fs::remove_all(dir, ec);
    return !ec;
}

// ---- Logs ----------------------------------------------------------------------------------

char* lichii_content_list_logs_json(void) {
    json out = json::array();
    std::error_code ec;
    for (const auto& path : LogFiles()) {
        const auto size = fs::file_size(path, ec);
        const auto modified = fs::last_write_time(path, ec);
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                                 modified.time_since_epoch() -
                                 fs::file_time_type::clock::now().time_since_epoch())
                                 .count() +
                             std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
        out.push_back({{"name", PathString(path.filename())},
                       {"path", PathString(path)},
                       {"size", ec ? 0 : size},
                       {"modified", seconds}});
    }
    return DupJson(out);
}

char* lichii_content_read_log(const char* path_text, uint64_t max_bytes) {
    const auto path = PathFrom(path_text);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return nullptr;
    }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<u64>(in.tellg());
    u64 start = 0;
    if (max_bytes != 0 && size > max_bytes) {
        start = size - max_bytes;
    }
    in.seekg(static_cast<std::streamoff>(start));
    std::string text(static_cast<std::size_t>(size - start), '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    if (start != 0) {
        if (const auto newline = text.find('\n'); newline != std::string::npos) {
            text.erase(0, newline + 1);
        }
    }
    std::replace(text.begin(), text.end(), '\0', ' ');
    return Dup(text);
}

bool lichii_content_export_logs(const char* zip_path, char** out_message) {
    std::vector<LichiiZip::Input> inputs;
    const auto crash_dir = Common::FS::GetEdenPath(EdenPath::CrashDumpsDir);
    for (const auto& path : LogFiles()) {
        const bool crash = path.parent_path() == crash_dir;
        inputs.push_back({path, (crash ? "crash_dumps/" : "log/") + PathString(path.filename())});
    }
    std::error_code ec;
    for (const auto& entry :
         fs::directory_iterator(Common::FS::GetEdenPath(EdenPath::ConfigDir), ec)) {
        if (entry.is_regular_file(ec) && Lower(PathString(entry.path().extension())) == ".ini") {
            inputs.push_back({entry.path(), "config/" + PathString(entry.path().filename())});
        }
    }
    if (inputs.empty()) {
        SetMessage(out_message, "There are no logs yet.");
        return false;
    }
    const auto result = LichiiZip::Create(PathFrom(zip_path), inputs);
    SetMessage(out_message, result == LichiiZip::Result::Success
                                ? fmt::format("{} files", inputs.size())
                                : std::string{LichiiZip::ResultString(result)});
    return result == LichiiZip::Result::Success;
}

// ---- User profiles -------------------------------------------------------------------------

char* lichii_content_list_profiles_json(void) {
    Lock lock{GetState().mutex};
    auto& manager = System().GetProfileManager();
    const auto current = Settings::values.current_user.GetValue();
    json out = json::array();
    std::size_t index = 0;
    for (const auto& uuid : manager.GetAllUsers()) {
        if (!uuid.IsValid()) {
            continue;
        }
        const auto avatar = AvatarPath(uuid);
        out.push_back({{"uuid", uuid.FormattedString()},
                       {"name", ProfileName(manager, uuid)},
                       {"current", static_cast<s32>(index) == current},
                       {"avatar", Common::FS::IsFile(avatar) ? PathString(avatar) : std::string{}},
                       {"index", index}});
        ++index;
    }
    return DupJson(out);
}

char* lichii_content_create_profile(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }
    Lock lock{GetState().mutex};
    auto& manager = System().GetProfileManager();
    if (manager.GetUserCount() >= Service::Account::MAX_USERS) {
        return nullptr;
    }
    const auto uuid = Common::UUID::MakeRandom();
    if (manager.CreateNewUser(uuid, ToUsername(name)).IsError()) {
        return nullptr;
    }
    manager.WriteUserSaveFile();
    return Dup(uuid.FormattedString());
}

bool lichii_content_rename_profile(const char* uuid_text, const char* name) {
    const auto uuid = ParseUuid(uuid_text);
    if (!uuid || name == nullptr || name[0] == '\0') {
        return false;
    }
    Lock lock{GetState().mutex};
    auto& manager = System().GetProfileManager();
    Service::Account::ProfileBase base{};
    if (!manager.GetProfileBase(*uuid, base)) {
        return false;
    }
    base.username = ToUsername(name);
    if (!manager.SetProfileBase(*uuid, base)) {
        return false;
    }
    manager.WriteUserSaveFile();
    return true;
}

bool lichii_content_delete_profile(const char* uuid_text) {
    const auto uuid = ParseUuid(uuid_text);
    if (!uuid) {
        return false;
    }
    Lock lock{GetState().mutex};
    auto& manager = System().GetProfileManager();
    const auto index = manager.GetUserIndex(*uuid);
    if (!index || manager.GetUserCount() <= 1) {
        return false;
    }
    const auto current = static_cast<std::size_t>(
        std::max(0, Settings::values.current_user.GetValue()));
    if (!manager.RemoveProfileAtIndex(*index)) {
        return false;
    }
    if (current == *index) {
        Settings::values.current_user = 0;
    } else if (current > *index) {
        Settings::values.current_user = static_cast<s32>(current - 1);
    }
    manager.WriteUserSaveFile();
    manager.ResetUserSaveFile();
    std::error_code ec;
    fs::remove(AvatarPath(*uuid), ec);
    SaveConfig();
    return true;
}

bool lichii_content_select_profile(const char* uuid_text) {
    const auto uuid = ParseUuid(uuid_text);
    if (!uuid) {
        return false;
    }
    Lock lock{GetState().mutex};
    const auto index = System().GetProfileManager().GetUserIndex(*uuid);
    if (!index) {
        return false;
    }
    Settings::values.current_user = static_cast<s32>(*index);
    SaveConfig();
    return true;
}

char* lichii_content_profile_avatar_path(const char* uuid_text) {
    const auto uuid = ParseUuid(uuid_text);
    return uuid ? Dup(PathString(AvatarPath(*uuid))) : nullptr;
}

bool lichii_content_set_profile_avatar(const char* uuid_text, const char* jpeg_path) {
    const auto uuid = ParseUuid(uuid_text);
    if (!uuid) {
        return false;
    }
    Lock lock{GetState().mutex};
    return CopyFileOver(PathFrom(jpeg_path), AvatarPath(*uuid));
}

// ---- System applets ------------------------------------------------------------------------

char* lichii_content_mii_edit_path(void) {
    Lock lock{GetState().mutex};
    const auto path = MiiEditPath();
    return path.empty() ? nullptr : Dup(path);
}

char* lichii_content_prepare_mii_edit(void) {
    Lock lock{GetState().mutex};
    const auto path = MiiEditPath();
    if (path.empty()) {
        return nullptr;
    }
    auto& state = GetState();
    std::scoped_lock applet_lock{state.applet_mutex};
    state.applet_path = path;
    return Dup(path);
}

} // extern "C"
