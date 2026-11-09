// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <fmt/format.h>
#include <unistd.h>

#include "common/fs/fs.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/ios_jit.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/settings_common.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/loader/loader.h"
#include "core/loader/nro.h"
#include "core/perf_stats.h"
#include "frontend_common/config.h"
#include "input_common/drivers/touch_screen.h"
#include "input_common/drivers/virtual_gamepad.h"
#include "input_common/main.h"
#include "ios_frontend/bridge_host.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"
#include "yuzu_cmd/ios_support.h"

namespace IosFrontend {
namespace {

Host g_host;
std::atomic<LichiiEmuState> g_state{LICHII_EMU_IDLE};
std::atomic<LichiiEventFn> g_event_handler{nullptr};
std::atomic<u64> g_jit_pool_bytes{0};
std::atomic<bool> g_stop_request{false};
std::atomic<bool> g_cancel_request{false};
std::uint64_t g_jit_background_task{};

std::mutex g_mutex;
std::optional<std::string> g_launch_request;
std::string g_error;
std::string g_game_path;
u64 g_program_id{};
void* g_game_window{};

// Loader and patch manager calls share the system's filesystem objects.
std::mutex g_loader_mutex;

char* Dup(std::string_view text) {
    char* out = static_cast<char*>(std::malloc(text.size() + 1));
    std::memcpy(out, text.data(), text.size());
    out[text.size()] = '\0';
    return out;
}

// Per-game ini (config/custom/<program id>.ini). Only the shared sections matter on iOS.
class GameConfig final : public Config {
public:
    explicit GameConfig(u64 program_id) : Config(ConfigType::PerGameConfig) {
        Initialize(fmt::format("{:016X}", program_id));
    }

    void ReloadAllValues() override {
        Reload();
    }
    void SaveAllValues() override {
        SaveValues();
    }

protected:
    void ReadHidbusValues() override {}
    void ReadDebugControlValues() override {}
    void ReadPathValues() override {}
    void ReadShortcutValues() override {}
    void ReadUIValues() override {}
    void ReadUIGamelistValues() override {}
    void ReadUILayoutValues() override {}
    void ReadMultiplayerValues() override {}
    void SaveHidbusValues() override {}
    void SaveDebugControlValues() override {}
    void SavePathValues() override {}
    void SaveShortcutValues() override {}
    void SaveUIValues() override {}
    void SaveUIGamelistValues() override {}
    void SaveUILayoutValues() override {}
    void SaveMultiplayerValues() override {}
    std::vector<Settings::BasicSetting*>& FindRelevantList(Settings::Category category) override {
        return Settings::values.linkage.by_category[category];
    }
};

std::unique_ptr<GameConfig> g_game_config;
u64 g_game_scope{};

std::filesystem::path GameConfigPath(u64 program_id) {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "custom" /
           fmt::format("{:016X}.ini", program_id);
}

} // Anonymous namespace

void Attach(const Host& host) {
    g_host = host;
}

Core::System& System() {
    return *g_host.system;
}

InputCommon::InputSubsystem& Input() {
    return *g_host.input;
}

void SaveGlobalConfig() {
    if (g_host.save_config) {
        g_host.save_config();
    }
}

std::optional<std::string> TakeLaunchRequest() {
    std::scoped_lock lock{g_mutex};
    return std::exchange(g_launch_request, std::nullopt);
}

bool TakeStopRequest() {
    return g_stop_request.exchange(false);
}

bool TakeCancelRequest() {
    return g_cancel_request.exchange(false);
}

void SetState(LichiiEmuState state, std::string error) {
    {
        std::scoped_lock lock{g_mutex};
        if (state == LICHII_EMU_FAILED) {
            g_error = std::move(error);
        } else if (state == LICHII_EMU_WAITING_FOR_JIT) {
            g_error.clear();
        }
    }
    if (g_state.exchange(state) != state) {
        PostEvent(LICHII_EVENT_STATE_CHANGED);
    }
}

LichiiEmuState GetState() {
    return g_state.load();
}

void SetRunningGame(std::string path, u64 program_id, void* ui_window) {
    {
        std::scoped_lock lock{g_mutex};
        g_game_path = std::move(path);
        g_program_id = program_id;
        g_game_window = ui_window;
    }
    PostEvent(LICHII_EVENT_GAME_WINDOW_READY);
}

void ClearRunningGame() {
    std::scoped_lock lock{g_mutex};
    g_game_path.clear();
    g_program_id = 0;
    g_game_window = nullptr;
}

u64 RunningProgramId() {
    std::scoped_lock lock{g_mutex};
    return g_program_id;
}

void ApplyGameConfig(u64 program_id) {
    if (program_id == 0 || !Common::FS::Exists(GameConfigPath(program_id))) {
        return;
    }
    LOG_INFO(Frontend, "iOS: applying per-game settings for {:016X}", program_id);
    GameConfig config{program_id};
}

void RestoreGlobal() {
    Settings::RestoreGlobalState(false);
}

u64 ReadProgramId(const std::string& path) {
    std::scoped_lock lock{g_loader_mutex};
    auto& system = System();
    const auto file = Core::GetGameFileFromPath(system.GetFilesystem(), path);
    if (file == nullptr) {
        return 0;
    }
    const auto loader = Loader::GetLoader(system, file);
    u64 program_id = 0;
    if (loader == nullptr || loader->ReadProgramId(program_id) != Loader::ResultStatus::Success) {
        return 0;
    }
    return program_id;
}

void PostEvent(LichiiEvent event) {
    dispatch_async_f(dispatch_get_main_queue(), reinterpret_cast<void*>(static_cast<intptr_t>(event)),
                     [](void* context) {
                         if (const auto handler = g_event_handler.load()) {
                             handler(static_cast<int32_t>(reinterpret_cast<intptr_t>(context)));
                         }
                     });
}

void SetJitPoolBytes(u64 bytes) {
    g_jit_pool_bytes = bytes;
}

void EndJitBackgroundTask() {
    IosSupport::EndBackgroundTask(std::exchange(g_jit_background_task, 0));
}

} // namespace IosFrontend

using namespace IosFrontend;

namespace {

/* ---------------------------------------------------------------------------------- games */

LichiiFileType ToFileType(Loader::FileType type) {
    switch (type) {
    case Loader::FileType::NSO:
        return LICHII_FILE_NSO;
    case Loader::FileType::NRO:
        return LICHII_FILE_NRO;
    case Loader::FileType::NCA:
        return LICHII_FILE_NCA;
    case Loader::FileType::NSP:
        return LICHII_FILE_NSP;
    case Loader::FileType::NSZ:
        return LICHII_FILE_NSZ;
    case Loader::FileType::XCI:
        return LICHII_FILE_XCI;
    case Loader::FileType::XCZ:
        return LICHII_FILE_XCZ;
    case Loader::FileType::NUP:
        return LICHII_FILE_NUP;
    case Loader::FileType::DeconstructedRomDirectory:
        return LICHII_FILE_DIRECTORY;
    default:
        return LICHII_FILE_UNKNOWN;
    }
}

std::string LowerExtension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

bool IsGameExtension(const std::string& ext) {
    return ext == ".nup" || ext == ".nsp" || ext == ".nsz" || ext == ".xci" || ext == ".xcz" ||
           ext == ".nca" || ext == ".nso" || ext == ".nro";
}

// Files, split parts ("<name>.nsp/00") and deconstructed ROM directories (main + main.npdm).
void CollectGames(const std::filesystem::path& dir, int depth,
                  std::vector<std::filesystem::path>& out) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto& path = entry.path();
        const std::string name = path.filename().string();
        if (name.empty() || name[0] == '.') {
            continue;
        }
        if (entry.is_directory(ec)) {
            if (depth == 0 && name == EDEN_DIR) {
                continue;
            }
            if (std::filesystem::exists(path / "main", ec) &&
                std::filesystem::exists(path / "main.npdm", ec)) {
                out.push_back(path);
            } else if (IsGameExtension(LowerExtension(path)) &&
                       std::filesystem::exists(path / "00", ec)) {
                out.push_back(path / "00");
            } else if (depth < 2) {
                CollectGames(path, depth + 1, out);
            }
        } else if (entry.is_regular_file(ec) && IsGameExtension(LowerExtension(path))) {
            out.push_back(path);
        }
    }
}

u64 GameSize(const std::filesystem::path& path) {
    std::error_code ec;
    const auto root = path.filename() == "00" ? path.parent_path() : path;
    if (!std::filesystem::is_directory(root, ec)) {
        return std::filesystem::file_size(root, ec);
    }
    u64 total = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root, ec)) {
        if (entry.is_regular_file(ec)) {
            total += entry.file_size(ec);
        }
    }
    return total;
}

bool ReadGame(const std::string& path, LichiiGame& out) {
    std::scoped_lock lock{g_loader_mutex};
    auto& system = System();
    const auto file = Core::GetGameFileFromPath(system.GetFilesystem(), path);
    if (file == nullptr) {
        return false;
    }
    const auto loader = Loader::GetLoader(system, file);
    if (loader == nullptr) {
        return false;
    }
    const auto type = loader->GetFileType();
    if (type == Loader::FileType::Unknown || type == Loader::FileType::Error) {
        return false;
    }
    if ((type == Loader::FileType::NSP || type == Loader::FileType::XCI ||
         type == Loader::FileType::NSZ || type == Loader::FileType::XCZ) &&
        !Loader::IsBootableGameContainer(file, type)) {
        return false;
    }
    u64 program_id = 0;
    if (loader->ReadProgramId(program_id) != Loader::ResultStatus::Success) {
        return false;
    }

    std::string title;
    loader->ReadTitle(title);
    if (title.empty()) {
        const std::filesystem::path p{path};
        title = (p.filename() == "00" || p.filename() == "main" ? p.parent_path() : p)
                    .stem()
                    .string();
    }
    std::vector<u8> icon;
    loader->ReadIcon(icon);

    std::string developer;
    std::string version;
    const FileSys::PatchManager pm{program_id, system.GetFileSystemController(),
                                   system.GetContentProvider()};
    const auto control = pm.GetControlMetadata();
    if (control.first != nullptr) {
        developer = control.first->GetDeveloperName();
        version = control.first->GetVersionString();
    } else {
        FileSys::NACP nacp;
        if (loader->ReadControlData(nacp) == Loader::ResultStatus::Success) {
            developer = nacp.GetDeveloperName();
            version = nacp.GetVersionString();
        }
    }

    out = {};
    out.path = Dup(path);
    out.title = Dup(title);
    out.developer = Dup(developer);
    out.version = Dup(version);
    out.program_id = program_id;
    out.file_size = GameSize(std::filesystem::path{path});
    out.file_type = ToFileType(type);
    out.is_homebrew = type == Loader::FileType::NRO &&
                      static_cast<Loader::AppLoader_NRO*>(loader.get())->IsHomebrew();
    if (!icon.empty()) {
        out.icon = static_cast<uint8_t*>(std::malloc(icon.size()));
        std::memcpy(out.icon, icon.data(), icon.size());
        out.icon_size = icon.size();
    }
    return true;
}

/* ------------------------------------------------------------------------------- settings */

struct EnumChoice {
    std::string name;
    u32 value;
};

template <typename T>
void AddEnum(std::map<u32, std::vector<EnumChoice>>& table) {
    auto& choices = table[Settings::EnumMetadata<T>::Index()];
    for (const auto& [name, value] : Settings::EnumMetadata<T>::Canonicalizations()) {
        choices.push_back({std::string{name}, static_cast<u32>(value)});
    }
}

const std::map<u32, std::vector<EnumChoice>>& EnumTable() {
    static const auto table = [] {
        using namespace Settings;
        std::map<u32, std::vector<EnumChoice>> t;
        AddEnum<AudioEngine>(t);
        AddEnum<AudioMode>(t);
        AddEnum<Language>(t);
        AddEnum<Region>(t);
        AddEnum<TimeZone>(t);
        AddEnum<AnisotropyMode>(t);
        AddEnum<AstcDecodeMode>(t);
        AddEnum<AstcRecompression>(t);
        AddEnum<FramePacingMode>(t);
        AddEnum<VSyncMode>(t);
        AddEnum<VramUsageMode>(t);
        AddEnum<RendererBackend>(t);
        AddEnum<GpuAccuracy>(t);
        AddEnum<DmaAccuracy>(t);
        AddEnum<GpuFenceBehavior>(t);
        AddEnum<NativeComputeWriteMark>(t);
        AddEnum<CpuBackend>(t);
        AddEnum<CpuAccuracy>(t);
        AddEnum<CpuClock>(t);
        AddEnum<MemoryLayout>(t);
        AddEnum<ConfirmStop>(t);
        AddEnum<FullscreenMode>(t);
        AddEnum<NvdecEmulation>(t);
        AddEnum<ResolutionSetup>(t);
        AddEnum<ScalingFilter>(t);
        AddEnum<AntiAliasing>(t);
        AddEnum<OotAntiAliasing>(t);
        AddEnum<DlssPreset>(t);
        AddEnum<AspectRatio>(t);
        AddEnum<ConsoleMode>(t);
        AddEnum<AppletMode>(t);
        AddEnum<SpirvOptimizeMode>(t);
        AddEnum<GpuClock>(t);
        AddEnum<GpuUnswizzleSize>(t);
        AddEnum<GpuUnswizzle>(t);
        AddEnum<GpuUnswizzleChunk>(t);
        AddEnum<TemperatureUnits>(t);
        AddEnum<ExtendedDynamicState>(t);
        AddEnum<GpuLogLevel>(t);
        AddEnum<GameListMode>(t);
        AddEnum<SpeedMode>(t);
        AddEnum<FrameGenType>(t);
        return t;
    }();
    return table;
}

const std::vector<EnumChoice>* EnumChoices(const Settings::BasicSetting& setting) {
    if (!setting.IsEnum()) {
        return nullptr;
    }
    const auto& table = EnumTable();
    const auto it = table.find(setting.EnumIndex());
    return it == table.end() ? nullptr : &it->second;
}

const std::vector<Settings::BasicSetting*>& AllSettings() {
    static const auto list = [] {
        std::vector<Settings::BasicSetting*> out;
        for (const auto& [category, settings] : Settings::values.linkage.by_category) {
            for (auto* setting : settings) {
                if (setting->Save()) {
                    out.push_back(setting);
                }
            }
        }
        return out;
    }();
    return list;
}

Settings::BasicSetting* FindSetting(const char* key) {
    if (key == nullptr) {
        return nullptr;
    }
    const auto& by_key = Settings::values.linkage.by_key;
    const auto it = by_key.find(key);
    return it == by_key.end() ? nullptr : it->second;
}

LichiiSettingKind KindOf(const Settings::BasicSetting& setting) {
    if (setting.TypeId() == "bool") {
        return LICHII_SETTING_BOOL;
    }
    if (setting.IsEnum()) {
        return LICHII_SETTING_ENUM;
    }
    if (setting.IsFloatingPoint()) {
        return LICHII_SETTING_FLOAT;
    }
    if (setting.IsIntegral()) {
        return LICHII_SETTING_INT;
    }
    return LICHII_SETTING_STRING;
}

double ToNumber(const std::string& text) {
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    return end == text.c_str() ? 0.0 : value;
}

void FillInfo(const Settings::BasicSetting& setting, LichiiSettingInfo& out) {
    out = {};
    out.key = setting.GetLabel().c_str();
    out.category = Settings::TranslateCategory(setting.GetCategory());
    out.category_id = static_cast<uint32_t>(setting.GetCategory());
    out.kind = KindOf(setting);
    out.switchable = setting.Switchable();
    out.runtime_modifiable = setting.RuntimeModifiable();
    out.ranged = setting.Ranged();
    out.specialization = setting.Specialization();
    out.percentage = (setting.Specialization() & Settings::Specialization::Percentage) != 0;
    if (out.kind == LICHII_SETTING_INT || out.kind == LICHII_SETTING_FLOAT ||
        out.kind == LICHII_SETTING_ENUM) {
        out.minimum = ToNumber(setting.MinVal());
        out.maximum = ToNumber(setting.MaxVal());
    }
    if (const auto* paired = setting.PairedSetting()) {
        out.paired_key = paired->GetLabel().c_str();
    }
    if (const auto* choices = EnumChoices(setting)) {
        out.enum_count = static_cast<uint32_t>(choices->size());
    }
}

bool InGameScope() {
    return g_game_config != nullptr;
}

} // Anonymous namespace

extern "C" {

void lichii_free(void* pointer) {
    std::free(pointer);
}

/* ---------------------------------------------------------------------------------- paths */

char* lichii_documents_dir(void) {
    const char* home = std::getenv("HOME");
    return Dup(std::string(home ? home : ".") + "/Documents");
}

char* lichii_user_dir(void) {
    return Dup(Common::FS::PathToUTF8String(
        Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir)));
}

char* lichii_log_file(void) {
    return Dup(Common::FS::PathToUTF8String(
        Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / LOG_FILE));
}

char* lichii_build_version(void) {
    return Dup(fmt::format("{} | {}-{}", Common::g_build_name, Common::g_scm_branch,
                           Common::g_scm_desc));
}

/* ---------------------------------------------------------------------------------- games */

size_t lichii_games_scan(const char* dir, LichiiGame** out_games) {
    *out_games = nullptr;
    std::vector<std::filesystem::path> paths;
    CollectGames(std::filesystem::path{dir}, 0, paths);
    std::sort(paths.begin(), paths.end());
    std::vector<LichiiGame> games;
    for (const auto& path : paths) {
        LichiiGame game{};
        if (ReadGame(path.string(), game)) {
            games.push_back(game);
        }
    }
    if (games.empty()) {
        return 0;
    }
    *out_games = static_cast<LichiiGame*>(std::malloc(games.size() * sizeof(LichiiGame)));
    std::memcpy(*out_games, games.data(), games.size() * sizeof(LichiiGame));
    return games.size();
}

bool lichii_game_read(const char* path, LichiiGame* out_game) {
    *out_game = {};
    return path != nullptr && ReadGame(path, *out_game);
}

void lichii_game_clear(LichiiGame* game) {
    std::free(game->path);
    std::free(game->title);
    std::free(game->developer);
    std::free(game->version);
    std::free(game->icon);
    *game = {};
}

void lichii_games_free(LichiiGame* games, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        lichii_game_clear(&games[i]);
    }
    std::free(games);
}

/* ------------------------------------------------------------------------------- settings */

size_t lichii_settings_count(void) {
    return AllSettings().size();
}

bool lichii_settings_info(size_t index, LichiiSettingInfo* out_info) {
    const auto& settings = AllSettings();
    if (index >= settings.size()) {
        return false;
    }
    FillInfo(*settings[index], *out_info);
    return true;
}

bool lichii_settings_find(const char* key, LichiiSettingInfo* out_info) {
    const auto* setting = FindSetting(key);
    if (setting == nullptr) {
        return false;
    }
    FillInfo(*setting, *out_info);
    return true;
}

const char* lichii_settings_enum_name(const char* key, uint32_t index) {
    const auto* setting = FindSetting(key);
    const auto* choices = setting ? EnumChoices(*setting) : nullptr;
    return choices && index < choices->size() ? (*choices)[index].name.c_str() : "";
}

uint32_t lichii_settings_enum_value(const char* key, uint32_t index) {
    const auto* setting = FindSetting(key);
    const auto* choices = setting ? EnumChoices(*setting) : nullptr;
    return choices && index < choices->size() ? (*choices)[index].value : 0;
}

char* lichii_settings_get(const char* key) {
    const auto* setting = FindSetting(key);
    if (setting == nullptr) {
        return Dup("");
    }
    return Dup(InGameScope() ? setting->ToString() : setting->ToStringGlobal());
}

char* lichii_settings_get_default(const char* key) {
    const auto* setting = FindSetting(key);
    return Dup(setting ? setting->DefaultToString() : std::string{});
}

bool lichii_settings_set(const char* key, const char* value) {
    auto* setting = FindSetting(key);
    if (setting == nullptr || value == nullptr) {
        return false;
    }
    if (InGameScope()) {
        if (!setting->Switchable()) {
            return false;
        }
        setting->SetGlobal(false);
        setting->LoadString(value);
        return true;
    }
    // A running game may use its own value; the global one is written underneath.
    const bool custom = !setting->UsingGlobal();
    if (custom) {
        setting->SetGlobal(true);
    }
    setting->LoadString(value);
    if (custom) {
        setting->SetGlobal(false);
    }
    return true;
}

bool lichii_settings_begin_game(uint64_t program_id) {
    if (program_id == 0 || InGameScope()) {
        return false;
    }
    const u64 running = RunningProgramId();
    if (running != 0 && running != program_id) {
        return false;
    }
    Settings::SetConfiguringGlobal(false);
    g_game_config = std::make_unique<GameConfig>(program_id);
    g_game_scope = program_id;
    return true;
}

void lichii_settings_end_game(void) {
    if (!InGameScope()) {
        return;
    }
    g_game_config->SaveAllValues();
    g_game_config.reset();
    Settings::SetConfiguringGlobal(true);
    if (RunningProgramId() == g_game_scope && System().IsPoweredOn()) {
        System().ApplySettings();
    } else {
        Settings::RestoreGlobalState(false);
    }
    g_game_scope = 0;
}

bool lichii_settings_in_game_scope(void) {
    return InGameScope();
}

bool lichii_settings_has_game_config(uint64_t program_id) {
    return Common::FS::Exists(GameConfigPath(program_id));
}

bool lichii_settings_uses_global(const char* key) {
    const auto* setting = FindSetting(key);
    return setting == nullptr || setting->UsingGlobal();
}

void lichii_settings_set_use_global(const char* key, bool use_global) {
    if (auto* setting = FindSetting(key); setting != nullptr && InGameScope()) {
        setting->SetGlobal(use_global);
    }
}

void lichii_settings_save(void) {
    SaveGlobalConfig();
    if (System().IsPoweredOn()) {
        System().ApplySettings();
    }
}

/* ------------------------------------------------------------------------------ emulation */

bool lichii_emu_launch(const char* path) {
    const auto state = g_state.load();
    if (path == nullptr || state == LICHII_EMU_BOOTING || state == LICHII_EMU_RUNNING ||
        state == LICHII_EMU_PAUSED) {
        return false;
    }
    {
        std::scoped_lock lock{g_mutex};
        g_launch_request = path;
    }
    SetState(LICHII_EMU_WAITING_FOR_JIT);
    return true;
}

void lichii_emu_cancel_launch(void) {
    {
        std::scoped_lock lock{g_mutex};
        g_launch_request.reset();
    }
    g_cancel_request = true;
}

void lichii_emu_pause(void) {
    if (g_state.load() == LICHII_EMU_RUNNING && System().IsPoweredOn()) {
        void(System().Pause());
        SetState(LICHII_EMU_PAUSED);
    }
}

void lichii_emu_resume(void) {
    if (g_state.load() == LICHII_EMU_PAUSED && System().IsPoweredOn()) {
        void(System().Run());
        SetState(LICHII_EMU_RUNNING);
    }
}

void lichii_emu_stop(void) {
    g_stop_request = true;
}

int32_t lichii_emu_state(void) {
    return g_state.load();
}

char* lichii_emu_error(void) {
    std::scoped_lock lock{g_mutex};
    return Dup(g_error);
}

uint64_t lichii_emu_program_id(void) {
    return RunningProgramId();
}

char* lichii_emu_game_path(void) {
    std::scoped_lock lock{g_mutex};
    return Dup(g_game_path);
}

void* lichii_emu_window(void) {
    std::scoped_lock lock{g_mutex};
    return g_game_window;
}

bool lichii_emu_perf_stats(LichiiPerfStats* out_stats) {
    const auto state = g_state.load();
    if ((state != LICHII_EMU_RUNNING && state != LICHII_EMU_PAUSED) || !System().IsPoweredOn()) {
        return false;
    }
    const auto results = System().GetAndResetPerfStats();
    out_stats->fps = results.total_fps;
    out_stats->game_fps = results.average_game_fps;
    out_stats->generated_fps = results.generated_fps;
    out_stats->frametime_ms = results.frametime * 1000.0;
    out_stats->speed = results.emulation_speed;
    return true;
}

bool lichii_emu_screenshot(uint32_t height, LichiiScreenshotFn done, void* context) {
    if (g_state.load() != LICHII_EMU_RUNNING || !System().IsPoweredOn() || done == nullptr) {
        return false;
    }
    auto& renderer = System().GPU().Renderer();
    if (renderer.IsScreenshotPending()) {
        return false;
    }
    height = std::clamp<uint32_t>(height, 144, 2160);
    const auto layout = Layout::DefaultFrameLayout(height * 16 / 9, height);
    auto pixels = std::make_shared<std::vector<u8>>(size_t{layout.width} * layout.height * 4);
    renderer.RequestScreenshot(
        pixels->data(),
        [pixels, done, context, width = layout.width, height = layout.height](bool flip_y) {
            done(context, pixels->data(), width, height, flip_y);
        },
        layout);
    return true;
}

void lichii_set_event_handler(LichiiEventFn handler) {
    g_event_handler = handler;
}

void lichii_post_event(int32_t event) {
    PostEvent(static_cast<LichiiEvent>(event));
}

/* ------------------------------------------------------------------------------------ JIT */

int32_t lichii_jit_state(void) {
    switch (Common::IosJit::GetState()) {
    case Common::IosJit::State::Waiting:
        return LICHII_JIT_WAITING;
    case Common::IosJit::State::Acquiring:
        return LICHII_JIT_ACQUIRING;
    case Common::IosJit::State::Ready:
        return LICHII_JIT_READY;
    case Common::IosJit::State::Failed:
        return LICHII_JIT_FAILED;
    }
    return LICHII_JIT_FAILED;
}

uint64_t lichii_jit_prepared_bytes(void) {
    return Common::IosJit::PreparedBytes();
}

uint64_t lichii_jit_pool_bytes(void) {
    return g_jit_pool_bytes.load();
}

char* lichii_jit_error(void) {
    return Dup(Common::IosJit::GetError());
}

bool lichii_jit_debugged(void) {
    return Common::IosJit::Debugged();
}

bool lichii_jit_needs_script(void) {
    return IosSupport::HasTxm();
}

bool lichii_jit_stikdebug_installed(void) {
    return IosSupport::CanOpenScheme("stikdebug");
}

bool lichii_jit_request(void) {
    std::string bundle_id;
    if (CFStringRef id = CFBundleGetIdentifier(CFBundleGetMainBundle())) {
        char buffer[256];
        if (CFStringGetCString(id, buffer, sizeof(buffer), kCFStringEncodingUTF8)) {
            bundle_id = buffer;
        }
    }
    std::string url = "stikdebug://enable-jit?bundle-id=" + bundle_id +
                      "&pid=" + std::to_string(getpid());
    if (IosSupport::HasTxm()) {
        url += "&script-name=universal.js";
    }
    if (g_jit_background_task == 0) {
        g_jit_background_task = IosSupport::BeginBackgroundTask("lichii-jit");
    }
    LOG_INFO(Frontend, "iOS: opening {}", url);
    return IosSupport::OpenUrl(url.c_str());
}

/* ---------------------------------------------------------------------------------- input */

namespace {
constexpr std::size_t kVirtualPorts[] = {0, 8};
} // Anonymous namespace

void lichii_input_button(int32_t button, bool pressed) {
    if (auto* pad = Input().GetVirtualGamepad()) {
        for (const std::size_t port : kVirtualPorts) {
            pad->SetButtonState(port, button, pressed);
        }
    }
}

void lichii_input_stick(int32_t stick, float x, float y) {
    if (auto* pad = Input().GetVirtualGamepad()) {
        for (const std::size_t port : kVirtualPorts) {
            pad->SetStickPosition(port, stick, x, y);
        }
    }
}

void lichii_input_touch_down(uint32_t finger, float x, float y) {
    Input().GetTouchScreen()->TouchPressed(x, y, finger);
}

void lichii_input_touch_move(uint32_t finger, float x, float y) {
    Input().GetTouchScreen()->TouchMoved(x, y, finger);
}

void lichii_input_touch_up(uint32_t finger) {
    Input().GetTouchScreen()->TouchReleased(finger);
}

void lichii_input_release_all(void) {
    if (auto* pad = Input().GetVirtualGamepad()) {
        pad->ResetControllers();
    }
    Input().GetTouchScreen()->ReleaseAllTouch();
}

} // extern "C"
